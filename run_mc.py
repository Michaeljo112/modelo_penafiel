"""
Monte Carlo runner for the extended model, parallelised with
multiprocessing.Pool. Each of the R*|SCENARIOS| runs is an independent task
dispatched to a worker process.

Usage:
    python run_mc.py                # full run: R=100, T=500
    python run_mc.py --fast         # quick test: R=10, T=100
    python run_mc.py --original     # ablation: reproduce the pre-revision
                                     # (hard-threshold, non-adaptive,
                                     # deterministic-disintegration) model
    python run_mc.py --workers 4    # override worker count
"""

import argparse
import time
import numpy as np
from numpy.random import SeedSequence
import pandas as pd
from pathlib import Path
from multiprocessing import Pool, cpu_count

from simulation import run, Params

OUT = Path("results")
OUT.mkdir(exist_ok=True)

# ─── Reproducibility ─────────────────────────────────────────────────────────
# Same base seed as the root study, so scenario draws are comparable.
BASE_SEED = 20250624

# ─── Scenario definitions ────────────────────────────────────────────────────
SCENARIOS = {
    "altruistic":    (0.80, 0.10),   # d_A=0.80 > u=0.75: starts altruistic regime
    "borderline":    (0.55, 0.25),   # d_A=0.55 < u=0.75: near tipping point
    "extractive":    (0.10, 0.80),   # d_X=0.80: strongly extractive
    "institutional": (0.10, 0.10),   # d_I=0.80: norm-followers dominate
}


# ─── Worker (must be module-level for pickling on Windows) ───────────────────

def _worker(args: tuple) -> tuple:
    """Run one replication and return (scenario_name, rep, result_dict)."""
    name, d_A, d_X, rep, seed, p = args
    result = run(d_A, d_X, seed=seed, p=p)
    return name, rep, result


# ─── Task builder ────────────────────────────────────────────────────────────

def _build_tasks(scenarios: dict, R: int, p: Params) -> list:
    """Enumerate all (scenario, replication) tasks with deterministic seeds."""
    sc_names = list(scenarios.keys())
    top_ss = SeedSequence(BASE_SEED).spawn(len(sc_names))   # one per scenario

    tasks = []
    for sc_idx, (name, (d_A, d_X)) in enumerate(scenarios.items()):
        child_seeds = [
            int(s.generate_state(1)[0])
            for s in top_ss[sc_idx].spawn(R)
        ]
        for rep in range(R):
            tasks.append((name, d_A, d_X, rep, child_seeds[rep], p))
    return tasks


# ─── Results assembler ───────────────────────────────────────────────────────

def _assemble(raw: list, T: int) -> pd.DataFrame:
    """Convert list of (name, rep, result_dict) to tidy DataFrame."""
    records = []
    for name, rep, result in raw:
        for t in range(T):
            if result["N"][t] <= 0:
                continue
            records.append({
                "scenario": name,
                "rep":      rep,
                "t":        t,
                "Y":        result["Y"][t],
                "G":        result["G"][t],
                "alpha":    result["alpha"][t],
                "regime":   result["regime"][t],
                "congress": result["congress"][t],
                "disint":   result["disint"][t],
                "poverty":  result["poverty"][t],
                "deaths":   result["deaths"][t],
                "mortality": result["mortality"][t],
                "N":        int(result["N"][t]),
            })
    return pd.DataFrame(records)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",    action="store_true",
                        help="Quick test: R=10, T=100")
    parser.add_argument("--original", action="store_true",
                        help="Ablation: reproduce pre-revision model "
                             "(hard thresholds, no adaptation, deterministic disintegration)")
    parser.add_argument("--no-hierarchy", action="store_true",
                        help="Ablation for Proposition 3: disable the elite-congress "
                             "override, holding smooth thresholds, adaptation and "
                             "stochastic disintegration fixed at their revised-model "
                             "values (isolates the hierarchy mechanism)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Worker processes (default: all logical CPUs)")
    args = parser.parse_args()

    p = Params()
    if args.original:
        p.smooth_thresholds = False
        p.adaptive = False
        p.psi = 0.0
        p.stochastic_disint = False
    if args.no_hierarchy:
        p.elite_override = False

    R = 10 if args.fast else 100
    if args.fast:
        p.T = 100

    n_workers = args.workers or cpu_count()
    suffix    = "_fast" if args.fast else ""
    suffix   += "_original" if args.original else ""
    suffix   += "_nohierarchy" if args.no_hierarchy else ""

    n_tasks = len(SCENARIOS) * R
    print(f"Monte Carlo: {len(SCENARIOS)} scenarios x {R} reps x {p.T} periods "
          f"= {n_tasks} tasks on {n_workers} workers")
    print(f"N={p.N}, g={p.g}, tau={p.tau}, plus_rate={p.plus_rate}, u={p.u}, "
          f"disint_scale={p.disint_scale}, p_max={p.p_max}, "
          f"smooth_thresholds={p.smooth_thresholds}, adaptive={p.adaptive} (psi={p.psi}), "
          f"stochastic_disint={p.stochastic_disint}")
    print()

    tasks = _build_tasks(SCENARIOS, R, p)
    t0 = time.time()
    if n_workers == 1:
        raw = []
        done = 0
        for result in map(_worker, tasks):
            raw.append(result)
            done += 1
            if done % max(1, n_tasks // 20) == 0 or done == n_tasks:
                elapsed = time.time() - t0
                eta = elapsed / done * (n_tasks - done)
                print(f"  {done}/{n_tasks} done  "
                      f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]",
                      flush=True)

        elapsed = time.time() - t0
        print(f"\nAll runs finished in {elapsed:.1f}s  "
              f"({elapsed/n_tasks:.2f}s per run, {n_workers} workers)")

        df_all = _assemble(raw, p.T)
        df_all.to_parquet(OUT / f"mc_results{suffix}.parquet", index=False)
        summary = (
            df_all.groupby(["scenario", "t"])
            .agg(
                Y_mean=("Y", "mean"),
                Y_p25=("Y", lambda x: np.quantile(x, 0.25)),
                Y_p75=("Y", lambda x: np.quantile(x, 0.75)),
                G_mean=("G", "mean"),
                G_p25=("G", lambda x: np.quantile(x, 0.25)),
                G_p75=("G", lambda x: np.quantile(x, 0.75)),
                alpha_mean=("alpha", "mean"),
                regime_frac=("regime", "mean"),
                congress_frac=("congress", "mean"),
                disint_rate=("disint", "mean"),
                poverty_mean=("poverty", "mean"),
                deaths_mean=("deaths", "mean"),
                mortality_mean=("mortality", "mean"),
                N_mean=("N", "mean"),
            )
            .reset_index()
        )
        summary.to_parquet(OUT / f"mc_summary{suffix}.parquet", index=False)
        summary.to_csv(OUT / f"mc_summary{suffix}.csv", index=False)

        print(f"Saved: {OUT}/mc_results{suffix}.parquet")
        print(f"Saved: {OUT}/mc_summary{suffix}.csv")
        return

    with Pool(n_workers) as pool:
        # imap_unordered streams results as they finish → progress feedback
        raw = []
        done = 0
        for result in pool.imap_unordered(_worker, tasks, chunksize=1):
            raw.append(result)
            done += 1
            if done % max(1, n_tasks // 20) == 0 or done == n_tasks:
                elapsed = time.time() - t0
                eta = elapsed / done * (n_tasks - done)
                print(f"  {done}/{n_tasks} done  "
                      f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]",
                      flush=True)

    elapsed = time.time() - t0
    print(f"\nAll runs finished in {elapsed:.1f}s  "
          f"({elapsed/n_tasks:.2f}s per run, {n_workers} workers)")

    # ── Save full results ──
    df_all = _assemble(raw, p.T)
    df_all.to_parquet(OUT / f"mc_results{suffix}.parquet", index=False)

    # ── Save summary (mean ± IQR per scenario per period) ──
    summary = (
        df_all.groupby(["scenario", "t"])
        .agg(
            Y_mean    = ("Y",      "mean"),
            Y_p25     = ("Y",      lambda x: np.quantile(x, 0.25)),
            Y_p75     = ("Y",      lambda x: np.quantile(x, 0.75)),
            G_mean    = ("G",      "mean"),
            G_p25     = ("G",      lambda x: np.quantile(x, 0.25)),
            G_p75     = ("G",      lambda x: np.quantile(x, 0.75)),
            alpha_mean= ("alpha",  "mean"),
            regime_frac=("regime", "mean"),
            congress_frac=("congress", "mean"),
            disint_rate=("disint", "mean"),
            poverty_mean=("poverty", "mean"),
            deaths_mean=("deaths", "mean"),
            mortality_mean=("mortality", "mean"),
            N_mean    = ("N",      "mean"),
        )
        .reset_index()
    )
    summary.to_parquet(OUT / f"mc_summary{suffix}.parquet", index=False)
    summary.to_csv(OUT / f"mc_summary{suffix}.csv", index=False)

    print(f"Saved: {OUT}/mc_results{suffix}.parquet")
    print(f"Saved: {OUT}/mc_summary{suffix}.csv")


if __name__ == "__main__":
    main()
