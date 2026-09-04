"""
Systematic threshold / bifurcation sensitivity analysis — response to the
JASSS observation that "there is weak systematic analysis of model
parameters... no test of the crucial parameter thresholds."

Two 2D sweeps, each run over a grid of (param_x, param_y) with R replications
per cell:

  1. u (institutional threshold) x d_A (initial altruistic share)
  2. sigma (intra-faction sharing rate) x plus_rate (elite extraction rate)

For each cell we record, averaged over replications:
  - persistence: P(regime is altruistic in the final period)
  - alpha_final, G_final: mean over the last 20 periods
  - disint_rate: mean disintegration-event rate over the full run

We then fit a logistic curve to persistence as a function of the swept
parameter (holding the other fixed per row/column) via a linear fit in
logit-space, and report the estimated critical threshold (P=0.5 crossing)
and its slope — the formal test of the critical threshold the observation
asks for.

Usage:
    python sensitivity.py            # full grid
    python sensitivity.py --fast     # small grid, quick validation
    python sensitivity.py --workers 4
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

BASE_SEED = 20250624
TERMINAL_WINDOW = 20   # periods averaged for "final" alpha/G


# ─── Worker ──────────────────────────────────────────────────────────────────

def _worker(args: tuple) -> dict:
    (sweep, x_name, x_val, y_name, y_val, d_A, d_X, rep, seed, p) = args
    result = run(d_A, d_X, seed=seed, p=p)
    live_periods = np.where(result["N"] > 0)[0]
    if len(live_periods) == 0:
        last = 0
        terminal = slice(0, 0)
        extinct = True
    else:
        last = int(live_periods[-1])
        first = max(0, last - TERMINAL_WINDOW + 1)
        terminal = slice(first, last + 1)
        extinct = last < p.T - 1
    return {
        "sweep":  sweep,
        x_name:   x_val,
        y_name:   y_val,
        "rep":    rep,
        "regime_final": float(result["regime"][last]) if len(live_periods) else np.nan,
        "alpha_final":  float(np.nanmean(result["alpha"][terminal])) if len(live_periods) else np.nan,
        "G_final":      float(np.nanmean(result["G"][terminal])) if len(live_periods) else np.nan,
        "disint_rate":  float(np.nanmean(result["disint"][live_periods])) if len(live_periods) else np.nan,
        "extinct":      extinct,
        "last_live_t":  last if len(live_periods) else np.nan,
    }


def _build_tasks(sweep: str, x_name: str, x_vals, y_name: str, y_vals,
                  R: int, T: int, param_setter) -> list:
    """param_setter(p, x_val, y_val) -> (p, d_A, d_X) configures one cell."""
    tasks = []
    cell_idx = 0
    for x_val in x_vals:
        for y_val in y_vals:
            # deterministic per-cell sub-stream, independent of iteration order
            cell_ss = SeedSequence([BASE_SEED, cell_idx])
            child_seeds = [int(s.generate_state(1)[0]) for s in cell_ss.spawn(R)]
            for rep in range(R):
                p = Params()
                p.T = T
                p, d_A, d_X = param_setter(p, x_val, y_val)
                tasks.append((sweep, x_name, x_val, y_name, y_val,
                              d_A, d_X, rep, child_seeds[rep], p))
            cell_idx += 1
    return tasks


# ─── Sweep 1: u x d_A ──────────────────────────────────────────────────────

def _setter_u_dA(p: Params, u_val: float, dA_val: float):
    p.u = u_val
    d_A = dA_val
    d_X = 0.15
    return p, d_A, d_X


# ─── Sweep 2: sigma x plus_rate ──────────────────────────────────────────────

def _setter_sigma_rho(p: Params, sigma_val: float, rho_val: float):
    p.sigma = sigma_val
    p.sigma_hat = sigma_val / 1.5   # keep the intra/inter ratio used elsewhere
    p.plus_rate = rho_val
    d_A, d_X = 0.65, 0.15           # near the u=0.75 borderline region
    return p, d_A, d_X


# ─── Logistic (logit-space) threshold fit ───────────────────────────────────

def _fit_critical_threshold(df: pd.DataFrame, x_name: str, group_name: str = None
                             ) -> pd.DataFrame:
    """
    For each level of `group_name` (or the whole df if None), fit
    logit(persistence) = a + b*x by OLS on the per-x aggregated persistence
    rate, and report x0 = -a/b (the P=0.5 crossing) and slope b.
    """
    rows = []
    groups = df.groupby(group_name) if group_name else [(None, df)]
    for gval, gdf in groups:
        agg = gdf.groupby(x_name)["regime_final"].apply(
            lambda s: 1.0 - s.mean()   # regime_final: 0=altruistic -> persistence=1-mean
        ).reset_index(name="persistence")
        eps = 1e-3
        p_clip = agg["persistence"].clip(eps, 1 - eps)
        logit = np.log(p_clip / (1 - p_clip))
        if agg[x_name].nunique() < 2 or logit.nunique() < 2:
            continue
        b, a = np.polyfit(agg[x_name].values, logit.values, 1)
        x0 = -a / b if b != 0 else np.nan
        rows.append({
            group_name or "group": gval,
            "x0_critical": x0,
            "slope": b,
            "n_x_levels": agg[x_name].nunique(),
        })
    return pd.DataFrame(rows)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Small grid, few replications, short T — for validation")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    if args.fast:
        R, T = 4, 100
        u_vals  = np.linspace(0.60, 0.85, 4)
        dA_vals = np.linspace(0.40, 0.85, 5)
        sigma_vals = np.linspace(1/12, 1/3, 4)
        rho_vals   = np.linspace(0.02, 0.10, 4)
    else:
        R, T = 20, 300
        u_vals  = np.linspace(0.55, 0.90, 8)
        dA_vals = np.linspace(0.30, 0.90, 13)
        sigma_vals = np.linspace(1/12, 1/3, 8)
        rho_vals   = np.linspace(0.01, 0.15, 8)

    n_workers = args.workers or cpu_count()
    suffix = "_fast" if args.fast else ""

    sweeps = [
        ("u_dA",       "u",     u_vals,  "d_A",       dA_vals, _setter_u_dA),
        ("sigma_rho",  "sigma", sigma_vals, "plus_rate", rho_vals, _setter_sigma_rho),
    ]

    for sweep_name, x_name, x_vals, y_name, y_vals, setter in sweeps:
        tasks = _build_tasks(sweep_name, x_name, x_vals, y_name, y_vals, R, T, setter)
        n_tasks = len(tasks)
        print(f"[{sweep_name}] {len(x_vals)}x{len(y_vals)} grid x {R} reps "
              f"= {n_tasks} tasks on {n_workers} workers (T={T})")

        t0 = time.time()
        raw = []
        done = 0
        if n_workers == 1:
            iterator = map(_worker, tasks)
        else:
            pool = Pool(n_workers)
            iterator = pool.imap_unordered(_worker, tasks, chunksize=1)

        try:
            for rec in iterator:
                raw.append(rec)
                done += 1
                if done % max(1, n_tasks // 10) == 0 or done == n_tasks:
                    elapsed = time.time() - t0
                    eta = elapsed / done * (n_tasks - done)
                    print(f"  {done}/{n_tasks} done "
                          f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]", flush=True)
        finally:
            if n_workers != 1:
                pool.close()
                pool.join()

        df = pd.DataFrame(raw)
        df.to_parquet(OUT / f"sensitivity_{sweep_name}{suffix}.parquet", index=False)

        cell = (
            df.groupby([x_name, y_name])
            .agg(
                persistence = ("regime_final", lambda s: 1.0 - s.mean()),
                alpha_final = ("alpha_final",  "mean"),
                G_final     = ("G_final",      "mean"),
                disint_rate = ("disint_rate",  "mean"),
                extinction_rate = ("extinct",  "mean"),
                last_live_t  = ("last_live_t", "mean"),
                n_reps      = ("regime_final", "count"),
            )
            .reset_index()
        )
        cell.to_csv(OUT / f"sensitivity_{sweep_name}_cells{suffix}.csv", index=False)
        print(f"  Saved results/sensitivity_{sweep_name}{suffix}.parquet "
              f"and _cells{suffix}.csv")

        # Critical threshold: for u_dA, fit u0 per d_A level; for sigma_rho,
        # fit plus_rate0 per sigma level.
        crit = _fit_critical_threshold(df, x_name, group_name=y_name)
        crit.to_csv(OUT / f"critical_threshold_{sweep_name}{suffix}.csv", index=False)
        print(f"  Saved results/critical_threshold_{sweep_name}{suffix}.csv")
        print(crit.to_string(index=False))
        print()

    print("Done.")


if __name__ == "__main__":
    main()
