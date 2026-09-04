"""
Generate figures for the extended model from the outputs in results/.
Includes trajectory figures and diagnostic figures for the model revisions:

  fig6_phase_diagram: heatmap of P(altruistic regime persists) over the
    (u, d_A) grid from sensitivity.py — the "formal threshold test" figure.
  fig7_hard_vs_smooth: side-by-side comparison of the original (hard
    threshold, non-adaptive) model against the extended one, run with the
    same scenarios, to show the qualitative results survive the revision.
  fig8_poverty_mortality: pre-transfer poverty count and mortality rate by
    scenario.
  fig9_hierarchy_ablation: side-by-side comparison of the fraction of
    replications with an inclusive congress, with and without the
    elite--congress override mechanism (Proposition 3 ablation), holding
    smooth thresholds, adaptation, and stochastic disintegration fixed.

Usage:  python make_figures.py [--fast]
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

OUT_FIG = Path("figures")
OUT_FIG.mkdir(exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       10,
    "axes.labelsize":  10,
    "axes.titlesize":  10,
    "legend.fontsize": 8,
    "figure.dpi":      150,
    "lines.linewidth": 1.2,
})

COLORS = {
    "altruistic":    "#2ca02c",
    "borderline":    "#1f77b4",
    "extractive":    "#d62728",
    "institutional": "#ff7f0e",
}
LABELS = {
    "altruistic":    r"Altruistic ($d_A=0.80 > u$)",
    "borderline":    r"Borderline ($d_A=0.55 < u$)",
    "extractive":    r"Extractive ($d_X=0.80$)",
    "institutional": r"Institutional ($d_I=0.80$)",
}
SCENARIOS = ["altruistic", "borderline", "extractive", "institutional"]


def load(fast: bool, suffix_extra: str = "") -> pd.DataFrame:
    suf = ("_fast" if fast else "") + suffix_extra
    p = Path("results") / f"mc_summary{suf}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"No results file at {p}. Run run_mc.py first.")
    return pd.read_parquet(p)


# ── Figures 1-4: same as the root model ───────────────────────────────────────

def fig_prosperity(df: pd.DataFrame, fast: bool):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for sc in SCENARIOS:
        d = df[df.scenario == sc]
        ax.plot(d.t, d.Y_mean, color=COLORS[sc], label=LABELS[sc])
        ax.fill_between(d.t, d.Y_p25, d.Y_p75, color=COLORS[sc], alpha=0.15)
    ax.set_xlabel("Period $t$")
    ax.set_ylabel("Per-capita spice $Y_t$")
    ax.set_title("Figure 1: Prosperity Trajectories by Scenario")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    fig.tight_layout()
    suf = "_fast" if fast else ""
    fig.savefig(OUT_FIG / f"fig1_prosperity{suf}.pdf")
    fig.savefig(OUT_FIG / f"fig1_prosperity{suf}.png")
    plt.close(fig)
    print("  Saved fig1_prosperity")


def fig_gini(df: pd.DataFrame, fast: bool):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for sc in SCENARIOS:
        d = df[df.scenario == sc]
        ax.plot(d.t, d.G_mean, color=COLORS[sc], label=LABELS[sc])
        ax.fill_between(d.t, d.G_p25, d.G_p75, color=COLORS[sc], alpha=0.15)
    ax.set_xlabel("Period $t$")
    ax.set_ylabel("Gini coefficient $G_t$")
    ax.set_title("Figure 2: Inequality Trajectories by Scenario")
    ax.set_ylim(0, None)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    suf = "_fast" if fast else ""
    fig.savefig(OUT_FIG / f"fig2_gini{suf}.pdf")
    fig.savefig(OUT_FIG / f"fig2_gini{suf}.png")
    plt.close(fig)
    print("  Saved fig2_gini")


def fig_alpha_regime(df: pd.DataFrame, fast: bool):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    ax = axes[0]
    for sc in SCENARIOS:
        d = df[df.scenario == sc]
        ax.plot(d.t, d.alpha_mean, color=COLORS[sc], label=LABELS[sc])
    ax.axhline(0.75, ls="--", color="k", lw=0.8, label="$u = 0.75$")
    ax.set_xlabel("Period $t$")
    ax.set_ylabel(r"Altruistic proportion $\alpha_t$")
    ax.set_title("(a) Altruistic Population Share")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7, framealpha=0.9)

    ax = axes[1]
    for sc in SCENARIOS:
        d = df[df.scenario == sc]
        ax.plot(d.t, 1 - d.regime_frac, color=COLORS[sc], label=LABELS[sc])
    ax.set_xlabel("Period $t$")
    ax.set_ylabel("Fraction of replications\nin altruistic regime")
    ax.set_title("(b) Altruistic Regime Persistence")
    ax.set_ylim(-0.05, 1.10)
    ax.axhline(1.0, color="gray", lw=0.5, ls=":")
    ax.axhline(0.0, color="gray", lw=0.5, ls=":")
    ax.legend(loc="center left", fontsize=7, framealpha=0.9)

    fig.suptitle("Figure 3: Moral Composition and Institutional Regime", fontsize=10)
    fig.tight_layout()
    suf = "_fast" if fast else ""
    fig.savefig(OUT_FIG / f"fig3_alpha_regime{suf}.pdf")
    fig.savefig(OUT_FIG / f"fig3_alpha_regime{suf}.png")
    plt.close(fig)
    print("  Saved fig3_alpha_regime")


def fig_disintegration(df: pd.DataFrame, fast: bool):
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    for sc in SCENARIOS:
        d = df[df.scenario == sc]
        rate = d.disint_rate.rolling(window=10, center=True, min_periods=1).mean()
        ax.plot(d.t.values, rate.values, color=COLORS[sc], label=LABELS[sc])
    ax.set_xlabel("Period $t$")
    ax.set_ylabel("Disintegration event rate\n(10-period rolling average)")
    ax.set_title("Figure 4: Elite Oversupply — Disintegration Events")
    ax.set_ylim(0, None)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    suf = "_fast" if fast else ""
    fig.savefig(OUT_FIG / f"fig4_disintegration{suf}.pdf")
    fig.savefig(OUT_FIG / f"fig4_disintegration{suf}.png")
    plt.close(fig)
    print("  Saved fig4_disintegration")


def fig_terminal_boxplot(fast: bool):
    suf = "_fast" if fast else ""
    p = Path("results") / f"mc_results{suf}.parquet"
    if not p.exists():
        print("  Skipping fig5 (full results file not found)")
        return

    df_full = pd.read_parquet(p)
    T_max = df_full.t.max()
    terminal = df_full[df_full.t >= T_max - 20].groupby(
        ["scenario", "rep"]
    ).agg(Y_term=("Y", "mean"), G_term=("G", "mean")).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, var, label in [
        (axes[0], "Y_term", "Terminal per-capita spice $Y_T$"),
        (axes[1], "G_term", "Terminal Gini coefficient $G_T$"),
    ]:
        data = [terminal[terminal.scenario == sc][var].values for sc in SCENARIOS]
        bp = ax.boxplot(data, patch_artist=True, medianprops=dict(color="black", lw=1.5))
        for patch, sc in zip(bp["boxes"], SCENARIOS):
            patch.set_facecolor(COLORS[sc])
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels(["Altruistic", "Borderline", "Extractive", "Institutional"],
                            rotation=12, fontsize=8)
        ax.set_ylabel(label)

    axes[0].set_title("(a) Prosperity")
    axes[1].set_title("(b) Inequality")
    fig.suptitle("Figure 5: Terminal Distributions across Monte Carlo Runs", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"fig5_terminal{suf}.pdf")
    fig.savefig(OUT_FIG / f"fig5_terminal{suf}.png")
    plt.close(fig)
    print("  Saved fig5_terminal")


# ── Figure 6: Phase diagram from sensitivity.py (u x d_A) ────────────────────

def fig_phase_diagram(fast: bool):
    suf = "_fast" if fast else ""
    p = Path("results") / f"sensitivity_u_dA_cells{suf}.csv"
    if not p.exists():
        print(f"  Skipping fig6 ({p} not found — run sensitivity.py first)")
        return

    df = pd.read_csv(p)
    u_vals = np.sort(df["u"].unique())
    dA_vals = np.sort(df["d_A"].unique())
    Z = df.pivot(index="d_A", columns="u", values="persistence") \
          .reindex(index=dA_vals, columns=u_vals).values

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    im = ax.pcolormesh(u_vals, dA_vals, Z, shading="auto", cmap="RdYlGn", vmin=0, vmax=1)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$P(\mathrm{altruistic\ regime\ persists})$")

    # 0.5-persistence contour = empirical critical threshold curve
    try:
        ax.contour(u_vals, dA_vals, Z, levels=[0.5], colors="black", linewidths=1.2)
    except Exception:
        pass

    ax.set_xlabel(r"Institutional threshold $u$")
    ax.set_ylabel(r"Initial altruistic share $d_A$")
    ax.set_title("Figure 6: Threshold Sensitivity — Phase Diagram")
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"fig6_phase_diagram{suf}.pdf")
    fig.savefig(OUT_FIG / f"fig6_phase_diagram{suf}.png")
    plt.close(fig)
    print("  Saved fig6_phase_diagram")


# ── Figure 7: Original (hard) vs extended (smooth+adaptive) comparison ───────

def fig_hard_vs_smooth(fast: bool):
    suf = "_fast" if fast else ""
    p_new = Path("results") / f"mc_summary{suf}.parquet"
    p_old = Path("results") / f"mc_summary{suf}_original.parquet"
    if not (p_new.exists() and p_old.exists()):
        print(f"  Skipping fig7 (need both {p_new} and {p_old} — "
              f"run run_mc.py and run_mc.py --original)")
        return

    df_new = pd.read_parquet(p_new)
    df_old = pd.read_parquet(p_old)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, df, title in [(axes[0], df_old, "(a) Original (hard threshold)"),
                           (axes[1], df_new, "(b) Extended (smooth + adaptive)")]:
        for sc in SCENARIOS:
            d = df[df.scenario == sc]
            ax.plot(d.t, d.alpha_mean, color=COLORS[sc], label=LABELS[sc])
        ax.axhline(0.75, ls="--", color="k", lw=0.8)
        ax.set_xlabel("Period $t$")
        ax.set_ylabel(r"Altruistic proportion $\alpha_t$")
        ax.set_title(title)
        ax.set_ylim(0, 1)
    axes[0].legend(fontsize=6, framealpha=0.9, loc="lower left")

    fig.suptitle("Figure 7: Robustness of Qualitative Results to the Revision", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"fig7_hard_vs_smooth{suf}.pdf")
    fig.savefig(OUT_FIG / f"fig7_hard_vs_smooth{suf}.png")
    plt.close(fig)
    print("  Saved fig7_hard_vs_smooth")


# -- Figure 8: Poverty and mortality (defined in the paper, previously unreported) --

def fig_poverty_mortality(df: pd.DataFrame, fast: bool):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    ax = axes[0]
    for sc in SCENARIOS:
        d = df[df.scenario == sc]
        rate = d.poverty_mean.rolling(window=10, center=True, min_periods=1).mean()
        ax.plot(d.t.values, rate.values, color=COLORS[sc], label=LABELS[sc])
    ax.set_xlabel("Period $t$")
    ax.set_ylabel("Agents in poverty\n(10-period rolling average)")
    ax.set_title(r"(a) Pre-transfer Poverty $N^{\mathrm{pov}}_t$")
    ax.set_ylim(0, None)
    ax.legend(fontsize=7, framealpha=0.9)

    ax = axes[1]
    for sc in SCENARIOS:
        d = df[df.scenario == sc]
        rate = d.mortality_mean.rolling(window=10, center=True, min_periods=1).mean()
        ax.plot(d.t.values, rate.values, color=COLORS[sc], label=LABELS[sc])
    ax.set_xlabel("Period $t$")
    ax.set_ylabel("Mortality rate\n(10-period rolling average)")
    ax.set_title(r"(b) Mortality Rate $\mu_t$")
    ax.set_ylim(0, None)
    ax.legend(fontsize=7, framealpha=0.9)

    fig.suptitle("Figure 8: Poverty and Mortality by Scenario", fontsize=10)
    fig.tight_layout()
    suf = "_fast" if fast else ""
    fig.savefig(OUT_FIG / f"fig8_poverty_mortality{suf}.pdf")
    fig.savefig(OUT_FIG / f"fig8_poverty_mortality{suf}.png")
    plt.close(fig)
    print("  Saved fig8_poverty_mortality")


# -- Figure 9: Elite-congress hierarchy ablation (Proposition 3) --

def fig_hierarchy_ablation(fast: bool):
    suf = "_fast" if fast else ""
    p_with = Path("results") / f"mc_summary{suf}.parquet"
    p_without = Path("results") / f"mc_summary{suf}_nohierarchy.parquet"
    if not (p_with.exists() and p_without.exists()):
        print(f"  Skipping fig9 (need both {p_with} and {p_without} -- "
              f"run run_mc.py and run_mc.py --no-hierarchy)")
        return

    df_with = pd.read_parquet(p_with)
    df_without = pd.read_parquet(p_without)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, df, title in [(axes[0], df_without, "(a) Without elite override"),
                           (axes[1], df_with, "(b) With elite override")]:
        for sc in SCENARIOS:
            d = df[df.scenario == sc]
            ax.plot(d.t, 1 - d.congress_frac, color=COLORS[sc], label=LABELS[sc])
        ax.set_xlabel("Period $t$")
        ax.set_ylabel("Fraction of replications\nwith inclusive congress")
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize=6, framealpha=0.9, loc="upper right")

    fig.suptitle("Figure 9: Elite-Congress Hierarchy Ablation (Proposition 3)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_FIG / f"fig9_hierarchy_ablation{suf}.pdf")
    fig.savefig(OUT_FIG / f"fig9_hierarchy_ablation{suf}.png")
    plt.close(fig)
    print("  Saved fig9_hierarchy_ablation")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    print("Loading summary data ...")
    df = load(args.fast)
    print(f"  {len(df):,} rows, scenarios: {df.scenario.unique().tolist()}")
    print("Generating figures ...")

    fig_prosperity(df, args.fast)
    fig_gini(df, args.fast)
    fig_alpha_regime(df, args.fast)
    fig_disintegration(df, args.fast)
    fig_terminal_boxplot(args.fast)
    fig_phase_diagram(args.fast)
    fig_hard_vs_smooth(args.fast)
    fig_poverty_mortality(df, args.fast)
    fig_hierarchy_ablation(args.fast)

    print(f"Done. Figures saved to {OUT_FIG}/")


if __name__ == "__main__":
    main()
