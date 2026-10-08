"""Step 14: tables and figures of the synthetic benchmark, read from outputs/13_benchmark only."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..io import read_csv, write_csv
from ..plotting import apply_style, color, label, save
from ..stats import a12, a12_magnitude, friedman_nemenyi, holm, mann_whitney
from .common import sdir
from .s11_figures import _cd_plot
from .s12_tables import to_latex

CODE_DEPS = ["co2sr.plotting", "co2sr.stats", "co2sr.steps.s11_figures", "co2sr.steps.s12_tables"]
NAME = "14_benchmark_report"
REFERENCE = "dcsr_v"


def config_part(cfg):
    return {"plotting": cfg["plotting"], "statistics": cfg["statistics"],
            "benchmark": {k: cfg["benchmark"][k] for k in ("problems", "methods")}}


def inputs(cfg):
    p = sdir(cfg, "13") / "benchmark_runs.csv"
    return [p] if p.exists() else []


def _order(cfg, present):
    return [m for m in cfg["benchmark"]["methods"] if m in present]


def run(cfg, log, **_):
    import matplotlib.pyplot as plt
    src = sdir(cfg, "13") / "benchmark_runs.csv"
    if not src.exists():
        log.warning("no benchmark runs; run step 13 first")
        return []
    df = read_csv(src)
    out = sdir(cfg, "14")
    outs = []
    alpha = cfg["statistics"]["alpha"]
    for c in ("test_nmse", "extrap_nmse", "interp_nmse"):
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df["bb_certified"] = (df["bb_verdict"] == "certified").astype(float)
    df["bb_refuted"] = (df["bb_verdict"] == "refuted").astype(float)
    df["bb_undecided"] = (df["bb_verdict"] == "undecided").astype(float)
    df["any_test_violation"] = (df["test_violation_share"] > 0).astype(float)
    df["ia_false_rejection"] = ((df["ia_certified"] == 0) & (df["bb_verdict"] == "certified")).astype(float)
    df["failed_numerically"] = df["test_nmse"].isna().astype(float)

    # 1. summary per problem and method --------------------------------------------------
    s = df.groupby(["problem", "method"]).agg(
        n_runs=("seed", "count"), test_nmse_median=("test_nmse", "median"),
        extrap_nmse_median=("extrap_nmse", "median"), interp_nmse_median=("interp_nmse", "median"),
        share_bb_certified=("bb_certified", "mean"), share_bb_refuted=("bb_refuted", "mean"),
        share_bb_undecided=("bb_undecided", "mean"), share_ia_certified=("ia_certified", "mean"),
        share_ia_false_rejection=("ia_false_rejection", "mean"),
        share_runs_test_violation=("any_test_violation", "mean"),
        share_numerical_failure=("failed_numerically", "mean"),
        n_nodes_median=("n_nodes", "median"), fit_time_median_s=("fit_time_s", "median")).reset_index()
    outs.append(write_csv(s, out / "benchmark_summary_by_problem.csv"))
    agg = df.groupby("method").agg(
        n_runs=("seed", "count"), share_bb_certified=("bb_certified", "mean"),
        share_bb_refuted=("bb_refuted", "mean"), share_bb_undecided=("bb_undecided", "mean"),
        share_ia_false_rejection=("ia_false_rejection", "mean"),
        share_runs_test_violation=("any_test_violation", "mean"),
        fit_time_median_s=("fit_time_s", "median"), n_nodes_median=("n_nodes", "median")).reset_index()
    agg = agg.set_index("method").loc[_order(cfg, set(agg["method"]))].reset_index()
    outs.append(write_csv(agg, out / "benchmark_summary_by_method.csv"))

    # 2. DCSR-V against every other method, per problem (extrapolation NMSE) --------------
    rows = []
    for pb, g in df.groupby("problem"):
        xa = g.loc[g["method"] == REFERENCE, "extrap_nmse"].dropna().to_numpy()
        for m in _order(cfg, set(g["method"])):
            if m == REFERENCE:
                continue
            xb = g.loc[g["method"] == m, "extrap_nmse"].dropna().to_numpy()
            if len(xa) and len(xb):
                A = a12(xa, xb)
                rows.append({"problem": pb, "method_a": REFERENCE, "method_b": m, "median_a": np.median(xa),
                             "median_b": np.median(xb), "A12": A, "A12_magnitude": a12_magnitude(A),
                             "p_value": mann_whitney(xa, xb)})
    pw = pd.DataFrame(rows)
    if len(pw):
        pw["p_holm"] = pw.groupby("problem", group_keys=False)["p_value"].apply(holm)
        pw["significant"] = (pw["p_holm"] < alpha).astype(int)
    outs.append(write_csv(pw, out / "benchmark_pairwise_dcsr_v.csv"))

    # 3. Friedman across problems on median extrapolation NMSE ------------------------------
    tab = s.pivot_table(index="problem", columns="method", values="extrap_nmse_median")
    tab = tab[_order(cfg, set(tab.columns))].dropna()
    fr = None
    if len(tab) >= 2 and tab.shape[1] >= 3:
        fr = friedman_nemenyi(tab, alpha)
        outs.append(write_csv(pd.DataFrame([{"n_problems": fr["n_blocks"], "k_methods": fr["k_methods"],
                                             "friedman_chi2": fr["statistic"], "p_value": fr["p_value"],
                                             "critical_difference": fr["cd"]}]),
                              out / "benchmark_friedman.csv"))
        outs.append(write_csv(pd.Series(fr["avg_ranks"], name="average_rank").rename_axis("method").reset_index(),
                              out / "benchmark_average_ranks.csv"))

    # 4. tables -------------------------------------------------------------------------
    t = s.copy()
    t["method"] = [label(cfg, m) for m in t["method"]]
    t = t[["problem", "method", "test_nmse_median", "extrap_nmse_median", "share_bb_certified",
           "share_bb_refuted", "share_runs_test_violation", "n_nodes_median", "fit_time_median_s"]]
    t.columns = ["Problem", "Method", "Test NMSE", "Extrap. NMSE", "Certified", "Refuted", "Runs violating",
                 "Nodes", "Time (s)"]
    outs += to_latex(t, out / "tab11_benchmark_results.tex", digits=3)
    a = agg.copy()
    a["method"] = [label(cfg, m) for m in a["method"]]
    a.columns = ["Method", "Runs", "Certified", "Refuted", "Undecided", "IA false rejection", "Runs violating",
                 "Time (s)", "Nodes"]
    outs += to_latex(a, out / "tab12_benchmark_verification.tex", digits=3)
    if len(pw):
        p2 = pw.copy()
        p2["method_b"] = [label(cfg, m) for m in p2["method_b"]]
        p2 = p2[["problem", "method_b", "median_a", "median_b", "A12", "A12_magnitude", "p_holm"]]
        p2.columns = ["Problem", "Compared with", "DCSR-V", "Other", "A12", "Magnitude", "p (Holm)"]
        outs += to_latex(p2, out / "tab13_benchmark_a12_dcsr_v.tex", digits=3)

    # 5. figures ------------------------------------------------------------------------
    apply_style(cfg)
    problems = [p for p in cfg["benchmark"]["problems"] if p in set(df["problem"])]
    methods = _order(cfg, set(df["method"]))
    ncol = min(4, len(problems))
    nrow = int(np.ceil(len(problems) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.3 * ncol + 0.6, 3.3 * nrow), squeeze=False)
    for i, pb in enumerate(problems):
        ax = axes[i // ncol][i % ncol]
        g = df[df["problem"] == pb]
        data = [g.loc[g["method"] == m, "extrap_nmse"].dropna().clip(lower=1e-8).to_numpy() for m in methods]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(color(cfg, m)); patch.set_alpha(0.8)
        for med in bp["medians"]:
            med.set_color("#222222")
        ax.set_yscale("log")
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels([label(cfg, m) for m in methods], rotation=60, ha="right", fontsize=7)
        ax.text(0.0, 1.02, pb, transform=ax.transAxes, va="bottom", ha="left", fontsize=9)
        if i % ncol == 0:
            ax.set_ylabel("Extrap. NMSE")
    for j in range(len(problems), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.tight_layout()
    outs += save(fig, out, "fig16_benchmark_extrapolation_nmse", cfg)

    fig, ax = plt.subplots(figsize=(5.5, 0.32 * len(methods) + 1.0))
    yv = np.arange(len(methods))
    left = np.zeros(len(methods))
    shades = {"certified": "#2a78d6", "undecided": "#bbbbbb", "refuted": "#e34948"}
    ag = agg.set_index("method").loc[methods]
    for v in ("certified", "undecided", "refuted"):
        w = 100 * ag[f"share_bb_{v}"].to_numpy()
        ax.barh(yv, w, left=left, color=shades[v], height=0.6, label=v.capitalize())
        left += w
    ax.set_yticks(yv)
    ax.set_yticklabels([label(cfg, m) for m in methods])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Final models over all problems and seeds (%)")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout()
    outs += save(fig, out, "fig17_benchmark_verification_verdicts", cfg)

    if fr is not None:
        fig, ax = plt.subplots(figsize=(6.5, 0.25 * len(methods) + 1.4))
        _cd_plot(ax, pd.Series(fr["avg_ranks"]), fr["cd"], cfg)
        outs += save(fig, out, "fig18_benchmark_critical_difference", cfg)
    log.info("%d benchmark report file(s) written to %s", len(outs), out)
    return outs
