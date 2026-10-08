"""Step 11: publication figures, generated only from files written by earlier steps.

Each figure is saved as PNG and PDF (300 dpi) under results/figures/, without a
title, named <number>_<content>_<dataset>[_<protocol>].  A figure whose inputs
are missing is skipped with a warning, so this step can run at any stage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import enabled_datasets
from ..io import read_csv
from ..plotting import DCSR_MARKERS, apply_style, color, label, save
from .common import clean_file, load_clean, sdir

CODE_DEPS = ['co2sr.plotting']
NAME = "11_figures"
FUEL_ORDER = ["X", "Z", "D", "E"]
FUEL_COLORS = {"X": "#2a78d6", "Z": "#eb6834", "D": "#1baf7a", "E": "#eda100"}
FUEL_LABELS = {"X": "Regular gasoline", "Z": "Premium gasoline", "D": "Diesel", "E": "E85"}
SET_LABELS = {"all": "all", "new_config": "new configurations", "low_tail": "lower tail",
              "high_tail": "upper tail", "newer": "newer group", "cyl3": "3-cylinder"}


def config_part(cfg):
    return {"plotting": cfg["plotting"], "statistics": cfg["statistics"]}


def inputs(cfg):
    d8, d9 = sdir(cfg, "08"), sdir(cfg, "09")
    return [p for p in [*(clean_file(cfg, ds) for ds in enabled_datasets(cfg)), d8 / "run_summary.csv",
                        d8 / "run_metrics.csv", d9 / "summary_by_block.csv"] if p.exists()]


def _methods_in_order(cfg, present):
    order = list(cfg["statistics"]["main_methods"]) + [m for m in cfg["methods"]]
    seen, out = set(), []
    for m in order:
        if m in present and m not in seen:
            out.append(m)
            seen.add(m)
    return out


# ----------------------------------------------------------------------------- data
def fig_rows_by_year(cfg, out, ds, plt):
    df = load_clean(cfg, ds)
    if df["model_year"].isna().all():
        return []
    tab = df.pivot_table(index="model_year", columns="fuel_type", values="row_id", aggfunc="count", fill_value=0)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    bottom = np.zeros(len(tab))
    for f in [f for f in FUEL_ORDER if f in tab.columns]:
        ax.bar(tab.index.astype(int), tab[f], bottom=bottom, color=FUEL_COLORS[f], label=FUEL_LABELS[f],
               edgecolor="white", linewidth=0.8, width=0.8)
        bottom += tab[f].to_numpy()
    ax.set_xlabel("Model year")
    ax.set_ylabel("Vehicle records")
    ax.set_xticks(tab.index.astype(int))
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    return save(fig, out, f"fig01_records_by_model_year_and_fuel_{ds}", cfg)


def fig_co2_vs_engine(cfg, out, ds, plt):
    df = load_clean(cfg, ds)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for f in [f for f in FUEL_ORDER if f in df["fuel_type"].unique()]:
        s = df[df["fuel_type"] == f]
        ax.scatter(s["engine_size"], s["co2"], s=8, alpha=0.45, color=FUEL_COLORS[f], label=FUEL_LABELS[f],
                   edgecolors="none")
    ax.set_xlabel("Engine size (L)")
    ax.set_ylabel("CO$_2$ emissions (g/km)")
    ax.legend(frameon=False, markerscale=2)
    return save(fig, out, f"fig02_co2_vs_engine_size_by_fuel_{ds}", cfg)


def fig_prior_by_year(cfg, out, ds, plt):
    p = sdir(cfg, "03") / f"{ds}_prior_support.csv"
    if not p.exists():
        return []
    t = read_csv(p)
    t = t[t["group"].str.startswith("model_year=")]
    if t.empty:
        return []
    t["year"] = t["group"].str.split("=").str[1].astype(float).astype(int)
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.bar(t["year"], 100 * t["violation_share"], color="#2a78d6", width=0.7)
    for x, y, n in zip(t["year"], 100 * t["violation_share"], t["pairs"]):
        ax.text(x, y + 0.5, f"n={n}", ha="center", va="bottom", fontsize=7, color="#555555")
    ax.set_xlabel("Model year")
    ax.set_ylabel("Matched pairs with larger engine\nand lower CO$_2$ (%)")
    ax.set_xticks(t["year"])
    return save(fig, out, f"fig03_prior_violation_share_by_year_{ds}", cfg)


def fig_proxy_k(cfg, out, ds, plt):
    p = sdir(cfg, "04") / f"{ds}_proxy_k_selection.csv"
    if not p.exists():
        return []
    t = read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    axes[0].plot(t["k"], t["silhouette"], marker="o", color="#2a78d6", markersize=4)
    sel = t[t["selected"] == 1]
    axes[0].scatter(sel["k"], sel["silhouette"], s=60, facecolors="none", edgecolors="#e34948", zorder=3)
    axes[0].set_xlabel("Number of clusters $K$")
    axes[0].set_ylabel("Silhouette score")
    axes[1].plot(t["k"], t["inertia"], marker="o", color="#2a78d6", markersize=4)
    axes[1].set_xlabel("Number of clusters $K$")
    axes[1].set_ylabel("Inertia")
    fig.tight_layout()
    return save(fig, out, f"fig04_proxy_k_selection_{ds}", cfg)


def fig_proxy_vs_year(cfg, out, ds, plt):
    p = sdir(cfg, "04") / f"{ds}_proxy_assignments.csv"
    if not p.exists():
        return []
    a = read_csv(p).merge(load_clean(cfg, ds)[["row_id", "model_year"]], on="row_id")
    if a["model_year"].isna().all():
        return []
    years = sorted(a["model_year"].dropna().unique())
    data = [a.loc[a["model_year"] == y, "technology_score"].to_numpy() for y in years]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    bp = ax.boxplot(data, positions=range(len(years)), widths=0.6, patch_artist=True, showfliers=False)
    for b in bp["boxes"]:
        b.set(facecolor="#cde2fb", edgecolor="#2a78d6")
    for k in ("whiskers", "caps"):
        for l in bp[k]:
            l.set_color("#2a78d6")
    for m in bp["medians"]:
        m.set_color("#104281")
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(int(y)) for y in years])
    ax.set_xlabel("Model year")
    ax.set_ylabel("Technology score of cluster\n(higher = classed as older)")
    return save(fig, out, f"fig05_proxy_technology_score_by_model_year_{ds}", cfg)


# ----------------------------------------------------------------------------- results
def fig_rmse_boxes(cfg, out, plt):
    met = read_csv(sdir(cfg, "08") / "run_metrics.csv")
    paths = []
    for (ds, p), g in met[met["set"] != "val"].groupby(["dataset", "protocol"]):
        g = g[g["method"].isin(cfg["statistics"]["main_methods"])]
        if g.empty:
            continue
        methods = _methods_in_order(cfg, set(g["method"]))
        units = g[["block", "set"]].drop_duplicates().sort_values(["block", "set"]).to_records(index=False)
        n = len(units)
        ncol = min(3, n)
        nrow = int(np.ceil(n / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.6 * nrow + 0.6), squeeze=False, sharey=False)
        for ax, (b, s) in zip(axes.flat, units):
            sub = g[(g["block"] == b) & (g["set"] == s)]
            data = [sub.loc[sub["method"] == m, "rmse"].dropna().to_numpy() for m in methods]
            for i, (m, x) in enumerate(zip(methods, data)):
                if len(x) == 0:
                    continue
                if len(x) == 1:
                    ax.scatter([i], x, color=color(cfg, m), s=22, zorder=3, marker="D")
                else:
                    bp = ax.boxplot([x], positions=[i], widths=0.6, patch_artist=True, showfliers=False)
                    bp["boxes"][0].set(facecolor=color(cfg, m), alpha=0.35, edgecolor=color(cfg, m))
                    for k in ("whiskers", "caps"):
                        for l in bp[k]:
                            l.set_color(color(cfg, m))
                    bp["medians"][0].set_color(color(cfg, m))
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels([label(cfg, m) for m in methods], rotation=60, ha="right", fontsize=8)
            ax.text(0.0, 1.02, f"{b}, {SET_LABELS.get(s, s)}", transform=ax.transAxes, va="bottom", ha="left",
                    fontsize=8)
            ax.set_ylabel("Test RMSE (g/km)")
        for ax in axes.flat[n:]:
            ax.axis("off")
        fig.tight_layout()
        paths += save(fig, out, f"fig06_test_rmse_by_method_{ds}_{p}", cfg)
    return paths


def _cd_plot(ax, ranks: pd.Series, cd: float, cfg):
    k = len(ranks)
    ranks = ranks.sort_values()
    lo, hi = 1, max(k, 2)
    ax.set_xlim(lo - 0.3, hi + 0.3)
    ax.axis("off")
    y0 = 0.85
    ax.plot([lo, hi], [y0, y0], color="#333333", linewidth=1)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [y0, y0 + 0.03], color="#333333", linewidth=1)
        ax.text(r, y0 + 0.05, str(r), ha="center", va="bottom", fontsize=9)
    # CD bar
    ax.plot([lo, lo + cd], [0.98, 0.98], color="#333333", linewidth=1.5)
    ax.text(lo + cd / 2, 1.0, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8)
    vals = ranks.to_numpy()
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for n, (i, j) in enumerate(cliques):
        yy = y0 - 0.05 - 0.035 * n
        ax.plot([vals[i] - 0.03, vals[j] + 0.03], [yy, yy], color="#333333", linewidth=2.5)
    top = y0 - 0.08 - 0.035 * len(cliques)
    names = list(ranks.index)
    half = int(np.ceil(k / 2))
    for i, m in enumerate(names):
        r = ranks[m]
        if i < half:
            yy = top - 0.1 * i
            ax.plot([r, r, lo - 0.1], [y0, yy, yy], color=color(cfg, m), linewidth=1, zorder=1)
            ax.text(lo - 0.15, yy, f"{label(cfg, m)} ({r:.2f})", ha="right", va="center", fontsize=8)
        else:
            j = k - 1 - i
            yy = top - 0.1 * j
            ax.plot([r, r, hi + 0.1], [y0, yy, yy], color=color(cfg, m), linewidth=1, zorder=1)
            ax.text(hi + 0.15, yy, f"({r:.2f}) {label(cfg, m)}", ha="left", va="center", fontsize=8)
    ax.set_ylim(top - 0.1 * half, 1.05)


def fig_cd(cfg, out, plt):
    p = sdir(cfg, "09") / "friedman_average_ranks.csv"
    if not p.exists():
        return []
    t = read_csv(p)
    paths = []
    for (gname, ds), g in t.groupby(["group", "dataset"]):
        ranks = g.set_index("method")["average_rank"]
        fig, ax = plt.subplots(figsize=(7.0, 1.2 + 0.3 * np.ceil(len(ranks) / 2)))
        _cd_plot(ax, ranks, float(g["critical_difference"].iloc[0]), cfg)
        paths += save(fig, out, f"fig07_critical_difference_{gname}_{ds}", cfg)
    return paths


def fig_domain_validity(cfg, out, plt):
    p = sdir(cfg, "09") / "constraint_validity.csv"
    if not p.exists():
        return []
    t = read_csv(p)
    paths = []
    for (ds, pr), g in t.groupby(["dataset", "protocol"]):
        methods = _methods_in_order(cfg, set(g["method"]) & set(cfg["experiments"].get("main", {}).get("methods", g["method"])))
        if not methods:
            methods = _methods_in_order(cfg, set(g["method"]))
        g = g.set_index("method").loc[methods]
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 0.35 * len(methods) + 1.0), sharey=True)
        yv = np.arange(len(methods))
        axes[0].barh(yv, 100 * g["share_runs_domain_valid"], color=[color(cfg, m) for m in methods], height=0.6)
        axes[0].set_xlabel("Runs with no violation on the domain grid (%)")
        axes[0].set_xlim(0, 100)
        axes[1].barh(yv, 100 * g["domain_violation_rate_median"], color=[color(cfg, m) for m in methods], height=0.6)
        axes[1].set_xlabel("Grid points with negative slope,\nmedian over runs (%)")
        axes[0].set_yticks(yv)
        axes[0].set_yticklabels([label(cfg, m) for m in methods])
        axes[0].invert_yaxis()
        fig.tight_layout()
        paths += save(fig, out, f"fig08_constraint_validity_{ds}_{pr}", cfg)
    return paths


def fig_rmse_vs_complexity(cfg, out, plt):
    summ = read_csv(sdir(cfg, "08") / "run_summary.csv")
    met = read_csv(sdir(cfg, "08") / "run_metrics.csv")
    if "n_nodes_simplified" not in summ:
        return []
    keys = ["dataset", "protocol", "block", "method", "seed"]
    m = met[~met["set"].isin(["val", "new_config"])].groupby(keys)["rmse"].mean().reset_index()
    d = summ.merge(m, on=keys)
    d = d[d["method"].isin(["gp_standard", "dcsr_t", "dcsr_s", "dcsr_f", "dcsr_ia"])]
    paths = []
    for (ds, pr), g in d.groupby(["dataset", "protocol"]):
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        for meth in _methods_in_order(cfg, set(g["method"])):
            s = g[g["method"] == meth]
            ax.scatter(s["n_nodes_simplified"], s["rmse"], s=14, alpha=0.6, color=color(cfg, meth),
                       marker=DCSR_MARKERS.get(meth, "o"), label=label(cfg, meth), edgecolors="none")
        ax.set_xlabel("Nodes in simplified equation")
        ax.set_ylabel("Test RMSE (g/km), mean over test sets")
        ax.legend(frameon=False)
        paths += save(fig, out, f"fig09_test_rmse_vs_equation_size_{ds}_{pr}", cfg)
    return paths


def fig_response(cfg, out, plt):
    p = sdir(cfg, "08") / "response_curves.csv"
    if not p.exists():
        return []
    cur = read_csv(p)
    met = read_csv(sdir(cfg, "08") / "run_metrics.csv")
    val = met[met["set"] == "val"]
    paths = []
    main = cfg["statistics"]["main_methods"]
    for (ds, pr), g in cur[cur["method"].isin(main)].groupby(["dataset", "protocol"]):
        block = sorted(g["block"].unique())[0]
        g = g[g["block"] == block]
        cyls = sorted(g["curve_cylinders"].unique())
        fig, axes = plt.subplots(1, len(cyls), figsize=(3.0 * len(cyls), 3.0), sharey=False, squeeze=False)
        for ax, c in zip(axes[0], cyls):
            for meth in _methods_in_order(cfg, set(g["method"])):
                v = val[(val["dataset"] == ds) & (val["protocol"] == pr) & (val["block"] == block)
                        & (val["method"] == meth)].sort_values("rmse")
                if v.empty:
                    continue
                seed = int(v.iloc[(len(v) - 1) // 2]["seed"])       # median-validation run
                s = g[(g["method"] == meth) & (g["seed"] == seed) & (g["curve_cylinders"] == c)]
                ax.plot(s["engine_size"], s["yhat"], color=color(cfg, meth), label=label(cfg, meth),
                        linewidth=1.5 if meth.startswith("dcsr") else 1.2)
            ax.set_xlabel("Engine size (L)")
            ax.text(0.03, 0.95, f"{c} cylinders", transform=ax.transAxes, va="top", fontsize=9)
        axes[0][0].set_ylabel("Predicted CO$_2$ (g/km)")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
        fig.tight_layout()
        paths += save(fig, out, f"fig10_engine_size_response_{ds}_{pr}_{block}", cfg)
    return paths


def fig_lambda(cfg, out, plt):
    p = sdir(cfg, "09") / "lambda_ablation.csv"
    if not p.exists():
        return []
    t = read_csv(p)
    order = cfg["statistics"]["lambda_ablation"]
    paths = []
    for (ds, pr), g in t.groupby(["dataset", "protocol"]):
        g = g.copy()
        g["lam"] = [float(cfg["methods"].get(m, {}).get("lam", 0.0)) for m in g["method"]]
        g = g.sort_values("lam")
        g["x"] = 100 * g["domain_violation_rate_median"]
        fig, ax = plt.subplots(figsize=(5.0, 3.4))
        ax.plot(g["x"], g["val_rmse_median"], color="#9a9a9a", linewidth=1, zorder=1)
        labels_at = {}
        for _, r in g.iterrows():
            ax.scatter(r["x"], r["val_rmse_median"], color=color(cfg, r["method"]), s=36, zorder=2)
            txt = "Standard GP ($\\lambda$=0)" if r["method"] == "gp_standard" else f"$\\lambda$={r['lam']:g}"
            labels_at.setdefault((round(r["x"], 3), round(r["val_rmse_median"], 3)), []).append(txt)
        for (x, y), txts in labels_at.items():
            ax.annotate(", ".join(txts), (x, y), textcoords="offset points", xytext=(6, 3), fontsize=8)
        xr = max(1.0, 1.15 * g["x"].max())
        ax.set_xlim(-0.04 * xr, xr)
        ax.set_xlabel("Negative-slope grid points, median (%)")
        ax.set_ylabel("Validation RMSE (g/km), median")
        paths += save(fig, out, f"fig11_lambda_tradeoff_{ds}_{pr}", cfg)
    return paths


def fig_verdicts(cfg, out, plt):
    """Branch-and-bound verdict of the final model of every GP run (certified / undecided / refuted)."""
    p = sdir(cfg, "09") / "constraint_validity.csv"
    if not p.exists():
        return []
    t = read_csv(p)
    if "share_bb_certified" not in t:
        return []
    t = t[t["share_bb_certified"].notna()]
    paths = []
    shades = {"certified": "#2a78d6", "undecided": "#bbbbbb", "refuted": "#e34948"}
    for (ds, pr), g in t.groupby(["dataset", "protocol"]):
        methods = _methods_in_order(cfg, set(g["method"]))
        g = g.set_index("method").loc[methods]
        fig, ax = plt.subplots(figsize=(5.5, 0.32 * len(methods) + 1.0))
        yv = np.arange(len(methods))
        left = np.zeros(len(methods))
        for v in ("certified", "undecided", "refuted"):
            w = 100 * g[f"share_bb_{v}"].fillna(0).to_numpy()
            ax.barh(yv, w, left=left, color=shades[v], height=0.6, label=v.capitalize())
            left += w
        ax.set_yticks(yv)
        ax.set_yticklabels([label(cfg, m) for m in methods])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Final models (%)")
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.25 if len(methods) < 6 else -0.12))
        fig.tight_layout()
        paths += save(fig, out, f"fig15_verification_verdicts_{ds}_{pr}", cfg)
    return paths


def fig_convergence(cfg, out, plt):
    p = sdir(cfg, "08") / "gp_convergence.csv"
    if not p.exists():
        return []
    h = read_csv(p)
    h = h[h["method"].isin(["gp_standard", "dcsr_t", "dcsr_s", "dcsr_f", "dcsr_ia", "dcsr_v", "dcsr_dyn"])]
    paths = []
    for (ds, pr), g in h.groupby(["dataset", "protocol"]):
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0))
        for meth in _methods_in_order(cfg, set(g["method"])):
            s = g[g["method"] == meth].groupby("generation").agg(nmse=("best_nmse", "median"),
                                                                  feas=("feasible_share", "median"))
            ls = "-" if meth == "gp_standard" else (0, (4, 1 + list(DCSR_MARKERS).index(meth)))
            mk = DCSR_MARKERS.get(meth, None)
            every = max(1, len(s) // 8)
            axes[0].plot(s.index, s["nmse"], color=color(cfg, meth), label=label(cfg, meth), linestyle=ls,
                         marker=mk, markevery=every, markersize=4)
            if meth != "gp_standard":
                axes[1].plot(s.index, 100 * s["feas"], color=color(cfg, meth), label=label(cfg, meth),
                             linestyle=ls, marker=mk, markevery=every, markersize=4)
        axes[0].set_xlabel("Generation")
        axes[0].set_ylabel("Training NMSE of best individual")
        axes[1].set_xlabel("Generation")
        axes[1].set_ylabel("Feasible individuals (%)")
        handles, labels_ = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels_, frameon=False, ncol=len(labels_), loc="upper center",
                   bbox_to_anchor=(0.5, 1.08))
        fig.tight_layout()
        paths += save(fig, out, f"fig12_gp_convergence_{ds}_{pr}", cfg)
    return paths


def fig_audit(cfg, out, plt):
    p = sdir(cfg, "10") / "audit_by_make.csv"
    if not p.exists():
        return []
    t = read_csv(p).sort_values("median_residual")
    fig, ax = plt.subplots(figsize=(5.0, 0.22 * len(t) + 1.0))
    y = np.arange(len(t))
    ax.errorbar(t["median_residual"], y, xerr=[t["median_residual"] - t["ci_low"], t["ci_high"] - t["median_residual"]],
                fmt="o", color="#2a78d6", ecolor="#86b6ef", markersize=4, capsize=0)
    ax.axvline(0, color="#6b6b6b", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(t["make"])
    ax.set_xlabel("Median residual, observed minus predicted (g/km)")
    a = cfg["audit"]
    return save(fig, out, f"fig13_residual_by_make_{a['dataset']}_{a['protocol']}_{a['method']}", cfg)


def fig_ceiling(cfg, out, plt):
    p = sdir(cfg, "09") / "summary_by_block.csv"
    if not p.exists():
        return []
    t = read_csv(p)
    cols = [c for c in ("r2_ceiling_F1", "r2_ceiling_F2", "r2_ceiling_F3") if c in t]
    if not cols:
        return []
    paths = []
    for (ds, pr), g in t[~t["set"].isin(["val"])].groupby(["dataset", "protocol"]):
        units = g[["block", "set"]].drop_duplicates().sort_values(["block", "set"])
        labels_u = [f"{b}\n{SET_LABELS.get(s, s)}" for b, s in units.to_records(index=False)]
        x = np.arange(len(units))
        fig, ax = plt.subplots(figsize=(max(4.5, 0.9 * len(units) + 2), 3.2))
        grey = {"r2_ceiling_F1": "#c3c2b7", "r2_ceiling_F2": "#8f8e86", "r2_ceiling_F3": "#5b5a55"}
        mk = {"r2_ceiling_F1": "v", "r2_ceiling_F2": "s", "r2_ceiling_F3": "^"}
        for c in cols:
            vals = [g[(g["block"] == b) & (g["set"] == s)][c].iloc[0] for b, s in units.to_records(index=False)]
            ax.scatter(x, vals, marker=mk[c], color=grey[c], s=30, label=f"Ceiling, {c[-2:]}", zorder=2)
        for meth in ("gp_standard", "dcsr_s"):
            vals = []
            for b, s in units.to_records(index=False):
                r = g[(g["block"] == b) & (g["set"] == s) & (g["method"] == meth)]
                vals.append(r["r2_median"].iloc[0] if len(r) else np.nan)
            ax.scatter(x, vals, color=color(cfg, meth), marker=DCSR_MARKERS.get(meth, "o"), s=30,
                       label=f"{label(cfg, meth)}, median", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_u, fontsize=8)
        ax.set_ylabel("Test $R^2$")
        ax.legend(frameon=False, fontsize=8, ncol=2)
        paths += save(fig, out, f"fig14_r2_against_information_ceiling_{ds}_{pr}", cfg)
    return paths


def run(cfg, log, **_):
    import matplotlib.pyplot as plt
    apply_style(cfg)
    out = sdir(cfg, "11")
    paths = []
    jobs = []
    for ds in enabled_datasets(cfg):
        if clean_file(cfg, ds).exists():
            jobs += [(fig_rows_by_year, ds), (fig_co2_vs_engine, ds), (fig_prior_by_year, ds),
                     (fig_proxy_k, ds), (fig_proxy_vs_year, ds)]
    jobs += [(f, None) for f in (fig_rmse_boxes, fig_cd, fig_domain_validity, fig_rmse_vs_complexity,
                                 fig_response, fig_lambda, fig_convergence, fig_audit, fig_ceiling,
                                 fig_verdicts)]
    for f, ds in jobs:
        try:
            paths += f(cfg, out, ds, plt) if ds else f(cfg, out, plt)
        except FileNotFoundError as exc:
            log.warning("%s skipped: %s", f.__name__, exc)
        except Exception as exc:
            log.warning("%s failed: %s: %s", f.__name__, type(exc).__name__, exc)
    log.info("%d figure file(s) written to %s", len(paths), out)
    return paths
