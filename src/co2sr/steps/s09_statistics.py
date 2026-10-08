"""Step 09: aggregate run metrics and run the statistical tests (plan, Section 9).

Reads only outputs/08_evaluation/*.csv and outputs/05_splits/information_ceiling.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from ..io import read_csv, write_csv
from ..stats import a12, a12_magnitude, friedman_nemenyi, holm, mann_whitney
from .common import sdir

CODE_DEPS = ['co2sr.stats']
NAME = "09_statistics"
UNIT = ["dataset", "protocol", "block", "set"]
EXCLUDE_FROM_BLOCKS = {"val", "new_config"}     # new_config is a subset of 'all'


def config_part(cfg):
    return {"statistics": cfg["statistics"]}


def inputs(cfg):
    d = sdir(cfg, "08")
    return [d / "run_metrics.csv", d / "run_summary.csv", sdir(cfg, "05") / "information_ceiling.csv"]


def _q(p):
    return lambda s: float(np.nanquantile(s, p)) if s.notna().any() else np.nan


def run(cfg, log, **_):
    sc = cfg["statistics"]
    metric = sc["metric"]
    d8 = sdir(cfg, "08")
    met = read_csv(d8 / "run_metrics.csv")
    summ = read_csv(d8 / "run_summary.csv")
    out_dir = sdir(cfg, "09")
    outs = []

    test = met[met["set"] != "val"].merge(
        summ[["dataset", "protocol", "block", "method", "seed", "domain_violation_rate", "domain_min_slope"]
             + [c for c in ("certified", "n_nodes", "n_nodes_simplified", "fit_time_s") if c in summ]],
        on=["dataset", "protocol", "block", "method", "seed"], how="left")

    # 1. per block summary -------------------------------------------------------------
    agg = {"n_runs": ("seed", "count"),
           "rmse_median": ("rmse", "median"), "rmse_q1": ("rmse", _q(0.25)), "rmse_q3": ("rmse", _q(0.75)),
           "mae_median": ("mae", "median"), "r2_median": ("r2", "median"),
           "points_violation_rate_median": ("points_violation_rate", "median"),
           "domain_violation_rate_median": ("domain_violation_rate", "median"),
           "domain_violation_rate_max": ("domain_violation_rate", "max"),
           "share_runs_domain_valid": ("domain_violation_rate", lambda s: float(np.mean(s == 0))),
           "domain_min_slope_median": ("domain_min_slope", "median")}
    if "certified" in test:
        agg["share_certified"] = ("certified", "mean")
    if "n_nodes" in test:
        agg["n_nodes_median"] = ("n_nodes", "median")
    if "n_nodes_simplified" in test:
        agg["n_nodes_simplified_median"] = ("n_nodes_simplified", "median")
    if "fit_time_s" in test:
        agg["fit_time_median_s"] = ("fit_time_s", "median")
    by_block = test.groupby(UNIT + ["method"]).agg(**agg).reset_index()
    ceil = read_csv(sdir(cfg, "05") / "information_ceiling.csv").rename(columns={"test_set": "set"})
    ceil = ceil.pivot_table(index=["dataset", "protocol", "block", "set"], columns="feature_set",
                            values="r2_ceiling").add_prefix("r2_ceiling_").reset_index()
    by_block = by_block.merge(ceil, on=UNIT, how="left")
    outs.append(write_csv(by_block, out_dir / "summary_by_block.csv"))

    # 2. per protocol summary (mean of block medians, and mean rank within block) ---------
    bb = by_block[~by_block["set"].isin(EXCLUDE_FROM_BLOCKS)].copy()
    bb["rank_in_block"] = bb.groupby(UNIT)["rmse_median"].rank(method="average")
    by_proto = bb.groupby(["dataset", "protocol", "method"]).agg(
        n_blocks=("block", "count"), rmse_mean_of_medians=("rmse_median", "mean"),
        mae_mean_of_medians=("mae_median", "mean"), r2_mean_of_medians=("r2_median", "mean"),
        mean_rank=("rank_in_block", "mean"),
        domain_violation_rate_median=("domain_violation_rate_median", "median"),
        share_runs_domain_valid=("share_runs_domain_valid", "mean")).reset_index()
    outs.append(write_csv(by_proto, out_dir / "summary_by_protocol.csv"))

    # 3. degradation relative to in-distribution (P4) ------------------------------------
    p4 = by_proto[by_proto["protocol"] == "P4"][["dataset", "method", "rmse_mean_of_medians"]] \
        .rename(columns={"rmse_mean_of_medians": "rmse_P4"})
    deg = by_proto[by_proto["protocol"] != "P4"].merge(p4, on=["dataset", "method"], how="inner")
    deg["degradation_ratio"] = deg["rmse_mean_of_medians"] / deg["rmse_P4"]
    outs.append(write_csv(deg[["dataset", "protocol", "method", "rmse_mean_of_medians", "rmse_P4",
                               "degradation_ratio"]], out_dir / "degradation_vs_P4.csv"))

    # 4. pairwise tests within each block -------------------------------------------------
    rows = []
    for unit, g in test[~test["set"].isin({"val"})].groupby(UNIT):
        for a, b in sc["comparisons"]:
            xa = g.loc[g["method"] == a, metric].dropna().to_numpy()
            xb = g.loc[g["method"] == b, metric].dropna().to_numpy()
            if len(xa) == 0 or len(xb) == 0:
                continue
            A = a12(xa, xb)
            rows.append({**dict(zip(UNIT, unit)), "method_a": a, "method_b": b, "n_a": len(xa), "n_b": len(xb),
                         "median_a": np.median(xa), "median_b": np.median(xb), "A12": A,
                         "A12_magnitude": a12_magnitude(A), "p_value": mann_whitney(xa, xb)})
    pw = pd.DataFrame(rows)
    if len(pw):
        pw["p_holm"] = pw.groupby(["dataset", "protocol"], group_keys=False)["p_value"].apply(holm)
        pw["significant"] = (pw["p_holm"] < sc["alpha"]).astype(int)
    outs.append(write_csv(pw, out_dir / "pairwise_tests.csv"))

    # 5. Friedman + Nemenyi across blocks -------------------------------------------------
    fr_rows, rank_rows = [], []
    for gname, protos in sc["friedman_groups"].items():
        for ds, gds in bb.groupby("dataset"):
            sub = gds[gds["protocol"].isin(protos) & gds["method"].isin(sc["main_methods"])]
            if sub.empty:
                continue
            tab = sub.pivot_table(index=UNIT, columns="method", values="rmse_median")
            tab = tab[[m for m in sc["main_methods"] if m in tab.columns]]
            res = friedman_nemenyi(tab, sc["alpha"])
            fr_rows.append({"group": gname, "dataset": ds, "protocols": "+".join(protos),
                            "n_blocks": res["n_blocks"], "k_methods": res["k_methods"],
                            "friedman_chi2": res["statistic"], "p_value": res["p_value"],
                            "critical_difference": res["cd"]})
            for m, r in res["avg_ranks"].items():
                rank_rows.append({"group": gname, "dataset": ds, "method": m, "average_rank": r,
                                  "critical_difference": res["cd"], "n_blocks": res["n_blocks"]})
    outs.append(write_csv(pd.DataFrame(fr_rows), out_dir / "friedman_summary.csv"))
    outs.append(write_csv(pd.DataFrame(rank_rows), out_dir / "friedman_average_ranks.csv"))

    # 6. constraint validity per method (H3) ---------------------------------------------
    sv = summ.groupby(["dataset", "protocol", "method"]).agg(
        n_runs=("seed", "count"),
        share_runs_domain_valid=("domain_violation_rate", lambda s: float(np.mean(s == 0))),
        domain_violation_rate_median=("domain_violation_rate", "median"),
        domain_violation_rate_max=("domain_violation_rate", "max"),
        most_negative_slope=("domain_min_slope", "min"),
        share_runs_using_engine_size=("uses_engine_size", "mean"),
        mean_slope_median=("domain_mean_slope", "median")).reset_index()
    if "certified" in summ:
        cert = summ.groupby(["dataset", "protocol", "method"])["certified"].mean().rename("share_certified")
        sv = sv.merge(cert.reset_index(), on=["dataset", "protocol", "method"], how="left")
    if "bb_verdict" in summ:
        gp = summ[summ["bb_verdict"].notna() & (summ["bb_verdict"] != "")]
        vs = pd.crosstab([gp["dataset"], gp["protocol"], gp["method"]], gp["bb_verdict"], normalize="index") \
            .reindex(columns=["certified", "refuted", "undecided"], fill_value=0.0).add_prefix("share_bb_").reset_index()
        sv = sv.merge(vs, on=["dataset", "protocol", "method"], how="left")
    outs.append(write_csv(sv, out_dir / "constraint_validity.csv"))

    # 6a. pessimism of plain interval arithmetic: runs that the fixed-partition check fails to certify
    #     but branch-and-bound certifies (false rejections), and the reverse (must be zero: soundness)
    if {"bb_verdict", "certified"} <= set(summ.columns):
        gp = summ[summ["bb_verdict"].isin(["certified", "refuted", "undecided"])].copy()
        gp["ia_certified"] = gp["certified"].astype(float)
        gp["bb_certified"] = (gp["bb_verdict"] == "certified").astype(float)
        cmp_ = gp.groupby(["dataset", "protocol", "method"]).agg(
            n_runs=("seed", "count"),
            share_ia_certified=("ia_certified", "mean"),
            share_bb_certified=("bb_certified", "mean"),
            share_ia_false_rejection=("bb_certified", lambda s: float(np.mean((s == 1) & (gp.loc[s.index, "ia_certified"] == 0)))),
            n_ia_certified_bb_refuted=("bb_verdict", lambda s: int(np.sum((s == "refuted") & (gp.loc[s.index, "ia_certified"] == 1)))),
            share_grid_false_acceptance=("bb_verdict", lambda s: float(np.mean((s == "refuted") & (gp.loc[s.index, "domain_violation_rate"] == 0)))),
            bb_boxes_median=("bb_boxes", "median")).reset_index()
        outs.append(write_csv(cmp_, out_dir / "interval_vs_branch_and_bound.csv"))
        if (cmp_["n_ia_certified_bb_refuted"] > 0).any():
            log.error("soundness violation: a model certified by interval arithmetic was refuted by branch-and-bound")

    # 6d. verification cost of DCSR-V during search
    vcols = [c for c in summ.columns if c.startswith("verify_")]
    if vcols:
        vv = summ[summ[vcols[0]].notna()]
        if len(vv):
            outs.append(write_csv(vv.groupby(["dataset", "protocol", "method"])[vcols + ["fit_time_s"]].median()
                                  .reset_index(), out_dir / "verification_cost.csv"))

    # 6b. stability (plan, Section 8): share of runs beating OLS in the same block and test set,
    #     and share of runs whose equation uses the most common variable set
    keys = ["dataset", "protocol", "block", "set"]
    ols = test[test["method"] == "ols"][keys + [metric]].rename(columns={metric: "ols_" + metric})
    st = test.merge(ols, on=keys, how="left")
    st["beats_ols"] = (st[metric] < st["ols_" + metric]).astype(float).where(st["ols_" + metric].notna())
    stab = st.groupby(["dataset", "protocol", "method"]).agg(
        share_runs_beating_ols=("beats_ols", "mean"),
        rmse_iqr_median_over_blocks=(metric, lambda s: np.nan)).reset_index()
    iqr = by_block.assign(iqr=by_block["rmse_q3"] - by_block["rmse_q1"]) \
        .groupby(["dataset", "protocol", "method"])["iqr"].median().rename("rmse_iqr_median_over_blocks")
    stab = stab.drop(columns="rmse_iqr_median_over_blocks").merge(iqr.reset_index(), on=["dataset", "protocol", "method"])
    if "variables_used" in summ:
        def modal_share(s):
            s = s.dropna()
            return float(s.value_counts(normalize=True).iloc[0]) if len(s) else np.nan
        vs = summ.groupby(["dataset", "protocol", "block", "method"])["variables_used"].agg(modal_share) \
            .groupby(["dataset", "protocol", "method"]).median().rename("share_runs_modal_variable_set")
        stab = stab.merge(vs.reset_index(), on=["dataset", "protocol", "method"], how="left")
    outs.append(write_csv(stab, out_dir / "stability.csv"))

    # 6c. engine-size effect: model slopes by fuel type (domain grid) against matched pairs (step 03)
    slope_cols = [c for c in summ.columns if c.startswith("domain_mean_slope_")]
    if slope_cols:
        me = summ.groupby(["dataset", "protocol", "method"])[slope_cols].median().reset_index()
        outs.append(write_csv(me, out_dir / "engine_size_effect_by_fuel.csv"))

    # 7. lambda ablation (A1): validation error against domain violation ------------------
    lam = met[(met["set"] == "val") & met["method"].isin(sc["lambda_ablation"])] \
        .merge(summ[["dataset", "protocol", "block", "method", "seed", "domain_violation_rate"]],
               on=["dataset", "protocol", "block", "method", "seed"])
    if len(lam):
        la = lam.groupby(["dataset", "protocol", "method"]).agg(
            val_rmse_median=("rmse", "median"), domain_violation_rate_median=("domain_violation_rate", "median"),
            share_runs_domain_valid=("domain_violation_rate", lambda s: float(np.mean(s == 0)))).reset_index()
        outs.append(write_csv(la, out_dir / "lambda_ablation.csv"))

    # 8. proxy ranking vs real temporal ranking (RQ1) --------------------------------------
    kt = []
    for ds, g in by_proto.groupby("dataset"):
        r1 = g[g["protocol"] == "P1"].set_index("method")["mean_rank"]
        r3 = g[g["protocol"] == "P3"].set_index("method")["mean_rank"]
        common = [m for m in sc["main_methods"] if m in r1.index and m in r3.index]
        if len(common) >= 3:
            tau, p = sps.kendalltau(r1[common], r3[common])
            kt.append({"dataset": ds, "n_methods": len(common), "kendall_tau": tau, "p_value": p,
                       "methods": ";".join(common)})
    outs.append(write_csv(pd.DataFrame(kt, columns=["dataset", "n_methods", "kendall_tau", "p_value", "methods"]),
                          out_dir / "proxy_vs_temporal_ranking.csv"))
    log.info("statistics written to %s", out_dir)
    return outs
