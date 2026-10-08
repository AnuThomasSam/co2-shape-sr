"""Evaluation protocols (plan, Section 7). Splits are stored as row-id lists.

Each block has: fit rows, inner validation rows, and one or more named test
sets. No test row is ever in fit or validation; tests/test_splits.py checks this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def _grouped_val(df: pd.DataFrame, frac: float, seed: int) -> tuple[pd.Index, pd.Index]:
    gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    fit_i, val_i = next(gss.split(df, groups=df["make_model"]))
    return df.index[fit_i], df.index[val_i]


def _block(df, block_id, fit_idx, val_idx, tests: dict, meta: dict | None = None) -> dict:
    return {"block_id": block_id,
            "fit": df.loc[fit_idx, "row_id"].tolist(),
            "val": df.loc[val_idx, "row_id"].tolist(),
            "tests": {k: df.loc[v, "row_id"].tolist() for k, v in tests.items()},
            "meta": meta or {}}


def p1_rolling(df: pd.DataFrame, pc: dict) -> list[dict]:
    years = sorted(int(y) for y in df["model_year"].dropna().unique())
    w = int(pc["window_years"])
    if w < 2:
        raise ValueError("P1 window_years must be >= 2 (fit years + one validation year)")
    blocks = []
    for test_year in years:
        fit_years = list(range(test_year - w, test_year - 1))
        val_year = test_year - 1
        if fit_years[0] < years[0]:
            continue
        fit_idx = df.index[df["model_year"].isin(fit_years)]
        val_idx = df.index[df["model_year"] == val_year]
        test_idx = df.index[df["model_year"] == test_year]
        if len(fit_idx) < pc["min_fit_rows"] or len(val_idx) == 0 or len(test_idx) == 0:
            continue
        seen = set(df.loc[fit_idx.union(val_idx), "spec_key"])
        new_idx = test_idx[~df.loc[test_idx, "spec_key"].isin(seen)]
        blocks.append(_block(df, f"T{test_year}", fit_idx, val_idx,
                             {"all": test_idx, "new_config": new_idx},
                             {"fit_years": fit_years, "val_year": val_year, "test_year": test_year}))
    return blocks


def p2_extrapolation(df: pd.DataFrame, pc: dict, seed: int) -> list[dict]:
    lo = df["engine_size"].quantile(pc["lower_quantile"])
    hi = df["engine_size"].quantile(pc["upper_quantile"])
    blocks = []
    mid = df[(df["engine_size"] >= lo) & (df["engine_size"] <= hi)]
    fit_idx, val_idx = _grouped_val(mid, pc["validation_fraction"], seed)
    blocks.append(_block(df, "displacement", fit_idx, val_idx,
                         {"low_tail": df.index[df["engine_size"] < lo],
                          "high_tail": df.index[df["engine_size"] > hi]},
                         {"engine_low_cut": float(lo), "engine_high_cut": float(hi)}))
    c = pc.get("unseen_cylinders")
    if c is not None and (df["cylinders"] == c).sum() > 0:
        rest = df[df["cylinders"] != c]
        f2, v2 = _grouped_val(rest, pc["validation_fraction"], seed)
        blocks.append(_block(df, f"cyl{c}", f2, v2, {f"cyl{c}": df.index[df["cylinders"] == c]},
                             {"unseen_cylinders": c}))
    return blocks


def p3_proxy(df: pd.DataFrame, assign: pd.DataFrame, pc: dict, seed: int) -> list[dict]:
    m = df[["row_id", "make_model"]].merge(assign[["row_id", "proxy_group"]], on="row_id", how="left")
    m.index = df.index
    older = df[m["proxy_group"] == "older"]
    fit_idx, val_idx = _grouped_val(older, pc["validation_fraction"], seed)
    return [_block(df, "proxy", fit_idx, val_idx, {"newer": df.index[m["proxy_group"] == "newer"]})]


def p4_grouped_cv(df: pd.DataFrame, pc: dict, seed: int) -> list[dict]:
    blocks = []
    gkf = GroupKFold(n_splits=pc["n_folds"])
    # shuffle group order reproducibly so folds are not alphabetical
    rng = np.random.default_rng(seed)
    groups = df["make_model"].to_numpy()
    uniq = np.unique(groups)
    perm = dict(zip(uniq, rng.permutation(len(uniq))))
    gid = np.array([perm[g] for g in groups])
    for k, (tr, te) in enumerate(gkf.split(df, groups=gid)):
        train = df.iloc[tr]
        fit_idx, val_idx = _grouped_val(train, pc["validation_fraction"], seed + k)
        blocks.append(_block(df, f"fold{k + 1}", fit_idx, val_idx, {"all": df.index[te]}))
    return blocks


def check_no_overlap(blocks: list[dict]) -> None:
    for b in blocks:
        fit, val = set(b["fit"]), set(b["val"])
        assert not fit & val, f"{b['block_id']}: fit/val overlap"
        for name, ids in b["tests"].items():
            s = set(ids)
            assert not s & fit and not s & val, f"{b['block_id']}/{name}: test overlaps training"
