"""Statistical tests (plan, Section 9)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def a12(x: np.ndarray, y: np.ndarray) -> float:
    """Vargha-Delaney A12 for errors: probability that a value from x is LOWER than one from y
    (ties count one half). A12 > 0.5 means x tends to have smaller error than y."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = (x[:, None] < y[None, :]).sum()
    eq = (x[:, None] == y[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(x) * len(y)))


def a12_magnitude(a: float) -> str:
    d = abs(a - 0.5) * 2          # distance from 0.5, mapped to [0, 1]
    if np.isnan(a):
        return "n/a"
    # thresholds of Vargha and Delaney (2000): 0.56, 0.64, 0.71
    m = max(a, 1 - a)
    return "negligible" if m < 0.56 else "small" if m < 0.64 else "medium" if m < 0.71 else "large"


def mann_whitney(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or len(y) < 2 or (np.all(x == x[0]) and np.all(y == y[0]) and x[0] == y[0]):
        return np.nan
    return float(stats.mannwhitneyu(x, y, alternative="two-sided").pvalue)


def holm(pvalues: pd.Series) -> pd.Series:
    p = pvalues.to_numpy(float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.where(ok)[0]
    order = idx[np.argsort(p[idx])]
    m = len(order)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * p[i])
        running = max(running, adj)
        out[i] = running
    return pd.Series(out, index=pvalues.index)


def friedman_nemenyi(table: pd.DataFrame, alpha: float = 0.05) -> dict:
    """table: rows = blocks, columns = methods, values = error (lower is better)."""
    t = table.dropna(axis=0, how="any")
    n, k = t.shape
    res = {"n_blocks": n, "k_methods": k}
    if n < 2 or k < 2:
        res.update({"statistic": np.nan, "p_value": np.nan, "cd": np.nan,
                    "avg_ranks": pd.Series(dtype=float)})
        return res
    ranks = t.rank(axis=1, method="average")
    avg = ranks.mean(axis=0).sort_values()
    stat, p = stats.friedmanchisquare(*[t[c].to_numpy() for c in t.columns]) if k >= 3 else (np.nan, np.nan)
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2.0)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n))
    res.update({"statistic": float(stat), "p_value": float(p), "cd": float(cd), "q_alpha": float(q),
                "avg_ranks": avg})
    return res


def kruskal_dunn(groups: dict[str, np.ndarray]) -> tuple[dict, pd.DataFrame]:
    """Kruskal-Wallis test and Dunn's pairwise test with tie correction and Holm adjustment."""
    names = list(groups)
    data = [np.asarray(groups[n], float) for n in names]
    h, p = stats.kruskal(*data)
    allv = np.concatenate(data)
    ranks = stats.rankdata(allv)
    N = len(allv)
    _, counts = np.unique(allv, return_counts=True)
    tie = 1 - np.sum(counts ** 3 - counts) / (N ** 3 - N)
    mean_rank, start = {}, 0
    for n, d in zip(names, data):
        mean_rank[n] = ranks[start:start + len(d)].mean()
        start += len(d)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            se = np.sqrt(tie * N * (N + 1) / 12.0 * (1 / len(groups[a]) + 1 / len(groups[b])))
            z = (mean_rank[a] - mean_rank[b]) / se
            rows.append({"group_a": a, "group_b": b, "z": z, "p_value": 2 * stats.norm.sf(abs(z))})
    pw = pd.DataFrame(rows)
    if len(pw):
        pw["p_holm"] = holm(pw["p_value"])
    return {"H": float(h), "p_value": float(p), "n_groups": len(names), "N": N}, pw


def bootstrap_median_ci(x: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    meds = np.median(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    return float(np.quantile(meds, alpha / 2)), float(np.quantile(meds, 1 - alpha / 2))
