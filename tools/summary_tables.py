"""Summary tables for the thesis, read from the existing pipeline outputs (no training).

Run from the repository root, after steps 08 and 09:
    python tools/summary_tables.py

It writes four tables, each as CSV (outputs/extra/) and LaTeX (results/tables/):

  T1 results_by_block     median test RMSE (g/km) of the main methods in every P1 test year
                          and every P4 fold (one row per test set, one column per method)
  T2 validity             for each GP method and protocol: share of final equations proved valid
                          by branch and bound, and share of valid equations that plain interval
                          checking fails to prove (interval false rejection)
  T3 false_rejection_by_size
                          interval false rejection against equation size, over every valid final
                          GP equation on the NRCan data (all methods, protocols and seeds)
  T4 pooled_comparison    DCSR-V against each other method over all NRCan test sets of
                          P1, P2, P3 and P4: number of test sets where DCSR-V has the lower median
                          RMSE, mean relative difference, and the Wilcoxon signed-rank p-value

Every number comes from outputs/08_evaluation/run_summary.csv and
outputs/09_statistics/summary_by_block.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = ROOT / "outputs" / "extra"
OUT_TEX = ROOT / "results" / "tables"

MAIN = ["ols", "loglinear", "gam_mono", "xgb", "xgb_mono", "gp_standard", "dcsr_s", "dcsr_f",
        "dcsr_dyn", "dcsr_ia", "dcsr_v"]
GP_METHODS = ["gp_standard", "dcsr_t", "dcsr_s", "dcsr_f", "dcsr_dyn", "dcsr_ia", "dcsr_v"]
LABEL = {"ols": "OLS", "loglinear": "Log-linear OLS", "gam_mono": "Monotone GAM", "xgb": "XGBoost",
         "xgb_mono": "Monotone XGBoost", "gp_standard": "Standard GP", "dcsr_t": "DCSR-T",
         "dcsr_s": "DCSR-S", "dcsr_f": "DCSR-F", "dcsr_dyn": "Dynamic penalty", "dcsr_ia": "DCSR-IA",
         "dcsr_v": "DCSR-V", "pysr": "PySR"}


def _cell(v, float_fmt):
    if isinstance(v, (float, np.floating)):
        return "--" if np.isnan(v) else float_fmt % v
    s = str(v)
    for a, b in [("\\", "\\textbackslash{}"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#")]:
        s = s.replace(a, b)
    return s


def to_latex(df: pd.DataFrame, float_fmt: str) -> str:
    """Plain tabular with \\hline rules (no extra packages needed)."""
    align = "".join("l" if df[c].dtype == object else "r" for c in df.columns)
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\hline",
             " & ".join(_cell(c, float_fmt) for c in df.columns) + " \\\\", "\\hline"]
    for _, row in df.iterrows():
        lines.append(" & ".join(_cell(v, float_fmt) for v in row) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", ""]
    return "\n".join(lines)


def save(df: pd.DataFrame, name: str, caption: str, index=False, float_fmt="%.2f"):
    OUT_CSV.mkdir(parents=True, exist_ok=True)
    OUT_TEX.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV / f"{name}.csv", index=index)
    (OUT_TEX / f"{name}.tex").write_text(f"% {caption}\n{to_latex(df, float_fmt)}", encoding="utf-8")
    print(f"\n{caption}\n{df.to_string(index=index)}")


def main():
    blocks = pd.read_csv(ROOT / "outputs/09_statistics/summary_by_block.csv")
    runs = pd.read_csv(ROOT / "outputs/08_evaluation/run_summary.csv")
    nb = blocks[(blocks.dataset == "nrcan") & (~blocks.set.isin(["val", "new_config"]))]

    # T1: results by block (P1 years and P4 folds)
    t1 = nb[nb.protocol.isin(["P1", "P4"])].pivot_table(index=["protocol", "block"], columns="method",
                                                        values="rmse_median")
    cols = [m for m in MAIN if m in t1.columns]
    t1 = t1[cols]
    mean_rows = t1.groupby(level=0).mean()
    mean_rows.index = pd.MultiIndex.from_tuples([(p, "Mean") for p in mean_rows.index],
                                                names=["protocol", "block"])
    t1 = pd.concat([t1, mean_rows]).sort_index(level=0, sort_remaining=False)
    t1 = t1.rename(columns=LABEL).reset_index().rename(columns={"protocol": "Protocol", "block": "Test set"})
    t1.columns.name = None
    save(t1, "T1_results_by_block", "T1. Median test RMSE (g/km) by test set, NRCan, P1 and P4")

    # T2: validity of final GP equations
    r = runs[(runs.dataset == "nrcan") & runs.method.isin(GP_METHODS) & runs.bb_verdict.notna()].copy()
    r["valid"] = r.bb_verdict == "certified"
    r["ia_proved"] = r.certified == 1
    rows = []
    for (p, m), g in r.groupby(["protocol", "method"]):
        v = g[g.valid]
        rows.append({"Protocol": p, "Method": LABEL[m], "Runs": len(g),
                     "Proved valid (share)": g.valid.mean(),
                     "Interval false rejection (share of valid)": (~v.ia_proved).mean() if len(v) else np.nan,
                     "Median nodes": g.n_nodes.median()})
    t2 = pd.DataFrame(rows)
    order = {LABEL[m]: i for i, m in enumerate(GP_METHODS)}
    t2 = t2.sort_values(["Protocol", "Method"], key=lambda s: s.map(order) if s.name == "Method" else s)
    save(t2, "T2_validity", "T2. Final GP equations proved valid by branch and bound, NRCan", float_fmt="%.3f")

    # T3: interval false rejection against equation size. Every GP run is used here (main methods and
    # ablations), because the quantity measured is a property of the checking method, not of the search.
    # Methods whose search already uses interval checks (DCSR-IA and every DCSR-V variant) are left out:
    # their equations were selected to pass those checks, which would bias the share downwards.
    a = runs[(runs.dataset == "nrcan") & runs.bb_verdict.notna()
             & ~runs.method.str.startswith(("dcsr_ia", "dcsr_v"))].copy()
    v = a[a.bb_verdict == "certified"].copy()
    v["ia_proved"] = v.certified == 1
    v["valid"] = True
    v["Nodes"] = pd.cut(v.n_nodes, [0, 10, 15, 20, 25, 30, 41],
                        labels=["1-10", "11-15", "16-20", "21-25", "26-30", "31-41"])
    t3 = v.groupby("Nodes", observed=True).agg(**{"Valid equations": ("valid", "size"),
                                                  "Interval false rejection": ("ia_proved",
                                                                               lambda s: 1 - s.mean())})
    t3 = t3.reset_index()
    save(t3, "T3_false_rejection_by_size",
         "T3. Share of valid final equations that plain interval checking fails to prove, by size",
         float_fmt="%.3f")

    # T4: DCSR-V against each other method over all NRCan test sets
    p = nb.pivot_table(index=["protocol", "block", "set"], columns="method", values="rmse_median")
    rows = []
    for m in MAIN:
        if m == "dcsr_v" or m not in p.columns:
            continue
        x = p[["dcsr_v", m]].dropna()
        d = x["dcsr_v"] - x[m]
        pv = wilcoxon(d).pvalue if (d != 0).any() else np.nan
        rows.append({"Compared with": LABEL[m], "Test sets": len(x), "DCSR-V lower": int((d < 0).sum()),
                     "Mean relative difference (%)": 100 * (d / x[m]).mean(), "Wilcoxon p": pv})
    t4 = pd.DataFrame(rows)
    save(t4, "T4_pooled_comparison",
         "T4. DCSR-V against other methods over all NRCan test sets (negative difference: DCSR-V lower)",
         float_fmt="%.3f")


if __name__ == "__main__":
    main()
