"""Sensitivity of the RQ1 proxy result to the number of clusters K.

Silhouette keeps rising with K on the NRCan data (the clustering inputs take only about 171 distinct
values), so the silhouette maximum sits at whatever upper limit k_max is given. This script fixes K
at several values and reports the proxy-year association for each, so the thesis can show whether the
RQ1 conclusion depends on K. Reads outputs/02_clean only; writes outputs/extra/proxy_sensitivity.csv.

    python tools/proxy_sensitivity.py [--config configs/default.yaml]
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from co2sr.config import load_config, outputs_dir  # noqa: E402
from co2sr.proxy import technology_split  # noqa: E402
from co2sr.steps.common import load_clean  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--k", default="2,4,6,8,10,15,20,25,30")
    a = ap.parse_args()
    cfg = load_config(a.config)
    df = load_clean(cfg, "nrcan").set_index("row_id")
    rows = []
    for K in [int(k) for k in a.k.split(",")]:
        pc = copy.deepcopy(cfg["proxy"])
        pc["k_min"] = pc["k_max"] = K
        ksel, prof, d, info = technology_split(df.reset_index(), pc, cfg["project"]["random_seed"])
        yr = df.loc[d["row_id"], "model_year"].to_numpy()
        rho, p = stats.spearmanr(d["technology_score"], yr)
        late = (yr > np.median(yr)).astype(int)
        newer = (d["proxy_group"] == "newer").astype(int).to_numpy()
        rows.append({"K": K, "silhouette": float(ksel["silhouette"].iloc[0]), "spearman_rho_score_year": rho,
                     "p_value": p, "auc_newer_label_late_year": roc_auc_score(late, newer),
                     "rows_newer": info["rows_newer"]})
    out = outputs_dir(cfg) / "extra"
    out.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(rows)
    t.to_csv(out / "proxy_sensitivity.csv", index=False)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
