"""K-Means technology proxy (RQ1), ported from the original 03 and 04 scripts.

Changes relative to the original code:
  - the K range is configurable and wider; a boundary choice is flagged;
  - the number of distinct points in the clustering space is reported;
  - adversarial validation is run with and without the variables that define
    the clusters, and never uses the model name;
  - when model years exist, the proxy is validated against them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def clustering_matrix(df: pd.DataFrame, pcfg: dict) -> pd.DataFrame:
    X = df[pcfg["features"]].copy()
    if "gears" in X and pcfg.get("cvt_gears") == "median":
        X["gears"] = X["gears"].fillna(X["gears"].median())
    return X


def select_k(Z: np.ndarray, pcfg: dict, seed: int) -> tuple[pd.DataFrame, int]:
    rows = []
    for k in range(pcfg["k_min"], pcfg["k_max"] + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=pcfg["n_init"]).fit(Z)
        sample = min(len(Z), 5000)
        rows.append({"k": k, "silhouette": silhouette_score(Z, km.labels_, sample_size=sample, random_state=seed),
                     "inertia": km.inertia_})
    tab = pd.DataFrame(rows)
    best = int(tab.loc[tab["silhouette"].idxmax(), "k"])
    tab["selected"] = (tab["k"] == best).astype(int)
    tab["selected_at_boundary"] = int(best in (pcfg["k_min"], pcfg["k_max"]))
    return tab, best


def technology_split(df: pd.DataFrame, pcfg: dict, seed: int):
    Xraw = clustering_matrix(df, pcfg)
    scaler = StandardScaler().fit(Xraw)
    Z = scaler.transform(Xraw)
    ksel, k = select_k(Z, pcfg, seed)
    km = KMeans(n_clusters=k, random_state=seed, n_init=pcfg["n_init"]).fit(Z)
    d = df[["row_id"]].copy()
    d["cluster"] = km.labels_
    prof = Xraw.assign(cluster=km.labels_, co2=df["co2"].values).groupby("cluster").agg(
        rows=("co2", "size"), engine_size=("engine_size", "mean"), cylinders=("cylinders", "mean"),
        gears=("gears", "mean"), co2=("co2", "mean"))
    z = (prof[["engine_size", "cylinders", "gears"]] - prof[["engine_size", "cylinders", "gears"]].mean()) \
        / prof[["engine_size", "cylinders", "gears"]].std(ddof=0)
    prof["technology_score"] = z["engine_size"] + z["cylinders"] - z["gears"]     # higher = "older"
    prof = prof.sort_values("technology_score", ascending=False)
    if pcfg.get("split_rule", "clusters") == "clusters":
        older = prof.index[: len(prof) // 2].tolist()
    else:                                      # half of the rows, cutting at cluster boundaries
        cum = prof["rows"].cumsum() / prof["rows"].sum()
        older = prof.index[cum <= 0.5].tolist() or prof.index[:1].tolist()
    prof["group"] = np.where(prof.index.isin(older), "older", "newer")
    d["technology_score"] = d["cluster"].map(prof["technology_score"])
    d["proxy_group"] = np.where(d["cluster"].isin(older), "older", "newer")
    info = {"k": k, "distinct_points": int(len(np.unique(Xraw.to_numpy(), axis=0))),
            "rows_older": int((d["proxy_group"] == "older").sum()),
            "rows_newer": int((d["proxy_group"] == "newer").sum())}
    return ksel, prof.reset_index(), d, info


def adversarial_auc(df: pd.DataFrame, labels: np.ndarray, numeric: list[str], categorical: list[str],
                    folds: int, seed: int) -> float:
    X = df[numeric + categorical].copy()
    if "gears" in X:
        X["gears"] = X["gears"].fillna(-1)
    transformers = []
    if numeric:
        transformers.append(("num", StandardScaler(), numeric))
    if categorical:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
    pipe = Pipeline([("prep", ColumnTransformer(transformers)),
                     ("clf", LogisticRegression(max_iter=2000))])
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    p = cross_val_predict(pipe, X, labels, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(labels, p))


def validity(df: pd.DataFrame, assign: pd.DataFrame, pcfg: dict, seed: int) -> pd.DataFrame:
    m = df.merge(assign, on="row_id")
    lab = (m["proxy_group"] == "newer").astype(int).to_numpy()
    rows = []
    with_def = adversarial_auc(m, lab, ["engine_size", "cylinders", "gears"],
                               ["transmission_type", "body_group", "fuel_type"], pcfg["adversarial_folds"], seed)
    without_def = adversarial_auc(m, lab, [], ["transmission_type", "body_group", "fuel_type"],
                                  pcfg["adversarial_folds"], seed)
    rows.append({"measure": "adversarial AUC, with defining variables", "value": with_def})
    rows.append({"measure": "adversarial AUC, without defining variables", "value": without_def})
    if m["model_year"].notna().any():
        yr = m["model_year"].astype(float)
        rho, p = stats.spearmanr(m["technology_score"], yr)
        # positive = higher "older" score goes with later years (the proxy is wrong)
        rows.append({"measure": "Spearman rho, technology score vs model year", "value": rho, "p_value": p})
        newer_year = (yr > yr.median()).astype(int)
        rows.append({"measure": "AUC, proxy 'newer' label predicting model year > median",
                     "value": roc_auc_score(newer_year, lab) if newer_year.nunique() > 1 else np.nan})
        rows.append({"measure": "AUC, -technology score predicting model year > median",
                     "value": roc_auc_score(newer_year, -m["technology_score"]) if newer_year.nunique() > 1 else np.nan})
        rows.append({"measure": "mean model year, older group", "value": yr[lab == 0].mean()})
        rows.append({"measure": "mean model year, newer group", "value": yr[lab == 1].mean()})
    for grp, sub in m.groupby("proxy_group"):
        rows.append({"measure": f"engine size range, {grp}", "value": f"{sub.engine_size.min()}-{sub.engine_size.max()}"})
        rows.append({"measure": f"cylinder range, {grp}", "value": f"{sub.cylinders.min()}-{sub.cylinders.max()}"})
        rows.append({"measure": f"mean gears, {grp}", "value": sub["gears"].mean()})
        rows.append({"measure": f"mean CO2, {grp}", "value": sub["co2"].mean()})
    return pd.DataFrame(rows)
