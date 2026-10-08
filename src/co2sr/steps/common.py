"""Paths and loaders shared by the steps. Each step reads only files written by earlier steps."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import outputs_dir, results_dir
from ..io import read_csv, read_json

STEP_DIRS = {
    "01": "01_ingest", "02": "02_clean", "03": "03_prior", "04": "04_proxy", "05": "05_splits",
    "06": "06_domain", "07": "07_models", "08": "08_evaluation", "09": "09_statistics",
    "10": "10_audit", "11": "figures", "12": "tables", "13": "13_benchmark", "14": "benchmark",
}


def sdir(cfg: dict, step: str) -> Path:
    if step in ("11", "12", "14"):
        d = results_dir(cfg) / STEP_DIRS[step]
    else:
        d = outputs_dir(cfg) / STEP_DIRS[step]
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------- file names
def harmonised_file(cfg, ds):      return sdir(cfg, "01") / f"{ds}_harmonised.csv"
def clean_file(cfg, ds):           return sdir(cfg, "02") / f"{ds}_clean.csv"
def proxy_assign_file(cfg, ds):    return sdir(cfg, "04") / f"{ds}_proxy_assignments.csv"
def split_file(cfg, ds, p):        return sdir(cfg, "05") / f"{ds}_{p}_splits.json"
def domain_dir(cfg, ds, p, b):     return sdir(cfg, "06") / ds / p / b
def run_dir(cfg, ds, p, b, m, s):  return sdir(cfg, "07") / ds / p / b / m / f"seed{s:02d}"


def load_clean(cfg: dict, ds: str) -> pd.DataFrame:
    df = read_csv(clean_file(cfg, ds), dtype={"model": str, "make": str})
    df["model_year"] = df["model_year"].astype("Int64")
    return df


def load_splits(cfg: dict, ds: str, protocol: str) -> dict:
    p = split_file(cfg, ds, protocol)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run step 05 (protocol {protocol} may not apply to {ds}).")
    return read_json(p)


def block_frames(df: pd.DataFrame, block: dict) -> dict[str, pd.DataFrame]:
    idx = df.set_index("row_id")
    out = {"fit": idx.loc[block["fit"]].reset_index(), "val": idx.loc[block["val"]].reset_index()}
    out["full"] = pd.concat([out["fit"], out["val"]], ignore_index=True)
    out["tests"] = {k: idx.loc[v].reset_index() for k, v in block["tests"].items()}
    return out


def load_domain(cfg: dict, ds: str, p: str, b: str, which=("sample", "boxes_train")) -> dict:
    d = domain_dir(cfg, ds, p, b)
    out = {}
    if "sample" in which:
        out["sample"] = read_csv(d / "domain_sample.csv")
    if "grid" in which:
        out["grid"] = read_csv(d / "domain_grid.csv")
    if "boxes_train" in which:
        out["boxes_lo"] = read_csv(d / "boxes_train_lo.csv")
        out["boxes_hi"] = read_csv(d / "boxes_train_hi.csv")
    if "boxes_verify" in which:
        out["boxes_verify_lo"] = read_csv(d / "boxes_verify_lo.csv")
        out["boxes_verify_hi"] = read_csv(d / "boxes_verify_hi.csv")
    if "boxes_eval" in which:
        out["boxes_eval_lo"] = read_csv(d / "boxes_eval_lo.csv")
        out["boxes_eval_hi"] = read_csv(d / "boxes_eval_hi.csv")
    return out


def protocols_for(cfg: dict, ds: str) -> list[str]:
    ps = ["P2", "P3", "P4"]
    if cfg["datasets"][ds].get("has_model_year"):
        ps = ["P1"] + ps
    return ps
