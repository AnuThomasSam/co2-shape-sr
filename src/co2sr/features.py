"""Feature sets and design matrices.

All methods in a comparison receive the same information. Encodings differ:
symbolic regression can take fuel type as a group-wise linear scaling, the
other methods take fuel indicators. Engine size is always column 0, so the
derivative with respect to column 0 is the derivative with respect to engine
size: no engineered feature contains engine size.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRANSMISSION_VARS = ["gears_f", "gears_missing", "tt_AM", "tt_AS", "tt_AV", "tt_M"]


def fuel_levels(cfg: dict) -> list[str]:
    ref = cfg["features"]["fuel_reference"]
    fuels = cfg["cleaning"]["allowed_fuels"]
    return [ref] + [f for f in fuels if f != ref]


def body_levels(cfg: dict) -> list[str]:
    return list(cfg["features"]["body_groups"])


def variable_names(cfg: dict, feature_set: str, fuel_mode: str) -> list[str]:
    blocks = cfg["features"]["sets"][feature_set]
    names: list[str] = []
    for b in blocks:
        if b == "engine_size":
            names.append("engine_size")
        elif b == "cylinders":
            names.append("cylinders")
        elif b == "transmission":
            names += TRANSMISSION_VARS
        elif b == "body_group":
            names += [f"bg_{g}" for g in body_levels(cfg)[1:]]
        elif b == "fuel":
            if fuel_mode == "indicators":
                names += [f"fuel_{f}" for f in fuel_levels(cfg)[1:]]
        else:
            raise ValueError(f"Unknown feature block '{b}'")
    if names[0] != "engine_size":
        raise ValueError("engine_size must be the first variable of every feature set")
    return names


def uses_fuel(cfg: dict, feature_set: str) -> bool:
    return "fuel" in cfg["features"]["sets"][feature_set]


def design_matrix(df: pd.DataFrame, names: list[str]) -> np.ndarray:
    cols = []
    for n in names:
        if n == "engine_size":
            cols.append(df["engine_size"].to_numpy(float))
        elif n == "cylinders":
            cols.append(df["cylinders"].to_numpy(float))
        elif n == "gears_f":
            cols.append(df["gears"].fillna(0.0).to_numpy(float))
        elif n == "gears_missing":
            cols.append(df["gears_missing"].to_numpy(float))
        elif n.startswith("tt_"):
            cols.append((df["transmission_type"] == n[3:]).to_numpy(float))
        elif n.startswith("bg_"):
            cols.append((df["body_group"] == n[3:]).to_numpy(float))
        elif n.startswith("fuel_"):
            cols.append((df["fuel_type"] == n[5:]).to_numpy(float))
        else:
            raise ValueError(f"Unknown variable '{n}'")
    return np.column_stack(cols) if cols else np.zeros((len(df), 0))


def fuel_group_index(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    levels = fuel_levels(cfg)
    lookup = {f: i for i, f in enumerate(levels)}
    return df["fuel_type"].map(lookup).fillna(-1).astype(int).to_numpy()


def ml_variable_names(cfg: dict, feature_set: str) -> list[str]:
    """Variables for non-SR methods: fuel always enters as indicators."""
    return variable_names(cfg, feature_set, fuel_mode="indicators")
