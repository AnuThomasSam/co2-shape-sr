"""Cleaning rules applied to harmonised data (step 02).

Every rule that removes rows writes a line to the cleaning log, so the thesis
can report exactly what was removed and why.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


def body_group(vehicle_class: str) -> str:
    """Map NRCan vehicle classes (both spelling styles) to five body groups.

    The mapping is fixed a priori and does not look at CO2.
    """
    c = str(vehicle_class).lower()
    if "suv" in c:
        return "suv"
    if "pickup" in c:
        return "pickup"
    if "special purpose" in c:
        return "special"
    if "van" in c:            # minivan, van - passenger, van: cargo
        return "van"
    return "car"              # minicompact ... full-size, two-seater, station wagons


def parse_transmission(code: str, cvt_without_gears: list[str]) -> tuple[str | None, float]:
    m = re.fullmatch(r"([A-Z]+?)(\d*)", str(code).strip().upper())
    if not m:
        return None, np.nan
    ttype, digits = m.group(1), m.group(2)
    if str(code).strip().upper() in cvt_without_gears or digits == "":
        return ttype, np.nan
    return ttype, float(digits)


def clean(df: pd.DataFrame, cfg: dict, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ccfg, dcfg, dscfg = cfg["cleaning"], cfg["domain"], cfg["datasets"][dataset]
    log = []

    def record(rule: str, mask: pd.Series):
        log.append({"dataset": dataset, "rule": rule, "rows_removed": int(mask.sum()),
                    "rows_before": int(len(mask))})

    n0 = len(df)
    log.append({"dataset": dataset, "rule": "rows after ingest", "rows_removed": 0, "rows_before": n0})

    bad = df[["engine_size", "cylinders", "co2"]].isna().any(axis=1) | (df["co2"] <= 0)
    record("missing or non-positive engine size, cylinders or CO2", bad)
    df = df[~bad].copy()

    if dscfg.get("has_model_year"):
        ymin = dscfg.get("year_min")
        ymax = dscfg.get("year_max")
        out = pd.Series(False, index=df.index)
        if ymin is not None:
            out |= df["model_year"] < ymin
        if ymax is not None:
            out |= df["model_year"] > ymax
        record(f"model year outside [{ymin}, {ymax}]", out)
        df = df[~out].copy()

    parsed = df["transmission"].apply(lambda c: parse_transmission(c, ccfg["cvt_codes_without_gears"]))
    df["transmission_type"] = [p[0] for p in parsed]
    df["gears"] = [p[1] for p in parsed]
    bad_t = ~df["transmission_type"].isin(ccfg["transmission_types"])
    record("unrecognised transmission code", bad_t)
    df = df[~bad_t].copy()
    df["gears_missing"] = df["gears"].isna().astype(int)

    fuel_counts = df["fuel_type"].value_counts()
    rare = df["fuel_type"].map(fuel_counts) < ccfg["min_rows_per_fuel"]
    not_allowed = ~df["fuel_type"].isin(ccfg["allowed_fuels"])
    record(f"fuel type with < {ccfg['min_rows_per_fuel']} rows or not in {ccfg['allowed_fuels']}",
           rare | not_allowed)
    df = df[~(rare | not_allowed)].copy()

    df["body_group"] = df["vehicle_class"].apply(body_group)

    key_cols = ["model_year", "make", "model", "vehicle_class", "engine_size", "cylinders",
                "transmission", "fuel_type", "co2"]
    dup = df.duplicated(subset=key_cols, keep="first")
    record("exact duplicate within model year", dup)
    df = df[~dup].copy()

    ratio = df["engine_size"] / df["cylinders"]
    in_domain = (
        df["engine_size"].between(dcfg["engine_size_min"], dcfg["engine_size_max"])
        & df["cylinders"].isin(dcfg["cylinders"])
        & ratio.between(dcfg["litres_per_cylinder_min"], dcfg["litres_per_cylinder_max"])
    )
    df["in_domain"] = in_domain.astype(int)
    if ccfg.get("drop_outside_domain"):
        record("outside the declared domain", ~in_domain)
        df = df[in_domain].copy()
    else:
        log.append({"dataset": dataset, "rule": "outside the declared domain (flagged, kept)",
                    "rows_removed": 0, "rows_before": int(len(df)),
                    "rows_flagged": int((~in_domain).sum())})

    # Keys used to recognise the same vehicle across years. NRCan changed the spelling of
    # vehicle classes (e.g. "SUV - SMALL" vs "SUV: Small"), so classes are normalised here.
    model_u = df["model"].str.upper().str.replace(r"\s+", " ", regex=True)
    class_n = df["vehicle_class"].str.lower().str.replace(r"\s*[-:]\s*", ": ", regex=True)
    df["make_model"] = df["make"] + "|" + model_u
    df["spec_key"] = (df["make"] + "|" + model_u + "|" + class_n + "|"
                      + df["engine_size"].astype(str) + "|" + df["cylinders"].astype(int).astype(str)
                      + "|" + df["transmission"] + "|" + df["fuel_type"])
    df["cylinders"] = df["cylinders"].astype(int)
    sort_cols = ["model_year", "make", "model", "engine_size", "transmission", "fuel_type"]
    df = df.sort_values([c for c in sort_cols if df[c].notna().any()], kind="mergesort")
    df = df.reset_index(drop=True)
    df.insert(0, "row_id", [f"{dataset}-{i:06d}" for i in range(len(df))])
    log.append({"dataset": dataset, "rule": "rows after cleaning", "rows_removed": 0,
                "rows_before": int(len(df))})
    cols = ["row_id", "dataset", "model_year", "make", "model", "make_model", "vehicle_class",
            "body_group", "engine_size", "cylinders", "transmission", "transmission_type", "gears",
            "gears_missing", "fuel_type", "co2", "in_domain", "spec_key", "source_file"]
    return df[cols], pd.DataFrame(log)


def data_card(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Descriptive tables for the data chapter."""
    has_year = df["model_year"].notna().any()
    by = "model_year" if has_year else None
    cards = {}
    if has_year:
        cards["rows_by_year_fuel"] = (df.pivot_table(index="model_year", columns="fuel_type",
                                                     values="row_id", aggfunc="count", fill_value=0)
                                      .reset_index())
        cards["rows_by_year_body"] = (df.pivot_table(index="model_year", columns="body_group",
                                                     values="row_id", aggfunc="count", fill_value=0)
                                      .reset_index())
        g = df.groupby("model_year")
    else:
        g = df.assign(model_year="all").groupby("model_year")
    rng = g.agg(rows=("row_id", "count"),
                engine_size_min=("engine_size", "min"), engine_size_mean=("engine_size", "mean"),
                engine_size_max=("engine_size", "max"),
                cylinders_min=("cylinders", "min"), cylinders_max=("cylinders", "max"),
                gears_mean=("gears", "mean"), cvt_share=("gears_missing", "mean"),
                co2_min=("co2", "min"), co2_mean=("co2", "mean"), co2_max=("co2", "max"),
                outside_domain=("in_domain", lambda s: int((s == 0).sum())))
    cards["feature_ranges"] = rng.round(3).reset_index()
    cards["category_levels"] = pd.DataFrame({
        "variable": ["make", "model", "make_model", "vehicle_class", "transmission", "fuel_type", "body_group"],
        "levels": [df[c].nunique() for c in ["make", "model", "make_model", "vehicle_class",
                                              "transmission", "fuel_type", "body_group"]],
    })
    return cards
