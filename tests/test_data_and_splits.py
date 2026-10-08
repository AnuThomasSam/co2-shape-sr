"""Tests for ingestion, cleaning and splits."""
import csv

import numpy as np
import pandas as pd

from co2sr.data.clean import body_group, parse_transmission
from co2sr.data.ingest import read_raw_file
from co2sr.splits import check_no_overlap, p1_rolling, p2_extrapolation, p4_grouped_cv


def _toy(n=600, seed=0):
    rng = np.random.default_rng(seed)
    years = rng.integers(2015, 2022, n)
    mm = rng.integers(0, 60, n)
    eng = np.round(rng.uniform(1.0, 6.0, n), 1)
    df = pd.DataFrame({
        "row_id": [f"t-{i:05d}" for i in range(n)], "model_year": years, "make_model": [f"M|{m}" for m in mm],
        "engine_size": eng, "cylinders": rng.choice([3, 4, 6, 8], n), "co2": 100 + 30 * eng,
    })
    df["spec_key"] = df["make_model"] + "|" + df["engine_size"].astype(str)
    return df


def test_p1_is_strictly_temporal_and_disjoint():
    df = _toy()
    blocks = p1_rolling(df, {"window_years": 3, "min_fit_rows": 10})
    check_no_overlap(blocks)
    yr = df.set_index("row_id")["model_year"]
    for b in blocks:
        test_years = yr.loc[b["tests"]["all"]]
        assert yr.loc[b["fit"] + b["val"]].max() < test_years.min()
        assert set(b["tests"]["new_config"]) <= set(b["tests"]["all"])


def test_p4_groups_do_not_cross_folds():
    df = _toy()
    blocks = p4_grouped_cv(df, {"n_folds": 5, "validation_fraction": 0.15}, 1)
    check_no_overlap(blocks)
    mm = df.set_index("row_id")["make_model"]
    for b in blocks:
        assert not set(mm.loc[b["fit"] + b["val"]]) & set(mm.loc[b["tests"]["all"]])
    covered = sorted(sum((b["tests"]["all"] for b in blocks), []))
    assert covered == sorted(df["row_id"])


def test_p2_tails_are_outside_training_range():
    df = _toy()
    blocks = p2_extrapolation(df, {"lower_quantile": 0.1, "upper_quantile": 0.9, "unseen_cylinders": 3,
                                   "validation_fraction": 0.15}, 0)
    check_no_overlap(blocks)
    e = df.set_index("row_id")["engine_size"]
    b = blocks[0]
    tr = e.loc[b["fit"] + b["val"]]
    assert e.loc[b["tests"]["low_tail"]].max() < tr.min()
    assert e.loc[b["tests"]["high_tail"]].min() > tr.max()


def test_transmission_and_body_group_parsing():
    cvt = ["AV", "AV1"]
    assert parse_transmission("AS10", cvt) == ("AS", 10.0)
    t, g = parse_transmission("AV", cvt)
    assert t == "AV" and np.isnan(g)
    t, g = parse_transmission("AV1", cvt)
    assert t == "AV" and np.isnan(g)
    assert parse_transmission("AV7", cvt) == ("AV", 7.0)
    assert body_group("SUV: Small") == "suv" and body_group("SUV - STANDARD") == "suv"
    assert body_group("Minivan") == "van" and body_group("Pickup truck: Standard") == "pickup"
    assert body_group("Station wagon: Mid-size") == "car"
    assert body_group("Special purpose vehicle") == "special"


def test_ingest_two_row_header_and_footer(tmp_path):
    p = tmp_path / "f.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Model", "Make", "Model", "Vehicle", "Engine", "Cylinders", "Transmission", "Fuel",
                    "Fuel consumption", "CO2", "CO2", "Smog"])
        w.writerow(["year", "", "", "class", "size (L)", "", "", "type", "Comb (L/100 km)",
                    "emissions (g/km)", "rating", "rating"])
        w.writerow([2021, "ACURA", "ILX", "Compact", 2.4, 4, "AM8", "Z", 8.6, 199, 6, 3])
        w.writerow([])
        w.writerow(["Understanding the table"])
    df, rep = read_raw_file(p, "nrcan")
    assert len(df) == 1
    assert "Comb (L/100 km)" in rep["dropped_columns"] and "rating" in rep["dropped_columns"]
    assert set(["model_year", "make", "engine_size", "co2"]) <= set(df.columns)
    assert not any("consumption" in c.lower() for c in df.columns)
