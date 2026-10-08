"""Read raw CSV files and harmonise them to one schema.

The reader is deliberately defensive because NRCan files differ between years:
header row position, one- or two-row headers, column spelling, footer notes,
and file encoding. Only a whitelist of columns is kept. Every other column,
including all fuel-consumption columns, the CO2 rating and the smog rating,
is dropped here, so no later step can use them by mistake.
"""
from __future__ import annotations

import csv
import glob
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT

# Canonical name -> accepted normalised spellings (lower case, letters and digits only)
SYNONYMS = {
    "model_year": ["modelyear", "year", "my"],
    "make": ["make", "manufacturer"],
    "model": ["model", "model1"],
    "vehicle_class": ["vehicleclassraw", "vehicleclass", "class"],
    "engine_size": ["enginesizel", "enginesize", "enginesizelitres", "engine", "displacementl"],
    "cylinders": ["cylinders", "cyl", "numberofcylinders"],
    "transmission": ["transmission", "trans"],
    "fuel_code": ["fueltypecode", "fueltype", "fuel"],
    "co2": ["co2emissionsgkm", "co2emissionsgkm1", "co2emissions", "co2gkm", "co2emissionsg"],
}

FUEL_NAME_TO_CODE = {
    "regular gasoline": "X", "premium gasoline": "Z", "diesel": "D",
    "ethanol": "E", "ethanol (e85)": "E", "e85": "E", "natural gas": "N",
}

KEEP = list(SYNONYMS.keys())


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _read_lines(path: Path) -> list[list[str]]:
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    return list(csv.reader(io.StringIO(text)))


def _find_header(rows: list[list[str]]) -> tuple[int, list[str]]:
    """Locate the header row (contains 'make' and 'model'); merge a second header row if present."""
    for i, row in enumerate(rows[:40]):
        cells = [_norm(c) for c in row]
        if "make" in cells and any(c.startswith("model") and c != "modelyear" for c in cells):
            header = [c.strip() for c in row]
            # A second header row holds unit/sub labels and no numbers.
            if i + 1 < len(rows):
                nxt = rows[i + 1]
                non_empty = [c for c in nxt if c.strip()]
                has_numbers = any(re.fullmatch(r"-?\d+(\.\d+)?", c.strip()) for c in non_empty)
                if non_empty and not has_numbers and len(non_empty) < len(header):
                    header = [(h + " " + s).strip() for h, s in
                              zip(header, nxt + [""] * (len(header) - len(nxt)))]
                    return i + 1, header
            return i, header
    raise ValueError("Could not find a header row containing 'Make' and 'Model'.")


def _map_columns(header: list[str]) -> dict[str, int]:
    normed = [_norm(h) for h in header]
    mapping: dict[str, int] = {}
    for canon, options in SYNONYMS.items():
        for opt in options:
            hits = [j for j, h in enumerate(normed) if h == opt]
            if hits:
                mapping[canon] = hits[0]
                break
    # Prefer a fuel code column over a fuel name column when both exist
    code_hits = [j for j, h in enumerate(normed) if h == "fueltypecode"]
    if code_hits:
        mapping["fuel_code"] = code_hits[0]
    return mapping


def read_raw_file(path: Path, dataset: str) -> tuple[pd.DataFrame, dict]:
    rows = _read_lines(path)
    hdr_idx, header = _find_header(rows)
    mapping = _map_columns(header)
    missing = [c for c in KEEP if c not in mapping and not (c == "model_year" and dataset == "kaggle")]
    if missing:
        raise ValueError(f"{path.name}: required columns not found: {missing}. Header: {header}")
    data = []
    for r in rows[hdr_idx + 1:]:
        if not any(c.strip() for c in r):
            continue
        rec = {c: (r[j].strip() if j < len(r) else "") for c, j in mapping.items()}
        data.append(rec)
    df = pd.DataFrame(data)
    report = {
        "file": path.name,
        "header_row": hdr_idx,
        "n_raw_rows": len(df),
        "kept_columns": ";".join(f"{c}<-{header[j]}" for c, j in mapping.items()),
        "dropped_columns": ";".join(h for j, h in enumerate(header) if j not in mapping.values()),
    }
    # Footer notes: rows without a numeric engine size or (for NRCan) a numeric year
    df["engine_size"] = pd.to_numeric(df["engine_size"], errors="coerce")
    ok = df["engine_size"].notna() & (df["make"].astype(str).str.len() > 0)
    if "model_year" in df:
        df["model_year"] = pd.to_numeric(df["model_year"], errors="coerce")
        ok &= df["model_year"].notna()
    report["n_footer_or_invalid_rows"] = int((~ok).sum())
    df = df[ok].copy()
    df["source_file"] = path.name
    return df, report


def harmonise(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    out = pd.DataFrame(index=df.index)
    out["model_year"] = (df["model_year"].round().astype("Int64") if "model_year" in df
                         else pd.Series([pd.NA] * len(df), dtype="Int64"))
    out["make"] = df["make"].astype(str).str.strip().str.upper()
    out["model"] = df["model"].astype(str).str.strip()
    out["vehicle_class"] = df["vehicle_class"].astype(str).str.strip()
    out["engine_size"] = pd.to_numeric(df["engine_size"], errors="coerce")
    out["cylinders"] = pd.to_numeric(df["cylinders"], errors="coerce")
    out["transmission"] = df["transmission"].astype(str).str.strip().str.upper()
    fuel = df["fuel_code"].astype(str).str.strip()
    fuel = fuel.where(fuel.str.len() <= 2, fuel.str.lower().map(FUEL_NAME_TO_CODE))
    out["fuel_type"] = fuel.str.upper()
    out["co2"] = pd.to_numeric(df["co2"], errors="coerce")
    out["source_file"] = df["source_file"].values
    out["dataset"] = dataset
    return out.reset_index(drop=True)


def ingest_dataset(dataset: str, raw_glob: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = sorted(Path(p) for p in glob.glob(str(PROJECT_ROOT / raw_glob)))
    if not files:
        raise FileNotFoundError(
            f"No raw files for dataset '{dataset}' matching {raw_glob}. "
            f"See data/raw/{dataset}/README.md for where to obtain them.")
    frames, reports = [], []
    for f in files:
        df, rep = read_raw_file(f, dataset)
        frames.append(harmonise(df, dataset))
        reports.append(rep)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(reports)


def raw_files(dataset: str, raw_glob: str) -> list[Path]:
    return sorted(Path(p) for p in glob.glob(str(PROJECT_ROOT / raw_glob)))
