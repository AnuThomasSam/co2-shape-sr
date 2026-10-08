"""Step 01: read raw CSV files and harmonise them. Keeps whitelisted columns only."""
from __future__ import annotations

from ..config import enabled_datasets
from ..data.ingest import ingest_dataset, raw_files
from ..io import write_csv
from .common import harmonised_file, sdir

CODE_DEPS = ['co2sr.data.ingest']
NAME = "01_ingest"


def config_part(cfg):
    return {"datasets": cfg["datasets"]}


def inputs(cfg):
    files = []
    for ds in enabled_datasets(cfg):
        files += raw_files(ds, cfg["datasets"][ds]["raw_glob"])
    return files


def run(cfg, log, **_):
    outs = []
    for ds in enabled_datasets(cfg):
        df, report = ingest_dataset(ds, cfg["datasets"][ds]["raw_glob"])
        outs.append(write_csv(df, harmonised_file(cfg, ds)))
        outs.append(write_csv(report, sdir(cfg, "01") / f"{ds}_ingest_report.csv"))
        log.info("%s: %d rows from %d file(s)", ds, len(df), len(report))
    return outs
