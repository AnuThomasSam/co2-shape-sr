"""Step 02: cleaning rules, body groups, domain flags and the data card."""
from __future__ import annotations

from ..config import enabled_datasets
from ..data.clean import clean, data_card
from ..io import read_csv, write_csv
from .common import clean_file, harmonised_file, sdir

CODE_DEPS = ['co2sr.data.clean']
NAME = "02_clean"


def config_part(cfg):
    return {"cleaning": cfg["cleaning"], "domain": {k: cfg["domain"][k] for k in
            ("engine_size_min", "engine_size_max", "cylinders", "litres_per_cylinder_min",
             "litres_per_cylinder_max")}, "datasets": cfg["datasets"]}


def inputs(cfg):
    return [harmonised_file(cfg, ds) for ds in enabled_datasets(cfg)]


def run(cfg, log, **_):
    outs = []
    for ds in enabled_datasets(cfg):
        raw = read_csv(harmonised_file(cfg, ds), dtype={"model": str, "make": str})
        df, clog = clean(raw, cfg, ds)
        outs.append(write_csv(df, clean_file(cfg, ds)))
        outs.append(write_csv(clog, sdir(cfg, "02") / f"{ds}_cleaning_log.csv"))
        for name, tab in data_card(df).items():
            outs.append(write_csv(tab, sdir(cfg, "02") / f"{ds}_card_{name}.csv"))
        log.info("%s: %d rows after cleaning", ds, len(df))
    return outs
