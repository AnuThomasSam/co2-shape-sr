"""Step 03: empirical support for the monotone prior (matched pairs)."""
from __future__ import annotations

from ..config import enabled_datasets
from ..io import write_csv
from ..prior import matched_pairs, support_summary
from .common import clean_file, load_clean, sdir

CODE_DEPS = ['co2sr.prior']
NAME = "03_prior"


def config_part(cfg):
    return {"prior": cfg["prior"]}


def inputs(cfg):
    return [clean_file(cfg, ds) for ds in enabled_datasets(cfg)]


def run(cfg, log, **_):
    outs = []
    for ds in enabled_datasets(cfg):
        df = load_clean(cfg, ds)
        has_year = cfg["datasets"][ds].get("has_model_year")
        keys = cfg["prior"]["match_keys_with_year" if has_year else "match_keys_without_year"]
        pairs = matched_pairs(df, keys)
        summ = support_summary(pairs, "model_year" if has_year else None)
        summ.insert(0, "match_keys", "+".join(keys))
        outs.append(write_csv(pairs, sdir(cfg, "03") / f"{ds}_matched_pairs.csv"))
        outs.append(write_csv(summ, sdir(cfg, "03") / f"{ds}_prior_support.csv"))
        log.info("%s: %d matched pairs, larger engine lower CO2 in %.1f%%", ds, len(pairs),
                 100 * summ.iloc[0]["violation_share"] if len(pairs) else float("nan"))
    return outs
