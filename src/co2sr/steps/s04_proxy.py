"""Step 04: K-Means technology proxy and its validity (RQ1)."""
from __future__ import annotations

from ..config import enabled_datasets
from ..io import write_csv, write_json
from ..proxy import technology_split, validity
from .common import clean_file, load_clean, proxy_assign_file, sdir

CODE_DEPS = ['co2sr.proxy']
NAME = "04_proxy"


def config_part(cfg):
    return {"proxy": cfg["proxy"], "seed": cfg["project"]["random_seed"]}


def inputs(cfg):
    return [clean_file(cfg, ds) for ds in enabled_datasets(cfg)]


def run(cfg, log, **_):
    outs = []
    seed = cfg["project"]["random_seed"]
    for ds in enabled_datasets(cfg):
        df = load_clean(cfg, ds)
        ksel, prof, assign, info = technology_split(df, cfg["proxy"], seed)
        val = validity(df, assign, cfg["proxy"], seed)
        d = sdir(cfg, "04")
        outs.append(write_csv(ksel, d / f"{ds}_proxy_k_selection.csv"))
        outs.append(write_csv(prof, d / f"{ds}_proxy_cluster_profile.csv"))
        outs.append(write_csv(assign, proxy_assign_file(cfg, ds)))
        outs.append(write_csv(val, d / f"{ds}_proxy_validity.csv"))
        outs.append(write_json(info, d / f"{ds}_proxy_info.json"))
        log.info("%s: K=%d, %d distinct points, older=%d newer=%d", ds, info["k"],
                 info["distinct_points"], info["rows_older"], info["rows_newer"])
    return outs
