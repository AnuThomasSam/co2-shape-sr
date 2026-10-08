"""Step 05: evaluation protocols P1-P4 as row-id lists, plus information ceilings."""
from __future__ import annotations

import pandas as pd

from ..config import enabled_datasets
from ..io import read_csv, write_csv, write_json
from ..metrics import CEILING_KEYS, information_ceiling
from ..splits import check_no_overlap, p1_rolling, p2_extrapolation, p3_proxy, p4_grouped_cv
from .common import block_frames, clean_file, load_clean, protocols_for, proxy_assign_file, sdir, split_file

CODE_DEPS = ['co2sr.splits', 'co2sr.metrics']
NAME = "05_splits"


def config_part(cfg):
    return {"protocols": cfg["protocols"], "seed": cfg["project"]["random_seed"]}


def inputs(cfg):
    out = []
    for ds in enabled_datasets(cfg):
        out += [clean_file(cfg, ds), proxy_assign_file(cfg, ds)]
    return out


def run(cfg, log, **_):
    outs, summary, ceilings = [], [], []
    seed = cfg["project"]["random_seed"]
    pc = cfg["protocols"]
    for ds in enabled_datasets(cfg):
        df = load_clean(cfg, ds)
        for p in protocols_for(cfg, ds):
            if p == "P1":
                blocks = p1_rolling(df, pc["P1"])
            elif p == "P2":
                blocks = p2_extrapolation(df, pc["P2"], seed)
            elif p == "P3":
                blocks = p3_proxy(df, read_csv(proxy_assign_file(cfg, ds)), pc["P3"], seed)
            else:
                blocks = p4_grouped_cv(df, pc["P4"], seed)
            check_no_overlap(blocks)
            if not blocks:
                log.warning("%s %s: no blocks could be formed", ds, p)
                continue
            outs.append(write_json({"dataset": ds, "protocol": p, "blocks": blocks}, split_file(cfg, ds, p)))
            for b in blocks:
                fr = block_frames(df, b)
                for tname, tdf in fr["tests"].items():
                    summary.append({"dataset": ds, "protocol": p, "block": b["block_id"], "test_set": tname,
                                    "n_fit": len(b["fit"]), "n_val": len(b["val"]), "n_test": len(tdf),
                                    **{k: str(v) for k, v in b["meta"].items()}})
                    for fs, keys in CEILING_KEYS.items():
                        ceilings.append({"dataset": ds, "protocol": p, "block": b["block_id"],
                                         "test_set": tname, "feature_set": fs,
                                         "r2_ceiling": information_ceiling(tdf, keys)})
            log.info("%s %s: %d block(s)", ds, p, len(blocks))
    outs.append(write_csv(pd.DataFrame(summary), sdir(cfg, "05") / "split_summary.csv"))
    outs.append(write_csv(pd.DataFrame(ceilings), sdir(cfg, "05") / "information_ceiling.csv"))
    return outs
