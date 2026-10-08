"""Step 06: per block, the domain contexts, the sample Q_D, the evaluation grid and the boxes.

Contexts come from the training rows (fit + validation) of the block only.
"""
from __future__ import annotations

from ..config import enabled_datasets
from ..data.domain import build_contexts, domain_boxes, domain_grid, domain_sample
from ..io import write_csv
from .common import block_frames, domain_dir, load_clean, load_splits, protocols_for, split_file

CODE_DEPS = ['co2sr.data.domain']
NAME = "06_domain"


def config_part(cfg):
    return {"domain": cfg["domain"]}


def inputs(cfg):
    out = []
    for ds in enabled_datasets(cfg):
        out += [p for p in (split_file(cfg, ds, pr) for pr in protocols_for(cfg, ds)) if p.exists()]
    return out


def run(cfg, log, **_):
    dc = cfg["domain"]
    outs = []
    for ds in enabled_datasets(cfg):
        df = load_clean(cfg, ds)
        for p in protocols_for(cfg, ds):
            if not split_file(cfg, ds, p).exists():
                continue
            for b in load_splits(cfg, ds, p)["blocks"]:
                fr = block_frames(df, b)
                ctx = build_contexts(fr["full"], dc)
                d = domain_dir(cfg, ds, p, b["block_id"])
                outs.append(write_csv(ctx, d / "contexts.csv"))
                outs.append(write_csv(domain_sample(ctx, dc["sample_size"], dc["sample_seed"]),
                                      d / "domain_sample.csv"))
                outs.append(write_csv(domain_grid(ctx, dc["grid_points_per_context"], dc["max_grid_contexts"],
                                                  dc["grid_seed"]), d / "domain_grid.csv"))
                lo, hi = domain_boxes(ctx, dc["interval_segments_train"], dc["max_interval_contexts_train"],
                                      dc["sample_seed"])
                outs += [write_csv(lo, d / "boxes_train_lo.csv"), write_csv(hi, d / "boxes_train_hi.csv")]
                lo, hi = domain_boxes(ctx, dc["interval_segments_eval"], dc["max_grid_contexts"], dc["grid_seed"])
                outs += [write_csv(lo, d / "boxes_eval_lo.csv"), write_csv(hi, d / "boxes_eval_hi.csv")]
                # DCSR-V root boxes: all contexts (no subsampling), refined adaptively during search
                lo, hi = domain_boxes(ctx, dc.get("verify_root_segments", 1), len(ctx), dc["sample_seed"])
                outs += [write_csv(lo, d / "boxes_verify_lo.csv"), write_csv(hi, d / "boxes_verify_hi.csv")]
            log.info("%s %s: domain files written", ds, p)
    return outs
