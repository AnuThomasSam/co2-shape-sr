#!/usr/bin/env python
"""Command-line entry point.

Examples
  python pipeline.py run all                        # every step, skipping up-to-date ones
  python pipeline.py run 01 02 03                   # selected steps
  python pipeline.py run 07 --experiment main --methods dcsr_s gp_standard --seeds 0-9 --n-jobs 8
  python pipeline.py run 11 --force                 # redraw all figures
  python pipeline.py status                         # which steps are up to date
  python pipeline.py --config configs/smoke.yaml run all

Each step reads only the files written by earlier steps (outputs/<NN_step>/...),
so a step can be re-run on its own after a change, without re-running the rest.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from co2sr.config import load_config, outputs_dir  # noqa: E402
from co2sr.io import code_hash, get_logger, is_up_to_date, write_manifest  # noqa: E402
from co2sr.steps.common import sdir  # noqa: E402

STEPS = {
    "01": "s01_ingest", "02": "s02_clean", "03": "s03_prior", "04": "s04_proxy", "05": "s05_splits",
    "06": "s06_domain", "07": "s07_train", "08": "s08_evaluate", "09": "s09_statistics",
    "10": "s10_audit", "11": "s11_figures", "12": "s12_tables",
    "13": "s13_benchmark", "14": "s14_benchmark_report",
}


def parse_seeds(text):
    if text is None:
        return None
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def run_steps(cfg, steps, args):
    log_dir = outputs_dir(cfg) / "logs"
    for s in steps:
        mod = importlib.import_module(f"co2sr.steps.{STEPS[s]}")
        log = get_logger(f"step{s}", log_dir / f"step{s}_{mod.NAME}.log")
        part = {"config": mod.config_part(cfg),
                "code": code_hash([mod.__name__, "co2sr.steps.common"] + list(getattr(mod, "CODE_DEPS", [])))}
        ins = mod.inputs(cfg)
        incremental = getattr(mod, "INCREMENTAL", False)
        if not incremental and not args.force and is_up_to_date(sdir(cfg, s), part, ins):
            log.info("up to date, skipped (use --force to re-run)")
            continue
        t0 = time.time()
        log.info("started")
        outs = mod.run(cfg, log, force=args.force, experiments=args.experiment, methods=args.methods,
                       protocols=args.protocols, blocks=args.blocks, seeds=parse_seeds(args.seeds),
                       n_jobs=args.n_jobs)
        if not incremental:
            write_manifest(sdir(cfg, s), mod.NAME, part, ins, outs)
        log.info("finished in %.1f s", time.time() - t0)


def status(cfg):
    for s, name in STEPS.items():
        mod = importlib.import_module(f"co2sr.steps.{name}")
        if getattr(mod, "INCREMENTAL", False):
            state = "incremental (per-run caching)"
        else:
            part = {"config": mod.config_part(cfg),
                    "code": code_hash([mod.__name__, "co2sr.steps.common"] + list(getattr(mod, "CODE_DEPS", [])))}
            state = "up to date" if is_up_to_date(sdir(cfg, s), part, mod.inputs(cfg)) else "needs run"
        print(f"{s} {mod.NAME:<16} {state}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("steps", nargs="+", help="'all' or step numbers, e.g. 01 02 07")
    r.add_argument("--force", action="store_true")
    r.add_argument("--experiment", nargs="*", help="step 07: experiments to run (default: all)")
    r.add_argument("--methods", nargs="*")
    r.add_argument("--protocols", nargs="*")
    r.add_argument("--blocks", nargs="*")
    r.add_argument("--seeds", help="step 07: e.g. 0-9 or 0,3,5")
    r.add_argument("--n-jobs", type=int, default=None)
    sub.add_parser("status")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.cmd == "status":
        status(cfg)
        return
    steps = list(STEPS) if args.steps == ["all"] else [s.zfill(2) for s in args.steps]
    bad = [s for s in steps if s not in STEPS]
    if bad:
        ap.error(f"unknown step(s): {bad}")
    run_steps(cfg, steps, args)


if __name__ == "__main__":
    main()
