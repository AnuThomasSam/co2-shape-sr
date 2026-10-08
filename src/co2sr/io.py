"""File helpers, logging and step manifests.

Each step writes a manifest (outputs/<step>/_manifest.json) recording the hash of
the configuration it used and a fingerprint of its input files. The pipeline
skips a step when neither has changed and all recorded outputs still exist.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import stable_hash

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(LOG_FORMAT, "%H:%M:%S"))
        logger.addHandler(h)
        logger.propagate = False
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_file.resolve()
                   for h in logger.handlers):
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter(LOG_FORMAT))
            logger.addHandler(fh)
    return logger


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"Not JSON serialisable: {type(o)}")


def write_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
    tmp.replace(path)
    return path


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    return path


def read_csv(path: Path, **kw) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} does not exist. Run the step that produces it first "
            f"(see README, 'Pipeline steps').")
    return pd.read_csv(path, **kw)


def file_fingerprint(paths: Iterable[Path]) -> str:
    items = []
    for p in sorted(Path(x) for x in paths):
        if p.exists():
            st = p.stat()
            items.append([str(p), st.st_size, int(st.st_mtime)])
        else:
            items.append([str(p), None, None])
    return stable_hash(items, 16)


def manifest_path(step_dir: Path) -> Path:
    return step_dir / "_manifest.json"


def is_up_to_date(step_dir: Path, config_part, inputs: Iterable[Path]) -> bool:
    mp = manifest_path(step_dir)
    if not mp.exists():
        return False
    m = read_json(mp)
    if m.get("config_hash") != stable_hash(config_part):
        return False
    if m.get("inputs_fingerprint") != file_fingerprint(inputs):
        return False
    return all(Path(o).exists() for o in m.get("outputs", []))


def write_manifest(step_dir: Path, step: str, config_part, inputs: Iterable[Path],
                   outputs: Iterable[Path], extra: dict | None = None) -> None:
    inputs = list(inputs)
    write_json({
        "step": step,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config_hash": stable_hash(config_part),
        "inputs": [str(p) for p in inputs],
        "inputs_fingerprint": file_fingerprint(inputs),
        "outputs": [str(p) for p in outputs],
        **(extra or {}),
    }, manifest_path(step_dir))


def code_hash(module_names: Iterable[str]) -> str:
    """Hash of the source code of the given modules, so that a code change re-runs a step."""
    import hashlib
    import importlib
    h = hashlib.sha256()
    for name in sorted(set(module_names)):
        mod = importlib.import_module(name)
        h.update(Path(mod.__file__).read_bytes())
    return h.hexdigest()[:16]
