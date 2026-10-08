"""Configuration loading, merging and hashing."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path) -> dict:
    """Load a YAML config. A file may declare `inherits: other.yaml` to override it."""
    path = Path(path)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    parent = cfg.pop("inherits", None)
    if parent:
        base = load_config(path.parent / parent)
        cfg = _deep_merge(base, cfg)
    cfg.setdefault("_meta", {})["config_path"] = str(path)
    return cfg


def stable_hash(obj: Any, length: int = 12) -> str:
    """Hash of a JSON-serialisable object, independent of key order."""
    text = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def method_spec(cfg: dict, name: str) -> dict:
    """Full specification of a method: defaults merged with its overrides."""
    if name not in cfg["methods"]:
        raise KeyError(f"Method '{name}' is not defined in the configuration.")
    spec = copy.deepcopy(cfg["methods"][name])
    if spec["type"] == "gp":
        spec = _deep_merge(cfg["gp_defaults"], spec)
    spec["name"] = name
    spec["stochastic"] = spec["type"] in cfg.get("stochastic_types", ["gp", "pysr"])
    return spec


def outputs_dir(cfg: dict) -> Path:
    return PROJECT_ROOT / cfg["project"]["outputs_dir"]


def results_dir(cfg: dict) -> Path:
    return PROJECT_ROOT / cfg["project"]["results_dir"]


def step_dir(cfg: dict, step: str) -> Path:
    d = outputs_dir(cfg) / step
    d.mkdir(parents=True, exist_ok=True)
    return d


def enabled_datasets(cfg: dict) -> list[str]:
    return [name for name, d in cfg["datasets"].items() if d.get("enabled", True)]
