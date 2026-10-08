"""Model registry."""
from __future__ import annotations

from .base import BaseModel
from .gam import MonotoneGAM
from .gp_model import GPModel
from .linear import LogLinearModel, OLSModel
from .pysr_model import PySRModel, pysr_available
from .xgb import XGBModel

REGISTRY = {
    "ols": OLSModel,
    "loglinear": LogLinearModel,
    "gam": MonotoneGAM,
    "xgb": XGBModel,
    "pysr": PySRModel,
    "gp": GPModel,
}


def build_model(spec: dict, cfg: dict, seed: int) -> BaseModel:
    return REGISTRY[spec["type"]](spec, cfg, seed)


def load_model(run_dir) -> BaseModel:
    return BaseModel.load(run_dir)


__all__ = ["build_model", "load_model", "pysr_available", "REGISTRY"]
