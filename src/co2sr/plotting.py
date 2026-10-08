"""Shared figure style. Figures carry no titles (captions are written in the thesis)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DCSR_MARKERS = {"dcsr_t": "v", "dcsr_s": "o", "dcsr_f": "s", "dcsr_ia": "D", "dcsr_v": "P", "dcsr_dyn": "X"}


def apply_style(cfg: dict) -> None:
    plt.style.use(cfg["plotting"].get("style", "seaborn-v0_8-whitegrid"))
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": cfg["plotting"].get("dpi", 300),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.linewidth": 0.5,
        "grid.color": "#dddddd",
        "lines.linewidth": 1.5,
    })


def save(fig, out_dir: Path, name: str, cfg: dict) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in cfg["plotting"].get("formats", ["png", "pdf"]):
        p = out_dir / f"{name}.{fmt}"
        fig.savefig(p, format=fmt, dpi=cfg["plotting"].get("dpi", 300), bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    return paths


def color(cfg: dict, method: str) -> str:
    cols = cfg["plotting"]["colors"]
    if method in cols:
        return cols[method]
    if method.startswith("dcsr"):
        return cols.get("dcsr", "#e34948")
    if method.startswith("gp"):
        return cols.get("gp_standard", "#008300")
    return "#6b6b6b"


def label(cfg: dict, method: str) -> str:
    return cfg["plotting"]["labels"].get(method, method)
