"""Shared journal figure style: serif, muted print-safe palette, IEEE column widths, PDF + 300 dpi PNG."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

IEEE_COL_IN = 3.5   # single column width (inches)
IEEE_PAGE_IN = 7.16  # double column (full page) width
PALETTE = ["#2F4B7C", "#4D7F5E", "#8C6D31", "#A05195", "#665191", "#D45087", "#5A5A5A"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "lines.linewidth": 1.0, "lines.markersize": 3,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "axes.grid": True, "grid.linewidth": 0.3, "grid.alpha": 0.5,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def save(fig, stem: str | Path) -> None:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    plt.close(fig)
