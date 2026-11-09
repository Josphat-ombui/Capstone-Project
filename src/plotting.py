"""Shared plot style, colours and figure saving."""
import matplotlib.pyplot as plt

from src.config import REPORTS_FIGURES

PRIMARY = "#1F4E79"    # dark blue: main series
ALERT = "#C0392B"      # red: failures, targets, warnings
ACCENT = "#E08E0B"     # amber: second series, limits
POSITIVE = "#2E8B57"   # green: good outcome, best model
NEUTRAL = "#7F8C8D"    # grey: reference lines, history


def set_plot_style():
    """Apply one clean style to every chart."""
    plt.rcParams.update({
        "figure.dpi": 100,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "legend.frameon": False,
        "font.size": 10,
    })


def save_figure(fig, name):
    """Save a figure as reports/figures/<name>.png."""
    REPORTS_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(REPORTS_FIGURES / f"{name}.png", dpi=150, bbox_inches="tight")
