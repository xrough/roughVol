from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

DEEP_BLUE = "#274c77"
MID_BLUE = "#4f6d8c"
SLATE_GREY = "#6b7280"
LIGHT_GREY = "#d9dee7"
PALE_BLUE = "#c7d3e3"


def configure_libertine_style() -> None:
    available_fonts = {font.name for font in fm.fontManager.ttflist}
    libertine_candidates = [
        "Linux Libertine O",
        "Libertinus Serif",
        "LinLibertine",
    ]
    preferred_serif = [font for font in libertine_candidates if font in available_fonts]
    preferred_serif += ["DejaVu Serif"]

    plt.rcParams.update(
        {
            "figure.facecolor": "#f5f7fa",
            "axes.facecolor": "#f5f7fa",
            "axes.edgecolor": SLATE_GREY,
            "axes.labelcolor": "#243447",
            "axes.titlecolor": "#16202a",
            "xtick.color": "#243447",
            "ytick.color": "#243447",
            "grid.color": LIGHT_GREY,
            "font.family": "serif",
            "font.serif": preferred_serif,
            "mathtext.fontset": "dejavuserif",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
