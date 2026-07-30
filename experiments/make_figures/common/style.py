"""The visual grammar shared by every figure.

Two base hues, from which the sequential and diverging colormaps are derived, plus
the rcParams theme applied on import. The grammar: color and marker encode the
delay type, line style and hatch encode the condition. Where only the condition
varies, color encodes it instead.

Nothing else in the analysis hard-codes a color, editing the two hues re-themes
every figure.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from cycler import cycler

# ========================== EDIT HERE, PALETTE =========================== #
# Two base hues, the only knobs. Everything else (categorical assignment, the
# sequential and diverging ramps) is derived from these two. A muted purple and a
# soft coral: the repo's two main plot colors.
BLUE = "#7753a1"    # cool, axonal / learned (muted purple)
ORANGE = "#d3444c"  # warm, synaptic / fixed (warm red, slight orange lean)

# Color = delay type.
TYPE_COLORS = {"axonal": BLUE, "synaptic": ORANGE}
# Marker = delay type too (a redundant channel, for line plots that also carry the
# condition as line style / marker fill).
TYPE_MARKERS = {"axonal": "o", "synaptic": "s"}
# Color = condition, used only where type is constant (e.g. SSC learned vs fixed).
CONDITION_COLORS = {"learned": BLUE, "fixed": ORANGE}
# Condition as style (used when color already encodes type, e.g. HAR).
CONDITION_LS = {"learned": "-", "fixed": "--"}      # line style
CONDITION_HATCH = {"learned": None, "fixed": "///"}  # bar / histogram fill hatch

# Neutral gray midpoint for the diverging ramp (never a hue at the center).
MID_GRAY = "#f2f1ee"

# Ink & chrome.
INK_PRIMARY = "#0b0b0b"    # titles, values
INK_SECONDARY = "#52514e"  # axis labels
INK_MUTED = "#898781"      # ticks, minor labels
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"       # axis spines
SURFACE = "#ffffff"        # figure / axes background (white for print)

# =========================== EDIT HERE, RANGES =========================== #
FIGSIZE_LINE = (5.0, 3.6)       # single line/scatter panel
FIGSIZE_PAIR = (5.0, 3.6)       # fixed-vs-learned pair panel
ACC_YLIM = None                 # accuracy axes. None = auto, or e.g. (0, 100)
# Ordinal ramp sampling window: skip the lightest steps (they recede into the
# surface) so discrete ordered lines/markers stay legible.
SEQ_ORDINAL_LO = 0.28
SEQ_ORDINAL_HI = 1.0
# ========================================================================= #


# --------------------------------------------------------------------------- #
# Derived colormaps, built from BLUE / ORANGE so the two base hues are the only
# thing to edit. (Registered too, so cmap="delrec_seq"/"delrec_div" also work.)
# --------------------------------------------------------------------------- #
def _mix(hex_c, t, toward="#ffffff"):
    """Blend ``hex_c`` a fraction ``t`` toward ``toward`` (white by default)."""
    from matplotlib.colors import to_rgb
    a, b = to_rgb(hex_c), to_rgb(toward)
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))

# Sequential: one hue light→dark (ordered magnitude: std sweep, heatmaps).
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "delrec_seq", [_mix(BLUE, 0.90), BLUE, _mix(BLUE, 0.55, "#000000")])
# Diverging: warm ↔ gray ↔ purple (signed: Δ, gradients). Purple is the positive
# (high) end and the warm red the negative (low) end, so on a Δ(learned−fixed)
# map, "learned wins" reads purple. Poles deepened toward black.
DIVERGING = LinearSegmentedColormap.from_list(
    "delrec_div", [_mix(ORANGE, 0.35, "#000000"), ORANGE, MID_GRAY,
                   BLUE, _mix(BLUE, 0.35, "#000000")])
for _cm in (SEQUENTIAL, DIVERGING):
    try:
        mpl.colormaps.register(_cm, force=True)
    except Exception:
        pass


def type_of(mkey):
    """'synaptic' if the model key names a synaptic family, else 'axonal'."""
    return "synaptic" if "syn" in mkey else "axonal"


def condition_of(mkey):
    """'fixed' if the model key names a fixed-delay family, else 'learned'."""
    return "fixed" if "fixed" in mkey else "learned"


def ordinal_colors(n):
    """``n`` discrete colors from the sequential ramp (ordered categories such as
    the std sweep), skipping the near-surface lightest steps."""
    import numpy as np
    xs = np.linspace(SEQ_ORDINAL_LO, SEQ_ORDINAL_HI, max(n, 1))
    return [SEQUENTIAL(x) for x in xs]


# --------------------------------------------------------------------------- #
# rcParams
# --------------------------------------------------------------------------- #
RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "text.color": INK_PRIMARY,
    "axes.labelcolor": INK_SECONDARY,
    "axes.edgecolor": BASELINE,
    "axes.linewidth": 0.8,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.grid": False,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "svg.fonttype": "none",   # keep text as selectable text in the SVG
    # Fallback categorical order for any plot that doesn't assign colors by role.
    "axes.prop_cycle": cycler(color=[BLUE, ORANGE, "#1baf7a", "#4a3aa7",
                                     "#e34948", "#eda100", "#e87ba4", "#008300"]),
}


# Presentation-size text: RC's text elements scaled ~1.3×, for figures that end up
# projected or printed small enough that the default sizes stop being legible. Scoped
# on purpose, use it per figure, so the rest of the gallery keeps RC's sizes:
#
#     with plt.rc_context(st.RC_LARGE_TEXT):
#         fig, ax = plt.subplots(...)
#         ...                       # tight_layout + save inside, they measure text
#
# Only sizes. Every color/spine/grid convention still comes from RC.
RC_LARGE_TEXT = {
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "legend.fontsize": 12,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.titlesize": 17,
}


# Print-size text: RC's text elements scaled ~0.75×, plus thinner chrome, for
# figures that go into the paper at (or near) 1:1 as a sub-panel of a larger
# figure, at a ~3 in panel width RC's 11-12 pt would be oversized. Scoped like
# RC_LARGE_TEXT (use it per figure, inside a ``plt.rc_context``). Only sizes and
# line widths, every color convention still comes from RC.
RC_PAPER = {
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.titlesize": 9,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
}


def apply():
    """Install the DelRec theme into matplotlib's global rcParams."""
    plt.rcParams.update(RC)


# Applied on import so simply importing ``style`` themes every figure.
apply()
