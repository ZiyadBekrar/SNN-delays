"""One delay shared by a whole layer, against per-neuron and per-connection delays.

The shared-delay variant carries one scalar per recurrent layer, broadcast to
every connection, two delay parameters on HAR instead of hundreds or tens of
thousands. It never reaches the heterogeneous families, and barely moves from its
initialization, which is the argument for parameterizing delays per neuron or per
connection despite the extra parameters.

Outputs:  figures/HAR/common/
Usage:    python experiments/make_figures/HAR/analyze_commondelay.py
"""

import argparse
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "experiments", "make_figures")]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import common.utils as cu
from common import style
import runs as har

# Squarer panels than the reference's (8, 5).
FIGSIZE_SQUARE = (5.4, 5.0)

# The accuracy comparison is a paper panel (see plot_acc_vs_std): squarer than
# plot_final_acc_compare.py's (4.4, 7.2) stack, whose height only exists to separate
# the accuracy curves from the delay panel sharing their x.
FIGSIZE_ACC_PANEL = (5.4, 4.6)

# Per-layer column of the init-vs-final panel: the layers sit side by side, each a
# delay panel over a shorter displacement (final − init) panel sharing its x (width
# per layer, height of the delay row).
FIGSIZE_LAYER_PANEL = (4.4, 3.0)
DELTA_ROW_HEIGHT = 1.5

# Color = condition, the repo chart's encoding for figures where the delay type is
# constant (experiments/make_figures/common/style.py, SSC uses it the same way). Here the only
# contrast is the training itself, so the delay as drawn takes the "fixed" hue and the
# delay after training the "learned" one. The family's green is dropped in this figure:
# it names the family, which is not what this panel is contrasting.
INIT_COLOR = style.CONDITION_COLORS["fixed"]
FINAL_COLOR = style.CONDITION_COLORS["learned"]

# Color = delay type from the chart (axonal purple / synaptic red). Common is the
# chart's 3rd categorical hue (green). Marker = type. Line style / fill = condition
# (learned solid + filled, fixed dashed + open), the plot_final_acc_compare.py
# grammar, recolored onto style.py.
COMMON_COLOR = "#1baf7a"  # style.py's 3rd categorical hue
TYPE_COLORS = {"axonal": style.TYPE_COLORS["axonal"],
               "synaptic": style.TYPE_COLORS["synaptic"],
               "common": COMMON_COLOR}
TYPE_MARKERS = {"axonal": "o", "synaptic": "s", "common": "^"}
# (mkey, delay_type, is_fixed, label)
FAMILIES = [
    ("ax_learned",  "axonal",   False, "axonal learned"),
    ("ax_fixed",    "axonal",   True,  "axonal fixed"),
    ("syn_learned", "synaptic", False, "synaptic learned"),
    ("syn_fixed",   "synaptic", True,  "synaptic fixed"),
    ("common",      "common",   False, "common"),
]


def _resolve(mkey, std, exp_dir, seeds):
    if mkey == har.COMMON_KEY:
        return har.resolve_common_run_dirs(std, exp_dir, seeds)
    return har.resolve_run_dirs(mkey, std, exp_dir, seeds)


def _std_colors(stds):
    """One color per std from the chart's sequential (purple) ramp, ordered."""
    cols = style.ordinal_colors(len(stds))
    return {s: cols[i] for i, s in enumerate(sorted(stds))}


# --------------------------------------------------------------------------- #
# 1. Accuracy comparison
# --------------------------------------------------------------------------- #
def plot_acc_vs_std(stds, exp_dir, seeds, out_dir):
    """Final test accuracy vs delay_std_init, one line per family."""
    ticks = sorted(stds)
    pos = {v: i for i, v in enumerate(ticks)}
    rows = []
    with plt.rc_context(style.RC_LARGE_TEXT):
        fig, ax = plt.subplots(figsize=FIGSIZE_ACC_PANEL)
        for mkey, dtype, is_fixed, label in FAMILIES:
            color = TYPE_COLORS[dtype]
            marker = TYPE_MARKERS[dtype]
            linestyle = style.CONDITION_LS["fixed" if is_fixed else "learned"]
            xs, means, sems = [], [], []
            for std in ticks:
                accs = [har.read_test_accuracy(rd)
                        for rd in _resolve(mkey, std, exp_dir, seeds).values()]
                accs = [a for a in accs if a is not None]
                if not accs:
                    continue
                xs.append(pos[std])
                means.append(float(np.mean(accs)))
                sems.append(float(np.std(accs) / np.sqrt(len(accs))))
                rows.append({"curve": label, "delay_std_init": std,
                             "mean": float(np.mean(accs)), "std": float(np.std(accs)),
                             "sem": float(np.std(accs) / np.sqrt(len(accs))),
                             "n_seeds": len(accs)})
            if not xs:
                print(f"  WARNING: no data for {mkey}")
                continue
            xs, means, sems = np.array(xs), np.array(means), np.array(sems)
            ax.plot(xs, means, color=color, linestyle=linestyle, marker=marker,
                    markerfacecolor=style.SURFACE if is_fixed else color, label=label)
            ax.fill_between(xs, means - sems, means + sems, color=color, alpha=0.18,
                            linewidth=0)

        ax.set_xticks(list(pos.values()))
        ax.set_xticklabels([format(v, har.STD_AXIS.value_fmt) for v in ticks])
        ax.set_xlabel(har.STD_AXIS.label)
        ax.set_ylabel("final test accuracy (%)")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
        fig.tight_layout()
        # Inside the rc_context: savefig re-measures the text (bbox="tight").
        cu.save_fig(fig, os.path.join(out_dir, "acc_vs_std.svg"))
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, "acc_vs_std.csv"), index=False)



# --------------------------------------------------------------------------- #
# 2. Evolution of the shared (scalar) delay, common family only
# --------------------------------------------------------------------------- #
def _common_scalar_delays(std, exp_dir, seeds, ckpt):
    """Return {seed: (init_per_layer, final_per_layer)} of scalar delays for the
    common family at a given std. Both are 1-D arrays over recdel layers (positional
    pairing: init_delays.npz's layer{i} matches the i-th recurrent layer in best.pth)."""
    out = {}
    for seed, rd in sorted(_resolve(har.COMMON_KEY, std, exp_dir, seeds).items()):
        npz_path = os.path.join(rd, "init_delays.npz")
        if not os.path.exists(npz_path):
            continue
        z = np.load(npz_path)
        init = np.array([float(z[k].ravel()[0]) for k in sorted(z.files)])
        final = np.array([float(l["delays"].ravel()[0])
                          for l in cu.load_recurrent_params(rd, ckpt)])
        out[seed] = (init, final)
    return out


def plot_common_delay_init_vs_final(stds, exp_dir, seeds, ckpt, out_dir):
    """Per-layer scalar delay at init vs end, per seed + across-seed mean, vs std."""
    per_std = {std: _common_scalar_delays(std, exp_dir, seeds, ckpt) for std in stds}
    n_layers = max((len(next(iter(d.values()))[0]) for d in per_std.values() if d),
                   default=0)
    if n_layers == 0:
        print("[warn] no common-delay runs found, skipping init-vs-final plot")
        return
    rows = []
    with plt.rc_context(style.RC_LARGE_TEXT):
        _plot_init_vs_final_panel(per_std, stds, n_layers, rows, out_dir)
    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(out_dir, "common_delay_init_vs_final.csv"), index=False)


def _plot_init_vs_final_panel(per_std, stds, n_layers, rows, out_dir):
    """One column per layer, each a delay panel over a displacement panel."""
    fig, axes = plt.subplots(2, n_layers, sharex=True, sharey="row", squeeze=False,
                             figsize=(FIGSIZE_LAYER_PANEL[0] * n_layers,
                                      FIGSIZE_LAYER_PANEL[1] + DELTA_ROW_HEIGHT),
                             gridspec_kw={"height_ratios":
                                          [FIGSIZE_LAYER_PANEL[1], DELTA_ROW_HEIGHT]})
    for layer in range(n_layers):
        ax, ax_d = axes[0, layer], axes[1, layer]
        im, fm, dm, xs = [], [], [], []  # per-std across-seed means (init / final / Δ)
        for std in sorted(stds):
            seeds_d = per_std[std]
            if not seeds_d:
                continue
            inits = np.array([v[0][layer] for v in seeds_d.values()])
            finals = np.array([v[1][layer] for v in seeds_d.values()])
            jit = (np.arange(len(inits)) - (len(inits) - 1) / 2) * 0.25
            first = std == sorted(stds)[0]
            # The connector is what makes a sub-step move visible at all on this axis:
            # the two markers overlap, the stick between them does not.
            ax.vlines(np.full_like(inits, std) + jit, inits, finals,
                      color=style.INK_MUTED, linewidth=1.2, zorder=1)
            # Init drawn over final: the two coincide to within a marker, and an open
            # ring on top of a filled dot keeps both readable (a filled dot on top of a
            # ring hides it).
            ax.scatter(np.full_like(inits, std) + jit, inits, s=32,
                       facecolors="none", edgecolors=INIT_COLOR, linewidths=1.4,
                       zorder=4, label="init (seeds)" if first else None)
            ax.scatter(np.full_like(finals, std) + jit, finals, s=32, color=FINAL_COLOR,
                       zorder=3, label="final (seeds)" if first else None)
            ax_d.scatter(np.full_like(finals, std) + jit, finals - inits, s=32,
                         color=FINAL_COLOR, zorder=3)
            xs.append(std)
            im.append(inits.mean())
            fm.append(finals.mean())
            dm.append((finals - inits).mean())
            for seed, (iv, fv) in seeds_d.items():
                rows.append({"layer": layer, "std": std, "seed": seed,
                             "delay_init": float(iv[layer]),
                             "delay_final": float(fv[layer]),
                             "delta": float(fv[layer] - iv[layer])})
        ax.plot(xs, im, color=INIT_COLOR, ls="--", lw=2.0, label="init (mean)")
        ax.plot(xs, fm, color=FINAL_COLOR, lw=2.0, label="final (mean)")
        # Zero = the frozen-delay control this family has no trained twin of: a point
        # on the line is a run whose shared delay came out exactly as drawn.
        ax_d.axhline(0.0, color=INIT_COLOR, ls="--", lw=2.0, zorder=0)
        ax_d.plot(xs, dm, color=FINAL_COLOR, lw=2.0)
        ax.set_xticks(sorted(stds))
        ax.set_title(f"layer{layer}")
        ax_d.set_xlabel(har.STD_AXIS.label)
        ax.grid(alpha=0.3)
        ax_d.grid(alpha=0.3)
    # sharey: only the left column is labeled and ticked.
    axes[0, 0].set_ylabel("shared delay (steps)")
    axes[1, 0].set_ylabel("final − init\n(steps)")
    # One key for the figure, above the columns: the entries mean the same thing in
    # every panel, and at panel width an in-axes legend covers the points.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0),
               bbox_transform=fig.transFigure, ncol=4, frameon=False)
    fig.tight_layout()
    # Inside the caller's rc_context: savefig re-measures the text (bbox="tight").
    cu.save_fig(fig, os.path.join(out_dir, "common_delay_init_vs_final.svg"))



# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-dir", default=".",
                    help="Root the run paths are resolved against (default '.').")
    ap.add_argument("--values", type=int, nargs="+", default=None,
                    help="Restrict to these delay_std_init values. Default: all in config.STDS.")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Restrict to these seeds. Default: all in config.SEEDS.")
    ap.add_argument("--ckpt", default="best.pth",
                    help="Checkpoint to read the final shared delay from.")
    ap.add_argument("--out-dir", default=None,
                    help="Default: exp/HAR/make_figures/common")
    args = ap.parse_args()

    stds = args.values or har.STDS
    seeds = args.seeds
    out_dir = args.out_dir or os.path.join(har.OUT_ROOT, "common")
    os.makedirs(out_dir, exist_ok=True)
    print(f"Common-delay analysis → {out_dir}\n  stds={stds}  seeds={seeds or har.SEEDS}")

    plot_acc_vs_std(stds, args.exp_dir, seeds, out_dir)
    plot_common_delay_init_vs_final(stds, args.exp_dir, seeds, args.ckpt, out_dir)


if __name__ == "__main__":
    main()
