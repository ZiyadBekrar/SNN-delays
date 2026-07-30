"""MG (Mackey-Glass) glue for the shared analysis code.

The MG runs are the (family × mg_tau × horizon × seed) grid written by
``experiments/sweeps/sweep_MG.py`` (or its parallel launcher)::

    exp/MG/mackey_glass_sweep/<family>/tau<t>_H<h>/seed<N>/

Two swept axes (the system delay ``mg_tau`` and the prediction horizon ``H``) rather
than HAR's single one, so no ``SweepAxis``: :func:`discover_axes` simply reads the
grid back off the directory tree, and every analysis is a function of (tau, H).

MG is a regression task, so runs carry NMSE / R² (from each run's ``final_test.json``
and ``predictions.npz``) instead of accuracy. There is no rebuild and no forward pass
anywhere in ``experiments/make_figures/MG``, hence no GPU or dataset needed.

All four families are axonal. What varies is the delay condition. As in SSC (where
type is likewise constant) color is therefore freed to encode the condition, with the
delay-free control carried in gray/dotted as ``analyze_spikepenalty`` does for vanilla.
"""

import glob
import json
import os
import re

import numpy as np

# Importing common.utils inserts the repo root on sys.path so ``from src...`` /
# ``from configs...`` resolve when run as ``python experiments/make_figures/MG/<script>.py``.
from common import paths
from common.utils import AnalysisSpec
from common import style as _style


DATASET = "MG"

SWEEP_ROOT = paths.runs("MG/mackey_glass_sweep")
OUT_ROOT = paths.figures("MG")

SPLITS = ["train", "val", "test"]

# The reference condition every "improvement" figure is measured against.
REFERENCE = "learned_delays"

# family -> delrec.networks class name it was trained with (must match experiments/sweeps/sweep_MG.FAMILIES)
MODEL_PREFIXES = {
    "learned_delays":       "SNN_recurrent_delays",
    "learned_no_annealing": "SNN_recurrent_delays",
    "fixed_delays":         "SNN_fixed_recurrent_delays",
    "no_delays":            "SNN_vanilla_recurrent",
}

FAMILY_ORDER = ["learned_delays", "learned_no_annealing", "fixed_delays", "no_delays"]

SPEC = AnalysisSpec(
    labels={
        "learned_delays":       "Learned delays",
        "learned_no_annealing": "Learned delays (no anneal)",
        "fixed_delays":         "Fixed delays",
        "no_delays":            "No delays",
    },
    colors={
        "learned_delays":       _style.CONDITION_COLORS["learned"],
        "learned_no_annealing": _style.CONDITION_COLORS["learned"],
        "fixed_delays":         _style.CONDITION_COLORS["fixed"],
        "no_delays":            _style.INK_MUTED,
    },
    prefixes=MODEL_PREFIXES,
    linestyles={
        "learned_delays":       "-",
        "learned_no_annealing": "-.",
        "fixed_delays":         "--",
        "no_delays":            ":",
    },
    hatches={
        "learned_delays":       None,
        "learned_no_annealing": "\\\\\\",
        "fixed_delays":         "///",
        "no_delays":            "...",
    },
)


# --------------------------------------------------------------------------- #
# Run discovery
# --------------------------------------------------------------------------- #
_TAG_RE = re.compile(r"^tau([\d.]+)_H(\d+)$")


def axis_tag(mg_tau, horizon):
    """Must match sweep_MG._axis_tag."""
    return f"tau{mg_tau:g}_H{horizon:g}"


def run_dir(family, mg_tau, horizon, seed, sweep_root=SWEEP_ROOT):
    return os.path.join(sweep_root, family, axis_tag(mg_tau, horizon), f"seed{seed}")


def discover_axes(sweep_root=SWEEP_ROOT):
    """Read the sweep grid back off the directory tree."""
    families, taus, horizons, seeds = set(), set(), set(), set()
    for path in glob.glob(os.path.join(sweep_root, "*", "*", "seed*", "final_test.json")):
        seed_dir, tag_dir, fam_dir = (os.path.basename(os.path.dirname(path)),
                                      os.path.basename(os.path.dirname(os.path.dirname(path))),
                                      os.path.basename(os.path.dirname(os.path.dirname(
                                          os.path.dirname(path)))))
        m = _TAG_RE.match(tag_dir)
        if not m:
            continue
        families.add(fam_dir)
        taus.add(float(m.group(1)))
        horizons.add(int(m.group(2)))
        seeds.add(int(seed_dir[len("seed"):]))

    ordered = [f for f in FAMILY_ORDER if f in families] + \
              sorted(f for f in families if f not in FAMILY_ORDER)
    return ordered, sorted(taus), sorted(horizons), sorted(seeds)


# --------------------------------------------------------------------------- #
# Per-run readers: recorded measurements only, no model rebuild, no forward pass
# --------------------------------------------------------------------------- #
def read_metrics(rdir):
    """The run's final_test.json (metrics of the best, σ=0, rounded model), or None."""
    path = os.path.join(rdir, "final_test.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def read_predictions(rdir):
    """{split: {'preds': np, 'tgts': np}} from the run's predictions.npz, or None."""
    path = os.path.join(rdir, "predictions.npz")
    if not os.path.exists(path):
        return None
    npz = np.load(path)
    out = {}
    for split in SPLITS:
        if f"{split}_preds" in npz and f"{split}_tgts" in npz:
            out[split] = {"preds": npz[f"{split}_preds"], "tgts": npz[f"{split}_tgts"]}
    return out or None


def read_delays(rdir, ckpt_name="best.pth"):
    """Per-layer recurrent delays + weights of the tested model."""
    from common.utils import load_recurrent_params
    path = os.path.join(rdir, "final_delays.npz")
    if not os.path.exists(path):
        return None
    npz = np.load(path)
    layers = load_recurrent_params(rdir, ckpt_name)
    if len(layers) != len(npz.files):
        return None
    return [{"delays": npz[f"layer{i}"], "weights": layers[i]["weights"]}
            for i in range(len(layers))]
