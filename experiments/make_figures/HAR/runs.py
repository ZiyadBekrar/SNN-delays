"""HAR-specific glue for the shared analysis code.

HAR runs are a sweep, one run per (family, swept value, seed)::

    trained_models/HAR/<sweep_root>/<ModelClass>/<value>/seed<N>/

Four families are compared, axonal and synaptic, each learned and fixed. HAR reuses
SSC's ``delrec.networks`` classes and rebuilds models with ``configs.perf_HAR.Config``.
Test accuracy is read straight from each run's ``final_test.json``, which the HAR
trainer writes, so no rebuild is needed for accuracy.
"""

import os
import json
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from common import paths
from common.utils import (AnalysisSpec, load_config_json, load_recurrent_params,
                          load_model as _load_model)


# --------------------------------------------------------------------------- #
# Hardcoded run set: model key -> (sweep_root, delrec.networks class name)
# --------------------------------------------------------------------------- #
_SW = paths.runs("HAR/std_init_sweep")
MODEL_INFO = {
    "ax_learned":  (_SW, "SNN_recurrent_delays"),
    "ax_fixed":    (_SW, "SNN_fixed_recurrent_delays"),
    "syn_learned": (_SW, "SNN_synaptic_recurrent_delays"),
    "syn_fixed":   (_SW, "SNN_fixed_synaptic_recurrent_delays"),
}

MODEL_PREFIXES = {k: v[1] for k, v in MODEL_INFO.items()}

# HAR varies both delay type and condition: color = type (axonal/synaptic),
# line style / hatch = condition (learned solid, fixed dashed/hatched). See
# common.style.
from common import style as _style
SPEC = AnalysisSpec(
    labels={
        "ax_learned": "Axonal learned", "ax_fixed": "Axonal fixed",
        "syn_learned": "Synaptic learned", "syn_fixed": "Synaptic fixed",
    },
    colors={k: _style.TYPE_COLORS[_style.type_of(k)] for k in MODEL_PREFIXES},
    prefixes=MODEL_PREFIXES,
    linestyles={k: _style.CONDITION_LS[_style.condition_of(k)] for k in MODEL_PREFIXES},
    hatches={k: _style.CONDITION_HATCH[_style.condition_of(k)] for k in MODEL_PREFIXES},
)

DATASET = "HAR"

# delay_std_init values swept + seeds per (family, std).
STDS = [3, 8, 13, 18, 23]
SEEDS = [0, 1, 2]

# --------------------------------------------------------------------------- #
# Common-delay family (SNN_common_recurrent_delays): a SINGLE shared scalar delay
# per layer (common_recdel, recurrent_delays shape (1,)), swept over the same
# delay_std_init grid as the four per-neuron/per-synapse families. It is kept OUT
# of MODEL_INFO / STD_AXIS on purpose. Its scalar delay makes the per-neuron
# distribution / weight-delay-scatter analyses degenerate, and the four-family
# learned-vs-fixed comparisons must stay unchanged. Analyze_commondelay.py is its
# sole consumer (accuracy-vs-others + shared-delay evolution).
# --------------------------------------------------------------------------- #
COMMON_KEY = "common"
COMMON_CLASS = "SNN_common_recurrent_delays"

OUT_ROOT = paths.figures("HAR")


def make_config():
    from configs.perf_HAR import Config
    return Config()


def load_model(model_class_name, ckpt_path, device, **kw):
    """HAR ``load_model``: binds a fresh ``configs.perf_HAR.Config``."""
    cfg = kw.pop("config", None) or make_config()
    return _load_model(model_class_name, ckpt_path, device, cfg, **kw)


# --------------------------------------------------------------------------- #
# Run discovery
# --------------------------------------------------------------------------- #
def resolve_run_dirs(mkey, std, exp_dir=".", seeds=None):
    """Return {seed: abs_run_dir} for a (model family, std). Paths are relative to
    the repo root. ``exp_dir`` defaults to '.'."""
    sweep_root, cls = MODEL_INFO[mkey]
    base = sweep_root if os.path.isabs(sweep_root) else os.path.join(exp_dir, sweep_root)
    out = {}
    for seed in (seeds if seeds is not None else SEEDS):
        path = os.path.join(base, cls, f"std{std}", f"seed{seed}")
        if not os.path.isdir(path):
            continue
        out[seed] = path
    return out


def resolve_common_run_dirs(std, exp_dir=".", seeds=None):
    """Return {seed: abs_run_dir} for the common-delay family at a given std.
    Mirrors ``resolve_run_dirs`` but for ``SNN_common_recurrent_delays`` (which is
    deliberately absent from MODEL_INFO). Same ``std_init_sweep`` layout."""
    base = _SW if os.path.isabs(_SW) else os.path.join(exp_dir, _SW)
    out = {}
    for seed in (seeds if seeds is not None else SEEDS):
        path = os.path.join(base, COMMON_CLASS, f"std{std}", f"seed{seed}")
        if not os.path.isdir(path):
            continue
        out[seed] = path
    return out


# --------------------------------------------------------------------------- #
# Weight-decay-on-delays sweep (exp/HAR/weight_decay_sweep), an alternative
# route to smaller delays: fixed delay_std_init=8, varying L2 (AdamW) weight
# decay on the delay parameters only. Same two learned families as the std
# sweep. There are NO fixed families in this sweep, so the wd analyses are
# 2-family and skip the fixed-vs-learned comparisons.
# --------------------------------------------------------------------------- #
WD_SWEEP_ROOT = paths.runs("HAR/weight_decay_sweep")
WD_MODEL_KEYS = ["ax_learned", "syn_learned"]  # the only families present
WD_VALUES = [0.0, 0.001, 0.003, 0.01, 0.03, 0.04, 0.05, 0.06, 0.07, 0.1]


def resolve_wd_run_dirs(mkey, wd, exp_dir=".", seeds=None):
    """Return {seed: abs_run_dir} for a (learned family, weight-decay value) in the
    weight-decay sweep. Directories are ``<root>/<cls>/wd<g>/seed<N>`` (wd0,
    wd0.001, ...). Mirrors ``resolve_run_dirs`` but for the wd axis."""
    cls = MODEL_PREFIXES[mkey]
    base = WD_SWEEP_ROOT if os.path.isabs(WD_SWEEP_ROOT) else os.path.join(exp_dir, WD_SWEEP_ROOT)
    out = {}
    for seed in (seeds if seeds is not None else SEEDS):
        path = os.path.join(base, cls, f"wd{wd:g}", f"seed{seed}")
        if not os.path.isdir(path):
            continue
        out[seed] = path
    return out


def read_test_accuracy(run_dir):
    """Final test accuracy from the run's final_test.json (None if missing)."""
    path = os.path.join(run_dir, "final_test.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return float(json.load(f).get("acc"))


def delay_spread(delays_per_layer):
    """A run's delay spread: the mean over recurrent layers of the per-layer delay
    std, in timesteps. Averaging the per-layer stds (rather than pooling every
    layer's delays into one sample) keeps a shift between the layers' delay means
    out of the spread. Shape-agnostic: axonal (N,) and synaptic (N, N) alike."""
    return float(np.mean([np.std(d) for d in delays_per_layer]))


def delay_mean(delays_per_layer):
    """A run's mean delay, in timesteps: the mean over recurrent layers of the
    per-layer mean, where the delay distribution sits, as opposed to how wide it is
    (``delay_spread``). Averaged per layer for the same reason as ``delay_spread``,
    so the two are reductions of the same per-layer quantities."""
    return float(np.mean([np.mean(d) for d in delays_per_layer]))


def delay_max(delays_per_layer):
    """A run's longest delay, in timesteps: the max over recurrent layers. This is the
    ring-buffer depth the model needs, on-chip memory / worst-case reaction latency,
    the same quantity ``analyze_weightdecay`` calls the buffer depth. Being a max it is
    pooling-invariant, unlike the mean/std."""
    return float(np.max([np.max(d) for d in delays_per_layer]))


# The per-run reductions of the trained delays that a sweep's x axis can be
# re-expressed in. Keys name the stat. See final_delay_stat.
DELAY_STATS = {"std": delay_spread, "mean": delay_mean, "max": delay_max}

_FINAL_DELAY_CACHE = {}


def final_delay_stat(mkey, value, axis, stat="std", exp_dir=".", seeds=None,
                     ckpt="best.pth"):
    """Mean over seeds of a trained-delay statistic (``DELAY_STATS[stat]`` applied to
    ``ckpt``'s delays) for one (family, swept value) cell, what the runs of that cell
    actually ended with, as opposed to the ``delay_std_init`` knob they started from.
    """
    key = (axis.key, mkey, value, stat, exp_dir, ckpt, tuple(seeds) if seeds else None)
    if key not in _FINAL_DELAY_CACHE:
        reduce = DELAY_STATS[stat]
        vals = []
        for rd in axis.resolve(mkey, value, exp_dir, seeds).values():
            if not os.path.exists(os.path.join(rd, ckpt)):
                continue
            vals.append(reduce([l["delays"] for l in load_recurrent_params(rd, ckpt)]))
        _FINAL_DELAY_CACHE[key] = float(np.mean(vals)) if vals else None
    return _FINAL_DELAY_CACHE[key]


# --------------------------------------------------------------------------- #
# Test-set model rebuild (for the jitter sweep)
# --------------------------------------------------------------------------- #
_TEST_LOADER_CACHE = {}


def get_test_loader(config):
    """Build (and cache) the HAR test loader (returned as valid==test by the
    dataset). Independent of seed/model, so build once and reuse."""
    from delrec.datasets import load_dataset
    key = config.datasets_path
    if key not in _TEST_LOADER_CACHE:
        _, _, test_loader = load_dataset(config)
        _TEST_LOADER_CACHE[key] = test_loader
    return _TEST_LOADER_CACHE[key]


def build_test_model(mkey, run_dir, ckpt_name, forward_version=None):
    """Rebuild a HAR model in the test regime (σ=0, rounded delays) ready for inference."""
    import torch as _torch
    from delrec.utils import seed_everything


    cfg_json = load_config_json(run_dir)
    config = make_config()
    if cfg_json.get("seed") is not None:
        config.seed = cfg_json["seed"]
    if cfg_json.get("delay_std_init") is not None:
        config.delay_std_init = cfg_json["delay_std_init"]

    seed_everything(seed=config.seed, is_cuda=True)
    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")

    model, _cfg, _state = load_model(
        MODEL_PREFIXES[mkey], os.path.join(run_dir, ckpt_name), device,
        config=config, sigma_zero=True, round_positions=True,
        strict=True, verbose=False, forward_version=forward_version,
    )
    return model, config, device


# --------------------------------------------------------------------------- #
# Sweep axis. The single abstraction over "which HAR sweep are we analyzing?".
#
# The HAR runs come in two grids that share everything but the swept variable:
#   * the delay-std-init sweep (``std_init_sweep``): 4 families (axonal & synaptic,
#     each learned & fixed), x = delay_std_init.
#   * the weight-decay-on-delays sweep (``weight_decay_sweep``): the 2 learned
#     families only (no fixed runs), x = L2 weight decay on the delays.
#
# Every analysis is identical apart from (a) which values / families / resolver it
# walks and (b) whether a learned-vs-fixed comparison exists. A ``SweepAxis`` bundles
# exactly those differences so one script handles both via ``--sweep {std,wd}``. The
# fixed-vs-learned comparison figures are emitted iff ``comparisons`` is non-empty
# (so the wd axis, lacking fixed runs, cleanly yields only per-family figures).
#
# Outputs nest by sweep: ``figures/HAR/<out_prefix>/<analysis>/`` (see
# ``out_dir``), keeping the std and wd figure trees side by side.
# --------------------------------------------------------------------------- #
@dataclass
class SweepAxis:
    key: str                    # "std" | "wd"
    col: str                    # DataFrame column holding the swept value
    label: str                  # x-axis label on figures
    values: list                # default swept values
    families: list              # model keys present in this sweep
    comparisons: dict           # {modality: (fixed_key, learned_key)}, empty if no fixed runs
    resolve: Callable           # (mkey, value, exp_dir, seeds) -> {seed: run_dir}
    out_prefix: str             # analysis output subtree ("std" | "wd")
    value_fmt: str = "g"        # format spec for a value in dir names / labels

    def value_dir(self, v):
        """Per-value output subfolder name, e.g. 'std8' / 'wd0.01'."""
        return f"{self.key}{format(v, self.value_fmt)}"

    def pretty(self, v):
        return format(v, self.value_fmt)

    def out_dir(self, analysis):
        """Output dir for an analysis under this axis, e.g.
        'figures/HAR/std/jitter' or 'figures/HAR/wd/HAR_recdel'."""
        return os.path.join(OUT_ROOT, self.out_prefix, analysis)


STD_AXIS = SweepAxis(
    # "s_init", not "delay std init": delays are drawn as |N(0, delay_std_init)|, whose
    # std is 0.6028·s, labeling the knob a std invites reading the fixed families'
    # final-delay-std curve as a broken identity line (see plot_final_acc_compare).
    # The paper's symbol for it is s, not σ (which is the annealed mask width).
    key="std", col="std", label=r"$s_\mathrm{init}$", values=STDS,
    families=list(MODEL_PREFIXES),
    comparisons={"axonal": ("ax_fixed", "ax_learned"),
                 "synaptic": ("syn_fixed", "syn_learned")},
    resolve=resolve_run_dirs, out_prefix="std", value_fmt="d")

WD_AXIS = SweepAxis(
    key="wd", col="wd", label="weight decay on delays", values=WD_VALUES,
    families=list(WD_MODEL_KEYS), comparisons={},
    resolve=resolve_wd_run_dirs, out_prefix="wd", value_fmt="g")

AXES = {"std": STD_AXIS, "wd": WD_AXIS}


def add_sweep_arg(ap):
    """Add the shared ``--sweep`` / ``--values`` options to an argparse parser."""
    ap.add_argument("--sweep", choices=list(AXES), default="std",
                    help="Which HAR sweep axis to analyze (default: std).")
    ap.add_argument("--values", type=float, nargs="+", default=None,
                    help="Restrict to these swept values (delay_std_init's or wd's). "
                         "Default: all in the axis.")


def get_axis(args):
    """Resolve the ``SweepAxis`` selected by ``--sweep``, applying a ``--values``
    override (coerced to int for the std axis) if given."""
    axis = AXES[args.sweep]
    vals = getattr(args, "values", None)
    if vals is not None:
        vals = [int(v) for v in vals] if axis.key == "std" else list(vals)
        axis = replace(axis, values=vals)
    return axis
