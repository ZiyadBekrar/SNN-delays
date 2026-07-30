"""AL-specific glue for the shared analysis code.

AL runs are a flat 4-family by N-seed grid with no swept variable, so no ``SweepAxis``
(contrast HAR)::

    trained_models/AL/<ModelClass>/perf_triton/*seed<N>_*/

The recorded runs predate ``final_test.json``, so :func:`evaluate_test_accuracy`
recomputes accuracy the way the entry point did (best checkpoint, sigma=0) and
:func:`get_test_accuracy` caches it into the run. Only the first pass needs a GPU.

Both an unrounded and a rounded accuracy are measured, because AL trains with
``round_pos_each_epoch = False`` and its learned delays stay fractional. The eval
pins its own rounding regime rather than reading that training-time knob, which
otherwise made the reported number change whenever the config was edited.
"""

import os
import glob
import json

# Importing common.utils inserts the repo root on sys.path so ``from src...`` /
# ``from configs...`` resolve when run as ``python experiments/make_figures/AL/<script>.py``.
from common import paths
from common.utils import (
    AnalysisSpec, kernel_for, load_config_json, load_model as _load_model,
)
from common import style as _style


# Recurrent forward kernel the models are rebuilt with for the test-set eval, per the
# repo convention: axonal → 'triton_exact' (the exact Triton scan), synaptic →
# 'eventdriven' (its exact counterpart. Synaptic has no triton_exact path).
# ``common.utils.kernel_for`` applies that mapping. All kernels are numerically
# equivalent, so this is a speed choice only.
FORWARD_VERSION = "triton_exact"


# --------------------------------------------------------------------------- #
# Hardcoded run set: model key -> delrec.networks class name (== the exp/AL subdir)
# --------------------------------------------------------------------------- #
RUN_ROOT = paths.runs("AL")
RUN_SUBDIR = "perf_triton"  # the per-family folder the AL runs live in

MODEL_PREFIXES = {
    "ax_learned":  "SNN_recurrent_delays",
    "ax_fixed":    "SNN_fixed_recurrent_delays",
    "syn_learned": "SNN_synaptic_recurrent_delays",
    "syn_fixed":   "SNN_fixed_synaptic_recurrent_delays",
}

# AL varies both delay type and condition (as HAR/SSC do): color = type
# (axonal/synaptic), line style / hatch = condition (learned solid, fixed
# dashed/hatched). See common.style.
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

DATASET = "AL"

SEEDS = [0, 1, 2, 3, 4]

OUT_ROOT = paths.figures("AL")


def make_config():
    from configs.perf_AL import Config
    return Config()


def load_model(model_class_name, ckpt_path, device, **kw):
    """AL ``load_model``: binds a fresh ``configs.perf_AL.Config``."""
    cfg = kw.pop("config", None) or make_config()
    return _load_model(model_class_name, ckpt_path, device, cfg, **kw)


# --------------------------------------------------------------------------- #
# Run discovery
# --------------------------------------------------------------------------- #
def resolve_run_dirs(mkey, exp_dir=".", seeds=None):
    """Return {seed: run_dir} for a model family. Run folders are named
    ``<anything>seed<N>_<timestamp>`` (the kept ones carry a ``TOKEEP_`` prefix). The latest timestamp wins when a seed has several."""
    base = os.path.join(exp_dir, RUN_ROOT, MODEL_PREFIXES[mkey], RUN_SUBDIR)
    out = {}
    for seed in (seeds if seeds is not None else SEEDS):
        matches = sorted(glob.glob(os.path.join(base, f"*seed{seed}_*")))
        if not matches:
            continue
        out[seed] = matches[-1]
    return out


def read_test_accuracy(run_dir, key="acc"):
    """Final test accuracy from the run's final_test.json (None if not computed yet).
    ``key`` selects the regime: 'acc' (delays as trained) or 'acc_rounded'.
    Mirrors ``HAR.config.read_test_accuracy``."""
    path = os.path.join(run_dir, "final_test.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        val = json.load(f).get(key)
    return None if val is None else float(val)


# --------------------------------------------------------------------------- #
# Test-set evaluation (rebuilds the model and runs a real forward pass)
# --------------------------------------------------------------------------- #
_TEST_LOADER_CACHE = {}


def get_test_loader(config):
    """Build (and cache) the AL test loader. AL has no separate validation set, so
    load_dataset returns the test loader as valid too. Shuffle=False, so it is
    model- and seed-independent, build once, reuse."""
    from delrec.datasets import load_dataset
    key = config.datasets_path
    if key not in _TEST_LOADER_CACHE:
        _, _, test_loader = load_dataset(config)
        _TEST_LOADER_CACHE[key] = test_loader
    return _TEST_LOADER_CACHE[key]


def evaluate_test_accuracy(mkey, run_dir, ckpt_name="best.pth",
                           forward_version=FORWARD_VERSION):
    """Rebuild the model from ``configs/perf_AL.py``, load <ckpt_name>, and run the AL
    test-set forward pass, the number ``experiments/train.py --dataset al`` prints as final_test/acc
    but never writes to disk. Matches that script's final-test regime: σ=0 (sharp delays,
    independent of the training anneal).
    """
    import torch
    from delrec.training.al import test
    from delrec.utils import seed_everything


    config = make_config()
    cfg_json = load_config_json(run_dir)
    if cfg_json.get("seed") is not None:
        config.seed = cfg_json["seed"]
    # Pin the rounding regime here instead of inheriting the training-time knob:
    # trainer.test() rounds the delays in place iff round_pos_each_epoch, which would
    # make this eval depend on whatever configs/perf_AL.py currently says. Rounding
    # is driven explicitly below.
    config.round_pos_each_epoch = False

    seed_everything(seed=config.seed, is_cuda=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = get_test_loader(config)
    kernel = kernel_for(mkey, forward_version, device)

    model, _cfg, state = load_model(
        MODEL_PREFIXES[mkey], os.path.join(run_dir, ckpt_name), device,
        config=config, sigma_zero=True, round_positions=False,
        strict=True, verbose=False, forward_version=kernel,
    )

    best_epoch = int(state["epoch"]) if isinstance(state, dict) and "epoch" in state else 0
    best_val = float(state["acc"]) if isinstance(state, dict) and "acc" in state else None
    acc, loss = test(test_loader, model, best_epoch, device, config)

    # Same model, delays rounded to integers (in place, the run is done with).
    with torch.no_grad():
        model.round_pos()
    acc_r, loss_r = test(test_loader, model, best_epoch, device, config)

    return (float(acc), float(loss), float(acc_r), float(loss_r),
            best_val, best_epoch, kernel)


def get_test_accuracy(mkey, run_dir, ckpt_name="best.pth", recompute=False,
                      forward_version=FORWARD_VERSION):
    """Cached final test accuracy: read the run's final_test.json, else evaluate it (rebuild +
    forward pass) and write it there.
    """
    if not recompute:
        acc = read_test_accuracy(run_dir)
        if acc is not None and read_test_accuracy(run_dir, "acc_rounded") is not None:
            return acc, True

    acc, loss, acc_r, loss_r, best_val, best_epoch, kernel = evaluate_test_accuracy(
        mkey, run_dir, ckpt_name, forward_version)
    with open(os.path.join(run_dir, "final_test.json"), "w") as f:
        json.dump({"acc": acc, "loss": loss,
                   "acc_rounded": acc_r, "loss_rounded": loss_r,
                   "best_val": best_val, "best_epoch": best_epoch,
                   "ckpt": ckpt_name, "kernel": kernel}, f, indent=2)
    return acc, False
