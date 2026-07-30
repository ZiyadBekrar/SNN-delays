"""PS-MNIST-specific glue for the shared analysis code.

A flat 4-family by N-seed grid with no swept variable, so no ``SweepAxis``::

    trained_models/PSMNIST/<ModelClass>/perf/*seed<N>_*/

The permutation. Each run draws its own pixel order and saves it as ``perm.pt``.
Evaluating a run under any other permutation is meaningless, so
:func:`evaluate_test_accuracy` loads the run's own file.

The trainer never wrote ``final_test.json``, so accuracy is recomputed the way
``experiments/train.py --dataset psmnist`` reports it (best checkpoint, sigma=0, rounded delays)
and cached into the run. Only the first pass needs a GPU. Unlike AL and HAR, PS-MNIST
has a real held-out test set, so ``best.pth`` is selected on a separate validation
split.
"""

import os
import glob
import json

# Importing common.utils inserts the repo root on sys.path so ``from src...`` /
# ``from configs...`` resolve when run as ``python experiments/make_figures/PSMNIST/<script>.py``.
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
# Hardcoded run set: model key -> delrec.networks class name (== the exp subdir)
# --------------------------------------------------------------------------- #
RUN_ROOT = paths.runs("PSMNIST")
RUN_SUBDIR = "perf"  # the per-family folder the PSMNIST runs live in

MODEL_PREFIXES = {
    "ax_learned":  "SNN_recurrent_delays",
    "ax_fixed":    "SNN_fixed_recurrent_delays",
    "syn_learned": "SNN_synaptic_recurrent_delays",
    "syn_fixed":   "SNN_fixed_synaptic_recurrent_delays",
}

# Color = delay type (axonal/synaptic), line style / hatch = condition (learned
# solid, fixed dashed/hatched). See common.style.
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

DATASET = "PSMNIST"

SEEDS = [0, 1, 2, 3, 4]

OUT_ROOT = paths.figures("PSMNIST")


def make_config():
    from configs.perf_PSMNIST import Config
    return Config()


def load_model(model_class_name, ckpt_path, device, **kw):
    """PSMNIST ``load_model``: binds a fresh ``configs.perf_PSMNIST.Config``."""
    cfg = kw.pop("config", None) or make_config()
    return _load_model(model_class_name, ckpt_path, device, cfg, **kw)


# --------------------------------------------------------------------------- #
# Run discovery
# --------------------------------------------------------------------------- #
def resolve_run_dirs(mkey, exp_dir=".", seeds=None):
    """Return {seed: run_dir} for a model family. Run folders are named
    ``seed<N>_<timestamp>``. The latest timestamp wins when a seed has several."""
    base = os.path.join(exp_dir, RUN_ROOT, MODEL_PREFIXES[mkey], RUN_SUBDIR)
    out = {}
    for seed in (seeds if seeds is not None else SEEDS):
        matches = sorted(glob.glob(os.path.join(base, f"*seed{seed}_*")))
        if not matches:
            continue
        out[seed] = matches[-1]
    return out


def read_test_accuracy(run_dir):
    """Final test accuracy from the run's final_test.json (None if not computed yet).
    Mirrors ``HAR.config.read_test_accuracy``."""
    path = os.path.join(run_dir, "final_test.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return float(json.load(f).get("acc"))


# --------------------------------------------------------------------------- #
# Test-set evaluation (rebuilds the model and runs a real forward pass)
# --------------------------------------------------------------------------- #
_TEST_LOADER_CACHE = {}


def get_test_loader(config):
    """Build (and cache) the PSMNIST test loader (MNIST-test, shuffle=False, and
    unpermuted. The permutation is applied per batch by the trainer, from each
    run's own perm.pt). Model- and seed-independent, so build once and reuse."""
    from delrec.datasets import load_dataset
    key = config.datasets_path
    if key not in _TEST_LOADER_CACHE:
        _, _, test_loader = load_dataset(config)
        _TEST_LOADER_CACHE[key] = test_loader
    return _TEST_LOADER_CACHE[key]


def load_perm(run_dir):
    """The run's own 784-pixel permutation (perm.pt), as saved by ``experiments/train.py --dataset psmnist``."""
    import torch
    path = os.path.join(run_dir, "perm.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing, the run's pixel permutation is required to evaluate it.")
    return torch.load(path, map_location="cpu")


def build_test_model(mkey, run_dir, ckpt_name="best.pth",
                     forward_version=FORWARD_VERSION):
    """Rebuild a model in the exact PSMNIST test regime and return it ready for inference, σ=0
    (sharp single-tap delays), ``use_sig_p`` off, delays rounded, and the recurrent kernel
    :func:`kernel_for` picks for the family. Shares its setup with
    :func:`evaluate_test_accuracy` (which runs the clean test pass and rounds inside the
    trainer's ``test()``, hence ``round_positions=False`` there).
    """
    import torch

    config = make_config()
    cfg_json = load_config_json(run_dir)
    if cfg_json.get("seed") is not None:
        config.seed = cfg_json["seed"]

    from delrec.utils import seed_everything
    seed_everything(seed=config.seed, is_cuda=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    get_test_loader(config)  # sets config.time_window / input_size / output_size
    kernel = kernel_for(mkey, forward_version, device)

    model, _cfg, _state = load_model(
        MODEL_PREFIXES[mkey], os.path.join(run_dir, ckpt_name), device,
        config=config, sigma_zero=True, round_positions=True,
        strict=True, verbose=False, forward_version=kernel,
    )
    return model, config, device, kernel


def evaluate_test_accuracy(mkey, run_dir, ckpt_name="best.pth",
                           forward_version=FORWARD_VERSION):
    """Rebuild the model from ``configs/perf_PSMNIST.py``, load <ckpt_name>, and run the MNIST
    test-set forward pass under the run's own permutation, the number ``experiments/train.py
    --dataset psmnist`` prints as final_test/acc but never writes to disk. Matches that
    script's final-test regime: σ=0 (sharp single-tap delays, independent of the training
    anneal), ``use_sig_p`` off (p_spread is inert at σ=0), and delays rounded to integers
    (the trainer's ``test()`` rounds them itself via round_pos_each_epoch).
    """
    import torch
    from delrec.training.psmnist import test
    from delrec.utils import seed_everything


    config = make_config()
    cfg_json = load_config_json(run_dir)
    if cfg_json.get("seed") is not None:
        config.seed = cfg_json["seed"]

    seed_everything(seed=config.seed, is_cuda=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = get_test_loader(config)  # sets config.time_window/input_dim/output_dim
    perm = load_perm(run_dir)
    kernel = kernel_for(mkey, forward_version, device)

    model, _cfg, state = load_model(
        MODEL_PREFIXES[mkey], os.path.join(run_dir, ckpt_name), device,
        config=config, sigma_zero=True, round_positions=False,  # test() rounds
        strict=True, verbose=False, forward_version=kernel,
    )

    best_epoch = int(state["epoch"]) if isinstance(state, dict) and "epoch" in state else 0
    best_val = float(state["acc"]) if isinstance(state, dict) and "acc" in state else None
    acc, loss = test(test_loader, model, best_epoch, device, config, perm)
    return float(acc), float(loss), best_val, best_epoch, kernel


def get_test_accuracy(mkey, run_dir, ckpt_name="best.pth", recompute=False,
                      forward_version=FORWARD_VERSION):
    """Cached final test accuracy: read the run's final_test.json, else evaluate it
    (rebuild + forward pass) and write it there. Returns (acc, from_cache)."""
    if not recompute:
        acc = read_test_accuracy(run_dir)
        if acc is not None:
            return acc, True

    acc, loss, best_val, best_epoch, kernel = evaluate_test_accuracy(
        mkey, run_dir, ckpt_name, forward_version)
    with open(os.path.join(run_dir, "final_test.json"), "w") as f:
        json.dump({"acc": acc, "loss": loss, "best_val": best_val,
                   "best_epoch": best_epoch, "ckpt": ckpt_name, "kernel": kernel}, f,
                  indent=2)
    return acc, False
