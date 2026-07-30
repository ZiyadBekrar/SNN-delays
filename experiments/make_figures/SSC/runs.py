"""SSC-specific glue for the shared analysis code.

Holds the hardcoded SSC run set, the presentation :class:`AnalysisSpec`, run
discovery, run verification, and the SSC test-set evaluation. Everything generic
lives in ``experiments/make_figures/common``.
"""

import glob
import os
import re

# Importing common.utils inserts the repo root on sys.path so ``from src...`` /
# ``from configs...`` resolve when run as ``python experiments/make_figures/SSC/<script>.py``.
from common import paths
from common.utils import (
    AnalysisSpec, find_run_dirs, kernel_for, load_config_json,
    load_model as _load_model,
)

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Hardcoded run set (single source of truth for both SSC scripts)
# --------------------------------------------------------------------------- #

# The delrec.networks class each model key rebuilds to (also the glob prefix for
# auto-discovery when a RUN_DIRS entry is empty).
MODEL_PREFIXES = {
    "ax_learned":  "SNN_recurrent_delays",
    "ax_fixed":    "SNN_fixed_recurrent_delays",
    "syn_learned": "SNN_synaptic_recurrent_delays",
    "syn_fixed":   "SNN_fixed_synaptic_recurrent_delays",
}

# SSC now varies both delay type and condition (as HAR does): color = type
# (axonal/synaptic), line style / hatch = condition (learned solid, fixed
# dashed/hatched). See common.style.
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

DATASET = "SSC"

# Default output root, code lives in experiments/make_figures/, generated figures + CSVs go under figures/.
RUN_ROOT = paths.runs("SSC")
OUT_ROOT = paths.figures("SSC")


def make_config():
    from configs.perf_SSC import Config
    return Config()


def load_model(model_class_name, ckpt_path, device, **kw):
    """SSC ``load_model``: binds a fresh ``configs.perf_SSC.Config``."""
    cfg = kw.pop("config", None) or make_config()
    return _load_model(model_class_name, ckpt_path, device, cfg, **kw)


# --------------------------------------------------------------------------- #
# Run discovery
# --------------------------------------------------------------------------- #
RUN_SUBDIR = "perf"   # the per-family folder the SSC runs live in
SEEDS = [0, 1, 2, 3, 4]


def resolve_run_dirs(mkey, exp_dir=".", seeds=None):
    """Return {seed: run_dir} for a model family."""
    base = os.path.join(exp_dir, RUN_ROOT, MODEL_PREFIXES[mkey], RUN_SUBDIR)
    out = {}
    for seed in (seeds if seeds is not None else SEEDS):
        matches = sorted(glob.glob(os.path.join(base, f"*seed{seed}_*")))
        if matches:
            out[seed] = matches[-1]
    return out


def parse_run_overrides(run_dir):
    """Recover per-run overrides from config.json + the run name."""
    cfg = load_config_json(run_dir)
    base = os.path.basename(run_dir.rstrip("/"))

    seed = cfg.get("seed")
    if seed is None:
        m = re.search(r"_seed(\d+)", base)
        seed = int(m.group(1)) if m else None

    sigma_init = cfg.get("sigma_init")
    if sigma_init is None:
        m = re.search(r"_sig([0-9.]+)", base)
        if m:
            sigma_init = float(m.group(1))

    forward_version = None
    for cand in ("triton_exact", "triton_approx", "triton_ste"):
        if cand in base:
            forward_version = cand
            break
    return seed, sigma_init, forward_version


# --------------------------------------------------------------------------- #
# Run verification: are the seeds trained the same way?
# --------------------------------------------------------------------------- #
def verify_runs(results, ckpt_name, out_dir):
    """Check that every run we are about to aggregate was trained the same way
    (identical architecture fingerprint, config.json apart from per-run keys, and
    epoch count). Prints + saves run_verification.csv. Returns True if consistent."""
    from common.utils import arch_fingerprint, n_epochs_trained

    rows = []
    fingerprints = {}
    cfgs = {}
    for mkey, data in results.items():
        for seed, rd in sorted(data["run_dirs"].items()):
            s, sigma_init, fv = parse_run_overrides(rd)
            fp = arch_fingerprint(rd, ckpt_name)
            fingerprints[(mkey, seed)] = fp
            cfgs[(mkey, seed)] = load_config_json(rd)
            rows.append({
                "model": mkey, "seed": seed,
                "sigma_init": sigma_init,
                "forward_version": fv or "v2 (default)",
                "n_epochs": n_epochs_trained(rd),
                "n_params": sum(int(np.prod(shp)) for _, shp in fp),
                "run_dir": os.path.basename(rd.rstrip("/")),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("[verify] no runs to verify.")
        return False
    df.to_csv(os.path.join(out_dir, "run_verification.csv"), index=False)

    print("\n=== Run verification (are all seeds trained the same way?) ===")
    print(df.to_string(index=False))

    ok = True
    uniq_fp = set(fingerprints.values())
    if len(uniq_fp) == 1:
        print("[verify] OK  - all runs share one architecture fingerprint.")
    else:
        ok = False
        print(f"[verify] WARNING - {len(uniq_fp)} distinct architecture "
              f"fingerprints found; runs are NOT structurally identical.")

    n_ep = df["n_epochs"].dropna().unique()
    if len(n_ep) <= 1:
        print(f"[verify] OK  - all runs trained for {n_ep.tolist()} epochs.")
    else:
        ok = False
        print(f"[verify] WARNING - runs trained for different #epochs: {sorted(n_ep)}")

    ignore = {"seed", "results_dir", "sigma_init"}
    all_keys = set().union(*[set(c) for c in cfgs.values()]) - ignore
    for k in sorted(all_keys):
        vals = {repr(c.get(k)) for c in cfgs.values()}
        if len(vals) > 1:
            ok = False
            print(f"[verify] WARNING - config.json key '{k}' differs across runs: {vals}")

    print("[verify] NOTE  - config.json only stores per-run overrides "
          "(seed/results_dir/sigma_init), the bulk of the hyperparameters live as "
          "class attributes in configs/perf_SSC.py and are not recorded per run, "
          "so they are verified here only via the architecture fingerprint.")
    return ok


# --------------------------------------------------------------------------- #
# Test-set evaluation (rebuilds the model and runs a real forward pass)
# --------------------------------------------------------------------------- #
_TEST_LOADER_CACHE = {}


def _get_test_loader(config, load_dataset):
    """Build (and cache) the SSC test loader. shuffle=False so it is seed- and
    call-independent, build once, reuse across all seeds/models."""
    key = config.datasets_path
    if key not in _TEST_LOADER_CACHE:
        _, _, test_loader = load_dataset(config)
        _TEST_LOADER_CACHE[key] = test_loader
    return _TEST_LOADER_CACHE[key]


def get_test_loader(config):
    """Public wrapper: cached SSC test loader for the given config."""
    from delrec.datasets import load_dataset
    return _get_test_loader(config, load_dataset)


def build_test_model(mkey, run_dir, ckpt_name, forward_version_override=None):
    """Rebuild a model in the exact SSC test regime (σ=0, rounded delays, the forward kernel
    ``experiments/train.py --dataset ssc`` evaluates with) and return it ready for inference.
    """
    import torch as _torch
    from delrec.delay_layers import axonal_recdel
    from delrec.utils import seed_everything


    seed, sigma_init, fv = parse_run_overrides(run_dir)
    if forward_version_override:
        fv = forward_version_override
    # else: fv is a run-name kernel (or None). Leave it unset when None so the
    # model's forward() default picks eventdriven on CUDA / v2 on CPU.

    config = make_config()
    if seed is not None:
        config.seed = seed
    if sigma_init is not None:
        config.sigma_init = sigma_init

    seed_everything(seed=config.seed, is_cuda=True)
    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")

    model, _cfg, _state = load_model(
        MODEL_PREFIXES[mkey], os.path.join(run_dir, ckpt_name), device,
        config=config, sigma_zero=True, round_positions=True,
        strict=True, verbose=False,
    )
    if fv is not None:
        # Map onto the kernel the family can actually run fast: synaptic has no
        # triton_exact path (would silently fall through to the slow pure-torch v2),
        # so kernel_for routes it to eventdriven. Numerically equivalent, speed only.
        fv = kernel_for(mkey, fv, device)
        for mod in model.modules():
            if isinstance(mod, axonal_recdel):
                mod.forward_version = fv
    return model, config, device, (fv or ("eventdriven" if device.type == "cuda" else "v2"))


def evaluate_test_accuracy(mkey, run_dir, ckpt_name, forward_version_override=None):
    """Rebuild the model from configs/perf_SSC.py, load <ckpt_name>, and run the
    SSC test set forward pass, the number ``experiments/train.py --dataset ssc`` prints as final_test/acc
    but never writes to disk. sigma is set to 0 to match ``experiments/train.py --dataset ssc``."""
    import torch as _torch
    from delrec.training.ssc import test
    from delrec.delay_layers import axonal_recdel
    from delrec.datasets import load_dataset
    from delrec.utils import seed_everything


    seed, sigma_init, fv = parse_run_overrides(run_dir)
    if forward_version_override:
        fv = forward_version_override
    # else: fv is a run-name kernel (or None). Leave it unset when None so the
    # model's forward() default picks eventdriven on CUDA / v2 on CPU.

    config = make_config()
    if seed is not None:
        config.seed = seed
    if sigma_init is not None:
        config.sigma_init = sigma_init

    seed_everything(seed=config.seed, is_cuda=True)
    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")

    test_loader = _get_test_loader(config, load_dataset)

    model, _cfg, state = load_model(
        MODEL_PREFIXES[mkey], os.path.join(run_dir, ckpt_name), device,
        config=config, sigma_zero=True, round_positions=False,
        strict=True, verbose=False,
    )
    if fv is not None:
        for mod in model.modules():
            if isinstance(mod, axonal_recdel):
                mod.forward_version = fv

    last_epoch = int(state["epoch"]) if isinstance(state, dict) and "epoch" in state else 0
    test_loader.reset()
    acc, loss = test(test_loader, model, last_epoch, device, config)
    return float(acc), float(loss), (fv or ("eventdriven" if device.type == "cuda" else "v2"))
