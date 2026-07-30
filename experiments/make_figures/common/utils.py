"""Dataset-agnostic infrastructure for the DelRec analyses.

Device selection, SVG figure saving, lightweight checkpoint / results readers
(no spikingjelly / DCLS needed) and a generic model-rebuilder. The per-dataset
packages pass their own ``Config`` instance into :func:`load_model`, so nothing
here is tied to SSC or HAR.

Design notes
------------
* This module lives under ``experiments/make_figures/common/``. The scripts are run from the
  repo root (``python experiments/make_figures/<dataset>/<script>.py``). Importing it inserts
  the repo root on ``sys.path`` so ``from src...`` / ``from configs...`` resolve
  regardless of the script's own directory.
* Heavy imports (spikingjelly / DCLS via ``delrec.networks``) are done lazily
  inside :func:`load_model` so a pure read-the-checkpoints analysis (no forward
  pass) keeps working anywhere torch is installed.
* Every figure is written as SVG (see :func:`save_fig`): analysis figures
  are vector by project convention (CLAUDE.md).
"""

import os
import sys

# Make the repo root importable when run as ``python experiments/make_figures/<dataset>/<script>.py``
# (the script's own dir, not the repo root, is what Python puts on sys.path[0]).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import glob
import json
import re
from dataclasses import dataclass, field

import torch
import matplotlib.pyplot as plt

# pandas is imported lazily (only the val_res.csv readers need it) so the
# gradmap path (which never touches pandas) stays importable even where a
# broken pandas/libstdc++ would otherwise fail at import time.


# --------------------------------------------------------------------------- #
# Model spec: pretty names / colors a dataset attaches to its model keys.
# --------------------------------------------------------------------------- #
@dataclass
class AnalysisSpec:
    """Per-dataset presentation of the model families being compared."""
    labels: dict = field(default_factory=dict)
    colors: dict = field(default_factory=dict)
    prefixes: dict = field(default_factory=dict)
    linestyles: dict = field(default_factory=dict)
    hatches: dict = field(default_factory=dict)

    def label(self, mkey):
        return self.labels.get(mkey, mkey)

    def color(self, mkey):
        return self.colors.get(mkey)

    def linestyle(self, mkey):
        return self.linestyles.get(mkey, "-")

    def hatch(self, mkey):
        return self.hatches.get(mkey, None)


# --------------------------------------------------------------------------- #
# Device / figures
# --------------------------------------------------------------------------- #
def get_device(verbose=True):
    """Return the best available torch device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        if verbose:
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        torch.set_default_dtype(torch.float32)
        if verbose:
            print("Using Apple Silicon GPU (MPS)")
    else:
        dev = torch.device("cpu")
        if verbose:
            print("GPU not available, using CPU")
    return dev


def iso_crossing(x, y, target, interpolate=True):
    """The smallest ``x`` at which the curve ``y(x)`` reaches ``target``. Budget sweeps report
    a ratio of x-values read at a matched y ("N× fewer spikes", "N× smaller delay
    budget").
    """
    import numpy as _np
    x, y = _np.asarray(x, dtype=float), _np.asarray(y, dtype=float)
    order = _np.argsort(x)
    x, y = x[order], y[order]
    hit = _np.nonzero(y >= target)[0]
    if hit.size == 0:
        return None
    i = int(hit[0])
    if i == 0 or not interpolate:
        return float(x[i])
    lo, hi = x[i - 1], x[i]
    # y[i-1] < target <= y[i], so the denominator is strictly positive.
    t = (target - y[i - 1]) / (y[i] - y[i - 1])
    return float(lo + t * (hi - lo))


def save_fig(fig, path, dpi=200):
    """Save ``fig`` to ``path`` and close it. Figures default to SVG: any
    path without a ``.svg``/``.pdf``/``.png`` suffix is coerced to ``.svg`` so
    every analysis figure is vector by default (CLAUDE.md convention)."""
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix not in {".svg", ".pdf", ".png"}:
        path = path.with_suffix(".svg")
        suffix = ".svg"
    if suffix == ".svg":
        fig.savefig(path, format="svg", bbox_inches="tight")
    elif suffix == ".pdf":
        fig.savefig(path, format="pdf", bbox_inches="tight")
    else:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(path)
    return path


# --------------------------------------------------------------------------- #
# Run discovery (glob fallback. The per-dataset packages own the hardcoded set)
# --------------------------------------------------------------------------- #
def find_run_dirs(exp_dir, prefix, seeds):
    """Return {seed: run_dir} for the most recent run matching each seed."""
    out = {}
    for seed in seeds:
        pattern = os.path.join(exp_dir, f"{prefix}_seed{seed}_*")
        matches = sorted(glob.glob(pattern))
        matches = [
            m for m in matches
            if re.match(rf"^{re.escape(prefix)}_seed{seed}_[0-9]", os.path.basename(m))
        ]
        if not matches:
            continue
        out[seed] = matches[-1]  # latest timestamp wins
    return out


# --------------------------------------------------------------------------- #
# Lightweight checkpoint / results readers (no spikingjelly / DCLS needed)
# --------------------------------------------------------------------------- #
def load_config_json(run_dir):
    """Return the saved config.json as a dict (empty dict if missing)."""
    path = os.path.join(run_dir, "config.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_val_accuracy(run_dir):
    """Per-epoch validation accuracy curve (row 0 of val_res.csv)."""
    import pandas as pd
    df = pd.read_csv(os.path.join(run_dir, "val_res.csv"), index_col=0)
    return df.loc[0].to_numpy(dtype=float)


def n_epochs_trained(run_dir):
    """Number of epochs run = number of columns in val_res.csv."""
    import pandas as pd
    path = os.path.join(run_dir, "val_res.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, index_col=0).shape[1]


def _load_state_dict(ckpt_path, map_location="cpu"):
    """Return (raw_checkpoint, state_dict). Handles the 'model' / 'net' wrappers
    used by the SSC and HAR trainers respectively."""
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            return ckpt, ckpt["model"]
        if "net" in ckpt:
            return ckpt, ckpt["net"]
    return ckpt, ckpt


def best_acc_from_ckpt(run_dir, ckpt_name):
    ckpt, _ = _load_state_dict(os.path.join(run_dir, ckpt_name))
    if isinstance(ckpt, dict) and "acc" in ckpt:
        return float(ckpt["acc"])
    return None


def arch_fingerprint(run_dir, ckpt_name):
    """Hashable architecture fingerprint: sorted (param_name, shape) of the
    checkpoint state_dict. Identical fingerprints = identical network structure."""
    _, sd = _load_state_dict(os.path.join(run_dir, ckpt_name))
    return tuple(sorted((k, tuple(v.shape)) for k, v in sd.items()))


def load_recurrent_params(run_dir, ckpt_name, offset_steps=0.0):
    """Extract recurrent delays/weights per recurrent layer from a checkpoint. Returns an
    ordered list of dicts:: {"layer_idx": i, "delays": np, "weights": (N_in, N_out) np}
    ``delays`` keeps its native shape: ``(N,)`` for axonal (per-source-neuron) delays and
    ``(N_in, N_out)`` for synaptic (per-synapse) delays.
    """
    ckpt = os.path.join(run_dir, ckpt_name)
    if not os.path.exists(ckpt):
        # The trained delays are recorded as final_delays.npz, so an analysis that
        # needs only delays runs off the measurements and never loads a checkpoint.
        # The ones that also need the recurrent weights get weights=None and must say so.
        snap = os.path.join(run_dir, "final_delays.npz")
        if not os.path.exists(snap):
            raise FileNotFoundError(
                f"neither {ckpt_name} nor final_delays.npz in {run_dir}. "
                f"Fetch the weights: python fetch_checkpoints.py --list")
        import numpy as np
        with np.load(snap) as z:
            keys = sorted(z.files, key=lambda k: int(re.sub(r"\D", "", k) or 0))
            return [{"layer_idx": i, "delays": z[k] + offset_steps, "weights": None}
                    for i, k in enumerate(keys)]
    _, sd = _load_state_dict(ckpt)
    layers = []
    for key in sd:
        m = re.match(r"layers\.(\d+)\.recurrent_delays$", key)
        if not m:
            continue
        idx = int(m.group(1))
        layers.append({
            "layer_idx": idx,
            "delays": sd[key].detach().cpu().numpy() + offset_steps,
            "weights": sd[f"layers.{idx}.recurrent_weights"].detach().cpu().numpy(),
        })
    layers.sort(key=lambda d: d["layer_idx"])
    return layers


# --------------------------------------------------------------------------- #
# Model rebuilding (heavy: pulls spikingjelly / DCLS lazily)
# --------------------------------------------------------------------------- #
def kernel_for(mkey, forward_version, device):
    """The recurrent kernel to actually evaluate a model family with. The repo's convention
    (and the layers' own CUDA defaults, see ``src/delrec/delay_layers.py``
    AXONAL_CUDA_DEFAULT / SYNAPTIC_CUDA_DEFAULT): axonal → 'triton_exact', synaptic →
    'eventdriven', the exact Triton scan has no synaptic implementation, so
    ``synaptic_recdel.forward`` would silently fall through to the slow pure-torch ``v2``
    if handed 'triton_exact'.
    """
    from common import style
    if device.type != "cuda":
        return "v2"
    if forward_version == "triton_exact" and style.type_of(mkey) == "synaptic":
        return "eventdriven"
    return forward_version


def load_model(model_class_name, ckpt_path, device, config, *,
               surrogate_fn=None, store_v_seq=True, sigma_zero=True,
               zero_dcls_sig=False, round_positions=True, strict=False,
               verbose=True, forward_version=None):
    """Rebuild a ``delrec.networks`` model (SSC and HAR share this module), optionally load a
    checkpoint, and put it in the discrete-delay eval regime shared by the analysis
    scripts. ``config`` is a dataset-specific ``Config`` instance (``configs.perf_SSC`` or
    ``configs.perf_HAR``) supplied by the caller.
    """
    import delrec.networks as snn_mod
    from delrec.delay_layers import axonal_recdel

    cfg = config
    cfg.store_v_seq = store_v_seq
    if surrogate_fn is not None:
        cfg.surrogate_function = surrogate_fn
    if model_class_name == "SNN_axonal_feedforward_delays":
        cfg.DCLSversion = "gauss"

    model = getattr(snn_mod, model_class_name)(cfg).to(device)
    model.eval()

    state = None
    if ckpt_path is not None:
        state, sd = _load_state_dict(ckpt_path, map_location=device)
        incompatible = model.load_state_dict(sd, strict=strict)
        if verbose:
            print(f"Loaded {model_class_name} from {ckpt_path}")
            if not strict:
                print(f"  missing_keys:    {incompatible.missing_keys}")
                print(f"  unexpected_keys: {incompatible.unexpected_keys}")

    if sigma_zero:
        # sigma=0 is the eval regime. Also disable use_sig_p so p_spread (inert at
        # sigma=0, mask width = 1) is turned off explicitly and the eventdriven
        # kernel (which requires use_sig_p=False) is usable on learned models too.
        for name, m in model.named_modules():
            if isinstance(m, axonal_recdel):
                m.sigma = 0.0
                m.use_sig_p = False

    if zero_dcls_sig:
        dcls_module = getattr(snn_mod, "dcls_module")
        for name, m in model.named_modules():
            if isinstance(m, dcls_module):
                with torch.no_grad():
                    m.SIG.fill_(0)

    if round_positions and hasattr(model, "round_pos"):
        with torch.no_grad():
            model.round_pos()

    if forward_version is not None:
        for name, m in model.named_modules():
            if isinstance(m, axonal_recdel):
                m.forward_version = forward_version

    return model, cfg, state
