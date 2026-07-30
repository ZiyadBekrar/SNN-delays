"""Sweep over weight decay applied to the delay parameters only.

Produces: compressing learned delays toward zero, and the buffer depth that costs.
Outputs:  trained_models/HAR/weight_decay_sweep/  (axis 'std' -> trained_models/HAR/fixed_compression_sweep/)
Usage:    python experiments/sweeps/sweep_HAR_weightdecay.py

Each run writes the same artifacts as a single training run. The
`parallel_` twin of this script runs one process per run across the
available GPUs, giving each its own Triton cache so concurrent JITs do
not race.
"""

import os
import sys
# Repo-root on sys.path so `configs`/`delrec` resolve when run as `python experiments/sweeps/<script>.py`.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src")]

import torch
from datetime import datetime
from pathlib import Path
import pandas as pd
import json
import os
import numpy as np

from configs.perf_HAR import Config

from delrec.training.har import *
from delrec.networks import *
import delrec.networks as snn_module
from delrec.datasets import load_dataset


# ── Recurrent kernel choice (hardcode here. None = library default) ────────────
# Library default on GPU: axonal -> 'triton_exact', synaptic -> 'eventdriven'
# (both exact vs v2). Set to force a kernel, e.g. AXONAL_FORWARD_VERSION =
# 'eventdriven'. Env REC_FWD / SYN_REC_FWD take precedence if set.
AXONAL_FORWARD_VERSION = os.environ.get("REC_FWD") or None
SYN_FORWARD_VERSION    = os.environ.get("SYN_REC_FWD") or None

# Weight-decay-on-delays sweep for HAR, built on sweep_HAR_delaystd.py.
# For every (learned-delay model, weight_decay_positions, seed) triple it trains the
# HAR pipeline at a fixed delay_std_init (default 8, matching the existing std8
# runs) while applying L2 (decoupled AdamW) weight decay to the delay parameters
# only (the "positions" optimizer group == recurrent_delays. See src/HAR/trainer.py
# init_optim_sche). Weight_decay on the weights is left at config.weight_decay.
#
# The recurrent kernel is left at its DEFAULT: the spike-sparse 'eventdriven' Triton
# scan on CUDA (HAR has decay_input=False, so it qualifies) and pure-torch 'v2' on
# CPU, forward_version is intentionally NOT overridden here.
#
# Per run it persists:
#   config.json, train_res.csv, val_res.csv, best.pth, last.pth   (as perf_HAR)
#   init_delays.npz, per-layer recurrent_delays right after init
#   final_delays.npz, per-layer recurrent_delays of the tested (best, rounded) model
#   delay_variation.csv, per-epoch mean-abs delay variation per recdel layer
#   final_test.json, final test acc/loss + best val/epoch + weight_decay
# A manifest.json at the sweep root indexes all runs for the plotting/analysis script.
#
# ``run_single`` trains ONE triple and returns its manifest row. It is reused by the
# sequential loop below and by parallel_sweep_HAR_weightdecay.py (one process per run).


def _recdel_layers(model):
    """Return the axonal_recdel (incl. synaptic_recdel) layers in model.layers order."""
    return [m for m in model.layers if isinstance(m, axonal_recdel)]


def _wd_tag(wd):
    """Filesystem-safe, stable directory tag for a weight-decay value: wd0, wd0.001, wd0.01."""
    return f"wd{wd:g}"


def run_single(model_name, wd, seed, sweep_root, device, *,
               delay_std_init=8, round_pos=True, epochs_override="", batch_override="", subdir=None):
    """Train a single run and persist all artifacts under
    ``sweep_root/model_name/<subdir>/seed<seed>/``.
    """
    tag = subdir if subdir is not None else _wd_tag(wd)
    config = Config()
    config.seed = seed
    config.delay_std_init = delay_std_init
    config.weight_decay_positions = wd
    config.round_pos_each_epoch = round_pos
    if epochs_override:
        config.epochs = int(epochs_override)
    if batch_override:
        config.batch_size = int(batch_override)
    seed_everything(seed=config.seed, is_cuda=True)

    train_loader, valid_loader, test_loader = load_dataset(config)
    model = getattr(snn_module, model_name)(config).to(device)

    # Apply the hardcoded kernel choice per layer type (None -> library default:
    # axonal triton_exact / synaptic eventdriven on CUDA, v2 on CPU).
    from delrec.delay_layers import synaptic_recdel
    for _layer in model.layers:
        if isinstance(_layer, synaptic_recdel):              # check subclass first
            if SYN_FORWARD_VERSION:
                _layer.forward_version = SYN_FORWARD_VERSION
        elif isinstance(_layer, axonal_recdel):
            if AXONAL_FORWARD_VERSION:
                _layer.forward_version = AXONAL_FORWARD_VERSION
    n_recdel = len(_recdel_layers(model))
    print(f"\n=== {model_name} | wd={wd:g} | std{delay_std_init} | seed={seed} ===")
    print(f"recurrent layers: {n_recdel} | axonal={AXONAL_FORWARD_VERSION or 'default(triton_exact)'} "
          f"synaptic={SYN_FORWARD_VERSION or 'default(eventdriven)'}")

    optimizer, scheduler = init_optim_sche(model, config)
    count_parameters(model)


    # Run directory for this run.
    run_dir = os.path.join(sweep_root, model_name, tag, f"seed{seed}")
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    config.results_dir = run_dir
    with open(os.path.join(run_dir, 'config.json'), 'w') as fid:
        json.dump(config.__dict__, fid, indent=2)

    # Initial (post-init, pre-training) recurrent delays per layer.
    recdel_layers = _recdel_layers(model)
    init_delays = {f'layer{i}': l.recurrent_delays.detach().cpu().numpy()
                   for i, l in enumerate(recdel_layers)}
    np.savez(os.path.join(run_dir, 'init_delays.npz'), **init_delays)

    # Reference for per-epoch delay-variation tracking.
    prev_recurrent_delays_vals = [l.recurrent_delays.detach().clone()
                                  for l in recdel_layers]

    # For storing results
    train_res = pd.DataFrame()
    val_res = pd.DataFrame()
    # rows = layer{i}, cols = epoch . Mean-abs delay variation per epoch
    delay_var_res = pd.DataFrame()
    best_val_acc = 0.0

    for epoch in range(config.epochs):

        # If update sigma at each epoch :
        for m in model.layers:
            if isinstance(m, axonal_recdel):
                m.update_sigma(epoch)

        train_acc, train_loss = train(train_loader, model, optimizer, epoch, device, config)
        val_acc, val_loss = test(valid_loader, model, epoch, device, config)

        for sc in scheduler:
            sc.step()

        # for logs
        train_res[str(epoch)] = [train_acc, train_loss]
        val_res[str(epoch)] = [val_acc, val_loss]
        train_res.to_csv(os.path.join(run_dir, 'train_res.csv'), index=True)
        val_res.to_csv(os.path.join(run_dir, 'val_res.csv'), index=True)

        # Following delay variation (per recdel layer)
        epoch_var = {}
        for rec_idx, m in enumerate(recdel_layers):
            current = m.recurrent_delays.detach()
            prev = prev_recurrent_delays_vals[rec_idx]
            delta = (current - prev).abs().mean().item()
            epoch_var[f'layer{rec_idx}'] = delta
            prev_recurrent_delays_vals[rec_idx] = current.clone()
        delay_var_res[str(epoch)] = pd.Series(epoch_var)
        delay_var_res.to_csv(os.path.join(run_dir, 'delay_variation.csv'), index=True)

        state = {'net': model.state_dict(), 'acc': val_acc, 'epoch': epoch}
        torch.save(state, os.path.join(run_dir, 'last.pth'))

        if val_acc >= best_val_acc:
            torch.save(state, os.path.join(run_dir, 'best.pth'))
            best_val_acc = val_acc

        print(
            'Val Epoch: [{}/{}], lr: {:.6f}, lr_pos: {:.6f}, acc: {:.4f}, best: {:.4f}'
            .format(epoch, config.epochs,
                    optimizer[0].param_groups[0]['lr'],
                    optimizer[1].param_groups[0]['lr'],
                    val_acc, best_val_acc))

    ### Testing the best model ###
    best_ckpt = torch.load(os.path.join(run_dir, 'best.pth'))
    model.load_state_dict(best_ckpt['net'])
    best_epoch = best_ckpt['epoch']
    best_val = best_ckpt['acc']

    # Final test at sigma=0 (sharp single-tap delays), rounded, matches perf_HAR.
    for m in model.modules():
        if isinstance(m, axonal_recdel):
            m.sigma = 0.0
            m.use_sig_p = False
    if getattr(config, 'round_pos_each_epoch', False) and hasattr(model, 'round_pos'):
        model.round_pos()

    final_test_acc, final_test_loss = test(test_loader, model, best_epoch, device, config)

    # Delays of the exact model that was tested (best, sigma=0, rounded).
    final_delays = {f'layer{i}': l.recurrent_delays.detach().cpu().numpy()
                    for i, l in enumerate(_recdel_layers(model))}
    np.savez(os.path.join(run_dir, 'final_delays.npz'), **final_delays)

    print(f"\nFinal Test ({model_name}, {tag}, seed {seed}) "
          f"Acc: {final_test_acc:.4f}, Loss: {final_test_loss:.4f} "
          f"(from best val_acc={best_val:.4f}@epoch={best_epoch})")


    with open(os.path.join(run_dir, 'final_test.json'), 'w') as fid:
        json.dump({"acc": final_test_acc, "loss": final_test_loss,
                   "best_val": best_val, "best_epoch": int(best_epoch),
                   "weight_decay_positions": wd,
                   "delay_std_init": delay_std_init}, fid, indent=2)


    return {"model": model_name, "weight_decay_positions": wd,
            "delay_std_init": delay_std_init, "seed": seed,
            "run_dir": os.path.relpath(run_dir, sweep_root),
            "final_test_acc": final_test_acc}


if __name__ == "__main__":

    # Sweep axes (all overridable via env for splitting / smoke tests).
    WEIGHT_DECAYS = [float(s) for s in os.environ.get(
        "HAR_WEIGHT_DECAYS", "0,1e-3,3e-3,1e-2,3e-2,1e-1").split(",")]
    MODELS = [s for s in os.environ.get(
        "HAR_MODELS", "SNN_recurrent_delays,SNN_synaptic_recurrent_delays").split(",") if s]
    seed_list = [int(s) for s in os.environ.get("HAR_SEEDS", "0,1,2").split(",")]
    DELAY_STD_INIT = int(os.environ.get("HAR_DELAY_STD_INIT", "8"))
    # round_pos=1 matches the existing std8 comparison runs (roundpos1).
    ROUND_POS = os.environ.get("HAR_ROUND_POS", "1").lower() in ("1", "true", "yes")
    SWEEP_ROOT = os.environ.get("HAR_SWEEP_ROOT", './exp/HAR/weight_decay_sweep')
    # optional epoch override (handy for range-finding / smoke tests). Defaults to Config.epochs
    EPOCHS_OVERRIDE = os.environ.get("HAR_EPOCHS", "")
    # optional batch-size override. Synaptic (N,N) delays are memory-heavier.
    BATCH_OVERRIDE = os.environ.get("HAR_BATCH_SIZE", "")

    Path(SWEEP_ROOT).mkdir(parents=True, exist_ok=True)
    print(f"Sweep root: {SWEEP_ROOT}")
    print(f"Models: {MODELS} | delay_std_init: {DELAY_STD_INIT} | "
          f"weight_decays: {WEIGHT_DECAYS} | seeds: {seed_list} | round_pos: {ROUND_POS}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.set_default_dtype(torch.float32)
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU")

    manifest = {
        "sweep_root": SWEEP_ROOT,
        "models": MODELS,
        "delay_std_init": DELAY_STD_INIT,
        "weight_decays": WEIGHT_DECAYS,
        "seeds": seed_list,
        "round_pos": ROUND_POS,
        "runs": [],
    }
    # (model, wd) -> list of final test accuracies, for the end-of-sweep summary
    group_acc = {}

    for model_name in MODELS:
        for wd in WEIGHT_DECAYS:
            for run_seed in seed_list:
                row = run_single(model_name, wd, run_seed, SWEEP_ROOT, device,
                                 delay_std_init=DELAY_STD_INIT, round_pos=ROUND_POS,
                                 epochs_override=EPOCHS_OVERRIDE, batch_override=BATCH_OVERRIDE)

                manifest["runs"].append(row)
                with open(os.path.join(SWEEP_ROOT, 'manifest.json'), 'w') as fid:
                    json.dump(manifest, fid, indent=2)

                group_acc.setdefault((model_name, wd), []).append(row["final_test_acc"])

    print("\n==================== SWEEP SUMMARY ====================")
    for (model_name, wd), accs in sorted(group_acc.items()):
        accs = np.array(accs)
        print(f"{model_name:32s} wd={wd:<8g} "
              f"test acc = {accs.mean():.4f} +/- {accs.std():.4f}  (seeds={list(seed_list)})")
    print(f"\nManifest: {os.path.join(SWEEP_ROOT, 'manifest.json')}")
