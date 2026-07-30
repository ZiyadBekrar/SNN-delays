"""Sweep over the delay-initialization width s_init.

Produces: the accuracy/initialization-width curve, and the model set every
          robustness analysis perturbs.
Outputs:  trained_models/HAR/std_init_sweep/
Usage:    python experiments/sweeps/sweep_HAR_delaystd.py

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
import numpy as np

from configs.perf_HAR import Config

from delrec.training.har import *
from delrec.networks import *
import delrec.networks as snn_module
from delrec.datasets import load_dataset


# Sweep built on top of ``experiments/train.py --dataset har``. For every (model, delay_std_init, seed)
# triple it trains the exact same HAR pipeline with the exact Triton kernel
# (forward_version='triton_exact', bit-equivalent to v2) and persists, per run:
#   config.json, train_res.csv, val_res.csv, best.pth, last.pth   (as ``experiments/train.py --dataset har``)
#   init_delays.npz, per-layer recurrent_delays right after init (the "beginning")
#   delay_variation.csv, per-epoch mean-abs delay variation per recdel layer
#   final_test.json, final test acc/loss + best val/epoch
# Runs land under exp/HAR/std_init_sweep/<ModelClass>/std<X>/seed<N>/ (the tree
# analysis/HAR/runs.py reads). Manifest_delaystd.json at the sweep root indexes them.
#
# ``run_single`` trains ONE (model, delay_std_init, seed) run and returns its
# manifest row. It is reused by the sequential loop below and by
# parallel_sweep_HAR_delaystd.py (one process per run).
FORWARD_VERSION = os.environ.get("HAR_FORWARD_VERSION", "triton_exact")  # axonal: 'triton_exact'|'triton_approx'|'triton_ste'|'eventdriven'
# Per-synapse delays: the accelerated-exact path is 'eventdriven' (spike-sparse
# scatter, fastest). 'eventdriven' also works for the axonal layers above.
# Applied to synaptic_recdel layers only.
SYN_FORWARD_VERSION = os.environ.get("HAR_SYN_FORWARD_VERSION", "eventdriven")  # 'eventdriven'|'v2'


def _recdel_layers(model):
    """Return the axonal_recdel layers in model.layers order (the 'two layers')."""
    return [m for m in model.layers if isinstance(m, axonal_recdel)]


def run_single(model_name, dsi, seed, sweep_root, device, *,
               round_pos=True, epochs_override="", batch_override="", subdir=None):
    """Train a single (model, delay_std_init, seed) run and persist all artifacts
    under ``sweep_root/model_name/<subdir>/seed<seed>/``. ``subdir`` defaults to the
    std tag ``std<dsi>``. ``round_pos`` defaults to True, as in the runs already
    trained under exp/HAR/std_init_sweep/. Returns the manifest row."""
    tag = subdir if subdir is not None else f"std{dsi}"
    config = Config()
    config.seed = seed
    config.delay_std_init = dsi
    config.round_pos_each_epoch = round_pos
    if epochs_override:
        config.epochs = int(epochs_override)
    if batch_override:
        config.batch_size = int(batch_override)
    seed_everything(seed=config.seed, is_cuda=True)

    train_loader, valid_loader, test_loader = load_dataset(config)
    model = getattr(snn_module, model_name)(config).to(device)

    # --- switch the recurrent-delay layers to their fast forward path ---
    # synaptic_recdel (per-synapse, subclass of axonal_recdel) -> eventdriven.
    # Axonal layers -> the Triton kernel.
    n_ax = n_syn = 0
    for m in model.modules():
        if isinstance(m, synaptic_recdel):
            m.forward_version = SYN_FORWARD_VERSION
            n_syn += 1
        elif isinstance(m, axonal_recdel):
            m.forward_version = FORWARD_VERSION
            n_ax += 1
    print(f"\n=== {model_name} | delay_std_init={dsi} | seed={seed} ===")
    print(f"forward_version: '{FORWARD_VERSION}' on {n_ax} axonal layers, "
          f"'{SYN_FORWARD_VERSION}' on {n_syn} synaptic layers")

    optimizer, scheduler = init_optim_sche(model, config)
    count_parameters(model)


    # Run directory for this run.
    run_dir = os.path.join(sweep_root, model_name, tag, f"seed{seed}")
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    config.results_dir = run_dir
    with open(os.path.join(run_dir, 'config.json'), 'w') as fid:
        json.dump(config.__dict__, fid, indent=2)

    # Initial (post-init, pre-training) recurrent delays per layer -> "beginning".
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

    final_test_acc, final_test_loss = test(test_loader, model, best_epoch, device, config)

    print(f"\nFinal Test ({model_name}, {tag}, seed {seed}) "
          f"Acc: {final_test_acc:.4f}, Loss: {final_test_loss:.4f} "
          f"(from best val_acc={best_val:.4f}@epoch={best_epoch})")


    with open(os.path.join(run_dir, 'final_test.json'), 'w') as fid:
        json.dump({"acc": final_test_acc, "loss": final_test_loss,
                   "best_val": best_val, "best_epoch": int(best_epoch),
                   "delay_std_init": dsi}, fid, indent=2)


    return {"model": model_name, "delay_std_init": dsi, "seed": seed,
            "run_dir": os.path.relpath(run_dir, sweep_root),
            "final_test_acc": final_test_acc}


if __name__ == "__main__":

    # Sweep axes (all overridable via env for splitting / smoke tests).
    DELAY_STD_INITS = [int(s) for s in os.environ.get("HAR_DELAY_STD_INITS", "3,8,13,18,23").split(",")]
    MODELS = [s for s in os.environ.get(
        "HAR_MODELS", "SNN_recurrent_delays,SNN_fixed_recurrent_delays").split(",") if s]
    seed_list = [int(s) for s in os.environ.get("HAR_SEEDS", "0,1,2").split(",")]
    # round_pos defaults ON: the runs already trained under exp/HAR/std_init_sweep/
    # are roundpos1, so new seeds must match them.
    ROUND_POS = os.environ.get("HAR_ROUND_POS", "1").lower() in ("1", "true", "yes")
    SWEEP_ROOT = os.environ.get("HAR_SWEEP_ROOT", './exp/HAR/std_init_sweep')
    # optional epoch override (handy for smoke tests). Defaults to Config.epochs
    EPOCHS_OVERRIDE = os.environ.get("HAR_EPOCHS", "")
    # optional batch-size override. Synaptic (N,N) delays are memory-heavy at large
    # delay_std_init, so a smaller batch avoids CUDA OOM.
    BATCH_OVERRIDE = os.environ.get("HAR_BATCH_SIZE", "")

    Path(SWEEP_ROOT).mkdir(parents=True, exist_ok=True)
    print(f"Sweep root: {SWEEP_ROOT}")
    print(f"Models: {MODELS} | delay_std_init: {DELAY_STD_INITS} | seeds: {seed_list} "
          f"| round_pos: {ROUND_POS}")

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
        "delay_std_inits": DELAY_STD_INITS,
        "seeds": seed_list,
        "round_pos": ROUND_POS,
        "forward_version": FORWARD_VERSION,
        "syn_forward_version": SYN_FORWARD_VERSION,
        "runs": [],
    }
    # (model, dsi) -> list of final test accuracies, for the end-of-sweep summary
    group_acc = {}

    for model_name in MODELS:
        for dsi in DELAY_STD_INITS:
            for run_seed in seed_list:
                row = run_single(model_name, dsi, run_seed, SWEEP_ROOT, device,
                                 round_pos=ROUND_POS, epochs_override=EPOCHS_OVERRIDE,
                                 batch_override=BATCH_OVERRIDE)

                manifest["runs"].append(row)
                with open(os.path.join(SWEEP_ROOT, 'manifest_delaystd.json'), 'w') as fid:
                    json.dump(manifest, fid, indent=2)

                group_acc.setdefault((model_name, dsi), []).append(row["final_test_acc"])

    print("\n==================== SWEEP SUMMARY ====================")
    for (model_name, dsi), accs in sorted(group_acc.items()):
        accs = np.array(accs)
        print(f"{model_name:34s} std={dsi:<3d} "
              f"test acc = {accs.mean():.4f} +/- {accs.std():.4f}  (seeds={list(seed_list)})")
    print(f"\nManifest: {os.path.join(SWEEP_ROOT, 'manifest_delaystd.json')}")
