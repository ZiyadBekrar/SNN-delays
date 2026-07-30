"""Sweep over the chaos parameter tau x the prediction horizon H.

Produces: Mackey-Glass forecasting across chaos and horizon.
Outputs:  trained_models/MG/mackey_glass_sweep/
Usage:    python experiments/sweeps/sweep_MG.py

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

import json
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path

from configs.perf_MG import Config

from delrec.training.mg import train, evaluate, init_optim_sche
import delrec.networks as snn_module
from delrec.networks import axonal_recdel
from delrec.utils import count_parameters, seed_everything
from delrec.datasets import load_dataset


# ── Recurrent kernel choice (hardcode here. None = library default) ────────────
# Library default on GPU for the axonal layers this sweep trains: 'triton_exact'
# (the exact persistent Triton scan). 'v2' (pure torch) on CPU. Set to force a
# kernel, e.g. AXONAL_FORWARD_VERSION = 'eventdriven'. Env REC_FWD takes precedence.
AXONAL_FORWARD_VERSION = os.environ.get("REC_FWD") or None

# ── Sweep grid (hardcode here. Env MG_TAUS / MG_HORIZONS / MG_SEEDS override) ──
# The (system delay tau x prediction horizon H) grid swept by __main__. Kept in sync
# with parallel_sweep_MG.py's USER CONFIG block.
MG_TAUS = [17, 30, 50, 70, 100]
MG_HORIZONS = [6, 15, 30, 50, 84]
MG_SEEDS = [0, 1, 2, 3, 4]

# Mackey-Glass sweep: chaotic time-series prediction (regression), swept over the
# system's own delay mg_tau × the prediction horizon H. Both axes set how far back the
# network must look, so they are the natural probe for learnable recurrent delays.
#
# The four families are the delay conditions. The delay type (axonal per-neuron vs
# synaptic per-synapse) is chosen by DELAY_TYPE below. The learned + fixed families
# follow that choice, no_delays is delay-free either way:
#   learned_delays        SNN_[synaptic_]recurrent_delays        learned, sigma annealed from sigma_init
#   learned_no_annealing  SNN_[synaptic_]recurrent_delays        learned, no annealing (sigma=0 throughout)
#   fixed_delays          SNN_fixed_[synaptic_]recurrent_delays  delays frozen at their random init (control)
#   no_delays             SNN_vanilla_recurrent                  delays frozen at 0 (plain RSNN)
#
# Per run it persists:
#   config.json, train_res.csv, val_res.csv, best.pth, last.pth   (as the other sweeps)
#   init_delays.npz, per-layer recurrent_delays right after init
#   final_delays.npz, per-layer recurrent_delays of the tested (best, rounded) model
#   delay_variation.csv, per-epoch mean-abs delay variation per recdel layer
#   predictions.npz, train/val/test predictions + targets of the tested model
#   final_test.json, final train/val/test metrics + best val epoch + the sweep axes
# A manifest.json at the sweep root indexes all runs for analysis/MG/report.py.
#
# ``run_single`` trains ONE (family, mg_tau, H, seed) run and returns its manifest row. It
# is reused by the sequential loop below and by parallel_sweep_MG.py (one process per run).

# ── Recurrent delay type (hardcode here. Env MG_DELAY_TYPE overrides) ──────────
# 'axonal'   -> per-neuron delays  (recurrent_delays shape (N,))
# 'synaptic' -> per-synapse delays (recurrent_delays shape (N, N))
# The learned + fixed families use the matching class pair. No_delays is unaffected.
# NB: both types write the same family dir names, so run each type into its own
# MG_SWEEP_ROOT (or they overwrite each other).
DELAY_TYPE = os.environ.get("MG_DELAY_TYPE", "synaptic")

# delay type -> (learned class, fixed class) in src/SSC/snn.py
_RECDEL_CLASSES = {
    'axonal':   ('SNN_recurrent_delays', 'SNN_fixed_recurrent_delays'),
    'synaptic': ('SNN_synaptic_recurrent_delays', 'SNN_fixed_synaptic_recurrent_delays'),
}


def _build_families(delay_type):
    """family tag -> (model class in src/SSC/snn.py, config overrides)."""
    learned_cls, fixed_cls = _RECDEL_CLASSES[delay_type]
    return {
        'learned_delays':       (learned_cls, {}),
        'learned_no_annealing': (learned_cls, {'sigma_init': 0.0}),
        'fixed_delays':         (fixed_cls, {}),
        'no_delays':            ('SNN_vanilla_recurrent', {}),
    }


FAMILIES = _build_families(DELAY_TYPE)


def _recdel_layers(model):
    """Return the axonal_recdel layers in model.layers order."""
    return [m for m in model.layers if isinstance(m, axonal_recdel)]


def _axis_tag(mg_tau, horizon):
    """Filesystem-safe, stable directory tag for a point of the (tau, H) grid."""
    return f"tau{mg_tau:g}_H{horizon:g}"


def run_single(family, mg_tau, horizon, seed, sweep_root, device, *,
               epochs_override="", batch_override=""):
    """Train a single run and persist all artifacts under
    ``sweep_root/<family>/tau<tau>_H<H>/seed<seed>/``.
    """
    model_name, overrides = FAMILIES[family]
    tag = _axis_tag(mg_tau, horizon)

    config = Config()
    config.seed = seed
    config.mg_tau = mg_tau
    config.prediction_horizon = horizon
    for k, v in overrides.items():
        setattr(config, k, v)
    if epochs_override:
        config.epochs = int(epochs_override)
    if batch_override:
        config.batch_size = int(batch_override)
    seed_everything(seed=config.seed, is_cuda=True)

    train_loader, valid_loader, test_loader = load_dataset(config)
    model = getattr(snn_module, model_name)(config).to(device)

    # Apply the hardcoded kernel choice (None -> library default: triton_exact on CUDA, v2 on CPU).
    if AXONAL_FORWARD_VERSION:
        for _layer in _recdel_layers(model):
            _layer.forward_version = AXONAL_FORWARD_VERSION

    default_kernel = 'triton_exact' if DELAY_TYPE == 'axonal' else 'eventdriven'
    print(f"\n=== {family} ({model_name}, {DELAY_TYPE}) | tau={mg_tau:g} | H={horizon:g} | seed={seed} ===")
    print(f"recurrent layers: {len(_recdel_layers(model))} | "
          f"kernel={AXONAL_FORWARD_VERSION or f'default({default_kernel} on CUDA)'}")

    optimizer, scheduler = init_optim_sche(model, config)
    count_parameters(model)


    # Run directory for this run.
    run_dir = os.path.join(sweep_root, family, tag, f"seed{seed}")
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    config.results_dir = run_dir
    with open(os.path.join(run_dir, 'config.json'), 'w') as fid:
        json.dump(config.__dict__, fid, indent=2, default=str)

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
    best_val_nmse = float('inf')

    for epoch in range(config.epochs):

        # If update sigma at each epoch :
        for m in model.layers:
            if isinstance(m, axonal_recdel):
                m.update_sigma(epoch)

        train_nmse, train_loss = train(train_loader, model, optimizer, epoch, device, config)
        val = evaluate(valid_loader, model, epoch, device, config, split='val')

        for sc in scheduler:
            sc.step()

        # for logs
        train_res[str(epoch)] = [train_nmse, train_loss]
        val_res[str(epoch)] = [val['nmse'], val['mse']]
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

        state = {'net': model.state_dict(), 'nmse': val['nmse'], 'epoch': epoch}
        torch.save(state, os.path.join(run_dir, 'last.pth'))

        if val['nmse'] <= best_val_nmse:
            torch.save(state, os.path.join(run_dir, 'best.pth'))
            best_val_nmse = val['nmse']

        print(
            'Val Epoch: [{}/{}], lr: {:.6f}, lr_pos: {:.6f}, nmse: {:.6f}, best: {:.6f}'
            .format(epoch, config.epochs,
                    optimizer[0].param_groups[0]['lr'],
                    optimizer[1].param_groups[0]['lr'],
                    val['nmse'], best_val_nmse))

    ### Testing the best model ###
    best_ckpt = torch.load(os.path.join(run_dir, 'best.pth'))
    model.load_state_dict(best_ckpt['net'])
    best_epoch = best_ckpt['epoch']
    best_val = best_ckpt['nmse']

    # Final eval at sigma=0 (sharp single-tap delays), rounded, matches the other datasets.
    for m in model.modules():
        if isinstance(m, axonal_recdel):
            m.sigma = 0.0
            m.use_sig_p = False
    if getattr(config, 'round_pos_each_epoch', False) and hasattr(model, 'round_pos'):
        model.round_pos()

    final = {split: evaluate(loader, model, best_epoch, device, config, split=split)
             for split, loader in [('train', train_loader), ('val', valid_loader),
                                   ('test', test_loader)]}

    np.savez(
        os.path.join(run_dir, 'predictions.npz'),
        **{f'{split}_{k}': final[split][k]
           for split in final for k in ('preds', 'tgts')},
    )

    # Delays of the exact model that was tested (best, sigma=0, rounded).
    final_delays = {f'layer{i}': l.recurrent_delays.detach().cpu().numpy()
                    for i, l in enumerate(_recdel_layers(model))}
    np.savez(os.path.join(run_dir, 'final_delays.npz'), **final_delays)

    test = final['test']
    print(f"\nFinal Test ({family}, {tag}, seed {seed}) "
          f"NMSE: {test['nmse']:.6f}, NRMSE: {test['nrmse']:.6f}, R2: {test['r2']:.4f} "
          f"(from best val NMSE={best_val:.6f}@epoch={best_epoch})")


    metrics = {split: {k: final[split][k] for k in ('mse', 'nmse', 'nrmse', 'r2')}
               for split in final}
    with open(os.path.join(run_dir, 'final_test.json'), 'w') as fid:
        json.dump({**metrics,
                   "best_val_nmse": best_val, "best_epoch": int(best_epoch),
                   "family": family, "model_name": model_name,
                   "mg_tau": mg_tau, "horizon": horizon, "seed": seed}, fid, indent=2)


    return {"family": family, "model_name": model_name, "mg_tau": mg_tau,
            "horizon": horizon, "seed": seed,
            "run_dir": os.path.relpath(run_dir, sweep_root),
            "final_test_nmse": test['nmse'], "final_test_r2": test['r2']}


if __name__ == "__main__":

    # Sweep axes (all overridable via env for splitting / smoke tests).
    TAUS = [float(s) for s in os.environ.get(
        "MG_TAUS", ",".join(str(t) for t in MG_TAUS)).split(",")]
    HORIZONS = [int(s) for s in os.environ.get(
        "MG_HORIZONS", ",".join(str(h) for h in MG_HORIZONS)).split(",")]
    seed_list = [int(s) for s in os.environ.get(
        "MG_SEEDS", ",".join(str(s) for s in MG_SEEDS)).split(",")]
    MODELS = [s for s in os.environ.get(
        "MG_FAMILIES",
        "learned_delays,learned_no_annealing,fixed_delays,no_delays").split(",") if s]
    SWEEP_ROOT = os.environ.get("MG_SWEEP_ROOT", './exp/MG/mackey_glass_sweep')
    # optional epoch / batch overrides (handy for smoke tests). Default to Config's
    EPOCHS_OVERRIDE = os.environ.get("MG_EPOCHS", "")
    BATCH_OVERRIDE = os.environ.get("MG_BATCH_SIZE", "")

    Path(SWEEP_ROOT).mkdir(parents=True, exist_ok=True)
    print(f"Sweep root: {SWEEP_ROOT}")
    print(f"Delay type: {DELAY_TYPE}")
    print(f"Families: {MODELS} | taus: {TAUS} | horizons: {HORIZONS} | seeds: {seed_list}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU")

    manifest = {
        "sweep_root": SWEEP_ROOT,
        "families": MODELS,
        "taus": TAUS,
        "horizons": HORIZONS,
        "seeds": seed_list,
        "runs": [],
    }
    # (family, tau, H) -> list of final test NMSEs, for the end-of-sweep summary
    group_nmse = {}

    for family in MODELS:
        for mg_tau in TAUS:
            for horizon in HORIZONS:
                for run_seed in seed_list:
                    row = run_single(family, mg_tau, horizon, run_seed, SWEEP_ROOT, device,
                                     epochs_override=EPOCHS_OVERRIDE,
                                     batch_override=BATCH_OVERRIDE)

                    manifest["runs"].append(row)
                    with open(os.path.join(SWEEP_ROOT, 'manifest.json'), 'w') as fid:
                        json.dump(manifest, fid, indent=2)

                    group_nmse.setdefault((family, mg_tau, horizon), []).append(
                        row["final_test_nmse"])

    print("\n==================== SWEEP SUMMARY ====================")
    for (family, mg_tau, horizon), nmses in sorted(group_nmse.items()):
        nmses = np.array(nmses)
        print(f"{family:22s} {_axis_tag(mg_tau, horizon):>14s} "
              f"test NMSE = {nmses.mean():.6f} +/- {nmses.std():.6f}  (seeds={list(seed_list)})")
    print(f"\nManifest: {os.path.join(SWEEP_ROOT, 'manifest.json')}")
