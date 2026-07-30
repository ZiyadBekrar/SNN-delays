"""Train one DelRec model on one benchmark.

The five benchmarks differ only in their config, trainer and a handful of flags
collected in ``SPECS`` below, the training loop is shared. Hyperparameters live in
``configs/perf_<DATASET>.py``, edit those rather than this file.

Delays are evaluated at sigma=0, the annealed spread collapsed so each connection
carries a single sharp delay. AL is the exception: it trains with
``round_pos_each_epoch = False`` and is reported unrounded, because projecting its
fractional delays onto the integer grid costs it 12 to 17 accuracy points.

Outputs:  exp/<dataset>/<ModelClass>/perf/seed<N>_<timestamp>/ with config.json,
          train_res.csv, val_res.csv, best.pth, last.pth, init_delays.npz,
          final_delays.npz, final_test.json (+ perm.pt on PS-MNIST)
Usage:    python experiments/train.py --dataset ssc --model SNN_recurrent_delays --seeds 0,1,2,3,4
          python experiments/train.py --dataset ssc --smoke
"""

import argparse
import importlib
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "src")]

from delrec.datasets import load_dataset
from delrec.delay_layers import axonal_recdel, synaptic_recdel
from delrec.utils import count_parameters, seed_everything


class _FewBatches:
    """Loader view exposing only the first ``n`` batches, for ``--smoke``."""

    def __init__(self, loader, n):
        self._loader, self._n = loader, n

    def __iter__(self):
        for i, batch in enumerate(self._loader):
            if i >= self._n:
                return
            yield batch

    def __len__(self):
        return min(self._n, len(self._loader))

    def __getattr__(self, name):
        return getattr(self._loader, name)


@dataclass(frozen=True)
class Spec:
    """What distinguishes one benchmark's training run from another's."""
    config: str          # module holding the Config class
    trainer: str         # module holding train / test / init_optim_sche
    zoo: str             # module holding the network classes
    default_model: str
    resettable_loaders: bool = False   # SSC's HDF5 iterator needs an explicit reset
    needs_perm: bool = False           # PS-MNIST draws a pixel order per run
    sigma_zero_at_test: bool = True    # collapse the delay spread for the final test


SPECS = {
    "ssc":     Spec("configs.perf_SSC",     "delrec.training.ssc",     "delrec.networks",
                    "SNN_recurrent_delays", resettable_loaders=True),
    "psmnist": Spec("configs.perf_PSMNIST", "delrec.training.psmnist", "delrec.networks",
                    "SNN_recurrent_delays", needs_perm=True),
    "har":     Spec("configs.perf_HAR",     "delrec.training.har",     "delrec.networks",
                    "SNN_recurrent_delays"),
    "al":      Spec("configs.perf_AL",      "delrec.training.al",      "delrec.networks",
                    "SNN_recurrent_delays"),
    # The SHD stack (the SHD appendix) predates the others and has its own zoo,
    # trainer and readout. Its final test uses the training-time spread.
    "shd":     Spec("configs.perf_SHD",     "delrec.training.shd",     "delrec.networks_shd",
                    "SNN_recurrent_delays", sigma_zero_at_test=False),
}


def pick_device():
    if torch.cuda.is_available():
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.set_default_dtype(torch.float32)
        print("Using Apple Silicon GPU (MPS)")
        return torch.device("mps")
    print("GPU not available, using CPU")
    return torch.device("cpu")


def apply_kernels(model, axonal, synaptic):
    """A recurrent scan per delay type. None keeps the library default
    (``triton_exact`` axonal / ``eventdriven`` synaptic on CUDA, ``v2`` on CPU).
    All kernels are numerically equivalent, see tests/test_kernel_equivalence.py,
    so this is only ever a speed choice."""
    for m in model.modules():
        if isinstance(m, synaptic_recdel):        # subclass of axonal_recdel: check first
            if synaptic:
                m.forward_version = synaptic
        elif isinstance(m, axonal_recdel):
            if axonal:
                m.forward_version = axonal
    print(f"recurrent kernels: axonal={axonal or 'default'}, synaptic={synaptic or 'default'}")


def effective_config(config):
    """The full hyperparameter set actually in force, as a JSON-able dict."""
    merged = {k: v for k, v in vars(type(config)).items() if not k.startswith("__")}
    merged.update(vars(config))
    return {k: v for k, v in sorted(merged.items()) if not callable(v)}


def delay_snapshot(model):
    """Per-layer recurrent delays, in layer order, for the .npz artifacts."""
    return {f"layer{i}": m.recurrent_delays.detach().cpu().numpy()
            for i, m in enumerate(m for m in model.modules() if isinstance(m, axonal_recdel))}


def run_seed(spec, args, seed):
    cfg_mod = importlib.import_module(spec.config)
    trainer = importlib.import_module(spec.trainer)
    zoo = importlib.import_module(spec.zoo)

    config = cfg_mod.Config()
    config.seed = seed
    if args.epochs:
        config.epochs = args.epochs
    if args.smoke:
        config.epochs = 2
    seed_everything(seed=config.seed, is_cuda=True)

    device = pick_device()
    train_loader, valid_loader, test_loader = load_dataset(config)
    if args.smoke:
        train_loader = _FewBatches(train_loader, 8)
        valid_loader = _FewBatches(valid_loader, 4)
        test_loader = _FewBatches(test_loader, 4)

    model_cls = getattr(zoo, args.model or spec.default_model)
    print(f"Using model class: {model_cls.__name__}")
    model = model_cls(config).to(device)
    apply_kernels(model, args.kernel, args.synaptic_kernel)
    optimizer, scheduler = trainer.init_optim_sche(model, config)
    count_parameters(model)

    model_name = getattr(model, "module", model).__class__.__name__
    run_dir = Path(args.out or
                   f"exp/{config.dataset}/{model_name}/perf/"
                   f"seed{config.seed}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    config.results_dir = str(run_dir)
    (run_dir / "config.json").write_text(
        json.dumps(effective_config(config), indent=2, default=str))

    # PS-MNIST scrambles the pixel order once per run. Every later evaluation of
    # this checkpoint is meaningless under any other permutation, so it ships
    # alongside the weights.
    extra = {}
    if spec.needs_perm:
        perm = torch.randperm(784)
        torch.save(perm, run_dir / "perm.pt")
        extra["perm"] = perm
        print(f"Saved permutation to {run_dir / 'perm.pt'}")

    init_delays = delay_snapshot(model)
    if init_delays:
        np.savez(run_dir / "init_delays.npz", **init_delays)

    train_res, val_res = pd.DataFrame(), pd.DataFrame()
    best_val_acc, epoch = 0.0, 0

    for epoch in range(config.epochs):
        if spec.resettable_loaders:
            train_loader.reset()
            valid_loader.reset()

        # Eq. (13): anneal the triangular spread toward a single sharp delay.
        for m in model.modules():
            if isinstance(m, axonal_recdel):
                m.update_sigma(epoch)

        train_acc, train_loss = trainer.train(train_loader, model, optimizer, epoch,
                                              device, config, **extra)
        val_acc, val_loss = trainer.test(valid_loader, model, epoch, device, config, **extra)
        for sc in scheduler:
            sc.step()

        train_res[str(epoch)] = [train_acc, train_loss]
        val_res[str(epoch)] = [val_acc, val_loss]
        train_res.to_csv(run_dir / "train_res.csv", index=True)
        val_res.to_csv(run_dir / "val_res.csv", index=True)

        state = {
            "epoch": epoch,
            "acc": float(val_acc),
            "model": getattr(model, "module", model).state_dict(),
            "optim": [o.state_dict() for o in optimizer],
            "sched": [s.state_dict() for s in scheduler],
            "rng": {"torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        }
        torch.save(state, run_dir / "last.pth")
        if val_acc >= best_val_acc:
            torch.save(state, run_dir / "best.pth")
            best_val_acc = val_acc

        print(f"Val Epoch: [{epoch}/{config.epochs}], "
              f"lr: {optimizer[0].param_groups[0]['lr']:.6f}, "
              f"lr_pos: {optimizer[1].param_groups[0]['lr']:.6f}, "
              f"acc: {val_acc:.4f}, best: {best_val_acc:.4f}")

    # ---- final test on the best validation checkpoint -----------------------
    best_ckpt = torch.load(run_dir / "best.pth", weights_only=False)
    model.load_state_dict(best_ckpt["model"])

    if spec.sigma_zero_at_test:
        # sigma is a plain attribute, not part of state_dict, so it has to be set
        # here. P_spread (Eq. 14) is inert once sigma is 0, but turn it off too.
        for m in model.modules():
            if isinstance(m, axonal_recdel):
                m.sigma = 0.0
                m.use_sig_p = False

    if spec.resettable_loaders:
        test_loader.reset()
    final_acc, final_loss = trainer.test(test_loader, model, epoch, device, config, **extra)

    final_delays = delay_snapshot(model)
    if final_delays:
        np.savez(run_dir / "final_delays.npz", **final_delays)
    (run_dir / "final_test.json").write_text(json.dumps({
        "acc": final_acc, "loss": final_loss,
        "best_val": best_ckpt["acc"], "best_epoch": int(best_ckpt["epoch"]),
        "model": model_name, "seed": seed,
    }, indent=2))

    print(f"\nFinal Test (seed {seed}) - Acc: {final_acc:.4f}, Loss: {final_loss:.4f} "
          f"(from best val={best_ckpt['acc']:.4f} @ epoch {best_ckpt['epoch']})")
    return final_acc


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(SPECS))
    p.add_argument("--model", default=None,
                   help="network class from the benchmark's zoo "
                        "(default: SNN_recurrent_delays, i.e. learned axonal delays)")
    p.add_argument("--seeds", default="0", help="comma-separated, e.g. 0,1,2,3,4")
    p.add_argument("--kernel", default=os.environ.get("REC_FWD") or None,
                   help="axonal recurrent scan: v1|v2|triton_exact")
    p.add_argument("--synaptic-kernel", default=os.environ.get("SYN_REC_FWD") or None,
                   help="synaptic recurrent scan: v1|v2|eventdriven")
    p.add_argument("--epochs", type=int, default=None, help="override the config")
    p.add_argument("--out", default=None, help="run directory (default: exp/<dataset>/...)")
    p.add_argument("--smoke", action="store_true",
                   help="2 epochs over a handful of batches, to check the pipeline runs")
    args = p.parse_args()

    spec = SPECS[args.dataset]
    seeds = [int(s) for s in args.seeds.split(",")]
    accs = [run_seed(spec, args, s) for s in seeds]

    print(f"\nSeeds: {seeds}")
    print(f"Mean final test accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")


if __name__ == "__main__":
    main()
