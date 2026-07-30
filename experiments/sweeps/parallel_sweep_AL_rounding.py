"""Run the AL rounding sweep, one process per run across the GPUs.

Sweeps nothing, it retrains the AL models under per-epoch rounding.

Each worker trains exactly one (model, value, seed) and writes the same artifacts
as a single training run. Each GPU slot gets its own Triton cache directory, so
concurrent just-in-time compilations cannot race.

Usage:    python experiments/sweeps/parallel_sweep_AL_rounding.py
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

# Repo-root on sys.path so `configs`/`delrec` resolve when run as
# `python experiments/sweeps/parallel_sweep_AL_rounding.py`.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src")]


# ============================ USER CONFIG (hardcode here) ==================== #
# GPUS: None -> auto-detect every GPU on the box, or force a subset, e.g. [1] to
#       use only GPU 1, or [0, 1] to use both. (Env AL_GPUS="1" overrides.)
GPUS = None
# MAX_RUNS_PER_GPU: how many training runs may execute concurrently on one GPU.
#       (Env AL_MAX_RUNS_PER_GPU overrides.) Synaptic (N,N) delays + Triton are
#       memory-heavy, so 1 is the safe default.
MAX_RUNS_PER_GPU = 1
# PRINT_EPOCH_PROGRESS: also echo a compact per-epoch line per running run to this
#       terminal (full detail always stays in each run's run.log). Set False for
#       launch/done lines only. (Env AL_PRINT_EPOCH_PROGRESS=0 overrides.)
PRINT_EPOCH_PROGRESS = True
# ============================================================================ #

POLL_SECONDS = 3.0

# Matches the worker's per-epoch line, e.g.
#   Val Epoch: [12/200], lr: 0.000500, lr_pos: 0.005000, acc: 84.3456, best: 85.1234
_EPOCH_RE = re.compile(r"Val Epoch: \[(\d+)/(\d+)\].*?acc: ([\d.]+), best: ([\d.]+)")

# Recurrent forward kernel, per the repo convention: axonal -> the exact Triton scan,
# synaptic -> its exact spike-sparse counterpart ('triton_exact' has no synaptic path).
# All kernels are numerically equivalent, so this is a speed choice only.
FORWARD_VERSION = os.environ.get("AL_FORWARD_VERSION", "triton_exact")
SYN_FORWARD_VERSION = os.environ.get("AL_SYN_FORWARD_VERSION", "eventdriven")


# --------------------------------------------------------------------------- #
# The worker: one (model, seed) training run
# --------------------------------------------------------------------------- #
def run_single(model_name, seed, sweep_root, device, *,
               round_pos=True, epochs_override="", batch_override=""):
    """Train a single (model, seed) AL run and persist all artifacts under
    ``sweep_root/model_name/seed<seed>/``.
    """
    import torch
    from datetime import datetime
    from pathlib import Path
    import pandas as pd
    import numpy as np


    from configs.perf_AL import Config
    from delrec.training.al import train, test, init_optim_sche
    from delrec.utils import seed_everything, count_parameters
    from delrec.delay_layers import axonal_recdel, synaptic_recdel
    import delrec.networks as snn_module
    from delrec.datasets import load_dataset

    config = Config()
    config.seed = seed
    config.round_pos_each_epoch = round_pos
    if epochs_override:
        config.epochs = int(epochs_override)
    if batch_override:
        config.batch_size = int(batch_override)
    seed_everything(seed=config.seed, is_cuda=True)

    # AL has no validation split: load_dataset returns the test loader as valid too.
    train_loader, valid_loader, test_loader = load_dataset(config)
    model = getattr(snn_module, model_name)(config).to(device)

    # --- switch the recurrent-delay layers to their fast forward path ---
    n_ax = n_syn = 0
    for m in model.modules():
        if isinstance(m, synaptic_recdel):
            m.forward_version = SYN_FORWARD_VERSION
            n_syn += 1
        elif isinstance(m, axonal_recdel):
            m.forward_version = FORWARD_VERSION
            n_ax += 1

    print(f"\n=== {model_name} | seed={seed} | round_pos_each_epoch={round_pos} ===")
    print(f"forward_version: '{FORWARD_VERSION}' on {n_ax} axonal layers, "
          f"'{SYN_FORWARD_VERSION}' on {n_syn} synaptic layers")

    optimizer, scheduler = init_optim_sche(model, config)
    count_parameters(model)


    run_dir = os.path.join(sweep_root, model_name, f"seed{seed}")
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    config.results_dir = run_dir
    with open(os.path.join(run_dir, 'config.json'), 'w') as fid:
        json.dump(config.__dict__, fid, indent=2)

    # Initial (post-init, pre-training) recurrent delays per layer.
    recdel_layers = [m for m in model.layers if isinstance(m, axonal_recdel)]
    np.savez(os.path.join(run_dir, 'init_delays.npz'),
             **{f'layer{i}': l.recurrent_delays.detach().cpu().numpy()
                for i, l in enumerate(recdel_layers)})

    # Reference for per-epoch delay-variation tracking.
    prev_recurrent_delays_vals = [l.recurrent_delays.detach().clone()
                                  for l in recdel_layers]

    train_res = pd.DataFrame()
    val_res = pd.DataFrame()
    # rows = layer{i}, cols = epoch . Mean-abs delay variation per epoch
    delay_var_res = pd.DataFrame()
    best_val_acc = 0.0

    for epoch in range(config.epochs):

        for m in model.layers:
            if isinstance(m, axonal_recdel):
                m.update_sigma(epoch)

        train_acc, train_loss = train(train_loader, model, optimizer, epoch, device, config)
        # test() projects the delays onto the integer grid before evaluating when
        # config.round_pos_each_epoch is True. This is where the rounding happens.
        val_acc, val_loss = test(valid_loader, model, epoch, device, config)

        for sc in scheduler:
            sc.step()

        train_res[str(epoch)] = [train_acc, train_loss]
        val_res[str(epoch)] = [val_acc, val_loss]
        train_res.to_csv(os.path.join(run_dir, 'train_res.csv'), index=True)
        val_res.to_csv(os.path.join(run_dir, 'val_res.csv'), index=True)

        epoch_var = {}
        for rec_idx, m in enumerate(recdel_layers):
            current = m.recurrent_delays.detach()
            delta = (current - prev_recurrent_delays_vals[rec_idx]).abs().mean().item()
            epoch_var[f'layer{rec_idx}'] = delta
            prev_recurrent_delays_vals[rec_idx] = current.clone()
        delay_var_res[str(epoch)] = pd.Series(epoch_var)
        delay_var_res.to_csv(os.path.join(run_dir, 'delay_variation.csv'), index=True)

        state = {'net': model.state_dict(), 'acc': val_acc, 'epoch': epoch}
        torch.save(state, os.path.join(run_dir, 'last.pth'))

        if val_acc >= best_val_acc:
            torch.save(state, os.path.join(run_dir, 'best.pth'))
            best_val_acc = val_acc

        print('Val Epoch: [{}/{}], lr: {:.6f}, lr_pos: {:.6f}, acc: {:.4f}, best: {:.4f}'
              .format(epoch, config.epochs,
                      optimizer[0].param_groups[0]['lr'],
                      optimizer[1].param_groups[0]['lr'],
                      val_acc, best_val_acc))

    ### Testing the best model ###
    best_ckpt = torch.load(os.path.join(run_dir, 'best.pth'))
    model.load_state_dict(best_ckpt['net'])
    best_epoch = best_ckpt['epoch']
    best_val = best_ckpt['acc']

    # ``experiments/train.py --dataset al``'s final-test regime: sharp single-tap delays, no p_spread. Sigma is a
    # plain attribute (not in state_dict), so it must be set explicitly here.
    for m in model.modules():
        if isinstance(m, axonal_recdel):
            m.sigma = 0.0
            m.use_sig_p = False

    # Under round_pos the checkpoint's delays are already integers (test() projected
    # them in place before the epoch that saved it), so this final round is a no-op.
    final_test_acc, final_test_loss = test(test_loader, model, best_epoch, device, config)

    # Delays as actually used at test time, for the downstream analyses.
    np.savez(os.path.join(run_dir, 'final_delays.npz'),
             **{f'layer{i}': l.recurrent_delays.detach().cpu().numpy()
                for i, l in enumerate(recdel_layers)})

    print(f"\nFinal Test ({model_name}, seed {seed}) "
          f"Acc: {final_test_acc:.4f}, Loss: {final_test_loss:.4f} "
          f"(from best val_acc={best_val:.4f}@epoch={best_epoch})")


    with open(os.path.join(run_dir, 'final_test.json'), 'w') as fid:
        # acc_rounded == acc: these delays are integers already. Written so the AL
        # analysis glue (analysis/AL/runs.py) reads the file instead of recomputing.
        json.dump({"acc": final_test_acc, "acc_rounded": final_test_acc,
                   "loss": final_test_loss, "best_val": best_val,
                   "best_epoch": int(best_epoch),
                   "round_pos_each_epoch": round_pos}, fid, indent=2)


    return {"model": model_name, "seed": seed,
            "run_dir": os.path.relpath(run_dir, sweep_root),
            "final_test_acc": final_test_acc}


# --------------------------------------------------------------------------- #
# Launcher
# --------------------------------------------------------------------------- #
def _run_dir(sweep_root, run):
    model, seed = run
    return os.path.join(sweep_root, model, f"seed{seed}")


def _latest_epoch(run_log):
    """(epoch, epochs, val_acc, best) from the last 'Val Epoch' line in a run.log,
    or None if none logged yet."""
    try:
        with open(run_log, errors="ignore") as f:
            data = f.read()
    except OSError:
        return None
    m = None
    for m in _EPOCH_RE.finditer(data):
        pass  # keep only the last match
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4))


def detect_gpus():
    """Resolve the GPU list: env AL_GPUS > GPUS constant > auto (nvidia-smi -L)."""
    env = os.environ.get("AL_GPUS")
    if env:
        return [int(x) for x in env.split(",") if x.strip() != ""]
    if GPUS is not None:
        return list(GPUS)
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        n = sum(1 for line in out.splitlines() if line.startswith("GPU "))
        return list(range(max(n, 1)))
    except Exception:
        return [0]


def build_pairs():
    """(model, seed) list + run params. No swept value: the config is perf_AL's as-is."""
    seeds = [int(s) for s in os.environ.get("AL_SEEDS", "0,1,2,3,4").split(",")]
    # Rounding ON is the point of this sweep. AL_ROUND_POS=0 reproduces the published
    # (fractional) regime, e.g. As a sanity check of this launcher against exp/AL.
    round_pos = os.environ.get("AL_ROUND_POS", "1").lower() in ("1", "true", "yes")
    epochs = os.environ.get("AL_EPOCHS", "")
    batch = os.environ.get("AL_BATCH_SIZE", "")
    # Learned families only: the fixed ones are integer-valued at init, so rounding
    # them is a no-op.
    models = [s for s in os.environ.get(
        "AL_MODELS", "SNN_recurrent_delays,SNN_synaptic_recurrent_delays").split(",") if s]
    sweep_root = os.environ.get("AL_SWEEP_ROOT", './exp/AL/rounding_sweep')

    pairs = [(m, s) for m in models for s in seeds]
    params = dict(round_pos=round_pos, sweep_root=sweep_root, epochs=epochs,
                  batch=batch, models=models, seeds=seeds)
    return pairs, params


def worker_main(args):
    """--run-one: train exactly one (model, seed) run on the visible GPU."""
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_single(args.model, int(args.seed), args.sweep_root, device,
               round_pos=bool(int(args.round_pos)),
               epochs_override=args.epochs, batch_override=args.batch)


def _worker_env(gpu, cache_dir):
    """Environment for a worker: pin the GPU, isolate the Triton JIT cache, make the
    env interpreter's libstdc++ visible."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.makedirs(cache_dir, exist_ok=True)
    env["TRITON_CACHE_DIR"] = cache_dir
    lib = os.path.join(sys.prefix, "lib")
    env["LD_LIBRARY_PATH"] = lib + ":" + env.get("LD_LIBRARY_PATH", "")
    return env


def _signal_slot(slot, sig):
    """Send `sig` to a worker's whole process group (kills the worker + anything it
    spawned). Safe if the worker has already exited."""
    p = slot.get("proc")
    if p is None:
        return
    try:
        os.killpg(os.getpgid(p.pid), sig)
    except (ProcessLookupError, PermissionError):
        pass
    except Exception:
        try:
            p.send_signal(sig)
        except Exception:
            pass


def _shutdown(slots, grace=3.0):
    """Terminate every running worker: SIGTERM their groups, then SIGKILL any that don't exit
    within `grace` seconds (torch can't catch SIGKILL).
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    running = [s for s in slots if s.get("proc") is not None and s["proc"].poll() is None]
    if not running:
        return
    print(f"[shutdown] terminating {len(running)} worker(s) "
          f"(SIGTERM, then SIGKILL after {grace:g}s)...", flush=True)
    for s in running:
        _signal_slot(s, signal.SIGTERM)
    deadline = time.time() + grace
    while time.time() < deadline and any(s["proc"].poll() is None for s in running):
        time.sleep(0.3)
    for s in running:
        if s["proc"].poll() is None:
            print(f"[shutdown] SIGKILL gpu{s['gpu']} slot{s['k']} (did not stop)", flush=True)
            _signal_slot(s, signal.SIGKILL)
    # Reap the killed workers so we don't return while they are still zombies.
    for s in running:
        try:
            s["proc"].wait(timeout=5)
        except Exception:
            pass
    for s in running:
        try:
            if s.get("log"):
                s["log"].close()
        except Exception:
            pass


def _term_handler(signum, frame):
    """Turn SIGTERM (e.g. `kill <launcher>`, tmux kill-session) into KeyboardInterrupt
    so the same teardown path runs."""
    raise KeyboardInterrupt


def orchestrate(args):
    gpus = detect_gpus()
    max_per_gpu = int(os.environ.get("AL_MAX_RUNS_PER_GPU", MAX_RUNS_PER_GPU))
    pairs, params = build_pairs()
    sweep_root = params["sweep_root"]
    os.makedirs(sweep_root, exist_ok=True)
    cache_root = os.path.join(sweep_root, ".triton_cache")

    # One slot per (gpu, k). Each slot has a fixed cache dir (safe: a slot runs its
    # jobs sequentially. Different slots -> different caches).
    slots = []
    for g in gpus:
        for k in range(max_per_gpu):
            slots.append({"gpu": g, "k": k, "proc": None, "run": None, "log": None,
                          "cache_dir": os.path.join(cache_root, f"gpu{g}_slot{k}")})

    print(f"GPUs: {gpus} | max runs/GPU: {max_per_gpu} | total slots: {len(slots)}")
    print(f"Runs: {len(pairs)}  (models={params['models']}, seeds={params['seeds']}, "
          f"round_pos={params['round_pos']})")
    print(f"Sweep root: {sweep_root}\n")

    pending = list(pairs)
    results = []  # {model, seed, gpu, rc, final_test_acc}
    launched = 0

    def launch(slot, run):
        nonlocal launched
        model, seed = run
        run_dir = _run_dir(sweep_root, run)
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "run.log")
        logf = open(log_path, "w")
        cmd = [sys.executable, os.path.abspath(__file__), "--run-one",
               "--model", model, "--seed", str(seed), "--sweep-root", sweep_root,
               "--round-pos", "1" if params["round_pos"] else "0",
               "--epochs", params["epochs"], "--batch", params["batch"]]
        env = _worker_env(slot["gpu"], slot["cache_dir"])
        # start_new_session=True -> each worker is its own process-group leader, so
        # the terminal's Ctrl+C does NOT reach it directly (torch can't swallow
        # a half-delivered SIGINT). The launcher is the sole owner of its lifecycle
        # and tears it down explicitly via _shutdown() on interrupt.
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)
        slot.update(proc=proc, run=run, log=logf, last_epoch=None)
        launched += 1
        print(f"[{time.strftime('%H:%M:%S')}] launch #{launched}/{len(pairs)} "
              f"gpu{slot['gpu']} slot{slot['k']}: {model} seed{seed} "
              f"-> {log_path}", flush=True)

    def reap(slot):
        proc, run = slot["proc"], slot["run"]
        rc = proc.returncode
        slot["log"].close()
        model, seed = run
        acc = None
        ft = os.path.join(_run_dir(sweep_root, run), "final_test.json")
        if rc == 0 and os.path.exists(ft):
            try:
                acc = float(json.load(open(ft)).get("acc"))
            except Exception:
                pass
        status = "ok" if rc == 0 else f"FAILED(rc={rc})"
        print(f"[{time.strftime('%H:%M:%S')}] done   gpu{slot['gpu']} slot{slot['k']}: "
              f"{model} seed{seed}  {status}"
              + (f"  acc={acc:.4f}" if acc is not None else ""), flush=True)
        results.append({"model": model, "seed": seed, "gpu": slot["gpu"], "rc": rc,
                        "final_test_acc": acc})
        slot.update(proc=None, run=None, log=None)
        _write_manifest(sweep_root, params, results)

    print_epochs = os.environ.get("AL_PRINT_EPOCH_PROGRESS",
                                  "1" if PRINT_EPOCH_PROGRESS else "0").lower() \
        in ("1", "true", "yes")

    def maybe_print_epoch(slot):
        info = _latest_epoch(os.path.join(_run_dir(sweep_root, slot["run"]), "run.log"))
        if info is None:
            return
        e, E, acc, best = info
        if slot.get("last_epoch") == e:
            return
        slot["last_epoch"] = e
        model, seed = slot["run"]
        print(f"[{time.strftime('%H:%M:%S')}] gpu{slot['gpu']} s{slot['k']} "
              f"{model} seed{seed}  epoch {e}/{E}  "
              f"val {acc:.2f}%  best {best:.2f}%", flush=True)

    # Schedule loop. Ctrl+C (SIGINT) and `kill` (SIGTERM) both funnel into the
    # KeyboardInterrupt handler below, which tears down every running worker.
    # Reset SIGINT to the default handler: if this launcher was started as a
    # background job (`&`/nohup), the shell inherits SIGINT as SIG_IGN, which would
    # otherwise make Ctrl+C / `kill -INT` silently do nothing.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _term_handler)
    interrupted = False
    try:
        while pending or any(s["proc"] is not None for s in slots):
            for s in slots:
                if s["proc"] is not None and s["proc"].poll() is not None:
                    reap(s)
            for s in slots:
                if s["proc"] is None and pending:
                    launch(s, pending.pop(0))
            if print_epochs:
                for s in slots:
                    if s["proc"] is not None:
                        maybe_print_epoch(s)
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        interrupted = True
        print("\n[interrupt] stopping, no new runs will launch.", flush=True)
    finally:
        _shutdown(slots)

    if interrupted:
        print(f"[interrupt] workers terminated. Partial manifest: "
              f"{os.path.join(sweep_root, 'manifest_rounding.json')}", flush=True)
        raise SystemExit(130)

    _write_manifest(sweep_root, params, results)
    _summarize(results, params)


def _write_manifest(sweep_root, params, results):
    manifest = {
        "sweep_root": sweep_root,
        "models": params["models"],
        "seeds": params["seeds"],
        "round_pos": params["round_pos"],
        "runs": [{"model": r["model"], "seed": r["seed"], "gpu": r["gpu"], "rc": r["rc"],
                  "final_test_acc": r["final_test_acc"],
                  "run_dir": os.path.join(r["model"], f"seed{r['seed']}")}
                 for r in results],
    }
    with open(os.path.join(sweep_root, "manifest_rounding.json"), "w") as fid:
        json.dump(manifest, fid, indent=2)


def _summarize(results, params):
    import statistics as stats
    print("\n==================== SWEEP SUMMARY ====================")
    n_fail = sum(1 for r in results if r["rc"] != 0)
    group = {}
    for r in results:
        if r["final_test_acc"] is not None:
            group.setdefault(r["model"], []).append(r["final_test_acc"])
    for model, accs in sorted(group.items()):
        mu = stats.mean(accs)
        sd = stats.pstdev(accs) if len(accs) > 1 else 0.0
        print(f"{model:34s} test acc = {mu:.4f} +/- {sd:.4f}  (n={len(accs)})")
    if n_fail:
        print(f"\n[warn] {n_fail} run(s) FAILED, see per-run run.log files.")
    print(f"\nManifest: {os.path.join(params['sweep_root'], 'manifest_rounding.json')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-one", action="store_true",
                    help="internal: train exactly one run on the visible GPU")
    ap.add_argument("--model")
    ap.add_argument("--seed")
    ap.add_argument("--sweep-root")
    ap.add_argument("--round-pos", default="1")
    ap.add_argument("--epochs", default="")
    ap.add_argument("--batch", default="")
    args = ap.parse_args()

    if args.run_one:
        worker_main(args)
    else:
        orchestrate(args)
