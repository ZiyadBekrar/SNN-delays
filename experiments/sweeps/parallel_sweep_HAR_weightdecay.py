"""Run the weightdecay sweep, one process per run across the GPUs.

Sweeps weight decay on the delay parameters (axis 'wd') or a smaller
    initialization for the fixed families (axis 'std').

Each worker trains exactly one (model, value, seed) and writes the same artifacts
as a single training run. Each GPU slot gets its own Triton cache directory, so
concurrent just-in-time compilations cannot race.

Usage:    python experiments/sweeps/parallel_sweep_HAR_weightdecay.py
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

# Repo-root on sys.path so the worker's `from sweep_HAR_weightdecay import ...`
# (and that module's `configs`/`delrec` imports) resolve when run as
# `python experiments/sweeps/parallel_sweep_HAR_weightdecay.py`.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src")]


# ============================ USER CONFIG (hardcode here) ==================== #
# GPUS: None -> auto-detect every GPU on the box, or force a subset, e.g. [1] to
#       use only GPU 1, or [0, 1] to use both. (Env HAR_GPUS="1" overrides.)
GPUS = None
# MAX_RUNS_PER_GPU: how many training runs may execute concurrently on one GPU.
#       (Env HAR_MAX_RUNS_PER_GPU overrides.) Synaptic (N,N) delays are memory-
#       heavier, so keep this modest if the sweep includes the synaptic model.
MAX_RUNS_PER_GPU = 2
# PRINT_EPOCH_PROGRESS: also echo a compact per-epoch line per running run to this
#       terminal (full detail always stays in each run's run.log). Set False for
#       launch/done lines only. (Env HAR_PRINT_EPOCH_PROGRESS=0 overrides.)
PRINT_EPOCH_PROGRESS = True
# SWEEP_AXIS: what to sweep (env HAR_SWEEP_AXIS overrides).
#   'wd'  -> weight decay on delays, learned models, fixed delay_std_init  (the
#           weight-decay compression sweep. Runs land in wd<val>/ subdirs).
#   'std' -> delay_std_init, FIXED-delay models, no weight decay  (the fixed-delay
#           compression sweep. Runs land in std<val>/ subdirs).
SWEEP_AXIS = "wd"
# AXONAL_FORWARD_VERSION / SYN_FORWARD_VERSION: recurrent kernel per layer type,
#       propagated to every worker. None -> library default (axonal 'triton_exact',
#       synaptic 'eventdriven' on GPU. Both exact vs v2). Set to force a kernel,
#       e.g. AXONAL_FORWARD_VERSION = 'eventdriven'. Shell env REC_FWD / SYN_REC_FWD
#       take precedence over these.
AXONAL_FORWARD_VERSION = None
SYN_FORWARD_VERSION    = None
# ============================================================================ #

POLL_SECONDS = 3.0

# Matches the worker's per-epoch line, e.g.
#   Val Epoch: [12/100], lr: 0.001500, lr_pos: 0.050000, acc: 78.3456, best: 79.1234
_EPOCH_RE = re.compile(r"Val Epoch: \[(\d+)/(\d+)\].*?acc: ([\d.]+), best: ([\d.]+)")


def _wd_tag(wd):
    """Must match sweep_HAR_weightdecay._wd_tag."""
    return f"wd{wd:g}"


def _value_tag(axis, value):
    """Subdir tag for a swept value: wd<val> for the wd axis, std<val> for the std axis."""
    return _wd_tag(value) if axis == "wd" else f"std{value:g}"


def _run_dir(sweep_root, axis, run):
    model, value, seed = run
    return os.path.join(sweep_root, model, _value_tag(axis, value), f"seed{seed}")


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
    """Resolve the GPU list: env HAR_GPUS > GPUS constant > auto (nvidia-smi -L)."""
    env = os.environ.get("HAR_GPUS")
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


def build_triples():
    """(model, value, seed) list + sweep params. `value` is a weight-decay (wd axis)
    or a delay_std_init (std axis). Model/value defaults switch on the axis."""
    axis = os.environ.get("HAR_SWEEP_AXIS", SWEEP_AXIS)
    seeds = [int(s) for s in os.environ.get("HAR_SEEDS", "0,1,2").split(",")]
    round_pos = os.environ.get("HAR_ROUND_POS", "1").lower() in ("1", "true", "yes")
    epochs = os.environ.get("HAR_EPOCHS", "")
    batch = os.environ.get("HAR_BATCH_SIZE", "")

    if axis == "wd":
        # Gap-fill values between wd0.03 (~83% plateau) and wd0.1 (accuracy cliff),
        # for BOTH learned families, to trace the learned frontier through 79-83%.
        # The original base sweep ("0,1e-3,3e-3,1e-2,3e-2,1e-1") is already complete
        # for both families. These land in new wd<val>/ subdirs alongside it.
        values = [float(s) for s in os.environ.get(
            "HAR_WEIGHT_DECAYS", "0.04,0.05,0.06,0.07").split(",")]
        models = [s for s in os.environ.get(
            "HAR_MODELS", "SNN_recurrent_delays,SNN_synaptic_recurrent_delays").split(",") if s]
        dsi = float(os.environ.get("HAR_DELAY_STD_INIT", "8"))
        sweep_root = os.environ.get("HAR_SWEEP_ROOT", './exp/HAR/weight_decay_sweep')
    elif axis == "std":
        values = [float(s) for s in os.environ.get(
            "HAR_DELAY_STD_INITS", "0.5,1,2,3,5,8").split(",")]
        models = [s for s in os.environ.get(
            "HAR_MODELS",
            "SNN_fixed_recurrent_delays,SNN_fixed_synaptic_recurrent_delays").split(",") if s]
        dsi = None  # value IS the delay_std_init
        sweep_root = os.environ.get("HAR_SWEEP_ROOT", './exp/HAR/fixed_compression_sweep')
    else:
        raise ValueError(f"HAR_SWEEP_AXIS must be 'wd' or 'std', got {axis!r}")

    triples = [(m, v, s) for m in models for v in values for s in seeds]
    params = dict(axis=axis, delay_std_init=dsi, round_pos=round_pos, sweep_root=sweep_root,
                  epochs=epochs, batch=batch, models=models, values=values, seeds=seeds)
    return triples, params


def worker_main(args):
    """--run-one: train exactly one (model, value, seed) run on the visible GPU.
    On the 'wd' axis `value` is the weight decay (delay_std_init fixed). On the
    'std' axis `value` is the delay_std_init (weight decay 0, fixed-delay model)."""
    import torch
    from sweep_HAR_weightdecay import run_single

    axis, value = args.axis, float(args.value)
    if axis == "wd":
        wd, dsi = value, float(args.delay_std_init)
    else:  # std
        wd, dsi = 0.0, value
    subdir = _value_tag(axis, value)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_single(
        args.model, wd, int(args.seed), args.sweep_root, device,
        delay_std_init=dsi, round_pos=bool(int(args.round_pos)),
        epochs_override=args.epochs, batch_override=args.batch, subdir=subdir,
    )


def _worker_env(gpu, cache_dir):
    """Environment for a worker: pin the GPU, isolate the Triton cache, make the
    env interpreter's libstdc++ visible."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.makedirs(cache_dir, exist_ok=True)
    env["TRITON_CACHE_DIR"] = cache_dir
    lib = os.path.join(sys.prefix, "lib")
    env["LD_LIBRARY_PATH"] = lib + ":" + env.get("LD_LIBRARY_PATH", "")
    # Propagate the hardcoded kernel choice to the worker (which reads REC_FWD /
    # SYN_REC_FWD when it imports sweep_HAR_weightdecay). Shell env wins if set.
    if AXONAL_FORWARD_VERSION and "REC_FWD" not in env:
        env["REC_FWD"] = AXONAL_FORWARD_VERSION
    if SYN_FORWARD_VERSION and "SYN_REC_FWD" not in env:
        env["SYN_REC_FWD"] = SYN_FORWARD_VERSION
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
    max_per_gpu = int(os.environ.get("HAR_MAX_RUNS_PER_GPU", MAX_RUNS_PER_GPU))
    triples, params = build_triples()
    axis = params["axis"]
    sweep_root = params["sweep_root"]
    os.makedirs(sweep_root, exist_ok=True)
    cache_root = os.path.join(sweep_root, ".triton_cache")

    # One slot per (gpu, k). Each slot has a fixed Triton cache dir (safe: a slot
    # runs its jobs sequentially. Different slots -> different caches).
    slots = []
    for g in gpus:
        for k in range(max_per_gpu):
            slots.append({"gpu": g, "k": k, "proc": None, "run": None, "log": None,
                          "cache_dir": os.path.join(cache_root, f"gpu{g}_slot{k}")})

    print(f"GPUs: {gpus} | max runs/GPU: {max_per_gpu} | total slots: {len(slots)}")
    print(f"Axis: {axis} | Runs: {len(triples)}  (models={params['models']}, "
          f"values={params['values']}, seeds={params['seeds']}, "
          f"round_pos={params['round_pos']})")
    print(f"Sweep root: {sweep_root}\n")

    pending = list(triples)
    results = []  # {model, value, seed, gpu, rc, acc}
    launched = 0

    def launch(slot, run):
        nonlocal launched
        model, value, seed = run
        run_dir = os.path.join(sweep_root, model, _value_tag(axis, value), f"seed{seed}")
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "run.log")
        logf = open(log_path, "w")
        cmd = [sys.executable, os.path.abspath(__file__), "--run-one",
               "--axis", axis, "--model", model, "--value", repr(value),
               "--seed", str(seed), "--sweep-root", sweep_root,
               "--delay-std-init", repr(params["delay_std_init"]),
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
        print(f"[{time.strftime('%H:%M:%S')}] launch #{launched}/{len(triples)} "
              f"gpu{slot['gpu']} slot{slot['k']}: {model} {_value_tag(axis, value)} seed{seed} "
              f"-> {log_path}", flush=True)

    def reap(slot):
        proc, run = slot["proc"], slot["run"]
        rc = proc.returncode
        slot["log"].close()
        model, value, seed = run
        acc = None
        ft = os.path.join(sweep_root, model, _value_tag(axis, value), f"seed{seed}",
                          "final_test.json")
        if rc == 0 and os.path.exists(ft):
            try:
                acc = float(json.load(open(ft)).get("acc"))
            except Exception:
                pass
        status = "ok" if rc == 0 else f"FAILED(rc={rc})"
        print(f"[{time.strftime('%H:%M:%S')}] done   gpu{slot['gpu']} slot{slot['k']}: "
              f"{model} {_value_tag(axis, value)} seed{seed}  {status}"
              + (f"  acc={acc:.4f}" if acc is not None else ""), flush=True)
        results.append({"model": model, "value": value, "seed": seed,
                        "gpu": slot["gpu"], "rc": rc, "final_test_acc": acc})
        slot.update(proc=None, run=None, log=None)
        _write_manifest(sweep_root, params, results)

    print_epochs = os.environ.get("HAR_PRINT_EPOCH_PROGRESS",
                                  "1" if PRINT_EPOCH_PROGRESS else "0").lower() \
        in ("1", "true", "yes")

    def maybe_print_epoch(slot):
        info = _latest_epoch(os.path.join(_run_dir(sweep_root, axis, slot["run"]), "run.log"))
        if info is None:
            return
        e, E, acc, best = info
        if slot.get("last_epoch") == e:
            return
        slot["last_epoch"] = e
        model, value, seed = slot["run"]
        print(f"[{time.strftime('%H:%M:%S')}] gpu{slot['gpu']} s{slot['k']} "
              f"{model} {_value_tag(axis, value)} seed{seed}  epoch {e}/{E}  "
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
              f"{os.path.join(sweep_root, 'manifest.json')}", flush=True)
        raise SystemExit(130)

    _write_manifest(sweep_root, params, results)
    _summarize(results, params)


def _write_manifest(sweep_root, params, results):
    axis = params["axis"]
    manifest = {
        "sweep_root": sweep_root,
        "axis": axis,
        "models": params["models"],
        "delay_std_init": params["delay_std_init"],
        "values": params["values"],
        "seeds": params["seeds"],
        "round_pos": params["round_pos"],
        "runs": [{"model": r["model"], "axis": axis, "value": r["value"],
                  "seed": r["seed"], "gpu": r["gpu"], "rc": r["rc"],
                  "final_test_acc": r["final_test_acc"],
                  "run_dir": os.path.join(r["model"], _value_tag(axis, r["value"]),
                                          f"seed{r['seed']}")}
                 for r in results],
    }
    with open(os.path.join(sweep_root, "manifest.json"), "w") as fid:
        json.dump(manifest, fid, indent=2)


def _summarize(results, params):
    import statistics as stats
    axis = params["axis"]
    print("\n==================== SWEEP SUMMARY ====================")
    n_fail = sum(1 for r in results if r["rc"] != 0)
    group = {}
    for r in results:
        if r["final_test_acc"] is not None:
            group.setdefault((r["model"], r["value"]), []).append(r["final_test_acc"])
    for (model, value), accs in sorted(group.items()):
        mu = stats.mean(accs)
        sd = stats.pstdev(accs) if len(accs) > 1 else 0.0
        print(f"{model:34s} {_value_tag(axis, value):>8s} "
              f"test acc = {mu:.4f} +/- {sd:.4f}  (n={len(accs)})")
    if n_fail:
        print(f"\n[warn] {n_fail} run(s) FAILED, see per-run run.log files.")
    print(f"\nManifest: {os.path.join(params['sweep_root'], 'manifest.json')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-one", action="store_true",
                    help="internal: train exactly one run on the visible GPU")
    ap.add_argument("--axis", default="wd", choices=["wd", "std"])
    ap.add_argument("--model")
    ap.add_argument("--value")
    ap.add_argument("--seed")
    ap.add_argument("--sweep-root")
    ap.add_argument("--delay-std-init", default="8")
    ap.add_argument("--round-pos", default="1")
    ap.add_argument("--epochs", default="")
    ap.add_argument("--batch", default="")
    args = ap.parse_args()

    if args.run_one:
        worker_main(args)
    else:
        orchestrate(args)
