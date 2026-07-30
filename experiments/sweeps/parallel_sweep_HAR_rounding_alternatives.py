"""Run the rounding alternatives sweep, one process per run across the GPUs.

Sweeps the initialization width x the discretization scheme.

Each worker trains exactly one (model, value, seed) and writes the same artifacts
as a single training run. Each GPU slot gets its own Triton cache directory, so
concurrent just-in-time compilations cannot race.

Usage:    python experiments/sweeps/parallel_sweep_HAR_rounding_alternatives.py
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

# Repo-root on sys.path so the worker's `from sweep_HAR_rounding_alternatives import ...`
# (and that module's `configs`/`delrec` imports) resolve when run as
# `python experiments/sweeps/parallel_sweep_HAR_rounding_alternatives.py`.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src")]


# ============================ USER CONFIG (hardcode here) ==================== #
# GPUS: None -> auto-detect every GPU on the box, or force a subset, e.g. [1] to
#       use only GPU 1, or [0, 1] to use both. (Env HAR_GPUS="1" overrides.)
GPUS = None
# MAX_RUNS_PER_GPU: how many training runs may execute concurrently on one GPU.
#       (Env HAR_MAX_RUNS_PER_GPU overrides.) Synaptic (N,N) delays + Triton are
#       memory-heavy, so 1 is the safe default.
MAX_RUNS_PER_GPU = 2
# PRINT_EPOCH_PROGRESS: also echo a compact per-epoch line per running run to this
#       terminal (full detail always stays in each run's run.log). Set False for
#       launch/done lines only. (Env HAR_PRINT_EPOCH_PROGRESS=0 overrides.)
PRINT_EPOCH_PROGRESS = True
# ============================================================================ #

POLL_SECONDS = 3.0

# Matches the worker's per-epoch line, e.g.
#   Val Epoch: [12/100], lr: 0.001500, lr_pos: 0.050000, acc: 78.3456, best: 79.1234
_EPOCH_RE = re.compile(r"Val Epoch: \[(\d+)/(\d+)\].*?acc: ([\d.]+), best: ([\d.]+)")

# Kept in sync with sweep_HAR_rounding_alternatives.MODES (imported lazily in the
# worker. The launcher only needs the names for validation).
KNOWN_MODES = ('ste', 'stochastic', 'nearest')


def _mode_tag(mode, std):
    """Subdir tag for a (mode, delay_std_init) cell: ste_std8, stochastic_std13, ...
    (matches sweep_HAR_rounding_alternatives.mode_tag)."""
    return f"{mode}_std{std:g}"


def _run_dir(sweep_root, run):
    model, mode, std, seed = run
    return os.path.join(sweep_root, model, _mode_tag(mode, std), f"seed{seed}")


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


def build_quads():
    """(model, mode, std, seed) list + sweep params. Already-finished runs (a
    final_test.json in the run dir) are dropped unless HAR_OVERWRITE=1."""
    seeds = [int(s) for s in os.environ.get("HAR_SEEDS", "0,1,2").split(",")]
    epochs = os.environ.get("HAR_EPOCHS", "")
    batch = os.environ.get("HAR_BATCH_SIZE", "")

    values = [int(s) for s in os.environ.get(
        "HAR_DELAY_STD_INITS", "3,8,13,18,23").split(",")]
    models = [s for s in os.environ.get(
        "HAR_MODELS", "SNN_recurrent_delays,SNN_synaptic_recurrent_delays").split(",") if s]
    modes = [s for s in os.environ.get("HAR_ROUND_MODES", "ste,stochastic").split(",") if s]
    sweep_root = os.environ.get("HAR_SWEEP_ROOT", './exp/HAR/rounding_alternatives_sweep')
    overwrite = os.environ.get("HAR_OVERWRITE", "0").lower() in ("1", "true", "yes")

    unknown = [m for m in modes if m not in KNOWN_MODES]
    if unknown:
        raise SystemExit(f"Unknown rounding mode(s) {unknown}, expected from {list(KNOWN_MODES)}.")

    quads = [(m, mo, v, s) for m in models for mo in modes for v in values for s in seeds]
    if not overwrite:
        done = [q for q in quads
                if os.path.exists(os.path.join(_run_dir(sweep_root, q), "final_test.json"))]
        if done:
            print(f"[skip] {len(done)} run(s) already have a final_test.json "
                  f"(set HAR_OVERWRITE=1 to retrain them):")
            for m, mo, v, s in done:
                print(f"       {m} {_mode_tag(mo, v)} seed{s}")
        quads = [q for q in quads if q not in done]

    params = dict(sweep_root=sweep_root, epochs=epochs, batch=batch,
                  models=models, modes=modes, values=values, seeds=seeds)
    return quads, params


def worker_main(args):
    """--run-one: train exactly one (model, mode, std, seed) run on the visible GPU."""
    import torch
    from sweep_HAR_rounding_alternatives import run_single

    std = int(args.value)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_single(
        args.model, std, int(args.seed), args.sweep_root, device,
        mode=args.mode,
        epochs_override=args.epochs, batch_override=args.batch,
        subdir=_mode_tag(args.mode, std),
    )


def _worker_env(gpu, cache_dir):
    """Environment for a worker: pin the GPU, isolate the (torch/inductor) cache,
    make the env interpreter's libstdc++ visible."""
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
    max_per_gpu = int(os.environ.get("HAR_MAX_RUNS_PER_GPU", MAX_RUNS_PER_GPU))
    quads, params = build_quads()
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
    print(f"Runs: {len(quads)}  (models={params['models']}, modes={params['modes']}, "
          f"std={params['values']}, seeds={params['seeds']})")
    print(f"Sweep root: {sweep_root}\n")

    pending = list(quads)
    results = []  # {model, mode, std, seed, gpu, rc, acc}
    launched = 0

    def launch(slot, run):
        nonlocal launched
        model, mode, std, seed = run
        run_dir = _run_dir(sweep_root, run)
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "run.log")
        logf = open(log_path, "w")
        cmd = [sys.executable, os.path.abspath(__file__), "--run-one",
               "--model", model, "--mode", mode, "--value", str(std),
               "--seed", str(seed), "--sweep-root", sweep_root,
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
        print(f"[{time.strftime('%H:%M:%S')}] launch #{launched}/{len(quads)} "
              f"gpu{slot['gpu']} slot{slot['k']}: {model} {_mode_tag(mode, std)} seed{seed} "
              f"-> {log_path}", flush=True)

    def reap(slot):
        proc, run = slot["proc"], slot["run"]
        rc = proc.returncode
        slot["log"].close()
        model, mode, std, seed = run
        acc = None
        ft = os.path.join(_run_dir(sweep_root, run), "final_test.json")
        if rc == 0 and os.path.exists(ft):
            try:
                acc = float(json.load(open(ft)).get("acc"))
            except Exception:
                pass
        status = "ok" if rc == 0 else f"FAILED(rc={rc})"
        print(f"[{time.strftime('%H:%M:%S')}] done   gpu{slot['gpu']} slot{slot['k']}: "
              f"{model} {_mode_tag(mode, std)} seed{seed}  {status}"
              + (f"  acc={acc:.4f}" if acc is not None else ""), flush=True)
        results.append({"model": model, "mode": mode, "std": std, "seed": seed,
                        "gpu": slot["gpu"], "rc": rc, "final_test_acc": acc})
        slot.update(proc=None, run=None, log=None)
        _write_manifest(sweep_root, params, results)

    print_epochs = os.environ.get("HAR_PRINT_EPOCH_PROGRESS",
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
        model, mode, std, seed = slot["run"]
        print(f"[{time.strftime('%H:%M:%S')}] gpu{slot['gpu']} s{slot['k']} "
              f"{model} {_mode_tag(mode, std)} seed{seed}  epoch {e}/{E}  "
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
              f"{os.path.join(sweep_root, 'manifest_rounding_alternatives.json')}", flush=True)
        raise SystemExit(130)

    _write_manifest(sweep_root, params, results)
    _summarize(results, params)


def _write_manifest(sweep_root, params, results):
    manifest = {
        "sweep_root": sweep_root,
        "models": params["models"],
        "round_modes": params["modes"],
        "delay_std_inits": params["values"],
        "seeds": params["seeds"],
        "runs": [{"model": r["model"], "mode": r["mode"],
                  "delay_std_init": r["std"], "seed": r["seed"],
                  "gpu": r["gpu"], "rc": r["rc"],
                  "final_test_acc": r["final_test_acc"],
                  "run_dir": os.path.join(r["model"], _mode_tag(r["mode"], r["std"]),
                                          f"seed{r['seed']}")}
                 for r in results],
    }
    with open(os.path.join(sweep_root, "manifest_rounding_alternatives.json"), "w") as fid:
        json.dump(manifest, fid, indent=2)


def _summarize(results, params):
    import statistics as stats
    print("\n==================== SWEEP SUMMARY ====================")
    n_fail = sum(1 for r in results if r["rc"] != 0)
    group = {}
    for r in results:
        if r["final_test_acc"] is not None:
            group.setdefault((r["model"], r["mode"], r["std"]), []).append(r["final_test_acc"])
    for (model, mode, std), accs in sorted(group.items()):
        mu = stats.mean(accs)
        sd = stats.pstdev(accs) if len(accs) > 1 else 0.0
        print(f"{model:34s} {_mode_tag(mode, std):>20s} "
              f"test acc = {mu:.4f} +/- {sd:.4f}  (n={len(accs)})")
    if n_fail:
        print(f"\n[warn] {n_fail} run(s) FAILED, see per-run run.log files.")
    print(f"\nManifest: {os.path.join(params['sweep_root'], 'manifest_rounding_alternatives.json')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-one", action="store_true",
                    help="internal: train exactly one run on the visible GPU")
    ap.add_argument("--model")
    ap.add_argument("--mode", default="ste")
    ap.add_argument("--value")
    ap.add_argument("--seed")
    ap.add_argument("--sweep-root")
    ap.add_argument("--epochs", default="")
    ap.add_argument("--batch", default="")
    args = ap.parse_args()

    if args.run_one:
        worker_main(args)
    else:
        orchestrate(args)
