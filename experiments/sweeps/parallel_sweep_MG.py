"""Run the MG sweep, one process per run across the GPUs.

Sweeps the chaos parameter x the prediction horizon.

Each worker trains exactly one (model, value, seed) and writes the same artifacts
as a single training run. Each GPU slot gets its own Triton cache directory, so
concurrent just-in-time compilations cannot race.

Usage:    python experiments/sweeps/parallel_sweep_MG.py
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

# Repo-root on sys.path so the worker's `from sweep_MG import ...`
# (and that module's `configs`/`delrec` imports) resolve when run as
# `python experiments/sweeps/parallel_sweep_MG.py`.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src")]


# ============================ USER CONFIG (hardcode here) ==================== #
# GPUS: None -> auto-detect every GPU on the box, or force a subset, e.g. [1] to
#       use only GPU 1, or [0, 1] to use both. (Env MG_GPUS="1" overrides.)
GPUS = None
# MAX_RUNS_PER_GPU: how many training runs may execute concurrently on one GPU.
#       (Env MG_MAX_RUNS_PER_GPU overrides.) MG is a small model (1x128 hidden,
#       T=150), so several runs fit comfortably on one card.
MAX_RUNS_PER_GPU = 10
# PRINT_EPOCH_PROGRESS: also echo a compact per-epoch line per running run to this
#       terminal (full detail always stays in each run's run.log). Set False for
#       launch/done lines only. (Env MG_PRINT_EPOCH_PROGRESS=0 overrides.)
PRINT_EPOCH_PROGRESS = True
# AXONAL_FORWARD_VERSION: recurrent kernel, propagated to every worker. None ->
#       library default ('triton_exact' on GPU, 'v2' on CPU. Both exact). Set to force
#       a kernel, e.g. 'eventdriven'. Shell env REC_FWD takes precedence over this.
AXONAL_FORWARD_VERSION = None

# MG_TAUS / MG_HORIZONS: the (system delay tau x prediction horizon H) grid.
#       (Env MG_TAUS / MG_HORIZONS override.)
MG_TAUS = [17, 30, 40, 50]
MG_HORIZONS = [20, 30, 40, 50] #[30, 50, 100]
MG_SEEDS = [0, 1, 2] #[0, 1, 2, 3, 4]
# MG_DELAY_TYPE: 'axonal' (per-neuron delays) or 'synaptic' (per-synapse). The learned
#       + fixed families follow this. Propagated to every worker (which reads it when it
#       imports sweep_MG). Shell env MG_DELAY_TYPE wins. NB: both types write the same
#       family dir names, so give each type its own MG_SWEEP_ROOT.
MG_DELAY_TYPE = "axonal"
# ============================================================================ #

POLL_SECONDS = 3.0

# Matches the worker's per-epoch line, e.g.
#   Val Epoch: [12/100], lr: 0.000500, lr_pos: 0.050000, nmse: 0.1234, best: 0.1200
_EPOCH_RE = re.compile(r"Val Epoch: \[(\d+)/(\d+)\].*?nmse: ([\d.]+), best: ([\d.]+)")


def _axis_tag(mg_tau, horizon):
    """Must match sweep_MG._axis_tag."""
    return f"tau{mg_tau:g}_H{horizon:g}"


def _run_dir(sweep_root, run):
    family, mg_tau, horizon, seed = run
    return os.path.join(sweep_root, family, _axis_tag(mg_tau, horizon), f"seed{seed}")


def _latest_epoch(run_log):
    """(epoch, epochs, val_nmse, best) from the last 'Val Epoch' line in a run.log,
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
    """Resolve the GPU list: env MG_GPUS > GPUS constant > auto (nvidia-smi -L)."""
    env = os.environ.get("MG_GPUS")
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


def build_runs():
    """(family, tau, horizon, seed) list + the sweep params, read like sweep_MG.py."""
    default_taus = ",".join(str(t) for t in MG_TAUS)
    taus = [float(s) for s in os.environ.get("MG_TAUS", default_taus).split(",")]
    default_horizons = ",".join(str(h) for h in MG_HORIZONS)
    horizons = [int(s) for s in os.environ.get("MG_HORIZONS", default_horizons).split(",")]
    default_seeds = ",".join(str(s) for s in MG_SEEDS)
    seeds = [int(s) for s in os.environ.get("MG_SEEDS", default_seeds).split(",")]
    families = [s for s in os.environ.get(
        "MG_FAMILIES",
        "learned_delays,learned_no_annealing,fixed_delays,no_delays").split(",") if s]
    sweep_root = os.environ.get("MG_SWEEP_ROOT", './exp/MG/mackey_glass_sweep')
    epochs = os.environ.get("MG_EPOCHS", "")
    batch = os.environ.get("MG_BATCH_SIZE", "")

    runs = [(f, t, h, s) for f in families for t in taus for h in horizons for s in seeds]
    params = dict(sweep_root=sweep_root, epochs=epochs, batch=batch,
                  families=families, taus=taus, horizons=horizons, seeds=seeds)
    return runs, params


def worker_main(args):
    """--run-one: train exactly one (family, tau, horizon, seed) run on the visible GPU."""
    import torch
    from sweep_MG import run_single

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_single(
        args.family, float(args.tau), int(args.horizon), int(args.seed),
        args.sweep_root, device,
        epochs_override=args.epochs, batch_override=args.batch,
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
    # Propagate the hardcoded kernel choice to the worker (which reads REC_FWD when it
    # imports sweep_MG). Shell env wins if set.
    if AXONAL_FORWARD_VERSION and "REC_FWD" not in env:
        env["REC_FWD"] = AXONAL_FORWARD_VERSION
    # Propagate the hardcoded delay type to the worker (read when it imports sweep_MG).
    if "MG_DELAY_TYPE" not in env:
        env["MG_DELAY_TYPE"] = MG_DELAY_TYPE
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
    max_per_gpu = int(os.environ.get("MG_MAX_RUNS_PER_GPU", MAX_RUNS_PER_GPU))
    runs, params = build_runs()
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
    print(f"Runs: {len(runs)}  (families={params['families']}, taus={params['taus']}, "
          f"horizons={params['horizons']}, seeds={params['seeds']})")
    print(f"Delay type: {os.environ.get('MG_DELAY_TYPE', MG_DELAY_TYPE)}")
    print(f"Sweep root: {sweep_root}\n")

    pending = list(runs)
    results = []  # {family, mg_tau, horizon, seed, gpu, rc, final_test_nmse}
    launched = 0

    def launch(slot, run):
        nonlocal launched
        family, mg_tau, horizon, seed = run
        run_dir = _run_dir(sweep_root, run)
        os.makedirs(run_dir, exist_ok=True)
        log_path = os.path.join(run_dir, "run.log")
        logf = open(log_path, "w")
        cmd = [sys.executable, os.path.abspath(__file__), "--run-one",
               "--family", family, "--tau", repr(mg_tau), "--horizon", str(horizon),
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
        print(f"[{time.strftime('%H:%M:%S')}] launch #{launched}/{len(runs)} "
              f"gpu{slot['gpu']} slot{slot['k']}: {family} {_axis_tag(mg_tau, horizon)} "
              f"seed{seed} -> {log_path}", flush=True)

    def reap(slot):
        proc, run = slot["proc"], slot["run"]
        rc = proc.returncode
        slot["log"].close()
        family, mg_tau, horizon, seed = run
        nmse = None
        ft = os.path.join(_run_dir(sweep_root, run), "final_test.json")
        if rc == 0 and os.path.exists(ft):
            try:
                nmse = float(json.load(open(ft))["test"]["nmse"])
            except Exception:
                pass
        status = "ok" if rc == 0 else f"FAILED(rc={rc})"
        print(f"[{time.strftime('%H:%M:%S')}] done   gpu{slot['gpu']} slot{slot['k']}: "
              f"{family} {_axis_tag(mg_tau, horizon)} seed{seed}  {status}"
              + (f"  nmse={nmse:.6f}" if nmse is not None else ""), flush=True)
        results.append({"family": family, "mg_tau": mg_tau, "horizon": horizon,
                        "seed": seed, "gpu": slot["gpu"], "rc": rc,
                        "final_test_nmse": nmse})
        slot.update(proc=None, run=None, log=None)
        _write_manifest(sweep_root, params, results)

    print_epochs = os.environ.get("MG_PRINT_EPOCH_PROGRESS",
                                  "1" if PRINT_EPOCH_PROGRESS else "0").lower() \
        in ("1", "true", "yes")

    def maybe_print_epoch(slot):
        info = _latest_epoch(os.path.join(_run_dir(sweep_root, slot["run"]), "run.log"))
        if info is None:
            return
        e, E, nmse, best = info
        if slot.get("last_epoch") == e:
            return
        slot["last_epoch"] = e
        family, mg_tau, horizon, seed = slot["run"]
        print(f"[{time.strftime('%H:%M:%S')}] gpu{slot['gpu']} s{slot['k']} "
              f"{family} {_axis_tag(mg_tau, horizon)} seed{seed}  epoch {e}/{E}  "
              f"val NMSE {nmse:.4f}  best {best:.4f}", flush=True)

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
    manifest = {
        "sweep_root": sweep_root,
        "families": params["families"],
        "taus": params["taus"],
        "horizons": params["horizons"],
        "seeds": params["seeds"],
        "runs": [{"family": r["family"], "mg_tau": r["mg_tau"], "horizon": r["horizon"],
                  "seed": r["seed"], "gpu": r["gpu"], "rc": r["rc"],
                  "final_test_nmse": r["final_test_nmse"],
                  "run_dir": os.path.join(r["family"],
                                          _axis_tag(r["mg_tau"], r["horizon"]),
                                          f"seed{r['seed']}")}
                 for r in results],
    }
    with open(os.path.join(sweep_root, "manifest.json"), "w") as fid:
        json.dump(manifest, fid, indent=2)


def _summarize(results, params):
    import statistics as stats
    print("\n==================== SWEEP SUMMARY ====================")
    n_fail = sum(1 for r in results if r["rc"] != 0)
    group = {}
    for r in results:
        if r["final_test_nmse"] is not None:
            group.setdefault((r["family"], r["mg_tau"], r["horizon"]), []).append(
                r["final_test_nmse"])
    for (family, mg_tau, horizon), nmses in sorted(group.items()):
        mu = stats.mean(nmses)
        sd = stats.pstdev(nmses) if len(nmses) > 1 else 0.0
        print(f"{family:22s} {_axis_tag(mg_tau, horizon):>14s} "
              f"test NMSE = {mu:.6f} +/- {sd:.6f}  (n={len(nmses)})")
    if n_fail:
        print(f"\n[warn] {n_fail} run(s) FAILED, see per-run run.log files.")
    print(f"\nManifest: {os.path.join(params['sweep_root'], 'manifest.json')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-one", action="store_true",
                    help="internal: train exactly one run on the visible GPU")
    ap.add_argument("--family")
    ap.add_argument("--tau")
    ap.add_argument("--horizon")
    ap.add_argument("--seed")
    ap.add_argument("--sweep-root")
    ap.add_argument("--epochs", default="")
    ap.add_argument("--batch", default="")
    args = ap.parse_args()

    if args.run_one:
        worker_main(args)
    else:
        orchestrate(args)
