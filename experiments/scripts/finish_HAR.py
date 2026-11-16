"""Complete HAR feedforward seeds and rerun learned recurrent hybrid after the fix.

Preview: python experiments/scripts/finish_HAR.py --dry-run
Run: python experiments/scripts/finish_HAR.py
Use the training environment's Python. Existing results are never overwritten.
"""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def completed(model, seed, search_root):
    for path in search_root.rglob("final_test.json"):
        try:
            result = json.loads(path.read_text())
            cfg = json.loads(path.with_name("config.json").read_text())
            if (result.get("model") == model and result.get("seed") == seed
                    and cfg.get("epochs") == 100
                    and math.isfinite(float(result["acc"]))):
                return path.parent
        except (ValueError, KeyError, TypeError, OSError):
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output = ROOT / "exp/HAR/completion" / f"{stamp}_{os.getpid()}"
    models = ["SNN_axonal_feedforward_delays", "SNN_synaptic_feedforward_delays",
              "SNN_hybrid_feedforward_delays", "SNN_recurrent_hybrid_delays"]
    for model in models:
        recurrent = model == "SNN_recurrent_hybrid_delays"
        for seed in (0, 1, 2):
            # Historical recurrent results precede the gradient fix: rerun them.
            done = None if recurrent else completed(model, seed, ROOT / "exp/HAR")
            if done:
                print(f"SKIP {model} seed{seed}: {done}", flush=True)
                continue
            out = output / model / f"seed{seed}"
            cmd = [sys.executable, "-u", "experiments/train.py", "--dataset", "har",
                   "--model", model, "--seeds", str(seed), "--epochs", "100",
                   "--out", str(out)]
            if recurrent:
                cmd += ["--init-rec-delay", "half_normal", "--delay-std-init", "3",
                        "--hybrid-offset-scale", "4", "--save-delay-history"]
            print(shlex.join(cmd), flush=True)
            if args.dry_run:
                continue
            out.mkdir(parents=True, exist_ok=False)
            with (out / "console.log").open("w") as log:
                with subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True) as proc:
                    for line in proc.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    code = proc.wait()
                if code:
                    raise SystemExit(code)


if __name__ == "__main__":
    main()
