"""Run the 18 HAR experiments used for the six configurations of Table 8.

Preview commands:
    python experiments/scripts/run_HAR_table8.py --dry-run

Run all experiments:
    python experiments/scripts/run_HAR_table8.py

Every seed is launched separately because ``experiments/train.py --out``
designates one run directory. Results are grouped below one timestamped folder.
"""

import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]

MODELS = (
    "SNN_axonal_feedforward_delays",
    "SNN_hybrid_feedforward_delays",
    "SNN_synaptic_feedforward_delays",
    "SNN_axonal_recurrent_delays",
    "SNN_recurrent_hybrid_delays",
    "SNN_synaptic_recurrent_delays",
)

RECURRENT_MODELS = {
    "SNN_axonal_recurrent_delays",
    "SNN_recurrent_hybrid_delays",
    "SNN_synaptic_recurrent_delays",
}

HYBRID_MODELS = {
    "SNN_hybrid_feedforward_delays",
    "SNN_recurrent_hybrid_delays",
}


def build_command(model, seed, output):
    command = [
        sys.executable,
        "-u",
        "experiments/train.py",
        "--dataset",
        "har",
        "--model",
        model,
        "--seeds",
        str(seed),
        "--epochs",
        "100",
        "--out",
        str(output),
    ]

    if model in RECURRENT_MODELS:
        command += [
            "--init-rec-delay",
            "half_normal",
            "--delay-std-init",
            "3",
        ]

    if model in HYBRID_MODELS:
        command += ["--hybrid-offset-scale", "4"]

    return command


def run_command(command, output):
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "console.log"

    with log_path.open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()

            return_code = process.wait()

    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print every command without starting training",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="common output directory (default: exp/HAR/table8/<timestamp>_<pid>)",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="comma-separated training seeds (default: 0,1,2)",
    )
    args = parser.parse_args()

    try:
        seeds = [int(seed.strip()) for seed in args.seeds.split(",")]
    except ValueError as error:
        parser.error(f"invalid --seeds value: {error}")

    if not seeds:
        parser.error("--seeds must contain at least one seed")

    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_root = args.output or ROOT / "exp" / "HAR" / "table8" / f"{stamp}_{os.getpid()}"
    if not output_root.is_absolute():
        output_root = ROOT / output_root

    jobs = []
    for model in MODELS:
        for seed in seeds:
            output = output_root / model / f"seed{seed}"
            command = build_command(model, seed, output)
            jobs.append({
                "model": model,
                "seed": seed,
                "output": str(output),
                "command": command,
            })

    print(f"Output root: {output_root}")
    print(f"Runs: {len(jobs)} ({len(MODELS)} models x {len(seeds)} seeds)")

    for index, job in enumerate(jobs, start=1):
        print(f"\n[{index}/{len(jobs)}] {shlex.join(job['command'])}", flush=True)
        if not args.dry_run:
            run_command(job["command"], Path(job["output"]))

    if not args.dry_run:
        manifest = {
            "dataset": "har",
            "epochs": 100,
            "models": list(MODELS),
            "seeds": seeds,
            "jobs": jobs,
        }
        (output_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        print(f"\nAll runs completed: {output_root}")


if __name__ == "__main__":
    main()
