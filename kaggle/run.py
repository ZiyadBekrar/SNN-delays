"""Kaggle GPU entrypoint for the push-triggered run.

This file is uploaded as a Kaggle *script* kernel by
``.github/workflows/kaggle-gpu.yml``. Script kernels accept no arguments, so the
workflow rewrites ``COMMIT`` below with ``sed`` before ``kaggle kernels push``.
It clones the repo at that commit, installs only the packages Kaggle's image
lacks (never torch), and runs ``experiments/compare_mem_delays.py`` on the GPU,
writing every artifact under ``/kaggle/working`` so the workflow can download it.
"""

import os
import subprocess
import sys

REPO_URL = "https://github.com/ZiyadBekrar/SNN-delays.git"
COMMIT = "__COMMIT_SHA__"          # rewritten by the workflow; unset -> default branch HEAD
CHECKOUT = "/tmp/snn-delays"
RESULTS = "/kaggle/working/delay_comparison"


def sh(*cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def main():
    # 1. Source at the exact commit under test.
    sh("git", "clone", REPO_URL, CHECKOUT)
    if COMMIT and not COMMIT.startswith("__"):
        sh("git", "-C", CHECKOUT, "checkout", "--detach", COMMIT)
    sh("git", "-C", CHECKOUT, "log", "-1", "--oneline")

    # 2. Dependencies. Kaggle's image ships a very new torch (2.10+cu128) whose
    #    binaries dropped Pascal (sm_60) kernels, and the GPU a pushed kernel is
    #    given is a Tesla P100 (sm_60): every CUDA launch then fails with "no
    #    kernel image is available for execution on the device". Pin the repo's
    #    own torch 2.5.1 / torchvision 0.20.1 - its cu121 wheels include sm_60,
    #    and this is the environment requirements.txt already specifies.
    sh(sys.executable, "-m", "pip", "install", "-q",
       "torch==2.5.1", "torchvision==0.20.1",
       "--index-url", "https://download.pytorch.org/whl/cu121")
    sh(sys.executable, "-m", "pip", "install", "-q",
       "DCLS==0.1.1",
       "spikingjelly @ git+https://github.com/fangwei123456/spikingjelly.git"
       "@d4fee3a1715bf42ce15f52fde1b1d4709d7ea25e",
       "opt_einsum", "prettytable")
    sh(sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", CHECKOUT)

    # 3. A usable GPU must actually be attached - and torch must have kernels for
    #    it. Do a real launch so a mismatch fails here, with a clear message,
    #    instead of mid-experiment.
    import torch
    print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
          "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no device",
          flush=True)
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA device - enable the GPU accelerator on this kernel.")
    try:
        (torch.ones(16, device="cuda") * 2).sum().item()
    except Exception as exc:
        raise SystemExit(f"GPU present but unusable: {exc}. torch {torch.__version__} "
                         f"has no kernels for {torch.cuda.get_device_name(0)}.")

    # 4. Run the comparison on the GPU.
    env = dict(os.environ, MPLBACKEND="Agg")
    sh(sys.executable, os.path.join(CHECKOUT, "experiments", "compare_mem_delays.py"),
       "--device", "cuda", "--out", RESULTS, env=env)

    print("\n=== comparison.json ===", flush=True)
    with open(os.path.join(RESULTS, "comparison.json")) as fh:
        print(fh.read(), flush=True)


if __name__ == "__main__":
    main()
