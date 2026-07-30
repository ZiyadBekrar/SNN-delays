"""Helper for the top-level figure scripts.

Each script in this directory is a thin front end: it documents what it produces and
what that needs, then invokes the per-benchmark analysis modules where the work lives.
Those modules can still be run directly with their own flags.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def run(script, *args):
    """Run one analysis module from the repository root, as if from a shell."""
    cmd = [sys.executable, str(script), *map(str, args)]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def main(*steps):
    """Run each (script, *args) step in order."""
    for step in steps:
        run(*(step if isinstance(step, (tuple, list)) else (step,)))
