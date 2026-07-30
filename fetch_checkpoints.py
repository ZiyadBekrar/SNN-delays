"""Download the trained weights and unpack them into trained_models/.

The repository ships each run's derived artifacts but not its ``best.pth``, 385 MB
across 691 runs being more than belongs in git. The archives unpack into the same run
directories, so a run looks identical whether or not the weights have been fetched.
Only the figures that run a model forward need them.

Usage:    python fetch_checkpoints.py --list
          python fetch_checkpoints.py --dataset ssc
          python fetch_checkpoints.py --all
"""

import argparse
import hashlib
import sys
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "trained_models"
SUMS_FILE = ROOT / "docs" / "checksums.sha256"

# The Zenodo record holding the weights, cited in the docs as 10.5281/zenodo.21704323.
# That DOI always resolves to the newest deposit, so the download pins one specific
# version instead: the checksums in docs/checksums.sha256 describe one version's
# archives, and only those bytes will verify. See docs/checkpoints.md.
ZENODO_RECORD = "21704324"
RECORD_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

# name: (archive, compressed download in MB, contents). The unpacked best.pth are
# larger: 107, 207, 43, 15 and 13 MB respectively, 385 MB in total.
ARCHIVES = {
    "ssc":     ("delrec-checkpoints-ssc.tar.gz",      93, "23 runs: 4 delay families x 5 seeds, plus 3 feedforward-delay baselines"),
    "har":     ("delrec-checkpoints-har.tar.gz",     171, "426 runs: the delay-init, weight-decay, spike-penalty and rounding sweeps"),
    "psmnist": ("delrec-checkpoints-psmnist.tar.gz",  37, "20 runs: 4 delay families x 5 seeds"),
    "al":      ("delrec-checkpoints-al.tar.gz",       12, "30 runs: 4 delay families x 5 seeds, plus the rounding sweep"),
    "mg":      ("delrec-checkpoints-mg.tar.gz",       12, "192 runs: 4 families x 4 tau x 4 horizons x 3 seeds"),
}


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def expected_sums():
    """{filename: sha256} from the checksums file recorded alongside this script."""
    local = SUMS_FILE
    if local.exists():
        return dict(reversed(line.split()) for line in
                    local.read_text().splitlines() if line.strip())
    return {}


def _safe_members(tf):
    """Members that cannot escape the extraction root.

    Rejects absolute paths, drive letters and any ``..`` component, rather than only a
    leading one: ``a/../../etc`` starts with neither "/" nor ".." but still escapes.
    """
    safe = []
    for m in tf.getmembers():
        parts = PurePosixPath(m.name).parts
        if m.name.startswith("/") or ".." in parts or PurePosixPath(m.name).is_absolute():
            sys.exit(f"  refusing to unpack {tf.name}: unsafe member path {m.name!r}")
        if m.issym() or m.islnk():
            sys.exit(f"  refusing to unpack {tf.name}: link member {m.name!r}")
        safe.append(m)
    return safe


def fetch(name):
    fn, mb, _ = ARCHIVES[name]
    dest = ROOT / fn
    if not dest.exists():
        url = f"{RECORD_URL}/{fn}?download=1"
        print(f"  downloading {fn} (~{mb} MB) from {url}")
        urllib.request.urlretrieve(url, dest)

    # The checksums ship in this repository, so an unknown one means a truncated
    # download, a tampered archive or an incomplete clone. None of those should be
    # unpacked silently.
    want = expected_sums().get(fn)
    if not want:
        sys.exit(f"  no checksum recorded for {fn} in {SUMS_FILE.name}; refusing to unpack")
    got = sha256(dest)
    if got != want:
        sys.exit(f"  checksum mismatch for {fn}\n    expected {want}\n    got      {got}")
    print("  checksum OK")

    print(f"  unpacking into {RESULTS.relative_to(ROOT)}/")
    with tarfile.open(dest) as tf:
        tf.extractall(RESULTS, members=_safe_members(tf))
    n = len(list((RESULTS / name.upper()).rglob("best.pth"))) if (RESULTS / name.upper()).exists() else 0
    print(f"  {name}: {n} checkpoints in place")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(ARCHIVES), action="append", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list or not (args.dataset or args.all):
        total = sum(mb for _, mb, _ in ARCHIVES.values())
        print(f"{'dataset':9s} {'download':>11s}  contents")
        for k, (_, mb, what) in ARCHIVES.items():
            here = len(list((RESULTS / k.upper()).rglob("best.pth"))) if (RESULTS / k.upper()).exists() else 0
            print(f"{k:9s} {mb:5d} MB  {what}" + (f"   [{here} already present]" if here else ""))
        print(f"{'all':9s} {total:5d} MB")
        return

    for name in (sorted(ARCHIVES) if args.all else args.dataset):
        print(f"\n{name}:")
        fetch(name)


if __name__ == "__main__":
    main()
