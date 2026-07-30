"""Make Triton's CUDA-12 kernels loadable on hosts with an older NVIDIA driver.

Triton wheels bundle a CUDA-12 ``ptxas``, which emits cubin that a pre-CUDA-12
driver cannot load -> the kernel launch fails with
``Triton Error [CUDA]: device kernel image is invalid`` (seen on an A40 running
driver 470 / CUDA 11.4). If the running driver is older than CUDA 12 and a
compatible system ``ptxas`` exists (e.g. ``/usr/local/cuda-11.4/bin/ptxas``),
we point Triton at it via ``TRITON_PTXAS_PATH`` so it emits cubin the driver can
load and derives a matching PTX ISA version.

No-op when: ``TRITON_PTXAS_PATH`` is already set, there is no CUDA driver, the
driver is already >= CUDA 12, or no compatible system ptxas is found.
"""
import ctypes
import glob
import os
import re
import subprocess


def _driver_cuda_version():
    """CUDA version the installed NVIDIA driver supports as an int (11040 == 11.4),
    via ``cuDriverGetVersion``. None if libcuda is unavailable."""
    for lib in ("libcuda.so.1", "libcuda.so"):
        try:
            libcuda = ctypes.CDLL(lib)
        except OSError:
            continue
        try:
            if libcuda.cuInit(0) != 0:
                return None
            v = ctypes.c_int()
            if libcuda.cuDriverGetVersion(ctypes.byref(v)) != 0:
                return None
            return int(v.value)
        except Exception:
            return None
    return None


def _ptxas_cuda_version(path):
    """CUDA version reported by ``<path> --version`` as an int (11040 == 11.4)."""
    try:
        out = subprocess.check_output([path, "--version"], text=True,
                                      stderr=subprocess.STDOUT)
    except Exception:
        return None
    m = re.search(r"release (\d+)\.(\d+)", out)
    return int(m.group(1)) * 1000 + int(m.group(2)) * 10 if m else None


def configure_ptxas_for_driver(verbose=False):
    """Set ``TRITON_PTXAS_PATH`` to a driver-compatible system ptxas when needed.

    Returns the chosen ptxas path, or None if no change was made.
    """
    if os.environ.get("TRITON_PTXAS_PATH"):
        return None
    driver = _driver_cuda_version()
    if driver is None or driver >= 12000:   # no driver, or new enough for Triton's ptxas
        return None
    best = None  # newest system ptxas whose CUDA version the driver still supports
    for cand in sorted(glob.glob("/usr/local/cuda*/bin/ptxas")):
        ver = _ptxas_cuda_version(cand)
        if ver is not None and ver <= driver and (best is None or ver > best[1]):
            best = (cand, ver)
    if best is None:
        if verbose:
            print(f"[delrec.triton_kernels] driver CUDA {driver / 1000:.1f} < 12.0 but no "
                  f"compatible system ptxas found; Triton kernels may fail to load.")
        return None
    os.environ["TRITON_PTXAS_PATH"] = best[0]
    if verbose:
        print(f"[delrec.triton_kernels] driver CUDA {driver / 1000:.1f}: using ptxas {best[0]} "
              f"(CUDA {best[1] / 1000:.1f}) so Triton emits loadable cubin.")
    return best[0]
