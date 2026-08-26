"""ctypes loader for the Mojo sieve kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess
from typing import Any

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJO_PRIMESIEVE_LIB") or os.path.join(
    ROOT, "dist", "libmojo-primesieve.so"
)

I = ctypes.c_int64
P = ctypes.c_void_p

_SIGNATURES = {
    "mps_small_sieve": ([I, P, I], I),
    "mps_segment_sieve": ([I, I, P, I, P, I], I),
    "mps_sieve_batch": ([I, I, P, I, P, I, P, I, I], I),
    "mps_collect_segment": ([I, P, I, P], I),
    "mps_collect_batch": ([I, P, I, I, P, I, P, I], I),
    "mps_select_flag": ([I, P, I, I], I),
    "mps_count_constellations": ([P, I, I], I),
    "mps_count_flag_constellations": ([P, I, I], I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    """Build the shared library when it is absent or older than its source."""
    source = os.path.join(ROOT, "src", "primesieve.mojo")
    if os.environ.get("MOJO_PRIMESIEVE_LIB") and os.path.exists(LIB) and not force:
        return LIB
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(source):
        return LIB
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def addr(array: Any, dtype: np.dtype, *, writable: bool = False) -> int:
    """Return a checked address for a one-dimensional native NumPy buffer."""
    if not isinstance(array, np.ndarray):
        raise TypeError("native buffers must be NumPy arrays")
    if array.ndim != 1 or not array.flags.c_contiguous:
        raise ValueError("native buffers must be one-dimensional and C-contiguous")
    if array.dtype != np.dtype(dtype) or not array.dtype.isnative:
        raise TypeError(f"native buffer must have dtype {np.dtype(dtype)}")
    if writable and not array.flags.writeable:
        raise ValueError("native output buffer must be writable")
    address = int(array.ctypes.data)
    if array.size and address == 0:
        raise ValueError("non-empty native buffer has a null address")
    return address


def checked(result: int, operation: str) -> int:
    """Turn a native validation failure into a visible Python exception."""
    if result < 0:
        raise RuntimeError(f"native {operation} rejected its arguments")
    return int(result)
