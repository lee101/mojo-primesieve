"""NumPy-returning variants of the prime generation functions."""

from __future__ import annotations

from ._api import _as_nonnegative, _bounds, _first_n_numpy, _numpy_primes


def primes(from_limit, to_limit=0):
    start, stop = _bounds(from_limit, to_limit)
    return _numpy_primes(start, stop).astype("uint64", copy=False)


def n_primes(n, start=0):
    n = _as_nonnegative(n, "n")
    start = _as_nonnegative(start, "start")
    return _first_n_numpy(n, start).astype("uint64", copy=False)


__all__ = ["n_primes", "primes"]
