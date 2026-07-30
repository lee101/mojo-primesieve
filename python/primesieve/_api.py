"""Python-compatible API backed by an odd-only segmented Mojo sieve."""

from __future__ import annotations

import array
import math
import os
import operator
import sys
from collections.abc import Iterator as IteratorABC
from functools import lru_cache

import numpy as np

from ._lib import addr, build, checked, lib

arr_type = "Q"
_sieve_size_kib = 1024
_segment_cache: dict[int, np.ndarray] = {}
_MAX_STOP = sys.maxsize
_BATCH_SEGMENTS = 32
_PARALLEL_MIN_ODDS = 4_000_000
_num_threads = os.cpu_count() or 1
_FLAG_PATTERNS = {
    2: ((0, 1),),
    3: ((0, 1, 3), (0, 2, 3)),
    4: ((0, 1, 3, 4),),
    5: ((0, 1, 3, 4, 6), (0, 2, 3, 5, 6)),
    6: ((0, 2, 3, 5, 6, 8),),
}


def _as_nonnegative(value, name: str) -> int:
    try:
        value = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer") from None
    if value < 0:
        raise OverflowError(f"{name} cannot be negative")
    if value > _MAX_STOP:
        raise OverflowError(f"{name} exceeds this port's signed 64-bit limit")
    return value


def _bounds(from_limit, to_limit=0) -> tuple[int, int]:
    start = _as_nonnegative(from_limit, "from_limit")
    stop = _as_nonnegative(to_limit, "to_limit")
    return (0, start) if stop == 0 else (start, stop)


@lru_cache(maxsize=8)
def _small_primes(stop: int) -> np.ndarray:
    if stop < 2:
        return np.empty(0, dtype=np.int64)
    odd_count = max(0, (stop - 1) // 2)
    if odd_count == 0:
        return np.array([2], dtype=np.int64)
    flags = np.empty(odd_count, dtype=np.uint8)
    checked(
        lib().mps_small_sieve(
            stop, addr(flags, np.uint8, writable=True), odd_count
        ),
        "small sieve",
    )
    odds = 2 * np.flatnonzero(flags).astype(np.int64) + 3
    result = np.empty(odds.size + 1, dtype=np.int64)
    result[0] = 2
    result[1:] = odds
    return result


def _scratch(required: int) -> np.ndarray:
    if _segment_cache:
        scratch = next(iter(_segment_cache.values()))
        if scratch.size >= required:
            return scratch[:required]
    scratch = np.empty(required, dtype=np.uint8)
    _segment_cache.clear()
    _segment_cache[required] = scratch
    return scratch


def _should_parallel(n: int, segment_capacity: int, nsegments: int) -> bool:
    return n >= max(_PARALLEL_MIN_ODDS, 4 * segment_capacity) and nsegments >= 4


def _parallel_workers(n: int, segment_capacity: int, nsegments: int) -> int:
    if _num_threads > 1 and _should_parallel(n, segment_capacity, nsegments):
        return min(_num_threads, nsegments)
    return 1


def _sieve_batches(start: int, stop: int):
    low = max(3, start)
    if low % 2 == 0:
        low += 1
    if low > stop:
        return
    base = _small_primes(math.isqrt(stop))
    segment_capacity = max(512, _sieve_size_kib * 1024)
    batch_capacity = segment_capacity * _BATCH_SEGMENTS
    while low <= stop:
        n = min(batch_capacity, (stop - low) // 2 + 1)
        flags = _scratch(n)
        nsegments = (n + segment_capacity - 1) // segment_capacity
        counts = np.empty(nsegments, dtype=np.int64)
        checked(lib().mps_sieve_batch(
            low,
            low + 2 * (n - 1),
            addr(base, np.int64),
            base.size,
            addr(flags, np.uint8, writable=True),
            segment_capacity,
            addr(counts, np.int64, writable=True),
            nsegments,
            _parallel_workers(n, segment_capacity, nsegments),
        ), "batch sieve")
        yield low, flags, counts, segment_capacity
        low += 2 * n


def _odd_segments(start: int, stop: int):
    for batch_low, flags, counts, segment_capacity in _sieve_batches(start, stop):
        for segment, count in enumerate(counts):
            offset = segment * segment_capacity
            n = min(segment_capacity, flags.size - offset)
            yield batch_low + 2 * offset, flags[offset : offset + n], int(count)


def _numpy_primes(start: int, stop: int) -> np.ndarray:
    if stop < start or stop < 2:
        return np.empty(0, dtype=np.int64)
    chunks: list[np.ndarray] = []
    if start <= 2 <= stop:
        chunks.append(np.array([2], dtype=np.int64))
    for batch_low, flags, counts, segment_capacity in _sieve_batches(start, stop):
        values = np.empty(int(counts.sum()), dtype=np.int64)
        offsets = np.empty(counts.size, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts[:-1], out=offsets[1:])
        checked(lib().mps_collect_batch(
            batch_low,
            addr(flags, np.uint8),
            flags.size,
            segment_capacity,
            addr(offsets, np.int64),
            counts.size,
            addr(values, np.int64, writable=True),
            _parallel_workers(flags.size, segment_capacity, counts.size),
        ), "batch collection")
        chunks.append(values)
    if not chunks:
        return np.empty(0, dtype=np.int64)
    if len(chunks) == 1:
        return chunks[0]
    return np.concatenate(chunks)


def primes(from_limit, to_limit=0):
    """Generate an ``array('Q')`` of primes in the inclusive interval."""
    start, stop = _bounds(from_limit, to_limit)
    result = array.array(arr_type)
    result.frombytes(memoryview(_numpy_primes(start, stop)).cast("B"))
    return result


def _nth_upper_bound(n: int) -> int:
    if n < 6:
        return 13
    x = float(n)
    return math.ceil(x * (math.log(x) + math.log(math.log(x)))) + 3


def _first_n_numpy(n: int, start: int = 0) -> np.ndarray:
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if start <= 2:
        stop = _nth_upper_bound(n)
    else:
        stop = min(_MAX_STOP, start + max(1024, math.ceil(1.4 * n * math.log(start + n))))
    while True:
        values = _numpy_primes(start, stop)
        if values.size >= n:
            return values[:n].copy()
        if stop == _MAX_STOP:
            raise OverflowError("not enough representable primes")
        span = max(1024, stop - start + 1)
        stop = min(_MAX_STOP, stop + 2 * span)


def n_primes(n, start=0):
    """Return the first ``n`` primes greater than or equal to ``start``."""
    n = _as_nonnegative(n, "n")
    start = _as_nonnegative(start, "start")
    result = array.array(arr_type)
    result.frombytes(memoryview(_first_n_numpy(n, start)).cast("B"))
    return result


def nth_prime(n, start=0):
    """Find the nth prime after ``start``; negative n searches backwards."""
    try:
        n = operator.index(n)
    except TypeError:
        raise TypeError("n must be an integer") from None
    start = _as_nonnegative(start, "start")
    if n >= 0:
        rank = max(1, n)
        return int(_first_n_numpy(rank, min(_MAX_STOP, start + 1))[-1])
    need = -n
    if start <= 2:
        raise RuntimeError("nth prime < 2 is impossible")
    span = max(1024, int(need * max(4.0, math.log(start))))
    while True:
        low = max(0, start - span)
        values = _numpy_primes(low, start - 1)
        if values.size >= need:
            return int(values[-need])
        if low == 0:
            raise RuntimeError("nth prime < 2 is impossible")
        span *= 2


def count_primes(from_limit, to_limit=0):
    """Count primes in the inclusive interval."""
    start, stop = _bounds(from_limit, to_limit)
    if stop < start or stop < 2:
        return 0
    total = int(start <= 2 <= stop)
    for _, _, counts, _ in _sieve_batches(start, stop):
        total += int(counts.sum())
    return total


def _constellation_count(kind: int, from_limit, to_limit=0) -> int:
    start, stop = _bounds(from_limit, to_limit)
    patterns = _FLAG_PATTERNS[kind]
    span = patterns[0][-1]
    total = 0
    previous_tail = None
    for _, flags, _, _ in _sieve_batches(start, stop):
        total += checked(
            lib().mps_count_flag_constellations(
                addr(flags, np.uint8), flags.size, kind
            ),
            "constellation count",
        )
        if previous_tail is not None:
            first = flags[:span]
            boundary = previous_tail.size
            combined = np.concatenate((previous_tail, first))
            for offset in range(max(0, boundary - span), boundary):
                if offset + span >= combined.size:
                    continue
                if any(
                    all(combined[offset + delta] != 0 for delta in pattern)
                    for pattern in patterns
                ):
                    total += 1
        previous_tail = flags[-span:].copy()
    return total


def count_twins(from_limit, to_limit=0):
    return _constellation_count(2, from_limit, to_limit)


def count_triplets(from_limit, to_limit=0):
    return _constellation_count(3, from_limit, to_limit)


def count_quadruplets(from_limit, to_limit=0):
    return _constellation_count(4, from_limit, to_limit)


def count_quintuplets(from_limit, to_limit=0):
    return _constellation_count(5, from_limit, to_limit)


def count_sextuplets(from_limit, to_limit=0):
    return _constellation_count(6, from_limit, to_limit)


def _constellations(kind: int, from_limit, to_limit=0):
    start, stop = _bounds(from_limit, to_limit)
    values = _numpy_primes(start, stop)
    patterns = {
        2: {(2,)},
        3: {(2, 4), (4, 2)},
        4: {(2, 4, 2)},
        5: {(2, 4, 2, 4), (4, 2, 4, 2)},
        6: {(4, 2, 4, 2, 4)},
    }
    for i in range(kind - 1, values.size):
        window = values[i - kind + 1 : i + 1]
        if tuple(np.diff(window)) in patterns[kind]:
            yield tuple(map(int, window))


def _print(kind: int, from_limit, to_limit=0) -> None:
    if kind == 1:
        for prime in primes(from_limit, to_limit):
            print(prime)
    else:
        for values in _constellations(kind, from_limit, to_limit):
            print(values)


def print_primes(from_limit, to_limit=0):
    _print(1, from_limit, to_limit)


def print_twins(from_limit, to_limit=0):
    _print(2, from_limit, to_limit)


def print_triplets(from_limit, to_limit=0):
    _print(3, from_limit, to_limit)


def print_quadruplets(from_limit, to_limit=0):
    _print(4, from_limit, to_limit)


def print_quintuplets(from_limit, to_limit=0):
    _print(5, from_limit, to_limit)


def print_sextuplets(from_limit, to_limit=0):
    _print(6, from_limit, to_limit)


def get_max_stop():
    return _MAX_STOP


def get_sieve_size():
    return _sieve_size_kib


def set_sieve_size(sieve_size):
    global _sieve_size_kib
    try:
        size = operator.index(sieve_size)
    except TypeError:
        raise TypeError("sieve_size must be an integer") from None
    if not 16 <= size <= 4096:
        raise ValueError("sieve_size must be between 16 and 4096 KiB")
    _sieve_size_kib = size
    _segment_cache.clear()


def get_num_threads():
    return _num_threads


def set_num_threads(threads):
    global _num_threads
    try:
        threads = operator.index(threads)
    except TypeError:
        raise TypeError("threads must be an integer") from None
    if threads < 1:
        raise ValueError("threads must be at least 1")
    _num_threads = threads


def primesieve_version():
    return "mojo-primesieve 0.1.0"


class Iterator(IteratorABC):
    """Bidirectional prime iterator compatible with ``primesieve.Iterator``."""

    def __init__(self):
        self._position = 0

    def skipto(self, start, stop_hint=2**62):
        del stop_hint
        self._position = _as_nonnegative(start, "start")

    def next_prime(self):
        value = nth_prime(1, self._position)
        self._position = value
        return value

    def prev_prime(self):
        value = nth_prime(-1, self._position)
        self._position = value
        return value

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_prime()


__all__ = [
    "Iterator",
    "arr_type",
    "build",
    "count_primes",
    "count_quadruplets",
    "count_quintuplets",
    "count_sextuplets",
    "count_triplets",
    "count_twins",
    "get_max_stop",
    "get_num_threads",
    "get_sieve_size",
    "n_primes",
    "nth_prime",
    "primes",
    "primesieve_version",
    "print_primes",
    "print_quadruplets",
    "print_quintuplets",
    "print_sextuplets",
    "print_triplets",
    "print_twins",
    "set_num_threads",
    "set_sieve_size",
]
