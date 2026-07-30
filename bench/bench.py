"""Locked, direct benchmarks against the upstream Python package."""

from __future__ import annotations

import importlib
import math
import os
import platform
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_PYTHON = os.path.join(ROOT, "python")


def load_packages():
    saved_path = sys.path[:]
    blocked = {os.path.abspath(LOCAL_PYTHON), os.path.abspath(ROOT)}
    sys.path[:] = [
        path
        for path in sys.path
        if os.path.abspath(path or os.getcwd()) not in blocked
    ]
    for name in list(sys.modules):
        if name == "primesieve" or name.startswith("primesieve."):
            del sys.modules[name]
    upstream = importlib.import_module("primesieve")
    for name in list(sys.modules):
        if name == "primesieve" or name.startswith("primesieve."):
            del sys.modules[name]
    sys.path[:] = saved_path
    mojo = importlib.import_module("primesieve")
    return mojo, upstream


MOJO, UPSTREAM = load_packages()


def best_time(function, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


CASES = [
    ("primes(10,000,000)", lambda package: package.primes(10_000_000)),
    ("primes(10^12, 10^12 + 5M)", lambda package: package.primes(10**12, 10**12 + 5_000_000)),
    ("count_primes(100,000,000)", lambda package: package.count_primes(100_000_000)),
    ("count_twins(10,000,000)", lambda package: package.count_twins(10_000_000)),
    ("nth_prime(100,000)", lambda package: package.nth_prime(100_000)),
]


def cpu_model():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    print(f"Machine: {cpu_model()}, {os.cpu_count()} logical CPUs")
    print()
    print("| Operation | Mojo | upstream primesieve | upstream / Mojo |")
    print("|---|---:|---:|---:|")
    for name, operation in CASES:
        mojo_call = lambda: operation(MOJO)
        upstream_call = lambda: operation(UPSTREAM)
        expected = upstream_call()
        actual = mojo_call()
        if actual != expected:
            raise AssertionError(f"benchmark parity failed for {name}")
        mojo_seconds = best_time(mojo_call)
        upstream_seconds = best_time(upstream_call)
        ratio = upstream_seconds / mojo_seconds
        print(
            f"| {name} | {mojo_seconds * 1000:.2f} ms | "
            f"{upstream_seconds * 1000:.2f} ms | {ratio:.2f}x |"
        )


if __name__ == "__main__":
    main()
