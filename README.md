# mojo-primesieve

`mojo-primesieve` is a standalone Mojo port of the compute-heavy prime-sieving
parts of the Python [`primesieve`](https://pypi.org/project/primesieve/)
package. It provides the same import name, function names, argument order, and
inclusive range behavior for the covered API.

This is a real odd-only segmented sieve, not a wrapper around libprimesieve.
It is useful when a Mojo-native sieve kernel and a small ctypes boundary are
more important than matching the highly tuned upstream C++ library's peak
performance.

## Coverage

The following upstream Python API is covered and exercised by parity or
published-value tests:

- `primes`, `n_primes`, `nth_prime`, and `count_primes`
- `count_twins`, `count_triplets`, `count_quadruplets`,
  `count_quintuplets`, and `count_sextuplets`
- all corresponding `print_*` helpers
- forward and backward `Iterator`, including `skipto`
- `primesieve.numpy.primes` and `primesieve.numpy.n_primes`
- meaningful sieve-buffer control through `get_sieve_size` and
  `set_sieve_size`

Large independent segment batches use a native worker pool.
`get_num_threads()` reports its configured maximum and `set_num_threads()`
changes it; small calls remain serial to avoid launch overhead. Inputs are
limited to `sys.maxsize` because the C ABI uses signed integers; unlike
upstream, the port does not accept the upper half of the unsigned 64-bit
range. Very large stops also require all base primes through the square root
to fit in memory.

There is no GPU path. Odd-multiple crossing is an irregular, store-heavy
kernel with well under two arithmetic operations per byte moved, and flag
scans are directly memory-bandwidth-bound. Host/device transfers and launch
overhead therefore have no arithmetic intensity to amortize; CPU remains the
only execution device rather than exposing a GPU option that loses.

This port does not implement upstream's C++ API, CLI, iterator range hints,
architecture-specific wheel packaging, or bucket sieve.
`primesieve_version()` identifies this port instead of reporting the linked
C++ library version.

## Install and run

Install the pinned Mojo toolchain, Python dependencies, and the upstream
reference package:

```bash
pixi install
pixi run build
```

The Pixi environment puts `python/` on `PYTHONPATH`, so normal upstream-style
imports use this port:

```bash
pixi run python - <<'PY'
import primesieve
from primesieve.numpy import primes as numpy_primes

print(primesieve.primes(40))
print(primesieve.count_primes(1_000_000))
print(numpy_primes(100, 120))
PY
```

Output:

```text
array('Q', [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37])
78498
[101 103 107 109 113]
```

Run the complete validation with:

```bash
pixi run build
pixi run test
pixi run bench
```

## Benchmarks

Measured with the machine-wide benchmark lock on an Intel Xeon E5-2697 v4 at
2.30 GHz with 72 logical CPUs. Times are the best of five warm runs. The ratio
is upstream time divided by Mojo time, so values below 1 mean upstream is
faster.

| Operation | Mojo | upstream primesieve | upstream / Mojo |
|---|---:|---:|---:|
| `primes(10,000,000)` | 15.13 ms | 12.48 ms | 0.82x |
| `primes(10^12, 10^12 + 5M)` | 15.79 ms | 7.47 ms | 0.47x |
| `count_primes(100,000,000)` | 8.23 ms | 1.97 ms | 0.24x |
| `count_twins(10,000,000)` | 2.49 ms | 0.88 ms | 0.35x |
| `nth_prime(100,000)` | 1.46 ms | 0.09 ms | 0.06x |

Upstream wins every measured case. Its C++ implementation uses wheel
factorization, bit compression, cache-tuned bucket sieving, and multiple
threads for supported operations. This port still uses one byte per odd
candidate and remains slower on this host despite SIMD scans, adaptive
segments, and parallel batches.

These are measured results from `pixi run bench`, not estimates. The benchmark
first asserts that both implementations return identical results.

## How it works

Python allocates a contiguous NumPy `uint8` segment and an `int64` base-prime
array. Buffer addresses cross the C ABI as signed 64-bit integers through
Mojo's exported integer parameters and as pointer-typed `ctypes` arguments;
exported Mojo functions rebuild them as mutable `UnsafePointer` values. No
Python objects cross the ABI.

Each byte in a segment represents one odd integer. Python checks dtype,
contiguity, writability, and native return status around each synchronous C
call; the NumPy arrays remain strongly referenced for the full call. Mojo
rejects invalid lengths and null pointers before constructing an
`UnsafePointer`. Mojo then fills the bytes,
crosses out odd multiples of the base primes beginning at `max(p*p, ceil(low /
p)*p)`, counts survivors, and optionally compacts them into a contiguous
`uint64` output buffer. High-base ranges use a SIMD-repeated 3-by-5 presieve;
the periodic vector copy and its remainder, survivor reduction, rank
selection, and direct constellation counting all have scalar tail handling.
The default uses cache-sized 256 KiB segments, coalescing them to 1 MiB when a
large base-prime table makes repeated traversal more expensive than cache
locality. Explicit `set_sieve_size` values are used unchanged. Up to 256
independent segments are processed in one batch; batches with at least two
segments and two million odd candidates use `parallelize`, while smaller work
stays serial. The NumPy buffers pass directly through `ctypes`, collection
writes into the final unsigned NumPy dtype, and Python constructs `array('Q')`
results through the buffer protocol instead of iterating over NumPy scalars.

The tests compare the covered numerical functions and iterator behavior with
the installed upstream `primesieve` 2.3.4 extension. They also assert each
print helper's output, exercise configuration and native error paths, and
check published values of `pi(10^k)` through `10^7`.
