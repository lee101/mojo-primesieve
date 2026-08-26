from __future__ import annotations

import array
import importlib

import numpy as np
import pytest


@pytest.mark.parametrize(
    "args",
    [
        (0,),
        (1,),
        (2,),
        (3,),
        (100,),
        (100, 120),
        (120, 100),
        (10_000, 20_000),
        (999_000, 1_001_000),
    ],
)
def test_primes_matches_upstream(ours, upstream, args):
    actual = ours.primes(*args)
    expected = upstream.primes(*args)
    assert isinstance(actual, array.array)
    assert actual.typecode == expected.typecode == "Q"
    assert actual == expected


@pytest.mark.parametrize(
    ("stop", "expected"),
    [
        (10, 4),
        (100, 25),
        (1_000, 168),
        (10_000, 1_229),
        (100_000, 9_592),
        (1_000_000, 78_498),
        (10_000_000, 664_579),
    ],
)
def test_published_prime_counts(ours, stop, expected):
    assert ours.count_primes(stop) == expected


@pytest.mark.parametrize(
    "args",
    [(0,), (1,), (10,), (1_000,), (25, 1000), (500, 1_000_000)],
)
def test_n_primes_matches_upstream(ours, upstream, args):
    assert ours.n_primes(*args) == upstream.n_primes(*args)


@pytest.mark.parametrize(
    "args",
    [
        (0,),
        (1,),
        (2,),
        (10_000,),
        (0, 100),
        (1, 100),
        (50, 1_000_000),
        (-1, 100),
        (-2, 100),
        (-100, 10_000),
    ],
)
def test_nth_prime_matches_upstream(ours, upstream, args):
    assert ours.nth_prime(*args) == upstream.nth_prime(*args)


@pytest.mark.parametrize(
    "name",
    [
        "count_primes",
        "count_twins",
        "count_triplets",
        "count_quadruplets",
        "count_quintuplets",
        "count_sextuplets",
    ],
)
@pytest.mark.parametrize("args", [(100,), (10_000,), (1_000, 100_000)])
def test_count_functions_match_upstream(ours, upstream, name, args):
    assert getattr(ours, name)(*args) == getattr(upstream, name)(*args)


@pytest.mark.parametrize("args", [(100,), (100, 10_000), (1_000_000, 1_020_000)])
def test_numpy_primes_matches_upstream(ours_numpy, upstream, args):
    actual = ours_numpy.primes(*args)
    expected = np.asarray(upstream.primes(*args), dtype=np.uint64)
    assert actual.dtype == expected.dtype == np.uint64
    assert np.array_equal(actual, expected)


def test_numpy_n_primes_matches_upstream(ours_numpy, upstream):
    assert np.array_equal(
        ours_numpy.n_primes(5000),
        np.asarray(upstream.n_primes(5000), dtype=np.uint64),
    )
    assert np.array_equal(
        ours_numpy.n_primes(1000, 1_000_000),
        np.asarray(upstream.n_primes(1000, 1_000_000), dtype=np.uint64),
    )


def test_iterator_forward_backward_and_skipto(ours, upstream):
    actual = ours.Iterator()
    expected = upstream.Iterator()
    assert [actual.next_prime() for _ in range(20)] == [
        expected.next_prime() for _ in range(20)
    ]
    actual.skipto(10_000)
    expected.skipto(10_000)
    assert [actual.next_prime() for _ in range(20)] == [
        expected.next_prime() for _ in range(20)
    ]
    actual.skipto(10_000)
    expected.skipto(10_000)
    assert [actual.prev_prime() for _ in range(20)] == [
        expected.prev_prime() for _ in range(20)
    ]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("print_primes", ["2", "3", "5", "7", "11", "13", "17", "19", "23", "29"]),
        ("print_twins", ["(3, 5)", "(5, 7)", "(11, 13)", "(17, 19)"]),
        ("print_triplets", ["(5, 7, 11)", "(7, 11, 13)", "(11, 13, 17)", "(13, 17, 19)", "(17, 19, 23)"]),
        ("print_quadruplets", ["(5, 7, 11, 13)", "(11, 13, 17, 19)"]),
        ("print_quintuplets", ["(5, 7, 11, 13, 17)", "(7, 11, 13, 17, 19)", "(11, 13, 17, 19, 23)"]),
        ("print_sextuplets", ["(7, 11, 13, 17, 19, 23)"]),
    ],
)
def test_print_helpers(ours, name, expected, capsys):
    getattr(ours, name)(30)
    assert capsys.readouterr().out.splitlines() == expected


def test_small_segment_configuration_exercises_boundaries(ours, upstream):
    original = ours.get_sieve_size()
    try:
        ours.set_sieve_size(16)
        assert ours.get_sieve_size() == 16
        assert ours.primes(100_000, 1_000_000) == upstream.primes(100_000, 1_000_000)
        assert ours.count_primes(10_000_000) == upstream.count_primes(10_000_000)
    finally:
        ours.set_sieve_size(original)


def test_invalid_inputs_and_thread_configuration(ours):
    with pytest.raises(OverflowError):
        ours.primes(-1)
    with pytest.raises(OverflowError):
        ours.n_primes(-1)
    with pytest.raises(RuntimeError):
        ours.nth_prime(-1, 2)
    with pytest.raises(ValueError):
        ours.set_sieve_size(1)
    with pytest.raises(TypeError):
        ours.primes(3.5)
    with pytest.raises(TypeError):
        ours.n_primes(2.5)
    with pytest.raises(TypeError):
        ours.set_sieve_size(32.5)
    original_threads = ours.get_num_threads()
    try:
        ours.set_num_threads(2)
        assert ours.get_num_threads() == 2
        with pytest.raises(ValueError):
            ours.set_num_threads(0)
        with pytest.raises(TypeError):
            ours.set_num_threads(2.5)
    finally:
        ours.set_num_threads(original_threads)


def test_simd_tail_and_parallel_threshold(ours, upstream, monkeypatch):
    api = importlib.import_module("primesieve._api")
    original_size = ours.get_sieve_size()
    original_threads = ours.get_num_threads()
    try:
        assert ours.primes(3, 69) == upstream.primes(3, 69)
        assert ours.nth_prime(20) == upstream.nth_prime(20)
        assert ours.primes(10**12, 10**12 + 132) == upstream.primes(
            10**12, 10**12 + 132
        )
        ours.set_sieve_size(16)
        ours.set_num_threads(4)
        monkeypatch.setattr(api, "_PARALLEL_MIN_ODDS", 10**18)
        serial = ours.primes(3, 200_123)
        monkeypatch.setattr(api, "_PARALLEL_MIN_ODDS", 0)
        parallel = ours.primes(3, 200_123)
        assert serial == parallel == upstream.primes(3, 200_123)
    finally:
        ours.set_num_threads(original_threads)
        ours.set_sieve_size(original_size)


def test_constellation_batch_boundary(ours, upstream, monkeypatch):
    api = importlib.import_module("primesieve._api")
    original_size = ours.get_sieve_size()
    original_threads = ours.get_num_threads()
    try:
        ours.set_sieve_size(16)
        ours.set_num_threads(4)
        monkeypatch.setattr(api, "_PARALLEL_MIN_ODDS", 0)
        for name in (
            "count_twins",
            "count_triplets",
            "count_quadruplets",
            "count_quintuplets",
            "count_sextuplets",
        ):
            assert getattr(ours, name)(1_100_123) == getattr(upstream, name)(1_100_123)
    finally:
        ours.set_num_threads(original_threads)
        ours.set_sieve_size(original_size)


def test_build_artifact_and_version(ours):
    assert ours.build().endswith("dist/libmojo-primesieve.so")
    assert ours.primesieve_version() == "mojo-primesieve 0.1.0"
    assert ours.get_max_stop() == 2**63 - 1


def test_native_boundary_rejects_null_and_invalid_lengths(ours):
    native = importlib.import_module("primesieve._lib").lib()
    assert native.mps_small_sieve(100, None, 10) == -1
    assert native.mps_small_sieve(100, None, -1) == -1
    assert native.mps_select_flag(3, None, 10, 1) == -1
    assert native.mps_count_flag_constellations(None, 10, 2) == -1
    assert native.mps_count_flag_constellations(None, 0, 2) == 0
