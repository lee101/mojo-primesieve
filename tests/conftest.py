from __future__ import annotations

import importlib
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_PYTHON = os.path.join(ROOT, "python")


def _without_local_path():
    blocked = {os.path.abspath(LOCAL_PYTHON), os.path.abspath(ROOT)}
    return [path for path in sys.path if os.path.abspath(path or os.getcwd()) not in blocked]


def _load_packages():
    saved_path = sys.path[:]
    sys.path[:] = _without_local_path()
    for name in list(sys.modules):
        if name == "primesieve" or name.startswith("primesieve."):
            del sys.modules[name]
    upstream = importlib.import_module("primesieve")

    for name in list(sys.modules):
        if name == "primesieve" or name.startswith("primesieve."):
            del sys.modules[name]
    sys.path[:] = saved_path
    ours = importlib.import_module("primesieve")
    ours_numpy = importlib.import_module("primesieve.numpy")
    return upstream, ours, ours_numpy


UPSTREAM, OURS, OURS_NUMPY = _load_packages()


@pytest.fixture(scope="session")
def upstream():
    return UPSTREAM


@pytest.fixture(scope="session")
def ours():
    return OURS


@pytest.fixture(scope="session")
def ours_numpy():
    return OURS_NUMPY
