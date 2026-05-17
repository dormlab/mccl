"""Pytest fixtures backed by tests/common.py."""
import sys
import pathlib

# Make `from common import ...` work for tests/collectives/*.py
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import pytest
import common as _c


@pytest.fixture
def spawn():
    return _c.spawn


@pytest.fixture
def harness():
    return _c
