"""
Root-level pytest conftest.py — shared configuration for all test modules.

Sets the matplotlib backend to 'Agg' (non-interactive) once, so individual
test files don't each need to do it themselves.
"""

import matplotlib
matplotlib.use("Agg")


import pytest as _pytest


@_pytest.fixture(autouse=True)
def _shared_host_flag_reset():
    """``funhouse_agent._fileio.require_turn_binding`` is process-wide (a
    multi-user identity sets it); keep one test's shared-host mode from
    leaking into the next."""
    try:
        from funhouse_agent import _fileio
    except Exception:                                  # noqa: BLE001
        yield
        return
    saved = _fileio._REQUIRE_BINDING
    yield
    _fileio._REQUIRE_BINDING = saved
