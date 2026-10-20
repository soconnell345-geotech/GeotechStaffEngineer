"""
Root-level pytest conftest.py — shared configuration for all test modules.

Sets the matplotlib backend to 'Agg' (non-interactive) once, so individual
test files don't each need to do it themselves.
"""

import os
import shutil
import tempfile

import matplotlib
matplotlib.use("Agg")


import pytest as _pytest


def pytest_configure(config):
    """Under pytest-xdist (``-n``), give each worker its own temp directory.

    Code that keeps a file at a FIXED name in the system temp dir -- the
    geotech-references FTS and figure databases, which some tests rebuild --
    would otherwise be rebuilt by one worker while another reads it.
    ``tmp_path`` is unaffected (xdist already gives each worker its own)."""
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if not worker:
        return
    path = tempfile.mkdtemp(prefix=f"pytest-{worker}-")
    os.environ["TMPDIR"] = path
    tempfile.tempdir = path
    config.add_cleanup(lambda: shutil.rmtree(path, ignore_errors=True))


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
