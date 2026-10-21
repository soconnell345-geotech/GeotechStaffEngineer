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


def pytest_terminal_summary(terminalreporter):
    """Say so when every test of a top-level package skipped.

    Smart App Control blocks the openseespy DLL on the Windows host, so every
    ``opensees_agent`` test skips with nothing but an ``s`` in the dots. This
    only prints; it never changes the exit code."""
    stats = terminalreporter.stats
    total = {}
    skipped = {}
    reasons = {}
    for outcome, reports in stats.items():
        if outcome not in ("passed", "failed", "error", "skipped",
                           "xfailed", "xpassed"):
            continue
        for rep in reports:
            nodeid = getattr(rep, "nodeid", "")
            if "/" not in nodeid:
                continue
            if outcome in ("passed", "xpassed") and getattr(
                    rep, "when", "call") != "call":
                continue                      # count a test once, at its call
            pkg = nodeid.split("/", 1)[0]
            total[pkg] = total.get(pkg, 0) + 1
            if outcome == "skipped":
                skipped[pkg] = skipped.get(pkg, 0) + 1
                if pkg not in reasons:
                    lr = getattr(rep, "longrepr", None)
                    text = lr[2] if isinstance(lr, tuple) and len(lr) == 3 \
                        else str(lr)
                    reasons[pkg] = text.removeprefix("Skipped: ").strip()
    all_skipped = [p for p in sorted(total) if skipped.get(p) == total[p]]
    if not all_skipped:
        return
    terminalreporter.section("ALL SKIPPED (no test of these packages ran)",
                             yellow=True)
    for pkg in all_skipped:
        terminalreporter.write_line(
            f"ALL SKIPPED: {pkg} ({total[pkg]} tests): {reasons[pkg]}")
