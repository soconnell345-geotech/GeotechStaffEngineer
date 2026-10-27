"""
Root-level pytest conftest.py — shared configuration for all test modules.

Sets the matplotlib backend to 'Agg' (non-interactive) once, so individual
test files don't each need to do it themselves.
"""

import os
import shutil
import re as _re
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


@_pytest.fixture
def fail_import(monkeypatch):
    """``fail_import(name, exc)``: ``name`` is found, as an installed package
    is, and its import raises ``exc`` (BACKLOG INF-54). ``name`` and its
    submodules leave ``sys.modules`` for the test, and a finder at the front
    of ``sys.meta_path`` serves the failure; both are restored after."""
    import importlib.abc
    import importlib.util
    import sys

    class _Loader(importlib.abc.Loader):
        def __init__(self, exc):
            self.exc = exc

        def create_module(self, spec):
            return None

        def exec_module(self, module):
            raise self.exc

    class _Finder(importlib.abc.MetaPathFinder):
        def __init__(self, name, exc):
            self.name, self.exc = name, exc

        def find_spec(self, fullname, path=None, target=None):
            if fullname != self.name:
                return None
            return importlib.util.spec_from_loader(
                fullname, _Loader(self.exc), is_package=True)

    def fail(name, exc):
        for mod in [m for m in sys.modules
                    if m == name or m.startswith(name + ".")]:
            monkeypatch.delitem(sys.modules, mod)
        monkeypatch.setattr(sys, "meta_path",
                            [_Finder(name, exc)] + sys.meta_path)

    return fail


_MISSING_DEP = _re.compile(
    r"not installed|not importable|No module named|could not import|DLL",
    _re.IGNORECASE)


def pytest_terminal_summary(terminalreporter):
    """Say so when skips hide a missing dependency or a whole package.

    Smart App Control blocks the openseespy DLL on the Windows host, so the
    openseespy tests skip with nothing but an ``s`` in the dots. This only
    prints; it never changes the exit code."""
    stats = terminalreporter.stats
    total = {}
    skipped = {}
    first_reason = {}
    missing = {}                         # reason -> {pkg: count}
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
            if outcome != "skipped":
                continue
            skipped[pkg] = skipped.get(pkg, 0) + 1
            lr = getattr(rep, "longrepr", None)
            text = lr[2] if isinstance(lr, tuple) and len(lr) == 3 else str(lr)
            reason = text.removeprefix("Skipped: ").strip()
            first_reason.setdefault(pkg, reason)
            if _MISSING_DEP.search(reason):
                by_pkg = missing.setdefault(reason, {})
                by_pkg[pkg] = by_pkg.get(pkg, 0) + 1
    if missing:
        terminalreporter.section(
            "SKIPPED, a dependency missing (these tests did not run)",
            yellow=True)
        for reason, by_pkg in sorted(missing.items()):
            terminalreporter.write_line(
                f"SKIPPED, a dependency missing: {reason} "
                f"({sum(by_pkg.values())} tests: {', '.join(sorted(by_pkg))})")
    all_skipped = [p for p in sorted(total) if skipped.get(p) == total[p]]
    if all_skipped:
        terminalreporter.section(
            "ALL SKIPPED (no test of these packages ran)", yellow=True)
        for pkg in all_skipped:
            terminalreporter.write_line(
                f"ALL SKIPPED: {pkg} ({total[pkg]} tests): "
                f"{first_reason[pkg]}")
