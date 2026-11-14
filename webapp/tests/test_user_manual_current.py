"""The user manual is what its build makes of manual_content.py and the
registry (2026-11-11 reference review).

``docs/user_manual/manual.html`` and ``toc.json`` are built by
``docs/user_manual/build_manual.py`` from ``manual_content.py`` and the
dispatch registry, and nothing checked that the committed files still match:
the release step in CLAUDE.md asks for a rebuild, but a change to a tool's
description or a method's parameters between releases left the manual behind
without a sound. This runs ``module_work/doc10/check_manual.py --rebuild``,
which rebuilds the manual in memory, diffs it against the committed files and
checks the version on its cover and that every module, analysis method,
parameter and registered default is named; it fails on any miss.

To fix a failure: ``python module_work/doc10/check_manual.py --rebuild
--write``, read the diff and commit ``manual.html`` and ``toc.json``.
"""

import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).parents[2]
CHECK = ROOT / "module_work" / "doc10" / "check_manual.py"
BUILD = ROOT / "docs" / "user_manual" / "build_manual.py"
CONTENT = ROOT / "docs" / "user_manual" / "manual_content.py"


def test_the_manual_is_what_its_build_makes():
    missing = [p.name for p in (CHECK, BUILD, CONTENT) if not p.is_file()]
    if missing:
        pytest.skip(f"the manual's build tool is not here: {missing}")
    env = dict(os.environ)
    # this checkout's shared core first, as the root conftest does
    core = str(ROOT / "packages" / "aec-agent-core")
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (core, env.get("PYTHONPATH", "")) if p)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("MPLBACKEND", "Agg")
    run = subprocess.run(
        [sys.executable, str(CHECK), "--rebuild"], cwd=str(ROOT), env=env,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=900)
    tail = "\n".join((run.stdout + run.stderr).splitlines()[-60:])
    assert run.returncode == 0, (
        "docs/user_manual/manual.html is not what manual_content.py and the "
        "registry build; run `python module_work/doc10/check_manual.py "
        "--rebuild --write`, read the diff and commit manual.html and "
        "toc.json.\n" + tail)
