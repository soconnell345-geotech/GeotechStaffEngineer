"""
The code runs on numpy 1.26 as well as numpy 2 (BACKLOG INF-53): Foundry's
transforms repos hold numpy at 1.26.4. A name numpy 2 added, or one a later
numpy removed (trapz), is taken from geotech_common.numpy_compat, never from
numpy itself.

The scan runs on every host, so a new numpy-2-only call fails here on
numpy 2 too, not only in the numpy 1.26 CI job.
"""

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from geotech_common.numpy_compat import trapezoid

ROOT = Path(__file__).resolve().parents[2]

#: Names in numpy 2's namespace that numpy 1.26 does not have (from a diff
#: of dir(numpy) between 1.26.4 and 2.5.4), and ndarray attributes likewise.
NUMPY2_NAMES = (
    "trapezoid", "long", "ulong", "bool", "concat", "isdtype", "unstack",
    "astype", "pow", "acos", "acosh", "asin", "asinh", "atan", "atan2",
    "atanh", "bitwise_count", "bitwise_invert", "bitwise_left_shift",
    "bitwise_right_shift", "cumulative_sum", "cumulative_prod",
    "matrix_transpose", "permute_dims", "vecdot", "matvec", "vecmat",
    "unique_all", "unique_counts", "unique_inverse", "unique_values",
    "strings",
)
NUMPY2_LINALG = (
    "cross", "diagonal", "matmul", "matrix_norm", "matrix_transpose",
    "outer", "svdvals", "tensordot", "trace", "vecdot", "vector_norm",
)
#: Names numpy 1.26 has and a later numpy removed: trapz went in 2.4.
REMOVED_LATER = ("trapz",)
_PATTERN = re.compile(
    r"\b(?:np|numpy)\.(?:(?:%s)|linalg\.(?:%s))\b"
    % ("|".join(NUMPY2_NAMES + REMOVED_LATER), "|".join(NUMPY2_LINALG)))
_ARRAY_ATTRS = re.compile(r"\.(?:mT|to_device)\b")


def _tracked_sources():
    """The repository's tracked .py files (untracked scratch is not code)."""
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    run = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT,
                         capture_output=True, text=True)
    if run.returncode:     # e.g. a Windows worktree read from WSL
        pytest.skip(f"git ls-files failed: {run.stderr.strip()[:200]}")
    for name in run.stdout.splitlines():
        path = ROOT / name
        # Not the module that holds the names, nor this file, which tests
        # the scan on them.
        skip = ("numpy_compat.py", Path(__file__).name)
        if path.name not in skip and path.is_file():
            yield path


def test_the_scan_flags_both_kinds():
    assert _PATTERN.search("y = np.trapezoid(f, x)")
    assert _PATTERN.search("y = numpy.trapz(f, x)")
    assert not _PATTERN.search("y = trapezoid(f, x)")


def test_no_numpy2_only_names_in_the_code():
    found = []
    for path in _tracked_sources():
        text = path.read_text(encoding="utf-8", errors="replace")
        for n, line in enumerate(text.splitlines(), 1):
            code = line.split("#", 1)[0]
            if _PATTERN.search(code) or _ARRAY_ATTRS.search(code):
                found.append(f"{path.relative_to(ROOT)}:{n}: {line.strip()}")
    assert not found, (
        "numpy names that numpy 1.26 or the newest numpy lacks (take them "
        "from geotech_common.numpy_compat):\n" + "\n".join(found))


def test_trapezoid_is_the_trapezoid_rule():
    x = np.array([0.0, 1.0, 3.0])
    assert trapezoid([2.0, 4.0, 4.0], x) == pytest.approx(3.0 + 8.0)
    assert trapezoid(y=[1.0, 1.0], x=[0.0, 2.0]) == pytest.approx(2.0)
