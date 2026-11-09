"""No local reference pack is tracked in this public repo (BACKLOG REF-64).

A deployment's reference packs (``GEOTECH_REFERENCES_PACKS``; the format is
the geotech-references repo's ``docs/REFERENCE_PACKS.md``) hold its own,
often private, manuals. They live on the host, never here. A tracked
``pack.json`` means one was committed; the refs repo has the same guard
(``tests/test_packs.py::test_no_pack_is_tracked_in_this_repo``).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def test_no_pack_json_is_tracked():
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "ls-files", "-z"], capture_output=True,
            check=True).stdout.decode("utf-8", "replace")
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    files = [f for f in out.split("\0") if f]
    assert len(files) > 1000, "the git walk found too few files"
    packs = [f for f in files if f.rsplit("/", 1)[-1] == "pack.json"]
    assert packs == [], (
        "a local reference pack is tracked in this public repo: "
        f"{packs}; packs stay on the host (GEOTECH_REFERENCES_PACKS)")
