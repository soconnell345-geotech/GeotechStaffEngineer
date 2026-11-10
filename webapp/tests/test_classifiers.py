"""Every classifier in the three pyprojects is one PyPI accepts.

Why: PyPI refuses an upload whose metadata names a classifier it does not
list, and it refuses it only at upload time, after the tag is pushed.
aec-document-review 0.1.0's first upload (2026-11-10) was refused for
"Framework :: Streamlit". This checks the root pyproject and the two under
``packages/`` against ``trove-classifiers``, PyPI's own list.

It needs the ``trove-classifiers`` package. Where it is missing it skips,
saying so; with ``GEOTECH_REQUIRE_WHEEL_TEST=1`` (CI's Wheels job, which
installs it) that is a failure.
"""

from __future__ import annotations

import os
import pathlib
import tomllib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PYPROJECTS = (
    ROOT / "pyproject.toml",
    ROOT / "packages" / "aec-agent-core" / "pyproject.toml",
    ROOT / "packages" / "aec-document-review" / "pyproject.toml",
)


def _known():
    try:
        import trove_classifiers
    except ImportError:
        if os.environ.get("GEOTECH_REQUIRE_WHEEL_TEST") == "1":
            pytest.fail("trove-classifiers is not installed")
        pytest.skip("trove-classifiers is not installed")
    return trove_classifiers.classifiers


@pytest.mark.parametrize("path", PYPROJECTS, ids=lambda p: p.parent.name)
def test_classifiers_are_known_to_pypi(path):
    known = _known()
    project = tomllib.loads(path.read_text(encoding="utf-8"))["project"]
    unknown = [c for c in project.get("classifiers", []) if c not in known]
    assert not unknown, (
        f"{path.relative_to(ROOT)} names classifiers PyPI does not accept "
        f"(it would refuse the upload): {unknown}")
