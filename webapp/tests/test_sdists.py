"""The sdists of the two packages under ``packages/`` carry no tests (the app
split's step 13, INF-33).

Why: setuptools puts ``tests/test*.py`` in an sdist by default, and PyPI
keeps an upload for good. The packages' tests stay in this repository: each
package's MANIFEST.in prunes them. This builds each sdist the way a release
does (``python -m build``) and lists it.

How it runs:
- each build works on a copy of the package's folder in a temporary folder,
  so nothing is written into the source tree and it can run beside other
  tests (seconds);
- it needs the ``build`` package and a setuptools that reads the packages'
  licence fields (77 or later). Where either is missing it skips, saying so;
  with ``GEOTECH_REQUIRE_WHEEL_TEST=1`` (CI's Wheels job) that is a failure.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
#: Each package (its folder under packages/) and files its sdist must still
#: have (a prune that went too far fails here).
PACKAGES = {
    "aec-agent-core": ("aec_agent_core/__init__.py",
                       "aec_agent_core/hooks.py"),
    "aec-document-review": ("aec_document_review/__init__.py",
                            "aec_document_review/app.py",
                            "aec_document_review/ws_upload_component/"
                            "index.html"),
}


def _need_build():
    missing = None
    if importlib.util.find_spec("build") is None:
        missing = "the 'build' package"
    else:
        try:
            major = int(importlib.metadata.version("setuptools").split(".")[0])
        except importlib.metadata.PackageNotFoundError:
            major = 0
        if major < 77:
            missing = "setuptools 77 or later"
    if missing is None:
        return
    if os.environ.get("GEOTECH_REQUIRE_WHEEL_TEST") == "1":
        pytest.fail(f"the sdist test needs {missing}")
    pytest.skip(f"the sdist test needs {missing}")


def _sdist(name: str, tmp_path: pathlib.Path) -> list[str]:
    src = tmp_path / "src"
    shutil.copytree(ROOT / "packages" / name, src, ignore=shutil.ignore_patterns(
        "*.egg-info", "build", "dist", "__pycache__", ".pytest_cache"))
    out = tmp_path / "dist"
    proc = subprocess.run(
        [sys.executable, "-m", "build", "--sdist", "--no-isolation",
         "--outdir", str(out), str(src)],
        capture_output=True, text=True)
    assert proc.returncode == 0, \
        f"{name}: sdist build failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    (tarball,) = out.glob("*.tar.gz")
    with tarfile.open(tarball) as tar:
        # drop the "<name>-<version>/" folder every member starts with
        return [m.name.split("/", 1)[1] for m in tar.getmembers()
                if m.isfile() and "/" in m.name]


@pytest.mark.parametrize("name", sorted(PACKAGES))
def test_the_sdist_carries_the_package_and_no_tests(name, tmp_path):
    _need_build()
    names = _sdist(name, tmp_path)
    tests = [n for n in names if "tests" in n.split("/")[:-1]]
    assert not tests, f"{name}: tests in the sdist: {tests[:10]}"
    missing = [n for n in PACKAGES[name] if n not in names]
    assert not missing, f"{name}: missing from the sdist: {missing}"
