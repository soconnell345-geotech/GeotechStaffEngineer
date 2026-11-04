"""The Tiny Apps wrapper template picks how our code arrives (OWN-72, INF-34).

``tinyapps/wrapper_repo/run.sh`` takes route ``source`` (``app_src/`` on
PYTHONPATH), ``wheel`` (``wheels/*.whl``) or ``nexus`` (``packages.txt``),
named by ``APP_SOURCE`` or chosen by what the repo holds. These tests run its
``APP_DRY_RUN=1`` mode with bash on made-up wrapper folders; nothing is
installed or started.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

WRAPPER = Path(__file__).resolve().parents[2] / "tinyapps" / "wrapper_repo"
INDEX = "https://nexus.example/simple"

_BASH = shutil.which("bash")
#: On Windows the WSL launcher is also called bash; it does not see the
#: test's environment, so only Git Bash (or a Linux bash) runs these.
_WSL = _BASH and os.name == "nt" and any(
    part in _BASH.lower() for part in ("system32", "windowsapps"))
needs_bash = pytest.mark.skipif(not _BASH or bool(_WSL),
                                reason="needs bash (not WSL's launcher)")


def _wrapper(tmp_path, src=False, wheel=False, src_reqs=False):
    """A made-up wrapper repo: the real run.sh beside placeholder files."""
    d = tmp_path / "wrapper"
    d.mkdir()
    # LF endings whatever the checkout did: this tests the logic, not git
    (d / "run.sh").write_bytes(
        (WRAPPER / "run.sh").read_bytes().replace(b"\r\n", b"\n"))
    for name in ("app.py", "packages.txt", "requirements-thirdparty.txt"):
        (d / name).write_text("# placeholder\n")
    if src:
        (d / "app_src" / "madeup_pkg").mkdir(parents=True)
        (d / "app_src" / "madeup_pkg" / "__init__.py").write_text("")
    if src_reqs:
        (d / "app_src" / "requirements.txt").write_text("madeup-core==0.1\n")
    if wheel:
        (d / "wheels").mkdir()
        (d / "wheels" / "madeup_app-0.1-py3-none-any.whl").write_bytes(b"")
    return d


def _run(d, **env):
    full = {k: v for k, v in os.environ.items()
            if k not in ("APP_SOURCE", "PYTHONPATH")}
    full.update(APP_DRY_RUN="1", PIP_INDEX_URL=INDEX, **env)
    return subprocess.run([_BASH, "run.sh"], cwd=d, env=full,
                          capture_output=True, text=True, timeout=60)


def _pip_line(out):
    lines = [ln for ln in out.splitlines()
             if ln.startswith("DRY RUN: python -m pip install")]
    assert len(lines) == 1, out
    return lines[0]


@needs_bash
def test_source_folder_wins_when_present(tmp_path):
    d = _wrapper(tmp_path, src=True, wheel=True, src_reqs=True)
    r = _run(d)
    assert r.returncode == 0, r.stderr
    assert "Code route: source" in r.stdout
    assert "chosen by what the repo holds" in r.stdout
    pip = _pip_line(r.stdout)
    assert "requirements-thirdparty.txt" in pip
    assert "app_src/requirements.txt" in pip
    assert "packages.txt" not in pip and ".whl" not in pip
    assert f"--index-url {INDEX}" in pip
    start = r.stdout.splitlines()[-1]
    assert start.startswith("DRY RUN: PYTHONPATH=") and "app_src" in start
    assert "Starting Streamlit" not in r.stdout


@needs_bash
def test_wheels_when_no_source_folder(tmp_path):
    d = _wrapper(tmp_path, wheel=True)
    r = _run(d)
    assert r.returncode == 0, r.stderr
    assert "Code route: wheel (madeup_app-0.1-py3-none-any.whl;" in r.stdout
    pip = _pip_line(r.stdout)
    assert "madeup_app-0.1-py3-none-any.whl" in pip
    assert "requirements-thirdparty.txt" in pip
    assert "packages.txt" not in pip
    assert f"--index-url {INDEX}" in pip


@needs_bash
def test_nexus_when_the_repo_holds_neither(tmp_path):
    d = _wrapper(tmp_path)
    r = _run(d)
    assert r.returncode == 0, r.stderr
    assert "Code route: nexus" in r.stdout
    pip = _pip_line(r.stdout)
    assert "packages.txt" in pip and "requirements-thirdparty" not in pip
    assert "app_src" not in r.stdout.splitlines()[-1]


@needs_bash
@pytest.mark.parametrize("route, expect", [
    ("wheel", "Code route: wheel"), ("nexus", "Code route: nexus")])
def test_app_source_overrides_what_is_present(tmp_path, route, expect):
    d = _wrapper(tmp_path, src=True, wheel=True)
    r = _run(d, APP_SOURCE=route)
    assert r.returncode == 0, r.stderr
    assert expect in r.stdout and "set by APP_SOURCE" in r.stdout
    assert "app_src" not in r.stdout.splitlines()[-1]


@needs_bash
@pytest.mark.parametrize("route", ["source", "wheel", "pypi"])
def test_a_named_route_without_its_files_fails_loudly(tmp_path, route):
    d = _wrapper(tmp_path)
    r = _run(d, APP_SOURCE=route)
    assert r.returncode != 0
    assert "ERROR: APP_SOURCE=" in r.stderr
    assert "DRY RUN" not in r.stdout


def test_app_py_finds_the_source_folder_without_run_sh(tmp_path):
    """A plain ``python app.py`` (as ``streamlit run app.py`` on dosdev)
    imports the entry from app_src/ ahead of any installed copy."""
    d = tmp_path / "wrapper"
    entry = d / "app_src" / "webapp"
    entry.mkdir(parents=True)
    shutil.copy(WRAPPER / "app.py", d / "app.py")
    (entry / "__init__.py").write_text("")
    (entry / "tinyapps_entry.py").write_text(
        "def main():\n    print('made-up entry from app_src')\n")
    env = {k: v for k, v in os.environ.items()
           if k not in ("APP_SOURCE", "PYTHONPATH")}
    r = subprocess.run([sys.executable, "app.py"], cwd=d, env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert "made-up entry from app_src" in r.stdout


def test_template_keeps_our_package_apart_from_the_third_party_pins():
    def reqs(name):
        return [ln.strip() for ln in
                (WRAPPER / name).read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")]
    packages = reqs("packages.txt")
    third = reqs("requirements-thirdparty.txt")
    assert "-r requirements-thirdparty.txt" in packages
    assert any(ln.startswith("geotech-staff-engineer") for ln in packages)
    assert not any("geotech" in ln for ln in third)
    for pin in ("streamlit==1.62.0", "PyJWT[crypto]==2.13.0", "msal==1.37.0",
                "azure-identity==1.21.0", "azure-keyvault-secrets==4.9.0"):
        assert pin in third
