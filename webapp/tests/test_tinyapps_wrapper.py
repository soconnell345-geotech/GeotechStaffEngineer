"""The Tiny Apps wrapper template for Document Review (OWN-72, INF-34).

``tinyapps/wrapper_repo/run.sh`` takes route ``source`` (``app_src/`` on
PYTHONPATH), ``wheel`` (``wheels/*.whl``) or ``nexus`` (``packages.txt``),
named by ``APP_SOURCE`` or chosen by what the repo holds. These tests run its
``APP_DRY_RUN=1`` mode with bash on made-up wrapper folders; nothing is
installed or started. Since the app split's step 14 the template runs
``aec-document-review``: its ``app.py`` imports that package's entry and
``packages.txt`` pins it with the exact core and planlens its pyproject
names, beside the fleet pins with pandas==3.0.2.
"""

import fnmatch
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
WRAPPER = REPO / "tinyapps" / "wrapper_repo"
DR_PYPROJECT = REPO / "packages" / "aec-document-review" / "pyproject.toml"
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
    entry = d / "app_src" / "aec_document_review"
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


def test_template_keeps_run_sh_lf():
    """App Service runs ``bash run.sh``; a CRLF copy fails at the first line."""
    attrs = (WRAPPER / ".gitattributes").read_text(encoding="utf-8")
    assert "*.sh text eol=lf" in attrs.splitlines()


def test_template_keeps_the_keys_and_the_venv_out_of_commits():
    """The owner's checklist makes ``.venv`` and ``.env`` (the keys) inside
    the clone; a stage-all commit must not take them. Route B commits its
    wheels, so nothing of the template or of ``wheels/`` is ignored."""
    ignored = _reqs(".gitignore")
    assert ".env" in ignored and ".venv/" in ignored
    for kept in (".env.example", "run.sh", "app.py", "packages.txt",
                 "wheels", "aec_document_review-0.1.0-py3-none-any.whl"):
        assert not any(fnmatch.fnmatch(kept, pat.rstrip("/"))
                       for pat in ignored), kept


def _reqs(name):
    return [ln.strip() for ln in
            (WRAPPER / name).read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]


def _dr_project():
    """Document Review's version and its exact pins, from its pyproject."""
    text = DR_PYPROJECT.read_text(encoding="utf-8")
    version = re.search(r'^version = "([^"]+)"', text, re.M).group(1)
    pins = dict(re.findall(r'"(aec-agent-core|planlens)==([^"]+)"', text))
    return version, pins


def test_app_py_runs_the_document_review_entry():
    code = [ln.strip() for ln in
            (WRAPPER / "app.py").read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]
    imports = [ln for ln in code if ln.startswith(("from ", "import "))
               and "tinyapps_entry" in ln]
    assert imports == ["from aec_document_review.tinyapps_entry import main"
                       "  # noqa: E402  (after the env is set)"]
    assert code[-1] == "main()"
    assert not any("webapp" in ln for ln in imports)


def test_packages_txt_pins_the_package_and_its_exact_companions():
    version, pins = _dr_project()
    assert set(pins) == {"aec-agent-core", "planlens"}, pins
    packages = _reqs("packages.txt")
    assert f"aec-document-review=={version}" in packages
    assert f"aec-agent-core=={pins['aec-agent-core']}" in packages
    assert f"planlens=={pins['planlens']}" in packages
    assert not any(ln.startswith("geotech") for ln in packages)


def test_template_keeps_our_packages_apart_from_the_third_party_pins():
    packages = _reqs("packages.txt")
    third = _reqs("requirements-thirdparty.txt")
    assert "-r requirements-thirdparty.txt" in packages
    for ours in ("aec-document-review", "aec-agent-core", "planlens",
                 "geotech"):
        assert not any(ln.startswith(ours) for ln in third), ours
    for pin in ("streamlit==1.62.0", "pandas==3.0.2",
                "PyJWT[crypto]==2.13.0", "msal==1.37.0",
                "azure-identity==1.21.0", "azure-keyvault-secrets==4.9.0"):
        assert pin in third


def test_nothing_in_the_package_caps_pandas():
    """The fleet's pandas==3.0.2 needs no cap from Document Review or its
    core (the cap below 3 came with the geotechnical app's python-ags4)."""
    for pyproject in (DR_PYPROJECT, REPO / "packages" / "aec-agent-core"
                      / "pyproject.toml"):
        deps = [ln for ln in pyproject.read_text(encoding="utf-8").splitlines()
                if ln.strip().startswith('"') and "pandas" in ln]
        assert deps == [], (pyproject, deps)


@needs_bash
def test_the_real_template_takes_route_nexus_by_default(tmp_path):
    """The template as shipped (no app_src/, no wheels/) installs
    packages.txt and starts Streamlit on app.py."""
    d = tmp_path / "wrapper"
    shutil.copytree(WRAPPER, d)
    (d / "run.sh").write_bytes(
        (WRAPPER / "run.sh").read_bytes().replace(b"\r\n", b"\n"))
    r = _run(d)
    assert r.returncode == 0, r.stderr
    assert "Code route: nexus" in r.stdout
    assert "packages.txt" in _pip_line(r.stdout)
    assert r.stdout.splitlines()[-1].endswith("app.py --server.port 8000 ...")


# --- deployment review 2026-11-10 -------------------------------------------

def _origins(out):
    lines = [ln for ln in out.splitlines()
             if ln.startswith("DRY RUN: origins ")]
    assert len(lines) == 1, out
    return lines[0].split()[3:]


@needs_bash
def test_the_published_address_comes_from_an_app_setting(tmp_path):
    """The address was a line to edit in run.sh, lost at every update of
    the file; left as it was, the page never loaded and nothing said why."""
    d = _wrapper(tmp_path)
    r = _run(d, APP_ORIGINS="https://apps.example, https://review.example")
    assert r.returncode == 0, r.stderr
    assert _origins(r.stdout) == [
        "--server.corsAllowedOrigins", "https://apps.example",
        "--server.corsAllowedOrigins", "https://review.example"]
    assert "WARNING: the published address is not set" not in r.stderr


@needs_bash
def test_an_address_never_set_is_said_at_the_start(tmp_path):
    d = _wrapper(tmp_path)
    full = {k: v for k, v in os.environ.items()
            if k not in ("APP_SOURCE", "PYTHONPATH", "APP_ORIGINS")}
    full.update(APP_DRY_RUN="1", PIP_INDEX_URL=INDEX)
    r = subprocess.run([_BASH, "run.sh"], cwd=d, env=full,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert _origins(r.stdout) == ["--server.corsAllowedOrigins",
                                  "https://CHANGE-ME"]
    assert "WARNING: the published address is not set (APP_ORIGINS" in r.stderr


def _start(tmp_path, packages_present):
    """Run run.sh for real against a made-up ``python``: its ``pip install``
    fails, its import check answers ``packages_present``, and it prints a
    line where Streamlit would start."""
    d = _wrapper(tmp_path)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "python"
    stub.write_bytes((
        "#!/bin/bash\n"
        'case "$1 $2" in\n'
        '    "-m pip") echo "made-up pip: no such index"; exit 1 ;;\n'
        f'    "-c "*) exit {0 if packages_present else 1} ;;\n'
        '    "-m streamlit") echo "made-up streamlit started"; exit 0 ;;\n'
        "esac\n"
        "exit 0\n").encode())
    stub.chmod(0o755)
    full = {k: v for k, v in os.environ.items()
            if k not in ("APP_SOURCE", "PYTHONPATH", "APP_DRY_RUN")}
    full.update(PIP_INDEX_URL=INDEX, APP_ORIGINS="https://apps.example",
                PATH=str(bin_dir) + os.pathsep + full.get("PATH", ""))
    return subprocess.run([_BASH, "run.sh"], cwd=d, env=full,
                          capture_output=True, text=True, timeout=60)


@needs_bash
def test_a_failed_install_with_nothing_installed_stops_the_start(tmp_path):
    """The start went on to Streamlit, and the log's first error was an
    import error far from the cause."""
    r = _start(tmp_path, packages_present=False)
    assert r.returncode != 0
    assert "ERROR: the package install failed" in r.stderr
    assert "ERROR: the app cannot start" in r.stderr
    assert "Starting Streamlit" not in r.stdout
    assert "made-up streamlit started" not in r.stdout


@needs_bash
def test_a_failed_install_over_an_earlier_one_starts_with_a_warning(tmp_path):
    r = _start(tmp_path, packages_present=True)
    assert r.returncode == 0, r.stderr
    assert "ERROR: the package install failed" in r.stderr
    assert "WARNING: starting with the packages already installed" in r.stderr
    assert "made-up streamlit started" in r.stdout


@pytest.mark.parametrize("env, expect", [
    ({"WEBSITE_SITE_NAME": "madeup-site"}, "0"),          # the App Service
    ({"WEBSITE_SITE_NAME": "madeup-site",
      "GEOTECH_PLOT_CODE": "1"}, "1"),                    # the app setting wins
    ({}, "None"),                                         # a local run
])
def test_app_py_turns_model_written_plot_code_off_on_the_app_service(
        tmp_path, env, expect):
    """One process holds every tester's conversations there, and the box the
    code runs in limits Python, not the operating system (INF-16)."""
    d = tmp_path / "wrapper"
    entry = d / "app_src" / "aec_document_review"
    entry.mkdir(parents=True)
    shutil.copy(WRAPPER / "app.py", d / "app.py")
    (entry / "__init__.py").write_text("")
    (entry / "tinyapps_entry.py").write_text(
        "import os\n\n\ndef main():\n"
        "    print('plot code:', os.environ.get('GEOTECH_PLOT_CODE'))\n")
    full = {k: v for k, v in os.environ.items()
            if k not in ("APP_SOURCE", "PYTHONPATH", "GEOTECH_PLOT_CODE",
                         "WEBSITE_SITE_NAME", "IDENTITY_ENDPOINT")}
    full.update(env)
    r = subprocess.run([sys.executable, "app.py"], cwd=d, env=full,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert f"plot code: {expect}" in r.stdout


@needs_bash
def test_an_address_list_of_separators_only_is_an_address_not_set(tmp_path):
    d = _wrapper(tmp_path)
    r = _run(d, APP_ORIGINS=" , ")
    assert r.returncode == 0, r.stderr
    assert _origins(r.stdout) == []
    assert "WARNING: the published address is not set" in r.stderr


@needs_bash
def test_an_address_is_taken_as_written(tmp_path):
    """A star is passed on as a star, not matched against the files beside
    run.sh."""
    d = _wrapper(tmp_path)
    r = _run(d, APP_ORIGINS="https://apps.example  *")
    assert r.returncode == 0, r.stderr
    assert _origins(r.stdout) == [
        "--server.corsAllowedOrigins", "https://apps.example",
        "--server.corsAllowedOrigins", "*"]
