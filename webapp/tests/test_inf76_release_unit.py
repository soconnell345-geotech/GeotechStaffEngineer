"""The shared core and Document Review release as ONE unit (BACKLOG INF-76).

- Both packages carry one version, and every pin of the core (GSE,
  Document Review, the Tiny Apps wrapper's files) names it.
- One tag pattern, ``aec-v<version>``, starts both publish workflows; GSE's
  own ``v*`` never matches it. The two carry the same version guard, word
  for word.
- A release refuses when the core's folder has changed since the tag of the
  core version it pins (the guard is run here on a throwaway git repo).
- Document Review's publish waits, with a bound, for the core of its
  version to reach PyPI (run here against a local stand-in for PyPI).

Needs only Python, PyYAML and git: no network, nothing geotechnical.
"""

from __future__ import annotations

import fnmatch
import http.server
import os
import re
import shutil
import subprocess
import sys
import threading
import tomllib
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOWS = REPO / ".github" / "workflows"
CORE_DIR = REPO / "packages" / "aec-agent-core"
DR_DIR = REPO / "packages" / "aec-document-review"
WRAPPER = REPO / "tinyapps" / "wrapper_repo"

UNIT_WORKFLOWS = ("publish-core.yml", "publish-document-review.yml")
VERSION_STEP = "One version, one tag (the core and Document Review)"
CORE_STEP = "The pinned core is the core in this checkout"
PYPI_STEP = ("The aec-agent-core and planlens versions the package pins "
             "are on PyPI")


def _project(path: Path) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)["project"]


def _version_line(init: Path) -> str:
    return re.search(r'^__version__ = "([^"]+)"', init.read_text("utf-8"),
                     re.M).group(1)


def _pin(deps, name):
    out = []
    for dep in deps:
        dist = re.split(r"[\s<>=!~;\[(]", dep, maxsplit=1)[0]
        if re.sub(r"[-_.]+", "-", dist).lower() == name:
            out.append(re.sub(r"\s+", "", dep.split(";")[0]))
    return out


def _unit_version() -> str:
    return _project(CORE_DIR / "pyproject.toml")["version"]


def _workflow(name: str) -> dict:
    return yaml.safe_load((WORKFLOWS / name).read_text("utf-8"))


def _triggers(doc: dict) -> dict:
    # PyYAML reads the bare key `on` as True.
    return doc.get("on", doc.get(True))


def _step(name: str, workflow: str) -> dict:
    steps = _workflow(workflow)["jobs"]["build"]["steps"]
    found = [s for s in steps if s.get("name") == name]
    assert len(found) == 1, (workflow, name)
    return found[0]


def _run_step(step: dict, cwd: Path, **env) -> subprocess.CompletedProcess:
    assert step.get("shell") == "python"
    full = {k: v for k, v in os.environ.items()
            if k not in ("PYTHONPATH", "PYTHONHOME")}
    full.update({k: str(v) for k, v in (step.get("env") or {}).items()
                 if "${{" not in str(v)})
    full.update(env)
    return subprocess.run([sys.executable, "-I", "-c", step["run"]],
                          cwd=str(cwd), env=full, capture_output=True,
                          text=True, timeout=120)


# -- one version ----------------------------------------------------------

def test_the_core_and_document_review_carry_one_version():
    version = _unit_version()
    assert _project(DR_DIR / "pyproject.toml")["version"] == version
    assert _version_line(CORE_DIR / "aec_agent_core" / "__init__.py") == \
        version
    assert _version_line(DR_DIR / "aec_document_review" / "__init__.py") == \
        version


def test_every_pin_of_the_core_names_that_version():
    version = _unit_version()
    assert _pin(_project(DR_DIR / "pyproject.toml")["dependencies"],
                "aec-agent-core") == [f"aec-agent-core=={version}"]
    assert _pin(_project(REPO / "pyproject.toml")["dependencies"],
                "aec-agent-core") == [f"aec-agent-core=={version}"]
    packages = (WRAPPER / "packages.txt").read_text("utf-8").split()
    assert f"aec-agent-core=={version}" in packages
    assert f"aec-document-review=={version}" in packages


@pytest.mark.parametrize("doc", [
    WRAPPER / "packages.txt",
    WRAPPER / "README.md",
    REPO / "tinyapps" / "TINYAPPS.md",
])
def test_the_wrapper_files_state_no_other_version(doc):
    """Every pin and wheel name of either package in the Tiny Apps files is
    the one version (history lives elsewhere: the packages' READMEs)."""
    version = _unit_version()
    text = doc.read_text("utf-8")
    seen = re.findall(r"aec-(?:agent-core|document-review)==([0-9][\w.]*)",
                      text)
    seen += re.findall(r"aec_(?:agent_core|document_review)-([0-9][\w.]*?)"
                       r"-py3-none-any\.whl", text)
    seen += re.findall(r"`aec-(?:agent-core|document-review)` ([0-9][\w.]*)",
                       text)
    assert seen, doc
    assert set(seen) == {version}, (doc, sorted(set(seen)))


# -- one tag --------------------------------------------------------------

@pytest.mark.parametrize("name", [*UNIT_WORKFLOWS, "publish.yml"])
def test_the_release_workflows_parse(name):
    doc = _workflow(name)
    assert doc["jobs"]["publish"]["environment"] == "pypi"
    assert _triggers(doc)["push"]["tags"]


def test_one_tag_pattern_starts_both_and_never_gse():
    for name in UNIT_WORKFLOWS:
        assert _triggers(_workflow(name))["push"]["tags"] == ["aec-v*"], name
    gse = _triggers(_workflow("publish.yml"))["push"]["tags"]
    assert gse == ["v*"]
    tag = f"aec-v{_unit_version()}"
    assert fnmatch.fnmatchcase(tag, "aec-v*")
    assert not any(fnmatch.fnmatchcase(tag, p) for p in gse)
    assert not fnmatch.fnmatchcase("v5.36.0", "aec-v*")


def test_both_workflows_carry_the_same_guards():
    a, b = (_step(VERSION_STEP, w) for w in UNIT_WORKFLOWS)
    assert a == b
    core_a = _step(CORE_STEP, "publish.yml")
    core_b = _step(CORE_STEP, "publish-document-review.yml")
    assert core_a["run"] == core_b["run"]
    assert core_a["env"] == {"PYPROJECT": "pyproject.toml"}
    assert core_b["env"] == {
        "PYPROJECT": "packages/aec-document-review/pyproject.toml"}


def test_the_guard_runs_before_anything_is_built():
    for name in UNIT_WORKFLOWS:
        steps = _workflow(name)["jobs"]["build"]["steps"]
        names = [s.get("name") or s.get("run") or s.get("uses")
                 for s in steps]
        first_build = next(i for i, s in enumerate(names)
                           if "python -m build" in str(s))
        assert names.index(VERSION_STEP) < first_build, name


def test_the_checks_needing_tags_fetch_them():
    for name in ("publish.yml", "publish-document-review.yml"):
        checkout = _workflow(name)["jobs"]["build"]["steps"][0]
        assert checkout["uses"].startswith("actions/checkout@")
        assert checkout["with"]["fetch-depth"] == 0, name


def test_document_review_waits_for_the_core_with_a_bound():
    step = _step(PYPI_STEP, "publish-document-review.yml")
    wait = float(step["env"]["CORE_WAIT_SECONDS"])
    poll = float(step["env"]["CORE_POLL_SECONDS"])
    assert 0 < poll < wait <= 3600
    timeout = _workflow("publish-document-review.yml")["jobs"]["build"][
        "timeout-minutes"]
    assert timeout * 60 > wait


# -- the version guard, run ----------------------------------------------

def test_the_version_guard_passes_this_tree_at_its_tag():
    step = _step(VERSION_STEP, "publish-core.yml")
    out = _run_step(step, REPO, TAG=f"aec-v{_unit_version()}")
    assert out.returncode == 0, out.stdout + out.stderr


def _tree(tmp_path: Path, core: str, review: str, pin: str) -> Path:
    for sub, version, deps in (
            ("aec-agent-core", core, []),
            ("aec-document-review", review,
             [f"aec-agent-core=={pin}", "planlens==0.15.1"])):
        d = tmp_path / "packages" / sub
        d.mkdir(parents=True)
        (d / "pyproject.toml").write_text(
            f'[project]\nname = "{sub}"\nversion = "{version}"\n'
            f"dependencies = {deps!r}\n".replace("'", '"'), "utf-8")
    return tmp_path


@pytest.mark.parametrize("core,review,pin,tag,says", [
    ("2.0.0", "2.0.0", "2.0.0", "aec-v2.0.1", "does not match"),
    ("2.0.0", "1.9.0", "2.0.0", "aec-v2.0.0", "one version"),
    ("2.0.0", "2.0.0", "1.9.0", "aec-v2.0.0", "exactly"),
    ("2.0.0", "2.0.0", "2.0.0", "core-v2.0.0", "does not match"),
])
def test_the_version_guard_refuses(tmp_path, core, review, pin, tag, says):
    step = _step(VERSION_STEP, "publish-document-review.yml")
    out = _run_step(step, _tree(tmp_path, core, review, pin), TAG=tag)
    assert out.returncode == 1
    assert says in out.stdout


def test_the_version_guard_passes_a_matching_tree(tmp_path):
    step = _step(VERSION_STEP, "publish-document-review.yml")
    out = _run_step(step, _tree(tmp_path, "2.0.0", "2.0.0", "2.0.0"),
                    TAG="aec-v2.0.0")
    assert out.returncode == 0, out.stdout + out.stderr


# -- the pinned-core guard, run on a throwaway repository -----------------

def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
         "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args],
        cwd=str(cwd), check=True, capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    r = tmp_path / "r"
    (r / "packages" / "aec-agent-core" / "aec_agent_core").mkdir(parents=True)
    _git(r, "init", "-q")
    (r / "packages" / "aec-agent-core" / "aec_agent_core" / "a.py") \
        .write_text("A = 1\n", "utf-8")
    (r / "pyproject.toml").write_text(
        '[project]\nname = "app"\nversion = "9.0.0"\n'
        'dependencies = ["numpy>=1.26", "aec-agent-core == 2.0.0"]\n',
        "utf-8")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "core 2.0.0")
    return r


def _core_guard(cwd):
    step = _step(CORE_STEP, "publish.yml")
    return _run_step(step, cwd)


def test_the_core_guard_passes_an_unchanged_core(repo):
    _git(repo, "tag", "aec-v2.0.0")
    (repo / "app.py").write_text("x = 1\n", "utf-8")  # the app moved on
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "app only")
    out = _core_guard(repo)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "unchanged since aec-v2.0.0" in out.stdout


def test_the_core_guard_reads_a_tag_from_before_inf76(repo):
    _git(repo, "tag", "core-v2.0.0")
    out = _core_guard(repo)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "core-v2.0.0" in out.stdout


def test_the_core_guard_refuses_a_core_changed_since_its_tag(repo):
    _git(repo, "tag", "-a", "-m", "release", "aec-v2.0.0")
    new = repo / "packages" / "aec-agent-core" / "aec_agent_core" / \
        "foundry_dataset_store.py"
    new.write_text("STORE = 1\n", "utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "a module moves into the core")
    out = _core_guard(repo)
    assert out.returncode == 1
    assert "has changed since aec-v2.0.0" in out.stdout
    assert "foundry_dataset_store.py" in out.stdout


def test_the_core_guard_refuses_without_a_tag(repo):
    out = _core_guard(repo)
    assert out.returncode == 1
    assert "No tag aec-v2.0.0 or core-v2.0.0" in out.stdout


# -- the wait for the core, run against a local stand-in for PyPI --------

class _FakePyPI(http.server.BaseHTTPRequestHandler):
    published: set = set()
    core_after = 0  # the core's page answers 200 from this request on
    asked = 0

    def do_GET(self):  # noqa: N802
        cls = type(self)
        if "aec-agent-core" in self.path:
            cls.asked += 1
            ok = cls.asked > cls.core_after
        else:
            ok = any(p in self.path for p in cls.published)
        self.send_response(200 if ok else 404)
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *a):
        pass


@pytest.fixture
def pypi():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakePyPI)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _FakePyPI.published = {"/planlens/"}
    _FakePyPI.asked = 0
    yield server, f"http://127.0.0.1:{server.server_address[1]}/pypi"
    server.shutdown()


def test_document_review_waits_until_the_core_is_up(tmp_path, pypi):
    server, url = pypi
    _FakePyPI.core_after = 2  # up on the third look
    step = _step(PYPI_STEP, "publish-document-review.yml")
    tree = _tree(tmp_path, "2.0.0", "2.0.0", "2.0.0")
    out = _run_step(step, tree, PYPI_JSON_URL=url, CORE_WAIT_SECONDS="30",
                    CORE_POLL_SECONDS="0.05")
    assert out.returncode == 0, out.stdout + out.stderr
    assert "not on PyPI yet; waiting" in out.stdout
    assert "aec-agent-core 2.0.0 is on PyPI." in out.stdout
    assert _FakePyPI.asked == 3


def test_document_review_stops_waiting_at_the_bound(tmp_path, pypi):
    server, url = pypi
    _FakePyPI.core_after = 10 ** 6  # never
    step = _step(PYPI_STEP, "publish-document-review.yml")
    tree = _tree(tmp_path, "2.0.0", "2.0.0", "2.0.0")
    out = _run_step(step, tree, PYPI_JSON_URL=url, CORE_WAIT_SECONDS="0.3",
                    CORE_POLL_SECONDS="0.05")
    assert out.returncode == 1
    assert "after waiting" in out.stdout


def test_planlens_missing_is_refused_without_waiting(tmp_path, pypi):
    server, url = pypi
    _FakePyPI.core_after = 0
    _FakePyPI.published = set()
    step = _step(PYPI_STEP, "publish-document-review.yml")
    tree = _tree(tmp_path, "2.0.0", "2.0.0", "2.0.0")
    out = _run_step(step, tree, PYPI_JSON_URL=url, CORE_WAIT_SECONDS="600",
                    CORE_POLL_SECONDS="0.05")
    assert out.returncode == 1
    assert "planlens==0.15.1, which is not on PyPI," in out.stdout
