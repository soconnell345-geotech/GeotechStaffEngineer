"""The two wheels this repository builds (the app split's step 10, INF-32):
geotech-staff-engineer from the root and aec-agent-core from
``packages/aec-agent-core``.

Why: the 5.19.0 lesson. A wheel can build and install happily and simply
lack a package (``report_ingest`` was one ``include`` line from missing), or
carry what must never ship. So the slow test builds BOTH wheels the way a
release does (sdist first, then the wheel from the sdist), installs them into
a FRESH venv and imports the entry points there, and lists every file in
them.

The fresh venv gets the third-party packages it needs (the agent stack,
streamlit, ...) by a plain path line to the running interpreter's
site-packages. A path line does not run that folder's ``.pth`` files, so the
editable installs of this repository are NOT visible in it: an import that
works there came out of the wheels. Each imported module's file is checked
to be inside the venv all the same.

How it runs:
- the fast tests below (versions and pins in step) run everywhere;
- the build tests are marked ``slow`` and are OPT-IN: they run only when
  ``GEOTECH_REQUIRE_WHEEL_TEST=1`` is set, and otherwise skip, saying so.
  They build the app's sdist and wheel from the repository root (under a
  minute), and setuptools writes into the source tree while it does, so two
  runs at once (two xdist workers, or a local run beside another) must
  never build it. With the variable set, a missing ``build`` package is a
  failure, not a skip. CI's "Wheels" job (tests.yml) installs ``build``,
  sets that variable and runs this file alone, in one process.
- On Windows, Smart App Control may refuse to run the freshly built
  ``geotech-disclaimer.exe`` (WinError 4551); the console-script check then
  skips with that reason. CI runs it on Linux.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import re
import subprocess
import sys
import sysconfig
import textwrap
import zipfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
CORE = ROOT / "packages" / "aec-agent-core"
CORE_NAME = "aec-agent-core"


def _project_version(pyproject: pathlib.Path) -> str:
    # A regex rather than tomllib, as test_docs_currency does (written while
    # the floor was 3.10, which has none).
    text = pyproject.read_text(encoding="utf-8")
    for line in text.split("[project]", 1)[-1].splitlines():
        if line.startswith("["):
            break
        m = re.match(r"""version\s*=\s*["']([^"']+)""", line.strip())
        if m:
            return m.group(1)
    raise AssertionError(f"no [project] version in {pyproject}")


def _core_pin() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    pins = re.findall(rf'"{CORE_NAME}\s*==\s*([^"\s;]+)"', text)
    assert len(pins) == 1, f"expected one exact {CORE_NAME} pin, found {pins}"
    return pins[0]


# -- fast: versions and pins in step ---------------------------------------

def test_the_app_pins_the_core_exactly_at_its_version():
    assert _core_pin() == _project_version(CORE / "pyproject.toml")


def test_the_core_module_names_its_own_version():
    text = (CORE / "aec_agent_core" / "__init__.py").read_text(
        encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.M)
    assert m and m.group(1) == _project_version(CORE / "pyproject.toml")


def test_the_core_carries_the_app_s_licence():
    assert (CORE / "LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()


def test_the_app_does_not_ship_the_core_and_the_core_ships_only_itself():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = text.split("[tool.setuptools.packages.find]", 1)[1]
    include = block.split("exclude", 1)[0]
    assert "aec_agent_core" not in include and '"packages' not in include
    core = (CORE / "pyproject.toml").read_text(encoding="utf-8")
    cblock = core.split("[tool.setuptools.packages.find]", 1)[1]
    assert re.search(r'include\s*=\s*\["aec_agent_core\*"\]', cblock)


def test_publish_builds_the_root_only():
    wf = (ROOT / ".github" / "workflows" / "publish.yml").read_text(
        encoding="utf-8")
    builds = re.findall(r"python -m build[^\n]*", wf)
    assert builds and all(b.rstrip().endswith(" .") for b in builds), builds
    # INF-76: the pinned-core guard names the core's folder to diff it
    # against the core's tag; it builds nothing there.
    assert "packages/" not in wf.replace(
        "packages/aec-agent-core is NOT", "").replace(
        'CORE = "packages/aec-agent-core/"', "")


def test_ci_installs_the_core_before_the_app():
    for name in ("tests.yml", "nightly.yml"):
        wf = (ROOT / ".github" / "workflows" / name).read_text(
            encoding="utf-8")
        core = re.search(r"^\s*pip install[^\n]* -e packages/aec-agent-core",
                         wf, re.M)
        app = re.search(r"^\s*pip install -e \.\s*$", wf, re.M)
        assert core, f"{name} does not install the core"
        assert app is None or core.start() < app.start(), \
            f"{name} installs the app first"


# -- slow: build, install, import, list ------------------------------------

#: Path parts that must never be in a wheel.
FORBIDDEN_PARTS = {"tests", "module_work", "raw", "packages"}
#: Modules imported in the fresh venv: the packages' entry points, all
#: import-safe (no Streamlit page runs on import).
IMPORTS = {
    # The core and what step 11 (a) moved into it (each imports with only
    # the core's own requirements and planlens).
    "aec_agent_core": (
        "aec_agent_core", "aec_agent_core.hooks",
        "aec_agent_core.engine_config", "aec_agent_core.palantir_sdk_engine",
        "aec_agent_core.http_trace", "aec_agent_core.tinyapps_engine",
        "aec_agent_core.vision_tools", "aec_agent_core.vision_view",
        "aec_agent_core.image_view", "aec_agent_core.vision_probe",
        "aec_agent_core.inline_store", "aec_agent_core.turn_question",
        "aec_agent_core.error_text", "aec_agent_core._fileio",
        "aec_agent_core.deep", "aec_agent_core.deep.vision_engine",
        "aec_agent_core.deep.databricks_bridge",
        "aec_agent_core.deep.pacing", "aec_agent_core.deep.inline_images",
        # step 11 (b): the chart stack, the plotting box, the figure helper
        "aec_agent_core.charts", "aec_agent_core.charts.data_plot",
        "aec_agent_core.charts.plot_tool", "aec_agent_core.charts.layout_ink",
        "aec_agent_core.charts.plan_furniture",
        "aec_agent_core.charts.plotting",
        "aec_agent_core.charts.figure_examples", "aec_agent_core.plot_box",
        "aec_agent_core.plot_box.child", "aec_agent_core.plot_box.figures",
        "aec_agent_core.deep.figure_helper", "aec_agent_core.drawing_wait",
        "aec_agent_core.progress_notes", "aec_agent_core.arg_checks",
        "aec_agent_core.method_call",
        # step 11 (c): guards and records, Word and Excel, the checks, the
        # document tools' binding, the assembly and the suite's framework
        "aec_agent_core.activity_log", "aec_agent_core.cited_documents",
        "aec_agent_core.cited_pages", "aec_agent_core.coverage",
        "aec_agent_core.document_tools", "aec_agent_core.docx_renderer",
        "aec_agent_core.dxf_notes", "aec_agent_core.measure_tool",
        "aec_agent_core.office_text", "aec_agent_core.review_flags",
        "aec_agent_core.run_mirror", "aec_agent_core.runtime_check",
        "aec_agent_core.safe_expr", "aec_agent_core.scale_labels",
        "aec_agent_core.xlsx_writer", "aec_agent_core.deep.assembly",
        "aec_agent_core.deep.busy_retry", "aec_agent_core.deep.calculate_tool",
        "aec_agent_core.deep.delivered_log",
        "aec_agent_core.deep.held_records", "aec_agent_core.deep.keep_work",
        "aec_agent_core.deep.limits", "aec_agent_core.deep.result_map",
        "aec_agent_core.deep.scratch_guard",
        "aec_agent_core.deep.strict_args", "aec_agent_core.deep.tool_guards",
        "aec_agent_core.deep.vision_kit", "aec_agent_core.deep.warm_imports",
        "aec_agent_core.review_eval", "aec_agent_core.review_eval.runner",
        "aec_agent_core.review_eval.checks",
        "aec_agent_core.review_eval.documents",
        "aec_agent_core.review_eval.tasks",
    ),
    "geotech-staff-engineer": (
        "funhouse_agent", "funhouse_agent._disclaimer",
        "funhouse_agent.deep", "funhouse_agent.deep.agent",
        "webapp", "webapp.core", "webapp.profiles",
        "webapp.databricks_launcher", "report_ingest", "geotech_common",
        "calc_package", "profile_figure",
    ),
}
#: Page entry points: present in the wheel, not executed (they run a page).
PAGES = ("webapp.app", "webapp.pages_entry", "webapp.tinyapps_entry",
         "webapp.foundry_entry")


def _need_build():
    if os.environ.get("GEOTECH_REQUIRE_WHEEL_TEST") != "1":
        pytest.skip("the wheel build test is opt-in: set "
                    "GEOTECH_REQUIRE_WHEEL_TEST=1 and run this file alone, in "
                    "one process (CI's Wheels job does)")
    if importlib.util.find_spec("build") is None:
        pytest.fail("the wheel test needs the 'build' package "
                    "(pip install build)")


def _run(cmd, **kw):
    proc = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if proc.returncode:
        raise AssertionError(
            f"{' '.join(map(str, cmd))} failed ({proc.returncode}):\n"
            f"{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}")
    return proc


def _build(project: pathlib.Path, out: pathlib.Path) -> pathlib.Path:
    """sdist, then the wheel from the sdist, as a release builds.

    --no-isolation uses the setuptools already installed, so no network.
    Building from the sdist also keeps setuptools' build/ folder out of the
    source tree (a stale build/ once put a deleted module in a wheel).
    """
    _run([sys.executable, "-m", "build", "--no-isolation", "--outdir",
          str(out), str(project)])
    wheels = sorted(out.glob("*.whl"))
    assert len(wheels) == 1, wheels
    assert len(sorted(out.glob("*.tar.gz"))) == 1
    return wheels[0]


@pytest.fixture(scope="module")
def wheels(tmp_path_factory):
    _need_build()
    base = tmp_path_factory.mktemp("wheels")
    return {"geotech-staff-engineer": _build(ROOT, base / "app"),
            CORE_NAME: _build(CORE, base / "core")}


def _names(wheel: pathlib.Path) -> list[str]:
    with zipfile.ZipFile(wheel) as z:
        return z.namelist()


@pytest.mark.slow
def test_neither_wheel_ships_what_must_not_ship(wheels):
    for dist, wheel in wheels.items():
        names = _names(wheel)
        bad = [n for n in names
               if FORBIDDEN_PARTS & set(n.split("/")[:-1])
               or "tinyapps/reference" in n
               or n.endswith((".pyc", ".env"))]
        assert not bad, f"{dist}: {bad[:20]}"


@pytest.mark.slow
def test_the_core_wheel_is_the_core_alone(wheels):
    names = _names(wheels[CORE_NAME])
    tops = {n.split("/")[0] for n in names}
    assert {t for t in tops if not t.endswith(".dist-info")} == \
        {"aec_agent_core"}, tops
    assert "aec_agent_core/__init__.py" in names
    for moved in ("aec_agent_core/hooks.py", "aec_agent_core/vision_tools.py",
                  "aec_agent_core/engine_config.py",
                  "aec_agent_core/deep/__init__.py",
                  "aec_agent_core/deep/vision_engine.py",
                  "aec_agent_core/charts/layout_ink.py",
                  "aec_agent_core/plot_box/child.py",
                  "aec_agent_core/deep/figure_helper.py",
                  # step 11 (c)
                  "aec_agent_core/deep/assembly.py",
                  "aec_agent_core/document_tools.py",
                  "aec_agent_core/activity_log.py",
                  "aec_agent_core/review_eval/runner.py",
                  "aec_agent_core/review_eval/tasks.py"):
        assert moved in names, moved
    assert any(n.endswith(".dist-info/licenses/LICENSE") for n in names)
    with zipfile.ZipFile(wheels[CORE_NAME]) as z:
        meta = z.read(next(n for n in names
                           if n.endswith(".dist-info/METADATA"))).decode()
    assert f"Version: {_project_version(CORE / 'pyproject.toml')}" in meta
    assert 'Provides-Extra: anthropic' in meta
    assert "Requires-Dist: langchain-anthropic" in meta


@pytest.mark.slow
def test_the_app_wheel_has_its_packages_and_data_and_pins_the_core(wheels):
    names = set(_names(wheels["geotech-staff-engineer"]))
    assert not any(n.startswith("aec_agent_core/") for n in names)
    for must in ("report_ingest/__init__.py", "webapp/__init__.py",
                 "funhouse_agent/__init__.py",
                 "webapp/ws_upload_component/index.html",
                 "funhouse_agent/geotech_test_suite.json",
                 "funhouse_agent/worked_examples.json",
                 "report_ingest/report_review_checklist.json"):
        assert any(n == must or n.startswith(must + "/") for n in names), must
    meta_name = next(n for n in names if n.endswith(".dist-info/METADATA"))
    with zipfile.ZipFile(wheels["geotech-staff-engineer"]) as z:
        meta = z.read(meta_name).decode()
    assert f"Requires-Dist: {CORE_NAME}=={_core_pin()}" in meta


def _site_dirs() -> list[str]:
    """The running interpreter's site-packages folders (the third-party
    packages), plus the folder of any package the app imports that is an
    editable install of ANOTHER repository (geotech-references), never a
    folder of this repository."""
    dirs = []
    for key in ("purelib", "platlib"):
        d = sysconfig.get_paths()[key]
        if d not in dirs:
            dirs.append(d)
    for mod in ("geotech_references", "planlens"):
        spec = importlib.util.find_spec(mod)
        if spec is None or not spec.origin:
            continue
        # the folder that holds the package: site-packages (already in), or
        # an editable checkout such as the geotech-references submodule,
        # which holds that package alone. Never this repository's root.
        holder = pathlib.Path(spec.origin).resolve().parents[1]
        if holder != ROOT and str(holder) not in dirs:
            dirs.append(str(holder))
    return dirs


@pytest.fixture(scope="module")
def venv(wheels, tmp_path_factory):
    base = tmp_path_factory.mktemp("venv")
    env_dir = base / "env"
    _run([sys.executable, "-m", "venv", str(env_dir)])
    py = env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    purelib = _run([str(py), "-c",
                    "import sysconfig; print(sysconfig.get_paths()"
                    "['purelib'])"]).stdout.strip()
    (pathlib.Path(purelib) / "_borrowed_site.pth").write_text(
        "\n".join(_site_dirs()) + "\n", encoding="utf-8")
    _run([str(py), "-m", "pip", "install", "--no-deps", "--no-index",
          "--disable-pip-version-check", "-q",
          str(wheels[CORE_NAME]), str(wheels["geotech-staff-engineer"])],
         cwd=str(base))
    return env_dir, py


@pytest.mark.slow
def test_the_entry_points_import_from_the_installed_wheels(venv):
    env_dir, py = venv
    mods = [*IMPORTS["aec_agent_core"], *IMPORTS["geotech-staff-engineer"]]
    script = textwrap.dedent(f"""
        import importlib, importlib.util, importlib.metadata as md, os, sys
        env = os.path.realpath({str(env_dir)!r})
        bad = []
        for name in {mods!r}:
            m = importlib.import_module(name)
            f = os.path.realpath(getattr(m, "__file__", "") or "")
            if not f.startswith(env + os.sep):
                bad.append((name, f))
        for name in {list(PAGES)!r}:
            spec = importlib.util.find_spec(name)
            if spec is None or not os.path.realpath(spec.origin).startswith(
                    env + os.sep):
                bad.append((name, spec and spec.origin))
        assert not bad, bad
        import aec_agent_core
        assert md.version("aec-agent-core") == aec_agent_core.__version__
        reqs = md.requires("geotech-staff-engineer")
        assert "aec-agent-core==" + aec_agent_core.__version__ in reqs, reqs
        # the old names of what moved into the core are the core's modules
        import funhouse_agent, webapp
        for old, new in {{**funhouse_agent.CORE_ALIASES,
                         **webapp.CORE_ALIASES}}.items():
            assert importlib.import_module(old) is \
                importlib.import_module(new), old
        print("ok", len({mods!r}))
        """)
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME")}
    env["GEOTECH_NO_DISCLAIMER"] = "1"
    # -I: no PYTHONPATH, no user site, and not the current folder, which
    # is set outside the repository anyway.
    out = _run([str(py), "-I", "-c", script], cwd=str(env_dir), env=env)
    assert out.stdout.strip().startswith("ok")


@pytest.mark.slow
def test_the_console_script_runs_from_the_wheel(venv):
    env_dir, _ = venv
    exe = env_dir / ("Scripts/geotech-disclaimer.exe" if os.name == "nt"
                     else "bin/geotech-disclaimer")
    assert exe.exists()
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    try:
        out = _run([str(exe)], cwd=str(env_dir), env=env)
    except OSError as exc:
        if getattr(exc, "winerror", None) != 4551:
            raise
        pytest.skip("Windows refused to run the freshly built "
                    "geotech-disclaimer.exe (WinError 4551: an Application "
                    "Control policy, Smart App Control); CI runs this check "
                    "on Linux")
    assert "disclaimer" in (out.stdout + out.stderr).lower()
