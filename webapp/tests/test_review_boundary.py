"""The line between Document Review and GeotechStaffEngineer, measured.

BACKLOG INF-30 (``module_work/APP_SPLIT_PLAN.md`` §2.7 and §4). The review
page's three builds are built and run for one turn on a fake model, in a
fresh interpreter, and every import that crosses from shared or review
code into GSE's own modules (the geotech packages, the reference library,
and the GSE files of ``funhouse_agent`` and ``webapp``) is recorded. The
turn calls the tools whose code reaches into geotech modules when it runs
(``plot_data``, ``calculate``, ``write_docx``) and one helper; a tool the
turn does not call can still hide a coupling, which the docreview-only CI
job of step 12 catches.

The known couplings below may only SHRINK: a new one fails the test, and so
does one that has gone and is still listed. Since step 8 only the soft ones
remain (inside ``try``: the page works without them), so the blocking test
is a plain test the page must pass with every GSE module refused (its
strict-xfail mark lifts itself while only soft couplings are listed). The
tests at the end read the split's exits off the source: the chart stack
(step 6), the shell (step 7), and the review build and the assembly both
builds share (step 8) import nothing of GSE outside the listed hook.

What neither the recorder nor the blocker sees: a module loaded from its
file (``importlib.util.spec_from_file_location``), which the plotting box
does with ``layout_ink.py``, ``plan_furniture.py`` (both in
``funhouse_agent/charts/`` since step 6) and the chart-lines file an app
registered; anything the plotting box's own child interpreter
(``python -I``) imports; and data files read by path.
The docreview-only CI job of step 12, with no geotech package installed,
is the check for those.
"""

import importlib.util
import subprocess
import sys

import pytest

from webapp.tests import page_builds

pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(m) is None
        for m in ("deepagents", "langchain", "planlens")),
    reason="the agent stack or planlens is not installed")


#: "importer -> imported": where the review page reaches into GSE today,
#: with the plan's item (§2.7) and the step that cuts it.
KNOWN_COUPLINGS = {
    # 1. (The review page was built by GSE's builder: gone at step 8; it
    # builds itself, funhouse_agent.review_build, on the assembly both
    # builds share, funhouse_agent.deep.assembly.)
    # 2. (The review tools' factory sat in GSE's tools module: gone at
    # step 4, now the vision kit.)
    # 3. The package door shows GSE's first-import disclaimer notice, inside
    # a ``try`` (so soft); its other names load on first use since step 2.
    # The disclaimer goes when the review files leave the package at step
    # 13 (the earlier plan's "step 10"; INF-31 relabelled it).
    "funhouse_agent -> funhouse_agent._disclaimer":
        "2.7 #3, soft, step 13",
    # 4. (The chart stack behind plot_data and the plotting box: gone at
    # step 6, now funhouse_agent.charts; plot_data runs through
    # funhouse_agent.method_call, not GSE's dispatch.)
    # 5. (The review prompt's module was GSE's: gone at step 3.)
    # (Further couplings: calculate's evaluator, the Word renderer and the
    # figure helper's result cap, gone at step 5.)
    # (The plotting box's look for geotech_references.chart_lines: gone at
    # step 6; GSE registers the chart lines with funhouse_agent.charts.)
    # Soft: inside ``try``, the page works without them.
    # The review build hands the reference library's chart lines to the
    # chart stack as the geotech page does (a hook GSE fills), so the figure
    # helper reads as before; step 13 drops it from Document Review (the
    # one deliberate fingerprint change; the earlier plan's "step 12").
    "funhouse_agent.review_build -> "
    "funhouse_agent.adapters.profile_figure_adapter": "2.7 soft, step 13",
    # (vision_tools -> reference_docs, geotech_references and _figures_db:
    # gone at step 11 (a); vision_tools is the shared core's, and asks the
    # hook vision_tools.reference_read_roots for the reference library's
    # folders.) GSE fills that hook from its package door, which the review
    # page still loads, so the read tools reach the reference folders as
    # before; inside a ``try``. Step 13 leaves it unfilled for Document
    # Review.
    "aec_agent_core.hooks -> funhouse_agent.reference_vision":
        "2.7 soft, step 13",
    # (webapp.core -> webapp.reference_fetch: gone at step 7; the geotech
    # page's on_build hook registers the fetcher.)
}

#: The couplings inside ``try``, which a page without GSE survives.
SOFT_COUPLINGS = frozenset(k for k, v in KNOWN_COUPLINGS.items()
                           if "soft" in v)

#: "importer -> imported": where GSE's code reaches into the review page's
#: own files (plan §2.7, "geotech code into review code"; steps 8 and 15).
KNOWN_REVERSE_COUPLINGS = {
    # (The review build inside GSE's builder: gone at step 8;
    # build_deep_agent(review_page=True) hands its callers to
    # funhouse_agent.review_build until step 15.)
    # The old import path of the review prompt (step 3), re-exported.
    "funhouse_agent.deep.prompt -> funhouse_agent.review_prompt":
        "re-export for the old import path, step 15",
}


@pytest.fixture(scope="module")
def review_run():
    """The review page's three builds, built and run once, recording."""
    return page_builds.run_child(
        ["boundary", "record", ",".join(page_builds.REVIEW_BUILDS)])


@pytest.fixture(scope="module")
def geotech_run():
    """The geotech page, built and run once, recording."""
    return page_builds.run_child(["boundary", "record", "geotech_default"])


def _built(run):
    failed = {n: b.get("error") for n, b in run["builds"].items()
              if not b.get("ok")}
    assert not failed, f"a build or its turn failed: {failed}"


def test_review_page_reaches_into_gse_only_where_known(review_run):
    _built(review_run)
    seen = set(review_run["into_gse"])
    new = sorted(seen - set(KNOWN_COUPLINGS))
    gone = sorted(set(KNOWN_COUPLINGS) - seen)
    assert not new, (
        "The review page now imports GeotechStaffEngineer code it did not "
        "before (the split's line, plan §2.7):\n  " + "\n  ".join(new))
    assert not gone, (
        "These couplings are gone. Take them out of KNOWN_COUPLINGS (the "
        "list only shrinks):\n  " + "\n  ".join(gone))


def test_gse_reaches_into_review_files_only_where_known(review_run,
                                                        geotech_run):
    _built(geotech_run)
    seen = set(review_run["gse_into_review"]) | set(
        geotech_run["gse_into_review"])
    new = sorted(seen - set(KNOWN_REVERSE_COUPLINGS))
    gone = sorted(set(KNOWN_REVERSE_COUPLINGS) - seen)
    assert not new, (
        "GeotechStaffEngineer's code now imports the review page's own "
        "files:\n  " + "\n  ".join(new))
    assert not gone, (
        "These are gone. Take them out of KNOWN_REVERSE_COUPLINGS:\n  "
        + "\n  ".join(gone))


@pytest.mark.xfail(
    condition=bool(set(KNOWN_COUPLINGS) - SOFT_COUPLINGS), strict=True,
    raises=AssertionError,
    reason="The review page imports GSE modules outside a try "
           "(KNOWN_COUPLINGS); since step 8 of the split it must build and "
           "run without them, and this mark lifts itself while only soft "
           "couplings are listed.")
def test_review_page_works_with_gse_modules_blocked():
    run = page_builds.run_child(
        ["boundary", "block", ",".join(page_builds.REVIEW_BUILDS)])
    failed = {n: b.get("error") for n, b in run["builds"].items()
              if not b.get("ok")}
    assert not failed, (
        f"With every GSE module refused ({run.get('refused')}), "
        f"the review page failed: {failed}")


def test_the_blocker_reads_as_not_installed():
    """A refused module looks as it would with the geotech packages not
    installed: ``find_spec`` answers None, an import raises."""
    code = ("import importlib.util\n"
            "from webapp.tests import page_builds as pb\n"
            "pb._Blocker().install()\n"
            "print(importlib.util.find_spec('geotech_common') is None,\n"
            "      importlib.util.find_spec('funhouse_agent.deep.agent')"
            " is None,\n"
            "      importlib.util.find_spec('json') is not None)\n"
            "import funhouse_agent.deep.agent\n")
    proc = subprocess.run([sys.executable, "-c", code], cwd=page_builds.REPO,
                          capture_output=True, text=True, timeout=120)
    assert proc.stdout.split() == ["True", "True", "True"], proc.stderr
    assert ("blocked for the boundary test: funhouse_agent.deep.agent"
            in proc.stderr), proc.stderr


# ---------------------------------------------------------------------------
# The split's exits, read off the source (INF-31): what a step cut stays cut
# even where the turn above does not reach it.
# ---------------------------------------------------------------------------

def _imports_of(path: str) -> set:
    """Every module ``path`` imports, at any depth (inside functions too),
    with relative imports resolved."""
    import ast
    import os
    rel = os.path.relpath(path, page_builds.REPO).replace(os.sep, "/")
    module = rel[:-3].replace("/", ".")
    if module.endswith(".__init__"):
        package = module[:-len(".__init__")]
    else:
        package = module.rpartition(".")[0]
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")
                up = parts[:len(parts) - node.level + 1]
                base = ".".join(up + ([base] if base else []))
            out.add(base)
            out |= {f"{base}.{a.name}" for a in node.names if a.name != "*"}
    return out


def _python_files(*where: str) -> list:
    import os
    files = []
    for w in where:
        path = os.path.join(page_builds.REPO, w)
        if path.endswith(".py"):
            files.append(path)
            continue
        for root, dirs, names in os.walk(path):
            dirs[:] = [d for d in dirs if d not in ("tests", "__pycache__")]
            files += [os.path.join(root, n) for n in names
                      if n.endswith(".py")]
    return sorted(files)


#: Step 6's exit (plan §4): the chart stack, the plotting box and the call
#: plot_data runs through import nothing of GeotechStaffEngineer; the
#: reference charts are handed in (``funhouse_agent.charts.
#: register_reference_charts``).
CHART_STACK = ("funhouse_agent/charts", "funhouse_agent/plot_box",
               "funhouse_agent/method_call.py",
               "funhouse_agent/arg_checks.py")


def test_the_chart_stack_imports_nothing_of_gse():
    found = []
    for path in _python_files(*CHART_STACK):
        found += [f"{path} -> {m}" for m in sorted(_imports_of(path))
                  if page_builds.is_geotech(m)]
    assert not found, ("The chart stack imports GeotechStaffEngineer code "
                       "(INF-31 step 6):\n  " + "\n  ".join(found))


#: Step 7's exit (plan §4): the shell asks the page, not geotech. These
#: three files import nothing of these (the geotech page's own parts are in
#: webapp/geotech_page.py, reached through the page's hooks).
SHELL = ("webapp/app.py", "webapp/core.py", "webapp/diagnostics.py")
NOT_IN_THE_SHELL = ("funhouse_agent.dispatch", "funhouse_agent.reviewers",
                    "funhouse_agent._disclaimer", "funhouse_agent.disclaimer",
                    "funhouse_agent.reference_docs",
                    "funhouse_agent.worked_examples", "geotech_references",
                    "webapp.reference_fetch", "webapp.geotech_page")


def test_the_shell_imports_nothing_of_geotech():
    found = []
    for path in _python_files(*SHELL):
        found += [f"{path} -> {m}" for m in sorted(_imports_of(path))
                  if any(m == n or m.startswith(n + ".")
                         for n in NOT_IN_THE_SHELL)]
    assert not found, ("The shell imports the geotech page's own code "
                       "(INF-31 step 7):\n  " + "\n  ".join(found))


#: Step 8's exit (plan §4): the review page builds itself and the assembly
#: both builds share imports nothing of GSE; the review build's one reach,
#: the chart-lines hook inside a ``try``, is listed (it goes at step 13).
REVIEW_BUILD = ("funhouse_agent/review_build.py",
                "funhouse_agent/deep/assembly.py")
REVIEW_BUILD_HOOKS = {
    "funhouse_agent/review_build.py -> "
    "funhouse_agent.adapters.profile_figure_adapter":
        "the reference charts, soft, step 13",
}


def test_the_review_build_and_the_assembly_import_nothing_of_gse():
    import os
    found = set()
    for path in _python_files(*REVIEW_BUILD):
        rel = os.path.relpath(path, page_builds.REPO).replace(os.sep, "/")
        mods = _imports_of(path)
        # a module, not the names imported from it
        gse = {m for m in mods if page_builds.is_geotech(m)}
        found |= {f"{rel} -> {m}" for m in gse
                  if not any(m.startswith(o + ".") for o in gse)}
    assert found == set(REVIEW_BUILD_HOOKS), (
        "The review build or the shared assembly imports GeotechStaffEngineer "
        "code other than the listed hooks (INF-31 step 8):\n  "
        + "\n  ".join(sorted(found ^ set(REVIEW_BUILD_HOOKS))))
