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
does one that has gone and is still listed. From step 8 only the soft ones
remain (inside ``try``: the page works without them), and the blocking test
turns into a plain test the page must pass with every GSE module refused.
Until then it is a strict xfail that expects the page's own failure (an
``AssertionError``): an unexpected pass, or the child process itself
failing, is a failure.

What neither the recorder nor the blocker sees: a module loaded from its
file (``importlib.util.spec_from_file_location``), which the plotting box
does with ``geotech_common/layout_ink.py`` (listed below only because the
box's module is also imported in process) and ``plan_furniture.py`` (not
listed; plan 2.7 names it); anything the plotting box's
own child interpreter (``python -I``) imports; and data files read by path.
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
    # 1. The review page is built by GSE's builder (step 8).
    "webapp.core -> funhouse_agent.deep.agent": "2.7 #1, step 8",
    "funhouse_agent.deep.review_agent -> funhouse_agent.deep.agent":
        "2.7 #1, step 8",
    # 2. The review tools' factory sits in GSE's tools module (step 4).
    "funhouse_agent.deep.review_agent -> funhouse_agent.deep.tools":
        "2.7 #2, step 4",
    "funhouse_agent.deep.minimal_agent -> funhouse_agent.deep.tools":
        "2.7 #2, step 4",
    # 3. The package door shows GSE's first-import disclaimer notice, inside
    # a ``try`` (so soft); its other names load on first use since step 2
    # (the disclaimer goes with the package, steps 10-15).
    "funhouse_agent -> funhouse_agent._disclaimer":
        "2.7 #3, soft, step 10",
    # 4. The chart stack behind plot_data and the plotting box (step 6).
    "funhouse_agent.plot_box.child -> geotech_common": "2.7 figure, step 6",
    "funhouse_agent.plot_box.child -> geotech_common.layout_ink":
        "2.7 figure, step 6",
    # 5. (The review prompt's module was GSE's: gone at step 3.)
    # Further couplings (step 5).
    "funhouse_agent.deep.calculate_tool -> "
    "funhouse_agent.adapters.reliability_adapter": "2.7 further, step 5",
    "funhouse_agent.vision_tools -> calc_package.docx_renderer":
        "2.7 further, step 5",
    # The figure helper on the review page (GM-227; steps 6 and 8).
    "funhouse_agent.deep.figure_helper -> funhouse_agent.deep.agent":
        "2.7 figure, step 8",
    "funhouse_agent.deep.figure_helper -> funhouse_agent.deep.tools":
        "2.7 figure, step 4",
    # Soft: inside ``try``, the page works without them (hooks, step 11).
    "funhouse_agent.plot_box -> geotech_references.chart_lines":
        "2.7 soft, step 12",
    "funhouse_agent.vision_tools -> funhouse_agent.reference_docs":
        "2.7 soft, step 11",
    "funhouse_agent.vision_tools -> geotech_references": "2.7 soft, step 11",
    "funhouse_agent.vision_tools -> geotech_references._figures_db":
        "2.7 soft, step 11",
    "webapp.core -> webapp.reference_fetch": "2.7 soft, step 7",
}

#: The couplings inside ``try``, which a page without GSE survives.
SOFT_COUPLINGS = frozenset(k for k, v in KNOWN_COUPLINGS.items()
                           if "soft" in v)

#: "importer -> imported": where GSE's code reaches into the review page's
#: own files (plan §2.7, "geotech code into review code"; steps 8 and 15).
KNOWN_REVERSE_COUPLINGS = {
    "funhouse_agent.deep.agent -> funhouse_agent.deep.review_agent":
        "the review build inside GSE's builder, step 8",
    "funhouse_agent.deep.agent -> funhouse_agent.deep.minimal_agent":
        "the review build inside GSE's builder, step 8",
    "funhouse_agent.deep.agent -> funhouse_agent.review_prompt":
        "the review build inside GSE's builder, step 8",
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
    reason="The review page still imports GSE modules outside a try "
           "(KNOWN_COUPLINGS); from step 8 of the split it must build and "
           "run without them, and this mark lifts itself.")
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
