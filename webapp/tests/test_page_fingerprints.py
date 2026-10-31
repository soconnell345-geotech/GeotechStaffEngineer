"""Each page's agent, pinned: the app split must not change what it is handed.

BACKLOG INF-30 (``module_work/APP_SPLIT_PLAN.md`` §4). What is pinned, per
build (the review page's default, ``lean`` and ``minimal`` builds, and the
geotech page), with every switch at its default: the system prompt; each
tool's name, description and argument schema; each helper's name,
description, prompt, tools and middleware; the middleware classes in order,
with our budgets, their texts and the tools they hide; and the recursion
limit the build asks for. Texts are pinned as hashes, so a failure names
WHAT changed (which build, which helper, which tool, which field) and
``git diff`` of the code says how.

**When this test fails.**

* In a step of the split (INF-30 to INF-35): do NOT re-record. The step
  changed what a page's agent is handed; find the moved text or the import
  that now resolves differently and put it back. The one exception is the
  deliberate change the plan names at step 12 (Document Review without the
  reference library: ``plot_data`` loses ``reference_chart``, ``plot_code``
  loses ``chart_lines``), re-recorded in its own commit.
* After a deliberate change to a prompt, a tool description or schema, a
  helper or a budget (any other branch): re-record, in a commit of its own
  that says what changed and why, from the repository root::

      python -m webapp.tests.page_builds record

  Never fold a re-record into a split step's commit, and never re-record to
  make an unexplained difference go away.

**The planlens part.** The document tools planlens serves follow the
installed planlens, so they are recorded under the planlens they were built
with (``planlens`` in the file, keyed by a hash of its tool specs) and
compared only against the same planlens. ``record`` replaces the entry for
the installed planlens and keeps the others: the PyPI release CI installs
and the planlens main a developer often has. On CI a planlens with no entry
FAILS rather than skips (a new planlens release on PyPI: record it, in a
venv with that release, or with planlens main at its tag); elsewhere it
skips.

**The app part** is the same on every install but for two tools:
``plot_data`` offers ``reference_chart``, and ``plot_code``'s description
names the standard charts, only where the reference library has
``chart_lines`` (BACKLOG INF-58). What they read is recorded beside the
fingerprints (``app_environment``: the charts of the submodule CI
installs). Where the installed library differs (geotech-references 1.4.1
from PyPI has no ``chart_lines``), the differences in those two tools are
named and the test skips, and any other difference still fails; on CI it
fails rather than skips.

**Cost.** One child interpreter builds all four agents (about 14 s on the
Windows laptop); the check that the test has teeth builds one more in
process (about 8 s).
"""

import importlib.util
import json
import os

import pytest

from webapp.tests import page_builds

pytestmark = pytest.mark.skipif(
    any(importlib.util.find_spec(m) is None
        for m in ("deepagents", "langchain", "planlens")),
    reason="the agent stack or planlens is not installed")


@pytest.fixture(scope="module")
def golden():
    with open(page_builds.GOLDEN, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def current():
    """The fingerprints of this tree, built in a fresh interpreter."""
    return page_builds.run_child(["fingerprints"])


def test_every_build_is_recorded(golden):
    assert sorted(golden["builds"]) == sorted(page_builds.BUILDS)


_REFRESH = (
    "\nIn a split step (INF-30 to INF-35): do NOT re-record; find what "
    "moved (the one exception is plan step 12, named in this test's "
    "docstring). After a deliberate change to a prompt, "
    "tool, helper or budget: re-record in a commit of its own that says "
    "why, from the repository root: python -m webapp.tests.page_builds "
    "record. Texts are hashed; git diff of the code shows how they "
    "changed.")


def _absent_module_prompt_only(diff):
    """BACKLOG INF-61: the module table in the geotech prompt lists the
    modules the registry has, so where an optional package is missing
    (PyNiteFEA on numpy 1.26: 90 characters, the pynite row) only the
    prompt's hash and length differ. Returns the skip reason for that
    case, else None; a difference anywhere else still fails."""
    from funhouse_agent.adapters import ABSENT_MODULES
    if ABSENT_MODULES and diff and all(
            d.rsplit("/", 1)[-1].startswith(("system_prompt:",
                                             "system_prompt_chars:"))
            for d in diff):
        return ("the module table in the system prompt omits "
                + ", ".join(f"{m} ({p} is not installed)"
                            for m, p in sorted(ABSENT_MODULES.items()))
                + "; every other part of the build matches")
    return None


@pytest.mark.parametrize("name", sorted(page_builds.BUILDS))
def test_app_fingerprint_unchanged(name, golden, current):
    diff = page_builds.differences(golden["builds"][name]["app"],
                                   current["builds"][name]["app"])
    recorded_env = golden.get("app_environment")
    env = current.get("app_environment")
    library = []
    if env != recorded_env:
        library, diff = page_builds.split_library_differences(diff)
    absent = _absent_module_prompt_only(diff)
    assert absent or not diff, (
        f"The {name} build changed ({len(diff)} differences):\n  "
        + "\n  ".join(diff[:40]) + _REFRESH)
    if absent:
        pytest.skip(absent)
    if library:
        message = (
            f"The {name} build's {' and '.join(page_builds.LIBRARY_TOOLS)} "
            f"follow the installed reference library, which differs from "
            f"the one recorded (recorded {recorded_env}, installed {env}): "
            f"{len(library)} differences there, none elsewhere. Install "
            "the reference library CI installs (pip install -e "
            "geotech-references/) to compare them.")
        if os.environ.get("GITHUB_ACTIONS") == "true":
            pytest.fail(message)
        pytest.skip(message)


@pytest.mark.parametrize("name", sorted(page_builds.BUILDS))
def test_planlens_tools_unchanged(name, golden, current):
    surface = current["planlens_surface"]
    recorded = golden["planlens"].get(surface)
    if recorded is None:
        message = (
            f"No planlens tools are recorded for the installed planlens "
            f"({current.get('planlens_version')}, surface {surface}; "
            f"recorded: "
            + ", ".join(f"{v['planlens']} ({s})"
                        for s, v in sorted(golden["planlens"].items()))
            + "). Record them with this planlens installed: "
              "python -m webapp.tests.page_builds record")
        if os.environ.get("GITHUB_ACTIONS") == "true":
            pytest.fail(message)
        pytest.skip(message)
    diff = page_builds.differences(recorded["builds"][name],
                                   current["builds"][name]["planlens"])
    assert not diff, (
        f"The {name} build's planlens tools changed (planlens "
        f"{recorded['planlens']}):\n  " + "\n  ".join(diff[:40]) + _REFRESH)


def test_fingerprint_sees_a_changed_prompt(monkeypatch):
    """The check has teeth: one more sentence in the review prompt changes
    the prompt's hash and nothing else."""
    page_kwargs = page_builds.page_kwargs

    def changed(page):
        kw = page_kwargs(page)
        kw["extra_system_prompt"] = "One more sentence."
        return kw

    before = page_builds.fingerprint("review_lean")["app"]
    monkeypatch.setattr(page_builds, "page_kwargs", changed)
    after = page_builds.fingerprint("review_lean")["app"]
    diff = page_builds.differences(before, after)
    assert diff and all("system_prompt" in d for d in diff), diff


def test_a_swapped_tool_order_fails():
    """Tools are hashed by name, so the order is recorded as a list and a
    reorder (which changes what the model sees first) is a difference."""
    old = {"tools": {"a": {"x": 1}, "b": {"x": 2}}, "tool_order": ["a", "b"]}
    new = {"tools": {"b": {"x": 2}, "a": {"x": 1}}, "tool_order": ["b", "a"]}
    diff = page_builds.differences(old, new)
    assert diff and all("tool_order" in d for d in diff), diff
    assert not page_builds.differences(old, dict(old))


def test_only_the_library_tools_are_set_apart():
    """Under another reference library only plot_code's and plot_data's
    changed texts are set apart; a tool gone, or a difference anywhere
    else, still fails."""
    diff = ["/calls[0]/tools/plot_data/args: 'a' -> 'b'",
            "/calls[0]/subagents[1]/tools/plot_code/description: 'a' -> 'b'",
            "/calls[0]/tools/plot_code: gone",
            "/calls[0]/tools/plot_data_extra/args: 'a' -> 'b'",
            "/calls[0]/system_prompt: 'a' -> 'b'"]
    library, rest = page_builds.split_library_differences(diff)
    assert library == diff[:2]
    assert rest == diff[2:]


def test_the_recorded_environment_is_the_ci_one(golden):
    """The recorded charts are the ones a reference library with
    chart_lines names, so CI, which installs the submodule, compares."""
    charts = golden["app_environment"]["reference_chart_lines"]
    assert charts and all(isinstance(c, str) for c in charts)


def test_every_recorded_call_has_its_tool_order(golden):
    for name, build in golden["builds"].items():
        for call in build["app"]["calls"]:
            assert list(call["tools"]) == call["tool_order"] or                 sorted(call["tools"]) == sorted(call["tool_order"]), name


def test_record_keeps_the_other_planlens_entries():
    """Recording replaces the installed planlens' entry and keeps the
    others, so a developer on planlens main never drops CI's."""
    current = {"planlens_surface": "main", "planlens_version": "0.14",
               "builds": {"b": {"app": {"x": 1}, "planlens": ["new"]}}}
    old = {"builds": {"b": {"app": {"x": 0}}},
           "planlens": {"pypi": {"planlens": "0.13", "builds": {"b": ["ci"]}},
                        "main": {"planlens": "0.13", "builds": {"b": ["old"]}}}}
    rec = page_builds.golden_from(current, old)
    assert rec["builds"] == {"b": {"app": {"x": 1}}}
    assert rec["planlens"] == {
        "pypi": {"planlens": "0.13", "builds": {"b": ["ci"]}},
        "main": {"planlens": "0.14", "builds": {"b": ["new"]}}}
