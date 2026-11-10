"""Document Review's page agent, pinned (BACKLOG INF-33, the app split's
step 13; copied from GeotechStaffEngineer's ``test_page_fingerprints.py``).

What is pinned, per build (the page's default build and its ``minimal``
build), with every switch at its default: the system prompt; each tool's
name, description and argument schema; each helper's name, description,
prompt, tools and middleware; the middleware classes in order, with the
budgets, their texts and the tools they hide; and the recursion limit the
build asks for. Texts are pinned as hashes.

These fingerprints started at step 13 from GeotechStaffEngineer's, with two
deliberate changes. The split's: the figure helper's ``plot_data`` has no
``reference_chart`` and its ``plot_code`` no ``chart_lines()`` (no standard
chart lines are registered here). The step's review fixes': one line of the
default build's prompt ("What you are not") no longer sends the person to
the GeotechStaffEngineer page, which a Document Review deployment does not
have. Everything else equals that app's record of its review page at the
copy.

**When this test fails** after a deliberate change to a prompt, a tool, a
helper or a budget: re-record, in a commit of its own that says what changed
and why, from ``packages/aec-document-review``::

    python -m tests.page_builds record

**The planlens part** follows the installed planlens, so it is recorded
under the planlens it was built with and compared only against the same
one; on CI a planlens with no entry fails rather than skips.
"""

import json
import os

import pytest

from tests import page_builds


@pytest.fixture(scope="module")
def golden():
    with open(page_builds.GOLDEN, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def current():
    """The fingerprints of this tree, built in a fresh interpreter."""
    return page_builds.run_child(["fingerprints"])


_REFRESH = ("\nAfter a deliberate change to a prompt, tool, helper or "
            "budget: re-record in a commit of its own that says why, from "
            "packages/aec-document-review: python -m tests.page_builds "
            "record.")


def test_every_build_is_recorded(golden):
    assert sorted(golden["builds"]) == sorted(page_builds.BUILDS)


def test_no_chart_lines_are_registered(golden, current):
    """The split's deliberate change stays made: nothing registers the
    standard chart lines on this page."""
    assert golden["app_environment"] == {"reference_charts_registered": False}
    assert current["app_environment"] == golden["app_environment"]


@pytest.mark.parametrize("name", sorted(page_builds.BUILDS))
def test_app_fingerprint_unchanged(name, golden, current):
    diff = page_builds.differences(golden["builds"][name]["app"],
                                   current["builds"][name]["app"])
    assert not diff, (f"The {name} build changed ({len(diff)} "
                      f"differences):\n  " + "\n  ".join(diff[:40])
                      + _REFRESH)


@pytest.mark.parametrize("name", sorted(page_builds.BUILDS))
def test_planlens_tools_unchanged(name, golden, current):
    surface = current["planlens_surface"]
    recorded = golden["planlens"].get(surface)
    if recorded is None:
        message = (f"No planlens tools are recorded for the installed "
                   f"planlens ({current.get('planlens_version')}, surface "
                   f"{surface}). Record them with this planlens installed: "
                   "python -m tests.page_builds record")
        if os.environ.get("GITHUB_ACTIONS") == "true":
            pytest.fail(message)
        pytest.skip(message)
    diff = page_builds.differences(recorded["builds"][name],
                                   current["builds"][name]["planlens"])
    assert not diff, (f"The {name} build's planlens tools changed:\n  "
                      + "\n  ".join(diff[:40]) + _REFRESH)


def test_fingerprint_sees_a_changed_prompt(monkeypatch):
    """The check has teeth: one more sentence in the prompt changes the
    prompt's hash and nothing else."""
    page_kwargs = page_builds.page_kwargs

    def changed(page):
        kw = page_kwargs(page)
        kw["extra_system_prompt"] = "One more sentence."
        return kw

    before = page_builds.fingerprint("review_default")["app"]
    monkeypatch.setattr(page_builds, "page_kwargs", changed)
    after = page_builds.fingerprint("review_default")["app"]
    diff = page_builds.differences(before, after)
    assert diff and all("system_prompt" in d for d in diff), diff
