"""Each page's agent, pinned: the app split must not change what it is handed.

BACKLOG INF-30 (``module_work/APP_SPLIT_PLAN.md`` §4). Every step of the
split marked "no behaviour change" leaves these fingerprints identical. A
deliberate change re-records them in its own commit, with the reason:

    python -m webapp.tests.page_builds record

What is pinned, per build (the review page's default, ``lean`` and
``minimal`` builds, and the geotech page), with every switch at its
default: the system prompt; each tool's name, description and argument
schema; each helper's name, description, prompt, tools and middleware; the
middleware classes in order, with our budgets, their texts and the tools
they hide; and the recursion limit the build asks for. Texts are pinned as
hashes, so a failure names WHAT changed; ``git diff`` of the code says how.

The document tools planlens serves are pinned apart, and compared only
where the installed planlens is the one they were recorded with (CI
installs planlens from PyPI; a developer often has planlens main).
"""

import importlib.util
import json

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


@pytest.mark.parametrize("name", sorted(page_builds.BUILDS))
def test_app_fingerprint_unchanged(name, golden, current):
    diff = page_builds.differences(golden["builds"][name]["app"],
                                   current["builds"][name]["app"])
    assert not diff, (
        f"The {name} build changed ({len(diff)} differences):\n  "
        + "\n  ".join(diff[:40])
        + "\nIf the change is deliberate, re-record in its own commit with "
          "the reason: python -m webapp.tests.page_builds record")


@pytest.mark.parametrize("name", sorted(page_builds.BUILDS))
def test_planlens_tools_unchanged(name, golden, current):
    if current["planlens_surface"] != golden["planlens_surface"]:
        pytest.skip("the installed planlens is not the one recorded")
    diff = page_builds.differences(golden["builds"][name]["planlens"],
                                   current["builds"][name]["planlens"])
    assert not diff, (
        f"The {name} build's planlens tools changed:\n  "
        + "\n  ".join(diff[:40]))


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
