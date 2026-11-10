"""The review method as deployment instructions and as a suite arm
(BACKLOG DR-06), offline.

The method is a draft: nothing loads it unless ``GEOTECH_INSTRUCTIONS``
names its folder. These tests hold that it fits what the loader takes,
reaches the page's agent and its reading helper when asked for, names no
tool the page does not have, and runs as an arm of the review suite.
"""

import os
import re

import pytest

from aec_agent_core import review_flags
from aec_agent_core.deep import instructions as ins
from aec_document_review import review_method
from aec_document_review.review_eval import method_arm, score_review_suite
from tests import page_builds


def _main_call(calls):
    return next(c for c in calls if c["kind"] == "create_deep_agent")


def test_the_two_files_ship_with_the_package_and_fit_the_loader():
    assert sorted(os.listdir(review_method.FOLDER))[:1] != []
    for role in review_method.ROLES:
        text = review_method.text(role)
        # under the loader's cap, so nothing is cut when it is appended
        assert 200 < len(text) <= ins.CAP_CHARS, (role, len(text))
        assert "\r" not in text.replace("\r\n", "\n")
    with pytest.raises(KeyError):
        review_method.text("calc")
    main = review_method.text("main")
    for heading in ("### Before the sheets", "### On each sheet",
                    "### Across sheets", "### What a comment rests on",
                    "### The comment log"):
        assert heading in main
    # it cites no code or standard and gives no clause number
    assert not re.search(r"\b(?:ACI|AISC|ASCE|IBC|NFPA|ASTM|EN \d|BS \d)",
                         main)


def test_unset_the_page_is_unchanged():
    agent, calls, _ = page_builds.build("review_default", capture=True)
    assert "appended" not in agent.geotech_build
    assert "Reviewing a drawing set" not in _main_call(
        calls)["kwargs"]["system_prompt"]


def test_named_the_method_reaches_the_agent_and_its_reading_helper():
    agent, calls, _ = page_builds.build(
        "review_default", capture=True,
        settings={ins.ENV: review_method.FOLDER})
    kw = _main_call(calls)["kwargs"]
    prompt = kw["system_prompt"]
    assert prompt.rstrip().endswith(review_method.text("main").strip())
    assert prompt.count(ins.PROTECTED_RULES) == 1
    reader = next(s for s in kw["subagents"]
                  if s.get("name") == "general-purpose")
    assert reader["system_prompt"].rstrip().endswith(
        review_method.text("general-purpose").strip())
    record = agent.geotech_build["appended"]
    assert record["roles"]["main"]["files"] == ["main.md"]
    assert record["roles"]["general-purpose"]["files"] == [
        "general-purpose.md"]
    assert not record["roles"]["main"]["cut"]
    # nothing in the folder is left unused, and nothing else is read
    assert not record.get("unused") and not record.get("file_errors")


def test_the_method_names_only_tools_the_page_has():
    """A prompt that names a tool the agent lacks sends it looking for it
    (BACKLOG DR-42). The method names its tools in backticks."""
    agent, calls, _ = page_builds.build("review_default", capture=True)
    kw = _main_call(calls)["kwargs"]
    have = {getattr(t, "name", None) for t in kw["tools"]}
    have |= {"write_todos", "task"}              # deepagents' own
    for role in review_method.ROLES:
        named = set(re.findall(r"`([a-z_]+)`", review_method.text(role)))
        assert named <= have, (role, named - have)


def test_the_suite_arm():
    name, env = method_arm()
    assert name == "method"
    assert env == {ins.ENV: review_method.FOLDER}
    assert method_arm("method_r2")[0] == "method_r2"
    # the arm sets it for its runs and the baseline that follows is clean
    with review_flags.switches(env):
        assert os.environ[ins.ENV] == review_method.FOLDER
    assert ins.ENV not in os.environ


def test_the_dry_run_takes_the_arm(tmp_path, monkeypatch):
    monkeypatch.delenv(ins.ENV, raising=False)
    out = score_review_suite(out_dir=str(tmp_path), dry_run=True,
                             arms=["baseline", method_arm()],
                             ids=["coord-sheet-index"])
    assert "1 tasks x 2 arms" in out["results_md"]
    assert ins.ENV not in os.environ
