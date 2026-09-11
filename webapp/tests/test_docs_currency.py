"""Docs-currency guard: the files that ASSERT a current state must name the
version actually in ``pyproject.toml``.

Why this exists. ``CLAUDE.md`` spent 2026-09-09 to 09-11 telling every agent
that read it "BOTH TREES UNCOMMITTED" — written truthfully during the 5.13.0
release and then never refreshed, because the 5.14.0 commit updated
``HANDOFF.md`` (+228 lines) and did not touch ``CLAUDE.md``. Two files carried
the same volatile state, only one was designated authoritative, and the release
updated that one. Nothing detected the contradiction: the trees were committed
and released while the repo's own orientation file said they were not.

This is the same class of defect as the published planlens figures that drifted
twice, and it gets the same remedy that fixed those: make the claim checkable
by the gate instead of by somebody remembering. A version bump now MUST be
accompanied by a touch of each state-asserting doc, or the gate goes red.

These tests deliberately check only that the CURRENT VERSION IS NAMED. They do
not police prose — an agent that bumps the version is forced to open each file,
and once it is open the surrounding staleness is obvious.
"""

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).parents[2]


def _version() -> str:
    """The ``[project]`` version from pyproject.toml, read with a regex.

    Deliberately NOT tomllib: that is 3.11+ stdlib while pyproject declares
    ``requires-python = ">=3.10"``, and on 3.10 the ImportError would take this
    whole file out at collection — silently disabling every guard in it, which
    is the drift it exists to catch.
    """
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    for line in text.split("[project]", 1)[-1].splitlines():
        if line.startswith("["):
            break                      # next table; version must precede it
        match = re.match(r"""version\s*=\s*["']([^"']+)""", line.strip())
        if match:
            return match.group(1)
    raise AssertionError("no [project] version in pyproject.toml")


def _section(path: pathlib.Path, heading_startswith: str) -> str:
    """The text from a ``##`` heading up to the next ``##`` heading."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.startswith("## ") and heading_startswith in line:
            start = i
            break
    assert start is not None, (
        f"{path.name} has no '## ...{heading_startswith}...' heading. If you "
        "renamed it, update this guard so the docs stay checkable.")
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            return "\n".join(lines[start:j])
    return "\n".join(lines[start:])


def test_claude_md_current_state_names_the_shipped_version():
    """CLAUDE.md is loaded into every agent's context — it must not describe a
    superseded release."""
    version = _version()
    section = _section(ROOT / "CLAUDE.md", "CURRENT WORKING STATE")
    assert version in section, (
        f"CLAUDE.md '## CURRENT WORKING STATE' does not mention {version}. "
        "The version was bumped without refreshing the state block, which is "
        "exactly how it came to claim BOTH TREES UNCOMMITTED for two days "
        "after 5.14.0 shipped. Update the section (commit/tag/gate numbers and "
        "what is actually released), not just the version string.")


def test_handoff_pickup_list_names_the_shipped_version():
    """HANDOFF.md §0a-current is the authoritative pickup list."""
    version = _version()
    section = _section(ROOT / "HANDOFF.md", "0a-current")
    assert version in section, (
        f"HANDOFF.md '## 0a-current' does not mention {version}. It is the "
        "file every other doc points at as authoritative; refresh the pickup "
        "list for this release.")


def test_install_triage_history_has_a_row_for_this_version():
    """docs/DATABRICKS_INSTALL.md carries the install-log triage history."""
    version = _version()
    doc = ROOT / "docs" / "DATABRICKS_INSTALL.md"
    assert doc.exists(), (
        "docs/DATABRICKS_INSTALL.md is missing — it is the standing answer to "
        "'here is the install log, any concerns?'. Restore it rather than "
        "re-deriving the triage from scratch a fourth time.")
    text = doc.read_text(encoding="utf-8")
    assert re.search(rf"^\|\s*{re.escape(version)}\s*\|", text, re.M), (
        f"docs/DATABRICKS_INSTALL.md has no history row for {version}. Add "
        "one when the cluster install is confirmed (or note that it has not "
        "been attempted yet) so the next reviewer can see whether this "
        "version has ever actually been installed.")


def test_claude_md_triggers_are_near_the_top():
    """The pointer must survive a SKIM, not merely exist.

    CLAUDE.md is ~73 KB and is auto-loaded whole; a pointer buried at line 52
    is a pointer that gets skimmed past. After a context compaction this file
    and MEMORY.md are the only docs a fresh agent still has, so the triggers
    that send it to HANDOFF.md and the install guide have to be at the top.
    2500 chars is roughly the first screen.
    """
    head = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")[:2500]
    for target in ("docs/DATABRICKS_INSTALL.md", "HANDOFF.md"):
        assert target in head, (
            f"{target} is no longer pointed at in the first 2500 chars of "
            "CLAUDE.md. Keep the READ-FIRST TRIGGERS block at the top: it is "
            "the only thing a post-compaction agent sees before it starts "
            "answering, and re-deriving the install triage from the raw log "
            "is exactly what it exists to prevent.")


@pytest.mark.parametrize("doc", ["CLAUDE.md", "HANDOFF.md"])
def test_state_docs_point_at_the_install_triage_guide(doc):
    """Both orientation docs must route an install-log question to the guide,
    so the triage is not re-derived from the raw log again."""
    text = (ROOT / doc).read_text(encoding="utf-8")
    assert "docs/DATABRICKS_INSTALL.md" in text, (
        f"{doc} no longer points at docs/DATABRICKS_INSTALL.md. That pointer "
        "is the only thing standing between the next agent and a from-scratch "
        "re-derivation of the numpy cascade and the conflict-warning triage.")
