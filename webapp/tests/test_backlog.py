"""Backlog guard: ``BACKLOG.md`` is THE one list of open work, and stays usable.

Why this exists. On 2026-10-12 the owner found that sessions had "lost track
of some of our future plans and started new, more-narrow todo lists in the
handoff doc". A sweep found 278 open items across the app, planlens and
geotech-references; HANDOFF's list held about 60. Everything went into
``BACKLOG.md`` with stable IDs, and HANDOFF §0a-current now names only this
week's slice by ID.

Like ``test_docs_currency.py``, these tests do not police prose. They check
that the structure the next session relies on is intact:
- every item has a unique ID and a tier;
- every ID cited in the backlog or in HANDOFF §0a-current exists;
- CLAUDE.md and HANDOFF still send a session to the backlog.

Closed items leave in two steps: "Done since the last sweep" at the foot of
``BACKLOG.md``, then, at a sweep, ``BACKLOG_DONE.md`` (the archive,
``module_work/backlog_dashboard/sweep_done.py``). An ID closed in either
place still counts as defined when an open item cites it, and is never
reused.
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).parents[2]
BACKLOG = ROOT / "BACKLOG.md"
ARCHIVE = ROOT / "BACKLOG_DONE.md"

PREFIXES = ("OWN", "DR", "VIS", "TA", "PL", "RI", "GM", "REF", "EV", "INF",
            "LT", "DOC")
TIERS = ("Now", "Next", "Later", "Parked")
_ITEM = re.compile(r"^- \*\*([A-Z]{2,3}-\d{2,3})\*\* \[([A-Za-z]+)\]")
_ID = re.compile(r"\b(" + "|".join(PREFIXES) + r")-(\d{2,3})\b")


def _items():
    """(id, tier) for every item line in BACKLOG.md."""
    out = []
    for line in BACKLOG.read_text(encoding="utf-8").splitlines():
        m = _ITEM.match(line)
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def _section(path: pathlib.Path, heading_startswith: str) -> str:
    """The text from a ``##`` heading up to the next ``##`` heading."""
    lines = path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines)
                 if ln.startswith("## ") and heading_startswith in ln)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("## "):
            return "\n".join(lines[start:j])
    return "\n".join(lines[start:])


def test_backlog_exists_and_has_items():
    assert BACKLOG.exists(), "BACKLOG.md is the one list of open work"
    assert len(_items()) > 0


def test_ids_are_unique_known_and_tiered():
    seen = {}
    for item_id, tier in _items():
        assert item_id.split("-")[0] in PREFIXES, (
            f"{item_id}: unknown prefix; use one of {PREFIXES} or add the new "
            "area to this guard and to BACKLOG.md's sections")
        assert tier in TIERS, f"{item_id}: tier [{tier}] is not one of {TIERS}"
        assert item_id not in seen, (
            f"{item_id} is used twice in BACKLOG.md; IDs are never reused, "
            "take the next free number")
        seen[item_id] = tier


_DONE = "## Done since the last sweep"
_DONE_ITEM = re.compile(r"^- \*\*([A-Z]{2,3}-\d{2,3})\*\*", re.M)


def _done_ids():
    """Closed IDs (bold at the start of a bullet): BACKLOG.md's Done list
    plus every earlier sweep in the archive."""
    text = BACKLOG.read_text(encoding="utf-8")
    done = text.split(_DONE, 1)[1] if _DONE in text else ""
    if ARCHIVE.exists():
        done += "\n" + ARCHIVE.read_text(encoding="utf-8")
    return set(_DONE_ITEM.findall(done))


def test_ids_cited_in_the_backlog_exist():
    defined = {i for i, _ in _items()}
    text = BACKLOG.read_text(encoding="utf-8")
    cited = {f"{p}-{n}" for p, n in _ID.findall(text)}
    missing = sorted(cited - defined - _done_ids())
    assert not missing, (
        f"BACKLOG.md cites IDs no item defines: {missing}. Fix the reference "
        "or add the item.")


def test_closed_ids_are_not_reused():
    reused = sorted(_done_ids() & {i for i, _ in _items()})
    assert not reused, (
        f"{reused} are listed as done (BACKLOG.md or BACKLOG_DONE.md) AND as "
        "open items; a closed ID is never reused, take the next free number "
        "(sweep_done.py --next PREFIX)")


def test_archive_holds_only_swept_entries():
    """The archive is sweeps of Done entries, nothing open."""
    if not ARCHIVE.exists():
        return
    text = ARCHIVE.read_text(encoding="utf-8")
    assert "## Swept " in text, (
        "BACKLOG_DONE.md has no '## Swept <date>' heading; write it with "
        "module_work/backlog_dashboard/sweep_done.py, not by hand")
    tiered = re.findall(r"^- \*\*([A-Z]{2,3}-\d{2,3})\*\* \[(?:"
                        + "|".join(TIERS) + r")\]", text, re.M)
    assert not tiered, (
        f"{tiered} sit in BACKLOG_DONE.md as tiered open items; open work "
        "lives only in BACKLOG.md")


def test_sweep_keeps_every_closed_id_and_open_item(tmp_path):
    """A sweep of today's files (on copies) loses no closed ID or open item."""
    import importlib.util
    import shutil
    spec = importlib.util.spec_from_file_location(
        "sweep_done", ROOT / "module_work" / "backlog_dashboard" / "sweep_done.py")
    sweep_done = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep_done)
    shutil.copy(BACKLOG, tmp_path / "BACKLOG.md")
    if ARCHIVE.exists():
        shutil.copy(ARCHIVE, tmp_path / "BACKLOG_DONE.md")
    before = _done_ids()
    open_before = _items()
    sweep_done.sweep(tmp_path / "BACKLOG.md", tmp_path / "BACKLOG_DONE.md",
                     "2000-01-01")
    swept = (tmp_path / "BACKLOG.md").read_text(encoding="utf-8")
    archive = (tmp_path / "BACKLOG_DONE.md")
    after = set(_DONE_ITEM.findall(swept.split(_DONE, 1)[1]))
    if archive.exists():
        after |= set(_DONE_ITEM.findall(archive.read_text(encoding="utf-8")))
    assert after == before
    assert [(m.group(1), m.group(2)) for m in map(_ITEM.match, swept.splitlines())
            if m] == open_before
    assert sweep_done.next_id("GM", tmp_path / "BACKLOG.md", archive) == (
        sweep_done.next_id("GM", BACKLOG, ARCHIVE))


def test_apply_triage_moves_merged_and_closed_items_to_done(tmp_path):
    """A triage's merges and closes leave the open list as Done entries, so
    every ID the backlog cites stays defined."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "apply_triage", ROOT / "module_work" / "backlog_dashboard" / "apply_triage.py")
    apply_triage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(apply_triage)
    backlog = tmp_path / "BACKLOG.md"
    backlog.write_text(
        "# Backlog\n\n## 7. Modules (GM-nn)\n\n"
        "- **GM-01** [Later] **One**: text. *Src:* a\n"
        "- **GM-02** [Later] **Two**: see GM-01.\n"
        "- **GM-03** [Later] **Three**: text.\n"
        "- **GM-04** [Later] **Four**: text.\n\n"
        f"{_DONE}\n\nIntro.\n\n- **GM-05** closed. Commit `abc1234`.\n",
        encoding="utf-8")
    rows = [
        {"id": "GM-01", "action": "MERGE", "target": "GM-03", "reason": "same",
         "evidence": "", "merge_text": "one's text."},
        {"id": "GM-02", "action": "CLOSE-COVERED", "target": "",
         "reason": "built", "evidence": "commit abc1234"},
        {"id": "GM-04", "action": "DROP?", "target": "", "reason": "low",
         "evidence": ""},
    ]
    assert apply_triage.apply(backlog, rows, "2000-01-01") == 0
    text = backlog.read_text(encoding="utf-8")
    open_part, done = text.split(_DONE, 1)
    assert re.findall(r"^- \*\*(GM-\d+)\*\*", open_part, re.M) == ["GM-03", "GM-04"]
    assert "Also (was GM-01, merged 2000-01-01): one's text." in open_part
    assert _DONE_ITEM.findall(done) == ["GM-01", "GM-02", "GM-05"]
    # the owner's drops later, from the same file: applied rows are skipped
    assert apply_triage.apply(backlog, rows, "2000-01-02", drop=["GM-04"]) == 0
    text = backlog.read_text(encoding="utf-8")
    open_part, done = text.split(_DONE, 1)
    assert re.findall(r"^- \*\*(GM-\d+)\*\*", open_part, re.M) == ["GM-03"]
    assert _DONE_ITEM.findall(done) == ["GM-04", "GM-01", "GM-02", "GM-05"]
    # a merge target that is not open refuses, changing nothing
    backlog.write_text(text.replace("- **GM-03** [Later]",
                                    "- **GM-03** [Later] see\n- **GM-06** [Later]"),
                       encoding="utf-8")
    before = backlog.read_text(encoding="utf-8")
    bad = {"id": "GM-06", "action": "MERGE", "target": "GM-09",
           "reason": "", "evidence": "", "merge_text": "x"}
    assert apply_triage.apply(backlog, [bad], "2000-01-03") == 1
    assert backlog.read_text(encoding="utf-8") == before


def test_handoff_slice_cites_only_existing_ids():
    defined = {i for i, _ in _items()}
    section = _section(ROOT / "HANDOFF.md", "0a-current")
    assert "BACKLOG.md" in section, (
        "HANDOFF §0a-current must point at BACKLOG.md as the one list")
    cited = {f"{p}-{n}" for p, n in _ID.findall(section)}
    missing = sorted(cited - defined)
    assert not missing, (
        f"HANDOFF §0a-current cites backlog IDs that do not exist: {missing}")


def test_claude_md_sends_sessions_to_the_backlog():
    head = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")[:6000]
    assert "BACKLOG.md" in head, (
        "CLAUDE.md's read-first block must name BACKLOG.md, so a fresh "
        "session finds the one list")


def test_every_owner_item_has_a_desk_card():
    """The Backlog Desk explains every item that waits on the owner."""
    import json
    cards = {}
    for p in (ROOT / "module_work" / "backlog_dashboard" / "context").glob("*.json"):
        cards.update(json.loads(p.read_text(encoding="utf-8")))
    text = BACKLOG.read_text(encoding="utf-8")
    owner = re.findall(r"^- \*\*([A-Z]{2,3}-\d{2,3})\*\* \[[A-Za-z]+\] \[owner\]",
                       text, re.M)
    missing = sorted(set(owner) - set(cards))
    assert not missing, (
        f"[owner] items without a Backlog Desk card: {missing}. Add one to "
        "module_work/backlog_dashboard/context/*.json (fields in its README), "
        "then rebuild and republish the page.")
