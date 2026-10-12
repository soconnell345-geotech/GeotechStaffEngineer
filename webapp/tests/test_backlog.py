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
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).parents[2]
BACKLOG = ROOT / "BACKLOG.md"

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
    """IDs closed since the last sweep (bold at the start of a Done bullet)."""
    text = BACKLOG.read_text(encoding="utf-8")
    done = text.split(_DONE, 1)[1] if _DONE in text else ""
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
        f"{reused} are listed as done AND as open items; a closed ID is never "
        "reused, take the next free number")


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
