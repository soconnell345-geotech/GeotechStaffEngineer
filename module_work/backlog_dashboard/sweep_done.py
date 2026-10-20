"""Move BACKLOG.md's "Done since the last sweep" entries into the archive.

    .venv/Scripts/python module_work/backlog_dashboard/sweep_done.py
    .venv/Scripts/python module_work/backlog_dashboard/sweep_done.py --next GM

The first form is the sweep (owner, 2026-10-20: the Done list had grown to
over half of BACKLOG.md). Every entry under ``## Done since the last sweep``
moves, text unchanged and in its order (newest first), to the top of
``BACKLOG_DONE.md`` under a dated heading; earlier sweeps stay below it. The
Done heading stays in BACKLOG.md with its one-line pointer, so
``close_item`` keeps adding new entries there.

``--next PREFIX`` prints the next free ID for a prefix. IDs are never
reused, and after a sweep the highest number in use may live only in the
archive, so it reads both files.

Options: ``--backlog`` / ``--archive`` (another copy, for a dry run on a
temp dir), ``--date`` (the sweep heading; default today), ``--note`` (added
to the heading, e.g. the master commit), ``--dry-run`` (report only).
"""

import argparse
import io
import pathlib
import re
import sys
from datetime import date

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BACKLOG = ROOT / "BACKLOG.md"
ARCHIVE = ROOT / "BACKLOG_DONE.md"

DONE_HEADING = "## Done since the last sweep"
# The paragraph under the Done heading; close_item inserts new entries
# after it and its blank line.
DONE_INTRO = ("Older docs still list these as open. Items closed before the "
              "last sweep are in `BACKLOG_DONE.md`, newest sweep first.")
ARCHIVE_HEAD = """# BACKLOG_DONE: closed backlog items, by sweep

The archive of `BACKLOG.md`'s "Done since the last sweep" list. At each
sweep, `module_work/backlog_dashboard/sweep_done.py` moves every entry here,
text unchanged, under a dated heading; the newest sweep is on top.

IDs here are never reused. The next free ID for a prefix is the highest
number in `BACKLOG.md` or in this file, plus one:
`python module_work/backlog_dashboard/sweep_done.py --next GM`.
"""
SWEEP_HEADING = "## Swept "
_ID = re.compile(r"\b([A-Z]{2,3})-(\d{2,3})\b")


def _read(path: pathlib.Path) -> str:
    with io.open(path, encoding="utf-8", newline="") as f:
        return f.read()


def _write(path: pathlib.Path, text: str) -> None:
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def split_done(text: str):
    """(before, done_lines, after) around the Done section's body.

    ``before`` ends with the Done heading line; ``done_lines`` are the lines
    under it, with their own line endings; ``after`` is any later ``##``
    section (none today: Done is the last section).
    """
    lines = text.splitlines(keepends=True)
    start = [i for i, ln in enumerate(lines)
             if ln.rstrip("\r\n") == DONE_HEADING]
    if len(start) != 1:
        raise SystemExit(f"{DONE_HEADING!r}: found {len(start)} times, "
                         "expected once")
    s = start[0]
    end = next((j for j in range(s + 1, len(lines))
                if lines[j].startswith("## ")), len(lines))
    return "".join(lines[:s + 1]), lines[s + 1:end], "".join(lines[end:])


def entries(done_lines):
    """The Done section's entries: each bullet with its continuation lines.

    Lines before the first bullet are the intro paragraph, which is replaced
    by ``DONE_INTRO``. A continuation line is any non-blank line that is not
    a new bullet; blank lines are not part of an entry.
    """
    out = []
    first = next((i for i, ln in enumerate(done_lines)
                  if ln.startswith("- ")), None)
    if first is None:
        return out
    for ln in done_lines[first:]:
        if ln.startswith("- "):
            out.append([ln])
        elif ln.strip():
            out[-1].append(ln)
    return ["".join(e) for e in out]


def sweep(backlog: pathlib.Path, archive: pathlib.Path, day: str,
          note: str = "", dry_run: bool = False) -> int:
    text = _read(backlog)
    eol = "\r\n" if "\r\n" in text else "\n"
    before, done_lines, after = split_done(text)
    moved = entries(done_lines)
    if not moved:
        print("nothing to sweep: the Done list is empty")
        return 0
    if not moved[-1].endswith("\n"):
        moved[-1] += eol  # the file's last line had no line ending
    body = "".join(moved)

    heading = f"{SWEEP_HEADING}{day}" + (f" ({note})" if note else "")
    section = f"{heading}{eol}{eol}{body}"
    old_archive = _read(archive) if archive.exists() else ""
    if old_archive:
        cut = old_archive.find(eol + SWEEP_HEADING)
        if cut < 0:
            new_archive = old_archive.rstrip("\r\n") + eol + eol + section
        else:
            cut += len(eol)
            new_archive = old_archive[:cut] + section + eol + old_archive[cut:]
    else:
        new_archive = ARCHIVE_HEAD.replace("\n", eol) + eol + section
    # The intro keeps its blank line, so the list's shape (and close_item's
    # anchor, DONE_INTRO plus a blank line) is the same empty or full.
    new_backlog = before + eol + DONE_INTRO + eol + eol + after

    # Nothing is lost: every entry line of the old Done section is in the
    # new archive, and the rest of BACKLOG.md is as it was.
    kept = [ln.rstrip("\r\n") for e in moved for ln in e.splitlines()]
    arch_lines = set(new_archive.splitlines())
    lost = [ln for ln in kept if ln not in arch_lines]
    if lost or not new_backlog.startswith(before) or after not in new_backlog:
        raise SystemExit(f"sweep check failed: {len(lost)} lines would be lost")

    ids = {f"{p}-{n}" for p, n in re.findall(
        r"^- \*\*([A-Z]{2,3})-(\d{2,3})\b", body, re.M)}
    print(f"{len(moved)} entries ({len(ids)} distinct lead IDs) -> "
          f"{archive.name} under {heading!r}")
    print(f"{backlog.name}: {len(text.encode('utf-8')):,} -> "
          f"{len(new_backlog.encode('utf-8')):,} bytes; {archive.name}: "
          f"{len(old_archive.encode('utf-8')):,} -> "
          f"{len(new_archive.encode('utf-8')):,} bytes")
    if dry_run:
        return 0
    _write(archive, new_archive)
    _write(backlog, new_backlog)
    return 0


def next_id(prefix: str, backlog: pathlib.Path, archive: pathlib.Path) -> str:
    """The highest number in use for ``prefix`` in either file, plus one."""
    nums = [0]
    for p in (backlog, archive):
        if p.exists():
            nums += [int(n) for pre, n in _ID.findall(_read(p)) if pre == prefix]
    return f"{prefix}-{max(nums) + 1:02d}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backlog", type=pathlib.Path, default=BACKLOG)
    ap.add_argument("--archive", type=pathlib.Path, default=None,
                    help="default: BACKLOG_DONE.md beside --backlog")
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--note", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--next", metavar="PREFIX",
                    help="print the next free ID for PREFIX and stop")
    a = ap.parse_args(argv)
    archive = a.archive or a.backlog.with_name(ARCHIVE.name)
    if a.next:
        print(next_id(a.next.upper(), a.backlog, archive))
        return 0
    return sweep(a.backlog, archive, a.date, a.note, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
