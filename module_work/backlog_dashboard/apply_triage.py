"""Apply a backlog triage file to BACKLOG.md: merges, closes and agreed drops.

    .venv/Scripts/python module_work/backlog_dashboard/apply_triage.py module_work/backlog_dashboard/triage_2026-10-20.json
    ... --drop OWN-23,VIS-18      (the DROP? rows the owner agreed to)
    ... --dry-run                 (report only)

A triage file is a list of ``{id, action, target, reason, evidence}`` rows
(``merge_text`` on a MERGE). For each row:

- ``MERGE``: the item's line leaves BACKLOG.md, its ``merge_text`` is added
  to the end of the target item as "Also (was ID, merged DATE): ...", the
  backlog's own convention for a follow-up, and a Done entry records the
  merge.
- ``CLOSE-COVERED``: the item's line leaves and a Done entry gives the
  reason and the evidence.
- ``DROP?``: applied only when the ID is named in ``--drop`` (the owner's
  call); the line leaves and a Done entry says the owner dropped it.
- ``KEEP``: untouched.

Every Done entry starts ``- **ID**``, so the ID stays defined for whatever
still cites it and is never reused (``webapp/tests/test_backlog.py``). A row
whose ID is already closed (in Done or ``BACKLOG_DONE.md``) is skipped, so
the same file runs again with ``--drop`` once the owner answers. The script
refuses, changing nothing, when an item or a target is not an open item, or
a target is itself leaving. It prints the HANDOFF §0a-current
lines that cite a removed ID: the backlog test accepts only open IDs there,
so those lines need editing in the same commit.
"""

import argparse
import io
import json
import pathlib
import re
import sys
from datetime import date

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BACKLOG = ROOT / "BACKLOG.md"
DONE_HEADING = "## Done since the last sweep"
_ITEM = re.compile(r"^- \*\*([A-Z]{2,3}-\d{2,3})\*\* \[[A-Za-z]+\]")
_DONE_ID = re.compile(r"^- \*\*([A-Z]{2,3}-\d{2,3})\*\*", re.M)


def _read(path: pathlib.Path) -> str:
    with io.open(path, encoding="utf-8", newline="") as f:
        return f.read()


def _write(path: pathlib.Path, text: str) -> None:
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def _body(line: str) -> str:
    return line.rstrip("\r\n")


def done_entry(row: dict, day: str) -> str:
    """The Done line for one applied row."""
    reason = row.get("reason", "").strip()
    evidence = row.get("evidence", "").strip()
    if row["action"] == "MERGE":
        return (f"- **{row['id']}** (merged into {row['target']} by the backlog "
                f"tidy, {day}): {reason}")
    if row["action"] == "CLOSE-COVERED":
        tail = f" Evidence: {evidence}" if evidence else ""
        return (f"- **{row['id']}** (closed as covered by the backlog tidy, "
                f"{day}): {reason}{tail}")
    return f"- **{row['id']}** (dropped by the owner, {day}): {reason}"


def apply(backlog: pathlib.Path, rows: list, day: str, drop=(),
          dry_run: bool = False) -> int:
    text = _read(backlog)
    eol = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines(keepends=True)
    done_at = next((i for i, ln in enumerate(lines)
                    if _body(ln) == DONE_HEADING), None)
    if done_at is None:
        print(f"{DONE_HEADING!r} not found")
        return 1
    where = {}
    for i, ln in enumerate(lines[:done_at]):
        m = _ITEM.match(ln)
        if m:
            where[m.group(1)] = i

    # Rows already applied (the ID closed in Done or the archive) are
    # skipped, so the same file can be run again with --drop later.
    closed = set(_DONE_ID.findall("".join(lines[done_at:])))
    archive = backlog.with_name("BACKLOG_DONE.md")
    if archive.exists():
        closed |= set(_DONE_ID.findall(_read(archive)))
    drop = {d.strip().upper() for d in drop if d.strip()}
    chosen = [r for r in rows
              if r["action"] in ("MERGE", "CLOSE-COVERED")
              or (r["action"] == "DROP?" and r["id"] in drop)]
    done_already = [r["id"] for r in chosen
                    if r["id"] not in where and r["id"] in closed]
    if done_already:
        print(f"already applied, skipped: {', '.join(done_already)}")
    chosen = [r for r in chosen if r["id"] not in done_already]
    leaving = {r["id"] for r in chosen}
    errors = []
    unknown_drop = drop - {r["id"] for r in rows if r["action"] == "DROP?"}
    if unknown_drop:
        errors.append(f"--drop names IDs that are not DROP? rows: {sorted(unknown_drop)}")
    for r in chosen:
        i = where.get(r["id"])
        if i is None:
            errors.append(f"{r['id']}: not an open item in {backlog.name}")
            continue
        nxt = _body(lines[i + 1]) if i + 1 < len(lines) else ""
        if nxt.strip() and not nxt.startswith(("- ", "#", "|")):
            errors.append(f"{r['id']}: the next line continues the item "
                          f"({nxt[:40]!r}...); fix it by hand first")
        if r["action"] == "MERGE":
            if r.get("target") not in where:
                errors.append(f"{r['id']}: merge target {r.get('target')} is "
                              "not an open item")
            elif r["target"] in leaving:
                errors.append(f"{r['id']}: merge target {r['target']} is "
                              "itself leaving")
    if errors:
        print("nothing changed:\n  " + "\n  ".join(errors))
        return 1

    appends = {}
    for r in chosen:
        if r["action"] == "MERGE" and r.get("merge_text", "").strip():
            appends.setdefault(r["target"], []).append(
                f"Also (was {r['id']}, merged {day}): {r['merge_text'].strip()}")
    gone = {where[r["id"]] for r in chosen}
    out = []
    for i, ln in enumerate(lines):
        if i in gone:
            continue
        m = _ITEM.match(ln) if i < done_at else None
        if m and m.group(1) in appends:
            ln = (_body(ln) + " " + " ".join(appends[m.group(1)])
                  + ln[len(_body(ln)):])
        out.append(ln)

    # The Done entries go on top of the Done list (newest first); with an
    # empty list, after the intro paragraph and its blank line.
    start = next(i for i, ln in enumerate(out) if _body(ln) == DONE_HEADING)
    end = next((j for j in range(start + 1, len(out))
                if out[j].startswith("## ")), len(out))
    first = next((j for j in range(start + 1, end)
                  if out[j].startswith("- ")), None)
    new = [done_entry(r, day) + eol for r in chosen]
    if not new:
        pass
    elif first is not None:
        out[first:first] = new
    else:
        tail = end
        if tail > 0 and not out[tail - 1].endswith("\n"):
            out[tail - 1] += eol
        if tail > 0 and out[tail - 1].strip():
            new.insert(0, eol)
        out[tail:tail] = new
    new_text = "".join(out)

    counts = {}
    for r in chosen:
        counts[r["action"]] = counts.get(r["action"], 0) + 1
    print(f"{len(chosen)} rows applied {counts}; {len(appends)} targets "
          f"extended; {backlog.name}: {len(text.encode('utf-8')):,} -> "
          f"{len(new_text.encode('utf-8')):,} bytes")
    cite = re.compile(r"\b(" + "|".join(map(re.escape, sorted(leaving)))
                      + r")\b") if leaving else None
    handoff = backlog.with_name("HANDOFF.md")
    if cite and handoff.exists():
        hl = handoff.read_text(encoding="utf-8").splitlines()
        s = next((i for i, ln in enumerate(hl)
                  if ln.startswith("## ") and "0a-current" in ln), None)
        if s is not None:
            e = next((j for j in range(s + 1, len(hl))
                      if hl[j].startswith("## ")), len(hl))
            for j in range(s, e):
                found = sorted(set(cite.findall(hl[j])))
                if found:
                    print(f"EDIT HANDOFF.md:{j + 1} cites {found}: "
                          f"{hl[j].strip()[:100]}")
    if cite:
        for j, ln in enumerate(new_text.split(DONE_HEADING, 1)[0].splitlines()):
            ln = re.sub(r"\(was [A-Z]{2,3}-\d{2,3}, merged [^)]*\)", "", ln)
            found = sorted(set(cite.findall(ln)))
            if found:
                print(f"note {backlog.name}:{j + 1} still cites {found} "
                      "(defined by its Done entry; point it at the target "
                      "if that reads better)")
    if not dry_run:
        _write(backlog, new_text)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("triage", type=pathlib.Path)
    ap.add_argument("--backlog", type=pathlib.Path, default=BACKLOG)
    ap.add_argument("--drop", default="",
                    help="comma-separated DROP? IDs the owner agreed to")
    ap.add_argument("--date", default=date.today().isoformat())
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    rows = json.loads(a.triage.read_text(encoding="utf-8"))
    return apply(a.backlog, rows, a.date, a.drop.split(","), a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
