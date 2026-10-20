"""Build the Backlog Desk page from BACKLOG.md plus the owner-card context.

    .venv/Scripts/python module_work/backlog_dashboard/build.py

Reads ``BACKLOG.md`` (the one list of open work), ``BACKLOG_DONE.md`` (the
archive of earlier sweeps; only its IDs and sweep dates) and
``context/*.json`` (plain-English cards for the items that need the
owner). Writes
``out/backlog_desk.html``, which a session publishes as a private
claude.ai artifact. See README.md beside this file for the artifact's URL
and how answers come back.
"""

import json
import pathlib
import re
import subprocess
import sys
from datetime import date

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BACKLOG = ROOT / "BACKLOG.md"
ARCHIVE = ROOT / "BACKLOG_DONE.md"
TEMPLATE = HERE / "template.html"
OUT = HERE / "out" / "backlog_desk.html"

_ITEM = re.compile(
    r"^- \*\*(?P<id>[A-Z]{2,3}-\d{2,3})\*\* \[(?P<tier>[A-Za-z]+)\]"
    r"(?P<owner> \[owner\])? (?P<rest>.*)$")
_TITLE = re.compile(r"^\*\*(?P<title>.+?)\*\*:?\s*(?P<body>.*)$")
_SECTION = re.compile(r"^## (?:(?P<num>\d+)\. )?(?P<name>.+?)\s*$")
_PREFIX = re.compile(r"\(([A-Z]{2,3})-nn\)")


def _split_refs(body: str):
    """Pull ``*Src:* ...`` and ``*was:* ...`` off the end of an item body."""
    src = was = ""
    m = re.search(r"\*was:\*\s*(.+)$", body)
    if m:
        was = m.group(1).strip()
        body = body[:m.start()].rstrip(" ;")
    m = re.search(r"\*Src:\*\s*(.+)$", body)
    if m:
        src = m.group(1).strip().rstrip(";").strip()
        body = body[:m.start()].rstrip(" ;")
    return body.strip(), src, was


def parse_backlog(text: str) -> dict:
    lines = text.splitlines()
    data = {"sections": [], "items": [], "this_week": [], "switches": [],
            "done": [], "header": {}}
    m = re.search(r"Last full sweep:\*\*\s*(.+)", text)
    data["header"]["last_sweep"] = m.group(1).strip() if m else ""
    section = None
    sub = None
    in_table = False
    for raw in lines:
        line = raw.rstrip()
        sm = _SECTION.match(line)
        if sm and line.startswith("## "):
            name = sm.group("name")
            num = sm.group("num")
            pm = _PREFIX.search(name)
            section = {"num": int(num) if num is not None else None,
                       "name": _PREFIX.sub("", name).strip(),
                       "prefix": pm.group(1) if pm else None,
                       "kind": ("week" if num == "0" else
                                "done" if name.startswith("Done") else
                                "items" if num is not None else "meta")}
            data["sections"].append(section)
            sub = None
            in_table = False
            continue
        if section is None or section["kind"] == "meta":
            continue
        if line.startswith("### "):
            sub = line[4:].strip()
            in_table = False
            continue
        if section["kind"] == "week" and line.startswith("- "):
            data["this_week"].append(line[2:])
            continue
        if section["kind"] == "done" and line.startswith("- "):
            data["done"].append(line[2:])
            continue
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            if cells and cells[0] == "Switch":
                in_table = True
                continue
            if in_table and len(cells) >= 3:
                data["switches"].append(
                    {"switch": cells[0], "default": cells[1], "awaits": cells[2]})
            continue
        im = _ITEM.match(line)
        if im:
            tm = _TITLE.match(im.group("rest"))
            title, body = ((tm.group("title"), tm.group("body")) if tm
                           else (im.group("rest"), ""))
            body, src, was = _split_refs(body)
            data["items"].append({
                "id": im.group("id"), "tier": im.group("tier"),
                "owner": bool(im.group("owner")), "title": title,
                "body": body, "src": src, "was": was,
                "section": section["num"], "group": sub or ""})
    return data


def archive_summary(path=ARCHIVE) -> dict:
    """The archive's closed IDs, entry count and sweep dates (newest first).

    The page does not carry the archived entries themselves (about 160 KB
    at the first sweep); the Done tab names the file and the count.
    """
    if not path.exists():
        return {"path": path.name, "entries": 0, "sweeps": [], "ids": []}
    text = path.read_text(encoding="utf-8")
    return {"path": path.name,
            "entries": len(re.findall(r"^- ", text, re.M)),
            "sweeps": re.findall(r"^## Swept (\S+)", text, re.M),
            "ids": sorted(set(re.findall(
                r"^- \*\*([A-Z]{2,3}-\d{2,3})\b", text, re.M)))}


def load_context() -> dict:
    ctx = {}
    for p in sorted((HERE / "context").glob("*.json")):
        part = json.loads(p.read_text(encoding="utf-8"))
        overlap = set(ctx) & set(part)
        if overlap:
            raise SystemExit(f"{p.name}: IDs already carded elsewhere: {sorted(overlap)}")
        ctx.update(part)
    return ctx


def main() -> int:
    data = parse_backlog(BACKLOG.read_text(encoding="utf-8"))
    ids = {i["id"] for i in data["items"]}
    ctx = load_context()
    archive = archive_summary()
    closed = {re.match(r"\*\*([A-Z]{2,3}-\d{2,3})\b", d) for d in data["done"]}
    closed = {m.group(1) for m in closed if m} | set(archive.pop("ids"))
    unknown = sorted(set(ctx) - ids)
    owner_ids = {i["id"] for i in data["items"] if i["owner"]}
    missing = sorted(owner_ids - set(ctx))
    if set(unknown) & closed:
        print("note: cards for closed IDs (Done or the archive): "
              f"{sorted(set(unknown) & closed)}")
    if set(unknown) - closed:
        print("note: cards for IDs found nowhere in BACKLOG.md or "
              f"{archive['path']}: {sorted(set(unknown) - closed)}")
    if missing:
        print(f"note: owner items without a card yet: {missing}")
    data["cards"] = {k: v for k, v in ctx.items() if k in ids}
    data["archive"] = archive
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "log", "-1",
                                 "--format=%h", "--", "BACKLOG.md"],
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    data["header"]["built"] = date.today().isoformat()
    data["header"]["backlog_commit"] = commit
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        "/*__BACKLOG_DATA__*/null", payload)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT} ({len(html):,} bytes): {len(data['items'])} items, "
          f"{len(owner_ids)} owner items, {len(data['cards'])} cards")
    return 0


if __name__ == "__main__":
    sys.exit(main())
