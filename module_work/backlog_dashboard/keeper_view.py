"""Print one keeper's own open items from BACKLOG.md (run from the repo root).

    python module_work/backlog_dashboard/keeper_view.py maintainer
    python module_work/backlog_dashboard/keeper_view.py developer
    python module_work/backlog_dashboard/keeper_view.py owner
    python module_work/backlog_dashboard/keeper_view.py --check

BACKLOG.md stays the one list. This prints a view of it and writes nothing:
the items one keeper works on, by tier, with the owner's decisions those
items cite. The item text itself is in BACKLOG.md.

Who keeps an item is read from BACKLOG.md section 0 (a):
- the line "By section" gives each section's keeper;
- four lines list the exceptions by id: kept by the maintainer, kept by the
  developer's sessions, shared, the owner's own.
A shared item is printed for both the maintainer and the developer, marked
"shared". ``--check`` fails when a listed id is not open, when an id is in
two lists, or when an open item has no keeper.
"""
import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
BACKLOG = ROOT / "BACKLOG.md"
DONE = "## Done since the last sweep"
TIERS = ("Now", "Next", "Later", "Parked")
ITEM = re.compile(r"^- \*\*([A-Z]{2,3}-\d{2,3})\*\* \[([A-Za-z]+)\]((?: \[[a-z]+\])*) \*\*(.+?)\*\*")
ANY_ID = re.compile(r"\b([A-Z]{2,3}-\d{2,3})\b")
SECTION = re.compile(r"^## (\d+)\. (.+)$")
# The start of each exceptions line in section 0 (a), and the keeper it gives.
LISTS = (
    ("- **Kept by the maintainer", "maintainer"),
    ("- **Kept by the developer", "developer"),
    ("- **Shared", "shared"),
    ("- **The owner's own", "owner"),
)
# Read from the "By section" line; these are used only if that line is absent.
DEFAULTS = {1: "owner", 2: "maintainer", 3: "maintainer", 4: "maintainer",
            5: "maintainer", 6: "developer", 7: "developer", 8: "developer",
            9: "developer", 10: "developer"}
WORDS = (("maintainer", "maintainer"), ("developer", "developer"), ("owner", "owner"))


def read(backlog=BACKLOG):
    """Open items, section names, each list of exceptions, section defaults."""
    text = pathlib.Path(backlog).read_text(encoding="utf-8").replace("\r\n", "\n")
    open_part = text.split(DONE, 1)[0]
    items, names, lists, defaults = [], {}, {}, dict(DEFAULTS)
    section = None
    for line in open_part.split("\n"):
        m = SECTION.match(line)
        if m:
            section = int(m.group(1))
            names[section] = m.group(2).strip()
            continue
        if section == 0:
            for start, keeper in LISTS:
                if line.startswith(start):
                    lists[keeper] = ANY_ID.findall(line.split(":**", 1)[-1])
            if line.startswith("- **By section:**"):
                defaults = {}
                for clause in line.split(":**", 1)[1].split(";"):
                    keeper = next((k for w, k in WORDS if w in clause), None)
                    if keeper and "by the lists" not in clause:
                        for n in re.findall(r"\b(\d{1,2})\b", clause):
                            defaults[int(n)] = keeper
            continue
        m = ITEM.match(line)
        if m and section is not None:
            items.append({"id": m.group(1), "tier": m.group(2), "title": m.group(4),
                          "section": section, "line": line})
    return items, names, lists, defaults


def keepers(items, lists, defaults):
    """id -> keeper ('maintainer', 'developer', 'shared', 'owner' or None)."""
    listed = {}
    for keeper, ids in lists.items():
        for i in ids:
            listed.setdefault(i, keeper)
    return {it["id"]: listed.get(it["id"], defaults.get(it["section"])) for it in items}


def problems(backlog=BACKLOG):
    """What ``--check`` reports; an empty list when every open item has one keeper."""
    items, _names, lists, defaults = read(backlog)
    open_ids = {it["id"] for it in items}
    out, seen = [], {}
    if len(lists) != len(LISTS):
        out.append("section 0 (a) lacks a list: " +
                   ", ".join(k for _s, k in LISTS if k not in lists))
    for keeper, ids in lists.items():
        for i in ids:
            if i not in open_ids:
                out.append(f"{i} is in the '{keeper}' list but is not an open item")
            if i in seen and seen[i] != keeper:
                out.append(f"{i} is in two lists: '{seen[i]}' and '{keeper}'")
            seen[i] = keeper
    who = keepers(items, lists, defaults)
    out += [f"{i} (section {it['section']}) has no keeper: its section has no default "
            "and no list names it" for it in items for i in [it["id"]] if who[i] is None]
    return out


def view(keeper, backlog=BACKLOG):
    items, names, lists, defaults = read(backlog)
    who = keepers(items, lists, defaults)
    mine = [it for it in items if who[it["id"]] in (keeper, "shared" if keeper != "owner" else keeper)]
    label = {"maintainer": "the maintainer", "developer": "the developer's sessions",
             "owner": "the owner"}[keeper]
    shared = sum(1 for it in mine if who[it["id"]] == "shared")
    out = [f"# Open items kept by {label}: {len(mine)}"
           + (f" ({shared} of them shared)" if shared else ""), "",
           "A view of `BACKLOG.md`, printed by "
           "`module_work/backlog_dashboard/keeper_view.py`. The item text is in "
           "`BACKLOG.md` under the same id; this list holds no work of its own.", ""]
    for tier in TIERS:
        rows = [it for it in mine if it["tier"] == tier]
        if not rows:
            continue
        out += [f"## {tier} ({len(rows)})", ""]
        section = None
        for it in rows:
            if it["section"] != section:
                section = it["section"]
                out.append(f"**Section {section}. {names.get(section, '')}**")
            mark = " (shared)" if who[it["id"]] == "shared" else ""
            out.append(f"- {it['id']}{mark}: {it['title']}")
        out.append("")
    if keeper != "owner":
        own = {it["id"]: it for it in items if it["id"].startswith("OWN-")}
        cited = {}
        for it in mine:
            for i in ANY_ID.findall(it["line"]):
                if i in own and i != it["id"]:
                    cited.setdefault(i, []).append(it["id"])
        if cited:
            out += [f"## The owner's open decisions these items cite ({len(cited)})", ""]
            for i in sorted(cited, key=lambda x: int(x.split("-")[1])):
                by = ", ".join(cited[i][:8]) + (" ..." if len(cited[i]) > 8 else "")
                out.append(f"- {i}: {own[i]['title']} (cited by {by})")
            out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("keeper", nargs="?", choices=("maintainer", "developer", "owner"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--backlog", type=pathlib.Path, default=BACKLOG)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.check:
        bad = problems(args.backlog)
        print("\n".join(bad) if bad else "every open item has one keeper")
        raise SystemExit(1 if bad else 0)
    if not args.keeper:
        ap.error("name a keeper, or use --check")
    print(view(args.keeper, args.backlog))


if __name__ == "__main__":
    main()
