#!/usr/bin/env python
"""DOC-10: check docs/user_manual/manual.html against the shipped registry.

Reports (exit code 1 on any miss):
  * the manual's version against pyproject.toml;
  * every module in the dispatch catalog named in the manual;
  * every non-alias method of every ANALYSIS module named in the manual
    (reference modules are listed by method count, by design);
  * every parameter of those methods named in the manual;
  * every optional parameter with a registered default shown as
    ``default: <value>`` (searched in the whole page, not per table).

Usage (repo root):  python module_work/doc10/check_manual.py
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs" / "user_manual"))

from funhouse_agent import dispatch  # noqa: E402
from funhouse_agent.adapters import MODULE_REGISTRY  # noqa: E402
import build_manual  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--rebuild", action="store_true",
                help="rebuild the manual in memory and diff it against the committed files")
ap.add_argument("--write", action="store_true",
                help="with --rebuild, write the rebuilt manual.html and toc.json in place")
args = ap.parse_args()

MANUAL = ROOT / "docs" / "user_manual"
rebuild_problems = []
if args.rebuild:
    new_html, toc_entries = build_manual.build()
    new_toc = json.dumps(toc_entries, ensure_ascii=False, indent=2)
    for name, new in (("manual.html", new_html), ("toc.json", new_toc)):
        path = MANUAL / name
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        if old == new:
            print(f"rebuild: {name} unchanged")
            continue
        diff = list(difflib.unified_diff(old.splitlines(), new.splitlines(),
                                         f"committed/{name}", f"rebuilt/{name}",
                                         n=0, lineterm=""))
        changed = sum(1 for d in diff if d[:1] in "+-" and d[:3] not in ("+++", "---"))
        rebuild_problems.append(f"rebuilt {name} differs from the committed file "
                                f"({changed} lines changed)")
        print(f"rebuild: {name} differs ({changed} lines changed); first lines:")
        for d in diff[:80]:
            print("  " + (d if len(d) <= 300 else d[:300] + " ..."))
        if args.write:
            path.write_text(new, encoding="utf-8")
            print(f"rebuild: wrote {path}")
    if args.write:
        rebuild_problems = []   # the files on disk now match the rebuild

html = (MANUAL / "manual.html").read_text(encoding="utf-8")
version = re.search(r'^version\s*=\s*"([^"]+)"',
                    (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M).group(1)
problems = []
if f"Version {version}<" not in html:
    problems.append(f"cover does not say Version {version}")

n_meth = n_par = n_def = 0
for name in sorted(MODULE_REGISTRY):
    if f"<code>{name}</code>" not in html:
        problems.append(f"module {name} not in the manual")
    if name in dispatch.REFERENCE_MODULES:
        continue
    mod = dispatch._load_adapter(name)
    for meth, info in getattr(mod, "METHOD_INFO", {}).items():
        if info.get("alias_of"):
            continue
        n_meth += 1
        if f"<code>{meth}</code>" not in html:
            problems.append(f"method {name}.{meth} not in the manual")
        for pn, pi in (info.get("parameters") or {}).items():
            n_par += 1
            if f"<code>{pn}</code>" not in html:
                problems.append(f"parameter {name}.{meth}.{pn} not in the manual")
            if isinstance(pi, dict) and pi.get("default") not in (None, "") \
                    and not pi.get("required"):
                n_def += 1
                shown = "default: <code>" + build_manual.esc(
                    build_manual._default_text(pi["default"])) + "</code>"
                if shown not in html:
                    problems.append(f"default of {name}.{meth}.{pn} not shown")

problems = rebuild_problems + problems
print(f"version {version}; {len(MODULE_REGISTRY)} modules; {n_meth} analysis methods; "
      f"{n_par} parameters; {n_def} registered defaults")
for p in problems:
    print("MISS", p)
print("OK" if not problems else f"{len(problems)} misses")
sys.exit(1 if problems else 0)
