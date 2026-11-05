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

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs" / "user_manual"))

from funhouse_agent import dispatch  # noqa: E402
from funhouse_agent.adapters import MODULE_REGISTRY  # noqa: E402
import build_manual  # noqa: E402

html = (ROOT / "docs" / "user_manual" / "manual.html").read_text(encoding="utf-8")
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

print(f"version {version}; {len(MODULE_REGISTRY)} modules; {n_meth} analysis methods; "
      f"{n_par} parameters; {n_def} registered defaults")
for p in problems:
    print("MISS", p)
print("OK" if not problems else f"{len(problems)} misses")
sys.exit(1 if problems else 0)
