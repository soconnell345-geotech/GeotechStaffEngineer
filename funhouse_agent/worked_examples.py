"""Worked-examples corpus — validated, published calculations as agent exemplars.

The registry (``worked_examples.json``, packaged alongside this module) holds
curated worked examples drawn from REAL published design reports and
verification manuals (FHWA GEC design examples, Caltrans Trenching & Shoring,
AASHTO 1993, UFC 3-250-01, Slide2/FLAC verification problems). Every entry's
``dispatch_calls`` run clean through ``funhouse_agent.dispatch.call_agent`` and
its ``computed_result`` was produced by actually running them — the corpus is
mechanically verifiable (see ``verify_example`` and the offline test).

Purpose: when the calc agent works a nontrivial problem or builds a report, a
matching exemplar shows (1) how a real problem of this type is framed, (2) the
method calls and parameters that solve it, (3) the published answer it should
be able to reproduce, and (4) what a professional calc report presents
(``report_notes``). This is few-shot grounding with provenance, not extra
methods.

Owner-extensible: entries are plain JSON; the schema is documented by
``REQUIRED_KEYS`` and enforced by ``validate_entry``. Future direction (see
module_work/FUTURE_IDEAS.md): a second, user-supplied corpus harvested from
the owner's own project reports.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import List, Optional

REGISTRY_FILENAME = "worked_examples.json"

#: Every entry must carry these keys (validate_entry enforces).
REQUIRED_KEYS = (
    "id", "title", "domain", "keywords", "source", "problem",
    "dispatch_calls", "published_result", "computed_result",
    "report_notes", "provenance",
)


def registry_path() -> str:
    """Absolute path of the packaged registry JSON (source tree or wheel)."""
    try:
        from importlib.resources import files
        p = str(files("funhouse_agent") / REGISTRY_FILENAME)
        if os.path.isfile(p):
            return p
    except Exception:
        pass
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        REGISTRY_FILENAME)


@lru_cache(maxsize=1)
def load_examples() -> tuple:
    """The full corpus as a tuple of entry dicts (cached; empty if missing)."""
    path = registry_path()
    if not os.path.isfile(path):
        return ()
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return tuple(data)


def validate_entry(entry: dict) -> List[str]:
    """Schema problems for one entry ([] = valid)."""
    problems = []
    for k in REQUIRED_KEYS:
        if k not in entry:
            problems.append(f"missing key '{k}'")
    for call in entry.get("dispatch_calls", []):
        for k in ("module", "method", "params"):
            if k not in call:
                problems.append(f"dispatch_call missing '{k}'")
    if not isinstance(entry.get("keywords", []), list):
        problems.append("'keywords' must be a list")
    return problems


_WORD = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> set:
    return set(_WORD.findall(str(text).lower()))


def search_examples(query: str, domain: Optional[str] = None,
                    limit: int = 3) -> List[dict]:
    """Rank corpus entries against a free-text query (simple token overlap,
    keywords and title weighted over body text). Returns up to ``limit`` full
    entries, best first; [] when nothing scores."""
    q = _tokens(query)
    if not q:
        return []
    scored = []
    for e in load_examples():
        if domain and e.get("domain") != domain:
            continue
        kw = _tokens(" ".join(e.get("keywords", [])))
        title = _tokens(e.get("title", ""))
        body = _tokens(e.get("problem", "")) | _tokens(e.get("source", ""))
        dom = _tokens(e.get("domain", ""))
        score = (3.0 * len(q & kw) + 2.0 * len(q & title)
                 + 1.0 * len(q & body) + 2.0 * len(q & dom))
        if score > 0:
            scored.append((score, e))
    scored.sort(key=lambda t: -t[0])
    return [e for _, e in scored[:int(max(1, limit))]]


def get_example(example_id: str) -> Optional[dict]:
    """One entry by exact id (case-insensitive), else None."""
    want = str(example_id).strip().lower()
    for e in load_examples():
        if str(e.get("id", "")).lower() == want:
            return e
    return None


def summarize(entry: dict) -> dict:
    """Compact listing form (no params/body) for search results the agent can
    scan cheaply before fetching the full entry."""
    return {
        "id": entry.get("id"),
        "title": entry.get("title"),
        "domain": entry.get("domain"),
        "source": entry.get("source"),
        "published_result": entry.get("published_result"),
    }


def verify_example(entry: dict) -> List[str]:
    """Run the entry's dispatch calls; return error strings ([] = clean).
    Used by the offline corpus test so every packaged example stays runnable."""
    from funhouse_agent.dispatch import call_agent
    errors = []
    for call in entry.get("dispatch_calls", []):
        try:
            result = call_agent(call["module"], call["method"],
                                call.get("params", {}))
        except Exception as exc:
            errors.append(f"{call.get('module')}.{call.get('method')}: "
                          f"{type(exc).__name__}: {exc}")
            continue
        if isinstance(result, dict) and result.get("error"):
            errors.append(f"{call.get('module')}.{call.get('method')}: "
                          f"{str(result['error'])[:200]}")
    return errors
