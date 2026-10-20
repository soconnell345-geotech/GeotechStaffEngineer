"""The records a stopped helper held, found by their shape (BACKLOG RI-31).

Field session 2026-10-14, turn 3: helpers that extracted boring logs typed
their records into their replies or into calls that were refused, and the
primary retyped 29,000 characters of them into ``write_diggs``; a blow
count, a bottom depth, a printed 60/15" and one exploration's coordinates
went wrong on the way. A helper that saves its records with ``save_file``
hands them over by name (``keep_work.saved_note``). One that STOPS first
(an error, its model-call budget, a short answer after a lot of reading)
left them only in its kept ``.md``, so the primary would retype them again.

This module finds those records in what the helper wrote: its answer (or,
for a helper that raised, its last words) and the arguments of each of its
calls. Only records in the shape ``write_diggs`` takes are taken -- an
exploration (``investigation_id`` with layers, samples, SPTs, water levels,
a total depth or a cone/probe/pit record) or a laboratory test (a lab
``kind`` with its result or the hole and sample it came from), the models
of :mod:`report_ingest.model` -- never any other JSON. The source holding
the most records is the one kept; records a call of the helper already
delivered (a ``save_file`` or a ``write_diggs`` that succeeded) are left
out, so the same exploration is never handed over twice.

Nothing here changes a value: a record is kept exactly as the helper wrote
it, and ``write_diggs`` checks it when the primary passes the file.
"""

from __future__ import annotations

import functools
import json
import re
from typing import Any, Dict, List, Optional, Tuple

#: An exploration's own content: at least one of these, of the right type,
#: makes a dict with an ``investigation_id`` an exploration record.
_INVESTIGATION_LISTS = ("layers", "samples", "spt", "water")
_INVESTIGATION_DICTS = ("total_depth", "cpt", "dcp", "pit")

#: Nested records are not walked deeper than this.
_MAX_DEPTH = 12

#: Text longer than this is not searched for JSON (a helper's answer or a
#: call's argument; the field session's largest was 53,000 characters).
_MAX_TEXT_CHARS = 2_000_000


@functools.lru_cache(maxsize=1)
def _fields() -> Tuple[frozenset, frozenset, frozenset, frozenset]:
    """(exploration fields, lab-test fields, exploration kinds, lab kinds),
    read off the record models; empty where report_ingest is missing."""
    try:
        from typing import get_args

        from report_ingest.model import (Investigation, InvestigationKind,
                                         LabKind, LabTest)
    except Exception:  # noqa: BLE001 - no record models, no records
        return frozenset(), frozenset(), frozenset(), frozenset()
    return (frozenset(Investigation.model_fields),
            frozenset(LabTest.model_fields),
            frozenset(get_args(InvestigationKind)),
            frozenset(get_args(LabKind)))


def _mostly_known(d: dict, known: frozenset) -> bool:
    """At least as many of ``d``'s keys are the model's as are not (a
    misspelt key or two is still the record it plainly is; write_diggs
    names it)."""
    keys = set(map(str, d))
    return len(keys & known) >= len(keys - known)


def is_investigation(d: Any) -> bool:
    """Is ``d`` an exploration record as write_diggs takes one?"""
    inv_fields, _, inv_kinds, _ = _fields()
    if not inv_fields or not isinstance(d, dict):
        return False
    iid = d.get("investigation_id")
    if not isinstance(iid, str) or not iid.strip() or "result" in d:
        return False
    kind = d.get("kind")
    if kind is not None and kind not in inv_kinds:
        return False
    content = any(isinstance(d.get(k), list) and d.get(k)
                  for k in _INVESTIGATION_LISTS) or any(
        isinstance(d.get(k), dict) and d.get(k)
        for k in _INVESTIGATION_DICTS)
    return content and _mostly_known(d, inv_fields)


def is_lab_test(d: Any) -> bool:
    """Is ``d`` a laboratory-test record as write_diggs takes one?"""
    _, lab_fields, _, lab_kinds = _fields()
    if not lab_fields or not isinstance(d, dict):
        return False
    if d.get("kind") not in lab_kinds:
        return False
    if any(k in d for k in _INVESTIGATION_LISTS + _INVESTIGATION_DICTS):
        return False
    result = d.get("result")
    if isinstance(result, dict) and result.get("kind"):
        linked = True
    else:
        linked = d.get("kind") != "other" and any(
            isinstance(d.get(k), str) and d.get(k).strip()
            for k in ("investigation_id", "sample_id"))
    return linked and _mostly_known(d, lab_fields)


_OPENS = re.compile(r"[\[{]")


def json_values(text: str) -> list:
    """Every JSON object or array written in ``text``, in order: fenced or
    not, each decoded where it starts. A value that does not decode is
    skipped (its inner values are tried in turn)."""
    if not isinstance(text, str) or len(text) > _MAX_TEXT_CHARS:
        return []
    out: list = []
    decoder = json.JSONDecoder()
    at = 0
    while True:
        m = _OPENS.search(text, at)
        if m is None:
            return out
        try:
            value, end = decoder.raw_decode(text, m.start())
        except ValueError:
            at = m.start() + 1
            continue
        if isinstance(value, (dict, list)):
            out.append(value)
        at = max(end, m.start() + 1)


def records_in(value: Any) -> Tuple[List[dict], List[dict]]:
    """``(investigations, lab_tests)``: the records ``value`` holds, in
    order, each taken whole and once. JSON written in a string (an answer's
    prose, a fenced block, a call's ``parameters`` sent as text) is read; a
    record's own parts are not searched again."""
    inv: List[dict] = []
    lab: List[dict] = []
    seen: set = set()

    def take(target: list, d: dict) -> None:
        key = _key(d)
        if key not in seen:
            seen.add(key)
            target.append(d)

    def walk(v: Any, depth: int) -> None:
        if depth > _MAX_DEPTH:
            return
        if isinstance(v, dict):
            if is_investigation(v):
                take(inv, v)
                return
            if is_lab_test(v):
                take(lab, v)
                return
            for item in v.values():
                walk(item, depth + 1)
        elif isinstance(v, (list, tuple)):
            for item in v:
                walk(item, depth + 1)
        elif isinstance(v, str) and ("{" in v or "[" in v):
            for item in json_values(v):
                walk(item, depth + 1)

    walk(value, 0)
    return inv, lab


def _key(d: dict) -> str:
    try:
        return json.dumps(d, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return repr(d)


def _delivered(rec) -> set:
    """The ids of the helper's calls that wrote what they were given to the
    real disk: a result without an error that names what it ``saved`` or
    the ``output_path`` it wrote (save_file, write_diggs)."""
    out: set = set()
    for s in rec.steps:
        if s.get("kind") != "result" or s.get("status") == "error":
            continue
        text = s.get("text") or ""
        if '"saved"' not in text and '"output_path"' not in text:
            continue
        try:
            data = json.loads(text)
        except ValueError:
            continue
        if isinstance(data, dict) and not data.get("error") and (
                data.get("saved") or data.get("output_path")):
            out.add(s.get("id"))
    return out


def held_records(rec, answer: Optional[str] = None) -> Optional[Dict]:
    """The records a helper held and did not deliver: ``{"investigations",
    "lab_tests", "source"}`` from the one source that holds the most of them
    -- its answer (its last words when it gave none) or one of its calls --
    or ``None`` when it held none."""
    sources: list = []
    text = answer if isinstance(answer, str) and answer.strip() else None
    if text is not None:
        sources.append(("its answer", text))
    else:
        last = rec.last_words()
        if last:
            sources.append(("its last words", last))
    delivered = _delivered(rec)
    done_ids: set = set()
    done_lab: set = set()
    calls: list = []
    for s in rec.steps:
        if s.get("kind") != "model":
            continue
        for c in s.get("calls") or []:
            if c.get("id") and c.get("id") in delivered:
                inv, lab = records_in(c.get("args"))
                done_ids.update(str(i.get("investigation_id")).strip()
                                for i in inv)
                done_lab.update(_key(t) for t in lab)
                continue
            size = len(_key(c.get("args")))
            calls.append((size, c))
    calls.sort(key=lambda sc: -sc[0])
    for size, c in calls:
        sources.append((f"its {c.get('name') or 'tool'} call ({size:,} "
                        "characters of arguments)", c.get("args")))
    best = None
    for label, value in sources:
        inv, lab = records_in(value)
        inv = [i for i in inv
               if str(i.get("investigation_id")).strip() not in done_ids]
        lab = [t for t in lab if _key(t) not in done_lab]
        if inv or lab:
            if best is None or len(inv) + len(lab) > \
                    len(best["investigations"]) + len(best["lab_tests"]):
                best = {"investigations": inv, "lab_tests": lab,
                        "source": label}
    return best


def file_content(held: Dict) -> str:
    """The records file: ``{"investigations": [...]}``, ``{"lab_tests":
    [...]}`` or both, as write_diggs reads a records file."""
    data: dict = {}
    if held.get("investigations"):
        data["investigations"] = held["investigations"]
    if held.get("lab_tests"):
        data["lab_tests"] = held["lab_tests"]
    return json.dumps(data, ensure_ascii=False, indent=1, default=str)


def describe(held: Dict) -> str:
    """'3 explorations (B-1, B-2, B-3) and 2 lab tests'."""
    parts = []
    inv = held.get("investigations") or []
    lab = held.get("lab_tests") or []
    if inv:
        ids = [str(i.get("investigation_id")) for i in inv]
        shown = ", ".join(ids[:12]) + (f" and {len(ids) - 12} more"
                                       if len(ids) > 12 else "")
        parts.append(f"{len(inv)} exploration{'s' if len(inv) != 1 else ''} "
                     f"({shown})")
    if lab:
        parts.append(f"{len(lab)} lab test{'s' if len(lab) != 1 else ''}")
    return " and ".join(parts)


def records_note(name: str, held: Dict) -> str:
    """What the primary is told about a records file saved for a stopped
    helper: its name, what it holds and how to pass it to write_diggs."""
    inv = bool(held.get("investigations"))
    lab = bool(held.get("lab_tests"))
    if inv and lab:
        how = ("as investigations_file AND as lab_tests_file (the same name "
               "for both)")
    elif inv:
        how = "as investigations_file"
    else:
        how = "as lab_tests_file"
    return (f"[Records this helper held, from {held.get('source')}, are saved "
            f"as '{name}' in the working folder, exactly as it wrote them: "
            f"{describe(held)}. Pass that file to write_diggs {how} -- never "
            "retype them; write_diggs checks every record and names any "
            "problem by file and field.]")


__all__ = ["is_investigation", "is_lab_test", "json_values", "records_in",
           "held_records", "file_content", "describe", "records_note"]
