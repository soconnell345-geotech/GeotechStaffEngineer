"""OFF against ON, request by request and item by item (GM-314).

The baseline is plot code OFF; the other arm is ON. For one item of one
request:

* both marked the same (yes and yes, no and no, na and na): ``same``;
* ON ``yes`` where OFF ``no``: ``better``; ON ``no`` where OFF ``yes``:
  ``worse``;
* one arm ``na`` and the other ``yes`` or ``no`` (the two figures are not
  the same kind of figure for that item): ``one_na``, counted apart, never
  folded into better or worse;
* the checker gave no mark in either arm: ``unmarked``, counted apart,
  never ``same``.

A run that finished and drew no figure has every item ``no``, so it loses
to any figure that passes an item; the report names those requests. Three
kinds of request are kept out of better / same / worse, counted apart and
named (:data:`APART`): a run that failed in either arm (an error, a rate
limit, a timeout, a crash), a vision reply that could not be read even
when asked twice, and a request where neither arm drew a figure. A request
is ``better`` when ON has more better items than worse ones, ``worse`` when
fewer, otherwise ``same``. No single blended score: the counts are the
result, ties shown as plainly as gains and losses.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

OUTCOMES = ("better", "same", "worse", "one_na", "unmarked")
MARK_KEYS = ("yes", "no", "na", "unmarked")
#: Requests kept out of better / same / worse, and the check flag of each.
APART = {"run_failed": "run_failed", "not_scored": "not_scored",
         "neither_drew": "no_figure"}


def item_outcome(base: Optional[str], other: Optional[str]) -> str:
    """One item: ``base`` is the OFF arm's mark, ``other`` the ON arm's."""
    base = base or "unmarked"
    other = other or "unmarked"
    if "unmarked" in (base, other):
        return "unmarked"
    if base == other:
        return "same"
    if "na" in (base, other):
        return "one_na"
    return "better" if other == "yes" else "worse"


def _empty() -> Dict[str, List[str]]:
    return {o: [] for o in OUTCOMES}


def _apart(a: Dict[str, Any], b: Dict[str, Any]) -> Optional[str]:
    """Why a request checked in both arms is kept apart, if it is."""
    if a.get("run_failed") or b.get("run_failed"):
        return "run_failed"
    if a.get("not_scored") or b.get("not_scored"):
        return "not_scored"
    if a.get("no_figure") and b.get("no_figure"):
        return "neither_drew"
    return None


def compare(requests: Sequence[Dict[str, Any]],
            checks: Dict[str, Dict[str, Optional[Dict[str, Any]]]],
            base: str = "off", other: str = "on") -> Dict[str, Any]:
    """``checks[request id][arm]`` is a check (``checklist.check_run``) or
    ``None`` (not checked yet). A request is compared only when both arms
    are checked (the others are listed under ``unchecked``) and it is not
    kept apart (``apart[reason][request id]`` names the arms concerned)."""
    per_request: Dict[str, Dict[str, Any]] = {}
    per_item: Dict[str, Dict[str, List[str]]] = {}
    overall: Dict[str, List[str]] = {"better": [], "same": [], "worse": []}
    apart: Dict[str, Dict[str, List[str]]] = {k: {} for k in APART}
    drew_none: Dict[str, str] = {}
    items_total = {o: 0 for o in OUTCOMES}
    marks_total: Dict[str, Dict[str, int]] = {
        base: dict.fromkeys(MARK_KEYS, 0), other: dict.fromkeys(MARK_KEYS, 0)}
    unchecked: List[str] = []
    for req in requests:
        rid = req["id"]
        a = (checks.get(rid) or {}).get(base)
        b = (checks.get(rid) or {}).get(other)
        if not a or not b:
            unchecked.append(rid)
            continue
        why = _apart(a, b)
        if why:
            apart[why][rid] = [arm for arm, c in ((base, a), (other, b))
                               if c.get(APART[why])]
            continue
        counts = {o: 0 for o in OUTCOMES}
        items = {}
        for iid in a["marks"]:
            ma = a["marks"][iid]["mark"]
            mb = (b["marks"].get(iid) or {}).get("mark") or "unmarked"
            out = item_outcome(ma, mb)
            counts[out] += 1
            items_total[out] += 1
            items[iid] = {"outcome": out, base: ma, other: mb}
            key = iid if not iid.startswith("x") else "request's own items"
            per_item.setdefault(key, _empty())[out].append(rid)
            for arm, m in ((base, ma), (other, mb)):
                marks_total[arm][m if m in MARK_KEYS else "unmarked"] += 1
        verdict = ("better" if counts["better"] > counts["worse"] else
                   "worse" if counts["worse"] > counts["better"] else
                   "same")
        overall[verdict].append(rid)
        no_fig = [arm for arm, c in ((base, a), (other, b))
                  if c.get("no_figure")]
        if no_fig:
            drew_none[rid] = no_fig[0]
        per_request[rid] = {
            "verdict": verdict, "counts": counts, "items": items,
            "no_figure": {base: bool(a.get("no_figure")),
                          other: bool(b.get("no_figure"))}}
    return {"base": base, "other": other, "per_request": per_request,
            "per_item": per_item, "requests": overall, "apart": apart,
            "drew_none": drew_none, "items": items_total,
            "marks": marks_total, "unchecked": unchecked}
