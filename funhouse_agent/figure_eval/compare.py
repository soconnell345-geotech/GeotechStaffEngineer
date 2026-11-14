"""OFF against ON, request by request and item by item (GM-314).

The baseline is plot code OFF; the other arm is ON. For one item of one
request:

* both marked the same (yes and yes, no and no, na and na): ``same``;
* ON ``yes`` where OFF ``no``: ``better``; ON ``no`` where OFF ``yes``:
  ``worse``;
* one arm ``na`` and the other ``yes`` or ``no`` (the two figures are not
  the same kind of figure for that item): ``one_na``, counted apart, never
  folded into better or worse.

A run that drew no figure (or failed) has every item ``no``, so it loses to
any figure that passes an item. A request is ``better`` when ON has more
better items than worse ones, ``worse`` when fewer, otherwise ``same``. No
single blended score: the counts are the result, ties shown as plainly as
gains and losses.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

OUTCOMES = ("better", "same", "worse", "one_na")


def item_outcome(base: Optional[str], other: Optional[str]) -> str:
    """One item: ``base`` is the OFF arm's mark, ``other`` the ON arm's."""
    base = base or "na"
    other = other or "na"
    if base == other:
        return "same"
    if "na" in (base, other):
        return "one_na"
    return "better" if other == "yes" else "worse"


def _empty() -> Dict[str, List[str]]:
    return {o: [] for o in OUTCOMES}


def compare(requests: Sequence[Dict[str, Any]],
            checks: Dict[str, Dict[str, Optional[Dict[str, Any]]]],
            base: str = "off", other: str = "on") -> Dict[str, Any]:
    """``checks[request id][arm]`` is a check (``checklist.check_run``) or
    ``None`` (not checked yet). A request is compared only when both arms
    are checked; the others are listed under ``unchecked``."""
    per_request: Dict[str, Dict[str, Any]] = {}
    per_item: Dict[str, Dict[str, List[str]]] = {}
    overall: Dict[str, List[str]] = {"better": [], "same": [], "worse": []}
    items_total = {o: 0 for o in OUTCOMES}
    marks_total: Dict[str, Dict[str, int]] = {
        base: {"yes": 0, "no": 0, "na": 0},
        other: {"yes": 0, "no": 0, "na": 0}}
    unchecked: List[str] = []
    for req in requests:
        rid = req["id"]
        a = (checks.get(rid) or {}).get(base)
        b = (checks.get(rid) or {}).get(other)
        if not a or not b:
            unchecked.append(rid)
            continue
        counts = {o: 0 for o in OUTCOMES}
        items = {}
        for iid in a["marks"]:
            ma = a["marks"][iid]["mark"]
            mb = (b["marks"].get(iid) or {}).get("mark")
            out = item_outcome(ma, mb)
            counts[out] += 1
            items_total[out] += 1
            items[iid] = {"outcome": out, base: ma, other: mb or "na"}
            key = iid if not iid.startswith("x") else "request's own items"
            per_item.setdefault(key, _empty())[out].append(rid)
            marks_total[base][ma] += 1
            marks_total[other][mb or "na"] += 1
        verdict = ("better" if counts["better"] > counts["worse"] else
                   "worse" if counts["worse"] > counts["better"] else
                   "same")
        overall[verdict].append(rid)
        per_request[rid] = {
            "verdict": verdict, "counts": counts, "items": items,
            "no_figure": {base: bool(a.get("no_figure")),
                          other: bool(b.get("no_figure"))}}
    return {"base": base, "other": other, "per_request": per_request,
            "per_item": per_item, "requests": overall,
            "items": items_total, "marks": marks_total,
            "unchecked": unchecked}
