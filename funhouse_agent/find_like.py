"""``find_like`` for the agent: find every copy of a tag, verify each one.

The reviewer's question on 2026-09-25 was "where are the GCE penetrations?"
on an 85-sheet set whose 0.06 in lettering is drawn as lines. Whole-sheet
vision read "GCE" as "QCE" and found none of 34. This tool answers it the
robust way (owner: "robust first"):

1. **Search** — planlens ``Document.find_like`` image-matches ONE example of
   the mark (a box the agent zoomed to and confirmed; a legend row is fine)
   on every page, at any scale and rotation, and labels each hit from the
   drawing's own geometry: *callout* (a leader is drawn from it, with where it
   points), *unanchored* (on the plan, no leader), *legend* (a legend or
   schedule row, or the same place on most sheets).
2. **Verify** — matching cannot tell GCE from GCG, so every candidate is cut
   out, turned upright, enlarged (lettering ~34 px) and numbered on contact
   sheets, and the vision model READS each number: the characters as drawn,
   and whether a leader is drawn from it. Characters it cannot be sure of come
   back bracketed ([G/Q]CE) and the candidate is reported as uncertain rather
   than counted either way.
3. **Answer** — the confirmed instances by page, callouts with the point each
   leader reaches, the legend entries counted apart, what was rejected and
   why, and the contact sheets saved to the conversation folder so the
   reviewer can check every call the tool made.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence

#: Candidates per contact sheet. 20 cells of 460 x 220 px: the lettering stays
#: above 20 px even after a tile model shrinks the sheet to 768 px.
PER_SHEET = 20

#: Parallel vision calls reading contact sheets.
VERIFY_WORKERS = 4

#: Result size the tool keeps itself under (the reference cap is 16,000).
RESULT_BUDGET_CHARS = 14000

VERIFY_PROMPT = (
    "Each numbered cell (#N) shows ONE candidate mark cut from an engineering "
    "drawing, enlarged and turned upright; the lines around it are its "
    "surroundings on the sheet. For EVERY number on this sheet reply with "
    "exactly one line:\n"
    "#N | <the characters exactly as drawn> | <callout|legend|other>\n"
    "callout = a leader line or arrow is drawn from the mark; legend = the "
    "mark sits in a table, legend or schedule row; other = anything else. "
    "Read character by character. Where a character could be another "
    "(G/Q/O/C/D, E/F, B/8, S/5, I/1/L, Z/2), write the alternatives in "
    "brackets, e.g. [G/Q]CE — never guess. If a cell shows no lettering, "
    "write '-' for the characters. Output only those lines.")

_LINE = re.compile(r"^\s*#?\s*(\d+)\s*[|:]\s*([^|]*?)\s*(?:\|\s*([A-Za-z]+))?\s*$")


def _norm(s: str) -> str:
    return re.sub(r"[\s\-_.]", "", (s or "").upper())


def parse_readings(text: str) -> Dict[int, Dict[str, str]]:
    """``{number: {"read", "kind"}}`` from the model's ``#N | text | kind``
    lines; anything else in the reply is ignored."""
    out: Dict[int, Dict[str, str]] = {}
    for line in (text or "").splitlines():
        m = _LINE.match(line.strip().strip("`"))
        if m:
            out[int(m.group(1))] = {"read": m.group(2).strip(),
                                    "kind": (m.group(3) or "").lower()}
    return out


def _verdict(read: str, target: Optional[str]) -> str:
    if not read or read.strip() == "-":
        return "no_lettering"
    if "[" in read or "?" in read:
        # Every way the brackets resolve: if one of them is the target, the
        # candidate is uncertain; if none can be, it is rejected outright.
        if target:
            alts = _expand(read)
            if _norm(target) in {_norm(a) for a in alts}:
                return "uncertain"
            return "rejected"
        return "uncertain"
    if target is None:
        return "read"
    return "confirmed" if _norm(read) == _norm(target) else "rejected"


def _expand(read: str, limit: int = 64) -> List[str]:
    parts = re.split(r"(\[[^\]]*\])", read)
    outs = [""]
    for p in parts:
        if p.startswith("[") and p.endswith("]"):
            opts = [o for o in re.split(r"[/|,]", p[1:-1]) if o] or ["?"]
            outs = [o + x for o in outs for x in opts][:limit]
        else:
            outs = [o + p for o in outs]
    return outs


def find_like(pdf, page: int, bbox: Sequence[float], engine, *,
              text: Optional[str] = None, pages=None,
              include_legend: bool = False,
              threshold: Optional[float] = None,
              save_dir: Optional[str] = None,
              save_prefix: str = "find_like") -> Dict[str, Any]:
    """Search, verify and report — see the module docstring.

    ``pdf`` is PDF bytes or a path; ``bbox`` the example's box in PDF points
    (displayed frame); ``text`` the characters the mark is expected to read
    (e.g. ``"GCE"``) — with it, candidates are confirmed or rejected; without
    it, they are grouped by what they read. ``engine`` reads the contact
    sheets (``analyze_image``); ``None`` skips verification and returns the
    unverified candidates.
    """
    from planlens.document import Document
    from funhouse_agent import vision_view

    doc = (Document(content=pdf) if isinstance(pdf, (bytes, bytearray))
           else Document(filepath=str(pdf)))
    try:
        kw = {"threshold": threshold} if threshold else {}
        res = doc.find_like(int(page), bbox, pages, **kw)
        hits = res["hits"]
        cands = [h for h in hits if include_legend or h.context != "legend"]
        legend = [h for h in hits if h.context == "legend"]
        sheets = doc.like_sheets(cands, per_sheet=PER_SHEET) if cands else []
    finally:
        doc.close()

    saved: List[str] = []
    if save_dir and sheets:
        os.makedirs(save_dir, exist_ok=True)
        tag = re.sub(r"[^A-Za-z0-9]+", "_", text or "mark").strip("_") or "mark"
        for n, (png, _ids) in enumerate(sheets, start=1):
            path = os.path.join(save_dir, f"{save_prefix}_{tag}_sheet{n}.png")
            with open(path, "wb") as fh:
                fh.write(png)
            saved.append(path)

    readings: Dict[int, Dict[str, str]] = {}
    errors: List[str] = []
    if engine is not None and sheets:
        prompt = VERIFY_PROMPT + (
            f"\n\n(The mark being looked for reads \"{text}\"; report what "
            f"each cell ACTUALLY shows, whether or not it matches.)"
            if text else "")

        def read(sheet):
            png, ids = sheet
            try:
                return ids, engine.analyze_image(png, prompt), None
            except Exception as exc:              # a sheet, not the search
                return ids, "", f"{type(exc).__name__}: {exc}"

        with ThreadPoolExecutor(max_workers=VERIFY_WORKERS) as ex:
            for ids, reply, err in ex.map(read, sheets):
                if err:
                    errors.append(f"sheet #{ids[0]}-{ids[-1]}: {err}")
                got = parse_readings(reply)
                for i in ids:
                    if i in got:
                        readings[i] = got[i]

    rows = []
    for i, h in enumerate(cands, start=1):
        r = readings.get(i, {})
        verdict = (_verdict(r.get("read", ""), text) if r
                   else ("unverified" if engine is None or not sheets
                         else "unread"))
        rows.append({"id": i, "page": h.page,
                     "bbox": [round(v, 1) for v in h.bbox],
                     "context": h.context,
                     "seen_as": r.get("kind") or None,
                     "read": r.get("read") or None,
                     "verdict": verdict,
                     "score": round(h.score, 2),
                     **({"points_to": [round(v, 1) for v in h.points_to]}
                        if h.points_to else {})})

    def pick(v):
        return [r for r in rows if r["verdict"] == v]

    confirmed = pick("confirmed") if text else pick("read")
    instances = [r for r in confirmed
                 if r["context"] != "legend" and r.get("seen_as") != "legend"]
    by_page: Dict[int, int] = {}
    for r in instances:
        by_page[r["page"]] = by_page.get(r["page"], 0) + 1
    rejected = pick("rejected")
    rejected_as: Dict[str, int] = {}
    for r in rejected:
        rejected_as[r["read"] or "?"] = rejected_as.get(r["read"] or "?", 0) + 1
    out: Dict[str, Any] = {
        "target": text, "example": res["example"],
        "pages_searched": len(res["pages"]),
        "instances": len(instances),
        "instances_by_page": {str(k): v for k, v in sorted(by_page.items())},
        "callouts": sum(1 for r in instances if r["context"] == "callout"
                        or r.get("seen_as") == "callout"),
        "legend_entries": len(legend) + sum(
            1 for r in confirmed if r["context"] == "legend"
            or r.get("seen_as") == "legend"),
        "legend_pages": sorted({h.page for h in legend}),
        "uncertain": [{k: r[k] for k in ("id", "page", "bbox", "read")}
                      for r in pick("uncertain") + pick("unread")],
        "rejected_as": rejected_as,
        "contact_sheets": saved,
        "found": [{k: v for k, v in r.items() if k not in ("verdict", "read")}
                  for r in instances],
    }
    if engine is None:
        out["note"] = ("UNVERIFIED: no vision engine — these are image-match "
                       "candidates, look-alikes included.")
        out["unverified"] = [
            {k: r[k] for k in ("id", "page", "bbox", "context", "score")}
            for r in pick("unverified")]
    if errors:
        out["verify_errors"] = errors
    if res.get("warnings"):
        out["warnings"] = res["warnings"]
    _fit(out)
    return out


def _fit(out: Dict[str, Any]) -> None:
    """Keep the JSON under :data:`RESULT_BUDGET_CHARS` by shortening the
    per-instance list (the counts by page always stay)."""
    found = out["found"]
    while found and len(json.dumps(out)) > RESULT_BUDGET_CHARS:
        keep = max(1, int(len(found) * 0.8))
        if keep == len(found):
            keep -= 1
        out["found_truncated"] = (f"{len(out['found']) and keep} of "
                                  f"{out['instances']} instances listed — "
                                  f"narrow `pages` to see the rest")
        found = found[:keep]
        out["found"] = found
        if not found:
            break
    for key in ("unverified", "uncertain"):
        if len(json.dumps(out)) > RESULT_BUDGET_CHARS and out.get(key):
            out[key + "_total"] = len(out[key])
            out[key] = out[key][:20]


__all__ = ["find_like", "parse_readings", "VERIFY_PROMPT", "PER_SHEET"]
