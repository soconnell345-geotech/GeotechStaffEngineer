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

An example that matches linework rather than the mark floods a page with
candidates (Foundry brief 4: 367 and 400 on a sheet of seven callouts, 18-20
vision calls to read them). Past :data:`FLOOD_PER_PAGE` on any page nothing
is read, and the result says to box a copy not crossed by linework. A looser
example can stay under that guard on every page and still flood the whole
set (brief 5: 978 candidates, 49 sheets, 795 s), so the whole call also
keeps to one time budget (:data:`BUDGET_ENV`): past it no further sheet is
read and the result says the search was cut short.
"""

from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Dict, List, Optional, Sequence

#: Seconds one ``find_like`` call may spend in all — the search and the
#: reads of its contact sheets (``GEOTECH_FIND_LIKE_BUDGET_S``; ``0`` = no
#: limit). Foundry brief 5 (2026-10-08): a loose example (a callout with its
#: leader) matched 978 places at about 41 a page — under the per-page flood
#: guard — and the tool read 49 sheets in 795 s for 3 true instances. Past
#: the budget no further sheet is sent; the reads already under way finish,
#: and the result says the search was cut short and which pages were not
#: read.
BUDGET_ENV = "GEOTECH_FIND_LIKE_BUDGET_S"
DEFAULT_BUDGET_S = 300.0

#: The clock the budget is kept on (a test replaces it).
_clock = time.monotonic


def budget_s() -> Optional[float]:
    """The whole-search budget in seconds (:data:`BUDGET_ENV`), or ``None``
    for none."""
    raw = (os.environ.get(BUDGET_ENV) or "").strip()
    try:
        limit = float(raw) if raw else DEFAULT_BUDGET_S
    except ValueError:
        limit = DEFAULT_BUDGET_S
    return limit if limit > 0 else None


CUT_SHORT_NOTE = (
    "The search was CUT SHORT: its time budget ({budget:g} s) ran out after "
    "{read} of {total} candidates were read. Every count above covers ONLY "
    "the candidates read; {left} candidate(s) on page(s) {pages} were NOT "
    "read and are counted neither way. To finish, search again with "
    "pages=\"{pages}\". A box tight round the lettering of one copy (no "
    "leader, no linework) matches fewer look-alikes and reads faster.")

#: Candidates per contact sheet. 20 cells of 460 x 220 px: the lettering stays
#: above 20 px even after a tile model shrinks the sheet to 768 px.
PER_SHEET = 20

#: Parallel vision calls reading contact sheets.
VERIFY_WORKERS = 4

#: Result size the tool keeps itself under (the vision cap is 32,000 —
#: ``deep.tools.DEFAULT_VISION_RESULT_CHARS``).
RESULT_BUDGET_CHARS = 30000

#: More candidates than this on ONE page is not a mark's copies but the
#: example matching linework: the read pass is skipped and the agent is told
#: to box a cleaner copy. Foundry brief 4 (2026-10-07): an example whose
#: lettering sat on a heavy grid line gave 367 and 400 candidates on a sheet
#: holding seven callouts and a few dozen look-alikes, and reading them took
#: 18-20 vision calls and 180-289 s; a clean example gave 43. Half planlens'
#: own per-page cap (400), and well past what a sheet holds of one mark.
FLOOD_PER_PAGE = 200

FLOOD_NOTE = (
    "The example matched {n} places on page(s) {pages} — far more than one "
    "mark's copies on a sheet — so it is matching LINEWORK, not the mark: a "
    "grid line, a wall, a leader or a table rule running through the example "
    "box. The candidates were NOT read and this is NOT a count of instances. "
    "Box another copy of the mark, one not crossed by linework (a legend "
    "row is often clean), tight round its lettering, and search again.")

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

    budget = budget_s()
    deadline = None if budget is None else _clock() + budget
    doc = (Document(content=pdf) if isinstance(pdf, (bytes, bytearray))
           else Document(filepath=str(pdf)))
    try:
        kw = {"threshold": threshold} if threshold else {}
        res = doc.find_like(int(page), bbox, pages, **kw)
        hits = res["hits"]
        flooded = flooded_pages(hits)
        if flooded:
            return _flood_result(res, hits, flooded, text)
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
    not_sent: set = set()          # candidate ids on sheets past the budget
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

        # Each read runs in a copy of the caller's context, so the run's
        # callbacks (activity log, token count) see these vision calls; a
        # plain thread pool starts every worker with an empty context. The
        # sheets go out in page order, a few at a time, and none is sent
        # once the budget is spent.
        import contextvars
        queue = list(sheets)
        with ThreadPoolExecutor(max_workers=VERIFY_WORKERS) as ex:
            running: set = set()
            while queue or running:
                while queue and len(running) < VERIFY_WORKERS:
                    if deadline is not None and _clock() >= deadline:
                        for _png, ids in queue:
                            not_sent.update(ids)
                        queue = []
                        break
                    running.add(ex.submit(contextvars.copy_context().run,
                                          read, queue.pop(0)))
                if not running:
                    break
                finished, running = wait(running, return_when=FIRST_COMPLETED)
                for f in finished:
                    ids, reply, err = f.result()
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
                   else "not_read" if i in not_sent
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
    left = pick("not_read")
    if left:
        pages_left = _page_ranges(r["page"] for r in left)
        out["status"] = "cut_short"
        out["budget_s"] = budget
        out["candidates"] = len(cands)
        out["candidates_read"] = len(cands) - len(left)
        out["not_read"] = len(left)
        out["not_read_pages"] = pages_left
        out["note"] = CUT_SHORT_NOTE.format(
            budget=budget, read=len(cands) - len(left), total=len(cands),
            left=len(left), pages=pages_left)
    if res.get("warnings"):
        out["warnings"] = res["warnings"]
    _fit(out)
    return out


def _page_ranges(pages) -> str:
    """``"3-5,9"`` for pages 3, 4, 5 and 9 (0-based, as ``pages`` takes
    them)."""
    ps = sorted({int(p) for p in pages})
    parts: List[str] = []
    i = 0
    while i < len(ps):
        j = i
        while j + 1 < len(ps) and ps[j + 1] == ps[j] + 1:
            j += 1
        parts.append(str(ps[i]) if i == j else f"{ps[i]}-{ps[j]}")
        i = j + 1
    return ",".join(parts)


def flooded_pages(hits: Sequence[Any],
                  per_page: int = FLOOD_PER_PAGE) -> Dict[int, int]:
    """``{page: candidates}`` for every page holding more than ``per_page``
    candidates — empty when the search looks like a mark's copies."""
    counts: Dict[int, int] = {}
    for h in hits:
        counts[h.page] = counts.get(h.page, 0) + 1
    return {p: n for p, n in sorted(counts.items()) if n > per_page}


def _flood_result(res: Dict[str, Any], hits: Sequence[Any],
                  flooded: Dict[int, int], text: Optional[str]
                  ) -> Dict[str, Any]:
    """What the tool says, without reading anything, when the example
    floods the search (:data:`FLOOD_PER_PAGE`)."""
    by_page: Dict[int, int] = {}
    for h in hits:
        by_page[h.page] = by_page.get(h.page, 0) + 1
    out: Dict[str, Any] = {
        "target": text, "example": res["example"],
        "pages_searched": len(res["pages"]),
        "status": "example_matches_linework",
        "candidates": len(hits),
        "candidates_by_page": {str(k): v for k, v in sorted(by_page.items())},
        "note": FLOOD_NOTE.format(
            n=max(flooded.values()),
            pages=", ".join(str(p) for p in list(flooded)[:10])),
    }
    if res.get("warnings"):
        out["warnings"] = res["warnings"]
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


__all__ = ["find_like", "parse_readings", "VERIFY_PROMPT", "PER_SHEET",
           "FLOOD_PER_PAGE", "flooded_pages", "BUDGET_ENV",
           "DEFAULT_BUDGET_S", "budget_s"]
