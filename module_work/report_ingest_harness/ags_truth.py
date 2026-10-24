"""Answer sheets for the logs stage, written from AGS4 data (BACKLOG RI-79).

Every hand-truthed log so far was written by a Claude session off the
rendered page. The public blind set's BGS scans come with something no
session wrote: the AGS4 data file the investigation contractor delivered for
the same holes. This turns that file into answer sheets in the SAME shape the
logs stage's truth files have (``raw/truth/logs/<ID>_p<page>.json``, read by
:mod:`report_ingest.log_scoring` and :mod:`.truth_records`), so the log reader
is scored on the scans against it with the scorer unchanged.

WHAT GOES WHERE::

    PROJ   project (PROJ_NAME), contract_number (PROJ_ID), client, contractor,
           engineer and location (the last two recorded, not scored)
    LOCA   investigation_id; boring_id (test_pit_id on a pit); total_depth
           (LOCA_FDEP); ground_surface_elevation (LOCA_GL, or LOCA_LOCZ when
           GL is blank, which the unit note says); easting and northing
           (recorded, not scored); method (LOCA_TYPE spelled out by the
           file's own ABBR rows -- a code the file does not spell out is
           kept, unscored, as loca_type); remarks (LOCA_REM, LOCA_TERM)
    GEOL   layers: top, bottom, description
    SAMP   samples: id (SAMP_REF), top, bottom, type, recovery (SAMP_RECV),
           blows (SAMP_UBLO, the driven tube's blow count, as a ring
           sample's blows are kept in the hand truth)
    ISPT   blows and N on the small disturbed sample taken from the spoon at
           that depth (same top, an interval no longer than the drive), else
           on an SPT row of its own; the blows are the counts of the seating
           and test increments in order, as ISPT_REP reports them
    WSTG   water: each strike, with its casing depth
    WSTD   water: the level the strike rose to at its LAST reading (a log
           prints where it settled); earlier readings are not slotted
    CORE   core runs: recovery (CORE_PREC) and RQD (CORE_RQD)
    LNMC   wc;  LLPL  ll, pl, pi;  GRAG  fines (GRAG_FINE) -- on the sample
           they were tested on (its top, reference and type)

UNITS. AGS4 depths are metres. A depth heading's own UNIT row is checked: a
blank one is taken as metres (the AGS4 standard) and ``units`` says so; one
declaring anything else stops the build rather than being converted. Every
other value keeps the unit the file gives it, recorded in ``units``.

WHAT HAS NO SLOT is listed, row by row, in ``ags_unslotted`` in the same
answer sheet -- the scorer reads only the keys it knows, as it already does
with the hand truth's ``tests_other`` -- so nothing is dropped silently.
Laboratory density (LDEN) and strength groups are there: the schema's
``duw`` is a dry UNIT WEIGHT and ``qu`` an unconfined compressive strength,
and turning a density or an undrained shear strength into either would be a
conversion this file made, not a value AGS holds.

DATES are recorded but not scored unless asked (``score_dates``): AGS writes
them ISO (2006-05-26), a log prints them its own way (26/05/06), and the
scorer compares text, so scoring them would score the format.

AGS AND THE PRINTED LOG CAN LEGITIMATELY DIFFER -- a depth rounded, a
description shortened on the log, a reading the log never printed. The AGS
value is the answer and the scorer's tolerances stay as they are.

THE ITEMS (``build``). One PDF per set ID is assembled from its scans, in
the natural order of the hole names, then page, so page numbers are stable;
the page map is written beside it. A hole that has both a scan and AGS data
gets one answer sheet, keyed by its first page (``P01_p0.json``, pages
0-based as everywhere in the stage) with every page of its scan in
``pages``: the reader is handed the whole hole, and the AGS data describes
the whole hole. Holes are matched to scans through the BGS index
(``onshore_index_records.json``): the scan's BGS ID gives the hole name and
the AGS location IDs, which are the AGS file's ``BGS_LOCA_ID``.

BLINDNESS. Nothing here renders, reads or describes a scanned page: the scan
PDFs are only counted and concatenated.

PRIVACY. Everything is written under the set's gitignored ``raw/`` folder,
and a path inside a checkout that git would track is refused. What this
prints is set IDs, page numbers and counts.

Run from the repo root (or a worktree) with the app venv::

    .venv/Scripts/python -m module_work.report_ingest_harness.ags_truth build
    .venv/Scripts/python -m module_work.report_ingest_harness.ags_truth build \\
        --ids P01,P02 --set-dir <the set's raw folder>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

__all__ = [
    "SET_REL", "read_ags", "abbreviations", "parse_spt_report",
    "truth_for_hole", "counts", "assemble", "build",
]

#: The public blind set's private folder, relative to the main checkout.
SET_REL = Path("module_work/field_feedback/2026-10-24_public_blind/raw")

#: Groups that describe the file, not a hole. PROJ is read for the header
#: fields; the rest only for units and abbreviations.
FILE_GROUPS = ("PROJ", "TRAN", "UNIT", "TYPE", "DICT", "ABBR", "FILE")

#: Headings holding a depth below ground. Their unit must be metres.
DEPTH_HEADINGS = ("LOCA_FDEP", "GEOL_TOP", "GEOL_BASE", "SAMP_TOP",
                  "SAMP_BASE", "SPEC_DPTH", "ISPT_TOP", "WSTG_DPTH",
                  "WSTD_POST", "CORE_TOP", "CORE_BASE")

#: The laboratory groups with a slot, heading -> the truth's key.
LAB_SLOTS: Dict[str, Dict[str, str]] = {
    "LNMC": {"LNMC_MC": "wc"},
    "LLPL": {"LLPL_LL": "ll", "LLPL_PL": "pl", "LLPL_PI": "pi"},
    "GRAG": {"GRAG_FINE": "fines"},
}
#: What identifies the sample a laboratory row was tested on.
LAB_KEYS = ("SAMP_TOP", "SAMP_REF", "SAMP_TYPE", "SAMP_ID", "SPEC_REF",
            "SPEC_DPTH")

#: The index-value keys an answer sheet's sample can carry.
INDEX_KEYS = ("wc", "ll", "pl", "pi", "fines", "rqd")

#: Two depths are the same depth when they differ by less than this (AGS
#: writes depths to the centimetre).
SAME_DEPTH_M = 0.005
#: A sample taken from the spoon is no longer than the drive (0.45 m).
SPT_DRIVE_M = 0.45


# ---------------------------------------------------------------------------
# reading the file
# ---------------------------------------------------------------------------

def read_ags(path: Any) -> Dict[str, Dict[str, Any]]:
    """Every group of an AGS4 file, through python-ags4.

    ``{group: {"headings": [...], "units": {heading: unit}, "types":
    {heading: type}, "rows": [{heading: value}, ...]}}``, every value a
    stripped string. A file that is not UTF-8 is read as cp1252, which is
    what the older exports are written in.
    """
    from python_ags4 import AGS4

    try:
        tables, _headings = AGS4.AGS4_to_dict(str(path), encoding="utf-8")
    except UnicodeDecodeError:
        tables, _headings = AGS4.AGS4_to_dict(str(path), encoding="cp1252")
    out: Dict[str, Dict[str, Any]] = {}
    for group, columns in tables.items():
        kinds = list(columns.get("HEADING") or [])
        headings = [h for h in columns if h != "HEADING"]
        one: Dict[str, Any] = {"headings": headings, "units": {},
                               "types": {}, "rows": []}
        for i, kind in enumerate(kinds):
            row = {h: (str(columns[h][i]).strip()
                       if i < len(columns[h]) else "") for h in headings}
            if kind == "UNIT":
                one["units"] = row
            elif kind == "TYPE":
                one["types"] = row
            elif kind == "DATA":
                one["rows"].append(row)
        out[group] = one
    return out


def abbreviations(ags: Dict[str, Any]) -> Dict[Tuple[str, str], str]:
    """``(heading, code) -> description`` from the file's own ABBR rows."""
    return {(r.get("ABBR_HDNG", ""), r.get("ABBR_CODE", "")):
            r.get("ABBR_DESC", "")
            for r in (ags.get("ABBR") or {}).get("rows") or []
            if r.get("ABBR_DESC")}


def _num(text: Any) -> Optional[float]:
    text = str(text if text is not None else "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _rows(ags: Dict[str, Any], group: str, loca_id: str) -> List[dict]:
    return [r for r in (ags.get(group) or {}).get("rows") or []
            if r.get("LOCA_ID") == loca_id]


def _unit(ags: Dict[str, Any], group: str, heading: str) -> str:
    return str(((ags.get(group) or {}).get("units") or {}).get(heading)
               or "").strip()


def _at(row: Dict[str, str]) -> Optional[float]:
    """The depth a row stands at, for the unslotted list."""
    for heading, value in row.items():
        if heading.endswith(("_TOP", "_DPTH")) and _num(value) is not None:
            return _num(value)
    return None


class _Unslotted:
    """What a hole's rows hold that no slot took, row by row."""

    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def add(self, group: str, row: Dict[str, str], used: Iterable[str] = (),
            why: str = "") -> None:
        used = set(used) | {"LOCA_ID"}
        values = {h: v for h, v in row.items() if v and h not in used}
        if not values:
            return
        entry: Dict[str, Any] = {"group": group, "at": _at(row),
                                 "values": values}
        if why:
            entry["why"] = why
        self.rows.append(entry)


# ---------------------------------------------------------------------------
# the SPT report
# ---------------------------------------------------------------------------

_N_RE = re.compile(r"\bN\s*=\s*(\d+)", re.I)
_INCREMENT_RE = re.compile(r"(\d+)(?:\s*for\s*(\d+(?:\.\d+)?)\s*mm)?", re.I)


def parse_spt_report(text: Any) -> Dict[str, Any]:
    """An ISPT_REP string, e.g. ``N=12 (1,2/2,3,3,4)`` or
    ``50 (4,21 for 60mm/40,10)``.

    ``n`` is the N the report states (``N=``), else None; ``blows`` the blow
    count of every seating and test increment in order; ``partial`` each
    increment driven short of its length, as the report words it;
    ``refusal`` is a report with blows and no ``N=`` (the drive was stopped);
    ``parsed`` is False when the bracket held something this cannot read,
    and the report then gives no blows at all rather than some of them.
    """
    text = str(text or "").strip()
    out: Dict[str, Any] = {"n": None, "blows": [], "partial": [],
                           "refusal": False, "parsed": True}
    if not text:
        out["parsed"] = False
        return out
    stated = _N_RE.search(text)
    if stated:
        out["n"] = int(stated.group(1))
    bracket = re.search(r"\(([^)]*)\)", text)
    if bracket:
        for part in re.split(r"[,/]", bracket.group(1)):
            part = part.strip()
            if not part:
                continue
            one = _INCREMENT_RE.fullmatch(part)
            if one is None:
                return {**out, "blows": [], "partial": [], "parsed": False}
            out["blows"].append(int(one.group(1)))
            if one.group(2) is not None:
                out["partial"].append(
                    f"{one.group(1)} blows for {one.group(2)} mm")
    elif out["n"] is None:
        out["parsed"] = False
    out["refusal"] = out["n"] is None and bool(out["blows"])
    return out


# ---------------------------------------------------------------------------
# one hole
# ---------------------------------------------------------------------------

def _is_pit(code: str, spelled: str) -> bool:
    return "pit" in spelled.lower() or "trench" in spelled.lower() \
        or code.upper() == "TP"


def _spell(code: str, abbr: Dict[Tuple[str, str], str]) -> Optional[str]:
    """LOCA_TYPE in words from the file's ABBR rows: ``CP+RC`` needs both
    parts spelled out, or None."""
    parts = [p.strip() for p in re.split(r"[+/]", code) if p.strip()]
    words = [abbr.get(("LOCA_TYPE", p)) for p in parts]
    if not parts or not all(words):
        return None
    return " + ".join(str(w) for w in words)


def _check_depth_units(ags: Dict[str, Any], units: Dict[str, str]) -> None:
    for group, one in ags.items():
        for heading in one.get("headings") or []:
            if heading not in DEPTH_HEADINGS:
                continue
            unit = _unit(ags, group, heading)
            if unit and unit.lower() != "m":
                raise ValueError(
                    f"{group}.{heading} is in {unit!r}: AGS4 depths are "
                    f"metres and this does not convert them")
            units.setdefault(heading, unit or
                             "m (blank in the file; the AGS4 standard)")


def _same(a: Optional[float], b: Optional[float]) -> bool:
    return a is not None and b is not None and abs(a - b) < SAME_DEPTH_M


def _header(ags: Dict[str, Any], loca: Dict[str, str],
            abbr: Dict[Tuple[str, str], str], pit: bool, score_dates: bool,
            units: Dict[str, str], unslotted: _Unslotted
            ) -> Tuple[Dict[str, Any], str]:
    fields: Dict[str, Any] = {}
    used: Set[str] = {"LOCA_ID"}
    fields["test_pit_id" if pit else "boring_id"] = loca["LOCA_ID"]

    for proj in (ags.get("PROJ") or {}).get("rows") or []:
        taken = set()
        for heading, key in (("PROJ_NAME", "project"),
                             ("PROJ_ID", "contract_number"),
                             ("PROJ_CLNT", "client"),
                             ("PROJ_CONT", "contractor"),
                             ("PROJ_ENG", "engineer"),
                             ("PROJ_LOC", "location")):
            value = proj.get(heading, "")
            # A location of "0" is a placeholder, not a place.
            if value and value != "0":
                fields.setdefault(key, value)
                taken.add(heading)
        unslotted.add("PROJ", proj, taken)

    code = loca.get("LOCA_TYPE", "")
    if code:
        spelled = _spell(code, abbr)
        if spelled:
            fields["method"] = spelled
        else:
            fields["loca_type"] = code
        used.add("LOCA_TYPE")
    for heading, key in (("LOCA_FDEP", "total_depth"),
                         ("LOCA_NATE", "easting"),
                         ("LOCA_NATN", "northing")):
        value = _num(loca.get(heading))
        if value is not None:
            fields[key] = value
            used.add(heading)
            units[heading] = _unit(ags, "LOCA", heading) or units.get(
                heading, "")
    level = _num(loca.get("LOCA_GL"))
    if level is not None:
        fields["ground_surface_elevation"] = level
        used.add("LOCA_GL")
        units["LOCA_GL"] = _unit(ags, "LOCA", "LOCA_GL")
    elif _num(loca.get("LOCA_LOCZ")) is not None:
        fields["ground_surface_elevation"] = _num(loca["LOCA_LOCZ"])
        used.add("LOCA_LOCZ")
        units["LOCA_LOCZ"] = (_unit(ags, "LOCA", "LOCA_LOCZ")
                              + " (LOCA_GL is blank; the ground level "
                                "taken from LOCA_LOCZ)").strip()
    for heading, key in (("LOCA_STAR", "date_started"),
                         ("LOCA_ENDD", "date_finished")):
        value = loca.get(heading, "")
        if value:
            fields[key if score_dates else f"{key}_ags"] = value
            used.add(heading)
    remarks = [loca.get(h, "") for h in ("LOCA_REM", "LOCA_TERM")
               if loca.get(h)]
    used.update(h for h in ("LOCA_REM", "LOCA_TERM") if loca.get(h))
    unslotted.add("LOCA", loca, used)
    return fields, " ".join(remarks)


def _layers(ags: Dict[str, Any], loca_id: str,
            unslotted: _Unslotted) -> List[Dict[str, Any]]:
    out = []
    for row in _rows(ags, "GEOL", loca_id):
        top = _num(row.get("GEOL_TOP"))
        if top is None:
            unslotted.add("GEOL", row, why="no top depth")
            continue
        out.append({"top": top, "bottom": _num(row.get("GEOL_BASE")),
                    "description": row.get("GEOL_DESC", ""), "uscs": None})
        unslotted.add("GEOL", row, ("GEOL_TOP", "GEOL_BASE", "GEOL_DESC"))
    out.sort(key=lambda layer: layer["top"])
    return out


def _sample_row(sample_id: str, top: float, bottom: Optional[float],
                kind: str, desc: str = "") -> Dict[str, Any]:
    row: Dict[str, Any] = {"id": sample_id, "top": top, "bottom": bottom,
                           "type": kind, "blows": None, "n": None,
                           "recovery": None}
    if desc:
        row["type_desc"] = desc
    return row


def _samples(ags: Dict[str, Any], loca_id: str,
             abbr: Dict[Tuple[str, str], str], units: Dict[str, str],
             unslotted: _Unslotted) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in _rows(ags, "SAMP", loca_id):
        top = _num(row.get("SAMP_TOP"))
        if top is None:
            unslotted.add("SAMP", row, why="no top depth")
            continue
        kind = row.get("SAMP_TYPE", "")
        sample = _sample_row(row.get("SAMP_REF", ""), top,
                             _num(row.get("SAMP_BASE")), kind,
                             abbr.get(("SAMP_TYPE", kind), ""))
        used = {"SAMP_TOP", "SAMP_BASE", "SAMP_REF", "SAMP_TYPE"}
        recovery = _num(row.get("SAMP_RECV"))
        if recovery is not None:
            sample["recovery"] = recovery
            used.add("SAMP_RECV")
            units["SAMP_RECV"] = _unit(ags, "SAMP", "SAMP_RECV") or "%"
        blows = _num(row.get("SAMP_UBLO"))
        if blows is not None:
            sample["blows"] = [int(blows)]
            used.add("SAMP_UBLO")
        if row.get("SAMP_REM"):
            sample["note"] = row["SAMP_REM"]
            used.add("SAMP_REM")
        out.append(sample)
        unslotted.add("SAMP", row, used)
    return out


def _spoon_sample(samples: List[Dict[str, Any]], top: float
                  ) -> Optional[Dict[str, Any]]:
    """The small disturbed sample taken from the spoon at ``top``: same
    top, an interval no longer than the drive, no blows of its own."""
    for sample in samples:
        if not _same(sample["top"], top) or sample.get("blows"):
            continue
        if str(sample.get("type", "")).upper() not in ("D", "SPT", "SPTS",
                                                       "SPTLS"):
            continue
        bottom = sample.get("bottom")
        if bottom is None or bottom - top <= SPT_DRIVE_M + SAME_DEPTH_M:
            return sample
    return None


def _spt(ags: Dict[str, Any], loca_id: str, samples: List[Dict[str, Any]],
         fields: Dict[str, Any], unslotted: _Unslotted) -> None:
    rows = _rows(ags, "ISPT", loca_id)
    hammers = {r.get("ISPT_HAM", "") for r in rows if r.get("ISPT_HAM")}
    one_hammer = len(hammers) == 1
    if one_hammer:
        fields["hammer"] = next(iter(hammers))
    for row in rows:
        top = _num(row.get("ISPT_TOP"))
        if top is None:
            unslotted.add("ISPT", row, why="no top depth")
            continue
        used = {"ISPT_TOP"}
        report = parse_spt_report(row.get("ISPT_REP"))
        n = report["n"]
        if _num(row.get("ISPT_NVAL")) is not None:
            n = int(_num(row["ISPT_NVAL"]))
            used.add("ISPT_NVAL")
        blows = list(report["blows"])
        if report["parsed"] and row.get("ISPT_REP"):
            used.add("ISPT_REP")
        if not blows:
            incs = [f"ISPT_INC{i}" for i in range(1, 7)
                    if _num(row.get(f"ISPT_INC{i}")) is not None]
            blows = [int(_num(row[h])) for h in incs]
            used.update(incs)
        if one_hammer:
            used.add("ISPT_HAM")
        if n is None and not blows:
            unslotted.add("ISPT", row, why="the report gives no N and no "
                                           "blows this could read")
            continue
        sample = _spoon_sample(samples, top)
        if sample is None:
            sample = _sample_row("", top, None, "spt")
            samples.append(sample)
        sample["blows"] = blows or None
        sample["n"] = n
        if n is None:
            sample["refusal"] = True
        if row.get("ISPT_REP"):
            sample["spt_report"] = row["ISPT_REP"]
        if report["partial"]:
            sample["note"] = "; ".join(
                ([sample["note"]] if sample.get("note") else [])
                + report["partial"])
        unslotted.add("ISPT", row, used)


def _water(ags: Dict[str, Any], loca_id: str, units: Dict[str, str],
           unslotted: _Unslotted) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    strikes = _rows(ags, "WSTG", loca_id)
    for row in strikes:
        depth = _num(row.get("WSTG_DPTH"))
        if depth is None:
            unslotted.add("WSTG", row, why="no depth")
            continue
        water: Dict[str, Any] = {"depth": depth,
                                 "when": "encountered (water strike)"}
        used = {"WSTG_DPTH"}
        casing = _num(row.get("WSTG_CAS"))
        if casing is not None:
            water["casing"] = casing
            used.add("WSTG_CAS")
            units["WSTG_CAS"] = _unit(ags, "WSTG", "WSTG_CAS") or (
                "m (blank in the file; the AGS4 standard)")
        if row.get("WSTG_DTIM"):
            water["date"] = row["WSTG_DTIM"]
            used.add("WSTG_DTIM")
        if row.get("WSTG_REM"):
            water["note"] = row["WSTG_REM"]
            used.add("WSTG_REM")
        out.append(water)
        unslotted.add("WSTG", row, used)

    readings: Dict[Optional[float], List[dict]] = {}
    for row in _rows(ags, "WSTD", loca_id):
        readings.setdefault(_num(row.get("WSTG_DPTH")), []).append(row)
    for strike, rows in readings.items():
        rows = sorted(rows, key=lambda r: (_num(r.get("WSTD_NMIN"))
                                           if _num(r.get("WSTD_NMIN"))
                                           is not None else -1.0))
        last = rows[-1]
        for earlier in rows[:-1]:
            unslotted.add("WSTD", earlier, why="an earlier reading; the "
                                               "last one is slotted")
        post = _num(last.get("WSTD_POST"))
        if post is None or _same(post, strike):
            unslotted.add("WSTD", last, ("WSTG_DPTH",),
                          why="" if post is None else "no rise")
            continue
        minutes = _num(last.get("WSTD_NMIN"))
        when = "rose to" + (f", {minutes:g} min after the strike"
                            if minutes is not None else "")
        if strike is not None:
            when += f" at {strike:g} m"
        water = {"depth": post, "when": when}
        if last.get("WSTD_REM"):
            water["note"] = last["WSTD_REM"]
        out.append(water)
        unslotted.add("WSTD", last, ("WSTG_DPTH", "WSTD_NMIN", "WSTD_POST",
                                     "WSTD_REM"))
    out.sort(key=lambda w: w["depth"])
    return out


def _core(ags: Dict[str, Any], loca_id: str, samples: List[Dict[str, Any]],
          units: Dict[str, str], unslotted: _Unslotted) -> None:
    for row in _rows(ags, "CORE", loca_id):
        top = _num(row.get("CORE_TOP"))
        if top is None:
            unslotted.add("CORE", row, why="no top depth")
            continue
        run = _sample_row("", top, _num(row.get("CORE_BASE")), "core")
        used = {"CORE_TOP", "CORE_BASE"}
        for heading, key in (("CORE_PREC", "recovery"), ("CORE_RQD", "rqd")):
            value = _num(row.get(heading))
            if value is not None:
                run[key] = value
                used.add(heading)
                units[heading] = _unit(ags, "CORE", heading) or "%"
        samples.append(run)
        unslotted.add("CORE", row, used)


def _lab(ags: Dict[str, Any], loca_id: str, samples: List[Dict[str, Any]],
         units: Dict[str, str], unslotted: _Unslotted) -> None:
    for group, slots in LAB_SLOTS.items():
        for row in _rows(ags, group, loca_id):
            top = _num(row.get("SAMP_TOP"))
            if top is None:
                top = _num(row.get("SPEC_DPTH"))
            if top is None:
                unslotted.add(group, row, why="no sample depth")
                continue
            ref, kind = row.get("SAMP_REF", ""), row.get("SAMP_TYPE", "")
            # The sample with this top and reference (and type, where more
            # than one shares them); with no reference, the first at the
            # top. A reference SAMP does not list gets a row of its own
            # rather than another sample's.
            same_top = [s for s in samples if _same(s["top"], top)]
            match = [s for s in same_top if s["id"] == ref] if ref \
                else same_top
            sample = next((s for s in match if s["type"] == kind),
                          match[0] if match else None)
            if sample is None:
                sample = _sample_row(ref, top, None, kind)
                samples.append(sample)
            used = set()
            for heading, key in slots.items():
                value = row.get(heading, "")
                if not value:
                    continue
                if sample.get(key) is not None:
                    continue                   # a second specimen: unslotted
                number = _num(value)
                sample[key] = number if number is not None else value
                used.add(heading)
                units[heading] = _unit(ags, group, heading)
            if used:
                used.update(LAB_KEYS)
            unslotted.add(group, row, used)


def truth_for_hole(ags: Dict[str, Any], loca_id: str, *, item_id: str,
                   pages: Sequence[int], source_file: str = "",
                   score_dates: bool = False) -> Dict[str, Any]:
    """One hole of an AGS4 file as a logs-stage answer sheet."""
    loca = next((r for r in (ags.get("LOCA") or {}).get("rows") or []
                 if r.get("LOCA_ID") == loca_id), None)
    if loca is None:
        raise KeyError(f"{loca_id} is not in the file's LOCA group")
    units: Dict[str, str] = {}
    _check_depth_units(ags, units)
    abbr = abbreviations(ags)
    code = loca.get("LOCA_TYPE", "")
    pit = bool(code) and _is_pit(code, _spell(code, abbr) or "")
    unslotted = _Unslotted()

    fields, remarks = _header(ags, loca, abbr, pit, score_dates, units,
                              unslotted)
    layers = _layers(ags, loca_id, unslotted)
    samples = _samples(ags, loca_id, abbr, units, unslotted)
    _spt(ags, loca_id, samples, fields, unslotted)
    _core(ags, loca_id, samples, units, unslotted)
    _lab(ags, loca_id, samples, units, unslotted)
    samples.sort(key=lambda s: (s["top"], str(s.get("id") or "")))
    water = _water(ags, loca_id, units, unslotted)

    # Groups with no slot at all: every row of theirs for this hole.
    slotted = {"LOCA", "GEOL", "SAMP", "ISPT", "WSTG", "WSTD", "CORE",
               *LAB_SLOTS}
    for group in ags:
        if group in FILE_GROUPS or group in slotted:
            continue
        for row in _rows(ags, group, loca_id):
            unslotted.add(group, row, why="no slot for this group")

    truth: Dict[str, Any] = {
        "id": item_id,
        "investigation_id": loca_id,
        "pages": [int(p) for p in pages],
        "depth_unit": "m",
        "source": "AGS4 data file (BGS export); not read off the page",
        "kind": "test_pit" if pit else "boring",
        "fields": fields,
        "layers": layers,
        "samples": samples,
        "water": water,
        "units": units,
        "ags": {"file": source_file, "loca_id": loca_id,
                "bgs_loca_id": loca.get("BGS_LOCA_ID", ""),
                "dates_scored": score_dates},
        "ags_unslotted": unslotted.rows,
    }
    if remarks:
        truth["remarks"] = remarks
    return truth


def counts(truth: Dict[str, Any]) -> Dict[str, int]:
    """How many answer values of each kind an answer sheet holds."""
    from report_ingest.log_scoring import FIELD_MAP

    samples = truth.get("samples") or []
    spt = [s for s in samples if s.get("spt_report") or s.get("type") == "spt"]
    return {
        "layers": sum(1 for x in truth.get("layers") or []
                      if x.get("top") is not None),
        "samples": len(samples),
        "spt_n": sum(1 for s in spt if s.get("n") is not None),
        "spt_blows": sum(1 for s in spt if s.get("blows")),
        "tube_blows": sum(1 for s in samples
                          if s.get("blows") and s not in spt),
        "water": sum(1 for w in truth.get("water") or []
                     if w.get("depth") is not None),
        "lab": sum(1 for s in samples for k in INDEX_KEYS
                   if s.get(k) is not None),
        "recovery": sum(1 for s in samples if s.get("recovery") is not None),
        "fields_scored": sum(1 for k in (truth.get("fields") or {})
                             if k in FIELD_MAP),
        "unslotted_rows": len(truth.get("ags_unslotted") or []),
    }


# ---------------------------------------------------------------------------
# the items: one PDF per set ID, and its answer sheets
# ---------------------------------------------------------------------------

def _natural(name: str) -> Tuple:
    return tuple(int(t) if t.isdigit() else t.lower()
                 for t in re.split(r"(\d+)", name) if t)


def _hole_name(record: Dict[str, Any]) -> str:
    """The hole's own name: the last word of the index's ``name``, which is
    the project's name and then the hole's."""
    words = str(record.get("name") or "").split()
    return words[-1] if words else ""


def _scan_entries(folder: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    index_path = folder / "onshore_index_records.json"
    records = (json.loads(index_path.read_text(encoding="utf-8"))
               if index_path.is_file() else [])
    by_bgs = {int(r["bgs_id"]): r for r in records
              if r.get("bgs_id") is not None}
    entries, notes = [], []
    for scan in sorted(folder.glob("scan_*.pdf")):
        found = re.search(r"(\d+)", scan.stem)
        bgs = int(found.group(1)) if found else -1
        record = by_bgs.get(bgs)
        if record is None:
            notes.append(f"{scan.name}: not in the index, so which hole it "
                         f"shows is not known; it is assembled but has no "
                         f"answer sheet")
        entries.append({
            "bgs_id": bgs, "scan": scan,
            "hole": _hole_name(record) if record else "",
            "loca_ids": re.findall(r"loca_id=(\d+)",
                                   str((record or {}).get("ags_log_url")
                                       or ""))})
    entries.sort(key=lambda e: (not e["hole"], _natural(e["hole"]),
                                e["bgs_id"]))
    return entries, notes


def assemble(folder: Path, target: Path
             ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """Concatenate a set ID's scans into ``target`` (natural order of the
    hole names, then page): ``(holes, page map, notes)``. Each hole carries
    the pages it fills. The scans are only counted and copied: no page is
    rendered or read."""
    import pymupdf

    entries, notes = _scan_entries(folder)
    page_map: List[Dict[str, Any]] = []
    out = pymupdf.open()
    try:
        for entry in entries:
            with pymupdf.open(str(entry["scan"])) as src:
                n = src.page_count
                out.insert_pdf(src)
            first = len(page_map)
            entry["pages"] = list(range(first, first + n))
            for k in range(n):
                page_map.append({"page": first + k, "bgs_id": entry["bgs_id"],
                                 "hole": entry["hole"],
                                 "scan_file": entry["scan"].name,
                                 "scan_page": k, "scan_pages": n})
        target.parent.mkdir(parents=True, exist_ok=True)
        out.save(str(target), garbage=3, deflate=True)
    finally:
        out.close()
    holes = [{**e, "scan": e["scan"].name} for e in entries]
    return holes, page_map, notes


def _refuse_tracked(paths: Iterable[Path]) -> None:
    """Stop before writing a path a checkout would track."""
    from module_work.report_ingest_harness.run_claude_local import (
        REPO, _inside, git_ignored, main_checkout,
    )

    for path in paths:
        for root in {main_checkout(), REPO}:
            if _inside(path, root) and git_ignored(path, root) is False:
                raise PermissionError(
                    f"refused: {path} is inside the repo and git does not "
                    f"ignore it; the answer sheets and scans are private")


def _hole_view(hole: Dict[str, Any]) -> Dict[str, Any]:
    return {k: hole[k] for k in ("hole", "bgs_id", "scan", "pages",
                                 "loca_ids", "answer_sheet") if k in hole}


def build(set_dir: Path, ids: Sequence[str], *,
          score_dates: bool = False) -> Dict[str, Dict[str, Any]]:
    """Assemble each set ID's PDF and write its answer sheets.

    Writes ``<set_dir>/assembled/<ID>.pdf`` and ``<ID>_pages.json`` (the
    page map, which hole each page shows and which answer sheet it has) and
    ``<set_dir>/truth/logs/<ID>_p<first page>.json``; a set ID's earlier
    answer sheets are replaced. ``OPEN.txt`` there names no report, so
    every item is scored as blind. Returns counts per set ID.
    """
    set_dir = Path(set_dir)
    assembled = set_dir / "assembled"
    truth_dir = set_dir / "truth" / "logs"
    summary: Dict[str, Dict[str, Any]] = {}
    for pid in ids:
        folder = set_dir / pid
        if not folder.is_dir():
            raise FileNotFoundError(f"no folder for {pid} at {folder}")
        target = assembled / f"{pid}.pdf"
        map_path = assembled / f"{pid}_pages.json"
        _refuse_tracked([target, map_path, truth_dir / f"{pid}_p0.json",
                         truth_dir / "OPEN.txt"])
        holes, page_map, notes = assemble(folder, target)
        by_loca = {lid: hole for hole in holes for lid in hole["loca_ids"]}
        by_name: Dict[str, List[Dict[str, Any]]] = {}
        for hole in holes:
            if hole["hole"]:
                by_name.setdefault(hole["hole"], []).append(hole)

        sheets: Dict[str, Dict[str, Any]] = {}
        no_scan: List[Dict[str, str]] = []
        locations = 0
        for ags_path in sorted((folder / "ags").glob("*.ags")):
            ags = read_ags(ags_path)
            for loca in (ags.get("LOCA") or {}).get("rows") or []:
                locations += 1
                hole = by_loca.get(loca.get("BGS_LOCA_ID", ""))
                if hole is None and len(by_name.get(loca["LOCA_ID"], ())) == 1:
                    hole = by_name[loca["LOCA_ID"]][0]
                if hole is None:
                    no_scan.append({"loca_id": loca["LOCA_ID"],
                                    "bgs_loca_id": loca.get("BGS_LOCA_ID",
                                                            ""),
                                    "file": ags_path.name})
                    continue
                item_id = f"{pid}_p{hole['pages'][0]}"
                if item_id in sheets:
                    notes.append(f"{item_id}: a second AGS location for the "
                                 f"same scan in {ags_path.name}; the first "
                                 f"is kept")
                    continue
                sheets[item_id] = truth_for_hole(
                    ags, loca["LOCA_ID"], item_id=item_id,
                    pages=hole["pages"],
                    source_file=f"{pid}/ags/{ags_path.name}",
                    score_dates=score_dates)
                hole["answer_sheet"] = item_id

        truth_dir.mkdir(parents=True, exist_ok=True)
        for stale in truth_dir.glob(f"{pid}_p*.json"):
            stale.unlink()
        for item_id, truth in sheets.items():
            (truth_dir / f"{item_id}.json").write_text(
                json.dumps(truth, indent=1, ensure_ascii=False),
                encoding="utf-8")
        map_path.write_text(json.dumps({
            "set": pid,
            "order": "holes in the natural order of their names, then page; "
                     "pages 0-based",
            "pages": page_map,
            "holes": [_hole_view(h) for h in holes],
            "ags_locations_with_no_scan": no_scan,
            "notes": notes}, indent=1, ensure_ascii=False), encoding="utf-8")

        totals: Dict[str, int] = {}
        for truth in sheets.values():
            for key, value in counts(truth).items():
                totals[key] = totals.get(key, 0) + value
        summary[pid] = {
            "pages": len(page_map), "scans": len(holes),
            "ags_locations": locations, "answer_sheets": len(sheets),
            "pages_with_answers": sum(len(t["pages"])
                                      for t in sheets.values()),
            "ags_locations_with_no_scan": len(no_scan),
            "notes": len(notes), **totals}

    truth_dir.mkdir(parents=True, exist_ok=True)
    (truth_dir / "OPEN.txt").write_text("(none)\n", encoding="utf-8")
    return summary


def summary_line(pid: str, row: Dict[str, Any]) -> str:
    return (f"{pid}: {row['pages']} page(s) from {row['scans']} scan(s); "
            f"{row['ags_locations']} AGS location(s), "
            f"{row['answer_sheets']} matched to a scan "
            f"({row['pages_with_answers']} page(s)), "
            f"{row['ags_locations_with_no_scan']} with no scan. Answer "
            f"values: layers {row.get('layers', 0)}, samples "
            f"{row.get('samples', 0)}, SPT N {row.get('spt_n', 0)}, SPT "
            f"blows {row.get('spt_blows', 0)}, tube blows "
            f"{row.get('tube_blows', 0)}, water "
            f"{row.get('water', 0)}, lab {row.get('lab', 0)}, recovery "
            f"{row.get('recovery', 0)}, scored header fields "
            f"{row.get('fields_scored', 0)}; unslotted rows "
            f"{row.get('unslotted_rows', 0)}.")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="assemble the PDFs and write the "
                                     "answer sheets ($0, no model call)")
    b.add_argument("--set-dir", default=None,
                   help="the set's raw folder (default: the main "
                        "checkout's public blind set)")
    b.add_argument("--ids", default="P01,P02,P03")
    b.add_argument("--score-dates", action="store_true",
                   help="put the AGS dates where the scorer reads them")
    args = ap.parse_args(argv)
    if args.set_dir:
        set_dir = Path(args.set_dir)
    else:
        from module_work.report_ingest_harness.run_claude_local import (
            main_checkout,
        )
        set_dir = main_checkout() / SET_REL
    ids = [t.strip() for t in args.ids.split(",") if t.strip()]
    summary = build(set_dir, ids, score_dates=args.score_dates)
    for pid, row in summary.items():
        print(summary_line(pid, row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
