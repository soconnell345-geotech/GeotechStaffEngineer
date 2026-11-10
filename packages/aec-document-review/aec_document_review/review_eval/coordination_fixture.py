"""A made-up building's documents with coordination problems planted in them.

WHY. The suite's other tasks ask about one place in one document, and every
build of the page now answers nearly all of them (BACKLOG DR-25). A design
review is mostly the other kind of question: does this sheet agree with that
one, does the callout lead anywhere, does the schedule hold what the plan
tags, do the drawings follow the specification and the geotechnical
recommendations, what changed between two issues. Such a question can only
be scored where every disagreement in the documents is known, so these
documents are built here, from one small model of a building, and the
problems are planted in the model by name.

WHAT IS BUILT. Everything belongs to one invented project (a small office
building; every name and number is made up):

* the drawing set, issue 1 (15 sheets, 34 x 22 in, real text at drawing
  sizes: 6 to 9 pt), with 13 planted problems (:data:`BASE_PLANTED`);
* the same set at issue 0, before three changes: two clouded and listed in
  their sheets' revision blocks, one made silently (:data:`REVISIONS`);
* the issue 1 set as a scan: each sheet a grey image at 150 dpi, no text
  layer, so only a look at the page reads it (:func:`build_scanned_set`);
* a five-storey version of the set (40 sheets, with reflected ceiling,
  mechanical and power plans) that keeps the 13 problems and adds six
  (:data:`LONG_PLANTED`);
* the project manual (a table of contents and three specification
  sections), the geotechnical recommendations memorandum, and a concrete mix
  submittal with four things wrong with it (:data:`SUBMITTAL_PLANTED`).

Callouts are drawn as a reviewer meets them: a bubble with the view number
over the sheet number, two separate pieces of text. A search for "7/A-501"
finds nothing; the page has to be read.

:func:`audit` derives the problems again from the model by rule, and the
tests hold it equal to the planted lists, so the documents carry the planted
problems and no others of those kinds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT = "NORTH YARD ANNEX"
PROJECT_NO = "24-117"
MEMO_NO = "G-24-031"
ISSUE_DATES = ("2026-01-12", "2026-03-02")

FLOOR_WORDS = ("FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH")
FIRST_FLOOR_EL = 100.00
STOREY_M = 4.20
PARAPET_M = 0.90
#: Grid bays, in millimetres: (from, to, length).
BAYS_X = (("A", "B", 6000), ("B", "C", 7200), ("C", "D", 6000))
BAYS_Y = (("1", "2", 5400), ("2", "3", 5400))

#: The sections of the project manual, as its table of contents lists them.
MANUAL_SECTIONS = (
    ("01 33 00", "SUBMITTAL PROCEDURES"),
    ("03 30 00", "CAST-IN-PLACE CONCRETE"),
    ("04 20 00", "UNIT MASONRY"),
    ("05 12 00", "STRUCTURAL STEEL FRAMING"),
    ("05 31 00", "STEEL DECKING"),
    ("07 21 00", "THERMAL INSULATION"),
    ("07 92 00", "JOINT SEALANTS"),
    ("08 11 13", "HOLLOW METAL DOORS AND FRAMES"),
    ("08 51 13", "ALUMINUM WINDOWS"),
    ("08 71 00", "DOOR HARDWARE"),
    ("09 29 00", "GYPSUM BOARD"),
    ("09 51 00", "ACOUSTICAL CEILINGS"),
    ("09 91 00", "PAINTING"),
    ("31 20 00", "EARTH MOVING"),
)
#: The sections the drawings' general notes cite (G-002).
CITED_SECTIONS = (
    ("03 30 00", "CAST-IN-PLACE CONCRETE"),
    ("04 20 00", "UNIT MASONRY"),
    ("07 84 00", "FIRESTOPPING AT RATED PENETRATIONS"),
    ("07 92 00", "JOINT SEALANTS"),
    ("08 11 13", "HOLLOW METAL DOORS AND FRAMES"),
    ("08 71 00", "DOOR HARDWARE"),
    ("09 91 00", "PAINTING"),
    ("01 33 00", "SUBMITTALS"),
)
#: 28-day concrete strengths in MPa: (use, the specification, the drawings).
CONCRETE = (
    ("FOOTINGS", 30, 25),
    ("SLABS ON GRADE", 30, 30),
    ("COLUMNS AND SUSPENDED SLABS", 35, 35),
)
#: Foundation values: (what, the memorandum, the drawings at issue 0, at
#: issue 1), printed with the unit given.
FOUNDATION = (
    ("bearing", "150 kPa", "150 kPa", "200 kPa"),
    ("depth", "1.5 m", "1.2 m", "1.2 m"),
    ("site_class", "C", "C", "C"),
    ("groundwater", "EL. 96.50", "EL. 96.50", "EL. 96.50"),
    ("subgrade", "40 MN/m3", "40 MN/m3", "40 MN/m3"),
)


@dataclass(frozen=True)
class Planted:
    """One planted problem: its name, the sheets (or document) that show it,
    and what a reviewer would write."""
    key: str
    sheets: Tuple[str, ...]
    truth: str


BASE_PLANTED: Tuple[Planted, ...] = (
    Planted("index-missing-sheet", ("G-001",),
            "The sheet index lists A-202 INTERIOR ELEVATIONS; there is no "
            "sheet A-202 in the set."),
    Planted("index-title", ("G-001", "S-201"),
            "The sheet index calls S-201 ROOF FRAMING PLAN; the sheet's own "
            "title is SECOND FLOOR FRAMING PLAN."),
    Planted("callout-missing-detail", ("A-101", "A-501"),
            "A-101 calls out detail 7/A-501; sheet A-501 holds details 1 to "
            "6 only."),
    Planted("callout-missing-sheet", ("A-102",),
            "A-102 calls out section 2/A-302; there is no sheet A-302 (the "
            "sections are on A-301)."),
    Planted("door-not-scheduled", ("A-101", "A-601"),
            "Door 109 is tagged on A-101 but is not in the door schedule."),
    Planted("door-not-on-plan", ("A-102", "A-601"),
            "Door 204 is in the door schedule but is not tagged on A-102."),
    Planted("level-second-floor", ("A-201", "A-301"),
            "SECOND FLOOR is EL. 104.20 on the elevations (A-201) and the "
            "framing plan (S-201) but EL. 104.50 on the sections (A-301)."),
    Planted("bay-b-c", ("A-101", "A-401"),
            "Grids B to C are 7200 on the floor plans and the structural "
            "plans but 7500 on the enlarged plans (A-401)."),
    Planted("spec-section-missing", ("G-002",),
            "General note 6 cites Section 07 84 00, which is not in the "
            "project manual."),
    Planted("bearing-pressure", ("S-001",),
            "S-001 gives an allowable bearing pressure of 200 kPa; the "
            "geotechnical memorandum recommends 150 kPa."),
    Planted("footing-depth", ("S-001",),
            "S-001 puts the bottom of exterior footings at least 1.2 m below "
            "finished grade; the geotechnical memorandum asks for 1.5 m."),
    Planted("footing-concrete", ("S-001",),
            "S-001 gives footings 25 MPa concrete; Section 03 30 00 requires "
            "30 MPa for footings."),
    Planted("footing-not-scheduled", ("S-101", "S-501"),
            "Footing F4 is tagged on S-101 but is not in the footing "
            "schedule on S-501, which lists F1 to F3."),
)

LONG_PLANTED: Tuple[Planted, ...] = BASE_PLANTED + (
    Planted("callout-missing-ceiling-detail", ("A-114", "A-502"),
            "A-114 calls out detail 9/A-502; sheet A-502 holds details 1 to "
            "8 only."),
    Planted("door-not-scheduled-4", ("A-104", "A-602"),
            "Door 407 is tagged on A-104 but is not in the door schedule."),
    Planted("door-not-on-plan-5", ("A-105", "A-602"),
            "Door 509 is in the door schedule but is not tagged on A-105."),
    Planted("level-fifth-floor", ("A-201", "A-301"),
            "FIFTH FLOOR is EL. 116.80 on the elevations (A-201) and the "
            "framing plan (S-204) but EL. 116.50 on the sections (A-301)."),
    Planted("fcu-not-scheduled", ("M-103", "M-601"),
            "FCU-3-4 is shown on M-103 but is not in the mechanical "
            "equipment schedule."),
    Planted("panel-not-scheduled", ("E-104", "E-601"),
            "Panel LP-4B is shown on E-104 but is not in the panel "
            "schedule."),
)

#: What issue 1 changed: (key, sheet, what, clouded and listed in the
#: sheet's revision block).
REVISIONS = (
    ("core-dimension", "A-401",
     "Grids B to C on the enlarged plans changed from 7200 to 7500.", True),
    ("door-hardware", "A-601",
     "Door 107's hardware set changed from HW-1 to HW-2.", True),
    ("bearing-pressure", "S-001",
     "The allowable bearing pressure in structural note 2 changed from 150 "
     "kPa to 200 kPa, with no cloud and no entry in the revision block.",
     False),
)

SUBMITTAL_PLANTED: Tuple[Planted, ...] = (
    Planted("water-cement", ("submittal",),
            "The mix's water-cementitious ratio is 0.48; Class C allows 0.45 "
            "at most."),
    Planted("fly-ash", ("submittal",),
            "Fly ash is 30 % of the cementitious material; the "
            "specification allows 25 % at most."),
    Planted("reactivity-report", ("submittal",),
            "The aggregate alkali-silica reactivity test report the "
            "specification requires (1.3 D) is missing from the submittal."),
    Planted("strength-record", ("submittal",),
            "The strength record holds 12 tests; the specification asks for "
            "30 consecutive tests or laboratory trial batches."),
)


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

@dataclass
class Sheet:
    number: str
    title: str
    kind: str
    floor: int = 0
    #: (kind, view number, the sheet it points to)
    callouts: List[Tuple[str, str, str]] = field(default_factory=list)
    #: view numbers drawn on this sheet
    views: List[str] = field(default_factory=list)
    #: tags drawn on this sheet, by group ("door", "footing", ...)
    tags: Dict[str, List[str]] = field(default_factory=dict)
    #: schedule rows on this sheet, by group; a row's first cell is its mark
    rows: Dict[str, List[Tuple[str, ...]]] = field(default_factory=dict)
    #: level name -> the elevation printed on this sheet
    levels: Dict[str, float] = field(default_factory=dict)
    #: bay ("B-C") -> the length printed on this sheet
    bays: Dict[str, int] = field(default_factory=dict)
    #: revision block rows: (number, date, description)
    revisions: List[Tuple[str, str, str]] = field(default_factory=list)


@dataclass
class Project:
    floors: int
    issue: int
    sheets: List[Sheet]
    #: the cover's sheet index: (number, title)
    index: List[Tuple[str, str]]

    def sheet(self, number: str) -> Optional[Sheet]:
        for s in self.sheets:
            if s.number == number:
                return s
        return None


def levels(floors: int) -> List[Tuple[str, float]]:
    """The building's levels, bottom up: (name, elevation in metres)."""
    out = [(f"{FLOOR_WORDS[k]} FLOOR", FIRST_FLOOR_EL + STOREY_M * k)
           for k in range(floors)]
    roof = FIRST_FLOOR_EL + STOREY_M * floors
    return out + [("ROOF", roof), ("T.O. PARAPET", roof + PARAPET_M)]


def _true_bays() -> Dict[str, int]:
    return {f"{a}-{b}": n for a, b, n in BAYS_X + BAYS_Y}


def doors_on_plan(floor: int) -> List[str]:
    if floor == 1:
        return [str(n) for n in range(101, 110)]
    if floor == 2:
        return ["201", "202", "203", "205", "206"]
    return [str(floor * 100 + n) for n in range(1, 9)]


def doors_scheduled(floor: int) -> List[str]:
    if floor == 1:
        return [str(n) for n in range(101, 109)]
    if floor == 2:
        return [str(n) for n in range(201, 207)]
    if floor == 4:
        return [str(n) for n in (401, 402, 403, 404, 405, 406, 408)]
    if floor == 5:
        return [str(n) for n in range(501, 510)]
    return [str(floor * 100 + n) for n in range(1, 9)]


def room_names(floor: int) -> Tuple[str, ...]:
    """A floor's six rooms: the north row west to east, then the south
    row."""
    if floor == 1:
        return ("OPEN OFFICE", "RECEPTION", "MEETING ROOM", "LOBBY", "CORE",
                "STORAGE")
    return ("OPEN OFFICE", "OFFICE", "MEETING ROOM", "OPEN OFFICE", "CORE",
            "COPY ROOM")


def _door_room(number: str) -> str:
    """The room a door opens into: its place on the plan decides (a door's
    last digit is its place, :func:`_door_slots`)."""
    floor, place = int(number) // 100, int(number) % 100
    rooms = room_names(floor)
    if place <= 7:
        return rooms[0 if place <= 2 else 1 if place <= 5 else 2]
    if place == 8:
        return rooms[3] if floor == 1 else rooms[5]
    return rooms[4]


#: Hardware set by the room a door opens into (the manual's Section
#: 08 71 00 lists the sets).
_HARDWARE = {"OPEN OFFICE": "HW-2", "OFFICE": "HW-2", "RECEPTION": "HW-1",
             "MEETING ROOM": "HW-1", "LOBBY": "HW-4", "CORE": "HW-4",
             "STORAGE": "HW-3", "COPY ROOM": "HW-3"}


def _door_row(number: str, issue: int) -> Tuple[str, ...]:
    room = _door_room(number)
    hardware = _HARDWARE[room]
    if number == "107" and issue >= 1:
        hardware = "HW-2"
    rating = {"CORE": "90 MIN", "STORAGE": "45 MIN"}.get(room, "-")
    return (number, room, "915", "2135", "A", "HM", rating, hardware)


def footing_rows(floors: int) -> Tuple[Tuple[str, ...], ...]:
    """The footing schedule (the five-storey building's are larger)."""
    if floors > 2:
        return (("F1", "1800 x 1800", "500", "10-15M EACH WAY"),
                ("F2", "2400 x 2400", "600", "12-20M EACH WAY"),
                ("F3", "3000 x 3000", "700", "14-20M EACH WAY"))
    return (("F1", "1500 x 1500", "400", "8-15M EACH WAY"),
            ("F2", "1800 x 1800", "450", "10-15M EACH WAY"),
            ("F3", "2200 x 2200", "500", "10-20M EACH WAY"))


def _footing_sizes(floors: int) -> Dict[str, int]:
    """Each mark's plan size in millimetres as drawn (F4 has no row)."""
    sizes = {row[0]: int(row[1].split()[0]) for row in footing_rows(floors)}
    sizes["F4"] = 2800 if floors > 2 else 2600
    return sizes


def _footing_at(i: int, j: int) -> str:
    """The footing mark at grid column ``i`` (0-3), row ``j`` (0-2)."""
    if (i, j) == (2, 1):
        return "F4"
    if (i, j) == (1, 1):
        return "F3"
    return "F1" if i in (0, 3) and j in (0, 2) else "F2"


def fcus_on_plan(floor: int) -> List[str]:
    return ([f"AHU-1"] if floor == 1 else []) + [
        f"FCU-{floor}-{n}" for n in range(1, 5)]


def panels_on_plan(floor: int) -> List[str]:
    return (["MDP"] if floor == 1 else []) + [f"LP-{floor}A", f"LP-{floor}B"]


def _project(floors: int = 2, issue: int = 1) -> Project:
    long = floors > 2
    bays = _true_bays()
    lv = dict(levels(floors))
    rev0 = [("0", ISSUE_DATES[0], "ISSUED FOR PERMIT")]
    sheets: List[Sheet] = []

    def add(number: str, title: str, kind: str, **kw) -> Sheet:
        s = Sheet(number, title, kind, revisions=list(rev0), **kw)
        sheets.append(s)
        return s

    cover = add("G-001", "COVER SHEET AND SHEET INDEX", "cover")
    if issue >= 1:
        cover.revisions.append(("1", ISSUE_DATES[1], "SHEET INDEX UPDATED"))
    add("G-002", "GENERAL NOTES", "general_notes")
    add("C-101", "SITE PLAN", "site",
        levels={"FIRST FLOOR": lv["FIRST FLOOR"]}, views=["1"])
    for k in range(1, floors + 1):
        s = add(f"A-10{k}", f"{FLOOR_WORDS[k - 1]} FLOOR PLAN", "plan",
                floor=k, views=["1"], bays=dict(bays),
                tags={"door": doors_on_plan(k)})
        s.callouts = [("section", "1", "A-301"), ("section", "2", "A-301")]
        if k == 1:
            s.callouts += [("plan", "1", "A-401"), ("detail", "3", "A-501"),
                           ("detail", "7", "A-501")]
        elif k == 2:
            s.callouts[1] = ("section", "2", "A-302")
            s.callouts += [("plan", "2", "A-401"), ("detail", "5", "A-501")]
        else:
            s.callouts += [("detail", {3: "3", 4: "5", 5: "3"}[k], "A-501")]
    if long:
        pairs = {1: ("1", "2"), 2: ("3", "4"), 3: ("5", "6"), 4: ("7", "9"),
                 5: ("8", "1")}
        for k in range(1, floors + 1):
            add(f"A-11{k}", f"{FLOOR_WORDS[k - 1]} FLOOR REFLECTED CEILING "
                "PLAN", "rcp", floor=k, views=["1"],
                callouts=[("detail", n, "A-502") for n in pairs[k]])
    add("A-201", "EXTERIOR ELEVATIONS", "elevations", views=["1", "2"],
        levels=dict(lv))
    wrong = dict(lv)
    wrong["SECOND FLOOR"] = 104.50
    if long:
        wrong["FIFTH FLOOR"] = 116.50
    add("A-301", "BUILDING SECTIONS", "sections", views=["1", "2"],
        levels=wrong, callouts=[("detail", "2", "A-501"),
                                ("detail", "4", "A-501"),
                                ("detail", "6", "A-501")])
    core = dict(bays)
    core_bays = {"B-C": 7500 if issue >= 1 else core["B-C"]}
    s = add("A-401", "ENLARGED PLANS", "enlarged", views=["1", "2"],
            bays=core_bays, tags={"door": ["109"]},
            callouts=[("detail", "1", "A-501")])
    if issue >= 1:
        s.revisions.append(("1", ISSUE_DATES[1], "REVISED CORE LAYOUT"))
    add("A-501", "DETAILS", "details", views=[str(n) for n in range(1, 7)])
    if long:
        add("A-502", "CEILING DETAILS", "details",
            views=[str(n) for n in range(1, 9)])
    first = range(1, 4) if long else range(1, floors + 1)
    s = add("A-601", "DOOR SCHEDULE", "door_schedule",
            rows={"door": [_door_row(d, issue) for k in first
                           for d in doors_scheduled(k)]})
    if issue >= 1:
        s.revisions.append(("1", ISSUE_DATES[1], "REVISED DOOR HARDWARE"))
    if long:
        add("A-602", "DOOR SCHEDULE (CONTINUED)", "door_schedule",
            rows={"door": [_door_row(d, issue) for k in (4, 5)
                           for d in doors_scheduled(k)]})
    add("S-001", "STRUCTURAL GENERAL NOTES", "structural_notes")
    add("S-101", "FOUNDATION PLAN", "foundation", views=["1"],
        bays=dict(bays),
        tags={"footing": sorted({_footing_at(i, j) for i in range(4)
                                 for j in range(3)})},
        callouts=[("detail", "1", "S-501"), ("detail", "2", "S-501"),
                  ("detail", "3", "S-501")])
    for k in range(2, floors + 2):
        roof = k == floors + 1
        name = "ROOF" if roof else f"{FLOOR_WORDS[k - 1]} FLOOR"
        add(f"S-20{k - 1}", f"{name} FRAMING PLAN", "framing", floor=k,
            views=["1"], bays=dict(bays), levels={name: lv[name]},
            callouts=[] if roof else [("detail", "4", "S-501")])
    add("S-501", "FOUNDATION DETAILS AND FOOTING SCHEDULE",
        "details_schedule", views=["1", "2", "3", "4"],
        rows={"footing": list(footing_rows(floors))})
    if long:
        for k in range(1, floors + 1):
            add(f"M-10{k}", f"{FLOOR_WORDS[k - 1]} FLOOR MECHANICAL PLAN",
                "mech", floor=k, views=["1"], tags={"fcu": fcus_on_plan(k)})
        marks = [m for k in range(1, floors + 1) for m in fcus_on_plan(k)
                 if m != "FCU-3-4"]
        add("M-601", "MECHANICAL EQUIPMENT SCHEDULE", "mech_schedule",
            rows={"fcu": [
                (m, "WHOLE BUILDING" if m == "AHU-1" else
                 f"{FLOOR_WORDS[int(m[4]) - 1]} FLOOR, ZONE {m[-1]}",
                 "4200" if m == "AHU-1" else str(280 + 20 * int(m[-1])),
                 "62.0" if m == "AHU-1" else f"{5.5 + int(m[-1]) * 0.5:.1f}",
                 "400 V / 3 PH" if m == "AHU-1" else "230 V / 1 PH")
                for m in marks]})
        for k in range(1, floors + 1):
            add(f"E-10{k}", f"{FLOOR_WORDS[k - 1]} FLOOR POWER PLAN",
                "power", floor=k, views=["1"],
                tags={"panel": panels_on_plan(k)})
        marks = [m for k in range(1, floors + 1) for m in panels_on_plan(k)
                 if m != "LP-4B"]
        add("E-601", "PANEL SCHEDULE", "panel_schedule",
            rows={"panel": [
                (m, "FIRST FLOOR CORE" if m == "MDP" else
                 f"{FLOOR_WORDS[int(m[3]) - 1]} FLOOR CORE",
                 "400 / 230 V", "800" if m == "MDP" else "100",
                 "UTILITY" if m == "MDP" else "MDP")
                for m in marks]})

    index = [(s.number, s.title) for s in sheets]
    at = [n for n, _t in index].index("A-301")
    index.insert(at, ("A-202", "INTERIOR ELEVATIONS"))
    index = [(n, "ROOF FRAMING PLAN" if n == "S-201" else t)
             for n, t in index]
    return Project(floors, issue, sheets, index)


# ---------------------------------------------------------------------------
# The problems, found again by rule
# ---------------------------------------------------------------------------

def audit(project: Project) -> List[str]:
    """Every disagreement of the kinds planted, found from the model by
    rule, as planted keys. The tests hold this equal to the planted list, so
    a change to the model that adds or removes a disagreement shows."""
    found: List[str] = []
    numbers = {s.number: s for s in project.sheets}
    for n, t in project.index:
        if n not in numbers:
            found.append("index-missing-sheet")
        elif numbers[n].title != t:
            found.append("index-title")
    for s in project.sheets:
        for _kind, view, target in s.callouts:
            if target not in numbers:
                found.append("callout-missing-sheet")
            elif view not in numbers[target].views:
                found.append("callout-missing-ceiling-detail"
                             if target == "A-502" else
                             "callout-missing-detail")
    for group, absent, unused in (
            ("door", "door-not-scheduled", "door-not-on-plan"),
            ("footing", "footing-not-scheduled", "footing-not-on-plan"),
            ("fcu", "fcu-not-scheduled", "fcu-not-on-plan"),
            ("panel", "panel-not-scheduled", "panel-not-on-plan")):
        drawn = {t for s in project.sheets for t in s.tags.get(group, [])}
        listed = {r[0] for s in project.sheets
                  for r in s.rows.get(group, [])}
        for t in sorted(drawn - listed):
            found.append(absent + ("-4" if t == "407" else ""))
        for t in sorted(listed - drawn):
            found.append(unused + ("-5" if t == "509" else ""))
    for name, _el in levels(project.floors):
        printed = {round(s.levels[name], 2) for s in project.sheets
                   if name in s.levels}
        if len(printed) > 1:
            found.append("level-" + name.lower().replace(" ", "-"))
    for bay in _true_bays():
        printed = {s.bays[bay] for s in project.sheets if bay in s.bays}
        if len(printed) > 1:
            found.append("bay-" + bay.lower())
    manual = {n for n, _t in MANUAL_SECTIONS}
    for n, _t in CITED_SECTIONS:
        if n not in manual:
            found.append("spec-section-missing")
    for use, spec, drawn in CONCRETE:
        if spec != drawn:
            found.append(use.lower().rstrip("s").replace(" ", "-")
                         + "-concrete")
    names = {"bearing": "bearing-pressure", "depth": "footing-depth"}
    for what, memo, issue0, issue1 in FOUNDATION:
        if memo != (issue1 if project.issue >= 1 else issue0):
            found.append(names.get(what, what))
    return sorted(found)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

W, H = 2448.0, 1584.0               # 34 x 22 in
MARGIN = 36.0
TB = 288.0                          # the title block's width
DX0, DY0 = MARGIN + 24.0, MARGIN + 24.0
DX1, DY1 = W - MARGIN - TB - 24.0, H - MARGIN - 24.0
BLACK = (0, 0, 0)


def _k(scale: int) -> float:
    """Paper points per real millimetre at 1:``scale``."""
    return 72.0 / 25.4 / scale


class _Canvas:
    """One sheet: lines and text in the order they are drawn."""

    def __init__(self, page):
        import pymupdf
        self._pm = pymupdf
        self.page = page
        self.shape = page.new_shape()
        self._texts: List[Tuple[float, float, str, float, str]] = []

    def line(self, a, b, width=0.6, dashes=None):
        self.shape.draw_line(a, b)
        self.shape.finish(width=width, color=BLACK, dashes=dashes,
                          closePath=False)

    def rect(self, x0, y0, x1, y1, width=0.6, dashes=None):
        self.shape.draw_rect(self._pm.Rect(x0, y0, x1, y1))
        self.shape.finish(width=width, color=BLACK, dashes=dashes)

    def circle(self, c, r, width=0.6, dashes=None, fill=None):
        self.shape.draw_circle(c, r)
        self.shape.finish(width=width, color=BLACK, dashes=dashes, fill=fill)

    def poly(self, points, width=0.6, fill=None, close=True):
        self.shape.draw_polyline(list(points))
        self.shape.finish(width=width, color=BLACK, fill=fill,
                          closePath=close)

    def text(self, x, y, s, size=8.0, bold=False, align="l"):
        font = "hebo" if bold else "helv"
        if align != "l":
            w = self._pm.get_text_length(str(s), fontname=font,
                                         fontsize=size)
            x -= w if align == "r" else w / 2.0
        self._texts.append((x, y, str(s), size, font))

    def bubble(self, cx, cy, top, bottom, r=15.0):
        """A reference bubble: the view number over the sheet number."""
        self.circle((cx, cy), r, width=0.9)
        self.line((cx - r, cy), (cx + r, cy))
        self.text(cx, cy - 3.0, top, size=9, bold=True, align="c")
        self.text(cx, cy + 8.5, bottom, size=6, align="c")

    def tag(self, cx, cy, s, r=11.0):
        """A hexagon tag (a door number)."""
        import math
        self.poly([(cx + r * math.cos(math.radians(a)),
                    cy + r * math.sin(math.radians(a)))
                   for a in range(0, 360, 60)], width=0.7)
        self.text(cx, cy + 2.5, s, size=7, bold=True, align="c")

    def view_title(self, x, y, number, sheet, title, scale):
        self.bubble(x + 15, y, number, sheet)
        self.text(x + 38, y - 2, title, size=12, bold=True)
        self.line((x + 36, y + 2), (x + 60 + 7.4 * len(title), y + 2),
                  width=1.1)
        self.text(x + 38, y + 12, f"SCALE {scale}", size=7)

    def cloud(self, x0, y0, x1, y1, number="1", r=7.0):
        """A revision cloud round a box, with its numbered triangle."""
        pts = []
        nx = max(2, int((x1 - x0) / (2 * r)))
        ny = max(2, int((y1 - y0) / (2 * r)))
        pts += [(x0 + (x1 - x0) * i / nx, y0, 0, -1) for i in range(nx)]
        pts += [(x1, y0 + (y1 - y0) * i / ny, 1, 0) for i in range(ny)]
        pts += [(x1 - (x1 - x0) * i / nx, y1, 0, 1) for i in range(nx)]
        pts += [(x0, y1 - (y1 - y0) * i / ny, -1, 0) for i in range(ny)]
        for i, (ax, ay, ox, oy) in enumerate(pts):
            bx, by = pts[(i + 1) % len(pts)][:2]
            mid = ((ax + bx) / 2 + ox * r, (ay + by) / 2 + oy * r)
            self.shape.draw_curve((ax, ay), mid, (bx, by))
            self.shape.finish(width=0.9, color=BLACK, closePath=False)
        tx, ty = x1 + 16, y0 - 8
        self.poly([(tx, ty - 11), (tx - 10, ty + 6), (tx + 10, ty + 6)],
                  width=0.9)
        self.text(tx, ty + 3.5, number, size=7, bold=True, align="c")

    def table(self, x, y, widths, header, rows, row_h=17.0, size=8.0,
              title=""):
        """A ruled table; returns ``{mark: (x0, y0, x1, y1)}`` for each
        row's last cell (for a revision cloud)."""
        if title:
            self.text(x, y - 8, title, size=12, bold=True)
        total = sum(widths)
        cells = {}
        for r, row in enumerate([tuple(header)] + [tuple(r) for r in rows]):
            top = y + r * row_h
            self.line((x, top), (x + total, top), width=0.9 if r < 2 else 0.4)
            cx = x
            for c, cell in enumerate(row):
                self.text(cx + 4, top + row_h - 5, cell, size=size,
                          bold=r == 0)
                cx += widths[c]
            cells[row[0]] = (cx - widths[-1], top, cx, top + row_h)
        bottom = y + (len(rows) + 1) * row_h
        self.line((x, bottom), (x + total, bottom), width=0.9)
        cx = x
        for w in list(widths) + [0]:
            self.line((cx, y), (cx, bottom), width=0.4)
            cx += w
        return cells

    def notes(self, x, y, title, lines, size=8.0, width=640.0, lead=1.45):
        """Numbered notes, wrapped to ``width``; returns the next y."""
        self.text(x, y, title, size=12, bold=True)
        y += 22
        for n, note in enumerate(lines, 1):
            first = True
            for piece in _wrap(self._pm, f"{n}. {note}", size, width):
                self.text(x if first else x + 14, y, piece, size=size)
                first = False
                y += size * lead
            y += size * 0.5
        return y

    def done(self):
        """Write the sheet. The text goes in last and in no helpful order
        (by a checksum of each piece), as a CAD plot's does: a bubble's
        view number and sheet number are not neighbours in the file, so
        pairing them takes their positions or a look at the page."""
        import zlib
        for x, y, s, size, font in sorted(
                self._texts, key=lambda t: zlib.crc32(
                    f"{t[2]}|{t[0]:.0f}|{t[1]:.0f}".encode())):
            self.shape.insert_text((x, y), s, fontsize=size, fontname=font)
        self.shape.commit()


def _wrap(pm, text: str, size: float, width: float,
          font: str = "helv") -> List[str]:
    out, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and pm.get_text_length(trial, fontname=font,
                                       fontsize=size) > width:
            out.append(line)
            line = word
        else:
            line = trial
    return out + ([line] if line else [])


def _frame(c: _Canvas, sheet: Sheet, n: int, total: int, issue: int) -> None:
    """The border and the title block down the right edge."""
    c.rect(MARGIN, MARGIN, W - MARGIN, H - MARGIN, width=1.6)
    x = W - MARGIN - TB
    c.line((x, MARGIN), (x, H - MARGIN), width=1.2)
    c.text(x + 14, MARGIN + 44, PROJECT, size=17, bold=True)
    c.text(x + 14, MARGIN + 64, f"PROJECT NO. {PROJECT_NO}", size=9)
    c.text(x + 14, MARGIN + 80,
           f"ISSUE {issue}  -  {ISSUE_DATES[min(issue, 1)]}", size=9)
    c.line((x, MARGIN + 100), (W - MARGIN, MARGIN + 100))
    c.text(x + 14, MARGIN + 124, "REVISIONS", size=9, bold=True)
    c.table(x + 14, MARGIN + 134, (30, 70, 160), ("NO.", "DATE",
                                                  "DESCRIPTION"),
            sheet.revisions, row_h=15, size=7)
    y = H - MARGIN - 250
    c.line((x, y), (W - MARGIN, y))
    c.text(x + 14, y + 22, "SHEET TITLE", size=7)
    for i, piece in enumerate(_wrap(c._pm, sheet.title, 13, TB - 28,
                                    font="hebo")):
        c.text(x + 14, y + 44 + 17 * i, piece, size=13, bold=True)
    y = H - MARGIN - 130
    c.line((x, y), (W - MARGIN, y))
    c.text(x + 14, y + 22, "SHEET NO.", size=7)
    c.text(x + 14, y + 78, sheet.number, size=44, bold=True)
    c.text(x + 14, y + 108, f"SHEET {n} OF {total}", size=8)


class _Plan:
    """A plan's grid on the sheet: where each grid line falls."""

    def __init__(self, x0: float, y0: float, scale: int):
        self.k = _k(scale)
        self.x0, self.y0 = x0, y0
        self.xs = [0.0]
        for _a, _b, n in BAYS_X:
            self.xs.append(self.xs[-1] + n)
        self.ys = [0.0]
        for _a, _b, n in BAYS_Y:
            self.ys.append(self.ys[-1] + n)

    def px(self, mm: float) -> float:
        return self.x0 + mm * self.k

    def py(self, mm: float) -> float:
        return self.y0 + mm * self.k

    @property
    def right(self) -> float:
        return self.px(self.xs[-1])

    @property
    def bottom(self) -> float:
        return self.py(self.ys[-1])


def _grid(c: _Canvas, p: _Plan, bays: Dict[str, int], dims: bool = True,
          outline: bool = True) -> None:
    """Grid lines and bubbles, the bay dimensions as PRINTED on this sheet
    (``bays``), and the building's outline."""
    top, left = p.y0, p.x0
    for i, letter in enumerate("ABCD"):
        x = p.px(p.xs[i])
        c.line((x, top - 48), (x, p.bottom + 24), width=0.35,
               dashes="[14 3 2 3] 0")
        c.circle((x, top - 62), 14, width=0.8)
        c.text(x, top - 57.5, letter, size=12, bold=True, align="c")
    for j, digit in enumerate("123"):
        y = p.py(p.ys[j])
        c.line((left - 48, y), (p.right + 24, y), width=0.35,
               dashes="[14 3 2 3] 0")
        c.circle((left - 62, y), 14, width=0.8)
        c.text(left - 62, y + 4.5, digit, size=12, bold=True, align="c")
    if dims:
        y = top - 22
        c.line((left, y), (p.right, y), width=0.4)
        for i, (a, b, _n) in enumerate(BAYS_X):
            xa, xb = p.px(p.xs[i]), p.px(p.xs[i + 1])
            for x in (xa, xb):
                c.line((x - 3, y + 3), (x + 3, y - 3), width=0.8)
            c.text((xa + xb) / 2, y - 3, str(bays[f"{a}-{b}"]), size=7,
                   align="c")
        x = left - 22
        c.line((x, top), (x, p.bottom), width=0.4)
        for j, (a, b, _n) in enumerate(BAYS_Y):
            ya, yb = p.py(p.ys[j]), p.py(p.ys[j + 1])
            for y in (ya, yb):
                c.line((x - 3, y + 3), (x + 3, y - 3), width=0.8)
            c.text(x - 4, (ya + yb) / 2 + 2.5, str(bays[f"{a}-{b}"]), size=7,
                   align="r")
    if outline:
        c.rect(left, top, p.right, p.bottom, width=1.6)
        c.rect(left + 200 * p.k, top + 200 * p.k, p.right - 200 * p.k,
               p.bottom - 200 * p.k, width=0.5)
    for i in range(4):
        for j in range(3):
            x, y = p.px(p.xs[i]), p.py(p.ys[j])
            c.poly([(x - 4, y - 4), (x + 4, y - 4), (x + 4, y + 4),
                    (x - 4, y + 4)], width=0.5, fill=BLACK)


def _callouts(c: _Canvas, p: _Plan, callouts) -> None:
    """A plan's callouts: section cuts across it, details off its east
    wall, the enlarged plan round the core."""
    walls = [1900.0, 4300.0, 7100.0, 9500.0]
    n_cut = n_det = 0
    for kind, view, target in callouts:
        if kind == "section" and n_cut == 0:
            # the first cut runs east-west (the long section), looking north
            n_cut += 1
            y = p.py(1500.0)
            c.line((p.x0 - 96, y), (p.right + 40, y), width=1.3,
                   dashes="[18 4 4 4] 0")
            c.bubble(p.x0 - 111, y, view, target)
            c.poly([(p.x0 - 118, y - 15), (p.x0 - 111, y - 27),
                    (p.x0 - 104, y - 15)], width=0.6, fill=BLACK)
        elif kind == "section":
            # the second runs north-south (the cross section), looking east
            x = p.x0 + (p.right - p.x0) * 0.86
            c.line((x, p.y0 - 96), (x, p.bottom + 40), width=1.3,
                   dashes="[18 4 4 4] 0")
            c.bubble(x, p.y0 - 111, view, target)
            c.poly([(x + 15, p.y0 - 118), (x + 27, p.y0 - 111),
                    (x + 15, p.y0 - 104)], width=0.6, fill=BLACK)
        elif kind == "detail":
            y = p.py(walls[n_det % len(walls)])
            n_det += 1
            c.circle((p.right, y), 12, width=0.8, dashes="[4 3] 0")
            c.line((p.right + 12, y), (p.right + 85, y))
            c.bubble(p.right + 100, y, view, target)
        else:
            xa, xb = p.px(p.xs[1]), p.px(p.xs[2])
            ya, yb = p.py(p.ys[1]), p.py(p.ys[2])
            c.rect(xa + 8, ya + 8, xb - 8, yb - 8, width=0.9,
                   dashes="[8 4] 0")
            mid = (xa + xb) / 2
            c.line((mid, yb - 8), (mid, p.bottom + 60))
            c.bubble(mid, p.bottom + 75, view, target)
            c.text(mid + 22, p.bottom + 78, "ENLARGED PLAN", size=7)


def _door_slots(floor: int):
    """Where a floor's doors go, by the door's last digit, in millimetres
    from grid A/1: (x, y, the wall runs east-west). Seven on the partition
    along grid 2, opening north; the eighth is the entrance in the south
    wall on the first floor and the door into the south-east room above it;
    the ninth opens from grid B into the core."""
    north = [(1500, 5400, True), (4200, 5400, True), (7400, 5400, True),
             (9600, 5400, True), (11800, 5400, True), (15000, 5400, True),
             (17800, 5400, True)]
    eighth = (3000, 10800, True) if floor == 1 else (13200, 8100, False)
    return north + [eighth, (6000, 8100, False)]


def _rooms(c: _Canvas, p: _Plan, floor: int) -> None:
    names = room_names(floor)
    order = ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1))
    for name, (i, j) in zip(names, order):
        c.text((p.px(p.xs[i]) + p.px(p.xs[i + 1])) / 2,
               (p.py(p.ys[j]) + p.py(p.ys[j + 1])) / 2 + 26, name, size=9,
               align="c")
    for x in (p.xs[1], p.xs[2]):
        c.line((p.px(x), p.y0), (p.px(x), p.bottom), width=1.0)
    c.line((p.x0, p.py(p.ys[1])), (p.right, p.py(p.ys[1])), width=1.0)


def _doors(c: _Canvas, p: _Plan, doors: Sequence[str]) -> None:
    for number in doors:
        x, y, along = _door_slots(int(number) // 100)[int(number) % 100 - 1]
        px, py = p.px(x), p.py(y)
        leaf = 915 * p.k
        if along:
            c.line((px, py), (px, py - leaf), width=0.8)
            c.shape.draw_curve((px, py - leaf), (px + leaf, py - leaf),
                               (px + leaf, py))
            c.shape.finish(width=0.4, color=BLACK, closePath=False)
            c.tag(px + leaf / 2, py - leaf - 16, number)
        else:
            c.line((px, py), (px + leaf, py), width=0.8)
            c.shape.draw_curve((px + leaf, py), (px + leaf, py + leaf),
                               (px, py + leaf))
            c.shape.finish(width=0.4, color=BLACK, closePath=False)
            c.tag(px + leaf + 18, py + leaf / 2, number)


def _level_marks(c: _Canvas, x0: float, x1: float, base_y: float,
                 k: float, printed: Dict[str, float], floors: int) -> None:
    """Level lines across an elevation or a section, each with the level's
    name and the elevation PRINTED on this sheet."""
    for name, true in levels(floors):
        y = base_y - (true - FIRST_FLOOR_EL) * 1000 * k
        c.line((x0 - 20, y), (x1 + 70, y), width=0.4, dashes="[10 4] 0")
        c.circle((x1 + 78, y), 5, width=0.6)
        c.line((x1 + 70, y), (x1 + 86, y), width=0.6)
        c.text(x1 + 90, y - 3, name, size=8, bold=True)
        c.text(x1 + 90, y + 8, f"EL. {printed[name]:.2f}", size=8)


def _sketch(c: _Canvas, x: float, y: float, w: float, h: float,
            notes: Sequence[str]) -> None:
    """A detail's stand-in: two hatched pieces and three leader notes."""
    c.rect(x + w * 0.12, y + h * 0.18, x + w * 0.42, y + h * 0.78, width=1.2)
    c.rect(x + w * 0.42, y + h * 0.42, x + w * 0.60, y + h * 0.78, width=0.9)
    for i in range(12):
        t = y + h * 0.18 + i * h * 0.05
        c.line((x + w * 0.12, t + h * 0.05), (x + w * 0.20, t), width=0.25)
    for i, note in enumerate(notes):
        ny = y + h * (0.24 + 0.2 * i)
        c.line((x + w * 0.42, ny), (x + w * 0.66, ny), width=0.35)
        c.text(x + w * 0.67, ny + 3, note, size=7)


_DETAIL_TITLES = {
    # each is called where it belongs: 1 from the core's wall (A-401), 2,
    # 4 and 6 from the section (A-301: parapet, second floor, grade), 3 and
    # 5 from the plans' exterior wall
    "A-501": ("CORE WALL AT FLOOR SLAB", "PARAPET AT ROOF EDGE",
              "WINDOW HEAD AT EXTERIOR WALL", "SLAB EDGE AT EXTERIOR WALL",
              "WINDOW JAMB AT EXTERIOR WALL",
              "BASE OF EXTERIOR WALL AT GRADE"),
    # all called from the ceiling plans' exterior wall
    "A-502": ("CEILING AT EXTERIOR WALL", "CEILING AT WINDOW HEAD",
              "BULKHEAD AT EXTERIOR WALL", "CEILING AT BLIND POCKET",
              "CEILING EDGE TRIM", "CEILING AT EXPANSION JOINT",
              "PERIMETER HANGER AND BRACING",
              "CEILING AT COLUMN ENCLOSURE"),
    "S-501": ("TYPICAL SPREAD FOOTING", "FOOTING AT COLUMN",
              "SLAB ON GRADE AT EXTERIOR WALL", "SLAB EDGE AT STEEL BEAM"),
}
_DETAIL_NOTES = ("SEALANT AND BACKER ROD", "CONTINUOUS FLASHING",
                 "RIGID INSULATION", "GYPSUM BOARD ON METAL STUDS",
                 "SHIM AND ANCHOR AT 600 O.C.", "VAPOUR RETARDER",
                 "GROUT FRAME SOLID", "FINISH AS SCHEDULED",
                 "COMPACTED GRANULAR BASE")


def _details(c: _Canvas, sheet: Sheet, x0: float, x1: float,
             cols: int) -> None:
    titles = _DETAIL_TITLES[sheet.number]
    rows = (len(titles) + cols - 1) // cols
    w, h = (x1 - x0) / cols, (DY1 - DY0) / rows
    for i, title in enumerate(titles):
        x, y = x0 + (i % cols) * w, DY0 + (i // cols) * h
        c.rect(x + 6, y + 6, x + w - 6, y + h - 6, width=0.4)
        _sketch(c, x, y, w, h - 60,
                [_DETAIL_NOTES[(i * 3 + j + len(sheet.number)) %
                               len(_DETAIL_NOTES)] for j in range(3)])
        c.view_title(x + 24, y + h - 40, str(i + 1), sheet.number, title,
                     "1:5" if sheet.number != "S-501" else "1:20")


_GENERAL_NOTES = (
    "THESE DRAWINGS ARE TO BE READ WITH THE PROJECT MANUAL. WHERE THE "
    "DRAWINGS AND THE PROJECT MANUAL CONFLICT, ASK THE ARCHITECT BEFORE "
    "PROCEEDING.",
    "DO NOT SCALE THE DRAWINGS. WRITTEN DIMENSIONS GOVERN. DIMENSIONS ARE IN "
    "MILLIMETRES AND LEVELS ARE IN METRES UNLESS NOTED OTHERWISE.",
    "THE PROJECT DATUM IS THE FIRST FLOOR FINISHED FLOOR, EL. 100.00.",
)
_ABBREVIATIONS = ("A.F.F.  ABOVE FINISHED FLOOR", "EL.  ELEVATION",
                  "HM  HOLLOW METAL", "O.C.  ON CENTRE",
                  "T.O.  TOP OF", "U.N.O.  UNLESS NOTED OTHERWISE")


def _structural_notes(issue: int) -> List[str]:
    f = {what: (i1 if issue >= 1 else i0)
         for what, _memo, i0, i1 in FOUNDATION}
    strengths = "; ".join(f"{use} {drawn} MPa" for use, _s, drawn in CONCRETE)
    return [
        "FOUNDATION DESIGN IS BASED ON THE GEOTECHNICAL RECOMMENDATIONS "
        f"MEMORANDUM, REPORT NO. {MEMO_NO}.",
        f"ALLOWABLE NET SOIL BEARING PRESSURE: {f['bearing']}.",
        "BOTTOM OF EXTERIOR FOOTINGS: NOT LESS THAN "
        f"{f['depth']} BELOW FINISHED GRADE.",
        f"SEISMIC SITE CLASS: {f['site_class']}.",
        f"DESIGN GROUNDWATER LEVEL: {f['groundwater']}.",
        f"CONCRETE 28-DAY COMPRESSIVE STRENGTH: {strengths}.",
        "REINFORCING STEEL: GRADE 420, DEFORMED BARS.",
        "CONCRETE COVER TO REINFORCING: FOOTINGS 75 mm; SLABS ON GRADE 40 "
        "mm; COLUMNS 40 mm.",
        "SLAB ON GRADE: 150 mm ON 150 mm COMPACTED GRANULAR BASE; MODULUS OF "
        f"SUBGRADE REACTION {f['subgrade']}.",
        "STRUCTURAL STEEL: GRADE 350W. BOLTS: GRADE 8.8.",
        "REMOVE ALL FILL BELOW FOOTINGS AND SLABS ON GRADE. THE GEOTECHNICAL "
        "ENGINEER IS TO INSPECT EVERY FOOTING SUBGRADE BEFORE CONCRETE IS "
        "PLACED.",
        "EXCAVATIONS DEEPER THAN 1.2 m ARE TO BE SLOPED OR SHORED.",
    ]


def _draw(c: _Canvas, project: Project, sheet: Sheet) -> None:
    kind, floors, issue = sheet.kind, project.floors, project.issue
    if kind == "cover":
        c.text(DX0 + 40, DY0 + 110, PROJECT, size=60, bold=True)
        c.text(DX0 + 44, DY0 + 150, f"PROJECT NO. {PROJECT_NO}", size=16)
        c.text(DX0 + 44, DY0 + 176, f"DRAWING SET  -  ISSUE {issue}  -  "
               f"{ISSUE_DATES[min(issue, 1)]}", size=16)
        latest = {s.number: s.revisions[-1][0] for s in project.sheets}
        c.table(DX0 + 44, DY0 + 250, (90, 430, 50),
                ("SHEET", "TITLE", "REV."),
                [(n, t, latest.get(n, "0")) for n, t in project.index],
                row_h=18 if floors <= 2 else 16.5, size=9,
                title="SHEET INDEX")
        c.notes(DX0 + 760, DY0 + 250, "PROJECT DATA", [
            f"BUILDING: {floors} STOREYS, STEEL FRAME ON SPREAD FOOTINGS.",
            "GROSS FLOOR AREA: "
            f"{19.2 * 10.8 * floors:.0f} m2.",
            "READ THESE DRAWINGS WITH THE PROJECT MANUAL AND THE "
            f"GEOTECHNICAL RECOMMENDATIONS MEMORANDUM, REPORT NO. {MEMO_NO}.",
        ], size=9, width=560)
    elif kind == "general_notes":
        lines = list(_GENERAL_NOTES)
        for n, t in CITED_SECTIONS[:-1]:
            lines.append(f"{t}: SEE SECTION {n}.")
        lines.append(f"SUBMITTALS: SEE SECTION {CITED_SECTIONS[-1][0]}.")
        c.notes(DX0 + 40, DY0 + 60, "GENERAL NOTES", lines)
        c.notes(DX0 + 800, DY0 + 60, "ABBREVIATIONS", _ABBREVIATIONS)
    elif kind == "structural_notes":
        c.notes(DX0 + 40, DY0 + 60, "STRUCTURAL GENERAL NOTES",
                _structural_notes(issue))
    elif kind == "site":
        k = _k(200)
        x, y = DX0 + 420, DY0 + 300
        c.rect(x - 300, y - 180, x + 19200 * k + 420, y + 10800 * k + 300,
               width=0.8, dashes="[20 5 3 5] 0")
        c.text(x - 290, y - 188, "PROPERTY LINE", size=7)
        c.rect(x, y, x + 19200 * k, y + 10800 * k, width=1.6)
        c.text(x + 19200 * k / 2, y + 10800 * k / 2 - 6, "PROPOSED BUILDING",
               size=9, bold=True, align="c")
        c.text(x + 19200 * k / 2, y + 10800 * k / 2 + 8,
               f"FINISHED FLOOR EL. {sheet.levels['FIRST FLOOR']:.2f}",
               size=8, align="c")
        for name, (bx, by) in (("B-1", (-90, -70)), ("B-2", (330, -80)),
                               ("B-3", (-70, 230)), ("B-4", (350, 250))):
            c.circle((x + bx, y + by), 4, width=0.8)
            c.line((x + bx - 6, y + by), (x + bx + 6, y + by), width=0.4)
            c.line((x + bx, y + by - 6), (x + bx, y + by + 6), width=0.4)
            c.text(x + bx + 9, y + by + 3, name, size=7, bold=True)
        c.poly([(DX1 - 120, DY0 + 150), (DX1 - 132, DY0 + 190),
                (DX1 - 108, DY0 + 190)], width=0.8, fill=BLACK)
        c.text(DX1 - 120, DY0 + 140, "N", size=12, bold=True, align="c")
        c.notes(DX0 + 40, DY1 - 220, "SITE NOTES", [
            "BORING LOCATIONS B-1 TO B-4 ARE FROM THE GEOTECHNICAL "
            f"RECOMMENDATIONS MEMORANDUM, REPORT NO. {MEMO_NO}.",
            "FINISHED GRADE SLOPES AWAY FROM THE BUILDING AT 2 % FOR 3 m."])
        c.view_title(DX0 + 40, DY1 - 40, "1", sheet.number, "SITE PLAN",
                     "1:200")
    elif kind in ("plan", "rcp", "foundation", "framing", "mech", "power"):
        p = _Plan(DX0 + 190, DY0 + 230, 50)
        _grid(c, p, sheet.bays or _true_bays(), dims=bool(sheet.bays))
        if kind == "plan":
            _rooms(c, p, sheet.floor)
            _doors(c, p, sheet.tags["door"])
        elif kind == "rcp":
            x = 1200.0
            while x < p.xs[-1]:
                c.line((p.px(x), p.y0), (p.px(x), p.bottom), width=0.2)
                x += 1200.0
            y = 1200.0
            while y < p.ys[-1]:
                c.line((p.x0, p.py(y)), (p.right, p.py(y)), width=0.2)
                y += 1200.0
            c.text(p.px(9600), p.py(2700), "ACOUSTIC TILE CEILING AT 2700 "
                   "A.F.F.", size=9, align="c")
        elif kind == "foundation":
            size = _footing_sizes(floors)
            for i in range(4):
                for j in range(3):
                    mark = _footing_at(i, j)
                    half = size[mark] * p.k / 2
                    x, y = p.px(p.xs[i]), p.py(p.ys[j])
                    c.rect(x - half, y - half, x + half, y + half, width=0.8,
                           dashes="[6 3] 0")
                    c.text(x + half + 5, y + half + 9, mark, size=8,
                           bold=True)
            c.text(p.px(9600), p.bottom + 150, "TOP OF FOOTING EL. 98.00 "
                   "U.N.O. SEE FOOTING SCHEDULE ON S-501.", size=8,
                   align="c")
        elif kind == "framing":
            for j in range(3):
                c.line((p.x0, p.py(p.ys[j])), (p.right, p.py(p.ys[j])),
                       width=2.0)
            for i in range(4):
                c.line((p.px(p.xs[i]), p.y0), (p.px(p.xs[i]), p.bottom),
                       width=2.0)
            for i in range(3):
                c.text((p.px(p.xs[i]) + p.px(p.xs[i + 1])) / 2,
                       p.py(p.ys[1]) - 6, "W410x60", size=7, align="c")
            (name, el), = sheet.levels.items()
            c.text(p.px(9600), p.py(2700),
                   f"{name}  -  T.O. SLAB EL. {el:.2f}", size=9, bold=True,
                   align="c")
            c.text(p.px(9600), p.py(2700) + 14,
                   "150 mm CONCRETE SLAB ON METAL DECK", size=8, align="c")
        elif kind == "mech":
            # four fan coil units in the rooms; the air handler, last, in
            # the core
            slots = ((3000, 2700), (9600, 2700), (16200, 2700),
                     (3000, 8100), (9600, 8100))
            marks = sheet.tags["fcu"]
            if "AHU-1" in marks:
                marks = marks[1:] + marks[:1]
            for mark, (x, y) in zip(marks, slots):
                c.rect(p.px(x) - 26, p.py(y) - 14, p.px(x) + 26,
                       p.py(y) + 14, width=0.9)
                c.text(p.px(x), p.py(y) + 30, mark, size=7, bold=True,
                       align="c")
                c.line((p.px(x) + 26, p.py(y)), (p.px(x) + 120, p.py(y)),
                       width=0.5)
        elif kind == "power":
            slots = ((9000, 6200), (10200, 6200), (9600, 7600))
            for mark, (x, y) in zip(sheet.tags["panel"], slots):
                c.rect(p.px(x) - 14, p.py(y) - 8, p.px(x) + 14, p.py(y) + 8,
                       width=1.2)
                c.text(p.px(x), p.py(y) + 22, mark, size=7, bold=True,
                       align="c")
        _callouts(c, p, sheet.callouts)
        c.view_title(DX0 + 40, DY1 - 40, "1", sheet.number, sheet.title,
                     "1:50")
    elif kind in ("elevations", "sections"):
        k = _k(100)
        base = DY0 + 150 + (levels(floors)[-1][1] - FIRST_FLOOR_EL) * 1000 * k
        names = (("SOUTH ELEVATION", "EAST ELEVATION")
                 if kind == "elevations" else
                 ("BUILDING SECTION, EAST-WEST",
                  "BUILDING SECTION, NORTH-SOUTH"))
        x = DX0 + 120
        for view, (name, width) in enumerate(zip(names, (19200, 10800)), 1):
            x1 = x + width * k
            top = base - (levels(floors)[-1][1] - FIRST_FLOOR_EL) * 1000 * k
            c.rect(x, top, x1, base, width=1.5)
            c.line((x - 60, base + 4), (x1 + 60, base + 4), width=2.0)
            if kind == "sections":
                for _n, el in levels(floors)[1:-1]:
                    y = base - (el - FIRST_FLOOR_EL) * 1000 * k
                    c.rect(x, y, x1, y + 6, width=0.8)
            else:
                for f in range(floors):
                    y = base - (f * STOREY_M + 2.6) * 1000 * k
                    for i in range(int(width / 2400)):
                        wx = x + (1200 + i * 2400 - 600) * k
                        c.rect(wx, y, wx + 1200 * k, y + 1500 * k, width=0.4)
            _level_marks(c, x, x1, base, k, sheet.levels, floors)
            c.view_title(x, base + 70, str(view), sheet.number, name, "1:100")
            x = x1 + 420
        # the first view's west wall, top down: the parapet, the second
        # floor's slab edge, the base of the wall
        wall = DX0 + 120
        spots = (base - (levels(floors)[-1][1] - FIRST_FLOOR_EL) * 1000 * k,
                 base - STOREY_M * 1000 * k, base)
        for (_kind, view, target), y in zip(sheet.callouts, spots):
            c.circle((wall, y), 12, width=0.8, dashes="[4 3] 0")
            c.line((wall - 10, y - 7), (wall - 52, y - 30))
            c.bubble(wall - 66, y - 38, view, target)
    elif kind == "enlarged":
        printed = sheet.bays["B-C"]
        for view in (1, 2):
            x0 = DX0 + 120 + (view - 1) * 1000
            y0 = DY0 + 260
            k = _k(25)
            xb, xc = x0, x0 + 7200 * k
            yb = y0 + 5400 * k
            for x, letter in ((xb, "B"), (xc, "C")):
                c.line((x, y0 - 48), (x, yb + 24), width=0.35,
                       dashes="[14 3 2 3] 0")
                c.circle((x, y0 - 62), 14, width=0.8)
                c.text(x, y0 - 57.5, letter, size=12, bold=True, align="c")
            c.rect(xb, y0, xc, yb, width=1.6)
            c.rect(xb + 300 * k, y0 + 300 * k, xb + 3000 * k, yb - 300 * k,
                   width=0.8)
            for i in range(11):
                sx = xb + (450 + i * 230) * k
                c.line((sx, y0 + 300 * k), (sx, y0 + 2400 * k), width=0.3)
            c.text(xb + 1650 * k, yb - 500 * k, "STAIR 1", size=9, align="c")
            c.rect(xb + 3600 * k, y0 + 300 * k, xc - 300 * k, yb - 300 * k,
                   width=0.8)
            c.text(xb + 5250 * k, yb - 500 * k, "TOILET", size=9, align="c")
            if view == 1:
                leaf = 915 * k
                dy = y0 + 2700 * k
                c.line((xb, dy), (xb + leaf, dy), width=0.8)
                c.shape.draw_curve((xb + leaf, dy), (xb + leaf, dy + leaf),
                                   (xb, dy + leaf))
                c.shape.finish(width=0.4, color=BLACK, closePath=False)
                c.tag(xb - 22, dy + leaf / 2, sheet.tags["door"][0])
            y = y0 - 22
            c.line((xb, y), (xc, y), width=0.4)
            for x in (xb, xc):
                c.line((x - 3, y + 3), (x + 3, y - 3), width=0.8)
            c.text((xb + xc) / 2, y - 3, str(printed), size=7, align="c")
            if project.issue >= 1:
                c.cloud((xb + xc) / 2 - 30, y - 16, (xb + xc) / 2 + 30,
                        y + 5)
            if view == 1:
                for _kind, v, target in sheet.callouts:
                    c.circle((xc, y0 + 2700 * k), 12, width=0.8,
                             dashes="[4 3] 0")
                    c.line((xc + 12, y0 + 2700 * k),
                           (xc + 70, y0 + 2700 * k))
                    c.bubble(xc + 85, y0 + 2700 * k, v, target)
            c.view_title(x0, yb + 80, str(view), sheet.number,
                         "ENLARGED CORE PLAN, "
                         f"{FLOOR_WORDS[view - 1]} FLOOR", "1:25")
    elif kind == "details":
        _details(c, sheet, DX0, DX1, 3 if len(sheet.views) <= 6 else 4)
    elif kind == "details_schedule":
        split = DX0 + (DX1 - DX0) * 0.62
        _details(c, sheet, DX0, split, 2)
        c.table(split + 40, DY0 + 90, (70, 130, 110, 190),
                ("MARK", "SIZE (mm)", "THICKNESS (mm)", "REINFORCING"),
                sheet.rows["footing"], title="FOOTING SCHEDULE")
    elif kind == "door_schedule":
        cells = c.table(
            DX0 + 60, DY0 + 90, (60, 170, 80, 80, 60, 70, 100, 110),
            ("NO.", "ROOM", "WIDTH", "HEIGHT", "TYPE", "FRAME",
             "FIRE RATING", "HARDWARE SET"), sheet.rows["door"],
            title=sheet.title)
        if issue >= 1 and "107" in cells:
            x0, y0, x1, y1 = cells["107"]
            c.cloud(x0 + 2, y0 + 1, x1 - 40, y1 - 1, r=5)
        c.notes(DX0 + 900, DY0 + 90, "DOOR NOTES", [
            "DOOR TYPE A: SINGLE, FLUSH.",
            "FRAMES: HM, HOLLOW METAL, SEE SECTION 08 11 13.",
            "HARDWARE SETS: SEE SECTION 08 71 00."], width=520)
    elif kind == "mech_schedule":
        c.table(DX0 + 60, DY0 + 90, (90, 250, 110, 110, 130),
                ("MARK", "SERVES", "AIRFLOW (L/s)", "COOLING (kW)",
                 "POWER"), sheet.rows["fcu"], title=sheet.title)
    elif kind == "panel_schedule":
        c.table(DX0 + 60, DY0 + 90, (80, 280, 110, 90, 110),
                ("MARK", "LOCATION", "VOLTAGE", "MAIN (A)", "FED FROM"),
                sheet.rows["panel"], title=sheet.title)
    else:                                                 # pragma: no cover
        raise ValueError(f"unknown sheet kind {kind!r}")


@dataclass(frozen=True)
class DrawingSet:
    """A built set: the PDF, its sheets in page order, and what is planted."""
    pdf: bytes
    floors: int
    issue: int
    sheets: Tuple[Tuple[str, str], ...]
    planted: Tuple[Planted, ...]

    def page_of(self, number: str) -> int:
        """The 1-based PDF page of sheet ``number``."""
        return [n for n, _t in self.sheets].index(number) + 1


@lru_cache(maxsize=None)
def build_drawing_set(floors: int = 2, issue: int = 1) -> DrawingSet:
    """The drawing set as one PDF (``floors=2``: 15 sheets; ``floors=5``: 40
    sheets), at issue 0 or issue 1."""
    import pymupdf
    project = _project(floors, issue)
    doc = pymupdf.open()
    try:
        for n, sheet in enumerate(project.sheets, 1):
            c = _Canvas(doc.new_page(width=W, height=H))
            _frame(c, sheet, n, len(project.sheets), issue)
            _draw(c, project, sheet)
            c.done()
        doc.set_metadata({"title": f"{PROJECT} drawings, issue {issue}"})
        pdf = doc.tobytes(garbage=3, deflate=True, no_new_id=True)
    finally:
        doc.close()
    planted = LONG_PLANTED if floors > 2 else BASE_PLANTED
    if issue < 1:
        planted = tuple(p for p in planted
                        if p.key not in ("bay-b-c", "bearing-pressure"))
    return DrawingSet(pdf, floors, issue,
                      tuple((s.number, s.title) for s in project.sheets),
                      planted)


# ---------------------------------------------------------------------------
# The text documents
# ---------------------------------------------------------------------------

class _Paper:
    """A letter-size text document: headings, paragraphs and ruled tables."""

    LEFT, RIGHT, TOP, BOTTOM = 72.0, 540.0, 84.0, 716.0

    def __init__(self, running_head: str):
        import pymupdf
        self._pm = pymupdf
        self.doc = pymupdf.open()
        self.head = running_head
        self.page = None
        self.y = 0.0
        self.new_page()

    def new_page(self):
        self.page = self.doc.new_page(width=612, height=792)
        n = len(self.doc)
        self.page.insert_text((self.LEFT, 52), self.head, fontsize=8,
                              fontname="helv")
        self.page.insert_text((self.RIGHT - 40, 752), f"Page {n}",
                              fontsize=8, fontname="helv")
        self.y = self.TOP

    def _room(self, need: float):
        if self.y + need > self.BOTTOM:
            self.new_page()

    def h(self, text: str, size: float = 13.0):
        self._room(size * 3)
        self.y += size * 0.6
        self.page.insert_text((self.LEFT, self.y), text, fontsize=size,
                              fontname="hebo")
        self.y += size * 1.2

    def p(self, text: str, size: float = 10.0, indent: float = 0.0):
        for piece in _wrap(self._pm, text, size,
                           self.RIGHT - self.LEFT - indent):
            self._room(size * 1.4)
            self.page.insert_text((self.LEFT + indent, self.y), piece,
                                  fontsize=size, fontname="helv")
            self.y += size * 1.4
        self.y += size * 0.5

    def table(self, widths, rows, size: float = 9.0):
        row_h = size + 8
        self._room(row_h * (len(rows) + 1))
        total = sum(widths)
        for r, row in enumerate(rows):
            top = self.y + r * row_h
            self.page.draw_line((self.LEFT, top), (self.LEFT + total, top),
                                width=0.5)
            x = self.LEFT
            for cell, w in zip(row, widths):
                self.page.insert_text((x + 4, top + row_h - 5), str(cell),
                                      fontsize=size,
                                      fontname="hebo" if r == 0 else "helv")
                x += w
        bottom = self.y + len(rows) * row_h
        self.page.draw_line((self.LEFT, bottom), (self.LEFT + total, bottom),
                            width=0.5)
        x = self.LEFT
        for w in list(widths) + [0]:
            self.page.draw_line((x, self.y), (x, bottom), width=0.5)
            x += w
        self.y = bottom + 14

    def bytes(self, title: str) -> bytes:
        try:
            self.doc.set_metadata({"title": title})
            return self.doc.tobytes(garbage=3, deflate=True, no_new_id=True)
        finally:
            self.doc.close()


@lru_cache(maxsize=None)
def build_project_manual() -> bytes:
    """The project manual: a cover, the table of contents, and Sections
    01 33 00, 03 30 00 and 08 71 00."""
    d = _Paper(f"{PROJECT}  -  PROJECT MANUAL  -  ISSUE 1")
    d.y = 240
    d.h("PROJECT MANUAL", 26)
    d.h(PROJECT, 18)
    d.p(f"Project No. {PROJECT_NO}")
    d.p(f"Issue 1, {ISSUE_DATES[1]}")
    d.new_page()
    d.h("TABLE OF CONTENTS", 15)
    d.table((90, 330), [("SECTION", "TITLE")] + [
        (n, t) for n, t in MANUAL_SECTIONS])
    d.p("Sections not listed here are not part of this project manual. "
        "This volume binds Sections 01 33 00, 03 30 00 and 08 71 00 in "
        "full; the other sections listed are issued separately. Mechanical "
        "and electrical work is specified in Volume 2.")
    d.new_page()
    d.h("SECTION 01 33 00  -  SUBMITTAL PROCEDURES", 13)
    d.h("PART 1  GENERAL", 11)
    d.p("1.1  Submit each item the technical sections ask for, complete, "
        "under one transmittal that names the specification section.")
    d.p("1.2  The Contractor reviews each submittal before sending it and "
        "marks every deviation from the contract documents on the "
        "transmittal. A submittal with unmarked deviations will be "
        "returned.")
    d.p("1.3  Allow 10 working days for review.")
    d.new_page()
    d.h("SECTION 03 30 00  -  CAST-IN-PLACE CONCRETE", 13)
    d.h("PART 1  GENERAL", 11)
    d.p("1.1  SUMMARY. Cast-in-place concrete for footings, slabs on "
        "grade, columns and suspended slabs.")
    d.p("1.2  REFERENCES. Test methods are those named in this section.")
    d.p("1.3  SUBMITTALS. Submit, for each class of concrete:")
    for line in (
            "A.  Mix design, with the proportions of every material and the "
            "water-cementitious ratio.",
            "B.  Strength record: not fewer than 30 consecutive strength "
            "tests of the proposed mix, or the results of laboratory trial "
            "batches.",
            "C.  Cement mill certificate.",
            "D.  Aggregate test reports: grading, and alkali-silica "
            "reactivity by the accelerated mortar-bar method, not more than "
            "12 months old.",
            "E.  Product data for each admixture."):
        d.p(line, indent=18)
    d.h("PART 2  PRODUCTS", 11)
    d.p("2.1  MATERIALS. Portland cement, Type GU. Fly ash: not more than "
        "25 percent of the cementitious material by mass. Coarse aggregate: "
        "20 mm nominal maximum size.")
    d.p("2.2  MIXES. Proportion each class to the following:")
    d.table((70, 190, 110, 98), [
        ("CLASS", "USE", "28-DAY STRENGTH", "MAX. w/cm"),
        ("A", "Footings", f"{CONCRETE[0][1]} MPa", "0.50"),
        ("B", "Slabs on grade", f"{CONCRETE[1][1]} MPa", "0.45"),
        ("C", "Columns and suspended slabs", f"{CONCRETE[2][1]} MPa",
         "0.45")])
    d.p("2.3  SLUMP. 100 mm, plus or minus 25 mm, at the point of "
        "discharge.")
    d.p("2.4  AIR CONTENT. Concrete exposed to freezing and thawing: 5 to "
        "8 percent. Interior concrete: not air entrained.")
    d.h("PART 3  EXECUTION", 11)
    d.p("3.1  Place concrete within 90 minutes of batching.")
    d.p("3.2  Cure for not less than 7 days.")
    d.new_page()
    d.h("SECTION 08 71 00  -  DOOR HARDWARE", 13)
    d.h("PART 2  PRODUCTS", 11)
    d.p("2.1  HARDWARE SETS. The door schedule on the drawings names the "
        "set for each door.")
    d.table((70, 398), [
        ("SET", "CONTENTS"),
        ("HW-1", "Hinges, passage latch, wall stop."),
        ("HW-2", "Hinges, office lock, wall stop."),
        ("HW-3", "Hinges, storeroom lock, closer, kick plate."),
        ("HW-4", "Hinges, exit device, closer, kick plate, smoke seals.")])
    return d.bytes(f"{PROJECT} project manual")


@lru_cache(maxsize=None)
def build_geotechnical_memo() -> bytes:
    """The geotechnical recommendations memorandum (four pages)."""
    f = {what: memo for what, memo, _i0, _i1 in FOUNDATION}
    d = _Paper(f"GEOTECHNICAL RECOMMENDATIONS MEMORANDUM  -  REPORT NO. "
               f"{MEMO_NO}")
    d.h("GEOTECHNICAL RECOMMENDATIONS MEMORANDUM", 16)
    d.p(f"Report No. {MEMO_NO}.  Project: {PROJECT}, Project No. "
        f"{PROJECT_NO}.  Date: 2025-11-18.")
    d.h("1  PURPOSE AND SCOPE")
    d.p("This memorandum gives foundation recommendations for the proposed "
        "building, a steel-framed office building on spread footings with a "
        "slab on grade. It is based on four borings, B-1 to B-4, drilled to "
        "12 m near the corners of the building, and on laboratory tests of "
        "samples from them.")
    d.h("2  SUBSURFACE CONDITIONS")
    d.p("The borings met the same sequence across the site:")
    d.table((110, 110, 248), [
        ("DEPTH (m)", "ELEVATION", "MATERIAL"),
        ("0.0 to 0.9", "99.8 to 98.9", "Fill: loose silty sand with gravel"),
        ("0.9 to 6.5", "98.9 to 93.3",
         "Dense, well-graded sand with gravel"),
        ("6.5 to 12.0", "93.3 to 87.8", "Very stiff silty clay")])
    d.p("Groundwater was measured in standpipes in B-2 and B-4 between El. "
        "96.3 and El. 96.5 over four weeks.")
    d.new_page()
    d.h("3  RECOMMENDATIONS")
    d.p("3.1  FOUNDATION TYPE. Spread footings bearing on the dense sand, "
        "below all fill.")
    d.p(f"3.2  BEARING PRESSURE. Proportion footings for an allowable net "
        f"bearing pressure of {f['bearing']}. This value limits total "
        "settlement to about 25 mm and holds for footings between 1.2 m and "
        "3.0 m wide.")
    d.p(f"3.3  FOOTING DEPTH. Found exterior footings at least {f['depth']} "
        "below finished grade, for frost protection and to bear below the "
        "fill.")
    d.p(f"3.4  GROUNDWATER. Use a design groundwater level of "
        f"{f['groundwater'].title()}.")
    d.p(f"3.5  SEISMIC SITE CLASS. Site Class {f['site_class']}.")
    d.p("3.6  SLAB ON GRADE. Support the slab on 150 mm of compacted "
        "granular base over the dense sand, with all fill removed. Use a "
        f"modulus of subgrade reaction of {f['subgrade']}.")
    d.new_page()
    d.h("4  CONSTRUCTION")
    d.p("4.1  Have the geotechnical engineer inspect every footing "
        "subgrade before concrete is placed.")
    d.p("4.2  Excavations deeper than 1.2 m are to be sloped or shored.")
    d.p("4.3  Dewatering is not expected for footings founded above El. "
        "97.0.")
    d.new_page()
    d.h("5  LIMITATIONS")
    d.p("The recommendations apply to the building described in Section 1. "
        "If the building, its loads or its location change, this "
        "memorandum is to be reviewed.")
    return d.bytes(f"Geotechnical recommendations memorandum {MEMO_NO}")


@lru_cache(maxsize=None)
def build_concrete_submittal() -> bytes:
    """A concrete mix submittal for Class C (five pages)."""
    d = _Paper(f"{PROJECT}  -  SUBMITTAL 03 30 00-02")
    d.h("SUBMITTAL TRANSMITTAL", 16)
    d.table((150, 318), [
        ("ITEM", "ENTRY"),
        ("Project", f"{PROJECT}, Project No. {PROJECT_NO}"),
        ("Submittal No.", "03 30 00-02"),
        ("Specification section", "03 30 00, Cast-in-Place Concrete"),
        ("Description", "Concrete mix design, Class C"),
        ("Use", "Columns and suspended slabs"),
        ("Date", "2026-03-16")])
    d.p("ATTACHED:")
    for line in ("1.  Mix design, Mix No. C-35-20.",
                 "2.  Strength record of Mix No. C-35-20.",
                 "3.  Cement mill certificate.",
                 "4.  Admixture product data."):
        d.p(line, indent=18)
    d.p("CONTRACTOR'S REVIEW: Reviewed for conformance with the contract "
        "documents. DEVIATIONS: NONE.")
    d.new_page()
    d.h("1  MIX DESIGN, MIX NO. C-35-20", 13)
    d.table((250, 110, 108), [
        ("MATERIAL", "QUANTITY", "UNIT"),
        ("Portland cement, Type GU", "310", "kg/m3"),
        ("Fly ash, Class F", "133", "kg/m3"),
        ("Total cementitious material", "443", "kg/m3"),
        ("Water", "213", "kg/m3"),
        ("Coarse aggregate, 20 mm", "1020", "kg/m3"),
        ("Fine aggregate", "760", "kg/m3"),
        ("Water-reducing admixture", "1.3", "L/m3")])
    d.table((250, 218), [
        ("PROPERTY", "VALUE"),
        ("Specified 28-day strength", "35 MPa"),
        ("Water-cementitious ratio", "0.48"),
        ("Fly ash, share of cementitious material", "30 %"),
        ("Slump", "100 mm"),
        ("Air content", "Not air entrained (interior concrete)")])
    d.new_page()
    d.h("2  STRENGTH RECORD, MIX NO. C-35-20", 13)
    d.p("28-day compressive strength, consecutive tests, each the average "
        "of two cylinders:")
    import datetime
    tests = (41.2, 39.8, 42.5, 40.1, 43.0, 38.9, 41.7, 40.6, 42.2, 39.5,
             41.0, 40.3)
    first = datetime.date(2025, 9, 2)
    d.table((80, 130, 130), [("TEST", "DATE CAST", "STRENGTH (MPa)")] + [
        (str(i), (first + datetime.timedelta(days=7 * (i - 1))).isoformat(),
         f"{v:.1f}") for i, v in enumerate(tests, 1)])
    d.p(f"Number of tests: {len(tests)}.  Average: "
        f"{sum(tests) / len(tests):.1f} MPa.")
    d.new_page()
    d.h("3  CEMENT MILL CERTIFICATE", 13)
    d.p("Portland cement, Type GU. Production period: February 2026.")
    d.table((250, 218), [
        ("PROPERTY", "RESULT"),
        ("Fineness (Blaine)", "385 m2/kg"),
        ("Initial setting time", "140 min"),
        ("Compressive strength, 28 days", "43.5 MPa"),
        ("Sulphur trioxide", "2.9 %")])
    d.new_page()
    d.h("4  ADMIXTURE PRODUCT DATA", 13)
    d.p("Water-reducing admixture, mid-range. Dosage 0.2 to 0.5 L per 100 "
        "kg of cementitious material. Chloride ion content: none added.")
    return d.bytes("Submittal 03 30 00-02, concrete mix design, Class C")


#: Dots per inch of the scanned set: 6 pt lettering is about 12 pixels
#: tall, legible in a zoom and not in a view of the whole sheet.
SCAN_DPI = 150


@lru_cache(maxsize=None)
def build_scanned_set(dpi: int = SCAN_DPI) -> bytes:
    """The issue 1 set as a scan: each sheet one grey image, no text layer.
    The same sheets with the same 13 problems; only a look reads them."""
    import pymupdf
    src = pymupdf.open(stream=build_drawing_set(2, 1).pdf, filetype="pdf")
    out = pymupdf.open()
    try:
        for page in src:
            pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
            new = out.new_page(width=page.rect.width,
                               height=page.rect.height)
            new.insert_image(new.rect,
                             stream=pix.tobytes("jpeg", jpg_quality=60))
        out.set_metadata({"title": f"{PROJECT} drawings, issue 1 (scan)"})
        return out.tobytes(garbage=3, deflate=True, no_new_id=True)
    finally:
        out.close()
        src.close()


#: Fixture name -> what it builds (the core's ``review_eval.fixture`` hook).
_BUILDERS = {
    "coord_set": lambda: build_drawing_set(2, 1).pdf,
    "coord_set_prev": lambda: build_drawing_set(2, 0).pdf,
    "coord_set_scan": build_scanned_set,
    "coord_set_long": lambda: build_drawing_set(5, 1).pdf,
    "coord_manual": build_project_manual,
    "coord_memo": build_geotechnical_memo,
    "coord_submittal": build_concrete_submittal,
}


def fixture(name: str) -> Optional[bytes]:
    """The PDF bytes of fixture ``name``, or None when it is not one of
    these (the core then reports an unknown fixture)."""
    build = _BUILDERS.get(name)
    return build() if build is not None else None


__all__ = ["Planted", "BASE_PLANTED", "LONG_PLANTED", "REVISIONS",
           "SUBMITTAL_PLANTED", "MANUAL_SECTIONS", "CITED_SECTIONS",
           "CONCRETE", "FOUNDATION", "DrawingSet", "build_drawing_set",
           "build_scanned_set", "SCAN_DPI", "footing_rows",
           "build_project_manual", "build_geotechnical_memo",
           "build_concrete_submittal", "audit", "fixture", "levels",
           "doors_on_plan", "doors_scheduled", "room_names"]
