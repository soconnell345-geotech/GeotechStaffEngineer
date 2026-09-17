"""One boring or test-pit log -> one :class:`Investigation`.

GEOMETRY SAYS WHERE, THE MODEL SAYS WHAT. planlens' ``log_grid`` has already
found the form's columns, fitted the depth ruler and put every text line at a
(column, depth) with the box it came from. It asserts nothing about MEANING:
it does not know that the cell reading ``5-9-12`` is three drives of a split
spoon, that the little triangle beside the ruler is a water level, or that
``50/5"`` is refusal rather than the number fifty. That is what the model is
for, and it is all the model is for.

So the reader hands over the rows as the primary source and the rendered page
as the second opinion, with one rule: the rows are what the page says, and the
picture is for resolving what the rows leave ambiguous -- a sample symbol, a
water symbol, refusal notation, stacked drives, which of two numbers beside
each other is the N value. Every value comes back with the page and the box it
came from.

WHAT PYTHON CHECKS AFTERWARDS, AND WHY. The model can return a depth that is
not on the page. A ruler fitted to a page states exactly which depths that
page covers, so a depth outside that range plus a tolerance is not a reading,
it is an invention -- and it is REFUSED here, moved into ``unresolved`` with
the range it fell outside. The same holds for a depth on a page with no ruler
at all: there is no scale, so there is no depth. Refusing costs a real value
now and then; accepting one costs a reviewer's trust in all of them.

THE BUDGET. One model call reads the log. A continuation sheet gets a call of
its own when the first call says it could not finish, and the ceiling is
:data:`MAX_MODEL_CALLS` per log so a confused loop cannot run away. Nothing
here is an agent loop: there is one bounded thing to read and the reader reads
it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from report_ingest.engine import Engine, image_block, text_block, user
from report_ingest.model import (
    DrillingDetails, Investigation, Layer, Provenance, Quantity, Sample, SPT,
    WaterLevel,
)

__all__ = [
    "read_log", "LogReadResult", "LogReading", "LOG_READER_SYSTEM",
    "MAX_MODEL_CALLS", "PAGE_DPI", "DEPTH_SLACK",
]

#: The ceiling in the brief: one main call plus follow-ups for continuation
#: pages. A log with more sheets than this reads the sheets it can and says
#: which it did not.
MAX_MODEL_CALLS = 6
#: About 110 dpi: a letter page at roughly 1200 px on its long side. Enough
#: to see a sample symbol and a water triangle, which is what the picture is
#: for; the words come from the rows.
PAGE_DPI = 110.0
#: Fit slack, as a fraction of one printed ruler step. The window a depth has
#: to fall in is the depth at the TOP of the paper to the depth at its
#: BOTTOM, read off the ruler's own linear map -- so a layer contact printed
#: below the last tick, or a total depth in a footer, is inside it, while a
#: depth that would be off the edge of the sheet is not. This is the margin on
#: top of that, to absorb the fit's own residual.
DEPTH_SLACK = 0.5
#: Rows serialised per page. A crowded log page runs to about 120 placed
#: cells; a page with many more than this is a table of several borings and
#: the extra rows are of no use to a reader of ONE log.
MAX_ROWS_PER_PAGE = 260
#: How much of one cell's text is sent. A description cell can run to a
#: paragraph; the layers come back from the grid's own layer list as well.
MAX_CELL_CHARS = 200


# ---------------------------------------------------------------------------
# what the model returns
# ---------------------------------------------------------------------------
#
# These mirror report_ingest.model but are NOT the same classes. The model
# fills in what it read off the page; Python turns that into the record,
# attaching provenance and refusing what the page cannot support. Keeping them
# apart is what makes the refusal possible: a schema the model fills cannot
# also be the schema that has already been checked.

class ReadProv(BaseModel):
    """Where the model says it read a value."""

    model_config = ConfigDict(extra="forbid")

    page: int = Field(description="0-based PDF page index")
    bbox: Optional[Tuple[float, float, float, float]] = Field(
        default=None,
        description="the box from the grid row this came from, copied "
                    "exactly; null when you read it off the picture")
    from_image: bool = Field(
        default=False,
        description="true when the picture, not the rows, is what told you")
    note: str = Field(
        default="", description="what was ambiguous and what settled it, in "
                                "15 words or fewer; empty when nothing was")


class ReadLayer(BaseModel):
    """One described stratum."""

    model_config = ConfigDict(extra="forbid")

    top: float = Field(description="depth to the top, in the log's own unit")
    bottom: Optional[float] = Field(
        default=None,
        description="depth to the base; null when this page does not print it")
    description: str = Field(description="the log's own words, verbatim")
    uscs: str = Field(
        default="",
        description="the group symbol the log PRINTS (CL, SM, SP-SM). Empty "
                    "when it prints none. Never work one out from the words")
    consistency: str = Field(default="", description="as printed, or empty")
    color: str = Field(default="", description="as printed, or empty")
    moisture: str = Field(default="", description="as printed, or empty")
    prov: ReadProv


class ReadSample(BaseModel):
    """One sample, and the index values printed against it on the log face."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str = Field(default="", description="the log's own label")
    top: float = Field(description="depth to the top of the sample")
    bottom: Optional[float] = Field(default=None)
    kind: str = Field(
        default="other",
        description="spt, ring, shelby, bulk, grab, core, cuttings or other. "
                    "'other' when the symbol cannot be read")
    recovery_percent: Optional[float] = Field(default=None)
    rqd_percent: Optional[float] = Field(default=None)
    water_content: Optional[float] = Field(
        default=None, description="percent, when printed against this sample")
    dry_unit_weight: Optional[float] = Field(
        default=None, description="in the log's own unit for it")
    dry_unit_weight_unit: str = Field(
        default="", description="pcf, kN/m3, g/cm3 ... as the column heads it")
    liquid_limit: Optional[float] = Field(default=None)
    plastic_limit: Optional[float] = Field(default=None)
    plasticity_index: Optional[float] = Field(default=None)
    fines_percent: Optional[float] = Field(default=None)
    qu: Optional[float] = Field(default=None)
    qu_unit: str = Field(default="", description="tsf, kPa, psf ... as headed")
    pocket_pen: Optional[float] = Field(default=None)
    pocket_pen_unit: str = Field(default="")
    uscs: str = Field(default="", description="as printed against the sample")
    note: str = Field(default="")
    prov: ReadProv


class ReadSPT(BaseModel):
    """One driven record, as the log prints it."""

    model_config = ConfigDict(extra="forbid")

    depth_top: float = Field(description="depth at the start of the drive")
    depth_bottom: Optional[float] = Field(default=None)
    blows: List[str] = Field(
        default_factory=list,
        description="one entry per increment, EXACTLY as printed. '12' for a "
                    "count, '50/5\"' for a refusal. Never add them up")
    n: Optional[int] = Field(
        default=None,
        description="the N value the log PRINTS. null when it prints only "
                    "the drives -- do not compute one")
    refusal: bool = Field(
        default=False,
        description="the record shows refusal: 50 blows for less than the "
                    "full increment, or the word itself")
    sample_id: str = Field(
        default="", description="the sample this drive belongs to, when one")
    prov: ReadProv


class ReadWater(BaseModel):
    """One water observation."""

    model_config = ConfigDict(extra="forbid")

    depth: Optional[float] = Field(
        default=None,
        description="depth to water; null when the log says none was "
                    "encountered")
    when: str = Field(
        default="unknown",
        description="while_drilling, at_completion, after_hours, "
                    "not_encountered or unknown")
    hours: Optional[float] = Field(
        default=None, description="hours after completion, when stated")
    date: str = Field(default="", description="as printed")
    casing_depth: Optional[float] = Field(default=None)
    caved_depth: Optional[float] = Field(default=None)
    note: str = Field(default="")
    prov: ReadProv


class ReadDrilling(BaseModel):
    """How the hole was made, as the header prints it."""

    model_config = ConfigDict(extra="forbid")

    method: str = Field(default="")
    equipment: str = Field(default="")
    hammer_type: str = Field(
        default="",
        description="automatic, safety, donut, cathead ... as printed")
    hammer_energy_ratio: Optional[float] = Field(
        default=None, description="the energy ratio as a PERCENT, when the "
                                  "log prints one")
    sampler: str = Field(default="")
    driller: str = Field(default="")
    contractor: str = Field(default="")
    logged_by: str = Field(default="")


class Unsettled(BaseModel):
    """Something on this log you could not read."""

    model_config = ConfigDict(extra="forbid")

    what: str = Field(description="what it is, in 15 words or fewer")
    page: Optional[int] = Field(default=None)
    why: str = Field(description="what stopped you, in 20 words or fewer")


class LogReading(BaseModel):
    """One log as the model read it."""

    model_config = ConfigDict(extra="forbid")

    investigation_id: str = Field(
        description="the log's own identifier, exactly as printed (B-2, "
                    "TP-4, SB-01). Empty string only if the page prints none")
    kind: str = Field(
        default="boring",
        description="boring, test_pit, cpt, dcp, hand_auger, well or other")
    depth_unit: str = Field(
        default="",
        description="'ft' or 'm' -- the unit EVERY depth you report is in. "
                    "Empty when neither the grid nor the page states one")
    units_known: bool = Field(
        default=True,
        description="false when no unit is printed anywhere and you are "
                    "reporting the ruler's own numbers")
    x: Optional[float] = Field(
        default=None, description="easting or longitude, as printed")
    y: Optional[float] = Field(
        default=None, description="northing or latitude, as printed")
    coordinate_system: str = Field(default="", description="as printed")
    elevation: Optional[float] = Field(
        default=None, description="ground surface elevation, as printed")
    elevation_unit: str = Field(default="", description="ft or m, as printed")
    total_depth: Optional[float] = Field(
        default=None, description="in the depth unit above")
    date_started: str = Field(default="", description="as printed")
    date_finished: str = Field(default="", description="as printed")
    station: str = Field(default="", description="as printed")
    offset: str = Field(default="", description="as printed")
    sheet: str = Field(
        default="", description="the log's own 'Page 1 of 3', as printed")
    drilling: ReadDrilling = Field(default_factory=ReadDrilling)
    layers: List[ReadLayer] = Field(default_factory=list)
    samples: List[ReadSample] = Field(default_factory=list)
    spt: List[ReadSPT] = Field(default_factory=list)
    water: List[ReadWater] = Field(default_factory=list)
    remarks: str = Field(
        default="", description="the log's own notes, verbatim; empty if none")
    pages_read: List[int] = Field(
        default_factory=list,
        description="the pages you actually read in this answer")
    pages_left: List[int] = Field(
        default_factory=list,
        description="pages of this log you have NOT read yet, if any")
    unsettled: List[Unsettled] = Field(
        default_factory=list,
        description="everything on this log you could not read, and why")


# ---------------------------------------------------------------------------
# the result
# ---------------------------------------------------------------------------

@dataclass
class LogReadResult:
    """One log read, and everything needed to audit the reading."""

    investigation: Investigation
    #: Every value the reader took from the PICTURE rather than the rows, and
    #: every grid reading it overrode. This is what a reviewer checks first:
    #: the rows can be gone back to, a look cannot.
    changes: List[Dict[str, Any]] = field(default_factory=list)
    #: What was not settled: the model's own list, plus everything Python
    #: refused (a depth off the page, a page with no ruler).
    unresolved: List[Dict[str, Any]] = field(default_factory=list)
    cost: Dict[str, Any] = field(default_factory=dict)
    model_calls: int = 0
    model: str = ""
    warnings: List[str] = field(default_factory=list)
    pages: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "investigation": self.investigation.model_dump(mode="json"),
            "changes": [dict(c) for c in self.changes],
            "unresolved": [dict(u) for u in self.unresolved],
            "cost": dict(self.cost),
            "model_calls": self.model_calls,
            "model": self.model,
            "warnings": list(self.warnings),
            "pages": list(self.pages),
        }


# ---------------------------------------------------------------------------
# the prompt
# ---------------------------------------------------------------------------

LOG_READER_SYSTEM = """\
You are reading ONE exploration log -- a boring, a test pit, a sounding --
off an engineering report, and returning it as data. One log, however many
sheets it runs to.

WHAT YOU ARE GIVEN. A form reader has already found the log's columns, fitted
its depth scale and placed every line of text at a column and a depth, with
the box it occupies on the page. Those ROWS are your primary source: they are
what the page says, with the geometry already worked out. You are also given
the page as a picture, the layers and header fields the form reader pulled
out, and its warnings.

THE ONE RULE ABOUT THE PICTURE. The rows are the source; the picture resolves
what the rows leave ambiguous. Look at it for:
- sample symbols: the rows carry a label, the picture shows whether the
  sampler was a split spoon, a ring, a thin-wall tube, a core barrel;
- water symbols: the inverted triangle and its variants say which reading is
  while drilling and which at completion, and the rows rarely say;
- refusal notation: 50/5", 50/0.1m, 100/3" -- the rows may have split it;
- stacked drives: which numbers belong to one drive set and in what order;
- which of two numbers beside each other is the N value and which is a
  sample number or a recovery.
When the picture is what told you, say so: set from_image on that value's
provenance. Do not use the picture to re-read text the rows already carry.

NEVER INVENT A DEPTH. Every depth you report must come from the scale the
form reader fitted, or from a depth printed on the page. If a value has no
depth you can point at, leave it out and list it under unsettled. If the form
reader found no scale at all, the page carries no depths: report the header
fields and the descriptions, report NO depths, and say so in unsettled. A
depth that is not on the page will be thrown away and counted against this
reading.

THE UNIT. Every depth you report is in ONE unit, and you state it:
depth_unit 'ft' or 'm'. Take it from the form reader when it found one,
otherwise from the page -- a column head reading DEPTH (FT.), an elevation in
feet, a note. If nothing on the page states a unit, set units_known false,
leave depth_unit empty, and report the scale's own numbers unchanged. Do not
convert anything, ever. A log that prints 21.5 feet is reported as 21.5.

WHAT TO RECORD, WHEN THE LOG PRINTS IT.
- The log's own identifier, exactly as printed.
- Ground surface elevation, total depth, the dates, the station and offset,
  the coordinates.
- How the hole was made: the drilling method, the rig, the sampler, and above
  all the HAMMER -- its type (automatic, safety, donut, cathead) and its
  energy ratio if a number is printed. An N value means a different thing
  behind different hammers, so a hammer that is printed and not recorded is a
  number thrown away.
- Every described layer, with its top depth, its base where the page prints
  one, the description in the log's OWN WORDS verbatim, and the USCS group
  symbol ONLY where the log prints one. Never work a symbol out from the
  words: a record that says the log printed CL when it did not is worse than
  one that says nothing.
- Every sample: its label, its depth interval, what the sampler was, the
  recovery and RQD where printed, and any index value printed against it ON
  THE LOG FACE -- water content, dry unit weight, Atterberg limits, percent
  fines, unconfined strength, pocket penetrometer. Give the unit as the
  column heads it.
- Every driven record separately from the sample: the drives EXACTLY as
  printed, one entry per increment, as strings. '12' is a count, '50/5"' is a
  refusal and stays a string. Do NOT add drives together. Record n only where
  the log PRINTS an N value; where it prints only the drives, leave n null.
  That is not a gap -- it is what the page says.
- Every water observation: the depth, whether it was read while drilling, at
  completion or after a stated number of hours, the date, the casing and
  caved depths. A log that says water was not encountered is a water entry
  with when 'not_encountered' and a null depth, not an empty list.
- The log's remarks, verbatim.

PROVENANCE. Every layer, sample, driven record and water level carries a
provenance: the page, and the box copied EXACTLY from the grid row it came
from. Where the picture is what told you, set from_image true and leave the
box null. Where a value is ambiguous and you resolved it, say how in the
note, briefly.

WHAT NOT TO DO. Do not compute. Do not convert. Do not fill a field from what
logs usually say. Do not give a layer a symbol, a sample a type or a water
reading a timing the page does not support -- 'other' and 'unknown' are
answers. Anything you cannot settle goes in unsettled with the reason.
"""

_CONTINUE_INSTRUCTION = (
    "You have pages of this log left to read: {pages}. Here are their rows "
    "and their pictures. Return the WHOLE log again -- everything you "
    "already read plus what these pages add -- in the same shape. Depths "
    "continue from the previous sheet; do not restart them at zero unless "
    "the sheet itself does."
)


# ---------------------------------------------------------------------------
# what the model is shown
# ---------------------------------------------------------------------------

def _clip(text: str, n: int = MAX_CELL_CHARS) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n - 1] + "…"


def serialise_rows(grid: Any, pages: Sequence[int]) -> str:
    """The grid's cells as compact lines, one per cell, grouped by page.

    ``col | depth | conf | x0,y0,x1,y1 | text``. The box is in the line
    because the model has to copy it back as provenance; a reader that had to
    invent boxes would invent them.
    """
    out: List[str] = []
    for page in pages:
        ruler = grid.rulers.get(page)
        cells = [c for c in grid.rows if c.page == page]
        out.append(f"--- page {page}: {len(cells)} placed cell(s) ---")
        if ruler is None:
            out.append("  NO DEPTH SCALE ON THIS PAGE. Nothing here carries "
                       "a depth.")
        else:
            depths = [c.depth for c in cells if c.depth is not None]
            span = (f"{min(depths):g} to {max(depths):g}" if depths
                    else "no placed depths")
            out.append(
                f"  depth scale: {ruler.kind}, unit "
                f"{ruler.unit or 'NOT STATED'}"
                f"{' (' + ruler.unit_source + ')' if ruler.unit_source else ''}"
                f", step {ruler.step:g}, fit residual {ruler.residual:.2f}, "
                f"confidence {ruler.confidence:.2f}; "
                f"cells placed between {span}")
        elev = grid.elevation_rulers.get(page)
        if elev is not None:
            out.append(f"  an elevation scale is also on this page "
                       f"(unit {elev.unit or 'not stated'})")
        columns = [c for c in grid.columns if c.page == page]
        if columns:
            out.append("  columns: " + "; ".join(
                f"{c.id}={c.name}"
                + (f" [{c.header.strip()[:40]}]" if c.header.strip() else "")
                + (f" {c.unit}" if c.unit else "")
                for c in columns))
        out.append("  col | depth | conf | x0,y0,x1,y1 | text")
        shown = cells[:MAX_ROWS_PER_PAGE]
        for cell in shown:
            depth = "-" if cell.depth is None else f"{cell.depth:g}"
            x0, y0, x1, y1 = cell.bbox
            out.append(
                f"  {cell.column} | {depth} | {cell.confidence:.2f} | "
                f"{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f} | {_clip(cell.text)}")
        if len(cells) > len(shown):
            out.append(f"  [{len(cells) - len(shown)} further cell(s) not "
                       f"listed; this page carries more rows than one log]")
    return "\n".join(out)


def _brief(grid: Any, pages: Sequence[int], ledger: Sequence[str],
           item_title: str, report_id: str) -> str:
    layers = [
        {"top": ly.top, "bottom": ly.bottom,
         "description": _clip(ly.description, 300),
         "pages": list(ly.pages), "confidence": round(ly.confidence, 2)}
        for ly in grid.layers]
    parts: List[str] = [
        f"ONE EXPLORATION LOG, on page(s) {', '.join(str(p) for p in pages)}"
        + (f" of report {report_id}" if report_id else "") + ".",
    ]
    if item_title:
        parts.append(f"The document titles it: {item_title}")
    parts += [
        "",
        "WHAT THE PAGE LEDGER SAYS ABOUT THESE PAGES",
        "\n".join(ledger) if ledger else "(no ledger line)",
        "",
        "THE DEPTH UNIT THE FORM READER SETTLED ON: "
        + (grid.unit or "NONE -- it could not read one off these pages"),
        "",
        "HEADER FIELDS THE FORM READER PULLED OUT (key = value, as printed)",
        ("\n".join(f"  {k} = {_clip(v, 160)}"
                   for k, v in sorted(grid.fields.items()))
         or "  (none)"),
        "",
        "LAYERS THE FORM READER BOUND TO DEPTHS (its own reading, to check "
        "against the rows, not to copy)",
        (json.dumps(layers, indent=1) if layers else "  (none)"),
        "",
        "WHAT THE FORM READER WARNS ABOUT THESE PAGES",
        ("\n".join(f"  - {w}" for w in grid.warnings) or "  (nothing)"),
        "",
        "THE ROWS",
        serialise_rows(grid, pages),
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# turning the reading into the record
# ---------------------------------------------------------------------------

def _depth_window(doc: Any, grid: Any, pages: Sequence[int]
                  ) -> Optional[Tuple[float, float]]:
    """What depths the paper of this log can carry, over all its sheets.

    A ruler is a linear map from y to depth, so it says what depth sits at the
    TOP edge of the page and what depth sits at the BOTTOM edge. Everything
    printed on the sheet lies between those two, and nothing else does. That
    is the window -- not the range of the ticks, which stops short of the
    bottom of the form and would refuse the last layer of every real log, and
    not the range of the placed cells, which is narrower still.

    None when no page of this log has a ruler: there is then no depth on the
    log at all, and every depth the reader returns is refused.
    """
    lo: Optional[float] = None
    hi: Optional[float] = None
    for page in pages:
        ruler = grid.rulers.get(page)
        if ruler is None:
            continue
        try:
            height = float(doc.summary(page).height)
        except Exception:                       # a page that will not measure
            height = 792.0                      # falls back to a letter sheet
        edges = [ruler.value_at(0.0), ruler.value_at(height)]
        slack = DEPTH_SLACK * abs(ruler.step or 0.0)
        low, high = min(edges) - slack, max(edges) + slack
        lo = low if lo is None else min(lo, low)
        hi = high if hi is None else max(hi, high)
    if lo is None or hi is None:
        return None
    return (lo, hi)


class _Builder:
    """Turns one :class:`LogReading` into an :class:`Investigation`.

    Every depth passes :meth:`_depth` first. That is the only place a value
    can be refused, and it is deliberately the narrow gate the whole reading
    goes through.
    """

    def __init__(self, reading: LogReading, window: Optional[Tuple[float, float]],
                 pages: Sequence[int], report_id: str) -> None:
        self.reading = reading
        self.window = window
        self.pages = list(pages)
        self.report_id = report_id
        self.unresolved: List[Dict[str, Any]] = []
        self.changes: List[Dict[str, Any]] = []
        self.unit = (reading.depth_unit or "").strip()
        self.units_known = bool(reading.units_known and self.unit)

    # -- the gate ----------------------------------------------------------
    def _depth(self, value: Optional[float], what: str,
               page: Optional[int] = None) -> Optional[Quantity]:
        """A depth as a Quantity, or None with the refusal recorded."""
        if value is None:
            return None
        value = float(value)
        if self.window is None:
            self.unresolved.append({
                "what": what, "page": page, "value": value,
                "why": "no depth scale was found on any page of this log, so "
                       "the page carries no depths; the reader returned one "
                       "anyway and it was refused",
                "refused_by": "python"})
            return None
        lo, hi = self.window
        if not (lo <= value <= hi):
            self.unresolved.append({
                "what": what, "page": page, "value": value,
                "why": f"depth {value:g} is outside what the scale on this "
                       f"log reads ({lo:g} to {hi:g}); refused rather than "
                       f"accepted",
                "refused_by": "python"})
            return None
        return Quantity(value=value, unit=self.unit)

    def _prov(self, prov: Optional[ReadProv], what: str) -> Optional[Provenance]:
        if prov is None:
            return None
        page = int(prov.page)
        if page not in self.pages:
            # A page outside this log is a reading of a different page. The
            # value is kept -- the depth gate is what protects the numbers --
            # but the box is not trusted to point at anything.
            self.unresolved.append({
                "what": what, "page": page,
                "why": f"provenance names page {page}, which is not part of "
                       f"this log ({self.pages}); the box was dropped",
                "refused_by": "python"})
            return Provenance(page=self.pages[0] if self.pages else page,
                              method="vision" if prov.from_image else "grid",
                              confidence=0.4, note=prov.note)
        method = "vision" if prov.from_image else "grid"
        if prov.from_image:
            self.changes.append({
                "what": what, "page": page,
                "why": prov.note or "read off the picture, not the rows"})
        return Provenance(page=page,
                          bbox=tuple(prov.bbox) if prov.bbox else None,
                          method=method,
                          confidence=0.9 if not prov.from_image else 0.8,
                          note=prov.note)

    def _q(self, value: Optional[float], unit: str) -> Optional[Quantity]:
        """A non-depth quantity, in whatever unit the column headed it."""
        if value is None:
            return None
        return Quantity(value=float(value), unit=(unit or "").strip())

    # -- the pieces --------------------------------------------------------
    def layers(self) -> List[Layer]:
        out: List[Layer] = []
        for n, ly in enumerate(self.reading.layers):
            what = f"layer {n + 1} ({_clip(ly.description, 40)})"
            top = self._depth(ly.top, what + " top", ly.prov.page)
            if top is None:
                continue
            bottom = self._depth(ly.bottom, what + " base", ly.prov.page)
            out.append(Layer(
                top=top, bottom=bottom,
                description=ly.description, uscs=ly.uscs.strip(),
                consistency=ly.consistency, color=ly.color,
                moisture=ly.moisture, prov=self._prov(ly.prov, what)))
        return out

    def samples(self) -> List[Sample]:
        out: List[Sample] = []
        for n, s in enumerate(self.reading.samples):
            what = f"sample {s.sample_id or n + 1}"
            top = self._depth(s.top, what + " top", s.prov.page)
            if top is None:
                continue
            kind = s.kind if s.kind in {
                "spt", "ring", "shelby", "bulk", "grab", "core", "cuttings",
                "other"} else "other"
            out.append(Sample(
                sample_id=s.sample_id, top=top,
                bottom=self._depth(s.bottom, what + " base", s.prov.page),
                kind=kind,
                recovery_percent=_pct(s.recovery_percent),
                rqd_percent=_pct(s.rqd_percent),
                water_content=s.water_content,
                dry_unit_weight=self._q(s.dry_unit_weight,
                                        s.dry_unit_weight_unit),
                liquid_limit=s.liquid_limit,
                plastic_limit=s.plastic_limit,
                plasticity_index=s.plasticity_index,
                fines_percent=_pct(s.fines_percent),
                qu=self._q(s.qu, s.qu_unit),
                pocket_pen=self._q(s.pocket_pen, s.pocket_pen_unit),
                uscs=s.uscs.strip(), note=s.note,
                prov=self._prov(s.prov, what)))
        return out

    def spt(self) -> List[SPT]:
        out: List[SPT] = []
        for n, rec in enumerate(self.reading.spt):
            what = f"driven record {n + 1} at {rec.depth_top:g}"
            top = self._depth(rec.depth_top, what, rec.prov.page)
            if top is None:
                continue
            blows: List[Any] = []
            for entry in rec.blows:
                text = str(entry).strip()
                # A plain integer is kept as one; anything else -- a
                # refusal, a range, a note -- stays the string the log
                # printed, because that is what it says.
                blows.append(int(text) if text.isdigit() else text)
            out.append(SPT(
                depth_top=top,
                depth_bottom=self._depth(rec.depth_bottom, what + " base",
                                         rec.prov.page),
                blows=blows, n=rec.n, refusal=bool(rec.refusal),
                hammer=self.reading.drilling.hammer_type,
                sample_id=rec.sample_id,
                prov=self._prov(rec.prov, what)))
        return out

    def water(self) -> List[WaterLevel]:
        out: List[WaterLevel] = []
        for n, w in enumerate(self.reading.water):
            what = f"water observation {n + 1}"
            when = w.when if w.when in {
                "while_drilling", "at_completion", "after_hours",
                "not_encountered", "unknown"} else "unknown"
            depth = self._depth(w.depth, what, w.prov.page)
            if w.depth is not None and depth is None:
                # The depth was refused; the observation itself is still a
                # fact about the hole and is kept without one.
                pass
            out.append(WaterLevel(
                depth=depth, when=when, hours=w.hours, date=w.date,
                casing_depth=self._depth(w.casing_depth, what + " casing",
                                         w.prov.page),
                caved_depth=self._depth(w.caved_depth, what + " caved",
                                        w.prov.page),
                note=w.note, prov=self._prov(w.prov, what)))
        return out

    def build(self) -> Investigation:
        r = self.reading
        kind = r.kind if r.kind in {
            "boring", "test_pit", "cpt", "dcp", "hand_auger", "well",
            "other"} else "other"
        elevation = None
        if r.elevation is not None:
            elevation = Quantity(value=float(r.elevation),
                                 unit=(r.elevation_unit or self.unit).strip())
        # Total depth is a HEADER value, not a reading placed on the scale:
        # on sheet 1 of 3 it names the bottom of the whole hole, which is
        # below this sheet's own paper. Gating it on the window would refuse
        # the very number the header exists to state. Only nonsense is
        # refused.
        total = None
        if r.total_depth is not None:
            if float(r.total_depth) >= 0.0 and self.units_known:
                total = Quantity(value=float(r.total_depth), unit=self.unit)
            elif float(r.total_depth) < 0.0:
                self.unresolved.append({
                    "what": "total depth", "page": None,
                    "value": float(r.total_depth),
                    "why": "a hole cannot have a negative total depth",
                    "refused_by": "python"})
        for item in r.unsettled:
            self.unresolved.append({
                "what": item.what, "page": item.page, "why": item.why,
                "refused_by": "reader"})
        for page in r.pages_left:
            self.unresolved.append({
                "what": f"page {page} of this log", "page": page,
                "why": "the reader did not get to this page within its "
                       "model-call budget",
                "refused_by": "budget"})
        return Investigation(
            investigation_id=r.investigation_id.strip(),
            kind=kind,
            depth_unit=self.unit,
            units_known=self.units_known,
            x=r.x, y=r.y, coordinate_system=r.coordinate_system,
            elevation=elevation, total_depth=total,
            date_started=r.date_started, date_finished=r.date_finished,
            drilling=DrillingDetails(
                method=r.drilling.method, equipment=r.drilling.equipment,
                hammer_type=r.drilling.hammer_type,
                hammer_energy_ratio=r.drilling.hammer_energy_ratio,
                sampler=r.drilling.sampler, driller=r.drilling.driller,
                contractor=r.drilling.contractor,
                logged_by=r.drilling.logged_by),
            layers=self.layers(), samples=self.samples(), spt=self.spt(),
            water=self.water(), remarks=r.remarks,
            station=r.station, offset=r.offset,
            pages=list(self.pages), source_report=self.report_id,
            sheet=r.sheet)


def _pct(value: Optional[float]) -> Optional[float]:
    """A percentage clamped to 0-100, or None. The record's field is bounded
    and a reader that returns 105 % must not fail the whole log."""
    if value is None:
        return None
    return max(0.0, min(100.0, float(value)))


# ---------------------------------------------------------------------------
# the reader
# ---------------------------------------------------------------------------

def read_log(doc, item_pages: Sequence[int], engine: Engine, *,
             budget: int = MAX_MODEL_CALLS, grid: Any = None,
             ledger: Optional[Sequence[str]] = None, item_title: str = "",
             report_id: str = "", dpi: float = PAGE_DPI) -> LogReadResult:
    """Read ONE exploration log and return it as an :class:`Investigation`.

    ``item_pages`` are the pages of one log, continuation sheets included --
    ``planlens.document.roles.document_items`` gives them. ``budget`` is the
    model-call ceiling for this log; the first call reads it, and a further
    call is spent only when the reader itself says it has pages left.

    Pass ``grid`` when ``log_grid`` has already been run over these pages (a
    scorer runs it to measure the grid alone), otherwise it is run here.
    """
    pages = [int(p) for p in item_pages]
    if not pages:
        raise ValueError("read_log needs at least one page")
    budget = max(1, min(int(budget), MAX_MODEL_CALLS))

    if grid is None:
        from planlens.document.loggrid import log_grid
        grid = log_grid(doc, pages)
    if ledger is None:
        try:
            from planlens.document.roles import page_ledger
            lines = page_ledger(doc)
            ledger = [ln for ln in lines
                      if any(ln.startswith(f"p{p:03d} ") for p in pages)]
        except Exception:                        # a ledger is a nicety
            ledger = []

    spent = {"calls": 0, "input_tokens": 0, "output_tokens": 0,
             "cache_read_tokens": 0, "seconds": 0.0, "dollars": 0.0}

    def charge(reply) -> None:
        spent["calls"] += 1
        spent["input_tokens"] += reply.usage.input_tokens
        spent["output_tokens"] += reply.usage.output_tokens
        spent["cache_read_tokens"] += reply.usage.cache_read_tokens
        spent["seconds"] += reply.seconds
        spent["dollars"] += reply.usage.dollars(reply.model)

    def images_for(wanted: Sequence[int]) -> List[bytes]:
        out: List[bytes] = []
        for page in wanted:
            try:
                png, _info = doc.render(page, dpi=dpi)
            except Exception:                    # a page that will not draw
                continue                         # is read from its rows alone
            out.append(png)
        return out

    brief = _brief(grid, pages, ledger, item_title, report_id)
    messages: List[Dict[str, Any]] = [
        user(text_block(brief),
             *[image_block(png) for png in images_for(pages)])]
    reply = engine.complete(messages, system=LOG_READER_SYSTEM,
                            output_format=LogReading)
    charge(reply)
    reading = reply.parsed
    model_calls = 1
    if reading is None:
        raise RuntimeError(
            f"the log reader returned no structured reading for pages "
            f"{pages} (stop_reason {reply.stop_reason!r})")

    # Continuation sheets: only when the reader itself says it has pages
    # left, and only within the budget. Each round re-reads the whole log, so
    # the last answer is the complete one.
    while reading.pages_left and model_calls < budget:
        left = [p for p in reading.pages_left if p in pages]
        if not left:
            break
        messages.append({"role": "assistant",
                         "content": reply.content or [text_block("")]})
        messages.append(user(
            text_block(_CONTINUE_INSTRUCTION.format(
                pages=", ".join(str(p) for p in left))),
            text_block(serialise_rows(grid, left)),
            *[image_block(png) for png in images_for(left)]))
        reply = engine.complete(messages, system=LOG_READER_SYSTEM,
                                output_format=LogReading)
        charge(reply)
        model_calls += 1
        if reply.parsed is None:
            break
        if reply.parsed.pages_left == reading.pages_left:
            reading = reply.parsed
            break                     # no progress: stop rather than spin
        reading = reply.parsed

    builder = _Builder(reading, _depth_window(doc, grid, pages), pages,
                       report_id)
    investigation = builder.build()
    spent["seconds"] = round(spent["seconds"], 2)
    spent["dollars"] = round(spent["dollars"], 5)
    return LogReadResult(
        investigation=investigation,
        changes=builder.changes,
        unresolved=builder.unresolved,
        cost=spent,
        model_calls=model_calls,
        model=reply.model or getattr(engine, "name", ""),
        warnings=list(grid.warnings),
        pages=pages)
