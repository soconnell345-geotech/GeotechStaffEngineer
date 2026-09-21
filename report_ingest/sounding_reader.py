"""One cone sounding or dynamic probe -> one :class:`Investigation` with a series.

A SOUNDING IS NOT A LOG, AND THAT IS THE WHOLE DESIGN. A boring log is a form:
a ruler down one side, a description column, samples and drives placed against
it, and a reader's job is to say what each placed cell MEANS. A cone sounding
prints none of that. It is three traces against depth -- tip resistance,
sleeve friction, pore pressure -- and everything anyone wants from it (a
profile, a soil behaviour type, a bearing estimate, a liquefaction screen) is
computed from those traces. So the record's payload is a SERIES, and the
reader's job is to get the numbers off the paper at a stated depth step with a
stated confidence. The same holds for a dynamic cone or dynamic probe, whose
series is blows against depth.

TWO SHAPES OF SHEET, AND THEY ARE READ DIFFERENTLY.

**TABULATED.** The sheet prints the series as a table: a depth column and a
column per channel, two hundred rows of it. Then THE TABLE IS THE RECORD. It
is read deterministically -- planlens' detected tables first, and where a
sheet rules no table at all (which the corpus's tabulated soundings mostly do
not), the text lines banded by their own geometry: a header band naming the
columns, and every band below it whose cells line up under those columns. A
model is not asked to retype two hundred rows it would get wrong; one call
goes on the HEADER -- the identifier, the cone, the datum, the standard --
and on whatever the floor could not settle.

**PLOTTED.** The sheet prints the traces and no numbers at all, which is the
usual case and the hard one. The floor is then the header fields and THE AXIS
RANGES read off the plot's own text: the axis titles with their units, the
tick values, the depth scale. Those ranges are what makes a digitised reading
checkable -- a qc of 42 MPa on an axis that runs to 20 is refused here, not
argued about later. The traces themselves are digitised THROUGH THE PICTURE:
the reader calls ``zoom_plot`` on a panel with its axes in the box and reports
a value per channel at a fixed depth step (:data:`CPT_STEP_M`, or the printed
grid where the sheet states one), each point carrying its own confidence,
which FALLS where one trace crosses another and the reader says which of the
two it followed.

THE BUDGET IS TWO CALLS. One per plot panel group -- a CPT sheet is one panel
group of two or three traces, and a sheet carrying four separate soundings is
four items, not one call each -- and one for the header and the reader's own
unsettled list. A tabulated sheet spends one.

WHAT IS REFUSED, AND WHY EACH. A depth outside the sheet's own depth axis is
not a reading. A channel value outside its axis range by more than a tick is
not a reading. A negative tip resistance is not a reading. A series with no
depth unit is not a series. Each refusal is recorded with the range it fell
outside, because a reviewer needs to know the reader tried.

THE FLOOR AND THE MERGE are the package's own
(:mod:`report_ingest.floor`): the deterministic pass is the first voter, the
model may ADD, may CORRECT only with evidence, and may NEVER DROP. On a
tabulated sheet that rule is doing real work -- the table has the numbers and
the model must not re-type them worse.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field

from report_ingest.engine import (
    Engine, image_block, text_block, tool_result_block, user,
)
from report_ingest.floor import MergeLog, has_evidence, settle, show
from report_ingest.log_floor import merge_investigations
from report_ingest.model import (
    CPTData, CPTPoint, DCPData, DCPPoint, DrillingDetails, Investigation,
    Provenance, Quantity, _canonical_unit,
)

__all__ = [
    "read_sounding", "SoundingReadResult", "SoundingReading",
    "SOUNDING_READER_SYSTEM", "SOUNDING_TOOLS", "SOUNDING_KINDS",
    "MAX_SOUNDING_CALLS", "PAGE_DPI", "ZOOM_DPI",
    "CPT_STEP_M", "DCP_STEP_M", "MIN_STEP_M", "MAX_SERIES_POINTS",
    "AXIS_SLACK_TICKS", "default_step",
    "SoundingFloor", "sounding_floor", "serialise_floor", "merge_sounding",
    "ColumnRole", "column_role", "TabularSeries", "read_tabulated",
    "PlotAxis", "axes_from_text",
]

#: The two kinds this module reads. They share a shape -- a series against
#: depth off one sheet -- and differ in what the series holds, so they share
#: a reader and differ in one section of the prompt and one builder.
SOUNDING_KINDS: Tuple[str, ...] = ("cpt", "dcp")

#: The ceiling in the brief. A TABULATED sheet costs one: the table is the
#: record and the call is spent on the header. A PLOTTED one costs both: one
#: per plot panel group for the traces, one for the header and the unsettled
#: list.
MAX_SOUNDING_CALLS = 2
#: About 110 dpi, as every other reader uses for the whole-page picture.
PAGE_DPI = 110.0
#: What ``zoom_plot`` renders at. A trace has to be read against its own tick
#: labels and the ticks on a CPT sheet are 2 mm apart on the paper.
ZOOM_DPI = 300.0

#: The depth step a PLOTTED cone sounding is digitised at when the sheet
#: states no reading interval of its own. Half a metre over a 20 m sounding
#: is 40 points, which is a profile a reviewer can use and a number of
#: readings one call can actually do carefully. A sheet that prints its own
#: grid spacing overrides it.
CPT_STEP_M = 0.5
#: The same for a dynamic probe. A DCP's own increment is 100 mm and a
#: dynamic probe's 200 mm, and a plotted one is read at the printed grid
#: where there is one; half a metre is the fallback.
DCP_STEP_M = 0.5
#: The finest step a plotted sounding is ever digitised at. A trace drawn at
#: 1 m to the inch cannot be read every centimetre however fine its grid.
MIN_STEP_M = 0.1
#: How many points one series may carry. A 30 m sounding tabulated at 20 mm
#: is 1,500 rows; past this the floor records the cap as a warning rather
#: than silently holding a truncated series.
MAX_SERIES_POINTS = 1200
#: How far outside an axis's printed range a digitised value may fall before
#: it is refused, as a fraction of one tick interval. A trace drawn hard
#: against the frame reads a hair past the last tick, and refusing that would
#: refuse the densest part of every sounding.
AXIS_SLACK_TICKS = 1.0

#: Text lines serialised per page of a sounding sheet.
MAX_LINES_PER_PAGE = 260
MAX_LINE_CHARS = 160
#: Rows of a tabulated series shown to the model. It is NOT asked to re-type
#: them; it is shown the head and the tail so it can see what the floor read
#: and say if the floor read the wrong columns.
SHOW_SERIES_ROWS = 12

#: How sure the floor is. A ruled table on a vector text layer is as good as
#: the page; a band of text lines read as a table is nearly as good; an axis
#: range read off a tick label is a reading of a label.
CONF_TABLE = 0.9
CONF_BANDS = 0.8
CONF_AXIS = 0.7
CONF_FIELD = 0.75
#: What a digitised point is worth: a clean trace, and one crossing another.
CONF_DIGITISED = 0.6
CONF_CROSSED = 0.35


# ---------------------------------------------------------------------------
# what a column of a sounding table is
# ---------------------------------------------------------------------------
#
# The vocabulary is small and deliberately so: these are the headings the
# corpus's own soundings print, in the four languages they are printed in.
# A heading that is in none of them is not guessed at -- the column is left
# unread and named in the floor's warnings, which a reviewer can act on.

#: ``role -> the headings that name it``, matched against a heading folded
#: to lowercase letters and digits. Longer names are tried first, so
#: ``depth_ref`` beats ``depth`` and ``sleeve friction`` beats ``friction``.
_COLUMN_VOCAB: Dict[str, Tuple[str, ...]] = {
    # where the reading is
    "depth": ("depth", "dm", "dft", "d", "z", "diepte", "profondeur",
              "prof", "profundidad", "profundidade", "tiefe", "depthm",
              "depthft", "depthbgs", "sondeerdiepte"),
    # "niveau", "level" and "altitude" are DELIBERATELY absent: they name a
    # header field on these sheets far more often than an axis ("Niveau
    # piezometrique", "Terrain altitude : 106,8 m"), and reading one as a
    # depth axis put an elevation scale on a resistance plot.
    "elevation": ("elevation", "elev", "cote", "peil", "elevacion",
                  "dref", "drefm", "niveaumaaiveld", "elevationm",
                  "elevationft"),
    # a cone sounding's three channels, and the ratio it prints beside them
    "qc": ("qc", "qt", "qcmpa", "coneresistance", "tipresistance",
           "conetipresistance", "correctedconeresistance",
           "resistancedepointe", "resistanceenpointe", "puntweerstand",
           "conusweerstand", "resistenciadepunta", "resistenciaporpunta",
           "spitzendruck", "qcbar", "qctsf"),
    "fs": ("fs", "sleevefriction", "localfriction", "localsidefriction",
           "frictionsleeve", "frottementlateral", "frottement",
           "plaatselijkewrijving", "wrijving", "mantelwrijving",
           "friccionlateral", "manteldruck", "fsmpa", "fskpa"),
    "u2": ("u2", "u", "porepressure", "porewaterpressure", "dynamicpore",
           "pressioninterstitielle", "waterspanning", "presionintersticial",
           "porenwasserdruck", "u2mpa", "u2kpa"),
    "rf": ("rf", "frictionratio", "rapportdefrottement", "wrijvingsgetal",
           "razondefriccion", "rfpct", "rf%"),
    "sbt": ("sbt", "sbtn", "soilbehaviourtype", "soilbehaviortype",
            "soiltype", "icsbt", "ic", "grondsoort", "typedesol"),
    # a dynamic probe's
    "blows": ("blows", "numberofblows", "nblows", "blowcount", "n",
              "n10", "n20", "n30", "nbdecoups", "nbredecoups",
              "nombredecoups", "coups", "nbdecoupspour", "golpes",
              "numerodegolpes", "schlagzahl", "slagen"),
    "penetration": ("penetration", "penetrationperblow", "mmblow",
                    "mmperblow", "enfoncement", "penetracion"),
    "index": ("dpi", "dn", "dcpindex", "penetrationindex", "rd", "qd",
              "rdmpa", "qdmpa", "resistancedynamique",
              "resistancedynamiqueapparente", "dynamicresistance",
              "resistdynamique", "resistdyn", "resistencedynamique",
              "sigmaa", "a"),
    "cbr": ("cbr", "cbrpct", "cbr%", "estimatedcbr"),
}

#: The order roles are tested in: the specific before the general, so
#: ``dref (m)`` is an elevation and not a depth, and ``N`` is a blow count
#: and not something a CPT column could claim.
_ROLE_ORDER: Tuple[str, ...] = (
    "elevation", "qc", "fs", "u2", "rf", "sbt", "cbr", "index",
    "penetration", "blows", "depth",
)

#: Which roles belong to which kind, so a DCP table's ``N`` column is never
#: read as a cone channel and a CPT's ``qc`` never as a dynamic resistance.
_KIND_ROLES: Dict[str, Tuple[str, ...]] = {
    "cpt": ("depth", "elevation", "qc", "fs", "u2", "rf", "sbt"),
    "dcp": ("depth", "elevation", "blows", "penetration", "index", "cbr"),
}

#: Where a heading may print its unit: inside parentheses or brackets
#: (``Qc (MPa)``), after a comma or a slash (``Depth, m``), or spelled out
#: at the end (``Cone resistance (qc) in MPa``, ``Resistance en Bar``).
#: EVERY candidate is validated before it is taken, because the commonest
#: thing inside a heading's parentheses is not a unit at all -- it is the
#: SYMBOL (``(qc)``, ``(Rf)``) or the datum (``(TAW)``), and a reader that
#: took those would record a tip resistance in qc.
_UNIT_SPOTS: Tuple["re.Pattern[str]", ...] = (
    re.compile(r"[\(\[]\s*([^)\]]{1,14})\s*[\)\]]"),
    re.compile(r"\b(?:in|en|em)\s+([A-Za-z%°/·^0-9.\-]{1,12})\s*$",
               re.I),
    re.compile(r"[,/]\s*([A-Za-z%°/0-9.\-]{1,10})\s*$"),
    # And spelled out in the MIDDLE of the title, which is how a depth axis
    # names its own unit: "Depth in m to reference level (TAW)".
    re.compile(r"\b(?:in|en|em)\s+([A-Za-z%°/0-9]{1,6})\b", re.I),
)
#: Units a sounding sheet prints that the record's conversion table has no
#: entry for, and which are still the unit the column says. A blow rate per
#: a stated increment and a penetration per blow are the two that matter.
_EXTRA_UNITS = re.compile(
    r"^(?:mm|cm|in)\s*/\s*blow$|^blows?\s*/.{1,8}$|^coups\s*/.{1,8}$"
    r"|^golpes\s*/.{1,8}$|^bar$|^mpa$|^kpa$", re.I)
#: A number as a sounding sheet writes one, decimal comma included.
_NUMBER = re.compile(r"^[-+]?\d{1,3}(?:[ .,]\d{3})*(?:[.,]\d+)?$|^[-+]?\d*[.,]?\d+$")
#: A value the sheet prints where it measured nothing.
_BLANK_CELL = frozenset({"", "-", "--", "/", "n/a", "na", ".", "*"})
#: How long a folded heading may be before a substring match is refused. A
#: column heading and an axis title are short; a caption is not.
MAX_HEADING_CHARS = 40
#: The same, on the printed text of an axis title.
MAX_AXIS_TITLE_CHARS = 45
#: How short an axis title may be when it prints NO unit. A title with a
#: unit is a title however short ("Depth (m)"); one without has to be a
#: word rather than a symbol.
MIN_AXIS_TITLE_CHARS = 6

ColumnRole = str


def _fold_head(text: str) -> str:
    """A column heading folded to plain letters and digits.

    ACCENTS ARE STRIPPED. Half these sheets are French, Spanish or
    Portuguese and their headings carry them -- ``Resistance dynamique
    apparente`` is printed ``Resistance`` with an acute accent, which is a
    letter Python calls alphanumeric and which no entry in the vocabulary
    has. Before this, the entire French dynamic-penetrometer family matched
    nothing at all.
    """
    raw = str(text or "")
    raw = re.sub(r"[\(\[][^)\]]{0,12}[\)\]]", " ", raw)
    raw = unicodedata.normalize("NFKD", raw)
    return "".join(ch.lower() for ch in raw
                   if ch.isalnum() and not unicodedata.combining(ch)
                   and ord(ch) < 128)


def column_role(heading: str, kind: str = "cpt") -> Optional[ColumnRole]:
    """What a printed column heading names, or None when nothing does.

    ``kind`` narrows the vocabulary: a dynamic probe's ``N`` is a blow count
    and a cone sounding has no such column, so asking with the wrong kind
    returns nothing rather than a wrong role.
    """
    key = _fold_head(heading)
    if not key:
        return None
    allowed = _KIND_ROLES.get(kind, tuple(_COLUMN_VOCAB))
    for role in _ROLE_ORDER:
        if role not in allowed:
            continue
        for name in _COLUMN_VOCAB[role]:
            if key == name:
                return role
    # A substring match, but only on something SHORT enough to be a heading.
    # A sentence is not a column heading: "Undrained shear strength
    # interpreted from cone resistance and pore pressure response" contains
    # "coneresistance" and is a figure's caption, and reading it as a qc
    # column put a tip-resistance axis on a plot of undrained shear strength.
    if len(key) > MAX_HEADING_CHARS:
        return None
    for role in _ROLE_ORDER:
        if role not in allowed:
            continue
        for name in _COLUMN_VOCAB[role]:
            if len(name) >= 4 and name in key:
                return role
    return None


def _unit_token(token: str) -> str:
    """One candidate as a unit, or an empty string when it is not one.

    A unit the record's conversion table knows, or one of the compound
    spellings a sounding sheet prints that no table converts. Anything else
    -- a symbol, a datum, a word -- is not a unit and is refused, because a
    wrong unit on a number is worse than no unit at all.
    """
    text = " ".join(str(token or "").split())
    if not text or len(text) > 14:
        return ""
    if _canonical_unit(text):
        return text
    if _EXTRA_UNITS.match(text):
        return text
    return ""


def head_unit(heading: str) -> str:
    """The unit a heading prints, as printed, or an empty string."""
    text = str(heading or "")
    whole = _unit_token(text)
    if whole:
        return whole     # a column headed nothing but its unit: "mm/blow"
    for pattern in _UNIT_SPOTS:
        for match in pattern.finditer(text):
            unit = _unit_token(match.group(1))
            if unit:
                return unit
    return ""


def _number(cell: Any) -> Optional[float]:
    """One cell as a number, decimal comma and thousands separator included."""
    text = " ".join(str(cell or "").split())
    if not text or text.lower() in _BLANK_CELL:
        return None
    text = text.replace("−", "-").replace("–", "-")
    if not _NUMBER.match(text):
        return None
    cleaned = text.replace(" ", "")
    if "," in cleaned and "." in cleaned:
        # Whichever comes last is the decimal mark; the other groups digits.
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        # A comma is a DECIMAL MARK unless the whole number is written as
        # thousands groups. "0,20" is a fifth on a Dutch sheet and "1,200" is
        # twelve hundred on an American one; "0,200" is a fifth, because
        # nobody writes two hundred with a leading zero.
        if re.fullmatch(r"[-+]?[1-9]\d{0,2}(?:,\d{3})+", cleaned):
            cleaned = cleaned.replace(",", "")
        else:
            cleaned = cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# a tabulated series
# ---------------------------------------------------------------------------

@dataclass
class TabularSeries:
    """A series read off a table: which column was what, and the rows."""

    roles: Dict[ColumnRole, int] = field(default_factory=dict)
    units: Dict[ColumnRole, str] = field(default_factory=dict)
    headings: Dict[ColumnRole, str] = field(default_factory=dict)
    rows: List[List[Any]] = field(default_factory=list)
    page: int = 0
    bbox: Optional[Tuple[float, float, float, float]] = None
    method: str = "tables"
    confidence: float = CONF_TABLE
    warnings: List[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    def value(self, row: Sequence[Any], role: ColumnRole) -> Optional[float]:
        i = self.roles.get(role)
        if i is None or i >= len(row):
            return None
        return _number(row[i])

    def text(self, row: Sequence[Any], role: ColumnRole) -> str:
        i = self.roles.get(role)
        if i is None or i >= len(row):
            return ""
        return " ".join(str(row[i] or "").split())


def _roles_of(headings: Sequence[Any], kind: str
              ) -> Tuple[Dict[ColumnRole, int], Dict[ColumnRole, str],
                         Dict[ColumnRole, str]]:
    roles: Dict[ColumnRole, int] = {}
    units: Dict[ColumnRole, str] = {}
    printed: Dict[ColumnRole, str] = {}
    for i, heading in enumerate(headings):
        role = column_role(heading, kind)
        if role is None or role in roles:
            continue
        roles[role] = i
        units[role] = head_unit(heading)
        printed[role] = " ".join(str(heading or "").split())
    return roles, units, printed


def _series_from_table(table: Any, kind: str) -> Optional[TabularSeries]:
    """One detected table as a series, or None when it is not one.

    A table is a series when its headings name a depth (or an elevation) and
    at least one measured column of this kind, and at least three of its rows
    carry a number under the depth column. Three, because two numbers is a
    header restated and a summary box.
    """
    headings = list(table.header or [])
    rows = [list(r) for r in (table.rows or [])]
    if not headings and rows:
        headings, rows = rows[0], rows[1:]
    roles, units, printed = _roles_of(headings, kind)
    depth_role = "depth" if "depth" in roles else (
        "elevation" if "elevation" in roles else None)
    measured = [r for r in roles if r not in ("depth", "elevation")]
    if depth_role is None or not measured:
        return None
    series = TabularSeries(
        roles=roles, units=units, headings=printed, rows=[],
        page=int(getattr(table, "page", 0) or 0),
        bbox=tuple(table.bbox) if getattr(table, "bbox", None) else None,
        method="tables", confidence=CONF_TABLE)
    for row in rows:
        if series.value(row, depth_role) is None:
            continue
        series.rows.append(row)
    return series if series.n_rows >= 3 else None


#: Two text lines are on the same printed ROW when their tops are within
#: this many points. A sounding table sets its depth column and its value
#: columns in one size, so this is tighter than a spreadsheet's.
BAND_TOL = 4.5
#: How many printed lines in a row may carry no depth before the series is
#: taken to have ENDED. One is not enough: a long table's depth cell drifts a
#: point or two off its own row's baseline as the page fills, so a row comes
#: apart into a line of values and a line holding its depth, and stopping at
#: the first of those cut a 56-row dynamic probe off at 6.8 m.
MAX_GAP_BANDS = 3
#: How far a cell's centre may sit outside EVERY heading's span and still be
#: claimed by the nearest of them. Containment is tried first, so this only
#: catches a number set just past the edge of its own heading.
COLUMN_SLACK = 40.0
#: How many consecutive printed lines one column header may run to. Three
#: covers every sheet in the corpus: the name, the rest of the name, and the
#: unit in its own line of parentheses.
MAX_HEADER_BANDS = 5


def _bands(lines: Sequence[Any]) -> List[List[Any]]:
    """Text lines grouped into printed rows by their tops."""
    ordered = sorted(lines, key=lambda ln: (round(float(ln.bbox[1]), 1),
                                            float(ln.bbox[0])))
    out: List[List[Any]] = []
    for line in ordered:
        top = float(line.bbox[1])
        if out and abs(float(out[-1][0].bbox[1]) - top) <= BAND_TOL:
            out[-1].append(line)
        else:
            out.append([line])
    for band in out:
        band.sort(key=lambda ln: float(ln.bbox[0]))
    return out


def _centre(line: Any) -> float:
    return (float(line.bbox[0]) + float(line.bbox[2])) / 2.0


def _columns_in(window: Sequence[Sequence[Any]]
                ) -> List[Tuple[float, float, str]]:
    """The window's lines clustered into columns: ``(x0, x1, joined text)``.

    A HEADER IS NOT A LINE. The corpus's French dynamic-probe sheets stagger
    their header cells over five printed lines -- "Nbre de | Resist. |
    Contrainte dyn." on one, the plot's own title on the next, "Profondeur"
    alone on the one after, then "coups | dynamique | Admissible", then
    "[m] | (daN/cm2) | (daN/cm2)" -- so no line of it names both a depth and
    a measured column, and a reader that took one line at a time found
    nothing at all on the whole family. Clustering the window's lines by
    their x OVERLAP puts each column's scattered cells back together.
    """
    flat = [ln for band in window for ln in band]
    flat.sort(key=lambda ln: float(ln.bbox[0]))
    clusters: List[List[Any]] = []
    for line in flat:
        x0, x1 = float(line.bbox[0]), float(line.bbox[2])
        if clusters and x0 <= max(float(l.bbox[2])
                                  for l in clusters[-1]) + 2.0:
            clusters[-1].append(line)
        else:
            clusters.append([line])
    out: List[Tuple[float, float, str]] = []
    for cluster in clusters:
        cluster.sort(key=lambda ln: (round(float(ln.bbox[1]), 1),
                                     float(ln.bbox[0])))
        out.append((min(float(l.bbox[0]) for l in cluster),
                    max(float(l.bbox[2]) for l in cluster),
                    " ".join(str(l.text or "").strip() for l in cluster)))
    return out


def _series_from_bands(lines: Sequence[Any], page: int, kind: str
                       ) -> Optional[TabularSeries]:
    """A series read off text lines that line up in columns.

    The corpus's tabulated soundings are printed with no ruling a detector
    can find -- a header of column names and two hundred lines of numbers
    under them -- so planlens reports no table and the geometry is still a
    table: a line of cells sitting under the header's cells IS a row of it.

    EVERY column of the header becomes a column, not only the ones this
    package has a name for. A dynamic probe's table prints Depth,
    Coefficient, Number of Blows, Rd and sigma-a; with only the three named
    columns in the frame, the Coefficient cells snapped into the nearest
    named neighbour and were recorded as blow counts. A column with no role
    is read and then ignored, which is the only way its numbers cannot
    become somebody else's.

    A cell belongs to the column whose HEADING SPAN it falls under, and to
    the nearest heading centre only when it falls under none. A heading is
    wide and its numbers are set left or right inside it, so a centre
    distance alone loses a column.
    """
    bands = _bands(lines)
    best: Optional[TabularSeries] = None
    best_rows = 0
    best_score: Tuple[int, int, int] = (0, 0, 0)
    for i in range(len(bands)):
        for depth_of_header in range(1, MAX_HEADER_BANDS + 1):
            if i + depth_of_header > len(bands):
                break
            columns = _columns_in(bands[i:i + depth_of_header])
            if len(columns) < 2:
                continue
            roles, units, printed = _roles_of([c[2] for c in columns], kind)
            depth_role = "depth" if "depth" in roles else (
                "elevation" if "elevation" in roles else None)
            measured = [r for r in roles if r not in ("depth", "elevation")]
            if depth_role is None or not measured:
                continue
            spans = [(x0, x1, (x0 + x1) / 2.0) for x0, x1, _t in columns]
            band = bands[i]
            series = TabularSeries(
                roles=dict(roles), units=dict(units), headings=dict(printed),
                rows=[], page=int(page), method="text",
                confidence=CONF_BANDS,
                bbox=(min(float(ln.bbox[0]) for ln in band),
                      min(float(ln.bbox[1]) for ln in band),
                      max(float(ln.bbox[2]) for ln in band),
                      max(float(ln.bbox[3]) for ln in band)))
            leading: List[List[Any]] = []
            gap = 0
            for below in bands[i + depth_of_header:]:
                row: List[Any] = [None] * len(spans)
                placed = 0
                for line in below:
                    x = _centre(line)
                    under = [n for n, (x0, x1, _c) in enumerate(spans)
                             if x0 - 4.0 <= x <= x1 + 4.0]
                    if under:
                        near = min(under, key=lambda n: abs(spans[n][2] - x))
                    else:
                        near = min(range(len(spans)),
                                   key=lambda n: abs(spans[n][2] - x))
                        if abs(spans[near][2] - x) > COLUMN_SLACK:
                            continue
                    if row[near] is None:
                        row[near] = line.text
                        placed += 1
                if not placed:
                    continue
                if series.value(row, depth_role) is None:
                    # A line with no number under the depth column is a
                    # footer, a note, the next block's title -- or a row
                    # whose depth cell drifted onto its own printed line.
                    # The series ends after MAX_GAP_BANDS of them in a row.
                    if series.n_rows >= 3:
                        gap += 1
                        if gap >= MAX_GAP_BANDS:
                            break
                        continue
                    leading.append(row)
                    continue
                gap = 0
                series.rows.append(row)
            # THE UNIT LINE. A sheet that sets its column names on one line
            # and their units on the next leaves the header window with no
            # unit in it. Any line BEFORE the first reading that carries no
            # depth is part of the heading, so it is folded back in and the
            # units re-read -- which is how "Profondeur" gets its "[m]" and
            # a resistance column gets its "(daN/cm2)".
            if leading and not all(series.units.get(r) for r in roles):
                merged = [c[2] for c in columns]
                for row in leading:
                    for n, cell in enumerate(row):
                        if cell:
                            merged[n] = (merged[n] + " "
                                         + str(cell).strip()).strip()
                _r, again, _p = _roles_of(merged, kind)
                for role, unit in again.items():
                    if unit and not series.units.get(role):
                        series.units[role] = unit
            # A ROW IS ONLY A READING IF IT CARRIES ONE. Clustering a wide
            # window of a busy page can assemble a "header" out of a project
            # block and a plot title, with three lines of prose under it that
            # happen to start with a number. Counting only the rows that
            # carry a NUMBER UNDER A MEASURED COLUMN throws those out, and
            # taking the candidate with the most such rows picks the real
            # table on a page that holds more than one thing.
            good = sum(
                1 for row in series.rows
                if any(series.value(row, r) is not None for r in measured))
            # MORE READINGS FIRST, then MORE COLUMNS NAMED, then the
            # NARROWEST header that found them. A wide window on a busy page
            # can assemble a header out of a project block and a plot title
            # and still land on the real rows, so narrowness breaks a tie;
            # but a header that names a depth, a blow count AND a resistance
            # beats one that names only the first two, however wide it had
            # to reach to do it.
            # A COLUMN THAT CARRIES NO NUMBERS IS NOT A COLUMN. The plot
            # drawn beside a French dynamic-probe table is titled
            # "RESISTANCE DYNAMIQUE (daN/cm2)", which names a role as
            # squarely as the table's own "Resist. dynamique" heading does
            # and has not one reading under it. Counting only the roles that
            # actually carry readings is what tells the two apart.
            filled = sum(1 for role in roles
                         if sum(1 for row in series.rows
                                if series.value(row, role) is not None) >= 3)
            score = (good, filled, -depth_of_header)
            if good >= 3 and score > best_score:
                best, best_score, best_rows = series, score, good
    return best

def read_tabulated(doc: Any, pages: Sequence[int], kind: str
                   ) -> Tuple[Optional[TabularSeries], List[str]]:
    """The sheet's series as a table, and what could not be read.

    Detected tables first, then the text lines banded by their geometry. The
    FIRST series found wins and later pages EXTEND it where their columns
    agree, because a sounding tabulated over three pages is one sounding.
    """
    warnings: List[str] = []
    found: Optional[TabularSeries] = None
    for page in pages:
        try:
            content = doc.page(page)
        except Exception as exc:                 # a page that will not read
            warnings.append(f"page {page}: {type(exc).__name__}: {exc}")
            continue
        here: Optional[TabularSeries] = None
        for table in getattr(content, "tables", ()) or ():
            here = _series_from_table(table, kind)
            if here is not None:
                break
        if here is None:
            here = _series_from_bands(getattr(content, "lines", ()) or (),
                                      page, kind)
        if here is None:
            continue
        if found is None:
            found = here
            continue
        if set(found.roles) == set(here.roles):
            found.rows.extend(here.rows)
        else:
            warnings.append(
                f"page {page}: a second table with different columns "
                f"({sorted(here.roles)} against {sorted(found.roles)}); its "
                f"{here.n_rows} row(s) were not folded into the series")
    if found is not None and found.n_rows > MAX_SERIES_POINTS:
        warnings.append(
            f"the tabulated series runs to {found.n_rows} rows and was cut "
            f"to {MAX_SERIES_POINTS}; the deepest rows are not in the record")
        found.rows = found.rows[:MAX_SERIES_POINTS]
    return found, warnings


# ---------------------------------------------------------------------------
# a plotted sheet's axes
# ---------------------------------------------------------------------------

@dataclass
class PlotAxis:
    """One axis of a plotted sounding: what it measures, and its range.

    ``ticks`` are the tick VALUES read off the axis, in the order they were
    printed. They are what makes a digitised reading checkable and what
    gives the tolerance the scorer uses: one tick.
    """

    role: ColumnRole
    unit: str = ""
    title: str = ""
    ticks: List[float] = field(default_factory=list)
    page: int = 0
    bbox: Optional[Tuple[float, float, float, float]] = None

    @property
    def low(self) -> Optional[float]:
        return min(self.ticks) if self.ticks else None

    @property
    def high(self) -> Optional[float]:
        return max(self.ticks) if self.ticks else None

    @property
    def tick(self) -> Optional[float]:
        """The interval between ticks, where they are evenly spaced."""
        if len(self.ticks) < 2:
            return None
        values = sorted(self.ticks)
        steps = [b - a for a, b in zip(values, values[1:]) if b > a]
        if not steps:
            return None
        return min(steps)

    def holds(self, value: float) -> bool:
        """Is this value inside the axis, allowing one tick of slack?"""
        if self.low is None or self.high is None:
            return True
        step = self.tick or (abs(self.high - self.low) / 10.0 or 1.0)
        slack = AXIS_SLACK_TICKS * step
        return (self.low - slack) <= value <= (self.high + slack)

    def to_dict(self) -> Dict[str, Any]:
        return {"role": self.role, "unit": self.unit, "title": self.title,
                "low": self.low, "high": self.high, "tick": self.tick,
                "page": self.page,
                "ticks": [round(t, 4) for t in self.ticks[:40]]}


#: How far from an axis title its tick labels may sit, ACROSS the axis, in
#: page points. Narrow on purpose: a CPT sheet sets two scales on the same
#: printed line -- cone resistance rising to the right, friction ratio
#: falling to the right -- and a wide band hands one axis the other's ticks.
TICK_BAND_PT = 26.0
#: The same along a vertical axis, whose title is set rotated beside it.
TICK_COLUMN_PT = 60.0
#: Fewest tick labels an axis must carry. Two numbers is a caption.
MIN_TICKS = 3
#: How far a tick scale's intervals may vary and still be a SCALE. A printed
#: axis is evenly spaced, linearly or in decades; three numbers that happen
#: to rise -- a cone's area, its sleeve's area, a page number -- are not.
#: This is what keeps the little cone symbol printed in the corner of every
#: Dutch sounding from being recorded as a pore-pressure axis.
TICK_REGULARITY = 0.25


def _regular(ticks: Sequence[float], linear_only: bool = False) -> bool:
    """Is this run of tick values an evenly spaced scale, linear or log?

    ``linear_only`` for a DEPTH axis, which is always linear. A run of 1,
    10, 100, 1000 is a perfectly good resistance scale and is never a depth
    scale, and accepting one as a depth scale put a log resistance axis on
    the depth of a French dynamic penetrometer.
    """
    values = sorted(float(t) for t in ticks)
    if len(values) < MIN_TICKS:
        return False
    steps = [b - a for a, b in zip(values, values[1:])]
    if all(s > 0 for s in steps):
        mean = sum(steps) / len(steps)
        if mean > 0 and max(abs(s - mean) for s in steps) \
                <= TICK_REGULARITY * mean:
            return True
    if values[0] > 0 and not linear_only:
        logs = [math.log10(v) for v in values]
        steps = [b - a for a, b in zip(logs, logs[1:])]
        if all(s > 0 for s in steps):
            mean = sum(steps) / len(steps)
            if mean > 0 and max(abs(s - mean) for s in steps) \
                    <= TICK_REGULARITY * mean:
                return True
    return False


def _run(values: Sequence[Tuple[float, float]]) -> List[float]:
    """The longest monotone run of ``(position, value)``, in position order.

    A tick scale is monotone along its own axis, rising or falling, and a
    stray number that is not on the scale breaks the run rather than joining
    it. So the longest monotone run of the numbers in a band IS the scale.
    """
    ordered = [v for _p, v in sorted(values, key=lambda pv: pv[0])]
    best: List[float] = []
    for direction in (1, -1):
        current: List[float] = []
        for value in ordered:
            if not current or direction * (value - current[-1]) > 0:
                current.append(value)
            else:
                if len(current) > len(best):
                    best = current
                current = [value]
        if len(current) > len(best):
            best = current
    return best


def _tick_candidates(title: Any, numeric: Sequence[Tuple[Any, float]],
                     linear_only: bool = False
                     ) -> Tuple[List[Any], List[float], float]:
    """``(the lines, their values, how far the title is from them)``.

    Both orientations are tried and the one with the longer monotone run
    wins, so a depth scale down the left of a sheet and a resistance scale
    along its top are found by the same code.
    """
    tx0, ty0, tx1, ty1 = (float(v) for v in title.bbox)
    tcx, tcy = (tx0 + tx1) / 2.0, (ty0 + ty1) / 2.0

    def band(across: bool) -> List[Tuple[float, float, Any]]:
        out: List[Tuple[float, float, Any]] = []
        for line, value in numeric:
            if line is title:
                continue
            cx = (float(line.bbox[0]) + float(line.bbox[2])) / 2.0
            cy = (float(line.bbox[1]) + float(line.bbox[3])) / 2.0
            if across and abs(cy - tcy) <= TICK_BAND_PT:
                out.append((cx, value, line))
            elif not across and abs(cx - tcx) <= TICK_COLUMN_PT:
                out.append((cy, value, line))
        return out

    best: Tuple[List[Any], List[float], float] = ([], [], 1e9)
    for across in (True, False):
        candidates = band(across)
        if len(candidates) < MIN_TICKS:
            continue
        run = _run([(p, v) for p, v, _ln in candidates])
        if len(run) < MIN_TICKS or not _regular(run, linear_only):
            continue
        wanted = list(run)
        lines: List[Any] = []
        for _p, value, line in sorted(candidates, key=lambda c: c[0]):
            if wanted and value == wanted[0]:
                wanted.pop(0)
                lines.append(line)
        centre = sum(((float(ln.bbox[0]) + float(ln.bbox[2])) / 2.0
                      if across else
                      (float(ln.bbox[1]) + float(ln.bbox[3])) / 2.0)
                     for ln in lines) / max(1, len(lines))
        distance = abs(centre - (tcx if across else tcy))
        if len(run) > len(best[1]) or (len(run) == len(best[1])
                                       and distance < best[2]):
            best = (lines, run, distance)
    return best


def axes_from_text(doc: Any, pages: Sequence[int], kind: str
                   ) -> Tuple[List[PlotAxis], List[str]]:
    """The axis ranges a plotted sheet PRINTS, off its own text.

    An axis is found where a line NAMES one of this kind's measured
    quantities and a run of at least three numbers stands in a narrow band
    beside it. Those numbers are the tick labels and their span is the axis
    range.

    A TICK LABEL BELONGS TO ONE AXIS. A cone sounding prints its tip
    resistance and its friction ratio along the SAME printed line, one
    rising to the right and one falling, and the first version of this
    function gave both scales to whichever title it met first -- so a
    friction ratio that runs 10 down to 2 was recorded as running 2 up to 20.
    Titles now claim their ticks exclusively, nearest first, and each
    remaining title is re-read against what is left.

    Nothing here looks at the picture. That is the point: the ranges come
    off the text layer, so the digitising the model does afterwards is
    checked against something the model did not supply.
    """
    out: List[PlotAxis] = []
    warnings: List[str] = []
    for page in pages:
        try:
            content = doc.page(page)
        except Exception as exc:
            warnings.append(f"page {page}: {type(exc).__name__}: {exc}")
            continue
        lines = list(getattr(content, "lines", ()) or ())
        numeric = [(ln, _number(ln.text)) for ln in lines]
        numeric = [(ln, v) for ln, v in numeric if v is not None]
        titles: List[Tuple[Any, str, str]] = []
        for line in lines:
            text = " ".join(str(line.text or "").split())
            if not text or len(text) > MAX_AXIS_TITLE_CHARS:
                continue
            role = column_role(text, kind)
            if role is None:
                continue
            unit = head_unit(text)
            if not unit and len(text) < MIN_AXIS_TITLE_CHARS:
                # A bare symbol is not an axis title. Every Dutch sounding
                # prints "u2" inside the little cone diagram in its corner,
                # beside the cone's area and its sleeve's area -- three
                # rising numbers, which is a scale to anything that does not
                # ask whether the label was a title in the first place.
                continue
            if any(t[1] == role for t in titles):
                continue
            titles.append((line, role, unit))
        claimed: set = set()
        remaining = list(titles)
        while remaining:
            best = None
            for entry in remaining:
                line, role, unit = entry
                pool = [(ln, v) for ln, v in numeric if id(ln) not in claimed]
                found, run, distance = _tick_candidates(
                    line, pool, linear_only=role in ("depth", "elevation"))
                if len(run) < MIN_TICKS:
                    continue
                if best is None or distance < best[0]:
                    best = (distance, entry, found, run)
            if best is None:
                break
            _distance, entry, found, run = best
            line, role, unit = entry
            remaining.remove(entry)
            for one in found:
                claimed.add(id(one))
            if any(a.role == role for a in out):
                continue
            out.append(PlotAxis(
                role=role, unit=unit,
                title=" ".join(str(line.text or "").split()),
                ticks=run, page=int(page), bbox=tuple(line.bbox)))
    if not out:
        warnings.append(
            "no axis with a readable range was found on these pages; a "
            "digitised value cannot be checked against the paper")
    return out, warnings


# ---------------------------------------------------------------------------
# the floor
# ---------------------------------------------------------------------------

#: Header fields a sounding sheet prints, in the corpus's four languages.
#: The grid's own vocabulary is for boreholes and misses every one of them.
_HEADER_KEYS: Dict[str, Tuple[str, ...]] = {
    "sounding_id": ("cpt no", "cpt number", "sounding no", "sounding number",
                    "test no", "dcp no", "dpt no", "sondage", "sondeo",
                    "no du sondage", "n du sondage", "sondering",
                    "numero de sondeo", "essai no", "probe no"),
    "project": ("project", "projet", "proyecto", "projectnummer",
                "project no", "project number", "projectnumber",
                "no de dossier", "n de dossier", "affaire"),
    "client": ("client", "cliente", "opdrachtgever", "customer", "owner"),
    "location": ("location", "site", "lieu", "ubicacion", "implantation",
                 "omschrijving", "locatie"),
    # "Datum" is the DATE in Dutch and in German and is printed on half the
    # Continental soundings in the corpus. The vertical datum is named
    # below, by the phrases that actually mean one.
    "date": ("date", "datum", "fecha", "data", "test date", "date of test",
             "date du forage", "date realised", "date de l essai"),
    "ground_level": ("ground level", "gl", "surface elevation",
                     "niveau maaiveld", "terrain altitude", "cote du sol",
                     "maaiveld", "ground surface elevation",
                     "elevation of ground"),
    "datum": ("reference level", "referentiepunt", "reference point",
              "vertical datum", "niveau de reference", "elevation datum",
              "nivel de referencia"),
    "water": ("water level", "groundwater", "grondwaterpeil",
              "niveau piezometrique", "nivel freatico", "water table",
              "assumed ground water level", "niveau de l eau",
              "niveau de leau"),
    "cone": ("cone no", "cone number", "cone", "conus", "penetrometer",
             "cone type", "type de pointe", "tip"),
    "standard": ("test according", "standard", "norme", "norma", "method",
                 "according to", "conformement a la norme"),
    "hammer": ("hammer", "marteau", "hammer mass", "hammer weight",
               "masse du mouton", "mouton"),
    "increment": ("increment", "blows per", "nb de coups pour",
                  "coups pour", "penetration increment"),
    "refusal": ("refusal", "refus", "rechazo", "weigering", "arret"),
}
_HEADER_PAIR = re.compile(r"^\s*([^:=]{2,44}?)\s*[:=]\s*(.+?)\s*$")


def _header_fields(doc: Any, pages: Sequence[int]) -> Dict[str, Any]:
    """``key -> (value, page, bbox)`` for the header fields these pages print."""
    out: Dict[str, Any] = {}
    extra: Dict[str, Any] = {}
    for page in pages:
        try:
            content = doc.page(page)
        except Exception:
            continue
        for line in getattr(content, "lines", ()) or ():
            text = " ".join(str(line.text or "").split())
            if not text or len(text) > 120:
                continue
            pair = _HEADER_PAIR.match(text)
            if pair is None:
                continue
            label = "".join(ch.lower() if ch.isalnum() else " "
                            for ch in pair.group(1))
            label = " ".join(label.split())
            value = pair.group(2).strip()
            if not value or len(value) > 120:
                continue
            key = None
            for name, phrases in _HEADER_KEYS.items():
                if label in phrases:
                    key = name
                    break
            if key is None:
                for name, phrases in _HEADER_KEYS.items():
                    if any(p in label for p in phrases if len(p) >= 5):
                        key = name
                        break
            target = out if key else extra
            target.setdefault(key or pair.group(1).strip(),
                              (value, int(page), tuple(line.bbox)))
    out["_extra"] = extra
    return out


#: A sounding's own identifier as a title block prints it with no label at
#: all: ``S1.1``, ``CPT-4``, ``DPT-1``, ``B - 2``, ``SC03``. Letters then
#: digits, optionally separated, and short.
_BARE_ID = re.compile(r"^[A-Za-z]{1,4}\s?[-.–]?\s?\d{1,3}(?:[.\-]\d{1,2})?$")
#: Words that look like an identifier and are not one.
_NOT_AN_ID = frozenset({"page", "sheet", "fig", "figure", "no", "nr", "n",
                        "rev", "table", "appendix", "annex", "vol"})
#: How far down the page the unlabelled identifier may sit, as a fraction of
#: the page height. A title block is at the top of the sheet.
ID_BAND_FRAC = 0.2


def bare_identifier(doc: Any, page: int) -> Optional[Tuple[str, Any]]:
    """The identifier a title block prints with no label, or None.

    Half the corpus's sounding sheets never write "CPT no."; they set the
    sounding's name alone and large at the top of the page. The labelled
    fields are read first and this is only reached when they found nothing,
    so a sheet that does say is never second-guessed.
    """
    try:
        content = doc.page(page)
    except Exception:
        return None
    height = float(getattr(content, "height", 0.0) or 0.0)
    cut = height * ID_BAND_FRAC if height else 0.0
    lines = sorted(getattr(content, "lines", ()) or (),
                   key=lambda ln: (round(float(ln.bbox[1]), 1),
                                   float(ln.bbox[0])))
    for line in lines:
        if cut and float(line.bbox[1]) > cut:
            break
        text = " ".join(str(line.text or "").split())
        if not text or len(text) > 12:
            continue
        if text.lower().strip(".:") in _NOT_AN_ID:
            continue
        if _BARE_ID.match(text) and any(ch.isdigit() for ch in text) \
                and any(ch.isalpha() for ch in text):
            return " ".join(text.split()), line
    return None


@dataclass
class SoundingFloor:
    """What these pages say before a model is called.

    ``tabulated`` is the fork the whole reader turns on: a table means the
    numbers are already the record's and the call goes on the header; no
    table means the traces have to be digitised and the axes are what will
    check them.
    """

    kind: str = "cpt"
    investigation: Optional[Investigation] = None
    series: Optional[TabularSeries] = None
    axes: List[PlotAxis] = field(default_factory=list)
    fields: Dict[str, Any] = field(default_factory=dict)
    pages: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def tabulated(self) -> bool:
        return self.series is not None and self.series.n_rows >= 3

    @property
    def n_points(self) -> int:
        data = None if self.investigation is None else (
            self.investigation.cpt or self.investigation.dcp)
        return 0 if data is None else data.n_points

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "tabulated": self.tabulated,
            "n_points": self.n_points,
            "axes": [a.to_dict() for a in self.axes],
            "pages": list(self.pages),
            "warnings": list(self.warnings),
            "investigation": (self.investigation.model_dump(mode="json")
                              if self.investigation is not None else None),
        }


def _depth_unit_of(series: Optional[TabularSeries],
                   axes: Sequence[PlotAxis], fields: Dict[str, Any]) -> str:
    """The unit every depth in this sounding is in, or an empty string."""
    for role in ("depth", "elevation"):
        if series is not None and role in series.units:
            unit = series.units.get(role) or ""
            if _canonical_unit(unit) and _canonical_unit(unit) in (
                    "m", "ft", "cm", "mm"):
                return unit
        for axis in axes:
            if axis.role == role and _canonical_unit(axis.unit) in (
                    "m", "ft", "cm", "mm"):
                return axis.unit
    return ""


def sounding_floor(doc: Any, pages: Sequence[int], kind: str = "cpt",
                   report_id: str = "") -> SoundingFloor:
    """What a sounding sheet states, read with patterns and no model.

    The header fields, the series where the sheet TABULATES one, and the
    axis ranges where it PLOTS one. Nothing is inferred: a cone area is
    recorded only where one is printed, an identifier only where the sheet
    names one, a refusal only where the sheet says so.
    """
    pages = [int(p) for p in pages]
    kind = kind if kind in SOUNDING_KINDS else "cpt"
    floor = SoundingFloor(kind=kind, pages=pages)
    if not pages:
        return floor
    series, warnings = read_tabulated(doc, pages, kind)
    floor.series = series
    floor.warnings.extend(warnings)
    axes: List[PlotAxis] = []
    if series is None:
        axes, axis_warnings = axes_from_text(doc, pages, kind)
        floor.warnings.extend(axis_warnings)
    floor.axes = axes
    fields = _header_fields(doc, pages)
    floor.fields = fields
    unit = _depth_unit_of(series, axes, fields)

    def field_value(key: str) -> Tuple[str, Optional[Provenance]]:
        got = fields.get(key)
        if not got:
            return "", None
        value, page, bbox = got
        return value, Provenance(page=page, bbox=bbox, method="text",
                                 confidence=CONF_FIELD,
                                 note=f"header field {key}")

    identifier, id_prov = field_value("sounding_id")
    if not identifier:
        bare = bare_identifier(doc, pages[0])
        if bare is not None:
            identifier, line = bare
            id_prov = Provenance(
                page=pages[0], bbox=tuple(line.bbox), method="text",
                confidence=0.5,
                note="the title block names it, with no field label")
    date, _ = field_value("date")
    datum, _ = field_value("datum")
    level, level_prov = field_value("ground_level")
    cone, cone_prov = field_value("cone")
    standard, _ = field_value("standard")
    hammer, _ = field_value("hammer")

    elevation: Optional[Quantity] = None
    number = _number(level.split()[0]) if level.split() else None
    if number is not None and unit:
        elevation = Quantity(value=number, unit=unit, prov=level_prov)

    investigation = Investigation(
        investigation_id=identifier,
        kind=kind,  # type: ignore[arg-type]
        depth_unit=unit, units_known=bool(unit),
        elevation=elevation,
        date_started=date,
        drilling=DrillingDetails(
            method=standard, hammer_type=hammer,
            prov=[p for p in (id_prov,) if p is not None]),
        pages=pages, source_report=report_id,
        fields={k: v[0] for k, v in fields.get("_extra", {}).items()
                if isinstance(v, tuple)},
        prov=[p for p in (id_prov,) if p is not None])
    if datum:
        investigation.fields.setdefault("datum", datum)
    water, water_prov = field_value("water")
    if water:
        investigation.fields.setdefault("water_level_as_printed", water)

    data = _floor_series(series, axes, kind, unit, cone, standard,
                         cone_prov, floor)
    if kind == "cpt":
        investigation.cpt = data  # type: ignore[assignment]
        if data is not None and axes:
            investigation.cpt.vertical_axis = (
                "elevation" if any(a.role == "elevation" for a in axes)
                and not any(a.role == "depth" for a in axes) else "depth")
            investigation.cpt.datum = datum
    else:
        investigation.dcp = data  # type: ignore[assignment]
    floor.investigation = investigation
    return floor


def _floor_series(series: Optional[TabularSeries], axes: Sequence[PlotAxis],
                  kind: str, unit: str, cone: str, standard: str,
                  cone_prov: Optional[Provenance],
                  floor: SoundingFloor) -> Optional[Any]:
    """The floor's own series object, tabulated or empty with the axes on it."""
    if kind == "cpt":
        data = CPTData(cone_type=cone, standard=standard,
                       prov=[p for p in (cone_prov,) if p is not None])
    else:
        data = DCPData(test_type=cone, standard=standard,
                       prov=[p for p in (cone_prov,) if p is not None])
    if series is None:
        # A PLOTTED sheet's floor holds no points. What it does hold is the
        # unit each axis is printed in, which is the one part of a plotted
        # series a pattern can read exactly.
        for axis in axes:
            name = f"{axis.role}_unit"
            if axis.unit and hasattr(data, name):
                setattr(data, name, axis.unit)
        return data
    if not unit:
        floor.warnings.append(
            "the tabulated series names no depth unit, so its depths have no "
            "meaning and no point was placed")
        return data
    depth_role = "depth" if "depth" in series.roles else "elevation"
    if kind == "cpt":
        data.vertical_axis = "elevation" if depth_role == "elevation" \
            else "depth"
        data.qc_unit = series.units.get("qc", "")
        data.fs_unit = series.units.get("fs", "")
        data.u2_unit = series.units.get("u2", "")
    steps: List[float] = []
    last: Optional[float] = None
    for row in series.rows:
        depth = series.value(row, depth_role)
        if depth is None:
            continue
        prov = Provenance(
            page=series.page, bbox=series.bbox, method=series.method,  # type: ignore[arg-type]
            confidence=series.confidence,
            note=f"tabulated row under {series.headings.get(depth_role, '')}")
        if last is not None and depth != last:
            steps.append(abs(depth - last))
        last = depth
        if kind == "cpt":
            point = CPTPoint(
                depth=Quantity(value=depth, unit=unit),
                qc=_quantity(series.value(row, "qc"), series.units.get("qc")),
                fs=_quantity(series.value(row, "fs"), series.units.get("fs")),
                u2=_quantity(series.value(row, "u2"),
                             series.units.get("u2")),
                rf_percent=series.value(row, "rf"),
                sbt=series.text(row, "sbt"),
                prov=prov)
            if point.qc is None and point.fs is None and point.u2 is None \
                    and point.rf_percent is None and not point.sbt:
                continue
            data.points.append(point)
        else:
            blows = series.value(row, "blows")
            point = DCPPoint(
                depth=Quantity(value=depth, unit=unit),
                blows=blows,
                penetration=_quantity(series.value(row, "penetration"),
                                      series.units.get("penetration")),
                index=_quantity(series.value(row, "index"),
                                series.units.get("index")),
                cbr_percent=series.value(row, "cbr"),
                prov=prov)
            if point.blows is None and point.penetration is None \
                    and point.index is None and point.cbr_percent is None:
                continue
            data.points.append(point)
    if kind == "dcp":
        data.index_name = series.headings.get("index", "")
    if steps:
        # The step the sheet actually printed, which is the MODE of the gaps
        # and not their mean: a sheet with one missing row would otherwise
        # report a step no row stands at.
        rounded = [round(s, 3) for s in steps]
        common = max(set(rounded), key=rounded.count)
        if common > 0:
            data.step = Quantity(value=common, unit=unit)
            # On a cone sounding the step is the reading interval. On a
            # dynamic probe it is the INCREMENT the blows were counted over,
            # which is a different field and a different fact -- and it is
            # only taken where the sheet did not print one of its own.
            if isinstance(data, CPTData):
                data.depth_interval = Quantity(value=common, unit=unit)
            elif data.increment is None:
                data.increment = Quantity(value=common, unit=unit)
    return data


def _quantity(value: Optional[float], unit: Optional[str]
              ) -> Optional[Quantity]:
    if value is None:
        return None
    return Quantity(value=float(value), unit=(unit or "").strip())


def serialise_floor(floor: SoundingFloor) -> str:
    """The floor as compact lines the model can build on and cite."""
    out: List[str] = []
    inv = floor.investigation
    if inv is not None:
        head = [f"identifier '{inv.investigation_id}'"
                if inv.investigation_id else "identifier: not found",
                f"depth unit {inv.depth_unit or 'NOT STATED'}"]
        if inv.elevation is not None:
            head.append(f"ground level {show(inv.elevation)}")
        if inv.date_started:
            head.append(f"date '{inv.date_started}'")
        out.append("header: " + "; ".join(head))
        for key, value in sorted(inv.fields.items()):
            out.append(f"header field {key} = '{str(value)[:80]}'")
    if floor.tabulated and floor.series is not None:
        series = floor.series
        out.append(
            f"THE SHEET TABULATES ITS SERIES: {series.n_rows} row(s) on page "
            f"{series.page}, columns "
            + ", ".join(f"{role} ('{series.headings.get(role, '')}')"
                        for role in series.roles)
            + f" -- read by {series.method}, confidence "
              f"{series.confidence:.2f}.")
        data = None if inv is None else (inv.cpt or inv.dcp)
        if data is not None and data.points:
            out.append(f"the floor placed {data.n_points} point(s)"
                       + (f" at a step of {show(data.step)}"
                          if data.step is not None else "") + ":")
            shown = data.points[:SHOW_SERIES_ROWS // 2] \
                + data.points[-(SHOW_SERIES_ROWS // 2):]
            for point in shown:
                out.append("  " + _point_line(point))
            if data.n_points > SHOW_SERIES_ROWS:
                out.append(f"  [{data.n_points - SHOW_SERIES_ROWS} further "
                           f"point(s) not listed; they are IN the record]")
    else:
        out.append("THE SHEET PLOTS ITS SERIES; no table of values was found.")
        if floor.axes:
            out.append("the axes the sheet's own text states:")
            for axis in floor.axes:
                out.append(
                    f"  {axis.role}: '{axis.title}' unit '{axis.unit}' "
                    f"from {axis.low:g} to {axis.high:g}"
                    + (f", tick {axis.tick:g}" if axis.tick else "")
                    + f" (page {axis.page})")
        else:
            out.append("  no axis range could be read off the text.")
    for warning in floor.warnings:
        out.append(f"WARNING: {warning}")
    return "\n".join(out) or "(nothing was read off these pages)"


def _point_line(point: Any) -> str:
    if isinstance(point, CPTPoint):
        parts = [f"depth {show(point.depth)}"]
        for name in ("qc", "fs", "u2"):
            value = getattr(point, name)
            if value is not None:
                parts.append(f"{name} {show(value)}")
        if point.rf_percent is not None:
            parts.append(f"Rf {point.rf_percent:g} %")
        if point.sbt:
            parts.append(f"SBT '{point.sbt}'")
    else:
        parts = [f"depth {show(point.depth)}"]
        if point.blows is not None:
            parts.append(f"blows {point.blows:g}")
        if point.penetration is not None:
            parts.append(f"penetration {show(point.penetration)}")
        if point.index is not None:
            parts.append(f"index {show(point.index)}")
        if point.cbr_percent is not None:
            parts.append(f"CBR {point.cbr_percent:g} %")
    confidence = ("" if point.prov is None
                  else f"  [confidence {point.prov.confidence:.2f}]")
    return ", ".join(parts) + confidence


# ---------------------------------------------------------------------------
# what the model returns
# ---------------------------------------------------------------------------

class ReadProv(BaseModel):
    """Where the model says it read a value."""

    model_config = ConfigDict(extra="forbid")

    page: int = Field(description="0-based PDF page index")
    bbox: Optional[Tuple[float, float, float, float]] = Field(
        default=None,
        description="the box of the line or cell this came from, copied "
                    "exactly; null when you read it off the picture")
    from_image: bool = Field(
        default=False,
        description="true when the picture, not the text, is what told you")
    note: str = Field(
        default="",
        description="what was ambiguous and what settled it, 15 words or "
                    "fewer; empty when nothing was")


class ReadPoint(BaseModel):
    """One depth of the series, as the model read it."""

    model_config = ConfigDict(extra="forbid")

    depth: float = Field(
        description="the vertical-axis value at this point, in the depth "
                    "unit you stated")
    qc: Optional[float] = Field(
        default=None, description="tip resistance, in the qc unit you stated")
    fs: Optional[float] = Field(
        default=None, description="sleeve friction, in the fs unit stated")
    u2: Optional[float] = Field(
        default=None, description="pore pressure, in the u2 unit stated")
    rf_percent: Optional[float] = Field(
        default=None, description="friction ratio as a percent, when the "
                                  "sheet plots or prints one")
    sbt: str = Field(
        default="",
        description="the soil behaviour type printed at this depth, in the "
                    "sheet's own words; empty when none is")
    blows: Optional[float] = Field(
        default=None,
        description="a dynamic probe's blows over the increment ending at "
                    "this depth")
    penetration: Optional[float] = Field(
        default=None,
        description="how far it went for the stated number of blows, on a "
                    "sheet printed that way round, in the depth unit")
    index: Optional[float] = Field(
        default=None,
        description="the printed penetration index or dynamic resistance at "
                    "this depth, in the index unit you stated")
    cbr_percent: Optional[float] = Field(
        default=None, description="the CBR the sheet DERIVES and prints")
    crossed: bool = Field(
        default=False,
        description="true when a trace you read here CROSSES another and "
                    "you had to decide which was which. Say which in the "
                    "note. This lowers the point's confidence, which is "
                    "the honest outcome and not a penalty")
    refusal: bool = Field(
        default=False, description="the sheet records refusal at this depth")
    note: str = Field(default="", description="anything a reviewer needs")


class ReadCone(BaseModel):
    """The instrument, as the sheet describes it."""

    model_config = ConfigDict(extra="forbid")

    cone_type: str = Field(
        default="",
        description="the cone, penetrometer or probe as printed: its model, "
                    "its class, 'piezocone', 'mechanical', 'DPSH'")
    cone_area: Optional[float] = Field(
        default=None, description="the cone's base area as printed")
    cone_area_unit: str = Field(default="", description="cm2, mm2, in2")
    sleeve_area: Optional[float] = Field(default=None)
    sleeve_area_unit: str = Field(default="")
    standard: str = Field(
        default="",
        description="the standard the sheet names: 'ASTM D5778', "
                    "'NEN 5140 class 1', 'NF P 94-115', 'EN ISO 22476-2'")
    hammer_mass: Optional[float] = Field(
        default=None, description="a dynamic probe's hammer mass, as printed")
    hammer_mass_unit: str = Field(default="", description="kg or lb")
    hammer_drop: Optional[float] = Field(default=None)
    hammer_drop_unit: str = Field(default="", description="m, mm, cm, in")
    increment: Optional[float] = Field(
        default=None,
        description="the fixed increment the blows are counted over, as "
                    "printed")
    increment_unit: str = Field(default="", description="mm, cm, m, in")
    blows_per_set: Optional[int] = Field(
        default=None,
        description="the fixed number of blows a penetration is measured "
                    "over, on a sheet printed that way round")
    penetration_rate: Optional[float] = Field(default=None)
    penetration_rate_unit: str = Field(default="", description="mm/s, m/s")


class Unsettled(BaseModel):
    """Something on this sheet you could not read."""

    model_config = ConfigDict(extra="forbid")

    what: str = Field(description="what it is, in 15 words or fewer")
    page: Optional[int] = Field(default=None)
    why: str = Field(description="what stopped you, in 20 words or fewer")


class SoundingReading(BaseModel):
    """ONE sounding as the model read it."""

    model_config = ConfigDict(extra="forbid")

    investigation_id: str = Field(
        description="the sheet's own identifier, exactly as printed (CPT-1, "
                    "S1.1, DPT-3, B-2). Empty only if the sheet prints none")
    kind: str = Field(
        default="cpt",
        description="cpt for a cone sounding, dcp for a dynamic cone or "
                    "dynamic probe")
    depth_unit: str = Field(
        default="",
        description="'m' or 'ft' -- the unit EVERY depth you report is in")
    units_known: bool = Field(
        default=True,
        description="false when no unit is printed anywhere and you are "
                    "reporting the axis's own numbers")
    vertical_axis: str = Field(
        default="depth",
        description="'depth' when the axis is a depth below the ground "
                    "surface, 'elevation' when it is a level on a datum "
                    "(m TAW, mean sea level). Read the axis title")
    datum: str = Field(
        default="",
        description="the elevation datum as printed, when the axis is an "
                    "elevation")
    ground_level: Optional[float] = Field(
        default=None, description="the ground surface level, as printed")
    total_depth: Optional[float] = Field(
        default=None, description="how deep the sounding went, as printed")
    refusal_depth: Optional[float] = Field(
        default=None,
        description="the depth the sheet says it refused at, when it says so")
    date_started: str = Field(default="", description="as printed")
    x: Optional[float] = Field(default=None, description="as printed")
    y: Optional[float] = Field(default=None, description="as printed")
    coordinate_system: str = Field(default="")
    station: str = Field(default="", description="as printed")
    cone: ReadCone = Field(default_factory=ReadCone)
    qc_unit: str = Field(
        default="",
        description="the unit the qc axis or column is printed in: MPa, "
                    "kPa, tsf, kg/cm2, bar")
    fs_unit: str = Field(default="", description="the same, for fs")
    u2_unit: str = Field(default="", description="the same, for u2")
    index_unit: str = Field(
        default="",
        description="the unit of a dynamic probe's index or resistance "
                    "column: mm, mm/blow, MPa, bar")
    digitised: bool = Field(
        default=False,
        description="true when you read the points off PLOTTED traces "
                    "because the sheet tabulates them nowhere")
    step: Optional[float] = Field(
        default=None,
        description="the depth step your points stand at, in the depth unit")
    points: List[ReadPoint] = Field(
        default_factory=list,
        description="the series, in depth order, at the step you were told")
    water_depth: Optional[float] = Field(
        default=None,
        description="depth to water where the sheet prints one; null when it "
                    "prints none, and null when it says there was none")
    water_note: str = Field(
        default="", description="what the sheet says about water, as printed")
    remarks: str = Field(
        default="", description="the sheet's own notes, verbatim")
    language: str = Field(
        default="",
        description="a two-letter code when the sheet is not in English: "
                    "fr, es, pt, nl. Empty for English")
    pages_read: List[int] = Field(default_factory=list)
    prov: Optional[ReadProv] = Field(
        default=None, description="where the header was read")
    unsettled: List[Unsettled] = Field(
        default_factory=list,
        description="everything on this sheet you could not read, and why")


# ---------------------------------------------------------------------------
# the tool
# ---------------------------------------------------------------------------

SOUNDING_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "zoom_plot",
        "description": (
            "A magnified crop of one region of the page, rendered far finer "
            "than the whole-page picture you were given. Use it to read a "
            "PLOTTED trace against its own scale. Put the AXES AND THEIR "
            "TICK LABELS inside the box on BOTH sides -- the depth scale and "
            "the measured scale -- because a trace read without them is a "
            "guess. Crop a band of the sounding at a time rather than the "
            "whole of it: a 20 m sounding crammed into 600 pixels cannot be "
            "read at half-metre steps. Give the box in the same coordinates "
            "as the text lines you were shown."),
        "input_schema": {
            "type": "object",
            "properties": {
                "page": {"type": "integer",
                         "description": "0-based PDF page index"},
                "bbox": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "x0, y0, x1, y1 of the region, including "
                                   "both axes and their tick labels",
                },
                "why": {"type": "string",
                        "description": "what you are reading, ten words or "
                                       "fewer"},
            },
            "required": ["page", "bbox"],
        },
    },
]


class _Tools:
    """The one tool the reader has, bound to one open document."""

    def __init__(self, doc: Any, pages: Sequence[int]) -> None:
        self.doc = doc
        self.pages = list(pages)
        self.zooms: List[Dict[str, Any]] = []

    def zoom_plot(self, arguments: Dict[str, Any]) -> Any:
        page = int(arguments.get("page", self.pages[0]))
        if page not in self.pages:
            raise ValueError(
                f"page {page} is not part of this sounding ({self.pages})")
        bbox = arguments.get("bbox")
        if not bbox or len(list(bbox)) != 4:
            raise ValueError("bbox must be four numbers: x0, y0, x1, y1")
        box = [float(v) for v in bbox]
        png, info = self.doc.render(page, bbox=box, dpi=ZOOM_DPI)
        self.zooms.append({"page": page, "bbox": box,
                           "why": str(arguments.get("why") or "")})
        return [text_block(
            f"page {page}, box {[round(v, 1) for v in info['clip']]} at "
            f"{info['dpi']} dpi ({info['width_px']}x{info['height_px']} px). "
            f"Read each trace against the tick labels you can see, set "
            f"digitised true, and say in the note where a trace crosses "
            f"another."),
            image_block(png)]

    def run(self, name: str, arguments: Dict[str, Any]) -> Tuple[Any, bool]:
        """``(content, is_error)`` -- a tool mistake is an answer, not a stop."""
        handler = getattr(self, name, None)
        if handler is None or name not in {t["name"] for t in SOUNDING_TOOLS}:
            return (f"unknown tool {name!r}; the tools are "
                    f"{[t['name'] for t in SOUNDING_TOOLS]}", True)
        try:
            return handler(arguments), False
        except (IndexError, KeyError, ValueError, TypeError) as exc:
            return f"{type(exc).__name__}: {exc}", True


# ---------------------------------------------------------------------------
# the prompt
# ---------------------------------------------------------------------------

SOUNDING_READER_SYSTEM = """\
You are reading ONE sounding out of an engineering report and returning what
it says as data. A sounding is not a boring log: it has no strata column, no
samples and no drives. It is a SERIES against depth, and the series is what
the record is for.

WHAT YOU ARE GIVEN. Every text line on the pages with the box it occupies,
the page as a picture, and THE STARTING RECORD -- what a deterministic pass
already read off these pages with patterns and no judgement. The starting
record says which of the two kinds of sheet this is, and that decides your
whole job.

IF THE SHEET TABULATES ITS SERIES, the starting record already holds it and
you must NOT re-type it. Retyping two hundred rows introduces errors into
numbers that were read exactly. Your job is then the HEADER: the identifier,
the date, the ground level and its datum, the cone or the hammer, the
standard, the water, the refusal depth, what the sheet says in words. Return
points ONLY where you can see the starting record read a column wrongly or
missed one, and say in the note what the page prints there.

IF THE SHEET PLOTS ITS SERIES, you digitise it. This is the harder job and
it is the one worth doing well:
- call zoom_plot on a BAND of the sounding with BOTH axes and their tick
  labels inside the box, and work down the sounding band by band;
- read a value for every trace at the depth step you are told, and report one
  point per depth with the values of all the traces at it;
- read against the TICK LABELS. The starting record tells you each axis's
  printed range; a value outside that range is a misread scale and will be
  refused;
- where a trace CROSSES another -- a friction trace over a tip trace, two
  soundings drawn on one frame -- set crossed true on that point and say in
  the note which trace you followed. A point that says it is uncertain is
  worth more than one that is quietly wrong;
- do not invent points between the ones you can see. Where the trace is off
  the scale, or the sheet stops, stop and say so in unsettled.

THE DEPTH AXIS IS NOT ALWAYS A DEPTH. Many Continental sheets plot against
an ELEVATION on a datum (m TAW, NGF, mean sea level) with the numbers going
UP the page. Read the axis title: set vertical_axis to 'elevation' and name
the datum, and report the axis's own values unchanged. Do NOT subtract them
from the ground level -- that is arithmetic, and the record does none.

UNITS STAY AS PRINTED. A tip resistance in MPa is reported as MPa, one in tsf
as tsf, one in kg/cm2 as kg/cm2. State qc_unit, fs_unit and u2_unit from the
axis titles or the column headings, and never convert. State depth_unit once
and report every depth in it. Decimal commas are the same number as decimal
points: report 1,45 as 1.45.

A DYNAMIC CONE OR DYNAMIC PROBE is the same shape of record with a different
series: blows over a fixed increment against depth, OR penetration for a
fixed number of blows, whichever way round the sheet prints it -- and you
report the one it prints, never the other. Record the increment and the
hammer's mass and drop where they are printed, because a blow count means
nothing without them. Where the sheet prints an index column of its own (DPI,
DN, mm/blow, Rd, qd) record it in the unit it is printed in and say what the
column is CALLED; never compute one.

NOTHING IS COMPUTED AND NOTHING IS INFERRED. Do not work out a friction ratio
from qc and fs. Do not classify a soil behaviour type the sheet does not
print. Do not give a cone an area the sheet does not state. Do not turn an
elevation into a depth. An empty field is an answer.

THE STARTING RECORD, AND THE THREE THINGS YOU MAY DO TO IT. You may ADD what
it lacks. You may CORRECT a value it has, but only with evidence: name the
page and the box, or say you read it off the picture and what the picture
shows. You may NEVER DROP one -- a value you leave out is kept anyway, so
leaving it out gains nothing.

Anything you cannot settle goes in unsettled with the reason.
"""

_FINAL_INSTRUCTION = (
    "Return the structured reading now. If you could not read part of the "
    "sheet, return what you did read and list the rest under unsettled."
)

_UNSETTLED_INSTRUCTION = (
    "You listed {n} thing(s) on this sheet you could not settle. Settle what "
    "you now can -- zoom in on it, or say plainly that the sheet does not "
    "state it -- and return the WHOLE reading again with everything you had "
    "plus whatever this call added. Do not drop anything."
)

_CONTINUE_INSTRUCTION = (
    "You have the rest of this sounding to digitise: you stopped at depth "
    "{depth}. Work on down it at the same step, and return the WHOLE series "
    "again -- the points you already read plus the new ones -- in depth "
    "order."
)


# ---------------------------------------------------------------------------
# the brief
# ---------------------------------------------------------------------------

def _clip(text: str, n: int = MAX_LINE_CHARS) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n - 1] + "…"


def serialise_pages(doc: Any, pages: Sequence[int]) -> str:
    """The sheets' text lines with their boxes, as compact lines."""
    out: List[str] = []
    for page in pages:
        try:
            content = doc.page(page)
        except Exception as exc:
            out.append(f"--- page {page}: could not be read "
                       f"({type(exc).__name__}: {exc}) ---")
            continue
        out.append(f"--- page {page}: {len(content.lines)} text line(s), "
                   f"{len(content.tables)} table(s) ---")
        if not content.lines:
            out.append("  THIS PAGE CARRIES NO TEXT. Read it from the "
                       "picture, and say so on every value.")
        out.append("  x0,y0,x1,y1 | text")
        lines = sorted(content.lines,
                       key=lambda ln: (round(ln.bbox[1], 1), ln.bbox[0]))
        for line in lines[:MAX_LINES_PER_PAGE]:
            x0, y0, x1, y1 = line.bbox
            out.append(f"  {x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f} | "
                       f"{_clip(line.text)}")
        if len(lines) > MAX_LINES_PER_PAGE:
            out.append(f"  [{len(lines) - MAX_LINES_PER_PAGE} further "
                       f"line(s) not listed]")
    return "\n".join(out)


def _brief(doc: Any, pages: Sequence[int], ledger: Sequence[str],
           item_title: str, report_id: str, budget: int,
           floor: SoundingFloor, step: float) -> str:
    kind_word = ("cone penetration sounding (CPT)" if floor.kind == "cpt"
                 else "dynamic cone or dynamic probe record (DCP)")
    parts: List[str] = [
        f"ONE {kind_word.upper()}, on page(s) "
        f"{', '.join(str(p) for p in pages)}"
        + (f" of report {report_id}" if report_id else "") + ".",
        f"You may make {budget} model call(s) in all, the last of them your "
        f"answer.",
    ]
    if item_title:
        parts.append(f"The document titles it: {item_title}")
    parts += [
        "",
        "WHAT THE PAGE LEDGER SAYS ABOUT THESE PAGES",
        "\n".join(ledger) if ledger else "(no ledger line)",
        "",
        "THE STARTING RECORD",
        serialise_floor(floor),
        "",
    ]
    if floor.tabulated:
        parts += [
            "THIS SHEET TABULATES ITS SERIES. The series above is already in "
            "the record. Do NOT re-type it. Spend this call on the header "
            "and on any column the starting record read wrongly or missed, "
            "and return points only for those.",
            "",
        ]
    else:
        unit = (floor.investigation.depth_unit
                if floor.investigation is not None else "") or "the axis unit"
        parts += [
            f"THIS SHEET PLOTS ITS SERIES. Digitise it through zoom_plot at "
            f"a step of {step:g} {unit}, working down the sounding band by "
            f"band with both axes in every crop. Report one point per depth "
            f"with every trace's value at it, set digitised true, and set "
            f"crossed true on any point where one trace crosses another.",
            "",
        ]
    parts += [
        "THE PAGES",
        serialise_pages(doc, pages),
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# turning the reading into the record
# ---------------------------------------------------------------------------

class _Builder:
    """Turns one :class:`SoundingReading` into an :class:`Investigation`.

    Every value passes a gate. The gates are the axes the floor read off the
    sheet's own text and the sheet's own stated depth range, so what checks
    the model is something the model did not supply.
    """

    def __init__(self, reading: SoundingReading, floor: SoundingFloor,
                 pages: Sequence[int], report_id: str,
                 zooms: Sequence[Dict[str, Any]] = ()) -> None:
        self.reading = reading
        self.floor = floor
        self.pages = list(pages)
        self.report_id = report_id
        self.zooms = list(zooms)
        self.unresolved: List[Dict[str, Any]] = []
        self.changes: List[Dict[str, Any]] = []
        self.unit = (reading.depth_unit or "").strip()
        self.units_known = bool(reading.units_known and self.unit)
        self.axes = {a.role: a for a in floor.axes}

    # -- the gates ---------------------------------------------------------
    def _depth(self, value: Optional[float], what: str) -> Optional[Quantity]:
        if value is None:
            return None
        value = float(value)
        if not self.unit:
            self.unresolved.append({
                "what": what, "page": None, "value": value,
                "why": "no depth unit was stated anywhere on this sheet, so "
                       "the number has no meaning",
                "refused_by": "python"})
            return None
        axis = self.axes.get("depth") or self.axes.get("elevation")
        if axis is not None and not axis.holds(value):
            self.unresolved.append({
                "what": what, "page": axis.page, "value": value,
                "why": f"depth {value:g} is outside the sheet's own depth "
                       f"axis ({axis.low:g} to {axis.high:g} "
                       f"{axis.unit or self.unit}); refused rather than "
                       f"accepted",
                "refused_by": "python"})
            return None
        return Quantity(value=value, unit=self.unit)

    def _channel(self, value: Optional[float], role: str, unit: str,
                 depth: float) -> Optional[Quantity]:
        """One measured value, checked against its own printed axis."""
        if value is None:
            return None
        value = float(value)
        if role == "qc" and value < 0.0:
            self.unresolved.append({
                "what": f"{role} at depth {depth:g}", "page": None,
                "value": value,
                "why": "a tip resistance cannot be negative",
                "refused_by": "python"})
            return None
        axis = self.axes.get(role)
        if axis is not None and not axis.holds(value):
            self.unresolved.append({
                "what": f"{role} at depth {depth:g}", "page": axis.page,
                "value": value,
                "why": f"{role} {value:g} is outside the sheet's own "
                       f"{role} axis ({axis.low:g} to {axis.high:g} "
                       f"{axis.unit}); the scale was misread",
                "refused_by": "python"})
            return None
        printed = unit or (axis.unit if axis is not None else "")
        return Quantity(value=value, unit=printed.strip())

    def _prov(self, read: ReadPoint, digitised: bool,
              page: int) -> Provenance:
        confidence = CONF_CROSSED if read.crossed else (
            CONF_DIGITISED if digitised else 0.85)
        note = read.note or ("digitised off the plotted trace" if digitised
                             else "read off the sheet")
        if read.crossed:
            note = (note + "; a trace crosses another here and the reader "
                           "said which it followed").strip("; ")
        return Provenance(
            page=int(page), method="model_from_picture" if digitised
            else "model",  # type: ignore[arg-type]
            confidence=confidence, note=note)

    def _q(self, value: Optional[float], unit: str) -> Optional[Quantity]:
        if value is None or not str(unit or "").strip():
            return None
        return Quantity(value=float(value), unit=str(unit).strip())

    # -- the pieces --------------------------------------------------------
    def points(self) -> List[Any]:
        r = self.reading
        page = self.pages[0]
        digitised = bool(r.digitised)
        out: List[Any] = []
        seen: set = set()
        for n, read in enumerate(r.points):
            depth = self._depth(read.depth, f"point {n + 1}")
            if depth is None:
                continue
            key = round(float(depth.value), 4)
            if key in seen:
                continue                 # one point per depth, the first won
            seen.add(key)
            prov = self._prov(read, digitised, page)
            if r.kind == "dcp" or self.floor.kind == "dcp":
                point = DCPPoint(
                    depth=depth,
                    blows=read.blows,
                    penetration=self._q(read.penetration, self.unit),
                    index=self._channel(read.index, "index", r.index_unit,
                                        float(depth.value)),
                    cbr_percent=read.cbr_percent,
                    refusal=bool(read.refusal),
                    note=read.note,
                    prov=prov)
                if point.blows is None and point.penetration is None \
                        and point.index is None and point.cbr_percent is None:
                    continue
            else:
                point = CPTPoint(
                    depth=depth,
                    qc=self._channel(read.qc, "qc", r.qc_unit,
                                     float(depth.value)),
                    fs=self._channel(read.fs, "fs", r.fs_unit,
                                     float(depth.value)),
                    u2=self._channel(read.u2, "u2", r.u2_unit,
                                     float(depth.value)),
                    rf_percent=read.rf_percent,
                    sbt=read.sbt,
                    prov=prov)
                if point.qc is None and point.fs is None \
                        and point.u2 is None and point.rf_percent is None \
                        and not point.sbt:
                    continue
            if digitised:
                self.changes.append({
                    "what": f"point at depth {depth.value:g}",
                    "page": page,
                    "why": prov.note})
            out.append(point)
        out.sort(key=lambda p: float(p.depth.value))
        return out

    def build(self) -> Investigation:
        r = self.reading
        kind = r.kind if r.kind in SOUNDING_KINDS else self.floor.kind
        cone = r.cone
        points = self.points()
        for item in r.unsettled:
            self.unresolved.append({
                "what": item.what, "page": item.page, "why": item.why,
                "refused_by": "reader"})
        elevation = None
        if r.ground_level is not None and self.unit:
            elevation = Quantity(value=float(r.ground_level), unit=self.unit)
        total = None
        if r.total_depth is not None and self.unit \
                and float(r.total_depth) >= 0.0:
            total = Quantity(value=float(r.total_depth), unit=self.unit)
        refusal = None
        if r.refusal_depth is not None and self.unit:
            refusal = Quantity(value=float(r.refusal_depth), unit=self.unit)
        step = None
        if r.step is not None and self.unit and float(r.step) > 0.0:
            step = Quantity(value=float(r.step), unit=self.unit)

        investigation = Investigation(
            investigation_id=r.investigation_id.strip(),
            kind=kind,  # type: ignore[arg-type]
            depth_unit=self.unit, units_known=self.units_known,
            x=r.x, y=r.y, coordinate_system=r.coordinate_system,
            elevation=elevation, total_depth=total,
            date_started=r.date_started, station=r.station,
            remarks=r.remarks,
            drilling=DrillingDetails(method=cone.standard),
            pages=list(self.pages), source_report=self.report_id)
        if r.water_depth is not None and self.unit:
            from report_ingest.model import WaterLevel

            investigation.water.append(WaterLevel(
                depth=Quantity(value=float(r.water_depth), unit=self.unit),
                when="unknown", note=r.water_note,
                prov=Provenance(page=self.pages[0], method="model",
                                confidence=0.8, note=r.water_note)))
        if r.language:
            investigation.fields["language"] = r.language
        if r.datum:
            investigation.fields["datum"] = r.datum

        if kind == "cpt":
            data = CPTData(
                points=points,
                vertical_axis=("elevation" if r.vertical_axis == "elevation"
                               else "depth"),
                datum=r.datum,
                step=step, digitised=bool(r.digitised),
                cone_type=cone.cone_type,
                cone_area=self._q(cone.cone_area, cone.cone_area_unit),
                sleeve_area=self._q(cone.sleeve_area, cone.sleeve_area_unit),
                standard=cone.standard,
                penetration_rate=self._q(cone.penetration_rate,
                                         cone.penetration_rate_unit),
                qc_unit=r.qc_unit, fs_unit=r.fs_unit, u2_unit=r.u2_unit,
                refusal_depth=refusal,
                prov=[Provenance(
                    page=self.pages[0],
                    method="model_from_picture" if r.digitised else "model",  # type: ignore[arg-type]
                    confidence=0.8,
                    note=(f"{len(points)} point(s), "
                          + ("digitised off the plotted traces"
                             if r.digitised else "read off the sheet")))])
            investigation.cpt = data
        else:
            data = DCPData(
                points=points,
                increment=self._q(cone.increment, cone.increment_unit),
                blows_per_set=cone.blows_per_set,
                test_type=cone.cone_type, standard=cone.standard,
                hammer_mass=self._q(cone.hammer_mass, cone.hammer_mass_unit),
                hammer_drop=self._q(cone.hammer_drop, cone.hammer_drop_unit),
                cone_area=self._q(cone.cone_area, cone.cone_area_unit),
                index_name=r.index_unit,
                digitised=bool(r.digitised), step=step,
                refusal_depth=refusal,
                prov=[Provenance(
                    page=self.pages[0],
                    method="model_from_picture" if r.digitised else "model",  # type: ignore[arg-type]
                    confidence=0.8,
                    note=f"{len(points)} increment(s)")])
            investigation.dcp = data
        return investigation


# ---------------------------------------------------------------------------
# the merge
# ---------------------------------------------------------------------------

#: Two points are the same reading when their depths are this close, in the
#: depth unit's own numbers. A digitised point at a half-metre step and a
#: tabulated one at 0.2 m are NOT the same point, and this keeps them apart.
POINT_TOL = 0.05


def _series_of(inv: Optional[Investigation]) -> Optional[Any]:
    if inv is None:
        return None
    return inv.cpt or inv.dcp


def merge_sounding(floor: Investigation, model: Investigation,
                   log: Optional[MergeLog] = None
                   ) -> Tuple[Investigation, MergeLog]:
    """The floor and the model's answer as one record, and what the merge did.

    The header, the water and everything a boring shares go through the log
    reader's own merge, which is the same rule and already tested. The
    SERIES is merged here: paired on depth, the floor's value kept where the
    model brought no evidence, the model's added where the floor had none,
    and every split recorded. A tabulated floor beating a digitised model is
    the case this exists for.
    """
    log = log if log is not None else MergeLog()
    merged, log = merge_investigations(floor, model, log)
    fdata, mdata = _series_of(floor), _series_of(model)
    if fdata is None and mdata is None:
        return merged, log
    if fdata is None or not fdata.points:
        _attach(merged, mdata)
        if mdata is not None and mdata.points:
            log.add(f"investigations[{merged.investigation_id}].series",
                    f"{len(mdata.points)} point(s)",
                    merged.pages[0] if merged.pages else None,
                    None, "model")
        return merged, log
    if mdata is None or not mdata.points:
        _attach(merged, fdata)
        log.keep(f"investigations[{merged.investigation_id}].series",
                 f"{len(fdata.points)} point(s) the reader did not return",
                 merged.pages[0] if merged.pages else None)
        return merged, log

    where = f"investigations[{merged.investigation_id}].series"
    by_depth: Dict[float, Any] = {
        round(float(p.depth.value), 4): p for p in fdata.points}
    paired: set = set()
    out: List[Any] = list(fdata.points)
    for point in mdata.points:
        key = round(float(point.depth.value), 4)
        near = min(by_depth, key=lambda d: abs(d - key), default=None)
        if near is not None and abs(near - key) <= POINT_TOL:
            twin = by_depth[near]
            paired.add(near)
            _settle_point(where, twin, point, log)
            continue
        out.append(point)
        by_depth[key] = point
        log.add(where, f"point at depth {show(point.depth)}",
                point.prov.page if point.prov else None, point.depth,
                point.prov.method if point.prov else "model")
    # EVERY FLOOR POINT THE READER DID NOT RETURN IS KEPT, and the merge
    # says so. One line rather than one per point: a two-hundred-row
    # tabulated series the reader was told not to re-type would otherwise
    # fill the QA section with two hundred identical notes, which is the
    # same fact written two hundred times.
    missed = [d for d in by_depth if d not in paired
              and any(abs(d - round(float(p.depth.value), 4)) < 1e-9
                      for p in fdata.points)]
    if missed:
        log.keep(where,
                 f"{len(missed)} floor reading(s) the reader did not return, "
                 f"from depth {min(missed):g} to {max(missed):g}",
                 merged.pages[0] if merged.pages else None)
    out.sort(key=lambda p: float(p.depth.value))
    settled = fdata.model_copy(deep=True)
    settled.points = out
    # What the MODEL alone knows about the instrument and the sheet, where
    # the floor's pattern found nothing: a cone type, a standard, a datum.
    for name in ("cone_type", "standard", "index_name", "test_type", "datum"):
        if hasattr(settled, name) and not str(getattr(settled, name) or ""):
            value = getattr(mdata, name, "")
            if value:
                setattr(settled, name, value)
    for name in ("cone_area", "sleeve_area", "hammer_mass", "hammer_drop",
                 "increment", "penetration_rate", "refusal_depth", "step",
                 "depth_interval"):
        if hasattr(settled, name) and getattr(settled, name) is None:
            value = getattr(mdata, name, None)
            if value is not None:
                setattr(settled, name, value)
    if hasattr(settled, "blows_per_set") and settled.blows_per_set is None:
        settled.blows_per_set = getattr(mdata, "blows_per_set", None)
    _attach(merged, settled)
    return merged, log


def _attach(inv: Investigation, data: Optional[Any]) -> None:
    if data is None:
        return
    if isinstance(data, CPTData):
        inv.cpt = data
        inv.dcp = None
    else:
        inv.dcp = data
        inv.cpt = None


#: The channels a point carries, per kind, for the slot-by-slot settle.
_CPT_SLOTS = ("qc", "fs", "u2")
_DCP_SLOTS = ("blows", "penetration", "index", "cbr_percent")


def _settle_point(where: str, floor_point: Any, model_point: Any,
                  log: MergeLog) -> None:
    """One depth both voters read, slot by slot, in place on the floor's point."""
    slots = _CPT_SLOTS if isinstance(floor_point, CPTPoint) else _DCP_SLOTS
    evidence = has_evidence(
        model_point.prov,
        floor_point.prov.bbox if floor_point.prov is not None else None)
    depth = show(floor_point.depth)
    for name in slots:
        fv, mv = getattr(floor_point, name, None), getattr(model_point, name,
                                                           None)
        same = _same_value(fv, mv)
        s = settle(name, fv, mv, same,
                   floor_method=(floor_point.prov.method
                                 if floor_point.prov else "tables"),
                   floor_confidence=(floor_point.prov.confidence
                                     if floor_point.prov else CONF_TABLE),
                   model_method=(model_point.prov.method
                                 if model_point.prov else "model"),
                   model_confidence=(model_point.prov.confidence
                                     if model_point.prov else 0.6),
                   evidence=evidence,
                   floor_unit=fv.unit if isinstance(fv, Quantity) else "",
                   model_unit=mv.unit if isinstance(mv, Quantity) else "")
        if s.verdict == "reconciled":
            log.reconciled += 1
        elif s.verdict in ("floor_wins", "model_wins"):
            log.disagree(
                where, f"{name} at depth {depth}", fv, mv,
                kept="floor" if s.verdict == "floor_wins" else "model",
                why=("the reader read the plotted trace differently from the "
                     "table; both are on the record"),
                page=floor_point.prov.page if floor_point.prov else None,
                floor_method=(floor_point.prov.method
                              if floor_point.prov else "tables"),
                floor_confidence=(floor_point.prov.confidence
                                  if floor_point.prov else CONF_TABLE),
                model_confidence=(model_point.prov.confidence
                                  if model_point.prov else 0.6),
                model_method=(model_point.prov.method
                              if model_point.prov else "model"))
        elif s.verdict == "model_only":
            log.add(where, f"{name} at depth {depth}",
                    model_point.prov.page if model_point.prov else None, mv,
                    model_point.prov.method if model_point.prov else "model")
        if s.value is not None:
            setattr(floor_point, name, s.value)
        if s.alternative is not None and floor_point.prov is not None:
            floor_point.prov.alternatives.append(s.alternative)
    if isinstance(floor_point, CPTPoint) and not floor_point.sbt \
            and getattr(model_point, "sbt", ""):
        floor_point.sbt = model_point.sbt


def _same_value(a: Any, b: Any) -> Optional[bool]:
    """Are two readings of one channel the same reading?

    Five per cent, because a value read off a plot and one printed in a table
    are the same reading when they agree that closely, and a scorer that
    demanded more would call every digitised point a disagreement.
    """
    if a is None or b is None:
        return None
    left = a.value if isinstance(a, Quantity) else float(a)
    right = b.value if isinstance(b, Quantity) else float(b)
    if isinstance(a, Quantity) and isinstance(b, Quantity):
        la, lb = a.si_value, b.si_value
        if la is not None and lb is not None:
            left, right = la, lb
    scale = max(abs(float(left)), abs(float(right)), 1e-9)
    return abs(float(left) - float(right)) <= 0.05 * scale


# ---------------------------------------------------------------------------
# the result
# ---------------------------------------------------------------------------

@dataclass
class SoundingReadResult:
    """One sounding read, and everything needed to audit the reading."""

    investigation: Optional[Investigation] = None
    changes: List[Dict[str, Any]] = field(default_factory=list)
    unresolved: List[Dict[str, Any]] = field(default_factory=list)
    cost: Dict[str, Any] = field(default_factory=dict)
    model_calls: int = 0
    tool_calls: int = 0
    model: str = ""
    warnings: List[str] = field(default_factory=list)
    pages: List[int] = field(default_factory=list)
    kind: str = "cpt"
    tabulated: bool = False
    #: The two voters kept apart, so the scorecard can score each alone.
    floor: Optional[Investigation] = None
    model_investigation: Optional[Investigation] = None
    floor_points: int = 0
    axes: List[Dict[str, Any]] = field(default_factory=list)
    disagreements: List[Dict[str, Any]] = field(default_factory=list)
    kept: List[Dict[str, Any]] = field(default_factory=list)
    added: List[Dict[str, Any]] = field(default_factory=list)
    reconciled: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "investigation": (self.investigation.model_dump(mode="json")
                              if self.investigation is not None else None),
            "changes": [dict(c) for c in self.changes],
            "unresolved": [dict(u) for u in self.unresolved],
            "cost": dict(self.cost),
            "model_calls": self.model_calls,
            "tool_calls": self.tool_calls,
            "model": self.model,
            "warnings": list(self.warnings),
            "pages": list(self.pages),
            "kind": self.kind,
            "tabulated": self.tabulated,
            "floor": (self.floor.model_dump(mode="json")
                      if self.floor is not None else None),
            "model_investigation": (
                self.model_investigation.model_dump(mode="json")
                if self.model_investigation is not None else None),
            "floor_points": self.floor_points,
            "axes": [dict(a) for a in self.axes],
            "disagreements": [dict(d) for d in self.disagreements],
            "kept": [dict(k) for k in self.kept],
            "added": [dict(a) for a in self.added],
            "reconciled": self.reconciled,
        }


def default_step(floor: SoundingFloor) -> float:
    """The depth step a PLOTTED sounding is digitised at.

    The sheet's own printed grid where the axis's ticks state one, and
    :data:`CPT_STEP_M` or :data:`DCP_STEP_M` otherwise. A sheet whose depth
    axis is ticked every metre is digitised every metre: asking for readings
    between its own gridlines asks the reader to interpolate, which is the
    one thing a digitised series must not do.
    """
    fallback = CPT_STEP_M if floor.kind == "cpt" else DCP_STEP_M
    for axis in floor.axes:
        if axis.role in ("depth", "elevation") and axis.tick:
            tick = abs(float(axis.tick))
            if tick > 0:
                # THE FINER OF THE TWO, never the coarser. A cone sounding's
                # traces swing further between two of the sheet's own
                # gridlines than they do across the whole of the rest of the
                # hole -- on the corpus's own sheets qc goes from 0.2 to
                # 9.9 MPa in a fifth of the 1 m grid -- so asking for one
                # reading per printed gridline asks for a number that is not
                # there. Floored, because a reading every centimetre is not
                # a reading either.
                return max(MIN_STEP_M, min(tick, fallback))
    return fallback


def read_sounding(doc, item_pages: Sequence[int], engine: Engine, *,
                  kind: str = "cpt", budget: int = MAX_SOUNDING_CALLS,
                  ledger: Optional[Sequence[str]] = None,
                  item_title: str = "", report_id: str = "",
                  dpi: float = PAGE_DPI) -> SoundingReadResult:
    """Read ONE sounding sheet into an :class:`Investigation` with a series.

    ``kind`` is ``"cpt"`` or ``"dcp"`` and comes from the page label the
    voters settled on. ``budget`` is the model-call ceiling: a tabulated
    sheet spends one of its two, a plotted one both.
    """
    pages = [int(p) for p in item_pages]
    if not pages:
        raise ValueError("read_sounding needs at least one page")
    kind = kind if kind in SOUNDING_KINDS else "cpt"
    budget = max(1, min(int(budget), MAX_SOUNDING_CALLS))

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

    warnings: List[str] = []
    for page in pages:
        try:
            content = doc.page(page)
        except Exception as exc:                 # a page that will not read
            warnings.append(f"page {page}: {type(exc).__name__}: {exc}")
        else:
            warnings.extend(f"page {page}: {w}" for w in content.warnings)

    floor = sounding_floor(doc, pages, kind, report_id)
    warnings.extend(floor.warnings)
    step = default_step(floor)

    images: List[bytes] = []
    for page in pages[:2]:
        try:
            png, _info = doc.render(page, dpi=dpi)
        except Exception:                        # a page that will not draw
            continue                             # is read from its text alone
        images.append(png)

    brief = _brief(doc, pages, ledger, item_title, report_id, budget, floor,
                   step)
    tools = _Tools(doc, pages)
    messages: List[Dict[str, Any]] = [
        user(text_block(brief), *[image_block(png) for png in images])]

    model_calls = 0
    tool_calls = 0
    reading: Optional[SoundingReading] = None
    final: Any = None
    followed_up = False

    while model_calls < budget:
        last = model_calls == budget - 1
        reply = engine.complete(messages, system=SOUNDING_READER_SYSTEM,
                                tools=None if last else SOUNDING_TOOLS,
                                output_format=SoundingReading)
        model_calls += 1
        charge(reply)
        final = reply
        if reply.tool_calls and not last:
            messages.append({"role": "assistant", "content": reply.content})
            results: List[Dict[str, Any]] = []
            for call in reply.tool_calls:
                tool_calls += 1
                content, is_error = tools.run(call.name, call.arguments)
                results.append(tool_result_block(call.id, content,
                                                 is_error=is_error))
            messages.append({"role": "user", "content": results})
            continue
        if reply.parsed is not None:
            reading = reply.parsed
            if last or followed_up:
                break
            if reading.unsettled:
                followed_up = True
                messages.append({"role": "assistant",
                                 "content": reply.content or [text_block("")]})
                messages.append(user(text_block(
                    _UNSETTLED_INSTRUCTION.format(n=len(reading.unsettled)))))
                continue
            break
        # Neither a tool call nor an answer: say what is wanted, once, and
        # spend another call on it rather than returning nothing.
        messages.append({"role": "assistant",
                         "content": reply.content or [text_block("")]})
        messages.append(user(text_block(_FINAL_INSTRUCTION)))

    if reading is None:
        raise RuntimeError(
            f"the sounding reader returned no structured reading for pages "
            f"{pages} in {model_calls} call(s) (stop_reason "
            f"{getattr(final, 'stop_reason', None)!r})")

    builder = _Builder(reading, floor, pages, report_id, tools.zooms)
    model_investigation = builder.build()
    floor_inv = floor.investigation or Investigation(
        kind=kind, pages=pages,  # type: ignore[arg-type]
        source_report=report_id)
    merged, merge_log = merge_sounding(floor_inv, model_investigation,
                                       MergeLog())
    spent["seconds"] = round(spent["seconds"], 2)
    spent["dollars"] = round(spent["dollars"], 5)
    return SoundingReadResult(
        investigation=merged,
        changes=builder.changes,
        unresolved=builder.unresolved,
        cost=spent,
        model_calls=model_calls,
        tool_calls=tool_calls,
        model=getattr(final, "model", "") or getattr(engine, "name", ""),
        warnings=warnings,
        pages=pages,
        kind=kind,
        tabulated=floor.tabulated,
        floor=floor_inv,
        model_investigation=model_investigation,
        floor_points=floor.n_points,
        axes=[a.to_dict() for a in floor.axes],
        disagreements=merge_log.disagreements,
        kept=merge_log.kept,
        added=merge_log.added,
        reconciled=merge_log.reconciled)
