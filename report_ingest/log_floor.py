"""The floor under the log reader: the grid's own record, and the merge.

WHY THERE IS A FLOOR. On the first full cluster run the log reader took the
grid's rows as its source and re-emitted the whole record from the model's
answer -- and on seven of ten blind logs the answer held FEWER values than the
grid had already placed: recovery 15/15 -> 3/15, index 7/16 -> 0/16, blows
25/25 -> 19/25. Nothing the grid knew was wrong; the model simply left it out.
So the grid is now the first voter. :func:`seed_from_grid` turns what
``log_grid`` placed into an :class:`~report_ingest.model.Investigation`
BEFORE any model is called, with the grid's own confidence on every value;
the model is shown that record as its starting point; and
:func:`merge_investigations` folds the model's answer back onto it under one
rule -- a value the model adds is accepted, a value it corrects with evidence
replaces the floor's with the floor's kept as the alternative, a value it
contradicts without evidence stays the floor's with the model's kept as the
alternative, and a value it omits is kept with a note. Every contradiction
is a disagreement for the QA section. Nothing is ever dropped.

WHAT THE GRID CAN AND CANNOT SEED. It places text at a column and a depth
and asserts no meaning, so everything here is the smallest reading of a
placed cell that the scorer already credits it for: a number in a blows
column at a depth is a blow record at that depth, ``N=21`` is an N value,
a number under WATER CONTENT (%) is a water content, a layer the grid bound
to a depth is a layer, a header field is a header field. A sampler symbol,
a water triangle's timing, a refusal's meaning and a symbol the log draws
rather than prints are the model's to add, and the model adds them.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from report_ingest.floor import (
    DEPTH_TOL_M, LAYER_TOL_M, MergeLog, Settled, carry, depth_m, fold,
    has_evidence, same_depth, same_number, same_text, settle, show,
)
from report_ingest.model import (
    DrillingDetails, Investigation, Layer, PitDimensions, Provenance,
    Quantity, SPT, Sample, WaterLevel, _canonical_unit,
)

__all__ = ["seed_from_grid", "serialise_seed", "merge_investigations",
           "GROUP_SPAN_M", "pit_dimensions", "bucket_width", "header_pairs",
           "PIT_METHOD_KEYS"]

#: How far apart two placed cells may be and still belong to one sample: a
#: driven sample is 0.46 m (18 in) long and its blow record, its N and its
#: index values are printed against it, one to three lines apart.
GROUP_SPAN_M = 0.46
FT_PER_M = 3.280839895

#: The columns whose cells mark a sample. A description, a depth tick, a
#: graphic and a remark do not.
_SAMPLE_FAMILY = frozenset({
    "sample_id", "sample_type", "blows", "n_value", "recovery", "rqd",
    "water_content", "dry_unit_weight", "liquid_limit", "plastic_limit",
    "plasticity_index", "fines", "qu", "pocket_pen",
})
_INDEX_NAMES = ("water_content", "dry_unit_weight", "liquid_limit",
                "plastic_limit", "plasticity_index", "fines", "qu",
                "pocket_pen")

#: The sampler codes a log prints in its type column. Only read in a column
#: whose header names the sample TYPE; anywhere else "S" is a label.
_TYPE_CODES: Dict[str, str] = {
    "ss": "spt", "spt": "spt", "sp": "spt", "sb": "spt", "s": "spt",
    "st": "shelby", "sh": "shelby", "tw": "shelby", "u": "shelby",
    "ud": "shelby", "shelby": "shelby",
    "r": "ring", "ring": "ring", "mc": "ring", "ca": "ring", "cal": "ring",
    "ms": "ring", "cs": "ring",
    "b": "bulk", "bulk": "bulk", "bag": "bulk", "bs": "bulk",
    "g": "grab", "gb": "grab", "grab": "grab",
    "c": "core", "core": "core", "nx": "core", "nq": "core", "hq": "core",
    "rc": "core", "cr": "core",
    "ct": "cuttings", "cuttings": "cuttings", "auger": "cuttings",
    "undist": "shelby", "und": "shelby",
}

#: The index family, plus the two columns whose cells carry their own label
#: on a form that stacks several results under one heading.
_LABELLED_FAMILY = frozenset(_INDEX_NAMES) | {"tests", "recovery", "rqd"}

#: ``LABEL = value`` as a form prints it inside one cell: ``MC = 10.1%``,
#: ``LL = 38``, ``% Passing #200 = 68.7``, ``REC=29cm, 64%``. The label is
#: folded to letters and digits and looked up in :data:`_LABEL_NAMES`.
_LABELLED = re.compile(r"^\s*([^=]{1,30}?)\s*[=:]\s*(.+)$", re.DOTALL)
_LABEL_NAMES: Dict[str, str] = {
    "mc": "water_content", "w": "water_content", "wc": "water_content",
    "moisturecontent": "water_content", "watercontent": "water_content",
    "moisture": "water_content", "wn": "water_content",
    "ll": "liquid_limit", "liquidlimit": "liquid_limit",
    "pl": "plastic_limit", "plasticlimit": "plastic_limit",
    "pi": "plasticity_index", "plasticityindex": "plasticity_index",
    "passing200": "fines", "p200": "fines", "200": "fines",
    "fines": "fines", "percentpassing200": "fines",
    "qu": "qu", "pp": "pocket_pen", "pocketpen": "pocket_pen",
    "dd": "dry_unit_weight", "drydensity": "dry_unit_weight",
    "dryunitweight": "dry_unit_weight",
    "rec": "recovery", "recovery": "recovery",
    "rqd": "rqd",
}
#: A number with the unit the cell printed beside it, and a percentage
#: anywhere after it: ``29cm, 64%`` -> ``(29.0, "cm", 64.0)``.
_LABEL_VALUE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(cm|mm|in|ft|m|%|\"|pcf|kpa|psf|ksf|tsf)?", re.I)
_LABEL_PERCENT = re.compile(r"(-?\d+(?:\.\d+)?)\s*%")

#: A sample named and typed in ONE cell, the way a form that stacks its
#: sampling data prints it: ``S-1, SPT``, ``GR-3, GRAB``, ``UD-1, UNDIST``.
_ID_AND_TYPE = re.compile(
    r"^\s*([A-Za-z]{1,4}-?\d+[A-Za-z]?)\s*[,/]\s*([A-Za-z]{1,8})\s*$")

_USCS = re.compile(
    r"\b((?:GW|GP|GM|GC|SW|SP|SM|SC|ML|CL|OL|MH|CH|OH|PT)"
    r"(?:-(?:GW|GP|GM|GC|SW|SP|SM|SC|ML|CL|OL|MH|CH|OH|PT|ML))?)\b")
_N_VALUE = re.compile(r"\bN\s*[=:]\s*(\d+)\b")
_INTERVAL = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*$")
_PAREN = re.compile(r"\(([^)]{1,14})\)")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


# ---------------------------------------------------------------------------
# small readers of one cell
# ---------------------------------------------------------------------------

def _paren_unit(header: str) -> str:
    """The unit a header prints in parentheses, folded to the record's own
    spelling when the table knows it: ``DRY UNIT WEIGHT (pcf)`` -> ``pcf``."""
    for token in _PAREN.findall(str(header or "")):
        key = _canonical_unit(token)
        if key:
            return token.strip()
    return ""


def _blow_tokens(text: str) -> List[Any]:
    """A blow record as the log printed it, or ``[]`` when the text is not
    one. ``5-9-12`` -> ``[5, 9, 12]``; ``12-30-50/5"`` -> ``[12, 30,
    '50/5"']``; ``5/9/12`` -> ``[5, 9, 12]``; ``2+1+2`` -> ``[2, 1, 2]``.

    The plus is gINT's own spelling of a driven record and is as common in
    the corpus as the hyphen, so it is read here rather than left on the
    page.
    """
    raw = " ".join(str(text or "").split())
    if not raw or not re.fullmatch(r'[\d\s+\-–/",.\'inmc]+', raw):
        return []
    if "+" in raw:
        pieces = re.split(r"\s*\+\s*", raw)
    elif "-" in raw or "–" in raw:
        pieces = re.split(r"\s*[-–]\s*", raw)
    elif raw.count("/") >= 2 and '"' not in raw and "in" not in raw:
        pieces = raw.split("/")
    else:
        pieces = [raw]
    out: List[Any] = []
    for piece in pieces:
        piece = piece.strip(" ,")
        if not piece:
            continue
        if re.fullmatch(r"\d+", piece):
            out.append(int(piece))
        elif re.fullmatch(r'\d+\s*/\s*[\d.]+\s*(?:"|\'\'|in|cm|mm)?', piece):
            out.append(piece)
        else:
            return []
    if len(out) < 2 and not (out and isinstance(out[0], str)):
        return []
    return out


def _names(grid: Any, cell: Any,
           by_column: Optional[Dict[str, Tuple[str, ...]]] = None
           ) -> Tuple[str, ...]:
    """What this cell's column carries.

    ``by_column`` is a recognised TEMPLATE's own reading of the columns
    (:func:`report_ingest.log_templates.column_names`). It wins where it has
    something to say, because a form that prints ``DATA`` over three stacked
    values gives the general header vocabulary nothing to classify, and the
    fingerprint knows what that column holds.
    """
    if by_column:
        named = by_column.get(cell.column_id)
        if named:
            return tuple(named)
    column = grid.column(cell.column_id)
    return tuple(column.names) if column is not None else ()


def _header(grid: Any, cell: Any) -> str:
    column = grid.column(cell.column_id)
    return str(column.header or "") if column is not None else ""


def _unit_of_column(grid: Any, cell: Any, fallback: str = "") -> str:
    column = grid.column(cell.column_id)
    if column is None:
        return fallback
    return (str(column.unit or "") or _paren_unit(column.header)
            or fallback).strip()


def _labelled_value(text: str) -> Optional[Tuple[str, float, str,
                                                 Optional[float]]]:
    """``(what it is, the number, its unit, a percentage)`` off one cell.

    A form that stacks several results under one heading prints each one
    with its own label -- ``MC = 10.1%``, ``LL = 38``, ``REC=29cm, 56%`` --
    and the label is the only thing that says which result it is. Read here
    and nowhere else, so the grid's floor asserts exactly what the cell
    prints and nothing more. ``None`` when the cell carries no label this
    recognises, which is the normal case and leaves every other reader
    exactly as it was.
    """
    match = _LABELLED.match(" ".join(str(text or "").split()))
    if match is None:
        return None
    key = "".join(ch for ch in match.group(1).lower() if ch.isalnum())
    name = _LABEL_NAMES.get(key)
    if name is None:
        return None
    rest = match.group(2)
    value = _LABEL_VALUE.search(rest)
    if value is None:
        return None
    unit = (value.group(2) or "").strip()
    percent = None
    for hit in _LABEL_PERCENT.finditer(rest):
        if hit.start() > value.end(1):
            percent = float(hit.group(1))
            break
    if unit == "%" and percent is None:
        percent = float(value.group(1))
    return name, float(value.group(1)), ("" if unit == "%" else unit), percent


def _limits_order(header: str) -> List[str]:
    """Which of LL, PL, PI a header prints, in the order it prints them."""
    found: List[Tuple[int, str]] = []
    for token, name in (("LL", "liquid_limit"), ("PL", "plastic_limit"),
                        ("PI", "plasticity_index")):
        match = re.search(rf"\b{token}\b", str(header or ""), re.I)
        if match:
            found.append((match.start(), name))
    return [name for _pos, name in sorted(found)]


# ---------------------------------------------------------------------------
# the seed
# ---------------------------------------------------------------------------

def _span_in_unit(unit: str, grid: Any) -> float:
    if unit == "ft":
        return GROUP_SPAN_M * FT_PER_M
    if unit == "m":
        return GROUP_SPAN_M
    steps = [abs(r.step) for r in grid.rulers.values() if r.step]
    return 1.5 if steps and min(steps) >= 2.0 else 0.5


def _q(value: Optional[float], unit: str,
       prov: Optional[Provenance] = None) -> Optional[Quantity]:
    if value is None:
        return None
    return Quantity(value=float(value), unit=unit, prov=prov)


def _first_number(text: str) -> Optional[float]:
    match = _NUMBER.search(str(text or ""))
    return float(match.group(0)) if match else None


#: Words a title pattern can catch where an identifier should be.
_NOT_AN_ID = frozenset({"log", "logs", "no", "number", "boring", "borings",
                        "pit", "pits", "test", "of", "record", "sheet",
                        "page", "data", "hole", "borehole", "sondage",
                        "sondeo", "calicata"})


def _looks_like_an_id(text: str) -> bool:
    """Is this a hole's name rather than a word off the title?

    Every identifier in the corpus carries a digit (B-1, TP-04, SB-11, 3) or
    is a letter or two (A, BH). A whole word with no digit -- LOG, BORING --
    is the title's, not the hole's.
    """
    token = str(text or "").strip()
    if not token:
        return False
    if token.lower().strip(".:-#") in _NOT_AN_ID:
        return False
    return any(ch.isdigit() for ch in token) or len(token) <= 2


class _Group:
    """The cells of one sample, gathered by depth.

    Depth alone, not page: a hole has one sample at one depth, so a row
    placed at the same depth on a second sheet -- a form that repeats its
    last line, a continuation sheet that restates the sample it broke on --
    is the same sample read twice, not a second one.
    """

    def __init__(self, cell: Any) -> None:
        self.cells: List[Any] = [cell]
        self.page = cell.page
        self.first = cell.depth

    def takes(self, cell: Any, span: float) -> bool:
        return abs(float(cell.depth) - float(self.first)) <= span

    def add(self, cell: Any) -> None:
        if not any(c.page == cell.page and c.column_id == cell.column_id
                   and c.text == cell.text and c.bbox == cell.bbox
                   for c in self.cells):
            self.cells.append(cell)


def _seed_sample(group: _Group, grid: Any, unit: str, confidence_floor: float,
                 by_column: Optional[Dict[str, Tuple[str, ...]]] = None,
                 template_note: str = ""
                 ) -> Tuple[Optional[Sample], Optional[SPT]]:
    top = min(float(c.depth) for c in group.cells)
    bottom: Optional[float] = None
    sample_id = ""
    kind = "other"
    blows: List[Any] = []
    n: Optional[int] = None
    refusal = False
    values: Dict[str, Any] = {}
    units: Dict[str, str] = {}
    texts: List[str] = []
    confidence = min(float(c.confidence) for c in group.cells)
    blow_cell = None
    stacked: List[Tuple[Tuple[float, float, float], int]] = []
    for cell in group.cells:
        names = _names(grid, cell, by_column)
        text = " ".join(str(cell.text or "").split())
        texts.append(text)
        code = text.lower()
        # A sampler code: SS, ST, MC ... in any sample column; a single
        # letter only where the column is headed as the sample TYPE, since
        # a log may label its samples A, B, C.
        if code in _TYPE_CODES and (
                "sample_type" in names
                or (len(code) >= 2 and ("sample_id" in names
                                        or "blows" in names))):
            if kind == "other":
                kind = _TYPE_CODES[code]
            continue
        # A sample named and typed in one cell: "S-1, SPT". Only where the
        # column carries sample identifiers, so a comma in a description
        # cannot become a sampler code.
        pair = _ID_AND_TYPE.match(text) if "sample_id" in names else None
        if pair is not None and pair.group(2).lower() in _TYPE_CODES:
            if not sample_id:
                sample_id = pair.group(1)
            if kind == "other":
                kind = _TYPE_CODES[pair.group(2).lower()]
            continue
        # A result that names itself: "MC = 10.1%", "REC=29cm, 56%".
        labelled = (_labelled_value(text)
                    if set(names) & _LABELLED_FAMILY else None)
        if labelled is not None:
            what, number, cell_unit, percent = labelled
            if what == "recovery":
                if percent is not None:
                    values.setdefault("recovery_percent", percent)
                if cell_unit:
                    values.setdefault("recovery", number)
                    units.setdefault("recovery", cell_unit)
            elif what == "rqd":
                values.setdefault("rqd_percent",
                                  percent if percent is not None else number)
            elif what == "fines":
                values.setdefault("fines_percent", number)
            else:
                values.setdefault(what, number)
                if cell_unit and what in ("dry_unit_weight", "qu",
                                          "pocket_pen"):
                    units.setdefault(what, cell_unit)
            continue
        match = _N_VALUE.search(text)
        if match and ("blows" in names or "n_value" in names
                      or "sample_id" in names):
            n = int(match.group(1))
            continue
        tokens = _blow_tokens(text) if ("blows" in names or "n_value" in names
                                        or "sample_id" in names) else []
        if tokens:
            blows = tokens
            refusal = any(isinstance(t, str) for t in tokens)
            blow_cell = cell
            continue
        # A blow record printed one increment to a line, each in its own
        # cell, which is what the two commonest gINT forms in the corpus do.
        # Gathered here and assembled after the loop, in the order the form
        # printed them; a LONE number is left where it is, because one
        # number in a blows column is as likely to be an N value.
        if "blows" in names and re.fullmatch(r"\d{1,3}", text):
            stacked.append(((float(cell.depth), float(cell.bbox[1]),
                             float(cell.bbox[0])), int(text)))
            continue
        interval = _INTERVAL.match(text)
        if interval and "sample_id" in names:
            top = min(top, float(interval.group(1)))
            bottom = float(interval.group(2))
            continue
        if ("n_value" in names and "blows" not in names
                and len(cell.numbers) == 1 and re.fullmatch(r"\d+", text)):
            n = int(cell.numbers[0])
            continue
        if "sample_id" in names and text and not cell.numbers:
            if not sample_id:
                sample_id = text
            continue
        if "sample_id" in names and text and len(cell.numbers) == 1 \
                and re.fullmatch(r"[A-Za-z]*-?\d+[A-Za-z]?", text) \
                and "blows" not in names:
            if not sample_id:
                sample_id = text
            continue
        if "rqd" in names and cell.numbers:
            values["rqd_percent"] = float(cell.numbers[0])
            continue
        if "recovery" in names and cell.numbers:
            header = _header(grid, cell)
            col_unit = _unit_of_column(grid, cell)
            number = float(cell.numbers[0])
            if "%" in text or col_unit == "%" or "%" in header \
                    or re.search(r"\bpercent\b|\bpct\b", header, re.I) \
                    or (not col_unit and number <= 100.0):
                values["recovery_percent"] = number
            else:
                values["recovery"] = number
                units["recovery"] = col_unit
            continue
        limits = [name for name in ("liquid_limit", "plastic_limit",
                                    "plasticity_index") if name in names]
        if limits and cell.numbers:
            order = _limits_order(_header(grid, cell)) or (
                ["liquid_limit", "plastic_limit", "plasticity_index"]
                if len(cell.numbers) == 3 else limits)
            for name, number in zip(order, cell.numbers):
                values.setdefault(name, float(number))
            continue
        for name in ("water_content", "dry_unit_weight", "fines", "qu",
                     "pocket_pen"):
            if name in names and cell.numbers:
                key = "fines_percent" if name == "fines" else name
                values.setdefault(key, float(cell.numbers[0]))
                if name in ("dry_unit_weight", "qu", "pocket_pen"):
                    units[key] = _unit_of_column(
                        grid, cell, "pcf" if name == "dry_unit_weight"
                        and "pcf" in _header(grid, cell).lower() else "")
                break
    if not blows and len(stacked) >= 2:
        stacked.sort(key=lambda row: row[0])
        blows = [number for _key, number in stacked]
    if kind == "other" and blows:
        kind = "spt"
    anchor = blow_cell or group.cells[0]
    page = int(anchor.page)
    prov = Provenance(
        page=page, bbox=tuple(anchor.bbox), method="grid",
        confidence=max(confidence_floor, min(1.0, confidence)),
        note=("seeded from the grid rows"
              + (f" ({template_note})" if template_note else "") + ": "
              + "; ".join(t for t in texts if t)[:160]))
    sample = Sample(
        sample_id=sample_id, top=_q(top, unit), bottom=_q(bottom, unit),
        kind=kind,  # type: ignore[arg-type]
        recovery=_q(values.get("recovery"), units.get("recovery", "")),
        recovery_percent=_pct(values.get("recovery_percent")),
        rqd_percent=_pct(values.get("rqd_percent")),
        water_content=values.get("water_content"),
        dry_unit_weight=_q(values.get("dry_unit_weight"),
                           units.get("dry_unit_weight", "")),
        liquid_limit=values.get("liquid_limit"),
        plastic_limit=values.get("plastic_limit"),
        plasticity_index=values.get("plasticity_index"),
        fines_percent=_pct(values.get("fines_percent")),
        qu=_q(values.get("qu"), units.get("qu", "")),
        pocket_pen=_q(values.get("pocket_pen"), units.get("pocket_pen", "")),
        prov=prov)
    spt = None
    if blows or n is not None:
        spt = SPT(depth_top=_q(top, unit), depth_bottom=_q(bottom, unit),
                  blows=blows, n=n, refusal=refusal, sample_id=sample_id,
                  prov=prov.model_copy(deep=True))
    return sample, spt


def _pct(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return max(0.0, min(100.0, float(value)))


# ---------------------------------------------------------------------------
# what a PIT has and a hole does not
# ---------------------------------------------------------------------------
#
# A test pit is an EXCAVATION with a plan size. The grid's header vocabulary
# was written for boreholes and has no term for a length, a width or a
# bucket, so those never reached the record: a pit came back as a hole with
# strata in it. They are read here, off the header lines of the first page,
# by the same rule as everything else in this module -- a printed label
# beside a printed number, and nothing inferred.

#: ``normalised label -> which dimension``, in the four languages the corpus
#: prints its pit logs in. A label that names two or three dimensions at
#: once ("Dimensions (L x W x D)") is handled by :data:`_DIM_RUN` below.
_DIM_KEYS: Dict[str, str] = {
    "length": "length", "pit length": "length", "trench length": "length",
    "excavation length": "length", "longueur": "length",
    "comprimento": "length", "largo": "length", "longitud": "length",
    "width": "width", "pit width": "width", "trench width": "width",
    "excavation width": "width", "bucket width": "width",
    "largeur": "width", "largura": "width", "ancho": "width",
    "depth": "depth", "pit depth": "depth", "trench depth": "depth",
    "excavation depth": "depth", "depth of pit": "depth",
    "depth of excavation": "depth", "profondeur de la fouille": "depth",
    "profundidade": "depth", "profundidad": "depth",
}

#: Labels that introduce a run of dimensions in one value:
#: ``Pit Dimensions: 2.5 m x 1.0 m x 3.5 m``, ``Size (L x W x D)``.
_DIM_RUN_KEYS: Tuple[str, ...] = (
    "dimensions", "pit dimensions", "test pit dimensions",
    "excavation dimensions", "trench dimensions", "pit size", "size",
    "plan dimensions", "dimension", "dimensoes", "dimensiones",
)

#: How a pit log names how it was dug. Set on ``DrillingDetails.method``,
#: which is where the record keeps "how the hole was made" whatever made it.
PIT_METHOD_KEYS: Tuple[str, ...] = (
    "excavation method", "method of excavation", "excavated by",
    "excavation equipment", "metodo de escavacao", "metodo de excavacion",
    "methode d excavation", "methode de fouille", "equipment",
)

#: ``label : value`` on one header line, in either punctuation a form uses.
_HEADER_PAIR = re.compile(r"^\s*([^:=]{2,40}?)\s*[:=]\s*(.+?)\s*$")
#: A label with nothing after it, which is how a printed form sets one: the
#: label is its own text run and the value is another run to the right of it
#: on the same baseline.
_BARE_LABEL = re.compile(r"^\s*([^:=]{2,40}?)\s*[:=]\s*$")
#: How far apart two text runs' tops may be and still be on one printed line.
_PAIR_BAND_PT = 4.0


def header_pairs(lines: Sequence[Any]) -> List[Tuple[str, str]]:
    """``(label, value)`` for every header field these lines state.

    BOTH SHAPES, because forms use both. A line reading ``Equipment: CAT
    428E, 55 cm bucket`` is one text run and splits on its colon; a printed
    form sets ``Equipment:`` as its own run with the value as a second run
    to the right of it on the same baseline, and nothing splits at all. The
    second is the commoner of the two on a drawn form and was invisible to a
    colon-splitting reader.
    """
    out: List[Tuple[str, str]] = []
    placed: List[Tuple[float, float, str]] = []      # (top, x0, text)
    for raw in lines:
        text = " ".join(str(getattr(raw, "text", raw) or "").split())
        if not text or len(text) > 160:
            continue
        pair = _HEADER_PAIR.match(text)
        if pair is not None and pair.group(2).strip():
            out.append((pair.group(1).strip(), pair.group(2).strip()))
        bbox = getattr(raw, "bbox", None)
        if bbox is not None:
            placed.append((float(bbox[1]), float(bbox[0]), text))
    placed.sort()
    for top, x0, text in placed:
        label = _BARE_LABEL.match(text)
        if label is None:
            continue
        # The nearest run to the RIGHT on the same printed line. Sorting by
        # the top alone is not enough to find it: a form sets a label and
        # its value a tenth of a point apart vertically, so either can come
        # first in a sort, and the whole band has to be looked at.
        on_the_line = [(other_x0, other)
                       for other_top, other_x0, other in placed
                       if abs(other_top - top) <= _PAIR_BAND_PT
                       and other_x0 > x0 and other.strip()]
        on_the_line.sort()
        for _x, other in on_the_line:
            if _BARE_LABEL.match(other) is not None:
                break               # the next label, not this one's value
            out.append((label.group(1).strip(), other.strip()))
            break
    return out
#: ``2.5 m x 1.0 m x 3.5 m``, ``8' x 3' x 10'``, ``2,5 x 1,0 x 3,5 m``.
_DIM_RUN = re.compile(
    r"(-?\d+(?:[.,]\d+)?)\s*"
    r"(mm|cm|m|ft|in|'|\")?\s*"
    r"(?:x|×|by|par|por)\s*"
    r"(-?\d+(?:[.,]\d+)?)\s*"
    r"(mm|cm|m|ft|in|'|\")?"
    r"(?:\s*(?:x|×|by|par|por)\s*(-?\d+(?:[.,]\d+)?)\s*"
    r"(mm|cm|m|ft|in|'|\")?)?", re.I)
#: One number with the unit printed beside it.
_DIM_ONE = re.compile(
    r"(-?\d+(?:[.,]\d+)?)\s*(mm|cm|m|ft|in|'|\")?", re.I)

#: THE BUCKET, which is how a test pit log actually states its width.
#: Fourteen pits across four reports of this corpus were checked and NOT ONE
#: prints a labelled length or width; every one of them names the machine and
#: its bucket -- "... with a 55 cm bucket", "... Rubber Tire Backhoe 90 cm
#: Bucket", "... w/ 1 m wide bucket", "1.06 m Wide Mechanical Bucket". A
#: trench dug with a 90 cm bucket is 90 cm wide, and that is what
#: the page is telling a reader. It is recorded as the pit's WIDTH at a
#: lower confidence than a labelled field, with the provenance saying it
#: came off the bucket, so a reviewer can see exactly what was read and what
#: was taken from it.
_BUCKET = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(mm|cm|m|ft|in|'|\")\s*(?:\w+\s+){0,2}"
    r"(?:bucket|godet|balde|cazo|cuchar[oa])"
    r"|(?:bucket|godet|balde)\D{0,12}?(\d+(?:[.,]\d+)?)\s*(mm|cm|m|ft|in|'|\")",
    re.I)
#: The header fields whose VALUE may name the bucket. Only these: a bucket
#: mentioned in a remark ("backfilled with the bucket") says nothing about
#: how wide the pit was.
_BUCKET_FIELDS: Tuple[str, ...] = (
    "equipment", "drilling equipment", "excavation equipment", "machine",
    "rig", "drill rig", "equipement", "equipo", "maquina", "material",
    "excavation method", "method of excavation", "excavated by",
)


#: The grid's own field keys whose VALUE may name the bucket. A printed form
#: sets its label and its value as two separate text spans, so a
#: ``label: value`` pattern over the lines never sees them as one string --
#: the grid has already paired them by geometry and that pairing is what is
#: read here.
_BUCKET_GRID_KEYS: Tuple[str, ...] = (
    "drilling_equipment", "drilling_method", "abandonment",
)


def bucket_width(lines: Sequence[Any] = (), unit: str = "",
                 fields: Optional[Dict[str, Any]] = None
                 ) -> Optional[Tuple[Quantity, str]]:
    """``(the bucket's width, what it was printed in)``, or None.

    Both the grid's own header FIELDS and the raw lines are looked at: a
    a printed form pairs "Equipment:" with its value geometrically and sets
    them as two spans, while a form that writes "Equipment: backhoe with a
    55 cm bucket" on one line is read straight off the line.
    """
    candidates: List[str] = []
    for key in _BUCKET_GRID_KEYS:
        value = str((fields or {}).get(key) or "").strip()
        if value:
            candidates.append(value)
    for raw_label, value in header_pairs(lines):
        if _norm_label(raw_label) in _BUCKET_FIELDS:
            candidates.append(value)
    for value in candidates:
        match = _BUCKET.search(value)
        if match is None:
            continue
        number = _dim_number(match.group(1) or match.group(3))
        token = match.group(2) or match.group(4)
        if number is None or number <= 0.0:
            continue
        return (Quantity(value=number, unit=_dim_unit(token, unit)),
                value.strip())
    return None


def _dim_number(text: str) -> Optional[float]:
    """A dimension as a number, decimal comma and all."""
    try:
        return float(str(text).replace(",", "."))
    except (TypeError, ValueError):
        return None


def _dim_unit(token: Optional[str], fallback: str) -> str:
    key = _canonical_unit(str(token or "")) if token else None
    return key or fallback


def _norm_label(text: str) -> str:
    """A header label folded to words, for the dimension tables above."""
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", str(text or "").lower())
    return " ".join(cleaned.split())


def pit_dimensions(lines: Sequence[Any], unit: str = "",
                   page: int = 0,
                   fields: Optional[Dict[str, Any]] = None
                   ) -> Optional[PitDimensions]:
    """The pit's plan size and its own depth, off the header lines.

    ``lines`` are text lines (anything with a ``.text``) or plain strings.
    Two shapes are read and no others: a label naming ONE dimension with a
    number beside it, and a label introducing a RUN of two or three
    (``Dimensions: 2.5 m x 1.0 m x 3.5 m``), in which case the first is the
    length, the second the width and the third, if any, the depth. A unit
    printed against a number wins; where none is, the log's depth unit
    stands in, because a pit log that prints its depths in feet does not
    print its width in metres.

    Returns None when the header printed nothing about the pit's size,
    which is the common case and is not a failure.
    """
    length = width = depth = None
    seen: List[str] = []
    for raw_label, value in header_pairs(lines):
        label = _norm_label(raw_label)
        if not value:
            continue
        if label in _DIM_RUN_KEYS or (
                label.endswith(" dimensions") or label.startswith("dimension")
                or label.startswith("size ") or label == "size"):
            run = _DIM_RUN.search(value)
            if run is None:
                continue
            numbers = [(_dim_number(run.group(1)), run.group(2)),
                       (_dim_number(run.group(3)), run.group(4))]
            if run.group(5) is not None:
                numbers.append((_dim_number(run.group(5)), run.group(6)))
            # A run states its unit once, usually on the LAST number. So a
            # number with no unit of its own takes the run's own unit before
            # it falls back to the log's.
            stated = next((u for _n, u in reversed(numbers) if u), None)
            slots = ["length", "width", "depth"]
            for (number, token), slot in zip(numbers, slots):
                if number is None:
                    continue
                quantity = Quantity(
                    value=number, unit=_dim_unit(token or stated, unit))
                if slot == "length":
                    length = quantity
                elif slot == "width":
                    width = quantity
                else:
                    depth = quantity
            seen.append(raw_label)
            continue
        slot = _DIM_KEYS.get(label)
        if slot is None:
            continue
        one = _DIM_ONE.search(value)
        if one is None:
            continue
        number = _dim_number(one.group(1))
        if number is None:
            continue
        quantity = Quantity(value=number, unit=_dim_unit(one.group(2), unit))
        if slot == "length" and length is None:
            length = quantity
        elif slot == "width" and width is None:
            width = quantity
        elif slot == "depth" and depth is None:
            depth = quantity
        else:
            continue
        seen.append(raw_label)
    note = ("pit dimensions from the header field(s) "
            + ", ".join(sorted(set(seen)))) if seen else ""
    confidence = 0.7
    if width is None:
        bucket = bucket_width(lines, unit, fields)
        if bucket is not None:
            width, said = bucket
            confidence = 0.6
            note = (note + "; " if note else "") + (
                f"the width is the BUCKET the equipment field names "
                f"({said[:60]!r}); the header states no pit dimension of its "
                f"own")
    if length is None and width is None and depth is None:
        return None
    return PitDimensions(
        length=length, width=width, depth=depth,
        prov=Provenance(page=int(page), method="grid", confidence=confidence,
                        note=note or "pit dimensions read off the header"))


def _pit_method(lines: Sequence[Any]) -> str:
    """How the pit was dug, off a header field, or an empty string."""
    for raw_label, value in header_pairs(lines):
        if _norm_label(raw_label) in PIT_METHOD_KEYS and value:
            return value
    return ""


def seed_from_grid(grid: Any, pages: Sequence[int], report_id: str = "",
                   confidence_floor: float = 0.3,
                   template: Any = None,
                   lines: Sequence[Any] = ()) -> Investigation:
    """The grid's own reading of one log as an :class:`Investigation`.

    Every value carries a ``grid`` provenance at the grid's own confidence
    for that cell. Nothing here is inferred: a sampler is ``other`` unless
    the type column prints a code, an N is recorded only where the log
    prints ``N=``, a USCS symbol only where the description prints one in
    the standard letters, and a water level only where the header's
    groundwater field prints a depth.

    ``template`` is a
    :class:`~report_ingest.log_templates.TemplateMatch` when the page was
    recognised as a known printed FORM. Its ``column_map`` names the columns
    the general header vocabulary could not -- a heading of ``DATA`` over a
    stacked sample id, blow record and recovery is the case -- and the
    method stays ``grid``, with the template named in the note, because the
    value was still placed by geometry and nothing was inferred from the
    form beyond what its own columns carry.

    ``lines`` are the text lines of the log's pages, when the caller has
    them. They are read for the two things a HEADER states that the grid's
    borehole vocabulary has no term for: a test pit's plan dimensions and
    how the pit was dug. Left empty, nothing changes and no pit gets
    dimensions -- which is what every caller before this got.
    """
    pages = [int(p) for p in pages]
    unit = (grid.unit or "").strip()
    fields = dict(grid.fields or {})
    used: set = set()

    by_column: Dict[str, Tuple[str, ...]] = {}
    template_note = ""
    if template is not None:
        from report_ingest.log_templates import column_names, ledger_note

        by_column = column_names(grid, template)
        if by_column:
            template_note = f"{ledger_note(template)}, columns named by it"

    def fprov(key: str, confidence: float = 0.8) -> Provenance:
        page, bbox = grid.field_boxes.get(key, (pages[0], None)) \
            if getattr(grid, "field_boxes", None) else (pages[0], None)
        return Provenance(page=int(page), bbox=tuple(bbox) if bbox else None,
                          method="grid", confidence=confidence,
                          note=f"header field {key}")

    def text(key: str) -> str:
        value = str(fields.get(key) or "").strip()
        if value:
            used.add(key)
        return value

    investigation_id = text("boring_id") or text("test_pit_id")
    if investigation_id and not _looks_like_an_id(investigation_id):
        # The grid read a word off the title -- "LOG", "NO", "BORING" --
        # where an identifier should be. A word is not an identifier; it is
        # kept as a printed field and the model names the hole.
        extra_word = investigation_id
        investigation_id = ""
        fields["title_word"] = extra_word
    kind = "test_pit" if (fields.get("test_pit_id")
                          and not fields.get("boring_id")) else "boring"
    elevation = None
    raw = text("ground_surface_elevation")
    if raw and _first_number(raw) is not None:
        elevation = Quantity(value=_first_number(raw),
                             unit=_paren_unit(raw) or unit,
                             prov=fprov("ground_surface_elevation"))
    total = None
    raw = text("total_depth")
    if raw and _first_number(raw) is not None \
            and _first_number(raw) >= 0.0 and unit:
        total = Quantity(value=_first_number(raw), unit=unit,
                         prov=fprov("total_depth"))

    header_prov: List[Provenance] = []
    for key in ("boring_id", "test_pit_id"):
        if key in used:
            header_prov.append(fprov(key, 0.9))
    if template is not None:
        # The template claim travels ON the record, so a reviewer of the
        # finished investigation sees which form it was read off and how
        # sure the recogniser was.
        from report_ingest.log_templates import ledger_note as _note

        header_prov.append(Provenance(
            page=pages[0], method="grid",
            confidence=max(0.0, min(1.0, float(
                getattr(template, "confidence", 0.0) or 0.0))),
            note=_note(template) + (", columns named by it" if by_column
                                    else ", no column map")))

    drilling = DrillingDetails(
        method=text("drilling_method"), equipment=text("drilling_equipment"),
        hammer_type=text("hammer_type"),
        driller=text("driller"), contractor=text("contractor"),
        logged_by=text("logged_by"),
        prov=[fprov(k) for k in ("drilling_method", "drilling_equipment",
                                 "hammer_type", "driller", "contractor",
                                 "logged_by") if k in used])

    # THE PIT. A pit's plan size and the way it was dug are printed in the
    # header, in words the grid's borehole vocabulary has no term for, so
    # they are read off the lines rather than off the fields.
    pit = pit_dimensions(lines, unit, pages[0], fields)         if (lines or fields) else None
    if pit is not None and kind != "test_pit":
        # A hole with a length and a width is a pit however the header
        # titled itself: a borehole has a diameter, not a plan size. The
        # grid's own answer is kept in the fields so a reviewer sees both.
        if pit.length is not None and pit.width is not None:
            fields["grid_kind"] = kind
            kind = "test_pit"
    if pit is not None:
        method = _pit_method(lines)
        if method and not drilling.method.strip():
            drilling.method = method
            drilling.prov.append(Provenance(
                page=pages[0], method="grid", confidence=0.7,
                note="how the pit was dug, from its header"))
        pit.method = method or drilling.method

    water: List[WaterLevel] = []
    raw = text("groundwater")
    if raw:
        lowered = raw.lower()
        number = _first_number(raw)
        if re.search(r"not encountered|none|dry|no water|n/e|n\.e\.",
                     lowered) and number is None:
            water.append(WaterLevel(depth=None, when="not_encountered",
                                    note=raw, prov=fprov("groundwater", 0.6)))
        elif number is not None and unit:
            water.append(WaterLevel(
                depth=Quantity(value=number, unit=unit,
                               prov=fprov("groundwater", 0.5)),
                when="unknown", note=raw, prov=fprov("groundwater", 0.5)))

    layers: List[Layer] = []
    seen_layers: set = set()
    for ly in grid.layers:
        if ly.top is None:
            continue
        description = " ".join(str(ly.description or "").split())
        if not description:
            # A band the grid opened between two rules with no words in it
            # is not a stratum; a record that carried it would export an
            # empty lithology. The words are what make a layer.
            continue
        key = (round(float(ly.top), 2), fold(description))
        if key in seen_layers:
            continue                  # the same stratum restated on a sheet
        seen_layers.add(key)
        page = int(ly.pages[0]) if ly.pages else pages[0]
        symbol = _USCS.search(description)
        layers.append(Layer(
            top=_q(ly.top, unit), bottom=_q(ly.bottom, unit),
            description=str(ly.description or ""),
            uscs=symbol.group(1) if symbol else "",
            prov=Provenance(page=page,
                            bbox=tuple(ly.bbox) if ly.bbox else None,
                            method="grid",
                            confidence=max(confidence_floor,
                                           min(1.0, float(ly.confidence))),
                            note=f"layer bound by the grid ({ly.source})")))

    samples: List[Sample] = []
    spt: List[SPT] = []
    if unit or grid.rulers:
        span = _span_in_unit(unit, grid)
        anchors = [c for c in grid.rows
                   if c.depth is not None and c.page in pages
                   and set(_names(grid, c, by_column)) & _SAMPLE_FAMILY
                   and (c.numbers or str(c.text or "").strip())]
        anchors.sort(key=lambda c: (c.page, float(c.depth), c.bbox[0]))
        groups: List[_Group] = []
        for cell in anchors:
            # Any group at this depth takes the cell, whichever sheet it is
            # on: the rows are sorted by page first, so a second sheet's
            # restatement of a depth would otherwise open a second sample.
            home = next((g for g in groups if g.takes(cell, span)), None)
            if home is not None:
                home.add(cell)
            else:
                groups.append(_Group(cell))
        for group in groups:
            sample, drive = _seed_sample(group, grid, unit, confidence_floor,
                                         by_column, template_note)
            if sample is not None:
                samples.append(sample)
            if drive is not None:
                spt.append(drive)

    extra = {k: str(v) for k, v in fields.items()
             if k not in used and str(v or "").strip()}
    return Investigation(
        investigation_id=investigation_id,
        kind=kind,  # type: ignore[arg-type]
        depth_unit=unit, units_known=bool(unit),
        elevation=elevation, total_depth=total,
        date_started=text("date_started"),
        date_finished=text("date_finished"),
        drilling=drilling, layers=layers, samples=samples, spt=spt,
        water=water, pit=pit, sheet=text("sheet"),
        pages=pages, source_report=report_id, fields=extra,
        prov=header_prov)


# ---------------------------------------------------------------------------
# what the model is shown
# ---------------------------------------------------------------------------

def _where(prov: Optional[Provenance]) -> str:
    if prov is None:
        return ""
    box = ("" if prov.bbox is None else
           " box " + ",".join(f"{v:.0f}" for v in prov.bbox))
    return f"page {prov.page}{box}"


def serialise_seed(inv: Investigation) -> str:
    """The seed as compact lines the model can build on and cite."""
    unit = inv.depth_unit or "(unit not stated)"
    out: List[str] = []
    header = [f"identifier '{inv.investigation_id}'" if inv.investigation_id
              else "identifier: not found"]
    if inv.elevation is not None:
        header.append(f"elevation {show(inv.elevation)}")
    if inv.total_depth is not None:
        header.append(f"total depth {show(inv.total_depth)}")
    for name in ("method", "equipment", "hammer_type", "driller",
                 "contractor", "logged_by"):
        value = getattr(inv.drilling, name)
        if value:
            header.append(f"{name} '{value}'")
    for name in ("date_started", "date_finished", "sheet"):
        value = getattr(inv, name)
        if value:
            header.append(f"{name} '{value}'")
    out.append("header: " + "; ".join(header))
    if inv.pit is not None:
        size = [f"{name} {show(getattr(inv.pit, name))}"
                for name in ("length", "width", "depth")
                if getattr(inv.pit, name) is not None]
        out.append(f"pit dimensions ({_where(inv.pit.prov)}): "
                   + ", ".join(size))
    for key, value in sorted(inv.fields.items()):
        out.append(f"header field {key} = '{value[:80]}'")
    for ly in inv.layers:
        out.append(
            f"layer {show(ly.top)} to {show(ly.bottom) or '?'} "
            f"({_where(ly.prov)}): '{ly.description[:120]}'"
            + (f", uscs {ly.uscs}" if ly.uscs else "")
            + (f"  [confidence {ly.prov.confidence:.2f}]" if ly.prov else ""))
    for s in inv.samples:
        parts = [f"kind {s.kind}"]
        if s.sample_id:
            parts.insert(0, f"sample_id '{s.sample_id}'")
        for name in ("recovery_percent", "rqd_percent", "water_content",
                     "liquid_limit", "plastic_limit", "plasticity_index",
                     "fines_percent"):
            value = getattr(s, name)
            if value is not None:
                parts.append(f"{name} {value:g}")
        for name in ("recovery", "dry_unit_weight", "qu", "pocket_pen"):
            value = getattr(s, name)
            if value is not None:
                parts.append(f"{name} {show(value)}")
        out.append(f"sample at {show(s.top)}"
                   + (f" to {show(s.bottom)}" if s.bottom else "")
                   + f" ({_where(s.prov)}): " + ", ".join(parts)
                   + (f"  [confidence {s.prov.confidence:.2f}]" if s.prov
                      else ""))
    for r in inv.spt:
        out.append(f"driven record at {show(r.depth_top)} "
                   f"({_where(r.prov)}): blows {show(r.blows) or '?'}"
                   + (f", N {r.n}" if r.n is not None else ", N not printed")
                   + (", refusal" if r.refusal else ""))
    for w in inv.water:
        out.append(f"water: {show(w.depth) or 'none'} ({w.when}) "
                   f"({_where(w.prov)}): '{w.note[:80]}'")
    return "\n".join(out) or "(the form reader placed nothing on this log)"


# ---------------------------------------------------------------------------
# the merge
# ---------------------------------------------------------------------------

def _pair(floor_items: Sequence[Any], model_items: Sequence[Any],
          key: Any, tol: float
          ) -> Tuple[List[Tuple[Any, Any]], List[Any], List[Any]]:
    """Nearest-neighbour pairs within ``tol`` metres, and the leftovers."""
    taken: set = set()
    pairs: List[Tuple[Any, Any]] = []
    floor_only: List[Any] = []
    for f in floor_items:
        fd = depth_m(key(f))
        best = None
        best_gap = None
        if fd is not None:
            for i, m in enumerate(model_items):
                if i in taken:
                    continue
                md = depth_m(key(m))
                if md is None:
                    continue
                gap = abs(md - fd)
                if gap <= tol and (best_gap is None or gap < best_gap):
                    best, best_gap = i, gap
        if best is None:
            floor_only.append(f)
        else:
            taken.add(best)
            pairs.append((f, model_items[best]))
    model_only = [m for i, m in enumerate(model_items) if i not in taken]
    return pairs, floor_only, model_only


def _conf(prov: Optional[Provenance], default: float) -> float:
    return float(prov.confidence) if prov is not None else default


def _same_quantity(a: Optional[Quantity], b: Optional[Quantity]
                   ) -> Optional[bool]:
    if a is None or b is None:
        return None
    left, right = a.si_value, b.si_value
    if left is None or right is None or a.unit.strip() == "" \
            or b.unit.strip() == "":
        return same_number(a.value, b.value)
    return same_number(left, right)


def _blow_numbers(blows: Sequence[Any]) -> List[float]:
    out: List[float] = []
    for entry in blows:
        if isinstance(entry, (int, float)):
            out.append(float(entry))
        else:
            out.extend(float(t) for t in _NUMBER.findall(str(entry)))
    return out


def _same_blows(a: Sequence[Any], b: Sequence[Any]) -> Optional[bool]:
    if not a or not b:
        return None
    left, right = _blow_numbers(a), _blow_numbers(b)
    if len(left) != len(right):
        return False
    return all(abs(x - y) <= 1e-9 for x, y in zip(left, right))


class _Merger:
    """Folds one model investigation onto one floor investigation."""

    def __init__(self, floor: Investigation, model: Investigation,
                 log: MergeLog) -> None:
        self.floor = floor
        self.model = model
        self.log = log
        self.inv_id = model.investigation_id or floor.investigation_id or "?"

    # -- one slot --------------------------------------------------------
    def slot(self, where: str, what: str, name: str, fv: Any, mv: Any,
             same: Optional[bool], floor_prov: Optional[Provenance],
             model_prov: Optional[Provenance], evidence: bool,
             page: Optional[int]) -> Settled:
        s = settle(name, fv, mv, same,
                   floor_confidence=_conf(floor_prov, 0.5),
                   model_method=(model_prov.method if model_prov is not None
                                 and model_prov.method in
                                 ("model", "model_from_picture", "vision")
                                 else "model"),
                   model_confidence=_conf(model_prov, 0.9),
                   evidence=evidence,
                   floor_unit=fv.unit if isinstance(fv, Quantity) else "",
                   model_unit=mv.unit if isinstance(mv, Quantity) else "",
                   model_note=(model_prov.note if model_prov is not None
                               else ""))
        if s.verdict == "reconciled":
            self.log.reconciled += 1
        elif s.verdict == "model_only":
            self.log.add(where, f"{what}: {name}", page, mv, s.method)
        elif s.verdict == "floor_only":
            self.log.keep(where, f"{what}: {name}", page, fv)
        elif s.verdict in ("floor_wins", "model_wins"):
            self.log.disagree(
                where, f"{what}: {name}", fv, mv,
                kept="model" if s.verdict == "model_wins" else "floor",
                why=("the model named a box and a note, so its reading "
                     "replaces the floor's; the floor's is kept beside it"
                     if s.verdict == "model_wins" else
                     "the model's reading differs from the floor's and it "
                     "named no box or note for it; the floor's stands and "
                     "the model's is kept beside it"),
                page=page,
                floor_method=floor_prov.method if floor_prov else "grid",
                floor_confidence=_conf(floor_prov, 0.5),
                model_confidence=_conf(model_prov, 0.9),
                model_method=s.method if s.verdict == "model_wins"
                else (model_prov.method if model_prov else "model"))
        return s

    # -- objects ---------------------------------------------------------
    def merge_pair(self, where: str, what: str, f: Any, m: Any,
                   specs: Sequence[Tuple[str, str]], model_only: Sequence[str]
                   ) -> Any:
        """One floor object and one model object into one merged object.

        ``specs`` is ``(field, kind)`` for the voted slots, in the order they
        are settled; ``model_only`` names the fields the floor never seeds,
        which are taken from the model as they are.
        """
        floor_prov, model_prov = f.prov, m.prov
        evidence = has_evidence(model_prov,
                                floor_prov.bbox if floor_prov else None)
        page = (model_prov.page if model_prov is not None
                else floor_prov.page if floor_prov is not None else None)
        merged = m.model_copy(deep=True)
        prov = (model_prov or floor_prov)
        prov = prov.model_copy(deep=True) if prov is not None else None
        alternatives = []
        methods: List[str] = []
        confidences: List[float] = []
        for name, kind in specs:
            fv, mv = getattr(f, name), getattr(m, name)
            if kind == "depth":
                same = same_depth(fv, mv)
            elif kind == "layer_depth":
                same = same_depth(fv, mv, LAYER_TOL_M)
            elif kind == "quantity":
                same = _same_quantity(fv, mv)
            elif kind == "number":
                same = same_number(fv, mv)
            elif kind == "exact":
                same = same_number(fv, mv, 0.0)
            elif kind == "text":
                same = same_text(fv, mv)
            elif kind == "exact_text":
                same = (None if not (fold(fv) and fold(mv))
                        else fold(fv) == fold(mv))
            elif kind == "blows":
                same = _same_blows(fv, mv)
            elif kind == "kind":
                # The floor's sampler is a guess from a type code and the
                # model's from the symbol; 'other' on either side is not an
                # answer, so only two real answers can disagree.
                fv = None if fv == "other" else fv
                mv = None if mv == "other" else mv
                same = None if fv is None or mv is None else fv == mv
            elif kind == "when":
                fv = None if fv == "unknown" else fv
                mv = None if mv == "unknown" else mv
                same = None if fv is None or mv is None else fv == mv
            elif kind == "flag":
                setattr(merged, name, bool(fv) or bool(mv))
                continue
            else:
                same = None
            if kind == "blows" and (not fv and not mv):
                setattr(merged, name, [])
                continue
            if kind == "blows":
                fv = fv or None
                mv = mv or None
            if kind in ("text", "exact_text"):
                fv = fv or None
                mv = mv or None
            s = self.slot(where, what, name, fv, mv, same, floor_prov,
                          model_prov, evidence, page)
            value = s.value
            if kind == "kind" and value is None:
                value = "other"
            if kind == "when" and value is None:
                value = "unknown"
            if kind in ("text", "exact_text", "blows") and value is None:
                value = "" if kind != "blows" else []
            if isinstance(value, Quantity):
                value = value.model_copy(deep=True)
                if value.prov is None:
                    value.prov = (prov.model_copy(deep=True) if prov
                                  else None)
                if value.prov is not None:
                    value.prov.method = s.method  # type: ignore[assignment]
                    value.prov.confidence = max(0.0, min(1.0, s.confidence))
            setattr(merged, name, value)
            if s.alternative is not None:
                alternatives.append(s.alternative)
            if s.verdict != "empty":
                methods.append(s.method)
                confidences.append(s.confidence)
        for name in model_only:
            setattr(merged, name, getattr(m, name))
        if prov is not None:
            # The object's own method and confidence are those of the slot
            # that identifies it -- its depth, settled last -- which for a
            # pair the two voters matched on is ``reconciled``. A slot the
            # voters split on does not change what the object IS; its loser
            # stands in ``alternatives`` under the slot's name.
            if methods:
                prov.method = methods[-1]  # type: ignore[assignment]
                prov.confidence = max(0.0, min(1.0, confidences[-1]))
            prov.alternatives.extend(alternatives)
            if floor_prov is not None and model_prov is not None:
                prov.note = (f"{model_prov.note}; " if model_prov.note
                             else "") + "matched to the grid's row" \
                    + (f" ({floor_prov.note[:60]})" if floor_prov.note else "")
        merged.prov = prov
        return merged

    def keep_whole(self, where: str, what: str, f: Any) -> Any:
        kept = f.model_copy(deep=True)
        self.log.keep(where, what, f.prov.page if f.prov else None)
        return kept

    def add_whole(self, where: str, what: str, m: Any) -> Any:
        added = m.model_copy(deep=True)
        self.log.add(where, what, m.prov.page if m.prov else None,
                     method=m.prov.method if m.prov else "model")
        return added

    # -- the lists -------------------------------------------------------
    def samples(self) -> List[Sample]:
        base = f"investigations[{self.inv_id}].samples"
        pairs, floor_only, model_only = _pair(
            self.floor.samples, self.model.samples, lambda s: s.top,
            DEPTH_TOL_M)
        out: List[Sample] = []
        for f, m in pairs:
            what = f"sample at {show(f.top)}"
            out.append(self.merge_pair(
                f"{base}[{show(f.top)}]", what, f, m,
                specs=[("sample_id", "exact_text"), ("kind", "kind"),
                       ("bottom", "depth"), ("recovery", "quantity"),
                       ("recovery_percent", "number"),
                       ("rqd_percent", "number"),
                       ("water_content", "number"),
                       ("dry_unit_weight", "quantity"),
                       ("liquid_limit", "number"),
                       ("plastic_limit", "number"),
                       ("plasticity_index", "number"),
                       ("fines_percent", "number"), ("qu", "quantity"),
                       ("pocket_pen", "quantity"), ("uscs", "exact_text"),
                       ("top", "depth")],
                model_only=["note"]))
        for f in floor_only:
            out.append(self.keep_whole(f"{base}[{show(f.top)}]",
                                       f"sample at {show(f.top)}", f))
        for m in model_only:
            out.append(self.add_whole(f"{base}[{show(m.top)}]",
                                      f"sample at {show(m.top)}", m))
        out.sort(key=lambda s: (depth_m(s.top) if depth_m(s.top) is not None
                                else 1e9))
        return out

    def spt(self) -> List[SPT]:
        base = f"investigations[{self.inv_id}].spt"
        pairs, floor_only, model_only = _pair(
            self.floor.spt, self.model.spt, lambda r: r.depth_top,
            DEPTH_TOL_M)
        out: List[SPT] = []
        for f, m in pairs:
            what = f"driven record at {show(f.depth_top)}"
            out.append(self.merge_pair(
                f"{base}[{show(f.depth_top)}]", what, f, m,
                specs=[("depth_bottom", "depth"), ("blows", "blows"),
                       ("n", "exact"), ("refusal", "flag"),
                       ("sample_id", "exact_text"),
                       ("depth_top", "depth")],
                model_only=["hammer", "increment"]))
        for f in floor_only:
            out.append(self.keep_whole(f"{base}[{show(f.depth_top)}]",
                                       f"driven record at "
                                       f"{show(f.depth_top)}", f))
        for m in model_only:
            out.append(self.add_whole(f"{base}[{show(m.depth_top)}]",
                                      f"driven record at "
                                      f"{show(m.depth_top)}", m))
        out.sort(key=lambda r: (depth_m(r.depth_top)
                                if depth_m(r.depth_top) is not None else 1e9))
        return out

    def layers(self) -> List[Layer]:
        base = f"investigations[{self.inv_id}].layers"
        pairs, floor_only, model_only = _pair(
            self.floor.layers, self.model.layers, lambda ly: ly.top,
            LAYER_TOL_M)
        out: List[Layer] = []
        for f, m in pairs:
            what = f"layer at {show(f.top)}"
            out.append(self.merge_pair(
                f"{base}[{show(f.top)}]", what, f, m,
                specs=[("bottom", "layer_depth"), ("description", "text"),
                       ("uscs", "exact_text"), ("top", "layer_depth")],
                model_only=["consistency", "color", "moisture"]))
        for f in floor_only:
            out.append(self.keep_whole(f"{base}[{show(f.top)}]",
                                       f"layer at {show(f.top)}", f))
        for m in model_only:
            out.append(self.add_whole(f"{base}[{show(m.top)}]",
                                      f"layer at {show(m.top)}", m))
        out.sort(key=lambda ly: (depth_m(ly.top)
                                 if depth_m(ly.top) is not None else 1e9))
        return out

    def water(self) -> List[WaterLevel]:
        base = f"investigations[{self.inv_id}].water"
        floor_items = list(self.floor.water)
        model_items = list(self.model.water)
        pairs, floor_only, model_only = _pair(
            [w for w in floor_items if w.depth is not None],
            [w for w in model_items if w.depth is not None],
            lambda w: w.depth, DEPTH_TOL_M)
        # A floor entry saying "not encountered" matches a model entry
        # saying the same; otherwise the two stand side by side.
        for f in [w for w in floor_items if w.depth is None]:
            twin = next((w for w in model_only + [w for w in model_items
                                                  if w.depth is None]
                         if w.depth is None and w.when == f.when), None)
            if twin is not None and twin in model_items:
                pairs.append((f, twin))
                if twin in model_only:
                    model_only.remove(twin)
            else:
                floor_only.append(f)
        model_only += [w for w in model_items
                       if w.depth is None and not any(w is p[1]
                                                      for p in pairs)]
        out: List[WaterLevel] = []
        for f, m in pairs:
            what = f"water at {show(f.depth) or 'none'}"
            out.append(self.merge_pair(
                f"{base}[{show(f.depth) or 'none'}]", what, f, m,
                specs=[("when", "when"), ("depth", "depth")],
                model_only=["hours", "date", "casing_depth", "caved_depth",
                            "elevation", "note"]))
        # The grid read a depth off the header's groundwater field and the
        # model says none was encountered, or a depth nowhere near it: two
        # entries stand in the record and the split is on the QA list. The
        # two cannot be one slot -- a depth and "none" -- so both are kept.
        if floor_only and model_only:
            for f in floor_only:
                for m in model_only:
                    self.log.disagree(
                        f"{base}[{show(f.depth) or 'none'}]",
                        f"water at {show(f.depth) or 'none'}: depth",
                        f.depth if f.depth is not None else f.when,
                        m.depth if m.depth is not None else m.when,
                        kept="both",
                        why="the grid's header field and the model give "
                            "different water readings; both are in the "
                            "record",
                        page=f.prov.page if f.prov else None,
                        floor_method=f.prov.method if f.prov else "grid",
                        floor_confidence=_conf(f.prov, 0.5),
                        model_confidence=_conf(m.prov, 0.9),
                        model_method=m.prov.method if m.prov else "model")
        for f in floor_only:
            out.append(self.keep_whole(
                f"{base}[{show(f.depth) or 'none'}]",
                f"water at {show(f.depth) or 'none'}", f))
        for m in model_only:
            out.append(self.add_whole(
                f"{base}[{show(m.depth) or 'none'}]",
                f"water at {show(m.depth) or 'none'}", m))
        return out

    # -- the header ------------------------------------------------------
    def header(self, merged: Investigation) -> None:
        f, m = self.floor, self.model
        where = f"investigations[{self.inv_id}]"
        fprov = f.prov[0] if f.prov else None
        mprov = m.prov[0] if m.prov else None
        page = f.pages[0] if f.pages else (m.pages[0] if m.pages else None)

        def text_slot(name: str, fv: str, mv: str, floor_method: str = "grid"
                      ) -> str:
            same = same_text(fv, mv)
            s = self.slot(where, "header", name, fv or None, mv or None,
                          same, fprov, mprov, False, page)
            if s.alternative is not None:
                merged.prov.append(Provenance(
                    page=int(page) if page is not None else 0,
                    method=s.method,  # type: ignore[arg-type]
                    confidence=s.confidence,
                    note=f"header {name}",
                    alternatives=[s.alternative]))
            return s.value or ""

        merged.investigation_id = text_slot(
            "investigation_id", f.investigation_id, m.investigation_id)
        merged.kind = m.kind if m.kind != "other" else f.kind
        # The unit: the ruler's is measured off the page, the model's is
        # read off it; where both exist and differ the floor's stands and
        # the disagreement is on the record.
        s = self.slot(where, "header", "depth_unit",
                      f.depth_unit or None, m.depth_unit or None,
                      (fold(f.depth_unit) == fold(m.depth_unit)
                       if f.depth_unit and m.depth_unit else None),
                      fprov, mprov, False, page)
        merged.depth_unit = s.value or ""
        merged.units_known = bool(merged.depth_unit) and (
            f.units_known if f.depth_unit else m.units_known)
        for name in ("elevation", "total_depth"):
            fv, mv = getattr(f, name), getattr(m, name)
            s = self.slot(where, "header", name, fv, mv,
                          _same_quantity(fv, mv), fprov, mprov, False, page)
            value = s.value
            if isinstance(value, Quantity):
                value = value.model_copy(deep=True)
                if value.prov is None and page is not None:
                    value.prov = Provenance(page=int(page), method="grid")
                if value.prov is not None:
                    value.prov.method = s.method  # type: ignore[assignment]
                    value.prov.confidence = max(0.0, min(1.0, s.confidence))
                    if s.alternative is not None:
                        value.prov.alternatives.append(s.alternative)
            setattr(merged, name, value)
        for name in ("date_started", "date_finished", "station", "offset",
                     "sheet"):
            setattr(merged, name,
                    text_slot(name, getattr(f, name), getattr(m, name)))
        for name in ("method", "equipment", "hammer_type", "driller",
                     "contractor", "logged_by"):
            setattr(merged.drilling, name,
                    text_slot(f"drilling.{name}", getattr(f.drilling, name),
                              getattr(m.drilling, name)))
        merged.drilling.hammer_energy_ratio = m.drilling.hammer_energy_ratio
        merged.drilling.prov = list(f.drilling.prov) + list(m.drilling.prov)
        merged.x, merged.y = m.x, m.y
        merged.coordinate_system = m.coordinate_system
        merged.remarks = m.remarks or f.remarks
        fields = dict(f.fields)
        for key, value in m.fields.items():
            fields.setdefault(key, value)
        merged.fields = fields
        merged.pages = list(m.pages or f.pages)
        merged.source_report = m.source_report or f.source_report
        merged.prov = list(f.prov) + list(m.prov) + [
            p for p in merged.prov if p not in f.prov and p not in m.prov]

    def run(self) -> Investigation:
        merged = self.model.model_copy(deep=True)
        merged.prov = []
        self.header(merged)
        merged.layers = self.layers()
        merged.samples = self.samples()
        merged.spt = self.spt()
        merged.water = self.water()
        merged.pit = self.pit()
        for r in merged.spt:
            if not r.hammer:
                r.hammer = merged.drilling.hammer_type
        return merged

    # -- the pit ---------------------------------------------------------
    def pit(self) -> Optional[PitDimensions]:
        """The pit's dimensions, slot by slot, under the same rule.

        Three numbers and a word, so there is nothing to pair up: each slot
        is settled on its own, the floor's value stands where the model
        brought no evidence, and every split is a disagreement.
        """
        f, m = self.floor.pit, self.model.pit
        if f is None and m is None:
            return None
        if f is None:
            return m
        if m is None:
            return self.keep_whole(f"investigations[{self.inv_id}].pit",
                                   "the pit's dimensions", f)
        where = f"investigations[{self.inv_id}].pit"
        out = PitDimensions(prov=f.prov or m.prov)
        for name in ("length", "width", "depth"):
            fv, mv = getattr(f, name), getattr(m, name)
            s = self.slot(where, "pit", name, fv, mv, _same_quantity(fv, mv),
                          f.prov, m.prov,
                          has_evidence(m.prov, f.prov.bbox if f.prov else None),
                          f.prov.page if f.prov else None)
            setattr(out, name, s.value)
            if s.alternative is not None and out.prov is not None:
                out.prov = out.prov.model_copy(deep=True)
                out.prov.alternatives.append(s.alternative)
        out.method = m.method or f.method
        return out


def merge_investigations(floor: Investigation, model: Investigation,
                         log: Optional[MergeLog] = None
                         ) -> Tuple[Investigation, MergeLog]:
    """The floor and the model's answer as one record, and what the merge did.

    The floor is every value the grid placed; the model's answer is the same
    log as the model read it. The rule is in :mod:`report_ingest.floor`:
    add, correct with evidence, never drop. The merged record is a new
    object; neither input is changed.
    """
    log = log if log is not None else MergeLog()
    merged = _Merger(floor, model, log).run()
    return merged, log
