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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from report_ingest.floor import (
    DEPTH_TOL_M, LAYER_TOL_M, MergeLog, Settled, carry, depth_m, fold,
    has_evidence, same_depth, same_number, same_text, settle, show,
)
from report_ingest.model import (
    Alternative, DrillingDetails, Investigation, Layer, PitDimensions,
    Provenance, Quantity, SPT, Sample, WaterLevel, _canonical_unit,
)
from report_ingest.printed_depths import (
    PrintedDepth, PrintedDepths, PrintedRow, chained, on_line,
    printed_depths,
)

__all__ = ["seed_from_grid", "serialise_seed", "merge_investigations",
           "GROUP_SPAN_M", "pit_dimensions", "bucket_width", "header_pairs",
           "PIT_METHOD_KEYS", "n_from_drives", "header_increment",
           "N_RULE_SOURCE", "N_HALT_RULE", "ends_in_halt", "halts_before_n",
           "shows_halt"]

#: How far apart two placed cells may be and still belong to one sample: a
#: driven sample is three 6 in (150 mm) drives, 18 in (0.4572 m), long (FHWA
#: NHI-06-088 Vol. I, Sec. 3.7, p. 3-51), and its blow record, its N and its
#: index values are printed against it. It measures DEPTH; a stack of
#: printed lines is read by its lines (:func:`_read_stacks`), not by this.
GROUP_SPAN_M = 0.4572
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
#: "N=13": an N printed as a whole number -- not the start of a bound
#: ("N=50+") or of blows over a distance ('N=50/3"'), which are
#: :data:`_N_PRINTED` (RI-48).
_N_VALUE = re.compile(r"\bN\s*[=:]\s*(\d+)\b(?!\s*[+/])")
#: An N the log prints as words or a bound, not a whole number: ">50",
#: "50+", "R", "Refusal", '50/3"'. Kept as printed in ``SPT.n_printed``
#: (RI-30), never as an exact N.
_N_PRINTED = re.compile(
    r"\bN\s*[=:]\s*(>\s*\d+|\d+\s*\+"
    r"|\d+\s*/\s*\d+(?:\.\d+)?\s*(?:\"|''|in\b|cm\b|mm\b)?"
    r"|REF(?:USAL)?|R)(?![\w])", re.I)
#: The same printed bare, where the N goes -- in an N column, or on the
#: line a stack's header names the N (RI-48): "50+", ">50", "R", "REF".
_N_MARK = re.compile(r">\s*\d+|\d+\s*\+|REF(?:USAL)?\.?|R", re.I)
#: A header naming an N corrected for the hammer's energy or for
#: overburden: N60, "the energy-corrected SPT N-value adjusted to 60%
#: efficiency" (FHWA NHI-06-088 Vol. I, Sec. 3.7.1, p. 3-55, Eq. 3-2), and
#: (N1)60, N60 "normalized for overburden" (Sec. 3.7.2, p. 3-56, Eq. 3-3).
#: Neither is the N measured in the field, which SPT.n holds (RI-48).
_CORRECTED_N = re.compile(
    r"\(?\bN\s?1?\)?\s?,?\s?60\b|N₆₀|N₁₆₀", re.I)
#: A header that prints the N in parentheses after the blows: "BLOW COUNT
#: (N)", "BLOWS PER 6 IN (N)".
_WRAPPED_N = re.compile(r"\(\s*N(?:[- ]?VALUE)?\s*\)", re.I)
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


#: A drive printed as words: the sampler sank under the weight of the hammer
#: or of the rods. It is a drive, but not a count.
_WORD_DRIVE = re.compile(r"\bWO[HR]\b", re.I)


def _word_drive(entry: Any) -> bool:
    """Is this drive ``WOH`` or ``WOR`` as printed?"""
    return isinstance(entry, str) and fold(entry) in ("woh", "wor")


def _blow_tokens(text: str) -> List[Any]:
    """A blow record as the log printed it, or ``[]`` when the text is not
    one. ``5-9-12`` -> ``[5, 9, 12]``; ``12-30-50/5"`` -> ``[12, 30,
    '50/5"']``; ``5/9/12`` -> ``[5, 9, 12]``; ``2+1+2`` -> ``[2, 1, 2]``;
    ``WOH-2-3`` -> ``['WOH', 2, 3]`` (RI-44: the words stay as printed; a
    ``WOH`` alone is not a record).

    The plus is gINT's own spelling of a driven record and is as common in
    the corpus as the hyphen, so it is read here rather than left on the
    page.
    """
    raw = " ".join(str(text or "").split())
    if not raw or not re.fullmatch(r'[\d\s+\-–/",.\'inmc]+',
                                   _WORD_DRIVE.sub("0", raw)):
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
        elif _word_drive(piece):
            out.append(piece)
        elif re.fullmatch(r'\d+\s*/\s*[\d.]+\s*(?:"|\'\'|in|cm|mm)?', piece):
            out.append(piece)
        else:
            return []
    if len(out) < 2 and not (out and isinstance(out[0], str)
                             and not _word_drive(out[0])):
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
    if name in _PERCENT_LIKE and unit not in ("", "%"):
        # "W = 2.5 m" in a remark is a level, not a water content.
        return None
    if name == "qu" and re.search(r"\bPP\b|pocket|penetrom", rest, re.I):
        # "qu = 75 kPa (PP)": the pocket penetrometer's reading, which the
        # instrument prints as an unconfined strength.
        name = "pocket_pen"
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
# what a cell IS, before it is read (RI-04)
# ---------------------------------------------------------------------------
#
# The grid places text by geometry and is sometimes wrong about the column: a
# description runs into the sample-id column, a sample's id sits in a DATA
# column the header vocabulary called blows and recovery, a label and its
# value are printed on two lines. The floor reads a cell for what its TEXT
# says it is, and the column only decides between readings the text allows.

#: The columns whose cells say a sample is THERE: its id, its sampler, its
#: drives, its recovery. An index column only says what was measured on it.
_SAMPLING = frozenset({"sample_id", "sample_type", "blows", "n_value",
                       "recovery", "rqd"})
#: Columns that are not sample columns but print results with their own
#: label -- ``MC = 21%`` under TESTS, ``qu = 95 kPa`` under REMARKS. Only
#: a labelled result is read off them.
_RESULT_COLUMNS = frozenset({"tests", "remarks"})
#: A sample's own name: ``S-4``, ``SS-3``, ``4A``, ``U-2-1``, ``ST 2``, ``12``.
_ID_LIKE = re.compile(
    r"^[A-Za-z]{0,4}[-\s#.]?\d{1,4}[A-Za-z]?(?:[-/.]\d{1,4}[A-Za-z]?)?$")
#: Words a sample column prints about its sample -- the sampler, the drive,
#: the recovery -- so a cell carrying one is the sample's, not prose that
#: strayed into a sample column.
_SAMPLER_WORDS = frozenset({
    "recovery", "refusal", "hammer", "weight", "rods", "rod", "lost",
    "spoon", "split", "shelby", "tube", "ring", "core", "coring", "bulk",
    "grab", "california", "sampler", "sample", "barrel", "push", "pushed",
    "piston", "pitcher", "denison", "osterberg", "spt", "auger", "bag",
    "jar", "block", "cuttings", "drive", "driven", "standard", "thin",
    "walled", "modified", "liner", "undisturbed", "disturbed"})
#: Quantities that are percentages or plain numbers, never a length or a
#: stress: a labelled ``W = 2.5 m`` in a remark is not a water content.
_PERCENT_LIKE = frozenset({"water_content", "liquid_limit", "plastic_limit",
                           "plasticity_index", "fines"})
#: How a header prints two quantities in one column, ``A (B)``: the number
#: printed bare is A's, the number in parentheses B's.
_PAREN_WORDS: Tuple[Tuple[str, str], ...] = (
    ("rqd", r"\bRQD\b"), ("recovery", r"\bREC"),
    ("n_value", r"\bN\b|\bN[- ]?VALUE"), ("blows", r"\bBLOW"))
_TRAILING_UNIT = re.compile(r"\s*(?:%|\"|'|[A-Za-z][A-Za-z/.²³\d]{0,7})\s*$")


def _text(cell: Any) -> str:
    return " ".join(str(getattr(cell, "text", "") or "").split())


#: What a log prints in a sample-id column that is never a sample's id: no
#: recovery, weight of hammer or of rods, not applicable.
_NOT_IDS = frozenset({"nr", "nrec", "norec", "woh", "wor", "rec", "na",
                      "nd", "lost",
                      # a refusal printed where the N goes (RI-48)
                      "r", "ref", "refusal"})
#: A sample id with a letter prefix: ``S-4``, ``SS-3``, ``U-2-1``, ``ST 2``.
_PREFIXED_ID = re.compile(
    r"^[A-Za-z]{1,4}[-\s#.]?\d{1,4}[A-Za-z]?(?:[-/.]\d{1,4}[A-Za-z]?)?$")
#: A sample id that is a number: ``12``, ``4A``.
_NUMBER_ID = re.compile(r"^\d{1,4}[A-Za-z]?$")
#: A header that says its column numbers the samples.
_NUMBER_HEADER = re.compile(
    r"\bNO\b\.?|\bNUMBER\b|\bNUM\b|#|\bN[\u00ba\u00b0]", re.I)


def _id_like(text: str) -> bool:
    """Could this be a sample's name at all? Not ``NR`` or ``WOH``."""
    token = text.strip()
    if not token or fold(token) in _NOT_IDS:
        return False
    return bool(_ID_LIKE.match(token)) or (token.isalpha()
                                           and len(token) <= 3)


def _id_shape(text: str) -> str:
    """``prefixed`` for ``S-4``, ``number`` for ``12``, else empty."""
    token = text.strip()
    if not token or fold(token) in _NOT_IDS or token.lower() in _TYPE_CODES:
        return ""
    if _PREFIXED_ID.match(token):
        return "prefixed"
    if _NUMBER_ID.match(token):
        return "number"
    return ""


def _id_styles(grid: Any, cells: Sequence[Any],
               by_column: Optional[Dict[str, Tuple[str, ...]]] = None
               ) -> Dict[str, str]:
    """How each sample-id column names its samples, from the column itself.

    A column that prints any id with a letter prefix (``S-3``) names its
    samples that way, and a bare number in it is a drive or a value printed
    under the id, not a second id. A column of bare numbers names samples by
    number only when its header says so (``SAMPLE NO.``); in a column headed
    ``DATA`` a stack of ``4``, ``6``, ``7`` is a blow record. Where the header
    names the blows too, only the first line of each stack is a number
    (:func:`_read_stacks`).
    """
    shapes: Dict[str, set] = {}
    for cell in cells:
        if "sample_id" in _names(grid, cell, by_column):
            shapes.setdefault(cell.column_id, set()).add(
                _id_shape(_text(cell)))
    styles: Dict[str, str] = {}
    for column_id, found in shapes.items():
        if "prefixed" in found:
            styles[column_id] = "prefixed"
        elif "number" in found:
            column = grid.column(column_id)
            header = str(getattr(column, "header", "") or "")
            styles[column_id] = ("number" if _NUMBER_HEADER.search(header)
                                 else "")
    return styles


# ---------------------------------------------------------------------------
# a column headed as both the sample numbers and the blows (RI-44)
# ---------------------------------------------------------------------------
#
# "SAMPLE NO. / BLOWS / REC" over a stack of bare numbers, one to a line:
# the number, the drives, the recovery. The numbers look alike and the
# header names them all, so the only clue is each line's place in its stack,
# and a stack is found by the printed LINES, never by a depth: at the scale a
# log is printed, a stack of four or five lines may run well past the
# sample's own 18 in. Where the stacks cannot be found and read with
# certainty, the column is read as it was before RI-44 (no sample number
# from it, no N) and the QA section lists it: a missing N is a gap the
# reader fills, a wrong one is a wrong record.

#: How many drives a standard SPT record holds: three 6 in (150 mm)
#: increments (FHWA NHI-06-088 Vol. I, Sec. 3.7, p. 3-51).
SPT_DRIVES = 3

# The two thresholds below -- half a text height for "one line", a line and
# a half of the column's own pitch for "a blank line" -- are judgments about
# how text is laid out, not rules printed anywhere. Every case they get
# wrong leaves a stack that does not read, and the column is then read as
# before RI-44 and listed (floor_stack): they can lose a reading, not make
# one up.


def _centre(cell: Any) -> float:
    return (float(cell.bbox[1]) + float(cell.bbox[3])) / 2.0


def _same_line(a: Any, b: Any) -> bool:
    """Are two cells printed on one line of text (centres within half the
    taller one's height; layout judgment, see above)?"""
    height = max(float(a.bbox[3]) - float(a.bbox[1]),
                 float(b.bbox[3]) - float(b.bbox[1]), 1e-6)
    return abs(_centre(a) - _centre(b)) <= 0.5 * height


def _above(cell: Any, line: Any) -> bool:
    """Is ``cell`` printed on a line above ``line``'s?"""
    return _centre(cell) < _centre(line) and not _same_line(cell, line)


def _naming_line(group: Any, grid: Any,
                 by_column: Optional[Dict[str, Tuple[str, ...]]] = None
                 ) -> Any:
    """The top cell of a sample that names it -- its id, its number read by
    place in a stack, or its sampler code -- or None (RI-44)."""
    found = []
    for cell in group.cells:
        names = set(_names(grid, cell, by_column))
        text = _text(cell)
        code = text.lower()
        if (id(cell) in group.heads or _ID_AND_TYPE.match(text)
                or (code in _TYPE_CODES and (
                    "sample_type" in names
                    or (len(code) >= 2 and names & {"sample_id", "blows"})))
                or ("sample_id" in names and (
                    _id_shape(text) == "prefixed"
                    or ("blows" not in names and _id_like(text))))):
            found.append(cell)
    return min(found, key=_centre) if found else None


def _line_pitch(lines: Sequence[Any]) -> Optional[float]:
    """The distance from one printed line to the next in a column: the
    smallest step between two of its lines that are not on one line."""
    steps = [_centre(b) - _centre(a) for a, b in zip(lines, lines[1:])
             if not _same_line(a, b)]
    return min(steps) if steps else None


def _runs(lines: Sequence[Any], pitch: Optional[float]) -> List[List[Any]]:
    """The column's lines cut where a blank line opens: a step of more than
    a line and a half (layout judgment, see above). Lines printed one under
    the next stay together."""
    out: List[List[Any]] = []
    for cell in lines:
        if out and (pitch is None
                    or _centre(cell) - _centre(out[-1][-1]) <= 1.5 * pitch):
            out[-1].append(cell)
        else:
            out.append([cell])
    return out


#: What a header over such a column prints, found by its words: the sample
#: number, the blows, the N, a corrected N (:data:`_CORRECTED_N`), the
#: recovery (REC, RECOV, RECOVERY or RECOVERED as a word, never RECORD). The
#: N is the letter as a word, not "No."/"N°", not a corrected N ("N60",
#: "(N1)60") and not an abbreviation ("N.R." for no recovery, "N/A"): layout
#: judgment (RI-48).
_LAYOUT_PARTS: Tuple[Tuple[str, Any], ...] = (
    ("number", _NUMBER_HEADER),
    ("blows", re.compile(r"\bBLOW", re.I)),
    ("n", re.compile(r"\bN\b(?![º°]|\s?1?\)?\s?,?\s?60\b|\.\s?[A-Za-z]"
                     r"|/[A-Za-z])|\bN[- ]?VALUE", re.I)),
    ("n60", _CORRECTED_N),
    ("rec", re.compile(r"\bREC(?:OV(?:ERY|ERED)?)?\b", re.I)),
)
#: The column name each part of the layout places.
_LAYOUT_NAMES = {"n": "n_value", "n60": "n_value", "rec": "recovery"}


def _stack_layout(header: str) -> Optional[List[str]]:
    """What each stack prints after its drives, in the header's order --
    ``[]``, ``["rec"]``, ``["n", "rec"]`` -- or None when the header does not
    print the sample number first and the blows next ("SAMPLE NO. / REC /
    BLOWS"): a recovery printed before the drives, where it may or may not
    be there, cannot be told from the first drive, so such a column is read
    as before."""
    found = sorted((m.start(), name) for name, pattern in _LAYOUT_PARTS
                   for m in [pattern.search(str(header or ""))] if m)
    order = [name for _at, name in found]
    if order[:2] != ["number", "blows"]:
        return None
    return order[2:]


@dataclass
class _Stack:
    """One sample's lines in a column headed as both the sample numbers and
    the blows, top down: its number, its drives, and what the header prints
    after them."""

    lines: List[Any]
    recovery: Any = None
    #: The line the header's N names, read as the printed N.
    printed_n: Any = None
    #: The line the header's corrected N (N60) names: never the N (RI-48).
    n60: Any = None
    #: Lines after the drives that cannot be told apart: fewer than the
    #: header names. Kept in the note, not read as any of them.
    unassigned: List[Any] = field(default_factory=list)
    #: A cell on the number's line that anchored the stack (a sampler code).
    anchor: Any = None
    #: Set when the last drive read may instead be the N or the recovery the
    #: header prints after the blows: the drives are kept as printed and no
    #: N is computed from them.
    doubt: bool = False
    #: A sampler code ("R") printed after the number, ahead of the drives:
    #: neither a drive nor a refusal, and not read.
    sampler: Any = None
    #: Set when a line where the header prints the N or the recovery reads
    #: as blows: it is in ``unassigned``, never either.
    blows_in_tail: bool = False


def _read_stack(lines: Sequence[Any], after: Sequence[str], complete: bool
                ) -> Optional[_Stack]:
    """A stack read as a sample number, one to :data:`SPT_DRIVES` drives,
    then what the header prints after the blows (``after``, from
    :func:`_stack_layout`), one line each, in the header's order; None when
    it does not read so.

    The drives are bare counts, ``WOH`` or ``WOR`` one to a line, or one
    printed record (``4-6-7``). They end at the third drive or at a refusal
    (``50/3"``), where the drive halts (p. 3-51); the lines after that are
    the header's N and recovery. A refusal printed in place of a drive
    ("12" / "30" / "R") ends them too, and is the record's printed N, in
    place of the N line the header names; so is one printed after the
    drives where the header names no N line (layout judgment, RI-48). An
    "R" printed after the number and ahead of every drive is the sampler's
    code (a ring sampler), as the floor read it before RI-48, and is
    skipped; a "REF" or "50+" there is a refusal. Where a refusal ends the
    drives, a line after it that reads as blows is not its N or recovery:
    the stack does not read (layout judgment). A number with no drive under
    it does not
    read: it may be a sample started a line early. With ``complete``, a
    stack whose drives do not end so does not read either: with no sampler
    code or drawn box to say where a sample starts, a short stack may be the
    tail of another.
    """
    head = lines[0]
    if _id_shape(_text(head)) != "number":
        return None
    entries, done, last = 0, False, ""
    tail: List[Any] = []
    mark: Any = None
    sampler: Any = None
    for above, cell in zip(lines, lines[1:]):
        if _same_line(above, cell):
            return None               # two items on one line: not a stack
        text = _text(cell)
        if _N_VALUE.search(text) or _N_PRINTED.search(text):
            continue                  # a printed N, read as one
        if (entries == 0 and sampler is None and mark is None
                and _sampler_letter(text)):
            sampler = cell            # "R" ahead of the drives: the sampler
            continue
        if done and mark is None and "n" not in after \
                and _marks_refusal(text):
            mark = cell               # its refusal, where no N line is named
            continue
        if done:
            tail.append(cell)
            continue
        tokens = _blow_tokens(text)
        if re.fullmatch(r"\d{1,3}", text) or _word_drive(text):
            entries += 1
            last = "count" if text.isdigit() else "word"
        elif tokens:
            entries += len(tokens)
            last = "record"
            done = any(isinstance(t, str) and not _word_drive(t)
                       for t in tokens)
        elif _marks_refusal(text):
            mark, done = cell, True   # a refusal in place of a drive
            continue
        else:
            return None
        if entries > SPT_DRIVES:
            return None
        done = done or entries == SPT_DRIVES
    if mark is not None:
        rest = [part for part in after if part != "n"]
        if len(tail) > len(rest) or any(_blow_tokens(_text(cell))
                                        for cell in tail):
            return None
        stack = _Stack(list(lines))
        stack.printed_n = mark
        stack.sampler = sampler
        if len(tail) == len(rest):
            for part, cell in zip(rest, tail):
                if _marks_refusal(_text(cell)):
                    return None
                if part == "rec":
                    stack.recovery = cell
                else:
                    stack.n60 = cell
        else:
            stack.unassigned = tail   # the drives end at the mark: no doubt
        return stack
    if entries == 0 or (complete and not done) or len(tail) > len(after):
        return None
    stack = _Stack(list(lines))
    stack.sampler = sampler
    if tail and len(tail) < len(after) and after[0] == "n" \
            and _marks_refusal(_text(tail[0])):
        # Fewer lines than the header names, the first a refusal printed
        # where the N goes, and the header prints the N first: that line is
        # the N, so the last count before it is a drive (layout judgment,
        # RI-48). The lines after it cannot be told apart.
        stack.printed_n = tail[0]
        stack.unassigned = tail[1:]
        return stack
    if len(tail) == len(after):
        for part, cell in zip(after, tail):
            if _marks_refusal(_text(cell)) and part not in ("n", "n60"):
                return None           # a printed refusal only where an N is
            if _blow_tokens(_text(cell)):
                # A line that reads as blows ('33-18-25', '50/3"') where the
                # header prints the N or the recovery is neither: not a
                # recovery of 33 %. Kept in the note and listed (layout
                # judgment).
                stack.unassigned.append(cell)
                stack.blows_in_tail = True
            elif part == "rec":
                stack.recovery = cell
            elif part == "n60":
                stack.n60 = cell
            else:
                stack.printed_n = cell
    else:
        # Fewer lines than the header names after the blows: which is which
        # is not known, and the last count may itself be one of them.
        stack.unassigned = tail
        stack.doubt = last == "count"
    return stack


def _marks_refusal(text: str) -> bool:
    """Is this line a refusal printed where an N goes ("R", "50+",
    "(REF)")?"""
    return bool(_N_MARK.fullmatch(_unwrap(text)))


def _sampler_letter(text: str) -> bool:
    """Is this line an "R" alone ("R", "(R)"): a ring sampler's code ahead
    of the drives, a refusal after one (:func:`_read_stack`)?"""
    return _marks_refusal(text) and len(_unwrap(text)) == 1


def _unwrap(text: str) -> str:
    """The text inside the parentheses it is printed in, if any."""
    t = text.strip()
    return t[1:-1].strip() if _wrapped(t) else t


def _number_of(cell: Any) -> int:
    return int(re.match(r"\d+", _text(cell)).group(0))


def _read_stacks(grid: Any, cells: Sequence[Any], styles: Dict[str, str],
                 by_column: Optional[Dict[str, Tuple[str, ...]]] = None
                 ) -> Tuple[List[_Stack], List[List[Any]]]:
    """``(stacks read, runs not read)`` for every column whose header names
    both the samples' numbers and their blows and which numbers its samples
    (:func:`_id_styles`); a column printing ``S-3`` names its samples by
    that, and a bare number in it is never an id.

    A sample starts on the line of its sampler code in a TYPE column, or at
    the first line inside a box ``log_grid`` drew; with neither on the
    page, at a blank line. A column is read only when EVERY line of it on
    the page falls in a stack that reads (:func:`_read_stack`) and the
    numbers its stacks start with run on down the hole, each the one before
    or the next: a sampler code printed a line below the number, or a stack
    the blank lines cut wrongly, would otherwise start a sample at a blow
    count. Any other column is returned as its runs, to be read as before
    and listed.
    """
    columns: Dict[Tuple[int, str], List[Any]] = {}
    anchors: Dict[int, List[Any]] = {}
    for cell in cells:
        names = set(_names(grid, cell, by_column))
        both = {"sample_id", "blows"} <= names
        # A lone "R" in this column is read by its place in the stack
        # (:func:`_read_stack`): ahead of the drives the sampler's code,
        # after one the refusal it prints (RI-48).
        if _text(cell).lower() in _TYPE_CODES and not (
                both and "sample_type" not in names
                and _N_MARK.fullmatch(_text(cell))):
            if "sample_type" in names and not both:
                anchors.setdefault(cell.page, []).append(cell)
            continue                  # the sampler: neither number nor drive
        if both and styles.get(cell.column_id) == "number":
            columns.setdefault((cell.page, cell.column_id), []).append(cell)
    boxes = list(getattr(grid, "samples", None) or [])
    read: List[_Stack] = []
    unread: List[List[Any]] = []
    for (page, column_id), lines in sorted(columns.items()):
        lines.sort(key=lambda c: (_centre(c), float(c.bbox[0])))
        header = str(getattr(grid.column(column_id), "header", "") or "")
        pitch = _line_pitch(lines)
        after = _stack_layout(header)
        # Every quantity the column is named for must have its place in the
        # layout: a recovery or an N the header's words do not place may be
        # any line of the stack (a lost drive would shift it into the
        # drives), so such a column is read as before.
        named = set(_names(grid, lines[0], by_column)) & (
            _SAMPLING | set(_INDEX_NAMES))
        unplaced = named - {"sample_id", "blows", "sample_type"} - {
            _LAYOUT_NAMES[part] for part in (after or ())}
        if after is None or unplaced:
            unread.extend(_runs(lines, pitch))
            continue
        # No stack starts at an "R" alone: a sampler code printed on its
        # line, or a box whose first line it is, is a line low, and does
        # not start a sample there (layout judgment).
        starts: Dict[int, Any] = {}
        for anchor in anchors.get(page, ()):
            on = [k for k, c in enumerate(lines) if _same_line(c, anchor)]
            if on and not _sampler_letter(_text(lines[on[0]])):
                starts.setdefault(on[0], anchor)
        for box in boxes:
            pages = (set(getattr(box, "pages", ()) or ())
                     | {getattr(box, "page", None)})
            if page not in pages:
                continue
            tol = float(getattr(box, "plus_minus", None) or 0.05)
            inside = [k for k, c in enumerate(lines)
                      if float(box.top) - tol <= float(c.depth)
                      <= float(box.bottom) + tol]
            if inside and not _sampler_letter(_text(lines[inside[0]])):
                starts.setdefault(inside[0], None)
        if starts:
            order = sorted(starts)
            pieces = [lines[k:order[n + 1]] if n + 1 < len(order)
                      else _runs(lines[k:], pitch)[0]
                      for n, k in enumerate(order)]
            stacks = [_read_stack(p, after, complete=False)
                      for p in pieces]
            covered = (order[0] == 0
                       and order[-1] + len(pieces[-1]) == len(lines))
            for stack, k in zip(stacks, order):
                if stack is not None:
                    stack.anchor = starts[k]
        else:
            pieces = _runs(lines, pitch)
            stacks = [_read_stack(p, after, complete=True)
                      for p in pieces]
            covered = True
        numbers = [_number_of(s.lines[0]) for s in stacks if s is not None]
        if (covered and all(s is not None for s in stacks)
                and all(b - a in (0, 1) and x.lines[0] is not y.lines[0]
                        and _text(x.lines[0]) != _text(y.lines[0])
                        for a, b, x, y in zip(numbers, numbers[1:],
                                              stacks, stacks[1:]))):
            read.extend(stacks)                      # type: ignore[arg-type]
        else:
            unread.extend(_runs(lines, pitch))
    return read, unread


def _is_words(text: str) -> bool:
    """Is this cell words -- a description, a remark, a rule of dashes --
    rather than anything a sample column holds?

    A sample column holds an id, a sampler code, drives, an N, a recovery,
    a labelled result or a number. A cell of two or more words, or one long
    word, that is none of those is text the grid placed in the wrong column
    (``see attached log``, ``grey silty SAND, moist``, ``Bottom of hole at
    6.2 m``); a cell with no letter or digit at all is a drawn rule.
    """
    t = " ".join(text.split())
    if not t:
        return False
    if not any(ch.isalnum() for ch in t):
        return True
    if (_id_like(t) or t.lower() in _TYPE_CODES or _ID_AND_TYPE.match(t)
            or _blow_tokens(t) or _labelled_value(t) is not None
            or _N_VALUE.search(t) or _N_PRINTED.search(t)
            or _INTERVAL.match(t)):
        return False
    words = re.findall(r"[A-Za-z]{3,}", t)
    if any(w.lower() in _SAMPLER_WORDS for w in words):
        return False
    return len(words) >= 2 or any(len(w) >= 5 for w in words)


#: A number and the unit printed after it, as a numeric cell holds them.
_NUMBER_AND_UNIT = re.compile(
    r"-?\d+(?:[.,]\d+)?\s*(?:%|\"|''|'|cm|mm|inch(?:es)?|in|ft|feet|m|pcf|psf"
    r"|ksf|tsf|psi|kpa|mpa|kn/m[3³]|lbs?/ft[3³]|g/cm[3³]"
    r"|kg/cm[2²]|tons?/ft[2²]?)?(?![A-Za-z])", re.I)


def _numeric(text: str) -> bool:
    """Is the cell numbers, each with at most its unit -- ``17``, ``31,4``,
    ``(8)``, ``45 cm``, ``42-21-21``, ``29cm, 64%``? A cell with words in it
    (``S-1, SS``, ``Run = 3.5 ft``, ``% Passing #200``) is something else,
    and its number is not that column's value."""
    t = text.strip()
    if not any(ch.isdigit() for ch in t):
        return False
    rest = _NUMBER_AND_UNIT.sub(" ", t)
    return bool(re.fullmatch(r"[\s.,()\-–/<>~+]*", rest))


def _numbers_of(cell: Any) -> List[float]:
    """The cell's numbers, with a decimal comma read as one: ``31,4`` is
    31.4, where a comma before three digits stays a thousands mark."""
    t = _TRAILING_UNIT.sub("", _text(cell))
    match = re.fullmatch(r"\s*\(?\s*(-?\d+),(\d{1,2})\s*\)?\s*", t)
    if match:
        return [float(f"{match.group(1)}.{match.group(2)}")]
    return [float(n) for n in (getattr(cell, "numbers", None) or [])]


def _wrapped(text: str) -> bool:
    t = text.strip()
    return t.startswith("(") and t.endswith(")")


def _paren_names(header: str, names: Sequence[str]
                 ) -> Optional[Tuple[str, str]]:
    """``(bare, in parentheses)`` for a column whose header prints two of its
    quantities as ``A (B)``: ``REC % (RQD %)`` -> ``(recovery, rqd)``,
    ``BLOWS PER 6 IN (N)`` -> ``(blows, n_value)``. None otherwise, which
    leaves the column read as before."""
    match = re.search(r"^(.*?)\(([^)]*)\)", str(header or ""))
    if match is None:
        return None

    def which(part: str) -> Optional[str]:
        found = [name for name, pattern in _PAREN_WORDS
                 if name in names and re.search(pattern, part, re.I)]
        return found[0] if len(found) == 1 else None

    bare, wrapped = which(match.group(1)), which(match.group(2))
    if bare and wrapped and bare != wrapped:
        return bare, wrapped
    return None


def _marks_a_sample(text: str, names: Sequence[str]) -> bool:
    """Does this cell say a sample is here (its id, sampler, drives, N,
    recovery), as against a value measured on one?"""
    here = set(names)
    if not here & _SAMPLING:
        return False
    code = text.lower()
    if code in _TYPE_CODES and ("sample_type" in here or (
            len(code) >= 2 and here & {"sample_id", "blows"})):
        return True
    if _ID_AND_TYPE.match(text):
        return True
    if here & {"blows", "n_value", "sample_id"} and (
            _blow_tokens(text) or _N_VALUE.search(text)
            or _N_PRINTED.search(text)):
        return True
    labelled = _labelled_value(text)
    if labelled is not None:
        return labelled[0] in ("recovery", "rqd")
    if "sample_id" in here and (_INTERVAL.match(text) or _id_like(text)):
        return True
    if fold(text) in _NOT_IDS:
        return False
    return not (here & set(_INDEX_NAMES)) and bool(_NUMBER.search(text))


def _sample_key(text: str, names: Sequence[str], style: str = ""
                ) -> Optional[Tuple[str, str]]:
    """What makes this cell ONE sample's and no other's: its id, or its
    whole blow record. Two cells with different keys of one kind on one
    sheet are two samples, however close they are printed. An id counts
    only in the shape its column names samples in (``style``, from
    :func:`_id_styles`): a bare ``4`` under ``S-3`` is not a second id."""
    pair = _ID_AND_TYPE.match(text)
    if pair is not None and pair.group(2).lower() in _TYPE_CODES:
        return "id", fold(pair.group(1))
    if "sample_id" in names and style and _id_shape(text) == style:
        return "id", fold(text)
    if set(names) & {"blows", "n_value", "sample_id"} and _blow_tokens(text):
        return "drives", fold(text)
    return None


class _Joined:
    """A label and its value printed on two lines of one column -- ``%
    Passing #200`` above ``= 41.5`` -- read as the one cell they are. The
    label's line places it."""

    __slots__ = ("_cell", "text", "bbox", "numbers")

    def __init__(self, label: Any, value: Any) -> None:
        self._cell = label
        self.text = f"{_text(label)} {_text(value)}"
        a, b = label.bbox, value.bbox
        self.bbox = (min(a[0], b[0]), min(a[1], b[1]),
                     max(a[2], b[2]), max(a[3], b[3]))
        self.numbers = list(getattr(value, "numbers", None) or [])

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cell, name)


def _join_labels(cells: Sequence[Any]) -> List[Any]:
    """Each label printed without its value, joined to the ``= value`` line
    under it in the same column; every other cell as it was."""
    out: List[Any] = []
    used: set = set()
    ordered = sorted(cells, key=lambda c: (c.page, c.column_id, c.bbox[1]))
    for k, cell in enumerate(ordered):
        if id(cell) in used:
            continue
        text = _text(cell)
        nxt = ordered[k + 1] if k + 1 < len(ordered) else None
        key = "".join(ch for ch in text.lower() if ch.isalnum())
        if (nxt is not None and key in _LABEL_NAMES and "=" not in text
                and ":" not in text and nxt.page == cell.page
                and nxt.column_id == cell.column_id
                and re.match(r"^\s*[=:]", _text(nxt))
                and 0.0 <= float(nxt.bbox[1]) - float(cell.bbox[1])
                <= 2.5 * max(1.0, float(cell.bbox[3]) - float(cell.bbox[1]))):
            out.append(_Joined(cell, nxt))
            used.add(id(nxt))
            continue
        out.append(cell)
    return out


# ---------------------------------------------------------------------------
# the seed
# ---------------------------------------------------------------------------

def _span_in_unit(unit: str, grid: Any) -> float:
    """:data:`GROUP_SPAN_M` in the log's depth unit.

    With no unit, the span is 1.5 where the rulers step by 2 or more and
    0.5 otherwise: 18 in written in feet, or rounded to metres, by a guess
    at the unit from the ruler's step. That fallback is a working
    assumption with no printed source (RI-47); only the 18 in it stands for
    is printed (p. 3-51, at :data:`GROUP_SPAN_M`).
    """
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
        #: Values printed off the sample's own lines and bound to it by the
        #: page's geometry (RI-04): read with the sample, but they neither
        #: set its depth nor have to sit inside its drawn box.
        self.bound: List[Any] = []
        #: Values printed off every sample's lines on a sheet that has
        #: samples, which the geometry could not bind to one (RI-04).
        self.unbound = False
        #: From a stack in a column headed as both the sample numbers and
        #: the blows (:func:`_read_stacks`, RI-44): the cell that is the
        #: sample's number by its place, and the one that is its recovery.
        self.heads: set = set()
        self.recovery: set = set()
        #: The stack's line the header's N names, and lines after the drives
        #: that cannot be told apart (kept in the note).
        self.printed_n: set = set()
        self.unassigned: set = set()
        #: The stack's line a header's corrected N (N60) names (RI-48).
        self.n60: set = set()
        #: The stack's sampler code printed ahead of its drives ("R").
        self.sampler: set = set()
        #: Why no N is computed from this sample's drives (RI-44), or empty.
        self.doubt = ""
        #: Set when the group holds lines of a stack that did not read.
        self.unread_stack = False

    def takes(self, cell: Any, span: float,
              keys: Optional[Dict[int, Tuple[str, str]]] = None) -> bool:
        if abs(float(cell.depth) - float(self.first)) > span:
            return False
        mine = keys.get(id(cell)) if keys else None
        if mine is None:
            return True
        # A second id, or a second blow record, on the same sheet is a
        # second sample however close it is printed (RI-04).
        return not any(c.page == cell.page and keys.get(id(c)) is not None
                       and keys[id(c)][0] == mine[0]
                       and keys[id(c)][1] != mine[1]
                       for c in self.cells)

    def add(self, cell: Any) -> None:
        if not any(c.page == cell.page and c.column_id == cell.column_id
                   and c.text == cell.text and c.bbox == cell.bbox
                   for c in self.cells):
            self.cells.append(cell)


def _drawn_box(group: _Group, boxes: Sequence[Any]) -> Any:
    """The sample box ``log_grid`` drew round this group's rows, or None.

    A box holds the group when its interval, widened by its own
    uncertainty, takes every row's depth on a sheet the box is on; the
    narrowest such box wins. A grid with no ``samples`` (an older planlens,
    or a form that draws none) leaves the rows as they were.
    """
    best, best_span = None, None
    for box in boxes:
        pages = set(getattr(box, "pages", ()) or ()) | {getattr(box, "page", None)}
        tol = float(getattr(box, "plus_minus", None) or 0.05)
        if all(c.page in pages
               and float(box.top) - tol <= float(c.depth) <= float(box.bottom) + tol
               for c in group.cells):
            span = float(box.bottom) - float(box.top)
            if best is None or span < best_span:
                best, best_span = box, span
    return best


def _seed_sample(group: _Group, grid: Any, unit: str, confidence_floor: float,
                 by_column: Optional[Dict[str, Tuple[str, ...]]] = None,
                 template_note: str = "", drawn: Any = None,
                 printed_row: Optional[PrintedRow] = None,
                 printed_bottom: Optional[float] = None,
                 notes: Optional["_FloorNotes"] = None
                 ) -> Tuple[Optional[Sample], Optional[SPT]]:
    # The sample's depth is where its own cells -- id, sampler, drives,
    # recovery -- stand; a value printed a line above them is the sample's
    # but does not move it (RI-04). A group with none of those is placed by
    # all of its cells, as before.
    marked = [c for c in group.cells
              if _marks_a_sample(_text(c), _names(grid, c, by_column))]
    top = min(float(c.depth) for c in (marked or group.cells))
    bottom: Optional[float] = None
    sample_id = ""
    kind = "other"
    blows: List[Any] = []
    n: Optional[int] = None
    refusal = False
    n_printed = ""
    values: Dict[str, Any] = {}
    units: Dict[str, str] = {}
    texts: List[str] = []
    confidence = min(float(c.confidence) for c in group.cells)
    blow_cell = None
    stacked: List[Tuple[Tuple[float, float, float], int]] = []
    stacked_cells: List[Any] = []
    #: What a drive column printed that is not a count: an N computed with
    #: it left out would be wrong.
    unread: List[str] = []
    #: Numbers printed in a column that holds several quantities, which
    #: their position cannot tell apart: ``(the number as printed, header)``.
    unassigned: List[Tuple[str, str]] = []
    #: A corrected N (N60) the log prints: never the N (RI-48).
    corrected: List[Tuple[str, str]] = []
    #: An N or a refusal printed where the floor cannot tell whose it is
    #: (RI-48): ``(the text as printed, header)``.
    unplaced: List[Tuple[str, str]] = []
    for cell in group.cells + group.bound:
        names = _names(grid, cell, by_column)
        text = _text(cell)
        texts.append(text)
        if id(cell) in group.heads:
            # The first line of a stack under a header naming both the
            # sample numbers and the blows: the sample's number (RI-44).
            if not sample_id:
                sample_id = text
            continue
        if id(cell) in group.sampler:
            continue                  # the stack's sampler code, not read
        if id(cell) in group.recovery:
            names = ("recovery",)     # the stack's line after its drives
        if id(cell) in group.n60:
            corrected.append((text, _header(grid, cell)))
            continue
        if id(cell) in group.printed_n:
            # The line the header's N names: the N, bare or in parentheses
            # ("(13)" under "BLOW COUNT (N)"), or a refusal printed there
            # ("R", "(REF)").
            inner = text.strip()[1:-1].strip() if _wrapped(text) else text
            if re.fullmatch(r"\d{1,3}", inner):
                n = int(inner)
                continue
            if _N_MARK.fullmatch(inner):
                n_printed, refusal = re.sub(r"\s+", "", inner), True
                continue
        if id(cell) in group.printed_n or id(cell) in group.unassigned:
            unassigned.append((text, _header(grid, cell)))
            continue
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
        # A sample named and typed in one cell: "S-1, SPT". Only in a column
        # that carries sampling data -- a form's DATA column stacks the id,
        # the drives and the recovery -- so a comma in a description cannot
        # become a sampler code.
        pair = (_ID_AND_TYPE.match(text)
                if set(names) & (_SAMPLING - {"n_value"}) else None)
        if pair is not None and pair.group(2).lower() in _TYPE_CODES:
            if not sample_id:
                sample_id = pair.group(1)
            if kind == "other":
                kind = _TYPE_CODES[pair.group(2).lower()]
            continue
        # A label printed without its value ("% Passing #200" with no "= n"
        # under it) is a label: its digits are not the column's number.
        if "".join(ch for ch in code if ch.isalnum()) in _LABEL_NAMES \
                and not _numeric(text):
            continue
        # A result that names itself: "MC = 21%", "REC=29cm, 56%", and
        # under TESTS or REMARKS, "qu = 95 kPa".
        labelled = (_labelled_value(text)
                    if set(names) & (_LABELLED_FAMILY | _RESULT_COLUMNS)
                    else None)
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
        printed = _N_PRINTED.search(text)
        if printed and ("blows" in names or "n_value" in names
                        or "sample_id" in names):
            n_printed = re.sub(r"\s+", "", printed.group(1))
            refusal = True
            continue
        if (("n_value" in names or "blows" in names)
                and set(names) <= {"n_value", "blows"}
                and (_marks_refusal(text)
                     or re.fullmatch(r"\d{1,3}", text))):
            n_header = _header(grid, cell)
            if ("n_value" in names and _CORRECTED_N.search(n_header)
                    and not _drive_header(n_header)):
                # A column the header names N60 or (N1)60 and no drive
                # increment: its numbers are corrected Ns, never the N, and
                # a BLOWS there is the N's unit ("SPT N60 (blows/ft)")
                # (RI-48; layout judgment).
                corrected.append((text, n_header))
                continue
            mark = text.strip()[1:-1].strip() if _wrapped(text) else text
            if _N_MARK.fullmatch(mark) and (
                    mark[0].isalpha() or "blows" not in names):
                # A refusal printed bare where the N or the blows go (RI-48):
                # "50+" and ">50" in an N column, "R" or "REF" in either --
                # a single letter is a sampler only under a TYPE header.
                # Kept as printed, and the record is a refusal; the drives'
                # N is set against it below.
                n_printed, refusal = re.sub(r"\s+", "", mark), True
                continue
        tokens = _blow_tokens(text) if ("blows" in names or "n_value" in names
                                        or "sample_id" in names) else []
        if tokens:
            blows = tokens
            refusal = any(isinstance(t, str) and not _word_drive(t)
                          for t in tokens)
            unread.extend(t for t in tokens if _word_drive(t))
            blow_cell = cell
            continue
        drive_column = ("blows" in names
                        and not set(names) & set(_INDEX_NAMES))
        # A drive printed as words -- WOH, WOR: the sampler sank under the
        # weight of the hammer or of the rods -- keeps its place among the
        # drives AS PRINTED, never as a number (RI-04).
        if drive_column and fold(text) in ("woh", "wor"):
            stacked.append(((float(cell.depth), float(cell.bbox[1]),
                             float(cell.bbox[0])), text))
            stacked_cells.append(cell)
            unread.append(text)
            continue
        # From here on a cell is read as a NUMBER of its column, so it must
        # be one: a cell with words in it ("S-1, SS" under a DATA column the
        # vocabulary called recovery, "Run = 3.5 ft" under RQD) is not that
        # column's value, whatever digits it holds (RI-04). Its text stays
        # in the note.
        if not _numeric(text) and not (
                "sample_id" in names and _id_like(text)):
            if drive_column and text and not (
                    "sample_id" in names and _id_shape(text)):
                unread.append(text)
            continue
        header = _header(grid, cell)
        index_names = [name for name in _INDEX_NAMES if name in names]
        paren = (_paren_names(header, names)
                 if len(set(names) & _SAMPLING) >= 2 else None)
        # A blow record printed one increment to a line, each in its own
        # cell, which is what the two commonest gINT forms in the corpus do.
        # Gathered here and assembled after the loop, in the order the form
        # printed them; a LONE number is left where it is, because one
        # number in a blows column is as likely to be an N value. Not in a
        # column that also holds index values: there a lone number may be
        # any of them.
        if ("blows" in names and not index_names
                and re.fullmatch(r"\d{1,3}", text)):
            stacked.append(((float(cell.depth), float(cell.bbox[1]),
                             float(cell.bbox[0])), int(text)))
            stacked_cells.append(cell)
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
        if ("blows" in names and _wrapped(text)
                and _CORRECTED_N.search(header)):
            # "BLOWS PER 6 IN (N60)": the number in parentheses is the
            # corrected N, never the N (RI-48).
            corrected.append((text, header))
            continue
        if ((paren == ("blows", "n_value")
             or ("blows" in names and _WRAPPED_N.search(header)))
                and _wrapped(text)):
            # "BLOWS PER 6 IN (N)", "BLOW COUNT (N)": what is printed in
            # parentheses is the N -- a number, or a refusal printed there
            # ("(50+)", "(R)"). Not in a stack of sample numbers and blows
            # that did not read: which sample it is the N of is not known
            # there, so it is kept in the note (RI-48).
            inner = text.strip()[1:-1].strip()
            if group.unread_stack and "sample_id" in names:
                unplaced.append((text, header))
                continue
            if re.fullmatch(r"\d+", inner):
                n = int(inner)
                continue
            if _N_MARK.fullmatch(inner):
                n_printed, refusal = re.sub(r"\s+", "", inner), True
                continue
        if _marks_refusal(text) and ("blows" in names or "n_value" in names
                                     or "sample_id" in names):
            # A refusal printed where none of the above places it -- in a
            # stack that did not read, or under a header that also names a
            # recovery: kept in the note, never a recovery, a value or an id
            # (RI-48).
            unplaced.append((text, header))
            continue
        if "sample_id" in names and text and not cell.numbers:
            if not sample_id and _id_like(text):
                sample_id = text
            continue
        if "sample_id" in names and _id_shape(text) == "prefixed":
            # "S-3" under a DATA column the vocabulary also called blows
            # and recovery: the sample's name, not a recovery of 3 %.
            if not sample_id:
                sample_id = text
            continue
        if "sample_id" in names and text and len(cell.numbers) == 1 \
                and re.fullmatch(r"[A-Za-z]*-?\d+[A-Za-z]?", text) \
                and "blows" not in names:
            if not sample_id:
                sample_id = text
            continue
        numbers = _numbers_of(cell)
        if not numbers:
            continue
        if paren in (("recovery", "rqd"), ("rqd", "recovery")):
            # "REC % (RQD %)": the bare number is the recovery, the one
            # in parentheses the RQD -- or the other way round.
            name = paren[1] if _wrapped(text) else paren[0]
            names = (name,)
        if "rqd" in names:
            values["rqd_percent"] = numbers[0]
            continue
        if "recovery" in names:
            col_unit = _unit_of_column(grid, cell)
            number = numbers[0]
            if "%" in text or col_unit == "%" or "%" in header \
                    or re.search(r"\bpercent\b|\bpct\b", header, re.I) \
                    or (not col_unit and number <= 100.0):
                values["recovery_percent"] = number
            else:
                values["recovery"] = number
                units["recovery"] = col_unit
            continue
        if not index_names:
            continue
        limits = [name for name in ("liquid_limit", "plastic_limit",
                                    "plasticity_index") if name in names]
        order = _limits_order(header) or (
            ["liquid_limit", "plastic_limit", "plasticity_index"]
            if len(numbers) == 3 else limits)
        if limits and len(numbers) >= 2:
            # "42-21-21" under LL-PL-PI: the header says which is which.
            for name, number in zip(order, numbers):
                values.setdefault(name, number)
            continue
        if len(index_names) > 1:
            # One number under a heading of several quantities -- a plot
            # column with a scale for each -- is any of them, and its
            # position does not say which (RI-04). It is the sample's, so it
            # is kept with the sample, in its note, not guessed into a slot.
            unassigned.append((text, header))
            continue
        name = index_names[0]
        key = "fines_percent" if name == "fines" else name
        values.setdefault(key, numbers[0])
        if name in ("dry_unit_weight", "qu", "pocket_pen"):
            units[key] = _unit_of_column(
                grid, cell, "pcf" if name == "dry_unit_weight"
                and "pcf" in header.lower() else "")
    if not blows and len(stacked) >= 2:
        stacked.sort(key=lambda row: row[0])
        blows = [number for _key, number in stacked]
    elif (blows and blow_cell is not None and len(blows) < SPT_DRIVES
          and not group.doubt):
        # A record printed in one cell with fewer than three drives takes the
        # drives printed one to a line beside it, in the place they are
        # printed (RI-44): a WOH or WOR line ("WOH" over "2-3"), and the
        # counts printed ABOVE a record that ends in a refusal ("25" over
        # '50/3"'), since the drive is halted at the refusal (p. 3-51). Never
        # past three drives, never from above the line that names the
        # sample (its id or sampler code: above it is the sample before), and
        # never a count in a sample-number column unless a stack read it as
        # a drive (there it may be a sample's number). A lone number beside
        # a full record, or below a refusal, stays out, as above: it may be
        # an N. A sample whose stack did not read gets none of this.
        at = (float(blow_cell.depth), float(blow_cell.bbox[1]),
              float(blow_cell.bbox[0]))
        naming = _naming_line(group, grid, by_column)
        rows = [(row, c) for row, c in zip(stacked, stacked_cells)
                if naming is None or not _above(c, naming)]
        extra = [row for row, _c in rows if _word_drive(row[1])]
        if (naming is not None and isinstance(blows[-1], str)
                and not _word_drive(blows[-1])):
            extra += [row for row, c in rows
                      if isinstance(row[1], int) and row[0] < at
                      and (group.heads or "sample_id"
                           not in _names(grid, c, by_column))]
        extra.sort(key=lambda row: row[0])
        if len(blows) + len(extra) <= SPT_DRIVES:
            blows = ([w for key, w in extra if key < at] + list(blows)
                     + [w for key, w in extra if key >= at])
    if n_printed:
        # A refusal printed where the N goes is the log's own word: the
        # record is a refusal whatever its drives say (RI-48).
        refusal = True
    elif refusal and not _halted(blows):
        # Blows over a whole increment ('4/6"-6/6"-7/6"') are drives, not a
        # halt (RI-44). A halt of whole numbers alone ([21, 57]) is flagged
        # in the merge, after the reader's brief (RI-50: halt_flag).
        refusal = False
    # The sampler is the one the log prints. Drives alone do not say SPT: a
    # ring or California sampler is driven too, and the model reads the
    # symbol (RI-04; the floor no longer calls every driven sample an SPT).
    anchor = blow_cell or group.cells[0]
    page = int(anchor.page)
    drawn_note = ""
    if drawn is not None:
        # The sample's box, drawn against the depth scale (planlens
        # log_grid, PL-24). An interval the log PRINTS as numbers stays; the
        # drawn box seats the rows that print none.
        if bottom is None:
            top, bottom = float(drawn.top), float(drawn.bottom)
            pm = getattr(drawn, "plus_minus", None)
            drawn_note = ("; interval from the sample box drawn on the log"
                          + (f" (+/- {float(pm):.3g} {unit})".rstrip()
                             if pm else ""))
        label = str(getattr(drawn, "label", "") or "").strip()
        if label and not sample_id:
            sample_id = label
            drawn_note += "; sample id from the drawn box"
    printed_note = ""
    if printed_row is not None:
        # A page with no ruler (RI-18): these cells stand on the line of a
        # printed depth, and that depth is the sample's. An interval the
        # sample prints in its own cell, read above, stands; the base the
        # line prints (``printed_bottom``) fills one it did not print. The
        # grid gives every cell on such a page its no-ruler confidence,
        # which the printed depth replaces.
        if bottom is None:
            bottom = printed_bottom
        if not sample_id and printed_row.label:
            sample_id = printed_row.label
        confidence = printed_row.first.confidence
        printed_note = "; " + printed_row.note()
    at = f"{top:g} {unit}".strip()
    # THE CHECKS (RI-04): a value the grid may have put in the wrong row or
    # column keeps a lower confidence and a note, and goes to QA. Nothing
    # is dropped; a percentage the record cannot hold stays in the note.
    remarks: List[str] = []
    if unassigned:
        remarks.append("printed with it but not assigned to a quantity: "
                       + ", ".join(f"{t} (column '{h[:60]}')"
                                   for t, h in unassigned))
        if notes is not None:
            for t, h in unassigned:
                notes.add(page, "unassigned", f"{t} at {at}", header=h)
    if unplaced:
        remarks.append("an N or a refusal printed near it whose sample "
                       "cannot be told; not read as its N: "
                       + ", ".join(f"{t} (column '{h[:60]}')"
                                   for t, h in unplaced))
        if notes is not None:
            for t, h in unplaced:
                notes.add(page, "unplaced_n", f"{t} at {at}", header=h)
    if group.unbound:
        confidence = min(confidence, confidence_floor)
        remarks.append("values printed off every sample's lines and not "
                       "bound to one sample; placed where printed")
    doubts = _doubts(values, units)
    if doubts:
        confidence = min(confidence, confidence_floor)
        remarks.append("doubtful: " + "; ".join(doubts))
        if notes is not None:
            notes.add(page, "doubtful", f"at {at}: " + "; ".join(doubts))
    def drives_n() -> Optional[Tuple[int, str]]:
        """:func:`n_from_drives` on this sample's drives."""
        drive_cell = blow_cell or (stacked_cells[0] if stacked_cells
                                   else None)
        printed_inc = (header_increment(_header(grid, drive_cell))
                       if drive_cell is not None else None)
        interval = (None if bottom is None
                    else _metres(float(bottom) - float(top), unit))
        return n_from_drives(
            blows, sampler=kind, increment_m=printed_inc,
            interval_m=interval,
            increment_from=(f"the column header "
                            f"'{_header(grid, drive_cell)[:40]}'"
                            if printed_inc is not None else ""))

    if corrected:
        # An N60 or (N1)60 is the N corrected for the hammer's energy or for
        # overburden (FHWA NHI-06-088 Vol. I, Sec. 3.7.1, p. 3-55, Eq. 3-2;
        # Sec. 3.7.2, p. 3-56, Eq. 3-3), not the N measured in the field
        # that SPT.n holds, and the record has no place of its own for it
        # (RI-48): kept in the note, listed.
        remarks.append("printed as a corrected N, not the field N: "
                       + "; ".join(f"{t} (column '{h[:40]}')"
                                   for t, h in corrected))
        if notes is not None:
            notes.add(page, "n60", "; ".join(t for t, _h in corrected)
                      + f" at {at}")
    # N, where the log prints a standard SPT's three drives and no N.
    n_basis = ""
    drives_alt: Optional[Alternative] = None
    if unread and n is None and not n_printed and (
            kind == "spt"
            or (kind == "other" and any(_word_drive(t) for t in blows))):
        # A drive the floor could not count -- WOH, or a word it cannot
        # read -- leaves the record's N to the reader (RI-04). So does a WOH
        # kept in a record whose sampler the log does not name: it may be an
        # SPT's (RI-44).
        remarks.append("a drive printed as '" + "', '".join(unread)
                       + "' is not a count; no N computed from the drives")
        if notes is not None:
            notes.add(page, "drives", f"'{'; '.join(unread)}' at {at}")
    elif group.doubt and blows and n is None and not n_printed:
        # A stack whose lines cannot all be told apart (RI-44): the drives
        # stay as printed, the N is left to the reader. Listed in QA by
        # _sample_groups.
        remarks.append(group.doubt + "; no N computed from the drives")
    elif n is None and not n_printed and not refusal and blows:
        computed = drives_n()
        if computed is not None:
            n, n_basis = computed
    elif n_printed and blows and not group.doubt:
        # A refusal printed where the N goes ("R", "50+", ">50") beside the
        # drives (RI-48): it stands as printed, and the record is a refusal.
        # Where the drives are three whole 6 in drives of a named SPT, their
        # 2nd + 3rd (p. 3-51) is set against it: a word, or a bound they do
        # not meet ("50+" over 6 + 7), is a disagreement listed with the
        # drives' N beside it.
        computed = drives_n()
        if computed is not None:
            bound = re.fullmatch(r"(>)?\s*(\d+)\s*(\+)?", n_printed)
            met = bound is not None and (
                computed[0] > int(bound.group(2)) if bound.group(1)
                else computed[0] >= int(bound.group(2)))
            if met:
                remarks.append(f"the N printed as '{n_printed}' agrees with "
                               f"the 2nd + 3rd of the printed drives, "
                               f"{computed[0]} ({N_RULE_SOURCE})")
                drives_alt = Alternative(
                    field="n", value=show(computed[0]), method="derived",
                    confidence=max(confidence_floor, min(1.0, confidence)),
                    note=(f"the drives' N, which agrees with the N printed "
                          f"as '{n_printed}': " + computed[1].replace(
                              "not printed; ", "", 1)))
            else:
                remarks.append(f"the N printed as '{n_printed}' differs from "
                               f"the 2nd + 3rd of the printed drives, "
                               f"{computed[0]}; the printed N stands, the "
                               f"drives' N is kept beside it")
                if notes is not None:
                    notes.add(page, "n_drives",
                              f"printed {n_printed}, drives {computed[0]} "
                              f"at {at}")
                drives_alt = Alternative(
                    field="n", value=show(computed[0]), method="derived",
                    confidence=max(confidence_floor, min(1.0, confidence)),
                    note="the drives' N: " + computed[1].replace(
                        "not printed; ", "", 1))
    elif n is not None and n < SPT_HALT_BLOWS and _halted(blows):
        # A printed N under 50 on drives that end in a halt: the record has
        # no N (p. 3-51), so the number is something else -- the next
        # sample's number, a recovery (RI-44). Kept in the note, never as
        # the N.
        remarks.append(f"{n} printed as the N of drives that end in a "
                       f"halt, which have none (the halts printed in "
                       f"{N_RULE_SOURCE}); not read as the N")
        if notes is not None:
            notes.add(page, "refusal_n", f"{n} at {at}")
        n = None
    elif n is not None and blows and not group.doubt:
        # A printed N set against the drives where the printed rule applies
        # (RI-44). The printed N stays in the slot; a different N from the
        # drives stands beside it and the QA section lists both, as a
        # model's N is checked at the merge (RI-04).
        computed = drives_n()
        if computed is not None and computed[0] == n:
            remarks.append(f"the printed N {n} is the 2nd + 3rd of the "
                           f"printed drives ({N_RULE_SOURCE})")
        elif computed is not None and computed[0] is not None:
            remarks.append(f"the printed N {n} differs from the 2nd + 3rd "
                           f"of the printed drives, {computed[0]}; the "
                           f"printed N stands, the drives' N is kept beside "
                           f"it")
            if notes is not None:
                notes.add(page, "n_drives",
                          f"printed {n}, drives {computed[0]} at {at}")
            drives_alt = Alternative(
                field="n", value=show(computed[0]), method="derived",
                confidence=max(confidence_floor, min(1.0, confidence)),
                note="the drives' N: " + computed[1].replace(
                    "not printed; ", "", 1))
    prov = Provenance(
        page=page, bbox=tuple(anchor.bbox), method="grid",
        confidence=max(confidence_floor, min(1.0, confidence)),
        note=("seeded from the grid rows"
              + (f" ({template_note})" if template_note else "") + ": "
              + "; ".join(t for t in texts if t)[:160] + drawn_note
              + printed_note))
    sample = Sample(
        note="; ".join(remarks),
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
    if blows or n is not None or n_printed:
        spt = SPT(depth_top=_q(top, unit), depth_bottom=_q(bottom, unit),
                  blows=blows, n=n, n_basis=n_basis, n_printed=n_printed,
                  refusal=refusal, sample_id=sample_id,
                  prov=prov.model_copy(deep=True))
        if drives_alt is not None:
            spt.prov.alternatives.append(drives_alt)
    if (notes is not None and not marked and spt is None and not sample_id
            and kind == "other" and drawn is None and printed_row is None
            and not any(v is not None for v in values.values())
            and not unassigned):
        # Nothing a sample has -- no id, sampler, drives, N or value, no
        # drawn box: a stray mark in a sample column, not a sample (RI-04).
        notes.add(page, "empty", f"'{'; '.join(t for t in texts if t)[:40]}'"
                                 f" at {at}")
        return None, None
    return sample, spt


def _pct(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return max(0.0, min(100.0, float(value)))


def _metres(length: float, unit: str) -> Optional[float]:
    if unit == "m":
        return length
    if unit == "ft":
        return length / FT_PER_M
    return None


#: Where the rule the floor computes an N by is printed. FHWA NHI-06-088
#: Vol. I, Sec. 3.7, p. 3-51 (procedure: ASTM D 1586): the sampler is driven
#: "three successive 6 in (150 mm) increments"; "the first recorded increment
#: is considered as a 'seating' penetration, while the number of blows to
#: advance the second and third increments are summed to give the N-value".
N_RULE_SOURCE = "FHWA NHI-06-088 Vol. I, Sec. 3.7, p. 3-51 (ASTM D 1586)"
#: The halts printed on the same page, the whole sentence: a record stopped
#: by one short of its three increments has no N. A halt for failing to
#: advance can stop under 50 blows ('10/0"') and is still a halt with no N.
N_HALT_RULE = ("'The SPT can be halted when a total of 100 blows have been "
               "counted or if the number of blows exceeds 50 in any given "
               "6 in (150 mm) increment, or if the sampler fails to advance "
               "during 10 consecutive blows'")
SPT_HALT_BLOWS = 50
SPT_HALT_TOTAL = 100
#: The standard increment, and how far a printed or measured one may sit from
#: it and still be it (6 in is 0.1524 m; a metric log prints 150 mm).
SPT_INCREMENT_M = 0.1524
_INCREMENT_TOL_M = 0.006
#: A sample's drawn or printed interval over its three drives may be off the
#: standard by this much a drive (a box is drawn to a line's width).
_INTERVAL_TOL_M = 0.012
_LENGTH_UNIT = (r"(\"|''|inch(?:es)?\b|in\b\.?|cm\b|mm\b|m\b|ft\b|foot\b"
                r"|feet\b)")
#: "BLOWS/6 IN", "Blows per 6\"", "Blows per 6-inch", "blows/15 cm",
#: "blows/150 mm", "por 15 cm".
_INCREMENT = re.compile(
    r"(?:/|\bper\b|\bpor\b|\bpour\b)\s*(?:(\d+(?:[.,]\d+)?)\s*-?\s*)?"
    + _LENGTH_UNIT, re.I)
#: An increment the header names as one: '(6" INCREMENTS)', "6-in. incr.".
_NAMED_INCREMENT = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*-?\s*" + _LENGTH_UNIT + r"\s*-?\s*incr", re.I)
#: A hammer's weight before the slash, or "drop" after the length: the fall.
_HAMMER_BEFORE = re.compile(r"(?:\blbs?\.?|\bkg|#|\bpounds?)\s*$", re.I)
_DROP_AFTER = re.compile(r"\s*\.?\s*(?:drop|fall)\b", re.I)


def _drive_header(header: str) -> bool:
    """Does a header print the 6 in (150 mm) drive increment, so that its
    bare numbers are drives rather than one N per sample?"""
    inc = header_increment(header)
    return inc is not None and abs(inc - SPT_INCREMENT_M) <= _INCREMENT_TOL_M


def header_increment(header: str) -> Optional[float]:
    """The drive increment a blows column's header prints, in metres, or
    None when it prints none.

    An increment the header states wins (RI-44): one it names as the
    increment ('BLOWS / FOOT (6" INCREMENTS)'), else one printed with its
    length ("Blows per 6 in" after "N-VALUE (blows/ft)"). A bare
    ``BLOWS/FT`` is a foot: the N's own unit, not a drive's, and never the
    standard increment, so no N is computed under it.
    """
    text = str(header or "")
    match = _NAMED_INCREMENT.search(text)
    if match is None:
        # "140 lb / 30 in drop" is the hammer's weight and fall, not a drive.
        found = [m for m in _INCREMENT.finditer(text)
                 if not _HAMMER_BEFORE.search(text[:m.start()])
                 and not _DROP_AFTER.match(text[m.end():])]
        match = next((m for m in found if m.group(1)),
                     found[0] if found else None)
    if match is None:
        return None
    number = float((match.group(1) or "1").replace(",", "."))
    unit = match.group(2).lower().rstrip(".")
    factor = {"\"": 0.0254, "''": 0.0254, "in": 0.0254, "inch": 0.0254,
              "inches": 0.0254, "cm": 0.01, "mm": 0.001, "m": 1.0,
              "ft": 0.3048, "foot": 0.3048, "feet": 0.3048}.get(unit)
    return None if factor is None else number * factor


#: A drive printed as blows over a distance: '50/3"', '10/0"', '25/150 mm'.
_BLOWS_OVER = re.compile(r'(\d+)\s*/\s*([\d.]+)\s*("|\'\'|in|cm|mm)?')
#: The increment written with no unit, in inches, centimetres and
#: millimetres (6 in, 150 mm).
_BARE_INCREMENT = (6.0, 15.0, 150.0)


def _short_of_increment(length: float, unit: Optional[str]) -> bool:
    """Is a drive's printed distance short of the 6 in (150 mm) increment?

    With no unit, 6, 15 and 150 are the increment written per increment
    ('4/6-6/6-7/6'), and every other number under 150 is read as short of
    it. That is a layout judgment, not a printed rule, and it is right only
    in part: under 150 is short in millimetres, but in centimetres only
    under 15 and in inches only under 6. A bare 16-149 read in centimetres
    ('20/45': 45 cm, three increments), or 7-149 read in inches, is a drive
    of a whole increment or more taken as a halt. For the N the error is on
    the safe side: such a record gets no N rather than one it may not have
    (RI-47). The floor's own record of that form is flagged a refusal, as
    before; the merge does not add the flag to a reading on such a halt
    alone (:func:`shows_halt`).
    """
    if unit is None:
        return length < _BARE_INCREMENT[-1] and length not in _BARE_INCREMENT
    metres = length * {"\"": 0.0254, "''": 0.0254, "in": 0.0254,
                       "cm": 0.01, "mm": 0.001}[unit]
    return metres < SPT_INCREMENT_M - _INCREMENT_TOL_M


def _halted(blows: Sequence[Any],
            lengths: Optional[Sequence[Optional[float]]] = None) -> bool:
    """Do these printed drives END in a halt, so that the record has no N?

    N is the 2nd + 3rd of three 6 in (150 mm) increments, and the halts
    :data:`N_HALT_RULE` prints are ones the test CAN make (p. 3-51): a
    record that drove all three whole increments is complete and has its N,
    however hard any of them was ([52, 4, 28], [4, 28, 52]). A record ends
    in a halt only where it stops short: its last drive is printed as blows
    over a distance short of the increment ('50/3"'; '10/0"', which failed
    to advance under 50), or it stops before its third increment at blows
    over a distance, at a drive over 50 blows or at 100 blows in all. Blows
    over a whole increment ('4/6"', '6/6"', '7/6"': a record written per
    increment) are a drive like any count (RI-44, RI-45).

    ``lengths`` are the record's own drive lengths in metres, one per entry
    of ``blows`` (``SPT.increments``). A last one short of the increment is
    the same halt as blows printed over that distance: 16-29-50 over 6, 6
    and 2.5 in ends as 16-29-'50/2.5"' does (RI-50). Lengths that do not line
    up with the blows, or that will not convert, say nothing.
    """
    if (lengths and len(lengths) == len(blows)
            and lengths[-1] is not None
            and lengths[-1] < SPT_INCREMENT_M - _INCREMENT_TOL_M):
        return True
    if not blows:
        return False
    counts: List[int] = []
    for entry in blows:
        if isinstance(entry, bool):
            continue
        if isinstance(entry, int):
            counts.append(entry)
            continue
        match = _BLOWS_OVER.fullmatch(str(entry).strip())
        if match:
            counts.append(int(match.group(1)))
    last = blows[-1]
    over = (None if isinstance(last, (int, bool))
            else _BLOWS_OVER.fullmatch(str(last).strip()))
    if len(blows) < SPT_DRIVES and (
            over is not None or sum(counts) >= SPT_HALT_TOTAL
            or any(c > SPT_HALT_BLOWS for c in counts)):
        return True
    if over is None:
        return False
    try:
        length = float(over.group(2))
    except ValueError:
        return True
    return _short_of_increment(length, over.group(3))


def _lengths_m(record: Any) -> List[Optional[float]]:
    """A driven record's own drive lengths in metres (``SPT.increments``)."""
    return [q.si_value for q in getattr(record, "increments", None) or []]


def ends_in_halt(record: Any) -> bool:
    """Does this driven record end in a halt, by its printed drives or by its
    own drive lengths (:func:`_halted`)? (RI-50)"""
    return _halted(record.blows, _lengths_m(record))


def _first_three(record: Any) -> Tuple[List[Any], List[Optional[float]]]:
    """The drives the N is taken from, the first three, and their lengths
    where the record's own lengths line up with its drives."""
    blows = list(record.blows)
    lengths = _lengths_m(record)
    if len(lengths) != len(blows):
        lengths = []
    return blows[:SPT_DRIVES], lengths[:SPT_DRIVES]


def halts_before_n(record: Any) -> bool:
    """Does this record halt within the three increments its N is taken
    from, so that it has no N under 50 (RI-44, RI-50)? A drive after the
    third that ends short does not: Fig. 3-20 (FHWA NHI-06-088 p. 3-52)
    prints an "Occasional Fourth Increment Used to provide additional soil
    material", and the N is still the 2nd + 3rd (RI-50 review)."""
    return _halted(*_first_three(record))


def _halt_how(record: Any) -> str:
    """How a record halts within its first three increments, for a QA
    line."""
    blows, lengths = _first_three(record)
    if not _halted(blows) and _halted(blows, lengths):
        given = ", ".join(f"{q.value:g} {q.unit}"
                          for q in record.increments[:SPT_DRIVES])
        return (f"its own drive lengths ({given}) end short of the 6 in "
                f"(150 mm) increment")
    if len(blows) < SPT_DRIVES:
        return "it stops before the third increment"
    which = "last" if len(record.blows) == SPT_DRIVES else "third"
    return (f"its {which} drive, {show(blows[-1])}, is short of its "
            f"increment")


def _a_lone_count(record: Any) -> bool:
    """One whole number for drives, no distance and no lengths: as likely
    the N filed as a drive as a drive of its own (RI-50 review)."""
    if len(record.blows) != 1 or _lengths_m(record):
        return False
    entry = record.blows[0]
    if isinstance(entry, bool):
        return False
    return isinstance(entry, int) or bool(re.fullmatch(r"\s*\d+\s*",
                                                      str(entry)))


def shows_halt(record: Any) -> bool:
    """Does this record end in a halt that it shows plainly: by its own drive
    lengths, by stopping before the third increment, or by a last drive
    printed over a distance with its unit ('50/2"')? Not plain, and left as
    read by the refusal flag and the reconciler's drive check: a halt that
    rests only on a bare number read as short ('7/13':
    :func:`_short_of_increment`'s layout judgment, wrong in centimetres from
    16 and in inches from 7; the notation may be another one), and a lone
    whole number over 50 with no distance or lengths (:func:`_a_lone_count`:
    it may be the N filed as a drive, which the drive check then catches)."""
    if not ends_in_halt(record) or _a_lone_count(record):
        return False
    if not _halted(record.blows) or len(record.blows) < SPT_DRIVES:
        return True
    over = _BLOWS_OVER.fullmatch(str(record.blows[-1]).strip())
    return over is not None and over.group(3) is not None


def _whole_lengths(record: Any) -> bool:
    """No drive lengths of its own, or one whole 6 in (150 mm) increment for
    each drive: lengths that are not rule out an N computed at the 6 in
    increment the floor read off the header (RI-45, RI-50)."""
    lengths = _lengths_m(record)
    if not lengths:
        return True
    return len(lengths) == len(record.blows) and all(
        v is not None and abs(v - SPT_INCREMENT_M) <= _INCREMENT_TOL_M
        for v in lengths)


def n_from_drives(blows: Sequence[Any], *, sampler: str,
                  increment_m: Optional[float] = None,
                  interval_m: Optional[float] = None,
                  increment_from: str = ""
                  ) -> Optional[Tuple[int, str]]:
    """``(N, how it was had)`` from a standard SPT's three printed drives, or
    None when the rule does not apply.

    The rule is printed in :data:`N_RULE_SOURCE`: N is the second drive plus
    the third. It applies only to an SPT sampler the log names (``sampler``
    is ``"spt"``), to exactly three whole-number drives (a refusal, a
    weight-of-hammer entry or a fourth drive is not this case), and to drives
    of 6 in or 150 mm: the increment a header prints (``increment_m``) must
    be that, and with none printed the sample's interval (``interval_m``)
    must be three of them. A drive over 50 blows does not stop it: the halts
    the same page prints (:data:`N_HALT_RULE`) are ones the test CAN make,
    and three whole drives show it did not (RI-45).
    """
    if sampler != "spt":
        return None
    counts: List[int] = []
    for entry in blows:
        if isinstance(entry, bool):
            return None
        if isinstance(entry, int):
            counts.append(entry)
        elif isinstance(entry, str) and entry.strip().isdigit():
            counts.append(int(entry.strip()))
        else:
            return None
    if len(counts) != 3:
        return None
    if increment_m is not None:
        if abs(increment_m - SPT_INCREMENT_M) > _INCREMENT_TOL_M:
            return None
        how = increment_from or "the printed increment"
    elif interval_m is not None and abs(
            interval_m / 3.0 - SPT_INCREMENT_M) <= _INTERVAL_TOL_M:
        how = f"the sample's interval ({interval_m:.3g} m over three drives)"
    else:
        return None
    n = counts[1] + counts[2]
    return n, (f"not printed; the 2nd + 3rd of three 6 in (150 mm) drives, "
               f"{counts[1]} + {counts[2]} = {n}, by the rule printed in "
               f"{N_RULE_SOURCE}; 6 in drives from {how}")


def _doubts(values: Dict[str, Any], units: Dict[str, str]) -> List[str]:
    """What in one sample's values contradicts its own column or the other
    values (RI-04). A percentage outside 0-100 cannot stand in the record's
    slot; it is taken out of ``values`` and the note says what was printed.
    Everything else stays where it is, doubted."""
    out: List[str] = []
    for key in ("recovery_percent", "rqd_percent", "fines_percent"):
        value = values.get(key)
        if value is not None and not 0.0 <= float(value) <= 100.0:
            out.append(f"{key.replace('_', ' ')} printed {value:g}, outside "
                       f"0-100")
            values.pop(key)
    for key in ("water_content", "liquid_limit", "plastic_limit",
                "plasticity_index", "dry_unit_weight", "qu", "pocket_pen",
                "recovery"):
        value = values.get(key)
        if value is not None and float(value) < 0.0:
            out.append(f"{key.replace('_', ' ')} {value:g} is negative")
    ll, pl = values.get("liquid_limit"), values.get("plastic_limit")
    pi = values.get("plasticity_index")
    if ll is not None and pl is not None and float(pl) > float(ll):
        out.append(f"plastic limit {pl:g} above liquid limit {ll:g}")
    if None not in (ll, pl, pi) and abs(float(ll) - float(pl)
                                        - float(pi)) > 1.0:
        out.append(f"LL - PL = {float(ll) - float(pl):g} but PI printed "
                   f"{pi:g}")
    return out


class _FloorNotes:
    """What the floor could not settle, for the QA section (RI-04).

    One row per page and reason, naming every depth, so a log with twenty
    such values is one entry to check rather than twenty.
    """

    WHY: Dict[str, str] = {
        "unassigned": ("numbers printed in a column that holds several "
                       "quantities ({header}); which one each is cannot be "
                       "told from its position, so each is kept in its "
                       "sample's note and left to the reader"),
        "unbound": ("values printed off every sample's lines that the "
                    "page's geometry does not bind to one sample (none "
                    "near enough, two about equally near, or on the other "
                    "side of their sample from the rest of their column); "
                    "each stays where it is printed"),
        "doubtful": ("values that contradict their column or the sample's "
                     "other values, kept at a lower confidence"),
        "order": ("samples whose depth contradicts their neighbours' "
                  "(their numbers run out of order down the hole), kept at "
                  "a lower confidence"),
        "drives": ("drive records of an SPT, or of a sampler the log does "
                   "not name, with an entry that is not a count (WOH, WOR, "
                   "or a word the floor cannot read); no N is computed from "
                   "them"),
        "empty": ("rows in the sample columns with nothing a sample has "
                  "(no id, sampler, drives, N or value, no drawn box); not "
                  "seeded as samples"),
        "stack": ("runs of lines in a column {header} that do not read, "
                  "line by line, as a sample number, at most three drives "
                  "(FHWA NHI-06-088 Vol. I, Sec. 3.7, p. 3-51) and a "
                  "recovery, with each sample's number the one before or "
                  "the next; read as before, with no sample number taken "
                  "from them and no N computed"),
        "stack_tail": ("samples in a column {header} whose last count may "
                       "be a drive or the N or recovery the header prints "
                       "after the blows (the lines are kept as drives and "
                       "no N is computed from them), or with a line that "
                       "reads as blows where the header prints the N or "
                       "the recovery (kept in the sample's note, as "
                       "neither)"),
        "refusal_n": ("numbers under 50 printed as the N of drives that "
                      "end in a halt (stopped before the third increment "
                      "at 100 blows, more than 50 in an increment or blows "
                      "over a distance, or a last drive short of its "
                      "increment); such a record has no N (FHWA NHI-06-088 "
                      "Vol. I, Sec. 3.7, p. 3-51), so each is kept in its "
                      "sample's note, not as the N"),
        "n_drives": ("printed Ns that differ from the 2nd + 3rd of the "
                     "printed drives (FHWA NHI-06-088 Vol. I, Sec. 3.7, "
                     "p. 3-51); the printed N stands and the drives' N is "
                     "kept beside it"),
        "n60": ("numbers printed as a corrected N (N60, the N adjusted to "
                "60 % hammer efficiency, FHWA NHI-06-088 Vol. I, Sec. "
                "3.7.1, p. 3-55, Eq. 3-2; or (N1)60, Sec. 3.7.2, p. 3-56, "
                "Eq. 3-3), not the N measured in the field; the record has "
                "no place for them, so each is kept in its sample's note, "
                "not as the N"),
        "unplaced_n": ("Ns or refusals printed in a column {header} where "
                       "the floor cannot tell which sample they belong to "
                       "(a stack of lines that does not read as a sample "
                       "number and its drives) or what they are (a column "
                       "that also names a recovery or values); each is "
                       "kept in the nearest sample's note, not as its N"),
    }

    def __init__(self) -> None:
        self.items: Dict[Tuple[int, str, str], List[str]] = {}

    def add(self, page: int, reason: str, item: str,
            header: str = "") -> None:
        self.items.setdefault((int(page), reason, header[:60]),
                              []).append(item)

    def qa_rows(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for (page, reason, header), items in sorted(self.items.items()):
            why = self.WHY[reason].format(
                header=f"headed '{header}'" if header else "")
            out.append({"what": f"log floor, page {page}", "page": page,
                        "why": f"The grid's reading of page {page} has "
                               f"{len(items)} {why}.",
                        "value": "; ".join(items)[:400],
                        "flag": f"floor_{reason}"})
        return out


# ---------------------------------------------------------------------------
# a page with no ruler: the depths it prints (RI-18)
# ---------------------------------------------------------------------------
#
# Where ``log_grid`` fitted no ruler, no cell has a depth and nothing above
# seeds anything. The page may still PRINT its depths: in a depth column, as a
# sample's interval, at the head of a description
# (:mod:`report_ingest.printed_depths`). A cell standing on the same line as
# one of those is given that depth, and only that depth; a cell standing on
# no printed depth's line is left without one.
#
# A line that prints a depth is read by what stands beside it:
# - sample or test values: a SAMPLE line, and its depth is the sample's;
# - a description: the description above it stops there. The words on the
#   line may run on from above or start a new stratum, and their position
#   cannot say which, so the stratum above keeps only the words it surely
#   has. A layer opens on the line only where it prints an INTERVAL and no
#   sample stands on it (beside a sample, the interval may be the
#   sample's), or where the interval heads the description itself. One
#   depth beside a description does not say which end of a stratum it is.

class _AtPrintedDepth:
    """A grid cell on a page with no ruler, given the depth printed on its
    line. Everything else is the cell's own."""

    __slots__ = ("_cell", "depth")

    def __init__(self, cell: Any, depth: float) -> None:
        self._cell = cell
        self.depth = depth

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cell, name)


@dataclass
class _PrintedLine:
    """One line of a page with no ruler that prints a depth, and the
    sample or test values and the description lines standing on it."""

    row: PrintedRow
    values: List[Any]
    words: List[Any]


def _printed_lines(grid: Any, printed: PrintedDepths, page: int,
                   by_column: Optional[Dict[str, Tuple[str, ...]]] = None,
                   unit: Optional[str] = None, placing: bool = True
                   ) -> List[_PrintedLine]:
    values = [c for c in grid.rows if c.page == page
              and set(_names(grid, c, by_column)) & _SAMPLE_FAMILY
              and (c.numbers or str(c.text or "").strip())]
    words = sorted((c for c in grid.rows if c.page == page
                    and "description" in _names(grid, c, by_column)
                    and getattr(c, "column", "") != "depth"
                    and str(c.text or "").strip()),
                   key=lambda c: (c.bbox[1], c.bbox[0]))
    return [_PrintedLine(row,
                         [c for c in values if on_line(c.bbox, row.first)],
                         [c for c in words if on_line(c.bbox, row.first)])
            for row in printed.rows(page, unit, placing)]


def _printed_rows(grid: Any, printed: PrintedDepths, unit: str,
                  by_column: Optional[Dict[str, Tuple[str, ...]]] = None
                  ) -> List[Tuple[PrintedRow, List[Any], Optional[float]]]:
    """Each sample line's values, given the line's depth, and the base the
    line prints for the sample (or None).

    A line places its values when a depth column or a sample's own interval
    prints its depth; a range at the head of a description opens a stratum
    and says nothing of a sample beside it.
    """
    out: List[Tuple[PrintedRow, List[Any], Optional[float]]] = []
    for page in printed.pages:
        taken: set = set()
        for line in _printed_lines(grid, printed, page, by_column, unit):
            if not line.row.places_values:
                continue
            cells = [c for c in line.values if id(c) not in taken]
            if not cells:
                continue
            taken.update(id(c) for c in cells)
            out.append((line.row,
                        [_AtPrintedDepth(c, line.row.top) for c in cells],
                        line.row.sample_bottom(bool(line.words))))
    return out


def _printed_layers(grid: Any, printed: PrintedDepths, unit: str,
                    confidence_floor: float,
                    by_column: Optional[Dict[str, Tuple[str, ...]]] = None
                    ) -> List[Layer]:
    """The strata whose two depths a page with no ruler prints.

    A stratum's words run from its line down to the next line that prints a
    depth beside a description (see the rules above). Only intervals that
    CHAIN, each top at another's base or base at another's top, open
    layers: an interval standing alone may be a thickness or a lens ("2 to 4
    ft thick sand lenses"), not a stratum, and a seam printed between two
    strata does not break their chain. A stratum printed with one depth, or whose
    words share a line with a sample, is left to the reader, whose top and
    base are accepted where the page prints them.
    """
    out: List[Layer] = []
    for page in printed.pages:
        desc = sorted((c for c in grid.rows if c.page == page
                       and "description" in _names(grid, c, by_column)
                       and getattr(c, "column", "") != "depth"),
                      key=lambda c: (c.bbox[1], c.bbox[0]))
        if not desc:
            continue
        rests = {tuple(r.bbox): r.rest for r in printed.on_page(page)
                 if r.source == "description"}
        # Every line printing a depth beside a description cuts the words
        # above it, whether or not its depth can place anything.
        cuts: List[Tuple[Optional[PrintedDepth], int]] = []
        for line in _printed_lines(grid, printed, page, by_column,
                                   placing=False):
            if not line.words:
                continue
            at = desc.index(line.words[0])
            if cuts and at <= cuts[-1][1]:
                continue
            reading = line.row.stratum()
            opens = (reading is not None and reading.bottom is not None
                     and reading.seeds and printed.unit_on(reading) == unit
                     and (reading.source == "description" or not line.values))
            cuts.append((reading if opens else None, at))
        intervals = [k for k, (reading, _at) in enumerate(cuts)
                     if reading is not None]
        links = chained([cuts[k][0] for k in intervals])
        for k, link in zip(intervals, links):
            if not link:
                cuts[k] = (None, cuts[k][1])
        for k, (reading, start) in enumerate(cuts):
            if reading is None:
                continue
            stop = cuts[k + 1][1] if k + 1 < len(cuts) else len(desc)
            words = [rests.get(tuple(c.bbox), str(c.text or ""))
                     for c in desc[start:stop]]
            description = " ".join(" ".join(w.split()) for w in words
                                   if w.strip())
            if not description:
                continue
            symbol = _USCS.search(description)
            out.append(Layer(
                top=_q(reading.top, unit), bottom=_q(reading.bottom, unit),
                description=description,
                uscs=symbol.group(1) if symbol else "",
                prov=Provenance(
                    page=page, bbox=tuple(desc[start].bbox), method="grid",
                    confidence=max(confidence_floor, reading.confidence),
                    note=(f"layer bound by its printed depths "
                          f"('{reading.text}', {reading.how}); no depth "
                          f"ruler on page {page}"))))
    return out


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


def _sample_groups(grid: Any, pages: Sequence[int], span: float,
                   by_column: Dict[str, Tuple[str, ...]],
                   notes: _FloorNotes) -> List[_Group]:
    """The sample columns' cells gathered into one group per sample.

    First the samples: the cells that mark one (its id, sampler, drives, N,
    recovery) within ``span`` of a sample's first such row are that
    sample's; a second id or a second blow record on the same sheet opens a
    second sample however close it is printed. Then the values -- a moisture
    content, a labelled result under TESTS -- each go to the sample the
    page's geometry points at (RI-04): the one whose lines are nearest, on
    any sheet, when it is clearly nearer than the next (the next half as far
    again) and, for a value printed off the sample's lines, within two spans
    and on the side of its sample the rest of its column's values are. A
    tie never goes to either. A value that fails any of those stays where it
    is printed, as its own row at the floor's lowest confidence, and the QA
    section lists it; a labelled result from outside the sample columns is
    not a sample on its own and is only listed. A log with no sample marked
    anywhere has only its values, read as they always were.
    """
    family: List[Any] = []
    results: List[Any] = []
    for cell in grid.rows:
        if cell.depth is None or cell.page not in pages:
            continue
        if not (getattr(cell, "numbers", None) or _text(cell)):
            continue
        here = set(_names(grid, cell, by_column))
        if here & _SAMPLE_FAMILY:
            family.append(cell)
        elif here & _RESULT_COLUMNS:
            results.append(cell)
    # Words the grid placed in a sample column -- a description run long, a
    # remark, an end-of-boring line, a drawn rule -- are not sample data:
    # they neither open a sample nor name one. They are the layer's or the
    # remark's, read there, so nothing is lost by leaving them out here.
    cells: List[Any] = [c for c in _join_labels(family)
                        if not _is_words(_text(c))]
    labelled = [c for c in _join_labels(results)
                if _labelled_value(_text(c)) is not None]
    from_results = {id(c) for c in labelled}
    cells.extend(labelled)
    cells.sort(key=lambda c: (c.page, float(c.depth), c.bbox[0]))
    styles = _id_styles(grid, cells, by_column)
    # A column headed as both the sample numbers and the blows, read stack
    # by stack (RI-44): each stack that reads is one sample, whatever depth
    # its lines run over.
    stacks, unread = _read_stacks(grid, cells, styles, by_column)
    groups: List[_Group] = []
    for stack in stacks:
        group = _Group(stack.lines[0])
        for cell in stack.lines[1:] + [c for c in (stack.anchor,)
                                       if c is not None]:
            group.add(cell)
        group.heads = {id(stack.lines[0])}
        group.recovery = {id(c) for c in (stack.recovery,) if c is not None}
        group.printed_n = {id(c) for c in (stack.printed_n,)
                           if c is not None}
        group.unassigned = {id(c) for c in stack.unassigned}
        group.n60 = {id(c) for c in (stack.n60,) if c is not None}
        group.sampler = {id(c) for c in (stack.sampler,) if c is not None}
        if stack.doubt:
            group.doubt = ("the last count may be the N or the recovery the "
                           "header prints after the blows")
        if stack.doubt or stack.blows_in_tail:
            notes.add(group.page, "stack_tail",
                      f"sample {_text(stack.lines[0])} at "
                      f"{float(group.first):g}",
                      header=_header(grid, stack.lines[0]))
        groups.append(group)
    placed = {id(c) for g in groups for c in g.cells}
    drives = {id(c) for s in stacks for c in s.lines[1:]}
    in_unread = {id(c) for run in unread for c in run}
    for run in unread:
        notes.add(run[0].page, "stack",
                  f"'{'; '.join(_text(c) for c in run)[:60]}' at "
                  f"{float(run[0].depth):g}", header=_header(grid, run[0]))
    keys: Dict[int, Tuple[str, str]] = {}
    marks: set = set()
    for cell in cells:
        names = _names(grid, cell, by_column)
        key = _sample_key(_text(cell), names, styles.get(cell.column_id, ""))
        if key is not None and not (key[0] == "id" and id(cell) in drives):
            keys[id(cell)] = key
        if id(cell) not in from_results and _marks_a_sample(_text(cell),
                                                            names):
            marks.add(id(cell))

    for cell in cells:
        if id(cell) not in marks or id(cell) in placed:
            continue
        # Any sample at this depth takes the cell, whichever sheet it is
        # on: the rows are sorted by page first, so a second sheet's
        # restatement of a depth would otherwise open a second sample.
        fits = [g for g in groups if g.takes(cell, span, keys)]
        if fits:
            min(fits, key=lambda g: abs(float(cell.depth)
                                        - float(g.first))).add(cell)
        else:
            groups.append(_Group(cell))

    def gap(group: _Group, at: float) -> Tuple[float, int]:
        """How far ``at`` is from the lines the sample is printed on, and
        on which side: -1 above them, 1 below, 0 among them."""
        lines = [float(c.depth) for c in group.cells]
        if at < min(lines):
            return min(lines) - at, -1
        if at > max(lines):
            return at - max(lines), 1
        return 0.0, 0

    values = [c for c in cells if id(c) not in marks and id(c) not in placed]
    left: List[Any] = []
    off_lines: List[Tuple[Any, _Group, int]] = []
    for cell in values:
        if not groups:
            left.append(cell)
            continue
        at = float(cell.depth)
        found = sorted(((*gap(g, at), k) for k, g in enumerate(groups)),
                       key=lambda row: row[0])
        nearest, side, home = found[0][0], found[0][1], groups[found[0][2]]
        clear = len(found) == 1 or found[1][0] > 1.5 * nearest + 1e-9
        if not clear or nearest > 2.0 * span:
            left.append(cell)
        elif nearest <= span:
            home.add(cell)                       # on the sample's own rows
        else:
            off_lines.append((cell, home, side))
    # The side of its sample a column prints its values on, by majority.
    sides: Dict[Tuple[int, str], Dict[int, int]] = {}
    for cell, _home, side in off_lines:
        tally = sides.setdefault((cell.page, cell.column_id), {})
        tally[side] = tally.get(side, 0) + 1
    for cell, home, side in off_lines:
        tally = sides[(cell.page, cell.column_id)]
        if tally.get(-side, 0) > tally.get(side, 0):
            left.append(cell)
        elif not any(c is cell for c in home.bound):
            home.bound.append(cell)

    # What is left stays where it is printed. On a log with samples it is
    # doubted and listed; on a log with none it is all the samples there
    # are, read as before.
    loose: List[_Group] = []
    for cell in sorted(left, key=lambda c: (c.page, float(c.depth),
                                             c.bbox[0])):
        fits = [g for g in loose if g.takes(cell, span)]
        if fits:
            min(fits, key=lambda g: abs(float(cell.depth)
                                        - float(g.first))).add(cell)
        else:
            loose.append(_Group(cell))
    kept: List[_Group] = list(groups)
    for group in loose:
        results_only = all(id(c) in from_results for c in group.cells)
        if not groups and not results_only:
            kept.append(group)
            continue
        if not any(re.search(r"\d", _text(c)) for c in group.cells):
            continue                  # a tick box, a dash: no value at all
        what = "; ".join(_text(c) for c in group.cells)[:60]
        notes.add(group.page, "unbound",
                  f"'{what}' at {float(group.first):g}")
        if results_only:
            continue
        group.unbound = True
        kept.append(group)
    for group in kept:
        if any(id(c) in in_unread for c in group.cells + group.bound):
            # Read as before RI-44, and no N from it: which of its numbers
            # are drives is not known.
            group.doubt = ("printed in a stack that does not read as a "
                           "sample number and its drives")
            group.unread_stack = True
    kept.sort(key=lambda g: (g.page, float(g.first)))
    return kept


def _check_order(samples: List[Sample], spt: List[SPT], unit: str,
                 confidence_floor: float, notes: _FloorNotes) -> None:
    """A sample whose number runs out of order down the hole -- S-7 between
    S-2 and S-4 -- sits at a depth its neighbours contradict: the grid put
    it on the wrong row, or the log misprinted it. It stays, doubted."""
    numbered: Dict[str, List[Tuple[float, float, Sample]]] = {}
    for sample in samples:
        match = re.fullmatch(r"([A-Za-z]*)[-\s#.]?(\d{1,4})[A-Za-z]?",
                             sample.sample_id.strip())
        if match is None or sample.top is None:
            continue
        numbered.setdefault(match.group(1).upper(), []).append(
            (float(sample.top.value), float(match.group(2)), sample))
    for rows in numbered.values():
        rows.sort(key=lambda row: row[0])
        for k in range(1, len(rows) - 1):
            before, here, after = rows[k - 1][1], rows[k][1], rows[k + 1][1]
            if before < after and not before <= here <= after:
                sample = rows[k][2]
                why = (f"sample {sample.sample_id} at {rows[k][0]:g} {unit} "
                       f"lies between {rows[k - 1][2].sample_id} and "
                       f"{rows[k + 1][2].sample_id}").replace("  ", " ")
                if sample.prov is not None:
                    sample.prov.confidence = min(sample.prov.confidence,
                                                 confidence_floor)
                sample.note = "; ".join(
                    t for t in (sample.note, "doubtful: " + why) if t)
                for drive in spt:
                    if drive.sample_id == sample.sample_id and drive.prov:
                        drive.prov.confidence = min(drive.prov.confidence,
                                                    confidence_floor)
                notes.add(sample.prov.page if sample.prov else 0, "order",
                          why)


def seed_from_grid(grid: Any, pages: Sequence[int], report_id: str = "",
                   confidence_floor: float = 0.3,
                   template: Any = None,
                   lines: Sequence[Any] = (),
                   printed: Optional[PrintedDepths] = None,
                   qa: Optional[List[Dict[str, Any]]] = None
                   ) -> Investigation:
    """The grid's own reading of one log as an :class:`Investigation`.

    Every value carries a ``grid`` provenance at the grid's own confidence
    for that cell. Nothing here is inferred: a sampler is ``other`` unless
    the log prints its code (drives alone do not make an SPT), an N is
    recorded where the log prints one -- or, for an SPT the log names with
    its three 6 in (150 mm) drives printed, computed by the rule
    :data:`N_RULE_SOURCE` prints and said so in ``n_basis`` -- a USCS symbol
    only where the description prints one in the standard letters, and a
    water level only where the header's groundwater field prints a depth.

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

    ``printed`` is :func:`report_ingest.printed_depths.printed_depths` over
    these pages, when the caller has already run it; left ``None`` it is run
    here. On a page whose ruler could not be fitted it is the only source of
    a depth: a sample or test printed on the same line as a printed depth is
    seeded at that depth, and a stratum whose interval is printed is seeded
    as a layer, each saying in its note how the depth was read. On a log
    whose every page has a ruler it finds nothing and changes nothing.

    ``qa``, when given, receives one unresolved row per page and reason for
    what the floor read but could not settle (RI-04): values bound to no
    sample, numbers in a column of several quantities, values that
    contradict their column or their neighbours.
    The rows have the shape the log reader's ``unresolved`` list carries.
    """
    pages = [int(p) for p in pages]
    unit = (grid.unit or "").strip()
    if printed is None:
        printed = printed_depths(grid, pages)
    # On a log with no ruler anywhere, a header depth is the number its
    # unit is bound to (RI-18), never the first number on the line, which
    # may be a date, a time or a year; and the unit is the one printed with
    # the depths or over the depth column. None on every other log, whose
    # header reads as it always did.
    header_depths: Optional[Dict[str, float]] = None
    if printed and not unit and not grid.rulers:
        unit = printed.unit
        header_depths = {r.column_id: r.top for r in printed.readings
                         if r.source == "header"
                         and printed.unit_on(r) == unit}
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
    if header_depths is not None:
        if header_depths.get("total_depth") is not None and unit:
            total = Quantity(value=header_depths["total_depth"], unit=unit,
                             prov=fprov("total_depth"))
    elif raw and _first_number(raw) is not None \
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
        number = (_first_number(raw) if header_depths is None
                  else header_depths.get("groundwater"))
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
        note = f"layer bound by the grid ({ly.source})"
        evidence = getattr(ly, "evidence", None) or {}
        if evidence.get("found_in") == "pixels":
            # Only with visual scales on (report_ingest.visual_scales): a
            # top MEASURED from the stratum line the scan draws, with its
            # +/- -- a geometry voter in the floor.
            pm = getattr(ly, "plus_minus", None)
            note = ("layer top measured from the stratum line drawn on the "
                    "scan (pixels)"
                    + (f", +/- {float(pm):.3g} {unit}".rstrip() if pm
                       else "")
                    + (", drawn dashed" if evidence.get("dashed") else ""))
        layers.append(Layer(
            top=_q(ly.top, unit), bottom=_q(ly.bottom, unit),
            description=str(ly.description or ""),
            uscs=symbol.group(1) if symbol else "",
            prov=Provenance(page=page,
                            bbox=tuple(ly.bbox) if ly.bbox else None,
                            method="grid",
                            confidence=max(confidence_floor,
                                           min(1.0, float(ly.confidence))),
                            note=note)))
    if printed and unit:
        for layer in _printed_layers(grid, printed, unit, confidence_floor,
                                     by_column):
            key = (round(float(layer.top.value), 2), fold(layer.description))
            if key not in seen_layers:
                seen_layers.add(key)
                layers.append(layer)

    samples: List[Sample] = []
    spt: List[SPT] = []
    notes = _FloorNotes()
    if unit or grid.rulers:
        span = _span_in_unit(unit, grid)
        groups = _sample_groups(grid, pages, span, by_column, notes)
        boxes = list(getattr(grid, "samples", None) or [])
        for group in groups:
            drawn = _drawn_box(group, boxes)
            if drawn is not None:
                boxes.remove(drawn)
            sample, drive = _seed_sample(group, grid, unit, confidence_floor,
                                         by_column, template_note, drawn,
                                         notes=notes)
            if sample is not None:
                samples.append(sample)
            if drive is not None:
                spt.append(drive)
    if printed and unit:
        span = _span_in_unit(unit, grid)
        ruled = [float(s.top.value) for s in samples if s.top is not None]
        for row, cells, base in _printed_rows(grid, printed, unit,
                                              by_column):
            if any(abs(row.top - top) <= span for top in ruled):
                # A sheet with no ruler restating a sample a ruled sheet has
                # already placed: the same sample read twice (see _Group).
                continue
            group = _Group(cells[0])
            for cell in cells[1:]:
                group.add(cell)
            sample, drive = _seed_sample(group, grid, unit, confidence_floor,
                                         by_column, template_note,
                                         printed_row=row, printed_bottom=base,
                                         notes=notes)
            if sample is not None:
                samples.append(sample)
            if drive is not None:
                spt.append(drive)
    _check_order(samples, spt, unit, confidence_floor, notes)
    if qa is not None:
        qa.extend(notes.qa_rows())

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
                   + (f"; {s.note}" if s.note else "")
                   + (f"  [confidence {s.prov.confidence:.2f}]" if s.prov
                      else ""))
    for r in inv.spt:
        out.append(f"driven record at {show(r.depth_top)} "
                   f"({_where(r.prov)}): blows {show(r.blows) or '?'}"
                   + (f", N {r.n} ({r.n_basis})" if r.n_basis
                      else f", N {r.n}" if r.n is not None
                      else ", N not printed")
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


def _same_drives(a: Sequence[Any], b: Sequence[Any]) -> Optional[bool]:
    """Per-drive lengths agree: same count, each within 4 mm."""
    if not a or not b:
        return None
    if len(a) != len(b):
        return False
    return all(same_depth(x, y, 0.004) is not False for x, y in zip(a, b))


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
            elif kind == "drives":
                same = _same_drives(fv, mv)
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
            if kind in ("blows", "drives") and (not fv and not mv):
                setattr(merged, name, [])
                continue
            if kind in ("blows", "drives"):
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
            if (kind in ("text", "exact_text", "blows", "drives")
                    and value is None):
                value = "" if kind in ("text", "exact_text") else []
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
                # The floor's drives' N beside a printed one (RI-44) stays
                # beside the merged N while those drives stand and it is not
                # the merged N itself; from drives that lost, it is no N of
                # this record, and nor is it where the record's own drive
                # lengths are not three whole 6 in increments (RI-50).
                for alt in floor_prov.alternatives:
                    if (alt.field == "n"
                            and getattr(merged, "blows", None)
                            == getattr(f, "blows", None)
                            and _whole_lengths(merged)
                            and show(getattr(merged, "n", None)) != alt.value
                            and alt not in prov.alternatives):
                        prov.alternatives.append(alt.model_copy(deep=True))
                prov.note = (f"{model_prov.note}; " if model_prov.note
                             else "") + "matched to the grid's row" \
                    + (f" ({floor_prov.note[:60]})" if floor_prov.note else "")
        merged.prov = prov
        return merged

    @staticmethod
    def _with_floor_note(f: Sample, merged: Sample) -> Sample:
        """The floor's own note on a sample -- a doubt, a number it could
        not assign -- stays on the merged sample beside the model's."""
        if f.note and f.note not in (merged.note or ""):
            merged.note = "; ".join(t for t in (f.note, merged.note) if t)
        return merged

    # -- N, against the drives (RI-04) -----------------------------------
    def sampler_at(self, record: SPT) -> str:
        """The sampler of the merged sample this drive record belongs to:
        the one with its id, else the one at its depth."""
        merged = getattr(self, "merged_samples", None) or []
        if record.sample_id:
            for sample in merged:
                if sample.sample_id and fold(sample.sample_id) == fold(
                        record.sample_id):
                    return sample.kind
        for sample in merged:
            if same_depth(sample.top, record.depth_top):
                return sample.kind
        return "other"

    def derived_n(self, record: SPT) -> Optional[Tuple[int, str]]:
        """N computed from the record's printed drives by the printed rule
        (:func:`n_from_drives`), or None where the rule does not apply."""
        increment: Optional[float] = None
        how = ""
        if record.increments:
            lengths = [q.si_value for q in record.increments]
            if len(lengths) != len(record.blows) or any(
                    v is None or abs(v - SPT_INCREMENT_M) > _INCREMENT_TOL_M
                    for v in lengths):
                return None
            increment, how = SPT_INCREMENT_M, "the record's drive lengths"
        elif record.increment is not None:
            increment = record.increment.si_value
            if increment is None:
                return None
            how = "the record's printed increment"
        interval = None
        if record.depth_bottom is not None:
            top = record.depth_top.si_value
            bottom = record.depth_bottom.si_value
            if top is not None and bottom is not None:
                interval = bottom - top
        return n_from_drives(record.blows, sampler=self.sampler_at(record),
                             increment_m=increment, interval_m=interval,
                             increment_from=how)

    def check_n(self, merged: SPT, f: Optional[SPT], m: SPT, where: str,
                what: str) -> None:
        """How the merged N was had, and a model's N checked against the
        drives (RI-04).

        An N the floor computed keeps its basis while it is the N that
        stands; where the model's printed N replaced it, the computed one is
        the alternative and says how it was had. An N only the model gives
        is set against the second plus the third drive where the printed
        rule applies, and a difference is a disagreement under the floor's
        one rule: the model's N stands only with a box and a note, the
        other is kept beside it, and the QA section lists both.

        An N under 50 on a record that halts within its first three
        increments is never in the slot, whoever read it (RI-44), whether
        its drives show the halt or its own drive lengths do (RI-50): such a
        record has no N (p. 3-51; :func:`halts_before_n`).
        """
        if (merged.n is not None and merged.n < SPT_HALT_BLOWS
                and halts_before_n(merged)):
            self.halt_n(merged, f, m, where, what)
            return
        if f is not None and f.n_basis:
            if list(merged.blows) != list(f.blows) or merged.increments:
                self.recompute_n(merged, f, m, where, what)
                return
            if merged.n == f.n:
                merged.n_basis = f.n_basis
                return
            merged.n_basis = ""
            for alt in (merged.prov.alternatives if merged.prov else []):
                if alt.field == "n" and alt.value == show(f.n):
                    alt.method = "derived"  # type: ignore[assignment]
                    alt.note = f"the floor's N, {f.n_basis}"
            return
        if m.n is None or (f is not None and f.n is not None):
            return
        derived = self.drives_n(merged, f)
        if derived is None or derived[0] == m.n:
            return
        self.check_model_n(merged, f, m, where, what, derived)

    def check_model_n(self, merged: SPT, f: Optional[SPT], m: SPT,
                      where: str, what: str,
                      derived: Tuple[int, str]) -> None:
        """The model's printed N set against the 2nd + 3rd of the drives
        that stand, ``derived``, which differs from it: a disagreement under
        the floor's one rule (the model's N stands only with a box and a
        note, the other is kept beside it), listed (RI-04)."""
        n, basis = derived
        page = (m.prov.page if m.prov is not None
                else f.prov.page if f is not None and f.prov is not None
                else None)
        computed_confidence = _conf(f.prov if f is not None else m.prov, 0.5)
        s = settle("n", n, m.n, False, floor_method="derived",
                   floor_confidence=computed_confidence,
                   model_method=(m.prov.method if m.prov is not None
                                 and m.prov.method in
                                 ("model", "model_from_picture", "vision")
                                 else "model"),
                   model_confidence=_conf(m.prov, 0.9),
                   evidence=has_evidence(
                       m.prov, f.prov.bbox if f is not None and f.prov
                       else None),
                   model_note=m.prov.note if m.prov is not None else "")
        merged.n = s.value
        merged.n_basis = basis if s.verdict == "floor_wins" else ""
        if s.alternative is not None and merged.prov is not None:
            if s.verdict == "model_wins":
                s.alternative.note = f"N {basis}"
            merged.prov.alternatives.append(s.alternative)
        self.log.disagree(
            where, f"{what}: n", n, m.n,
            kept="model" if s.verdict == "model_wins" else "floor",
            why=("the model's N differs from the second plus the third of "
                 "the printed drives; "
                 + ("it named a box and a note, so it stands and the "
                    "drives' N is kept beside it"
                    if s.verdict == "model_wins" else
                    "it named no box or note, so the drives' N stands and "
                    "the model's is kept beside it")),
            page=page, floor_method="derived",
            floor_confidence=computed_confidence,
            model_confidence=_conf(m.prov, 0.9),
            model_method=m.prov.method if m.prov is not None else "model")

    def drives_n(self, merged: SPT, f: Optional[SPT],
                 beside: bool = False) -> Optional[Tuple[int, str]]:
        """The 2nd + 3rd of the merged record's drives: by the record's own
        drive lengths, increment or interval (:meth:`derived_n`), else the
        floor's N from the same drives. The floor read the 6 in increment
        off the column's header, which the record does not carry, and a
        model's record seldom gives any of the three (RI-45). Where the
        record gives its drive lengths, they alone decide. With ``beside``,
        also the drives' N the floor kept beside a printed N that differs
        from it."""
        derived = self.derived_n(merged)
        if derived is not None or merged.increments:
            return derived
        if f is None or list(f.blows) != list(merged.blows):
            return None
        if f.n_basis and f.n is not None:
            return f.n, f.n_basis
        if not beside:
            return None
        for alt in (f.prov.alternatives if f.prov is not None else []):
            if (alt.field == "n" and alt.method == "derived"
                    and str(alt.value).isdigit()):
                return int(alt.value), alt.note
        return None

    def recompute_n(self, merged: SPT, f: SPT, m: SPT, where: str,
                    what: str) -> None:
        """The floor computed its N from its own drives by the 6 in
        increment it read off the header, and the record that stands says
        otherwise: the model's drives replaced the floor's, or the record
        gives its own drive lengths (RI-45). The floor's N does not stay in
        the slot as it was had: the N is the 2nd + 3rd of the drives that
        stand, by the record's drive lengths where it gives them
        (:meth:`derived_n`), else by the increment the floor read, where
        they are three whole 6 in drives; otherwise the floor's N is ruled
        out, kept beside the slot and listed (:meth:`replace_n`), and an N
        the model read takes the slot where nothing rules it out too.

        An N the model printed stands as the slot merge left it, and is set
        against the drives that stand as any model N is. That holds where
        it is the very number the floor computed (RI-47): it is the model's
        reading, not the floor's computation, and is not dropped with it."""
        drives = "-".join(str(b) for b in f.blows)
        replaced = list(merged.blows) != list(f.blows)
        lengths = ", ".join(f"{q.value:g} {q.unit}"
                            for q in merged.increments)
        ruled_out = ("" if _whole_lengths(merged) else
                     f"; the record gives its drive lengths ({lengths}), "
                     f"not three whole 6 in drives, so this is no N of it")
        base_note = (f"the floor's N, of its drives {drives}, which the "
                     f"model's drives replaced: {f.n_basis}" if replaced
                     else f"the floor's N, {f.n_basis}")
        note = base_note + ruled_out
        if merged.increments:
            new = self.derived_n(merged)
        else:
            how = re.search(r"6 in drives from (.+)$", f.n_basis)
            new = n_from_drives(
                merged.blows, sampler=self.sampler_at(merged),
                increment_m=SPT_INCREMENT_M,
                increment_from=how.group(1) if how else "the floor's reading")
        if merged.n != f.n or m.n == f.n:
            # The model's printed N is in the slot: it won the vote, or it
            # is the number the floor computed, read off the page.
            merged.n_basis = ""
            for alt in (merged.prov.alternatives if merged.prov else []):
                if alt.field == "n" and alt.value == show(f.n):
                    alt.method = "derived"  # type: ignore[assignment]
                    alt.note = note
            if new is None or (new[0] == f.n and merged.n != f.n):
                return        # nothing to set it against, or already listed
            if new[0] == merged.n:
                if merged.n == f.n:
                    # As where the floor's drives stand: the drives give the
                    # N the model read, and the record says how.
                    merged.n_basis = new[1]
                return
            self.check_model_n(merged, f, m, where, what, new)
            return
        if new is not None:
            merged.n, merged.n_basis = new
            return
        out = {"floor": (
            f"the floor's N {f.n} is the 2nd + 3rd of its drives {drives} "
            f"at the 6 in increment it read"
            + ("; the model's drives replaced them" if replaced else "")
            + (f"; the record gives its drive lengths ({lengths})"
               if lengths else "")
            + f"; the drives that stand are not three whole 6 in drives, "
              f"so the record has no N from them ({N_RULE_SOURCE})")}
        if (m.n is not None and m.n < SPT_HALT_BLOWS
                and halts_before_n(merged)):
            out["model"] = self._halt_reason("model", m.n, merged)
        self.replace_n(merged, f, m, where, what, out,
                       floor_note=base_note)

    def check_refusal(self, merged: SPT, f: Optional[SPT], m: SPT,
                      where: str, what: str) -> None:
        """A refusal the model alone claims for a record of three whole 6 in
        drives of a named SPT (RI-45). The test drove all three increments,
        so it did not halt and the record has its N, the 2nd + 3rd
        (p. 3-51); the flag is cleared and the claim listed. A refusal
        printed where the N goes (``n_printed``) is left as it stands."""
        if not merged.refusal or merged.n_printed or not m.refusal:
            return
        if f is not None and f.refusal:
            return
        derived = self.drives_n(merged, f, beside=True)
        if derived is None:
            return
        merged.refusal = False
        self.log.disagree(
            where, f"{what}: refusal", False, True, kept="floor",
            why=("the model calls a record of three whole 6 in drives a "
                 "refusal; the test drove all three increments, so it did "
                 f"not halt and its N is the 2nd + 3rd, {derived[0]} "
                 f"({N_RULE_SOURCE}); not kept as a refusal"),
            page=(m.prov.page if m.prov is not None
                  else f.prov.page if f is not None and f.prov is not None
                  else None),
            floor_method="derived",
            floor_confidence=_conf(f.prov if f is not None else m.prov, 0.5),
            model_confidence=_conf(m.prov, 0.9),
            model_method=m.prov.method if m.prov is not None else "model")

    @staticmethod
    def halt_flag(merged: SPT) -> None:
        """A record that shows plainly that it ends in a halt, by its drives
        or its own drive lengths (:func:`shows_halt`), carries the refusal
        flag (RI-50): the flag follows how the record ends, whoever read it,
        and is set here, after the reader's brief, which the floor's seed
        leaves as it was. Where the halt rests only on a bare number read as
        short, or on a lone whole number, the flag stays as read."""
        if not merged.refusal and shows_halt(merged):
            merged.refusal = True

    @staticmethod
    def _halt_reason(whose: str, n: int, merged: SPT) -> str:
        return (f"the {whose}'s N {n} is under 50 on a record that halts "
                f"within its first three increments ({_halt_how(merged)}); "
                f"such a record has no N under 50 ({N_RULE_SOURCE})")

    def halt_n(self, merged: SPT, f: Optional[SPT], m: SPT, where: str,
               what: str) -> None:
        """An N under 50 in the slot of a record that halts within its
        first three increments, by its drives or its own drive lengths
        (RI-44, RI-50; :func:`halts_before_n`): it is ruled out, and
        so is any other N under 50 either voter read, and an N the floor
        computed from its drives, which a halted record has none of
        (:meth:`replace_n`)."""
        out: Dict[str, str] = {}
        if f is not None and f.n is not None:
            if f.n_basis:
                out["floor"] = (
                    f"the floor's N {f.n} is the 2nd + 3rd of its drives at "
                    f"the 6 in increment it read; the record ends in a halt "
                    f"({_halt_how(merged)}), so it has no N from its drives "
                    f"({N_RULE_SOURCE})")
            elif f.n < SPT_HALT_BLOWS:
                out["floor"] = self._halt_reason("floor", f.n, merged)
        if m.n is not None and m.n < SPT_HALT_BLOWS:
            out["model"] = self._halt_reason("model", m.n, merged)
        note = ""
        if f is not None and f.n_basis:
            note = f"the floor's N, {f.n_basis}"
        self.replace_n(merged, f, m, where, what, out, floor_note=note)

    def replace_n(self, merged: SPT, f: Optional[SPT], m: SPT, where: str,
                  what: str, out: Dict[str, str],
                  floor_note: str = "") -> None:
        """The N in the slot is ruled out; ``out`` says, by voter, why each
        N that cannot be this record's is not. The slot takes the other
        voter's N where nothing rules it out -- the N the record would hold
        had the ruled-out one never been read -- and is otherwise empty.
        Every N ruled out is kept beside the slot with its reason. One row
        lists what each voter read and which stands, in place of the vote's
        row or "added" entry for this slot, which no longer say what the
        record holds (RI-47)."""
        floor_n = f.n if f is not None else None
        model_n = m.n
        stands = ("floor" if floor_n is not None and "floor" not in out
                  else "model" if model_n is not None and "model" not in out
                  else "")
        merged.n = (floor_n if stands == "floor"
                    else model_n if stands == "model" else None)
        merged.n_basis = (f.n_basis if stands == "floor" and f is not None
                          else "")
        model_method = (m.prov.method if m.prov is not None
                        and m.prov.method in ("model", "model_from_picture",
                                              "vision") else "model")
        floor_method = ("derived" if f is not None and f.n_basis
                        else f.prov.method if f is not None
                        and f.prov is not None else "grid")
        if merged.prov is not None:
            alternatives = merged.prov.alternatives
            if stands:
                held = show(merged.n)
                alternatives[:] = [a for a in alternatives
                                   if not (a.field == "n"
                                           and a.value == held)]
            for whose, n, method, confidence, note in (
                    ("floor", floor_n, floor_method,
                     _conf(f.prov if f is not None else None, 0.5),
                     floor_note or "the floor's N"),
                    ("model", model_n, model_method, _conf(m.prov, 0.9),
                     "the model's N")):
                if whose not in out or n is None or (
                        stands and n == merged.n):
                    continue
                alt_note = f"{note}; {out[whose]}"
                same = next((a for a in alternatives
                             if a.field == "n" and a.value == show(n)), None)
                if same is not None:
                    same.method = method  # type: ignore[assignment]
                    same.note = alt_note
                else:
                    alternatives.append(Alternative(
                        field="n", value=show(n),
                        method=method,  # type: ignore[arg-type]
                        confidence=confidence, note=alt_note))
        slot = f"{what}: n"
        self.log.disagreements[:] = [
            r for r in self.log.disagreements
            if not (r["where"] == where and r["what"] == slot)]
        self.log.added[:] = [r for r in self.log.added
                             if not (r["where"] == where
                                     and r["what"] == slot)]
        kept = (stands or ("floor" if floor_n is None
                           else "model" if model_n is None else "neither"))
        if stands:
            other = "model" if stands == "floor" else "floor"
            tail = (f"the {stands}'s N {merged.n} stands, as it would had "
                    f"the {other}'s not been read")
        elif floor_n is not None and model_n is not None:
            tail = "neither is the N; both are kept beside the slot"
        else:
            tail = "the N is kept beside the slot, not in it"
        self.log.disagree(
            where, slot,
            "none" if floor_n is None else floor_n,
            "none" if model_n is None else model_n,
            kept=kept, why="; ".join(list(out.values()) + [tail]),
            page=(m.prov.page if m.prov is not None
                  else f.prov.page if f is not None and f.prov is not None
                  else None),
            floor_method=floor_method,
            floor_confidence=_conf(f.prov if f is not None else None, 0.5),
            model_confidence=_conf(m.prov, 0.9),
            model_method=model_method)

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
            out.append(self._with_floor_note(f, self.merge_pair(
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
                model_only=["note"])))
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
            where = f"{base}[{show(f.depth_top)}]"
            merged = self.merge_pair(
                where, what, f, m,
                specs=[("depth_bottom", "depth"), ("blows", "blows"),
                       ("n", "exact"), ("n_printed", "exact_text"),
                       ("increments", "drives"), ("refusal", "flag"),
                       ("sample_id", "exact_text"),
                       ("depth_top", "depth")],
                model_only=["hammer", "increment"])
            self.check_n(merged, f, m, where, what)
            self.check_refusal(merged, f, m, where, what)
            self.halt_flag(merged)
            out.append(merged)
        for f in floor_only:
            kept = self.keep_whole(f"{base}[{show(f.depth_top)}]",
                                   f"driven record at {show(f.depth_top)}",
                                   f)
            self.halt_flag(kept)
            out.append(kept)
        for m in model_only:
            what = f"driven record at {show(m.depth_top)}"
            where = f"{base}[{show(m.depth_top)}]"
            added = self.add_whole(where, what, m)
            self.check_n(added, None, m, where, what)
            self.check_refusal(added, None, m, where, what)
            self.halt_flag(added)
            out.append(added)
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
        self.merged_samples = merged.samples
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
