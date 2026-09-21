"""The floor under the lab reader: the page's own tables, and the merge.

WHY. The first full cluster run scored the lab reader at 87 % over 31 sheets
against 68 % for the page's detected tables alone -- and on six sheets the
reader came back BELOW the tables: a gradation 28/28 -> 11/30, a chemical
18/18 -> 14/22, a compaction 4/8 -> 1/10. The numbers were on the page, in a
table planlens had already found, and the model's answer left them out. So
the tables are the first voter here, the way the grid is for a log.

WHAT A TABLE CAN SEED, AND WHAT IT CANNOT. A detected table is a grid of
strings with the words printed beside the numbers, so the smallest honest
reading of it is: the label a row or a column carries names the value. A
row reading ``Liquid limit | 31`` is a liquid limit of 31; a column headed
``PERCENT FINER`` beside one headed ``SIEVE`` is a grading series; a table
with a BORING column and a DEPTH column and four rows is four specimens. The
sheet's TITLE names the test kind, and the lines reading ``Boring: B-4`` and
``Depth: 7.5 ft`` are the link to the ground, both read off the text layer
with a plain pattern. What a table cannot say -- which of two tables is the
result and which the trial weighings, what a curve reads where nothing is
tabulated, what a French sheet's column means -- is the model's, and the
model adds it.

THE MERGE is the same rule as the log's (:mod:`report_ingest.floor`): add,
correct with evidence, never drop. A floor value with no home in the model's
tests becomes a test of its own, kept with a note; a model value the floor
has no opinion on is accepted; a value they split on stays the floor's with
the model's beside it, unless the model named the box and said what it
read, in which case the reverse. Every split is a disagreement for QA.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from report_ingest.floor import (
    EXACT_TOL, MergeLog, fold, has_evidence, same_depth, same_number,
    same_text, settle, show,
)
from report_ingest.model import (
    Alternative, AtterbergResult, CBRResult, ChemicalResult, CompactionResult,
    ConsolidationResult, GradationResult, LabTest, MoistureDensityResult,
    OtherResult, Provenance, Quantity, RESULT_CLASS, SievePoint,
    StrengthResult, SummaryRow, SummaryTableResult, _canonical_unit,
)

__all__ = ["FloorSheet", "floor_from_tables", "serialise_floor",
           "merge_lab_tests", "kind_from_title", "TITLE_KINDS",
           "PASSING_TOL"]

#: A grading's percent-passing values agree within one percentage point
#: (``lab_scoring.PASSING_TOL``).
PASSING_TOL = 1.0
#: How sure the floor is of a value read off a detected table: a ruled table
#: on a vector text layer is exact; one read optically is not.
CONF_TABLE = 0.85
CONF_TABLE_OCR = 0.6
CONF_TITLE = 0.75
CONF_LINK = 0.7

#: What a sheet's title says it is. Tested in this order, so the specific
#: names win: a SUMMARY OF LABORATORY TESTS names every test kind on it and
#: is none of them, and a CERTIFICATE OF ANALYSIS reports nothing.
TITLE_KINDS: Tuple[Tuple[str, str], ...] = (
    (r"summary of lab|laboratory (?:test )?summary|summary of (?:soil )?test|"
     r"resumen de ensayos|recapitulatif", "summary_table"),
    (r"certificate|chain of custody|sample receipt|samples received|"
     r"laboratory order", "other"),
    (r"atterberg|liquid limit|plastic(?:ity)? (?:limit|index|chart)|"
     r"limites? d'?atterberg|limites? de atterberg|limite liquide|"
     r"limite liquido", "atterberg"),
    (r"particle[- ]size|grain[- ]size|sieve|gradation|granulom|hydrometer|"
     r"tamiz|gradacion", "gradation"),
    (r"consolidation|swell|oedometer|collapse|expansion index",
     "swell_consolidation"),
    (r"triaxial|triaxiale?", "triaxial"),
    (r"direct shear|shear box|cisaillement|corte directo", "direct_shear"),
    (r"(?:unconfined|uniaxial).{0,30}(?:rock|core)|"
     r"(?:rock|core).{0,30}(?:unconfined|uniaxial)|point load",
     "unconfined_rock"),
    (r"unconfined|compression simple|compresion simple", "unconfined"),
    (r"proctor|compaction|moisture[- ]density relation|densite optimale",
     "compaction"),
    (r"\bcbr\b|bearing ratio", "cbr"),
    (r"moisture content|water content|teneur en eau|humedad natural|"
     r"contenido de (?:agua|humedad)", "moisture_content"),
    (r"organic|loss on ignition|ignition", "organic_content"),
    (r"corros|resistivity|chemical|sulfate|sulphate|chloride|\bph\b",
     "chemical"),
    (r"specific gravity|gravedad especifica|densite des grains",
     "specific_gravity"),
    (r"permeability|hydraulic conductivity|permeabilite|permeabilidad",
     "permeability"),
    (r"\bdensity\b|unit weight|densite|densidad", "density"),
)

#: A printed label -> the record's own name for the value. Applied to a
#: row's first cell in a property/value table and to a column head in a
#: header-row table, after folding to lower case and one space. The first
#: pattern that matches wins, so the longer names come first.
LABEL_FIELDS: Tuple[Tuple[str, str], ...] = (
    (r"max(?:imum)?\.? dry (?:density|unit weight)|mdd", "max_dry_density"),
    (r"optimum (?:moisture|water)|omc|opt\.? (?:w|m)", "optimum_wc"),
    (r"liquid limit|\bll\b|limite liquide|limite liquido|\bwl\b", "ll"),
    (r"plastic limit|\bpl\b|limite plastique|limite plastico|\bwp\b", "pl"),
    (r"plasticity index|\bpi\b|\bip\b|indice de plasticit", "pi"),
    (r"shrinkage limit|\bsl\b", "shrinkage_limit"),
    (r"percent finer|% finer|percent passing|% passing|passing|finer|"
     r"tamisat|que pasa|pasante", "percent_passing"),
    (r"sieve|tamis|tamiz|mesh", "sieve"),
    (r"size|opening|diam|ouverture|abertura", "size"),
    (r"gravel|gravier|grava", "gravel_percent"),
    (r"\bsand\b|sable|arena", "sand_percent"),
    (r"fines|-200|no\.? 200|silt (?:and|&) clay|silt/clay|passing 200|"
     r"fins|finos", "fines_percent"),
    (r"\bsilt\b|limon", "silt_percent"),
    (r"\bclay\b|argile|arcilla", "clay_percent"),
    (r"\bd100\b", "d100"), (r"\bd90\b", "d90"), (r"\bd85\b", "d85"),
    (r"\bd60\b", "d60"), (r"\bd50\b", "d50"), (r"\bd30\b", "d30"),
    (r"\bd10\b", "d10"),
    (r"coefficient of uniformity|\bcu\b", "cu"),
    (r"coefficient of curvature|\bcc\b", "cc"),
    (r"dry (?:density|unit weight)|densite seche|densidad seca|\bgamma ?d\b",
     "dry_density"),
    (r"(?:wet|bulk|moist|total) (?:density|unit weight)|unit weight|"
     r"densite humide|densidad humeda", "wet_density"),
    (r"specific gravity|\bgs\b|gravedad especifica", "specific_gravity"),
    (r"void ratio|\be0\b|\be_0\b", "void_ratio"),
    (r"saturation|\bsr\b", "saturation_percent"),
    (r"(?:natural )?(?:water|moisture) content|moisture|\bwc\b|\bw%|"
     r"\bmc\b|\bwn\b|teneur en eau|humedad|\bw\b", "wc"),
    (r"\bph\b", "pH"),
    (r"resistivity|resistivit", "resistivity"),
    (r"sulfate|sulphate|sulfato", "sulfate"),
    (r"chloride|chlorure|cloruro", "chloride"),
    (r"sulfide|sulphide|sulfuro", "sulfides"),
    (r"redox", "redox"),
    (r"\bcbr\b|bearing ratio", "cbr_percent"),
    (r"cohesion|\bc'?\b(?!\w)", "c"),
    (r"friction angle|phi|\bφ|angle de frottement|angulo de friccion",
     "phi_deg"),
    (r"unconfined|uniaxial|\bqu\b|\bucs\b|compressive strength", "qu"),
    (r"undrained shear|\bsu\b|\bcu\b(?= ?\()", "su"),
    (r"swell|expansion", "swell_percent"),
    (r"organic|loss on ignition|\bloi\b", "organic_percent"),
    (r"preconsolidation|\bpc\b|\bp'?c\b", "pc"),
    (r"compression index", "compression_index"),
    (r"recompression index", "recompression_index"),
    (r"date|sampled|received|tested|matrix|lab id|laboratory id|order",
     "date"),
    (r"boring|borehole|bore ?hole|\bhole\b|exploration|sondage|sondeo|"
     r"test pit|\bpit\b|location", "investigation_id"),
    (r"sample|spec(?:imen)?\b|muestra|echantillon|\bno\.?\b", "sample_id"),
    (r"depth|prof(?:ondeur|undidad)?\b|elevation", "depth_top"),
    (r"uscs|class|group symbol|symbol", "uscs"),
    (r"description|material|soil type|descripcion", "description"),
    (r"type", "sample_type"),
)

#: Which result class a floor value lives in. ``wc`` and the densities move
#: with the sheet's kind; everything else has one home.
FIELD_HOME: Dict[str, str] = {
    "ll": "atterberg", "pl": "atterberg", "pi": "atterberg",
    "shrinkage_limit": "atterberg",
    "gravel_percent": "gradation", "sand_percent": "gradation",
    "fines_percent": "gradation", "silt_percent": "gradation",
    "clay_percent": "gradation", "d10": "gradation", "d30": "gradation",
    "d50": "gradation", "d60": "gradation", "d85": "gradation",
    "d90": "gradation", "d100": "gradation", "cu": "gradation",
    "cc": "gradation", "percent_passing": "gradation",
    "max_dry_density": "compaction", "optimum_wc": "compaction",
    "pH": "chemical", "resistivity": "chemical", "sulfate": "chemical",
    "chloride": "chemical", "sulfides": "chemical", "redox": "chemical",
    "cbr_percent": "cbr",
    "c": "strength", "phi_deg": "strength", "qu": "strength",
    "su": "strength",
    "swell_percent": "swell_consolidation", "pc": "swell_consolidation",
    "compression_index": "swell_consolidation",
    "recompression_index": "swell_consolidation",
    "specific_gravity": "moisture", "void_ratio": "moisture",
    "saturation_percent": "moisture", "organic_percent": "moisture",
    "wc": "moisture", "dry_density": "moisture", "wet_density": "moisture",
}
#: The record's field for a floor name, where it differs.
_RENAME: Dict[str, Dict[str, str]] = {
    "atterberg": {"wc": "water_content"},
    "gradation": {"wc": "water_content"},
    "swell_consolidation": {"compression_index": "cc",
                            "recompression_index": "cr",
                            "dry_density": "dry_unit_weight"},
}
_STRENGTH_KINDS = ("triaxial", "direct_shear", "unconfined",
                   "unconfined_rock")
_MOISTURE_KINDS = ("moisture_content", "density", "organic_content")

_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")
_PAREN = re.compile(r"\(([^)]{1,14})\)")
_LINK_BORING = re.compile(
    r"(?:boring|borehole|bore ?hole|test ?pit|sondage|sondeo|exploration|"
    r"hole|location)\s*(?:no\.?|number|#|id)?\s*[:.\-]?\s*"
    r"([A-Za-z]{1,4}[- ]?\d+[A-Za-z]?)\b", re.I)
_LINK_SAMPLE = re.compile(
    r"(?:sample|specimen|muestra|echantillon)\s*(?:no\.?|number|#|id)?\s*"
    r"[:.\-]?\s*([A-Za-z]*[- ]?\d+[A-Za-z]?)\b", re.I)
_LINK_DEPTH = re.compile(
    r"(?:depth|profondeur|profundidad|prof\.?)\s*[:.\-]?\s*"
    r"(\d+(?:[.,]\d+)?)\s*(?:(?:-|–|to|a)\s*(\d+(?:[.,]\d+)?))?\s*"
    r"(ft|feet|foot|m|metres|meters|'|\")?", re.I)


# ---------------------------------------------------------------------------
# reading the page
# ---------------------------------------------------------------------------

def _num(text: Any) -> Optional[float]:
    """The number a cell prints, or None. ``1,45`` is 1.45; ``<10`` is None
    -- below the reporting limit is words, not the number ten."""
    raw = str(text or "").strip()
    if not raw or raw.startswith("<") or raw.startswith(">"):
        return None
    match = _NUMBER.fullmatch(raw.replace(" ", "").rstrip("%"))
    if match is None:
        return None
    return float(match.group(0).replace(",", "."))


def _field_of(label: str) -> str:
    text = " ".join(str(label or "").lower().replace("_", " ").split())
    text = text.replace("(", " (").replace("%", " % ")
    text = " ".join(text.split())
    for pattern, name in LABEL_FIELDS:
        if re.search(pattern, text):
            return name
    return ""


def _unit_in(label: str) -> str:
    for token in _PAREN.findall(str(label or "")):
        key = _canonical_unit(token)
        if key:
            return token.strip()
    return ""


def kind_from_title(lines: Sequence[str]) -> Tuple[str, str]:
    """``(kind, the title line)`` from the top of the page, or ``("", "")``."""
    for line in lines:
        text = " ".join(str(line or "").lower().split())
        for pattern, kind in TITLE_KINDS:
            if re.search(pattern, text):
                return kind, str(line).strip()
    return "", ""


@dataclass
class _Link:
    investigation_id: str = ""
    sample_id: str = ""
    depth_top: Optional[float] = None
    depth_bottom: Optional[float] = None
    unit: str = ""
    page: int = 0
    bbox: Optional[Tuple[float, float, float, float]] = None


@dataclass
class FloorSheet:
    """What the page's own tables and title say, before any model call."""

    kind: str = ""
    title: str = ""
    link: _Link = field(default_factory=_Link)
    tests: List[LabTest] = field(default_factory=list)
    n_values: int = 0
    pages: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def _lines_of(doc: Any, page: int) -> List[Any]:
    try:
        content = doc.page(page)
    except Exception:
        return []
    return sorted(content.lines, key=lambda ln: (round(ln.bbox[1], 1),
                                                 ln.bbox[0]))


def _tables_of(doc: Any, page: int) -> List[Any]:
    try:
        return list(doc.page(page).tables)
    except Exception:
        return []


def _link_from_lines(lines: Sequence[Any], page: int) -> _Link:
    link = _Link(page=page)
    for line in lines:
        text = str(line.text or "")
        if not link.investigation_id:
            match = _LINK_BORING.search(text)
            if match:
                link.investigation_id = match.group(1).strip().replace(" ", "-")
                link.bbox = tuple(line.bbox)
        if not link.sample_id:
            match = _LINK_SAMPLE.search(text)
            if match:
                link.sample_id = match.group(1).strip()
        if link.depth_top is None:
            match = _LINK_DEPTH.search(text)
            if match:
                link.depth_top = float(match.group(1).replace(",", "."))
                if match.group(2):
                    link.depth_bottom = float(match.group(2).replace(",", "."))
                unit = (match.group(3) or "").lower()
                link.unit = ("ft" if unit in ("ft", "feet", "foot", "'")
                             else "m" if unit in ("m", "metres", "meters")
                             else "")
    return link


def _table_rows(table: Any) -> List[List[str]]:
    rows = [[str(c or "").strip() for c in row] for row in (table.rows or [])]
    if table.header:
        rows.insert(0, [str(c or "").strip() for c in table.header])
    return [r for r in rows if any(r)]


def _is_header_row(row: Sequence[str]) -> bool:
    words = [c for c in row if c and _num(c) is None]
    return len(words) >= max(1, len([c for c in row if c]) // 2)


def _table_conf(table: Any) -> float:
    source = str(getattr(table, "source", "") or "").lower()
    return CONF_TABLE_OCR if ("ocr" in source or "di" in source
                              or "azure" in source) else CONF_TABLE


@dataclass
class _Values:
    """One specimen's floor values, keyed by the record's names."""

    index: Dict[str, Tuple[Any, str, float]] = field(default_factory=dict)
    series: List[Tuple[str, Optional[float], float, float]] = field(
        default_factory=list)   # (sieve, size mm, percent, confidence)
    provs: List[Provenance] = field(default_factory=list)

    def put(self, name: str, value: Any, unit: str, conf: float) -> None:
        if name and value is not None and name not in self.index:
            self.index[name] = (value, unit, conf)


def _read_property_table(rows: List[List[str]], values: _Values,
                         conf: float) -> int:
    """``label | value`` rows. Returns how many values were read."""
    n = 0
    for row in rows:
        cells = [c for c in row if c]
        if len(cells) < 2:
            continue
        label, cell = cells[0], cells[1]
        name = _field_of(label)
        if not name or name in ("investigation_id", "sample_id", "depth_top",
                                "uscs", "description", "sample_type",
                                "sieve", "size", "percent_passing", "date"):
            continue
        number = _num(cell)
        unit = _unit_in(label) or _unit_in(cell)
        if number is None:
            if name in ("pH", "resistivity", "sulfate", "chloride",
                        "sulfides", "redox") and cell:
                values.put(name, cell, unit, conf)
                n += 1
            continue
        values.put(name, number, unit, conf)
        n += 1
    return n


def _series_from(rows: List[List[str]], header: List[str],
                 columns: Dict[int, str], conf: float
                 ) -> List[Tuple[str, Optional[float], float, float]]:
    sieve_col = next((i for i, n in columns.items() if n == "sieve"), None)
    size_col = next((i for i, n in columns.items() if n == "size"), None)
    pass_col = next((i for i, n in columns.items()
                     if n == "percent_passing"), None)
    if pass_col is None or (sieve_col is None and size_col is None):
        return []
    size_unit = _unit_in(header[size_col]) if size_col is not None else ""
    out: List[Tuple[str, Optional[float], float, float]] = []
    for row in rows:
        if pass_col >= len(row):
            continue
        percent = _num(row[pass_col])
        if percent is None or not (0.0 <= percent <= 100.0):
            continue
        sieve = row[sieve_col] if sieve_col is not None \
            and sieve_col < len(row) else ""
        size = _num(row[size_col]) if size_col is not None \
            and size_col < len(row) else None
        if size is not None and size_unit and size_unit.lower() != "mm":
            converted = Quantity(value=size, unit=size_unit).si_value
            size = converted * 1000.0 if converted is not None else None
        if not sieve and size is not None:
            sieve = f"{size:g} mm"
        out.append((sieve, size, percent, conf))
    return out


def _summary_rows(rows: List[List[str]], header: List[str],
                  columns: Dict[int, str], unit: str, conf: float,
                  prov: Provenance) -> List[SummaryRow]:
    out: List[SummaryRow] = []
    for row in rows:
        cells = {i: (row[i] if i < len(row) else "") for i in columns}
        get = {name: cells[i] for i, name in columns.items()}
        if not any(get.get(k) for k in ("investigation_id", "sample_id",
                                        "depth_top")):
            continue
        passing: List[SievePoint] = []
        other: List[Tuple[str, str]] = []
        kwargs: Dict[str, Any] = {}
        for i, name in columns.items():
            text = cells[i]
            if not text:
                continue
            head = header[i] if i < len(header) else ""
            if name == "fines_percent" and re.search(
                    r"200|fines|-200", head, re.I) and _num(text) is not None:
                passing.append(SievePoint(
                    percent_passing=max(0.0, min(100.0, _num(text))),
                    sieve="No. 200",
                    size=Quantity(value=0.075, unit="mm")))
                kwargs["fines_percent"] = max(0.0, min(100.0, _num(text)))
                continue
            if name in ("investigation_id", "sample_id", "description",
                        "uscs", "sample_type"):
                kwargs[name] = text
                continue
            if name == "depth_top":
                number = _num(text)
                if number is not None:
                    kwargs["depth_top"] = Quantity(value=number,
                                                   unit=unit or _unit_in(head))
                continue
            number = _num(text)
            row_unit = _unit_in(head)
            if name == "wc" and number is not None:
                kwargs["wc"] = number
            elif name in ("ll", "pi", "sand_percent", "gravel_percent",
                          "optimum_wc", "phi_deg", "swell_percent",
                          "organic_percent") and number is not None:
                kwargs[name] = number
            elif name == "pl":
                kwargs["pl"] = number if number is not None else text
            elif name in ("silt_percent", "clay_percent") and number is not None:
                kwargs["silt_clay_percent"] = number
            elif name in ("wet_density", "dry_density", "max_dry_density",
                          "qu", "su", "c") and number is not None:
                kwargs[name] = Quantity(value=number, unit=row_unit)
            elif name in ("pH", "resistivity", "sulfate", "chloride",
                          "sulfides", "redox"):
                kwargs[name] = (Quantity(value=number, unit=row_unit)
                                if number is not None and row_unit
                                else number if number is not None else text)
            else:
                other.append((head, text))
        try:
            out.append(SummaryRow(percent_passing=passing, other=other,
                                  **kwargs))
        except Exception:                      # a row the model cannot hold
            continue
    return out


def _read_header_table(rows: List[List[str]], values_by_key: Dict[str, _Values],
                       link: _Link, kind: str, conf: float, prov: Provenance,
                       ) -> Tuple[int, List[SummaryRow]]:
    """A table whose first row names its columns.

    Three shapes: a grading series (a sieve or size column beside a percent
    passing column); a table of specimens (a boring, sample or depth column
    and more than one row), which seeds summary rows; and a one-row table of
    values for the sheet's one specimen.
    """
    header = rows[0]
    body = rows[1:]
    columns = {i: _field_of(h) for i, h in enumerate(header)
               if _field_of(h) and _field_of(h) != "date"}
    if not body or not columns:
        return 0, []
    series = _series_from(body, header, columns, conf)
    if series:
        key = _specimen_key(link)
        values_by_key.setdefault(key, _Values()).series.extend(series)
        values_by_key[key].provs.append(prov)
        return len(series), []
    link_names = ("investigation_id", "sample_id", "depth_top", "uscs",
                  "description", "sample_type")
    has_link = any(n in link_names[:3] for n in columns.values())
    has_value = any(n not in link_names for n in columns.values())
    if not has_value:
        # A list of samples with no result column -- a certificate's
        # receipt table, a chain of custody -- seeds nothing.
        return 0, []
    if has_link and (len(body) > 1 or kind == "summary_table"):
        summary = _summary_rows(body, header, columns, link.unit, conf, prov)
        return sum(len([v for v in r.model_dump().values() if v])
                   for r in summary), summary
    n = 0
    row = body[0]
    key = _specimen_key(link)
    values = values_by_key.setdefault(key, _Values())
    values.provs.append(prov)
    for i, name in columns.items():
        if i >= len(row) or not row[i]:
            continue
        if name in ("investigation_id", "sample_id", "depth_top", "uscs",
                    "description", "sample_type", "sieve", "size",
                    "percent_passing"):
            continue
        number = _num(row[i])
        unit = _unit_in(header[i])
        if number is None:
            if name in ("pH", "resistivity", "sulfate", "chloride",
                        "sulfides", "redox"):
                values.put(name, row[i], unit, conf)
                n += 1
            continue
        values.put(name, number, unit, conf)
        n += 1
    return n, []


def _specimen_key(link: _Link) -> str:
    return f"{fold(link.investigation_id)}|{fold(link.sample_id)}|" \
           f"{'' if link.depth_top is None else round(link.depth_top, 2)}"


# ---------------------------------------------------------------------------
# the seed
# ---------------------------------------------------------------------------

def _q(value: Any, unit: str) -> Any:
    if isinstance(value, str):
        return value
    if value is None:
        return None
    return Quantity(value=float(value), unit=unit) if unit else float(value)


def _reported(value: Any, unit: str) -> Any:
    if isinstance(value, str):
        return value
    if value is None:
        return None
    return Quantity(value=float(value), unit=unit) if unit else float(value)


def _build_result(kind: str, values: _Values, hint: str) -> Any:
    """The typed result a set of floor values makes, for one kind."""
    idx = values.index

    def num(name: str) -> Optional[float]:
        got = idx.get(name)
        return None if got is None or isinstance(got[0], str) else \
            float(got[0])

    def qty(name: str) -> Optional[Quantity]:
        got = idx.get(name)
        if got is None or isinstance(got[0], str):
            return None
        return Quantity(value=float(got[0]), unit=got[1])

    def rep(name: str) -> Any:
        got = idx.get(name)
        return None if got is None else _reported(got[0], got[1])

    if kind == "atterberg":
        return AtterbergResult(ll=num("ll"), pl=num("pl"), pi=num("pi"),
                               shrinkage_limit=num("shrinkage_limit"),
                               water_content=num("wc"))
    if kind == "gradation":
        points = [SievePoint(percent_passing=max(0.0, min(100.0, pct)),
                             size=(Quantity(value=size, unit="mm")
                                   if size is not None else None),
                             sieve=sieve)
                  for sieve, size, pct, _c in values.series]
        return GradationResult(
            percent_passing=points,
            d10=qty("d10") or _mm(idx, "d10"), d30=qty("d30") or _mm(idx, "d30"),
            d50=qty("d50") or _mm(idx, "d50"), d60=qty("d60") or _mm(idx, "d60"),
            d85=qty("d85") or _mm(idx, "d85"), d90=qty("d90") or _mm(idx, "d90"),
            d100=qty("d100") or _mm(idx, "d100"),
            cu=num("cu"), cc=num("cc"),
            gravel_percent=num("gravel_percent"),
            sand_percent=num("sand_percent"),
            silt_percent=num("silt_percent"), clay_percent=num("clay_percent"),
            fines_percent=num("fines_percent"), water_content=num("wc"))
    if kind == "compaction":
        return CompactionResult(max_dry_density=qty("max_dry_density"),
                                optimum_wc=num("optimum_wc"))
    if kind == "chemical":
        return ChemicalResult(pH=rep("pH"), resistivity=rep("resistivity"),
                              sulfate=rep("sulfate"), chloride=rep("chloride"),
                              sulfides=rep("sulfides"), redox=rep("redox"),
                              organic_percent=num("organic_percent"),
                              wc=num("wc"))
    if kind == "cbr":
        return CBRResult(cbr_percent=num("cbr_percent"),
                         swell_percent=num("swell_percent"),
                         dry_density=qty("dry_density"), wc=num("wc"))
    if kind in _STRENGTH_KINDS:
        return StrengthResult(kind=kind, c=qty("c"), phi_deg=num("phi_deg"),
                              qu=qty("qu"), su=qty("su"), wc=num("wc"),
                              dry_density=qty("dry_density"),
                              wet_density=qty("wet_density"))
    if kind == "swell_consolidation":
        return ConsolidationResult(
            swell_percent=num("swell_percent"), pc=qty("pc"),
            cc=num("compression_index"), cr=num("recompression_index"),
            dry_unit_weight=qty("dry_density"), wc=num("wc"),
            saturation_percent=num("saturation_percent"))
    if kind in _MOISTURE_KINDS:
        return MoistureDensityResult(
            kind=kind, wc=num("wc"), wet_density=qty("wet_density"),
            dry_density=qty("dry_density"),
            specific_gravity=num("specific_gravity"),
            void_ratio=num("void_ratio"),
            saturation_percent=num("saturation_percent"),
            organic_percent=num("organic_percent"))
    if kind == "summary_table":
        return SummaryTableResult()
    return OtherResult(kind=kind if kind in ("other", "specific_gravity",
                                             "permeability") else "other",
                       no_results=(hint == "other"),
                       fields={k: show(v[0]) for k, v in idx.items()})


def _mm(idx: Dict[str, Any], name: str) -> Optional[Quantity]:
    got = idx.get(name)
    if got is None or isinstance(got[0], str):
        return None
    return Quantity(value=float(got[0]), unit="mm")


def _kinds_for(values: _Values, hint: str) -> List[str]:
    """Which result kinds one specimen's floor values call for."""
    homes: List[str] = []
    for name in values.index:
        home = FIELD_HOME.get(name, "")
        if home == "moisture":
            home = (hint if hint in _MOISTURE_KINDS + ("atterberg",
                                                       "gradation", "cbr",
                                                       "swell_consolidation")
                    + _STRENGTH_KINDS else "moisture_content")
            if hint in _STRENGTH_KINDS:
                home = hint
        elif home == "strength":
            home = hint if hint in _STRENGTH_KINDS else (
                "unconfined" if "qu" in values.index else "direct_shear")
        if home and home not in homes:
            homes.append(home)
    if values.series and "gradation" not in homes:
        homes.append("gradation")
    if hint and hint not in homes and hint != "summary_table":
        if not homes or hint in ("other",):
            homes.append(hint)
        elif hint not in ("moisture_content", "density", "organic_content"):
            homes.append(hint)
    # 'wc' alone on an Atterberg sheet is the natural water content beside
    # the limits, not a moisture test of its own.
    if hint in ("atterberg", "gradation") and "moisture_content" in homes \
            and hint in homes:
        homes.remove("moisture_content")
    order = [k for k in ([hint] if hint else []) + homes if k]
    seen: List[str] = []
    for k in order:
        if k not in seen and k in RESULT_CLASS:
            seen.append(k)
    return seen


def floor_from_tables(doc: Any, pages: Sequence[int],
                      report_id: str = "") -> FloorSheet:
    """The page's own tables and title as :class:`LabTest` records.

    Nothing here is inferred beyond the word printed beside the number. A
    sheet whose title names no kind and whose tables hold no labelled value
    comes back with an empty test list, which is an honest floor.
    """
    pages = [int(p) for p in pages]
    sheet = FloorSheet(pages=pages)
    if not pages:
        return sheet
    first = _lines_of(doc, pages[0])
    try:
        height = float(doc.summary(pages[0]).height)
    except Exception:
        height = 792.0
    top = [ln.text for ln in first if ln.bbox[1] <= 0.22 * height] \
        or [ln.text for ln in first[:6]]
    sheet.kind, sheet.title = kind_from_title(top)
    sheet.link = _link_from_lines(first, pages[0])
    link = sheet.link

    values_by_key: Dict[str, _Values] = {}
    summary_rows: List[SummaryRow] = []
    table_provs: List[Provenance] = []
    for page in pages:
        for table in _tables_of(doc, page):
            rows = _table_rows(table)
            if not rows:
                continue
            conf = _table_conf(table)
            prov = Provenance(page=page, bbox=tuple(table.bbox),
                              method="tables", confidence=conf,
                              note=f"table {getattr(table, 'id', '')} "
                                   f"({len(rows)} rows)".strip())
            table_provs.append(prov)
            if _is_header_row(rows[0]) and len(rows) >= 2 and \
                    len([c for c in rows[0] if c]) >= 2:
                n, summary = _read_header_table(rows, values_by_key, link,
                                                sheet.kind, conf, prov)
                if summary:
                    summary_rows.extend(summary)
                sheet.n_values += n
                if n or summary:
                    continue
            key = _specimen_key(link)
            values = values_by_key.setdefault(key, _Values())
            n = _read_property_table(rows, values, conf)
            if n:
                values.provs.append(prov)
                sheet.n_values += n

    link_prov = Provenance(page=link.page, bbox=link.bbox, method="text",
                           confidence=CONF_LINK,
                           note="the boring and depth printed on the sheet")
    depth = (Quantity(value=link.depth_top, unit=link.unit)
             if link.depth_top is not None else None)
    depth_bottom = (Quantity(value=link.depth_bottom, unit=link.unit)
                    if link.depth_bottom is not None else None)

    if summary_rows:
        prov = table_provs[0] if table_provs else link_prov
        sheet.tests.append(LabTest(
            kind="summary_table",
            result=SummaryTableResult(rows=summary_rows, title=sheet.title),
            pages=list(pages), source_report=report_id,
            prov=[prov]))

    for key, values in values_by_key.items():
        for kind in _kinds_for(values, sheet.kind):
            if kind == "summary_table":
                continue
            result = _build_result(kind, values, sheet.kind)
            provs = [link_prov] + [p for p in values.provs]
            sheet.tests.append(LabTest(
                kind=kind, investigation_id=link.investigation_id,
                sample_id=link.sample_id, depth_top=depth,
                depth_bottom=depth_bottom, pages=list(pages),
                source_report=report_id, result=result, prov=provs))

    if not sheet.tests and sheet.kind:
        # The title names the kind and the tables hold no value: the kind
        # alone is the floor, and it is an answer.
        result = (OtherResult(no_results=True) if sheet.kind == "other"
                  else None)
        sheet.tests.append(LabTest(
            kind=sheet.kind, investigation_id=link.investigation_id,
            sample_id=link.sample_id, depth_top=depth,
            depth_bottom=depth_bottom, pages=list(pages),
            source_report=report_id, result=result,
            prov=[Provenance(page=pages[0], method="text",
                             confidence=CONF_TITLE,
                             note=f"kind from the title: {sheet.title[:60]}")]))
    return sheet


# ---------------------------------------------------------------------------
# what the model is shown
# ---------------------------------------------------------------------------

def _result_lines(test: LabTest) -> List[str]:
    result = test.result
    if result is None:
        return ["(no value in a table; the kind alone)"]
    out: List[str] = []
    if isinstance(result, SummaryTableResult):
        for row in result.rows:
            bits = [f"{n} {show(getattr(row, n))}" for n in (
                "wc", "ll", "pl", "pi", "fines_percent", "sand_percent",
                "gravel_percent", "dry_density", "wet_density", "qu")
                if getattr(row, n) is not None]
            bits += [f"passing {p.sieve} {p.percent_passing:g}"
                     for p in row.percent_passing]
            bits += [f"{h} '{v}'" for h, v in row.other]
            out.append(f"  row {row.investigation_id} {row.sample_id} "
                       f"{show(row.depth_top)}: " + ", ".join(bits))
        return out
    bits: List[str] = []
    for name, value in result.model_dump().items():
        if name in ("kind", "percent_passing", "points", "specimens",
                    "flow_curve", "pl_trials", "water_contents",
                    "description", "fields", "no_results", "test_type",
                    "uscs", "method", "rock_type", "weathering",
                    "lab_sample_id", "hydrometer") \
                or value in (None, "", [], False):
            continue
        bits.append(f"{name} {show(getattr(result, name))}")
    if isinstance(result, GradationResult) and result.percent_passing:
        bits.append("percent passing: " + ", ".join(
            f"{p.sieve} {p.percent_passing:g}"
            + (f" ({show(p.size)})" if p.size else "")
            for p in result.percent_passing))
    if isinstance(result, OtherResult):
        bits += [f"{k} '{v}'" for k, v in result.fields.items()]
        if result.no_results:
            bits.append("no results on this page")
    out.append("  " + (", ".join(bits) or "(no value)"))
    return out


def serialise_floor(sheet: FloorSheet) -> str:
    """The floor as compact lines the model builds on."""
    out: List[str] = []
    out.append(f"kind from the title: {sheet.kind or 'none read'}"
               + (f" ('{sheet.title[:70]}')" if sheet.title else ""))
    link = sheet.link
    bits = []
    if link.investigation_id:
        bits.append(f"boring '{link.investigation_id}'")
    if link.sample_id:
        bits.append(f"sample '{link.sample_id}'")
    if link.depth_top is not None:
        bits.append(f"depth {link.depth_top:g}"
                    + (f" to {link.depth_bottom:g}" if link.depth_bottom
                       is not None else "")
                    + (f" {link.unit}" if link.unit else " (unit not printed)"))
    out.append("link printed on the sheet: " + (", ".join(bits) or "none read"))
    for test in sheet.tests:
        where = ", ".join(
            f"page {p.page}" + (" box " + ",".join(f"{v:.0f}" for v in p.bbox)
                                if p.bbox else "")
            for p in test.prov if p.method == "tables") or \
            f"page {test.pages[0] if test.pages else '?'}"
        conf = max((p.confidence for p in test.prov), default=0.5)
        out.append(f"{test.kind} ({where})  [confidence {conf:.2f}]")
        out.extend(_result_lines(test))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# the merge
# ---------------------------------------------------------------------------

def _same_specimen(a: LabTest, b: LabTest) -> bool:
    ida, idb = fold(a.investigation_id), fold(b.investigation_id)
    if ida and idb and ida != idb:
        return False
    same = same_depth(a.depth_top, b.depth_top)
    if same is False:
        return False
    if not (ida and idb) and same is None:
        # Neither side names a boring and a depth: one sheet, one specimen,
        # unless the sample labels say otherwise.
        sa, sb = fold(a.sample_id), fold(b.sample_id)
        return not (sa and sb and sa != sb)
    return True


def _same_value(name: str, fv: Any, mv: Any) -> Optional[bool]:
    if fv is None or mv is None or fv == "" or mv == "":
        return None
    if isinstance(fv, str) or isinstance(mv, str):
        return same_text(fv, mv)
    if isinstance(fv, Quantity) or isinstance(mv, Quantity):
        fq = fv if isinstance(fv, Quantity) else Quantity(value=float(fv),
                                                          unit="")
        mq = mv if isinstance(mv, Quantity) else Quantity(value=float(mv),
                                                          unit="")
        if fq.unit and mq.unit and fold(fq.unit) != fold(mq.unit):
            left, right = fq.si_value, mq.si_value
            if left is None or right is None:
                return same_number(fq.value, mq.value, EXACT_TOL)
            return same_number(left, right, max(EXACT_TOL, abs(right) * 0.005))
        return same_number(fq.value, mq.value, EXACT_TOL)
    if isinstance(fv, bool) or isinstance(mv, bool):
        return bool(fv) == bool(mv)
    if isinstance(fv, (list, tuple)) or isinstance(mv, (list, tuple)):
        return None
    return same_number(float(fv), float(mv), EXACT_TOL)


def _scalar_fields(result: Any) -> List[str]:
    """The fields of a result a floor can have an opinion on."""
    if result is None:
        return []
    out: List[str] = []
    for name in type(result).model_fields:
        if name in ("kind", "percent_passing", "points", "specimens",
                    "flow_curve", "pl_trials", "water_contents", "rows",
                    "fields", "description"):
            continue
        out.append(name)
    return out


class _LabMerger:
    def __init__(self, floor: Sequence[LabTest], model: Sequence[LabTest],
                 log: MergeLog) -> None:
        self.floor = list(floor)
        self.model = [t.model_copy(deep=True) for t in model]
        self.log = log
        self.paired: set = set()

    @staticmethod
    def _label(test: LabTest) -> str:
        return (f"lab_tests[{test.kind} "
                f"{test.investigation_id or 'no boring'}"
                + (f" {show(test.depth_top)}" if test.depth_top else "")
                + "]")

    @staticmethod
    def _prov0(test: LabTest) -> Provenance:
        if not test.prov:
            test.prov.append(Provenance(page=test.pages[0] if test.pages
                                        else 0, method="model"))
        return test.prov[0]

    def _floor_conf(self, test: LabTest) -> float:
        return max((p.confidence for p in test.prov
                    if p.method in ("tables", "text")), default=0.6)

    def _settle_into(self, target: LabTest, holder: Any, name: str,
                     fv: Any, floor_test: LabTest, where: str,
                     what: str) -> None:
        """Settle one floor value against the model's ``holder.name``."""
        mv = getattr(holder, name, None)
        if mv == "" or mv == []:
            mv = None
        same = _same_value(name, fv, mv)
        mprov = self._prov0(target)
        page = mprov.page
        evidence = has_evidence(mprov, None)
        s = settle(name, fv, mv, same,
                   floor_method="tables",
                   floor_confidence=self._floor_conf(floor_test),
                   model_method=mprov.method if mprov.method in (
                       "model", "model_from_picture", "vision") else "model",
                   model_confidence=float(mprov.confidence),
                   evidence=evidence,
                   floor_unit=fv.unit if isinstance(fv, Quantity) else "",
                   model_unit=mv.unit if isinstance(mv, Quantity) else "",
                   model_note=mprov.note)
        if s.verdict == "reconciled":
            self.log.reconciled += 1
        elif s.verdict == "floor_only":
            self.log.keep(where, f"{what}: {name}", page, fv)
        elif s.verdict in ("floor_wins", "model_wins"):
            self.log.disagree(
                where, f"{what}: {name}", fv, mv,
                kept="model" if s.verdict == "model_wins" else "floor",
                why=("the model named a box and a note, so its reading "
                     "replaces the table's; the table's is kept beside it"
                     if s.verdict == "model_wins" else
                     "the model's reading differs from the table's and it "
                     "named no box or note for it; the table's stands and "
                     "the model's is kept beside it"),
                page=page, floor_method="tables",
                floor_confidence=self._floor_conf(floor_test),
                model_confidence=float(mprov.confidence),
                model_method=mprov.method)
        if s.verdict in ("floor_only", "reconciled", "floor_wins",
                         "model_wins"):
            value = s.value
            try:
                setattr(holder, name, value)
            except Exception:                    # a slot that cannot hold it
                target.fields[name] = show(value)
        if s.alternative is not None:
            mprov.alternatives.append(s.alternative)

    def _merge_series(self, target: LabTest, result: GradationResult,
                      floor_points: Sequence[SievePoint], floor_test: LabTest,
                      where: str) -> None:
        mprov = self._prov0(target)
        evidence = has_evidence(mprov, None)
        for fp in floor_points:
            twin = None
            for mp in result.percent_passing:
                if fold(mp.sieve) and fold(mp.sieve) == fold(fp.sieve):
                    twin = mp
                    break
                if fp.size is not None and mp.size is not None:
                    fs, ms = fp.size.si_value, mp.size.si_value
                    if fs is not None and ms is not None and \
                            abs(fs - ms) <= max(1e-5, 0.02 * abs(fs)):
                        twin = mp
                        break
            what = f"passing {fp.sieve or show(fp.size)}"
            if twin is None:
                result.percent_passing.append(fp.model_copy(deep=True))
                self.log.keep(where, what, mprov.page, fp.percent_passing)
                continue
            same = same_number(fp.percent_passing, twin.percent_passing,
                               PASSING_TOL)
            s = settle("percent_passing", fp.percent_passing,
                       twin.percent_passing, same, floor_method="tables",
                       floor_confidence=self._floor_conf(floor_test),
                       model_method=mprov.method, evidence=evidence,
                       model_confidence=float(mprov.confidence),
                       model_note=mprov.note)
            if s.verdict == "reconciled":
                self.log.reconciled += 1
            elif s.verdict in ("floor_wins", "model_wins"):
                self.log.disagree(
                    where, what, fp.percent_passing, twin.percent_passing,
                    kept="model" if s.verdict == "model_wins" else "floor",
                    why=("the model's series differs from the table's at "
                         "this sieve"), page=mprov.page,
                    floor_method="tables",
                    floor_confidence=self._floor_conf(floor_test),
                    model_confidence=float(mprov.confidence),
                    model_method=mprov.method)
                twin.percent_passing = float(s.value)
                if s.alternative is not None:
                    mprov.alternatives.append(s.alternative)
            if twin.size is None and fp.size is not None:
                twin.size = fp.size.model_copy(deep=True)
        result.percent_passing.sort(
            key=lambda p: -(p.size.si_value or 0.0) if p.size else 0.0)

    def _merge_rows(self, target: LabTest, result: SummaryTableResult,
                    floor_rows: Sequence[SummaryRow], floor_test: LabTest,
                    where: str) -> None:
        mprov = self._prov0(target)
        for fr in floor_rows:
            twin = None
            for mr in result.rows:
                if fold(fr.investigation_id) != fold(mr.investigation_id):
                    continue
                if fr.sample_id and mr.sample_id and \
                        fold(fr.sample_id) == fold(mr.sample_id):
                    twin = mr
                    break
                if same_depth(fr.depth_top, mr.depth_top):
                    twin = mr
                    break
            label = f"row {fr.investigation_id} {fr.sample_id} " \
                    f"{show(fr.depth_top)}".strip()
            if twin is None:
                result.rows.append(fr.model_copy(deep=True))
                self.log.keep(where, label, mprov.page)
                continue
            for name in type(fr).model_fields:
                if name in ("percent_passing", "other", "investigation_id",
                            "sample_id"):
                    continue
                fv = getattr(fr, name)
                if fv in (None, "", []):
                    continue
                self._settle_into(target, twin, name, fv, floor_test, where,
                                  label)
            for fp in fr.percent_passing:
                match = next((mp for mp in twin.percent_passing
                              if fold(mp.sieve) == fold(fp.sieve)), None)
                if match is None:
                    twin.percent_passing.append(fp.model_copy(deep=True))
                    self.log.keep(where, f"{label}: passing {fp.sieve}",
                                  mprov.page, fp.percent_passing)
                elif not same_number(fp.percent_passing,
                                     match.percent_passing, PASSING_TOL):
                    self.log.disagree(
                        where, f"{label}: passing {fp.sieve}",
                        fp.percent_passing, match.percent_passing,
                        kept="floor", why="the model's row differs from the "
                                          "table's at this sieve",
                        page=mprov.page, floor_method="tables",
                        floor_confidence=self._floor_conf(floor_test),
                        model_confidence=float(mprov.confidence),
                        model_method=mprov.method)
                    mprov.alternatives.append(Alternative(
                        field=f"{label}: passing {fp.sieve}",
                        value=show(match.percent_passing),
                        method=mprov.method,
                        confidence=float(mprov.confidence)))
                    match.percent_passing = fp.percent_passing
                else:
                    self.log.reconciled += 1
            for head, text in fr.other:
                if not any(fold(h) == fold(head) for h, _t in twin.other):
                    twin.other.append((head, text))

    def merge_pair(self, f: LabTest, m: LabTest) -> None:
        where = self._label(m)
        what = f.kind
        mprov = self._prov0(m)
        # The link, then the values.
        for name in ("investigation_id", "sample_id", "depth_top",
                     "depth_bottom"):
            fv = getattr(f, name)
            if fv in (None, ""):
                continue
            self._settle_into(m, m, name, fv, f, where, "link")
        if f.result is None:
            pass
        elif m.result is None:
            m.result = f.result.model_copy(deep=True)
            self.log.keep(where, f"{what}: every value", mprov.page)
        elif isinstance(f.result, SummaryTableResult) and \
                isinstance(m.result, SummaryTableResult):
            self._merge_rows(m, m.result, f.result.rows, f, where)
        else:
            for name in _scalar_fields(f.result):
                fv = getattr(f.result, name, None)
                if fv in (None, "", False):
                    continue
                if not hasattr(m.result, name):
                    m.fields.setdefault(name, show(fv))
                    self.log.keep(where, f"{what}: {name}", mprov.page, fv)
                    continue
                self._settle_into(m, m.result, name, fv, f, where, what)
            if isinstance(f.result, GradationResult) and \
                    isinstance(m.result, GradationResult):
                self._merge_series(m, m.result, f.result.percent_passing,
                                   f, where)
        for p in f.prov:
            if p not in m.prov:
                m.prov.append(p)

    def place_values(self, f: LabTest) -> Optional[LabTest]:
        """A floor test with no twin: its values find homes, or it stays."""
        candidates = [m for m in self.model if _same_specimen(f, m)]
        if f.result is None or isinstance(f.result, (SummaryTableResult,
                                                     OtherResult)):
            return f.model_copy(deep=True) if not any(
                m.kind == f.kind for m in candidates) else None
        homeless: Dict[str, Any] = {}
        homed = 0
        for name in _scalar_fields(f.result):
            fv = getattr(f.result, name, None)
            if fv in (None, "", False):
                continue
            home = next((m for m in candidates
                         if m.result is not None and hasattr(m.result, name)),
                        None)
            if home is None:
                homeless[name] = fv
                continue
            self._settle_into(home, home.result, name, fv, f,
                              self._label(home), f.kind)
            homed += 1
        series = (f.result.percent_passing
                  if isinstance(f.result, GradationResult) else [])
        if series:
            home = next((m for m in candidates
                         if isinstance(m.result, GradationResult)), None)
            if home is not None:
                self._merge_series(home, home.result, series, f,
                                   self._label(home))
                homed += 1
            else:
                homeless["percent_passing"] = series
        if not homeless:
            return None
        if homed == 0:
            return f.model_copy(deep=True)
        residue = f.model_copy(deep=True)
        cls = type(f.result)
        kept = {name: value for name, value in homeless.items()
                if name != "percent_passing"}
        try:
            residue.result = cls(**kept, **({"percent_passing": homeless[
                "percent_passing"]} if "percent_passing" in homeless else {}))
        except Exception:
            residue.result = f.result
        return residue

    def run(self) -> List[LabTest]:
        out: List[LabTest] = list(self.model)
        extra: List[LabTest] = []
        for f in self.floor:
            twin = None
            for i, m in enumerate(self.model):
                if i in self.paired or m.kind != f.kind:
                    continue
                if _same_specimen(f, m):
                    twin = i
                    break
            if twin is not None:
                self.paired.add(twin)
                self.merge_pair(f, self.model[twin])
                continue
            kept = self.place_values(f)
            if kept is not None:
                page = kept.prov[0].page if kept.prov else None
                if any(_same_specimen(f, m) for m in self.model) and \
                        f.result is None or (
                            isinstance(f.result, OtherResult)
                            and f.result.no_results):
                    others = sorted({m.kind for m in self.model
                                     if _same_specimen(f, m)})
                    if others and f.kind not in others:
                        self.log.disagree(
                            self._label(f), "kind", f.kind, ", ".join(others),
                            kept="both",
                            why="the sheet's title names one kind and the "
                                "model read another; both are in the record",
                            page=page, floor_method="text",
                            floor_confidence=self._floor_conf(f),
                            model_confidence=0.9)
                self.log.keep(self._label(f), f"{f.kind} test", page)
                extra.append(kept)
        for i, m in enumerate(self.model):
            if i not in self.paired:
                self.log.add(self._label(m), f"{m.kind} test",
                             m.prov[0].page if m.prov else None,
                             method=m.prov[0].method if m.prov else "model")
        return out + extra


def merge_lab_tests(floor: Sequence[LabTest], model: Sequence[LabTest],
                    log: Optional[MergeLog] = None
                    ) -> Tuple[List[LabTest], MergeLog]:
    """The tables' tests and the model's as one list, and what the merge did.

    The model's tests come first, in the model's own order, each merged
    with the floor test of the same kind on the same specimen where there
    is one; floor tests with no twin follow, whole where none of their
    values found a home and as a residue where some did.
    """
    log = log if log is not None else MergeLog()
    merged = _LabMerger(floor, model, log).run()
    return merged, log
