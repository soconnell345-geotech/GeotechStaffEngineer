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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from report_ingest.floor import (
    EXACT_TOL, MergeLog, fold, has_evidence, same_depth, same_number,
    same_text, settle, show,
)
from report_ingest.model import (
    Alternative, AtterbergResult, CBRResult, ChemicalResult, CompactionResult,
    ConsolidationResult, GradationResult, LabTest, MoistureDensityResult,
    OtherResult, Provenance, Quantity, RESULT_CLASS, SievePoint,
    StrengthResult, SummaryRow, SummaryTableResult, UNIT_TO_SI,
    _canonical_unit,
)
from report_ingest.turned import (
    ROW_TOL, Turn, line_thickness as _row_thickness, page_turn,
    reading_order, top_band, upright_box,
)

__all__ = ["FloorSheet", "floor_from_tables", "serialise_floor",
           "merge_lab_tests", "kind_from_title", "TITLE_KINDS",
           "PASSING_TOL", "title_kinds", "page_title_kinds",
           "legend_specimens", "LEGEND_NOTE"]

#: A grading's percent-passing values agree within one percentage point
#: (``lab_scoring.PASSING_TOL``).
PASSING_TOL = 1.0
#: How sure the floor is of a value read off a detected table: a ruled table
#: on a vector text layer is exact; one read optically is not.
CONF_TABLE = 0.85
CONF_TABLE_OCR = 0.6
CONF_TITLE = 0.75
CONF_LINK = 0.7
#: A value read off a legend row (:func:`legend_specimens`): printed beside
#: its label on one line, read from the text layer or the optical text.
CONF_LEGEND = 0.75
#: The top share of a page, as it reads, where the floor looks for the
#: sheet's title.
TITLE_BAND = 0.22

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

#: What a laboratory COMPACTION sheet prints that a density or water content
#: test does not: the test's own names (a Proctor, a moisture-density or
#: dry density and water content relationship) and the parts the DIGGS
#: writer records for one (``LabCompactionTest``: optimum water content,
#: rammer, blows per layer). Any one of these is enough. See
#: :func:`kind_from_title`.
COMPACTION_PARTS = (
    r"proctor|(?:moisture|water)[- ]density relation|"
    r"density ?/ ?(?:moisture|water) content relation|"
    r"optimum (?:moisture|water)|\bomc\b|rammer|blows? per layer")
#: Parts other density tests print too: a mould (a unit weight, a maximum
#: index density), the word compaction (a field test's relative
#: compaction), a maximum dry density (quoted by a field test, or a relative
#: density's maximum). None counts on its own; two different ones do.
COMPACTION_WEAK_PARTS: Tuple[str, ...] = (
    r"\bmou?ld\b", r"\bcompact(?:ion|ed)?\b",
    r"max(?:imum|\.)? dry (?:density|unit weight)|\bmdd\b")
#: Density tests that are not a laboratory compaction, though they quote
#: one (its maximum dry density, the relative compaction it gives): a sheet
#: naming any of these at its top keeps its kind: the field methods by name,
#: and a compaction given as a percent ("percent compaction", "degree of
#: compaction", "Compaction (%)"), which is a specimen measured against a
#: compaction; a compaction never reports that of itself.
NOT_COMPACTION = (r"\bfield\b|in[- ]place|in[- ]?situ|nuclear|sand[- ]cone|"
                  r"drive[- ](?:tube|cylinder)|core[- ]cutter|balloon|"
                  r"relative density|index density|"
                  r"(?:percent|per cent|degree of|relative)\s+compaction|"
                  r"compaction\s*\(?\s*%|%\s*compaction")
#: The kinds that report one measurement "and nothing else"
#: (``lab_reader.KIND_DEFINITIONS``): a compaction sheet prints both.
_ONE_MEASUREMENT = ("density", "moisture_content")
_CBR_TITLE = r"\bcbr\b|bearing ratio"
#: A title naming the density of the solid particles: that is the specific
#: gravity of solids, which "is a measure of solid particle density and is
#: referenced to an equivalent volume of water", printed as Gs = ρs / ρd
#: with ρd "the mass density of water" (FHWA-NHI-06-088, Soils and
#: Foundations Reference Manual Vol. I, 5.3.6, p. 5-19), as
#: :data:`TITLE_KINDS` already reads "densite des grains". A title that also
#: names a bulk, dry or wet density reports those too and keeps its kind.
_PARTICLE_DENSITY = r"particle density"
_OTHER_DENSITY = (r"\b(?:bulk|dry|wet|moist|total|field|in[- ]?situ|"
                  r"in[- ]place)\b")

#: A heading for the share of a specimen that is dry solids -- percent
#: solids (SM 2540 G), total solids, dry matter -- which a chemistry lab
#: prints beside its results and which is no water content: a specimen of
#: 85 % solids holds about 18 % water by dry weight (RI-143). Not the
#: specific gravity, density or mass of the solids, nor dissolved,
#: suspended, volatile, fixed or settleable solids (a water analysis's
#: concentrations and fractions).
SOLIDS_HEADING = (r"(?:%|\bpercent|\bper cent|\bpct\.?) ?(?:total )?solids\b|"
                  r"\btotal solids\b|"
                  r"(?<!dissolved )(?<!suspended )(?<!volatile )(?<!fixed )"
                  r"(?<!settleable )"
                  r"\bsolids,? ?(?:content|\( ?% ?\)|%|\[ ?% ?\])|"
                  r"^solids$|\bdry matter\b|\bdry residue\b|"
                  r"\bmati[eè]re s[eè]che\b|\bresiduo seco\b")
#: A water content a test was run AT, not the specimen's own: the one a
#: resistivity test reached its minimum at (a heading naming minimum,
#: resistivity and water content, in any order) (RI-143).
_TEST_WATER_CONTENT = (r"(?=.*\bmin(?:imum|\.)?\b)(?=.*resistivit)"
                       r"(?=.*(?:water|moisture) content)")
#: The percent solids S at which a specimen's own water content w, in
#: percent, is the same number as S. With water mass Mw, solids mass Ms and
#: total mass M = Mw + Ms, percent solids is S = 100 Ms / M; a water content
#: on a wet basis is 100 Mw / M = 100 - S, equal to S only at S = 50; on a
#: dry basis it is 100 Mw / Ms = 100 (100 - S) / S, equal to S only where
#: S^2 + 100 S - 10000 = 0, at S = 50 (sqrt 5 - 1) = 61.80.
SOLIDS_EQUAL_TO_WATER = (50.0, 50.0 * (5.0 ** 0.5 - 1.0))


def names_solids(label: Any) -> bool:
    """Whether a printed heading names a share of dry solids
    (:data:`SOLIDS_HEADING`), folded as :func:`_field_of` folds one."""
    return bool(re.search(SOLIDS_HEADING, _fold_label(label)))


def solids_can_equal_water(printed: Any) -> bool:
    """Whether a percent solids, as printed, can be the same number as the
    same specimen's water content (RI-143).

    It can only near the two values of :data:`SOLIDS_EQUAL_TO_WATER`: 50 on
    a wet basis, 61.80 on a dry basis. A printed number stands for any value
    within half its last digit, and both water contents fall as S rises
    while S itself rises, so the two print alike only where one of those
    values lies within half a last digit of the number: "50" or "61.8" may
    be both, "73.7" may not -- a specimen of 73.7 % solids holds 26.3 %
    water on a wet basis and 35.7 % on a dry one."""
    raw = str(printed or "").strip()
    value = _num(raw)
    if value is None:
        return False
    digits = _NUMBER.search(raw.replace(" ", ""))
    decimals = re.split(r"[.,]", digits.group(0))[1:] if digits else []
    half = 0.5 * 10.0 ** -(len(decimals[0]) if decimals else 0)
    return any(abs(value - equal) <= half + 1e-9
               for equal in SOLIDS_EQUAL_TO_WATER)


#: A printed label -> the record's own name for the value. Applied to a
#: row's first cell in a property/value table and to a column head in a
#: header-row table, after folding to lower case and one space. The first
#: pattern that matches wins, so the longer names come first. A share of
#: dry solids and the water content a resistivity test ran at are kept as
#: printed (:data:`_AS_PRINTED`), never read as the water content.
LABEL_FIELDS: Tuple[Tuple[str, str], ...] = (
    (SOLIDS_HEADING + "|" + _TEST_WATER_CONTENT, "as_printed"),
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
#: A hole named beside its field, the name bare or in brackets ("BORE HOLE
#: NO. (B-2)", RI-149).
_LINK_BORING = re.compile(
    r"(?:boring|borehole|bore ?hole|test ?pit|sondage|sondeo|exploration|"
    r"hole|location)\s*(?:no\.?|number|#|id)?\s*[:.\-]?\s*\(?\s*"
    r"([A-Za-z]{1,4}[- ]?\d+[A-Za-z]?)\b", re.I)
_LINK_SAMPLE = re.compile(
    r"(?:sample|specimen|muestra|echantillon)\s*(?:no\.?|number|#|id)?\s*"
    r"[:.\-]?\s*([A-Za-z]*[- ]?\d+[A-Za-z]?)\b", re.I)
_LINK_DEPTH = re.compile(
    r"(?:depth|profondeur|profundidad|prof\.?)\s*[:.\-]?\s*"
    r"(\d+(?:[.,]\d+)?)\s*(?:(?:-|–|to|a)\s*(\d+(?:[.,]\d+)?))?\s*"
    r"(ft|feet|foot|m|metres|meters|'|\")?", re.I)

#: THE LEGEND OF SEVERAL SPECIMENS (RI-149). A grading chart that plots
#: several samples prints a legend or key with one row per sample: the
#: sample's label (its depth, sometimes with its hole), its description and
#: its percentages ("SAND = 53%, Gravel = 18%"), along one printed row.
#: Which numbers belong to which sample is which row they are printed on,
#: and nothing else says so. The floor reads such a legend in the text's
#: own frame (:mod:`report_ingest.turned`, de-skewed on a page scanned
#: sideways) and binds each row's values to that row's label and no other
#: (:func:`legend_specimens`); the model's answer is merged onto them
#: (:func:`merge_lab_tests` lets a legend value be overruled only by a
#: reading boxed on the line that prints it). Read only where at least
#: :data:`LEGEND_MIN_ROWS` entries each print a label and at least two
#: labelled values: a sheet of one specimen is left as it was.
#:
#: The note every legend value's provenance carries; the merge knows a
#: legend test by it.
LEGEND_NOTE = ("read off a legend row: the specimen's label and its values "
               "printed on one line")
LEGEND_MIN_ROWS = 2
#: A grading share named and printed with its percent: "SAND = 53%",
#: "Gravel: 18 %", "Silt 39%".
_LEGEND_SHARE = re.compile(
    r"\b(cobbles?|gravel|sand|silt\s*(?:&|and|/|\+)\s*clay|fines|silt|clay)"
    r"\s*[=:]?\s*(\d+(?:[.,]\d+)?)\s*%", re.I)
#: A diameter or a coefficient printed with its sign: "D60 = 0.42 mm",
#: "Cu = 12", "Cc: 1.1".
_LEGEND_SIZE = re.compile(
    r"\b(d\s?(?:10|30|50|60|85|90|100)|cu|cc)\s*[=:]\s*(\d+(?:[.,]\d+)?)"
    r"\s*(mm|in)?\b", re.I)
#: A legend row's label, at the row's start: an optional hole or sample
#: code, then a depth range ("4.2-5.7", "13.-14.5", "2.3 to 3.8 m") or one
#: depth with its unit or after an at sign ("@ 9 ft", "13.85 m"). A bare
#: number is a symbol's or a sample's number, not a depth, and is no label.
_LEGEND_LABEL = re.compile(
    r"^[^\w@]*(?:(?P<code>[A-Za-z]{1,4}[- ]?\d+[A-Za-z]?)[\s,;:/]+)?"
    r"(?P<at>@\s*)?(?P<top>\d+(?:[.,]\d*)?)\s*"
    r"(?P<unit1>m|ft|feet|metres|meters|')?"
    r"(?:\s*(?:-|–|—|to)\s*(?P<bottom>\d+(?:[.,]\d*)?)\s*"
    r"(?P<unit2>m|ft|feet|metres|meters|')?)?(?![\w.])", re.I)
_LEGEND_FIELD = {"cobble": "cobbles_percent", "cobbles": "cobbles_percent",
                 "gravel": "gravel_percent", "sand": "sand_percent",
                 "silt": "silt_percent", "clay": "clay_percent",
                 "fines": "fines_percent", "cu": "cu", "cc": "cc"}


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


def _fold_label(label: Any) -> str:
    """A printed label folded as :data:`LABEL_FIELDS` is matched against:
    lower case, one space, a space before a bracket and round a percent."""
    text = " ".join(str(label or "").lower().replace("_", " ").split())
    text = text.replace("(", " (").replace("%", " % ")
    return " ".join(text.split())


def _field_of(label: str) -> str:
    text = _fold_label(label)
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


def kind_from_title(lines: Sequence[str],
                    corrected: bool = True) -> Tuple[str, str]:
    """``(kind, the title line)`` from the top of the page, or ``("", "")``.

    Line by line, each line against :data:`TITLE_KINDS` in order. Then the
    lab module's own kinds correct a title that names only a density or a
    water content (RI-56): a particle density is a specific gravity
    (:data:`_PARTICLE_DENSITY`); and those tests report that one
    measurement and nothing else, so where the top of the page also names a
    compaction's own part (:data:`COMPACTION_PARTS`), or two different parts
    other density tests print too (:data:`COMPACTION_WEAK_PARTS`), and no
    CBR and no field or index density test (:data:`NOT_COMPACTION`), the
    sheet is a compaction, and the first line that names a part is the
    title.
    ``corrected=False`` is the rule before RI-56, which the model's brief
    shows unless ``GEOTECH_INGEST_LAB_FLOOR_PER_PAGE`` is on
    (:attr:`FloorSheet.as_before`).
    """
    kind, title = "", ""
    texts = [" ".join(str(line or "").lower().split()) for line in lines]
    for line, text in zip(lines, texts):
        kind = next((name for pattern, name in TITLE_KINDS
                     if re.search(pattern, text)), "")
        if kind:
            title = str(line).strip()
            break
    if not corrected:
        return kind, title
    folded = " ".join(title.lower().split())
    if kind == "density" and re.search(_PARTICLE_DENSITY, folded) and \
            not re.search(_OTHER_DENSITY, folded):
        return "specific_gravity", title
    if kind not in _ONE_MEASUREMENT or any(
            re.search(_CBR_TITLE, text) or re.search(NOT_COMPACTION, text)
            for text in texts):
        return kind, title
    weak = {i for text in texts
            for i, part in enumerate(COMPACTION_WEAK_PARTS)
            if re.search(part, text)}
    if not any(re.search(COMPACTION_PARTS, text) for text in texts) and \
            len(weak) < 2:
        return kind, title
    for line, text in zip(lines, texts):
        if re.search(COMPACTION_PARTS, text) or any(
                re.search(part, text) for part in COMPACTION_WEAK_PARTS):
            return "compaction", str(line).strip()
    return kind, title


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
    #: The pages whose tables gave values under no boring and no depth of
    #: their own while another page of the sheet prints one (RI-56): those
    #: values are left unbound, never given the other page's link. Each is
    #: ``{"page", "values", "linked"}``, ``linked`` naming the pages that
    #: print a link.
    unbound: List[Dict[str, Any]] = field(default_factory=list)
    #: The floor as it was before RI-56: the first page's link on every
    #: page's values and the title's kind by line order alone. The model's
    #: brief shows it (:func:`serialise_floor`) unless
    #: ``GEOTECH_INGEST_LAB_FLOOR_PER_PAGE`` is on; the merge never uses it.
    #: None on that floor itself.
    as_before: Optional["FloorSheet"] = None


def _lines_of(doc: Any, page: int) -> List[Any]:
    """A page's lines in the order it reads: top to bottom and left to
    right as displayed, or, on a page whose text runs a quarter turn (a
    sheet scanned sideways), row by row in the turned page's own frame
    (:func:`report_ingest.turned.reading_order`, RI-149)."""
    try:
        content = doc.page(page)
    except Exception:
        return []
    return reading_order(content.lines, _turn_of_content(content))


def _turn_of_content(content: Any) -> Turn:
    return page_turn(content.lines,
                     float(getattr(content, "width", 612.0) or 612.0),
                     float(getattr(content, "height", 792.0) or 792.0))


def _page_turn(doc: Any, page: int) -> Turn:
    try:
        return _turn_of_content(doc.page(page))
    except Exception:
        return Turn()


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


def _depth_unit(text: str) -> str:
    text = str(text or "").lower()
    return ("ft" if text in ("ft", "feet", "foot", "'")
            else "m" if text in ("m", "metres", "meters") else "")


def _legend_field(name: str) -> str:
    folded = " ".join(str(name or "").lower().split())
    if re.match(r"silt\s*(?:&|and|/|\+)\s*clay", folded):
        return "fines_percent"
    folded = folded.replace(" ", "")
    if re.fullmatch(r"d(?:10|30|50|60|85|90|100)", folded):
        return folded
    return _LEGEND_FIELD.get(folded, "")


def _union(boxes: Sequence[Sequence[float]]) -> Tuple[float, float, float,
                                                       float]:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


@dataclass
class _LegendRow:
    """One legend entry read: the specimen's link and its labelled values,
    with the label's box and the box of the line(s) printing the values."""

    link: _Link
    values: Dict[str, Tuple[Any, str]]
    label_bbox: Tuple[float, float, float, float]
    values_bbox: Tuple[float, float, float, float]


def _label_of(text: str, page: int, sheet: _Link
              ) -> Optional[Tuple[_Link, int]]:
    """A specimen label at the start of a line, and where it ends, or None:
    a depth range, or one depth with its unit or after an at sign, with an
    optional hole or sample code before it (:data:`_LEGEND_LABEL`)."""
    label = _LEGEND_LABEL.match(str(text or ""))
    if label is None:
        return None
    if not (label.group("bottom") or label.group("at")
            or label.group("unit1") or label.group("unit2")):
        return None
    top = _num(label.group("top").rstrip(".,"))
    if top is None:
        return None
    unit = (_depth_unit(label.group("unit2") or label.group("unit1") or "")
            or sheet.unit)
    link = _Link(investigation_id=sheet.investigation_id, unit=unit,
                 page=page, depth_top=top,
                 depth_bottom=(_num(label.group("bottom").rstrip(".,"))
                               if label.group("bottom") else None))
    code = (label.group("code") or "").strip()
    if code:
        # A code before the depth: the sheet's own hole printed again on the
        # row says nothing more; a sample's series (S-1, ST-2) names the
        # row's sample; any other names the row's own hole, which may be
        # another than the one the sheet's header prints.
        series = re.match(r"[A-Za-z]+", code)
        if link.investigation_id and \
                _hole_fold(code) == _hole_fold(link.investigation_id):
            pass
        elif series is not None and series.group(0).upper() in \
                _SAMPLE_SERIES:
            link.sample_id = code
        else:
            link.investigation_id = code.replace(" ", "-")
    return link, label.end()


#: The series letters that name a sample, not a hole, where a legend row
#: prints a code before its depth: split spoon (S, SS, SPT), tube (ST, SH,
#: TW, U, UD), disturbed (D, DS), core (C, CS, R, RC, RUN), grab, bag and
#: block samples (G, GS, BAG, BLK). LAYOUT JUDGMENT, as
#: :data:`report_ingest.log_holes._NOT_ALONE` keeps them from naming a
#: hole.
_SAMPLE_SERIES = frozenset((
    "S", "SS", "SPT", "ST", "SH", "TW", "U", "UD", "D", "DS", "C", "CS",
    "R", "RC", "RUN", "G", "GS", "BAG", "BLK"))


def _hole_fold(text: Any) -> str:
    """A hole's code folded for comparison: lower case, letters and digits
    only, a number's leading zeros dropped ("QX-04" is "qx4")."""
    folded = re.sub(r"[^0-9a-z]+", "", str(text or "").lower())
    return re.sub(r"(?<![0-9])0+(?=[0-9])", "", folded)


def _values_in(text: str) -> Optional[Dict[str, Tuple[Any, str]]]:
    """The labelled grading values a piece of text prints ("SAND = 53%",
    "D60 = 0.42 mm"), or None where one is named twice (two entries'
    values run together)."""
    values: Dict[str, Tuple[Any, str]] = {}
    for match in _LEGEND_SHARE.finditer(text):
        name = _legend_field(match.group(1))
        if not name:
            continue
        if name in values:
            return None
        values[name] = (_num(match.group(2)), "%")
    for match in _LEGEND_SIZE.finditer(text):
        name = _legend_field(match.group(1))
        if not name:
            continue
        if name in values:
            return None
        unit = "" if name in ("cu", "cc") else (match.group(3) or "mm")
        values[name] = (_num(match.group(2)), unit)
    return {k: v for k, v in values.items() if v[0] is not None}


#: How far apart, as a share of the legend's row pitch, a label and the
#: values bound to it by nearness may lie, and by how much a second label
#: must be further off for the binding to be called unambiguous.
LEGEND_NEAR = 0.3


def _bind_by_rank(labels: Sequence[float], values: Sequence[float],
                  pitch: float) -> Optional[List[int]]:
    """Each value row's label by order, where there are as many value rows
    as labels and every pair sits the same way off its label: the rows of
    one legend are parallel, so a scan's skew moves every row's values the
    same distance off its label, and that distance may be half a row."""
    if len(labels) != len(values) or len(labels) < LEGEND_MIN_ROWS:
        return None
    offsets = [v - lab for lab, v in zip(labels, values)]
    if max(offsets) - min(offsets) >= 0.5 * pitch:
        return None
    # Off by most of a row, the order could as well be one row out (a stray
    # label above the column and one missing below it).
    if max(abs(o) for o in offsets) >= 0.75 * pitch:
        return None
    return list(range(len(values)))


def _bind_by_nearness(labels: Sequence[float], values: Sequence[float],
                      pitch: float) -> Optional[List[Optional[int]]]:
    """Each value row's label by nearness across the rows: the nearest
    label where it lies within :data:`LEGEND_NEAR` of the pitch and every
    other is further off by as much again; None for a value row with no
    label that near (its label was not read: it makes no entry). The whole
    answer is None where a value row lies near two labels alike."""
    out: List[Optional[int]] = []
    for v in values:
        ranked = sorted(range(len(labels)), key=lambda i: abs(labels[i] - v))
        d1 = abs(labels[ranked[0]] - v)
        d2 = abs(labels[ranked[1]] - v) if len(ranked) > 1 else float("inf")
        if d1 > LEGEND_NEAR * pitch:
            out.append(None)
            continue
        if d2 - d1 < LEGEND_NEAR * pitch:
            return None
        out.append(ranked[0])
    return out


def _label_column(labels: List[Tuple[float, _Link, Any]], turn: Turn,
                  tol: float) -> List[Tuple[float, _Link, Any]]:
    """The labels that stand in the legend's column of labels.

    A legend prints its labels one under another, starting (or ending) at
    one place along the row; a line of the key's symbols read optically
    ("-6-9-" for a dashed line with its markers) also opens with something
    shaped like a depth range, but it stands in another column. Where at
    least :data:`LEGEND_MIN_ROWS` labels share a start (or an end) within
    four times ``tol``, only those are labels."""
    if len(labels) <= LEGEND_MIN_ROWS:
        return labels
    best: List[Tuple[float, _Link, Any]] = labels
    best_n = 0
    for side in (0, 2):
        at = [turn.frame(item[2].bbox)[side] for item in labels]
        for here in at:
            kept = [item for item, there in zip(labels, at)
                    if abs(there - here) <= 4.0 * tol]
            if len(kept) > best_n:
                best, best_n = kept, len(kept)
    return best if best_n >= LEGEND_MIN_ROWS else labels


def legend_specimens(doc: Any, page: int, sheet: Optional[_Link] = None
                     ) -> List[_LegendRow]:
    """The specimens a page's legend prints, each with its own values
    (RI-149), or an empty list.

    A legend row is a specimen's label at the start of a line (its depth,
    perhaps with its hole) and, on that row, its labelled grading values,
    printed on the same line or on others further along the row. Rows are
    taken in the text's own frame (:mod:`report_ingest.turned`, de-skewed),
    and each value row is bound to a label:

    * on one line with its label: to that label;
    * else by ORDER, where the legend has as many value rows as labels and
      every value row sits the same way off its label (a skew moves all of
      them alike);
    * else by NEARNESS, where each value row is clearly nearer one label
      than any other (:data:`LEGEND_NEAR`).

    Where neither binds every value row the page's legend is not read at
    all: the floor then says nothing about it rather than guess, and the
    model's reading stands. A label with no values, and values with no
    label, make no entry. Empty unless at least :data:`LEGEND_MIN_ROWS`
    entries are read. ``sheet`` is the page's own link: its hole and depth
    unit go to entries that print none."""
    try:
        content = doc.page(page)
    except Exception:
        return []
    if sheet is None:
        sheet = _link_from_lines(_lines_of(doc, page), page)
    turn = _turn_of_content(content)
    whole: List[_LegendRow] = []
    labels: List[Tuple[float, _Link, Any]] = []
    loose: List[Tuple[float, Dict[str, Tuple[Any, str]], Any]] = []
    for line in content.lines:
        text = str(getattr(line, "text", "") or "")
        if not text.strip() or not getattr(line, "bbox", None):
            continue
        got = _label_of(text, page, sheet)
        rest = text[got[1]:] if got else text
        values = _values_in(rest)
        if values is None:
            return []                       # two entries run together
        v = turn.centre_v(line.bbox)
        if got is not None and len(values) >= 2:
            link = got[0]
            link.bbox = tuple(line.bbox)
            whole.append(_LegendRow(link=link, values=values,
                                    label_bbox=tuple(line.bbox),
                                    values_bbox=tuple(line.bbox)))
        elif got is not None:
            labels.append((v, got[0], line))
        elif values:
            loose.append((v, values, line))
    # The values a row prints over more than one line, one value row.
    tol = ROW_TOL * _row_thickness(content.lines, turn)
    rows: List[List[Tuple[float, Dict[str, Tuple[Any, str]], Any]]] = []
    for item in sorted(loose, key=lambda it: it[0]):
        if rows and abs(item[0] - rows[-1][-1][0]) <= tol and not (
                set(item[1]) & {n for _v, vals, _l in rows[-1] for n in vals}):
            rows[-1].append(item)
        else:
            rows.append([item])
    rows = [row for row in rows
            if sum(len(vals) for _v, vals, _l in row) >= 2]
    labels = _label_column(labels, turn, tol)
    labels.sort(key=lambda it: it[0])
    out = list(whole)
    if rows and labels:
        lv = [it[0] for it in labels]
        rv = [sum(v for v, _vals, _l in row) / len(row) for row in rows]
        gaps = sorted(b - a for a, b in zip(lv, lv[1:]) if b - a > tol)
        pitch = gaps[len(gaps) // 2] if gaps else 4.0 * tol
        bound: Optional[List[Optional[int]]] = _bind_by_rank(lv, rv, pitch)
        if bound is None:
            bound = _bind_by_nearness(lv, rv, pitch)
        named = [i for i in bound or () if i is not None]
        if bound is None or len(set(named)) != len(named):
            return []
        for row, i in zip(rows, bound):
            if i is None:
                continue
            link = labels[i][1]
            link.bbox = tuple(labels[i][2].bbox)
            values: Dict[str, Tuple[Any, str]] = {}
            for _v, vals, _l in row:
                values.update(vals)
            out.append(_LegendRow(
                link=link, values=values,
                label_bbox=tuple(labels[i][2].bbox),
                values_bbox=_union([tuple(l.bbox) for _v, _vals, l in row])))
    return out if len(out) >= LEGEND_MIN_ROWS else []


#: The note a legend test's label carries.
LEGEND_LABEL_NOTE = "the specimen's label printed at the start of its legend row"


def _legend_test(row: _LegendRow, report_id: str) -> LabTest:
    """One legend entry as a grading test: the values its row prints, under
    the label that row prints, nothing computed (no fines from silt and
    clay, no Cu from the diameters)."""
    link = row.link
    fields: Dict[str, Any] = {}
    for name, (value, unit) in row.values.items():
        if name.startswith("d"):
            fields[name] = Quantity(value=float(value), unit=unit or "mm")
        else:
            fields[name] = float(value)
    return LabTest(
        kind="gradation", investigation_id=link.investigation_id,
        sample_id=link.sample_id,
        depth_top=Quantity(value=link.depth_top, unit=link.unit),
        depth_bottom=(Quantity(value=link.depth_bottom, unit=link.unit)
                      if link.depth_bottom is not None else None),
        pages=[link.page], source_report=report_id,
        result=GradationResult(**fields),
        prov=[Provenance(page=link.page, bbox=row.label_bbox, method="text",
                         confidence=CONF_LINK, note=LEGEND_LABEL_NOTE),
              Provenance(page=link.page, bbox=row.values_bbox, method="text",
                         confidence=CONF_LEGEND, note=LEGEND_NOTE)])


def _legend_box(test: LabTest) -> Optional[Tuple[int, Tuple[float, ...]]]:
    """``(page, box)`` of the line(s) a legend test's values are printed on,
    or None for a test not read off a legend."""
    for p in test.prov:
        if p.note == LEGEND_NOTE and p.bbox is not None:
            return int(p.page), tuple(float(v) for v in p.bbox)
    return None


def _legend_boxes(test: LabTest
                  ) -> Optional[Tuple[int, Optional[Tuple[float, ...]],
                                      Tuple[float, ...]]]:
    """``(page, label box, values box)`` of a legend test's row: the line
    printing its label and the line(s) printing its values (one line where
    the row prints both), or None for a test not read off a legend."""
    values = _legend_box(test)
    if values is None:
        return None
    label = next((tuple(float(v) for v in p.bbox) for p in test.prov
                  if p.note == LEGEND_LABEL_NOTE and p.bbox is not None
                  and int(p.page) == values[0]), None)
    return values[0], label, values[1]


#: How much of the smaller of two boxes must lie in the other for a reading
#: boxed there to be ON a legend row's line: the displayed boxes of a skewed
#: sheet's neighbouring rows overlap at their edges.
ON_LINE_SHARE = 0.5


def _on_line(box: Sequence[float], line: Sequence[float]) -> bool:
    """Whether ``box`` lies on ``line``: at least :data:`ON_LINE_SHARE` of
    the smaller of the two inside the other."""
    w = min(box[2], line[2]) - max(box[0], line[0])
    h = min(box[3], line[3]) - max(box[1], line[1])
    if w <= 0 or h <= 0:
        return False
    smaller = min((box[2] - box[0]) * (box[3] - box[1]),
                  (line[2] - line[0]) * (line[3] - line[1]))
    return smaller > 0 and w * h >= ON_LINE_SHARE * smaller


def title_kinds(lines: Sequence[str]) -> Tuple[str, ...]:
    """Every kind :data:`TITLE_KINDS` names in the first of ``lines`` that
    names any (the line :func:`kind_from_title` takes as the title), in the
    table's order; empty when none does. A title naming two tests ("GRAIN
    SIZE DISTRIBUTION & ATTERBERG LIMITS TEST RESULTS", a table of both)
    names another sheet than one naming one of them."""
    for line in lines:
        text = " ".join(str(line or "").lower().split())
        named = tuple(name for pattern, name in TITLE_KINDS
                      if re.search(pattern, text))
        if named:
            return tuple(dict.fromkeys(named))
    return ()


def page_title_kinds(doc: Any, page: int) -> Tuple[str, ...]:
    """:func:`title_kinds` of a page's title band, the floor's band (the
    top :data:`TITLE_BAND` of the page as it reads, turned or not)."""
    try:
        content = doc.page(page)
    except Exception:
        return ()
    turn = _turn_of_content(content)
    band = top_band(content.lines, turn, TITLE_BAND)
    texts = [ln.text for ln in band] or [
        ln.text for ln in reading_order(content.lines, turn)[:6]]
    return title_kinds(texts)


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


#: A row whose label names a mass or a weight is the working of a water
#: content (the mass of water, in grams), never the water content itself
#: (RI-56). Two signs of a mass, told apart (RI-71):
#:
#: * a UNIT of mass, where a unit is printed -- in brackets, after a comma
#:   or a colon at the end, after "in" or after a number ("(g)", "[gr]",
#:   ", gm", "in lb", "25 g") -- or a unit's whole name anywhere ("grams",
#:   "pounds"). It is a mass even where a percent is printed too. The gram's
#:   one letter is a lower-case g: a capital G, in brackets, after a comma
#:   or anywhere else, is no unit on its own (a specimen "G", "G-14").
#: * a WORD for a mass or a weight, in the floor's languages (mass,
#:   weight, wt, Mw; masse, poids; masa, peso; massa). It is a mass unless
#:   the label prints a percent ("% by weight", "% of dry mass"), and is
#:   no word for a mass with a number run on to it: "MW-14" is a well.
#:
#: A RATIO of masses or a mass per volume (g/g, kg/kg, g/100 g, g/cm3) is no
#: mass: a water content printed as a fraction is one (:func:`_wc_factor`).
_MASS_UNIT = r"(?:(?i:kg|grs?|gms?|lbs?)|g)\.?"
_NOT_A_NAME = r"(?![\w/]|\s?[-#]\s?\d)"
_MASS_UNIT_PRINTED = re.compile(
    rf"[(\[]\s*{_MASS_UNIT}\s*[)\]]"
    rf"|[,:;]\s*{_MASS_UNIT}\s*(?:$|[)\]])"
    rf"|\b(?i:in)\s+{_MASS_UNIT}{_NOT_A_NAME}"
    rf"|\d\s*{_MASS_UNIT}{_NOT_A_NAME}"
    r"|\b(?i:grams?|grammes?|gramos?|kilograms?|pounds?)\b")
_MASS_WORD = (r"\b(?:mass|weight|wt|mw|masse|poids|masa|peso|massa)\b"
              r"(?!\s?[-#]?\s?\d)")
_MASS_NAMES = r"mg|k?g|grs?|gms?|grams?|lbs?|mass|weight|wt"
_MASS_RATIO = re.compile(
    rf"(?<![a-z])(?:\d+(?:[.,]\d+)?\s*)?(?:{_MASS_NAMES})\.?\s*/\s*"
    r"(?:\d+(?:[.,]\d+)?\s*)?[a-zµμ]+\d?", re.I)
#: A water content printed as a fraction of masses: the mass a unit names,
#: in grams; a word for a mass is the same mass on either side.
_GRAMS = {"mg": 0.001, "g": 1.0, "gr": 1.0, "grs": 1.0, "gm": 1.0,
          "gms": 1.0, "gram": 1.0, "grams": 1.0, "kg": 1000.0,
          "lb": 453.59237, "lbs": 453.59237}
_MASS_FRACTION = re.compile(
    rf"(?<![a-z])({_MASS_NAMES})\.?\s*/\s*(\d+(?:[.,]\d+)?)?\s*"
    rf"({_MASS_NAMES})\b")
_PERCENT_LABEL = r"%|\bper ?cent|\bpct\b|pourcent|porcent"
#: A ratio a label says is multiplied by 100 is a percent already.
_TIMES_100 = r"(?<![a-z])[x×*]\s*100(?![\d.,]?\d)"
#: The values a trial table repeats across its columns, one a trial point:
#: the first of such a row is no one specimen's value (RI-56); a CBR's
#: swell and CBR are one a mould too (RI-71).
_ONE_A_SPECIMEN = ("wc", "dry_density", "wet_density", "swell_percent",
                   "cbr_percent")


def _a_mass(label: str) -> bool:
    """Whether a water content ``label`` names a mass (see
    :data:`_MASS_UNIT_PRINTED`)."""
    text = _MASS_RATIO.sub(" ", " ".join(str(label or "").split()))
    return bool(_MASS_UNIT_PRINTED.search(text)) or bool(
        re.search(_MASS_WORD, text, re.I)) and not re.search(
            _PERCENT_LABEL, text, re.I)


def _wc_factor(label: str) -> Optional[float]:
    """What a water content printed under ``label`` is multiplied by to be
    a percent: 1 for a percent, 100 for a fraction of one mass in the same
    mass (g/g, kg/kg), 1 for grams per 100 g, by the units for any other
    ratio of masses; None for a ratio of a word and a unit, which says no
    amount.

    The water content is the ratio times 100: "the gravimetric water or
    moisture content, w, expressed as a percentage as follows: w = Ww/Ws
    x100" (FHWA-NHI-06-088 Vol. I, 2.1.2, Eq. 2-4, pp. 2-4 to 2-5), and
    "water content is understood to be a weight ratio" (p. 2-5). A label
    printing the ratio alone (g/g) gives the fraction; one that prints a
    percent or the "x 100" too ("% wt/wt", "g/g x 100") gives the percent
    already, and is read as printed (RI-71 review).
    """
    text = " ".join(str(label or "").lower().split())
    match = _MASS_FRACTION.search(text)
    if match is None or re.search(_PERCENT_LABEL, text) or re.search(
            _TIMES_100, text):
        return 1.0
    top, count, bottom = match.groups()
    count = float(count.replace(",", ".")) if count else 1.0
    if top in _GRAMS and bottom in _GRAMS and count > 0:
        return 100.0 * _GRAMS[top] / (count * _GRAMS[bottom])
    return 100.0 / count if top == bottom and count > 0 else None


def _not_the_specimens(name: str, label: str, numbers: int) -> bool:
    """Whether a ``label`` row read as ``name`` is a mass of water, or a row
    of ``numbers`` trial points, rather than the specimen's value."""
    if name not in _ONE_A_SPECIMEN:
        return False
    return numbers > 1 or (name == "wc" and _a_mass(label))


def _row_numbers(cells: Sequence[str]) -> int:
    """How many different numbers a property row prints for its label, in
    ``cells`` (the row after the label): up to a cell naming another value
    ("Water content (%) | 15.2 | Specific gravity | 2.69" prints one), and
    one number printed across the columns once ("Average water content |
    12.85 | 12.85 | 12.85", a cell spanning the trial columns) (RI-71)."""
    found: Set[float] = set()
    for cell in cells:
        number = _num(cell)
        if number is not None:
            found.add(number)
        elif _field_of(cell) in FIELD_HOME:
            break
    return len(found)


#: A row read as a cohesion by its one letter ("C", "c'") is one only on
#: a shear test's sheet or where the row prints a unit of stress (RI-71): a
#: grading chart's legend key, a person's initial and a composite's mark
#: are one letter C too. Never where the row, label or cells, names a
#: temperature or prints degrees C or F. The word itself is a cohesion
#: anywhere else.
_COHESION_WORD = r"coh[eé]si[oó]n"
_TEMPERATURE = (r"[°º]\s?[cf](?![\w'’])|\bdeg(?:rees?)?\.?\s?[cf](?![\w'’])"
                r"|\btemp")
#: The name a column of a table of specimens goes by when the per-page
#: floor reads none of its cells as a value: a summary row keeps them in
#: its ``other``, as printed (RI-71 review).
_AS_PRINTED = "as_printed"
#: A number and the unit printed straight after it ("59 kPa", "0.3kg/cm2").
_NUMBER_UNIT = re.compile(
    r"(?<![\w.,])[-+]?\d+(?:[.,]\d+)?\s*([^\s\d,;()\[\]][^\s,;()\[\]]*)")


def _a_stress_unit(text: str) -> bool:
    text = re.sub(r"^[-+]?\d+(?:[.,]\d+)?", "", str(text or "").strip())
    key = _canonical_unit(text.strip()) if text.strip() else None
    return bool(key) and UNIT_TO_SI[key][0] == "stress"


def _prints_stress(head: str, cells: Sequence[str]) -> bool:
    """Whether a row or column prints a unit of stress (kPa, psf, kg/cm2):
    in its ``head`` (the label or the column's header), anywhere; in one of
    its ``cells``, only as the whole cell (a column of units), in brackets
    or straight after a number. A word that only reads as a unit ("Bar 4",
    a sample's name) is not one (RI-71 review)."""
    head = str(head or "")
    if any(_a_stress_unit(t) for t in [head] + _PAREN.findall(head)
           + re.split(r"[\s,;()\[\]]+", head)):
        return True
    for cell in cells:
        cell = str(cell or "")
        if any(_a_stress_unit(t) for t in [cell] + _PAREN.findall(cell)
               + _NUMBER_UNIT.findall(cell)):
            return True
    return False


def _not_a_cohesion(label: str, cells: Sequence[str], kind: str) -> bool:
    """Whether a ``label`` read as a cohesion is a letter C that is no
    cohesion on a sheet of ``kind`` (``cells``: the rest of the row, or the
    column's cells under it)."""
    row = " ".join([str(label or "")] + [str(c or "") for c in cells])
    if re.search(_TEMPERATURE, row.lower()):
        return True
    if re.search(_COHESION_WORD, str(label or "").lower()):
        return False
    return kind not in _STRENGTH_KINDS and not _prints_stress(label, cells)


#: A column a table heads with the word unit(s): it holds units, and the
#: entry a test without one prints there ("pos/neg") (RI-71).
_UNITS_HEAD = re.compile(r"\s*units?\b", re.I)


def _reported_text(cell: str, head: str = "") -> bool:
    """Whether a chemical result printed as words ("Trace", "Not
    detected", "<0.5") is one, and not header text (RI-71): a mark that
    prints nothing ("-", "N/A", "NT"), a unit ("ohm-cm", "mV"), a word
    with no digit naming a value of its own (the next row's label,
    "Chloride"), or the entry of a column whose ``head`` is the word
    unit(s)."""
    text = " ".join(str(cell or "").split())
    if not re.search(r"[0-9A-Za-z]", text) or _canonical_unit(text):
        return False
    if re.fullmatch(r"n/?a|nt", text, re.I):
        return False
    if not re.search(r"\d", text) and _field_of(text):
        return False
    return not _UNITS_HEAD.match(str(head or ""))


def _head_over(rows: Sequence[Sequence[str]], row: Sequence[str]) -> str:
    """The first row's cell over ``row``'s value (its second cell that
    prints anything), or ``""`` for the first row itself."""
    if not rows or row is rows[0]:
        return ""
    printed = [i for i, c in enumerate(row) if c]
    if len(printed) < 2 or printed[1] >= len(rows[0]):
        return ""
    return str(rows[0][printed[1]] or "")


def _read_property_table(rows: List[List[str]], values: _Values,
                         conf: float, strict: bool = False,
                         kind: str = "") -> int:
    """``label | value`` rows. Returns how many values were read.

    ``strict`` (the per-page floor, RI-56) reads no water content from a
    mass or weight row and none of the per-specimen values from a row of
    several different numbers across (:func:`_not_the_specimens`,
    :func:`_row_numbers`); a water content printed as a fraction of masses
    as its percent (:func:`_wc_factor`); a letter C as a cohesion only on a
    sheet of a shear test (``kind``) or with a unit of stress
    (:func:`_not_a_cohesion`); and no header text as a chemical result
    (:func:`_reported_text`).
    """
    n = 0
    for row in rows:
        cells = [c for c in row if c]
        if len(cells) < 2:
            continue
        label, cell = cells[0], cells[1]
        name = _field_of(label)
        if not name or name in ("investigation_id", "sample_id", "depth_top",
                                "uscs", "description", "sample_type",
                                "sieve", "size", "percent_passing", "date",
                                _AS_PRINTED):
            continue
        if strict and _not_the_specimens(name, label,
                                         _row_numbers(cells[1:])):
            continue
        if strict and name == "c" and _not_a_cohesion(label, cells[1:], kind):
            continue
        number = _num(cell)
        unit = _unit_in(label) or _unit_in(cell)
        if number is None:
            if name in ("pH", "resistivity", "sulfate", "chloride",
                        "sulfides", "redox") and cell and (
                    not strict or _reported_text(cell, _head_over(rows,
                                                                  row))):
                values.put(name, cell, unit, conf)
                n += 1
            continue
        if strict and name == "wc" and _wc_factor(label) != 1.0:
            factor = _wc_factor(label)
            if factor is None:
                continue
            number, unit = number * factor, "%"
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


def _an_elevation(head: str) -> bool:
    """A column heading that names an elevation and no depth (RI-123):
    :data:`LABEL_FIELDS` files both under ``depth_top`` as a link, but an
    elevation is on a datum, not below the ground surface."""
    text = str(head or "").lower()
    return bool(re.search(r"\belev(?:ation)?\b", text)) and not re.search(
        r"depth|prof(?:ondeur|undidad)?\b", text)


def _summary_rows(rows: List[List[str]], header: List[str],
                  columns: Dict[int, str], unit: str, conf: float,
                  prov: Provenance,
                  wc_factors: Optional[Dict[int, Optional[float]]] = None
                  ) -> List[SummaryRow]:
    """The rows of a table of specimens. ``wc_factors`` (the per-page floor)
    turns a water content column printed as a fraction into a percent, or
    leaves it unread (None) where its units say no amount."""
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
                if number is not None and _an_elevation(head):
                    # An elevation column is not a depth below the ground
                    # (RI-123): the row has a slot of its own for it.
                    kwargs["elevation_top"] = Quantity(
                        value=number, unit=_unit_in(head) or unit)
                elif number is not None and number < 0.0:
                    # No depth below the ground is negative (RI-123): kept
                    # as printed under its heading, not as the row's depth.
                    other.append((head, text))
                elif number is not None:
                    kwargs["depth_top"] = Quantity(value=number,
                                                   unit=unit or _unit_in(head))
                continue
            number = _num(text)
            row_unit = _unit_in(head)
            if name == "wc" and number is not None:
                factor = (wc_factors or {}).get(i, 1.0)
                if factor is not None:
                    kwargs["wc"] = number * factor
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
                       key: str = "", strict: bool = False
                       ) -> Tuple[int, List[SummaryRow]]:
    """A table whose first row names its columns.

    Three shapes: a grading series (a sieve or size column beside a percent
    passing column); a table of specimens (a boring, sample or depth column
    and more than one row), which seeds summary rows; and a one-row table of
    values for the sheet's one specimen. ``key`` files the values under a
    key other than the link's own (a bare page's, see :func:`_floor`);
    ``strict`` reads no water content from a mass or weight column, a water
    content printed as a fraction as its percent, and a column headed by a
    letter C as a cohesion only as :func:`_read_property_table` does; such
    a column's cells stay in a summary row's ``other``, as printed. Nor,
    where no column names a boring, sample or depth, any value from a
    column with a number in more than one row: those are one a row (a
    compaction's or a CBR's trial points), not the specimen's; nor header
    text as a chemical result (:func:`_reported_text`) (RI-71).
    """
    header = rows[0]
    body = rows[1:]
    columns = {i: _field_of(h) for i, h in enumerate(header)
               if _field_of(h) and _field_of(h) != "date"
               and not (strict and _not_the_specimens(_field_of(h), h, 1))}
    for i, name in columns.items():
        if strict and name == "c" and _not_a_cohesion(
                header[i], [r[i] for r in body if i < len(r)], kind):
            columns[i] = _AS_PRINTED
    if not body or not columns:
        return 0, []
    factors: Dict[int, Optional[float]] = {
        i: _wc_factor(header[i]) for i, name in columns.items()
        if strict and name == "wc"}
    series = _series_from(body, header, columns, conf)
    key = key or _specimen_key(link)
    if series:
        values_by_key.setdefault(key, _Values()).series.extend(series)
        values_by_key[key].provs.append(prov)
        return len(series), []
    link_names = ("investigation_id", "sample_id", "depth_top", "uscs",
                  "description", "sample_type")
    has_link = any(n in link_names[:3] for n in columns.values())
    has_value = any(n not in link_names + (_AS_PRINTED,)
                    for n in columns.values())
    if not has_value:
        # A list of samples with no result column -- a certificate's
        # receipt table, a chain of custody -- seeds nothing.
        return 0, []
    if has_link and (len(body) > 1 or kind == "summary_table"):
        summary = _summary_rows(body, header, columns, link.unit, conf, prov,
                                factors)
        return sum(len([v for v in r.model_dump().values() if v])
                   for r in summary), summary
    n = 0
    row = body[0]
    values = values_by_key.setdefault(key, _Values())
    values.provs.append(prov)
    for i, name in columns.items():
        if i >= len(row) or not row[i]:
            continue
        if name in ("investigation_id", "sample_id", "depth_top", "uscs",
                    "description", "sample_type", "sieve", "size",
                    "percent_passing", _AS_PRINTED):
            continue
        if strict and sum(1 for r in body
                          if i < len(r) and _num(r[i]) is not None) > 1:
            # A column with a number in more than one row and no boring,
            # sample or depth beside it: one value a row -- a compaction's
            # or a CBR's trial points, a test's specimens -- and the first
            # is no one specimen's value (RI-71).
            continue
        number = _num(row[i])
        unit = _unit_in(header[i])
        if number is None:
            if name in ("pH", "resistivity", "sulfate", "chloride",
                        "sulfides", "redox") and (
                    not strict or _reported_text(row[i])):
                values.put(name, row[i], unit, conf)
                n += 1
            continue
        if factors.get(i, 1.0) != 1.0:
            if factors[i] is None:
                continue
            number, unit = number * factors[i], "%"
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


#: Sheets whose water content and densities belong to their own test, not
#: a moisture test of their own: in its result where it has the slot, else
#: carried in the test's ``fields`` (a compaction's, RI-56).
_HOLDS_MOISTURE = _MOISTURE_KINDS + ("atterberg", "gradation", "cbr",
                                     "swell_consolidation") + _STRENGTH_KINDS


def _kinds_for(values: _Values, hint: str,
               per_page: bool = False) -> List[str]:
    """Which result kinds one specimen's floor values call for.

    ``per_page`` (the floor the merge takes, RI-56) holds a compaction
    sheet's water content and densities in its compaction test as well.
    """
    homes: List[str] = []
    holds = _HOLDS_MOISTURE + (("compaction",) if per_page else ())
    for name in values.index:
        home = FIELD_HOME.get(name, "")
        if home == "moisture":
            home = hint if hint in holds else "moisture_content"
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
    # the limits, not a moisture test of its own; nor, on the per-page
    # floor, on a compaction or CBR sheet (RI-56).
    beside = ("atterberg", "gradation") + (("compaction", "cbr")
                                           if per_page else ())
    if hint in beside and "moisture_content" in homes and hint in homes:
        homes.remove("moisture_content")
    order = [k for k in ([hint] if hint else []) + homes if k]
    seen: List[str] = []
    for k in order:
        if k not in seen and k in RESULT_CLASS:
            seen.append(k)
    return seen


def _no_value(result: Any) -> bool:
    """Whether a floor result holds no value at all."""
    if result is None:
        return True
    return all(value in (None, "", [], {}, False)
               for name, value in result.model_dump().items()
               if name != "kind")


def _held_moisture(values: _Values, result: Any) -> Dict[str, str]:
    """A compaction sheet's water content and densities its result has no
    slot for, as printed, so that they stay with the test (RI-56).

    Only on a sheet whose title names a compaction (RI-71). A compaction
    another sheet quotes -- the maximum dry density a field density test is
    measured against -- holds none of that sheet's values: they are the
    sheet's own test's.
    """
    out: Dict[str, str] = {}
    for name, (value, unit, _conf) in values.index.items():
        if FIELD_HOME.get(name) == "moisture" and \
                getattr(result, name, None) is None:
            out[name] = show(value) if unit in ("", "%") else \
                f"{show(value)} {unit}"
    return out


def floor_from_tables(doc: Any, pages: Sequence[int],
                      report_id: str = "") -> FloorSheet:
    """The page's own tables and title as :class:`LabTest` records.

    Nothing here is inferred beyond the word printed beside the number. A
    sheet whose title names no kind and whose tables hold no labelled value
    comes back with an empty test list, which is an honest floor.

    EACH PAGE ITS OWN LINK (RI-56). A page's values carry the boring, sample
    and depth that page prints. A page that prints neither a boring nor a
    depth leaves its values unbound -- never given another page's -- and is
    named in :attr:`FloorSheet.unbound` where another page of the sheet
    prints one. The title's kind is corrected as :func:`kind_from_title`
    says. The merge always takes this floor. The model's brief shows the
    floor as it was before RI-56 (:attr:`FloorSheet.as_before`) unless
    ``GEOTECH_INGEST_LAB_FLOOR_PER_PAGE`` is on.
    """
    sheet = _floor(doc, pages, report_id, per_page=True)
    sheet.as_before = _floor(doc, pages, report_id, per_page=False)
    return sheet


def _page_link(own: _Link, first: _Link) -> _Link:
    """A later page's link: what the page prints itself, never the first
    page's boring, sample or depth. Where the page prints no unit, the first
    page's depth UNIT stands: for a depth printed bare, and for a summary
    table's depth column that prints none."""
    if not own.unit:
        own.unit = first.unit
    return own


def _bound(link: _Link) -> bool:
    return bool(link.investigation_id) or link.depth_top is not None


def _floor(doc: Any, pages: Sequence[int], report_id: str,
           per_page: bool) -> FloorSheet:
    """:func:`floor_from_tables`, with each page's own link and the
    corrected kind (``per_page``), or with the first page's link on every
    page and the kind by line order alone, as before RI-56."""
    pages = [int(p) for p in pages]
    sheet = FloorSheet(pages=pages)
    if not pages:
        return sheet
    first = _lines_of(doc, pages[0])
    turn = _page_turn(doc, pages[0])
    if turn.turned:
        # A sheet scanned sideways: its title is at the top of the page as
        # it reads, which is a side of the page as displayed (RI-149).
        top = [ln.text for ln in top_band(first, turn, TITLE_BAND)] \
            or [ln.text for ln in first[:6]]
    else:
        try:
            height = float(doc.summary(pages[0]).height)
        except Exception:
            height = 792.0
        top = [ln.text for ln in first if ln.bbox[1] <= TITLE_BAND * height] \
            or [ln.text for ln in first[:6]]
    sheet.kind, sheet.title = kind_from_title(top, corrected=per_page)
    sheet.link = _link_from_lines(first, pages[0])
    links: Dict[int, _Link] = {pages[0]: sheet.link}
    for page in pages[1:]:
        links[page] = (_page_link(_link_from_lines(_lines_of(doc, page),
                                                   page), sheet.link)
                       if per_page else sheet.link)

    values_by_key: Dict[str, _Values] = {}
    link_of: Dict[str, _Link] = {}
    keyed: Dict[int, int] = {}                  # page -> values it keyed
    summary_rows: List[SummaryRow] = []
    table_provs: List[Provenance] = []
    linked = [p for p in pages if _bound(links[p])]
    bare: List[str] = []
    for page in pages:
        link = links[page]
        key = _specimen_key(link)
        if per_page and len(pages) > 1 and not _bound(link):
            # A bare page of a run: its own specimen, not one shared with
            # every other bare page.
            key = f"{key}#page{page}"
            bare.append(key)
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
                                                sheet.kind, conf, prov, key,
                                                per_page)
                if summary:
                    summary_rows.extend(summary)
                elif n:
                    keyed[page] = keyed.get(page, 0) + n
                sheet.n_values += n
                if n or summary:
                    continue
            values = values_by_key.setdefault(key, _Values())
            n = _read_property_table(rows, values, conf, per_page,
                                     kind=sheet.kind)
            if n:
                values.provs.append(prov)
                sheet.n_values += n
                keyed[page] = keyed.get(page, 0) + n
        if key in values_by_key:
            link_of.setdefault(key, link)

    # The legend of a sheet plotting several specimens, read row by row on
    # each page whose tables gave no value (RI-149): one grading test per
    # specimen, each with the values its own row prints. Each page's hole is
    # that page's own, on either floor.
    for page in pages:
        if keyed.get(page):
            continue
        kinds = page_title_kinds(doc, page)
        if kinds and "gradation" not in kinds:
            continue
        own = _page_link(_link_from_lines(_lines_of(doc, page), page),
                         sheet.link)
        sheet.tests.extend(_legend_test(row, report_id)
                           for row in legend_specimens(doc, page, own))

    if per_page and linked:
        sheet.unbound = [{"page": p, "values": keyed[p], "linked": linked}
                         for p in pages
                         if keyed.get(p) and not _bound(links[p])]

    def link_prov(link: _Link) -> Provenance:
        return Provenance(page=link.page, bbox=link.bbox, method="text",
                          confidence=CONF_LINK,
                          note=("the boring and depth printed on the sheet"
                                if _bound(link) or not per_page else
                                "no boring or depth printed on this page"))

    def depths(link: _Link) -> Tuple[Optional[Quantity], Optional[Quantity]]:
        return ((Quantity(value=link.depth_top, unit=link.unit)
                 if link.depth_top is not None else None),
                (Quantity(value=link.depth_bottom, unit=link.unit)
                 if link.depth_bottom is not None else None))

    if summary_rows:
        prov = table_provs[0] if table_provs else link_prov(sheet.link)
        sheet.tests.append(LabTest(
            kind="summary_table",
            result=SummaryTableResult(rows=summary_rows, title=sheet.title),
            pages=list(pages), source_report=report_id,
            prov=[prov]))

    # A particle density corrected to a specific gravity (RI-56) is told to
    # the model, and makes no test of its own where the tables read no value
    # for it: the sheet's one value is the model's to read. A corrected
    # compaction keeps its test of the kind alone, which is an answer.
    corrected = per_page and sheet.kind == "specific_gravity" and \
        sheet.kind != kind_from_title(top, corrected=False)[0]
    for key, values in values_by_key.items():
        if key in bare and not values.index and not values.series:
            continue                # a bare page with no value: no test
        link = link_of.get(key, sheet.link)
        depth, depth_bottom = depths(link)
        for kind in _kinds_for(values, sheet.kind, per_page):
            if kind == "summary_table":
                continue
            result = _build_result(kind, values, sheet.kind)
            fields = (_held_moisture(values, result) if per_page and
                      kind == sheet.kind == "compaction" else {})
            if corrected and kind == sheet.kind and not fields and \
                    _no_value(result):
                continue
            provs = [link_prov(link)] + [p for p in values.provs]
            sheet.tests.append(LabTest(
                kind=kind, investigation_id=link.investigation_id,
                sample_id=link.sample_id, depth_top=depth,
                depth_bottom=depth_bottom, pages=list(pages),
                source_report=report_id, result=result, prov=provs,
                fields=fields))

    # A run of pages none of which prints a boring or a depth: each page is
    # its own specimen, and a page with no value is none, so a run whose
    # pages hold no value has no specimen to give the kind alone (RI-56).
    all_bare = per_page and len(pages) > 1 and not linked
    if not sheet.tests and sheet.kind and not corrected and not all_bare:
        # The title names the kind and the tables hold no value: the kind
        # alone is the floor, and it is an answer.
        depth, depth_bottom = depths(sheet.link)
        result = (OtherResult(no_results=True) if sheet.kind == "other"
                  else None)
        sheet.tests.append(LabTest(
            kind=sheet.kind, investigation_id=sheet.link.investigation_id,
            sample_id=sheet.link.sample_id, depth_top=depth,
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


def _link_words(boring: str, sample: str, top: Optional[float],
                bottom: Optional[float], unit: str) -> str:
    bits = []
    if boring:
        bits.append(f"boring '{boring}'")
    if sample:
        bits.append(f"sample '{sample}'")
    if top is not None:
        bits.append(f"depth {top:g}"
                    + (f" to {bottom:g}" if bottom is not None else "")
                    + (f" {unit}" if unit else " (unit not printed)"))
    return ", ".join(bits)


def _test_link_words(test: LabTest) -> str:
    top, bottom = test.depth_top, test.depth_bottom
    return _link_words(test.investigation_id, test.sample_id,
                       top.value if top is not None else None,
                       bottom.value if bottom is not None else None,
                       (top.unit if top is not None else "") or "")


def serialise_floor(sheet: FloorSheet, per_page: bool = False,
                    turns: Optional[Mapping[int, Turn]] = None) -> str:
    """The floor as compact lines the model builds on.

    By default the floor as the brief showed it before RI-56
    (:attr:`FloorSheet.as_before`), byte for byte. ``per_page``
    (``GEOTECH_INGEST_LAB_FLOOR_PER_PAGE``) shows the floor the merge takes:
    the corrected kind, and, where a test's page prints a link other than
    the first page's, each test under the link its own page prints. A sheet
    whose floor is the same either way reads the same either way.

    ``turns``: a page's turn where the reader lists that page and shows its
    picture in the upright frame (RI-149); a box on such a page is shown in
    that frame too (:func:`report_ingest.turned.upright_box`). Every other
    page's boxes are shown as they always were.
    """
    turned = {int(p): t for p, t in (turns or {}).items()
              if t is not None and t.turned}

    def box_on(page: int, bbox: Sequence[float]) -> Sequence[float]:
        turn = turned.get(int(page))
        return upright_box(bbox, turn) if turn is not None else bbox

    if not per_page:
        sheet = sheet.as_before or sheet
    out: List[str] = []
    out.append(f"kind from the title: {sheet.kind or 'none read'}"
               + (f" ('{sheet.title[:70]}')" if sheet.title else ""))
    link = sheet.link
    words = _link_words(link.investigation_id, link.sample_id,
                        link.depth_top, link.depth_bottom, link.unit)
    own = {id(test): _test_link_words(test) for test in sheet.tests
           if test.kind != "summary_table"} if per_page else {}
    per_test = any(text != words for text in own.values())
    out.append((f"link printed on page {link.page}: " if per_test else
                "link printed on the sheet: ") + (words or "none read"))
    for test in sheet.tests:
        where = ", ".join(
            f"page {p.page}" + (" box " + ",".join(
                f"{v:.0f}" for v in box_on(p.page, p.bbox)) if p.bbox else "")
            for p in test.prov if p.method == "tables") or \
            f"page {test.pages[0] if test.pages else '?'}"
        conf = max((p.confidence for p in test.prov), default=0.5)
        out.append(f"{test.kind} ({where})  [confidence {conf:.2f}]")
        if per_test and id(test) in own:
            out.append("  link printed on its page: " + (
                own[id(test)] or "none (its values are left unbound)"))
        legend = _legend_box(test)
        if legend is not None:
            out.append("  read off one legend row, the label and these "
                       "values printed on that row (values box "
                       + ",".join(f"{v:.0f}"
                                  for v in box_on(legend[0], legend[1]))
                       + "): " + (_test_link_words(test) or "no label read"))
        out.extend(_result_lines(test))
        if per_page and test.fields:
            out.append("  also printed: " + ", ".join(
                f"{name} {value}" for name, value in test.fields.items()))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# the merge
# ---------------------------------------------------------------------------

def _hole_named(text: Any) -> str:
    """A hole's name folded, read out of a printed field where the text is
    one ("BORE HOLE NO. (B-2)" is B-2, as the floor reads it off the
    sheet), else the text folded."""
    match = _LINK_BORING.search(str(text or ""))
    return fold(match.group(1) if match else text)


def _is_legend(test: LabTest) -> bool:
    return any(p.note == LEGEND_NOTE for p in test.prov)


def _depths_meet(a: LabTest, b: LabTest) -> Optional[bool]:
    """:func:`same_depth` of two tests' tops; but a legend label that prints
    no unit, on a sheet that prints none, is set against the other's
    printed number as printed: its unit is the other's (RI-149), where
    converting the other's alone would set feet against bare numbers."""
    da, db = a.depth_top, b.depth_top
    if da is not None and db is not None and \
            (_is_legend(a) or _is_legend(b)) and \
            bool(str(da.unit or "").strip()) != bool(str(db.unit or "").strip()):
        return same_number(da.value, db.value, 0.05)
    return same_depth(da, db)


def _same_specimen(a: LabTest, b: LabTest) -> bool:
    ida, idb = _hole_named(a.investigation_id), _hole_named(b.investigation_id)
    if ida and idb and ida != idb:
        return False
    same = _depths_meet(a, b)
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


def _slot(kind: str, name: str) -> str:
    """The result field of a ``kind`` test that holds the value a floor
    test names ``name``."""
    return _RENAME.get(kind, {}).get(name, name)


#: Why a floor value that does not pin one specimen lost (RI-56).
_UNPINNED_WHY = ("the table's value comes from a page whose boring and "
                 "depth do not pin one specimen, so it fills an empty slot "
                 "but does not overrule the model; the model's reading "
                 "stands and the table's is kept beside it")
_UNPINNED_NOTE = ("the table's value; its page's boring and depth do not "
                  "pin one specimen, so it did not overrule the model")
#: Why a legend row's value stood against the model's (RI-149).
_LEGEND_WHY = ("the legend prints this value on the specimen's own row and "
               "the model's box is on neither the line printing its label "
               "nor the line printing its values; the legend's stands and "
               "the model's is kept beside it")
#: Why a legend row's value stood against a model's value that another
#: specimen's row prints (RI-149).
_SWAP_WHY = ("the model's value is the one another specimen's legend row "
             "prints on this sheet, so it was read off that row; the "
             "legend's own row stands and the model's is kept beside it")
#: Why a model's value its legend row does not print was set aside (RI-149).
_LEGEND_ASIDE_WHY = ("the model read this specimen off another specimen's "
                     "legend row (its values there overruled), and this "
                     "value, which the specimen's own row does not print, is "
                     "the one that other row prints; it is set beside the "
                     "record")
#: The link fields: a legend row prints them on its label's line.
_LINK_FIELDS = ("investigation_id", "sample_id", "depth_top", "depth_bottom")
#: The legend values one row prints in one family: a reader that took
#: another row's values may have taken them under another name too (each
#: row prints its shares in its own order), so a value is another row's
#: where that row prints it under any name of the family.
_LEGEND_FAMILY: Dict[str, Tuple[str, ...]] = {
    name: family
    for family in (("cobbles_percent", "gravel_percent", "sand_percent",
                    "silt_percent", "clay_percent", "fines_percent"),
                   ("d10", "d30", "d50", "d60", "d85", "d90", "d100"),
                   ("cu", "cc"))
    for name in family}


def _row_prints(test: LabTest, name: str, value: Any) -> bool:
    """Whether a legend test's row prints ``value`` under ``name`` or any
    other name of its family (:data:`_LEGEND_FAMILY`)."""
    if test.result is None or value is None:
        return False
    return any(_same_value(other, getattr(test.result, other, None),
                           value) is True
               for other in _LEGEND_FAMILY.get(name, (name,)))
#: The grading values a legend row prints, and a model's reading of a
#: specimen off another row carries: the shares, the diameters, Cu and Cc.
LEGEND_FIELDS = ("cobbles_percent", "gravel_percent", "sand_percent",
                 "silt_percent", "clay_percent", "fines_percent", "d10", "d30",
                 "d50", "d60", "d85", "d90", "d100", "cu", "cc")
#: Why a floor value carried into a test's fields lost (RI-56).
_FIELD_WHY = ("the model's test already holds another value here; the "
              "model's stands and the table's is kept beside it")


class _LabMerger:
    def __init__(self, floor: Sequence[LabTest], model: Sequence[LabTest],
                 log: MergeLog, title_kind: str = "") -> None:
        self.floor = list(floor)
        self.model = [t.model_copy(deep=True) for t in model]
        self.log = log
        self.paired: set = set()
        self.title_kind = title_kind
        #: The other specimens' legend rows the pair being merged took
        #: values from (RI-149): a model value its own row overruled because
        #: another row prints it (the swap signature), by that row's test.
        self._swap_from: List[LabTest] = []

    def _swap_rows(self, floor_test: LabTest, name: str, mv: Any
                   ) -> List[LabTest]:
        """The other specimens' legend rows on ``floor_test``'s page that
        print ``mv`` (under ``name`` or another name of its family,
        :func:`_row_prints`): where the model's value is one of them, the
        model read this specimen off that row."""
        legend = _legend_box(floor_test)
        if legend is None or mv is None or name not in LEGEND_FIELDS:
            return []
        out: List[LabTest] = []
        for other in self.floor:
            if other is floor_test or other.result is None:
                continue
            there = _legend_box(other)
            if there is None or there[0] != legend[0]:
                continue
            if _row_prints(other, name, mv):
                out.append(other)
        return out

    def _pinned(self, f: LabTest, kind: str) -> bool:
        """Whether a floor test's link pins ONE of the model's ``kind``
        tests: it prints a boring or a depth, and matches no more than one.

        A floor test that is not pinned may fill an empty slot but never
        wins a contradiction (RI-56): a page printing no boring or depth
        matches every specimen, a depth matches every one within the
        tolerance, and its value may be another specimen's.
        """
        if not fold(f.investigation_id) and f.depth_top is None:
            return False
        return sum(1 for m in self.model
                   if m.kind == kind and _same_specimen(f, m)) <= 1

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
                     what: str, pinned: bool = True) -> None:
        """Settle one floor value against the model's ``holder.name``.

        A floor value that is not ``pinned`` (:meth:`_pinned`) fills an
        empty slot and loses every contradiction, kept beside the model's.
        """
        mv = getattr(holder, name, None)
        if mv == "" or mv == []:
            mv = None
        same = _same_value(name, fv, mv)
        mprov = self._prov0(target)
        page = mprov.page
        evidence = has_evidence(mprov, None)
        # A value read off a legend row is that row's (RI-149). The model
        # overrules it only with a box on that row: on the line printing
        # the specimen's label (the box the reader is asked for) or, for a
        # value, on the line printing its values. A value read off the
        # picture with a note keeps the evidence it has. And a value that
        # ANOTHER specimen's row prints on the sheet (the swap signature) is
        # how a reader that paired the rows wrongly gives one specimen
        # another's values: the row's own stands against it.
        legend = _legend_boxes(floor_test)
        off_line = False
        swapped_from: List[LabTest] = []
        if legend is not None:
            l_page, label_box, values_box = legend
            if evidence and mprov.bbox is not None and len(mprov.bbox) == 4:
                lines = [label_box] if name in _LINK_FIELDS else \
                    [label_box, values_box]
                off_line = not (int(mprov.page) == l_page and any(
                    b is not None and _on_line(mprov.bbox, b)
                    for b in lines))
            if same is False:
                swapped_from = self._swap_rows(floor_test, name, mv)
            if off_line or swapped_from:
                evidence = False
        unpinned = not pinned and not evidence
        s = settle(name, fv, mv, same,
                   floor_method="tables",
                   floor_confidence=self._floor_conf(floor_test),
                   model_method=mprov.method if mprov.method in (
                       "model", "model_from_picture", "vision") else "model",
                   model_confidence=float(mprov.confidence),
                   evidence=evidence or not pinned,
                   floor_unit=fv.unit if isinstance(fv, Quantity) else "",
                   model_unit=mv.unit if isinstance(mv, Quantity) else "",
                   model_note=mprov.note)
        if unpinned and s.alternative is not None:
            s.alternative = s.alternative.model_copy(update={
                "note": _UNPINNED_NOTE})
        if swapped_from and s.verdict == "floor_wins":
            self._swap_from.extend(swapped_from)
        if s.verdict == "reconciled":
            self.log.reconciled += 1
        elif s.verdict == "floor_only":
            self.log.keep(where, f"{what}: {name}", page, fv)
        elif s.verdict in ("floor_wins", "model_wins"):
            self.log.disagree(
                where, f"{what}: {name}", fv, mv,
                kept="model" if s.verdict == "model_wins" else "floor",
                why=(_UNPINNED_WHY if unpinned else
                     _SWAP_WHY if swapped_from and s.verdict == "floor_wins"
                     else
                     _LEGEND_WHY if off_line and s.verdict == "floor_wins"
                     else
                     "the model named a box and a note, so its reading "
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
                      where: str, pinned: bool = True) -> None:
        mprov = self._prov0(target)
        evidence = has_evidence(mprov, None) or not pinned
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

    def _fill_fields(self, target: LabTest, fields: Dict[str, str],
                     floor_test: LabTest, where: str, what: str) -> None:
        """Values the floor holds as printed text, into the empty slots of
        ``target.fields``; one the model filled otherwise stays the
        model's, with the table's beside it."""
        mprov = self._prov0(target)
        conf = self._floor_conf(floor_test)
        for name, value in fields.items():
            have = target.fields.get(name)
            if have is None:
                target.fields[name] = value
                self.log.keep(where, f"{what}: {name}", mprov.page, value)
            elif same_text(have, value):
                self.log.reconciled += 1
            else:
                self.log.disagree(
                    where, f"{what}: {name}", value, have, kept="model",
                    why=_FIELD_WHY, page=mprov.page,
                    floor_method="tables", floor_confidence=conf,
                    model_confidence=float(mprov.confidence),
                    model_method=mprov.method)
                mprov.alternatives.append(Alternative(
                    field=name, value=value, method="tables",
                    confidence=conf, note="the table's value"))

    def _set_aside(self, m: LabTest, f: LabTest, where: str, what: str,
                   sources: Sequence[LabTest]) -> None:
        """The model's grading values its specimen's legend row does not
        print and the row it read the specimen off does, set beside the
        record rather than in it (RI-149). ``sources`` are the other
        specimens' rows whose values overruled the model's on this one (the
        swap signature, :meth:`_swap_rows`). A value no such row prints
        stays the model's: the sheet may print it elsewhere (a diameter off
        the curve, the fines in a box of their own), where the floor did
        not read it."""
        mprov = self._prov0(m)
        for name in LEGEND_FIELDS:
            if getattr(f.result, name, None) is not None:
                continue
            mv = getattr(m.result, name, None)
            if mv is None:
                continue
            if not any(_row_prints(src, name, mv) for src in sources):
                continue
            setattr(m.result, name, None)
            mprov.alternatives.append(Alternative(
                field=name, value=show(mv),
                unit=mv.unit if isinstance(mv, Quantity) else "",
                method=mprov.method if mprov.method in (
                    "model", "model_from_picture", "vision") else "model",
                confidence=float(mprov.confidence),
                note="the model's reading, set aside: its legend row "
                     "prints no such value, and the row the model read "
                     "this specimen off prints this one"))
            self.log.disagree(
                where, f"{what}: {name}", "not printed on the legend row",
                mv, kept="floor", why=_LEGEND_ASIDE_WHY, page=mprov.page,
                floor_method="text",
                floor_confidence=self._floor_conf(f),
                model_confidence=float(mprov.confidence),
                model_method=mprov.method)

    def merge_pair(self, f: LabTest, m: LabTest) -> None:
        where = self._label(m)
        what = f.kind
        mprov = self._prov0(m)
        pinned = self._pinned(f, m.kind)
        # The link, then the values.
        for name in ("investigation_id", "sample_id", "depth_top",
                     "depth_bottom"):
            fv = getattr(f, name)
            if fv in (None, ""):
                continue
            self._settle_into(m, m, name, fv, f, where, "link", pinned)
        if f.fields:
            self._fill_fields(m, f.fields, f, where, what)
        if f.result is None:
            pass
        elif m.result is None:
            m.result = f.result.model_copy(deep=True)
            self.log.keep(where, f"{what}: every value", mprov.page)
        elif isinstance(f.result, SummaryTableResult) and \
                isinstance(m.result, SummaryTableResult):
            self._merge_rows(m, m.result, f.result.rows, f, where)
        else:
            self._swap_from = []
            for name in _scalar_fields(f.result):
                fv = getattr(f.result, name, None)
                if fv in (None, "", False):
                    continue
                if not hasattr(m.result, name):
                    m.fields.setdefault(name, show(fv))
                    self.log.keep(where, f"{what}: {name}", mprov.page, fv)
                    continue
                self._settle_into(m, m.result, name, fv, f, where, what,
                                  pinned)
            if self._swap_from:
                self._set_aside(m, f, where, what, self._swap_from)
            self._swap_from = []
            if isinstance(f.result, GradationResult) and \
                    isinstance(m.result, GradationResult):
                self._merge_series(m, m.result, f.result.percent_passing,
                                   f, where, pinned)
        for p in f.prov:
            if p not in m.prov:
                m.prov.append(p)

    def _order(self, f: LabTest) -> Optional[List[int]]:
        """The order a floor test looks through the model's tests for its
        specimen, or None for the model's own order.

        Each page's values carry that page's link (RI-56), and a link
        matches more than one specimen: a page that prints no boring or
        depth matches every one, a depth matches every specimen within the
        tolerance whatever its sample. Where the model put its tests on more
        than one page, the page decides among the specimens that match: the
        model's test on the floor test's page, else the nearest that starts
        before it (a sheet runs down its pages), else the rest in the
        model's order.
        """
        pages = sorted({p.page for p in f.prov if p.method == "tables"
                        or p.note == LEGEND_NOTE})
        if not pages:
            return None

        def page_of(m: LabTest) -> Optional[int]:
            return m.prov[0].page if m.prov else None

        if len({page_of(m) for m in self.model}) < 2:
            return None

        def rank(i: int) -> Tuple[int, int, int]:
            page = page_of(self.model[i])
            if page in pages:
                return (0, 0, i)
            if page is not None and page < pages[0]:
                return (1, pages[0] - page, i)
            return (2, 0, i)

        return sorted(range(len(self.model)), key=rank)

    def place_values(self, f: LabTest) -> Optional[LabTest]:
        """A floor test with no twin: its values find homes, or it stays."""
        order = self._order(f) or range(len(self.model))
        candidates = [self.model[i] for i in order
                      if _same_specimen(f, self.model[i])]
        if f.result is None or isinstance(f.result, (SummaryTableResult,
                                                     OtherResult)):
            return f.model_copy(deep=True) if not any(
                m.kind == f.kind for m in candidates) else None
        homeless: Dict[str, Any] = {}
        homed = 0
        first_home: Optional[LabTest] = None
        by_values = f.kind in _MOISTURE_KINDS and f.kind != self.title_kind

        def slot(m: LabTest, name: str) -> str:
            return _slot(m.kind, name) if by_values else name

        for name in _scalar_fields(f.result):
            fv = getattr(f.result, name, None)
            if fv in (None, "", False):
                continue
            # A test named by its values (``by_values``) finds its slot in
            # the home's own kind: an Atterberg test's water content is its
            # ``water_content`` (:data:`_RENAME`).
            home = next((m for m in candidates if m.result is not None
                         and hasattr(m.result, slot(m, name))), None)
            if home is None:
                homeless[name] = fv
                continue
            self._settle_into(home, home.result, slot(home, name), fv,
                              f, self._label(home), f.kind,
                              self._pinned(f, home.kind))
            homed += 1
            first_home = first_home or home
        series = (f.result.percent_passing
                  if isinstance(f.result, GradationResult) else [])
        if series:
            home = next((m for m in candidates
                         if isinstance(m.result, GradationResult)), None)
            if home is not None:
                self._merge_series(home, home.result, series, f,
                                   self._label(home),
                                   self._pinned(f, home.kind))
                homed += 1
                first_home = first_home or home
            else:
                homeless["percent_passing"] = series
        # A water content or density the floor named by its values (no
        # title kind was read, or the title names another kind), on a
        # specimen the model reported as another kind, is part of that test
        # -- an Atterberg sheet's natural water content, a compaction's --
        # not a test of its own: what has no slot there goes into its fields
        # (RI-56).
        other = next((m for m in candidates if m.kind != f.kind), None)
        if other is not None and by_values and (f.fields or homeless):
            moved = dict(f.fields)
            for name in list(homeless):
                moved[name] = show(homeless.pop(name))
            self._fill_fields(other, moved, f, self._label(other), f.kind)
            return None
        if not homeless:
            if f.fields and first_home is not None:
                self._fill_fields(first_home, f.fields, f,
                                  self._label(first_home), f.kind)
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
            order = self._order(f)
            if order is not None:
                # The specimen its page names, paired or not: a later one is
                # another specimen, and a paired one takes its values below.
                twin = next((i for i in order if self.model[i].kind == f.kind
                             and _same_specimen(f, self.model[i])), None)
                if twin in self.paired:
                    twin = None
            else:
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
                    log: Optional[MergeLog] = None, title_kind: str = ""
                    ) -> Tuple[List[LabTest], MergeLog]:
    """The tables' tests and the model's as one list, and what the merge did.

    The model's tests come first, in the model's own order, each merged
    with the floor test of the same kind on the same specimen where there
    is one; floor tests with no twin follow, whole where none of their
    values found a home and as a residue where some did. ``title_kind`` is
    the kind the sheet's title names (:attr:`FloorSheet.kind`, ``""`` where
    none was read): a floor test of any other kind was named by its values.
    """
    log = log if log is not None else MergeLog()
    merged = _LabMerger(floor, model, log, title_kind).run()
    return merged, log
