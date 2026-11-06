"""Recompute a report's calculations through the app's own modules (DR-11).

Report ingest writes each calculation printout as a
:class:`~report_ingest.model.Calculation`: its kind, the method and program
as printed, and the labelled values it was GIVEN and WORKED OUT. This module
takes each one, finds the module inputs among the printed labels, runs the
module the geotech page's agent runs (through
:func:`funhouse_agent.dispatch.call_agent`, so the result carries the
module's own ``choices`` and ``defaults_used``), and returns one row per
claimed result: the claimed value, the recomputed value, the difference and
a flag past a stated tolerance, or the reason it was not recomputed.

Rules (``module_work/dr11/PLAN.md``):

* **Nothing is guessed.** An input the recompute needs that the printout did
  not give, or gave twice with different values, makes the row
  ``not_recomputed`` and names it. An optional input the module documents a
  case for is left to the module, and the row lists the module's
  ``defaults_used`` under ``assumed``.
* **No new formula.** Only the existing modules compute; this file maps
  labels and converts units (:func:`report_ingest.model.calc_si`).
* **Open methods.** A method the report names is used; where it names none,
  bearing capacity runs the module's default and lists the module's other
  cited options beside it with their values, and settlement takes the method
  its printed inputs fit, and says so.
* **Never a confident wrong flag.** Where the printout does not settle a
  module option (the method, the depth factors, the elastic Iw), the
  module's own alternatives run beside the default, and a claim is flagged
  ``differs`` only outside the range they span. Printed inputs the module
  cannot take as printed (an undrained and a drained strength together, a
  strain influence factor, a compression ratio) make the row
  ``not_recomputed`` with the reason.

First slice: ``shallow_foundation_bearing`` and ``settlement`` (elastic, and
one-layer Cc/Cr consolidation). Every other kind is a row saying why it was
not recomputed; ``slope_stability`` is held (GM-140).

Second slice (``module_work/dr11/CHECK_SLICE2.md``): a printout that puts
several CASES in one calculation (scenario x stratum x load case, a footing
shape) is split by the case its labels print, wherever in the label the
qualifier sits ('phi, stratum 2, long term', 'B (Case A)', 'Stratum 2 phi'),
and each case is recomputed on its own, its rows naming the case. Labels are
matched with the case words taken out. A printout that gives only gamma'
follows the setting :data:`GAMMA_PRIME_ENV` (OWN-130, the owner's open
decision): refused by default.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from report_ingest.model import Calculation, NamedQuantity, calc_si

__all__ = ["TOLERANCE", "TOLERANCE_BASIS", "NOT_RECOMPUTED",
           "GAMMA_PRIME_ENV", "GAMMA_PRIME_RULES", "recompute_record",
           "recompute_calculation", "table_markdown", "find_record",
           "make_recompute_tool", "RECOMPUTE_TOOL_DESCRIPTION"]

#: A row is flagged ``differs`` when the recomputed value is more than this
#: fraction away from the claimed one. A QC threshold of this table, not an
#: engineering value: printed factors are often read off charts and rounded,
#: and a few per cent between two honest calculations is common.
TOLERANCE = 0.10

#: What the tolerance is, said in every table and in the tool's description.
TOLERANCE_BASIS = ("this table's QC threshold, not an engineering "
                   "criterion: printed factors are read off charts and "
                   "rounded")

#: The setting for a bearing printout that gives only the effective unit
#: weight gamma' and no total unit weight (OWN-130, the owner's open
#: decision): ``refuse`` (the default) lists the calculation as 'water level
#: not printed'; ``submerged`` runs it as the submerged case (gamma' above
#: and below the footing base) and says so on the row. The function argument
#: ``gamma_prime`` overrides it.
GAMMA_PRIME_ENV = "GEOTECH_RECOMPUTE_GAMMA_PRIME"
GAMMA_PRIME_RULES = ("refuse", "submerged")


def _gamma_prime_rule(value: Optional[str]) -> str:
    rule = (value if value is not None
            else os.environ.get(GAMMA_PRIME_ENV, "")).strip().lower()
    rule = rule or "refuse"
    if rule not in GAMMA_PRIME_RULES:
        raise ValueError(f"gamma_prime rule {rule!r}: one of "
                         f"{', '.join(GAMMA_PRIME_RULES)} ({GAMMA_PRIME_ENV})")
    return rule


#: Why each kind the first slice does not build is not recomputed.
NOT_RECOMPUTED: Dict[str, str] = {
    "slope_stability": (
        "slope stability is held: not recomputed yet (no slope_stability "
        "work until the owner's integral review, GM-140)"),
    "axial_pile": (
        "not recomputed yet: the axial_pile and drilled_shaft modules need "
        "the layered profile and the pile section as rows, which a later "
        "slice reads"),
    "lateral_pile": (
        "not recomputed yet: the lateral_pile module needs a layered p-y "
        "profile and the section, which a later slice reads"),
    "pile_group": (
        "not recomputed yet: the pile_group module needs the group layout "
        "and each pile's stiffness, which a later slice reads"),
    "retaining_wall": (
        "not recomputed yet: retaining_walls, sheet_pile and soe recompute "
        "walls in a later slice"),
    "liquefaction": (
        "not recomputed yet: the liquefaction tool needs a layer table of N "
        "or cone readings, which a later slice reads"),
    "site_response": (
        "not recomputed yet: site class and site response come in a later "
        "slice"),
    "pavement": (
        "not recomputed yet: pavement_design (AASHTO 1993, UFC 3-250-01) "
        "comes in a later slice"),
    "ground_improvement": (
        "not recomputed yet: ground_improvement comes in a later slice"),
    "seepage": (
        "none in the app: no module recomputes seepage from a printout's "
        "labelled values (fem2d's seepage needs a full mesh geometry)"),
    "other": "no calculation kind: nothing to recompute",
}

# ---------------------------------------------------------------------------
# reading the printed labels
# ---------------------------------------------------------------------------

_GREEK = {"φ": "phi", "ϕ": "phi", "γ": "gamma", "σ": "sigma", "δ": "delta",
          "Δ": "delta", "ν": "nu", "ρ": "rho", "′": "'", "’": "'"}


def _norm(text: str) -> str:
    out = str(text or "")
    for k, v in _GREEK.items():
        out = out.replace(k, v)
    return " ".join(out.lower().split())


# -- the case a label prints --------------------------------------------------

#: A case's id: a number (with a letter), a roman numeral or one letter.
_CASE_ID = r"(?:no\.?\s*)?([0-9]+[a-z]?|[ivx]+(?![a-z])|[a-z](?![a-z]))"

_SHAPE_CASE = {"strip": "strip", "continuous": "strip", "square": "square",
               "circular": "circular", "round": "circular",
               "rectangular": "rectangular"}

#: The case dimensions a printout's labels carry, in the order a case is
#: named: (dimension, pattern, value from the match).
_CASE_DIMS: Tuple[Tuple[str, "re.Pattern[str]", Callable[[Any], str]],
                  ...] = (
    ("scenario", re.compile(r"\bscenario\s*" + _CASE_ID),
     lambda m: m.group(1)),
    ("case", re.compile(r"\b(?:load\s+)?case\s*" + _CASE_ID),
     lambda m: m.group(1)),
    ("stratum", re.compile(r"\b(?:soil\s+)?(?:stratum|layer)\s*" + _CASE_ID),
     lambda m: m.group(1)),
    ("term", re.compile(r"\b(short|long)[- ]?term\b"), lambda m: m.group(1)),
    ("shape", re.compile(r"\b(strip|continuous|square|circular|round|"
                         r"rectangular)\b(?:\s+(?:footing|foundation)s?\b)?"),
     lambda m: _SHAPE_CASE[m.group(1)]),
)


def _case_of(label: str) -> Tuple[Dict[str, str], str]:
    """``(the case the label prints, the label without it)``, normalised:
    'phi, stratum 2, long term' -> ``({"stratum": "2", "term": "long"},
    "phi")``; 'B (Case A)' -> ``({"case": "a"}, "b")``."""
    lab = _norm(label)
    case: Dict[str, str] = {}
    for dim, pat, value in _CASE_DIMS:
        def _take(m, dim=dim, value=value):
            case.setdefault(dim, value(m))
            return " "
        lab = pat.sub(_take, lab)
    if case:
        lab = re.sub(r"[(\[]\s*[-,;:/&\s]*\s*[)\]]", " ", lab)
        lab = re.sub(r"(\s*[-\u2013,;:/&]\s*){2,}", ", ", lab)
        lab = re.sub(r"^[\s\-\u2013,;:/&]+|[\s\-\u2013,;:/&]+$", "", lab)
        lab = re.sub(r"\s+([,;:)])", r"\1", lab)
        lab = re.sub(r"[,;:]\s*\(", " (", lab)
        lab = " ".join(lab.split())
    return case, lab


def _lab(label: str) -> str:
    """A label as matched: normalised, its case words taken out."""
    return _case_of(label)[1]


def _case_name(case: Dict[str, str]) -> str:
    parts = []
    for dim, _pat, _v in _CASE_DIMS:
        if dim not in case:
            continue
        v = case[dim]
        v = v.upper() if len(v) == 1 and v.isalpha() else v
        parts.append(f"{v} term" if dim == "term" else
                     v if dim == "shape" else f"{dim} {v}")
    if "name" in case:
        parts.append(case["name"])
    return ", ".join(parts)


# -- a case printed by its name ----------------------------------------------

#: The words of the module's own inputs and claims. What is left of a label
#: without them, its case words, units, numbers, one-letter symbols and small
#: words is the NAME of the case the label prints: a stratum printed by its
#: soil ('phi (<soil>)', '<soil> phi', 'phi, <soil>'), or a structure.
_NAME_NOISE = re.compile(
    r"\b(bearing capacity|bearing pressure|bearing|capacity|pressure|"
    r"allowable|ultimate|net|gross|footing|foundation|dimensions?|"
    r"factor of safety|safety|factor|width|length|embedment|depth|"
    r"friction angle|angle of (internal )?friction|friction|angle|cohesion|"
    r"undrained|shear strength|strength|unit weight|unit wt|weight|density|"
    r"effective|total|moist|saturated|buoyant|submerged|water table|"
    r"groundwater|water|phi|gamma|settlement|immediate|elastic|"
    r"consolidation|primary|cumulative|last row of the table|"
    r"scenarios?|cases?|strat(um|a)|layers?|"
    r"q_?all?|q_?ult|q_?a|q_?u)\b")

#: A symbol of a factor or a ratio, never a name.
_NAME_SYMBOL = re.compile(
    r"^(n_?[cqg]|n_?gamma|n_?y|[sdi]_?[cqg]|[sdi]_?gamma|f[cqy]s|k[pa0o]|"
    r"fs|fos|s_?u|c_?u|c'|e_?s|nu|ocr|d_?f|b/l|l/b|b/d_?f)$")


def _names_of(lab: str) -> List[str]:
    """The names a label (without its case words) prints: each part between
    commas, colons or brackets, kept to its words once the module's own
    words, units, numbers, one-letter symbols and small words are out."""
    out = []
    for part in re.split(r"[,;:()\[\]]", _NAME_NOISE.sub(" ", lab)):
        words = []
        for w in re.findall(r"[^\s/]+", part):
            w = w.strip("-–—=.'\"")
            if (len(w) < 2 or w in _STOP_WORDS or _UNIT_WORD.match(w)
                    or _NAME_SYMBOL.match(w)
                    or re.fullmatch(r"[\d.,+=\-]+", w)):
                continue
            words.append(w)
        if words:
            out.append(" ".join(words))
    return out


#: Words a label's length is counted without.
_STOP_WORDS = frozenset(
    "the of at to in a an for from on and or below above under per by with "
    "is be".split())

#: A unit printed as a word of a label ('ft', 'ksf', 'kN/m3').
_UNIT_WORD = re.compile(
    r"^(k?pa|mpa|[kpt]sf|ksi|psi|ft|feet|foot|m|meters?|metres?|in|inch|"
    r"inches|mm|cm|deg|degrees?|°|[kp]cf|kn|kips?|k|lbs?|t)"
    r"(/[a-z0-9^³²]+)*$")


def _skip(nq: NamedQuantity, claim: bool = False) -> bool:
    """A printed setting or a sentence, not a labelled value: printed as
    words, a yes/no choice, an enumerated item ('a) ...') or a label of more
    than eight words (its case, the parentheses, units, small words and
    words with a digit left out: 'P1', 'L=10', '200 ft' are not words).
    A claim (``claim``) is skipped only when printed as words: a result
    label naming its structure and size is still the claim."""
    if nq.value is None:
        return True
    if claim:
        return False
    raw = _norm(nq.name)
    if re.search(r"\byes\b.*\bno\b|\by/n\b", raw) or re.match(
            r"^\(?[a-z0-9]\)\s", raw):
        return True
    text = re.sub(r"\([^)]*\)", " ", _lab(nq.name))
    words = [w for w in re.findall(r"[^\s,;:\[\]]+", text)
             if w not in _STOP_WORDS and re.search(r"[a-z]", w)
             and not re.search(r"\d", w) and not _UNIT_WORD.match(w)]
    return len(words) > 8


class _Spec:
    """One module input: its name, quantity kind and the printed labels
    that name it (regular expressions on the lower-cased label), with the
    words that rule a label out."""

    def __init__(self, param: str, kind: str, patterns: Sequence[str],
                 exclude: Sequence[str] = ()):
        self.param = param
        self.kind = kind
        self.patterns = [re.compile(p) for p in patterns]
        self.exclude = [re.compile(p) for p in exclude]

    def matches(self, label: str) -> bool:
        lab = _lab(label)
        return (any(p.search(lab) for p in self.patterns)
                and not any(p.search(lab) for p in self.exclude))


_LENGTH_UNITS = {"length": "m", "stress": "kPa", "unit_weight": "kN/m3"}


def _page(nq: NamedQuantity) -> Optional[int]:
    """The 1-based page a value was printed on, when the record says."""
    prov = nq.prov or (nq.value.prov if nq.value is not None else None)
    return None if prov is None else int(prov.page) + 1


def _printed(nq: NamedQuantity) -> str:
    if nq.value is not None:
        return f"{nq.value.value:g} {nq.value.unit}".strip()
    return nq.text


def _si(nq: NamedQuantity, kind: str) -> Tuple[Optional[float], str]:
    """``(value in the module's unit, note)``; None with the reason when the
    printed value is not a number of that kind."""
    q = nq.value
    if q is None:
        return None, f"'{nq.name}' is printed as words ({nq.text!r})"
    unit = str(q.unit or "").strip()
    if kind in ("angle", "ratio"):
        if not unit or unit in ("-", "--"):
            note = ("printed without a unit: read as degrees"
                    if kind == "angle" else "")
            return float(q.value), note
        if kind == "ratio":
            return None, f"'{nq.name}' has unit {unit!r}, not a ratio"
    got = calc_si(q)
    want = _LENGTH_UNITS.get(kind, "deg" if kind == "angle" else None)
    if got is None:
        return None, (f"'{nq.name}' is printed in {unit!r}, which the record "
                      f"cannot convert" if unit else
                      f"'{nq.name}' is printed without a unit")
    value, si_unit = got
    if si_unit != want:
        return None, f"'{nq.name}' is in {unit!r}, not a {kind}"
    return value, ""


def _collect(rows: Sequence[NamedQuantity], specs: Sequence[_Spec]
             ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """The module inputs found among the printed ``rows``: ``(found,
    problems)``. A label goes to the first spec it matches. A parameter
    printed twice with different values, or in a unit of the wrong kind, is a
    problem, never a choice between them."""
    found: Dict[str, Dict[str, Any]] = {}
    problems: Dict[str, str] = {}
    for nq in rows:
        if _skip(nq):
            continue
        spec = next((s for s in specs if s.matches(nq.name)), None)
        if spec is None:
            continue
        value, note = _si(nq, spec.kind)
        if value is None:
            if spec.param not in found:
                problems.setdefault(spec.param, note)
            continue
        entry = {"value": value, "label": nq.name, "printed": _printed(nq),
                 "page": _page(nq)}
        if note:
            entry["note"] = note
        old = found.get(spec.param)
        if old is not None:
            if abs(old["value"] - value) > 1e-9 * max(1.0, abs(value)):
                problems[spec.param] = (
                    f"printed more than once with different values "
                    f"('{old['label']}' {old['printed']}, '{nq.name}' "
                    f"{_printed(nq)})")
            continue
        found[spec.param] = entry
        problems.pop(spec.param, None)
    for param in problems:
        found.pop(param, None)
    return found, problems


def _all_words(calc: Calculation) -> str:
    parts = [calc.method, calc.program or "", calc.subject, calc.summary]
    for nq in list(calc.inputs) + list(calc.results):
        parts += [nq.name, nq.text]
    return _norm(" ".join(parts))


def _method_words(calc: Calculation) -> str:
    return _norm(f"{calc.method} {calc.program or ''}")


_SHAPE_WORDS = (("strip", r"\bstrip\b|\bcontinuous\b|\bwall footing\b"),
                ("square", r"\bsquare\b"),
                ("circular", r"\bcircular\b|\bcircle\b|\bround\b"),
                ("rectangular", r"\brectang"))


def _shape(calc: Calculation, found: Dict[str, Dict[str, Any]]
           ) -> Tuple[Optional[str], str]:
    """``(shape, how)`` from the printed words, else from this calculation's
    (this case's) own labels when its words name several shapes or none,
    else from L against B; None with the reason when nothing settles it."""
    words = _all_words(calc)
    named = [s for s, pat in _SHAPE_WORDS if re.search(pat, words)]
    if len(named) == 1:
        return named[0], f"printed: {named[0]}"
    labels = _norm(" ".join(f"{nq.name} {nq.text}" for nq in
                            list(calc.inputs) + list(calc.results)))
    own = [s for s, pat in _SHAPE_WORDS if re.search(pat, labels)]
    if len(own) == 1:
        return own[0], f"printed in this case's labels: {own[0]}"
    B = (found.get("width") or found.get("B") or {}).get("value")
    L = (found.get("length") or found.get("L") or {}).get("value")
    if len(named) > 1:
        return None, ("footing shape: the printout names several "
                      f"({', '.join(named)})")
    if B is not None and L is not None:
        if abs(L - B) <= 1e-6 * max(B, 1.0):
            return "square", "from the printed L = B"
        if L > B:
            return "rectangular", "from the printed L > B"
    return None, "footing shape (no shape word printed, and no L to set it)"


# ---------------------------------------------------------------------------
# the rows
# ---------------------------------------------------------------------------

def _base_row(calc: Calculation, index: int) -> Dict[str, Any]:
    return {"calc_index": index, "kind": calc.kind, "subject": calc.subject,
            "pages": [p + 1 for p in calc.pages],
            "program": calc.program, "method_printed": calc.method}


def _not_row(calc: Calculation, index: int, reason: str,
             status: str = "not_recomputed", **extra) -> Dict[str, Any]:
    row = _base_row(calc, index)
    row.update(status=status, reason=reason, **extra)
    return row


#: The flags a claim can get past ``agrees``, in the order the tool shows
#: them first when its result is cut.
_FLAG_ORDER = ("differs", "below", "within_options")

_FLAG_BASIS = {
    "agrees": "the claim is within {tol} of the recomputed value",
    "within_options": ("the claim is more than {tol} from the recomputed "
                       "value but inside the range of the module's own "
                       "options, so the printout does not settle a "
                       "difference"),
    "below": "the claim is more than {tol} below every option run",
    "differs": "the claim is more than {tol} outside every option run",
}


def _compare(row: Dict[str, Any], nq: NamedQuantity, claimed: float,
             recomputed: float, unit: str, tolerance: float,
             options: Sequence[Dict[str, Any]] = (), default: str = "",
             upper_bound: bool = False) -> None:
    """Fill the comparison of one claim.

    ``options``: the module's other options run beside its default (each
    with its ``option`` label and ``recomputed_si``, or refused). A claim
    is ``differs`` only outside the range the default and the options span,
    past the tolerance; inside it but away from the default it is
    ``within_options``, because the printout does not settle which option it
    used. ``upper_bound``: an allowable pressure, which settlement may set
    below q_ult / FS, so a claim below every option is ``below``."""
    diff = None if claimed == 0 else (recomputed - claimed) / abs(claimed)
    ran = [o for o in options if o.get("recomputed_si") is not None]
    values = [recomputed] + [o["recomputed_si"] for o in ran]
    lo, hi = min(values), max(values)
    if diff is None:
        flag = "differs" if recomputed != 0 else "agrees"
    elif abs(diff) <= tolerance:
        flag = "agrees"
    elif lo <= claimed <= hi or any(
            abs(v - claimed) <= tolerance * abs(claimed) for v in values):
        flag = "within_options"
    elif upper_bound and claimed < lo:
        flag = "below"
    else:
        flag = "differs"
    row.update(
        quantity=nq.name,
        claimed={"printed": _printed(nq), "page": _page(nq)},
        claimed_si=round(claimed, 6), recomputed_si=round(recomputed, 6),
        unit_si=unit,
        difference_pct=None if diff is None else round(100 * diff, 2),
        flag=flag, status="recomputed", reason="")
    if ran:
        row["options_range_si"] = [round(lo, 6), round(hi, 6)]
        row["flag_basis"] = (
            f"the module ran {default} (the recomputed value), and beside "
            f"it " + "; ".join(o["option"] for o in ran)
            + f": {lo:.4g} to {hi:.4g} {unit}; "
            + _FLAG_BASIS[flag].format(tol=f"{100 * tolerance:g} %"))


def _call(module: str, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    from funhouse_agent.dispatch import call_agent
    return call_agent(module, method, params)


def _basis(result: Dict[str, Any], what: str) -> str:
    for c in result.get("choices") or []:
        if what in str(c.get("what", "")):
            return str(c.get("basis", ""))
    return ""


# -- bearing capacity -------------------------------------------------------

#: A ratio of two printed quantities (B/L, L/B, B/Df, q/Nc, su/sigma'v, a
#: stress ratio) is never one of them. A unit's slash (kN/m2) is not a ratio.
_RATIO = [r"\bratio\b",
          r"(^|[\s(,])(b|l|d_?f|q_?c?|s_?u|c_?u|sigma\S*|width|length|depth)"
          r"\s*/"]

#: The footing's width and length; a ratio (B/L, L/B, B/Df) is neither.
_WIDTH = _Spec("width", "length", [r"\bwidth\b", r"^b\b"],
               [r"effective", r"b'"] + _RATIO)
_LENGTH = _Spec("length", "length", [r"\blength\b", r"^l\b"],
                [r"effective", r"l'"] + _RATIO)

#: The total unit weight: gamma' is the effective one, a relative density
#: no unit weight.
_UNIT_WEIGHT = _Spec(
    "unit_weight", "unit_weight",
    [r"unit w(eigh)?t\b", r"\bdensity\b", r"^gamma\b"],
    [r"effective", r"buoyant", r"submerged", r"water", r"gamma ?'",
     r"\bprime\b", r"\bgamma_?(sub|b)\b", r"relative density", r"\bd_?r\b"])

#: The effective (buoyant) unit weight gamma': used only under the OWN-130
#: setting, never as the total.
_GAMMA_PRIME = _Spec(
    "gamma_prime", "unit_weight",
    [r"effective unit w(eigh)?t\b", r"buoyant", r"submerged unit",
     r"gamma ?'", r"\bgamma_?(sub|b)\b", r"\bprime\b"], [r"water"])

_BEARING_INPUTS = (
    _WIDTH,
    _LENGTH,
    _Spec("depth", "length",
          [r"embedment", r"\b(footing|foundation) depth\b",
           r"\bdepth of (the )?(footing|foundation)\b", r"\bdf\b",
           r"\bd_f\b"], [r"water", r"gwt"] + _RATIO),
    _Spec("friction_angle", "angle",
          [r"friction angle", r"angle of (internal )?friction", r"^phi\b",
           r"\bphi'?\b"]),
    # an undrained strength is its own input: run as the cohesion with
    # phi = 0, never together with a drained c' or phi
    # an unconfined strength (qu = 2 su) is not su
    _Spec("su", "stress",
          [r"undrained (shear )?strength", r"undrained cohesion",
           r"\bs_?u\b", r"\bc_?u\b"], [r"unconfined"] + _RATIO),
    _Spec("cohesion", "stress", [r"cohesion", r"^c'?\b"],
          [r"undrained", r"unconfined"] + _RATIO),
    _UNIT_WEIGHT,
    _GAMMA_PRIME,
    _Spec("gwt_depth", "length",
          [r"water table", r"groundwater", r"\bgwt\b", r"\bd_?w\b",
           r"depth to (ground ?)?water", r"water depth"]),
    _Spec("factor_of_safety", "ratio",
          [r"factor of safety", r"\bf\.?s\.?\b", r"\bfos\b"]),
)

#: Claimed results, most specific first: the module's result key and the
#: printed labels that claim it.
_ALLOWABLE = r"allowable|\bq_?all?\b|\bq_?a\b"
_BEARING_RESULTS = (
    ("q_allowable_net_kPa", _ALLOWABLE, r"\bnet\b"),
    ("q_allowable_kPa", _ALLOWABLE, None),
    ("q_ultimate_kPa", r"ultimate|\bq_?ult\b", None),
)

_BEARING_METHODS = ("vesic", "meyerhof", "terzaghi")

#: A printed label naming a depth factor: the report used depth factors.
_DEPTH_FACTOR_LABEL = re.compile(r"depth factor|\bd ?_?(c|q|gamma|g)\b")

#: The key of a "net ultimate" claim, which is not compared.
_NET_ULTIMATE = "net_ultimate"

#: An allowable pressure below q_ult / FS is consistent with the print
#: (checked in the reference library's UFC 3-220-20 text and print).
_BELOW_NOTE = (
    "lower than q_ult/FS: UFC 3-220-20 Sec. 5-2.4 defines the allowable "
    "bearing pressure as the highest pressure that meets both the factored "
    "bearing capacity and settlement (p. 278), prints Eq. 5-14 as an upper "
    "bound, q_all,net <= q_ult/F_BC - sigma_zD, and lowers the allowable "
    "where settlement controls (step 3b, p. 279); check the report's basis")


def _bearing_claims(calc: Calculation):
    out = []
    for nq in calc.results:
        lab = _lab(nq.name)
        if _skip(nq, claim=True) or "factor" in lab:
            continue
        for key, must, also in _BEARING_RESULTS:
            if re.search(must, lab) and (also is None or re.search(also, lab)):
                if key == "q_allowable_kPa" and re.search(r"\bnet\b", lab):
                    continue
                if key == "q_ultimate_kPa" and re.search(r"\bnet\b", lab):
                    key = _NET_ULTIMATE
                out.append((key, nq))
                break
    return out


def _alternative(alt: Dict[str, Any], key: str, claimed: Optional[float]
                 ) -> Dict[str, Any]:
    """One of the module's other options, and what it gives."""
    res = alt["result"]
    out = {k: v for k, v in alt.items() if k != "result"}
    if "error" in res:
        out["refused"] = str(res["error"])[:300]
        return out
    value = float(res[key])
    out["recomputed_si"] = round(value, 6)
    if claimed:
        out["difference_pct"] = round(100 * (value - claimed) / abs(claimed),
                                      2)
    return out


def _net_ultimate_row(calc: Calculation, index: int, nq: NamedQuantity
                      ) -> Dict[str, Any]:
    return _not_row(calc, index, status="not_compared", quantity=nq.name,
                    claimed={"printed": _printed(nq), "page": _page(nq)},
                    reason=("a net ultimate pressure (q_ult - q): the "
                            "module's q_ult is the gross one"))


#: A water depth printed below the footing, not below the ground.
_BELOW_FOOTING = re.compile(r"below (the )?(footing|foundation|base|bottom|"
                            r"bearing)")

_SUBMERGED_NOTE = (
    "only gamma' is printed: run as the submerged case, gamma' above and "
    "below the footing base (water at or above the ground), the setting's "
    "'submerged' rule (OWN-130); with the water between the ground and the "
    "base the overburden would be larger")


def _gamma_prime(found: Dict[str, Dict[str, Any]],
                 problems: Dict[str, str], missing: List[str], rule: str,
                 extra: Dict[str, Any]) -> None:
    """The OWN-130 rule for a printout that gives gamma' and no total unit
    weight. ``refuse``: 'water level not printed', the row not run.
    ``submerged``: gamma' runs as the unit weight with no water table, and
    the row says so. A printed water depth below the ground makes it a
    refusal under both: the total unit weight above the water is missing."""
    gp = found.get("gamma_prime")
    problems.pop("gamma_prime", None)
    if gp is None:
        return
    if "unit_weight" in found or "unit_weight" in problems:
        gp["note"] = "not used: the total unit weight is printed"
        return
    gwt = found.get("gwt_depth")
    what = f"'{gp['label']}' {gp['printed']}"
    if gwt is not None and gwt["value"] > 0:
        missing.append(
            f"unit_weight: only the effective unit weight is printed ({what}) "
            f"with a water depth below the ground ('{gwt['label']}' "
            f"{gwt['printed']}): the total unit weight above the water is not "
            f"printed")
        return
    if rule == "submerged":
        found["unit_weight"] = {**gp, "note": _SUBMERGED_NOTE}
        gp["note"] = "run as the unit weight (the submerged case)"
        if gwt is not None:
            gwt["note"] = ("not passed: the submerged case already takes "
                           "gamma' everywhere")
        extra["gamma_prime_rule"] = "submerged: " + _SUBMERGED_NOTE
        return
    note = (f"water level not printed: only the effective unit weight is "
            f"printed ({what}) and no total unit weight; where the water "
            f"stands is not printed (OWN-130; the setting {GAMMA_PRIME_ENV}"
            f"='submerged' runs it as the submerged case)")
    missing.append(note)
    extra["water_level"] = "not printed"
    extra["gamma_prime_rule"] = "refuse: " + note


def _water_below_footing(found: Dict[str, Dict[str, Any]],
                         missing: List[str]) -> None:
    """A water depth printed below the footing: Df is added, and said."""
    gwt = found.get("gwt_depth")
    if gwt is None or not _BELOW_FOOTING.search(_lab(gwt["label"])):
        return
    depth = found.get("depth")
    if depth is None:
        found.pop("gwt_depth")
        missing.append(f"gwt_depth: '{gwt['label']}' is below the footing, "
                       f"and the footing depth Df is not printed")
        return
    gwt["value"] = gwt["value"] + depth["value"]
    gwt["note"] = (f"printed below the footing: Df ('{depth['label']}' "
                   f"{depth['printed']}) added, {gwt['value']:g} m below "
                   f"the ground")


def _recompute_bearing(calc: Calculation, index: int, tolerance: float,
                       gamma_prime: str = "refuse"
                       ) -> List[Dict[str, Any]]:
    claims = _bearing_claims(calc)
    if not claims:
        return [_not_row(calc, index, status="not_compared", reason=(
            "no ultimate or allowable bearing pressure among the printed "
            "results"))]
    net_rows = [_net_ultimate_row(calc, index, nq) for key, nq in claims
                if key == _NET_ULTIMATE]
    claims = [c for c in claims if c[0] != _NET_ULTIMATE]
    if not claims:
        return net_rows
    words = _method_words(calc)
    if re.search(r"two.?layer|layered", words):
        return [_not_row(calc, index, "a two-layer bearing case: a later "
                         "slice maps its second layer")] + net_rows
    found, problems = _collect(calc.inputs, _BEARING_INPUTS)
    su = found.get("su")
    if su is not None:
        drained = [found[p] for p in ("cohesion", "friction_angle")
                   if p in found and (p == "cohesion"
                                      or found[p]["value"] > 0)]
        if drained:
            return [_not_row(
                calc, index, "an undrained and a drained strength are both "
                "printed (" + "; ".join(f"'{e['label']}' {e['printed']}"
                                        for e in [su] + drained)
                + "): the short- and long-term cases are separate runs, not "
                "recomputed together", inputs_used=found,
                module="bearing_capacity",
                module_method="bearing_capacity_analysis")] + net_rows
    missing = [f"{p}: {why}" for p, why in problems.items()
               if p != "gamma_prime"]
    extra: Dict[str, Any] = {}
    gp_missing: List[str] = []
    _water_below_footing(found, gp_missing)
    _gamma_prime(found, problems, gp_missing, gamma_prime, extra)
    for need in ("width", "depth", "unit_weight"):
        if need not in found and need not in problems:
            missing.append(f"{need}: not printed")
    missing += gp_missing
    if not {"friction_angle", "cohesion", "su"} & (set(found) | set(problems)):
        missing.append("friction_angle or cohesion (c' or su): none printed")
    shape, shape_how = _shape(calc, found)
    if shape is None:
        missing.append(shape_how)
    elif shape == "rectangular" and "length" not in found:
        missing.append("length: a rectangular footing needs L, not printed")
    if missing:
        return [_not_row(calc, index, "inputs not printed: "
                         + "; ".join(missing), inputs_missing=missing,
                         inputs_used=found, module="bearing_capacity",
                         module_method="bearing_capacity_analysis", **extra)
                ] + net_rows
    params: Dict[str, Any] = {k: v["value"] for k, v in found.items()
                              if k != "gamma_prime"}
    if extra.get("gamma_prime_rule", "").startswith("submerged"):
        params.pop("gwt_depth", None)
    if su is not None:
        params["cohesion"] = params.pop("su")
        su["note"] = "run as the cohesion c with phi = 0 (the undrained case)"
    params["shape"] = shape
    if shape in ("strip", "square", "circular"):
        params.pop("length", None)

    named = [m for m in _BEARING_METHODS if m in words]
    hansen = "hansen" in words
    if len(named) == 1:
        method, how = named[0], f"printed: {calc.method or calc.program}"
    elif not named and hansen:
        method, how = "vesic", ("printed: Hansen; the module takes Hansen "
                                "as the N-gamma option with Vesic's factors")
        params["ngamma_method"] = "hansen"
    else:
        method = None
        how = ("the printout names no method the module offers: the "
               "module's default (Vesic) is the recomputed value, its other "
               "cited options beside it" if not named else
               f"the printout names several ({', '.join(named)}): the "
               f"module's default (Vesic) is the recomputed value, the "
               f"others beside it")
    run = dict(params)
    if method is not None:
        run["factor_method"] = method
    result = _call("bearing_capacity", "bearing_capacity_analysis", run)
    if "error" in result:
        return [_not_row(calc, index, f"the module refused: {result['error']}",
                         inputs_used=found, module="bearing_capacity",
                         module_method="bearing_capacity_analysis")
                ] + net_rows
    used = result.get("factor_method", method or "vesic")
    alternatives = []
    if method is None:
        for other in _BEARING_METHODS:
            if other == used:
                continue
            alt = _call("bearing_capacity", "bearing_capacity_analysis",
                        {**params, "factor_method": other})
            alternatives.append({"option": f"factor_method '{other}'",
                                 "method": other, "result": alt})
    # the module's default depth factors are Hansen's; a printout that names
    # none may have used none (GEC-6 Sec. 5.2.3.3), so both run
    labels = [_lab(n.name) for n in list(calc.inputs) + list(calc.results)]
    depth_alt = (params.get("depth", 0) > 0 and used in ("vesic", "meyerhof")
                 and not any(_DEPTH_FACTOR_LABEL.search(lab)
                             for lab in labels))
    if depth_alt:
        alt = _call("bearing_capacity", "bearing_capacity_analysis",
                    {**run, "factor_method": used,
                     "depth_factor_method": "none"})
        alternatives.append({
            "option": f"factor_method '{used}' with depth_factor_method "
                      "'none'",
            "method": used, "depth_factor_method": "none",
            "why": ("no depth factor is printed: the module's default "
                    "(depth_factor_method 'hansen') and none both run"),
            "basis": _basis(alt, "depth factors"), "result": alt})
    default = (f"factor_method '{used}'"
               + (", ngamma_method 'hansen'"
                  if params.get("ngamma_method") else "")
               + (", depth_factor_method 'hansen'" if depth_alt else ""))
    rows = []
    for key, nq in claims:
        row = _base_row(calc, index)
        row.update(module="bearing_capacity",
                   module_method="bearing_capacity_analysis",
                   method_used=used, method_basis=how,
                   method_reference=_basis(result, "bearing capacity method"),
                   inputs_used=found, inputs_missing=[],
                   assumed=result.get("defaults_used") or {},
                   shape=f"{shape} ({shape_how})", **extra)
        claimed = _si(nq, "stress")[0]
        if claimed is None:
            row.update(quantity=nq.name, status="not_compared",
                       reason=_si(nq, "stress")[1])
            rows.append(row)
            continue
        if key != "q_ultimate_kPa" and "factor_of_safety" not in found:
            row.update(quantity=nq.name,
                       claimed={"printed": _printed(nq), "page": _page(nq)},
                       claimed_si=round(claimed, 6), unit_si="kPa",
                       status="not_compared", reason=(
                           "an allowable pressure, and the printout gives no "
                           "factor of safety (the module's 3.0 would be a "
                           "guess)"))
            rows.append(row)
            continue
        row["alternatives"] = [_alternative(a, key, claimed)
                               for a in alternatives]
        _compare(row, nq, claimed, float(result[key]), "kPa", tolerance,
                 options=row["alternatives"], default=default,
                 upper_bound=key != "q_ultimate_kPa")
        notes = []
        if key == "q_allowable_kPa":
            notes.append("compared with the gross allowable q_ult / FS "
                         "(UFC 3-220-20 Eq. 5-13); a net allowable label "
                         "says 'net'")
        if row["flag"] == "below":
            notes.append(_BELOW_NOTE)
        if notes:
            row["note"] = "; ".join(notes)
        rows.append(row)
    return rows + net_rows


# -- settlement --------------------------------------------------------------

_ELASTIC_INPUTS = (
    _Spec("q_net", "stress",
          [r"pressure", r"\bbearing stress\b", r"^q\b", r"^q_?net\b",
           r"load intensity", r"applied stress", r"contact stress"],
          [r"preconsolidation", r"overburden", r"effective", r"initial",
           r"allowable", r"ultimate", r"increase", r"added", r"pore",
           r"water", r"^q_?c?\s*/"] + _RATIO),
    _Spec("B", "length", _WIDTH.patterns, _WIDTH.exclude),
    _Spec("L", "length", _LENGTH.patterns, _LENGTH.exclude),
    _Spec("Es", "stress",
          [r"modulus", r"^e_?s\b", r"^e\b", r"young"],
          [r"subgrade", r"reaction", r"shear modulus"]),
    _Spec("nu", "ratio", [r"poisson", r"^nu\b"]),
    # GEC-6 Eq. 5-36's Iw (Table 5-12's shape and rigidity factor) only:
    # Schmertmann's Iz, Fox's depth factor and Bowles' Is are other factors
    _Spec("Iw", "ratio",
          [r"\bi_?w\b", r"^influence factor$", r"shape and rigidity factor"],
          [r"strain", r"depth", r"embedment"]),
)

#: A printed factor elastic_settlement (Se = q B (1 - nu^2) Iw / Es) has no
#: place for, when it is not the Iw above.
_OTHER_FACTOR = re.compile(r"influence|\b(depth|embedment) (correction )?"
                           r"factor\b|\bi_?zp?\b|\bi_[sf]\b")

_CONSOLIDATION_INPUTS = (
    _Spec("OCR", "ratio", [r"\bocr\b", r"over.?consolidation ratio"]),
    _Spec("sigma_p", "stress",
          [r"preconsolidation", r"maximum past", r"^sigma'?_?p\b", r"^p_?c\b"
           r"|^p'_?c\b"]),
    _Spec("delta_sigma", "stress",
          [r"stress increase", r"increase in (vertical )?stress",
           r"added stress", r"change in stress", r"^delta ?sigma",
           r"^delta ?p\b"]),
    _Spec("sigma_v0", "stress",
          [r"(initial|existing|in.?situ).*(stress|overburden)",
           r"effective (overburden|vertical) stress", r"overburden",
           r"^sigma'?_?v ?0\b", r"^sigma'?_?vo\b", r"^p_?0\b", r"^p_?o\b"],
          _RATIO),
    _Spec("thickness", "length",
          [r"thickness", r"^h\b", r"height of (the )?(soil )?layer",
           r"layer height"], [r"drain"]),
    _Spec("depth_to_center", "length",
          [r"depth to (the )?(mid|cent)", r"mid.?depth", r"(center|centre) "
           r"(depth|of (the )?layer)"]),
    _Spec("e0", "ratio",
          [r"void ratio", r"^e_?0\b", r"^e_?o\b"]),
    # the indices only: a compression or recompression RATIO (CR, RR) is
    # strain-based, and the secondary compression index C-alpha is no Cc
    _Spec("Cr", "ratio",
          [r"recompression", r"swelling", r"\bcr\b", r"\bcs\b"],
          [r"\bratio\b"]),
    _Spec("Cc", "ratio", [r"compression index", r"\bcc\b"],
          [r"secondary", r"\bratio\b"]),
)

_CONSOLIDATION_REQUIRED = ("thickness", "depth_to_center", "e0", "Cc", "Cr",
                           "sigma_v0", "delta_sigma")


def _settlement_method(calc: Calculation) -> Tuple[Optional[str], str]:
    """``(method, how)``: printed, else from the inputs; None with why."""
    for words, src in ((_method_words(calc), "printed"),
                       (_norm(calc.summary), "the summary names")):
        if re.search(r"schmertmann", words):
            return None, ("Schmertmann's strain influence method: a later "
                          "slice reads its layer table")
        if re.search(r"\bhough\b", words):
            return None, ("Hough's method: a later slice reads its layer "
                          "table")
        elastic = bool(re.search(r"elastic|immediate", words))
        consol = bool(re.search(r"consolidation|e.?log|compression index",
                                words))
        if elastic and consol:
            return None, ("immediate and consolidation together: a later "
                          "slice recomputes the combined analysis")
        if elastic:
            return "elastic", f"{src}: elastic"
        if consol:
            return "consolidation", f"{src}: consolidation"
    el, _ = _collect(calc.inputs, _ELASTIC_INPUTS)
    co, _ = _collect(calc.inputs, _CONSOLIDATION_INPUTS)
    has_es, has_cc = "Es" in el, "Cc" in co
    if has_es and not has_cc:
        return "elastic", ("the printout names no method: elastic, from the "
                           "inputs printed (a modulus and no compression "
                           "index)")
    if has_cc and not has_es:
        return "consolidation", ("the printout names no method: Cc/Cr "
                                 "consolidation, from the inputs printed (a "
                                 "compression index and no modulus)")
    return None, ("the printout names no method, and its inputs fit "
                  + ("both elastic and consolidation" if has_es else
                     "neither elastic nor consolidation"))


#: A claimed settlement: the word, or the symbols rho, Se, delta, delta-h.
_SETTLEMENT_CLAIM = re.compile(
    r"settlement|^rho(_[a-z0-9]+)?\b|^s_?e\b|^delta[ _]?h?\b")


def _settlement_candidates(calc: Calculation) -> List[NamedQuantity]:
    """The printed results that claim a settlement: a settlement label and a
    length unit (a time, a ratio or a pressure named with the word is not
    one)."""
    return [nq for nq in calc.results
            if not _skip(nq, claim=True)
            and _SETTLEMENT_CLAIM.search(_lab(nq.name))
            and _si(nq, "length")[0] is not None]


def _settlement_claim(calc: Calculation, method: str):
    """``(the claimed settlement, None)`` or ``(None, why)``."""
    cands = _settlement_candidates(calc)
    if not cands:
        return None, "no settlement among the printed results"
    if len(cands) > 1:
        pat = (r"immediate|elastic" if method == "elastic"
               else r"consolidation|primary")
        named = [nq for nq in cands if re.search(pat, _lab(nq.name))]
        if len(named) == 1:
            return named[0], ""
        return None, ("more than one settlement printed ("
                      + "; ".join(f"'{nq.name}'" for nq in cands)
                      + "): which one this recompute matches is not clear")
    return cands[0], ""


def _stress_history(calc: Calculation, found: Dict[str, Dict[str, Any]],
                    problems: Dict[str, str], missing: List[str]) -> None:
    """sigma_p from a printed OCR, or the reason the row cannot be run.

    sigma_p = OCR x sigma_v0 is OCR's definition, taken only with the
    printed effective stress the module runs the layer at; a sigma_v0
    labelled as a total stress refuses the row. A printout that calls the
    soil overconsolidated and gives neither sigma_p nor OCR is not run as
    normally consolidated."""
    if "sigma_p" in found or "sigma_p" in problems:
        if "OCR" in found:
            found["OCR"]["note"] = "not used: sigma_p is printed"
        return
    ocr, sv0 = found.get("OCR"), found.get("sigma_v0")
    if ocr is not None:
        if sv0 is None:
            return                  # sigma_v0 is already named as missing
        if re.search(r"\btotal\b", _norm(sv0["label"])):
            missing.append(
                f"sigma_p: the printout gives OCR ('{ocr['label']}'), not "
                f"sigma_p, and its sigma_v0 ('{sv0['label']}') is a total "
                f"stress, not the effective stress sigma_p = OCR x sigma'_v0 "
                f"takes")
            return
        found["sigma_p"] = {
            "value": ocr["value"] * sv0["value"], "label": "OCR x sigma_v0",
            "printed": f"{ocr['printed']} x {sv0['printed']}",
            "page": ocr["page"],
            "note": (f"not printed: sigma_p = OCR x sigma_v0 (the definition "
                     f"of OCR), from '{ocr['label']}' and '{sv0['label']}', "
                     f"the stress the module runs the layer at")}
        return
    if "OCR" not in problems and re.search(r"\bover.?consolidated\b",
                                           _all_words(calc)):
        missing.append("sigma_p: the printout calls the soil "
                       "overconsolidated and prints neither sigma_p nor OCR "
                       "(normally consolidated would be a guess)")


def _recompute_settlement(calc: Calculation, index: int, tolerance: float,
                          gamma_prime: str = "refuse"
                          ) -> List[Dict[str, Any]]:
    method, how = _settlement_method(calc)
    if method is None:
        return [_not_row(calc, index, how, module="settlement")]
    nq, why = _settlement_claim(calc, method)
    if nq is None:
        return [_not_row(calc, index, why, status="not_compared",
                         module="settlement", method_used=method,
                         method_basis=how)]
    iw_options: List[Tuple[str, Dict[str, str]]] = []
    if method == "elastic":
        module_method = "elastic_settlement"
        other = [n.name for n in calc.inputs
                 if not _skip(n) and _OTHER_FACTOR.search(_lab(n.name))
                 and not _ELASTIC_INPUTS[-1].matches(n.name)]
        if other:
            return [_not_row(
                calc, index, "a factor that is not GEC-6 Eq. 5-36's Iw is "
                "printed (" + "; ".join(f"'{n}'" for n in other) + "): the "
                "report's method uses it, and the module's elastic_settlement "
                "(Se = q B (1 - nu^2) Iw / Es) has no place for it, so the "
                "two are not compared", module="settlement",
                module_method=module_method, method_used=method,
                method_basis=how)]
        found, problems = _collect(calc.inputs, _ELASTIC_INPUTS)
        missing = [f"{p}: {w}" for p, w in problems.items()]
        missing += [f"{p}: not printed" for p in ("q_net", "B", "Es")
                    if p not in found and p not in problems]
        params = {k: v["value"] for k, v in found.items()}
        shape_how = ""
        if "Iw" not in found:
            shape, shape_how = _shape(calc, found)
            if shape is None:
                missing.append(shape_how + " (or the influence factor Iw)")
            elif shape == "strip":
                missing.append("Iw: a strip has no GEC-6 Table 5-12 factor, "
                               "and none is printed")
            elif shape == "rectangular" and "L" not in found:
                missing.append("L: a rectangular footing needs L, not "
                               "printed")
            else:
                params["shape"] = shape
                if shape != "rectangular":
                    params.pop("L", None)
                words = _all_words(calc)
                if re.search(r"\brigid\b", words):
                    params["rigidity"] = "rigid"
                else:
                    # Table 5-12's other points beside the module's default
                    # (the centre of a flexible footing)
                    if not re.search(r"\bflexible\b", words):
                        iw_options.append(("rigidity 'rigid'",
                                           {"rigidity": "rigid"}))
                    iw_options.append(("point 'average'",
                                       {"point": "average"}))
        else:
            params.pop("L", None)
        key = "immediate_settlement_m"
    else:
        module_method = "consolidation_settlement"
        found, problems = _collect(calc.inputs, _CONSOLIDATION_INPUTS)
        missing = [f"{p}: {w}" for p, w in problems.items()]
        strain = [n.name for n in calc.inputs
                  if re.search(r"\bratio\b", _lab(n.name))
                  and re.search(r"compression|\bcr\b|\brr\b", _lab(n.name))]
        for p in _CONSOLIDATION_REQUIRED:
            if p not in found and p not in problems:
                missing.append(f"{p}: not printed" + (
                    " (the printout gives "
                    + ", ".join(f"'{n}'" for n in strain)
                    + ", a strain-based ratio; the module takes the indices "
                    "Cc and Cr)" if strain and p in ("Cc", "Cr") else ""))
        _stress_history(calc, found, problems, missing)
        layer = {k: found[k]["value"] for k in found
                 if k not in ("delta_sigma", "OCR")}
        params = {"layers": [layer],
                  "delta_sigma": found.get("delta_sigma", {}).get("value")}
        shape_how = ""
        key = "consolidation_settlement_m"
    if missing:
        return [_not_row(calc, index, "inputs not printed: "
                         + "; ".join(missing), inputs_missing=missing,
                         inputs_used=found, module="settlement",
                         module_method=module_method, method_used=method,
                         method_basis=how)]
    result = _call("settlement", module_method, params)
    row = _base_row(calc, index)
    row.update(module="settlement", module_method=module_method,
               method_used=method, method_basis=how, inputs_used=found,
               inputs_missing=[])
    if "error" in result:
        row.update(quantity=nq.name, status="not_recomputed",
                   reason=f"the module refused: {result['error']}")
        return [row]
    assumed = dict(result.get("defaults_used") or {})
    if method == "consolidation" and "sigma_p" not in found:
        assumed["sigma_p"] = ("not printed: run as normally consolidated, "
                              "the module's case for no sigma_p")
    row["assumed"] = assumed
    if shape_how:
        row["shape"] = f"{params.get('shape')} ({shape_how})"
    claimed, why = _si(nq, "length")
    if claimed is None:
        row.update(quantity=nq.name, status="not_compared", reason=why)
        return [row]
    default = ""
    if iw_options:
        row["alternatives"] = []
        for label, extra in iw_options:
            res = _call("settlement", module_method, {**params, **extra})
            alt = _alternative({"option": label, "result": res}, key, claimed)
            if "Iw" in res:
                alt["Iw"] = res["Iw"]
            row["alternatives"].append(alt)
        default = (f"Iw {result.get('Iw', 0):.3g} (GEC-6 Table 5-12, the "
                   f"centre of a flexible footing)")
    _compare(row, nq, claimed, float(result[key]), "m", tolerance,
             options=row.get("alternatives") or (), default=default)
    return [row]


_RECIPES: Dict[str, Callable[..., List[Dict[str, Any]]]] = {
    "shallow_foundation_bearing": _recompute_bearing,
    "settlement": _recompute_settlement,
}

#: Whether a calculation (or one case of it) prints a claim the recipe
#: compares: a case with none is not a row of its own.
_HAS_CLAIM: Dict[str, Callable[[Calculation], bool]] = {
    "shallow_foundation_bearing": lambda c: bool(_bearing_claims(c)),
    "settlement": lambda c: bool(_settlement_candidates(c)),
}


def _cases(calc: Calculation) -> List[Tuple[Dict[str, str], Calculation]]:
    """The calculation split by the case its labels print, or ``[]`` when
    it is one case.

    A case is each distinct combination (scenario, case, stratum, short or
    long term, shape) the printed RESULTS carry. Each case gets the results
    that carry exactly its combination, and every input whose own case does
    not contradict it: an input with no case word is shared, 'phi, stratum
    2' goes to every stratum-2 case, 'phi, stratum 1' to none of them.

    An input naming a dimension that the cases with a claim use reaches only
    the cases naming it with the same value: with claims for 'scenario 1'
    and 'scenario 1, stratum 2', 'phi, stratum 2' never reaches the
    'scenario 1' case, which names no stratum, so that case is not run on
    another stratum's input. An input naming a dimension no claim uses
    stays in every case, so two such values show as printed twice, never as
    a choice between them.

    A case is also printed by its NAME (:func:`_names_of`): a stratum by its
    soil, in any word order ('phi (<soil>)', '<soil> phi', 'ultimate
    bearing capacity, scenario 1, <soil>'). A name counts only when a
    claim's label and an input's label both print it, and it splits like
    the dimensions above (dimension ``name``)."""
    claim = _HAS_CLAIM[calc.kind]

    def read(rows):
        out = []
        for n in rows:
            case, lab = _case_of(n.name)
            out.append((dict(case), _names_of(lab), n))
        return out

    res_n, inp_n = read(calc.results), read(calc.inputs)
    claimed = {x for _c, names, n in res_n
               if claim(calc.model_copy(update={"results": [n]}))
               for x in names}
    shared = claimed & {x for _c, names, _n in inp_n for x in names}

    def named(case: Dict[str, str], names: List[str]) -> Dict[str, str]:
        mine = sorted(set(names) & shared)
        if mine:
            case["name"] = " / ".join(mine)
        return case

    res = [(named(c, names), n) for c, names, n in res_n]
    inp = [(named(c, names), n) for c, names, n in inp_n]
    combos: List[Dict[str, str]] = []
    for case, _n in res:
        if case not in combos:
            combos.append(case)
    if not combos or combos == [{}]:
        return []
    subs = [(combo, [n for case, n in res if case == combo])
            for combo in combos]
    in_play = {d for combo, results in subs
               if claim(calc.model_copy(update={"results": results}))
               for d in combo}

    def fits(case: Dict[str, str], combo: Dict[str, str]) -> bool:
        for d, v in case.items():
            if d in combo:
                if combo[d] != v:
                    return False
            elif d in in_play:
                return False
        return True

    return [(combo, calc.model_copy(update={
        "inputs": [n for case, n in inp if fits(case, combo)],
        "results": results}))
        for combo, results in subs]


# ---------------------------------------------------------------------------
# the public functions
# ---------------------------------------------------------------------------

def recompute_calculation(calc: Any, index: int = 0,
                          tolerance: float = TOLERANCE,
                          gamma_prime: Optional[str] = None
                          ) -> List[Dict[str, Any]]:
    """The rows for one calculation (a :class:`Calculation` or its dict).

    A calculation whose labels print several cases is recomputed case by
    case, each row naming its ``case``. ``gamma_prime``: the OWN-130 rule
    for a printout with only gamma' (``refuse`` or ``submerged``); None
    reads :data:`GAMMA_PRIME_ENV`, default ``refuse``."""
    rule = _gamma_prime_rule(gamma_prime)
    if not isinstance(calc, Calculation):
        calc = Calculation.model_validate(calc)
    recipe = _RECIPES.get(calc.kind)
    if recipe is None:
        return [_not_row(calc, index, NOT_RECOMPUTED.get(
            calc.kind, f"no recompute for kind {calc.kind!r}"))]
    try:
        cases = [(c, sub) for c, sub in _cases(calc)
                 if _HAS_CLAIM[calc.kind](sub)]
        if not cases:
            return recipe(calc, index, tolerance, gamma_prime=rule)
        rows: List[Dict[str, Any]] = []
        for case, sub in cases:
            name = _case_name(case)
            for row in recipe(sub, index, tolerance, gamma_prime=rule):
                if name:
                    row["case"] = name
                rows.append(row)
        return rows
    except Exception as exc:  # noqa: BLE001 - one bad calc never ends the table
        return [_not_row(calc, index, f"the recompute failed: "
                         f"{type(exc).__name__}: {exc}")]


def recompute_record(record: Any, tolerance: float = TOLERANCE,
                     gamma_prime: Optional[str] = None
                     ) -> Dict[str, Any]:
    """The claimed-against-recomputed table for one report.

    ``record``: a :class:`~report_ingest.model.ReportRecord`, its dict, or
    the path of ``report.record.json`` or the folder report ingest wrote it
    into. Returns ``{"report_id", "tolerance", "tolerance_basis",
    "gamma_prime_rule", "rows", "counts"}``: one row per claimed result, per
    case where the printout prints several (or one per calculation or case
    that was not recomputed). ``gamma_prime``: as in
    :func:`recompute_calculation`.
    """
    from report_ingest.checklist import load_record
    rule = _gamma_prime_rule(gamma_prime)
    rec = load_record(record)
    rows: List[Dict[str, Any]] = []
    for i, calc in enumerate(rec.calculations):
        rows += recompute_calculation(calc, i, tolerance, gamma_prime=rule)
    counts: Dict[str, int] = {"calculations": len(rec.calculations),
                              "rows": len(rows)}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        if row.get("flag"):
            counts[row["flag"]] = counts.get(row["flag"], 0) + 1
    return {"report_id": rec.document.report_id, "tolerance": tolerance,
            "tolerance_basis": TOLERANCE_BASIS, "gamma_prime_rule": rule,
            "rows": rows,
            "counts": counts}


def _fmt(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.4g}"


def table_markdown(table: Dict[str, Any]) -> str:
    """The table as Markdown, one line a row."""
    lines = [f"Recomputed calculations, report {table.get('report_id') or '?'}"
             f" (flag past {100 * table['tolerance']:g} %: "
             f"{table.get('tolerance_basis') or TOLERANCE_BASIS})", "",
             "| # | kind | quantity | claimed | recomputed (SI) | diff % | "
             "result | method | pages |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in table["rows"]:
        claimed = (r.get("claimed") or {}).get("printed", "")
        rec = (f"{_fmt(r.get('recomputed_si'))} {r.get('unit_si', '')}"
               if r.get("recomputed_si") is not None else "")
        diff = r.get("difference_pct")
        result = r.get("flag") or f"{r['status']}: {r.get('reason', '')}"
        method = r.get("method_used") or ""
        if r.get("method_basis"):
            method = f"{method} ({r['method_basis']})".strip()
        if r.get("assumed"):
            method = f"{method}; assumed: {', '.join(r['assumed'])}".strip(
                "; ")
        quantity = r.get("quantity", "")
        if r.get("case"):
            quantity = f"{quantity} [{r['case']}]".strip()
        lines.append("| " + " | ".join(str(x).replace("|", "/") for x in (
            r["calc_index"] + 1, r["kind"], quantity, claimed,
            rec.strip(), "" if diff is None else f"{diff:+.1f}", result,
            method, ", ".join(str(p) for p in r.get("pages") or []))) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the tool on the geotech page
# ---------------------------------------------------------------------------

RECORD_NAME = "report.record.json"

#: The shared report library's setting (``funhouse_agent.deep.tools.
#: REPORT_LIBRARY_ENV``): with it set, the ingest writes outside the working
#: folder.
_LIBRARY_ENV = "GEOTECH_REPORT_LIBRARY"

RECOMPUTE_TOOL_DESCRIPTION = (
    "Recompute the calculations a report claims, through this app's own "
    "modules, from a report already read by report_ingest. `record`: the "
    "record path the report_ingest tool returned (paths.record), or the "
    "folder it wrote (empty: the one record in the working folder). Returns "
    "one row per claimed result: the claimed value as printed with its "
    "page, the recomputed value in SI, the difference in per cent, and a "
    f"flag: 'differs' past {100 * TOLERANCE:g} % (this table's own QC "
    "threshold, not an engineering criterion) from every option the module "
    "ran; 'within_options' where the claim misses the module's default but "
    "falls inside the range of its own options run beside it (method, depth "
    "factors, influence factor); 'below' for an allowable pressure under "
    "q_ult/FS, which settlement may set. Each row gives the method used and "
    "why (printed, the module's default with its other options beside it, "
    "or inferred from the printed inputs), the inputs used with their pages, "
    "what the module assumed, and for a calculation not recomputed, why (an "
    "input not printed is named, never filled in). Bearing capacity and "
    "settlement are recomputed so far; slope stability is held. Show the "
    "user the table, the flagged rows first, with the reasons and "
    "assumptions as given.")


def find_record(where: str = "") -> str:
    """The record file ``where`` names (a file, a folder holding one, a path
    relative to the working folder), or with ``where`` empty the one record
    under the working folder. Raises ``ValueError`` saying why."""
    from funhouse_agent._fileio import host_output_dir
    folder = host_output_dir()
    if where:
        path = os.path.expanduser(str(where))
        if not os.path.isabs(path) and folder and not os.path.exists(path):
            path = os.path.join(folder, path)
        if os.path.isdir(path):
            path = os.path.join(path, RECORD_NAME)
        if not os.path.isfile(path):
            raise ValueError(f"no record at {where!r}: give the record path "
                             f"report_ingest returned (paths.record), or "
                             f"the folder it wrote")
        return path
    if not folder or not os.path.isdir(folder):
        raise ValueError("no working folder to look in: give the record "
                         "path report_ingest returned (paths.record)")
    hits = []
    for root, _dirs, files in os.walk(folder):
        if RECORD_NAME in files:
            hits.append(os.path.join(root, RECORD_NAME))
    if len(hits) == 1:
        return hits[0]
    if not hits and (os.environ.get(_LIBRARY_ENV) or "").strip():
        # the ingest wrote into the shared library, not the working folder
        raise ValueError(f"no {RECORD_NAME} in the working folder (report "
                         f"ingest writes to the shared report library): "
                         f"give the record path report_ingest returned "
                         f"(paths.record)")
    if not hits:
        raise ValueError(f"no {RECORD_NAME} in the working folder: read the "
                         f"report with report_ingest first, or give the "
                         f"record path report_ingest returned "
                         f"(paths.record)")
    rel = sorted(os.path.relpath(h, folder) for h in hits)
    raise ValueError(f"{len(hits)} records in the working folder; name one: "
                     + ", ".join(rel[:20]))


def _fit(table: Dict[str, Any], cap: int) -> str:
    text = json.dumps(table, ensure_ascii=False, default=str)
    if cap <= 0 or len(text) <= cap:
        return text
    # detail goes first: the inputs, then the alternatives and assumptions
    # of the rows not flagged "differs", then those rows' alternatives; a
    # "differs" row keeps what the module assumed
    for field, differs_too in (("inputs_used", True),
                               ("alternatives", False), ("assumed", False),
                               ("alternatives", True)):
        for row in table["rows"]:
            if differs_too or row.get("flag") != "differs":
                row.pop(field, None)
        text = json.dumps(table, ensure_ascii=False, default=str)
        if len(text) <= cap:
            return text
    rank = {flag: i for i, flag in enumerate(_FLAG_ORDER)}
    ordered = sorted(table["rows"],
                     key=lambda r: rank.get(r.get("flag"), len(rank)))
    keep: List[Dict[str, Any]] = []
    for row in ordered:
        trial = {**table, "rows": keep + [row]}
        if len(json.dumps(trial, ensure_ascii=False, default=str)) > cap - 200:
            break
        keep.append(row)
    table = {**table, "rows": keep,
             "cut": (f"{len(keep)} of {table['counts']['rows']} rows shown, "
                     f"flagged first; the rest did not fit")}
    return json.dumps(table, ensure_ascii=False, default=str)


def make_recompute_tool(max_result_chars: int = 16000):
    """The ``recompute_report_calcs`` tool (the geotech page, with
    report_ingest)."""
    from langchain_core.tools import StructuredTool

    def recompute_report_calcs(record: str = "") -> str:
        try:
            table = recompute_record(find_record(record))
        except ValueError as exc:
            return json.dumps({"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - reported to the model
            return json.dumps({"error": f"the recompute failed: "
                                        f"{type(exc).__name__}: {exc}"})
        return _fit(table, max_result_chars)

    return StructuredTool.from_function(
        recompute_report_calcs, name="recompute_report_calcs",
        description=RECOMPUTE_TOOL_DESCRIPTION)
