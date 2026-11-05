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

First slice: ``shallow_foundation_bearing`` and ``settlement`` (elastic, and
one-layer Cc/Cr consolidation). Every other kind is a row saying why it was
not recomputed; ``slope_stability`` is held (GM-140).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from report_ingest.model import Calculation, NamedQuantity, calc_si

__all__ = ["TOLERANCE", "NOT_RECOMPUTED", "recompute_record",
           "recompute_calculation", "table_markdown", "find_record",
           "make_recompute_tool", "RECOMPUTE_TOOL_DESCRIPTION"]

#: A row is flagged ``differs`` when the recomputed value is more than this
#: fraction away from the claimed one. A QC threshold of this table, not an
#: engineering value: printed factors are often read off charts and rounded,
#: and a few per cent between two honest calculations is common.
TOLERANCE = 0.10

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
          "Δ": "delta", "ν": "nu", "′": "'", "’": "'"}


def _norm(text: str) -> str:
    out = str(text or "")
    for k, v in _GREEK.items():
        out = out.replace(k, v)
    return " ".join(out.lower().split())


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
        lab = _norm(label)
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
    """``(shape, how)`` from the printed words, else from L against B; None
    with the reason when the printout settles neither."""
    words = _all_words(calc)
    named = [s for s, pat in _SHAPE_WORDS if re.search(pat, words)]
    if len(named) == 1:
        return named[0], f"printed: {named[0]}"
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


def _compare(row: Dict[str, Any], nq: NamedQuantity, claimed: float,
             recomputed: float, unit: str, tolerance: float) -> None:
    diff = None if claimed == 0 else (recomputed - claimed) / abs(claimed)
    row.update(
        quantity=nq.name,
        claimed={"printed": _printed(nq), "page": _page(nq)},
        claimed_si=round(claimed, 6), recomputed_si=round(recomputed, 6),
        unit_si=unit,
        difference_pct=None if diff is None else round(100 * diff, 2),
        flag=("differs" if diff is None and recomputed != 0
              or diff is not None and abs(diff) > tolerance else "agrees"),
        status="recomputed", reason="")


def _call(module: str, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
    from funhouse_agent.dispatch import call_agent
    return call_agent(module, method, params)


def _basis(result: Dict[str, Any], what: str) -> str:
    for c in result.get("choices") or []:
        if what in str(c.get("what", "")):
            return str(c.get("basis", ""))
    return ""


# -- bearing capacity -------------------------------------------------------

_BEARING_INPUTS = (
    _Spec("width", "length", [r"\bwidth\b", r"^b\b"],
          [r"effective", r"b'"]),
    _Spec("length", "length", [r"\blength\b", r"^l\b"],
          [r"effective", r"l'"]),
    _Spec("depth", "length",
          [r"embedment", r"\b(footing|foundation) depth\b",
           r"\bdepth of (the )?(footing|foundation)\b", r"\bdf\b",
           r"\bd_f\b"], [r"water", r"gwt"]),
    _Spec("friction_angle", "angle",
          [r"friction angle", r"angle of (internal )?friction", r"^phi\b",
           r"\bphi'?\b"]),
    _Spec("cohesion", "stress",
          [r"cohesion", r"^c'?\b", r"undrained shear strength", r"\bsu\b",
           r"\bcu\b"]),
    _Spec("unit_weight", "unit_weight", [r"unit weight", r"^gamma\b"],
          [r"effective", r"buoyant", r"submerged", r"water"]),
    _Spec("gwt_depth", "length",
          [r"water table", r"groundwater", r"\bgwt\b", r"\bd_?w\b"]),
    _Spec("factor_of_safety", "ratio",
          [r"factor of safety", r"\bf\.?s\.?\b", r"\bfos\b"]),
)

#: Claimed results, most specific first: the module's result key and the
#: printed labels that claim it.
_BEARING_RESULTS = (
    ("q_allowable_net_kPa", r"allowable", r"\bnet\b"),
    ("q_allowable_kPa", r"allowable", None),
    ("q_ultimate_kPa", r"ultimate", None),
)

_BEARING_METHODS = ("vesic", "meyerhof", "terzaghi")


def _bearing_claims(calc: Calculation):
    out = []
    for nq in calc.results:
        lab = _norm(nq.name)
        if nq.value is None or "factor" in lab:
            continue
        for key, must, also in _BEARING_RESULTS:
            if re.search(must, lab) and (also is None or re.search(also, lab)):
                if key == "q_allowable_kPa" and re.search(r"\bnet\b", lab):
                    continue
                out.append((key, nq))
                break
    return out


def _alternative(alt: Dict[str, Any], key: str, claimed: float
                 ) -> Dict[str, Any]:
    """One of the module's other cited methods, and what it gives."""
    res = alt["result"]
    if "error" in res:
        return {"method": alt["method"], "refused": str(res["error"])[:300]}
    value = float(res[key])
    out = {"method": alt["method"], "recomputed_si": round(value, 6)}
    if claimed:
        out["difference_pct"] = round(100 * (value - claimed) / abs(claimed),
                                      2)
    return out


def _recompute_bearing(calc: Calculation, index: int, tolerance: float
                       ) -> List[Dict[str, Any]]:
    claims = _bearing_claims(calc)
    if not claims:
        return [_not_row(calc, index, status="not_compared", reason=(
            "no ultimate or allowable bearing pressure among the printed "
            "results"))]
    words = _method_words(calc)
    if re.search(r"two.?layer|layered", words):
        return [_not_row(calc, index, "a two-layer bearing case: a later "
                         "slice maps its second layer")]
    found, problems = _collect(calc.inputs, _BEARING_INPUTS)
    missing = [f"{p}: {why}" for p, why in problems.items()]
    for need in ("width", "depth", "unit_weight"):
        if need not in found and need not in problems:
            missing.append(f"{need}: not printed")
    if ("friction_angle" not in found and "cohesion" not in found
            and not {"friction_angle", "cohesion"} & set(problems)):
        missing.append("friction_angle or cohesion: neither printed")
    shape, shape_how = _shape(calc, found)
    if shape is None:
        missing.append(shape_how)
    elif shape == "rectangular" and "length" not in found:
        missing.append("length: a rectangular footing needs L, not printed")
    if missing:
        return [_not_row(calc, index, "inputs not printed: "
                         + "; ".join(missing), inputs_missing=missing,
                         inputs_used=found, module="bearing_capacity",
                         module_method="bearing_capacity_analysis")]
    params: Dict[str, Any] = {k: v["value"] for k, v in found.items()}
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
                         module_method="bearing_capacity_analysis")]
    used = result.get("factor_method", method or "vesic")
    alternatives = []
    if method is None:
        for other in _BEARING_METHODS:
            if other == used:
                continue
            alt = _call("bearing_capacity", "bearing_capacity_analysis",
                        {**params, "factor_method": other})
            alternatives.append({"method": other, "result": alt})
    rows = []
    for key, nq in claims:
        row = _base_row(calc, index)
        row.update(module="bearing_capacity",
                   module_method="bearing_capacity_analysis",
                   method_used=used, method_basis=how,
                   method_reference=_basis(result, "bearing capacity method"),
                   inputs_used=found, inputs_missing=[],
                   assumed=result.get("defaults_used") or {},
                   shape=f"{shape} ({shape_how})")
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
        _compare(row, nq, claimed, float(result[key]), "kPa", tolerance)
        if key == "q_allowable_kPa":
            row["note"] = ("compared with the gross allowable q_ult / FS "
                           "(UFC 3-220-20 Eq. 5-13); a net allowable label "
                           "says 'net'")
        row["alternatives"] = [_alternative(a, key, claimed)
                               for a in alternatives]
        rows.append(row)
    return rows


# -- settlement --------------------------------------------------------------

_ELASTIC_INPUTS = (
    _Spec("q_net", "stress",
          [r"pressure", r"\bbearing stress\b", r"^q\b", r"^q_?net\b",
           r"load intensity", r"applied stress", r"contact stress"],
          [r"preconsolidation", r"overburden", r"effective", r"initial",
           r"allowable", r"ultimate", r"increase", r"added", r"pore"]),
    _Spec("B", "length", [r"\bwidth\b", r"^b\b"], [r"effective", r"b'"]),
    _Spec("L", "length", [r"\blength\b", r"^l\b"], [r"effective", r"l'"]),
    _Spec("Es", "stress",
          [r"modulus", r"^e_?s\b", r"^e\b", r"young"],
          [r"subgrade", r"reaction", r"shear modulus"]),
    _Spec("nu", "ratio", [r"poisson", r"^nu\b"]),
    _Spec("Iw", "ratio", [r"influence factor", r"^i_?w\b"]),
)

_CONSOLIDATION_INPUTS = (
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
           r"^sigma'?_?v ?0\b", r"^sigma'?_?vo\b", r"^p_?0\b", r"^p_?o\b"]),
    _Spec("thickness", "length", [r"thickness", r"^h\b"]),
    _Spec("depth_to_center", "length",
          [r"depth to (the )?(mid|cent)", r"mid.?depth", r"(center|centre) "
           r"(depth|of (the )?layer)"]),
    _Spec("e0", "ratio",
          [r"void ratio", r"^e_?0\b", r"^e_?o\b"]),
    _Spec("Cr", "ratio",
          [r"recompression", r"swelling", r"\bcr\b", r"\bcs\b"]),
    _Spec("Cc", "ratio", [r"compression index", r"\bcc\b"]),
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


def _settlement_claim(calc: Calculation, method: str):
    """``(the claimed settlement, None)`` or ``(None, why)``."""
    cands = [nq for nq in calc.results
             if nq.value is not None and "settlement" in _norm(nq.name)]
    if not cands:
        return None, "no settlement among the printed results"
    if len(cands) > 1:
        pat = (r"immediate|elastic" if method == "elastic"
               else r"consolidation|primary")
        named = [nq for nq in cands if re.search(pat, _norm(nq.name))]
        if len(named) == 1:
            return named[0], ""
        return None, ("more than one settlement printed ("
                      + "; ".join(f"'{nq.name}'" for nq in cands)
                      + "): which one this recompute matches is not clear")
    return cands[0], ""


def _recompute_settlement(calc: Calculation, index: int, tolerance: float
                          ) -> List[Dict[str, Any]]:
    method, how = _settlement_method(calc)
    if method is None:
        return [_not_row(calc, index, how, module="settlement")]
    nq, why = _settlement_claim(calc, method)
    if nq is None:
        return [_not_row(calc, index, why, status="not_compared",
                         module="settlement", method_used=method,
                         method_basis=how)]
    if method == "elastic":
        module_method = "elastic_settlement"
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
            params.pop("L", None)
        key = "immediate_settlement_m"
    else:
        module_method = "consolidation_settlement"
        found, problems = _collect(calc.inputs, _CONSOLIDATION_INPUTS)
        missing = [f"{p}: {w}" for p, w in problems.items()]
        missing += [f"{p}: not printed" for p in _CONSOLIDATION_REQUIRED
                    if p not in found and p not in problems]
        layer = {k: found[k]["value"] for k in found
                 if k not in ("delta_sigma",)}
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
    _compare(row, nq, claimed, float(result[key]), "m", tolerance)
    return [row]


_RECIPES: Dict[str, Callable[[Calculation, int, float],
                             List[Dict[str, Any]]]] = {
    "shallow_foundation_bearing": _recompute_bearing,
    "settlement": _recompute_settlement,
}


# ---------------------------------------------------------------------------
# the public functions
# ---------------------------------------------------------------------------

def recompute_calculation(calc: Any, index: int = 0,
                          tolerance: float = TOLERANCE
                          ) -> List[Dict[str, Any]]:
    """The rows for one calculation (a :class:`Calculation` or its dict)."""
    if not isinstance(calc, Calculation):
        calc = Calculation.model_validate(calc)
    recipe = _RECIPES.get(calc.kind)
    if recipe is None:
        return [_not_row(calc, index, NOT_RECOMPUTED.get(
            calc.kind, f"no recompute for kind {calc.kind!r}"))]
    try:
        return recipe(calc, index, tolerance)
    except Exception as exc:  # noqa: BLE001 - one bad calc never ends the table
        return [_not_row(calc, index, f"the recompute failed: "
                         f"{type(exc).__name__}: {exc}")]


def recompute_record(record: Any, tolerance: float = TOLERANCE
                     ) -> Dict[str, Any]:
    """The claimed-against-recomputed table for one report.

    ``record``: a :class:`~report_ingest.model.ReportRecord`, its dict, or
    the path of ``report.record.json`` or the folder report ingest wrote it
    into. Returns ``{"report_id", "tolerance", "rows", "counts"}``: one row
    per claimed result (or one per calculation that was not recomputed).
    """
    from report_ingest.checklist import load_record
    rec = load_record(record)
    rows: List[Dict[str, Any]] = []
    for i, calc in enumerate(rec.calculations):
        rows += recompute_calculation(calc, i, tolerance)
    counts: Dict[str, int] = {"calculations": len(rec.calculations),
                              "rows": len(rows)}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
        if row.get("flag"):
            counts[row["flag"]] = counts.get(row["flag"], 0) + 1
    return {"report_id": rec.document.report_id, "tolerance": tolerance,
            "rows": rows, "counts": counts}


def _fmt(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.4g}"


def table_markdown(table: Dict[str, Any]) -> str:
    """The table as Markdown, one line a row."""
    lines = [f"Recomputed calculations, report {table.get('report_id') or '?'}"
             f" (flag past {100 * table['tolerance']:g} %)", "",
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
        lines.append("| " + " | ".join(str(x).replace("|", "/") for x in (
            r["calc_index"] + 1, r["kind"], r.get("quantity", ""), claimed,
            rec.strip(), "" if diff is None else f"{diff:+.1f}", result,
            method, ", ".join(str(p) for p in r.get("pages") or []))) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# the tool on the geotech page
# ---------------------------------------------------------------------------

RECORD_NAME = "report.record.json"

RECOMPUTE_TOOL_DESCRIPTION = (
    "Recompute the calculations a report claims, through this app's own "
    "modules, from a report already read by report_ingest. Give `record`: "
    "the path of its report.record.json or of the folder report_ingest "
    "wrote it into (empty: the one record in the working folder). Returns "
    "one row per claimed result: the claimed value as printed with its "
    "page, the recomputed value in SI, the difference in per cent, flag "
    "'differs' past the tolerance, the method used and why (printed, the "
    "module's default with its other cited options beside it, or inferred "
    "from the printed inputs), the inputs used with their pages, what the "
    "module assumed, and for a calculation not recomputed, why (an input "
    "not printed is named, never filled in). Bearing capacity and "
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
            raise ValueError(f"no record at {where!r}: give the path of the "
                             f"{RECORD_NAME} report_ingest wrote, or of its "
                             f"folder")
        return path
    if not folder or not os.path.isdir(folder):
        raise ValueError("no working folder to look in: give the path of "
                         f"the {RECORD_NAME} report_ingest wrote")
    hits = []
    for root, _dirs, files in os.walk(folder):
        if RECORD_NAME in files:
            hits.append(os.path.join(root, RECORD_NAME))
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ValueError(f"no {RECORD_NAME} in the working folder: read the "
                         f"report with report_ingest first")
    rel = sorted(os.path.relpath(h, folder) for h in hits)
    raise ValueError(f"{len(hits)} records in the working folder; name one: "
                     + ", ".join(rel[:20]))


def _fit(table: Dict[str, Any], cap: int) -> str:
    text = json.dumps(table, ensure_ascii=False, default=str)
    if cap <= 0 or len(text) <= cap:
        return text
    for field in ("assumed", "alternatives", "inputs_used"):
        for row in table["rows"]:
            row.pop(field, None)
        text = json.dumps(table, ensure_ascii=False, default=str)
        if len(text) <= cap:
            return text
    flagged = [r for r in table["rows"] if r.get("flag") == "differs"]
    rest = [r for r in table["rows"] if r.get("flag") != "differs"]
    keep: List[Dict[str, Any]] = []
    for row in flagged + rest:
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
