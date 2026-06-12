"""Deterministic Project validation — the agent's pre-flight checklist.

``validate(project)`` returns a list of :class:`Issue` (level 'error' |
'warning' | 'info'). Errors BLOCK ``project_run``; warnings/infos are shown
to the human at the confirmation gates.

Checks
------
Geometry      GEOM001 too few surface points; GEOM002 non-monotonic surface x;
              GEOM003 no layers; GEOM004 layer gap/overlap; GEOM005 layers do
              not reach the surface crest; GEOM006 section bottom above the
              surface low point (slip surfaces need depth); GEOM007 vision-
              draft geometry not yet human-confirmed (BLOCKING — the design
              inversion); GEOM008 undetermined layer elevations; GEOM009
              boundary polyline does not span the section width.
Water         WATER001 ponded GWT (info when declared via water.ponded,
              warning otherwise); WATER002 GWT entirely below the section;
              WATER003 ru out of range.
Loads         LOAD001 negative surcharge; LOAD002 multiple surcharges (LE
              maps the first only); LOAD003 surcharge band outside section;
              LOAD004 kh out of sane range.
Reinforcement REINF001 element outside the section.
Materials     MAT001 gamma missing/invalid; MAT002 mohr_coulomb incomplete;
              MAT003 undrained incomplete; MAT004 shansep incomplete (OCR
              needs a source!); MAT005 hoek_brown incomplete; MAT006 FEM
              stiffness (E, nu) missing; MAT007 FEM with shansep/hoek_brown
              layer (unsupported — pre-linearize to c-phi); MAT008 unknown
              probabilistic parameter key.
Unit sanity   UNIT001 gamma outside 10–25 kN/m3; UNIT002 phi outside 0–50
              deg; UNIT003 nu outside (0, 0.49); UNIT004 suspicious strength
              magnitude (c' > 1000 kPa / cu > 2000 kPa — psf or psi?);
              UNIT005 E outside 500–2,000,000 kPa.
Analyses      ANAL001 unknown analysis type / method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from geo_project.schema import (
    FEMAnalysis,
    LEAnalysis,
    Project,
)

#: Valid LE methods (slope_stability.analysis.analyze_slope).
LE_METHODS = ("fellenius", "bishop", "janbu", "spencer",
              "morgenstern_price", "gle")

#: Valid probabilistic variable parameters (slope_stability.probabilistic).
PROB_PARAMS = ("phi", "c_prime", "cu", "gamma")

_TOL = 1e-6


@dataclass
class Issue:
    """One validation finding."""
    level: str = "error"      # 'error' | 'warning' | 'info'
    code: str = ""
    message: str = ""
    fix_hint: str = ""

    def to_dict(self) -> dict:
        return {"level": self.level, "code": self.code,
                "message": self.message, "fix_hint": self.fix_hint}


def _err(code, msg, hint=""):
    return Issue("error", code, msg, hint)


def _warn(code, msg, hint=""):
    return Issue("warning", code, msg, hint)


def _info(code, msg, hint=""):
    return Issue("info", code, msg, hint)


def has_errors(issues: List[Issue]) -> bool:
    """True when any issue is level 'error' (blocks project_run)."""
    return any(i.level == "error" for i in issues)


def summarize(issues: List[Issue]) -> str:
    """One-line-per-issue plain-text report."""
    if not issues:
        return "Validation: clean (no issues)."
    lines = []
    for i in issues:
        line = f"[{i.level.upper()}] {i.code}: {i.message}"
        if i.fix_hint:
            line += f"  (fix: {i.fix_hint})"
        lines.append(line)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def validate(project: Project) -> List[Issue]:
    """Run all deterministic checks on a Project."""
    issues: List[Issue] = []
    issues += check_geometry(project)
    issues += check_vision_draft(project)
    issues += check_water(project)
    issues += check_loads(project)
    issues += check_reinforcement(project)
    issues += check_materials(project)
    issues += check_unit_sanity(project)
    issues += check_analyses(project)
    return issues


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def check_geometry(project: Project) -> List[Issue]:
    issues: List[Issue] = []
    pts = project.geometry.surface_points

    if len(pts) < 2:
        issues.append(_err(
            "GEOM001",
            f"Ground surface has {len(pts)} point(s); at least 2 required.",
            "Provide surface_points left-to-right as (x, z) pairs."))
        return issues  # nothing else is meaningful without a surface

    xs = [x for x, _ in pts]
    for i in range(len(xs) - 1):
        if xs[i] >= xs[i + 1]:
            issues.append(_err(
                "GEOM002",
                f"Surface x not strictly increasing at index {i} "
                f"(x[{i}]={xs[i]:g} >= x[{i + 1}]={xs[i + 1]:g}).",
                "Sort surface points by x and remove duplicates."))
            break

    layers = project.stratigraphy
    if not layers:
        issues.append(_err(
            "GEOM003", "No stratigraphy layers defined.",
            "Add at least one layer with a bottom elevation/boundary."))
        return issues

    z_min_surf, z_max_surf = project.geometry.z_surface_range
    x_min, x_max = project.geometry.x_range

    # Resolve representative elevations; report undetermined ones.
    tops, bots = [], []
    undetermined = False
    for i, layer in enumerate(layers):
        top = project.layer_top(i)
        bot = project.layer_bottom(i)
        if top is None or bot is None:
            issues.append(_err(
                "GEOM008",
                f"Layer '{layer.name or i}' has undetermined "
                f"{'top' if top is None else 'bottom'} elevation.",
                "Set bottom_elevation or reference a named boundary "
                "polyline in geometry.layer_boundaries."))
            undetermined = True
        tops.append(top)
        bots.append(bot)
    if undetermined:
        return issues

    # Coverage: first layer must reach the surface crest.
    if tops[0] < z_max_surf - _TOL:
        issues.append(_err(
            "GEOM005",
            f"Top layer '{layers[0].name or 0}' top ({tops[0]:g} m) is below "
            f"the surface crest ({z_max_surf:g} m) — the section is not "
            f"covered.",
            "Raise the top layer's top_elevation to the surface crest (or "
            "leave top_elevation None to track the surface)."))

    # Chain: each layer's bottom is the next layer's top (no gaps/overlaps).
    for i in range(len(layers) - 1):
        gap = tops[i + 1] - bots[i]
        if abs(gap) > _TOL:
            kind = "gap" if gap < 0 else "overlap"
            issues.append(_err(
                "GEOM004",
                f"Layer {kind} between '{layers[i].name or i}' (bottom "
                f"{bots[i]:g} m) and '{layers[i + 1].name or i + 1}' (top "
                f"{tops[i + 1]:g} m).",
                "Make each layer's bottom the next layer's top (leave "
                "top_elevation None to chain automatically)."))
        if bots[i] >= tops[i] - _TOL:
            issues.append(_err(
                "GEOM004",
                f"Layer '{layers[i].name or i}' has zero/negative thickness "
                f"(top {tops[i]:g} m, bottom {bots[i]:g} m).",
                "Check the layer's elevations."))

    # Section depth below the surface low point (slip surfaces need room).
    if bots[-1] > z_min_surf - _TOL:
        issues.append(_warn(
            "GEOM006",
            f"Section bottom ({bots[-1]:g} m) is at/above the surface low "
            f"point ({z_min_surf:g} m) — deep slip surfaces cannot form.",
            "Extend the bottom layer below the toe (>= 0.5x slope height "
            "is typical)."))

    # Boundary polylines should span the section width.
    for name, bpts in project.geometry.layer_boundaries.items():
        if not bpts:
            issues.append(_err(
                "GEOM009", f"Boundary '{name}' has no points.",
                "Add (x, z) points or remove the boundary."))
            continue
        bx = sorted(x for x, _ in bpts)
        if bx[0] > x_min + _TOL or bx[-1] < x_max - _TOL:
            issues.append(_warn(
                "GEOM009",
                f"Boundary '{name}' spans x=[{bx[0]:g}, {bx[-1]:g}] but the "
                f"section spans x=[{x_min:g}, {x_max:g}]; it will be "
                f"extrapolated flat beyond its ends.",
                "Extend the boundary polyline across the full section."))

    return issues


def check_vision_draft(project: Project) -> List[Issue]:
    """The design-inversion gate: vision geometry blocks until confirmed."""
    if (project.geometry.provenance == "vision_draft"
            and not project.confirmations.geometry):
        return [_err(
            "GEOM007",
            "Geometry was extracted by VISION (provenance='vision_draft') "
            "and has NOT been human-confirmed. Vision-read geometry is a "
            "draft, never trusted input.",
            "Render the echo-back cross-section (project_render), have the "
            "human verify it against the original drawing, then confirm "
            "the geometry stage (request_confirmation).")]
    return []


# ---------------------------------------------------------------------------
# Water
# ---------------------------------------------------------------------------

def check_water(project: Project) -> List[Issue]:
    issues: List[Issue] = []
    w = project.water
    pts = project.geometry.surface_points

    if not (0.0 <= w.ru < 1.0):
        issues.append(_err(
            "WATER003", f"ru = {w.ru:g} is outside [0, 1).",
            "ru is the pore-pressure ratio u/(gamma*z); typical 0-0.5."))

    if w.gwt_points and len(pts) >= 2:
        def surf_at(x):
            if x <= pts[0][0]:
                return pts[0][1]
            if x >= pts[-1][0]:
                return pts[-1][1]
            for i in range(len(pts) - 1):
                x0, z0 = pts[i]
                x1, z1 = pts[i + 1]
                if x0 <= x <= x1:
                    t = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
                    return z0 + t * (z1 - z0)
            return pts[-1][1]

        ponded_at = [x for x, z in w.gwt_points if z > surf_at(x) + _TOL]
        if ponded_at:
            if w.ponded:
                issues.append(_info(
                    "WATER001",
                    f"GWT is above the ground surface near x={ponded_at[0]:g}"
                    f" m (ponded water, declared intentional).",
                    ""))
            else:
                issues.append(_warn(
                    "WATER001",
                    f"GWT is above the ground surface near x={ponded_at[0]:g}"
                    f" m — ponded water. Intentional?",
                    "Set water.ponded=true to declare intentional ponding, "
                    "or correct the GWT elevations."))

        bot = project.section_bottom()
        if bot is not None and all(z < bot - _TOL for _, z in w.gwt_points):
            issues.append(_warn(
                "WATER002",
                "GWT is entirely below the modeled section — it has no "
                "effect.",
                "Remove the GWT or check its elevations."))

    return issues


# ---------------------------------------------------------------------------
# Loads
# ---------------------------------------------------------------------------

def check_loads(project: Project) -> List[Issue]:
    issues: List[Issue] = []
    x_min, x_max = project.geometry.x_range

    for k, s in enumerate(project.loads.surcharges):
        label = s.label or f"surcharge[{k}]"
        if s.q < 0:
            issues.append(_err(
                "LOAD001", f"{label}: negative surcharge q={s.q:g} kPa.",
                "Surcharge must be >= 0."))
        if s.q > 1000:
            issues.append(_warn(
                "LOAD002", f"{label}: q={s.q:g} kPa is unusually large.",
                "Check units (kPa expected; 1000 kPa = ~20 ksf)."))
        if (s.x_start is not None and s.x_end is not None
                and project.geometry.surface_points):
            if s.x_end <= s.x_start:
                issues.append(_err(
                    "LOAD003", f"{label}: x_end <= x_start.",
                    "Give x_start < x_end."))
            elif s.x_end < x_min or s.x_start > x_max:
                issues.append(_err(
                    "LOAD003",
                    f"{label}: band [{s.x_start:g}, {s.x_end:g}] lies "
                    f"outside the section x=[{x_min:g}, {x_max:g}].",
                    "Move the band onto the section."))

    if len(project.loads.surcharges) > 1:
        issues.append(_warn(
            "LOAD002",
            f"{len(project.loads.surcharges)} surcharges defined; the LE "
            "model applies only the FIRST (SlopeGeometry carries one "
            "uniform band).",
            "Combine bands or keep the governing one first."))

    if not (0.0 <= project.loads.kh <= 0.5):
        level = _err if project.loads.kh < 0 else _warn
        issues.append(level(
            "LOAD004", f"kh = {project.loads.kh:g} is outside 0-0.5.",
            "Typical pseudo-static kh is 0.05-0.3."))

    return issues


# ---------------------------------------------------------------------------
# Reinforcement
# ---------------------------------------------------------------------------

def check_reinforcement(project: Project) -> List[Issue]:
    issues: List[Issue] = []
    if not project.geometry.surface_points:
        return issues
    x_min, x_max = project.geometry.x_range
    z_min_surf, z_max_surf = project.geometry.z_surface_range
    bot = project.section_bottom()
    z_lo = bot if bot is not None else z_min_surf - 1e9

    def outside(x, z):
        return not (x_min - _TOL <= x <= x_max + _TOL
                    and z_lo - _TOL <= z <= z_max_surf + _TOL)

    for i, n in enumerate(project.reinforcement.nails):
        if outside(n.x_head, n.z_head):
            issues.append(_err(
                "REINF001",
                f"Nail {i} head ({n.x_head:g}, {n.z_head:g}) is outside the "
                f"section.",
                "Place the head on the slope face inside the section."))
    for i, a in enumerate(project.reinforcement.anchors):
        if outside(a.x_head, a.z_head):
            issues.append(_err(
                "REINF001",
                f"Anchor {i} head ({a.x_head:g}, {a.z_head:g}) is outside "
                f"the section.",
                "Place the head on the wall/slope face inside the section."))
    for i, g in enumerate(project.reinforcement.geosynthetics):
        if not (z_lo - _TOL <= g.elevation <= z_max_surf + _TOL):
            issues.append(_err(
                "REINF001",
                f"Geosynthetic {i} elevation {g.elevation:g} m is outside "
                f"the section [{z_lo:g}, {z_max_surf:g}].",
                "Set the layer elevation within the section."))
    return issues


# ---------------------------------------------------------------------------
# Materials (completeness FOR THE REQUESTED ANALYSES)
# ---------------------------------------------------------------------------

def _requested(project: Project):
    le = any(isinstance(a, LEAnalysis) for a in project.analyses)
    fem = any(isinstance(a, FEMAnalysis) for a in project.analyses)
    return le, fem


def check_materials(project: Project) -> List[Issue]:
    issues: List[Issue] = []
    le, fem = _requested(project)
    if not (le or fem):
        return issues  # nothing requested yet — incompleteness is fine

    for i, layer in enumerate(project.stratigraphy):
        m = layer.material
        name = layer.name or f"layer[{i}]"

        if m.gamma is None or m.gamma <= 0:
            issues.append(_err(
                "MAT001",
                f"{name}: unit weight gamma is "
                f"{'missing' if m.gamma is None else 'non-positive'}.",
                "Set material.gamma (kN/m3); typical soil 16-22."))

        if m.strength_model == "mohr_coulomb":
            if m.phi is None or m.c_prime is None:
                issues.append(_err(
                    "MAT002",
                    f"{name}: mohr_coulomb needs BOTH phi and c_prime set "
                    f"(missing: "
                    f"{', '.join(p for p, v in (('phi', m.phi), ('c_prime', m.c_prime)) if v is None)}).",
                    "Set both (0 is allowed but must be deliberate)."))
        elif m.strength_model == "undrained":
            if m.cu is None or m.cu <= 0:
                issues.append(_err(
                    "MAT003",
                    f"{name}: undrained model needs cu > 0 "
                    f"({'missing' if m.cu is None else f'cu={m.cu:g}'}).",
                    "Set material.cu (kPa)."))
        elif m.strength_model == "shansep":
            missing = [p for p, v in (("shansep_S", m.shansep_S),
                                      ("shansep_m", m.shansep_m),
                                      ("ocr", m.ocr)) if v is None]
            if missing:
                issues.append(_err(
                    "MAT004",
                    f"{name}: SHANSEP needs {', '.join(missing)}. The OCR "
                    f"must come from a documented source (consolidation "
                    f"tests / CPTu / geologic history).",
                    "Set shansep_S (~0.22), shansep_m (~0.8) and ocr; cite "
                    "the OCR source in the assumption ledger."))
            else:
                if m.shansep_S <= 0 or not (0.0 < m.shansep_m <= 1.5) \
                        or m.ocr < 1.0:
                    issues.append(_err(
                        "MAT004",
                        f"{name}: SHANSEP parameters out of range "
                        f"(S={m.shansep_S:g}, m={m.shansep_m:g}, "
                        f"OCR={m.ocr:g}).",
                        "Require S>0, 0<m<=1.5, OCR>=1."))
        elif m.strength_model == "hoek_brown":
            missing = [p for p, v in (("hb_sigci", m.hb_sigci),
                                      ("hb_gsi", m.hb_gsi),
                                      ("hb_mi", m.hb_mi)) if v is None]
            if missing:
                issues.append(_err(
                    "MAT005",
                    f"{name}: Hoek-Brown needs {', '.join(missing)}.",
                    "Set hb_sigci (intact UCS, kPa), hb_gsi (0-100], hb_mi; "
                    "hb_D defaults to 0."))
            else:
                if (m.hb_sigci <= 0 or not (0.0 < m.hb_gsi <= 100.0)
                        or m.hb_mi <= 0 or not (0.0 <= m.hb_D <= 1.0)):
                    issues.append(_err(
                        "MAT005",
                        f"{name}: Hoek-Brown parameters out of range "
                        f"(sigci={m.hb_sigci:g}, GSI={m.hb_gsi:g}, "
                        f"mi={m.hb_mi:g}, D={m.hb_D:g}).",
                        "Require sigci>0, 0<GSI<=100, mi>0, 0<=D<=1."))

        if fem:
            if m.E is None or m.nu is None:
                issues.append(_err(
                    "MAT006",
                    f"{name}: FEM (SRM) requested but stiffness is "
                    f"incomplete (missing: "
                    f"{', '.join(p for p, v in (('E', m.E), ('nu', m.nu)) if v is None)}).",
                    "Set material.E (kPa) and material.nu."))
            if m.strength_model in ("shansep", "hoek_brown"):
                issues.append(_err(
                    "MAT007",
                    f"{name}: FEM-SRM supports Mohr-Coulomb strengths only; "
                    f"'{m.strength_model}' is not directly supported.",
                    "Linearize to an equivalent c-phi (or cu) for the FEM "
                    "run, or drop the FEM analysis."))

        for key in m.probabilistic:
            if key.split(":", 1)[0] not in PROB_PARAMS:
                issues.append(_err(
                    "MAT008",
                    f"{name}: probabilistic key '{key}' is not one of "
                    f"{PROB_PARAMS}.",
                    "Use 'phi', 'c_prime', 'cu' or 'gamma'."))

    return issues


# ---------------------------------------------------------------------------
# Unit-sanity heuristics
# ---------------------------------------------------------------------------

def check_unit_sanity(project: Project) -> List[Issue]:
    issues: List[Issue] = []
    for i, layer in enumerate(project.stratigraphy):
        m = layer.material
        name = layer.name or f"layer[{i}]"
        if m.gamma is not None and m.gamma > 0 and not (10.0 <= m.gamma <= 25.0):
            issues.append(_warn(
                "UNIT001",
                f"{name}: gamma={m.gamma:g} kN/m3 is outside the typical "
                f"10-25 range.",
                "Check units (kN/m3 expected; 120 pcf = 18.9 kN/m3)."))
        if m.phi is not None and not (0.0 <= m.phi < 50.0):
            issues.append(_warn(
                "UNIT002",
                f"{name}: phi={m.phi:g} deg is outside 0-50.",
                "Check the value (degrees expected)."))
        if m.nu is not None and not (0.0 < m.nu < 0.49):
            issues.append(_warn(
                "UNIT003",
                f"{name}: nu={m.nu:g} is outside (0, 0.49).",
                "Poisson's ratio for soils is typically 0.2-0.4."))
        if m.c_prime is not None and m.c_prime > 1000:
            issues.append(_warn(
                "UNIT004",
                f"{name}: c'={m.c_prime:g} kPa is unusually large.",
                "Check units (kPa expected; 1000 psf = 47.9 kPa)."))
        if m.cu is not None and m.cu > 2000:
            issues.append(_warn(
                "UNIT004",
                f"{name}: cu={m.cu:g} kPa is unusually large.",
                "Check units (kPa expected)."))
        if m.E is not None and not (500.0 <= m.E <= 2.0e6):
            issues.append(_warn(
                "UNIT005",
                f"{name}: E={m.E:g} kPa is outside 500-2,000,000.",
                "Check units (kPa expected; 30 MPa = 30000 kPa)."))
    return issues


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------

def check_analyses(project: Project) -> List[Issue]:
    issues: List[Issue] = []
    for k, a in enumerate(project.analyses):
        if isinstance(a, LEAnalysis):
            if a.method not in LE_METHODS:
                issues.append(_err(
                    "ANAL001",
                    f"analyses[{k}]: unknown LE method '{a.method}'.",
                    f"Use one of {LE_METHODS}."))
            if a.probabilistic is not None:
                if a.probabilistic.kind not in ("fosm", "monte_carlo"):
                    issues.append(_err(
                        "ANAL001",
                        f"analyses[{k}]: unknown probabilistic kind "
                        f"'{a.probabilistic.kind}'.",
                        "Use 'fosm' or 'monte_carlo'."))
                for key in a.probabilistic.variables:
                    if key.split(":", 1)[0] not in PROB_PARAMS:
                        issues.append(_err(
                            "ANAL001",
                            f"analyses[{k}]: probabilistic variable '{key}' "
                            f"is not one of {PROB_PARAMS}.",
                            "Use 'phi', 'c_prime', 'cu' or 'gamma', "
                            "optionally ':LayerName'."))
        elif isinstance(a, FEMAnalysis):
            if a.element_type not in ("t6", "cst"):
                issues.append(_err(
                    "ANAL001",
                    f"analyses[{k}]: unknown element_type "
                    f"'{a.element_type}'.",
                    "Use 't6' (recommended) or 'cst'."))
        else:
            issues.append(_err(
                "ANAL001",
                f"analyses[{k}]: unknown analysis object "
                f"{type(a).__name__}.",
                "Use LEAnalysis or FEMAnalysis."))
    return issues


__all__ = ["Issue", "validate", "has_errors", "summarize",
           "LE_METHODS", "PROB_PARAMS"]
