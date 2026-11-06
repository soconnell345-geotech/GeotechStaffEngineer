"""
Calculation package steps for retaining wall analysis.

Provides input summaries, step-by-step equation output, and figures
for Mathcad-style calc package generation.

Handles both cantilever retaining walls and MSE walls.

The `analysis` parameter should be a dict with:
  Common keys:
    - "wall_type": "cantilever" or "mse"
    - "geom": CantileverWallGeometry or MSEWallGeometry
    - "gamma_backfill": float (kN/m3)
    - "phi_backfill": float (degrees)
    - "c_backfill": float (kPa, default 0)
    - "pressure_method": "rankine" or "coulomb" (cantilever only, default "rankine")
  Cantilever-specific:
    - "gamma_concrete": float (kN/m3, default 23.6)
    - "phi_foundation": float (degrees, optional)
    - "c_foundation": float (kPa, default 0)
    - "q_allowable": float (kPa, optional)
    - "FOS_sliding": float (default 1.5, 2.0 with include_passive)
    - "FOS_overturning": float (default 2.0, 1.5 on rock)
    - and the other inputs of ``analyze_cantilever_wall`` (GM-288:
      "wall_friction_deg", "interface_gap_possible", "founded_on_rock",
      "bearing_pressure"; GM-292: "gamma_sat_backfill", "base_uplift",
      "uplift_in_bearing"), each at its default when left out
  MSE-specific:
    - "reinforcement": Reinforcement dataclass
    - "gamma_foundation": float (kN/m3, optional)
    - "phi_foundation": float (degrees, optional)
    - "c_foundation": float (kPa, default 0)
    - "q_allowable": float (kPa, optional)

References:
    FHWA GEC-11 (FHWA-NHI-10-024) — MSE Walls and Reinforced Slopes
    AASHTO LRFD Bridge Design Specifications, Section 11
    Das, B.M., Principles of Foundation Engineering, Ch 13
    Coulomb (1776), Rankine (1857) — Earth pressure theories
"""

import math
from typing import List

from calc_package.data_model import (
    InputItem, CalcStep, CheckItem, FigureData, CalcSection, TableData,
)
from calc_package.renderer import figure_to_base64

DISPLAY_NAME = "Retaining Wall Stability Analysis"

REFERENCES = [
    'FHWA GEC-11 (FHWA-NHI-10-024): Design of Mechanically Stabilized '
    'Earth Walls and Reinforced Slopes. FHWA, 2009.',
    'AASHTO LRFD Bridge Design Specifications, Sections 11.6 and 11.10.',
    'Das, B.M. Principles of Foundation Engineering, 9th Ed., Ch. 13.',
    'Coulomb, C.A. (1776). "Essai sur une application des regles de '
    'maximis et minimis a quelques problemes de statique." Memoires de '
    'Mathematique et de Physique, Academie Royale des Sciences.',
    'Rankine, W.J.M. (1857). "On the Stability of Loose Earth." '
    'Philosophical Transactions of the Royal Society, 147.',
]


# ── Helper to detect wall type ──────────────────────────────────────
def _wall_type(result, analysis) -> str:
    """Return 'cantilever' or 'mse' based on analysis dict or result class."""
    if isinstance(analysis, dict) and "wall_type" in analysis:
        return analysis["wall_type"]
    # Fall back to result class inspection
    from retaining_walls.results import CantileverWallResult, MSEWallResult
    if isinstance(result, MSEWallResult):
        return "mse"
    return "cantilever"


def _get(analysis, key, default=None):
    """Safely get a value from analysis dict."""
    if isinstance(analysis, dict):
        return analysis.get(key, default)
    return getattr(analysis, key, default)


def _used(analysis, key):
    """An input the cantilever run used: the analysis dict's, else the
    default of ``analyze_cantilever_wall``, which the run then took
    (GM-141: the steps had defaults of their own)."""
    if isinstance(analysis, dict) and key in analysis:
        return analysis[key]
    import inspect
    from retaining_walls.cantilever import analyze_cantilever_wall
    default = inspect.signature(
        analyze_cantilever_wall).parameters[key].default
    if default is inspect.Parameter.empty:
        raise ValueError(
            f"retaining_walls calc steps: the analysis gives no {key}, "
            f"which analyze_cantilever_wall requires; pass the inputs the "
            f"run was given")
    return default


# ════════════════════════════════════════════════════════════════════
#  INPUT SUMMARY
# ════════════════════════════════════════════════════════════════════

def get_input_summary(result, analysis) -> List[InputItem]:
    """Build input parameter table for retaining wall calc package.

    Parameters
    ----------
    result : CantileverWallResult or MSEWallResult
        Computed results.
    analysis : dict
        Analysis parameters (see module docstring).

    Returns
    -------
    list of InputItem
    """
    wtype = _wall_type(result, analysis)

    if wtype == "mse_lrfd":
        return _input_summary_mse_lrfd(result, analysis)
    if wtype == "mse":
        return _input_summary_mse(result, analysis)
    else:
        return _input_summary_cantilever(result, analysis)


def _input_summary_cantilever(result, analysis) -> List[InputItem]:
    """Input summary for cantilever walls."""
    # the inputs the run used (the cantilever_wall tool's, GM-141)
    geom = _get(analysis, "geom")
    gamma = _used(analysis, "gamma_backfill")
    phi = _used(analysis, "phi_backfill")
    c = _used(analysis, "c_backfill")
    method = _used(analysis, "pressure_method")
    gamma_c = _used(analysis, "gamma_concrete")
    phi_f = _used(analysis, "phi_foundation")
    c_f = _used(analysis, "c_foundation")
    q_all = _used(analysis, "q_allowable")
    FOS_s = result.FOS_sliding_required
    FOS_o = result.FOS_overturning_required

    items = [
        InputItem("Type", "Wall type", "Cantilever", ""),
        InputItem("H", "Wall height (underside of base to top of stem)",
                  f"{geom.wall_height:.2f}", "m"),
        InputItem("h_stem", "Stem height (H - base thickness)",
                  f"{geom.stem_height:.2f}", "m"),
        InputItem("B", "Base width", f"{geom.base_width:.2f}", "m"),
        InputItem("t_toe", "Toe length", f"{geom.toe_length:.2f}", "m"),
        InputItem("t_heel", "Heel length", f"{geom.heel_length:.2f}", "m"),
        InputItem("t_stem_top", "Stem thickness (top)", f"{geom.stem_thickness_top:.2f}", "m"),
        InputItem("t_stem_base", "Stem thickness (base)", f"{geom.stem_thickness_base:.2f}", "m"),
        InputItem("t_base", "Base slab thickness", f"{geom.base_thickness:.2f}", "m"),
    ]
    # GM-288: the battered face, the embedment and the key
    if geom.stem_thickness_base > geom.stem_thickness_top:
        items.append(InputItem("batter", "Battered face of the stem",
                               getattr(geom, "stem_batter", "back"), ""))
    if getattr(geom, "embedment_depth", None) is not None:
        items.append(InputItem("D_f", "Embedment (ground in front to the "
                               "underside of the base)",
                               f"{geom.embedment_depth:.2f}", "m"))
    if geom.has_shear_key:
        items.append(InputItem("d_key", "Shear key depth below the base",
                               f"{geom.key_depth:.2f}", "m"))
    if result.founded_on_rock:
        items.append(InputItem("Rock", "Founded on rock", "yes", ""))

    if geom.backfill_slope > 0:
        items.append(InputItem("\u03b2", "Backfill slope", f"{geom.backfill_slope:.1f}", "deg"))
    # The live and the dead-load surcharge (GM-157)
    if geom.surcharge > 0:
        items.append(InputItem("q_s", "Live-load surcharge", f"{geom.surcharge:.1f}", "kPa"))
    if getattr(geom, "dead_load_surcharge", 0.0) > 0:
        items.append(InputItem("q_D", "Dead-load surcharge",
                               f"{geom.dead_load_surcharge:.1f}", "kPa"))
    # GM-292: the water table behind the wall
    d_w = getattr(geom, "water_table_depth", None)
    if d_w is not None:
        items.append(InputItem("d_w", "Water table behind the wall, below "
                               "the top of the backfill", f"{d_w:.2f}", "m"))
        if geom.water_height > 0:
            g_sat = _used(analysis, "gamma_sat_backfill")
            items.append(InputItem(
                "\u03b3_sat", "Backfill unit weight below the water table",
                f"{gamma if g_sat is None else g_sat:.1f}"
                + (" (= \u03b3_backfill)" if g_sat is None else ""),
                "kN/m\u00b3"))
            uplift = _used(analysis, "base_uplift") or "linear"
            items.append(InputItem(
                "u", "Water pressure under the base",
                "linear, heel to toe" if uplift == "linear"
                else "uniform", ""))
            items.append(InputItem(
                "U_brg", "Uplift in the bearing pressures",
                "counted" if _used(analysis, "uplift_in_bearing")
                else "left out", ""))

    items.extend([
        InputItem("\u03b3_backfill", "Backfill unit weight", f"{gamma:.1f}", "kN/m\u00b3"),
        InputItem("\u03c6_backfill", "Backfill friction angle", f"{phi:.1f}", "deg"),
    ])
    if c > 0:
        items.append(InputItem("c_backfill", "Backfill cohesion", f"{c:.1f}", "kPa"))
    items.append(InputItem("\u03b3_concrete", "Concrete unit weight", f"{gamma_c:.1f}", "kN/m\u00b3"))

    if phi_f is not None and phi_f != phi:
        items.append(InputItem("\u03c6_foundation", "Foundation friction angle",
                               f"{phi_f:.1f}", "deg"))
    if c_f > 0:
        items.append(InputItem("c_foundation", "Foundation cohesion", f"{c_f:.1f}", "kPa"))
    if q_all is not None:
        items.append(InputItem("q_all", "Allowable bearing pressure", f"{q_all:.1f}", "kPa"))
    else:
        items.append(InputItem("q_all", "Allowable bearing pressure",
                               "not given: bearing NOT CHECKED", ""))

    items.extend([
        InputItem("Method", "Earth pressure theory", method.capitalize(), ""),
        InputItem("FOS_s_req", "Required FOS (sliding)", f"{FOS_s:.1f}", ""),
        InputItem("FOS_ot_req", "Required FOS (overturning)", f"{FOS_o:.1f}", ""),
    ])

    return items


def _per_level_text(values, fmt=".3f") -> str:
    """A value given for the wall, or per level from the top (GM-261):
    runs of the same value as 'v x n'; a level with none, a dash."""
    def one(v):
        return "—" if v is None else f"{v:{fmt}}"
    if not isinstance(values, (list, tuple)):
        return one(values)
    runs = []
    for v in values:
        if runs and runs[-1][0] == v:
            runs[-1][1] += 1
        else:
            runs.append([v, 1])
    return ", ".join(one(v) + (f" × {n}" if n > 1 else "")
                     for v, n in runs)


def _level_reinforcements(reinforcement, geom):
    """The reinforcement at each level and whether it was given per level
    (GM-266), or (None, False) with none."""
    if reinforcement is None:
        return None, False
    from retaining_walls import mse as _mse
    return (_mse.level_reinforcements(reinforcement,
                                      geom.reinforcement_depths),
            isinstance(reinforcement, (list, tuple)))


def _reinforcement_keys(rs):
    """A letter for each distinct reinforcement, in the order they first
    appear from the top, for a table column and its notes (GM-266): the
    name with its Tal -> the letter."""
    keys = {}
    for r in rs:
        keys.setdefault(_reinforcement_label(r), chr(ord("A") + len(keys)))
    return keys


def _reinforcement_label(r) -> str:
    return f"{r.name}, T_al {r.Tallowable:.4g} kN/m"


def _kind_text(r) -> str:
    return ("Metallic" if r.is_metallic else
            {"geogrid": "Geogrid", "geotextile": "Geotextile"}.get(
                r.type, "Geosynthetic"))


def _from_top(values) -> str:
    return (", per level from the top"
            if isinstance(values, (list, tuple)) else "")


def _sigma_H_text(analysis) -> str:
    """How sigma_H was averaged over each contributory height, in the
    calc package's symbols (GM-273)."""
    from retaining_walls import mse as _mse
    average = _mse.sigma_H_average_of(_get(analysis, "sigma_H_average"))
    return (_mse.SIGMA_H_AVERAGE_TEXT[average]
            .replace("sigma_H", "\u03c3_H").replace("Svt", "S_vt"))


def _input_summary_mse(result, analysis) -> List[InputItem]:
    """Input summary for MSE walls."""
    geom = _get(analysis, "geom")
    gamma = _get(analysis, "gamma_backfill", 18.0)
    phi = _get(analysis, "phi_backfill", 34.0)
    reinforcement = _get(analysis, "reinforcement")
    phi_f = _get(analysis, "phi_foundation")
    c_f = _get(analysis, "c_foundation", 0.0)
    q_all = _get(analysis, "q_allowable")

    items = [
        InputItem("Type", "Wall type", "MSE (Mechanically Stabilized Earth)", ""),
        InputItem("H", "Wall height", f"{geom.wall_height:.2f}", "m"),
        InputItem("L", "Reinforcement length", f"{geom.reinforcement_length:.2f}", "m"),
        InputItem("S_v", "Reinforcement spacing", f"{geom.reinforcement_spacing:.2f}", "m"),
        InputItem("N_levels", "Number of reinforcement levels", geom.n_reinforcement_levels, ""),
    ]
    if geom.level_depths is not None:
        # GM-273: the levels at the depths given, each over its
        # contributory height (GEC-11 Vol I pp. 4-46, 4-47)
        items[3] = InputItem(
            "z", "Reinforcement level depths below the top of the wall, "
            "from the top", ", ".join(f"{z:.2f}" for z in geom.level_depths),
            "m")
        items.append(InputItem(
            "\u03c3_H", "\u03c3_H over each level's contributory height S_vt",
            _sigma_H_text(analysis), ""))

    if geom.backfill_slope > 0:
        items.append(InputItem("\u03b2", "Backfill slope", f"{geom.backfill_slope:.1f}", "deg"))
    if getattr(geom, "face_batter", 0.0) > 0:
        items.append(InputItem("\u03c9", "Face batter from vertical",
                               f"{geom.face_batter:g}", "deg"))
    if geom.surcharge > 0:
        items.append(InputItem("q_s", "Live-load surcharge", f"{geom.surcharge:.1f}", "kPa"))

    items.extend([
        InputItem("\u03b3_backfill", "Backfill unit weight", f"{gamma:.1f}", "kN/m\u00b3"),
        InputItem("\u03c6_backfill", "Backfill friction angle", f"{phi:.1f}", "deg"),
    ])

    rs, per_r = _level_reinforcements(reinforcement, geom)
    if per_r:
        # GM-266: a reinforcement per level, each with its own Tal, b, Sh
        from retaining_walls import mse as _mse
        keys = _reinforcement_keys(rs)
        items.extend([
            InputItem("Reinf.", "Reinforcement per level, from the top",
                      _per_level_text(tuple(keys[_reinforcement_label(r)]
                                            for r in rs), "s")
                      + " (" + "; ".join(f"{k} = {n}" for n, k
                                         in keys.items()) + ")", ""),
            InputItem("T_al", "Allowable tensile strength per unit width of "
                      "reinforcement, per level from the top",
                      _per_level_text(tuple(r.Tallowable for r in rs),
                                      ".1f"), "kN/m"),
            InputItem("Type_r", "Reinforcement type", _kind_text(rs[0]), ""),
        ])
        if any(r.horizontal_spacing is not None for r in rs):
            items.extend([
                InputItem("b", "Reinforcement width, per level from the top",
                          _per_level_text(tuple(r.width for r in rs)), "m"),
                InputItem("S_h", "Horizontal spacing, centre to centre, per "
                          "level from the top",
                          _per_level_text(tuple(r.horizontal_spacing
                                                for r in rs)), "m"),
            ])
        Rcs = _mse.coverage_ratios(reinforcement, geom.reinforcement_depths)
        items.append(InputItem(
            "R_c", "Coverage ratio b/S_h (1 for a continuous layer), per "
            "level from the top", _per_level_text(tuple(Rcs)), ""))
    elif reinforcement is not None:
        items.extend([
            InputItem("Reinf.", "Reinforcement product", reinforcement.name, ""),
            InputItem("T_al", "Allowable tensile strength per unit width of "
                      "reinforcement", f"{reinforcement.Tallowable:.1f}",
                      "kN/m"),
            InputItem("Type_r", "Reinforcement type",
                      "Metallic" if reinforcement.is_metallic else
                      {"geogrid": "Geogrid", "geotextile": "Geotextile"}.get(
                          reinforcement.type, "Geosynthetic"), ""),
        ])
        # GM-256: the coverage ratio, GEC-11 Vol I Eq. 3-1
        from retaining_walls import mse as _mse
        if reinforcement.horizontal_spacing is not None:
            items.extend([
                InputItem("b", "Reinforcement width",
                          f"{reinforcement.width:.3f}", "m"),
                InputItem("S_h", "Horizontal spacing, centre to centre"
                          + _from_top(reinforcement.horizontal_spacing),
                          _per_level_text(reinforcement.horizontal_spacing),
                          "m"),
            ])
        # GM-261: one value a level when the spacing is given per level
        Rcs = _mse.coverage_ratios(reinforcement, geom.reinforcement_depths)
        items.append(InputItem(
            "R_c", "Coverage ratio b/S_h (1 for a continuous layer)"
            + _from_top(Rcs if reinforcement.per_level else None),
            _per_level_text(tuple(Rcs) if reinforcement.per_level
                            else Rcs[0]), ""))

    if phi_f is not None:
        items.append(InputItem("\u03c6_foundation", "Foundation friction angle",
                               f"{phi_f:.1f}", "deg"))
    else:
        # GM-302 review: GEC-11 Vol I p. 4-10's 30 deg for no data
        from retaining_walls import mse as _mse
        items.append(InputItem(
            "\u03c6_foundation", "Foundation friction angle, not given: "
            "assumed for preliminary sizing (GEC-11 Vol I p. 4-10)",
            f"{_mse.PHI_ASSUMED:.1f}", "deg"))
    if c_f > 0:
        items.append(InputItem("c_foundation", "Foundation cohesion", f"{c_f:.1f}", "kPa"))
    if q_all is not None:
        items.append(InputItem("q_all", "Allowable bearing pressure", f"{q_all:.1f}", "kPa"))

    return items


# ════════════════════════════════════════════════════════════════════
#  CALCULATION STEPS
# ════════════════════════════════════════════════════════════════════

def get_calc_steps(result, analysis) -> List[CalcSection]:
    """Build step-by-step calculation sections.

    Parameters
    ----------
    result : CantileverWallResult or MSEWallResult
        Computed results.
    analysis : dict
        Analysis parameters (see module docstring).

    Returns
    -------
    list of CalcSection
    """
    wtype = _wall_type(result, analysis)

    if wtype == "mse_lrfd":
        return _calc_steps_mse_lrfd(result, analysis)
    if wtype == "mse":
        return _calc_steps_mse(result, analysis)
    else:
        return _calc_steps_cantilever(result, analysis)


def _calc_steps_cantilever(result, analysis) -> List[CalcSection]:
    """Calculation steps for cantilever retaining wall."""
    # the inputs the run used (the cantilever_wall tool's, GM-141)
    geom = _get(analysis, "geom")
    gamma = _used(analysis, "gamma_backfill")
    phi = _used(analysis, "phi_backfill")
    c = _used(analysis, "c_backfill")
    method = _used(analysis, "pressure_method")
    phi_f = _used(analysis, "phi_foundation")
    q_all = _used(analysis, "q_allowable")
    FOS_s_req = result.FOS_sliding_required
    FOS_o_req = result.FOS_overturning_required

    if phi_f is None:
        phi_f = phi

    sections = []

    # ── Section 1: Earth Pressure Coefficient ──────────────────
    ep_items = []

    # The terms the analysis used (GM-143): the result's own, so the
    # printout cannot drift from the run.
    ep, forces, sl, ot, br = _cantilever_terms(result, analysis)

    Ka_val = ep["Ka"]
    if method == "coulomb":
        delta = ep["delta_deg"]
        beta = geom.backfill_slope

        ep_items.append(CalcStep(
            title="Coulomb Active Earth Pressure Coefficient",
            equation=(
                "K_a = sin\u00b2(\u03b1 + \u03c6) / "
                "[sin\u00b2(\u03b1) \u00d7 sin(\u03b1 - \u03b4) \u00d7 "
                "(1 + \u221a(sin(\u03c6+\u03b4)\u00d7sin(\u03c6-\u03b2) / "
                "(sin(\u03b1-\u03b4)\u00d7sin(\u03b1+\u03b2))))\u00b2]"
            ),
            substitution=(
                f"\u03c6 = {phi:.1f}\u00b0, \u03b4 = {delta:.1f}\u00b0, "
                f"\u03b2 = {beta:.1f}\u00b0"
            ),
            result_name="K_a",
            result_value=f"{Ka_val:.4f}",
            reference="UFC 3-220-20 Eq. 4-10 (vertical plane, θ = 0)",
            notes=(f"δ on the vertical plane through the heel: "
                   + ("wall_friction_deg as given." if ep.get("delta_given")
                      else "the backfill slope β, soil on soil (FHWA "
                           "NHI-06-089 p. 10-20).")),
        ))
    else:
        if geom.backfill_slope > 0:
            ep_items.append(CalcStep(
                title="Rankine Active Earth Pressure Coefficient (sloped backfill)",
                equation=(
                    "K_a = cos\u03b2 \u00d7 (cos\u03b2 - \u221a(cos\u00b2\u03b2 - "
                    "cos\u00b2\u03c6)) / (cos\u03b2 + \u221a(cos\u00b2\u03b2 - cos\u00b2\u03c6))"
                ),
                substitution=(
                    f"\u03c6 = {phi:.1f}\u00b0, \u03b2 = {geom.backfill_slope:.1f}\u00b0"
                ),
                result_name="K_a",
                result_value=f"{Ka_val:.4f}",
                reference="Rankine (1857), sloped backfill",
            ))
        else:
            ep_items.append(CalcStep(
                title="Rankine Active Earth Pressure Coefficient",
                equation="K_a = tan\u00b2(45\u00b0 - \u03c6/2)",
                substitution=f"K_a = tan\u00b2(45\u00b0 - {phi:.1f}\u00b0/2)",
                result_name="K_a",
                result_value=f"{Ka_val:.4f}",
                reference="Rankine (1857)",
            ))

    # Active height
    H = ep["H_active"]
    ep_items.append(CalcStep(
        title="Active Pressure Height",
        equation="H_active = H + heel \u00d7 tan(\u03b2)" if geom.backfill_slope > 0
        else "H_active = H (no backfill slope)",
        substitution=f"H_active = {geom.wall_height:.2f}" + (
            f" + {geom.heel_length:.2f} \u00d7 tan({geom.backfill_slope:.1f}\u00b0)"
            if geom.backfill_slope > 0 else ""
        ),
        result_name="H_active",
        result_value=f"{H:.3f}",
        result_unit="m",
    ))

    # Active force, as the analysis summed it; q is the live and the
    # dead-load surcharge together (GM-157)
    Pa_total = ep["Pa"]
    q_live = ep.get("q_live", geom.surcharge)
    q_dead = ep.get("q_dead", getattr(geom, "dead_load_surcharge", 0.0))
    q = q_live + q_dead
    ep_items.append(CalcStep(
        title="Total Active Thrust",
        equation="P_a = 0.5 \u00d7 K_a \u00d7 \u03b3 \u00d7 H\u00b2"
                 + (" + K_a \u00d7 q \u00d7 H" if q > 0 else "")
                 + (" - 2c\u221aK_a \u00d7 H" if c > 0 else ""),
        substitution=(
            f"P_a = 0.5 \u00d7 {Ka_val:.4f} \u00d7 {gamma:.1f} \u00d7 {H:.3f}\u00b2"
            + (f" + {Ka_val:.4f} \u00d7 {q:.1f} \u00d7 {H:.3f}"
               if q > 0 else "")
            + (f" - 2 \u00d7 {c:.1f} \u00d7 \u221a{Ka_val:.4f} \u00d7 {H:.3f}"
               if c > 0 else "")
        ),
        result_name="P_a",
        result_value=f"{Pa_total:.1f}",
        result_unit="kN/m",
        reference="UFC 3-220-20 Eq. 4-6; FHWA NHI-06-089 Eq. 10-10",
        notes=(f"q = q_live + q_dead = {q_live:.1f} + {q_dead:.1f} kPa: "
               "both surcharges' thrust drives every check."
               if q_live > 0 and q_dead > 0 else ""),
    ))
    z_t = ep.get("z_tension", 0.0) or 0.0
    if z_t > 0:
        # GM-288: no negative pressure in the tension zone; the step above
        # is replaced by the triangle below it
        ep_items[-1] = CalcStep(
            title="Total Active Thrust (tension zone)",
            equation=("z_0 = (2c√K_a - K_a q) / (K_a γ),  "
                      "P_a = 0.5 × K_a × γ × (H - z_0)²"),
            substitution=(
                f"z_0 = (2 × {c:.1f} × √{Ka_val:.4f} - "
                f"{Ka_val:.4f} × {q:.1f}) / ({Ka_val:.4f} × "
                f"{gamma:.1f}) = {z_t:.3f} m; P_a = 0.5 × "
                f"{Ka_val:.4f} × {gamma:.1f} × ({H:.3f} - "
                f"{z_t:.3f})²" if z_t < H else
                f"2c√K_a ≥ K_a (q + γ H): no pressure over "
                f"the whole height H = {H:.3f} m"),
            result_name="P_a",
            result_value=f"{Pa_total:.1f}",
            result_unit="kN/m",
            reference="UFC 3-220-20 Eq. 4-6 and Fig. 4-5; FHWA NHI-06-089 "
                      "p. 10-15",
            notes=("The pressure K_a (q + γz) - 2c√K_a is negative "
                   "above z_0 and taken as zero there, never a pull on the "
                   "wall (UFC 3-220-20 Fig. 4-5: 'In no case should negative "
                   "pressure be considered'); the pull over it, "
                   f"{ep['Pa_tension_zone']:.1f} kN/m, is not counted. P_a "
                   "acts at (H - z_0)/3. "
                   "Water in the tension crack is not modelled."),
        )
    h_w = ep.get("h_w") or 0.0
    if h_w > 0:
        # GM-292: the effective earth pressure below the water table, and
        # the water's own; the steps above are replaced by these
        z_w, g_b = ep["z_w"], ep["gamma_b"]
        closed = not z_t > 0
        ep_items[-1] = CalcStep(
            title="Active Thrust with a Water Table (effective stress)",
            equation=(
                "σ'_v = q + γz above the water table (z ≤ z_w), q + γz_w + "
                "γ_b(z - z_w) below it, γ_b = γ_sat - γ_w; "
                + ("P_a = K_a [q H + ½γz_w² + γz_w(H - z_w) + ½γ_b(H - z_w)²]"
                   + (" - 2c√K_a H" if c > 0 else "") if closed else
                   "p_a = K_a σ'_v - 2c√K_a, zero where negative; P_a = ∫p_a dz")),
            substitution=(
                f"z_w = {z_w:.3f} m below the top of the pressure plane, "
                f"γ_b = {ep['gamma_sat']:.2f} - {ep['gamma_w']:.2f} = "
                f"{g_b:.2f} kN/m³; "
                + (f"P_a = {Ka_val:.4f} × [{q:.1f} × {H:.3f} + 0.5 × "
                   f"{gamma:.1f} × {z_w:.3f}² + {gamma:.1f} × {z_w:.3f} × "
                   f"{H - z_w:.3f} + 0.5 × {g_b:.2f} × {H - z_w:.3f}²]"
                   + (f" - 2 × {c:.1f} × √{Ka_val:.4f} × {H:.3f}"
                      if c > 0 else "") if closed else
                   f"zero over the top {z_t:.3f} m")),
            result_name="P_a",
            result_value=f"{Pa_total:.1f}",
            result_unit="kN/m",
            reference=("UFC 3-220-20 Sec. 4-4.1 and Figs. 4-5, 4-16(a); FHWA "
                       "NHI-06-089 Fig. 10-12"),
            notes=(f"The effective earth pressure's thrust, {ep['z_Pa']:.3f} m "
                   "above the underside of the base; the water's is the "
                   "next step. UFC 3-220-20 p. 200: 'the buoyant or "
                   "effective unit weight (γb) is used in the earth "
                   "pressure equations for soils below the water table'."
                   + ("" if closed else " Water in the tension crack is not "
                      "modelled.")),
        )
        ep_items.append(CalcStep(
            title="Water Pressure Thrust",
            equation="P_w = 0.5 × γ_w × h_w²,  z_Pw = h_w / 3",
            substitution=(f"h_w = {geom.wall_height:.2f} - "
                          f"{geom.water_table_depth:.2f} = {h_w:.3f} m; "
                          f"P_w = 0.5 × {ep['gamma_w']:.2f} × {h_w:.3f}²"),
            result_name="P_w",
            result_value=f"{ep['Pw']:.1f}",
            result_unit="kN/m",
            reference=("UFC 3-220-20 Sec. 4-4.1 (an earth pressure "
                       "coefficient of unity); FHWA NHI-06-089 Fig. 10-12"),
            notes=(f"Horizontal, at z_Pw = {ep['z_Pw']:.3f} m above the "
                   "underside of the base; it drives sliding and overturning "
                   "with P_h. h_w is the water table's height above the "
                   "underside of the base; no water is taken in front of "
                   "the wall."),
        ))

    # Thrust inclination & components (cantilever._active_thrust, RW-1):
    # Coulomb -> the wall friction on the plane through the heel (beta by
    # default, GM-288); sloped Rankine -> delta = beta; else horizontal.
    delta_incl = ep["delta_deg"]
    Ph = ep["Ph"]
    Pv = ep["Pv"]

    if delta_incl > 0:
        # The live load's share of P_v the analysis left out of the
        # resisting side (GM-157), from the forces it summed
        Pv_left_out = sum(f["W"] for f in forces
                          if f.get("kind") == "thrust"
                          and not f.get("resists", True))
        ep_items.append(CalcStep(
            title="Thrust Components (inclined thrust)",
            equation="P_h = P_a \u00d7 cos(\u03b4),  P_v = P_a \u00d7 sin(\u03b4)",
            substitution=(
                f"\u03b4 = {delta_incl:.2f}\u00b0; "
                f"P_h = {Pa_total:.1f} \u00d7 cos({delta_incl:.2f}\u00b0), "
                f"P_v = {Pa_total:.1f} \u00d7 sin({delta_incl:.2f}\u00b0)"
            ),
            result_name="P_h, P_v",
            result_value=f"{Ph:.1f}, {Pv:.1f}",
            result_unit="kN/m",
            notes=(
                "P_h drives sliding/overturning; P_v acts downward at the "
                "pressure plane (back of heel) and is stabilizing."
                + (f" The live load's share, {Pv_left_out:.1f} kN/m, is "
                   "left out of sliding, overturning and the eccentricity "
                   "check." if Pv_left_out > 0 else "")
            ),
            reference=("Coulomb wall friction \u03b4 on the plane through "
                       "the heel" + ("" if ep.get("delta_given") else
                                     " = \u03b2, FHWA NHI-06-089 p. 10-20"))
            if method == "coulomb"
            else "Rankine thrust parallel to backfill slope",
        ))

    sections.append(CalcSection(title="Earth Pressure Computation", items=ep_items))

    # ── Section 2: Wall Weight Components ──────────────────────
    weight_items = []
    B = geom.base_width

    # Every vertical force the analysis summed, with its arm (GM-143)
    weight_rows = [[f["component"], f"{f['W']:.1f}", f"{f['x']:.3f}",
                    f"{f['M']:.1f}"] for f in forces]

    # Totals, as the analysis summed them. A surcharge over the heel that
    # is not a resisting force (GM-157) makes two sets of loads: the one
    # sliding, overturning and the eccentricity check count, and the one
    # the bearing pressures count. Each check's own total is printed.
    total_W = br["V"]
    M_stab = br["M_stab"]
    split = _loads_differ(ot, br)
    headers = ["Component", "Weight (kN/m)", "Arm from Toe (m)",
               "Moment (kN\u00b7m/m)"]
    notes = "Moments computed about toe for overturning check."
    if split or any(not f.get("resists", True) or not f.get("bearing", True)
                    for f in forces):
        headers.append("Counted in")
        for row, f in zip(weight_rows, forces):
            row.append(_counted_in(f))
        notes = ("Counted in: the checks that sum the force. A surcharge "
                 "over the heel that is not a resisting force is left out "
                 "of sliding, overturning and the eccentricity check "
                 "(see Choices).")
    uplift = [f for f in forces if f.get("kind") == "uplift"]
    if uplift:
        # GM-292: the uplift comes off V; its moment is in M_ot, so the
        # totals' moments leave it out
        notes += (" The uplift (negative) comes off V; its moment about "
                  "the toe is counted in the overturning moment M_ot, so "
                  "the TOTAL moments leave it out (see Choices).")
    if split:
        weight_rows.append(["TOTAL, sliding / overturning / eccentricity",
                            f"{ot['V']:.1f}", "\u2014",
                            f"{ot['M_stabilizing']:.1f}", ""])
        weight_rows.append(["TOTAL, bearing", f"{total_W:.1f}", "\u2014",
                            f"{M_stab:.1f}", ""])
    else:
        weight_rows.append(["TOTAL", f"{total_W:.1f}", "\u2014",
                            f"{M_stab:.1f}"]
                           + ([""] if len(headers) == 5 else []))

    if uplift:
        f = uplift[0]
        uplift_step = CalcStep(
            title="Uplift on the Base",
            equation=("U = 0.5 × γ_w × h_w × B,  x_U = 2B/3"
                      if f.get("base_uplift", "linear") == "linear" else
                      "U = γ_w × h_w × B,  x_U = B/2"),
            substitution=(f"U = {'0.5 × ' if f.get('base_uplift', 'linear') == 'linear' else ''}"
                          f"{ep['gamma_w']:.2f} × {ep['h_w']:.3f} × {B:.2f}"),
            result_name="U",
            result_value=f"{-f['W']:.1f}",
            result_unit="kN/m",
            reference=("UFC 3-220-20 Fig. 5-40 (uniform)"
                       if f.get("base_uplift") == "uniform" else
                       "EM 1110-2-2104 para. E-12 (p. 115)"),
            notes=(f"At x_U = {f['x']:.3f} m from the toe: "
                   + ("γ_w h_w at the heel to zero at the toe, no water in "
                      "front of the wall."
                      if f.get("base_uplift", "linear") == "linear" else
                      "γ_w h_w under the whole base.")
                   + " It comes off V in sliding, its moment U x_U is an "
                   "overturning moment, and the bearing pressures "
                   + ("count it." if f.get("bearing") else "leave it out.")),
        )
        weight_items.append(uplift_step)

    weight_items.append(TableData(
        title="Wall Component Weights and Moments about Toe",
        headers=headers,
        rows=weight_rows,
        notes=notes,
    ))

    sections.append(CalcSection(title="Wall Weights & Stabilizing Moments", items=weight_items))

    # ── Section 3: Sliding Check ───────────────────────────────
    slide_items = []

    # The base interface and passive resistance the analysis USED (GM-103:
    # the package printed (2/3)phi and no passive even when delta_base,
    # base_adhesion or include_passive set them otherwise).
    delta_b = sl["delta_b"]
    ca = sl["ca"]
    Pp = sl["Pp"]
    # GM-212: the printed rule for the foundation cohesion
    rule = sl.get("sliding_cohesion", "drained")
    width = sl.get("adhesion_width")
    # the theory of the passive coefficient in front of the wall (GM-242)
    passive_name = "Rankine passive resistance"
    if sl.get("passive_theory") == "log_spiral":
        passive_name = (
            f"log-spiral passive resistance (Kp {sl['Kp']:.3g} at wall "
            f"friction {sl['passive_delta']:.3g}°, log_spiral_source "
            f"'{sl['log_spiral_source']}')")
    elif Pp > 0:
        passive_name = f"Rankine passive resistance (Kp {sl['Kp']:.3g})"

    if delta_b is None:
        pass
    elif sl["delta_b_given"]:
        slide_items.append(CalcStep(
            title="Base Friction Angle",
            equation="\u03b4_b given directly (no 2/3 factor)",
            substitution=f"\u03b4_b = {delta_b:.2f}\u00b0",
            result_name="\u03b4_b",
            result_value=f"{delta_b:.2f}",
            result_unit="deg",
            reference="input delta_base",
        ))
    else:
        slide_items.append(CalcStep(
            title="Base Friction Angle",
            equation="\u03b4_b = (2/3) \u00d7 \u03c6_foundation",
            substitution=f"\u03b4_b = (2/3) \u00d7 {phi_f:.1f}\u00b0",
            result_name="\u03b4_b",
            result_value=f"{delta_b:.2f}",
            result_unit="deg",
            reference=("EN 1997-1 6.5.3(10), 2/3 \u03c6'cv;d for smooth "
                       "precast foundations; UFC 3-220-20 Table 4-1"),
        ))

    if rule == "friction_plus_adhesion":
        if sl["ca_given"]:
            slide_items.append(CalcStep(
                title="Base Adhesion",
                equation="C_a given directly",
                substitution=f"C_a = {ca:.1f} kPa",
                result_name="C_a", result_value=f"{ca:.1f}",
                result_unit="kPa", reference="input base_adhesion"))
        else:
            slide_items.append(CalcStep(
                title="Base Adhesion",
                equation=("C_a from the undrained shear strength c_f, "
                          "read linearly within its row"),
                substitution=f"c_f = {sl['c_foundation']:.1f} kPa",
                result_name="C_a", result_value=f"{ca:.1f}",
                result_unit="kPa", reference="UFC 3-220-20 Table 4-1"))
        e_s = sl["eccentricity"]
        slide_items.append(CalcStep(
            title="Effective Base Width",
            equation="B_e = B \u2212 2|e|",
            substitution=f"B_e = {B:.2f} \u2212 2 \u00d7 {abs(e_s):.3f}",
            result_name="B_e", result_value=f"{width:.3f}", result_unit="m",
            reference="UFC 3-220-20 Sec. 4-5.1.3, Fig. 4-27",
            notes=("e from the loads the sliding check counts, the uniform "
                   "bearing pressure method; NHI-06-089 Fig. 10-17 takes "
                   "the full B.")))

    # The passive resistance in front of the wall, as the analysis took it
    # (GM-288: over the embedment Df, and cu under the undrained rule)
    if Pp > 0:
        D, D_top, Kp = sl["D_passive"], sl["D_passive_top"], sl["Kp"]
        gf = sl["gamma_foundation"]
        Ppc = sl.get("Pp_cohesion", 0.0) or 0.0
        slide_items.append(CalcStep(
            title="Passive Resistance in Front of the Wall",
            equation=("P_p = 0.5 \u00d7 K_p \u00d7 \u03b3_f \u00d7 "
                      "(D\u00b2 - D_top\u00b2)"
                      + (" + 2c_u\u221aK_p \u00d7 (D - D_top)"
                         if Ppc > 0 else "")),
            substitution=(
                f"P_p = 0.5 \u00d7 {Kp:.3f} \u00d7 {gf:.1f} \u00d7 "
                f"({D:.2f}\u00b2 - {D_top:.2f}\u00b2)"
                + (f" + 2 \u00d7 {sl['c_foundation']:.1f} \u00d7 "
                   f"\u221a{Kp:.3f} \u00d7 {D - D_top:.2f}"
                   if Ppc > 0 else "")),
            result_name="P_p",
            result_value=f"{Pp:.1f}",
            result_unit="kN/m",
            reference=("UFC 3-220-20 Eq. 4-7, Figs. 4-27 and 4-29; GEC-6 "
                       "Eq. 5-41"),
            notes=(f"{passive_name}; "
                   + (f"over the shear key, from the underside of the base "
                      f"(D_top = {D_top:.2f} m below the ground in front) to "
                      f"the key's bottom (D = {D:.2f} m). "
                      if D_top else
                      f"over the embedment, the ground in front {D:.2f} m "
                      "above the underside of the base. ")
                   + ("c_foundation counted as cu (the undrained rule)."
                      if Ppc > 0 else "")),
        ))

    R_sliding = sl["resisting"]
    if rule == "undrained":
        equation = "R = c_a \u00d7 B"
        substitution = f"R = {ca:.1f} \u00d7 {width:.2f}"
        reference = "GEC-6 Eq. 5-39"
        rule_note = ("Undrained: adhesion alone, c_a = "
                     + ("base_adhesion given directly"
                        if sl["ca_given"] else
                        "c_foundation as the undrained shear strength")
                     + ", no base friction (GEC-6: Eqs. 5-38 and 5-39 "
                     "'should not be combined'). "
                     + (f"Held to 0.4 V = {sl['R_adhesion']:.1f} kN/m "
                        "(EN 1997-1 Eq. 6.5: water or air can reach the "
                        "interface). " if sl.get("gap_limited") else ""))
    elif rule == "friction_plus_adhesion":
        equation = "R = V \u00d7 tan(\u03b4_b) + C_a \u00d7 B_e"
        substitution = (f"R = {sl['V']:.1f} \u00d7 tan({delta_b:.2f}\u00b0)"
                        f" + {ca:.1f} \u00d7 {width:.2f}")
        reference = "UFC 3-220-20 Fig. 4-27"
        rule_note = ("Friction and adhesion added (UFC 3-220-20 Sec. "
                     "4-5.1.3, which adds that the minimum FS may need to "
                     "be increased when base adhesion is included). ")
    else:
        equation = "R = V \u00d7 tan(\u03b4_b)"
        substitution = f"R = {sl['V']:.1f} \u00d7 tan({delta_b:.2f}\u00b0)"
        reference = "GEC-6 Eq. 5-38"
        rule_note = (f"Drained: c_foundation = {sl['c_foundation']:g} kPa "
                     "not counted (GEC-6: Eqs. 5-38 and 5-39 'should not "
                     "be combined'; EN 1997-1 6.5.3(10)). "
                     if sl.get("c_foundation") else "")
    slide_items.append(CalcStep(
        title="Resisting Force (Sliding)",
        equation=equation + (" + P_p" if Pp > 0 else ""),
        substitution=substitution + (f" + {Pp:.1f}" if Pp > 0 else ""),
        result_name="R",
        result_value=f"{R_sliding:.1f}",
        result_unit="kN/m",
        reference=reference,
        notes=rule_note
              + (f"V leaves out {sl['V_left_out']:.1f} kN/m of surcharge "
                 "over the heel, not a resisting force (see Choices)."
                 if sl.get("V_left_out", 0.0) > 0 else "")
              + (f" V is net of the uplift U = {sl['U']:.1f} kN/m."
                 if (sl.get("U") or 0.0) > 0 else ""),
    ))

    if result.FOS_sliding is None:
        # No driving force (GM-100): say so, never a placeholder FOS.
        slide_items.append(
            "Sliding: no driving force — the active thrust P_h is zero, "
            "so nothing pushes the wall. The check passes; there is no "
            "finite factor of safety to report.")
    else:
        Pw_s = sl.get("Pw", 0.0) or 0.0
        slide_items.append(CalcStep(
            title="Factor of Safety Against Sliding",
            equation=("FOS_sliding = R / (P_h + P_w)" if Pw_s > 0
                      else "FOS_sliding = R / P_h"),
            substitution=(f"FOS_sliding = {R_sliding:.1f} / ({Ph:.1f} + "
                          f"{Pw_s:.1f})" if Pw_s > 0 else
                          f"FOS_sliding = {R_sliding:.1f} / {sl['driving']:.1f}"),
            result_name="FOS_sliding",
            result_value=f"{result.FOS_sliding:.3f}",
            reference=("GEC-6 Eq. 5-40 (min 1.5); UFC 3-220-20 Fig. 4-27 "
                       "(min 2.0 with the passive resistance)"),
            notes=("P_h = P_a × cos(δ) — horizontal thrust component."
                   if delta_incl > 0 else ""),
        ))

        slide_items.append(CheckItem(
            description="Sliding stability",
            demand=FOS_s_req,
            demand_label="FOS_required",
            capacity=result.FOS_sliding,
            capacity_label="FOS_sliding",
            passes=result.passes_sliding,
        ))

    sections.append(CalcSection(title="Sliding Stability Check", items=slide_items))

    # ── Section 4: Overturning Check ───────────────────────────
    ot_items = []

    z_Pa = ot["z_Pa"]
    M_over = ot["M_overturning"]

    Pw_o, M_U = ot.get("Pw", 0.0) or 0.0, ot.get("M_uplift", 0.0) or 0.0
    ot_items.append(CalcStep(
        title="Overturning Moment about Toe",
        equation=("M_ot = P_h \u00d7 z_Pa + P_w \u00d7 z_Pw + U \u00d7 x_U"
                  if Pw_o > 0 else "M_ot = P_h \u00d7 z_Pa"),
        substitution=(f"M_ot = {ot['Ph']:.1f} \u00d7 {z_Pa:.3f} + "
                      f"{Pw_o:.1f} \u00d7 {ot['z_Pw']:.3f} + {M_U:.1f}"
                      if Pw_o > 0 else
                      f"M_ot = {ot['Ph']:.1f} \u00d7 {z_Pa:.3f}"),
        result_name="M_ot",
        result_value=f"{M_over:.1f}",
        result_unit="kN\u00b7m/m",
        notes=("P_h = P_a \u00d7 cos(\u03b4); the vertical component P_v "
               "appears as a stabilizing entry in the weight table."
               if delta_incl > 0 else "")
              + (" U \u00d7 x_U is the uplift's moment about the toe, "
                 "counted as overturning, the module's assumption (see "
                 "Choices)." if M_U > 0 else ""),
    ))

    ot_items.append(CalcStep(
        title="Stabilizing Moment about Toe",
        equation="M_stab = \u03a3(W_i \u00d7 x_i)",
        substitution=("(see weight table above: TOTAL, sliding / "
                      "overturning / eccentricity)" if split
                      else "(see weight table above)"),
        result_name="M_stab",
        result_value=f"{ot['M_stabilizing']:.1f}",
        result_unit="kN\u00b7m/m",
    ))

    if result.FOS_overturning is None:
        ot_items.append(
            "Overturning: no driving force — the overturning moment "
            "M_ot is zero. The check passes; there is no finite factor of "
            "safety to report.")
    else:
        ot_items.append(CalcStep(
            title="Factor of Safety Against Overturning",
            equation="FOS_ot = M_stab / M_ot",
            substitution=(f"FOS_ot = {ot['M_stabilizing']:.1f} / "
                          f"{M_over:.1f}"),
            result_name="FOS_ot",
            result_value=f"{result.FOS_overturning:.3f}",
            reference=("FHWA NHI-06-089 Sec. 10.5.5; GEC-6 Sec. 7.3 (about "
                       "2.0 on soil, 1.5 on rock)"),
        ))

        ot_items.append(CheckItem(
            description="Overturning stability",
            demand=FOS_o_req,
            demand_label="FOS_required",
            capacity=result.FOS_overturning,
            capacity_label="FOS_overturning",
            passes=result.passes_overturning,
        ))

    sections.append(CalcSection(title="Overturning Stability Check", items=ot_items))

    # ── Section 5: Bearing Pressure & Eccentricity ─────────────
    brg_items = []

    # The eccentricity check, on the loads the overturning check counts,
    # as the analysis worked it out (GM-157); B/6, or B/4 on rock (GM-288)
    x_R_check = ot["x_resultant"]
    within = result.within_eccentricity_limit
    part, frac = (("middle half", "B/4") if result.founded_on_rock
                  else ("middle third", "B/6"))
    brg_items.append(CalcStep(
        title="Resultant Location from Toe"
              + (" (sliding / overturning loads)" if split else ""),
        equation="x_R = (M_stab - M_ot) / V",
        substitution=(f"x_R = ({ot['M_stabilizing']:.1f} - "
                      f"{ot['M_overturning']:.1f}) / {ot['V']:.1f}"),
        result_name="x_R",
        result_value=f"{x_R_check:.3f}",
        result_unit="m",
    ))

    brg_items.append(CalcStep(
        title="Eccentricity of Resultant",
        equation="e = B/2 - x_R",
        substitution=f"e = {B:.2f}/2 - {x_R_check:.3f}",
        result_name="e",
        result_value=f"{result.eccentricity:.3f}",
        result_unit="m",
        notes=f"{frac} = {result.eccentricity_limit:.3f} m"
              f"{' (founded on rock)' if result.founded_on_rock else ''}. "
              f"{'Resultant within the ' + part if within else 'OUTSIDE the ' + part}.",
        reference="GEC-6 Sec. 6.4.1; UFC 3-220-20 Sec. 4-5.1.2",
    ))

    # The bearing pressures, on the bearing check's loads
    x_R = br["x_resultant"]
    e = br["eccentricity"]
    if split:
        brg_items.append(CalcStep(
            title="Resultant for Bearing (bearing loads)",
            equation="x_R,b = (M_stab,b - M_ot) / V_b,  e_b = B/2 - x_R,b",
            substitution=(f"x_R,b = ({br['M_stab']:.1f} - "
                          f"{br['M_over']:.1f}) / {total_W:.1f}; "
                          f"e_b = {B:.2f}/2 - {x_R:.3f}"),
            result_name="e_b",
            result_value=f"{e:.3f}",
            result_unit="m",
            notes=("The eccentricity for bearing is not the eccentricity "
                   "check's (GEC-11 Vol I p. 4-26): the bearing pressures "
                   "count the surcharge over the heel"
                   + (" and the uplift" if uplift else "")
                   + " as the Choices say."),
        ))

    # Bearing pressures, by the distribution the analysis used
    if br["distribution"] == "trapezoidal":
        brg_items.append(CalcStep(
            title="Toe Bearing Pressure (Trapezoidal Distribution)",
            equation="q_toe = (V/B)(1 + 6e/B)",
            substitution=f"q_toe = ({total_W:.1f}/{B:.2f})(1 + 6\u00d7{e:.3f}/{B:.2f})",
            result_name="q_toe",
            result_value=f"{result.q_toe:.1f}",
            result_unit="kPa",
        ))
        brg_items.append(CalcStep(
            title="Heel Bearing Pressure",
            equation="q_heel = (V/B)(1 - 6e/B)",
            substitution=f"q_heel = ({total_W:.1f}/{B:.2f})(1 - 6\u00d7{e:.3f}/{B:.2f})",
            result_name="q_heel",
            result_value=f"{result.q_heel:.1f}",
            result_unit="kPa",
        ))
    elif not (br["B_eff"] or 0) > 0:
        # x_R <= 0 (or >= B): the resultant falls outside the base
        brg_items.append(
            "Bearing pressure: the resultant falls outside the base "
            f"(x_R = {x_R:.3f} m from the toe, B = {B:.2f} m), so no "
            "bearing pressure distribution balances it.")
    elif br["distribution"] == "triangular_toe":
        brg_items.append(CalcStep(
            title="Toe Bearing Pressure (Triangular Distribution)",
            equation="q_toe = 2V / (3 \u00d7 x_R)  [resultant outside middle third]",
            substitution=f"q_toe = 2 \u00d7 {total_W:.1f} / (3 \u00d7 {x_R:.3f})",
            result_name="q_toe",
            result_value=f"{result.q_toe:.1f}",
            result_unit="kPa",
            notes="q_heel = 0 kPa (resultant outside middle third).",
        ))
    else:
        brg_items.append(CalcStep(
            title="Heel Bearing Pressure (Triangular Distribution)",
            equation=("q_heel = 2V / (3 \u00d7 (B - x_R))  [resultant outside "
                      "middle third, heel side]"),
            substitution=(f"q_heel = 2 \u00d7 {total_W:.1f} / (3 \u00d7 "
                          f"({B:.2f} - {x_R:.3f}))"),
            result_name="q_heel",
            result_value=f"{result.q_heel:.1f}",
            result_unit="kPa",
            notes="q_toe = 0 kPa (resultant outside middle third).",
        ))

    # The equivalent uniform pressure over B' = B - 2e (GM-288)
    uniform = br.get("bearing_pressure", "linear") == "uniform"
    if br.get("B_prime", 0.0) > 0 and "q_eq" in br:
        brg_items.append(CalcStep(
            title="Equivalent Uniform Bearing Pressure",
            equation="B' = B - 2|e|,  q_eq = V / B'",
            substitution=(f"B' = {B:.2f} - 2 \u00d7 {abs(e):.3f} = "
                          f"{br['B_prime']:.3f} m; q_eq = {total_W:.1f} / "
                          f"{br['B_prime']:.3f}"),
            result_name="q_eq",
            result_value=f"{br['q_eq']:.1f}",
            result_unit="kPa",
            reference=("GEC-6 Eq. 6-6 and Sec. 6.4.2; FHWA NHI-06-089 Fig. "
                       "10-17; UFC 3-220-20 Eq. 4-27"),
            notes=("Compared with q_allowable on soil; the trapezoid or "
                   "triangle above is for rock and structural design."
                   if uniform else
                   "Reported; the larger edge pressure is the one compared "
                   "with q_allowable (see Choices)."),
        ))

    # Bearing check (GM-100): run only against a given q_allowable, on the
    # pressure the analysis compared; otherwise say plainly that it was
    # not run.
    q_max = (br["q_bearing"] if "q_bearing" in br
             else max(result.q_toe, result.q_heel))
    q_name = "q_eq" if uniform else "q_max"
    if q_all is None or result.FOS_bearing is None:
        brg_items.append(
            "Bearing capacity: NOT CHECKED — no allowable bearing "
            "pressure (q_allowable) was given. The bearing pressures above "
            f"and the eccentricity check are reported; compare {q_name} "
            "with the foundation's allowable bearing pressure.")
    elif q_max > 0:
        brg_items.append(CalcStep(
            title="Factor of Safety Against Bearing Failure",
            equation=(f"FOS_bearing = q_allowable / {q_name}"
                      + ("" if uniform else ",  q_max = max(q_toe, q_heel)")),
            substitution=f"FOS_bearing = {q_all:.1f} / {q_max:.1f}",
            result_name="FOS_bearing",
            result_value=f"{result.FOS_bearing:.3f}",
            reference=("GEC-6 Sec. 6.4.2; FHWA NHI-06-089 Sec. 10.5.4.1"
                       if uniform else "UFC 3-220-20 Sec. 4-5.1.4"),
        ))
        brg_items.append(CheckItem(
            description="Bearing capacity adequacy",
            demand=q_max,
            demand_label=q_name,
            capacity=q_all,
            capacity_label="q_allowable",
            unit="kPa",
            passes=bool(result.passes_bearing),
        ))
    else:
        brg_items.append(
            "Bearing capacity: FAILS — the resultant falls outside the "
            "base, so no bearing pressure can balance it (the wall tips "
            "over the toe).")

    sections.append(CalcSection(
        title="Bearing Pressure & Eccentricity", items=brg_items
    ))

    # ── Section 6: Summary ─────────────────────────────────────
    summary_items = []

    def _fos_row(name, fos, required, passes):
        if fos is None:
            return [name, "\u2014", required, "OK (no driving force)"]
        return [name, f"{fos:.3f}", required, "OK" if passes else "FAIL"]

    if result.FOS_bearing is None:
        bearing_row = ["Bearing", "\u2014", "N/A", "NOT CHECKED"]
    else:
        bearing_row = _fos_row("Bearing", result.FOS_bearing, "1.0",
                               result.passes_bearing)
    summary_table = TableData(
        title="Stability Check Summary",
        headers=["Check", "FOS Computed", "FOS Required", "Status"],
        rows=[
            _fos_row("Sliding", result.FOS_sliding, f"{FOS_s_req:.1f}",
                     result.passes_sliding),
            _fos_row("Overturning", result.FOS_overturning,
                     f"{FOS_o_req:.1f}", result.passes_overturning),
            bearing_row,
            [f"Eccentricity ({part})", "\u2014", f"e \u2264 {frac}",
             "OK" if within else "FAIL"],
        ],
    )
    summary_items.append(summary_table)

    eccentricity_status = (f"Within the {part}" if within
                           else f"OUTSIDE the {part}")
    summary_items.append(
        f"Eccentricity: e = {result.eccentricity:.3f} m "
        f"({frac} = {result.eccentricity_limit:.3f} m) \u2014 "
        f"{eccentricity_status}."
    )
    # The verdict in plain words: failed checks first, then not checked.
    summary_items.append(f"Result: {result.verdict}")

    sections.append(CalcSection(title="Stability Summary", items=summary_items))

    return sections


#: Where the MSE package's ASD requirements come from (GM-114).
_MSE_ASD = ("the module's ASD value: GEC-11 (2009) designs by LRFD and "
            "does not update the ASD procedure (Vol II App. F.1, p. F-2)")


def _mse_level_steps(t, geom):
    """The earth pressure steps for a level backfill (GEC-11 Vol I Eqs.
    4-1, 4-5, 4-6): the steps and P_a."""
    H = geom.wall_height
    q = geom.surcharge
    ep_items = []

    Ka = t["Ka"]
    phi_b, gamma_b = t["phi_retained"], t["gamma_retained"]
    ep_items.append(CalcStep(
        title="Rankine Active Earth Pressure Coefficient (retained fill)",
        equation="K_a = tan²(45° - φ_b/2)",
        substitution=f"K_a = tan²(45° - {phi_b:.1f}°/2)",
        result_name="K_a",
        result_value=f"{Ka:.4f}",
        reference="GEC-11 Vol I Eq. 4-1 (p. 4-15), vertical wall, level "
                  "backfill",
    ))

    # Active force: the thrusts the analysis summed
    Pa_total = t["Pa"]
    ep_items.append(CalcStep(
        title="Total Active Force on Reinforced Zone",
        equation="P_a = 0.5 × K_a × γ_b × H²"
                 + (" + K_a × q × H" if q > 0 else ""),
        substitution=(
            f"P_a = 0.5 × {Ka:.4f} × {gamma_b:.1f} × {H:.2f}²"
            + (f" + {Ka:.4f} × {q:.1f} × {H:.2f}" if q > 0 else "")
        ),
        result_name="P_a",
        result_value=f"{Pa_total:.1f}",
        result_unit="kN/m",
        reference="GEC-11 Vol I Eqs. 4-5, 4-6 (p. 4-20), Fig. 4-2",
        notes=("The live-load surcharge's thrust (K_a q H) drives every "
               "check." if q > 0 else ""),
    ))

    return ep_items, Pa_total


def _mse_backslope_steps(t, ld, geom):
    """The earth pressure steps for a sloping or broken backslope (GM-307),
    GEC-11 Vol I Sec. 4.4.4 as the analysis summed them
    (:func:`retaining_walls.mse.backslope_loads`): the slope's geometry,
    Kab by Eqs. 4-2 and 4-3, F_T by Eq. 4-7 and its components (Eqs.
    4-10, 4-13), the traffic thrust behind a broken backslope's crest, and
    the slope fill over the reinforced zone, V2. On a battered face, theta,
    the components at delta - omega and, under a slope, h and V2 to where
    the inclined back of the zone meets it (Fig. 4-5)."""
    H = geom.wall_height
    L = geom.reinforcement_length
    q = geom.surcharge
    bs = ld["geometry"]
    beta, I, h, S = bs["beta_deg"], ld["I_deg"], ld["h"], bs["S"]
    phi_b, gamma_b = t["phi_retained"], t["gamma_retained"]
    theta = ld.get("theta_deg", 90.0)
    omega = ld.get("omega_deg", 0.0)
    items = []
    if omega > 0:
        # GM-307: a face battered 10 deg or more (GEC-11 Vol I p. 4-15)
        items.append(CalcStep(
            title="Battered Face",
            equation="θ = 90° + ω;  δ = β",
            substitution=(f"ω = {omega:g}° (10° or more from vertical): θ = "
                          f"90° + {omega:g}° = {theta:g}°;  δ = β = "
                          f"{beta:g}°"),
            result_name="θ",
            result_value=f"{theta:g}",
            result_unit="deg",
            reference=("GEC-11 Vol I p. 4-15 ('Battered Wall with or "
                       "without Backslope'), Fig. 4-5 (p. 4-16)"),
            notes=("The reinforced zone is a parallelogram of horizontal "
                   "width L, its back L + y tan ω from the toe at height y; "
                   f"V1 at L/2 + (H/2) tan ω = {ld['x_V1']:.3f} m."),
        ))
    if beta <= 0:
        pass
    elif bs.get("h_rule") == "Fig. 4-5":
        # GM-307 review: on a battered face Fig. 4-5 carries the inclined
        # back of the reinforced zone up to the slope
        h7 = H + L * math.tan(math.radians(beta))
        items.append(CalcStep(
            title="Backslope Geometry (slope that continues)",
            equation="h = H + L tan β / (1 - tan ω tan β);  δ = β",
            substitution=(f"h = {H:.2f} + {L:.2f} × tan {beta:.1f}° / (1 - "
                          f"tan {omega:g}° × tan {beta:.1f}°) = {h:.3f} m;  "
                          f"δ = β = {beta:.1f}°"),
            result_name="h",
            result_value=f"{h:.3f}",
            result_unit="m",
            reference=("GEC-11 Vol I Fig. 4-5 (p. 4-16): h up to where the "
                       "back of the reinforced zone, inclined at θ, meets "
                       "the slope; Eq. 4-7 (p. 4-20); δ = β, p. 4-15"),
            notes=(f"Eq. 4-7's h = H + L tan β = {h7:.3f} m is written for "
                   "a vertical wall (Fig. 4-3) and gives a lower F_T (see "
                   "Choices)."),
        ))
    elif S is None:
        items.append(CalcStep(
            title="Backslope Geometry (slope that continues)",
            equation="h = H + L tan β;  δ = β",
            substitution=(f"h = {H:.2f} + {L:.2f} × tan {beta:.1f}° = "
                          f"{h:.3f} m;  δ = β = {beta:.1f}°"),
            result_name="h",
            result_value=f"{h:.3f}",
            result_unit="m",
            reference=("GEC-11 Vol I Eq. 4-7 (p. 4-20), Fig. 4-3; δ = β, "
                       "p. 4-15"),
        ))
    else:
        rise_2H = min(2.0 * H * math.tan(math.radians(beta)), S)
        items.append(CalcStep(
            title="Backslope Geometry (broken backslope)",
            equation=("I = atan(rise at 2H / 2H), the rise not more than S;"
                      "  β = δ = I;  h = H + L tan β ≤ H + S"),
            substitution=(
                f"I = atan({rise_2H:.3f} / {2.0 * H:.2f}) = {I:.2f}°;  "
                f"h = min({H:.2f} + {L:.2f} × tan {beta:.1f}°, {H:.2f} + "
                f"{S:.2f}) = {h:.3f} m"
                + (f"; crest {bs['crest_x']:.3f} m behind the face"
                   if bs["crest_x"] is not None else "")),
            result_name="h",
            result_value=f"{h:.3f}",
            result_unit="m",
            reference=("GEC-11 Vol I p. 4-15 and Fig. 4-4 (p. 4-14) for I; "
                       "Eq. 4-7 and its note (p. 4-20) for h"),
        ))
    # Kab, Eqs. 4-2 and 4-3, with delta and the slope both I
    pr, dr, tr, br = (math.radians(phi_b), math.radians(I),
                      math.radians(theta), math.radians(I))
    Gamma = (1.0 + math.sqrt(math.sin(pr + dr) * math.sin(pr - br)
                             / (math.sin(tr - dr) * math.sin(tr + br)))) ** 2
    Kab = ld["Kab"]
    items.append(CalcStep(
        title="Coulomb Active Earth Pressure Coefficient (retained fill)",
        equation=("K_ab = sin²(θ + φ_b) / [Γ sin²θ sin(θ - δ)];  Γ = [1 + "
                  "√(sin(φ_b + δ) sin(φ_b - β) / (sin(θ - δ) sin(θ + β)))]²"),
        substitution=(f"θ = {theta:.1f}°, φ_b = {phi_b:.1f}°, δ = β = "
                      f"{I:.2f}°: Γ = {Gamma:.4f}; K_ab = sin²({theta:.1f}° + "
                      f"{phi_b:.1f}°) / [{Gamma:.4f} × sin²{theta:.1f}° × "
                      f"sin({theta:.1f}° - {I:.2f}°)]"),
        result_name="K_ab",
        result_value=f"{Kab:.4f}",
        reference="GEC-11 Vol I Eqs. 4-2, 4-3 (p. 4-15)",
    ))
    FT, FTH, FTV = ld["FT"], ld["FTH"], ld["FTV"]
    items.append(CalcStep(
        title="Retained Backfill Thrust",
        equation="F_T = 0.5 × K_ab × γ_b × h²",
        substitution=f"F_T = 0.5 × {Kab:.4f} × {gamma_b:.1f} × {h:.3f}²",
        result_name="F_T",
        result_value=f"{FT:.1f}",
        result_unit="kN/m",
        reference=("GEC-11 Vol I Eq. 4-7 (p. 4-20), Fig. "
                   + ("4-5 (p. 4-16)" if omega > 0 else "4-3")),
    ))
    if omega > 0:
        a = I - omega
        items.append(CalcStep(
            title="Components of F_T (at δ to the normal of the back)",
            equation=("F_H = F_T cos(δ - ω), at h/3;  F_V = F_T sin(δ - ω), "
                      "at L + (h/3) tan ω"),
            substitution=(f"δ - ω = {I:.2f}° - {omega:g}° = {a:.2f}°: F_H = "
                          f"{FT:.1f} × cos({a:.2f}°) = {FTH:.1f};  F_V = "
                          f"{FT:.1f} × sin({a:.2f}°) = {FTV:.1f}, at "
                          f"{ld['x_FT']:.3f} m"),
            result_name="F_H",
            result_value=f"{FTH:.1f}",
            result_unit="kN/m",
            reference=("GEC-11 Vol I Fig. 4-5 (p. 4-16): F_T at δ to the "
                       "normal of the back of the reinforced zone, F_H at "
                       "θ - 90 to that normal, at h/3"),
            notes=("F_V below 0 points up and lessens the vertical load in "
                   "every check." if FTV < 0 else ""),
        ))
    else:
        items.append(CalcStep(
            title="Components of F_T (inclined at δ = β)",
            equation="F_H = F_T cos β, at h/3;  F_V = F_T sin β, at L",
            substitution=(f"F_H = {FT:.1f} × cos {I:.2f}° = {FTH:.1f};  "
                          f"F_V = {FT:.1f} × sin {I:.2f}° = {FTV:.1f}"),
            result_name="F_H",
            result_value=f"{FTH:.1f}",
            result_unit="kN/m",
            reference=("GEC-11 Vol I Eq. 4-10 (p. 4-21), F_T sin β in Eq. "
                       "4-13 (p. 4-21) and Eq. 4-16 (p. 4-23), arms h/3 "
                       "and L (Fig. 4-8, p. 4-25)"),
        ))
    if ld["F2"] > 0 and beta <= 0:
        items.append(CalcStep(
            title="Live-Load Surcharge Thrust",
            equation=("F2 = K_ab × q × H, at h/2 inclined as F_T: "
                      "F2 cos(δ - ω), F2 sin(δ - ω)"),
            substitution=(f"F2 = {Kab:.4f} × {q:.1f} × {h:.3f} = "
                          f"{ld['F2']:.1f};  F2 cos = {ld['F2H']:.1f}, "
                          f"F2 sin = {ld['F2V']:.1f}"),
            result_name="F2",
            result_value=f"{ld['F2']:.1f}",
            result_unit="kN/m",
            reference=("GEC-11 Vol I Eq. 4-6 (p. 4-20), Fig. 4-2 (p. "
                       "4-12); inclined as F_T, Fig. 4-5 (p. 4-16)"),
            notes=("The live-load surcharge's thrust drives every check; "
                   "its vertical component, pointing up, lessens V in "
                   "every check." if ld["F2V"] < 0 else
                   "The live-load surcharge's thrust drives every check."),
        ))
    elif ld["F2"] > 0:
        items.append(CalcStep(
            title="Traffic Thrust behind the Crest",
            equation=("F2 = K_ab × q × h, inclined at I: F2 cos I at h/2, "
                      "F2 sin I at L"),
            substitution=(f"F2 = {Kab:.4f} × {q:.1f} × {h:.3f} = "
                          f"{ld['F2']:.1f};  F2 cos I = {ld['F2H']:.1f}, "
                          f"F2 sin I = {ld['F2V']:.1f}"),
            result_name="F2",
            result_value=f"{ld['F2']:.1f}",
            result_unit="kN/m",
            reference=("GEC-11 Vol I Eq. 4-6 (p. 4-20) on h; Vol II "
                       "Example E1, pp. E1-7 to E1-9"),
            notes=_TRAFFIC_NOTE[bool(t.get("traffic_vertical_resists"))],
        ))
    if ld["V2"] > 0 and bs.get("h_rule") == "Fig. 4-5":
        # GM-307 review: the slope fill up to the inclined back (Fig. 4-5)
        items.append(CalcStep(
            title="Slope Fill over the Reinforced Zone",
            equation=("V2 = γ_r L (h - H) / 2, the triangle up to the "
                      "inclined back of the zone, at (2L + (h - H) tan ω) / "
                      "3 from the top of the face"),
            substitution=(f"V2 = {ld['V1'] / (H * L):.1f} × {L:.2f} × "
                          f"({h:.3f} - {H:.2f}) / 2 = {ld['V2']:.1f}, at "
                          f"{bs['V2_arm']:.3f} m from the top of the face"),
            result_name="V2",
            result_value=f"{ld['V2']:.1f}",
            result_unit="kN/m",
            reference=("GEC-11 Vol I Fig. 4-3 (p. 4-13) for V2 = γ_r L (h - "
                       "H) / 2, with Fig. 4-5's h (p. 4-16)"),
        ))
    elif ld["V2"] > 0:
        items.append(CalcStep(
            title="Slope Fill over the Reinforced Zone",
            equation=("V2 = γ_r × (area of the slope fill over L); "
                      "V2 = γ_r L (h - H) / 2 for the triangle"),
            substitution=(f"V2 = {ld['V1'] / (H * L):.1f} × "
                          f"{bs['V2_area']:.3f} m² = {ld['V2']:.1f}, at "
                          f"{bs['V2_arm']:.3f} m from the face"),
            result_name="V2",
            result_value=f"{ld['V2']:.1f}",
            result_unit="kN/m",
            reference=("GEC-11 Vol I Figs. 4-3 (p. 4-13), 4-4 (p. 4-14) and "
                       "4-8 (p. 4-25), Eq. 4-13 (p. 4-21)"),
        ))
    items.append(CalcStep(
        title="Horizontal Driving Force",
        equation="P_a = Σ horizontal components",
        substitution="P_a = " + " + ".join(
            f"{f['P']:.1f}" for f in t["horizontal_forces"]),
        result_name="P_a",
        result_value=f"{t['Pa']:.1f}",
        result_unit="kN/m",
        # GM-307 review: a battered face's thrust is at delta - omega
        reference=(("GEC-11 Vol I Fig. 4-5 (p. 4-16): F_H = F_T cos(δ - ω)"
                    + (", with F2 cos(δ - ω) added as Eq. 4-8 (p. 4-21) "
                       "adds F2" if ld["F2"] > 0 else "")
                    + ", unfactored") if omega > 0 else
                   "GEC-11 Vol I Eq. 4-10 (p. 4-21), unfactored"),
    ))
    return items


#: The traffic thrust's vertical component, by traffic_vertical_resists
_TRAFFIC_NOTE = {
    False: ("F2 sin I is counted in bearing only (GEC-11 Vol I p. 4-22: "
            "loads that increase sliding resistance only if permanent; p. "
            "4-26)."),
    True: ("F2 sin I is counted in sliding, overturning and bearing, as "
           "GEC-11 Vol II Example E1 does (traffic_vertical_resists)."),
}


def _calc_steps_mse(result, analysis) -> List[CalcSection]:
    """Calculation steps for MSE wall. The external steps print the terms
    the ASD analysis used (GM-143, GM-164; :func:`_mse_terms`)."""
    geom = _get(analysis, "geom")
    reinforcement = _get(analysis, "reinforcement")
    q_all = _get(analysis, "q_allowable")

    t = _mse_terms(result, analysis)
    phi, phi_f = t["phi_backfill"], t["phi_foundation"]

    sections = []
    H = geom.wall_height
    L = geom.reinforcement_length
    q = geom.surcharge
    forces = t["vertical_forces"]
    thrusts = t["horizontal_forces"]

    # ── Section 1: Earth Pressure ──────────────────────────────
    ld = t.get("backslope_loads")
    if ld is not None:
        # GM-307: a sloping or broken backslope, GEC-11 Vol I Sec. 4.4.4
        ep_items = _mse_backslope_steps(t, ld, geom)
        Pa_total = t["Pa"]
    else:
        ep_items, Pa_total = _mse_level_steps(t, geom)

    sections.append(CalcSection(title="Earth Pressure Computation", items=ep_items))

    # ── Section 2: External Stability ──────────────────────────
    ext_items = []

    # Every vertical force the analysis summed and the checks that count it
    # (GM-164: a live load over the reinforced zone is left out of sliding
    # and overturning and counted in bearing, GEC-11 Fig. 4-2)
    V, M_stab = t["V"], t["M_stabilizing"]
    V_b, M_b = t["V_bearing"], t["M_bearing"]
    split = abs(V - V_b) > 1e-9 or abs(M_stab - M_b) > 1e-9
    marked = split or any(not f["resists"] or not f["bearing"]
                          for f in forces)
    rows = []
    for f in forces:
        row = [f["component"], f"{f['W']:.1f}", f"{f['x']:.3f}",
               f"{f['M']:.1f}"]
        if marked:
            row.append(_counted_in(f, "sliding, overturning"))
        rows.append(row)
    if split:
        rows.append(["TOTAL, sliding / overturning", f"{V:.1f}", "—",
                     f"{M_stab:.1f}", ""])
        rows.append(["TOTAL, bearing", f"{V_b:.1f}", "—", f"{M_b:.1f}",
                     ""])
    else:
        rows.append(["TOTAL", f"{V:.1f}", "—", f"{M_stab:.1f}"]
                    + ([""] if marked else []))
    headers = ["Component", "Weight (kN/m)", "Arm from Toe (m)",
               "Moment (kN·m/m)"] + (["Counted in"] if marked else [])
    left_out = sum(f["W"] for f in forces
                   if f["kind"] == "surcharge" and not f["resists"])
    notes = "Moments about the toe of the reinforced zone."
    if marked:
        notes += (" Counted in: the checks that sum the force. The live load "
                  "over the reinforced zone is not a stabilizing force in "
                  "sliding or overturning (GEC-11 Vol I pp. 4-19, 4-21 "
                  "Eq. 4-12, 4-22) and is applied for bearing (p. 4-26; "
                  "Fig. 4-2); see Choices."
                  if left_out > 0 else
                  " Counted in: the checks that sum the force (see Choices).")
    ext_items.append(TableData(
        title="Vertical Forces and Moments about the Toe",
        headers=headers,
        rows=rows,
        notes=notes,
    ))

    # GM-297: the rule for the foundation cohesion; 2/3 c L added before
    R_slide = t["resisting"]
    slide_notes = (f"V leaves out the live load over the reinforced zone, "
                   f"{left_out:.1f} kN/m (GEC-11 p. 4-19)."
                   if left_out > 0 else "")
    if t.get("sliding_cohesion", "drained") == "drained":
        delta_b = t["delta_b"]
        c_f = t.get("c_foundation") or 0.0
        ext_items.append(CalcStep(
            title="Sliding Resistance",
            equation="R = V × tan(δ_b),  δ_b = min(φ_r, φ_f)",
            substitution=(
                f"δ_b = min({phi:.1f}°, {phi_f:.1f}°); "
                f"R = {V:.1f} × tan({delta_b:.2f}°)"),
            result_name="R",
            result_value=f"{R_slide:.1f}",
            result_unit="kN/m",
            reference="GEC-11 Vol I Eq. 4-12 (p. 4-21), μ p. 4-22",
            notes=(slide_notes + (
                f" c_foundation = {c_f:g} kPa is not counted: Eq. 4-12 has "
                f"no cohesion term (drained rule; see Choices)."
                if c_f > 0 else "")).strip(),
        ))
    elif t.get("gap_limited"):
        # EN 1997-1 Eq. 6.5 held the foundation's resistance to 0.4 V
        ext_items.append(CalcStep(
            title="Sliding Resistance (undrained foundation)",
            equation="R = min(c_u × L, 0.4 V, V × tan φ_r)",
            substitution=(
                f"R = min({t['c_foundation']:.1f} × {L:.2f}, 0.4 × {V:.1f}, "
                f"{V:.1f} × tan {phi:.1f}°) = min("
                f"{t['c_foundation'] * L:.1f}, {t['R_foundation']:.1f}, "
                f"{t['R_fill']:.1f}): along {t['sliding_along']}"),
            result_name="R",
            result_value=f"{R_slide:.1f}",
            result_unit="kN/m",
            reference=("GEC-11 Vol I p. 4-21 Step 3; UFC 3-220-20 p. 223; "
                       "EN 1997-1 Eq. 6.5 (interface_gap_possible)"),
            notes=slide_notes,
        ))
    else:
        ext_items.append(CalcStep(
            title="Sliding Resistance (undrained foundation)",
            equation="R = min(c_u × L, V × tan φ_r)",
            substitution=(
                f"R = min({t['c_foundation']:.1f} × {L:.2f}, {V:.1f} × "
                f"tan {phi:.1f}°) = min({t['R_foundation']:.1f}, "
                f"{t['R_fill']:.1f}): along {t['sliding_along']}"),
            result_name="R",
            result_value=f"{R_slide:.1f}",
            result_unit="kN/m",
            reference=("GEC-11 Vol I p. 4-21 Step 3; UFC 3-220-20 p. 223; "
                       "GEC-6 Eq. 5-39"),
            notes=slide_notes,
        ))

    sheet = t.get("sheet")
    FOS_base = R_slide / Pa_total if Pa_total > 0 else 99.9
    if sheet is not None:
        # GM-297 review: the third surface, the lowest sheet
        ext_items.append(CalcStep(
            title="Sliding Along the Lowest Sheet",
            equation=("FOS_s = V_s × tan ρ / P_a,s, the block above the "
                      "lowest sheet"),
            substitution=(
                f"z = {sheet['depth']:g} m; tan ρ = "
                + (f"tan {sheet['rho_tested']:g}°"
                   if sheet["rho_tested"] is not None
                   else f"2/3 × tan {phi:.1f}°")
                + f" = {sheet['tan_rho']:.4f}; FOS_s = {sheet['V']:.1f} × "
                f"{sheet['tan_rho']:.4f} / {sheet['Pa']:.1f}"),
            result_name="FOS_s",
            result_value=f"{sheet['FOS']:.3f}",
            reference=("GEC-11 Vol I p. 4-21 Step 3(iii), Eq. 3-9 (p. "
                       "3-21); Vol II E1 p. E1-7"),
            notes=("V_s and P_a,s are the weight and thrust of the wall "
                   "above the sheet, as at the base"
                   + (f"; V_s leaves out {sheet['V_left_out']:.1f} kN/m of "
                      "live load" if sheet["V_left_out"] > 0 else "")
                   + ". Sheet reinforcement: sliding 'along the weaker of "
                   "the upper and lower soil-reinforcement interfaces'."),
        ))
        ext_items.append(CalcStep(
            title="Factor of Safety Against Sliding",
            equation="FOS_sliding = min(R / P_a, FOS_s)",
            substitution=(f"FOS_sliding = min({R_slide:.1f} / "
                          f"{Pa_total:.1f}, {sheet['FOS']:.3f}) = min("
                          f"{FOS_base:.3f}, {sheet['FOS']:.3f}): "
                          + ("the lowest sheet governs"
                             if t.get("sliding_surface") == "lowest sheet"
                             else "the base governs")),
            result_name="FOS_sliding",
            result_value=f"{result.FOS_sliding:.3f}",
            reference=("GEC-11 Vol I p. 4-21 Step 3: the least of the "
                       "surfaces; ASD, required "
                       f"{result.FOS_sliding_required:g} (FOS_sliding_req), "
                       f"{_MSE_ASD}"),
        ))
    else:
        ext_items.append(CalcStep(
            title="Factor of Safety Against Sliding",
            equation="FOS_sliding = R / P_a",
            substitution=f"FOS_sliding = {R_slide:.1f} / {Pa_total:.1f}",
            result_name="FOS_sliding",
            result_value=f"{result.FOS_sliding:.3f}",
            reference=("GEC-11 Vol I Eq. 4-12 (p. 4-21) over Eqs. 4-5, 4-6 "
                       f"(p. 4-20), unfactored; ASD, required "
                       f"{result.FOS_sliding_required:g} (FOS_sliding_req), "
                       f"{_MSE_ASD}"),
        ))

    # Overturning: the thrusts' moments as the analysis summed them
    M_over = t["M_overturning"]
    ext_items.append(CalcStep(
        title="Overturning Moment about Toe",
        equation="M_ot = Σ P × z",
        substitution="M_ot = " + " + ".join(
            f"{f['P']:.1f} × {f['z']:.3f}" for f in thrusts),
        result_name="M_ot",
        result_value=f"{M_over:.1f}",
        result_unit="kN·m/m",
        reference=(("GEC-11 Vol I Fig. 4-5 (p. 4-16): F_H at h/3"
                    + (", F2 at h/2" if len(thrusts) > 1 else "")
                    + ", moments about the toe")
                   if getattr(geom, "battered", False) else
                   ("GEC-11 Vol I Eq. 4-16 (p. 4-23): F_T cos β at h/3"
                    + (", F2 cos I at h/2 (Vol II Example E1, pp. E1-7 to E1-9)"
                       if len(thrusts) > 1 else "")
                    + ", moments about the toe")
                   if t.get("backslope_loads") is not None else
                   "GEC-11 Vol I Eqs. 4-5, 4-6 (p. 4-20), F1 at H/3 and F2 "
                   "at H/2 as in Eq. 4-19 (p. 4-27), moments about the toe"),
    ))

    ext_items.append(CalcStep(
        title="Stabilizing Moment",
        equation="M_stab = Σ W × x (the forces counted in overturning)",
        substitution="M_stab = " + " + ".join(
            f"{f['W']:.1f} × {f['x']:.3f}" for f in forces
            if f["resists"]),
        result_name="M_stab",
        result_value=f"{M_stab:.1f}",
        result_unit="kN·m/m",
        reference=("GEC-11 Vol I Fig. 4-7 (p. 4-24), moments about the "
                   "toe; the live load over the reinforced zone left out "
                   "(p. 4-22)"),
    ))

    ext_items.append(CalcStep(
        title="Factor of Safety Against Overturning",
        equation="FOS_ot = M_stab / M_ot",
        substitution=f"FOS_ot = {M_stab:.1f} / {M_over:.1f}",
        result_name="FOS_ot",
        result_value=f"{result.FOS_overturning:.3f}",
        reference=(f"ASD, required {result.FOS_overturning_required:g} "
                   f"(FOS_overturning_req), {_MSE_ASD}; GEC-11 Vol I p. "
                   f"4-18: 'Limiting eccentricity (formerly known as "
                   f"overturning)', checked on pp. 4-22, 4-23"),
    ))

    # Bearing — Meyerhof uniform pressure over effective width (RW-3), on
    # the bearing loads: e_B is not the overturning check's (GEC-11 p. 4-26)
    e_B = t["eccentricity"]
    sigma_v = t["sigma_v"]
    ext_items.append(CalcStep(
        title="Eccentricity & Bearing Pressure (Meyerhof effective width)",
        equation=("e_B = L/2 - (M_b - M_ot)/V_b;  "
                  "σ_v = V_b / (L - 2e_B)"),
        substitution=(
            f"e_B = {L:.2f}/2 - ({M_b:.1f} - {M_over:.1f})/{V_b:.1f} "
            f"= {e_B:.3f} m;  σ_v = {V_b:.1f} / ({L:.2f} - "
            f"2×{max(e_B, 0.0):.3f})"
        ),
        result_name="σ_v",
        result_value=f"{sigma_v:.1f}",
        result_unit="kPa",
        # GM-307 review: keyed on the slope and the batter, not on the
        # backslope loads (a battered level wall has them too)
        reference=(("GEC-11 Vol I Eq. 4-18 (p. 4-26), Eq. 4-21's terms (p. "
                    "4-28) with F_T sin(δ - ω) in place of F_T sin β (Fig. "
                    "4-5, p. 4-16), unfactored"
                    if getattr(geom, "battered", False)
                    and geom.backfill_slope > 0 else
                    "GEC-11 Vol I Eq. 4-18 (p. 4-26) with Fig. 4-5's terms "
                    "(p. 4-16), unfactored"
                    if getattr(geom, "battered", False) else
                    "GEC-11 Vol I Eq. 4-18 (p. 4-26), Eq. 4-21 (p. 4-28) "
                    "with Eq. 4-16's terms (p. 4-23), unfactored"
                    if geom.backfill_slope > 0 else
                    "GEC-11 Vol I Eq. 4-18 (p. 4-26), Eqs. 4-19, 4-20 (p. "
                    "4-27), unfactored") + "; AASHTO 11.10.5.4"),
        notes=("V_b and M_b count the live load over the reinforced zone "
               "(GEC-11 p. 4-26), so e_B is not the eccentricity of the "
               "sliding and overturning loads."
               if split else ""),
    ))

    bearing_checked = q_all is not None and result.FOS_bearing is not None
    if bearing_checked:
        ext_items.append(CalcStep(
            title="Factor of Safety Against Bearing",
            equation="FOS_bearing = q_allowable / σ_v",
            substitution=f"FOS_bearing = {q_all:.1f} / {sigma_v:.1f}",
            result_name="FOS_bearing",
            result_value=f"{result.FOS_bearing:.3f}",
            reference=("GEC-11 Vol I Eq. 4-17 (p. 4-26) with the given "
                       "allowable bearing pressure in place of the factored "
                       "resistance; ASD, required 1.0"),
        ))
    else:
        # GM-100: a bearing check not run is said, never counted as a pass.
        ext_items.append(
            "Bearing capacity: NOT CHECKED — no allowable bearing "
            "pressure (q_allowable) was given. The bearing pressure σ_v "
            "above is reported; compare it with the foundation's allowable "
            "bearing pressure.")

    # GM-302: GEC-11 Eq. 4-24 on a cohesive foundation
    ls = result.local_shear
    if ls is not None:
        ext_items.append(CalcStep(
            title="Local Shear on a Cohesive Foundation",
            equation="γ_r H ≤ 3 c_u",
            substitution=(f"γ_r H = {ls['gamma_r_H_kPa'] / H:.1f} × "
                          f"{H:.2f} = {ls['gamma_r_H_kPa']:.1f} kPa;  3 c_u = "
                          f"3 × {ls['three_cu_kPa'] / 3:.1f} = "
                          f"{ls['three_cu_kPa']:.1f} kPa"),
            result_name="γ_r H ≤ 3 c_u",
            result_value="yes" if ls["passes"] else "NO",
            reference="GEC-11 Vol I Eq. 4-24 (p. 4-30)",
            notes=("'To prevent local shear of structures bearing on weak "
                   "cohesive soils'; c_u = c_foundation, the undrained "
                   "sliding rule."),
        ))
        ext_items.append(CheckItem(
            description="Local shear (GEC-11 Eq. 4-24)",
            demand=ls["gamma_r_H_kPa"],
            demand_label="γ_r H (kPa)",
            capacity=ls["three_cu_kPa"],
            capacity_label="3 c_u (kPa)",
            passes=ls["passes"],
        ))

    # External stability summary: only the checks that were run
    fos_s_req = result.FOS_sliding_required
    fos_o_req = result.FOS_overturning_required
    pass_s = result.FOS_sliding >= fos_s_req
    pass_o = result.FOS_overturning >= fos_o_req
    run_fos = [result.FOS_sliding, result.FOS_overturning]
    run_pass = [pass_s, pass_o]
    if bearing_checked:
        run_fos.append(result.FOS_bearing)
        run_pass.append(result.FOS_bearing >= 1.0)
    ext_items.append(CheckItem(
        description=("External stability (sliding, overturning, bearing)"
                     if bearing_checked else
                     "External stability (sliding, overturning; bearing "
                     "NOT CHECKED)"),
        demand=1.0,
        demand_label="Required",
        capacity=min(run_fos),
        capacity_label="Min FOS",
        passes=all(run_pass),
    ))

    ext_summary = TableData(
        title="External Stability Summary",
        headers=["Check", "FOS", "Required", "Status"],
        rows=[
            ["Sliding", f"{result.FOS_sliding:.3f}", f"{fos_s_req:.1f}",
             "OK" if pass_s else "FAIL"],
            ["Overturning", f"{result.FOS_overturning:.3f}",
             f"{fos_o_req:.1f}", "OK" if pass_o else "FAIL"],
            (["Bearing", f"{result.FOS_bearing:.3f}", "1.0",
              "OK" if result.FOS_bearing >= 1.0 else "FAIL"]
             if bearing_checked else
             ["Bearing", "—", "N/A", "NOT CHECKED"]),
        ],
    )
    ext_items.append(ext_summary)

    sections.append(CalcSection(title="External Stability", items=ext_items))

    # ── Section 3: Internal Stability ──────────────────────────
    int_items = []

    rs, per_r = _level_reinforcements(reinforcement, geom)
    first = rs[0] if rs else None
    r_type = "Metallic" if (first and first.is_metallic) else "Geosynthetic"
    # GM-266: the curve for each kind of steel reinforcement the wall has
    kr_text = "; ".join(
        text for kind, text in (
            ("strip", "strips: Kr/Ka = 1.7 at z=0 to 1.2 at z\u226520 ft "
                      "(6.096 m)"),
            ("grid", "bar mats and welded wire grids: Kr/Ka = 2.5 at z=0 "
                     "to 1.2 at z\u226520 ft (6.096 m)"))
        if any((r.is_grid if kind == "grid" else not r.is_grid)
               for r in (rs or [])))

    int_items.append(CalcStep(
        title="Lateral Earth Pressure Ratio (Kr/Ka)",
        equation="Kr/Ka varies with depth per GEC-11 Fig 4-10"
                 if r_type == "Metallic"
                 else "Kr/Ka = 1.0 for geosynthetic reinforcement",
        substitution=(
            "Metallic " + kr_text
            if r_type == "Metallic"
            else "Geosynthetic: Kr/Ka = 1.0 (constant)"
        ),
        result_name="Kr/Ka",
        result_value="variable" if r_type == "Metallic" else "1.0",
        reference="GEC-11 Vol I Fig. 4-10 (p. 4-37)",
    ))

    sloping = geom.backfill_slope > 0
    psi = None
    if getattr(geom, "battered", False):
        # GM-307: Eq. 4-26 (p. 4-36) and Fig. 4-9(b) (p. 4-34)
        from retaining_walls import mse as _mse
        phi_r = _mse_used(analysis, "phi_backfill")
        theta = geom.theta_deg
        Ka_i = _mse.battered_internal_Ka(phi_r, theta)
        printed = _mse.battered_internal_Ka_as_printed(phi_r, theta)
        int_items.append(CalcStep(
            title="Internal Ka for a Battered Face",
            equation=("K_a = sin²(θ + φ_r) / [sin³θ (1 + sin φ_r / "
                      "sin θ)²]"),
            substitution=(f"θ = {theta:g}°, φ_r = {phi_r:.1f}°: K_a = "
                          f"sin²({theta:g}° + {phi_r:.1f}°) / [sin³"
                          f"{theta:g}° × (1 + sin {phi_r:.1f}° / sin "
                          f"{theta:g}°)²]"),
            result_name="K_a",
            result_value=f"{Ka_i:.4f}",
            reference=("GEC-11 Vol I Eq. 4-26 (p. 4-36), taken as Eqs. "
                       "4-2, 4-3 (p. 4-15) with δ = β = 0"),
            notes=("Eq. 4-26 is printed with no square on the bracket, "
                   f"which gives K_a = {printed:.4f} here and 1 - sin φ_r, "
                   "not Eq. 4-25, at θ = 90°; p. 4-36 calls it the Coulomb "
                   "K_a with no wall friction and β = 0, which reduces to "
                   "Eq. 4-25 for a vertical wall, so the square is kept "
                   "(see Choices)."),
        ))
        if rs and any(not r.is_metallic for r in rs):
            psi = _mse.extensible_failure_angle(phi_r, theta,
                                                geom.backfill_slope)
            int_items.append(CalcStep(
                title="Failure Surface for a Battered Face (extensible)",
                equation=("tan(ψ - φ) = [-tan(φ - β) + √(tan(φ - β)[tan(φ - "
                          "β) + cot(φ + θ - 90)][1 + tan(δ + 90 - θ) cot(φ + "
                          "θ - 90)])] / [1 + tan(δ + 90 - θ)[tan(φ - β) + "
                          "cot(φ + θ - 90)]],  δ = β;  L_a = (H - z)(cot ψ "
                          "- tan ω)"),
                substitution=(f"φ = {phi_r:.1f}°, θ = {theta:g}°, β = δ = "
                              f"{geom.backfill_slope:g}°"),
                result_name="ψ",
                result_value=f"{psi:.2f}",
                result_unit="deg",
                reference=("GEC-11 Vol I p. 4-33, Fig. 4-9(b) (p. 4-34), "
                           "through the toe"),
                notes=("The figure's left side reads tan(ψ - θ); only "
                       "tan(ψ - φ) gives its 'For vertical face ψ = 45 + "
                       "φ/2', so it is read so (see Choices)."),
            ))
    q_text = "q"
    if sloping:
        # GM-307: the slope's surcharge on the reinforcement (Fig. 4-11)
        from retaining_walls import mse as _mse
        bs = _mse.backslope_geometry(geom)
        gamma_r = _mse_used(analysis, "gamma_backfill")
        S = bs["S"]
        q_int = q if bs["level_length"] > 0 else 0.0
        q_text = ("σ_2 + q" if q_int > 0 else "σ_2")
        int_items.append(CalcStep(
            title="Sloping Backfill Surcharge on the Reinforcement",
            equation=("S_eq = ½ (0.7H) tan β" + (" ≤ S" if S else "")
                      + ";  σ_2 = γ_r S_eq"),
            substitution=(
                f"S_eq = 0.5 × 0.7 × {H:.2f} × tan {bs['beta_deg']:.1f}°"
                + (f", not more than S = {S:.2f}" if S else "")
                + f" = {bs['S_eq']:.3f} m;  σ_2 = {gamma_r:.1f} × "
                f"{bs['S_eq']:.3f} = {gamma_r * bs['S_eq']:.2f} kPa"
                + (f"; q = {q:.1f} kPa counted, level ground lying over "
                   "the reinforced zone behind the crest" if q_int > 0 else
                   (f"; q = {q:.1f} kPa not counted, no level ground over "
                    "the reinforced zone" if q > 0 else ""))),
            result_name="σ_2",
            result_value=f"{gamma_r * bs['S_eq']:.2f}",
            result_unit="kPa",
            # GM-307 review: a battered face takes Eq. 4-26's Ka
            reference=("GEC-11 Vol I Fig. 4-11 (p. 4-38), Eq. 4-29 (p. "
                       "4-42), p. 4-36: Ka by Eq. "
                       + ("4-26" if getattr(geom, "battered", False)
                          else "4-25")
                       + " as for a level backfill"),
        ))
        if rs and any(r.is_metallic for r in rs):
            int_items.append(CalcStep(
                title="Height of the Zone of Maximum Stress (inextensible)",
                equation="H1 = H + tan β × 0.3H / (1 - 0.3 tan β)"
                         + (", the rise not more than S" if S else ""),
                substitution=(f"H1 = {H:.2f} + tan {bs['beta_deg']:.1f}° × "
                              f"0.3 × {H:.2f} / (1 - 0.3 tan "
                              f"{bs['beta_deg']:.1f}°)"),
                result_name="H1",
                result_value=f"{bs['H1']:.3f}",
                result_unit="m",
                reference="GEC-11 Vol I Fig. 4-9(a) (p. 4-34)",
                notes=("L_a = 0.3 H1 over the upper H1/2, then tapering "
                       "to the toe."),
            ))

    if geom.level_depths is None:
        # GM-277: each level over its contributory height, the bottom
        # level's down to the base (GEC-11 Vol I p. 4-47, Fig. 4-14)
        substitution = ("\u03c3_v = \u03b3\u00b7z + " + q_text + "; each level's "
                        "contributory height S_v with the level at its "
                        "middle, the top level's from the top of the wall")
        a, b = geom.contributory_heights[-1]
        if geom.n_reinforcement_levels == 1:
            substitution = (
                f"\u03c3_v = \u03b3\u00b7z + {q_text}; one level, its contributory "
                f"height the whole wall, S_vt = {b:.2f} m, with \u03c3_v "
                f"and Kr/Ka at its middle, z = {b / 2:.2f} m")
        elif geom.whole_spacings:
            substitution += " and the bottom level's to its base"
        else:
            substitution += (
                f"; the bottom level's from {a:.2f} m down to the base, "
                f"S_vt = {b - a:.2f} m, with \u03c3_v and Kr/Ka at its "
                f"middle, z = {(a + b) / 2:.2f} m")
        int_items.append(CalcStep(
            title="Maximum Reinforcement Tension",
            equation="T_max = (Kr/Ka) \u00d7 K_a \u00d7 \u03c3_v \u00d7 S_v",
            substitution=substitution,
            result_name="T_max",
            result_value="(per level)",
            result_unit="kN/m",
            reference="GEC-11 Vol I Eq. 4-32a, pp. 4-46, 4-47 (Fig. 4-14)",
        ))
    else:
        # GM-273: each level over its contributory height S_vt
        int_items.append(CalcStep(
            title="Maximum Reinforcement Tension",
            equation=("T_max = \u03c3_H \u00d7 S_vt, \u03c3_H = (Kr/Ka) "
                      "\u00d7 K_a \u00d7 \u03c3_v"),
            substitution=(
                "\u03c3_v = \u03b3\u00b7z + " + q_text + "; S_vt from the midpoints "
                "between levels, the top level's from the top of the wall "
                "and the bottom level's to its base; "
                + _sigma_H_text(analysis)),
            result_name="T_max",
            result_value="(per level)",
            result_unit="kN/m",
            reference="GEC-11 Vol I Eq. 4-32a, pp. 4-46, 4-47 (Fig. 4-14)",
        ))

    from retaining_walls import mse as _mse
    f_tested = _get(analysis, "F_star_tested")
    if f_tested is not None:
        # GM-256: F* from pullout tests (GEC-11 Table 3-6 note (i))
        int_items.append(CalcStep(
            title="Pullout Resistance Factor F* (from pullout tests)",
            equation="F* from product- and backfill-specific pullout tests",
            substitution=_mse._tested_text(f_tested),
            result_name="F*",
            result_value="tested",
            reference="GEC-11 Vol I Table 3-6 note (i) (p. 3-17), "
                      "Appendix B",
        ))
    elif rs is not None and any(_mse.uses_strip_F_star(r) for r in rs):
        # GM-246: the ribbed steel strip F*, from the backfill's Cu
        cu_given = _get(analysis, "Cu_backfill")
        cu = _mse.CU_UNKNOWN if cu_given is None else cu_given
        phi_b = _get(analysis, "phi_backfill", 34.0)
        f_top = _mse.F_star_strip_top(cu)
        f_deep = math.tan(math.radians(phi_b))
        int_items.append(CalcStep(
            title="Pullout Resistance Factor F* (ribbed steel strips)",
            equation=("F* = 1.2 + log C_u ≤ 2.0 at the top of the "
                      "wall; F* = tan φ at 20 ft (6 m) and below; "
                      "linear between"),
            substitution=(
                f"C_u = {cu:g}"
                + (" (not given: GEC-11 p. 3-20 says to assume C_u = 4 "
                   "when the backfill's is unknown)" if cu_given is None
                   else "")
                + (" (below 4: GEC-11 Table 3-6 note (i), 'For fill "
                   "outside these limits, tests must be run')"
                   if cu < _mse.CU_UNKNOWN else "")
                + f": F* = {f_top:.2f} at z = 0; tan {phi_b:.1f}° = "
                f"{f_deep:.3f} at z ≥ 6.10 m"),
            result_name="F*",
            result_value=f"{f_top:.2f} → {f_deep:.3f}",
            reference="GEC-11 Vol I Eqs. 3-4, 3-5 (p. 3-20)",
        ))

    # GM-261: one R_c a level when the spacing is given per level
    per_level = reinforcement is not None and (
        per_r or reinforcement.per_level)
    Rcs = (_mse.coverage_ratios(reinforcement, geom.reinforcement_depths)
           if reinforcement is not None else [1.0])
    Rc_text = (_per_level_text(tuple(Rcs)) + " per level from the top"
               if per_level else f"{Rcs[0]:.3f}")
    alpha_t = _get(analysis, "alpha_tested")
    alphas = ([_mse.pullout_alpha(r, alpha_t) for r in rs] if rs
              else [alpha_t if alpha_t is not None else 1.0])
    alpha_text = (f"{alphas[0]:g}" if len(set(alphas)) == 1 else
                  _per_level_text(tuple(alphas), "g")
                  + " per level from the top")
    int_items.append(CalcStep(
        title="Pullout Resistance",
        equation=("P_r = F* \u00d7 \u03b1 \u00d7 \u03c3_v' \u00d7 L_e "
                  "\u00d7 C \u00d7 R_c"),
        substitution=(
            (("L_e = L - L_a; L_a per bilinear coherent-gravity surface "
              "(0.3H1 over upper H1/2, tapering to toe; Fig. 4-9(a), p. "
              "4-34) for inextensible (metallic) reinforcement"
              if sloping else
              "L_e = L - L_a; L_a per bilinear coherent-gravity surface "
              "(0.3H over upper H/2, tapering to toe) for inextensible "
              "(metallic) reinforcement")
             + ("; on the battered face the upper part parallel to the "
                "face at 0.3H1 from it (Fig. 4-9(a) note, p. 4-34)"
                if getattr(geom, "battered", False) else "")
             if r_type == "Metallic" else
             f"L_a = (H - z)(cot ψ - tan ω), ψ = {psi:.2f}° (Fig. 4-9(b), "
             "p. 4-34)" if psi is not None else
             "L_e = L - L_a; L_a from Rankine 45+\u03c6/2 plane "
             "(extensible/geosynthetic reinforcement)")
            + f"; C = 2; \u03b1 = {alpha_text}"
            + (" (from pullout tests)" if alpha_t is not None
               else " (GEC-11 Table 3-6, p. 3-17)")
            + f"; R_c = {Rc_text}"
            + ("; \u03c3_v' = \u03b3_r Z_p, Z_p the depth below the slope "
               "surface averaged over L_e (GEC-11 Vol I Fig. 4-15, p. "
               "4-50; Vol II Example E3), per level in the table"
               if sloping else "")
        ),
        result_name="P_r",
        result_value="(per level)",
        result_unit="kN/m of wall",
        reference="GEC-11 Vol I Eq. 3-2 (p. 3-16), Eq. 4-35 (p. 4-49); "
                  "AASHTO Fig. 11.10.6.3.1-1",
    ))

    # GM-262 review: 1.0 on a steel allowable, 1.5 on a geosynthetic's
    # Tult / RF (GEC-11's 1.35 / phi)
    FS_ru = (_mse.rupture_FOS_required(reinforcement)
             if reinforcement is not None else 1.0)
    if per_r:
        # GM-266: each level's own Tal times its own R_c
        tal_rc = [r.Tallowable * Rc for r, Rc in zip(rs, Rcs)]
        int_items.append(CalcStep(
            title="Tensile Resistance per Metre of Wall",
            equation=("T_al \u00d7 R_c \u2265 T_max" if FS_ru == 1.0 else
                      f"T_al \u00d7 R_c \u2265 {FS_ru:g} T_max"),
            substitution=(
                "T_al " + _per_level_text(tuple(r.Tallowable for r in rs),
                                          ".1f")
                + " kN/m of reinforcement width per level from the top "
                f"\u00d7 R_c {Rc_text}"),
            result_name="T_al \u00d7 R_c",
            result_value=f"{min(tal_rc):.1f} to {max(tal_rc):.1f}",
            result_unit="kN/m of wall",
            reference="GEC-11 Vol I Eq. 3-11a (p. 3-22), p. 4-47, "
                      "Eqs. 4-33 and 4-34 (pp. 4-48, 4-49)",
        ))
    elif reinforcement is not None:
        int_items.append(CalcStep(
            title="Tensile Resistance per Metre of Wall",
            equation=("T_al \u00d7 R_c \u2265 T_max" if FS_ru == 1.0 else
                      f"T_al \u00d7 R_c \u2265 {FS_ru:g} T_max"),
            substitution=(f"{reinforcement.Tallowable:.1f} kN/m of "
                          f"reinforcement width \u00d7 {Rc_text}"),
            result_name="T_al \u00d7 R_c",
            result_value=(
                f"{reinforcement.Tallowable * min(Rcs):.1f} to "
                f"{reinforcement.Tallowable * max(Rcs):.1f}" if per_level
                else f"{reinforcement.Tallowable * Rcs[0]:.1f}"),
            result_unit="kN/m of wall",
            reference="GEC-11 Vol I Eq. 3-11a (p. 3-22), p. 4-47, "
                      "Eqs. 4-33 and 4-34 (pp. 4-48, 4-49)",
        ))

    # Per-level results table
    if result.internal_results:
        int_rows = []
        keys = _reinforcement_keys(rs) if per_r else {}
        svt = geom.level_depths is not None   # GM-273
        for i, r in enumerate(result.internal_results):
            status = "OK" if r.get("passes", False) else "FAIL"
            int_rows.append([
                f"{r['depth_m']:.2f}"] + (
                [keys[_reinforcement_label(rs[i])]] if per_r else []) + (
                [f"{r['Svt_m']:.2f}"] if svt else []) + (
                [f"{r['Zp_m']:.2f}"] if sloping else []) + [
                f"{r.get('Kr_Ka', 0):.3f}",
                f"{r['Tmax_kN_per_m']:.2f}",
                f"{r.get('Tal_Rc_kN_per_m', 0):.1f}",
                f"{r.get('Le_m', 0):.2f}",
                f"{r.get('F_star', 0):.3f}",
                f"{r['Pr_kN_per_m']:.2f}",
                f"{r['FOS_pullout']:.2f}",
                f"{r.get('FOS_rupture', 0):.2f}",
                status,
            ])

        int_items.append(TableData(
            title="Internal Stability per Reinforcement Level",
            headers=["z (m)"] + (["Reinf."] if per_r else [])
            + (["S_vt (m)"] if svt else [])
            + (["Z_p (m)"] if sloping else []) + [
                     "Kr/Ka", "T_max (kN/m)", "T_al R_c (kN/m)",
                     "L_e (m)", "F*", "P_r (kN/m)",
                     "FOS_po", "FOS_ru", "Status"],
            rows=int_rows,
            notes=(
                "L_e = L - L_a, at least 3 ft (0.91 m) (GEC-11 Eq. 4-36): "
                "a level with less fails. "
                f"FOS_pullout required: 1.5. FOS_rupture required: "
                + ("1.0 (T_al the steel allowable, 0.55 or 0.48 F_y). "
                   if FS_ru == 1.0 else
                   f"{FS_ru:g} (1.35 / phi 0.90 on T_al = T_ult / RF). ")
                + f"T_al R_c and P_r per metre of wall, coverage ratio "
                f"R_c = {Rc_text}. "
                + ("Reinforcement per level: " + "; ".join(
                    f"{k} = {n}" for n, k in keys.items()) if per_r else
                   f"Reinforcement: {reinforcement.name if reinforcement else 'N/A'}")
            ),
        ))

    int_items.append(CheckItem(
        description="Internal stability (all levels)",
        demand=1.0,
        demand_label="Required",
        capacity=1.0 if result.all_pass_internal else 0.0,
        capacity_label="All pass",
        passes=result.all_pass_internal,
    ))

    # GM-302: a steel grid's transverse-bar spacing (GEC-11 p. 3-21)
    gs = result.grid_spacing
    if gs is not None:
        int_items.append(CheckItem(
            description="Steel grid transverse-bar spacing St "
                        "(GEC-11 Vol I p. 3-21: at most 24 in.)",
            demand=gs["St_max_m"] * 1000,
            demand_label="St (mm)",
            capacity=gs["St_limit_m"] * 1000,
            capacity_label="Maximum (mm)",
            passes=gs["passes"],
        ))

    sections.append(CalcSection(title="Internal Stability (GEC-11)", items=int_items))

    # ── Section 4: Overall Summary ─────────────────────────────
    summary_items = []

    def _status(passes):
        if passes is None:
            return "NOT ALL CHECKED (bearing not checked)"
        return "OK" if passes else "FAIL"

    grid_ok = gs is None or gs["passes"]
    if (result.passes_external is False or not result.all_pass_internal
            or not grid_ok):
        overall_pass = False
    elif result.passes_external is None:
        overall_pass = None
    else:
        overall_pass = True
    summary_items.append(TableData(
        title="Overall MSE Wall Assessment",
        headers=["Category", "Status"],
        rows=[
            ["External Stability", _status(result.passes_external)],
            ["Internal Stability", _status(result.all_pass_internal)],
        ] + ([["Grid Spacing St", _status(gs["passes"])]]
             if gs is not None else []) + [
            ["Overall", _status(overall_pass)],
        ],
    ))
    summary_items.append(f"Result: {result.verdict}")
    # GM-302: GEC-11's advice the wall goes against
    if result.warnings:
        summary_items.append(TableData(
            title="Warnings (GEC-11 advice, not a check)",
            headers=["Warning"],
            rows=[[w] for w in result.warnings],
        ))

    sections.append(CalcSection(title="Overall Assessment", items=summary_items))

    return sections


# ════════════════════════════════════════════════════════════════════
#  MSE LRFD EXTERNAL STABILITY (GM-307)
# ════════════════════════════════════════════════════════════════════

#: Where each LRFD factor comes from, GEC-11 (FHWA-NHI-10-024/025) as
#: printed (GM-307): name in the result's load_factors -> (what, source).
_LRFD_FACTOR_SOURCES = (
    ("EV_max", "γ_EV max, vertical earth pressure (retaining walls)",
     "GEC-11 Vol I Table 4-2 (p. 4-4): 1.35"),
    ("EV_min", "γ_EV min", "GEC-11 Vol I Table 4-2 (p. 4-4): 1.00"),
    ("EH_max", "γ_EH max, horizontal earth pressure (active)",
     "GEC-11 Vol I Table 4-2 (p. 4-4): 1.50"),
    ("EH_min", "γ_EH min", "GEC-11 Vol I Table 4-2 (p. 4-4): 0.90"),
    ("LL", "γ_LL, live load and live-load surcharge, Strength I",
     "GEC-11 Vol I Table 4-1 (p. 4-4): 1.75; Service I 1.00"),
    ("phi_sliding", "φ_τ, sliding resistance",
     "GEC-11 Vol I Table 4-5 (p. 4-19): 1.0; Eq. 4-4"),
    ("ecc_limit_ratio", "e_max / L, Strength I",
     "GEC-11 Vol I p. 4-23: L/4 on soil, 3/8 L on rock"),
    ("ecc_limit_ratio_service", "e_max / L, Service I",
     "GEC-11 Vol II Tables E3-6.3 (p. E3-11), E4-6.3 (p. E4-9): L/6"),
)

#: The values GEC-11 prints for them (the sources above).
_LRFD_PRINTED = {"EV_max": 1.35, "EV_min": 1.00, "EH_max": 1.50,
                 "EH_min": 0.90, "LL": 1.75, "phi_sliding": 1.0,
                 "ecc_limit_ratio": (0.25, 0.375),
                 "ecc_limit_ratio_service": 1 / 6}

#: The combinations the tool checks (GEC-11 Vol II Table E4-5.1, p. E4-6;
#: the critical one, Vol I p. 4-23 and Vol II Table E4-6.3, p. E4-9).
_LRFD_COMBINATIONS = (
    ("Strength I (max)", "EV_max", "EH_max", "LL"),
    ("Strength I (min)", "EV_min", "EH_min", "LL"),
    ("Critical (EV min, EH max)", "EV_min", "EH_max", "LL"),
    ("Service I", None, None, None),
)


def _f(value, fmt=".3f", none="—"):
    return none if value is None else format(value, fmt)


def _input_summary_mse_lrfd(result, analysis) -> List[InputItem]:
    """The LRFD package's inputs (GM-307): the geometry and the inputs the
    mse_lrfd_external_stability run used."""
    geom = _get(analysis, "geom")
    items = [
        InputItem("H", "Wall height", f"{geom.wall_height:.2f}", "m"),
        InputItem("L", "Reinforcement length",
                  f"{geom.reinforcement_length:.2f}", "m"),
    ]
    if geom.backfill_slope > 0:
        items.append(InputItem("β", "Backfill slope",
                               f"{geom.backfill_slope:.1f}", "deg"))
    if geom.broken_backslope_height is not None:
        items.append(InputItem("S", "Broken backslope height",
                               f"{geom.broken_backslope_height:.2f}", "m"))
    q = _get(analysis, "live_load")
    q = geom.surcharge if q is None else q
    if q:
        items.append(InputItem("q", "Live-load surcharge", f"{q:.1f}", "kPa"))
    for key, sym, name, unit in (
            ("gamma_backfill", "γ_r", "Reinforced fill unit weight",
             "kN/m³"),
            ("phi_backfill", "φ_r", "Reinforced fill friction angle", "deg"),
            ("phi_retained", "φ_b", "Retained fill friction angle", "deg"),
            ("gamma_retained", "γ_b", "Retained fill unit weight", "kN/m³"),
            ("phi_foundation", "φ_f", "Foundation friction angle", "deg"),
            ("c_foundation", "c_f", "Foundation cohesion", "kPa"),
            ("gamma_foundation", "γ_f", "Foundation unit weight", "kN/m³"),
            ("bearing_resistance_strength", "q_R",
             "Factored bearing resistance (Strength I), given", "kPa"),
            ("bearing_resistance_service", "q_R,s",
             "Bearing resistance (Service I), given", "kPa")):
        value = _get(analysis, key)
        if value is not None:
            items.append(InputItem(sym, name, f"{value:g}", unit))
    return items


def _calc_steps_mse_lrfd(result, analysis) -> List[CalcSection]:
    """The LRFD external-stability package's steps (GM-307): the result of
    mse_lrfd_external_stability as the tool returned it, each combination
    (Strength I max and min, the critical one, Service I) with the
    equations and the load and resistance factors as GEC-11 prints them.
    Nothing is worked out here: every number is the tool's."""
    geom = _get(analysis, "geom")
    lf = result["load_factors"]
    uf = result["unfactored_kN_m"]
    sl, ec, br = (result["sliding"], result["eccentricity"],
                  result["bearing"])
    sloping = geom.backfill_slope > 0
    sections = []

    # ── Factors ───────────────────────────────────────────────
    rows, changed = [], []
    for key, what, src in _LRFD_FACTOR_SOURCES:
        if key in lf:
            rows.append([what, f"{lf[key]:.3f}" if key.startswith("ecc")
                         else f"{lf[key]:.2f}", src])
            printed = _LRFD_PRINTED[key]
            if all(abs(lf[key] - v) > 1e-9 for v in (
                    printed if isinstance(printed, tuple) else (printed,))):
                changed.append(what)
    if br.get("phi_bearing") is not None:
        rows.append(["φ_b, bearing resistance", f"{br['phi_bearing']:.2f}",
                     "GEC-11 Vol I Table 4-5 (p. 4-19): 0.65; Eq. 4-23 (p. "
                     "4-30)"])
        if abs(br["phi_bearing"] - 0.65) > 1e-9:
            changed.append("φ_b")
    combo_rows = [[name] + ([f"{lf[ev]:.2f}", f"{lf[eh]:.2f}",
                             f"{lf[ll]:.2f}"] if ev else
                            ["1.00", "1.00", "1.00"])
                  for name, ev, eh, ll in _LRFD_COMBINATIONS]
    sections.append(CalcSection(title="Load and Resistance Factors", items=[
        TableData(title="Factors used", headers=["Factor", "Value",
                                                 "Printed source"],
                  rows=rows,
                  notes=(("Given, not the printed value: " + ", ".join(
                      changed) + ".") if changed else
                      "Each as printed.")),
        TableData(title="Load combinations checked",
                  headers=["Combination", "EV", "EH", "LL / LS"],
                  rows=combo_rows,
                  notes=("GEC-11 Vol II Table E4-5.1 (p. E4-6): Strength I "
                         "(maximum), Strength I (minimum), Service I; the "
                         "critical one pairs EV min with EH max (Vol I p. "
                         "4-23: 'use the minimum EV and maximum EH load "
                         "factors'; Vol II Table E4-6.3, p. E4-9). The live "
                         "load over the reinforced zone is left out of "
                         "sliding and eccentricity and counted in bearing "
                         "(Vol I pp. 4-19, 4-22, 4-26; Fig. 4-2).")),
    ]))

    # ── Unfactored loads ──────────────────────────────────────
    ld_items = []
    bs = result.get("backslope") or {}
    # GM-307 review: a broken backslope's thrust is inclined at I (p. 4-15,
    # Fig. 4-4), as the ASD steps print it; I = beta for a slope that
    # continues
    broken = sloping and geom.broken_backslope_height is not None
    a = "I" if broken else "β"
    traffic = bool(uf.get("F2"))
    f2_resists = traffic and bool(bs.get("traffic_vertical_resists"))
    if sloping:
        ld_items.append(CalcStep(
            title="Retained Fill Thrust (sloping backfill)",
            equation=(f"F_T = 0.5 K_ab γ_b h²;  F_H = F_T cos {a}, F_V = "
                      f"F_T sin {a}"),
            substitution=(f"K_ab = {uf['Ka_retained']:.4f} (Eqs. 4-2, 4-3), "
                          f"h = {_f(bs.get('h_m'))} m"
                          + (f", I = {_f(bs.get('interface_angle_deg'), '.2f')}°"
                             if broken else "")
                          + f": F_T = {uf['FT']:.1f}, "
                          f"F_H = {uf['FT_H']:.1f}, F_V = {uf['FT_V']:.1f}"),
            result_name="F_T",
            result_value=f"{uf['FT']:.1f}",
            result_unit="kN/m",
            reference=("GEC-11 Vol I Eqs. 4-2, 4-3 (p. 4-15), Eq. 4-7 (p. "
                       "4-20), Eqs. 4-10, 4-11 (p. 4-21)"
                       + ("; β = δ = I for a broken backslope, p. 4-15 and "
                          "Fig. 4-4 (p. 4-14)" if broken else "")),
        ))
        rows = [["V1 (reinforced fill)", "EV", f"{uf['V1']:.1f}"],
                ["V2 (slope fill over the zone)", "EV", f"{uf['V2']:.1f}"],
                [f"F_T cos {a} (driving)", "EH", f"{uf['FT_H']:.1f}"],
                [f"F_T sin {a} (resisting)", "EH", f"{uf['FT_V']:.1f}"]]
        if uf.get("F2"):
            rows += [["F2 cos I (traffic behind the crest)", "LL",
                      f"{uf['F2_H']:.1f}"],
                     ["F2 sin I", "LL", f"{uf['F2_V']:.1f}"],
                     ["Vs (live load over the zone)", "LL",
                      f"{uf['Vs']:.1f}"]]
    else:
        ld_items.append(CalcStep(
            title="Retained Fill and Surcharge Thrusts (level backfill)",
            equation="K_ab = tan²(45° - φ_b/2);  F1 = 0.5 K_ab γ_b H²;  "
                     "F2 = K_ab q H",
            substitution=(f"K_ab = {uf['Ka_retained']:.4f}: F1 = "
                          f"{uf['F1']:.1f}, F2 = {uf['F2']:.1f}"),
            result_name="F1",
            result_value=f"{uf['F1']:.1f}",
            result_unit="kN/m",
            reference=("GEC-11 Vol I Eq. 4-1 (p. 4-15), Eqs. 4-5, 4-6 (p. "
                       "4-20)"),
        ))
        rows = [["V1 (reinforced fill)", "EV", f"{uf['V1']:.1f}"],
                ["F1 (retained fill)", "EH", f"{uf['F1']:.1f}"]]
        if uf.get("F2"):
            rows += [["F2 (live-load surcharge)", "LS", f"{uf['F2']:.1f}"],
                     ["Vs = q L (live load over the zone)", "LL",
                      f"{uf['Vs']:.1f}"]]
    ld_items.append(TableData(
        title="Unfactored Loads", headers=["Load", "Type", "kN/m"],
        rows=rows,
        notes="Load types as GEC-11 Vol I Sec. 4.2 (p. 4-3) names them."))
    sections.append(CalcSection(title="Unfactored Loads", items=ld_items))

    # ── Sliding ───────────────────────────────────────────────
    # GM-307 review: the terms the tool sums (_lrfd_sliding_loads), the
    # traffic thrust behind a broken backslope's crest included
    if sloping:
        r_tau = (f"R_τ = [γ_EV (V1 + V2) + γ_EH F_T sin {a}"
                 + (" + γ_LL F2 sin I" if f2_resists else "") + "] μ")
        p_d = (f"P_d = γ_EH F_T cos {a}"
               + (" + γ_LL F2 cos I" if traffic else ""))
    else:
        r_tau, p_d = "R_τ = γ_EV V μ", "P_d = γ_EH F1 + γ_LS F2"
    s_items = [CalcStep(
        title="Sliding CDR",
        equation=f"CDR = R_R / P_d;  R_R = φ_τ R_τ;  {r_tau};  {p_d}",
        substitution=(f"Strength I (max) {_f(sl.get('CDR_strength_max'))}, "
                      f"(min) {_f(sl.get('CDR_strength_min'))}, critical "
                      f"{_f(sl.get('CDR_critical'))}; μ by the "
                      f"{sl.get('sliding_cohesion', 'drained')} rule"),
        result_name="CDR_sliding",
        result_value=_f(sl.get("CDR_governing")),
        reference=("GEC-11 Vol I Eq. 4-4 (p. 4-19), Eqs. 4-9, 4-11 to "
                   "4-13 (p. 4-21), μ p. 4-22"
                   + ("; the traffic thrust F2 behind the crest, Vol II "
                      "Example E1, pp. E1-7 to E1-9" if sloping and traffic
                      else "") + "; required 1.0"),
    ), CheckItem(description="Sliding (governing combination)",
                 demand=1.0, demand_label="Required",
                 capacity=sl.get("CDR_governing"),
                 capacity_label="CDR", passes=bool(sl.get("passes")))]
    sections.append(CalcSection(title="Sliding", items=s_items))

    # ── Eccentricity ──────────────────────────────────────────
    e_rows = [["Strength I (max)", _f(ec.get("eL_strength_max_m")),
               _f(ec.get("e_limit_m"))],
              ["Strength I (min)", _f(ec.get("eL_strength_min_m")),
               _f(ec.get("e_limit_m"))],
              ["Critical (EV min, EH max)", _f(ec.get("eL_critical_m")),
               _f(ec.get("e_limit_m"))],
              ["Service I", _f(ec.get("eL_service_m")),
               _f(ec.get("e_limit_service_m"))]]
    e_items = [CalcStep(
        title="Eccentricity",
        equation="e = (ΣM_D - ΣM_R) / ΣV about the centre of L, ≤ e_max",
        substitution=(f"governing e = {_f(ec.get('eL_governing_m'))} m ≤ "
                      f"e_max = {_f(ec.get('e_limit_m'))} m"),
        result_name="e",
        result_value=_f(ec.get("eL_governing_m")),
        result_unit="m",
        reference=("GEC-11 Vol I Eq. 4-14 (p. 4-22), "
                   + ("Eq. 4-16" if sloping else "Eq. 4-15")
                   + " (p. 4-23), e_max p. 4-23"),
    ), TableData(title="Eccentricity by Combination",
                 headers=["Combination", "e (m)", "e_max (m)"], rows=e_rows),
        CheckItem(description="Eccentricity, Strength I", demand=ec.get(
            "eL_governing_m"), demand_label="e (m)",
            capacity=ec.get("e_limit_m"), capacity_label="e_max (m)",
            passes=bool(ec.get("passes_strength"))),
        CheckItem(description="Eccentricity, Service I", demand=ec.get(
            "eL_service_m"), demand_label="e (m)",
            capacity=ec.get("e_limit_service_m"),
            capacity_label="e_max (m)",
            passes=bool(ec.get("passes_service")))]
    sections.append(CalcSection(title="Eccentricity", items=e_items))

    # ── Bearing ───────────────────────────────────────────────
    worked = br.get("Nc") is not None
    b_rows = [
        ["Strength I (max)", _f(br.get("sigma_v_strength_max_kPa"), ".1f"),
         _f(br.get("B_eff_strength_max_m")),
         _f(br.get("qR_strength_max_kPa"), ".1f"),
         _f(br.get("CDR_strength"))],
        ["Strength I (min)", _f(br.get("sigma_v_strength_min_kPa"), ".1f"),
         _f(br.get("B_eff_strength_min_m")),
         _f(br.get("qR_strength_min_kPa"), ".1f"),
         _f(br.get("CDR_strength_min"))],
        ["Critical (EV min, EH max)",
         _f(br.get("sigma_v_critical_kPa"), ".1f"),
         _f(br.get("B_eff_critical_m")),
         _f(br.get("qR_critical_kPa"), ".1f"),
         _f(br.get("CDR_critical"))],
        ["Service I", _f(br.get("sigma_v_service_kPa"), ".1f"), "—",
         _f(_get(analysis, "bearing_resistance_service"), ".1f"),
         _f(br.get("CDR_service"))]]
    if not worked:
        given = _get(analysis, "bearing_resistance_strength")
        for row in b_rows[:3]:
            row[3] = _f(given, ".1f")
    b_items = [CalcStep(
        title="Bearing Pressure (Meyerhof)",
        equation="σ_v = ΣV / (L - 2e_B), L - 2e_B = L when e_B < 0",
        substitution=(f"Strength I (max) {_f(br.get('sigma_v_strength_max_kPa'), '.1f')}, "
                      f"(min) {_f(br.get('sigma_v_strength_min_kPa'), '.1f')}, "
                      f"critical {_f(br.get('sigma_v_critical_kPa'), '.1f')}, "
                      f"Service I {_f(br.get('sigma_v_service_kPa'), '.1f')} kPa"),
        result_name="σ_v (Strength I max)",
        result_value=_f(br.get("sigma_v_strength_max_kPa"), ".1f"),
        result_unit="kPa",
        reference=("GEC-11 Vol I Eq. 4-18 (p. 4-26), "
                   + ("Eq. 4-21 (p. 4-28)" if sloping else
                      "Eqs. 4-19, 4-20 (p. 4-27)")
                   + "; the live load over the zone at γ_LL (Vol II "
                   "Table E4-6.3, p. E4-9)"),
    )]
    if worked:
        b_items.append(CalcStep(
            title="Nominal and Factored Bearing Resistance",
            equation="q_n = c_f N_c + 0.5 L' γ_f N_γ;  q_R = φ_b q_n",
            substitution=(f"N_c = {br['Nc']:g}, N_γ = {br['Ngamma']:g} "
                          f"(Table 4-6), φ_b = {br['phi_bearing']:g}; q_n "
                          f"over each combination's L'"),
            result_name="q_R (Strength I max)",
            result_value=_f(br.get("qR_strength_max_kPa"), ".1f"),
            result_unit="kPa",
            reference=("GEC-11 Vol I Eq. 4-22 (p. 4-28), Eq. 4-23 (p. "
                       "4-30), Table 4-6 (p. 4-29)"),
        ))
    b_items.append(TableData(
        title="Bearing by Combination",
        headers=["Combination", "σ_v (kPa)", "L' (m)", "q_R (kPa)", "CDR"],
        rows=b_rows,
        notes=("CDR = q_R / σ_v (Eq. 4-17, p. 4-26); '—': not checked, no "
               "resistance given or worked out (GM-100).")))
    if br.get("checked"):
        b_items.append(CheckItem(
            description="Bearing (governing combination)", demand=1.0,
            demand_label="Required", capacity=br.get("CDR_governing"),
            capacity_label="CDR", passes=bool(br.get("passes"))))
    else:
        b_items.append("Bearing capacity: NOT CHECKED — no bearing "
                       "resistance given (bearing_resistance_strength) or "
                       "worked out (gamma_foundation). σ_v above is "
                       "reported.")
    ls = result.get("local_shear")
    if ls is not None:
        b_items.append(CheckItem(
            description="Local shear (GEC-11 Vol I Eq. 4-24, p. 4-30)",
            demand=ls["gamma_r_H_kPa"], demand_label="γ_r H (kPa)",
            capacity=ls["three_cu_kPa"], capacity_label="3 c_u (kPa)",
            passes=ls["passes"]))
    sections.append(CalcSection(title="Bearing", items=b_items))

    # ── Summary ───────────────────────────────────────────────
    summary = [TableData(
        title="LRFD External Stability Summary",
        headers=["Check", "Governing", "Required", "Status"],
        rows=[["Sliding CDR", _f(sl.get("CDR_governing")), "1.0",
               "OK" if sl.get("passes") else "FAIL"],
              ["Eccentricity e (m), Strength I",
               _f(ec.get("eL_governing_m")), f"≤ {_f(ec.get('e_limit_m'))}",
               "OK" if ec.get("passes_strength") else "FAIL"],
              ["Eccentricity e (m), Service I", _f(ec.get("eL_service_m")),
               f"≤ {_f(ec.get('e_limit_service_m'))}",
               "OK" if ec.get("passes_service") else "FAIL"],
              ["Bearing CDR", _f(br.get("CDR_governing")), "1.0",
               ("OK" if br.get("passes") else "FAIL") if br.get("checked")
               else "NOT CHECKED"]])]
    failed = result.get("checks_failed") or []
    not_run = result.get("checks_not_run") or []
    summary.append("Result: " + (
        ("FAILS: " + ", ".join(failed)) if failed else "no check failed")
        + ("; not checked: " + ", ".join(not_run) if not_run else "") + ".")
    if result.get("warnings"):
        summary.append(TableData(
            title="Warnings (GEC-11 advice, not a check)",
            headers=["Warning"], rows=[[w] for w in result["warnings"]]))
    summary.append("Internal stability is not part of this package: the "
                   "ASD package (wall_type 'mse') reports it.")
    sections.append(CalcSection(title="Overall Assessment", items=summary))
    return sections


# ════════════════════════════════════════════════════════════════════
#  FIGURES
# ════════════════════════════════════════════════════════════════════

def get_figures(result, analysis) -> List[FigureData]:
    """Generate figures for the retaining wall calc package.

    Parameters
    ----------
    result : CantileverWallResult or MSEWallResult
        Computed results.
    analysis : dict
        Analysis parameters.

    Returns
    -------
    list of FigureData
    """
    wtype = _wall_type(result, analysis)
    figures = []
    if wtype == "mse_lrfd":
        # GM-307: the LRFD package reports external stability; its section
        # is the ASD package's (wall_type 'mse')
        return figures

    if wtype == "mse":
        # MSE wall cross-section
        try:
            fig = _plot_mse_section(result, analysis)
            b64 = figure_to_base64(fig, dpi=150)
            import matplotlib.pyplot as plt
            plt.close(fig)
            figures.append(FigureData(
                title="MSE Wall Cross-Section",
                image_base64=b64,
                caption=(
                    f"Figure 1: MSE wall cross-section. "
                    f"H = {result.wall_height:.2f} m, "
                    f"L = {result.reinforcement_length:.2f} m, "
                    f"{result.n_levels} reinforcement levels."
                ),
                width_percent=80,
            ))
        except ImportError:
            pass

        # Internal stability profile
        if result.internal_results:
            try:
                fig2 = _plot_mse_internal(result, analysis)
                b64_2 = figure_to_base64(fig2, dpi=150)
                import matplotlib.pyplot as plt
                plt.close(fig2)
                figures.append(FigureData(
                    title="Internal Stability Profile",
                    image_base64=b64_2,
                    caption=(
                        "Figure 2: T_max and pullout resistance vs. depth. "
                        "All levels must have FOS_pullout >= 1.5."
                    ),
                    width_percent=75,
                ))
            except ImportError:
                pass
    else:
        # Cantilever wall cross-section
        try:
            fig = _plot_cantilever_section(result, analysis)
            b64 = figure_to_base64(fig, dpi=150)
            import matplotlib.pyplot as plt
            plt.close(fig)
            figures.append(FigureData(
                title="Cantilever Wall Cross-Section",
                image_base64=b64,
                caption=(
                    f"Figure 1: Cantilever wall cross-section. "
                    f"H = {result.wall_height:.2f} m, "
                    f"B = {result.base_width:.2f} m."
                ),
                width_percent=80,
            ))
        except ImportError:
            pass

        # Bearing pressure diagram
        try:
            fig2 = _plot_bearing_pressure(result, analysis)
            b64_2 = figure_to_base64(fig2, dpi=150)
            import matplotlib.pyplot as plt
            plt.close(fig2)
            figures.append(FigureData(
                title="Base Bearing Pressure Distribution",
                image_base64=b64_2,
                caption=(
                    f"Figure 2: Bearing pressure distribution along base. "
                    f"q_toe = {result.q_toe:.1f} kPa, "
                    f"q_heel = {result.q_heel:.1f} kPa."
                ),
                width_percent=70,
            ))
        except ImportError:
            pass

    return figures


# ═══════════════════════════════════════════════════════════════════
#  PRIVATE HELPERS
# ═══════════════════════════════════════════════════════════════════

def _cantilever_terms(result, analysis):
    """(earth pressure, vertical forces, sliding, overturning, bearing):
    the terms the cantilever analysis used (GM-143). The result's own;
    a result built by hand carries none, and then the module's own term
    functions work them out from the inputs the run used."""
    if getattr(result, "bearing_terms", None) is not None:
        return (result.earth_pressure, result.vertical_forces,
                result.sliding_terms, result.overturning_terms,
                result.bearing_terms)
    from retaining_walls import cantilever as cw
    geom = _get(analysis, "geom")
    gamma = _used(analysis, "gamma_backfill")
    phi = _used(analysis, "phi_backfill")
    c = _used(analysis, "c_backfill")
    method = _used(analysis, "pressure_method")
    gamma_c = _used(analysis, "gamma_concrete")
    wf = _used(analysis, "wall_friction_deg")
    gamma_f = _used(analysis, "gamma_foundation")
    rock = bool(_used(analysis, "founded_on_rock"))
    # GM-292: the water table's inputs
    g_sat = _used(analysis, "gamma_sat_backfill")
    uplift = _used(analysis, "base_uplift")
    up_brg = bool(_used(analysis, "uplift_in_bearing"))
    ep = cw._active_thrust_terms(geom, gamma, phi, c, method, wf, g_sat)
    live = bool(_used(analysis, "live_load_resists"))
    dead = bool(_used(analysis, "dead_load_resists"))
    in_bearing = bool(_used(analysis, "surcharge_in_bearing"))
    return (
        ep, cw._vertical_force_items(geom, gamma, gamma_c, ep, live, dead,
                                     in_bearing, gamma_front=gamma_f,
                                     base_uplift=uplift,
                                     uplift_in_bearing=up_brg),
        cw._sliding_terms(
            geom, gamma, phi, c, _used(analysis, "phi_foundation"),
            _used(analysis, "c_foundation"), gamma_c, method,
            bool(_used(analysis, "include_passive")), gamma_f,
            delta_base=_used(analysis, "delta_base"),
            base_adhesion=_used(analysis, "base_adhesion"),
            live_load_resists=live, dead_load_resists=dead,
            sliding_cohesion=_used(analysis, "sliding_cohesion"),
            passive_theory=_used(analysis, "passive_theory"),
            log_spiral_source=_used(analysis, "log_spiral_source"),
            passive_wall_friction_deg=_used(analysis,
                                            "passive_wall_friction_deg"),
            wall_friction_deg=wf,
            interface_gap_possible=bool(
                _used(analysis, "interface_gap_possible")),
            gamma_sat_backfill=g_sat, base_uplift=uplift),
        cw._overturning_terms(geom, gamma, phi, c, gamma_c, method,
                              live_load_resists=live,
                              dead_load_resists=dead,
                              wall_friction_deg=wf, gamma_foundation=gamma_f,
                              founded_on_rock=rock, gamma_sat_backfill=g_sat,
                              base_uplift=uplift),
        cw._bearing_terms(geom, gamma, phi, c,
                          _used(analysis, "q_allowable"), gamma_c, method,
                          surcharge_in_bearing=in_bearing,
                          wall_friction_deg=wf, gamma_foundation=gamma_f,
                          bearing_pressure=_used(analysis,
                                                 "bearing_pressure"),
                          founded_on_rock=rock, gamma_sat_backfill=g_sat,
                          base_uplift=uplift, uplift_in_bearing=up_brg))


def _loads_differ(ot, br) -> bool:
    """Whether the sliding / overturning / eccentricity loads and the
    bearing loads differ (GM-157: a surcharge over the heel counted in
    one and not the other)."""
    return (abs(ot["V"] - br["V"]) > 1e-9
            or abs(ot["M_stabilizing"] - br["M_stab"]) > 1e-9)


def _counted_in(force, resisting="sliding, overturning, eccentricity") -> str:
    """The checks a vertical force is counted in (GM-157); ``resisting``
    names the checks its ``resists`` flag stands for."""
    resists, bearing = force.get("resists", True), force.get("bearing", True)
    if force.get("kind") == "uplift":
        # GM-292: its moment is in M_ot
        return ("all checks" if bearing else resisting) + ", moment in M_ot"
    if resists and bearing:
        return "all checks"
    if bearing:
        return "bearing only"
    if resists:
        return resisting
    return "none"


def _mse_used(analysis, key):
    """An input the MSE run used: the analysis dict's, else the default of
    ``analyze_mse_wall``, which the run then took."""
    if isinstance(analysis, dict) and key in analysis:
        return analysis[key]
    if not isinstance(analysis, dict) and hasattr(analysis, key):
        return getattr(analysis, key)
    import inspect
    from retaining_walls.mse import analyze_mse_wall
    default = inspect.signature(
        analyze_mse_wall).parameters[key].default
    if default is inspect.Parameter.empty:
        raise ValueError(
            f"retaining_walls calc steps: the analysis gives no {key}, "
            f"which analyze_mse_wall requires; pass the inputs the run "
            f"was given")
    return default


def _mse_terms(result, analysis):
    """The ASD external terms the MSE analysis used (GM-143, GM-164): the
    result's own; a result built by hand carries none, and then the
    module's term function works them out from the inputs the run used."""
    if getattr(result, "external_terms", None) is not None:
        return result.external_terms
    from retaining_walls import mse
    phi = _mse_used(analysis, "phi_backfill")
    phi_f = _mse_used(analysis, "phi_foundation")
    geom = _get(analysis, "geom")
    return mse._external_terms(
        geom, _mse_used(analysis, "gamma_backfill"), phi,
        # GM-302 review: GEC-11 Vol I p. 4-10's 30 deg for no data
        mse.PHI_ASSUMED if phi_f is None else phi_f,
        _mse_used(analysis, "c_foundation"),
        phi_retained=_mse_used(analysis, "phi_retained"),
        gamma_retained=_mse_used(analysis, "gamma_retained"),
        traffic_vertical_resists=bool(
            _mse_used(analysis, "traffic_vertical_resists")),
        live_load_resists=bool(_mse_used(analysis, "live_load_resists")),
        surcharge_in_bearing=bool(
            _mse_used(analysis, "surcharge_in_bearing")),
        sliding_cohesion=_mse_used(analysis, "sliding_cohesion"),
        interface_gap_possible=bool(
            _mse_used(analysis, "interface_gap_possible")),
        lowest_sheet_depth=mse.lowest_sheet_depth(
            geom, _get(analysis, "reinforcement")),
        rho_tested=_mse_used(analysis, "rho_tested"))


def _plot_cantilever_section(result, analysis):
    """Create a schematic cross-section of a cantilever retaining wall."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    geom = _get(analysis, "geom")
    gamma = _used(analysis, "gamma_backfill")
    phi = _used(analysis, "phi_backfill")
    method = _used(analysis, "pressure_method")

    H = geom.wall_height
    B = geom.base_width
    t_toe = geom.toe_length
    t_stem_b = geom.stem_thickness_base
    t_stem_t = geom.stem_thickness_top
    h_stem = geom.stem_height
    t_base = geom.base_thickness
    heel = geom.heel_length

    fig, ax = plt.subplots(figsize=(8, 6))

    # Coordinate system: x from left, y from bottom of base (y=0)

    # Base slab
    base = patches.Rectangle(
        (0, 0), B, t_base,
        facecolor='#94a3b8', edgecolor='#1a1a1a', linewidth=1.5,
    )
    ax.add_patch(base)

    # Stem (trapezoid: wider at base, narrower at top), its battered face
    # as the analysis has it (GM-288: the back, or the front)
    batter = t_stem_b - t_stem_t
    front_top = t_toe + (batter if geom.stem_batter == "front" else 0.0)
    stem_x = [
        t_toe,                              # bottom-left (front face)
        front_top,                          # top-left (front face)
        front_top + t_stem_t,               # top-right
        t_toe + t_stem_b,                   # bottom-right
    ]
    stem_y = [
        t_base,                             # bottom-left
        t_base + h_stem,                    # top-left
        t_base + h_stem,                    # top-right
        t_base,                             # bottom-right
    ]
    ax.fill(stem_x, stem_y, facecolor='#94a3b8', edgecolor='#1a1a1a', linewidth=1.5)

    # The earth pressure as the analysis takes it (GM-143), and its height:
    # the vertical plane through the back of the heel, H_active high
    ep = _cantilever_terms(result, analysis)[0]
    Ka_val = ep["Ka"]
    H_a = ep["H_active"]
    q = (ep.get("q_live") or 0.0) + (ep.get("q_dead") or 0.0)
    p_q = Ka_val * q                       # the surcharge's, kPa
    p_soil = Ka_val * gamma * H_a          # the earth's, at the base
    # GM-292: under a water table the earth's on the effective stress, the
    # water's beside it
    h_w = ep.get("h_w") or 0.0
    p_w = 0.0
    if h_w > 0:
        z_w = ep["z_w"]
        p_wt = Ka_val * gamma * z_w        # the earth's at the water table
        p_soil = p_wt + Ka_val * ep["gamma_b"] * (H_a - z_w)
        p_w = ep["gamma_w"] * h_w          # the water's, at the base
    # the diagram's widest, a set share of B, and the room right of it
    p_scale = B * 0.5 / max(p_q + p_soil + p_w, 1e-9)
    w_diagram = (p_q + p_soil + p_w) * p_scale

    # The ground on each side, as the analysis has it (GM-200: the toe
    # side's ground was drawn at H, level with the backfill). Behind the
    # wall the backfill meets the stem at its top and rises at the
    # backfill slope from the back of the stem at the heel, where the
    # analysis's wedge over the heel starts, so it meets the plane through
    # the back of the heel at H_active (GM-216); in front, the ground is
    # at the top of the base: H is measured from the underside of the
    # base, the wall retains H less the base thickness above the top of
    # the footing, and passive resistance is taken on the base's
    # embedment (``cantilever._sliding_terms``), Df below the ground in
    # front (GM-288: the base thickness unless embedment_depth is given).
    x_left, x_right = -B * 0.4, B + w_diagram + B * 0.3
    slope = math.tan(math.radians(geom.backfill_slope or 0.0))
    x_back_top = front_top + t_stem_t
    y_front = geom.Df
    x_heel = t_toe + t_stem_b

    def _backfill_surface(x):
        return H + max(x - x_heel, 0.0) * slope

    backfill_x = [x_heel, B, B, x_right, x_right, x_heel, x_back_top]
    backfill_y = [t_base, t_base, 0.0, 0.0, _backfill_surface(x_right), H,
                  H]
    ax.fill(backfill_x, backfill_y, facecolor='#f5e6c8', edgecolor='none',
            alpha=0.7, zorder=0)

    # A shear key: where it is given, else under the stem, an assumption
    # (UFC 3-220-20 Fig. 4-29, p. 224, draws a gravity wall, GEC-6 Fig.
    # 5-22 a footing; neither prints a cantilever key's position), and the
    # figure says so (GM-216, GM-224); the analysis has no place for it
    key_depth = geom.key_depth if geom.has_shear_key else 0.0
    key_lines = []
    if key_depth > 0:
        key_x = _get(analysis, "key_distance_from_toe")
        key_w = _get(analysis, "key_width")
        if key_x is None or key_w is None:
            missing = ("position or width" if key_x is None
                       and key_w is None else
                       "position" if key_x is None else "width")
            key_lines += [("Shear key drawn under the stem: an assumption, no "
                           "printed basis (UFC 3-220-20 Fig. 4-29 draws a "
                           "gravity wall);"),
                          f"no key {missing} given."]
        key_x = t_toe if key_x is None else float(key_x)
        key_w = t_stem_b if key_w is None else float(key_w)
        ax.add_patch(patches.Rectangle(
            (key_x, -key_depth), key_w, key_depth, facecolor='#94a3b8',
            edgecolor='#1a1a1a', linewidth=1.5))
        if not _used(analysis, "include_passive"):
            key_lines.append("No passive resistance in the sliding check: "
                             "the key is not counted.")

    # Foundation soil under the base and the soil in front of the toe
    y_bottom = -B * 0.45 - key_depth
    ax.fill([x_left, x_right, x_right, 0.0, 0.0, x_left],
            [y_bottom, y_bottom, 0.0, 0.0, y_front, y_front],
            facecolor='#e8dcc8', edgecolor='none', alpha=0.5, zorder=0)
    if y_front > t_base and t_toe > 0:
        # the soil over the toe, behind the stem where it meets it
        ax.add_patch(patches.Rectangle(
            (0.0, t_base), t_toe + t_stem_b, y_front - t_base,
            facecolor='#e8dcc8', edgecolor='none', alpha=0.5, zorder=0))

    # The pressure diagram on the plane through the back of the heel, from
    # the underside of the base to H_active, where the analysis applies it
    # (it was drawn on the stem's face, over the stem only; GM-216): the
    # surcharge's K_a q next to the plane, the earth's K_a gamma z beyond
    press = dict(linewidth=1.2, alpha=0.5, zorder=1)
    if p_q > 0:
        ax.add_patch(patches.Rectangle(
            (B, 0.0), p_q * p_scale, H_a, facecolor='#fdba74',
            edgecolor='#c2410c', hatch='..', **press))
    x_q = B + p_q * p_scale
    if h_w > 0:
        # the earth's kinks at the water table; the water's on its edge
        x_wt, x_b = x_q + p_wt * p_scale, x_q + p_soil * p_scale
        ax.fill([x_q, x_wt, x_b, x_q], [H_a, h_w, 0.0, 0.0],
                facecolor='#fca5a5', edgecolor='#dc2626', **press)
        ax.fill([x_wt, x_b, x_b + p_w * p_scale], [h_w, 0.0, 0.0],
                facecolor='#93c5fd', edgecolor='#1d4ed8', **press)
        # the water table behind the wall, over the heel
        from matplotlib.transforms import offset_copy as _offset
        ax.plot([x_heel, B], [h_w, h_w], color='#1d4ed8', lw=1.0, ls='--',
                zorder=2)
        ax.plot(x_heel + 0.5 * (B - x_heel), h_w, marker='v',
                color='#1d4ed8', markersize=7, zorder=3,
                transform=_offset(ax.transData, fig=fig, y=5,
                                  units='points'))
        ax.annotate('', xy=(B, ep["z_Pw"]),
                    xytext=(B + w_diagram + 0.12 * B, ep["z_Pw"]),
                    arrowprops=dict(arrowstyle='-|>', color='#1d4ed8',
                                    lw=1.6, mutation_scale=12), zorder=4)
    else:
        ax.fill([x_q, x_q + p_soil * p_scale, x_q], [0.0, 0.0, H_a],
                facecolor='#fca5a5', edgecolor='#dc2626', **press)
    # The thrust, at its height above the base and inclination as the
    # analysis has them, named in the key (written at its tail it ran
    # past the frame at the printed width)
    delta = math.radians(ep.get("delta_deg") or 0.0)
    z_pa = ep.get("z_Pa", H_a / 3.0)
    reach = w_diagram + 0.12 * B
    tail = (B + reach * math.cos(delta), z_pa + reach * math.sin(delta))
    ax.annotate('', xy=(B, z_pa), xytext=tail,
                arrowprops=dict(arrowstyle='-|>', color='#991b1b', lw=1.6,
                                mutation_scale=12), zorder=4)
    # The surcharge on the backfill it stands for
    if q > 0:
        for i in range(7):
            x = x_heel + 0.1 * B + i * (x_right - 0.18 * B - x_heel) / 6
            ax.annotate('', xy=(x, _backfill_surface(x)), xytext=(0, 12),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='#c2410c',
                                        lw=0.8))

    # Dimension annotations
    # Wall height
    ax.annotate('', xy=(-B * 0.15, 0), xytext=(-B * 0.15, H),
                arrowprops=dict(arrowstyle='<->', color='#1a1a1a', lw=1.2))
    ax.text(-B * 0.22, H / 2, f'H = {H:.2f} m', ha='right', va='center',
            fontsize=9, fontweight='bold', rotation=90)

    # The dimensions under the base, set off from it by points (by metres
    # they ran into each other and their values onto their lines at the
    # printed width; GM-194): toe and heel 4 pt under it, B under those;
    # under the shear key where there is one (GM-216).
    from matplotlib.transforms import offset_copy

    def _under(pts):
        return offset_copy(ax.transData, fig=fig, y=-pts, units='points')

    def _dimension(x0, x1, pts, text, **kw):
        line = dict(color=kw.pop('color', '#1a1a1a'), lw=kw.pop('lw', 1.2))
        ax.annotate('', xy=(x0, -key_depth), xytext=(x1, -key_depth),
                    xycoords=_under(pts), textcoords=_under(pts),
                    arrowprops=dict(arrowstyle='<->', **line))
        ax.text((x0 + x1) / 2, -key_depth, text, transform=_under(pts + 4),
                ha='center', va='top', color=line['color'], **kw)

    if t_toe > 0:
        _dimension(0, t_toe, 4, f'toe={t_toe:.2f}', fontsize=7,
                   color='#666', lw=0.8)
    if heel > 0:
        _dimension(t_toe + t_stem_b, B, 4, f'heel={heel:.2f}', fontsize=7,
                   color='#666', lw=0.8)
    _dimension(0, B, 20, f'B = {B:.2f} m', fontsize=9, fontweight='bold')

    # Ground surface lines: the backfill behind the stem, the ground in
    # front Df above the underside of the base, to the stem's front face
    ax.plot([x_back_top, x_heel, x_right],
            [H, H, _backfill_surface(x_right)], 'g-', linewidth=2)
    x_front = 0.0
    if y_front > t_base:
        x_front = t_toe + (front_top - t_toe) * min(
            (y_front - t_base) / h_stem, 1.0) if h_stem > 0 else t_toe
    ax.plot([x_left, x_front], [y_front, y_front], 'g-', linewidth=2)

    # Resultant location marker, its point on the base's underside, and
    # its label beside it inside the base (under the base it ran into B)
    x_R_scaled = (result.base_width / 2.0 - result.eccentricity)
    ax.plot(x_R_scaled, 0, 'rv', markersize=10,
            transform=offset_copy(ax.transData, fig=fig, y=4,
                                  units='points'))
    ax.annotate(f'R (e={result.eccentricity:.3f}m)', (x_R_scaled, 0),
                xytext=(7, 5), textcoords='offset points', ha='left',
                va='center', fontsize=7, color='#dc2626')

    # The soil and the factors of safety under the drawing: in boxes in
    # the axes they covered the stem and ran over the frame at the
    # printed width (GM-194). A check with no FOS says why (GM-100).
    def _fos_text(fos, word):
        return word if fos is None else f"{fos:.2f}"

    lines = [
        f"Active pressure on the vertical plane through the back of the "
        f"heel, {H_a:.2f} m high",
        f'\u03b3 = {gamma:.1f} kN/m\u00b3, \u03c6 = {phi:.1f}\u00b0, '
        f'Ka = {Ka_val:.3f}']
    if ep.get("Pa_cohesion"):
        # net of the tension zone, where no negative pressure is counted
        # (GM-288)
        cut = -(ep["Pa_cohesion"] + (ep.get("Pa_tension_zone") or 0.0))
        lines.append(f"The backfill's cohesion takes {cut:.0f} kN/m off "
                     f"P_a (not drawn).")
    lines.append(
        f"FOS_slide = {_fos_text(result.FOS_sliding, 'no driving force')}, "
        f"FOS_OT = {_fos_text(result.FOS_overturning, 'no driving force')}, "
        f"FOS_brg = {_fos_text(result.FOS_bearing, 'not checked')}")
    lines += key_lines
    from matplotlib.lines import Line2D
    handles = [Line2D(
        [], [], color='#991b1b', lw=1.6, marker='<', markersize=6,
        label=(f'P_a = {ep["Pa"]:.0f} kN/m, {z_pa:.2f} m above the base'
               + (f', at {ep["delta_deg"]:.0f}° to horizontal'
                  if ep.get("delta_deg") else '')))]
    if h_w > 0:
        # GM-292
        handles += [patches.Patch(
            facecolor='#fca5a5', edgecolor='#dc2626', alpha=0.5,
            label=f"Earth, K_a\u03c3'_v (\u03b3_b below the water table): "
                  f"{p_soil:.1f} kPa at the base"),
            Line2D([], [], color='#1d4ed8', lw=1.6, marker='<', markersize=6,
                   label=f'P_w = {ep["Pw"]:.0f} kN/m, {ep["z_Pw"]:.2f} m '
                         f'above the base'),
            patches.Patch(
            facecolor='#93c5fd', edgecolor='#1d4ed8', alpha=0.5,
            label=f'Water, \u03b3_w \u00d7 depth below the water table: '
                  f'{p_w:.1f} kPa at the base, {h_w:.2f} m below it')]
        up = [f for f in _cantilever_terms(result, analysis)[1]
              if f.get("kind") == "uplift"]
        if up:
            lines.insert(2, f"Uplift on the base U = {-up[0]['W']:.0f} kN/m "
                            f"({up[0].get('base_uplift', 'linear')}; not "
                            "drawn).")
    else:
        handles += [patches.Patch(
            facecolor='#fca5a5', edgecolor='#dc2626', alpha=0.5,
            label=f'Earth, K_a\u03b3z: {p_soil:.1f} kPa at the base')]
    if p_q > 0:
        handles.append(patches.Patch(
            facecolor='#fdba74', edgecolor='#c2410c', hatch='..', alpha=0.5,
            label=f'Surcharge, K_a q: {p_q:.1f} kPa (q = {q:g} kPa)'))

    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, max(H * 1.15, H_a + 0.05 * H,
                              _backfill_surface(x_right)
                              + (0.12 if q > 0 else 0.05) * H))
    # the wall's own frame, said as such: the inputs give no elevation
    # (the axis read "Elevation" with 0 at the underside of the base)
    ax.set_xlabel('Distance from the toe (m)', fontsize=10)
    ax.set_ylabel('Height above the underside of the base (m)', fontsize=10)
    ax.set_title('Cantilever Retaining Wall Cross-Section', fontsize=11, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)
    from geotech_common.plotting import key_under_axes
    key_under_axes(fig, ax, handles=handles, note="\n".join(lines), ncol=1)
    return fig


def _plot_bearing_pressure(result, analysis):
    """Create a bearing pressure distribution diagram."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    geom = _get(analysis, "geom")
    B = geom.base_width
    q_toe = result.q_toe
    q_heel = result.q_heel

    fig, ax = plt.subplots(figsize=(7, 3.5))

    # x along base from toe (0) to heel (B); outside the middle third the
    # pressure is a triangle over the width the analysis used (GM-143)
    terms = getattr(result, "bearing_terms", None) or {}
    B_eff = terms.get("B_eff")
    if terms.get("distribution") == "triangular_toe" and B_eff:
        x, q = [0, min(B_eff, B)], [q_toe, 0.0]
    elif terms.get("distribution") == "triangular_heel" and B_eff:
        x, q = [max(B - B_eff, 0.0), B], [0.0, q_heel]
    else:
        x = [0, B]
        q = [q_toe, q_heel]

    # Fill the pressure distribution
    ax.fill_between(x, 0, q, color='#fca5a5', alpha=0.5, edgecolor='#dc2626', linewidth=1.5)
    ax.plot(x, q, 'r-', linewidth=2)

    # Base line
    ax.axhline(y=0, color='k', linewidth=1)

    # Middle-third zone
    b6 = B / 6.0
    ax.axvspan(B / 2 - b6, B / 2 + b6, alpha=0.1, color='#16a34a',
               label=f'Middle third (B/6 = {b6:.2f} m)')

    # Eccentricity marker: the resultant of the loads drawn, the bearing
    # check's (GM-157: it can differ from the eccentricity check's)
    e = terms.get("eccentricity", result.eccentricity)
    x_R = B / 2.0 - e
    differs = getattr(result, "eccentricity_bearing", None) is not None
    ax.axvline(x=x_R, color='#dc2626', linestyle='--', linewidth=1.2,
               label=(f'Resultant for bearing (e_b = {e:.3f} m)' if differs
                      else f'Resultant (e = {e:.3f} m)'))
    ax.axvline(x=B / 2.0, color='#666', linestyle=':', linewidth=0.8,
               label='Base center')

    q_all = _used(analysis, "q_allowable")
    if q_all is not None:
        ax.axhline(y=q_all, color='#16a34a', linestyle='--', linewidth=1.2,
                   label=f'q_allowable = {q_all:.0f} kPa')

    ax.set_xlabel('Distance from Toe (m)', fontsize=10)
    ax.set_ylabel('Bearing Pressure (kPa)', fontsize=10)
    ax.set_title('Base Bearing Pressure Distribution', fontsize=11, fontweight='bold')
    ax.set_xlim(-B * 0.05, B * 1.05)
    ax.set_ylim(0, max(q_toe, q_heel, q_all or 0) * 1.3 + 5)
    ax.grid(True, alpha=0.2)
    # the key under the axes, headed by the edge pressures: inside them it
    # covered the q_toe note once the figure was drawn for the width it
    # prints at (GM-186), and the notes over the diagram lay on the
    # resultant's line and the pressure line (GM-194)
    from geotech_common.plotting import key_under_axes
    key_under_axes(fig, ax, fontsize=7,
                   note=(f'q_toe = {q_toe:.1f} kPa, '
                         f'q_heel = {q_heel:.1f} kPa'))
    return fig


def _plot_mse_section(result, analysis):
    """Create a schematic cross-section of an MSE wall."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    geom = _get(analysis, "geom")
    reinforcement = _get(analysis, "reinforcement")
    gamma = _get(analysis, "gamma_backfill", 18.0)
    phi = _get(analysis, "phi_backfill", 34.0)

    H = geom.wall_height
    L = geom.reinforcement_length
    Sv = geom.reinforcement_spacing

    fig, ax = plt.subplots(figsize=(9, 6))

    # GM-307: a face battered 10 deg or more leans back at omega (GEC-11
    # Vol I Fig. 4-5); the zone is a parallelogram of horizontal width L
    tw = (math.tan(math.radians(geom.theta_deg - 90.0))
          if getattr(geom, "battered", False) else 0.0)
    # Reinforced soil block
    ax.add_patch(patches.Polygon(
        [(0, 0), (L, 0), (L + H * tw, H), (H * tw, H)],
        facecolor='#f5e6c8', edgecolor='#1a1a1a', linewidth=2,
    ))

    # Wall face (thicker line on the left)
    ax.plot([0, H * tw], [0, H], 'k-', linewidth=3)

    # Reinforcement layers
    rs, per_r = _level_reinforcements(reinforcement, geom)
    first = rs[0] if rs else None
    r_color = '#2563eb' if (first and first.is_metallic) else '#16a34a'
    # GM-266: one line style a reinforcement when given per level
    styles = ['-', '--', '-.', ':']
    keys = _reinforcement_keys(rs) if per_r else {}
    for i, z in enumerate(geom.reinforcement_depths):
        y = H - z  # convert depth to elevation
        style = (styles[(ord(keys[_reinforcement_label(rs[i])])
                         - ord("A")) % len(styles)] if per_r else '-')
        ax.plot([y * tw, y * tw + L], [y, y], color=r_color, linewidth=1.5,
                alpha=0.7, linestyle=style)
        ax.plot(y * tw + L, y, '|', color=r_color, markersize=8,
                markeredgewidth=2)

    # Failure surface: bilinear coherent-gravity surface for inextensible
    # (metallic) reinforcement, Rankine 45+phi/2 plane for extensible (RW-2)
    if geom.backfill_slope > 0:
        # GM-307: the backslope, rising at beta from the top of the face
        # (GEC-11 Vol I Figs. 4-3, 4-4), level behind a broken one's crest
        from retaining_walls import mse as _mse
        tb = math.tan(math.radians(geom.backfill_slope))
        S_b = geom.broken_backslope_height
        xs = [0.0, 1.4 * L] if S_b is None else sorted(
            {0.0, min(S_b / tb, 1.4 * L), 1.4 * L})
        ax.plot([H * tw + x for x in xs],
                [H + (x * tb if S_b is None else min(x * tb, S_b))
                 for x in xs], 'k-', linewidth=1.5)
    if first is not None and first.is_metallic:
        H1 = (_mse.backslope_geometry(geom)["H1"]
              if geom.backfill_slope > 0 else H)
        x_fail_top = 0.3 * H1
        ax.plot([0, 0.3 * H1 + H1 / 2.0 * tw, 0.3 * H1 + H * tw],
                [0, H1 / 2.0, H], 'r--',
                linewidth=2, alpha=0.7,
                label=('Coherent-gravity surface (0.3H1, inextensible)'
                       if H1 != H else
                       'Coherent-gravity surface (0.3H, inextensible)'))
    elif tw > 0:
        # GM-307: GEC-11 Fig. 4-9(b)'s Coulomb plane through the toe
        from retaining_walls import mse as _mse
        psi = _mse.extensible_failure_angle(phi, geom.theta_deg,
                                            geom.backfill_slope)
        x_fail_top = H / math.tan(math.radians(psi))
        ax.plot([0, x_fail_top], [0, H], 'r--', linewidth=2, alpha=0.7,
                label=f'Failure plane (Coulomb, \u03c8 = {psi:.0f}\u00b0)')
        x_fail_top -= H * tw          # from the face, as below
    else:
        failure_angle = math.radians(45 + phi / 2)
        # Line from toe going up at the failure angle
        x_fail_top = H / math.tan(failure_angle)
        ax.plot([0, x_fail_top], [0, H], 'r--', linewidth=2, alpha=0.7,
                label=f'Failure plane (45+\u03c6/2 = {45 + phi / 2:.0f}\u00b0)')

    # Retained soil behind reinforced zone
    ax.add_patch(patches.Polygon(
        [(L, 0), (L * 1.4 + H * tw, 0), (L * 1.4 + H * tw, H),
         (L + H * tw, H)],
        facecolor='#e8dcc8', edgecolor='none', alpha=0.5,
    ))

    # Earth pressure arrows on back face of reinforced zone
    Ka = math.tan(math.radians(45 - phi / 2)) ** 2
    n_arrows = 5
    for i in range(1, n_arrows + 1):
        y_arr = H * (1 - i / n_arrows)
        z_arr = H - y_arr  # depth
        sigma = Ka * gamma * z_arr
        arrow_len = sigma / (Ka * gamma * H) * L * 0.3
        ax.annotate('', xy=(L + y_arr * tw, y_arr),
                    xytext=(L + y_arr * tw + arrow_len, y_arr),
                    arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.5))

    ax.text(L + L * 0.15, H * 0.1, 'P_a', fontsize=10, color='#dc2626', fontweight='bold')

    # Dimension annotations
    # Height
    ax.annotate('', xy=(-L * 0.08, 0), xytext=(-L * 0.08, H),
                arrowprops=dict(arrowstyle='<->', color='#1a1a1a', lw=1.2))
    ax.text(-L * 0.15, H / 2, f'H = {H:.2f} m', ha='right', va='center',
            fontsize=9, fontweight='bold', rotation=90)

    # Reinforcement length
    ax.annotate('', xy=(0, -H * 0.06), xytext=(L, -H * 0.06),
                arrowprops=dict(arrowstyle='<->', color='#1a1a1a', lw=1.2))
    ax.annotate(f'L = {L:.2f} m', (L / 2, -H * 0.06), xytext=(0, -2),
                textcoords='offset points', ha='center', va='top',
                fontsize=9, fontweight='bold')

    # Spacing annotation
    if len(geom.reinforcement_depths) >= 2:
        z1 = geom.reinforcement_depths[0]
        z2 = geom.reinforcement_depths[1]
        y1 = H - z1
        y2 = H - z2
        ax.annotate('', xy=(L * 1.05, y1), xytext=(L * 1.05, y2),
                    arrowprops=dict(arrowstyle='<->', color='#666', lw=0.8))
        if geom.level_depths is not None:
            Sv = z2 - z1   # GM-273: the depths given set the spacing
        ax.text(L * 1.08, (y1 + y2) / 2, f'Sv = {Sv:.2f} m',
                fontsize=7, va='center', color='#666')

    # Labels: the zone's name between the failure surface and the back of
    # the zone, between two reinforcement levels, and ringed so a level
    # stops short of it (centred in the zone the surface ran through it;
    # GM-194). Measured from the surface where it crosses the upper of the
    # two levels: from the plane's top, it sat across the zone's back.
    from geotech_common.plotting import halo
    levels = sorted(H - z for z in geom.reinforcement_depths)
    bands = list(zip(levels, levels[1:])) or [(0.4 * H, 0.6 * H)]
    lo, hi = min(bands, key=lambda b: abs((b[0] + b[1]) / 2 - H / 2))
    if first is not None and first.is_metallic:
        x_surface = x_fail_top * min(1.0, hi / (H / 2.0))
    else:
        x_surface = x_fail_top * hi / H
    halo([ax.text((min(x_surface, L) + L) / 2, (lo + hi) / 2,
                  'Reinforced\nSoil Zone', ha='center', va='center',
                  fontsize=10, fontweight='bold', color='#6b6358')],
         colour='#f5e6c8')
    ax.text(L * 1.2 + H * tw * 0.75, H / 2, 'Retained\nSoil', ha='center', va='center',
            fontsize=9, alpha=0.5)

    # Reinforcement legend
    if per_r:
        for name, k in keys.items():
            ax.plot([], [], color=r_color, linewidth=2,
                    linestyle=styles[(ord(k) - ord("A")) % len(styles)],
                    label=name)
    else:
        r_name = reinforcement.name if reinforcement else "Reinforcement"
        ax.plot([], [], color=r_color, linewidth=2, label=r_name)

    # The factors of safety head the key, under the drawing: the box in
    # the axes covered the zone's top corner and the key covered L at the
    # printed width (GM-194). Red for a check that fails, green when every
    # check passes, dark grey when one was not made (bearing with no
    # allowable pressure): in red, "all pass" read as a failure.
    verdicts = (result.passes_external, result.all_pass_internal)
    if any(v is False for v in verdicts):
        status_color = '#dc2626'
    elif all(v is True for v in verdicts):
        status_color = '#16a34a'
    else:
        status_color = '#333333'
    fos_brg = getattr(result, 'FOS_bearing', None)
    fos_text = (
        f"External: slide={result.FOS_sliding:.2f}, OT={result.FOS_overturning:.2f}, "
        f"bearing={'not checked' if fos_brg is None else f'{fos_brg:.2f}'}; "
        f"Internal: {'all pass' if result.all_pass_internal else 'FAIL'}"
    )

    ax.set_xlim(-L * 0.25, L * 1.5)
    ax.set_ylim(-H * 0.15, H * 1.1)
    # the wall's own frame, said as such: the inputs give no elevation
    # (the axis read "Elevation" with 0 at the base of the reinforced
    # zone; GM-218, as GM-200 did for the cantilever section)
    ax.set_xlabel('Distance from the wall face (m)', fontsize=10)
    ax.set_ylabel('Height above the base of the reinforced zone (m)',
                  fontsize=10)
    ax.set_title('MSE Wall Cross-Section', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)
    from geotech_common.plotting import key_under_axes
    key = key_under_axes(fig, ax, note=fos_text)
    key.get_title().set_color(status_color)
    return fig


def _plot_mse_internal(result, analysis):
    """Create a Tmax vs pullout resistance depth profile."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = result.internal_results
    if not data:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.text(0.5, 0.5, 'No internal stability data', transform=ax.transAxes,
                ha='center', fontsize=12)
        return fig

    depths = [r['depth_m'] for r in data]
    Tmax = [r['Tmax_kN_per_m'] for r in data]
    Pr = [r['Pr_kN_per_m'] for r in data]
    passes = [r.get('passes', False) for r in data]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot Tmax and Pr vs depth (depth increasing downward)
    ax.plot(Tmax, depths, 'ro-', linewidth=2, markersize=6, label='T_max (demand)')
    ax.plot(Pr, depths, 'bs-', linewidth=2, markersize=6, label='P_r (capacity)')

    # Mark failures
    for i, p in enumerate(passes):
        if not p:
            ax.plot(Tmax[i], depths[i], 'rx', markersize=12, markeredgewidth=3)

    ax.invert_yaxis()  # depth increases downward
    ax.set_xlabel('Force (kN/m)', fontsize=10)
    ax.set_ylabel('Depth below the top of the wall, z (m)', fontsize=10)
    ax.set_title('Internal Stability: T_max vs Pullout Resistance', fontsize=11,
                 fontweight='bold')
    ax.grid(True, alpha=0.3)
    # the key under the axes: inside them the deepest P_r ran through it
    # (GM-194)
    from geotech_common.plotting import key_under_axes
    key_under_axes(fig, ax, fontsize=9)
    return fig
