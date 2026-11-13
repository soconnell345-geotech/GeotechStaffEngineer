"""
Structural design of a cantilever retaining wall's concrete (GM-312):
stem, toe and heel, each with its factored moment and shear at its
critical section, the steel required, a bar size and spacing, the checks
the manual prints and the cover.

On USACE EM 1110-2-2104, "Strength Design for Reinforced Concrete
Hydraulic Structures" (1 November 2023; it supersedes the 2016 edition and
has no hydraulic factor). ACI 318 is not held: where the manual sends the
reader to ACI 318-19 and prints nothing itself, that part is NOT designed
and the result's ``not_designed`` says so.

The design runs :func:`retaining_walls.cantilever.analyze_cantilever_wall`
on the same inputs first and keeps its result (``stability``), so the
stability checks and the concrete are one wall. Inputs and outputs are SI
(m, kN, kPa, MPa, mm); the manual is US customary, the section design is
worked in its units (:mod:`retaining_walls.structural_section`) and every
structural quantity is returned in both.

What the print gives, and the calls made where it gives nothing, are in
the result's ``choices``, each with its page.

Load cases (Table E-2, "Load combinations for an earth retaining wall",
p. 93):

- Case 1, normal operating (usual): serviceability 2.2 (D + EH + EV + Hs);
  strength (1.2 or 0.9) D + 1.5 EH + (1.35 or 1.0) EV + 1.0 Hs.
- Case 2, normal operating + surcharge (unusual), when there is a
  live-load surcharge: serviceability 1.6 (D + EH + EV + ES + Hs);
  strength the same as Case 1 + 1.6 ES.

Not designed: the earthquake cases (3A, 3B), the maximum hydrostatic case
(4) and the construction case (5).
"""

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from geotech_common.units import (
    MPa_to_psi, in2_per_ft_to_mm2_per_m, kN_per_m_to_kip_per_ft,
    kNm_per_m_to_kipft_per_ft, kip_per_ft_to_kN_per_m,
    kipft_per_ft_to_kNm_per_m,
)
from geotech_common.water import GAMMA_W
from retaining_walls import structural_section as S
from retaining_walls.cantilever import (
    GAMMA_CONCRETE, _active_thrust_terms, _vertical_force_items,
    analyze_cantilever_wall,
)
from retaining_walls.earth_pressure import (
    active_force_components, active_force_components_water,
)
from retaining_walls.geometry import CantileverWallGeometry
from retaining_walls.results import CantileverWallResult

EM = S.EM
SCHEMA = "reinforced_wall_section/1"
BASIS = ("USACE EM 1110-2-2104 (1 November 2023), strength design with the "
         "serviceability single load factors")
BASIS_SERVICE_STRESS = (
    "USACE EM 1110-2-2104 (1 November 2023), strength design with the "
    "service stress limits of Table 3-3")

_MM_PER_IN = 25.4

#: Para. 3-5, p. 27: "Design should normally be based on 60,000 psi (414
#: MPa), the yield strength of ASTM Grade 60 reinforcement"; "Other
#: grades, up to a maximum of 80,000 psi (552 MPa), may be used".
FY_DEFAULT_MPA = 414.0
FY_MAX_MPA = 552.0
_FY_60_PSI = 60000.0

#: Table E-2, p. 93, and para. 3-3d, p. 22 (D 1.2 or 0.9; EV 1.35 or 1.0;
#: EH, active pressure, 1.5); Table 3-2, pp. 24-25 (Hs as a companion
#: load 1.0; ES as the principal load of an unusual case 1.6).
STRENGTH_D = (1.2, 0.9)
STRENGTH_EV = (1.35, 1.0)
STRENGTH_EH = 1.5
STRENGTH_ES = 1.6
STRENGTH_HS = 1.0

CASES = {
    "1": ("Normal operating", "usual"),
    "2": ("Normal operating + surcharge", "unusual"),
}

#: The manual's printed metric pairs (pp. 4-5): a length this module
#: takes at the manual's inch value is reported at the metric value the
#: manual prints beside it.
_PRINTED_MM = {12.0: 300.0, 6.0: 150.0, 4.0: 100.0, 3.0: 75.0}


class _NoPressure(Exception):
    """The factored loads' resultant falls outside the base."""


def _mm(inches: float) -> float:
    return _PRINTED_MM.get(inches, round(inches * _MM_PER_IN, 1))


def _segment_force(seg: Optional[Tuple[float, float, float, float]],
                   x0: float, x1: float) -> Tuple[float, float]:
    """(force, its first moment about x = 0) of a pressure that runs in a
    straight line from (xa, qa) to (xb, qb) and is zero elsewhere, over
    x0 to x1."""
    if seg is None:
        return 0.0, 0.0
    xa, qa, xb, qb = seg
    lo, hi = max(x0, xa), min(x1, xb)
    if hi <= lo or xb <= xa:
        return 0.0, 0.0
    slope = (qb - qa) / (xb - xa)
    q0, q1 = qa + slope * (lo - xa), qa + slope * (hi - xa)
    L = hi - lo
    F = 0.5 * (q0 + q1) * L
    if F == 0.0:
        return 0.0, 0.0
    x = lo + L * (q0 + 2.0 * q1) / (3.0 * (q0 + q1))
    return F, F * x


def _q_at(seg: Tuple[float, float, float, float], x: float) -> float:
    """The pressure of a straight-line segment at ``x`` (0 outside it)."""
    xa, qa, xb, qb = seg
    if not xa <= x <= xb or xb <= xa:
        return 0.0
    return qa + (qb - qa) * (x - xa) / (xb - xa)


def _base_pressure(V: float, x_r: float, B: float
                   ) -> Tuple[Tuple[float, float, float, float], str]:
    """The bearing pressure under the base for structural design: the
    trapezoid, or the triangle where the resultant is outside the middle
    third (FHWA NHI-06-089 Fig. 10-17, p. 10-39: "use trapezoidal
    distribution with qmax and qmin for rocks and structural analysis";
    FHWA GEC-6 Sec. 6.6, pp. 121-122). As a segment (x from the toe)."""
    if V <= 0 or not 0.0 < x_r < B:
        raise _NoPressure()
    e = B / 2.0 - x_r
    if abs(e) <= B / 6.0:
        return ((0.0, V / B * (1.0 + 6.0 * e / B),
                 B, V / B * (1.0 - 6.0 * e / B)), "trapezoidal")
    if e > 0:
        w = 3.0 * x_r
        return (0.0, 2.0 * V / w, w, 0.0), "triangular_toe"
    w = 3.0 * (B - x_r)
    return (B - w, 0.0, B, 2.0 * V / w), "triangular_heel"


def _load_type(item: Dict[str, Any]) -> str:
    """Table E-1's load (p. 92) of one of the wall module's vertical
    force items: D (the concrete), EV (soil and a permanent surcharge on
    the base), EH (the active thrust's vertical component) or Hs (the
    uplift)."""
    if item["kind"] == "uplift":
        return "Hs"
    if item["kind"] == "thrust":
        return "EH"
    if item["kind"] == "weight" and item["component"].startswith(
            ("Base slab", "Stem (")):
        return "D"
    return "EV"


class _Loads:
    """The unfactored loads on the wall, by Table E-1's load names, on
    the wall module's own thrust and weights."""

    def __init__(self, geom: CantileverWallGeometry, soil: Dict[str, Any]):
        self.geom = geom
        g = geom
        gamma, c = soil["gamma_backfill"], soil["c_backfill"]
        gamma_c = soil["gamma_concrete"]
        thrust_kw = dict(
            gamma_backfill=gamma, phi_backfill=soil["phi_backfill"],
            c_backfill=c, pressure_method=soil["pressure_method"],
            wall_friction_deg=soil["wall_friction_deg"],
            gamma_sat_backfill=soil["gamma_sat_backfill"])
        g0 = dataclasses.replace(g, surcharge=0.0)
        t0 = _active_thrust_terms(g0, **thrust_kw)
        t1 = _active_thrust_terms(g, **thrust_kw)
        self.thrust, self.thrust_no_live = t1, t0
        self.q_live, self.q_dead = g.surcharge, g.dead_load_surcharge
        self.B = g.base_width
        self.heel = heel = max(g.heel_length, 0.0)
        self.toe = g.toe_length
        self.x_back = g.toe_length + g.stem_thickness_base
        self.h_w = t1["h_w"]
        self.u_heel = t1["u_heel"]
        self.base_uplift = soil["base_uplift"] or "linear"

        # vertical forces (kN/m) and arms from the toe, without the live
        # load; every item counted (the structure carries what is there)
        items = _vertical_force_items(
            g0, gamma, gamma_c, t0, live_load_resists=True,
            dead_load_resists=True, surcharge_in_bearing=True,
            gamma_front=soil["gamma_foundation"],
            base_uplift=soil["base_uplift"], uplift_in_bearing=True)
        self.vertical = [{"name": i["component"], "type": _load_type(i),
                          "W": i["W"], "x": i["x"]} for i in items]
        # the live load: its weight on the heel and its share of the thrust
        self.W_live_heel = self.q_live * heel
        self.x_live_heel = self.x_back + heel / 2.0
        self.Pv_live = t1["Pv"] - t0["Pv"]
        self.Pv_earth = t0["Pv"]
        # horizontal forces and their moments about the underside of base
        self.horizontal = {
            "EH": (t0["Ph"], t0["Ph"] * t0["z_Pa"]),
            "ES": (t1["Ph"] - t0["Ph"],
                   t1["Ph"] * t1["z_Pa"] - t0["Ph"] * t0["z_Pa"]),
            "Hs": (t1["Pw"], t1["Pw"] * t1["z_Pw"]),
        }

        # distributed loads on the toe and the heel slabs (kPa)
        self.w_slab = gamma_c * g.base_thickness
        g_front = (gamma if soil["gamma_foundation"] is None
                   else soil["gamma_foundation"])
        self.w_soil_toe = g_front * max(g.Df - g.base_thickness, 0.0)
        s_w = max(min(self.h_w, g.wall_height) - g.base_thickness, 0.0)
        excess = ((t1["gamma_sat"] - gamma) if self.h_w > 0 else 0.0)
        self.w_soil_heel = (gamma * g.stem_height + excess * s_w
                            + self.q_dead)
        rise = heel * math.tan(math.radians(g.backfill_slope))
        self.W_wedge = 0.5 * gamma * heel * rise

        # the stem: the pressure on the vertical plane through the back
        # of the stem's base, over the stem's height
        taper = (max(g.stem_thickness_base - g.stem_thickness_top, 0.0)
                 if g.stem_batter == "back" else 0.0)
        h = g.stem_height + taper * math.tan(math.radians(g.backfill_slope))
        self.h_stem_plane = h
        hw_s = 0.0
        if g.water_table_depth is not None:
            hw_s = max(g.stem_height - g.water_table_depth, 0.0)
        self.h_w_stem = hw_s
        Ka = t1["Ka"]
        cos_d = math.cos(math.radians(t1["delta_deg"]))

        def parts(q):
            if hw_s > 0:
                p = active_force_components_water(
                    gamma, t1["gamma_b"], h, h - hw_s, Ka, c, q)
            else:
                p = active_force_components(gamma, h, Ka, c, q)
            return p["Pa"] * cos_d, p["Pa"] * cos_d * p["z_Pa"]

        e_V, e_M = parts(self.q_dead)
        a_V, a_M = parts(self.q_dead + self.q_live)
        Pw = 0.5 * GAMMA_W * hw_s ** 2
        self.stem = {"EH": (e_V, e_M), "ES": (a_V - e_V, a_M - e_M),
                     "Hs": (Pw, Pw * hw_s / 3.0)}

    # -- the base under one set of load factors -------------------------
    def resolve(self, f: Dict[str, float], live_on_heel: bool
                ) -> Dict[str, Any]:
        V = sum(f[i["type"]] * i["W"] for i in self.vertical)
        M = sum(f[i["type"]] * i["W"] * i["x"] for i in self.vertical)
        if live_on_heel:
            V += f["ES"] * self.W_live_heel
            M += f["ES"] * self.W_live_heel * self.x_live_heel
        V += f["ES"] * self.Pv_live
        M += f["ES"] * self.Pv_live * self.B
        M_over = sum(f[k] * m for k, (_, m) in self.horizontal.items())
        x_r = (M - M_over) / V if V > 0 else -1.0
        seg, shape = _base_pressure(V, x_r, self.B)
        uplift = None
        if self.h_w > 0:
            u = f["Hs"] * self.u_heel
            uplift = ((0.0, 0.0, self.B, u) if self.base_uplift == "linear"
                      else (0.0, u, self.B, u))
        return {"V": V, "x_r": x_r, "e": self.B / 2.0 - x_r,
                "pressure": seg, "shape": shape, "uplift": uplift,
                "q_toe": seg[1] if seg[0] == 0.0 else 0.0,
                "q_heel": seg[3] if seg[2] == self.B else 0.0}

    def stem_actions(self, f, live_on_heel=None, d=None):
        """(M, V) at the base of the stem, kN-m/m and kN/m."""
        M = sum(f[k] * m for k, (_, m) in self.stem.items())
        V = sum(f[k] * v for k, (v, _) in self.stem.items())
        return M, V, {"parts": {"factors": dict(f)}}

    def toe_actions(self, f, live_on_heel, d):
        """(M, V): the moment at the front face of the stem (tension in
        the bottom positive) and the shear at ``d`` from it (para. 5-3,
        pp. 35-36)."""
        st = self.resolve(f, live_on_heel)
        xf = self.toe
        w_down = f["D"] * self.w_slab + f["EV"] * self.w_soil_toe
        F = Mx = 0.0
        for seg in (st["pressure"], st["uplift"]):
            dF, dMx = _segment_force(seg, 0.0, xf)
            F, Mx = F + dF, Mx + dMx
        M = (F * xf - Mx) - w_down * xf ** 2 / 2.0
        xs = xf - d
        V = Fs = 0.0
        if xs > 0:
            Fs = sum(_segment_force(seg, 0.0, xs)[0]
                     for seg in (st["pressure"], st["uplift"]))
            V = Fs - w_down * xs
        st["parts"] = {
            "factors": dict(f), "length": xf, "w_down": w_down,
            "up_force": F, "up_moment": F * xf - Mx,
            "down_moment": w_down * xf ** 2 / 2.0,
            "shear_section": max(xs, 0.0), "up_force_shear": Fs,
            "down_force_shear": w_down * max(xs, 0.0),
            "q_at_face": _q_at(st["pressure"], xf)}
        return M, V, st

    def heel_actions(self, f, live_on_heel, d=None):
        """(M, V) at the back face of the stem (tension in the top
        positive), the section para. 5-3 gives for the heel's shear."""
        st = self.resolve(f, live_on_heel)
        L, xb = self.heel, self.x_back
        w = f["D"] * self.w_slab + f["EV"] * self.w_soil_heel
        if live_on_heel:
            w += f["ES"] * self.q_live
        tip = f["EH"] * self.Pv_earth + f["ES"] * self.Pv_live
        down_F = w * L + f["EV"] * self.W_wedge + tip
        down_M = (w * L ** 2 / 2.0 + f["EV"] * self.W_wedge * 2.0 * L / 3.0
                  + tip * L)
        F = Mx = 0.0
        for seg in (st["pressure"], st["uplift"]):
            dF, dMx = _segment_force(seg, xb, self.B)
            F, Mx = F + dF, Mx + dMx
        st["parts"] = {
            "factors": dict(f), "length": L, "w_down": w, "tip": tip,
            "wedge": f["EV"] * self.W_wedge, "down_force": down_F,
            "down_moment": down_M, "up_force": F,
            "up_moment": Mx - F * xb,
            "q_at_face": _q_at(st["pressure"], xb)}
        return down_M - (Mx - F * xb), down_F - F, st


def _unit_factors(case2: bool) -> Dict[str, float]:
    return {"D": 1.0, "EV": 1.0, "EH": 1.0, "Hs": 1.0,
            "ES": 1.0 if case2 else 0.0}


def _case_rows(actions, d: Optional[float], has_live: bool,
               single_factor: bool) -> Tuple[List[dict], List[dict],
                                              List[str]]:
    """Every load combination's moment and shear on one element:
    ``(rows, service, warnings)``. ``service`` holds the unfactored
    moments, one per case, for the serviceability checks."""
    rows, service, warnings = [], [], []
    for case in ("1", "2") if has_live else ("1",):
        name, category = CASES[case]
        case2 = case == "2"
        places = (True, False) if case2 else (None,)
        # serviceability: unfactored loads, the results times the single
        # load factor (para. 3-1f, pp. 13-14; Table 3-4, p. 26)
        best = None
        for place in places:
            try:
                M, V, st = actions(_unit_factors(case2), bool(place), d)
            except _NoPressure:
                warnings.append(
                    f"case {case}, unfactored loads: the resultant falls "
                    "outside the base, so there is no bearing pressure to "
                    "design the base slab for")
                continue
            if best is None or M > best[0]:
                best = (M, place, st)
            lf = S.SINGLE_LOAD_FACTOR[category]
            if single_factor:
                rows.append({
                    "case": case, "description": name,
                    "category": category, "column": "serviceability",
                    "factors": f"{lf:g} (D + EH + EV"
                               + (" + ES" if case2 else "") + " + Hs)",
                    "surcharge_over_heel": place,
                    "M": lf * M, "V": lf * V, "state": st})
        if best is not None:
            service.append({"case": case, "category": category,
                            "M": best[0], "surcharge_over_heel": best[1],
                            "state": best[2]})
        # strength: factored loads (para. 3-1f; Table E-2, p. 93)
        for place in places:
            for fD in STRENGTH_D:
                for fEV in STRENGTH_EV:
                    f = {"D": fD, "EV": fEV, "EH": STRENGTH_EH,
                         "Hs": STRENGTH_HS,
                         "ES": STRENGTH_ES if case2 else 0.0}
                    try:
                        M, V, st = actions(f, bool(place), d)
                    except _NoPressure:
                        warnings.append(
                            f"case {case}, strength, {fD:g} D + "
                            f"{STRENGTH_EH:g} EH + {fEV:g} EV: the "
                            "factored resultant falls outside the base; "
                            "the combination gives no bearing pressure "
                            "and is left out")
                        continue
                    rows.append({
                        "case": case, "description": name,
                        "category": category, "column": "strength",
                        "factors": f"{fD:g} D + {STRENGTH_EH:g} EH + "
                                   f"{fEV:g} EV"
                                   + (f" + {STRENGTH_ES:g} ES"
                                      if case2 else "")
                                   + f" + {STRENGTH_HS:g} Hs",
                        "surcharge_over_heel": place,
                        "M": M, "V": V, "state": st})
    return rows, service, sorted(set(warnings))


def _both_moment(M_kNm: Optional[float], key: str) -> Dict[str, Any]:
    if M_kNm is None:
        return {f"{key}_kNm_per_m": None, f"{key}_kipft_per_ft": None}
    return {f"{key}_kNm_per_m": round(M_kNm, 2),
            f"{key}_kipft_per_ft": round(kNm_per_m_to_kipft_per_ft(M_kNm),
                                         3)}


def _both_force(V_kN: Optional[float], key: str) -> Dict[str, Any]:
    if V_kN is None:
        return {f"{key}_kN_per_m": None, f"{key}_kip_per_ft": None}
    return {f"{key}_kN_per_m": round(V_kN, 2),
            f"{key}_kip_per_ft": round(kN_per_m_to_kip_per_ft(V_kN), 3)}


def _both_area(As_in2: Optional[float], key: str) -> Dict[str, Any]:
    if As_in2 is None:
        return {f"{key}_mm2_per_m": None, f"{key}_in2_per_ft": None}
    return {f"{key}_mm2_per_m": round(in2_per_ft_to_mm2_per_m(As_in2), 1),
            f"{key}_in2_per_ft": round(As_in2, 3)}


_ELEMENT = {
    "stem": {"section": "the base of the stem (the top of the base slab), "
                        "for moment and for shear",
             "section_basis": f"{EM} para. 5-3 (pp. 35-36): with a "
                              "construction joint at the base of a "
                              "cantilevered stem 'the critical section "
                              "for shear must be taken at the base of the "
                              "stem'; D-3 (p. 73) designs for 'the moment "
                              "at the base of a retaining wall stem'",
             "tension_face": "back"},
    "toe": {"section": "moment at the front face of the stem; shear at d "
                       "from the front face of the stem",
            "section_basis": f"shear: {EM} para. 5-3 (p. 36) and Fig. 5-1, "
                             "'at a distance, d, from the front face of "
                             "the wall stem for the toe section'. Moment: "
                             "at the front face of the stem, by statics; "
                             "no held reference prints the section (the "
                             "manual, p. 87, refers to EM 1110-2-2502, "
                             "which is not held)",
            "tension_face": "bottom"},
    "heel": {"section": "moment and shear at the back face of the stem",
             "section_basis": f"shear: {EM} para. 5-3 (p. 36) and Fig. "
                              "5-1, 'at the back face of the wall stem "
                              "for the heel section'. Moment: at the same "
                              "face, by statics; no held reference prints "
                              "the section (the manual, p. 87, refers to "
                              "EM 1110-2-2502, which is not held)",
             "tension_face": "top"},
}


def _governing(rows: List[dict], key: str) -> Optional[dict]:
    live = [r for r in rows if r[key] is not None]
    if not live:
        return None
    if key == "V":
        return max(live, key=lambda r: abs(r["V"]))
    return max(live, key=lambda r: r["M"])


def _row_out(r: dict) -> Dict[str, Any]:
    out = {"case": r["case"], "description": r["description"],
           "category": r["category"], "column": r["column"],
           "factors": r["factors"]}
    if r["surcharge_over_heel"] is not None:
        out["surcharge_over_heel"] = r["surcharge_over_heel"]
    out.update(_both_moment(r["M"], "M"))
    out.update(_both_force(r["V"], "V"))
    st = r.get("state")
    if st is not None and "x_r" in st:
        out["base_pressure"] = {
            "V_kN_per_m": round(st["V"], 2),
            "resultant_from_toe_m": round(st["x_r"], 4),
            "eccentricity_m": round(st["e"], 4), "shape": st["shape"],
            "q_toe_kPa": round(st["q_toe"], 2),
            "q_heel_kPa": round(st["q_heel"], 2)}
    return out


def _case_summary(rows: List[dict]) -> List[Dict[str, Any]]:
    """One line per case and column: the largest moment and the largest
    shear of its combinations, each with the factors that gave it."""
    out, seen = [], []
    for r in rows:
        key = (r["case"], r["column"])
        if key not in seen:
            seen.append(key)
    for case, column in seen:
        group = [r for r in rows if (r["case"], r["column"]) == (case,
                                                                 column)]
        gm, gv = _governing(group, "M"), _governing(group, "V")
        line = {"case": case, "description": gm["description"],
                "category": gm["category"], "column": column,
                **_both_moment(gm["M"], "M"), "M_factors": gm["factors"],
                **_both_force(abs(gv["V"]), "V"),
                "V_factors": gv["factors"]}
        if gm["surcharge_over_heel"] is not None:
            line["M_surcharge_over_heel"] = gm["surcharge_over_heel"]
            line["V_surcharge_over_heel"] = gv["surcharge_over_heel"]
        out.append(line)
    return out


def _design_element(name: str, actions, *, h_m: float, cover_in: float,
                    cover_given: bool, has_live: bool, single_factor: bool,
                    mat: Dict[str, Any], bar_size: Optional[str],
                    spacing_mm: Optional[float], As_min_in2: float
                    ) -> Dict[str, Any]:
    """One of stem, toe and heel: its load cases, its governing moment and
    shear and its section design."""
    h_in = h_m * 1000.0 / _MM_PER_IN
    spacing_in = None if spacing_mm is None else spacing_mm / _MM_PER_IN

    def design(rows, service):
        gm, gv = _governing(rows, "M"), _governing(rows, "V")
        Mu = max(gm["M"], 0.0) if gm else 0.0
        Vu = abs(gv["V"]) if gv else 0.0
        sec = S.design_section(
            kNm_per_m_to_kipft_per_ft(Mu), kN_per_m_to_kip_per_ft(Vu),
            h_in, cover_in, mat["fc_psi"], mat["fy_psi"],
            beta1=mat["beta1"], As_min_in2=As_min_in2, bar_size=bar_size,
            spacing_in=spacing_in,
            service_moments=[
                {"case": s["case"], "category": s["category"],
                 "M_kipft": kNm_per_m_to_kipft_per_ft(max(s["M"], 0.0))}
                for s in service],
            modular_ratio=mat["modular_ratio"],
            max_aggregate_in=mat["max_aggregate_in"])
        return gm, gv, Mu, Vu, sec

    # the toe's shear section is d from the stem, and d follows the bar:
    # the bar is chosen on the moment, then the shear is taken at its d
    d_m = None
    if name == "toe":
        rows, service, warnings = _case_rows(actions, 0.0, has_live,
                                             single_factor)
        d_m = design(rows, service)[4]["d_in"] * _MM_PER_IN / 1000.0
    rows, service, warnings = _case_rows(actions, d_m, has_live,
                                         single_factor)
    gm, gv, Mu, Vu, sec = design(rows, service)
    reverse = min((r["M"] for r in rows), default=0.0)

    info = _ELEMENT[name]
    spacing_out = (round(spacing_mm, 1) if spacing_mm is not None
                   else _mm(sec["spacing_in"]))
    el = {
        "element": name, "section": info["section"],
        "section_basis": info["section_basis"],
        "tension_face": info["tension_face"],
        "thickness_mm": round(h_m * 1000.0, 1),
        "thickness_in": round(h_in, 2),
        "cover_mm": mat["cover_mm"][name],
        "cover_in": round(cover_in, 3),
        "cover_given": cover_given,
        "d_mm": round(sec["d_in"] * _MM_PER_IN, 1),
        "d_in": round(sec["d_in"], 3),
        "governing_moment_case": _row_out(gm) if gm else None,
        "governing_shear_case": _row_out(gv) if gv else None,
        **_both_moment(Mu, "Mu"), **_both_force(Vu, "Vu"),
        "Ku": None if sec["Ku"] is None else round(sec["Ku"], 4),
        **_both_area(sec["As_flexure_in2"], "As_flexure"),
        **_both_area(sec["As_min_in2"], "As_minimum"),
        **_both_area(sec["As_required_in2"], "As_required"),
        **_both_area(sec["As_provided_in2"], "As_provided"),
        "bar": {"size": sec["bar_size"],
                "diameter_mm": round(sec["bar_diameter_in"] * _MM_PER_IN
                                     + 1e-9, 1),
                "diameter_in": sec["bar_diameter_in"],
                "spacing_mm": spacing_out,
                "spacing_in": round(sec["spacing_in"], 2),
                "chosen_by": sec["chosen_by"]},
        **_both_moment(kipft_per_ft_to_kNm_per_m(sec["phi_Mn_kipft"]),
                       "phi_Mn"),
        **_both_force(kip_per_ft_to_kN_per_m(sec["phi_Vc_kip"]), "phi_Vc"),
        "phi_Vc_SI_form_kN_per_m": round(
            kip_per_ft_to_kN_per_m(sec["phi_Vc_SI_form_kip"]), 2),
        "rho": round(sec["rho"], 5), "rho_balanced": round(sec["rho_b"], 5),
        "service": [
            {"case": s["case"], "category": s["category"],
             **_both_moment(kipft_per_ft_to_kNm_per_m(s["M_kipft"]),
                            "M_service"),
             "fs_MPa": (None if s["fs_psi"] is None
                        else round(s["fs_psi"] / MPa_to_psi(1.0), 1)),
             "fs_ksi": (None if s["fs_psi"] is None
                        else round(s["fs_psi"] / 1000.0, 2)),
             "stress_basis": s["stress_basis"],
             "s_max_mm": (None if s["s_max_in"] is None
                          else round(s["s_max_in"] * _MM_PER_IN, 0)),
             "s_max_in": (None if s["s_max_in"] is None
                          else round(s["s_max_in"], 2))}
            for s in sec["service"]],
        "load_cases": _case_summary(rows),
        "checks": sec["checks"],
        "warnings": warnings,
        "_section": sec, "_rows": rows, "_service": service,
    }
    if reverse < -1e-9:
        el["reverse_moment"] = {
            **_both_moment(-reverse, "M"),
            "note": ("a load combination puts the tension in the other "
                     "face of the slab (the "
                     + ("top" if name == "toe" else "bottom")
                     + "); the bars of that face run the full width of "
                     "the base and are not checked for it here")}
    el["passes"] = all(c["passes"] for c in sec["checks"]
                       if c["required"])
    return el


@dataclass
class CantileverWallStructuralResult:
    """The structural design of a cantilever retaining wall's concrete.

    Attributes
    ----------
    stability : CantileverWallResult
        The stability analysis of the same wall, run first.
    geometry : dict
        The wall as the module resolved it (m, deg).
    materials : dict
        fc and fy in MPa and psi, beta1, the modular ratio where given.
    load_cases : list of dict
        Table E-2's cases and which were designed.
    elements : dict
        "stem", "toe", "heel": each the section's place, its governing
        load case, Mu and Vu, d, the steel required and provided, the bar
        set, phi Mn, phi Vc and each check with its page.
    secondary : list of dict
        The temperature and shrinkage bar sets.
    development : dict
        The stem bars' standard hook into the base.
    checks : list of dict
        The wall-level checks (minimum thickness, the hook).
    reinforcement : dict
        The description a drawing is made from
        (``"reinforced_wall_section/1"``).
    choices, not_designed, warnings : list
    """
    stability: CantileverWallResult
    geometry: Dict[str, Any]
    materials: Dict[str, Any]
    load_cases: List[Dict[str, Any]]
    elements: Dict[str, Dict[str, Any]]
    secondary: List[Dict[str, Any]]
    development: Dict[str, Any]
    checks: List[Dict[str, Any]]
    reinforcement: Dict[str, Any]
    choices: List[Dict[str, Any]] = field(default_factory=list)
    not_designed: List[Dict[str, str]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    basis: str = BASIS
    #: The structural design's own choices (``choices`` holds the
    #: stability analysis's too).
    structural_choices: List[Dict[str, Any]] = field(default_factory=list)
    #: The unfactored loads the design worked from (for the calc package).
    loads: Any = field(default=None, repr=False)

    # -- the checks, failed first ---------------------------------------
    def _all_checks(self) -> List[Tuple[str, Dict[str, Any]]]:
        out = [(name, c) for name, el in self.elements.items()
               for c in el["checks"]]
        return out + [("wall", c) for c in self.checks]

    @property
    def structural_checks_failed(self) -> List[str]:
        return [f"{where}: {c['name']}" for where, c in self._all_checks()
                if c["required"] and not c["passes"]]

    @property
    def recommendations_not_met(self) -> List[str]:
        return [f"{where}: {c['name']}" for where, c in self._all_checks()
                if not c["required"] and not c["passes"]]

    @property
    def checks_failed(self) -> List[str]:
        return (list(self.stability.checks_failed)
                + self.structural_checks_failed)

    @property
    def checks_not_run(self) -> List[str]:
        out = list(self.stability.checks_not_run)
        if not self.development.get("computed", False):
            out.append("stem bars: hook development length")
        return out

    @property
    def all_checks_pass(self) -> bool:
        return not self.checks_failed

    @staticmethod
    def _words(where: str, c: Dict[str, Any]) -> str:
        def n(v):
            return f"{v:.4g}" if isinstance(v, (int, float)) else str(v)
        unit = f" {c['unit']}" if c["unit"] else ""
        return (f"{where}: {c['name']} ({n(c['demand'])} against "
                f"{n(c['capacity'])}{unit})")

    @property
    def verdict(self) -> str:
        """The result in plain words: failed checks first, the stability
        checks' own verdict, then what was not designed."""
        failed = [self._words(w, c) for w, c in self._all_checks()
                  if c["required"] and not c["passes"]]
        parts = []
        if failed:
            parts.append("CONCRETE FAILS: " + "; ".join(failed) + ".")
        else:
            parts.append("Concrete: no check failed.")
        bars = "; ".join(
            f"{name} {el['bar']['size']} @ {el['bar']['spacing_mm']:g} mm"
            for name, el in self.elements.items())
        parts.append(f"Main bars: {bars}.")
        rec = self.recommendations_not_met
        if rec:
            parts.append("Recommended limit not met: " + "; ".join(rec)
                         + ".")
        parts.append("Stability: " + self.stability.verdict)
        parts.append("Not designed: " + "; ".join(
            n["what"] for n in self.not_designed) + ".")
        return " ".join(parts)

    def summary(self) -> str:
        lines = ["=" * 60,
                 "  CANTILEVER RETAINING WALL: STRUCTURAL DESIGN",
                 "=" * 60, "", f"  Basis: {self.basis}",
                 f"  f'c = {self.materials['fc_MPa']:g} MPa "
                 f"({self.materials['fc_psi']:,.0f} psi), fy = "
                 f"{self.materials['fy_MPa']:g} MPa "
                 f"({self.materials['fy_psi']:,.0f} psi)", ""]
        for name, el in self.elements.items():
            lines += [
                f"  {name.upper()}  ({el['section']})",
                f"    Mu = {el['Mu_kNm_per_m']:.1f} kN-m/m "
                f"({el['Mu_kipft_per_ft']:.2f} kip-ft/ft), Vu = "
                f"{el['Vu_kN_per_m']:.1f} kN/m "
                f"({el['Vu_kip_per_ft']:.2f} kips/ft)",
                f"    d = {el['d_mm']:.0f} mm ({el['d_in']:.2f} in.), "
                f"cover {el['cover_mm']:g} mm",
                f"    {el['bar']['size']} @ {el['bar']['spacing_mm']:g} mm "
                f"({el['bar']['spacing_in']:g} in.), {el['tension_face']} "
                f"face: As = {el['As_provided_mm2_per_m']:.0f} mm2/m "
                f"(required "
                + ("none: section too thin"
                   if el['As_required_mm2_per_m'] is None
                   else f"{el['As_required_mm2_per_m']:.0f}") + ")",
                f"    phi Mn = {el['phi_Mn_kNm_per_m']:.1f} kN-m/m, "
                f"phi Vc = {el['phi_Vc_kN_per_m']:.1f} kN/m  "
                f"[{'OK' if el['passes'] else 'FAIL'}]", ""]
        lines.append("  Temperature and shrinkage steel:")
        for b in self.secondary:
            lines.append(f"    {b['callout']}")
        d = self.development
        if d.get("computed"):
            lines += ["", f"  Stem bars' standard hook into the base: ldh = "
                          f"{d['ldh_mm']:.0f} mm ({d['ldh_in']:.1f} in.), "
                          f"{d['available_mm']:.0f} mm available"]
        failed = self.structural_checks_failed
        lines += ["", "  Failed concrete checks: "
                  + ("; ".join(failed) if failed else "none"),
                  "  Stability: " + self.stability.verdict,
                  "  Not designed: "
                  + "; ".join(n["what"] for n in self.not_designed)]
        if self.choices:
            from geotech_common.choices import choices_lines
            lines.extend(choices_lines(self.choices))
        lines += ["", "=" * 60]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        def public(el):
            return {k: v for k, v in el.items() if not k.startswith("_")}
        return {
            "verdict": self.verdict,
            "checks_failed": self.checks_failed,
            "checks_not_run": self.checks_not_run,
            "recommendations_not_met": self.recommendations_not_met,
            "basis": self.basis,
            "materials": self.materials,
            "geometry": self.geometry,
            "load_cases": self.load_cases,
            "stem": public(self.elements["stem"]),
            "toe": (public(self.elements["toe"])
                    if "toe" in self.elements else None),
            "heel": (public(self.elements["heel"])
                     if "heel" in self.elements else None),
            "temperature_shrinkage": self.secondary,
            "development": self.development,
            "wall_checks": self.checks,
            "reinforcement": self.reinforcement,
            "not_designed": self.not_designed,
            "warnings": self.warnings,
            "choices": self.choices,
            # its choices are in ``choices`` above
            "stability": {k: v for k, v in self.stability.to_dict().items()
                          if k != "choices"},
        }


def _cover(name: str, given_mm: Optional[float], thickness_in: float,
           against_ground: bool, input_name: str, what: str
           ) -> Tuple[float, float, bool]:
    """(cover in., the mm reported, given?) for one face: the engineer's,
    else Table 2-1's (p. 5); required where the table prints none."""
    if given_mm is not None:
        if given_mm <= 0:
            raise ValueError(f"{input_name} must be positive (mm), got "
                             f"{given_mm:g}")
        return given_mm / _MM_PER_IN, round(float(given_mm), 1), True
    cover = S.table_2_1_cover_in(thickness_in, against_ground)
    if cover is None:
        raise ValueError(
            f"{input_name} is required: the {what} is "
            f"{thickness_in * _MM_PER_IN:.0f} mm ({thickness_in:.1f} in.) "
            f"thick, and for a section 12 in. (30 cm) thick or less {EM} "
            "Table 2-1 (p. 5) prints no cover, only 'According to ACI "
            "318-19', which is not held. Give the clear cover in mm (the "
            "manual's own 9 in. wall example, D-3, p. 75, takes 2 in., "
            "about 50 mm), or make the section thicker than 305 mm.")
    return cover, _mm(cover), False


def _ts_set(mark: str, element: str, face: str, runs: str, bar: dict,
            cover_mm: float, where: str) -> Dict[str, Any]:
    s_mm, s_in = _mm(bar["spacing_in"]), bar["spacing_in"]
    return {
        "mark": mark, "element": element, "face": face, "runs": runs,
        "role": "temperature_shrinkage", "size": bar["bar_size"],
        "diameter_mm": round(bar["bar_diameter_in"] * _MM_PER_IN + 1e-9, 1),
        "spacing_mm": s_mm, "cover_mm": cover_mm, "end": None,
        "callout": f"{bar['bar_size']} @ {s_mm:g} mm ({s_in:g} in.), "
                   f"{where}",
        "As_required_mm2_per_m": round(in2_per_ft_to_mm2_per_m(
            bar["As_required_in2"]), 1),
        "As_provided_mm2_per_m": round(in2_per_ft_to_mm2_per_m(
            bar["As_provided_in2"]), 1),
    }


def _flexure_set(mark: str, el: Dict[str, Any], where: str,
                 end: Optional[dict]) -> Dict[str, Any]:
    b = el["bar"]
    return {
        "mark": mark, "element": el["element"],
        "face": el["tension_face"], "runs": "in_section",
        "role": "flexure", "size": b["size"],
        "diameter_mm": b["diameter_mm"], "spacing_mm": b["spacing_mm"],
        "cover_mm": el["cover_mm"], "end": end,
        "callout": f"{b['size']} @ {b['spacing_mm']:g} mm "
                   f"({b['spacing_in']:g} in.), {where}",
        "As_required_mm2_per_m": el["As_required_mm2_per_m"],
        "As_provided_mm2_per_m": el["As_provided_mm2_per_m"],
    }


def analyze_cantilever_wall_structure(
        geom: CantileverWallGeometry, gamma_backfill: float,
        phi_backfill: float, *, fc_MPa: float, joint_spacing: float,
        fy_MPa: float = FY_DEFAULT_MPA,
        stem_cover_mm: Optional[float] = None,
        base_top_cover_mm: Optional[float] = None,
        base_bottom_cover_mm: Optional[float] = None,
        stem_bar_size: Optional[str] = None,
        stem_bar_spacing_mm: Optional[float] = None,
        toe_bar_size: Optional[str] = None,
        toe_bar_spacing_mm: Optional[float] = None,
        heel_bar_size: Optional[str] = None,
        heel_bar_spacing_mm: Optional[float] = None,
        beta1: Optional[float] = None,
        modular_ratio: Optional[float] = None,
        max_aggregate_mm: Optional[float] = None,
        lap_length_mm: Optional[float] = None,
        stability: Optional[CantileverWallResult] = None,
        **wall_inputs) -> CantileverWallStructuralResult:
    """Design the concrete of a cantilever retaining wall (stem, toe and
    heel) on USACE EM 1110-2-2104 (1 November 2023).

    Parameters
    ----------
    geom, gamma_backfill, phi_backfill, **wall_inputs
        The wall, as :func:`analyze_cantilever_wall` takes it; that
        analysis runs first on the same inputs (``result.stability``).
    fc_MPa : float
        Concrete compressive strength f'c (MPa). Required: the manual
        prints no default.
    joint_spacing : float
        Length between control joints along the wall (m). Required: the
        temperature and shrinkage steel along the wall is by Table 2-3
        (p. 7), which goes by it.
    fy_MPa : float
        Steel yield strength (MPa). Default 414 (60,000 psi), para. 3-5,
        p. 27; at most 552 (80,000 psi). The single load factors of Table
        3-4 are printed for 60,000 psi only: with another fy the design is
        by the strength combinations and the service stress limits of
        Table 3-3, which needs ``modular_ratio``.
    stem_cover_mm, base_top_cover_mm, base_bottom_cover_mm : float
        Clear cover (mm) to the stem's bars (both faces), the base's top
        bars and its bottom bars. None: Table 2-1 (p. 5): 100 mm (4 in.)
        against the foundation; 100 mm for a section 24 in. (61 cm) thick
        or more and 75 mm (3 in.) for one over 12 in. (30 cm). Required
        for a section 12 in. thick or less, where the table prints none.
    *_bar_size, *_bar_spacing_mm : optional
        The main bars of the stem, the toe and the heel where the engineer
        fixes them ("No. 4" to "No. 11"; mm centre to centre).
    beta1 : float, optional
        None: ACI 318-19's value by the reference library
        (:data:`structural_section.BETA1_BASIS`).
    modular_ratio : float, optional
        n = Es / Ec. Given, the service stress in the main bars is worked
        out and checked against Table 3-3 (p. 26), and the crack-control
        spacing is on it. The manual prints no rule for n.
    max_aggregate_mm : float, optional
        Largest aggregate (mm), for the clear distance between bars and
        the cover (para. 2-6a, p. 4; Table 2-1 Note, p. 5). None: those
        two limits are on the bar diameter alone.
    lap_length_mm : float, optional
        A lap length the engineer supplies. It is printed in the notes
        and never checked: lap lengths are not designed.
    stability : CantileverWallResult, optional
        The stability analysis of this wall on these inputs, where the
        caller has run it already; None: it is run here.
    """
    from geotech_common.choices import DEFAULT, MODULE, choice, how_given

    if fc_MPa is None or fc_MPa <= 0:
        raise ValueError(f"fc_MPa (concrete strength, MPa) must be "
                         f"positive, got {fc_MPa}")
    if joint_spacing is None or joint_spacing <= 0:
        raise ValueError(
            "joint_spacing (the length between control joints along the "
            f"wall, m) must be positive, got {joint_spacing}: {EM} Table "
            "2-3 (p. 7) sets the temperature and shrinkage steel by it")
    if not 0 < fy_MPa <= FY_MAX_MPA + 0.5:
        raise ValueError(
            f"fy_MPa must be at most {FY_MAX_MPA:g} MPa (80,000 psi), got "
            f"{fy_MPa:g}: {EM} para. 3-5 (p. 27)")
    fy_psi = MPa_to_psi(fy_MPa)
    grade60 = abs(fy_psi - _FY_60_PSI) <= 0.005 * _FY_60_PSI
    if grade60:
        fy_psi = _FY_60_PSI
    elif modular_ratio is None:
        raise ValueError(
            f"fy_MPa {fy_MPa:g}: the single load factors that stand in "
            f"for the service stress limits ({EM} Table 3-4, p. 26) "
            "'apply only to design using reinforcing steel with yield "
            "strength of 60,000 psi (414 MPa)'. For another steel give "
            "modular_ratio (n = Es / Ec), and the design is by the "
            "strength load combinations with the service stresses checked "
            "against Table 3-3; or give fy_MPa 414.")
    if modular_ratio is not None and modular_ratio <= 0:
        raise ValueError(f"modular_ratio must be positive, got "
                         f"{modular_ratio}")
    if beta1 is not None and not 0.65 <= beta1 <= 0.85:
        raise ValueError(f"beta1 must be from 0.65 to 0.85, got {beta1}")
    for label, size in (("stem_bar_size", stem_bar_size),
                        ("toe_bar_size", toe_bar_size),
                        ("heel_bar_size", heel_bar_size)):
        if size is not None and size not in S.BARS:
            raise ValueError(f"{label} must be one of {list(S.BARS)}, got "
                             f"{size!r}")
    for label, s in (("stem_bar_spacing_mm", stem_bar_spacing_mm),
                     ("toe_bar_spacing_mm", toe_bar_spacing_mm),
                     ("heel_bar_spacing_mm", heel_bar_spacing_mm)):
        if s is not None and s <= 0:
            raise ValueError(f"{label} must be positive (mm), got {s}")

    # 1. the stability analysis of the same wall
    if stability is None:
        stability = analyze_cantilever_wall(geom, gamma_backfill,
                                            phi_backfill, **wall_inputs)
    soil = {
        "gamma_backfill": gamma_backfill, "phi_backfill": phi_backfill,
        "c_backfill": wall_inputs.get("c_backfill", 0.0),
        "pressure_method": wall_inputs.get("pressure_method", "rankine"),
        "wall_friction_deg": wall_inputs.get("wall_friction_deg"),
        "gamma_sat_backfill": wall_inputs.get("gamma_sat_backfill"),
        "gamma_concrete": wall_inputs.get("gamma_concrete",
                                          GAMMA_CONCRETE),
        "gamma_foundation": wall_inputs.get("gamma_foundation"),
        "base_uplift": wall_inputs.get("base_uplift"),
    }
    loads = _Loads(geom, soil)
    has_live = geom.surcharge > 0
    single_factor = grade60

    fc_psi = MPa_to_psi(fc_MPa)
    beta1_used = S.beta1_default(fc_psi) if beta1 is None else beta1
    t_stem_in = geom.stem_thickness_base * 1000.0 / _MM_PER_IN
    t_base_in = geom.base_thickness * 1000.0 / _MM_PER_IN
    c_stem, c_stem_mm, c_stem_given = _cover(
        "stem", stem_cover_mm, t_stem_in, False, "stem_cover_mm",
        "stem at its base")
    c_top, c_top_mm, c_top_given = _cover(
        "heel", base_top_cover_mm, t_base_in, False, "base_top_cover_mm",
        "base slab")
    c_bot, c_bot_mm, c_bot_given = _cover(
        "toe", base_bottom_cover_mm, t_base_in, True,
        "base_bottom_cover_mm", "base slab")
    mat = {"fc_psi": fc_psi, "fy_psi": fy_psi, "beta1": beta1_used,
           "modular_ratio": modular_ratio,
           "max_aggregate_in": (None if max_aggregate_mm is None
                                else max_aggregate_mm / _MM_PER_IN),
           "cover_mm": {"stem": c_stem_mm, "toe": c_bot_mm,
                        "heel": c_top_mm}}

    # 2. temperature and shrinkage steel (para. 2-9, pp. 6-8)
    joint_ft = joint_spacing / 0.3048
    ratio_along = S.temperature_shrinkage_ratio(joint_ft)
    ts = {
        "stem_vertical": S.temperature_shrinkage_area(0.003, t_stem_in),
        "stem_along": S.temperature_shrinkage_area(ratio_along, t_stem_in),
        "base_across": S.temperature_shrinkage_area(0.003, t_base_in),
        "base_along": S.temperature_shrinkage_area(ratio_along, t_base_in),
    }

    # 3. stem, toe and heel
    elements: Dict[str, Dict[str, Any]] = {}
    common = dict(has_live=has_live, single_factor=single_factor, mat=mat)
    elements["stem"] = _design_element(
        "stem", loads.stem_actions, h_m=geom.stem_thickness_base,
        cover_in=c_stem, cover_given=c_stem_given, bar_size=stem_bar_size,
        spacing_mm=stem_bar_spacing_mm, As_min_in2=ts["stem_vertical"],
        **common)
    if geom.toe_length > 0:
        elements["toe"] = _design_element(
            "toe", loads.toe_actions, h_m=geom.base_thickness,
            cover_in=c_bot, cover_given=c_bot_given, bar_size=toe_bar_size,
            spacing_mm=toe_bar_spacing_mm, As_min_in2=ts["base_across"],
            **common)
    if loads.heel > 0:
        elements["heel"] = _design_element(
            "heel", loads.heel_actions, h_m=geom.base_thickness,
            cover_in=c_top, cover_given=c_top_given, bar_size=heel_bar_size,
            spacing_mm=heel_bar_spacing_mm, As_min_in2=ts["base_across"],
            **common)
    for name, size, s in (("toe", toe_bar_size, toe_bar_spacing_mm),
                          ("heel", heel_bar_size, heel_bar_spacing_mm)):
        if name not in elements and (size is not None or s is not None):
            raise ValueError(
                f"the wall has no {name} (its length is 0), so "
                f"{name}_bar_size and {name}_bar_spacing_mm do nothing; "
                "leave them out")

    # 4. the bar sets, as the drawing takes them
    bar_B = S.temperature_shrinkage_bar(ts["stem_vertical"])
    bar_C = S.temperature_shrinkage_bar(ts["stem_along"])
    bar_F = S.temperature_shrinkage_bar(ts["base_along"])
    bar_base = S.temperature_shrinkage_bar(ts["base_across"])

    # the stem bars' standard hook into the base (D-5 Step 6, pp. 86-87)
    stem_sec = elements["stem"]["_section"]
    hook = S.hook_development_length(stem_sec["bar_diameter_in"], fc_psi,
                                     fy_psi)
    bottom_bar_in = (elements["toe"]["_section"]["bar_diameter_in"]
                     if "toe" in elements else bar_base["bar_diameter_in"])
    available_in = (t_base_in - c_bot - bottom_bar_in
                    - bar_F["bar_diameter_in"])
    wall_checks = []
    development = {
        "bars": "the stem's main bars (mark A), hooked into the base",
        "computed": hook["computed"],
        "ldh_mm": None, "ldh_in": None,
        "available_mm": round(available_in * _MM_PER_IN, 0),
        "available_in": round(available_in, 2),
        "available_basis": (
            "the base thickness less the bottom cover and the bottom "
            "mat's two layers: 'Typically, the hooks will rest on the "
            f"bottom mat' ({EM} D-5 Step 6(d), p. 87)"),
        "formula": ("ldh = [fy psi_e psi_c psi_r psi_o / (55 lambda "
                    "sqrt(f'c))] db^1.5, psi_e = psi_r = psi_o = lambda = "
                    "1.0, psi_c = f'c/15,000 + 0.6"),
        "reference": f"{EM} D-5 Step 6 (pp. 86-87), 'From ACI 318-19'",
    }
    end_A = None
    if hook["computed"]:
        development.update(
            ldh_mm=round(hook["ldh_in"] * _MM_PER_IN, 0),
            ldh_in=round(hook["ldh_in"], 2),
            psi_c=round(hook["psi_c"], 4))
        end_A = {"type": "standard_hook_90", "into": "base",
                 "ldh_mm": development["ldh_mm"]}
        wall_checks.append(S._check(
            "stem bars' hook development length fits in the base",
            round(hook["ldh_in"], 2), round(available_in, 2), "in.",
            hook["ldh_in"] <= available_in + 1e-9,
            development["reference"],
            note="thicken the base, or use a smaller stem bar at a closer "
                 "spacing" if hook["ldh_in"] > available_in else ""))
    else:
        development["why_not"] = hook["why"]
        end_A = {"type": "standard_hook_90", "into": "base", "ldh_mm": None}

    # minimum wall thickness (para. 3-7, p. 27)
    top_in = geom.stem_thickness_top * 1000.0 / _MM_PER_IN
    least = 12.0 if geom.stem_height > 3.048 else 8.0
    wall_checks.append(S._check(
        "stem thickness at its top >= the minimum wall thickness",
        least, round(top_in, 2), "in.", top_in >= least - 1e-9,
        f"{EM} para. 3-7 (p. 27): 'Walls with height greater than 10 ft "
        "(3 m) must be a minimum of 12 in. (30.5 cm) thick ... Walls must "
        "not be less than 8 in. (20 cm) thick'"))

    stem_el = elements["stem"]
    bars = [_flexure_set("A", stem_el, "back face", end_A),
            _ts_set("B", "stem", "front", "in_section", bar_B, c_stem_mm,
                    "front face"),
            _ts_set("C", "stem", "both", "along_wall", bar_C, c_stem_mm,
                    "each face, along the wall")]
    if "toe" in elements:
        bars.append(_flexure_set("D", elements["toe"], "bottom", None))
    else:
        bars.append(_ts_set("D", "base", "bottom", "in_section", bar_base,
                            c_bot_mm, "bottom"))
    if "heel" in elements:
        bars.append(_flexure_set("E", elements["heel"], "top", None))
    else:
        bars.append(_ts_set("E", "base", "top", "in_section", bar_base,
                            c_top_mm, "top"))
    bars.append(_ts_set("F", "base", "both", "along_wall", bar_F,
                        c_top_mm, "top and bottom, along the wall"))
    secondary = [b for b in bars if b["role"] == "temperature_shrinkage"]

    # 5. what was not designed, and why
    aci = ("the manual sends it to ACI 318-19 "
           "(para. 2-4, p. 4), which is not held")
    not_designed = [
        {"what": "earthquake load cases (Table E-2 cases 3A and 3B)",
         "why": "the seismic earth pressure is not part of this design; "
                f"{EM} para. 3-3h (p. 25) sends it to other manuals"},
        {"what": "the maximum hydrostatic case (Table E-2 case 4)",
         "why": "it needs a second water level and its return period; the "
                "water table given is taken as the normal operating level "
                "of cases 1 and 2"},
        {"what": "the construction case (Table E-2 case 5)",
         "why": "it needs the construction surcharge and the stage of the "
                "backfill"},
        {"what": "lap lengths and straight development lengths",
         "why": aci},
        {"what": "bar cut-offs up the stem",
         "why": "every bar runs the full height or width; cut-offs need "
                "development lengths, and " + aci},
        {"what": "shear friction across the joint at the base of the stem",
         "why": f"{EM} para. 5-3 (p. 36) asks that the vertical steel of "
                "both faces be developed into the base for it and prints "
                "no check"},
        {"what": "deflection",
         "why": f"{EM} para. 3-4b (p. 27) leaves the limits to the "
                "engineer; the 0.25 rho_b recommendation is reported"},
        {"what": "counterforts, and a stem or base with shear "
                 "reinforcement",
         "why": "the manual's walls and slabs are designed without shear "
                "reinforcement (para. 5-2b, p. 35)"},
    ]
    if geom.has_shear_key:
        not_designed.insert(3, {
            "what": "the shear key's steel",
            "why": "no held reference prints the key's loads or its "
                   "design section"})

    # 6. the description the drawing is made from
    def psi(MPa):
        return f"{round(MPa_to_psi(MPa), -1):,.0f}"
    notes = [
        f"Concrete f'c = {fc_MPa:g} MPa ({psi(fc_MPa)} psi). "
        f"Reinforcement fy = {fy_MPa:g} MPa ({fy_psi:,.0f} psi).",
        f"Clear cover: stem {c_stem_mm:g} mm ({c_stem:.3g} in.), base top "
        f"{c_top_mm:g} mm ({c_top:.3g} in.), base bottom {c_bot_mm:g} mm "
        f"({c_bot:.3g} in.).",
        "The main bars (A, D, E) are the outer layer of their face; the "
        "bars along the wall lie inside them.",
        f"Design basis: {BASIS if single_factor else BASIS_SERVICE_STRESS}"
        "; load cases 1 (normal operating)"
        + (" and 2 (normal operating + surcharge)" if has_live else "")
        + " of its Table E-2.",
        "Sizes, spacings and covers are the manual's inch values with the "
        "metric values it prints beside them (12 in. = 30 cm, 4 in. = 10 "
        "cm, 3 in. = 7.5 cm).",
        "Lap and straight development lengths, bar cut-offs"
        + (", the shear key's steel" if geom.has_shear_key else "")
        + " and the earthquake, maximum hydrostatic and construction load "
        "cases are not designed.",
    ]
    if development["computed"]:
        notes.insert(3, (
            f"Bars A end in a standard 90-degree hook in the base, hook "
            f"development length {development['ldh_mm']:.0f} mm "
            f"({development['ldh_in']:.1f} in.)."))
    if lap_length_mm is not None:
        notes.append(f"Lap length {lap_length_mm:g} mm, as supplied by the "
                     "engineer; not checked.")
    geometry = {
        "wall_height": geom.wall_height, "base_width": geom.base_width,
        "toe_length": geom.toe_length,
        "heel_length": round(geom.heel_length, 4),
        "stem_thickness_top": geom.stem_thickness_top,
        "stem_thickness_base": geom.stem_thickness_base,
        "base_thickness": geom.base_thickness,
        "stem_batter": geom.stem_batter,
        "embedment_depth": geom.Df,
        "backfill_slope_deg": geom.backfill_slope,
        "has_shear_key": bool(geom.has_shear_key),
        "key_depth": geom.key_depth if geom.has_shear_key else 0.0,
        "key_width": None, "key_distance_from_toe": None,
    }
    reinforcement = {
        "schema": SCHEMA,
        "title": "Cantilever retaining wall: reinforcement",
        "geometry": geometry,
        "materials": {"fc_MPa": float(fc_MPa), "fy_MPa": float(fy_MPa)},
        "bars": bars, "notes": notes,
        "basis": BASIS if single_factor else BASIS_SERVICE_STRESS,
    }

    # 7. the choices, each with its page
    lf = S.SINGLE_LOAD_FACTOR
    serviceability = (
        "the single load factors of Table 3-4: unfactored results x "
        f"{lf['usual']:g} (usual loads)"
        + (f" and x {lf['unusual']:g} (unusual loads)" if has_live else "")
        + (", and the service steel stress checked against Table 3-3 on "
           f"n = {modular_ratio:g}" if modular_ratio is not None else "")
        if single_factor else
        "the service steel stress checked against Table 3-3 (25 ksi "
        f"usual, 35 ksi unusual) on n = {modular_ratio:g}; the single "
        "load factors are not printed for this fy")
    choices = [
        choice("design basis", BASIS if single_factor
               else BASIS_SERVICE_STRESS, MODULE,
               "the manual the reference library holds for reinforced "
               "concrete walls; ACI 318 is not held"),
        choice("load cases", "Table E-2 case 1 (normal operating, usual)"
               + (" and case 2 (normal operating + surcharge, unusual)"
                  if has_live else "; case 2 needs a live-load surcharge "
                  "and there is none"), MODULE,
               f"{EM} Table E-2 (p. 93) and its notes (pp. 93-94); the "
               "serviceability column and the strength column are both "
               "worked and the larger moment and shear govern. Para. 3-1e "
               "(p. 13): usual and unusual loads 'are designed using "
               "serviceability requirements'"),
        choice("strength load factors",
               "(1.2 or 0.9) D + 1.5 EH + (1.35 or 1.0) EV + 1.0 Hs"
               + (" + 1.6 ES" if has_live else ""), MODULE,
               f"{EM} Table E-2 (p. 93); para. 3-3d (p. 22): 'EH, active "
               "pressure, 1.5', and 'Maximum and minimum load factors "
               "must be applied to provide the greatest effect' (each of "
               "the four D and EV pairs is worked on the whole wall); "
               "Table 3-2 (pp. 24-25): Hs as a companion load 1.0, ES 1.6. "
               "Para. 3-1f (pp. 13-14): the base pressure is found from "
               "the factored loads"),
        choice("serviceability", serviceability,
               MODULE if modular_ratio is None
               else how_given(modular_ratio, None, name="modular_ratio"),
               f"{EM} para. 3-4a(2) and Table 3-4 (p. 26): 'Service loads "
               "are multiplied by a single load factor ... These load "
               "factors apply only to design using reinforcing steel with "
               "yield strength of 60,000 psi (414 MPa)'; para. 3-1f (p. "
               "14): 'analysis can be performed with unfactored loads. "
               "The resulting moments, shears, and thrusts may then be "
               "multiplied by the single load factor'. Table 3-3 (p. 26) "
               "is the service stress route; the manual prints no rule "
               "for the modular ratio it needs (its examples take Es = "
               "29,000 ksi and Ec = 3,605 ksi at 4,000 psi, p. 75)",
               ["give modular_ratio (n = Es / Ec) for the service steel "
                "stress and the crack-control spacing on it"]
               if modular_ratio is None else []),
        choice("which loads are which",
               "D: the concrete. EV: the soil over the heel and the toe, "
               "and a dead-load surcharge on the heel. EH: the active "
               "thrust without the live load, a dead-load surcharge's "
               "thrust included. ES: the live-load surcharge, its thrust "
               "and its weight on the heel. Hs: the water's thrust and "
               "the uplift", MODULE,
               f"{EM} Table E-1 (p. 92) names the loads; para. 3-3d (p. "
               "22): 'Permanent loads ... consist of dead loads, vertical "
               "and lateral earth loads'; Table E-2 note b (p. 93): case "
               "2 is case 1 with 'a temporary surcharge'. That the "
               "permanent (dead-load) surcharge is an earth load and the "
               "live load the temporary surcharge ES is this module's "
               "reading: the manual prints no rule for a permanent "
               "surcharge"),
        choice("strength reduction factors",
               "phi = 0.90 for bending, 0.75 for shear", MODULE,
               S.PHI_BASIS),
        choice("beta1", f"{beta1_used:.3f}",
               how_given(beta1, None, name="beta1") if beta1 is not None
               else MODULE, S.BETA1_BASIS,
               [] if beta1 is not None else ["give beta1"]),
        choice("effective depth",
               "d = thickness - cover - half the main bar's diameter: the "
               "main bars are the outer layer of their face", MODULE,
               f"{EM} para. 2-3 (p. 4): 'Unless the plans and "
               "specifications specify that the primary reinforcement is "
               "to be on the outside, the width of secondary "
               "reinforcement should be subtracted'; the examples take "
               "the main bars outside, d = h - cover - 0.5 (bar diameter) "
               "(D-3 Step 4, p. 75; D-5 Step 3(d), p. 84). The "
               "reinforcement description's notes say so for the "
               "drawing"),
        choice("stem loads",
               "the active pressure's horizontal component and the water "
               "pressure on the vertical plane through the back of the "
               "stem's base, over the stem's height, with the surcharge "
               "from the back of the stem; the stem's weight and the "
               "thrust's vertical component left out (no axial load)",
               MODULE,
               "FHWA NHI-06-089 Sec. 10.5.3 (p. 10-41): 'the surcharge "
               "loads within the width of the base slab are considered "
               f"for the structural design of the wall stem'; {EM} D-5 "
               "Step 2 (p. 84): 'Weight of stem is neglected', and Step "
               "3(h) (p. 85) takes Eq. 5-1 'without axial terms'"),
        choice("base pressure for the toe and the heel",
               "the trapezoid, or the triangle when the resultant is "
               "outside the middle third; the water pressure under the "
               "base added to it", MODULE,
               "FHWA NHI-06-089 Fig. 10-17 (p. 10-39): 'use trapezoidal "
               "distribution with qmax and qmin for rocks and structural "
               "analysis'; FHWA GEC-6 Sec. 6.6 (pp. 121-122). The passive "
               "resistance in front of the wall is left out"),
        choice("toe and heel moment sections",
               "the toe at the front face of the stem, the heel at the "
               "back face, by statics on the slab: the base pressure up; "
               "the slab's weight and the soil, surcharge and water above "
               "it down; the thrust's vertical component at the end of "
               "the heel", MODULE,
               "not printed in a held reference: "
               f"{EM} para. 5-3 (pp. 35-36) gives the sections for shear "
               "only, and p. 87 sends the base slab's shear and moment to "
               "EM 1110-2-2502, which is not held"),
        choice("shear strength",
               "Vc = 2 sqrt(f'c) b d, f'c in psi (the US form of Eq. 5-1, "
               "no axial load), on the values converted to US units; the "
               "printed SI form, 0.17 sqrt(f'c) b d, is about 2 % higher "
               "and is reported beside it (phi_Vc_SI_form)", MODULE,
               f"{EM} Eq. 5-1 and para. 5-2 (p. 35); the examples use the "
               "US form (D-5 Step 3(h), p. 85). 2 sqrt(psi) is 0.166 "
               "sqrt(MPa); the print rounds it to 0.17"),
        choice("minimum flexural steel",
               "the shrinkage and temperature steel of the face (0.003 of "
               "the gross area, half in each face, not less than No. 4 at "
               "12 in.)", MODULE,
               f"{EM} para. 2-9g (p. 8): 'the primary reinforcement must "
               "not be less than that required for shrinkage and "
               "temperature'; para. 2-9a (pp. 6-7). The manual prints no "
               "other minimum"),
        choice("temperature and shrinkage steel",
               f"along the wall {ratio_along:g} of the gross area for "
               f"joints {joint_spacing:g} m ({joint_ft:.0f} ft) apart, "
               "half in each face; up the stem and across the base 0.003; "
               "each face not less than No. 4 at 12 in. nor more than No. "
               "9 at 12 in.; a tapered stem on its thickness at the base",
               how_given(joint_spacing, None, name="joint_spacing"),
               f"{EM} para. 2-9a, 2-9b and Table 2-3 (pp. 6-7); para. "
               "2-9d (p. 7): 'Shrinkage and temperature reinforcement in "
               "the transverse (shorter) direction must be according to "
               "paragraph 2-9a'. The stem's thickness at its base for the "
               "whole stem is this module's reading, the larger steel"),
        choice("bar choice",
               "; ".join(f"{n}: {el['bar']['chosen_by']}"
                         for n, el in elements.items()), MODULE,
               f"{EM} D-3 Step 3 (p. 75): 'Use No. 6 bars at 12-in. on "
               "center'; D-5 Step 3(j)-(k) (p. 85): No. 11 at 6 in. where "
               "12 in. does not give the steel. Bars No. 4 to No. 11 "
               f"(para. 2-8b, p. 6); areas: {S.BARS_BASIS}"),
        choice("crack control",
               "usual loads: met by the 12 in. spacing when the "
               "serviceability design is met"
               + ("; unusual loads: s_max = 15 (40,000 / fs) - 2.5 cc on "
                  + ("the actual service stress" if modular_ratio
                     is not None else "the 35 ksi limit of Table 3-3 (no "
                     "modular ratio for the actual stress), which is 10.9 "
                     "in. at 2.5 in. of cover or more, so 12 in. is "
                     "passed over for 6 in.")
                  if has_live else ""), MODULE,
               f"{EM} para. 2-6b (p. 5): cc 'the lesser of 2.5 inches or "
               "the actual clear cover'; usual loads 'met by 12 in. (30 "
               "cm) maximum spacing when the serviceability requirements "
               "specified in Chapter 3 are met'; unusual loads on 'either "
               "the reinforcement service stress limit in paragraph 3-4 "
               "or the actual service stress'. The spacing formula is "
               "ACI 318-19's as the manual prints it in D-3 Step 5(b) "
               "(p. 76) and D-5 Step 5(b) (p. 86)",
               ["give modular_ratio for the actual service stress, or "
                "the bar spacing"] if has_live and modular_ratio is None
               else []),
        choice("clear cover",
               f"stem {c_stem_mm:g} mm, base top {c_top_mm:g} mm, base "
               f"bottom {c_bot_mm:g} mm",
               "given" if (c_stem_given or c_top_given or c_bot_given)
               else DEFAULT,
               f"{EM} Table 2-1 (p. 5): 4 in. (10 cm) for unformed "
               "surfaces in contact with the foundation; formed and "
               "screeded surfaces 4 in. for a section 24 in. (61 cm) "
               "thick or more, 3 in. (7.5 cm) over 12 in. (30 cm); for 12 "
               "in. or less 'According to ACI 318-19', so the engineer "
               "gives it. The stem's row is by its thickness at the "
               "base"),
        choice("stem bars' anchorage",
               (f"a standard 90-degree hook in the base, ldh = "
                f"{development['ldh_mm']:.0f} mm "
                f"({development['ldh_in']:.1f} in.), "
                f"{development['available_mm']:.0f} mm available"
                if development["computed"] else
                "a standard hook in the base; its development length is "
                "not computed"), MODULE,
               f"{EM} para. 2-10a and Fig. 2-1 (p. 9): 'Full moment "
               "transfer is attained by turning the hooks'; the "
               "development length of a standard hook as D-5 Step 6 "
               "prints it (pp. 86-87), with its factors: uncoated bars, "
               "no confining reinforcement, side cover of 6 bar diameters "
               "or more, normal weight concrete"
               + ("" if development["computed"]
                  else ". " + development["why_not"])),
    ]

    load_case_table = [
        {"case": "1", "description": "Normal operating",
         "category": "usual", "designed": True,
         "serviceability": "2.2 (D + EH + EV + Hs)" if single_factor
         else "1.0 (D + EH + EV + Hs), service stress",
         "strength": "(1.2 or 0.9) D + 1.5 EH + (1.35 or 1.0) EV + 1.0 Hs"},
        {"case": "2", "description": "Normal operating + surcharge",
         "category": "unusual", "designed": has_live,
         "serviceability": "1.6 (D + EH + EV + ES + Hs)" if single_factor
         else "1.0 (D + EH + EV + ES + Hs), service stress",
         "strength": "(1.2 or 0.9) D + 1.5 EH + (1.35 or 1.0) EV + 1.6 ES "
                     "+ 1.0 Hs",
         **({} if has_live else {"why_not": "no live-load surcharge"})},
        {"case": "3A, 3B", "description": "Earthquake (OBE, MDE)",
         "category": "unusual, extreme", "designed": False},
        {"case": "4", "description": "Maximum hydrostatic loading",
         "category": "by its return period", "designed": False},
        {"case": "5", "description": "Construction",
         "category": "unusual", "designed": False},
    ]
    warnings = sorted({w for el in elements.values()
                       for w in el["warnings"]})
    for name, el in elements.items():
        if "reverse_moment" in el:
            warnings.append(f"{name}: " + el["reverse_moment"]["note"])

    return CantileverWallStructuralResult(
        stability=stability,
        geometry={**geometry, "stem_height": round(geom.stem_height, 4)},
        materials={
            "fc_MPa": float(fc_MPa), "fc_psi": round(fc_psi, 0),
            "fy_MPa": float(fy_MPa), "fy_psi": round(fy_psi, 0),
            "beta1": round(beta1_used, 4), "modular_ratio": modular_ratio,
            "max_aggregate_mm": max_aggregate_mm,
            "joint_spacing_m": joint_spacing,
            "lap_length_mm": lap_length_mm,
            "phi_flexure": S.PHI_FLEXURE, "phi_shear": S.PHI_SHEAR},
        load_cases=load_case_table, elements=elements,
        secondary=secondary, development=development,
        checks=wall_checks, reinforcement=reinforcement,
        choices=list(stability.choices or []) + choices,
        structural_choices=choices, loads=loads,
        not_designed=not_designed, warnings=warnings,
        basis=BASIS if single_factor else BASIS_SERVICE_STRESS)


__all__ = ["analyze_cantilever_wall_structure",
           "CantileverWallStructuralResult", "SCHEMA", "BASIS"]
