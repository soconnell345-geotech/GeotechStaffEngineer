"""
Main bearing capacity calculation engine.

Computes ultimate and allowable bearing capacity using the general
bearing capacity equation (Vesic/Meyerhof/Hansen forms), or Terzaghi's
(1943) equation with his own factors, for one-layer and two-layer soil
systems.

References:
    FHWA GEC-6 (FHWA-SA-02-054), Chapter 5
    FHWA-SA-94-034 (CBEAR User's Guide)
    NAVFAC DM-7.01 Ch. 4 / Bowles 5th ed. Sec. 4-7 — Two-layer load-spread method
    Meyerhof & Hanna (1978) — Two-layer punching shear (recognized alternative)
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

from bearing_capacity.footing import Footing
from bearing_capacity.soil_profile import BearingSoilProfile, SoilLayer
from bearing_capacity.factors import (
    all_N_factors,
    shape_factors,
    depth_factors,
    depth_parameter_k,
    ufc_table_5_8_printed_dc,
    DEPTH_FACTOR_METHODS,
    DEPTH_FACTOR_BASIS, DEPTH_K_BASIS,
    inclination_factors,
    vesic_inclination_needs_V,
    _meyerhof_inclination_factors,
    base_inclination_factors,
    ground_inclination_factors,
    bearing_capacity_Nc,
    bearing_capacity_Nq,
    bearing_capacity_Ngamma,
    TERZAGHI_NGAMMA_SOURCES,
    terzaghi_Nc,
    terzaghi_Nq,
    terzaghi_Ngamma,
    terzaghi_shape_factors,
    terzaghi_local_shear,
)
from bearing_capacity.results import BearingCapacityResult

#: factor_method values: the Vesic and Meyerhof correction-factor sets, or
#: Terzaghi (1943) -- his own Nc, Nq, Ngamma and shape coefficients.
FACTOR_METHODS = ("vesic", "meyerhof", "terzaghi")
#: N-gamma choices that go with the Vesic/Meyerhof factor sets.
GENERAL_NGAMMA_METHODS = ("vesic", "meyerhof", "hansen")
#: Each method's own N-gamma, the default for ``ngamma_method`` (GM-188):
#: Vesic's factors with the FHWA/AASHTO N-gamma, Meyerhof's with his own,
#: Terzaghi's with Kumbhojkar's tabulation of his chart.
OWN_NGAMMA = {"vesic": "vesic", "meyerhof": "meyerhof",
              "terzaghi": "kumbhojkar"}

#: Where each method and N-gamma is printed (GM-188).
METHOD_BASIS = {
    "vesic": ("GEC-6 Sec. 5.2.2 (p. 53): Eq. 5-7, N-gamma Eq. 5-8, "
              "Table 5-1 (p. 54); shape factors Sec. 5.2.3.1, Table 5-2 "
              "(p. 56)"),
    "meyerhof": ("UFC 3-220-20 Sec. 5-3.3.2, Table 5-6 (p. 294): Meyerhof "
                 "(1963) bearing, shape, depth and inclination factors; "
                 "Table 5-3 (p. 291): any conditions except a tilted base, "
                 "shear strength dominated by friction"),
    "terzaghi": ("UFC 3-220-20 Table 5-2 (p. 289) and Table 5-5 (p. 294): "
                 "Terzaghi (1943); Table 5-3 (p. 291): undrained soils, "
                 "Df/B <= 1, no inclined load, slope or tilted base"),
}
NGAMMA_BASIS = {
    "vesic": ("2 (Nq + 1) tan phi, Caquot and Kerisel (1948), GEC-6 "
              "Eq. 5-8 (p. 53): 'currently used by FHWA and AASHTO', "
              "Table 5-1 (p. 54)"),
    "meyerhof": ("(Nq - 1) tan(1.4 phi), UFC 3-220-20 Table 5-6 (p. 294), "
                 "Table 5-2 (p. 289)"),
    "hansen": ("1.5 (Nq - 1) tan phi, Brinch Hansen (1970), UFC 3-220-20 "
               "Table 5-2 (p. 289)"),
}


@dataclass
class BearingCapacityAnalysis:
    """Bearing capacity analysis for a shallow foundation.

    Parameters
    ----------
    footing : Footing
        Footing geometry definition.
    soil : BearingSoilProfile
        Soil profile (1 or 2 layers).
    load_inclination : float, optional
        Angle of applied load from vertical (degrees). Default 0.
    ground_slope : float, optional
        Ground surface slope angle (degrees). Default 0.
    vertical_load : float, optional
        Total vertical load V on the footing (kN; kN/m for a strip). Vesic's
        load inclination factors take H/(V + A' c cot phi), so with
        cohesion (or phi = 0) an inclined load needs it, and the analysis
        raises without it (GM-203: it fell back to Meyerhof's factors with
        a warning). With c = 0, H/V = tan(load_inclination) and V cancels.
    factor_of_safety : float, optional
        Required factor of safety. Default 3.0.
    ngamma_method : str, optional
        Method for Ngamma: "vesic", "meyerhof", or "hansen"; with
        ``factor_method="terzaghi"``, "kumbhojkar" or "coduto" (see
        ``factors.TERZAGHI_NGAMMA_SOURCES``). Default: the factor method's
        own (:data:`OWN_NGAMMA`; GM-188: Meyerhof took Vesic's before).
        Another one makes a mixed method, and the result's ``choices``
        say so.
    factor_method : str, optional
        Method for shape/depth/inclination factors: "vesic" (default) or
        "meyerhof"; or "terzaghi" for Terzaghi (1943): his own Nc, Nq,
        Ngamma and shape coefficients (1.3 c Nc, 0.4 / 0.3 gamma B Ngamma),
        no depth or inclination factors. Terzaghi is not for inclined loads,
        sloping ground or tilted bases (UFC 3-220-20 Table 5-3) and raises
        on them.
    failure_mode : str, optional
        "general" (default) or, with ``factor_method="terzaghi"``, "local":
        Terzaghi's local-shear reduction c* = 2/3 c, tan(phi*) = 2/3 tan(phi).
    depth_factor_method : str, optional
        How the depth factors are worked out (GM-197;
        ``factors.DEPTH_FACTOR_BASIS``): "hansen" (default), Brinch
        Hansen's depth parameter k in the method's factors; "gec6"
        (``factor_method="vesic"`` only), GEC-6's d_q of Table 5-4 on the
        surcharge term alone, 1.0 below Df/B = 1; "none", no depth factors.

    Examples
    --------
    Square footing on sand:
    >>> footing = Footing(width=2.0, depth=1.5, shape="square")
    >>> soil = BearingSoilProfile(
    ...     layer1=SoilLayer(friction_angle=30, unit_weight=18.0)
    ... )
    >>> analysis = BearingCapacityAnalysis(footing=footing, soil=soil)
    >>> result = analysis.compute()
    >>> print(f"qult = {result.q_ultimate:.0f} kPa")
    """
    footing: Footing = None
    soil: BearingSoilProfile = None
    load_inclination: float = 0.0
    ground_slope: float = 0.0
    vertical_load: Optional[float] = None
    factor_of_safety: float = 3.0
    ngamma_method: Optional[str] = None
    factor_method: str = "vesic"
    failure_mode: str = "general"
    depth_factor_method: str = "hansen"

    def __post_init__(self):
        if self.footing is None:
            raise ValueError("Footing must be provided")
        if self.soil is None:
            raise ValueError("Soil profile must be provided")
        if self.factor_of_safety <= 0:
            raise ValueError(f"Factor of safety must be positive, got {self.factor_of_safety}")
        self.factor_method = str(self.factor_method).strip().lower()
        if self.factor_method not in FACTOR_METHODS:
            raise ValueError(
                f"factor_method must be one of {list(FACTOR_METHODS)}, got "
                f"'{self.factor_method}'")
        terzaghi = self.factor_method == "terzaghi"
        if self.ngamma_method is None:
            # the method's own N-gamma (GM-188: Meyerhof took Vesic's)
            self.ngamma_method = OWN_NGAMMA[self.factor_method]
        self.ngamma_method = str(self.ngamma_method).strip().lower()
        if terzaghi and self.ngamma_method not in TERZAGHI_NGAMMA_SOURCES:
            raise ValueError(
                f"factor_method='terzaghi' takes Terzaghi's own N-gamma: "
                f"ngamma_method {list(TERZAGHI_NGAMMA_SOURCES)} (default "
                f"'kumbhojkar'), got '{self.ngamma_method}'")
        if not terzaghi and self.ngamma_method in TERZAGHI_NGAMMA_SOURCES:
            raise ValueError(
                f"ngamma_method '{self.ngamma_method}' is a Terzaghi (1943) "
                f"N-gamma; use it with factor_method='terzaghi', or pick one "
                f"of {list(GENERAL_NGAMMA_METHODS)}")
        self.failure_mode = str(self.failure_mode).strip().lower()
        if self.failure_mode not in ("general", "local"):
            raise ValueError(
                f"failure_mode must be 'general' or 'local', got "
                f"'{self.failure_mode}'")
        if self.failure_mode == "local" and not terzaghi:
            raise ValueError(
                "failure_mode='local' is Terzaghi's local-shear reduction "
                "(c* = 2/3 c, tan phi* = 2/3 tan phi); use it with "
                "factor_method='terzaghi'.")
        self.depth_factor_method = str(
            self.depth_factor_method).strip().lower()
        if self.depth_factor_method not in DEPTH_FACTOR_METHODS:
            raise ValueError(
                f"depth_factor_method must be one of "
                f"{list(DEPTH_FACTOR_METHODS)}, got "
                f"'{self.depth_factor_method}'")
        if self.depth_factor_method == "gec6" \
                and self.factor_method != "vesic":
            raise ValueError(
                "depth_factor_method='gec6' is GEC-6's depth factor (Eq. "
                "5-14, Table 5-4), which goes with GEC-6's own factors: "
                "factor_method='vesic'.")
        if terzaghi:
            bad = [name for name, v in (
                ("load_inclination", self.load_inclination),
                ("ground_slope", self.ground_slope),
                ("base_tilt", self.footing.base_tilt)) if v]
            if bad:
                raise ValueError(
                    f"Terzaghi (1943) has no factors for {', '.join(bad)} "
                    "(UFC 3-220-20 Table 5-3: do not use it for inclined "
                    "loads, slopes or tilted bases); use factor_method="
                    "'vesic' or 'meyerhof'.")

    def compute(self) -> BearingCapacityResult:
        """Run the bearing capacity analysis.

        Returns
        -------
        BearingCapacityResult
            Complete results including qult, qallowable, and all factors.
        """
        if self.soil.is_two_layer:
            result = self._compute_two_layer()
        else:
            result = self._compute_single_layer()
        # the area the allowable loads are worked on (GM-170)
        result.is_strip = self.footing.shape == "strip"
        result.area_m2 = (self.footing.B_eff if result.is_strip
                          else self.footing.A_eff)
        f = self.footing
        result.area_full_m2 = (
            f.width if result.is_strip else math.pi / 4.0 * f.width ** 2
            if f.shape == "circular" else f.width * f.length)
        result.choices = self._choices(result)
        return result

    def _choices(self, r: BearingCapacityResult) -> list:
        """What the analysis chose (GM-188, ``geotech_common.choices``):
        the method, its N-gamma, the shear mode and the allowable pressure's
        basis; for an embedded footing the depth factors (GM-197); under an
        inclined load Vesic's inclination factors (GM-203) or Meyerhof's
        shape factors."""
        from geotech_common.choices import MODULE, choice, how_given
        name = {"vesic": "Vesic (1973)", "meyerhof": "Meyerhof (1963)",
                "terzaghi": "Terzaghi (1943)"}
        fm = self.factor_method
        out = [choice(
            "bearing capacity method",
            f"{name[fm]} factors (factor_method '{fm}')",
            how_given(fm, "vesic", name="factor_method"), METHOD_BASIS[fm],
            [f"{name[m]} (factor_method '{m}'): {METHOD_BASIS[m]}"
             for m in FACTOR_METHODS if m != fm])]

        ng, own = self.ngamma_method, OWN_NGAMMA[fm]
        how = how_given(ng, own, name="ngamma_method")
        if fm == "terzaghi":
            out.append(choice(
                "N-gamma",
                f"{r.Ngamma:.2f}, {TERZAGHI_NGAMMA_SOURCES[ng]} "
                f"(ngamma_method '{ng}')", how,
                "'Terzaghi (1943) only provided N-gamma values for 0, 34, "
                "and 48 degrees' (UFC 3-220-20 Table 5-2 note, p. 289), so "
                "every Terzaghi N-gamma is someone's evaluation; the table "
                "prints Coduto et al.'s (2016)",
                [f"{v:.2f}, {TERZAGHI_NGAMMA_SOURCES[k]} (ngamma_method "
                 f"'{k}')" for k, v in r.Ngamma_alternatives.items()]))
        elif self.soil.layer1.friction_angle == 0:
            # every N-gamma formula is 0 at phi = 0: nothing to choose
            # (GM-230)
            out.append(choice(
                "N-gamma", f"0 at phi = 0 (ngamma_method '{ng}')", how,
                "every N-gamma formula gives 0 at phi = 0, so the unit "
                "weight term drops out and which formula does not matter"))
        else:
            phi = self.soil.layer1.friction_angle
            out.append(choice(
                "N-gamma",
                f"{r.Ngamma:.2f} = {NGAMMA_BASIS[ng].split(',')[0]} "
                f"(ngamma_method '{ng}')", how,
                NGAMMA_BASIS[ng] + (
                    ", the method's own" if ng == own else
                    f"; a mixed method: {name[fm]} factors with another "
                    "N-gamma"),
                [f"{bearing_capacity_Ngamma(phi, m):.2f} = "
                 f"{NGAMMA_BASIS[m]} (ngamma_method '{m}')"
                 + (", the method's own" if m == own else "")
                 for m in GENERAL_NGAMMA_METHODS if m != ng]))

        local = self.failure_mode == "local"
        out.append(choice(
            "shear mode",
            ("local shear: c* = 2/3 c, phi* = atan(2/3 tan phi) "
             "(failure_mode 'local')") if local else
            "general shear (failure_mode 'general')",
            # failure_mode is Terzaghi's input; the other methods assume
            # general shear
            how_given(self.failure_mode, "general", name="failure_mode")
            if fm == "terzaghi" else MODULE,
            "UFC 3-220-20 Sec. 5-3.1.3 (p. 291): the bearing capacity "
            "theories assume general shear, appropriate only for dense "
            "and/or stiff soils; for local or punching shear settlement "
            "almost always controls; Table 5-4 (p. 291)",
            ["general shear (failure_mode 'general')" if local else
             "local shear, Terzaghi's c* = 0.67 c', phi* = atan(0.67 tan "
             "phi'), UFC 3-220-20 Table 5-4 (factor_method 'terzaghi', "
             "failure_mode 'local')"]))

        out.append(choice(
            "allowable bearing pressure",
            f"gross: q_all = q_ult / FS = {r.q_allowable:,.1f} kPa "
            f"(q_allowable); the net beside it, q_all,net = "
            f"{r.q_allowable_net:,.1f} kPa (q_allowable_net)",
            MODULE,
            "UFC 3-220-20 Sec. 5-2.4 Eq. 5-13 and 5-14 (p. 279); GEC-6 Sec. "
            "5.2.2 (p. 53) recommends the gross, with the footing and "
            "backfill weight in the applied load; the allowable structural "
            "load is on the net (allowable_load)",
            ["net: q_all,net = q_ult / FS - q (UFC 3-220-20 Eq. 5-14), "
             "set against the structural load over the area (Eq. 5-3)"]))

        if fm != "terzaghi" and self.footing.depth > 0:
            out.append(self._depth_choice(r))
        if fm == "vesic" and self.load_inclination:
            out.append(self._inclination_choice(r))
        if fm == "meyerhof" and self.load_inclination:
            out.append(choice(
                "shape factors under inclined load", "sc = sq = sgamma = 1",
                MODULE,
                "UFC 3-220-20 Sec. 5-3.3.2 and Table 5-6 note A (p. 294): "
                "'Do not combine factors for shape and inclined load'"))
        return out

    def _no_cohesion(self) -> bool:
        """Whether no layer below the base has cohesion, so the c N_c term
        and its factors drop out."""
        layers = [self.soil.layer1] + ([self.soil.layer2]
                                       if self.soil.is_two_layer else [])
        return all(not l.cohesion for l in layers)

    def _depth_choice(self, r: BearingCapacityResult) -> dict:
        """The depth factors, how they were worked out and what the other
        depth_factor_method values give (GM-197)."""
        from geotech_common.choices import choice, how_given
        fm, dm = self.factor_method, self.depth_factor_method
        phi = self.soil.layer1.friction_angle
        Df, B = self.footing.depth, self.footing.B_for_factors
        df_b = Df / B

        def factors(d):
            return ", ".join(f"{v:.3f}" for v in d)

        if dm == "none":
            chosen = "none: d_c = d_q = d_gamma = 1"
        elif dm == "gec6":
            chosen = (f"GEC-6: d_q = {r.dq:.3f} on the surcharge term "
                      f"alone at Df/B = {df_b:.2f}"
                      + (" (1.0 below Df/B = 1)" if df_b < 1.0 else
                         " (Table 5-4)"))
        else:
            k = depth_parameter_k(Df, B, fm)
            form = "Df/B" if k == df_b else "arctan(Df/B)"
            chosen = (f"k = {form} = {k:.3f} at Df/B = {df_b:.2f}: d_c, "
                      f"d_q, d_gamma = {r.dc:.3f}, {r.dq:.3f}, "
                      f"{r.dgamma:.3f}")
        chosen += f" (depth_factor_method '{dm}')"
        if fm == "meyerhof" and dm == "hansen":
            basis = ("Meyerhof's depth factors, UFC 3-220-20 Table 5-6 (p. "
                     "294), with k = Df/B as that table prints it up to "
                     "Df/B = 1 and Brinch Hansen's k = arctan(Df/B) beyond "
                     "it (Tables 5-7 and 5-8 note B), the smaller factors")
        elif dm == "hansen" and phi == 0:
            # Table 5-7's d_c at phi = 0, added as Eq. 5-21 adds it
            # (GM-281); the note C formula and the printed Table 5-8 d_c do
            # not bear (GM-230)
            basis = (DEPTH_K_BASIS + "; at phi = 0 Table 5-7 (p. 295) "
                     "prints d_c = 0.4 k, added in Eq. 5-21's 1 + s_c + "
                     "d_c - i_c - b_c - g_c. The cohesion term adds it the "
                     "same way, c N_c (s_c i_c b_c g_c + 0.4 k), the other "
                     "factors multiplied as GEC-6 Eq. 5-14 (p. 63) and EN "
                     "1997-1 D.1 (p. 158) print them; with the shape factor "
                     "alone that is Eq. 5-21, as UFC 3-220-20 Example B-3 "
                     "works it (Table B-17, p. 622). d_q = 1, as Table 5-7 "
                     "puts no correction on the q term, and d_gamma = 1")
        elif dm == "hansen" and self._no_cohesion():
            basis = (DEPTH_K_BASIS + "; d_q = 1 + 2 k tan phi (1 - sin "
                     "phi)^2, Table 5-8; d_gamma = 1; d_c does not enter "
                     "with c = 0")
        else:
            basis = DEPTH_FACTOR_BASIS[dm]
        alternatives = []
        for other in DEPTH_FACTOR_METHODS:
            if other == dm or (other == "gec6" and fm != "vesic"):
                continue
            d = depth_factors(phi, Df, B, fm, other)
            alternatives.append(
                f"d_c, d_q, d_gamma = {factors(d)} (depth_factor_method "
                f"'{other}'): {DEPTH_FACTOR_BASIS[other]}")
        if fm == "meyerhof" and df_b > 1.0:
            alternatives.append(
                f"k = Df/B = {df_b:.2f}, as UFC 3-220-20 Table 5-6 (p. 294) "
                "prints for Meyerhof at any depth (larger depth factors; "
                "not an option of this tool)")
        if (fm == "vesic" and dm == "hansen" and phi > 0
                and not self._no_cohesion()):
            # GM-214: the table's printed d_c is half its note C formula's
            alternatives.append(
                f"d_c = {ufc_table_5_8_printed_dc(phi, Df, B):.3f}, as UFC "
                "3-220-20 Table 5-8 (p. 296) prints it, 1 + k (1 - sin "
                "phi)^2 N_q/N_c: half of what the correspondence formula "
                "its note C names gives (not an option of this tool)")
        return choice(
            "depth factors", chosen,
            how_given(dm, "hansen", name="depth_factor_method"), basis,
            alternatives)

    def _inclination_choice(self, r: BearingCapacityResult) -> dict:
        """Vesic's load inclination factors, and what they took (GM-203:
        with no V they had fallen back to Meyerhof's with a warning)."""
        from geotech_common.choices import choice, how_given
        layer = self.soil.layer1
        beta = self.load_inclination
        B, L = self.footing.B_for_factors, self.footing.L_for_factors
        m = (2.0 + B / L) / (1.0 + B / L)
        i = f"i_c, i_q, i_gamma = {r.ic:.3f}, {r.iq:.3f}, {r.igamma:.3f}"
        if not vesic_inclination_needs_V(layer.friction_angle,
                                         layer.cohesion):
            chosen = (f"Vesic (1975), H along B (m = {m:.2f}): with c = 0, "
                      f"H/V = tan({beta:g} deg) and V cancels; {i}")
        else:
            per = " per metre" if self.footing.shape == "strip" else ""
            chosen = (f"Vesic (1975), H along B (m = {m:.2f}): H = V "
                      f"tan({beta:g} deg) = "
                      f"{self.vertical_load * math.tan(math.radians(beta)):.1f}"
                      f" kN{per} with V = {self.vertical_load:g} kN{per}, "
                      f"A' = {self._inclination_area():.2f} m2{per}; {i}")
        try:
            mic, miq, mig = _meyerhof_inclination_factors(
                layer.friction_angle, beta)
            meyerhof = (f"i_c = i_q = {mic:.3f}, i_gamma = {mig:.3f} "
                        "(factor_method 'meyerhof', with its shape factors "
                        "1 under an inclined load)")
        except ValueError:
            # beta >= phi: the table takes theta < phi' (GM-214)
            meyerhof = (f"none at {beta:g} deg: the table takes theta = "
                        f"atan(H/V) < phi' = {layer.friction_angle:g} deg")
        alternatives = [
            "Meyerhof's angle-only factors, UFC 3-220-20 Table 5-6 (p. "
            "294): " + meyerhof,
            "none, with sliding checked against the horizontal load: GEC-6 "
            "Sec. 5.2.3.7 (p. 62) (load_inclination 0)"]
        if layer.friction_angle == 0:
            # Vesic's phi = 0 form is not the one EN 1997-1 prints (GM-203)
            basis = ("Vesic (1975) for phi = 0: i_c = 1 - m H/(A' c N_c), "
                     "m = (2 + B'/L')/(1 + B'/L') for H along B; GEC-6 Sec. "
                     "5.2.3.7 (p. 62) omits load inclination factors where "
                     "sliding is checked separately")
            H = self.vertical_load * math.tan(math.radians(beta))
            Ac = self._inclination_area() * layer.cohesion
            alternatives.insert(0, (
                "i_c = [1 + sqrt(1 - H/(A' c))]/2 = "
                + (f"{0.5 * (1.0 + math.sqrt(1.0 - H / Ac)):.3f}"
                   if H <= Ac else "undefined (H > A' c)")
                + ", EN 1997-1:2004 D.3 (p. 158); UFC 3-220-20 Table 5-7 "
                "(p. 295) prints the same in its additive form (not an "
                "option of this tool)"))
        else:
            basis = ("Vesic (1975) as EN 1997-1:2004 Annex D.4 (p. 159) "
                     "prints it: i_q = [1 - H/(V + A' c cot phi)]^m, "
                     "i_gamma = [...]^(m+1), i_c = i_q - (1 - i_q)/(N_c tan "
                     "phi), m = (2 + B'/L')/(1 + B'/L') for H along B; "
                     "GEC-6 Sec. 5.2.3.7 (p. 62) omits load inclination "
                     "factors where sliding is checked separately")
        return choice(
            "load inclination factors", chosen,
            how_given(self.factor_method, "vesic", name="factor_method"),
            basis, alternatives)

    def _inclination_area(self) -> float:
        """A' in Vesic's adhesion term: B' per metre for a strip (whose L is
        taken as infinite), else the effective area (GM-203: a strip took
        B x 1e6 m)."""
        f = self.footing
        return f.B_for_factors if f.shape == "strip" else f.A_eff

    def _compute_single_layer(self) -> BearingCapacityResult:
        """Compute bearing capacity for a single-layer soil profile.

        Uses the general bearing capacity equation:
            qult = c*Nc*sc*dc*ic*bc*gc
                 + q*Nq*sq*dq*iq*bq*gq
                 + 0.5*gamma*B'*Ng*sg*dg*ig*bg*gg
        except that at phi = 0 Vesic's cohesion term adds Brinch Hansen's
        d_c, as UFC 3-220-20 Eq. 5-21 (p. 295) adds its Table 5-7 d_c = 0.4 k:
            c*Nc*(sc*ic*bc*gc + dc - 1)
        the other factors multiplied as GEC-6 Eq. 5-14 (p. 63) and EN 1997-1
        D.1 (p. 158) print them (GM-281).
        """
        if self.factor_method == "terzaghi":
            return self._compute_terzaghi()
        layer = self.soil.layer1
        phi = layer.friction_angle
        c = layer.cohesion

        # Effective footing dimensions
        B = self.footing.B_for_factors
        L = self.footing.L_for_factors
        Df = self.footing.depth

        # Bearing capacity factors
        Nc, Nq, Ng = all_N_factors(phi, self.ngamma_method)

        # Shape factors; Meyerhof's are 1 under an inclined load (UFC
        # 3-220-20 Table 5-6 note A: do not combine shape and inclination)
        sc, sq, sg = shape_factors(phi, B, L, self.factor_method)
        if self.factor_method == "meyerhof" and self.load_inclination:
            sc = sq = sg = 1.0

        # Depth factors
        dc, dq, dg = depth_factors(phi, Df, B, self.factor_method,
                                   self.depth_factor_method)

        # Inclination factors
        ic, iq, ig = inclination_factors(
            phi, self.load_inclination, c, B, L,
            self.vertical_load, self.factor_method,
            A=self._inclination_area()
        )

        # Base inclination factors
        bc, bq, bg = base_inclination_factors(phi, self.footing.base_tilt)

        # Ground inclination factors
        gc, gq, gg = ground_inclination_factors(phi, self.ground_slope)

        # Overburden pressure at footing base
        q = self.soil.overburden_pressure(Df)

        # Effective unit weight below footing
        gamma = self.soil.gamma_below_footing(Df, B)

        # General bearing capacity equation
        # Term 1: cohesion term. At phi = 0 Vesic's d_c is Brinch Hansen's
        # 0.4 k, which Eq. 5-21 adds (UFC 3-220-20 p. 295; GM-281: it was
        # multiplied). Meyerhof's d_c multiplies (Table 5-6, Eq. 5-18)
        dc_added = phi == 0 and self.factor_method == "vesic"
        if dc_added:
            term_c = c * Nc * (sc * ic * bc * gc + dc - 1.0)
        else:
            term_c = c * Nc * sc * dc * ic * bc * gc
        # Term 2: overburden term
        term_q = q * Nq * sq * dq * iq * bq * gq
        # Term 3: self-weight term
        term_g = 0.5 * gamma * B * Ng * sg * dg * ig * bg * gg

        q_ultimate = term_c + term_q + term_g
        q_allowable = q_ultimate / self.factor_of_safety

        # Net ultimate bearing capacity (subtract overburden)
        q_net = q_ultimate - q

        return BearingCapacityResult(
            q_ultimate=q_ultimate,
            q_allowable=q_allowable,
            q_net=q_net,
            factor_of_safety=self.factor_of_safety,
            Nc=Nc, Nq=Nq, Ngamma=Ng,
            sc=sc, sq=sq, sgamma=sg,
            dc=dc, dq=dq, dgamma=dg,
            ic=ic, iq=iq, igamma=ig,
            bc=bc, bq=bq, bgamma=bg,
            gc=gc, gq=gq, ggamma=gg,
            q_overburden=q,
            gamma_eff=gamma,
            B_eff=B,
            L_eff=L,
            term_cohesion=term_c,
            term_overburden=term_q,
            term_selfweight=term_g,
            ngamma_method=self.ngamma_method,
            factor_method=self.factor_method,
            c_term_adds_dc=dc_added,
        )

    def _compute_terzaghi(self) -> BearingCapacityResult:
        """Terzaghi (1943), one layer.

            qult = c Nc sc + q Nq + 0.5 gamma B Ngamma sgamma

        with Terzaghi's Nc, Nq, the chosen Ngamma source and his shape
        coefficients; every other factor is 1 (UFC 3-220-20 Sec 5-3.3.1).
        For local shear, c and phi are first reduced to c* = 2/3 c and
        phi* = atan(2/3 tan phi). The other Ngamma source is reported beside
        the one used.
        """
        layer = self.soil.layer1
        c, phi = layer.cohesion, layer.friction_angle
        notes = []
        c_used, phi_used = c, phi
        if self.failure_mode == "local":
            c_used, phi_used = terzaghi_local_shear(c, phi)
            notes.append(
                f"Local shear (Terzaghi 1943; UFC 3-220-20 Table 5-4): "
                f"c* = 2/3 c = {c_used:.2f} kPa, phi* = atan(2/3 tan phi) = "
                f"{phi_used:.2f} deg. UFC 3-220-20 notes it is probably too "
                "conservative for sands.")

        B = self.footing.B_for_factors
        L = self.footing.L_for_factors
        Df = self.footing.depth

        Nc = terzaghi_Nc(phi_used)
        Nq = terzaghi_Nq(phi_used)
        Ng = terzaghi_Ngamma(phi_used, self.ngamma_method)
        alternatives = {
            src: terzaghi_Ngamma(phi_used, src)
            for src in TERZAGHI_NGAMMA_SOURCES
            if src != self.ngamma_method
            and (src != "kumbhojkar" or phi_used <= 50)
        }
        note = (f"Terzaghi (1943) bearing capacity; N-gamma = {Ng:.2f} from "
                f"{TERZAGHI_NGAMMA_SOURCES[self.ngamma_method]}.")
        if alternatives:
            note += " Other cited N-gamma: " + "; ".join(
                f"ngamma_method='{src}' gives {v:.2f}"
                for src, v in alternatives.items()) + "."
        notes.append(note)

        shape = self.footing.shape
        if shape in ("square", "rectangular") and abs(B - L) <= 1e-9 * L:
            shape = "square"
        sc, sq, sg, shape_note = terzaghi_shape_factors(shape, B, L)
        if shape_note:
            notes.append(shape_note)
        if Df > B:
            notes.append(
                f"Df/B = {Df / B:.2f} > 1: Terzaghi's method is for Df/B <= 1 "
                "(UFC 3-220-20 Table 5-3); it ignores the shear strength of "
                "the soil above the base. Consider factor_method='meyerhof'.")

        q = self.soil.overburden_pressure(Df)
        gamma = self.soil.gamma_below_footing(Df, B)

        term_c = c_used * Nc * sc
        term_q = q * Nq * sq
        term_g = 0.5 * gamma * B * Ng * sg
        q_ultimate = term_c + term_q + term_g

        return BearingCapacityResult(
            q_ultimate=q_ultimate,
            q_allowable=q_ultimate / self.factor_of_safety,
            q_net=q_ultimate - q,
            factor_of_safety=self.factor_of_safety,
            Nc=Nc, Nq=Nq, Ngamma=Ng,
            sc=sc, sq=sq, sgamma=sg,
            q_overburden=q,
            gamma_eff=gamma,
            B_eff=B,
            L_eff=L,
            term_cohesion=term_c,
            term_overburden=term_q,
            term_selfweight=term_g,
            ngamma_method=self.ngamma_method,
            factor_method=self.factor_method,
            failure_mode=self.failure_mode,
            Ngamma_alternatives=alternatives,
            c_used=c_used,
            phi_used=phi_used,
            notes=notes,
        )

    def _compute_two_layer(self) -> BearingCapacityResult:
        """Compute bearing capacity for a two-layer soil profile.

        Uses the load-spread (projected-area) method. The footing bears within
        the upper layer; the capacity is governed by the smaller of the
        upper-layer (semi-infinite) capacity and the capacity controlled by the
        lower layer reached through the upper layer.

        * **Stronger-over-weaker** (``q_top >= q_bottom``): the net footing
          pressure spreads 2V:1H through the competent upper layer to a larger
          area at the top of the weak layer, which must carry the spread
          pressure. As the upper layer thickens the spread area grows and the
          upper layer governs (``-> q_top``); as it thins the weak layer governs
          (``-> q_bottom``).
        * **Weaker-over-stronger** (``q_top < q_bottom``): the weak upper layer
          governs when thick; as it thins (``H < B``) the stronger lower layer
          contributes, so capacity is taken as a bounded transition from
          ``q_top`` (at ``H >= B``) up to ``q_bottom`` (as ``H -> 0``). This
          branch is a bounded engineering estimate, not a closed-form solution.

        ``layer1.thickness`` is interpreted as the upper-layer thickness *below
        the footing base* (``H``). The result is always bounded between the two
        single-layer capacities. The reported term/factor breakdown is the
        bearing (upper) layer's, proportionally scaled so the three terms sum to
        the two-layer ``q_ultimate``.

        References
        ----------
        NAVFAC DM-7.01, Ch. 4 (load spread through a stronger stratum);
        Bowles, "Foundation Analysis and Design," 5th ed., Sec. 4-7.
        Meyerhof & Hanna (1978), Canadian Geotechnical Journal 15(4) — the
        punching-shear alternative (requires the K_s punching-shear chart).
        """
        layer1 = self.soil.layer1
        layer2 = self.soil.layer2
        H = layer1.thickness  # upper-layer thickness BELOW the footing base (m)

        B = self.footing.B_for_factors
        L = self.footing.L_for_factors
        Df = self.footing.depth

        # Upper layer treated as semi-infinite (footing bears at Df within it)
        result_top = self._compute_single_layer()
        qt = result_top.q_ultimate

        # Lower layer beneath the same footing (reference capacity)
        soil_bottom = BearingSoilProfile(layer1=SoilLayer(
            cohesion=layer2.cohesion,
            friction_angle=layer2.friction_angle,
            unit_weight=layer2.unit_weight,
        ), gwt_depth=self.soil.gwt_depth)
        result_bottom = BearingCapacityAnalysis(
            footing=self.footing,
            soil=soil_bottom,
            load_inclination=self.load_inclination,
            ground_slope=self.ground_slope,
            vertical_load=self.vertical_load,
            factor_of_safety=self.factor_of_safety,
            ngamma_method=self.ngamma_method,
            factor_method=self.factor_method,
            failure_mode=self.failure_mode,
            depth_factor_method=self.depth_factor_method,
        )._compute_single_layer()
        qb = result_bottom.q_ultimate

        q_ovb = result_top.q_overburden  # effective overburden at footing base (kPa)
        q_lo = min(qt, qb)
        q_hi = max(qt, qb)

        if qt >= qb:
            # Stronger over weaker: 2V:1H load spread through the upper layer.
            # The net footing pressure at the base spreads to an area
            # (B+H)(L+H) at the top of the weak layer, which limits the net
            # pressure it can carry to its own net capacity (qb - overburden).
            qb_net = max(qb - result_bottom.q_overburden, 0.0)
            spread = ((B + H) * (L + H)) / (B * L)
            q_ultimate = qb_net * spread + q_ovb
        else:
            # Weaker over stronger: bounded transition q_top (H>=B) -> q_bottom (H->0).
            if H >= B:
                q_ultimate = qt
            else:
                q_ultimate = qt + (qb - qt) * (1.0 - H / B)

        # Always bounded between the two single-layer capacities.
        q_ultimate = min(max(q_ultimate, q_lo), q_hi)

        q_allowable = q_ultimate / self.factor_of_safety
        q_net = q_ultimate - q_ovb

        # Self-consistent breakdown: report the bearing (upper) layer's factors,
        # with the term breakdown proportionally scaled so the three terms sum
        # to the two-layer q_ultimate (the combined mechanism is punching/spread,
        # not a single general-shear equation, so attribution is proportional).
        result = result_top
        scale = q_ultimate / qt if qt > 0 else 0.0
        result.term_cohesion *= scale
        result.term_overburden *= scale
        result.term_selfweight *= scale
        result.q_ultimate = q_ultimate
        result.q_allowable = q_allowable
        result.q_net = q_net
        result.is_two_layer = True
        result.q_upper_layer = qt
        result.q_lower_layer = qb
        return result
