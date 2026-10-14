"""
Main bearing capacity calculation engine.

Computes ultimate and allowable bearing capacity using the general
bearing capacity equation (Vesic/Meyerhof/Hansen forms), or Terzaghi's
(1943) equation with his own factors, for one-layer and two-layer soil
systems.

References:
    FHWA GEC-6 (FHWA-IF-02-054), Chapter 6
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
    inclination_factors,
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
        Total vertical load on footing (kN). Needed for Vesic inclination
        factors. Default 0 (factors computed without V).
    factor_of_safety : float, optional
        Required factor of safety. Default 3.0.
    ngamma_method : str, optional
        Method for Ngamma: "vesic" (default), "meyerhof", or "hansen"; with
        ``factor_method="terzaghi"``, "kumbhojkar" (default) or "coduto"
        (see ``factors.TERZAGHI_NGAMMA_SOURCES``).
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
    vertical_load: float = 0.0
    factor_of_safety: float = 3.0
    ngamma_method: Optional[str] = None
    factor_method: str = "vesic"
    failure_mode: str = "general"

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
            self.ngamma_method = "kumbhojkar" if terzaghi else "vesic"
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
            return self._compute_two_layer()
        return self._compute_single_layer()

    def _compute_single_layer(self) -> BearingCapacityResult:
        """Compute bearing capacity for a single-layer soil profile.

        Uses the general bearing capacity equation:
            qult = c*Nc*sc*dc*ic*bc*gc
                 + q*Nq*sq*dq*iq*bq*gq
                 + 0.5*gamma*B'*Ng*sg*dg*ig*bg*gg
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

        # Shape factors
        sc, sq, sg = shape_factors(phi, B, L, self.factor_method)

        # Depth factors
        dc, dq, dg = depth_factors(phi, Df, B, self.factor_method)

        # Inclination factors
        ic, iq, ig = inclination_factors(
            phi, self.load_inclination, c, B, L,
            self.vertical_load, self.factor_method
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
        # Term 1: cohesion term
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
