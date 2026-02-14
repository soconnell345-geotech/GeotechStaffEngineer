"""
Main bearing capacity calculation engine.

Computes ultimate and allowable bearing capacity using the general
bearing capacity equation (Vesic/Meyerhof/Hansen forms) for one-layer
and two-layer soil systems.

References:
    FHWA GEC-6 (FHWA-IF-02-054), Chapter 6
    FHWA-SA-94-034 (CBEAR User's Guide)
    Meyerhof & Hanna (1978) — Two-layer bearing capacity
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
)
from bearing_capacity.results import BearingCapacityResult


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
        Method for Ngamma: "vesic" (default), "meyerhof", or "hansen".
    factor_method : str, optional
        Method for shape/depth/inclination factors: "vesic" (default) or "meyerhof".

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
    ngamma_method: str = "vesic"
    factor_method: str = "vesic"

    def __post_init__(self):
        if self.footing is None:
            raise ValueError("Footing must be provided")
        if self.soil is None:
            raise ValueError("Soil profile must be provided")
        if self.factor_of_safety <= 0:
            raise ValueError(f"Factor of safety must be positive, got {self.factor_of_safety}")

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
        gamma = self.soil.gamma_below_footing(Df)

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

    def _compute_two_layer(self) -> BearingCapacityResult:
        """Compute bearing capacity for a two-layer soil profile.

        Uses the Meyerhof & Hanna (1978) approach:
        - Computes bearing capacity of each layer as if it were semi-infinite
        - Applies correction based on the relative strength and layer geometry
        - Result is bounded between the two single-layer solutions

        References
        ----------
        Meyerhof & Hanna (1978), Canadian Geotechnical Journal, Vol. 15, No. 4.
        FHWA GEC-6, Section 6.5.
        """
        layer1 = self.soil.layer1
        layer2 = self.soil.layer2
        H = layer1.thickness  # thickness of upper layer below footing

        B = self.footing.B_for_factors
        L = self.footing.L_for_factors
        Df = self.footing.depth

        # Bearing capacity of bottom layer (as if footing directly on it)
        soil_bottom = BearingSoilProfile(layer1=SoilLayer(
            cohesion=layer2.cohesion,
            friction_angle=layer2.friction_angle,
            unit_weight=layer2.unit_weight,
        ), gwt_depth=self.soil.gwt_depth)

        analysis_bottom = BearingCapacityAnalysis(
            footing=self.footing,
            soil=soil_bottom,
            load_inclination=self.load_inclination,
            ground_slope=self.ground_slope,
            vertical_load=self.vertical_load,
            factor_of_safety=self.factor_of_safety,
            ngamma_method=self.ngamma_method,
            factor_method=self.factor_method,
        )
        result_bottom = analysis_bottom._compute_single_layer()
        qb = result_bottom.q_ultimate

        # Bearing capacity of top layer (as if semi-infinite)
        soil_top = BearingSoilProfile(layer1=SoilLayer(
            cohesion=layer1.cohesion,
            friction_angle=layer1.friction_angle,
            unit_weight=layer1.unit_weight,
        ), gwt_depth=self.soil.gwt_depth)

        analysis_top = BearingCapacityAnalysis(
            footing=self.footing,
            soil=soil_top,
            load_inclination=self.load_inclination,
            ground_slope=self.ground_slope,
            vertical_load=self.vertical_load,
            factor_of_safety=self.factor_of_safety,
            ngamma_method=self.ngamma_method,
            factor_method=self.factor_method,
        )
        result_top = analysis_top._compute_single_layer()
        qt = result_top.q_ultimate

        # Meyerhof & Hanna correction
        # For strong-over-weak: qult is limited by punching through
        # For weak-over-strong: qult approaches the stronger bottom layer
        # Modified bearing capacity:
        #   q_ult = qt + (qb - qt) * (1 - H²/(B*H_max))
        #   where H_max = B (maximum influence depth)
        # Bounded: qt <= qult <= qb (for weak-over-strong)
        #          qb <= qult <= qt (for strong-over-weak)

        if H >= B:
            # Upper layer is thick enough; use top layer capacity
            q_ultimate = qt
        else:
            # Interpolate between top and bottom layer capacities
            # Punching coefficient approach (simplified Meyerhof & Hanna)
            ratio = H / B
            if qt >= qb:
                # Strong over weak: punching shear failure possible
                # Ks is a punching shear coefficient (depends on q1/q2 ratio)
                q1_over_q2 = qb / qt if qt > 0 else 0
                # Simplified: linear interpolation weighted by H/B
                q_ultimate = qb + (qt - qb) * ratio
                # Upper bound: cannot exceed top layer capacity
                q_ultimate = min(q_ultimate, qt)
            else:
                # Weak over strong: projection through weak layer
                q_ultimate = qt + (qb - qt) * ratio
                # Upper bound: cannot exceed bottom layer capacity
                q_ultimate = min(q_ultimate, qb)

        q_allowable = q_ultimate / self.factor_of_safety
        q = self.soil.overburden_pressure(Df)
        q_net = q_ultimate - q

        # Return result with top layer factors for reference
        result = result_top
        result.q_ultimate = q_ultimate
        result.q_allowable = q_allowable
        result.q_net = q_net
        result.is_two_layer = True
        result.q_upper_layer = qt
        result.q_lower_layer = qb
        return result
