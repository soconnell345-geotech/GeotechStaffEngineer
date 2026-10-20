"""
Soil layer and profile definitions for downdrag analysis.

Defines soil layers with both strength parameters (for skin friction)
and consolidation parameters (for settlement), plus a soil profile
that computes effective stress at any depth.

All units are SI: meters (m), kilonewtons (kN), kilopascals (kPa).

References
----------
- Fellenius, B.H. (2006). "Results of static loading tests on driven piles."
- AASHTO LRFD Bridge Design Specifications, Section 10.7.3.7.
- UFC 3-220-20, Chapter 6.
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np


#: How a layer's unit skin friction is worked out (GM-249): the Beta method
#: (fs = beta sigma'v, UFC 3-220-20 Eq 6-7) or the Alpha method (fs = alpha
#: cu, UFC 3-220-20 Eq 6-8). A cohesionless layer takes the Beta method; a
#: cohesive layer the Alpha method unless ``skin_friction_method="beta"``.
SKIN_FRICTION_METHODS = ("alpha", "beta")

#: Where the Beta method is printed for cohesive soil (long-term, drained
#: conditions), and its values in clay.
BETA_IN_CLAY_CITE = "UFC 3-220-20 §6-5.4.1, p. 411; FHWA GEC-12 §7.2.1.3.4, p. 255"
BETA_IN_CLAY_VALUES = (
    "from local experience, else from phi' by GEC-12 Table 7-9, p. 256 "
    "(after Fellenius 2014): clay 0.15-0.35 for phi' 25-30 deg; UFC "
    "3-220-20 Table 6-15, p. 414, the same for driven piles in clay")


@dataclass
class DowndragSoilLayer:
    """A soil layer for downdrag analysis.

    Combines strength parameters (for skin friction computation) with
    consolidation parameters (for settlement computation in settling layers).

    Parameters
    ----------
    thickness : float
        Layer thickness (m).
    soil_type : str
        "cohesionless" or "cohesive".
    unit_weight : float
        Total unit weight above GWT, or saturated unit weight below GWT (kN/m^3).
    phi : float
        Effective friction angle (degrees). Required for cohesionless; a
        cohesive layer does not take it (nothing would use it).
    cu : float
        Undrained shear strength (kPa). Required for a cohesive layer with
        the Alpha method, and for a cohesive layer the pile toe sits in
        (toe resistance Nt x cu, 9 cu unless Nt is given).
    beta : float, optional
        Beta coefficient for skin friction (fs = beta sigma'v). Cohesionless:
        if None, computed from phi using Fellenius (1991): beta =
        (1-sin(phi))*tan(phi). Cohesive: only with
        ``skin_friction_method="beta"``, which requires it.
    alpha : float, optional
        Alpha coefficient for cohesive skin friction (fs = alpha cu), Alpha
        method only. If None, defaults to 1.0 (conservative for soft clay).
        A cohesionless layer does not take it.
    Cc : float
        Compression index. Required if settling=True and cohesive. Given
        with e0 on any clay, C_ec and C_er are derived from them, settling
        or not: below the pile toe the toe load compresses any clay with a
        compressibility (``settling`` decides only whether the fill /
        drawdown does).
    Cr : float
        Recompression index (C_er = Cr/(1+e0); 0 if not given).
    e0 : float
        Initial void ratio. Required if settling=True and cohesive.
    C_ec : float, optional
        Modified compression index = Cc/(1+e0). If provided, overrides Cc/e0.
        Used in UFC Eq 6-53.
    C_er : float, optional
        Modified recompression index = Cr/(1+e0). If provided, overrides Cr/e0.
        Used in UFC Eq 6-53. Required with C_ec when sigma_p is given (the
        clay is then overconsolidated where sigma_p exceeds sigma'v0).
    sigma_p : float, optional
        Preconsolidation pressure (kPa). If None, layer is normally consolidated.
    E_s : float, optional
        Young's modulus of soil (kPa). Required for coarse-grained settling layers.
        Used in UFC Eq 6-54.
    nu_s : float
        Poisson's ratio of soil. Default 0.3. Used for coarse-grained settlement
        per UFC Eq 6-54.
    settling : bool
        True if this layer settles due to fill placement or GW drawdown.
    description : str
        Layer description.
    skin_friction_method : str, optional
        ``"alpha"`` or ``"beta"``; None (default) = the layer's default: the
        Alpha method (fs = alpha cu, UFC 3-220-20 Eq 6-8) for a cohesive
        layer, the Beta method (fs = beta sigma'v, Eq 6-7) for a
        cohesionless one. ``"beta"`` on a cohesive layer is the effective
        stress option for long-term, drained conditions (UFC 3-220-20
        §6-5.4.1; FHWA GEC-12 §7.2.1.3.4) and needs ``beta`` (GEC-12
        Table 7-9: clay 0.15-0.35). A cohesionless layer takes the Beta
        method only. :attr:`fs_method` is the method used.
    """
    thickness: float
    soil_type: str
    unit_weight: float
    phi: float = 0.0
    cu: float = 0.0
    beta: Optional[float] = None
    alpha: Optional[float] = None
    Cc: float = 0.0
    Cr: float = 0.0
    e0: float = 0.0
    C_ec: Optional[float] = None
    C_er: Optional[float] = None
    sigma_p: Optional[float] = None
    E_s: Optional[float] = None
    nu_s: float = 0.3
    settling: bool = False
    description: str = ""
    skin_friction_method: Optional[str] = None

    @property
    def fs_method(self) -> str:
        """The skin friction method the layer uses: ``"alpha"`` or
        ``"beta"`` (``skin_friction_method`` or the soil type's default)."""
        if self.skin_friction_method is not None:
            return self.skin_friction_method
        return "alpha" if self.soil_type == "cohesive" else "beta"

    def _check_skin_friction_inputs(self):
        """Refuse a skin friction input the layer would not use (GM-249:
        a ``beta`` on a clay was dropped without a word), naming the input
        that takes effect."""
        method = self.skin_friction_method
        if method is not None and method not in SKIN_FRICTION_METHODS:
            raise ValueError(
                f"skin_friction_method must be one of "
                f"{list(SKIN_FRICTION_METHODS)}, got '{method}'")
        if self.soil_type == "cohesionless":
            if method == "alpha":
                raise ValueError(
                    "A cohesionless layer does not take skin_friction_method "
                    "'alpha': its skin friction is beta x sigma'v (the Beta "
                    "method, UFC 3-220-20 Eq 6-7), beta as given or from "
                    "phi. The Alpha method is for cohesive layers.")
            if self.alpha is not None:
                raise ValueError(
                    "A cohesionless layer does not take alpha: its skin "
                    "friction is beta x sigma'v (the Beta method, UFC "
                    "3-220-20 Eq 6-7), beta as given or from phi. Leave "
                    "alpha out.")
            return
        if self.phi > 0:
            raise ValueError(
                "A cohesive layer does not take phi: its skin friction is "
                "alpha x cu (the Alpha method, the default), or beta x "
                "sigma'v with skin_friction_method 'beta' and beta given, "
                "and a toe in it takes Nt x cu (9 cu unless Nt is given). "
                "For a clay in long-term, drained conditions, give "
                "skin_friction_method 'beta' and the beta for its phi' "
                f"({BETA_IN_CLAY_VALUES}); the module does not work beta "
                "out from phi' in clay. Leave phi out.")
        if self.fs_method == "alpha":
            if self.beta is not None:
                raise ValueError(
                    "A cohesive layer does not take beta with the Alpha "
                    "method (the default for cohesive soil): its skin "
                    "friction is alpha x cu (UFC 3-220-20 Eq 6-8). For "
                    "beta x sigma'v, the Beta method for long-term, drained "
                    f"conditions ({BETA_IN_CLAY_CITE}), give "
                    "skin_friction_method 'beta' with beta "
                    f"({BETA_IN_CLAY_VALUES}); otherwise leave beta out.")
            if self.cu <= 0:
                raise ValueError(
                    f"Cohesive layer must have cu > 0, got {self.cu}"
                )
            return
        if self.alpha is not None:
            raise ValueError(
                "A cohesive layer with skin_friction_method 'beta' does not "
                "take alpha: its skin friction is beta x sigma'v (UFC "
                "3-220-20 Eq 6-7). Leave alpha out, or use "
                "skin_friction_method 'alpha' for alpha x cu.")
        if self.beta is None or self.beta <= 0:
            raise ValueError(
                "A cohesive layer with skin_friction_method 'beta' needs "
                f"beta > 0, {BETA_IN_CLAY_VALUES} (GEC-12 §7.2.1.3.4 Step "
                "2, p. 257).")
        if self.cu < 0:
            raise ValueError(f"cu must be non-negative, got {self.cu}")

    def __post_init__(self):
        if self.thickness <= 0:
            raise ValueError(
                f"Layer thickness must be positive, got {self.thickness}"
            )
        valid_types = ("cohesionless", "cohesive")
        if self.soil_type not in valid_types:
            raise ValueError(
                f"soil_type must be one of {valid_types}, got '{self.soil_type}'"
            )
        if self.unit_weight <= 0:
            raise ValueError(
                f"Unit weight must be positive, got {self.unit_weight}"
            )
        if self.soil_type == "cohesionless" and self.phi <= 0:
            raise ValueError(
                f"Cohesionless layer must have phi > 0, got {self.phi}"
            )
        self._check_skin_friction_inputs()
        if self.soil_type == "cohesive":
            # The modified indices whenever Cc and e0 are given, settling or
            # not: below the pile toe the toe load compresses any clay with
            # a compressibility; ``settling`` only says whether the fill /
            # drawdown compresses the layer (GM-49).
            has_traditional = self.Cc > 0 and self.e0 > 0
            if self.Cr < 0:
                raise ValueError(
                    f"Cr must be non-negative, got {self.Cr}"
                )
            if self.C_ec is None and has_traditional:
                self.C_ec = self.Cc / (1.0 + self.e0)
            if self.C_er is None and has_traditional:
                self.C_er = self.Cr / (1.0 + self.e0)
            if (self.C_ec is not None and self.C_er is None
                    and self.sigma_p is not None):
                # sigma_p makes the clay overconsolidated where it exceeds
                # sigma'v0, and UFC Eq 6-53 then needs the recompression
                # index; there is no default for it.
                raise ValueError(
                    f"A cohesive layer with C_ec and sigma_p "
                    f"({self.sigma_p:g} kPa) needs the recompression index "
                    f"too: give C_er (= Cr/(1+e0)), or Cc, Cr and e0 "
                    f"(UFC 3-220-20 Eq 6-53)"
                    + (f" [{self.description}]" if self.description else ""))
        if self.settling:
            if self.soil_type == "cohesive":
                # Cohesive: need Cc/e0 or C_ec/C_er
                has_traditional = self.Cc > 0 and self.e0 > 0
                has_modified = (self.C_ec is not None and self.C_ec > 0)
                if not has_traditional and not has_modified:
                    raise ValueError(
                        "Cohesive settling layer must have (Cc > 0 and e0 > 0)"
                        " or C_ec > 0"
                    )
            else:
                # Cohesionless: need E_s for elastic settlement (Eq 6-54)
                if self.E_s is None or self.E_s <= 0:
                    raise ValueError(
                        "Cohesionless settling layer must have E_s > 0"
                    )

        # Compute beta from phi if not provided (cohesionless; a cohesive
        # layer has no phi, and the Beta method there needs beta given)
        if self.beta is None and self.phi > 0:
            phi_rad = math.radians(self.phi)
            self.beta = (1.0 - math.sin(phi_rad)) * math.tan(phi_rad)

        # Default alpha for cohesive layers with the Alpha method
        if self.alpha is None and self.fs_method == "alpha":
            self.alpha = 1.0


@dataclass
class DowndragSoilProfile:
    """Soil profile for downdrag analysis.

    A stack of layers from the ground surface downward, with a
    groundwater table depth.

    Parameters
    ----------
    layers : list of DowndragSoilLayer
        Soil layers from surface downward.
    gwt_depth : float
        Depth to groundwater table from ground surface (m). Default 0 (at surface).
    gamma_w : float
        Unit weight of water (kN/m^3). Default 9.81.
    """
    layers: List[DowndragSoilLayer]
    gwt_depth: float = 0.0
    gamma_w: float = 9.81

    def __post_init__(self):
        if not self.layers:
            raise ValueError("At least one soil layer is required")
        if self.gwt_depth < 0:
            raise ValueError(
                f"GWT depth must be non-negative, got {self.gwt_depth}"
            )

    @property
    def total_depth(self) -> float:
        """Total depth of all layers (m)."""
        return sum(layer.thickness for layer in self.layers)

    def effective_stress_at_depth(self, z: float) -> float:
        """Compute effective vertical stress at depth z.

        Parameters
        ----------
        z : float
            Depth from ground surface (m).

        Returns
        -------
        float
            Effective vertical stress (kPa).
        """
        sigma_total = 0.0
        u = 0.0
        depth = 0.0

        for layer in self.layers:
            top = depth
            bot = depth + layer.thickness

            if z <= top:
                break

            z_in_layer = min(z, bot) - top

            # Total stress contribution from this layer
            sigma_total += layer.unit_weight * z_in_layer

            depth = bot

        # Pore water pressure (hydrostatic below GWT)
        if z > self.gwt_depth:
            u = self.gamma_w * (z - self.gwt_depth)

        return sigma_total - u

    def layer_at_depth(self, z: float) -> DowndragSoilLayer:
        """Return the layer containing depth z.

        Parameters
        ----------
        z : float
            Depth from ground surface (m).

        Returns
        -------
        DowndragSoilLayer

        Raises
        ------
        ValueError
            If z is below all layers.
        """
        depth = 0.0
        for layer in self.layers:
            bot = depth + layer.thickness
            if z <= bot:
                return layer
            depth = bot
        raise ValueError(
            f"Depth {z} m is below the soil profile (total depth {self.total_depth} m)"
        )

    def get_layer_boundaries(self) -> List[float]:
        """Return cumulative depth to the bottom of each layer.

        Returns
        -------
        list of float
            Depths to bottom of each layer (m).
        """
        boundaries = []
        depth = 0.0
        for layer in self.layers:
            depth += layer.thickness
            boundaries.append(depth)
        return boundaries
