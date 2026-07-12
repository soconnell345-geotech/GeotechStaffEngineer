"""
Slope geometry and soil layer definitions for limit equilibrium analysis.

Slope is defined by a ground surface profile (x, z) and soil layers
by elevation ranges. All units SI: meters, kPa, kN/m3, degrees.

References:
    Duncan, Wright & Brandon (2014) — Soil Strength and Slope Stability
    Abramson et al. (2002) — Slope Stability and Stabilization Methods
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from slope_stability.nails import SoilNail


def _interp_polyline(points, x):
    """Linearly interpolate a polyline at x with constant extrapolation.

    Parameters
    ----------
    points : list of (float, float)
        Polyline as (x, z) pairs, sorted by x.
    x : float
        Query x-coordinate.

    Returns
    -------
    float
        Interpolated z value.
    """
    if not points:
        raise ValueError("Empty polyline")
    if x <= points[0][0]:
        return points[0][1]
    if x >= points[-1][0]:
        return points[-1][1]
    for i in range(len(points) - 1):
        x0, z0 = points[i]
        x1, z1 = points[i + 1]
        if x0 <= x <= x1:
            t = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
            return z0 + t * (z1 - z0)
    return points[-1][1]


@dataclass
class SlopeSoilLayer:
    """A soil layer within the slope, defined by elevation boundaries.

    Parameters
    ----------
    name : str
        Layer identifier / description.
    top_elevation : float
        Top elevation of layer (m). Higher = up.
    bottom_elevation : float
        Bottom elevation of layer (m).
    gamma : float
        Total unit weight (kN/m3).
    gamma_sat : float, optional
        Saturated unit weight (kN/m3). If None, uses gamma.
    phi : float
        Effective friction angle (degrees). For drained analysis.
    c_prime : float
        Effective cohesion (kPa). For drained analysis.
    cu : float
        Undrained shear strength (kPa). For undrained (phi=0) analysis.
    analysis_mode : str
        'drained' (uses c', phi') or 'undrained' (uses cu, phi=0).
        Default 'drained'.
    bottom_boundary_points : list of (float, float), optional
        Non-horizontal bottom boundary as (x, z) polyline, sorted by x.
        When set, bottom_at(x) interpolates this polyline instead of
        using the flat bottom_elevation.
    """
    name: str
    top_elevation: float
    bottom_elevation: float
    gamma: float
    gamma_sat: Optional[float] = None
    phi: float = 0.0
    c_prime: float = 0.0
    cu: float = 0.0
    analysis_mode: str = "drained"
    ru: float = 0.0
    bottom_boundary_points: Optional[List[Tuple[float, float]]] = None
    # --- strength model selection -------------------------------------
    # 'mohr_coulomb' (default): c'/phi' or cu per analysis_mode.
    # 'shansep':   su = shansep_S * OCR^shansep_m * sigma'_v  (phi = 0)
    # 'hoek_brown': Generalized Hoek-Brown (GSI, mi, sigci, D) ->
    #               instantaneous c-phi tangent at the slice base
    #               effective normal stress (Fellenius estimate).
    # 'anisotropic': undrained su varies with the slice-base inclination alpha
    #               (ADP active/DSS/passive; su_active/su_dss/su_passive, phi=0).
    #               See _anisotropic_su and strength_at.
    strength_model: str = "mohr_coulomb"
    shansep_S: float = 0.22
    shansep_m: float = 0.8
    ocr: float = 1.0
    su_min: float = 0.0
    hb_sigci: float = 0.0
    hb_gsi: float = 50.0
    hb_mi: float = 10.0
    hb_D: float = 0.0
    # --- anisotropic undrained strength (ADP: active / DSS / passive) --------
    # Undrained strength as a function of the slice-base inclination alpha (the
    # failure-plane angle to horizontal). su_active applies in the ACTIVE zone
    # (alpha >= +45 deg, steep bases under the crest / driving side), su_passive
    # in the PASSIVE zone (alpha <= -45 deg, reverse-dip bases at the toe /
    # resisting side), su_dss at direct simple shear (alpha = 0). Only used when
    # strength_model == 'anisotropic'. See _anisotropic_su.
    su_active: float = 0.0
    su_dss: float = 0.0
    su_passive: float = 0.0
    # --- rapid-drawdown R-envelope (Corps/Duncan) ---------------------
    # Total-stress (consolidated-undrained, "R") strength envelope used ONLY
    # by the rapid-drawdown 3-stage / 2-stage analysis. When R_phi is set, the
    # layer is treated as LOW-PERMEABILITY (undrained during rapid drawdown);
    # when None it is FREE-DRAINING (keeps its effective strength throughout).
    # See slope_stability/rapid_drawdown.py.
    R_c: float = 0.0            # R-envelope cohesion intercept (kPa)
    R_phi: Optional[float] = None   # R-envelope friction angle (deg); None = free-draining

    def __post_init__(self):
        if self.bottom_boundary_points is None:
            if self.bottom_elevation >= self.top_elevation:
                raise ValueError(
                    f"Layer '{self.name}': bottom_elevation ({self.bottom_elevation}) "
                    f"must be less than top_elevation ({self.top_elevation})"
                )
        if self.gamma <= 0:
            raise ValueError(
                f"Layer '{self.name}': gamma must be positive, got {self.gamma}"
            )
        if self.gamma_sat is None:
            self.gamma_sat = self.gamma
        if self.analysis_mode not in ("drained", "undrained"):
            raise ValueError(
                f"analysis_mode must be 'drained' or 'undrained', "
                f"got '{self.analysis_mode}'"
            )
        if self.analysis_mode == "drained":
            if self.phi < 0:
                raise ValueError(f"phi must be non-negative, got {self.phi}")
            if self.c_prime < 0:
                raise ValueError(f"c_prime must be non-negative, got {self.c_prime}")
        else:
            if self.cu < 0:
                raise ValueError(f"cu must be non-negative, got {self.cu}")
        if self.strength_model not in ("mohr_coulomb", "shansep",
                                       "hoek_brown", "anisotropic"):
            raise ValueError(
                f"strength_model must be 'mohr_coulomb', 'shansep', "
                f"'hoek_brown' or 'anisotropic', got '{self.strength_model}'")
        if self.strength_model == "anisotropic":
            for nm in ("su_active", "su_dss", "su_passive"):
                if getattr(self, nm) < 0:
                    raise ValueError(f"{nm} must be non-negative, got "
                                     f"{getattr(self, nm)}")
            if max(self.su_active, self.su_dss, self.su_passive) <= 0:
                raise ValueError(
                    "anisotropic strength_model needs a positive su_active, "
                    "su_dss or su_passive")
        if self.strength_model == "shansep":
            if self.shansep_S <= 0:
                raise ValueError(f"shansep_S must be positive, got {self.shansep_S}")
            if self.ocr < 1.0:
                raise ValueError(f"OCR must be >= 1, got {self.ocr}")
            if not (0.0 < self.shansep_m <= 1.5):
                raise ValueError(f"shansep_m out of range: {self.shansep_m}")
        if self.strength_model == "hoek_brown":
            if self.hb_sigci <= 0:
                raise ValueError(
                    "hb_sigci (intact UCS, kPa) must be positive for the "
                    f"Hoek-Brown model, got {self.hb_sigci}")
            if not (0.0 < self.hb_gsi <= 100.0):
                raise ValueError(f"hb_gsi out of (0, 100]: {self.hb_gsi}")
            if self.hb_mi <= 0:
                raise ValueError(f"hb_mi must be positive, got {self.hb_mi}")
            if not (0.0 <= self.hb_D <= 1.0):
                raise ValueError(f"hb_D out of [0, 1]: {self.hb_D}")

    @property
    def thickness(self) -> float:
        """Layer thickness (m)."""
        return self.top_elevation - self.bottom_elevation

    @property
    def shear_strength_params(self) -> Tuple[float, float]:
        """Return (cohesion, phi) depending on analysis mode.

        For drained: (c', phi')
        For undrained: (cu, 0)
        """
        if self.analysis_mode == "undrained":
            return (self.cu, 0.0)
        return (self.c_prime, self.phi)

    def strength_at(self, sigma_n_eff: float, sigma_v_eff: float,
                    alpha: Optional[float] = None) -> Tuple[float, float]:
        """Base shear-strength parameters (c, phi) for this layer.

        Parameters
        ----------
        sigma_n_eff : float
            Effective normal stress estimate on the slice base (kPa) —
            Fellenius estimate (W*cos(a)/l - u), used by the Hoek-Brown
            tangent. The instantaneous c-phi is exact on the GHB
            envelope at this stress; methods then iterate FOS with these
            fixed parameters (standard LE treatment; Slide2's
            'Generalized Hoek-Brown' does the same per-slice
            linearization).
        sigma_v_eff : float
            Effective vertical (overburden) stress at the slice base
            (kPa), used by SHANSEP.
        alpha : float, optional
            Slice-base inclination (radians, failure-plane angle to
            horizontal). Required by the 'anisotropic' model; ignored by
            the others.

        Returns
        -------
        (c, phi) : tuple of float — cohesion (kPa) and friction angle
            (degrees) to use at the slice base.
        """
        if self.strength_model == "shansep":
            su = self.shansep_S * (self.ocr ** self.shansep_m) \
                * max(sigma_v_eff, 0.0)
            return (max(su, self.su_min), 0.0)
        if self.strength_model == "hoek_brown":
            return _hoek_brown_instantaneous(
                max(sigma_n_eff, 0.0), self.hb_sigci, self.hb_gsi,
                self.hb_mi, self.hb_D)
        if self.strength_model == "anisotropic":
            return (self._anisotropic_su(0.0 if alpha is None else alpha), 0.0)
        return self.shear_strength_params

    def _anisotropic_su(self, alpha_rad: float) -> float:
        """Anisotropic undrained strength su at base inclination ``alpha_rad``.

        ADP interpolation between su_passive (alpha <= -45 deg, passive/toe),
        su_dss (alpha = 0, direct simple shear) and su_active (alpha >= +45 deg,
        active/crest). Within each half the Casagrande & Carrillo (1944)
        elliptical variation su = su_h + (su_v - su_h)*sin^2(i) — with i the
        major-principal-stress inclination from horizontal and the failure plane
        at 45 deg to sigma1 (phi_u = 0, so i = alpha + 45 deg) — reduces to
        sin(2*alpha) (since 2*sin^2(alpha+45) - 1 = sin(2*alpha)):

            alpha in [0, 45) deg:   su_dss + (su_active  - su_dss)*sin(2*alpha)
            alpha in (-45, 0] deg:  su_dss + (su_passive - su_dss)*sin(2*|alpha|)

        held constant beyond +/-45 deg (the triaxial-compression/extension
        failure-plane angles). su_active >= su_dss >= su_passive for K>1 natural
        clays, but any ordering is accepted.

        ALPHA SIGN CONVENTION (the #1 error source — be explicit): positive
        alpha = base dipping toward +x = the ACTIVE (driving/crest) side for the
        module's standard slope (toe at low x, crest at high x, sliding toward
        low x — the orientation every validation geometry uses). A mirrored
        slope (crest at low x) must swap su_active/su_passive. NOTE alpha is the
        failure-PLANE angle, not the major-principal-stress inclination i (they
        differ by ~45 deg for phi_u = 0).
        """
        a = alpha_rad
        quarter = math.pi / 4.0        # 45 deg
        if a >= quarter:
            return self.su_active
        if a <= -quarter:
            return self.su_passive
        if a >= 0.0:
            return self.su_dss + (self.su_active - self.su_dss) \
                * math.sin(2.0 * a)
        return self.su_dss + (self.su_passive - self.su_dss) \
            * math.sin(-2.0 * a)

    def bottom_at(self, x):
        """Bottom elevation at x. Uses polyline if set, else flat bottom_elevation."""
        if self.bottom_boundary_points is not None:
            return _interp_polyline(self.bottom_boundary_points, x)
        return self.bottom_elevation

    def top_at(self, x):
        """Top elevation at x. Returns flat top_elevation (surface clipping done in slices)."""
        return self.top_elevation


@dataclass
class SlopeGeometry:
    """Complete slope definition: ground surface + soil layers + water.

    Parameters
    ----------
    surface_points : list of (float, float)
        Ground surface profile as (x, z) coordinates, left to right.
        z = elevation. Must have at least 2 points.
    soil_layers : list of SlopeSoilLayer
        Soil layers sorted by decreasing top_elevation.
    gwt_points : list of (float, float), optional
        Groundwater table as (x, z_gwt) points. None = no water.
    surcharge : float
        Uniform vertical surcharge on slope surface (kPa). Default 0.
    surcharge_x_range : tuple of (float, float), optional
        (x_start, x_end) over which surcharge is applied.
        None = entire surface.
    reinforcement_force : float
        Horizontal reinforcement force (kN/m). Default 0.
    reinforcement_elevation : float, optional
        Elevation at which reinforcement force acts (m).
    kh : float
        Horizontal seismic coefficient. Default 0 (no seismic).
    nails : list of SoilNail, optional
        Soil nails for reinforcement. Default None (no nails).
    anchors : list of reinforcement.Anchor, optional
        Tiebacks/anchors with user-specified allowable tension.
    geosynthetics : list of reinforcement.Geosynthetic, optional
        Horizontal reinforcement layers (T_allow per crossing).
    """
    surface_points: List[Tuple[float, float]]
    soil_layers: List[SlopeSoilLayer]
    gwt_points: Optional[List[Tuple[float, float]]] = None
    surcharge: float = 0.0
    surcharge_x_range: Optional[Tuple[float, float]] = None
    # Additional surcharge zones (v5.4 E8) for problems with several distinct
    # loaded areas (e.g. a bench load + a crest load). Each is a
    # (pressure_kPa, x_start, x_end) triple; ``surcharge_at`` SUMS every zone
    # covering x, ON TOP OF the single ``surcharge``/``surcharge_x_range`` pair
    # (which is unchanged). Default None = single-surcharge behaviour preserved.
    surcharges: Optional[List[Tuple[float, float, float]]] = None
    reinforcement_force: float = 0.0
    reinforcement_elevation: Optional[float] = None
    kh: float = 0.0
    nails: Optional[List[SoilNail]] = None
    anchors: Optional[list] = None
    geosynthetics: Optional[list] = None
    stabilizing_piles: Optional[list] = None
    tension_crack_depth: float = 0.0
    tension_crack_water_depth: float = 0.0
    # Tension-crack placement + model (v5.4 E4). Defaults reproduce the historical
    # entry-side, strength-truncation behaviour byte-for-byte.
    #  * ``tension_crack_side``: 'entry' (default, low-x / slip-surface entry) or
    #    'exit' (high-x). The crack forms at the CREST; put it on whichever side
    #    of the surface the crest is on (no need to mirror the slope any more).
    #  * ``tension_crack_model``: 'strength' (default) keeps the cracked wedge as
    #    zero-shear-strength DRIVING soil; 'truncation' REMOVES it from the sliding
    #    mass (the mass ends at the vertical crack face, as Slide2/UTEXAS do).
    #    Both apply the hydrostatic crack-water thrust on the retained face.
    tension_crack_side: str = "entry"
    tension_crack_model: str = "strength"
    # Discrete pore-pressure field: scattered (x, z, u) points (kPa), e.g. from a
    # flow net / TIN. When set, the base pore pressure at each slice is
    # INTERPOLATED from this field (linear on the Delaunay triangulation, with a
    # nearest fallback outside the hull) instead of the piezometric-line / ru
    # models. The ponded-water buttress is still driven by ``gwt_points`` (set
    # both for a pool over a flow-net field). Default None = unchanged.
    pore_pressure_points: Optional[List[Tuple[float, float, float]]] = None

    def __post_init__(self):
        if len(self.surface_points) < 2:
            raise ValueError("Surface must have at least 2 points")
        xs = [p[0] for p in self.surface_points]
        if any(xs[i] >= xs[i + 1] for i in range(len(xs) - 1)):
            raise ValueError("Surface points must be sorted left-to-right by x")
        if len(self.soil_layers) == 0:
            raise ValueError("At least one soil layer is required")
        if self.kh < 0:
            raise ValueError(f"kh must be non-negative, got {self.kh}")
        if self.surcharge < 0:
            raise ValueError(f"surcharge must be non-negative, got {self.surcharge}")
        if self.surcharges is not None:
            for z in self.surcharges:
                if len(z) != 3:
                    raise ValueError(
                        "each surcharges entry must be a (pressure, x_start, "
                        f"x_end) triple, got {z!r}")
                p, x0, x1 = z
                if p < 0:
                    raise ValueError(
                        f"surcharge zone pressure must be non-negative, got {p}")
                if x1 <= x0:
                    raise ValueError(
                        f"surcharge zone needs x_start < x_end, got {(x0, x1)}")
        if self.tension_crack_depth < 0:
            raise ValueError(f"tension_crack_depth must be non-negative, got {self.tension_crack_depth}")
        if self.tension_crack_water_depth < 0:
            raise ValueError(f"tension_crack_water_depth must be non-negative, got {self.tension_crack_water_depth}")
        if self.tension_crack_water_depth > self.tension_crack_depth:
            self.tension_crack_water_depth = self.tension_crack_depth
        if self.tension_crack_side not in ("entry", "exit"):
            raise ValueError("tension_crack_side must be 'entry' or 'exit', "
                             f"got '{self.tension_crack_side}'")
        if self.tension_crack_model not in ("strength", "truncation"):
            raise ValueError("tension_crack_model must be 'strength' or "
                             f"'truncation', got '{self.tension_crack_model}'")
        if self.pore_pressure_points is not None:
            if len(self.pore_pressure_points) < 1:
                raise ValueError("pore_pressure_points, if given, must be a "
                                 "non-empty list of (x, z, u) triples")
            for p in self.pore_pressure_points:
                if len(p) != 3:
                    raise ValueError(
                        "each pore_pressure_points entry must be an "
                        f"(x, z, u) triple, got {p!r}")

    def ground_elevation_at(self, x: float) -> float:
        """Linearly interpolate ground surface elevation at x.

        Extrapolates as constant beyond surface extent.
        """
        pts = self.surface_points
        if x <= pts[0][0]:
            return pts[0][1]
        if x >= pts[-1][0]:
            return pts[-1][1]
        for i in range(len(pts) - 1):
            x0, z0 = pts[i]
            x1, z1 = pts[i + 1]
            if x0 <= x <= x1:
                t = (x - x0) / (x1 - x0)
                return z0 + t * (z1 - z0)
        return pts[-1][1]

    def gwt_elevation_at(self, x: float) -> Optional[float]:
        """Linearly interpolate GWT elevation at x. None if no GWT."""
        if self.gwt_points is None:
            return None
        pts = self.gwt_points
        if len(pts) == 0:
            return None
        if x <= pts[0][0]:
            return pts[0][1]
        if x >= pts[-1][0]:
            return pts[-1][1]
        for i in range(len(pts) - 1):
            x0, z0 = pts[i]
            x1, z1 = pts[i + 1]
            if x0 <= x <= x1:
                t = (x - x0) / (x1 - x0)
                return z0 + t * (z1 - z0)
        return pts[-1][1]

    def layer_at_elevation(self, z: float) -> Optional[SlopeSoilLayer]:
        """Return the soil layer that contains elevation z.

        Searches from top to bottom. Returns None if z is outside
        all layers.
        """
        for layer in self.soil_layers:
            if layer.bottom_elevation <= z <= layer.top_elevation:
                return layer
        return None

    def layer_at_point(self, x: float, z: float) -> Optional[SlopeSoilLayer]:
        """Return the soil layer containing point (x, z).

        Uses position-dependent boundaries when layers have
        bottom_boundary_points set. Falls back to flat elevations.
        Returns None if outside all layers.
        """
        for layer in self.soil_layers:
            top_z = layer.top_at(x)
            bot_z = layer.bottom_at(x)
            if bot_z <= z <= top_z:
                return layer
        return None

    @property
    def x_range(self) -> Tuple[float, float]:
        """(x_min, x_max) of the ground surface."""
        return (self.surface_points[0][0], self.surface_points[-1][0])

    @property
    def z_range(self) -> Tuple[float, float]:
        """(z_min, z_max) of the ground surface."""
        zs = [p[1] for p in self.surface_points]
        return (min(zs), max(zs))

    def surcharge_at(self, x: float) -> float:
        """Return the TOTAL surcharge pressure at x (kPa).

        Sums the single ``surcharge`` (within ``surcharge_x_range``, or
        everywhere if that is None) and every ``surcharges`` zone
        ``(pressure, x_start, x_end)`` that covers x. With ``surcharges=None``
        this is byte-identical to the single-surcharge behaviour.
        """
        q = 0.0
        if self.surcharge > 0:
            if self.surcharge_x_range is None:
                q += self.surcharge
            else:
                x_lo, x_hi = self.surcharge_x_range
                if x_lo <= x <= x_hi:
                    q += self.surcharge
        if self.surcharges:
            for p, x0, x1 in self.surcharges:
                if x0 <= x <= x1:
                    q += p
        return q

    def pore_pressure_at(self, x: float, z: float) -> Optional[float]:
        """Interpolate the discrete pore-pressure field at (x, z).

        Returns the pore pressure (kPa, clamped >= 0) linearly interpolated on
        the Delaunay triangulation of ``pore_pressure_points`` (nearest-node
        fallback outside the convex hull / for degenerate point sets), or None
        if no field is set. This is the convenience/adapter entry point; the
        slice builder builds the interpolator once per call for the hot path.
        """
        if self.pore_pressure_points is None:
            return None
        interp = build_pore_pressure_interpolator(self.pore_pressure_points)
        return interp(x, z)


def _hoek_brown_instantaneous(sigma_n: float, sigci: float, gsi: float,
                              mi: float, D: float):
    """Instantaneous (c, phi) on the Generalized Hoek-Brown envelope.

    GHB (Hoek, Carranza-Torres & Corkum 2002):
        sigma1 = sigma3 + sigci*(mb*sigma3/sigci + s)^a
        mb = mi*exp((GSI-100)/(28-14D)),  s = exp((GSI-100)/(9-3D)),
        a = 1/2 + (exp(-GSI/15) - exp(-20/3))/6

    Balmer's equations give the normal/shear stress on the failure
    plane for a given sigma3:
        d1 = dsigma1/dsigma3 = 1 + a*mb*(mb*sigma3/sigci + s)^(a-1)
        sigma_n = sigma3 + (sigma1 - sigma3)/(d1 + 1)
        tau     = (sigma_n - sigma3)*sqrt(d1)

    sigma3 is found by bisection so that Balmer's sigma_n equals the
    requested base normal stress; the tangent then yields
        phi_i = atan( (d1 - 1) / (2*sqrt(d1)) ),
        c_i   = tau - sigma_n*tan(phi_i).

    Returns (c_i_kPa, phi_i_deg).
    """
    mb = mi * math.exp((gsi - 100.0) / (28.0 - 14.0 * D))
    s = math.exp((gsi - 100.0) / (9.0 - 3.0 * D))
    a = 0.5 + (math.exp(-gsi / 15.0) - math.exp(-20.0 / 3.0)) / 6.0
    sig_t = -s * sigci / mb  # tensile cutoff (sigma3 at zero strength)

    def balmer(sigma3):
        term = mb * sigma3 / sigci + s
        if term <= 0:
            return None
        d1 = 1.0 + a * mb * term ** (a - 1.0)
        sigma1 = sigma3 + sigci * term ** a
        sn = sigma3 + (sigma1 - sigma3) / (d1 + 1.0)
        tau = (sn - sigma3) * math.sqrt(d1)
        return sn, tau, d1

    # bisection on sigma3 so balmer sigma_n == sigma_n requested
    lo = sig_t * 0.999
    hi = max(sigma_n, sigci) * 2.0 + 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        b = balmer(mid)
        if b is None:
            lo = mid
            continue
        if b[0] > sigma_n:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-9 * max(abs(hi), 1.0):
            break
    b = balmer(0.5 * (lo + hi))
    if b is None:
        return (0.0, 0.0)
    sn, tau, d1 = b
    phi_i = math.atan((d1 - 1.0) / (2.0 * math.sqrt(d1)))
    c_i = max(tau - sn * math.tan(phi_i), 0.0)
    return (c_i, math.degrees(phi_i))


def build_pore_pressure_interpolator(points):
    """Build a callable u(x, z) -> kPa from scattered pore-pressure points.

    ``points`` is a list of (x, z, u) triples (a flow-net / TIN sampling).
    Returns a function that linearly interpolates u on the Delaunay
    triangulation of the (x, z) nodes, falling back to the nearest node outside
    the convex hull or when the triangulation is degenerate (< 3 points, or
    collinear points). Values are clamped to >= 0 (suction is not carried into
    the effective-stress reduction). SI throughout (m, kPa).

    scipy is imported lazily (same pattern as search.search_de) so importing
    geometry stays dependency-light.
    """
    import numpy as np

    pts = np.asarray([(float(x), float(z)) for x, z, _ in points], dtype=float)
    vals = np.asarray([float(u) for _, _, u in points], dtype=float)

    # Nearest-node fallback (always available, incl. 1-2 points / collinear).
    try:
        from scipy.interpolate import NearestNDInterpolator
        nearest = NearestNDInterpolator(pts, vals)
    except Exception:
        nearest = None

    linear = None
    if len(points) >= 3:
        try:
            from scipy.interpolate import LinearNDInterpolator
            linear = LinearNDInterpolator(pts, vals)
        except Exception:
            linear = None

    def _interp(x: float, z: float) -> float:
        u = float("nan")
        if linear is not None:
            u = float(linear(x, z))
        if (u != u or u is None) and nearest is not None:  # nan -> outside hull
            u = float(nearest(x, z))
        if u != u:  # still nan: no usable interpolator
            # last-resort inverse-distance on the raw points
            d2 = (pts[:, 0] - x) ** 2 + (pts[:, 1] - z) ** 2
            if len(d2) == 0:
                return 0.0
            j = int(np.argmin(d2))
            u = float(vals[j])
        return max(u, 0.0)

    return _interp
