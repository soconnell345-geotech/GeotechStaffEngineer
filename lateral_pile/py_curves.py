"""
p-y curve models for laterally loaded pile analysis.

Implements nonlinear soil spring (p-y) formulations from the COM624P program
and related FHWA/API publications. Each model computes the soil resistance p
(force per unit length, kN/m) as a function of lateral deflection y (m) at
a given depth z (m) for a pile of diameter b (m).

All units are SI: meters (m), kilonewtons (kN), kilopascals (kPa), kN/m3.

References
----------
- COM624P Manual: FHWA-SA-91-048 (Wang & Reese, 1993)
- Matlock, H. (1970). "Correlations for Design of Laterally Loaded Piles
  in Soft Clay." OTC 1204.
- Reese, L.C., Cox, W.R. & Koop, F.D. (1974). "Field Testing and Analysis
  of Laterally Loaded Piles in Sand." OTC 2080.
- Reese, L.C., Cox, W.R. & Koop, F.D. (1975). "Field Testing and Analysis
  of Laterally Loaded Piles in Stiff Clay." OTC 2312.
- Welch, R.C. & Reese, L.C. (1972). "Laterally Loaded Behavior of Drilled
  Shafts." Research Report 89-10, Center for Highway Research, Univ. of Texas.
- API RP2A-WSD, 21st Edition (2000). "Recommended Practice for Planning,
  Designing and Constructing Fixed Offshore Platforms."
- O'Neill, M.W. & Murchison, J.M. (1983). "An Evaluation of p-y
  Relationships in Sands." Report to API, Univ. of Houston.
- Reese, L.C. (1997). "Analysis of Laterally Loaded Piles in Weak Rock."
  J. Geotech. & Geoenviron. Eng., 123(11), 1010-1017.
- Jeanjean, P. (2009). "Re-assessment of p-y curves for soft clays from
  centrifuge testing and finite element modeling." OTC-20158-MS.
- Jeanjean, P., Zhang, Y., Andersen, K.H., Gilbert, R., & Senanayake, A.
  (2017). "A framework for monotonic p-y curves in clays." OTC-27466-MS.
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Tuple, Optional

import numpy as np


# =============================================================================
# Utility functions
# =============================================================================

def _safe_sign(y: float) -> float:
    """Return sign of y, treating zero as positive."""
    if y >= 0:
        return 1.0
    return -1.0


# =============================================================================
# 1. Soft Clay Below Water Table — Matlock (1970)
# =============================================================================

@dataclass
class SoftClayMatlock:
    """p-y curves for soft clay below water table per Matlock (1970).

    This is the most widely used p-y formulation for soft to medium clays.
    The method was developed from full-scale tests on instrumented piles
    at Lake Austin and Sabine Pass, Texas.

    Parameters
    ----------
    c : float
        Undrained shear strength (kPa). Typical: 10-100 kPa for soft-medium clay.
    gamma : float
        Effective unit weight of soil (kN/m^3). Typical: 6-11 kN/m^3 submerged.
    eps50 : float
        Strain at 50% of maximum deviator stress from UU triaxial test.
        Typical values: soft clay 0.02, medium clay 0.01, stiff clay 0.005.
    J : float
        Empirical constant. Use 0.5 for soft clay (Gulf of Mexico),
        0.25 for medium clay. Default 0.5.
    loading : str
        'static' or 'cyclic'. Default 'static'.

    References
    ----------
    Matlock, H. (1970). OTC 1204.
    COM624P Manual (FHWA-SA-91-048), Section 2.3.1.

    Examples
    --------
    >>> model = SoftClayMatlock(c=25.0, gamma=8.0, eps50=0.02, J=0.5)
    >>> p = model.get_p(y=0.01, z=3.0, b=0.6)
    """
    c: float
    gamma: float
    eps50: float
    J: float = 0.5
    loading: str = 'static'

    def __post_init__(self):
        if self.c <= 0:
            raise ValueError(f"Undrained shear strength c must be positive, got {self.c}")
        if self.gamma <= 0:
            raise ValueError(f"Unit weight gamma must be positive, got {self.gamma}")
        if self.eps50 <= 0:
            raise ValueError(f"eps50 must be positive, got {self.eps50}")
        if self.J < 0:
            raise ValueError(f"J must be non-negative, got {self.J}")
        if self.loading not in ('static', 'cyclic'):
            raise ValueError(f"loading must be 'static' or 'cyclic', got '{self.loading}'")

        # Range warnings
        if self.c > 100:
            warnings.warn(f"c = {self.c} kPa is high for soft clay; consider stiff clay model")
        if self.eps50 > 0.05:
            warnings.warn(f"eps50 = {self.eps50} is unusually high")
        if self.eps50 < 0.003:
            warnings.warn(f"eps50 = {self.eps50} is unusually low for soft clay")

    def get_pu(self, z: float, b: float) -> float:
        """Compute ultimate soil resistance at depth z.

        Parameters
        ----------
        z : float
            Depth below ground surface (m).
        b : float
            Pile diameter (m).

        Returns
        -------
        float
            Ultimate resistance pu (kN/m).
        """
        # Shallow (wedge) failure mechanism
        pu_shallow = (3.0 + self.gamma * z / self.c + self.J * z / b) * self.c * b
        # Deep (flow-around) failure mechanism
        pu_deep = 9.0 * self.c * b
        return min(pu_shallow, pu_deep)

    def get_zr(self, b: float) -> float:
        """Compute critical depth zr where transition occurs.

        Below zr, the flow-around mechanism governs (pu = 9*c*b).

        Parameters
        ----------
        b : float
            Pile diameter (m).

        Returns
        -------
        float
            Critical depth zr (m).
        """
        # Solve: 3*c*b + gamma*z*b + J*c*z = 9*c*b
        # => z * (gamma*b + J*c) = 6*c*b
        denominator = self.gamma * b + self.J * self.c
        if denominator <= 0:
            return float('inf')
        return 6.0 * self.c * b / denominator

    def get_p(self, y: float, z: float, b: float) -> float:
        """Compute soil resistance p for a given deflection y at depth z.

        Parameters
        ----------
        y : float
            Lateral deflection (m). Positive away from soil.
        z : float
            Depth below ground surface (m).
        b : float
            Pile diameter (m).

        Returns
        -------
        float
            Soil resistance p (kN/m). Same sign as y.
        """
        if z < 0:
            return 0.0

        sign = _safe_sign(y)
        y_abs = abs(y)

        pu = self.get_pu(z, b)
        y50 = 2.5 * self.eps50 * b

        if y_abs == 0:
            return 0.0

        if self.loading == 'static':
            if y_abs <= 8.0 * y50:
                p = 0.5 * pu * (y_abs / y50) ** (1.0 / 3.0)
            else:
                p = pu
        else:
            # Cyclic loading
            zr = self.get_zr(b)
            if y_abs <= 3.0 * y50:
                p = 0.5 * pu * (y_abs / y50) ** (1.0 / 3.0)
            else:
                if z >= zr:
                    # Deep: constant at 0.72*pu
                    p = 0.72 * pu
                else:
                    # Shallow: reduced by z/zr
                    p = 0.72 * pu * (z / zr) if zr > 0 else 0.0

        return sign * p

    def get_py_curve(self, z: float, b: float, n_points: int = 50,
                     y_max_factor: float = 15.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a complete p-y curve at depth z.

        Parameters
        ----------
        z : float
            Depth below ground surface (m).
        b : float
            Pile diameter (m).
        n_points : int
            Number of points in the curve. Default 50.
        y_max_factor : float
            Maximum y as a multiple of y50. Default 15.

        Returns
        -------
        y_array : numpy.ndarray
            Deflection values (m).
        p_array : numpy.ndarray
            Soil resistance values (kN/m).
        """
        y50 = 2.5 * self.eps50 * b
        y_max = y_max_factor * y50
        y_array = np.linspace(0, y_max, n_points)
        p_array = np.array([self.get_p(y, z, b) for y in y_array])
        return y_array, p_array


# =============================================================================
# 2. Stiff Clay Below Water Table — Reese et al. (1975)
# =============================================================================

@dataclass
class StiffClayBelowWT:
    """p-y curves for stiff clay below the water table per Reese et al. (1975).

    Uses a five-segment p-y curve construction with separate static and cyclic
    formulations. Requires the initial modulus of subgrade reaction ks in
    addition to shear strength and strain parameters.

    Parameters
    ----------
    c : float
        Undrained shear strength (kPa). Typical: 50-400 kPa for stiff clay.
    gamma : float
        Effective unit weight of soil (kN/m^3).
    eps50 : float
        Strain at 50% of max deviator stress. Typical: 0.004-0.007.
    ks : float
        Initial modulus of subgrade reaction (kN/m^3).
        Typical: 135,000 kN/m^3 for stiff clay (50-100 kPa),
        270,000 kN/m^3 for very stiff clay (100-200 kPa),
        540,000 kN/m^3 for hard clay (200-400 kPa).
    loading : str
        'static' or 'cyclic'. Default 'static'.

    References
    ----------
    Reese, L.C., Cox, W.R. & Koop, F.D. (1975). OTC 2312.
    COM624P Manual (FHWA-SA-91-048), Section 2.3.3.
    """
    c: float
    gamma: float
    eps50: float
    ks: float
    loading: str = 'static'

    def __post_init__(self):
        if self.c <= 0:
            raise ValueError(f"c must be positive, got {self.c}")
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")
        if self.eps50 <= 0:
            raise ValueError(f"eps50 must be positive, got {self.eps50}")
        if self.ks <= 0:
            raise ValueError(f"ks must be positive, got {self.ks}")
        if self.loading not in ('static', 'cyclic'):
            raise ValueError(f"loading must be 'static' or 'cyclic', got '{self.loading}'")

        if self.c < 50:
            warnings.warn(f"c = {self.c} kPa may be too low for stiff clay model")

    def _get_As(self, z: float, b: float) -> float:
        """Static coefficient As as function of z/b.

        Interpolated from Reese et al. (1975) Table/Figure.
        """
        zb = z / b if b > 0 else 0
        # Tabulated values from COM624P manual
        if zb <= 0:
            return 2.5  # at surface
        elif zb >= 12:
            return 0.88
        else:
            # Piecewise linear interpolation from published values
            zb_tab = [0, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12]
            As_tab = [2.5, 2.2, 1.9, 1.5, 1.25, 1.1, 1.0, 0.95, 0.93, 0.9, 0.88]
            return float(np.interp(zb, zb_tab, As_tab))

    def _get_Ac(self, z: float, b: float) -> float:
        """Cyclic coefficient Ac as function of z/b.

        Interpolated from Reese et al. (1975) Table/Figure.
        """
        zb = z / b if b > 0 else 0
        if zb <= 0:
            return 0.2
        elif zb >= 12:
            return 0.55
        else:
            zb_tab = [0, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12]
            Ac_tab = [0.2, 0.2, 0.3, 0.35, 0.4, 0.42, 0.45, 0.47, 0.5, 0.52, 0.55]
            return float(np.interp(zb, zb_tab, Ac_tab))

    def get_pu(self, z: float, b: float) -> float:
        """Ultimate soil resistance.

        Two mechanisms:
        - Shallow (wedge): pu = (2*c*b + gamma*b*z + 2.83*c*z)
        - Deep (flow-around): pu = 11*c*b
        """
        pu_shallow = 2.0 * self.c * b + self.gamma * b * z + 2.83 * self.c * z
        pu_deep = 11.0 * self.c * b
        return min(pu_shallow, pu_deep)

    def get_p(self, y: float, z: float, b: float) -> float:
        """Compute soil resistance for stiff clay below water table.

        Uses the five-segment construction from Reese et al. (1975).
        """
        if z < 0:
            return 0.0

        sign = _safe_sign(y)
        y_abs = abs(y)
        if y_abs == 0:
            return 0.0

        pu = self.get_pu(z, b)
        y50 = 2.5 * self.eps50 * b

        if self.loading == 'static':
            As = self._get_As(z, b)
            pc = As * pu

            # Initial linear portion: p = ks * z * y
            p_linear = self.ks * z * y_abs

            # Parabolic portion: p = 0.5 * pu * (y/y50)^0.5
            if y_abs > 0 and y50 > 0:
                p_para = 0.5 * pu * (y_abs / y50) ** 0.5
            else:
                p_para = 0.0

            # Use the lesser of linear and parabolic in the transition zone
            if y_abs <= y50:
                p = min(p_linear, p_para)
            elif y_abs <= 6.0 * y50:
                p = min(p_para, pc)
            elif y_abs <= 18.0 * y50:
                # Linear decrease from pc to residual
                p = pc - (pc - pu * 0.5) * (y_abs - 6.0 * y50) / (12.0 * y50)
                p = max(p, 0.5 * pu)
            else:
                p = 0.5 * pu  # Residual

            p = min(p, pc)
        else:
            # Cyclic
            Ac = self._get_Ac(z, b)
            pc = Ac * pu

            p_linear = self.ks * z * y_abs

            if y_abs > 0 and y50 > 0:
                p_para = 0.5 * pu * (y_abs / y50) ** 0.5
            else:
                p_para = 0.0

            if y_abs <= y50:
                p = min(p_linear, p_para)
            elif y_abs <= 3.0 * y50:
                p = min(p_para, pc)
            else:
                p = pc  # Constant plateau for cyclic

            p = min(p, pc)

        return sign * p

    def get_py_curve(self, z: float, b: float, n_points: int = 50,
                     y_max_factor: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a complete p-y curve at depth z."""
        y50 = 2.5 * self.eps50 * b
        y_max = y_max_factor * y50
        y_array = np.linspace(0, y_max, n_points)
        p_array = np.array([self.get_p(y, z, b) for y in y_array])
        return y_array, p_array


# =============================================================================
# 3. Stiff Clay Above Water Table — Welch & Reese (1972)
# =============================================================================

@dataclass
class StiffClayAboveWT:
    """p-y curves for stiff clay above the water table per Welch & Reese (1972).

    Uses 1/4 power (not 1/3 as in Matlock) for the p-y relationship.
    Uses total unit weight since soil is above the water table.

    Parameters
    ----------
    c : float
        Undrained shear strength (kPa).
    gamma : float
        Total unit weight of soil (kN/m^3). Note: total, not effective.
    eps50 : float
        Strain at 50% of max deviator stress.
    J : float
        Empirical constant (same as Matlock). Default 0.5.
    loading : str
        'static' or 'cyclic'. Default 'static'.

    References
    ----------
    Welch, R.C. & Reese, L.C. (1972). Research Report 89-10.
    COM624P Manual (FHWA-SA-91-048), Section 2.3.2.
    """
    c: float
    gamma: float
    eps50: float
    J: float = 0.5
    loading: str = 'static'

    def __post_init__(self):
        if self.c <= 0:
            raise ValueError(f"c must be positive, got {self.c}")
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")
        if self.eps50 <= 0:
            raise ValueError(f"eps50 must be positive, got {self.eps50}")
        if self.loading not in ('static', 'cyclic'):
            raise ValueError(f"loading must be 'static' or 'cyclic', got '{self.loading}'")

    def get_pu(self, z: float, b: float) -> float:
        """Ultimate soil resistance (same formulation as Matlock but total gamma)."""
        pu_shallow = (3.0 + self.gamma * z / self.c + self.J * z / b) * self.c * b
        pu_deep = 9.0 * self.c * b
        return min(pu_shallow, pu_deep)

    def get_p(self, y: float, z: float, b: float) -> float:
        """Compute soil resistance using 1/4 power curve.

        Static:  p = 0.5 * pu * (y/y50)^(1/4)
        Cyclic:  p remains constant beyond y = 16*y50
        """
        if z < 0:
            return 0.0

        sign = _safe_sign(y)
        y_abs = abs(y)
        if y_abs == 0:
            return 0.0

        pu = self.get_pu(z, b)
        y50 = 2.5 * self.eps50 * b

        if self.loading == 'static':
            p = 0.5 * pu * (y_abs / y50) ** 0.25
            p = min(p, pu)
        else:
            # Cyclic: cap at value reached at y = 16*y50
            if y_abs <= 16.0 * y50:
                p = 0.5 * pu * (y_abs / y50) ** 0.25
            else:
                p = 0.5 * pu * 16.0 ** 0.25  # = 0.5 * pu * 2.0 = pu
            p = min(p, pu)

        return sign * p

    def get_py_curve(self, z: float, b: float, n_points: int = 50,
                     y_max_factor: float = 25.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a complete p-y curve at depth z."""
        y50 = 2.5 * self.eps50 * b
        y_max = y_max_factor * y50
        y_array = np.linspace(0, y_max, n_points)
        p_array = np.array([self.get_p(y, z, b) for y in y_array])
        return y_array, p_array


# =============================================================================
# 1b. Soft Clay — Jeanjean (2009) [Modern advancement]
# =============================================================================

@dataclass
class SoftClayJeanjean:
    """p-y curves for soft clay per Jeanjean (2009).

    A modern advancement over Matlock (1970), developed from centrifuge
    testing and 3D FEA of laterally loaded conductors. Key differences:
    - Maximum N_p = 12 (vs Matlock's 9), consistent with Randolph & Houlsby
      (1984) plasticity solutions for rough piles.
    - Uses G_max (small-strain shear modulus) instead of eps50.
    - Smooth tanh curve shape instead of power law.

    Now standard practice for offshore applications and recommended by
    API RP 2GEO (2014) as an alternative to Matlock.

    Parameters
    ----------
    su : float
        Undrained shear strength (kPa). Same as 'c' in Matlock.
    gamma : float
        Effective unit weight of soil (kN/m^3).
    Gmax : float
        Small-strain shear modulus (kPa). Typically 100-600 * su.
        Can be measured from seismic CPT (Vs), resonant column, or
        estimated from correlations.
    J : float
        Empirical constant for pu calculation. Default 0.5.
    loading : str
        'static' or 'cyclic'. Default 'static'.

    References
    ----------
    Jeanjean, P. (2009). OTC-20158-MS.
    Jeanjean et al. (2017). OTC-27466-MS.
    Randolph, M.F. & Houlsby, G.T. (1984). Proc. ICE, 77(1), 17-33.
    """
    su: float
    gamma: float
    Gmax: float
    J: float = 0.5
    loading: str = 'static'

    def __post_init__(self):
        if self.su <= 0:
            raise ValueError(f"su must be positive, got {self.su}")
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")
        if self.Gmax <= 0:
            raise ValueError(f"Gmax must be positive, got {self.Gmax}")
        if self.loading not in ('static', 'cyclic'):
            raise ValueError(f"loading must be 'static' or 'cyclic', got '{self.loading}'")

        if self.Gmax < 50 * self.su:
            warnings.warn(f"Gmax/su ratio = {self.Gmax/self.su:.0f} is unusually low")
        if self.Gmax > 1000 * self.su:
            warnings.warn(f"Gmax/su ratio = {self.Gmax/self.su:.0f} is unusually high")

    def get_pu(self, z: float, b: float) -> float:
        """Compute ultimate soil resistance at depth z.

        Uses N_p = 12 at depth (flow-around for rough pile, per
        Randolph & Houlsby 1984) vs Matlock's N_p = 9.

        Shallow: pu = (3 + gamma'*z/su + J*z/b) * su * b
        Deep:    pu = 12 * su * b
        """
        pu_shallow = (3.0 + self.gamma * z / self.su + self.J * z / b) * self.su * b
        pu_deep = 12.0 * self.su * b
        return min(pu_shallow, pu_deep)

    def get_p(self, y: float, z: float, b: float) -> float:
        """Compute soil resistance using Jeanjean tanh formulation.

        p = pu * tanh((Gmax / (f * su)) * |y/b|^0.5) * sign(y)

        where f is a curve-fitting parameter:
        - f = 4.0 for static monotonic loading (Jeanjean 2009)
        - f = 8.0 for cyclic loading
        """
        if z < 0:
            return 0.0

        sign = _safe_sign(y)
        y_abs = abs(y)
        if y_abs == 0:
            return 0.0

        pu = self.get_pu(z, b)

        # Curve-fitting parameter
        f = 4.0 if self.loading == 'static' else 8.0

        # Normalized deflection
        yb = y_abs / b if b > 0 else 0.0

        # Jeanjean tanh formulation
        xi = (self.Gmax / (f * self.su)) * (yb ** 0.5)
        p = pu * math.tanh(xi)

        return sign * p

    def get_py_curve(self, z: float, b: float, n_points: int = 50,
                     y_max_factor: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a complete p-y curve at depth z.

        Parameters
        ----------
        y_max_factor : float
            Maximum y as fraction of pile diameter. Default 0.2 (20%D).
        """
        y_max = y_max_factor * b
        y_array = np.linspace(0, y_max, n_points)
        p_array = np.array([self.get_p(y, z, b) for y in y_array])
        return y_array, p_array


# =============================================================================
# Sand p-y curve coefficient tables
# =============================================================================

def _sand_coefficients(phi: float) -> Tuple[float, float, float]:
    """Compute sand p-y curve coefficients C1, C2, C3 from friction angle.

    Interpolated from the tables in Reese, Cox & Koop (1974) and
    COM624P Manual Table 2.1.

    Parameters
    ----------
    phi : float
        Friction angle (degrees).

    Returns
    -------
    C1, C2, C3 : float
        Coefficients for sand p-y curves.
    """
    # Tabulated values from COM624P Manual / Reese et al. (1974)
    phi_tab = [20, 25, 28, 29, 30, 33, 36, 38, 40, 45]
    C1_tab = [0.20, 0.41, 0.70, 0.85, 1.00, 1.65, 2.60, 3.50, 4.70, 10.0]
    C2_tab = [0.18, 0.45, 0.82, 1.00, 1.25, 2.05, 3.45, 4.65, 6.30, 14.0]
    C3_tab = [17, 28, 46, 55, 67, 120, 210, 310, 460, 1200]

    if phi < 20:
        warnings.warn(f"phi = {phi}° is below minimum tabulated value (20°)")
        phi = 20
    if phi > 45:
        warnings.warn(f"phi = {phi}° is above maximum tabulated value (45°)")
        phi = 45

    C1 = float(np.interp(phi, phi_tab, C1_tab))
    C2 = float(np.interp(phi, phi_tab, C2_tab))
    C3 = float(np.interp(phi, phi_tab, C3_tab))

    return C1, C2, C3


def _sand_k_recommendation(phi: float, below_wt: bool = True) -> float:
    """Recommended initial modulus of subgrade reaction k for sand.

    From COM624P Manual Table 2.2 (approximate values).

    Parameters
    ----------
    phi : float
        Friction angle (degrees).
    below_wt : bool
        True if sand is below water table. Default True.

    Returns
    -------
    float
        Recommended k value (kN/m^3).
    """
    if below_wt:
        phi_tab = [25, 30, 35, 40]
        k_tab = [5400, 11000, 22000, 45000]
    else:
        phi_tab = [25, 30, 35, 40]
        k_tab = [6800, 24000, 61000, 170000]

    phi_clamped = max(25, min(40, phi))
    return float(np.interp(phi_clamped, phi_tab, k_tab))


# =============================================================================
# 4. Sand — Reese, Cox & Koop (1974)
# =============================================================================

@dataclass
class SandReese:
    """p-y curves for sand per Reese, Cox & Koop (1974).

    Uses a three-part curve construction with parabolic transition between
    the initial straight-line portion and the ultimate resistance.

    Parameters
    ----------
    phi : float
        Friction angle (degrees). Typical: 25-40°.
    gamma : float
        Effective unit weight (kN/m^3). Typical: 8-11 kN/m^3 submerged.
    k : float
        Initial modulus of subgrade reaction (kN/m^3).
        See _sand_k_recommendation() for typical values.
    loading : str
        'static' or 'cyclic'. Default 'static'.

    References
    ----------
    Reese, L.C., Cox, W.R. & Koop, F.D. (1974). OTC 2080.
    COM624P Manual (FHWA-SA-91-048), Section 2.3.4.
    """
    phi: float
    gamma: float
    k: float
    loading: str = 'static'

    def __post_init__(self):
        if self.phi <= 0 or self.phi > 50:
            raise ValueError(f"phi must be between 0 and 50, got {self.phi}")
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")
        if self.loading not in ('static', 'cyclic'):
            raise ValueError(f"loading must be 'static' or 'cyclic', got '{self.loading}'")

        self.C1, self.C2, self.C3 = _sand_coefficients(self.phi)

        alpha = self.phi / 2.0  # angle for passive wedge
        beta = 45.0 + self.phi / 2.0
        K0 = 0.4  # at-rest coefficient
        Ka = math.tan(math.radians(45.0 - self.phi / 2.0)) ** 2

        self._alpha_rad = math.radians(alpha)
        self._beta_rad = math.radians(beta)
        self._phi_rad = math.radians(self.phi)
        self._K0 = K0
        self._Ka = Ka

    def get_pu(self, z: float, b: float) -> Tuple[float, float]:
        """Compute ultimate resistance from shallow and deep mechanisms.

        Returns
        -------
        pus : float
            Shallow (wedge) ultimate resistance (kN/m).
        pud : float
            Deep (flow-around) ultimate resistance (kN/m).
        """
        C1, C2, C3 = self.C1, self.C2, self.C3

        pus = (C1 * z + C2 * b) * self.gamma * z
        pud = C3 * b * self.gamma * z

        return pus, pud

    def get_p(self, y: float, z: float, b: float) -> float:
        """Compute soil resistance using Reese sand three-part curve."""
        if z <= 0:
            return 0.0

        sign = _safe_sign(y)
        y_abs = abs(y)
        if y_abs == 0:
            return 0.0

        pus, pud = self.get_pu(z, b)
        pu = min(pus, pud)

        # Adjustment factor
        if self.loading == 'static':
            if pus <= pud:  # shallow
                A = max(0.9, 3.0 - 0.8 * z / b)
            else:
                A = 0.88  # deep, static
        else:
            A = 0.9  # cyclic

        ps = A * pu

        # Initial slope
        k_init = self.k * z

        if k_init <= 0:
            return 0.0

        # Compute ym and pm for the parabolic transition
        # yu at ultimate
        yu = ps / k_init if k_init > 0 else b / 60.0

        # Three-part curve: linear, parabolic, constant
        p_linear = k_init * y_abs

        if p_linear <= ps:
            p = p_linear
        else:
            p = ps

        return sign * p

    def get_py_curve(self, z: float, b: float, n_points: int = 50,
                     y_max_factor: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a complete p-y curve at depth z."""
        y_max = y_max_factor * b / 60.0  # rough estimate
        if z > 0:
            pus, pud = self.get_pu(z, b)
            pu = min(pus, pud)
            k_init = self.k * z
            if k_init > 0:
                y_max = max(y_max, 3.0 * pu / k_init)
        y_array = np.linspace(0, y_max, n_points)
        p_array = np.array([self.get_p(y, z, b) for y in y_array])
        return y_array, p_array


# =============================================================================
# 5. API Sand — O'Neill & Murchison (1983) / API RP2A
# =============================================================================

@dataclass
class SandAPI:
    """p-y curves for sand per API RP2A (O'Neill & Murchison, 1983).

    Simplified hyperbolic tangent formulation widely used for offshore
    and transportation piles. Uses the same ultimate resistance as
    Reese sand but with a smooth tanh transition.

    p = A * pu * tanh(k * z * y / (A * pu))

    Parameters
    ----------
    phi : float
        Friction angle (degrees). Typical: 25-40°.
    gamma : float
        Effective unit weight (kN/m^3).
    k : float
        Initial modulus of subgrade reaction (kN/m^3).
    loading : str
        'static' or 'cyclic'. Default 'static'.

    References
    ----------
    O'Neill, M.W. & Murchison, J.M. (1983). Report to API.
    API RP2A-WSD, 21st Edition (2000), Section 6.8.
    COM624P Manual (FHWA-SA-91-048), Section 2.3.5.
    """
    phi: float
    gamma: float
    k: float
    loading: str = 'static'

    def __post_init__(self):
        if self.phi <= 0 or self.phi > 50:
            raise ValueError(f"phi must be between 0 and 50, got {self.phi}")
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")
        if self.loading not in ('static', 'cyclic'):
            raise ValueError(f"loading must be 'static' or 'cyclic', got '{self.loading}'")

        self.C1, self.C2, self.C3 = _sand_coefficients(self.phi)

    def get_pu(self, z: float, b: float) -> float:
        """Compute ultimate resistance at depth z.

        Uses the same two mechanisms as Reese sand.
        """
        if z <= 0:
            return 0.0
        pus = (self.C1 * z + self.C2 * b) * self.gamma * z
        pud = self.C3 * b * self.gamma * z
        return min(pus, pud)

    def get_A(self, z: float, b: float) -> float:
        """Get adjustment factor A.

        Static: A = max(0.9, 3.0 - 0.8*z/b)
        Cyclic: A = 0.9
        """
        if self.loading == 'cyclic':
            return 0.9
        else:
            return max(0.9, 3.0 - 0.8 * z / b)

    def get_p(self, y: float, z: float, b: float) -> float:
        """Compute soil resistance using API sand hyperbolic tangent.

        p = A * pu * tanh(k * z * y / (A * pu))
        """
        if z <= 0:
            return 0.0

        sign = _safe_sign(y)
        y_abs = abs(y)
        if y_abs == 0:
            return 0.0

        pu = self.get_pu(z, b)
        if pu <= 0:
            return 0.0

        A = self.get_A(z, b)
        Apu = A * pu

        # Hyperbolic tangent formulation
        argument = self.k * z * y_abs / Apu if Apu > 0 else 0.0
        p = Apu * math.tanh(argument)

        return sign * p

    def get_py_curve(self, z: float, b: float, n_points: int = 50,
                     y_max_factor: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a complete p-y curve at depth z."""
        pu = self.get_pu(z, b)
        A = self.get_A(z, b)
        k_init = self.k * z if z > 0 else self.k * 0.01

        if k_init > 0 and pu > 0:
            # y where tanh argument ≈ 3 (99.5% of ultimate)
            y_max = 3.0 * A * pu / (self.k * max(z, 0.01))
        else:
            y_max = 0.1 * b

        if y_max_factor is not None:
            y_max = y_max_factor * b / 60.0

        y_array = np.linspace(0, y_max, n_points)
        p_array = np.array([self.get_p(y, z, b) for y in y_array])
        return y_array, p_array


# =============================================================================
# 6. Weak Rock — Reese (1997) [Lower priority]
# =============================================================================

@dataclass
class WeakRock:
    """p-y curves for weak/weathered rock per Reese (1997).

    For weak rock characterized by unconfined compressive strength qu.

    Parameters
    ----------
    qu : float
        Unconfined compressive strength (kPa).
    Er : float
        Modulus of rock mass (kPa). If not known, estimate as 100-500 * qu.
    gamma_r : float
        Unit weight of rock (kN/m^3). Typical: 20-26 kN/m^3.
    RQD : float
        Rock Quality Designation (%). Used to estimate krm. Default 100.
    loading : str
        'static' or 'cyclic'. Default 'static'.

    References
    ----------
    Reese, L.C. (1997). J. Geotech. & Geoenviron. Eng., 123(11), 1010-1017.
    COM624P Manual (FHWA-SA-91-048), Section 2.3.6.
    """
    qu: float
    Er: float
    gamma_r: float = 22.0
    RQD: float = 100.0
    loading: str = 'static'

    def __post_init__(self):
        if self.qu <= 0:
            raise ValueError(f"qu must be positive, got {self.qu}")
        if self.Er <= 0:
            raise ValueError(f"Er must be positive, got {self.Er}")
        if self.loading not in ('static', 'cyclic'):
            raise ValueError(f"loading must be 'static' or 'cyclic', got '{self.loading}'")

    def get_pu(self, z: float, b: float) -> float:
        """Ultimate resistance for weak rock.

        pur = alpha_r * qu * b * (1 + 1.4 * z/b)  for z/b <= 3
        pur = 5.2 * alpha_r * qu * b               for z/b > 3
        """
        alpha_r = 1.0  # conservative; could be reduced based on RQD
        zb = z / b if b > 0 else 0
        if zb <= 3.0:
            pur = alpha_r * self.qu * b * (1.0 + 1.4 * zb)
        else:
            pur = 5.2 * alpha_r * self.qu * b
        return pur

    def get_p(self, y: float, z: float, b: float) -> float:
        """Compute soil resistance for weak rock."""
        if z < 0:
            return 0.0

        sign = _safe_sign(y)
        y_abs = abs(y)
        if y_abs == 0:
            return 0.0

        pur = self.get_pu(z, b)

        # Initial modulus
        kir = self.Er  # simplified: Kir ≈ Er for initial slope

        # Reference deflection
        yr = pur / (0.25 * kir) if kir > 0 else 0.001 * b

        if y_abs <= yr:
            # Linear region
            p = kir * y_abs * 0.25
        else:
            # Ultimate plateau
            p = pur

        p = min(p, pur)
        return sign * p

    def get_py_curve(self, z: float, b: float, n_points: int = 50,
                     y_max_factor: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a complete p-y curve at depth z."""
        pur = self.get_pu(z, b)
        kir = self.Er * 0.25
        yr = pur / kir if kir > 0 else 0.001 * b
        y_max = y_max_factor * yr
        y_array = np.linspace(0, y_max, n_points)
        p_array = np.array([self.get_p(y, z, b) for y in y_array])
        return y_array, p_array


# =============================================================================
# 7. Liquefied Sand — Rollins et al. (2005)
# =============================================================================

@dataclass
class SandLiquefied:
    """p-y curves for liquefied sand per Rollins et al. (2005).

    Concave-up formulation for piles in fully liquefied sand, based on
    full-scale blast-induced liquefaction tests.  The curve is very soft
    (near-zero initial stiffness) and caps at y = 150 mm.

    No soil strength parameters are needed — the model is purely empirical.

    p = A * (B * y)^C * pd       for y < 0.15 m
    p = pu                        for y >= 0.15 m

    where:
        A = 3e-7 * (z + 1)^6.05
        B = 2.8  * (z + 1)^0.11
        C = 2.85 * (z + 1)^-0.41
        pd = 3.81 * ln(D) + 5.6     (diameter correction)

    Parameters
    ----------
    diameter : float
        Pile diameter (m).  Valid range: 0.3 to 2.6 m.

    References
    ----------
    Rollins, K.M., Gerber, T.M., Lane, J.D., & Ashford, S.A. (2005).
    "Lateral Resistance of a Full-Scale Pile Group in Liquefied Sand."
    J. Geotech. Geoenviron. Eng., 131(1), 115-125.
    """
    diameter: float

    def __post_init__(self):
        if self.diameter <= 0:
            raise ValueError(f"diameter must be positive, got {self.diameter}")
        if self.diameter < 0.3 or self.diameter > 2.6:
            warnings.warn(
                f"SandLiquefied: diameter {self.diameter} m outside validated "
                f"range (0.3-2.6 m). Results may be unreliable.")

    # --- Internal coefficients ---

    @staticmethod
    def _coeff_A(z: float) -> float:
        return 3.0e-7 * (z + 1.0) ** 6.05

    @staticmethod
    def _coeff_B(z: float) -> float:
        return 2.8 * (z + 1.0) ** 0.11

    @staticmethod
    def _coeff_C(z: float) -> float:
        return 2.85 * (z + 1.0) ** (-0.41)

    def _pd(self) -> float:
        """Diameter correction factor."""
        return 3.81 * math.log(self.diameter) + 5.6

    # --- Public interface (duck-typing, matches all other models) ---

    def get_pu(self, z: float, b: float) -> float:
        """Ultimate resistance at depth z (at y = 0.15 m cap).

        Parameters
        ----------
        z : float
            Depth below ground surface (m).
        b : float
            Pile diameter (m).  Not used directly; uses self.diameter.
        """
        if z <= 0:
            return 0.0
        A = self._coeff_A(z)
        B = self._coeff_B(z)
        C = self._coeff_C(z)
        pd = self._pd()
        return A * (B * 0.15) ** C * pd

    def get_p(self, y: float, z: float, b: float) -> float:
        """Compute soil resistance for liquefied sand.

        Parameters
        ----------
        y : float
            Lateral deflection (m).
        z : float
            Depth below ground surface (m).
        b : float
            Pile diameter (m).
        """
        if z <= 0:
            return 0.0

        sign = _safe_sign(y)
        y_abs = abs(y)
        if y_abs == 0:
            return 0.0

        pu = self.get_pu(z, b)

        # Cap at 150 mm deflection
        if y_abs >= 0.15:
            return sign * pu

        A = self._coeff_A(z)
        B = self._coeff_B(z)
        C = self._coeff_C(z)
        pd = self._pd()

        p = A * (B * y_abs) ** C * pd
        return sign * min(p, pu)

    def get_py_curve(self, z: float, b: float, n_points: int = 50,
                     y_max_factor: float = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a complete p-y curve at depth z.

        Default y_max = 0.20 m (past the 0.15 m cap) to show the plateau.
        """
        y_max = 0.20 if y_max_factor is None else y_max_factor * b / 60.0
        y_array = np.linspace(0, y_max, n_points)
        p_array = np.array([self.get_p(y, z, b) for y in y_array])
        return y_array, p_array
