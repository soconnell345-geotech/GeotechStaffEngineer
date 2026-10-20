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
  J. Geotech. & Geoenviron. Eng., 123(11), 1010-1017; as printed in FHWA
  GEC 9 (FHWA-HIF-18-031, 2018), Appendix A.5.
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
    in soft clay.

    The criterion as printed in the COM624P manual (FHWA-SA-91-048),
    pp. 323-327, Eqs. 3.8-3.13 and Fig. 3.5, and in FHWA GEC 9
    (FHWA-HIF-18-031, 2018), Appendix A.1, pp. 216-219, Eqs. A-1 to A-6.
    pu is the smaller of (3 + gamma' x / c + J x / b) c b and 9 c b
    (3.8, 3.9; A-1, A-2), y50 = 2.5 eps50 b (3.10, A-3).

    Static (Fig. 3.5a):

        p/pu = 0.5 (y/y50)^(1/3)                     (3.11, A-4)
        p = pu beyond y = 8 y50

    Cyclic (Fig. 3.5b): the static curve while p is less than 0.72 pu,
    then, past y = 3 y50,

        x >= x_r: p = 0.72 pu
        x <  x_r: p falls on a straight line from 0.72 pu at 3 y50 to
                  p = 0.72 pu (x / x_r) at 15 y50    (3.13, A-6)
                  and stays there beyond 15 y50

    with x_r = 6 c b / (gamma' b + J c) (3.12, A-5), the depth where the
    two expressions for pu meet.

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
    COM624P Manual (FHWA-SA-91-048), pp. 323-327, Eqs. 3.8-3.13.
    FHWA GEC 9 (FHWA-HIF-18-031, 2018), Appendix A.1, pp. 216-219.

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
        elif y_abs <= 3.0 * y50:
            # Cyclic: the static curve for p below 0.72 pu (COM624P p. 326,
            # step 1), which it reaches at 2.99 y50
            p = min(0.5 * pu * (y_abs / y50) ** (1.0 / 3.0), 0.72 * pu)
        else:
            zr = self.get_zr(b)
            if z >= zr:
                p = 0.72 * pu
            else:
                # above x_r, from 0.72 pu at 3 y50 down to 0.72 pu z/z_r at
                # 15 y50 (Eq. 3.13, A-6), constant beyond
                p_15 = 0.72 * pu * z / zr
                frac = min((y_abs - 3.0 * y50) / (12.0 * y50), 1.0)
                p = 0.72 * pu + (p_15 - 0.72 * pu) * frac

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

#: A_s and A_c of the stiff-clay-below-water criterion against x/b, read off
#: COM624P Fig. 3.7 (FHWA-SA-91-048 p. 330) and its redraw, GEC 9 Fig. A-6
#: (FHWA-HIF-18-031 p. 224). Each curve was traced on both prints: the
#: COM624P scan at 300 dpi, deskewed against its frame (its A ticks read
#: back within 0.003), and the GEC 9 raster against its grid lines; the
#: line's centre was taken across each row where the curve is steep and
#: down each column where it is flat, away from the plotted test points.
#: The two prints agree within 0.013 (GEC 9's redraw sits up to 0.013
#: above COM624P's between x/b 0.25 and 1.5); the table is their mean,
#: rounded to 0.005, but for the deep A_s, 0.60 as COM624P prints it (GEC
#: 9's redraw: 0.593). Both curves start at 0.2 at the ground. A_s reaches
#: 0.60 at x/b about 4.5 and A_c 0.305 at about 1.75; each is drawn dashed,
#: constant, to x/b = 12, where the figure ends, and is held at that value
#: below it.
_STIFF_CLAY_A_ZB = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5,
                    3.0, 3.5, 4.0, 4.5)
_STIFF_CLAY_AS = (0.200, 0.270, 0.340, 0.395, 0.435, 0.470, 0.500, 0.520,
                  0.535, 0.565, 0.580, 0.590, 0.595, 0.600)
_STIFF_CLAY_AC = (0.200, 0.235, 0.255, 0.270, 0.285, 0.290, 0.300, 0.305,
                  0.305, 0.305, 0.305, 0.305, 0.305, 0.305)

#: COM624P Table 3.2 (GEC 9 Table A-1): representative k for stiff clay by
#: the average undrained shear strength to five pile diameters, 0.5-1,
#: 1-2 and 2-4 ton/ft^2 (48-96, 96-192 and 192-383 kPa): ks 500, 1,000 and
#: 2,000 lb/in^3, kc 200, 400 and 800 lb/in^3, here in kN/m^3.
STIFF_CLAY_K_TABLE = {
    "ks": ((48.0, 96.0, 135_700.0), (96.0, 192.0, 271_400.0),
           (192.0, 383.0, 542_900.0)),
    "kc": ((48.0, 96.0, 54_300.0), (96.0, 192.0, 108_600.0),
           (192.0, 383.0, 217_200.0)),
}


@dataclass
class StiffClayBelowWT:
    """p-y curves for stiff clay below the water surface per Reese, Cox &
    Koop (1975).

    The criterion as printed in the COM624P manual (FHWA-SA-91-048),
    pp. 328-335, Eqs. 3.14-3.26, Figs. 3.6-3.8 and Tables 3.2-3.3, and in
    FHWA GEC 9 (FHWA-HIF-18-031, 2018), Appendix A.2, pp. 219-224,
    Eqs. A-7 to A-18 ("stiff clay with free water"). At depth x:

        pct = 2 ca b + gamma' b x + 2.83 ca x          (3.14, A-7)
        pcd = 11 c b                                   (3.15, A-8)

    pc the smaller, with ca, the average undrained shear strength over the
    depth x, taken as the layer's c; y50 = eps50 b (3.17, A-10; not the
    2.5 eps50 b of the other clay criteria); the initial straight line
    p = (k x) y (3.16, A-9), k = ks for static loading and kc for cyclic.

    Static (Fig. 3.6), A_s from Fig. 3.7 at x/b:

        p = 0.5 pc (y/y50)^0.5                       to As y50     (3.18)
        p = 0.5 pc (y/y50)^0.5
            - 0.055 pc ((y - As y50)/(As y50))^1.25  to 6 As y50   (3.19)
        p = 0.5 pc (6 As)^0.5 - 0.411 pc
            - (0.0625/y50) pc (y - 6 As y50)         to 18 As y50  (3.20)
        p = 0.5 pc (6 As)^0.5 - 0.411 pc - 0.75 pc As   beyond     (3.21)

    (GEC 9 A-11 to A-14.) Cyclic (Fig. 3.8), A_c from Fig. 3.7 at x/b,
    yp = 4.1 Ac y50 (3.23):

        p = Ac pc [1 - |(y - 0.45 yp)/(0.45 yp)|^2.5]  to 0.6 yp   (3.24)
        p = 0.936 Ac pc - (0.085/y50) pc (y - 0.6 yp)  to 1.8 yp   (3.25)
        p = 0.936 Ac pc - (0.102/y50) pc yp             beyond     (3.26)

    (GEC 9 A-16 to A-18.) The initial line holds until it meets the curve;
    where they do not meet, the smaller p holds (COM624P's notes, pp. 333
    and 335). The curve's p/y falls with y, so this is the smaller of the
    two at every y; at the ground (x = 0) the line is p = 0. Two readings
    differ between the manuals. GEC 9 prints A-15 as yp = 4.1 As y50;
    COM624P's Eq. 3.23 and Fig. 3.8 give 4.1 Ac y50, used here. GEC 9's
    static step 9 (p. 223) discards the line when it meets Eq. A-11 beyond
    As y50 and carries A-11 to y = 0; that differs from COM624P's note only
    near the ground, where k x is small (a 0.9 m shaft in c = 100 kPa clay,
    ks = 271,400 kN/m^3: the top 0.2 m), and COM624P's note is followed
    here. Where As is below 0.223 (x/b below about 0.08) Eq. 3.21 gives a
    negative p; p is held at zero there.

    Parameters
    ----------
    c : float
        Undrained shear strength (kPa). The criterion comes from tests in
        clay of about 1 to 3 ton/ft^2 (about 100-290 kPa; COM624P p. 328).
    gamma : float
        Effective (submerged) unit weight gamma' (kN/m^3).
    eps50 : float
        Strain at half the maximum principal-stress difference. Table 3.3
        (A-2): 0.007, 0.005 and 0.004 for 48-96, 96-192 and 192-383 kPa.
    ks : float, optional
        k for static loading (kN/m^3), required with loading='static' and
        refused with 'cyclic'. Table 3.2 (A-1): 135,700, 271,400 and
        542,900 kN/m^3 (500, 1,000 and 2,000 lb/in^3) for an average
        undrained shear strength to five diameters of 48-96, 96-192 and
        192-383 kPa.
    loading : str
        'static' or 'cyclic'. Default 'static'.
    kc : float, optional
        k for cyclic loading (kN/m^3), required with loading='cyclic' and
        refused with 'static'. Table 3.2 (A-1): 54,300, 108,600 and 217,200
        kN/m^3 (200, 400 and 800 lb/in^3) for the same ranges.

    References
    ----------
    Reese, L.C., Cox, W.R. & Koop, F.D. (1975). OTC 2312.
    COM624P Manual (FHWA-SA-91-048), pp. 328-335, Eqs. 3.14-3.26.
    FHWA GEC 9 (FHWA-HIF-18-031, 2018), Appendix A.2, pp. 219-224.
    """
    c: float
    gamma: float
    eps50: float
    ks: Optional[float] = None
    loading: str = 'static'
    kc: Optional[float] = None

    def __post_init__(self):
        if self.c <= 0:
            raise ValueError(f"c must be positive, got {self.c}")
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")
        if self.eps50 <= 0:
            raise ValueError(f"eps50 must be positive, got {self.eps50}")
        if self.loading not in ('static', 'cyclic'):
            raise ValueError(f"loading must be 'static' or 'cyclic', got '{self.loading}'")
        need, other = (("ks", "kc") if self.loading == 'static'
                       else ("kc", "ks"))
        value = getattr(self, need)
        if value is None:
            raise ValueError(
                f"{self.loading} loading of stiff clay below the water "
                f"table needs {need}, the k of the initial line p = (k x) y "
                f"(COM624P Eq. 3.16, Table 3.2; GEC 9 Eq. A-9, Table A-1): "
                f"{_k_table_text(need)}")
        if value <= 0:
            raise ValueError(f"{need} must be positive, got {value}")
        if getattr(self, other) is not None:
            raise ValueError(
                f"{other} ({getattr(self, other):g}) is the k for "
                f"{'cyclic' if other == 'kc' else 'static'} loading; "
                f"loading='{self.loading}' uses {need} only (COM624P "
                f"Table 3.2)")

        if self.c < 50:
            warnings.warn(f"c = {self.c} kPa may be too low for stiff clay model")

    @property
    def k(self) -> float:
        """The k of the initial line in force: ks static, kc cyclic."""
        return self.ks if self.loading == 'static' else self.kc

    def _get_As(self, z: float, b: float) -> float:
        """A_s at x/b, from COM624P Fig. 3.7 (GEC 9 Fig. A-6)."""
        zb = z / b if b > 0 else 0.0
        return float(np.interp(zb, _STIFF_CLAY_A_ZB, _STIFF_CLAY_AS))

    def _get_Ac(self, z: float, b: float) -> float:
        """A_c at x/b, from COM624P Fig. 3.7 (GEC 9 Fig. A-6)."""
        zb = z / b if b > 0 else 0.0
        return float(np.interp(zb, _STIFF_CLAY_A_ZB, _STIFF_CLAY_AC))

    def get_pu(self, z: float, b: float) -> float:
        """pc, the smaller of pct = 2 ca b + gamma' b x + 2.83 ca x and
        pcd = 11 c b (COM624P Eqs. 3.14-3.15, GEC 9 A-7, A-8), ca = c."""
        pu_shallow = 2.0 * self.c * b + self.gamma * b * z + 2.83 * self.c * z
        pu_deep = 11.0 * self.c * b
        return min(pu_shallow, pu_deep)

    def _curve(self, y: float, z: float, b: float) -> float:
        """The static (3.18-3.21) or cyclic (3.24-3.26) curve at y >= 0,
        without the initial line."""
        pc = self.get_pu(z, b)
        y50 = self.eps50 * b
        if self.loading == 'static':
            As = self._get_As(z, b)
            if y <= As * y50:
                return 0.5 * pc * math.sqrt(y / y50)
            if y <= 6.0 * As * y50:
                return pc * (0.5 * math.sqrt(y / y50) - 0.055
                             * ((y - As * y50) / (As * y50)) ** 1.25)
            top = 0.5 * pc * math.sqrt(6.0 * As) - 0.411 * pc
            if y <= 18.0 * As * y50:
                return top - 0.0625 / y50 * pc * (y - 6.0 * As * y50)
            return top - 0.75 * pc * As
        Ac = self._get_Ac(z, b)
        yp = 4.1 * Ac * y50
        if y <= 0.6 * yp:
            return Ac * pc * (1.0 - abs((y - 0.45 * yp) / (0.45 * yp)) ** 2.5)
        if y <= 1.8 * yp:
            return 0.936 * Ac * pc - 0.085 / y50 * pc * (y - 0.6 * yp)
        return 0.936 * Ac * pc - 0.102 / y50 * pc * yp

    def get_p(self, y: float, z: float, b: float) -> float:
        """Soil resistance (kN/m): the smaller of the initial line
        (k x) y and the curve, never below zero; the sign of y."""
        if z < 0:
            return 0.0
        sign = _safe_sign(y)
        y_abs = abs(y)
        if y_abs == 0:
            return 0.0
        p = min(self.k * z * y_abs, self._curve(y_abs, z, b))
        return sign * max(p, 0.0)

    def get_py_curve(self, z: float, b: float, n_points: int = 50,
                     y_max_factor: float = 30.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a complete p-y curve at depth z, to ``y_max_factor``
        times y50 = eps50 b."""
        y50 = self.eps50 * b
        y_max = y_max_factor * y50
        y_array = np.linspace(0, y_max, n_points)
        p_array = np.array([self.get_p(y, z, b) for y in y_array])
        return y_array, p_array


def _k_table_text(name: str) -> str:
    """COM624P Table 3.2's ``ks`` or ``kc`` in kN/m^3, for messages."""
    rows = STIFF_CLAY_K_TABLE[name]
    return ("; ".join(f"{k:,.0f} kN/m3 for {lo:g}-{hi:g} kPa"
                      for lo, hi, k in rows)
            + ", by the average undrained shear strength to five diameters")


# =============================================================================
# 3. Stiff Clay Above Water Table — Welch & Reese (1972)
# =============================================================================

@dataclass
class StiffClayAboveWT:
    """p-y curves for stiff clay above the water table per Welch & Reese (1972).

    The criterion as printed in the COM624P manual (FHWA-SA-91-048),
    pp. 336-340, Eqs. 3.27-3.29 and Figs. 3.9-3.10, and in FHWA GEC 9
    (FHWA-HIF-18-031, 2018), Appendix A.3, pp. 224-226, Eqs. A-19 to A-21.
    pu is the smaller of Eqs. 3.8 and 3.9 (GEC 9 A-1, A-2) with J = 0.5
    and y50 = 2.5 eps50 b (Eq. 3.10, GEC 9 A-3).

    Static (Fig. 3.9):

        p/pu = 0.5 (y/y50)^(1/4)                         (3.27, A-19)
        p = pu beyond y = 16 y50

    Cyclic (Fig. 3.10), for N cycles of the design lateral load: for each
    p/pu, the deflection on the static curve y_s grows to

        C   = 9.6 (p/pu)^4                               (3.28, A-20)
        y_c = y_s + y50 C log N                          (3.29, A-21)

    with log the common logarithm. Eq. 3.27 gives y_s = 16 y50 (p/pu)^4,
    so y_c = y50 (p/pu)^4 (16 + 9.6 log N): the static curve with y50
    multiplied by (1 + 0.6 log N), reaching pu at y = 16 y50 + 9.6 y50
    log N as Fig. 3.10 labels it. N = 1 gives the static curve.

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
    n_cycles : float, optional
        N, the number of times the design lateral load will be applied
        (COM624P p. 338, step 2; GEC 9 p. 226, step 2), at least 1.
        Required for 'cyclic' loading, which is refused without it: the
        references give no default number. Refused with 'static' loading,
        whose curve takes no N.

    References
    ----------
    Welch, R.C. & Reese, L.C. (1972). Research Report 89-10.
    COM624P Manual (FHWA-SA-91-048), pp. 336-340, Eqs. 3.27-3.29.
    FHWA GEC 9 (FHWA-HIF-18-031, 2018), Appendix A.3, pp. 224-226.
    """
    c: float
    gamma: float
    eps50: float
    J: float = 0.5
    loading: str = 'static'
    n_cycles: Optional[float] = None

    def __post_init__(self):
        if self.c <= 0:
            raise ValueError(f"c must be positive, got {self.c}")
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")
        if self.eps50 <= 0:
            raise ValueError(f"eps50 must be positive, got {self.eps50}")
        if self.loading not in ('static', 'cyclic'):
            raise ValueError(f"loading must be 'static' or 'cyclic', got '{self.loading}'")
        if self.loading == 'cyclic' and self.n_cycles is None:
            raise ValueError(
                "cyclic loading of stiff clay above the water table needs "
                "n_cycles, the number of times the design lateral load will "
                "be applied: the curve softens with log N (y_c = y_s + y50 C "
                "log N, COM624P Eq. 3.29, GEC 9 Eq. A-21) and the references "
                "give no default")
        if self.loading == 'static' and self.n_cycles is not None:
            raise ValueError(
                f"n_cycles ({self.n_cycles:g}) applies to loading='cyclic' "
                "only; the static curve takes no number of cycles")
        if self.n_cycles is not None and self.n_cycles < 1:
            raise ValueError(f"n_cycles must be >= 1, got {self.n_cycles}")

    @property
    def cyclic_factor(self) -> float:
        """y_c / y_s at every p: 1 + 0.6 log10 N (COM624P Eqs. 3.27-3.29);
        1.0 for static loading."""
        if self.loading != 'cyclic':
            return 1.0
        return 1.0 + 0.6 * math.log10(self.n_cycles)

    def get_pu(self, z: float, b: float) -> float:
        """Ultimate soil resistance (same formulation as Matlock but total gamma)."""
        pu_shallow = (3.0 + self.gamma * z / self.c + self.J * z / b) * self.c * b
        pu_deep = 9.0 * self.c * b
        return min(pu_shallow, pu_deep)

    def get_p(self, y: float, z: float, b: float) -> float:
        """Compute soil resistance using the 1/4 power curve.

        Static:  p = 0.5 pu (y/y50)^(1/4), p = pu beyond 16 y50
        Cyclic:  the same with y50 times (1 + 0.6 log10 N), which is
                 Eqs. 3.27-3.29 solved for p; p = pu beyond
                 16 y50 + 9.6 y50 log10 N
        """
        if z < 0:
            return 0.0

        sign = _safe_sign(y)
        y_abs = abs(y)
        if y_abs == 0:
            return 0.0

        pu = self.get_pu(z, b)
        y50 = 2.5 * self.eps50 * b * self.cyclic_factor
        p = min(0.5 * pu * (y_abs / y50) ** 0.25, pu)
        return sign * p

    def get_py_curve(self, z: float, b: float, n_points: int = 50,
                     y_max_factor: float = 25.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a complete p-y curve at depth z, to ``y_max_factor``
        times y50 (times 1 + 0.6 log10 N for cyclic loading)."""
        y50 = 2.5 * self.eps50 * b * self.cyclic_factor
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

    From COM624P Manual (FHWA-SA-91-048) Tables 3.4 (submerged) and 3.5 (above
    water table), printed p. 346 — verified against the source PDF 2026-07-18
    (ledger: module_work/wiki_verification/reese_sand_com624p.md). The manual
    keys k by RELATIVE DENSITY (Loose/Medium/Dense); the phi keying here maps
    Loose=25 / Medium=30 / Dense=35 deg (an interpretation, applied identically
    to both branches). Printed anchors, converted at 1 lb/in^3 = 271.45 kN/m^3:
    submerged 20/60/125 lb/in^3 -> 5,400/16,300/33,900 kN/m^3; above-WT
    25/90/225 -> 6,800/24,400/61,100. The phi=40 points are NOT in the manual
    (extrapolations retained for continuity; flagged non-primary).

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
        k_tab = [5400, 16300, 33900, 45000]
    else:
        phi_tab = [25, 30, 35, 40]
        k_tab = [6800, 24400, 61100, 170000]

    phi_clamped = max(25, min(40, phi))
    return float(np.interp(phi_clamped, phi_tab, k_tab))


# =============================================================================
# 4. Sand — Reese, Cox & Koop (1974)
# =============================================================================

# Reese (1974) non-dimensional A and B coefficients vs depth ratio z/b,
# RE-DIGITIZED 2026-07-18 from the COM624P Manual (FHWA-SA-91-048) Figures 3.12
# (A) and 3.13 (B), printed pp. 344-345 — the source PDF in the owner's library
# (page-calibrated marker reads, tolerance +/-0.05 nominal / 0.10 outer; printed
# asymptote annotations exact). Full ledger:
# module_work/wiki_verification/reese_sand_com624p.md.
# A scales the ultimate curve resistance (pu = A*ps at yu = 3b/80); B scales the
# m-point (pm = B*ps at ym = b/60).
# Printed asymptotes (x/b > 5.0): Fig 3.12 "A = 0.88" (ONE annotation — static
# and cyclic merge at depth); Fig 3.13 "B_c = 0.55, B_s = 0.5".
# The manual's A_c is a LOW hump curve (~0.75 surface -> ~1.08 at z/b 1.5 ->
# 0.88 deep), NOT a descent from 2.9; likewise B_c (~0.50 -> ~0.86 -> 0.55).
# A >= B holds everywhere in the printed charts as-is (A_c >= B_c: 0.75/0.50,
# 1.08/0.86, 0.88/0.55) — the pre-2026-07-18 tables' fabricated cyclic descents
# (2.9 -> 0.55, premised on a wrong A_c asymptote of 0.55, which is actually
# B_c's) are retired.
_REESE_AB_ZB = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
_REESE_A_STATIC = [2.90, 2.54, 2.13, 1.77, 1.48, 1.24, 1.04, 0.88, 0.88]
_REESE_A_CYCLIC = [0.75, 0.91, 1.06, 1.08, 1.01, 0.97, 0.93, 0.88, 0.88]
_REESE_B_STATIC = [2.20, 1.85, 1.56, 1.26, 1.05, 0.88, 0.70, 0.54, 0.50]
_REESE_B_CYCLIC = [0.50, 0.71, 0.84, 0.86, 0.82, 0.78, 0.69, 0.56, 0.55]


def _reese_A(zb: float, cyclic: bool) -> float:
    """Reese (1974) ultimate-resistance coefficient A at depth ratio z/b."""
    tab = _REESE_A_CYCLIC if cyclic else _REESE_A_STATIC
    return float(np.interp(zb, _REESE_AB_ZB, tab))


def _reese_B(zb: float, cyclic: bool) -> float:
    """Reese (1974) m-point coefficient B at depth ratio z/b."""
    tab = _REESE_B_CYCLIC if cyclic else _REESE_B_STATIC
    return float(np.interp(zb, _REESE_AB_ZB, tab))


@dataclass
class SandReese:
    """p-y curves for sand per Reese, Cox & Koop (1974).

    Two curve constructions are available (``construction``):
    - ``"simplified"`` (**default**) — an initial linear branch (k*z*y), a
      1/3-power softening parabola anchored at the ultimate point (yu = 3b/80),
      and the ultimate plateau p_ult = A*min(pus,pud) with A = max(0.9, 3-0.8 z/b)
      (static). Chart-free; the historical module behavior.
    - ``"reese1974"`` — the FULL four-segment Reese (1974) construction: initial
      linear (k*z*y) up to yk, a parabola p = C*y^(1/n), a straight m-segment from
      the m-point (ym = b/60, pm = B*pu) to the ultimate point (yu = 3b/80,
      pu_curve = A*pu), then the plateau. A and B come from the Reese charts
      (``_reese_A``/``_reese_B``). This is stiffer in the working range than the
      simplified curve (it tracks the initial linear further before softening).

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
    construction : str
        'simplified' (default) or 'reese1974' (full four-segment curve).

    References
    ----------
    Reese, L.C., Cox, W.R. & Koop, F.D. (1974). OTC 2080.
    Reese, L.C. & Van Impe, W.F. (2001). Single Piles and Pile Groups Under
    Lateral Loading, Figs 3.30/3.31.
    COM624P Manual (FHWA-SA-91-048), Section 2.3.4, Figs 2.19/2.20.
    """
    phi: float
    gamma: float
    k: float
    loading: str = 'static'
    construction: str = 'simplified'

    def __post_init__(self):
        if self.phi <= 0 or self.phi > 50:
            raise ValueError(f"phi must be between 0 and 50, got {self.phi}")
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")
        if self.loading not in ('static', 'cyclic'):
            raise ValueError(f"loading must be 'static' or 'cyclic', got '{self.loading}'")
        if self.construction not in ('simplified', 'reese1974'):
            raise ValueError(
                f"construction must be 'simplified' or 'reese1974', got '{self.construction}'")

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
        """Compute soil resistance using the Reese (1974) sand p-y curve.

        Three-part construction: an initial linear branch with slope k*z, a
        1/3-power parabolic softening branch, and the ultimate plateau
        p_ult = A * min(pus, pud) reached at yu = 3b/80 (Reese 1974). The
        returned resistance is the lower envelope of the three branches, which
        is monotonic, concave, and continuous.

        Notes
        -----
        The full Reese (1974) construction additionally fixes an intermediate
        m-point (pm = B*pu) from the B-factor charts and derives a variable
        parabola exponent. This implementation anchors the softening parabola
        at the ultimate point with a 1/3-power exponent -- a recognized,
        chart-free simplification. For the smooth API / O'Neill & Murchison
        equivalent, use ``SandAPI``. (Previously this method was bilinear
        despite the three-part docstring -- LP-1.)
        """
        if z <= 0:
            return 0.0

        sign = _safe_sign(y)
        y_abs = abs(y)
        if y_abs == 0:
            return 0.0

        pus, pud = self.get_pu(z, b)
        pu = min(pus, pud)

        k_init = self.k * z  # initial slope (kN/m of resistance per m deflection)
        if k_init <= 0 or pu <= 0:
            return 0.0

        if self.construction == 'reese1974':
            return sign * self._p_reese1974(y_abs, z, b, pu, k_init)

        # --- simplified (default) construction ---------------------------------
        # Adjustment factor A (COM624P): static A = max(0.9, 3 - 0.8 z/b),
        # cyclic A = 0.9.
        A = 0.9 if self.loading == 'cyclic' else max(0.9, 3.0 - 0.8 * z / b)
        p_ult = A * pu
        # Deflection at ultimate resistance (Reese 1974).
        yu = 3.0 * b / 80.0
        # Lower envelope of: initial linear, 1/3-power softening parabola
        # (passing through the ultimate point at yu), and the ultimate plateau.
        p_linear = k_init * y_abs
        p_parabola = p_ult * (y_abs / yu) ** (1.0 / 3.0) if yu > 0 else p_ult
        p = min(p_linear, p_parabola, p_ult)

        return sign * p

    def _p_reese1974(self, y_abs: float, z: float, b: float,
                     pu: float, k_init: float) -> float:
        """Full four-segment Reese (1974) sand p-y resistance (magnitude).

        Initial linear (k*z*y) -> parabola (C*y^(1/n)) -> straight m-segment
        (pm..pu_curve) -> plateau. The A/B chart coefficients set the ultimate and
        m-point resistances; the initial linear governs (via the lower envelope)
        below yk where it undercuts the parabola.
        """
        cyclic = self.loading == 'cyclic'
        zb = z / b if b > 0 else 0.0
        pu_curve = _reese_A(zb, cyclic) * pu          # ultimate at yu = 3b/80
        pm = _reese_B(zb, cyclic) * pu                # m-point at ym = b/60
        yu = 3.0 * b / 80.0
        ym = b / 60.0
        if pm <= 0 or ym <= 0 or yu <= ym:
            # Geometry degenerate (zero diameter etc.); linear-to-plateau.
            return min(k_init * y_abs, pu_curve)
        if pu_curve < pm:
            # A < B: m-point above the ultimate -- unphysical, and impossible
            # with the corrected A/B tables (A_c >= B_c enforced). Guard defense-
            # in-depth: warn and fall back to a linear-to-plateau curve.
            warnings.warn(
                "Reese (1974) sand p-y: A < B at z/b=%.2f (%s) gives an m-point "
                "above the ultimate; check the A/B coefficient tables. Falling "
                "back to a linear-to-plateau curve." % (
                    zb, 'cyclic' if cyclic else 'static'))
            return min(k_init * y_abs, pu_curve)
        if pu_curve == pm:
            # A == B (deep cyclic asymptote): zero-length m-segment. The parabola
            # runs straight to the ultimate/plateau -- no m-segment, no division
            # by zero. Anchor a 1/3-power parabola at the ultimate point.
            p_upper = (pu_curve * (y_abs / yu) ** (1.0 / 3.0)
                       if y_abs <= yu else pu_curve)
            return min(k_init * y_abs, p_upper)
        m = (pu_curve - pm) / (yu - ym)               # slope of the m-segment
        n = pm / (m * ym)                             # parabola exponent 1/n
        C = pm / ym ** (1.0 / n)
        # Composite parabola -> m-segment -> plateau (upper branch).
        if y_abs <= ym:
            p_upper = C * y_abs ** (1.0 / n)
        elif y_abs <= yu:
            p_upper = pm + m * (y_abs - ym)
        else:
            p_upper = pu_curve
        # Initial linear governs until it meets the parabola at yk (lower envelope).
        return min(k_init * y_abs, p_upper)

    def get_py_curve(self, z: float, b: float, n_points: int = 50,
                     y_max_factor: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a complete p-y curve at depth z.

        ``y_max`` defaults to ``y_max_factor * yu`` (``yu = 3b/80``) so the
        plateau beyond ultimate is captured.
        """
        yu = 3.0 * b / 80.0
        y_max = max(y_max_factor, 1.0) * yu
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
    """p-y curves for weak rock per Reese (1997).

    Reese's interim criterion for rock with an unconfined compressive
    strength of 0.5 to 5 MPa, as printed in FHWA GEC 9 (FHWA-HIF-18-031,
    2018), Appendix A.5, pp. 230-232, Eqs. A-33 to A-42, with z_r the
    depth below the rock surface:

        p_ur = alpha_r qu b (1 + 1.4 z_r/b)   0 <= z_r <= 3b    (A-33)
        p_ur = 5.2 alpha_r qu b               z_r >= 3b         (A-34)
        alpha_r = 1 - (2/3)(RQD/100): 1/3 at RQD 100 %, rising
                  linearly to 1.0 at RQD 0                       (p. 231)
        K_ir = k_ir E_m                                          (A-35)
        k_ir = 100 + 400 z_r/(3b)   0 <= z_r <= 3b;  500 below   (A-36, A-37)
        p = K_ir y                         y <= y_A              (A-38)
        p = (p_ur/2)(y/y_rm)^0.25          y >= y_A, p <= p_ur   (A-39)
        p = p_ur                                                 (A-40)
        y_rm = k_rm b                                            (A-41)
        y_A = [p_ur / (2 y_rm^0.25 K_ir)]^1.333                  (A-42)

    GEC 9 prints A-38 with a lower-case k_ir and A-41 as y_m; its Figure
    A-12, A-35 and A-42 make the initial slope K_ir and the deflection the
    y_rm of A-39, which is what is used here. The exponent of A-42 is taken
    as 4/3, the exact crossing of A-38 and A-39 (printed 1.333).

    Parameters
    ----------
    qu : float
        Uniaxial compressive strength of intact rock, sigma_ci (kPa).
    Er : float
        Modulus of the rock mass, E_m (kPa): from a pressuremeter or
        dilatometer test (GEC 9 p. 231). GEC 10 (FHWA-NHI-18-024) p. 9-31
        modelled a decomposed quartzite with 150 qu.
    gamma_r : float
        Unit weight of rock (kN/m^3). Not used: the criterion's p_ur and
        K_ir do not depend on it (GEC 9 Eqs. A-33 to A-42).
    RQD : float
        Rock quality designation (%), 0-100: sets alpha_r. Default 100,
        which gives the printed lower bound alpha_r = 1/3.
    loading : str
        'static' or 'cyclic'. Default 'static'. The criterion is for static
        loading (GEC 9 p. 92), so 'cyclic' gives the same curve.
    krm : float
        k_rm, the strain at 50 % of the ultimate (GEC 9 p. 232: 0.00005 to
        0.0005; GEC 10 p. 9-31: the lower value gives a stiffer initial
        response). Default 0.0005, the upper end: the softer curve.
    rock_top : float, optional
        Depth of the rock surface below the ground (m); z_r = z - rock_top.
        Default None: 0 for the model alone, and in a
        :class:`~lateral_pile.analysis.LateralPileAnalysis` the top of the
        weak-rock layers that hold this one.

    References
    ----------
    Reese, L.C. (1997). J. Geotech. & Geoenviron. Eng., 123(11), 1010-1017.
    FHWA GEC 9 (FHWA-HIF-18-031, 2018), Appendix A.5, pp. 230-232.
    FHWA GEC 10 (FHWA-NHI-18-024, 2018), Section 9.3.3.4.4, p. 9-31.
    """
    qu: float
    Er: float
    gamma_r: float = 22.0
    RQD: float = 100.0
    loading: str = 'static'
    krm: float = 0.0005
    rock_top: Optional[float] = None

    def __post_init__(self):
        if self.qu <= 0:
            raise ValueError(f"qu must be positive, got {self.qu}")
        if self.Er <= 0:
            raise ValueError(f"Er must be positive, got {self.Er}")
        if not 0.0 <= self.RQD <= 100.0:
            raise ValueError(f"RQD must be between 0 and 100 %, got {self.RQD}")
        if self.krm <= 0:
            raise ValueError(f"krm must be positive, got {self.krm}")
        if self.rock_top is not None and self.rock_top < 0:
            raise ValueError(f"rock_top must be >= 0, got {self.rock_top}")
        if self.loading not in ('static', 'cyclic'):
            raise ValueError(f"loading must be 'static' or 'cyclic', got '{self.loading}'")

    @property
    def alpha_r(self) -> float:
        """Strength reduction factor: 1/3 at RQD 100 %, rising linearly to
        1.0 at RQD 0 (GEC 9 p. 231)."""
        return 1.0 - (2.0 / 3.0) * self.RQD / 100.0

    def depth_in_rock(self, z: float) -> float:
        """z_r, the depth below the rock surface (m), for a depth z below
        the ground; 0 above the rock surface."""
        return max(z - (self.rock_top or 0.0), 0.0)

    def get_pu(self, z: float, b: float) -> float:
        """Ultimate resistance p_ur (kN/m), GEC 9 Eqs. A-33 and A-34.

        p_ur = alpha_r * qu * b * (1 + 1.4 * z_r/b)  for z_r <= 3b
        p_ur = 5.2 * alpha_r * qu * b                for z_r > 3b
        """
        zb = self.depth_in_rock(z) / b if b > 0 else 0.0
        if zb <= 3.0:
            return self.alpha_r * self.qu * b * (1.0 + 1.4 * zb)
        return 5.2 * self.alpha_r * self.qu * b

    def get_Kir(self, z: float, b: float) -> float:
        """Initial slope K_ir = k_ir * E_m (kN/m^2), GEC 9 Eqs. A-35 to
        A-37: k_ir = 100 + 400 z_r/(3b) to z_r = 3b, 500 below."""
        zr = self.depth_in_rock(z)
        kir = 100.0 + 400.0 * zr / (3.0 * b) if zr <= 3.0 * b else 500.0
        return kir * self.Er

    def get_p(self, y: float, z: float, b: float) -> float:
        """Rock resistance p (kN/m) at deflection y: the three segments of
        GEC 9 Eqs. A-38 to A-40."""
        if z < 0:
            return 0.0

        sign = _safe_sign(y)
        y_abs = abs(y)
        if y_abs == 0:
            return 0.0

        pur = self.get_pu(z, b)
        Kir = self.get_Kir(z, b)
        y_rm = self.krm * b
        y_A = (pur / (2.0 * y_rm ** 0.25 * Kir)) ** (4.0 / 3.0)
        if y_abs <= y_A:
            p = Kir * y_abs
        else:
            p = 0.5 * pur * (y_abs / y_rm) ** 0.25
        return sign * min(p, pur)

    def get_py_curve(self, z: float, b: float, n_points: int = 50,
                     y_max_factor: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a complete p-y curve at depth z, to ``y_max_factor``
        times the deflection at which p reaches p_ur (16 y_rm, or p_ur/K_ir
        where the first segment reaches p_ur first)."""
        y_u = max(16.0 * self.krm * b, self.get_pu(z, b) / self.get_Kir(z, b))
        y_array = np.linspace(0, y_max_factor * y_u, n_points)
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
