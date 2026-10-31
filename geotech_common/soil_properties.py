"""
Common soil property correlations, as the manuals we hold print them.

Field-test corrections (raw field N -> N60 -> (N1)60) and correlations
between SPT blow counts and soil parameters. They are for preliminary
design and should be checked against laboratory tests for final design
(GEC-12 p. 129: correlating N with the undrained strength of clays "is
crude and unreliable for final design").

Checked against the print (GM-295, 2026-10-31):

* FHWA NHI-06-088 (Soils and Foundations Vol. I, Dec 2006): Eq 3-2,
  N60 = (Ef/60) Nmeas (p. 3-55); Eq 3-3, Peck et al. (1974) C_N
  (p. 3-56); Tables 4-1 and 4-2 (after Peck et al. 1974, p. 4-5).
* FHWA GEC-12 Vol I (FHWA-NHI-16-009): Eq. 5-1 to 5-4 (p. 108), Tables
  5-5 and 5-6 (p. 123), Eq. 5-14 to 5-16 (p. 124), Eq. 5-19 (p. 128),
  Table 5-9 (p. 129).
* FHWA GEC-5 (FHWA-NHI-16-072, Nov 2016): Eq 5.10 (p. 5-31); Table 7-5
  and Eq 7.22 (p. 7-54); Eq 7.23 (p. 7-55); pa = 100 kPa (p. 6-19).
* FHWA GEC-6 Figure 5-18 (p. 90): Liao & Whitman's C_N, "2.0 max."
* UFC 3-220-10 (16 Jan 2026) §8-2.1.2 and §8-3.4: Table 8-3 (p. 396),
  Eq 8-2 and 8-3 (p. 397), Eq 8-4 (p. 398), Eq 8-5, Eq 8-6 and Fig 8-4
  (p. 399), Eq 8-26 (p. 428), Fig 8-32 (p. 429).
* FHWA NHI-06-089 (Vol. II): Eq 10-2 to 10-5 (pp. 10-10 to 10-12).
* EN 1997-2:2007 Annex F, Tables F.1 and F.2 (p. 125).

Not in the reference library, so not checked against a print: Youd et al.
(2001) Table 2 (the borehole, rod-length and sampler factors and the 1.7
cap on Liao & Whitman's C_N), EN ISO 22476-3 (the "iso" C_N) and Bolton
(1986).
"""

import math
import warnings
from typing import Optional, Sequence


def _interp(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    """Linear between the nodes (xs ascending); x within them."""
    for k in range(1, len(xs)):
        if x <= xs[k]:
            t = (x - xs[k - 1]) / (xs[k] - xs[k - 1])
            return ys[k - 1] + t * (ys[k] - ys[k - 1])
    return ys[-1]


# ---------------------------------------------------------------------------
# SPT -> friction angle
# ---------------------------------------------------------------------------

#: Atmospheric pressure pa (kPa) as GEC-5 takes it ("100 kPa = 14.7 psi",
#: p. 6-19).
PA_KPA = 100.0

#: UFC 3-220-10 Fig 8-4 (p. 399) draws Peck et al. (1974) and Hatanaka &
#: Uchida (1996) for N1,60 from 0 to 60.
FIG_8_4_N_MAX = 60.0

#: GEC-5 Table 7-5 (after Meyerhof 1956, p. 7-54; GEC-12 Table 5-6, UFC
#: 3-220-10 Table 8-3): the SPT N band edges and phi' (deg) at them.
MEYERHOF_N = (4.0, 10.0, 30.0, 50.0)
MEYERHOF_PHI = (30.0, 35.0, 40.0, 45.0)

PHI_METHODS = ("peck", "hatanaka_uchida", "kulhawy_mayne", "kulhawy_chen",
               "shioi_fukui_roads", "shioi_fukui_buildings", "meyerhof")


def spt_to_phi(N: float, method: str = "peck",
               sigma_v: Optional[float] = None) -> float:
    """Drained friction angle phi' (deg) of a clean sand from an SPT blow
    count, by one of the correlations the manuals print. Each takes the
    blow count its print names.

    Parameters
    ----------
    N : float
        The SPT blow count the method takes (below).
    method : str, optional
        - "peck" (default): Peck et al. (1974), the curve of UFC 3-220-10
          Fig 8-4 (p. 399) as Wolff (1989) approximated it, Eq 8-5:
          phi' = 27.1 + 0.3 N1,60 - 0.00054 N1,60^2. ``N`` is N1,60
          (corrected for 60 % energy and to 1 atm overburden). UFC: "this
          correlation is conservative" (Ameratunga et al. 2016).
        - "hatanaka_uchida": GEC-5 Eq 7.23 (p. 7-55; UFC 3-220-10 Eq 8-6):
          phi' = sqrt(15.4 (N1)60) + 20, "for an average state-of-practice
          hammer with 60 percent efficiency". ``N`` is (N1)60.
        - "kulhawy_mayne": Kulhawy & Mayne's (1990) fit to Schmertmann's
          (1975) chart, GEC-5 Eq 7.22 (p. 7-54; GEC-12 Eq. 5-14, UFC
          3-220-10 Eq 8-4): tan phi' = [N60 / (12.2 + 20.3 sigma'v/pa)]^0.34,
          pa = 100 kPa. ``N`` is N60; needs ``sigma_v``. GEC-5: "somewhat
          conservative, especially for depths less than 6 ft".
        - "kulhawy_chen": Kulhawy & Chen (2007), sands and gravels,
          GEC-12 Eq. 5-16 (p. 124): phi' = 27.5 + 9.2 log10 (N1)60
          (r^2 = 0.356, n = 57). ``N`` is (N1)60, above 0.
        - "shioi_fukui_roads" / "shioi_fukui_buildings": Shioi & Fukui
          (1982), UFC 3-220-10 Eq 8-2, phi' = sqrt(15 N60) + 15 (roads and
          bridges), and Eq 8-3, phi' = 0.3 N60 + 27 (buildings) (p. 397).
          ``N`` is N60.
        - "meyerhof": GEC-5 Table 7-5 (after Meyerhof 1956, p. 7-54; GEC-12
          Table 5-6, UFC 3-220-10 Table 8-3): 30, 35, 40 and 45 deg at the
          band edges N = 4, 10, 30 and 50, read linearly between them (the
          reading is this function's). The table gives only "< 30" below
          N = 4 and "> 45" above 50, so those are refused. UFC takes N60,
          GEC-5 the uncorrected N. UFC's note under Table 8-3 takes 5 deg
          less for a clayey sand and 5 more for a gravelly sand (GEC-12
          p. 123 credits it to GEC-5's 2002 edition); GEC-5's own note
          gives only N = 15 + (N' - 15)/2 for N' > 15 in a saturated very
          fine or silty sand.
    sigma_v : float, optional
        Vertical effective stress at the test depth (kPa), for
        "kulhawy_mayne".

    Returns
    -------
    float
        phi' (degrees).

    Past Fig 8-4's N1,60 = 60, "peck" and "hatanaka_uchida" hold the angle
    at 60's and warn. All are for relatively clean sands: not for
    micaceous sands, and for gravelly soils only conservatively (GEC-5
    p. 7-55). Before GM-295 "peck" was a hand-drawn piecewise line on N60
    (26 to 42 deg, up to 2.4 deg above Eq 8-5) and "meyerhof" was
    25 + 0.3 N60, which no manual we hold prints.
    """
    if N < 0:
        raise ValueError(f"SPT N-value must be non-negative, got {N}")
    method = method.lower()
    if method in ("peck", "hatanaka_uchida"):
        if N > FIG_8_4_N_MAX:
            warnings.warn(
                f"N1,60 = {N:g} is past UFC 3-220-10 Fig 8-4's N1,60 axis "
                f"(0-{FIG_8_4_N_MAX:g}); phi' was held at N1,60 = "
                f"{FIG_8_4_N_MAX:g}.")
            N = FIG_8_4_N_MAX
        if method == "peck":
            return 27.1 + 0.3 * N - 0.00054 * N ** 2
        return math.sqrt(15.4 * N) + 20.0
    if method == "kulhawy_mayne":
        if sigma_v is None or sigma_v <= 0:
            raise ValueError(
                "method 'kulhawy_mayne' needs sigma_v, the vertical "
                "effective stress at the test depth (kPa)")
        x = N / (12.2 + 20.3 * sigma_v / PA_KPA)
        return math.degrees(math.atan(x ** 0.34))
    if method == "kulhawy_chen":
        if N <= 0:
            raise ValueError(
                "method 'kulhawy_chen' takes the log of (N1)60, which must "
                "be above 0")
        return 27.5 + 9.2 * math.log10(N)
    if method == "shioi_fukui_roads":
        return math.sqrt(15.0 * N) + 15.0
    if method == "shioi_fukui_buildings":
        return 0.3 * N + 27.0
    if method == "meyerhof":
        if not MEYERHOF_N[0] <= N <= MEYERHOF_N[-1]:
            raise ValueError(
                f"N = {N:g}: GEC-5 Table 7-5 (Meyerhof 1956) gives phi' only "
                f"as < 30 deg below N = 4 and > 45 deg above N = 50; use "
                f"another method or judgment")
        return _interp(N, MEYERHOF_N, MEYERHOF_PHI)
    raise ValueError(
        f"Unknown method '{method}'. Options: {', '.join(PHI_METHODS)}")


# ---------------------------------------------------------------------------
# SPT -> undrained shear strength
# ---------------------------------------------------------------------------

#: 1 kgf/cm^2 in kPa. UFC 3-220-10 Fig 8-32 prints Hara's su = 0.297 N^0.72
#: and draws it through about 800 kPa at N = 100, so in kgf/cm^2.
KGF_CM2_KPA = 98.0665

#: UFC 3-220-10 Fig 8-31 (p. 428) draws Eq 8-26 for PI from 15 to 62.
FIG_8_31_PI = (15.0, 62.0)

CU_METHODS = ("terzaghi_peck", "hara", "stroud_butler")


def spt_to_cu(N60: float, method: str = "terzaghi_peck",
              PI: Optional[float] = None) -> float:
    """Undrained shear strength su (kPa) of a clay from an SPT blow count,
    for preliminary estimates only (GEC-12 p. 129).

    Parameters
    ----------
    N60 : float
        SPT blow count corrected for 60 % energy.
    method : str, optional
        - "terzaghi_peck" (default): su = 6.25 N60 kPa, the consistency
          table of NHI-06-088 Table 4-2 (after Peck et al. 1974, p. 4-5)
          with su = qu/2: qu 25, 50 and 100 kPa at N60 = 2, 4 and 8, so
          6.25 kPa a blow. At its 15 and 30 edges the table gives 6.67 a
          blow (qu 200 and 400 kPa).
        - "hara": Hara et al. (1974), from triaxial compression tests,
          su = 0.297 N^0.72 kgf/cm^2 (UFC 3-220-10 Fig 8-32, p. 429), here
          29.13 N^0.72 kPa.
        - "stroud_butler": Stroud & Butler (1975) for overconsolidated
          clays, Carter & Bentley's (2016) fit, UFC 3-220-10 Eq 8-26
          (p. 428): su/N = 8910/PI^3 + 4.36 kPa. Needs ``PI``; Fig 8-31
          draws it for PI 15 to 62 and a warning says when PI is outside.
          GEC-12 Eq. 5-19 (p. 128) prints Stroud's su = f1 N60 pa/100 with
          f1 = 4.5 at PI 50 and 5.5 at PI 15.
    PI : float, optional
        Plasticity index (%), for "stroud_butler".

    Returns
    -------
    float
        su (kPa).
    """
    if N60 < 0:
        raise ValueError(f"SPT N-value must be non-negative, got {N60}")

    method = method.lower()
    if method == "terzaghi_peck":
        return 6.25 * N60
    if method == "hara":
        return 0.297 * KGF_CM2_KPA * N60 ** 0.72
    if method == "stroud_butler":
        if PI is None or PI <= 0:
            raise ValueError(
                "method 'stroud_butler' needs PI, the plasticity index (%)")
        if not FIG_8_31_PI[0] <= PI <= FIG_8_31_PI[1]:
            warnings.warn(
                f"PI = {PI:g} is outside UFC 3-220-10 Fig 8-31's "
                f"{FIG_8_31_PI[0]:g}-{FIG_8_31_PI[1]:g}, where Eq 8-26 is "
                f"drawn.")
        return N60 * (8910.0 / PI ** 3 + 4.36)
    raise ValueError(
        f"Unknown method '{method}'. Options: {', '.join(CU_METHODS)}")


# ---------------------------------------------------------------------------
# SPT -> relative density
# ---------------------------------------------------------------------------

#: EN 1997-2:2007 Table F.1 (p. 125), normally consolidated natural sand
#: deposits: (N1)60 band edges and the density index ID (%) at them.
EN_TABLE_F1_N = (0.0, 3.0, 8.0, 25.0, 42.0, 58.0)
EN_TABLE_F1_ID = (0.0, 15.0, 35.0, 65.0, 85.0, 100.0)

#: EN 1997-2 Table F.2 (p. 125, Skempton 1986), normally consolidated fine
#: sands: (N1)60 / ID^2 by age.
SKEMPTON_AGEING = {"laboratory": 35.0, "recent_fill": 40.0,
                   "natural": 55.0}


def spt_to_relative_density(N1_60: float, method: str = "en1997",
                            age: str = "natural") -> float:
    """Relative density (density index ID, %) of a sand from (N1)60.

    Parameters
    ----------
    N1_60 : float
        SPT blow count corrected for 60 % energy and to 1 atm overburden.
    method : str, optional
        - "en1997" (default): EN 1997-2:2007 Table F.1 (p. 125), normally
          consolidated natural sand deposits: ID 0, 15, 35, 65, 85 and
          100 % at (N1)60 = 0, 3, 8, 25, 42 and 58, read linearly between
          them (the reading is this function's); 100 % past 58.
        - "skempton": EN 1997-2 Table F.2 (p. 125, Skempton 1986),
          normally consolidated fine sands: ID = sqrt((N1)60 / C) with
          C = 35 in laboratory tests, 40 in recent fills and 55 in natural
          deposits (``age``); at most 100 %.
    age : str, optional
        "laboratory", "recent_fill" or "natural" (default), for
        "skempton".

    Returns
    -------
    float
        ID (percent, 0-100).

    EN 1997-2 F.1: for fine sands N is reduced in the ratio 55:60 and for
    coarse sands increased 65:60 (4); the correlations are for
    predominantly silica sands (7). NHI-06-088 Table 4-1 (N60), UFC
    3-220-10 Table 8-4 (N1,60, after Mitchell 1981) and GEC-5 Table 7-5
    (N, after Meyerhof 1956) print class tables like Table F.1. Before
    GM-295 this took N60 and sigma'v, sqrt(N60 / (0.46 sigma'v/pa)) x 100,
    and returned 100 % for any N60 above about 0.5.
    """
    if N1_60 < 0:
        raise ValueError(f"SPT N-value must be non-negative, got {N1_60}")
    method = method.lower()
    if method == "en1997":
        if N1_60 >= EN_TABLE_F1_N[-1]:
            return 100.0
        return _interp(N1_60, EN_TABLE_F1_N, EN_TABLE_F1_ID)
    if method == "skempton":
        if age not in SKEMPTON_AGEING:
            raise ValueError(
                f"age must be one of {list(SKEMPTON_AGEING)}, got '{age}'")
        return min(100.0 * math.sqrt(N1_60 / SKEMPTON_AGEING[age]), 100.0)
    raise ValueError(
        f"Unknown method '{method}'. Options: en1997, skempton")


# ---------------------------------------------------------------------------
# Earth pressure coefficients
# ---------------------------------------------------------------------------

def phi_to_Ka(phi_deg: float) -> float:
    """Rankine active earth pressure coefficient, NHI-06-089 Eq 10-4
    (p. 10-12): Ka = (1 - sin phi')/(1 + sin phi') = tan^2(45 - phi'/2),
    for a vertical wall, level ground and no wall friction.

    Parameters
    ----------
    phi_deg : float
        Drained friction angle (degrees).

    Returns
    -------
    float
        Active earth pressure coefficient Ka.
    """
    if phi_deg < 0 or phi_deg > 50:
        raise ValueError(f"Friction angle must be 0-50 degrees, got {phi_deg}")
    phi_rad = math.radians(phi_deg)
    return math.tan(math.pi / 4.0 - phi_rad / 2.0) ** 2


def phi_to_Kp(phi_deg: float) -> float:
    """Rankine passive earth pressure coefficient, NHI-06-089 Eq 10-5
    (p. 10-12): Kp = tan^2(45 + phi'/2), on the same terms as Ka.

    Parameters
    ----------
    phi_deg : float
        Drained friction angle (degrees).

    Returns
    -------
    float
        Passive earth pressure coefficient Kp.
    """
    if phi_deg < 0 or phi_deg > 50:
        raise ValueError(f"Friction angle must be 0-50 degrees, got {phi_deg}")
    phi_rad = math.radians(phi_deg)
    return math.tan(math.pi / 4.0 + phi_rad / 2.0) ** 2


def phi_to_K0(phi_deg: float, OCR: float = 1.0) -> float:
    """At-rest earth pressure coefficient.

    K0 = 1 - sin phi' (Jaky 1944), NHI-06-089 Eq 10-2 (p. 10-10), for a
    near normally consolidated soil; with ``OCR`` above 1,
    K0 = (1 - sin phi') OCR^sin phi' (Schmidt 1966, Omega = sin phi' after
    Mayne & Kulhawy 1982), Eq 10-3 (p. 10-11). NHI: "sufficiently accurate
    for normally to lightly overconsolidated soils provided the
    overconsolidation ratio has been evaluated from laboratory
    consolidation testing".

    Parameters
    ----------
    phi_deg : float
        Drained friction angle (degrees).
    OCR : float, optional
        Overconsolidation ratio (default 1).

    Returns
    -------
    float
        At-rest earth pressure coefficient K0.
    """
    if phi_deg < 0 or phi_deg > 50:
        raise ValueError(f"Friction angle must be 0-50 degrees, got {phi_deg}")
    if OCR < 1.0:
        raise ValueError(f"OCR must be at least 1, got {OCR}")
    sin_phi = math.sin(math.radians(phi_deg))
    return (1.0 - sin_phi) * OCR ** sin_phi


# ---------------------------------------------------------------------------
# SPT field corrections (raw field N  ->  N60  ->  (N1)60)
# ---------------------------------------------------------------------------

#: Rod-length correction C_R per Youd et al. (2001), Table 2 (not held).
_ROD_LENGTH_BANDS = (
    (3.0, 0.75),    # rod length < 3 m
    (4.0, 0.80),    # 3-4 m
    (6.0, 0.85),    # 4-6 m
    (10.0, 0.95),   # 6-10 m
    (float("inf"), 1.00),  # 10-30 m
)


def spt_energy_correction(
    N_field: float,
    energy_ratio_pct: float = 60.0,
    rod_length_m: float = 10.0,
    borehole_diameter_mm: float = 100.0,
    sampler_correction: float = 1.0,
) -> dict:
    """Correct a raw field SPT blow count to the standardized N60.

    N60 = N_field * CE * CB * CR * CS. CE = ER/60 is NHI-06-088 Eq 3-2
    (p. 3-55; GEC-12 Eq. 5-1). CB, CR and CS are Youd et al. (2001) Table
    2, which the reference library does not hold; at their defaults
    (100 mm borehole, 10 m of rod, standard sampler) all three are 1.

    Parameters
    ----------
    N_field : float
        Raw field SPT blow count N (blows/300 mm).
    energy_ratio_pct : float, optional
        Measured hammer energy ratio ER (percent of theoretical free-fall
        energy). CE = ER/60. Default 60 (already-standardized energy).
        NHI-06-088 p. 3-55: about 35 to 85 for cathead systems with donut
        or safety hammers, about 60 on average; 80 to 100 for automatic
        trip hammers; without a measurement AASHTO takes 60 for rope and
        cathead and 80 for automatic hammers.
    rod_length_m : float, optional
        Rod length from hammer to sampler (m); sets CR per Youd et al.
        (2001) Table 2 bands (0.75 below 3 m rising to 1.0 beyond 10 m).
        Default 10.0 (CR = 1.0).
    borehole_diameter_mm : float, optional
        Borehole diameter; CB = 1.0 for 65-115 mm (default), 1.05 for
        150 mm, 1.15 for 200 mm.
    sampler_correction : float, optional
        CS: 1.0 for a standard sampler (default); 1.1-1.3 for a sampler
        used without liners (Youd et al. 2001).

    Returns
    -------
    dict
        Keys "CE", "CB", "CR", "CS", "N60".

    References
    ----------
    Youd et al. (2001). "Liquefaction resistance of soils: summary report
    from the 1996 NCEER and 1998 NCEER/NSF workshops." J. Geotech.
    Geoenviron. Eng., 127(10), Table 2.
    """
    if N_field < 0:
        raise ValueError(f"N_field must be non-negative, got {N_field}")
    if not 30.0 <= energy_ratio_pct <= 120.0:
        raise ValueError(
            f"energy_ratio_pct outside credible 30-120 range: {energy_ratio_pct}")
    if rod_length_m < 0:
        raise ValueError(f"rod_length_m must be non-negative, got {rod_length_m}")
    if not 1.0 <= sampler_correction <= 1.3:
        raise ValueError(
            f"sampler_correction (CS) must be 1.0-1.3, got {sampler_correction}")
    CE = energy_ratio_pct / 60.0
    if borehole_diameter_mm <= 115.0:
        CB = 1.0
    elif borehole_diameter_mm <= 150.0:
        CB = 1.05
    elif borehole_diameter_mm <= 200.0:
        CB = 1.15
    else:
        raise ValueError(
            f"borehole_diameter_mm above 200 mm not covered: {borehole_diameter_mm}")
    CR = next(cr for upper, cr in _ROD_LENGTH_BANDS if rod_length_m < upper)
    CS = sampler_correction
    return {"CE": CE, "CB": CB, "CR": CR, "CS": CS,
            "N60": N_field * CE * CB * CR * CS}


#: 1 tsf (2000 lb/ft^2) in kPa: NHI-06-088 Eq 3-3 takes p_o in tsf.
TSF_KPA = 95.76052

#: The overburden corrections spt_overburden_correction offers, and each
#: one's default cap on C_N.
OVERBURDEN_METHODS = {"liao_whitman": 1.7, "peck": 2.0, "iso": 2.0}


def spt_overburden_correction(
    N: float,
    sigma_vo_eff_kPa: float,
    method: str = "liao_whitman",
    CN_cap: float | None = None,
) -> dict:
    """Overburden correction C_N for granular soils: N1 = C_N * N.

    Pass the raw field N to get N1, or an energy-corrected N60 to get
    (N1)60 -- the correction is multiplicative either way (NHI-06-088
    Eq 3-3, GEC-12 Eq. 5-2).

    Parameters
    ----------
    N : float
        SPT blow count (field N or N60).
    sigma_vo_eff_kPa : float
        Vertical effective stress at the test depth (kPa).
    method : str, optional
        - "liao_whitman" (default): C_N = (pa / sigma'v)^0.5 with
          pa = 100 kPa, Liao & Whitman (1986): GEC-5 Eq 5.10 (p. 5-31),
          GEC-12 Eq. 5-4 with n = 0.5 for sands (p. 108). Capped at 1.7
          per Youd et al. (2001), which the reference library does not
          hold; GEC-6 Figure 5-18 (p. 90) draws the same correction (pa
          1 tsf) to "2.0 max."
        - "peck": C_N = 0.77 log10(20 / p_o), p_o in tsf, at most 2.0,
          Peck et al. (1974): NHI-06-088 Eq 3-3 (p. 3-56) and GEC-12 Eq.
          5-3 (p. 108, in ksf), which GEC-12 calls "the recommended
          correction in AASHTO specifications". It reaches zero at
          p_o = 20 tsf (1915 kPa), where it is refused.
        - "iso": C_N = sqrt(98 / sigma'v) per EN ISO 22476-3 guidance
          (not held); capped at 2.0.
    CN_cap : float, optional
        Override the default cap (1.7 liao_whitman, 2.0 peck and iso).

    Returns
    -------
    dict
        Keys "CN", "N1".

    References
    ----------
    Liao, S. & Whitman, R. (1986). "Overburden correction factors for SPT
    in sand." J. Geotech. Eng., 112(3). Youd et al. (2001) Table 2 (cap).
    EN ISO 22476-3 (SPT).
    """
    if N < 0:
        raise ValueError(f"N must be non-negative, got {N}")
    if sigma_vo_eff_kPa <= 0:
        raise ValueError(
            f"sigma_vo_eff_kPa must be positive, got {sigma_vo_eff_kPa}")
    if method not in OVERBURDEN_METHODS:
        raise ValueError(
            f"Unknown method '{method}' ({' | '.join(OVERBURDEN_METHODS)})")
    if method == "liao_whitman":
        CN = math.sqrt(PA_KPA / sigma_vo_eff_kPa)
    elif method == "peck":
        CN = 0.77 * math.log10(20.0 * TSF_KPA / sigma_vo_eff_kPa)
        if CN <= 0:
            raise ValueError(
                f"sigma_vo_eff_kPa = {sigma_vo_eff_kPa:g}: Peck et al. "
                f"(1974) C_N reaches zero at 20 tsf "
                f"({20.0 * TSF_KPA:.0f} kPa)")
    else:
        CN = math.sqrt(98.0 / sigma_vo_eff_kPa)
    cap = OVERBURDEN_METHODS[method] if CN_cap is None else CN_cap
    CN = min(CN, cap)
    return {"CN": CN, "N1": CN * N}


def spt_n1_60(
    N_field: float,
    sigma_vo_eff_kPa: float,
    energy_ratio_pct: float = 60.0,
    rod_length_m: float = 10.0,
    borehole_diameter_mm: float = 100.0,
    sampler_correction: float = 1.0,
    overburden_method: str = "liao_whitman",
) -> dict:
    """Full field-N -> (N1)60 correction chain (energy then overburden).

    Convenience wrapper: spt_energy_correction then
    spt_overburden_correction, (N1)60 = C_N N60 (NHI-06-088 Eq 3-3,
    GEC-12 Eq. 5-2). Returns every intermediate factor so the calculation
    can be reported step by step.

    Returns
    -------
    dict
        Keys "CE", "CB", "CR", "CS", "N60", "CN", "N1_60".
    """
    e = spt_energy_correction(N_field, energy_ratio_pct, rod_length_m,
                              borehole_diameter_mm, sampler_correction)
    o = spt_overburden_correction(e["N60"], sigma_vo_eff_kPa,
                                  method=overburden_method)
    return {**e, "CN": o["CN"], "N1_60": o["N1"]}


# ---------------------------------------------------------------------------
# Bolton (1986) stress-dilatancy
# ---------------------------------------------------------------------------

def stress_dilatancy_bolton(
    relative_density: float,
    p_eff_kPa: float,
    Q: float = 10.0,
    R: float = 1.0,
    stress_condition: str = "triaxial",
) -> dict:
    """Bolton (1986) relative dilatancy index and peak-friction excess.

    I_R = D_r * (Q - ln p') - R

    phi_max - phi_crit = 3 * I_R   (triaxial)
    phi_max - phi_crit = 5 * I_R   (plane strain; = 0.8 * psi_max)
    (-d eps_v / d eps_1)_max = 0.3 * I_R

    Not checked against a print: the reference library does not hold
    Bolton (1986). The triaxial dilation angle taken equal to the friction
    excess is this function's own approximation.

    Parameters
    ----------
    relative_density : float
        Relative density D_r as a FRACTION (0-1).
    p_eff_kPa : float
        Mean effective stress p' (kPa).
    Q, R : float, optional
        Bolton's mineralogy fitting constants (defaults 10 and 1 for
        silica sand).
    stress_condition : str, optional
        "triaxial" (default) or "plane_strain".

    Returns
    -------
    dict
        Keys "IR", "phi_excess_deg", "dilation_angle_deg",
        "max_dilation_rate". I_R is clipped at 0 (no dilation predicted
        for loose states / high stress); Bolton recommends 0 <= I_R <= 4.

    References
    ----------
    Bolton, M.D. (1986). "The strength and dilatancy of sands."
    Geotechnique, 36(1), 65-78.
    """
    if not 0.0 <= relative_density <= 1.0:
        raise ValueError(
            f"relative_density must be a 0-1 fraction, got {relative_density}")
    if p_eff_kPa <= 0:
        raise ValueError(f"p_eff_kPa must be positive, got {p_eff_kPa}")
    IR = relative_density * (Q - math.log(p_eff_kPa)) - R
    IR = max(IR, 0.0)
    if stress_condition == "triaxial":
        phi_excess = 3.0 * IR
        dilation_angle = phi_excess       # psi ~= dphi for triaxial
    elif stress_condition == "plane_strain":
        phi_excess = 5.0 * IR
        dilation_angle = phi_excess / 0.8  # dphi = 0.8 * psi_max
    else:
        raise ValueError(
            f"Unknown stress_condition '{stress_condition}' "
            "(triaxial | plane_strain)")
    return {"IR": IR, "phi_excess_deg": phi_excess,
            "dilation_angle_deg": dilation_angle,
            "max_dilation_rate": 0.3 * IR}
