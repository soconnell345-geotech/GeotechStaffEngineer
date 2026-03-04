"""ACI 318-19 concrete footing design.

Checks one-way shear, two-way (punching) shear, flexure, and
development length for a rectangular spread footing supporting
a rectangular column.

All external units: kN, m, kPa, MPa.
Internal computations use MPa and mm for ACI equations.

References:
    ACI 318-19 Building Code Requirements for Structural Concrete
    Sections 7.6.1, 8.5.1, 22.5, 22.6, 25.4
"""

import math
from dataclasses import dataclass
from typing import Optional

from lateral_pile.pile import rebar_diameter


# Rebar areas (mm^2) keyed by designation
_REBAR_AREAS = {
    "#3": 71.0,
    "#4": 129.0,
    "#5": 199.0,
    "#6": 284.0,
    "#7": 387.0,
    "#8": 510.0,
    "#9": 645.0,
    "#10": 819.0,
    "#11": 1006.0,
    "#14": 1452.0,
    "#18": 2581.0,
}


def rebar_area(designation: str) -> float:
    """Return rebar cross-sectional area in mm^2."""
    key = designation.strip()
    if key not in _REBAR_AREAS:
        available = ", ".join(sorted(_REBAR_AREAS.keys(),
                                     key=lambda s: int(s[1:])))
        raise ValueError(
            f"Unknown rebar size '{designation}'. Available: {available}"
        )
    return _REBAR_AREAS[key]


@dataclass
class ConcreteFootingResult:
    """Results from ACI 318-19 concrete footing design.

    Attributes
    ----------
    P_kN : float
        Column factored load (kN).
    B_m, L_m : float
        Footing plan dimensions (m).
    h_m : float
        Footing total thickness (m).
    d_m : float
        Effective depth to centroid of tension steel (m).
    fc_MPa : float
        Concrete compressive strength (MPa).
    fy_MPa : float
        Rebar yield strength (MPa).
    col_b_m, col_d_m : float
        Column cross-section dimensions (m).
    Vu_oneway_kN : float
        Factored one-way shear demand (kN).
    phi_Vc_oneway_kN : float
        Design one-way shear capacity phi*Vc (kN).
    oneway_ok : bool
        True if phi*Vc >= Vu for one-way shear.
    Vu_twoway_kN : float
        Factored two-way (punching) shear demand (kN).
    phi_Vc_twoway_kN : float
        Design two-way shear capacity phi*Vc (kN).
    bo_m : float
        Perimeter of critical section for two-way shear (m).
    twoway_ok : bool
        True if phi*Vc >= Vu for two-way shear.
    Mu_kNm : float
        Factored flexural demand at column face (kN-m).
    As_req_mm2 : float
        Required flexural steel area (mm^2).
    As_min_mm2 : float
        Minimum steel per ACI 318-19 Section 7.6.1 (mm^2).
    As_provided_mm2 : float
        Steel area provided by selected bars (mm^2).
    bar_size : str
        Bar designation used.
    n_bars : int
        Number of bars provided.
    spacing_mm : float
        Center-to-center bar spacing (mm).
    ld_req_mm : float
        Required development length (mm).
    ld_avail_mm : float
        Available development length (mm).
    ld_ok : bool
        True if ld_avail >= ld_req.
    """
    # Inputs echoed
    P_kN: float
    B_m: float
    L_m: float
    h_m: float
    d_m: float
    fc_MPa: float
    fy_MPa: float
    col_b_m: float
    col_d_m: float

    # One-way shear
    Vu_oneway_kN: float
    phi_Vc_oneway_kN: float
    oneway_ok: bool

    # Two-way shear
    Vu_twoway_kN: float
    phi_Vc_twoway_kN: float
    bo_m: float
    twoway_ok: bool

    # Flexure
    Mu_kNm: float
    As_req_mm2: float
    As_min_mm2: float
    As_provided_mm2: float
    bar_size: str
    n_bars: int
    spacing_mm: float

    # Development length
    ld_req_mm: float
    ld_avail_mm: float
    ld_ok: bool

    def summary(self) -> str:
        """Human-readable summary."""
        sep = "=" * 55
        dash = "-" * 55
        lines = [
            sep,
            "  CONCRETE FOOTING DESIGN (ACI 318-19)",
            sep,
            "",
            f"  Footing: {self.B_m:.2f} x {self.L_m:.2f} m, "
            f"h = {self.h_m * 1000:.0f} mm, d = {self.d_m * 1000:.0f} mm",
            f"  Column:  {self.col_b_m:.2f} x {self.col_d_m:.2f} m",
            f"  f'c = {self.fc_MPa:.0f} MPa,  fy = {self.fy_MPa:.0f} MPa",
            f"  Pu = {self.P_kN:.0f} kN",
            "",
            "  ONE-WAY SHEAR",
            "  " + dash,
            f"  Vu  = {self.Vu_oneway_kN:.1f} kN",
            f"  phiVc = {self.phi_Vc_oneway_kN:.1f} kN"
            f"  {'OK' if self.oneway_ok else 'FAIL'}",
            "",
            "  TWO-WAY (PUNCHING) SHEAR",
            "  " + dash,
            f"  Vu  = {self.Vu_twoway_kN:.1f} kN",
            f"  phiVc = {self.phi_Vc_twoway_kN:.1f} kN"
            f"  {'OK' if self.twoway_ok else 'FAIL'}",
            f"  bo = {self.bo_m:.3f} m",
            "",
            "  FLEXURE",
            "  " + dash,
            f"  Mu   = {self.Mu_kNm:.1f} kN-m",
            f"  As_req  = {self.As_req_mm2:.0f} mm2",
            f"  As_min  = {self.As_min_mm2:.0f} mm2",
            f"  As_prov = {self.As_provided_mm2:.0f} mm2"
            f"  ({self.n_bars} {self.bar_size} @ {self.spacing_mm:.0f} mm)",
            "",
            "  DEVELOPMENT LENGTH",
            "  " + dash,
            f"  ld_req   = {self.ld_req_mm:.0f} mm",
            f"  ld_avail = {self.ld_avail_mm:.0f} mm"
            f"  {'OK' if self.ld_ok else 'FAIL'}",
            "",
            sep,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Dict representation for LLM agents."""
        return {
            "P_kN": self.P_kN,
            "B_m": self.B_m,
            "L_m": self.L_m,
            "h_m": self.h_m,
            "d_m": self.d_m,
            "fc_MPa": self.fc_MPa,
            "fy_MPa": self.fy_MPa,
            "col_b_m": self.col_b_m,
            "col_d_m": self.col_d_m,
            "Vu_oneway_kN": round(self.Vu_oneway_kN, 2),
            "phi_Vc_oneway_kN": round(self.phi_Vc_oneway_kN, 2),
            "oneway_ok": self.oneway_ok,
            "Vu_twoway_kN": round(self.Vu_twoway_kN, 2),
            "phi_Vc_twoway_kN": round(self.phi_Vc_twoway_kN, 2),
            "bo_m": round(self.bo_m, 4),
            "twoway_ok": self.twoway_ok,
            "Mu_kNm": round(self.Mu_kNm, 2),
            "As_req_mm2": round(self.As_req_mm2, 1),
            "As_min_mm2": round(self.As_min_mm2, 1),
            "As_provided_mm2": round(self.As_provided_mm2, 1),
            "bar_size": self.bar_size,
            "n_bars": self.n_bars,
            "spacing_mm": round(self.spacing_mm, 1),
            "ld_req_mm": round(self.ld_req_mm, 1),
            "ld_avail_mm": round(self.ld_avail_mm, 1),
            "ld_ok": self.ld_ok,
        }


def design_concrete_footing(
    P_kN: float,
    B_m: float,
    L_m: float,
    h_m: float = 0.5,
    fc_MPa: float = 28.0,
    fy_MPa: float = 420.0,
    col_b_m: float = 0.4,
    col_d_m: float = 0.4,
    cover_mm: float = 75.0,
    bar_size: str = "#8",
) -> ConcreteFootingResult:
    """Design a reinforced concrete spread footing per ACI 318-19.

    Parameters
    ----------
    P_kN : float
        Factored column load (kN).
    B_m : float
        Footing width (m), short direction.
    L_m : float
        Footing length (m), long direction.
    h_m : float
        Footing total thickness (m). Default 0.5.
    fc_MPa : float
        Concrete compressive strength (MPa). Default 28.
    fy_MPa : float
        Rebar yield strength (MPa). Default 420.
    col_b_m : float
        Column width in B direction (m). Default 0.4.
    col_d_m : float
        Column depth in L direction (m). Default 0.4.
    cover_mm : float
        Clear cover to bottom steel (mm). Default 75.
    bar_size : str
        Rebar designation (e.g. "#8"). Default "#8".

    Returns
    -------
    ConcreteFootingResult
    """
    # Convert to mm for ACI equations
    B = B_m * 1000.0          # mm
    L = L_m * 1000.0          # mm
    h = h_m * 1000.0          # mm
    col_b = col_b_m * 1000.0  # mm
    col_d = col_d_m * 1000.0  # mm
    P = P_kN * 1000.0         # N

    db = rebar_diameter(bar_size) * 1000.0  # mm
    Ab = rebar_area(bar_size)               # mm^2

    # Effective depth (to centroid of bottom layer of steel)
    d = h - cover_mm - db / 2.0  # mm
    if d <= 0:
        raise ValueError(
            f"Effective depth d = {d:.0f} mm <= 0. "
            f"Increase footing thickness or reduce cover."
        )

    # Factored net soil pressure (uniform)
    q_u = P / (B * L)  # N/mm^2 = MPa

    # ── ONE-WAY SHEAR (ACI 318-19 Section 22.5) ──
    # Critical section at d from column face (long direction governs)
    cantilever_B = (B - col_b) / 2.0  # mm
    shear_dist_B = cantilever_B - d     # mm from footing edge to crit section
    if shear_dist_B < 0:
        shear_dist_B = 0.0
    Vu_oneway = q_u * shear_dist_B * L  # N

    # Vc = 0.17 * sqrt(f'c) * bw * d   (ACI Eq. 22.5.5.1)
    phi_shear = 0.75
    Vc_oneway = 0.17 * math.sqrt(fc_MPa) * L * d  # N
    phi_Vc_oneway = phi_shear * Vc_oneway

    oneway_ok = phi_Vc_oneway >= Vu_oneway

    # ── TWO-WAY (PUNCHING) SHEAR (ACI 318-19 Section 22.6) ──
    # Critical perimeter at d/2 from column face
    b1 = col_b + d   # mm
    b2 = col_d + d   # mm
    bo = 2.0 * (b1 + b2)  # mm

    # Area within critical perimeter
    A_crit = b1 * b2  # mm^2
    Vu_twoway = P - q_u * A_crit  # N (net shear outside critical area)
    if Vu_twoway < 0:
        Vu_twoway = 0.0

    # Three ACI two-way shear equations (ACI 318-19 Table 22.6.5.2)
    beta_col = max(col_b, col_d) / min(col_b, col_d)  # aspect ratio
    alpha_s = 40.0  # interior column

    Vc_1 = 0.33 * math.sqrt(fc_MPa) * bo * d                          # N
    Vc_2 = (0.17 + 0.33 / beta_col) * math.sqrt(fc_MPa) * bo * d     # N
    Vc_3 = (0.083 * (alpha_s * d / bo + 2)) * math.sqrt(fc_MPa) * bo * d  # N

    Vc_twoway = min(Vc_1, Vc_2, Vc_3)
    phi_Vc_twoway = phi_shear * Vc_twoway

    twoway_ok = phi_Vc_twoway >= Vu_twoway

    # ── FLEXURE (ACI 318-19 Section 8.5.1) ──
    # Critical section at column face
    # Cantilever length in short direction (governs for rectangular)
    cant = cantilever_B  # mm
    # Mu = q_u * L * cant^2 / 2
    Mu = q_u * L * cant ** 2 / 2.0  # N-mm

    # Required steel from Whitney stress block
    phi_flex = 0.9
    As_req = _compute_As_flexure(Mu, fc_MPa, fy_MPa, B, d, phi_flex)

    # Minimum steel per ACI 318-19 Section 7.6.1.1
    As_min_1 = 0.0018 * B * h  # mm^2 (for fy >= 420 MPa)
    As_min_2 = 0.0014 * B * d  # mm^2
    As_min = max(As_min_1, As_min_2)

    As_design = max(As_req, As_min)

    # Bar selection
    n_bars = max(math.ceil(As_design / Ab), 2)
    As_provided = n_bars * Ab
    # Spacing = (B - 2*cover) / (n_bars - 1)
    if n_bars > 1:
        spacing = (B - 2.0 * cover_mm) / (n_bars - 1)
    else:
        spacing = B - 2.0 * cover_mm

    # ── DEVELOPMENT LENGTH (ACI 318-19 Section 25.4.2.4 simplified) ──
    # ld = (fy * psi_t * psi_e / (1.1 * sqrt(f'c))) * db
    # For bottom bars: psi_t = 1.0, psi_e = 1.0 (uncoated), psi_s per bar size
    psi_t = 1.0  # bottom bars
    psi_e = 1.0  # uncoated
    psi_s = 0.8 if int(bar_size[1:]) <= 6 else 1.0
    ld_req = (fy_MPa * psi_t * psi_e * psi_s / (1.1 * math.sqrt(fc_MPa))) * db
    ld_req = max(ld_req, 300.0)  # minimum 300 mm per ACI

    # Available: from column face to footing edge minus cover
    ld_avail = cant - cover_mm

    ld_ok = ld_avail >= ld_req

    # ── Convert back to kN/m units ──
    d_m = d / 1000.0
    bo_m = bo / 1000.0

    return ConcreteFootingResult(
        P_kN=P_kN,
        B_m=B_m,
        L_m=L_m,
        h_m=h_m,
        d_m=d_m,
        fc_MPa=fc_MPa,
        fy_MPa=fy_MPa,
        col_b_m=col_b_m,
        col_d_m=col_d_m,
        Vu_oneway_kN=Vu_oneway / 1000.0,
        phi_Vc_oneway_kN=phi_Vc_oneway / 1000.0,
        oneway_ok=oneway_ok,
        Vu_twoway_kN=Vu_twoway / 1000.0,
        phi_Vc_twoway_kN=phi_Vc_twoway / 1000.0,
        bo_m=bo_m,
        twoway_ok=twoway_ok,
        Mu_kNm=Mu / 1e6,
        As_req_mm2=As_req,
        As_min_mm2=As_min,
        As_provided_mm2=As_provided,
        bar_size=bar_size,
        n_bars=n_bars,
        spacing_mm=spacing,
        ld_req_mm=ld_req,
        ld_avail_mm=ld_avail,
        ld_ok=ld_ok,
    )


def _compute_As_flexure(
    Mu: float,
    fc: float,
    fy: float,
    b: float,
    d: float,
    phi: float,
) -> float:
    """Compute required flexural steel area via Whitney stress block.

    All units: N, mm, MPa.

    Parameters
    ----------
    Mu : float
        Factored moment (N-mm).
    fc : float
        f'c (MPa).
    fy : float
        fy (MPa).
    b : float
        Section width (mm).
    d : float
        Effective depth (mm).
    phi : float
        Strength reduction factor.

    Returns
    -------
    float
        Required As (mm^2). Returns 0 if Mu <= 0.
    """
    if Mu <= 0:
        return 0.0

    # Rn = Mu / (phi * b * d^2)
    Rn = Mu / (phi * b * d ** 2)

    # As = (0.85*f'c/fy) * b * d * (1 - sqrt(1 - 2*Rn/(0.85*f'c)))
    term = 2.0 * Rn / (0.85 * fc)
    if term >= 1.0:
        # Section is over-reinforced; return a large value to flag issue
        # In practice, increase footing thickness
        return 0.85 * fc * b * d / fy

    As = (0.85 * fc / fy) * b * d * (1.0 - math.sqrt(1.0 - term))
    return As
