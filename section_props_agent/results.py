"""Result dataclasses for the section properties agent."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SectionPropertiesResult:
    """Cross-section properties (mm-based units).

    Attributes
    ----------
    shape : str
        Shape name ('rectangle', 'circle', 'chs', 'rhs', 'i_section',
        'polygon').
    area_mm2 : float
        Cross-sectional area.
    perimeter_mm : float
        Section perimeter.
    cx_mm, cy_mm : float
        Centroid location (from the shape's construction origin).
    ixx_mm4, iyy_mm4, ixy_mm4 : float
        Second moments of area about centroidal x/y axes.
    zxx_plus_mm3, zxx_minus_mm3, zyy_plus_mm3, zyy_minus_mm3 : float
        Elastic section moduli (extreme-fibre, both directions).
    sxx_mm3, syy_mm3 : float
        Plastic section moduli.
    rx_mm, ry_mm : float
        Radii of gyration.
    j_mm4 : float
        St. Venant torsion constant.
    gamma_mm6 : float or None
        Warping constant (from the warping analysis).
    i11_mm4, i22_mm4 : float
        Principal second moments of area.
    phi_deg : float
        Principal axis angle (degrees from x-axis).
    """
    shape: str = ""
    area_mm2: float = 0.0
    perimeter_mm: float = 0.0
    cx_mm: float = 0.0
    cy_mm: float = 0.0
    ixx_mm4: float = 0.0
    iyy_mm4: float = 0.0
    ixy_mm4: float = 0.0
    zxx_plus_mm3: float = 0.0
    zxx_minus_mm3: float = 0.0
    zyy_plus_mm3: float = 0.0
    zyy_minus_mm3: float = 0.0
    sxx_mm3: float = 0.0
    syy_mm3: float = 0.0
    rx_mm: float = 0.0
    ry_mm: float = 0.0
    j_mm4: float = 0.0
    gamma_mm6: Optional[float] = None
    i11_mm4: float = 0.0
    i22_mm4: float = 0.0
    phi_deg: float = 0.0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  CROSS-SECTION PROPERTIES",
            "=" * 60,
            f"  Shape:              {self.shape}",
            f"  Area:               {self.area_mm2:,.1f} mm^2",
            f"  Centroid (x, y):    ({self.cx_mm:.1f}, {self.cy_mm:.1f}) mm",
            f"  Ixx (centroidal):   {self.ixx_mm4:,.3e} mm^4",
            f"  Iyy (centroidal):   {self.iyy_mm4:,.3e} mm^4",
            f"  Zxx (min elastic):  {min(self.zxx_plus_mm3, self.zxx_minus_mm3):,.3e} mm^3",
            f"  Zyy (min elastic):  {min(self.zyy_plus_mm3, self.zyy_minus_mm3):,.3e} mm^3",
            f"  Sxx (plastic):      {self.sxx_mm3:,.3e} mm^3",
            f"  Syy (plastic):      {self.syy_mm3:,.3e} mm^3",
            f"  rx, ry:             {self.rx_mm:.1f}, {self.ry_mm:.1f} mm",
            f"  J (torsion):        {self.j_mm4:,.3e} mm^4",
        ]
        if self.gamma_mm6 is not None:
            lines.append(f"  Gamma (warping):    {self.gamma_mm6:,.3e} mm^6")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = {
            "shape": self.shape,
            "area_mm2": float(self.area_mm2),
            "perimeter_mm": float(self.perimeter_mm),
            "cx_mm": float(self.cx_mm),
            "cy_mm": float(self.cy_mm),
            "ixx_mm4": float(self.ixx_mm4),
            "iyy_mm4": float(self.iyy_mm4),
            "ixy_mm4": float(self.ixy_mm4),
            "zxx_plus_mm3": float(self.zxx_plus_mm3),
            "zxx_minus_mm3": float(self.zxx_minus_mm3),
            "zyy_plus_mm3": float(self.zyy_plus_mm3),
            "zyy_minus_mm3": float(self.zyy_minus_mm3),
            "sxx_mm3": float(self.sxx_mm3),
            "syy_mm3": float(self.syy_mm3),
            "rx_mm": float(self.rx_mm),
            "ry_mm": float(self.ry_mm),
            "j_mm4": float(self.j_mm4),
            "i11_mm4": float(self.i11_mm4),
            "i22_mm4": float(self.i22_mm4),
            "phi_deg": float(self.phi_deg),
        }
        if self.gamma_mm6 is not None:
            d["gamma_mm6"] = float(self.gamma_mm6)
        return d
