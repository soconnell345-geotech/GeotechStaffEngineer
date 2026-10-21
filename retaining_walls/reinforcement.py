"""
MSE wall reinforcement properties.

Defines reinforcement types and built-in common products.

All units are SI: kN/m, m, kPa.

References:
    FHWA GEC-11, Chapter 3
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Reinforcement:
    """MSE wall reinforcement properties.

    Parameters
    ----------
    name : str
        Product/designation name.
    type : str
        "metallic_strip", "metallic_grid", "geogrid", "geotextile", or
        "geosynthetic". The type sets the scale effect correction factor
        alpha of the pullout resistance, GEC-11 Vol I Table 3-6 (p. 3-17):
        1.0 for steel strips and grids, 0.8 for a geogrid, 0.6 for a
        geotextile; a "geosynthetic" of no stated kind has no row and needs
        a tested alpha (``mse.pullout_alpha``, GM-256).
    Tallowable : float
        Long-term allowable tensile strength per unit width of the
        reinforcement, Tal (kN/m): GEC-11 Vol I Eq. 3-11a (p. 3-22), Fy Ac /
        b for a steel strip or grid. The rupture check takes Tal x Rc, the
        strength per metre of wall (p. 4-47: "the product of the nominal
        long-term strength, coverage ratio, and applicable resistance
        factor"); a strength per strip or per mat (Ptal, Eq. 3-11b) is Ptal
        / b (GM-256).
    width : float, optional
        Gross width b (m) of the strip, sheet or grid; for a grid, from the
        centre to centre of the outside longitudinal bars (GEC-11 Vol I p.
        3-12, Fig. 3-3). With ``horizontal_spacing`` it gives the coverage
        ratio Rc = b/Sh. Default None (not given).
    Fy : float, optional
        Yield strength of steel (kPa). Default 0 (not metallic).
    thickness : float, optional
        Strip thickness (m) for strips; transverse-bar diameter (m) for steel
        bar-mat / welded-grid reinforcement (the ``t`` in the F* = 20·t/St
        bearing-resistance ratio). Default 0.
    coverage_ratio : float, optional
        Coverage ratio Rc = b/Sh (GEC-11 Vol I Eq. 3-1, p. 3-12), given
        directly, 0 < Rc <= 1. Default None: worked out from ``width`` and
        ``horizontal_spacing``, or 1 for a geosynthetic given neither (a
        continuous layer, p. 3-13); a steel strip or grid given neither is
        refused (``mse.coverage_ratio``, GM-256).
    transverse_spacing : float, optional
        Transverse-bar (grid) spacing St (m) for steel bar-mat / welded-grid
        reinforcement (the ``St`` in F* = 20·t/St). Only used for
        ``metallic_grid``. Default 0 (then the per-call ``t_over_St`` /
        F* default applies).
    horizontal_spacing : float, optional
        Centre-to-centre horizontal spacing Sh (m) between the strips,
        sheets or grids of one level (GEC-11 Vol I p. 3-12). Default None.
    """
    name: str
    type: str
    Tallowable: float
    width: Optional[float] = None
    Fy: float = 0.0
    thickness: float = 0.0
    coverage_ratio: Optional[float] = None
    transverse_spacing: float = 0.0
    horizontal_spacing: Optional[float] = None

    def __post_init__(self):
        valid_types = ("metallic_strip", "metallic_grid", "geogrid",
                       "geotextile", "geosynthetic")
        if self.type not in valid_types:
            raise ValueError(f"type must be one of {valid_types}, got '{self.type}'")
        if self.Tallowable <= 0:
            raise ValueError(f"Tallowable must be positive, got {self.Tallowable}")
        for name in ("width", "horizontal_spacing"):
            value = getattr(self, name)
            if value is not None and not value > 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.coverage_ratio is not None \
                and not 0 < self.coverage_ratio <= 1:
            raise ValueError(
                "coverage_ratio is Rc = b/Sh, the reinforcement's width over "
                "its horizontal spacing (GEC-11 Vol I Eq. 3-1): more than 0 "
                f"and at most 1; got {self.coverage_ratio}")

    @property
    def is_metallic(self) -> bool:
        return self.type.startswith("metallic")

    @property
    def is_grid(self) -> bool:
        """True for steel bar-mat / welded-grid reinforcement (uses the
        bar-mat Kr/Ka and F* = 20·t/St curves, not the ribbed-strip curves)."""
        return self.type == "metallic_grid"

    @property
    def t_over_St(self):
        """Transverse-bar diameter / spacing (t/St), unit-free, for the
        steel-grid F* curve. Returns None if the grid geometry is not set."""
        if self.is_grid and self.thickness > 0 and self.transverse_spacing > 0:
            return self.thickness / self.transverse_spacing
        return None


# Built-in reinforcement types. The values are representative of their
# class, not a manufacturer's catalog (DESIGN.md). A steel strip or grid has
# no coverage ratio of its own: the wall's horizontal spacing Sh sets it,
# Rc = b/Sh (GEC-11 Vol I Eq. 3-1), and a call gives Sh or Rc (GM-256).
RIBBED_STEEL_STRIP_75x4 = Reinforcement(
    name="Ribbed Steel Strip 75x4mm",
    type="metallic_strip",
    # Tal per metre of strip width (GEC-11 Eq. 3-11a): the representative
    # allowable load of 43.1 kN per strip (Grade 65 steel, 75 mm x 4 mm,
    # after corrosion) over b = 75 mm. Until GM-256 the module compared 43.1
    # "kN/m" with the tension per metre of wall, at no spacing.
    Tallowable=43.1 / 0.075,
    width=0.075,
    Fy=450000.0,
    thickness=0.004,
)

WELDED_WIRE_GRID_W11 = Reinforcement(
    name="Welded Wire Grid W11xW11",
    type="metallic_grid",
    Tallowable=32.0,  # kN/m of grid width
    Fy=450000.0,
    thickness=0.0095,           # W11 transverse-bar diameter t = 0.374 in
    transverse_spacing=0.1524,  # St = 6 in (densest level) -> t/St = 0.0623
)

# The geogrids are type "geogrid": GEC-11 Vol I Table 3-6's alpha 0.8 for
# extensible grids (p. 3-17). They took alpha 1.0 as "geosynthetic" before
# GM-256.
GEOGRID_UX1600 = Reinforcement(
    name="Geogrid UX1600 (HDPE)",
    type="geogrid",
    Tallowable=21.9,  # kN/m (LTDS after creep, damage, durability)
)

GEOGRID_UX1700 = Reinforcement(
    name="Geogrid UX1700 (HDPE)",
    type="geogrid",
    Tallowable=29.2,  # kN/m
)
