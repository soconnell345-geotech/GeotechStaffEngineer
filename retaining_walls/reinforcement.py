"""
MSE wall reinforcement properties.

Defines reinforcement types and the built-in reinforcements, each as a
GEC-11 Vol II worked example prints it (GM-262, GM-267, GM-271).

All units are SI: kN/m, m, kPa.

References:
    FHWA GEC-11, Chapter 3
"""

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

#: One value for the wall, or one per reinforcement level from the top down
#: (GM-261).
PerLevel = Union[float, Sequence[float]]

#: The allowable stress of a steel bar mat as a fraction of Fy, for the
#: module's rupture check, which compares Tal x Rc with the unfactored Tmax
#: at a factor of safety of 1.0: GEC-11 Vol I p. 4-77 (checklist IX.5), "Is
#: the correct value of nominal strength of steel used? (e.g., 0.55 Fy for
#: strips and 0.48Fy for bar mats)". The LRFD check of Example E4 (Vol II
#: p. E4-15, Table E4-7.4) takes phi 0.65 (a grid on a rigid facing, Vol I
#: Table 4-7, p. 4-48) against Tmax times 1.35: 0.65 / 1.35 = 0.48 (GM-262).
BAR_MAT_ALLOWABLE_FY = 0.48

#: The same for a steel strip: "0.55 Fy for strips" (GEC-11 Vol I p. 4-77);
#: phi 0.75 (Table 4-7, p. 4-48) against Tmax times 1.35, 0.75 / 1.35 =
#: 0.56 (GM-262 review).
STRIP_ALLOWABLE_FY = 0.55


def steel_strip_Tal(E_corroded: float, Fy: float,
                    fraction: float = 1.0) -> float:
    """Tal of a steel strip per metre of its own width (kN/m; GM-262
    review): ``fraction`` x Fy Ac / b, GEC-11 Vol I Eq. 3-11a (p. 3-22),
    with Ac = b Ec (Fig. 3-5a, p. 3-24), so Fy Ec whatever the width; Ec
    the strip's thickness after corrosion (Eq. 3-10). ``fraction`` as
    :func:`steel_grid_Tal`; :data:`STRIP_ALLOWABLE_FY` the allowable.

    Parameters: E_corroded (m); Fy (kPa)."""
    if E_corroded <= 0 or Fy <= 0:
        raise ValueError("steel_strip_Tal needs a positive corroded "
                         "thickness and Fy")
    return fraction * Fy * E_corroded


def steel_grid_Tal(n_longitudinal: int, D_corroded: float, Fy: float,
                   b: float, fraction: float = 1.0) -> float:
    """Tal of a steel grid or bar mat per metre of its own width b (kN/m;
    GM-262): ``fraction`` x Fy Ac / b, GEC-11 Vol I Eq. 3-11a (p. 3-22),
    with Ac = (No. of longitudinal bars) x pi D*^2 / 4, D* the bar's
    diameter after corrosion (Fig. 3-5, p. 3-24, which prints D*^2 / 4 with
    no pi; Vol II Example E4, p. E4-15, takes pi (0.318 in.)^2 / 4 = 0.0795
    in.^2 a wire). ``fraction`` 1 gives GEC-11's nominal long-term strength,
    which its LRFD check multiplies by the resistance factor (Table 4-7, p.
    4-48); :data:`BAR_MAT_ALLOWABLE_FY` the allowable this module's rupture
    check takes.

    Parameters: n_longitudinal, the number of longitudinal bars in the grid;
    D_corroded (m); Fy (kPa); b (m), centre to centre of the outside
    longitudinal bars (p. 3-12)."""
    if n_longitudinal < 1 or D_corroded <= 0 or Fy <= 0 or b <= 0:
        raise ValueError(
            "steel_grid_Tal needs at least one longitudinal bar and a "
            "positive corroded diameter, Fy and grid width b")
    Ac = n_longitudinal * math.pi * D_corroded ** 2 / 4.0
    return fraction * Fy * Ac / b


def geosynthetic_Tal(T_ult: float, RF_ID: float, RF_CR: float,
                     RF_D: float) -> float:
    """Tal of a geosynthetic, the long-term strength Tult / RF with RF =
    RFID x RFCR x RFD (GEC-11 Vol I Eq. 3-12, p. 3-28; GM-271), in the
    units of ``T_ult``. The rupture check takes it at a factor of safety
    of 1.5 (``mse.rupture_FOS_required``)."""
    if T_ult <= 0 or min(RF_ID, RF_CR, RF_D) < 1:
        raise ValueError("geosynthetic_Tal needs a positive Tult and "
                         "reduction factors of at least 1")
    return T_ult / (RF_ID * RF_CR * RF_D)


def _per_level(value, name):
    """A list given per level as a tuple of floats; a single value as it
    is (GM-261)."""
    if value is None or not isinstance(value, (list, tuple)):
        return value
    if not value:
        raise ValueError(f"{name} per level needs at least one value")
    try:
        return tuple(float(v) for v in value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} per level is a list of numbers, one per reinforcement "
            f"level from the top down; got {list(value)!r}") from None


def _values(value) -> Tuple[float, ...]:
    """Every value of a per-level input, or the one value, or none."""
    if value is None:
        return ()
    return value if isinstance(value, tuple) else (value,)


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
        reinforcement, Tal (kN/m). The rupture check takes Tal x Rc, the
        strength per metre of wall (p. 4-47: "the product of the nominal
        long-term strength, coverage ratio, and applicable resistance
        factor"), against the unfactored Tmax at a factor of safety of 1.0,
        so Tal here is an allowable: for steel a fraction of GEC-11 Vol I
        Eq. 3-11a (p. 3-22), Fy Ac / b with Ac after corrosion, 0.55 Fy for
        strips and 0.48 Fy for bar mats (p. 4-77; :func:`steel_strip_Tal`,
        :func:`steel_grid_Tal`, GM-262). A strength per strip or per mat
        (Ptal, Eq. 3-11b) is Ptal / b (GM-256).
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
    coverage_ratio : float or sequence of float, optional
        Coverage ratio Rc = b/Sh (GEC-11 Vol I Eq. 3-1, p. 3-12), given
        directly, 0 < Rc <= 1: one value for the wall, or one per
        reinforcement level from the top down (GM-261). Default None: worked
        out from ``width`` and ``horizontal_spacing``, or 1 for a
        geosynthetic given neither (a continuous layer, p. 3-13); a steel
        strip or grid given neither is refused (``mse.coverage_ratio``,
        GM-256).
    transverse_spacing : float, optional
        Transverse-bar (grid) spacing St (m) for steel bar-mat / welded-grid
        reinforcement (the ``St`` in F* = 20·t/St). Only used for
        ``metallic_grid``. Default 0 (then the per-call ``t_over_St`` /
        F* default applies).
    horizontal_spacing : float or sequence of float, optional
        Centre-to-centre horizontal spacing Sh (m) between the strips,
        sheets or grids of one level (GEC-11 Vol I p. 3-12): one value for
        the wall, or one per reinforcement level from the top down, as Vol
        II Example E3 changes it with depth (Table E3-7.3, p. E3-19: 2
        strips a 5 ft panel at Levels 1 to 8, 3 at Levels 9 to 12; GM-261).
        Default None.
    basis : str, optional
        Where ``Tallowable`` comes from, for the result's choices. Default
        None (not stated).
    """
    name: str
    type: str
    Tallowable: float
    width: Optional[float] = None
    Fy: float = 0.0
    thickness: float = 0.0
    coverage_ratio: Optional[PerLevel] = None
    transverse_spacing: float = 0.0
    horizontal_spacing: Optional[PerLevel] = None
    basis: Optional[str] = None

    def __post_init__(self):
        valid_types = ("metallic_strip", "metallic_grid", "geogrid",
                       "geotextile", "geosynthetic")
        if self.type not in valid_types:
            raise ValueError(f"type must be one of {valid_types}, got '{self.type}'")
        if self.Tallowable <= 0:
            raise ValueError(f"Tallowable must be positive, got {self.Tallowable}")
        if self.width is not None and not self.width > 0:
            raise ValueError(f"width must be positive, got {self.width}")
        self.horizontal_spacing = _per_level(self.horizontal_spacing,
                                             "horizontal_spacing")
        self.coverage_ratio = _per_level(self.coverage_ratio,
                                         "coverage_ratio")
        for value in _values(self.horizontal_spacing):
            if not value > 0:
                raise ValueError(
                    f"horizontal_spacing must be positive, got {value}")
        for value in _values(self.coverage_ratio):
            if not 0 < value <= 1:
                raise ValueError(
                    "coverage_ratio is Rc = b/Sh, the reinforcement's width "
                    "over its horizontal spacing (GEC-11 Vol I Eq. 3-1): "
                    f"more than 0 and at most 1; got {value}")

    @property
    def per_level(self) -> bool:
        """True when the spacing or the coverage ratio is given per
        reinforcement level (GM-261)."""
        return (isinstance(self.horizontal_spacing, tuple)
                or isinstance(self.coverage_ratio, tuple))

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


# Built-in reinforcements, each as a GEC-11 Vol II worked example prints it
# (DESIGN.md), not a manufacturer's catalog. A steel strip or grid has no
# coverage ratio of its own: the wall's horizontal spacing Sh sets it, Rc =
# b/Sh (GEC-11 Vol I Eq. 3-1), and a call gives Sh or Rc (GM-256).
_IN = 0.0254
_FT = 0.3048
_KSI = 6894.757                 # kPa
_LB_PER_FT = 4.4482216 / _FT / 1000.0    # kN/m
_FY_GRADE_65 = 65.0 * _KSI

# GEC-11 Vol II Example E3's strip, read on the page (GM-262 review):
# "Grade 65 (Fy = 65 ksi), 1.969 in. x 0.157 in. (50 mm wide x 4 mm) thick
# galvanized steel ribbed strips"; after 75 years "EC = 0.157 in. - 0.055
# in. = 0.102 in." (p. E3-16: the zinc gone in 16 years, then 0.47
# mil/year/side for 59).
_E3_EC = 0.102 * _IN

STRIP_BASIS = (
    "GEC-11 Vol II Example E3's strip, 'Grade 65 (Fy = 65 ksi), 1.969 in. "
    "x 0.157 in. (50 mm wide x 4 mm) thick galvanized steel ribbed strips' "
    "(p. E3-16), the ribbed strip GEC-11 Vol I gives as 'currently "
    "commercially available' (p. 2-16): b = 50 mm. After E3's 75-year design "
    "life Ec = 0.157 in. - 0.055 in. = 0.102 in. (2.59 mm; p. E3-16: the "
    "zinc gone in 16 years, then 0.47 mil/year/side of steel for 59). Tal = "
    "0.55 Fy Ec per metre of strip width (Vol I Eq. 3-11a, p. 3-22, with Ac "
    "= b Ec, Fig. 3-5a, p. 3-24; '0.55 Fy for strips', p. 4-77, the "
    "allowable for a check at a factor of safety of 1.0 on the unfactored "
    "Tmax): 638.6 kN/m of strip width, 31.9 kN a strip (E3: Tn = 65 ksi x "
    "0.200 in.^2 = 13.00 k a strip, 0.55 Tn = 7.15 k = 31.8 kN). Another "
    "strip, design life or backfill is a custom reinforcement "
    "(reinforcement_Tallowable)")

RIBBED_STEEL_STRIP_50x4 = Reinforcement(
    name="Ribbed steel strip 50 x 4 mm (GEC-11 Vol II Example E3)",
    type="metallic_strip",
    # kN/m of strip width, times Rc per metre of wall (GM-256)
    Tallowable=steel_strip_Tal(_E3_EC, _FY_GRADE_65, STRIP_ALLOWABLE_FY),
    width=0.050,                    # 1.969 in.; E3's b = 0.164 ft (p. E3-18)
    Fy=_FY_GRADE_65,
    thickness=0.004,
    basis=STRIP_BASIS,
)

# Retained for the retired foundry/ wrapper only, which imports it; no
# longer offered: the tool refuses its name and gives the 50 mm strip
# instead (GM-267). Its 75 mm width was the tool's own.
_STRIP_75_BASIS = (
    "Tal = 0.55 Fy Ec per metre of strip width (GEC-11 Vol I Eq. 3-11a, "
    "p. 3-22, with Ac = b Ec, Fig. 3-5a, p. 3-24; '0.55 Fy for strips', p. "
    "4-77, the allowable for a check at a factor of safety of 1.0 on the "
    "unfactored Tmax), Ec = 0.102 in. (2.59 mm), the 4 mm (0.157 in.) "
    "galvanized ribbed strip of GEC-11 Vol II Example E3 after 75 years of "
    "corrosion (p. E3-16), Fy = 65 ksi (Grade 65, as E3): 638.6 kN/m of "
    "strip width, 47.9 kN a 75 mm strip. The 75 mm width is the tool's "
    "own: GEC-11 gives the ribbed strips 'currently commercially available' "
    "as '2 in. (50 mm) wide and 5/32-inch (4 mm) thick' (Vol I p. 2-16), "
    "E3's. Another width, design life or backfill is a custom "
    "reinforcement (reinforcement_Tallowable)")

RIBBED_STEEL_STRIP_75x4 = Reinforcement(
    name="Ribbed Steel Strip 75x4mm",
    type="metallic_strip",
    # kN/m of strip width, times Rc per metre of wall (GM-256). It was 43.1
    # kN a strip / 0.075 m = 574.7, of no recorded source (GM-262 review).
    Tallowable=steel_strip_Tal(_E3_EC, _FY_GRADE_65, STRIP_ALLOWABLE_FY),
    width=0.075,
    Fy=_FY_GRADE_65,
    thickness=0.004,
    basis=_STRIP_75_BASIS,
)

# GEC-11 Vol II (FHWA-NHI-10-025) Example E4's bar mat at Levels 1, 3 and
# 4, read on the pages (GM-262): "4W11 + W11x0.5' which means a bar mat
# with 4 W11 longitudinal wires spaced at 0.5 ft on centers with W11
# transverse wires spaced at 0.5 ft on centers" (p. E4-18); a W11 wire is
# D = 0.374 in., 0.318 in. after 75 years (p. E4-14: galvanized, the AASHTO
# (2007) Art. 11.10.6.4.2a rates, 0.47 mil/year/side of carbon steel after
# the zinc's 16 years), 0.0795 in.^2; Grade 65 steel, Tn = 65 ksi x 0.0795
# in.^2 = 5.17 k a wire (p. E4-15).
_W11_D = 0.374 * _IN
_W11_D75 = 0.318 * _IN
_W11_MAT_WIRES = 4
_W11_MAT_WIDTH = round((_W11_MAT_WIRES - 1) * 0.5 * _FT, 4)  # b 1.5 ft

W11_MAT_BASIS = (
    "GEC-11 Vol II Example E4's bar mat at Levels 1, 3 and 4, '4W11 + "
    "W11x0.5'' (p. E4-18): 4 W11 longitudinal wires at 0.5 ft (0.152 m), "
    "b = 1.5 ft (0.457 m) between the outside wires, W11 transverse wires "
    "at 0.5 ft; Tal = 0.48 Fy Ac / b (GEC-11 Vol I Eq. 3-11a, p. 3-22; "
    "'0.48Fy for bar mats', p. 4-77, the allowable for a check at a factor "
    "of safety of 1.0 on the unfactored Tmax), Ac = 4 pi (0.318 in.)^2 / 4, "
    "the W11 wire after E4's 75 years of corrosion (p. E4-14), Fy = 65 ksi "
    "(p. E4-15). The 0.48 Fy is for a grid connected to a rigid facing, as "
    "E4's precast panels: phi 0.65 (p. 3-23; Table 4-7, p. 4-48) against "
    "Tmax times 1.35, 0.65 / 1.35 = 0.48. E4 uses this mat to Level 4 "
    "(9.37 ft, 2.86 m deep) and 4W15 mats from Level 5 (11.87 ft, 3.62 m) "
    "down, where four W11 wires carry 0.95 of the factored Tmax (Table "
    "E4-7.4, p. E4-18); a 4W15 mat, W15 0.381 in. after 75 years (p. "
    "E4-14), is Tal = 0.48 Fy 4 pi (0.381 in.)^2 / 4 / 1.5 ft = 138.4 kN/m "
    "of mat width, with W11 transverse wires at 1 ft or 1.5 ft. Another "
    "mat, design life or backfill is a custom reinforcement "
    "(reinforcement_Tallowable)")

WELDED_WIRE_GRID_W11 = Reinforcement(
    name="Bar mat 4W11 + W11x0.5' (GEC-11 Vol II Example E4)",
    type="metallic_grid",
    # kN/m of mat width, times Rc per metre of wall (GM-256). It was 32.0,
    # of no recorded source (Fy x ONE uncorroded W11 wire), about a third
    # of this mat's 96.4 (FOS_rupture about 0.36 at E4's Level 4; GM-262).
    Tallowable=steel_grid_Tal(_W11_MAT_WIRES, _W11_D75, _FY_GRADE_65,
                              _W11_MAT_WIDTH, BAR_MAT_ALLOWABLE_FY),
    width=_W11_MAT_WIDTH,
    Fy=_FY_GRADE_65,
    thickness=_W11_D,               # transverse W11, t = 0.374 in.
    transverse_spacing=0.5 * _FT,   # St = 6 in. -> t/St = 0.0623 (p. E4-15)
    basis=W11_MAT_BASIS,
)

# GEC-11 Vol II (FHWA-NHI-10-025) Example E1's geogrids, read on the pages
# (GM-271). Table E1-7.3 (p. E1-15), "Geogrid Nominal and Factored
# Resistance": Tult 3,000, 6,000 and 9,000 lb/ft for GG-I, GG-II and
# GG-III, RFCR 1.85 and RFD 1.15 for all three, RFID 1.3, 1.3 and 1.2, Tal
# 1,085, 2,169 and 3,525 lb/ft, Tr = 0.90 Tal. Table E1-7.1 (p. E1-11)
# lists the three grades' ultimate tensile strengths as 6,000, 9,000 and
# 12,000 lb/ft; E1's Tal and its rupture CDRs (Table E1-7.4, p. E1-16)
# follow Table E1-7.3, and so do these. Step 7.8 (p. E1-16): "F* = Pullout
# resistance factor, = 0.45 for these geogrids", "alpha = Scale correction
# factor, = 0.8 for these geogrids", C = 2, Rc = 1. Type "geogrid": GEC-11
# Vol I Table 3-6's alpha 0.8 for extensible grids (p. 3-17).
_E1_RF_CR = 1.85
_E1_RF_D = 1.15
#: grade: (Tult lb/ft, RFID, Tal lb/ft as printed), Table E1-7.3
_E1_GEOGRIDS = {"GG-I": (3000.0, 1.3, 1085), "GG-II": (6000.0, 1.3, 2169),
                "GG-III": (9000.0, 1.2, 3525)}
#: grade: ultimate tensile strength lb/ft, Table E1-7.1
_E1_TABLE_7_1 = {"GG-I": 6000, "GG-II": 9000, "GG-III": 12000}

_E1_GEOGRID_BASIS = (
    "GEC-11 Vol II Example E1's geogrid {grade}, as Table E1-7.3 (p. E1-15) "
    "prints it: Tult {Tult:,.0f} lb/ft, RFCR 1.85, RFD 1.15, RFID {RFID:g}; "
    "Tal = Tult / (RFID x RFCR x RFD) (Vol I Eq. 3-12, p. 3-28) = "
    "{Tal_lb:,} lb/ft, {Tal:.4g} kN/m. Table E1-7.1 (p. E1-11) lists "
    "{grade}'s ultimate tensile strength as {T71:,} lb/ft; E1's Tal and its "
    "rupture CDRs (Table E1-7.4, p. E1-16) follow Table E1-7.3, as this "
    "does. For pullout E1 takes F* = 0.45 (0.67 tan 34 deg) and alpha = 0.8 "
    "'for these geogrids', with C = 2 and Rc = 1 (Step 7.8, p. E1-16); this "
    "tool takes the same for a geogrid, F* = 0.67 tan phi_backfill (Vol I "
    "Eq. 3-8, p. 3-21) and alpha 0.8 (Table 3-6, p. 3-17). The reduction "
    "factors are E1's; a product's own Tult and RF for the wall's backfill "
    "and design life are a custom reinforcement (reinforcement_Tallowable "
    "with reinforcement_type 'geogrid')")


def _e1_geogrid(grade: str) -> Reinforcement:
    T_ult, RF_ID, Tal_lb = _E1_GEOGRIDS[grade]
    Tal = geosynthetic_Tal(T_ult, RF_ID, _E1_RF_CR, _E1_RF_D) * _LB_PER_FT
    return Reinforcement(
        name=f"Geogrid {grade} (GEC-11 Vol II Example E1)",
        type="geogrid",
        Tallowable=Tal,                 # kN/m, Tult / RF
        basis=_E1_GEOGRID_BASIS.format(
            grade=grade, Tult=T_ult, RFID=RF_ID, Tal_lb=Tal_lb, Tal=Tal,
            T71=_E1_TABLE_7_1[grade]),
    )


GEOGRID_GEC11_E1_GG1 = _e1_geogrid("GG-I")      # Tal 15.83 kN/m
GEOGRID_GEC11_E1_GG2 = _e1_geogrid("GG-II")     # Tal 31.66 kN/m
GEOGRID_GEC11_E1_GG3 = _e1_geogrid("GG-III")    # Tal 51.45 kN/m

# Retained for the retired foundry/ wrapper only, which imports them; no
# longer offered: the tool refuses their names and gives E1's geogrids or a
# custom Tal instead (GM-271). Their Tal had no recorded source.
_GEOGRID_BASIS = (
    "the long-term strength after installation damage, creep and "
    "durability, Tult / RF (GEC-11 Vol I Eq. 3-12, p. 3-28): {Tal:g} kN/m "
    "(about {lbft:,.0f} lb/ft). No source is "
    "recorded for this value and it has not been checked against the "
    "product's published strength; a product's own Tult / RF for the "
    "wall's backfill and design life is a custom reinforcement "
    "(reinforcement_Tallowable)")

GEOGRID_UX1600 = Reinforcement(
    name="Geogrid UX1600 (HDPE)",
    type="geogrid",
    Tallowable=21.9,  # kN/m (LTDS after creep, damage, durability)
    basis=_GEOGRID_BASIS.format(Tal=21.9, lbft=round(21.9e3 * _FT / 4.4482216, -1)),
)

GEOGRID_UX1700 = Reinforcement(
    name="Geogrid UX1700 (HDPE)",
    type="geogrid",
    Tallowable=29.2,  # kN/m
    basis=_GEOGRID_BASIS.format(Tal=29.2, lbft=round(29.2e3 * _FT / 4.4482216, -1)),
)
