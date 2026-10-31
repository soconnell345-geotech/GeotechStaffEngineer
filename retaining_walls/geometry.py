"""
Wall geometry definitions for cantilever and MSE retaining walls.

All units are SI: meters, degrees, kPa.

References:
    Das, B.M., Principles of Foundation Engineering, Ch 13
    FHWA GEC-11, Chapter 4
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, List, Sequence, Tuple


@dataclass
class CantileverWallGeometry:
    """Cantilever retaining wall geometry.

    Parameters
    ----------
    wall_height : float
        Total wall height H (m), measured from the UNDERSIDE of the base
        slab (footing) to the top of the stem: H = base_thickness + stem
        height. The backfill surface meets the stem at its top, and the
        active thrust acts on the vertical plane through the back of the
        heel over H (plus the rise of a sloping backfill over the heel,
        ``H_active``). A wall retaining 4.0 m above the top of a 0.5 m
        thick footing is H = 4.5 m.
    base_width : float, optional
        Base slab width B (m). If None, 0.6 H: within the typical 2/5 H to
        2/3 H of FHWA NHI-06-089 Fig. 10-16(a) (p. 10-38, after Teng 1962)
        and the 50 to 70 % of H of UFC 3-220-20 App. B-1.3 (p. 600).
    toe_length : float, optional
        Toe projection from front face of stem (m). If None, H/10, the
        short end of NHI-06-089 Fig. 10-16(a)'s H/10 to H/8 (GM-288).
    stem_thickness_top : float, optional
        Stem thickness at top (m). Default 0.30 (NHI-06-089 Fig. 10-16(a):
        200 mm minimum, 300 mm preferable).
    stem_thickness_base : float, optional
        Stem thickness at base of stem (m). If None, H/10 (NHI-06-089 Fig.
        10-16(a): H/12 to H/10), at least the top thickness (GM-288).
    base_thickness : float, optional
        Base slab thickness (m). If None, H/10 (NHI-06-089 Fig. 10-16(a):
        H/12 to H/10; GM-288).
    stem_batter : str, optional
        Which face of a tapered stem (thicker at its base than at its top)
        is battered: "back" (default; the front face vertical, the backfill
        over the battered back counted as resisting weight, GM-288) or
        "front" (the back face vertical, as UFC 3-220-20 App. B-1 Fig. B-2
        (p. 598) and NHI-06-089 Fig. 10-16(a) and Example 10-3 (p. 10-52)
        draw it).
    embedment_depth : float, optional
        Depth of the underside of the base below the ground in front of
        the wall, Df (m). If None, the base thickness: the ground in front
        at the top of the base, no soil over the toe. Deeper, the soil
        over the toe, (Df - base_thickness) x toe_length, counts as
        resisting weight (UFC 3-220-20 Fig. 4-27, p. 220: "Weight includes
        ... weight of soil above footing"; App. B-1.3, p. 599, W_ff), and
        the passive resistance in front of the wall acts over Df.
    has_shear_key : bool, optional
        Whether wall has a shear key. Default False.
    key_depth : float, optional
        Shear key depth below base slab (m). Default 0.
    backfill_slope : float, optional
        Slope of backfill behind wall (degrees), rising away from the
        wall. Default 0. A descending backfill (below 0) is refused
        (GM-32: it was taken as level without a word).
    surcharge : float, optional
        Uniform live-load surcharge on the backfill (kPa), e.g. traffic.
        Default 0. Its thrust K q H drives every check; over the heel it is
        not a resisting force for sliding and overturning by default
        (``cantilever``, GM-157).
    dead_load_surcharge : float, optional
        Uniform dead-load (permanent) surcharge on the backfill (kPa), e.g.
        a pavement or an approach slab. Default 0. Its thrust drives every
        check, as the live load's does; how its weight over the heel counts
        is a separate choice (``cantilever``, GM-157).
    """
    wall_height: float
    base_width: Optional[float] = None
    toe_length: Optional[float] = None
    stem_thickness_top: float = 0.30
    stem_thickness_base: Optional[float] = None
    base_thickness: Optional[float] = None
    has_shear_key: bool = False
    key_depth: float = 0.0
    backfill_slope: float = 0.0
    surcharge: float = 0.0
    dead_load_surcharge: float = 0.0
    stem_batter: str = "back"
    embedment_depth: Optional[float] = None

    def __post_init__(self):
        if self.wall_height <= 0:
            raise ValueError(f"Wall height must be positive, got {self.wall_height}")
        # Auto-size if not specified: the typical proportions of FHWA
        # NHI-06-089 Fig. 10-16(a) (p. 10-38) and UFC 3-220-20 App. B-1.3
        # (p. 600), GM-288
        if self.base_thickness is None:
            self.base_thickness = round(self.wall_height / 10.0, 2)
        if self.base_thickness <= 0:
            raise ValueError(f"Base thickness must be positive, got {self.base_thickness}")
        if self.has_shear_key and not self.key_depth > 0:
            raise ValueError(
                f"has_shear_key needs the key's depth below the base slab, "
                f"key_depth > 0 (m), got {self.key_depth:g}: the passive "
                f"resistance acts over the key's depth (UFC 3-220-20 Fig. "
                f"4-29, p. 224). Give key_depth, or leave has_shear_key out.")
        if self.backfill_slope < 0:
            raise ValueError(
                f"backfill_slope {self.backfill_slope:g} deg: a backfill "
                "descending away from the wall is not modelled (Rankine's "
                "sloped Ka and the backfill wedge over the heel are for a "
                "surface rising from the top of the stem). Give 0, level "
                "ground, whose thrust is the larger.")
        if self.stem_batter not in ("back", "front"):
            raise ValueError(f"stem_batter must be 'back' or 'front', got "
                             f"{self.stem_batter!r}")

        if self.base_width is None:
            self.base_width = round(0.6 * self.wall_height, 2)
        if self.toe_length is None:
            self.toe_length = round(self.wall_height / 10.0, 2)
        if self.stem_thickness_base is None:
            self.stem_thickness_base = round(
                max(self.wall_height / 10.0, self.stem_thickness_top), 2)
        if self.embedment_depth is not None and not (
                0.0 <= self.embedment_depth <= self.wall_height):
            raise ValueError(
                f"embedment_depth {self.embedment_depth:g} m (the depth of "
                "the underside of the base below the ground in front of the "
                f"wall) must be from 0 to wall_height {self.wall_height:g} m")

    @property
    def Df(self) -> float:
        """Depth of the underside of the base below the ground in front of
        the wall (m): ``embedment_depth``, or the base thickness when it is
        not given (GM-288)."""
        return (self.base_thickness if self.embedment_depth is None
                else self.embedment_depth)

    @property
    def stem_height(self) -> float:
        """Height of stem above base slab (m)."""
        return self.wall_height - self.base_thickness

    @property
    def heel_length(self) -> float:
        """Heel projection behind back face of stem (m)."""
        return self.base_width - self.toe_length - self.stem_thickness_base

    @property
    def H_active(self) -> float:
        """Height for active pressure computation (m).
        Includes backfill slope contribution.
        """
        if self.backfill_slope > 0:
            i_rad = math.radians(self.backfill_slope)
            return self.wall_height + self.heel_length * math.tan(i_rad)
        return self.wall_height


@dataclass
class MSEWallGeometry:
    """MSE (Mechanically Stabilized Earth) wall geometry.

    Parameters
    ----------
    wall_height : float
        Total wall height H (m).
    reinforcement_length : float, optional
        Length of reinforcement strips/grids (m).
        If None, auto-sized to max(0.7*H, 2.5).
    reinforcement_spacing : float, optional
        Vertical spacing between reinforcement levels Sv (m).
        Default 0.60. Without ``level_depths`` the levels sit at Sv (i +
        1/2) below the top, H / Sv of them rounded down (on a wall a whole
        number of spacings high, one a spacing, whatever the floating-point
        rounding of H / Sv; GM-277), each over its contributory height:
        Sv, the bottom level's from the midpoint above it down to the base
        (:attr:`contributory_heights`). A spacing that puts the first level
        at or below the base (Sv / 2 >= H) is refused. Not used with
        ``level_depths``.
    level_depths : sequence of float, optional
        Depths z of the reinforcement levels below the top of the wall (m),
        from the top down, as GEC-11 Vol II's examples print Z (Example E4,
        Table E4-7.1, p. E4-11: 1.87 ft, then 2.5 ft apart to 24.37 ft).
        Each must lie inside the wall (0 < z < H) and below the one above.
        Each level then takes its contributory height (:attr:`
        contributory_heights`, GEC-11 Vol I pp. 4-46, 4-47). Default None:
        the levels at Sv (i + 1/2) (GM-273).
    backfill_slope : float, optional
        Slope of the backfill surface above the wall, beta (degrees from
        horizontal), rising away from the face. Default 0 (level).
    surcharge : float, optional
        Uniform surcharge (kPa). Default 0. With a sloping backfill it is
        taken only with ``broken_backslope_height``: traffic on the level
        ground behind the crest (GEC-11 Example E1).
    broken_backslope_height : float, optional
        For a broken backslope (GEC-11 Fig. 4-4): the height S of the slope
        crest above the top of the wall (m), with level ground behind it.
        Default None: the slope continues (GEC-11 Fig. 4-3).
    """
    wall_height: float
    reinforcement_length: Optional[float] = None
    reinforcement_spacing: float = 0.60
    backfill_slope: float = 0.0
    surcharge: float = 0.0
    broken_backslope_height: Optional[float] = None
    level_depths: Optional[Sequence[float]] = None

    def __post_init__(self):
        if self.wall_height <= 0:
            raise ValueError(f"Wall height must be positive, got {self.wall_height}")
        if self.reinforcement_spacing <= 0:
            raise ValueError(f"Spacing must be positive, got {self.reinforcement_spacing}")
        if self.level_depths is not None:
            self.level_depths = _checked_depths(self.level_depths,
                                                self.wall_height)
        elif self.reinforcement_spacing / 2.0 >= self.wall_height:
            # GM-277: the first level at Sv / 2 would be outside the wall
            raise ValueError(
                f"reinforcement_spacing {self.reinforcement_spacing:g} m "
                f"puts the first level at Sv / 2 = "
                f"{self.reinforcement_spacing / 2.0:g} m, at or below the "
                f"base of the wall (wall_height {self.wall_height:g} m): "
                "give a spacing under twice the wall height, or the "
                "levels' depths (level_depths)")
        if not 0 <= self.backfill_slope < 90:
            raise ValueError(
                f"backfill_slope must be from 0 to below 90 degrees, rising "
                f"away from the wall face; got {self.backfill_slope}")
        if self.broken_backslope_height is not None:
            if self.broken_backslope_height <= 0:
                raise ValueError(
                    "broken_backslope_height (the height of the slope crest "
                    "above the top of the wall) must be positive, got "
                    f"{self.broken_backslope_height}")
            if self.backfill_slope <= 0:
                raise ValueError(
                    "broken_backslope_height needs a backfill_slope: it is "
                    "the height of the crest of that slope")

        if self.reinforcement_length is None:
            self.reinforcement_length = round(max(0.7 * self.wall_height, 2.5), 2)

    @property
    def n_reinforcement_levels(self) -> int:
        """Number of reinforcement levels: one a depth of ``level_depths``,
        else H / Sv rounded down, at least one. A wall a whole number of
        spacings high counts every spacing: 4.8 m at Sv 0.8 m is six levels,
        though 4.8 / 0.8 is 5.999... in floating point (GM-277)."""
        if self.level_depths is not None:
            return len(self.level_depths)
        n = math.floor(self.wall_height / self.reinforcement_spacing
                       + WHOLE_SPACING_TOLERANCE)
        return max(1, n)

    @property
    def reinforcement_depths(self) -> List[float]:
        """Depth of each reinforcement level from top of wall (m): the
        ``level_depths`` given, else Sv (i + 1/2)."""
        if self.level_depths is not None:
            return list(self.level_depths)
        n = self.n_reinforcement_levels
        return [
            round(self.reinforcement_spacing * (i + 0.5), 3)
            for i in range(n)
        ]

    @property
    def contributory_heights(self) -> List[Tuple[float, float]]:
        """Each level's contributory height (z-, z+), depths below the top
        of the wall (m). GEC-11 Vol I p. 4-46: "The contributory height is
        defined as the midpoint between vertically adjacent reinforcement
        elevations, except for the top and bottom layers reinforcement";
        p. 4-47: for those, Sv "is the distance from top or bottom of wall,
        respectively, to the midpoint between the first and second layer"
        (Fig. 4-14). Vol II Example E4 tabulates it as Svt = Z+ - Z- (Table
        E4-7.1, p. E4-11). With ``level_depths`` (GM-273) from the depths
        given; without them, the levels at Sv (i + 1/2), from i Sv to (i +
        1) Sv, the top level's from the top of the wall and the bottom
        level's from (n - 1) Sv down to the base, Sv plus the part of the
        wall below the last whole spacing when H is not a whole number of
        spacings (GM-277)."""
        if self.level_depths is None:
            Sv, n = self.reinforcement_spacing, self.n_reinforcement_levels
            tops = [i * Sv for i in range(n)]
            return list(zip(tops, tops[1:] + [self.wall_height]))
        z = list(self.level_depths)
        mids = [(a + b) / 2.0 for a, b in zip(z, z[1:])]
        return list(zip([0.0] + mids, mids + [self.wall_height]))

    @property
    def whole_spacings(self) -> bool:
        """Without ``level_depths``: whether the wall is a whole number of
        spacings high, so the bottom level's contributory height is Sv
        like every other's (GM-277); False with ``level_depths``."""
        if self.level_depths is not None:
            return False
        Sv = self.reinforcement_spacing
        rest = self.wall_height - self.n_reinforcement_levels * Sv
        return abs(rest) <= WHOLE_SPACING_TOLERANCE * Sv


#: A remainder of the wall height under this fraction of a spacing is
#: floating-point rounding, not a strip of wall (GM-277): H / Sv for 4.8 m
#: at 0.8 m is 5.999999999999999.
WHOLE_SPACING_TOLERANCE = 1e-9


def _checked_depths(depths, H: float) -> Tuple[float, ...]:
    """``level_depths`` as a tuple of floats, refused unless each lies
    inside the wall and below the one above it (GM-273)."""
    takes = ("level_depths takes the depths z (m) of the reinforcement "
             "levels below the top of the wall, from the top down, as GEC-11 "
             "Vol II's examples print Z (Example E4, Table E4-7.1, p. E4-11)")
    if isinstance(depths, (str, bytes)) or not hasattr(depths, "__len__") \
            or not len(depths):
        raise ValueError(f"{takes}; got {depths!r}")
    try:
        z = tuple(float(v) for v in depths)
    except (TypeError, ValueError):
        raise ValueError(f"{takes}: one number a level; got "
                         f"{depths!r}") from None
    if any(isinstance(v, bool) or not math.isfinite(f)
           for v, f in zip(depths, z)):
        raise ValueError(f"{takes}: one number a level; got {depths!r}")
    for i, zi in enumerate(z, start=1):
        if not 0.0 < zi < H:
            raise ValueError(
                f"level_depths: level {i} at z = {zi:g} m is outside the "
                f"wall: give each level's depth below the top of the wall "
                f"(z > 0) and above its base (z < wall_height = {H:g} m), "
                "from the top down")
    for i in range(1, len(z)):
        if not z[i] > z[i - 1]:
            raise ValueError(
                f"level_depths must increase from the top down, each level "
                f"below the one above it: level {i + 1} at z = {z[i]:g} m is "
                f"not below level {i} at z = {z[i - 1]:g} m; give the depths "
                "in order, one a level")
    return z
