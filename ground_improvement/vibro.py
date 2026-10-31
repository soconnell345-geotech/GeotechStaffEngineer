"""
Vibro-compaction (vibro-flotation) feasibility assessment.

Screens a granular soil for vibro-compaction by its fines content and its
SPT N, and reads the probe spacing for a target relative density off the
chart GEC-13 prints for it (GM-291).

All units SI: meters, percent.

What GEC-13 (FHWA-NHI-16-027) Vol. I, Ch. 4 prints, and the tool follows:

- Fines (p. 4-65): "proper densification generally cannot be achieved when
  the granular soil contains more than 12 to 15 percent silt OR more than 2
  percent clay"; p. 4-67: soils in Fig. 4-27's zone B "are ideally suited
  for vibro-compaction with fines content below 10 percent". Table 4-5
  (Degen 1997) adds, by USCS: silty sand (SM) "Compaction inhibited if silt
  content >8%", clayey sand (SC) "if clay content >2%".
- Density: Table 4-8 (p. 4-74), SPT N to the equivalent relative density of
  a normally consolidated sand; p. 4-75: "It is unlikely, even for heavy
  loading, that it will be necessary to achieve a relative density above 85
  percent."
- Targets (p. 4-77): "60 percent relative density for floor slabs, flat
  bottom tanks, embankments; 70 to 75 percent relative density for column
  footings, bridge footings; 80 percent relative density for machinery and
  mat foundations".
- Spacing: Fig. 4-34 (p. 4-77), "Relative density versus probe spacing for
  silty sands", read off the printed figure (:data:`FIG_4_34`). It "is
  based on the lower bound soil gradation (silty sand) indicated in Figure
  4-33" (p. 4-76), and the area each point compacts "depends on the
  compaction method used and the specific characteristics of the vibrator,
  which may not be known in advance" (p. 4-75).

Until GM-291 the screen also refused a D50 under 0.1 mm and an N over 25,
took fines of 15 to 20 % as marginal, and estimated the spacing from the
ratio of a target N to the initial N: none of that is printed in GEC-13,
and it is gone.
"""

import math
from typing import List, Optional, Tuple

from ground_improvement.results import VibroResult

#: GEC-13 p. 4-67: Fig. 4-27's zone B, "ideally suited for vibro-compaction
#: with fines content below 10 percent".
FINES_IDEAL_PERCENT = 10.0
#: GEC-13 p. 4-65: "proper densification generally cannot be achieved when
#: the granular soil contains more than 12 to 15 percent silt".
FINES_LIMIT_PERCENT = (12.0, 15.0)

#: GEC-13 Table 4-8 (p. 4-74), for a normally consolidated sand: SPT N
#: (blows/foot) upper bound -> (density, equivalent relative density range
#: in percent). The last row has no upper bound.
TABLE_4_8 = (
    (4.0, "very loose", (0.0, 15.0)),
    (10.0, "loose", (15.0, 35.0)),
    (30.0, "medium dense", (35.0, 65.0)),
    (50.0, "dense", (65.0, 85.0)),
    (math.inf, "very dense", (85.0, 100.0)),
)

#: GEC-13 Fig. 4-34 (p. 4-77), "Relative density versus probe spacing for
#: silty sands": target relative density (%) -> spacing of center of
#: vibration (m). Read off the printed figure (a 674 x 591 px chart image,
#: no vector layer, 2026-10-31): the grid lines found and fitted (x within
#: 0.002 m, y within 0.05 %), the curve's centre column read on every pixel
#: row clear of a grid line, a line fitted through the rows within 1.5 % of
#: each 5 % grid line, and the reads drawn back over the figure on the
#: curve. The curve is printed from 90 % at 1.40 m to 55 % at 2.50 m; the
#: tool reads it there only, linear in relative density between nodes.
FIG_4_34 = {
    55.0: 2.50, 60.0: 2.30, 65.0: 2.13, 70.0: 1.95, 75.0: 1.80,
    80.0: 1.67, 85.0: 1.52, 90.0: 1.40,
}

#: GEC-13 p. 4-77, the performance criteria by structure, as printed.
TARGETS_GEC13 = (
    "60 percent relative density for floor slabs, flat bottom tanks, "
    "embankments; 70 to 75 percent relative density for column footings, "
    "bridge footings; 80 percent relative density for machinery and mat "
    "foundations (GEC-13 p. 4-77)")


def density_from_N(N_spt: float) -> Tuple[str, Tuple[float, float]]:
    """(density, equivalent relative density range %) for an SPT N, as
    GEC-13 Table 4-8 (p. 4-74) prints them for a normally consolidated
    sand. A boundary N (4, 10, 30, 50) falls in the looser class."""
    if N_spt < 0:
        raise ValueError(f"N_spt must be >= 0, got {N_spt}")
    for upper, name, dr in TABLE_4_8:
        if N_spt <= upper:
            return name, dr
    return TABLE_4_8[-1][1], TABLE_4_8[-1][2]


def probe_spacing_fig_4_34(target_relative_density: float) -> float:
    """Spacing of center of vibration (m) for a target relative density
    (%), off GEC-13 Fig. 4-34 (p. 4-77), silty sands, read at 55 to 90 %
    and refused outside it: the curve is printed there only."""
    lo, hi = min(FIG_4_34), max(FIG_4_34)
    dr = target_relative_density
    if not lo - 1e-9 <= dr <= hi + 1e-9:
        raise ValueError(
            f"target_relative_density {dr:g} %: GEC-13 Fig. 4-34 (p. 4-77) "
            f"prints the probe spacing for silty sands at a target relative "
            f"density of {lo:g} to {hi:g} % only, and the tool does not "
            "extend it. GEC-13 p. 4-75: 'It is unlikely, even for heavy "
            "loading, that it will be necessary to achieve a relative "
            "density above 85 percent.'")
    xs = sorted(FIG_4_34)
    dr = min(max(dr, lo), hi)
    for x0, x1 in zip(xs, xs[1:]):
        if dr <= x1:
            t = (dr - x0) / (x1 - x0)
            return FIG_4_34[x0] + t * (FIG_4_34[x1] - FIG_4_34[x0])
    return FIG_4_34[xs[-1]]


def vibro_feasibility(
    fines_content: float,
    initial_N_spt: float = 0.0,
) -> Tuple[bool, List[str]]:
    """Go/no-go assessment for vibro-compaction, as GEC-13 prints it.

    Parameters
    ----------
    fines_content : float
        Percent passing #200 sieve (0-100).
    initial_N_spt : float
        Current SPT blow count (blows/foot); 0 = not given.

    Returns
    -------
    tuple of (bool, list of str)
        (is_feasible, list of reasons/notes).
    """
    if fines_content < 0 or fines_content > 100:
        raise ValueError(f"Fines content must be 0-100%, got {fines_content}")

    reasons = []
    feasible = True
    lo, hi = FINES_LIMIT_PERCENT

    if fines_content > hi:
        feasible = False
        reasons.append(
            f"Fines content ({fines_content:.0f}%) is above GEC-13's 12 to "
            "15 percent: 'proper densification generally cannot be "
            "achieved when the granular soil contains more than 12 to 15 "
            "percent silt OR more than 2 percent clay' (p. 4-65); such "
            "soils 'can be improved by vibro-replacement, as described in "
            "Chapter 5 Aggregate Columns' (p. 4-67)")
    elif fines_content > lo:
        reasons.append(
            f"Fines content ({fines_content:.0f}%) is within GEC-13's 12 to "
            "15 percent, above which 'proper densification generally "
            "cannot be achieved' (p. 4-65): marginal")
    elif fines_content > FINES_IDEAL_PERCENT:
        reasons.append(
            f"Fines content ({fines_content:.0f}%) is above the 10 percent "
            "of GEC-13 Fig. 4-27's zone B, 'ideally suited for "
            "vibro-compaction with fines content below 10 percent' (p. "
            "4-67), and below the 12 to 15 percent of p. 4-65")
    else:
        reasons.append(
            f"Fines content ({fines_content:.0f}%) is below 10 percent: "
            "GEC-13 Fig. 4-27's zone B soils are 'ideally suited for "
            "vibro-compaction with fines content below 10 percent' (p. "
            "4-67)")
    reasons.append(
        "The clay fraction is not screened (the tool takes the fines "
        "content): GEC-13 p. 4-65 sets 2 percent clay apart from the silt, "
        "and Table 4-5 (p. 4-66) inhibits compaction of a silty sand (SM) "
        "above 8 percent silt and of a clayey sand (SC) above 2 percent "
        "clay")

    if initial_N_spt > 0:
        name, (dr_lo, dr_hi) = density_from_N(initial_N_spt)
        dr = (f"above {dr_lo:.0f}" if dr_hi >= 100 else
              f"below {dr_hi:.0f}" if dr_lo <= 0 else
              f"{dr_lo:.0f} to {dr_hi:.0f}")
        if dr_lo >= 85.0:
            feasible = False
            reasons.append(
                f"N_spt ({initial_N_spt:.0f}) is {name} by GEC-13 Table 4-8 "
                f"(p. 4-74), an equivalent relative density {dr} percent; "
                "p. 4-75: 'It is unlikely, even for heavy loading, that it "
                "will be necessary to achieve a relative density above 85 "
                "percent.'")
        else:
            reasons.append(
                f"N_spt ({initial_N_spt:.0f}) is {name} by GEC-13 Table 4-8 "
                f"(p. 4-74): an equivalent relative density {dr} percent "
                "for a normally consolidated sand")

    return feasible, reasons


def analyze_vibro_compaction(
    fines_content: float,
    initial_N_spt: float,
    target_relative_density: Optional[float] = None,
    pattern: str = "triangular",
) -> VibroResult:
    """Full vibro-compaction feasibility analysis.

    Parameters
    ----------
    fines_content : float
        Percent passing #200 sieve (0-100).
    initial_N_spt : float
        Current SPT blow count (blows/foot).
    target_relative_density : float, optional
        Target relative density (%), 55 to 90: the probe spacing is read
        off GEC-13 Fig. 4-34 (p. 4-77) for it. GEC-13 p. 4-77 gives 60 %
        for floor slabs, flat bottom tanks and embankments, 70 to 75 % for
        column and bridge footings, 80 % for machinery and mat foundations.
        Not given: no spacing.
    pattern : str
        Probe pattern: 'triangular' or 'square'. Default 'triangular'.
        GEC-13 p. 4-77: a square pattern "requires 5 to 8 percent more
        points to achieve the same minimum densities in large areas".

    Returns
    -------
    VibroResult
        Complete feasibility results.
    """
    if pattern not in ("triangular", "square"):
        raise ValueError(f"Pattern must be 'triangular' or 'square', got '{pattern}'")

    is_feasible, reasons = vibro_feasibility(fines_content, initial_N_spt)

    spacing = 0.0
    if target_relative_density is None:
        reasons.append(
            "No probe spacing: give target_relative_density, read off "
            "GEC-13 Fig. 4-34 at 55 to 90 percent. GEC-13's criteria: "
            + TARGETS_GEC13)
    elif is_feasible:
        spacing = probe_spacing_fig_4_34(target_relative_density)
        reasons.append(
            f"Probe spacing {spacing:.2f} m for a target relative density "
            f"of {target_relative_density:g} percent, off GEC-13 Fig. 4-34 "
            "(p. 4-77), 'Relative density versus probe spacing for silty "
            "sands': the lower bound gradation (p. 4-76), and the vibrator "
            "sets the area each point compacts, 'which may not be known in "
            "advance' (p. 4-75)")
        if initial_N_spt > 0:
            name, (dr_lo, _) = density_from_N(initial_N_spt)
            if dr_lo >= target_relative_density:
                reasons.append(
                    f"N_spt ({initial_N_spt:.0f}) is already {name} by "
                    f"GEC-13 Table 4-8, an equivalent relative density of "
                    f"at least {dr_lo:.0f} percent, at or above the target "
                    f"{target_relative_density:g} percent")
        if pattern == "square":
            reasons.append(
                "Square pattern: GEC-13 p. 4-77, 'The use of a square "
                "pattern instead of an equilateral triangular pattern "
                "requires 5 to 8 percent more points to achieve the same "
                "minimum densities in large areas.' Fig. 4-34 does not "
                "name its pattern")
    else:
        probe_spacing_fig_4_34(target_relative_density)  # refuse off the curve

    return VibroResult(
        is_feasible=is_feasible,
        fines_content_percent=fines_content,
        initial_N_spt=initial_N_spt,
        target_relative_density_percent=target_relative_density,
        recommended_spacing_m=spacing,
        probe_pattern=pattern if is_feasible else "",
        reasons=reasons,
    )
