"""
Embedment depth calculations for SOE walls (free earth support).

Braced / anchored walls: moments about the lowest support (California
Trenching and Shoring Manual Sec 8-4.02, hinge method) -- the whole wall for a
single support level, the portion below the lowest support for two or more.
Above the excavation the pressure is the apparent envelope plus surcharge and
water (the same pressure the support loads use); below it, layered Rankine
active and passive pressures with net water. D solves MR = FS x MD. By default
no depth increase is applied: the free-earth-support depth with its FS is the
design embedment, as in Caltrans Example 8-1.

Cantilever walls: the Caltrans Simplified Method, D = 1.2 D0.

Replaced 2026-09-15 (field feedback N2). The previous routine used one soil
layer (the one at excavation level), started the active triangle at zero at
the support instead of carrying the overburden, and used moment arms d/3 and
D/3 where the resultants act at about 2d/3 and 2D/3 -- errors in both
directions. See ``soe.free_earth`` and DESIGN.md.

All units SI: m, kPa, kN/m³, degrees.

References:
    California Trenching and Shoring Manual (2025), Sec 7-5.02, 8-4
    FHWA-IF-99-015, GEC-4, Section 5.4
    USACE EM 1110-2-2504, Chapter 5
"""

from typing import Optional, Union

from soe.geometry import ExcavationGeometry
from soe.free_earth import (
    CantileverSolution,
    SupportSolution,
    apparent_pressure_profile,
    profile_for,
    solve_about_support,
    solve_cantilever,
)


def embedment_detail(geometry: ExcavationGeometry, FOS_passive: float = 1.5,
                     ) -> Union[CantileverSolution, SupportSolution]:
    """Full free-earth-support solution behind :func:`compute_embedment`.

    A ``SupportSolution`` (D with FS, D at FS = 1, the lowest support's load
    from below, moment and shear) for a braced wall; a ``CantileverSolution``
    (D0, D = 1.2 D0, ...) when there are no supports.
    """
    geometry.validate()
    prof = profile_for(geometry)
    if not geometry.support_levels:
        return solve_cantilever(prof, FS_passive=FOS_passive,
                                embedment_increase=1.2)
    _, p = apparent_pressure_profile(geometry)
    depths = [s.depth for s in geometry.support_levels]
    pivot = max(depths)
    body_top = 0.0 if len(depths) == 1 else pivot
    return solve_about_support(prof, pivot=pivot, body_top=body_top,
                               FS=FOS_passive, above=p)


def compute_embedment(geometry: ExcavationGeometry,
                      FOS_passive: float = 1.5,
                      embedment_increase: Optional[float] = None) -> float:
    """Compute required wall embedment below excavation level (m).

    Parameters
    ----------
    geometry : ExcavationGeometry
        Complete excavation geometry.
    FOS_passive : float
        Braced walls: MR = FS x MD about the lowest support (Caltrans uses
        1.3). Cantilever walls: factor on passive pressure. Default 1.5.
    embedment_increase : float, optional
        Multiplier on the solved depth. Default 1.0 for braced walls and 1.2
        (D = 1.2 D0) for cantilever walls.

    Returns
    -------
    float
        Required embedment depth below excavation level (m).
    """
    detail = embedment_detail(geometry, FOS_passive=FOS_passive)
    if isinstance(detail, CantileverSolution):
        inc = 1.2 if embedment_increase is None else embedment_increase
        base = detail.D0
    else:
        inc = 1.0 if embedment_increase is None else embedment_increase
        base = detail.D
    if inc < 1.0:
        raise ValueError("embedment_increase must be >= 1.0")
    return base * inc
