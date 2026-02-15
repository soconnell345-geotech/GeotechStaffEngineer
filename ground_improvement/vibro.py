"""
Vibro-compaction (vibro-flotation) feasibility assessment.

Evaluates whether vibro-compaction is feasible based on grain size
(fines content, D50) and initial density (N_spt), and estimates
probe spacing for treatment.

All units SI: meters, percent.

References:
    FHWA GEC-13: Ground Modification Methods Reference Manual
    Brown (1977) — Vibroflotation compaction of cohesionless soils
    Mitchell & Jardine (2002) — A Guide to Ground Treatment
"""

import warnings
from typing import List, Tuple

from ground_improvement.results import VibroResult


def vibro_feasibility(
    fines_content: float,
    D50: float = None,
    initial_N_spt: float = 0.0,
) -> Tuple[bool, List[str]]:
    """Go/no-go assessment for vibro-compaction.

    Parameters
    ----------
    fines_content : float
        Percent passing #200 sieve (0-100).
    D50 : float, optional
        Median grain size (mm).
    initial_N_spt : float
        Current SPT blow count.

    Returns
    -------
    tuple of (bool, list of str)
        (is_feasible, list of reasons/notes).
    """
    if fines_content < 0 or fines_content > 100:
        raise ValueError(f"Fines content must be 0-100%, got {fines_content}")

    reasons = []
    feasible = True

    # Fines content criteria (primary screening)
    if fines_content > 20:
        feasible = False
        reasons.append(
            f"Fines content ({fines_content:.0f}%) exceeds 20% — "
            "vibro-compaction not effective in cohesive/silty soils"
        )
    elif fines_content > 15:
        reasons.append(
            f"Fines content ({fines_content:.0f}%) is marginal (15-20%) — "
            "vibro-compaction may have limited effectiveness"
        )
    elif fines_content > 10:
        reasons.append(
            f"Fines content ({fines_content:.0f}%) is in the 10-15% range — "
            "feasible but reduced effectiveness compared to clean sands"
        )
    else:
        reasons.append(
            f"Fines content ({fines_content:.0f}%) is favorable (< 10%)"
        )

    # D50 criteria
    if D50 is not None:
        if D50 < 0.1:
            feasible = False
            reasons.append(
                f"D50 ({D50:.2f} mm) is too fine (< 0.1 mm) for vibro-compaction"
            )
        elif D50 < 0.2:
            reasons.append(
                f"D50 ({D50:.2f} mm) is marginal — coarser sands respond better"
            )
        else:
            reasons.append(f"D50 ({D50:.2f} mm) is suitable")

    # Density criteria
    if initial_N_spt > 0:
        if initial_N_spt > 25:
            feasible = False
            reasons.append(
                f"N_spt ({initial_N_spt:.0f}) is already dense — "
                "limited improvement potential"
            )
        elif initial_N_spt > 20:
            reasons.append(
                f"N_spt ({initial_N_spt:.0f}) is medium-dense — "
                "moderate improvement potential"
            )
        else:
            reasons.append(
                f"N_spt ({initial_N_spt:.0f}) indicates loose sand — "
                "good improvement potential"
            )

    return feasible, reasons


def estimate_probe_spacing(
    initial_N_spt: float,
    target_N_spt: float,
    fines_content: float = 5.0,
) -> float:
    """Estimate probe spacing for vibro-compaction.

    Empirical relationship based on FHWA GEC-13 guidance.
    Tighter spacing is needed for looser soils and higher targets.

    Parameters
    ----------
    initial_N_spt : float
        Current SPT blow count.
    target_N_spt : float
        Desired SPT blow count after treatment.
    fines_content : float
        Percent fines (affects effectiveness). Default 5%.

    Returns
    -------
    float
        Estimated probe spacing (m), center-to-center.
    """
    if initial_N_spt <= 0:
        raise ValueError(f"Initial N_spt must be positive, got {initial_N_spt}")
    if target_N_spt <= initial_N_spt:
        raise ValueError(
            f"Target N_spt ({target_N_spt}) must exceed initial ({initial_N_spt})"
        )

    # Base spacing: 2.0 - 3.5 m depending on improvement ratio
    # Higher improvement ratio → tighter spacing
    improvement_ratio = target_N_spt / initial_N_spt

    # Linear interpolation: ratio 1.5 → 3.0 m, ratio 4.0 → 1.5 m
    if improvement_ratio <= 1.5:
        base_spacing = 3.0
    elif improvement_ratio >= 4.0:
        base_spacing = 1.5
    else:
        base_spacing = 3.0 - (improvement_ratio - 1.5) / (4.0 - 1.5) * (3.0 - 1.5)

    # Fines content adjustment: more fines → tighter spacing
    if fines_content > 10:
        fines_factor = 1.0 - 0.02 * (fines_content - 10)  # reduce by 2% per %fines above 10
        fines_factor = max(fines_factor, 0.7)
        base_spacing *= fines_factor

    return round(base_spacing, 2)


def analyze_vibro_compaction(
    fines_content: float,
    initial_N_spt: float,
    target_N_spt: float = 25.0,
    D50: float = None,
    pattern: str = "triangular",
) -> VibroResult:
    """Full vibro-compaction feasibility analysis.

    Parameters
    ----------
    fines_content : float
        Percent passing #200 sieve (0-100).
    initial_N_spt : float
        Current SPT blow count.
    target_N_spt : float
        Desired SPT blow count. Default 25.
    D50 : float, optional
        Median grain size (mm).
    pattern : str
        Probe pattern: 'triangular' or 'square'. Default 'triangular'.

    Returns
    -------
    VibroResult
        Complete feasibility results.
    """
    if pattern not in ("triangular", "square"):
        raise ValueError(f"Pattern must be 'triangular' or 'square', got '{pattern}'")

    is_feasible, reasons = vibro_feasibility(fines_content, D50, initial_N_spt)

    spacing = 0.0
    if is_feasible and initial_N_spt > 0 and target_N_spt > initial_N_spt:
        spacing = estimate_probe_spacing(initial_N_spt, target_N_spt, fines_content)

    return VibroResult(
        is_feasible=is_feasible,
        fines_content_percent=fines_content,
        initial_N_spt=initial_N_spt,
        target_N_spt=target_N_spt,
        recommended_spacing_m=spacing,
        probe_pattern=pattern if is_feasible else "",
        reasons=reasons,
    )
