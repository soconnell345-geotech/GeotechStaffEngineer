"""
Aggregate pier / rammed aggregate pier analysis.

Computes area replacement ratio, composite modulus, settlement reduction
factor, and bearing capacity improvement for stone column / aggregate
pier ground improvement.

All units SI: kPa, meters.

The settlement over the columns is by the stress ratio n (the default) or,
with the column modulus, by GEC-13 Fig. 5-21's weighted-modulus and Balaam
curves (``settlement_ratio``, GM-220). The Priebe curve GEC-13 prints, Fig.
5-21 curve (4), is read by ``settlement_ratio.priebe_curve_settlement_ratio``
(GM-221); since GM-276 it is the only Priebe basis, with no formula.

References:
    Barksdale & Bachus (1983) — Design and Construction of Stone Columns
    FHWA GEC-13: Ground Modification Methods Reference Manual
"""

import math
import warnings
from typing import Optional

from ground_improvement.results import AggregatePierResult


def area_replacement_ratio(column_diameter: float, spacing: float,
                           pattern: str = "triangular") -> float:
    """Compute area replacement ratio as = Ac / A_tributary.

    Parameters
    ----------
    column_diameter : float
        Pier column diameter (m).
    spacing : float
        Center-to-center spacing (m).
    pattern : str
        'triangular' or 'square'.

    Returns
    -------
    float
        Area replacement ratio (dimensionless, 0 to 1).
    """
    if column_diameter <= 0:
        raise ValueError(f"Column diameter must be positive, got {column_diameter}")
    if spacing <= 0:
        raise ValueError(f"Spacing must be positive, got {spacing}")
    if spacing <= column_diameter:
        raise ValueError(
            f"Spacing ({spacing} m) must exceed column diameter ({column_diameter} m)"
        )

    Ac = math.pi / 4.0 * column_diameter**2

    if pattern == "triangular":
        A_trib = math.sqrt(3) / 2.0 * spacing**2
    elif pattern == "square":
        A_trib = spacing**2
    else:
        raise ValueError(f"Pattern must be 'triangular' or 'square', got '{pattern}'")

    return Ac / A_trib


def settlement_reduction_factor(as_ratio: float, n: float) -> float:
    """Compute settlement reduction factor.

    SRF = 1 / (1 + as * (n - 1))

    Parameters
    ----------
    as_ratio : float
        Area replacement ratio.
    n : float
        Stress concentration ratio (typically 3-8).

    Returns
    -------
    float
        Settlement reduction factor (0 to 1).
    """
    if n < 1.0:
        raise ValueError(f"Stress concentration ratio must be >= 1, got {n}")
    return 1.0 / (1.0 + as_ratio * (n - 1.0))


def composite_modulus(as_ratio: float, E_soil: float,
                      n: float) -> float:
    """Compute composite modulus of improved ground.

    E_comp = Es * [1 + as * (n - 1)]

    Derived from the unit-cell equal-strain model: the settlement
    reduction factor SRF = 1 / [1 + as*(n-1)] equals Es / E_comp,
    so E_comp = Es / SRF = Es * [1 + as*(n-1)].

    This is the composite modulus consistent with the stress-ratio
    settlement reduction factor (GEC-13 Eq. 5-4, the stress reduction
    method; not Priebe's). It differs from the simple Voigt weighted
    average (as*Ec + (1-as)*Es), which is only correct when the stress
    concentration ratio equals the modular ratio (n = Ec/Es).

    Parameters
    ----------
    as_ratio : float
        Area replacement ratio.
    E_soil : float
        Modulus of surrounding soil (kPa).
    n : float
        Stress concentration ratio (typically 3-8).

    Returns
    -------
    float
        Composite modulus (kPa).

    References
    ----------
    FHWA GEC-13 (FHWA-NHI-16-027) Vol. I, Secs. 4.1.2.1, 4.1.2.4 and Eq. 5-4,
    pp. 5-28 to 5-32: the unit cell under equal strain, the stress split by n.
    """
    return E_soil * (1.0 + as_ratio * (n - 1.0))


#: What the Priebe tool reads (GM-221, GM-276), for a result's choices and
#: notes: GEC-13's printed curve, its only basis.
PRIEBE_BASIS = (
    "GEC-13 (FHWA-NHI-16-027) Vol. I, Fig. 5-21, p. 5-36, curve (4) "
    "'PRIEBE phi = 37.5 deg', read off the printed figure (A/A_c 1.6 to "
    "8.0, about +/- 0.03 in n'; from A/A_c about 4.0 on it is drawn on "
    "one line with Balaam's curve (5a)). GEC-13 prints this curve and no "
    "formula for it, so the tool takes the area ratio alone: the column "
    "friction angle is the curve's 37.5 deg. GEC-13 reads it in Example 2 "
    "(Sec. 4.3.2, Fig. 5-27, p. 5-49), and p. 5-37: 'For preliminary "
    "estimates, the Priebe curve may be used to evaluate the upper bound "
    "effectiveness'")


def priebe_choices(as_ratio: float, n_prime: float) -> list:
    """What ``priebe_improvement_factor`` stands on (GM-221, GM-276):
    GEC-13's printed Priebe curve, the tool's one basis, with the other
    Fig. 5-21 curves the module reads as the alternatives."""
    from geotech_common.choices import MODULE, choice
    return [choice(
        "settlement ratio n'",
        f"GEC-13 Fig. 5-21 curve (4), Priebe phi = 37.5 deg: n' = "
        f"{n_prime:.2f} at A/A_c = {1.0 / as_ratio:.2f}", MODULE,
        PRIEBE_BASIS,
        ["GEC-13 Fig. 5-21's Balaam curves (5), by E_c/E_s; p. 5-37: "
         "'an average or lower bound estimate suitable for preliminary "
         "analyses' (aggregate_piers settlement_method 'balaam_rigid' or "
         "'balaam_flexible')",
         "GEC-13 Fig. 5-21's weighted-modulus curve (2), by E_c/E_s "
         "(aggregate_piers settlement_method 'weighted_modulus')"])]


def improved_bearing_capacity(q_unreinforced: float, as_ratio: float,
                              n: float) -> float:
    """Compute improved bearing capacity with aggregate piers.

    q_improved = q_unreinforced * (1 + as * (n - 1))

    .. note:: First-order estimate, not Priebe (GI-2)
        This applies the same equal-strain stress-concentration factor
        ``1 + as*(n - 1)`` used for the composite modulus / settlement
        reduction — a simple first-order estimate of the bearing
        improvement. It is not the Priebe curve GEC-13 prints (Fig. 5-21
        curve (4), p. 5-36), which is a settlement ratio, not a bearing
        factor.

    Parameters
    ----------
    q_unreinforced : float
        Unreinforced bearing capacity (kPa).
    as_ratio : float
        Area replacement ratio.
    n : float
        Stress concentration ratio.

    Returns
    -------
    float
        Improved bearing capacity (kPa).

    References
    ----------
    Barksdale & Bachus (1983) stress-concentration model; FHWA GEC-13.
    """
    return q_unreinforced * (1.0 + as_ratio * (n - 1.0))


def improved_settlement(S_unreinforced: float, as_ratio: float,
                        n: float) -> float:
    """Compute improved (reduced) settlement.

    S_improved = SRF * S_unreinforced

    Parameters
    ----------
    S_unreinforced : float
        Unreinforced settlement (mm).
    as_ratio : float
        Area replacement ratio.
    n : float
        Stress concentration ratio.

    Returns
    -------
    float
        Improved settlement (mm).
    """
    srf = settlement_reduction_factor(as_ratio, n)
    return srf * S_unreinforced


#: How the settlement over the columns is worked out: the stress ratio n
#: (the default), or the settlement ratio n' by E_c / E_s off GEC-13 Fig.
#: 5-21 (``ground_improvement.settlement_ratio``, GM-220).
SETTLEMENT_METHODS = ("stress_ratio", "weighted_modulus", "balaam_rigid",
                      "balaam_flexible")


def analyze_aggregate_piers(
    column_diameter: float,
    spacing: float,
    pattern: str = "triangular",
    E_column: Optional[float] = None,
    E_soil: float = 5000.0,
    n: float = 5.0,
    q_unreinforced: float = 0.0,
    S_unreinforced: float = 0.0,
    settlement_method: str = "stress_ratio",
) -> AggregatePierResult:
    """Run a complete aggregate pier analysis.

    Parameters
    ----------
    column_diameter : float
        Pier column diameter (m).
    spacing : float
        Center-to-center spacing (m).
    pattern : str
        'triangular' or 'square'. Default 'triangular'.
    E_column : float, optional
        Modulus of aggregate column (kPa). Needed by the settlement
        methods of GEC-13 Fig. 5-21 (``settlement_method`` other than
        'stress_ratio'), which take E_c / E_s. With 'stress_ratio' it is
        used only to warn when it is below E_soil: the stress split is
        set by ``n``, not worked out from the moduli.
    E_soil : float
        Modulus of surrounding soil (kPa). Default 5,000 kPa. With
        'stress_ratio' used only for the composite modulus; with the
        Fig. 5-21 methods it also sets E_c / E_s.
    n : float
        Stress concentration ratio sigma_c / sigma_s. Default 5.0 (GEC-13:
        "between 2.0 and 5.0" for stone columns, Sec. 4.1.2.4; for rammed
        aggregate piers "4 to 45 for rigid footings" and 5 to 10 under
        embankments, p. 5-45).
    q_unreinforced : float
        Unreinforced bearing capacity (kPa). Default 0 (not computed).
    S_unreinforced : float
        Unreinforced settlement (mm). Default 0 (not computed).
    settlement_method : str
        How the settlement over the columns is worked out (GM-220):
        'stress_ratio' (default): SRF = 1 / [1 + a_s (n - 1)] from n.
        'weighted_modulus', 'balaam_rigid', 'balaam_flexible': the
        settlement ratio n' (settlement without columns over settlement
        with them) by E_c / E_s off GEC-13 Fig. 5-21, p. 5-36; SRF = 1 /
        n' and the composite modulus E_s n'. Balaam's curves are drawn
        for E_c / E_s = 5 and 10 (rigid) and 10 (flexible) only. ``n``
        still sets the improved bearing.

    Returns
    -------
    AggregatePierResult
        Complete analysis results, with its ``choices`` (GM-142): the
        settlement method, n, the soil modulus and the pattern, each given
        or default.
    """
    as_ratio = area_replacement_ratio(column_diameter, spacing, pattern)

    if as_ratio > 0.4:
        warnings.warn(
            f"Area replacement ratio {as_ratio:.3f} is unusually high (> 0.4). "
            "Verify column spacing and diameter."
        )
    if settlement_method not in SETTLEMENT_METHODS:
        raise ValueError(
            f"settlement_method must be one of {SETTLEMENT_METHODS}, got "
            f"{settlement_method!r}")
    by_moduli = settlement_method != "stress_ratio"
    if by_moduli and E_column is None:
        raise ValueError(
            f"settlement_method {settlement_method!r} works the settlement "
            "ratio out from E_c / E_s (GEC-13 Fig. 5-21): give E_column")
    if E_column is not None and E_column < E_soil and not by_moduli:
        warnings.warn(
            f"Column modulus ({E_column} kPa) is less than soil modulus "
            f"({E_soil} kPa). This is unusual for aggregate piers."
        )

    modular_ratio = None
    if by_moduli:
        from ground_improvement.settlement_ratio import settlement_ratio
        modular_ratio = E_column / E_soil
        n_prime = settlement_ratio(settlement_method, as_ratio,
                                   modular_ratio)
        srf = 1.0 / n_prime
        E_comp = E_soil * n_prime
    else:
        srf = settlement_reduction_factor(as_ratio, n)
        E_comp = composite_modulus(as_ratio, E_soil, n)

    q_improved = 0.0
    if q_unreinforced > 0:
        q_improved = improved_bearing_capacity(q_unreinforced, as_ratio, n)

    S_improved = 0.0
    if S_unreinforced > 0:
        S_improved = (srf * S_unreinforced if by_moduli
                      else improved_settlement(S_unreinforced, as_ratio, n))

    return AggregatePierResult(
        area_replacement_ratio=as_ratio,
        stress_concentration_ratio=n,
        composite_modulus_kPa=E_comp,
        settlement_reduction_factor=srf,
        improved_bearing_kPa=q_improved,
        unreinforced_bearing_kPa=q_unreinforced,
        settlement_improved_mm=S_improved,
        settlement_unreinforced_mm=S_unreinforced,
        column_diameter_m=column_diameter,
        column_spacing_m=spacing,
        pattern=pattern,
        settlement_method=settlement_method,
        settlement_ratio=1.0 / srf,
        modular_ratio=modular_ratio,
        choices=aggregate_pier_choices(
            n, E_soil, pattern, settlement_method, as_ratio, modular_ratio,
            1.0 / srf, q_unreinforced),
    )


#: The settlement methods besides the stress ratio, for the choices.
_WEIGHTED = ("settlement_method 'weighted_modulus' with E_column: GEC-13 "
             "Fig. 5-21 curve (2), n' = 1 + a_s (E_c/E_s - 1)")
_BALAAM = ("settlement_method 'balaam_rigid' (E_c/E_s 5 to 10) or "
           "'balaam_flexible' (E_c/E_s = 10) with E_column: GEC-13 Fig. "
           "5-21 Balaam curves (5); p. 5-37: the FHWA (1983) Equilibrium "
           "Method, 'roughly equivalent to the Balaam relationships', "
           "'represents an average or lower bound estimate suitable for "
           "preliminary analyses'")
_STRESS_RATIO = ("settlement_method 'stress_ratio' (the default): SRF = "
                 "1 / [1 + a_s (n - 1)] from the given n")
_PRIEBE = ("the settlement ratio off GEC-13's printed Priebe curve, Fig. "
           "5-21 curve (4), phi = 37.5 deg (priebe_improvement_factor); "
           "GEC-13 p. 5-37: the Priebe curve 'may be used to evaluate the "
           "upper bound effectiveness'")
_NAMES = {"weighted_modulus": "weighted modulus",
          "balaam_rigid": "Balaam, rigid",
          "balaam_flexible": "Balaam, flexible"}


def _settlement_method_choice(method, as_ratio, modular_ratio, n_prime):
    from geotech_common.choices import choice, how_given
    how = how_given(method, "stress_ratio", name="settlement_method")
    if method == "stress_ratio":
        return choice(
            "settlement method", "stress ratio, SRF = 1 / [1 + a_s (n - 1)]",
            how,
            "unit cell under equal strain, the stress split by the given n "
            "(GEC-13 Eq. 5-4, the stress reduction method); S = SRF x the "
            "unreinforced settlement",
            [_WEIGHTED, _BALAAM, _PRIEBE])
    from ground_improvement.settlement_ratio import BASIS
    between = (method == "balaam_rigid"
               and not any(math.isclose(modular_ratio, r, rel_tol=0.01)
                           for r in (5.0, 10.0)))
    others = [_BALAAM] if method == "weighted_modulus" else [_WEIGHTED]
    if method == "balaam_rigid" and math.isclose(modular_ratio, 10.0,
                                                 rel_tol=0.01):
        others.append("settlement_method 'balaam_flexible': Balaam curve "
                      "(5a), E_c/E_s = 10, flexible")
    if method == "balaam_flexible":
        others.append("settlement_method 'balaam_rigid': Balaam curve "
                      "(5b), E_c/E_s = 10, rigid")
    return choice(
        "settlement method",
        f"{_NAMES[method]}: n' = {n_prime:.2f} at A/A_c = "
        f"{1.0 / as_ratio:.2f}, E_c/E_s = {modular_ratio:g}", how,
        BASIS[method]
        + ("; interpolated linearly in E_c/E_s between (5c) and (5b)"
           if between else "")
        + ". n' is the settlement without columns over that with them: "
        "S = the unreinforced settlement / n'",
        [_STRESS_RATIO, *others, _PRIEBE])


def aggregate_pier_choices(n: float, E_soil: float, pattern: str,
                           settlement_method: str = "stress_ratio",
                           as_ratio: Optional[float] = None,
                           modular_ratio: Optional[float] = None,
                           n_prime: Optional[float] = None,
                           q_unreinforced: float = 0.0) -> list:
    """What :func:`analyze_aggregate_piers` chose (GM-142, GM-220), as a
    result's ``choices`` (``geotech_common.choices``). With a Fig. 5-21
    settlement method n sets only the improved bearing, so it is a choice
    only with ``q_unreinforced``."""
    from geotech_common.choices import choice, how_given
    by_moduli = settlement_method != "stress_ratio"
    sets = ("the improved bearing; the settlement is by E_c/E_s "
            f"(settlement_method {settlement_method!r})" if by_moduli
            else "the settlement reduction factor 1 / [1 + a_s (n - 1)], "
            "the improved bearing and the composite modulus")
    n_choice = [
        choice("stress concentration ratio n", f"{n:g}",
               how_given(n, 5.0, name="n"),
               "n = sigma_c / sigma_s, taken as given and not worked out "
               "from the moduli; GEC-13 (FHWA-NHI-16-027): for stone "
               "columns n is 'between 2.0 and 5.0' (Sec. 4.1.2.4); for "
               "rammed aggregate piers '4 to 45 for rigid footings' and "
               "5 to 10 under embankments (p. 5-45). It sets " + sets,
               ["n from the moduli, n = E_c / E_s (equal-strain modular "
                "ratio; settlement_method 'weighted_modulus')",
                "the settlement ratio off GEC-13's printed Priebe curve, "
                "Fig. 5-21 curve (4) (priebe_improvement_factor)"]),
    ] if not by_moduli or q_unreinforced > 0 else []
    return [
        _settlement_method_choice(settlement_method, as_ratio,
                                  modular_ratio, n_prime),
        *n_choice,
        choice("soil modulus E_s", f"{E_soil:g} kPa",
               how_given(E_soil, 5000.0, name="E_soil"),
               (f"with E_column it sets E_c/E_s = {modular_ratio:g} and "
                "the composite modulus E_comp = E_s n'" if by_moduli else
                "used only for the composite modulus E_comp = "
                "E_s [1 + a_s (n - 1)]")),
        choice("column pattern", pattern,
               how_given(pattern, "triangular", name="pattern"),
               "unit cell of equal area: d_e = 1.05 S (triangular) or "
               "1.13 S (square), a_s = (d / d_e)^2",
               ["square (pattern 'square')" if pattern == "triangular"
                else "triangular (pattern 'triangular')"]),
    ]
