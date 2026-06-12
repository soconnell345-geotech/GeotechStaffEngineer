"""
Pre-canned geotechnical reliability wrappers.

Thin adapters that build the performance function g(values) for common
analyses and push it through any engine:

- :func:`bearing_capacity_reliability` — FOS = q_ult / q_applied around
  ``bearing_capacity.BearingCapacityAnalysis`` (single-layer).
- :func:`axial_pile_reliability` — FOS = Q_ult / applied_load around
  ``axial_pile.AxialPileAnalysis``.
- :func:`slope_reliability` — pure delegate to
  ``slope_stability.probabilistic`` (fosm_fos / monte_carlo_fos), which is
  the validated slope implementation (Duncan 2000 anchor).

Variable means default to the deterministic base value (give only cov/std),
matching the slope_stability spec convention.
"""

from __future__ import annotations

from typing import Dict, Optional

from reliability.form import form
from reliability.fosm import fosm
from reliability.monte_carlo import monte_carlo
from reliability.pem import pem

ENGINES = ("fosm", "pem", "monte_carlo", "form")


def run_engine(g, variables, engine: str = "fosm", correlation=None,
               convention: str = "fos", **engine_kwargs):
    """Dispatch g + variables to the named engine."""
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}, got '{engine}'.")
    fn = {"fosm": fosm, "pem": pem, "monte_carlo": monte_carlo,
          "form": form}[engine]
    return fn(g, variables, correlation=correlation, convention=convention,
              **engine_kwargs)


def _fill_means(variables: Dict[str, Dict], base_values: Dict[str, float],
                context: str) -> Dict[str, Dict]:
    """Default each variable's mean to the deterministic base value."""
    out = {}
    for name, spec in variables.items():
        if name not in base_values:
            raise ValueError(
                f"{context}: unknown variable '{name}'. "
                f"Valid variables: {sorted(base_values)}.")
        if not isinstance(spec, dict):
            raise ValueError(
                f"{context}: variable '{name}' spec must be a dict "
                f"(e.g. {{'cov': 0.1}}).")
        spec = dict(spec)
        spec.setdefault("mean", base_values[name])
        out[name] = spec
    if not out:
        raise ValueError(f"{context}: variables dict is empty.")
    return out


# ---------------------------------------------------------------------------
# Bearing capacity
# ---------------------------------------------------------------------------

def bearing_capacity_reliability(footing: Dict,
                                 soil: Dict,
                                 applied_pressure: float,
                                 variables: Dict[str, Dict],
                                 engine: str = "fosm",
                                 correlation=None,
                                 ngamma_method: str = "vesic",
                                 factor_method: str = "vesic",
                                 **engine_kwargs):
    """Reliability of FOS = q_ult / q_applied for a shallow foundation.

    Parameters
    ----------
    footing : dict
        Passed to ``bearing_capacity.Footing`` (width, depth, shape, ...).
    soil : dict
        Single-layer properties: friction_angle (deg), cohesion (kPa),
        unit_weight (kN/m3) (+ optional gwt_depth passed to the profile).
    applied_pressure : float
        Applied bearing pressure q (kPa). Must be positive.
    variables : dict
        Random variables among: 'friction_angle', 'cohesion',
        'unit_weight', 'applied_pressure'. Means default to the base
        values; give {'cov': ...} or {'std': ...} (+ optional dist).
    engine : str
        'fosm' (default), 'pem', 'monte_carlo' or 'form'.

    Returns
    -------
    Engine result (FOSMResult / PEMResult / MonteCarloResult / FORMResult),
    convention="fos" (failure at FOS < 1).
    """
    from bearing_capacity import (
        BearingCapacityAnalysis, BearingSoilProfile, Footing, SoilLayer,
    )
    if applied_pressure <= 0:
        raise ValueError("applied_pressure must be positive.")
    base = {
        "friction_angle": float(soil.get("friction_angle", 0.0)),
        "cohesion": float(soil.get("cohesion", 0.0)),
        "unit_weight": float(soil.get("unit_weight", 18.0)),
        "applied_pressure": float(applied_pressure),
    }
    variables = _fill_means(variables, base, "bearing_capacity_reliability")
    wt = soil.get("gwt_depth", soil.get("water_table_depth"))
    ft = Footing(**footing)

    def g(values: Dict[str, float]) -> float:
        p = dict(base)
        p.update(values)
        layer = SoilLayer(
            cohesion=max(p["cohesion"], 0.0),
            friction_angle=min(max(p["friction_angle"], 0.0), 45.0),
            unit_weight=max(p["unit_weight"], 1.0),
        )
        profile = (BearingSoilProfile(layer1=layer, gwt_depth=wt)
                   if wt is not None else BearingSoilProfile(layer1=layer))
        res = BearingCapacityAnalysis(
            footing=ft, soil=profile,
            ngamma_method=ngamma_method, factor_method=factor_method,
        ).compute()
        return res.q_ultimate / p["applied_pressure"]

    return run_engine(g, variables, engine=engine, correlation=correlation,
                      convention="fos", **engine_kwargs)


# ---------------------------------------------------------------------------
# Axial pile
# ---------------------------------------------------------------------------

def axial_pile_reliability(pile: Dict,
                           soil_layers,
                           pile_length: float,
                           applied_load: float,
                           variables: Dict[str, Dict],
                           gwt_depth: Optional[float] = None,
                           method: str = "auto",
                           engine: str = "fosm",
                           correlation=None,
                           **engine_kwargs):
    """Reliability of FOS = Q_ult / applied_load for a driven pile.

    Parameters
    ----------
    pile : dict
        Passed to ``axial_pile.PileSection`` (name, pile_type, area,
        perimeter, tip_area, width, ...).
    soil_layers : list of dict
        Each passed to ``axial_pile.AxialSoilLayer`` (thickness, soil_type,
        unit_weight, friction_angle / cohesion).
    pile_length : float
        Embedded length (m).
    applied_load : float
        Applied axial load (kN). Must be positive.
    variables : dict
        Keys: 'friction_angle', 'cohesion', 'unit_weight' — applied to ALL
        layers carrying that property — or scoped to one layer by 1-based
        index, e.g. 'cohesion:2'. 'applied_load' is also available. Means
        default to base values.
    method : str
        axial_pile method ('auto', 'beta').

    Returns
    -------
    Engine result, convention="fos".
    """
    from axial_pile import AxialPileAnalysis, AxialSoilLayer, \
        AxialSoilProfile, PileSection
    if applied_load <= 0:
        raise ValueError("applied_load must be positive.")
    layers = [dict(d) for d in soil_layers]
    if not layers:
        raise ValueError("soil_layers must not be empty.")

    base: Dict[str, float] = {"applied_load": float(applied_load)}
    for prop in ("friction_angle", "cohesion", "unit_weight"):
        vals = [d.get(prop) for d in layers if d.get(prop) is not None]
        if vals:
            base[prop] = float(vals[0])  # global default = first carrier
        for i, d in enumerate(layers, start=1):
            if d.get(prop) is not None:
                base[f"{prop}:{i}"] = float(d[prop])

    variables = _fill_means(variables, base, "axial_pile_reliability")
    section = PileSection(**pile)

    def g(values: Dict[str, float]) -> float:
        mod = [dict(d) for d in layers]
        load = float(applied_load)
        for name, v in values.items():
            v = float(v)
            if name == "applied_load":
                load = max(v, 1e-6)
            elif ":" in name:
                prop, idx = name.split(":", 1)
                mod[int(idx) - 1][prop] = max(v, 1e-6)
            else:
                for d in mod:
                    if d.get(name) is not None:
                        d[name] = max(v, 1e-6)
        profile = AxialSoilProfile(
            layers=[AxialSoilLayer(**d) for d in mod], gwt_depth=gwt_depth)
        res = AxialPileAnalysis(
            pile=section, soil=profile, pile_length=pile_length,
            method=method,
        ).compute()
        return res.Q_ultimate / load

    return run_engine(g, variables, engine=engine, correlation=correlation,
                      convention="fos", **engine_kwargs)


# ---------------------------------------------------------------------------
# Slope (delegate)
# ---------------------------------------------------------------------------

def slope_reliability(geom,
                      variables: Dict[str, Dict],
                      engine: str = "fosm",
                      **kwargs):
    """Slope probabilistic analysis — delegates to the validated
    ``slope_stability.probabilistic`` implementation.

    Parameters
    ----------
    geom : slope_stability.geometry.SlopeGeometry
        Slope geometry (layers carry the mean property values).
    variables : dict
        slope_stability variable spec, e.g.
        ``{"cu": {"cov": 0.2, "dist": "lognormal"}}`` (keys may be scoped
        ':LayerName'; means default to layer values).
    engine : str
        'fosm' (Duncan 2000 Taylor series) or 'monte_carlo'.
    **kwargs
        Passed through (xc/yc/radius or slip_surface, method, n_slices,
        n, seed, ...).

    Returns
    -------
    slope_stability.probabilistic.FOSMResult or MonteCarloResult.
    """
    from slope_stability.probabilistic import fosm_fos, monte_carlo_fos
    if engine == "fosm":
        return fosm_fos(geom, variables, **kwargs)
    if engine == "monte_carlo":
        return monte_carlo_fos(geom, variables, **kwargs)
    raise ValueError(
        "slope_reliability engine must be 'fosm' or 'monte_carlo' "
        f"(got '{engine}'). For PEM/FORM on a slope, wrap the slope FOS "
        "in your own g() and call reliability.pem/form directly.")
