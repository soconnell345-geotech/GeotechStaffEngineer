"""Project → analysis inputs: SlopeGeometry, fem2d kwargs, and runs.

The schema's field names mirror the module APIs, so this is a thin,
mechanical mapping:

* :func:`to_slope_geometry`  — Project → slope_stability.SlopeGeometry
  (+ SoilNail / Anchor / Geosynthetic objects).
* :func:`to_fem_kwargs`      — Project → ``analyze_slope_srm`` kwargs
  (surface_points, soil_layers dicts, gwt array, layer_polylines).
* :func:`run_analyses`       — execute every requested analysis and return
  a results dict (LE search + optional FOSM/Monte-Carlo, FEM-SRM).

Builders RAISE on documents that fail validation with errors — the agent is
expected to run :func:`geo_project.validate.validate` (and the human gates)
first; the raise is the engineering backstop, not the UX.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from geo_project.schema import (
    FEMAnalysis,
    LEAnalysis,
    Project,
)
from geo_project.validate import has_errors, summarize, validate


class ProjectValidationError(ValueError):
    """Raised when a builder is invoked on a Project with blocking errors."""

    def __init__(self, issues):
        self.issues = issues
        super().__init__(
            "Project has blocking validation errors:\n" + summarize(
                [i for i in issues if i.level == "error"]))


def _require_valid(project: Project) -> None:
    issues = validate(project)
    if has_errors(issues):
        raise ProjectValidationError(issues)


# ---------------------------------------------------------------------------
# LE: Project -> SlopeGeometry
# ---------------------------------------------------------------------------

def to_slope_geometry(project: Project, check: bool = True):
    """Build a :class:`slope_stability.geometry.SlopeGeometry`.

    Mapping notes
    -------------
    * schema ``strength_model='undrained'`` → SlopeSoilLayer
      ``analysis_mode='undrained'`` (+ cu); the other models map straight to
      the layer's ``strength_model``.
    * a layer's named ``bottom_boundary`` polyline →
      ``bottom_boundary_points`` (position-dependent bottom).
    * ``water.ru`` is applied to every layer's ``ru``.
    * only the FIRST surcharge maps (SlopeGeometry carries one band) —
      validate() warns about extras.

    Parameters
    ----------
    project : Project
    check : bool
        Run validate() first and raise :class:`ProjectValidationError` on
        blocking errors (default True).
    """
    from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
    from slope_stability.nails import SoilNail
    from slope_stability.reinforcement import Anchor as SSAnchor
    from slope_stability.reinforcement import Geosynthetic as SSGeosynthetic

    if check:
        _require_valid(project)

    layers = []
    for i, layer in enumerate(project.stratigraphy):
        m = layer.material
        kwargs: Dict[str, Any] = dict(
            name=layer.name or f"layer_{i}",
            top_elevation=float(project.layer_top(i)),
            bottom_elevation=float(project.layer_bottom(i)),
            gamma=float(m.gamma),
            gamma_sat=(float(m.gamma_sat)
                       if m.gamma_sat is not None else None),
            ru=float(project.water.ru),
        )
        if m.strength_model == "undrained":
            kwargs.update(analysis_mode="undrained", cu=float(m.cu),
                          strength_model="mohr_coulomb")
        elif m.strength_model == "mohr_coulomb":
            kwargs.update(analysis_mode="drained", phi=float(m.phi),
                          c_prime=float(m.c_prime),
                          strength_model="mohr_coulomb")
        elif m.strength_model == "shansep":
            kwargs.update(
                analysis_mode="drained", strength_model="shansep",
                shansep_S=float(m.shansep_S), shansep_m=float(m.shansep_m),
                ocr=float(m.ocr), su_min=float(m.su_min or 0.0))
        elif m.strength_model == "hoek_brown":
            kwargs.update(
                analysis_mode="drained", strength_model="hoek_brown",
                hb_sigci=float(m.hb_sigci), hb_gsi=float(m.hb_gsi),
                hb_mi=float(m.hb_mi), hb_D=float(m.hb_D or 0.0))
        bpts = project.boundary_points(i)
        if bpts:
            kwargs["bottom_boundary_points"] = bpts
        layers.append(SlopeSoilLayer(**kwargs))

    nails = [SoilNail(x_head=n.x_head, z_head=n.z_head, length=n.length,
                      inclination=n.inclination,
                      bar_diameter=n.bar_diameter,
                      drill_hole_diameter=n.drill_hole_diameter,
                      fy=n.fy, bond_stress=n.bond_stress,
                      spacing_h=n.spacing_h)
             for n in project.reinforcement.nails] or None
    anchors = [SSAnchor(x_head=a.x_head, z_head=a.z_head, length=a.length,
                        T_allow=a.T_allow, inclination=a.inclination)
               for a in project.reinforcement.anchors] or None
    geos = [SSGeosynthetic(elevation=g.elevation, T_allow=g.T_allow,
                           x_start=g.x_start, x_end=g.x_end)
            for g in project.reinforcement.geosynthetics] or None

    surcharge = 0.0
    surcharge_x_range = None
    if project.loads.surcharges:
        s = project.loads.surcharges[0]
        surcharge = float(s.q)
        if s.x_start is not None and s.x_end is not None:
            surcharge_x_range = (float(s.x_start), float(s.x_end))

    return SlopeGeometry(
        surface_points=[(float(x), float(z))
                        for x, z in project.geometry.surface_points],
        soil_layers=layers,
        gwt_points=([(float(x), float(z))
                     for x, z in project.water.gwt_points]
                    if project.water.gwt_points else None),
        surcharge=surcharge,
        surcharge_x_range=surcharge_x_range,
        kh=float(project.loads.kh),
        nails=nails,
        anchors=anchors,
        geosynthetics=geos,
    )


# ---------------------------------------------------------------------------
# FEM: Project -> analyze_slope_srm kwargs
# ---------------------------------------------------------------------------

def to_fem_kwargs(project: Project, analysis: Optional[FEMAnalysis] = None,
                  check: bool = True) -> Dict[str, Any]:
    """Build the kwargs dict for :func:`fem2d.analysis.analyze_slope_srm`.

    Returns surface_points, soil_layers (fem2d dict format: name,
    bottom_elevation, E, nu, c, phi, psi, gamma), gwt as an (M, 2) array,
    layer_polylines (when named boundaries exist) and the analysis settings.

    Undrained layers map to c=cu, phi=0 (total stress). SHANSEP/Hoek-Brown
    layers are rejected by validate() (MAT007) before this runs.
    """
    if check:
        _require_valid(project)
    if analysis is None:
        fems = [a for a in project.analyses if isinstance(a, FEMAnalysis)]
        analysis = fems[0] if fems else FEMAnalysis()

    soil_layers: List[Dict[str, Any]] = []
    layer_polylines: List[np.ndarray] = []
    any_boundary = False
    for i, layer in enumerate(project.stratigraphy):
        m = layer.material
        if m.strength_model == "undrained":
            c, phi = float(m.cu), 0.0
        else:
            c, phi = float(m.c_prime or 0.0), float(m.phi or 0.0)
        soil_layers.append({
            "name": layer.name or f"layer_{i}",
            "top_elevation": float(project.layer_top(i)),
            "bottom_elevation": float(project.layer_bottom(i)),
            "gamma": float(m.gamma),
            "phi": phi,
            "c": c,
            "E": float(m.E),
            "nu": float(m.nu),
            "psi": float(m.psi or 0.0),
        })
        bpts = project.boundary_points(i)
        if bpts and i < len(project.stratigraphy) - 1:
            any_boundary = True
            layer_polylines.append(np.array(bpts, dtype=float))
        elif i < len(project.stratigraphy) - 1:
            bot = float(project.layer_bottom(i))
            x_min, x_max = project.geometry.x_range
            layer_polylines.append(
                np.array([(x_min, bot), (x_max, bot)], dtype=float))

    gwt = None
    if project.water.gwt_points:
        gwt = np.array(sorted(project.water.gwt_points,
                              key=lambda p: p[0]), dtype=float)

    kwargs: Dict[str, Any] = {
        "surface_points": [(float(x), float(z))
                           for x, z in project.geometry.surface_points],
        "soil_layers": soil_layers,
        "gwt": gwt,
        "nx": analysis.nx,
        "ny": analysis.ny,
        "srf_tol": analysis.srf_tol,
        "element_type": analysis.element_type,
        "srf_range": tuple(analysis.srf_range),
    }
    if analysis.depth is not None:
        kwargs["depth"] = analysis.depth
    if analysis.x_extend is not None:
        kwargs["x_extend"] = analysis.x_extend
    if any_boundary and layer_polylines:
        kwargs["layer_polylines"] = layer_polylines
    return kwargs


# ---------------------------------------------------------------------------
# Run everything requested
# ---------------------------------------------------------------------------

def _probabilistic_variables(project: Project, spec) -> Dict[str, Dict]:
    """Resolve the probabilistic variables dict for an LE analysis.

    An explicit ``spec.variables`` wins. Otherwise it is assembled from each
    layer Material's ``probabilistic`` entries, scoped per layer
    (``"phi:LayerName"``) so per-layer COVs stay distinct.
    """
    if spec.variables:
        return dict(spec.variables)
    variables: Dict[str, Dict] = {}
    for layer in project.stratigraphy:
        for param, v in layer.material.probabilistic.items():
            key = param if ":" in param else f"{param}:{layer.name}"
            entry = {"cov": float(v.get("cov", 0.1)),
                     "dist": v.get("dist", "lognormal")}
            if v.get("mean") is not None:
                entry["mean"] = float(v["mean"])
            variables[key] = entry
    return variables


def run_analyses(project: Project) -> Dict[str, Any]:
    """Execute every analysis in ``project.analyses``.

    Returns ``{analysis_name: {...}}``. LE entries carry the SearchResult's
    critical surface (FOS, method, xc/yc/R) plus optional 'probabilistic'
    results (FOSMResult / MonteCarloResult fields). FEM entries carry the
    FEMResult summary (FOS, converged, mesh size).

    Raw result objects are returned under '_result' keys for downstream
    plotting/calc-package use; everything else is JSON-safe scalars.
    """
    _require_valid(project)
    out: Dict[str, Any] = {}

    geom = None
    for k, a in enumerate(project.analyses):
        name = a.name or f"analysis_{k}"
        if isinstance(a, LEAnalysis):
            from slope_stability.analysis import search_critical_surface
            if geom is None:
                geom = to_slope_geometry(project, check=False)
            search = search_critical_surface(
                geom,
                method=a.method,
                n_slices=a.n_slices,
                surface_type=a.search.surface_type,
                nx=a.search.nx, ny=a.search.ny,
                n_trials=a.search.n_trials,
                n_points=a.search.n_points,
                seed=a.search.seed,
            )
            crit = search.critical
            entry: Dict[str, Any] = {
                "type": "le",
                "method": a.method,
                "FOS": float(crit.FOS) if crit is not None else None,
                "n_surfaces_evaluated": search.n_surfaces_evaluated,
                "critical": (None if crit is None else {
                    "xc": crit.xc, "yc": crit.yc, "radius": crit.radius,
                    "x_entry": crit.x_entry, "x_exit": crit.x_exit,
                }),
                "_result": search,
            }
            if a.probabilistic is not None and crit is not None:
                variables = _probabilistic_variables(project, a.probabilistic)
                if variables:
                    entry["probabilistic"] = _run_probabilistic(
                        geom, a, crit, variables)
                else:
                    entry["probabilistic"] = {
                        "error": "no probabilistic variables defined "
                                 "(set material.probabilistic or "
                                 "analysis.probabilistic.variables)"}
            out[name] = entry
        elif isinstance(a, FEMAnalysis):
            from fem2d.analysis import analyze_slope_srm
            kwargs = to_fem_kwargs(project, analysis=a, check=False)
            result = analyze_slope_srm(**kwargs)
            out[name] = {
                "type": "fem_srm",
                "FOS": (float(result.FOS)
                        if getattr(result, "FOS", None) is not None else None),
                "converged": bool(result.converged),
                "n_nodes": result.n_nodes,
                "n_elements": result.n_elements,
                "_result": result,
            }
    return out


def _run_probabilistic(geom, a: LEAnalysis, crit, variables) -> Dict[str, Any]:
    """FOSM or Monte Carlo on the critical surface; JSON-safe summary."""
    spec = a.probabilistic
    common = dict(method=a.method, n_slices=a.n_slices)
    if crit.radius > 0:
        common.update(xc=crit.xc, yc=crit.yc, radius=crit.radius)
    else:
        from slope_stability.slip_surface import PolylineSlipSurface
        common.update(slip_surface=PolylineSlipSurface(crit.slip_points))
    if spec.kind == "monte_carlo":
        from slope_stability.probabilistic import monte_carlo_fos
        mc = monte_carlo_fos(geom, variables, n=spec.n, seed=spec.seed,
                             keep_samples=False, **common)
        return {
            "kind": "monte_carlo", "n": mc.n,
            "fos_mean": mc.fos_mean, "fos_cov": mc.fos_cov,
            "pf": mc.pf, "pf_lognormal": mc.pf_lognormal,
            "beta_lognormal": mc.beta_lognormal,
            "variables": variables,
            "_result": mc,
        }
    from slope_stability.probabilistic import fosm_fos
    fr = fosm_fos(geom, variables, **common)
    return {
        "kind": "fosm",
        "fos_mlv": fr.fos_mlv, "sigma_f": fr.sigma_f, "cov_f": fr.cov_f,
        "beta_normal": fr.beta_normal, "beta_lognormal": fr.beta_lognormal,
        "pf_normal": fr.pf_normal, "pf_lognormal": fr.pf_lognormal,
        "variables": variables,
        "_result": fr,
    }


__all__ = [
    "ProjectValidationError",
    "to_slope_geometry",
    "to_fem_kwargs",
    "run_analyses",
]
