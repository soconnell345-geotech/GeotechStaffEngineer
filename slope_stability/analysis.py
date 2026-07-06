"""
Top-level slope stability analysis orchestrator.

Provides analyze_slope() and search_critical_surface() as the
primary entry points, following the project pattern of analyze_*()
functions returning result dataclasses.

References:
    Duncan, Wright & Brandon (2014) — Soil Strength and Slope Stability
"""

import math
from typing import Optional, Tuple

from slope_stability.geometry import SlopeGeometry
from slope_stability.slip_surface import CircularSlipSurface
from slope_stability.slices import build_slices, compute_slice_forces
from slope_stability.methods import (
    fellenius_fos, bishop_fos, spencer_fos, morgenstern_price_fos,
)
from slope_stability.gle import janbu_fos
from slope_stability.search import (
    grid_search, search_noncircular,
    search_pso, search_weak_layer_biased, search_entry_exit, search_de,
)
from slope_stability.results import (
    SlopeStabilityResult, SliceData, SearchResult, InfiniteSlopeResult,
)


def analyze_slope(geom: SlopeGeometry,
                  xc: float = None,
                  yc: float = None,
                  radius: float = None,
                  slip_surface=None,
                  method: str = "bishop",
                  n_slices: int = 30,
                  tol: float = 1e-4,
                  f_interslice: str = "half_sine",
                  include_slice_data: bool = False,
                  compare_methods: bool = False,
                  ) -> SlopeStabilityResult:
    """Run slope stability analysis for a specified slip surface.

    Parameters
    ----------
    geom : SlopeGeometry
        Complete slope definition.
    xc : float, optional
        Circle center x-coordinate (m). Used with yc/radius for circular.
    yc : float, optional
        Circle center z-coordinate / elevation (m).
    radius : float, optional
        Circle radius (m).
    slip_surface : CircularSlipSurface or PolylineSlipSurface, optional
        Explicit slip surface object. If provided, xc/yc/radius are ignored.
    method : str
        'fellenius', 'bishop', 'janbu', 'spencer', 'morgenstern_price'
        or 'gle'. 'gle' is an alias for the rigorous Morgenstern-Price
        solution with a selectable interslice force function. 'janbu'
        reports the f0-corrected FOS (uncorrected value in
        FOS_janbu_uncorrected). Default 'bishop'.
    n_slices : int
        Number of slices. Default 30.
    tol : float
        Convergence tolerance for iterative methods. Default 1e-4.
    f_interslice : str
        Interslice force function for 'morgenstern_price' / 'gle':
        'constant', 'half_sine' (default), 'clipped_sine', 'trapezoidal'.
    include_slice_data : bool
        If True, include per-slice breakdown in results.
    compare_methods : bool
        If True, compute FOS for all four methods (Fellenius, Bishop,
        Spencer, Morgenstern-Price). For noncircular surfaces, Bishop
        is skipped (requires circular).

    Returns
    -------
    SlopeStabilityResult
    """
    # Build slip surface from either explicit object or xc/yc/radius
    if slip_surface is not None:
        slip = slip_surface
    else:
        if xc is None or yc is None or radius is None:
            raise ValueError(
                "Either provide slip_surface or all of xc, yc, radius"
            )
        slip = CircularSlipSurface(xc, yc, radius)

    is_circular = getattr(slip, 'is_circular', True)
    x_entry, x_exit = slip.find_entry_exit(geom)
    slices = build_slices(geom, slip, n_slices)

    # Reinforcement crossings (nails / anchors / geosynthetics)
    from slope_stability.reinforcement import compute_reinforcement_forces
    reinf_forces = compute_reinforcement_forces(geom, slip, x_entry, x_exit)
    if not reinf_forces:
        reinf_forces = None

    # For result: store circle params if circular, zeros otherwise
    r_xc = slip.xc if is_circular else 0.0
    r_yc = slip.yc if is_circular else 0.0
    r_radius = slip.radius if is_circular else 0.0
    r_slip_points = None if is_circular else list(slip.points)

    # For noncircular, force Spencer (Bishop doesn't work)
    if not is_circular and method == "bishop":
        method = "spencer"

    # Primary FOS
    gle_result = None      # rich GLE state for rigorous methods
    theta_spencer = None
    fos_spencer = None
    fos_mp = None
    lambda_mp = None
    fos_janbu = None
    fos_janbu_uncorr = None
    f0_janbu = None
    if method == "fellenius":
        fos = fellenius_fos(slices, slip, reinf_forces=reinf_forces)
        method_name = "Fellenius"
    elif method == "janbu":
        fos_c, fos_u, f0 = janbu_fos(slices, slip, tol=min(tol, 1e-4) * 0.1,
                                     reinf_forces=reinf_forces)
        fos = fos_c
        fos_janbu = fos_c
        fos_janbu_uncorr = fos_u
        f0_janbu = f0
        method_name = "Janbu"
    elif method == "spencer":
        gle_result = _try_gle(slices, slip, "constant", tol, reinf_forces)
        if gle_result is not None:
            fos = gle_result.fos
            theta = math.degrees(math.atan(gle_result.lam))
        else:
            fos, theta = spencer_fos(slices, slip, tol=tol,
                                     reinf_forces=reinf_forces)
        fos_spencer = fos
        theta_spencer = theta
        method_name = "Spencer"
    elif method in ("morgenstern_price", "gle"):
        gle_result = _try_gle(slices, slip, f_interslice, tol, reinf_forces)
        if gle_result is not None:
            fos, lam = gle_result.fos, gle_result.lam
        else:
            fos, lam = morgenstern_price_fos(
                slices, slip, f_interslice=f_interslice, tol=tol,
                reinf_forces=reinf_forces)
        fos_mp = fos
        lambda_mp = lam
        method_name = ("GLE" if method == "gle" else "Morgenstern-Price")
    else:
        fos = bishop_fos(slices, slip, tol=tol, reinf_forces=reinf_forces)
        method_name = "Bishop"

    # Comparison FOS values
    fos_fellenius = None
    fos_bishop = None
    if compare_methods:
        fos_fellenius = fellenius_fos(slices, slip,
                                      reinf_forces=reinf_forces)
        if is_circular:
            fos_bishop = bishop_fos(slices, slip, tol=tol,
                                    reinf_forces=reinf_forces)
        if fos_spencer is None:
            fos_sp, theta = spencer_fos(slices, slip, tol=tol,
                                        reinf_forces=reinf_forces)
            fos_spencer = fos_sp
            theta_spencer = theta
        if fos_mp is None:
            fos_mp, lambda_mp = morgenstern_price_fos(
                slices, slip, f_interslice=f_interslice, tol=tol,
                reinf_forces=reinf_forces)
        if fos_janbu is None:
            fos_janbu, fos_janbu_uncorr, f0_janbu = janbu_fos(
                slices, slip, tol=min(tol, 1e-4) * 0.1,
                reinf_forces=reinf_forces)

    # Slice data (per-slice force table) and thrust line
    slice_data = None
    thrust_line = None
    if include_slice_data:
        slice_data = []
        rich = gle_result  # rigorous per-slice/boundary forces, if any
        for i, s in enumerate(slices):
            sf = compute_slice_forces(s)
            dl = s.base_length
            U = s.pore_pressure * dl
            if rich is not None:
                # rigorous base forces at the converged GLE state
                N_eff = rich.base_normal_eff[i]
                S_mob = rich.shear_mobilized[i]
                T_avail = S_mob * rich.fos
                E_l, E_r = rich.interslice_E[i], rich.interslice_E[i + 1]
                X_l, X_r = rich.interslice_X[i], rich.interslice_X[i + 1]
            else:
                # Fellenius decomposition (no interslice forces)
                N_eff = sf.N_prime
                S_mob = sf.S_mobilized
                T_avail = sf.T_available
                E_l = E_r = X_l = X_r = None
            sigma_n = N_eff / dl if dl > 0 else 0.0
            tau_mob = S_mob / dl if dl > 0 else 0.0
            tau_avail = T_avail / dl if dl > 0 else 0.0
            slice_data.append(SliceData(
                x_mid=s.x_mid,
                z_top=s.z_top,
                z_base=s.z_base,
                width=s.width,
                height=s.height,
                alpha_deg=math.degrees(s.alpha),
                weight=s.weight,
                pore_pressure=s.pore_pressure,
                c=s.c,
                phi=s.phi,
                base_length=s.base_length,
                normal_stress_kPa=sigma_n,
                shear_stress_kPa=tau_mob,
                shear_resistance_kPa=tau_avail,
                in_tension_crack=s.in_tension_crack,
                N_eff_kN=N_eff,
                S_mob_kN=S_mob,
                U_base_kN=U,
                E_left_kN=E_l,
                E_right_kN=E_r,
                X_left_kN=X_l,
                X_right_kN=X_r,
            ))
        if rich is not None and rich.boundary_x:
            thrust_line = list(zip(rich.boundary_x, rich.thrust_elevation))

    return SlopeStabilityResult(
        FOS=fos,
        method=method_name,
        xc=r_xc,
        yc=r_yc,
        radius=r_radius,
        x_entry=x_entry,
        x_exit=x_exit,
        theta_spencer=theta_spencer,
        FOS_fellenius=fos_fellenius,
        FOS_bishop=fos_bishop,
        FOS_spencer=fos_spencer,
        FOS_morgenstern_price=fos_mp,
        lambda_mp=lambda_mp,
        FOS_janbu=fos_janbu,
        FOS_janbu_uncorrected=fos_janbu_uncorr,
        janbu_f0=f0_janbu,
        reinforcements=[
            {
                "kind": f.kind, "index": f.index,
                "x_m": round(f.x, 3), "z_m": round(f.z, 3),
                "T_kN_per_m": round(f.T, 2),
                "controlled_by": f.controlled_by,
            } for f in (reinf_forces or [])
        ] or None,
        n_slices=len(slices),
        has_seismic=geom.kh > 0,
        kh=geom.kh,
        slice_data=slice_data,
        thrust_line=thrust_line,
        slip_points=r_slip_points,
        tension_crack_depth=geom.tension_crack_depth,
        tension_crack_water_depth=geom.tension_crack_water_depth,
    )


def _try_gle(slices, slip, f_interslice, tol, reinf_forces):
    """Run the rigorous GLE engine; return the GLEResult or None.

    Mirrors the convergence acceptance used by the spencer_fos /
    morgenstern_price_fos wrappers so analyze_slope can keep the rich
    per-slice state (interslice E/X, thrust line) without solving twice.
    """
    from slope_stability.gle import gle_fos
    try:
        res = gle_fos(slices, slip, f_interslice=f_interslice,
                      tol=min(tol, 1e-4) * 0.1, reinf_forces=reinf_forces)
        if res.converged and 0.0 < res.fos < 900.0:
            return res
    except (ValueError, ZeroDivisionError, OverflowError):
        pass
    return None


def compare_methods_table(geom: SlopeGeometry,
                          xc: float = None,
                          yc: float = None,
                          radius: float = None,
                          slip_surface=None,
                          n_slices: int = 30,
                          tol: float = 1e-4,
                          f_interslice: str = "half_sine") -> dict:
    """Run ALL limit-equilibrium methods on one slip surface.

    The classic Fredlund & Krahn (1977) style comparison: one surface,
    every method, side by side — useful for judging method sensitivity
    and for validation against published tables.

    Parameters
    ----------
    geom : SlopeGeometry
    xc, yc, radius : float, optional
        Trial circle (or pass slip_surface).
    slip_surface : CircularSlipSurface or PolylineSlipSurface, optional
    n_slices : int
    tol : float
    f_interslice : str
        Interslice function for the Morgenstern-Price row.

    Returns
    -------
    dict with keys:
        'rows' : list of dicts {method, FOS, detail}
        'surface' : surface description dict
        'summary' : formatted text table
    """
    res = analyze_slope(geom, xc=xc, yc=yc, radius=radius,
                        slip_surface=slip_surface, method="bishop"
                        if (slip_surface is None or
                            getattr(slip_surface, "is_circular", True))
                        else "spencer",
                        n_slices=n_slices, tol=tol,
                        f_interslice=f_interslice, compare_methods=True)

    rows = []
    if res.FOS_fellenius is not None:
        rows.append({"method": "Fellenius (OMS)",
                     "FOS": round(res.FOS_fellenius, 4), "detail": ""})
    if res.FOS_bishop is not None:
        rows.append({"method": "Bishop simplified",
                     "FOS": round(res.FOS_bishop, 4), "detail": ""})
    if res.FOS_janbu_uncorrected is not None:
        rows.append({"method": "Janbu simplified (uncorrected)",
                     "FOS": round(res.FOS_janbu_uncorrected, 4),
                     "detail": ""})
    if res.FOS_janbu is not None:
        rows.append({"method": "Janbu simplified (corrected)",
                     "FOS": round(res.FOS_janbu, 4),
                     "detail": f"f0={res.janbu_f0:.3f}"
                     if res.janbu_f0 else ""})
    if res.FOS_spencer is not None:
        rows.append({"method": "Spencer",
                     "FOS": round(res.FOS_spencer, 4),
                     "detail": f"theta={res.theta_spencer:.2f} deg"
                     if res.theta_spencer is not None else ""})
    if res.FOS_morgenstern_price is not None:
        rows.append({"method": f"Morgenstern-Price ({f_interslice})",
                     "FOS": round(res.FOS_morgenstern_price, 4),
                     "detail": f"lambda={res.lambda_mp:.3f}"
                     if res.lambda_mp is not None else ""})

    surface = {"type": "circular" if res.is_circular else "noncircular",
               "xc": res.xc, "yc": res.yc, "radius": res.radius,
               "x_entry": round(res.x_entry, 3),
               "x_exit": round(res.x_exit, 3)}

    lines = ["Method comparison (one surface, all methods)",
             "-" * 52]
    for r in rows:
        lines.append(f"  {r['method']:<34s} {r['FOS']:>7.3f}  {r['detail']}")
    summary = "\n".join(lines)

    return {"rows": rows, "surface": surface, "summary": summary}


def search_critical_surface(
    geom: SlopeGeometry,
    x_range: Tuple[float, float] = None,
    y_range: Tuple[float, float] = None,
    nx: int = 10,
    ny: int = 10,
    method: str = "bishop",
    n_slices: int = 30,
    tol: float = 1e-4,
    surface_type: str = "circular",
    x_entry_range: Tuple[float, float] = None,
    x_exit_range: Tuple[float, float] = None,
    n_trials: int = 500,
    n_points: int = 5,
    seed: Optional[int] = None,
) -> SearchResult:
    """Search for the critical slip surface (minimum FOS).

    Auto-computes search bounds from slope geometry if not provided.

    Parameters
    ----------
    geom : SlopeGeometry
        Slope definition.
    x_range : (float, float), optional
        Circle center x search range (circular only).
    y_range : (float, float), optional
        Circle center y search range (circular only).
    nx, ny : int
        Grid resolution. Default 10x10.
    method : str
        FOS method. Default 'bishop'.
    n_slices : int
        Number of slices per analysis.
    tol : float
        Convergence tolerance for iterative methods. Default 1e-4.
    surface_type : str
        'circular' (default, centre-grid search), 'entry_exit' (circular
        arcs between entry/exit windows), 'noncircular' (random
        polylines), 'noncircular_de' (differential-evolution refinement
        seeded by a random search), 'pso', or 'weak_layer'.
    x_entry_range : (float, float), optional
        Allowed entry x-coordinate range.
    x_exit_range : (float, float), optional
        Allowed exit x-coordinate range.
    n_trials : int
        Number of random trials for noncircular search. Default 500.
    n_points : int
        Number of polyline vertices for noncircular search. Default 5.
    seed : int, optional
        Random seed for noncircular search reproducibility.

    Returns
    -------
    SearchResult
    """
    # Auto-compute entry/exit ranges from slope geometry if not provided
    x_min_geo = geom.surface_points[0][0]
    x_max_geo = geom.surface_points[-1][0]
    if x_entry_range is None:
        x_entry_range = (x_min_geo, x_min_geo + (x_max_geo - x_min_geo) * 0.4)
    if x_exit_range is None:
        x_exit_range = (x_min_geo + (x_max_geo - x_min_geo) * 0.6, x_max_geo)

    noncirc_method = method if method != "bishop" else "spencer"

    if surface_type == "noncircular":
        return search_noncircular(
            geom,
            x_entry_range=x_entry_range,
            x_exit_range=x_exit_range,
            n_trials=n_trials,
            n_points=n_points,
            n_slices=n_slices,
            tol=tol,
            seed=seed,
            method=noncirc_method,
        )

    if surface_type == "noncircular_de":
        return search_de(
            geom,
            x_entry_range=x_entry_range,
            x_exit_range=x_exit_range,
            n_points=max(n_points, 5),
            method=noncirc_method,
            n_slices=n_slices,
            tol=tol,
            seed=seed,
        )

    if surface_type == "pso":
        return search_pso(
            geom,
            x_entry_range=x_entry_range,
            x_exit_range=x_exit_range,
            n_particles=max(nx * ny, 30),
            n_iterations=n_trials // max(nx * ny, 30) if n_trials > 30 else 50,
            n_points=n_points,
            n_slices=n_slices,
            tol=tol,
            seed=seed,
        )

    if surface_type == "weak_layer":
        return search_weak_layer_biased(
            geom,
            x_entry_range=x_entry_range,
            x_exit_range=x_exit_range,
            n_trials=n_trials,
            n_points=n_points,
            n_slices=n_slices,
            tol=tol,
            seed=seed,
        )

    if surface_type == "entry_exit":
        return search_entry_exit(
            geom,
            x_entry_range=x_entry_range,
            x_exit_range=x_exit_range,
            n_entry=nx,
            n_exit=ny,
            method=method,
            n_slices=n_slices,
            tol=tol,
        )

    # Circular search
    if x_range is None:
        x_min = geom.surface_points[0][0]
        x_max = geom.surface_points[-1][0]
        x_range = (x_min, x_max)

    if y_range is None:
        z_min = min(z for _, z in geom.surface_points)
        z_max = max(z for _, z in geom.surface_points)
        slope_height = z_max - z_min
        y_range = (z_max + 1.0, z_max + 2.0 * slope_height)

    result = grid_search(geom, x_range, y_range, nx, ny, method, n_slices,
                         tol=tol, x_entry_range=x_entry_range,
                         x_exit_range=x_exit_range)

    return result


def rapid_drawdown_fos(geom: SlopeGeometry,
                       drawdown_from_elevation: float,
                       drawdown_to_elevation: float,
                       **kwargs):
    """Rapid drawdown analysis — NOT YET IMPLEMENTED (designed stub).

    Planned implementation (Duncan, Wright & Brandon 2014, Ch. 9 /
    USACE three-stage procedure):

    1. Stage 1: long-term (drained) analysis with the high pool
       (consolidation stresses sigma'_fc and tau_fc on the slip surface
       from the pre-drawdown steady state).
    2. Stage 2: undrained analysis after drawdown using composite
       drained/undrained strength envelopes per slice — su tied to the
       stage-1 consolidation stresses (Kc-dependent strengths).
    3. Stage 3: drained check with the low pool; the governing FOS per
       slice is min(undrained, drained).

    Requires per-slice consolidation-stress bookkeeping and a composite
    strength interpolation that the current Slice pipeline does not yet
    carry. Until then, model drawdown conservatively by keeping the GWT
    high inside the slope with the pond removed (gwt_points unchanged,
    surface water at the drawn-down level).

    Raises
    ------
    NotImplementedError
    """
    raise NotImplementedError(
        "rapid_drawdown_fos is a designed stub (see docstring for the "
        "planned Duncan 3-stage procedure). Conservative interim "
        "approach: analyze with the pre-drawdown GWT inside the slope "
        "and the external pool at the drawn-down elevation."
    )


def infinite_slope_fos(slope_angle: float,
                       phi: float,
                       gamma: float,
                       c: float = 0.0,
                       depth: float = 1.0,
                       water_condition: str = "dry",
                       gamma_w: float = 9.81,
                       ru: float = 0.0,
                       water_depth: float = 0.0) -> InfiniteSlopeResult:
    """Infinite-slope (planar translational) factor of safety — closed form.

    Effective-stress limit equilibrium of a slip plane parallel to an
    infinitely long slope at depth ``z`` below the surface (Duncan, Wright &
    Brandon 2014, Ch. 14; Skempton & DeLory 1957):

        sigma_n = gamma * z * cos^2(beta)             (total normal on plane)
        tau     = gamma * z * sin(beta) * cos(beta)   (driving shear)
        FOS     = [ c' + (sigma_n - u) * tan(phi') ] / tau

    Pore pressure ``u`` on the slip plane by ``water_condition``:

    * ``'dry'``              : u = 0.
    * ``'seepage_parallel'`` : steady seepage parallel to the slope, phreatic
      surface a depth ``water_depth`` below the ground;
      u = gamma_w * (z - water_depth) * cos^2(beta)  (0 if z <= water_depth).
      ``water_depth = 0`` -> water table at the surface (the usual case).
    * ``'ru'``               : u = ru * gamma * z  (pore-pressure ratio).

    For a cohesionless soil (c' = 0) the depth cancels:
        dry:                FOS = tan(phi') / tan(beta)
        seepage (d_w = 0):  FOS = (gamma'/gamma) * tan(phi') / tan(beta).

    Parameters
    ----------
    slope_angle : float
        Slope inclination beta (degrees from horizontal), in (0, 90).
    phi : float
        Effective friction angle (degrees).
    gamma : float
        Total (moist / saturated) unit weight (kN/m3).
    c : float
        Effective cohesion (kPa). Default 0 (cohesionless).
    depth : float
        Slip-plane depth z below the surface (m). Cancels for c' = 0; must be
        positive. Default 1.0.
    water_condition : str
        'dry' (default), 'seepage_parallel', or 'ru'.
    gamma_w : float
        Unit weight of water (kN/m3). Default 9.81.
    ru : float
        Pore-pressure ratio, used when ``water_condition='ru'``. Default 0.
    water_depth : float
        Depth of the phreatic surface below the ground (m), used when
        ``water_condition='seepage_parallel'``. Default 0 (table at surface).

    Returns
    -------
    InfiniteSlopeResult
    """
    if not (0.0 < slope_angle < 90.0):
        raise ValueError(
            f"slope_angle must be in (0, 90) deg, got {slope_angle}")
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth}")
    if phi < 0 or c < 0:
        raise ValueError("phi and c must be non-negative")
    if water_condition not in ("dry", "seepage_parallel", "ru"):
        raise ValueError(
            "water_condition must be 'dry', 'seepage_parallel' or 'ru', "
            f"got '{water_condition}'")

    beta = math.radians(slope_angle)
    cosb, sinb = math.cos(beta), math.sin(beta)
    sigma_n = gamma * depth * cosb * cosb
    tau = gamma * depth * sinb * cosb

    if water_condition == "seepage_parallel":
        head = max(depth - water_depth, 0.0)
        u = gamma_w * head * cosb * cosb
    elif water_condition == "ru":
        u = ru * gamma * depth
    else:
        u = 0.0

    sigma_eff = max(sigma_n - u, 0.0)
    resist = c + sigma_eff * math.tan(math.radians(phi))
    fos = resist / tau if tau > 0 else float("inf")

    return InfiniteSlopeResult(
        FOS=fos, slope_angle=slope_angle, depth=depth, phi=phi, c=c,
        gamma=gamma, water_condition=water_condition,
        normal_stress=sigma_n, shear_stress=tau, pore_pressure=u, ru=ru,
    )
