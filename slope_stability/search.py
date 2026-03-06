"""
Critical slip surface search algorithms.

Algorithms:
1. Grid search — 2D grid of trial centers (xc, yc), optimizes radius at each.
2. Random noncircular — random polyline generation with Spencer's method.
3. PSO noncircular — Particle Swarm Optimization over control points.
4. Weak-layer biased — SNIFF-inspired search biasing toward weak layers.
5. Entry/exit region — generates trial arcs from entry/exit zones.

References:
    Duncan, Wright & Brandon (2014) — Chapter 14
    Kennedy & Eberhart (1995) — Particle Swarm Optimization
    Borselli (2002) — SNIFF Random Search concept
"""

import math
import random
import warnings
from typing import List, Tuple, Optional, Dict, Any

from slope_stability.geometry import SlopeGeometry
from slope_stability.slip_surface import CircularSlipSurface, PolylineSlipSurface
from slope_stability.slices import build_slices
from slope_stability.methods import fellenius_fos, bishop_fos, spencer_fos, morgenstern_price_fos
from slope_stability.results import SearchResult, SlopeStabilityResult


_FOS_MAX = 999.9


def _compute_fos(geom: SlopeGeometry,
                 slip,
                 method: str,
                 n_slices: int,
                 tol: float = 1e-4) -> float:
    """Safely compute FOS, returning _FOS_MAX on any error."""
    try:
        slices = build_slices(geom, slip, n_slices)
        if len(slices) < 3:
            return _FOS_MAX

        if method == "fellenius":
            fos = fellenius_fos(slices, slip)
        elif method == "bishop":
            if not getattr(slip, 'is_circular', True):
                # Bishop can't handle noncircular, fall back to Spencer
                fos, _ = spencer_fos(slices, slip, tol=tol)
            else:
                fos = bishop_fos(slices, slip, tol=tol)
        elif method == "spencer":
            fos, _ = spencer_fos(slices, slip, tol=tol)
        elif method == "morgenstern_price":
            fos, _ = morgenstern_price_fos(slices, slip, tol=tol)
        else:
            fos = bishop_fos(slices, slip, tol=tol)

        # Guard against non-convergence producing negative or absurd values
        # FOS < 0.05 is physically meaningless and indicates numerical issues
        if fos < 0.05 or math.isnan(fos) or math.isinf(fos):
            return _FOS_MAX
        return fos
    except (ValueError, ZeroDivisionError, RuntimeError):
        return _FOS_MAX


def _check_entry_exit(geom, slip, x_entry_range, x_exit_range):
    """Check if slip surface entry/exit are within allowed x-ranges.

    Returns True if acceptable, False if should be rejected.
    """
    if x_entry_range is None and x_exit_range is None:
        return True
    try:
        x_entry, x_exit = slip.find_entry_exit(geom)
    except ValueError:
        return False

    if x_entry_range is not None:
        if x_entry < x_entry_range[0] or x_entry > x_entry_range[1]:
            return False
    if x_exit_range is not None:
        if x_exit < x_exit_range[0] or x_exit > x_exit_range[1]:
            return False
    return True


def optimize_radius(geom: SlopeGeometry,
                    xc: float, yc: float,
                    method: str = "bishop",
                    n_slices: int = 30,
                    r_min: Optional[float] = None,
                    r_max: Optional[float] = None,
                    n_radii: int = 20,
                    tol: float = 1e-4,
                    x_entry_range: Optional[Tuple[float, float]] = None,
                    x_exit_range: Optional[Tuple[float, float]] = None,
                    ) -> Tuple[float, float]:
    """Find the radius that minimizes FOS for a given circle center.

    Uses a coarse scan followed by golden-section refinement.

    Parameters
    ----------
    geom : SlopeGeometry
        Slope definition.
    xc, yc : float
        Circle center coordinates.
    method : str
        'fellenius', 'bishop', or 'spencer'. Default 'bishop'.
    n_slices : int
        Number of slices.
    r_min, r_max : float, optional
        Radius search bounds. Auto-computed if None.
    n_radii : int
        Number of trial radii in coarse scan.
    tol : float
        Convergence tolerance. Default 1e-4.
    x_entry_range : (float, float), optional
        Allowed entry x-coordinate range. Surfaces outside are rejected.
    x_exit_range : (float, float), optional
        Allowed exit x-coordinate range. Surfaces outside are rejected.

    Returns
    -------
    (R_opt, FOS_min) : tuple
        Optimal radius and minimum FOS.
    """
    # Auto-compute radius bounds
    if r_min is None or r_max is None:
        zs = [p[1] for p in geom.surface_points]
        z_max_ground = max(zs)

        dist_to_ground = yc - z_max_ground
        if dist_to_ground <= 0:
            dist_to_ground = 0.5
        if r_min is None:
            r_min = max(dist_to_ground + 0.5, 1.0)

        min_layer_bot = min(L.bottom_elevation for L in geom.soil_layers)
        max_dist = math.sqrt((xc - geom.surface_points[0][0])**2 +
                             (yc - min_layer_bot)**2)
        if r_max is None:
            r_max = max(max_dist, r_min + 5.0)

    if r_min >= r_max:
        r_max = r_min + 5.0

    # Coarse scan
    best_r = r_min
    best_fos = _FOS_MAX

    for i in range(n_radii):
        r = r_min + (r_max - r_min) * i / max(n_radii - 1, 1)
        slip = CircularSlipSurface(xc, yc, r)

        # Entry/exit range check
        if not _check_entry_exit(geom, slip, x_entry_range, x_exit_range):
            continue

        fos = _compute_fos(geom, slip, method, n_slices, tol=tol)
        if fos < best_fos:
            best_fos = fos
            best_r = r

    # Golden-section refinement around best_r
    golden = (math.sqrt(5) - 1) / 2
    margin = (r_max - r_min) / n_radii
    a = max(r_min, best_r - 2 * margin)
    b = min(r_max, best_r + 2 * margin)

    for _ in range(20):
        if b - a < 0.01:
            break
        r1 = b - golden * (b - a)
        r2 = a + golden * (b - a)

        slip1 = CircularSlipSurface(xc, yc, r1)
        slip2 = CircularSlipSurface(xc, yc, r2)

        ok1 = _check_entry_exit(geom, slip1, x_entry_range, x_exit_range)
        ok2 = _check_entry_exit(geom, slip2, x_entry_range, x_exit_range)

        f1 = _compute_fos(geom, slip1, method, n_slices, tol=tol) if ok1 else _FOS_MAX
        f2 = _compute_fos(geom, slip2, method, n_slices, tol=tol) if ok2 else _FOS_MAX

        if f1 < f2:
            b = r2
            if f1 < best_fos:
                best_fos = f1
                best_r = r1
        else:
            a = r1
            if f2 < best_fos:
                best_fos = f2
                best_r = r2

    return (best_r, best_fos)


def grid_search(geom: SlopeGeometry,
                x_range: Tuple[float, float],
                y_range: Tuple[float, float],
                nx: int = 10,
                ny: int = 10,
                method: str = "bishop",
                n_slices: int = 30,
                tol: float = 1e-4,
                x_entry_range: Optional[Tuple[float, float]] = None,
                x_exit_range: Optional[Tuple[float, float]] = None,
                ) -> SearchResult:
    """Grid search for the critical circular slip surface.

    Parameters
    ----------
    geom : SlopeGeometry
        Slope definition.
    x_range : (float, float)
        (x_min, x_max) for circle center search.
    y_range : (float, float)
        (y_min, y_max) for circle center search.
    nx, ny : int
        Grid resolution. Default 10x10.
    method : str
        FOS method. Default 'bishop'.
    n_slices : int
        Number of slices per analysis.
    tol : float
        Convergence tolerance. Default 1e-4.
    x_entry_range : (float, float), optional
        Allowed entry x-coordinate range.
    x_exit_range : (float, float), optional
        Allowed exit x-coordinate range.

    Returns
    -------
    SearchResult
    """
    best_fos = _FOS_MAX
    best_xc = 0.0
    best_yc = 0.0
    best_r = 0.0
    grid_fos = []
    n_evaluated = 0

    x_lo, x_hi = x_range
    y_lo, y_hi = y_range

    for ix in range(nx):
        xc = x_lo + (x_hi - x_lo) * ix / max(nx - 1, 1) if nx > 1 else (x_lo + x_hi) / 2
        for iy in range(ny):
            yc = y_lo + (y_hi - y_lo) * iy / max(ny - 1, 1) if ny > 1 else (y_lo + y_hi) / 2

            r_opt, fos_opt = optimize_radius(
                geom, xc, yc, method=method, n_slices=n_slices, tol=tol,
                x_entry_range=x_entry_range, x_exit_range=x_exit_range,
            )
            n_evaluated += 1

            grid_fos.append({
                "xc": round(xc, 2),
                "yc": round(yc, 2),
                "R": round(r_opt, 2),
                "FOS": round(fos_opt, 4),
            })

            if fos_opt < best_fos:
                best_fos = fos_opt
                best_xc = xc
                best_yc = yc
                best_r = r_opt

    # Build result for the critical surface
    critical = None
    if best_fos < _FOS_MAX:
        slip = CircularSlipSurface(best_xc, best_yc, best_r)
        try:
            x_entry, x_exit = slip.find_entry_exit(geom)
            slices = build_slices(geom, slip, n_slices)
            critical = SlopeStabilityResult(
                FOS=best_fos,
                method=method.capitalize(),
                xc=best_xc,
                yc=best_yc,
                radius=best_r,
                x_entry=x_entry,
                x_exit=x_exit,
                n_slices=len(slices),
                has_seismic=geom.kh > 0,
                kh=geom.kh,
            )
        except ValueError:
            pass

    return SearchResult(
        critical=critical,
        n_surfaces_evaluated=n_evaluated,
        grid_fos=grid_fos,
    )


def search_noncircular(
    geom: SlopeGeometry,
    x_entry_range: Tuple[float, float],
    x_exit_range: Tuple[float, float],
    n_trials: int = 500,
    n_points: int = 5,
    n_slices: int = 30,
    tol: float = 1e-4,
    seed: Optional[int] = None,
) -> SearchResult:
    """Search for the critical noncircular slip surface using random polylines.

    Generates random polyline slip surfaces within the specified entry/exit
    ranges and evaluates Spencer's FOS for each.

    Parameters
    ----------
    geom : SlopeGeometry
        Slope definition.
    x_entry_range : (float, float)
        Allowed entry x-coordinate range (where slip enters ground).
    x_exit_range : (float, float)
        Allowed exit x-coordinate range (where slip exits ground).
    n_trials : int
        Number of random trial surfaces. Default 500.
    n_points : int
        Number of polyline vertices (including endpoints). Default 5.
    n_slices : int
        Number of slices per analysis. Default 30.
    tol : float
        Convergence tolerance. Default 1e-4.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    SearchResult
    """
    if seed is not None:
        random.seed(seed)

    x_lo_entry, x_hi_entry = x_entry_range
    x_lo_exit, x_hi_exit = x_exit_range

    # Get ground surface elevation range for depth generation
    z_min_ground = min(z for _, z in geom.surface_points)
    z_max_ground = max(z for _, z in geom.surface_points)
    min_layer_bot = min(L.bottom_elevation for L in geom.soil_layers)

    best_fos = _FOS_MAX
    best_slip = None
    n_evaluated = 0
    grid_fos = []

    for _ in range(n_trials):
        # Random entry and exit x-coordinates
        x_entry = random.uniform(x_lo_entry, x_hi_entry)
        x_exit = random.uniform(x_lo_exit, x_hi_exit)

        if x_exit <= x_entry + 1.0:
            continue

        # Ground elevation at entry and exit
        z_entry = geom.ground_elevation_at(x_entry)
        z_exit = geom.ground_elevation_at(x_exit)

        # Generate intermediate points
        n_interior = max(n_points - 2, 1)
        points = [(x_entry, z_entry)]

        for j in range(n_interior):
            frac = (j + 1) / (n_interior + 1)
            x_pt = x_entry + frac * (x_exit - x_entry)
            z_ground = geom.ground_elevation_at(x_pt)
            # Random depth below ground surface (but above lowest layer)
            max_depth = z_ground - min_layer_bot
            if max_depth <= 0:
                max_depth = 1.0
            depth = random.uniform(0.5, max_depth * 0.9)
            z_pt = z_ground - depth
            points.append((x_pt, z_pt))

        points.append((x_exit, z_exit))

        try:
            slip = PolylineSlipSurface(points=points)
            fos = _compute_fos(geom, slip, "spencer", n_slices, tol=tol)
            n_evaluated += 1

            grid_fos.append({
                "xc": round(points[len(points) // 2][0], 2),
                "yc": round(points[len(points) // 2][1], 2),
                "R": 0.0,
                "FOS": round(fos, 4),
            })

            if fos < best_fos:
                best_fos = fos
                best_slip = slip
        except (ValueError, ZeroDivisionError, RuntimeError):
            continue

    # Build result
    critical = None
    if best_fos < _FOS_MAX and best_slip is not None:
        try:
            x_entry, x_exit = best_slip.find_entry_exit(geom)
            slices = build_slices(geom, best_slip, n_slices)
            critical = SlopeStabilityResult(
                FOS=best_fos,
                method="Spencer",
                xc=0.0,
                yc=0.0,
                radius=0.0,
                x_entry=x_entry,
                x_exit=x_exit,
                n_slices=len(slices),
                has_seismic=geom.kh > 0,
                kh=geom.kh,
                slip_points=list(best_slip.points),
            )
        except ValueError:
            pass

    return SearchResult(
        critical=critical,
        n_surfaces_evaluated=n_evaluated,
        grid_fos=grid_fos,
    )


# ── PSO Noncircular Search ────────────────────────────────────────────────

def search_pso(
    geom: SlopeGeometry,
    x_entry_range: Tuple[float, float],
    x_exit_range: Tuple[float, float],
    n_particles: int = 30,
    n_iterations: int = 50,
    n_points: int = 5,
    n_slices: int = 30,
    tol: float = 1e-4,
    seed: Optional[int] = None,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
) -> SearchResult:
    """Particle Swarm Optimization for critical noncircular slip surface.

    Each particle represents a polyline slip surface defined by control
    point coordinates. The swarm evolves toward the minimum FOS surface.

    Parameters
    ----------
    geom : SlopeGeometry
        Slope definition.
    x_entry_range : (float, float)
        Allowed entry x-coordinate range.
    x_exit_range : (float, float)
        Allowed exit x-coordinate range.
    n_particles : int
        Swarm size. Default 30.
    n_iterations : int
        Maximum iterations. Default 50.
    n_points : int
        Number of polyline vertices (including endpoints). Default 5.
    n_slices : int
        Number of slices per analysis.
    tol : float
        FOS convergence tolerance.
    seed : int, optional
        Random seed for reproducibility.
    w : float
        Inertia weight. Default 0.7.
    c1 : float
        Cognitive (personal best) coefficient. Default 1.5.
    c2 : float
        Social (global best) coefficient. Default 1.5.

    Returns
    -------
    SearchResult

    References
    ----------
    Kennedy & Eberhart (1995) — Particle Swarm Optimization
    """
    if seed is not None:
        random.seed(seed)

    min_layer_bot = min(L.bottom_elevation for L in geom.soil_layers)
    n_interior = max(n_points - 2, 1)

    # Particle = [x_entry, x_exit, depth_frac_1, ..., depth_frac_n_interior]
    dim = 2 + n_interior

    def _random_particle():
        x_entry = random.uniform(*x_entry_range)
        x_exit = random.uniform(*x_exit_range)
        depths = [random.uniform(0.05, 0.95) for _ in range(n_interior)]
        return [x_entry, x_exit] + depths

    def _particle_to_slip(pos):
        x_entry = max(x_entry_range[0], min(x_entry_range[1], pos[0]))
        x_exit = max(x_exit_range[0], min(x_exit_range[1], pos[1]))
        if x_exit <= x_entry + 1.0:
            return None

        z_entry = geom.ground_elevation_at(x_entry)
        z_exit = geom.ground_elevation_at(x_exit)
        points = [(x_entry, z_entry)]

        for j in range(n_interior):
            frac = (j + 1) / (n_interior + 1)
            x_pt = x_entry + frac * (x_exit - x_entry)
            z_ground = geom.ground_elevation_at(x_pt)
            max_depth = z_ground - min_layer_bot
            if max_depth <= 0:
                max_depth = 1.0
            depth_frac = max(0.05, min(0.95, pos[2 + j]))
            z_pt = z_ground - depth_frac * max_depth
            points.append((x_pt, z_pt))

        points.append((x_exit, z_exit))
        try:
            return PolylineSlipSurface(points=points)
        except (ValueError, IndexError):
            return None

    def _evaluate(pos):
        slip = _particle_to_slip(pos)
        if slip is None:
            return _FOS_MAX
        return _compute_fos(geom, slip, "spencer", n_slices, tol=tol)

    # Initialize swarm
    positions = [_random_particle() for _ in range(n_particles)]
    velocities = [[random.uniform(-0.3, 0.3) for _ in range(dim)]
                  for _ in range(n_particles)]
    fitness = [_evaluate(p) for p in positions]

    pbest_pos = [list(p) for p in positions]
    pbest_fit = list(fitness)

    gbest_idx = min(range(n_particles), key=lambda i: fitness[i])
    gbest_pos = list(positions[gbest_idx])
    gbest_fit = fitness[gbest_idx]

    n_evaluated = n_particles
    grid_fos = []

    for i in range(n_particles):
        slip = _particle_to_slip(positions[i])
        if slip and fitness[i] < _FOS_MAX:
            mid = slip.points[len(slip.points) // 2]
            grid_fos.append({
                "xc": round(mid[0], 2),
                "yc": round(mid[1], 2),
                "R": 0.0,
                "FOS": round(fitness[i], 4),
            })

    for iteration in range(n_iterations):
        w_iter = w - (w - 0.4) * iteration / max(n_iterations - 1, 1)

        for i in range(n_particles):
            for d in range(dim):
                r1 = random.random()
                r2 = random.random()
                velocities[i][d] = (
                    w_iter * velocities[i][d]
                    + c1 * r1 * (pbest_pos[i][d] - positions[i][d])
                    + c2 * r2 * (gbest_pos[d] - positions[i][d])
                )
                if d < 2:
                    velocities[i][d] = max(-5.0, min(5.0, velocities[i][d]))
                else:
                    velocities[i][d] = max(-0.3, min(0.3, velocities[i][d]))
                positions[i][d] += velocities[i][d]

            positions[i][0] = max(x_entry_range[0], min(x_entry_range[1], positions[i][0]))
            positions[i][1] = max(x_exit_range[0], min(x_exit_range[1], positions[i][1]))
            for d in range(2, dim):
                positions[i][d] = max(0.05, min(0.95, positions[i][d]))

            fos = _evaluate(positions[i])
            n_evaluated += 1
            fitness[i] = fos

            slip = _particle_to_slip(positions[i])
            if slip and fos < _FOS_MAX:
                mid = slip.points[len(slip.points) // 2]
                grid_fos.append({
                    "xc": round(mid[0], 2),
                    "yc": round(mid[1], 2),
                    "R": 0.0,
                    "FOS": round(fos, 4),
                })

            if fos < pbest_fit[i]:
                pbest_fit[i] = fos
                pbest_pos[i] = list(positions[i])

            if fos < gbest_fit:
                gbest_fit = fos
                gbest_pos = list(positions[i])

    critical = None
    if gbest_fit < _FOS_MAX:
        best_slip = _particle_to_slip(gbest_pos)
        if best_slip is not None:
            try:
                x_entry, x_exit = best_slip.find_entry_exit(geom)
                slices = build_slices(geom, best_slip, n_slices)
                critical = SlopeStabilityResult(
                    FOS=gbest_fit,
                    method="Spencer",
                    xc=0.0,
                    yc=0.0,
                    radius=0.0,
                    x_entry=x_entry,
                    x_exit=x_exit,
                    n_slices=len(slices),
                    has_seismic=geom.kh > 0,
                    kh=geom.kh,
                    slip_points=list(best_slip.points),
                )
            except ValueError:
                pass

    return SearchResult(
        critical=critical,
        n_surfaces_evaluated=n_evaluated,
        grid_fos=grid_fos,
    )


# ── Weak-Layer Biased Search (SNIFF-inspired) ────────────────────────────

def search_weak_layer_biased(
    geom: SlopeGeometry,
    x_entry_range: Tuple[float, float],
    x_exit_range: Tuple[float, float],
    n_trials: int = 500,
    n_points: int = 5,
    n_slices: int = 30,
    tol: float = 1e-4,
    seed: Optional[int] = None,
) -> SearchResult:
    """Weak-layer biased random search inspired by SSAP's SNIFF algorithm.

    Biases random surface generation toward passing through soil layers
    with the lowest shear strength. Surfaces are weighted to preferentially
    route through weak zones.

    Parameters
    ----------
    geom : SlopeGeometry
        Slope definition.
    x_entry_range : (float, float)
        Allowed entry x-coordinate range.
    x_exit_range : (float, float)
        Allowed exit x-coordinate range.
    n_trials : int
        Number of random trial surfaces. Default 500.
    n_points : int
        Number of polyline vertices. Default 5.
    n_slices : int
        Number of slices per analysis.
    tol : float
        FOS convergence tolerance.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    SearchResult

    References
    ----------
    Borselli (2002) — SNIFF Random Search concept (SSAP2010)
    """
    if seed is not None:
        random.seed(seed)

    min_layer_bot = min(L.bottom_elevation for L in geom.soil_layers)

    # Compute weakness score for each layer: inverse of shear strength
    layer_weakness = []
    for layer in geom.soil_layers:
        c, phi = layer.shear_strength_params
        strength = c + 50.0 * math.tan(math.radians(phi))
        if strength <= 0:
            strength = 0.1
        layer_weakness.append(1.0 / strength)

    total_weakness = sum(layer_weakness)
    if total_weakness > 0:
        layer_weights = [lw / total_weakness for lw in layer_weakness]
    else:
        layer_weights = [1.0 / len(geom.soil_layers)] * len(geom.soil_layers)

    best_fos = _FOS_MAX
    best_slip = None
    n_evaluated = 0
    grid_fos = []

    for _ in range(n_trials):
        x_entry = random.uniform(*x_entry_range)
        x_exit = random.uniform(*x_exit_range)
        if x_exit <= x_entry + 1.0:
            continue

        z_entry = geom.ground_elevation_at(x_entry)
        z_exit = geom.ground_elevation_at(x_exit)

        n_interior = max(n_points - 2, 1)
        points = [(x_entry, z_entry)]

        target_idx = _weighted_choice(layer_weights)
        target_layer = geom.soil_layers[target_idx]

        for j in range(n_interior):
            frac = (j + 1) / (n_interior + 1)
            x_pt = x_entry + frac * (x_exit - x_entry)
            z_ground = geom.ground_elevation_at(x_pt)

            if random.random() < 0.7:
                layer_mid = (target_layer.top_elevation + target_layer.bottom_elevation) / 2.0
                thickness = target_layer.top_elevation - target_layer.bottom_elevation
                noise = random.gauss(0, max(thickness / 4.0, 0.1))
                z_pt = layer_mid + noise
                z_pt = max(min_layer_bot, min(z_ground - 0.3, z_pt))
            else:
                max_depth = z_ground - min_layer_bot
                if max_depth <= 0:
                    max_depth = 1.0
                depth = random.uniform(0.5, max_depth * 0.9)
                z_pt = z_ground - depth

            points.append((x_pt, z_pt))

        points.append((x_exit, z_exit))

        try:
            slip = PolylineSlipSurface(points=points)
            fos = _compute_fos(geom, slip, "spencer", n_slices, tol=tol)
            n_evaluated += 1

            grid_fos.append({
                "xc": round(points[len(points) // 2][0], 2),
                "yc": round(points[len(points) // 2][1], 2),
                "R": 0.0,
                "FOS": round(fos, 4),
            })

            if fos < best_fos:
                best_fos = fos
                best_slip = slip
        except (ValueError, ZeroDivisionError, RuntimeError):
            continue

    critical = None
    if best_fos < _FOS_MAX and best_slip is not None:
        try:
            x_entry, x_exit = best_slip.find_entry_exit(geom)
            slices = build_slices(geom, best_slip, n_slices)
            critical = SlopeStabilityResult(
                FOS=best_fos,
                method="Spencer",
                xc=0.0,
                yc=0.0,
                radius=0.0,
                x_entry=x_entry,
                x_exit=x_exit,
                n_slices=len(slices),
                has_seismic=geom.kh > 0,
                kh=geom.kh,
                slip_points=list(best_slip.points),
            )
        except ValueError:
            pass

    return SearchResult(
        critical=critical,
        n_surfaces_evaluated=n_evaluated,
        grid_fos=grid_fos,
    )


def _weighted_choice(weights):
    """Weighted random selection, returns index."""
    r = random.random()
    cumulative = 0.0
    for i, w in enumerate(weights):
        cumulative += w
        if r <= cumulative:
            return i
    return len(weights) - 1


# ── Entry/Exit Region Search ─────────────────────────────────────────────

def search_entry_exit(
    geom: SlopeGeometry,
    x_entry_range: Tuple[float, float],
    x_exit_range: Tuple[float, float],
    n_entry: int = 10,
    n_exit: int = 10,
    method: str = "bishop",
    n_slices: int = 30,
    tol: float = 1e-4,
) -> SearchResult:
    """Entry/exit region search for circular slip surfaces.

    Instead of searching on a (xc, yc) grid, generates trial arcs
    connecting entry and exit points on the ground surface. For each
    entry/exit pair, finds the optimal circle passing through both points.

    Parameters
    ----------
    geom : SlopeGeometry
        Slope definition.
    x_entry_range : (float, float)
        x-range where slip enters the ground (typically upslope).
    x_exit_range : (float, float)
        x-range where slip exits the ground (typically at toe).
    n_entry : int
        Number of trial entry points. Default 10.
    n_exit : int
        Number of trial exit points. Default 10.
    method : str
        FOS method. Default 'bishop'.
    n_slices : int
        Number of slices per analysis.
    tol : float
        FOS convergence tolerance.

    Returns
    -------
    SearchResult
    """
    best_fos = _FOS_MAX
    best_xc = 0.0
    best_yc = 0.0
    best_r = 0.0
    grid_fos = []
    n_evaluated = 0

    entry_xs = [x_entry_range[0] + (x_entry_range[1] - x_entry_range[0]) * i / max(n_entry - 1, 1)
                for i in range(n_entry)] if n_entry > 1 else [(x_entry_range[0] + x_entry_range[1]) / 2]
    exit_xs = [x_exit_range[0] + (x_exit_range[1] - x_exit_range[0]) * i / max(n_exit - 1, 1)
               for i in range(n_exit)] if n_exit > 1 else [(x_exit_range[0] + x_exit_range[1]) / 2]

    for x_en in entry_xs:
        z_en = geom.ground_elevation_at(x_en)
        for x_ex in exit_xs:
            if x_ex <= x_en + 1.0:
                continue
            z_ex = geom.ground_elevation_at(x_ex)

            # Circle center lies on perpendicular bisector of chord
            mid_x = (x_en + x_ex) / 2.0
            mid_z = (z_en + z_ex) / 2.0
            dx = x_ex - x_en
            dz = z_ex - z_en
            chord_len = math.sqrt(dx**2 + dz**2)
            if chord_len < 0.1:
                continue

            perp_x = -dz / chord_len
            perp_z = dx / chord_len
            half_chord = chord_len / 2.0

            t_min = 0.5
            t_max = 3.0 * chord_len
            n_t = 15

            local_best_fos = _FOS_MAX
            local_best_t = t_min

            for it in range(n_t):
                t = t_min + (t_max - t_min) * it / max(n_t - 1, 1)
                xc = mid_x + t * perp_x
                yc = mid_z + t * perp_z
                r = math.sqrt(half_chord**2 + t**2)

                if yc < mid_z:
                    continue

                slip = CircularSlipSurface(xc, yc, r)
                fos = _compute_fos(geom, slip, method, n_slices, tol=tol)
                n_evaluated += 1

                if fos < local_best_fos:
                    local_best_fos = fos
                    local_best_t = t

            # Golden-section refinement
            margin = (t_max - t_min) / n_t
            a = max(t_min, local_best_t - 2 * margin)
            b = min(t_max, local_best_t + 2 * margin)
            golden = (math.sqrt(5) - 1) / 2

            for _ in range(15):
                if b - a < 0.05:
                    break
                t1 = b - golden * (b - a)
                t2 = a + golden * (b - a)

                xc1 = mid_x + t1 * perp_x
                yc1 = mid_z + t1 * perp_z
                r1 = math.sqrt(half_chord**2 + t1**2)

                xc2 = mid_x + t2 * perp_x
                yc2 = mid_z + t2 * perp_z
                r2 = math.sqrt(half_chord**2 + t2**2)

                f1 = _compute_fos(geom, CircularSlipSurface(xc1, yc1, r1),
                                  method, n_slices, tol=tol)
                f2 = _compute_fos(geom, CircularSlipSurface(xc2, yc2, r2),
                                  method, n_slices, tol=tol)
                n_evaluated += 2

                if f1 < f2:
                    b = t2
                    if f1 < local_best_fos:
                        local_best_fos = f1
                        local_best_t = t1
                else:
                    a = t1
                    if f2 < local_best_fos:
                        local_best_fos = f2
                        local_best_t = t2

            t_final = local_best_t
            xc_f = mid_x + t_final * perp_x
            yc_f = mid_z + t_final * perp_z
            r_f = math.sqrt(half_chord**2 + t_final**2)

            grid_fos.append({
                "xc": round(xc_f, 2),
                "yc": round(yc_f, 2),
                "R": round(r_f, 2),
                "FOS": round(local_best_fos, 4),
            })

            if local_best_fos < best_fos:
                best_fos = local_best_fos
                best_xc = xc_f
                best_yc = yc_f
                best_r = r_f

    critical = None
    if best_fos < _FOS_MAX:
        slip = CircularSlipSurface(best_xc, best_yc, best_r)
        try:
            x_entry, x_exit = slip.find_entry_exit(geom)
            slices = build_slices(geom, slip, n_slices)
            critical = SlopeStabilityResult(
                FOS=best_fos,
                method=method.capitalize(),
                xc=best_xc,
                yc=best_yc,
                radius=best_r,
                x_entry=x_entry,
                x_exit=x_exit,
                n_slices=len(slices),
                has_seismic=geom.kh > 0,
                kh=geom.kh,
            )
        except ValueError:
            pass

    return SearchResult(
        critical=critical,
        n_surfaces_evaluated=n_evaluated,
        grid_fos=grid_fos,
    )
