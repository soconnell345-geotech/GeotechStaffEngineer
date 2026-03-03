# slope_stability — Limit Equilibrium Slope Stability

## Purpose
Circular and noncircular slip surface analysis using three methods of slices
(Fellenius, Bishop, Spencer) with critical surface search and entry/exit
range constraints.

## References
- Fellenius (1927) — Ordinary Method of Slices
- Bishop (1955) — Geotechnique, Vol. 5, pp. 7-17
- Spencer (1967) — Geotechnique, Vol. 17, pp. 11-26
- Duncan, Wright & Brandon (2014) — Soil Strength and Slope Stability
- FHWA GEC-7, Lazarte et al. (2003) — Soil nail walls (nails.py, currently disconnected)

## Files
- `geometry.py` — SlopeSoilLayer (elevation-based), SlopeGeometry (polyline surface/GWT)
- `slip_surface.py` — CircularSlipSurface (center + radius), PolylineSlipSurface (polyline points)
- `slices.py` — Slice dataclass, build_slices() with multi-layer weight/pore pressure
- `methods.py` — fellenius_fos(), bishop_fos(), spencer_fos() (circular + noncircular)
- `search.py` — optimize_radius(), grid_search(), search_noncircular(), entry/exit filtering
- `analysis.py` — analyze_slope(), search_critical_surface() orchestrators
- `results.py` — SliceData, SlopeStabilityResult, SearchResult
- `nails.py` — SoilNail, NailContribution (kept but disconnected from analysis pipeline)
- `tests/test_slope_stability.py` — 83 tests (core + entry/exit limits + noncircular search)
- `tests/test_duncan_verification.py` — 19 tests (Duncan §7.6-7.7 verification examples)
- `tests/test_nails.py` — 38 tests + 16 skipped (nail intersection/pullout/capacity)

## Public API
```python
# Circular analysis
analyze_slope(geom, xc, yc, radius, method="bishop", tol=1e-4, ...) -> SlopeStabilityResult

# Noncircular analysis (polyline slip surface)
analyze_slope(geom, slip_surface=PolylineSlipSurface(points), method="spencer", tol=1e-4, ...)

# Critical surface search — circular (grid search)
search_critical_surface(geom, x_range, y_range, nx, ny, method="bishop",
                        surface_type="circular", x_entry_range=None, x_exit_range=None,
                        tol=1e-4, ...) -> SearchResult

# Critical surface search — noncircular (random polyline)
search_critical_surface(geom, surface_type="noncircular",
                        x_entry_range=None, x_exit_range=None,
                        n_trials=500, n_points=5, seed=None,
                        tol=1e-4, ...) -> SearchResult
```

## Slip Surface Types

### Circular (CircularSlipSurface)
- Defined by center (xc, yc) and radius
- Entry/exit found by bisection against ground surface
- Works with all three methods (Fellenius, Bishop, Spencer)
- Spencer reduces to Bishop for circular (theta=0)

### Noncircular (PolylineSlipSurface)
- Defined by polyline points [(x1,z1), (x2,z2), ...]
- x-coordinates must be monotonically increasing
- Entry = first point, exit = last point (or intersection with ground)
- Spencer only (force-marching approach, no moment arm needed)
- `slip_elevation_at(x)` — linear interpolation between points
- `tangent_angle_at(x)` — angle of segment containing x
- Both classes share `is_circular` property for dispatch

## Search Algorithms

### Circular Grid Search
- `grid_search(geom, x_range, y_range, nx, ny, method, ...)` — evaluates nx*ny circle centers
- `optimize_radius(geom, xc, yc, method, ...)` — golden-section radius optimization at each center
- Optional `x_entry_range` / `x_exit_range` — reject surfaces whose entry/exit fall outside bounds
- FOS guard: surfaces with FOS < 0.05 are rejected as numerical artifacts

### Noncircular Random Search
- `search_noncircular(geom, n_trials, n_points, x_entry_range, x_exit_range, ...)` — random polylines
- Generates random polyline surfaces between entry and exit x-ranges
- Intermediate points at random depths below ground surface
- Uses Spencer's method for each trial
- Returns best (minimum FOS) surface found

## Critical Sign Conventions
- **Driving moment**: uses `W*(x_mid - xc)/R` + `abs(driving)`, NOT `W*sin(alpha)`
  directly, because `sin(alpha)` can be negative for L-to-R slopes
- **Seismic moment**: computed separately from gravity driving, then
  `driving = abs(gravity) + abs(seismic)` so seismic always increases
  driving regardless of slope direction (L-to-R vs R-to-L)
- **Spencer m_alpha**: `cos(alpha - theta) + sin(alpha - theta)*tan(phi')/F`
  (reduces to Bishop when theta=0)

## Key Technical Notes
- **Spencer = Bishop for circular surfaces**: sin(alpha) = (x-xc)/R identity
  makes moment and force driving identical, forcing theta=0
- **Spencer noncircular**: uses interslice force marching (no moment arm).
  FOS is the value that makes the last interslice force = 0.
- **For c'=0 (cohesionless)**: circular analysis overestimates FOS vs infinite
  slope solution tan(phi)/tan(beta) — this is a well-known limitation
- **Coordinates**: elevation-based (x, z where z=elevation), NOT depth-based
- **SlopeSoilLayer**: separate from geotech_common/SoilLayer (no adapter yet)
- **Undrained**: phi=0, cu used. Bishop = Fellenius when phi=0 (m_alpha = cos(alpha))
- **FOS convergence guard**: in `_compute_fos` (search.py), FOS < 0.05 is treated
  as non-convergence and returns FOS_MAX (999). Prevents negative/absurd FOS from
  polluting search results.
- **Empty-space slices**: slices where the slip surface is below all soil layers
  are skipped (zero weight, no contribution)
- **Nails disconnected**: `nails.py` is importable but not called from the analysis
  pipeline. Nail fields removed from `SlopeStabilityResult`. Can be re-enabled later.

## Duncan Verification Examples (test_duncan_verification.py)

| Example | Description | Published FOS | Method | Tolerance |
|---------|-------------|--------------|--------|-----------|
| 1 | Saturated clay, undrained (cu=600psf, phi=0) | ~1.0 Fellenius, ~1.08 Bishop | Fellenius, Bishop | ±10% |
| 2 | Cohesionless slope (c=0, phi=40, 2H:1V) | ~1.17 | Bishop, Spencer | ±15% |
| 3 | Cohesionless with seismic (kh=0.15) | < Ex 2 | Spencer | directional |
| 4 | Two-layer (sand over clay) | ~1.4 Bishop, ~1.3 Fellenius | Bishop, Fellenius | ±20% |
| 6 | Submerged slope with GWT | varies | Bishop | ±15% |

All dimensions converted from imperial (ft/psf/pcf) to SI (m/kPa/kN/m³).
Example 5 skipped (curved Mohr-Coulomb envelope not supported).

## Qt GUI
- **Standalone app**: `slope_stability_qt.py` (separate from main `geotech_qt_gui.py`)
- **Panel**: `qt_panels/slope_stability_panel.py`
- Features: circular/noncircular toggle, convergence tolerance, entry/exit limits,
  slice visualization on cross-section, save/load/export
