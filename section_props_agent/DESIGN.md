# section_props_agent — design notes

Wraps `sectionproperties` (MIT, R. van Leeuwen; FE-based cross-section
analysis) for parametric shapes and arbitrary polygons.

## Units — documented exception

Structural cross-sections are universally tabulated in mm, so THIS module's
interface is **mm in, mm-based out** (mm^2, mm^3, mm^4, mm^6) — the same
kind of documented exception as `pavement_design`'s US-customary interface.
The library is unit-agnostic; no conversion is applied.

## Method

Geometry is meshed (triangular FE) and analyzed:
- geometric analysis: area, centroid, centroidal I, elastic moduli
  (extreme-fibre, both signs), radii of gyration, principal axes;
- warping analysis (optional, default on): St. Venant torsion constant J
  and warping constant Γ — FE solution, mesh-dependent;
- plastic analysis: plastic section moduli Sxx/Syy.

Mesh size auto-selects at ~area/400 per element (min 1 mm^2); pass
`mesh_size` explicitly for convergence studies. Closed-form checks
(rectangle) reproduce exactly; J for non-circular shapes converges from
below with mesh refinement (rectangle 100x200 at default mesh: J within
~0.3% of the exact series solution 45.79e6 mm^4).

## Shapes

`rectangle(d,b)`, `circle(d)`, `chs(d,t)`, `rhs(d,b,t[,r_out=2t])`,
`i_section(d,b,t_f,t_w[,r=0])`, plus `analyze_polygon_section(points)`
for arbitrary closed outlines (validated via shapely before meshing).

## Sign/axis conventions

x = horizontal, y = vertical, origin at the library's construction origin
(shape corner for rectangles/RHS; centroid offsets are reported via
cx/cy). Principal angle `phi_deg` measured from the x-axis.

## Edge cases

- Non-positive dimensions and <3-point or self-intersecting polygons are
  rejected with actionable errors.
- `warping=False` skips the slower warping solve; J/Γ then report 0/None.
