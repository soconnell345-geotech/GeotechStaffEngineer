# section_props_agent — design notes

Cross-section properties for parametric shapes and arbitrary polygons,
computed natively on numpy/scipy.

## Why there is no third-party section library (5.13.0)

This module wrapped `sectionproperties` (MIT, R. van Leeuwen) until 5.13.0.
That package requires `cytriangle` — a compiled Cython wrapper around
Shewchuk's Triangle mesher — whose only manylinux wheel is **quarantined by
the corporate Nexus malware-defense proxy** the app deploys behind. Because
it was a core dependency, one blocked wheel failed the *entire*
`pip install geotech-staff-engineer` on the Databricks cluster (observed on
5.12.0, 2026-09-08), and the project is too small to justify a security
waiver.

So the wrapper was replaced by native mechanics, the same route taken for the
`groundhog` removal in 5.11.2: the library's outputs were pinned as numerical
oracles **before** deletion (`module_work/structural_native/oracles.json`,
regenerable with `pin_oracles.py`), no library code was copied, and
`tests/test_retired_library_oracle.py` holds the engine to those numbers.

The rewrite also made the module ~17x faster (no meshing, no FE solve) and
dropped the transitive `shapely` dependency.

## Units — documented exception

Structural cross-sections are universally tabulated in mm, so THIS module's
interface is **mm in, mm-based out** (mm^2, mm^3, mm^4, mm^6) — the same
kind of documented exception as `pavement_design`'s US-customary interface.

## Method

**Geometry (`polygon_props.py`) — exact.** Every geometric property is a
closed-form integral over the outline via Green's theorem: area, centroid,
centroidal Ixx/Iyy/Ixy, principal I11/I22 and their angle, elastic moduli at
both extreme fibres, and radii of gyration. Hollow shapes are described as a
solid ring plus a negative ring, so a CHS or RHS is integrated exactly the
same way. Plastic moduli come from bisecting for the area-halving axis and
summing the two halves' first moments, with the halves obtained by clipping
each ring to a half-plane — also exact per iteration, so the only error is
the root-finding tolerance.

Circles and CHS are handled analytically rather than as polygons. Rounded
RHS corners and I-section root fillets are sampled as 48-segment arcs, which
is exact to ~1e-6 of the section area.

**Torsion and warping (`torsion.py`) — published closed forms.**

| shape | J | source |
|---|---|---|
| rectangle | exact St. Venant series | Timoshenko & Goodier, Art. 109 |
| circle / CHS | polar second moment | exact |
| RHS | Bredt single-cell | thin-wall closed section |
| I-section | strips + fillet term | El Darwish & Johnston (1965) / AISC DG9 |
| polygon | FD Prandtl solve | no closed form exists |

The rectangle series reproduces Roark's tabulated coefficients (Table 10.1
case 4) across aspect ratios 1–10, and the retired FE library's value for a
2:1 rectangle to 0.001%.

**The one approximate path** is the polygon torsion solve: a finite-difference
solution of `lap(phi) = -2` with `phi = 0` on the boundary, `J = 2*int(phi)`.
Holding phi = 0 at the first node outside the outline puts the zero contour
about half a cell beyond the true boundary, making the raw solve first-order
in cell size (measured: error halves exactly per grid halving), so the solver
runs twice and extrapolates to zero cell size. Against the exact rectangle
series that lands within **0.013%**. Nodes are offset half a cell to keep
them off axis-aligned edges — without that the inside/outside test turns on
round-off and the error stops falling smoothly, which breaks the
extrapolation.

## Documented differences from the retired library

- **Warping constant.** Reported for the I-section (`Cw = Iy*h0^2/4`, the
  value steel tables quote — within 0.9% of the retired FE solve) and as
  exactly 0 for circular sections, which do not warp. For a solid rectangle,
  a closed box or an arbitrary polygon there is no defensible closed form and
  design practice neglects warping restraint there, so `gamma_mm6` is `None`
  rather than a fabricated number. This is the only capability the rewrite
  did not carry over.
- **J on I-sections** runs 1.6–3.3% above the FE value: the fillet term is
  the AISC/Darwish-Johnston approximation that steel tables are built on.
- **Circles** are analytic, so they differ ~0.2–0.3% from the library, which
  meshed them as 64-gons.
- **Product moment** of a symmetric section is exactly 0 here (the library
  returned round-off), which also keeps `phi_deg` from flipping 180 degrees.

## Shapes

`rectangle(d,b)`, `circle(d)`, `chs(d,t)`, `rhs(d,b,t[,r_out=2t])`,
`i_section(d,b,t_f,t_w[,r=0])`, plus `analyze_polygon_section(points)` for
arbitrary closed outlines (validated natively for self-intersection and
zero area — no shapely).

## Sign/axis conventions

x = horizontal, y = vertical, origin at the shape's construction origin
(bottom-left corner for rectangles/RHS/I-sections, centre for circles);
centroid offsets are reported via cx/cy. `phi_deg` is the major (11) axis
measured from the x-axis, so a section stiffer about y reports -90.

## Edge cases

- Non-positive dimensions, walls thicker than the section, flanges deeper
  than the section, and <3-point or self-intersecting polygons are rejected
  with actionable errors.
- `warping=False` skips the torsion work; J/Γ then report 0/None.
- `mesh_size` is retained for API compatibility. The geometry needs no mesh;
  the value is only used as a target cell area for the polygon torsion solve.
