# concrete_props_agent — design notes

Rectangular RC section analysis computed natively from the ACI 318-19
material model. No Python-version floor (the retired library needed 3.12).

## Why there is no third-party section library (5.13.0)

This module wrapped `concreteproperties` (MIT, R. van Leeuwen) until 5.13.0,
and through it `sectionproperties` for the geometry builder. Both require
`cytriangle`, whose only manylinux wheel is **quarantined by the corporate
Nexus malware-defense proxy** the app deploys behind — one blocked wheel
failed the *entire* install on the Databricks cluster (observed on 5.12.0,
2026-09-08), with no security waiver available. Following the `groundhog`
precedent from 5.11.2, the library's outputs were pinned as numerical oracles
before deletion (`module_work/structural_native/oracles.json`), no library
code was copied, and `tests/test_retired_library_oracle.py` holds the native
mechanics to those numbers: **every reported scalar agrees to better than
0.04%.**

## Units — documented exception

Dimensions mm, strengths MPa (a consistent N/mm set); moments are converted
to kN*m and axial forces to kN on output.

## Material models (defaults, all overridable)

- Concrete service: linear-elastic, Ec = 4700*sqrt(f'c) MPa
  (ACI 318-19 Eq. 19.2.2.1.b, normal-weight).
- Concrete ultimate: rectangular stress block, alpha = 0.85,
  gamma = ACI beta1 (0.85 down to 0.65 by 0.05 per 7 MPa above 28 MPa;
  ACI 318-19 Table 22.2.2.4.3), eps_cu = 0.003.
- Modulus of rupture fr = 0.62*sqrt(f'c) MPa (ACI 318-19 Eq. 19.2.3.1)
  — sets the cracking moment.
- Steel: elastic-perfectly-plastic, Es = 200 GPa.

Capacities are **NOMINAL (Mn)** — no phi/strength-reduction factors are
applied; the result dict says so explicitly.

## Method (`rc_native.py`)

- **Transformed gross section**: steel enters as `(n-1)*As` — the bar
  displaces concrete, so only the excess is added — then the parallel-axis
  shift about the transformed centroid. Bar self-inertia is neglected, as in
  the standard hand calculation (~0.01% of a beam's Ixx).
- **Cracked section**: concrete carries no tension; the neutral axis is found
  by bisecting for zero net transformed first moment, with bars on the
  compression side transformed at `(n-1)` and tension bars at `n`.
- **Cracking moment**: `M_cr = fr * I_transformed / y_tension`.
- **Ultimate capacity**: strain compatibility — `eps_cu` at the compression
  fibre, elastic-perfectly-plastic steel, the equivalent rectangular block
  over `beta1*c`, and bars inside the block credited only with their stress
  above the concrete they displace. The neutral-axis depth follows from axial
  equilibrium by bisection. Hogging runs the same solve on the flipped
  section.
- **Interaction diagram**: sampled uniformly in axial force (that is how the
  diagram is read) rather than in neutral-axis depth, which would crowd every
  point into the bending end. The pure-compression, balanced, pure-bending
  and pure-tension control points are always included.

Moments are taken about the **gross concrete centroid** (h/2), so a pure
tension force on eccentric steel correctly carries a moment; compression is
positive.

## Geometry

Bottom bars (n, dia) and optional top bars; `cover_mm` = CLEAR cover to the
bar surface on both faces (bar centre = cover + dia/2). Rectangular sections
only, one layer per face.

## Outputs

Gross transformed area/Ixx, cracked transformed Ixx (sagging), cracking
moment, ultimate Mn sagging and hogging (when top steel exists), and an
optional N-M interaction diagram (list of (N kN compression-positive,
M kN*m) points; control points are added beyond `n_interaction_points`).

## Validation anchors

- Singly-reinforced 300x550, 3-28mm bars fy=500, f'c=32, clear cover 48:
  hand ACI rectangular-block calc a = As*fy/(0.85*f'c*b) = 113.2 mm,
  Mn = As*fy*(d - a/2) = 398.7 kN*m; module returns 398.46 kN*m (0.06%).
  Tests assert 0.2%.
- Squash load `P0 = 0.85*f'c*(Ag - As) + As*fy` and the pure-tension point
  reproduce the retired library to the newton.
- A section with identical top and bottom steel returns identical sagging and
  hogging capacities, and zero moment at the squash point — symmetry
  invariants that catch sign errors the scalar anchors would not.

## Documented difference from the retired library

Above roughly 70% of the squash load the two interaction curves separate (up
to ~6% of the peak moment). The native model holds `eps_cu` at the extreme
compression fibre for every neutral-axis depth — textbook ACI strain
compatibility, hand-checkable — while the retired library appears to pivot
the strain profile once the whole section is in compression. ACI 318 caps Pn
at 0.80*P0 for tied columns, so the curves agree everywhere the diagram is
usable, and the control points match to better than 0.01%.
