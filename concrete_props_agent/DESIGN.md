# concrete_props_agent — design notes

Wraps `concreteproperties` (MIT, R. van Leeuwen) for rectangular RC
section analysis. Requires Python >= 3.12 (library floor); on older
interpreters `has_concreteproperties()` is False and the adapter returns
an actionable error.

## Units — documented exception

Dimensions mm, strengths MPa (the library's native N/mm consistent set);
moments are converted to kN*m and axial forces to kN on output.

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

## Geometry

`concrete_rectangular_section` builder: bottom bars (n, dia) and optional
top bars, `cover_mm` = CLEAR cover to the bar surface on both faces (bar
centre = cover + dia/2). Bars are discretized (n_circle=4 area-preserving
polygons), standard for this library.

## Outputs

Gross transformed area/Ixx, cracked transformed Ixx (theta=0, sagging),
cracking moment, ultimate Mn sagging (theta=0) and hogging (theta=pi,
when top steel exists), optional N-M interaction diagram (list of
(N kN compression-positive, M kN*m) points; the library may add anchor
points beyond `n_interaction_points`).

## Validation anchor

Singly-reinforced 300x550, 3-28mm bars fy=500, f'c=32, clear cover 48:
hand ACI rectangular-block calc a = As*fy/(0.85*f'c*b) = 113.2 mm,
Mn = As*fy*(d - a/2) = 398.7 kN*m; module returns 398.5 kN*m (0.06%).
Tolerance in tests: 2% (the library integrates the actual stress block
and bar positions rather than the lumped hand formula).
