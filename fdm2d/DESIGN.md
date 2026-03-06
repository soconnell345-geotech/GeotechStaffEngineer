# FDM2D — Explicit Lagrangian Finite Difference Module

## Overview

FLAC-style explicit Lagrangian finite difference solver for 2D plane-strain
geomechanics. Named "fdm2d" (not "flac2d") for copyright reasons.

**Key difference from fem2d**: No global stiffness matrix. All operations are
element-by-element with explicit time-stepping. Converges to static equilibrium
via dynamic relaxation (local damping).

| Aspect | fem2d (implicit FEM) | fdm2d (explicit FDM) |
|--------|---------------------|---------------------|
| Element type | CST triangles, Q4 quads | Quad zones → 4 sub-triangles |
| Global matrix | Sparse K assembled, solved | None — element-by-element |
| Solver | Newton-Raphson | Central-difference time stepping |
| Convergence | Residual norm ratio | Unbalanced force ratio |
| Nonlinearity | Tangent stiffness reform | Direct stress update |

## Sign Conventions

- **Tension-positive** (matches fem2d and all other modules)
- Stress vector: [sigma_xx, sigma_yy, tau_xy]
- Gravity body force: by = -gamma (downward)
- Compression = negative stress
- Settlement = negative y-displacement

## Units

All SI: meters (m), kilopascals (kPa = kN/m²), kilonewtons (kN), degrees.

Forces in kN, stress in kPa, mass density rho = gamma/g (kN·s²/m⁴).

## Core Algorithm

Each timestep of the explicit loop:

```
1. STRAIN RATES:     eps_dot = B · v_element      (CST B-matrix per sub-triangle)
2. MIXED DISCR:      Average volumetric eps_dot across 4 sub-tris per zone
3. CONSTITUTIVE:     sigma_new = sigma_old + D · (eps_dot · dt), then MC return mapping
4. INTERNAL FORCES:  F_int = t · A · B^T · sigma  (averaged across 2 overlays)
5. NET FORCE:        F_net = F_gravity + F_surface - F_int
6. DAMPING:          F_damped = F_net - alpha · sign(v) · |F_net|  (alpha ≈ 0.8)
7. VELOCITY:         v_new = v_old + (dt/m) · F_damped; apply BCs
8. POSITION:         x_new = x + dt · v_new
9. CONVERGENCE:      force_ratio = max(|F_unbal|) / |F_applied| < tol
```

## Mixed Discretization (Marti & Cundall 1982)

Each quad zone split into 2 overlapping triangle pairs to prevent volumetric
locking:

```
Zone: 3---2     Overlay A: tri(0,1,2) + tri(0,2,3)   [diagonal 0-2]
      |   |     Overlay B: tri(0,1,3) + tri(1,2,3)   [diagonal 1-3]
      0---1
```

- Stress tracked per sub-triangle (4 per zone)
- Volumetric strain rate averaged across all 4 sub-tris per zone
- Deviatoric strain rates preserved per sub-tri
- Internal forces averaged from both overlays

## Critical Timestep

```
dt = safety × min(sqrt(A_zone) / v_p)    over all zones
v_p = sqrt((K + 4G/3) / rho)             P-wave speed
rho = gamma / 9.81                       mass density
safety = 0.5
```

## Local Damping

Non-viscous (Cundall) damping for static solutions:

```
F_damped = F_net - alpha · sign(v) · |F_net|
```

Default alpha = 0.8. This removes energy when force and velocity have the
same sign (motion in the direction of net force) and adds energy when they
don't, but the net effect is always dissipative for convergence to static
equilibrium.

## Convergence

```
force_ratio = max(|F_unbalanced|) / |F_applied|
```

Converged when force_ratio < tol (default 1e-5).
Checked every report_interval (default 1000) steps, not every step.

## Mohr-Coulomb Return Mapping

Standalone implementation in materials.py (no fem2d imports).
2D in-plane Mohr circle formulation:

```
f = q + p·sin(phi) - c·cos(phi)
p = (sxx + syy) / 2
q = sqrt(((sxx - syy)/2)^2 + txy^2)
```

Returns only (sigma_new, yielded) — no tangent D_ep needed for explicit solver.

## Validation

### Analytical Checks
- **Gravity column**: sigma_yy(z) = gamma·z (compression at depth)
  K0 = nu/(1-nu), max settlement ≈ gamma·H²/(2M) where M = E(1-nu)/((1+nu)(1-2nu))
- **Patch test**: linear displacement → uniform strain in all sub-triangles
- **Mixed discretization**: nu=0.499 should not lock

### Cross-validation
- Gravity column: fdm2d vs fem2d within 20% on displacement and stress

## References

1. Cundall, P. (1976). "Explicit finite difference method in geomechanics."
   Proc. EF Conf. Numerical Methods in Geomechanics, Blacksburg, VA.
2. Marti, J. & Cundall, P. (1982). "Mixed discretization procedure for
   accurate modelling of plastic collapse." Int. J. Num. Anal. Meth. Geomech.
3. Itasca (2019). FLAC 8.1 Theory and Background. Minneapolis, MN.
4. Wilkins, M.L. (1964). "Calculation of elastic-plastic flow." Methods in
   Computational Physics, 3.
