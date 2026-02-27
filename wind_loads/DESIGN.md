# Wind Loads Module — Design Notes

## Reference
ASCE 7-22 *Minimum Design Loads and Associated Criteria for Buildings and Other Structures*
- Chapter 26: Wind Load Parameters
- Chapter 29.3: Wind Loads on Other Structures — Freestanding Walls and Fences

## Scope
Wind loads on **freestanding solid walls** and **fences** (open structures with porosity).
Use case: retaining walls with above-grade portions, boundary/screen walls, standalone fences.

Wind speed V is a **direct user input** (no US-specific map lookups) for worldwide applicability.

## Key Formulas

### Velocity Pressure (Section 26.10, Eq. 26.10-1)
```
qz = 0.613 * Kz * Kzt * Kd * Ke * V²  [Pa, with V in m/s]
```

The 0.613 factor comes from 0.5 * rho_air = 0.5 * 1.225 kg/m³ = 0.6125, rounded to 0.613.

### Velocity Pressure Exposure Coefficient Kz (Table 26.10-1)
```
Kz = 2.01 * (z/zg)^(2/alpha)
```

| Exposure | alpha | zg (m) | zg (ft) |
|----------|-------|--------|---------|
| B | 7.0 | 365.76 | 1200 |
| C | 9.5 | 274.32 | 900 |
| D | 11.5 | 213.36 | 700 |

**Critical**: Heights below 4.6 m (15 ft) are evaluated at z = 4.6 m per Section 26.10.1.
No separate zmin clamp per exposure — the formula is applied directly.

Verification against published Table 26.10-1:

| z (m) | Exp B | Exp C | Exp D |
|--------|-------|-------|-------|
| 4.6 | 0.57 | 0.85 | 1.03 |
| 9.1 | 0.70 | 0.98 | 1.16 |
| 30.5 | 0.99 | 1.26 | 1.43 |

### Topographic Factor Kzt (Section 26.8, Eq. 26.8-1)
```
Kzt = (1 + K1*K2*K3)²
```
- K1: hill shape factor (depends on H/Lh and feature type)
- K2: horizontal attenuation from crest
- K3: vertical attenuation above surface

For flat terrain, Kzt = 1.0. Only needed near 2D ridges, 2D escarpments, or 3D hills.

### Ground Elevation Factor Ke (Table 26.9-1)
```
Ke = e^(-0.0000362 * ze)
```
where ze = ground elevation above sea level in meters.
At sea level, Ke = 1.0. At 1000 m, Ke ≈ 0.964.

### Wind Directionality Factor Kd (Table 26.6-1)
Kd = 0.85 for freestanding walls (Other Structures, Solid Signs).

### Gust-Effect Factor G (Section 26.11)
G = 0.85 for rigid structures. Freestanding walls are always treated as rigid.

### Net Force Coefficient Cf (Figure 29.3-1)
Two cases:
- **Case A**: Wall on ground (clearance_ratio h/s = 0)
- **Case C**: Wall fully elevated (clearance_ratio h/s ≥ 1.0)

| B/s | Case A | Case C |
|-----|--------|--------|
| 1 | 1.30 | 1.80 |
| 2 | 1.40 | 1.85 |
| 5 | 1.55 | 1.90 |
| 10 | 1.70 | 1.95 |
| ≥40 | 1.75 | 2.00 |

For intermediate clearance ratios (0 < h/s < 1.0), linearly interpolate between A and C.

### Porosity Reduction (Figure 29.3-1, Note 4)
For open/porous fences:
```
Cf_effective = Cf_solid * solidity_ratio
```
where solidity_ratio = solid_area / gross_area (epsilon).

### Design Forces
```
p = qh * G * Cf              [Pa] — design wind pressure
f = p * s / 1000              [kN/m] — force per unit length
F = f * B                     [kN] — total horizontal force
M = f * (h + s/2)             [kN*m/m] — overturning moment per unit length about base
```
where h = clearance height, s = wall height.

## Sign Conventions
- All forces are horizontal, acting in the wind direction
- Overturning moment is computed about the base of the wall (ground level)
- Moment arm measured from ground to centroid of pressure on the wall

## Edge Cases
- **Short walls (z < 4.6 m)**: Kz evaluated at z = 4.6 m per code
- **Very long walls (B/s > 40)**: Cf clamped at maximum table values
- **Solid fence (solidity = 1.0)**: Equivalent to solid wall analysis
- **No clearance (h = 0)**: Standard wall-on-ground Case A
- **High elevation**: Ke < 1.0 reduces velocity pressure (thinner air)
