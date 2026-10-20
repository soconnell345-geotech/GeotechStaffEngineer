# wave_equation — Smith 1-D Wave Equation Analysis

## Purpose
Simulates pile driving using Smith (1960) 1-D wave equation method.
Produces bearing graphs (capacity vs blow count) and drivability analyses.

## References
- Smith (1960) — original wave equation formulation
- Goble & Rausche (1976) — WEAP methodology
- FHWA GEC-12 (Driven Piles) — wave equation chapter

**Source basis (default quake & damping constants, `soil_model.py`).** The
governing algebra (Smith spring-dashpot, `R = R_static·(1 + J·v)`) is coded from
the equation (text). The DEFAULT numeric quake values (~2.5 mm shaft,
2.5–5 mm toe) and Smith damping factors J are transcribed standard values
(Smith 1960 / GRLWEAP defaults). Checked 2026-10-20 (GM-178) against FHWA
GEC-12 (FHWA-NHI-16-009) Vol II Sec. 12.6.7, p. 211: shaft damping 0.05 s/ft
non-cohesive (0.16 s/m), 0.20 s/ft cohesive (0.66 s/m), toe about 0.15 s/ft
(0.49 s/m), shaft quake 0.10 in (2.5 mm). GEC-12 (2016) Table 12-3 is an
example's output table (p. 190), not these values; the code no longer cites
it. They are user-overridable inputs, anchored by the drivability and
bearing-graph behavior checks.

## Files
- `hammer.py` — 14 built-in hammers (diesel, hydraulic, air/steam)
- `cushion.py` — hammer cushion and pile cushion stiffness
- `pile_model.py` — mass-spring discretization of pile
- `soil_model.py` — Smith quake and damping parameters
- `time_integration.py` — explicit central difference time stepping
- `bearing_graph.py` — capacity vs blow count analysis
- `drivability.py` — blow count vs depth profile
- `results.py` — WaveEqResult, BearingGraphResult, DrivabilityResult

## Public API
```python
analyze_bearing_graph(hammer, pile, soil, capacities, ...) -> BearingGraphResult
analyze_drivability(hammer, pile, soil_profile, ...) -> DrivabilityResult
```

## Key Notes
- `hammer_type` is a label (result, calc package): every hammer is a rigid
  ram striking at sqrt(2 g h efficiency); diesel combustion is not modelled,
  so a type's losses go in `efficiency` (GEC-12 Sec. 12.6.2, p. 205).
  `cushion_cor` applies to the default 500,000 kN/m cushion too (GM-160).
- Time step automatically chosen for Courant stability
- Static soil spring is a true elasto-plastic (kinematic) spring with memory:
  loads at slope R_ult/Q to R_ult, unloads along the same elastic slope from
  the peak, leaving a plastic offset. The toe spring is no-tension (gaps on
  rebound). Springs are stateful per blow; `simulate_blow` creates fresh
  springs each blow.
- Permanent set per blow = the toe spring's plastic (residual) displacement —
  the physical quantity. For monotonic toe loading this equals
  max(D_max,toe − Q_toe, 0), so it coincides exactly with the v5.0 (WE-1)
  peak-minus-quake value; it stays correct under re-loading oscillations.
- Smith damping (`damping_model` on SoilSetup / generate_bearing_graph /
  drivability_study):
  - `"smith"` (default): R = R_static + J·|R_static|·v — damping proportional
    to the *mobilized* static resistance (Smith 1960, R_s·(1+J·v); GRLWEAP
    default "Smith damping"). Acts throughout the blow; vanishes when no
    static resistance is mobilized.
  - `"smith_viscous"`: R = R_static + J·R_ultimate·v, loading only — damping
    proportional to the *ultimate* resistance (GRLWEAP "Smith viscous"
    variant; this module's pre-v5.1 behavior).
  Switching the default from ∝R_ultimate to ∝R_static (v5.1, WE-3) reduces
  early-blow damping, giving slightly larger sets / lower blow counts at a
  given R_ult (a few percent on the standard test case).
- Quake: elastic limit of soil spring (typically 2.5mm shaft, 2.5-5mm toe)
