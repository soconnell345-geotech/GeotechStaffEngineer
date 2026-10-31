# wave_equation — Smith 1-D Wave Equation Analysis

## Purpose
Simulates pile driving using Smith (1960) 1-D wave equation method.
Produces bearing graphs (capacity vs blow count) and drivability analyses.

## References
- Smith (1960) — original wave equation formulation
- FHWA GEC-12 (FHWA-NHI-16-009) Vol 2, Chapter 12 — the wave equation, its
  inputs and examples; Vol 1 Chapter 8 — driving stress limits
- UFC 3-220-20 (2025), Sec. 6-9.2 — quake, Smith and Smith viscous damping

**Source basis.** Checked against GEC-12 and UFC 3-220-20 as printed on
2026-10-31 (GM-294; module_work/gm294/CHECK.md has every item and page).
The Smith spring-dashpot, `R = R_static·(1 + J·v)`, is GEC-12 Vol 2
p. 156 and UFC p. 487. The default quakes and damping factors are GEC-12
Vol 2 Sec. 12.6.7, p. 211: shaft quake 0.10 in (2.5 mm); shaft damping
0.05 s/ft non-cohesive (0.16 s/m), 0.10 (0.33), 0.15 (0.49) and 0.20 s/ft
cohesive (0.66 s/m); toe damping about 0.15 s/ft (0.49 s/m). The default toe
quake, 0.10 in, is the non-displacement pile's (p. 171); a displacement pile
takes D/120 to D/60 (p. 211). GEC-12 (2016) Table 12-3 is an example's
output table (p. 190), not these values.

## Files
- `hammer.py` — 14 built-in hammers (diesel, air/steam), GEC-12 Appendix C
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
- The model simplifies GEC-12 Vol 2 Fig. 12-2 (p. 157): one cushion spring
  and the helmet's weight lumped with the first pile segment, where GEC-12
  has a hammer cushion, a helmet mass and a pile cushion. The pile-top
  force is the cushion force less the helmet's inertia (GM-294 review; it
  was the cushion force, and Example 6's Vulcan 014 read 52.5 ksi against
  Fig. 12-15's about 29; now 28.8).
- `hammer_type` is a label (result, calc package): every hammer is a rigid
  ram striking at sqrt(2 g h efficiency) (GEC-12 Vol 2 Eq. 15-1, p. 358,
  with the stroke times the efficiency, p. 205); diesel combustion is not
  modelled, so a diesel strikes with its rated energy at every resistance
  and its result warns (`diesel_warning`): p. 206 calls an optimistic
  stroke non-conservative, and Example 6's ICE 42-S gives 50 and 103
  blows/ft here against 99 and 209 (p. 182). The diesel cycle is GM-29.
  Efficiencies: 0.67 air, 0.80 diesel (p. 185). Ram weights,
  strokes and rated energies are GEC-12 Vol 2 Appendix C's (pp. 519-536),
  since GM-294: the Vulcan 08 and 010 strokes were 0.914 m for 3.25 ft, the
  diesels' strokes and energies were lower, and the ICE I-30V2 and I-46V2
  were "hydraulic" at 1.22 m and 0.95 (they are open end diesels, p. 206).
  `cushion_cor` applies to the default 500,000 kN/m cushion too (GM-160);
  neither default is printed.
- Time step: the Courant step (segment / wave speed, or the cushion's
  sqrt(m/k)) divided by 1.6, Smith's numerical safety factor as GRLWEAP
  normally takes it (GEC-12 Vol 2 p. 213). It was 0.8 x before GM-294.
- Static soil spring is a true elasto-plastic (kinematic) spring with memory:
  loads at slope R_ult/Q to R_ult, unloads along the same elastic slope from
  the peak, leaving a plastic offset. The toe spring is no-tension (gaps on
  rebound) and carries no damping while the gap is open. Springs are
  stateful per blow; `simulate_blow` creates fresh springs each blow.
- Permanent set per blow, GEC-12 Vol 2 p. 158: the maximum pile toe
  displacement less a weighted average of the shaft and toe quakes. The page
  gives no weights; the shares of R_ult on the shaft and at the toe are the
  module's assumption (`SoilSetup.weighted_quake`), stated in the results'
  choices with the blow counts of a plain average and of the toe quake
  alone (`quake_weighting_choice`; Example 6 at 0.40 in: 92.6, 63 and 303
  blows/ft, GRLWEAP 94, p. 182). Before GM-294 the set was the toe
  spring's plastic offset, max(D_max,toe − Q_toe, 0): the same with equal
  quakes, but zero (refusal) for a pile with no toe resistance.
- Smith damping (`damping_model` on SoilSetup / generate_bearing_graph /
  drivability_study), UFC 3-220-20 pp. 487-488:
  - `"smith"` (default): R = R_static + J·|R_static|·v — damping proportional
    to the *mobilized* static resistance (GEC-12 Vol 2 p. 242: "in most
    instances"). Acts throughout the blow; vanishes when no static
    resistance is mobilized.
  - `"smith_viscous"`: R = R_static + J·R_ultimate·v throughout the blow —
    proportional to the *fully mobilized* resistance (GEC-12 p. 242: for a
    residual stress analysis or a vibratory hammer). Before GM-294 it
    damped only while velocity and displacement had the same sign.
- Drivability refusal: more than 120 blows/ft (393.7 blows/m), GEC-12 Vol 2
  p. 161 and Vol 3 p. 43; 240 blows/ft (787.4) for piles seated on hard
  rock (p. 161). It was 3000 blows/m (914 blows/ft) before GM-294. The pile
  analysed at each depth is as long as the depth, the skin spread evenly.
- Steel: E 200e6 kPa (29,000 ksi, GEC-12 Vol 1 p. 317); unit weight 77.3
  kN/m3 (492 lb/ft3, Vol 2 p. 462; 78.5 before GM-294).
- Driving stress: the calc package gives the yield stress the maximum
  compression needs, F_y ≥ σ / 0.9 (GEC-12 Vol 1 Eq. 8-33, φ_da = 1.0);
  concrete Eqs. 8-62 and 8-63 are named, not run (no f'c input).
