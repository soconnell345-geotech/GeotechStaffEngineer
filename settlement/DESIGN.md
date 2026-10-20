# settlement — Foundation Settlement Analysis

## Purpose
Computes immediate (elastic) and consolidation settlement for shallow
foundations, matching CSETT program output. Includes Schmertmann CPT method,
1-D consolidation, and time-rate analysis.

## References
- Schmertmann (1970, 1978) — CPT-based immediate settlement
- Hough (1959) — granular (sand/gravel) C'-index settlement
- Terzaghi 1-D consolidation theory
- FHWA GEC-6 (Shallow Foundations)

## Files
- `immediate.py` — elastic_settlement(), schmertmann_settlement()
- `hough.py` — hough_settlement() granular C'-index (HoughLayer/HoughResult)
- `consolidation.py` — consolidation_settlement() with Cc/Cr/Csec
- `time_rate.py` — time_factor(), degree_of_consolidation()
- `results.py` — SettlementResult with summary()/to_dict()
- `__init__.py` — exports analyze_settlement(), analyze_consolidation()

## Public API
```python
analyze_settlement(q_net, B, L, footing_depth, ...) -> SettlementResult
analyze_consolidation(layers, delta_sigma, ...) -> ConsolidationResult
```

## Key Notes
- Schmertmann uses SchmertmannLayer list with depth_top/bottom/Es/Izp
- Hough (granular): `hough_settlement(layers, q_net, B, L=None)` over HoughLayer
  (thickness/depth_to_center/sigma_v0/C_prime); per-layer
  dH=H/C'·log10[(σ'vo+Δσ)/σ'vo] with a 2:1 Δσ (reuses `approximate_2to1`). C' is
  the Hough bearing-capacity index (from corrected SPT N', GEC-6 Fig 5-19), NOT
  Cc/(1+e0). sigma_v0 is supplied per layer (not computed from a γ-profile).
  Validated vs GEC-6 Ex B-1 Tables B1-2/B1-3 (V-022).
- time_rate module reused by ground_improvement/surcharge.py and wick_drains.py
- Parameter `q0` (not `q_overburden`) for Schmertmann overburden correction

## Defaults made correct or stated (GM-142, 2026-10-18)

- **Footing length.** A strip with no L is a strip (`L = math.inf`): the
  stress is the strip's closed form (2:1 `approximate_2to1_strip`,
  Boussinesq `boussinesq_center_strip` = q/pi (alpha + sin alpha),
  Westergaard `westergaard_center_strip`, each the L -> infinity limit of
  the module's rectangle) and Schmertmann takes the plane-strain diagram.
  The elastic method's Iw grows without limit with L/B (GEC-6 Table 5-12),
  so an elastic strip with no L is refused unless `Iw_immediate` is given.
  A square or circle with no L takes L = B and says so; a rectangle with no
  L is refused. Before: every footing with no L was a B x B square (and
  `SettlementAnalysis` with no L used L = 1.0 m).
- **Secondary compression e0.** Each consolidating layer settles by
  C_alpha / (1 + e0) H log10(t2/t1) with its own e0 (FHWA NHI-06-088 Eq.
  7-10), summed; `e0_secondary` given still applies one e0 to every layer.
  Before: e0 = 1.0 whatever the layers' e0.
- **The consolidation tool** needs `q_net` and `B` when no `delta_sigma` is
  given (it took B = 1.0 m and q_net = 0).
- Each choice is in the result's `choices` with given / default / module.
- **The elastic method's Iw (GM-203, 2026-10-19).** `elastic_settlement`
  took Iw = 1.0 for every footing, which is no printed point of a square.
  Not given, Iw is now GEC-6 Table 5-12's C_d (p. 97, after Winterkorn &
  Fang 1975) for `shape`, `rigidity` and `point`
  (`immediate.gec6_influence_factor`): every printed cell, a rectangle
  between the printed L/B rows interpolated in log(L/B); a strip, a rigid
  rectangle (not printed) and a point the shape lacks are refused. Default:
  the centre of a flexible square, 1.12 (before: 1.0, so +12 %). The tool
  states the value and the table's other points in `choices`. Note the
  table prints a rigid square (0.99) above the flexible square's average
  (0.95), where the rigid circle (0.79) is below the flexible circle's
  (0.85). `SettlementAnalysis` keeps Schleicher's closed form for the
  centre of a flexible footing (1.122 for a square).
- **The rigid square, 0.99 (GM-214, 2026-10-19).** Kept as printed. The
  library was searched for another printed value: FHWA NHI-06-089 Table
  8-13 (p. 8-59) prints the same Winterkorn & Fang table with 0.99; UFC
  3-220-10 (2026) / DM 7.1 give the elastic method through Christian &
  Carrier's mu_0 and mu_1 (Fig. 5-6) with no rigid factor; UFC 3-220-20,
  NHI-06-088 and EN 1997-1 and -2 print none. Bowles-type tables are not
  in the library. So there is no cited alternative to offer; a square's
  `choices` basis carries the note (`immediate.GEC6_RIGID_SQUARE_NOTE`)
  and says to give Iw for another value. No number changes.

## The values the analysis used (GM-143)

`SettlementResult` carries `B`, `L` (None for a strip), `footing_shape`,
`Iw`, `schmertmann_terms` (C1, C2, C3, z_peak, z_max, Iz_peak, the sum and
Se), `consolidation_terms` (per layer delta_sigma, sigma_final, case, Sc)
and `secondary_terms` (t1, t2, where t1 came from, per-layer e0 and Ss).
The calc steps print these; they no longer work out the influence factors
or the stress increase again (`immediate.schmertmann_terms`,
`consolidation.consolidation_layer_terms` are the module's own arithmetic).
