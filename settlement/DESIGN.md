# settlement — Foundation Settlement Analysis

## Purpose
Computes immediate (elastic) and consolidation settlement for shallow
foundations, matching CSETT program output. Includes Schmertmann CPT method,
1-D consolidation, and time-rate analysis.

## References
Every equation, chart and default is read from these as printed
(`module_work/gm285/CHECK.md`, GM-285):
- FHWA GEC-6 (FHWA-SA-02-054), Ch. 5 and Appendix B: 2:1 stress (Eq.
  5-17), consolidation (Eqs. 5-20, 5-22), Hough (Eq. 5-24, Figs. 5-18,
  5-19), elastic (Eq. 5-36, Table 5-12)
- FHWA NHI-06-089 Sec. 8.5: Schmertmann (Eqs. 8-16 to 8-18, Fig. 8-21,
  Table 8-11, Examples 8-2, 8-3)
- FHWA NHI-06-088 Ch. 7: time rate (Eq. 7-8, Table 7-4), secondary
  compression (Eq. 7-10)
- UFS 3-220-10 (2026) Chs. 4 and 5: Boussinesq (Table 4-2), Westergaard
  (Fig. 4-10), consolidation (Eqs. 5-12 to 5-14), time rate (Fig. 5-16),
  secondary (Eq. 5-18)
- UFC 3-220-20 sends settlement to UFS 3-220-10 Ch. 5

## Files
- `immediate.py` — elastic_settlement(), schmertmann_settlement()
- `hough.py` — hough_settlement() granular C'-index (HoughLayer/HoughResult)
- `consolidation.py` — consolidation_settlement() with Cc/Cr/Csec
- `time_rate.py` — time_factor(), degree_of_consolidation()
- `results.py` — SettlementResult with summary()/to_dict()
- `__init__.py` — exports analyze_settlement(), analyze_consolidation()

## Public API
```python
# exports of settlement/__init__.py (see __all__ for the rest)
SettlementAnalysis(q_applied, q_overburden, B, L=None, ...)  # .compute() -> SettlementResult
SettlementResult
stress_at_depth, approximate_2to1, boussinesq_center_rectangular
elastic_settlement, schmertmann_settlement, SchmertmannLayer
ConsolidationLayer, consolidation_settlement_layer, total_consolidation_settlement
HoughLayer, HoughResult, hough_settlement, hough_settlement_layer
time_factor, degree_of_consolidation, time_for_consolidation, settlement_at_time
secondary_settlement
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
  (0.85). `SettlementAnalysis` without `Iw_immediate` takes the table's
  centre of a flexible footing (GM-285: it kept Schleicher's closed form,
  1.122 for a square, which no reference in the library prints).
- **The rigid square, 0.99 (GM-214, 2026-10-19).** Kept as printed. The
  library was searched for another printed value: FHWA NHI-06-089 Table
  8-13 (p. 8-59) prints the same Winterkorn & Fang table with 0.99; UFC
  3-220-10 (2026) / DM 7.1 give the elastic method through Christian &
  Carrier's mu_0 and mu_1 (Fig. 5-6) with no rigid factor; UFC 3-220-20,
  NHI-06-088 and EN 1997-1 and -2 print none. Bowles-type tables are not
  in the library. So there is no cited alternative to offer; a square's
  `choices` basis carries the note (`immediate.GEC6_RIGID_SQUARE_NOTE`)
  and says to give Iw for another value. No number changes.

## Checked against the print (GM-285, 2026-10-31)

Per item page, printed form, module form and verdict in
`module_work/gm285/CHECK.md`. What changed:
- **Westergaard.** The formula used, I = atan(m n / sqrt(1 + m^2 + n^2))
  / 2 pi, was not a Westergaard form: it is the first (arctangent) term of
  UFS Table 4-2's Boussinesq corner formula alone, up to about half below
  UFS Fig. 4-10 (I = 0.083 where the figure reads 0.116 at m = n = 1, 0.124
  against 0.151 at m = 1, n = 10). The influence factor is now read off Fig.
  4-10 (`stress_distribution.westergaard_influence_factor`, the chart's
  curves digitised, bilinear in m and n); stresses rise 11 % to 81 % in the
  recorded runs, toward double deep below a small area.
- **Boussinesq** below a corner is written as UFS Table 4-2 prints it (the
  same numbers).
- **Schmertmann.** `gamma_soil` is required (sigma'_vp at the peak was
  taken as q0 without it); the diagram for 1 < L/B < 10 is interpolated
  linearly between the axisymmetric and plane-strain ones (NHI-06-089
  Example 8-2, Table 8-11; it was one or the other); no cap on Izp (it
  was held to 1.0; none is printed).
- **Elastic Iw** in `SettlementAnalysis`: GEC-6 Table 5-12 (above).
- **Consolidation.** A layer within 5 % of sigma'_p is no longer taken as
  normally consolidated (the recompression part runs into the NC form as
  sigma'_p falls to sigma'_v0); sigma'_p below sigma'_v0 reads as
  underconsolidated (UFS Sec. 5-2.3) and the analysis warns (the
  arithmetic is as before).
- **Time rate.** U(T) switches form at T = 0.2 as UFS Fig. 5-16 prints
  (it switched at U = 60 %, T = 0.283, where the two forms differ by
  0.0035 in U; at T = 0.2 by 0.0005).
- **Secondary.** t1 = t90 from cv (NHI-06-088 Eq. 7-10; it was t95);
  `secondary_settlement` has no default e0 (it was 1.0). t2 = t1 +
  `t_secondary`: the print's t is counted from loading.

## Layers that do not cover the diagram, and stress lists of another length (GM-320, 2026-11-13)

- **Schmertmann.** Eq. 8-16 sums over "the soil layers within the zone of
  strain influence" (NHI-06-089 p. 8-45); Example 8-2 takes its layers to
  the bottom of the diagram (p. 8-50). App. A.7 (pp. A.7-10 to A.7-12)
  prints the case of layers that stop above it: a 7 ft pier footing
  (L/B 14.3, plane strain) on 11 ft of sand over clay, the diagram to 28 ft unchanged, the sum stopped at the
  bottom of the sand ("The immediate settlement will occur only in the
  sand layer"), the clay settled by consolidation (Step 4, p. A.7-13).
  UFS 3-220-10 Sec. 5-4.1.2 (p. 231) gives the method for coarse-grained
  soils. Until GM-320 any depth of the diagram (2B at L/B = 1, 4B at
  L/B >= 10) the layers did not cover added nothing and nothing was said.
  Now the layers must cover the diagram from the footing base down: a
  first layer that starts below the base, a gap between layers, or the
  depth below the deepest layer is refused, naming the depth ranges,
  unless `below_profile` (`schmertmann_below_profile` on
  `SettlementAnalysis` and its tools) says what lies there:
  - `"settled_by_another_method"`: not granular soil, a clay say, whose
    settlement is worked out by another method (App. A.7's case);
  - `"incompressible"`: rock or other ground that does not compress;
  - `"last_layer_continued"`: the deepest layer's Es to the bottom of the
    diagram (only when the depth below the deepest layer is all that is
    not covered).
  The first two count nothing in the depth not covered, on the same
  diagram, as App. A.7 does; they differ in what the result says. The
  terms carry `profile_bottom`, `uncovered` (the ranges) and
  `below_profile`; the result's `choices` and `warnings` say which was
  taken and where, the tool returns `depths_not_covered_m`, and the
  settlement package's sublayer table shows the continued layer and the
  same note. In `SettlementAnalysis`, where a consolidation layer lies in a
  depth not covered, only `"settled_by_another_method"` is taken (and the
  refusal offers only it): the same result cannot call that clay
  incompressible, or count it with the sand's Es, while consolidating it.
  That refusal does not offer layers down to the bottom of the diagram
  (Schmertmann layers over the clay would count its depth in both sums);
  it names any depth in neither the Schmertmann nor the consolidation
  layers, and asks for Schmertmann layers there if that ground is
  granular. The terms carry `settled` (the depth not covered that the
  consolidation layers settle) and `neither` (with
  `"settled_by_another_method"` in `SettlementAnalysis`: the depth in
  neither list, with or without consolidation layers). The choice then lists only what the call
  takes, and the choice and the warning name the `neither` depth as not in
  the total settlement. Every message names the caller's own parameter (`below_param`). The
  diagram itself is not redrawn for a shallow rigid layer: no held page
  prints that.
- **Tolerance.** Depths within `COVER_TOL_M` = 10 mm count as covered: a
  numerical tolerance, so that the depth the refusal prints (rounded up to
  the centimetre) passes, and a bottom given to the centimetre under an
  interpolated diagram (5.33 m for 5.3333 m) is not refused for the tip of
  the triangle, of the order of 1e-5 of the strain.
- **`consolidation_settlement` (the tool).** `delta_sigmas` must be a list
  of numbers, one per layer; another length is refused (it was zipped,
  dropping the extra layers or stresses from the total).

## The values the analysis used (GM-143)

`SettlementResult` carries `B`, `L` (None for a strip), `footing_shape`,
`Iw`, `schmertmann_terms` (C1, C2, C3, z_peak, z_max, Iz_peak, the sum and
Se), `consolidation_terms` (per layer delta_sigma, sigma_final, case, Sc)
and `secondary_terms` (t1, t2, where t1 came from, per-layer e0 and Ss).
The calc steps print these; they no longer work out the influence factors
or the stress increase again (`immediate.schmertmann_terms`,
`consolidation.consolidation_layer_terms` are the module's own arithmetic).
