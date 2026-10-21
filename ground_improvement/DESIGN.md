# ground_improvement — Ground Improvement Methods (GEC-13)

## Purpose
Design tools for four ground improvement techniques: aggregate piers,
wick drains, surcharge preloading, and vibro-compaction.

## References
- FHWA GEC-13 (Ground Modification Methods)
- Barron (1948), Hansbo (1981) — radial consolidation theory
- Priebe's curve as GEC-13 prints it (Fig. 5-21 curve (4), p. 5-36): the
  module's only Priebe basis since GM-276; no Priebe formula

## Files
- `aggregate_piers.py` — area replacement, SRF, composite modulus, bearing improvement
- `settlement_ratio.py` — the settlement ratio n' off GEC-13 Fig. 5-21: by E_c/E_s (weighted modulus, Balaam) and the Priebe curve (4)
- `wick_drains.py` — Barron/Hansbo radial consolidation, drain spacing design (bisection)
- `surcharge.py` — preloading with/without drains (imports settlement.time_rate)
- `vibro.py` — vibro-compaction feasibility (fines content go/no-go)
- `feasibility.py` — decision support: soil + problem -> method recommendations
- `results.py` — 5 result dataclasses

## Public API
```python
analyze_aggregate_piers(B, L, q_applied, ...) -> AggregatePierResult
analyze_wick_drains(cv, ch, drain_spacing, ...) -> WickDrainResult
analyze_surcharge_preloading(target_consolidation, ...) -> SurchargeResult
analyze_vibro_compaction(fines_content, N_spt, ...) -> VibroResult
evaluate_feasibility(soil_type, fines_content, ...) -> FeasibilityResult
```

## Key Notes
- Combined consolidation: U_total = 1 - (1-Uv)*(1-Ur)
- Wick drains use bisection to find spacing for target consolidation
- feasibility.py takes plain floats, NOT SoilProfile objects
- Vibro: FC<10% feasible, 10-20% marginal, >20% not feasible
- `improved_bearing_capacity` is a FIRST-ORDER estimate using the
  equal-strain stress-concentration factor 1 + as·(n−1) (GI-2). It is not
  the Priebe curve, which GEC-13 prints as a settlement ratio (Fig. 5-21).
- `equivalent_drain_diameter = 2(w+t)/π` (Hansbo; GI-1 fixed in 5.0)
- Aggregate piers state their choices (GM-142, 2026-10-18): n (default
  5.0; GEC-13: "between 2.0 and 5.0" for stone columns, Sec. 4.1.2.4; for
  rammed aggregate piers "4 to 45 for rigid footings" and 5 to 10 under
  embankments, p. 5-45; taken as given, not worked out from the moduli), E_soil (used only for the composite modulus) and the pattern. The calc
  package no longer prints E_column as an input. Since GM-179 the tool
  refuses E_column (and its other names column_modulus, E_pier) with the
  stress-ratio settlement method: there it changes no result, only a
  warning below E_soil.
- Settlement by E_c/E_s (GM-220, 2026-10-20): `settlement_method`
  'stress_ratio' (the default, SRF = 1/[1 + a_s(n - 1)] from n, unchanged)
  or one of GEC-13 (FHWA-NHI-16-027) Fig. 5-21's (p. 5-36) settlement
  ratio curves n' (settlement without columns over settlement with them,
  against A/A_c): 'weighted_modulus', curve (2), n' = 1 + a_s(E_c/E_s - 1),
  any E_c/E_s (traced, within 1 % of (2a) up to A/A_c = 5 and 2.2 % beyond,
  within 0.6 % of (2b)); 'balaam_rigid', curves (5b) E_c/E_s = 10 and (5c)
  5, A/A_c 1.75 to 9, linear in E_c/E_s between them; 'balaam_flexible',
  curve (5a) E_c/E_s = 10 only, A/A_c 1.5 to 8.5. The Balaam curves were
  traced off the rendered page, calibrated on the grid lines, checked by
  overlay and tabulated at A/A_c steps of 0.25 to 0.5; between nodes n' is
  linear in a_s (within 0.5 % of the trace). Off the drawn ranges they are
  refused, not extended. p. 5-37: the FHWA (1983) Equilibrium Method "is
  roughly equivalent to the Balaam relationships shown in Figure 5-21 and
  represents an average or lower bound estimate suitable for preliminary
  analyses"; the Priebe curve "may be used to evaluate the upper bound
  effectiveness". These methods need E_column (and the tool, E_soil);
  then SRF = 1/n', S = S_unreinforced/n', E_comp = E_s n', and n sets
  only the improved bearing (a choice only with q_unreinforced). GEC-13
  has no worked example by these curves (Example 2, p. 5-49, reads the
  Priebe curve).
- Priebe (GM-221, GM-276): `priebe_improvement_factor` reads GEC-13's
  printed Priebe curve, Fig. 5-21 curve (4), phi = 37.5 deg
  (`settlement_ratio.PRIEBE_CURVE_4`, A/A_c 1.6 to 8.0, traced and drawn
  back as the Balaam curves, +/- 0.03), and nothing else. GM-221 added the
  curve beside Priebe's basic n0 formula (default phi_column 42.5, printed
  nowhere in GEC-13, 20 % above the curve at Example 2); the owner's answer
  to OWN-70 (2026-10-21): "just the printed curve. discard your formula.
  don't trust formulas from your memory!" GM-276 removed the formula,
  `priebe_basic_improvement_factor` and the `method` choice; the tool
  refuses `phi_column`, `nu_soil` (and their other names) and `method`, each
  with what it uses now, and returns `settlement_ratio` (was `n0`) with
  `phi_column_deg` 37.5. Off A/A_c 1.6 to 8.0 it is refused, not extended.
  GM-276 checked the table on both printed copies: on p. 5-36 it sits on the
  line within 0.015 (95 % of reads) from A/A_c 2.25 to 8 and within 0.04
  above that, where the curve is steep; Fig. 5-27 (p. 5-49) agrees. From
  A/A_c about 4.0 the printed (4) runs on one line with (5a). Example 2: at
  A/A_sc = (5.7/3.0)^2 = 3.61 the curve reads 2.55 on both copies, so 27 in
  becomes 10.6 in; GEC-13 prints 2.7 and 10 in, its arrows (Fig. 5-27)
  drawn at A/A_sc 3.53, n' 2.63 on the curve, and labelled 3.6 and 2.7.
  The settlement and composite-modulus steps cited "Priebe (1995)": they
  are GEC-13's stress reduction method (Sec. 4.1.2.6, p. 5-36) with the
  soil stress by Eq. 5-4 (p. 5-32), on the unit cell under equal strain
  (Secs. 4.1.2.1, 4.1.2.4); GEC-13 names Priebe (1995) only as a design
  method (p. 5-27) and for its curve.
- Feasibility screening (GM-179): the depth to the bottom of the
  treatment, depth_to_top_m + thickness_m, is held against GEC-13's
  printed depth limits: aggregate columns "Typically, the maximum
  practical depth of stone columns and rammed aggregate piers is 100 feet
  and 35 feet respectively" (p. 5-9, item 7; 30.5 m and 10.7 m), which
  replaces an uncited 12 m thickness limit; vibro-compaction "a depth of
  about 165 feet, with generally recommended depth of about 100 feet"
  (p. 4-65; 50 m and 30.5 m). The water table is refused: no screen here
  uses it, and GEC-13 finds vibro-compaction "effective both above and
  below the natural groundwater level" (p. 4-65).
- Tests in tests/test_ground_improvement.py
