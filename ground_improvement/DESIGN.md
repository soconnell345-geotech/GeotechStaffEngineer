# ground_improvement — Ground Improvement Methods (GEC-13)

## Purpose
Design tools for four ground improvement techniques: aggregate piers,
wick drains, surcharge preloading, and vibro-compaction.

## References
- FHWA GEC-13 (Ground Modification Methods)
- Barron (1948), Hansbo (1981) — radial consolidation theory
- Priebe (1995) — stone column improvement factors

## Files
- `aggregate_piers.py` — area replacement, SRF, composite modulus, bearing improvement
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
  equal-strain stress-concentration factor 1 + as·(n−1) — not Priebe's n₀
  (GI-2). The genuine Priebe (1995) basic improvement factor
  n₀(as, φ_col, ν_s) is available as
  `aggregate_piers.priebe_basic_improvement_factor`
- `equivalent_drain_diameter = 2(w+t)/π` (Hansbo; GI-1 fixed in 5.0)
- Aggregate piers state their choices (GM-142, 2026-10-18): n (default
  5.0; GEC-13: "between 2.0 and 5.0" for stone columns, Sec. 4.1.2.4; for
  rammed aggregate piers "4 to 45 for rigid footings" and 5 to 10 under
  embankments, p. 5-45; taken as given, not worked out from the moduli), E_soil (used only for the composite modulus) and the pattern. The calc
  package no longer prints E_column as an input. Since GM-179 the tool
  refuses E_column (and its other names column_modulus, E_pier): it
  changed no result, only a warning below E_soil; the library function
  keeps it for that warning, out of the choices.
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
