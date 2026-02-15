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
- 40 tests
