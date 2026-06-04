# Ledger — foundations

**Owned modules:** bearing_capacity, settlement, retaining_walls, ground_improvement, downdrag, wind_loads

## Reference map
- Module source + tests: `<module>/` with `DESIGN.md`, `tests/test_*.py`, `results.py`.
- Agent adapter: `funhouse_agent/adapters/<module>.py`.
- Standards: DM7.1/7.2 (shallow foundations, settlement); GEC-11 (retaining + MSE walls);
  GEC-13 (ground improvement); UFC 3-220-20 (downdrag); ASCE 7-22 Ch.29 (wind_loads) — refs in `geotech-references/`.
- Test one module: `.venv/Scripts/python.exe -m pytest <module>/ -v`.

## Backlog
- ◐ **bearing_capacity** (pilot): Phase 0 ergonomics applied — `shape`, `factor_method`,
  `method` allowed_values. Live tests: BC-2/3/4 (undrained, eccentric, layered) hit
  errors; **review specifics from triage** (likely param naming for 2-layer / eccentricity).
- ☐ Extend `allowed_values` to settlement, retaining_walls, ground_improvement, downdrag, wind_loads adapters.
- ☐ Per-module real issues — populate from `geotech_test_suite_results.json` triage
  (settlement had 2 errors; retaining_walls, ground_improvement, downdrag 1+ each).

## Progress log
- 2026-06-03 (lead, Phase 0): bearing_capacity adapter METHOD_INFO updated (allowed_values
  on shape/factor_method/method). funhouse_agent tests green.
