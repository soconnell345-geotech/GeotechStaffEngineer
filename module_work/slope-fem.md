# Ledger — slope-fem

**Owned modules:** slope_stability, fem2d

## Reference map
- Module source + tests: `<module>/` with `DESIGN.md`, `tests/test_*.py`, `results.py`.
  - `slope_stability/IMPROVEMENT_PLAN.md` — existing manual roadmap (HYRCAN/SSAP comparison); read before deep work.
- Agent adapter: `funhouse_agent/adapters/<module>.py`.
- Standards/theory: DM7.1 (slopes); Duncan & Wright (limit equilibrium); SRM/FEM theory in module DESIGN.md.
- Test one module: `.venv/Scripts/python.exe -m pytest <module>/ -v` (slope_stability has 169 + 17 skipped).

## Backlog
- ◐ **slope_stability** (pilot): Phase 0 — `method`/`surface_type` allowed_values;
  `analyze_slope` brief now steers to `search_critical_surface` when no trial circle is
  known; clarified `analysis_mode` (drained/undrained). Live tests: guessed
  `circular_bishop` (nonexistent), and `analyze_slope` failed repeatedly with "circle
  does not intersect the ground surface" from guessed centers/radii. System-prompt nudge
  + adapter steer should route to search_critical_surface; **verify SS-1..SS-3 after triage.**
- ☐ Extend `allowed_values` to fem2d adapter.
- ☐ Consider whether `analyze_slope` should auto-fallback to a search when given an
  invalid circle (design decision — discuss with lead; this is a real-behavior change).
- ☐ Per-module real issues — populate from `geotech_test_suite_results.json` triage (fem2d had 2 errors).

## Progress log
- 2026-06-03 (lead, Phase 0): slope_stability adapter METHOD_INFO updated (allowed_values +
  search_critical_surface steer). funhouse_agent tests green.
