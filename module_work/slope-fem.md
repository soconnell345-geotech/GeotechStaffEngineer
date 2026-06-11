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
- ☑ Extend `allowed_values` to fem2d adapter (fem-modern branch: fem2d_slope_srm element_type/srm_field/n_gp + new outputs).
- ☐ Consider whether `analyze_slope` should auto-fallback to a search when given an
  invalid circle (design decision — discuss with lead; this is a real-behavior change).
- ☐ Per-module real issues — populate from `geotech_test_suite_results.json` triage (fem2d had 2 errors).

## Progress log
- 2026-06-11 (fem-modern branch): fem2d modernization phases 3-7 — SRM
  robustness (GL99 failure detection, shared factorization, srf curve),
  NR hardening tests, published-benchmark validation suite
  (fem2d/VALIDATION.md: GL99 Ex1 FOS 1.34-1.37 vs 1.4 published; Prandtl
  Nc 5.10-5.25 vs 5.14 with T6, CST locks >9; Bishop cross-check +21%
  converging). T6 now the default element for slope SRM. Found+fixed:
  x_extend default (2x width) starved slope-face mesh and overpredicted
  FOS by up to +91%; CST documented as collapse-unsafe. Adapter exposes
  element_type/srm_field/blowup_factor/srf_range/n_gp with
  allowed_values, returns fos_basis + srf_curve.
- 2026-06-03 (lead, Phase 0): slope_stability adapter METHOD_INFO updated (allowed_values +
  search_critical_surface steer). funhouse_agent tests green.
- 2026-06-11 (le-modern close-out): slope_stability LE modernization P0-P10 complete — rigorous GLE/M-P engine, Janbu+f0, DE/entry-exit searches, reinforcement, FOSM+MC, SHANSEP/Hoek-Brown, ponded water, per-slice force table + thrust line, modernized funhouse adapter, VALIDATION.md (F&K/ACADS/Duncan + fem2d SRM cross-check). Suites: slope_stability 348p/17s (incl. slow SRM cross-check), funhouse 668p/5s.
