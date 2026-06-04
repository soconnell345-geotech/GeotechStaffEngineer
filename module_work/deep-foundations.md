# Ledger — deep-foundations

**Owned modules:** axial_pile, drilled_shaft, lateral_pile, pile_group, wave_equation

## Reference map
- Module source + tests: `<module>/` (e.g. `axial_pile/`), each with `DESIGN.md`, `tests/test_*.py`, `results.py`.
- Agent adapter (LLM-facing METHOD_INFO/METHOD_REGISTRY): `funhouse_agent/adapters/<module>.py`.
- Standards: DM7.2 (deep foundations); GEC-10 / FHWA-NHI-18-024 (drilled shafts) — in `geotech-references/`; FHWA driven-pile guidance.
- Test one module: `.venv/Scripts/python.exe -m pytest <module>/ -v`.

## Backlog
- ◐ **axial_pile** (pilot): Phase 0 ergonomics applied to METHOD_INFO — `pile_type`
  allowed_values; clarified that pipe piles use diameter+wall_thickness, concrete piles
  use `width`; `soil_type` must be cohesionless/cohesive. Live-test errors seen:
  guessed `beta_method` (real methods: axial_pile_capacity / capacity_vs_depth /
  make_pile_section), `soil_type="sand"`, concrete_circular missing `width`. These are
  addressed doc-side; **verify by re-running AP-1..AP-4 after triage.**
- ☐ Extend `allowed_values` to drilled_shaft, lateral_pile, pile_group, wave_equation adapters.
- ☐ Per-module real issues — populate from `geotech_test_suite_results.json` triage.

## Progress log
- 2026-06-03 (lead, Phase 0): axial_pile adapter METHOD_INFO updated (allowed_values +
  clarified geometry/soil_type). funhouse_agent tests green.
