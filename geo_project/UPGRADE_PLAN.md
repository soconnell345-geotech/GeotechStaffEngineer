# MODEL-SETUP build plan (geo_project + setup sub-agent)

Branch: `model-setup` (worktree). Base: master f758ddb.

## Motivation / design inversion

LLMs are not trusted to read geometry off an image. The system NEVER claims to
have read geometry correctly — it echoes back a rendered cross-section
(numbers → image, easy for a human to verify) and the human confirms visually.
A specialized "setup" sub-agent builds LE/FEM models in STAGES with explicit
human confirmation gates; vision-derived geometry is quarantined
(`provenance='vision_draft'`) and BLOCKS analysis until confirmed.

## Targets

1. `geo_project/` — canonical, versioned Project document (numpy-only hard dep):
   schema.py, validate.py, builders.py, templates.py, ingest.py, render.py, tests.
2. `funhouse_agent/deep/setup_tools.py` + `setup_agent.py` — staged setup
   sub-agent with `request_confirmation` human gate (langgraph `interrupt()` +
   chat fallback), wired into `build_deep_agent` behind `enable_setup_agent`
   (OFF by default).
3. Docs + demo: geo_project/DESIGN.md, docs/examples/model_setup_walkthrough.md
   (+ echo-back PNG), CLAUDE.md one-liner, pyproject packaging/testpaths.

## Key API facts (from phase-0 study)

- `SlopeSoilLayer`: name, top/bottom_elevation, gamma, gamma_sat, phi, c_prime,
  cu, analysis_mode ('drained'|'undrained'), ru, bottom_boundary_points,
  strength_model ('mohr_coulomb'|'shansep'|'hoek_brown'), shansep_S/m/ocr/su_min,
  hb_sigci/gsi/mi/D. `SlopeGeometry`: surface_points, soil_layers, gwt_points,
  surcharge(+x_range), kh, nails/anchors/geosynthetics, tension_crack_*.
- `fem2d.analyze_slope_srm(surface_points, soil_layers=[{name, bottom_elevation,
  E, nu, c, phi, psi, gamma}], gwt=(M,2) array, layer_polylines=...)`.
- `dxf_import`: discover_layers(path) → DxfDiscoveryResult.to_dict();
  parse_dxf_geometry(path, layer_mapping=LayerMapping(surface, soil_boundaries
  {dxf_layer: soil_name}, water_table, nails), units, flip_y) → DxfParseResult.
- `pdf_import`: extract_vector_geometry(path, role_mapping={hex: role}, scale,
  page) → PdfParseResult; to_dxf_parse_result(pdf) adapts. vision.py = draft-only.
- `reliability.cov_database.cov_guidance(property, soil_type, test, category)`
  → list[CovEntry] with published sources (values in PERCENT).
- `slope_stability.analysis.analyze_slope(geom, xc/yc/radius|slip_surface,
  method, ...)`, `search_critical_surface(geom, method, surface_type, ...)`;
  `probabilistic.fosm_fos / monte_carlo_fos(geom, variables={key: {mean, cov,
  dist}}, xc/yc/radius, ...)` — keys 'phi'|'c_prime'|'cu'|'gamma' optionally
  ':layer_name'.
- deepagents sub-agent spec: `{name, description, system_prompt, tools,
  middleware?}` (see `build_references_subagent`); tools are
  `StructuredTool.from_function` closures (see `make_core_tools`).
- `langgraph.types.interrupt(value)` raises GraphInterrupt (needs checkpointer);
  resume via `Command(resume=...)`; result surfaces under `__interrupt__`.
  Outside a runnable context / without checkpointer it raises non-GraphInterrupt
  errors → chat fallback path. `langgraph.errors.GraphInterrupt` must be
  re-raised, never swallowed.
- venv has NO ipywidgets → notebook form renderer ships as designed stub +
  chat fallback (owner deprioritized form polish).

## Phases & PROGRESS

- [x] Phase 0 — study (geometry/nails/reinforcement, fem2d.analysis, dxf/pdf
      import, plotting, cov_database, deep agent/tools/notebook, interrupt API).
- [x] Phase 1 — geo_project/schema.py (versioned dataclasses, to_json/from_json
      round-trip, unknown-key tolerance, provenance + confirmations) +
      validate.py (Issue list: monotonic surface, layer coverage/overlap, GWT
      sanity incl. ponded info, reinforcement inside section, materials complete
      per requested analyses, unit-sanity heuristics, vision_draft blocking) +
      tests. COMMIT per phase.
- [x] Phase 2 — builders.py (to_slope_geometry, to_fem_kwargs, run_analyses) +
      templates.py (simple_slope, benched_slope, embankment_on_foundation,
      cut_with_berm; LLM-friendly docstrings) + tests vs hand-built reference.
- [x] Phase 3 — ingest.py (discover_dxf, from_dxf, from_pdf_vector, from_points,
      from_vision_draft) + render.py (echo_back PNG + numbered vertex table,
      matplotlib-guarded) + tests (synthetic ezdxf DXF; vision-draft blocking;
      every template renders + validates clean).
- [x] Phase 4 — funhouse_agent/deep/setup_tools.py: ProjectStore +
      make_setup_tools (project_new/show/patch/render/validate, dxf_discover,
      cov_lookup, project_run gate, request_confirmation interrupt+fallback) +
      offline tests (incl. real interrupt round-trip on a tiny StateGraph with
      InMemorySaver; gate ordering; patch resets stage confirmations).
- [ ] Phase 5 — funhouse_agent/deep/setup_agent.py: SETUP_SYSTEM_PROMPT (staged
      protocol + assumption ledger + never-proceed-unconfirmed + dxf_discover-
      and-ask), build_setup_subagent, build_deep_agent(enable_setup_agent=False
      default) wiring + scripted-model offline test walking all stages
      (template → render → confirm → materials via cov_lookup → confirm →
      water/loads confirm → run LE; project_run refuses early, succeeds after).
- [ ] Phase 6 — geo_project/DESIGN.md, demo script → docs/examples/
      model_setup_walkthrough.md + PNG, CLAUDE.md module line, pyproject
      packages/testpaths, notebook-forms stub note.

## Next action

Implement Phase 5 (deep/setup_agent.py + build_deep_agent wiring + staged scripted-model test), run both suites, commit.

## Descoped (designed path noted)

- ipywidgets form renderer: venv lacks ipywidgets; designed mapping documented
  in DESIGN.md (number → BoundedFloatText, choice → Dropdown, bool → Checkbox,
  text → Text; Approve / Request-changes buttons resolving to
  `Command(resume={approved, edits})`). Chat fallback (numbered questions) is
  the shipped path.
- Multiple surcharges per project: schema stores a list; LE builder maps the
  FIRST surcharge only (SlopeGeometry supports one uniform band) and validate
  warns on extras.
- FEM with SHANSEP/Hoek-Brown layers: validate emits a blocking error
  (fem2d MC-only); designed path = pre-linearize to c-phi per layer.

## Build note (session mechanics)

This session's Write tool is pinned to another worktree; files are authored in
a staging dir and copied here via shell. No content differences result.
