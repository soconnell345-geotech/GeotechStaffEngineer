# Calc-Package Visualization Plan (branch: calc-viz)

Professional calc-package output (plots + tables) for the modernized
slope_stability (LE) and fem2d (FEM) modules — target quality: "what
PLAXIS or SLOPE/W put out". PRIORITY 1 = calc-package integration;
STRETCH = interactive single-file HTML viewer. A modeling GUI is OUT
of scope.

## Ground rules
- All matplotlib/plotly use is optional-dep-guarded (import inside
  functions; calc_steps get_figures swallows ImportError) — packages
  must degrade gracefully to text+tables.
- CS-3: calc_steps RENDER result fields; they never re-derive
  engineering values. Anything new to display (trial surfaces,
  plastic Gauss points) is computed in the ENGINE and stored on the
  result additively.
- Tests: `.venv python -m pytest calc_package slope_stability fem2d
  -q -m "not slow"` stays green every phase. Commit per phase.

## PROGRESS

- [x] Phase 0 — study calc_package data model/renderer, slope results
      (slice_data E/X, thrust_line, grid_fos), fem2d results
      (FEMResult arrays, srf_history/srf_curve, SeepageResult,
      StagedConstructionResult), house style (sheet_pile calc_steps).
- [ ] Phase 1 — slope_stability/plotting.py (matplotlib layer):
      cross_section (layers/hatch/GWT/crack/surface/slices),
      trial_surface_map (SearchResult.trial_surfaces — additive
      storage of polyline trials; circular trials reconstructed from
      grid_fos xc/yc/R), slice_force_diagram (N'/S/W bars),
      interslice_distribution (E/X + thrust line), reinforcement
      layout overlay, mc_histogram (+lognormal fit, beta/pf box),
      fosm_tornado (variance contribution). Unit tests.
- [ ] Phase 2 — slope_stability/calc_steps.py upgrade: method+
      interslice statement (GLE/Spencer/M-P lambda/theta, Janbu f0),
      per-slice force TableData (W, alpha, N', S, u*l, E/X),
      reinforcement table, method-comparison table, probabilistic
      section (inputs w/ COV+dist, beta both conventions, pf,
      histogram + tornado figures), search summary section, figures
      from Phase 1 embedded. analysis dict gains optional keys:
      search (SearchResult), mc (MonteCarloResult), fosm (FOSMResult),
      variables (probabilistic spec), FOS_required. Tests.
- [ ] Phase 3 — fem2d/plotting.py + fem2d/calc_steps.py (NEW) +
      registry entry in calc_package/__init__. Figures: mesh plot
      (material coloring, BC symbols), deformed mesh (auto scale
      factor annotated), tricontourf contours (|u|, ux, uy, sigma_yy,
      tau_max) on corner-node triangulation (T6 -> corners), plastic
      GP map (engine additively stores plastic_gp coords at failure
      state in srm.py result + FEMResult.plastic_points), SRF vs
      dimensionless displacement curve w/ FOS annotation, failure
      mechanism |u| contour, seepage head contours + flow vectors.
      calc_steps sections: model summary (mesh stats, materials
      table, BCs), analysis narrative, results tables (max disp,
      FOS + basis + srf_history table, beam forces), staged = one
      section per phase. Tests.
- [ ] Phase 4 — end-to-end render tests: LE (reinforced + MC),
      LE search, FEM gravity/footing, FEM slope SRM, seepage, staged
      -> render_html with base64 figure asserts + table asserts;
      no-matplotlib degradation test (monkeypatch import) for both
      modules.
- [ ] Phase 5 (STRETCH) — calc_package/interactive.py: plotly-based
      single-file HTML viewer, save_interactive_report(result, path,
      ...). LE view: section + toggleable trial surfaces colored by
      FOS + slice hover (W, N', S, E/X) + thrust toggle. FEM view:
      mesh/deformed toggle, contour dropdown, plastic points toggle,
      SRF curve subplot. plotly.js INLINE (include_plotlyjs=True) so
      no network needed; plotly optional dep (pyproject extra
      `interactive`). Tests: file generated, trace-name string
      asserts, no-plotly degradation.
- [ ] Phase 6 — demo artifacts to docs/examples/ (committed HTML):
      slope_le_calc_package.html (reinforced + MC + search),
      fem_srm_calc_package.html, slope_le_interactive.html,
      fem_srm_interactive.html. Final report.

## Baseline (Phase 0)
- Tests before (branch @ 2789278): recorded in Phase 1 commit.

## Next action
- Implement Phase 1 (slope_stability/plotting.py + trial_surfaces
  storage in search.py/results.py) and its tests.

## Notes / decisions
- SearchResult gains `trial_surfaces: list[dict]` (additive,
  default []) — each {"FOS", "points": [(x,z)...]} for noncircular
  trials; circular trials use existing grid_fos (xc, yc, R) and are
  re-drawn geometrically (drawing arcs is rendering, not
  re-derivation).
- fem2d srm.py `_result` additively returns `plastic_gp` = Gauss
  points on the reduced-strength MC yield surface at the last stable
  state; analyze_slope_srm stores it on FEMResult.plastic_points.
  calc_steps only renders it.
- fem2d calc_steps `analysis` param = dict (like slope/sheet_pile):
  optional keys model_description, soil_layers (list of dict as
  passed to analyze_*), surface_points, element_type, notes.
  Dispatch on result type (FEMResult / SeepageResult /
  ConsolidationResult / StagedConstructionResult).
- pyproject: new optional extra `interactive = ["plotly>=5"]`,
  added to `full`.
- NOTE (session): harness Write/Edit tools were pinned to another
  worktree; all files in this branch are written via shell heredocs.
