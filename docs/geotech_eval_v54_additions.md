# Eval-suite v5.3/v5.4 additions (E10 + E11)

29 answer-keyed questions added to `funhouse_agent/geotech_test_suite.json`
(71 → 100), exercising the tools shipped in v5.3/v5.4. Every answer key was
produced by RUNNING the actual module (not from memory) and is re-verified by
`funhouse_agent/tests/test_eval_suite_v54.py`, which recomputes each key and
asserts it lands within the question's own tolerance. The LIVE model run stays
owner-gated; this deliverable is the suite + keys + this manifest.

Weighting: 17 of 29 exercise the slope / rapid-drawdown / seismic surface (the
biggest new area), the rest cover the composite-EI, earth-pressure/MSE-LRFD,
fem2d (elastic footing + monolithic consolidation + footing bearing capacity +
SRM mesh-refinement study), pdf and drawing-IR tools. The three fem2d-owned
questions CON-1 / FTG-1 / MRS-1 are the E11 addition (2026-07-07).

## New questions

| id | module | tool exercised | pinned key |
|----|--------|----------------|-----------|
| RDD-1 | slope_stability | `rapid_drawdown_fos` USACE 2-stage | FOS 1.317 |
| RDD-2 | slope_stability | `rapid_drawdown_fos` Duncan-Wright-Wong 3-stage | FOS 1.385 |
| RDD-3 | slope_stability | 3-stage `stage3_effective_normal='gle'` option (E2) | FOS 1.428 |
| RDD-4 | slope_stability | 2-stage with `stage1_phreatic_points` steady-seepage | FOS 1.441 |
| RDS-1 | slope_stability | `search_rapid_drawdown` (E1) critical circle | min FOS 0.958 |
| INF-1 | slope_stability | `infinite_slope_fos` dry cohesionless | FOS 1.349 |
| INF-2 | slope_stability | `infinite_slope_fos` ru pore pressure | FOS 1.401 |
| INF-3 | slope_stability | `infinite_slope_fos` seepage-parallel | FOS 0.905 |
| NMK-1 | slope_stability | `yield_acceleration` ky | ky 0.360 |
| NMK-2 | slope_stability | `newmark_displacement` (rectangular pulse) | 79.4 cm |
| NMK-3 | slope_stability | `newmark_jibson2007` regression | 2.86 cm |
| SPL-1 | slope_stability | `StabilizingPile` specified shear, active | FOS 1.336 |
| SPL-2 | slope_stability | pile `support_convention='passive'` (E6) | FOS 1.280 |
| SPL-3 | slope_stability | `ito_matsui_lateral_force` (1975) | 412 kN/pile |
| PPG-1 | slope_stability | `pore_pressure_points` flow-net grid (E3) | FOS 1.223 |
| TCK-1 | slope_stability | `tension_crack_side='exit'`+`model='truncation'` (E4) | FOS 1.597 |
| SUR-1 | slope_stability | multiple `surcharges` zones (E8) | FOS 1.773 |
| CEI-1 | lateral_pile | `composite_section_ei` filled steel pipe (E5) | EI 330,780 kN·m² |
| CEI-2 | lateral_pile | `composite_section_ei` reinforced concrete (E5) | EI 175,652 kN·m² |
| EPC-1 | retaining_walls | `earth_pressure_coefficient` Rankine Ka | 0.307 |
| EPC-2 | retaining_walls | `earth_pressure_coefficient` Rankine Kp | 3.255 |
| EPC-3 | retaining_walls | `earth_pressure_coefficient` Caquot-Kerisel log-spiral Kp | 7.2 |
| MSE-1 | retaining_walls | `mse_lrfd_external_stability` sliding CDR | 1.84 |
| FF-1 | fem2d | `fem2d_foundation` elastic strip footing | 0.01946 m |
| CON-1 | fem2d | `fem2d_consolidation` monolithic Taylor-Hood Biot (E11) | U 0.599 at t=1e8 s |
| FTG-1 | fem2d | `fem2d_footing_capacity` load-control collapse, Prandtl (E11) | Nc 5.12 / q_ult 512 kPa |
| MRS-1 | fem2d | `srm_mesh_refinement_study` mesh-consistency study (E11) | finest FOS 1.531 |
| CAL-1 | pdf_import | `calibrate_scale` two-point scale | 0.05 m/unit |
| DIR-1 | drawing_ir | `digitize_drawing` DXF entity counts | 3 polyline / 2 text |

DIR-1 reuses the bundled `funhouse_agent/eval_samples/sample_section.dxf`.
CON-1 keys the degree of consolidation (the load-induced undrained pressure the
monolithic scheme uniquely captures shows a known Taylor-Hood boundary-layer
overshoot at the drained face, so the robust reproducible output is U). FTG-1 is
the FE analogue of a bearing-capacity calc (Prandtl anchor, VALIDATION.md §3).
MRS-1 exercises the `srm_mesh_refinement_study` fem2d utility (which drives the
`fem2d_slope_srm` tool at each mesh); FTG-1/MRS-1 are `-m slow` (~9 s / ~19 s),
CON-1 is fast.

## Deferred / not added (flagged for the owner)

These v5.3/v5.4 items were on the list but left out with a reason; each has a
ready answer key the owner can drop in once the gap is closed:

- **Log-spiral Caquot-Kerisel Kp** — DONE (follow-up). Wired
  `soe.earth_pressure.caquot_kerisel_Kp` into the `earth_pressure_coefficient`
  adapter as `theory='caquot_kerisel'` (passive-only, with the wall-friction
  `delta_deg`); question EPC-3 added (phi=35°, delta=23.33° → Kp 7.2, recomputed
  via the adapter path).
- **Monolithic Taylor-Hood consolidation** (`fem2d_consolidation`, scheme
  `monolithic`) — ADDED (E11, CON-1). Calibrated to the V-023 column
  (E=529412 kPa, nu=0.3235, M=4e6 kPa, mobility 1e-10) with time points that
  give a partial U; the key is the degree of consolidation (0.599), robust and
  fast (~1 s). The undrained interior p0=83.9 kPa is not exposed by the adapter
  (only the peak, which sits in the drained-face boundary layer), so U is the
  reproducible choice.
- **SRM mesh-refinement study + footing-SRM** (`fem2d_footing_capacity`,
  `srm_mesh_refinement_study`) — ADDED (E11, FTG-1 + MRS-1). FTG-1 keys the
  Prandtl footing Nc=5.12 via the new `fem2d_footing_capacity` adapter tool;
  MRS-1 keys the finest-mesh FOS of a 2-mesh refinement study. Both `-m slow`.
- **pdf label→region** (`propose_role_mapping`) — not added; `calibrate_scale`
  (CAL-1) covers the pdf-geometry surface. Easy to add later if wanted.

## Harness note

`eval_harness.run_suite` sizes itself from `len(questions)`, so it needs no
change — it picks up all 100. Docs that named the suite count (now 100) —
`docs/funhouse_agent_guide.md`, `CLAUDE.md`, `HANDOFF.md` — carry the earlier 97
and can be bumped by their owners; the dated run reports
(`docs/geotech_eval*.{md,json}`) are historical and left as-is.
