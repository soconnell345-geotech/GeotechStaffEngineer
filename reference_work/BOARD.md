# Reference-layer work board

Tracks the reference build/QC effort. Workers: `.claude/agents/figure-catalog-builder.md`
(vision catalogs) + general-purpose agents (deletions, new-reference digitization).
Orchestrator commits; workers produce files + report. No version bumps until EOD publish.

## Workstream 1 — prune + fill UFC vision  ✅ DONE (2026-06-05)
- [DONE] DELETED `noaa_frost`, `ufc_dewatering`, `fema_p2192` from both repos
  (submodule d4b9679, parent c4dc8aa).
- [DONE] Built figure catalogs:
  - `ufc_backfill`  ← `ufc_3_220_04fa_2004.pdf` → 14 figs, 100% confirmed.
  - `ufc_expansive` ← `ufc_3_220_07.pdf` → only 3 confirmable (scanned PDF, no OCR
    caption layer; remaining figures need OCR-first — FOLLOW-UP).
  (committed: submodule eda8bbd, parent pin 36467be)

## Workstream 2 — all vision accuracy to 100% (page_estimated -> 0)
- gec_8       37% -> ✅ 100% (132 figs; 9 indices fixed)  [committed eda8bbd]
- gec_11      42% -> ✅ 100% (142 figs, multi-vol; 61 fixed) [committed eda8bbd]
- micropile   83% -> [in progress, agent vision-cleanup]
- gec_10      94% -> [in progress]
- gec_5       95% -> [in progress]
- gec_13      96% -> [in progress, multi-vol]
- ufc_pavement 97% -> [in progress]
- dm7_1/dm7_2, gec_4/6/7/9/14, gec_12  already 100%
Method: for each estimated figure, find its true page by caption/label search in the
source PDF (see figure-catalog-builder). One worker per reference for the big two.

## Workstream 3 — new references from orphan/added PDFs (full pipeline)
Each = structured chapter-text JSON + python lookups + figure catalog + registry wiring in BOTH repos.
- ✅ FEMA P-2082 (2020 NEHRP Provisions) — `fema_p2082` module. Site classes BC/CD/DE (BC baseline),
  Fa/Fv removed (USGS geodatabase), 19 figs 100%, 86 tests. submodule 1612e6a / parent 0ed7248.
- ✅ California Trenching & Shoring (`california_trenching`) — 15 tables, 16 eqns, 7 chapters,
  177 figs 100%, 83 tests. submodule c1fb7ce / parent 8483b31.
- ✅ FHWA-NHI-05-037 Geotechnical Aspects of Pavements (`fhwa_pavements`) — 14 tables, 11 eqns,
  3 geotech chapters (4/5/7), 177 figs 100%, 89 tests. Resilient modulus Mr (default by AASHTO/USCS
  class + CBR/R-value/DCP/plasticity correlations + stress-dependent granular + seasonal/backcalc),
  CBR, soil-as-pavement-material, drainage mi/Cd + permeability, frost F1-F4, swell, stabilization/
  geosynthetics, compaction. Distinct from ufc_pavement (UFC 3-250-01). submodule c0552e7.

## fdm2d hang fix (analysis module, from the funhouse run)
✅ Wall-clock guard (max_seconds, default 30s) on the explicit solver so it can't grind ~20 min on a
non-converging case. parent d5e4362. Plus a runner-side per-question signal.alarm timeout for the suite.

## Sequencing
DEL (registries) must commit before Workstream-3 new modules touch the same registries.
Workstream 2 + the UFC catalogs touch only figures_catalog.json — safe to run after DEL.
