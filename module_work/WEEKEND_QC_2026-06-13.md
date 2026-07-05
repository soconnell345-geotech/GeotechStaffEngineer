# Weekend QC run — 2026-06-13 (owner away until Monday)

Owner ask: "QC of all code and reference retrieval, make any necessary fixes
(no wheel rebuild until Monday). Also find references that give example
validation problems (FLAC manual came to mind, also our existing reference
manuals) and run modules against them to check correctness. Stagger agents."

Standing rules: one subagent at a time; commit+push at milestones; NO version
bump / wheel / publish. Worktree: .claude/worktrees/v5.1-todos (current with
master fb34502).

## Plan / status

- [x] **A. Baseline** — full repo suite (`pytest -q`) + geotech-references
      suite; record counts. (Closed 2026-07-05: full suite re-run green with
      the eval fixes — 8056 passed / 48 skipped; geotech-references 3703
      recorded in Phase C.)
- [x] **B. Code QC** — adversarial review of this week's unreviewed diffs
      (c974247 adapter sweep, d7a3b30 _fileio) + general code QC; fix findings.
- [x] **C. Reference-retrieval QC** — registry wiring of all 21 reference
      modules, FTS DB integrity, semantic aliases, figure_db/figure catalogs,
      query-expansion eval (scripts/eval_retrieval_recall.py), reference_mode
      consult path offline. Fix findings.
- [x] **D. Worked-example inventory** — sweep reference library text JSONs +
      DM7/GEC/UFC for worked example problems with numeric answers; map to
      modules. Add public FLAC/Itasca verification problems (web). (Done —
      committed a9f81a1: `validation_examples/INVENTORY.md`, 25 problems,
      14 modules; checkbox was stale.)
- [x] **E. Validation runs** — all 25 inventory problems implemented as
      offline pytest checks (`validation_examples/test_published_v0*.py`, 87
      tests) with `RESULTS.md`. Done 2026-06-14 (resumed under Opus after the
      Fable→Opus switch). No analysis-result bugs found; one additive fem2d
      capability gained (see below). Verdicts + gaps in RESULTS.md.
- [x] **F. Wrap-up** — summary report for owner, memory update, final push.
      (Covered by the follow-on docs session: `docs/V5.1_SUMMARY.html`,
      HANDOFF refresh 70888e8; checkbox was stale.)

## Log

- 2026-06-12 ~21:30 UTC: plan written; baseline started.
- B done: adversarial review of 2c6e1c9..fb34502 — 0 blockers, 4 minors +
  2 nits fixed in c374fce (_fileio CRLF boundary + rescue collisions, latex
  verification scope, mse_wall alias, fem2d layer keys, valid-list hints).
- C done: 21 modules wired clean, 2910 figures page_estimated=0, eval
  off 11% / auto 44% / 0 disturbance (matches baseline), spot values correct.
  Fixed in submodule 99e67b7: reference_search zero-hit OR fallback
  (auto/fill/rerank only) + gec_11 title normalization. 3703 sub tests pass.
  Deferred: synonym-map curation (CRR group has dead surface forms).
- E done (2026-06-14, under Opus): 25 published worked examples run as 87
  offline tests across 8 module-grouped batches (commits d599f0d, 59a249e,
  f0c9a33, 07efb7d, a1795d0, 9d9b500, 3c42474, + fem2d batch). NO analysis
  bugs — every discrepancy resolved to units / method-variant / datum /
  convention after investigation; modules NOT tuned to any single example.
  PASS where methods align (incl. M-O KAE 0.4782 regression anchor, V-021
  Vesic bearing <0.2%, V-004 downdrag NP <1%, V-024 Salencon cavity R0 exact,
  V-001 Nordlund ±15%, V-015 Fellenius 0.863). Documented coverage gaps for
  v5.2 (see RESULTS.md owner notes): drilled_shaft rational GEC-10 chains;
  axial_pile per-layer toe-phi + stiff-clay alpha; wave_equation diesel model;
  MSE LRFD/bar-mat curves; soe FHWA apparent-pressure + sidewall-shear heave;
  RAP upper-zone + Priebe convention; Hough settlement; fem2d undrained
  consolidation transient (staggered-Biot gives correct drained end-state
  only). ADDED to fem2d (additive, opt-in, general — not benchmark-tuned):
  roller_base BC + initial_stress_relaxation (excavation/cavity unloading),
  which enabled V-024. This is the only shipped-code change in Phase E ->
  justifies the rc4 wheel rebuild.
- 2026-07-05 (Fable, eval-review session): owner ran the 71-Q suite on the rc5
  wheel (results: `docs/geotech_eval_20260705.json` / `docs/geotech_eval.md`).
  Verdict: 26/31 auto-graded pass (83.9%); 4 of the 5 fails were missing
  optional packages on the cluster (gstools/pygef/python-ags4/ezdxf — agent
  refused honestly), 1 real content miss (REF-1 alpha narrative). All 5 P1
  flags verified as recovered-after-error false positives. Fixes landed on
  master: drilled_shaft to_dict per-layer alpha/fs breakdown, 6 new dispatch
  aliases, eval-harness P1 recovered-split + optional-dep preflight,
  `[deep,full]` install docs. Phase A closed with the post-fix full-suite
  run: **8056 passed, 48 skipped** (14:20); board checkboxes D/F reconciled.
