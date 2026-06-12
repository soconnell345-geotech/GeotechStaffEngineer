# Weekend QC run — 2026-06-13 (owner away until Monday)

Owner ask: "QC of all code and reference retrieval, make any necessary fixes
(no wheel rebuild until Monday). Also find references that give example
validation problems (FLAC manual came to mind, also our existing reference
manuals) and run modules against them to check correctness. Stagger agents."

Standing rules: one subagent at a time; commit+push at milestones; NO version
bump / wheel / publish. Worktree: .claude/worktrees/v5.1-todos (current with
master fb34502).

## Plan / status

- [ ] **A. Baseline** — full repo suite (`pytest -q`) + geotech-references
      suite; record counts. (background shell)
- [x] **B. Code QC** — adversarial review of this week's unreviewed diffs
      (c974247 adapter sweep, d7a3b30 _fileio) + general code QC; fix findings.
- [x] **C. Reference-retrieval QC** — registry wiring of all 21 reference
      modules, FTS DB integrity, semantic aliases, figure_db/figure catalogs,
      query-expansion eval (scripts/eval_retrieval_recall.py), reference_mode
      consult path offline. Fix findings.
- [ ] **D. Worked-example inventory** — sweep reference library text JSONs +
      DM7/GEC/UFC for worked example problems with numeric answers; map to
      modules. Add public FLAC/Itasca verification problems (web).
- [ ] **E. Validation runs** — implement inventory items as checks, run
      modules, classify pass/fail, fix real module bugs, keep a results table.
- [ ] **F. Wrap-up** — summary report for owner, memory update, final push.

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
