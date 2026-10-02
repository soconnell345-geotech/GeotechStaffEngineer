# Session state — Document Review 5.32 and the Foundry test bed (2026-10-02)

Written before a context compaction so the next context loses nothing. The
short pickup is `HANDOFF.md` §0a-current (top entry); this file is the long
form. Session "document-review-architecture" (Claude Code, Opus 5.5).

## 1. The owner's standing directions (this session and before)

- **Vision first.** "Systematic visual enumeration is much more flexible; even
  if it's more expensive, we don't care much here. The tools can be a fallback
  or supporting evidence, but the LLM shouldn't use them in lieu of vision"
  (2026-10-01). Robust first, cost later (2026-09-25).
- **No overfitting to one example.** Tester feedback becomes a suite TASK, not
  a prompt rule; special-purpose tools stay optional; a switch goes on by
  default only when the suite says so (total not down, no category regressed).
- **Markups:** "never draws anything based on inferred locations; always use
  confirmed coordinates ... should also visually confirm its markups."
- **Nothing is published** (PyPI, tags, version bumps) until the Foundry
  feedback is in — for 5.31's switches AND for 5.32 itself. Release on the
  owner's word only. "Let's just get a big build for .32."
- **Foundry is the test bed; Tiny Apps is production.** The owner's Funhouse
  budget is small; Foundry (Palantir, State Dept enclave, cleared to SBU) has
  ample model quota. Report TOKENS and TIME on Foundry, never dollars (no
  per-token rate there; the $2.50/M in brief 2 was Funhouse's and misled the
  FDE's estimates).
- **Data:** the report-ingest bundle (manifest, truth, templates.json) is NOT
  SBU (owner, 2026-10-01) — it may pass through this chat but never into a
  commit. Material marked SBU (the IZD drawing set and its screenshots) never
  goes to Anthropic and never into the public repos; it lives only under
  gitignored `raw/` folders. Internal hostnames/URLs never go into commits.
- One subagent at a time; builders run suites in the FOREGROUND; gate on
  pytest's exit code; edit by exact text (Edit tool) — scripted replaces fail
  on CRLF files.
- The owner is not a developer: minimal git/dev explanation.

## 2. Branches (nothing pushed, nothing published)

**App `release/5.32.0`** (from master 38435c9 = v5.31.0), 18 commits:
f75b04c Foundry vision door · 0f80ab4 per-page past-conversation restore ·
9c3a167 failed-calls column + Foundry run 2 + field findings · b686e0a
annotate_document checks its marks (`funhouse_agent/markup_check.py`) ·
4a05888 look-first review prompts · 3a3dbc9 SharePoint browser URLs · e576e38
no auto-orientation for a mid-conversation screenshot · aec07a3 + af54367
paste a screenshot (chat box with `accept_file` where HTTP uploads work; the
ws uploader's paste target on Funhouse; verified live in a sandboxed app) ·
04bb821 `produce-circle-tags` + check `markups_on_targets` · ad83eb2 stale
prompt notes (deep reviewer `REVIEWER_DEEP_PROMPT`, structural refs,
seismic md, calc delegation) · 4b08784 report_ingest `triage_model` reaches
the ingest triage (`ingest_report(triage_engine=)`) · b9f8fdb check faults
fixed (`produce-markup` file name; hyphenated terms) + `set-long-rare-tag` +
check `pages_listed` + Foundry run 3 recorded · f617fe4 the `minimal` arm
(`funhouse_agent/deep/minimal_agent.py`) · 6aee307 README/guide look-first
wording · df9e349 Foundry wrapper sends `original` as AUTO · HANDOFF commits.

**planlens `release/0.11.0`** (from main 574f7d7 = v0.10.1), 3 commits:
0d9787b drawing-sheet advice wording · f5bee8a circle markups, visible
labels, `view`+`image_box` anchors, `box`/`page_bbox` aliases · b577851
`tag_fixtures` `gce_growth` / `extra_callouts`.

**Test wheels (NOT published)** in `C:/Users/socon/OneDrive/dev/foundry_handoff/`:
`planlens-0.11.0rc1` (from b577851) and `geotech_staff_engineer-5.32.0rc2`
(from df9e349; pin `planlens>=0.11.0rc1`). Built in throwaway worktrees with
the versions edited only there; the branches still say 5.31.0 / 0.10.1.

**Tests run on the branch** (each on pytest's exit code): deep + reviewer
599/1; webapp 392; review_eval 75; report_ingest 1,298; planlens tools +
document 605, tag/find_like 19; minimal 6 (+39 m1); palantir wrapper 17.
**The FULL gate has NOT run** — a background run was stopped for low memory
(funhouse_agent 29 % with no failures; planlens full suite not started). Run
it in a clean worktree before the release.

## 3. Foundry (the AI FDE — Claude Opus 5.5 inside the owner's Foundry)

Briefs in `foundry_handoff/`: `AI_FDE_BRIEF.md` (Document Review suite),
`AI_FDE_BRIEF_2.md` (report-ingest first live checks + the 106-question
geotech eval on GPT-5.4), `AI_FDE_BRIEF_3.md` (rc2: baseline / sweep /
minimal, 37 tasks, Sol at full resolution through the FDE's Responses glue).
The FDE runs lightweight Python transforms, one incremental dataset per
out_dir, a retry layer (connection drops, timeouts, rate limits) and a
throttle; its glue: `foundry_vision_model.py` (= brief 1 Appendix A),
`foundry_responses_model.py` (GPT-5.4 is Bedrock-only on Foundry: Responses
route), `foundry_full_res.py` (`FullResResponsesChatModel`,
`FullResSdkChatModel`) — ASK FOR THESE and fold the Responses route into
`webapp/palantir_sdk_engine.py` before the release.

**Foundry facts** (memory `reference-foundry-model-route`): `ImageDetail` has
AUTO/HIGH/LOW/UNKNOWN only; HIGH CAPS images (Sol 692 image tokens at 1024
and 2048 px ≈ 768 px); AUTO/unset = full size (Sol chat route ≈ 2048 px cap,
Responses ≥ 4096 px; GPT-5.4 Responses ≈ 10M px); raw ORIGINAL → 400. An 11 px
code was misread at HIGH and read exactly at AUTO. Every Foundry run before
rc2 was at ~768 px with heavy tiling. GPT-5.4's project limit bites on
TOKENS per minute. Report-ingest full pipeline ≈ 2.7M input tokens and 47 min
per report on GPT-5.4.

**Results so far.**
- Run 2 (5.31, 8 tasks): GPT-5.4 baseline 8/8, lean 8/8; Sol 8/8, 8/8.
- Run 3 (5.31, Sol, 7 arms × 35 tasks, ~768 px), RESCORED after two check
  faults: baseline 35/35 (3.6M in, 49 min), sweep 35/35, geometry 35/35,
  lean 34, grounded 34, digest 34, inline 32 (loses 3 locate tasks),
  overview 9/10 vs grounded 10/10 on the UFC tasks. The suite is saturated on
  Sol except for coverage. Files: `module_work/review_eval_results/
  2026-10-02_foundry_5.31.0/`; ledger `module_work/REVIEW_HARNESS.md`.
- Brief 2 (report ingest, GPT-5.4): stages c (calc + soundings), d (logs, lab,
  narrative) and e (whole-report vision labels, GPT-4.1-mini) DONE — their
  RESULTS.md NOT yet received. Stage a skipped. Stage b (whole pipeline): 6 of
  38 done, then the owner was told to run ONLY reports with answer keys
  (in the label spreadsheet, oos_labels.json, or a truth/ file name); status
  unknown. FDE package findings: the ingest stage ignored `triage_model`
  (FIXED, 4b08784); R11 wrote no DIGGS file and 4 of 5 DIGGS read back
  differently from the record (pydiggs not installed so no schema check) —
  NEEDS the details from the hand-back.
- Brief 2 geotech eval (GPT-5.4, 106 questions): 62/68 keyed correct (91.2 %
  vs 43/60 = 71.7 % in July on Funhouse "medium"); tool-error rate 38.7 % (vs
  23 %), hallucination-on-error 5.7 % (6, 4 likely false positives); 85 s per
  question; 10.35M tokens; optional packages missing on Foundry: eqsig,
  gstools, liquepy, pydiggs, pystra, pystrata — separate those from real tool
  errors when the per-question results arrive.
- Brief 3 (rc2) STARTED on the owner's word; the FDE runs baseline first.
  Claude's suggested trim (if the owner wants it after round 1): baseline on
  all 37 once at full res; the two new tasks × baseline/sweep/minimal × 3
  repeats; baseline at forced HIGH (`GEOTECH_VISION_BUDGET=openai-high` in a
  custom arm) on the 15 drawn-lettering + 2 new tasks for a resolution A/B;
  minimal on the 21 drawing tasks + ~5 text-heavy ones; skip sweep/minimal on
  the long manuals at full res.

## 4. Open items, in order

1. Read brief 3's first round (baseline at full res): tokens per task, any
   regression from the look-first prompt or the markup check; then the other
   arms or the trimmed plan.
2. Receive brief 2's hand-back: c/d/e RESULTS.md, stage b on keyed reports,
   the geotech eval per-question results; investigate the DIGGS read-backs.
3. Fold the FDE's Responses-route glue into the package.
4. Decide 5.32 defaults from rc2 (does `sweep`/lean become default?).
5. Release: planlens 0.11.0 → app 5.32.0 (pin `planlens>=0.11`), docs
   (CLAUDE.md state block, HANDOFF §0a, DATABRICKS_INSTALL §11 — the
   docs-currency test), FULL gate in a clean worktree, tags — owner's word.
6. 5.33 candidates: an image crop/zoom tool (raster images and pasted
   screenshots have no zoom today) + a Chartography run (license to check on
   the Hugging Face card; BenchCAD judged not relevant) + a small geotech chart
   truth set (`module_work/FUTURE_IDEAS.md` "VISION HARNESS BENCHMARK").

## 5. Field feedback of 2026-10-01

`module_work/field_feedback/2026-10-01_doc-review-penetrations_v5.31.0/FINDINGS.md`
(F1-F6, all built into 5.32); raw export (SBU-marked screenshots) under its
gitignored `raw/`.
