# DR-39: strip the review experiments that never earned a place

Branch `claude/dr-39-strip-review-experiments`, from master `3b30181`.
BACKLOG DR-39 (owner 2026-11-09: "strip down the non-beneficial parts of the
package and finish the split"). This is the first step of the clean Document
Review package for Tiny Apps.

The task is code, not a calculation, so there are no printed-page citations.
Each verdict below cites the code and the measured runs instead.

**Evidence used:**
- `module_work/REVIEW_HARNESS.md`, results ledger, runs 1-7;
- `module_work/review_eval_results/2026-10-02_foundry_5.31.0/` (run 3: every arm on GPT-5.6 Sol, 35 tasks);
- `module_work/review_eval_results/2026-10-04_foundry_5.32.0rc3/` (run 4: baseline, sweep and minimal at full resolution, 37 tasks);
- `module_work/review_eval_results/2026-10-08_foundry_5.33.0rc1/TRACE_REVIEW.md` (run 7: coverage, checklist, patch align);
- `module_work/review_eval_results/2026-11-08_chart_suite_ev35/TRACE_REVIEW.md` §6;
- BACKLOG.md's switch table (the rows from `GEOTECH_REVIEW_AGENT=lean` down to `GEOTECH_CHART_CODE_VOTER`);
- `module_work/REVIEW_ARCHITECTURE.md`.

## 1. Inventory

How to read the verdicts:
- **remove:** off by default, and no run showed a gain.
- **keep:** on by default, or default code uses it.
- **keep for now:** the reason is in the row.

"Arms" means the entries of `review_flags.ARMS` that set the switch.

| Switch | Default | Code, modules, tests, arms, docs | Measured evidence | Verdict |
|---|---|---|---|---|
| `GEOTECH_REVIEW_AGENT=lean` | unset | `deep/review_agent.build_review_agent`, `make_reader_tool`, `TODO_*`, `TASK_DESCRIPTION`; `review_build` lean branch; `webapp/core.attachment_note(review=)` lean wording; `review_prompt.DOCUMENT_REVIEW_PROMPT_LEAN` / `build_document_review_prompt(lean=)`; fingerprint build `review_lean`; arm `lean`; tests in `test_review_harness_offline`, `test_coverage_offline`, `test_wave2b_guards_offline`, `test_vision_images` | Run 1 (8 tasks, GPT-5.4): lean 7/8 vs baseline 8/8. Run 2 (Foundry, 8 tasks): 8/8 both. Run 3 (Sol, 35 tasks): lean 34/35 vs baseline 35/35; it breaks bioretention-section-dims at 86 vs 49 min. Never run on GPT or after 5.32 (BACKLOG table). The default build took its three fixes (DR-01 to DR-03). | **remove** |
| `GEOTECH_REVIEW_AGENT=minimal` | unset | `deep/minimal_agent.py`, `deep/inline_images.py`, `inline_store.py`; arm `minimal`; fingerprint build `review_minimal`; `test_review_minimal_offline` | Run 4: 33/35 at ~4x tokens. EV-35 chart suite: matches baseline on Opus at ~1/5 the input tokens; fixed two Sol runs and broke none. | **keep for now:** the brief says so; the chart suite (EV-35) measures it as an arm. |
| `GEOTECH_VISION_INLINE` | off | `review_agent.INLINE_DESCRIPTIONS` and the lean build's `inline` / `inline_files`; arms `inline`, (with overview) the lean contact-sheet path; tests in `test_review_harness_offline`, `test_review_m1_offline`, `test_review_prompt_rules`, `test_vision_images` | Run 3: 32/35. It loses 3 of 15 locate tasks (BACKLOG table: DR-28, VIS-13). | **remove** (switch only). The inline image machinery stays: the `minimal` build uses it through `make_vision_tools(inline_images=True)`, not through the switch. |
| `GEOTECH_REVIEW_SWEEP` | off | `deep/sweep.py` (`sweep_pages`); arm `sweep`; 3 tests in `test_review_harness_offline` | Run 3: 35/35, a tie, at 4.85M vs 3.62M input. Run 4: 35/35 and 6/6 vs 5/6, at 2.4x the tokens and 2.7x the time. The extra pass was the markup case, not coverage. | **remove** |
| `GEOTECH_REVIEW_FINDINGS` | off | `review_findings.py`, `deep/findings_tools.py` (`record_finding`, `update_finding`, `list_findings`, `findings_report`); `cited_pages.unopened_citations` (only caller); most of `test_review_m1_offline`, 2 tests in `test_cited_pages`, 1 in `test_orientation_overview` | Never measured: it had no arm (BACKLOG table, DR-27). | **remove** |
| `GEOTECH_REVIEW_OVERVIEW` (+ `_OVERVIEW_PAGES`) | off | `webapp/review_page.ORIENTATION_REQUEST_LONG` / `_SHORT`, `_ORIENTATION_ASK`, `pdf_page_count`, the switch branch of `orientation_request_for`; `review_flags.overview_pages`; arm `overview`; 7 tests in `webapp/tests/test_orientation_overview.py` | Run 3 (UFC tasks with `orientation=True`): 9/10 vs grounded's 10/10; it breaks ufc04-table-5-1. | **remove** |
| `GEOTECH_REVIEW_GEOMETRY` | off | `deep/geometry_tools.py` (`drawing_callouts`, `drawing_dimensions`, `title_block`, `revision_clouds`); arm `geometry`; `test_review_geometry_offline.py` | Run 3: 35/35, a tie, at 5.17M vs 3.62M input. "The construct finders reach no user while off" (BACKLOG table). | **remove** |
| `GEOTECH_REVIEW_DIGEST` (+ `_SHAPE1_PAGES`) | off | `review_digest/` (4 files), `deep/digest_tools.py`; arm `digest`; `review_digest/tests`, `test_review_digest_tools_offline.py` | Run 3: 34/35; it breaks ramp-detail-callouts (read 3/4 as 34). | **remove** |
| `GEOTECH_REVIEW_KEPT_READINGS` | off | `review_flags.kept_readings_on_review_page`; the `_drop` line in `review_build`; `review_agent.KEPT_READINGS_TOOL`; arm `kept_readings`; 1 test | Built 2026-10-15 (TA-28). The arm was never run on the review suite (none of the results folders has it). | **remove.** The review page now always leaves `kept_readings` out, which is what it did by default; the geotech page keeps the tool. |
| `GEOTECH_VISION_TEXT_CONTEXT` / `GEOTECH_VISION_STRUCTURED` | off | Read in `vision_tools.py` (lines 1366-1367, 3303, 3556, 3565) and `vision_view.text_context` / `split_located`; arms `grounded` / `inline` (on the lean build, now gone) | Run 3 `grounded` (lean + both): 34/35; it breaks calc-bearing-consistency. Structured is partly superseded by pixel boxes (BACKLOG table). | **keep for now:** the code paths are in `vision_tools.py`, which this branch may not edit. The switches stay usable as a custom arm `(name, {env: "1"})`; their only named arms went with lean. Follow-up §4. |
| `GEOTECH_VISION_TURN_RETRY` | off | `vision_tools.py` (1368, 3488, 3498); arm `turn_retry`; `test_vision_turned` | Built for VIS-01; not measured on the suite yet (EV-13). | **keep for now:** the code is in `vision_tools.py` (excluded). |
| `GEOTECH_VISION_PATCH_ALIGN` | off | `vision_view.patch_align` / `align_to_patches`, read in `vision_tools.py:1365`; `test_vision_brief4` | Run 7 / brief 5 part F: undecided, "stay OFF" until ≥10 repeats (EV-02). | **keep for now:** read in `vision_tools.py` (excluded). |
| `GEOTECH_COVERAGE` | off | `coverage.py`, `deep/coverage_tools.py`; arms `coverage`, `checklist` | Run 7: 39/42 either way, at +47 % input; not ready. CV1-CV4 fixed, not re-measured (DR-23). | **keep for now:** `document_tools.py` (excluded) imports the ledger and `cited_pages.py` (default code) reads it. Removing the gate and tool alone would leave the ledger in place. Follow-up §4. |
| `GEOTECH_REVIEW_CHECKLIST` | off | `report_ingest/checklist.py`; offered on the geotech page only (RI-34) | Run 7: not ready (a false fail, a false pass, a stuck "unsure"). Since RI-34 it is never on the Document Review page. | **keep for now:** a GeotechStaffEngineer report-ingest feature, not a Document Review experiment (OWN-31 / RI-105). |
| `GEOTECH_REPORT_INGEST` | off | `report_ingest` sub-agent on the geotech page | Not a review-suite switch; the report-ingest train has its own measurements. | **keep for now:** a geotech-page feature, never on the review page. |
| `GEOTECH_CITED_PAGES` | off (the check always runs and is recorded) | `cited_pages.py` (default code: `write_diggs` cross-checks, the `cited_pages` event); arms `cited_pages_note`, `cited_pages_retry` | Never measured (EV-18 open: "measure it" before either mode goes on). | **keep for now:** default code uses the module. Only the note/retry modes are switched, and EV-18 is still an open measuring item on both pages. |
| `GEOTECH_CITED_DOCUMENTS` | off (always recorded) | `cited_documents.py`, hooked from `dispatch.py` and `vision_tools.py:325`; arms `cited_documents_note` / `_retry` | Built for EV-21; not measured (EV-23). | **keep for now:** hooked from `vision_tools.py` (excluded), and mainly a geotech-page deliverable check. |
| `GEOTECH_STEP_CAP_ANSWER` | off | `webapp/turn_jobs.py:313`, `webapp.core.answer_at_step_cap`; arm `step_cap_answer` | "Never measured live" (TA-33). | **keep for now:** shared turn machinery for both pages. The geotech page has no model-call budget and still ends at the step cap, so this is not a Document Review experiment. The owner can rule it in or out with the geotech page. |
| `GEOTECH_FIGURE_HELPER` / `GEOTECH_PLOT_CODE` / `GEOTECH_JUDGMENT_CALLS` | **on** | `review_flags.DEFAULT_ON_ENVS` | Owner's rulings (GM-226, GM-227). | **keep** |
| `GEOTECH_CHART_CODE_VOTER` | **on** | `chart_reading.py`, read in `vision_tools.py:4695` | On by default (BACKLOG table). | **keep** |
| `GEOTECH_VISION_POLICY` (robust), `_BUDGET`, `_DETAIL`, `GEOTECH_CHART_BUDGET`, `_MAX_PX`, `_PROBE`, `_REPEAT_CACHE`, `_MAX_INFLIGHT`, `_MAX_PER_CONVERSATION`, `_BUSY_TRIES`, `_CALL_TIMEOUT_S`, `_MAX_TOKENS`, `GEOTECH_FIND_LIKE_BUDGET_S` | settings with working defaults | `vision_view.py`, `vision_tools.py`, `vision_probe.py` | Settings the default looks run on, not experiments. | **keep** |
| `GEOTECH_VISION_INLINE_KEEP`, `GEOTECH_REVIEW_MAX_MODEL_CALLS` | 2, 40 | `inline_store.KEEP_ENV` (minimal build); `review_agent.MAX_MODEL_CALLS_ENV` (the default build's budget) | In use by default (the budget) or by `minimal`. | **keep** |

**Suite arms after this branch:** `baseline`, `minimal`, `coverage`, `checklist`, `turn_retry`, `cited_pages_note`, `cited_pages_retry`, `figure_helper`, `plot_code`, `figure_helper_off`, `plot_code_off`, `step_cap_answer`, `cited_documents_note`, `cited_documents_retry`, `judgment_calls`, `judgment_calls_off`.

**Removed arms:** `lean`, `grounded`, `inline`, `sweep`, `overview`, `geometry`, `digest`, `kept_readings`. A notebook that asks for one gets the runner's existing "unknown arm" error, which lists the known arms.

## 2. What was removed

**Modules deleted:**
- `deep/sweep.py`
- `deep/findings_tools.py`
- `review_findings.py`
- `deep/geometry_tools.py`
- `deep/digest_tools.py`
- `review_digest/` (`__init__`, `build`, `digest`, `references`, and its tests)

**Code paths removed:**
- `review_flags`: 7 switches and their readers (`lean_agent`, `vision_inline`, `sweep`, `findings`, `overview`, `geometry`, `digest`, `kept_readings_on_review_page`), `overview_pages`, the overview and digest settings, and 8 arms.
- `review_build`: the `lean` branch. `kept_readings` is now always dropped on the review page.
- `deep/review_agent.py`: `build_review_agent`, `make_reader_tool`, `INLINE_DESCRIPTIONS`, `KEPT_READINGS_TOOL`, `TODO_PROMPT`, `TODO_TOOL_DESCRIPTION`, `TASK_DESCRIPTION`. It keeps the shared tables, budgets and helper pieces that the default and `minimal` builds import.
- `review_prompt` and `deep/prompt`: `DOCUMENT_REVIEW_PROMPT_LEAN` and the `lean=` argument.
- `webapp/core.attachment_note`: the `review=` argument and its lean wording. Its callers are `webapp/app.py`, `review_eval/runner.py` and `module_work/live_smoke/session.py`.
- `webapp/review_page`: the overview variants and `pdf_page_count`. `webapp/profiles` re-exports updated.
- `cited_pages.unopened_citations` (its only caller was `record_finding`).
- `webapp/tests/page_builds`: the `review_lean` build and the deleted modules in `REVIEW_MODULES`.

**Tests:**
- Tests of removed code deleted.
- Tests of shared code ported:
  - the inline middleware tests and the failing-helper test now run on the `minimal` build;
  - the suite's two-arm resume test now uses `turn_retry` as its second arm;
  - the fingerprint "has teeth" test now uses `review_default`.
- `test_review_m1_offline.py` became `test_review_switches_offline.py`: what remains is the arms, the settings an arm clears, and the inline image tools.
- New `test_the_removed_experiments_are_gone`.

**Docs:**
- The `review_flags` module docstring.
- `funhouse_agent/DESIGN.md` (`kept_readings`, cited pages, the figure helper, RI-34 lines).
- Docstrings in `vision_kit`, `inline_images`, `inline_store`, `minimal_agent`, `review_eval/runner` and `webapp/core` (`CACHE_DIRS`).
- A dated DR-39 note at the top of these history docs, whose text was not rewritten:
  - `module_work/REVIEW_HARNESS.md`
  - `REVIEW_ARCHITECTURE.md`
  - `APP_SPLIT_PLAN.md`
  - `harness_theory/README.md`, `document_review.md`, `shared_machinery.md`, `locating_things_on_a_page.md`

No env-var mention of a removed switch is left in `docs/`, `README.md` or `tinyapps/` (grep: none before or after).

### Lines removed, per area

From `git diff --numstat origin/master`. Moving `test_review_m1_offline.py` to `test_review_switches_offline.py` counts as a delete plus an add.

| Area | Added | Removed |
|---|---|---|
| Code, `funhouse_agent/` (excluding `review_digest/`) | 81 | 3,802 |
| Code, `funhouse_agent/review_digest/` | 0 | 2,004 |
| Code, `webapp/` | 21 | 115 |
| Tests (including the 362 `review_lean` lines of `page_fingerprints.json`) | 339 | 3,588 |
| Docs (`.md`) | 78 (the DR-39 notes, plus this file) | 11 |
| `module_work/live_smoke/session.py` | 1 | 2 |

## 3. The default pages are unchanged

**`webapp/tests/test_page_fingerprints.py` passes with no re-record:** 18 passed together with `test_public_text.py`, none skipped.
- `page_fingerprints.json` lost only the `review_lean` entries: the build under `builds` and its two planlens copies.
- That is 362 lines deleted and 0 added. The file was rewritten with the recorder's own format (`indent=1, sort_keys=True`).
- Every hash of `review_default`, `review_minimal` and `geotech_default` is byte-identical.
- `test_every_build_is_recorded` requires the recorded builds to equal `page_builds.BUILDS`, so the entry of a build that no longer exists had to go with it.

**Other checks:**
- The review prompt's text is unchanged: `_REVIEW_SOURCES_LEAN` was only renamed, and `DOCUMENT_REVIEW_PROMPT` hashes the same.
- The geotech page's attachment note is the one it always was. The review page already used it with the lean switch off.

## 4. Follow-ups in the files this branch may not edit

`vision_probe.py`, `vision_tools.py` and `document_tools.py` belong to other branches. This is what I would remove there, plus the name tables I left in place:
1. **`vision_tools.py`:**
   - `GEOTECH_VISION_TEXT_CONTEXT` and `GEOTECH_VISION_STRUCTURED`: the `text_context=` / `structured=` arguments at 1366-1367, the text-layer block at 3303, and the `LOCATED` request and `split_located` handling at 3556-3565 (then `vision_view.text_context` / `split_located` / `page_lines` if nothing else uses them). Off; their one measurement was 34/35 on lean.
   - The comments naming `GEOTECH_VISION_INLINE` at 2836, 3656, 3678 and 3716: say "the minimal build" instead.
2. **`document_tools.py`:** its import of `coverage` (the ledger recorder). That import is what keeps `GEOTECH_COVERAGE` here; with it gone, `coverage_tools.py`, the gate and the two arms could go if the owner rules coverage out (run 7: not ready).
3. **`vision_probe.py`:** nothing found to remove.
4. **Kept name tables** (inert, noted, not fixed). The tool names `sweep_pages`, `digest_*`, `document_inventory`, `title_block` and `drawing_callouts` stay in:
   - `coverage.SWEEP_TOOLS` and `cited_pages.SWEEP_TOOLS` / `OPENED_HOWS`;
   - the reading-tool list in `cited_documents.py:780-786`;
   - the "sweep" `how` in `review_eval/checks.py`.

   They classify tool calls in a conversation's activity record, which can include conversations recorded before this branch. No tool of those names can be called now. `webapp/core.CACHE_DIRS` keeps `digest` for the same reason: a `digest/` folder an older conversation kept still is not a download card.

## 5. Tests

venv as in `COMMON_MODULES.md` (Python 3.12, `-e packages/aec-agent-core -e .`, planlens 0.14.0 and geotech-references 1.5.0 from PyPI). Each result was judged on pytest's own exit code.

**Before the change:** unmodified master, in a separate worktree. `funhouse_agent/deep/tests`, `funhouse_agent/review_eval`, `funhouse_agent/review_digest`, `test_vision_images`, `test_vision_turned`, `test_cited_pages` and `webapp/tests`: **2,728 passed, 12 skipped, exit 0.**

**After the change:**

| Run | Result | Exit |
|---|---|---|
| The same set (minus the deleted `review_digest` tests) | 2,531 passed, 9 skipped | 0 |
| `webapp/tests/test_page_fingerprints.py` + `test_public_text.py` | 18 passed, 0 skipped | 0 |
| The rest of `funhouse_agent/` (`--ignore` deep/tests and review_eval) | 4,257 passed, 32 skipped, **16 failed** | 1 |

The 16 failures fail identically on unmodified master: same 16 node ids, 16 failed, run in the master worktree. They are reference-library and calc-package tests that differ under geotech-references 1.5.0 from PyPI (`test_citations`, `test_fhwa_soils_foundations_reference` ×2, `test_reference_adapters` REF-61/62 ×7, `test_calc_package_figure_conventions`, `test_defaults_used[profile_figure|plot_data]`, and others in that list). This branch does not touch them.

## 6. Release note

None for users. Every removed switch was off by default, and the removed arms built agents that only a suite run or a developer setting reached. Both pages build exactly what they built before; the recorded fingerprints prove it.
- **Notebooks:** a notebook that sets one of the removed `GEOTECH_*` variables now gets the default page, silently. One that names a removed arm gets "unknown arm".
- **Library callers:**
  - `webapp.core.attachment_note` no longer takes `review=`;
  - `funhouse_agent.review_prompt.build_document_review_prompt` no longer takes `lean=`;
  - `funhouse_agent.cited_pages.unopened_citations` is gone;
  - `funhouse_agent.review_findings`, `funhouse_agent.review_digest` and the four `deep` tool modules are gone.

  A grep of the repo found no other caller.
