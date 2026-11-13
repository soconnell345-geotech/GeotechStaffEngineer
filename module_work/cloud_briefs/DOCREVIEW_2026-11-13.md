# Cloud helper briefs: Document Review builds (2026-11-13)

Read `COMMON_DOCREVIEW.md` (beside this file) first: its rules bind every task here. This file adds what is common to this round, then has one section per task. Your routine's prompt names your section; do that section only.

## This round

The owner approved the next round of Document Review builds and wants them run in the cloud. They come from two reads: the deployment review of Document Review (2026-11-10, `module_work/docreview_review_2026-11-10/REVIEW.md` section 2) and the full read of Foundry brief 7's 84 suite runs (2026-11-12, `module_work/review_eval_results/2026-11-12_foundry_brief7/TRACE_REVIEW.md` sections 7, 8 and 10, findings F-6 to F-15). Nothing here calls a model. Where an item needs a live model, a live SharePoint or the testers' host, write in CHECK.md, under "For the lead", exactly what to check and how.

- **Confirm first.** Names and line numbers are from app master 1b9b499 and planlens main 0f87fa6 and may have moved. Open each cited function and confirm the defect before changing anything. If it is already fixed, or the item is wrong, say so in CHECK.md and leave the code.
- **Repairs and settings.** Each section says which pieces are repairs (no setting) and which are new behaviour (behind a setting that is off unless set): COMMON_DOCREVIEW, "No switch on by default".
- **Two waves.** Tasks A to G and P1 run at the same time. P2 starts after P1 is merged into planlens main.
- **Releases.** Nothing here reaches testers until the lead releases the core and Document Review (one unit, once task G's work is in) and, for P1 and P2, planlens with a raised pin. Write each release note for that release.

### Who edits what

Tasks that run at the same time never edit the same file. Edit only the files listed for your task, plus new files of your own (tests named for your item, your CHECK.md). A file listed for another task is not yours even if your change seems to need it: stop short of that part and say so in CHECK.md.

DR = `packages/aec-document-review/aec_document_review/`; core = `packages/aec-agent-core/aec_agent_core/`.

| Task | Repository | May edit |
|---|---|---|
| A | app | DR `tinyapps_keys.py`, `identity.py`, `profiles.py`; core `vision_probe.py` |
| B | app | core `deep/vision_engine.py`, `palantir_sdk_engine.py`, `tinyapps_engine.py`, `engine_config.py`, `deep/busy_retry.py`, `activity_log.py`, `review_eval/runner.py` |
| C | app | DR `turn_jobs.py`, `core.py`, `app.py`, `budget_panel.py`, `review_agent.py`, `review_build.py`, `stream_format.py`, `turn_record.py`; core `deep/keep_work.py`, `deep/limits.py`, `deep/tool_guards.py`, `deep/assembly.py`, `error_text.py`; GSE `webapp/core.py` (one constant); the two fingerprint files (a commit of their own) |
| D | app | DR `sharepoint_store.py`, `graph_sharepoint.py`, `sharepoint_tools.py`, `sharepoint_log.py`; `docs/SHAREPOINT_SETUP.md` |
| E | app | core `vision_tools.py`, `turn_question.py` |
| F | app | core `docx_renderer.py`, `office_text.py` |
| G | app | `.github/workflows/publish.yml`, `publish-core.yml`, `publish-document-review.yml` (and a new workflow file only if your design needs one); both packages' `pyproject.toml`, their `README.md` and the version line of their `__init__.py`; the root `pyproject.toml` (the core pin and its comment only); `tinyapps/TINYAPPS.md` and `tinyapps/wrapper_repo/`; `docs/FOUNDRY.md` |
| P1 | planlens | `planlens/tools/toolkit.py`, `planlens/tools/specs.py`, `planlens/review/specs.py`, `planlens/document/document.py`, `planlens/document/quantities.py`, `planlens/review/tests/app_tool_specs.json`, `CHANGELOG.md` (its Unreleased section), the `DESIGN.md` of what you change |
| P2, after P1 | planlens | P1's files, plus `planlens/tools/formatting.py`, `planlens/document/structure.py`, `planlens/document/sheets.py`, `planlens/review/marked_copy.py`, `planlens/review/tools.py`, `planlens/review/host.py` |

Existing test files are not split between tasks: edit one only where your change makes one of its assertions untrue (COMMON_DOCREVIEW, "Tests"). If two branches touch the same test file in different places, the lead merges them.

**No task edits**, besides COMMON_DOCREVIEW's list:
- the files another cloud session's branch, `claude/gm-316-317-warnings-allowed-values`, is changing: core `method_call.py`, `hooks.py`, `call_warnings.py` (new there), `charts/plot_tool.py`; `funhouse_agent/__init__.py`, `funhouse_agent/core_hooks.py` and GSE's adapters. Before you start, run `git fetch origin claude/gm-316-317-warnings-allowed-values && git diff --name-only origin/master...FETCH_HEAD` and leave every file it lists alone;
- GSE's copies of the page under `webapp/` (except task C's one constant in `webapp/core.py`), and `funhouse_agent/`;
- `docs/user_manual/` (the geotech page's manual).

**Fingerprints across tasks.** Only task C is expected to move the page fingerprints. Any other task that moves them re-records in a last commit of its own and says so in CHECK.md and in its final message; the lead merges the branches in turn and re-records once on master where two branches moved them.

## Task A: testers told apart (TA-39)

Branch `claude/ta-39-testers-apart`. CHECK at `module_work/ta39/CHECK.md`. App repository. Model: Opus. Venvs (a) and (b).

Closes BACKLOG TA-39, except keying the running-turn table and the conversation-folder lookup by folder and id, which stays with INF-71 (it waits on the owner). Evidence: REVIEW.md section 2.2.

**Confirm first:**
- `tinyapps_keys.tester_suffix` turns every run of characters other than a letter or digit into one hyphen, so `corp\j.doe`, `corp\j-doe` and `corp\j_doe` all ask Key Vault for `PROMPTER-API-KEY--CORP-J-DOE`; `resolve_for` tries that per-tester secret before the exact-name `PROMPTER-KEYS` mapping.
- `identity.Identity.key`, the folder key (also the owner segment of the SharePoint folders, through `app.py`), keeps letters, digits, `.`, `_` and `-` and turns every other run into `_`: `o'brien` and `o_brien` share a folder, and a name with an accented letter can share one with another name. Since TA-36 a sign-in name may hold any character Windows allows. (`j.doe` and `j_doe` already get different folders.)
- `vision_probe.profile_for` holds a failed probe's fallback for `FAILED_HOLD_S` (120 s) under `_cache_key(model)`, which is the model's class, name and address and not its key. A probe that failed because one tester's key was refused (budget spent, no access) lowers every tester's image size on that model for two minutes.

**Change.** All three are repairs (a tester holding another's key or folder, or paying for another's spent key); no setting.
1. **Keys.** Two sign-in names that differ only in punctuation or accents can be given different keys, and no tester who has a key today loses it. The item offers two ways: the exact-name `PROMPTER-KEYS` mapping consulted first; or a short hash of the exact sign-in name in the per-tester secret's name, looked up first, with today's name as the fallback so existing secrets keep working. Choose, record the call, and say in the module docstring how a deployment provisions two such testers.
2. **Folders.** Two different sign-in names never share a folder. A name made only of letters, digits, `.`, `_` and `-` keeps today's key exactly, so no ordinary tester's folder or SharePoint mirror moves. A name holding any other character gets a short hash of its exact, case-folded spelling appended. The release note says whose folder moves (only such names) and where their earlier conversations are. Do not move an old folder for them: two people may have shared it.
3. **The probe.** When the probe failed because the caller's key was refused (budget spent, unauthorised, no access), keep no verdict for the model: that failure answers that caller only. A verdict about the model itself (a measured size, an image detail it refuses) stays shared, as now.

**Must not change:** every per-tester secret name that resolves today; the folder of any name in that safe alphabet; single-user hosts (`multi_user` false); the sharing of a successful probe.

**Tests.** New `packages/aec-document-review/tests/test_ta39_testers_apart.py`: for each kind of collision, two made-up names get different keys and folders; today's names keep theirs; the chosen key route serves the right key. New `packages/aec-agent-core/tests/test_ta39_probe_refusal.py`: a refused key leaves no held verdict for the next caller; a failure about the model still holds. Run DR `tests/test_tinyapps_keys.py`, `test_app_profiles.py`, `test_tinyapps_wiring.py`, `test_sharepoint_privacy.py` (venv a); `packages/aec-agent-core/tests/test_vision_probe_retry.py` and `funhouse_agent/tests/test_vision_probe.py` (venv b).

**For the lead:** `tinyapps/TINYAPPS.md` is task G's file this round, so write the text its key section needs into CHECK.md; a live check on the test host with two colliding sign-in names. GSE's own `webapp/identity.py` and `webapp/tinyapps_keys.py` are not changed: note whether any GSE host is multi-user.

## Task B: model calls, one ladder and every retry on the record (INF-75, VIS-31 part 2, TA-24)

Branch `claude/inf-75-model-call-record`. CHECK at `module_work/inf75/CHECK.md`. App repository. Model: Opus. Venvs (a) and (b).

Closes INF-75; VIS-31's part 2 (VIS-31 closes once task E's part 1 is merged too); and TA-24's structural part (what stays: the live check of the rollout gateway). Evidence: TRACE_REVIEW F-12 to F-15 (section 10) and sections 7 and 8; REVIEW.md section 2.1.

**Why together, and the order.** All three change how a model call is timed out, retried and recorded, in the same files, `deep/vision_engine.py` above all. Do them in this order, each in one commit or more, so the record exists before the limits move and the next suite run can measure both.

1. **INF-75 (1): a side call that times out is on the record and in the results table.**
   - Confirm: `vision_engine.invoke_side_call` gives a call up after `call_timeout_s()` and asks once more, with only a `log.warning`. `_invoke_once` leaves the abandoned call running on its thread and holding its vision slot, so the record keeps a `model_start` with no `model_end` (44 of them in brief 7's 84 records). `review_eval/runner.py`'s results table counts only `model_error` events as "failed calls" and only failed runs as "errors", so both read 0 even where three tool calls were refused.
   - Change: write an app event each time a side call times out, dispatched as `busy_retry._event` does and added to `activity_log.APP_EVENTS` and its docstring: the limit, the call's kind, page and tile (`current_side_call()`), the attempt, and whether it is asked again. If the abandoned call ends late, record that too, so a slot held past the limit shows. In `RESULTS.md` and `results.json`, count per run the timed-out side calls and the refused tool calls (read how the record marks a tool call that raised or was refused).
2. **INF-75 (2): an engine retry is on the record.**
   - Confirm: `palantir_sdk_engine.call_with_retries` retries a timed-out or dropped call with only a `log.info`.
   - Change: one app event per retry there (the kind, the attempt, the wait, the error's class and HTTP status where known), reaching the activity log from inside the model call; test that it arrives, with a fake engine. Leave the retry policy itself as it is (Foundry is not the testers' host; say so under "Noted, not fixed").
3. **INF-75 (3): the record shows what was offered and used.**
   - Confirm: `ActivityLogger._turn_facts` hashes the sorted tool names only (`tools_offered_sha`), so baseline and `show_me` runs look the same; the runner does not count looks made with `show_me=true`.
   - Change: per turn, also hash each offered tool's name, description and argument schema, in a new field beside `tools_offered_sha` so older records still read; count `show_me` looks per run in RESULTS, from the tool calls' arguments in the record.
4. **VIS-31 (2): the side-call limit set from the measured tail**, in a commit of its own after 1. `DEFAULT_TIMEOUT_S` is 180 s. In brief 7, 15,545 tile calls took 5 s at the median and under 22 s for 99 %; brief 5 (another deployment, 2026-10-08) had a 99th percentile of 74 s; the trace review suggests 60 s with the existing single retry. Choose and record the call, with both numbers. `GEOTECH_VISION_CALL_TIMEOUT_S` still overrides it; the primary agent's own calls keep the engine's timeout.
5. **TA-24: one ladder.** A repair: an unbounded wait.
   - Confirm: on the Prompter route (`tinyapps_engine`) the OpenAI client retries a busy or timed-out call `DEFAULT_MAX_RETRIES` (4) times at `REQUEST_TIMEOUT_S` (180 s) inside each try of the agent's busy ladder (`busy_retry.AgentBusyRetry`, five tries, which counts a timeout as busy) and inside the vision ladder, so one stuck call can hold a turn 15 to 20 minutes before the tester sees an error. `FairSlots.acquire` waits for a vision slot with no bound, while other testers' looks queue with no message.
   - Change: client retries at 0 or 1 by default on that route (the ladder already honours `retry-after`; `GEOTECH_PROMPTER_MAX_RETRIES` still overrides; choose and record the call); a timeout asked again at most once in all, client and ladder together; a bound on the wait for a vision slot, after which the look is reported as not read because the vision model is busy with other looks (one plain line through `describe_error`; the bound is a call).
   - Leave: the ladder's waits (15, 30, 60 and 120 s), the vision ladder's tries and `GEOTECH_TOKENS_PER_MINUTE`, which wait on the live check (does the gateway's 429 carry `retry-after` or `x-ratelimit` headers; which wait clears its window); deepagents' summarization call, made inside a before-model hook outside the ladder (`deep/assembly.py` is task C's file: noted); the vision probe's own retries (`vision_probe.py`, task A's).

**Must not change:** what a successful call returns; the busy ladder's bound per turn; the Foundry engine's retry policy; any tool description.

**Tests.** New `packages/aec-agent-core/tests/test_inf75_*.py` for the events, the limit and the ladder, and `packages/aec-document-review/tests/test_inf75_results_counts.py` for the table. Run core `test_vision_side_call_timeout_offline.py`, `test_vision_busy_and_fairness_offline.py`, `test_vision_inflight_offline.py`, `test_core_suite_and_records.py`; `funhouse_agent/deep/tests/test_busy_ladder_offline.py`, `test_foundry_rate_limit_offline.py`; `webapp/tests/test_palantir_sdk_engine.py`, `test_activity_log.py`, `test_activity_cache_tokens.py`, `test_agent_busy_routes.py`; `funhouse_agent/review_eval/tests/test_review_eval_offline.py` (venv b); DR `tests/test_agent_busy_routes.py`, `test_record_ev06.py`, `test_record_ev14.py`, `test_review_eval_offline.py`, `test_wave2b_turns.py` (venv a).

**For the lead:** the next suite run measures the new limit through the timeout events (brief 8's wheels predate this branch); the live check above; whether to change the Foundry engine's retries once their events are seen.

## Task C: a turn the tester can stop, ceilings over the whole turn, a spend line (TA-37, TA-38, TA-05, DR-43 part 1)

Branch `claude/ta-37-stop-and-turn-ceilings`. CHECK at `module_work/ta37/CHECK.md`. App repository. Model: Opus. Venvs (a) and (b).

Closes TA-37; TA-38 except the count and ceiling of image calls (below); TA-05; and DR-43's part 1 (its part 2 is a measurement the lead makes). Evidence: REVIEW.md section 2.1; TRACE_REVIEW F-11.

**Why together.** All of it lives in the turn: the pass loop in DR `core.stream_turn`, the turn job in `turn_jobs.py`, the page's sidebar in `app.py`, and DR-43's constant sits in the same `core.py`.

1. **TA-37: a Stop button.** New behaviour: behind a setting, off unless set (name it, for example `GEOTECH_STOP_BUTTON`).
   - Confirm: nothing stops a turn once sent (`turn_jobs.TurnJob`, `start_turn_job`, `_run_turn_job`; the running-turn view in `app.py`).
   - Change: with the setting on, a Stop button shows while the caller's own turn runs and sets a flag on that turn's job. The flag is checked before each model call of every agent in the turn (the primary and each helper), before each wait (the busy ladder's, through `keep_work.TurnScope.spend_busy`; the helper queue's) and before each pass (auto-continue and the page check's pass in `core.stream_turn`). A stopped turn keeps its work the way an interrupted turn's is kept (`keep_work`; TA-19's note in `turn_jobs.py`), and its answer says it was stopped by the user and what was done. Put the checks in hooks the turn already has (`keep_work.HelperRecorder`, `limits.ModelCallBudgetMiddleware`, the pass loop), so that with the setting unset no middleware is added and the fingerprints do not move. Record the stop with `activity_log.append_event` (do not edit `activity_log.py`: it is task B's).
   - Noted, not this task's: a vision side call already running ends at its own limit (task B's files), and a wait already under way ends at its time.
2. **TA-38: ceilings over the whole turn.** Repairs: the budget is the tester's money. Confirm each, then fix it:
   - a pass that ends without reaching the model-call budget can still auto-continue with a fresh budget (`core.stream_turn`, the branch on `continuations < MAX_AUTO_CONTINUES`; TA-36 fixed only the pass the budget itself ended): count the turn's budget over all its passes;
   - the continuation calls after a cut-off reply (`tool_guards.OutputLimitGuard`) are outside the budget, and an error in one re-runs the long call and drops the text already paid for: count them, and on an error keep that text;
   - the forced last reply can come back empty, and the answer then reads "(no answer text)" (`turn_jobs.py`, `app.py`): say instead that the model-call budget ended and what the turn did, from what the record holds.
   - Leave: a count and a ceiling of image calls per turn (a page look is up to 17 image calls, so a 50-sheet set read sheet by sheet is about 850). It needs `vision_tools.py` and `deep/vision_engine.py`, tasks E's and B's files: write its design in CHECK.md for a follow-up.
3. **TA-05: a spend line without the Funhouse SDK.** New behaviour: behind a setting, off unless set.
   - Confirm: `budget_panel.py` reads only the Funhouse SDK's meter.
   - Change: with the setting on, the sidebar shows the caller's own spend this calendar month, estimated from the token counts in the caller's own conversations' records and priced by settings for input and output tokens (with a monthly budget if one is set), and labelled as an estimate. Nothing from another tester's folder is read. Where the SDK is present its line stays as it is.
4. **DR-43 part 1: dead `sandbox:` links.** A repair.
   - Confirm: DR `core.ATTACHMENT_NOTE`, the note that `write_docx`, `write_xlsx` and the other saves add (`make_save_fn` sets it as the save function's `saved_note`, through `attachment_note_for`), lacks the sentence `method_call`'s note has: "Tell the user the file name; a /tmp path or a sandbox: link would not reach them." In brief 7, 15 Sol runs linked a file as `sandbox:/mnt/data/...`.
   - Change: add that sentence to `ATTACHMENT_NOTE` in DR `core.py` and in GSE's `webapp/core.py` (not frozen; the geotech page's saves use it). Do not edit `method_call.py`.

**Must not change:** with both settings unset, the page and its fingerprints, apart from the repairs in 2 and 4; GSE's copies of the page apart from that one constant.

**Tests.** New `packages/aec-document-review/tests/test_ta37_stop.py`, `test_ta38_turn_ceilings.py`, `test_ta05_spend_line.py`, `test_dr43_attachment_note.py`. Run DR `tests/test_turn_jobs.py`, `test_page_turn.py`, `test_final_answer.py`, `test_kept_turn_work.py`, `test_interrupted_turn_files.py`, `test_budget_panel.py`, `test_core.py`, `test_working_files_note.py`, `test_wave2b_turns.py`, `test_app_smoke.py` (venv a); `funhouse_agent/deep/tests/test_keep_work_offline.py`, `test_deep_call_budget_offline.py`, `test_wave2b_guards_offline.py`, `webapp/tests/test_core.py`, `webapp/tests/test_kept_turn_work.py` (venv b); both fingerprint tests.

**For the lead:** turning either setting on for a tester round is the owner's call; the image-call ceiling once B and E are merged; GSE's copies of the turn.

## Task D: SharePoint saves off the turn and off the page (TA-40)

Branch `claude/ta-40-sharepoint-off-turn`. CHECK at `module_work/ta40/CHECK.md`. App repository. Model: Opus. Venv (a).

Closes TA-40, except its live check. Evidence: REVIEW.md section 2.3; BACKLOG TA-40 says what TA-36 already fixed. All seven pieces are repairs (one tester holding up the others; files failing or lost without a word); no setting. Build against a fake Graph, as the existing SharePoint tests do.

**Confirm each, then fix it:**
1. Saves after a turn run inside the turn, under the store's one process-wide lock (`SharePointStore._lock`, `mirror_conversation`, reached from `turn_jobs.py` through `sharepoint_store.save_after_change`). One tester's first mirror of a large set holds every other tester's finished turn at "running", and "Sync now" (`app.py`) can freeze the page. The Foundry backend already saves in the background (`SharePointStore.background`, `save_soon`): give the Graph route the same, and lock per conversation, not per process, so one conversation's mirror never waits on another's. Do not edit `turn_jobs.py` or `app.py` (task C's): `save_after_change` decides. If "Sync now" still needs a change in `app.py`, write it in CHECK.md.
2. Each upload walks the folder chain again (`graph_sharepoint._ensure_folder`; about 300 calls for a 40-file first mirror): upload by path and let Graph make the folders (`PUT .../root:/{path}:/content`, and `createUploadSession` by path for a large file).
3. A large upload has no retry of a failed piece: resume the upload session from `nextExpectedRanges`, a bounded number of times.
4. Whole files are held in memory with no size limit: read and send them from disk in pieces; skip a file above a size limit with a message that names it and the limit (the limit is a call).
5. A listing that fails reads as "nothing mirrored yet": report it as a failure, in the summary and in the sidebar's text.
6. A file name SharePoint refuses (`" * : < > ? / \ |`, a trailing dot or space, a reserved name) fails the same way at every sync: map it to an accepted name the same way every time (kept in the manifest), or report it once and skip it; never fail it again without a word.
7. The waits TA-36 added for a throttled service are taken under the process-wide lock: after 1, they hold only their own conversation.

**Must not change:** where a conversation's mirror lives (folder names, and a manifest that older versions still read); restoring existing mirrors; the Foundry backend's behaviour; the TA-35 and TA-36 privacy rules (a session never shows another conversation's record or link).

**Tests.** New `packages/aec-document-review/tests/test_ta40_*.py`: two conversations mirrored at once do not wait on each other; a 40-file first mirror walks no folders; a failed piece is resumed; a large file goes in pieces; a failed listing reads as a failure; a refused name does not fail twice. Run DR `tests/test_sharepoint_store.py`, `test_sharepoint_tools.py`, `test_sharepoint_privacy.py`, `test_sharepoint_links_and_failures.py`, `test_sharepoint_ta35.py`, `test_graph_sharepoint_setup.py`, `test_past_conversations_pages.py`, `test_foundry_host.py`, `test_tinyapps_wiring.py`, `test_wave2a_bookkeeping.py`, `test_wave2c_fixes.py`.

**For the lead:** the steps of a live mirror on a real tenant (a 40-file first mirror, a file of several hundred MB, a refused name, two testers at once). GSE's copies (`webapp/sharepoint_store.py`, `webapp/graph_sharepoint.py`) are not changed.

## Task E: what a page reading is told (VIS-31 part 1, VIS-30 part 1)

Branch `claude/vis-31-one-image-readings`. CHECK at `module_work/vis31/CHECK.md`. App repository. Model: Sonnet. Venvs (a) and (b).

Closes VIS-31's part 1 (VIS-31 closes once task B's part 2 is merged too) and VIS-30's first part (what stays: tiling by the sheet's own regions). Evidence: TRACE_REVIEW F-8 (section 10) and section 7; REVIEW.md section 2.5. Both pieces are repairs (a wrong reading, a warning dropped wrongly); no setting.

1. **VIS-31 (1): a reading says it saw one image.**
   - Confirm: the page and tile prompts (core `vision_tools._tile_jobs` for pages, `_image_tile_jobs` for pictures, `_vision_prompt`, `_overview_prompt`) and `turn_question.with_question` add only "Report only what is IN this tile" and the user's question; nothing tells a reading that it cannot compare with an image it was not given. In brief 7 a reading asked what changed against the other issue answered "None identifiable in this tile", and two agents took the sheet as unchanged.
   - Change: every reading that goes back to the main model (`turn_question.QUESTION_KINDS`), whether or not a user question is bound, is told in one short constant that it sees one image (one page, region or tile); asked to compare with another issue, sheet or image, it transcribes what is in this one and says that the comparison needs the other image. Add it in one place (for example `ask_vision`). Plain words about what the call can see, not a correction of a model's habit.
2. **VIS-30 (1): the legibility note kept while the tiles are still too small to read.**
   - Confirm: `_read_pdf_page` removes the result's `legibility` note whenever the page was read in tiles (`out.pop("legibility", None)`), although on a 36 x 48 in sheet a 4 x 4 tile still shows 4.3 pt lettering at about 9 px, under `LEGIBLE_TEXT_PX` (12 px; planlens `review/frames.py`, re-exported by core `vision_view`).
   - Change: keep a note while any tile's lettering is under that line. Tile rows carry `text_px` where the text layer gives one; on a sheet whose lettering is not in its text layer, use the assumed CAD lettering size (`ASSUMED_CAD_LETTERING_PT`, 4.3 pt) as planlens' `tile_grid` does. The note names those tiles, says their small lettering may be misread, and says to zoom with `render_region` before taking a dimension or keynote from them. Drop it only when every tile reads at or above the line. Check whether any other tiled path drops the note the same way.
   - Leave: tiling by the sheet's own regions (title block, notes, schedules, legend, each view) with a second level, which needs a design and a measurement; `MAX_TILES_PER_SIDE` (it is planlens').

**Must not change:** tool names, descriptions and argument schemas (if a fingerprint test fails, you changed one: undo it); the answers' shapes; how many tiles a page gets.

**Tests.** New `packages/aec-agent-core/tests/test_vis31_one_image.py` and `test_vis30_tile_legibility.py`, with a fake engine that records the prompts it is sent (see `test_page_reads_offline.py`). Run `funhouse_agent/tests/test_turn_question.py`, `test_turn_question_isolation.py`, `packages/aec-agent-core/tests/test_page_reads_offline.py` and every `funhouse_agent/tests/test_vision_*.py` that mentions `legibility` or the tile prompt (venv b); both fingerprint tests.

**For the lead:** VIS-30's rest (region tiling) needs a design and a measurement on E-size sets; the next run of the suite's revision task shows whether the one-image line changes what agents conclude.

## Task F: what the file tools lose (DR-41)

Branch `claude/dr-41-file-tools-say`. CHECK at `module_work/dr41/CHECK.md`. App repository. Model: Sonnet. Venv (b), and (a) for the Document Review tests.

Closes DR-41 except Excel-out formulas. Evidence: REVIEW.md section 2.5. All repairs (a file's content printed wrong or lost without a word); no setting.

**Confirm each, then fix it** (core `docx_renderer.py` writes Word and, in `docx_to_markdown`, reads it back; core `office_text.py` reads Word, through that function, and Excel):
1. **Word out.** `<sup>` and `<sub>` are printed as typed, "12 m<sup>2</sup>" (`_add_inline`): write superscript and subscript runs. Strike-through stays as tildes: write struck-through text. A tall image is scaled on width only (`_picture`) and runs off the page: fit it within the text width and the page's text height.
2. **Excel in** (`_xlsx`). A label merged over several rows reads as empty on every row but the first, so a schedule grouped by level loses its groups: give every row (and column) of a merged range the merged cell's value, and say in the table's note which ranges were merged.
3. **Word in** (`office_text._docx`, `docx_renderer.docx_to_markdown`). Comments, footnotes and endnotes, headers and footers, and text boxes are not read, and nothing says so: read them, each under its own heading or marked in place (a comment with its author and the text it is on). Where something cannot be read, say what is there and that it was not read.
- Leave: Excel-out formulas. Since TA-36 every cell that opens with "=" is written as text, and the tool says so; live totals would need a measurement with the suite.

**Must not change:** tool names, descriptions and argument schemas, which live in `vision_tools.py` and `deep/vision_kit.py` (not yours: if a description must change to stay true, write the new text in CHECK.md); the shape of `read_office`'s return value.

**Tests.** New `packages/aec-agent-core/tests/test_dr41_office_files.py`, building made-up .docx and .xlsx files inside the test (python-docx, openpyxl). Run `calc_package/tests/test_docx_renderer.py`, `calc_package/tests/test_docx_roundtrip.py`, `funhouse_agent/tests/test_read_office_files.py`, `packages/aec-agent-core/tests/test_saved_files_and_settings.py` (venv b); DR `tests/test_page_turn.py`, `test_review_binding_offline.py` (venv a). The geotech calc packages' Word output changes too wherever it holds `<sup>`: the release note says so.

## Task G: one release unit, prepared (INF-76)

Branch `claude/inf-76-one-release-unit`. CHECK at `module_work/inf76/CHECK.md`. App repository. Model: Opus. Venvs (a) and (b).

Prepares INF-76; the item stays open until the owner's word and the release itself. No code moves, and nothing changes for testers.

**Confirm first:** `aec-agent-core` publishes on a `core-v<version>` tag (`publish-core.yml`), and `aec-document-review` on a `document-review-v<version>` tag (`publish-document-review.yml`, which refuses until its exact pins are on PyPI). The core is 1.0.1; Document Review is 0.1.1 and pins `aec-agent-core==1.0.1` and `planlens==0.15.1`. GSE pins `aec-agent-core==1.0.1` (root `pyproject.toml`), and its `publish.yml` checks only that the pinned core is on PyPI. The pins are repeated in `tinyapps/wrapper_repo/packages.txt`, its `README.md` and `tinyapps/TINYAPPS.md`, and checked by `packages/aec-document-review/tests/test_package.py`, `webapp/tests/test_wheels.py` and `webapp/tests/test_tinyapps_wrapper.py`.

**Change:**
1. **One version, one tag.** Both packages carry the same version, and one tag pattern publishes both wheels, the core first. A guard refuses the tag unless both versions equal it and Document Review pins the core at exactly that version. PyPI's trusted publishing names the workflow file: aec-agent-core trusts `publish-core.yml` and aec-document-review trusts `publish-document-review.yml`. Prefer a design that needs no change on pypi.org, for example both existing workflows started by the one tag, the second waiting (with a bound) until the core of that version is on PyPI. A design that does need a change there lists it in CHECK.md as an owner action. Pick a tag pattern that GSE's `v*` does not match.
2. **The core a release goes out with.** INF-76's note of 2026-11-12: a GSE or Document Review release could go out pinned to a core that lacks a module both import (`foundry_dataset_store`), because the guard checks only that the pinned version exists. Add a guard to the release workflows that refuses when `packages/aec-agent-core/` has changed since the tag of the pinned core version.
3. **Tests and documents follow.** A new test that the two versions are equal, that every pin of the core (GSE, Document Review, the wrapper's files) names that version, and that the workflows' tag patterns agree; parse every workflow you change as YAML there. Update the existing tests and every document that states the pins or the release order (`tinyapps/`, `docs/FOUNDRY.md`, both packages' READMEs, the comments in the `pyproject.toml` files). The release steps in `CLAUDE.md` and `webapp/tests/test_docs_currency.py` are the lead's: write the proposed `CLAUDE.md` text in CHECK.md.
4. **The version bump, last and alone.** In the last commit, and only there, set both packages to one new version: above today's (1.0.1 and 0.1.1) and above anything on PyPI for either (`pip index versions aec-agent-core`, and the same for `aec-document-review`). Propose 1.1.0, the version brief 8's test core wheel carried as 1.1.0rc2. The lead may drop this commit and make the bump part of the release.

**Must not:** create a tag, publish, or start a publish workflow; edit `HANDOFF.md`, `CLAUDE.md` or the install-history rows of `docs/DATABRICKS_INSTALL.md`.

**Tests.** The new `webapp/tests/test_inf76_release_unit.py`; DR `tests/test_package.py` (venv a); in venv b, after `pip install build "setuptools>=68" wheel trove-classifiers`, `webapp/tests/test_wheels.py` and `webapp/tests/test_sdists.py` with `GEOTECH_REQUIRE_WHEEL_TEST=1` (they build the wheels), then `webapp/tests/test_tinyapps_wrapper.py` and `webapp/tests/test_docs_currency.py`.

**For the lead:** the owner's word; the tag; any pypi.org change; `CLAUDE.md`'s release steps; whether to keep the bump commit.

## Task P1: planlens, four results that mislead (PL-39)

Repository planlens. Branch `claude/pl-39-misleading-results`, from main. CHECK at `CHECK_PL39.md` in the repository root, in a last commit of its own. Model: Opus. Setup: COMMON_DOCREVIEW, "Setup: the planlens repository", with the app clone at `/tmp/gse-src`.

Closes PL-39. Evidence, read from the clone: TRACE_REVIEW F-6, F-7, F-9 and F-10 (section 10) and section 7. All four are repairs (a tool result that misleads any model that trusts it; (1) and (2) changed answers in brief 7), so no option gates them.

1. **`open_document`'s `pages_to_view`** (`tools/toolkit.py`, built from `VIEW_KINDS`) lists only pages classed as scans, figures, drawing sheets or forms, under the note "these pages are scans, figures or drawing sheets". On the suite's coordination sets it left out 22 of 40 and 9 of 15 titled drawing sheets, every callout destination among them. Make the list and its note true: either list every drawing-size page with a title block (what `Document.sheet_index()` finds is the yardstick, though the sheet tools stay opt-in), or say in the note that pages classed as text or mixed can be drawing sheets and that the list is not the set of sheets. `VIEW_KINDS` also drives `document_page_map`'s `look` and the search's list: keep the three consistent. Choose and record the call; the cost is part of it, since the page's rule is to look at every page in scope.
2. **Contact-sheet labels** (`document/document.py`, `Document.render_thumbnails`) are bare page numbers counted from 0, and the legend ("each thumbnail is labelled '<page> <kind>'") does not say which count, while the page tools take both `page` (from 0) and `pdf_page` (from 1). Print the PDF page number, counted from 1, or both, and say which in the legend.
3. **`search_document`** (`_tool_search_document`): (a) `pages_not_searchable_as_text` lists every page of those kinds, including pages with a full text layer and pages on which the same search just found hits: list only pages without a usable text layer, never one with a hit; (b) a pattern holding "|" without `regex=true` is matched literally, and the zero-hit hint offers `fuzzy`: when a literal pattern holds "|", say that `regex=true` reads it as "or".
4. **`find_quantities`** (`document/quantities.py`, `_RE_VALUE` with `_RANGE_TAIL`, read case-blind) takes the rebar callout "10-20M EACH WAY" as a length of 10 to 20 m: read a count and a bar size as one rebar callout, or not as a quantity at all; never as a range with a unit.

**Must not change:** tool names and argument schemas; any result, for a host that did not ask for the sheet tools, other than these four repairs; the sheet tools themselves (opt-in, PL-37).

**If a tool description changes:** change it in `planlens/review/specs.py` or `planlens/tools/specs.py`, and update `planlens/review/tests/app_tool_specs.json` for exactly that tool. That fixture is the app's capture ("Do not edit by hand"), and since 0.15 the app's specifications come from planlens, so re-capture it: build venv (b) of COMMON_DOCREVIEW from `/tmp/gse-src`, install your planlens checkout into it with `--no-deps --force-reinstall -e`, and run `python planlens/review/tests/capture_app_specs.py --app /tmp/gse-src` from your planlens checkout. Only if that venv cannot be built, edit that tool's entry by hand to the new text and say so in CHECK. Add one CHANGELOG "Unreleased" entry per repair.

**Tests.** New tests beside the code (`planlens/tools/tests/test_pl39_*.py`, `planlens/document/tests/test_pl39_*.py`), on made-up PDFs built in the test (see how `planlens/tools/tests/test_sheet_tools.py` builds a set). Run `planlens/tools/tests/test_toolkit.py`, `test_sheet_tools.py`, `test_structure_tools.py`; `planlens/document/tests/test_document.py`, `test_quantities.py`, `test_sheets.py`; `planlens/review/tests/test_host_specs.py`, `test_tools.py`, `test_end_to_end.py`. The app as a consumer: Document Review's `tests/test_review_binding_offline.py`, `test_review_sheet_tools_offline.py`, `test_review_eval_coordination_offline.py` and `test_review_fingerprints.py`.

**For the lead:** a planlens release, the app's pin raise, and Document Review's fingerprints (their planlens part) recorded under the new version.

## Task P2: planlens, sheet numbers as addresses and large files (PL-37 rest, PL-38)

Repository planlens. Starts after P1 is merged into planlens main; branch `claude/pl-37-38-sheets-and-large-files` from main at that point. CHECK at `CHECK_PL37.md` in the repository root, in a last commit of its own. Model: Opus. Setup as P1.

Closes PL-37's remaining parts, except a look at sheets that have no text layer (it needs a model: the host's job), and PL-38. Read, from the clone: BACKLOG PL-37 in full (its 2026-11-10 note lists what is built and what is open), PL-27 and PL-38; REVIEW.md section 2.5. In planlens: `planlens/document/DESIGN.md`, "Sheet index and cross-references".

1. **PL-37, the rest of the sheet tools.** New behaviour, only for a host that asks (`ReviewHost(toolkit_options={"sheet_tools": True})`); with it off, every tool list, specification and result stays byte-identical, as PL-37's first slice does.
   - The sheet index returned by `open_document`, within the result's size limit.
   - A sheet number accepted wherever a page is accepted, resolved through the index; a number not in the index is refused with the numbers it holds.
   - Each index row shows the sheet's scale where one is printed, and the stable markup ids of the marks on that sheet (PL-27).
2. **PL-37, `printed_numbers`** (`document/structure.py`). A repair, so it applies for every host: it reads only the top and bottom of a page, misses "SHEET NO. S-201A", the "DWG NO." forms and a label and value in separate cells, and returns "3 of 24" as a sheet number. Fix it by the rules `sheets.py` already uses, where they fit, and say where its results move.
3. **PL-38, measured first.** Build a made-up scanned set of about 300 MB inside the session (pages of image noise, in a scratch folder, never committed; the generator goes in CHECK, or as a small script under `planlens/testing/` if it fits there), and measure the peak memory and the time of opening it, of one `annotate_document` call, and of two calls on one copy. Then:
   - two annotate calls on one copy can lose marks (no lock on the output path, `review/marked_copy.py`): a repair, whatever the numbers;
   - near the result limit the viewer's page numbers are dropped from every row with no note: say so in the result (a repair);
   - each annotate call holds several whole copies of the file and re-reads it for the check, opening a document does work on every page, and the text caches (`Document._pages`) have no limit, for up to 8 documents in each of 8 live conversations: fix what the numbers show matters (a bound is a call), and report the numbers before and after.

**Must not change:** with the sheet tools off, every result except the repairs in 2 and 3; tool names; markup ids already handed out.

**Tests.** New tests beside the code, one file per piece. Run the test files of what you change, `planlens/tools/tests/test_sheet_tools.py`, `planlens/review/tests/test_sheet_tools.py`, `test_host_specs.py`, `test_end_to_end.py`, `test_marked_copy.py`, `planlens/document/tests/test_structure.py` and `test_sheets.py`. The app as a consumer: Document Review's `tests/test_review_sheet_tools_offline.py` and `test_review_binding_offline.py`.

**For the lead:** the planlens release and the app's pin; whether the app's `GEOTECH_REVIEW_SHEET_TOOLS` passes the new parts through once the pin is raised.

## Routines

One routine per task. A to G and P1 start at the same time; P2 starts once P1 is merged into planlens main. The lead merges this file to master before starting any of them, since each session reads it from its own clone.

| Section | Repository | Branch | Model | Must not run beside | One-line prompt |
|---|---|---|---|---|---|
| Task A: testers told apart (TA-39) | GeotechStaffEngineer | `claude/ta-39-testers-apart` | Opus | none | Read `module_work/cloud_briefs/COMMON_DOCREVIEW.md`, then `module_work/cloud_briefs/DOCREVIEW_2026-11-13.md`: its top part and the section "Task A: testers told apart (TA-39)". Both are binding. Do that section and nothing else. Branch `claude/ta-39-testers-apart`; push after each piece. |
| Task B: model calls, one ladder and every retry on the record (INF-75, VIS-31 part 2, TA-24) | GeotechStaffEngineer | `claude/inf-75-model-call-record` | Opus | none | Read `module_work/cloud_briefs/COMMON_DOCREVIEW.md`, then `module_work/cloud_briefs/DOCREVIEW_2026-11-13.md`: its top part and the section "Task B: model calls, one ladder and every retry on the record (INF-75, VIS-31 part 2, TA-24)". Both are binding. Do that section and nothing else. Branch `claude/inf-75-model-call-record`; push after each piece. |
| Task C: a turn the tester can stop, ceilings over the whole turn, a spend line (TA-37, TA-38, TA-05, DR-43 part 1) | GeotechStaffEngineer | `claude/ta-37-stop-and-turn-ceilings` | Opus | none | Read `module_work/cloud_briefs/COMMON_DOCREVIEW.md`, then `module_work/cloud_briefs/DOCREVIEW_2026-11-13.md`: its top part and the section "Task C: a turn the tester can stop, ceilings over the whole turn, a spend line (TA-37, TA-38, TA-05, DR-43 part 1)". Both are binding. Do that section and nothing else. Branch `claude/ta-37-stop-and-turn-ceilings`; push after each piece. |
| Task D: SharePoint saves off the turn and off the page (TA-40) | GeotechStaffEngineer | `claude/ta-40-sharepoint-off-turn` | Opus | none | Read `module_work/cloud_briefs/COMMON_DOCREVIEW.md`, then `module_work/cloud_briefs/DOCREVIEW_2026-11-13.md`: its top part and the section "Task D: SharePoint saves off the turn and off the page (TA-40)". Both are binding. Do that section and nothing else. Branch `claude/ta-40-sharepoint-off-turn`; push after each piece. |
| Task E: what a page reading is told (VIS-31 part 1, VIS-30 part 1) | GeotechStaffEngineer | `claude/vis-31-one-image-readings` | Sonnet | none | Read `module_work/cloud_briefs/COMMON_DOCREVIEW.md`, then `module_work/cloud_briefs/DOCREVIEW_2026-11-13.md`: its top part and the section "Task E: what a page reading is told (VIS-31 part 1, VIS-30 part 1)". Both are binding. Do that section and nothing else. Branch `claude/vis-31-one-image-readings`; push after each piece. |
| Task F: what the file tools lose (DR-41) | GeotechStaffEngineer | `claude/dr-41-file-tools-say` | Sonnet | none | Read `module_work/cloud_briefs/COMMON_DOCREVIEW.md`, then `module_work/cloud_briefs/DOCREVIEW_2026-11-13.md`: its top part and the section "Task F: what the file tools lose (DR-41)". Both are binding. Do that section and nothing else. Branch `claude/dr-41-file-tools-say`; push after each piece. |
| Task G: one release unit, prepared (INF-76) | GeotechStaffEngineer | `claude/inf-76-one-release-unit` | Opus | none | Read `module_work/cloud_briefs/COMMON_DOCREVIEW.md`, then `module_work/cloud_briefs/DOCREVIEW_2026-11-13.md`: its top part and the section "Task G: one release unit, prepared (INF-76)". Both are binding. Do that section and nothing else. Branch `claude/inf-76-one-release-unit`; push after each piece. |
| Task P1: planlens, four results that mislead (PL-39) | planlens | `claude/pl-39-misleading-results` | Opus | P2 | Clone the app repository read-only: `git clone --depth 1 https://github.com/soconnell345-geotech/GeotechStaffEngineer /tmp/gse-src`. Read `/tmp/gse-src/module_work/cloud_briefs/COMMON_DOCREVIEW.md`, then `/tmp/gse-src/module_work/cloud_briefs/DOCREVIEW_2026-11-13.md`: its top part and the section "Task P1: planlens, four results that mislead (PL-39)". Both are binding. Do that section and nothing else, in this planlens repository. Branch `claude/pl-39-misleading-results`; push after each piece. |
| Task P2: planlens, sheet numbers as addresses and large files (PL-37 rest, PL-38) | planlens | `claude/pl-37-38-sheets-and-large-files` | Opus | P1 (start after P1 is merged into main) | Clone the app repository read-only: `git clone --depth 1 https://github.com/soconnell345-geotech/GeotechStaffEngineer /tmp/gse-src`. Read `/tmp/gse-src/module_work/cloud_briefs/COMMON_DOCREVIEW.md`, then `/tmp/gse-src/module_work/cloud_briefs/DOCREVIEW_2026-11-13.md`: its top part and the section "Task P2: planlens, sheet numbers as addresses and large files (PL-37 rest, PL-38)". Both are binding. Do that section and nothing else, in this planlens repository. Branch `claude/pl-37-38-sheets-and-large-files`; push after each piece. |
