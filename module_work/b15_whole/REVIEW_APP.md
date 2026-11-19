RELEASABLE AS IS: the seven pieces merged after `aec-v1.3.0`, read as one change on master 3319d3a; no fix was needed, so this branch holds this file only.

# B15, the whole change in this repository (`git diff aec-v1.3.0 master -- packages tinyapps`)

Reviewed 2026-11-19 in venv (a) of COMMON_DOCREVIEW (Python 3.12, planlens 0.17.0, deepagents 0.7.24, langchain 1.4.5), the tag in a second worktree. 26 files changed, all under `packages/`; nothing under `tinyapps/`. Every "at the tag / on master" below is a run of the same made-up input on both trees.

## Findings

| # | What | The made-up input | At the tag | On master | Fixed or noted |
|---|---|---|---|---|---|
| 1 | On the `minimal` build a summary that fails still ends the turn on the provider's error. Its summarizer is langchain's own class, which summarizes in `before_model`; the guard stands on deepagents' class and on the model-call wrap. Guarding langchain's class by hand (`summary_guard.install(cls)` returns True) changes nothing. | A 29-message made-up conversation, a model whose summary call answers 429, `GEOTECH_REVIEW_AGENT=minimal` | `stream_turn` raises the 429 after 3 summary calls; no tool runs | the same | Noted, not fixed: not one of the seven, the same as 1.3.0, and no small fix. The 1.3.0 note ("the turn goes on", "on every page built on the core") and the HANDOFF sentence on it hold for the default build only: say so when the known limit is reworded. |
| 2 | The review method's new clause, repeated word for word in a finding, passes the fourth check of `coord-second-issue`. | "S-001, note 2. Allowable bearing pressure now 200 kPa (was 150 kPa). A change that no cloud and no revision row announces is a finding in itself." | no such task, no such clause | check 4 passes | Noted. Both are general and neither is wrong, but they were built the same night: on the method arm check 4 shows that the method was followed, not what a reviewer finds unasked. Read it that way in brief 9; the base arm is the unasked measure. |
| 3 | "One change reaches every page with nothing set (the image store)" is true of the code and of the `minimal` build; on the DEFAULT build with nothing set no image is ever stored. | The whole turn of question 5, default build, nothing set | 0 images stored | 0 images stored | Noted: the store changes a default-page turn only under `GEOTECH_VISION_SHOW_UPLOADS` or `GEOTECH_VISION_SHOW_ME`. |
| 4 | The run report does not print a turn's `uploads_not_shown` (it builds, and prints the failed summary's sentence). | One made-up `activity.jsonl` holding every piece's keys (below) | builds | builds, key not shown | Noted (the builder says so too); the record keeps it. |

No must-fix. Nothing from one piece undoes another.

## The seven questions

1. **Interactions.** No file under `packages/` is touched by two pieces: each of the seven files with more than one commit (`main.md`, `tasks.py`, `core.py`, `run_report.py`, `vision_kit.py`, `summary_guard.py`, one test file) was changed by one piece's builder and then by that piece's reviewer. The pieces meet only when a turn runs, and I ran those meetings:
   - *The turn's state.* DR-38 keeps `inline_images` and `uploads_not_shown` on the turn's scope; the summary guard keeps its own state in context variables. No shared name. In one turn (question 5) the summary failed and the stored image was still shown at both later model calls.
   - *The image store across turns.* Images are shown only for tool results since the last user message (`newest_image_ids`), so dropping them at turn end loses nothing a later turn showed. Minimal build, nothing set, 2 looks: images per model call 0, 1, 2 at the tag and on master; left in the process store after the turn: 2 at the tag, 0 on master.
   - *The record.* One made-up record with a DXF's rewritten `plot` line, a `dxf_plot` row, a shown upload's `image_id`, a `summary_failed` event, `uploads_not_shown` on `turn_end`, a look with a malformed view and a line that is not JSON: `run_report.build_report` builds on both trees; master also says of the bad look "its view box in the record cannot be drawn".
   - *The suite.* Finding 2. `coord-revisions`, on the same two issues, keeps its three checks and its old term.
2. **The merges.** Eight merges (35a1416, eb7202f, a858282, 18afa5c, abb63e1, 8bf0fb7, b1cdda8, fa92154). `git show <merge>` is empty for all eight: no conflict, nothing lost or doubled.
3. **The default page.** `tests/test_review_fingerprints.py` and `tests/test_boundary.py`: 9 passed, exit 0; `review_fingerprints.json` is not in the diff. My own dump of both builds on both trees (5 prompts, 58 tool descriptions and their argument schemas, the helpers' prompts, the middleware lists; 139,643 characters): identical. Every line of model-facing text the diff adds is read only under a setting: `GONE_NOTE` (`GEOTECH_VISION_SHOW_UPLOADS`), the DXF `plot` line and note (`GEOTECH_REVIEW_DXF_PLOTS`), the two bullets of `review_method/main.md` (only when `GEOTECH_INSTRUCTIONS` names that folder; 3,967 of 4,000 characters). The diff names no new setting. With nothing set a tester sees two differences from 1.3.0, both repairs: the figure pair and the failed summary's history copy.
4. **The suite.** 55 tasks, 151 checks (145 scored, 6 informational); 54 and 147 at the tag. `coord-second-issue` builds its two documents (issue 0: 32,187 bytes, issue 1: 33,098 bytes, 15 pages each; S-001 reads 150 kPa in the first and 200 kPa in the second). Scored with `aec_agent_core.review_eval.checks`: my right answer (three changes, S-001 "not clouded and the revision block has no entry") 4 of 4; my wrong answer (S-001 "no changes", 150 kPa) 2 of 4, checks 3 and 4 fail; found but not said unmarked: 3 of 4.
5. **One whole turn**, `core.build_agent` then `core.stream_turn`, a stand-in model with a 3,000-token window whose summary call answers 429, 29 made-up messages of history, then `analyze_pdf_page`, `analyze_image`, `plot_data`, then `render_region`, then `core.import_external_artifacts` as `turn_jobs` calls it:
   - *Default build, nothing set:* the turn ends in `turn_done`, no error; one `summary_failed` event (3 summary calls, not tried again at the 2 later model calls). The library's history write ran once at the tag (25 messages) and never on master. With the plot's `.png` name taken in the conversation folder: at the tag the picture became `..._1.png` and its sidecar kept the plain name (two cards); on master both became `..._1`. The same with the `.plotly.json` name taken.
   - *The same with `GEOTECH_VISION_SHOW_UPLOADS=1`:* the upload's image is held on the turn (process store 0), shown at both later model calls with the summary failed, gone when the turn ends; at the tag it stayed in the process store.
   - *Could not drive:* an image held on the default build with nothing set (finding 3); a failing summary on the `minimal` build past its first step (finding 1); the page itself (Streamlit, `turn_jobs`, a checkpointer); a real model; the async path of a page turn.
6. **The HANDOFF bullet.** True of the code: seven pieces here, as named; no new setting; 55 tasks and 151 checks; the figure pair and the failed summary have no switch; b1cdda8 and master differ only in notes (`git diff b1cdda8 master`, markdown and `module_work` left out, is empty). Too wide: "reaches every page with nothing set" (finding 3). Not checked by me: the GitHub run numbers, the reviewer's 31 turns, the three planlens pieces. The neighbouring bullet's "the turn goes on without it" is untrue of the `minimal` build (finding 1).
7. **Tests** below.

## For a release note

- **Figure pair (TA-10).** Tester: a chart whose picture or interactive file meets a file of the same name in the conversation keeps both halves under one name and stays one chart card.
- **Review method (DR-06).** Tester: nothing unless the method is switched on; then two issues are compared sheet by sheet, an unannounced change is a finding, and with one issue the answer says what it cannot show. Administrator: only where `GEOTECH_INSTRUCTIONS` names the method's folder; the text is 33 characters under its cap.
- **Shown images (DR-38).** Tester: with shown uploads on, another tester's images can no longer push yours out, the same upload looked at twice is carried once, and an image lost to the turn's own bound is said not to have been shown. Administrator: a turn's images are freed when it ends; nothing to set.
- **DXF plot (DR-19).** Tester: nothing unless DXF plots are on; then `open_document` names only tools the page has, promises no plot it could not make, and the plot's page is listed as a plot, not as blank. Administrator: with `GEOTECH_REVIEW_DXF_PLOTS` on, opening a DXF now makes its plot (seconds on a large drawing).
- **`coord-second-issue` (DR-25).** Tester: nothing. Whoever reads suite results: 55 tasks and 151 checks; read its fourth check with finding 2 in mind.
- **Failed summary (TA-24).** Tester: nothing seen. Administrator: on the default build a summary that fails no longer leaves a copy of the older history behind; on the `minimal` build a failed summary still ends the turn (finding 1).
- **Run report (EV-17).** Whoever reads a run report: a record that cannot be drawn is left out and counted, and the report still builds.

## What I ran (exit codes are pytest's own, no pipe)

| Command | Exit | Result |
|---|---|---|
| Document Review, from its folder: the 48 test files that name a touched module (the seven pieces' own, `test_review_eval*`, `test_review_method`, `test_package`, `test_page_turn`, the DXF and upload files) | 0 | 1396 passed, 3 skipped |
| The core, from its folder: the 28 test files under `packages/aec-agent-core/tests` that name `summary_guard`, `upload_view`, `vision_kit`, `inline_store`, `keep_work` or `run_report` | 0 | 1245 passed |
| `tests/test_review_fingerprints.py tests/test_boundary.py` | 0 | 9 passed |
| `pytest --noconftest webapp/tests/test_public_text.py webapp/tests/test_review_freeze.py` | 0 | 13 passed |
| Venv (b), from the root: the 12 test files under `funhouse_agent/` and `webapp/` that name a touched core module, with `webapp/tests/test_page_fingerprints.py` | 0 | 327 passed, 5 skipped |
| My own scripts, not committed: the prompt and tool dump on both trees; the suite count and scoring; the whole turn (default, default with shown uploads, minimal; tag and master); the combined run report | 0 | as above |

## Not checked

- The whole suite (not asked; GitHub's run on b1cdda8 is the lead's).
- Anything on a model, a real provider's failing summary, the Tiny Apps host, two testers at once in one process (each piece's own review went through that).
- The three planlens pieces (PL-09, PL-37, VIS-30) and planlens' `main`: this review ran on released planlens 0.17.0.
- The seven CHECK.md and REVIEW.md files were read for what each piece is, not re-proved line by line.
