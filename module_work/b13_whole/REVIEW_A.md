# Release review A: the whole unreleased change after `aec-v1.2.0`

**Verdict: MERGE AFTER FIXES**, reviewed head `299fb72` (`claude/b13-integration`). The three fixes are on this branch (`claude/rv-b13-whole-a`), so merging this branch is the fixed state. No must-fix finding.

One thing about this branch's own history: fix 1 is in two commits. `591dcb2` holds only its test (the source file fell out of the index), and `7c00b1b` holds the change, so the new test fails on the three commits between them. A rewrite of the pushed branch was refused, so it stays that way; squash the two when merging if the history matters. This report is therefore in two commits as well, the last of them alone.

The question asked: with every new setting unset, is the page what 1.2.0 was, and do the settings notes say what the code does?

- **Unset, the default page is what 1.2.0 was**, apart from the stated repairs and the three things in findings 1, 4 and 5 (one fixed here, two for the release note).
- **The notes match the code** for the seven new switches. One setting was in none of the three notes (finding 3, fixed here).

Scope: `git diff aec-v1.2.0..299fb72 -- packages tinyapps`, non-test source read in three slices by helpers and checked by me where a finding came out of it. Items were not reviewed again one by one.

## 1. The settings

All switches are read through `review_flags._on`: true values `1`, `true`, `yes`, `on`; anything else, or unset, is off. The notes say `1`, which is true.

| Setting | Default | Where read | Cleared by the suite | Documented (README / TINYAPPS / .env.example) | Inert when unset |
|---|---|---|---|---|---|
| `GEOTECH_REVIEW_PLAN_TOOLS` (new) | off | `review_build.py:316`, at build | yes, `LATER_ENVS` | all three | yes |
| `GEOTECH_REVIEW_DRAWING_TOOLS` (new) | off | `review_build.py:318`, at build | yes, `LATER_ENVS` | all three | yes |
| `GEOTECH_REVIEW_DXF_PLOTS` (new) | off | `review_build.py:320`, at build | yes, `LATER_ENVS` | all three | yes |
| `GEOTECH_REVIEW_FIGURE_EXAMPLES` (new) | off | `review_build.py:363`, at build | yes, `LATER_ENVS` | all three | yes |
| `GEOTECH_REVIEW_LEAN_PROMPT` (new) | off | `review_build.py:434`, at build | yes, `LATER_ENVS` | all three | yes |
| `GEOTECH_VISION_SHOW_UPLOADS` (new) | off | `review_build.py:186`, at build | yes, `LATER_ENVS` | all three, with the text-only-model caution | yes |
| `GEOTECH_VISION_PIXEL_REQUESTS` (new) | off | `vision_tools.py` `_pixel_requests()`, at each look | yes, `LATER_ENVS` | all three | yes |
| `GEOTECH_PLOT_CODE_KERNEL_LAYER` (new) | off | `plot_box/__init__.py:158`, at each `run_plot_code` | no, on purpose (finding 7) | none on the reviewed head; all three after `74e608c` | yes |
| `GEOTECH_REVIEW_SHEET_TOOLS` (reading changed) | off | at build, as before | yes, `ALL_ENVS` | all three, with the new `sheets` / `sheet` arguments | yes |
| `GEOTECH_TURN_MAX_IMAGE_CALLS` (reading changed) | no ceiling | as before; now also named by `review_flags.describe()` | yes, `LATER_SETTINGS_ENVS` (new) | all three, unchanged | yes |

- The kernel layer takes other values: unset, empty, `0`, `false`, `no`, `off` are off; `1`, `true`, `yes`, `on` are on; `require` refuses to run code where the file rules cannot go up; any other value refuses to run code.
- On the pinned planlens 0.16.0: the plan tools exist (`has_tool` true), the drawing tool does not (the switch does nothing there, as the notes say), and the sheet arguments do not appear.
- The other `GEOTECH_` names the diff touches (`GEOTECH_VISION_SHOW_ME`, `GEOTECH_VISION_TILE_ANSWERS`, `GEOTECH_VISION_INLINE_KEEP`, `GEOTECH_VISION_TOOLS`) are mentioned in comments and notes only; the diff adds no read of them.

## 2. Findings

| # | Severity | Where | What | Shown by | Outcome |
|---|---|---|---|---|---|
| 1 | should-fix | `error_text.py:543` `_PREFIXED_KEY` | The new masking of a tool's error line took a document's file name for a key. Unset behaviour, not part of the stated repair: the model was no longer told which file could not be opened. | `no document named 'sk-101-foundation-plan.pdf'` came back as `no document named '….pdf'` | Fixed, `591dcb2` (the test) and `7c00b1b` (the change). A name followed by a document extension is kept; keys are masked as before. |
| 2 | should-fix | `aec-document-review/tests/test_record_masks_fx11_page.py`, fixture `helper` | The fixture patched `langchain.agents.create_agent`; where deepagents was first imported under the patch it kept the failing stand-in for the whole process. | `pytest tests/test_record_masks_fx11_page.py tests/test_auth_refusal.py`: 3 failed on the reviewed head. The whole folder passes only because of its alphabetical order. | Fixed, `9d484c9` (test file only). |
| 3 | should-fix | `packages/aec-document-review/README.md`, `tinyapps/TINYAPPS.md`, `tinyapps/wrapper_repo/.env.example` | `GEOTECH_PLOT_CODE_KERNEL_LAYER` was in none of the three notes. | grep on the reviewed head | Fixed, `74e608c` (wording). |
| 4 | note | `aec_document_review/core.py:1912` `_unique_dest` (TA-10) | Changes with everything unset and is not on the list of stated repairs: a second import of `plot.plotly.json` is now saved as `plot_1.plotly.json`, where 1.2.0 wrote `plot.plotly_1.json` and showed a plain file card. It is a defect repair. | reading | Not changed. Add it to the release note's repairs. |
| 5 | note | `turn_jobs.py:374`, `feedback.py:214`, `sharepoint_tools.py:654` and `:1545`, `minimal_agent.py:291`, `diagnostics.py:102`, `review_eval/checks.py:1680` | The masking reaches further than "the record's error text and a tool's error line": the turn error a tester sees, the feedback and SharePoint failure lines, the diagnostics report and the suite's saved results. | reading | Not changed; within the repair's purpose. The release note should say a tester's own error text is masked too. |
| 6 | note | `error_text.py` `_LOOSE_SECRET`, and the docstrings of `error_line` and `mask_error_secrets` | "A message with no secret comes back character for character" is not true: a value of 16 or more characters with a letter and a digit after `key`, `token`, `code`, `secret` or `sig` is masked. | `no such key: boring_log_page_12_table` gives `no such key: …`; `unknown view key: p3_r2_x100y200w300h400` gives `unknown view key: …` | Not fixed: narrowing it is a call about leak risk, the lead's. I found no tool on the page that words an error this way, and did not search planlens. |
| 7 | note | `review_flags.py` | The kernel layer is in no clearing list, and `describe()` does not name it, so a turn's `build.switches` does not show it (the tool result's `box.kernel_layer` does). | reading | Right as it is: clearing `require` between arms would run code without the layer a host asked for. Now said in the README row. |
| 8 | note | `packages/aec-document-review/README.md`, the `GEOTECH_REVIEW_DXF_PLOTS` row | It lacks the caution the other two notes carry: ezdxf and Pillow are needed, and a large drawing is not plotted. | reading | Not changed. |
| 9 | note | `vision_tools.py:2576` `_note_image_read` | Unset, every `analyze_image` now also copies the bytes, hashes them and writes an `image_read` event. The tool result is unchanged. | reading | Record only. |

## 3. Unset paths traced

| Path | Holds |
|---|---|
| The default page's build: prompt, tools, schemas, helpers, middleware | Yes: `review_default` has not moved in `review_fingerprints.json` since the tag, and the test passes. |
| `document_tools.host_for`: `toolkit_options` | Yes: the three new arguments default to false and nothing new is passed. `sent_scale` is set only where planlens has the field (after 0.16.0). |
| `vision_kit`: plan, drawing and sheet-address rewrites | Yes: reached only with names a switch supplies. |
| `analyze_image` with show-uploads off | Yes: `upload_view.active()` is false and the call goes on as before. |
| Pixel requests off on a PDF page | Yes: no prompt text and no result key added. |
| The image-pixel repair on a PDF | Does not reach it: `_image_frame` needs `source_kind == "image"`, `_kept_on_image` needs `image_px`. |
| The plotting box with the kernel layer unset | Yes: `kernel_layer.py` is not imported and the child's payload has no key. |
| The figure helper with its examples switch off | Yes: `examples=None` gives the core's set and the old description text. |
| The to-do middleware's place | Yes: it moves only for a `keep_last` middleware, which only the lean prompt adds. |
| The cited-pages and cited-documents checks, off | The record's events change; nothing reaches the model or the tester. |
| `read_reference_table` (REF-114) | The page fills no hook; a call answers "Unknown extended tool" as at 1.2.0. |
| The review suite | The 52 existing tasks and 7 documents hash the same at base and head; 2 tasks and 3 documents are new, so saved scores compare. |

Stated repairs confirmed on these paths: the summary guard (`assembly.py:493`), the masking, the image file's pixels, the minimal page's sentence and `page_count`, the helper's open items (two zooms on one page no longer share a key, `keep_work.py:1551`), the image ceiling between arms.

Record-only additions, not page behaviour: `model_end.cache_reported`, `turn_end.cache_reported_calls`, `calls_not_started`, `n_calls_not_started`, `n_calls_not_answered`, `summaries_skipped`, the events `summary_failed` and `image_read`, `build.helper_tools`, and `usage_metadata.input_token_details` present when a provider reports cache counts of 0 (`palantir_sdk_engine.py:1545`).

## 4. Fingerprints

- `git diff aec-v1.2.0..299fb72 -- '*fingerprints.json'`: one build moved, Document Review's `review_minimal` system prompt, 1,637 to 1,749 characters. It is in a commit of its own, `1f10f8c`, which says what moved and why (the sentence on an uploaded picture, VIS-26).
- `webapp/tests/page_fingerprints.json` has not changed since the tag: neither of the app's pages moved.
- Both tests pass on the reviewed head and on this branch. None of the three fixes moves one.

## 5. Tests

Venvs (a) and (b) of COMMON_DOCREVIEW, Python 3.12, planlens 0.16.0, deepagents 0.7.24. Judged by exit code.

| Command | Result |
|---|---|
| (a) `tests/test_boundary.py tests/test_review_fingerprints.py` | 9 passed |
| (a) `--noconftest webapp/tests/test_public_text.py webapp/tests/test_review_freeze.py`, on the head and before each commit | 13 passed |
| (a) the whole of `packages/aec-document-review/tests`, on the reviewed head | exit 0 (run quiet; count not kept) |
| (b) `webapp/tests/test_page_fingerprints.py` | 13 passed |
| (b) `webapp/tests/test_wheels.py` with it | 19 passed, 5 skipped in the two files |
| (b) `-n 4 packages/aec-agent-core/tests`, with fix 1 | 1142 passed, 1 skipped |
| (b) the four `test_record_masks_*` core files and the new test, with fix 1 | 110 passed |
| (a) the four `test_record_masks_*_page` files and `test_auth_refusal.py`, in that order | 6 failed, 18 passed on the head; 24 passed after `9d484c9` |
| (b) `webapp/tests/test_tinyapps_wrapper.py webapp/tests/test_inf76_release_unit.py`, after the notes fix | 51 passed |

New test: `packages/aec-agent-core/tests/test_rv_b13_file_names_kept.py`, 10 cases. Before fix 1, 5 fail (the file names) and 5 pass (keys still masked, by design); after it, 10 pass.

## 6. Could not check

- **A planlens after 0.16.0.** The sheet arguments, the drawing tool, the DXF plot (and the limits TINYAPPS gives for it: 16 MB, 50,000 entities, 30 s) and `sent_scale` were read, not run. Install planlens' main as in COMMON_DOCREVIEW (c) and run `tests/test_b14_wiring_offline.py` and the fingerprint test.
- **Any model.** No switch was measured; that is brief 9 (EV-45).
- **The kernel layer on a host.** Run `python -m aec_agent_core.plot_box.kernel_probe` on the App Service.
- **Behaviour with a switch on.** `lean_prompt.py`, `upload_view.py` and `figure_examples.py` were checked for their gate and for no import-time effect, not for what they do when on. The item reviews cover that.
- **Not read:** `run_report.py` (the offline report), the body of the `cited_pages.check_answer` rewrite (only its gating), `kernel_probe.py`.
- **Suspected, not verified:** `measure_tool.run_log_grid` may open a document a second time when `handle` is a path not already open (`measure_tool.py:452-470`). `TurnPageCheck._check`'s `why` (`cited_pages.py:2655`) may reach the record unmasked, unlike its photo-pages twin.
- **The full suites on GitHub.** This branch was not pushed as `ci/...`; the lead does that.
