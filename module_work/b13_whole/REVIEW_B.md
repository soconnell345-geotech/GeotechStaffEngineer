# Release review B: the whole unreleased change of the two packages

**Verdict: MERGE AFTER FIXES** (the three fixes are on this branch). Reviewed head `299fb72` of `claude/b13-integration`, `git diff aec-v1.2.0..299fb72 -- packages` (92 files, 31 first-parent commits). No must-fix. Three should-fix, each fixed here in one commit with a test that fails before it. No fingerprint, setting, prompt or tool description moved.

## Findings

| # | Severity | Where (at `299fb72`) | What is wrong, and the made-up input that shows it | Outcome |
|---|---|---|---|---|
| 1 | should-fix | `aec_agent_core/deep/vision_kit.py:1448-1495` (`measure`, `log_grid`) | DR-19 meets `measure`. Every other document tool of a build runs under the build's own planlens host (`host_for`); these two ran under the process host. planlens opens a DXF as a plot for the host that asked, so with `GEOTECH_REVIEW_DXF_PLOTS=1` a two-line made-up `plan.dxf` that the model had just looked at as a plot was refused by `measure(source="plan.dxf", page=0, bbox=[10,10,60,60])` with "could not open 'plan.dxf' as a PDF or image: the file is damaged, incomplete or of another kind" and a hint to ask the user to check the file. | Fixed, `0b21f15`: both run inside `planlens.review.using(review_host)`. On planlens' `claude/b13-integration` the same call now measures. |
| 2 | should-fix | `aec_agent_core/vision_tools.py:4139` (`_vision_error`) | The record masks meet a tiled read. The masks work on a tool result's TOP-LEVEL `"error"`. A vision call that fails inside a tiled read leaves its text one level down (`tiles[].error`, `tiles_not_read`, `overview_not_read`). `RuntimeError("401 from https://gateway.example/v1?api_key=sk-madeup1234567890abcd")` on one tile: the key was in what the model read and in `tool_end`'s `result`. | Fixed, `53dcd3d`: the text is masked where it is made (`error_text.masked_error_text`); a text with no secret is unchanged. |
| 3 | should-fix | `aec_agent_core/deep/upload_view.py:193` (`_today`), `vision_tools.py:2576` | DR-38 meets EV-20. With `GEOTECH_VISION_SHOW_UPLOADS=1`, an upload that is not shown (a 64 x 48 made-up `photo.png` with the inline cap set to 10 bytes; or `notes.txt`) goes through `_dispatch_analyze_image` twice, and the record's `image_read` event was written twice for one tool call. The image-call count was right (1). | Fixed, `1d8d3ca`: the second pass does not write the event again. |
| 4 | note, left | `aec_agent_core/cited_pages.py:1995`, `:2015-2033` | A zoom on an uploaded picture is recorded as page 0 of a document `site.png`; "PDF page 1" of ANY document then counts as opened, and the printed labels come back unknown, so the page check goes quiet for the conversation. Made-up: a report alone gives `never_opened` = "PDF page 1", "page 12"; after one `render_region` on `site.png` it is empty. | Left: the same output on the 1.2.0 module, so no merge of today caused it; VIS-26 and DR-38 make picture work more common. The fix (leave image files out of the documents the check reads) changes what the check returns: the lead's call, a BACKLOG item. |
| 5 | note, left | `aec_agent_core/deep/upload_view.py:104` (`left_out`), `deep/inline_images.py:56` | With show-uploads on, the upload goes into the process-wide image store (96 images, oldest out). If 96 other images are stored between a tester's `analyze_image` result and that tester's next model call, the image is not shown, the result still says it will be, and no "not shown" note is added. | Left: needs 96 stores inside one step; the store and its bound are older than this change. Smallest fix: `left_out` treats an id the store lost as left out. |
| 6 | note, left | `aec_agent_core/cited_pages.py:897-900` | A table read of several pages puts the result's `printed_page` on the first page that succeeded, not the first sent (index 87 failed, 88 ok, `printed_page` "88": "p. 88" counts as opened). | Left: the table reader is on the geotech page only, which is quiet here. |
| 7 | note, left | `aec_document_review/minimal_agent.py:204` (`page_count`) | Its own `{"error": "<type>: <text>"}` is not masked where made. | Left: top-level `error`, so the guard and the record mask it; the errors are a missing file's or PyMuPDF's. |

## 1. The files merged by hand (each read whole at the head)

- `activity_log.py`: holds. `APP_EVENTS` has both new events once (`image_read`, also in `LOOK_EVENTS`; `summary_failed`). `turn_end`: `error` and `error_detail` masked before the cut; `calls_not_started` capped at 20 with the unanswered first; then four facts blocks, each in its own `try`.
- `deep/keep_work.py`: holds. `TurnScope` carries the image-call fields, `lean_prompt` and the summary fields side by side under one lock; `image_call_facts`, `lean_prompt_facts` and `summary_facts` return disjoint keys.
- `minimal_agent.py`: holds. The helper's failure line is masked and cut to 300; its open items are added after the answer and an exception there hands the answer back unchanged; the sentence on an uploaded picture and `page_count`'s image branch agree with the tool descriptions (page 0, its own pixels).
- `review_flags.py`: holds. Seven new `*_ENV`, each with a reader, each in `LATER_ENVS` and `__all__`; `LATER_SETTINGS_ENVS` is cleared by `switches()` and printed by `describe()`.
- `cited_pages.py`, `cited_documents.py`: hold (finding 4 and 6 are older). A citation is filed in at most one list; the top-level `why` is masked by the logger by its name and the nested `photo_pages.why` where it is made; a table read whose pages all failed grants no page and no source; an error in the table-reader hook cannot lose a result.
- `vision_tools.py`, `document_tools.py`, `measure_tool.py`: findings 1 to 3; the rest holds (below).

## 2. Settings built together

`/tmp` scripts built the default and the `minimal` build with each setting alone and each pair, and compared tool names, descriptions, schemas, system prompt lines, middleware and each helper's tools; a second script ran one offline turn and read what the model was handed after every middleware. On planlens 0.16.0 (PyPI) and, for the planlens parts, on planlens' own `claude/b13-integration` (`803e591`).

| Pair | What it showed |
|---|---|
| lean prompt + plan tools / drawing tool / DXF plots / sheet tools | Builds. At the model: the same tool set as the other setting alone, only `write_todos` and `task` changed, the system prompt shorter by the same 1,056 characters, `lean_prompt` on the turn `applied: true` with the same `cut`. |
| lean prompt + figure examples / to-dos with calls / show uploads / `show_me` / checklist / image ceiling / pixel requests | The same: nothing dropped or doubled, `lean_prompt` written once. |
| show uploads + pixel requests | Builds; no tool or prompt changed by both. An image file never takes the pixel-request path, so nothing is converted twice. |
| show uploads + `GEOTECH_TURN_MAX_IMAGE_CALLS` | A shown upload is one image call, an upload too large to show is one (through the engine); past the ceiling the upload is refused in the result. |
| show uploads + `show_me` | One middleware (`UploadImageMiddleware`) stands in for the inline-image one, by design; not attached twice. |
| pixel requests + ceiling; pixel requests + tile answers | No second vision call is added; each tile converts in its own frame, once. |
| photograph-page one look + photograph pages; both cited-pages modes on `retry` | One note with each prefix once, three records, each key once; an uploaded image goes to `left_out`, never `unseen`. |
| DXF plots + sheet tools / plan tools / drawing tool | Build; no tool twice; only `open_document`'s description moves for the plots; two looks at a plot count 2 of a ceiling of 3. Finding 1 was here. |
| plan tools + sheet tools; drawing tool + sheet tools | The three planlens tools gain one `sheet` argument, descriptions unchanged (PL-37's design). |
| every setting at once | Default build 29 tools (30 on the integration planlens), `minimal` 7 and 4: no name twice, no middleware twice. |

## 3. The turn's record with everything on

A `turn_end` written with a ceiling met, a refusal, a lean prompt, skipped summaries, a call not started, a cache count and a masked error: every key once, none overwritten. With each of the three facts functions made to raise in turn, only its own keys were missing.

## 4. A shared host

| State added or changed since 1.2.0 | Keyed | Holds |
|---|---|---|
| `summary_guard.install()` (a wrapper on the library's class) | process-wide, once, under a lock held for the patch only | yes |
| a failed summary "not again this turn" | on the `TurnScope`, by summarizer and helper | yes: two threads sharing one summarizer, A's failure did not stop B's summary nor reach A's next turn |
| `_guarding`, `_holding`, `upload_view._ON`, `vision_kit._SHEET_ADDRESS` | context variables, reset in `finally` | yes |
| the logger's asked / answered / cache counts | one logger per turn | yes |
| error masking | no state | yes |
| the ceiling "cleared between arms" | the environment variable, in the suite only; counts sit on the scope | yes |
| the lean prompt's `last`, `_warned` | on the middleware; the record reads the scope | yes |
| the inline image store, newly reached by shown uploads | process-wide, random ids | finding 5 |

## 5. Tests on this branch's head

- `packages/aec-document-review`, venv (a): `pytest tests` 1284 passed, 18 skipped (includes `test_review_fingerprints.py` and `test_boundary.py`: 9 passed at the reviewed head too).
- `packages/aec-agent-core/tests`, venv (b): 1140 passed, 1 skipped.
- `webapp/tests/test_page_fingerprints.py`, `test_wheels.py`: 19 passed, 5 skipped. `test_public_text.py`, `test_review_freeze.py`: 13 passed.
- New: `tests/test_rvb_measure_under_build_host.py` (2, both fail before), `test_rvb_tile_error_masked.py` (3; 2 fail before, 1 pins an ordinary error kept word for word), `test_rvb_image_read_once.py` (3; 2 fail before, 1 pins the shown upload's single record).

## Could not check

- The drawing tool, the DXF plot and the sheet arguments do not exist in planlens 0.16.0: checked only against planlens' `claude/b13-integration` installed over venv (a). The fingerprints have no entry for it.
- Any live model, the testers' host, a real 96-image race, and whether the agent library carries the turn's scope into every thread and task of a live turn (without a scope each call tries its summary).
- The lean prompt only on deepagents 0.7.24 (the installed one), not 0.6.8.
- GSE's full suite and the six other CI jobs: the lead's `ci/` push.
