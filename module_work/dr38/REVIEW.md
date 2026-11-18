# DR-38 review: `claude/dr-38-main-model-image`

## 1. Verdict

**MERGE AFTER FIXES**, reviewed head `a8d4e13` (base `519c425`). The two fixes are made on `claude/rv-dr-38` (`bc28506`, `b92517f`); one finding is left for a design decision (F3), and the merge with `origin/claude/b13-integration` has two textual conflicts (section 3, path 8).

## 2. Findings

| # | Severity | Where | What is wrong | Input that shows it | Fix |
|---|---|---|---|---|---|
| F1 | must-fix, **fixed `bc28506`** | `aec_agent_core/deep/upload_view.py`, `UploadImageMiddleware`; `inline_images.newest_image_ids` | A model call carries only the newest `GEOTECH_VISION_INLINE_KEEP` images (2). When one step looks at more uploads than that, every result says "The uploaded image is shown to you with your next step" and every look counts one image call, but the oldest images are never in front of the model, and nothing says which. The model can then answer about an image it was never shown. | Three uploads `first.png`, `second.png`, `third.png`, one model step with three `analyze_image` calls: the next model call held 2 image blocks, 3 results with `image_id`, `image_calls` 3, no vision call. Same on deepagents 0.6.8 and 0.7.24. | The middleware adds one line to the image message, straight after such a step, naming the uploads not shown and how to see them (`NOT_SHOWN_NOTE`, `left_out`). Only uploads this switch stored (`"upload": True` in the store's entry); nothing changes unset or for a batch that fits. |
| F2 | should-fix, **fixed `b92517f`** | `upload_view.show_upload` | Today's read carries `image_note` (`image_view.open_picture`): "converted from TIFF to PNG (page 1 of 3: only the first was sent)", or that a photo was turned as its EXIF says. The shown look dropped it, so the other pages of a multi-page TIFF or GIF were lost without a word. | A made-up three-page TIFF: today's result has `image_note` with "page 1 of 3"; the shown result had none. | `show_upload` carries `picture.note` as `image_note`. |
| F3 | should-fix, **left: needs a design decision** | `upload_view.takes_images`; `review_build._show_uploads` | A wrong "on". The page's default engine is `LangChainVisionEngine(model)`, so `engine.model is model` holds for every model the page is built with, and the rule says "yes" unless the model's profile says `image_inputs: false`. A text-only model with no profile (a custom chat class, an OpenAI-compatible route) is "on". Today its look fails as one tool error and the turn goes on; under the switch the main model's own call carries the image and fails, and so does every later call of that turn, since the image rides each one. | A fake main model that raises "400: this model does not support image input" when a message holds an image: the turn raised at the first call after the look (one look, no answer). | Not made. The sure test is one live call when the page is built (`inline_images.probe(model)` exists and is unused here), which costs a call per build; or catch the refusal once and fall back. Either is the lead's call. Until then: set the switch only where the main model is known to take images, and read `build.show_uploads`. |
| F4 | note | `keep_work.TurnScope` (comment), `upload_view.show_upload` | A look left out by F1 is still counted as an image call, and seeing it costs a second one. `TurnScope`'s comment still says inline images are never counted. | The F1 input: `image_calls` 3 for 2 images seen. | None made (a refund at model-call time is a design change). |
| F5 | note | `upload_view.show_upload` | A second look at the same upload stores a second copy and counts a second image call; today's repeated read is served from the repeat cache with none. The image also rides every model call of the turn while it is among the newest 2 (4 calls after one look: 1 image each), as CHECK.md says. | One look, then three more steps: image blocks per call 1, 1, 1; the next turn 0. | None; brief 9 measures the tokens. |
| F6 | note | `inline_store` (`MAX_IMAGES = 96`, one table for the process) | Stored uploads can be up to 4 MB each, so the shared table can hold up to about 380 MB on the testers' host, where page views were far smaller. Ids are random, so one tester cannot name another's image. | By reading. | None; lower `MAX_IMAGES` or bound by bytes if the host's memory shows it. |
| F7 | note | `review_eval/runner.py`, `show_me_looks` | The suite's RESULTS column counts only calls made with `show_me=true`, so a shown upload is not in it. A report tells a shown image from a read one by the look's result (`image_id` and `note`, no `analysis`) and by `build.show_uploads`. | By reading. | A column for the arm when brief 9 is written. |

## 3. Paths checked

1. **Unset, byte for byte: holds.** `_show_uploads` returns `(None, None)` before importing `upload_view`; `vision_tools` reaches `upload_view.active()` only for an upload and it is false without the middleware. Both fingerprint tests pass unchanged (Document Review's on both deepagents; the geotech app's two pages in venv b). `show_upload` patched to raise is never reached.
2. **The mark: holds.** Only the main agent's `analyze_image` (`wrap_tool_call`, sync and async); reset in `finally`, so a call that errors leaves nothing (`active()` false after a look at a missing key). A helper's look is a vision call (the builder's test, re-run). The image is added to the request only: the saved conversation holds no image, a retried model step re-reads the same tool results (no double count), and a later turn of the conversation carries no image (0 blocks) until the model looks again. A batch of looks: F1.
3. **`takes_images`: a wrong "off" is visible** (`build.show_uploads: off: <why>`, the page is today's). A wrong "on": F3.
4. **The ceiling: holds.** The count is `keep_work.take_image_call` (the look's `image_calls` line is the vision read's), and the refusal is the same sentence a vision read returns (`IMAGE_CEILING_ERROR` through `{"error": ...}`). `show_me` looks are still not counted (the builder's note; unchanged).
5. **What the model is told: holds, after F1 and F2.** The result says the image comes with the next step, that no vision call read it and the prompt was not used; the image is in the very next model call. The image message's header still says "page/region views" for an upload (the builder's note).
6. **The bytes: holds.** The whole-image render (`image_view.render`, the model's budget, JPEG only where the engine takes it), the engine's `detail`, the same `image_url` block the side call sends. The 4 MB rule is applied after shrinking, and the fallback is today's read with `not_shown`. A file that is not an image, or one that cannot be decoded here, gets today's answer.
7. **`test_the_process_host_is_the_apps`: the environment, not this branch.** It fails the same at `519c425` and at the head when its file runs alone or after most single files, and passes in the whole Document Review suite (1154 passed at the head). `BASE_HOST.own_mark_words` is fixed when `aec_agent_core.document_tools` is first imported: "Document Review" if `aec_document_review` was imported before it, "AEC agent" if not, and the test file imports the core first. The page itself imports the package first, so testers get the right words. A test-order matter for the lead to file; no change made.
8. **Merge with `origin/claude/b13-integration` (`821ffc6`): two files conflict**, both "each side added beside the other": `review_flags.py` (four places: the docstring entry, the `*_ENV` names, `LATER_ENVS` and its comment, `__all__`, against DR-35's and DR-29's switches) and `review_build.py` (one place: `_show_uploads` beside `_lean_prompt`). Keep both sides in each; `LATER_ENVS` then holds seven names. The `page_middleware` lines merged cleanly. Nothing was pushed. After the merge, Document Review's fingerprints need no re-record for this branch (unset is unchanged).
9. **A shared host.** No lock, no thread and no new table; the mark is a context variable, set and reset inside one tool call; the store is the existing one (F6).

## 4. Tests

Venv a (deepagents 0.7.24, langchain 1.4.5, planlens 0.16.0), from `packages/aec-document-review`:
- `tests/test_dr38_show_uploads.py tests/test_review_fingerprints.py tests/test_review_show_me_offline.py tests/test_boundary.py`: all passed; with `tests/test_review_binding_offline.py` 38 passed, 1 failed (path 7).
- The whole suite at the reviewed head without the new file: 1154 passed, 3 skipped.
- After the fixes: `tests/test_dr38_review_fixes.py tests/test_dr38_show_uploads.py tests/test_review_fingerprints.py tests/test_boundary.py`: 26 passed.
- The same venv with `deepagents==0.6.8`: the builder's 14, the fingerprints and `test_review_show_me_offline` passed (35 with my turns); the fix tests and the 14: passed.

Venv b, from the root: `packages/aec-agent-core/tests/test_ta43_image_ceiling.py test_vis31_one_image.py test_vis34_tile_answers.py test_page_reads_offline.py webapp/tests/test_page_fingerprints.py test_wheels.py test_docs_currency.py test_backlog.py`: 113 passed, 5 skipped. Guards (`test_public_text.py`, `test_review_freeze.py`): 13 passed before each commit.

On the base: the builder's file does not collect at `519c425` (`review_flags.SHOW_UPLOADS_ENV` and `upload_view` do not exist), so all 14 fail there by import, as CHECK.md says. Two of them would pass in substance on the base (`test_off_...` apart from its patch, `test_set_but_without_the_page_s_middleware...`): by design, they pin "unchanged".

The fix tests (`tests/test_dr38_review_fixes.py`): `test_an_upload_left_out_of_a_batch_is_named_as_not_shown` failed before `bc28506`; `test_a_shown_upload_keeps_the_note_today_s_read_gives` failed before `b92517f`; `test_a_batch_that_fits_says_nothing_more` passes before and after (it pins that nothing is added).

My own turns (not committed), on both deepagents: the batch of three (F1); one look then three steps and a second turn (path 2); a look at a missing key (path 2); the multi-page TIFF (F2); a model that refuses images (F3). The tests check behaviour (what the model's messages hold, the counts, the record), not text alone.

## 5. Scope

Six files, all inside the item: the new module, `review_flags.py`, 8 lines of `vision_tools.py`, `review_build.py`, the new test file, CHECK.md. No existing test edited, no frozen file, nothing of the geotech app. My commits add to `upload_view.py` and one new test file.

## 6. CHECK.md

- "Every test fails on the base": true only by the import error (section 4).
- "`test_the_process_host_is_the_apps` fails on the laptop on the base commit as well, so the failure is not from this change": true; the cause is import order (path 7).
- "the image is shown to the main model at its next step": untrue for a batch larger than the keep (F1), now said to the model.
- "A model that takes no images ... is told apart when the page is built": only where its profile says so (F3).
- The release note is missing; with the switch off by default "None" is right.

## 7. Could not check

- Whether each hosted route accepts an image in a user message after tool results from the page's main model, and the request size with two 4 MB images: on the hosted platform, run `inline_images.probe(model)` once, then one turn with the switch set and two large uploads; read `build.show_uploads` and the model call's status.
- Answer quality, tokens and time: brief 9's arm `("show_uploads", {"GEOTECH_VISION_SHOW_UPLOADS": "1"})`, read in full.
- The full suites on GitHub: push the merged result as `ci/<name>` (not done here).
