# DR-38: the page's main model sees an uploaded image itself (check notes)

## What the item asked

On the chart suite (EV-35) every loss of the Document Review page traced to the text a vision side call wrote about an uploaded chart; the main model never saw the chart. The item asks for an uploaded image to be shown to the page's main model itself, behind a switch.

`GEOTECH_VISION_SHOW_ME` (built 2026-11-10) already lets the model ask for that per call (`show_me=true`), an upload included. In the runs that offered it, no call asked (EV-40), so that arm measured nothing. This build adds the standing form: behind a new switch the main model is shown an upload without having to ask.

## What was built, and where

- `packages/aec-agent-core/aec_agent_core/review_flags.py`: the switch `GEOTECH_VISION_SHOW_UPLOADS` (`SHOW_UPLOADS_ENV`, `show_uploads()`), documented with the others and kept in `LATER_ENVS` (not `ALL_ENVS` or `ARMS`, which the geotech app's tests pin whole). `switches()` clears it unless an arm names it, and `describe()` names it when set, so `build.switches` in a turn's record shows it.
- `packages/aec-agent-core/aec_agent_core/deep/upload_view.py` (new):
  - `UploadImageMiddleware`, an `InlineImageMiddleware` that also marks the main agent's `analyze_image` calls (`wrap_tool_call`, sync and async), and only those.
  - `show_upload()`: the look itself.
  - `takes_images()`: the check made before the page is built.
- `packages/aec-agent-core/aec_agent_core/vision_tools.py`: 8 lines in `_dispatch_analyze_image`. When the source is an upload (`src == "attachment"`) and the call is marked, the look goes to `upload_view.show_upload`. Nothing else in that file changed.
- `packages/aec-document-review/aec_document_review/review_build.py`:
  - `_show_uploads(model, engine)` returns the middleware and the record line.
  - The main agent's middleware takes `UploadImageMiddleware` in place of `show_me`'s `InlineImageMiddleware` (one middleware does both).
  - `geotech_build["show_uploads"]` is written only when the switch is set.

With the switch set, the main agent's `analyze_image` of an uploaded image file:
1. decodes it, as today's look does (`vision_tools._open_image`);
2. shrinks it to the model's budget by the rule a whole-image look uses (`image_view.render`);
3. counts one image call on the turn (`keep_work.take_image_call`), or returns the ceiling's own refusal past `GEOTECH_TURN_MAX_IMAGE_CALLS`;
4. stores it and returns `image_id`, `source`, the `view` payload and a note. No vision call runs.

At the next model call the middleware puts the image in front of the main model. It goes in the request only, as the same `image_url` block (with the engine's `detail`) that the vision side call sends (`LangChainVisionEngine.analyze_image`). Each engine family's chat model already translates that block: the OpenAI-compatible route passes it through, and the hosted platform's SDK and Claude routes read it with `_image_url`. No second encoder was written.

## Calls made

- **The model has no choice under this switch.** `show_me` already gives the choice, and it was never used.
- **Only the main agent.**
  - The reading helper holds the same tool, but its calls are not marked, so its look is today's vision call. A helper may run on a model of its own, and the item asks for the page's main model.
  - Helpers, the geotech page and the `minimal` build never carry the middleware, so nothing changes there whatever the variable says.
- **Only uploads, only image files.**
  - A PDF upload gets today's answer word for word.
  - A file opened by its path (a contact sheet) and every `render_region` zoom stay as they are. A zoom on the upload is how the model has a part read out, and the note says so.
- **One shown image = one image call.** It is counted once, when shown. The re-sends while it is among the newest `GEOTECH_VISION_INLINE_KEEP` images are not counted again; they show in the model calls' input tokens.
- **Too large.** The image is shrunk first. If it is still over `INLINE_IMAGE_MAX_BYTES` (4 MB, the existing inline rule), today's vision call reads it, and the result carries `not_shown` saying why, with the size.
- **A model that takes no images.** This is told apart when the page is built:
  - No, when the model's own LangChain profile says `image_inputs: false`, when there is no vision engine, or when the vision engine reads images with another model.
  - Yes, when the page's looking tools already send images to this same model object, as the default engine (`LangChainVisionEngine(model)`) does; no host passes another engine today.
  - On "no", the page is today's and the record says `show_uploads: off: <why>`.
- `tiles` and `turn` are not used on a shown image, and the result says so (`unused`).

## Settings

| Setting | Default | Turns on |
|---|---|---|
| `GEOTECH_VISION_SHOW_UPLOADS` | off | `1` / `true` / `yes` / `on`, read when the page is built |

Unset, every message the model is handed is today's, and no fingerprint moves (`tests/test_review_fingerprints.py` passes unchanged). Set, the page's middleware differs, so its fingerprint differs as `show_me`'s does; the tool descriptions and schemas do not change.

## Tests: `packages/aec-document-review/tests/test_dr38_show_uploads.py` (new, 14 tests)

Every test fails on the base (the switch and `upload_view` do not exist there) and passes on this branch.

- Off: an upload's look is a vision call, `show_upload` is never reached, no image goes to the model, and the record has no new line.
- Set, but the build has no middleware (the geotech page, `minimal`, a helper): the look is today's.
- On: the upload is shown at the next call (one `image_url` block holding a PNG), no vision call runs, nothing is kept in the saved conversation, and the record says `on`.
- On: one shown image is one image call, and the ceiling refuses the third look under a ceiling of 2 with its own words.
- On: a PDF upload gets today's answer word for word, and nothing is stored.
- On: a file opened by its path is not shown.
- On: an image too large to show is read by a vision call, and `not_shown` says why.
- On: `tiles` and `turn` are said to be unused.
- On: a model whose profile takes no images gets today's look, and the record says `off` and why. The same holds when images go to another model, or there is no engine.
- On: the reading helper's look stays a vision call.
- The middleware marks only `analyze_image`.
- The block is the side call's `image_url` shape, with the engine's `detail`. It passes through the OpenAI-compatible route and is read by the SDK route's `_image_url`.
- The switch is a later switch that the suite can set and `switches({})` clears.

Also run, all green: the Document Review tests `test_review_show_me_offline`, `test_review_fingerprints`, `test_review_harness_offline`, `test_review_switches_offline`, `test_ta43_image_ceiling_page`, `test_review_minimal_offline`, `test_dr48_photo_pages_turn`, `test_dr43_todos_with_calls`, `test_helper_models` and `test_pl27_plan_tools_offline`; the core's `test_ta43_image_ceiling`, `test_vis34`, `test_vis33`, `test_dr48`, `test_vis31`, `test_page_reads_offline` and `test_instructions`; and the app's `test_core_move_11c`, `test_record_ev06`, `test_wheels`, `test_foundry_model_family`, `test_build_record` and `test_public_text`.

`test_review_binding_offline.py::test_the_process_host_is_the_apps` fails on the laptop on the base commit as well, so the failure is not from this change.

## Noted, not fixed

- `show_me` looks are still not counted against the image ceiling; this switch's looks are.
- The image message's header says "your last page/region views" for an upload too (shared with `show_me`).

## For the lead: the measurement on the hosted platform

Run the same image questions (the chart suite's uploaded charts and a few uploaded photos and screenshots of drawings) with `GEOTECH_VISION_SHOW_UPLOADS` unset and set, on the same model. The suite takes the arm as `("show_uploads", {"GEOTECH_VISION_SHOW_UPLOADS": "1"})`. Compare:

- **Answer quality:** score per task, and read each loss against the image.
- **Tokens:** input tokens per turn, since the image rides every model call while it is among the newest two; output tokens; and the vision side calls saved.
- **Time:** per turn.
- **The record:** `build.show_uploads` says `on`; each shown look's result carries `image_id` and the upload `note`; and the turn's `image_calls` counts them.

A run where the record says `off` measured nothing. Also check that each image-bearing request was accepted, since a gateway's request size limit is why `GEOTECH_VISION_INLINE_KEEP` exists.
