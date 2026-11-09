# DR-36 and DR-37: CHECK

Branch `claude/dr-36-37-vision-probe-question`, from master `3b30181`.
Evidence: `module_work/review_eval_results/2026-11-08_chart_suite_ev35/TRACE_REVIEW.md`
sections 4.1 and 4.2. No calculation module changed, so no printed-page cites apply. The cites
here are to the code and to the measured runs.

## DR-36: a failed vision probe lowered the image detail, silently

**The fault** (TRACE_REVIEW 4.1): probe calls were not retried. A 409 on call 4 was read as
"detail=original refused". A failed probe's profile was cached for the process. On the chart
suite this hit 8 of 200 Sol page runs (detail `high` instead of `original`) and changed one
answer (2108 baseline).

**What changed**
- `funhouse_agent/vision_probe.py:241` `_invoke_retrying`: every probe call (text, small,
  large, original, wide) is asked again while the model answers busy. It uses the vision side
  calls' own classifier, tries and backoff: `vision_engine.busy_kind` (409, 429, 5xx, 529,
  timeouts, dropped connections, Foundry's named rate limits), `busy_tries` (default 3, from
  `GEOTECH_VISION_BUSY_TRIES`), `_busy_wait_s` (2 s doubled, with jitter, or `retry-after`).
  Each retry is on the record (`probe_call`, `probe_try` on the side-call label) and counted in
  `VisionProfile.retries`.
- `:362`: a call 4 that is busy to the last try raises; it is never read as a refusal. A real
  refusal (a non-busy error) is still read as "original refused" and cached, as before.
- `:397-401`, `:427`: any probe that could not finish (calls 1 to 3, a busy call 4, a busy
  call 5) sets `VisionProfile.failed`.
- `:477-479` (`profile_for`): a failed profile is never put in the process cache. It is held in
  `_FAILED` for `FAILED_HOLD_S` (120 s, `:201`) so a model that stays down is not re-probed on
  every look, and is measured again after that. The cache only ever holds finished probes.
  `cached_profile` / `cached_profiles` still return a held failure, so run records show it.
- **Said:**
  - `VisionProfile.fallback_note()` (`:148`) gives the text;
  - `vision_view.view_payload` puts it on every vision result as `vision_probe` (`vision_view.py:546-549`,
    helper `probe_fallback_note` at `:279`);
  - `LangChainVisionEngine.analyze_image` puts it on the side call's label as `probe_fallback`,
    which `activity.jsonl` writes on the call's `model_start` (`deep/vision_engine.py:736-743`;
    read from the cache only, so it never starts a probe the call would not have made);
  - `to_dict` gains `failed` and `retries`, and `summary()` says "probe FAILED" (shown in
    Connection diagnostics).

## DR-37: the side call never saw the user's question

**The fault** (TRACE_REVIEW 4.2): `analyze_image` and `render_region` sent only the main
model's rewording. On 2089, both Sol page arms answered a narrower question.

**What changed**
- New `funhouse_agent/turn_question.py`: the user's latest message is held in a context
  variable bound to the turn, the way `_fileio.bind_working_dir` binds the working folder.
  LangGraph copies the context into tool threads, so there is no new tool argument.
  - Capped at 600 characters (`MAX_CHARS`, `:30`).
  - `with_question` (`:83`) appends it under the main model's prompt. The added text (`_HEAD`,
    `:39`) says the request above is what to answer, and the user's message is context only:
    "Do not answer it instead of the request, and do not add what it asks for that the request
    leaves out".
  - Kinds that get it (`QUESTION_KINDS`, `:36`): image, page, overview, tile, region,
    reference_figure, worked_example_page. That is every reading handed back to the main model,
    on both pages.
  - Left out: the markup check's look at one mark (`check`) and find_like's candidate reads
    (`find`). They read one thing for code.
- `funhouse_agent/vision_tools.py:2232` `ask_vision`, the one choke point every side call
  passes, adds the question. The call's record carries `user_question_chars`.
- `webapp/turn_jobs.py:174-180` binds `ctx["prompt"]` (the user's own text, without the
  attachment notes) at the start of every turn. Both Document Review and the geotech page run
  this path, and it covers their helpers too.
- `funhouse_agent/review_eval/runner.py:256` binds each turn's text the same way, so the
  review suite measures what the app does.
- `funhouse_agent/document_tools.py` is **not touched**.

**Fingerprints:** no tool description or page prompt changed. The side calls' text is not part
of a page build. `webapp/tests/test_page_fingerprints.py` passes unchanged, so there is no
re-record commit.

## Tests (pytest exit codes)

- `funhouse_agent/tests/test_vision_probe_retry.py` (new, 13): exit 0.
  - A 409 on each of the four probe calls in turn, then success, gives the full profile:
    `openai-original`, not refused, cached, the backoff waited once.
  - A busy wide (call 5) is retried.
  - A persistent 409 on `original` is a failed probe, not a refusal, not cached, and stated.
  - A persistent failure on calls 1 to 3 gives the default budget, with `vision_probe` on the
    result.
  - A held failure is re-measured after the hold, and then cached.
  - The side-call label carries `probe_fallback`.
  - A real 400 refusal is still a refusal and cached.
- `funhouse_agent/tests/test_turn_question.py` (new, 12): exit 0. Covers:
  - cap, kinds and instructions;
  - through `analyze_image`, `render_region` (image and PDF) and `analyze_pdf_page`;
  - a copied context in a thread;
  - the app's turn worker binding `ctx["prompt"]`;
  - no binding gives the prompt alone.
- `funhouse_agent/tests/test_vision_probe.py`: 24 passed, exit 0. One expectation was updated:
  the probe label now carries `probe_call`.
- The vision files that use the probe, plus `webapp/tests/test_public_text.py` and
  `test_page_fingerprints.py`: 354 passed, exit 0. These are `test_vision_turned`, `_find_like_tool`,
  `_vision_locations`, `_find_like_brief4`, `_vision_brief4`, `_vision_view`, `_vision_images`,
  `deep/tests/test_strict_args_offline`, `_vision_side_call_timeout_offline`,
  `_wave2b_guards_offline` and `webapp/tests/test_palantir_claude_engine`.
- `webapp/tests/test_activity_log*.py`, `test_diagnostics*.py` and `funhouse_agent/review_eval`:
  159 passed, exit 0.
- Wide run of `funhouse_agent/tests webapp/tests funhouse_agent/review_eval
  funhouse_agent/deep/tests`: 6,748 passed, 42 skipped, 16 failed (exit 1).
  - **All 16 fail the same way on master `3b30181`** (re-run there: 16 failed). They are
    reference tests reading the geotech-references pin, which is newer than the PyPI 1.5.0 this
    venv has: REF-61/62 GEC-13 removals, the fhwa formulas, example ids and citations. Plus
    `test_calc_package_figure_conventions::test_large_pile_group...` and
    `test_defaults_used[profile_figure|plot_data]`.
  - Not caused by this branch.

## Release note (what a user notices)

- A vision model that answers "busy" while the app first measures it no longer quietly drops
  every later look to a smaller image. The measurement is retried. If it still fails, the
  fallback is not kept for the life of the process, and each vision result made on it says
  "the vision model could not be measured", until the model is measured again a couple of
  minutes later.
- The vision reading of an image, page, zoom or chart is now given the user's own question as
  context beside the agent's request. A reading should no longer answer a narrower or reworded
  question than the one the user asked.
- No calculated value changes.

## Noted, not fixed

- The repeat-read cache (`vision_tools._repeat_or_read`) keys on the main model's prompt and
  the image, not the user's question. The same request on the same image in a later turn with
  a different question reuses the earlier reading. No saved report's output changes.
- A vision call that waits more than `PROBE_WAIT_S` for another call's probe still goes ahead
  on the configured budget without a note. This was outside the item.
- A non-busy error on probe calls 1 to 3 (for example, a model that takes no images) is now
  re-probed every 120 s instead of once per process: about a cent a time.
