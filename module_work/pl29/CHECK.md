# PL-29 (a) check notes — the document tools move into planlens

App BACKLOG PL-29 part (a), app split step 9a (`module_work/APP_SPLIT_PLAN.md`
§2.6 and "Step 9"). Branch `claude/pl-29a-document-tools`. The lead moves this
file into the app's notes at merge; planlens keeps no check notes.

Taken from the app's master at `d45380a` (2026-11-05), after INF-31. The plan's
line numbers were stale; every part was found by name. No version bump, no
tag, nothing published. The app is untouched: part (b) is not in scope.

## 1. `ReviewToolkit` takes a `look`

`planlens/tools/toolkit.py`: `ReviewToolkit(..., look=None)`, documented next to
`resolve_source`; the type is `planlens.tools.Look`, `(image, question, kind)
-> str`. A non-callable is refused (`TypeError`). `ReviewToolkit.LOOK_TOOLS`
names the toolkit's own tools that need it; they are left out of `specs()` and
`tool_names` without a look.

**A choice the lead should see:** none of the toolkit's own seventeen tools
needs a look (its `find_like` is the image match alone, with no reads), so
`LOOK_TOOLS` is empty and the toolkit's surface is unchanged; the MCP server and
any other caller see exactly what they saw. The tool that does need the look
is the host layer's `find_like` (`planlens.review.specs.LOOK_TOOLS`), left out of
`planlens.review.tool_names()` / `tool_specs()` when the host gives no look.
Without a look, `annotate_document` still writes its copy and its result has no
`check` (as in the app when the engine cannot see images). The look the review
tools use is `ReviewHost.look`, read at call time from the host serving the
call; the toolkit only stores its own. A host with no look can be one
conversation's: `review.using(look=None)`, or `call(..., host=)` (section 3).

## 2. What moved, from where to where

| From (app, `funhouse_agent/`) | Lines moved | To (`planlens/review/`) | Lines |
|---|---|---|---|
| `document_tools.py`: one toolkit per conversation (`_Space`, `_space`, `handle_source`, budgets) | 179 | `conversation.py` | 1,368 in all |
| `document_tools.py`: `_resolve`, `_toolkit`, lock-wait timing, `dispatch_document_tool` | 187 | `conversation.py` | |
| `document_tools.py`: image coordinates, `pdf_pages`/`pdf_page`, range hints, handle or source, viewer pages, the Word/Excel/DXF side reads (DXF reader whole) | 853 | `conversation.py` | |
| `document_tools.py`: `forget_path`; `open_document_entry`, `document_entry`, `resolve_document_source` | 97 | `conversation.py` | |
| `document_tools.py`: `_document_name`, `markup_output_path`, and writing, removing, placing marks on a copy (`_MARKUP_ID` ... `_write_marked_copy`) | 884 | `marked_copy.py` | 924 |
| `document_tools.py`: `annotate_markups_schema` and its field lists | (copied) | `specs.py` | 356 |
| `markup_check.py` | 1,399 (all) | `markup_check.py` | 1,417 |
| `find_like.py` | 410 (all) | `find_like.py` | 415 |
| `page_numbers.py` | 185 (all) | `page_numbers.py` | 185 |
| `vision_view.py`: tiles, pixel/grid/page conversions, `boxes_to_grid`, location error, precision line, zoom window and pad, aim tolerance, `answer_boxes`; turned lettering, cut lines, `image_box_to_page` | 803 of 1,589 | `frames.py` | 851 |
| (new) | | `host.py` (`ReviewHost`, `configure`, `using`, small helpers) | 461 |
| `deep/vision_kit.py` tool functions and notes, `vision_tools._dispatch_find_like` (orchestration only) | (re-written as a dispatcher) | `tools.py` | 469 |

From `document_tools.py` 2,200 of 2,446 lines moved (the plan said about
1,600): besides the four named parts, `_resolve`, `dispatch_document_tool` and
the side-read dispatch came too, because the moved parts cannot run without
them. What stays in the app's `document_tools.py`: the module docstring, the
tool-name tuples, `MARKUP_AUTHOR_ENV` / `DEFAULT_MARKUP_AUTHOR` /
`markup_author()`, the two hints naming the app's vision tools
(`IMAGE_VIEW_HINT`, `VISION_HINT`), the feature detection (`available`,
`tool_description`, `_spec`, `has_tool`, `document_tool_names`,
`search_supports_fuzzy`), `REPORT_INGEST_TOOLS` / `report_ingest_supported`,
and the Word/Excel reader's call into `office_text` (now the host's
`read_office`). In part (b) `document_tools.py` becomes the binding.

Behaviour is kept; logic was changed only where an app import had to go:

- `find_like.find_like(pdf, page, bbox, look, ..., budget=300)`: `engine` is now
  `look`; the budget is an argument (the app's env var is the binding's to
  read). `budget_s(value)` reads a value instead of the environment, as the
  app read its setting: `None`, empty or unreadable is 300 s, `0` or less no
  limit. (Before the review `None` meant no limit, so a binding passing an
  unset setting would have run unbounded.)
- `tools.find_like` reads its source through the host's `find_like_source`
  when one is given (the app's vision resolver, whose words for a key it
  cannot find differ from the document tools'), else through the document
  tools' resolver.
- `markup_check.check_marks(output_pdf, result, specs, look, ...)`: `engine` is
  now `look`; "no engine, or one without `analyze_image`" becomes "no look"
  (the binding passes `None` for an engine that cannot see). The crop is the
  host's `render_crop` when given (the app passes its budget-sized
  `render_view`), else planlens' own render at 300 dpi, the app's fixed size
  for a zoom with no budget. The verdicts' key is `ReviewHost.verdict_key`.
- `marked_copy._ours`: the app's name as a literal became the host's
  `own_mark_words` (default `("(AI draft)",)`; the binding passes its name too).
- `conversation._read_office`: the reader is the host's `read_office(path, name,
  image_dir=, image_ref=) -> (markdown, fields)`. planlens passes the working
  folder's scratch folder (not created) and its name, or `None` / `None` with
  no working folder, as the app's `_read_office` passed `scratch_dir(create=
  False)` / `SCRATCH_DIR`. Without a reader a Word/Excel file is refused with
  a hint.
- `host.check_nested_keys` (a refused `remove` key): the did-you-mean is ranked
  as the app's `suggest_params` ranks keys without units: the same letters in
  any case or spacing first (`ID -> id`), then a spelling slip (0.85), then
  the leading words of a longer name. Its unit-suffix and synonym steps serve
  the analysis parameters and were not copied. Over 47 keys (every
  `_REMOVE_KEY_HINTS` word, case variants, slips, unrelated words) the refusal
  is word for word the app's (`arg_checks.check_nested_keys` run beside it).
- `tools.annotate_document`: a marked copy that fails to write, and the
  check's own failure line, are worded by the host's `error_json` like every
  other tool's error (the line is the host's error for the tool "the
  placement check"). With the app's `tool_error_json` and `hide_paths` both
  are the app's text exactly; with no `error_json` they read as before.
- Without `ReviewHost.images`, marks and searches on an image file take the
  app's "could not be decoded here (Pillow is missing...)" path, unchanged.
- Comments citing app-only paths were reworded; "Foundry brief N" and model
  names stay as the existing planlens files already write them.

Not moved (section 2.6 "proposed"): `measure_tool.py`, `scale_labels.py`,
`dxf_notes.py`. Not moved from `vision_view.py`: the image budget per model,
the detail level, `render_view`, the patch alignment, and every text given to
the model (`READING_INSTRUCTION`, `FRAME_INSTRUCTION`, `GRID_INSTRUCTION`,
`PIXEL_INSTRUCTION`, `with_location`, the text-layer context and
`split_located`). `TURN_TAG_INSTRUCTION` moved with `read_turn_tag`, which
parses the line it asks for.

## 3. What the host passes in

One `planlens.review.ReviewHost`. The fields that are the same for every
conversation go in one base host, installed for the process with
`planlens.review.configure(base)`. The fields that belong to one agent build
(`look`, `signer`, `render_crop`) go in that build's own host, passed on
every call its tool functions make (below, "One host per conversation").

| Field | Replaces in the app | Part (b) binding passes |
|---|---|---|
| `resolve_source` | `find_readable_file`, `read_roots`, `PathRefused`, `files_here_text` inside `_resolve` | the confinement and its refusal words (the app's `_resolve` after its attachment and handle steps, which planlens takes first) |
| `look` (per build) | `vision_tools.ask_vision(engine, ...)` | `lambda image, question, kind: ask_vision(engine, image, question, kind)` with that `make_vision_tools` call's engine; for `annotate_document`, `None` when the engine is `None` or has no `analyze_image` (the app's `check_marks` guard: no `check` block). The app offered `find_like` whatever the engine, so its `find_like` calls pass the host with the engine's look in any case |
| `working_folder` | `_fileio.host_output_dir` | `host_output_dir` |
| `signer` (per build) | `markup_author or markup_author()` | `lambda: markup_author or document_tools.markup_author()`, with that build's `markup_author` (the signed-in person); the setting is read at call time, as the app read it |
| `verdict_key` | `vision_tools.reads_file` | `reads_file` |
| `find_like_budget_s` | `GEOTECH_FIND_LIKE_BUDGET_S` (read on every call; unset = 300 s, `0` = no limit) | `lambda: os.environ.get("GEOTECH_FIND_LIKE_BUDGET_S")`: read on every call, and unset (`None`) is 300 s here too. Leaving the field at planlens' default (300 s, the app's `DEFAULT_BUDGET_S`) is the same except that the setting is then not read at all |
| `display_path`, `readable_file`, `output_path`, `hide_paths` | `vision_tools.display_path`, `find_readable_file`, `_fileio.resolve_output_path`, `hide_working_folder` | the same functions |
| `render_crop` (per build) | `vision_view.render_view(pdf, page=, bbox=, pad_frac=0.0, engine=)` | `lambda pdf, page, bbox: vision_view.render_view(pdf, page=page, bbox=bbox, pad_frac=0.0, engine=engine)` with that build's engine. Not `render_view` with its default `pad_frac` (0.1), which changes the crops the check is shown |
| `find_like_source` | `vision_tools._resolve_attachment_or_path` in `_dispatch_find_like` | `lambda key, attachments: vision_tools._resolve_attachment_or_path(key, attachments)[0]`: the app's own words for a key it cannot find. Its handle step (`document_tools.handle_source`) must then be planlens' `conversation.handle_source`, the one record of handles |
| `images` (`ImageSupport`) | `image_view` (Pillow), `vision_tools._open_image` | `open_picture`/`as_pdf`/`page_scale`/`shown_size` |
| `read_office` | `office_text.read_office(path, image_dir=, image_ref=)` with the `office_to_markdown` fallback | a wrapper, not `office_text.read_office` itself (planlens calls the reader with the file's name too): `def read_office(path, name, image_dir=None, image_ref=None):` `try: return office_text.read_office(path, image_dir=image_dir, image_ref=image_ref)` `except TypeError: return office_text.office_to_markdown(path), {}`. planlens supplies `image_dir` / `image_ref` itself, as the app's `_read_office` computed them (section 2) |
| `after_marked_copy` | `cited_documents.add_to_result` | that call |
| `record` | `dispatch_custom_event("document_lock_wait", ...)` | that call |
| `error_json` | `error_text.tool_error_json` (and `tool_error` / `error_line` in `annotate_document`) | `tool_error_json` |
| `vision_hint`, `image_view_hint`, `scratch_dir`, `own_mark_words`, `max_live_toolkits`, `idle_close_s` | module constants and env settings | the app's values |

**One host per conversation (review finding 1).** The app holds the engine
and `markup_author` per `make_vision_tools` call (seven callers; the sidebar
picks the model per conversation; two testers can be signed in to one
process). `planlens.review.current()` reads a context variable first:
`review.call(name, arguments, attachments, max_chars, host=h)` and
`with review.using(h):` serve the calls inside with `h`, in that context
only. Other threads keep theirs, and the markup check's and `find_like`'s
worker threads run in a copy of the calling context, so the host reaches
them. `using(look=None)` (or `dataclasses.replace(h, look=None)`) is "no
look for this conversation". Part (b) builds, once per `make_vision_tools`
call, `h = dataclasses.replace(base, look=..., signer=..., render_crop=...)`
and passes `host=h` on every `review.call` its tool functions make. Binding
the signer or the look with `configure()` would sign every marked copy with
the name of the user whose agent was built last.

The process-wide state (toolkits per conversation, kept verdicts, the
side-read cache) is as it was in the app; `configure()` with no argument
restores the defaults (file paths only, no look, no working folder).

## 4. The spec-equality check

`planlens/review/tests/capture_app_specs.py` (kept in the branch) builds the
app's own tools with `make_vision_tools` on a fake engine, offline, for the ten
moved tools (`open_document`, `document_structure`, `document_page_map`,
`read_document`, `search_document`, `find_quantities`, `document_markups`,
`render_page_thumbnails`, `annotate_document`, `find_like`) and writes each
one's name, description and input schema (`tool_call_schema`, as handed to
the model) to `app_tool_specs.json`, with the app commit (`d45380a`) and the
planlens version (0.14.0, this checkout). `planlens/review/specs.py` builds
the same from planlens' own sources: the toolkit's description plus the host
note, and the schema of the app's tool function written out (pydantic's form:
titles, defaults, `markups` from the toolkit's item schema).
`test_host_specs.py` asserts equality per tool: **10 of 10 descriptions and 10
of 10 input schemas equal**. `measure` and `log_grid` are not in it (they stay
in the app's `measure_tool.py`).

Re-capture: `python planlens/review/tests/capture_app_specs.py --app
../GeotechStaffEngineer` (needs the app's agent stack installed).

## 5. Tests

Ported from the app into `planlens/review/tests/`, rewritten against
`planlens.review` with a fake look `(image, question, kind) -> str`
(`conftest.look_of` turns the app tests' fake engines into looks, so they
stay close to verbatim). An autouse fixture restores the host, clears kept
verdicts and closes every conversation's toolkit around each test. Made-up
values only (the app's signer, sample names and a bot name replaced).

| New file | From (app) | Tests | Left in the app, and why |
|---|---|---|---|
| `test_host_specs.py` | new | 26 | the spec-equality check (section 4) and the toolkit's `look` |
| `test_markup_check.py` | `deep/tests/test_markup_check_offline.py` | 46 functions (71 collected) | none; crops now come from planlens' own 300 dpi render |
| `test_find_like.py` | `test_find_like_tool.py`, `test_find_like_brief4.py`, `test_find_like_budget.py` | 18 | tiled page reads, budget/policy, result cap, vision probe (the app's vision tools); budget tests now use `budget=` / `find_like_budget_s`, and an unset budget (`None`, or a callable returning it) is 300 s |
| `test_frames.py` | geometry parts of `test_vision_view.py`, `test_vision_turned.py`, `test_vision_locations.py`, `test_vision_brief4.py` | 24 (32 collected) | `render_view`, budgets, detail, prompt text, turn memory/retry, patch alignment, tool dispatch; new direct tests for `page_text_layer`, `turn_image`, `turned_note`, `tile_grid`, `zoom_window` |
| `test_page_numbers.py` | `deep/tests/test_wave2c_offline.py` (page parts) | 4 | page tests on the app's own vision tools, its coverage test |
| `test_tools.py` | `deep/tests/test_document_tools_offline.py`; search and schema parts of `test_annotate_keys_offline.py` | 24 | surface checks through `make_vision_tools`, feature detection, the older-planlens fuzzy refusal, the markup-author env and sign-in signer, the app's hint text, `convert_to_openai_tool` size checks |
| `test_conversation.py` | `test_conversation_isolation.py` (document parts), `deep/tests/test_wave2b_documents_offline.py` (C2/C3/C8/C9) | 20 | the app's file tools, reference and Databricks read roots, the real Word/Excel reader tests (three new tests on the host's reader: two hand a made-up reader the conversation's scratch folder, or none with no working folder; one has no reader) |
| `test_marked_copy.py` | wave2b C4, `deep/tests/test_wave2c_annotate_copy_offline.py`, `test_annotate_keys_offline.py`, wave2c D5 | 32 (43 collected; the Windows-only path test skips off Windows) | the `strict_tool` unknown-argument refusal (here `review.call` refuses an unknown argument by name) and converted-schema checks; new: the did-you-mean as the app's (`ID`, `Id`, `TEXT`, `idx`, `ids`) |
| `test_host_per_conversation.py` | new (review finding 1); the no-look case is the app's `test_check_false_and_no_engine_leave_the_result_alone` | 6 | — |
| `test_end_to_end.py` | new | 2 | — |

The end-to-end test, offline on a made-up sheet: `annotate_document` makes a
marked copy and the check confirms the mark; `find_like` finds all three
copies with the fake look reading the contact sheets; looks of kind `check`
and `find` both happen; and a child process shows no app or agent-framework
module (`funhouse_agent`, `webapp`, `langchain`, `langchain_core`,
`langgraph`, `deepagents`) is imported.

`test_host_per_conversation.py`: two threads mark one made-up sheet at the
same moment, each through its own host (a barrier in each signer and each
look holds both inside the call at once): each copy is signed by its own
signer, lands in its own folder, is checked by its own look and cropped by
its own `render_crop` in the check's worker thread, and the process's host
is never asked. A conversation with no look (`using(look=None)`, or
`call(..., host=)` with `look=None`) gets its copy with no `check` block and
no `find_like`, while the process's host still checks. Also: `using`
restores the host after the block (nested, and after an exception);
`find_like_source` gives the host's words for a missing key; and
`annotate_document`'s write failure and check failure take the host's
`error_json`, or read as before without one. Ignoring the context variable
(`current()` returning the process's host) fails the two-thread test.

No planlens source was changed by the ported tests.

Suite (after the review fixes, Python 3.14.5 on Windows): `pytest
planlens/review -q`: **246 passed, exit code 0** (233 before). `pytest planlens
-q` in full, run as three foreground chunks because the whole takes longer
than one foreground command may (`planlens/document` 812 passed;
`planlens --ignore=planlens/document --ignore=planlens/review` 971 passed;
`planlens/review` 246 passed; each exit code 0): **2,029 passed, 0 skipped**.
Before the review fixes the same venv gave 2,016 passed, 0 skipped (the
reviewer's run); on Python 3.13 this file had recorded 2,002 passed /
3 skipped. The baseline on the branch point was 1,770 passed / 2 skipped. The
docs-claims and packaging tests (`planlens/tests/test_readme_claims.py`,
`test_packaging.py`) are in that run.

No tool's description or input schema changed in the review fixes, so the
spec-equality fixture was not re-captured; `test_host_specs.py` passes.

## 6. Left in the app, and why

- `measure_tool.py`, `scale_labels.py`, `dxf_notes.py`: "proposed", not named
  for this step.
- The agent binding: the tool functions in `deep/vision_kit.py`, their
  `StructuredTool` wrapping, `strict_tool`, `_shown_as`; planlens has its own
  dispatcher (`planlens.review.tools.call`) so the moved code runs and is
  tested here, but the app's agent keeps calling its own functions until part
  (b) points them at planlens.
- Feature detection (`has_tool`, `search_supports_fuzzy`,
  `document_tool_names`): kept one release for an older planlens (plan step 9 (b)).
- `vision_view`'s budgets, detail, `render_view` and model texts; `image_view`
  (Pillow); `office_text`; `cited_documents`; the confinement in `vision_tools`.
- The app's own copies of every moved file: deleting them is part (b).

## 7. The independent review's findings (2026-11-05), and what changed

1. A host only per process: `current()` reads a context variable first;
   `call(..., host=)`, `using(host)` / `using(**fields)`; `tool_names(host=)`,
   `tool_specs(host=)`; "no look for this conversation" is `look=None` on
   that conversation's host (sections 1, 3, 5).
2. Three section 3 rows that changed behaviour, rewritten: `read_office` is a
   wrapper keeping the fallback (planlens now hands it `image_dir` /
   `image_ref` itself); `render_crop` passes `pad_frac=0.0`;
   `find_like_budget_s` unset is 300 s (planlens' `budget_s(None)` now means
   the default, and the field takes a callable read on every call). The
   reader's return type annotation now says `(markdown, fields)`.
3. `find_like`'s words for a key it cannot find: the host's
   `find_like_source` (the app passes its vision resolver).
4. A refused `remove` key `ID` / `Id`: the did-you-mean as the app's.
5. `annotate_document`'s errors through the host's `error_json`.
6. README and DESIGN say the look the review tools use is
   `ReviewHost.look`; the toolkit only stores its own.
7. Informational (other hosts' default crop has no long-side cap; app tool
   names in planlens' model-facing text): not changed, as the review says.
