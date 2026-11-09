# PL-29 (b) check notes: the app binds to planlens.review

App BACKLOG PL-29 part (b), app split step 9 (b) (`module_work/APP_SPLIT_PLAN.md`
§2.6 and "Step 9"). Branch `claude/pl-29b-binding`, from master `05eab2d`.
Built against planlens main `34261bf` (part (a); its version still reads
0.14.0, the release will be 0.15.0). No behaviour change: this is a move. No
version bump, no tag, nothing published; planlens untouched. The lead does
BACKLOG, HANDOFF and CLAUDE.md at merge.

**CI on this branch fails until planlens 0.15 is on PyPI.** The pin is now
`planlens>=0.15` and PyPI's 0.14 has no `planlens.review`, so the install
step cannot resolve. That is expected: the lead merges after the planlens
release.

## 1. What moved out, what stayed

| File | Lines before | After | What happened |
|---|---|---|---|
| `funhouse_agent/document_tools.py` | 2,446 | 467 | Became the binding (section 2). Deleted: the toolkit per conversation, `_resolve`'s attachment and handle steps, lock timing, `dispatch_document_tool`'s body, image coordinates, page ranges and viewer pages, the Word/Excel/DXF side reads, the marked copy (writing, removing, placing, image frames), `annotate_markups_schema` and its field lists. Kept: the tool-name tuples, `MARKUP_AUTHOR_ENV` / `DEFAULT_MARKUP_AUTHOR` / `markup_author()`, the two hints naming the app's vision tools, `MAX_LIVE_TOOLKITS` / `IDLE_CLOSE_S` (now host fields), the feature detection (`tool_description`, `_spec`, `has_tool`, `document_tool_names`, `search_supports_fuzzy`), `REPORT_INGEST_TOOLS` / `report_ingest_supported`, and the app-only tail of `_resolve` (now the host's `resolve_source`) |
| `funhouse_agent/markup_check.py` | 1,399 | 14 | An alias of `planlens.review.markup_check` (`sys.modules`, as the step 6 moves do), importing the binding first so the app's host is installed |
| `funhouse_agent/find_like.py` | 410 | 14 | An alias of `planlens.review.find_like`. Its `BUDGET_ENV` is now `document_tools.FIND_LIKE_BUDGET_ENV` |
| `funhouse_agent/page_numbers.py` | 185 | 14 | An alias of `planlens.review.page_numbers`; the app's own imports (`vision_tools`, `vision_kit`) name `planlens.review.page_numbers` directly |
| `funhouse_agent/vision_view.py` | 1,589 | 843 | The frame conversions, tiles, zoom window and pad, aim tolerance, `answer_boxes`, the turned lettering, cut lines and `image_box_to_page`: 64 top-level definitions, each AST-identical (docstrings included) to its copy in `planlens.review.frames`, deleted and imported from there under the same names. Kept: budgets, detail, `render_view`, patch alignment and every text given to the model |
| `funhouse_agent/vision_tools.py` | 5,204 | 5,037 | `find_like`'s orchestration (`_dispatch_find_like`'s body, `_find_like_on_image`, `_image_find_boxes`, `_named_contact_sheets`, `FIND_ON_IMAGE_FRAME`) replaced by one call into the binding; now `planlens.review.tools.find_like` |
| `funhouse_agent/deep/vision_kit.py` | 1,651 | 1,603 | The tool functions keep their signatures and docstrings, so names, descriptions and schemas don't change. `annotate_document`'s body (about 90 lines of orchestration) is now one call into the binding (`planlens.review.call`); the document tools and `find_like` pass the build's host |

In all: 14 files, 461 lines added, 5,330 deleted (`git diff 05eab2d --stat`,
the code commit plus the pin; the new test file and this note come on top).
One correction to the code commit's message: it says `document_tools.py` went
to "about 560 lines"; it is 467.

## 2. The binding table as built

`document_tools.app_host()` builds the base host. It is installed for the
process when `funhouse_agent.document_tools` is imported
(`BASE_HOST = review.configure(app_host())`), so library callers (the
coverage ledger, the findings, geometry, digest and sweep tools, the
cited-page check, `measure_tool`) resolve and open documents exactly as the
agent's tools do. `document_tools.host_for(engine, markup_author)` is one
agent build's host: `dataclasses.replace(BASE_HOST, look=..., signer=...,
render_crop=...)`. `make_vision_tools` builds it once per call, and every
document tool, `find_like` and `annotate_document` call that build makes
passes it (`dispatch_document_tool(..., host=)`, `find_like(..., host=)`,
`annotate_document(..., host=)` → `review.call(..., host=)`).

| Field | Passed (as CHECK §3 asks) |
|---|---|
| `resolve_source` | `_resolve_source`: the app's old `_resolve` after its attachment and handle steps (planlens takes those first): `find_readable_file`, `PathRefused` → `ToolError` with the attachment keys, and the two refusals naming the conversation's files or real paths, word for word |
| `look` (per build) | `lambda image, question, kind: ask_vision(engine, image, question, kind)` with the build's engine; `None` with no engine (`find_like` then returns unverified matches, as before). For `annotate_document`, `checking_host()` sets `look=None` when the engine is `None` or has no `analyze_image` (no `check` block, the app's old guard) |
| `working_folder` | `_fileio.host_output_dir` |
| `signer` (per build) | `markup_author or document_tools.markup_author()`, read at call time |
| `verdict_key` | `vision_tools.reads_file` |
| `find_like_budget_s` | `lambda: os.environ.get("GEOTECH_FIND_LIKE_BUDGET_S")`, read on every call; unset is 300 s, `0` no limit |
| `display_path`, `readable_file`, `output_path`, `hide_paths` | `vision_tools.display_path`, `vision_tools.find_readable_file`, `_fileio.resolve_output_path`, `_fileio.hide_working_folder` |
| `render_crop` (per build) | `vision_view.render_view(pdf, page=int(page), bbox=bbox, pad_frac=0.0, engine=engine)` with the build's engine, `render_view` looked up at call time (a test that patches it still bites) |
| `find_like_source` | `vision_tools._resolve_attachment_or_path(key, attachments)[0]` (its handle step is planlens' `handle_source`, through the re-export) |
| `images` | `ImageSupport(open_image=vision_tools._open_image, as_pdf=image_view.as_pdf, page_scale=image_view.page_scale, shown_size=image_view.shown_size)` |
| `read_office` | the wrapper: `office_text.read_office(path, image_dir=, image_ref=)`, and on `TypeError` `(office_text.office_to_markdown(path), {})` |
| `after_marked_copy` | `cited_documents.add_to_result(result, cited_documents.deliverable_texts(markups), where="annotate_document")` |
| `record` | `dispatch_custom_event(event, data)`, silent outside a run |
| `error_json` | `error_text.tool_error_json` |
| `vision_hint`, `image_view_hint` | the app's `VISION_HINT`, `IMAGE_VIEW_HINT` |
| `scratch_dir` | `vision_tools.SCRATCH_DIR` (`.scratch`) |
| `own_mark_words` | `("GeotechStaffEngineer", "(AI draft)")`, the words the app's `_ours` tested |
| `max_live_toolkits`, `idle_close_s` | the app's settings (`GEOTECH_DOCUMENT_TOOLKITS`, `GEOTECH_DOCUMENT_TOOLKIT_IDLE_S`), read at import as before |

A `find_like` reached through `vision_tools.dispatch_extended_tool` rather
than an agent build's tools (`_dispatch_find_like`) builds `host_for(engine)`
for that call; the agent's own tool no longer goes that way.

Aliases kept until step 16: the three `sys.modules` stubs; `vision_view`'s
imported names; and in `document_tools`, the names other modules and tests
import (`handle_source`, `looks_like_handle`, `document_entry`,
`open_document_entry`, `resolve_document_source`, `with_viewer_pages`,
`budget_for_cap`, `available`, `_toolkit`, `write_marked_copy`,
`markup_output_path`, `remove_problem`, `markups_with_pages`, `image_of`,
`image_frame_result`, `empty_comments_note`, `_document_name`,
`annotate_markups_schema` and its field lists, ...), imported from
planlens.review, plus a module `__getattr__` that finds any other moved name
there. They are the same objects, so reading one works; patching a moved
function now means patching it in `planlens.review`.

## 3. Exit checks

- **Page fingerprints:** `python -m webapp.tests.page_builds record` with
  planlens main installed wrote the entry for surface `709cbea6e23e8aaa`
  (version 0.14.0, main's label) and `git diff` of
  `webapp/tests/page_fingerprints.json` is empty: tool for tool (name,
  description, schema), helper for helper, all four builds equal the
  previous entry. `test_page_fingerprints.py` passes without a skip. When
  0.15.0 is on PyPI, record once in a venv with it; the surface hash should
  be the same (the toolkit's specs did not change in part (a)), so only the
  label should move. The 0.13.0 entry (`f440beafcd9dfd89`) can no longer be
  installed beside the app (pin >=0.15); the lead may delete it.
- **Boundary test** (`webapp/tests/test_review_boundary.py`): unchanged and
  passing.

## 4. Tests

All on Python 3.11 in a fresh venv, planlens main `34261bf`, each suite on
pytest's own exit code, chunked by folder with `-n 4`.

| Suite | Baseline (the branch point, same venv) | This branch |
|---|---|---|
| `funhouse_agent/deep/tests` | 1,187 passed, 4 skipped, exit 0 | 1,194 passed, 4 skipped, exit 0 (7 new) |
| `webapp/tests` (fingerprints, boundary) | 1,131 passed, exit 0 | 1,131 passed, exit 0 |
| `funhouse_agent/tests` | 4,240 passed, 28 skipped, **2 failed**, exit 1 | 4,239 passed, 28 skipped, **the same 2 failed**, exit 1 (one fewer: `test_find_like_budget.py` went from 5 tests to 4) |
| `webapp/tests/test_public_text.py` | | 4 passed |

The two `funhouse_agent/tests` failures fail the same way on the untouched
branch point in this venv and have nothing to do with documents:
`test_calc_package_figure_conventions.py::test_large_pile_group_leaves_the_forces_to_the_table`
and `test_defaults_used.py::test_every_default_a_run_lists_changes_its_result[profile_figure|plot_data]`.

**Changed** (they patched or called moved internals):

- `deep/tests/test_document_tools_offline.py::test_without_the_tool_layer_the_tools_are_hidden`:
  also patches `planlens.review.conversation.available` (the dispatcher's).
- `deep/tests/test_wave2c_offline.py::test_an_unwritable_temporary_copy_is_said_plainly`:
  patches `planlens.review.marked_copy._temp_beside`.
- `deep/tests/test_wave2c_second_pass_offline.py::test_the_markup_check_look_has_its_cap`:
  the check runs under `host_for(engine)` with its look (the "check" cap
  still reaches the engine through `ask_vision`).
- `tests/test_conversation_isolation.py::test_idle_toolkits_close_and_their_handles_reopen`:
  the limit through `review.using(max_live_toolkits=1)`.
- `tests/test_find_like_tool.py`: the two direct `find_like` calls take the
  build's look (`host_for(engine).look`) instead of an engine.
- `tests/test_find_like_budget.py`: became binding tests (the setting read
  on every call; unset stops at 300 s, `0` reads every sheet, a large value
  leaves the search alone, each through `document_tools.find_like` with a
  build's host). **Removed from it, because planlens has the same test**
  (`planlens/review/tests/test_find_like.py`): `test_the_default_budget`,
  `test_a_loose_example_stops_at_the_budget_and_says_so`,
  `test_with_no_budget_every_sheet_is_read`,
  `test_a_search_inside_the_budget_is_unchanged`, `test_page_ranges` (same
  names there).

**New:** `deep/tests/test_review_binding_offline.py` (7): the process's
host is the app's; two builds made one after the other sign and look with
their own signer and engine; an unnamed author signs with the setting read
at call time; an engine that cannot see leaves marks unchecked; the build
host's look reads through its engine; `find_like` through the agent's own
tool (unverified with no engine); the Word/Excel reader's fallback.

**End to end, offline, fake engines, as before:** a marked copy and its
markup check through `make_vision_tools`' `annotate_document`
(`deep/tests/test_markup_check_offline.py`, its stand-in engine reading the
crops, and the new binding file); a `find_like` search through
`vision_tools._dispatch_find_like` with a fake reader
(`tests/test_find_like_tool.py`'s dispatcher tests) and through the agent's
tool (new file).

**Kept, though planlens now has copies:** the app's other tests of moved
code (the geometry parts of `test_vision_view.py`, `test_vision_turned.py`,
`test_vision_locations.py`, `test_vision_brief4.py`; `test_markup_check_offline.py`;
the page parts of `test_wave2c_offline.py`; `test_annotate_keys_offline.py`,
`test_wave2c_annotate_copy_offline.py`, `test_wave2b_documents_offline.py`;
`find_like`'s `parse_readings` / `_verdict` checks). They pass unchanged
through the aliases and most go through the agent's tools, so they cover
the binding too. Removing them was optional; the ones that call moved
functions directly by the old names should be pruned at step 16, when the
aliases go, against `planlens/review/tests/` (`test_frames.py`,
`test_markup_check.py`, `test_page_numbers.py`, `test_marked_copy.py`,
`test_conversation.py`, `test_find_like.py`).

## 5. What is left

- The planlens 0.15.0 release (owner's word), then: CI green here, the
  fingerprint recorded under 0.15.0 (section 3), the merge.
- The aliases (three stubs, `vision_view`'s imports, `document_tools`'
  re-exports and `__getattr__`) and the duplicate tests: step 16.
- The feature detection (`has_tool`, `search_supports_fuzzy`): one release
  after this one.
- Not moved (section 2.6 "proposed"): `measure_tool.py`, `scale_labels.py`,
  `dxf_notes.py`; they call the binding's process host.
- The binding moves on into the core at step 11 (c) (INF-32).

## 6. Release note (three lines)

The document tools (marked copies, the markup check, find_like, page numbers,
view frames) now run from planlens.review; the app hands planlens its look,
signer and file rules per conversation. No change to what the model sees or
what the tools return. Requires planlens 0.15 or later.

## 7. Review and fixes (lead, 2026-11-06)

An independent review (`C:/dev/tmp/pl29b_review/REVIEW.md`, local) found
no behaviour drift (33 scenarios through the agent's own tools, old and new
trees, identical after normalising markup ids), no leak between two
conversations in two threads, no unresolved import, and a planlens test for
every cut app test. Two changes:

- `funhouse_agent/tests/test_cited_documents.py`: the written-copy case
  stubbed `document_tools._write_marked_copy`, which nothing calls any more;
  it now stubs `planlens.review.marked_copy._write_marked_copy` (99 passed).
- `.github/workflows/tests.yml`: the numpy-1.26 job installs
  `planlens>=0.15` with `--no-deps`, like the pin.

Release order (from the review): both pages import `planlens.review` at the
top, so the app needs planlens 0.15 installed; every `--no-deps` install
(Foundry wheels, the FDE brief, the numpy-1.26 job) must ship planlens
0.15.0 with the app. planlens 0.15.0 can go out ahead of the app: the
released 5.34.0 does not import `planlens.review`, and its tool specs and
fingerprint are unchanged; the one change a 5.34.0 user sees is PL-11
(`Label (unit): value` read as a quantity), as intended.

## Refresh 2026-11-09

Branch `claude/pl-29b-refresh`, from `origin/claude/pl-29b-binding`
`9630642`, with `origin/master` (`1d219c8`, 138 commits since this branch's
base `05eab2d`) merged in as merge commit `40805b1` (not a rebase). The pin
stays `planlens>=0.15`; tests ran against the cloned planlens main checkout
(`34261bf`, `pip install -e --no-deps`). Master brought DR-36/37 (merge
`111f3a9`), DR-39 (merge `b9c2843`), EV-35 (`funhouse_agent/chart_eval/`) and
GM-309, among others.

### Conflicts and how each was resolved

| Where (in `40805b1`) | Kind | Resolution |
|---|---|---|
| `.github/workflows/tests.yml:149` (the numpy-1.26 job's `--no-deps` install) | text conflict, the only one | Both sides kept: master's `-e packages/aec-agent-core` and this branch's `"planlens>=0.15"` (section 7). |
| `pyproject.toml:219` | auto-merged | Master's own edits (comments, agent-stack caps) plus this branch's `"planlens>=0.15"` and its comment line; `git diff origin/master 40805b1 -- pyproject.toml` is that one line. |
| `funhouse_agent/vision_view.py:275` (`probe_fallback_note`) and `:542` (`view_payload`'s `vision_probe` line) | auto-merged (DR-36) | Both sit in what this branch KEPT in `vision_view` (budgets, `render_view`, `view_payload`); none of the 64 definitions moved to `planlens.review.frames` was touched by DR-36. A failed probe's fallback is still stated on every page, zoom and tile result. |
| `funhouse_agent/vision_tools.py:2232-2251` (`ask_vision` hands the turn's question through `turn_question.with_question`) | auto-merged (DR-37) | Kept as master has it. This branch's binding gives planlens its LOOK as `ask_vision` with the build's engine (`document_tools.py:397-398`, `host_for` at `:373`), so the markup check and `find_like` reach the model through the same function. Their kinds (`check`, `find_like`) are not in `turn_question.QUESTION_KINDS`, which is exactly master's behaviour: DR-36/37 left them out on purpose (`module_work/dr36_37/CHECK.md`, "Left out"). Every kind that goes back to the main model (`image`, `page`, `overview`, `region`, `tile`, `reference_figure`, `worked_example_page`) still carries the question. |
| `funhouse_agent/deep/vision_kit.py:559`, `:573` | auto-merged (DR-39 docstring) | Master's wording kept; this branch's change to the file (`annotate_document` and `find_like` passing the build's host) is in other lines. |
| DR-39's deleted files (`deep/sweep.py`, `deep/findings_tools.py`, `review_findings.py`, `deep/geometry_tools.py`, `deep/digest_tools.py`, `review_digest/`) | no conflict | This branch edited none of them, so git reports no modify/delete conflict and they stay deleted. |
| `funhouse_agent/document_tools.py:20`, `:466` (comments naming "the findings tools" DR-39 removed, and the coverage ledger) | stale text, not a conflict | Fixed in the DR-40 (b) commit `96a14d1` (the comments now name the cited-page check and `measure_tool`). |

No code fix was needed after the merge: the suites below pass on `40805b1`
as they do on master, and the page fingerprints pass unchanged.

### Nothing DR-39 removed survives

```
$ git grep -nE 'review_findings|findings_tools|geometry_tools|digest_tools|review_digest|deep\.sweep|build_review_agent|make_reader_tool|INLINE_DESCRIPTIONS|KEPT_READINGS_TOOL|DOCUMENT_REVIEW_PROMPT_LEAN|unopened_citations|GEOTECH_REVIEW_(SWEEP|FINDINGS|OVERVIEW|GEOMETRY|DIGEST|KEPT_READINGS)|lean_agent|record_finding|pdf_page_count|ORIENTATION_REQUEST_LONG' -- '*.py'
```
On `40805b1` this finds only what master itself has: DR-39's own
"removed" test (`test_review_harness_offline.py:137`, `:153`), docstring
notes (`deep/inline_images.py:7`, `deep/review_agent.py:8`,
`webapp/review_page.py:67`, `webapp/tests/test_orientation_overview.py:4`),
the name tables DR-39 kept for old records (`cited_pages.py` / `coverage.py`
`SWEEP_TOOLS`, `cited_documents.py:785`, `cited_pages.py:670`'s comment,
`test_review_minimal_offline.py:27`), and the three `GEOTECH_VISION_INLINE`
comments in `vision_tools.py` that DR-40 (a) then reworded. Over
`planlens/review` (the planlens checkout) the same grep, with `sweep_pages`,
finds one comment, `planlens/review/markup_check.py:1129` ("as sweep_pages
does"): listed for the lead in `module_work/dr40/CHECK.md` §3, not edited.

### Page fingerprints

`webapp/tests/test_page_fingerprints.py` on `40805b1`: 13 passed, no skip,
no re-record. The recorded `review_default`, `review_minimal` and
`geotech_default` builds are what this branch builds after the merge, so no
field differs from what section 3 already justified (and master's DR-39
re-record, which only dropped `review_lean`, merged cleanly).

### Tests on the merge commit

Python 3.12 venv as in `COMMON_MODULES.md`, planlens main editable, each
suite on pytest's own exit code, `-n 6`.

| Suite | master `1d219c8` | merge `40805b1` |
|---|---|---|
| `funhouse_agent/tests` | 4,285 passed, 32 skipped, 16 failed, exit 1 | 4,284 passed, 32 skipped, **the same 16 failed**, exit 1 |
| `funhouse_agent/deep/tests` | 1,091 passed, 4 skipped, exit 0 | 1,098 passed, 4 skipped, exit 0 |
| `funhouse_agent/review_eval` | 112 passed, exit 0 | 112 passed, exit 0 |
| `funhouse_agent/chart_eval/tests` | 33 passed, exit 0 | 33 passed, exit 0 |
| `report_ingest/tests` | (stopped, see below) | 4,856 passed, 23 skipped, exit 0 |
| `webapp/tests` | (stopped) | 1,155 passed, 5 skipped, exit 0 |
| `webapp/tests/test_public_text.py` | | 5 passed, exit 0 |
| `webapp/tests/test_page_fingerprints.py` | | 13 passed, exit 0 |
| planlens `planlens/review/tests` (planlens main; they import no app module, and `test_end_to_end` asserts so) | | 245 passed, 1 skipped, exit 0 |

The 16 `funhouse_agent/tests` failures are the same node ids on unmodified
master in this venv (`origin/master` 1d219c8 in its own worktree, diffed by
node id: identical). They are the reference-library and calc-package tests
DR-39 already recorded under geotech-references 1.5.0 from PyPI:
`test_citations::test_every_cited_number_is_in_its_manual`,
`test_calc_package_figure_conventions::test_large_pile_group_leaves_the_forces_to_the_table`,
`test_fhwa_soils_foundations_reference` (2), `test_reference_adapters`
REF-61/62 (10), `test_reference_example_ids::test_examples_come_from_the_reference_itself`
and `test_defaults_used[profile_figure|plot_data]`. Master's own
`report_ingest/tests` and `webapp/tests` runs were stopped at the session's
background time limit; both pass on the merge and the head, so no baseline
is needed for them.
