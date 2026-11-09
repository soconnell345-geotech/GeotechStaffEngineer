# DR-40: the rest of DR-39's strip (vision switches, coverage)

Branch `claude/pl-29b-refresh`, on top of PL-29 (b) brought up to master
(`module_work/pl29/CHECK_B.md`, "Refresh 2026-11-09"). BACKLOG DR-40, the
follow-up of DR-39 (`module_work/dr39/CHECK.md` sections 1 and 4). The owner,
2026-11-09: "strip down the non-beneficial parts of the package and finish
the split", with no reruns.

This is code, not a calculation: there are no printed pages to cite. Each
verdict cites the code and the measured runs.

## 1. What went, and why

| Switch | Default | Evidence (from `dr39/CHECK.md` §1) | Removed |
|---|---|---|---|
| `GEOTECH_VISION_TEXT_CONTEXT` | off | Run 3 `grounded` (lean + both switches): 34/35, breaks calc-bearing-consistency. Its only named arms went with lean in DR-39. | the text-layer block in every page / region / tile prompt; `_lines_for_context`; the `text_layer` on inline results; `vision_view.page_lines`, `text_context`, `TEXT_CONTEXT_CHARS`, `_UNMAPPED_LINE` |
| `GEOTECH_VISION_STRUCTURED` | off | Same run; "partly superseded by pixel boxes" (BACKLOG switch table). | the `LOCATED` request (`vision_view.with_locations`, `LOCATED_INSTRUCTION`, `LOCATED_PX_INSTRUCTION`), `split_located`, `vision_tools._split_located`, `_image_located`, `IMAGE_LOCATED_NOTE` / `_WIDE_NOTE`, `_fit_located` (it trimmed only `located` lists) |
| `GEOTECH_VISION_TURN_RETRY` | off | Built for VIS-01; never run on the suite (no results folder has the `turn_retry` arm; EV-13). | the TURNED line request, the second look (`_settle_turn`), the per-page turn memory (`_note_turn`, `remembered_turn`, `clear_turn_memory`, `_start_turn`, `_TURN_SEEN`), `turn_retry_failed` on results, `render_view(default_turn=)`, `image_view.frame_of` (only the memory used it), the `turn_retry` arm |
| `GEOTECH_VISION_PATCH_ALIGN` | off | Run 7 / brief 5 part F: undecided, "stay OFF" until ten repeats (EV-02). | `vision_view.patch_align`, `align_to_patches`, `PATCH_ALIGN_ENV`, `PATCH_PX`; LIVE_TEST_QUEUE check 7 marked retired |
| `GEOTECH_COVERAGE` | off | Run 7: 39/42 either way, at +47 % input; CV1-CV4 fixed and never re-measured (DR-23). | `deep/coverage_tools.py` (recorder, gate, `document_coverage`, the extraction step allowance); the ledger, inventory and gate note in `coverage.py`; `coverage_dir` on both builds; the assembly's coverage hooks; `webapp/core`'s held-back gated reply, `GATE_STATUS`, `coverage_gate_spoke` and extraction cap; the `coverage_gate` activity event; `cited_pages`' reading of `coverage.json`; the `coverage` arm |

Kept from those switches' surroundings, on purpose:
- the agent's own `turn` argument (on by default; VIS-01's lever) and the
  text-layer upright (`render_view(upright=True)`, VIS-05);
- the stale `GEOTECH_VISION_INLINE` comments in `vision_tools.py` now say
  "the minimal build" (the inline paths are the minimal build's);
- `coverage.py`'s page parsers (`pages_from_call`, `parse_pages`, `compact`,
  `TEXT_TOOLS` / `LOOK_TOOLS` / `SWEEP_TOOLS` / `GRID_TOOLS`), which
  `cited_pages.py` and `review_eval/checks.py` read tool calls with; their
  tests stay as `deep/tests/test_coverage_pages_offline.py`;
- the review suite's coverage TASKS (`report-extract-all`,
  `report-extract-diggs`) and `check_states_coverage`: they score the agent's
  own reads and answer, not the ledger;
- `activity_log`'s `step_allowance` field on `turn_end` (nothing sets it
  now; old records still carry it): noted, not fixed.

**The checklist first (DR-40 (b)'s condition).** `GEOTECH_REVIEW_CHECKLIST`
does not need the ledger: since RI-34 `report_ingest/checklist.py` reads
report ingest's record (`load_record` refuses anything else, including a
ledger, by type; `funhouse_agent/tests/test_report_checklist_shim.py`), and
`deep/agent.py` adds `make_checklist_tool` with no reference to coverage. The
`checklist` arm set `GEOTECH_COVERAGE` too; it is now
`{GEOTECH_REVIEW_CHECKLIST: "1"}`, so it measures the checklist alone. That
is the one arm whose meaning changes.

**PL-29 (b) makes (b) possible here:** on master `document_tools.py` imported
the ledger recorder; on this branch it is the binding to `planlens.review`
and imports nothing from `coverage` (grep below).

**Arms after this branch:** `baseline`, `minimal`, `checklist`,
`cited_pages_note`, `cited_pages_retry`, `figure_helper`, `plot_code`,
`figure_helper_off`, `plot_code_off`, `step_cap_answer`,
`cited_documents_note`, `cited_documents_retry`, `judgment_calls`,
`judgment_calls_off`. Removed: `turn_retry`, `coverage`.

## 2. Lines, per area

`git show --numstat` of each commit. "Tests" includes `webapp/tests/page_builds.py`.

| Commit | Area | Added | Removed |
|---|---|---|---|
| DR-40 (a) `671d56e` | code (`vision_tools.py` 91/425, `vision_view.py` 10/296, `review_flags.py` 8/49, `image_view.py` 1/9) | 110 | 779 |
| | tests | 50 | 400 |
| | docs (`funhouse_agent/DESIGN.md`) | 2 | 2 |
| DR-40 (b) `96a14d1` | code (`coverage.py` 22/1,044, `coverage_tools.py` 0/553, `webapp/core.py` 8/90, `cited_pages.py` 16/50, `assembly.py` 7/34, `review_flags.py` 12/29, `review_build.py` 12/25, `agent.py` 9/24, `activity_log.py` 4/13, `document_tools.py` 2/2) | 92 | 1,864 |
| | tests (`test_coverage_offline.py` 1,200 removed, 91 kept as `test_coverage_pages_offline.py`) | 120 | 1,327 |
| | docs | 4 | 6 |
| follow-up `c3c85be` | code (`vision_tools.py`: `_read_tiles`, uncalled on master too, left calling `_tile_jobs` the old way) | 0 | 7 |
| **Total** | | **378** | **4,385** |

## 3. Grep proofs

On the branch head, over `*.py` (app) and the planlens checkout:

```
$ git grep -nE 'VISION_TEXT_ENV|VISION_STRUCTURED_ENV|VISION_TURN_RETRY_ENV|vision_text_context|vision_structured|vision_turn_retry|PATCH_ALIGN|patch_align|align_to_patches|split_located|text_context\(|page_lines\(|clear_turn_memory|remembered_turn|_fit_located|turn_retry|_lines_for_context|frame_of|IMAGE_LOCATED|located_note' -- '*.py' ':!module_work'
```
Matches only in the two tests that assert the names are gone
(`deep/tests/test_review_harness_offline.py::test_the_unmeasured_vision_switches_are_gone`,
`tests/test_vision_turned.py::test_a_look_at_turned_strokes_is_one_call_that_asks_nothing_more`)
and the docstring notes saying they were removed.

```
$ git grep -nE 'coverage_tools|CoverageLedger|CoverageGate|coverage_kit|COVERAGE_ENV|review_flags\.coverage\b|GATE_PREFIX|GATE_STATUS|coverage_gate_spoke|geotech_coverage_ledger|geotech_extraction_recursion_limit|EXTRACTION_STEP_LIMIT|ALLOWANCE_KEY|document_coverage|coverage_dir|ledger_path\(' -- '*.py'
```
Matches only in removal notes (docstrings of `coverage.py`, `agent.py`,
`review_flags.py`, the parser test file) and in
`test_report_checklist_shim.py`, whose stand-in class is named
`CoverageLedger` so the checklist's refusal message is still pinned.

Each vision_view helper named in the brief was removed only after a grep of
the whole repo for callers: `text_context`, `split_located`, `page_lines`,
`with_locations`, `patch_align`, `align_to_patches`, `LOCATED_*`,
`TEXT_CONTEXT_CHARS`, `PATCH_PX`. The other hits of those words are other
things: `read_document`'s `with_locations` argument (a planlens tool
parameter, in tool descriptions) and local variables named `page_lines` in
`report_ingest/log_reader.py` and `log_holes.py`.

**In `planlens/review` (not edited; for the lead):** only the removed retry
used these, and they stay there and as the app's step-16 aliases in
`vision_view`:
- `planlens/review/frames.py:620-660`: `TURN_TAG_INSTRUCTION`, `_TURN_TAG`,
  `read_turn_tag` (and their test, `planlens/review/tests/test_frames.py:316`);
- `planlens/review/frames.py:605`: `turned_note`'s `"remembered"` wording (no
  caller passes `turn_from="remembered"` any more);
- comments: `planlens/review/markup_check.py:1129` (names `sweep_pages`,
  removed in DR-39) and `planlens/review/conversation.py:1307-1308` (names
  "the host's coverage ledger").

## 4. The default pages are unchanged

`webapp/tests/test_page_fingerprints.py` passes on each commit with no
re-record (no skip): every removed switch was off by default, and no tool
description or prompt text changed. The `vision_kit` docstring that says
"LOCATED" is a tool description ("an earlier vision result LOCATED") and was
left as it is.

## 5. Tests

venv as in `COMMON_MODULES.md` (Python 3.12, `-e packages/aec-agent-core
-e .`), with planlens from the cloned main checkout (`pip install -e
../planlens --no-deps`, 0.14.0 by its label, carrying `planlens.review`),
geotech-references 1.5.0 from PyPI. Each suite judged on pytest's own exit
code, `-n 6`.

| Suite | Result on the branch head (`c3c85be`, code as pushed) | Exit |
|---|---|---|
| `funhouse_agent/tests` | 4,263 passed, 32 skipped, 16 failed (the same 16 node ids as master) | 1 |
| `funhouse_agent/deep/tests` | 1,043 passed, 4 skipped | 0 |
| `funhouse_agent/review_eval` | 110 passed | 0 |
| `funhouse_agent/chart_eval/tests` | 33 passed | 0 |
| `report_ingest/tests` | 4,856 passed, 23 skipped | 0 |
| `webapp/tests` | 1,153 passed, 5 skipped | 0 |
| `webapp/tests/test_public_text.py` | 5 passed | 0 |
| `webapp/tests/test_page_fingerprints.py` | 13 passed, no skip, no re-record | 0 |
| planlens `planlens/review/tests` | 245 passed, 1 skipped | 0 |

Fewer tests than the merge: the removed switches' tests (section 1). The 16 `funhouse_agent/tests` failures are the same node ids on unmodified
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

## 6. Release note

None for users. Every removed switch was off by default; both pages build
and answer exactly as before (the fingerprints prove the builds).
- **Notebooks / suite runs:** setting one of the removed variables now does
  nothing; asking for the `turn_retry` or `coverage` arm gets the runner's
  "unknown arm" error. The `checklist` arm no longer turns on the coverage
  ledger.
- **Library callers:** `build_deep_agent` and `build_review_page_agent` no
  longer take `coverage_dir`; `funhouse_agent.deep.coverage_tools` is gone and
  `funhouse_agent.coverage` keeps only its page parsers; the vision helpers
  listed in section 1 are gone. A grep of the repo found no other caller.

## 7. Noted, not fixed

- `activity_log.note_step_cap`'s `allowance` and `turn_end`'s
  `step_allowance` stay, unused.
- The removed switches are still named in history documents
  (`module_work/REVIEW_HARNESS.md`, `harness_theory/*.md`,
  `LIVE_TEST_QUEUE.md` check 7); each now opens with a dated DR-40 note, as
  DR-39 did.
- BACKLOG's switch table rows for these switches (`GEOTECH_VISION_TEXT_CONTEXT`
  / `_STRUCTURED`, `GEOTECH_COVERAGE`, `GEOTECH_VISION_TURN_RETRY`,
  `GEOTECH_VISION_PATCH_ALIGN`) and EV-02 / EV-13 / DR-23, which name them,
  are the lead's to fold in.
