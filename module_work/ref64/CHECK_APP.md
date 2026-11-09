# CHECK: REF-64 part (A), the app half (local reference packs)

Branch `claude/ref-64-app`, from app master `9a4c552`. Plan of record:
`module_work/REFERENCE_PACKS_PLAN.md` sections 2.4 to 2.8 (GSE rows and the
app tests in 2.8); what the refs half hands over: `CHECK_REFS.md`, "What the
app half needs" and "Review fixes". Every test value is made up (a fictitious
"Example Practice Manual" about a fictitious fill, zorblite, built in
`tmp_path`). No real manual or non-public document is used. Nothing merged,
tagged or released; only this branch is pushed.

## What was built

| Piece | Where |
|---|---|
| **Figure reads (plan 2.5).** A catalog figure's `bbox` goes to `vision_view.render_view(..., bbox=)` as a crop; the kwarg is passed only when the record has a box, so a built-in figure's render call is exactly as before | `funhouse_agent/reference_vision.py:226-237` |
| `read_notes` appended to the read prompt under a fixed heading, `READ_NOTES_HEADING` = "Notes from the reference's catalog on reading this figure"; only when the record has notes, and before the code voter's read lines | `funhouse_agent/reference_vision.py:180-182`, `:276-280` |
| A pack figure whose PDF is in neither the pack folder nor `GEOTECH_REFERENCES_DOCS`: refs raises `FileNotFoundError`, and the existing fallback asks the host's fetcher (`reference_docs.fetch`) by the catalog's bare file name. No code change; pinned by a test | `reference_vision.py` (existing `except FileNotFoundError`) |
| **Feature detection and pack access**, one place: `_packs_module()` is `geotech_references._packs` only when `hasattr(_retrieval_db, "pack_dirs")` (refs 1.6.0+), else None; `pack_records()`, `pack_problems()`, `packs_line()` all return empty then | `funhouse_agent/reference_packs.py:23-76` (new) |
| **The packs line (plan 2.4)**: "This deployment adds: <title> (`local_x`), ...", appended to the librarian's system prompt (`\n\n` + line) and to its description (`" "` + line), only when non-empty | `funhouse_agent/deep/agent.py:265-276` (in `build_references_subagent`) |
| **The record (plan 2.7)**: `geotech_build["reference_packs"] = _packs.pack_records()` (`id, title, reference_id, edition, files, sha`) after `record_build`; absent with no packs | `funhouse_agent/deep/agent.py:1406-1422` (`_record_reference_packs`) |
| **Diagnostics (plan 2.6)**: the reference-PDF row reads catalogs through `_figures_db._iter_catalogs()` when packs are supported (built-ins then packs; a pack's catalog is labelled with its pack id), counts files in each pack's folder as present, and adds one line per skipped pack ("reference pack skipped: ...") from `_packs.problems()` | `webapp/geotech_page.py:237-270` (`_figure_catalogs`, `_pack_folder_files`), `:280`, `:370-374`, `:415` |
| **Pin**: `geotech-references>=1.6.0` | `pyproject.toml:40` |
| **Submodule** pointer to refs main `76f494d`. The submodule is not checked out in this clone, so the gitlink was set with `git update-index --cacheinfo 160000,76f494d…,geotech-references` | commit `e3b17e8` |
| **Docs**: where packs go on Foundry (`references/packs` beside the app file, `setdefault` in the app file) and Databricks (a Workspace folder or volume, set before `run_on_databricks`), what the app shows of them, never commit one, link to the refs repo's `docs/REFERENCE_PACKS.md` | `docs/FOUNDRY.md` §3d; `webapp/README.md` §3 "Your deployment's own references"; `docs/DATABRICKS_INSTALL.md` §9 |
| **Tests** | `funhouse_agent/tests/test_reference_packs.py` (13), `webapp/tests/test_no_reference_packs.py` (1) |

The other `>=1.5.0` mentions (the 5.34.0 history row and §1 of
`DATABRICKS_INSTALL.md`, CLAUDE.md) describe the 5.34.0 release and were left
as they are; CLAUDE.md is the lead's.

Files the app split's step 11 (b) moves were not touched (no change to the
figure helper, the plotting box, the chart stack or the worked examples;
`reference_vision.py` imports `chart_reading` as before).

## Unchanged without packs: the proof

1. **Fingerprints.** `webapp/tests/test_page_fingerprints.py` passes and
   `git diff origin/master -- webapp/tests/page_fingerprints.json` is empty.
2. **The librarian spec, byte for byte.** A script (every `GEOTECH_*`
   cleared) dumped `build_references_subagent()`'s description, system prompt
   and tool names as JSON, once from an unmodified master worktree
   (`9a4c552`) and once from this branch: `cmp` identical (2,497 bytes).
3. **A built-in figure's prompt and render.**
   `test_builtin_figure_prompt_and_render_unchanged` pins dm7_2 Figure 4-12's
   whole read prompt as a literal and asserts the render call's keywords are
   exactly `{allow_jpeg, engine}` (no `bbox`), with packs unset and with a
   pack set. The same test, run against master's `reference_vision.py`
   (PYTHONPATH on the master worktree), passes: 2 passed, exit 0. So the pin
   is master's prompt.
4. **The build record.** `_record_reference_packs` adds nothing when
   `pack_records()` is empty (test).
5. **Diagnostics.** With no packs, `_figure_catalogs()` returns the
   package's catalogs (refs' `_iter_catalogs()` is the old glob plus an empty
   pack list; refs' own unset proof), `_pack_folder_files()` is empty and no
   skipped-pack line is added.
6. **Older refs.** With `pack_dirs` removed from `_retrieval_db` (as on refs
   1.5.0) and a pack set, the line, the records and the problems are all
   empty (`test_no_packs_on_older_refs`).

## Tests, with exit codes

Environment: Python 3.12 venv,
`pip install -e packages/aec-agent-core -e ../geotech-references -e . pytest pytest-xdist`
(refs at `76f494d`, planlens 0.15.0, PyMuPDF installed, so no test skipped
for it).

| Run | Result | Exit |
|---|---|---|
| `pytest funhouse_agent/tests/test_reference_packs.py` | 13 passed | 0 |
| Reference and librarian tests: the 24 test files naming `reference_db_adapter`, `figure_db` or `read_reference_figure` (list below) | 652 passed, 5 skipped (live tests) | 0 |
| `webapp/tests/test_public_text.py` + `test_page_fingerprints.py` | 18 passed | 0 |
| `webapp/tests/test_no_reference_packs.py` | 1 passed | 0 |
| `pytest -n 8 funhouse_agent/tests funhouse_agent/deep/tests webapp/tests` | 6,439 passed, 39 skipped, 2 failed | 1 |
| The 2 failures on unmodified master (worktree `9a4c552`, same venv) | 2 failed, the same assertions | 1 |
| refs `tests/test_packs.py` in this venv | 64 passed | 0 |

The two failures, which this branch does not touch:
`test_calc_package_figure_conventions.py::test_large_pile_group_leaves_the_forces_to_the_table`
(the figure writes every pile force instead of the "Too many piles" note)
and `test_defaults_used.py::...[profile_figure|plot_data]` (`logx`/`logy`
change nothing in the result). Both are plotting checks, and both fail the
same way on master in this environment. They are probably caused by the
matplotlib version the fresh resolve picked; not investigated (not this
item).

Reference/librarian test files: `funhouse_agent/deep/tests/`
`test_deep_phase2_offline`, `test_deep_tools_offline`,
`test_deep_truncation_offline`, `test_measure_tool_offline`,
`test_review_harness_offline`, `test_review_switches_offline`;
`funhouse_agent/tests/` `test_catalog_routing_hints`, `test_cited_documents`,
`test_fhwa_soils_foundations_reference`, `test_figure_db_adapter`,
`test_live_figure_readoff`, `test_read_reference_figure`,
`test_reference_docs`, `test_reference_packs`, `test_reference_result_sizes`,
`test_render_region`, `test_reviewers`, `test_top_level_params`,
`test_vision_locations`, `test_vision_probe`, `test_vision_view`,
`test_worked_examples`; `webapp/tests/` `test_core_move_11a`,
`test_helper_models`.

What `test_reference_packs.py` covers: on a pack figure with a fake vision
engine, `bbox` reaches `render_view` and `read_notes` ends the prompt under
the heading; a pack figure without them renders the whole page with no
heading; the built-in prompt and render are pinned (packs unset and set);
with PyMuPDF, a real render of the pack's PDF is cropped to the box (padded);
a missing pack PDF goes to the host fetcher by its bare name and keeps the
crop; the packs line, and the librarian's prompt and description, only with
packs; nothing on a refs without `pack_dirs`; `reference_packs` in the build
record only with packs; diagnostics list a pack PDF under its pack id, find
it in the pack folder, and name a skipped pack.

## Release note (for HANDOFF)

> **Local reference packs (REF-64, the app half; needs geotech-references
> 1.6.0, now the pin).** A deployment can add its own manuals to the reference
> library: point `GEOTECH_REFERENCES_PACKS` at a pack folder or a folder of
> packs (the format and the `python -m geotech_references.packs check` command
> are in the refs repo's `docs/REFERENCE_PACKS.md`; where packs go on Foundry
> and Databricks is in `docs/FOUNDRY.md` §3d and `webapp/README.md` §3). The
> references helper then searches and cites them like the built-in
> references. Its prompt and its description to the main agent carry one
> line, "This deployment adds: <title> (`local_x`), ...". `read_reference_figure`
> crops to a catalog figure's `bbox` and adds its `read_notes` to the read
> prompt. Each turn's `turn_start` records the packs in force
> (`reference_packs`: id, title, files, sha256), and the Connection
> diagnostics' reference-PDF row lists pack PDFs and names any skipped pack.
> With no packs set, every prompt, tool description, figure read and record
> is unchanged (page fingerprints unchanged). No calculated value changes.

## Noted, not fixed

- With packs set, the diagnostics row no longer gives the "not set" message
  when the only PDFs found are in pack folders. It reports the built-in PDFs
  as missing instead (WARN/FAIL with the list), which also tells the user
  what to upload.
- A pack's `bbox` is passed to `render_view` with its default 10 % padding
  (`pad_frac=0.1`), as `render_region` pads a box. A tighter crop would need
  its own setting; nothing in the plan asks for one.
- The packs line uses `reference_title` from `pack.json`. A pack with no
  title is named by its id alone.
