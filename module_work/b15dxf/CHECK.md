# B15: what the page says of a DXF's plot (DR-19's two noted limits)

Branch `claude/b15-dxf-plot`, from `claude/b13-integration` at 9367934 (1.3.0). Repair commit 6294601; this file alone in the commit after it. Venv (a) of COMMON_DOCREVIEW.md, planlens 0.17.0 from PyPI, ezdxf 1.4.5, Pillow 12.3.0, Python 3.12. Both halves are done.

## Confirmed on the head, before changing anything

A made-up DXF (a closed polyline, a line, one text), `make_vision_tools(..., dxf_plots=True)`:

- **The line.** `open_document` returned `"plot": "render_page, render_region and the other page tools take this file too: ..."`. The words are planlens' own (`planlens/review/conversation.py`, `_side_document`, the `name == "open_document" and plots` branch). `render_page` is a tool of planlens' own server; no build here has it: the default page, its reading helper and `minimal` look at a page with `analyze_pdf_page` and `render_region`.
- **The row.** `document_page_map` returned `"kind": "blank"` for page 0, and `document_structure` a segment titled `blank` with `kinds: {"blank": 1}`. Decided in planlens: `planlens/document/classify.py`, `classify_page`, first rule (`chars < 5 and n_vector_paths < 5 and image_coverage < 0.02`). The plot's lettering is drawn as shapes (no text) and a drawing of few entities makes fewer than 5 paths.
- **`minimal`.** `minimal_agent.py` calls `make_vision_tools` without `dxf_plots`, so the plot never reaches it; nothing to repair there.

## What changed (both are repairs; no new setting)

`packages/aec-agent-core/aec_agent_core/deep/vision_kit.py`, 83 lines:

- `_dxf_plot_said` (new) and `_PLOT_PAGE_TOOLS` (new); `make_vision_tools._dispatch_tool` passes a document tool's result through it only when the build's `plots` is true (the setting on and the installed planlens able).
- (1) `open_document`: the `plot` line is rewritten to name those of `document_page_map, render_page_thumbnails, document_markups, analyze_pdf_page, render_region` that are in the build's `include`, then planlens' `PLOT_NOTE` as before. No tool that writes is named: the reading helper is handed the page's own tool objects (one line for both) and has no `annotate_document`. A build with none of the five gets no `plot` line.
- (2) `document_page_map` and `document_structure` on a DXF (by file name or `doc_` handle): each row or segment gets `plot: {of, n_entities, extent, drawing_units, what}`, the first four read from planlens' own `open_document` result for the file, `what` = `PLOT_NOTE`; a kind `blank` becomes `dxf_plot`, a segment title `blank` becomes `DXF plot`. Nothing is guessed: if the header cannot be read or carries no `plot`, the result is returned as planlens gave it.

Before: `{"page":0,"kind":"blank","words":0,...}`. After: `{"page":0,"kind":"dxf_plot","words":0,...,"plot":{"of":"yard.dxf","n_entities":3,"extent":[0.0,0.0,48.0,20.0],"drawing_units":"m","what":"a PDF plot of the DXF's model space made here (...)"}}`.

## Tests

New file `packages/aec-document-review/tests/test_b15_dxf_plot_lines.py`, under Document Review's tests and not the core's, because half (1) needs the page and its reading helper as built (`tests.page_builds`). On the head (the source change stashed): 5 failed, 1 passed. On the branch: 6 passed.

| Test | On the head |
|---|---|
| `test_the_plot_line_names_only_tools_of_the_kit` | fails |
| `test_the_plot_line_names_only_tools_of_the_page_and_its_reader` | fails |
| `test_a_build_with_fewer_page_tools_is_told_of_those_only` | fails |
| `test_the_page_map_says_the_page_is_a_plot_of_the_model_space` | fails |
| `test_the_structure_says_so_too` | fails |
| `test_a_pdf_is_mapped_as_before_and_unset_nothing_is_said` | passes: a guard that an empty PDF page stays `blank` and that with the setting unset nothing is said |

Runs (venv a, each judged by exit code 0; no existing test was edited):

- From `packages/aec-document-review`: `pytest tests/test_b15_dxf_plot_lines.py tests/test_b14_review_fixes_offline.py tests/test_b14_wiring_offline.py tests/test_review_fingerprints.py tests/test_boundary.py`: 38 passed, 1 skipped.
- From the root: `pytest --noconftest packages/aec-agent-core/tests/test_rvb_measure_under_build_host.py`: 2 passed. With the two b14 files above, that is every file `git grep -l "DXF_PLOTS\|dxf_plot" -- "packages/*/tests"` lists.
- `pytest --noconftest webapp/tests/test_public_text.py webapp/tests/test_review_freeze.py`: 13 passed.
- **Not run:** `webapp/tests/test_page_fingerprints.py` (venv b was not built in the time). The change touches no prompt, description, schema or setting.

## Fingerprints

Not moved; `tests/test_review_fingerprints.py` passes unchanged. With the setting unset `plots` is false and no result takes the new path.

## Calls made

- The line names only tools that write nothing, so one line is true for the page and its helper. The marked copy (`annotate_document`) is still named in the page's `open_document` description.
- `blank` becomes `dxf_plot`; any other kind planlens gives a denser plot (`figure`, `mixed`, `drawing_sheet`) is kept, with the `plot` entry beside it.
- `document_structure` is repaired too: it said `blank` of the same page.
- The entity count, extent and units come from a second `open_document` call per map call, under the build's host.

## Release note

With `GEOTECH_REVIEW_DXF_PLOTS` on: `open_document`'s plot line names the page's own tools, and the page map and structure say a DXF's page is a plot of its model space, with its entity count, extent and units. Unset: nothing changes.

## For the lead

- Push as `ci/...` for the full suite, which also runs `webapp/tests/test_page_fingerprints.py`.
- Finding for a later planlens change: `planlens/review/conversation.py`, `_side_document` writes the line with planlens' server tool names for every host; `planlens/document/classify.py`, `classify_page` has no kind for a plot, and the opened plot does not carry what it is to the page map (`planlens/dxf/plot.py`, `dxf_to_pdf`). With both fixed there, `_dxf_plot_said` can go.
- Not measured on a model (brief 9, EV-45, arm `dxf_plots`).

## Noted, not fixed

- With the setting on, `open_document`'s `note` for a DXF still ends "There are no pages here: to LOOK at the sheet, or for a marked-up PDF, ask the user for a PDF plot of it" (planlens' header note), which contradicts the plot line beside it.
- The reading helper shares the page's `open_document` description, whose rewrite (`_DXF_PLOT_REWRITE`) names `annotate_document`, a tool the helper lacks.
- `render_page_thumbnails`' thumbnail label for the plot's page is drawn by planlens and was not checked for the word `blank`.
