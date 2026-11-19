**Verdict: MERGE AFTER FIXES, the one fix is on this branch** (`claude/rv-b15-dxf-plot`, e4215ba on top of the reviewed head 43e31ed).

Review of B15 (`GEOTECH_REVIEW_DXF_PLOTS`, DR-19). Venv (a), Python 3.12, planlens 0.17.0 from PyPI, ezdxf 1.4.5, pymupdf 1.28.2. Before = 9367934 in a second worktree (its core and package first on `PYTHONPATH`); after = the branch. Every drawing was made up in the scripts.

## Findings

| # | What | Made-up input | Before (9367934) | After (43e31ed) | Status |
|---|---|---|---|---|---|
| 1 | `open_document` said two things of one DXF: the plot line, and a note ending "There are no pages here: ... ask the user for a PDF plot of it". | `shed.dxf`, three lines | both said | both said | **Fixed here** (e4215ba): with the plot made, the note ends "The DXF itself has no pages: to LOOK at the drawing, use the plot made here (`plot`)." |
| 2 | planlens gives the plot line before any plot is made, so a DXF it then refuses to plot was promised one; every page tool then answered "could not plot". | empty model space; 60,000 lines (over the 50,000 limit); 20,000 lines and 2,000 texts (30 s limit met) | line given, tools fail | same | **Fixed here**, same commit: `open_document` makes the plot when it has page tools to name; refused, it gives no plot line and planlens' note stands. |
| 3 | The plot line names only tools of the build. | default page and its reading helper, drawing, plan and sheet tools each on and off (8 builds each) | `render_page` named, in no list (16 of 16) | all five named tools in every list (16 of 16) | Holds. |
| 4 | The row never says `blank`, and claims nothing wrong. | text only; three lines far from the origin (easting 512000); millimetres | `blank` | `dxf_plot`, with `n_entities` 2 / 3 / 3, the extent in drawing units, `mm` where set | Holds. Counts agree with what was drawn (also 22,000 of 22,000). |
| 5 | A denser plot keeps planlens' kind, with the `plot` entry beside it. | three lines and a circle; 5,000 lines; 22,000 entities | `mixed` / `figure` / `drawing_sheet` | the same, plus `plot` | Holds; the row says what the page is through `plot`. |
| 6 | `document_page_map` takes `kind` as a filter and knows no `dxf_plot`. | text-only DXF | - | `kind="dxf_plot"`: no rows. `kind="blank"`: the row, shown as `dxf_plot` | **Noted.** A model that filters by the kind it was shown gets nothing. Needs the kind in planlens. |
| 7 | The thumbnail's drawn label still reads `blank`. | text-only DXF | `pg 0 \| PDF 1 blank` | same (seen in the image) | **Noted.** planlens draws it. |
| 8 | The reading helper's `open_document` description names `annotate_document`, which the helper lacks. | default page, setting on | same | same | **Noted** (the builder's own note is true). Moving it moves a description. |
| 9 | Each map or structure call on a DXF runs `open_document` again, and that is not kept. | 5,000 / 12,000 / 22,000 entities | - | about 0.2 s / 0.7 s / 1.5 to 2.1 s more per call | **Noted**, the cost asked for. On a refused plot nothing is added (the error returns first). |

## Question 3: who else reads a page's kind

- The core and Document Review: nothing. No code outside `_dxf_plot_said` branches on a page-map row's kind (searched `"blank"`, `kind`, `kinds` in both packages). The photograph-page check (`vision_tools._photo_page`) reads the page's pixels, the cited-pages record and the image budgets read calls, not kinds.
- The rename is made on the result on its way to the model, so inside planlens the page stays `blank`. Its own readers are unchanged from before: no `look: true` on the row (`tools/toolkit.py`), no duplicate hash (`document/imagehash.py`, `document.py`), the `kind` filter (finding 6) and the thumbnail label (finding 7).

## Question 4: unset, or not plotted

- Unset: `open_document` has no `plot`, the page tools answer as before, the note is planlens' own (my third test; the builder's sixth).
- Set, planlens too old: `plots` is false in `make_vision_tools`, so no result takes the new path (read, not run: only 0.17.0 was installed).
- Set, plot refused: the map returns planlens' error unchanged; with the fix `open_document` no longer promises the plot.

## Question 5: the planlens finding

True on both counts, confirmed in the installed 0.17.0: `review/conversation.py` writes `render_page` into the line for every host, and `document/classify.py` has no kind for a plot. What the core says without it is honest for the row and the line; findings 6 and 7 are where it still shows through.

## What the fix costs (a call for the lead)

The plot is now made at `open_document` instead of at the first page tool: 1.2 s more at 5,000 lines, 2.8 s at 12,000, and up to planlens' 30 s limit on a drawing near it (20,000 lines with 2,000 texts met the limit on one try and passed on the next). planlens keeps a made plot, so the first page tool is then quick; a refused plot is not kept and is tried again by each page tool. A model that only reads the DXF's text pays for a plot it never looks at. Source change: 19 lines in `vision_kit.py`.

## What I ran (exit codes)

- Builder's new tests on the base: 5 failed, 1 passed (exit 1), as CHECK.md says. On 43e31ed with the b14 files, fingerprints and boundary: 38 passed, 1 skipped (exit 0).
- `tests/test_rv_b15_dxf_plot.py` without the fix: 2 failed, 1 passed (exit 1). With it: 3 passed.
- On this branch, from `packages/aec-document-review`: `test_rv_b15_dxf_plot.py test_b15_dxf_plot_lines.py test_b14_review_fixes_offline.py test_b14_wiring_offline.py test_review_fingerprints.py test_boundary.py`: 41 passed, 1 skipped (exit 0). Fingerprints did not move.
- `--noconftest webapp/tests/test_public_text.py webapp/tests/test_review_freeze.py`: 13 passed (exit 0). `packages/aec-agent-core/tests/test_rvb_measure_under_build_host.py`: 2 passed (exit 0).
- One existing assertion edited, untrue by design after the fix: `test_b14_wiring_offline.py::test_on_the_page_tools_and_the_looks_take_a_dxf_as_its_plot` compared the whole `open_document` result with the setting on and off; it now expects the note to differ.

## CHECK.md

Its statements held where I checked them. One to tighten: a drawing "of few entities" is not always `blank`; three lines and a circle came out `mixed` before and after.

## Not checked

- `webapp/tests/test_page_fingerprints.py` and the full suite: venv (b) was not built. Push as `ci/...` before merging.
- The `minimal` build: the tool lists I captured for it (7 and 4 tools) hold no `open_document`, so I could not call it there; that the plot never reaches it rests on reading `minimal_agent.py`.
- `analyze_pdf_page` and `render_region` on the plot were run only by the builder's b14 test (passing), not on my drawings.
- An older planlens, a model, and two testers at once on one DXF.
