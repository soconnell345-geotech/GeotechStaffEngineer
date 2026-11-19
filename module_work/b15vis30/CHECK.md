# CHECK B15 / VIS-30, first slice: tiles cut along the sheet's own regions (planlens only)

Branch `claude/b15-vis30-region-tiles` from `main` 7eed79b (v0.17.0). Not wired: nothing calls the new function.

## Confirmed on today's code, before changing it

- `frames.tile_grid` picks 1 to 4 tiles per side from the page's small lettering (`text_px`, or 4.3 pt assumed under 40 characters of text); `frames.tile_boxes` cuts n x n with 8% overlap. A tile's pixel size is the budget's: `Document.render(budget=)` sizes a clip with `budget.fit_size`, and the core holds the budget's edge to `max_px` (2,048).
- The core (`vision_tools._tile_count`, `_tile_jobs`, `_tile_legibility` in the clone) asks for `tile_grid(info)` tiles and keeps the legibility note per tile under 12 px.
- VIS-30's arithmetic holds and is slightly worse with the default `openai-high` budget, whose patch count binds before its edge: on 36 x 48 in a 4 x 4 tile shows 4.3 pt lettering at 8.1 to 8.5 px (the backlog says about 9, from the edge alone).

## What changed (all additive)

- `planlens/review/frames.py`: `region_tiles(marks, budget, clip=None, max_tiles=16, overlap=0.08)` returning `RegionTiles(boxes, lettering_px, level, under, method, capped)`; `page_marks(doc, page)` returning `PageMarks` (text lines with lettering height, drawn pieces, drawn paths); helpers `_gaps`, `_in`, `_through`, `_cuts`, `_with_overlap`; five constants. `tile_boxes` and `tile_grid` are untouched. No repair, no setting, no tool.
- `planlens/testing/esize_fixtures.py`: `build_esize_sheet("right" | "bottom")`, two made-up 36 x 48 in sheets (PyMuPDF, as the other fixtures), with the known box of every dimension string and of the cloud.
- `CHANGELOG.md` (Unreleased) and `planlens/review/DESIGN.md` ("Tiles cut along the sheet's own regions": how the regions are found, what is guaranteed, what is not).
- Both levels are delivered. Size: 351 lines added to `frames.py` (docstrings and comments included), plus 239 for the fixture: over the "about 350" if the fixture counts.

## The comparison (budget `openai-high`; regenerate with `python -m planlens.review.tests.test_b15_vis30_region_tiles`)

Each text line is taken in the tile that holds it whole and shows it largest. The uniform grid is the one the core picks today (`tile_grid` on the whole-page render).

| Sheet | Tiles cut by | Tiles | Smallest lettering, px | ... in each tile, least to most | Lines under 12 px | Text lines cut by a tile edge | Text lines whole in no tile | Dimension strings whole in no tile |
|---|---|---|---|---|---|---|---|---|
| E-size, title block right | uniform grid | 16 | 8.1 | 8.1 to 17.7 | 473 | 263 | 0 | 4 |
| E-size, title block right | regions | 16 | 10.0 | 10.0 to 16.5 | 276 | 0 | 0 | 0 |
| E-size, title block right | regions, ceiling 40 | 22 | 12.2 | 12.2 to 27.5 | 0 | 0 | 0 | 0 |
| E-size, title block bottom | uniform grid | 16 | 8.1 | 8.1 to 17.0 | 437 | 5 | 0 | 8 |
| E-size, title block bottom | regions | 16 | 10.3 | 10.3 to 21.2 | 455 | 0 | 0 | 1 |
| E-size, title block bottom | regions, ceiling 40 | 29 | 12.1 | 12.1 to 37.2 | 0 | 0 | 0 | 2 |
| 34 x 22 in sheet-set fixture | uniform grid | 9 | 12.9 | 12.9 to 20.2 | 0 | 2 | 0 | n/a |
| 34 x 22 in sheet-set fixture | regions | 11 | 13.1 | 13.1 to 36.2 | 0 | 0 | 0 | n/a |
| 24 x 36 in review fixture | uniform grid | 16 | 32.1 | 32.1 to 38.4 | 0 | 1 | 0 | n/a |
| 24 x 36 in review fixture | grid | 16 | 32.1 | 32.1 to 38.4 | 0 | 1 | 0 | n/a |

Read it plainly:
- At the same 16 tiles the region tiles do NOT bring an E-size sheet over the line: the smallest lettering goes from 8.1 to about 10 px, and on the second sheet more lines are under 12 px than with the grid (455 against 437; they sit at 10.3 to 11.9 px where the grid has them at 8.1 and up). The result says so (`capped`, `under`). It takes 22 and 29 tiles to bring every line over the line.
- No text line is cut by any tile edge (the grid cuts 263 and 5). Dimension strings whole in no tile: 0 and 1 against 4 and 8. The ones lost are longer than a tile can take in (the string across the whole width of a view that had to be cut; a second one at 29 tiles).
- On a sparse 34 x 22 in sheet the regions take more tiles than the grid (11 against 9) for the same legibility.
- The existing 24 x 36 in fixtures (the review document's sheet, the scaled sheet) hold under 40 characters of text, so the answer is today's grid, box for box. The 34 x 22 in row is there to show a fixture with text.
- **A scan** (no text layer): `method="grid"`, exactly `tile_boxes(clip, tile_grid(...))`, `lettering_px` at the assumed 4.3 pt, and `under` names every tile under the line (all 16 on E-size). `marks=None` with a `clip` does the same.
- **A letter page, one column of text**: `method="whole"`, one tile that is the page; the grid is also 1.

## Made-up before and after

Sheet 1, the general notes (4.3 pt, a column 410 pt wide). Before: grid tiles r1c4 to r4c4, each about 933 x 700 pt with the title block in it, notes at 8.5 px, and 263 lines on the sheet cut by seams. After: the notes column on its own in four tiles cut between notes, at 11.3 to 13.5 px, no line cut; the tile at 11.3 px is in `under`.

## Tests

New file `planlens/review/tests/test_b15_vis30_region_tiles.py`, 29 tests (13 functions). On `main` the whole file fails at import (`planlens.testing.esize_fixtures` and `frames.region_tiles` do not exist), so every test in it fails there, including `test_tile_boxes_is_what_it_was`, which pins the old function and would pass alone.
- `/tmp/pl/bin/python -m pytest -q -p no:cacheprovider planlens/review/tests/test_b15_vis30_region_tiles.py`: 29 passed.
- The same with `test_frames.py`, `test_host_specs.py`, `test_tools.py` (review) and `test_toolkit.py`, `test_mcp_server.py` (tools): 222 passed, 1 skipped, exit 0. Alone: `test_frames.py` 63 passed.
- The default surface: `planlens/review/tests/test_host_specs.py` (the specs word for word against `app_tool_specs.json`), 26 passed, nothing re-recorded. No existing test was edited.
- Not run: the whole suite (for the lead's `ci/` push) and the app's tests (no result or spec changed).
- Also tried, not kept as tests: every page of the sheet-set, schedule-set, plan-location (also at /Rotate 90) and review fixtures under two budgets: cover held and no text line was cut.

## Fingerprints

Not moved: no tool specification, description or result changed. **Release note**: none (nothing a tester or a user can reach).

## Calls made

- **Cut while under the line, not into every region.** A region that already reads is left whole, so "level 1" is a cut along a border or gutter and "level 2" a cut where a region has neither left; a tile carries 2 if any cut that made it was second-level.
- **Which cut**: the one leaving the fewest tiles still to cut (the square of how far each half's lettering is under the line). Without it the first cuts were slivers.
- **The lettering height is the smallest in the tile** (font size from the text layer), stricter than the core's page-wide 25th percentile.
- **Ceiling 16 by default**, the grid's most. **Budget is a required argument**: planlens does not know the core's.
- **Overlap**: whole lines and paths within `WIDEN_MAX_FRAC` (a quarter) of the region; 8% more only on second-level tiles. A tile edge is drawn back off a text line before a path is kept whole.
- **Under 40 characters of text: the grid**, through `tile_grid` itself, so the answer is today's.
- Constants: `GUTTER_LETTERINGS` 2, `RULE_SPAN` 0.9, `RULE_THIN_PT` 1, `SECOND_CUT_BAND` 0.25, chosen by reasoning, not measured on real sheets. The fixture's views are lettered at 9 pt and its notes at 4.3 pt.

## For the lead

- **The owner's question this slice raises**: at 16 tiles an E-size sheet with 4.3 pt notes cannot be brought over 12 px with `openai-high` by any cutting (the sheet is 27 tiles' worth at that size). Region tiles buy whole text lines and whole dimension strings at 16; legibility needs a higher ceiling (22 to 29 here) or a larger budget. Which is wanted decides the default.
- Needs a run on real E-size sets before wiring: speed with 100,000 drawn pieces (`page_marks` loops in Python), stroked lettering, hatches, and whether the borders and gutters found are the ones a reader sees.
- `page_marks` reads `doc._doc` (the PyMuPDF page) for the drawings; a public accessor on `Document` would be cleaner.

## Noted, not fixed

- A blank page gets `tile_grid`'s 2 x 2 from the function (the core already reads a page with no ink as one tile before it tiles). A dimension string is not grouped as one object; see DESIGN, "What is not".

## To wire it (read from the clone; not written)

1. **A setting, off by default**, read in `vision_tools._read_pdf_page`: for example `GEOTECH_VISION_REGION_TILES`. Unset: `_tile_count` and `tile_boxes` as now, fingerprints unmoved.
2. **`_tile_jobs`** takes `boxes = vision_view.tile_boxes(info["clip"], n)` and names tiles `r{r}c{c}` from `divmod(k, n)`. With the setting on it would open the document once, call `page_marks` and `region_tiles(marks, vision_view.budget(engine), max_tiles=...)` (the budget already held to `max_px`), and iterate the returned boxes. Tile names and the sentence "tile row r of n, column c of n" need a form that does not assume a grid (a number, with the view).
3. **`_read_pdf_page`**: `n > 1` becomes "more than one tile"; the `tiling` note ("n x n overlapping tiles ... lettering n x larger") needs wording for region tiles; the blank-page and photo-page checks stay in front.
4. **`_tile_legibility`** can stay: each tile's row already carries `_lettering_px` from its own render. `RegionTiles.under` is the same answer in advance; the render's `text_px` is the page-wide quartile, so the two can differ by tile, and one of them should be chosen.
5. **`vision_view`**'s import list from `planlens.review.frames` gains the new names, and the core's planlens requirement moves to the release that has them (planlens published first).
6. `_fit_tiles`, `_merge_found` and `_tiles_not_read_note` read rows by `tile` name only and need no change beyond the names. Image tiles (`image_view.tile_boxes`) are another path and stay.
