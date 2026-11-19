**MERGE AFTER FIXES** (the one fix is on this branch, 0fd011b). Reviewed head 3ed9892, `claude/b15-vis30-region-tiles`, against 7eed79b (v0.17.0). No must-fix is left open.

## Findings (budget `openai-high`; "px" is the smallest lettering, each line in the tile that holds it whole and shows it largest)

| # | What | Made-up sheet | Uniform grid | Region tiles at 3ed9892 | |
|---|---|---|---|---|---|
| 1 | **must-fix.** A region with no text-free place to cut is marked `stuck` and left. The result is then far worse than the grid, and `capped` is False for it. With nothing free at all it is ONE tile, `method="whole"`, which DESIGN reads as "already reads whole". A caller that falls back on `capped` does not fall back. | E-size, 5,000 lines at made-up positions | 16 tiles, 7.9 px | 1 tile, 2.5 px, `whole`, `capped` False, `under` [0] | **Fixed here** |
| 2 | **must-fix.** One large plan, no gutters: the regions do worse at the same 16 tiles. `capped` and `under` say so, but a caller cannot tell from them that the grid is better without measuring the grid itself. | D-size plan, 70 scattered notes, 14 dimension strings | 16 tiles, 12.0 px, 58 lines under, 105 cut, 18 whole in no tile, 4 strings lost | 16 tiles, 7.5 px, 153 under, 0 cut, 3 strings lost | **Fixed here** (now the grid) |
| 3 | Same, milder. | 36 x 24 in, twelve details in a grid | 16 tiles, 11.5 px, 26 under, 40 cut, 4 strings lost | 16 tiles, 10.9 px, 82 under, 0 cut, 0 lost | **Fixed here** (now the grid; the price is the 40 cut lines and 4 strings) |
| 4 | A text line longer than a tile: the region holding it is a strip the width of the sheet. | E-size, 6 lines right across, 200 notes | 16 tiles, 7.9 px, 6 lines whole in none | 16 tiles, 2.5 px (also at ceiling 40) | **Fixed here** (now the grid) |
| 5 | Text turned 90 degrees, scattered. Its lettering height is measured right (4.3). | E-size, 300 turned lines, 100 level | 16 tiles, 7.9 px | 16 tiles, 3.2 px (a stuck region) | **Fixed here** (now the grid) |
| 6 | Where the regions win they are kept. | 30 x 42 in portrait, title block right | 16 tiles, 9.2 px, 274 under, 50 cut, 3 strings lost | 16 tiles, 8.8 px, 147 under, 0 cut, 2 lost; 25 tiles at ceiling 40, 12.0 px, 0 under | noted: unchanged by the fix |
| 7 | With room to finish, the plan of row 2 loses MORE dimension strings than the grid (second-level cuts through linework). DESIGN's "what is not" covers it; the result does not say it. | row 2's sheet, ceiling 40 | 4 strings lost | 29 tiles, 12.0 px, 0 under, 6 lost | noted |
| 8 | "No text line cut" holds to `CUT_TOL_PT` (1 pt), not exactly: edges shave 0.5 and 0.6 pt off two line boxes. Not a pixel a model would miss. | row 2's sheet, ceiling 40 | | 2 lines | noted; DESIGN now says the tolerance |
| 9 | `RULE_THIN_PT` and `SECOND_CUT_BAND` were not in DESIGN. Unstated still: collinear pieces are keyed by rounding to 1 pt (`_through`), the widening runs at most 4 passes, the cost is the square of the shortfall (stated). No sheet size, region name or coordinate in the source. | | | | DESIGN lines added here |
| 10 | Lettering so small the ceiling stops it: honest. | E-size, 400 lines at 1 pt | 1.9 px at 4 x 4 | 16 tiles, 1.9 px, `capped`, all 16 in `under` | none needed |

**The fix (0fd011b, one commit, 37 added lines in `frames.py`, a little over the 30 asked):** when cutting ends with a tile under the line, the uniform grid of no more tiles than `max_tiles` is measured the same way. It is returned (`method="grid"`, `level` 0, per-tile `lettering_px` and `under` from the real lettering, `capped` as the regions left it) when its smallest lettering is larger AND its lines are on the whole no smaller (heights summed, each at no more than 12 px, 0 where no grid tile holds the line whole). The second test keeps the regions on row 6 and on a letter page of fine print right across (every grid tile cuts every line). It is a judgement the owner may want otherwise: on row 3 it gives up whole dimension strings for 0.6 px.

**The builder's row (second E-size sheet, 455 lines under against 437).** Regenerated with `python -m planlens.review.tests.test_b15_vis30_region_tiles`: all ten rows match the check file, number for number. There the regions are not worse: the smallest lettering is 10.3 px against 8.1 and the lines are on the whole larger, so the function keeps them, before and after the fix. `capped` and `under` are enough there. They were not enough on rows 1 to 5, which is why the function now returns the grid itself.

## The guarantees (question 2), before and after the fix

- **Cover**: exact (coordinate-compressed union, not a raster) on all 14 sheets and both ceilings: 0 pt2 uncovered. Holds at /Rotate 90 and 270 (my portrait sheet and the builder's first sheet): the same tile counts and lettering sizes as unrotated.
- **No line cut, overlap**: hold for `"regions"` and `"whole"` to 1 pt (row 8). The grid, where returned, cuts lines: DESIGN now limits the guarantees to the other two methods.
- **Ceiling**: never exceeded (1, 4, 16, 40).
- **No text**: `"grid"`, 16 tiles, all in `under` at the assumed 4.3 pt. **Blank letter page**: 2 x 2, as the check says. **Letter page, two columns**: 2 tiles, 16.7 px, nothing cut. **Fine print across a letter page**: 2 tiles, 7.8 px, `under` [1], `capped` False: a stuck region, true as stated.

## Cost (question 4)

E-size sheet, 96 blocks, 4,800 text lines, 104,064 drawn pieces: `page_marks` 2.1 s (2.6 s on a second sheet of 105,000), `region_tiles` 0.17 to 0.30 s at ceilings 16 to 200; rendering that page once is 0.6 s. Linear in the pieces: 200 such sheets is about 7 to 9 minutes of `page_marks`, all in its Python loop. Nothing quadratic found (5,000 lines: 0.04 s). `page_marks` reads `doc._doc`: if `Document` renames it, `page_marks` raises `AttributeError` and there is no fallback. The builder's tests would catch it (their module fixture calls `page_marks` on a real `Document`, so most of the 29 error). No caller is exposed today.

## Nothing moves (question 5)

- `tile_boxes` and `tile_grid`: source identical to 7eed79b (compared by AST segment). The only line removed from `frames.py` in the builder's diff is the end of `__all__`.
- `test_host_specs.py` 26 passed, `app_tool_specs.json` and every existing test untouched in the diff. `test_frames.py` 63 passed.
- `planlens.testing.esize_fixtures` imports only the standard library at import and PyMuPDF inside its functions (a base dependency). Nothing imports it but the new tests. `import planlens`: 71 modules before and after; `import planlens.review`: 103 before and after; numpy and PyMuPDF not loaded by either.

## 'To wire it' (question 6): right as far as it goes; it leaves out

1. `TILE_WORKERS = 17` is the page plus 16 tiles in one wave, and `TILED_RESULT_CHARS = 30000` is shared by all tiles: at the 22 to 29 tiles the check says legibility needs, a page is two waves and `_fit_tiles` shortens every tile's text further. It says `_fit_tiles` needs no change.
2. The agent's own `tiles` argument (`N` or `NxN`, 2 to 4, in the tool description): what an explicit N means with the setting on is not said, and that description is fingerprinted.
3. `method="grid"` (more likely after this fix) still wants the `r{r}c{c}` names; the caller must branch on `method`, and on `capped` for a higher ceiling.
4. `_tile_jobs` gets `pdf_bytes` and a `clip`; `page_marks` needs an open `Document` and `region_tiles` takes `clip=`. The 2 s of `page_marks` on a heavy sheet lands before the first tile call.

## What I ran

- Builder's file: 29 passed at 3ed9892. After the fix, with ONE assertion of it edited (`test_the_tiles_cover...`: cut lines allowed when `method == "grid"`; it failed at ceiling 4 on the second sheet, which now gets the 2 x 2 grid): 29 passed.
- New `test_rv_b15_vis30_region_tiles.py`: 4 tests; 2 fail without the fix (rows 1 and 2), 2 pass before and after by design (they hold the fix back from sheets it must not touch).
- Both files with `test_frames.py`, `test_host_specs.py`, `test_tools.py`, `test_toolkit.py`, `test_mcp_server.py`: 226 passed, 1 skipped.
- 14 made-up sheets of my own in a scratch script (not committed), each at ceilings 16 and 40.

## Not checked

The whole planlens suite and the app's tests (not run). Any real drawing set or model: whether the grid or the regions is read better is arithmetic here, not measured. Stroked lettering, hatches, hidden CAD text. Budgets other than `openai-high`. The fallback at a ceiling that is not a square (17 to 24 falls back to 4 x 4, 5 to 8 to 2 x 2).
