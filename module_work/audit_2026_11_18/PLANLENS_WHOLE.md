<!-- Filed here by the lead on 2026-11-18 from planlens' branch `claude/rv-b15-whole` (planlens keeps no review file on its main). Its one repair (2048e7b) is merged to planlens' main on 2026-11-18 after a green run of the whole suite on GitHub (after 0.17.0, not released, no tag). -->

**RELEASABLE AFTER THE FIXES ON THIS BRANCH** (`claude/rv-b15-whole`, one fix, 2048e7b: five source lines and a test). Reviewed: `main` f9b5b38 against v0.17.0 (7eed79b), `git diff v0.17.0 main -- planlens`, the three pieces as one change. The fix is wording in one opt-in result; without it nothing found here is wrong, only unsaid.

# Review B15, the whole change since 0.17.0: PL-09 reading options, PL-37 printed scale, VIS-30 region tiles

Every answer is from a run. Two trees: v0.17.0 in a second worktree and `main`, each imported by `PYTHONPATH` (the tree printed at each run), one venv (Python 3.11, PyMuPDF 2.1, OpenCV 5.1.0.95, numpy 2.4.6, Pillow, mcp). The made-up document: six 36 x 24 in sheets of a made-up firm, title block down the right edge with a scale cell (NTS, AS NOTED, AS NOTED, 1/4" = 1'-0", 1:20, none), general notes in two columns drawn row by row on G-301, one ruled schedule running on from G-302 to G-303, three views with "SCALE: 1:50" under each on the other sheets. The same set at 150 sheets for size.

## Findings

| # | What | Made-up document | At v0.17.0 | On main | |
|---|---|---|---|---|---|
| 1 | The `sheet_index` result's own note names every key of a row but the two PL-37 added: nothing in the result says `scale` is the title block's and never a view's, or what `scale_from` is | the six-sheet set, sheet tools on | no scale keys; note complete | rows carry `scale`, `scale_from`; note unchanged | **fixed here**, 2048e7b: where a row has a scale the note says so (as it does for `from` and `duplicate`); a set that prints no scale has the note word for word; no specification moves |
| 2 | A row with a scale is 75 to 85 characters longer (the `scale_from` text is 44 of them, the same words on every such row), so a long index pages sooner | the 150-sheet set through the application's own `dispatch_document_tool` at its 16,000 cap | 1 call, 150 rows, 10,917 characters | 2 calls: 114 rows and `next_offset`, 15,743 characters | noted. Valid JSON, nothing cut. Saying `scale_from` once would undo a call the builder listed; the owner's |
| 3 | `Document.sheet_index()` gives the scale to a caller that set nothing: `SheetRow.scale`, `scale_how`, and both keys in `to_dict()`. "Rides the sheet tools" is true of the tools, not of the library method | the six-sheet set, `Document(content=...)` with defaults | no such keys | both keys on the five rows whose block prints a scale | noted; additive; CHANGELOG says it; a release note must (below) |
| 4 | Reading options and the index together: nothing moves the rows or the scales | the six-sheet set under all 8 combinations of the three options (toolkit), and three hosts in one conversation (all options + sheet tools; sheet tools; nothing), each given its own and the others' handles, opened in both orders | (a host naming an option raised `TypeError`) | index, cross-reference check, schedule-mark check, page map and `open_document` byte for byte the same in all 8; the warning and the table note appear only for the asker, also when the page is named by `sheets=`; the 54 host results are identical in both orders | none needed |
| 5 | Region tiles beside the index | each of the six sheets, 4 `Document` option combinations, tiles before and after the index is made | (no `region_tiles`) | tiles identical before and after, index identical before and after `page_marks`; `method="regions"`, 7 to 12 tiles, none capped | none needed |
| 6 | `CHECK_B15_PL37.md` is stale where its review reversed it: it lists `HOW_SCALE_CORNER` and "a sheet read alone takes a label in its lower right corner" | | | no such name on main; a sheet read alone has no scale | noted: do not write a release note from that file |
| 7 | `review/DESIGN.md` has no sentence on the open-document limit (PL-09's review, finding 1: an asking host's document is a second open document of the file) | `grep max_open planlens/review/DESIGN.md` | | nothing | for the documents session |

## The six questions

1. **Interactions.** Only `planlens/tools/toolkit.py` is touched by two pieces (PL-09 and PL-37; `CHANGELOG.md` and the two `DESIGN.md` by more). Read whole as merged: `_entry` answers a handle with the calling host's options and `_tool_sheet_index` reads that document's index; rows 4 and 5 are the runs. No piece changes what another returns.
2. **The merges.** Three (7428ecc, 85fe5e5, f9b5b38), each replayed with `git merge-tree`. Conflicts: `CHANGELOG.md` only, in the second and third. Recorded merge against replayed merge: the `CHECK_`/`REVIEW_` files dropped (meant), and in the third the lead added three CHANGELOG lines (the grid is returned where it reads better). For `toolkit.py`, both `DESIGN.md` and `CHANGELOG.md` the lines main adds are exactly the union of the three branches' lines but for those three: nothing lost, nothing doubled.
3. **Nothing moves for a caller that sets nothing.**
   - Pinned surface: `planlens/review/tests/test_host_specs.py` green; `app_tool_specs.json` and the test have no diff since the tag. No existing test file is modified in the diff (every test file in it is new).
   - Comparison (script kept in my scratch folder, not committed): the 34 fixtures the builders under `planlens/testing/` make with default arguments (the four sheet-set styles included; `esize_fixtures` is new and was read by both trees as files), built once and read by both trees. **4,732 results compared, 0 differ** (after the random temp-folder name in image paths is taken out): 14 `Document` methods and 6 more on each of the first four pages, every tool of a default `ReviewToolkit` and of a default `ReviewHost` with 1 to 4 argument sets each, the same with the sheet and drawing tools on, `tile_grid` and `tile_boxes`, the name lists and the specifications in three styles. With my made-up set added: 4,882 compared, 3 differ, all `sheet_index` (rows 1 and 3 above). 638 of the results are refusals of made-up arguments, equal on both sides. `annotate_document` and `find_like` were not called.
   - Imports, 7 runs each tree: `import planlens` loads 1 module in under 1 ms; `import planlens.review` loads 31 new modules, 7 of them planlens', the same set by hash, no third-party module, 14 ms median. Identical at the tag and on main.
   - Making the index on 150 sheets: 1.02 s at the tag, 1.03 s on main.
4. **The application's callers.** `review_flags.py` and `document_tools.py` do not read the rows: they hand planlens' JSON on under a character cap. I imported both from the clone against main (`PYTHONPATH`, nothing installed, clone unchanged) and ran `host_for(sheet_tools=True)` and `dispatch_document_tool`: valid JSON, `mask_secrets` leaves the new rows as they are (the `"` and `'` of an imperial scale included), a host that did not ask is still refused `sheet_index`, `open_document` is byte for byte the tag's for both hosts. What changes is row 2. `deep/vision_kit.py` (its `_truncate`) could not be imported here (the agent stack is not installed); it cuts only above the cap planlens already keeps under.
5. **The suite.** `pytest planlens -q -p no:cacheprovider`, exit code read directly: on main f9b5b38 **2,746 passed, 5 skipped, 0 failed**, exit 0, 6 min 50 s. On this branch (2048e7b): **2,751 passed, 5 skipped, 0 failed**, exit 0 (the 5 new tests). The skips are the same five: the OCR extra is not installed (1), Windows-only path cases (4).
6. **For a release note**, below.

## For a release note

- **Reading options (PL-09).** A host that asks (`column_warning`, `continued_tables`) gets a warning line in `read_document` on a page whose columns are drawn row by row and a note under a table that runs on. The CHECK file leaves out: that host's document has its own handle and counts as a second open document of the file; the added lines count against a result's size, so a long read pages sooner; `captions_on_every_page` changes nothing through `ReviewHost` (only `document_roles` on a `ReviewToolkit`); and at 0.17.0 naming any of the three in `toolkit_options` raised `TypeError`.
- **Printed scale (PL-37).** With the sheet tools on, each row of the sheet index shows the scale the sheet's title block prints and where it was read. The CHECK file leaves out: `Document.sheet_index()` rows hold it for every library caller, asked or not; a sheet read on its own has no scale (the check file says the opposite); a long index takes more calls (the 150-sheet made-up set: one call at 0.17.0, two now); the `sheet_index` description is 7 characters longer, so the application's fingerprint of its sheet-tools build is re-recorded when its pin moves; and, with this branch, the result's note says what the scale is.
- **Region tiles (VIS-30).** The CHECK file says "none". A release adds public names nobody calls yet (`region_tiles`, `page_marks`, `RegionTiles`, `PageMarks` and five constants in `planlens.review.frames`; `planlens.testing.esize_fixtures`), and the function may answer with the uniform grid (`method="grid"`), which cuts text lines: say both, and that no tool, specification or result moves.

## For the documents session (I edited no document)

- CHANGELOG, PL-37 entry: one sentence for the fix (where a row has a scale the result's note says it is the title block's, never a view's, and what `scale_from` is), and that a long index pages sooner.
- `planlens/document/DESIGN.md`, "The scale": the same sentence beside "The tool's rows show `scale` and `scale_from`".
- `planlens/review/DESIGN.md`: row 7. `README.md`: the 0.17.0 line "three `Document` options no tool passes yet" is true of 0.17.0; the next release's line must say they are now passed for a host that asks.

## What I ran

- Interactions: 8 toolkits x 8 results; 54 host results x 2 opening orders (the two files identical by `cmp`); 26 results of the two hosts that ask no reading option, tag against main (2 differ: `sheet_index`, by `scale` and `scale_from` alone; old keys and the rest of the result equal); 4 x 6 region-tile results before and after the index.
- New file `planlens/review/tests/test_rv_b15_whole.py`, 5 tests: before the fix 1 failed (the note) and 4 passed by design (they hold the three pieces together); after it 5 passed. With every test file that names `sheet_index`, the PL-09 and PL-37 files and `test_host_specs.py`: 248 passed, exit 0.
- The tag-against-main comparison, the import counts, the application's dispatch at 150 sheets, the three merges replayed: as above.

## What I did not check

- Any real drawing set, any model, the MCP server, OCR text sources, a scanned set.
- The application's own tests and fingerprints (not installed here; only its two modules were imported and its dispatch run).
- `annotate_document` and `find_like` in the tag-against-main comparison; pages beyond the fourth of a fixture for the per-page `Document` methods.
- Python 3.10 and 3.12 (CI's); this venv is 3.11 and newer libraries than the declared floors.
- Two hosts calling at the same moment on threads.
