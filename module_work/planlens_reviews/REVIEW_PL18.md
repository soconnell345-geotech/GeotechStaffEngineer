<!-- Filed here by the lead on 2026-11-19 from planlens' branch `claude/rv-pl-18` (planlens keeps no review file on its main); the branch was then deleted. -->

# Review: PL-18, the MINSERT part (recheck of `272fb28`)

**Verdict: MERGE AFTER FIXES** — reviewed head `272fb28` (base `6196f13`). No must-fix is left; the three should-fix findings are fixed on this branch (`claude/rv-pl-18`): `dd3211c`, `7717e3c`, `af71457`. Merge those three with the branch; drop this file's commit.

## The first review's findings, rechecked

| # | First review | At `272fb28` |
|---|---|---|
| 1 | must-fix: a MINSERT nested in a scaled or mirrored block landed its cells in the wrong places | **Fixed.** Checked against positions worked out separately (below): one and two levels deep, parents scaled, mirrored in x, in y, in both, turned, extruded (0, 0, -1), and an array inside an array. Every cell and every attribute is where the parent puts it. |
| 2 | the cell cap was per array | **Fixed.** One count for the file (`n_placed`, `ingest.py:360`; per space in `truth.py`). 20 arrays of 1,000,000 empty cells: 1.0 s ingest, 0.8 s truth. An array of an array of an array (1e18 cells): 3.0 s, three warnings. |
| 3 | a zero spacing looped over every cell | **Fixed.** That direction is walked once (`minsert.py` `_cells`); 32767 x 3000 with coinciding rows gives 3000 cells in under a second. |
| 4 | wording in `ir/DESIGN.md` and `CHANGELOG.md` | **Holds**, with two small points under Findings (notes 6 and 7). |

## Findings

| Severity | Where | What is wrong | Input that shows it | Fix |
|---|---|---|---|---|
| should-fix, **fixed `dd3211c`** | `planlens/dxf/minsert.py` `_cells` (the ATTRIB loop) | The grid step is in the reference's OCS and was added unchanged to each ATTRIB's insert and align point, which are in the ATTRIB's own OCS. Where the two extrusions differ, every attribute after the first cell landed mirrored about the first, silently. ezdxf's `multi_insert` has the same fault. | A 2 x 3 array at (2, 3), extrusion (0, 0, -1), with an ATTRIB at (2.5, 3.5) of extrusion (0, 0, 1) (what ezdxf's `add_attrib` writes): the column-1 attribute came at x = 8.5, it is drawn at x = -3.5. | The step goes through the WCS when the extrusions differ; equal extrusions take the old path unchanged. Test `planlens/ir/tests/test_pl18_review_attrib_ocs.py` (3 of 5 cases fail on `272fb28`; the 2 same-extrusion cases pass there by design). |
| should-fix, **fixed `7717e3c`** | `planlens/ir/ingest.py:573-596`, `planlens/dxf/truth.py:297-316` | With the budget already spent, an array got no cell at all and returned. At `6196f13` it was read as one INSERT: its ATTRIBs arrived whatever the budget, as a plain INSERT's still do. So (a) an attributed array after the entity budget lost its attribute; (b) after an array of an empty block had spent the cell budget, a later array lost every cell though no block entity had been walked. The warning said "the remaining copies are omitted" with none placed, and gave no count. | `max_block_entities=20`: a 30-line block, then a 1 x 3 array carrying ATTRIB "IN-ARRAY", then a plain INSERT carrying "PLAIN". Head returned "PLAIN" only; the base returned both. | When no cell of an array was placed, both readers fall through and read it as the single reference ezdxf resolves (nothing is copied, so the bound on work holds). The warning now reads "...budget after N of them; the remaining copies are omitted." `ir/DESIGN.md` says so. Test `planlens/ir/tests/test_pl18_review_budget_first_cell.py` (3 tests, all fail on `272fb28`). |
| should-fix (tests), **fixed `af71457`** | `planlens/ir/tests/test_ingest_dxf.py`, `planlens/dxf/tests/test_truth.py` | The tests did not pin the grid for an array with a scale of its own, nor (in truth) a non-square grid, nor an extrusion of (0, 0, -1). With `minsert.py` changed to scale the grid by the reference's `xscale`, both files still passed; with rows and columns swapped, the truth tests still passed. | The two changes just named. | `planlens/dxf/tests/test_pl18_review_grid.py`: 8 top-level arrays in each reader against trigonometry written in the test. Tests only: it passes on `272fb28` (the code is right) and fails on each of those changes and on an unturned grid. |
| note, **not fixed: needs a decision** | ezdxf `InsertCoordinateSystem.transform`, reached from `virtual_entities()` | A reference that is TURNED and sits under an UNEVENLY scaled parent is drawn sheared. ezdxf cannot hold a shear in an INSERT, tests only the untilted axes for it, and so returns a turned, unsheared reference **without raising or calling the skip callback**. The contents are then misplaced, silently. This is the same at `6196f13` for every plain INSERT; the branch neither adds nor mends it. For an array, the cell insert points are right when the array is directly under the uneven parent; the contents of each cell are not. | CELL = line (0,0)-(1,0.5); HOST holds CELL at (2,3) turned 30; HOST placed at (100,100) scaled (3, 0.5). The line's far end is drawn at (107.8481, 101.9665); both trees return (108.9623, 102.0362), no warning. | The lead's call: detect it (turned child, parent `xscale` != ±`yscale`) and warn, or transform the block's content by the composed matrix instead of trusting the virtual INSERT. A backlog item, not this branch. |
| note | `planlens/dxf/minsert.py` `_cells` | One ATTRIB that cannot be copied or moved ends the whole array with "expansion truncated" (the generator dies), where a plain INSERT loses only that ATTRIB. Not reproduced with a file ezdxf can read; no change made. | — | — |
| note 6 | `planlens/ir/DESIGN.md` ("with the old warning") | The fallback warning's text is new ("its block source is missing"), not the old one. Harmless. | — | — |
| note 7 | `planlens/ir/DESIGN.md` ("so the two cannot agree by sharing an error") | True only for the nested cases the builder tested; `af71457` makes it true for top-level arrays too. | — | — |

## Paths checked

1. **Cell and attribute positions (`_cells`), against positions worked out separately.** My script builds each drawing with ezdxf and computes every expected point with its own 3 x 3 matrices (no `minsert.py`, no `multi_insert`, no ezdxf transform), then compares the ingest's lines and attribute texts and the truth record's leaders. 155 layouts: top-level plain; turned; scaled unevenly (with and without a turn); mirrored in x, in y, in both; extrusion (0, 0, -1); negative spacing; each with an ATTRIB; ATTRIBs with halign 1 to 5 and a valign (insert through the ingest, align point read off the cells); one level under 8 parents; two levels under 20 parent pairs; an array in an array at the top, under 4 parents, and two levels down. On `272fb28`: 138 right, 2 wrong for the ATTRIB extrusion (fixed), 15 wrong for the ezdxf shear above (17 before the fix; same 15 remain, all "a turn below an uneven scale", none without). On the fixed head: 140 right, the same 15 remain.
2. **Does ezdxf scale and mirror a reference but not its grid spacing? Yes.** Installed ezdxf **1.4.5**. `Insert.transform` (`entities/insert.py:463-494`) rewrites `insert`, `rotation`, `extrusion`, `xscale`, `yscale`, `zscale` and transforms the ATTRIBs; it never touches `row_spacing` or `column_spacing`. `multi_insert` (`:668-702`) turns the offset by the rotation and says "Do not apply scaling to the grid!". `virtual_entities()` yields a nested MINSERT as one transformed copy, so its grid would stay in the unscaled frame; building from `source_of_copy` and moving by `parent.matrix44()` (the matrix ezdxf itself explodes with) is right.
3. **The two readers share `minsert.py`: does that weaken the check? Yes, in one way.** Truth agreeing with the ingest (`test_truth_walks_every_minsert_placement_as_ingest_does`) now proves nothing about where a cell is: one wrong line moves both. What still catches a wrong position is only a test with hand-worked numbers. I changed `minsert.py` six ways and ran both files on `272fb28`: no turn of the grid, no parent transform, nested built from the virtual copy — caught by both files; rows and columns swapped, attribute not moved — caught by the ingest tests only; grid scaled by the reference's own scale — caught by **neither**. `af71457` closes the last two gaps for cells in both readers. Attribute positions are checked in the ingest only (the truth record holds no ATTRIBs). The rest of truth (records, layers, block names, caps) is still its own code.
4. **The cap.** With the budget spent the caller gets: the cells placed so far; one warning per array (`MINSERT '<name>' places <n> copies; expansion stopped at the <cap>-entity budget after <k> of them; ...`, the count added by `7717e3c`); the existing "Block explosion stopped" warning when the entity budget is what ran out; and, after `7717e3c`, the first cell of any later array read as a plain INSERT. Nothing is short without a warning. There is no cell count in `metadata` (`n_block_entities` counts IR entities as before); two arrays of the same block, size and count placed share one warning (`_warn_once`).
5. **A DXF with no MINSERT.** 40 made-up drawings (1512 IR entities: nested, mirrored, extruded and attributed INSERTs, an empty block, a missing block, INSERTs carrying grid tags with zero spacing or 1 x 1 counts, small budgets), each tree run in its own subprocess, every entity, warning, metadata value and truth record dumped: `6196f13` and `272fb28` byte-identical; `6196f13` and the fixed head byte-identical.
6. **Merge with `origin/claude/b13-integration` (`a17272a`).** Tried in a scratch worktree from `272fb28`: clean, `CHANGELOG.md` auto-merged, no conflict; the two touched test files pass on the merge (130). Aborted, nothing pushed.

## Tests

- Venv: `python3.12 -m venv /tmp/pl; pip install -e ".[mcp]" pytest` (ezdxf 1.4.5).
- On `272fb28`: `pytest -q planlens/ir/tests/test_ingest_dxf.py planlens/dxf/tests/test_truth.py` — 130 passed.
- The builder's new tests on the base (both files copied into a worktree of `6196f13`): 24 failed, 106 passed. Every new MINSERT test fails there except `test_no_definition_source_falls_back_to_the_first_cell`, which passes by design (the fallback is the base's behaviour).
- On the fixed head (`af71457`): `pytest -q planlens/ir planlens/dxf` — 721 passed, exit 0.
- New tests of mine: `test_pl18_review_attrib_ocs.py` (3 fail / 2 pass on `272fb28`), `test_pl18_review_budget_first_cell.py` (3 fail), `test_pl18_review_grid.py` (16 pass on `272fb28`; fails on the mutations named above).
- The tests check positions and counts, not text, apart from the warning substrings.
- The builder's two timing tests (`< 30 s`, `< 10 s`) ran in about 1 s each here; they are bounds on a machine's speed and could fail on a very slow runner.
- Fingerprints: none in planlens; no tool specification changed. I did not run the Document Review suite against this branch: no document tool's result or specification changes, only what `from_dxf` returns for a file with a MINSERT.

## Scope

- Files changed by the builder: `CHANGELOG.md`, `planlens/dxf/minsert.py` (new), `planlens/dxf/truth.py`, `planlens/ir/ingest.py`, `planlens/ir/DESIGN.md`, and the two test files. All belong to the item.
- One existing test edited: `test_minsert_array_reports_its_missing_copies` became `test_minsert_array_ingests_every_placement`. The reason is true: its assertion (a "12 copies" warning, one cell) is what the item removes.
- New tests went into the two existing test files, not new files named for the item as COMMON_DOCREVIEW asks. No harm.
- Public text: the diff holds made-up values only; none of the listed words.
- Nothing beyond the item, bar the zero-spacing and cap work the first review asked for.

## CHECK file

There is none on the branch (the prompt named no task file). The claims I checked are the ones in `CHANGELOG.md`, `ir/DESIGN.md` and the docstrings: all hold, bar notes 6 and 7 and the shear note ("an unevenly scaled parent ... stretches the grid too" is true of the cell positions; the contents of a turned cell under it are ezdxf's shear case).

## Could not check

- A real CAD-written MINSERT (the ten validation sheets have none; every drawing here was built with ezdxf). To check: export one array from a CAD program, mirrored once with the mirror command, and compare `from_dxf`'s cells with the plot; this also shows which extrusion that program gives a mirrored reference's ATTRIBs.
- 3D extrusions other than (0, 0, ±1): not tried.
