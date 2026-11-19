MERGE AFTER FIXES, with the fixes on this branch (`claude/rv-b15-pl37-sheet-scale`, from 95aed91).

# REVIEW B15 / PL-37: the sheet's printed scale in the sheet index's rows

Reviewed: b818d8e and 95aed91 (`git diff 7eed79b HEAD -- planlens`). Before = 7eed79b in a second worktree; after = the builder's head; every row below comes from made-up 36 x 24 in sheets I drew (a made-up firm, title block down the right edge, across the bottom, or turned up the sheet), not the builder's.

## Findings

| # | What | Made-up input | Builder's head | This branch | |
|---|---|---|---|---|---|
| 1 | A key plan the set repeats in the title block's band is read as the sheet's scale, and takes the true scale away elsewhere (the check file's first "Noted, not fixed"). **Blocking.** | 4 sheets; "KEY PLAN" with "SCALE: NTS" (also "SCALE" over "N.T.S.", and "LOCATION MAP / SCALE: 1:5000") in the block's column or bottom strip; the block's own cell prints 1:100, 1:200, AS NOTED, nothing | none, none, none, **NTS** (three true scales lost, one wrong given) | 1:100, 1:200, AS NOTED, none | fixed, 7d539a5 and 34c2b80 |
| 2 | A revision row that tells of a scale is read as the sheet's scale | revision row B whose description cell reads "SCALE 1:50" or "SCALE: 1:50"; block prints none / prints 1:100 | **1:50** / none | none / 1:100 | fixed, cf1ec72 |
| 3 | The lone-sheet rule hands a detail's scale to the sheet | (a) one sheet, block prints none, "ANCHOR DETAIL / SCALE: 1:10" in the lower right; (b) the same, block prints 1:100; (c) bottom block whose scale cell (1" = 40') is left of the corner, a detail (3" = 1'-0") in it; also (a) as two sheets | (a) **1:10**, (b) none, (c) **3" = 1'-0"**, two sheets **1:10** each, all marked "title block not located" | none on all | fixed, 682dee7: a sheet read alone has no scale |
| 4 | My first repair of 1 was too eager: one stray scale line on one sheet unseated the scale every other sheet shares | 5 sheets all "SCALE: 1:100", one with a second "SCALE: 1:50" in the block's column | (builder's head: 1:100 x4, none on that sheet) | the same as the builder's head | fixed, 34c2b80 |
| 5 | Residual of 1, cannot be told apart | key plan with a scale line, and NO sheet's block prints a scale of its own | the key plan's scale on every row | the same | noted. The line is fixed wording in the block's band, exactly what a block that prints one scale on every sheet looks like. It is wrong only for a set that has a key plan with a labelled scale AND a title block with no filled scale cell on any sheet |
| 6 | Residual of 2 | a revision description wrapped onto a line of its own under the last row, reading "SCALE: 1:50" | 1:50 (none if the block prints one) | the same | noted: the line lies outside the revision block's box |
| 7 | Forms not read (no scale, safe) | "1:100 m", "1:500 (A1)", "1 IN. = 20 FT." (full stops), "FULL SIZE" | none | none | noted |

Fix 1 keeps a fixed scale line out only where it stands beside a sheet's own scale on at least two sheets and a quarter of the set; a scale most sheets share with nothing beside it is still theirs (tested). Fix 3 reverses a call the builder listed under "Calls made" and changes one of their assertions (`test_one_sheet_alone_...`, renamed, now asserts none); drop that commit to keep the corner reading. The fixes come to 66 changed lines of `sheets.py` over four commits, docstrings included: more than the 20 a single fix was allowed, said here plainly; each has a test that fails without it.

## The five questions

1. **Read where printed: yes.** 10 values (1:50; 1 cm = 2 m; 1/8" = 1'-0"; 1/8 IN = 1 FT; 1 in = 20 ft; 1" = 20'; AS NOTED; AS SHOWN; NTS; NOT TO SCALE) x 5 layouts (on the label's line; value in the next cell; label above value; "DWG SCALE:" above value; bottom block beside and under) and 6 values x 3 turned layouts: every row as printed, right `scale_how`. NONE, N.T.S., 3/16"=1'-0", 1"=20'-0", "1 : 100" also read. Before: no scale field at all.
2. **Given where not printed:** none for view titles with scales (7 a sheet, in the bands, three block positions), a detail in the block's column, notes holding "scale" ("DO NOT SCALE...", "SCALE OF DETAILS VARIES", a bare "SCALE" label), a graphic bar ("SCALE IN FEET", "GRAPHIC SCALE"), a detail sheet with five view scales (its block's AS NOTED is given), "SCALE CHANGED TO 1:50" in a revision row. Wrong on the builder's head: findings 1, 2, 3.
3. **A host that did not ask:** `TOOL_SPECS`, the toolkit's specs (anthropic, openai) and the host layer's default specs compare equal to 7eed79b's; 69 results from the 17 default tools on a made-up set are equal apart from the random document handle and temp folder in them. `git diff 7eed79b HEAD --stat` touches no file under `planlens/review/`, so `app_tool_specs.json` was not re-recorded, and `test_host_specs.py` passes. For a host that asked: the `sheet_index` description 758 to 765 characters (+7, as the check file states), and in results only `scale` and `scale_from` on rows that have one.
4. **Reuse:** yes, no second reader. Ratios and equations go through `planlens.document.scale.parse_ratio`; only the label and the words (AS NOTED, NTS...) are new. `sheets.SCALE_LABEL` is a second label pattern beside `scale._RE_SCALE_LABEL`; it changes no row.
5. **Suites and documents:** below. CHANGELOG and DESIGN matched the builder's code; I changed both where fixes 1 to 3 changed the behaviour.

## What I ran (`/tmp/pl`, Python 3.12, exit codes read directly)

- New file `planlens/document/tests/test_rv_b15_pl37_sheet_scale.py`: 10 tests; each fix's tests fail with that fix's source change stashed (4, 2, 1 and 1 failures) and pass with it.
- The 10 test files that name `sheet_index` plus all of `planlens/review/tests`: **437 passed, 6 skipped, 0 failed** on this branch.
- Whole suite `pytest planlens -q`: **2585 passed, 16 skipped, 0 failed**, exit code 0, on this branch (source as pushed in 682dee7).

## Not checked

- No real title block (made-up sheets only), no model run, and not the app as a consumer (the builder ran it; my fixes change no key or specification, only which rows have a scale).
- Scanned sheets, sets of mixed sheet sizes, a set with two title-block layouts, and `open_document`'s short index beyond the byte comparison above.
- `planlens.pdf.scale.parse_scale_annotations` against these cases: it is not the reader used.
