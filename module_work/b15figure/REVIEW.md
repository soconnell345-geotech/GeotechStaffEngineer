# B15 review: MERGE AFTER FIXES (one small fix, on this branch `claude/rv-b15-figure-pair`); reviewed head e1729f4

Nothing blocks: no import overwrote, renamed or removed an existing file in any case run. Before = 9367934 in a second worktree, after = e1729f4, fixed = this branch; every row was run in a temporary folder with made-up files, through both `import_external_artifacts` (E) and `import_reported_outputs` (R).

## Findings

| # | What | Made-up input | Before (9367934) | After (e1729f4) | Status |
|---|---|---|---|---|---|
| 1 | A lone file of the same name, from another source folder of the SAME import, is copied first and takes one of the figure's two names: the chart shows with the other folder's picture and Download serves it; the figure's own picture is a second card | `a/plot.png` (alone), `b/plot.png` + `b/plot.plotly.json`, empty folder; also `a/plot.plotly.json` alone | `plot.png` = a's, `plot.plotly.json` = b's, `plot_1.png` = b's | the same (not a regression) | **Fixed here** (`core.py`, both imports copy whole figures first, returned order unchanged; 9 source lines; 4 test cases fail without it) |
| 2 | The defect itself | `plot.png` taken; `plot.plotly.json` taken; three imports in a row; `plot_1.*` with `plot_1.png` taken; `Site Plan v1.2 rev.b.PNG`; `Site Plan v1.2.Plotly.JSON` | two cards; the chart pairs with the OLD picture or with none | one chart card, picture and Download are this import's (`plot_1.*`, `plot_2.*`, `plot_3.*`, `plot_1_1.*`, `Site Plan v1.2 rev.b_1.*`) | Gone |
| 3 | Both names taken; neither taken | `plot.*` both there; empty folder | `plot_1.*`; `plot.*` | the same | Unchanged |
| 4 | Two figures of one stem from two folders in one call | `a/plot.*`, `b/plot.*`, with `plot.png` taken and with nothing taken | one pair split (`plot_2.png` alone) | `plot_1.*` and `plot_2.*` (or `plot.*` and `plot_1.*`), two chart cards, four distinct destinations | Holds |
| 5 | Two files of one name in one call | `a/notes.csv`, `b/notes.csv`, `notes.csv` taken | `notes_1.csv`, `notes_2.csv` | the same | Unchanged |
| 6 | Not a figure, shares the stem | `plot.csv` with the figure, `plot.png` taken; `plot.csv` taken only | csv keeps `plot.csv`; `plot_1.csv` | the same; the figure moves alone | Unchanged for the csv |
| 7 | Picture alone; picture + HTML without a chart; chart alone | `plot.png` taken | `plot_1.png`; `plot_1.png` + `plot.html`; `plot_1.plotly.json` | the same | Unchanged |
| 8 | The same figure reported again (R) | `plot.*` twice, same bytes | reused, two files | the same | Unchanged |
| 9 | Same picture, new chart (R) | second `plot.plotly.json` differs | `plot.png` reused + `plot_1.plotly.json`, chart without picture | `plot_1.png` + `plot_1.plotly.json`, one card; one more copy of the picture (CHECK says so) | As designed |
| 10 | The HTML copy now moves with its figure | `plot.html` taken | `plot.*` + `plot_1.html`, HTML a second card | `plot_1.*` all three, one card | Better; in CHECK |
| 11 | Halves of a figure in two different folders, or stems that differ in case on a case-keeping disk (`Plot.PNG` + `plot.plotly.json`) | `a/plot.png`, `b/plot.plotly.json` | named file by file | the same | Noted, not fixed (not one figure by CHECK's definition) |

## Question by question
1. Gone, rows 2 to 4. A stem that ends in a number gets a second number (`plot_1_1`), which pairs.
2. In every case the folder was listed and every earlier file compared byte for byte after each round: none changed, none removed, no two sources of one call reached one destination. Both imports still pass every destination through `_unique_dest`, so the plan can only be renamed, never overwrite. The new test's second case pins this with two figures and a lone picture of one stem beside an earlier `plot.png`.
3. The card (`collect_turn_artifacts`), the picture named in the list of files (`plotly_picture_twin`) and Download (`plotly_download_twin`) find this import's picture in every pair case. A later import of the same figure: reused on the reported path (row 8); a fresh `plot_N.*` pair on the external path, as before.
4. Rows 6 and 7: unchanged.
5. `git diff 9367934 HEAD --stat -- webapp funhouse_agent`: empty. Fingerprints not moved; no prompt, tool description or setting touched, by the branch or by the fix.

## Run (venv (a) of COMMON_DOCREVIEW, Python 3.12, planlens 0.17.0 from PyPI)
- My case script on before, after, fixed: exit 0 each; before and after differ only in the pair rows above; after and fixed differ only in row 1.
- From `packages/aec-document-review`: the builder's file, `test_ta10_plotly_sidecar_name.py`, the eight other files that name the import or card functions, `test_rv_b15_figure_pair.py`, `test_review_fingerprints.py`, `test_boundary.py`: 189 passed, exit 0 (183 of the builder's count plus my 6).
- `test_b15_figure_pair_one_collision.py` on 9367934: 15 failed, 3 passed, as CHECK says; the 3 are the pins it names.
- `test_rv_b15_figure_pair.py`: 6 passed with the fix (exit 0); without it 4 failed, 2 passed (exit 1; the 2 pin row 4 and "no earlier file changes").
- `pytest --noconftest webapp/tests/test_public_text.py webapp/tests/test_review_freeze.py`: 13 passed, exit 0; `test_public_text.py` again with the new test file staged: exit 0.

## CHECK.md
Every statement I tested holds. "52 lines added, 3 removed" is right for `core.py`. One thing it does not say: row 1.

## Not checked
- The list of a conversation's files and the save to the connected document library were not run end to end: the list only through `plotly_picture_twin`, the library not at all (no live service; its code is not in the diff, and it copies files under the names the folder already holds).
- The page itself in a browser, a shared host with two testers at one moment, Windows (a disk that ignores case), venv (b), the core's tests and the full suites (the lead's `ci/` push).
- The four other points of REVIEW_DOCREVIEW's shared-host list: the change adds no lock, thread or module-level table, read from the diff only.
