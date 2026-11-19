# B15: a figure whose one name is taken keeps its chart (the limit TA-10 left)

Branch `claude/b15-figure-pair`, from `claude/b13-integration` at 9367934 (1.3.0).

## Confirmed on the head, before the change
- `aec_document_review/core.py`, `import_external_artifacts` and `import_reported_outputs`: each file's destination came from `_unique_dest` alone. With `plot.png` in the folder, a new `plot.png` + `plot.plotly.json` became `plot_1.png` + `plot.plotly.json`; with `plot.plotly.json` there, `plot.png` + `plot_1.plotly.json`. `plotly_picture_twin` then finds no picture (or the OLD one), and `collect_turn_artifacts` gives two cards.
- A second form of the same defect, in `import_reported_outputs` only: it reuses a file of the same bytes, so an unchanged picture with a new chart gave `plot.png` (reused) + `plot_1.plotly.json`.

## Changed (a repair; no setting; 52 lines added, 3 removed in `core.py`)
- New `core._figure_dests(sources, fd, reuse=False)`: the files of ONE figure imported together (a `*.plotly.json` with the `.png/.jpg/.jpeg/.html/.htm` of the same stem, from the same source folder; stems compared without case, as the card does) get one stem: `plot`, else the first `plot_N` under which none of them meets an existing file. `reuse` (reported outputs): a file of the same bytes does not count as taken under the figure's own stem.
- `import_external_artifacts` and `import_reported_outputs` ask it first and still pass the answer through `_unique_dest`, so nothing is ever overwritten. `import_reported_outputs` now picks its sources in a first loop and copies in a second (same filters, same order).
- `_unique_dest`, the card code, the list of files and `sharepoint_tools.py` are untouched.

Example (made up): the folder holds `plot.png`; a turn writes `plot.png` and `plot.plotly.json` elsewhere. Before: `plot_1.png`, `plot.plotly.json`: a picture card and a chart that downloads the OLD picture. After: `plot_1.png`, `plot_1.plotly.json`: one chart card.

## What reads the pair afterwards
- The card: `collect_turn_artifacts` (called from `turn_jobs.py` on the imported paths) gives the sidecar alone, and `plotly_download_twin` (the card's Download, `app.py`) serves the new picture. Asserted in every pair test.
- The list of a conversation's files (`core.py` about line 3863) names a chart's picture through `plotly_picture_twin`: it finds the new picture. Asserted through that function, not through the list itself.

## The neighbours
- **Three imports in a row** (one name taken at the start): `plot_1.*`, `plot_2.*`, `plot_3.*`, each one chart card with its own bytes; the first is still whole after the third. Tested, both import paths, each of the two names taken.
- **A number only one file holds** (`plot_1.png` alone in the way): both copies take `plot_2`. Tested.
- **A figure whose interactive file is missing** (a picture alone): named file by file as before, `plot_1.png`, one picture card. Tested; unchanged from the head.
- **A file that only shares the stem** (`plot.csv`): not part of the figure. It keeps its free name while the figure moves to `plot_1.*`, and moves alone (`plot_1.csv`) when only its own name is taken. Tested.
- **The HTML copy** moves with its figure (`plot_1.html`). Tested.
- **The very same figure reported twice**: reused, two files in the folder, as before. Tested; unchanged from the head.

## Tests
New file `packages/aec-document-review/tests/test_b15_figure_pair_one_collision.py`, 18 cases, made-up names. On the head 15 fail and 3 pass; the 3 pin behaviour that must NOT change (the picture alone, twice; the same figure reported twice). On the branch 18 pass.
- venv (a), from `packages/aec-document-review`: the new file, `test_ta10_plotly_sidecar_name.py`, the eight other test files that name the import or card functions, `test_review_fingerprints.py`, `test_boundary.py`: 183 passed, 0 failed.
- `pytest --noconftest webapp/tests/test_public_text.py webapp/tests/test_review_freeze.py`: 13 passed.
- Not run: venv (b) and the core's tests (no core file changed); the full suites (the lead's `ci/` push).

## Fingerprints
Not moved (`tests/test_review_fingerprints.py` passes); nothing re-recorded.

## The geotech page's copy (read only, not touched)
`webapp/core.py` has the same limit and also the older TA-10 defect: its `_unique_dest` (about line 1787) still splits at the last dot (`plot.plotly_1.json`), and its two imports (about lines 1815 and 1902) name file by file. Both repairs would port as they are. The maintainer's.

## Calls made
- The whole figure moves when any one name is taken, so a picture identical to the one in the folder is copied again as `plot_1.png` when its chart is new: one more file, for a pair that reads right.
- A figure is the files imported in ONE call from ONE folder. A picture already imported in an earlier turn is not renamed when its chart arrives later.
- The HTML copy counts as part of the figure (the card treats it so).
- The core is unchanged: none of this code lives there.

## Release note
A figure copied into a conversation that already held a file with the name of its picture or of its interactive file (but not both) showed as a plain picture beside a separate chart; it is now one card with its chart.

## For the lead
- Nothing live is needed. The SharePoint download (`sharepoint_tools._settle`) fetches one file at a time and still names file by file.
- TA-10's text in BACKLOG and the 1.3.0 note's "Known limits" line can drop the sentence for Document Review's copy once this merges.

## Noted, not fixed
- A sidecar imported alone that becomes `plot_1.plotly.json` pairs with an unrelated `plot_1.png` already in the folder (and a lone picture likewise with an unrelated chart). As on the head.
- A figure of the same bytes as an earlier `plot_1.*` is copied again as `plot_2.*` when `plot.*` differs; reuse is only under the figure's own stem. As on the head for single files.
