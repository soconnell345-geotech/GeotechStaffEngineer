<!-- Filed here by the lead on 2026-11-19 from planlens' branch `claude/rv-dr-44` (planlens keeps no review file on its main); the branch was then deleted. -->

# Review: `claude/dr-44-geometry-fallback` (DR-44, PL-13, DR-19)

**Verdict: MERGE AFTER FIXES**, reviewed head `5f02dc2` (base `6196f13`). Six fix commits are on `claude/rv-dr-44`; three decisions are left to the owner (findings 1, 3, 7).

## Findings

| # | Severity | Where | What is wrong | Input that shows it | Fix |
|---|---|---|---|---|---|
| 1 | must-fix | `planlens/dxf/plot.py:55` (imports), `toolkit.py` `_dxf_plot` | ezdxf's drawing add-on imports Pillow, which planlens does not require. On planlens' own CI install (`pip install -e ".[mcp]" pytest`) 6 of the branch's 24 new tests fail, and a host that asked is told the file "is not a DXF ezdxf can read (ModuleNotFoundError)". | A venv built as COMMON_DOCREVIEW says, any DXF. | **`835a203`**: `PlotUnavailable` names Pillow; the 6 plot tests skip without it. **Left to the owner:** make Pillow a dependency (core, or an extra that CI installs). Until then CI does not run the plot tests. |
| 2 | must-fix | `planlens/dxf/plot.py` `dxf_to_pdf`, called inside `ReviewToolkit.open` under `self._guard` | The plot had no bound and holds the toolkit's own lock, so every call of that conversation waits. Measured here: 100,000 lines 64 s and 390 MB; lettering 20-35 ms a note in the PDF writer (500 notes 17 s; 5,000 did not end in 2 min); a 21 KB file of blocks nested ten to a block draws ten times more per level, without end. | `ezdxf` file: block `L8` holding 10 x `L7` ... down to one line; one INSERT in model space. | **`879628d`**: refused as `PlotTooLarge` over `MAX_FILE_MB` 16, `MAX_ENTITIES` 50,000 (blocks expanded) or `MAX_SECONDS` 30 (checked per entity and per shape written). |
| 3 | must-fix | `planlens/document/drawing_geometry.py` `_rows` | A dimension's `value` is the NEAREST text within a reach sized off the sheet's typical line. On a sheet of long lines a dimension lettered in strokes came back with `value: "SEE NOTE 4"`, a note 150 pt away, under "measured from the PDF's drawn geometry". This is DR-44's own case (stroke lettering beside a text-layer title block). | 6 lines 740 pt long, one arrowed dimension with no text, one text 150 pt above it. | **`0bf6613`**: each row carries `value_offset_pt` / `text_offset_pt` and a note says the text is the nearest one. **Left to the owner:** a limit past which a text is not the value (`planlens.ir.queries` `text_radius`, not this branch's code). |
| 4 | must-fix | `planlens/review/conversation.py` `_new_toolkit` (`dxf_plots = True`), `toolkit.py` `open` | A host that did NOT ask got a plot. `_side_document` knows a DXF by the name given; the toolkit also by the path it resolves to. And a plot opened by one host was found by content key at the next open. | `ReviewHost(resolve_source=lambda s: ".../site.dxf")`, `open_document(source="site plan")`: base "could not open ... as a PDF or image"; branch a one-page PDF with `plot`. | **`24121bd`**: the shared toolkit asks the calling host at each open (`dxf_plots` takes a callable); a plot has its own key. |
| 5 | should-fix | `drawing_geometry.py` `drawing_geometry` | "None found" read as "none drawn". The finders need a filled arrowhead sized against the sheet's typical line: a dimension among 300 hatching strokes, one ended with tick marks, and a 40 pt one beside long walls all returned `dimensions: []`, count 0, no word. | Each of the three, built with the test fixtures. | **`742faff`**: a note names each empty kind and what the finders miss. |
| 6 | should-fix | `planlens/dxf/plot.py` page | "Scaled to fit" was untrue for a small drawing: one unit plots as 1 mm and is only scaled down. A detail 2 units wide (a drawing in metres) was a 2 mm speck on a 22 mm page of margin. | DXF with one line (0,0)-(0.5,0.2): page 58 x 57 pt, drawing 1.4 pt. | **`bcc9400`**: scaled up to A4's long side (`MIN_SIDE_MM`). |
| 7 | note, design | `plot.py` `PLOT_NOTE` | No result gives the plot's transform, so a position seen on the plot cannot be turned into DXF coordinates (or the reverse). The note only says the scale is "the fit to the page". Nothing wrong is stated; nothing checkable is stated either. | Any plot. | Not fixed: return origin and units-per-point with `plot`. Owner's call. |
| 8 | note | `toolkit.py` `open` | With finding 2 fixed the plot still runs under the toolkit's lock, for up to about 45 s (16 MB read about 13 s, plus 30 s). Moving it outside the lock is a restructure of `open`. | - | Not fixed. |
| 9 | note | `plot.py` `MAX_SECONDS` | 30 s refuses a DXF of more than roughly 1,000-1,500 lettered notes, which many real sheets have. ezdxf's `TextPolicy` could draw lettering faster (outline or boxes). | 600 notes with `max_seconds=3`: refused at 3.1 s. | Owner's call: the number, or a text policy. |
| 10 | note | `toolkit.py` `_drawn_ir` | Each first read of a page hands the WHOLE PDF's bytes to `from_pdf_vector` under the document's lock; not timed on a large set. | - | Not measured. |
| 11 | note | `document_structure`, `document_page_map` on a plot | A plot's page is classed `blank` (no words), so the structure reads "title: blank". | `dxf1` probe below. | Not fixed. |

`CHANGELOG.md` and the two `DESIGN.md` files do not yet mention the fix commits' new keys and constants (`value_offset_pt`, `text_offset_pt`, the none-found note, `MAX_*`, `MIN_SIDE_MM`, `dxf_plots` as a callable).

## Paths checked

1. **Not asked = unchanged: holds** (after `24121bd`). `/tmp` script dumped, on `6196f13` and on the branch: the toolkit's specs for every `sheet_tools` x look, `TOOL_SPECS`, `SHEET_TOOL_SPECS`, the host specs and names for every `sheet_tools` x `plan_tools` x look, in two styles, and 14 calls for four hosts (a PDF; a DXF by name through open, read, search, render, region, annotate, measure, `drawing_geometry`; DXF bytes under a `.pdf` name). 98 entries, identical apart from the temporary folder's name. A non-asking host calling by the toolkit's `doc_...` handle of a plot another host opened: refused as at the base.
2. **Wrong numbers.** Ends and tips are right at /Rotate 0, 90, 180, 270, with a crop box offset (50, 40), crop box plus rotation, and a MediaBox with origin (100, 100): all match `get_drawings` and `rotation_matrix`. Stacked dimensions 14 pt apart and a chain of three sharing tips: each value with its own ends. Stroke lettering with no text at all: values `None`, said. Stroke lettering beside other text: finding 3.
3. **None found:** finding 5. A scan is refused in plain words; a picture with lines over it is said.
4. **`_side_document`'s early return:** finding 4. With it fixed, the refusal for a `.dxf` name is unchanged and read/search still read the DXF's text for both hosts.
5. **The plot's bound and place:** finding 2. The plot is never written to disk: bytes in memory, opened as a `Document`; renders go to the toolkit's own temporary folder like any page's. A toolkit belongs to one conversation's working folder, and its cache is its own, so two conversations cannot see each other's plot.
6. **The plot's frame:** a drawing at (500000, 4000000) plots the same as one at the origin; a block inserted plain, turned 90 degrees and mirrored gives an 8 x 4 unit extent, as drawn. No mapping is stated to check (finding 7).
7. **Merge with `origin/claude/b13-integration`:** `git merge --no-commit` in a scratch worktree is clean for `5f02dc2` and for `claude/rv-dr-44` (`CHANGELOG.md` and `conversation.py` auto-merge); the DR-44/DR-19 test files pass on the merged tree (83 passed). No merge was pushed.

## Tests

Venv as COMMON_DOCREVIEW ("Setup: the planlens repository"), Python 3.12, ezdxf 1.4.5, PyMuPDF 2.1; Pillow added by hand after finding 1.

- Reviewed head, the four new files, without Pillow: **6 failed, 18 passed**. With Pillow: 24 passed.
- On the base `6196f13`: three of the four files do not import; `test_dr44_drawing_tools.py` 5 failed. None passes there.
- Review branch, `planlens/dxf/tests` + the two `test_dr19_*` review files: 65 passed (45 passed, 6 skipped before the last two fixes, without Pillow).
- Each fix's test file on the reviewed head: `test_plot_needs_pillow.py` 2 failed; `test_dr19_plot_bounds.py` 6 failed (the nested-blocks test left out, it would not end there); `test_dr44_tied_text.py` 2 failed, 1 passed (by design: no text, no offset); `test_dr44_none_found.py` 2 failed, 1 passed (by design: what is found is not called missing); `test_dr19_small_drawing.py` 3 failed, 1 passed (by design: a large drawing as before); `test_dr19_not_asked.py` 3 failed.
- The builder's tests check behaviour (points against where the fixture drew them), not wording alone.
- planlens has no fingerprint test; the app's is under "Could not check".

`python -m pytest -q -p no:cacheprovider planlens/tools/tests planlens/dxf planlens/review/tests` on `claude/rv-dr-44` at `24121bd`: exit code 0, 543 passed, 2 skipped, 0 failed.

## Scope

No task file or "Who edits what" table. All 15 files of the diff belong to the three items; no existing test was edited by the builder. I added `@needs_pillow` to six of the builder's new tests (finding 1). No words COMMON_DOCREVIEW forbids in the diff. The tool's description is general, not written around a model. Both new behaviours are off unless asked for.

## CHECK.md

There is none on the branch. The claims in the routine's prompt: "a host or toolkit that did not ask is offered and returns exactly what it did at the base" was untrue for one path (finding 4); "scaled to fit" untrue for small drawings (finding 6); "each result says it is measured from the drawn geometry" is true and is why finding 3 matters. The commit messages state no test counts.

## Could not check

- **Document Review against this branch** (its fingerprint test and the tests that read planlens' tool list): not run, for time. Build venv (a) from the app's repository, `pip install --no-deps --force-reinstall -e <planlens checkout>`, run `tests/test_review_fingerprints.py` and `tests/test_boundary.py`. Expect no movement: no spec changed for a host that did not ask.
- **A gain on real sheets** (DR-44's own condition): needs stroke-lettered sheets and a model.
- **The bounds on the testers' host:** the timings above are this container's.
- **Leader-to-note pairing with two leaders close together:** my made-up sheet found no leaders (my fixture, not shown to be the tool); only the builder's one-leader case is confirmed.
