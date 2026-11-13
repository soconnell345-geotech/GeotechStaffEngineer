# GM-315: the fixed plotting tool draws what only plot code drew

Branch `claude/gm-315-plot-data-panels`, from master 0dac9c2. Everything here
is drawing code and tool text; no calculation changes.

## What the item asked, and what is on the branch

`plot_data` drew one plot. Its own description said it "cannot draw a
structure to scale with diagrams beside it, or several panels sharing one
axis", and contours came only from model-written plot code, which is off for
Document Review on the shared hosts. It now draws all three from a JSON
description, with no code.

| Asked | Built | Where |
|---|---|---|
| Panels sharing the vertical axis: own series, x label, short title, reference lines, notes; relative widths; one x scale for one quantity; a line across every panel; a figure title and a footnote | `panels` (a list, one dict per panel), `width`, `share_x`, the figure's `hlines`, `title`, `footnote` | `charts/panel_plot.py` (new), reached from `render_data_plot` |
| A section to scale beside its diagrams; a fill between a curve and a base line; layer bands with a name and a hatch; a dimension line; an arrow for a resultant | panel `kind: "section"` (and `"column"` for a soil column); series `fill_to_x` / `fill_to_y`; panel `bands`, `dimensions`, `arrows` | `panel_plot.py`; the fill is in `data_plot.py::_draw_series`, so a single plot has it too |
| Contours from a grid or from scattered points (matplotlib's `tri`, no scipy): lines or filled, chosen or automatic levels, labelled lines, a named colour scale, to scale, outlines and points over it | `contour: {x, y, z, label, levels, filled, line_labels}` on the single plot, under the series | `data_plot.py` (`_clean_contour`, `_draw_contour`, `_contour_numbers`) |
| Existing inputs draw what they drew | The series loop moved into `_draw_series` with no change; the 156 existing `profile_figure` tests pass unchanged | `data_plot.py` |
| The interactive twin where it is straightforward, none otherwise, said in `warnings` | Twin for chart panels (subplots sharing y) and for grid contours at even levels; none for a section or column panel, bands, dimension lines, scattered or unevenly spaced contours | `panel_plot.py::build_panel_figure`, `data_plot.py::_plotly_contour` |
| The tool text, at most about 900 characters more | 5,298 to 6,157 characters (+859) | `plot_tool.py`, `vision_kit.py` |
| Every sentence that said only code draws these | `_PLOT_APP_NOTE` and the redraw note (`vision_kit.py`), the helper's note and its examples tool's description (`figure_helper.py`), the plotting box's description (`plot_box/__init__.py`), the calc helper's text and description (`funhouse_agent/deep/agent.py`) | see "What the model reads" |

One more key came with the section example: a note's `rotation` (a name
along a vertical plane).

## Calls made

No held reference prints a rule for any of these; each is a layout or naming
choice, stated in `profile_figure/DESIGN.md` ("Panels, sections to scale and
contours") and listed here for the lead to revisit.

1. **A new module.** The panel renderer is `charts/panel_plot.py` rather
   than more branches inside `render_data_plot`, so the single-plot path is
   untouched. It reuses that path's series cleaning and drawing, notes,
   reference-line naming and point-name placing.
2. **Three top-level keys**: `panels`, `contour`, `footnote`. `series` is no
   longer required when `panels` or `contour` is given (the page tool's
   `series` argument is optional now); with neither, a call with no series
   is refused as before.
3. **A chart panel's x axis starts at zero** when every value is of one sign
   and `xlim` is not given (the checklist's item 4: magnitudes compare
   truly). `xlim` overrides it.
4. **Sizes.** Axes 3.6 in high unless `height_in` is given. Sections take at
   most 62 % of the room beside chart panels: past that the axes are made
   lower (the section never goes off scale) and a `height_in` so reduced is
   said in the warnings. A chart panel under 0.75 in wide is refused; at
   most 8 panels.
5. **No grid and no legend in panels by default** (names go on the data). A
   panel's own `grid: true` draws one; the figure's `grid` is not used with
   panels.
6. **Axis numbers** that would run together in a narrow panel are thinned,
   and a panel's last number is left out where it would run into the next
   panel's first.
7. **`hlines` are named once, in the last panel.** A caller who wants the
   name elsewhere leaves the label empty and writes a note (the pile example
   does).
8. **Single-plot inputs given with `panels` are refused by name**, each with
   where it goes instead, rather than ignored.
9. **Contours.** Automatic levels are round numbers (matplotlib's
   `MaxNLocator`, 7 by default). Lines are numbered at the middle of each
   stretch long enough to carry a number, after the layout has settled,
   moved along the line off a drawn point, each on a ring of the
   background; a number the plot's edge would cut is taken off, and a level
   left with none is said in the warnings. Lines drawn alone are named in
   the legend; filled contours by their colour scale (viridis in both
   renderings), with no grid.
10. **`footnote`, `fill_to_x` / `fill_to_y` and a note's `rotation` work on
    every `plot_data` figure**, not only with panels.
11. **No twin is a warning that says there is nothing to fix**: "no
    interactive chart for this figure (...): the chat shows the PNG; nothing
    to fix". The brief asked for it in `warnings`; the wording is so a
    helper does not redraw to clear it.
12. **Example names and matching.** The four new examples are
    `wall_section_pressures`, `settlement_stress_panels`,
    `pile_design_panels` and `contour_top_of_rock`. Where plot code is on,
    a loose name ("settlement profile") now finds the `plot_data` example
    first; a script is still fetched by its own name. Plot code remains for
    what `plot_data` cannot draw.
13. **What `plot_data` still cannot draw**, as the tool text now says:
    panels stacked one above another, two vertical axes on one plot, bars,
    a coloured grid of cells, arcs and angles, a layout of the caller's own.
14. **`defaults_used`.** With `panels`, the single-plot defaults (`grid`,
    `logx`, `logy`, `equal_aspect`, `plan_view`, `north_deg`,
    `scale_units`, `x_on_top`, `legend_position`) are left out
    (`applies_when`), and a left-out `series` is listed as "no series of the
    figure's own: panels or contour gives what is drawn".
15. **"layer bands", not "soil column", in the tool text**: the Document
    Review page's tool descriptions carry no geotechnical words
    (`test_review_page_vocabulary_offline.py` caught my first wording).
16. **The manual.** Only the two `plot_data` rows of
    `docs/user_manual/manual.html` were replaced, generated with the
    manual's own `method_param_table`; no rebuild.

## The acceptance: the code examples, redrawn without code

Scripts: `render_code_examples.py` (the three scripts through the plotting
box) and `render_plot_data_examples.py` (the `plot_data` examples; exits
non-zero on any warning other than the no-twin note, or on lettering that
prints under 7 pt). The pairs are in this folder.

| Figure | Script, through the box | `plot_data`, no code |
|---|---|---|
| Wall and its pressures | `code_wall_pressures.png` | `plot_data_wall_pressures.png` |
| Settlement stresses | `code_settlement_stress_profile.png` | `plot_data_settlement_stress_profile.png` |
| Pile design profile | `code_pile_design_profile.png` | `plot_data_pile_design_profile.png` |
| Contours (new, made-up numbers) | none | `plot_data_contour_top_of_rock.png` |

The `plot_data` examples are built from the scripts' own `data`, so both
renderings of a figure show one set of made-up values.

**Measured on each `plot_data` figure**

| Figure | Layout check | Smallest lettering | Saved width | Prints at 6.5 in |
|---|---|---|---|---|
| wall_section_pressures | empty | 7.5 pt | 6.63 in | 7.35 pt |
| settlement_stress_panels | empty | 7.5 pt | 6.64 in | 7.34 pt |
| pile_design_panels | empty | 7.5 pt | 6.71 in | 7.26 pt |
| contour_top_of_rock | empty | 7.0 pt | 6.41 in | 7.00 pt |

**Against the checklist's eleven items** (I opened each PNG and its pair)

| Item | Wall | Settlement | Pile | Contour |
|---|---|---|---|---|
| 1 The data first | yes | yes | yes | yes |
| 2 Data-ink | yes: no grid, no frame | yes | yes | yes: lines only |
| 3 Labels on the data | yes: ordinates, resultants, plane and groundwater named in place; no legend | yes: lines named in their colour | yes | numbers on the lines, names at the borings; a legend for the three kinds of mark |
| 4 Scales tell the truth | section to scale, one pressure scale from zero | one stress scale from zero; elevation up | one force scale; elevation up | a plan: to scale, scale bar, north arrow, no axes |
| 5 Name everything | kPa, kN/m, elevation datum | kPa, datum | kPa, kN, datum | the quantity with its unit and datum; the coordinate system under the bar |
| 6 Lines mean continuity | yes | tests are open points, never joined | yes | contours are lines between borings, and the footnote says so |
| 7 Small multiples | three diagrams, one scale | four panels, one scale | two force panels, one scale | not a multiple |
| 8 Colour and shape | one colour per pressure | points open, lines solid or dashed | heavy and thin lines | one line colour, marks by shape |
| 9 Mark what the text quotes | resultants and their elevations | preconsolidation tests | nominal resistance, neutral plane, largest load | the levels |
| 10 The project's own words | the script's own | the script's own | the script's own | made up |
| 11 Readable at report size | yes (table above) | yes | yes | yes |

**What matches, and what differs from the script's figure**

- The same content, panel for panel, in all three: the section or column,
  the diagrams, every name, value, arrow, dimension and note, the title and
  the footnote.
- Differences I left: the typeface (the box's default against the app's);
  the axis numbers are thinned by rule, where the scripts list their ticks
  (the settlement panels show 0, 50, 100, 150 in both; the last panel also
  shows 200); the default axes are 3.6 in high where two scripts use
  3.95 in, because the page tool has no `height_in`; the dimension line's
  witness lines start a little off the base, as drafting does, where the
  script's start below it.

**What `plot_data` still cannot do for these figures**

- An interactive twin of a figure with a section or column panel: the chat
  shows the PNG.
- Tick positions chosen by the caller; a figure height from the page tool
  (`width_in`, `height_in`, `dpi` and `grid` are not arguments of the page's
  `plot_data` tool; they are of `call_agent`'s).
- A reference line named in a panel of the caller's choosing other than by
  a note.
- The assumed design line in the settlement script is not a separate input:
  it is one more line series and a note.

## What I looked at, and what I changed after looking

Every figure was rendered to PNG and opened; the twins were written to HTML
and screenshotted in a browser.

- First wall render: axis numbers too sparse (0, 20, 40) and "40" close to
  the next panel's "0"; point names placed by the automatic placer landed
  inside the filled diagram. Changed: denser ticks by panel width, the end
  number left out where it meets the next panel's first, and the example
  uses notes for the ordinates.
- Settlement: vertical axis numbered 2.5, 17.5, 22.5 (now round steps); the
  name of the preconsolidation points sat on the final-stress line (moved
  10 kPa right in the example).
- Pile: force numbers "1,000 1,500 2,000" ran together (numbers are now
  thinned until they stand apart).
- Contours: numbers were turned to the angle the line had before the layout
  settled, and matplotlib put them at line ends and on the plot's edge (now
  numbered after the layout, at mid-length, and taken off where the edge
  would cut them); the footprint ran through a number (a ring of the
  background round each number); "94" sat on a boring (numbers move along
  the line off a drawn point); point names crossed contour lines (the lines
  are obstacles for the placer now); the footnote of a plan stood far below
  the scale bar (it follows the plan's furniture now).
- Filled contours: the grid showed through the bands (no grid over filled
  contours); automatic levels came out at 0.8 steps (round steps now).
- Twins: the chart-panel twin and the filled-contour twin match their PNGs
  (same panels, fills, arrow, reference line and its name, bands of colour
  and scale).
- Lettering: the first figures printed at 6.9 pt at 6.5 in, because the
  saved PNG is a little wider than the figure; the footnote, band names and
  the examples' small notes are 7.5 pt now.

## What the model reads

- `plot_data`'s description (both pages): the brief names panels and
  contours; `series` says "Required unless panels or contour is given" and
  names `fill_to_x` / `fill_to_y`; a note has `rotation`; new entries
  `panels`, `contour`, `footnote`. `_PLOT_APP_NOTE` now says what it draws
  and what it cannot.
- The redraw note inside the `plot_data` tool (third redraw of one figure)
  no longer says a structure with diagrams needs plot code.
- `figure_helper.py`: `PLOT_CODE_DESCRIPTION_NOTE`, and the `figure_examples`
  tool's description (it lists the kinds of example, and said the scripts
  are "for figures plot_data cannot draw"). The checklist's wording is
  unchanged.
- `plot_box/__init__.py`: `plot_code`'s description.
- `funhouse_agent/deep/agent.py`: `_CALC_PLOT_CODE` and the calc helper's
  description sentence beside it.

## Tests

Judged by pytest's exit code. Venv (b), the core with the geotech app:

| Run | Result |
|---|---|
| `profile_figure/tests` (the six existing files and the new `test_data_plot_panels.py`) | 208 passed |
| The named set: `profile_figure/tests`, `geotech_common/tests/test_layout_ink.py`, `packages/aec-agent-core/tests`, `funhouse_agent/deep/tests/test_figure_helper.py`, `funhouse_agent/tests/test_defaults_used.py`, `webapp/tests/test_page_fingerprints.py`, `test_public_text.py`, `test_review_freeze.py`, `test_docs_currency.py`, and every test file that names `plot_data` (list in the commit history's test runs: plot box, figure checks, stand-ins, nested keys, hint targets, reference chart, the deep-agent figure and tool files, the webapp's core and card files, `calc_package/tests/test_story_figures.py`) | 2,066 passed, 23 skipped, 1 failed (below); both pages' fingerprint tests pass with the re-recorded file |
| Venv (a), `packages/aec-document-review/tests` | 940 passed, 6 skipped (the review fingerprint test passes with the re-recorded file) |
| Venv (a), `packages/aec-agent-core/tests` (the core-only install) | 282 passed |

**One failure, which master has too:**
`funhouse_agent/tests/test_defaults_used.py::test_every_default_a_run_lists_changes_its_result[profile_figure|plot_data]`
reports that `logx` and `logy` change nothing in the result of its probe. It
fails the same way on master 0dac9c2 in this environment (checked with my
changes stashed). Not touched.

**New tests** (`profile_figure/tests/test_data_plot_panels.py`, 52 tests):
all 51 that existed when I checked fail on master (run in a worktree of
origin/master: 51 failed) and pass on the branch; the 52nd
(`test_defaults_used_lists_only_what_a_panel_figure_used`) calls `panels`,
which master refuses.

**Existing tests edited**, each because the change made an assertion untrue:

- `packages/aec-agent-core/tests/test_core_charts.py`: the count of worked
  examples, 9 to 13.
- `funhouse_agent/deep/tests/test_figure_helper.py`, three places: every
  example "draws with no warning" now allows the one no-twin note and
  expects no `.plotly.json` for that figure; "every example has an xlabel"
  accepts each chart panel's own; the loose-name expectations follow call
  12 above.

## Fingerprints

Both recorded files move (the `plot_data` description and arguments on both
pages, the figure helper's description and its `figure_examples` tool, and on
the geotech page `plot_code` and the calc helper). They are re-recorded in
the LAST commit, with nothing else in it, so it can be dropped at the merge.
The Document Review file's `planlens` stamp also moved from 0.15.0 to 0.15.1:
that is the version its pin installs today, not something this change did.

## Merge with `claude/gm-316-317-warnings-allowed-values`

That branch adds one line to `plot_tool.py` (`"any_case": True` under
`legend_position`) and changes one line of `profile_figure_adapter.py`. I
changed neither place; `profile_figure_adapter.py` is untouched on this
branch (it reads `VALID_PLOT` and `PLOT_DATA_INFO` from the core, so the new
keys reach `call_agent` with no edit).

## Release note

None: no calculated value changes. For the next release's notes: `plot_data`
draws panels on one elevation axis (a wall or pile to scale with its
diagrams, a soil column beside stress panels) and contours, with no
model-written code, so hosts with plot code off get these figures.

## Noted, not fixed

- `funhouse_agent/tests/method_probes.json`: the `plot_data` row is
  unchanged; its probe still runs as written.
- Mathtext (`$P_A$`) is drawn by the PNG and shown raw by a Plotly twin.
  Single plots had this before; the panel examples that use it have no
  twin.
- With `plan_view`, a filled contour's colour scale is as tall as the whole
  drawing, the scale-bar band included.
- The figure helper's prompt is unchanged apart from the sentences named: it
  learns panels from the tool description and the examples. Whether figures
  got better or worse is GM-314's to measure.
- The page tool has no `height_in`, `width_in`, `dpi` or `grid` argument
  (before this change too).

## Review fixes (2026-11-13)

Independent review: `notes/review_gm315.txt` in the drawings lead's folder (not in the repo). One line per finding:

- B1 fixed: a single plot's footnote is broken to the figure's width (`data_plot.py`, panel_plot's `_fit_text`); the contour example's PNG is 6.37 in wide, smallest lettering 7.0 pt as printed (was 6.85 in, 6.64 pt; its test failed).
- B2 fixed (test): `test_fill_bands_dimension_and_arrow_are_drawn` put the arrow's tail at 25 kPa, and its text crossed the diagram (the layout check was right); the tail is now at 34 kPa. B1 and B2 were the main CI job's failures.
- B3 fixed: `reference_chart` with panels, or with a contour and no series, is refused by name in `run_plot_data`; it crashed with KeyError 'series' in the overlay.
- B4 fixed: a contour grid over 250,000 values, more than 50,000 scattered points or more than 30 chosen levels is refused in plain words; at most 3 numbers per level (longest lines first), none past 300 separate pieces (warned). A noisy 100 x 100 grid: 100 s before, 0.8 s now.
- B5 fixed: a panel's title or xlabel, a band label, dimension or arrow text that is not text, and a bad band or arrow colour, are refused under their own place (`panels[0] bands[0] label must be text`), not as "footnote" or "annotations[0]".
- B6 fixed: `legend`'s description says that with panels there is a legend only when it is true (+29 characters; both pages' fingerprints must be re-recorded at the merge).
- B7 not fixed, for the lead: where plot code is off, `nearest_example` attaches no example to a pile or settlement brief (the redraws' names need 'panels'); the fix changes `funhouse_agent/deep/tests/test_figure_runs.py:198` and `figure_helper.py` code, or renames the examples.
- C1 not fixed: `to_dict()` carries `contour`, `panels` and `footnote` on every result; the tool's response is unchanged.
- C2 fixed: `width_in` in the panels' echo is a plain float.
- C3 fixed: one chosen level has no twin "at a single level", not "unevenly spaced".
- C4 fixed: an all-missing z grid is refused as "has no values", with no numpy warning.
- C5 fixed: points on one line are refused in plain words, without qhull's own message.
- C6 not fixed (scope): `docs/user_manual/manual.html` was regenerated, and the figure_examples tool's description changed beyond the named sentences; both consistent.
- C7 not fixed: with panels, `legend_position` and `scale_units` are ignored without a word (the tool function always sends them).
- C8 not fixed: `filled: "false"` (a string) reads as true, as the existing flags do.
- C9 not fixed: the settlement footnote's last word stands alone on its line (cosmetic).
- The four `plot_data_*.png` pairs were re-drawn with `render_plot_data_examples.py` after these fixes, on the review machine (Windows, matplotlib 3.10.9). The committed ones were a different size for all four (the contour 961 x 528 px against 1028 x 506 here before B1's fix), so the builder's session laid out lettering differently from this machine.
