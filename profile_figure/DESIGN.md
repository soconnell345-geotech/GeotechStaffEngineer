# profile_figure — design notes

Generic, parameter-driven subsurface profile schematic (matplotlib → PNG).
Built 2026-09-04 from field feedback: asked for "a figure showing the pile and
a subsurface profile", the agent had **no drawing tool at all** and delivered a
PDF with an `[image]` placeholder plus an HTML table with coloured rows
presented as a "figure"
(the 2026-09-04 downdrag field session's `FINDINGS.md` under
`module_work/field_feedback/`, item 3). This module is the capability that was missing.

## Scope

It draws whatever it is handed. It knows **no analysis** — no consolidation, no
capacity, no pore pressure — so every module family (downdrag, settlement,
axial pile, bearing, walls, liquefaction, …) can call it with the profile it
already has. Anything computed must be passed in, either as a layer
`description` or as an `annotation`.

**Every specialist has it (since BACKLOG GM-40, 2026-10-20).** It is in
`DELIVERABLE_MODULES` in `funhouse_agent/dispatch.py`, beside `calc_package`,
which `funhouse_agent/reviewers.py` adds to every specialist's scope; each
specialist's own `*_MODULES` set is unchanged.

The 2026-09-05 decision kept it out of every specialist scope: subsurface
stratigraphy is a geotech capability, the scopes are minimal by doctrine,
and the unscoped agent the prompt nudge was written for reaches it anyway;
add it to a scope only if the owner decides it belongs there. That reason no
longer held:
- **The owner decided.** GM-40 ("Specialists cannot draw or report") is the
  owner's backlog item.
- **The shared prompt already asks for it.** The deep agent's working style
  (`funhouse_agent/deep/prompt.py`), which every specialist's prompt carries,
  tells the agent to draw a layered profile with
  `call_agent('profile_figure', 'subsurface_profile', ...)`. A specialist
  told to do that and refused the module got "Unknown module".
- **Minimal scopes are about methods, and this module has none.** The scopes
  are kept small so a specialist does not run another discipline's analysis.
  This module computes nothing and draws only what it is handed, so it cannot
  widen a domain. For the geotech specialists (foundations, earth retention,
  slope / FEM, seismic) a subsurface profile is core: a pile, a wall or a
  liquefiable layer in the ground.

`worked_examples` is still in no specialist scope. The scope counts pinned in
`test_reviewers.py` count each specialist's own `*_MODULES` set and did not
move.

## Conventions

- **SI, elevations internally.** Every input may be given as an elevation
  (`*_elevation`) or as a depth below ground (`*_depth`); depths are converted
  with `elevation = ground - depth` at resolve time. `axis` controls only how
  the vertical axis is *labelled*.
- **`axis="auto"`** picks `depth` when the caller expressed the profile purely
  in thicknesses/depths, and `elevation` as soon as any elevation appears. The
  drawn geometry is identical either way.
- **Vertical scale is true; horizontal is schematic.** A 1-D profile has no
  width, so bands span a unit-width axis and a pile/footing is drawn at an
  exaggerated width (`_FOUNDATION_WIDTH`) with its real dimension in the label.
  The figure footer says so — do not let a reader scale a width off it.
- **Layers must stack continuously.** A gap or overlap raises a `ValueError`
  naming the layer and the size of the gap. A silently-drawn gap would be a
  misleading figure, and an LLM assembling a profile from a boring log gets
  this wrong in exactly that way.
- **Never colour alone.** Each band carries its name as text plus a distinct
  hatch from the house palette (`slope_stability.plotting`'s earth tones), so
  the figure survives greyscale printing and colour-vision deficiency.
- **Axis ticks sit at the layer interfaces**, not on a regular grid: the
  interface elevations are the numbers an engineer reads off a profile. Ticks
  closer than ~3 % of the plot height are thinned so labels never overprint.

## Structure

| File | Role |
|------|------|
| `geometry.py` | Resolution + validation. Pure Python, no matplotlib. |
| `plotting.py` | Drawing; house palette; `build_profile_figure` → `(fig, ax)`. |
| `profile.py` | `render_profile_figure` — resolve, draw, PNG (file + base64). |
| `results.py` | `ProfileFigureResult` (resolved geometry + image + `summary()`). |
| `data_plot.py` | `plot_data`: `render_data_plot` (PNG) + `build_data_plot_figure` (its Plotly twin); see below. |
| `figure_examples.py` | Worked examples on made-up data for the figure helper: six `plot_data` calls and three `plot_code` scripts; see below. |

`build_profile_figure` is exported so a module can drop the section into a
bigger figure or a calc package; `render_profile_figure` closes its own figure
(long-running hosts must not accumulate open figures).

## Embedding in reports

`render_profile_figure` returns **both** a saved PNG (`output_path`) and the
same image as base64 (`image_base64` / `data_uri()` / `img_tag()`), because the
two consumers differ:

- the web app renders a saved `.png` artifact inline in chat automatically
  (`webapp/core.py` `_ARTIFACT_KIND_BY_EXT`), so the agent should save it;
- `calc_package.html_to_pdf` needs an embeddable image. It now inlines a real
  local PNG/JPEG path for the caller, so an agent can write
  `<img src="/abs/path/profile.png">` and never handle base64 itself; a data
  URI still works for callers that already have one.

## `plot_data` — the generic data plot (added 2026-09-11)

Second method, same module, same contract (PNG on the real disk + an
`html_img_tag`), from the owner's 2026-09-11 feedback that calc packages lacked
"plots of data". `profile_figure/data_plot.py::render_data_plot` draws one or
more `{x, y, label, style, color}` series with matplotlib: `depth_axis=True`
inverts y and puts the x axis on top (boring-log convention), `logx`/`logy`
(ticks written 1, 10, 100), `hlines`/`vlines` as reference lines (a water
table, an allowable value), each named on the line itself and kept out of
the legend; the name stands at whichever end of its line, on either side,
covers least of the data, the notes and the other names (the right end
above a horizontal line, the top right of a vertical one, when clear), and
the interactive twin puts it at the same end (BACKLOG GM-152). It knows no analysis and does no unit conversion — the caller labels
the axes WITH units. Non-finite points are dropped with a warning; unequal
`x`/`y` lengths, an empty series and an unknown style are errors that name the
series index. The agent's adapter also refuses, before anything is written, a
figure that only holds a place: a title, axis or series named as a
placeholder (BACKLOG GM-190; the rule is `funhouse_agent/plot_box/figures.py`
`stand_in`). Axes named 'x' and 'y' alone are not refused: a plan or a
coordinate plot is named so.
`render_data_plot` itself does not check. It exists because the agent has no code-execution tool: without
it, SPT-vs-depth or settlement-vs-time in a bespoke report was impossible.
It lives here rather than in a new module because the agent catalog is within
~35 characters of its 8,000-char budget — a method costs less than a module.

The same cleaned series also build an interactive Plotly twin
(`build_data_plot_figure`, on `result.figure`; 5.18.0), which the adapter
writes beside the PNG as `<name>.plotly.json` and the chat shows; the PNG
stays the report copy. `interactive=False` skips it, and a failure to build
it is a warning that never costs the PNG. The twin sets its own white paper
and dark text, so a dark page theme cannot hide its title or axis labels.

**Plans, named points and cases (2026-10-15)**, from the owner's field session
that day (tests: `tests/test_data_plot_maps.py`):

- **Per series:** `marker` (circle, square, triangle, triangle-down, diamond,
  x, cross, star, hexagon), `marker_size` (points), `marker_edge`,
  `marker_open` (hollow), `line_dash` (solid, dash, dot, dashdot),
  `line_width`, `in_legend`. Style `polygon` closes an outline (a footprint,
  a site boundary, a zone) with an optional `fill` (true or a colour),
  `fill_opacity` and `hatch`; outlines are drawn under the points and lines.
- **`labels`** names each point beside it (`""` skips one): each name goes
  in whichever of eight spots round its point covers least of the other
  names, the points, the lines, the reference lines (BACKLOG GM-146: a
  vertical LL = 50 line ran through two names), hatched outlines, the
  reference-line names, the plan's furniture and the notes with their
  leaders, and stays inside the plot frame; a line through a name's letters
  costs more than one in the margin round it. A name that can only stand on
  hatching (its point inside a hatched zone) gets a thin ring of the
  background round its letters, as a note's `halo`, in the PNG only. The
  Plotly twin puts each name in the same spot. **`hover`** is one
  text per point, shown on hover in the chat. Both need one entry per point,
  and a dropped point drops its name.
- **x given as names** (cases, scenarios, borings) puts them along x in the
  order first seen; every series uses names or every series numbers.
- **`equal_aspect`** draws one unit across as long as one unit up (a plan, a
  map, a section to scale) in a figure of the data's own proportions unless
  `height_in` is given; it is ignored, with a warning, on cases or a log axis.
  **`xlim`/`ylim`** are `[start, end]` in the order given: `[100, 0.001]`
  runs the axis from high to low, on a log axis too, in the PNG and the
  Plotly twin, and the result's `reversed_axes` and summary say so (live wave
  w9: a reversed range was sorted without a word; the twin ignored `xlim` on
  a log axis); with `depth_axis` y runs downward whatever the order, and a log
  range at or below zero is left out with a warning (tests:
  `tests/test_data_plot_axes.py`). **`x_on_top`** puts the x axis along the
  top alone, as a boring log reads, on a profile against elevation (BACKLOG
  GM-106). **`legend_position`** `auto` (outside a drawing to scale,
  inside otherwise), `inside`, `outside` or `below` (under whatever the x
  axis takes up, slanted case names included).
- **`plan_view`** draws a plan as a drawing, not a chart (the owner, after
  Tufte: "don't use unnecessary ink"): to scale, with no axes, axis numbers, grid or
  frame, and the title at the left. In a band below the site sit a scale bar
  of four black and white segments (a round 1, 2 or 5 × 10^k length, at most
  a third of the width, named only at its ends, in `scale_units`) with
  `xlabel`, the coordinate system, under it, and beside it a north arrow at
  `north_deg` (degrees clockwise from up, as the plan the data came from or
  the project's other plans show it), or straight up with a warning saying so
  when none is given. Nothing is drawn on the site itself. The view takes in
  every outline, line and point drawn with a margin all round, and the band
  and margins are sized in points as printed (`_PLAN_PT`), so the bar's end
  names stand clear of the site on a long flat site and the arrow clear of
  the bar on a narrow one; the equal aspect fits the axes' box and never
  trims the view (BACKLOG GM-147: a 40 x 25 m outline lost its top edge,
  the view stopping at 24.76). It is ignored, with
  a warning, on named cases, log axes or a depth axis. The scale bar and
  arrow are `geotech_common/plan_furniture.py`, which the gstools probability
  map shares, so the plans read alike.
- **`annotations`** places free text (BACKLOG GM-109; live wave w9 carried a
  plasticity chart's zone names on series of invisible markers, then moved it
  to `plot_code`): `[{text, x, y}]` in data units (a case's name for x on a
  case axis), or with `coords: "axes"` in fractions of the plot (0, 0 bottom
  left). Optional: `point` ([x, y], always data units), the point the note
  marks, drawn as a leader arrow from the text; `align`, the side of the text
  that stands at x, y (`center`, `left`, `right`, `top`, `bottom`, or one of
  each); `box` (true for white, or a colour) a plain box with no edge behind
  the text; `color`; `font_size` (pt, 6–20). A new line or `<br>` breaks a
  line. Unknown keys are refused with a did-you-mean, x and y must be finite
  numbers (above zero on a log axis), text is required. Notes are drawn on
  every mode, plan views included, and never moved: point names treat them
  and their leaders as fixed obstacles. The Plotly twin carries each note at
  the same place, side and box (log10 positions on a log axis; `x domain` /
  `y domain` for fractions; with a leader the tail sits in the head's units,
  so a note given in fractions with a leader is converted through the PNG's
  view and the twin is pinned to that view). A layout check after the draw
  reports in `warnings` a note that overlaps other lettering (another note,
  a point name, a reference-line name, the title, axis titles, tick numbers,
  the legend), one that a line, markers, an edge, hatching or another note's
  leader shows through (named: the series, the reference line, `the leader of
  annotations[j]`; a line under the note's own box does not show), one the
  figure edge cuts, one that stands outside the axis limits and one whose
  point lies outside them. The first three are judged by the pixels drawn,
  with the plotting box's own check (`geotech_common/layout_ink.py`, GM-108)
  at the PNG's dpi, about 50 ms a figure (the review of GM-109: a box check
  made 21 false reports in 236 one-note figures: rotated case names, legend
  padding, a two-line note's empty band, a boxed note under the title); the
  last two by the numbers. It moves nothing, and a failure of the check is
  a warning, never a lost figure. A `font_size` outside 6-20 pt is drawn at
  the nearer end with a warning (tests: `tests/test_data_plot_notes.py`,
  including two w9 figures redrawn). The same check covers the lettering
  `plot_data` places itself (BACKLOG GM-152): the reference lines' names
  (horizontal and vertical), the plan's scale-bar names, north arrow and
  coordinate system, and the point names, each warning naming the piece
  (`the name of the reference line at 50 ('LL = 50')`, `the scale bar's
  '20 m'`, `series[3] point name 'B-2'`) and what to do; a piece that meets
  a note is reported once, on the note. Tests:
  `tests/test_data_plot_lettering.py`.

**Worked examples** (`figure_examples.py`, the owner's option B of
2026-10-15): six complete `plot_data` calls on made-up data (`EXAMPLES`),
each with what it shows and why it is drawn that way: `plan_borings`,
`spt_profile`, `cases`, `time_settlement`, `load_settlement` and
`exceedance`. The figure helper (`funhouse_agent/deep/figure_helper.py`,
behind `GEOTECH_FIGURE_HELPER`, ON by default since 2026-10-20) fetches one by name through
its `figure_examples` tool, a near name matched by shared words and then by
spelling, and adapts it; `render_examples(out_dir)` draws them all. Every
number in the file is invented.

**The owner's conventions for profiles and design figures** (BACKLOG GM-97,
2026-10-15) are in the examples and the helper's checklist: elevation on a
profile's vertical axis by default ("most borings have a slightly different
ground surface elevation"), depth only when no elevations are given, never
both on one profile; "kPa and kN (everywhere!)"; preconsolidation stress as
test points when plotting the data and a line when it is an assumed design
profile, the figure saying which; the wall drawn by default, its pressure
diagram alone only when the wall's configuration is not known.
`spt_profile` (formerly `depth_profile`; "depth profile" still finds it) is
three borings with different ground levels against elevation, each test's
depth kept in its hover text.

**Code examples** (`CODE_EXAMPLES`, GM-97): figures `plot_data` cannot draw,
each a complete matplotlib script for `plot_code`'s locked box with the
`data` it reads, what it shows and why: `wall_pressures` (a cantilever wall
to scale with its active earth, surcharge and water pressure diagrams beside
it on one elevation axis and one kPa scale, each resultant in kN/m at its
elevation; the "why" says how to draw the diagrams alone),
`settlement_stress_profile` (the soil column and its fill, then the initial
vertical effective stress, the preconsolidation stress from consolidation
tests, the stress increase and the final stress against the preconsolidation
stress;
the script draws an assumed design line too when the data gives one) and
`pile_design_profile` (the soil column with the pile, then unit shaft
resistance, the axial resistance down the pile and the axial load with the
neutral plane where the load curve meets the resistance curve). The numbers
are made up but consistent (Rankine pressures, effective stresses, the
plane where the two curves cross). `example(name)` returns either kind with
its `kind`; the helper lists the code examples only when it has
`plot_code`. The tests draw each through the real box and require an empty
layout check (`funhouse_agent/deep/tests/test_figure_helper.py`).

## Deliberate non-goals

- **Not a boring log.** No sample intervals or lab-results columns on the
  schematic. `subsurface_characterization` owns real log data and its Plotly
  depth profiles; `plot_data` can draw a plain blow-count-vs-depth line, which
  is a chart, not a log.
- **Not a cross-section.** Horizontally varying stratigraphy, slope geometry
  and slip surfaces belong to `slope_stability.plotting` / `dxf_export`, which
  draw true 2-D sections. This is the 1-D column at a boring/pile location.
- **No unit conversion.** SI in, SI out, like every other analysis module
  (`pavement_design` is the documented US-customary exception).
