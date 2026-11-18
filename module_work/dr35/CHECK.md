# DR-35: Document Review's own worked examples for the figure helper

**What the item asked.** The owner's decision OWN-77, option (b): after the
split, Document Review gets general worked examples for its figure helper
(drawings, schedules, quantities) in place of the nine geotechnical ones it
carried along from the core, measured on the review suite. The lead's
ruling: the examples are text the helper's model is handed and the change
is not measured, so it sits behind one new switch, off by default; unset,
the page's figure helper is handed exactly today's text; the geotech page
keeps its examples whatever the switch says.

## What was built

- **A hook in the core** (`packages/aec-agent-core/aec_agent_core/deep/figure_helper.py`).
  `build_figure_subagent`, `FigureTasks`, `make_figure_examples_tool`,
  `nearest_example` and `brief_with_example` take `examples=`, an example
  set: a module or object with `EXAMPLES` and `CODE_EXAMPLES` shaped as in
  `aec_agent_core.charts.figure_examples`, and optionally `REDRAWS` and
  `TOOL_KINDS`. New helpers `_example_set` (the set handed in, or the
  core's own) and `_example_of` (one example, deep-copied, with its name
  and kind, as `figure_examples.example` gives it). `None`, the default,
  is the core's geotechnical set, so every caller that hands nothing in
  (the geotech page's two builders) is unchanged.
- **The tool description's list of kinds** moved, word for word, into
  `charts/figure_examples.py` as `TOOL_KINDS`; the tool reads it from the
  set. A set with no scripts says nothing of scripts in the description or
  the listing, even when the helper has `plot_code`.
- **Document Review's set**, `packages/aec-document-review/aec_document_review/figure_examples.py`,
  chosen from what the review suite's tasks ask a reviewer to produce
  (measure on a scaled plan, schedules against plan marks, levels and grid
  dimensions sheet to sheet, a comment log by sheet, a submittal against
  its specification, revisions between issues):
  - `plan_dimensions`: a floor plan's column grid redrawn to scale from the
    dimensions on the sheet, the exterior wall line, and a distance
    measured between two points (plan view, scale bar, north arrow at the
    sheet's angle, grid names at the ends of the grid lines, bay dimensions
    written between the lines, the distance written beside its line);
  - `schedule_count`: a door schedule against the tags counted on the floor
    plans, by type (two groups on a case axis, only the disagreeing types
    labelled, axis from zero);
  - `comments_by_sheet`: review comments by sheet in the sheet index's
    order, one colour and shape per discipline (quantities by sheet);
  - `values_against_limit`: the 28-day strength of each mix in a concrete
    submittal against the specified strength (the limit a named reference
    line, the one value below it labelled, axis from zero);
  - `level_by_sheet`: one value as each sheet gives it, the Level 2 floor
    level on the elevations and sections against the floor plans (an axis
    that does not start at zero, and the footnote says so);
  - `submittal_timeline` (a `plot_code` script): a submittal log as bars on
    a date-lettered axis, time with the reviewer and with the contractor,
    the part of a review past the allowed period in its own colour with the
    days over written just above it.
  Every number, sheet and name is made up. No plot_data redraws of the
  script (`REDRAWS` is empty).
- **The switch** `GEOTECH_REVIEW_FIGURE_EXAMPLES` (`review_flags.py`:
  `FIGURE_EXAMPLES_ENV`, `review_figure_examples()`, documented in the
  module docstring, in `LATER_ENVS` beside the three newest, not in
  `ALL_ENVS` or `ARMS`). Default off; `1` turns it on.
- **The page** (`packages/aec-document-review/aec_document_review/review_build.py`):
  with the switch on, the figure helper and `FigureTasks` (the worked
  example attached to each brief) get Document Review's set; unset, both
  get `examples=None`, the core's own.
- One row for the switch in `packages/aec-document-review/README.md`'s
  settings table.

## Calls made

- **Where the set lives.** In `aec-document-review`, not the core: the core
  names no application, and the set is this page's. The core keeps the
  geotechnical set as its default so the geotech page and every other
  caller is untouched.
- **A hook argument, not a registry in `aec_agent_core.hooks`.** The page
  already hands the helper its own tools and prompt by argument
  (`extra_tools`, `extra_system_prompt`); the set goes the same way, and
  the two places that need it (the helper and `FigureTasks`) get the same
  object from one build.
- **The helper's prompt and description are not changed** by the switch:
  only the examples. The prompt still opens "You are the FIGURE HELPER of a
  geotechnical agent" on this page (noted below).
- **One script in the set**, so that the helper's step for `plot_code`
  ("figure_examples lists worked plot_code scripts to start from") stays
  true on this page; a submittal timeline needs bars on a date axis, which
  `plot_data` does not draw. The script letters a day-count axis with dates
  itself instead of using `matplotlib.dates`: on Windows the date locator
  reads the registry for the time zone, which the plotting box refuses.
- **Five plot_data examples, not nine.** Chosen by the kinds of task, kept
  apart from each other (a drawing, a two-group count, a grouped quantity,
  a limit check from zero, a consistency check not from zero), so that the
  text handed over is smaller, not larger.
- **The level example names its reference line with a note under it**, not
  on the line: the points sit on the line, and `plot_data` warns that the
  line's own name is crossed by them.

## Sizes (characters the helper can be handed)

Measured by `test_the_text_stays_near_the_core_sets_size` (printed with
`-s`):

| | description | listing | every example in full | examples |
|---|---|---|---|---|
| core set, no plot_code | 466 | 2,393 | 28,384 | 10 |
| Document Review's, no plot_code | 424 | 1,354 | 10,076 | 5 |
| core set, with plot_code | 640 | 3,369 | 55,880 | 13 |
| Document Review's, with plot_code | 598 | 1,702 | about 15,100 | 6 |

The description is 42 characters shorter; the listing about half; the
worked example attached to a brief is one example's `shows` and `why`, of
the same length as before.

## Settings

| Setting | Default | Turns on |
|---|---|---|
| `GEOTECH_REVIEW_FIGURE_EXAMPLES` | off | `1` (also `true`, `yes`, `on`) |

## Tests

`packages/aec-agent-core/tests/test_dr35_example_sets.py`:
- `test_with_no_set_the_tool_is_handed_exactly_what_it_was` (with and
  without plot_code): the tool's description equals the pre-DR-35 text
  word for word, the listing equals the one built from the module's own
  functions, and each example in full is `figure_examples.example`'s.
- `test_with_no_set_the_attached_example_is_what_it_was`: `_example_of`
  equals `figure_examples.example` for every example; a plan brief still
  gets `plan_borings` attached; `FigureTasks()` holds no set.
- `test_a_hosts_set_replaces_the_core_set_everywhere`: a made-up one-example
  set replaces the core's in the listing, the description, the fuzzy name
  match, the nearest example and the attached brief; a set with no scripts
  says nothing of scripts.
- `test_the_switch_is_off_by_default_and_kept_apart`: in `LATER_ENVS`, not
  in `ALL_ENVS` or any arm; off unset; `describe()` names it; `switches({})`
  clears it.
- `test_the_core_default_ignores_the_switch`: with the switch on,
  `build_figure_subagent()` with no set (as the geotech page builds it)
  still lists the core's examples with the old description; a set changes
  only the examples tool, not the helper's prompt or description.

`packages/aec-document-review/tests/test_dr35_figure_examples.py`:
- `test_every_example_draws_cleanly_through_the_real_tool`: each plot_data
  example through the helper's own strict `plot_data` tool: success, no
  warning, a PNG and its interactive chart.
- `test_the_script_draws_cleanly_in_the_box`: the script in the real
  plotting box: passes the code check, success, nothing on the layout
  check, nothing refused, no warning, one PNG.
- `test_the_examples_follow_the_figure_checklist`: plan view with north and
  the coordinate system; the distance written equals the distance between
  the two points; case axes are markers only; counts and the limit check
  start at zero, the one axis that does not says so; each reference line is
  named with its value; the comment counts add up to the footnote's total.
- `test_a_brief_finds_its_example`: plain briefs find the right example
  (`nearest_example` with the set), the timeline with plot_code.
- `test_the_text_stays_near_the_core_sets_size`: listing and full text no
  larger than the core set's; prints the sizes above.
- `test_unset_the_page_is_handed_what_it_was`: the page built with every
  switch unset gives its figure helper the core's examples tool (same
  description, same listing) and `FigureTasks` no set.
- `test_switched_on_the_page_gets_its_own_examples`: with the switch on,
  the page's helper lists Document Review's set and `FigureTasks` holds
  it; the helper's prompt and description are the same as unset.

On master every one of these fails (the set, `examples=`, `TOOL_KINDS`,
`_example_of` and the switch do not exist there); the byte-for-byte parts
of the two "no set" tests would pass on master, which is their point.

Run here, with the switch unset: Document Review's
`tests/test_review_fingerprints.py` passes (no fingerprint moved),
with `test_review_switches_offline.py`, `test_figure_code_cards.py` and
`test_dr48_photo_pages_turn.py`; in the core and the app,
`test_core_charts.py`, `test_core_imports.py`, the three `LATER_ENVS`
tests, `funhouse_agent/deep/tests/test_figure_helper.py`,
`test_figure_runs.py`, `test_figure_turn.py` and
`webapp/tests/test_page_fingerprints.py`.

## Noted, not fixed

- The figure helper's prompt opens "You are the FIGURE HELPER of a
  geotechnical agent" on the Document Review page too, and its checklist
  speaks of borings and profiles in its examples of each rule. Not changed
  here: the item and the switch are about the examples. Worth its own item
  only if a measurement shows the helper drawing a review figure the
  geotechnical way.
- The Tiny Apps template's settings list (`tinyapps/`) does not name the
  new switch; that folder is outside this item's reach.

## For the lead: what a measurement should compare

- The suite arm, as a pair:
  `("figure_examples", {"GEOTECH_REVIEW_FIGURE_EXAMPLES": "1"})`, against
  the baseline (every switch unset), same model, same tasks.
- **The suite has no task that asks for a figure today**
  (`aec_document_review/review_eval/tasks.py`): the arm would measure
  nothing. A measurement needs a few figure tasks on the existing fixtures,
  for instance: the floor levels each sheet gives (coordination set), the
  door and footing schedules against the marks counted on the plans, the
  comment log of a full review by sheet, the scaled-plan distance drawn as
  a plan, the submittal against its specification's limits.
- What to compare per task: whether the helper called `figure_examples` or
  started from the attached example (activity record); the number of
  draws and looks (the helper's own budget); whether the figure obeys the
  checklist on its PNG (plan view and scale for a plan; markers, not lines,
  across sheets; axis from zero for counts; the limit named; no legend for
  one series; nothing drawn that the documents do not give); and the
  figure helper's tokens (the listing and the example fetched are smaller
  with the switch on).
