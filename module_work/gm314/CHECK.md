# GM-314 check: the figure test set

Branch `feat/gm314-figure-set` from master 2e7f6e9. No paid model run was
made: everything was tested offline with fake models.

## What was built, and where

`funhouse_agent/figure_eval/`, beside `chart_eval`, reusing its imaging
helpers, its JSON reader and its `foundry_model`. It also reuses the review
suite's page runner (`aec_agent_core.review_eval.runner.run_task`, page
`geotech`) and GM-328's `build_sample_packages.build_one`. No new machinery
for runs, switches or records.

- `requests.json`: the 22 requests. Each holds the engineer's ask on a
  made-up project, the sample row whose inputs it passes, the kind of
  figure, whether the tool texts or `figure_examples.py` name that figure,
  the checklist items it is chosen to test, and its own must-have items.
- `request_set.py`: resolves each request against GM-328's table (and one
  method probe), writes the data folder (`requests.json` plus `references/`
  built by GM-328's script, $0), and loads it.
- `checklist.py`: the 13 general items, the vision prompt, `check_run` (one
  call per run with all its figures, at most the four newest, detail AUTO
  for GPT and none for Claude), and the reader of the marks.
- `compare.py`: better / same / worse.
- `report.py`: `RESULTS.md`, `results.json` and `side_by_side.html` (OFF
  left, ON right, the marks under each, a box for the owner's mark, and a
  button that saves the owner's marks as JSON).
- `runner.py`: `score_figure_set` (resumable, rewrites the report after
  every run and check), `run_agent`, and the offline dry run
  (`python -m funhouse_agent.figure_eval.runner --dry-run --out DIR`).
- `tests/test_figure_eval_offline.py`: 11 tests.
- `module_work/gm314/FOUNDRY_SECTION_FIGURE_SET.md`: the section for the
  paid run.

## The requests

| Id | Module | What the figure shows | What it tests |
|---|---|---|---|
| R01 | bearing_capacity | ultimate and allowable bearing resistance against footing width (not named) | quantity on the right axis, kPa, design point marked, numbers |
| R02 | settlement | stress history of a made-up clay: effective overburden and preconsolidation pressure against elevation, three oedometer tests and a design profile (ground El. 30.0 m; made-up inputs; review round) | tests as points, the design profile a named line, elevation, kPa |
| R03 | lateral_pile | deflection, moment and shear against elevation, pile drawn (ground El. 20.0 m given) | elevation vertical, structure drawn, kN and kN-m |
| R04 | lateral_pile | p-y curves at 1, 3 and 6 m | curves named on the curve, units |
| R05 | slope_stability | cross-section with layers, the given circle and its factor of safety | to scale, layer names as given, numbers |
| R06 | slope_stability | entry-exit search: critical and ten lowest surfaces (not named) | to scale, data-ink, numbers |
| R07 | settlement | Schmertmann strain influence factor and Es against depth (not named) | depth only when no elevation, and said; Es per layer as steps |
| R08 | settlement | immediate settlement with Schmertmann's time factor, 0.1 to 50 years | settlement increasing downward, log time |
| R09 | axial_pile | shaft, toe and total resistance against toe elevation (ground El. 30.0 m) | elevation vertical, kN, layer names, groundwater drawn |
| R10 | drilled_shaft | unit and cumulative side resistance with depth, cased zone (not named) | structure drawn, kPa and kN |
| R11 | downdrag | axial load and soil and pile settlement with depth, neutral plane (numpy) | quantities defined in words, kN, numbers |
| R12 | seismic_geotech | wall with static active and Mononobe-Okabe pressures (not named) | wall drawn, the references' own names |
| R13 | seismic_geotech | liquefaction factor of safety, CSR and CRR at three SPT depths (ground El. 5.0 m; probe inputs; not named) | tests as points, elevation, FS 1.0 line named |
| R14 | retaining_walls | cantilever wall to scale with active pressure and base pressure | wall drawn, to scale, kPa |
| R15 | ground_improvement | degree of consolidation against time for three drain spacings and none | curves named, design time marked |
| R16 | ground_improvement | time to 90 % consolidation for three cases (not named) | cases as markers |
| R17 | pile_group | plan of the 11 x 2 group with each pile's load (not named) | plan as a drawing: no axes, scale bar, north arrow straight up and said |
| R18 | sheet_pile | cantilever soldier pile with net pressure and moment (not named; numpy) | wall drawn, kPa, embedment and moment written |
| R19 | soe | anchored wall with apparent pressure envelope and anchor loads (numpy) | wall and anchors drawn, kN |
| R20 | reliability | probability distribution of the factor of safety, P(F<1) shaded (not named) | defined in words, data-ink, numbers |
| R21 | pavement_design | flexible pavement section to scale (not named) | to scale, thickness in mm, layer names |
| R22 | fem2d | excess pore pressure isochrones with depth (not named) | depth vertical and said, kPa, curves named |

The set covers 17 of the 18 modules in GM-328's table. It leaves out
wave_equation (a bearing graph), to keep the set near 20.

## How the scorer decides

Each item is marked yes, no or n/a by the vision model, with one line of
reason. For one item of one request, ON is compared against OFF:

- the same mark in both arms is "same" (n/a with n/a included);
- ON yes where OFF no is "better", and the reverse is "worse";
- n/a in one arm and yes or no in the other is "one arm n/a", counted
  apart, never as a gain or a loss;
- an item the vision model left unmarked in either arm is "unmarked",
  counted apart, never "same" (review round).

A run that finished and drew nothing is marked no on every item, so the
arm that did draw wins every item its figure passes, and the headline names
the request. A failed run, a vision reply unreadable twice, and a request
neither arm drew are kept out of better, same and worse and named in the
headline (review round). A request is better when
ON has more better items than worse, worse when fewer, and same otherwise.
The report gives the counts per item and overall, with the requests named,
and no blended score. It also says how many ON runs called `plot_code` at
all, because a "same" without plot code means ON was not used, not that it
gained nothing.

## Tested offline

All on this machine, with fake models and no network:

- The set has 22 requests. Every row exists in GM-328's table, every
  checklist item is tested by some request, every kind of figure is there,
  15 or more modules are covered, and 8 or more figures are ones nobody
  names (12 after the review round). Each request passes the sample row's
  own inputs (or the probe's, for R13, or its own made-up inputs, for R02).
- The reader of marks takes fenced JSON and odd spellings ("n/a", "YES").
  An item that is missing or marked with another word is unmarked (not
  n/a, review round); a reply that marks fewer than half the items is not
  scored.
- Equal marks in both arms give "same" on every item and every request,
  and the report prints "same 2", "better 0", "worse 0".
- Different marks give the right better, worse and one-arm-n/a counts per
  item, per request and overall.
- A run that drew nothing, or failed, fails every item, and the vision
  model is not called for it.
- `check_run` sends at most four figures at detail AUTO for a GPT model
  and no detail for Claude, with the reference numbers in the prompt.
- The runner writes all three report files. The side-by-side page has OFF
  before ON, one owner box per request, and the failed run shown. A second
  call keeps finished runs and their checks and asks the failed run again.
- The dry run goes through the real geotech page agent: a scripted model
  calls `plot_data` through `call_agent`, and each arm's PNG lands in the
  run's `files/`. GM-328's reference build runs on the real row, and the
  fake marks end in the report. This was also run by hand for R01, R02,
  R13 and R17, with all four references built (R13's liquefaction probe
  returned a minimum factor of safety of 0.431 at 2 m).

Only the paid run can show:

- whether the real agent draws each figure at all;
- whether the vision model's marks agree with an engineer's;
- whether ON beats OFF on any item;
- the real tokens and time per run.

## Noted

- The data folder must be built at delivery
  (`python -m funhouse_agent.figure_eval.request_set --out figure_set_data`).
  Neither `docs/examples/` nor `funhouse_agent/tests/` ships in the wheel,
  so the requests travel resolved in that folder. Only four of the 22
  references were built here. The fem2d build (R22) may take a while; a
  failed reference only turns that request's numbers item to n/a.
- The numpy note: master has no direct `np.trapezoid` caller left. downdrag
  takes it from `geotech_common.numpy_compat`. R11, R18 and R19 are marked
  anyway, as the orchestrator asked.
- `run.json` keeps each tool call's arguments cut at 300 characters (the
  page runner's own record). The full calls are in `agent/activity.jsonl`.

## Review round (2026-11-14)

The independent review said merge after fixes (3 B and 5 C findings).
What each became:

| # | Finding | What changed |
|---|---|---|
| 1 (B) | An unreadable vision reply read as "same" | A missing item is now "unmarked", never n/a. A reply that marks fewer than half the items is "not scored". The runner asks the vision model once more at once; still unreadable, the check is saved as not scored, kept out of better / same / worse and named by request in the headline ("vision reply unreadable, not scored"). An item unmarked in either arm is its own outcome, never "same". |
| 2 (B) | A failed run decided a request | `check_run` flags a failed run (an error, a 429, a timeout, a crash). The comparison keeps it out of better / same / worse, and the headline names it with its arm ("run failed, not compared"). A run that finished and drew nothing still counts no on every item, and the headline names those requests. A request neither arm drew is its own row, never "same". The section asks the engineer to rerun the cell once after a pause and to report the runs that still fail. |
| 3 (B) | The preconsolidation convention was untested | R02 (the second bearing request: three bars of one number each, the weaker figure) is now R02-stress-history: a made-up soft clay under an embankment (ground El. 30.0 m) on the settlement package row. Three consolidation layers carry the effective overburden and the design preconsolidation profile (sigma_p); three made-up oedometer tests are in the ask. Its items: tests as points named as test data, the design profile a named line, elevation vertical, kPa. A `replace` without a probe now marks the row "made_up" and drops the printed example's note. Its reference builds (209 mm of consolidation). |
| 4 (C) | No general item for names on the data | New item `lines_named` (13 items now); R02, R04 and R15 list it. |
| 5 (C) | The plan item asked for a property line R17 lacks | It now says "a property line dash-dot where the plan shows one". |
| 6 (C) | "Not named" was overcounted | R08 and R11 are named (worked examples in `figure_examples.py`), and so is the new R02. 12 of the 22 are not named. |
| 7 (C) | R07's ask said strip; its inputs are a 1.8 m by 7.3 m rectangle | The ask now says "a rectangular footing 1.8 m by 7.3 m". |
| 8 (C) | Nothing guarded the arms' tool lists | New test: the page agent built under each `ARM_ENV` in one process (OFF, ON, OFF, ON) calls `plot_box.make_plot_code_tool` only under ON. |
| closing note | Drafts could be marked instead of final figures | The checker took the four newest PNGs of a run. Those could include the figure helper's look copies (`.scratch/<name>_look.png`, newer than their figure) and a draft drawn under another name. Now look copies are never sent and never listed as figures. `run.json` keeps every file under `files` and only the figures the reply cites under `figures` (`figures_by: reply`; every PNG left when it cites none, `figures_by: all`), and only `figures` are marked and shown side by side. The PNGs the reply names by file name are sent; when it names none, the newest, recorded as `picked_by: newest` and noted in `RESULTS.md`. A draft under another name cannot be told from a final figure by the file alone, so the section asks the engineer to compare R01's `check.json` with its `files/` before running the rest. New test. |

Tests: `test_figure_eval_offline.py` 15 passed, 4 of them new (the
unreadable reply, failed and neither-drew runs, the arms' tool lists, the
final figures marked), plus one more for a run that leaves a draft and a
final: only the final is sent. The
equal-marks test still gives "same 2, better 0, worse 0", and the new
not-compared rows show 0 and "-". `webapp/tests/test_public_text.py` 5
passed. The data folder was rebuilt with all 22 requests, and R02's
reference built. No agent-facing tool text changed.
