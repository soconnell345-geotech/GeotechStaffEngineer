# PL-27: planlens' plan tools, behind a switch; the core's side

## What the item asked

Wire planlens' plan-location and markup-id tools into the app: offer
`plan_symbols` and `plan_grid` (feature-detected), let the markup `remove`
take a `uid`, and make `forget_path` call `ReviewToolkit.forget`. Since
planlens 0.15.0 the last two live in planlens itself; they are done there
(planlens branch `claude/pl-27-wire`, not in this repository). This file is
the core's side: the two plan tools on the Document Review page, behind a
switch, off by default (the lead's ruling: a new behaviour the model is
offered).

## What was built, and where

- `aec_agent_core/review_flags.py`: `PLAN_TOOLS_ENV =
  "GEOTECH_REVIEW_PLAN_TOOLS"`, in `LATER_ENVS` (not `ALL_ENVS` or `ARMS`,
  which the geotech app's tests pin whole), `plan_tools()`, and its entry in
  the module's list of switches.
- `aec_agent_core/document_tools.py`: `PLAN_TOOL_NAMES`, kept OUT of
  `OPTIONAL_DOCUMENT_TOOL_NAMES` (every planlens the core takes publishes
  both tools in its standing list, so putting them there would turn them on
  for every page); `plan_tools_available()`; `plan_tools_need_asking()`;
  `PLAN_SHOWN_FROM_PLANLENS` and `plan_tool_schemas(name)`; `host_for(...,
  plan_tools=False)` adds `toolkit_options={"plan_tools": True}` only where
  the installed planlens needs asking.
- `aec_agent_core/deep/vision_kit.py`: `make_vision_tools(...,
  plan_tools=False)`; the two tool functions `plan_symbols` and `plan_grid`,
  each handing on only the arguments given; their notes (the viewer's
  `pdf_page`) in `_DOCUMENT_TOOL_NOTES`; their shown schemas in
  `_document_tool_schemas`. They are built after the sheet tools and only
  with the default tool set (a helper's own `include` never gets them).
- `aec_document_review/review_build.py`: the page passes
  `plan_tools=review_flags.plan_tools()`, as it passes the sheet tools'
  switch; its reading helper gets the two tools where the page has them
  (they read and write nothing, like the sheet tools).
- `aec_document_review/review_agent.py`: one phrase swap in
  `REVIEW_REWRITES` for `plan_symbols`, so the page's description names no
  geotechnical test (BACKLOG RI-34): the example labels "(B-1, TP-1, CPT-3
  ...)" become "(B-1, TP-1 ...)".
- `packages/aec-document-review/README.md`: one row for the switch.

## Which planlens the core's side needs, and what an older one does

- **planlens 0.16.0 (the pinned release):** works. It publishes both tools,
  and its dispatcher serves them to every host, so the page offers them
  with the switch on and does NOT pass the `plan_tools` option (0.16.0's
  toolkit would refuse an option it does not know: the core checks
  `plan_tools_need_asking()`, i.e. `planlens.review.specs.PLAN_TOOLS`).
  Document Review pins planlens 0.16.0 exactly. The core takes 0.15 or
  later, whose dispatcher serves the two tools to every host as 0.16.0's
  does (the tools are in planlens since 0.14).
- **planlens with PL-27 (after 0.16.0):** its dispatcher serves the plan
  tools only to a host that asks, and the page asks. Its host layer then
  specifies them exactly as the core shows them (a test compares the two).
- **A planlens without the tools** (none the core takes; kept as a guard):
  the page is the default build.
- The core change needs no new planlens. The planlens release is needed
  only for its own repairs (the `uid` in `remove`, `forget_path`) and for
  other hosts of `planlens.review`.

## Judgement calls

- **The page is Document Review's only.** The lead's ruling gates the plan
  tools as the sheet tools are, and the sheet tools reach only the review
  page (its `review_build`). The geotech page builds its tools in the app's
  own code, which is quiet here, so `make_vision_tools` takes
  `plan_tools=False` by default and the geotech page is unchanged. The tools
  are about exploration locations on a site plan, which is the geotech
  page's work as much as the review page's: the maintainer can pass
  `plan_tools=review_flags.plan_tools()` from the geotech page's build.
- **Each argument is shown with planlens' own schema of it** (type, limits
  and words: the size defaults, the label pattern's default, the shape of a
  control point), read off planlens' spec, so it cannot drift. `handle`,
  `page` and `pdf_page` are shown as every document tool shows them; every
  argument is optional but `plan_symbols`' handle (either `page` or
  `pdf_page`).
- **An argument not given is not handed on**, so planlens' own defaults
  apply and its result's `defaults` say so (for example `on_mirror`).
- **The swap drops one example label** rather than rewording the tool,
  which by default reads exploration ids; a reviewer of other plans passes
  `label_pattern`.

## Settings added

| Setting | Default | What `1` does |
|---|---|---|
| `GEOTECH_REVIEW_PLAN_TOOLS` | off | `plan_symbols` and `plan_grid` on the Document Review page and its reading helper, where the installed planlens has them |

Read when the page is built. Unset, the tool lists, descriptions and schemas
of both GSE pages and the Document Review page are byte for byte as before:
`tests/test_review_fingerprints.py` (Document Review) and
`webapp/tests/test_page_fingerprints.py` (GSE's two pages) pass unchanged
with planlens 0.16.0 and with the PL-27 planlens.

## Tests

`packages/aec-document-review/tests/test_pl27_plan_tools_offline.py` (new;
each fails on master, where the switch and names do not exist):

- the switch: its name, in `LATER_ENVS` and in neither `ALL_ENVS` nor any
  arm, off unless `1`, cleared by `switches()`, named by `describe()` when
  set;
- the names are kept out of the standing list;
- off: neither the page nor the kit has them, and the kit built with
  `plan_tools=False` shows every tool exactly as the default build does; no
  `plan_tools` option on the host;
- on, with a planlens without them: the default build;
- on: the page and its reading helper have them, described in planlens'
  words plus the viewer-page note and the page's swap (the swap still
  matches);
- on: the page's descriptions of them name no geotechnical report thing
  (the RI-34 vocabulary);
- on: every shown argument is planlens' own schema of it, with `default`
  None and its title;
- on, with the PL-27 planlens: its host layer's specs of the two tools
  equal the core's, description and schema (skipped on 0.16.0);
- on: through the kit, a file name as the handle and `pdf_page=1` find
  every symbol of planlens' made-up plan sheet; `plan_grid` converts the
  labelled centres right-handed, within 0.5 m of the made-up truth; a
  mirrored control set is refused (planlens' default, as nothing was
  handed on);
- the host passes `plan_tools` exactly when the installed planlens needs
  asking.

Run (each exit 0), with `PLANLENS_TREE` set to the PL-27 planlens and again
with the installed 0.16.0: `test_pl27_plan_tools_offline.py`,
`test_review_sheet_tools_offline.py`, `test_review_fingerprints.py`,
`test_review_switches_offline.py`, `test_review_page_vocabulary_offline.py`;
and from the root `test_core_imports.py`, `test_dr48_photo_pages.py`,
`test_vis33_photo_pages_one_look.py`, `test_vis34_tile_answers.py`,
`test_ta43_image_ceiling.py`, `test_core_suite_and_records.py`,
`funhouse_agent/deep/tests/test_document_gate.py`,
`webapp/tests/test_page_fingerprints.py`. No existing test was edited.

## Noted, not fixed

- `plan_symbols`' `label_pattern` description (planlens' words) still
  gives exploration ids as its default, CPT among them; the RI-34 guard
  reads descriptions, not argument schemas. Behind the switch only.

## For the lead: a measurement on the hosted platform

With `GEOTECH_REVIEW_PLAN_TOOLS=1` against unset, on a task whose drawing
set has a vector site plan with labelled point symbols and grid or
coordinate labels: whether the agent reports positions and local
coordinates from `plan_symbols` / `plan_grid` rather than from a look,
how far its reported coordinates sit from the drawing's, and the calls and
tokens each arm spends. The arm is the pair `("plan_tools",
{"GEOTECH_REVIEW_PLAN_TOOLS": "1"})`.
