# Brief 14: three unreleased planlens parts on the Document Review page, each off unless asked

Branch `claude/b14-core-wiring`, cut from `origin/claude/b13-integration`
(69d6c2c). One code commit, 7a50cf8, then this file alone. Nothing was
released, no fingerprint was re-recorded, and `tinyapps/`, `funhouse_agent/`,
`webapp/`, `BACKLOG.md` and planlens were not edited.

The planlens with the three parts is the branch `claude/b14-pl-preview` at
99af95f, cloned to `/tmp/planlens-preview` and installed with `pip install
--no-deps --force-reinstall -e`. It still reports version 0.16.0 ("the
preview" below; "0.16.0" below is the wheel from PyPI).

The model followed is the plan tools' wiring (5523575, `module_work/pl27/
CHECK.md`, `tests/test_pl27_plan_tools_offline.py`).

## What was confirmed on the base before changing it

- `aec_agent_core/deep/vision_kit.py` `make_vision_tools`: the page's
  document tools are functions that build their own arguments and call
  planlens' dispatcher (`document_tools.dispatch_document_tool`) under the
  build's host (`document_tools.host_for`). None takes a sheet number; a
  `sheets` argument is refused by name (`strict_tool`).
- The `minimal` build has no wrappers of its own: it calls the same
  `make_vision_tools` (`aec_document_review/minimal_agent.py`) and passes
  neither the sheet tools' nor the plan tools' switch, so it had none of
  those tools on the base and has none of this now. Left as it is.
- `document_tools._spec` read `TOOL_SPECS` and `SHEET_TOOL_SPECS` only, so
  it could not see `drawing_geometry` (kept in `DRAWING_TOOL_SPECS`).
- A DXF today (made-up drawing, base code, both planlens installs):
  `open_document`, `read_document` and `search_document` read it as text
  (entities, layers, lettering with positions). Every page tool refuses it:
  "'site.dxf' is a DXF drawing, read as CAD data: it has no pages here, so
  document_page_map does not apply to it", with the hint to ask the user for
  a PDF plot. `analyze_pdf_page` and `render_region` on it raise MuPDF's
  `FileDataError('Failed to open stream')` out of the tool function.

## What was built

### 1. Sheet numbers as addresses (PL-37), under the existing `GEOTECH_REVIEW_SHEET_TOOLS`

- `document_tools.sheet_address_args(name)`: `[(argument, schema)]` from the
  installed planlens' own `planlens.review.specs.sheet_args(name, True)`;
  empty where planlens has no such function (0.16.0) or the tool takes no
  page. No version is read.
- `vision_kit._taking_sheets(tool, sheet_args)`: the tool with those
  arguments after its others, shown in planlens' schema and words (the
  title is the only thing added, as pydantic writes one for every
  argument). The tool's function is not changed: what was given is held for
  the call (`_SHEET_ADDRESS`) and added to the arguments where the call is
  handed to planlens' dispatcher. planlens finds the pages, adds
  `sheet_pages` to the result, or refuses the number.
- Applied in `make_vision_tools` to every document tool, only where the
  build has the sheet tools. With the preview that is `sheets` on
  `render_page_thumbnails`, `document_page_map`, `read_document`,
  `search_document`, `find_quantities`, `document_markups`,
  `check_cross_references`, `check_schedule_marks`, and `sheet` on
  `plan_symbols`, `plan_grid` and `drawing_geometry` where the page has
  those. The reading helper holds the page's own tool objects, so it takes
  them too.

Made-up example (planlens' five-sheet fixture, sheets G-001, A-101, A-102,
A-501, A-601):

- before: `read_document(handle="set.pdf", sheets="A-101")` is refused as an
  unknown argument (also now, with the switch unset or on 0.16.0);
- after, switch on, preview: the text of page 1, with `"sheet_pages":
  [{"sheet": "A-101", "page": 1, "pdf_page": 2}]`;
- `sheets="ZZ-999"`: "sheet ZZ-999 is not in this document's sheet index,
  which reads 5 sheet numbers: G-001, A-101, A-102, A-501, A-601";
- `sheets="A101"`: refused, the hint naming "A101 -> A-101" without taking
  it; `pages="0", sheets="A-101"`: refused, "name different pages".

### 2. `drawing_geometry` (DR-44, PL-13), behind the new `GEOTECH_REVIEW_DRAWING_TOOLS`

- `review_flags`: `DRAWING_TOOLS_ENV`, `drawing_tools()`, in `LATER_ENVS`
  (not `ALL_ENVS` or `ARMS`), the docstring; so `switches` clears it and
  `describe` names it.
- `document_tools`: `DRAWING_TOOL_NAMES`; `drawing_tools_available()` (the
  toolkit publishes the tool AND `planlens.review.specs.DRAWING_TOOLS` names
  it AND a host can pass options); `drawing_tool_schemas()` (the arguments
  planlens' `DRAWING_SHOWN_FROM_TOOLKIT` lists, as its spec shows them);
  `_spec` also reads `DRAWING_TOOL_SPECS`; `host_for(drawing_tools=True)`
  adds `toolkit_options={"drawing_tools": True}` only where it is available.
- `vision_kit`: `make_vision_tools(drawing_tools=False)`; the tool function
  hands on only what was given; its description is planlens' words plus the
  viewer-page note; built after the plan tools.
- `aec_document_review/review_build.py`: the page passes the switch, and the
  reading helper gets the tool where the page has it (it reads and writes
  nothing).

Made-up example (planlens' dimensioned fixture): `drawing_geometry(handle=
"dims.pdf", pdf_page=1)` returns `measured_from`, `frame`, what the sheet
is, and its dimensions with value and end points; `find=["title_block"]`
returns only that.

### 3. A DXF also as a plot (DR-19), behind the new `GEOTECH_REVIEW_DXF_PLOTS`

- `review_flags`: `DXF_PLOTS_ENV`, `dxf_plots()`, as above.
- `document_tools.dxf_plots_available()`: the toolkit's constructor takes
  `dxf_plots`, and `planlens.dxf.plot` and `ezdxf` can be imported.
  `host_for(dxf_plots=True)` adds `toolkit_options={"dxf_plots": True}` only
  then.
- `vision_kit`: `make_vision_tools(dxf_plots=False)`. With it on:
  - planlens serves the page tools on the DXF's plot;
  - `analyze_pdf_page` and `render_region` on a DXF look at the same plot
    (`planlens.dxf.plot.dxf_to_pdf`, kept per build by the file's digest; the
    attachments handed in are not written to) and add `plot`, planlens' own
    words for what a plot is; a DXF that will not plot comes back as an
    error with a hint, not a raise;
  - `open_document`'s description swaps "the page, markup and vision tools
    do not apply to it." for words that say a DXF is also a plot
    (`_DXF_PLOT_REWRITE`, 433 characters more).

**What a tester would see with the switch on** (made-up DXF: a 60 x 30
outline, one line, the lettering "GRID B"):

| Call on `site.dxf` | Switch unset | Switch on (preview) |
|---|---|---|
| `open_document` | the DXF as text | the same, plus `plot`: what the plot is |
| `read_document`, `search_document` | the DXF's own text | unchanged |
| `document_page_map` | refused, "no pages here" | one row, page 0 |
| `render_page_thumbnails` | refused | a contact sheet of the plot |
| `analyze_pdf_page`, `render_region` | `FileDataError` | a look at the plot, with `plot` saying what it is |
| `annotate_document` | refused | `site_marked.pdf`: a marked PDF copy of the plot |
| `drawing_geometry`, `plan_symbols` (their switches on) | refused | read the plot's drawn lines |
| `measure`, `find_like`, `annotate_dxf` | as today | as today |

So a tester who uploads a DXF and asks "what does this show" or "mark the
line that..." gets a look and a marked PDF, where today the agent can only
read entity lists and asks for a PDF plot. Limits a tester would meet: the
plot is the model space only; its lettering is shapes, so text still comes
from the DXF read; its scale is the fit to the page; the marked copy is a
PDF of the plot, not a DXF.

## What must hold, and how it was shown

**Every switch unset: names, descriptions, schemas and results as on the
base.**

- `test_unset_the_page_shows_what_it_did_before_this_work` compares a hash
  of the page's document tools and its two page looks with one recorded on
  69d6c2c: `30d854b3...4c63`, the same on the base and on the branch, with
  0.16.0 and with the preview (planlens surface `3e415249e4551735` on both).
- A one-off comparison, base worktree against the branch, both planlens
  installs: all 24 page tools (description and schema), the two helpers'
  tool lists, and the results of 14 fixed calls on a made-up sheet set and a
  made-up DXF: identical in all four comparisons.
- Both fingerprint tests pass unchanged on every install:
  `packages/aec-document-review/tests/test_review_fingerprints.py` and
  `webapp/tests/test_page_fingerprints.py`. None failed on the preview, so
  there is no hash to name.

**planlens 0.16.0 with every switch set: the default build, nothing fails.**
`test_set_on_planlens_0_16_the_page_is_the_build_it_was` (runs only on
0.16.0): the two new switches give the unset page; all three set give the
page `GEOTECH_REVIEW_SHEET_TOOLS` alone gave; calls work and a DXF has no
`plot`. `test_set_without_the_parts_in_planlens_nothing_is_added` shows the
same on the preview with the three detections patched off.

## Commands and results

Venvs: `/tmp/dr` (brief's venv a: deepagents 0.7.24, langchain 1.4.5,
planlens 0.16.0); `/tmp/drp` (the same with the preview installed
editable); `/tmp/dr68` (deepagents 0.6.8, langchain 1.3.4, planlens 0.16.0;
run again with `PYTHONPATH=/tmp/planlens-preview`); `/tmp/gse` (venv b: the
core with the geotech app, planlens 0.16.0).

From `packages/aec-document-review`, `MPLBACKEND=Agg <python> -m pytest -q
-p no:cacheprovider tests` (the whole folder), each exit 0:

| deepagents | planlens | Result |
|---|---|---|
| 0.7.24 | 0.16.0 | 1179 passed, 16 skipped |
| 0.7.24 | preview | 1192 passed, 3 skipped |
| 0.6.8 | 0.16.0 | 1179 passed, 16 skipped |
| 0.6.8 | preview | 1192 passed, 3 skipped |

`tests/test_b14_wiring_offline.py` alone: 9 passed, 13 skipped on 0.16.0;
21 passed, 1 skipped on the preview (both deepagents). On the base it does
not import (`document_tools` has no `sheet_address_args`), so every test
fails there.

From the root in `/tmp/gse`, each exit 0: `packages/aec-agent-core/tests`
655 passed; `webapp/tests/test_page_fingerprints.py`,
`funhouse_agent/deep/tests/test_document_gate.py`, `webapp/tests/
test_wheels.py` 34 passed, 5 skipped; `test_page_fingerprints.py`, the two
`review_eval` offline files and `webapp/tests/test_tinyapps_wiring.py` 134
passed. Guards (`test_public_text.py`, `test_review_freeze.py`): 13 passed.

No existing test was edited.

## What each switch costs in characters handed to the model

The page's tool descriptions plus argument schemas, unset: 24 tools, 43,014
characters (both planlens installs). Added by each switch alone:

| Switch | planlens 0.16.0 | preview |
|---|---|---|
| `GEOTECH_REVIEW_SHEET_TOOLS` | +5,721 (three tools) | +9,277: the three tools 6,823, and 409 for `sheets` on each of six standing tools |
| of which sheet numbers | 0 | 3,272 (409 x 8 tools); 307 more for `sheet` on each plan or drawing tool the page has |
| `GEOTECH_REVIEW_DRAWING_TOOLS` | 0 | +1,713 (one tool; 2,020 with the sheet tools) |
| `GEOTECH_REVIEW_DXF_PLOTS` | 0 | +433 (`open_document`'s description) |
| all three and the plan tools | +10,225 | +16,848 |

The reading helper is handed the same tool objects, so each added tool or
argument is counted once more in a helper's own request. In results, with
the preview: `open_document` of the five-sheet set 1,584 -> 4,758 with the
sheet tools (3,351 on 0.16.0; the rest is planlens' own); a call that gives
sheet numbers adds one `sheet_pages` row per sheet; `open_document` of a
DXF +360 (the `plot` line); each look at a DXF carries the 263-character
`plot` note.

## Fingerprints

Not moved, not re-recorded.

## Calls made

- **The looks take the plot.** The item asks for the option; DR-19 itself
  is "render a DXF to PDF so the vision tools can look at it", and planlens'
  `plot` line tells the model `render_region` takes the file. The page's
  looks are the core's own and do not go through planlens, so behind the
  same switch they plot the DXF with planlens' function. Without this the
  switch would have pointed the model at a tool that raises.
- **`open_document`'s description changes with the DXF switch**: it says
  the page tools do not apply to a DXF, untrue with the switch on.
- **Sheet numbers only where planlens offers them for the tool.**
  `analyze_pdf_page`, `render_region`, `read_pdf_text`, `measure` and
  `log_grid` are the core's own; planlens' `sheet_args` names none of them
  and they take a source, not a handle with a sheet index. They take pages
  as before.
- **`dxf_plots` is detected from the toolkit's constructor**, planlens
  publishing nothing else to read (gap 3 below).
- **The drawing tool sits after the plan tools** in the page's list, before
  `annotate_document`, as the plan tools sit after the sheet tools.
- **One hash in the test for "as on the base"**, compared only on a planlens
  whose tool specs are the ones it was recorded with; on another planlens
  the test skips that line and the fingerprint tests are the comparison.

## Gaps found in planlens (not edited), each with a small test

All three pass on the preview today, which is the gap.

```python
import json, ezdxf, pytest
from planlens import review
from planlens.review import conversation, specs
from planlens.review.host import ReviewHost

ASK = ReviewHost(toolkit_options={"dxf_plots": True})

@pytest.fixture
def att(tmp_path):
    doc = ezdxf.new(); msp = doc.modelspace()
    msp.add_lwpolyline([(0, 0), (60, 0), (60, 30), (0, 30)], close=True)
    msp.add_text("GRID B", height=2.0).set_placement((4, 22))
    path = tmp_path / "site.dxf"; doc.saveas(str(path))
    conversation._SPACES.clear()
    return {"site.dxf": path.read_bytes()}

def test_gap_1_the_plot_line_names_tools_the_host_layer_does_not_offer(att):
    out = json.loads(review.call("open_document", {"source": "site.dxf"},
                                 att, host=ASK))
    offered = set(review.tool_names(ASK))
    assert out["plot"].startswith("render_page, render_region and the other")
    assert "render_page" not in offered and "render_region" not in offered

def test_gap_2_the_page_map_calls_a_plot_with_drawing_on_it_blank(att):
    rows = json.loads(review.call("document_page_map", {"handle": "site.dxf"},
                                  att, host=ASK))["rows"]
    assert rows[0]["kind"] == "blank" and rows[0]["words"] == 0

def test_gap_3_nothing_public_says_the_host_layer_takes_dxf_plots():
    assert not [n for n in dir(specs) if "DXF" in n.upper()]
    assert not hasattr(review, "HOST_OPTIONS")
```

1. The host layer's `plot` line names `render_page` (a toolkit tool no host
   is offered) and `render_region` (the host's own tool, which planlens
   cannot know takes the plot). The words should name what the host layer
   offers, or let the host supply them.
2. The page map gives the plot's page `kind: "blank"` (it has no words),
   though it holds the drawing. A model reading the map would pass it over.
3. A host cannot tell from planlens' specs that `dxf_plots` is taken;
   0.16.0's toolkit raises on the option. The sheet, plan and drawing tools
   each have a tuple in `planlens.review.specs` to read.

## Noted, not fixed

- With the switch unset, `analyze_pdf_page` and `render_region` on a DXF
  raise `FileDataError` out of the tool function, as on the base.
- planlens' host layer shows `status` of the two sheet checks as an enum or
  null; the core shows a string defaulting to "". Older than this work.
- The plot's page takes drawing units as millimetres up to A0: the made-up
  60 x 30 drawing plots on a 226 x 141 pt page. Lines are vector, so a zoom
  is sharp, but the contact sheet of a small drawing is small.
- `measure`, `log_grid` and `find_like` do not take a DXF's plot (the first
  two call planlens under a host with no options; `find_like` reads the
  attachment's bytes).

## Release note

None until a planlens with these parts is released and Document Review's
pin moves: on planlens 0.16.0 all three do nothing. After that, off unless
set: with `GEOTECH_REVIEW_SHEET_TOOLS=1` the document tools take printed
sheet numbers; `GEOTECH_REVIEW_DRAWING_TOOLS=1` adds `drawing_geometry`;
`GEOTECH_REVIEW_DXF_PLOTS=1` lets the page look at and mark up an uploaded
DXF as a plot.

## For the lead

- Push as `ci/<name>` for the eight jobs; here only the files above ran.
- When planlens with these parts reaches PyPI: Document Review's exact pin,
  the fingerprints' entry for that version (recorded from a full checkout),
  and the hash in `test_b14_wiring_offline.py` if the standing tools' specs
  moved (the test skips that line until then).
- Suite arms, none run on a model: `("drawing_tools",
  {"GEOTECH_REVIEW_DRAWING_TOOLS": "1"})`, `("dxf_plots",
  {"GEOTECH_REVIEW_DXF_PLOTS": "1"})`; sheet numbers ride on the existing
  `sheet_tools` arm. DR-44 asks for a task in which the user refuses the
  first answer before its arm can show a gain; the DXF arm needs a task
  with an uploaded DXF.
- Live check for the DXF switch: a real model's look at a plot of a real
  drawing (lettering legible at the plot's size), and whether it reads text
  from the DXF rather than from the plot.
- The three planlens gaps above, for the planlens branch before release.
- The geotech page is unchanged: its build passes none of the new
  arguments (the maintainer's).
