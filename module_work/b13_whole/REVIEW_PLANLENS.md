<!-- Filed here by the lead on 2026-11-19 from planlens' branch `claude/rv-b13-whole` (planlens keeps no review file on its main); the branch was then deleted. -->

# Release review: the whole unreleased change (v0.16.0 .. claude/b13-integration)

**Verdict: MERGE AFTER FIXES** — reviewed head `803e591`. One must-fix and one
should-fix, both where DR-19 (a DXF opened as a plot) meets another item;
both are fixed on this branch (`claude/rv-b13-whole`), each with a test.
Nothing else stands in the way of a release.

## 1. Options: is everything what 0.16.0 gave when they are unset?

Method: both trees (`/tmp/base` = v0.16.0, the head) driven by one script with
made-up files (a 4-sheet drawing set with a door schedule, a 5-page report, a
scan, a PNG, a DXF, an empty file, a missing file). Every toolkit tool and
every host tool was called, 1,287 calls a tree: a plain `ReviewToolkit` with
and without a look, and a `ReviewHost` with nothing asked, with a look, and
with `ImageSupport`. Handles and mark uids were normalised, then compared.

- **Specs:** `ReviewToolkit().specs()` in all three styles, `tool_names`,
  `TOOL_SPECS`, and `planlens.review` `host_specs` in three styles with and
  without a look: **identical**.
- **Results:** 1,241 of 1,287 identical. The 46 that differ are the four
  repairs the CHANGELOG states and nothing else: `measure` `side`/`at`
  (PL-26, 30 calls), the `min_score` 0 notes (PL-26, 13), the image-view
  refusal said in pixels (VIS-26, 2), `counts["page_units"]` (VIS-24, 1).

| Option (new, or changed in what it does) | Default | Read in | Unset = 0.16.0 |
|---|---|---|---|
| `ReviewToolkit(sheet_tools=)`: now also `sheets`/`sheet` arguments, the page map following the index, the schedule check's new wording | `False` | `toolkit.specs`, `_sheets_on` | yes |
| `ReviewToolkit(drawing_tools=)` | `False` | `toolkit._offered` | yes |
| `ReviewToolkit(dxf_plots=)` (bool or callable) | `False` | `toolkit.open` | yes |
| `ReviewHost(toolkit_options={"sheet_tools"})` | absent | `conversation._asked_for_sheet_tools` | yes |
| `ReviewHost(toolkit_options={"plan_tools"})` | absent | `conversation._asked_for_plan_tools` | yes for `review.call`; `dispatch_document_tool` called directly now refuses the plan tools (stated, PL-27) |
| `ReviewHost(toolkit_options={"drawing_tools"})` | absent | `conversation._asked_for_drawing_tools` | yes |
| `ReviewHost(toolkit_options={"dxf_plots"})` | absent | `conversation._asked_for_dxf_plots` | yes |
| `Document(column_warning=)` | `False` | `Document._read_page` | yes |
| `Document(continued_tables=)` | `False` | `Document.page` | yes |
| `document_outline(captions_on_every_page=)` | `False` | `roles.outline_from` | yes |
| `ImageSupport.sent_scale` | `None` | `marked_copy.judge_image_views` | no, as stated: the refusal's words are in pixels (VIS-26 repair); the decision is the same |

No environment variable is added. Changes that sit behind no option, each
stated in the CHANGELOG as a repair or a change: VIS-24, VIS-26, PL-26, PL-27
(`remove` by uid, `forget_path`), PL-18 (MINSERT), PL-16 (OpenCV < 6).

## 2. The public surface

A script imported every module of both trees and compared each public name
(`__all__`, or the module's own names), each signature, class member and
dataclass field; a second pass compared every module-level name, private
ones included.

- **Removed or renamed: none.** No module, function, class, tool name or
  argument of 0.16.0 is gone.
- **New required argument: none.** Every changed signature gained optional
  arguments at the end: `ReviewToolkit(drawing_tools, dxf_plots)`,
  `Document(continued_tables, column_warning)`, `document_outline` /
  `outline_from(captions_on_every_page)`, `ImageSupport.sent_scale`,
  `ScheduleMarks.unlettered`, `specs.input_schema(sheet_tools)`,
  `host_tool_names` / `host_specs(plan_tools, drawing_tools)`.
- **Result keys:** none removed with options unset (section 1). New keys
  appear only with an option or a stated repair (`sheet_pages`, `plot`,
  `min_score_note`, `counts["page_units"]`).
- New modules: `document.drawing_geometry`, `.continued_tables`,
  `.reading_order`, `dxf.minsert`, `dxf.plot`, `tools.sheet_addresses`.
- No private module-level name of 0.16.0 is gone either.

## 3. Findings

| # | Severity | Where | What is wrong, and the input that shows it | Fixed in |
|---|---|---|---|---|
| 1 | must-fix | `tools/toolkit.py` `_tool_annotate_document` (DR-19 meets the marks) | A DXF opened as a plot is held in memory, so `entry.path` is `None` and the rule "a marked copy is never the document itself" does not see the drawing. `ReviewToolkit(dxf_plots=True)`, open `site.dxf`, `annotate_document(output_path=<site.dxf>, append=False)`: the DXF is **replaced by a PDF** and the result says "the document you opened is unchanged". With `append=True` (the default) the DXF is read as the copy to add to and `FileDataError` is raised out of `ReviewToolkit.call`. The host layer names its own copies and is not affected. | `5b086a7` |
| 2 | should-fix | `tools/toolkit.py` `forget` (DR-19 meets PL-27) | `forget(path)` matches the file a document was opened from; a plot has none. Opened under a name the resolver maps to the DXF, the plot is not closed: `forget_path(<site.dxf>)` returns 0 and the old drawing's plot stays under its handle after the file is replaced. A PDF opened the same way is closed. | `c0838bb` |
| 3 | should-fix (words) | `CHANGELOG.md`, `review/DESIGN.md` | Untrue or unsaid: "a tool that takes `page` also takes `sheet`" (the host layer's `find_like` takes neither); the toolkit's specs no longer require `page` with the sheet tools (unsaid); a host that asks for `dxf_plots` "gets the page tools on a DXF" (not `find_like`, which fails to open a DXF for every host, as in 0.16.0); DESIGN put `sheet` on the plan tools only (the drawing tool takes it too); over a dozen new public names were not mentioned. | `9ed7b82` |
| 4 | note | `review/tools.py` `find_like` | On a DXF it answers "find_like failed: FileDataError" for every host. Same in 0.16.0, so not a regression; a host that asks for plots might expect a refusal in plain words. Not fixed (no edge-case chasing). | - |
| 5 | note | `document.py` `_sheets_from_index` (PL-41) | By reading, not run: in a conversation whose toolkit two hosts share, once the host that asked for the sheet tools makes the index, the other host's page map follows it too. "Without the sheet tools no result changes" holds for a host on its own. | - |
| 6 | note | `ReviewToolkit.call` | `render_region(marks=[{...}])` (a mark as an object) raises `KeyError: 0` out of `call`. Same in 0.16.0. | - |

## 4. Where two items meet (all run on the head, made-up input)

- **Sheet tools + drawing tool + plan tools** (toolkit and host):
  `drawing_geometry`, `plan_symbols`, `plan_grid`, `render_page`, `measure`
  take `sheet`; the result names `sheet_pages` (with `pdf_page` in the host
  layer); `sheet` beside a `page` / `pdf_page` that names another page is
  refused in the caller's own words. Holds.
- **Sheet tools + DXF plots:** a plot has no sheet numbers; `sheet` on it is
  refused in plain words, `sheet_index` and the two checks answer "no
  drawing sheets"; `sheets` on the DXF's own text is refused. Holds.
- **Drawing tool + DXF plot:** reads the plot, says its lettering is drawn
  as lines. Holds.
- **Marks + DXF plot:** written to `site_marked.pdf`, appended to, removed
  by uid; a small mark on the whole-page view is refused by the page rule.
  Holds, but for findings 1 and 2.
- **Marks on an image + `sent_scale` + page-unit boxes:** a mark on a
  1200 x 900 view sent at scale 1 is placed at the box its view names; a
  second image of another size under the same name goes to `_2`; `remove`
  by uid works on the first and removes nothing with the second;
  `boxes_to_grid` leaves `[..] pt` as written while converting a `px=` box
  in the same answer. Holds.
- Files changed by more than one merge (`conversation.py`, `specs.py`,
  `tools.py`, `toolkit.py`, `marked_copy.py`, `frames.py`, `document.py`):
  each change is present and the calls above pass through all of them.

## 5. CHANGELOG and DESIGN

Each Unreleased entry was checked against the code or a call: the limits of
the plot (16 MB, 50,000 entities, 30 s, 297 / 1189 mm), the twelve page-unit
forms of VIS-24 and the two that still convert, `{"uid": ...}` refused with
"did you mean id", the options' defaults. Untrue or missing statements are
finding 3. Not checked: PL-41's "22 made-up documents", PL-43's four limits
and PL-09's rules beyond their own tests (item reviews).

## 6. The suite

As `.github/workflows/tests.yml` runs it (`pip install -e ".[mcp]"`, pytest,
one process, Python 3.12.3):

| Tree | Passed | Skipped | Failed |
|---|---|---|---|
| Reviewed head `803e591` | 2560 | 16 | 0 |
| Head, with Pillow added | 2662 | 5 | 0 |
| This branch (the two fixes) | 2565 | 16 | 0 |

The new tests (5, in two files) fail on the head: 2 of 3 in
`test_rv_b13_marks_never_replace_the_dxf.py`, both in
`test_rv_b13_forget_closes_a_dxf_plot.py`. They stand in for the plot, so
they run without Pillow.

Skipped for a missing optional package (CI installs neither):

- **Pillow, 11 skips covering five whole files and six tests:**
  `dxf/tests/test_plot.py` (4), `test_dr19_plot_bounds.py`,
  `test_dr19_small_drawing.py`, `review/tests/test_dr19_not_asked.py`,
  `test_dr19_dxf_plots.py` (2), `test_vis26_image_views.py`,
  `test_markup_check.py`. **These hide paths a host runs:** every plot a
  host with `dxf_plots` gets, the rule that a host which did not ask gets no
  plot, all of VIS-26's marks on an image, and the markup check. CI has
  never run them; a host that plots or marks images has Pillow. Adding
  Pillow to the workflow's install is a change for the lead (no new
  dependency was added here).
- **rapidocr**, 1 (`tests/test_ocr.py`): left out on purpose (README). The
  other four are for Windows or a second spelling of a path, not packages.

## 7. Could not check

- **Python 3.10**, CI's other leg: no 3.10 here. The 64 changed `.py` files
  parse under the 3.10 grammar. Push this branch as `ci/...` to run it.
- **The app as a consumer** (Document Review's tests and fingerprints
  against this tree): not run. With every option unset the specs are
  identical to 0.16.0, so its fingerprints should not move until it asks.
- **A real DXF or drawing set, a vision model, OpenCV 6:** nothing here was
  run on real files or in front of a model.
