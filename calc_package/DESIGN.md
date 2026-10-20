# Calculation Package Generator — Design Notes

## Purpose

Generates professional, Mathcad-style calculation packages as self-contained
HTML files. Output mimics what a staff geotechnical engineer would submit
for review — inputs echoed, equations shown with substituted values,
step-by-step computation, engineering checks (pass/fail), and figures.

## Architecture

### Core (calc_package/)
- `data_model.py` — InputItem, CalcStep, CheckItem, FigureData, TableData, CalcSection, CalcPackageData
- `renderer.py` — Jinja2 HTML rendering, figure_to_base64(), preprocessing
- `template.py` — HTML/CSS template as Python string (no external files)
- `docx_renderer.py` — markdown -> Word (.docx) for prose the agent composed
  (see "Word output" below); nothing to do with the packages themselves.
  Since INF-30 step 5 it is `funhouse_agent/docx_renderer.py`, both pages'
  module; `calc_package.docx_renderer` is the same module under its old name
- `tables.py` — calculation tables (GM-39): one CSV per table and the printed
  "Calculation Tables" section (see "Calculation tables" below)
- `latex_renderer.py`, `latex_template.py`, `latex_compiler.py` — the LaTeX / PDF
  output (`format="latex"` / `"pdf"`)
- `__init__.py` — `generate_calc_package()` entry point + lazy module registry,
  `package_tables()`

### Per-Module (module_name/calc_steps.py)
Each supported module provides:
```python
DISPLAY_NAME = "Human-Readable Analysis Title"
REFERENCES = ["Citation 1", "Citation 2"]

def get_input_summary(result, analysis) -> list[InputItem]
def get_calc_steps(result, analysis) -> list[CalcStep] or list[CalcSection]
def get_figures(result, analysis) -> list[FigureData]
def get_tables(result, analysis) -> list[CalcTable]   # optional (GM-39)
```

### Rendering Pipeline
```
analyze_*() → Results object
                  ↓
calc_steps.get_input_summary(result, analysis) → list[InputItem]
calc_steps.get_calc_steps(result, analysis) → list[CalcSection]
calc_steps.get_figures(result, analysis) → list[FigureData]
                  ↓
            CalcPackageData assembled
                  ↓
            renderer.render_html(data) → str (HTML)
                  ↓
            save_html() or return string
```

### Figures-only rendering (`render_figures`, added 2026-09-11)

The agent adapter (`funhouse_agent/adapters/calc_package.py`) can run the same
pipeline and stop after `get_figures()`: inside a `render_figures` call
`_build_response` decodes each `FigureData.image_base64` and saves it as a
standalone PNG (`save_verified`, one file per figure, in the working folder or
`output_dir`) instead of assembling a package, returning
`[{title, caption, output_path, html_img_tag}]` plus the package's key
results. Every `*_package` generator routes through `_build_response`, so all
15 modules get it for free.

The mode rides a context var (`_FIGURE_MODE`), NOT a key in the params dict:
package handlers validate their params strictly (`reject_unknown_params`) and
forward them as `**kwargs` into the module function, so a router-injected key
would be rejected as unknown (`slope_report_package`) or arrive as an
unexpected kwarg (`pavement_design_package`). Out of band, a package needs to
know nothing about figure mode — which is what makes "all 15 for free" true. Purpose: a bespoke `html_to_pdf` report (SOE, a
multi-analysis narrative, anything with no canned template) can carry the
module's figures — before this they were reachable only inside a whole canned
package (owner feedback 2026-09-11).

The saved files are named from the figure's title (GM-171, 2026-10-19):
`[<name_prefix>_]<title slug>.png`, the slug cut at a word to 32 characters
(less where the folder is deep, `fig<i>` when under 8 would be left), and
`_2`, `_3` ... when the package or the folder already has the name. Before
this a deep folder gave `f1_<time>.png`.

### Figures in the Story PDF (GM-168, 2026-10-19)

`fitz.Story` shrinks a picture to the height left on the page, so a figure
placed after most of a page of text printed 0.1-2 in wide at the page foot
(live wave w13). `render_html_pdf_story` (behind `save_html_pdf_story`,
`html_to_pdf` and every canned Story PDF) lays the document out, finds each
figure printed narrower than it lays out on a page of its own (a pass with
one frame tall enough for the whole document), and moves it to the next page
with a page break before its block, the heading or "Figure ..." title just
above it, or the table holding it; the text after it follows. A figure that
does not fit even a fresh page starts one and is scaled to fit it, with room
for a caption. A picture cannot be split across pages, so nothing stops a
very tall figure printing small: below `MIN_FIGURE_SCALE` (0.5) of its width
`html_to_pdf` warns "draw it shorter". A figure that starts a page with its
heading or title left at the foot of the page before takes the heading
with it. Where most figures carry their "Figure ..." line under them (the
house style), the line under a figure stays with that figure and is never
taken as the next one's title.

The squeeze cost more than small figures: after a picture it shrank, the
engine draws the next picture past the page's edge and drops the text that
follows (a four-figure report lost 400 of its 1,040 words on master; GM-168
review). A figure drawn past the page foot is moved like a shrunk one.

Engine traps (PyMuPDF 1.27): a page break before the document's first
element, or anywhere after a table whose borders are not collapsed, makes
`Story` start the document again for ever. No break is put before the first
element; when a break would loop after a table, `table{border-collapse:
collapse}` is put first in the document (the HTML's own table rules still
win) and the layout tried again, and a figure whose move still loops is left
where the engine put it. `_MAX_STORY_PAGES` (and a tight limit on each
trial) turns any loop into an error rather than a hang. Edits to the
engine's DOM attributes are unreliable, so the breaks and widths are written
into the HTML string.

Each figure's PNG carries its smallest lettering and the width it was saved
at (a `Lettering` text chunk, `geotech_common.layout_ink.png_with_lettering`,
written by `figure_to_base64`, `profile_figure` and the plotting box), so
`html_to_pdf`'s `page_contents` gives every figure its printed `width_in`,
`height_in` and `lettering_pt`, and `figure_warnings` names a figure the
plotting box drew (its PNG records the print width its lettering check
measured it at, so the caller holds its code) whose lettering prints under
7 pt or that prints narrower than that width (the box's own constants), and
any figure taller than a page scaled below half its width. A module's or a
tool's own figure is reported, not warned: the caller cannot redraw it, and
every canned figure prints at 5.1-6.7 pt today (GM-186). A picture with no
record (a photo, a drawing, a logo) is not judged by its width.

## Calculation tables (`tables.py`, GM-39, 2026-10-14)

Engineers using LPILE, APILE, GROUP or SLIDE get, beside the answer, the
tables behind it; the owner asked for the same here. A module returns its
tables as `geotech_common.tables.CalcTable` objects, from its result object's
`get_tables()` or from its calc_steps' `get_tables(result, analysis)` when a
table needs the inputs too (a p-y curve, capacity against length). The
values are the module's own computed values, SI with the unit in each column
heading, columns named as the references and the Ensoft manuals name them.

`generate_calc_package(tables=True)` (the default) prints them in a
"Calculation Tables" section (a long table thinned to 50 rows, with a note
naming the CSV that holds every row) and, when `output_path` is given, saves
one CSV per table beside the package (`<stem>_t1_<name>.csv` ..., and the
input echo as `<stem>_inputs.csv`; `export_tables`). `tables=False` gives no
tables; a list gives those tables instead. `package_tables(module, result,
analysis)` returns them without building a package. A table that fails to
build never costs the package (a warning, and `render_report["tables_error"]`).

Modules with tables: lateral pile, axial pile, pile group, drilled shaft,
settlement, bearing capacity, sheet pile (cantilever and anchored), slope
stability (single surface and search), downdrag, fem2d (FEM, staged,
excavation, consolidation), liquefaction (seismic_geotech), the wave-equation
bearing graph, ground-improvement time curves. Retaining walls and pavement
hold only scalars; the SOE package is BACKLOG GM-41.

## Word output (`docx_renderer.py`, 2026-09-21)

The packages above are the calculation; this is everything else a reviewer
reads. The agent composes a memo, a findings summary or a review response as
markdown, and reviewers ask for it as `.docx` — a file they can track changes
in, paste into a report template and send on. `markdown_to_docx(markdown,
output_path, base_dir=..., title=...)` is that one step, and the agent tool
`write_docx` is it on the tool surface (`funhouse_agent/vision_tools.py`,
registered beside `save_file` in `funhouse_agent/deep/tools.py`).

It renders headings 1-4 to Word's own heading styles, runs of **bold**,
*italic* and `inline code`, bullet and numbered lists with one level of
nesting (`List Bullet` / `List Bullet 2`, `List Number` / `List Number 2`),
pipe tables as a `Table Grid` with a bold header row, block quotes as an
indented italic `Quote`, fenced code as one monospaced paragraph, and
`![alt](figure.png)` as an embedded picture scaled down to the text width with
its alt text as the caption. A `title` becomes a Title paragraph.

Five decisions worth stating:

1. **A horizontal rule `---` is a PAGE BREAK.** Word has no horizontal-rule
   paragraph, and in a document meant for print a rule between sections is
   almost always where the author wanted the next page to start. A reader who
   wants a thin line can draw one; nobody can recover a page break that was
   never emitted.
2. **An italic line immediately after a table is that table's CAPTION.**
   Markdown has no caption of its own, and that is the only way an author can
   say "this line names the table above", so it is taken as meant and rendered
   in the `Caption` style rather than as another paragraph.
3. **Nothing raises over content.** An unknown construct becomes a plain
   paragraph and a missing image becomes a line of text saying which file is
   missing; both are appended to the caller's `warnings` list. A figure that
   was not there must not cost the reader the other nine pages — but the
   agent is told, so it can say so rather than claim a figure is in the file.
4. **python-docx is imported LAZILY**, inside the function, so the module
   imports without it. That is what lets the tool layer feature-detect the
   package (`vision_tools.docx_available()`) and HIDE `write_docx` rather than
   offer it and fail in front of the user — the same rule the planlens
   document tools follow.
5. **The saving is `save_file`'s, byte for byte.** The document is rendered to
   a scratch file and handed to the same writer, so it gets the same path
   resolution into the conversation's working folder, the same Databricks
   `/Workspace` handling, the same read-back verification and the same rescue
   copy. `base_dir` defaults to that working folder, which is where
   `plot_data` and `render_figures` have already put the agent's own figures —
   so `![](profile.png)` finds the PNG written minutes earlier.

Markdown is parsed with `markdown-it-py` (CommonMark plus the table rule),
which arrived transitively through rich/streamlit long before this but is now
a declared dependency, because a tool of ours that imports a package names it.

## Design Decisions

1. **HTML output** (not PDF) — viewable in any browser, printable to PDF
   via browser print dialog. No extra dependencies (reportlab/weasyprint).
   Can be upgraded to direct PDF later.

2. **Equations as HTML text** — uses monospace font with sub/superscripts
   via HTML entities. No MathJax/LaTeX dependency. Simple, fast, readable.

3. **Figures as base64** — matplotlib figures saved to BytesIO, base64-encoded,
   embedded directly in HTML as data URIs. Fully self-contained single file.

4. **InputItems auto-grouped** — renderer preprocesses consecutive InputItems
   into a TableData with input-table CSS class. Clean table presentation.

5. **Lazy module registry** — calc_package doesn't import modules until
   `generate_calc_package()` is called. Avoids circular imports.

6. **Analysis object passed as-is** — the Analysis object holds all inputs.
   No need to capture or duplicate input data separately.

## Supported Modules (Tier 1)

(Historical first three. The live list is `calc_package._IMPORT_MAP` — 15
modules; `list_supported_modules()` derives from it since 2026-09-11 so the
two cannot drift.)

| Module | calc_steps.py | Plot Methods Added |
|--------|:---:|:---:|
| bearing_capacity | Yes | plot_term_breakdown() |
| lateral_pile | Yes | (existing plots captured) |
| slope_stability | Yes | plot_slip_circle() |

## CSS Design

- Serif body font (Georgia) — professional engineering document feel
- Monospace equations (Courier New) — Mathcad-like computation display
- Blue accent for calc steps and computed values
- Green/red for pass/fail checks
- Print-friendly: page-break-inside:avoid on steps and figures
- 8.5" max-width body mimics US Letter paper
