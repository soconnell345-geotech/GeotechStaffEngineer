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
  (see "Word output" below); nothing to do with the packages themselves
- `__init__.py` — `generate_calc_package()` entry point + lazy module registry

### Per-Module (module_name/calc_steps.py)
Each supported module provides:
```python
DISPLAY_NAME = "Human-Readable Analysis Title"
REFERENCES = ["Citation 1", "Citation 2"]

def get_input_summary(result, analysis) -> list[InputItem]
def get_calc_steps(result, analysis) -> list[CalcStep] or list[CalcSection]
def get_figures(result, analysis) -> list[FigureData]
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
