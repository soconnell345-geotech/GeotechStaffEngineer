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
