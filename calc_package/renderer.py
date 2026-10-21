"""
HTML renderer for calculation packages.

Converts CalcPackageData into self-contained HTML using Jinja2.
All figures embedded as base64 data URIs — no external files needed.
"""

import io
import base64
import re
from datetime import date
from pathlib import Path

from jinja2 import Environment, BaseLoader

from calc_package.data_model import (
    CalcPackageData, CalcSection, CalcStep, InputItem,
    CheckItem, FigureData, TableData,
)
from calc_package.template import (
    CALC_PACKAGE_TEMPLATE, CALC_SECTIONS_CSS, CALC_TABLES_HEADING_CSS,
    CALC_TABLES_TEMPLATE,
)


#: The width a package figure prints at in the package's PDF: the
#: template's text width, a letter page less the PDF's 0.75 in margins and
#: the page's 1 in padding each side. Every package figure is drawn for it,
#: so its lettering prints at 7 pt or larger (BACKLOG GM-186).
FIGURE_PRINT_WIDTH_IN = 5.0


def figure_to_base64(fig, dpi: int = 150, *,
                     print_width_in=FIGURE_PRINT_WIDTH_IN) -> str:
    """Convert a matplotlib Figure to a base64-encoded PNG string.

    The figure is first drawn for the width it prints at
    (``geotech_common.layout_ink.draw_for_print``): lettering under 7 pt is
    raised to 7 pt, and a figure that would print its smallest lettering
    under 7 pt at ``print_width_in`` is made smaller, its shape kept, and
    laid out again. ``fig`` is changed in place.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to convert.
    dpi : int
        Resolution in dots per inch. Default 150.
    print_width_in : float or None
        The width the figure prints at (default
        :data:`FIGURE_PRINT_WIDTH_IN`); ``None`` saves it as drawn.

    Returns
    -------
    str
        Base64-encoded PNG image data (no data URI prefix).
    """
    if print_width_in:
        from geotech_common.layout_ink import draw_for_print
        try:
            draw_for_print(fig, float(print_width_in))
        except Exception:                                  # noqa: BLE001
            pass                    # saved as drawn; its record says so
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    png = buf.read()
    buf.close()
    # the PNG carries its smallest lettering, so a PDF built from it can
    # tell how large that prints (GM-168)
    from geotech_common.layout_ink import png_with_lettering
    png = png_with_lettering(fig, png, dpi)
    return base64.b64encode(png).decode('utf-8')


def _preprocess_sections(sections: list) -> list:
    """Convert InputItem groups into TableData for cleaner rendering.

    Consecutive InputItems are gathered into a single input table.
    Other item types pass through unchanged.
    """
    processed = []
    for section in sections:
        new_items = []
        input_batch = []

        def flush_inputs():
            if input_batch:
                table = TableData(
                    title="",
                    headers=["Parameter", "Symbol", "Value", "Unit"],
                    rows=[
                        [inp.description, inp.name, inp.value, inp.unit]
                        for inp in input_batch
                    ],
                )
                # Mark it as an input table for CSS styling
                table._is_input_table = True
                new_items.append(table)
                input_batch.clear()

        for item in section.items:
            if isinstance(item, InputItem):
                input_batch.append(item)
            else:
                flush_inputs()
                new_items.append(item)
        flush_inputs()

        processed.append(CalcSection(title=section.title, items=new_items))
    return processed


def _item_type(item) -> str:
    """Return a string tag for template dispatch."""
    if isinstance(item, CalcStep):
        return "calc_step"
    if isinstance(item, CheckItem):
        return "check"
    if isinstance(item, FigureData):
        return "figure"
    if isinstance(item, TableData):
        if getattr(item, '_is_input_table', False):
            return "input_table"
        return "table"
    if isinstance(item, str):
        return "text"
    return "unknown"


def render_html(data: CalcPackageData) -> str:
    """Render a CalcPackageData object to a self-contained HTML string.

    All text fields (project_name, engineer, company, titles,
    descriptions, notes, references, table cells, ...) are
    HTML-escaped (CP-1) so hostile strings render inert. The only
    exceptions are the CalcStep ``equation`` and ``substitution``
    fields, which the data model documents as trusted equation markup
    ("HTML entities OK") — they are rendered raw via ``| safe`` in the
    template so entities like ``&radic;`` still display correctly.

    Parameters
    ----------
    data : CalcPackageData
        Complete calculation package data.

    Returns
    -------
    str
        Self-contained HTML document.
    """
    if not data.date:
        data.date = date.today().strftime("%Y-%m-%d")
    return _render(CALC_PACKAGE_TEMPLATE, data, heading="h2", numbered=True)


def _render(template_text: str, data: CalcPackageData, **extra) -> str:
    """``template_text`` rendered over ``data``'s sections (CP-1:
    autoescaped)."""
    # Preprocess: gather InputItems into tables
    processed_sections = _preprocess_sections(data.sections)

    # autoescape=True (CP-1): every interpolated field is HTML-escaped
    # unless explicitly marked | safe in the template (equations only).
    env = Environment(loader=BaseLoader(), autoescape=True)
    env.globals['item_type'] = _item_type
    template = env.from_string(template_text)
    return template.render(data=data, sections=processed_sections, **extra)


#: Every rule of a style sheet: its selectors and its body. The sections'
#: CSS has no at-rule, comment inside a rule or nested braces.
_CSS_RULE = re.compile(r"([^{}]+)\{([^{}]*)\}")
_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)

#: The class ``render_tables``' file puts round the sections; its CSS
#: applies only inside it, so a report's own classes keep their styles.
TABLES_SCOPE = "calc-tables"


def _scoped(css: str, scope: str) -> str:
    """``css`` with every selector inside ``.<scope>``."""
    def rule(m):
        sel = ",".join(f".{scope} {s.strip()}"
                       for s in m.group(1).split(",") if s.strip())
        return f"{sel} {{{m.group(2)}}}\n"
    return "".join(rule(m) for m in _CSS_RULE.finditer(
        _CSS_COMMENT.sub("", css)))


def tables_css() -> str:
    """The CSS of ``render_tables``' file: the package's sections' rules
    and its own section headings, scoped to ``.calc-tables``, with no
    background (the PDF draws none, BACKLOG GM-219, and says so of every
    fill; light text made dark)."""
    from calc_package.latex_renderer import _without_backgrounds
    sheet = _without_backgrounds(
        "<style>" + CALC_SECTIONS_CSS + CALC_TABLES_HEADING_CSS + "</style>")
    return _scoped(sheet[len("<style>"):-len("</style>")], TABLES_SCOPE)


def render_tables_html(data: CalcPackageData) -> str:
    """``data``'s sections and reference list as ``render_tables``' file
    (BACKLOG GM-67): the package's own rendering of them, with no title
    block or footer, the sections unnumbered under ``h3`` headings, for a
    report of the caller's own (``html_to_pdf``'s ``data-include``)."""
    if not data.date:
        data.date = date.today().strftime("%Y-%m-%d")
    return _render(CALC_TABLES_TEMPLATE, data, heading="h3", numbered=False,
                   css=tables_css())


def save_html(data: CalcPackageData, filepath: str) -> str:
    """Render and save a calculation package as an HTML file.

    Parameters
    ----------
    data : CalcPackageData
        Complete calculation package data.
    filepath : str
        Output file path (should end in .html).

    Returns
    -------
    str
        The absolute path of the saved file.
    """
    html = render_html(data)
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding='utf-8')
    return str(path.resolve())
