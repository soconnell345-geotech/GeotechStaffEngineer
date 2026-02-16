"""
HTML renderer for calculation packages.

Converts CalcPackageData into self-contained HTML using Jinja2.
All figures embedded as base64 data URIs — no external files needed.
"""

import io
import base64
from datetime import date
from pathlib import Path

from jinja2 import Environment, BaseLoader

from calc_package.data_model import (
    CalcPackageData, CalcSection, CalcStep, InputItem,
    CheckItem, FigureData, TableData,
)
from calc_package.template import CALC_PACKAGE_TEMPLATE


def figure_to_base64(fig, dpi: int = 150) -> str:
    """Convert a matplotlib Figure to a base64-encoded PNG string.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to convert.
    dpi : int
        Resolution in dots per inch. Default 150.

    Returns
    -------
    str
        Base64-encoded PNG image data (no data URI prefix).
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return encoded


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

    # Preprocess: gather InputItems into tables
    processed_sections = _preprocess_sections(data.sections)

    env = Environment(loader=BaseLoader(), autoescape=False)
    env.globals['item_type'] = _item_type
    template = env.from_string(CALC_PACKAGE_TEMPLATE)
    return template.render(data=data, sections=processed_sections)


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
