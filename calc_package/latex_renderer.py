"""
LaTeX renderer for calculation packages.

Converts CalcPackageData into a .tex document string using Jinja2,
and optionally compiles it to PDF via pdflatex.

Reuses the same preprocessing pipeline as the HTML renderer
(InputItem collapsing, item-type dispatch).
"""

import base64
import shutil
import tempfile
from datetime import date
from pathlib import Path

from jinja2 import Environment, BaseLoader

from calc_package.data_model import (
    CalcPackageData, CalcSection, CalcStep, InputItem,
    CheckItem, FigureData, TableData,
)
from calc_package.renderer import _preprocess_sections, _item_type
from calc_package.equation_converter import unicode_to_latex, escape_latex
from calc_package.latex_template import LATEX_TEMPLATE
from calc_package.latex_compiler import compile_pdf


# ── Jinja2 environment with LaTeX-safe delimiters ─────────────────────

def _make_env():
    """Create a Jinja2 Environment with custom delimiters for LaTeX."""
    env = Environment(
        loader=BaseLoader(),
        block_start_string=r"\BLOCK{",
        block_end_string="}",
        variable_start_string=r"\VAR{",
        variable_end_string="}",
        comment_start_string=r"\#{",
        comment_end_string="}",
        autoescape=False,
    )
    return env


# ── Item preparation ──────────────────────────────────────────────────

def _prepare_item(item):
    """Convert a data model item into a template-friendly dict.

    Applies unicode_to_latex for math fields and escape_latex for text.
    Returns a simple namespace object the template can access with dot
    notation.
    """
    itype = _item_type(item)

    class _NS:
        """Simple attribute namespace."""
        pass

    ns = _NS()
    ns._type = itype

    if itype == "calc_step":
        ns.title = escape_latex(item.title)
        # Keep originals for truthiness checks in template conditionals
        ns.equation = item.equation
        ns.substitution = item.substitution
        ns.equation_latex = unicode_to_latex(item.equation)
        ns.substitution_latex = unicode_to_latex(item.substitution)
        ns.result_name_latex = unicode_to_latex(item.result_name)
        # result_value is placed inside $...$ by the template, so use
        # unicode_to_latex (math-mode safe), not escape_latex (text-mode).
        ns.result_value = unicode_to_latex(str(item.result_value))
        ns.result_unit = escape_latex(item.result_unit)
        ns.reference = escape_latex(item.reference)
        ns.notes = escape_latex(item.notes)

    elif itype == "check":
        ns.description = escape_latex(item.description)
        ns.demand = item.demand
        ns.demand_label_latex = unicode_to_latex(item.demand_label)
        ns.capacity = item.capacity
        ns.capacity_label_latex = unicode_to_latex(item.capacity_label)
        ns.unit = escape_latex(item.unit)
        ns.passes = item.passes
        ns.relation = r"$\leq$" if item.passes else "$>$"
        dc = item.demand / item.capacity if item.capacity != 0 else 0
        ns.dc_ratio = f"{dc:.2f}"

    elif itype in ("input_table", "table"):
        ns.title = escape_latex(getattr(item, "title", ""))
        ns.headers = [escape_latex(str(h)) for h in item.headers]
        if itype == "input_table":
            # Input table: [Description, Symbol, Value, Unit]
            # Symbol column (index 1) is wrapped in $...$ by the template,
            # so it must use unicode_to_latex (math-mode), not escape_latex.
            ns.rows = []
            for row in item.rows:
                ns.rows.append([
                    escape_latex(str(row[0])),           # Description (text)
                    unicode_to_latex(str(row[1])),       # Symbol (math-mode)
                    escape_latex(str(row[2])),           # Value (text)
                    escape_latex(str(row[3])),           # Unit (text)
                ])
        else:
            ns.rows = [
                [escape_latex(str(cell)) for cell in row]
                for row in item.rows
            ]
        ns.notes = escape_latex(getattr(item, "notes", ""))
        # Column spec for longtable
        ncols = len(item.headers)
        if itype == "input_table":
            ns.col_spec = r"p{0.35\textwidth} >{\raggedright}p{0.15\textwidth} >{\raggedleft}p{0.20\textwidth} p{0.15\textwidth}"
        else:
            ns.col_spec = " ".join(["c"] * ncols)

    elif itype == "figure":
        ns.title = escape_latex(item.title)
        ns.caption = escape_latex(item.caption)
        ns.image_base64 = item.image_base64
        ns.image_path = ""  # Filled in by _write_figures()
        ns.width_frac = f"{item.width_percent / 100:.2f}"

    elif itype == "text":
        ns.text = escape_latex(str(item))

    return ns


def _prepare_sections(sections):
    """Prepare all sections for template rendering."""
    prepared = []
    for section in sections:
        class _Sec:
            pass
        sec = _Sec()
        sec.title = escape_latex(section.title)
        sec.items = [_prepare_item(item) for item in section.items]
        prepared.append(sec)
    return prepared


def _write_figures(prepared_sections, temp_dir: str):
    """Decode base64 figures to PNG files in temp_dir.

    Updates each figure item's image_path in-place.
    """
    fig_count = 0
    for section in prepared_sections:
        for item in section.items:
            if getattr(item, "_type", None) == "figure":
                fig_count += 1
                fig_path = Path(temp_dir) / f"fig_{fig_count}.png"
                img_data = base64.b64decode(item.image_base64)
                fig_path.write_bytes(img_data)
                # LaTeX needs forward slashes
                item.image_path = str(fig_path).replace("\\", "/")


# ── Public API ────────────────────────────────────────────────────────

def render_latex(data: CalcPackageData, figure_dir: str = None) -> str:
    """Render a CalcPackageData object to a LaTeX document string.

    Parameters
    ----------
    data : CalcPackageData
        Complete calculation package data.
    figure_dir : str, optional
        Directory where figure PNGs will be written.  If None, figures
        reference placeholder paths (useful for .tex-only output without
        compilation).

    Returns
    -------
    str
        Complete LaTeX document string (.tex content).
    """
    if not data.date:
        data.date = date.today().strftime("%Y-%m-%d")

    # Preprocess: gather InputItems into tables (reuse HTML logic)
    processed_sections = _preprocess_sections(data.sections)

    # Prepare for LaTeX (escape, convert equations)
    prepared = _prepare_sections(processed_sections)

    # Write figures if directory provided
    if figure_dir:
        _write_figures(prepared, figure_dir)

    # Build template context
    env = _make_env()
    template = env.from_string(LATEX_TEMPLATE)

    return template.render(
        title=escape_latex(data.analysis_type),
        project_name=escape_latex(data.project_name),
        project_number=escape_latex(data.project_number),
        engineer=escape_latex(data.engineer),
        checker=escape_latex(data.checker),
        company=escape_latex(data.company),
        date=escape_latex(data.date),
        sections=prepared,
        references=[escape_latex(r) for r in data.references],
        section_number=1,
    )


def save_latex(data: CalcPackageData, filepath: str) -> str:
    """Render and save a .tex file.

    Parameters
    ----------
    data : CalcPackageData
        Complete calculation package data.
    filepath : str
        Output file path (should end in .tex).

    Returns
    -------
    str
        Absolute path to the saved .tex file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create figure dir alongside the .tex file
    fig_dir = path.parent / "figures"
    fig_dir.mkdir(exist_ok=True)

    tex = render_latex(data, figure_dir=str(fig_dir))
    path.write_text(tex, encoding="utf-8")
    return str(path.resolve())


def save_pdf(
    data: CalcPackageData,
    filepath: str,
    keep_tex: bool = False,
    compiler: str = "pdflatex",
) -> str:
    """Render a .tex file, compile to PDF, and clean up.

    Parameters
    ----------
    data : CalcPackageData
        Complete calculation package data.
    filepath : str
        Output PDF file path.
    keep_tex : bool
        If True, keep the .tex source and figures alongside the PDF.
    compiler : str
        LaTeX compiler: ``"pdflatex"`` or ``"xelatex"``.

    Returns
    -------
    str
        Absolute path to the PDF file.

    Raises
    ------
    FileNotFoundError
        If the LaTeX compiler is not installed.
    RuntimeError
        If compilation fails.
    """
    output_path = Path(filepath).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_dir = tempfile.mkdtemp(prefix="geotech_calc_")
    try:
        tex_path = Path(temp_dir) / "calc_package.tex"

        # Render .tex with figures in temp dir
        tex_content = render_latex(data, figure_dir=temp_dir)
        tex_path.write_text(tex_content, encoding="utf-8")

        # Compile
        pdf_temp = compile_pdf(
            str(tex_path),
            output_dir=temp_dir,
            compiler=compiler,
        )

        # Copy PDF to final destination
        shutil.copy2(pdf_temp, str(output_path))

        # Optionally keep .tex and figures
        if keep_tex:
            tex_dest = output_path.with_suffix(".tex")
            shutil.copy2(str(tex_path), str(tex_dest))
            # Copy figure PNGs
            for png in Path(temp_dir).glob("fig_*.png"):
                shutil.copy2(str(png), str(output_path.parent / png.name))

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return str(output_path)
