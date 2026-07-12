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
from calc_package.renderer import _preprocess_sections, _item_type, render_html
from calc_package.equation_converter import unicode_to_latex, escape_latex
from calc_package.latex_template import LATEX_TEMPLATE
from calc_package.latex_compiler import compile_pdf, find_latex_compiler


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


def _save_pdf_latex(
    data: CalcPackageData,
    filepath: str,
    keep_tex: bool = False,
    compiler: str = "pdflatex",
) -> str:
    """Render a .tex file, compile to PDF via pdflatex, and clean up.

    The best-fidelity path (Mathcad-style layout, real math typesetting).
    Requires a LaTeX distribution on PATH. Returns the absolute PDF path.
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


# Wide-table styling injected ONLY into the Story-engine HTML (not the HTML or
# LaTeX outputs). The Story engine ignores ``table-layout:fixed``/``width`` and
# lays tables out at their natural width, and the base ``.data-table th/td``
# rule uses ``padding: 3px 8px`` — 16 px of horizontal padding per column, which
# for a 15-column table (the interslice slice-force table) pushes the table past
# the letter page and clips it at the right edge (the HTML output scrolls, but a
# PDF page cannot). ``_fit_wide_tables_for_pdf`` tags such tables with an extra
# ``wide-pdf-table`` class; this class-scoped rule (compound selector, so it
# out-specifies the base rule) collapses the padding and shrinks the font so all
# columns fit. Narrow tables are untouched.
_WIDE_TABLE_CLASS = "wide-pdf-table"
_PDF_STORY_CSS = (
    "<style>"
    f".data-table.{_WIDE_TABLE_CLASS} th,"
    f".data-table.{_WIDE_TABLE_CLASS} td"
    "{padding:0px 1px;font-size:6.5pt;"
    "word-break:break-word;overflow-wrap:anywhere;}"
    "</style>"
)

#: Header-column count above which a table is treated as "wide" and compacted.
_WIDE_TABLE_COLS = 11


def _fit_wide_tables_for_pdf(html: str, wide_cols: int = _WIDE_TABLE_COLS) -> str:
    """Tag WIDE tables so the Story PDF compacts them to fit the page width.

    A table with more than ``wide_cols`` columns (e.g. the 15-column interslice
    slice-force table) overflows the fixed letter page in the Story engine and
    clips at the right edge. Such tables get an extra ``wide-pdf-table`` class
    that :data:`_PDF_STORY_CSS` styles with collapsed padding + a smaller font;
    narrow tables keep the normal size. Column count is read from the ``<th>``
    cells of the table's first row.
    """
    import re

    def _proc(match: "re.Match") -> str:
        table = match.group(0)
        head = re.search(r"<tr.*?</tr>", table, re.DOTALL)
        n_cols = len(re.findall(r"<th\b", head.group(0))) if head else 0
        if n_cols > wide_cols:
            return table.replace(
                '<table class="data-table"',
                f'<table class="data-table {_WIDE_TABLE_CLASS}"', 1)
        return table

    return re.sub(r'<table class="data-table".*?</table>', _proc, html,
                  flags=re.DOTALL)


def _compress_pdf_images(html: str, jpg_quality: int = 90,
                         max_px: int = 1100) -> str:
    """Re-encode inline base64 PNG figures to JPEG for the Story PDF.

    ``fitz.Story`` embeds a PNG ``<img>`` by decoding it to a RAW, uncompressed
    pixmap — a 72 kB PNG plot balloons to a ~4 MB raw image in the PDF, so a
    handful of figures made the report ~13-19 MB. A JPEG source, by contrast, is
    preserved as DCT-compressed image data. So every figure is re-encoded to
    JPEG here (ALWAYS — even when the JPEG *base64* is larger than the PNG, its
    *embedded* form is far smaller), and downscaled if wildly oversampled for a
    letter page. This cuts the PDF to ~1-2 MB with no visible loss on line
    plots. Best-effort: an image that cannot be re-encoded is left untouched.
    Only the Story PDF path calls this; HTML/LaTeX outputs are unaffected.
    """
    import base64
    import re

    try:
        import fitz
    except Exception:  # pragma: no cover - fitz is required to reach here
        return html

    def _repl(match: "re.Match") -> str:
        try:
            raw = base64.b64decode(match.group(1))
            pix = fitz.Pixmap(raw)
            if pix.alpha:                       # JPEG has no alpha channel
                pix = fitz.Pixmap(pix, 0)
            longest = max(pix.width, pix.height)
            if longest > max_px:                # halve until within the cap
                n = 0
                while longest > max_px and n < 4:
                    n += 1
                    longest /= 2
                pix.shrink(n)
            jpg = pix.tobytes("jpeg", jpg_quality=jpg_quality)
            return ('src="data:image/jpeg;base64,'
                    + base64.b64encode(jpg).decode("ascii") + '"')
        except Exception:
            return match.group(0)

    return re.sub(r'src="data:image/png;base64,([^"]+)"', _repl, html)


def _save_pdf_story(data: CalcPackageData, filepath: str) -> str:
    """Pure-Python HTML -> PDF via PyMuPDF's Story engine (no LaTeX needed).

    Renders the SAME self-contained HTML as ``format="html"`` (``render_html``)
    into a paginated PDF, with two Story-only adjustments: inline PNG figures are
    re-encoded to JPEG (so they stay compressed in the PDF instead of bloating
    it) and wide tables are wrapped to the page width (so they cannot clip at the
    right edge). The Story engine supports a SUBSET of CSS: it lays out text,
    headings, tables, colours and inline base64 ``<img>`` figures, but ignores
    advanced CSS (flexbox/grid, positioned layout, some border/spacing rules), so
    the result is a plainer-looking document than the LaTeX PDF. The CONTENT is
    complete — only the styling fidelity differs. Returns the absolute PDF path.
    """
    try:
        import fitz  # PyMuPDF, installed via the [pdf] extra
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PyMuPDF (fitz) is not installed, so the pure-Python PDF fallback "
            "is unavailable. Install the [pdf] extra: "
            "pip install 'geotech-staff-engineer[pdf]'."
        ) from exc

    html = render_html(data)
    html = _compress_pdf_images(html)
    html = _fit_wide_tables_for_pdf(html)
    if "</head>" in html:
        html = html.replace("</head>", _PDF_STORY_CSS + "</head>", 1)
    else:
        html = _PDF_STORY_CSS + html
    output_path = Path(filepath).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    story = fitz.Story(html=html)
    writer = fitz.DocumentWriter(str(output_path))
    mediabox = fitz.paper_rect("letter")
    where = mediabox + (54, 54, -54, -54)      # ~0.75 in margins
    more = 1
    while more:
        device = writer.begin_page(mediabox)
        more, _filled = story.place(where)
        story.draw(device)
        writer.end_page()
    writer.close()
    return str(output_path)


def render_pdf(
    data: CalcPackageData,
    filepath: str,
    keep_tex: bool = False,
    compiler: str = "pdflatex",
    *,
    renderer: str = "auto",
) -> dict:
    """Render a calc package to PDF via a fallback chain; return a report dict.

    Chain (``renderer="auto"``, the default): **pdflatex** if it is on PATH
    (best fidelity, output byte-identical to the historical path) -> **PyMuPDF
    Story** pure-Python HTML->PDF (no LaTeX; simpler layout, CSS subset) -> a
    clear error with print-to-PDF advice if neither works. Force a single leg
    with ``renderer="pdflatex"`` (raises if unavailable) or
    ``renderer="pymupdf_story"``.

    Returns
    -------
    dict
        ``{"path": str, "renderer": "pdflatex"|"pymupdf_story", "warnings": [str]}``.

    Raises
    ------
    FileNotFoundError
        With ``renderer="pdflatex"`` when the compiler is not installed.
    RuntimeError
        When neither renderer can produce a PDF.
    """
    if renderer not in ("auto", "pdflatex", "pymupdf_story"):
        raise ValueError(
            "renderer must be 'auto', 'pdflatex', or 'pymupdf_story'")
    warnings: list = []
    have_latex = shutil.which(compiler) is not None

    if renderer == "pdflatex" and not have_latex:
        find_latex_compiler(compiler)          # raises the informative error

    if renderer == "pdflatex" or (renderer == "auto" and have_latex):
        path = _save_pdf_latex(data, filepath, keep_tex=keep_tex,
                               compiler=compiler)
        return {"path": path, "renderer": "pdflatex", "warnings": warnings}

    # Pure-Python fallback.
    if renderer == "auto":
        warnings.append(
            f"'{compiler}' not found on PATH; produced the PDF with the PyMuPDF "
            "Story HTML engine (simpler layout, CSS subset). Install a LaTeX "
            "distribution (MiKTeX / TeX Live) for the full-fidelity PDF.")
    try:
        path = _save_pdf_story(data, filepath)
    except Exception as exc:
        raise RuntimeError(
            "Could not produce a PDF: pdflatex is not installed and the "
            f"PyMuPDF Story fallback failed ({type(exc).__name__}: {exc}). "
            "Generate HTML output (format='html') and use your browser's "
            "Print -> Save as PDF, or install a LaTeX distribution "
            "(MiKTeX / TeX Live)."
        ) from exc
    return {"path": path, "renderer": "pymupdf_story", "warnings": warnings}


def save_pdf(
    data: CalcPackageData,
    filepath: str,
    keep_tex: bool = False,
    compiler: str = "pdflatex",
    *,
    renderer: str = "auto",
) -> str:
    """Render a calc package to PDF and return the absolute file path.

    Uses the :func:`render_pdf` fallback chain (pdflatex -> PyMuPDF Story ->
    clear error). Default-preserving: when ``pdflatex`` is on PATH the output is
    byte-for-byte the historical LaTeX PDF. Call :func:`render_pdf` instead if
    you need to know which renderer was used.

    Parameters
    ----------
    data : CalcPackageData
        Complete calculation package data.
    filepath : str
        Output PDF file path.
    keep_tex : bool
        If True, keep the .tex source and figures alongside the PDF (pdflatex
        leg only).
    compiler : str
        LaTeX compiler: ``"pdflatex"`` or ``"xelatex"``.
    renderer : str
        ``"auto"`` (default), ``"pdflatex"``, or ``"pymupdf_story"``.

    Returns
    -------
    str
        Absolute path to the PDF file.
    """
    return render_pdf(data, filepath, keep_tex=keep_tex, compiler=compiler,
                      renderer=renderer)["path"]
