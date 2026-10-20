"""
LaTeX renderer for calculation packages.

Converts CalcPackageData into a .tex document string using Jinja2,
and optionally compiles it to PDF via pdflatex.

Reuses the same preprocessing pipeline as the HTML renderer
(InputItem collapsing, item-type dispatch).
"""

import base64
import re
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
        autoescape=False,   # nosec B701 — this env renders LaTeX, not HTML;
                            # HTML-escaping would corrupt TeX markup. The
                            # HTML pipeline (renderer.py) autoescapes (CP-1).
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
        ns.compact = False
        if itype == "input_table":
            ns.col_spec = r"p{0.35\textwidth} >{\raggedright}p{0.15\textwidth} >{\raggedleft}p{0.20\textwidth} p{0.15\textwidth}"
        elif ncols >= _COMPACT_TABLE_COLS:
            # A wide table (the calculation tables, the slice forces):
            # wrapping columns that share the text width, in small type with
            # 2 pt column padding (the template sets both), instead of
            # natural-width "c" columns that run off the page.
            frac = (1.0 - 0.0095 * ncols) / ncols * 0.97
            ns.col_spec = " ".join(
                [r">{\centering\arraybackslash}p{%.4f\textwidth}" % frac]
                * ncols)
            ns.compact = True
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


#: Column count from which a data table is set compact (see _prepare_item).
_COMPACT_TABLE_COLS = 7


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
            if longest > max_px:                # scaled to the cap
                # (halving left a tall figure under the text width at the
                # engine's 96 px an inch, so it printed smaller; GM-186)
                s = max_px / longest
                pix = fitz.Pixmap(pix, max(1, round(pix.width * s)),
                                  max(1, round(pix.height * s)))
            jpg = pix.tobytes("jpeg", jpg_quality=jpg_quality)
            return ('src="data:image/jpeg;base64,'
                    + base64.b64encode(jpg).decode("ascii") + '"')
        except Exception:
            return match.group(0)

    return re.sub(r'src="data:image/png;base64,([^"]+)"', _repl, html)


def _save_pdf_story(data: CalcPackageData, filepath: str) -> dict:
    """Pure-Python HTML -> PDF via PyMuPDF's Story engine (no LaTeX needed).

    Renders the SAME self-contained HTML as ``format="html"`` (``render_html``)
    into a paginated PDF. See :func:`render_html_pdf_story` for the engine's
    capabilities and limits, and for what it returns: the absolute PDF path
    and each figure's printed size and lettering.
    """
    return render_html_pdf_story(render_html(data), filepath)


# ── Story PDF figure layout (BACKLOG GM-168) ──────────────────────────
#
# The Story engine shrinks a picture to whatever height is left on the page
# it reaches: live wave w13 printed figures 0.11, 0.3, 1.6 and 2.1 in wide
# at page feet, with lettering about 3 pt. The layout rule here:
#
# - every figure prints at the size the HTML lays it out at on a page of its
#   own (its CSS width, else its natural size, within the text width);
# - a figure that does not fit what is left of a page starts the next page
#   at that size, with the heading or "Figure n" title just before it; the
#   text after it follows it there (the engine has no floats), so the page
#   before ends short. So does one the engine drew past a page's foot (it
#   does that after a picture it shrank, and drops the text after it), and
#   one that starts a page with its heading left on the page before;
# - a figure taller than a whole page starts a page and is scaled to fit
#   it, leaving room for its caption. It is never shrunk to fit what is left
#   of a page. A picture cannot be split across pages, so one that a page
#   holds only under MIN_FIGURE_SCALE of its width still prints at the page
#   size, and its record says so (``scaled``) for the caller to warn.

#: The Story PDF page: US letter, ~0.75 in margins.
_STORY_MARGIN_PT = 54
#: A figure printed narrower than this share of its laid-out width was
#: shrunk by the engine to fit what was left of a page.
_SQUEEZED = 0.97
#: Room kept under a figure scaled to a page, for its caption (points).
_CAPTION_ROOM_PT = 36
#: The smallest share of its laid-out width a figure taller than a page
#: should be scaled to; under it the caller is told to draw it shorter.
MIN_FIGURE_SCALE = 0.5
#: The most pages a Story PDF runs to: a page break before the document's
#: first element, or anywhere after a table whose borders are not collapsed,
#: makes the engine start the document again for ever (PyMuPDF 1.27).
_MAX_STORY_PAGES = 2000
#: What starts a new page before a figure.
_PAGE_BREAK = '<div style="page-break-before:always"></div>'
#: Put before the HTML when a page break makes the engine loop: the loop
#: after a table goes when its borders are collapsed. First in the document,
#: so the HTML's own table rules still win.
_COLLAPSED_TABLES = "<style>table{border-collapse:collapse}</style>"


class _StoryLoop(RuntimeError):
    """The Story engine ran past its page limit without placing
    everything: it is repeating the document."""


#: Block elements a page break can go before, when the engine opens one
#: past a page's foot (GM-198, :func:`_story_bytes`). Table rows and list
#: items too: the engine drops a row or an item it opens there.
_BLOCK_START = re.compile(
    r"<(?:div|p|table|tr|li|h[1-6]|ul|ol|pre|blockquote|figure|section)"
    r"(?=[\s/>])", re.IGNORECASE)
#: Of those, the ones that hold lines of text, not other blocks: a table or
#: a section that runs past a page foot can be whole, one of these is not.
_TEXT_BLOCK_START = re.compile(r"<(?:p|li|h[1-6]|pre)(?=[\s/>])",
                               re.IGNORECASE)
_ID_ATTR = re.compile(r"""\sid\s*=\s*["']?([^"'\s>]+)""", re.IGNORECASE)
#: A heading's end tag, then nothing but space (to the element after it).
_HEADING_END = re.compile(r"</h([1-6])\s*>\s*$", re.IGNORECASE)
#: The tags that say which table row an offset is in.
_TABLE_PARTS = re.compile(r"<(/?)(table|thead|tr)(?=[\s/>])", re.IGNORECASE)
_HEADING_ROW = re.compile(r"\s*<th(?=[\s/>])", re.IGNORECASE)
#: Elements past a page foot dealt with, one per layout, in one Story PDF;
#: and the layouts tried for them, at most.
_MAX_FOOT_BREAKS = 25
_MAX_FOOT_TRIALS = 40
#: Containers a figure's own block climbs through, when the picture is the
#: first thing in them and their only picture.
_FIGURE_WRAPPERS = {"div", "figure", "p", "center", "a", "span", "picture"}
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input",
              "link", "meta", "source", "track", "wbr"}
_UNSEEN_TAGS = {"head", "style", "script", "title"}


def _drawn(src) -> bool:
    """An ``<img>`` the engine draws: one carrying its picture inline (the
    renderers embed every image as a data URI before laying out)."""
    return (src or "").strip().lower().startswith("data:image/")


def _figure_marks(html: str) -> dict:
    """Where each drawn ``<img>`` (:func:`_drawn`) sits in ``html``, in
    document order:
    ``{"spans": [(start, end)], "breaks": [offset or None],
    "titled": [bool]}``. ``breaks`` is where a page break moves the figure
    to the next page: before its own block (the containers it is the first
    thing in), a table holding it, and the heading or short "Figure ..."
    title just before that (not the caption under the figure before it).
    ``None`` where that would come before anything else in the document
    (a break there makes the engine loop). ``titled``: a heading or title
    goes with it."""
    from html.parser import HTMLParser

    line_starts = [0]
    pos = html.find("\n")
    while pos != -1:
        line_starts.append(pos + 1)
        pos = html.find("\n", pos + 1)

    root = {"tag": "#root", "start": 0, "children": [], "parent": None,
            "imgs": 0, "head": ""}
    spans, img_nodes = [], []
    state = {"first": None, "unseen": 0}

    class _Parser(HTMLParser):
        def __init__(self):
            super().__init__(convert_charrefs=True)
            self.stack = [root]

        def _offset(self):
            line, col = self.getpos()
            return line_starts[line - 1] + col

        def handle_starttag(self, tag, attrs):
            start = self._offset()
            text = self.get_starttag_text() or ""
            parent = self.stack[-1]
            node = {"tag": tag, "start": start, "children": [],
                    "parent": parent, "imgs": 0, "head": ""}
            if tag == "img" and _drawn(dict(attrs).get("src")):
                spans.append((start, start + len(text)))
                img_nodes.append(node)
                for anc in self.stack:
                    anc["imgs"] += 1
                if state["first"] is None and not state["unseen"]:
                    state["first"] = start
            if not state["unseen"]:
                parent["children"].append(node)
            if tag in _UNSEEN_TAGS:
                state["unseen"] += 1
            if tag not in _VOID_TAGS:
                self.stack.append(node)

        def handle_startendtag(self, tag, attrs):
            self.handle_starttag(tag, attrs)
            if tag not in _VOID_TAGS:
                self.handle_endtag(tag)

        def handle_endtag(self, tag):
            for i in range(len(self.stack) - 1, 0, -1):
                if self.stack[i]["tag"] == tag:
                    state["unseen"] -= sum(
                        1 for n in self.stack[i:] if n["tag"] in _UNSEEN_TAGS)
                    del self.stack[i:]
                    break

        def handle_data(self, data):
            text = " ".join(data.split())
            if not text or state["unseen"]:
                return
            if state["first"] is None:
                state["first"] = self._offset()
            self.stack[-1]["children"].append({"tag": "#text"})
            for anc in self.stack:
                if len(anc["head"]) < 40:
                    anc["head"] = (anc["head"] + " " + text).strip()

    parser = _Parser()
    parser.feed(html)
    parser.close()

    def _previous(node):
        sibs = node["parent"]["children"] if node["parent"] else []
        i = next((j for j, n in enumerate(sibs) if n is node), 0)
        return sibs[i - 1] if i > 0 else None

    def _next(node):
        sibs = node["parent"]["children"] if node["parent"] else []
        i = next((j for j, n in enumerate(sibs) if n is node), len(sibs))
        return sibs[i + 1] if i + 1 < len(sibs) else None

    def _fig_text(node) -> bool:
        return (node is not None and node["tag"] in ("p", "div")
                and not node["imgs"] and node["head"][:3].lower() == "fig")

    def _holds_picture(node) -> bool:
        return node is not None and (
            node.get("imgs") or any(node is i for i in img_nodes))

    def _block(img):
        node = img
        while (node["parent"] and node["parent"]["tag"] in _FIGURE_WRAPPERS
               and node["parent"]["imgs"] == 1
               and node["parent"]["children"][:1] == [node]):
            node = node["parent"]
        anc = img["parent"]
        while anc is not None:
            if anc["tag"] == "table":
                node = anc                         # the outermost table
            anc = anc["parent"]
        return node

    blocks = [_block(img) for img in img_nodes]
    # Captions under their figures (the house style: render_figures says
    # "use the caption text under it") or titles above them: a "Figure ..."
    # line between two figures belongs to the one before when most figures
    # have their "Figure ..." line under them.
    below = sum(1 for b in blocks if _fig_text(_next(b)))
    above = sum(1 for b in blocks if _fig_text(_previous(b))
                and not _holds_picture(_previous(_previous(b))))
    captions_below = below > above

    def _title(node) -> bool:
        if node["tag"] in _HEADINGS:
            return True
        if not _fig_text(node):
            return False
        # under a figure, it is that figure's caption
        return not (captions_below and _holds_picture(_previous(node)))

    breaks, titled = [], []
    for block in blocks:
        node = block
        for _ in range(2):                         # heading, then title
            prev = _previous(node)
            if prev is None or not _title(prev):
                break
            node = prev
        first = state["first"]
        breaks.append(None if first is None or node["start"] <= first
                      else node["start"])
        titled.append(node is not block)
    return {"spans": spans, "breaks": breaks, "titled": titled}


def _tag_blocks(html: str):
    """``(html, starts)``: ``html`` with an id on each block element
    (:data:`_BLOCK_START`) that has none, and ``{id: offset}``, where each
    one's start tag is in the ``html`` given. An id the HTML uses twice is
    left out."""
    starts, twice, parts, last = {}, set(), [], 0
    for m in _BLOCK_START.finditer(html):
        end = html.find(">", m.end())
        found = _ID_ATTR.search(html, m.end(),
                                len(html) if end == -1 else end)
        if found:
            key = found.group(1)
            if key in starts:
                twice.add(key)
            starts.setdefault(key, m.start())
            continue
        key = f"gse-block-{m.start()}"
        starts[key] = m.start()
        parts += [html[last:m.end()], f' id="{key}"']
        last = m.end()
    parts.append(html[last:])
    for key in twice:
        del starts[key]
    return "".join(parts), starts


def _story_bytes(html: str, mediabox, where, max_pages=None) -> bytes:
    """``html`` laid out by the Story engine into pages ``where`` on
    ``mediabox``: the PDF's bytes. :class:`_StoryLoop` past ``max_pages``
    (default :data:`_MAX_STORY_PAGES`).

    An element the engine opens past the foot of a page is not drawn, nor
    anything in it (GM-198: a pile group package's "Figures" section, plan
    view and caption, after a table that ran to 6 pt above the foot; a
    section of calculation steps in another; a table row or a list item),
    and the lines of a paragraph or list item it runs past the foot are
    not drawn either. A page break goes before it, or before the heading
    just above it so the heading goes with it (:func:`_foot_breaks`), and
    the HTML is laid out again. When no break moves it without looping the
    engine (a table row taller than a page), it is left as it was and the
    next one after it is moved; :data:`_MAX_FOOT_TRIALS` layouts in all."""
    import fitz

    pdf, at = _story_pass(html, mediabox, where, max_pages)
    if at is None:
        return pdf
    # a break adds a page at most, so a trial past this is the engine looping
    with fitz.open("pdf", pdf) as doc:
        limit = min(max_pages or _MAX_STORY_PAGES,
                    doc.page_count + _MAX_FOOT_BREAKS + 2)
    after = -1                     # elements up to here are left as they are
    trials = 0
    for _ in range(_MAX_FOOT_BREAKS):
        trial = None
        for (pos, insert), prefix in ((e, p) for e in _foot_breaks(html, at)
                                      for p in ("", _COLLAPSED_TABLES)):
            # a break after a table loops the engine unless the table
            # borders are collapsed (as in _lay_out_figures)
            if prefix and html.startswith(prefix):
                continue
            if trials >= _MAX_FOOT_TRIALS:
                return pdf
            trials += 1
            text = prefix + html[:pos] + insert + html[pos:]

            def _moved(offset, pos=pos, insert=insert, prefix=prefix):
                if offset < 0:
                    return offset
                return (offset + len(prefix)
                        + (len(insert) if pos <= offset else 0))
            try:
                got_pdf, got_at = _story_pass(text, mediabox, where, limit,
                                              after=_moved(after),
                                              loop_early=True)
            except _StoryLoop:
                continue
            # still the first one past a foot (or one before it is): the
            # break did not move it
            if got_at is not None and got_at <= _moved(at):
                continue
            trial = (text, got_pdf, got_at, _moved(after))
            break
        if trial is None:
            after = _element_end(html, at)
            try:
                at = _story_pass(html, mediabox, where, limit, after=after)[1]
            except _StoryLoop:
                break
        else:
            html, pdf, at, after = trial
        if at is None:
            break
    return pdf


def _element_end(html: str, at: int) -> int:
    """Where the element whose start tag is at ``at`` ends (``at`` itself
    when its end tag is not found): an element left as it is, is left with
    everything in it."""
    name = re.match(r"<([A-Za-z0-9]+)", html[at:at + 12])
    if name is None:
        return at
    depth = 0
    for m in re.finditer(rf"<(/?){name.group(1)}(?=[\s/>])", html[at:],
                         re.IGNORECASE):
        depth += -1 if m.group(1) else 1
        if depth == 0:
            close = html.find(">", at + m.end())
            return len(html) if close == -1 else close + 1
    return at


def _table_row(html: str, at: int):
    """``(table start, row start, body rows before)`` for the innermost
    table whose row starts at or holds ``at``: where its start tag and the
    row's are, and where its body rows (not in ``<thead>``, not headings
    only) before the row start. ``None`` when ``at`` is in no table."""
    tables = []
    for m in _TABLE_PARTS.finditer(html, 0, at):
        if m.group(2).lower() == "table":
            if m.group(1):
                if tables:
                    tables.pop()
            else:
                tables.append(m.start())
    if not tables:
        return None
    depth, head, row, rows = 0, False, None, []
    for m in _TABLE_PARTS.finditer(html, tables[-1], min(len(html), at + 4)):
        tag, closing = m.group(2).lower(), bool(m.group(1))
        if tag == "table":
            depth += -1 if closing else 1
        elif depth == 1 and tag == "thead":
            head = not closing
        elif depth == 1 and tag == "tr" and not closing and m.start() <= at:
            if row is not None and not row[1]:
                rows.append(row[0])
            # a row of <th> cells is the table's heading, kept with its rows
            row = (m.start(), head or bool(_HEADING_ROW.match(
                html, html.find(">", m.start()) + 1)))
    if row is None:
        return None
    return tables[-1], row[0], rows


def _foot_breaks(html: str, at: int) -> list:
    """The page breaks that may move the element starting at ``at`` to the
    next page (GM-198), in the order :func:`_story_bytes` tries them, as
    ``[(offset, text inserted)]``.

    A row of a table is moved by ending the table before it and starting
    it again on the next page (the engine places a break between rows
    before the table, and ignores one in a cell or on a row); its first
    body row, by a break before the table. Anything else gets a break
    before it. A break at a page foot can be placed past the foot too, and
    ignored (inside a calculation step, after the line or the row before
    it), so then the block or the row just before it goes too (a table's
    last row), then each block holding it, innermost first. Each is tried
    first before a heading just above it, so the heading goes with it.
    """
    edits = []
    row = _table_row(html, at)
    if row is not None:
        table, tr, before = row
        edits += _row_splits(html, table, [tr] + before[1:][-1:], before)
        at = table
    previous, holding = _blocks_before(html, at)
    targets = [at] + ([previous] if previous is not None else []) + holding
    for target in targets:
        heading = _HEADING_END.search(html, max(0, target - 2000), target)
        spots = [target]
        if heading:
            spots.insert(0, html.rfind(f"<h{heading.group(1)}", 0, target))
        if target == previous and re.match(r"<table(?=[\s/>])",
                                           html[target:target + 7],
                                           re.IGNORECASE):
            end = _element_end(html, target)
            last = (_table_row(html, html.rfind("<", target, end))
                    if end > target else None)
            if last is not None:
                edits += _row_splits(html, last[0], [last[1]], last[2])
        edits += [(spot, _PAGE_BREAK) for spot in spots if spot >= 0]
    return list(dict.fromkeys(edits))


def _row_splits(html: str, table: int, rows: list, before: list) -> list:
    """The edits that end the table starting at ``table`` before each of
    ``rows`` and start it again on the next page; none when no body row
    comes before (``before``: :func:`_table_row`)."""
    if not before:
        return []
    tag = html[table:html.find(">", table) + 1]
    again = "</table>" + _PAGE_BREAK + _ID_ATTR.sub("", tag)
    return [(start, again) for start in rows]


#: Block start and end tags, for the blocks before and holding an offset.
_BLOCK_TAGS = re.compile(
    r"<(/?)(div|section|figure|blockquote|ul|ol|li|table|td|th|tr|p|pre"
    r"|h[1-6])(?=[\s/>])", re.IGNORECASE)
#: Blocks climbed through, from an element no break moves.
_MAX_BLOCKS_CLIMBED = 3
_CELL_TAGS = ("td", "th", "tr")


def _blocks_before(html: str, at: int):
    """``(previous, holding)``: where the block just before the offset
    ``at`` starts, in the block holding it (``None``: none is), and where
    the blocks holding ``at`` start, innermost first, up to
    :data:`_MAX_BLOCKS_CLIMBED`; none from inside a table (its rows are
    moved by :func:`_foot_breaks`)."""
    stack = [["", 0, None]]         # tag, start, the last block closed in it
    for m in _BLOCK_TAGS.finditer(html, 0, at):
        tag = m.group(2).lower()
        if not m.group(1):
            stack.append([tag, m.start(), None])
            continue
        for i in range(len(stack) - 1, 0, -1):
            if stack[i][0] == tag:
                start = stack[i][1]
                del stack[i:]
                stack[-1][2] = start
                break
    if stack[-1][0] in _CELL_TAGS:
        return None, []
    holding = []
    for tag, start, _last in reversed(stack[1:]):
        if tag in _CELL_TAGS or len(holding) >= _MAX_BLOCKS_CLIMBED:
            break
        holding.append(start)
    return stack[-1][2], holding


def _story_pass(html: str, mediabox, where, max_pages=None, after=-1,
                loop_early=False):
    """``(pdf bytes, offset)``: one layout of ``html``
    (:func:`_story_bytes`), and where in ``html`` the first block element
    past ``after`` starts that the engine opened past a page's foot, or a
    paragraph, list item or heading it ran past one (``None``: none).
    :class:`_StoryLoop` past ``max_pages``, or with ``loop_early`` as soon
    as the engine places an element a second time (a trial page break that
    loops it)."""
    import io

    import fitz

    tagged, starts = _tag_blocks(html)
    texts = {key for key, at in starts.items()
             if _TEXT_BLOCK_START.match(html, at)}
    past_foot, opened, again = [], set(), []

    def _seen(pos):
        if pos.id not in starts:
            return
        if pos.open_close & 1:
            # the engine looping places the same element again
            if pos.id in opened:
                again.append(pos.id)
            opened.add(pos.id)
        if past_foot or starts[pos.id] <= after:
            return
        top, bottom = pos.rect[1], pos.rect[3]
        if bottom <= top:
            return
        if ((pos.open_close & 1 and top >= where.y1 - 0.5)
                or (pos.id in texts and bottom > where.y1 + 0.5)):
            past_foot.append(starts[pos.id])

    limit = max_pages or _MAX_STORY_PAGES
    story = fitz.Story(html=tagged)
    buf = io.BytesIO()
    writer = fitz.DocumentWriter(buf)
    more, pages = 1, 0
    while more:
        if pages >= limit or (loop_early and again):
            writer.close()
            raise _StoryLoop(
                f"the layout ran past {limit} pages without placing "
                "everything (a page break before the first element of the "
                "HTML makes the engine repeat an empty page): remove that "
                "page break")
        device = writer.begin_page(mediabox)
        more, _filled = story.place(where)
        if not past_foot or loop_early:
            story.element_positions(_seen)
        story.draw(device)
        writer.end_page()
        pages += 1
    writer.close()
    return buf.getvalue(), (past_foot[0] if past_foot else None)


def _story_images(pdf: bytes) -> list:
    """``[(page index, rect, at_top)]`` for each picture drawn in ``pdf``,
    in the order drawn (document order). ``at_top``: nothing is printed
    above it on its page."""
    import fitz

    out = []
    with fitz.open("pdf", pdf) as doc:
        for page in doc:
            infos = page.get_image_info()
            tops = [b[3] for b in page.get_text("blocks")]
            tops += [fitz.Rect(i["bbox"]).y1 for i in infos]
            for info in infos:
                r = fitz.Rect(info["bbox"])
                out.append((page.number, r,
                            not any(y <= r.y0 + 1 for y in tops)))
    return out


def _with_figure_edits(html: str, marks: dict, edits: dict) -> str:
    """``html`` with each figure's page break and width (``edits``:
    ``{k: {"break": bool, "width_pt": float}}``) written in."""
    import re

    inserts = {}                                   # (start, end) -> text
    for k, e in edits.items():
        if e.get("width_pt"):
            start, end = marks["spans"][k]
            tag = html[start:end]
            width = f"width:{e['width_pt']:.1f}pt"
            m = re.search(r"\bstyle\s*=\s*(\"([^\"]*)\"|'([^']*)')", tag)
            if m:
                old = m.group(2) if m.group(2) is not None else m.group(3)
                old = old.strip().rstrip(";")
                tag = (tag[:m.start()] + 'style="'
                       + (old + ";" if old else "") + width + '"'
                       + tag[m.end():])
            else:
                tag = tag[:4] + f' style="{width}"' + tag[4:]
            inserts[(start, end)] = tag
        if e.get("break") and marks["breaks"][k] is not None:
            at = marks["breaks"][k]
            inserts.setdefault((at, at), _PAGE_BREAK)
    for (start, end), text in sorted(inserts.items(), reverse=True):
        html = html[:start] + text + html[end:]
    return html


def _lay_out_figures(html: str, mediabox, where):
    """``(pdf bytes, figures)``: ``html`` laid out with no figure shrunk to
    fit what is left of a page (see the rule above). ``figures``: per
    picture, in document order, ``{"pdf_page", "width_pt", "height_pt",
    "laid_out_width_pt", "moved", "scaled"}``; ``None`` when the pictures
    drawn cannot be matched to the HTML's ``<img>`` tags."""
    import fitz

    try:
        pdf = _story_bytes(html, mediabox, where)
    except _StoryLoop:
        # the HTML's own page break after a table: collapsed borders end it
        html = _COLLAPSED_TABLES + html
        pdf = _story_bytes(html, mediabox, where)
    placed = _story_images(pdf)
    try:
        marks = _figure_marks(html)
    except Exception:                                  # noqa: BLE001
        marks = None
    n = len(placed)
    if not n or marks is None or len(marks["spans"]) != n:
        return pdf, None
    # the size each figure is laid out at on a page of its own: one frame
    # tall enough for the whole document, so nothing reaches a page foot
    with fitz.open("pdf", pdf) as doc:
        pages = doc.page_count
        tall_h = (pages + 2) * where.height
    tall_box = fitz.Rect(0, 0, mediabox.width, tall_h + 2 * where.y0)
    tall_where = fitz.Rect(where.x0, where.y0, where.x1, where.y0 + tall_h)
    laid_out = [r for _p, r, _t in
                _story_images(_story_bytes(html, tall_box, tall_where))]
    if len(laid_out) != n:
        return pdf, None
    # a move adds a page at most, so a trial past this is the engine looping
    trial_pages = pages + n + 4

    edits = {}
    settled = set()

    def _target(k):
        return edits.get(k, {}).get("width_pt") or laid_out[k].width

    def _misplaced(i, p, r, at_top):
        # shrunk to fit what was left of a page, or drawn past its foot
        # (the engine does that to a picture after one it shrank, and drops
        # the text after it), or starting a page with its heading or title
        # left at the foot of the page before
        return (r.width < _SQUEEZED * _target(i) or r.y1 > where.y1 + 1
                or (at_top and p > 0 and marks["titled"][i]
                    and not edits.get(i, {}).get("break")))

    def _trial(edits):
        trial_pdf = _story_bytes(_with_figure_edits(html, marks, edits),
                                 mediabox, where, max_pages=trial_pages)
        return trial_pdf, _story_images(trial_pdf)

    for _ in range(2 * n + 1):
        k = next((i for i, (p, r, t) in enumerate(placed)
                  if i not in settled and _misplaced(i, p, r, t)), None)
        if k is None:
            break
        p, r, at_top = placed[k]
        before = dict(edits.get(k, {}))
        e = edits.setdefault(k, {})
        if not e.get("break") and marks["breaks"][k] is not None \
                and (not at_top or marks["titled"][k]):
            e["break"] = True                  # the next page, at its width
        else:
            # it does not fit a page, or cannot move: the engine has fitted
            # it to the page; keep room under it for its caption
            tall = min(r.y1, where.y1) - r.y0
            if tall > 0:
                room = min(_CAPTION_ROOM_PT, 0.5 * tall)
                e["width_pt"] = r.width * (tall - room) / r.height
            settled.add(k)
        try:
            trial_pdf, trial_placed = _trial(edits)
        except _StoryLoop:
            # a page break after a table whose borders are not collapsed
            # makes the engine loop; collapsed, it does not
            loops = True
            if not html.startswith(_COLLAPSED_TABLES):
                html = _COLLAPSED_TABLES + html
                marks = _figure_marks(html)
                try:
                    trial_pdf, trial_placed = _trial(edits)
                    loops = False
                except _StoryLoop:
                    pass
            if loops:                          # leave this figure as it was
                edits[k] = before
                settled.add(k)
                continue
        if len(trial_placed) != n:
            break
        pdf, placed = trial_pdf, trial_placed

    figures = []
    for k, (page, r, _t) in enumerate(placed):
        e = edits.get(k, {})
        full = laid_out[k].width
        figures.append({
            "pdf_page": page + 1,
            "width_pt": r.width, "height_pt": r.height,
            "laid_out_width_pt": full,
            "moved": bool(e.get("break")) and marks["breaks"][k] is not None,
            "scaled": (round(r.width / full, 2)
                       if full and r.width < _SQUEEZED * full else None),
        })
    return pdf, figures


def _png_letterings(html: str) -> list:
    """Per drawn ``<img>`` in ``html`` (:func:`_drawn`), in document order,
    the lettering its PNG carries
    (``geotech_common.layout_ink.parse_lettering``), or ``{}``."""
    import re

    from geotech_common.layout_ink import (LETTERING_KEY, parse_lettering,
                                           png_texts)
    try:
        # the tags :func:`_figure_marks` matches to the pictures drawn (not
        # one in a comment, a script or the head)
        tags = [html[start:end]
                for start, end in _figure_marks(html)["spans"]]
    except Exception:                                      # noqa: BLE001
        tags = [m.group(0) for m in re.finditer(r"<img\b[^>]*>", html,
                                                flags=re.IGNORECASE)]
    out = []
    for tag in tags:
        src = re.search(r"\bsrc\s*=\s*[\"']([^\"']*)", tag)
        if not src or not _drawn(src.group(1)):
            continue
        m = re.match(r"\s*data:image/png;base64,(.+)", src.group(1),
                     flags=re.IGNORECASE | re.DOTALL)
        found = {}
        if m:
            try:
                png = base64.b64decode(m.group(1))
                found = parse_lettering(png_texts(png).get(LETTERING_KEY))
            except Exception:                              # noqa: BLE001
                found = {}
        out.append(found)
    return out


def save_html_pdf_story(html: str, filepath: str) -> str:
    """Render ARBITRARY self-contained HTML to a paginated PDF (fitz.Story).

    Public generic entry point (also exposed to LLM agents as the
    ``calc_package.html_to_pdf`` method) for report content that no canned
    ``*_package`` generator covers. Returns the absolute PDF path; see
    :func:`render_html_pdf_story` for what it does and for the figures'
    printed sizes.
    """
    return render_html_pdf_story(html, filepath)["path"]


def render_html_pdf_story(html: str, filepath: str) -> dict:
    """Render self-contained HTML to a paginated PDF (fitz.Story) and say
    where each figure printed.

    Three Story-only adjustments are applied: inline base64 PNG figures are
    re-encoded to JPEG (so they stay compressed in the PDF instead of
    bloating it), wide tables are wrapped to the page width (so they cannot
    clip at the right edge), and figures are laid out by the rule above
    (never shrunk to fit what is left of a page; BACKLOG GM-168). The Story
    engine supports a SUBSET of CSS: it lays out text, headings, tables,
    colours and inline base64 PNG/JPEG ``<img>`` figures, but ignores
    advanced CSS (flexbox/grid, positioned layout, some border/spacing
    rules) and does NOT render inline ``<svg>`` elements.

    Returns ``{"path": absolute PDF path, "figures": [...] or None}``:
    per ``<img>``, in document order, ``pdf_page`` (from 1), ``width_in``,
    ``height_in``, ``laid_out_width_in`` (its width on a page of its own),
    ``moved`` (to the next page, so as not to shrink), ``scaled`` (the
    share of its width it printed at, when under it: a figure taller than a
    page) and, when its PNG carries its lettering
    (``geotech_common.layout_ink.LETTERING_KEY``), ``lettering_pt`` (the
    smallest as printed) and ``checked_width_in`` (the width the layout
    check measured it at). ``figures`` is ``None`` when the pictures drawn
    cannot be matched to the HTML's ``<img>`` tags; then, when fewer
    printed than the HTML holds, ``figures_not_printed`` says how many did
    not (:func:`figures_not_printed_warning`).
    """
    try:
        import fitz  # PyMuPDF, installed via the [pdf] extra
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PyMuPDF (fitz) is not installed, so the pure-Python PDF fallback "
            "is unavailable. Install the [pdf] extra: "
            "pip install 'geotech-staff-engineer[pdf]'."
        ) from exc

    letterings = _png_letterings(html)
    html = _compress_pdf_images(html)
    html = _fit_wide_tables_for_pdf(html)
    if "</head>" in html:
        html = html.replace("</head>", _PDF_STORY_CSS + "</head>", 1)
    else:
        html = _PDF_STORY_CSS + html
    output_path = Path(filepath).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mediabox = fitz.paper_rect("letter")
    m = _STORY_MARGIN_PT
    where = mediabox + (m, m, -m, -m)
    pdf, placed = _lay_out_figures(html, mediabox, where)
    output_path.write_bytes(pdf)

    figures = None
    if placed is not None:
        figures = []
        for k, f in enumerate(placed):
            row = {"pdf_page": f["pdf_page"],
                   "width_in": round(f["width_pt"] / 72, 2),
                   "height_in": round(f["height_pt"] / 72, 2),
                   "laid_out_width_in": round(f["laid_out_width_pt"] / 72, 2),
                   "moved": f["moved"], "scaled": f["scaled"]}
            lt = letterings[k] if k < len(letterings) else {}
            if lt:
                row["lettering_pt"] = round(
                    lt["smallest_pt"] * f["width_pt"] / 72
                    / lt["saved_width_in"], 1)
                if lt.get("checked_width_in"):
                    row["checked_width_in"] = lt["checked_width_in"]
            figures.append(row)
    out = {"path": str(output_path), "figures": figures}
    if figures is None:
        try:
            short = len(_figure_marks(html)["spans"]) - len(_story_images(pdf))
        except Exception:                                  # noqa: BLE001
            short = 0
        if short > 0:
            out["figures_not_printed"] = short
    return out


def figures_not_printed_warning(rendered: dict) -> "str | None":
    """The warning for a Story PDF that left figures out
    (``rendered["figures_not_printed"]``, GM-198), else None."""
    short = rendered.get("figures_not_printed") if rendered else None
    if not short:
        return None
    return (f"{short} figure(s) in the HTML did not print in the PDF (the "
            "Story engine left them out where a page was full). Tell the "
            "user which are missing; do not describe the PDF as complete.")


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
        ``{"path": str, "renderer": "pdflatex"|"pymupdf_story", "warnings": [str]}``,
        plus, from the Story engine, ``"figures"``: each figure's printed
        size and lettering (see :func:`render_html_pdf_story`).

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
        rendered = _save_pdf_story(data, filepath)
    except Exception as exc:
        raise RuntimeError(
            "Could not produce a PDF: pdflatex is not installed and the "
            f"PyMuPDF Story fallback failed ({type(exc).__name__}: {exc}). "
            "Generate HTML output (format='html') and use your browser's "
            "Print -> Save as PDF, or install a LaTeX distribution "
            "(MiKTeX / TeX Live)."
        ) from exc
    missing = figures_not_printed_warning(rendered)
    if missing:
        warnings.append(missing)
    # each figure's printed size and lettering (GM-186), for the caller to
    # warn on one that prints too small
    return {"path": rendered["path"], "renderer": "pymupdf_story",
            "warnings": warnings, "figures": rendered.get("figures")}


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
