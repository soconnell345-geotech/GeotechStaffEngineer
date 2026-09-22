"""
Markdown -> Word (.docx) renderer.

The agent writes markdown fluently and reviewers want a Word file: something
they can track changes in, paste into a report template, and send on. This
turns one markdown string into one .docx — headings, runs of bold/italic/code,
bullet and numbered lists, pipe tables, images, fenced code, block quotes — so
a memo, a findings summary or a review response arrives as a document rather
than as chat text.

It is NOT a calculation package. `generate_calc_package()` renders a structured
`CalcPackageData` (inputs echoed, equations with substituted values, checks) as
self-contained HTML; this renders free prose the agent composed. Use the
package for a calculation, this for everything a reviewer reads.

Two rules shape the whole module:

- **Nothing here raises over content.** A construct the renderer does not know
  becomes a plain paragraph, and an image whose file is not there becomes a
  line of text saying so. A missing figure must not cost the reader the other
  nine pages, so every such case is appended to ``warnings`` and the document
  is still written.
- **python-docx is imported LAZILY**, inside the function. The module imports
  without it, which is what lets the agent tool feature-detect the package and
  hide itself rather than failing in front of the user.

A horizontal rule (``---``) becomes a PAGE BREAK. Word has no real "horizontal
rule" paragraph, and in a document meant for print a rule between sections is
almost always where the author wanted the next page to start; a reader who
wants a thin line can draw one, but nobody can recover a page break that was
never emitted.
"""

import os

#: Heading levels Word has styles for here. A deeper markdown heading is
#: rendered at this level rather than dropped, and noted in the warnings.
MAX_HEADING_LEVEL = 4

#: Font used for `inline code` and fenced blocks.
MONO_FONT = "Consolas"

#: Indent applied to a block quote, in inches.
QUOTE_INDENT_IN = 0.4


def markdown_to_docx(markdown: str, output_path: str, *,
                     base_dir: str = None, title: str = None,
                     warnings: list = None) -> str:
    """Render a markdown string as a Word document.

    Parameters
    ----------
    markdown : str
        The markdown source. CommonMark plus pipe tables.
    output_path : str
        Where to write the .docx. Parent directories are created.
    base_dir : str, optional
        Directory that relative image paths resolve against — normally the
        working folder the agent already saved its figures into, so
        ``![](profile.png)`` finds the PNG it just wrote.
    title : str, optional
        Rendered as a Title paragraph above everything else.
    warnings : list, optional
        If a list is passed it receives one line per thing that could not be
        rendered as asked (a missing image, a heading deeper than Word's
        styles, raw HTML). Nothing here raises over content, so this list is
        how a caller learns what the reader will not see.

    Returns
    -------
    str
        The absolute path of the file written.

    Raises
    ------
    ImportError
        If python-docx is not installed (the message says how to install it).
    """
    try:
        import docx
        from docx.enum.text import WD_BREAK
        from docx.shared import Inches
    except ImportError as exc:                       # pragma: no cover - env
        raise ImportError(
            "Word output needs the package python-docx: "
            "pip install python-docx"
        ) from exc
    from markdown_it import MarkdownIt

    notes = warnings if warnings is not None else []
    doc = docx.Document()
    section = doc.sections[0]
    text_width = section.page_width - section.left_margin - section.right_margin
    ctx = _Context(doc=doc, base_dir=base_dir, text_width=text_width,
                   warnings=notes, Inches=Inches, page_break=WD_BREAK.PAGE)

    if title:
        doc.add_heading(str(title), 0)

    tokens = MarkdownIt("commonmark").enable("table").parse(markdown or "")
    _render_blocks(tokens, ctx)

    abs_path = os.path.abspath(output_path)
    parent = os.path.dirname(abs_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    doc.save(abs_path)
    return abs_path


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

class _Context:
    """Everything the block and inline writers share for one document."""

    def __init__(self, doc, base_dir, text_width, warnings, Inches,
                 page_break):
        self.doc = doc
        self.base_dir = base_dir
        self.text_width = text_width
        self.warnings = warnings
        self.Inches = Inches
        self.page_break = page_break

    def style(self, paragraph, *names):
        """Apply the first of ``names`` this template actually defines.

        A .docx built from another template may not carry "List Bullet 2" or
        "Quote"; an unknown style name raises, so the fallbacks are tried in
        order and a paragraph that matches none keeps the default style (the
        caller has already applied the indent/italic that carry the meaning).
        """
        for name in names:
            try:
                paragraph.style = name
                return True
            except KeyError:
                continue
        return False


def _render_blocks(tokens, ctx):
    """Walk markdown-it's flat token stream and emit Word blocks."""
    lists = []              # 'bullet' / 'ordered', innermost last
    quote_depth = 0
    heading_level = None
    item_pending = False
    i = 0
    while i < len(tokens):
        t = tokens[i]
        kind = t.type

        if kind == "table_open":
            end = _matching(tokens, i, "table_open", "table_close")
            i = _table(tokens, i, end, ctx)
            continue

        if kind in ("bullet_list_open", "ordered_list_open"):
            lists.append("bullet" if kind[0] == "b" else "ordered")
        elif kind in ("bullet_list_close", "ordered_list_close"):
            if lists:
                lists.pop()
        elif kind == "list_item_open":
            item_pending = True
        elif kind == "list_item_close":
            item_pending = False
        elif kind == "blockquote_open":
            quote_depth += 1
        elif kind == "blockquote_close":
            quote_depth = max(0, quote_depth - 1)
        elif kind == "heading_open":
            heading_level = _heading_level(t.tag, ctx)
        elif kind == "heading_close":
            heading_level = None
        elif kind == "hr":
            # Documented in the module docstring: a rule is a page break.
            ctx.doc.add_paragraph().add_run().add_break(ctx.page_break)
        elif kind in ("fence", "code_block"):
            _code_block(t.content, ctx)
        elif kind in ("html_block",):
            ctx.warnings.append(
                "raw HTML was written as plain text: "
                f"{' '.join(t.content.split())[:60]}")
            ctx.doc.add_paragraph(t.content.strip())
        elif kind == "inline":
            if heading_level is not None:
                par = ctx.doc.add_heading("", level=heading_level)
                _add_inline(par, t.children, ctx)
            elif lists:
                par = ctx.doc.add_paragraph()
                _list_style(par, lists, item_pending, ctx)
                _add_inline(par, t.children, ctx)
                item_pending = False
            elif quote_depth:
                par = ctx.doc.add_paragraph()
                ctx.style(par, "Quote", "Intense Quote")
                par.paragraph_format.left_indent = ctx.Inches(
                    QUOTE_INDENT_IN * quote_depth)
                _add_inline(par, t.children, ctx, italic=True)
            else:
                _paragraph(t.children, ctx)
        i += 1


def _heading_level(tag, ctx):
    """``h3`` -> 3, capped at what Word styles here go to."""
    try:
        level = int(str(tag)[1:])
    except ValueError:                               # pragma: no cover
        return 1
    if level > MAX_HEADING_LEVEL:
        ctx.warnings.append(
            f"heading level {level} was rendered as level {MAX_HEADING_LEVEL}")
        return MAX_HEADING_LEVEL
    return max(1, level)


def _list_style(par, lists, item_pending, ctx):
    """One level of nesting, as Word's own list styles spell it."""
    depth = min(len(lists), 2)
    base = "List Bullet" if lists[-1] == "bullet" else "List Number"
    name = base if depth == 1 else f"{base} {depth}"
    if not ctx.style(par, name, base):
        par.paragraph_format.left_indent = ctx.Inches(0.25 * depth)
    if not item_pending:
        # A second paragraph inside one list item: keep it under the bullet
        # rather than starting a new one.
        ctx.style(par, "Body Text", "Normal")
        par.paragraph_format.left_indent = ctx.Inches(0.25 * depth + 0.25)


def _paragraph(children, ctx):
    """A body paragraph — or, when it holds nothing but images, the images."""
    images = [c for c in children if c.type == "image"]
    other = [c for c in children
             if c.type not in ("image", "softbreak", "hardbreak")
             and (c.type != "text" or c.content.strip())]
    if images and not other:
        for child in images:
            _picture(ctx.doc.add_paragraph(), child, ctx, block=True)
        return
    par = ctx.doc.add_paragraph()
    _add_inline(par, children, ctx)


def _code_block(content, ctx):
    """A fenced block: one paragraph, monospaced, line breaks kept."""
    par = ctx.doc.add_paragraph()
    ctx.style(par, "No Spacing", "Normal")
    par.paragraph_format.left_indent = ctx.Inches(0.25)
    lines = (content or "").rstrip("\n").split("\n")
    run = par.add_run()
    run.font.name = MONO_FONT
    for n, line in enumerate(lines):
        if n:
            run.add_break()
        run.add_text(line)


def _add_inline(par, children, ctx, bold=False, italic=False):
    """Runs for one inline token stream: bold, italic, code, images, links."""
    href = ""
    for child in children or ():
        kind = child.type
        if kind == "strong_open":
            bold = True
        elif kind == "strong_close":
            bold = False
        elif kind == "em_open":
            italic = True
        elif kind == "em_close":
            italic = False
        elif kind == "code_inline":
            run = par.add_run(child.content)
            run.font.name = MONO_FONT
        elif kind == "text":
            run = par.add_run(child.content)
            run.bold, run.italic = bold, italic
        elif kind == "softbreak":
            par.add_run(" ")
        elif kind == "hardbreak":
            par.add_run().add_break()
        elif kind == "image":
            _picture(par, child, ctx)
        elif kind == "link_open":
            href = dict(child.attrs or {}).get("href", "")
        elif kind == "link_close":
            # Word gets the words plus the address in brackets. python-docx
            # writes no real hyperlink without hand-built XML, and a link
            # rendered as its words alone leaves the reader no way to follow
            # it at all.
            if href and href not in par.text:
                par.add_run(f" [{href}]").italic = True
            href = ""
        elif kind == "html_inline":
            ctx.warnings.append(f"inline HTML kept as text: {child.content}")
            par.add_run(child.content)
        elif child.children:
            _add_inline(par, child.children, ctx, bold, italic)


def _picture(par, token, ctx, block=False):
    """Place an image, scaled down to the text width, or say why it is not there."""
    attrs = dict(token.attrs or {})
    src = attrs.get("src", "")
    alt = (token.content or attrs.get("alt") or "").strip()
    path = src
    if src.startswith(("http://", "https://")):
        ctx.warnings.append(f"image not embedded (remote URL): {src}")
        par.add_run(f"[image: {alt or src}]").italic = True
        return
    if not os.path.isabs(path) and ctx.base_dir:
        path = os.path.join(ctx.base_dir, path)
    try:
        shape = par.add_run().add_picture(path)
    except Exception as exc:
        ctx.warnings.append(
            f"image not found or unreadable, left as a note: {src} "
            f"({type(exc).__name__})")
        par.add_run(f"[image not found: {src}]").italic = True
        return
    if shape.width > ctx.text_width:
        shape.height = int(shape.height * ctx.text_width / shape.width)
        shape.width = ctx.text_width
    if block and alt:
        caption = ctx.doc.add_paragraph()
        ctx.style(caption, "Caption")
        run = caption.add_run(alt)
        run.italic = True


def _matching(tokens, start, open_type, close_type):
    """Index of the token closing the block that opens at ``start``."""
    depth = 0
    for i in range(start, len(tokens)):
        if tokens[i].type == open_type:
            depth += 1
        elif tokens[i].type == close_type:
            depth -= 1
            if depth == 0:
                return i
    return len(tokens) - 1                           # pragma: no cover


def _table(tokens, start, end, ctx):
    """Build one table, and take the italic line under it as its caption.

    Returns the index to continue from — past the table, and past the caption
    when one was consumed.
    """
    rows = []
    header = None
    in_head = False
    row = None
    for i in range(start, end + 1):
        t = tokens[i]
        if t.type == "thead_open":
            in_head = True
        elif t.type == "thead_close":
            in_head = False
        elif t.type == "tr_open":
            row = []
        elif t.type == "tr_close":
            if in_head and header is None:
                header = row
            else:
                rows.append(row)
            row = None
        elif t.type == "inline" and row is not None:
            # One inline token per cell, in column order — th_open / td_open
            # carry only the alignment, which Word takes from the style.
            row.append(t)
    grid = ([header] if header else []) + rows
    if not grid:
        return end + 1
    n_cols = max(len(r) for r in grid)
    table = ctx.doc.add_table(rows=len(grid), cols=n_cols)
    ctx.style(table, "Table Grid")
    for r, cells in enumerate(grid):
        for c in range(n_cols):
            cell = table.cell(r, c)
            par = cell.paragraphs[0]
            if c < len(cells):
                _add_inline(par, cells[c].children, ctx,
                            bold=bool(header) and r == 0)
    nxt = end + 1
    caption = _caption_after(tokens, nxt)
    if caption is not None:
        par = ctx.doc.add_paragraph()
        ctx.style(par, "Caption")
        _add_inline(par, caption.children, ctx, italic=True)
        return nxt + 3                               # open, inline, close
    return nxt


def _caption_after(tokens, index):
    """The inline token of a wholly-italic paragraph at ``index``, or None.

    A table's caption is written under it in italics, which is how a markdown
    author says "this line names the table above" — the only way they can say
    it, since markdown has no caption of its own.
    """
    if index + 2 >= len(tokens):
        return None
    if (tokens[index].type != "paragraph_open"
            or tokens[index + 1].type != "inline"
            or tokens[index + 2].type != "paragraph_close"):
        return None
    children = tokens[index + 1].children or []
    if len(children) < 3 or children[0].type != "em_open" \
            or children[-1].type != "em_close":
        return None
    return tokens[index + 1]
