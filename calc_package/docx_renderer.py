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

A single line break inside a paragraph is KEPT as a line break (the GitHub
"breaks" reading models write in): ``**To:** file`` / ``**Re:** memo`` on two
lines stays two lines, where CommonMark would run them together (live smoke
wave 2a, B4). ``<br>`` is a line break too, the only way to put one in a
table cell.

:func:`docx_to_markdown` is the inverse: it reads a Word document back as
the same Markdown dialect, so "revise the memo" can read it, edit it and
write it again without the headings, lists, tables and line breaks drifting.
"""

import datetime
import os
import re

#: Heading levels Word has styles for here. A deeper markdown heading is
#: rendered at this level rather than dropped, and noted in the warnings.
MAX_HEADING_LEVEL = 4

#: Font used for `inline code` and fenced blocks.
MONO_FONT = "Consolas"

#: Indent applied to a block quote, in inches.
QUOTE_INDENT_IN = 0.4

#: Author written into File > Info when the caller names nobody (python-docx's
#: template otherwise says "python-docx").
DEFAULT_AUTHOR = "GeotechStaffEngineer"

_BR_TAG = re.compile(r"^<br\s*/?>$", re.IGNORECASE)


def markdown_to_docx(markdown: str, output_path: str, *,
                     base_dir: str = None, title: str = None,
                     warnings: list = None, author: str = None) -> str:
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
        Rendered ONCE as a Title paragraph above everything else: an opening
        heading with exactly its text (case and whitespace aside) is not
        printed again beneath it; any other opening heading is kept.
    warnings : list, optional
        If a list is passed it receives one line per thing that could not be
        rendered as asked (a missing image, a heading deeper than Word's
        styles, raw HTML). Nothing here raises over content, so this list is
        how a caller learns what the reader will not see.
    author : str, optional
        Written to the document's properties (File > Info) as its author and
        last editor; :data:`DEFAULT_AUTHOR` when not given. The title there
        is ``title``, else the first heading.

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

    tokens = MarkdownIt("commonmark").enable("table").parse(markdown or "")
    if title:
        doc.add_heading(str(title), 0)
        tokens = _drop_restated_title(tokens, str(title))
    _render_blocks(tokens, ctx)
    _set_properties(doc, title or _first_heading(tokens), author)

    abs_path = os.path.abspath(output_path)
    parent = os.path.dirname(abs_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    doc.save(abs_path)
    return abs_path


def _plain_inline(token) -> str:
    """The words of one inline token, formatting dropped."""
    out = []
    for child in token.children or ():
        if child.type in ("text", "code_inline"):
            out.append(child.content)
        elif child.type in ("softbreak", "hardbreak"):
            out.append(" ")
        elif child.children:
            out.append(_plain_inline(child))
    return "".join(out)


def _normal(text: str) -> str:
    """Text as compared with the title: case folded, runs of whitespace one
    space, ends trimmed."""
    return " ".join(str(text or "").split()).casefold()


def _restates(heading: str, title: str) -> bool:
    """Whether an opening heading restates the title: exactly the title's
    text, case and whitespace aside. Anything else is kept, so no word of
    it is lost (live smoke 2c, E8: F44's heading "Review: Std. No. 21.01,
    Rev. 2 - Bioretention Cross-Section (BMP Fig. 4.1.3)" began with the
    title's words, was taken for the title and was dropped with its figure
    number)."""
    h = _normal(heading)
    return bool(h) and h == _normal(title)


def _drop_restated_title(tokens, title):
    """``tokens`` without an opening heading that restates ``title`` (live
    smoke wave 2a, B4: the title printed twice, as Title and as Heading 1;
    see :func:`_restates` for what counts)."""
    if len(tokens) >= 3 and tokens[0].type == "heading_open" \
            and tokens[1].type == "inline" \
            and tokens[2].type == "heading_close" \
            and _restates(_plain_inline(tokens[1]), title):
        return tokens[3:]
    return tokens


def _first_heading(tokens):
    for i, t in enumerate(tokens[:-1]):
        if t.type == "heading_open" and tokens[i + 1].type == "inline":
            return _plain_inline(tokens[i + 1]).strip() or None
    return None


def _set_properties(doc, title, author):
    """File > Info: the title and the author, never python-docx's own."""
    cp = doc.core_properties
    who = str(author or "").strip() or DEFAULT_AUTHOR
    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None,
                                                               microsecond=0)
    try:
        cp.author = who
        cp.last_modified_by = who
        cp.title = str(title or "").strip()
        cp.created = now
        cp.modified = now
        cp.revision = 1
    except Exception:  # noqa: BLE001 - properties never cost the document
        pass


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
            lists.append(_ListLevel("bullet" if kind[0] == "b" else "ordered",
                                    _list_start(t)))
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


class _ListLevel:
    """One open markdown list: bullet or ordered, the number it starts at,
    and (ordered) the Word numbering made for it."""

    def __init__(self, kind, start=1):
        self.kind = kind
        self.start = start
        self.num = None                 # (numId, ilvl) once made


def _list_start(token) -> int:
    try:
        return max(1, int(dict(token.attrs or {}).get("start", 1)))
    except (TypeError, ValueError):
        return 1


def _restart_numbering(par, ctx):
    """A fresh Word numbering for one ordered list, starting at its first
    number — else every numbered list in the document continues the count
    of the one before (Word's "List Number" style shares one sequence).
    ``(numId, ilvl)``, or ``None`` when the template has no such numbering
    (the style's own then applies)."""
    try:
        numPr = par.style.element.pPr.numPr
        ilvl = numPr.ilvl.val if numPr.ilvl is not None else 0
        numbering = ctx.doc.part.numbering_part.numbering_definitions._numbering
        abstract = numbering.num_having_numId(numPr.numId.val).abstractNumId.val
        return numbering.add_num(abstract).numId, ilvl
    except Exception:  # noqa: BLE001 - numbering is a nicety, never a failure
        return None


def _apply_numbering(par, level, ctx):
    if level.num is None:
        made = _restart_numbering(par, ctx)
        if made is None:
            level.num = False
            return
        level.num = made
        try:
            numbering = ctx.doc.part.numbering_part.numbering_definitions._numbering
            num = numbering.num_having_numId(made[0])
            num.add_lvlOverride(ilvl=made[1]).add_startOverride(level.start)
        except Exception:  # noqa: BLE001
            pass
    if not level.num:
        return
    num_id, ilvl = level.num
    try:
        numPr = par._p.get_or_add_pPr().get_or_add_numPr()
        numPr.get_or_add_ilvl().val = ilvl
        numPr.get_or_add_numId().val = num_id
    except Exception:  # noqa: BLE001
        pass


def _list_style(par, lists, item_pending, ctx):
    """One level of nesting, as Word's own list styles spell it."""
    depth = min(len(lists), 2)
    level = lists[-1]
    base = "List Bullet" if level.kind == "bullet" else "List Number"
    name = base if depth == 1 else f"{base} {depth}"
    if not ctx.style(par, name, base):
        par.paragraph_format.left_indent = ctx.Inches(0.25 * depth)
    elif item_pending and level.kind == "ordered":
        _apply_numbering(par, level, ctx)
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
        elif kind in ("softbreak", "hardbreak"):
            # A line break the author typed is a line break in Word too
            # (B4: "June 2026" and "Status: DRAFT" ran into one line).
            par.add_run().add_break()
        elif kind == "html_inline" and _BR_TAG.match(child.content.strip()):
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


# ---------------------------------------------------------------------------
# Reading: Word -> the same Markdown (live smoke wave 2a, B3)
# ---------------------------------------------------------------------------
#
# A memo the agent wrote, or one the user uploaded, comes back as the dialect
# markdown_to_docx writes: headings, list types and levels, pipe tables with
# their captions, bold/italic/code runs, line breaks, block quotes, fenced
# code, page breaks as ``---`` and pictures (saved beside, referenced by
# name), so read -> edit -> write keeps the document's shape. Before this the
# only route was planlens rendering the .docx to pages, which lost tables,
# list markers and line breaks, and every revision drifted.

#: Fonts a run is read as ``code`` in.
_MONO_FONTS = {"consolas", "courier new", "courier", "lucida console",
               "menlo", "monaco", "source code pro", "cascadia mono",
               "cascadia code"}

_VML = "{urn:schemas-microsoft-com:vml}"

#: A line that would start a Markdown block (list, heading, quote, fence,
#: setext underline) when it is only text in Word.
_BLOCK_START = re.compile(
    r"^(?:[-+*](?=\s|$)|\d{1,9}[.)](?=\s|$)|#{1,6}(?=\s|$)|>|```|~~~"
    r"|=+\s*$)")


def docx_to_markdown(source, *, image_dir: str = None, image_ref: str = None,
                     image_prefix: str = "") -> dict:
    """Read a Word document as the Markdown :func:`markdown_to_docx` writes.

    Parameters
    ----------
    source : str or file-like
        The .docx.
    image_dir : str, optional
        Folder the document's pictures are saved into (``<prefix>figN.<ext>``)
        so the Markdown can reference them and a rewrite embeds them again.
        Without it pictures are named in the text but not extracted.
    image_ref : str, optional
        How the Markdown names ``image_dir`` (e.g. ``.scratch``, relative to
        the folder write_docx resolves images in); default ``image_dir``.
    image_prefix : str
        Prefix of the saved pictures' names.

    Returns
    -------
    dict
        ``markdown``; ``title`` (the Title paragraph's text, or ``None``);
        ``images`` (the names the pictures were saved under, or a
        ``picture N`` placeholder each); ``properties`` (title, author).
    """
    import docx

    doc = docx.Document(source)
    reader = _DocxReader(doc, image_dir, image_ref, image_prefix)
    reader.read_container(doc.element.body)
    cp = doc.core_properties
    return {
        "markdown": _assemble(reader.blocks),
        "title": reader.title,
        "images": reader.images,
        "properties": {"title": cp.title or "", "author": cp.author or ""},
    }


def _qn(tag):
    from docx.oxml.ns import qn
    return qn(tag)


def _children(el, tag):
    """``el``'s children with ``tag``, looking through content controls."""
    wrap = (_qn("w:sdt"), _qn("w:sdtContent"), _qn("w:customXml"))
    for child in el.iterchildren():
        if child.tag == tag:
            yield child
        elif child.tag in wrap:
            yield from _children(child, tag)


def _escape_inline(text: str) -> str:
    return text.replace("*", r"\*").replace("`", r"\`")


def _escape_block_start(line: str) -> str:
    """``line`` with a leading list/heading/quote marker escaped, so text
    that only looks like Markdown stays text when it is written again."""
    line = line.lstrip()
    if not _BLOCK_START.match(line):
        return line
    d = re.match(r"\d{1,9}", line)
    if d:
        return line[:d.end()] + "\\" + line[d.end():]
    return "\\" + line


def _fmt_text(seg, drop_bold, drop_italic) -> str:
    s = seg["s"]
    lead = s[:len(s) - len(s.lstrip())]
    trail = s[len(s.rstrip()):]
    core = s.strip()
    if not core:
        return s
    if seg["c"]:
        tick = "``" if "`" in core else "`"
        pad = " " if core[0] == "`" or core[-1] == "`" else ""
        return f"{lead}{tick}{pad}{core}{pad}{tick}{trail}"
    core = _escape_inline(core)
    b = seg["b"] and not drop_bold
    i = seg["i"] and not drop_italic
    if not (b or i):
        return lead + core + trail
    mark = "***" if b and i else "**" if b else "*"
    return f"{lead}{mark}{core}{mark}{trail}"


def _segments_md(segs, drop_bold=False, drop_italic=False,
                 heading=False) -> str:
    """Markdown for a paragraph's pieces (page breaks already taken out)."""
    merged = []
    for s in segs:
        if s["t"] == "text" and merged and merged[-1]["t"] == "text" and all(
                merged[-1][k] == s[k] for k in ("b", "i", "c", "link")):
            merged[-1] = dict(merged[-1], s=merged[-1]["s"] + s["s"])
        else:
            merged.append(dict(s))
    out = []
    k = 0
    while k < len(merged):
        s = merged[k]
        if s["t"] == "text" and s["link"]:
            url, inner = s["link"], []
            while k < len(merged) and merged[k]["t"] == "text" \
                    and merged[k]["link"] == url:
                inner.append(_fmt_text(merged[k], drop_bold, drop_italic))
                k += 1
            words = "".join(inner).strip()
            target = f"<{url}>" if re.search(r"\s", url) else url
            out.append(f"[{words}]({target})" if words else "")
            continue
        if s["t"] == "text":
            out.append(_fmt_text(s, drop_bold, drop_italic))
        elif s["t"] == "br":
            out.append(" " if heading else "\n")
        elif s["t"] == "img":
            out.append(s["md"])
        k += 1
    return "".join(out)


class _DocxReader:
    """Walks one document body into Markdown blocks."""

    def __init__(self, doc, image_dir, image_ref, image_prefix):
        self.doc = doc
        self.body = doc._body
        self.image_dir = image_dir
        self.image_ref = image_ref if image_ref is not None else image_dir
        self.prefix = image_prefix or ""
        self.blocks = []             # (kind, text, extra)
        self.title = None
        self.images = []
        self._pending_caption_for = None   # index of an image block

    # -- containers ------------------------------------------------------
    def read_container(self, el):
        p_tag, tbl_tag = _qn("w:p"), _qn("w:tbl")
        wrap = (_qn("w:sdt"), _qn("w:sdtContent"), _qn("w:customXml"))
        for child in el.iterchildren():
            if child.tag == p_tag:
                self.paragraph(child)
            elif child.tag == tbl_tag:
                self.table(child)
            elif child.tag in wrap:
                self.read_container(child)

    # -- runs ------------------------------------------------------------
    def _is_mono(self, run) -> bool:
        try:
            name = run.font.name
            if not name and run.style is not None:
                name = run.style.font.name
                style = (run.style.name or "").lower()
                if style in ("html code", "code", "verbatim char",
                             "source code"):
                    return True
        except Exception:  # noqa: BLE001 - formatting is best effort
            return False
        return bool(name) and name.strip().lower() in _MONO_FONTS

    def _link_url(self, el):
        rid = el.get(_qn("r:id"))
        if not rid:
            return None
        try:
            rel = self.doc.part.rels[rid]
            return rel.target_ref if rel.is_external else None
        except Exception:  # noqa: BLE001
            return None

    def _image(self, el):
        blip = el.find(".//" + _qn("a:blip"))
        rid = blip.get(_qn("r:embed")) if blip is not None else None
        if rid is None:
            imd = el.find(".//" + _VML + "imagedata")
            rid = imd.get(_qn("r:id")) if imd is not None else None
        pr = el.find(".//" + _qn("wp:docPr"))
        alt = ((pr.get("descr") or pr.get("title") or "")
               if pr is not None else "").strip()
        n = len(self.images) + 1
        name = None
        if rid and self.image_dir:
            try:
                part = self.doc.part.related_parts[rid]
                ext = os.path.splitext(str(part.partname))[1] or ".png"
                fname = f"{self.prefix}fig{n}{ext}"
                os.makedirs(self.image_dir, exist_ok=True)
                with open(os.path.join(self.image_dir, fname), "wb") as fh:
                    fh.write(part.blob)
                ref = str(self.image_ref or "").rstrip("/\\")
                name = f"{ref}/{fname}" if ref else fname
            except Exception:  # noqa: BLE001 - a picture never costs the text
                name = None
        self.images.append(name or f"picture {n}")
        if name:
            md = f"![{alt}]({name})"
        else:
            md = f"[picture {n}{': ' + alt if alt else ''} — not extracted]"
        return {"t": "img", "md": md, "name": name, "alt": alt}

    def segments(self, p_el):
        """The pieces of one paragraph: text runs with their formatting,
        line and page breaks, pictures — tracked insertions in, deletions
        out, field codes out."""
        from docx.text.paragraph import Paragraph
        from docx.text.run import Run
        par = Paragraph(p_el, self.body)
        segs = []
        r_tag, link_tag = _qn("w:r"), _qn("w:hyperlink")
        skip = (_qn("w:del"), _qn("w:moveFrom"))
        into = (_qn("w:ins"), _qn("w:moveTo"), _qn("w:smartTag"),
                _qn("w:customXml"), _qn("w:fldSimple"), _qn("w:sdt"),
                _qn("w:sdtContent"))
        t_tag, tab_tag, br_tag, cr_tag = (_qn("w:t"), _qn("w:tab"),
                                          _qn("w:br"), _qn("w:cr"))
        pics = (_qn("w:drawing"), _qn("w:pict"))
        hyphen = _qn("w:noBreakHyphen")

        def run(r_el, link):
            r = Run(r_el, par)
            try:
                b, i = bool(r.bold), bool(r.italic)
            except Exception:  # noqa: BLE001
                b = i = False
            c = self._is_mono(r)
            for ch in r_el.iterchildren():
                if ch.tag == t_tag:
                    segs.append({"t": "text", "s": ch.text or "", "b": b,
                                 "i": i, "c": c, "link": link})
                elif ch.tag == tab_tag:
                    segs.append({"t": "text", "s": " ", "b": b, "i": i,
                                 "c": c, "link": link})
                elif ch.tag == hyphen:
                    segs.append({"t": "text", "s": "-", "b": b, "i": i,
                                 "c": c, "link": link})
                elif ch.tag == br_tag:
                    segs.append({"t": "page"} if ch.get(_qn("w:type")) ==
                                "page" else {"t": "br"})
                elif ch.tag == cr_tag:
                    segs.append({"t": "br"})
                elif ch.tag in pics:
                    segs.append(self._image(ch))

        def walk(el, link):
            for ch in el.iterchildren():
                if ch.tag == r_tag:
                    run(ch, link)
                elif ch.tag == link_tag:
                    walk(ch, self._link_url(ch) or link)
                elif ch.tag in skip:
                    continue
                elif ch.tag in into:
                    walk(ch, link)

        walk(p_el, None)
        return segs, par

    def inline(self, p_el, drop_bold=False) -> str:
        """One paragraph's text as inline Markdown (a table cell)."""
        segs, _par = self.segments(p_el)
        segs = [s for s in segs if s["t"] != "page"]
        return _segments_md(segs, drop_bold=drop_bold).strip()

    # -- paragraphs ------------------------------------------------------
    def _style_name(self, par) -> str:
        try:
            return par.style.name or "Normal"
        except Exception:  # noqa: BLE001
            return "Normal"

    def _list_info(self, p_el, par, style):
        m = re.match(r"^List (Bullet|Number)(?: (\d))?$", style)
        if m:
            return int(m.group(2) or 1) - 1, m.group(1) == "Number"
        num_pr = None
        try:
            ppr = p_el.pPr
            num_pr = ppr.numPr if ppr is not None else None
            if num_pr is None:
                st = par.style
                while st is not None and num_pr is None:
                    sp = st.element.pPr
                    num_pr = sp.numPr if sp is not None else None
                    st = st.base_style
        except Exception:  # noqa: BLE001
            num_pr = None
        if num_pr is None or num_pr.numId is None or not num_pr.numId.val:
            return None
        ilvl = num_pr.ilvl.val if num_pr.ilvl is not None else 0
        return int(ilvl), self._ordered(num_pr.numId.val, ilvl)

    def _ordered(self, num_id, ilvl) -> bool:
        try:
            numbering = self.doc.part.numbering_part.element
            abstract = numbering.num_having_numId(num_id).abstractNumId.val
            for an in numbering.findall(_qn("w:abstractNum")):
                if an.get(_qn("w:abstractNumId")) != str(abstract):
                    continue
                for lvl in an.findall(_qn("w:lvl")):
                    if lvl.get(_qn("w:ilvl")) == str(ilvl):
                        fmt = lvl.find(_qn("w:numFmt"))
                        val = fmt.get(_qn("w:val")) if fmt is not None \
                            else "decimal"
                        return val not in ("bullet", "none")
        except Exception:  # noqa: BLE001
            pass
        return False

    def _add(self, kind, text, extra=None):
        self.blocks.append((kind, text, extra))

    def paragraph(self, p_el):
        segs, par = self.segments(p_el)
        style = self._style_name(par)
        low = style.lower()
        pages = [k for k, s in enumerate(segs) if s["t"] == "page"]
        first_text = next((k for k, s in enumerate(segs)
                           if s["t"] in ("text", "img")
                           and (s["t"] == "img" or s["s"].strip())), None)
        break_before = bool(pages) and (first_text is None
                                        or pages[0] < first_text)
        break_after = any(first_text is not None and k > first_text
                          for k in pages)
        segs = [s for s in segs if s["t"] != "page"]
        if break_before:
            self._add("hr", "---")
        self._paragraph_body(p_el, par, segs, style, low)
        if break_after:
            self._add("hr", "---")

    def _paragraph_body(self, p_el, par, segs, style, low):
        texts = [s for s in segs if s["t"] == "text"]
        imgs = [s for s in segs if s["t"] == "img"]
        has_text = any(s["s"].strip() for s in texts)
        if not has_text and not imgs:
            return
        caption_for = self._pending_caption_for
        self._pending_caption_for = None

        if low == "caption" and has_text:
            words = _segments_md(segs, drop_italic=True).strip()
            if caption_for is not None:
                kind, _md, img = self.blocks[caption_for]
                if img and img.get("name"):
                    self.blocks[caption_for] = (
                        "img", f"![{words}]({img['name']})", img)
                    return
            self._add("p", f"*{words}*")
            return
        if imgs and not has_text:
            for s in imgs:
                self._add("img", s["md"], s)
            self._pending_caption_for = len(self.blocks) - 1
            return

        m = re.match(r"^heading (\d)$", low)
        if low == "title" or m:
            text = _segments_md(segs, drop_bold=True, heading=True).strip()
            if not text:
                return
            if low == "title":
                if self.title is None:
                    self.title = " ".join(
                        "".join(s["s"] for s in texts).split())
                level = 1
            else:
                level = min(6, max(1, int(m.group(1))))
            self._add("h", "#" * level + " " + text)
            return
        if low == "subtitle":
            self._add("p", "*" + _segments_md(segs, drop_italic=True).strip()
                      + "*")
            return

        # A code BLOCK is monospaced throughout and set apart (the "No
        # Spacing" paragraph a fence is written as, a code style, or more
        # than one line); one line of code in a body paragraph is `inline`.
        mono = texts and all(s["c"] for s in texts if s["s"].strip())
        block_style = (low in ("no spacing", "plain text", "html preformatted")
                       or "code" in low
                       or any(s["t"] == "br" for s in segs))
        if mono and block_style and not imgs:
            code = "".join("\n" if s["t"] == "br" else s.get("s", "")
                           for s in segs if s["t"] in ("text", "br"))
            self._add("code", code.rstrip("\n"))
            return

        info = self._list_info(p_el, par, style)
        if info is not None:
            level, ordered = info
            text = _segments_md(segs).strip()
            self._add("li", text, (level, ordered))
            return
        if (low.startswith("body text") or low.startswith("list continue")) \
                and self.blocks and self.blocks[-1][0] in ("li", "li_cont"):
            self._add("li_cont", _segments_md(segs).strip())
            return
        if low in ("quote", "intense quote"):
            text = _segments_md(segs, drop_italic=True).strip()
            self._add("quote", text)
            return
        text = _segments_md(segs).strip("\n").strip()
        lines = [_escape_block_start(ln) for ln in text.split("\n")]
        self._add("p", "\n".join(ln for ln in lines))

    # -- tables ----------------------------------------------------------
    def table(self, tbl_el):
        self._pending_caption_for = None
        rows = []
        for tr in _children(tbl_el, _qn("w:tr")):
            cells = []
            for tc in _children(tr, _qn("w:tc")):
                span = 1
                tcpr = tc.tcPr if hasattr(tc, "tcPr") else None
                try:
                    if tcpr is not None and tcpr.gridSpan is not None:
                        span = max(1, int(tcpr.gridSpan))
                except (TypeError, ValueError, AttributeError):
                    span = 1
                header = not rows
                parts = [self.inline(p, drop_bold=header)
                         for p in _children(tc, _qn("w:p"))]
                cell = "<br>".join(p for p in parts if p)
                cell = cell.replace("\n", "<br>").replace("|", r"\|")
                cells.append(cell)
                cells.extend([""] * (span - 1))
            rows.append(cells)
        rows = [r for r in rows if r]
        if not rows:
            return
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]
        lines = ["| " + " | ".join(rows[0]) + " |",
                 "| " + " | ".join(["---"] * width) + " |"]
        lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
        self._add("table", "\n".join(lines))


def _assemble(blocks) -> str:
    """Join the blocks: a blank line between blocks, list items of one list
    on consecutive lines, numbered and indented by level, consecutive code
    paragraphs in one fence."""
    out = []
    stack = []                       # per open list level: [ordered, n, width]
    indent_of_last = 0
    prev = None
    i = 0
    while i < len(blocks):
        kind, text, extra = blocks[i]
        if kind == "code":
            lines = [text]
            while i + 1 < len(blocks) and blocks[i + 1][0] == "code":
                i += 1
                lines.append(blocks[i][1])
            body = "\n".join(lines)
            fence = "````" if "```" in body else "```"
            piece = f"{fence}\n{body}\n{fence}"
        elif kind == "li":
            level, ordered = extra
            level = max(0, min(int(level), len(stack)))
            del stack[level + 1:]
            if len(stack) == level:
                stack.append([ordered, 0, 0])
            elif stack[level][0] != ordered:
                stack[level] = [ordered, 0, 0]
            stack[level][1] += 1
            marker = f"{stack[level][1]}." if ordered else "-"
            stack[level][2] = len(marker) + 1
            indent = sum(s[2] for s in stack[:level])
            pad = " " * (indent + stack[level][2])
            lines = (text or "").split("\n")
            piece = " " * indent + marker + " " + lines[0] + "".join(
                "\n" + pad + ln for ln in lines[1:])
            indent_of_last = indent + stack[level][2]
        elif kind == "li_cont":
            pad = " " * indent_of_last
            piece = "\n".join(pad + ln for ln in text.split("\n"))
        elif kind == "quote":
            piece = "\n".join("> " + ln for ln in text.split("\n"))
        else:
            piece = text
        if kind not in ("li", "li_cont"):
            stack = []
        if out:
            out.append("\n" if kind == "li" and prev == "li" else "\n\n")
        out.append(piece)
        prev = kind
        i += 1
    return "".join(out).strip("\n") + "\n" if out else ""
