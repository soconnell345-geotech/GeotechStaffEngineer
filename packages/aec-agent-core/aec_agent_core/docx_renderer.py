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

#: A photograph is stored in the .docx as JPEG at this quality (DR-47) ...
PHOTO_JPEG_QUALITY = 85
#: ... with no more pixels than this many per inch of the size it is
#: printed at (it is never printed wider than the text): a 600 x 400 px
#: photograph printed 2 in wide is stored at 400 x 267 px.
PHOTO_MAX_DPI = 200

# DEFAULT_AUTHOR: the author written into File > Info when the caller names
# nobody (python-docx's template otherwise says "python-docx"): the name of
# the application the page handed in (aec_agent_core.app_details; INF-33
# step 13), read at each use through the module's __getattr__ below.


def __getattr__(attr):
    if attr == "DEFAULT_AUTHOR":
        from aec_agent_core.app_details import name
        return name()
    raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")

_BR_TAG = re.compile(r"^<br\s*/?>$", re.IGNORECASE)
#: ``<sup>`` / ``<sub>`` and their closers, as inline HTML (DR-41).
_SCRIPT_TAG = re.compile(r"^<(/?)(sup|sub)\s*>$", re.IGNORECASE)
#: Characters XML 1.0 does not allow (tab, line feed and carriage return are).
_CONTROL_CHARS = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")


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

    # A form feed or another control character (text copied out of a PDF
    # carries them) is not allowed in a Word file: python-docx raises on it
    # and the whole document was lost.
    markdown = _CONTROL_CHARS.sub(" ", markdown or "")
    if title:
        title = _CONTROL_CHARS.sub(" ", str(title))

    notes = warnings if warnings is not None else []
    doc = docx.Document()
    section = doc.sections[0]
    text_width = section.page_width - section.left_margin - section.right_margin
    text_height = (section.page_height - section.top_margin
                   - section.bottom_margin)
    ctx = _Context(doc=doc, base_dir=base_dir, text_width=text_width,
                   text_height=text_height, warnings=notes, Inches=Inches, page_break=WD_BREAK.PAGE)

    tokens = MarkdownIt("commonmark").enable("table").enable(
        "strikethrough").parse(markdown or "")
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


def _words(text: str) -> list:
    """The words of ``text`` as compared with the title: runs of letters and
    digits, case folded. Punctuation (dashes, colons, commas, brackets,
    dots) and whitespace only separate words, so "A - B" and "A: B" have
    the same words."""
    return re.findall(r"\w+", str(text or "").casefold())


def _restates(heading: str, title: str) -> bool:
    """Whether an opening heading restates the title: the title's own words
    in the title's order, case, whitespace and punctuation aside (live
    smoke wave 3, F2: "Review Memo – Project Alpha …" as the title and
    "Review Memo: Project Alpha …" as the heading printed twice). A heading
    with any word more or fewer is kept, so no word of it is lost (live
    smoke 2c, E8: F44's heading "Review: Std. No. 21.01, Rev. 2 -
    Bioretention Cross-Section (BMP Fig. 4.1.3)" began with the title's
    words, was taken for the title and was dropped with its figure
    number)."""
    h = _words(heading)
    return bool(h) and h == _words(title)


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
    who = str(author or "").strip() or __getattr__("DEFAULT_AUTHOR")
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
                 page_break, text_height=None):
        self.doc = doc
        self.base_dir = base_dir
        self.text_width = text_width
        self.text_height = text_height
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
    strike = False
    script = None                       # "sup", "sub" or None
    for child in children or ():
        kind = child.type
        if kind == "s_open":
            strike = True
        elif kind == "s_close":
            strike = False
        elif kind == "html_inline" and _SCRIPT_TAG.match(
                child.content.strip()):
            m = _SCRIPT_TAG.match(child.content.strip())
            script = None if m.group(1) else m.group(2).lower()
        elif kind == "strong_open":
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
            _mark(run, strike, script)
        elif kind == "text":
            run = par.add_run(child.content)
            run.bold, run.italic = bold, italic
            _mark(run, strike, script)
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


def _mark(run, strike, script):
    """Struck-through, superscript and subscript formatting on one run."""
    if strike:
        run.font.strike = True
    if script == "sup":
        run.font.superscript = True
    elif script == "sub":
        run.font.subscript = True


def _picture(par, token, ctx, block=False):
    """Place an image, scaled down to the text width, or say why it is not there.

    A photograph goes in as JPEG at print size (:func:`_photo_as_jpeg`,
    DR-47); every other picture goes in as its file, as before.
    """
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
    photo = _photo_as_jpeg(path, ctx)
    try:
        if photo is None:
            shape = par.add_run().add_picture(path)
        else:
            import io
            data, (cx, cy) = photo
            # Printed at the size the file itself would have been.
            shape = par.add_run().add_picture(io.BytesIO(data), width=cx,
                                              height=cy)
    except Exception as exc:
        ctx.warnings.append(
            f"image not found or unreadable, left as a note: {src} "
            f"({type(exc).__name__})")
        par.add_run(f"[image not found: {src}]").italic = True
        return
    # Fit inside the text width AND the page's text height (a tall picture
    # scaled on width alone ran off the page); never enlarge.
    scale = 1.0
    if shape.width > ctx.text_width:
        scale = ctx.text_width / shape.width
    if ctx.text_height and shape.height * scale > ctx.text_height:
        scale = ctx.text_height / shape.height
    if scale < 1.0:
        shape.width = max(1, int(shape.width * scale))
        shape.height = max(1, int(shape.height * scale))
    if block and alt:
        caption = ctx.doc.add_paragraph()
        ctx.style(caption, "Caption")
        run = caption.add_run(alt)
        run.italic = True


#: The photograph test (DR-47) looks at a sample of at most this many pixels
#: along the longer side.
_PHOTO_SAMPLE = 256


def _is_photograph(rgb) -> bool:
    """Whether an RGB picture is a photograph, judged by its pixels alone.

    Measured on a sample of at most ``_PHOTO_SAMPLE`` pixels a side, taken
    nearest-pixel so that no colour is made up by the sampling. Both must
    hold, and every doubt answers "no", which leaves the picture as it is:

    - Continuous tone: the sample holds at least one distinct colour for
      every five pixels. Line work, lettering, charts and drawings are a
      few flat colours and their anti-aliased edges (about one in ten or
      fewer even for a dense scatter chart); a photograph's colours vary
      from pixel to pixel.
    - No paper: fewer than half of the sampled pixels lie within 32 levels,
      in every channel, of the picture's most common tone. A drawing, a
      chart, a scan or a page of text is mostly its background; a
      photograph has no one tone that covers half of it. (A photograph
      that is half sky fails this and stays as it is.)

    The most common tone is the centre of the fullest cell on eight grids
    of cells 32 levels wide, each channel's cells either as they are or
    shifted by 16 levels, and the largest share within 32 levels of such a
    centre is the one judged. On one of the eight grids any paper tone
    lies at least 8 levels inside its cell in every channel, so scanner
    noise of a few levels (checked up to a standard deviation of 5) keeps
    the paper in one cell; on a single grid a paper tone at a cell edge
    (about 224) split into up to eight fragments, and a flat coloured zone
    over a sixth of a scan outnumbered each and was taken for the paper.

    Some charts and drawing crops are photograph-like in their pixels and
    pass both: a smoothly shaded heat map, a shaded-relief map, line work
    drawn over a photograph, a scan under strongly uneven light. No rule
    on pixels tells them from photographs; they are stored as JPEG at full
    colour resolution (:func:`_photo_as_jpeg`), so their coloured lettering
    and symbols do not bleed.
    """
    import itertools

    import numpy as np
    from PIL import Image
    w, h = rgb.size
    k = min(1.0, _PHOTO_SAMPLE / max(w, h))
    sample = rgb if k >= 1.0 else rgb.resize(
        (max(1, round(w * k)), max(1, round(h * k))), Image.NEAREST)
    n = sample.size[0] * sample.size[1]
    if len(sample.getcolors(maxcolors=n)) * 5 < n:
        return False
    px = np.asarray(sample, dtype=np.int16).reshape(-1, 3)
    weights = np.array([81, 9, 1], dtype=np.int16)     # cells 0..8 a channel
    for shift in itertools.product((0, 16), repeat=3):
        cells = ((px + np.array(shift, dtype=np.int16)) >> 5) @ weights
        centre = px[cells == np.bincount(cells).argmax()].mean(axis=0)
        if float((np.abs(px - centre).max(axis=1) <= 32).mean()) >= 0.5:
            return False
    return True


def _photo_as_jpeg(path, ctx):
    """A photograph's bytes for the .docx, or ``None`` to place the file as
    it is (DR-47).

    Photo crops arrived as PNG at 200 to 300 dpi and were embedded
    unchanged, about 1 MB each: a 20-photo report passed a common 20 MB
    mail limit, and the agent dropped evidence photos to get under it. A
    photograph (:func:`_is_photograph`) is now stored as JPEG at
    ``PHOTO_JPEG_QUALITY`` and full colour resolution (4:4:4), with no
    more than ``PHOTO_MAX_DPI`` pixels per
    inch of the size it is printed at; it prints at the same size as
    before. Line work, lettering, charts and drawing crops are not
    photographs and stay as they are: JPEG blurs edges and lettering, and
    PNG already stores them small.

    Left as it is, always: a picture that is not RGB (a grey one: scans of
    drawings and text are grey; a palette, CMYK), one with any transparent
    pixel or an EXIF rotation, a JPEG already within the size, and any
    picture whose JPEG would not be smaller than its file. The source file
    is only read. Any failure returns ``None``, so a picture Pillow cannot
    open is placed exactly as before.

    Returns ``(jpeg bytes, (width, height))``, the size in EMU at which the
    file itself would have been placed (python-docx's own reading of it).
    """
    try:
        import io
        from PIL import Image
        from docx.image.image import Image as DocxImage
        with Image.open(path) as img:
            if img.mode not in ("RGB", "RGBA") \
                    or img.info.get("transparency") is not None \
                    or img.getexif().get(0x0112, 1) != 1:
                return None
            img.load()
            if img.mode == "RGBA":
                if img.getchannel("A").getextrema()[0] < 255:
                    return None
                rgb = img.convert("RGB")
            else:
                rgb = img
            if not _is_photograph(rgb):
                return None
            native = DocxImage.from_file(path)
            cx, cy = int(native.width), int(native.height)
            fit = min(1.0, ctx.text_width / cx)
            if ctx.text_height:
                fit = min(fit, ctx.text_height / cy)
            wide_in, high_in = cx * fit / 914400, cy * fit / 914400
            w, h = rgb.size
            k = min(1.0, wide_in * PHOTO_MAX_DPI / w,
                    high_in * PHOTO_MAX_DPI / h)
            if k >= 1.0 and img.format in ("JPEG", "MPO"):
                return None
            if k < 1.0:
                rgb = rgb.resize((max(1, round(w * k)), max(1, round(h * k))),
                                 Image.LANCZOS)
            # The JPEG's own dpi is its printed size, so reading the file
            # back and writing it again prints it at the same size.
            dpi = max(1, round(rgb.size[0] / wide_in))
            extra = {}
            if img.info.get("icc_profile"):
                extra["icc_profile"] = img.info["icc_profile"]
            buf = io.BytesIO()
            # Full colour resolution (4:4:4, ``subsampling=0``), not the
            # writer's default 4:2:0: a chart or drawing crop that is
            # photograph-like in its pixels converts too, and its coloured
            # lettering must not bleed. It costs about a fifth to a half
            # more bytes.
            rgb.save(buf, "JPEG", quality=PHOTO_JPEG_QUALITY, subsampling=0,
                     dpi=(dpi, dpi), **extra)
        data = buf.getvalue()
        if len(data) >= os.path.getsize(path):
            return None
        return data, (cx, cy)
    except Exception:  # noqa: BLE001 - never costs the picture its place
        return None


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

_MATH = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"

#: Word's own equations (``m:oMath``), counted as not read (DR-45).
_EQUATION = "equation (Word's own equation format)"

#: Fonts whose letters stand for other characters and for which no
#: published table is used here: text set in them is left as it is and the
#: run is counted as not read (DR-45).
_SYMBOL_FONTS_NO_TABLE = {"wingdings", "wingdings 2", "wingdings 3",
                          "webdings", "zapf dingbats", "zapfdingbats",
                          "itc zapf dingbats", "marlett", "mt extra"}

#: How such a run, or a Symbol-font run with a character outside the
#: table, is named in the not-read section.
_SYMBOL_RUN = ("run set in a symbol font (the letters shown are not the "
               "characters printed)")


def _symbol_char(ch):
    """A character of a Symbol-font run as the character printed, or
    ``None`` when the table has none. White space (a space, a non-breaking
    space) stays as it is; a character beyond Latin-1 and outside
    0xF000-0xF0FF is already a real character (Unicode text with the font
    applied) and stays as it is."""
    code = ord(ch)
    if ch.isspace():
        return ch
    if 0xF000 <= code <= 0xF0FF:
        code -= 0xF000
    elif code > 0xFF:
        return ch
    return _SYMBOL_FONT.get(code)


def _symbol_text(text):
    return "".join(_symbol_char(ch) for ch in text)


#: The heading of the tracked-changes section :func:`docx_to_markdown`
#: returns apart from the body, as ``tracked_changes`` (DR-45).
TRACKED_CHANGES_HEADING = "Tracked changes in the Word file"

#: A deleted passage longer than this is given in that section by its start
#: and its end, with the number of characters left out said between them
#: (DR-45; a whole deleted page went to the model) ...
_DELETION_MAX_CHARS = 300
#: ... and the section's deletion lines stop at this many characters; one
#: line then says how many more deletions the file holds.
_DELETIONS_MAX_CHARS = 4000

#: The Symbol font's characters, code -> Unicode, for a ``w:sym`` (DR-45).
#: From the Unicode Consortium's published table "Adobe Symbol Encoding to
#: Unicode" (unicode.org/Public/MAPPINGS/VENDORS/ADOBE/symbol.txt, table
#: version 1.0, 2011-07-12), whose code order the Symbol font follows; Word
#: writes the code itself or 0xF000 plus it. Where that table gives a code
#: two values, the Greek letter is taken (0x6D is mu, not the micro sign),
#: else the first listed. Its 29 codes assigned only to private-use values
#: (bracket pieces, arrow extenders, serif and sans-serif marks) and the
#: codes it leaves unassigned have no character here: such a symbol is
#: counted as not read, never guessed.
_SYMBOL_FONT = {
    int(code, 16): chr(int(uni, 16)) for code, uni in (
        pair.split(":") for pair in """
20:0020 21:0021 22:2200 23:0023 24:2203 25:0025 26:0026 27:220B 28:0028
29:0029 2A:2217 2B:002B 2C:002C 2D:2212 2E:002E 2F:002F 30:0030 31:0031
32:0032 33:0033 34:0034 35:0035 36:0036 37:0037 38:0038 39:0039 3A:003A
3B:003B 3C:003C 3D:003D 3E:003E 3F:003F 40:2245 41:0391 42:0392 43:03A7
44:0394 45:0395 46:03A6 47:0393 48:0397 49:0399 4A:03D1 4B:039A 4C:039B
4D:039C 4E:039D 4F:039F 50:03A0 51:0398 52:03A1 53:03A3 54:03A4 55:03A5
56:03C2 57:03A9 58:039E 59:03A8 5A:0396 5B:005B 5C:2234 5D:005D 5E:22A5
5F:005F 61:03B1 62:03B2 63:03C7 64:03B4 65:03B5 66:03C6 67:03B3 68:03B7
69:03B9 6A:03D5 6B:03BA 6C:03BB 6D:03BC 6E:03BD 6F:03BF 70:03C0 71:03B8
72:03C1 73:03C3 74:03C4 75:03C5 76:03D6 77:03C9 78:03BE 79:03C8 7A:03B6
7B:007B 7C:007C 7D:007D 7E:223C A0:20AC A1:03D2 A2:2032 A3:2264 A4:2044
A5:221E A6:0192 A7:2663 A8:2666 A9:2665 AA:2660 AB:2194 AC:2190 AD:2191
AE:2192 AF:2193 B0:00B0 B1:00B1 B2:2033 B3:2265 B4:00D7 B5:221D B6:2202
B7:2022 B8:00F7 B9:2260 BA:2261 BB:2248 BC:2026 BF:21B5 C0:2135 C1:2111
C2:211C C3:2118 C4:2297 C5:2295 C6:2205 C7:2229 C8:222A C9:2283 CA:2287
CB:2284 CC:2282 CD:2286 CE:2208 CF:2209 D0:2220 D1:2207 D5:220F D6:221A
D7:22C5 D8:00AC D9:2227 DA:2228 DB:21D4 DC:21D0 DD:21D1 DE:21D2 DF:21D3
E0:25CA E1:2329 E5:2211 F1:232A F2:222B F3:2320 F5:2321
""".split())}


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
        ``picture N`` placeholder each); ``properties`` (title, author);
        ``extras`` (the headings of what was read besides the body and
        written after it: comments, footnotes, endnotes, headers and
        footers, text boxes, and what was there but not read, such as
        charts and equations); ``tracked_changes`` (DR-45: a Markdown
        section, headed :data:`TRACKED_CHANGES_HEADING`, that says the file
        holds tracked changes and gives the deleted text; ``""`` when it
        holds none). ``markdown`` stays the writer's inverse: the body reads
        tracked changes as accepted, and the section is not in it.
    """
    import docx

    doc = docx.Document(source)
    reader = _DocxReader(doc, image_dir, image_ref, image_prefix)
    reader.read_container(doc.element.body)
    extras = reader.read_extras()
    cp = doc.core_properties
    return {
        "markdown": _assemble(reader.blocks),
        "extras": extras,
        "tracked_changes": reader.tracked_changes_text(),
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
        self._open_comments = set()        # comment ranges open right now
        self._comment_text = {}            # comment id -> text it is on
        self.textboxes = []                # text boxes' text, in order
        self.unread = {}                   # kind -> count, there, not read
        self.tracked = set()               # kinds of tracked change met
        self.deletions = []                # tracked deletions, in order
        self._comment_seen = set()         # comment ranges met in the text
        # The part being read (a header, footer, comments or notes part):
        # its pictures' and links' r:ids are its own, not the body's.
        self._part = None

    def _rels_part(self):
        """The part whose relationships the r:ids being read name."""
        return self._part if self._part is not None else self.doc.part

    def _range_mark(self, el):
        """A comment range opening or closing (inside a paragraph, or
        between paragraphs, tables, rows and cells)."""
        cid = el.get(_qn("w:id"))
        if el.tag == _qn("w:commentRangeStart"):
            self._open_comments.add(cid)
            self._comment_seen.add(cid)
        else:
            self._open_comments.discard(cid)

    def _with_ranges(self, el, tag):
        """``el``'s children with ``tag``, looking through content controls
        and noting the comment ranges that open and close between them."""
        marks = (_qn("w:commentRangeStart"), _qn("w:commentRangeEnd"))
        wrap = (_qn("w:sdt"), _qn("w:sdtContent"), _qn("w:customXml"))
        for child in el.iterchildren():
            if child.tag == tag:
                yield child
            elif child.tag in marks:
                self._range_mark(child)
            elif child.tag in wrap:
                yield from self._with_ranges(child, tag)

    # -- containers ------------------------------------------------------
    def read_container(self, el):
        p_tag, tbl_tag = _qn("w:p"), _qn("w:tbl")
        wrap = (_qn("w:sdt"), _qn("w:sdtContent"), _qn("w:customXml"))
        marks = (_qn("w:commentRangeStart"), _qn("w:commentRangeEnd"))
        for child in el.iterchildren():
            if child.tag == p_tag:
                self.paragraph(child)
            elif child.tag == tbl_tag:
                self.table(child)
            elif child.tag in marks:
                self._range_mark(child)
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
            rel = self._rels_part().rels[rid]
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
                part = self._rels_part().related_parts[rid]
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

    def _picture_or_box(self, el, segs):
        """A drawing or picture run child: a text box is read as text (kept
        for the "Text boxes" section), a chart, diagram or embedded object
        is counted as not read, anything else is a picture."""
        tb = _qn("w:txbxContent")
        if el.find(".//" + tb) is not None \
                and el.find(".//" + _qn("a:blip")) is None \
                and el.find(".//" + _VML + "imagedata") is None:
            boxes = [b for b in el.iter(tb)
                     if not any(a.tag == tb for a in b.iterancestors())]
            for box in boxes:
                text = self._paragraphs_text(box)
                if text:
                    self.textboxes.append(text)
            return
        for gd in el.iter(_qn("a:graphicData")):
            uri = (gd.get("uri") or "").lower()
            if uri.endswith("/chart"):
                self._not_read("chart")
                return
            if uri.endswith("/diagram"):
                self._not_read("SmartArt diagram")
                return
        obj = el if el.tag == _qn("w:object") \
            else el.find(".//" + _qn("w:object"))
        if obj is not None:
            # An OLE object (an Equation 3.0 or MathType equation, an
            # embedded workbook or drawing): named by its program, not read.
            prog = next((o.get("ProgID") for o in obj.iter(
                "{urn:schemas-microsoft-com:office:office}OLEObject")
                if o.get("ProgID")), "")
            self._not_read(f"embedded object ({prog})" if prog
                           else "embedded object")
            return
        segs.append(self._image(el))

    def _not_read(self, what):
        self.unread[what] = self.unread.get(what, 0) + 1

    def _sym(self, el, run=None) -> str:
        """A ``w:sym``'s character where its font's table is known (the
        Symbol font's, ``_SYMBOL_FONT``); otherwise ``""``, counted as not
        read (DR-45)."""
        font = (el.get(_qn("w:font")) or "").strip()
        if not font and run is not None:
            try:
                font = (run.font.name or "").strip()
            except Exception:  # noqa: BLE001 - formatting is best effort
                font = ""
        try:
            code = int(el.get(_qn("w:char")) or "", 16)
        except ValueError:
            code = -1
        if 0xF000 <= code <= 0xF0FF:
            code -= 0xF000
        if font.lower() == "symbol" and code in _SYMBOL_FONT:
            return _SYMBOL_FONT[code]
        self._not_read(f"symbol character in the {font} font" if font
                       else "symbol character in an unnamed font")
        return ""

    def _run_font(self, r_el, run) -> str:
        """The font a run's text is set in: the run's own ``w:rFonts``,
        else its character style's (``w:ascii``, else ``w:hAnsi``;
        ``w:cs`` for a complex-script run). Paragraph styles, the
        document defaults and theme fonts are not followed."""
        rpr = r_el.find(_qn("w:rPr"))
        if rpr is None:
            return ""
        attrs = ["w:ascii", "w:hAnsi"]
        if rpr.find(_qn("w:cs")) is not None \
                or rpr.find(_qn("w:rtl")) is not None:
            attrs.insert(0, "w:cs")
        holders = [rpr]
        if rpr.find(_qn("w:rStyle")) is not None:
            try:
                srpr = run.style.element.find(_qn("w:rPr"))
                if srpr is not None:
                    holders.append(srpr)
            except Exception:  # noqa: BLE001 - formatting is best effort
                pass
        for holder in holders:
            fonts = holder.find(_qn("w:rFonts"))
            if fonts is None:
                continue
            for a in attrs:
                name = (fonts.get(_qn(a)) or "").strip()
                if name:
                    return name
        return ""

    def _symbol_run(self, r_el, run):
        """For a run set in a symbol font: the function that turns its
        text into the characters printed (the Symbol font, when every
        character of the run is in ``_SYMBOL_FONT``), else ``None``. A run
        in a symbol font that cannot be turned so keeps its text and is
        counted once as not read (DR-45)."""
        font = self._run_font(r_el, run).lower()
        if font != "symbol" and font not in _SYMBOL_FONTS_NO_TABLE:
            return None
        text = "".join(t.text or "" for t in r_el.iterchildren(_qn("w:t")))
        if not text.strip():
            return None
        if font == "symbol" and all(_symbol_char(ch) is not None
                                    for ch in text):
            return _symbol_text
        self._not_read(_SYMBOL_RUN)
        return None

    def _deleted(self, el, segs, found):
        """A tracked deletion (``w:del``): left out of the text, as Word's
        "accept all" view has it, and kept with where it was for the
        "Tracked changes" section (DR-45). Word splits one deletion into a
        ``w:del`` per run: neighbours by one author are joined."""
        self.tracked.add("deletion")
        texts = (_qn("w:delText"), _qn("w:t"))
        parts = []
        for x in el.iter(*texts, _qn("w:sym"), _qn("w:tab"), _qn("w:br"),
                         _qn("w:cr")):
            if x.tag in texts:
                parts.append(x.text or "")
            elif x.tag == _qn("w:sym"):
                parts.append(self._sym(x))
            else:
                parts.append(" ")
        text = "".join(parts)
        if not text.strip():
            return
        who = el.get(_qn("w:author")) or "unknown author"
        when = (el.get(_qn("w:date")) or "")[:10]
        pos = sum(len(s["s"]) for s in segs if s["t"] == "text")
        if found and found[-1]["pos"] == pos and found[-1]["who"] == who:
            found[-1]["text"] += text
            return
        found.append({"who": who, "when": when, "text": text, "pos": pos})
        self.deletions.append(found[-1])

    def tracked_changes_text(self) -> str:
        """The "Tracked changes in the Word file" section, or ``""`` when
        the file holds none (DR-45). Kept out of the Markdown body, which
        stays the writer's inverse (Word's "accept all" reading): it says
        the file holds tracked changes and gives the deleted text."""
        if not self.tracked:
            return ""
        intro = ("The file holds tracked changes. The text above reads "
                 "them as accepted: insertions in, deletions out")
        if "move" in self.tracked:
            intro += ", moved text at its new place"
        intro += (". The deleted text, which is not part of the document's "
                  "current text:" if self.deletions else ".")
        parts = [f"## {TRACKED_CHANGES_HEADING}", intro]
        if self.deletions:
            # Bounded (_DELETIONS_MAX_CHARS); the count stays exact.
            lines, used = [], 0
            for d in self.deletions:
                line = self._deletion_line(d)
                if lines and used + len(line) + 1 > _DELETIONS_MAX_CHARS:
                    break
                lines.append(line)
                used += len(line) + 1
            left = len(self.deletions) - len(lines)
            if left:
                lines.append(
                    f"({left} more deletion"
                    f"{'s are' if left > 1 else ' is'} not shown here: this "
                    f"list stops at about {_DELETIONS_MAX_CHARS:,} "
                    f"characters.)")
            parts.append("\n".join(lines))
        return "\n\n".join(parts) + "\n"

    @staticmethod
    def _deletion_line(d) -> str:
        """One line of the "Tracked changes" section."""
        def clip(s, tail):
            if len(s) <= 40:
                return s
            if tail:
                return "..." + s[-40:].split(" ", 1)[-1]
            return s[:40].rsplit(" ", 1)[0] + "..."
        before = " ".join(d.get("before", "").split())
        after = " ".join(d.get("after", "").split())
        if before:
            where = f' after "{clip(before, True)}"'
        elif after:
            where = f' before "{clip(after, False)}"'
        else:
            where = " (its paragraph has no other text)"
        head = d["who"] + (f", {d['when']}" if d["when"] else "")
        text = " ".join(d["text"].split())
        if len(text) > _DELETION_MAX_CHARS:
            # Its start and its end, at word breaks; the count of the
            # characters between them is exact.
            keep = _DELETION_MAX_CHARS // 2
            start = text[:keep].rsplit(" ", 1)[0]
            end = text[-keep:].split(" ", 1)[-1]
            text = (f"{start} [... {len(text) - len(start) - len(end):,} "
                    f"characters not shown ...] {end}")
        return f'- {head}: deleted "{text}"{where}'

    def _paragraphs_text(self, container, sep=" / "):
        """The text of every paragraph under ``container``, joined."""
        parts = []
        for p in container.iter(_qn("w:p")):
            t = " ".join(self.inline(p).split())
            if t:
                parts.append(t)
        return sep.join(parts)

    def segments(self, p_el):
        """The pieces of one paragraph: text runs with their formatting,
        line and page breaks, pictures — tracked insertions in, deletions
        out (kept for the "Tracked changes" section), field codes out; a
        ``w:sym``, and a run set in the Symbol font, as the characters
        printed where the font's table is known, an equation in Word's own
        format counted as not read (DR-45)."""
        from docx.text.paragraph import Paragraph
        from docx.text.run import Run
        par = Paragraph(p_el, self.body)
        segs = []
        found = []                       # this paragraph's deletions
        r_tag, link_tag = _qn("w:r"), _qn("w:hyperlink")
        del_tag, move_from = _qn("w:del"), _qn("w:moveFrom")
        ins_tag, move_to = _qn("w:ins"), _qn("w:moveTo")
        math_para, math = _MATH + "oMathPara", _MATH + "oMath"
        sym_tag = _qn("w:sym")
        into = (_qn("w:ins"), _qn("w:moveTo"), _qn("w:smartTag"),
                _qn("w:customXml"), _qn("w:fldSimple"), _qn("w:sdt"),
                _qn("w:sdtContent"))
        t_tag, tab_tag, br_tag, cr_tag = (_qn("w:t"), _qn("w:tab"),
                                          _qn("w:br"), _qn("w:cr"))
        # w:object (an embedded OLE object) is a run child of its own, beside
        # w:drawing and w:pict: counted as not read (DR-41 review).
        pics = (_qn("w:drawing"), _qn("w:pict"), _qn("w:object"))
        hyphen = _qn("w:noBreakHyphen")
        mc = "{http://schemas.openxmlformats.org/markup-compatibility/2006}"
        alt_tag = mc + "AlternateContent"
        cr_start, cr_end = _qn("w:commentRangeStart"), _qn("w:commentRangeEnd")

        def run(r_el, link):
            r = Run(r_el, par)
            try:
                b, i = bool(r.bold), bool(r.italic)
            except Exception:  # noqa: BLE001
                b = i = False
            c = self._is_mono(r)
            as_printed = self._symbol_run(r_el, r)
            for ch in r_el.iterchildren():
                if ch.tag == t_tag:
                    s = ch.text or ""
                    if as_printed is not None:
                        s = as_printed(s)
                    segs.append({"t": "text", "s": s, "b": b,
                                 "i": i, "c": c, "link": link})
                    for cid in self._open_comments:
                        self._comment_text.setdefault(cid, []).append(s)
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
                elif ch.tag == sym_tag:
                    s = self._sym(ch, r)
                    if s:
                        segs.append({"t": "text", "s": s, "b": b, "i": i,
                                     "c": c, "link": link})
                        for cid in self._open_comments:
                            self._comment_text.setdefault(cid, []).append(s)
                elif ch.tag in pics:
                    self._picture_or_box(ch, segs)
                elif ch.tag == alt_tag:
                    # mc:AlternateContent holds the same drawing twice (the
                    # modern form and a fallback): read the first only.
                    pick = ch.find(mc + "Choice")
                    if pick is None:
                        pick = ch.find(mc + "Fallback")
                    for inner in (pick.iterchildren() if pick is not None
                                  else ()):
                        if inner.tag in pics:
                            self._picture_or_box(inner, segs)

        def walk(el, link):
            for ch in el.iterchildren():
                if ch.tag in (cr_start, cr_end):
                    self._range_mark(ch)
                elif ch.tag == r_tag:
                    run(ch, link)
                elif ch.tag == link_tag:
                    walk(ch, self._link_url(ch) or link)
                elif ch.tag == del_tag:
                    self._deleted(ch, segs, found)
                elif ch.tag == move_from:
                    # Moved away: read at the place it moved to.
                    self.tracked.add("move")
                elif ch.tag in into:
                    if ch.tag == ins_tag:
                        self.tracked.add("insertion")
                    elif ch.tag == move_to:
                        self.tracked.add("move")
                    walk(ch, link)
                elif ch.tag == math_para:
                    for _eq in ch.iterchildren(math):
                        self._not_read(_EQUATION)
                elif ch.tag == math:
                    self._not_read(_EQUATION)

        walk(p_el, None)
        if found:
            full = "".join(s["s"] for s in segs if s["t"] == "text")
            for d in found:
                d["before"], d["after"] = full[:d["pos"]], full[d["pos"]:]
        return segs, par

    def inline(self, p_el, drop_bold=False) -> str:
        """One paragraph's text as inline Markdown (a table cell)."""
        segs, _par = self.segments(p_el)
        segs = [s for s in segs if s["t"] != "page"]
        return _segments_md(segs, drop_bold=drop_bold).strip()

    # -- what is in the file besides the body --------------------------------
    def _part_roots(self, tail):
        """``(parsed XML, part)`` of each related part whose type ends
        ``tail``, and the error names of the parts that would not parse."""
        from lxml import etree
        roots, failed = [], []
        for rel in self.doc.part.rels.values():
            if rel.is_external or not rel.reltype.endswith(tail):
                continue
            try:
                part = rel.target_part
                roots.append((etree.fromstring(part.blob), part))
            except Exception as exc:  # noqa: BLE001
                failed.append(type(exc).__name__)
        return roots, failed

    def read_extras(self):
        """Comments, footnotes, endnotes, headers and footers and text
        boxes, added to the blocks under headings of their own, and what
        was there but could not be read. Returns the headings added."""
        added = []

        def section(title, lines):
            if lines:
                self._add("h", "## " + title)
                self._add("p", "\n".join(lines))
                added.append(title)

        # Notes, headers and footers are read first (a comment can sit on
        # their text), each with its own part's relationships, so a picture
        # or link in it is its own and not the body's (DR-41 review).
        notes = []
        for tail, title in (("/footnotes", "Footnotes in the Word file"),
                            ("/endnotes", "Endnotes in the Word file")):
            lines = []
            roots, failed = self._part_roots(tail)
            kind = "footnote" if tail == "/footnotes" else "endnote"
            for root, part in roots:
                tag = _qn("w:footnote" if kind == "footnote"
                          else "w:endnote")
                self._part = part
                try:
                    for n in root.iter(tag):
                        if n.get(_qn("w:type")) in ("separator",
                                                    "continuationSeparator",
                                                    "continuationNotice"):
                            continue
                        text = self._paragraphs_text(n)
                        if text:
                            lines.append(f"- {kind} {n.get(_qn('w:id'))}: "
                                         f"{text}")
                finally:
                    self._part = None
            if failed:
                self._not_read(f"{kind}s part (unparsable)")
            notes.append((title, lines))

        # Headers and footers: each distinct text once.
        heads, seen = [], set()
        try:
            for sec in self.doc.sections:
                for label, hf in (
                        ("header", sec.header), ("first-page header",
                                                 sec.first_page_header),
                        ("even-page header", sec.even_page_header),
                        ("footer", sec.footer), ("first-page footer",
                                                 sec.first_page_footer),
                        ("even-page footer", sec.even_page_footer)):
                    if hf.is_linked_to_previous:
                        continue
                    self._part = hf.part
                    try:
                        text = self._paragraphs_text(hf._element)
                    finally:
                        self._part = None
                    if text and (label, text) not in seen:
                        seen.add((label, text))
                        heads.append(f"- {label}: {text}")
        except Exception:  # noqa: BLE001
            self._not_read("headers and footers (unparsable)")

        # Comments: author, date, what each is attached to, its words.
        lines = []
        roots, failed = self._part_roots("/comments")
        for root, part in roots:
            for c in root.iter(_qn("w:comment")):
                cid = c.get(_qn("w:id"))
                who = c.get(_qn("w:author")) or "unknown author"
                when = (c.get(_qn("w:date")) or "")[:10]
                on = " ".join("".join(
                    self._comment_text.get(cid, [])).split())
                if len(on) > 120:
                    on = on[:117] + "..."
                self._part = part
                try:
                    text = self._paragraphs_text(c) or "(empty)"
                finally:
                    self._part = None
                head = who + (f", {when}" if when else "")
                if on:
                    where = f' on "{on}"'
                elif cid in self._comment_seen:
                    where = (" (attached to a place with no text, such as "
                             "a picture)")
                else:
                    where = " (the text it is attached to was not found)"
                lines.append(f"- {head}{where}: {text}")
        if failed:
            self._not_read("comments part (unparsable)")
        section("Comments in the Word file", lines)
        for title, note_lines in notes:
            section(title, note_lines)
        section("Headers and footers in the Word file", heads)

        section("Text boxes in the Word file",
                [f"- {t}" for t in self.textboxes])
        section("Not read in the Word file",
                [f"- {n} x {what}: present in the file, not read"
                 for what, n in self.unread.items()])
        return added

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
        for tr in self._with_ranges(tbl_el, _qn("w:tr")):
            cells = []
            for tc in self._with_ranges(tr, _qn("w:tc")):
                span = 1
                tcpr = tc.tcPr if hasattr(tc, "tcPr") else None
                try:
                    if tcpr is not None and tcpr.gridSpan is not None:
                        span = max(1, int(tcpr.gridSpan))
                except (TypeError, ValueError, AttributeError):
                    span = 1
                header = not rows
                parts = [self.inline(p, drop_bold=header)
                         for p in self._with_ranges(tc, _qn("w:p"))]
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
