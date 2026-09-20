"""The experiment: label a page by LOOKING at it, and nothing else.

planlens' rules label a page from what that page prints about itself, and
:mod:`report_ingest.label_review` corrects them with the whole report in
view. Both read TEXT. This module asks a different question, and asks it of
the cheapest model on the tier list: given the page as a PICTURE and the
eighteen-label vocabulary, what is this page?

WHY IT IS WORTH MEASURING. The rules reach 0.91 accuracy on the reports they
were built on and 0.79 on the next report anybody opens, and their worst
pages are the ones with no readable text at all -- a scanned appendix, a
plotted log, a photographed pit face. A rule cannot fire on those and the
review has to spend a tool call rendering each one. A vision-first pass
looks at every page by construction, so the pages the rules are worst at are
the ones it is under no handicap on. Whether that is worth its cost is a
measurement, not an opinion, and :mod:`report_ingest.cluster_scoring` makes
it against the SAME hand labels and the SAME scorer as the rules.

THREE MODES, AND THE TRADE BETWEEN THEM.

``mode="page"``
    One call per page, the page rendered whole. Every page gets the model's
    full attention and the page's own fine print is legible. A hundred pages
    is a hundred calls. **Nothing but the page is in view**, which is the
    mode's whole point and also its ceiling: the page-mode run on the
    cluster (2026-09-18, 307 pages) matched the rules-plus-review score and
    beat it on narrative and figure recall, while LOSING on plan and
    lab_test -- the two labels a reader settles by knowing which appendix
    the page is sitting in.
``mode="sheet"``
    One call per contact sheet of several pages, each thumbnail carrying its
    own page index, exactly as the label review's ``contact_sheet`` tool
    draws them. A hundred pages is seventeen calls at six a sheet, and each
    page is a thumbnail rather than a page. What that costs in accuracy is
    the thing being measured.
``mode="document"``
    The report's pages labelled with **the whole document in view**. The
    owner's own reason for it: *"I was mainly thinking about loading the
    full context up with all pages, not going one-by-one. Because the full
    context of the report is often needed to understand what's happening."*
    Each call carries a STRIP of contact sheets covering the report end to
    end, for orientation, and then a WINDOW of consecutive pages at full
    size, each stamped with its own page number. The model answers for the
    full-size pages only; the thumbnails are there so it can see that
    page 212 sits four pages into Appendix C, whose divider it can read.
    Consecutive windows OVERLAP, so a page skipped in one window can still
    be answered by the next.

    **The cap is the provider's, and it is 50 images in one request**
    (measured on the owner's cluster, 2026-09-18: a 51st image is refused
    with "Too many images in request: 51, maximum allowed: 50"). That, not
    the context window, is what makes this a sliding window rather than one
    call over the whole report: a 1M-token window would hold a 400-page
    report at 770 tokens a page with room to spare, and the image count
    would still refuse it. :data:`MAX_IMAGES_PER_CALL` is a parameter
    because the cap belongs to the endpoint, not to us.

``outline_context=True`` prepends what the document prints about ITSELF --
the contents list, the lists of figures, tables and appendices, and every
divider and fly sheet -- to each call, so a run can put pure vision beside
vision that knows what appendix it is standing in.

WHAT PYTHON REFUSES, rather than passes on. A label outside the vocabulary
becomes ``other`` and a QA note says what the model said; a confidence
outside 0 to 1 is clipped; a page the reply never mentioned is
``unresolved`` and is never guessed at from its neighbours; a reply naming a
page that was not on the sheet is dropped with a note. The budget caps model
calls and every page past it is ``unresolved`` too, so a capped run is
visibly incomplete instead of quietly short.

THE DPI, AND WHY IT IS WHAT IT IS. A 4.1-class vision stack scales an image
to fit its own working size before it looks at it or charges for it: the
short side lands at about 768 px, and the price is counted in 512 px tiles
of what is left. :data:`DEFAULT_DPI` is 100, which puts a letter page at
850 x 1100 and an A4 page at 827 x 1169, both just above that ceiling.

The arithmetic is the whole argument, and it is one-sided. A letter page at
72 dpi is 612 x 792, which is UNDER the ceiling and so is not scaled at all:
four tiles, about 765 tokens. The same page at 100 dpi is scaled down to
768 x 994: four tiles, about 765 tokens. **The same price for a quarter more
pixels on the short side**, so 72 dpi is not the cheap option, it is the
same option with detail thrown away. Going the other way, 200 dpi is scaled
to the same 768 x 994 and costs the same tokens again, for four times the
bytes on the wire and not one pixel the model keeps. Everything from about
90 dpi upwards lands on that identical 768 px short side; 100 leaves room
for a page that is not letter-sized. ``dpi`` is a parameter because the
ceiling is the provider's, not ours, and it moves.

THE DETAIL KEY. ``detail="low"`` tells an OpenAI-shaped endpoint to look at
one 512 px tile of an image and charge about 85 tokens for it, whatever the
image is; the default tiles a letter page into four and charges about 770.
Forty-eight pages at low detail cost about what five cost at default, which
is the difference between a window that is affordable to run over a whole
corpus and one that is not -- and what it costs in accuracy is the next
thing to measure. It is threaded through all three modes and defaults to
unset, so a call that does not ask for it is the request it always was.

NO CONSTRAINTS IN THE SCHEMA. ``confidence`` is described as 0 to 1 rather
than declared with a minimum and a maximum, and ``reason`` is asked for in
twenty words rather than capped with a maxLength. Strict structured output
refuses schema keywords it does not implement, and a rejected call returns
nothing at all, while a number out of range is a thing Python can clip. The
gate belongs here, not in the schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from report_ingest.engine import Engine, image_block, text_block, user
from report_ingest.label_review import LABEL_DEFINITIONS

__all__ = [
    "VISION_LABELS", "MODES", "DEFAULT_DPI", "SHEET_PAGES", "SHEET_COLUMNS",
    "SHEET_THUMB_PX", "MAX_CONTEXT_CHARS",
    "MAX_IMAGES_PER_CALL", "DOCUMENT_WINDOW", "DOCUMENT_OVERLAP",
    "DOCUMENT_MIN_WINDOW", "DOCUMENT_MIN_SPLIT", "STRIP_SHEETS_MAX",
    "STRIP_PER_SHEET", "STRIP_COLUMNS", "STRIP_THUMB_PX", "DETAIL_LEVELS",
    "VISION_JPEG_QUALITY",
    "VisionPageAnswer", "VisionSheetAnswer", "VisionPageLabel", "VisionLabels",
    "vision_system", "classify_pages_by_vision", "stamp_page_number",
    "document_windows", "encode_for_vision", "is_payload_too_large",
]

#: The vocabulary, and it is the label review's: a label this pass can return
#: that planlens cannot produce, or the other way round, would read as a miss
#: on every page carrying it. The one-line definitions are the review's too,
#: for the same reason -- two wordings of one vocabulary are two vocabularies,
#: and the second one's numbers would be the ones nobody checked.
VISION_LABELS: Tuple[str, ...] = tuple(LABEL_DEFINITIONS)

MODES: Tuple[str, ...] = ("page", "sheet", "document")

#: What OpenAI's ``image_url.detail`` accepts. ``None`` leaves it unset.
DETAIL_LEVELS: Tuple[str, ...] = ("auto", "low", "high")

#: See the module docstring: the largest render a 4.1-class vision stack will
#: keep, for a letter page and for A4.
DEFAULT_DPI = 100.0
#: Pages per call in ``sheet`` mode, and how they are laid out. Six in three
#: columns is two rows of thumbnails at planlens' largest thumbnail size.
SHEET_PAGES = 6
SHEET_COLUMNS = 3
SHEET_THUMB_PX = 400
#: Ceiling on the outline text when ``outline_context`` is on. It rides on
#: EVERY call in page mode, so a 30,000-character outline would cost more than
#: the pictures do.
MAX_CONTEXT_CHARS = 6000

#: The endpoint's own ceiling on images in ONE request, measured on the
#: owner's cluster on 2026-09-18: 50 went through and 51 came back "Too many
#: images in request: 51, maximum allowed: 50". It is a parameter on the
#: pass (``images_per_call``) because it belongs to the endpoint and a
#: different one will have a different number.
MAX_IMAGES_PER_CALL = 50
#: Full-size pages in one document-mode call, and how many of them the next
#: call sees again. The overlap is what lets a page the model skipped be
#: answered by the window after it; it is also why a page can be answered
#: twice, and the LATER answer -- the one made with more of the report
#: already decided -- is the one kept.
DOCUMENT_WINDOW = 36
DOCUMENT_OVERLAP = 3
#: However tight ``images_per_call`` is, a window of fewer than this many
#: pages is not worth a call: the strip is trimmed instead.
DOCUMENT_MIN_WINDOW = 12
#: How many contact sheets of the whole report ride on one call, at most. A
#: report longer than ``STRIP_SHEETS_MAX * STRIP_PER_SHEET`` pages cannot
#: show all of itself, so the sheets NEAREST the window are the ones sent.
STRIP_SHEETS_MAX = 12
#: The strip: 48 pages a sheet in 6 columns, each thumbnail captioned with
#: its page index and NOTHING else (see :func:`render_number_sheets` for why
#: planlens' own sheets, which also print the page's kind, are not used).
STRIP_PER_SHEET = 48
STRIP_COLUMNS = 6
STRIP_THUMB_PX = 140
#: The stamp drawn on a full-size page in document mode: a filled box at the
#: top-left corner with "p. N" in it, sized as a fraction of the page's
#: short side so it is legible at any dpi and covers the same sliver of the
#: page at every one.
STAMP_HEIGHT_FRAC = 0.030
STAMP_MIN_PX = 18
#: What every page picture is re-encoded to before it is sent. The pages are
#: RENDERED as PNG and TRAVEL as JPEG: see :func:`encode_for_vision` for the
#: gateway refusal that made it necessary. The pixel size does not change, so
#: neither does the token count.
VISION_JPEG_QUALITY = 80
#: A document-mode window smaller than this is not split again when the
#: gateway refuses it: at four pages the body is already small and the
#: refusal is saying something else, so the error is raised rather than
#: hidden behind a halving that will not help.
DOCUMENT_MIN_SPLIT = 4


# -- what the model returns -------------------------------------------------

class VisionPageAnswer(BaseModel):
    """One page, as the model read it off the picture."""

    page: int = Field(
        description="the 0-based page index of the page you are answering "
                    "for, as printed under the thumbnail or given to you")
    label: Literal[VISION_LABELS] = Field(     # type: ignore[valid-type]
        description="what this page IS; one of the eighteen labels")
    confidence: float = Field(
        description="how sure you are, from 0.0 to 1.0: 1.0 for a page that "
                    "names itself, about 0.5 for a guess from the layout, low "
                    "for a page you cannot make out")
    reason: str = Field(
        description="what on the page decided it, in 20 words or fewer")


class VisionSheetAnswer(BaseModel):
    """Every page on one contact sheet, one entry each."""

    pages: List[VisionPageAnswer] = Field(
        description="one entry for every page shown on the sheet and none "
                    "for any page that is not on it, in the order they are "
                    "shown")


# -- what this pass returns -------------------------------------------------

@dataclass(frozen=True)
class VisionPageLabel:
    """One page's label, as this pass settled it."""

    page: int
    label: str
    confidence: float
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {"page": self.page, "label": self.label,
                "confidence": round(self.confidence, 3), "reason": self.reason}


@dataclass
class VisionLabels:
    """The labels a vision-first pass produced, and what it cost."""

    labels: List[VisionPageLabel]
    cost: Dict[str, Any] = field(default_factory=dict)
    unresolved: List[Dict[str, Any]] = field(default_factory=list)
    #: What Python refused or repaired, one line each. A label outside the
    #: vocabulary, a confidence out of range, a page the reply invented.
    qa: List[Dict[str, Any]] = field(default_factory=list)
    mode: str = "page"
    dpi: float = DEFAULT_DPI
    outline_context: bool = False
    pages_asked: int = 0
    model_calls: int = 0
    budget: Optional[int] = None
    stopped_on_budget: bool = False
    model: str = ""

    @property
    def label_map(self) -> Dict[int, str]:
        """``page -> label``, which is what a scorer compares."""
        return {row.page: row.label for row in self.labels}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "labels": {str(row.page): row.label for row in self.labels},
            "detail": [row.to_dict() for row in self.labels],
            "unresolved": [dict(u) for u in self.unresolved],
            "qa": [dict(q) for q in self.qa],
            "mode": self.mode,
            "dpi": round(float(self.dpi), 1),
            "outline_context": bool(self.outline_context),
            "pages_asked": self.pages_asked,
            "model_calls": self.model_calls,
            "budget": self.budget,
            "stopped_on_budget": self.stopped_on_budget,
            "model": self.model,
            "cost": dict(self.cost),
        }


# -- the prompt -------------------------------------------------------------

_SYSTEM_HEAD = """\
You are looking at pages of one engineering report and saying what each page
IS. You are given the pages as PICTURES. {context}

THE LABEL VOCABULARY. Every page gets exactly one of these eighteen.
"""

#: What page and sheet mode say about the context a page arrives in: none.
_ALONE = ("There is no text layer, no rule label and no neighbouring context "
          "beyond what you can see: answer from the page in front of you.")
#: What document mode says instead. The sentence has to change, because in
#: that mode there IS neighbouring context and the whole point is to use it.
_IN_CONTEXT = ("There is no text layer and no rule label: answer from what "
               "you can see. You are shown the WHOLE report in thumbnail as "
               "well as the pages you are answering for, so read a page in "
               "the light of the pages around it.")

_SYSTEM_RULES = """
How to decide.

- Decide what the page IS, not what it is about. Prose discussing the
  laboratory results is 'narrative'; the laboratory's own result sheet is
  'lab_test'.
- If a page carries ONE exploration's own results, name that exploration --
  boring_log, test_pit_log, cpt_log or dcp_log -- and never figure. These
  results are often PLOTTED rather than tabulated, and a plot of one
  exploration's results is still that exploration's log however much it
  looks like a graph. 'figure' is for the report's own numbered figure
  series. When you can see it is an exploration's own results but not which
  kind, decide on what is being MEASURED, not on how it is drawn: blows per
  increment of penetration against depth is a DCP; continuous tip
  resistance, sleeve friction and pore pressure against depth is a CPT; a
  logged pit or trench face is a test pit; a drilled hole with driven
  samples and blow counts is a boring.
- 'other' is a real answer. Use it when the page is none of the above, or
  when you cannot tell from the picture, and say which in the reason.
- confidence is your own: 1.0 for a page that names itself, about 0.5 for a
  guess from the layout, low for a page you cannot make out. A low
  confidence costs nothing; a confident wrong answer costs a reader.
- reason: 20 words or fewer, naming what on the page decided it.
"""

_PAGE_TAIL = """
You are given ONE page at a time. Answer for that page, with the page index
you were given.
"""

_SHEET_TAIL = """
You are given a CONTACT SHEET of several pages laid out in a grid. Each
thumbnail is labelled beneath it with its 0-based page index, as 'p. N',
and nothing else. Answer for EVERY page on the sheet, once each, using the
page index printed under that thumbnail. Do not answer for a page that is not on the
sheet, and do not leave one out: a page you cannot make out is 'other' at a
low confidence, which is an answer, and a page you skip is a hole.
"""


_DOCUMENT_TAIL = """
You are given the WHOLE report and then part of it.

- First come THUMBNAIL CONTACT SHEETS of the whole report, several pages to
  a row, each thumbnail labelled beneath with its page number. They are for
  ORIENTATION and nothing else: they show you where this report's dividers,
  contents lists and appendices fall and how long each run of pages is. Do
  not answer for a page you have only seen as a thumbnail.
- Then come FULL-SIZE PAGES, in order. Each one is stamped 'p. N' in a box
  at its top-left corner, and N is the page number to answer with -- not a
  number printed by the report itself, which may start counting again in
  every appendix.

Answer for EVERY full-size page, once each, and for NO other page. One entry
per full-size page: a page you cannot make out is 'other' at a low
confidence, which is an answer, and a page you skip is a hole.

USE THE WHOLE REPORT TO DECIDE. This is the point of showing it to you. A
page that says nothing about itself is very often decidable from where it
sits: the divider or fly sheet that opens the run it is in names what the
run holds, the contents list and the lists of figures, tables and appendices
say what the report contains, and a page between two logs of one boring is
part of that log. When the pages before and after a page are all one kind
and the page itself is silent, it is usually that kind -- say so in the
reason, so a reader can tell a decision made from the page from a decision
made from its neighbours.
"""


def vision_system(mode: str = "page") -> str:
    """The system prompt for one mode, vocabulary and all."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; the modes are {list(MODES)}")
    vocab = "\n".join(f"  {name}: {text}"
                      for name, text in LABEL_DEFINITIONS.items())
    tail = {"page": _PAGE_TAIL, "sheet": _SHEET_TAIL,
            "document": _DOCUMENT_TAIL}[mode]
    head = _SYSTEM_HEAD.format(
        context=_IN_CONTEXT if mode == "document" else _ALONE)
    return head + vocab + "\n" + _SYSTEM_RULES + tail


# -- what a picture weighs on the wire ---------------------------------------

def encode_for_vision(png: bytes, quality: int = VISION_JPEG_QUALITY) -> bytes:
    """The same picture as a JPEG, at the same pixel size.

    THE GATEWAY HAS A BODY LIMIT AND IT IS NOT THE IMAGE COUNT. The 50-image
    cap was measured with tiny probe images and says nothing about bytes; on
    2026-09-20 every document-mode window of a 156-page SCANNED report came
    back ``The page was not displayed because the request entity is too
    large``, while the same windows of a text-page report went through. A
    text page at 100 dpi is a sparse PNG of some tens of kilobytes; a scanned
    page is a photograph, and PNG stores a photograph badly -- a few hundred
    kilobytes each, thirty-six of them, base64 at four bytes for three, and
    the request is tens of megabytes.

    So every page picture these modes send is re-encoded here. JPEG at
    quality 80 is the right trade for this job: the model is being asked what
    a page IS, which is settled by layout and headings, not by whether a
    hairline is one pixel or two. **The pixel size is unchanged**, so the
    provider scales and tiles exactly the render it always did and the token
    count does not move -- this is bytes on the wire, not tokens.

    AND IT IS NOT A WIN ON EVERY PAGE. A crisp vector text page is the case
    PNG is good at and JPEG is not: measured on the 22 synthetic pages, the
    JPEG runs between 0.84 and 1.20 of the PNG, and a nearly blank page can
    be several times it. A scan-like page is 0.32. The rule is unconditional
    anyway, because the page that refuses a request is the scanned one and
    picking per page would mean one report's windows were a different size
    from another's for reasons nothing in a run file records. If a text-page
    run ever needs those bytes back, the engines read the media type off the
    bytes, so sending whichever is smaller is a one-line change here.

    RGB, because JPEG has no alpha and a palette-mode PNG will not save as
    one. Pillow does the work; see :func:`stamp_page_number` for why that is
    not a new dependency.
    """
    import io

    from PIL import Image

    image = Image.open(io.BytesIO(png))
    if image.mode != "RGB":
        image = image.convert("RGB")
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=int(quality), optimize=True)
    return out.getvalue()


# -- stamping a page, and cutting a report into windows ----------------------

def stamp_page_number(png: bytes, page: int) -> bytes:
    """The same PNG with ``p. N`` drawn in a filled box at the top left.

    THE PAGE'S OWN NUMBERING IS NOT THE ANSWER. A geotechnical report
    restarts its printed numbering in every appendix -- three pages numbered
    "1" is normal -- so a model told to "use the page number printed on the
    page" would answer for the wrong page half the time. The 0-based index
    this pass counts in is stamped ON the picture instead, which is the one
    channel that cannot be separated from the page it belongs to when
    thirty-six of them arrive in one message.

    The PAGE SIZE IS UNCHANGED: the stamp is drawn over the top-left corner,
    not added as a margin, so the model sees the page at the dpi it was
    rendered at and the token count is the render's.

    Pillow does the drawing. It is not a new dependency -- it arrives with
    matplotlib and streamlit, both of which this package's app requires --
    and it is imported here rather than at module scope so that importing
    :mod:`report_ingest` still costs nothing.
    """
    import io

    from PIL import Image, ImageDraw

    image = Image.open(io.BytesIO(png))
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")
    width, height = image.size
    box_h = max(STAMP_MIN_PX, int(min(width, height) * STAMP_HEIGHT_FRAC))
    label = f"p. {int(page)}"
    draw = ImageDraw.Draw(image)
    font = _stamp_font(box_h)
    try:
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = right - left, bottom - top
    except (AttributeError, TypeError):            # a very old Pillow
        text_w, text_h = draw.textlength(label, font=font), box_h
        left = top = 0
    pad = max(2, box_h // 4)
    box_w = int(text_w) + 2 * pad
    box_h = max(box_h, int(text_h) + 2 * pad)
    draw.rectangle([(0, 0), (box_w, box_h)], fill=(255, 255, 255),
                   outline=(0, 0, 0), width=max(1, box_h // 12))
    draw.text((pad - left, pad - top), label, fill=(0, 0, 0), font=font)
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def _stamp_font(box_h: int) -> Any:
    """A font about ``box_h`` tall, or Pillow's own if none can be loaded.

    A bitmap default font is small but legible, and a stamp the model can
    read is the requirement; nothing here fails because a TrueType file is
    missing from a cluster image.
    """
    from PIL import ImageFont

    size = max(10, int(box_h * 0.8))
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "arialbd.ttf",
                 "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except (OSError, ImportError):
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                              # Pillow < 9.2
        return ImageFont.load_default()


#: The legend a number-only contact sheet carries into the model's message.
NUMBER_SHEET_LEGEND = ("each thumbnail is labelled beneath with its page "
                       "index only, as 'p. N', and nothing else")


def render_number_sheets(doc, pages: Optional[Sequence[int]], *,
                         columns: int, thumb_px: int, per_sheet: int
                         ) -> List[Tuple[bytes, Dict[str, Any]]]:
    """Contact sheets whose only caption is the page index.

    planlens' own ``render_thumbnails`` writes ``"<index> <kind>"`` under
    every tile -- the page's rule-derived shape: ``text``, ``figure``,
    ``form``, ``scanned`` -- and its legend tells the reader so, because the
    label review wants that evidence beside the picture. A vision pass must
    never see it. On the 2026-09-18 corpus run the sheet mode used those
    sheets, read "text" as narrative and "figure" as figure on thousands of
    pages, and scored 0.557 strict where page mode, which shows a bare page,
    scored 0.928 on the same reports. These sheets print ``p. N`` and
    nothing else, and the legend says so.

    Same shape as planlens' sheets -- ``(png, info)`` with ``info["pages"]``
    -- so the strip helpers read either. A page that will not render is an
    empty, captioned tile rather than a failed sheet.
    """
    import io

    from PIL import Image, ImageDraw

    wanted = (list(range(doc.n_pages)) if pages is None
              else [int(p) for p in pages])
    columns = max(1, int(columns))
    per_sheet = max(1, int(per_sheet))
    thumb = max(24, int(thumb_px))
    gap = 8
    label_h = max(14, thumb // 8)
    font = _stamp_font(label_h)
    dpi = max(12.0, thumb / 8.5)            # a letter page fills the tile
    sheets: List[Tuple[bytes, Dict[str, Any]]] = []
    for start in range(0, len(wanted), per_sheet):
        shown = wanted[start:start + per_sheet]
        cols = min(columns, len(shown))
        rows = (len(shown) + cols - 1) // cols
        cell_w, cell_h = thumb + gap, thumb + gap + label_h
        canvas = Image.new("RGB", (cols * cell_w + gap, rows * cell_h + gap),
                           (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        for i, page in enumerate(shown):
            x0 = gap + (i % cols) * cell_w
            y0 = gap + (i // cols) * cell_h
            try:
                png, _ = doc.render(page, dpi=dpi)
                tile = Image.open(io.BytesIO(png)).convert("RGB")
                tile.thumbnail((thumb, thumb))
                canvas.paste(tile, (x0, y0))
                draw.rectangle([(x0 - 1, y0 - 1),
                                (x0 + tile.width, y0 + tile.height)],
                               outline=(160, 160, 160))
            except Exception:                     # noqa: BLE001 - empty tile
                draw.rectangle([(x0, y0), (x0 + thumb, y0 + thumb)],
                               outline=(160, 160, 160))
            draw.text((x0, y0 + thumb + 2), f"p. {page}", fill=(0, 0, 0),
                      font=font)
        buf = io.BytesIO()
        canvas.save(buf, format="PNG")
        sheets.append((buf.getvalue(), {
            "pages": list(shown), "columns": cols, "thumb_px": thumb,
            "legend": NUMBER_SHEET_LEGEND}))
    return sheets


def document_windows(pages: Sequence[int], window: int,
                     overlap: int) -> List[List[int]]:
    """``pages`` cut into overlapping windows, covering every page.

    The last window is not padded backwards to a full size: a report whose
    length is not a multiple of the step ends with a short window, and a
    short window is cheaper, not wrong. Every page appears in at least one
    window, which is what makes "a page with no entry is unresolved" a
    statement about the model rather than about the arithmetic.
    """
    wanted = list(pages)
    if not wanted:
        return []
    size = max(1, int(window))
    step = max(1, size - max(0, int(overlap)))
    out: List[List[int]] = []
    start = 0
    while start < len(wanted):
        out.append(wanted[start:start + size])
        if start + size >= len(wanted):
            break
        start += step
    return out


#: What a gateway says when the REQUEST BODY, not the image count, is what
#: it will not take. Matched case-insensitively against the exception's text
#: because the wording belongs to whatever sits in front of the model -- the
#: owner's gateway answers "The page was not displayed because the request
#: entity is too large", an HTTP layer may say only 413 -- and because the
#: exception CLASS is the SDK's generic status error, which carries every
#: other refusal too.
PAYLOAD_TOO_LARGE_MARKERS: Tuple[str, ...] = (
    "too large", "request entity", "413",
)


def is_payload_too_large(exc: BaseException) -> bool:
    """Whether this exception is the gateway refusing the request BODY.

    A refused request bills nothing -- no model looked at it -- so a window
    that splits and retries costs time and no money. That is what makes
    splitting the right answer here and a plain failure the right answer to
    everything else: a rate limit, a refused parameter and an expired token
    are all things a smaller window would hit just as hard.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in PAYLOAD_TOO_LARGE_MARKERS)


def _halves(shown: Sequence[int]) -> Tuple[List[int], List[int]]:
    """One window as two, as near equal as an odd length allows."""
    pages = list(shown)
    cut = len(pages) // 2
    return pages[:cut], pages[cut:]


def _context_text(doc, outline: Any, max_chars: int) -> str:
    """What the document prints about itself, as lines a model reads."""
    from planlens.document.roles import document_outline

    from report_ingest.triage import outline_text

    if outline is None:
        outline = document_outline(doc)
    return outline_text(outline, max_chars=max_chars)


# -- the gates --------------------------------------------------------------

def _clip(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number:                       # NaN
        return 0.0
    return min(1.0, max(0.0, number))


def _accept(answer: Any, page: int, qa: List[Dict[str, Any]]
            ) -> VisionPageLabel:
    """One answer through the Python gates, onto one page.

    ``page`` is authoritative: in page mode it is the page that was rendered,
    and in sheet mode it is the page the entry was matched to. A label
    outside the vocabulary becomes ``other`` and says so; a confidence
    outside 0 to 1 is clipped.
    """
    label = str(getattr(answer, "label", "") or "")
    reason = str(getattr(answer, "reason", "") or "")
    if label not in LABEL_DEFINITIONS:
        qa.append({"page": page, "note": f"label {label!r} is not in the "
                                         f"vocabulary; recorded as 'other'"})
        label = "other"
    raw = getattr(answer, "confidence", 0.0)
    confidence = _clip(raw)
    try:
        if float(raw) != confidence:
            qa.append({"page": page,
                       "note": f"confidence {raw} clipped to {confidence}"})
    except (TypeError, ValueError):
        qa.append({"page": page,
                   "note": f"confidence {raw!r} is not a number; read as 0.0"})
    return VisionPageLabel(page=page, label=label, confidence=confidence,
                           reason=reason)


# -- the pass ---------------------------------------------------------------

def classify_pages_by_vision(doc, engine: Engine, *,
                             pages: Any = None,
                             mode: str = "page",
                             dpi: float = DEFAULT_DPI,
                             budget: Optional[int] = None,
                             outline_context: bool = False,
                             sheet_pages: int = SHEET_PAGES,
                             outline: Any = None,
                             max_context_chars: int = MAX_CONTEXT_CHARS,
                             detail: Optional[str] = None,
                             images_per_call: int = MAX_IMAGES_PER_CALL,
                             window: int = DOCUMENT_WINDOW,
                             overlap: int = DOCUMENT_OVERLAP,
                             strip_sheets_max: int = STRIP_SHEETS_MAX,
                             fallback: bool = True
                             ) -> VisionLabels:
    """Label pages from their pictures alone.

    Parameters
    ----------
    doc
        An open :class:`planlens.document.Document`.
    engine
        Anything satisfying :class:`report_ingest.engine.Engine`. The
        experiment is aimed at the cheapest Funhouse tier
        (``funhouse-gpt-low``), which is why it exists at all.
    pages
        Which pages to label: ``None`` for all of them, or anything
        planlens accepts -- an index, ``"0-9,14"``, or a sequence.
    mode
        ``"page"`` for one call per page, ``"sheet"`` for one call per
        contact sheet of ``sheet_pages`` pages, ``"document"`` for a window
        of full-size pages with the whole report in thumbnail beside them.
    dpi
        What a page is rendered at in page and document mode. See the module
        docstring for why the default is 100.
    budget
        A ceiling on MODEL CALLS. Every page past it comes back
        ``unresolved`` rather than unlabelled-and-unmentioned.
    outline_context
        Prepend what the document prints about itself to every call.
    sheet_pages
        Pages per contact sheet in sheet mode.
    outline
        :func:`planlens.document.roles.document_outline` when the caller
        already has it; only read when ``outline_context`` is on.
    detail
        OpenAI's ``image_url.detail`` for every picture this pass sends:
        ``"low"``, ``"high"``, ``"auto"``, or ``None`` to leave it unset and
        get the provider's own default. ``"low"`` is about 85 tokens an
        image instead of a page's four tiles.
    images_per_call
        The endpoint's ceiling on images in one request. The default is the
        50 measured on the owner's cluster; document mode never exceeds it.
    window, overlap
        Full-size pages in one document-mode call, and how many of them the
        next call sees again. ``window`` is lowered when the strip leaves it
        no room under ``images_per_call``.
    strip_sheets_max
        The most contact sheets of the whole report one document-mode call
        carries. A report too long to show all of itself sends the sheets
        nearest the window.
    fallback
        After a sheet or document pass, give every page still unresolved ONE
        page-mode call, budget allowing. A sheet or a window that skipped a
        page is the commonest way a page goes unlabelled -- on the cluster
        on 2026-09-20 a 151-page document run left three pages unanswered
        because no later window covered them -- and one call a page is a
        cheap answer to it. Off leaves those pages unresolved, which is what
        every run before this one did.
    """
    from planlens.document.document import parse_pages

    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; the modes are {list(MODES)}")
    if detail is not None and detail not in DETAIL_LEVELS:
        raise ValueError(f"unknown detail {detail!r}; the levels are "
                         f"{list(DETAIL_LEVELS)}")
    wanted = parse_pages(pages, doc.n_pages)
    system = vision_system(mode)
    context = (_context_text(doc, outline, max_context_chars)
               if outline_context else "")

    labels: List[VisionPageLabel] = []
    unresolved: List[Dict[str, Any]] = []
    qa: List[Dict[str, Any]] = []
    spent: Dict[str, Any] = {
        "calls": 0, "input_tokens": 0, "output_tokens": 0,
        "cache_read_tokens": 0, "seconds": 0.0, "dollars": 0.0,
        "mode": mode, "windows": 0, "strip_sheets": 0,
        "detail": detail or "",
        # A document window the gateway refused on size and this pass halved,
        # and a page a sheet or a window left out that one page-mode call
        # then answered. Both stay 0 in a run that needed neither.
        "splits": 0, "fallback_pages": 0,
    }
    model_name = ""
    stopped = False

    def charge(reply) -> None:
        nonlocal model_name
        spent["calls"] += 1
        spent["input_tokens"] += reply.usage.input_tokens
        spent["output_tokens"] += reply.usage.output_tokens
        spent["cache_read_tokens"] += reply.usage.cache_read_tokens
        spent["seconds"] += reply.seconds
        spent["dollars"] += reply.usage.dollars(reply.model)
        model_name = reply.model or model_name

    def out_of_budget() -> bool:
        return budget is not None and spent["calls"] >= int(budget)

    if mode == "document":
        stopped = _document_pass(
            doc, engine, wanted, system, context, dpi, detail,
            labels, unresolved, qa, charge, out_of_budget, spent,
            images_per_call=images_per_call, window=window, overlap=overlap,
            strip_sheets_max=strip_sheets_max, budget=budget)
    else:
        chunks: List[List[int]]
        if mode == "page":
            chunks = [[page] for page in wanted]
        else:
            step = max(1, int(sheet_pages))
            chunks = [wanted[i:i + step] for i in range(0, len(wanted), step)]

        for chunk in chunks:
            if out_of_budget():
                stopped = True
                for page in chunk:
                    unresolved.append({
                        "page": page,
                        "why": f"the budget of {budget} model call(s) was "
                               f"spent before this page"})
                continue
            if mode == "page":
                _one_page(doc, engine, chunk[0], system, context, dpi,
                          labels, unresolved, qa, charge, len(wanted), detail)
            else:
                _one_sheet(doc, engine, chunk, system, context,
                           labels, unresolved, qa, charge, detail)

    if fallback and mode != "page" and unresolved:
        # The fallback is page mode, so it gets page mode's prompt: the
        # document and sheet prompts tell the model to answer for every page
        # it was shown, which is one page here.
        stopped = _fallback_pass(
            doc, engine, vision_system("page"), context, dpi, detail, wanted,
            labels, unresolved, qa, charge, out_of_budget, spent,
            budget) or stopped

    spent["seconds"] = round(spent["seconds"], 1)
    spent["dollars"] = round(spent["dollars"], 4)
    return VisionLabels(
        labels=labels,
        cost=spent,
        unresolved=unresolved,
        qa=qa,
        mode=mode,
        dpi=float(dpi),
        outline_context=bool(outline_context),
        pages_asked=len(wanted),
        model_calls=spent["calls"],
        budget=None if budget is None else int(budget),
        stopped_on_budget=stopped,
        model=model_name or getattr(engine, "name", ""),
    )


def _ask(engine: Engine, content: Sequence[Any], system: str,
         output_format: Any, charge) -> Any:
    """One call: the words, the pictures, the shape the answer must take.

    The pictures ride as image BLOCKS inside the user turn rather than in
    the engine's ``images`` argument, because document mode interleaves
    captions with pictures and because an image block is the only place a
    ``detail`` can be attached to one image rather than to all of them.
    """
    reply = engine.complete([user(*content)], system=system,
                            output_format=output_format)
    charge(reply)
    return reply


def _one_page(doc, engine: Engine, page: int, system: str, context: str,
              dpi: float, labels: List[VisionPageLabel],
              unresolved: List[Dict[str, Any]], qa: List[Dict[str, Any]],
              charge, n_pages: int, detail: Optional[str] = None) -> None:
    png, info = doc.render(page, dpi=float(dpi))
    body = "\n".join(filter(None, [
        ("WHAT THIS DOCUMENT PRINTS ABOUT ITSELF\n" + context + "\n"
         if context else ""),
        f"The picture is page {page} of this document, rendered at "
        f"{info['dpi']:.0f} dpi. The document has {doc.n_pages} pages. Say "
        f"what page {page} is.",
    ]))
    reply = _ask(engine,
                 [text_block(body), image_block(encode_for_vision(png),
                                                detail)],
                 system, VisionPageAnswer, charge)
    answer = reply.parsed
    if answer is None:
        unresolved.append({"page": page,
                           "why": f"the model returned no structured answer "
                                  f"(stop_reason {reply.stop_reason!r})"})
        return
    got = getattr(answer, "page", page)
    try:
        if int(got) != page:
            qa.append({"page": page,
                       "note": f"the answer named page {got}; it was asked "
                               f"about page {page} and is recorded there"})
    except (TypeError, ValueError):
        pass
    labels.append(_accept(answer, page, qa))


def _fallback_pass(doc, engine: Engine, system: str, context: str,
                   dpi: float, detail: Optional[str],
                   wanted: Sequence[int], labels: List[VisionPageLabel],
                   unresolved: List[Dict[str, Any]],
                   qa: List[Dict[str, Any]], charge, out_of_budget,
                   spent: Dict[str, Any], budget: Optional[int]) -> bool:
    """One page-mode call for every page the pass left unresolved.

    WHY IT IS WORTH A CALL. In sheet and document mode a page goes
    unresolved because the REPLY left it out, not because the page could not
    be read: the model answered for the other thirty-five pages in the
    window and simply did not mention this one. Document mode's overlap is
    the first remedy and it is not always enough -- on the cluster on
    2026-09-20 three pages of a 151-page report ended unlabelled because the
    window that overlapped each of them skipped it too. Asking about that
    one page on its own is the mode that never skips a page by construction,
    and it costs one call.

    WHAT IT DOES NOT DO. It never overwrites an answer, because it only ever
    sees pages nothing answered for; a page it still cannot settle keeps its
    unresolved entry, with the fallback's own reason if the fallback is what
    failed; and it spends the same budget as everything else, so a run that
    was already capped does not quietly buy more calls. Returns whether the
    budget stopped it.
    """
    stopped = False
    kept: List[Dict[str, Any]] = []
    for entry in list(unresolved):
        try:
            page = int(entry["page"])
        except (KeyError, TypeError, ValueError):     # not a page: keep it
            kept.append(entry)
            continue
        if out_of_budget():
            stopped = True
            kept.append(entry)
            continue
        got: List[VisionPageLabel] = []
        missed: List[Dict[str, Any]] = []
        _one_page(doc, engine, page, system, context, dpi, got, missed, qa,
                  charge, len(wanted), detail)
        if not got:
            kept.append(missed[0] if missed else entry)
            continue
        labels.extend(got)
        spent["fallback_pages"] = int(spent.get("fallback_pages") or 0) + 1
        qa.append({"page": page,
                   "note": "the pass left this page unresolved; it was "
                           "answered by a page-mode fallback call"})
    unresolved[:] = kept
    order = {page: i for i, page in enumerate(wanted)}
    labels.sort(key=lambda row: order.get(row.page, row.page))
    return stopped


def _one_sheet(doc, engine: Engine, chunk: Sequence[int], system: str,
               context: str, labels: List[VisionPageLabel],
               unresolved: List[Dict[str, Any]], qa: List[Dict[str, Any]],
               charge, detail: Optional[str] = None) -> None:
    shown = list(chunk)
    sheets = render_number_sheets(doc, shown,
                                  columns=min(SHEET_COLUMNS, len(shown)),
                                  thumb_px=SHEET_THUMB_PX,
                                  per_sheet=len(shown))
    if not sheets:
        for page in shown:
            unresolved.append({"page": page,
                               "why": "no contact sheet could be rendered"})
        return
    png, info = sheets[0]
    body = "\n".join(filter(None, [
        ("WHAT THIS DOCUMENT PRINTS ABOUT ITSELF\n" + context + "\n"
         if context else ""),
        f"The picture is a contact sheet of pages "
        f"{', '.join(str(p) for p in shown)} of this document, which has "
        f"{doc.n_pages} pages. {info.get('legend', '')} Say what each of "
        f"those {len(shown)} pages is, one answer per page.",
    ]))
    reply = _ask(engine,
                 [text_block(body), image_block(encode_for_vision(png),
                                                detail)],
                 system, VisionSheetAnswer, charge)
    answer = reply.parsed
    if answer is None:
        for page in shown:
            unresolved.append({
                "page": page,
                "why": f"the model returned no structured answer for this "
                       f"sheet (stop_reason {reply.stop_reason!r})"})
        return

    by_page: Dict[int, Any] = {}
    for entry in (getattr(answer, "pages", None) or []):
        try:
            page = int(getattr(entry, "page"))
        except (TypeError, ValueError, AttributeError):
            qa.append({"page": None,
                       "note": "an entry on this sheet named no page index "
                               "and was dropped"})
            continue
        if page not in shown:
            qa.append({"page": page,
                       "note": f"the reply named page {page}, which is not on "
                               f"this sheet; dropped rather than guessed"})
            continue
        if page in by_page:
            qa.append({"page": page,
                       "note": "the reply answered for this page twice; the "
                               "first answer is the one kept"})
            continue
        by_page[page] = entry

    for page in shown:
        entry = by_page.get(page)
        if entry is None:
            unresolved.append({
                "page": page,
                "why": "the reply left this page out of the sheet it was "
                       "shown on"})
            continue
        labels.append(_accept(entry, page, qa))


# -- document mode ----------------------------------------------------------

def _ranges(pages: Sequence[int]) -> str:
    """``[33, 34, 35, 40]`` as ``"33-35, 40"``.

    A window is normally a consecutive run, and printing 36 numbers where
    two will do is 36 numbers the model has to read past. Where the caller
    asked for a broken set of pages, the breaks show.
    """
    wanted = sorted(set(int(p) for p in pages))
    if not wanted:
        return ""
    out: List[str] = []
    start = previous = wanted[0]
    for page in wanted[1:]:
        if page == previous + 1:
            previous = page
            continue
        out.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = page
    out.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(out)


def _runs(decided: Dict[int, str]) -> List[str]:
    """``page: label`` collapsed into runs, one line each.

    Run-length, not one line a page, for two reasons. It is far shorter on a
    400-page report, where the decided list would otherwise be most of the
    message. And it says the thing the model is being shown it FOR: that
    pages 61 to 118 are all boring logs is the shape of an appendix, which
    is exactly the evidence that settles the silent page at 119.
    """
    if not decided:
        return []
    pages = sorted(decided)
    out: List[str] = []
    start = previous = pages[0]
    label = decided[start]
    for page in pages[1:]:
        if page == previous + 1 and decided[page] == label:
            previous = page
            continue
        span = str(start) if start == previous else f"{start}-{previous}"
        out.append(f"  {span}: {label}")
        start = previous = page
        label = decided[page]
    span = str(start) if start == previous else f"{start}-{previous}"
    out.append(f"  {span}: {label}")
    return out


def _strip_sheets(doc, qa: List[Dict[str, Any]]
                  ) -> List[Tuple[bytes, Dict[str, Any]]]:
    """Contact sheets of the WHOLE report, captioned with page indexes only.

    A failure here is a note and an empty strip rather than a dead run: the
    strip is orientation, and a window of stamped full-size pages is still a
    usable call without it.
    """
    try:
        return render_number_sheets(doc, None, columns=STRIP_COLUMNS,
                                    thumb_px=STRIP_THUMB_PX,
                                    per_sheet=STRIP_PER_SHEET)
    except Exception as exc:                  # noqa: BLE001 - reported as QA
        qa.append({"page": None,
                   "note": f"no contact sheets of the whole report could be "
                           f"rendered ({type(exc).__name__}: {exc}); the "
                           f"windows went out without the strip"})
        return []


def _sheet_centre(sheet: Tuple[bytes, Dict[str, Any]]) -> float:
    shown = list((sheet[1] or {}).get("pages") or [])
    return (sum(shown) / len(shown)) if shown else 0.0


def _strip_for(sheets: Sequence[Tuple[bytes, Dict[str, Any]]],
               shown: Sequence[int], n_strip: int
               ) -> List[Tuple[bytes, Dict[str, Any]]]:
    """The sheets this window carries: all of them, or the nearest ones.

    A report longer than ``STRIP_SHEETS_MAX * STRIP_PER_SHEET`` pages cannot
    show all of itself inside the image cap, so it shows the part of itself
    the window is standing in. Which is the part that decides a page: the
    divider that opens this appendix is a few pages back, not 500.
    """
    if n_strip <= 0 or not sheets:
        return []
    if len(sheets) <= n_strip:
        return list(sheets)
    centre = (shown[0] + shown[-1]) / 2.0 if shown else 0.0
    nearest = sorted(range(len(sheets)),
                     key=lambda i: abs(_sheet_centre(sheets[i]) - centre))
    return [sheets[i] for i in sorted(nearest[:n_strip])]


def _document_message(doc, shown: Sequence[int],
                      strip: Sequence[Tuple[bytes, Dict[str, Any]]],
                      context: str, decided: Dict[int, str], dpi: float,
                      detail: Optional[str], n_wanted: int
                      ) -> List[Dict[str, Any]]:
    """One window's user turn: the words, the strip, the pages.

    The order is deliberate. The words say what is coming and what to answer
    for; the thumbnails come next, each after a caption naming the pages it
    shows, so a sheet is never an unlabelled picture; the full-size pages
    come last, closest to the answer, each stamped with its own number.
    """
    decided_lines = _runs({p: label for p, label in decided.items()
                           if p < shown[0]})
    body = "\n".join(filter(None, [
        ("WHAT THIS DOCUMENT PRINTS ABOUT ITSELF\n" + context + "\n"
         if context else ""),
        ("LABELS DECIDED SO FAR (earlier pages of this same report, as "
         "runs)\n" + "\n".join(decided_lines) + "\n" if decided_lines else ""),
        f"This document has {doc.n_pages} pages and {n_wanted} of them are "
        f"being labelled. {len(strip)} thumbnail contact sheet(s) of the "
        f"report follow, for orientation only, and then pages "
        f"{_ranges(shown)} at full size, rendered at {dpi:.0f} dpi and each "
        f"stamped 'p. N' at the top-left corner. Say what each of those "
        f"{len(shown)} pages is, one answer per page, using the stamped "
        f"number. Do not answer for any page you have seen only as a "
        f"thumbnail.",
    ]))
    content: List[Dict[str, Any]] = [text_block(body)]
    for png, info in strip:
        pages_on = list((info or {}).get("pages") or [])
        content.append(text_block(
            f"Thumbnails of pages {_ranges(pages_on)} of this report. "
            f"{(info or {}).get('legend', '')}".strip()))
        content.append(image_block(encode_for_vision(png), detail))
    content.append(text_block(
        f"The {len(shown)} full-size pages now follow in order, pages "
        f"{_ranges(shown)}. Answer for these and for no others."))
    for page in shown:
        png, _info = doc.render(page, dpi=float(dpi))
        # Stamped first, then re-encoded: the stamp keeps its PNG contract
        # and the picture that travels is the JPEG of the stamped page.
        content.append(image_block(
            encode_for_vision(stamp_page_number(png, page)), detail))
    return content


def _document_pass(doc, engine: Engine, wanted: Sequence[int], system: str,
                   context: str, dpi: float, detail: Optional[str],
                   labels: List[VisionPageLabel],
                   unresolved: List[Dict[str, Any]],
                   qa: List[Dict[str, Any]], charge, out_of_budget,
                   spent: Dict[str, Any], *, images_per_call: int,
                   window: int, overlap: int, strip_sheets_max: int,
                   budget: Optional[int]) -> bool:
    """Every page labelled with the whole report in view. Returns whether
    the budget stopped the run.

    THE ARITHMETIC, because it is the whole design. One call carries the
    strip and the window, and their sum may not exceed ``images_per_call``.
    The strip is capped first -- at ``strip_sheets_max``, at what the
    document actually has, and at whatever leaves the window
    :data:`DOCUMENT_MIN_WINDOW` pages -- and the window takes what is left,
    up to the ``window`` asked for. So a 151-page report sends 4 sheets and
    36 pages (40 images) and a 729-page report sends 12 sheets and 36 pages
    (48): under the cap either way, and the long report is the one that
    gives up seeing all of itself rather than the one that gives up pages.

    WHAT IS RESOLVED WHEN. Answers accumulate into one map across the
    windows and the LATER answer wins, because a later window was made with
    more of the report already decided. A page nothing answered for is
    unresolved at the end, with the reason its last chance gave -- which is
    why a page skipped in one window and answered in the next is simply
    answered, and why a budget that ran out is visible as the reason on
    every page it cost.

    AND THE WINDOW SPLITS ITSELF. The image cap is not the only ceiling: the
    gateway also refuses a request whose BODY is too big, which a report of
    scanned pages reaches long before 50 images (2026-09-20: every window of
    a 156-page scanned report came back "the request entity is too large",
    while a text-page report of the same length went through). So the
    windows are driven from a QUEUE rather than a list. A window the gateway
    refuses on size is halved, both halves go back at the front, the working
    size becomes the half, and every page still queued behind them is
    re-windowed at that size -- so one refusal narrows the rest of the
    report instead of being paid for again and again. A refused request
    bills nothing, so a split costs time alone. A window already at or under
    :data:`DOCUMENT_MIN_SPLIT` that is still refused raises: at four pages
    the body is small and the refusal means something else.
    """
    from collections import deque

    sheets = _strip_sheets(doc, qa)
    cap = max(1, int(images_per_call))
    n_strip = min(len(sheets), max(0, int(strip_sheets_max)),
                  max(0, cap - DOCUMENT_MIN_WINDOW))
    per_window = max(1, min(int(window), cap - n_strip))
    windows = document_windows(wanted, per_window, overlap)
    spent["windows"] = len(windows)
    spent["strip_sheets"] = n_strip
    spent["window_pages"] = per_window
    spent["images_per_call"] = cap

    answered: Dict[int, VisionPageLabel] = {}
    why_missing: Dict[int, str] = {}
    stopped = False
    queue = deque(windows)
    covered = 0

    while queue:
        shown = queue.popleft()
        if out_of_budget():
            stopped = True
            covered += 1
            for page in shown:
                why_missing[page] = (f"the budget of {budget} model call(s) "
                                     f"was spent before this window")
            continue
        strip = _strip_for(sheets, shown, n_strip)
        content = _document_message(doc, shown, strip, context,
                                    {p: row.label
                                     for p, row in answered.items()},
                                    dpi, detail, len(wanted))
        n_images = sum(1 for b in content if b.get("type") == "image")
        if n_images > cap:                       # never reachable; say so
            raise AssertionError(
                f"a document-mode call was built with {n_images} images, "
                f"over the cap of {cap}")
        try:
            reply = _ask(engine, content, system, VisionSheetAnswer, charge)
        except Exception as exc:                  # noqa: BLE001 - re-raised
            if not (is_payload_too_large(exc)
                    and len(shown) > DOCUMENT_MIN_SPLIT):
                raise
            first, second = _halves(shown)
            size = max(len(first), len(second))
            # Everything still queued is re-cut to the new size, so the
            # refusal is paid for once rather than at every window after it.
            seen_pending: set = set()
            pending: List[int] = []
            for later in queue:                   # the queue is in page order
                for page in later:
                    if page in seen_pending:      # the overlap repeats pages
                        continue
                    seen_pending.add(page)
                    pending.append(page)
            queue = deque([first, second]
                          + document_windows(pending, size, overlap))
            spent["splits"] = int(spent.get("splits") or 0) + 1
            spent["window_pages"] = size
            qa.append({"page": None,
                       "note": f"the gateway refused the window of pages "
                               f"{_ranges(shown)} as too large a request; it "
                               f"was split and the window size is now {size} "
                               f"page(s)"})
            continue
        covered += 1
        answer = reply.parsed
        if answer is None:
            for page in shown:
                why_missing[page] = (
                    f"the model returned no structured answer for the window "
                    f"holding this page (stop_reason {reply.stop_reason!r})")
            continue

        by_page: Dict[int, Any] = {}
        for entry in (getattr(answer, "pages", None) or []):
            try:
                page = int(getattr(entry, "page"))
            except (TypeError, ValueError, AttributeError):
                qa.append({"page": None,
                           "note": "an entry in this window named no page "
                                   "index and was dropped"})
                continue
            if page not in shown:
                qa.append({"page": page,
                           "note": f"the reply named page {page}, which is "
                                   f"not a full-size page of this window; "
                                   f"dropped rather than guessed"})
                continue
            if page in by_page:
                qa.append({"page": page,
                           "note": "the reply answered for this page twice "
                                   "within one window; the first answer is "
                                   "the one kept"})
                continue
            by_page[page] = entry

        for page in shown:
            entry = by_page.get(page)
            if entry is None:
                why_missing[page] = ("the reply left this page out of the "
                                     "window it was shown in")
                continue
            # A later window saw more of the report already decided, so its
            # answer replaces an earlier one rather than being dropped.
            answered[page] = _accept(entry, page, qa)

    # The windows the run ENDED with, not the ones it planned: a split
    # makes two where the plan had one, and the results table reads this
    # column as "one window is one call".
    spent["windows"] = covered
    for page in wanted:
        row = answered.get(page)
        if row is None:
            unresolved.append({
                "page": page,
                "why": why_missing.get(
                    page, "no window produced an answer for this page")})
            continue
        labels.append(row)
    return stopped
