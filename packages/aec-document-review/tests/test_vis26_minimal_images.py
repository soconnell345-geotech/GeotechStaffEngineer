"""BACKLOG VIS-26: the minimal arm on an uploaded picture.

Its system prompt said an uploaded picture "is shown" with
``analyze_image``, which answers by a vision call (only a zoom on it is
shown); and ``page_count`` answered an image with a PDF reader's error, or
(the PyMuPDF on this laptop) with a page in points at the dpi the file
declares, never the picture's own pixels.
A PDF's ``page_count`` is what it was. Pictures made here with Pillow, one
a JPEG with an EXIF rotation tag.
"""

import io
import json

import pytest

pymupdf = pytest.importorskip("pymupdf")
Image = pytest.importorskip("PIL.Image")

from aec_document_review import minimal_agent  # noqa: E402
from aec_document_review.minimal_agent import MINIMAL_PROMPT  # noqa: E402


def _jpeg_turned(w, h):
    im = Image.new("RGB", (w, h), "white")
    ex = Image.Exif()
    ex[0x0112] = 6                    # shown a quarter turn clockwise
    buf = io.BytesIO()
    im.save(buf, format="JPEG", exif=ex.tobytes())
    return buf.getvalue()


def _png(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, format="PNG")
    return buf.getvalue()


def _pdf(sizes):
    doc = pymupdf.open()
    for w, h in sizes:
        doc.new_page(width=w, height=h)
    return doc.tobytes()


def _count(attachments, source):
    tool = minimal_agent._page_count_tool(attachments)
    return tool.invoke({"source": source})


def test_the_prompt_no_longer_says_a_picture_is_shown():
    assert "An uploaded picture is shown with `analyze_image`" \
        not in MINIMAL_PROMPT
    assert "An uploaded picture is not shown to you whole" in MINIMAL_PROMPT
    assert "`render_region` zooms on it" in MINIMAL_PROMPT
    assert len(MINIMAL_PROMPT) < 3000


@pytest.mark.parametrize("name,data,size", [
    ("photo.jpg", _jpeg_turned(400, 300), [300, 400]),
    ("shot.png", _png(640, 480), [640, 480]),
])
def test_page_count_on_a_picture_says_one_page_in_pixels(name, data, size):
    out = json.loads(_count({name: data}, name))
    assert out == {"source": name, "pages": 1, "image_px": size,
                   "note": minimal_agent.IMAGE_PAGE_NOTE}


def test_page_count_on_a_pdf_is_what_it_was():
    one = _pdf([(612, 792), (612, 792)])
    two = _pdf([(612, 792), (1224, 792)])
    assert _count({"a.pdf": one}, "a.pdf") == json.dumps(
        {"source": "a.pdf", "pages": 2, "page_size_pt": [612, 792]})
    assert _count({"b.pdf": two}, "b.pdf") == json.dumps(
        {"source": "b.pdf", "pages": 2,
         "page_sizes_pt": [[612, 792], [1224, 792]]})


def test_page_count_on_neither_is_still_an_error():
    out = json.loads(_count({"notes.txt": b"made up text, not a file"},
                            "notes.txt"))
    assert set(out) == {"error"}
