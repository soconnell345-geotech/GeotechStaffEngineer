"""BACKLOG VIS-24: vision side calls asked for page coordinates.

A side call sees one image, never the page. Asked for positions "in PDF
points" it answered in whatever frame it guessed, and the numbers it wrote
were used as if measured. With ``GEOTECH_VISION_PIXEL_REQUESTS`` on, a
request for positions in page units is put to the vision model as one for
pixel boxes, which the tool converts to PDF points itself, and page
coordinates an answer states that the tool did not convert are said, once,
to be the vision model's own estimate. Off, every prompt and result is as
before.

Made-up pages and pictures built here, and a fake vision model that finds a
drawn square in the image it is sent: no model, no network.
"""

from __future__ import annotations

import io
import json
import re
import threading

import pytest

pymupdf = pytest.importorskip("pymupdf")
Image = pytest.importorskip("PIL.Image")

from aec_agent_core import review_flags, vision_tools, vision_view  # noqa: E402
from aec_agent_core.vision_tools import dispatch_extended_tool  # noqa: E402

ENV = review_flags.PIXEL_REQUESTS_ENV
#: The black square on the made-up sheet, in PDF points (top-left origin).
SQUARE = (300.0, 200.0, 320.0, 220.0)
CENTRE = (310.0, 210.0)
#: Within this many points of the square: a pixel or two of the image sent.
TOL = 1.5
ASK_POINTS = "Give the centre of the black square in PDF page points."
ASK_PLAIN = "Where is the black square? Give its location."
TILE = re.compile(r"tile row (\d) of \d, column (\d)")
LOCATION = "If you give the location of anything in it"
#: Made-up prose that states page coordinates the tool did not convert.
STATED = "Its centre is at (612.4, 396.0) pt on the page."
#: Made-up prose full of numbers that are not page coordinates.
ORDINARY = ("B-7 STA 12+50, EL 101.3 m; B-8 STA 14+00, EL 99.8 m. Sheet "
            "C-101, scale 1:100. The sheet is 1224 x 792 pt; general notes "
            "lettering about 8 pt tall. Boring B-7 is at N 1234.5, E 5678.9 "
            "(survey coordinates). A 12 in x 18 in footing.")
NEW_KEYS = ("page_points", "page_points_cut", "page_points_note",
            "coordinates_note")


def _sheet_pdf() -> bytes:
    """A made-up sheet with one black square on it."""
    d = pymupdf.open()
    page = d.new_page(width=1224, height=792)
    page.draw_rect(pymupdf.Rect(*SQUARE), color=(0, 0, 0), fill=(0, 0, 0))
    data = d.tobytes()
    d.close()
    return data


def _picture_png() -> bytes:
    """A made-up picture: one black square on white."""
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 800, 600), False)
    pix.set_rect(pix.irect, (255, 255, 255))
    pix.set_rect(pymupdf.IRect(300, 200, 340, 240), (0, 0, 0))
    return pix.tobytes("png")


class Finder:
    """Finds the black square in the image it is sent and gives its pixel
    box (``boxes=False``: gives none), then ``extra`` (made-up prose), in
    the looks ``extra_in`` names (``whole``, ``r1c1``, ...; all when
    ``None``)."""

    def __init__(self, extra="", extra_in=None, boxes=True):
        self.lock = threading.Lock()
        self.prompts = {}
        self.seen = []
        self.extra, self.extra_in, self.boxes = extra, extra_in, boxes

    def analyze_image(self, image, prompt=""):
        m = TILE.search(prompt)
        who = f"r{m.group(1)}c{m.group(2)}" if m else "whole"
        with self.lock:
            self.prompts[who] = prompt
            self.seen.append(prompt)
        img = Image.open(io.BytesIO(image)).convert("L")
        dark = img.point(lambda v: 255 if v < 100 else 0).getbbox()
        lines = []
        if dark and self.boxes:
            lines.append("Black square px=[{}, {}, {}, {}]".format(*dark))
        elif not dark:
            lines.append("No square in this view.")
        if self.extra and (self.extra_in is None or who in self.extra_in):
            lines.append(self.extra)
        return "\n".join(lines) or "Nothing to say."


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    for env in (ENV, review_flags.TILE_ANSWERS_ENV, vision_view.POLICY_ENV,
                vision_tools.VISION_CAP_ENV):
        monkeypatch.delenv(env, raising=False)
    vision_tools.clear_repeat_reads()
    vision_tools.clear_read_log()
    yield
    vision_tools.clear_repeat_reads()
    vision_tools.clear_read_log()


def _zoom(engine, prompt, **args):
    raw = dispatch_extended_tool(
        "render_region",
        {"attachment_key": "made_up.pdf", "page": 0, "prompt": prompt,
         "bbox": [250, 150, 400, 260], **args},
        engine=engine, attachments={"made_up.pdf": _sheet_pdf()})
    vision_tools.clear_repeat_reads()
    return raw


def _page(engine, prompt, tiles="off"):
    raw = dispatch_extended_tool(
        "analyze_pdf_page",
        {"attachment_key": "made_up.pdf", "page": 0, "prompt": prompt,
         "tiles": tiles},
        engine=engine, attachments={"made_up.pdf": _sheet_pdf()})
    vision_tools.clear_repeat_reads()
    return raw


def _tiled(engine, prompt):
    return _page(engine, prompt, tiles="2")


def _image(engine, prompt):
    raw = dispatch_extended_tool(
        "analyze_image",
        {"attachment_key": "made_up.png", "prompt": prompt, "tiles": "off"},
        engine=engine, attachments={"made_up.png": _picture_png()})
    vision_tools.clear_repeat_reads()
    return raw


def _image_zoom(engine, prompt):
    raw = dispatch_extended_tool(
        "render_region",
        {"attachment_key": "made_up.png", "prompt": prompt,
         "bbox": [250, 150, 400, 300]},
        engine=engine, attachments={"made_up.png": _picture_png()})
    vision_tools.clear_repeat_reads()
    return raw


def _near(a, b):
    return all(abs(float(x) - float(y)) <= TOL for x, y in zip(a, b))


# -- the switch ----------------------------------------------------------------

def test_the_switch_is_off_unless_set_and_kept_with_the_later_switches(
        monkeypatch):
    assert ENV == "GEOTECH_VISION_PIXEL_REQUESTS"
    assert not review_flags.pixel_requests()
    for off in ("", "0", "no", "off", "false"):
        monkeypatch.setenv(ENV, off)
        assert not review_flags.pixel_requests(), off
    monkeypatch.setenv(ENV, "1")
    assert review_flags.pixel_requests()
    # Beside the newest switches, apart from the two lists the geotech app's
    # own tests pin whole.
    assert ENV in review_flags.LATER_ENVS
    assert ENV not in review_flags.ALL_ENVS
    assert all(ENV not in arm for arm in review_flags.ARMS.values())


def test_the_turn_record_names_the_switch_and_an_arm_sets_it_alone(
        monkeypatch):
    """A turn's record carries ``describe()``: it says when the switch is
    on. An arm sets exactly what it names: the suite's pair turns it on; the
    baseline clears a value left in the notebook's environment."""
    assert ENV not in review_flags.describe()
    with review_flags.switches({ENV: "1"}):
        assert review_flags.pixel_requests()
        assert "GEOTECH_VISION_PIXEL_REQUESTS=1" in review_flags.describe()
    monkeypatch.setenv(ENV, "1")
    with review_flags.switches(review_flags.ARMS["baseline"]):
        assert not review_flags.pixel_requests()
    assert review_flags.pixel_requests()                # restored after


# -- what counts as a request for positions in page units ------------------------

@pytest.mark.parametrize("prompt", [
    "Give the centre of the north arrow in PDF points.",
    "List each symbol centre (x, y) in PDF page points.",
    "What are the page coordinates of the title block?",
    "Report the coordinates on the page of each boring symbol.",
    "Return the bbox of the stamp in pt.",
    "Give x, y in points for every tag.",
    "Where is the legend, in PDF units?",
    "Locate the revision cloud in PDF space and give its corners.",
    "Give the symbol's location as PDF coordinates.",
])
def test_a_request_for_positions_in_page_units_is_one(prompt):
    assert vision_tools.asks_page_positions(prompt)


@pytest.mark.parametrize("prompt", [
    # coordinates printed on the plan, not the page's
    "List the boring coordinates (northing and easting) printed on the plan.",
    # sizes in points, not positions
    "What is the lettering height in points?",
    "Give the sheet size in PDF points.",
    "Give the font size in pt of the general notes.",
    "Give the location of B-7 and the text height in points.",
    "What is the distance between the two gridlines in points?",
    # stations, elevations, sheet numbers, plain places
    "Read the station and elevation of each boring.",
    "Which sheet number is this, and what is its title?",
    "Where is the north arrow?",
    "Read page 3 and give the boring locations.",
    "Open made_up.pdf page 2 and say where the legend is.",
    "List the survey points shown on the plan.",
    "Find the point where the leader ends.",
    "",
])
def test_anything_else_is_left_alone(prompt):
    assert not vision_tools.asks_page_positions(prompt)


# -- what counts as page coordinates stated in an answer -------------------------

@pytest.mark.parametrize("text", [
    "Symbol centre: (612.4, 396.0) pt",
    "The tag sits at page coordinates [600, 380, 625, 410].",
    "x = 612 pt, y = 396 pt",
    "In PDF points: (612, 396).",
    "Centre (612, 396) in PDF points.",
    "B-7 at x=88.5, y=140.25 (PDF points)",
    "box [10, 20, 30, 40] pts",
    STATED,
])
def test_page_coordinates_in_prose_are_found(text):
    assert vision_tools.states_page_coordinates(text)


@pytest.mark.parametrize("text", [
    "B-7 STA 12+50, EL 101.3 m; B-8 STA 14+00, EL 99.8 m",
    "Sheet C-101, scale 1:100",
    "The sheet is 1224 x 792 pt.",
    "General notes lettering about 8 pt tall; two sizes, (8, 10) pt "
    "lettering.",
    "Boring B-7: N 1234.5, E 5678.9 (survey coordinates)",
    "Coordinates (1234.5, 5678.9) ft",
    "B-7 px=[12, 14, 80, 30]",
    "page coordinates px=[12, 14, 80, 30]",
    "Tag at [12, 14, 80, 30] px",
    "Tag box [12, 14, 80, 30]",
    "A 12 in x 18 in footing; 3.5 m wide strip",
    "4-6 pt lettering at (2, 3)",
    "The chart shows points (1, 2) and (3, 4) on the curve.",
    ORDINARY,
    "",
])
def test_ordinary_numbers_are_not_page_coordinates(text):
    assert not vision_tools.states_page_coordinates(text)


# -- off: byte-for-byte as before --------------------------------------------------

def _master_vision_prompt(prompt, size=None, shape=None, turn=0, cut=None,
                          positions=False):
    """``_vision_prompt`` as it was before VIS-24."""
    assert not positions
    parts = [prompt]
    if shape:
        parts.append(shape)
    if cut:
        parts.append(cut)
    return vision_view.with_location("\n\n".join(parts), size)


def _never(*_a, **_k):
    raise AssertionError("a VIS-24 rule ran with the switch off")


@pytest.mark.parametrize("read", [_zoom, _page, _tiled])
@pytest.mark.parametrize("unset", [None, "", "0"])
def test_off_every_prompt_and_result_is_as_before(monkeypatch, read, unset):
    """Off (unset, empty or 0), a request for page points with page
    coordinates in the answer gives the prompts and the result of the code
    before VIS-24, and none of its rules runs."""
    if unset is not None:
        monkeypatch.setenv(ENV, unset)
    now = Finder(extra=STATED)
    raw_now = read(now, ASK_POINTS)
    with monkeypatch.context() as m:
        m.setattr(vision_tools, "_vision_prompt", _master_vision_prompt)
        for name in ("asks_page_positions", "states_page_coordinates",
                     "_px_page_boxes", "_note_page_units"):
            m.setattr(vision_tools, name, _never)
        before = Finder(extra=STATED)
        raw_before = read(before, ASK_POINTS)
    assert raw_now == raw_before
    assert now.prompts == before.prompts
    out = json.loads(raw_now)
    assert "error" not in out, out
    assert not any(k in raw_now for k in NEW_KEYS)
    assert "_page_units" not in raw_now
    for prompt in now.seen:
        assert vision_tools.POSITIONS_AS_PIXELS not in prompt


# -- on: a request for page units is asked as pixel boxes the tool converts ------

@pytest.mark.parametrize("turn", [None, 90])
def test_on_a_zoom_asked_for_page_points_gets_them_converted(monkeypatch,
                                                             turn):
    """The zoom is asked for pixel boxes; the box it gives back is converted
    to PDF points by the tool and lands on the square, also when the image
    went turned (the turn is undone)."""
    monkeypatch.setenv(ENV, "1")
    engine = Finder()
    out = json.loads(_zoom(engine, ASK_POINTS,
                           **({"turn": turn} if turn else {})))
    assert "error" not in out, out
    (prompt,) = engine.seen
    assert prompt.startswith(
        ASK_POINTS + "\n\n" + vision_tools.POSITIONS_AS_PIXELS)
    at = [prompt.index(p) for p in (
        ASK_POINTS, vision_tools.POSITIONS_AS_PIXELS,
        vision_tools.ZOOM_ANSWER_SHAPE, LOCATION)]
    assert at == sorted(at)
    (row,) = out["page_points"]
    assert row["what"] == "Black square"
    assert _near(row["page_bbox"], SQUARE), row
    assert _near(row["centre"], CENTRE), row
    assert out["page_points_note"] == vision_tools.PAGE_POINTS_NOTE
    assert "coordinates_note" not in out
    if turn:
        assert out.get("turned")


@pytest.mark.parametrize("read", [_page, _tiled])
def test_on_a_page_read_whole_or_in_tiles_converts_too(monkeypatch, read):
    """A page read whole converts its own pixel boxes; a tiled read lists
    its merged 'found' boxes, each with where it was seen. Every look of
    the read is asked for pixel boxes."""
    monkeypatch.setenv(ENV, "1")
    engine = Finder()
    out = json.loads(read(engine, ASK_POINTS))
    assert "error" not in out, out
    assert engine.seen
    for prompt in engine.seen:
        assert prompt.count(vision_tools.POSITIONS_AS_PIXELS) == 1
    (row,) = out["page_points"]
    assert _near(row["centre"], CENTRE), row
    assert _near(row["page_bbox"], SQUARE), row
    if read is _tiled:
        assert len(engine.seen) == 5                    # whole + 2x2 tiles
        assert "r1c1" in row["seen_in"]
        assert not any("_page_units" in t for t in out["tiles"])
    else:
        assert "seen_in" not in row
    assert out["page_points_note"] == vision_tools.PAGE_POINTS_NOTE


@pytest.mark.parametrize("read", [_zoom, _page, _tiled])
def test_on_a_request_without_page_units_is_left_alone(monkeypatch, read):
    """A request for a place with no page unit already gets pixel boxes:
    its prompts and result are what they are with the switch off."""
    results = {}
    for value in (None, "1"):
        if value:
            monkeypatch.setenv(ENV, value)
        engine = Finder()
        results[value] = (read(engine, ASK_PLAIN), engine.prompts)
    assert results[None] == results["1"]
    assert "page_points" not in results["1"][0]


def test_on_an_answer_with_no_pixel_box_gives_no_page_points(monkeypatch):
    """Asked for page points, an answer that gives its own page numbers
    and no pixel box: nothing is listed as converted, and its numbers are
    said to be its own estimate."""
    monkeypatch.setenv(ENV, "1")
    engine = Finder(extra="The square's centre is (310, 210) in PDF points.",
                    boxes=False)
    out = json.loads(_zoom(engine, ASK_POINTS))
    assert "page_points" not in out
    assert out["page_points_note"] == vision_tools.NO_PAGE_POINTS_NOTE
    assert out["coordinates_note"] == vision_tools.COORDINATES_NOTE


# -- on: page coordinates stated in an answer are said once ----------------------

@pytest.mark.parametrize("read", [_zoom, _page])
def test_on_stated_page_coordinates_are_said_once(monkeypatch, read):
    monkeypatch.setenv(ENV, "1")
    out = json.loads(read(Finder(extra=STATED), ASK_PLAIN))
    assert out["coordinates_note"] == vision_tools.COORDINATES_NOTE
    for words in ("did not convert", "own estimate", "not measured"):
        assert words in out["coordinates_note"]
    assert "page_points" not in out and "page_points_note" not in out


def test_on_a_tile_stating_page_coordinates_is_said_once_for_the_read(
        monkeypatch):
    monkeypatch.setenv(ENV, "1")
    raw = _tiled(Finder(extra=STATED, extra_in={"r2c2"}), ASK_PLAIN)
    out = json.loads(raw)
    assert out["coordinates_note"] == vision_tools.COORDINATES_NOTE
    assert raw.count(vision_tools.COORDINATES_NOTE) == 1
    assert "_page_units" not in raw


@pytest.mark.parametrize("read", [_zoom, _page, _tiled])
def test_on_ordinary_numbers_and_the_tools_own_boxes_raise_no_note(
        monkeypatch, read):
    """Stations, elevations, sheet numbers, sizes, printed survey
    coordinates and the pixel boxes the tool converts are not page
    coordinates the model made up."""
    monkeypatch.setenv(ENV, "1")
    for ask in (ASK_PLAIN, ASK_POINTS):
        out = json.loads(read(Finder(extra=ORDINARY), ask))
        assert "error" not in out, out
        assert "coordinates_note" not in out, ask


# -- image files are not changed ------------------------------------------------

@pytest.mark.parametrize("read", [_image, _image_zoom])
def test_on_an_image_file_is_not_changed(monkeypatch, read):
    """An image file has no page frame: its looks are what they were."""
    results = {}
    for value in (None, "1"):
        if value:
            monkeypatch.setenv(ENV, value)
        engine = Finder(extra=STATED)
        results[value] = (read(engine, ASK_POINTS), engine.prompts)
    assert results[None] == results["1"]
    assert "error" not in json.loads(results["1"][0])
    assert not any(k in results["1"][0] for k in NEW_KEYS)


# -- the list stays short --------------------------------------------------------

def test_the_page_points_list_is_capped():
    entries = [(f"T-{k}", (k, k, k + 4.0, k + 2.0), None) for k in range(30)]
    out = {}
    vision_tools._note_page_units(out, True, entries, False)
    assert len(out["page_points"]) == vision_tools.MAX_PAGE_POINTS == 24
    assert out["page_points_cut"] == "6 more not listed"
    assert out["page_points"][0] == {"what": "T-0", "page_bbox": [0, 0, 4, 2],
                                     "centre": [2.0, 1.0]}
