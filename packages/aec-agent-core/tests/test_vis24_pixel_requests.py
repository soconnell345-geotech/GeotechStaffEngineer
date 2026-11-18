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


def _sheet_pdf(rect=SQUARE) -> bytes:
    """A made-up sheet with one black rectangle on it (the square)."""
    d = pymupdf.open()
    page = d.new_page(width=1224, height=792)
    page.draw_rect(pymupdf.Rect(*rect), color=(0, 0, 0), fill=(0, 0, 0))
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

    def __init__(self, extra="", extra_in=None, boxes=True,
                 label="Black square", tail=""):
        self.lock = threading.Lock()
        self.prompts = {}
        self.seen = []
        self.extra, self.extra_in, self.boxes = extra, extra_in, boxes
        self.label, self.tail = label, tail

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
            lines.append("{} px=[{}, {}, {}, {}]{}".format(self.label, *dark,
                                                         self.tail))
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


def _page(engine, prompt, tiles="off", rect=SQUARE, pdf=None):
    raw = dispatch_extended_tool(
        "analyze_pdf_page",
        {"attachment_key": "made_up.pdf", "page": 0, "prompt": prompt,
         "tiles": tiles},
        engine=engine,
        attachments={"made_up.pdf": pdf or _sheet_pdf(rect)})
    vision_tools.clear_repeat_reads()
    return raw


def _tiled(engine, prompt, rect=SQUARE, pdf=None):
    return _page(engine, prompt, tiles="2", rect=rect, pdf=pdf)


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
    "center≈(516,1170) PDF-view pt",
    "(x=612, y=396) pt",
    "(612 pt, 396 pt)",
    "centre at 612.4, 396.0 pt",
    '{"x": 612, "y": 396} points',
    'Centre in PDF points: {"x": 612, "y": 396}',
    'The centre in points: {"x": 612, "y": 396}',
    "Tag box px=[12, 14, 80, 30] pt",
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
    "px=[12, 14, 80, 30]",
    "General notes in 8 pt text at (2, 3).",
    "Text heights 8, 10 pt",
    "Text heights (8 pt, 10 pt)",
    "(8 pt, 10 pt) lettering",
    "Pad centre at (12 ft, 18 ft) from grid A-1",
    "Sheet C-101 (2 of 5), details 3, 4",
    "The arrow at (612, 396) points to the valve",
    "Tag px=[12, 14, 80, 30] points to the valve",
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
                     "_px_page_boxes", "_tiled_page_points",
                     "_note_page_units"):
            m.setattr(vision_tools, name, _never)
        before = Finder(extra=STATED)
        raw_before = read(before, ASK_POINTS)
    assert raw_now == raw_before
    assert now.prompts == before.prompts
    out = json.loads(raw_now)
    assert "error" not in out, out
    assert not any(k in raw_now for k in NEW_KEYS)
    assert "_page_units" not in raw_now and "_px_points" not in raw_now
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
    """A page read whole converts its own pixel boxes; a tiled read its
    overview's and each tile's, each in its own frame, one row per thing
    with where it was seen. Every look of the read is asked for pixel
    boxes."""
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
        assert row["seen_in"] == ["page", "r1c1"]
        assert not any("_page_units" in t or "_px_points" in t
                       for t in out["tiles"])
    else:
        assert "seen_in" not in row
    assert out["page_points_note"] == vision_tools.PAGE_POINTS_NOTE


#: Made-up prose giving a box in page units (the square's own place).
PT_BOX = "Its box is [300, 200, 320, 220] pt on the page."


@pytest.mark.parametrize("read", [_zoom, _page, _tiled])
def test_on_a_box_in_page_units_never_becomes_a_page_point(monkeypatch,
                                                         read):
    """Every look answers a pixel box and a box in page units: only the
    pixel box is converted (in a tiled read too, where the page-unit box
    used to come through 'found' as if it were pixels), and the page-unit
    box is said to be the model's own. The note is true with planlens
    0.16.0, which rewrites that box onto the grid as if it were pixels,
    and with a planlens that leaves it as written."""
    from planlens.review import frames
    monkeypatch.setenv(ENV, "1")
    raw = read(Finder(extra=PT_BOX), ASK_POINTS)
    out = json.loads(raw)
    assert "error" not in out, out
    (row,) = out["page_points"]
    assert _near(row["page_bbox"], SQUARE), row
    assert out["page_points_note"] == vision_tools.PAGE_POINTS_NOTE
    assert out["coordinates_note"] == vision_tools.COORDINATES_NOTE
    assert "_px_points" not in raw and "_page_units" not in raw
    if hasattr(frames, "_PAGE_UNIT_AFTER"):        # the repair: as written
        assert "[300, 200, 320, 220] pt" in out["analysis"]
        assert out["boxes"].startswith("1 box(es) with a page unit")
    else:                                          # 0.16.0: rewritten
        assert "[300, 200, 320, 220] pt" not in out["analysis"]
        assert re.search(r"\[\d+, \d+, \d+, \d+\] pt", out["analysis"])


#: The verb "points" after a pixel box (made-up answers).
VERBS = ["points right to the boring.", "points at the pump.",
         "points upward.", "points northeast.", "points left.",
         "points roughly north."]
#: A pixel box's frame for the rules read alone: 100 pt shown at 200 px.
INFO = {"clip": [0.0, 0.0, 100.0, 100.0], "width_px": 200, "height_px": 200}


@pytest.mark.parametrize("verb", VERBS)
def test_the_verb_points_is_not_a_page_unit(monkeypatch, verb):
    """A pixel box the verb follows is a pixel box: it is converted, listed
    in 'found' and 'page_points', and raises no note (with either
    planlens)."""
    line = f"Leader px=[10, 20, 60, 70] {verb}"
    assert not vision_tools.states_page_coordinates(line)
    (row,) = vision_tools._px_page_boxes(line, INFO)
    assert row[1] == pytest.approx((5.0, 10.0, 30.0, 35.0))
    monkeypatch.setenv(ENV, "1")
    out = json.loads(_tiled(Finder(tail=f" {verb}"), ASK_POINTS))
    assert any(_near(r["page_bbox"], SQUARE) for r in out["found"])
    (point,) = out["page_points"]
    assert _near(point["page_bbox"], SQUARE), point
    assert "coordinates_note" not in out
    assert "page unit" not in out.get("boxes", "")


@pytest.mark.parametrize("unit", [" points.", " points,", " points",
                                  " pt", " pts", " PDF points",
                                  " in points"])
def test_a_page_unit_after_a_box_still_holds(unit):
    assert vision_tools.states_page_coordinates(
        f"Tag [300, 200, 320, 220]{unit}")
    assert vision_tools._px_page_boxes(
        f"Tag px=[10, 20, 60, 70]{unit}", INFO) == []


def test_on_a_tiled_read_a_bare_box_is_not_a_page_point(monkeypatch):
    """A tile's box with no pixel tag may be the model's own page numbers:
    it is not converted, and the read says it got no box to convert."""
    monkeypatch.setenv(ENV, "1")
    raw = _tiled(Finder(extra="Black square [130, 120, 160, 150]",
                        boxes=False), ASK_POINTS)
    out = json.loads(raw)
    assert "page_points" not in out
    assert out["page_points_note"] == vision_tools.NO_PAGE_POINTS_NOTE
    assert "coordinates_note" not in out


TITLE = (1100.0, 20.0, 1200.0, 772.0)


def test_on_a_tiled_read_a_large_pixel_box_is_listed(monkeypatch):
    """A title block running down the sheet's edge is cut by the tiles'
    seam: each tile's part, converted in that tile's frame, joins into one
    row at its place, though 'found' leaves a box that big out."""
    monkeypatch.setenv(ENV, "1")
    out = json.loads(_tiled(
        Finder(label="Title block"),
        "What are the page coordinates of the title block?", rect=TITLE))
    assert "error" not in out, out
    (row,) = out["page_points"]
    assert row["what"] == "Title block"
    assert row["seen_in"] == ["page", "r1c2", "r2c2"]
    assert _near(row["page_bbox"], TITLE), row
    assert not any(_near(r["page_bbox"], TITLE)
                   for r in out.get("found") or ())


SEAM = (600.0, 380.0, 620.0, 400.0)


def test_on_a_tiled_read_one_thing_in_four_tiles_is_one_row(monkeypatch):
    """A square where all four tiles overlap is one row, at its place: each
    tile's box was converted in that tile's own frame."""
    monkeypatch.setenv(ENV, "1")
    out = json.loads(_tiled(Finder(), ASK_POINTS, rect=SEAM))
    (row,) = out["page_points"]
    assert row["seen_in"] == ["page", "r1c1", "r1c2", "r2c1", "r2c2"]
    assert _near(row["page_bbox"], SEAM), row


#: Two like squares 12 pt apart where tiles r1c1 and r1c2 overlap.
RED, BLUE = (580.0, 100.0, 590.0, 110.0), (602.0, 100.0, 612.0, 110.0)


class Pair:
    """Finds the red and the blue square in the image it is sent and names
    each "Square", red first, except in tile r1c2, which lists blue
    first."""

    def analyze_image(self, image, prompt=""):
        from PIL import ImageChops
        m = TILE.search(prompt)
        who = f"r{m.group(1)}c{m.group(2)}" if m else "whole"
        r, _g, b = Image.open(io.BytesIO(image)).convert("RGB").split()

        def where(high, low):
            return ImageChops.multiply(
                high.point(lambda v: 255 if v > 200 else 0),
                low.point(lambda v: 255 if v < 100 else 0)).getbbox()
        found = [where(r, b), where(b, r)]
        if who == "r1c2":
            found.reverse()
        return "\n".join("Square px=[{}, {}, {}, {}]".format(*box)
                         for box in found if box) or "No square here."


def test_on_a_tiled_read_a_box_joins_the_nearest_thing(monkeypatch):
    """Two like squares seen in two tiles that list them in opposite order
    stay two rows, each at its own place: a box joins the nearest row."""
    d = pymupdf.open()
    page = d.new_page(width=1224, height=792)
    for rect, colour in ((RED, (1, 0, 0)), (BLUE, (0, 0, 1))):
        page.draw_rect(pymupdf.Rect(*rect), color=colour, fill=colour)
    pdf = d.tobytes()
    d.close()
    monkeypatch.setenv(ENV, "1")
    out = json.loads(_tiled(Pair(), ASK_POINTS, pdf=pdf))
    red, blue = out["page_points"]
    assert _near(red["page_bbox"], RED), red
    assert _near(blue["page_bbox"], BLUE), blue
    assert red["seen_in"] == blue["seen_in"] == ["page", "r1c1", "r1c2"]


def test_on_a_tiled_read_the_cut_counts_the_whole_list(monkeypatch):
    """'page_points_cut' counts every converted row, not a list already
    capped ('found' keeps at most MAX_MERGED)."""
    monkeypatch.setenv(ENV, "1")
    tags = "\n".join(
        f"T-{k} px=[{10 + 20 * (k % 30)}, {10 + 20 * (k // 30)}, "
        f"{20 + 20 * (k % 30)}, {20 + 20 * (k // 30)}]" for k in range(70))
    out = json.loads(_tiled(Finder(extra=tags, extra_in={"r1c1"}),
                            ASK_POINTS))
    assert vision_tools.MAX_MERGED < 71
    assert len(out["page_points"]) == vision_tools.MAX_PAGE_POINTS
    assert out["page_points_cut"] == "47 more not listed"    # 71 rows


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
    for words in ("of its own", "estimate", "not measured",
                  "do not use it as a place"):
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
