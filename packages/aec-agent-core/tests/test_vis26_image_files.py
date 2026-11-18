"""BACKLOG VIS-26: the document tools on an IMAGE file (a scan saved as PNG,
a phone photograph) -- the parts that are the core's.

* ``measure`` takes positions on an image in its own pixels (a look's view
  + image_box, a bbox, ``to``) and puts them on planlens' page of the image,
  which is in points at the dpi the file declares; its result, and
  ``log_grid``'s, says how their positions read (``frame``).
* A build's host tells planlens the scale its looks send an image view at
  (``ImageSupport.sent_scale``), where the installed planlens takes it.
* ``kept_readings`` says a kept zoom on an image is in its pixels.

A PDF's results are what they were: proved below for each function.
Made-up scans from planlens' visual-scale fixtures, rendered here to image
files (one a JPEG with an EXIF rotation tag, made with Pillow); a fake
vision engine that reads the fixture's label crops. No model, no network.
"""

from __future__ import annotations

import io
import json

import pytest

pymupdf = pytest.importorskip("pymupdf")
Image = pytest.importorskip("PIL.Image")
pytest.importorskip("planlens.tools")

from aec_agent_core import document_tools, measure_tool  # noqa: E402
from aec_agent_core import scale_labels, vision_tools  # noqa: E402
from aec_agent_core import vision_view  # noqa: E402
from aec_agent_core.deep.vision_kit import make_vision_tools  # noqa: E402

if not document_tools.has_tool("measure"):            # pragma: no cover
    pytest.skip("installed planlens predates measure", allow_module_level=True)

from planlens.testing.visual_scale_fixtures import (  # noqa: E402
    LogVariant, _overlap, build_log,
)

#: The scan is rendered at this many dpi and the file says so: planlens
#: then opens it at 72/192 = 0.375 pt a pixel, so its page points are the
#: fixture's own and the fixture's label answers still apply.
DPI = 192
PX = DPI / 72.0                       # image pixels per fixture point
K = 72.0 / DPI                        # page points per image pixel


class LabelReader:
    """A vision engine whose label reads answer from the fixture (one
    ``#N | text`` line per numbered cell, as a model reading them would)."""

    accepts_jpeg = False

    def __init__(self, fx):
        self.fx = fx
        self.calls = []
        self.sheets = []

    def vision_profile(self):
        return None

    def analyze_image(self, image, prompt):
        self.calls.append(prompt)
        if "Each numbered cell" in prompt:
            lines = []
            for i, (_page, box) in enumerate(self.sheets[-1]):
                best = max(self.fx.labels, key=lambda lb: _overlap(box, lb.box))
                text = best.text if _overlap(box, best.box) > 0.2 else "-"
                lines.append(f"#{i + 1} | {text}")
            return "\n".join(lines)
        return "seen"


@pytest.fixture(autouse=True)
def _sheets(monkeypatch):
    real = scale_labels.label_sheets
    engines = []

    def spy(doc, items, *a, **kw):
        for e in engines:
            e.sheets.append(list(items))
        return real(doc, items, *a, **kw)

    monkeypatch.setattr(scale_labels, "label_sheets", spy)
    scale_labels.clear_cache()
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    yield engines
    scale_labels.clear_cache()


@pytest.fixture(scope="module")
def scan():
    return build_log(LogVariant("app_scan", skew_deg=0.4, seed=91))


def _picture(fx):
    """The fixture's page as a PIL image at :data:`DPI`."""
    with pymupdf.open(stream=fx.pdf, filetype="pdf") as doc:
        pix = doc[0].get_pixmap(dpi=DPI)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


@pytest.fixture(scope="module")
def png(scan):
    buf = io.BytesIO()
    _picture(scan).save(buf, format="PNG", dpi=(DPI, DPI))
    return buf.getvalue()


@pytest.fixture(scope="module")
def turned_jpeg(scan):
    """The same scan as a phone writes a photo taken turned: the pixels
    stored a quarter turn anticlockwise, and EXIF orientation 6 (turn them
    a quarter clockwise to show them)."""
    stored = _picture(scan).transpose(Image.Transpose.ROTATE_90)
    ex = Image.Exif()
    ex[0x0112] = 6
    buf = io.BytesIO()
    stored.save(buf, format="JPEG", quality=95, dpi=(DPI, DPI),
                exif=ex.tobytes())
    return buf.getvalue()


def _tools(engine, name, data):
    return {t.name: t for t in make_vision_tools(
        engine=engine, attachments={name: data})}


def _call(tool, **args):
    return json.loads(tool.invoke(args))


def _px(box_pt):
    return [round(v * PX, 1) for v in box_pt]


# -- measure ----------------------------------------------------------------

@pytest.mark.parametrize("name", ["log.png", "photo.jpg"])
def test_measure_takes_a_bbox_in_the_image_s_pixels(_sheets, scan, png,
                                                    turned_jpeg, name):
    data = png if name == "log.png" else turned_jpeg
    eng = LabelReader(scan)
    _sheets.append(eng)
    tools = _tools(eng, name, data)
    rd = scan.readings[1]
    out = _call(tools["measure"], source=name, page=0,
                bbox=_px(rd.box_pt), kind="line")
    v = out["value"]
    assert v is not None, out
    assert abs(v["depth"] - rd.value) <= v["plus_minus"], (v, rd.value)
    w, h = _picture(scan).size
    assert f"an image, {w} x {h} px" in out["frame"]
    assert f"{K:.4g} pt a pixel" in out["frame"]
    assert "bbox, to and a look's view + image_box" in out["frame"]
    # planlens' own positions stay in its page's points, as the note says
    assert out["where"] == pytest.approx(list(rd.box_pt), abs=0.2)


def test_a_zoom_s_view_and_image_box_on_an_image(_sheets, scan, png):
    eng = LabelReader(scan)
    _sheets.append(eng)
    tools = _tools(eng, "log.png", png)
    rd = scan.readings[2]
    b = _px(rd.box_pt)
    cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    view = [cx - 200.0, cy - 160.0, cx + 200.0, cy + 160.0]   # a zoom, px
    w, h = view[2] - view[0], view[3] - view[1]
    image_box = [(b[0] - view[0]) / w * 999, (b[1] - view[1]) / h * 999,
                 (b[2] - view[0]) / w * 999, (b[3] - view[1]) / h * 999]
    out = _call(tools["measure"], source="log.png", page=0, view=view,
                image_box=image_box, kind="line")
    assert abs(out["value"]["depth"] - rd.value) <= \
        out["value"]["plus_minus"], out
    ex, ey = vision_view.location_error(view)
    assert out["pad_pt"] == pytest.approx(max(ex, ey) * K, abs=0.02)
    note = out["located_from"]["note"]
    assert "400 x 320 px view of the image" in note
    assert f"within {max(ex, ey):.0f} px" in note and " pt " not in note


def test_a_pdf_measure_is_what_it_was(_sheets, scan, monkeypatch):
    """No ``frame``, the points note, and the same bytes as with the image
    path taken out."""
    rd = scan.readings[2]
    cx = (rd.box_pt[0] + rd.box_pt[2]) / 2.0
    cy = (rd.box_pt[1] + rd.box_pt[3]) / 2.0
    view = [cx - 75.0, cy - 60.0, cx + 75.0, cy + 60.0]
    w, h = view[2] - view[0], view[3] - view[1]
    image_box = [(rd.box_pt[0] - view[0]) / w * 999,
                 (rd.box_pt[1] - view[1]) / h * 999,
                 (rd.box_pt[2] - view[0]) / w * 999,
                 (rd.box_pt[3] - view[1]) / h * 999]
    calls = [dict(source="log.pdf", page=0, view=view, image_box=image_box,
                  kind="line"),
             dict(source="log.pdf", page=0, bbox=list(rd.box_pt),
                  kind="line")]
    outs = []
    for take_out in (False, True):
        if take_out:
            monkeypatch.setattr(measure_tool, "_image_frame",
                                lambda entry, attachments=None: None)
        eng = LabelReader(scan)
        _sheets.append(eng)
        scale_labels.clear_cache()
        tools = _tools(eng, "log.pdf", scan.pdf)
        outs.append([tools["measure"].invoke(c) for c in calls])
    assert outs[0] == outs[1]
    first = json.loads(outs[0][0])
    assert "frame" not in first
    assert "150 x 120 pt view" in first["located_from"]["note"]


def test_a_page_that_does_not_match_the_image_is_refused(scan, png,
                                                         monkeypatch):
    from aec_agent_core import image_view
    w, h = _picture(scan).size
    monkeypatch.setattr(image_view, "shown_size", lambda data: (h, w))
    tools = _tools(None, "log.png", png)
    out = _call(tools["measure"], source="log.png", page=0,
                bbox=[10, 10, 50, 50], kind="line")
    assert "another shape" in out["error"], out


# -- log_grid ---------------------------------------------------------------

def test_log_grid_on_an_image_says_how_its_positions_read(_sheets, scan,
                                                          png):
    eng = LabelReader(scan)
    _sheets.append(eng)
    tools = _tools(eng, "log.png", png)
    handle = _call(tools["open_document"], source="log.png")["handle"]
    out = _call(tools["log_grid"], handle=handle, pages="0", rows=False)
    tops = [ly for ly in out["layers"] if ly.get("source") == "stratum_rule"]
    assert len(tops) == len(scan.readings)
    for rd, ly in zip(scan.readings, tops):
        assert abs(ly["top"] - rd.value) <= max(ly["plus_minus"], 0.01)
    assert f"{K:.4g} pt a pixel" in out["frame"]
    assert "bbox, to" not in out["frame"]


def test_a_pdf_log_grid_is_what_it_was(_sheets, scan, monkeypatch):
    outs = []
    for take_out in (False, True):
        if take_out:
            monkeypatch.setattr(measure_tool, "_image_frame",
                                lambda entry, attachments=None: None)
        eng = LabelReader(scan)
        _sheets.append(eng)
        scale_labels.clear_cache()
        tools = _tools(eng, "log.pdf", scan.pdf)
        handle = _call(tools["open_document"], source="log.pdf")["handle"]
        outs.append(tools["log_grid"].invoke(
            {"handle": handle, "pages": "0", "rows": False}))
    assert outs[0] == outs[1]
    assert '"frame"' not in outs[0]


# -- the host's sent scale --------------------------------------------------

class _Seer:
    accepts_jpeg = False

    def vision_profile(self):
        return None

    def analyze_image(self, image, prompt):
        return "seen"


def test_a_build_host_tells_planlens_the_scale_it_sends_at():
    host = document_tools.host_for(_Seer())
    if not document_tools._takes_sent_scale(host.images):
        # the installed planlens judges an image view by its width alone
        assert host.images is document_tools.BASE_HOST.images
        pytest.skip("installed planlens has no ImageSupport.sent_scale")
    s = host.images.sent_scale
    assert s(400, 300) == pytest.approx(1.0)          # goes whole
    assert s(20000, 15000) < 0.5                       # shrunk to fit
    assert document_tools.host_for(None).images is \
        document_tools.BASE_HOST.images                # no engine, no scale


def test_a_mark_on_a_wide_image_view_sent_whole(tmp_path):
    """A screenshot's whole view (1200 x 900 px) goes to the model whole, so
    a small mark read off it is placed with a planlens that takes the
    scale; planlens 0.16.0 refuses it by the view's width, as before."""
    from aec_agent_core._fileio import working_dir_bound
    buf = io.BytesIO()
    Image.new("RGB", (1200, 900), "white").save(buf, format="PNG")
    mark = {"kind": "box", "page": 0, "comment": "check this tag",
            "view": [0, 0, 1200, 900], "image_box": [333, 333, 373, 373]}
    with working_dir_bound(str(tmp_path)):
        tools = _tools(_Seer(), "shot.png", buf.getvalue())
        out = _call(tools["annotate_document"], handle="shot.png",
                    markups=[mark])
    if document_tools._takes_sent_scale(
            document_tools.host_for(_Seer()).images):
        assert out.get("n_written") == 1, out
    else:
        assert out.get("nothing_written") is True, out


# -- kept_readings ----------------------------------------------------------

def test_a_kept_zoom_on_an_image_says_its_pixels(tmp_path):
    from aec_agent_core._fileio import working_dir_bound
    image_zoom = json.dumps({"analysis": "a tag reading A-1",
                             "view": [100, 80, 500, 400],
                             "image_px": [1200, 900], "frame": "px"})
    pdf_zoom = json.dumps({"analysis": "a tag reading B-2",
                           "view": [40, 300, 200, 420]})
    vision_tools.clear_read_log()
    try:
        with working_dir_bound(str(tmp_path)):
            vision_tools._keep_reading("shot.png", b"\x89PNG made up", 0,
                                       vision_tools.ZOOM_VIEW, "Read the tag",
                                       image_zoom, region=[100, 80, 500, 400])
            vision_tools._keep_reading("set.pdf", b"%PDF made up", 3,
                                       vision_tools.ZOOM_VIEW, "Read the tag",
                                       pdf_zoom, region=[40, 300, 200, 420])
        out = json.loads(vision_tools.dispatch_kept_readings(
            {}, folder=str(tmp_path)))
    finally:
        vision_tools.clear_read_log()
    image_ref = out["readings"]["shot.png"][0]
    pdf_ref = out["readings"]["set.pdf"][0]
    assert image_ref["pdf_page"] == 1
    assert image_ref["frame"] == vision_tools.KEPT_IMAGE_FRAME
    assert "own pixels" in image_ref["frame"]
    assert list(pdf_ref) == ["pdf_page", "kind", "bbox", "asked", "chars",
                             "summary"]
