"""The drawing layer (BACKLOG GM-313): the typed description, the DXF it
becomes, and the PDF page and PNG drawn from that DXF.

Every value here is made up.
"""

import copy
import io
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ezdxf = pytest.importorskip("ezdxf")
pymupdf = pytest.importorskip("pymupdf")

from dxf_export import (                                    # noqa: E402
    STAMP, Drawing, DrawingError, build_dxf, render_sheet,
    validate_wall_description, wall_section_drawing, write_sheet,
)
from dxf_export.drawing import SCALES, SHEETS               # noqa: E402
from dxf_export.sheet import (                              # noqa: E402
    DIM_STYLE, LAYERS, TEXT_H, choose_scale, second_unit_text,
)

FIXTURE = Path(__file__).parent / "fixtures" / "wall_section_example.json"


def wall_description() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def general() -> dict:
    """A small made-up sheet that uses every part of the description."""
    return {
        "outlines": [
            {"points": [[0, 0], [2000, 0], [2000, 400], [1200, 400],
                        [1200, 1800], [800, 1800], [800, 400], [0, 400]],
             "layer": "concrete", "hatch": "concrete"},
            {"points": [[2300, 0], [3300, 0], [3300, 600], [2300, 600]],
             "layer": "soil", "hatch": "gravel"}],
        "lines": [{"points": [[-600, 900], [800, 900]], "layer": "ground"},
                  {"points": [[1000, -200], [1000, 2000]],
                   "layer": "centre"}],
        "bars": [{"points": [[1120, 1750], [1120, 100], [300, 100]],
                  "diameter_mm": 20, "bend_radius_mm": 80},
                 {"points": [[900, 600], [900, 900], [900, 1200]],
                  "diameter_mm": 16, "kind": "dots"}],
        "dimensions": [
            {"p1": [0, 0], "p2": [2000, 0], "offset_mm": -16,
             "second_unit": "ft-in"},
            {"p1": [0, 0], "p2": [800, 0], "offset_mm": -8},
            {"p1": [0, 0], "p2": [800, 1800], "offset_mm": -10,
             "direction": "vertical", "suffix": " mm"},
            {"p1": [1130, 1750], "p2": [1200, 1750], "offset_mm": 8},
            {"p1": [2300, 0], "p2": [3300, 600], "offset_mm": 8,
             "direction": "aligned"}],
        "callouts": [
            {"at": [1120, 1300], "text_at": [1700, 1500],
             "text": "No. 6 @ 300 mm, back face", "mark": "A"},
            {"at": [900, 900], "text_at": [400, 1400],
             "text": "No. 4 @ 300 mm along the wall", "width_mm": 30}],
        "notes": ["All dimensions in millimetres.", "A made-up sheet."],
        "table": {"title": "BAR SCHEDULE",
                  "headers": ["Mark", "Size", "Spacing"],
                  "rows": [["A", "No. 6", "300 mm"], ["B", "No. 4", 300]]},
        "title_block": {"title": "A made-up test sheet", "project": "Test"},
    }


def by_type(doc) -> dict:
    out = {}
    for entity in doc.modelspace():
        out.setdefault(entity.dxftype(), []).append(entity)
    return out


# -- the description --------------------------------------------------------

def test_a_description_round_trips_through_json():
    drawing = Drawing.from_dict(general())
    again = Drawing.from_dict(json.loads(json.dumps(drawing.to_dict())))
    assert again.to_dict() == drawing.to_dict()
    assert drawing.sheet.size == "letter" and drawing.sheet.scale is None


@pytest.mark.parametrize("path, key", [
    ((), "colour"),
    (("outlines", 0), "lineweight"),
    (("lines", 0), "color"),
    (("bars", 0), "layer"),
    (("dimensions", 0), "text"),
    (("callouts", 0), "font"),
    (("table",), "widths"),
    (("title_block",), "stamp"),
    (("sheet",), "portrait"),
])
def test_an_unknown_key_is_refused_by_name(path, key):
    data = general()
    data.setdefault("sheet", {})
    target = data
    for step in path:
        target = target[step]
    target[key] = 1
    with pytest.raises(DrawingError, match=key):
        Drawing.from_dict(data)


@pytest.mark.parametrize("change, message", [
    (lambda d: d["outlines"][0].update(hatch="brick"), "hatch"),
    (lambda d: d["outlines"][0].update(points=[[0, 0], [1, 1]]),
     "at least 3"),
    (lambda d: d["lines"][0].update(layer="GSE-STAMP"), "layer"),
    (lambda d: d["bars"][0].update(diameter_mm=0), "diameter_mm"),
    (lambda d: d["bars"][0].update(points=[[0, float("nan")], [1, 1]]),
     "finite"),
    (lambda d: d["dimensions"][0].update(p2=[0, 0]), "nothing to dimension"),
    (lambda d: d["dimensions"][0].update(offset_mm=0), "offset_mm"),
    (lambda d: d["dimensions"][0].update(second_unit="yd"), "second_unit"),
    (lambda d: d["dimensions"][0].update(suffix="<> 99"), "suffix"),
    (lambda d: d["callouts"][0].update(text=""), "empty"),
    (lambda d: d["table"].update(rows=[["A", "B"]]), "3 cells"),
    (lambda d: d.update(sheet={"size": "a0"}), "size"),
    (lambda d: d.update(sheet={"scale": 30}), "scale"),
    (lambda d: d.update(schema="drawing_sheet/9"), "schema"),
    (lambda d: d.update(outlines=[], lines=[], bars=[]), "nothing to draw"),
])
def test_a_bad_value_is_refused_in_plain_words(change, message):
    data = general()
    change(data)
    with pytest.raises(DrawingError, match=message):
        Drawing.from_dict(data)


def test_a_dimension_cannot_carry_a_typed_number():
    """The description has no place for a dimension's number."""
    from dxf_export.drawing import Dimension
    assert "text" not in Dimension.KEYS and "value" not in Dimension.KEYS


def test_second_units():
    assert second_unit_text(2000, "ft-in") == "6'-6 3/4\""
    assert second_unit_text(76.2, "ft-in") == "3\""
    assert second_unit_text(304.8, "in") == "12.0 in."
    assert second_unit_text(3048, "ft") == "10.00 ft"
    assert second_unit_text(1500, "m") == "1.500 m"


# -- the DXF ----------------------------------------------------------------

def test_the_dxf_holds_real_entities_on_the_fixed_layers(tmp_path):
    drawing = Drawing.from_dict(general())
    result = write_sheet(drawing, str(tmp_path / "sheet.dxf"))
    doc = ezdxf.readfile(result.filepath)
    auditor = doc.audit()
    assert auditor.errors == [] and auditor.fixes == []
    kinds = by_type(doc)

    assert len(kinds["DIMENSION"]) == 5 == result.counts["dimensions"]
    assert len(kinds["LEADER"]) == 2 == result.counts["callouts"]
    patterns = sorted(h.dxf.pattern_name for h in kinds["HATCH"]
                      if not h.dxf.solid_fill)
    assert patterns == ["AR-CONC", "GRAVEL"]
    assert result.counts["hatches"] == 2
    # the three bars seen end-on are solid fills on the bars' layer
    dots = [h for h in kinds["HATCH"] if h.dxf.solid_fill
            and h.dxf.layer == LAYERS["bars"][0]]
    assert len(dots) == 3 == result.counts["bar_dots"]
    # a bar in the plane of the drawing: a wide polyline with a bend
    bars = [p for p in kinds["LWPOLYLINE"]
            if p.dxf.layer == LAYERS["bars"][0]]
    assert len(bars) == 1
    assert bars[0].dxf.const_width == pytest.approx(20.0)
    assert any(abs(b) > 0 for *_, b in bars[0].get_points("xyb"))

    names = {layer.dxf.name for layer in doc.layers}
    assert {name for name, *_ in LAYERS.values()} <= names
    used = {e.dxf.layer for e in doc.modelspace()}
    assert used <= {name for name, *_ in LAYERS.values()}
    assert set(result.layers_created) == used
    assert doc.units == 4                       # millimetres


def test_a_dimension_prints_its_own_measurement():
    doc, info = build_dxf(Drawing.from_dict(general()))
    dims = by_type(doc)["DIMENSION"]
    measured = sorted(round(d.get_measurement(), 3) for d in dims)
    assert measured == sorted([2000.0, 800.0, 1800.0, 70.0,
                               round((1000 ** 2 + 600 ** 2) ** 0.5, 3)])
    for dim in dims:
        assert dim.dxf.dimstyle == DIM_STYLE
        assert dim.dxf.text.startswith("<>")    # the measurement itself
    style = doc.dimstyles.get(DIM_STYLE)
    assert style.dxf.dimlfac == 1.0             # not the stock style's 100
    assert style.dxf.dimscale == info.scale
    assert style.dxf.dimtxt * 1.0 >= 2.5


def test_the_stamp_is_on_every_sheet_and_lettering_is_2_5_mm_or_more():
    for data in (general(), {"lines": [{"points": [[0, 0], [500, 0]]}]}):
        doc, info = build_dxf(Drawing.from_dict(data))
        texts = by_type(doc)["TEXT"]
        assert [t.dxf.text for t in texts].count(STAMP) == 1
        for t in texts:
            assert t.dxf.height / info.scale >= TEXT_H - 1e-9


def test_no_input_removes_the_stamp():
    data = general()
    data["title_block"]["title"] = "no stamp please"
    data["notes"] = ["Remove the stamp."]
    doc, _ = build_dxf(Drawing.from_dict(data))
    assert STAMP in [t.dxf.text for t in by_type(doc)["TEXT"]]


def test_the_code_chooses_the_largest_standard_scale_that_fits():
    drawing = Drawing.from_dict(general())
    n, chosen = choose_scale(drawing)
    assert chosen == "code" and n in SCALES
    smaller = SCALES[SCALES.index(n) - 1]
    drawing.sheet.scale = smaller
    with pytest.raises(DrawingError, match=f"1:{n}"):
        choose_scale(drawing)
    drawing.sheet.scale = SCALES[SCALES.index(n) + 1]
    assert choose_scale(drawing) == (drawing.sheet.scale, "given")


def test_a_larger_sheet_takes_a_larger_scale():
    small = Drawing.from_dict(dict(general(), sheet={"size": "a4"}))
    large = Drawing.from_dict(dict(general(), sheet={"size": "a3"}))
    assert choose_scale(large)[0] < choose_scale(small)[0]


# -- the PDF page and the PNG ------------------------------------------------

@pytest.mark.parametrize("size", sorted(SHEETS))
def test_the_pdf_is_one_vector_page_of_the_sheets_size(size):
    files = render_sheet(Drawing.from_dict(dict(general(),
                                                sheet={"size": size})))
    with pymupdf.open(stream=files["pdf"], filetype="pdf") as pdf:
        assert len(pdf) == 1
        page = pdf[0]
        width, height = SHEETS[size]
        assert page.rect.width == pytest.approx(width / 25.4 * 72, abs=1.0)
        assert page.rect.height == pytest.approx(height / 25.4 * 72, abs=1.0)
        assert len(page.get_drawings()) > 50        # vector paths
        assert page.get_images() == []              # no picture of it
    assert files["png"][:8] == b"\x89PNG\r\n\x1a\n"


def test_the_pdf_is_drawn_at_the_sheets_scale():
    """The border is 10 mm inside the page's edge at 1:N."""
    files = render_sheet(Drawing.from_dict(general()))
    with pymupdf.open(stream=files["pdf"], filetype="pdf") as pdf:
        page = pdf[0]
        box = pymupdf.Rect()
        for path in page.get_drawings():
            if path["rect"].width < page.rect.width - 5:    # not the paper
                box |= path["rect"]
        mm = 72 / 25.4
        assert box.x0 == pytest.approx(10 * mm, abs=1.5)
        assert box.y0 == pytest.approx(10 * mm, abs=1.5)
        assert box.x1 == pytest.approx(page.rect.width - 10 * mm, abs=1.5)
        assert box.y1 == pytest.approx(page.rect.height - 10 * mm, abs=1.5)


def test_three_files_are_written_and_nothing_on_a_refusal(tmp_path):
    drawing = Drawing.from_dict(general())
    result = write_sheet(drawing, str(tmp_path / "out" / "s"))
    for path in (result.filepath, result.pdf_path, result.png_path):
        assert os.path.getsize(path) > 1000
    assert result.to_dict()["scale"] == f"1:{result.scale}"
    assert "DRAWING SHEET" in result.summary()

    drawing.sheet.scale = 1
    with pytest.raises(DrawingError):
        write_sheet(drawing, str(tmp_path / "refused.dxf"))
    assert not list(tmp_path.glob("refused.*"))


# -- the first drawing: the reinforced wall section ---------------------------

def test_the_wall_fixture_draws_every_bar_set_dimension_and_callout(tmp_path):
    description = wall_description()
    drawing = wall_section_drawing(description)
    result = write_sheet(drawing, str(tmp_path / "wall.dxf"))
    doc = ezdxf.readfile(result.filepath)
    auditor = doc.audit()
    assert auditor.errors == [] and auditor.fixes == []
    kinds = by_type(doc)

    assert result.counts["bars"] == len(description["bars"]) == 6
    assert len(kinds["LEADER"]) == 6            # one callout per bar set
    texts = [t.dxf.text for t in kinds["TEXT"]]
    for bar in description["bars"]:
        assert texts.count(bar["mark"]) >= 2    # its callout, its schedule row
    assert STAMP in texts

    measured = sorted(round(d.get_measurement()) for d in kinds["DIMENSION"])
    # wall height, stem height, base thickness; base width, toe, stem at
    # its base, heel; stem at its top; the four covers
    for value in (4500, 4050, 450, 2700, 1800, 300, 100):
        assert value in measured
    assert measured.count(450) == 3             # base thickness, toe, stem
    assert measured.count(75) == 3              # stem back, front; base top
    assert len(measured) == 12

    hatch = [h for h in kinds["HATCH"] if not h.dxf.solid_fill]
    assert [h.dxf.pattern_name for h in hatch] == ["AR-CONC"]
    assert result.scale == 50 and result.sheet == "letter"


def test_a_hooked_back_face_bar_turns_toward_the_toe():
    """EM 1110-2-2104 (2023) Fig. 2-1, p. 9: the hook is turned toward
    the opposite face of the stem, across the joint."""
    drawing = wall_section_drawing(wall_description())
    hooked = [b for b in drawing.bars
              if b.kind == "path" and len(b.points) == 3]
    assert len(hooked) == 1
    top, corner, tail = hooked[0].points
    assert corner[1] == pytest.approx(tail[1])          # a flat tail
    assert tail[0] < corner[0]                          # toward the toe
    assert tail[0] == pytest.approx(450.0)              # the front face
    assert corner[1] < 450.0 < top[1]                   # down in the base
    assert hooked[0].bend_radius_mm > 0

    front = wall_description()
    front["bars"][0]["end"] = None
    front["bars"][1]["end"] = {"type": "standard_hook_90", "into": "base",
                               "ldh_mm": 150.0}
    drawing = wall_section_drawing(front)
    hooked = [b for b in drawing.bars
              if b.kind == "path" and len(b.points) == 3]
    _, corner, tail = hooked[0].points
    assert tail[0] > corner[0]                          # toward the heel


def test_bars_along_the_wall_sit_inside_the_bars_in_the_section():
    drawing = wall_section_drawing(wall_description())
    paths = [b for b in drawing.bars if b.kind == "path"]
    dots = [b for b in drawing.bars if b.kind == "dots"]
    back_bar = paths[0]                 # A, back face
    (xt, yt), (xb, yb) = back_bar.points[:2]

    def x_bar(y):                       # the bar follows the battered face
        return xb + (xt - xb) * (y - yb) / (yt - yb)

    stem_dots = dots[0].points          # C, each face
    back_dots = [p for p in stem_dots if p[0] > 600]
    assert back_dots and all(p[0] < x_bar(p[1]) - 10 for p in back_dots)
    ys = sorted(p[1] for p in back_dots)
    assert ys[1] - ys[0] == pytest.approx(300.0)        # at their spacing


def test_a_hook_that_does_not_fit_the_base_is_warned_of(tmp_path):
    description = wall_description()
    description["bars"][0]["end"]["ldh_mm"] = 400.0
    drawing = wall_section_drawing(description)
    result = write_sheet(drawing, str(tmp_path / "w"))
    assert any("Bar A" in w and "400 mm" in w for w in result.warnings)


WALLS = {
    "low": dict(wall_height=1.8, base_width=1.4, toe_length=0.3,
                heel_length=0.8, stem_thickness_top=0.25,
                stem_thickness_base=0.3, base_thickness=0.3,
                embedment_depth=0.6),
    "tall_long_heel": dict(wall_height=7.5, base_width=5.4, toe_length=0.9,
                           heel_length=3.8, stem_thickness_top=0.35,
                           stem_thickness_base=0.7, base_thickness=0.7,
                           embedment_depth=1.2, backfill_slope_deg=10.0),
    "front_batter_key": dict(wall_height=3.6, base_width=2.4,
                             toe_length=0.6, heel_length=1.3,
                             stem_thickness_top=0.25,
                             stem_thickness_base=0.5, base_thickness=0.4,
                             stem_batter="front", embedment_depth=0.8,
                             has_shear_key=True, key_depth=0.4,
                             key_width=0.4, key_distance_from_toe=0.9),
}


def _boxes_overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


@pytest.mark.parametrize("name", sorted(WALLS))
def test_walls_of_other_proportions_lay_out_with_no_callouts_overlapping(
        name):
    from dxf_export.sheet import _callout_geometry
    description = wall_description()
    description["geometry"].update(WALLS[name])
    drawing = wall_section_drawing(description, sheet={"size": "tabloid"})
    doc, info = build_dxf(drawing)
    assert doc.audit().errors == []
    boxes = []
    for callout in drawing.callouts:
        g = _callout_geometry(callout, info.scale)
        half = max(len(g.lines), 1) * 4.2 * info.scale / 2
        x0, x1 = sorted((g.text_x, g.box[2] if g.side > 0 else g.box[0]))
        boxes.append((x0, callout.text_at[1] - half, x1,
                      callout.text_at[1] + half))
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert not _boxes_overlap(a, b)
    # a callout's text is outside the concrete
    g = description["geometry"]
    for callout in drawing.callouts:
        assert (callout.text_at[0] > g["base_width"] * 1000
                or callout.text_at[0] < g["toe_length"] * 1000)
    if g.get("has_shear_key"):
        measured = [round(d.get_measurement())
                    for d in by_type(doc)["DIMENSION"]]
        assert measured.count(400) >= 3     # key depth, key width, base


@pytest.mark.parametrize("change, message", [
    (lambda d: d.update(extra=1), "extra"),
    (lambda d: d["geometry"].update(batter=1), "batter"),
    (lambda d: d["bars"][0].update(colour="red"), "colour"),
    (lambda d: d["bars"][0]["end"].update(radius=3), "radius"),
    (lambda d: d["geometry"].update(heel_length=1.0), "heel_length"),
    (lambda d: d["bars"][1].update(mark="A"), "used twice"),
    (lambda d: d["bars"][0].update(face="top"), "face"),
    (lambda d: d["bars"][3].update(end={"type": "standard_hook_90"}),
     "hook"),
    (lambda d: d.update(bars=[]), "no bar set"),
    (lambda d: d.update(schema="reinforced_wall_section/2"), "schema"),
    # a key needs its depth; its width and position may be left out
    # (GM-312 gives none: drawn under the stem, said to be assumed)
    (lambda d: d["geometry"].update(has_shear_key=True), "key_depth"),
])
def test_a_wall_description_is_checked(change, message):
    description = wall_description()
    change(description)
    with pytest.raises(DrawingError, match=message):
        validate_wall_description(description)


def test_validation_does_not_change_the_description_given():
    description = wall_description()
    before = copy.deepcopy(description)
    wall_section_drawing(description)
    assert description == before


# -- hosts with no system fonts and a read-only home --------------------------

_RENDER = textwrap.dedent("""
    import hashlib, json, sys
    from ezdxf.fonts import fonts
    if sys.argv[1] == "hidden":
        # the host has no fonts at all: empty ezdxf's font manager
        fonts.font_manager.clear()
        assert not fonts.font_manager.has_font("DejaVuSans.ttf")
        assert not fonts.font_manager.has_font("arial.ttf")
    import pymupdf
    from dxf_export import Drawing, render_sheet
    # lettering of every kind and no pattern hatch (ezdxf jiggles a
    # pattern's lines by a random hair, so two renders of one differ)
    description = {
        "outlines": [{"points": [[0, 0], [2000, 0], [2000, 400], [0, 400]],
                      "layer": "concrete"}],
        "dimensions": [{"p1": [0, 0], "p2": [2000, 0], "offset_mm": -8,
                        "second_unit": "ft-in"}],
        "callouts": [{"at": [1000, 200], "text_at": [2400, 900],
                      "text": "A made-up callout, wrapped over lines",
                      "mark": "A"}],
        "notes": ["A made-up note."],
        "table": {"title": "T", "headers": ["a", "b"], "rows": [["1", "2"]]},
        "title_block": {"title": "Fonts"}}
    files = render_sheet(Drawing.from_dict(description))
    page = pymupdf.open(stream=files["pdf"], filetype="pdf")[0]
    pix = page.get_pixmap(dpi=60)
    print(json.dumps({"digest": hashlib.sha1(pix.samples).hexdigest(),
                      "paths": len(page.get_drawings())}))
""")


def _render_in_a_fresh_process(tmp_path, mode: str, home: Path) -> dict:
    env = dict(os.environ, HOME=str(home), USERPROFILE=str(home),
               GEOTECH_NO_DISCLAIMER="1", MPLCONFIGDIR=str(tmp_path / "mpl"))
    env.pop("XDG_CACHE_HOME", None)
    env.pop("XDG_CONFIG_HOME", None)
    repo = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(
        [repo] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep)
                  if p])
    done = subprocess.run(
        [sys.executable, "-c", _RENDER, mode],
        capture_output=True, text=True, env=env, cwd=str(tmp_path),
        timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_a_sheet_renders_the_same_with_no_system_fonts_and_a_read_only_home(
        tmp_path):
    """The lettering comes from the DejaVu Sans matplotlib ships, so a
    host with no fonts draws the same sheet; and ezdxf's font cache,
    which it cannot write under a read-only home, is not needed."""
    home = tmp_path / "home"
    home.mkdir()
    normal = _render_in_a_fresh_process(tmp_path, "normal", home)
    # a home nothing can be written under, whoever runs the test: a path
    # below a regular file
    (tmp_path / "a_file").write_text("not a folder")
    locked = tmp_path / "a_file" / "home"
    hidden = _render_in_a_fresh_process(tmp_path, "hidden", locked)
    assert hidden["paths"] > 40         # drawn, lettering and all
    assert hidden == normal


def test_an_aligned_dimension_reads_upright_whichever_way_it_is_given():
    """Given right to left, it is measured left to right on the same
    side, so its number is not upside down."""
    from dxf_export.drawing import Dimension
    from dxf_export.sheet import _dim_geometry
    one = _dim_geometry(Dimension(p1=(0, 600), p2=(1000, 0), offset_mm=8,
                                  direction="aligned"), 10)
    other = _dim_geometry(Dimension(p1=(1000, 0), p2=(0, 600), offset_mm=-8,
                                    direction="aligned"), 10)
    assert one.p1 == other.p1 == (0, 600)
    assert one.a == pytest.approx(other.a) and one.b == pytest.approx(other.b)


def test_a_number_too_wide_for_its_dimension_is_set_beyond_p2():
    from dxf_export.drawing import Dimension
    from dxf_export.sheet import _dim_geometry
    wide = _dim_geometry(Dimension(p1=(0, 0), p2=(2000, 0), offset_mm=-8),
                         50)
    assert wide.fits and wide.location is None
    narrow = _dim_geometry(Dimension(p1=(100, 0), p2=(25, 0), offset_mm=8),
                           50)
    assert not narrow.fits and narrow.location[0] < 25      # beyond p2


# -- review fixes (2026-11-13) -------------------------------------------------

REAL = Path(__file__).parent / "fixtures" / "gm312_real_descriptions.json"


def real_descriptions() -> dict:
    """Three descriptions as GM-312's structural design returns them, for
    made-up walls: seven bar sets (the base's along-wall steel as F, top,
    and G, bottom), a shear key given by its depth only, a hook with no
    ldh, main bars at 150 mm."""
    return json.loads(REAL.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["plain", "key_surcharge",
                                  "front_batter_high_strength"])
def test_a_real_description_validates_and_draws_with_a_clean_dxf(
        name, tmp_path):
    from dxf_export.sheet import _dim_geometry
    description = real_descriptions()[name]
    validate_wall_description(description)
    drawing = wall_section_drawing(description)         # letter, by default
    result = write_sheet(drawing, str(tmp_path / f"{name}.dxf"))
    doc = ezdxf.readfile(result.filepath)
    auditor = doc.audit()
    assert auditor.errors == [] and auditor.fixes == []
    assert result.sheet == "letter"
    kinds = by_type(doc)
    assert len(kinds["LEADER"]) == len(description["bars"]) == 7
    texts = [t.dxf.text for t in kinds["TEXT"]]
    assert texts.count(STAMP) == 1
    assert not any("None" in t for t in texts)
    # the design basis is said once (GM-312's notes already carry it)
    assert sum(n.lower().startswith("design basis")
               for n in drawing.notes) == 1
    # no dimension's number prints over another's
    n = drawing.sheet.scale
    boxes = [_dim_geometry(d, n).text_box for d in drawing.dimensions]
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            assert not _boxes_overlap(a, b), (a, b)


def test_a_key_placed_but_not_sized_is_not_said_to_be_under_the_stem():
    """GM-335 review: with the key's place given and its width not, the
    note still named the stem as its place."""
    description = real_descriptions()["key_surcharge"]
    g = description["geometry"]
    g["key_distance_from_toe"] = round(g["toe_length"] + 0.2, 3)
    drawing = wall_section_drawing(description)
    note = [n for n in drawing.notes if n.startswith("Shear key")]
    assert len(note) == 1, drawing.notes
    assert "under the stem" not in note[0]
    assert "as wide as the stem's base" in note[0]
    assert "its width is assumed" in note[0]
    assert any(note[0] in w for w in drawing.layout_warnings)


def test_a_key_given_by_its_depth_only_is_drawn_under_the_stem_and_said_to_be_assumed():  # noqa: E501
    description = real_descriptions()["key_surcharge"]
    g = description["geometry"]
    assert g["key_width"] is None and g["key_distance_from_toe"] is None
    drawing = wall_section_drawing(description)
    depth = -g["key_depth"] * 1000.0
    key_x = sorted({x for x, y in drawing.outlines[0].points
                    if abs(y - depth) < 1e-6})
    toe = g["toe_length"] * 1000.0
    assert key_x == pytest.approx([toe, toe + g["stem_thickness_base"]
                                   * 1000.0])
    assert any(n.startswith("Shear key") and "assumed" in n
               for n in drawing.notes)
    assert any("assumed" in w for w in drawing.layout_warnings)
    # its depth is dimensioned; its assumed width and position are not
    assert not [d for d in drawing.dimensions if d.direction ==
                "horizontal" and abs(d.p1[1] - depth) < 1e-6]
    assert [d for d in drawing.dimensions if d.direction == "vertical"
            and {round(d.p1[1]), round(d.p2[1])} == {0, round(depth)}]


def test_stem_bars_stand_on_the_whole_bottom_mat():
    """The hook rests on the base's bottom steel, the bars along the wall
    included (it was drawn through them)."""
    description = real_descriptions()["plain"]
    drawing = wall_section_drawing(description)
    hooked = next(b for b in drawing.bars
                  if b.kind == "path" and len(b.points) == 3)
    tail_y = hooked.points[-1][1]
    lowest_dots = min(b.points[0][1] for b in drawing.bars
                      if b.kind == "dots")
    a, g = description["bars"][0], description["bars"][6]
    assert (a["mark"], g["mark"]) == ("A", "G")
    assert tail_y - lowest_dots >= (
        a["diameter_mm"] + g["diameter_mm"]) / 2 - 1e-6
    assert tail_y - lowest_dots < (a["diameter_mm"] + g["diameter_mm"])


def test_an_over_long_word_neither_leaves_its_column_nor_moves_the_sheet():
    """One long word in a note or the title used to run past the border,
    and the page, centred on the content, slid the sheet and its stamp off
    it. The page is now the sheet itself, and a long word is broken."""
    from dxf_export._fonts import text_width, wrap
    assert all(text_width(line, TEXT_H) <= 60.0
               for line in wrap("X" * 290, 60.0, TEXT_H))
    data = general()
    data["notes"] = ["X" * 290]
    data["title_block"]["title"] = "T" * 160
    files = render_sheet(Drawing.from_dict(data))
    page = pymupdf.open(stream=files["pdf"], filetype="pdf")[0]
    # every path but the white page behind the sheet
    rects = [d["rect"] for d in page.get_drawings()
             if d["rect"].width < page.rect.width - 1.0]
    margin = 10.0 / 25.4 * 72.0                 # the border, 10 mm in
    assert min(r.x0 for r in rects) == pytest.approx(margin, abs=1.5)
    assert min(r.y0 for r in rects) == pytest.approx(margin, abs=1.5)
    assert max(r.x1 for r in rects) == pytest.approx(
        page.rect.width - margin, abs=1.5)
    assert max(r.y1 for r in rects) == pytest.approx(
        page.rect.height - margin, abs=1.5)


def test_the_total_number_of_points_is_bounded():
    data = {"bars": [{"kind": "dots", "diameter_mm": 10,
                      "points": [[k, 100 * j] for k in range(5000)]}
                     for j in range(5)]}
    with pytest.raises(DrawingError, match="at most 20000 points"):
        Drawing.from_dict(data)
    description = wall_description()
    description["geometry"]["wall_height"] = 60.0
    description["bars"][2]["spacing_mm"] = 10.0
    with pytest.raises(DrawingError, match="dots"):
        wall_section_drawing(description)


def test_control_characters_are_dropped_from_text():
    drawing = Drawing.from_dict({"lines": [{"points": [[0, 0], [500, 0]]}],
                                 "notes": ["a\x00b\x07c"]})
    assert drawing.notes == ["a b c"]


def test_dimension_lines_are_the_thinnest_and_the_ground_is_stroked():
    weights = [w for _, _, w, _ in LAYERS.values() if w > 0]
    assert LAYERS["dims"][2] == min(weights)
    assert LAYERS["ground"][2] > LAYERS["dims"][2]
    doc, _ = build_dxf(wall_section_drawing(wall_description()))
    ground = [e for e in doc.modelspace()
              if e.dxf.layer == LAYERS["ground"][0]]
    assert len(ground) > 2 + 20         # the two ground lines and strokes
