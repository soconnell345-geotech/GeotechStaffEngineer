"""BACKLOG GM-313, the join: a reinforced cantilever wall's calc package
(``retaining_wall_package``, wall_type ``cantilever_reinforced``) ends with
its reinforcement sheet, drawn by ``dxf_export`` from the design's own
``reinforcement`` description, and writes the sheet's DXF and PNG beside
the package. Made-up walls only."""

import base64
import os

import pytest

pymupdf = pytest.importorskip("pymupdf")
ezdxf = pytest.importorskip("ezdxf")

import dxf_export                                           # noqa: E402
from funhouse_agent import dispatch as D                    # noqa: E402

#: A made-up 5 m wall with a live surcharge (too tall for letter at 1:50).
TALL = {"wall_height": 5.0, "base_width": 3.1, "toe_length": 0.65,
        "stem_thickness_top": 0.305, "stem_thickness_base": 0.5,
        "base_thickness": 0.5, "embedment_depth": 1.0, "surcharge": 12.0,
        "gamma_backfill": 19.0, "phi_backfill": 32.0, "q_allowable": 250.0,
        "fc_MPa": 28.0, "joint_spacing": 8.0,
        "wall_type": "cantilever_reinforced",
        "project_name": "Made-up wall project", "project_number": "MU-0001",
        "engineer": "A. Engineer", "checker": "B. Checker",
        "date": "2026-01-02"}

#: A made-up 3.5 m wall with a shear key (fits letter at 1:50).
LOW = {**TALL, "wall_height": 3.5, "base_width": 2.17, "toe_length": 0.46,
       "has_shear_key": True, "key_depth": 0.4}

TABLOID_IN = (17.0, 11.0)
LETTER_IN = (11.0, 8.5)


def _package(out_dir, name, params, fmt):
    r = D.call_agent("calc_package", "retaining_wall_package", {
        **params, "format": fmt,
        "output_path": os.path.join(str(out_dir), f"{name}.{fmt}")})
    assert "error" not in r, r
    assert r["file_exists"], r
    return r


def _pages(path):
    with pymupdf.open(path) as doc:
        return [(round(p.rect.width / 72, 2), round(p.rect.height / 72, 2))
                for p in doc]


def _texts(path, layer):
    doc = ezdxf.readfile(path)
    return [t.dxf.text for t in doc.modelspace().query("TEXT")
            if t.dxf.layer == layer]


def _key_x(path, depth_m):
    """The concrete outline's x (mm) at the key's bottom."""
    doc = ezdxf.readfile(path)
    (outline,) = [p for p in doc.modelspace().query("LWPOLYLINE")
                  if p.dxf.layer == "GSE-CONCRETE"]
    return sorted({round(x, 3) for x, y, *_ in outline.get_points()
                   if abs(y + depth_m * 1000.0) < 1e-6})


@pytest.fixture(scope="module")
def tall_pdf(tmp_path_factory):
    out = tmp_path_factory.mktemp("tall")
    return out, _package(out, "wall", TALL, "pdf")


@pytest.fixture(scope="module")
def tall_pdf_failed(tmp_path_factory):
    """The same package with the drawing made to fail."""
    out = tmp_path_factory.mktemp("failed")

    def broken(*args, **kwargs):
        raise RuntimeError("made to fail")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(dxf_export, "wall_section_drawing", broken)
        return out, _package(out, "wall", TALL, "pdf")


def test_the_pdf_package_ends_with_the_sheet_as_its_own_page(
        tall_pdf, tall_pdf_failed):
    out, r = tall_pdf
    _, failed = tall_pdf_failed
    pages, without = _pages(r["output_path"]), _pages(failed["output_path"])
    assert len(pages) == len(without) + 1
    assert pages[:-1] == without
    # the last page is the sheet's own size, landscape
    assert r["drawing_sheet"] == "tabloid" and pages[-1] == TABLOID_IN
    assert r["output_path"].endswith("wall.pdf")        # still the package
    stem = os.path.join(str(out), "wall_reinforcement")
    assert r["drawing_dxf_path"] == os.path.abspath(stem + ".dxf")
    assert r["drawing_png_path"] == os.path.abspath(stem + ".png")
    assert os.path.isfile(r["drawing_dxf_path"])
    assert os.path.isfile(r["drawing_png_path"])
    assert r["drawing_scale"] == "1:50"
    assert any("wider than to scale" in w for w in r["drawing_warnings"])
    assert not any("reinforcement sheet" in str(w)
                   for w in r.get("warnings") or [])


def test_the_dxf_reads_back_clean_with_the_designs_bars(tall_pdf):
    _, r = tall_pdf
    doc = ezdxf.readfile(r["drawing_dxf_path"])
    auditor = doc.audit()
    assert auditor.errors == [] and auditor.fixes == []
    bars = r["reinforcement"]["bars"]
    callouts = _texts(r["drawing_dxf_path"], "GSE-CALLOUTS")
    # one callout per bar set, its mark then its text: the design's own
    assert sorted(t for t in callouts if len(t) == 1) == sorted(
        b["mark"] for b in bars)
    joined = " ".join(callouts)
    for b in bars:
        assert f"{b['mark']} {b['callout']}" in joined, b["mark"]
    # the schedule: each set's mark, size and spacing as designed
    table = _texts(r["drawing_dxf_path"], "GSE-TABLE")
    for b in bars:
        i = table.index(b["mark"])
        assert table[i + 1] == b["size"]
        assert table[i + 2] == f"{b['spacing_mm']:.0f} mm"
    # the title block is the package's own header
    title = _texts(r["drawing_dxf_path"], "GSE-TITLE")
    for words in ("PROJECT: Made-up wall project", "NO.: MU-0001",
                  "DRAWN: A. Engineer", "CHECKED: B. Checker",
                  "DATE: 2026-01-02"):
        assert words in title, words
    assert dxf_export.STAMP in _texts(r["drawing_dxf_path"], "GSE-STAMP")


def test_a_tall_wall_goes_to_tabloid_because_letter_would_be_1_to_100(
        tall_pdf):
    _, r = tall_pdf
    on_letter = dxf_export.wall_section_drawing(r["reinforcement"],
                                                sheet={"size": "letter"})
    assert on_letter.sheet.scale > 50
    assert (r["drawing_sheet"], r["drawing_scale"]) == ("tabloid", "1:50")


def test_a_drawing_that_fails_costs_the_sheet_never_the_package(
        tall_pdf_failed):
    out, r = tall_pdf_failed
    assert r["status"] == "success" and os.path.isfile(r["output_path"])
    assert r["stem"]["Mu_kNm_per_m"] is not None       # the calculation
    assert r["drawing_dxf_path"] is None and r["drawing_png_path"] is None
    assert r["drawing_sheet"] is None
    (why,) = [w for w in r["warnings"] if "reinforcement sheet" in w]
    assert "RuntimeError: made to fail" in why
    assert "written without it" in why
    assert not [f for f in os.listdir(out)
                if f.startswith("wall_reinforcement.")]


@pytest.fixture(scope="module")
def low_html(tmp_path_factory):
    out = tmp_path_factory.mktemp("low")
    return out, _package(out, "low", {**LOW, "key_distance_from_toe": 1.0,
                                      "key_width": 0.35}, "html")


def test_an_html_package_shows_the_sheets_png_and_a_low_wall_stays_on_letter(
        low_html):
    _, r = low_html
    assert (r["drawing_sheet"], r["drawing_scale"]) == ("letter", "1:50")
    html = open(r["output_path"], encoding="utf-8").read()
    png = open(r["drawing_png_path"], "rb").read()
    assert base64.b64encode(png).decode("ascii") in html
    assert "Drawings" in html and "low_reinforcement.dxf" in html
    with pymupdf.open(stream=png, filetype="png") as img:
        page = img[0].rect
        assert page.width > page.height                 # landscape


def test_a_key_with_its_place_given_is_drawn_there(low_html):
    _, r = low_html
    g = r["reinforcement"]["geometry"]
    # the design gives the key's depth only; the package's place is drawn
    assert g["key_width"] is None and g["key_distance_from_toe"] is None
    assert _key_x(r["drawing_dxf_path"], 0.4) == [1000.0, 1350.0]
    notes = " ".join(_texts(r["drawing_dxf_path"], "GSE-NOTES"))
    assert "Shear key drawn under the stem" not in notes
    assert not any("assumed" in w for w in r["drawing_warnings"])


def test_a_key_without_its_place_is_drawn_under_the_stem_and_said(
        tmp_path):
    r = _package(tmp_path, "nokey", LOW, "html")
    assert _key_x(r["drawing_dxf_path"], 0.4) == [460.0, 960.0]
    notes = " ".join(_texts(r["drawing_dxf_path"], "GSE-NOTES"))
    assert "Shear key drawn under the stem" in notes
    assert any("assumed" in w for w in r["drawing_warnings"])


def test_render_figures_saves_the_sheets_png_with_the_figures(tmp_path):
    r = D.call_agent("calc_package", "render_figures", {
        "package": "retaining_wall_package", **LOW,
        "output_dir": str(tmp_path)})
    assert r["status"] == "success", r
    (sheet,) = [f for f in r["figures"]
                if f["title"] == "Reinforcement sheet"]
    assert sheet["file_exists"] and os.path.basename(
        sheet["output_path"]) == "reinforcement_sheet.png"
    assert sheet["index"] == r["n_figures"] == len(r["figures"]) > 1
    assert (r["drawing_sheet"], r["drawing_scale"]) == ("letter", "1:50")
    assert "drawing_dxf_path" not in r          # figures only: no package
