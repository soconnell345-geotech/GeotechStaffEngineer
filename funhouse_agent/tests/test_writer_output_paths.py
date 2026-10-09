"""Every file a ``call_agent`` writer saves lands in the conversation's folder.

Foundry brief 5 (2026-10-08): ``subsurface.write_diggs`` was valid in 6 of 6
runs and written to ``/tmp`` every time, because its own description gave
``'/tmp/site.diggs.xml'`` as the example and the writer kept whatever path it
was given. The suite never saw the file; in the app only the post-turn import
rescued it, and a DXF (reported as ``filepath``) was not rescued even there.

Now, once a host has set a working folder (``GEOTECH_DEFAULT_OUTPUT_DIR``,
the mechanism ``write_docx`` and ``annotate_document`` already use), every
writer reached through ``call_agent`` keeps the FILE NAME it was given and
writes into that folder, and its result says so. Library callers with no
folder keep exactly today's behaviour.
"""

from __future__ import annotations

import json
import os
import uuid

import pytest

from funhouse_agent._fileio import (
    DEFAULT_OUTPUT_DIR_ENV, host_output_dir, into_working_folder,
)
from funhouse_agent.dispatch import call_agent


def _j(value):
    return json.loads(value) if isinstance(value, str) else value


@pytest.fixture
def folder(tmp_path, monkeypatch):
    work = tmp_path / "conversation" / "files"
    work.mkdir(parents=True)
    monkeypatch.setenv(DEFAULT_OUTPUT_DIR_ENV, str(work))
    return work


@pytest.fixture
def elsewhere(tmp_path):
    """A directory outside the working folder, standing in for /tmp."""
    d = tmp_path / "tmp_elsewhere"
    d.mkdir()
    return d


# -- the rule ---------------------------------------------------------------

def test_into_working_folder(folder, elsewhere):
    f = str(folder)
    assert into_working_folder("site.diggs.xml") == os.path.join(f, "site.diggs.xml")
    assert into_working_folder("figs/a.png") == os.path.join(f, "figs", "a.png")
    inside = os.path.join(f, "x.pdf")
    assert into_working_folder(inside) == inside
    assert into_working_folder(str(elsewhere / "x.xml")) == os.path.join(f, "x.xml")
    assert into_working_folder("/tmp/harbour.diggs.xml") == os.path.join(
        f, "harbour.diggs.xml")
    assert into_working_folder(str(elsewhere), is_dir=True) == f
    assert into_working_folder("") == ""


def test_no_host_folder_changes_nothing(monkeypatch, elsewhere):
    monkeypatch.delenv(DEFAULT_OUTPUT_DIR_ENV, raising=False)
    assert host_output_dir() is None
    for p in ("/tmp/x.xml", "x.xml", str(elsewhere / "y.dxf")):
        assert into_working_folder(p) == p


# -- the writers, through the dispatch layer the agent uses ----------------

def _m(v, unit="m"):
    return {"value": v, "unit": unit}


BORING = {"investigation_id": "B-1", "kind": "boring", "depth_unit": "m",
          "total_depth": _m(6.0),
          "layers": [{"top": _m(0.0), "bottom": _m(6.0),
                      "description": "Brown silty SAND", "uscs": "SM"}]}


def test_write_diggs_to_tmp_lands_in_the_folder(folder):
    """The brief-5 call shape: an absolute /tmp path copied off the old
    example."""
    name = f"harbour_road_{uuid.uuid4().hex[:8]}.diggs.xml"
    res = _j(call_agent("subsurface", "write_diggs", {
        "investigations": [BORING], "output_path": f"/tmp/{name}"}))
    assert "error" not in res, res
    # named in the working folder, never by the server path (A6)
    assert res["output_path"] == name
    assert (folder / name).is_file() and res["file_exists"] is True
    assert not os.path.exists(os.path.join(os.path.abspath("/tmp"), name))
    assert "working folder" in res["output_note"]
    assert f"/tmp/{name}" in res["output_note"]      # says what was replaced


def test_write_diggs_bare_name_lands_in_the_folder_not_the_cwd(folder):
    res = _j(call_agent("subsurface", "write_diggs", {
        "investigations": [BORING], "output_path": "site.diggs.xml"}))
    assert res["output_path"] == "site.diggs.xml"
    assert (folder / "site.diggs.xml").is_file()
    assert "replaced" not in res["output_note"]


def test_the_description_no_longer_suggests_tmp():
    from funhouse_agent.dispatch import describe_method
    for agent, method in (("subsurface", "write_diggs"),
                          ("subsurface", "plot_parameter_vs_depth"),
                          ("dxf_export", "export_geometry_to_dxf")):
        desc = json.dumps(_j(describe_method(agent, method)))
        assert "/tmp" not in desc, (agent, method)


def test_dxf_export_lands_in_the_folder_and_reports_output_path(folder,
                                                                  elsewhere):
    pytest.importorskip("ezdxf")
    res = _j(call_agent("dxf_export", "export_geometry_to_dxf", {
        "surface_points": [[0, 10], [20, 10], [40, 5]],
        "output_path": str(elsewhere / "section.dxf")}))
    assert "error" not in res, res
    want = os.path.join(str(folder), "section.dxf")
    # output_path is the key the app's capture recognises a written file by
    assert res["output_path"] == res["filepath"] == "section.dxf"
    assert os.path.isfile(want) and not (elsewhere / "section.dxf").exists()
    from webapp.output_capture import paths_in
    captured = [p if os.path.isabs(p) else os.path.join(str(folder), p)
                for p in paths_in(json.dumps(res))[0]]
    assert want in captured


def test_html_to_pdf_lands_in_the_folder(folder, elsewhere):
    pytest.importorskip("fitz")
    res = _j(call_agent("calc_package", "html_to_pdf", {
        "html": "<h1>Memo</h1><p>Bearing 150 kPa.</p>",
        "output_path": str(elsewhere / "memo.pdf")}))
    assert "error" not in res, res
    assert res["output_path"] == "memo.pdf"
    assert (folder / "memo.pdf").is_file()


def test_plot_data_lands_in_the_folder(folder, elsewhere):
    pytest.importorskip("matplotlib")
    res = _j(call_agent("profile_figure", "plot_data", {
        "series": [{"x": [1, 2, 3], "y": [4, 5, 6], "label": "a"}],
        "output_path": str(elsewhere / "spt.png")}))
    assert "error" not in res, res
    assert res["output_path"] == "spt.png"
    assert (folder / "spt.png").is_file()


def test_snip_region_lands_in_the_folder(folder, elsewhere, tmp_path):
    fitz = pytest.importorskip("fitz")
    doc = fitz.open()
    doc.new_page(width=300, height=200).insert_text((40, 60), "B-1")
    src = tmp_path / "sheet.pdf"
    doc.save(str(src))
    res = _j(call_agent("drawing_ir", "snip_region", {
        "file_path": str(src), "output_path": str(elsewhere / "crop.png"),
        "bbox": [20, 30, 120, 80], "frame": "pdf"}))
    assert "error" not in res, res
    assert res["saved"] == "crop.png"
    assert (folder / "crop.png").is_file()


def test_a_library_caller_keeps_its_path(monkeypatch, elsewhere):
    """No host folder: the file goes exactly where it was asked, and the
    result carries no note about a folder there is not."""
    monkeypatch.delenv(DEFAULT_OUTPUT_DIR_ENV, raising=False)
    out = elsewhere / "lib.diggs.xml"
    res = _j(call_agent("subsurface", "write_diggs", {
        "investigations": [BORING], "output_path": str(out)}))
    assert res["output_path"] == str(out) and out.is_file()
    assert "output_note" not in res
