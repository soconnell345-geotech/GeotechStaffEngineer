"""Offline tests for the bundled eval sample files (NO API, NO network).

v5.1 closes the "file-input modules aren't exercised" eval-suite gap
(docs/V5.0_TODO.md section C) by bundling real sample input files under
``funhouse_agent/eval_samples/`` and pointing the file-input suite questions at
them. THE GATE: every sample must parse through the REAL underlying module API
(not a mock) —

  * sample_borehole.ags    -> subsurface_characterization.formats.ags4.read_ags4
  * sample_cpt.gef         -> subsurface_characterization.formats.gef.parse_cpt_file
  * sample_site_diggs.xml  -> subsurface_characterization.parse_diggs
  * sample_section.dxf     -> dxf_import.discover_layers (+ parse_dxf_geometry)
  * sample_section.pdf     -> pdf_import.discover_pdf_content (+ extract_vector_geometry)

hvsrpy / swprocess get NO sample file on purpose: their adapters take in-memory
arrays (ns/ew/vt or traces + dt), not file paths — there is nothing for a file
to feed (see funhouse_agent/adapters/hvsrpy_adapter.py / swprocess_adapter.py).

Also covered: the ``{sample_path}`` token convention — ``load_suite`` expands it
to an absolute path resolved against the installed ``funhouse_agent`` package
(works from any CWD / an installed wheel).

Run from the worktree root with the venv python::

    .venv/Scripts/python.exe -m pytest \
        funhouse_agent/deep/tests/test_eval_samples.py -v
"""

import os
from pathlib import Path

import pytest

from funhouse_agent.deep import eval_harness as eh


SAMPLES = [
    "sample_borehole.ags",
    "sample_cpt.gef",
    "sample_section.dxf",
    "sample_section.pdf",
    "sample_site_diggs.xml",
]


# ===========================================================================
# Path resolution (the {sample_path} convention)
# ===========================================================================

def test_all_samples_resolve_and_exist():
    """Every bundled sample resolves to an absolute, existing file."""
    for name in SAMPLES:
        p = eh.resolve_sample_path(name)
        assert p.is_absolute(), name
        assert p.exists(), f"missing sample: {p}"


def test_resolve_sample_path_is_cwd_independent(tmp_path, monkeypatch):
    """Resolution must not depend on the process CWD (wheel-install safety)."""
    before = eh.resolve_sample_path("sample_cpt.gef")
    monkeypatch.chdir(tmp_path)
    after = eh.resolve_sample_path("sample_cpt.gef")
    assert before == after
    assert after.exists()


def test_load_suite_expands_sample_path_token():
    """load_suite replaces {sample_path} with the resolved absolute path."""
    suite = eh.load_suite()
    file_questions = [q for q in suite if q.get("sample_file")]
    assert len(file_questions) >= 5  # PGEF/AGS/DIGGS/DXFI/PDF (+ SUB)
    for q in file_questions:
        assert eh.SAMPLE_PATH_TOKEN not in q["question"], q["id"]
        resolved = str(eh.resolve_sample_path(q["sample_file"]))
        assert resolved in q["question"], q["id"]
        assert Path(resolved).exists(), q["id"]


def test_suite_sample_files_are_all_bundled():
    """Every sample_file named in the suite exists under eval_samples/."""
    suite = eh.load_suite()
    referenced = {q["sample_file"] for q in suite if q.get("sample_file")}
    assert referenced  # the suite does reference samples
    for name in referenced:
        assert name in SAMPLES, f"suite references unbundled sample {name}"


# ===========================================================================
# THE GATE: every sample parses through the REAL module API
# ===========================================================================

def test_ags4_sample_parses():
    """sample_borehole.ags reads via python-ags4 with the expected groups."""
    from subsurface_characterization.formats.ags4 import read_ags4
    r = read_ags4(filepath=str(eh.resolve_sample_path("sample_borehole.ags")))
    d = r.to_dict()
    assert d["n_groups"] == 3
    assert set(d["group_names"]) == {"PROJ", "HOLE", "ISPT"}
    assert d["group_row_counts"]["ISPT"] == 6  # 3 SPTs in each of 2 boreholes


def test_ags4_sample_validates():
    """sample_borehole.ags passes AGS4 rule validation with zero errors."""
    from subsurface_characterization.formats.ags4 import validate_ags4
    r = validate_ags4(filepath=str(eh.resolve_sample_path("sample_borehole.ags")))
    d = r.to_dict()
    # The sample is intentionally minimal (no DICT group etc), so warnings/FYI
    # are acceptable; hard errors are not.
    assert d["n_errors"] == 0, d.get("errors")


def test_gef_cpt_sample_parses():
    """sample_cpt.gef parses via pygef with the suite's expected key numbers."""
    from subsurface_characterization.formats.gef import parse_cpt_file
    r = parse_cpt_file(file_path=str(eh.resolve_sample_path("sample_cpt.gef")))
    d = r.to_dict()
    assert d["n_points"] == 12
    assert d["final_depth_m"] == pytest.approx(6.0)
    assert d["gwl_m"] == pytest.approx(1.5)
    assert max(d["q_c_kPa"]) == pytest.approx(15000.0)  # 15 MPa peak


def test_diggs_sample_parses():
    """sample_site_diggs.xml parses via parse_diggs into 2 borings cleanly."""
    from subsurface_characterization import parse_diggs
    r = parse_diggs(filepath=str(eh.resolve_sample_path("sample_site_diggs.xml")))
    assert r.n_investigations == 2
    assert r.n_measurements == 6        # 6 SPT DrivenPenetrationTests
    assert r.n_lithology_intervals == 5
    assert r.warnings == []
    ids = {inv.investigation_id for inv in r.site.investigations}
    assert ids == {"B-1", "B-2"}


def test_dxf_sample_discovers_and_parses_geometry():
    """sample_section.dxf: discover_layers finds the slope layers AND the
    geometry parses end-to-end through parse_dxf_geometry."""
    from dxf_import import discover_layers, parse_dxf_geometry, LayerMapping
    path = str(eh.resolve_sample_path("sample_section.dxf"))

    disc = discover_layers(filepath=path).to_dict()
    names = {ly["name"] for ly in disc["layers"]}
    assert {"GROUND_SURFACE", "CLAY_TOP", "GWT"} <= names

    parsed = parse_dxf_geometry(
        filepath=path,
        layer_mapping=LayerMapping(
            surface="GROUND_SURFACE",
            # {dxf_layer_name: soil_name} — the boundary is the soil's BOTTOM.
            soil_boundaries={"CLAY_TOP": "clay"},
            water_table="GWT",
        ),
        units="m",
    ).to_dict()
    assert len(parsed["surface_points"]) >= 2
    assert "clay" in parsed["boundary_profiles"]
    assert parsed["gwt_points"]  # the GWT polyline came through


def test_pdf_sample_discovers_and_extracts_vectors():
    """sample_section.pdf: discover_pdf_content sees the drawing AND
    extract_vector_geometry pulls vector paths from it."""
    from pdf_import import discover_pdf_content, extract_vector_geometry
    path = str(eh.resolve_sample_path("sample_section.pdf"))

    disc = discover_pdf_content(filepath=path)
    assert disc["n_drawings"] == 5
    texts = " ".join(b["text"] for b in disc["text_blocks"])
    assert "FILL" in texts and "CLAY" in texts

    extracted = extract_vector_geometry(filepath=path).to_dict()
    # The vector method must find SOME geometry in the drawing.
    n_geom = (len(extracted.get("surface_points") or [])
              + sum(len(v) for v in (extracted.get("boundary_profiles") or {}).values())
              + len(extracted.get("gwt_points") or [])
              + len(extracted.get("unassigned_paths") or []))
    assert n_geom > 0, f"no vector geometry extracted: {list(extracted)}"


def test_hvsrpy_swprocess_need_no_sample_file():
    """hvsrpy/swprocess adapters are array-in (no file path parameter) — the
    documented reason they get no sample file. Guard that this stays true; if a
    file_path parameter ever appears, a sample should be added."""
    from funhouse_agent.adapters import hvsrpy_adapter, swprocess_adapter
    for adapter in (hvsrpy_adapter, swprocess_adapter):
        for method, info in adapter.METHOD_INFO.items():
            assert "file_path" not in info["parameters"], (
                f"{adapter.__name__}.{method} grew a file_path parameter — "
                "add an eval sample for it"
            )
