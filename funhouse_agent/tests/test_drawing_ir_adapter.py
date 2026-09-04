"""Tests for the drawing_ir funhouse adapter (digitize -> query -> get_entities)."""

import pytest

ezdxf = pytest.importorskip("ezdxf")

from funhouse_agent.adapters.drawing_ir_adapter import (
    METHOD_INFO, METHOD_REGISTRY, QUERY_NAMES,
)
from funhouse_agent.dispatch import (
    ANALYSIS_MODULES, call_agent, describe_method, list_agents, list_methods,
)

REQUIRED_INFO_FIELDS = {"category", "brief", "parameters", "returns"}


@pytest.fixture
def dxf_path(tmp_path):
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 6
    msp = doc.modelspace()
    msp.add_lwpolyline([(0, 10), (10, 10), (20, 5), (30, 5)],
                       dxfattribs={"layer": "SURFACE"})
    msp.add_line((0, 0), (30, 0), dxfattribs={"layer": "BASE"})
    msp.add_text("Clay", dxfattribs={"layer": "NOTES", "insert": (16, 7),
                                     "height": 1.0})
    p = tmp_path / "s.dxf"
    doc.saveas(str(p))
    return str(p)


class TestMethodInfo:
    def test_keys_match(self):
        assert set(METHOD_INFO) == set(METHOD_REGISTRY)

    def test_required_fields(self):
        for name, info in METHOD_INFO.items():
            for f in REQUIRED_INFO_FIELDS:
                assert f in info, f"{name} missing {f}"

    def test_expected_methods(self):
        assert set(METHOD_REGISTRY) == {
            "digitize_drawing", "query_drawing", "get_entities",
            "snip_region", "search_drawing_set"}

    def test_query_allowed_values_match_registry(self):
        allowed = METHOD_INFO["query_drawing"]["parameters"]["query"][
            "allowed_values"]
        assert set(allowed) == set(QUERY_NAMES)

    def test_source_allowed_values(self):
        av = METHOD_INFO["digitize_drawing"]["parameters"]["source"][
            "allowed_values"]
        assert set(av) == {"auto", "dxf", "pdf_vector", "raster"}


class TestDispatchVisibility:
    def test_in_analysis_modules(self):
        assert "drawing_ir" in ANALYSIS_MODULES
        assert "drawing_ir" in list_agents()

    def test_list_methods(self):
        result = list_methods("drawing_ir")
        total = sum(len(v) for v in result.values())
        assert total == 5

    def test_describe_method(self):
        info = describe_method("drawing_ir", "digitize_drawing")
        assert info["category"] == "Drawing IR"
        assert "file_path" in info["parameters"]


class TestEndToEnd:
    def test_digitize_returns_handle_and_stats(self, dxf_path):
        r = call_agent("drawing_ir", "digitize_drawing", {"file_path": dxf_path})
        assert "error" not in r
        assert r["handle"].startswith("dwg_")
        assert r["source"] == "dxf"
        assert r["counts_by_type"]["polyline"] == 1
        assert r["page"]["coordinate_space"] == "model"
        # full IR is NOT dumped by default
        assert "entities" not in r

    def test_auto_source_detection(self, dxf_path):
        r = call_agent("drawing_ir", "digitize_drawing",
                       {"file_path": dxf_path, "source": "auto"})
        assert r["source"] == "dxf"

    def test_query_and_get_entities_flow(self, dxf_path):
        r = call_agent("drawing_ir", "digitize_drawing", {"file_path": dxf_path})
        h = r["handle"]

        q = call_agent("drawing_ir", "query_drawing",
                       {"handle": h, "query": "text_items",
                        "params": {"pattern": "clay"}})
        assert q["n_results"] == 1
        assert q["result"][0]["content"] == "Clay"

        surf = call_agent("drawing_ir", "query_drawing",
                          {"handle": h, "query": "candidate_ground_surface"})
        cid = surf["result"]["candidate"]["id"]

        ge = call_agent("drawing_ir", "get_entities",
                        {"handle": h, "ids": [cid]})
        ent = ge["entities"][0]
        assert ent["type"] == "polyline"
        assert ent["vertices"][0] == [0.0, 10.0]

    def test_bbox_query(self, dxf_path):
        r = call_agent("drawing_ir", "digitize_drawing", {"file_path": dxf_path})
        q = call_agent("drawing_ir", "query_drawing",
                       {"handle": r["handle"], "query": "entities_in_bbox",
                        "params": {"x_min": 15, "y_min": 6, "x_max": 20,
                                   "y_max": 8}})
        assert any(e["type"] == "text" for e in q["result"])


class TestErrors:
    def test_missing_file_path(self):
        r = call_agent("drawing_ir", "digitize_drawing", {})
        assert "error" in r and "file_path" in r["error"]

    def test_unknown_param_rejected(self, dxf_path):
        r = call_agent("drawing_ir", "digitize_drawing",
                       {"file_path": dxf_path, "bogus": 1})
        assert "error" in r and "bogus" in r["error"]

    def test_bad_handle(self):
        r = call_agent("drawing_ir", "query_drawing",
                       {"handle": "nope", "query": "text_items"})
        assert "error" in r and "handle" in r["error"]

    def test_unknown_query(self, dxf_path):
        r = call_agent("drawing_ir", "digitize_drawing", {"file_path": dxf_path})
        q = call_agent("drawing_ir", "query_drawing",
                       {"handle": r["handle"], "query": "frobnicate"})
        assert "error" in q and "Unknown query" in q["error"]

    def test_query_missing_required_param(self, dxf_path):
        r = call_agent("drawing_ir", "digitize_drawing", {"file_path": dxf_path})
        q = call_agent("drawing_ir", "query_drawing",
                       {"handle": r["handle"], "query": "lines_by_angle",
                        "params": {"min_deg": 0}})
        assert "error" in q and "max_deg" in q["error"]

    def test_query_unknown_param(self, dxf_path):
        r = call_agent("drawing_ir", "digitize_drawing", {"file_path": dxf_path})
        q = call_agent("drawing_ir", "query_drawing",
                       {"handle": r["handle"], "query": "text_items",
                        "params": {"nope": 1}})
        assert "error" in q and "nope" in q["error"]

    def test_auto_source_unknown_extension(self, tmp_path):
        p = tmp_path / "x.xyz"
        p.write_text("nope")
        r = call_agent("drawing_ir", "digitize_drawing", {"file_path": str(p)})
        assert "error" in r


# ---------------------------------------------------------------------------
# Phase 2: composition queries, snip_region, search_drawing_set
# ---------------------------------------------------------------------------

fitz = pytest.importorskip("fitz")

from planlens.ir.tests.construct_fixtures import (  # noqa: E402
    build_synthetic_dimension_pdf, build_synthetic_drawing_set_pdf,
)
from planlens.ir.tests.leader_fixtures import (  # noqa: E402
    build_synthetic_leader_pdf,
)


class TestCompositionQueriesViaAdapter:
    def test_find_leaders_with_exclusion(self, tmp_path):
        path, gt = build_synthetic_leader_pdf(tmp_path, n_leaders=3,
                                              include_decoys=True)
        r = call_agent("drawing_ir", "digitize_drawing", {"file_path": path})
        q = call_agent("drawing_ir", "query_drawing",
                       {"handle": r["handle"], "query": "find_leaders",
                        "params": {"exclude_dimensions": True}})
        assert "error" not in q
        assert q["n_results"] >= 3
        assert all(p["proposal_only"] for p in q["result"])

    def test_find_dimensions(self, tmp_path):
        path, gt = build_synthetic_dimension_pdf(tmp_path)
        r = call_agent("drawing_ir", "digitize_drawing", {"file_path": path})
        q = call_agent("drawing_ir", "query_drawing",
                       {"handle": r["handle"], "query": "find_dimensions",
                        "params": {"min_confidence": 0.5}})
        assert "error" not in q
        assert q["n_results"] == len(gt["dimensions"])

    def test_text_anchored_geometry_runtime_pattern(self, tmp_path):
        path, gt = build_synthetic_leader_pdf(tmp_path, n_leaders=3,
                                              include_decoys=False)
        r = call_agent("drawing_ir", "digitize_drawing", {"file_path": path})
        q = call_agent("drawing_ir", "query_drawing",
                       {"handle": r["handle"],
                        "query": "text_anchored_geometry",
                        "params": {"pattern": "TYP"}})
        assert "error" not in q and q["n_results"] >= 1
        assert q["result"][0]["points_at"] is not None

    def test_entities_ending_near(self, tmp_path):
        path, gt = build_synthetic_leader_pdf(tmp_path, n_leaders=1,
                                              include_decoys=False)
        L = gt["leaders"][0]
        r = call_agent("drawing_ir", "digitize_drawing", {"file_path": path})
        q = call_agent("drawing_ir", "query_drawing",
                       {"handle": r["handle"], "query": "entities_ending_near",
                        "params": {"x": L.tip_xy[0], "y": L.tip_xy[1],
                                   "radius": 2.0}})
        assert "error" not in q and q["n_results"] >= 1


class TestSnipRegion:
    def test_ir_frame_conversion_and_save(self, tmp_path):
        path, gt = build_synthetic_leader_pdf(tmp_path, n_leaders=1,
                                              include_decoys=False)
        L = gt["leaders"][0]
        out = str(tmp_path / "tip.png")
        r = call_agent("drawing_ir", "snip_region",
                       {"file_path": path, "output_path": out,
                        "bbox": [L.tip_xy[0] - 20, L.tip_xy[1] - 20,
                                 L.tip_xy[0] + 20, L.tip_xy[1] + 20],
                        "marks": [[L.tip_xy[0], L.tip_xy[1], "1"]]})
        assert "error" not in r
        import os
        assert os.path.isfile(r["saved"])
        with open(r["saved"], "rb") as f:
            assert f.read(8) == b"\x89PNG\r\n\x1a\n"

    def test_pdf_frame_passthrough(self, tmp_path):
        path, _gt = build_synthetic_leader_pdf(tmp_path, n_leaders=1,
                                               include_decoys=False)
        out = str(tmp_path / "raw.png")
        r = call_agent("drawing_ir", "snip_region",
                       {"file_path": path, "output_path": out,
                        "bbox": [100, 100, 160, 160], "frame": "pdf"})
        assert "error" not in r

    def test_bad_frame_rejected(self, tmp_path):
        path, _gt = build_synthetic_leader_pdf(tmp_path, n_leaders=1,
                                               include_decoys=False)
        r = call_agent("drawing_ir", "snip_region",
                       {"file_path": path, "output_path": str(tmp_path / "x.png"),
                        "frame": "upside_down"})
        assert "error" in r


class TestSearchDrawingSet:
    def test_text_counts_across_pages(self, tmp_path):
        path, gt = build_synthetic_drawing_set_pdf(tmp_path)
        r = call_agent("drawing_ir", "search_drawing_set",
                       {"file_paths": path, "pattern": "W1"})
        assert "error" not in r
        assert r["total_count"] == sum(gt["w1_counts_by_page"].values())
        by_page = {p["page"]: p["count"] for p in r["files"][0]["pages"]}
        assert by_page == gt["w1_counts_by_page"]

    def test_leader_construct_counts(self, tmp_path):
        path, gt = build_synthetic_drawing_set_pdf(tmp_path)
        r = call_agent("drawing_ir", "search_drawing_set",
                       {"file_paths": [path], "construct": "leaders",
                        "min_confidence": 0.5})
        assert "error" not in r
        assert r["proposal_only"] is True
        by_page = {p["page"]: p["count"] for p in r["files"][0]["pages"]}
        assert by_page == gt["leader_counts_by_page"]

    def test_construct_plus_pattern_filter(self, tmp_path):
        path, gt = build_synthetic_drawing_set_pdf(tmp_path)
        r = call_agent("drawing_ir", "search_drawing_set",
                       {"file_paths": path, "construct": "leaders",
                        "pattern": "W1", "min_confidence": 0.5})
        assert r["total_count"] == sum(gt["w1_counts_by_page"].values())

    def test_title_block_pattern_matches_contained_texts(self, tmp_path):
        # Title-block proposals carry "texts" (a list of contained text
        # items), not "text" — the pattern filter must match against them.
        from planlens.ir.tests.construct_fixtures import (
            build_synthetic_title_block_pdf)
        path, gt = build_synthetic_title_block_pdf(tmp_path)
        hit = call_agent("drawing_ir", "search_drawing_set",
                         {"file_paths": path, "construct": "title_block",
                          "pattern": "S-101"})
        assert "error" not in hit
        assert hit["total_count"] >= 1
        miss = call_agent("drawing_ir", "search_drawing_set",
                          {"file_paths": path, "construct": "title_block",
                           "pattern": "NOT-ON-THIS-SHEET"})
        assert miss["total_count"] == 0

    def test_set_ir_cache_reuse(self, tmp_path):
        # Phase 3: repeated set queries over the same file reuse the
        # digitized IR (identical results, one cache entry per page).
        from funhouse_agent.adapters import drawing_ir_adapter as mod
        path, gt = build_synthetic_drawing_set_pdf(tmp_path)
        mod._SET_IR_CACHE.clear()
        mod._SET_IR_ORDER.clear()
        r1 = call_agent("drawing_ir", "search_drawing_set",
                        {"file_paths": path, "pattern": "W1"})
        n_cached = len(mod._SET_IR_CACHE)
        assert n_cached >= 1
        r2 = call_agent("drawing_ir", "search_drawing_set",
                        {"file_paths": path, "pattern": "W1"})
        assert len(mod._SET_IR_CACHE) == n_cached  # no re-digitization
        assert r1["total_count"] == r2["total_count"]

    def test_ocr_text_param_accepted_without_engine_use(self, tmp_path):
        # ocr_text=true on a page that HAS a text layer must not invoke
        # the OCR engine at all (the trigger is no_text_layer only).
        path, gt = build_synthetic_drawing_set_pdf(tmp_path)
        r = call_agent("drawing_ir", "search_drawing_set",
                       {"file_paths": path, "pattern": "W1",
                        "ocr_text": True})
        assert "error" not in r
        assert r["total_count"] == sum(gt["w1_counts_by_page"].values())

    def test_requires_pattern_or_construct(self, tmp_path):
        path, _gt = build_synthetic_drawing_set_pdf(tmp_path)
        r = call_agent("drawing_ir", "search_drawing_set",
                       {"file_paths": path})
        assert "error" in r

    def test_unknown_construct_rejected(self, tmp_path):
        path, _gt = build_synthetic_drawing_set_pdf(tmp_path)
        r = call_agent("drawing_ir", "search_drawing_set",
                       {"file_paths": path, "construct": "flux_capacitors"})
        assert "error" in r
