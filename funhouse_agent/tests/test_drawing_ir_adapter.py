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
            "digitize_drawing", "query_drawing", "get_entities"}

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
        assert total == 3

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
