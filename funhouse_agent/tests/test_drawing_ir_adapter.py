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
# DXF block explosion: the ingest flag must be reachable AND reported
# ---------------------------------------------------------------------------

@pytest.fixture
def blocks_dxf_path(tmp_path):
    """A sheet whose title-block border lives in a BLOCK, with the real
    section line-work drawn directly in model space — the shape that makes
    the two populations distinguishable (2 direct entities, +2 exploded)."""
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 6
    blk = doc.blocks.new("TITLEBLK")
    blk.add_lwpolyline([(0, 0), (120, 0), (120, 90), (0, 90)], close=True)
    blk.add_line((0, 10), (120, 10))
    msp = doc.modelspace()
    msp.add_lwpolyline([(2, 40), (30, 44), (60, 41), (100, 38)],
                       dxfattribs={"layer": "SURFACE"})
    msp.add_line((5, 20), (95, 20), dxfattribs={"layer": "BASE"})
    msp.add_blockref("TITLEBLK", (0, 0))
    p = tmp_path / "blocks.dxf"
    doc.saveas(str(p))
    return str(p)


def _n_entities(result):
    return sum((result.get("counts_by_type") or {}).values())


class TestBlockExplosionPassthrough:
    """explode_blocks silently changes the entity population every raw
    geometry query ranks over, so it must be BOTH steerable and visible."""

    def test_default_explodes_and_reports_block_share(self, blocks_dxf_path):
        r = call_agent("drawing_ir", "digitize_drawing",
                       {"file_path": blocks_dxf_path})
        assert "error" not in r
        # 2 directly drawn + the INSERT itself + 2 exploded block primitives.
        assert r["n_block_entities"] == 2
        assert _n_entities(r) > r["n_block_entities"]

    def test_explode_blocks_false_is_accepted(self, blocks_dxf_path):
        # Used to raise "unknown parameter(s) ['explode_blocks']" — there was
        # no way to ask for only the directly drawn model-space work.
        r = call_agent("drawing_ir", "digitize_drawing",
                       {"file_path": blocks_dxf_path, "explode_blocks": False})
        assert "error" not in r
        assert "n_block_entities" not in r
        on = call_agent("drawing_ir", "digitize_drawing",
                        {"file_path": blocks_dxf_path})
        assert _n_entities(r) == _n_entities(on) - on["n_block_entities"]

    def test_flag_changes_what_geometry_queries_rank_over(
            self, blocks_dxf_path):
        # The consequence that matters: with the block exploded, the widest
        # path on the sheet is the stamped title-block border (120 wide), not
        # the directly drawn surface (98 wide).
        def widest(params):
            r = call_agent("drawing_ir", "digitize_drawing", params)
            q = call_agent("drawing_ir", "query_drawing",
                           {"handle": r["handle"],
                            "query": "candidate_ground_surface"})
            return q["result"]["width"]

        direct_only = widest({"file_path": blocks_dxf_path,
                              "explode_blocks": False})
        assert direct_only == pytest.approx(98.0)
        assert widest({"file_path": blocks_dxf_path}) >= direct_only

    def test_documented_in_method_info(self):
        p = METHOD_INFO["digitize_drawing"]["parameters"]["explode_blocks"]
        assert p["default"] is True
        assert "block" in p["description"].lower()
        assert "n_block_entities" in METHOD_INFO["digitize_drawing"]["returns"]


class TestExplodeFlagAgainstAnOlderPlanlens:
    """The DXF leg used to pass ``explode_blocks=`` unconditionally, so on a
    RELEASED planlens 0.1.0 — which the app pin ``>=0.1`` resolved to, and
    whose ``from_dxf`` has no such parameter (verified against the published
    wheel) — every ``source='dxf'`` call raised TypeError. Neither repo's
    suite reached it: both develop against editable installs. The pin is now
    ``>=0.2``; this is the belt-and-braces half, so a mismatched environment
    degrades visibly instead of crashing."""

    @staticmethod
    def _pin_floor():
        import pathlib
        import re
        root = pathlib.Path(__file__).resolve().parents[2]
        text = (root / "pyproject.toml").read_text(encoding="utf-8")
        m = re.search(r'"planlens\[raster\]>=([0-9.]+)"', text)
        assert m, "planlens dependency line not found in pyproject.toml"
        return tuple(int(v) for v in m.group(1).split("."))

    def test_pin_floor_covers_the_explode_blocks_parameter(self):
        # The floor must not be below the version that introduced the
        # parameter the adapter's DXF leg wants to pass.
        assert self._pin_floor() >= (0, 2)

    def test_installed_planlens_supports_the_flag(self):
        from funhouse_agent.adapters.drawing_ir_adapter import (
            _dxf_supports_explode,
        )
        assert _dxf_supports_explode() is True

    def _old_planlens(self, monkeypatch, blocks_dxf_path):
        """Stand in a 0.1.x-shaped ``from_dxf`` (no ``explode_blocks``)."""
        import planlens.ir as pir
        real = pir.from_dxf

        def old_from_dxf(filepath=None, content=None, units=None,
                         flip_y=False, name="DXF import"):
            return real(filepath=filepath, content=content, units=units,
                        flip_y=flip_y, name=name, explode_blocks=False)

        monkeypatch.setattr(pir, "from_dxf", old_from_dxf)

    def test_degrades_with_a_loud_note_instead_of_typeerror(
            self, monkeypatch, blocks_dxf_path):
        self._old_planlens(monkeypatch, blocks_dxf_path)
        r = call_agent("drawing_ir", "digitize_drawing",
                       {"file_path": blocks_dxf_path})
        assert "error" not in r          # was: TypeError from from_dxf
        assert r["blocks_exploded"] is False
        assert "planlens>=0.2" in r["note"]
        # And it really is the unexploded population, not a silent success.
        assert "n_block_entities" not in r

    def test_explode_false_on_an_old_planlens_is_not_a_degrade(
            self, monkeypatch, blocks_dxf_path):
        # 0.1.x ingest IS explode_blocks=false, so that request is served
        # exactly — no warning, because nothing was lost.
        self._old_planlens(monkeypatch, blocks_dxf_path)
        r = call_agent("drawing_ir", "digitize_drawing",
                       {"file_path": blocks_dxf_path,
                        "explode_blocks": False})
        assert "error" not in r
        assert "blocks_exploded" not in r
        assert "planlens>=0.2" not in r["note"]


class TestGroundSurfaceBlockProvenance:
    """candidate_ground_surface ranks by widest x-extent over WHATEVER is in
    the IR. With explosion on that includes stamped block line-work, and the
    entity ref carries no provenance to say so — so the reply must at least
    tell the caller how much block geometry the ranking competed against."""

    def test_reply_echoes_block_share_and_the_escape_hatch(
            self, blocks_dxf_path):
        r = call_agent("drawing_ir", "digitize_drawing",
                       {"file_path": blocks_dxf_path})
        q = call_agent("drawing_ir", "query_drawing",
                       {"handle": r["handle"],
                        "query": "candidate_ground_surface"})
        assert "error" not in q
        assert q["n_block_entities"] == r["n_block_entities"] == 2
        assert "explode_blocks=false" in q["note"]
        # The provenance gap this documents: the winning candidate is the
        # 120-wide stamped border, and its ref does not say 'block'.
        assert q["result"]["width"] == pytest.approx(120.0)

    def test_no_echo_when_the_ir_has_no_block_geometry(self, blocks_dxf_path):
        r = call_agent("drawing_ir", "digitize_drawing",
                       {"file_path": blocks_dxf_path,
                        "explode_blocks": False})
        q = call_agent("drawing_ir", "query_drawing",
                       {"handle": r["handle"],
                        "query": "candidate_ground_surface"})
        assert "n_block_entities" not in q
        assert "note" not in q

    def test_block_risk_is_in_the_tool_description(self):
        desc = METHOD_INFO["query_drawing"]["parameters"]["params"][
            "description"]
        assert "candidate_ground_surface" in desc
        assert "explode_blocks=false" in desc
        assert "n_block_entities" in METHOD_INFO["query_drawing"]["returns"]


# ---------------------------------------------------------------------------
# Phase 2: composition queries, snip_region, search_drawing_set
# ---------------------------------------------------------------------------

fitz = pytest.importorskip("fitz")

# planlens.testing, NOT planlens.ir.tests: the *.tests packages are excluded
# from the planlens wheel, so importing fixtures from there works in a source
# checkout and dies with ModuleNotFoundError on a released install.
from planlens.testing.construct_fixtures import (  # noqa: E402
    build_synthetic_dimension_pdf, build_synthetic_drawing_set_pdf,
)
from planlens.testing.leader_fixtures import (  # noqa: E402
    build_synthetic_leader_pdf,
)


class TestFixtureImportsSurviveARelease:
    """planlens ships no ``*.tests`` package (its pyproject excludes them), so
    an app test importing ``planlens.ir.tests.*`` passes here and dies on a
    pip-installed planlens — measured: all 45 tests in these two modules were
    lost to a collection-time ModuleNotFoundError under a simulated wheel."""

    def test_no_module_imports_the_excluded_package(self):
        import pathlib
        import re
        # Import statements only — the surrounding comments name the old path
        # on purpose, to explain why nothing may import it.
        bad = re.compile(r"^\s*(?:from|import)\s+planlens\.ir\.tests\b")
        here = pathlib.Path(__file__).parent
        for name in ("test_drawing_ir_adapter.py", "test_render_region.py"):
            src = (here / name).read_text(encoding="utf-8").splitlines()
            offenders = [ln for ln in src if bad.match(ln)]
            assert not offenders, (
                f"{name} imports fixtures from a package the planlens wheel "
                f"excludes; use planlens.testing.*: {offenders}")

    def test_fixture_home_is_a_shipped_package(self):
        import planlens.testing.construct_fixtures as cf
        import planlens.testing.leader_fixtures as lf
        for mod in (lf, cf):
            assert "tests" not in mod.__name__.split(".")


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
        from planlens.testing.construct_fixtures import (
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
