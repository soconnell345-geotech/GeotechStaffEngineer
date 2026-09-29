"""Review milestone M1, S1.1: drawing geometry on the review page - offline.

Pins the ``GEOTECH_REVIEW_GEOMETRY`` switch on both sides (off: the page is
what it was; on: the lean agent and its reading helper carry four read-only
geometry tools), and what the tools return on planlens' synthetic sheets.

THE FRAME is the point of most of these tests. planlens' fixtures state their
truth in the IR's bottom-left frame of an UNROTATED page; the tools must
answer in the DISPLAYED frame (top-left origin, y down) of the page as a
viewer shows it. :func:`displayed` converts the truth BY HAND from the
/Rotate definition - never through the code under test or PyMuPDF's
rotation matrix - and each fixture is re-served at /Rotate 0, 90, 180 and
270 by setting the page's rotation, which moves every displayed coordinate
while the drawing itself stays put.
"""

import json
import math
import os

import pytest

pytest.importorskip("planlens.tools")
pytest.importorskip("planlens.ir")
fitz = pytest.importorskip("fitz")

from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    FakeMessagesListChatModel)
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402

from funhouse_agent import review_flags  # noqa: E402
from funhouse_agent.deep import geometry_tools as G  # noqa: E402
from funhouse_agent.deep.agent import build_deep_agent  # noqa: E402
from planlens.testing import (  # noqa: E402
    build_synthetic_cloud_pdf, build_synthetic_dimension_pdf,
    build_synthetic_leader_pdf, build_synthetic_review_document,
    build_synthetic_title_block_pdf)

GEOMETRY = set(G.GEOMETRY_TOOLS)
GEOTECH_WORDS = ("drawing_ir", "digitize_drawing", "query_drawing",
                 "snip_region", "call_agent", "figure_db", "geotech")
#: The synthetic construct sheets: 900 x 700 pt, unrotated.
W_UN, H_UN = 900.0, 700.0
BUDGET = 12000
MECK = os.path.join(os.path.dirname(__file__), "..", "..", "..", "module_work",
                    "drawing_ground_truth", "mecklenburg")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for env in review_flags.ALL_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")
    monkeypatch.delenv("GEOTECH_REVIEW_MAX_MODEL_CALLS", raising=False)
    G.clear_cache()
    yield
    G.clear_cache()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def displayed(pt_ir, rotation, w=W_UN, h=H_UN):
    """A point of the fixture's IR (bottom-left, y up, unrotated page w x h)
    -> the displayed frame once the page carries ``/Rotate rotation``.

    Unrotated y-down is (x, h - y). The /Rotate definitions (PDF 1.7 8.3.2,
    as planlens.testing.scale_fixtures writes them out): 90 -> (h - yu, xu),
    180 -> (w - xu, h - yu), 270 -> (yu, w - xu)."""
    xu, yu = float(pt_ir[0]), h - float(pt_ir[1])
    return {0: (xu, yu), 90: (h - yu, xu), 180: (w - xu, h - yu),
            270: (yu, w - xu)}[rotation]


def displayed_box(box_ir, rotation, w=W_UN, h=H_UN):
    x0, y0, x1, y1 = box_ir
    pts = [displayed(p, rotation, w, h)
           for p in ((x0, y0), (x1, y1), (x0, y1), (x1, y0))]
    return (min(p[0] for p in pts), min(p[1] for p in pts),
            max(p[0] for p in pts), max(p[1] for p in pts))


def rotated(path, rotation):
    doc = fitz.open(path)
    if rotation:
        doc[0].set_rotation(rotation)
    data = doc.tobytes()
    doc.close()
    return data


def call(name, attachments=None, **kw):
    tools = {t.name: t for t in G.make_geometry_tools(attachments or {})}
    out = tools[name].invoke(kw)
    assert len(out) < BUDGET, (name, len(out))
    return json.loads(out)


def near(p, q, tol):
    return math.dist(p, q) <= tol


def inside(p, box, pad=0.0):
    return (box[0] - pad <= p[0] <= box[2] + pad
            and box[1] - pad <= p[1] <= box[3] + pad)


def _review_kwargs():
    from webapp.profiles import DOCUMENT_REVIEW
    return DOCUMENT_REVIEW.build_kwargs()


class Scripted(FakeMessagesListChatModel):
    script: list = []
    seen: list = []

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kw):
        self.seen.append(list(messages))
        step = self.script.pop(0) if self.script else AIMessage(content="done")
        msg = step(messages) if callable(step) else step
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _model(script):
    return Scripted(responses=[AIMessage(content="x")], script=list(script),
                    seen=[])


def _tools_of(agent):
    return agent.nodes["tools"].bound.tools_by_name


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------

def test_geometry_tools_only_with_lean_and_the_switch(monkeypatch):
    def names():
        return set(_tools_of(build_deep_agent(_model([]),
                                              **_review_kwargs())))
    legacy_off = names()
    assert not legacy_off & GEOMETRY                      # everything off
    monkeypatch.setenv(review_flags.GEOMETRY_ENV, "1")
    assert names() == legacy_off                          # legacy: unchanged
    monkeypatch.setenv(review_flags.AGENT_ENV, "lean")
    monkeypatch.delenv(review_flags.GEOMETRY_ENV)
    lean_off = names()
    assert not lean_off & GEOMETRY                        # lean alone: no
    monkeypatch.setenv(review_flags.GEOMETRY_ENV, "1")
    agent = build_deep_agent(_model([]), **_review_kwargs())
    tools = _tools_of(agent)
    assert set(tools) == lean_off | GEOMETRY              # exactly the four
    from langchain_core.utils.function_calling import convert_to_openai_tool
    for name in GEOMETRY:
        desc = tools[name].description or ""
        for word in GEOTECH_WORDS:
            assert word not in desc.lower(), (name, word)
        assert "render_region" in desc or name == "revision_clouds"
        schema = json.dumps(convert_to_openai_tool(tools[name]))
        assert "$ref" not in schema and "$defs" not in schema
        props = convert_to_openai_tool(tools[name])["function"]["parameters"]
        assert {"source", "page", "bbox"} <= set(props["properties"])


def test_the_reading_helper_gets_them_only_with_the_switch(monkeypatch):
    from funhouse_agent.deep import review_agent
    seen = []
    real = review_agent.make_reader_tool

    def spy(model, reader_tools, *a, **kw):
        seen.append({t.name for t in reader_tools})
        return real(model, reader_tools, *a, **kw)

    monkeypatch.setattr(review_agent, "make_reader_tool", spy)
    monkeypatch.setenv(review_flags.AGENT_ENV, "lean")
    build_deep_agent(_model([]), **_review_kwargs())
    monkeypatch.setenv(review_flags.GEOMETRY_ENV, "1")
    build_deep_agent(_model([]), **_review_kwargs())
    off, on = seen
    assert not off & GEOMETRY
    assert on == off | GEOMETRY
    assert "annotate_document" not in on                 # still read-only


def test_the_arm_and_the_switch_are_registered(monkeypatch):
    A = review_flags
    assert A.GEOMETRY_ENV == "GEOTECH_REVIEW_GEOMETRY"
    assert A.GEOMETRY_ENV in A.ALL_ENVS
    assert A.ARMS["geometry"] == {
        A.AGENT_ENV: "lean", A.VISION_TEXT_ENV: "1",
        A.VISION_STRUCTURED_ENV: "1", A.GEOMETRY_ENV: "1"}
    monkeypatch.setenv(A.GEOMETRY_ENV, "1")
    with A.switches(A.ARMS["lean"]):
        assert not A.geometry()                           # an arm clears it
    assert A.geometry()
    with A.switches(A.ARMS["geometry"]):
        assert A.geometry() and A.lean_agent() and A.vision_structured()
    assert "GEOTECH_REVIEW_GEOMETRY=1" in A.describe()


# ---------------------------------------------------------------------------
# The frame, on planlens' own /Rotate 90 review sheet
# ---------------------------------------------------------------------------

def test_ir_to_displayed_frame_on_a_rotate_90_sheet():
    gt = build_synthetic_review_document()
    sheet = G._sheet(gt.pdf, gt.sheet_page)
    assert sheet.rotation == 90
    assert (round(sheet.width), round(sheet.height)) == (2592, 1728)
    # The 250 short lines, planted at displayed (xd, yd)-(xd + 30, yd).
    planted = {(200.0 + (i % 50) * 40.0, 700.0 + (i // 50) * 60.0)
               for i in range(gt.n_sheet_lines_drawn)}
    # (drawn with PyMuPDF's default closePath, each ingests as a 2-vertex
    # polyline rather than a Line - either is a straight piece of line-work)
    lines = [e.points() for e in sheet.ir.entities
             if getattr(e, "KIND", "") in ("line", "polyline")
             and len(e.points()) == 2
             and abs(math.dist(*e.points()) - 30.0) < 0.5]
    assert len(lines) == gt.n_sheet_lines_drawn
    for p0, p1 in lines:
        a, b = sheet.point(*p0), sheet.point(*p1)
        left, right = sorted((a, b))
        assert abs(left[1] - right[1]) < 0.01                # horizontal
        assert abs(right[0] - left[0] - 30.0) < 0.01
        assert (round(left[0], 2), round(left[1], 2)) in planted
    # The ring drawn round the reviewer's target (displayed 1500, 900).
    rings = [e for e in sheet.ir.entities
             if getattr(e, "KIND", "") == "polyline" and e.bbox
             and (e.bbox[2] - e.bbox[0]) < 20 and len(e.vertices) > 8]
    assert rings
    cx, cy = [(v0 + v1) / 2 for v0, v1 in zip(sheet.box(rings[0].bbox)[:2],
                                               sheet.box(rings[0].bbox)[2:])]
    assert near((cx, cy), gt.reviewer_target, 0.5)
    # Upright text: its IR insertion point is its displayed baseline origin.
    t = next(e for e in sheet.ir.entities if getattr(e, "KIND", "") == "text"
             and e.content == gt.sheet_text_upright)
    assert near(sheet.point(*t.position), gt.sheet_text_upright_origin, 1.0)


# ---------------------------------------------------------------------------
# drawing_callouts
# ---------------------------------------------------------------------------

def _match_leaders(found, planted, rotation):
    """For each planted leader, the callout whose tip and tail both land on
    it in the displayed frame (or None)."""
    out = []
    for L in planted:
        tip, tail = displayed(L.tip_xy, rotation), displayed(L.tail_xy,
                                                            rotation)
        hit = next((c for c in found if near(c["tip"], tip, 1.5)
                    and near(c["tail"], tail, 1.5)), None)
        out.append((L, hit))
    return out


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_callouts_recall_labels_and_frame_at_every_rotation(tmp_path,
                                                            rotation):
    path, gt = build_synthetic_leader_pdf(tmp_path, n_leaders=6)
    atts = {"sheet.pdf": rotated(path, rotation)}
    res = call("drawing_callouts", atts, source="sheet.pdf", page=0)
    assert (res["page"], res["pdf_page"]) == (0, 1)
    assert res["count"] == len(res["callouts"]) == res["shown"]
    pw, ph = (H_UN, W_UN) if rotation in (90, 270) else (W_UN, H_UN)
    assert res["page_size"] == [pw, ph]
    matched = _match_leaders(res["callouts"], gt["leaders"], rotation)
    assert all(hit for _, hit in matched), [L.label for L, h in matched
                                            if not h]          # recall 6/6
    assert len(res["callouts"]) == len(gt["leaders"])          # precision
    for L, hit in matched:
        assert hit["label"] == L.text
        assert hit["confidence"] >= 0.5
        text_at = displayed(L.text_pos, rotation)
        assert inside(text_at, hit["label_zoom"])
        for key in ("tip", "tail"):
            assert inside(hit[key], hit["zoom"])
        for box in (hit["label_zoom"], hit["zoom"]):
            assert 0 <= box[0] < box[2] <= pw and 0 <= box[1] < box[3] <= ph
    assert "PROPOSALS" in res["note"] and "render_region" in res["frame"]


def test_callouts_rank_every_decoy_below_every_planted_leader(tmp_path):
    path, gt = build_synthetic_leader_pdf(tmp_path, n_leaders=6)
    res = call("drawing_callouts", source=path, page=0, min_confidence=0.0)
    matched = {id(h) for _, h in _match_leaders(res["callouts"],
                                                gt["leaders"], 0) if h}
    assert len(matched) == 6
    true_conf = [c["confidence"] for c in res["callouts"] if id(c) in matched]
    decoys = [c["confidence"] for c in res["callouts"] if id(c) not in matched]
    assert not decoys or max(decoys) < min(true_conf)
    # the dimension decoy's arrowheads never surface as callouts at all
    for c in res["callouts"]:
        assert not (abs(c["tip"][1] - 90.0) < 2 and 290 < c["tip"][0] < 510)


def test_callouts_bbox_limits_the_region(tmp_path):
    path, gt = build_synthetic_leader_pdf(tmp_path, n_leaders=3)
    L = gt["leaders"][0]
    tip = displayed(L.tip_xy, 0)
    box = [tip[0] - 10, tip[1] - 10, tip[0] + 10, tip[1] + 10]
    res = call("drawing_callouts", source=path, page=0, bbox=box)
    assert res["count"] == 1 and res["callouts"][0]["label"] == L.text


def test_a_busy_sheet_stays_in_budget_and_says_what_was_left_out(tmp_path,
                                                                 monkeypatch):
    path, _gt = build_synthetic_leader_pdf(tmp_path, n_leaders=60)
    res = call("drawing_callouts", source=path, page=0, min_confidence=0.0)
    assert res["shown"] <= G.MAX_ITEMS
    assert res["left_out"] == res["count"] - res["shown"] > 0
    confs = [c["confidence"] for c in res["callouts"]]
    assert confs == sorted(confs, reverse=True)           # strongest first
    monkeypatch.setattr(G, "RESULT_CHARS", 2000)
    small = json.loads(G.make_geometry_tools({})[0].invoke(
        {"source": path, "page": 0, "min_confidence": 0.0}))
    assert 0 < small["shown"] < res["shown"]
    assert len(json.dumps(small)) <= 2000 + 50


# ---------------------------------------------------------------------------
# drawing_dimensions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rotation", [0, 90, 270])
def test_dimensions_ends_length_and_text_at_every_rotation(tmp_path,
                                                           rotation):
    path, gt = build_synthetic_dimension_pdf(tmp_path)
    atts = {"dims.pdf": rotated(path, rotation)}
    res = call("drawing_dimensions", atts, source="dims.pdf", page=0)
    assert res["count"] == len(gt["dimensions"]) == 2      # recall, precision
    for d in gt["dimensions"]:
        a, b = displayed(d["end_a"], rotation), displayed(d["end_b"], rotation)
        hit = next(x for x in res["dimensions"]
                   if (near(x["end_a"], a, 2) and near(x["end_b"], b, 2))
                   or (near(x["end_a"], b, 2) and near(x["end_b"], a, 2)))
        assert abs(hit["length_pt"] - math.dist(a, b)) < 2
        assert hit["text"] == d["value"]
        assert inside(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2),
                      hit["value_zoom"])
        assert "real_length" not in hit                   # no scale stored
    orient = sorted(x["orientation"] for x in res["dimensions"])
    assert orient == ["horizontal", "vertical"]
    assert "no calibrated scale" in res["scale"]


def test_the_leader_decoy_is_not_a_dimension(tmp_path):
    path, gt = build_synthetic_dimension_pdf(tmp_path)
    res = call("drawing_dimensions", source=path, page=0, min_confidence=0.0)
    planted = [(displayed(d["end_a"], 0), displayed(d["end_b"], 0))
               for d in gt["dimensions"]]

    def is_planted(x):
        return any((near(x["end_a"], a, 2) and near(x["end_b"], b, 2))
                   or (near(x["end_a"], b, 2) and near(x["end_b"], a, 2))
                   for a, b in planted)

    true = [x for x in res["dimensions"] if is_planted(x)]
    others = [x for x in res["dimensions"] if not is_planted(x)]
    assert len(true) == 2
    assert not others or max(o["confidence"] for o in others) < \
        min(t["confidence"] for t in true)
    # nothing near the decoy leader (tail 100,500 / tip 250,520, page frame)
    for x in res["dimensions"]:
        if x["confidence"] >= 0.5:
            assert not inside(x["end_a"], (90, 490, 260, 550))


_VP = ("[ << /Type /Viewport /BBox [ 0 0 900 700 ] /Measure << /Type /Measure "
       "/Subtype /RL /R ({ratio}) /X [ << /Type /NumberFormat /U ({unit}) "
       "/C {c} /D 100 >> ] /Y [ << /Type /NumberFormat /U ({unit}) /C {c} "
       "/D 100 >> ] /D [ << /Type /NumberFormat /U ({unit}) /C 1 /D 100 >> ] "
       "/A [ << /Type /NumberFormat /U (sq {unit}) /C 1 /D 100 >> ] >> >> ]")


def _with_viewport(path, ratio, unit, c):
    doc = fitz.open(path)
    doc.xref_set_key(doc[0].xref, "VP", _VP.format(ratio=ratio, unit=unit,
                                                   c=c))
    data = doc.tobytes()
    doc.close()
    return data


def test_real_length_only_through_a_scale_the_pdf_stores(tmp_path):
    path, gt = build_synthetic_dimension_pdf(tmp_path)
    # 1 in = 20 ft stored as /C = 20/72 ft per point: 250 pt -> 69.44 ft.
    scaled = _with_viewport(path, "1 in = 20 ft", "ft", "0.2777777778")
    res = call("drawing_dimensions", {"s.pdf": scaled}, source="s.pdf",
               page=0)
    assert res["count"] == 2
    for x in res["dimensions"]:
        assert x["real_length"] == f"{250.0 * 20.0 / 72.0:.2f} ft"
        assert "20 ft" in x["scale"]
    assert "stored in the PDF" in res["scale"]
    # The 1:1 default a producer writes when nobody set a scale is no scale.
    G.clear_cache()
    identity = _with_viewport(path, " ", " ", ".01389")
    res = call("drawing_dimensions", {"s.pdf": identity}, source="s.pdf",
               page=0)
    assert all("real_length" not in x for x in res["dimensions"])


# ---------------------------------------------------------------------------
# title_block
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rotation", [0, 90])
def test_title_block_box_and_fields(tmp_path, rotation):
    path, gt = build_synthetic_title_block_pdf(tmp_path)
    atts = {"tb.pdf": rotated(path, rotation)}
    res = call("title_block", atts, source="tb.pdf", page=0)
    tb = res["title_block"]
    want = displayed_box(gt["block_bbox_ir"], rotation)
    assert all(abs(u - v) <= 2 for u, v in zip(tb["box"], want)), \
        (tb["box"], want)
    assert tb["confidence"] >= 0.5
    assert tb["fields"]["sheet"] == "S-101"
    assert tb["fields"]["revision"] == "2"
    assert "20'" in tb["fields"]["scale"]
    assert "RETAINING WALL PLAN" in tb["lines"]
    assert inside(want[:2], tb["zoom"]) and inside(want[2:], tb["zoom"])
    # the mid-sheet schedule box is never the answer
    sched = displayed_box((300, H_UN - 470, 520, H_UN - 380), rotation)
    assert tb["box"] != list(sched)


def test_title_block_fields_from_label_and_value_on_separate_lines():
    lines = [("STD. NO.", (705.0, 547.0, 742.0, 558.0)),
             ("REV.", (752.0, 547.0, 772.0, 558.0)),
             ("10.31A", (700.0, 559.0, 750.0, 574.0)),
             ("DATE", (600.0, 547.0, 630.0, 558.0)),
             ("5/09/2023", (600.0, 559.0, 650.0, 570.0)),
             ("NOT TO SCALE", (688.0, 501.0, 772.0, 514.0))]
    f = G._fields(lines)
    assert f["sheet"] == "10.31A"
    assert "revision" not in f                  # REV. has nothing under it
    assert f["date"] == "5/09/2023"
    assert f["scale"] == "not to scale"
    assert "revision" not in G._fields([("REVISED PVMT. SECT.",
                                         (0, 0, 90, 10))])


# ---------------------------------------------------------------------------
# revision_clouds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("rotation", [0, 90])
def test_clouds_box_tag_and_decoys(tmp_path, rotation):
    path, gt = build_synthetic_cloud_pdf(tmp_path)
    atts = {"c.pdf": rotated(path, rotation)}
    res = call("revision_clouds", atts, source="c.pdf", page=0,
               min_confidence=0.0)
    assert res["clouds"], res
    want = displayed_box(gt["cloud_bbox_ir"], rotation)
    top = res["clouds"][0]
    # the scallops bulge out past the rectangle they were drawn round
    assert all(abs(u - v) <= 15 for u, v in zip(top["box"], want))
    assert len([c for c in res["clouds"] if c["confidence"] >= 0.3]) == 1
    tags = [top.get("tag")] if top.get("tag") else res.get("revision_tags")
    assert tags and tags[0]["text"] == gt["delta_text"]
    assert near(tags[0]["at"], displayed(gt["delta_center_ir"], rotation), 2)


# ---------------------------------------------------------------------------
# Cache, errors, a scan
# ---------------------------------------------------------------------------

def test_a_page_is_ingested_once_per_document_and_page(tmp_path, monkeypatch):
    path, _gt = build_synthetic_leader_pdf(tmp_path, n_leaders=3)
    calls = []
    real = G._ingest

    def counting(src, page):
        calls.append(page)
        return real(src, page)

    monkeypatch.setattr(G, "_ingest", counting)
    data = open(path, "rb").read()
    call("drawing_callouts", {"a.pdf": data}, source="a.pdf", page=0)
    call("drawing_dimensions", {"a.pdf": data}, source="a.pdf", page=0)
    call("revision_clouds", {"b.pdf": data}, source="b.pdf", page=0)
    call("title_block", source=path, page=0)     # same bytes, as a path
    assert calls == [0]


def test_errors_are_results_not_exceptions(tmp_path):
    path, _gt = build_synthetic_leader_pdf(tmp_path, n_leaders=1)
    res = call("drawing_callouts", {"a.pdf": b"%PDF"}, source="nope.pdf")
    assert "error" in res and "a.pdf" in json.dumps(res)
    res = call("drawing_dimensions", source=path, page=5)
    assert "out of range" in res["error"] and "1 pages" in res["error"]
    res = call("title_block", {"x.png": b"\x89PNG\r\n\x1a\n...."},
               source="x.png")
    assert "not a PDF" in res["error"]
    res = call("revision_clouds", source=path, page=0, bbox=[1, 2, 3])
    assert "bbox" in res["error"]


def test_a_scanned_page_has_no_geometry_and_says_look():
    gt = build_synthetic_review_document()
    res = call("drawing_callouts", {"r.pdf": gt.pdf}, source="r.pdf",
               page=gt.scanned_page)
    assert res["count"] == 0 and "analyze_pdf_page" in res["note"]
    res = call("title_block", {"r.pdf": gt.pdf}, source="r.pdf",
               page=gt.scanned_page)
    assert res["title_block"] is None


def test_the_lean_agent_calls_a_geometry_tool_end_to_end(tmp_path,
                                                         monkeypatch):
    path, gt = build_synthetic_dimension_pdf(tmp_path)
    monkeypatch.setenv(review_flags.AGENT_ENV, "lean")
    monkeypatch.setenv(review_flags.GEOMETRY_ENV, "1")
    model = _model([
        AIMessage(content="", tool_calls=[{
            "name": "drawing_dimensions", "id": "c1",
            "args": {"source": "d.pdf", "page": 0}}]),
        AIMessage(content="Two dimensions: 8'-0\" and 3.5 m.")])
    kw = _review_kwargs()
    kw["attachments"] = {"d.pdf": open(path, "rb").read()}
    agent = build_deep_agent(model, **kw)
    out = agent.invoke({"messages": [{"role": "user",
                                      "content": "what is dimensioned?"}]})
    msgs = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    body = json.loads(msgs[0].content)
    assert body["count"] == 2 and body["pdf_page"] == 1


# ---------------------------------------------------------------------------
# The suite tasks that need geometry
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("task_id,good,bad", [
    ("meck-trap-dimensions",
     ["Minimum dimensions: L=10' MIN., W=5' MIN., X=7' MIN., 1.5' MIN. and "
      "21\" MIN.",
      "| L | 10' | min |\n| W | 5 ft | minimum |\n| X | 7 ft | min |\nalso "
      "1.5 ft minimum and at least 21 inches",
      "Length L = 10 ft, width W = 5 ft, X = 7 ft (all minimums); the other "
      "minimums are 1.5' min and 21\" min."],
     ["L=12' MIN., W=6' MIN., X=8' MIN., 1.5' MIN., 21\" MIN.",
      "L=10' MIN., 5' MAX FILL, X=7' MIN., 1.5' MIN., 21\" MIN.",
      "L=10' MIN., W=5' MIN., X=7' MIN.; 12\" MIN of stone"]),
    ("meck-bioretention-section-dims",
     ["Section A-A: 10' MIN. across the top and 4' MIN. lower down.",
      "It shows a minimum of 10 feet and a 4 ft minimum."],
     ["A 10-foot wide maintenance easement and 2'-0\" to 4'-0\" of filter "
      "media.",
      "Section A-A: 12' MIN. and 4' MIN."]),
    ("meck-ramp-detail-callouts",
     ["The callouts: 3/4\" flowline depth at the ramp location, edge of "
      "pavement elevation, and 'TYP. MATCH RAMP SLOPE'.",
      "A ¾-inch flowline depth, the edge of pavement elevation, and it "
      "matches the ramp slope."],
     ["34\" FLOWLINE DEPTH AT RAMP LOCATION; EDGE OF PAVEMENT ELEVATION; "
      "TYP. MATCH RAMP SLOPE",
      "1/2\" flowline depth, edge of pavement, match ramp slope"]),
])
def test_the_geometry_tasks_take_right_answers_not_wrong_ones(task_id, good,
                                                              bad):
    from funhouse_agent.review_eval import checks as C
    from funhouse_agent.review_eval.tasks import OPEN_TASKS
    task = next(t for t in OPEN_TASKS if t.id == task_id)

    def passes(answer):
        return all(C.run_check(c, answer)["passed"]
                   for c in task.all_checks())

    assert passes(task.truth)
    for answer in good:
        assert passes(answer), answer
    for answer in bad:
        assert not passes(answer), answer


# ---------------------------------------------------------------------------
# The public stroke-lettered sheets (skipped where the corpus is absent)
# ---------------------------------------------------------------------------

def _meck(name):
    p = os.path.abspath(os.path.join(MECK, name))
    if not os.path.isfile(p):
        pytest.skip(f"{name} not in this checkout")
    return p


def test_sediment_trap_dimensions_are_all_found_with_zoom_boxes():
    """30.01 has no text layer; its DWG holds 10 DIMENSION entities."""
    res = call("drawing_dimensions", source=_meck("3001.pdf"), page=0)
    assert res["text_layer"] is False
    assert res["count"] == 10
    assert all(d["text"] is None for d in res["dimensions"])
    assert "zooming" in res["text_note"]
    for d in res["dimensions"]:
        mid = ((d["end_a"][0] + d["end_b"][0]) / 2,
               (d["end_a"][1] + d["end_b"][1]) / 2)
        assert inside(mid, d["value_zoom"])


def test_letter_strokes_and_border_lines_are_not_called_callouts():
    """30.01's DWG holds 7 LEADER entities, 25-50 pt long; before the caps
    the finder's output also carried the sheet border (500-720 pt) and 4-9 pt
    strokes of drawn letters above the call threshold."""
    path = _meck("3001.pdf")
    res = call("drawing_callouts", source=path, page=0)
    short = min(res["page_size"])
    for c in res["callouts"]:
        span = math.dist(c["tip"], c["tail"])
        assert G.LONG_LEADER * short >= span >= 4.0, c
    assert 6 <= res["count"] <= 12
    everything = call("drawing_callouts", source=path, page=0,
                      min_confidence=0.0)
    confs = [c["confidence"] for c in everything["callouts"]]
    assert confs == sorted(confs, reverse=True)


def test_rotated_ramp_sheet_callouts_carry_the_cad_text_labels():
    """10.31A is /Rotate 270 and carries AutoCAD's hidden SHX text; its DWG
    multileaders are labelled as below."""
    res = call("drawing_callouts", source=_meck("10.31A.pdf"), page=0)
    labels = [c["label"] or "" for c in res["callouts"]]
    assert sum("FLOWLINE DEPTH AT RAMP LOCATION" in t for t in labels) >= 2
    assert "EDGE OF PAVEMENT ELEVATION" in labels
    assert "TYP. MATCH RAMP SLOPE" in labels
    for c in res["callouts"]:            # every point on the displayed page
        assert inside(c["tip"], (0, 0, 792, 612))
        assert inside(c["tail"], (0, 0, 792, 612))
    tb = call("title_block", source=_meck("10.31A.pdf"), page=0)
    assert tb["title_block"]["fields"]["sheet"] == "10.31A"
    assert tb["title_block"]["box"][1] > 450          # along the bottom
