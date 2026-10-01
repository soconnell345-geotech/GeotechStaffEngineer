"""The orientation turn with and without ``GEOTECH_REVIEW_OVERVIEW`` (S1.2).

Switched off, the request is exactly the profile's own text (the app and the
review suite send the same). Switched on, the PDF pages of the new uploads
are counted: above ``GEOTECH_REVIEW_OVERVIEW_PAGES`` (default 20) the
request asks for the contact sheets by name, at or below it to skip them.
"""

import os

import pytest

fitz = pytest.importorskip("fitz")
pytest.importorskip("planlens.testing")

from planlens.testing import build_synthetic_submittal  # noqa: E402

from funhouse_agent import review_flags  # noqa: E402
from webapp import core, profiles  # noqa: E402

OVERVIEW = review_flags.OVERVIEW_ENV


@pytest.fixture(autouse=True)
def _no_switches(monkeypatch):
    for env in review_flags.ALL_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.delenv(review_flags.OVERVIEW_PAGES_ENV, raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")


@pytest.fixture(scope="module")
def submittal():
    return build_synthetic_submittal().pdf              # 11 pages


def _long(pdf: bytes, copies: int = 2) -> bytes:
    """``copies`` of ``pdf`` bound into one (22 pages from the submittal)."""
    out = fitz.open()
    for _ in range(copies):
        src = fitz.open(stream=pdf, filetype="pdf")
        out.insert_pdf(src)
        src.close()
    data = out.tobytes()
    out.close()
    return data


def _stage(tmp_path, files):
    return core.stage_uploads({}, str(tmp_path), files)


def test_switch_off_is_exactly_today_s_request(tmp_path, submittal):
    atts = _stage(tmp_path, [("big.pdf", _long(submittal)),
                             ("photo.png", b"\x89PNG not really")])
    text = profiles.orientation_request_for(profiles.DOCUMENT_REVIEW, atts)
    assert text == profiles.ORIENTATION_REQUEST.format(
        names="`big.pdf`, `photo.png`")
    assert "(and the contact sheets if it is long)" in text
    # the two new variants close with today's own words
    assert profiles._ORIENTATION_ASK in profiles.ORIENTATION_REQUEST
    for variant in (profiles.ORIENTATION_REQUEST_LONG,
                    profiles.ORIENTATION_REQUEST_SHORT):
        assert variant.endswith(profiles._ORIENTATION_ASK)


def test_a_short_upload_skips_the_contact_sheets(tmp_path, submittal,
                                                 monkeypatch):
    monkeypatch.setenv(OVERVIEW, "1")
    atts = _stage(tmp_path, [("sub.pdf", submittal)])
    text = profiles.orientation_request_for(profiles.DOCUMENT_REVIEW, atts)
    assert text == profiles.ORIENTATION_REQUEST_SHORT.format(names="`sub.pdf`")
    assert "skip the contact sheets" in text
    assert "render_page_thumbnails" not in text
    assert "Do not review it yet." in text


def test_a_long_upload_asks_for_the_contact_sheets(tmp_path, submittal,
                                                   monkeypatch):
    monkeypatch.setenv(OVERVIEW, "1")
    atts = _stage(tmp_path, [("big.pdf", _long(submittal))])
    text = profiles.orientation_request_for(profiles.DOCUMENT_REVIEW, atts)
    assert "render_page_thumbnails" in text and "(22 pages in all)" in text
    assert "look at them before you summarise" in text
    assert "`big.pdf`" in text


def test_pages_are_totalled_across_the_new_uploads(tmp_path, submittal,
                                                   monkeypatch):
    monkeypatch.setenv(OVERVIEW, "1")
    atts = _stage(tmp_path, [("a.pdf", submittal), ("b.pdf", submittal),
                             ("c.png", b"\x89PNG")])
    text = profiles.orientation_request_for(profiles.DOCUMENT_REVIEW, atts)
    assert "(22 pages in all)" in text


def test_the_threshold_is_a_setting_and_at_it_is_short(tmp_path, submittal,
                                                       monkeypatch):
    monkeypatch.setenv(OVERVIEW, "1")
    atts = _stage(tmp_path, [("sub.pdf", submittal)])
    monkeypatch.setenv(review_flags.OVERVIEW_PAGES_ENV, "11")      # at it
    assert "skip the contact sheets" in profiles.orientation_request_for(
        profiles.DOCUMENT_REVIEW, atts)
    monkeypatch.setenv(review_flags.OVERVIEW_PAGES_ENV, "10")      # above
    assert "render_page_thumbnails" in profiles.orientation_request_for(
        profiles.DOCUMENT_REVIEW, atts)


def test_page_count_never_raises(tmp_path, submittal):
    pdf = tmp_path / "s.pdf"
    pdf.write_bytes(submittal)
    png = tmp_path / "p.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n")
    fake = tmp_path / "fake.pdf"
    fake.write_bytes(b"%PDF-1.4 fake")
    assert profiles.pdf_page_count(str(pdf)) == 11
    assert profiles.pdf_page_count(str(png)) == 0
    assert profiles.pdf_page_count(str(fake)) == 0
    assert profiles.pdf_page_count(str(tmp_path / "missing.pdf")) == 0
    assert profiles.pdf_page_count(None) == 0


# ---------------------------------------------------------------------------
# The app sends it (AppTest, engine and agent mocked)
# ---------------------------------------------------------------------------

def _harness(monkeypatch, tmp_path):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    import webapp.engine_config as engine_config
    from webapp.engine_config import EngineResolution

    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path / "data"))
    for e in ("DEV_IDENTITY", "GEOTECH_USER_EMAIL", "GEOTECH_DEPLOYMENT"):
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setenv("GEOTECH_UPLOAD_MODE", "http")
    monkeypatch.setenv(profiles.PROFILE_ENV, "document_review")
    monkeypatch.setattr(engine_config, "resolve_engine",
                        lambda *a, **k: EngineResolution(
                            model=object(), source="prompter",
                            model_name="fake-model", message=""))
    monkeypatch.setattr(core, "build_agent", lambda *a, **k: object())

    def stream(_agent, _messages, _thread_id, **_kw):
        yield {"kind": "turn_done", "answer": "Oriented.", "turn_tokens": 1}
    monkeypatch.setattr(core, "stream_turn", stream)
    app = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "app.py")
    return AppTest.from_file(app, default_timeout=30)


def _orient(at, name, data):
    atts = core.stage_uploads(at.session_state["attachments"],
                              at.session_state["temp_dir"], [(name, data)])
    at.session_state["pending_orientation"] = [a.key for a in atts]
    at.run()
    assert not at.exception
    return at.session_state["transcript"][0]["text"]


def test_the_app_sends_the_overview_request(monkeypatch, tmp_path, submittal):
    monkeypatch.setenv(OVERVIEW, "1")
    at = _harness(monkeypatch, tmp_path).run()
    assert not at.exception
    text = _orient(at, "plans.pdf", _long(submittal))
    assert text == profiles.ORIENTATION_REQUEST_LONG.format(
        names="`plans.pdf`", pages=22)


def test_the_app_sends_today_s_request_with_the_switch_off(monkeypatch,
                                                           tmp_path,
                                                           submittal):
    at = _harness(monkeypatch, tmp_path).run()
    assert not at.exception
    text = _orient(at, "plans.pdf", _long(submittal))
    assert text == profiles.ORIENTATION_REQUEST.format(names="`plans.pdf`")


# ---------------------------------------------------------------------------
# The review suite sends the same text
# ---------------------------------------------------------------------------

def _answering_model():
    from langchain_core.language_models.fake_chat_models import (
        FakeMessagesListChatModel)
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kw):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            return ChatResult(generations=[ChatGeneration(
                message=AIMessage(content="An 11-sheet submittal."))])

    return Model(responses=[AIMessage(content="x")])


@pytest.mark.parametrize("arm", ["baseline", "overview"])
def test_the_suite_s_orientation_turn_is_the_app_s(arm, tmp_path, submittal):
    pytest.importorskip("planlens.tools")
    from funhouse_agent.review_eval.runner import run_task
    from funhouse_agent.review_eval.tasks import Task
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "long.pdf").write_bytes(_long(submittal))
    task = Task(id="orient", question="What is it?", documents=["long.pdf"],
                category="summarize", doc_type="submittal",
                checks=[{"type": "contains_all", "terms": ["submittal"]}])
    res = run_task(task, _answering_model(), arm=arm,
                   arm_env=review_flags.ARMS[arm], docs_dir=str(docs),
                   run_dir=str(tmp_path / "run"), orientation=True)
    assert res["error"] is None, res.get("traceback")
    first = res["turns"][0]["question"]
    with review_flags.switches(review_flags.ARMS[arm]):
        expected = profiles.orientation_request_for(
            profiles.DOCUMENT_REVIEW,
            [core.Attachment(key="long.pdf", path=str(docs / "long.pdf"),
                             size=0)])
    assert first == expected
    assert ("render_page_thumbnails" in first) == (arm == "overview")
    assert res["turns"][1]["question"] == "What is it?"


# ---------------------------------------------------------------------------
# The conversation's folder is bound when the page's agent is built
# ---------------------------------------------------------------------------

def _captured_build(monkeypatch, tmp_path, **build_kwargs):
    import funhouse_agent.deep.agent as deep_agent
    got = {}

    def fake(model, **kw):
        got.update(kw)
        return object()

    monkeypatch.setattr(deep_agent, "build_deep_agent", fake)
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path / "data"))
    files = os.path.join(core.conversation_dir("t_wd"), "files")
    os.makedirs(files, exist_ok=True)
    core.build_agent(object(), {}, files, [], **build_kwargs)
    return got, files


def test_the_review_page_build_carries_its_conversation_folder(monkeypatch,
                                                               tmp_path):
    """Review fix 6: another tab re-pointing GEOTECH_DEFAULT_OUTPUT_DIR must
    not redirect this conversation's findings or digests, so the review
    page's build is handed its folder; the geotech page's build is not."""
    got, files = _captured_build(monkeypatch, tmp_path,
                                 **profiles.DOCUMENT_REVIEW.build_kwargs())
    assert got["working_dir"] == files and got["review_page"] is True
    got, _files = _captured_build(monkeypatch, tmp_path)
    assert "working_dir" not in got


def test_the_page_agent_s_ledger_follows_its_own_conversation(monkeypatch,
                                                              tmp_path):
    """End to end through core.build_agent: two conversations' lean agents
    keep their findings apart while the process-wide folder points at a
    third."""
    pytest.importorskip("planlens.tools")
    from funhouse_agent.review_findings import FindingsLedger
    monkeypatch.setenv(review_flags.AGENT_ENV, "lean")
    monkeypatch.setenv(review_flags.FINDINGS_ENV, "1")
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path / "data"))
    agents = {}
    for tid in ("conv_a", "conv_b"):
        files = os.path.join(core.conversation_dir(tid), "files")
        os.makedirs(files, exist_ok=True)
        agents[tid] = (core.build_agent(
            _answering_model(), {}, files, [],
            **profiles.DOCUMENT_REVIEW.build_kwargs()), files)
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path / "other"))
    for tid, (agent, _files) in agents.items():
        tool = agent.nodes["tools"].bound.tools_by_name["record_finding"]
        tool.invoke({"statement": f"Found in {tid}.", "severity": "info",
                     "confidence": "high", "evidence": "read",
                     "citations": [{"document": "x.pdf", "page": 0}]})
    for tid, (_agent, files) in agents.items():
        saved = FindingsLedger(os.path.join(files, "findings.json")).load()
        assert [f.statement for f in saved] == [f"Found in {tid}."]
    assert not (tmp_path / "other").exists()


# -- a screenshot mid-conversation is evidence, not a document (2026-10-01) --

def test_a_first_upload_is_oriented_whatever_it_is():
    from webapp.profiles import orient_on_upload
    attach = [{"role": "attach", "text": "x"}]
    assert orient_on_upload(["set.pdf"], [])
    assert orient_on_upload(["photo.png"], attach)     # nothing asked yet


def test_a_screenshot_mid_conversation_is_not_oriented():
    from webapp.profiles import orient_on_upload
    started = [{"role": "user", "text": "find every tag"},
               {"role": "assistant", "text": "found 3"},
               {"role": "attach", "text": "missed.png"}]
    assert not orient_on_upload(["missed.png"], started)
    assert not orient_on_upload(["a.PNG", "b.jpeg"], started)
    # a document dropped in later is still introduced
    assert orient_on_upload(["addendum.pdf"], started)
    assert orient_on_upload(["shot.png", "addendum.pdf"], started)
    assert not orient_on_upload([], started)


def test_a_pasted_screenshot_gets_a_name_of_its_own():
    from webapp.core import pasted_upload_name
    when = 1790885085.0
    a = pasted_upload_name("image.png", 0, when)
    b = pasted_upload_name("image.png", 1, when)
    assert a.startswith("pasted_") and a.endswith(".png") and a != b
    assert pasted_upload_name("IMAGE.JPG", 0, when).endswith(".jpg")
    assert pasted_upload_name("sheet_31.png") == "sheet_31.png"   # kept
    assert pasted_upload_name("set.pdf") == "set.pdf"
