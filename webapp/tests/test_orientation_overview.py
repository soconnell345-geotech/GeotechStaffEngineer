"""The orientation turn: the request is exactly the profile's own text, and
the app and the review suite send the same.

(The ``GEOTECH_REVIEW_OVERVIEW`` variants that counted the uploads' pages
were removed in BACKLOG DR-39: the suite measured no gain.)
"""

import os

import pytest

pymupdf = pytest.importorskip("pymupdf")
pytest.importorskip("planlens.testing")

from planlens.testing import build_synthetic_submittal  # noqa: E402

from funhouse_agent import review_flags  # noqa: E402
from webapp import core, profiles  # noqa: E402



@pytest.fixture(autouse=True)
def _no_switches(monkeypatch):
    for env in review_flags.ALL_ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("GEOTECH_VISION_PROBE", "0")


@pytest.fixture(scope="module")
def submittal():
    return build_synthetic_submittal().pdf              # 11 pages


def _long(pdf: bytes, copies: int = 2) -> bytes:
    """``copies`` of ``pdf`` bound into one (22 pages from the submittal)."""
    out = pymupdf.open()
    for _ in range(copies):
        src = pymupdf.open(stream=pdf, filetype="pdf")
        out.insert_pdf(src)
        src.close()
    data = out.tobytes()
    out.close()
    return data


def _stage(tmp_path, files):
    return core.stage_uploads({}, str(tmp_path), files)


def test_the_request_is_exactly_the_profile_s_own(tmp_path, submittal):
    atts = _stage(tmp_path, [("big.pdf", _long(submittal)),
                             ("photo.png", b"\x89PNG not really")])
    text = profiles.orientation_request_for(profiles.DOCUMENT_REVIEW, atts)
    assert text == profiles.ORIENTATION_REQUEST.format(
        names="`big.pdf`, `photo.png`")
    assert "(and the contact sheets if it is long)" in text


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


def test_the_app_sends_the_profile_s_request(monkeypatch,
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


@pytest.mark.parametrize("arm", ["baseline"])
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
    assert "render_page_thumbnails" not in first
    assert res["turns"][1]["question"] == "What is it?"


# ---------------------------------------------------------------------------
# The conversation's folder is bound when the page's agent is built
# ---------------------------------------------------------------------------

def _captured_build(monkeypatch, tmp_path, **build_kwargs):
    import funhouse_agent.deep.agent as deep_agent
    import funhouse_agent.review_build as review_build
    got = {}

    def fake(model, **kw):
        got.update(kw)
        return object()

    # the geotech page builds with build_deep_agent, the review page with
    # its own build (INF-31 step 8)
    monkeypatch.setattr(deep_agent, "build_deep_agent", fake)
    monkeypatch.setattr(review_build, "build_review_page_agent", fake)
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path / "data"))
    files = os.path.join(core.conversation_dir("t_wd"), "files")
    os.makedirs(files, exist_ok=True)
    core.build_agent(object(), {}, files, [], **build_kwargs)
    return got, files


def test_the_review_page_build_carries_its_conversation_folder(monkeypatch,
                                                               tmp_path):
    """Review fix 6: the review page's build is handed its conversation's
    folder (another tab can re-point GEOTECH_DEFAULT_OUTPUT_DIR); the
    geotech page's build is not."""
    got, files = _captured_build(monkeypatch, tmp_path,
                                 **profiles.DOCUMENT_REVIEW.build_kwargs())
    assert got["working_dir"] == files and got["review_page"] is True
    got, _files = _captured_build(monkeypatch, tmp_path)
    assert "working_dir" not in got


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
