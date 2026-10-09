"""Live smoke wave 2c (module_work/live_smoke/runs/w2c-review-sonnet/
REVIEW.md), the app-side fixes.

* D2  when the AI budget runs out the tester sees ONE plain line -- no
      provider JSON, no request id, no "ask again"; the raw text goes to the
      activity log; the Tiny Apps client does not retry a spent quota, and
      still retries a plain rate limit.
* D3  the sidebar's SharePoint link is this conversation's own, never the
      last tester's.
* D6  "send me the links" for a file downloaded from SharePoint gives the
      ORIGINAL's link first, then the conversation's snapshot copy.
* D7  SharePoint search matches folder names and path segments, still
      inside the privacy scope.

Fakes only: no model, no network.
"""

from __future__ import annotations

import json
import os
import time
from types import SimpleNamespace

import pytest

import webapp.core as core
import webapp.sharepoint_store as sp

ANTHROPIC_BODY = {"type": "error", "error": {
    "type": "invalid_request_error",
    "message": ("You have reached your specified API usage limits. You will "
                "regain access on 2026-11-01 at 00:00 UTC.")},
    "request_id": "req_011CTest"}


class BadRequestError(Exception):
    """Shaped like anthropic.BadRequestError."""

    def __init__(self):
        super().__init__(f"Error code: 400 - {ANTHROPIC_BODY}")
        self.status_code = 400
        self.body = ANTHROPIC_BODY


class RateLimitError(Exception):
    """Shaped like openai.RateLimitError."""

    def __init__(self, code="insufficient_quota", message=None):
        inner = {"message": message or (
            "You exceeded your current quota, please check your plan and "
            "billing details."), "type": code, "param": None, "code": code}
        super().__init__(f"Error code: 429 - {{'error': {inner}}}")
        self.status_code = 429
        self.body = inner
        self.response = SimpleNamespace(status_code=429, headers={})


class BudgetExceededError(RuntimeError):
    """The Funhouse SDK's metered client raises this."""


# ---------------------------------------------------------------------------
# D2: what a tester sees when the budget runs out
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("exc, until", [
    (BadRequestError(), "until 2026-11-01 at 00:00 UTC"),
    (RateLimitError(), None),
    (BudgetExceededError("Monthly AI budget exceeded: $50.12 of $50.00"),
     None),
])
def test_a_spent_budget_is_one_plain_line(exc, until):
    text = core.friendly_turn_error(exc)
    assert text.startswith("The AI budget for this app is used up")
    assert text.endswith("Tell the app owner.")
    if until:
        assert until in text
    for raw in ("{", "request_id", "req_", "Error code", "400", "429",
                type(exc).__name__, "ask again", "Wait a minute",
                "rate limit"):
        assert raw not in text, raw
    assert core._busy_kind(exc) is None


def test_a_plain_rate_limit_still_says_wait_and_continue():
    text = core.friendly_turn_error(RateLimitError(
        code="rate_limit_exceeded",
        message="Rate limit reached for requests per minute"))
    assert "rate limit was reached" in text and "Wait a minute" in text


def _run_failing_turn(tmp_path, monkeypatch, exc, tid):
    import webapp.turn_jobs as tj
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    monkeypatch.delenv("GEOTECH_SHAREPOINT_SITE", raising=False)
    tj._JOBS.clear()

    def boom(agent, messages, thread_id, recursion_limit=None, **_kw):
        if False:
            yield {}
        raise exc

    monkeypatch.setattr(core, "stream_turn", boom)
    core.ensure_conversation(tid)
    files = core.conversation_files_dir(tid)
    os.makedirs(files, exist_ok=True)
    ctx = {"prompt": "what is on sheet 3?", "temp_dir": files,
           "before": core.snapshot_dir(files), "staged_inputs": set(),
           "working_dir": files, "before_wd": None, "artifacts": [],
           "artifacts_before_len": 0, "transcript": [], "trace_on": False,
           "model": "m", "behavior": core.default_behavior()}
    job = tj.start_turn_job(object(), [{"role": "user", "content": "q"}],
                            tid, 50, ctx)
    t0 = time.time()
    while not job.done and time.time() - t0 < 10:
        time.sleep(0.02)
    assert job.done
    return job


def test_a_turn_on_a_spent_budget_never_says_ask_again(tmp_path,
                                                       monkeypatch):
    """F43/F44/F46 t3: "ask again, or ask the agent to continue" over
    Anthropic's raw 400."""
    job = _run_failing_turn(tmp_path, monkeypatch, BadRequestError(),
                            "W2CBUDGET")
    assert job.result["final"] == core.NOTHING_KEPT_NO_RETRY
    assert "ask again" not in job.result["final"]
    assert job.result["error"].startswith(
        "The AI budget for this app is used up (until 2026-11-01")
    # The provider's own text is kept for the owner, in the activity log.
    from webapp import activity_log
    ends = [r for r in activity_log.load(core.conversation_dir("W2CBUDGET"))
            if r.get("event") == "turn_end"]
    assert ends and "req_011CTest" in ends[-1]["error_detail"]
    assert "req_011CTest" not in ends[-1]["error"]
    # ...and the transcript the tester reopens says the same plain thing.
    saved = core.load_transcript("W2CBUDGET")
    assert saved[-1]["error"] == job.result["error"]


def test_an_ordinary_failed_turn_still_offers_to_ask_again(tmp_path,
                                                           monkeypatch):
    job = _run_failing_turn(tmp_path, monkeypatch, ValueError("boom"),
                            "W2CPLAIN")
    assert job.result["final"] == core.NOTHING_KEPT


# -- the Tiny Apps client: a spent quota is not retried ----------------------

def _openai_client(handler, retries=3):
    openai = pytest.importorskip("openai")
    httpx = pytest.importorskip("httpx")
    import webapp.tinyapps_engine as te
    client = httpx.Client(
        transport=httpx.MockTransport(handler),
        event_hooks={"response": [te.no_retry_when_budget_spent]})
    return openai, openai.OpenAI(base_url="https://prompter.test/api/v1",
                                 api_key="k", http_client=client,
                                 max_retries=retries)


def _chat(client):
    return client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "hi"}])


def test_the_prompter_client_does_not_retry_a_spent_quota():
    httpx = pytest.importorskip("httpx")
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(429, json={"error": {
            "message": "You exceeded your current quota, please check your "
                       "plan and billing details.",
            "type": "insufficient_quota", "code": "insufficient_quota"}})

    openai, client = _openai_client(handler)
    with pytest.raises(openai.RateLimitError) as caught:
        _chat(client)
    assert len(calls) == 1                         # asked once, not 4 more
    assert core.friendly_turn_error(caught.value).startswith(
        "The AI budget for this app is used up")


def test_the_prompter_client_still_retries_a_rate_limit():
    httpx = pytest.importorskip("httpx")
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) < 3:
            return httpx.Response(
                429, headers={"retry-after-ms": "1"},
                json={"error": {"message": "Rate limit reached",
                                "type": "requests",
                                "code": "rate_limit_exceeded"}})
        return httpx.Response(200, json={
            "id": "c1", "object": "chat.completion", "created": 0,
            "model": "m", "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"}}]})

    _openai, client = _openai_client(handler)
    out = _chat(client)
    assert out.choices[0].message.content == "ok" and len(calls) == 3


def test_build_chat_model_installs_the_hook():
    pytest.importorskip("langchain_openai")
    import webapp.tinyapps_engine as te
    model = te.build_chat_model(prompter=te.PrompterSettings(
        url="https://prompter.test/api/v1/chat/completions", model="m",
        api_key="k"))
    hooks = model.http_client.event_hooks["response"]
    assert te.no_retry_when_budget_spent in hooks


# ---------------------------------------------------------------------------
# D3: the sidebar's SharePoint link is this conversation's own
# ---------------------------------------------------------------------------

class FakeFM:
    def __init__(self):
        self.uploads = []

    def create_folder(self, path):
        return True

    def upload_file(self, local, remote, overwrite=False):
        self.uploads.append(remote)
        return True

    def get_web_url(self, path):
        return f"https://sp.example/{path}"


def _conv(root, tid, title, owner):
    core.ensure_conversation(tid, root=str(root))
    core.touch_conversation(tid, root=str(root), title=title)
    meta = core.load_meta(tid, str(root))
    meta["owner"] = owner
    core.save_meta(tid, meta, str(root))
    files = os.path.join(core.conversation_dir(tid, str(root)), "files")
    os.makedirs(files, exist_ok=True)
    with open(os.path.join(files, "x.pdf"), "wb") as fh:
        fh.write(b"%PDF")


def test_a_fresh_conversation_never_shows_another_testers_folder(
        tmp_path, monkeypatch):
    for e in (sp.ENV_SITE, sp.ENV_TOKEN, sp.ENV_ROOT):
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    store = sp.SharePointStore(file_manager=FakeFM())
    alice_root, bob_root = tmp_path / "alice", tmp_path / "bob"
    _conv(alice_root, "ALICE01", "Alice review", "alice")
    _conv(bob_root, "BOB0001", "Bob review", "bob")
    core.register_thread_root("ALICE01", str(alice_root))
    core.register_thread_root("BOB0001", str(bob_root))

    alice = store.mirror_conversation("ALICE01", root=str(alice_root))
    assert "alice" in alice["folder"] and store.last_sync is alice
    # Bob's fresh session: nothing of Alice's, even with her summary in hand.
    assert sp.sync_for_conversation(store, "BOB0001") is None
    assert sp.sync_for_conversation(store, "BOB0001", alice) is None
    # After his own mirror, his own link only.
    bob = store.mirror_conversation("BOB0001", root=str(bob_root))
    shown = sp.sync_for_conversation(store, "BOB0001")
    assert shown is bob and "bob" in shown["web_url"]
    assert "alice" not in json.dumps(shown).lower()
    # Alice's own record is still hers.
    assert sp.sync_for_conversation(store, "ALICE01") is alice


def test_the_sidebar_of_a_fresh_session_shows_no_one_elses_link(
        tmp_path, monkeypatch):
    """The shell itself (AppTest): the store's last sync is another
    tester's; a new session's Permanent storage block must not show it."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    import webapp.engine_config as engine_config
    from webapp import profiles
    from webapp.engine_config import EngineResolution

    alice = {"uploaded": 3, "skipped": 0, "errors": [], "duration_s": 0.1,
             "folder": "R/conversations/alice/Alice_secret_review_2026-10-09",
             "web_url": "https://sp.example/alice-folder",
             "thread_id": "ALICE01"}

    class Store:
        configured = True
        last_sync = alice

        def last_sync_for(self, thread_id):
            return alice if thread_id == "ALICE01" else None

        def list_remote_conversations(self, owner=None, page=None):
            return []

        def mirror_conversation(self, thread_id, root=None):
            return {"uploaded": 0, "skipped": 0, "errors": [],
                    "duration_s": 0.0, "thread_id": thread_id}

    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path))
    for e in ("GEOTECH_APP_PROFILE", "DEV_IDENTITY", "GEOTECH_USER_EMAIL",
              "GEOTECH_DEPLOYMENT"):
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setenv("GEOTECH_UPLOAD_MODE", "http")
    monkeypatch.setattr(engine_config, "resolve_engine", lambda *a, **k:
                        EngineResolution(model=object(), source="prompter",
                                         model_name="fake", message=""))
    monkeypatch.setattr(core, "build_agent", lambda *a, **k: object())
    monkeypatch.setattr(core, "build_reviewer_agent", lambda *a, **k: object())
    store = Store()
    monkeypatch.setattr(sp, "get_store", lambda *a, **k: store)
    monkeypatch.setenv(profiles.PROFILE_ENV, "document_review")
    app = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "app.py")
    at = AppTest.from_file(app, default_timeout=30)
    at.run()
    assert not at.exception
    shown = " ".join([str(m.value) for m in at.sidebar.markdown]
                     + [str(c.value) for c in at.sidebar.caption])
    assert "Permanent storage" in " ".join(str(s.value)
                                           for s in at.sidebar.subheader)
    assert "alice-folder" not in shown and "Alice_secret" not in shown
    assert "mirroring is on" in shown


# ---------------------------------------------------------------------------
# D6 / D7: links for a downloaded file; search by folder names
# ---------------------------------------------------------------------------

pytest.importorskip("langchain_core")

import webapp.sharepoint_tools as spt  # noqa: E402
from webapp.tests.test_sharepoint_privacy import (  # noqa: E402
    ROOT, TreeFM, _conversation)

RIVERSIDE = f"{ROOT}/projects/Riverside Drive"
IFC = f"{RIVERSIDE}/drawings/Riverside Civil Set IFC 2026-09-30.pdf"
SPEC = f"{RIVERSIDE}/specs/Section 32 16 00.pdf"
LAKESIDE = f"{ROOT}/projects/Lakeside/plan.pdf"
ALICE_RIVERSIDE = (f"{ROOT}/conversations/alice/document_review/"
                   "Riverside_review_2026-10-08/files/notes.pdf")


@pytest.fixture
def site(monkeypatch, tmp_path):
    monkeypatch.setenv(sp.ENV_SITE, "https://t.sharepoint.com/sites/X")
    monkeypatch.setenv(sp.ENV_TOKEN, "tok")
    monkeypatch.setenv(sp.ENV_ROOT, ROOT)
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path / "data"))
    fm = TreeFM({IFC: b"%PDF ifc", SPEC: b"%PDF spec", LAKESIDE: b"%PDF lake",
                 ALICE_RIVERSIDE: b"%PDF alice"})
    monkeypatch.setattr(sp, "_STORE", sp.SharePointStore(file_manager=fm))
    spt._DOWNLOADS.clear()
    return fm


def test_a_downloaded_file_is_linked_to_its_original_first(
        site, monkeypatch, tmp_path):
    """F45 t4: the link handed out was the conversation's copy, which does
    not follow edits to the project drawing."""
    tid, rec, tools = _conversation(monkeypatch, tmp_path, "me", owner=False,
                                    multi_user=False)
    got = tools["sharepoint_download_file"].invoke({"path": IFC})
    assert got.startswith("Downloaded")
    out = tools["sharepoint_upload_file"].invoke(
        {"local_path": "Riverside Civil Set IFC 2026-09-30.pdf"})
    original = f"https://sp.example/{IFC}"
    assert "ORIGINAL is " + IFC in out and original in out
    session = sp.get_store().session_folder(tid)
    snapshot = f"{session}/files/Riverside Civil Set IFC 2026-09-30.pdf"
    assert "SNAPSHOT" in out and snapshot in out
    assert out.index(original) < out.index(snapshot)
    assert "no second copy" not in out and str(tmp_path) not in out


def test_a_file_the_conversation_made_keeps_the_one_copy_link(
        site, monkeypatch, tmp_path):
    tid, rec, tools = _conversation(monkeypatch, tmp_path, "me", owner=False,
                                    multi_user=False)
    (rec / "files" / "memo.docx").write_bytes(b"PK memo")
    out = tools["sharepoint_upload_file"].invoke({"local_path": "memo.docx"})
    assert "ONE copy" in out and "ORIGINAL" not in out


def test_search_finds_files_by_the_folder_they_are_in(site, monkeypatch,
                                                      tmp_path):
    """F45 t1: "riverside dr" found nothing although "Riverside Drive" held
    every set."""
    _tid, _rec, tools = _conversation(monkeypatch, tmp_path, "bob")
    out = tools["sharepoint_search_files"].invoke({"query": "riverside dr"})
    assert "Riverside Civil Set IFC" in out
    assert "matched by the folder" in out
    assert "Lakeside" not in out
    assert "alice" not in out.lower()                       # A1 still holds


def test_with_no_index_the_walk_lists_the_folder_and_its_files(
        site, monkeypatch, tmp_path):
    site.search_filenames = lambda query, path=None: []
    _tid, _rec, tools = _conversation(monkeypatch, tmp_path, "bob")
    out = tools["sharepoint_search_files"].invoke({"query": "riverside dr"})
    assert "[folder] Riverside Drive" in out
    assert "Riverside Civil Set IFC" in out and "Section 32 16 00" in out
    assert "Lakeside" not in out and "alice" not in out.lower()
    # A query that names nothing there still says so.
    none = tools["sharepoint_search_files"].invoke({"query": "hillside"})
    assert none.startswith("No files matching")
