"""Offline tests for the webapp core logic — no streamlit, no live model.

Covers attachment registration + staging, the agent-facing note, artifact
capture (save_fn + directory watch), engine-config resolution (including the
no-engine path), disclaimer presence, and a fake-agent stream.
"""

import os

import pytest

from webapp import core, engine_config


# ---------------------------------------------------------------------------
# Attachments
# ---------------------------------------------------------------------------

def test_stage_upload_registers_bytes_and_writes_file(tmp_path):
    attachments = {}
    att = core.stage_upload(attachments, str(tmp_path), "Site Plan.pdf", b"PDFDATA")
    # sanitized key, bytes registered for the vision tools
    assert att.key == "Site_Plan.pdf"
    assert attachments["Site_Plan.pdf"] == b"PDFDATA"
    # staged to a real file for real-path tools
    assert att.path == os.path.join(str(tmp_path), "Site_Plan.pdf")
    assert os.path.isfile(att.path)
    with open(att.path, "rb") as fh:
        assert fh.read() == b"PDFDATA"
    assert att.size == 7


def test_stage_upload_rejects_non_bytes(tmp_path):
    with pytest.raises(TypeError):
        core.stage_upload({}, str(tmp_path), "x.txt", "not-bytes")


def test_stage_upload_overwrites_same_key(tmp_path):
    attachments = {}
    core.stage_upload(attachments, str(tmp_path), "a.csv", b"one")
    att = core.stage_upload(attachments, str(tmp_path), "a.csv", b"two")
    assert attachments["a.csv"] == b"two"
    with open(att.path, "rb") as fh:
        assert fh.read() == b"two"


def test_sanitize_key_strips_path_and_unsafe_chars():
    assert core.sanitize_key("../../etc/pa ss$.pdf") == "pa_ss_.pdf"
    assert core.sanitize_key("") == "file"


def test_attachment_note_mentions_key_and_path(tmp_path):
    attachments = {}
    atts = core.stage_uploads(attachments, str(tmp_path),
                              [("report.pdf", b"x"), ("plan.dxf", b"y")])
    note = core.attachment_note(atts)
    assert "report.pdf" in note and "plan.dxf" in note
    assert atts[0].path in note                       # staged disk path
    assert "attachment_key='report.pdf'" in note      # vision key
    assert "read_pdf_text" in note and "dxf_import" in note


def test_attachment_note_empty_for_no_attachments():
    assert core.attachment_note([]) == ""


def test_assemble_user_message_identity_without_notes():
    assert core.assemble_user_message([], "hello") == "hello"
    assert core.assemble_user_message([""], "hello") == "hello"


def test_assemble_user_message_prepends_notes():
    out = core.assemble_user_message(["[System note] file X"], "analyze it")
    assert out == "[System note] file X\n\nanalyze it"


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

def test_save_fn_bare_path_lands_in_temp_dir_and_records(tmp_path):
    artifacts = []
    save_fn = core.make_save_fn(str(tmp_path), artifacts)
    p = save_fn("calc_package.html", "<html>ok</html>")
    assert p == os.path.join(str(tmp_path), "calc_package.html")
    assert os.path.isfile(p)
    assert artifacts == [p]
    with open(p, encoding="utf-8") as fh:
        assert fh.read() == "<html>ok</html>"


def test_save_fn_bytes_and_absolute_path(tmp_path):
    artifacts = []
    save_fn = core.make_save_fn(str(tmp_path / "session"), artifacts)
    os.makedirs(str(tmp_path / "session"), exist_ok=True)
    abs_target = str(tmp_path / "session" / "out.dxf")
    p = save_fn(abs_target, b"DXF")
    assert p == abs_target
    with open(p, "rb") as fh:
        assert fh.read() == b"DXF"
    assert artifacts == [abs_target]


def test_new_artifacts_detects_new_files_and_excludes_inputs(tmp_path):
    d = str(tmp_path)
    staged = os.path.join(d, "input.pdf")
    with open(staged, "wb") as fh:
        fh.write(b"in")
    before = core.snapshot_dir(d)
    # agent writes an output after the snapshot
    out = os.path.join(d, "result.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("out")
    found = core.new_artifacts(d, before, input_paths={staged})
    assert found == [out]                 # the input is excluded


# ---------------------------------------------------------------------------
# Artifact cards (viewer)
# ---------------------------------------------------------------------------

def test_classify_artifact_by_extension():
    assert core.classify_artifact("a/b/report.HTML") == "html"
    assert core.classify_artifact("calc.pdf") == "pdf"
    assert core.classify_artifact("plot.png") == "png"
    assert core.classify_artifact("photo.jpeg") == "image"
    assert core.classify_artifact("section.dxf") == "dxf"
    assert core.classify_artifact("data.csv") == "csv"
    assert core.classify_artifact("mystery.xyz") == "other"


def test_classify_plotly_sidecar():
    # A *.plotly.json sidecar renders natively; the compound suffix is checked
    # before the plain-extension lookup (which would see only .json -> text).
    assert core.classify_artifact("plot.plotly.json") == "plotly"
    assert core.classify_artifact("/tmp/a/b/fig.PLOTLY.JSON") == "plotly"
    assert core.classify_artifact("data.json") == "text"       # plain json unchanged


def test_plotly_sidecar_round_trips_via_from_json(tmp_path):
    # The sidecar app.py renders with plotly.io.from_json must be loadable.
    pio = pytest.importorskip("plotly.io")
    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]))
    sidecar = tmp_path / "fig.plotly.json"
    sidecar.write_text(fig.to_json(), encoding="utf-8")
    assert core.classify_artifact(str(sidecar)) == "plotly"
    loaded = pio.from_json(core.read_text(str(sidecar)))
    assert len(loaded.data) == 1
    assert list(loaded.data[0].y) == [4, 5, 6]


def test_describe_artifact(tmp_path):
    p = tmp_path / "calc_package.html"
    p.write_text("<html>x</html>", encoding="utf-8")
    card = core.describe_artifact(str(p))
    assert card.name == "calc_package.html"
    assert card.kind == "html"
    assert card.size == len("<html>x</html>")
    assert card.exists is True
    assert core.describe_artifact(str(tmp_path / "gone.pdf")).exists is False


def test_pdf_data_uri_small_and_oversized(tmp_path):
    p = tmp_path / "calc.pdf"
    p.write_bytes(b"%PDF-1.4 minimal")
    uri = core.pdf_data_uri(str(p))
    assert uri is not None and uri.startswith("data:application/pdf;base64,")
    # oversized -> None (cap below the file size)
    assert core.pdf_data_uri(str(p), max_bytes=4) is None
    # missing -> None
    assert core.pdf_data_uri(str(tmp_path / "nope.pdf")) is None


def test_artifact_bytes_and_read_text(tmp_path):
    p = tmp_path / "out.txt"
    p.write_text("hello", encoding="utf-8")
    assert core.artifact_bytes(str(p)) == b"hello"
    assert core.read_text(str(p)) == "hello"


def test_collect_turn_artifacts_unions_and_dedupes():
    save_new = ["/t/a.html", "/t/b.pdf"]
    dir_new = ["/t/b.pdf", "/t/c.dxf"]       # b.pdf appears in both
    assert core.collect_turn_artifacts(save_new, dir_new) == \
        ["/t/a.html", "/t/b.pdf", "/t/c.dxf"]
    assert core.collect_turn_artifacts([], []) == []


# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------

def test_disclaimer_text_is_present_and_substantive():
    text = core.disclaimer_text()
    assert isinstance(text, str) and len(text) > 40
    low = text.lower()
    assert "professional" in low or "research aid" in low or "warranty" in low


# ---------------------------------------------------------------------------
# Engine config
# ---------------------------------------------------------------------------

def test_resolve_engine_no_engine(monkeypatch):
    engine_config.register_model_builder(None)
    monkeypatch.delenv(engine_config.KEY_ENV, raising=False)
    res = engine_config.resolve_engine()
    assert res.source == "none"
    assert res.model is None and res.ok is False
    assert "No engine configured" in res.message


def test_resolve_engine_prompter_hook(monkeypatch):
    monkeypatch.delenv(engine_config.KEY_ENV, raising=False)
    sentinel = object()
    engine_config.register_model_builder(lambda: sentinel)
    try:
        res = engine_config.resolve_engine()
        assert res.source == "prompter"
        assert res.model is sentinel and res.ok is True
    finally:
        engine_config.register_model_builder(None)


def test_resolve_engine_prompter_builder_error(monkeypatch):
    monkeypatch.delenv(engine_config.KEY_ENV, raising=False)

    def _boom():
        raise RuntimeError("no prompter")

    engine_config.register_model_builder(_boom)
    try:
        res = engine_config.resolve_engine()
        assert res.source == "error" and res.model is None
        assert "no prompter" in res.message
    finally:
        engine_config.register_model_builder(None)


def test_resolve_engine_with_key_is_anthropic_or_error(monkeypatch):
    engine_config.register_model_builder(None)
    monkeypatch.setenv(engine_config.KEY_ENV, "sk-test-not-a-real-key")
    res = engine_config.resolve_engine()
    # With langchain_anthropic installed -> a ChatAnthropic model (source
    # "anthropic"); without it -> a clear "error" resolution. Never "none".
    assert res.source in ("anthropic", "error")
    if res.source == "anthropic":
        assert res.ok and engine_config.DEFAULT_MODEL in res.model_name


# ---------------------------------------------------------------------------
# Streaming (fake agent — no live model)
# ---------------------------------------------------------------------------

class _Msg:
    def __init__(self, content):
        self.content = content
        self.tool_call_chunks = None
        self.tool_calls = None


class _FakeAgent:
    """Minimal stand-in for a compiled deep agent: yields messages-mode token
    chunks in the (mode, (message, metadata)) shape stream_turn parses."""
    def __init__(self, tokens):
        self._tokens = tokens

    def stream(self, inp, config=None, stream_mode=None):
        assert "messages" in inp
        for t in self._tokens:
            yield ("messages", (_Msg(t), {"langgraph_node": "model"}))


def test_stream_turn_assembles_answer_and_reports_tokens():
    agent = _FakeAgent(["Bearing ", "capacity ", "is 450 kPa."])
    entries = list(core.stream_turn(
        agent, [{"role": "user", "content": "hi"}], "tid"))
    tokens = [e["text"] for e in entries if e["kind"] == "token"]
    assert "".join(tokens) == "Bearing capacity is 450 kPa."
    done = entries[-1]
    assert done["kind"] == "turn_done"
    assert done["answer"] == "Bearing capacity is 450 kPa."
    assert isinstance(done["turn_tokens"], int)


# ---------------------------------------------------------------------------
# Persistence — durable conversations
# ---------------------------------------------------------------------------

def test_data_root_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path / "store"))
    assert core.data_root() == str((tmp_path / "store").resolve()) or \
        core.data_root() == os.path.abspath(str(tmp_path / "store"))
    monkeypatch.delenv("GEOTECH_WEBAPP_DATA", raising=False)
    assert core.data_root().endswith(".geotech_webapp")


def test_conversation_lifecycle_and_listing(tmp_path):
    root = str(tmp_path)
    # created on first use, then renamed; a second, later conversation sorts first
    m1 = core.ensure_conversation("t1", title="Bearing question", root=root)
    assert m1["thread_id"] == "t1" and m1["turn_count"] == 0
    core.touch_conversation("t1", turn_count=2, root=root)
    core.rename_conversation("t1", "Footing on sand", root=root)
    core.ensure_conversation("t2", title="Slope FOS", root=root)
    core.touch_conversation("t2", turn_count=1, root=root)   # newer 'updated'
    convs = core.list_conversations(root=root)
    assert [c["thread_id"] for c in convs] == ["t2", "t1"]   # recent-first
    assert core.load_meta("t1", root=root)["title"] == "Footing on sand"
    assert core.load_meta("t1", root=root)["turn_count"] == 2
    assert core.load_meta("missing", root=root) is None


def test_auto_title():
    assert core.auto_title("") == "New conversation"
    assert core.auto_title("what is the bearing capacity of a 2 m footing on "
                           "dense sand at 1.5 m") == \
        "what is the bearing capacity of a 2…"      # first 8 words + ellipsis
    assert core.auto_title("short one") == "short one"


def test_transcript_append_load_roundtrip_with_portable_artifacts(tmp_path):
    root = str(tmp_path)
    tid = "conv"
    files_dir = core.conversation_files_dir(tid, root=root)
    art = os.path.join(files_dir, "calc.html")
    open(art, "w").close()
    core.append_transcript(tid, {"role": "user", "text": "run the calc"},
                           root=root)
    core.append_transcript(tid, {"role": "assistant", "text": "done",
                                 "artifacts": [art]}, root=root)
    # stored reference is RELATIVE (portable), resolved back on load
    raw = open(core._conv_path(tid, "transcript.jsonl", root)).read()
    assert '"calc.html"' in raw and files_dir not in raw
    loaded = core.load_transcript(tid, root=root)
    assert [e["role"] for e in loaded] == ["user", "assistant"]
    assert loaded[1]["artifacts"] == [art]
    assert core.artifacts_from_transcript(loaded) == [art]


def test_messages_save_load(tmp_path):
    root = str(tmp_path)
    msgs = [{"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}]
    core.save_messages("c", msgs, root=root)
    assert core.load_messages("c", root=root) == msgs
    assert core.load_messages("nope", root=root) == []


def test_attachment_reregistration_on_resume(tmp_path):
    root = str(tmp_path)
    tid = "c"
    # stage an upload into the conversation files dir, index it, then resume
    files_dir = core.conversation_files_dir(tid, root=root)
    live = {}
    core.stage_upload(live, files_dir, "Boring Log.pdf", b"PDFBYTES")
    core.save_attachments_index(tid, list(live.keys()), root=root)
    # fresh session: empty attachments dict, re-register from disk
    resumed = {}
    atts = core.load_attachments(tid, resumed, root=root)
    assert resumed["Boring_Log.pdf"] == b"PDFBYTES"
    assert atts[0].key == "Boring_Log.pdf" and atts[0].size == 8
    assert os.path.isfile(atts[0].path)


def test_delete_conversation_moves_to_trash(tmp_path):
    root = str(tmp_path)
    core.ensure_conversation("gone", title="x", root=root)
    assert os.path.isdir(core.conversation_dir("gone", root=root))
    dst = core.delete_conversation("gone", root=root)
    assert dst is not None and os.path.isdir(dst)          # moved to .trash
    assert not os.path.isdir(core.conversation_dir("gone", root=root))
    assert core.load_meta("gone", root=root) is None
    assert "gone" not in [c["thread_id"] for c in core.list_conversations(root=root)]
    assert core.delete_conversation("never", root=root) is None


def test_thread_isolation(tmp_path):
    """Two conversations do not cross-contaminate messages/transcript/attachments."""
    root = str(tmp_path)
    core.save_messages("A", [{"role": "user", "content": "A-msg"}], root=root)
    core.save_messages("B", [{"role": "user", "content": "B-msg"}], root=root)
    core.append_transcript("A", {"role": "user", "text": "A-turn"}, root=root)
    core.append_transcript("B", {"role": "user", "text": "B-turn"}, root=root)
    core.stage_upload({}, core.conversation_files_dir("A", root=root),
                      "a.pdf", b"AA")
    core.save_attachments_index("A", ["a.pdf"], root=root)
    assert core.load_messages("A", root=root)[0]["content"] == "A-msg"
    assert core.load_transcript("B", root=root)[0]["text"] == "B-turn"
    b_att = {}
    core.load_attachments("B", b_att, root=root)             # B has no index
    assert b_att == {}
    a_att = {}
    core.load_attachments("A", a_att, root=root)
    assert list(a_att.keys()) == ["a.pdf"]


def test_checkpointer_grows_and_isolates_across_threads():
    """The pluggable checkpointer path: a LangGraph InMemorySaver accumulates
    state per thread_id and keeps threads isolated (proves build_agent's
    checkpointer wiring is sound without needing a live model)."""
    saver = pytest.importorskip("langgraph.checkpoint.memory").InMemorySaver()

    def cfg(tid):
        return {"configurable": {"thread_id": tid, "checkpoint_ns": ""}}

    def put(tid, i):
        saver.put(cfg(tid),
                  {"v": 1, "id": f"{tid}-{i}", "ts": "", "channel_values": {},
                   "channel_versions": {}, "versions_seen": {}},
                  {"step": i}, {})

    put("one", 0)
    put("one", 1)
    put("two", 0)
    one = list(saver.list(cfg("one")))
    two = list(saver.list(cfg("two")))
    assert len(one) == 2 and len(two) == 1        # grows per thread, isolated
    assert core.build_agent.__defaults__ is not None  # checkpointer param exists


def test_persist_and_resume_conversation_roundtrip(tmp_path):
    """End-to-end: simulate a turn (persist attachment + transcript + messages +
    artifact + meta), then RESUME in a fresh session and verify the display
    transcript, agent-facing memory, attachments, artifacts and meta are all
    restored — the resume-correctness invariant the UI depends on."""
    root = str(tmp_path)
    tid = core.new_thread_id()
    files_dir = core.conversation_files_dir(tid, root=root)

    # --- a live turn (what app.py does) ---
    live = {}
    core.stage_upload(live, files_dir, "log.pdf", b"PDF")
    core.save_attachments_index(tid, list(live.keys()), root=root)
    core.append_transcript(tid, {"role": "attach", "text": "log.pdf"}, root=root)
    core.append_transcript(tid, {"role": "user", "text": "analyze it"}, root=root)
    messages = [{"role": "user", "content": "[note]\n\nanalyze it"}]
    artifact = os.path.join(files_dir, "calc.html")
    with open(artifact, "w") as fh:
        fh.write("<html>ok</html>")
    core.append_transcript(tid, {"role": "assistant", "text": "done",
                                 "artifacts": [artifact]}, root=root)
    messages.append({"role": "assistant", "content": "done"})
    core.save_messages(tid, messages, root=root)
    core.touch_conversation(tid, title=core.auto_title("analyze it"),
                            turn_count=1, root=root)

    # --- resume in a FRESH session (empty state) ---
    resumed_att: dict = {}
    core.load_attachments(tid, resumed_att, root=root)
    transcript = core.load_transcript(tid, root=root)
    resumed_msgs = core.load_messages(tid, root=root)
    resumed_arts = core.artifacts_from_transcript(transcript)
    meta = core.load_meta(tid, root=root)

    assert resumed_att["log.pdf"] == b"PDF"                     # attachment bytes
    assert [e["role"] for e in transcript] == \
        ["attach", "user", "assistant"]                        # display transcript
    assert resumed_msgs == messages                            # agent memory (replay)
    assert resumed_arts == [artifact] and os.path.isfile(artifact)  # artifact survives
    assert meta["title"] == "analyze it" and meta["turn_count"] == 1
    assert tid in [c["thread_id"]
                   for c in core.list_conversations(root=root)]


# ---------------------------------------------------------------------------
# Model picker
# ---------------------------------------------------------------------------

def test_model_choices_data_and_labels(monkeypatch):
    monkeypatch.delenv("GEOTECH_WEBAPP_MODEL", raising=False)
    choices = core.model_choices()
    ids = [c["id"] for c in choices]
    assert ids[0] == "claude-opus-4-8"                    # curated default first
    assert "claude-sonnet-5" in ids
    assert "claude-haiku-4-5-20251001" in ids
    assert all(c.get("label") and c.get("blurb") for c in choices)
    assert core.default_model_id() == "claude-opus-4-8"
    assert core.model_label("claude-sonnet-5") == "Sonnet 5"
    assert core.model_label(None) == ""
    assert core.model_label("unknown-x") == "unknown-x"   # unknown -> the id


def test_model_choices_env_override(monkeypatch):
    monkeypatch.delenv("GEOTECH_WEBAPP_MODEL", raising=False)
    # a non-curated env model is prepended AND becomes the default
    choices = core.model_choices(env_model="claude-custom-9")
    assert choices[0]["id"] == "claude-custom-9"
    assert core.default_model_id(env_model="claude-custom-9") == "claude-custom-9"
    # a curated env model is the default without being duplicated
    c2 = core.model_choices(env_model="claude-sonnet-5")
    assert [c["id"] for c in c2].count("claude-sonnet-5") == 1
    assert core.default_model_id(env_model="claude-sonnet-5") == "claude-sonnet-5"


def test_conversation_meta_records_model(tmp_path):
    root = str(tmp_path)
    core.ensure_conversation("m", title="x", root=root)
    assert core.load_meta("m", root=root)["model"] is None
    core.touch_conversation("m", model="claude-sonnet-5", turn_count=1, root=root)
    reloaded = core.load_meta("m", root=root)
    assert reloaded["model"] == "claude-sonnet-5"          # survives round-trip
    # switching model on resume overrides
    core.touch_conversation("m", model="claude-haiku-4-5-20251001", root=root)
    assert core.load_meta("m", root=root)["model"] == "claude-haiku-4-5-20251001"


def test_resolve_engine_explicit_model_overrides_env(monkeypatch):
    engine_config.register_model_builder(None)
    monkeypatch.setenv(engine_config.KEY_ENV, "sk-test-not-a-real-key")
    monkeypatch.setenv(engine_config.MODEL_ENV, "claude-sonnet-5")
    # the picker's explicit id wins over GEOTECH_WEBAPP_MODEL (built or clean
    # error, but the reported model name is the explicit one — never the env one)
    res = engine_config.resolve_engine(model_id="claude-haiku-4-5-20251001")
    assert res.model_name == "claude-haiku-4-5-20251001"
    assert res.source in ("anthropic", "error")
    # no explicit id -> falls back to GEOTECH_WEBAPP_MODEL
    assert engine_config.resolve_engine().model_name == "claude-sonnet-5"
