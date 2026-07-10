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
