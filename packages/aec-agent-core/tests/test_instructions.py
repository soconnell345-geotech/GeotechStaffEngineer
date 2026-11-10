"""A deployment's instructions appended to every agent and helper (REF-64
(B), ``aec_agent_core.deep.instructions``). Made-up content only; nothing
of the geotechnical toolkit is imported."""

import hashlib
import os
import sys

import pytest

from aec_agent_core.deep import instructions as ins


def _folder(tmp_path, **files):
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    return str(tmp_path)


def _specs():
    return [{"name": "references", "system_prompt": "REF PROMPT"},
            {"name": "figure", "system_prompt": "FIG PROMPT"},
            {"name": "general-purpose", "system_prompt": "GP PROMPT"},
            {"name": "report_library", "runnable": object()}]


def test_unset_changes_nothing(monkeypatch):
    monkeypatch.delenv(ins.ENV, raising=False)
    specs = _specs()
    prompt, out, record = ins.apply("MAIN", specs)
    assert prompt == "MAIN" and record == {}
    assert all(a is b for a, b in zip(out, specs))


def test_each_role_gets_all_then_its_own_file_last(monkeypatch, tmp_path):
    monkeypatch.setenv(ins.ENV, _folder(
        tmp_path, **{"all.md": "Use metres.", "main.md": "Main rule.",
                     "references.md": "Prefer pack A."}))
    specs = _specs()
    prompt, out, record = ins.apply("MAIN", specs)
    tail = ("\n\n" + ins.PROTECTED_RULES + "\n\n" + ins.HEADING + "\n\n")
    assert prompt == "MAIN" + tail + "Use metres.\n\nMain rule."
    by = {s["name"]: s for s in out}
    assert by["references"]["system_prompt"] == (
        "REF PROMPT" + tail + "Use metres.\n\nPrefer pack A.")
    # a missing role file: all.md alone
    assert by["figure"]["system_prompt"] == "FIG PROMPT" + tail + "Use metres."
    assert by["general-purpose"]["system_prompt"].endswith("Use metres.")
    # a compiled helper is skipped, unchanged
    assert by["report_library"] is specs[3]
    assert specs[0]["system_prompt"] == "REF PROMPT"   # the caller's untouched
    roles = record["roles"]
    assert set(roles) == {"main", "references", "figure", "general-purpose"}
    assert roles["references"]["files"] == ["all.md", "references.md"]
    assert roles["figure"]["files"] == ["all.md"]
    assert record["folder"] == os.path.basename(str(tmp_path))
    block = tail + "Use metres.\n\nPrefer pack A."
    assert roles["references"]["sha"] == hashlib.sha256(
        block.encode()).hexdigest()
    assert roles["references"]["chars"] == len("Use metres.\n\nPrefer pack A.")
    assert roles["references"]["text"] == "Use metres.\n\nPrefer pack A."
    assert roles["references"]["cut"] is False


def test_no_text_for_a_role_appends_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv(ins.ENV, _folder(tmp_path, **{
        "references.md": "Only the librarian.", "all.md": "   \n"}))
    prompt, out, record = ins.apply("MAIN", _specs())
    assert prompt == "MAIN"
    assert {s["name"]: s.get("system_prompt") for s in out}["figure"] == (
        "FIG PROMPT")
    assert set(record["roles"]) == {"references"}


def test_unused_and_empty_files_are_named(monkeypatch, tmp_path):
    """A misspelt role name, or a role this build lacks, appends nothing;
    the record names the file so the deployment can see why."""
    monkeypatch.setenv(ins.ENV, _folder(tmp_path, **{
        "all.md": "", "reviwer.md": "Misspelt rule.", "mian.md": "Main?",
        "references.md": "Prefer pack A.", "report_library.md": "Lib.",
        "notes.txt": "not a role file"}))
    prompt, _, record = ins.apply("MAIN", _specs())
    assert prompt == "MAIN"
    assert set(record["roles"]) == {"references"}
    # report_library is a compiled helper of this build: present, not unused
    assert record["unused"] == ["mian.md", "reviwer.md"]
    assert record["empty"] == ["all.md"]
    # nothing to name: neither key
    clean = tmp_path / "clean"
    clean.mkdir()
    monkeypatch.setenv(ins.ENV, _folder(clean, **{"main.md": "Main."}))
    _, _, record = ins.apply("MAIN", _specs())
    assert "unused" not in record and "empty" not in record


def test_the_cap_cuts_at_a_paragraph_and_says_so(monkeypatch, tmp_path):
    para = " ".join(["Made-up convention sentence."] * 20)  # ~580 chars
    text = "\n\n".join(f"{i} {para}" for i in range(10))  # ~5,800
    monkeypatch.setenv(ins.ENV, _folder(tmp_path, **{"all.md": text}))
    prompt, _, record = ins.apply("MAIN", [])
    entry = record["roles"]["main"]
    assert entry["cut"] is True
    assert entry["chars"] <= ins.CAP_CHARS
    appended = prompt.split(ins.HEADING + "\n\n", 1)[1]
    assert len(appended) == entry["chars"]
    assert text.startswith(appended) and text[len(appended):].startswith(
        "\n\n")                                         # at a paragraph
    assert len(entry["text"]) == ins.RECORD_TEXT_CHARS


def test_an_unreadable_folder_appends_nothing_and_names_the_error(
        monkeypatch, tmp_path):
    monkeypatch.setenv(ins.ENV, str(tmp_path / "not_there"))
    prompt, out, record = ins.apply("MAIN", _specs())
    assert prompt == "MAIN"
    assert record["roles"] == {}
    assert record["error"].startswith("FileNotFoundError: ")
    # the folder's basename only, as "folder": never the host's path
    assert record["folder"] == "not_there"
    assert str(tmp_path) not in record["error"]
    assert "not_there" not in record["error"]


def test_a_file_that_is_not_utf8_is_skipped_and_named(monkeypatch, tmp_path):
    (tmp_path / "main.md").write_bytes("Caf\xe9 rule".encode("latin-1"))
    monkeypatch.setenv(ins.ENV, _folder(tmp_path, **{"all.md": "Fine."}))
    prompt, _, record = ins.apply("MAIN", [])
    assert prompt.endswith("Fine.")
    assert record["roles"]["main"]["files"] == ["all.md"]
    assert record["file_errors"]["main.md"].startswith("UnicodeDecodeError")
    assert str(tmp_path) not in record["file_errors"]["main.md"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_an_unreadable_file_is_skipped_and_named(monkeypatch, tmp_path):
    bad = tmp_path / "main.md"
    bad.write_bytes(b"\xff\xfe\x00bad")
    monkeypatch.setenv(ins.ENV, _folder(tmp_path, **{"all.md": "Fine."}))
    prompt, _, record = ins.apply("MAIN", [])
    assert prompt.endswith("Fine.")
    assert "main.md" in record["file_errors"]


def test_describe_says_what_each_role_gets(monkeypatch, tmp_path):
    monkeypatch.delenv(ins.ENV, raising=False)
    assert ins.describe() == (False, [f"none: {ins.ENV} is not set"])
    folder = tmp_path / "made_up_inst"
    folder.mkdir()
    monkeypatch.setenv(ins.ENV, _folder(folder, **{
        "all.md": "Use metres.", "main.md": "Main rule.",
        "reviwer.md": "Misspelt.", "figure.md": ""}))
    look, parts = ins.describe()
    assert look is True                                  # an empty file
    assert parts[0] == "folder made_up_inst"
    assert "every agent and helper: all.md, 11 characters" in parts
    assert "main: all.md + main.md, 23 characters" in parts
    assert "reviwer: all.md + reviwer.md, 22 characters" in parts
    assert "empty: figure.md" in parts
    assert any("unused" in p for p in parts)
    monkeypatch.setenv(ins.ENV, str(tmp_path / "not_there"))
    look, parts = ins.describe()
    assert look is True and parts[0].startswith(
        "folder not_there could not be read (FileNotFoundError")


def test_block_for_and_merge_record(monkeypatch, tmp_path):
    monkeypatch.setenv(ins.ENV, _folder(tmp_path, **{
        "all.md": "Everyone.", "report_ingest.md": "Ingest rule."}))
    block, record = ins.block_for("report_ingest")
    assert block.endswith("Everyone.\n\nIngest rule.")
    agent = type("A", (), {})()
    agent.geotech_build = {"tools": []}
    ins.merge_record(agent, record)
    assert agent.geotech_build["appended"]["roles"]["report_ingest"][
        "files"] == ["all.md", "report_ingest.md"]
    monkeypatch.delenv(ins.ENV)
    assert ins.block_for("report_ingest") == ("", {})


def test_the_engine_wrapper_appends_to_system():
    seen = {}

    class Fake:
        name = "fake-engine"
        pacer = None

        def complete(self, messages, *, system=None, **kw):
            seen["system"] = system
            seen["kw"] = kw
            return "reply"

    inner = Fake()
    wrapped = ins.InstructedEngine(inner, "\n\nBLOCK", {"files": ["all.md"]})
    assert wrapped.complete([], system="SHIPPED", max_tokens=5) == "reply"
    assert seen == {"system": "SHIPPED\n\nBLOCK", "kw": {"max_tokens": 5}}
    assert wrapped.name == "fake-engine" and wrapped._inner is inner
    wrapped.pacer = "p"
    assert inner.pacer == "p"
    assert wrapped.appended == {"files": ["all.md"]}
    # instructed(): the wrapper only where there is a block and an engine
    assert ins.instructed(inner) is inner
    assert ins.instructed(inner, "", None) is inner
    assert ins.instructed(None, "\n\nBLOCK", None) is None
    assert ins.instructed(inner, "\n\nBLOCK", None)._block == "\n\nBLOCK"
