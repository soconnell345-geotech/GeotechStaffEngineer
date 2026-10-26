"""The report-library sub-agent on the app's pages (BACKLOG RI-95), offline.

On for the GeotechStaffEngineer page when a library exists, off when none
does, and never on the Document Review page. The library is the folder named
by ``GEOTECH_REPORT_LIBRARY`` or else the ``report_ingest`` folder in the
conversation's working folder. Made-up records only.
"""

import os

import pytest

pytest.importorskip("deepagents")

from langchain_core.language_models.fake_chat_models import (  # noqa: E402
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage  # noqa: E402

from funhouse_agent.deep.prompt import REPORT_LIBRARY_NUDGE  # noqa: E402
from funhouse_agent.deep.tools import (  # noqa: E402
    REPORT_LIBRARY_ENV, report_ingest_out_dir,
)
from webapp import core  # noqa: E402


@pytest.fixture()
def model():
    return FakeMessagesListChatModel(responses=[AIMessage(content="ok")])


@pytest.fixture()
def files(monkeypatch, tmp_path):
    monkeypatch.setenv("GEOTECH_WEBAPP_DATA", str(tmp_path / "data"))
    monkeypatch.delenv(REPORT_LIBRARY_ENV, raising=False)
    monkeypatch.delenv("GEOTECH_DEFAULT_OUTPUT_DIR", raising=False)
    folder = os.path.join(core.conversation_dir("t-ri95"), "files")
    os.makedirs(folder)
    return folder


def _record(root, name="R-test"):
    folder = os.path.join(str(root), name)
    os.makedirs(folder)
    with open(os.path.join(folder, "report.record.json"), "w",
              encoding="utf-8") as handle:
        handle.write("{}")


def _build(monkeypatch, model, files, **kw):
    """``(tool names, the system prompt)`` of the real build."""
    from funhouse_agent.deep import agent as agent_mod
    captured = {}
    original = agent_mod.create_deep_agent

    def spy(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(agent_mod, "create_deep_agent", spy)
    agent = core.build_agent(model, {}, files, [], **kw)
    return set(agent.nodes["tools"].bound.tools_by_name), \
        captured["system_prompt"]


def test_with_no_library_there_is_no_tool_and_no_prompt_line(
        monkeypatch, model, files):
    tools, prompt = _build(monkeypatch, model, files)
    assert "report_library" not in tools
    assert REPORT_LIBRARY_NUDGE not in prompt


def test_a_record_in_the_conversations_folder_turns_it_on(
        monkeypatch, model, files):
    _record(os.path.join(core.working_dir_for("t-ri95"), "report_ingest"))
    tools, prompt = _build(monkeypatch, model, files)
    assert "report_library" in tools
    assert REPORT_LIBRARY_NUDGE in prompt


def test_the_setting_names_the_library_the_agent_reads(
        monkeypatch, model, files, tmp_path):
    library = tmp_path / "shared"
    library.mkdir()
    _record(library)
    monkeypatch.setenv(REPORT_LIBRARY_ENV, str(library))
    tools, prompt = _build(monkeypatch, model, files)
    assert "report_library" in tools
    assert REPORT_LIBRARY_NUDGE in prompt
    seen = {}
    import funhouse_agent.deep.agent as deep_agent
    monkeypatch.setattr(deep_agent, "build_deep_agent",
                        lambda model=None, **kw: seen.update(kw) or object())
    core.build_agent(model, {}, files, [])
    assert seen["library_root"] == str(library)
    assert seen["enable_report_library"] is True


def test_the_setting_with_no_record_in_it_leaves_it_off(
        monkeypatch, model, files, tmp_path):
    monkeypatch.setenv(REPORT_LIBRARY_ENV, str(tmp_path / "empty"))
    tools, prompt = _build(monkeypatch, model, files)
    assert "report_library" not in tools
    assert REPORT_LIBRARY_NUDGE not in prompt


def test_the_ingests_output_folder_follows_the_setting(
        monkeypatch, tmp_path):
    monkeypatch.delenv(REPORT_LIBRARY_ENV, raising=False)
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path / "work"))
    own = report_ingest_out_dir("/anywhere/Report One.pdf")
    assert own == os.path.join(str(tmp_path / "work"), "report_ingest",
                               "Report One")
    monkeypatch.setenv(REPORT_LIBRARY_ENV, str(tmp_path / "shared"))
    # a file that cannot be read to hash falls back to the bare file name
    assert report_ingest_out_dir("/anywhere/Report One.pdf") == \
        os.path.join(str(tmp_path / "shared"), "Report One")


def _pdf(folder, name, body):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(str(folder), name)
    with open(path, "wb") as handle:
        handle.write(body)
    return path


def test_two_files_of_one_name_get_two_shared_folders(monkeypatch, tmp_path):
    monkeypatch.setenv(REPORT_LIBRARY_ENV, str(tmp_path / "shared"))
    one = _pdf(tmp_path / "a", "Report.pdf", b"%PDF made-up body one")
    two = _pdf(tmp_path / "b", "Report.pdf", b"%PDF made-up body two!")
    assert report_ingest_out_dir(one) != report_ingest_out_dir(two)
    for path in (one, two):
        folder = os.path.basename(report_ingest_out_dir(path))
        assert folder.startswith("Report-")
        assert len(folder) == len("Report-") + 8


def test_the_same_file_twice_gets_one_shared_folder(monkeypatch, tmp_path):
    monkeypatch.setenv(REPORT_LIBRARY_ENV, str(tmp_path / "shared"))
    one = _pdf(tmp_path / "a", "Report.pdf", b"%PDF made-up body one")
    again = _pdf(tmp_path / "b", "Report.pdf", b"%PDF made-up body one")
    assert report_ingest_out_dir(one) == report_ingest_out_dir(again)
    # the resolved file decides the hash, not the name the primary gave
    named = os.path.basename(report_ingest_out_dir("attachment-key", one))
    assert named.startswith("attachment-key-")
    assert named.split("-")[-1] == \
        os.path.basename(report_ingest_out_dir(one)).split("-")[-1]


def test_the_folder_digest_is_the_records_own_hash(monkeypatch, tmp_path):
    from report_ingest.writers import source_digest
    monkeypatch.setenv(REPORT_LIBRARY_ENV, str(tmp_path / "shared"))
    path = _pdf(tmp_path, "Report.pdf", b"%PDF made-up body one")
    assert os.path.basename(report_ingest_out_dir(path)) == \
        "Report-" + source_digest(path)[:8]


def test_the_conversation_local_folder_has_no_digest(monkeypatch, tmp_path):
    monkeypatch.delenv(REPORT_LIBRARY_ENV, raising=False)
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path / "work"))
    path = _pdf(tmp_path, "Report.pdf", b"%PDF made-up body one")
    assert os.path.basename(report_ingest_out_dir(path)) == "Report"


def test_the_review_page_never_gets_it_even_with_the_setting(
        monkeypatch, model, files, tmp_path):
    library = tmp_path / "shared"
    library.mkdir()
    _record(library)
    monkeypatch.setenv(REPORT_LIBRARY_ENV, str(library))
    tools, prompt = _build(monkeypatch, model, files, review_page=True,
                           enable_report_library=True,
                           library_root=str(library))
    assert "report_library" not in tools
    assert REPORT_LIBRARY_NUDGE not in prompt
    seen = {}
    import funhouse_agent.deep.agent as deep_agent
    monkeypatch.setattr(deep_agent, "build_deep_agent",
                        lambda model=None, **kw: seen.update(kw) or object())
    core.build_agent(model, {}, files, [], review_page=True)
    assert "enable_report_library" not in seen
    assert "library_root" not in seen
