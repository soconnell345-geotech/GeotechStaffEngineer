"""Finding 11 of ``module_work/record_masks/REVIEW.md``: three plain-text
failure messages the model reads carried an exception's text raw -- the
SharePoint upload of a conversation's own file, the minimal page's helper
("The helper failed: ...") and the feedback tool. Each now goes through
``error_text.mask_error_secrets``, and none is handed on raw if the masker
itself fails.

Every value here is made up (``Made0Up...``, ``example.test``), never a live
key of any vendor. A message with no secret is character for character what
it was.
"""

from __future__ import annotations

import pytest

from aec_agent_core import error_text
from aec_document_review import feedback, minimal_agent
from aec_document_review import sharepoint_store as sp
from aec_document_review import sharepoint_tools as spt

#: A made-up value: letters and digits, mixed, 21 characters.
VALUE = "Made0UpValue1For2Test"
SECRET = (f"403 Client Error: Forbidden for url: https://files.example.test"
          f"/a/d.pdf?download=1&tempauth={VALUE}")
PLAIN = ("403 Client Error: Forbidden for url: https://files.example.test"
         "/a/d.pdf?download=1 (request 0f3a)")


class Refused(Exception):
    """A made-up client error that quotes the request it was refused."""


@pytest.fixture()
def broken_masker(monkeypatch):
    def broken(_text):
        raise RuntimeError("the masker broke")
    return lambda: monkeypatch.setattr(error_text, "_mask", broken)


# ---------------------------------------------------------------------------
# the SharePoint upload of a file of this conversation
# ---------------------------------------------------------------------------

@pytest.fixture()
def upload(monkeypatch, tmp_path):
    """The upload tool of a made-up conversation whose mirrored copy cannot
    be shared: ``upload(text)`` is what the tool returns when that fails
    with ``text``."""
    monkeypatch.setenv(sp.ENV_SITE, "https://t.example.test/sites/x")
    monkeypatch.setenv(sp.ENV_TOKEN, "tok")
    monkeypatch.setenv(sp.ENV_ROOT, "Shared Documents/General/Made_up")
    monkeypatch.setattr(sp, "_STORE", sp.SharePointStore(file_manager=object()))
    conv = tmp_path / "conv"
    conv.mkdir()
    local = conv / "memo.docx"
    local.write_bytes(b"made up")
    tools, _ = spt.tools_if_configured(thread_id="t1", record_dir=str(conv))
    tool = next(t for t in tools if t.name == "sharepoint_upload_file")

    def run(text):
        def boom(*args, **kwargs):
            raise Refused(text)
        monkeypatch.setattr(spt, "_share_mirrored_copy", boom)
        return tool.invoke({"local_path": str(local)})
    return run


def test_a_failed_share_of_a_conversations_file_keeps_no_secret(upload):
    out = upload(SECRET)
    assert VALUE not in out
    assert out.startswith("SharePoint upload error: Refused: 403 Client")
    assert "tempauth=…" in out


def test_a_failed_share_with_no_secret_reads_as_before(upload):
    assert upload(PLAIN) == f"SharePoint upload error: Refused: {PLAIN}"


def test_a_failed_share_is_not_handed_on_raw_when_masking_fails(
        upload, broken_masker):
    broken_masker()
    assert upload(SECRET) == spt.MESSAGE_NOT_CHECKED


# ---------------------------------------------------------------------------
# the minimal page's helper
# ---------------------------------------------------------------------------

@pytest.fixture()
def helper(monkeypatch):
    """``helper(text)``: what the ``task`` tool returns when the helper
    fails with ``text``."""
    pytest.importorskip("langchain.agents")
    import langchain.agents

    def run(text):
        def boom(*args, **kwargs):
            raise Refused(text)
        monkeypatch.setattr(langchain.agents, "create_agent", boom)
        tool = minimal_agent._helper_tool(object(), None, [])
        return tool.invoke({"description": "look at page 3"})
    return run


def test_a_failed_helper_keeps_no_secret(helper):
    out = helper(SECRET)
    assert VALUE not in out and "tempauth=…" in out
    assert out.startswith("The helper failed: Refused: 403 Client")
    assert out.endswith(". Look yourself, or hand it a smaller job.")


def test_a_failed_helper_with_no_secret_reads_as_before(helper):
    assert helper(PLAIN) == (f"The helper failed: Refused: {PLAIN}. Look "
                             f"yourself, or hand it a smaller job.")
    long = "x" * 400
    assert helper(long) == (f"The helper failed: Refused: {long[:300]}. Look "
                            f"yourself, or hand it a smaller job.")


def test_a_failed_helper_is_not_handed_on_raw_when_masking_fails(
        helper, broken_masker):
    broken_masker()
    out = helper(SECRET)
    assert VALUE not in out and "tempauth" not in out
    assert out == (f"The helper failed: Refused: "
                   f"{minimal_agent.REASON_NOT_CHECKED}. Look yourself, or "
                   f"hand it a smaller job.")


# ---------------------------------------------------------------------------
# the feedback tool
# ---------------------------------------------------------------------------

@pytest.fixture()
def record_feedback(monkeypatch, tmp_path):
    def run(text):
        def boom(*args, **kwargs):
            raise Refused(text)
        monkeypatch.setattr(feedback, "record", boom)
        tool = feedback.make_record_feedback_tool(str(tmp_path))
        return tool.invoke({"kind": "other", "summary": "made up"})
    return run


def test_feedback_that_could_not_be_recorded_keeps_no_secret(record_feedback):
    out = record_feedback(SECRET)
    assert VALUE not in out and "tempauth=…" in out
    assert out.startswith("Feedback could not be recorded (Refused: 403 ")


def test_feedback_failing_with_no_secret_reads_as_before(record_feedback):
    assert record_feedback(PLAIN) == (
        f"Feedback could not be recorded (Refused: {PLAIN}). Continue with "
        f"the task; mention the gap in your answer instead.")


def test_feedback_is_not_handed_on_raw_when_masking_fails(
        record_feedback, broken_masker):
    broken_masker()
    assert record_feedback(SECRET) == (
        f"Feedback could not be recorded (Refused: "
        f"{feedback.REASON_NOT_CHECKED}). Continue with the task; mention "
        f"the gap in your answer instead.")
