"""The review of the record's masker, Document Review's part
(``module_work/record_masks/REVIEW.md``). Made-up values only."""

from __future__ import annotations

VALUE = "Made0UpValue1For2Test"
SIGNED = f"https://files.example.test/a/d.pdf?download=1&tempauth={VALUE}"


def test_a_sharepoint_failure_is_not_handed_on_raw_when_masking_fails(
        monkeypatch):
    from aec_agent_core import error_text
    from aec_document_review import sharepoint_log, sharepoint_tools
    monkeypatch.setattr(sharepoint_log, "record",
                        lambda step, message, path="", **kw: None)
    exc = RuntimeError(f"403 Client Error: Forbidden for url: {SIGNED}")
    masked = sharepoint_tools._failure("download", exc, "Docs/d.pdf")
    assert VALUE not in masked and "tempauth=…" in masked

    def broken(_text):
        raise RuntimeError("the masker broke")
    monkeypatch.setattr(error_text, "_mask", broken)
    said = sharepoint_tools._failure("download", exc, "Docs/d.pdf")
    assert said == sharepoint_tools.MESSAGE_NOT_CHECKED
    assert sharepoint_tools._logged("sharepoint list", "") == ""
