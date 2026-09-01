"""Offline tests for the agent-facing email tool (fake Funhouse SDK)."""

import sys
import types

import pytest

pytest.importorskip("langchain_core")

import webapp.email_tools as et  # noqa: E402


# ---------------------------------------------------------------------------
# Fake SDK (funhouse.services.email in sys.modules)
# ---------------------------------------------------------------------------

def _install_fake_sdk(monkeypatch, send_exc=None):
    sent = {}

    class FakeEmail:
        def __init__(self, config=None, password=None):
            pass

        def send_email(self, to, subject, body, cc=None, bcc=None,
                       attachments=None, is_html=False, is_markdown=False,
                       allow_large_send=False, remove_footer=False):
            if send_exc is not None:
                raise send_exc
            sent.update(to=to, subject=subject, body=body,
                        attachments=attachments)

    fh = types.ModuleType("funhouse")
    services = types.ModuleType("funhouse.services")
    email_mod = types.ModuleType("funhouse.services.email")
    email_mod.FunhouseEmail = FakeEmail
    fh.services, services.email = services, email_mod
    for name, mod in (("funhouse", fh), ("funhouse.services", services),
                      ("funhouse.services.email", email_mod)):
        monkeypatch.setitem(sys.modules, name, mod)
    return sent


def _uninstall_sdk(monkeypatch):
    for name in ("funhouse", "funhouse.services", "funhouse.services.email"):
        monkeypatch.setitem(sys.modules, name, None)


# ---------------------------------------------------------------------------

def test_send_success_attaches_real_bytes(monkeypatch, tmp_path):
    sent = _install_fake_sdk(monkeypatch)
    f = tmp_path / "calc_package.pdf"
    f.write_bytes(b"%PDF-real-content")
    out = et.email_file.invoke({"to": "jane.doe@state.gov",
                                "file_path": str(f)})
    assert "Emailed calc_package.pdf to jane.doe@state.gov" in out
    assert "no-reply" in out
    assert sent["to"] == "jane.doe@state.gov"
    assert sent["attachments"] == [("calc_package.pdf", b"%PDF-real-content")]
    name, content = sent["attachments"][0]
    assert isinstance(content, bytes)
    # default subject/body mention the app and the filename
    assert "GeotechStaffEngineer" in sent["subject"]
    assert "calc_package.pdf" in sent["subject"]
    assert "calc_package.pdf" in sent["body"]
    assert "no-reply" in sent["body"]


def test_custom_subject_and_body_pass_through(monkeypatch, tmp_path):
    sent = _install_fake_sdk(monkeypatch)
    f = tmp_path / "memo.pdf"
    f.write_bytes(b"x")
    et.email_file.invoke({"to": "a@army.mil", "file_path": str(f),
                          "subject": "Site A memo", "body": "See attached."})
    assert sent["subject"] == "Site A memo"
    assert sent["body"] == "See attached."


@pytest.mark.parametrize("addr", [
    "someone@gmail.com", "x@contractor.example.org", "not-an-email",
    "", "gov@x.commercial",
])
def test_blocked_domain_rejected_client_side(monkeypatch, tmp_path, addr):
    sent = _install_fake_sdk(monkeypatch)
    f = tmp_path / "report.pdf"
    f.write_bytes(b"x")
    out = et.email_file.invoke({"to": addr, "file_path": str(f)})
    assert "Recipient not allowed" in out and "no email was sent" in out
    assert sent == {}                              # SDK never touched


@pytest.mark.parametrize("addr", ["a@state.gov", "b@navy.MIL", "c@dept.sbu"])
def test_allowed_suffixes_case_insensitive(monkeypatch, tmp_path, addr):
    sent = _install_fake_sdk(monkeypatch)
    f = tmp_path / "x.pdf"
    f.write_bytes(b"x")
    out = et.email_file.invoke({"to": addr, "file_path": str(f)})
    assert "Emailed" in out and sent["to"] == addr


def test_missing_file(monkeypatch):
    sent = _install_fake_sdk(monkeypatch)
    out = et.email_file.invoke({"to": "a@state.gov",
                                "file_path": "/nope/missing.pdf"})
    assert "Local file not found" in out and "no email was sent" in out
    assert sent == {}


def test_sdk_exception_becomes_readable_error(monkeypatch, tmp_path):
    _install_fake_sdk(monkeypatch,
                      send_exc=RuntimeError("SMTP relay refused connection"))
    f = tmp_path / "x.pdf"
    f.write_bytes(b"x")
    out = et.email_file.invoke({"to": "a@state.gov", "file_path": str(f)})
    assert "Email error" in out and "SMTP relay refused connection" in out


def test_tools_if_available_with_sdk(monkeypatch):
    _install_fake_sdk(monkeypatch)
    tools, prompt = et.tools_if_available()
    assert [t.name for t in tools] == ["email_file"]
    assert "EMAIL" in prompt and "no-reply" in prompt
    assert ".gov" in prompt and ".mil" in prompt and ".sbu" in prompt
    assert et.available() is True


def test_tools_if_available_without_sdk(monkeypatch):
    _uninstall_sdk(monkeypatch)
    assert et.tools_if_available() == ([], "")
    assert et.available() is False


def test_tool_without_sdk_reports_not_raises(monkeypatch, tmp_path):
    """Even if injected somehow without the SDK, the tool degrades to text."""
    _uninstall_sdk(monkeypatch)
    f = tmp_path / "x.pdf"
    f.write_bytes(b"x")
    out = et.email_file.invoke({"to": "a@state.gov", "file_path": str(f)})
    assert "Email error" in out
