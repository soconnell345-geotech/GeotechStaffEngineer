"""Agent-facing email tool for the web app (Funhouse SDK backed).

One LangChain tool that lets the AGENT email a produced file (calc package,
report PDF, plot) to a colleague during a conversation:

    email_file(to, file_path, subject="", body="")

It is injected into the deep agent by ``webapp.core.build_agent`` via
``build_deep_agent(extra_tools=...)`` ONLY when the Funhouse email module is
importable (``tools_if_available()``, mirroring
``sharepoint_tools.tools_if_configured``), so non-Funhouse deployments carry
zero extra tool surface.

Delivery facts the agent (and user) should know:

* Mail goes out from the SHARED Funhouse no-reply mailbox — recipients cannot
  reply to it, and it is not the user's own address.
* Recipients are restricted to .gov / .mil / .sbu domains. We enforce that
  client-side BEFORE touching the SDK (cheaper and a clearer message than the
  SDK-side guard error); the SDK enforces it again regardless.
* An attachment-extension blocklist (e.g. .exe/.js) exists SDK-side.

Every tool returns a plain string and NEVER raises: errors come back as
readable text so the agent can report/retry rather than crash the turn.
"""

from __future__ import annotations

import os

from langchain_core.tools import tool

#: Recipient domains the Funhouse relay accepts (SDK default allowlist).
ALLOWED_SUFFIXES = (".gov", ".mil", ".sbu")

#: Env carrying the app user's own email, captured notebook-side at launch
#: (the sanctioned resolution is ``fh_config.get("session.user_name")`` — the
#: SDK's email example does exactly this; the launcher threads it through so
#: "email it to me" works in the app subprocess, where the session is absent).
USER_EMAIL_ENV = "GEOTECH_USER_EMAIL"


def _own_address() -> str:
    """The app user's own email: launch-captured env first, then a best-effort
    live session read (works when serving in-process on a driver)."""
    addr = os.environ.get(USER_EMAIL_ENV, "").strip()
    if addr:
        return addr
    try:
        from funhouse.config.funhouse_config import FunhouseConfig
        return str(FunhouseConfig.get_instance().get(
            "session.user_name", default="") or "").strip()
    except Exception:
        return ""


def available() -> bool:
    """True when the Funhouse email SDK is importable."""
    try:
        import funhouse.services.email  # noqa: F401
        return True
    except Exception:
        return False


def _domain_allowed(addr: str) -> bool:
    if "@" not in addr:
        return False
    domain = addr.rsplit("@", 1)[1].strip().lower()
    return bool(domain) and domain.endswith(ALLOWED_SUFFIXES)


@tool
def email_file(to: str, file_path: str, subject: str = "",
               body: str = "") -> str:
    """Email a local file (e.g. a calc package or report from the working
    folder) to a colleague as an attachment.

    to: the recipient's email address — must be a .gov, .mil, or .sbu
    address (the mail relay accepts nothing else). Pass "me" (or leave
    empty) to send to the app user's own address, which the app knows from
    the launch session — do NOT guess their address.
    file_path: the local file to attach (as returned by save_file /
    calc-package tools).
    subject, body: optional; sensible defaults mention GeotechStaffEngineer
    and the filename.

    The message is sent from a shared no-reply mailbox (not the user's own
    address), so tell the user who it went to.
    """
    addr = (to or "").strip()
    if not addr or addr.lower() in ("me", "myself", "self", "my email"):
        addr = _own_address()
        if not addr:
            return ("I don't know the user's own email address in this "
                    "deployment (no session identity was captured at launch) "
                    "— ask the user for their .gov/.mil/.sbu address "
                    "(no email was sent).")
    if not _domain_allowed(addr):
        return (f"Recipient not allowed: '{addr}'. The Funhouse mail relay "
                "only delivers to .gov, .mil, or .sbu addresses — check the "
                "address with the user (no email was sent).")
    if not os.path.isfile(file_path):
        return (f"Local file not found: {file_path} — check the path with "
                "list_files (no email was sent).")
    try:
        with open(file_path, "rb") as fh:
            data = fh.read()
    except OSError as exc:
        return (f"Could not read {file_path}: {type(exc).__name__}: {exc} "
                "(no email was sent).")
    filename = os.path.basename(file_path)
    subject = (subject or "").strip() or f"[GeotechStaffEngineer] {filename}"
    body = (body or "").strip() or (
        f"Attached: {filename} ({len(data):,} bytes), produced by the "
        "GeotechStaffEngineer web app.\n\nThis message was sent from a "
        "shared no-reply mailbox — please do not reply to it.")
    try:
        from funhouse.services.email import FunhouseEmail
        mailer = FunhouseEmail()
        mailer.send_email(to=addr, subject=subject, body=body,
                          attachments=[(filename, data)])
        return (f"Emailed {filename} to {addr} from the shared Funhouse "
                f"no-reply mailbox ({len(data):,} bytes attached, subject: "
                f"'{subject}').")
    except Exception as exc:
        return f"Email error: {type(exc).__name__}: {exc}"


#: Prompt block injected alongside the tool (build_agent), so the agent knows
#: the capability exists and its ground rules.
EMAIL_PROMPT = (
    "EMAIL: You have an email_file tool that emails a local file (a calc "
    "package, report, or plot you produced) to a colleague as an attachment. "
    "Use it when the user asks to send/email a deliverable to someone. "
    "Recipients must be .gov, .mil, or .sbu addresses — anything else is "
    "rejected. Mail is sent from a shared Funhouse no-reply mailbox (NOT the "
    "user's own address), so recipients cannot reply; mention that when "
    "confirming a send. When the user says 'email it to me', call the tool "
    "with to='me' — the app resolves their own address from the launch "
    "session. Never invent recipient addresses — only use to='me' or "
    "addresses the user supplied.")


def tools_if_available() -> tuple:
    """``(tools, prompt)`` when the Funhouse email SDK is importable, else
    ``([], "")``."""
    if not available():
        return [], ""
    return [email_file], EMAIL_PROMPT


__all__ = ["email_file", "tools_if_available", "available", "EMAIL_PROMPT",
           "ALLOWED_SUFFIXES"]
