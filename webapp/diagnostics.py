"""Connection diagnostics — staged self-tests that print WHY the app can't talk
to its model, instead of failing silently or flashing an error.

Born from the first live Foundry deployment (2026-07-17): the engine banner
said everything was fine, questions returned "(no answer text)", and the real
error flashed for a split second. This module runs the same calls the chat
turn makes — resolution, plain request, streaming request, tool-calling
request — one stage at a time, catching and REPORTING every failure verbatim.

Streamlit-free on purpose: callable from the sidebar panel (webapp/app.py),
from a notebook, or from a terminal::

    python -c "from webapp.diagnostics import run_diagnostics, format_report; \\
               print(format_report(run_diagnostics()))"

Each check returns a dict: ``{"name", "status": "pass"|"fail"|"warn"|"skip",
"detail"}``. A ``fail`` carries the full exception text — that string is the
diagnosis.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

# ---------------------------------------------------------------------------
# Report shape
# ---------------------------------------------------------------------------

PASS, FAIL, WARN, SKIP = "pass", "fail", "warn", "skip"

_ICON = {PASS: "✅", FAIL: "❌", WARN: "⚠️", SKIP: "⏭️"}


def _check(name: str, status: str, detail: str) -> dict:
    return {"name": name, "status": status, "detail": detail}


def format_report(checks: List[dict]) -> str:
    """Plain-text rendering of a diagnostics run (one line per check + detail
    lines for anything that isn't a clean pass)."""
    lines = []
    for c in checks:
        lines.append(f"{_ICON.get(c['status'], '?')} {c['name']}: {c['status'].upper()}")
        if c["detail"] and c["status"] != PASS:
            for dl in str(c["detail"]).splitlines():
                lines.append(f"    {dl}")
        elif c["detail"]:
            lines.append(f"    {c['detail']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _versions_check() -> dict:
    import importlib.metadata as md
    import sys
    parts = [f"python {sys.version.split()[0]}"]
    for pkg in ("geotech-staff-engineer", "geotech-references", "streamlit",
                "langchain", "langchain-openai", "langchain-anthropic",
                "deepagents", "websockets"):
        try:
            parts.append(f"{pkg} {md.version(pkg)}")
        except Exception:
            parts.append(f"{pkg} NOT INSTALLED")
    return _check("versions", PASS, " | ".join(parts))


def _env_check() -> dict:
    """Which engine-relevant env vars are set (values masked)."""
    envs = ("ANTHROPIC_API_KEY",
            "GEOTECH_FOUNDRY_TOKEN", "FOUNDRY_TOKEN",
            "GEOTECH_FOUNDRY_HOST", "FOUNDRY_HOSTNAME", "FOUNDRY_URL",
            "GEOTECH_FOUNDRY_MODELS", "GEOTECH_WEBAPP_MAX_TOKENS",
            "GEOTECH_FOUNDRY_DISABLE_STREAMING", "GEOTECH_TRACE")
    parts = []
    for e in envs:
        v = os.environ.get(e)
        if v is None or not str(v).strip():
            parts.append(f"{e}=unset")
        elif e in ("GEOTECH_FOUNDRY_MODELS", "GEOTECH_WEBAPP_MAX_TOKENS",
                   "GEOTECH_FOUNDRY_DISABLE_STREAMING", "GEOTECH_TRACE"):
            parts.append(f"{e}={v}")          # not secrets — show them
        else:
            parts.append(f"{e}=set({len(str(v))} chars)")
    return _check("environment", PASS, " | ".join(parts))


def _resolution_check(model_id: Optional[str]) -> dict:
    from webapp import engine_config
    try:
        eng = engine_config.resolve_engine(model_id=model_id)
    except Exception as exc:  # resolve_engine "never raises", but belt+braces
        return _check("engine resolution", FAIL,
                      f"{type(exc).__name__}: {exc}")
    status = PASS if eng.ok else FAIL
    return _check("engine resolution", status,
                  f"source={eng.source} model={eng.model_name} — {eng.message}")


def _get_model(model_id: Optional[str]):
    from webapp import engine_config
    eng = engine_config.resolve_engine(model_id=model_id)
    return eng.model if eng.ok else None


def _invoke_check(model: Any) -> dict:
    try:
        r = model.invoke("Reply with exactly: OK")
    except Exception as exc:
        return _check("plain request (invoke)", FAIL,
                      f"{type(exc).__name__}: {exc}")
    content = getattr(r, "content", r)
    text = content if isinstance(content, str) else str(content)
    if not text.strip():
        fr = ""
        meta = getattr(r, "response_metadata", None) or {}
        if meta.get("finish_reason"):
            fr = f" finish_reason={meta['finish_reason']}"
        usage = getattr(r, "usage_metadata", None) or {}
        if usage:
            fr += f" usage={usage}"
        return _check(
            "plain request (invoke)", FAIL,
            "Request succeeded but the reply text is EMPTY." + fr +
            " — a reasoning model likely spent the whole completion budget "
            "on hidden reasoning before any output. Raise "
            "GEOTECH_WEBAPP_MAX_TOKENS (e.g. 32000) and rerun.")
    return _check("plain request (invoke)", PASS, f"reply: {text.strip()[:80]}")


def _stream_check(model: Any) -> dict:
    try:
        chunks = []
        for c in model.stream("Reply with exactly: OK"):
            content = getattr(c, "content", "")
            if isinstance(content, str):
                chunks.append(content)
            else:
                chunks.append(str(content))
        text = "".join(chunks)
    except Exception as exc:
        return _check(
            "streaming request", FAIL,
            f"{type(exc).__name__}: {exc}\n"
            "If the plain request PASSED, set "
            "GEOTECH_FOUNDRY_DISABLE_STREAMING=1 in the app file — the app "
            "will fall back to non-streaming requests.")
    if not text.strip():
        return _check("streaming request", WARN,
                      "Stream completed but produced no text (see the plain-"
                      "request check for the token-budget suspicion).")
    return _check("streaming request", PASS, f"streamed: {text.strip()[:80]}")


def _tool_check(model: Any) -> dict:
    """Bind one trivial tool and confirm the model can emit a tool call —
    the agent is tool-calling on every real question."""
    try:
        from langchain_core.tools import tool as _tool_dec

        @_tool_dec
        def add_numbers(a: float, b: float) -> float:
            """Add two numbers and return the sum."""
            return a + b

        bound = model.bind_tools([add_numbers])
        r = bound.invoke(
            "Use the add_numbers tool to compute 2 + 3. Call the tool.")
    except Exception as exc:
        return _check("tool-calling request", FAIL,
                      f"{type(exc).__name__}: {exc}")
    calls = getattr(r, "tool_calls", None) or []
    if calls:
        return _check("tool-calling request", PASS,
                      f"model emitted tool call: {calls[0].get('name')}")
    content = getattr(r, "content", "")
    return _check(
        "tool-calling request", WARN,
        "Model replied WITHOUT a tool call (reply: "
        f"{str(content).strip()[:80]!r}). The agent needs tool calls for real "
        "calculations; if this persists, the proxy/model may not support the "
        "OpenAI tools API.")


# ---------------------------------------------------------------------------
# The battery
# ---------------------------------------------------------------------------

def run_diagnostics(model_id: Optional[str] = None) -> List[dict]:
    """Run every stage against the CURRENTLY CONFIGURED engine and return the
    check list. Never raises. Live checks each make one tiny model call (a few
    tokens) — three calls total when everything passes."""
    checks = [_versions_check(), _env_check(), _resolution_check(model_id)]
    if checks[-1]["status"] != PASS:
        checks.append(_check("plain request (invoke)", SKIP,
                             "engine did not resolve"))
        checks.append(_check("streaming request", SKIP, ""))
        checks.append(_check("tool-calling request", SKIP, ""))
        return checks
    model = _get_model(model_id)
    if model is None:
        checks.append(_check("plain request (invoke)", SKIP,
                             "engine did not resolve"))
        return checks
    checks.append(_invoke_check(model))
    checks.append(_stream_check(model))
    checks.append(_tool_check(model))
    return checks
