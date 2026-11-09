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
from pathlib import Path
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
    # numpy: the app runs on 1.26 and 2.x (INF-53); Foundry's transforms
    # repos hold 1.26.4, and scipy must then be 1.17 or older.
    for pkg in ("geotech-staff-engineer", "geotech-references", "planlens",
                "numpy", "streamlit", "langchain", "langchain-openai",
                "langchain-anthropic", "deepagents", "websockets"):
        try:
            parts.append(f"{pkg} {md.version(pkg)}")
        except Exception:
            parts.append(f"{pkg} NOT INSTALLED")
    return _check("versions", PASS, " | ".join(parts))


def _drift_check() -> dict:
    """Agent-stack versions vs the tested range (webapp.version_guard).

    WARN (not FAIL) — the app may still work, but untested versions are the
    prime suspect for weird agent behavior (see the 2026-08 deepagents-drift
    outage), so surface it prominently in every diagnostics run."""
    try:
        from webapp import version_guard
        warnings = version_guard.check_versions()
    except Exception as exc:
        return _check("version drift", SKIP,
                      f"guard unavailable: {type(exc).__name__}: {exc}")
    if not warnings:
        return _check("version drift", PASS,
                      "all agent-stack packages within the tested range")
    return _check("version drift", WARN, " || ".join(warnings))


#: The optional packages behind the analysis tools: (package, the tools that
#: need it, the module and function its tools ask whether it imports).
OPTIONAL_PACKAGES = (
    ("PyNiteFEA", "pynite", "pynite_agent.pynite_utils", "pynite_import_error"),
    ("gstools", "gstools", "gstools_agent.gstools_utils", "gstools_import_error"),
    ("SALib", "salib", "salib_agent.salib_utils", "salib_import_error"),
    ("pystra", "pystra", "pystra_agent.pystra_utils", "pystra_import_error"),
    ("liquepy", "liquepy, liquefaction", "liquepy_agent.liquepy_utils",
     "liquepy_import_error"),
    ("pystrata", "pystrata", "pystrata_agent.pystrata_utils",
     "pystrata_import_error"),
    ("openseespy", "opensees", "opensees_agent.opensees_utils",
     "opensees_import_error"),
    ("eqsig", "seismic_signals", "seismic_signals_agent.signal_utils",
     "eqsig_import_error"),
    ("pygef", "subsurface", "subsurface_characterization.formats.gef",
     "pygef_import_error"),
    ("python-ags4", "subsurface", "subsurface_characterization.formats.ags4",
     "ags4_import_error"),
    ("pydiggs", "subsurface",
     "subsurface_characterization.formats.diggs_validation",
     "pydiggs_import_error"),
)


def _optional_packages_check() -> dict:
    """The optional packages behind the analysis tools, by the check the
    tools themselves run: which import, which are not installed, and for one
    installed that will not import, its import's own error (BACKLOG INF-54:
    pandas 3 failing on numpy 1.26 once read as "SALib is not installed").

    WARN when an installed package cannot be imported. A package that is
    not installed is listed and not warned about: some hosts leave packages
    out on purpose (PyNiteFEA cannot install on Foundry's numpy 1.26)."""
    import importlib
    import importlib.metadata as md
    from geotech_common.optional_packages import CANNOT_IMPORT
    imports, missing, broken = [], [], []
    for package, tools, module, func in OPTIONAL_PACKAGES:
        try:
            error = getattr(importlib.import_module(module), func)()
        except Exception as exc:
            error = (f"{package}: the check itself failed: "
                     f"{type(exc).__name__}: {exc}")
        if error is None:
            try:
                imports.append(f"{package} {md.version(package)}")
            except Exception:
                imports.append(package)
        elif CANNOT_IMPORT in error or "the check itself failed" in error:
            broken.append(f"{error} (tools: {tools})")
        else:
            missing.append(f"{package} ({tools})")
    parts = [f"import: {', '.join(imports) or 'none'}"]
    if missing:
        parts.append(f"not installed: {', '.join(missing)}")
    summary = " | ".join(parts)
    if broken:
        return _check("optional packages", WARN, "\n".join(broken + [summary]))
    return _check("optional packages", PASS, summary)


def _env_check() -> dict:
    """Which engine-relevant env vars are set (values masked).

    On a Foundry deployment the Anthropic key is NOT part of the engine
    surface — it is neither read nor mentioned (enclave-IT requirement), so it
    is omitted from the report there.
    """
    from webapp import engine_config
    envs = ("GEOTECH_FOUNDRY_TOKEN", "FOUNDRY_TOKEN",
            "GEOTECH_FOUNDRY_HOST", "ANTHROPIC_BASE_URL", "OPENAI_BASE_URL",
            *engine_config.FOUNDRY_PROXY_URL_ENVS,
            "FOUNDRY_HOSTNAME", "FOUNDRY_URL",
            *engine_config.FOUNDRY_CA_ENVS,
            "GEOTECH_FOUNDRY_MODELS", "GEOTECH_HELPER_MODELS",
            "GEOTECH_WEBAPP_MAX_TOKENS",
            "GEOTECH_FOUNDRY_DISABLE_STREAMING",
            "GEOTECH_FOUNDRY_PROMPT_CACHE",
            engine_config.FOUNDRY_CLAUDE_SDK_ENV, "GEOTECH_TRACE")
    if engine_config.is_tinyapps_deployment():
        # The Tiny Apps engine settings, named as CfA names them; the key
        # (and the per-tester key mapping) itself only ever as a length.
        envs = ("PROMPTER_URL", "PROMPTER_MODEL", "PROMPTER_API_KEY",
                "PROMPTER_CA_BUNDLE", "PROMPTER_KEYS",
                "PROMPTER_SHARED_KEY_FALLBACK",
                "GEOTECH_PROMPTER_DISABLE_STREAMING",
                "GEOTECH_WEBAPP_MAX_TOKENS", "GEOTECH_TRACE")
    elif not engine_config.is_foundry_deployment():
        envs = ("ANTHROPIC_API_KEY",) + envs
    parts = []
    for e in envs:
        v = os.environ.get(e)
        if v is None or not str(v).strip():
            parts.append(f"{e}=unset")
        elif e in ("GEOTECH_FOUNDRY_MODELS", "GEOTECH_HELPER_MODELS",
                   "GEOTECH_WEBAPP_MAX_TOKENS",
                   "GEOTECH_FOUNDRY_DISABLE_STREAMING",
                   "GEOTECH_FOUNDRY_PROMPT_CACHE",
                   engine_config.FOUNDRY_CLAUDE_SDK_ENV, "GEOTECH_TRACE",
                   "PROMPTER_URL", "PROMPTER_MODEL",
                   "PROMPTER_SHARED_KEY_FALLBACK",
                   "GEOTECH_PROMPTER_DISABLE_STREAMING"):
            parts.append(f"{e}={v}")          # not secrets — show them
        else:
            parts.append(f"{e}=set({len(str(v))} chars)")
    return _check("environment", PASS, " | ".join(parts))


def _tls_check() -> dict:
    """Foundry: which certificates the model proxy's TLS is checked against.
    Says so plainly when no CA variable names a file and the SDKs fall back
    to certifi's public list (INF-44). Names variables, never paths."""
    from webapp import engine_config
    name = "TLS certificates"
    source = engine_config.foundry_ca_source()
    if source:
        return _check(name, PASS, f"the model proxy's certificate is "
                      f"checked against the CA file {source[0]} names")
    envs = engine_config.FOUNDRY_CA_ENVS
    broken = [e for e in envs if str(os.environ.get(e) or "").strip()]
    detail = (f"no CA variable names a file ({', '.join(envs)}), so the "
              "model proxy's certificate is checked against certifi, the "
              "public CA list that comes with Python's HTTP packages. If "
              "requests fail with a certificate error, set one of them to "
              "the CA file the container provides.")
    if broken:
        return _check(name, WARN, f"Set to a file that does not exist: "
                      f"{', '.join(broken)}. {detail[0].upper()}{detail[1:]}")
    return _check(name, PASS, detail)


def _conversation_storage_check() -> dict:
    """Foundry: where conversations are kept (INF-38). The conversation
    dataset when set, read once (one listing, about 3-4 s; nothing is
    written, the first turn's save shows that); otherwise why it is off,
    plainly, with the fix (``fds.not_kept_reason``: the sidebar's
    "Conversations are NOT kept" block shows the same text, and a failed
    read here reaches it through the store's shared record, INF-65). A new
    dataset with nothing committed yet passes as empty."""
    from webapp import foundry_dataset_store as fds
    name = "conversation storage"
    if not fds.dataset_alias():
        return _check(name, SKIP, fds.not_kept_reason())
    if fds.unconfigured_reason():
        return _check(name, FAIL, fds.not_kept_reason())
    fm = None
    try:
        fm = fds.DatasetFileManager()
        state = fm.probe()
    except Exception as exc:                         # noqa: BLE001
        reason = fds.not_kept_reason(fm) if fm is not None else None
        return _check(name, FAIL, reason or (
            f"the conversation dataset could not be read: "
            f"{type(exc).__name__}: {exc}"))
    return _check(name, PASS, f"Foundry storage: {fm.describe()}; {state}. "
                  "Conversations are saved there after each turn.")


def _sharepoint_started() -> bool:
    try:
        from webapp import graph_sharepoint
        return graph_sharepoint.started()
    except Exception:                                # noqa: BLE001
        return False


def _sharepoint_check() -> dict:
    """The SharePoint app registration, tested step by step (TA-34): which
    step failed and what the host's admin does about it. Run only when a
    registration setting is filled in; makes a sign-in request, one Graph
    request per site and library, and writes and deletes a small probe file
    (so a read-only grant shows here, not at the first save)."""
    from webapp import graph_sharepoint as gs
    name = "SharePoint connection"
    try:
        results = gs.check_connection()
    except Exception as exc:                         # noqa: BLE001
        return _check(name, FAIL, f"{type(exc).__name__}: {exc}")
    ok = bool(results) and all(r["ok"] for r in results)
    return _check(name, PASS if ok else FAIL, gs.format_check(results))


def sharepoint_download(checks: List[dict], who: str) -> Optional[str]:
    """TA-35: the SharePoint report to offer as a download in the sidebar
    when the connection test failed (sign-in refused, site unreachable, ...:
    then the report may not have reached the SharePoint log): the test's
    report, whether it reached the log, and the SharePoint errors met in
    the conversations of ``who`` (the signed-in person's key on a shared
    host, ``""`` on a single-user one) -- never another person's entries,
    which carry their folder and file names. ``None`` when the test passed
    or did not run. Never raises; carries no secret (the log's entries are
    scrubbed)."""
    try:
        sp = next((c for c in checks
                   if c.get("name") == "SharePoint connection"), None)
        if sp is None or sp.get("status") == PASS:
            return None
        from webapp import sharepoint_log
        sharepoint_log.flush(15.0)
        kept = sharepoint_log.last_write is True
        where = ("it was also appended to " + sharepoint_log.log_path()
                 if kept else "it could not be written to SharePoint"
                 + (f" ({sharepoint_log.last_write})"
                    if isinstance(sharepoint_log.last_write, str) else ""))
        return ("GeotechStaffEngineer SharePoint report\n"
                f"Status: {str(sp.get('status')).upper()}; {where}.\n\n"
                + sharepoint_log.scrub(str(sp.get("detail") or ""))
                + "\n\nSharePoint errors met in your conversations (UTC "
                "time, step, message, path):\n"
                + (sharepoint_log.recent_text(who) or "(none)\n"))
    except Exception:                                # noqa: BLE001
        return None


def _prompter_key_check(identity: Any = None) -> dict:
    """Tiny Apps: which Prompter key SOURCE serves this caller — per-tester
    secret, the PROMPTER-KEYS mapping or the shared key — and the per-tester
    secret's NAME. Never a key (``tinyapps_engine.key_source_report``)."""
    name = "Prompter key source"
    try:
        from webapp import tinyapps_engine
        ok, detail = tinyapps_engine.key_source_report(identity)
    except Exception as exc:                         # noqa: BLE001
        return _check(name, WARN, f"{type(exc).__name__}: {exc}")
    return _check(name, PASS if ok else WARN, detail)


def _engine_kwargs(model_id: Optional[str], identity: Any) -> dict:
    """``resolve_engine`` arguments: the caller only when known (older
    callers and test fakes take ``model_id`` alone)."""
    kw = {"model_id": model_id}
    if identity is not None:
        kw["identity"] = identity
    return kw


def _resolution_check(model_id: Optional[str], identity: Any = None) -> dict:
    from webapp import engine_config
    try:
        eng = engine_config.resolve_engine(**_engine_kwargs(model_id, identity))
    except Exception as exc:  # resolve_engine "never raises", but belt+braces
        return _check("engine resolution", FAIL,
                      f"{type(exc).__name__}: {exc}")
    status = PASS if eng.ok else FAIL
    route = getattr(eng, "route", "")
    return _check("engine resolution", status,
                  f"source={eng.source} model={eng.model_name} — {eng.message}"
                  + (f" | {route}" if route else ""))


def _get_model(model_id: Optional[str], identity: Any = None):
    from webapp import engine_config
    eng = engine_config.resolve_engine(**_engine_kwargs(model_id, identity))
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

def _upload_probe_check() -> dict:
    """PUT and POST a tiny body at the local Streamlit upload endpoint.

    Discriminates the browser-upload 403 (open since 5.10.0 behind the
    Databricks driver proxy): this probe goes to LOCALHOST — no proxy in the
    path. If localhost processes the PUT (any status except 403), the proxy
    is the blocker and the websocket uploader / SharePoint-fetch paths are
    the answer; a 403 HERE would mean Streamlit itself is still enforcing
    something (XSRF/origin) despite the launch flags.
    """
    try:
        import requests
        try:
            from streamlit import config as _st_config
            port = int(_st_config.get_option("server.port") or 8501)
        except Exception:
            port = int(os.environ.get("GEOTECH_APP_PORT", "8501"))
        results = []
        for method in ("PUT", "POST"):
            r = requests.request(
                method, f"http://localhost:{port}/_stcore/upload_file/diag/diag",
                data=b"x", timeout=5)
            results.append(f"{method} -> {r.status_code}")
        detail = "; ".join(results) + " (localhost, no proxy in path)"
        from webapp import engine_config
        if engine_config.is_foundry_deployment():
            # Measured 2026-10-20 (INF-25/INF-39): PUT 403 / POST 405 here,
            # and uploads from the browser worked in the published app.
            return _check("upload endpoint (localhost probe)", PASS,
                          detail + " — expected on Foundry: this probe "
                          "carries no browser session, so Streamlit refuses "
                          "it. Uploading from the browser works in the "
                          "published app (tested 2026-10-20).")
        if any(s.endswith("403") for s in results):
            return _check("upload endpoint (localhost probe)", WARN,
                          detail + " — 403 WITHOUT the proxy: Streamlit-side "
                          "block (XSRF/origin), not the proxy")
        return _check("upload endpoint (localhost probe)", PASS,
                      detail + " — methods reach Streamlit; a 403 seen only "
                      "through the browser therefore implicates the proxy")
    except Exception as exc:                     # noqa: BLE001
        return _check("upload endpoint (localhost probe)", SKIP,
                      f"probe not possible here: {type(exc).__name__}: {exc}")


#: Names that moved to the geotech page's own file with the reference-PDF
#: check (INF-31 step 7), still answered here for the callers that import
#: them from this module.
_MOVED_TO_GEOTECH_PAGE = {
    "REFERENCE_DOCS_ENV": "REFERENCE_DOCS_ENV",
    "reference_pdf_names": "reference_pdf_names",
    "_UNSET_MESSAGE": "_UNSET_MESSAGE",
    "_UNSET_MESSAGE_FOUNDRY": "_UNSET_MESSAGE_FOUNDRY",
    "_reference_docs_check": "reference_docs_check",
}


def __getattr__(name):
    moved = _MOVED_TO_GEOTECH_PAGE.get(name)
    if moved is not None:
        import importlib
        return getattr(importlib.import_module("webapp.geotech_page"), moved)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _page_checks(page_checks) -> List[dict]:
    """The page's own diagnostics lines (its ``diagnostics_checks`` hook;
    ``page_checks`` a hook, or None for the page this run serves). Never
    raises."""
    try:
        from webapp.page_profile import resolve
        if page_checks is None:
            from webapp import profiles
            page_checks = profiles.current().diagnostics_checks
        fn = resolve(page_checks)
        return list(fn()) if fn is not None else []
    except Exception as exc:                            # noqa: BLE001
        return [_check("page checks", SKIP,
                       f"not available: {type(exc).__name__}: {exc}")]


def _vision_check(model: Any) -> dict:
    """What model really answers, and what size of image it really takes.

    A deployment name is an alias (``tinyapp-gpt-medium`` was GPT-5.1, a
    tile model that ignores ``detail="original"``), so this reports the
    measured profile the vision tools will use — the same cached probe, so
    it costs its four small calls once per process at most.
    """
    name = "vision model"
    try:
        from funhouse_agent import vision_probe
    except ImportError as exc:                   # pragma: no cover
        return _check(name, SKIP, f"not available: {exc}")
    if not vision_probe.enabled():
        return _check(name, SKIP, f"probe switched off ({vision_probe.PROBE_ENV}=0)")
    prof = vision_probe.profile_for(model)
    if prof is None or not prof.general:
        return _check(name, WARN, prof.summary() if prof else "not measured")
    return _check(name, PASS, prof.summary())


def _helper_models(model_id: Optional[str], identity: Any,
                   overrides: Optional[dict]):
    """``(the "agent models" check or None, the helpers)``: the model each
    agent runs on (INF-41). No check on a host that gives no helper a model
    of its own."""
    from webapp import core, helper_models
    try:
        helpers = helper_models.resolve(model_id, overrides, identity)
    except Exception as exc:                         # noqa: BLE001
        return (_check("agent models", WARN, f"{type(exc).__name__}: {exc}"),
                [])
    if not helpers:
        return None, []
    main = core.model_label(model_id) or str(model_id or "")
    source = helper_models.main_source(model_id)
    if main != model_id and model_id:
        main = f"{main} ({model_id}; {source})"
    else:
        main = f"{main} ({source})"
    status = WARN if any(h.note for h in helpers) else PASS
    return (_check("agent models", status,
                   helper_models.summary(main, helpers)), helpers)


def _helper_request_checks(helpers) -> List[dict]:
    """A plain and a tool-calling request on each helper model that is not
    the main one (two tiny calls each)."""
    checks = []
    for h in helpers:
        if not h.own_model:
            continue
        for check in (_invoke_check(h.model), _tool_check(h.model)):
            check["name"] = f"{check['name']} ({h.name} helper)"
            if check["status"] != PASS:
                check["detail"] += (f" | {h.name} helper: {h.label}"
                                    + (f" | {h.route}" if h.route else ""))
            checks.append(check)
    return checks


def run_diagnostics(model_id: Optional[str] = None,
                    identity: Any = None,
                    helper_models: Optional[dict] = None,
                    page_checks=None) -> List[dict]:
    """Run every stage against the CURRENTLY CONFIGURED engine and return the
    check list. Never raises. Live checks each make one tiny model call (a few
    tokens) — three calls when everything passes, plus the vision probe's four
    the first time it runs in this process. ``identity`` is the signed-in
    caller: on Tiny Apps the engine is built with THEIR key, and the report
    says which key source that is (never the key). ``helper_models`` are the
    sidebar's picks for helpers on a model of their own; the report names
    the model each agent runs on and tries each helper model with two more
    tiny requests (:mod:`webapp.helper_models`). ``page_checks`` is the
    page's own lines (its ``diagnostics_checks`` hook, :mod:`webapp.
    page_profile`; the geotech page's says whether the reference PDFs can
    be opened); None takes the page this run serves."""
    from webapp import engine_config
    checks = [_versions_check(), _drift_check(), _optional_packages_check(),
              _env_check(), _upload_probe_check()] + _page_checks(page_checks)
    if engine_config.is_tinyapps_deployment():
        checks.append(_prompter_key_check(identity))
    if _sharepoint_started():
        checks.append(_sharepoint_check())
    if engine_config.is_foundry_deployment():
        checks.append(_tls_check())
        checks.append(_conversation_storage_check())
    resolved = _resolution_check(model_id, identity)
    checks.append(resolved)
    agents, helpers = _helper_models(model_id, identity, helper_models)
    if agents is not None:
        checks.append(agents)
    if resolved["status"] != PASS:
        checks.append(_check("plain request (invoke)", SKIP,
                             "engine did not resolve"))
        checks.append(_check("streaming request", SKIP, ""))
        checks.append(_check("tool-calling request", SKIP, ""))
        return checks
    model = _get_model(model_id, identity)
    if model is None:
        checks.append(_check("plain request (invoke)", SKIP,
                             "engine did not resolve"))
        return checks
    checks.append(_invoke_check(model))
    checks.append(_stream_check(model))
    checks.append(_tool_check(model))
    checks.append(_vision_check(model))
    checks.extend(_helper_request_checks(helpers))
    return checks
