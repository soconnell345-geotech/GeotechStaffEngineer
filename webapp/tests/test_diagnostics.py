"""Offline tests for webapp/diagnostics.py — the connection self-test battery.

Everything runs against fake chat models; no network, no API key. The battery
must (a) report the failure TEXT verbatim on any stage, (b) flag the
empty-reply case (reasoning-token burn) as a failure with actionable advice,
and (c) pass cleanly against a healthy model.
"""

import pytest

import webapp.diagnostics as diag
import webapp.engine_config as engine_config
from webapp.engine_config import EngineResolution


class _Reply:
    def __init__(self, content="OK", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.response_metadata = {}
        self.usage_metadata = {}


class _GoodModel:
    def invoke(self, _msg):
        return _Reply("OK")

    def stream(self, _msg):
        yield _Reply("O")
        yield _Reply("K")

    def bind_tools(self, _tools):
        outer = self

        class _Bound:
            def invoke(self, _msg):
                return _Reply("", tool_calls=[{"name": "add_numbers",
                                               "args": {"a": 2, "b": 3}}])
        return _Bound()


class _EmptyModel(_GoodModel):
    """Succeeds but returns no visible text — the reasoning-burn symptom."""
    def invoke(self, _msg):
        r = _Reply("")
        r.response_metadata = {"finish_reason": "length"}
        return r


class _BoomModel(_GoodModel):
    def invoke(self, _msg):
        raise RuntimeError("max_tokens is not supported with this model")

    def stream(self, _msg):
        raise RuntimeError("streaming is not permitted for this model")
        yield  # pragma: no cover


def _patch_engine(monkeypatch, model):
    monkeypatch.setattr(
        engine_config, "resolve_engine",
        lambda model_id=None: EngineResolution(
            model, "foundry", "ri.fake", "Using the Foundry OpenAI proxy."))


def _by_name(checks, name):
    return next(c for c in checks if c["name"] == name)


def test_all_pass_on_healthy_model(monkeypatch):
    _patch_engine(monkeypatch, _GoodModel())
    checks = diag.run_diagnostics("ri.fake")
    for name in ("engine resolution", "plain request (invoke)",
                 "streaming request", "tool-calling request"):
        assert _by_name(checks, name)["status"] == diag.PASS, checks


def test_empty_reply_is_flagged_with_advice(monkeypatch):
    _patch_engine(monkeypatch, _EmptyModel())
    checks = diag.run_diagnostics("ri.fake")
    c = _by_name(checks, "plain request (invoke)")
    assert c["status"] == diag.FAIL
    assert "EMPTY" in c["detail"]
    assert "GEOTECH_WEBAPP_MAX_TOKENS" in c["detail"]
    assert "finish_reason=length" in c["detail"]


def test_errors_reported_verbatim(monkeypatch):
    _patch_engine(monkeypatch, _BoomModel())
    checks = diag.run_diagnostics("ri.fake")
    inv = _by_name(checks, "plain request (invoke)")
    assert inv["status"] == diag.FAIL
    assert "max_tokens is not supported" in inv["detail"]
    str_c = _by_name(checks, "streaming request")
    assert str_c["status"] == diag.FAIL
    assert "streaming is not permitted" in str_c["detail"]
    assert "GEOTECH_FOUNDRY_DISABLE_STREAMING" in str_c["detail"]


def test_unresolved_engine_skips_live_checks(monkeypatch):
    monkeypatch.setattr(
        engine_config, "resolve_engine",
        lambda model_id=None: EngineResolution(
            None, "error", "ri.fake", "token env var(s) are not set"))
    checks = diag.run_diagnostics("ri.fake")
    assert _by_name(checks, "engine resolution")["status"] == diag.FAIL
    assert "token" in _by_name(checks, "engine resolution")["detail"]
    assert _by_name(checks, "plain request (invoke)")["status"] == diag.SKIP


def test_tls_line_names_the_ca_variable_or_says_certifi(monkeypatch,
                                                       tmp_path):
    for env in engine_config.FOUNDRY_CA_ENVS:
        monkeypatch.delenv(env, raising=False)
    c = diag._tls_check()
    assert c["status"] == diag.PASS
    assert "no CA variable names a file" in c["detail"]
    assert "against certifi" in c["detail"]
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "missing.pem"))
    c = diag._tls_check()
    assert c["status"] == diag.WARN
    assert "does not exist: SSL_CERT_FILE" in c["detail"]
    assert str(tmp_path) not in c["detail"]          # variables, not paths
    ca = tmp_path / "ca.pem"
    ca.write_text("x")
    monkeypatch.setenv("MAESTRO_CERT_PEM_PATH", str(ca))
    c = diag._tls_check()
    assert c["status"] == diag.PASS
    assert "MAESTRO_CERT_PEM_PATH" in c["detail"]
    assert "against certifi" not in c["detail"]
    assert str(ca) not in c["detail"]


def test_the_tls_line_is_foundrys_only(monkeypatch):
    _patch_engine(monkeypatch, _GoodModel())
    monkeypatch.delenv("GEOTECH_DEPLOYMENT", raising=False)
    names = [c["name"] for c in diag.run_diagnostics("ri.fake")]
    assert "TLS certificates" not in names
    monkeypatch.setenv("GEOTECH_DEPLOYMENT", "foundry")
    names = [c["name"] for c in diag.run_diagnostics("ri.fake")]
    assert names.index("TLS certificates") < names.index("engine resolution")


def test_format_report_readable():
    checks = [
        {"name": "a", "status": diag.PASS, "detail": "fine"},
        {"name": "b", "status": diag.FAIL, "detail": "line1\nline2"},
    ]
    text = diag.format_report(checks)
    assert "a: PASS" in text and "b: FAIL" in text
    assert "line1" in text and "line2" in text


def test_versions_and_env_never_raise(monkeypatch):
    monkeypatch.delenv("GEOTECH_FOUNDRY_MODELS", raising=False)
    assert diag._versions_check()["status"] == diag.PASS
    assert "unset" in diag._env_check()["detail"]


def test_foundry_openai_uses_max_completion_tokens(monkeypatch):
    """The Foundry OpenAI path must send max_completion_tokens (GPT-5/o-series
    reject max_tokens, and a RID gives no model-name hint to translate)."""
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("FOUNDRY_TOKEN", "fake-token")
    monkeypatch.setenv("FOUNDRY_HOSTNAME", "stack.example.palantirfoundry.com")
    monkeypatch.delenv("GEOTECH_FOUNDRY_DISABLE_STREAMING", raising=False)
    eng = engine_config.resolve_engine(model_id="ri.language-model-service..language-model.gpt-5-1")
    assert eng.ok, eng.message
    payload = eng.model._get_request_payload([("user", "hi")])
    assert payload.get("max_completion_tokens"), payload
    assert "max_tokens" not in payload, payload


def test_foundry_disable_streaming_env(monkeypatch):
    pytest.importorskip("langchain_openai")
    monkeypatch.setenv("FOUNDRY_TOKEN", "fake-token")
    monkeypatch.setenv("FOUNDRY_HOSTNAME", "stack.example.palantirfoundry.com")
    monkeypatch.setenv("GEOTECH_FOUNDRY_DISABLE_STREAMING", "1")
    eng = engine_config.resolve_engine(model_id="ri.some..language-model.gpt-5-1")
    assert eng.ok and eng.model.disable_streaming is True


# ---------------------------------------------------------------------------
# reference PDFs (chart read-off) — no model call
# ---------------------------------------------------------------------------

_NEEDED = {"A Doc.pdf": ["dm7_1"], "b.pdf": ["gec_6", "worked examples"],
           "C.pdf": ["gec_7"]}


def test_reference_docs_partial_folder_names_the_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(diag, "reference_pdf_names", lambda: _NEEDED)
    (tmp_path / "A Doc.pdf").write_bytes(b"%PDF")
    (tmp_path / "c.pdf").write_bytes(b"%PDF")          # wrong capitals
    monkeypatch.setenv("GEOTECH_REFERENCES_DOCS", str(tmp_path))
    c = diag._reference_docs_check()
    assert c["status"] == diag.WARN
    assert "1 of 3" in c["detail"]
    assert "- b.pdf  (gec_6, worked examples)" in c["detail"]
    assert "capital letters" in c["detail"] and "'c.pdf'" in c["detail"]


def test_reference_docs_all_found_passes(monkeypatch, tmp_path):
    monkeypatch.setattr(diag, "reference_pdf_names", lambda: _NEEDED)
    for n in _NEEDED:
        (tmp_path / n).write_bytes(b"%PDF")
    monkeypatch.setenv("GEOTECH_REFERENCES_DOCS", str(tmp_path))
    c = diag._reference_docs_check()
    assert c["status"] == diag.PASS and "all 3" in c["detail"]


def test_reference_docs_empty_or_absent_folder_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(diag, "reference_pdf_names", lambda: _NEEDED)
    monkeypatch.setenv("GEOTECH_REFERENCES_DOCS", str(tmp_path))
    assert diag._reference_docs_check()["status"] == diag.FAIL   # none found
    monkeypatch.setenv("GEOTECH_REFERENCES_DOCS", str(tmp_path / "nope"))
    c = diag._reference_docs_check()
    assert c["status"] == diag.FAIL and "does not exist" in c["detail"]


def test_reference_docs_unset_on_an_install_warns(monkeypatch, tmp_path):
    """A wheel install has no repo docs/ to fall back on."""
    from geotech_references import _figures_db
    monkeypatch.setattr(diag, "reference_pdf_names", lambda: _NEEDED)
    monkeypatch.delenv("GEOTECH_REFERENCES_DOCS", raising=False)
    monkeypatch.setattr(_figures_db, "_REPO_ROOT", tmp_path)
    c = diag._reference_docs_check()
    assert c["status"] == diag.WARN
    assert "GEOTECH_REFERENCES_DOCS" in c["detail"]


def test_reference_pdf_names_reads_the_real_catalogs():
    pytest.importorskip("geotech_references")
    names = diag.reference_pdf_names()
    assert "dm7_2" in names.get("ufc_3_220_20_2025.pdf", [])
    assert "worked examples" in names.get("GEC 12 Vol 3.pdf", [])
    assert len(names) >= 30


@pytest.fixture(autouse=True)
def _empty_reference_cache(monkeypatch, tmp_path):
    """PDFs fetched in other tests/sessions must not count as available here."""
    monkeypatch.setenv("GEOTECH_REFERENCES_CACHE", str(tmp_path / "ref_cache"))


def test_reference_docs_counts_what_sharepoint_can_supply(monkeypatch, tmp_path):
    """Owner decision 2026-09-15: PDFs in SharePoint primary_references/,
    fetched on first use. The row counts those and lists only what is
    available nowhere."""
    from geotech_references import _figures_db
    from webapp import reference_fetch, sharepoint_store
    monkeypatch.setattr(diag, "reference_pdf_names", lambda: _NEEDED)
    monkeypatch.delenv("GEOTECH_REFERENCES_DOCS", raising=False)
    monkeypatch.setattr(_figures_db, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(sharepoint_store, "configured", lambda: True)
    monkeypatch.setattr(reference_fetch, "available_names",
                        lambda: {"A Doc.pdf", "b.pdf"})
    c = diag._reference_docs_check()
    assert c["status"] == diag.WARN
    assert "SharePoint primary_references/" in c["detail"]
    assert "- C.pdf  (gec_7)" in c["detail"]
    assert "- b.pdf" not in c["detail"]
    monkeypatch.setattr(reference_fetch, "available_names", lambda: set(_NEEDED))
    c = diag._reference_docs_check()
    assert c["status"] == diag.PASS and "all 3" in c["detail"]
    assert "3 from SharePoint primary_references/" in c["detail"]


def test_reference_docs_unlistable_sharepoint_folder_is_reported(monkeypatch,
                                                                 tmp_path):
    from geotech_references import _figures_db
    from webapp import reference_fetch, sharepoint_store
    monkeypatch.setattr(diag, "reference_pdf_names", lambda: _NEEDED)
    monkeypatch.delenv("GEOTECH_REFERENCES_DOCS", raising=False)
    monkeypatch.setattr(_figures_db, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(sharepoint_store, "configured", lambda: True)
    monkeypatch.setattr(reference_fetch, "available_names", lambda: None)
    c = diag._reference_docs_check()
    assert c["status"] == diag.FAIL and "could not list" in c["detail"]


# ---------------------------------------------------------------------------
# Optional analysis packages (BACKLOG INF-54): one that is installed but will
# not import is reported with its import's error, not as "not installed".
# The failures are faked (root conftest ``fail_import``); the texts are made up.
# ---------------------------------------------------------------------------

def _only(monkeypatch, *packages):
    monkeypatch.setattr(diag, "OPTIONAL_PACKAGES", tuple(
        e for e in diag.OPTIONAL_PACKAGES if e[0] in packages))


@pytest.mark.parametrize("entry", diag.OPTIONAL_PACKAGES,
                         ids=[e[0] for e in diag.OPTIONAL_PACKAGES])
def test_every_optional_package_names_a_real_check(entry):
    import importlib
    _package, _tools, module, func = entry
    assert callable(getattr(importlib.import_module(module), func))


def test_an_installed_package_that_will_not_import_warns(monkeypatch,
                                                         fail_import):
    import salib_agent.salib_utils  # noqa: F401
    import sys
    _only(monkeypatch, "SALib", "PyNiteFEA")
    fail_import("SALib", ImportError(
        "inf54 made-up: C extension built against another numpy"))
    monkeypatch.setitem(sys.modules, "Pynite", None)
    c = diag._optional_packages_check()
    assert c["status"] == diag.WARN
    lines = c["detail"].splitlines()
    assert lines[0] == ("SALib is installed but cannot be imported: "
                        "ImportError: inf54 made-up: C extension built "
                        "against another numpy (tools: salib)")
    assert lines[-1] == "import: none | not installed: PyNiteFEA (pynite)"
    assert "SALib is not installed" not in c["detail"]
    text = diag.format_report([c])
    assert "optional packages: WARN" in text and "inf54 made-up" in text


def test_missing_packages_are_listed_without_a_warning(monkeypatch):
    import sys
    import types
    _only(monkeypatch, "SALib", "PyNiteFEA")
    monkeypatch.setitem(sys.modules, "SALib", types.ModuleType("SALib"))
    monkeypatch.setitem(sys.modules, "Pynite", None)
    c = diag._optional_packages_check()
    assert c["status"] == diag.PASS
    imports, missing = c["detail"].split(" | ")
    assert imports.startswith("import: SALib")
    assert missing == "not installed: PyNiteFEA (pynite)"


def test_a_check_that_fails_itself_warns(monkeypatch):
    monkeypatch.setattr(diag, "OPTIONAL_PACKAGES", (
        ("inf54-madeup", "madeup", "inf54_madeup_module", "madeup_error"),))
    c = diag._optional_packages_check()
    assert c["status"] == diag.WARN
    assert c["detail"].startswith("inf54-madeup: the check itself failed: "
                                  "ModuleNotFoundError")


def test_run_diagnostics_includes_the_optional_packages(monkeypatch):
    _patch_engine(monkeypatch, _GoodModel())
    _only(monkeypatch, "SALib")
    names = [c["name"] for c in diag.run_diagnostics("ri.fake")]
    assert names.index("optional packages") == names.index("version drift") + 1
