"""The Tiny Apps plumbing, offline: settings, identity, engine, SharePoint.

Nothing here touches a network or a real Key Vault. Every value is a fake set
through the environment or a temporary ``.env``; Graph is a recorded fake.
"""

from __future__ import annotations

import json
import os
import ssl
import urllib.parse
import urllib.request

import pytest

from webapp import engine_config, graph_sharepoint, identity
from webapp import tinyapps_engine as te
from webapp import tinyapps_settings as ts

_ALL_ENVS = (
    "APP_ENV", "IDENTITY_ENDPOINT", "KV_NAME", "GEOTECH_DOTENV",
    "PROMPTER_URL", "PROMPTER_MODEL", "PROMPTER_API_KEY", "PROMPTER_CA_BUNDLE",
    "GRAPH_TENANT_ID", "GRAPH_CLIENT_ID", "GRAPH_CLIENT_SECRET",
    "SHAREPOINT_SITE_URL", "GEOTECH_DEPLOYMENT", "GEOTECH_PROMPTER_MODELS",
    "GEOTECH_WEBAPP_MODEL", "ANTHROPIC_API_KEY", "DEV_IDENTITY",
    "GEOTECH_USER_EMAIL", "GEOTECH_MARKUP_AUTHOR", "WINDOWS_AUTH_HEADER",
    "GEOTECH_PROMPTER_DISABLE_STREAMING",
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for e in _ALL_ENVS:
        monkeypatch.delenv(e, raising=False)
    ts.reset()
    engine_config.register_model_builder(None)
    yield
    ts.reset()
    engine_config.register_model_builder(None)


# ---------------------------------------------------------------- settings

def test_env_var_wins_over_dotenv(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text('PROMPTER_MODEL="from-dotenv"\nPROMPTER_URL=u\n',
                        encoding="utf-8")
    monkeypatch.setenv("GEOTECH_DOTENV", str(env_file))
    monkeypatch.setenv("PROMPTER_MODEL", "from-env")
    assert ts.app_env() == "local"
    assert ts.get_setting("PROMPTER-MODEL") == "from-env"     # hyphen spelling
    assert ts.get_setting("PROMPTER_URL") == "u"              # from .env
    assert ts.get_setting("NOT_THERE", default="d") == "d"


def test_required_missing_names_the_dotenv(tmp_path, monkeypatch):
    monkeypatch.setenv("GEOTECH_DOTENV", str(tmp_path / ".env"))
    with pytest.raises(RuntimeError) as exc:
        ts.get_setting("PROMPTER_API_KEY", required=True)
    assert ".env" in str(exc.value) and "PROMPTER_API_KEY" in str(exc.value)


def test_azure_mode_autodetects_and_needs_kv_name(monkeypatch):
    monkeypatch.setenv("IDENTITY_ENDPOINT", "http://169.254.169.254/msi")
    assert ts.app_env() == "azure"
    with pytest.raises(RuntimeError) as exc:
        ts.get_setting("PROMPTER_API_KEY")
    assert "KV_NAME" in str(exc.value)


# ---------------------------------------------------------------- identity

def test_header_identity_parses_domain_user():
    ident = identity.from_header_values(["CORP\\jdoe"])
    assert ident.domain == "CORP" and ident.username == "jdoe"
    assert ident.qualified_name == "CORP\\jdoe"
    assert ident.key == "corp__jdoe"
    assert ident.authenticated and ident.source == "header"


def test_two_header_values_trust_neither():
    assert identity.from_header_values(["CORP\\a", "CORP\\b"]) is None


def test_dev_identity_and_email_fallbacks(monkeypatch):
    assert identity.current_identity() is identity.ANONYMOUS
    monkeypatch.setenv("GEOTECH_USER_EMAIL", "jane.doe@state.gov")
    ident = identity.current_identity()
    assert ident.source == "email" and ident.username == "jane.doe"
    assert ident.key == "state.gov__jane.doe"
    monkeypatch.setenv("DEV_IDENTITY", "CORP\\jdoe")
    assert identity.current_identity().source == "dev"


def test_key_is_case_insensitive_and_folder_safe():
    a = identity.parse_principal("CORP\\J Doe", "header")
    b = identity.parse_principal("corp\\j doe", "header")
    assert a.key == b.key == "corp__j_doe"
    assert "\\" not in a.key and " " not in a.key


def test_markup_author_names_the_user_and_the_model(monkeypatch):
    ident = identity.parse_principal("CORP\\jdoe", "header")
    assert ident.markup_author == "jdoe via GeotechStaffEngineer (AI draft)"
    assert identity.ANONYMOUS.markup_author == identity.DEFAULT_MARKUP_AUTHOR
    monkeypatch.setenv("GEOTECH_MARKUP_AUTHOR", "Review Bot")
    assert ident.markup_author == "Review Bot"


# ------------------------------------------------------------------ engine

def test_base_url_strips_chat_completions_and_query():
    assert te.base_url_from(
        "https://prompter.x/api/v1/chat/completions?api-version=1") == \
        "https://prompter.x/api/v1"
    assert te.base_url_from("https://prompter.x/api/v1/") == \
        "https://prompter.x/api/v1"


def test_ssl_context_accepts_pem_text_or_path_and_rejects_junk(tmp_path):
    assert te.ssl_context(None) is None
    pem = ssl.DER_cert_to_PEM_cert(_self_signed_der())
    ctx = te.ssl_context(pem)
    assert isinstance(ctx, ssl.SSLContext)
    assert ctx.verify_flags & ssl.VERIFY_X509_PARTIAL_CHAIN
    p = tmp_path / "ca.pem"
    p.write_text(pem, encoding="utf-8")
    assert isinstance(te.ssl_context(str(p)), ssl.SSLContext)
    with pytest.raises(ValueError):
        te.ssl_context("not-a-cert-and-not-a-path")


def test_settings_none_until_all_three_present(monkeypatch):
    monkeypatch.setenv("PROMPTER_URL", "https://p/api/v1/chat/completions")
    monkeypatch.setenv("PROMPTER_MODEL", "gpt-4.1")
    assert te.settings() is None and not te.configured()
    monkeypatch.setenv("PROMPTER_API_KEY", "k")
    ts.reset()
    ps = te.settings()
    assert ps.base_url == "https://p/api/v1" and ps.ca_bundle is None


def test_build_chat_model_sends_key_both_ways(monkeypatch):
    pytest.importorskip("langchain_openai")
    ps = te.PrompterSettings("https://p/api/v1/chat/completions", "gpt-4.1",
                             "secret-key")
    model = te.build_chat_model(prompter=ps)
    assert model.model_name == "gpt-4.1"
    assert str(model.openai_api_base).rstrip("/") == "https://p/api/v1"
    assert model.default_headers == {"api-key": "secret-key"}
    # the client sends the key as Bearer; the header above is the second way
    assert model.openai_api_key.get_secret_value() == "secret-key"
    assert model.max_tokens == engine_config._default_max_tokens()


def test_register_installs_builder_and_publishes_model(monkeypatch):
    pytest.importorskip("langchain_openai")
    assert te.register() is False
    monkeypatch.setenv("PROMPTER_URL", "https://p/api/v1/chat/completions")
    monkeypatch.setenv("PROMPTER_MODEL", "gpt-4.1")
    monkeypatch.setenv("PROMPTER_API_KEY", "k")
    assert te.register() is True
    assert engine_config.has_model_builder()
    assert os.environ["GEOTECH_PROMPTER_MODELS"] == "gpt-4.1=gpt-4.1"
    # the picker hands the builder its selection; register() made the
    # deployment's model the default one
    from webapp import core
    assert core.default_model_id() == "gpt-4.1"
    res = engine_config.resolve_engine(core.default_model_id())
    assert res.ok and res.source == "prompter" and res.model_name == "gpt-4.1"


def test_tinyapps_deployment_never_names_the_personal_key(monkeypatch):
    monkeypatch.setenv("GEOTECH_DEPLOYMENT", "tinyapps")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "should-never-be-read")
    assert engine_config.is_tinyapps_deployment()
    assert engine_config.is_keyless_deployment()
    res = engine_config.resolve_engine()
    assert not res.ok and res.source == "none"
    assert "ANTHROPIC" not in res.message and "PROMPTER_API_KEY" in res.message
    from webapp import core
    assert core.default_model_id() == ""
    assert all(c["id"].startswith("ri.") or c["id"] == ""
               for c in core.model_choices()) or core.model_choices() == []


# --------------------------------------------------------------- SharePoint

class _FakeGraph:
    """Records every Graph request and answers a tiny fake site."""

    def __init__(self):
        self.calls = []
        self.files = {}          # "lib/path" -> bytes

    def __call__(self, req, timeout=None):
        url, method = req.full_url, req.get_method()
        self.calls.append((method, url))
        body = req.data or b""
        if "oauth2/v2.0/token" in url:
            return _resp({"access_token": "T", "expires_in": 3600})
        if url.endswith("/sites/host.sharepoint.com:/sites/Geo"):
            return _resp({"id": "SITE"})
        if url.endswith("/sites/SITE/drives"):
            return _resp({"value": [
                {"id": "D1", "name": "Documents",
                 "webUrl": "https://host.sharepoint.com/sites/Geo/Shared%20Documents"},
                {"id": "D2", "name": "Site Assets",
                 "webUrl": "https://host.sharepoint.com/sites/Geo/SiteAssets"}]})
        if url.endswith("/drives/D1/root"):
            return _resp({"id": "ROOT", "name": "root", "folder": {}})
        if "/root:/" in url and url.endswith(":/content") and method == "GET":
            key = url.split("/root:/", 1)[1][: -len(":/content")]
            return _resp_bytes(self.files[urllib.parse.unquote(key)])
        if "/root:/" in url and method == "GET":
            key = urllib.parse.unquote(url.split("/root:/", 1)[1])
            if key in self.files:
                return _resp({"id": "F-" + key, "name": key.rsplit("/", 1)[-1],
                              "size": len(self.files[key]), "file": {},
                              "webUrl": "https://host/" + key})
            if key == "GSE":
                return _resp({"id": "GSE", "name": "GSE", "folder": {}})
            raise _http_error(url, 404)
        if url.endswith("/items/ROOT/children") and method == "GET":
            return _resp({"value": [{"id": "GSE", "name": "GSE", "folder": {}}]})
        if url.endswith("/items/GSE/children") and method == "GET":
            return _resp({"value": [
                {"id": "F-GSE/a.txt", "name": "a.txt", "size": 1, "file": {},
                 "webUrl": "https://host/GSE/a.txt",
                 "fileSystemInfo": {"lastModifiedDateTime": "2026-09-21T00:00:00Z"}},
                {"id": "SUB", "name": "sub", "folder": {}}]})
        if url.endswith("/children") and method == "POST":
            name = json.loads(body)["name"]
            return _resp({"id": "NEW-" + name, "name": name, "folder": {}})
        if ":/content?@microsoft.graph.conflictBehavior=" in url and method == "PUT":
            fname = urllib.parse.unquote(
                url.split("/items/", 1)[1].split(":/", 1)[1]
                .split(":/content", 1)[0])
            behaviour = url.rsplit("=", 1)[1]
            if fname == "exists.txt" and behaviour == "fail":
                raise _http_error(url, 409)
            self.files["GSE/" + fname] = body
            return _resp({"id": "UP", "name": fname, "webUrl": "https://host/x"})
        if "search(q=" in url:
            return _resp({"value": [
                {"id": "S1", "name": "meta.json", "size": 5, "file": {},
                 "webUrl": "https://host/GSE/conv1/meta.json",
                 "parentReference": {"path": "/drives/D1/root:/GSE/conv1"}},
                {"id": "S2", "name": "meta.json", "size": 5, "file": {},
                 "webUrl": "https://host/Other/meta.json",
                 "parentReference": {"path": "/drives/D1/root:/Other"}}]})
        raise AssertionError(f"unexpected Graph call {method} {url}")


class _Resp:
    def __init__(self, data: bytes):
        self._d = data

    def read(self):
        return self._d

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _resp(obj):
    return _Resp(json.dumps(obj).encode())


def _resp_bytes(b):
    return _Resp(b)


def _http_error(url, code):
    import io
    import urllib.error
    return urllib.error.HTTPError(url, code, "err", {}, io.BytesIO(b"{}"))


@pytest.fixture
def fm(monkeypatch):
    fake = _FakeGraph()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    for k, v in (("GRAPH_TENANT_ID", "tid"), ("GRAPH_CLIENT_ID", "cid"),
                 ("GRAPH_CLIENT_SECRET", "sec"),
                 ("SHAREPOINT_SITE_URL", "https://host.sharepoint.com/sites/Geo")):
        monkeypatch.setenv(k, v)
    ts.reset()
    mgr = graph_sharepoint.GraphFileManager()
    mgr._fake = fake
    return mgr


def test_graph_configured_needs_all_four(monkeypatch):
    assert not graph_sharepoint.configured()
    for k in ("GRAPH_TENANT_ID", "GRAPH_CLIENT_ID", "GRAPH_CLIENT_SECRET"):
        monkeypatch.setenv(k, "x")
    ts.reset()
    assert not graph_sharepoint.configured()
    monkeypatch.setenv("SHAREPOINT_SITE_URL", "https://h/sites/s")
    ts.reset()
    assert graph_sharepoint.configured()
    from webapp import sharepoint_store
    assert sharepoint_store.configured()          # the store sees it too


def test_ls_maps_the_shape_the_app_reads(fm):
    entries = fm.ls("Shared Documents/GSE")
    names = {e["name"]: e for e in entries}
    assert names["a.txt"]["type"] == "file" and not names["a.txt"]["is_folder"]
    assert names["a.txt"]["path"] == "Shared Documents/GSE/a.txt"
    assert names["sub"]["type"] == "folder" and names["sub"]["is_folder"]
    assert fm.ls("Shared Documents/GSE", type="files") == [names["a.txt"]]
    assert fm.ls("Shared Documents/does-not-exist") == []
    # the token was fetched once and the drive resolved by its UI name
    token_calls = [c for c in fm._fake.calls if "oauth2" in c[1]]
    assert len(token_calls) == 1


def test_upload_download_roundtrip_and_overwrite_contract(fm, tmp_path):
    src = tmp_path / "hello.txt"
    src.write_bytes(b"hello")
    assert fm.upload_file(str(src), "Shared Documents/GSE/hello.txt",
                          overwrite=True) is True
    dst = tmp_path / "back.txt"
    assert fm.download_file("Shared Documents/GSE/hello.txt",
                            local_path=str(dst), return_bytes=False,
                            overwrite=True) is True
    assert dst.read_bytes() == b"hello"
    assert fm.download_file("Shared Documents/GSE/hello.txt") == b"hello"
    # an existing remote name with overwrite=False is a False, not a raise
    exists = tmp_path / "exists.txt"
    exists.write_bytes(b"x")
    assert fm.upload_file(str(exists), "Shared Documents/GSE/exists.txt",
                          overwrite=False) is False


def test_create_folder_walks_and_creates_missing_levels(fm):
    details = fm.create_folder("Shared Documents/GSE/newdir")
    assert details["name"] == "newdir"
    assert details["web_url"].startswith(
        "https://host.sharepoint.com/sites/Geo/Shared%20Documents/GSE/newdir")
    posts = [c for c in fm._fake.calls if c[0] == "POST" and "children" in c[1]]
    assert len(posts) == 1                         # GSE existed; newdir did not


def test_search_scopes_to_the_folder(fm):
    hits = fm.search_filenames("meta.json", "Shared Documents/GSE")
    assert [h["path"] for h in hits] == ["Shared Documents/GSE/conv1/meta.json"]


def test_web_url_is_the_ui_path_with_web_flag(fm):
    assert fm.get_web_url("Shared Documents/GSE/a b.txt") == \
        "https://host.sharepoint.com/sites/Geo/Shared%20Documents/GSE/a%20b.txt?web=1"


def test_store_builds_the_graph_manager_when_configured(fm, monkeypatch):
    from webapp import sharepoint_store
    monkeypatch.delenv(sharepoint_store.ENV_SITE, raising=False)
    mgr = sharepoint_store._build_file_manager()
    assert isinstance(mgr, graph_sharepoint.GraphFileManager)


# ------------------------------------------------------------------ helpers

def _self_signed_der() -> bytes:
    """A throwaway self-signed certificate, so the SSL tests need no fixture
    file. Uses the ``cryptography`` package when present (it arrives with the
    Azure libraries); otherwise the test is skipped rather than shipping a
    certificate in the repo."""
    pytest.importorskip("cryptography")
    from datetime import datetime, timedelta, timezone
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives.serialization import Encoding
    from cryptography.x509.oid import NameOID
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test-ca")])
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(1)
            .not_valid_before(now - timedelta(days=1))
            .not_valid_after(now + timedelta(days=1))
            .sign(key, hashes.SHA256()))
    return cert.public_bytes(Encoding.DER)
