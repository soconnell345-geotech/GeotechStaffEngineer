"""TA-34: SharePoint set up by the host team on any tenant.

Every value here is made up: tenant and client ids, example.com /
contoso.sharepoint.com sites, and a certificate generated in the test. No
network: ``urllib.request.urlopen`` is replaced by a fake that answers as the
sign-in endpoint and Graph would.
"""

from __future__ import annotations

import base64
import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request

import pytest

from aec_document_review import graph_sharepoint as gs
from aec_document_review import tinyapps_settings as ts

TENANT = "00000000-1111-2222-3333-444444444444"
CLIENT = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
SITE = "https://contoso.sharepoint.com/sites/Alpha"
SITE_B = "https://contoso.sharepoint.com/sites/Beta"

_ENVS = (
    "APP_ENV", "IDENTITY_ENDPOINT", "KV_NAME", "GEOTECH_DOTENV",
    gs.ENV_TENANT, gs.ENV_CLIENT_ID, gs.ENV_CLIENT_SECRET, gs.ENV_SITE_URL,
    gs.ENV_SITE_URLS, gs.ENV_CLOUD, gs.ENV_LOGIN_URL, gs.ENV_API_URL,
    gs.ENV_CLIENT_CERT, gs.ENV_CLIENT_KEY, gs.ENV_CLIENT_KEY_PASSWORD,
    gs.ENV_CERT_THUMBPRINT, "GEOTECH_SHAREPOINT_ROOT",
    "GEOTECH_SHAREPOINT_SITE_URL",
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    for e in _ENVS:
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("GEOTECH_DOTENV", str(tmp_path / "none.env"))
    ts.reset()
    yield
    ts.reset()


def _resp(obj, code=200):
    class R:
        def __init__(self, data):
            self._d = data

        def read(self):
            return self._d

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    return R(json.dumps(obj).encode() if not isinstance(obj, bytes) else obj)


def _err(url, code, body):
    return urllib.error.HTTPError(url, code, "err", {},
                                  io.BytesIO(json.dumps(body).encode()))


class FakeCloud:
    """A sign-in endpoint and Graph, on whatever hosts the test names."""

    def __init__(self, sites=None, token_error=None, site_errors=None,
                 drives=("Shared Documents",), write_error=None,
                 delete_error=None):
        self.calls = []
        self.forms = []
        self.bodies = []
        self.timeouts = []
        self.puts = []
        self.made = []
        self.sites = sites or {"/sites/Alpha": "SA", "/sites/Beta": "SB"}
        self.token_error = token_error
        self.site_errors = site_errors or {}
        self.drives = drives
        self.write_error = write_error
        self.delete_error = delete_error

    def __call__(self, req, timeout=None):
        url, method = req.full_url, req.get_method()
        self.calls.append((method, url))
        self.timeouts.append(timeout)
        if "/oauth2/v2.0/token" in url:
            self.bodies.append(req.data)
            self.forms.append(dict(urllib.parse.parse_qsl(req.data.decode())))
            if self.token_error:
                code, body = self.token_error
                raise _err(url, code, body)
            return _resp({"access_token": "TOKEN", "expires_in": 3600})
        if ".sharepoint.com:/sites/" in url:
            path = "/sites/" + url.rsplit(":/sites/", 1)[1]
            if path in self.site_errors:
                code, body = self.site_errors[path]
                raise _err(url, code, body)
            if path not in self.sites:
                raise _err(url, 404, {"error": {"code": "itemNotFound"}})
            return _resp({"id": self.sites[path]})
        if url.endswith("/drives"):
            sid = url.rsplit("/sites/", 1)[1].split("/")[0]
            name = {"SA": "Alpha", "SB": "Beta"}.get(sid, "X")
            return _resp({"value": [
                {"id": f"D-{sid}-{i}", "name": lib,
                 "webUrl": f"https://contoso.sharepoint.com/sites/{name}/"
                           + urllib.parse.quote(lib)}
                for i, lib in enumerate(self.drives)]})
        if "/root:/" in url and method == "GET":
            return _resp({"id": "F", "name": url.rsplit("/", 1)[-1],
                          "file": {}, "size": 1, "webUrl": "https://x"})
        # the connection test's write probe
        if method == "GET" and url.endswith("/root"):
            return _resp({"id": "ROOT", "folder": {}})
        if method == "GET" and url.endswith("/children"):
            return _resp({"value": []})
        if method == "POST" and url.endswith("/children"):
            name = json.loads(req.data)["name"]
            self.made.append(name)
            return _resp({"id": f"FOLDER-{name}", "name": name, "folder": {}})
        if method == "PUT" and url.split("?")[0].endswith(":/content"):
            self.puts.append((url, req.data))
            if self.write_error:
                raise _err(url, *self.write_error)
            return _resp({"id": "PROBE"})
        if method == "DELETE":
            if self.delete_error:
                raise _err(url, *self.delete_error)
            return _resp(b"")
        raise AssertionError(f"unexpected call {method} {url}")


def _base_settings(monkeypatch, secret=True):
    monkeypatch.setenv(gs.ENV_TENANT, TENANT)
    monkeypatch.setenv(gs.ENV_CLIENT_ID, CLIENT)
    if secret:
        monkeypatch.setenv(gs.ENV_CLIENT_SECRET, "made-up-secret")
    monkeypatch.setenv(gs.ENV_SITE_URL, SITE)
    ts.reset()


@pytest.fixture
def cloud(monkeypatch):
    fake = FakeCloud()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return fake


# ---------------------------------------------------------------- clouds

@pytest.mark.parametrize("name,login,graph", [
    (None, "https://login.microsoftonline.com", "https://graph.microsoft.com"),
    ("commercial", "https://login.microsoftonline.com",
     "https://graph.microsoft.com"),
    ("gcc_high", "https://login.microsoftonline.us",
     "https://graph.microsoft.us"),
    ("GCC High", "https://login.microsoftonline.us",
     "https://graph.microsoft.us"),
    ("dod", "https://login.microsoftonline.us",
     "https://dod-graph.microsoft.us"),
])
def test_each_cloud_signs_in_and_calls_graph_on_its_own_addresses(
        monkeypatch, cloud, name, login, graph):
    _base_settings(monkeypatch)
    if name:
        monkeypatch.setenv(gs.ENV_CLOUD, name)
        ts.reset()
    mgr = gs.GraphFileManager()
    assert mgr.site_id == "SA"
    token_url = cloud.calls[0][1]
    assert token_url == f"{login}/{TENANT}/oauth2/v2.0/token"
    assert cloud.forms[0]["scope"] == f"{graph}/.default"
    assert cloud.calls[1][1].startswith(f"{graph}/v1.0/sites/")


def test_explicit_urls_override_the_cloud(monkeypatch, cloud):
    _base_settings(monkeypatch)
    monkeypatch.setenv(gs.ENV_CLOUD, "gcc_high")
    monkeypatch.setenv(gs.ENV_LOGIN_URL, "https://login.example.com/")
    monkeypatch.setenv(gs.ENV_API_URL, "https://graph.example.com")
    ts.reset()
    mgr = gs.GraphFileManager()
    mgr.site_id
    assert cloud.calls[0][1].startswith("https://login.example.com/" + TENANT)
    assert cloud.forms[0]["scope"] == "https://graph.example.com/.default"
    assert cloud.calls[1][1].startswith("https://graph.example.com/v1.0/")


def test_unknown_cloud_is_refused_plainly():
    with pytest.raises(ValueError, match="GRAPH_LOGIN_URL"):
        gs.cloud_urls("moon")


def test_no_new_settings_is_todays_behaviour(monkeypatch, cloud):
    _base_settings(monkeypatch)
    assert gs.configured()
    mgr = gs.GraphFileManager()
    assert mgr.credential_kind == "secret"
    assert [s.url for s in mgr.sites] == [SITE] and mgr.site_url == SITE
    mgr.site_id
    form = cloud.forms[0]
    assert form == {"grant_type": "client_credentials", "client_id": CLIENT,
                    "client_secret": "made-up-secret",
                    "scope": "https://graph.microsoft.com/.default"}
    # The same bytes as master's form (grant_type, client_id,
    # client_secret, scope, in that order), not only the same fields.
    assert cloud.bodies[0] == (
        b"grant_type=client_credentials"
        b"&client_id=aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        b"&client_secret=made-up-secret"
        b"&scope=https%3A%2F%2Fgraph.microsoft.com%2F.default")
    assert gs.site_note(mgr) == ""


# ------------------------------------------------------------ certificate

def _make_cert(tmp_path, together=True, password=None):
    crypto = pytest.importorskip("cryptography")
    del crypto
    import datetime
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test-app")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(1)
            .not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=30))
            .sign(key, hashes.SHA256()))
    enc = (serialization.BestAvailableEncryption(password.encode())
           if password else serialization.NoEncryption())
    key_pem = key.private_bytes(serialization.Encoding.PEM,
                                serialization.PrivateFormat.PKCS8,
                                enc).decode()
    cert_pem = cert.public_bytes(serialization.Encoding.PEM).decode()
    return key, cert, key_pem, cert_pem


def _b64d(part):
    return base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))


def _make_pfx(key, cert, password=None):
    """The certificate and key as PKCS #12 bytes (a .pfx), as Key Vault
    keeps a certificate by default."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.serialization import pkcs12
    enc = (serialization.BestAvailableEncryption(password.encode())
           if password else serialization.NoEncryption())
    return pkcs12.serialize_key_and_certificates(b"test-app", key, cert,
                                                 None, enc)


def _check_assertion(assertion, cert, token_url, alg="PS256"):
    """The assertion as Microsoft's certificate-credentials page gives it:
    PS256 with the SHA-256 thumbprint ``x5t#S256`` when the certificate is
    known; RS256 with the SHA-1 ``x5t`` when only a thumbprint is."""
    import hashlib
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    head, body, sig = assertion.split(".")
    header = json.loads(_b64d(head))
    claims = json.loads(_b64d(body))
    der = cert.public_bytes(serialization.Encoding.DER)
    assert header["alg"] == alg and header["typ"] == "JWT"
    if alg == "PS256":
        assert _b64d(header["x5t#S256"]) == hashlib.sha256(der).digest()
        assert "x5t" not in header
        pad = padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32)
    else:
        assert _b64d(header["x5t"]) == hashlib.sha1(der).digest()
        pad = padding.PKCS1v15()
    assert claims["aud"] == token_url
    assert claims["iss"] == CLIENT and claims["sub"] == CLIENT
    assert claims["exp"] > claims["nbf"] and claims["jti"]
    assert claims["exp"] - claims["nbf"] <= 600
    cert.public_key().verify(_b64d(sig), f"{head}.{body}".encode(),
                             pad, hashes.SHA256())


def test_certificate_from_a_file_signs_a_valid_assertion(
        monkeypatch, cloud, tmp_path):
    _key, cert, key_pem, cert_pem = _make_cert(tmp_path)
    pem = tmp_path / "app.pem"
    pem.write_text(key_pem + cert_pem)
    _base_settings(monkeypatch, secret=False)
    monkeypatch.setenv(gs.ENV_CLIENT_CERT, str(pem))
    ts.reset()
    assert gs.configured()
    mgr = gs.GraphFileManager()
    assert mgr.credential_kind == "certificate"
    mgr.site_id
    form = cloud.forms[0]
    assert "client_secret" not in form
    assert form["client_assertion_type"] == gs.ASSERTION_TYPE
    _check_assertion(form["client_assertion"], cert, mgr.token_url)
    # the key never shows in the credential's repr
    assert "PRIVATE" not in repr(mgr._certificate)


def test_certificate_wins_over_a_secret_and_takes_pem_text_and_a_separate_key(
        monkeypatch, cloud, tmp_path):
    _key, cert, key_pem, cert_pem = _make_cert(tmp_path, password="pw")
    _base_settings(monkeypatch)                         # a secret is set too
    monkeypatch.setenv(gs.ENV_CLOUD, "gcc_high")
    monkeypatch.setenv(gs.ENV_CLIENT_CERT, cert_pem.replace("\n", "\\n"))
    monkeypatch.setenv(gs.ENV_CLIENT_KEY, key_pem)
    monkeypatch.setenv(gs.ENV_CLIENT_KEY_PASSWORD, "pw")
    ts.reset()
    mgr = gs.GraphFileManager()
    mgr.site_id
    form = cloud.forms[0]
    assert "client_secret" not in form
    assert form["scope"] == "https://graph.microsoft.us/.default"
    _check_assertion(form["client_assertion"], cert,
                     f"https://login.microsoftonline.us/{TENANT}/oauth2/v2.0/token")


def test_key_with_only_a_thumbprint(tmp_path):
    import hashlib
    from cryptography.hazmat.primitives import serialization
    _key, cert, key_pem, _cert_pem = _make_cert(tmp_path)
    der = cert.public_bytes(serialization.Encoding.DER)
    thumb = hashlib.sha1(der).hexdigest().upper()
    cred = gs.CertificateCredential(key_pem=key_pem, thumbprint=thumb)
    _check_assertion(cred.assertion(CLIENT, "https://login.example.com/t"),
                     cert, "https://login.example.com/t", alg="RS256")
    with pytest.raises(ValueError, match="thumbprint"):
        gs.CertificateCredential(key_pem=key_pem)


def test_pkcs12_as_key_vault_returns_it_base64(monkeypatch, cloud, tmp_path):
    """A Key Vault certificate read by its secret name comes back as the
    base64 text of a PKCS #12 with no password."""
    key, cert, _key_pem, _cert_pem = _make_cert(tmp_path)
    value = base64.b64encode(_make_pfx(key, cert)).decode()
    _base_settings(monkeypatch, secret=False)
    monkeypatch.setenv(gs.ENV_CLIENT_CERT, value)
    ts.reset()
    assert gs.configured()
    mgr = gs.GraphFileManager()
    assert mgr.credential_kind == "certificate"
    mgr.site_id
    form = cloud.forms[0]
    assert form["client_assertion_type"] == gs.ASSERTION_TYPE
    _check_assertion(form["client_assertion"], cert, mgr.token_url)
    # field order: grant type, client id, the credential, scope
    assert list(form) == ["grant_type", "client_id", "client_assertion_type",
                          "client_assertion", "scope"]


def test_pkcs12_file_with_a_password(tmp_path):
    key, cert, _key_pem, _cert_pem = _make_cert(tmp_path)
    pfx = tmp_path / "app.pfx"
    pfx.write_bytes(_make_pfx(key, cert, password="made-up-pw"))
    cred = gs.CertificateCredential(cert_pem=str(pfx), password="made-up-pw")
    _check_assertion(cred.assertion(CLIENT, "https://login.example.com/t"),
                     cert, "https://login.example.com/t")
    # the wrong password, or none: a plain error, never a UnicodeDecodeError
    for pw in ("wrong", None):
        with pytest.raises(ValueError, match="PKCS #12") as info:
            gs.CertificateCredential(cert_pem=str(pfx), password=pw)
        assert gs.ENV_CLIENT_KEY_PASSWORD in str(info.value)
        assert "made-up-pw" not in str(info.value)


def test_a_certificate_setting_in_no_known_form_is_refused_plainly(
        monkeypatch, tmp_path):
    pytest.importorskip("cryptography")
    junk = tmp_path / "app.bin"
    junk.write_bytes(bytes(range(256)))         # a binary file, not PKCS #12
    for value in (str(junk), "C:\\no\\such\\app.pfx", "not a certificate"):
        with pytest.raises(ValueError, match="PEM text") as info:
            gs.CertificateCredential(cert_pem=value)
        assert not isinstance(info.value, UnicodeDecodeError)
    # the connection test names the setting problem in its first step
    _base_settings(monkeypatch, secret=False)
    monkeypatch.setenv(gs.ENV_CLIENT_CERT, str(junk))
    ts.reset()
    results = gs.check_connection()
    assert results[-1]["step"] == "settings" and not results[-1]["ok"]
    assert "PKCS #12" in results[-1]["message"]


# ------------------------------------------------------------------ sites

def test_site_list_parsing():
    sites = gs.parse_sites(SITE, f"beta={SITE_B}\n{SITE}, "
                                 "https://contoso.sharepoint.com/teams/Gamma/")
    assert [(s.name, s.url) for s in sites] == [
        ("Alpha", SITE), ("beta", SITE_B),
        ("Gamma", "https://contoso.sharepoint.com/teams/Gamma")]
    assert gs.parse_sites(None, f"{SITE_B}")[0].name == "Beta"


def test_paths_name_their_site_and_the_first_is_the_default(monkeypatch, cloud):
    _base_settings(monkeypatch)
    monkeypatch.setenv(gs.ENV_SITE_URLS, f"beta={SITE_B}")
    ts.reset()
    mgr = gs.GraphFileManager()
    assert mgr.site_names() == ["Alpha", "beta"]
    # no prefix: the default site, exactly as before
    entries = mgr.ls("Shared Documents/a.txt")
    assert entries[0]["path"] == "Shared Documents/a.txt"
    assert any("/sites/SA/drives" in u for _m, u in cloud.calls)
    # a prefix: that site, and results carry the prefix back
    entries = mgr.ls("beta:Shared Documents/b.txt")
    assert entries[0]["path"] == "beta:Shared Documents/b.txt"
    assert any("/sites/SB/drives" in u for _m, u in cloud.calls)
    # the server path of a configured site finds it too
    assert mgr._clean("/sites/Beta/Shared Documents/x") == \
        "beta:Shared Documents/x"
    assert mgr._clean("/sites/Alpha/Shared Documents/x") == \
        "Shared Documents/x"
    # web links use the site's own address
    assert mgr.get_web_url("beta:Shared Documents/b c.txt") == \
        f"{SITE_B}/Shared%20Documents/b%20c.txt?web=1"
    # an unknown name is just part of a path on the default site
    assert mgr._clean("zeta:notes") == "zeta:notes"
    note = gs.site_note(mgr)
    assert "'Alpha'" in note and "'beta'" in note and "beta:Shared" in note


def test_only_the_site_list_is_enough(monkeypatch):
    _base_settings(monkeypatch)
    monkeypatch.delenv(gs.ENV_SITE_URL)
    monkeypatch.setenv(gs.ENV_SITE_URLS, f"{SITE_B}")
    ts.reset()
    assert gs.configured()
    assert gs.GraphFileManager().site_url == SITE_B


def test_tools_name_the_sites_only_when_there_are_several(monkeypatch, cloud):
    from aec_document_review import sharepoint_store, sharepoint_tools as spt
    monkeypatch.setattr(sharepoint_store, "_foundry_selected", lambda: False)
    _base_settings(monkeypatch)
    monkeypatch.setattr(sharepoint_store, "_STORE", None, raising=False)
    store = sharepoint_store.SharePointStore(file_manager=gs.GraphFileManager())
    monkeypatch.setattr(sharepoint_store, "get_store", lambda: store)
    tools, prompt = spt.tools_if_configured()
    assert prompt == spt.SHAREPOINT_PROMPT
    assert "SharePoint sites" not in tools[0].description

    monkeypatch.setenv(gs.ENV_SITE_URLS, f"beta={SITE_B}")
    ts.reset()
    store2 = sharepoint_store.SharePointStore(
        file_manager=gs.GraphFileManager())
    monkeypatch.setattr(sharepoint_store, "get_store", lambda: store2)
    tools, prompt = spt.tools_if_configured()
    assert "SharePoint sites" in prompt and "beta:" in prompt
    names = {t.name: t for t in tools}
    for n in ("sharepoint_list_files", "sharepoint_download_file",
              "sharepoint_search_files"):
        assert "SharePoint sites" in names[n].description
    assert spt._resolve("beta:Shared Documents/x.pdf") == \
        "beta:Shared Documents/x.pdf"


# ------------------------------------------------------- connection test

def _run(monkeypatch, fake):
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    results = gs.check_connection()
    return results, gs.format_check(results)


def test_check_names_the_missing_settings():
    results = gs.check_connection()
    assert len(results) == 1 and not results[0]["ok"]
    msg = results[0]["message"]
    for n in ("GRAPH_TENANT_ID", "GRAPH_CLIENT_ID", "GRAPH_CLIENT_SECRET",
              "SHAREPOINT_SITE_URL"):
        assert n in msg


def test_check_passes_every_step(monkeypatch):
    _base_settings(monkeypatch)
    fake = FakeCloud()
    results, text = _run(monkeypatch, fake)
    assert [r["step"] for r in results] == [
        "settings", "sign-in", "site Alpha", "library Shared Documents",
        "write"]
    assert all(r["ok"] for r in results)
    assert "SharePoint is set up." in text
    assert "made-up-secret" not in text
    # the write probe: put in the library's root, then deleted, and said so
    assert "Wrote and deleted a probe file, Shared Documents/" \
        f"{gs.PROBE_NAME}" in text
    assert len(fake.puts) == 1 and not fake.made
    assert fake.puts[0][0].startswith(
        f"https://graph.microsoft.com/v1.0/drives/D-SA-0/items/ROOT:/"
        f"{gs.PROBE_NAME}:/content")
    methods = [m for m, _u in fake.calls]
    assert methods[-2:] == ["PUT", "DELETE"]
    assert fake.calls[-1][1].endswith("/drives/D-SA-0/items/PROBE")
    # every call of the check waits at most 30 s (sign-in and Graph alike)
    assert fake.timeouts and all(t == 30.0 for t in fake.timeouts)


def test_check_write_probe_goes_in_the_configured_root(monkeypatch):
    _base_settings(monkeypatch)
    monkeypatch.setenv("GEOTECH_SHAREPOINT_ROOT", "Shared Documents/Pilot")
    fake = FakeCloud()
    results, text = _run(monkeypatch, fake)
    assert results[-1]["step"] == "write" and results[-1]["ok"]
    assert fake.made == ["Pilot"]               # as the app's first save would
    assert f"/items/FOLDER-Pilot:/{gs.PROBE_NAME}:/content" in fake.puts[0][0]
    assert f"Shared Documents/Pilot/{gs.PROBE_NAME}" in text


def test_check_read_only_grant_fails_the_write_step(monkeypatch):
    _base_settings(monkeypatch)
    fake = FakeCloud(write_error=(403, {"error": {
        "code": "accessDenied", "message": "made-up access text"}}))
    results, text = _run(monkeypatch, fake)
    assert [r["ok"] for r in results] == [True, True, True, True, False]
    assert results[-1]["step"] == "write"
    assert "can read this site but not write" in results[-1]["message"]
    assert "Grant 'write'" in results[-1]["message"]
    assert "made-up access text" in text
    assert "SharePoint is set up" not in text
    assert "DELETE" not in [m for m, _u in fake.calls]


def test_check_probe_left_behind_is_named(monkeypatch):
    _base_settings(monkeypatch)
    fake = FakeCloud(delete_error=(500, {"error": {"code": "generalException"}}))
    results, text = _run(monkeypatch, fake)
    assert results[-1]["ok"]                    # saving works
    assert "delete it by hand" in results[-1]["message"]
    assert gs.PROBE_NAME in results[-1]["message"]


def test_check_read_only_skips_the_write_probe(monkeypatch, capsys):
    _base_settings(monkeypatch)
    fake = FakeCloud()
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    results = gs.check_connection(write=False)
    text = gs.format_check(results)
    assert results[-1]["step"] == "write" and results[-1].get("skipped")
    assert "SKIP  write" in text
    assert "saving was not tested" in text
    assert not fake.puts
    assert gs.main(["--check", "--read-only"]) == 0
    assert "SKIP  write" in capsys.readouterr().out
    assert not fake.puts


@pytest.mark.parametrize("code,expect", [
    ("AADSTS90002", "tenant was not found"),
    ("AADSTS700016", "client) ID was not found"),
    ("AADSTS7000215", "client secret is wrong"),
    ("AADSTS7000222", "client secret has expired"),
    ("AADSTS700027", "certificate was not accepted"),
    ("AADSTS900382", "different Microsoft cloud"),
    ("AADSTS900432", "different Microsoft cloud"),
])
def test_check_sign_in_refusals_in_plain_words(monkeypatch, code, expect):
    _base_settings(monkeypatch)
    body = {"error": "invalid_client",
            "error_description": f"{code}: made-up server text"}
    results, text = _run(monkeypatch, FakeCloud(token_error=(401, body)))
    assert results[-1]["step"] == "sign-in" and not results[-1]["ok"]
    assert expect in results[-1]["message"]
    assert "made-up server text" in text          # the server's own words
    assert "made-up-secret" not in text


def test_check_site_not_found(monkeypatch):
    _base_settings(monkeypatch)
    results, _ = _run(monkeypatch, FakeCloud(sites={"/sites/Other": "SO"}))
    assert results[-1]["step"] == "site Alpha"
    assert "was not found" in results[-1]["message"]
    # the government clouds' SharePoint addresses: GCC High, then DoD
    assert "GCC High: .sharepoint.us; DoD: .sharepoint-mil.us" in \
        results[-1]["message"]


def test_check_site_not_granted(monkeypatch):
    _base_settings(monkeypatch)
    fake = FakeCloud(site_errors={"/sites/Alpha": (
        403, {"error": {"code": "accessDenied", "message": "Access denied"}})})
    results, text = _run(monkeypatch, fake)
    msg = results[-1]["message"]
    assert "not been granted this site" in msg
    assert "Grant-PnPAzureADAppSitePermission" in msg
    assert "New-MgSitePermission" in msg
    assert "SharePoint administrator who is a member of the site" in msg
    assert "Access denied" in text


def test_check_no_permission_consented(monkeypatch):
    _base_settings(monkeypatch)
    fake = FakeCloud(site_errors={"/sites/Alpha": (
        401, {"error": {"code": "InvalidAuthenticationToken",
                        "message": "Either scp or roles claim need to be "
                                   "present in the token."}})})
    results, _ = _run(monkeypatch, fake)
    assert "Sites.Selected" in results[-1]["message"]
    assert "admin consent" in results[-1]["message"]


def test_check_second_site_failure_is_named(monkeypatch):
    _base_settings(monkeypatch)
    monkeypatch.setenv(gs.ENV_SITE_URLS, f"beta={SITE_B}")
    ts.reset()
    fake = FakeCloud(site_errors={"/sites/Beta": (
        403, {"error": {"code": "accessDenied"}})})
    results, _ = _run(monkeypatch, fake)
    assert results[-2]["step"] == "site Alpha" and results[-2]["ok"]
    assert results[-1]["step"] == "site beta" and not results[-1]["ok"]


def test_check_library_not_found(monkeypatch):
    _base_settings(monkeypatch)
    monkeypatch.setenv("GEOTECH_SHAREPOINT_ROOT", "Project Files/GSE")
    results, text = _run(monkeypatch, FakeCloud(drives=("Shared Documents",)))
    assert results[-1]["step"] == "library Project Files"
    assert "library was not found" in results[-1]["message"]
    assert "Shared Documents" in text             # the libraries it has


def test_check_cli_exit_codes(monkeypatch, capsys):
    assert gs.main(["--check"]) == 1
    assert "Settings missing" in capsys.readouterr().out
    _base_settings(monkeypatch)
    monkeypatch.setattr(urllib.request, "urlopen", FakeCloud())
    assert gs.main(["--check"]) == 0
    assert "SharePoint is set up." in capsys.readouterr().out


def test_diagnostics_runs_the_check_only_when_a_setting_is_filled(monkeypatch):
    from aec_document_review import diagnostics
    assert not diagnostics._sharepoint_started()
    _base_settings(monkeypatch)
    monkeypatch.setattr(urllib.request, "urlopen", FakeCloud())
    assert diagnostics._sharepoint_started()
    c = diagnostics._sharepoint_check()
    assert c["name"] == "SharePoint connection" and c["status"] == "pass"
