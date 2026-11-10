"""TA-35: SharePoint for testers.

The cloud found from the site's address, any site a user points at (opened
directly, never read on the default site in its place), and SharePoint
errors kept in a log file in SharePoint.

Every value here is made up: tenant and client ids, contoso sites, a
made-up secret. No network: ``urllib.request.urlopen`` is replaced by a fake
that answers as the sign-in endpoint and Graph would.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import pytest

pytest.importorskip("langchain_core")

from aec_document_review import graph_sharepoint as gs
from aec_document_review import sharepoint_log as slog
from aec_document_review import sharepoint_store as sp
from aec_document_review import sharepoint_tools as spt
from aec_document_review import tinyapps_settings as ts

TENANT = "00000000-1111-2222-3333-444444444444"
CLIENT = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
SECRET = "made-up-secret-value-123"
SITE = "https://contoso.sharepoint.com/sites/Alpha"

_ENVS = (
    "APP_ENV", "IDENTITY_ENDPOINT", "KV_NAME", "GEOTECH_DOTENV",
    gs.ENV_TENANT, gs.ENV_CLIENT_ID, gs.ENV_CLIENT_SECRET, gs.ENV_SITE_URL,
    gs.ENV_SITE_URLS, gs.ENV_CLOUD, gs.ENV_LOGIN_URL, gs.ENV_API_URL,
    gs.ENV_CLIENT_CERT, gs.ENV_CLIENT_KEY, gs.ENV_CLIENT_KEY_PASSWORD,
    gs.ENV_CERT_THUMBPRINT, "GEOTECH_SHAREPOINT_ROOT",
    "GEOTECH_SHAREPOINT_SITE_URL", "GEOTECH_SHAREPOINT_TOKEN",
)


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    for e in _ENVS:
        monkeypatch.delenv(e, raising=False)
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("GEOTECH_DOTENV", str(tmp_path / "none.env"))
    ts.reset()
    slog.reset()
    yield
    slog.flush(10.0)
    slog.reset()
    ts.reset()


def _resp(obj):
    class R:
        def __init__(self, data):
            self._d = data

        def read(self):
            return self._d

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    return R(obj if isinstance(obj, bytes) else json.dumps(obj).encode())


def _err(url, code, body=None):
    return urllib.error.HTTPError(url, code, "err", {}, io.BytesIO(
        json.dumps(body or {"error": {"code": "accessDenied"}}).encode()))


class FakeGraph:
    """Sign-in and Graph. ``sites`` maps ``host/sites/X`` to a site id;
    ``refused`` lists the ones that answer 403 (not granted) and anything
    not listed answers 404. ``shares`` maps a share id to
    ``(drive id, folder inside the library, name)`` or an HTTP status; a
    fourth element ``"no path"`` leaves the folder out of the sharing-link
    answer (the item read by its id carries it), ``"no folder"`` leaves it
    out of both."""

    def __init__(self, sites=None, refused=(), shares=None, files=None):
        self.calls = []
        self.forms = []
        self.sites = sites or {"contoso.sharepoint.com/sites/Alpha": "SA"}
        self.refused = set(refused)
        self.shares = shares or {}
        self.files = dict(files or {})       # (drive id, path) -> bytes
        self.put = {}
        self.items = {}                      # (drive id, item id) -> folder

    def _drive_site(self, drive_id):
        sid = drive_id.split("-")[1]
        return next(k for k, v in self.sites.items() if v == sid)

    def __call__(self, req, timeout=None):
        url, method = req.full_url, req.get_method()
        self.calls.append((method, url))
        if "/oauth2/v2.0/token" in url:
            self.forms.append(dict(urllib.parse.parse_qsl(req.data.decode())))
            return _resp({"access_token": "TOKEN", "expires_in": 3600})
        path = url.split("/v1.0/", 1)[1] if "/v1.0/" in url else url
        m = re.match(r"sites/([^:/]+):(/(?:sites|teams)/[^/?]+)$", path)
        if m:
            key = m.group(1) + m.group(2)
            if key in self.refused:
                raise _err(url, 403)
            if key not in self.sites:
                raise _err(url, 404, {"error": {"code": "itemNotFound"}})
            return _resp({"id": self.sites[key]})
        m = re.match(r"sites/([^/]+)/drives$", path)
        if m:
            sid = m.group(1)
            key = next(k for k, v in self.sites.items() if v == sid)
            return _resp({"value": [{
                "id": f"D-{sid}", "name": "Documents",
                "webUrl": f"https://{key}/Shared%20Documents"}]})
        m = re.match(r"shares/([^/]+)/driveItem$", path)
        if m:
            got = self.shares.get(m.group(1))
            if got is None:
                raise _err(url, 404, {"error": {"code": "itemNotFound"}})
            if isinstance(got, int):
                raise _err(url, got)
            drive, folder, name = got[:3]
            ref = {"driveId": drive}
            if len(got) < 4:
                ref["path"] = f"/drives/{drive}/root:/{folder}"
            self.items[(drive, "I")] = (folder, name, len(got) < 4
                                        or got[3] == "no path")
            return _resp({"id": "I", "name": name, "file": {},
                          "parentReference": ref})
        m = re.match(r"drives/([^/]+)/items/([^/]+)$", path)
        if m and method == "GET":
            folder, name, has_path = self.items[(m.group(1), m.group(2))]
            ref = {"driveId": m.group(1)}
            if has_path:
                ref["path"] = f"/drives/{m.group(1)}/root:/{folder}"
            return _resp({"id": m.group(2), "name": name, "file": {},
                          "parentReference": ref})
        m = re.match(r"drives/([^/]+)$", path)
        if m:
            key = self._drive_site(m.group(1))
            return _resp({"id": m.group(1),
                          "webUrl": f"https://{key}/Shared%20Documents"})
        m = re.match(r"drives/([^/]+)/root:/(.+?)(:/content)?$", path)
        if m and method == "GET":
            drive, inner = m.group(1), urllib.parse.unquote(m.group(2))
            if m.group(3):
                if (drive, inner) not in self.files:
                    raise _err(url, 404, {"error": {"code": "itemNotFound"}})
                return _resp(self.files[(drive, inner)])
            if inner == "f" or inner.endswith("/f"):
                return _resp({"id": "FOLD", "name": "f", "folder": {}})
            return _resp({"id": "F", "name": inner.rsplit("/", 1)[-1],
                          "file": {}, "size": 3, "webUrl": "https://x"})
        if method == "GET" and path.endswith("/children"):
            if "/items/FOLD/" in path:
                return _resp({"value": [{"id": "C", "name": "x.pdf",
                                         "file": {}, "size": 3}]})
            return _resp({"value": []})
        if method == "GET" and path.endswith("/root"):
            return _resp({"id": "ROOT", "folder": {}})
        if method == "POST" and path.endswith("/children"):
            name = json.loads(req.data)["name"]
            return _resp({"id": f"FOLDER-{name}", "name": name, "folder": {}})
        if method == "PUT" and ":/content" in path:
            self.put[path.split("?")[0]] = req.data
            return _resp({"id": "PUT"})
        if method == "DELETE":
            return _resp(b"")
        raise AssertionError(f"unexpected call {method} {url}")

    def touched_default(self):
        """Calls that read the default site's drive (D-SA) or its id."""
        return [u for _m, u in self.calls if "/drives/D-SA" in u
                or "/sites/SA/" in u]


def _settings(monkeypatch, site=SITE, many=None, cloud=None):
    monkeypatch.setenv(gs.ENV_TENANT, TENANT)
    monkeypatch.setenv(gs.ENV_CLIENT_ID, CLIENT)
    monkeypatch.setenv(gs.ENV_CLIENT_SECRET, SECRET)
    monkeypatch.setenv(gs.ENV_SITE_URL, site)
    if many:
        monkeypatch.setenv(gs.ENV_SITE_URLS, many)
    if cloud:
        monkeypatch.setenv(gs.ENV_CLOUD, cloud)
    ts.reset()


def _fake(monkeypatch, **kw):
    fake = FakeGraph(**kw)
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return fake


# ------------------------------------------------------- 1. the cloud

@pytest.mark.parametrize("host,login,graph,label", [
    ("contoso.sharepoint.com", "https://login.microsoftonline.com",
     "https://graph.microsoft.com", "commercial"),
    ("contoso.sharepoint.us", "https://login.microsoftonline.us",
     "https://graph.microsoft.us", "GCC High"),
    ("contoso.sharepoint-mil.us", "https://login.microsoftonline.us",
     "https://dod-graph.microsoft.us", "DoD"),
])
def test_the_cloud_comes_from_the_site_address(monkeypatch, host, login,
                                               graph, label):
    site = f"https://{host}/sites/Alpha"
    _settings(monkeypatch, site=site)
    fake = _fake(monkeypatch, sites={f"{host}/sites/Alpha": "SA"})
    mgr = gs.GraphFileManager()
    assert mgr.site_id == "SA"
    assert fake.calls[0][1] == f"{login}/{TENANT}/oauth2/v2.0/token"
    assert fake.forms[0]["scope"] == f"{graph}/.default"
    assert fake.calls[1][1].startswith(f"{graph}/v1.0/sites/{host}:")
    assert mgr.cloud_reason.startswith("from the site address")
    report = gs.format_check(gs.check_connection(mgr, write=False, log=False))
    assert f"cloud {label}" in report and "from the site address" in report


def test_an_explicit_cloud_or_address_still_wins(monkeypatch):
    _settings(monkeypatch, site="https://contoso.sharepoint.us/sites/Alpha",
              cloud="commercial")
    mgr = gs.GraphFileManager()
    assert (mgr.login_url, mgr.graph_url) == (gs.LOGIN_URL, gs.GRAPH_URL)
    assert mgr.cloud_reason == "GRAPH_CLOUD is set"
    monkeypatch.delenv(gs.ENV_CLOUD)
    monkeypatch.setenv(gs.ENV_API_URL, "https://graph.example.com")
    ts.reset()
    mgr = gs.GraphFileManager()
    assert mgr.graph_url == "https://graph.example.com"
    assert mgr.login_url == "https://login.microsoftonline.us"
    assert "GRAPH_API_URL set" in mgr.cloud_reason


def test_an_address_naming_no_cloud_is_commercial():
    cloud, why = gs.choose_cloud(None, [gs.Site("x", "https://files.example.com/sites/A")])
    assert cloud == "commercial" and why.startswith("the default")


def test_sites_in_two_clouds_are_refused(monkeypatch):
    _settings(monkeypatch,
              many="other=https://contoso.sharepoint.us/sites/Beta")
    with pytest.raises(ValueError, match="more than one Microsoft cloud"):
        gs.GraphFileManager()
    results = gs.check_connection(log=False)
    assert not results[0]["ok"] and "more than one Microsoft cloud" in \
        results[0]["message"]


# ------------------------------------------ 2. a site a user points at

def test_another_site_by_plain_address_opens_that_site(monkeypatch):
    _settings(monkeypatch)
    fake = _fake(monkeypatch, sites={
        "contoso.sharepoint.com/sites/Alpha": "SA",
        "contoso.sharepoint.com/sites/Gamma": "SG"})
    mgr = gs.GraphFileManager()
    url = "https://contoso.sharepoint.com/sites/Gamma/Shared%20Documents/f"
    entries = mgr.ls(url)
    assert [e["path"] for e in entries] == [
        "/sites/Gamma/Shared Documents/f/x.pdf"]
    assert any("sites/contoso.sharepoint.com:/sites/Gamma" in u
               for _m, u in fake.calls)
    assert fake.touched_default() == []
    # the path handed back finds the same site again
    fake.files[("D-SG", "f/x.pdf")] = b"PDF"
    assert mgr.download_file(entries[0]["path"]) == b"PDF"
    assert mgr.get_web_url(entries[0]["path"]).startswith(
        "https://contoso.sharepoint.com/sites/Gamma/Shared%20Documents/")
    assert fake.touched_default() == []


def test_a_path_on_an_unknown_site_is_never_read_on_the_default_site(
        monkeypatch):
    _settings(monkeypatch)
    fake = _fake(monkeypatch, refused={"contoso.sharepoint.com/sites/Gamma"},
                 files={("D-SA", "Reports/r.pdf"): b"WRONG"})
    mgr = gs.GraphFileManager()
    for call in (lambda: mgr.ls("/sites/Gamma/Shared Documents/Reports"),
                 lambda: mgr.download_file(
                     "/sites/Gamma/Shared Documents/Reports/r.pdf")):
        with pytest.raises(gs.SiteNotOpened) as info:
            call()
        assert "cannot open the site" in info.value.user_message
        assert "step 3" in info.value.user_message
        assert info.value.http_status == 403
    assert fake.touched_default() == []
    # an address not found at all says the address may be wrong
    with pytest.raises(gs.SiteNotOpened, match="address is wrong"):
        mgr.ls("https://contoso.sharepoint.com/sites/Nowhere/Shared Documents")


def test_another_site_by_sharing_link(monkeypatch):
    _settings(monkeypatch)
    link = ("https://contoso.sharepoint.com/:b:/s/Gamma/"
            "EaBcDeFgHiJkLmNoPqRsTuV?e=AbCdEf")
    sid = "u!" + base64.b64encode(link.encode()).decode().rstrip("=") \
        .replace("/", "_").replace("+", "-")
    assert gs.share_id(link) == sid
    fake = _fake(monkeypatch, sites={
        "contoso.sharepoint.com/sites/Alpha": "SA",
        "contoso.sharepoint.com/sites/Gamma": "SG"},
        shares={sid: ("D-SG", "Reports", "r.pdf")},
        files={("D-SG", "Reports/r.pdf"): b"PDF"})
    mgr = gs.GraphFileManager()
    assert spt.browser_url_to_path(link) is None      # a token link
    details = mgr.get_file_details(link)
    assert details["path"] == "/sites/Gamma/Shared Documents/Reports/r.pdf"
    assert details["name"] == "r.pdf"
    assert mgr.download_file(link) == b"PDF"
    assert fake.touched_default() == []


def test_a_sharing_link_to_the_default_site_reads_as_its_own_path(
        monkeypatch):
    _settings(monkeypatch)
    link = "https://contoso.sharepoint.com/:f:/g/ExYz012?e=x"
    _fake(monkeypatch, shares={gs.share_id(link): ("D-SA", "General", "f")})
    mgr = gs.GraphFileManager()
    assert mgr._clean(link) == "Shared Documents/General/f"


def test_a_sharing_link_the_app_cannot_read_asks_for_the_address(
        monkeypatch):
    """Microsoft does not list Sites.Selected for ``GET /shares``: a refused
    sharing-link read asks for the browser address or the path, and never
    sends the user to ask for a site grant, which cannot fix it."""
    _settings(monkeypatch)
    for status in (403, 401, 404):
        link = f"https://contoso.sharepoint.com/:b:/s/Gamma/EaBcD{status}?e=1"
        fake = _fake(monkeypatch, shares={gs.share_id(link): status})
        mgr = gs.GraphFileManager()
        with pytest.raises(gs.SiteNotOpened) as info:
            mgr.download_file(link)
        said = info.value.user_message
        assert "cannot open that sharing link" in said
        assert "cannot be opened with the app's permission" in said
        assert "browser bar" in said and "path" in said
        assert "grant" not in said.replace("granted sites", "")
        assert "step 3" not in said
        assert fake.touched_default() == []


def test_a_sharing_link_without_its_folder_is_read_by_its_id(monkeypatch):
    """The sharing-link answer may leave ``parentReference.path`` out: the
    item is read again by its drive and id; when that has no folder either,
    the user is told plainly -- never the library's top folder instead."""
    _settings(monkeypatch)
    link = "https://contoso.sharepoint.com/:b:/s/Gamma/EaNoPath?e=1"
    fake = _fake(monkeypatch, sites={
        "contoso.sharepoint.com/sites/Alpha": "SA",
        "contoso.sharepoint.com/sites/Gamma": "SG"},
        shares={gs.share_id(link): ("D-SG", "Reports", "r.pdf", "no path")},
        files={("D-SG", "Reports/r.pdf"): b"PDF"})
    mgr = gs.GraphFileManager()
    assert mgr._clean(link) == "/sites/Gamma/Shared Documents/Reports/r.pdf"
    assert any("/drives/D-SG/items/I" in u for _m, u in fake.calls)
    assert mgr.download_file(link) == b"PDF"
    lost = "https://contoso.sharepoint.com/:f:/s/Gamma/EaNoFolder?e=1"
    fake.shares[gs.share_id(lost)] = ("D-SG", "Reports", "f", "no folder")
    with pytest.raises(gs.SiteNotOpened) as info:
        mgr.ls(lost)
    assert "could not tell where that sharing link points" in \
        info.value.user_message
    assert "browser bar" in info.value.user_message
    assert fake.touched_default() == []


def test_a_site_in_another_cloud_is_refused(monkeypatch):
    _settings(monkeypatch)
    _fake(monkeypatch)
    mgr = gs.GraphFileManager()
    with pytest.raises(gs.SiteNotOpened, match="GCC High cloud"):
        mgr.ls("https://contoso.sharepoint.us/sites/Beta/Shared Documents")


@pytest.fixture
def graph_store(monkeypatch, tmp_path):
    """The tools over the real GraphFileManager on a fake Graph."""
    _settings(monkeypatch)
    monkeypatch.setenv("GEOTECH_SHAREPOINT_ROOT", "Shared Documents/APP")
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(work))
    fake = _fake(monkeypatch, refused={"contoso.sharepoint.com/sites/Gamma"},
                 files={("D-SA", "Reports/r.pdf"): b"WRONG"})
    monkeypatch.setattr(sp, "_STORE",
                        sp.SharePointStore(file_manager=gs.GraphFileManager()))
    return fake


def test_the_tools_say_plainly_when_a_site_is_not_granted(graph_store):
    path = "/sites/Gamma/Shared Documents/Reports/r.pdf"
    out = spt._download(path, "", False, None)
    assert out.startswith("The app cannot open the site")
    assert "step 3" in out
    out = spt._list("https://contoso.sharepoint.com/sites/Gamma/"
                    "Shared%20Documents/Reports")
    assert out.startswith("The app cannot open the site")
    out = spt._search_tool("r.pdf", "/sites/Gamma/Shared Documents")
    assert out.startswith("The app cannot open the site")
    # nothing was read on the default site in its place
    assert not any("/drives/D-SA" in u for _m, u in graph_store.calls)
    # and each was kept for the SharePoint log (in memory: the switch is off)
    assert sum("cannot open the site" in line for line in slog.recent()) == 3


def test_the_tools_keep_a_pasted_address_on_its_own_host(graph_store):
    """Through the tools, not only the file layer: an address in another
    cloud is refused for that cloud, and one on another host -- even with
    the default site's own name -- is opened there or refused, never read
    as the same path on the default site."""
    out = spt._list("https://contoso.sharepoint.us/sites/Beta/"
                    "Shared%20Documents/Reports")
    assert "GCC High cloud" in out and "contoso.sharepoint.us" in out
    other = ("https://host.sharepoint.com/sites/Alpha/"
             "Shared%20Documents/Reports/r.pdf")
    assert spt._resolve(other) == other
    out = spt._download(other, "", False, None)
    assert out.startswith("The app cannot open the site "
                          "https://host.sharepoint.com/sites/Alpha")
    assert graph_store.touched_default() == []
    # granted: it opens there, and the paths handed back find it again
    graph_store.sites["host.sharepoint.com/sites/Alpha"] = "SF"
    graph_store.files[("D-SF", "f/x.pdf")] = b"RIGHT"
    graph_store.files[("D-SA", "f/x.pdf")] = b"WRONG"
    sp._STORE._fm = gs.GraphFileManager()
    listed = spt._list("https://host.sharepoint.com/sites/Alpha/"
                       "Shared%20Documents/f")
    handed = ("https://host.sharepoint.com/sites/Alpha/"
              "Shared Documents/f/x.pdf")
    assert f"({handed})" in listed
    out = spt._download(handed, "", False, None)
    assert out.startswith("Downloaded")
    work = os.environ["GEOTECH_DEFAULT_OUTPUT_DIR"]
    with open(os.path.join(work, "x.pdf"), "rb") as fh:
        assert fh.read() == b"RIGHT"
    assert graph_store.touched_default() == []


# --------------------------------------------- 3. errors kept in SharePoint

class LogFM:
    """A file manager holding the log file."""

    def __init__(self, existing=None, upload_error=None, download_error=None):
        self.files = dict(existing or {})
        self.upload_error = upload_error
        self.download_error = download_error
        self.uploads = 0

    def download_file(self, path, local_path=None, return_bytes=True, **kw):
        if self.download_error:
            raise self.download_error
        if path not in self.files:
            raise gs.GraphError("GET ... -> HTTP 404: itemNotFound", 404)
        return self.files[path]

    def upload_file(self, local, path, overwrite=False, **kw):
        self.uploads += 1
        # a write that itself meets an error must not log to SharePoint
        assert slog.record("nested", "from inside the write") is None
        if self.upload_error:
            raise self.upload_error
        with open(local, "rb") as fh:
            self.files[path] = fh.read()
        return True


@pytest.fixture
def log_on(monkeypatch):
    monkeypatch.setenv("GEOTECH_SHAREPOINT_LOG", "1")
    monkeypatch.setattr(sp, "configured", lambda: True)

    def use(fm):
        monkeypatch.setattr(sp, "_STORE", sp.SharePointStore(file_manager=fm))
        return fm
    return use


def test_the_log_line_format_and_place(monkeypatch, log_on):
    fm = log_on(LogFM())
    monkeypatch.setenv("GEOTECH_SHAREPOINT_ROOT", "Shared Documents/APP/")
    line = slog.record("sharepoint download", "It failed\nbadly",
                       "Shared Documents/APP/r.pdf", wait=True)
    stamp, step, message, path = line.split("\t")
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", stamp)
    assert (step, message, path) == ("sharepoint download",
                                     "It failed | badly",
                                     "Shared Documents/APP/r.pdf")
    month = stamp[:7]
    remote = f"Shared Documents/APP/_app_log/sharepoint-{month}.log"
    assert slog.log_path() == remote
    assert fm.files[remote] == (line + "\n").encode()
    # a second entry is appended, the first kept
    line2 = slog.record("conversation save", "again", wait=True)
    assert fm.files[remote] == (line + "\n" + line2 + "\n").encode()
    # a worker that finds the queue empty clears itself, so a line queued
    # as it ends always starts a new one (never left for the next record)
    worker = slog._worker
    if worker is not None:
        worker.join(10.0)
    assert slog._worker is None
    line3 = slog.record("sharepoint list", "third", wait=True)
    assert fm.files[remote].endswith((line3 + "\n").encode())
    # with no root set: the app's own default folder, not the library's top
    monkeypatch.delenv("GEOTECH_SHAREPOINT_ROOT")
    assert slog.log_path().startswith(f"{sp.DEFAULT_ROOT}/_app_log/")


def test_the_log_never_carries_a_secret_or_token(monkeypatch, log_on):
    monkeypatch.setenv(gs.ENV_CLIENT_SECRET, SECRET)
    fm = log_on(LogFM())
    jwt = "eyJhbGciOiJSUzI1NiJ9.eyJzdWIiOiJ4eHh4eHh4eCJ9.c2lnbmF0dXJl"
    slog.record("sign-in", f"refused: client_secret={SECRET}&scope=x; "
                f"Authorization: Bearer {jwt}; access_token: \"abc123def\"; "
                f"the secret {SECRET} again", wait=True)
    text = b"".join(fm.files.values()).decode()
    for leak in (SECRET, jwt, "abc123def"):
        assert leak not in text
    assert "[removed]" in text
    # a secret the app read from Key Vault (not an environment variable)
    monkeypatch.delenv(gs.ENV_CLIENT_SECRET)
    vault_secret = "made-up-vault-secret-456"
    monkeypatch.setitem(ts._cache, gs.ENV_CLIENT_SECRET, vault_secret)
    assert vault_secret not in slog.scrub(f"refused: {vault_secret}")


def test_the_switch_turns_it_off(monkeypatch, log_on):
    fm = log_on(LogFM())
    monkeypatch.setenv("GEOTECH_SHAREPOINT_LOG", "off")
    assert not slog.enabled()
    slog.record("x", "y", wait=True)
    assert fm.uploads == 0 and fm.files == {}
    assert len(slog.recent()) == 1            # still shown in diagnostics
    # read like every other setting: Key Vault or .env when not in the env
    monkeypatch.delenv("GEOTECH_SHAREPOINT_LOG")
    slog.reset()
    monkeypatch.setattr(ts, "get_setting",
                        lambda name, *a, **k: "0"
                        if name == slog.ENV_SWITCH else None)
    assert not slog.enabled()


def test_a_failed_write_never_raises_or_loops(monkeypatch, log_on):
    fm = log_on(LogFM(upload_error=gs.GraphError("PUT -> HTTP 403", 403)))
    assert slog.record("list", "refused", wait=True)
    assert fm.uploads == 1                   # tried once, not again
    assert isinstance(slog.last_write, str)
    assert [l.split("\t")[1] for l in slog.recent()] == ["list"]
    # a log not read (sign-in refused) is never overwritten
    fm = log_on(LogFM(existing={slog.log_path(): b"old\n"},
                      download_error=gs.GraphError("HTTP 401", 401)))
    slog.record("list", "refused", wait=True)
    assert fm.uploads == 0 and fm.files[slog.log_path()] == b"old\n"

    class Broken:
        def file_manager(self):
            raise RuntimeError("no client")
    monkeypatch.setattr(sp, "get_store", lambda *a, **k: Broken())
    assert slog.record("list", "refused", wait=True)


def test_a_connection_test_report_is_logged(monkeypatch, log_on):
    _settings(monkeypatch)
    _fake(monkeypatch)
    fm = log_on(LogFM())
    gs.check_connection(gs.GraphFileManager(), write=False)
    slog.flush(10.0)
    text = b"".join(fm.files.values()).decode()
    assert "\tconnection test\tPASS | PASS  settings:" in text
    assert "cloud commercial" in text and SECRET not in text


def test_a_tool_failure_is_logged(monkeypatch, log_on):
    fm = log_on(LogFM())
    monkeypatch.setenv("GEOTECH_SHAREPOINT_ROOT", "Shared Documents/APP")
    spt._failure("download of Shared Documents/APP/r.pdf",
                 gs.GraphError("GET -> HTTP 403: accessDenied", 403),
                 "Shared Documents/APP/r.pdf")
    slog.flush(10.0)
    text = b"".join(fm.files.values()).decode()
    assert "\tsharepoint download of Shared Documents/APP/r.pdf\t" in text
    assert "REFUSED" in text


def test_a_failed_save_is_logged_by_its_thread_id(monkeypatch, tmp_path,
                                                  log_on):
    """The save path itself: ``mirror_conversation`` with a file manager
    that refuses the uploads. The entry names the conversation by its
    thread id; its folder (the owner and a title made from the first
    question) is in neither the path nor the error text."""
    from aec_document_review import core

    class FailingFM(LogFM):
        """Holds the log; every conversation upload is refused with an
        error that carries the folder, URL-quoted as Graph's URLs do."""

        def create_folder(self, path):
            return True

        def get_web_url(self, path):
            return f"https://contoso.sharepoint.com/{path}"

        def upload_file(self, local, path, overwrite=False, **kw):
            if f"/{slog.FOLDER}/" in path:
                return super().upload_file(local, path, overwrite)
            raise gs.GraphError(
                "PUT https://graph.microsoft.com/v1.0/drives/D/root:/"
                f"{urllib.parse.quote(path.split('/', 1)[1])}:/content -> "
                "HTTP 503: serviceNotAvailable", 503)

    fm = log_on(FailingFM())
    root = str(tmp_path / "data")
    tid = "T35SAVE01"
    core.ensure_conversation(tid, root=root)
    meta = core.load_meta(tid, root)
    meta.update(title="Bearing capacity at 12 Example Road",
                owner="corp__alice", created=time.time())
    core.save_meta(tid, meta, root)
    summary = sp.get_store().mirror_conversation(tid, root=root)
    assert summary["errors"] and "Bearing" in summary["folder"]
    slog.flush(10.0)
    text = b"".join(fm.files.values()).decode()
    entry = next(line for line in text.splitlines()
                 if "\tconversation save\t" in line)
    assert entry.endswith(f"\tconversations/{tid}")
    assert f"conversations/{tid}" in entry.split("\t")[2]
    for private in ("Bearing", "Example", "alice"):
        assert private not in entry
    assert "HTTP 503" in entry
    # in memory it is the owner's own, never another person's
    assert any("\tconversation save\t" in line
               for line in slog.recent("corp__alice"))
    assert not any("\tconversation save\t" in line
                   for line in slog.recent("corp__bob"))


def test_diagnostics_offer_the_report_when_sharepoint_is_unreachable(
        monkeypatch):
    from aec_document_review import diagnostics
    monkeypatch.setenv(gs.ENV_CLIENT_SECRET, SECRET)
    slog.record("sharepoint list", "SharePoint REFUSED the request")
    fail = [{"name": "SharePoint connection", "status": diagnostics.FAIL,
             "detail": f"FAIL  sign-in: Sign-in was refused.\n"
                       f"      Server said: client_secret={SECRET}"}]
    text = diagnostics.sharepoint_download(fail, who="")
    assert "Sign-in was refused" in text and "SharePoint REFUSED" in text
    assert "could not be written to SharePoint" in text
    assert SECRET not in text
    ok = [dict(fail[0], status=diagnostics.PASS)]
    assert diagnostics.sharepoint_download(ok, who="") is None
    assert diagnostics.sharepoint_download([], who="") is None


def test_the_download_holds_only_the_callers_own_entries():
    """On a shared host the download holds the connection test's report and
    the downloader's own entries -- never another person's, which carry
    their folder and file names."""
    from aec_document_review import diagnostics
    with slog.speaking_for("corp__alice"):
        slog.record("sharepoint list", "alice's failure",
                    "Shared Documents/APP/conversations/corp__alice/"
                    "Bearing_capacity_2026-11-05/files/alice_notes.pdf")
    with slog.speaking_for("corp__bob"):
        slog.record("sharepoint download of r.pdf", "bob's failure")
    slog.record("connection test", "FAIL | an earlier test")   # no owner
    fail = [{"name": "SharePoint connection", "status": diagnostics.FAIL,
             "detail": "FAIL  sign-in: Sign-in was refused."}]
    text = diagnostics.sharepoint_download(fail, who="corp__bob")
    assert "Sign-in was refused" in text and "bob's failure" in text
    for other in ("alice", "Bearing", "an earlier test"):
        assert other not in text
    # a single-user host (no owner anywhere) sees its own untagged entries
    assert "an earlier test" in diagnostics.sharepoint_download(fail, who="")
