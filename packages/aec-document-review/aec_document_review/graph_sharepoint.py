"""SharePoint files over Microsoft Graph with an app registration — no SDK.

The app has mirrored every conversation to SharePoint since 5.10.0 through
the Funhouse SDK's file manager, whose only login is an interactive device
code — fine in a notebook, impossible in a server process. Tiny Apps
provisions the alternative: an Entra APP REGISTRATION (client id + secret)
granted ``Sites.Selected`` on the team's site, the pattern in CfA's
``exampleCode/sharepoint.py`` (2026-08-28). Uploads then act as the app, not
as the person — SharePoint's audit trail shows the app's name.

:class:`GraphFileManager` speaks the SAME six methods the app already calls
on the SDK's manager (``ls``, ``upload_file``, ``download_file``,
``create_folder``, ``search_filenames``, ``get_web_url``), with the same
argument names and return shapes the callers read, so
:mod:`aec_document_review.sharepoint_store`, :mod:`aec_document_review.sharepoint_tools`,
:mod:`aec_document_review.reference_fetch` and ``aec_agent_core.run_mirror`` need no change
beyond how the manager is built.

Paths are the UI paths the rest of the app uses — ``Shared Documents/
General/GSE_app/...`` — where the first segment names the document library
(Graph calls it a drive) and the rest is the path inside it.

The login and Graph hosts default to the PUBLIC cloud ones: the
organisation's SharePoint tenant is commercial-side even though the App Service
runs in Azure Government (their note). Only the standard library is used for
the calls; ``urllib`` is what their connector uses and it keeps this file
dependency-free.

TA-34 (owner, 2026-11-05): the host team sets the registration up themselves,
on whatever tenant they have, with no code change from us:

* ``GRAPH_CLOUD`` picks Microsoft's national cloud (``commercial``, the
  default, ``gcc_high`` or ``dod``), and ``GRAPH_LOGIN_URL`` /
  ``GRAPH_API_URL`` override either address outright; the token scope
  follows the Graph address.
* A CERTIFICATE as well as a client secret: ``GRAPH_CLIENT_CERT`` (a PEM
  file path, or the PEM text, holding the certificate and, unless
  ``GRAPH_CLIENT_KEY`` names it apart, its private key; or a PKCS #12
  ``.pfx``/``.p12``, as a file path or as the base64 text Key Vault returns
  for a certificate's secret), optionally ``GRAPH_CLIENT_KEY_PASSWORD`` (the
  key's or the PKCS #12's password) and ``GRAPH_CLIENT_CERT_THUMBPRINT``.
  When a certificate is set it is used instead of the secret: a client
  assertion signed with PS256, header ``x5t#S256``, as Microsoft's
  certificate-credentials page gives it (RS256 with ``x5t`` when only a
  thumbprint is known). Needs the ``cryptography`` package, imported only
  then.
* More than one site: ``SHAREPOINT_SITE_URLS`` (one per line or
  comma-separated, each optionally ``name=URL``) beside or instead of
  ``SHAREPOINT_SITE_URL``. The first site is the default, so every path
  that worked before still does; ``name:Shared Documents/...`` names
  another, and so does its ``/sites/<site>/...`` server path.
* :func:`check_connection` (``python -m aec_document_review.graph_sharepoint --check``)
  runs each step in order and says which one failed and what the admin does.

TA-35 (owner, 2026-11-05):

* With ``GRAPH_CLOUD`` unset the cloud comes from the sites' addresses
  (:func:`choose_cloud`); sites in two clouds are refused.
* A site that is not in the settings, named by a user's address, server
  path or sharing link, is opened directly (``GET /sites/{host}:/sites/
  {name}``, ``GET /shares/{u!id}/driveItem``) and NEVER read on the
  default site in its place; one the app was not granted raises
  :class:`SiteNotOpened` with plain words.
* SharePoint errors and connection-test reports go to the app's log file
  in SharePoint (:mod:`aec_document_review.sharepoint_log`).
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

log = logging.getLogger(__name__)

#: The commercial cloud's addresses, the default (and today's behaviour).
LOGIN_URL = "https://login.microsoftonline.com"
GRAPH_URL = "https://graph.microsoft.com"

#: Microsoft's published sign-in and Graph addresses per national cloud
#: ("National cloud deployments", Microsoft Graph docs). GCC (moderate) uses
#: the commercial addresses.
CLOUDS: Dict[str, Tuple[str, str]] = {
    "commercial": (LOGIN_URL, GRAPH_URL),
    "gcc": (LOGIN_URL, GRAPH_URL),
    "gcc_high": ("https://login.microsoftonline.us",
                 "https://graph.microsoft.us"),
    "dod": ("https://login.microsoftonline.us",
            "https://dod-graph.microsoft.us"),
}

#: Graph's simple-upload limit; larger files go through an upload session.
SIMPLE_UPLOAD_MAX = 4 * 1024 * 1024
#: Upload-session chunk: a multiple of 320 KiB, as Graph requires.
CHUNK_SIZE = 10 * 320 * 1024

#: The setting names CfA's engineers provision (env or Key Vault spelling).
ENV_TENANT = "GRAPH_TENANT_ID"
ENV_CLIENT_ID = "GRAPH_CLIENT_ID"
ENV_CLIENT_SECRET = "GRAPH_CLIENT_SECRET"
ENV_SITE_URL = "SHAREPOINT_SITE_URL"
#: TA-34: the optional settings for any tenant (none set = today's behaviour).
ENV_SITE_URLS = "SHAREPOINT_SITE_URLS"
ENV_CLOUD = "GRAPH_CLOUD"
ENV_LOGIN_URL = "GRAPH_LOGIN_URL"
ENV_API_URL = "GRAPH_API_URL"
ENV_CLIENT_CERT = "GRAPH_CLIENT_CERT"
ENV_CLIENT_KEY = "GRAPH_CLIENT_KEY"
ENV_CLIENT_KEY_PASSWORD = "GRAPH_CLIENT_KEY_PASSWORD"
ENV_CERT_THUMBPRINT = "GRAPH_CLIENT_CERT_THUMBPRINT"

#: The JWT bearer assertion type of the client-credentials flow.
ASSERTION_TYPE = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"


class GraphError(RuntimeError):
    """A Graph call failed; ``status`` is the HTTP status when there was one."""

    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class LibraryNotFound(GraphError):
    """The site has no document library by the name a path starts with."""


class SiteNotOpened(GraphError):
    """TA-35: a site (or a sharing link) a user pointed at that the app
    cannot open: not granted to the app, not found, or in another cloud.

    ``status`` is None on purpose, so no caller reads it as an empty folder
    or a missing file and goes on to look on the default site;
    ``http_status`` keeps Graph's own. ``user_message`` is the plain words
    the tools hand back as they are."""

    def __init__(self, message: str, http_status: Optional[int] = None):
        super().__init__(message, status=None)
        self.http_status = http_status
        self.user_message = message


def _setting(name: str) -> str:
    from aec_document_review.tinyapps_settings import get_setting
    return (get_setting(name) or "").strip()


def configured() -> bool:
    """True when the app registration is complete: tenant, client id, a
    credential (the secret, or a certificate) and at least one site.

    The settings are read in an order that stops at the first one present,
    so a deployment with today's four settings reads exactly those four."""
    return bool(_setting(ENV_TENANT) and _setting(ENV_CLIENT_ID)
                and (_setting(ENV_CLIENT_SECRET) or _setting(ENV_CLIENT_CERT)
                     or _setting(ENV_CLIENT_KEY))
                and (_setting(ENV_SITE_URL) or _setting(ENV_SITE_URLS)))


def missing_settings() -> List[str]:
    """The settings still to fill in, by name (empty when :func:`configured`)."""
    out = [n for n in (ENV_TENANT, ENV_CLIENT_ID) if not _setting(n)]
    if not (_setting(ENV_CLIENT_SECRET) or _setting(ENV_CLIENT_CERT)
            or _setting(ENV_CLIENT_KEY)):
        out.append(f"{ENV_CLIENT_SECRET} (or {ENV_CLIENT_CERT})")
    if not (_setting(ENV_SITE_URL) or _setting(ENV_SITE_URLS)):
        out.append(f"{ENV_SITE_URL} (or {ENV_SITE_URLS})")
    return out


# ---------------------------------------------------------------- the cloud

#: TA-35: a SharePoint address's ending -> the cloud that serves it. Only
#: the SharePoint tenant's cloud matters, not where the app is hosted.
SHAREPOINT_CLOUDS: Tuple[Tuple[str, str], ...] = (
    (".sharepoint-mil.us", "dod"),
    (".sharepoint.us", "gcc_high"),
    (".sharepoint.com", "commercial"),      # also Microsoft 365 GCC
)

_CLOUD_LABELS = {"commercial": "commercial (also Microsoft 365 GCC)",
                 "gcc": "commercial (GCC)", "gcc_high": "GCC High",
                 "dod": "DoD"}


def _cloud_key(name: str) -> str:
    key = re.sub(r"[\s\-]+", "_", name.strip().lower())
    if key not in CLOUDS:
        raise ValueError(
            f"{ENV_CLOUD}={name!r} is not a cloud this app knows "
            f"({', '.join(sorted(CLOUDS))}); set {ENV_LOGIN_URL} and "
            f"{ENV_API_URL} for any other.")
    return key


def _host(url: str) -> str:
    return (urllib.parse.urlparse(url if "://" in url else f"https://{url}")
            .hostname or "").lower()


def cloud_of_site(url: str) -> Optional[str]:
    """The cloud a SharePoint site's address names (``None`` for an address
    that names none, e.g. a custom host)."""
    host = _host(url or "")
    for ending, cloud in SHAREPOINT_CLOUDS:
        if host.endswith(ending):
            return cloud
    return None


def _same_cloud(a: str, b: str) -> bool:
    family = {"gcc": "commercial"}
    return family.get(a, a) == family.get(b, b)


def choose_cloud(cloud: Optional[str] = None,
                 sites: Optional[List["Site"]] = None) -> Tuple[str, str]:
    """``(cloud, why)``: ``GRAPH_CLOUD`` when set, else the cloud the sites'
    addresses name (``*.sharepoint.com`` commercial and GCC,
    ``*.sharepoint.us`` GCC High, ``*.sharepoint-mil.us`` DoD), else
    commercial. Sites in two clouds are refused: one app registration signs
    in to one cloud."""
    named: Dict[str, str] = {}
    for s in sites or []:
        c = cloud_of_site(s.url)
        if c:
            named.setdefault(c, s.url)
    if len(named) > 1:
        listed = "; ".join(f"{u} is {_CLOUD_LABELS.get(c, c)}"
                           for c, u in named.items())
        raise ValueError(
            "The SharePoint sites are in more than one Microsoft cloud ("
            f"{listed}). One app registration signs in to one cloud: keep "
            f"only that cloud's sites in {ENV_SITE_URL} / {ENV_SITE_URLS}.")
    explicit = cloud if cloud is not None else _setting(ENV_CLOUD)
    if explicit:
        return _cloud_key(explicit), f"{ENV_CLOUD} is set"
    if named:
        c, url = next(iter(named.items()))
        ending = next(e for e, k in SHAREPOINT_CLOUDS if k == c)
        return c, f"from the site address ({_host(url)}, *{ending})"
    return "commercial", "the default: no site address names a cloud"


def cloud_label(cloud: str) -> str:
    return _CLOUD_LABELS.get(cloud, cloud)


def cloud_urls(cloud: Optional[str] = None, login_url: Optional[str] = None,
               api_url: Optional[str] = None,
               sites: Optional[List["Site"]] = None) -> Tuple[str, str]:
    """``(sign-in address, Graph address)`` for the configured cloud.

    ``GRAPH_CLOUD`` names one of :data:`CLOUDS` (spaces, hyphens and case
    are forgiven: ``GCC High`` works); when it is not set the cloud comes
    from the sites' addresses (:func:`choose_cloud`, TA-35), else
    ``commercial``. ``GRAPH_LOGIN_URL`` / ``GRAPH_API_URL`` replace either
    address."""
    key, _why = choose_cloud(cloud, sites)
    login, graph = CLOUDS[key]
    login = (login_url if login_url is not None
             else _setting(ENV_LOGIN_URL)) or login
    graph = (api_url if api_url is not None else _setting(ENV_API_URL)) or graph
    return login.rstrip("/"), graph.rstrip("/")


# ------------------------------------------------------------------ sites

@dataclass
class Site:
    """One SharePoint site the app may use; ``name`` is what a path's
    ``name:`` prefix says."""
    name: str
    url: str
    site_id: Optional[str] = None
    drive_ids: Dict[str, str] = field(default_factory=dict)
    #: TA-35: a site a user pointed at that is not in the settings.
    adhoc: bool = False

    @property
    def server_prefix(self) -> str:
        """``/sites/<Site>`` as the address spells it (case kept)."""
        return urllib.parse.unquote(urllib.parse.urlparse(
            self.url if "://" in self.url else f"https://{self.url}"
        ).path.rstrip("/"))

    @property
    def server_path(self) -> str:
        """``/sites/<site>`` (or ``/teams/...``), lower case, for matching."""
        return urllib.parse.urlparse(
            self.url if "://" in self.url else f"https://{self.url}"
        ).path.rstrip("/").lower()


def _default_site_name(url: str) -> str:
    path = urllib.parse.urlparse(url if "://" in url else f"https://{url}").path
    tail = urllib.parse.unquote(path.rstrip("/").rsplit("/", 1)[-1])
    return tail or urllib.parse.urlparse(
        url if "://" in url else f"https://{url}").netloc


def parse_sites(single: Optional[str] = None,
                many: Optional[str] = None) -> List[Site]:
    """The configured sites, the default first.

    ``single`` is ``SHAREPOINT_SITE_URL``; ``many`` is
    ``SHAREPOINT_SITE_URLS``: entries one per line or comma-separated, each
    ``URL`` or ``name=URL`` (the name defaults to the URL's last segment,
    ``/sites/Geo`` -> ``Geo``). A URL named twice is kept once."""
    entries: List[Tuple[str, str]] = []
    if single and single.strip():
        u = single.strip().rstrip("/")
        entries.append((_default_site_name(u), u))
    for raw in re.split(r"[\n,;]+", many or ""):
        raw = raw.strip()
        if not raw:
            continue
        name, url = "", raw
        head, sep, tail = raw.partition("=")
        if sep and "/" not in head and ":" not in head:
            name, url = head.strip(), tail.strip()
        url = url.rstrip("/")
        entries.append((name or _default_site_name(url), url))
    out: List[Site] = []
    for name, url in entries:
        if any(s.url.lower() == url.lower() for s in out):
            continue
        out.append(Site(name=name, url=url))
    return out


def share_id(url: str) -> str:
    """A sharing link as Graph's ``/shares/{id}`` takes it: ``u!`` and the
    unpadded base64url of the link (Microsoft's "Access shared items")."""
    return "u!" + base64.urlsafe_b64encode(url.encode("utf-8")).decode(
        "ascii").rstrip("=")


def not_opened_message(where: str, status: Optional[int] = None,
                       link: bool = False) -> str:
    """Plain words for a site or link a user pointed at that the app cannot
    open (TA-35)."""
    what = "that link" if link else f"the site {where}"
    why = ("it is not one of the sites the app has been granted, or the "
           "address is wrong" if status == 404 else
           "it is not one of the sites the app has been granted")
    if link:
        why += ", or the link has expired"
    return (f"The app cannot open {what}: {why}. The app opens only the "
            "SharePoint sites its administrator has granted to it. Ask the "
            "administrator to grant this site to the app "
            "(docs/SHAREPOINT_SETUP.md, step 3: grant the app the site), or "
            "copy the file to a site the app already uses."
            + (f" (Graph answered HTTP {status}.)" if status else ""))


#: What the user pastes instead of a sharing link the app cannot read.
LINK_INSTEAD = ("Paste the address from the browser bar while the file or "
                "folder is open (…/Forms/AllItems.aspx?id=…, or the file's "
                "own address), or its path (Shared Documents/…), instead.")


def link_not_opened_message(status: Optional[int] = None) -> str:
    """Plain words for a sharing link Graph's sharing-link call
    (``GET /shares/...``) refused (TA-35). Microsoft does not list the
    app's site-only permission (Sites.Selected) for that call, so a site
    grant cannot fix it; the address or the path opens the same file."""
    why = ("links of this kind cannot be opened with the app's permission "
           "(it is granted sites, not sharing links)")
    if status == 404:
        why = ("the link was not found (it may have expired), or " + why)
    return (f"The app cannot open that sharing link: {why}. " + LINK_INSTEAD
            + (f" (Graph answered HTTP {status}.)" if status else ""))


def configured_sites() -> List[Site]:
    """:func:`parse_sites` over the settings."""
    single = _setting(ENV_SITE_URL)
    many = _setting(ENV_SITE_URLS)
    return parse_sites(single, many)


# ----------------------------------------------------------- certificates

def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


#: What a certificate setting may hold, for the error when it holds none.
_CERT_FORMS = (
    "PEM text, the path of a PEM or PKCS #12 (.pfx/.p12) file, or the base64 "
    "text of a PKCS #12 (what Key Vault returns for a certificate's secret)")


def _cert_material(value: str) -> Tuple[str, Optional[bytes]]:
    """``(PEM text, None)`` or ``("", PKCS #12 bytes)`` for a certificate or
    key setting: PEM text, a PEM or PKCS #12 file's path, or the base64 text
    of a PKCS #12."""
    v = (value or "").strip()
    if "-----BEGIN" in v:                 # a setting may hold escaped newlines
        return v.replace("\\n", "\n"), None
    if os.path.isfile(v):
        with open(v, "rb") as fh:
            data = fh.read()
        if b"-----BEGIN" in data:
            return data.decode("utf-8", "replace"), None
        return "", data
    try:
        data = base64.b64decode("".join(v.split()), validate=True)
    except (binascii.Error, ValueError):
        data = b""
    if not data:
        raise ValueError(f"the certificate setting is not {_CERT_FORMS}")
    return "", data


def _load_pkcs12(data: bytes, password: Optional[bytes]):
    """``(key, certificate)`` from PKCS #12 bytes, or a plain error."""
    from cryptography.hazmat.primitives.serialization import pkcs12
    try:
        key, cert, _chain = pkcs12.load_key_and_certificates(data, password)
    except (ValueError, TypeError):
        raise ValueError(
            "the certificate setting is not PEM, and it did not load as "
            f"PKCS #12 (a wrong or missing {ENV_CLIENT_KEY_PASSWORD}, or not "
            f"a certificate at all). Give {_CERT_FORMS}.") from None
    return key, cert


class CertificateCredential:
    """A private key and its certificate's thumbprints, signing the client
    assertion Entra takes in place of a secret.

    The certificate and key come as PEM (text or a file) or as PKCS #12 (a
    ``.pfx``/``.p12`` file, or its base64 text); ``password`` opens an
    encrypted key or PKCS #12. The key never leaves this object: no ``repr``
    of it, no log line."""

    def __init__(self, cert_pem: Optional[str] = None,
                 key_pem: Optional[str] = None,
                 thumbprint: Optional[str] = None,
                 password: Optional[str] = None):
        try:
            from cryptography import x509
            from cryptography.hazmat.primitives import hashes, serialization
        except ImportError as exc:             # pragma: no cover - env
            raise RuntimeError(
                "A certificate credential needs the 'cryptography' package "
                "(pip install cryptography), or use GRAPH_CLIENT_SECRET "
                "instead.") from exc
        pw = password.encode() if password else None
        cert_text, cert_pfx = (_cert_material(cert_pem) if cert_pem
                               else ("", None))
        key_text, key_pfx = (_cert_material(key_pem) if key_pem
                             else (cert_text, cert_pfx))
        key = cert = None
        if "BEGIN CERTIFICATE" in cert_text:
            cert = x509.load_pem_x509_certificate(cert_text[
                cert_text.index("-----BEGIN CERTIFICATE"):].encode())
        if cert_pfx is not None:
            pfx_key, cert = _load_pkcs12(cert_pfx, pw)
            if not key_pem:
                key = pfx_key
        if key is None and key_pfx is not None and key_pem:
            key, pfx_cert = _load_pkcs12(key_pfx, pw)
            cert = cert or pfx_cert
        if key is None:
            if "PRIVATE KEY" not in key_text:
                raise ValueError(
                    f"no private key found: put it in {ENV_CLIENT_CERT} (with "
                    f"the certificate) or name it in {ENV_CLIENT_KEY}")
            # Load only the key block; certificates in the same text are
            # skipped.
            m = re.search(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?"
                          r"-----END [A-Z ]*PRIVATE KEY-----", key_text, re.S)
            key = serialization.load_pem_private_key(m.group(0).encode(),
                                                     password=pw)
        self._key = key
        self.x5t_s256: Optional[str] = None
        if cert is not None:
            der = cert.public_bytes(serialization.Encoding.DER)
            self.x5t = _b64url(hashlib.sha1(der).digest())
            self.x5t_s256 = _b64url(hashlib.sha256(der).digest())
            self.not_after = getattr(cert, "not_valid_after_utc", None) \
                or cert.not_valid_after
        elif thumbprint:
            hex_ = re.sub(r"[^0-9a-fA-F]", "", thumbprint)
            self.x5t = _b64url(bytes.fromhex(hex_))
            self.not_after = None
        else:
            raise ValueError(
                f"no certificate found: put it in {ENV_CLIENT_CERT} or give "
                f"its thumbprint in {ENV_CERT_THUMBPRINT}")
        self._hashes = hashes

    def __repr__(self) -> str:
        return f"CertificateCredential(x5t={self.x5t!r})"

    def assertion(self, client_id: str, token_url: str,
                  lifetime: int = 600) -> str:
        """The signed client assertion: PS256 with header ``x5t#S256`` (the
        SHA-256 thumbprint), as Microsoft's certificate-credentials page
        gives it; RS256 with ``x5t`` (SHA-1) when only a thumbprint is
        known, since the portal shows the SHA-1 one."""
        from cryptography.hazmat.primitives.asymmetric import padding
        now = int(time.time())
        if self.x5t_s256:
            header = {"alg": "PS256", "typ": "JWT", "x5t#S256": self.x5t_s256}
            pad = padding.PSS(mgf=padding.MGF1(self._hashes.SHA256()),
                              salt_length=32)
        else:                      # only a SHA-1 thumbprint was given
            header = {"alg": "RS256", "typ": "JWT", "x5t": self.x5t}
            pad = padding.PKCS1v15()
        claims = {"aud": token_url, "iss": client_id, "sub": client_id,
                  "jti": str(uuid.uuid4()), "nbf": now, "iat": now,
                  "exp": now + lifetime}
        signing_input = (_b64url(json.dumps(header, separators=(",", ":"))
                                 .encode()) + "." +
                         _b64url(json.dumps(claims, separators=(",", ":"))
                                 .encode()))
        sig = self._key.sign(signing_input.encode("ascii"), pad,
                             self._hashes.SHA256())
        return signing_input + "." + _b64url(sig)


def certificate_from_settings() -> Optional[CertificateCredential]:
    """The certificate credential the settings describe, or None when no
    certificate setting is present (the secret is used then)."""
    cert = _setting(ENV_CLIENT_CERT)
    key = _setting(ENV_CLIENT_KEY)
    if not (cert or key):
        return None
    return CertificateCredential(
        cert_pem=cert or None, key_pem=key or None,
        thumbprint=_setting(ENV_CERT_THUMBPRINT) or None,
        password=_setting(ENV_CLIENT_KEY_PASSWORD) or None)


class GraphFileManager:
    """Files on one or more SharePoint sites, as the app registration.

    Construct with explicit values or with none — then the settings are
    read through :func:`aec_document_review.tinyapps_settings.get_setting` (env var, then
    Key Vault on a deployment, then ``.env`` locally). Nothing is contacted
    until the first call; a bad secret therefore shows up as a request error
    in the log, not a process that will not start.

    The first site is the default: a path with no ``name:`` prefix is on it,
    exactly as when there was only one.
    """

    def __init__(self, site_url: Optional[str] = None,
                 tenant_id: Optional[str] = None,
                 client_id: Optional[str] = None,
                 client_secret: Optional[str] = None,
                 *, timeout: float = 300.0,
                 sites: Optional[List[Site]] = None,
                 certificate: Optional[CertificateCredential] = None,
                 cloud: Optional[str] = None,
                 login_url: Optional[str] = None,
                 api_url: Optional[str] = None):
        from aec_document_review.tinyapps_settings import get_setting
        if sites is None:
            sites = (parse_sites(site_url) if site_url
                     else configured_sites())
        if not sites:
            get_setting(ENV_SITE_URL, required=True)       # the plain error
        self.sites: List[Site] = list(sites)
        self._tenant = tenant_id or get_setting(ENV_TENANT, required=True)
        self._client_id = client_id or get_setting(ENV_CLIENT_ID, required=True)
        self.cloud, self.cloud_reason = choose_cloud(cloud, self.sites)
        overrides = [n for n, v in ((ENV_LOGIN_URL, login_url),
                                    (ENV_API_URL, api_url))
                     if v or (v is None and _setting(n))]
        if overrides:
            self.cloud_reason += f"; {' and '.join(overrides)} set"
        self.login_url, self.graph_url = cloud_urls(self.cloud, login_url,
                                                    api_url)
        #: TA-35: sites users pointed at (by server path) and resolved
        #: sharing links (link -> path), kept for this manager's life.
        self._adhoc: Dict[str, Site] = {}
        self._shares: Dict[str, str] = {}
        if certificate is None and client_secret is None:
            certificate = certificate_from_settings()
        self._certificate = certificate
        self._secret = None if certificate is not None else (
            client_secret or get_setting(ENV_CLIENT_SECRET, required=True))
        self._timeout = timeout
        self._token: Optional[str] = None
        self._token_expires = 0.0

    @property
    def site_url(self) -> str:
        """The default site's URL (the only one before TA-34)."""
        return self.sites[0].url

    @property
    def credential_kind(self) -> str:
        return "certificate" if self._certificate is not None else "secret"

    @property
    def token_url(self) -> str:
        return f"{self.login_url}/{self._tenant}/oauth2/v2.0/token"

    def site_names(self) -> List[str]:
        return [s.name for s in self.sites]

    # ------------------------------------------------------------------ auth
    def _get_token(self) -> str:
        """Client-credentials token for Graph (~1 h), renewed a minute early."""
        if self._token and time.time() < self._token_expires - 60:
            return self._token
        # Field order: grant type, client id, the credential, scope. With a
        # secret that is the same bytes as before TA-34.
        form = {"grant_type": "client_credentials",
                "client_id": self._client_id}
        if self._certificate is not None:
            form["client_assertion_type"] = ASSERTION_TYPE
            form["client_assertion"] = self._certificate.assertion(
                self._client_id, self.token_url)
        else:
            form["client_secret"] = self._secret
        form["scope"] = f"{self.graph_url}/.default"
        body = urllib.parse.urlencode(form).encode()
        req = urllib.request.Request(self.token_url, data=body, method="POST")
        payload = self._send(req, timeout=30.0)
        data = json.loads(payload or b"{}")
        if "access_token" not in data:
            raise GraphError("Token endpoint returned no access_token: "
                             f"{data.get('error_description') or data}")
        self._token = data["access_token"]
        self._token_expires = time.time() + int(data.get("expires_in", 3600))
        return self._token

    # -------------------------------------------------------------- plumbing
    def _send(self, req: urllib.request.Request, timeout: Optional[float] = None
              ) -> bytes:
        try:
            with urllib.request.urlopen(req, timeout=timeout or self._timeout) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:500]
            except Exception:
                pass
            raise GraphError(f"{req.get_method()} {req.full_url} -> "
                             f"HTTP {exc.code}: {detail}", status=exc.code)
        except urllib.error.URLError as exc:
            raise GraphError(f"{req.get_method()} {req.full_url} failed: "
                             f"{exc.reason}")

    def _request(self, method: str, url: str, data: Optional[bytes] = None,
                 content_type: str = "application/json",
                 extra_headers: Optional[Dict[str, str]] = None,
                 raw: bool = False) -> Any:
        """One Graph call; parsed JSON, or bytes when ``raw``."""
        if not url.startswith("http"):
            url = f"{self.graph_url}/v1.0/{url.lstrip('/')}"
        headers = {"Authorization": f"Bearer {self._get_token()}"}
        if data is not None:
            headers["Content-Type"] = content_type
        if extra_headers:
            headers.update(extra_headers)
        body = self._send(urllib.request.Request(url, data=data, headers=headers,
                                                 method=method))
        if raw:
            return body
        return json.loads(body) if body else {}

    def _site_id_of(self, site: Site) -> str:
        if site.site_id is None:
            parts = urllib.parse.urlparse(
                site.url if "://" in site.url else f"https://{site.url}")
            try:
                site.site_id = self._request(
                    "GET", f"sites/{parts.netloc}:{parts.path}")["id"]
            except GraphError as exc:
                if site.adhoc and exc.status in (401, 403, 404):
                    raise SiteNotOpened(not_opened_message(site.url,
                                                           exc.status),
                                        exc.status) from None
                raise
        return site.site_id

    @property
    def site_id(self) -> str:
        return self._site_id_of(self.sites[0])

    # ------------------------------------------- TA-35: any site a user names
    def _site_at(self, server: str, host: Optional[str] = None) -> Site:
        """The site at server path ``/sites/<name>`` (or ``/teams/...``): a
        configured one, else one a user pointed at, opened directly by its
        address (on ``host``, else the default site's host). Never the
        default site in its place."""
        low = server.rstrip("/").lower()
        for s in self.sites:
            if s.server_path == low and (host is None or _host(s.url) == host):
                return s
        host = host or _host(self.sites[0].url)
        known = self._adhoc.get(f"{host}{low}")
        if known is not None:
            return known
        url = f"https://{host}{server.rstrip('/')}"
        other = cloud_of_site(url)
        if other and not _same_cloud(other, self.cloud):
            raise SiteNotOpened(
                f"The app cannot open {url}: that site is in the "
                f"{cloud_label(other)} cloud and the app signs in to the "
                f"{cloud_label(self.cloud)} cloud. It can open only sites in "
                "its own cloud.")
        site = Site(name=_default_site_name(url), url=url, adhoc=True)
        self._adhoc[f"{host}{low}"] = site
        return site

    def _prefix_of(self, site: Site) -> str:
        """What the paths handed back carry so they find ``site`` again:
        a site on another host than the default site's keeps its whole
        address, so it is never read on the default host in its place."""
        if site is self.sites[0]:
            return ""
        if site.adhoc:
            host = _host(site.url)
            if host != _host(self.sites[0].url):
                return f"https://{host}{site.server_prefix}/"
            return f"{site.server_prefix}/"
        return f"{site.name}:"

    def _from_address(self, url: str) -> Tuple[Site, str]:
        """``(site, library path)`` a browser address or sharing link names:
        a plain address names its site (on ITS host) and path; a token link
        (``/:b:/s/...``, ``/:f:/g/...``) is read through Graph's sharing-link
        call, ``GET /shares/{u!base64url}/driveItem``."""
        if url in self._shares:
            return self._shares[url]
        parts = urllib.parse.urlsplit(url)
        host = (parts.hostname or "").lower()
        m = re.match(r"^/(sites|teams)/([^/]+)/?$",
                     urllib.parse.unquote(parts.path), re.IGNORECASE)
        if m:                                  # the site's own address
            return self._site_at(f"/{m.group(1)}/{m.group(2)}", host), ""
        try:
            from aec_document_review.sharepoint_tools import browser_url_to_path
            # Always the server-relative form, so the site is matched by
            # its name AND this address's host: the configured site's name
            # on another host is that host's site, never the default one.
            path = browser_url_to_path(url, server_relative=True)
        except Exception:                              # noqa: BLE001
            path = None
        if path and path.startswith("/"):
            segs = path.split("/")
            return self._site_at("/".join(segs[:3]), host), "/".join(segs[3:])
        return self._from_share(url)

    def _from_share(self, url: str) -> Tuple[Site, str]:
        try:
            item = self._request("GET", f"shares/{share_id(url)}/driveItem")
        except GraphError as exc:
            if exc.status in (401, 403, 404):
                # The sharing-link call itself: a site grant cannot fix it.
                raise SiteNotOpened(link_not_opened_message(exc.status),
                                    exc.status) from None
            raise
        ref = item.get("parentReference") or {}
        drive_id = ref.get("driveId")
        try:
            if drive_id and item.get("id") and not ref.get("path"):
                # Graph's sharing-link answer may leave the folder out; the
                # item read by its drive and id carries it.
                full = self._request(
                    "GET", f"drives/{drive_id}/items/{item['id']}")
                ref = full.get("parentReference") or ref
                drive_id = ref.get("driveId") or drive_id
            drive = self._request("GET", f"drives/{drive_id}") \
                if drive_id else {}
        except GraphError as exc:
            if exc.status in (401, 403, 404):
                raise SiteNotOpened(not_opened_message(url, exc.status,
                                                       link=True),
                                    exc.status) from None
            raise
        if not drive_id or "root:" not in str(ref.get("path", "")) \
                or not drive.get("webUrl"):
            # Never the library's top folder in its place.
            raise SiteNotOpened(
                "The app could not tell where that sharing link points: "
                "SharePoint did not say which folder holds it. "
                + LINK_INSTEAD)
        lib_url = str(drive.get("webUrl", "")).rstrip("/")
        site_url, _, library = lib_url.rpartition("/")
        library = urllib.parse.unquote(library)
        inside = urllib.parse.unquote(
            ref["path"].split("root:", 1)[1]).strip("/")
        name = item.get("name", "")
        site_parts = urllib.parse.urlparse(site_url)
        site = self._site_at(urllib.parse.unquote(site_parts.path),
                             (site_parts.hostname or "").lower())
        site.drive_ids.setdefault(library.lower(), drive_id)
        rel = "/".join(x for x in (library, inside, name) if x)
        self._shares[url] = (site, rel)
        return site, rel

    def _route(self, path: Optional[str]) -> Tuple[Site, str, str]:
        """``(site, library path, prefix)`` for a caller's path.

        ``name:Shared Documents/x`` is on the site called ``name``; a
        server-relative ``/sites/<site>/...`` is on the site at that
        address -- a configured one, or (TA-35) one a user pointed at,
        opened directly, never the default site in its place; a full
        address or sharing link is read for the site and path it names;
        anything else is on the default site. ``prefix`` is what the paths
        this manager hands back carry so they find the same site again:
        ``""`` on the default site, ``"name:"`` on another configured one,
        ``"/sites/<site>/"`` on one a user pointed at (its whole address
        when it is on another host)."""
        p = (path or "").strip().replace("\\", "/")
        if p.lower().startswith(("http://", "https://")):
            site, p = self._from_address(p)
            return site, p.strip("/"), self._prefix_of(site)
        site = self.sites[0]
        head, sep, tail = p.partition(":")
        if sep and "/" not in head:
            named = next((s for s in self.sites
                          if s.name.lower() == head.strip().lower()), None)
            if named is not None:
                site, p = named, tail
        for marker in ("/sites/", "/teams/"):
            if p.lower().startswith(marker):
                segs = p.split("/")
                site = self._site_at("/".join(segs[:3]))
                p = p.split("/", 3)[-1] if p.count("/") >= 3 else ""
        return site, p.strip("/"), self._prefix_of(site)

    def _clean(self, path: Optional[str]) -> str:
        """The path as this manager's results spell it: the library path,
        with the site's ``name:`` prefix when it is not the default site."""
        _site, p, prefix = self._route(path)
        return f"{prefix}{p}"

    def _split_path(self, path: str):
        """``Shared Documents/a/b.csv`` -> (drive id, ``a/b.csv``).

        The library is matched by the tail of its web URL (``Shared
        Documents``) OR its Graph name (``Documents``) — the default library
        has both, and callers use the UI spelling.
        """
        site, p, _prefix = self._route(path)
        library = p.split("/")[0] if p else "Shared Documents"
        rest = p[len(library):].strip("/") if p else ""
        key = library.lower()
        if key not in site.drive_ids:
            drives = self._request(
                "GET", f"sites/{self._site_id_of(site)}/drives")["value"]
            for d in drives:
                tail = urllib.parse.unquote(d["webUrl"].rstrip("/").split("/")[-1])
                if key in (tail.lower(), str(d.get("name", "")).lower()):
                    site.drive_ids[key] = d["id"]
                    break
            else:
                names = sorted({urllib.parse.unquote(
                    d["webUrl"].rstrip("/").split("/")[-1]) for d in drives})
                raise LibraryNotFound(f"No document library called {library!r} "
                                      f"on {site.url}. Libraries here: {names}")
        return site.drive_ids[key], rest

    def _item_url(self, drive_id: str, rest: str) -> str:
        return (f"drives/{drive_id}/root:/{urllib.parse.quote(rest)}"
                if rest else f"drives/{drive_id}/root")

    def _item(self, path: str) -> Dict[str, Any]:
        drive_id, rest = self._split_path(path)
        return self._request("GET", self._item_url(drive_id, rest))

    def _children(self, drive_id: str, item_id: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        url: Optional[str] = f"drives/{drive_id}/items/{item_id}/children"
        while url:
            page = self._request("GET", url)
            out.extend(page.get("value", []))
            url = page.get("@odata.nextLink")
        return out

    def _entry(self, item: Dict[str, Any], parent_path: str) -> Dict[str, Any]:
        """The dict shape the app's callers read (``name``/``type``/``path``/
        ``size``/``web_url``), plus Graph's own keys for anyone who wants them."""
        name = item.get("name", "")
        is_folder = "folder" in item
        return {
            "name": name,
            "type": "folder" if is_folder else "file",
            "is_folder": is_folder,
            "path": f"{parent_path}/{name}" if parent_path else name,
            "size": item.get("size", 0),
            "web_url": item.get("webUrl", ""),
            "webUrl": item.get("webUrl", ""),
            "lastModifiedDateTime": item.get("fileSystemInfo", {}).get(
                "lastModifiedDateTime", item.get("lastModifiedDateTime", "")),
            "id": item.get("id"),
        }

    # ----------------------------------------------------- the six methods
    def ls(self, path: Optional[str] = None, recursive: bool = False,
           type: str = "both", **_ignored) -> List[Dict[str, Any]]:
        """List a folder (``[]`` when it does not exist). ``type`` is
        ``files`` / ``folders`` / ``both``; ``recursive`` walks subfolders."""
        base = self._clean(path)
        try:
            drive_id, rest = self._split_path(base)
            item = self._request("GET", self._item_url(drive_id, rest))
        except GraphError as exc:
            if exc.status == 404:
                return []
            raise
        if "file" in item:
            return [self._entry(item, base.rsplit("/", 1)[0] if "/" in base else "")]
        out: List[Dict[str, Any]] = []
        for child in self._children(drive_id, item["id"]):
            entry = self._entry(child, base)
            if type == "both" or (type == "files") == (not entry["is_folder"]):
                out.append(entry)
            if recursive and entry["is_folder"]:
                out.extend(self.ls(entry["path"], recursive=True, type=type))
        return out

    def create_folder(self, folder_path: str) -> Dict[str, Any]:
        """Create the folder (and any missing parents); returns its details."""
        drive_id, rest = self._split_path(folder_path)
        parent = self._request("GET", f"drives/{drive_id}/root")
        made = parent
        for name in [p for p in rest.split("/") if p]:
            match = next((c for c in self._children(drive_id, parent["id"])
                          if "folder" in c and c["name"].lower() == name.lower()),
                         None)
            if match is None:
                match = self._request(
                    "POST", f"drives/{drive_id}/items/{parent['id']}/children",
                    data=json.dumps({"name": name, "folder": {},
                                     "@microsoft.graph.conflictBehavior": "fail"}
                                    ).encode())
            parent = made = match
        details = dict(made)
        details["web_url"] = self.get_web_url(folder_path)
        return details

    def upload_file(self, local_path: str, path: str, overwrite: bool = False,
                    **_ignored) -> bool:
        """Upload ``local_path`` to ``path``. False when the target exists and
        ``overwrite`` is False (the caller's retry-with-a-new-name relies on
        that), or when the upload fails; a missing local file raises."""
        with open(local_path, "rb") as fh:
            content = fh.read()
        drive_id, rest = self._split_path(path)
        folder, _, filename = rest.rpartition("/")
        behaviour = "replace" if overwrite else "fail"
        try:
            parent_id = self._ensure_folder(drive_id, folder)
            if len(content) <= SIMPLE_UPLOAD_MAX:
                url = (f"drives/{drive_id}/items/{parent_id}:/"
                       f"{urllib.parse.quote(filename)}:/content"
                       f"?@microsoft.graph.conflictBehavior={behaviour}")
                self._request("PUT", url, data=content,
                              content_type="application/octet-stream")
            else:
                self._upload_large(drive_id, parent_id, filename, content,
                                   behaviour)
            return True
        except GraphError as exc:
            if exc.status == 409 and not overwrite:
                return False
            log.error("SharePoint upload of %s -> %s failed: %s",
                      local_path, path, exc)
            return False

    def _ensure_folder(self, drive_id: str, folder: str) -> str:
        parent = self._request("GET", f"drives/{drive_id}/root")["id"]
        for name in [p for p in folder.split("/") if p]:
            match = next((c for c in self._children(drive_id, parent)
                          if "folder" in c and c["name"].lower() == name.lower()),
                         None)
            if match is None:
                match = self._request(
                    "POST", f"drives/{drive_id}/items/{parent}/children",
                    data=json.dumps({"name": name, "folder": {},
                                     "@microsoft.graph.conflictBehavior": "fail"}
                                    ).encode())
            parent = match["id"]
        return parent

    def _upload_large(self, drive_id: str, parent_id: str, filename: str,
                      content: bytes, behaviour: str) -> None:
        session = self._request(
            "POST", f"drives/{drive_id}/items/{parent_id}:/"
                    f"{urllib.parse.quote(filename)}:/createUploadSession",
            data=json.dumps({"item": {
                "@microsoft.graph.conflictBehavior": behaviour}}).encode())
        url, total = session["uploadUrl"], len(content)
        for start in range(0, total, CHUNK_SIZE):
            chunk = content[start:start + CHUNK_SIZE]
            end = start + len(chunk) - 1
            req = urllib.request.Request(
                url, data=chunk, method="PUT",
                headers={"Content-Range": f"bytes {start}-{end}/{total}",
                         "Content-Length": str(len(chunk))})
            self._send(req)

    def download_file(self, path: str, local_path: Optional[str] = None,
                      return_bytes: bool = True, overwrite: bool = False,
                      **_ignored) -> Union[bytes, bool]:
        """Bytes of ``path``; written to ``local_path`` when given (``True``
        back unless ``return_bytes``). An existing local file is kept unless
        ``overwrite``."""
        drive_id, rest = self._split_path(path)
        content = self._request(
            "GET", f"{self._item_url(drive_id, rest)}:/content"
            if rest else f"drives/{drive_id}/root/content", raw=True)
        if local_path:
            if os.path.exists(local_path) and not overwrite:
                return content if return_bytes else True
            os.makedirs(os.path.dirname(os.path.abspath(local_path)),
                        exist_ok=True)
            with open(local_path, "wb") as fh:
                fh.write(content)
            if not return_bytes:
                return True
        return content

    def search_filenames(self, filename_filter: Union[str, List[str]],
                         folder_path: str = "/", **_ignored
                         ) -> List[Dict[str, Any]]:
        """Files whose names match ANY filter string, under ``folder_path``."""
        filters = ([filename_filter] if isinstance(filename_filter, str)
                   else list(filename_filter))
        _site, inner, prefix = self._route(folder_path)
        drive_id, rest = self._split_path(
            f"{prefix}{inner or 'Shared Documents'}")
        library = prefix + (inner.split("/")[0] if inner
                            else "Shared Documents")
        scope = rest.lower()
        out: List[Dict[str, Any]] = []
        seen = set()
        for f in filters:
            q = urllib.parse.quote(f.replace("'", "''"))
            page = self._request(
                "GET", f"drives/{drive_id}/root/search(q='{q}')")
            for item in page.get("value", []):
                if "folder" in item or item.get("id") in seen:
                    continue
                parent = item.get("parentReference", {}).get("path", "")
                inside = urllib.parse.unquote(parent.split("/root:", 1)[-1]
                                              ).strip("/")
                if scope and not inside.lower().startswith(scope):
                    continue
                seen.add(item.get("id"))
                parent_path = "/".join(p for p in (library, inside) if p)
                out.append(self._entry(item, parent_path))
        return out

    def get_file_details(self, path: str) -> Dict[str, Any]:
        """A file's details, ``path`` in the form this manager routes (so
        a sharing link comes back as the place it points at)."""
        clean = self._clean(path)
        item = self._item(clean)
        parent = clean.rsplit("/", 1)[0] if "/" in clean else ""
        details = self._entry(item, parent)
        details["path"] = clean
        details["parentReference"] = item.get("parentReference", {})
        return details

    def get_web_url(self, path: str) -> str:
        """The browser URL of a file or folder: the site URL plus the UI path,
        ``?web=1`` so Office files open in the browser rather than download."""
        site, p, _prefix = self._route(path)
        return f"{site.url}/{urllib.parse.quote(p)}?web=1" if p \
            else f"{site.url}?web=1"


# ------------------------------------------------------- connection test
#
# TA-34 (4): the host team's admin runs this after filling in the settings.
# Each step runs in order; the first that fails stops the run, and says in
# plain words what is wrong and what the admin does, with the server's own
# error text after it.

#: Entra sign-in error codes (AADSTS...) -> what the admin does.
SIGN_IN_ADVICE: Dict[str, str] = {
    "90002": "The tenant was not found. Check GRAPH_TENANT_ID (the "
             "Directory (tenant) ID on the app registration's Overview page), "
             "and GRAPH_CLOUD: a tenant in a government cloud is not found by "
             "the commercial sign-in address, nor the other way round.",
    "900023": "GRAPH_TENANT_ID is not a valid tenant ID or domain name. Copy "
              "the Directory (tenant) ID from the app registration's "
              "Overview page.",
    "700016": "The app (client) ID was not found in this tenant. Check "
              "GRAPH_CLIENT_ID (the Application (client) ID on the Overview "
              "page), that GRAPH_TENANT_ID is the tenant it was registered "
              "in, and GRAPH_CLOUD.",
    "7000215": "The client secret is wrong. Put the secret's VALUE in "
               "GRAPH_CLIENT_SECRET (shown once, when it is created), not its "
               "Secret ID; if the value is lost, create a new secret.",
    "7000222": "The client secret has expired. Create a new one under "
               "Certificates & secrets and put its value in "
               "GRAPH_CLIENT_SECRET.",
    "700027": "The certificate was not accepted. Upload the certificate "
              "(the public .cer/.pem, not the key) under Certificates & "
              "secrets on this app registration, and check that "
              "GRAPH_CLIENT_CERT holds the same certificate and its private "
              "key.",
    "700024": "The signed sign-in request was outside its valid time. Check "
              "the host's clock.",
    "7000112": "The app registration is disabled in this tenant. An admin "
               "re-enables it under Enterprise applications.",
    "500011": "The Graph address was not recognised by this tenant: "
              "GRAPH_CLOUD (or GRAPH_API_URL) names a different cloud from "
              "the tenant's.",
    "7000218": "No credential was sent. Set GRAPH_CLIENT_SECRET or "
               "GRAPH_CLIENT_CERT.",
    # "Confidential Client isn't supported in Cross Cloud request": a
    # client-credentials sign-in sent to another cloud's sign-in address.
    "900382": "Wrong cloud: the tenant is in a different Microsoft cloud "
              "from the sign-in address. Set GRAPH_CLOUD to the tenant's "
              "cloud: commercial (also GCC), gcc_high or dod.",
    "900432": "Wrong cloud: the tenant is in a different Microsoft cloud "
              "from the sign-in address. Set GRAPH_CLOUD to the tenant's "
              "cloud: commercial (also GCC), gcc_high or dod.",
}

GRANT_ADVICE = (
    "The app has not been granted this site. Under Sites.Selected each site "
    "is granted on its own: a SharePoint administrator who is a member of "
    "the site grants it, for example with PnP PowerShell:\n"
    "    Grant-PnPAzureADAppSitePermission -AppId <GRAPH_CLIENT_ID> "
    "-DisplayName <app name> -Site <site URL> -Permissions Write\n"
    "or with Microsoft Graph PowerShell's New-MgSitePermission (the Graph "
    "call POST /sites/{site-id}/permissions; body: roles [\"write\"] and "
    "grantedToIdentities with the app's id). See docs/SHAREPOINT_SETUP.md "
    "step 3.")

WRITE_ADVICE = (
    "The app can read this site but not write to it: the grant is "
    "read-only ('read'). Grant 'write' (docs/SHAREPOINT_SETUP.md step 3), "
    "or conversations will not be saved.")

#: The probe file the connection test writes and deletes (TA-34 review 6).
PROBE_NAME = "gse-connection-check.txt"

CONSENT_ADVICE = (
    "Graph refused the app's sign-in token: the app has no Graph permission "
    "with admin consent. Add the Microsoft Graph APPLICATION permission "
    "Sites.Selected under API permissions and press \"Grant admin consent\".")


def _codes(text: str) -> Tuple[Optional[str], Optional[str]]:
    """``(AADSTS number, Graph error code)`` found in an error's text."""
    m = re.search(r"AADSTS(\d+)", text or "")
    g = re.search(r'"code"\s*:\s*"([^"]+)"', text or "")
    return (m.group(1) if m else None), (g.group(1) if g else None)


def _step(step: str, ok: bool, message: str, detail: str = "") -> Dict[str, Any]:
    return {"step": step, "ok": ok, "message": message, "detail": detail}


def started() -> bool:
    """True when any app-registration setting is filled in (so a
    diagnostics run should test the connection)."""
    return any(_setting(n) for n in (
        ENV_TENANT, ENV_CLIENT_ID, ENV_CLIENT_SECRET, ENV_CLIENT_CERT,
        ENV_SITE_URL, ENV_SITE_URLS))


def sign_in_message(exc: BaseException) -> str:
    """Plain words for a failed sign-in, the server's own text after them."""
    text = str(exc)
    aad, _graph = _codes(text)
    if aad and aad in SIGN_IN_ADVICE:
        return SIGN_IN_ADVICE[aad]
    if isinstance(exc, GraphError) and exc.status is None:
        return ("The sign-in address could not be reached: check the host's "
                "network or proxy, and GRAPH_CLOUD / GRAPH_LOGIN_URL.")
    return ("Sign-in was refused. Check GRAPH_TENANT_ID, GRAPH_CLIENT_ID, the "
            "secret or certificate, and GRAPH_CLOUD.")


def site_message(exc: BaseException, site: Site) -> str:
    """Plain words for a failed site or library request."""
    text = str(exc)
    status = getattr(exc, "status", None)
    _aad, code = _codes(text)
    if isinstance(exc, LibraryNotFound):
        return (f"The document library was not found on {site.url}. Check "
                "the first folder of GEOTECH_SHAREPOINT_ROOT (or of the path "
                "used) against the libraries listed below.")
    if status == 401:
        return CONSENT_ADVICE
    if status == 403 or code == "accessDenied":
        return GRANT_ADVICE
    if status == 404 or code == "itemNotFound":
        return (f"The site {site.url} was not found. Check the address: it "
                "is the site's own address, https://<tenant>.sharepoint.com/"
                "sites/<site> (GCC High: .sharepoint.us; DoD: "
                ".sharepoint-mil.us), without a library or folder after it.")
    if status is None:
        return ("Graph could not be reached: check the host's network or "
                "proxy, and GRAPH_CLOUD / GRAPH_API_URL.")
    return f"Graph refused the request (HTTP {status})."


def _write_probe(manager: "GraphFileManager", root: str) -> Dict[str, Any]:
    """Write a small probe file in ``root`` (created as the app's first
    save would), then delete it: the "write" step."""
    shown = f"{root.strip('/')}/{PROBE_NAME}"
    try:
        drive_id, rest = manager._split_path(root)
        parent_id = manager._ensure_folder(drive_id, rest)
        item = manager._request(
            "PUT", f"drives/{drive_id}/items/{parent_id}:/{PROBE_NAME}:/"
                   "content?@microsoft.graph.conflictBehavior=replace",
            data=(b"Written by the Document Review connection test "
                  b"(python -m aec_document_review.graph_sharepoint --check), which "
                  b"deletes it straight away.\n"),
            content_type="text/plain")
    except Exception as exc:                           # noqa: BLE001
        status = getattr(exc, "status", None)
        _aad, code = _codes(str(exc))
        message = (WRITE_ADVICE if status == 403 or code == "accessDenied"
                   else site_message(exc, manager.sites[0]))
        return _step("write", False, message, str(exc))
    try:
        manager._request("DELETE", f"drives/{drive_id}/items/{item['id']}")
    except Exception as exc:                           # noqa: BLE001
        return _step("write", True,
                     f"Wrote a probe file, {shown}, so the app can save "
                     "files; but it could not delete it: delete it by hand.",
                     str(exc))
    return _step("write", True,
                 f"Wrote and deleted a probe file, {shown}: the app can "
                 "save files.")


def check_connection(manager: Optional["GraphFileManager"] = None,
                     library_root: Optional[str] = None,
                     write: bool = True, log: bool = True
                     ) -> List[Dict[str, Any]]:
    """:func:`_check_connection`, whose report is then appended to the
    app's SharePoint log (TA-35, :mod:`aec_document_review.sharepoint_log`) unless
    ``log`` is False. Never raises."""
    results = _check_connection(manager, library_root, write)
    if log:
        try:
            from aec_document_review import sharepoint_log
            sharepoint_log.record_check(results)
        except Exception:                              # noqa: BLE001
            pass
    return results


def _check_connection(manager: Optional["GraphFileManager"] = None,
                      library_root: Optional[str] = None,
                      write: bool = True) -> List[Dict[str, Any]]:
    """Run the setup's steps in order and return one record per step:
    ``{"step", "ok", "message", "detail"}``. Stops at the first failure.

    Steps: the settings; the sign-in; each site; the document library the
    app writes to (the first folder of ``GEOTECH_SHAREPOINT_ROOT``, default
    ``Shared Documents``) on the default site; and, unless ``write`` is
    False, a write: a small probe file (:data:`PROBE_NAME`) put in
    ``GEOTECH_SHAREPOINT_ROOT`` (else the library's root) and deleted, so a
    read-only grant fails here and not at the first save. With no
    ``manager`` given, each Graph call waits at most 30 s. Never raises;
    never puts a secret or key in a message (the server's error text never
    carries one).
    """
    out: List[Dict[str, Any]] = []
    if manager is None:
        missing = missing_settings()
        if missing:
            out.append(_step("settings", False,
                             "Settings missing: " + ", ".join(missing)
                             + ". Fill them in (environment, Key Vault or "
                             ".env) and run this again."))
            return out
        try:
            manager = GraphFileManager(timeout=30.0)
        except Exception as exc:                       # noqa: BLE001
            out.append(_step("settings", False,
                             "A setting could not be used: "
                             f"{type(exc).__name__}: {exc}"))
            return out
    out.append(_step(
        "settings", True,
        f"{len(manager.sites)} site(s); sign-in with a "
        f"{manager.credential_kind}; cloud "
        f"{cloud_label(getattr(manager, 'cloud', 'commercial'))} "
        f"({getattr(manager, 'cloud_reason', '')}); sign-in address "
        f"{manager.login_url}, Graph {manager.graph_url}"))
    try:
        manager._token = None
        manager._get_token()
    except Exception as exc:                           # noqa: BLE001
        out.append(_step("sign-in", False, sign_in_message(exc), str(exc)))
        return out
    out.append(_step("sign-in", True, "Signed in as the app."))
    for site in manager.sites:
        try:
            manager._site_id_of(site)
        except Exception as exc:                       # noqa: BLE001
            out.append(_step(f"site {site.name}", False,
                             site_message(exc, site), str(exc)))
            return out
        out.append(_step(f"site {site.name}", True, f"Found {site.url}."))
    if library_root is None:
        library_root = (os.environ.get("GEOTECH_SHAREPOINT_ROOT", "").strip()
                        or "Shared Documents")
    library = library_root.strip("/").split("/")[0] or "Shared Documents"
    site = manager.sites[0]
    try:
        manager._split_path(library)
    except Exception as exc:                           # noqa: BLE001
        detail = str(exc)
        out.append(_step(f"library {library}", False,
                         site_message(exc, site), detail))
        return out
    out.append(_step(f"library {library}", True,
                     f"Found the library {library!r} on {site.url}."))
    if not write:
        skipped = _step("write", True, "Not tested (read-only check): a site "
                        "granted only 'read' passes the steps above, but the "
                        "app cannot save there.")
        skipped["skipped"] = True
        out.append(skipped)
        return out
    out.append(_write_probe(manager, library_root.strip("/") or library))
    return out


def format_check(results: List[Dict[str, Any]]) -> str:
    """The connection test as plain text, one line a step."""
    lines = []
    for r in results:
        label = "SKIP" if r.get("skipped") else ("PASS" if r["ok"] else "FAIL")
        lines.append(f"{label}  {r['step']}: {r['message']}")
        if r.get("detail"):
            lines.append(f"      Server said: {r['detail']}")
    if results and all(r["ok"] for r in results):
        lines.append("SharePoint is set up for reading; saving was not tested."
                     if any(r.get("skipped") for r in results)
                     else "SharePoint is set up.")
    return "\n".join(lines)


def site_note(manager: Any) -> str:
    """A line naming the sites, for the SharePoint tools, when there is more
    than one (``""`` otherwise, so a one-site deployment reads as before)."""
    sites = getattr(manager, "sites", None)
    if not isinstance(sites, list) or len(sites) < 2:
        return ""
    others = ", ".join(f"{s.name!r} ({s.url})" for s in sites[1:])
    return (f"SharePoint sites: the default is {sites[0].name!r} "
            f"({sites[0].url}); also {others}. A path on another site starts "
            f"with its name and a colon, e.g. '{sites[1].name}:Shared "
            "Documents/folder/file.pdf'; a path with no name is on the "
            "default site.")


def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="python -m aec_document_review.graph_sharepoint",
        description="SharePoint over Microsoft Graph with an app "
                    "registration (docs/SHAREPOINT_SETUP.md).")
    ap.add_argument("--check", action="store_true",
                    help="test the connection step by step (it writes and "
                         "deletes a small probe file, %s)" % PROBE_NAME)
    ap.add_argument("--read-only", action="store_true",
                    help="with --check: skip the write probe")
    args = ap.parse_args(argv)
    if not args.check:
        ap.print_help()
        return 2
    try:
        from aec_document_review.tinyapps_settings import export_dotenv
        export_dotenv()
    except Exception:                                  # noqa: BLE001
        pass
    results = check_connection(write=not args.read_only)
    print(format_check(results))
    try:
        from aec_document_review import sharepoint_log
        sharepoint_log.flush(30.0)
    except Exception:                                  # noqa: BLE001
        pass
    return 0 if results and all(r["ok"] for r in results) else 1


__all__ = ["GraphFileManager", "GraphError", "LibraryNotFound", "configured",
           "missing_settings", "cloud_urls", "CLOUDS", "Site", "parse_sites",
           "configured_sites", "CertificateCredential",
           "certificate_from_settings", "check_connection", "format_check",
           "site_note", "started", "PROBE_NAME", "SiteNotOpened",
           "choose_cloud", "cloud_of_site", "cloud_label", "share_id",
           "not_opened_message", "link_not_opened_message", "LINK_INSTEAD",
           "SHAREPOINT_CLOUDS",
           "ENV_TENANT", "ENV_CLIENT_ID", "ENV_CLIENT_SECRET", "ENV_SITE_URL",
           "ENV_SITE_URLS", "ENV_CLOUD", "ENV_LOGIN_URL", "ENV_API_URL",
           "ENV_CLIENT_CERT", "ENV_CLIENT_KEY", "ENV_CLIENT_KEY_PASSWORD",
           "ENV_CERT_THUMBPRINT", "LOGIN_URL", "GRAPH_URL"]


if __name__ == "__main__":                             # pragma: no cover
    raise SystemExit(main())
