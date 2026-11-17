"""Who is using the app — the signed-in person, on a deployment that says.

On Tiny Apps an IIS tier in front of the App Service performs Windows
authentication and passes the principal down as a request header,
``X-Windows-Auth-Header: DOMAIN\\user`` (CfA ``exampleCode``, ``auth.py`` and
the Streamlit starter, 2026-08/09). The app takes that header at face value:
the two controls that make it trustworthy — the App Service accepts traffic
only from the IIS servers, and the IIS module replaces any client-supplied
value — live outside this code and are load-bearing. This module therefore
never lets a user TYPE an identity; it only reads what the proxy supplies.

Elsewhere there is no header. Locally, ``DEV_IDENTITY='CORP\\jdoe'`` stands in
(their convention; it logs that it is doing so). On Databricks the launcher
captures the notebook user's email into ``GEOTECH_USER_EMAIL`` and that is the
identity. When nothing identifies the caller the result is
:data:`ANONYMOUS`, and the app keeps working in the single-user way it always
has.

What the identity is FOR here: a stable folder key so one person's
conversations, files and mirror are theirs (``Identity.key``), a display
name, and an author string for the review comments the agent writes onto a
PDF. It grants nothing — the access list on the App Service decides who may
open the app at all.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Optional

log = logging.getLogger(__name__)

#: The header IIS sets. Overridable the way CfA's module allows.
HEADER_ENV = "WINDOWS_AUTH_HEADER"
DEFAULT_HEADER = "X-Windows-Auth-Header"
#: Local stand-in principal (never set on a deployment).
DEV_IDENTITY_ENV = "DEV_IDENTITY"
#: The Databricks launcher's capture of the notebook user (5.11.0).
USER_EMAIL_ENV = "GEOTECH_USER_EMAIL"
#: Author attributed to comments the AGENT places; a deployment may override.
MARKUP_AUTHOR_ENV = "GEOTECH_MARKUP_AUTHOR"
#: Since the copy (step 13), this application's name (the core's
#: ``app_details``, handed in by the package): "Document Review (AI draft)".
DEFAULT_MARKUP_AUTHOR = "Document Review (AI draft)"

# DOMAIN\user — Windows account names cannot contain " / \ [ ] : ; | = , + * ? < >
# (and no control character). Anything else is a name: an apostrophe, an
# accented letter. Folder and secret names are made from it by rules of
# their own (``Identity.key``, ``tinyapps_keys.tester_secret_names``), which
# add a hash of the exact name where its characters do not fit (TA-39).
_PRINCIPAL = re.compile(
    r"^(?P<domain>[A-Za-z0-9._-]{1,64})\\"
    r"(?P<user>[^\"/\\\[\]:;|=,+*?<>\x00-\x1f]{1,256})$")
_UPN = re.compile(
    r"^(?P<user>[A-Za-z0-9._-]{1,256})@(?P<domain>[A-Za-z0-9._-]{1,255})$")
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
#: One part (domain or user) of a sign-in name whose folder key gives the
#: part back: the folder alphabet only, in lower case.
_PLAIN_PART = re.compile(r"[a-z0-9._-]+")
#: Hex digits of the sign-in name's hash carried by a folder key or a
#: per-tester secret name (40 bits: two names in a thousand share one with
#: a chance of about one in two million).
NAME_HASH_LEN = 10


@dataclass(frozen=True)
class Identity:
    """The caller. ``authenticated`` is False for :data:`ANONYMOUS` and
    :data:`UNIDENTIFIED` only."""
    domain: Optional[str]
    username: str
    source: str            # "header" | "dev" | "email" | "none" | "unidentified"

    @property
    def authenticated(self) -> bool:
        return self.source not in ("none", "unidentified")

    @property
    def refused(self) -> bool:
        """True for :data:`UNIDENTIFIED`: the deployed, shared host could not
        tell who is calling, and the page serves nobody it cannot name."""
        return self.source == "unidentified"

    @property
    def multi_user(self) -> bool:
        """True when a HOST identified the caller (the proxy header) — one
        process serving many people, whose conversations must be kept apart.
        ``DEV_IDENTITY`` and the Databricks launcher's email name ONE person
        in a process that serves only them: they give a display name and a
        markup author, and change no folder layout."""
        return self.source == "header"

    @property
    def qualified_name(self) -> str:
        """``DOMAIN\\user`` when a domain is known, else the bare user."""
        return f"{self.domain}\\{self.username}" if self.domain else self.username

    @property
    def display_name(self) -> str:
        """What the sidebar shows — the user part, never the domain."""
        return self.username

    @property
    def exact_name(self) -> str:
        """The sign-in name as the one spelling its hash is taken of:
        ``domain\\user``, case-folded (Windows account names are
        case-insensitive) and in Unicode's composed form (NFC), so an
        accented letter sent as one character or as a letter and a
        combining accent is one name."""
        return unicodedata.normalize("NFC", self.qualified_name.casefold())

    @property
    def name_hash(self) -> str:
        """:data:`NAME_HASH_LEN` hex digits of SHA-256 over
        :attr:`exact_name` in UTF-8: what tells two sign-in names apart
        where a folder or secret name cannot hold their own characters
        (BACKLOG TA-39)."""
        return hashlib.sha256(
            self.exact_name.encode("utf-8")).hexdigest()[:NAME_HASH_LEN]

    @property
    def legacy_key(self) -> str:
        """The folder key as it was made before TA-39: lower case, every
        run of characters outside ``A-Z a-z 0-9 . _ -`` turned into one
        ``_``, underscores trimmed from each part's ends. Several names can
        share one (``o'brien`` and ``o_brien``; any accented letter). Still
        the base of :attr:`key`, and what the per-tester secret's first
        name was made from (``tinyapps_keys.tester_suffix``)."""
        parts = [p for p in (self.domain, self.username) if p]
        return "__".join(_UNSAFE.sub("_", p.lower()).strip("_") for p in parts) \
            or "anonymous"

    @property
    def key_is_exact(self) -> bool:
        """Whether :attr:`legacy_key` gives this sign-in name back, so that
        no other name can have it: each part holds only ``a-z 0-9 . _ -``
        once in lower case, neither begins or ends with ``_`` (the old rule
        trimmed those) and neither holds ``__`` (the join between domain
        and user, and the mark before a hash)."""
        parts = [p.lower() for p in (self.domain, self.username) if p]
        return bool(parts) and all(
            _PLAIN_PART.fullmatch(p) and not p.startswith("_")
            and not p.endswith("_") and "__" not in p for p in parts)

    @property
    def key(self) -> str:
        """A folder-safe, case-insensitive key that no two sign-in names
        share: ``corp__jdoe``.

        Windows account names are case-insensitive, so two spellings of one
        person must land in one folder; the domain rides along so two
        tenants' ``jdoe`` do not.

        A name the folder alphabet can spell (:attr:`key_is_exact`: letters,
        digits, ``.``, ``_`` and ``-``) keeps the key it always had,
        ``corp__j.doe``. Any other name (an apostrophe, a space, an accented
        letter) gets :attr:`name_hash` appended after ``__``:
        ``corp__o_brien__3f9c1a2b7d``. Before TA-39 ``o'brien`` and
        ``o_brien``, or two names differing in an accent, shared one folder
        and one SharePoint mirror. An exact key holds ``__`` once at most
        and a hashed one twice or more, so the two kinds never meet.
        """
        if self.key_is_exact:
            return self.legacy_key
        # The readable part from the composed spelling the hash is taken
        # of, so one name sent in either Unicode form is one folder.
        parts = [unicodedata.normalize("NFC", p.casefold())
                 for p in (self.domain, self.username) if p]
        base = "__".join(_UNSAFE.sub("_", p).strip("_") for p in parts)
        return f"{base or 'user'}__{self.name_hash}"

    @property
    def markup_author(self) -> str:
        """The author written on review comments the agent places: the
        deployment's override if set, else ``<user> via <default author>``, so
        a reviewer opening the PDF in Bluebeam sees who asked AND that a
        model drafted it."""
        override = os.environ.get(MARKUP_AUTHOR_ENV, "").strip()
        if override:
            return override
        if not self.authenticated:
            return DEFAULT_MARKUP_AUTHOR
        return f"{self.display_name} via {DEFAULT_MARKUP_AUTHOR}"


#: The caller nobody identified.
ANONYMOUS = Identity(None, "anonymous", "none")

#: A caller the DEPLOYED host could not name: the proxy header was missing,
#: doubled or unreadable. On the App Service one process serves every tester,
#: and without a name the request would get the folder and the model key
#: every other unnamed request gets, so the page refuses it.
UNIDENTIFIED = Identity(None, "unidentified", "unidentified")

#: What the page shows a caller it refuses.
UNIDENTIFIED_MESSAGE = (
    "This app could not tell who you are: the sign-in header ({header}) was "
    "missing from this request, or could not be read. Open the app from its "
    "published address. If you did, tell the app owner: the web server in "
    "front of the app is not passing a usable header (the app's log says "
    "which).")


def _deployed() -> bool:
    """On the App Service: the platform's own ``WEBSITE_SITE_NAME`` is set
    (always, there), or ``tinyapps_settings.app_env`` is ``azure``."""
    if os.environ.get("WEBSITE_SITE_NAME", "").strip():
        return True
    try:
        from aec_document_review import tinyapps_settings
        return tinyapps_settings.app_env() == "azure"
    except Exception:                                  # noqa: BLE001
        return False


def _foundry() -> bool:
    """A Palantir Foundry deployment (``GEOTECH_DEPLOYMENT=foundry``, set by
    :mod:`aec_document_review.foundry_entry` or by whatever else starts the
    page there): one published app, one process, for everyone who opens it,
    and no header names them."""
    try:
        from aec_agent_core import engine_config
        return engine_config.is_foundry_deployment()
    except Exception:                                  # noqa: BLE001
        return False


def _bind_turns() -> None:
    """Several people share this process: from now on each turn's tools use
    the folder bound to that turn and never fall back to the process-wide
    working-folder env var, which belongs to whoever reran last (live smoke
    2c, D1). Process-wide and idempotent; best-effort."""
    try:
        from aec_agent_core._fileio import require_turn_binding
        require_turn_binding(True)
    except Exception:                                  # noqa: BLE001
        pass


def header_name() -> str:
    return os.environ.get(HEADER_ENV, "").strip() or DEFAULT_HEADER


def parse_principal(raw: Optional[str], source: str) -> Optional[Identity]:
    """``DOMAIN\\user`` or ``user@domain`` -> :class:`Identity`; else None."""
    if raw is None:
        return None
    value = raw.strip()
    if not value:
        return None
    m = _PRINCIPAL.match(value)
    if m:
        return Identity(m.group("domain"), m.group("user").strip(), source)
    m = _UPN.match(value)
    if m:
        return Identity(m.group("domain"), m.group("user"), source)
    log.warning("Unrecognised principal format from %s: %r", source, value)
    return None


def from_header_values(values: Iterable[str]) -> Optional[Identity]:
    """The identity in the proxy header, or None.

    Two values on one request means something upstream APPENDED rather than
    replaced — the IIS module should make that impossible — so neither is
    trusted (CfA's rule, kept here verbatim in spirit).
    """
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    if len(vals) > 1:
        log.warning("Multiple %s headers on one request (%r): trusting neither",
                    header_name(), vals)
        return None
    return parse_principal(vals[0], "header")


def _streamlit_header_values() -> list:
    """The proxy header from the current Streamlit request, if any.

    ``st.context.headers`` exists from Streamlit 1.37 and raises outside a
    script run; both cases come back as "no header".
    """
    try:
        import streamlit as st
        headers = st.context.headers
    except Exception:
        return []
    name = header_name()
    try:
        getter = getattr(headers, "get_all", None)
        if callable(getter):
            return list(getter(name))
        value = headers.get(name)
        return [value] if value else []
    except Exception:
        return []


def current_identity() -> Identity:
    """Resolve the caller: proxy header, else ``DEV_IDENTITY``, else the
    launcher-captured email, else :data:`ANONYMOUS`. Never raises.

    On the deployed host the header is the ONLY source: without it the
    answer is :data:`UNIDENTIFIED` (``DEV_IDENTITY`` is for a local run, and
    is not read there).

    A process that serves several people binds each turn's tools to its own
    conversation's folder (:func:`_bind_turns`): when the header names the
    caller, and on Foundry, where nobody is named (BACKLOG INF-80). Foundry
    still gets the one unnamed user, so its viewers share one list of
    conversations (OWN-152) and nobody is refused."""
    if _foundry():
        _bind_turns()
    ident = from_header_values(_streamlit_header_values())
    if ident is not None:
        if ident.multi_user:
            _bind_turns()
        return ident
    if _deployed():
        log.warning("No usable %s header on the deployed host: the request "
                    "is not served", header_name())
        return UNIDENTIFIED
    dev = os.environ.get(DEV_IDENTITY_ENV, "").strip()
    if dev:
        ident = parse_principal(dev, "dev")
        if ident is not None:
            log.warning("No %s header; using %s=%r (local development only)",
                        header_name(), DEV_IDENTITY_ENV, dev)
            return ident
    email = os.environ.get(USER_EMAIL_ENV, "").strip()
    if email:
        ident = parse_principal(email, "email")
        if ident is not None:
            return ident
    return ANONYMOUS


__all__ = ["Identity", "NAME_HASH_LEN", "ANONYMOUS", "UNIDENTIFIED", "UNIDENTIFIED_MESSAGE",
           "current_identity", "parse_principal",
           "from_header_values", "header_name", "DEFAULT_HEADER",
           "DEV_IDENTITY_ENV", "USER_EMAIL_ENV", "MARKUP_AUTHOR_ENV",
           "DEFAULT_MARKUP_AUTHOR"]
