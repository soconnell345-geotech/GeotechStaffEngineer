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

import logging
import os
import re
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
DEFAULT_MARKUP_AUTHOR = "GeotechStaffEngineer (AI draft)"

# DOMAIN\user — Windows account names cannot contain " / \ [ ] : ; | = , + * ? < >
_PRINCIPAL = re.compile(
    r"^(?P<domain>[A-Za-z0-9._-]{1,64})\\(?P<user>[A-Za-z0-9._@ -]{1,256})$")
_UPN = re.compile(
    r"^(?P<user>[A-Za-z0-9._-]{1,256})@(?P<domain>[A-Za-z0-9._-]{1,255})$")
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass(frozen=True)
class Identity:
    """The caller. ``authenticated`` is False for :data:`ANONYMOUS` only."""
    domain: Optional[str]
    username: str
    source: str            # "header" | "dev" | "email" | "none"

    @property
    def authenticated(self) -> bool:
        return self.source != "none"

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
    def key(self) -> str:
        """A folder-safe, case-insensitive key: ``corp__jdoe``.

        Windows account names are case-insensitive, so two spellings of one
        person must land in one folder; the domain rides along so two
        tenants' ``jdoe`` do not.
        """
        parts = [p for p in (self.domain, self.username) if p]
        return "__".join(_UNSAFE.sub("_", p.lower()).strip("_") for p in parts) \
            or "anonymous"

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
    launcher-captured email, else :data:`ANONYMOUS`. Never raises."""
    ident = from_header_values(_streamlit_header_values())
    if ident is not None:
        return ident
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


__all__ = ["Identity", "ANONYMOUS", "current_identity", "parse_principal",
           "from_header_values", "header_name", "DEFAULT_HEADER",
           "DEV_IDENTITY_ENV", "USER_EMAIL_ENV", "MARKUP_AUTHOR_ENV",
           "DEFAULT_MARKUP_AUTHOR"]
