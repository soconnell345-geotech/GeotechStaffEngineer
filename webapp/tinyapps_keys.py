"""Whose Prompter key serves a caller on a Tiny Apps deployment: the page's
half of the Prompter route.

The Prompter connection itself moved into the shared core
(:mod:`aec_agent_core.tinyapps_engine`; BACKLOG INF-32, the app split's
step 11 (a)), and its old name ``webapp.tinyapps_engine`` is an alias of it.
What names a Department deployment stayed with the page (plan
``module_work/APP_SPLIT_PLAN.md`` section 1.2): the settings read through
Key Vault (:mod:`webapp.tinyapps_settings`), the sign-in name
(:mod:`webapp.identity`) and the per-tester keys below. ``webapp/__init__.py``
hands them to the core as its hooks ``prompter.get_setting``,
``prompter.resolve_for`` and ``prompter.tester_keys_possible``. Document
Review's page takes this file over at step 13.

**Per-tester keys** (owner, 2026-10-09). On the published deployment each
tester gets an AI key, and so a budget, of their own. For a caller the
proxy header identified (``Identity.multi_user``), :func:`resolve_for` takes,
first that applies:

1. the per-tester secret ``PROMPTER-API-KEY--<SUFFIX>`` (optionally
   ``PROMPTER-MODEL--<SUFFIX>`` and ``PROMPTER-URL--<SUFFIX>``, else the
   shared ``PROMPTER-MODEL`` / ``PROMPTER-URL``). ``<SUFFIX>`` is the
   identity key in Key Vault's alphabet (:func:`tester_suffix`):
   ``CORP\\jdoe`` -> ``CORP-JDOE``, every run of characters other than a
   letter or digit -> one hyphen. As an App Service setting or a ``.env``
   line the same name is spelt with underscores
   (``PROMPTER_API_KEY__CORP_JDOE``);
2. the JSON secret ``PROMPTER-KEYS``:
   ``{"CORP\\\\jdoe": {"key": "...", "model": "...", "url": "..."}}`` —
   identities matched case-insensitively, ``model`` / ``url`` optional, a
   bare string value taken as the key;
3. the shared ``PROMPTER-API-KEY``, ONLY when
   ``PROMPTER_SHARED_KEY_FALLBACK`` is on (``1``/``true``/``yes``/``on``).

Nothing found is a polite refusal (:class:`NoPrompterKey`, shown as written:
"No AI key is set up for <name> yet — ask the app owner"), so the key list
is the pilot's access list. A single-user host (``DEV_IDENTITY`` on dosdev,
a laptop, Databricks, or no header at all) uses the shared key, as before.
The builder resolves the key for the caller it is handed EVERY time it is
called and caches no client: each conversation's agent is built with its
own user's key. Missing secrets are not cached
(:func:`webapp.tinyapps_settings.get_setting`), so a tester added to the
vault is served at their next new conversation, without a restart; a
rotated key is read at the next restart, as the shared key always was. No
key is ever logged or shown — only its SOURCE (:func:`key_source_report`).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Dict, Optional

from aec_agent_core.engine_config import EngineUnavailable
from aec_agent_core.tinyapps_engine import (
    ENV_CA_BUNDLE, ENV_KEY, ENV_MODEL, ENV_URL, NOT_CONFIGURED_MESSAGE,
    SHARED_FALLBACK_ENV, SOURCE_MAPPING, SOURCE_SHARED, SOURCE_SHARED_FALLBACK,
    SOURCE_TESTER,
    NoPrompterKey, PrompterSettings, ResolvedPrompter, _truthy,
    settings)
from webapp.tinyapps_settings import get_setting

log = logging.getLogger(__name__)

#: The optional JSON secret mapping ``DOMAIN\\user`` -> ``{key, model?, url?}``.
ENV_KEYS = "PROMPTER_KEYS"
#: Between a setting's name and the tester suffix: ``PROMPTER-API-KEY--CORP-JDOE``.
#: The suffix never holds two hyphens in a row, so this split is unambiguous.
TESTER_SEPARATOR = "--"
#: Key Vault secret names: 1-127 letters, digits and hyphens.
KEY_VAULT_NAME_MAX = 127


def _text(name: Optional[str]) -> str:
    """One setting as stripped text; "" when unnamed or unset."""
    if not name:
        return ""
    return str(get_setting(name) or "").strip()


def shared_fallback_allowed() -> bool:
    """``PROMPTER_SHARED_KEY_FALLBACK`` is on (env, Key Vault or ``.env``)."""
    return _truthy(get_setting(SHARED_FALLBACK_ENV))


def tester_suffix(ident) -> str:
    """The identity key in Key Vault's alphabet: ``corp__j.doe`` ->
    ``CORP-J-DOE``. Every run of characters other than a letter or a digit
    becomes ONE hyphen, ends trimmed; upper case, the spelling
    :func:`webapp.tinyapps_settings.get_setting` asks the vault for. Two
    names that differ only in punctuation (``j.doe`` / ``j_doe``) share a
    suffix: give such testers a ``PROMPTER-KEYS`` entry instead."""
    key = str(getattr(ident, "key", "") or "")
    return re.sub(r"[^A-Za-z0-9]+", "-", key).strip("-").upper()


def tester_secret_name(setting: str, ident) -> Optional[str]:
    """``PROMPTER-API-KEY--CORP-JDOE`` for ``setting="PROMPTER_API_KEY"`` and
    ``CORP\\jdoe``; None when the identity gives no suffix or the name would
    pass Key Vault's 127-character limit (then only ``PROMPTER-KEYS`` can
    serve that person)."""
    suffix = tester_suffix(ident)
    if not suffix:
        return None
    name = (setting.replace("_", "-").upper() + TESTER_SEPARATOR + suffix)
    return name if len(name) <= KEY_VAULT_NAME_MAX else None


def _normal_principal(text: str) -> str:
    """``CORP\\JDoe`` / ``jdoe@corp`` -> ``corp\\jdoe`` (case-insensitive,
    the spelling ``Identity.qualified_name`` gives)."""
    t = str(text or "").strip()
    if "\\" not in t and "@" in t:
        user, _, domain = t.rpartition("@")
        t = f"{domain}\\{user}"
    return t.lower()


def key_mapping() -> Dict[str, dict]:
    """``PROMPTER_KEYS`` parsed: ``{normalised principal: {key, model, url}}``.

    A value that is not JSON, or an entry without a key, is left out with a
    warning that names WHERE the fault is and never quotes the text (it holds
    keys). ``{}`` when the setting is absent."""
    raw = _text(ENV_KEYS)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError as exc:
        log.warning("%s is not valid JSON (%s at character %s); ignored",
                    ENV_KEYS, getattr(exc, "msg", "parse error"),
                    getattr(exc, "pos", "?"))
        return {}
    if not isinstance(data, dict):
        log.warning("%s must be a JSON object of identity -> key; ignored",
                    ENV_KEYS)
        return {}
    out: Dict[str, dict] = {}
    for who, entry in data.items():
        if isinstance(entry, str):
            entry = {"key": entry}
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("key") or entry.get("api_key") or "").strip()
        if not key:
            log.warning("%s: the entry for %r has no key; ignored",
                        ENV_KEYS, who)
            continue
        out[_normal_principal(who)] = {
            "key": key,
            "model": str(entry.get("model") or "").strip(),
            "url": str(entry.get("url") or "").strip()}
    return out


def no_key_message(ident, secret_name: Optional[str]) -> str:
    """The polite refusal: who, what to do, and — for the app owner — the
    secret NAME that would let this person in. Never a key."""
    who = getattr(ident, "display_name", "") or "you"
    msg = f"No AI key is set up for **{who}** yet — ask the app owner."
    if secret_name:
        msg += (f" (For the app owner: this person's key goes in the Key "
                f"Vault secret `{secret_name}`.)")
    return msg


def _complete(ident, source: str, key: str, url: str, model: str,
              own_model: bool, secret_name: Optional[str]) -> ResolvedPrompter:
    if not (url and model):
        gaps = [n for n, v in (("the endpoint URL", url),
                               ("the model name", model)) if not v]
        raise EngineUnavailable(
            f"An AI key is set up for **{ident.display_name}**, but "
            f"{' and '.join(gaps)} {'are' if len(gaps) > 1 else 'is'} "
            "missing — ask the app owner. (For the app owner: set the "
            "shared PROMPTER-URL / PROMPTER-MODEL secrets, or this person's "
            "own.)")
    ca = _text(ENV_CA_BUNDLE) or None
    return ResolvedPrompter(
        PrompterSettings(url=url, model=model, api_key=key, ca_bundle=ca),
        source, own_model=own_model, secret_name=secret_name)


def resolve_for(ident=None) -> ResolvedPrompter:
    """The Prompter settings for ``ident`` (default: the current caller).

    Single-user (no proxy header): the shared settings, as before. A caller
    the proxy identified: the per-tester secret, then ``PROMPTER-KEYS``, then
    — only when ``PROMPTER_SHARED_KEY_FALLBACK`` is on — the shared key.
    Raises :class:`NoPrompterKey` (or :class:`EngineUnavailable` for an
    incomplete set) — the polite refusals; never returns a half-filled set.
    """
    if ident is None:
        from webapp.identity import current_identity
        ident = current_identity()
    if not getattr(ident, "multi_user", False):
        ps = settings()
        if ps is None:
            raise EngineUnavailable(NOT_CONFIGURED_MESSAGE)
        return ResolvedPrompter(ps, SOURCE_SHARED)

    shared_url, shared_model = _text(ENV_URL), _text(ENV_MODEL)
    key_name = tester_secret_name(ENV_KEY, ident)
    # 1) this tester's own secret
    key = _text(key_name)
    if key:
        own_url = _text(tester_secret_name(ENV_URL, ident))
        own_model = _text(tester_secret_name(ENV_MODEL, ident))
        return _complete(ident, SOURCE_TESTER, key, own_url or shared_url,
                         own_model or shared_model, bool(own_model), key_name)
    # 2) the JSON mapping
    entry = key_mapping().get(_normal_principal(ident.qualified_name))
    if entry:
        return _complete(ident, SOURCE_MAPPING, entry["key"],
                         entry["url"] or shared_url,
                         entry["model"] or shared_model, bool(entry["model"]),
                         key_name)
    # 3) the shared key, only when the deployment allows it
    if shared_fallback_allowed():
        ps = settings()
        if ps is not None:
            return ResolvedPrompter(ps, SOURCE_SHARED_FALLBACK,
                                    secret_name=key_name)
    raise NoPrompterKey(no_key_message(ident, key_name))


def key_source_report(ident=None) -> tuple:
    """``(ok, text)`` for the diagnostics panel: which key SOURCE serves the
    caller, and the per-tester secret's NAME — never a key."""
    if ident is None:
        from webapp.identity import current_identity
        ident = current_identity()
    who = getattr(ident, "qualified_name", "") or "nobody"
    mode = ("multi-user (identity header)"
            if getattr(ident, "multi_user", False) else "single-user")
    name = (tester_secret_name(ENV_KEY, ident)
            if getattr(ident, "multi_user", False) else None)
    tail = f"; per-tester secret name: {name}" if name else ""
    try:
        rp = resolve_for(ident)
    except EngineUnavailable as exc:
        return False, (f"{who}, {mode}: no key — {exc}")
    except Exception as exc:  # noqa: BLE001 - a vault outage, say which
        return False, (f"{who}, {mode}: key lookup failed: "
                       f"{type(exc).__name__}: {exc}")
    model = rp.settings.model + (" (this tester's own)" if rp.own_model
                                 else "")
    return True, (f"{who}, {mode}: key source = {rp.source_text}; "
                  f"model {model}{tail}")


def tester_keys_possible() -> bool:
    """Could a tester be served although the shared set is incomplete? —
    the shared URL is there (their keys hang off it) or the mapping is."""
    return bool(_text(ENV_URL) or _text(ENV_KEYS))


__all__ = ["ENV_KEYS", "SHARED_FALLBACK_ENV", "TESTER_SEPARATOR",
           "KEY_VAULT_NAME_MAX", "NoPrompterKey", "ResolvedPrompter",
           "resolve_for", "tester_suffix", "tester_secret_name",
           "key_mapping", "shared_fallback_allowed", "no_key_message",
           "key_source_report", "tester_keys_possible", "get_setting"]
