"""Whose Prompter key serves a caller on a Tiny Apps deployment: the page's
half of the Prompter route.

The Prompter connection itself moved into the shared core
(:mod:`aec_agent_core.tinyapps_engine`; BACKLOG INF-32, the app split's
step 11 (a)), and its old name ``aec_agent_core.tinyapps_engine`` is an alias of it.
What names a Department deployment stayed with the page (plan
``module_work/APP_SPLIT_PLAN.md`` section 1.2): the settings read through
Key Vault (:mod:`aec_document_review.tinyapps_settings`), the sign-in name
(:mod:`aec_document_review.identity`) and the per-tester keys below. ``aec_document_review/__init__.py``
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
   sign-in name in Key Vault's alphabet (:func:`tester_suffix`):
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

**Names that differ only in punctuation or accents** (BACKLOG TA-39).
``<SUFFIX>`` cannot tell ``corp\\j.doe``, ``corp\\j-doe`` and ``corp\\j_doe``
apart (all ``CORP-J-DOE``), nor ``o'brien`` from ``o_brien``, nor a name
from the same name with an accent. So a sign-in name that holds anything
other than the letters A-Z and digits, in its domain or its user part, has
an EXACT secret name as well, and that one is looked up first::

    PROMPTER-API-KEY--<SUFFIX>--<HASH>      (PROMPTER_API_KEY__<SUFFIX>__<HASH>)

``<HASH>`` is the first ten hex digits, in upper case, of SHA-256 over the
sign-in name spelt ``domain\\user``, case-folded, in UTF-8
(``Identity.name_hash``). For such a name the order is: the exact secret
(with ``PROMPTER-MODEL--<SUFFIX>--<HASH>`` / ``PROMPTER-URL--...`` beside
it), then the ``PROMPTER-KEYS`` entry, which is matched on the exact name
too, then the secret named by ``<SUFFIX>`` alone, so a tester who had a key
before TA-39 still has it. A name of letters and digits only has one secret
name and the order above, unchanged.

To give two such testers different keys: give EACH an exact secret (or a
``PROMPTER-KEYS`` entry), and delete the secret named by ``<SUFFIX>`` alone
once both are served; while it exists, whoever of them has no exact secret
and no mapping entry is served by it. The exact name is shown, never a key:
in the refusal a tester without a key sees ("this person's key goes in the
Key Vault secret ..."), in the sidebar's Connection diagnostics, and by::

    python -m aec_document_review.tinyapps_keys "CORP\\j.doe"

Nothing found is a polite refusal (:class:`NoPrompterKey`, shown as written:
"No AI key is set up for <name> yet — ask the app owner"), so the key list
is the pilot's access list. A single-user host (``DEV_IDENTITY`` on dosdev,
a laptop, Databricks, or no header at all) uses the shared key, as before.
The builder resolves the key for the caller it is handed EVERY time it is
called and caches no client: each conversation's agent is built with its
own user's key. Missing secrets are not cached
(:func:`aec_document_review.tinyapps_settings.get_setting`), so a tester added to the
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
    no_key_message, settings)
from aec_document_review.tinyapps_settings import get_setting

log = logging.getLogger(__name__)

#: The optional JSON secret mapping ``DOMAIN\\user`` -> ``{key, model?, url?}``.
ENV_KEYS = "PROMPTER_KEYS"
#: Between a setting's name and the tester suffix: ``PROMPTER-API-KEY--CORP-JDOE``.
#: The suffix never holds two hyphens in a row, so this split is unambiguous.
#: It also stands before the hash of an exact name (TA-39):
#: ``PROMPTER-API-KEY--CORP-J-DOE--<HASH>``.
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
    """The sign-in name in Key Vault's alphabet: ``corp\\j.doe`` (folder key
    ``corp__j.doe``) -> ``CORP-J-DOE``. Every run of characters other than a
    letter or a digit becomes ONE hyphen, ends trimmed; upper case, the
    spelling :func:`aec_document_review.tinyapps_settings.get_setting` asks
    the vault for. Made from the folder key as it was before TA-39
    (``Identity.legacy_key``), so no secret name that resolved then has
    moved. Two names that differ only in punctuation or accents (``j.doe``
    / ``j_doe``) share a suffix: :func:`tester_exact_secret_name` tells
    them apart."""
    key = getattr(ident, "legacy_key", None)
    if key is None:
        key = getattr(ident, "key", "")
    return re.sub(r"[^A-Za-z0-9]+", "-", str(key or "")).strip("-").upper()


def tester_secret_name(setting: str, ident) -> Optional[str]:
    """``PROMPTER-API-KEY--CORP-JDOE`` for ``setting="PROMPTER_API_KEY"`` and
    ``CORP\\jdoe``; None when the identity gives no suffix or the name would
    pass Key Vault's 127-character limit (then the exact secret name or
    ``PROMPTER-KEYS`` serves that person)."""
    suffix = tester_suffix(ident)
    if not suffix:
        return None
    name = (setting.replace("_", "-").upper() + TESTER_SEPARATOR + suffix)
    return name if len(name) <= KEY_VAULT_NAME_MAX else None


def suffix_is_exact(ident) -> bool:
    """Whether :func:`tester_suffix` belongs to this sign-in name alone: its
    domain and user part hold only the letters A-Z and digits (case apart),
    so the suffix is ``DOMAIN-USER`` and reads back one way. Any other
    character became a hyphen that another name's character also becomes."""
    parts = [str(p) for p in (getattr(ident, "domain", None),
                              getattr(ident, "username", None)) if p]
    return bool(parts) and all(
        re.fullmatch(r"[A-Za-z0-9]+", p) for p in parts)


def tester_exact_secret_name(setting: str, ident) -> Optional[str]:
    """``PROMPTER-API-KEY--CORP-J-DOE--<HASH>``: the secret name only this
    sign-in name has (BACKLOG TA-39), for a name :func:`tester_suffix`
    cannot tell from another; None for a name of letters and digits only
    (its one name is :func:`tester_secret_name`) and for an identity with
    no name hash. ``<HASH>`` is ``Identity.name_hash`` in upper case. The
    readable part is cut where the whole would pass Key Vault's 127
    characters: the hash carries the identity."""
    if suffix_is_exact(ident):
        return None
    digest = str(getattr(ident, "name_hash", "") or "").upper()
    if not digest:
        return None
    head = setting.replace("_", "-").upper() + TESTER_SEPARATOR
    tail = TESTER_SEPARATOR + digest
    room = KEY_VAULT_NAME_MAX - len(head) - len(tail)
    suffix = tester_suffix(ident)[:max(room, 0)].strip("-") or "X"
    name = head + suffix + tail
    return name if len(name) <= KEY_VAULT_NAME_MAX else None


def tester_secret_names(setting: str, ident) -> list:
    """Every secret name ``setting`` is looked up under for ``ident``, in
    the order tried: the exact name first where there is one, then the
    name by suffix alone."""
    names = [tester_exact_secret_name(setting, ident),
             tester_secret_name(setting, ident)]
    return [n for n in names if n]


def advertised_secret_name(ident) -> Optional[str]:
    """The key secret's name the app owner is TOLD for ``ident`` (the
    refusal, the diagnostics): the exact name where there is one, so a new
    tester is never provisioned under a name another tester shares."""
    names = tester_secret_names(ENV_KEY, ident)
    return names[0] if names else None


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


# no_key_message (the polite refusal) is the core's, so a page that forgets
# the resolve_for hook refuses a tester in the same words (step 11 (a)
# review); it stays importable from here.


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
        from aec_document_review.identity import current_identity
        ident = current_identity()
    if not getattr(ident, "multi_user", False):
        ps = settings()
        if ps is None:
            raise EngineUnavailable(NOT_CONFIGURED_MESSAGE)
        return ResolvedPrompter(ps, SOURCE_SHARED)

    shared_url, shared_model = _text(ENV_URL), _text(ENV_MODEL)
    key_name = advertised_secret_name(ident)

    def _own_secret(make_name):
        name = make_name(ENV_KEY, ident)
        key = _text(name)
        if not key:
            return None
        own_url = _text(make_name(ENV_URL, ident))
        own_model = _text(make_name(ENV_MODEL, ident))
        return _complete(ident, SOURCE_TESTER, key, own_url or shared_url,
                         own_model or shared_model, bool(own_model), name)

    def _mapped():
        entry = key_mapping().get(_normal_principal(ident.qualified_name))
        if not entry:
            return None
        return _complete(ident, SOURCE_MAPPING, entry["key"],
                         entry["url"] or shared_url,
                         entry["model"] or shared_model, bool(entry["model"]),
                         key_name)

    # 1) this tester's own secret, 2) the JSON mapping. A name the suffix
    # cannot tell from another's (TA-39) is asked for by its exact secret
    # name and its exact mapping entry BEFORE the secret it may share.
    if tester_exact_secret_name(ENV_KEY, ident):
        steps = (lambda: _own_secret(tester_exact_secret_name), _mapped,
                 lambda: _own_secret(tester_secret_name))
    else:
        steps = (lambda: _own_secret(tester_secret_name), _mapped)
    for step in steps:
        found = step()
        if found is not None:
            return found
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
        from aec_document_review.identity import current_identity
        ident = current_identity()
    who = getattr(ident, "qualified_name", "") or "nobody"
    mode = ("multi-user (identity header)"
            if getattr(ident, "multi_user", False) else "single-user")
    names = (tester_secret_names(ENV_KEY, ident)
             if getattr(ident, "multi_user", False) else [])
    tail = f"; per-tester secret name: {names[0]}" if names else ""
    if len(names) > 1:
        tail += (f" (this sign-in name's own; {names[1]} can be shared with "
                 "a name that differs only in punctuation or accents, and "
                 "is tried last)")
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
           "tester_exact_secret_name", "tester_secret_names",
           "suffix_is_exact", "advertised_secret_name",
           "key_mapping", "shared_fallback_allowed", "no_key_message",
           "key_source_report", "tester_keys_possible", "get_setting"]


def _main(argv=None) -> int:
    """``python -m aec_document_review.tinyapps_keys "CORP\\j.doe" ...``:
    print, for each sign-in name, its folder and the secret names its key
    is looked up under, in order. Reads no setting and prints no key."""
    import sys
    from aec_document_review.identity import parse_principal
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print('usage: python -m aec_document_review.tinyapps_keys '
              '"DOMAIN\\user" [...]')
        return 2
    bad = 0
    for raw in args:
        ident = parse_principal(raw, "header")
        if ident is None:
            print(f"{raw}: not a sign-in name (DOMAIN\\user or user@domain)")
            bad = 1
            continue
        print(f"{ident.qualified_name}")
        print(f"  folder: users/{ident.key}/")
        for n in tester_secret_names(ENV_KEY, ident):
            print(f"  key secret: {n}")
        print(f"  {ENV_KEYS} entry: {_normal_principal(ident.qualified_name)}")
    return bad


if __name__ == "__main__":
    raise SystemExit(_main())
