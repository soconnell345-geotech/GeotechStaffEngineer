"""The Prompter engine for a Tiny Apps deployment — a key, not the SDK.

On Databricks the app reaches Prompter through the Funhouse SDK, which
authenticates a Windows identity with NTLM. Tiny Apps hands each app its OWN
model deployment and an API KEY for it (CfA ``exampleCode/prompter.py``,
2026-08-28), and the SDK is not installable there. Under the SDK Prompter was
always a plain OpenAI-compatible endpoint (``OpenAI(base_url=..., api_key=...)``
in its source), so the same endpoint is driven here with ``langchain-openai``
— which the Foundry route in :mod:`webapp.engine_config` already uses — and
nothing else new.

The four settings, in CfA's names, read through
:mod:`webapp.tinyapps_settings` (env var, else Key Vault on the deployment,
else a local ``.env``)::

    PROMPTER_URL         the endpoint their engineers give, e.g.
                         https://prompter.<host>/api/v1/chat/completions
                         (the OpenAI client wants the part BEFORE
                         /chat/completions; :func:`base_url_from` strips it)
    PROMPTER_MODEL       the model deployment's name — the only model the
                         pilot key serves, and the default picker entry
    PROMPTER_API_KEY     that deployment's key. Their reference sends it as
                         BOTH ``api-key`` and ``Authorization: Bearer``; so
                         does this module (the OpenAI client sends Bearer,
                         ``default_headers`` adds ``api-key``)
    PROMPTER_CA_BUNDLE   optional: the internal CA certificate, as the PEM
                         text itself (how the Key Vault secret holds it) or a
                         path to a ``.pem`` — Prompter's TLS certificate is
                         signed by the Department's own authority, so
                         without this the connection is refused

Behaviour knobs (env): ``GEOTECH_PROMPTER_DISABLE_STREAMING=1`` makes
``.stream()`` fall back to one non-streaming request, for a gateway that
rejects streaming; ``GEOTECH_WEBAPP_MAX_TOKENS`` is honoured as everywhere
else. :func:`register` installs the model builder the app already looks for
(:func:`webapp.engine_config.register_model_builder`) and publishes the
deployment's model to the sidebar picker, so nothing downstream knows which
host it is on.

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
import os
import re
import ssl
from dataclasses import dataclass
from typing import Dict, Optional

from webapp.engine_config import EngineUnavailable
from webapp.tinyapps_settings import get_setting

log = logging.getLogger(__name__)

ENV_URL = "PROMPTER_URL"
ENV_MODEL = "PROMPTER_MODEL"
ENV_KEY = "PROMPTER_API_KEY"
ENV_CA_BUNDLE = "PROMPTER_CA_BUNDLE"
DISABLE_STREAMING_ENV = "GEOTECH_PROMPTER_DISABLE_STREAMING"

#: The optional JSON secret mapping ``DOMAIN\\user`` -> ``{key, model?, url?}``.
ENV_KEYS = "PROMPTER_KEYS"
#: On a multi-user deployment, may a tester with no key of their own use the
#: shared ``PROMPTER_API_KEY``? Off unless set truthy.
SHARED_FALLBACK_ENV = "PROMPTER_SHARED_KEY_FALLBACK"
#: Between a setting's name and the tester suffix: ``PROMPTER-API-KEY--CORP-JDOE``.
#: The suffix never holds two hyphens in a row, so this split is unambiguous.
TESTER_SEPARATOR = "--"
#: Key Vault secret names: 1-127 letters, digits and hyphens.
KEY_VAULT_NAME_MAX = 127

#: Where a caller's key came from (the diagnostics panel shows THIS, never
#: the key).
SOURCE_TESTER = "per-tester"
SOURCE_MAPPING = "mapping"
SOURCE_SHARED = "shared"
SOURCE_SHARED_FALLBACK = "shared-fallback"
_SOURCE_TEXT = {
    SOURCE_TESTER: "per-tester secret",
    SOURCE_MAPPING: "PROMPTER-KEYS mapping",
    SOURCE_SHARED: "shared key (single-user host)",
    SOURCE_SHARED_FALLBACK: f"shared key ({SHARED_FALLBACK_ENV} is on)",
}

#: The banner when the deployment's own settings are incomplete — the same
#: words as :func:`webapp.engine_config.resolve_engine` uses when no builder
#: is registered at all.
NOT_CONFIGURED_MESSAGE = (
    "No model configured. This Tiny Apps deployment reads PROMPTER_URL, "
    "PROMPTER_MODEL and PROMPTER_API_KEY (and the optional "
    "PROMPTER_CA_BUNDLE) from Key Vault or the App Service app settings — "
    "locally, from the .env file. The app is running, but cannot answer "
    "questions until those are set and the app is restarted.")

#: CfA's module allows 180 s per request; the proxy in front of the app times
#: HTTP out near 120 s, but the app's turns run detached from the request.
REQUEST_TIMEOUT_S = 180.0
_CHAT_SUFFIX = "/chat/completions"

#: Retries of one model request that the provider refused as busy (429 rate
#: limit, 5xx, a dropped connection). Several testers may share ONE Prompter
#: key (the shared key, or one deployment's limits behind per-tester keys),
#: so a burst of rate limits is normal: the OpenAI client waits
#: as the provider's ``retry-after`` header asks (else an exponential backoff
#: from half a second) and tries again -- 4 retries, 5 tries in all. A turn
#: still refused after that ends with the plain-words message
#: (``webapp.core.friendly_turn_error``). Override with the env var.
MAX_RETRIES_ENV = "GEOTECH_PROMPTER_MAX_RETRIES"
DEFAULT_MAX_RETRIES = 4


def max_retries() -> int:
    """Retries per model request (``GEOTECH_PROMPTER_MAX_RETRIES``,
    default :data:`DEFAULT_MAX_RETRIES`; 0 turns them off)."""
    raw = str(os.environ.get(MAX_RETRIES_ENV, "")).strip()
    try:
        return max(0, min(int(raw), 10)) if raw else DEFAULT_MAX_RETRIES
    except ValueError:
        return DEFAULT_MAX_RETRIES


#: Refusals the OpenAI client would retry that may mean a SPENT budget.
_RETRIED_REFUSALS = (429,)


def no_retry_when_budget_spent(response) -> None:
    """httpx response hook: a refusal that says the AI budget or quota is
    used up is not retried (live smoke wave 2c, D2).

    The OpenAI client retries EVERY 429, and a gateway reports a spent
    budget as a 429 coded ``insufficient_quota``: four retries a request,
    then "wait a minute" for a stop that lasts until the budget is renewed.
    The client obeys an ``x-should-retry: false`` header, so the hook sets
    one when the body says the budget is spent
    (:func:`funhouse_agent.error_text.budget_signal`); a plain rate limit is
    left alone and still retried. Never raises."""
    try:
        if response.status_code not in _RETRIED_REFUSALS:
            return
        response.read()
        try:
            body = response.json()
        except Exception:  # noqa: BLE001 - a body that is not JSON
            body = None
        from funhouse_agent.error_text import budget_signal
        if budget_signal(response.status_code, body,
                         response.text or "") is not None:
            response.headers["x-should-retry"] = "false"
    except Exception:  # noqa: BLE001 - the client decides as it always has
        pass


@dataclass(frozen=True)
class PrompterSettings:
    url: str
    model: str
    api_key: str
    ca_bundle: Optional[str] = None

    @property
    def base_url(self) -> str:
        return base_url_from(self.url)


def base_url_from(url: str) -> str:
    """``https://h/api/v1/chat/completions?x=y`` -> ``https://h/api/v1``.

    Their engineers give the full chat-completions URL; the OpenAI client
    appends ``/chat/completions`` itself. A URL that already stops at the
    base is returned unchanged (less a trailing slash).
    """
    u = (url or "").strip().split("?", 1)[0].rstrip("/")
    if u.lower().endswith(_CHAT_SUFFIX):
        u = u[: -len(_CHAT_SUFFIX)]
    return u.rstrip("/")


def ssl_context(bundle: Optional[str]) -> Optional[ssl.SSLContext]:
    """Trust the internal CA named by the bundle — PEM text or a file path.

    None means "default trust" (no bundle given). The partial-chain flag lets
    the internal CA act as a trust anchor although it is not a self-signed
    root, which is what browsers do too (their ``_ssl_context``, kept).
    """
    b = (bundle or "").strip()
    if not b:
        return None
    if b.startswith("-----BEGIN"):
        ctx = ssl.create_default_context(cadata=b)
    elif os.path.exists(b):
        ctx = ssl.create_default_context(cafile=b)
    else:
        raise ValueError(
            f"{ENV_CA_BUNDLE} is neither PEM text nor a path that exists")
    ctx.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN
    return ctx


def settings() -> Optional[PrompterSettings]:
    """The four values, or None when the deployment has not been given them.

    Missing values are not an error here: the app must boot and show its
    "no engine configured" banner rather than crash on a reviewer's first
    click. A CA bundle that is set but unusable IS raised, because a wrong
    bundle is a misconfiguration to fix, not an absence to tolerate.
    """
    url = (get_setting(ENV_URL) or "").strip()
    model = (get_setting(ENV_MODEL) or "").strip()
    key = (get_setting(ENV_KEY) or "").strip()
    if not (url and model and key):
        return None
    return PrompterSettings(url=url, model=model, api_key=key,
                            ca_bundle=(get_setting(ENV_CA_BUNDLE) or "").strip()
                            or None)


def configured() -> bool:
    return settings() is not None


# --------------------------------------------------------- per-tester keys

class NoPrompterKey(EngineUnavailable):
    """A signed-in tester with no AI key of their own (and no shared-key
    fallback): the polite refusal the app shows as written."""


@dataclass(frozen=True)
class ResolvedPrompter:
    """The Prompter settings that serve ONE caller, and where the key came
    from. ``source`` names the source only (``SOURCE_*``); ``secret_name``
    is the per-tester secret's NAME — never a value."""
    settings: PrompterSettings
    source: str
    own_model: bool = False
    secret_name: Optional[str] = None

    @property
    def source_text(self) -> str:
        return _SOURCE_TEXT.get(self.source, self.source)


def _text(name: Optional[str]) -> str:
    """One setting as stripped text; "" when unnamed or unset."""
    if not name:
        return ""
    return str(get_setting(name) or "").strip()


def _truthy(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


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


def _streaming_disabled() -> bool:
    return str(os.environ.get(DISABLE_STREAMING_ENV, "")).strip().lower() in (
        "1", "true", "yes", "on")


def build_chat_model(model_id: Optional[str] = None, *,
                     prompter: Optional[PrompterSettings] = None):
    """A ``ChatOpenAI`` against the deployment's Prompter endpoint.

    ``model_id`` is the picker's choice; on the pilot's one-model key it can
    only ever be the deployment's own name, but the parameter keeps the
    builder shape the picker expects. Sends ``max_completion_tokens`` rather
    than ``max_tokens`` for the same reason the Foundry route does: the
    reasoning-model tiers reject the old name and a deployment name gives
    langchain-openai no hint. Its value is the app's output cap
    (``engine_config.DEFAULT_MAX_TOKENS``, 32,000 since live smoke wave 2b:
    on GPT-5.1 it holds the reasoning tokens as well as the reply). Busy
    refusals are retried :func:`max_retries` times, honouring retry-after;
    a refusal that says the budget is spent is not
    (:func:`no_retry_when_budget_spent`).
    """
    import httpx
    from langchain_openai import ChatOpenAI
    from webapp.engine_config import _default_max_tokens

    ps = prompter or settings()
    if ps is None:
        raise RuntimeError(
            f"Prompter is not configured: set {ENV_URL}, {ENV_MODEL} and "
            f"{ENV_KEY} (Key Vault secrets / app settings on the deployment, "
            "a .env file locally).")
    ctx = ssl_context(ps.ca_bundle)
    client = httpx.Client(verify=ctx if ctx is not None else True,
                          timeout=REQUEST_TIMEOUT_S,
                          event_hooks={"response":
                                       [no_retry_when_budget_spent]})
    return ChatOpenAI(
        model=model_id or ps.model,
        api_key=ps.api_key,
        base_url=ps.base_url,
        default_headers={"api-key": ps.api_key},
        http_client=client,
        max_completion_tokens=_default_max_tokens(),
        max_retries=max_retries(),
        disable_streaming=_streaming_disabled(),
    )


def build_for(ident=None, model_id: Optional[str] = None):
    """A chat model for ONE caller, with that caller's key
    (:func:`resolve_for`). Built afresh on every call — no client is cached,
    so no user's client can be handed to another. A tester's own model
    (``PROMPTER-MODEL--<SUFFIX>`` / the mapping's ``model``) wins over the
    picker's choice: a key serves its one deployment. Raises the polite
    refusals of :func:`resolve_for`."""
    rp = resolve_for(ident)
    log.info("Prompter key for %s: %s",
             getattr(ident, "qualified_name", None) or "the current caller",
             rp.source_text)
    return build_chat_model(None if rp.own_model else model_id,
                            prompter=rp.settings)


def tester_keys_possible() -> bool:
    """Could a tester be served although the shared set is incomplete? —
    the shared URL is there (their keys hang off it) or the mapping is."""
    return bool(_text(ENV_URL) or _text(ENV_KEYS))


def register() -> bool:
    """Install the Prompter builder when the deployment has settings to
    serve anyone: the shared set, or what per-tester keys need
    (:func:`tester_keys_possible`).

    Returns True when registered. The builder resolves the key for the
    caller it is handed each time (:func:`build_for`); it holds no key
    itself. Also publishes the deployment's model to the sidebar picker
    (``GEOTECH_PROMPTER_MODELS``) unless the deployment set its own list,
    and records the choice as the default model.
    """
    ps = settings()
    if ps is None and not tester_keys_possible():
        return False
    from webapp.engine_config import register_model_builder

    def _build(model_id: Optional[str] = None, identity=None):
        return build_for(identity, model_id)

    register_model_builder(_build)
    model = ps.model if ps is not None else _text(ENV_MODEL)
    if model:
        os.environ.setdefault("GEOTECH_PROMPTER_MODELS", f"{model}={model}")
        os.environ.setdefault("GEOTECH_WEBAPP_MODEL", model)
    return True


__all__ = ["PrompterSettings", "settings", "configured", "build_chat_model",
           "register", "base_url_from", "ssl_context",
           "ENV_URL", "ENV_MODEL", "ENV_KEY", "ENV_CA_BUNDLE",
           "DISABLE_STREAMING_ENV", "REQUEST_TIMEOUT_S", "MAX_RETRIES_ENV",
           "DEFAULT_MAX_RETRIES", "max_retries",
           "no_retry_when_budget_spent",
           # per-tester keys
           "ENV_KEYS", "SHARED_FALLBACK_ENV", "TESTER_SEPARATOR",
           "KEY_VAULT_NAME_MAX", "SOURCE_TESTER", "SOURCE_MAPPING",
           "SOURCE_SHARED", "SOURCE_SHARED_FALLBACK", "NOT_CONFIGURED_MESSAGE",
           "NoPrompterKey", "ResolvedPrompter", "resolve_for", "build_for",
           "tester_suffix", "tester_secret_name", "key_mapping",
           "shared_fallback_allowed", "no_key_message", "key_source_report",
           "tester_keys_possible"]
