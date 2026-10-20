"""Engine resolution for the geotech web chat app — env-driven, graceful.

The deep agent (``funhouse_agent.deep.agent.build_deep_agent``) needs a chat
MODEL. This module decides which one to build from the environment, WITHOUT ever
crashing the app when nothing is configured — a TinyApp reviewer may click the
app cold, and it must render a clear "no engine configured" banner rather than
throw.

Resolution order (first that applies wins):

1. **Injected builder** — a deployment (e.g. the TinyApp "Prompter" data
   connection) calls :func:`register_model_builder` at startup with a zero-arg
   callable that returns a LangChain-compatible chat model (a
   ``PrompterChatModel`` / ``fh_prompter`` wrapper the deployment supplies). We
   do NOT invent a Prompter HTTP client here — the deployment environment ships
   the SDK and provides the model object through this hook.
2. **ANTHROPIC_API_KEY** — the local / dev path: build a
   ``langchain_anthropic.ChatAnthropic`` (reads the key from the environment).
   Model id defaults to ``claude-opus-4-8`` (override with
   ``GEOTECH_WEBAPP_MODEL``).
3. **Nothing** — return a resolution with ``model=None`` and a banner message;
   the app shows it instead of building an agent.

Every failure mode (missing ``langchain_anthropic``, a builder that raises) is
caught and surfaced as ``source="error"`` with a readable message.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Optional

#: Default model id — the most capable general Claude (skill guidance).
DEFAULT_MODEL = "claude-opus-4-8"

#: Env var overriding the model id for the ANTHROPIC_API_KEY path.
MODEL_ENV = "GEOTECH_WEBAPP_MODEL"

#: Env var for the API key (LangChain's ChatAnthropic reads it itself too).
KEY_ENV = "ANTHROPIC_API_KEY"

#: Env var overriding the per-response output token cap.
MAX_TOKENS_ENV = "GEOTECH_WEBAPP_MAX_TOKENS"
#: The per-response output cap every engine the app builds sends
#: (``max_tokens`` / ``max_completion_tokens`` / ``max_output_tokens``).
#: Live smoke wave 2b (C5): at 8,192 a long tool call was cut silently --
#: F40 sent one cut save_file five times, F42's transcription stopped
#: mid-sentence -- and on GPT-5.x the same cap must also hold the reasoning
#: tokens. 32,000 holds a Word memo, a markup batch or a whole-page
#: transcription with reasoning room, and is within every production
#: model's own output limit (GPT-5.1 / 5.4 / Sol: 128K; GPT-4.1: 32,768;
#: Claude Opus 4.x and later: 32,000 or more). A cut reply is still caught
#: (``funhouse_agent.deep.tool_guards.OutputLimitGuard``).
DEFAULT_MAX_TOKENS = 32000

#: Seconds one model request may wait on the provider (to connect, to send,
#: or between two streamed chunks) before it fails as a timeout, which the
#: retries then treat as busy. langchain-openai and langchain-anthropic hand
#: their SDK client ``timeout=None`` unless given one, and None means NO
#: limit; it also overrides an httpx client's own timeout. In a field session
#: (2026-10-14, app 5.33.0) one request the gateway never answered held a
#: helper for over 3.5 hours. 600 s is the OpenAI and Anthropic SDKs' own
#: default. The Prompter key route sets its own (``tinyapps_engine``).
MODEL_REQUEST_TIMEOUT_S = 600.0

#: Deployment-supplied model builder (the Prompter hook). Set via
#: register_model_builder; None means "not injected".
_MODEL_BUILDER: Optional[Callable[[], object]] = None

# --- Palantir Foundry LLM proxy path (FOUNDRY_APP_PLAN.md) -------------------
# Foundry exposes provider-compatible proxies on the stack itself:
#   OpenAI shape:    https://<host>/api/v2/llm/proxy/openai/v1/chat/completions
#   Anthropic shape: https://<host>/api/v2/llm/proxy/anthropic/v1/messages
# and, inside a Code Workspace or a published app, through the container's
# Foundry proxy (https://<proxy>/language-model-service/api/proxy/anthropic
# and .../openai/v1); foundry_proxy_route picks the one to use (INF-36).
# Auth is the FOUNDRY token (bearer); the model is a Foundry RID like
# "ri.language-model-service..language-model.gpt-5-2". A model id starting with
# "ri." routes here automatically — by model family (stated as an
# "anthropic:" / "openai:" prefix, else read off the RID text) — so new model
# RIDs (e.g. Claude) plug in with NO code change: type the RID in the app or
# list it in GEOTECH_FOUNDRY_MODELS.

FOUNDRY_TOKEN_ENVS = ("GEOTECH_FOUNDRY_TOKEN", "FOUNDRY_TOKEN")
FOUNDRY_HOST_ENVS = ("GEOTECH_FOUNDRY_HOST", "FOUNDRY_HOSTNAME", "FOUNDRY_URL")

#: Deployment-mode marker. ``webapp.foundry_entry.main()`` sets this to
#: "foundry" so the app KNOWS it is running on a Foundry deployment: the
#: Anthropic-key path is then skipped entirely (never looked up, never
#: mentioned in any user-facing message — a hard requirement for enclave
#: deployments where an "API key" reference would raise IT flags) and the
#: model surface is Foundry RIDs only.
DEPLOYMENT_ENV = "GEOTECH_DEPLOYMENT"


def is_foundry_deployment() -> bool:
    """True when running as a Foundry deployment (set by foundry_entry)."""
    return os.environ.get(DEPLOYMENT_ENV, "").strip().lower() == "foundry"


def is_tinyapps_deployment() -> bool:
    """True when running on Tiny Apps (set by ``webapp.tinyapps_entry``).

    Same posture as Foundry: the only engine is the deployment's Prompter key
    (:mod:`webapp.tinyapps_engine`, which registers a model builder); the
    Anthropic-key path is never read and never named in a message.
    """
    return os.environ.get(DEPLOYMENT_ENV, "").strip().lower() == "tinyapps"


def is_keyless_deployment() -> bool:
    """A deployment where no personal API key may be read or mentioned."""
    return is_foundry_deployment() or is_tinyapps_deployment()


def foundry_token() -> Optional[str]:
    """The Foundry bearer token from the first set env var, else None."""
    for env in FOUNDRY_TOKEN_ENVS:
        val = os.environ.get(env)
        if val and val.strip():
            return val.strip()
    return None


def _with_scheme(value: str) -> str:
    """``value`` with ``https://`` added when it has no scheme, no trailing
    slash."""
    url = value.strip().rstrip("/")
    if not url.lower().startswith(("http://", "https://")):
        url = "https://" + url
    return url


def foundry_base_url() -> Optional[str]:
    """The stack base URL (``https://<host>``) from the first set env var,
    normalized (scheme added, trailing slash stripped), else None."""
    for env in FOUNDRY_HOST_ENVS:
        val = os.environ.get(env)
        if val and val.strip():
            return _with_scheme(val)
    return None


# --- Which proxy URL a Foundry model request goes to (INF-36) ---------------
# In a Code Workspace and in a published app, FOUNDRY_HOSTNAME is a data
# sidecar on localhost that answers every LLM-proxy path with 401 and an empty
# body (the AI FDE's run of 2026-10-20). The language-model service is reached
# through the container's Foundry proxy instead: some containers are given the
# provider base URLs ready-made, and every one seen carries FOUNDRY_PROXY_URL,
# from which those URLs follow.

#: The provider base URL a Foundry container may be given, by model family.
FOUNDRY_INJECTED_BASE_ENVS = {"anthropic": "ANTHROPIC_BASE_URL",
                              "openai": "OPENAI_BASE_URL"}
#: The container's Foundry proxy (a host and path, usually with no scheme).
FOUNDRY_PROXY_URL_ENV = "FOUNDRY_PROXY_URL"
#: Where a container may carry it, first set wins (the AI FDE's forwarder,
#: which worked in the published app, read both).
FOUNDRY_PROXY_URL_ENVS = (FOUNDRY_PROXY_URL_ENV, "MAESTRO_FOUNDRY_PROXY_URL")
#: The language-model service's provider-shaped doors behind that proxy.
FOUNDRY_PROXY_PATHS = {
    "anthropic": "/language-model-service/api/proxy/anthropic",
    "openai": "/language-model-service/api/proxy/openai/v1"}
#: The stack's own LLM-proxy paths (``foundry_base_url`` + these).
FOUNDRY_STACK_PATHS = {"anthropic": "/api/v2/llm/proxy/anthropic",
                       "openai": "/api/v2/llm/proxy/openai/v1"}
#: CA bundle files for the proxy's TLS, first existing file wins (the four
#: the AI FDE's forwarder read). The SDKs' HTTP client reads only
#: SSL_CERT_FILE itself.
FOUNDRY_CA_ENVS = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE",
                   "MAESTRO_CERT_PEM_PATH", "FOUNDRY_CLI_CERT_PEM_PATH")


@dataclass(frozen=True)
class FoundryRoute:
    """Where a Foundry model request goes: the provider base URL and the
    setting it came from (an env var name)."""

    url: str
    source: str

    @property
    def label(self) -> str:
        """The route as the sidebar says it: never an env var holding a
        provider's name (a Foundry deployment names no provider setting)."""
        if self.source in FOUNDRY_INJECTED_BASE_ENVS.values():
            return "the base URL the container provides"
        return self.source

    def describe(self) -> str:
        """``<setting>: <url>`` for Connection diagnostics, the URL without
        any user part or query (no token is ever in it)."""
        from urllib.parse import urlsplit, urlunsplit
        parts = urlsplit(self.url)
        host = parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
        return (f"{self.source}: "
                f"{urlunsplit((parts.scheme, host, parts.path, '', ''))}")


def foundry_proxy_route(family: str) -> Optional[FoundryRoute]:
    """The proxy base URL for ``family`` (``anthropic`` / ``openai``), else
    None. First that is set wins:

    1. ``GEOTECH_FOUNDRY_HOST``, the app's own explicit setting, + the stack
       path (what an admin or a workspace file names on purpose; a forwarder
       set there keeps working);
    2. ``ANTHROPIC_BASE_URL`` / ``OPENAI_BASE_URL``, when the container is
       given them;
    3. derived from ``FOUNDRY_PROXY_URL`` (or ``MAESTRO_FOUNDRY_PROXY_URL``)
       + the language-model service path;
    4. ``FOUNDRY_HOSTNAME`` / ``FOUNDRY_URL`` + the stack path (the route
       before INF-36).
    """
    family = family if family in FOUNDRY_STACK_PATHS else "openai"

    def _env(name: str) -> str:
        return str(os.environ.get(name) or "").strip()

    explicit = _env("GEOTECH_FOUNDRY_HOST")
    if explicit:
        return FoundryRoute(_with_scheme(explicit) + FOUNDRY_STACK_PATHS[family],
                            "GEOTECH_FOUNDRY_HOST")
    injected = FOUNDRY_INJECTED_BASE_ENVS[family]
    if _env(injected):
        return FoundryRoute(_with_scheme(_env(injected)), injected)
    for env in FOUNDRY_PROXY_URL_ENVS:
        if _env(env):
            return FoundryRoute(_with_scheme(_env(env))
                                + FOUNDRY_PROXY_PATHS[family], env)
    for env in FOUNDRY_HOST_ENVS[1:]:
        if _env(env):
            return FoundryRoute(_with_scheme(_env(env))
                                + FOUNDRY_STACK_PATHS[family], env)
    return None


def foundry_ca_bundle() -> Optional[str]:
    """The CA bundle file the Foundry proxy's TLS is checked against: the
    first of :data:`FOUNDRY_CA_ENVS` naming a file that exists, else None
    (the SDK's own default)."""
    for env in FOUNDRY_CA_ENVS:
        path = str(os.environ.get(env) or "").strip()
        if path and os.path.isfile(path):
            return path
    return None


_CA_HTTP_CLIENTS: dict = {}


def foundry_http_clients(ca_file: str):
    """``(sync, async)`` httpx clients that check TLS against ``ca_file``,
    one pair per file for the process (the SDKs share them)."""
    pair = _CA_HTTP_CLIENTS.get(ca_file)
    if pair is None:
        import ssl

        import httpx
        timeout = httpx.Timeout(MODEL_REQUEST_TIMEOUT_S, connect=30.0)
        pair = (httpx.Client(verify=ssl.create_default_context(cafile=ca_file),
                             timeout=timeout),
                httpx.AsyncClient(
                    verify=ssl.create_default_context(cafile=ca_file),
                    timeout=timeout))
        _CA_HTTP_CLIENTS[ca_file] = pair
    return pair


#: Model families: which provider shape a Foundry model speaks. A model id may
#: carry its family as a prefix -- ``anthropic:<RID or name>`` or
#: ``openai:<RID or name>`` -- so the choice travels with the id through the
#: sidebar box, ``GEOTECH_FOUNDRY_MODELS`` and a resumed conversation. Without
#: a prefix the family is read off the id (:data:`ANTHROPIC_ID_WORDS`). A
#: Foundry Claude RID need not name its maker (INF-25), hence the prefix.
MODEL_FAMILIES = ("anthropic", "openai")

#: Words in a model id that mean a Claude model when no family is stated.
#: ``anthropic`` counts anywhere in the id (the rule before INF-25); the
#: others count only as words of the model's name (:func:`_model_name_part`),
#: with no letter or digit before and no letter after -- "claude-opus-5-5",
#: "Claude_Opus_5_5" and "opus4" are Claude; "x7opus3k" (a random chunk of
#: an OpenAI RID) and "magnumopus" are not.
ANTHROPIC_ID_WORDS = ("anthropic", "claude", "opus", "sonnet", "haiku",
                      "fable")
_CLAUDE_NAME_WORD = re.compile(
    r"(?<![a-z0-9])(?:" + "|".join(ANTHROPIC_ID_WORDS[1:]) + r")(?![a-z])")


def _model_name_part(bare_id: str) -> str:
    """The part of an id that names the model: a Foundry RID's last field
    (``ri.<service>.<instance>.<type>.<name>``), else the whole id."""
    if bare_id.startswith("ri."):
        parts = bare_id.split(".", 4)
        if len(parts) == 5:
            return parts[4]
    return bare_id


def split_model_family(model_id: Optional[str]) -> "tuple[str, str]":
    """``(family, bare id)`` -- family is ``"anthropic"``, ``"openai"`` or
    ``"auto"`` (no prefix). Anything else before a colon is part of the id."""
    text = str(model_id or "").strip()
    head, sep, rest = text.partition(":")
    if sep and head.strip().lower() in MODEL_FAMILIES and rest.strip():
        return head.strip().lower(), rest.strip()
    return "auto", text


def with_model_family(model_id: Optional[str], family: Optional[str]) -> str:
    """The id carrying ``family`` as its prefix (replacing any it had);
    ``auto`` (or anything not in :data:`MODEL_FAMILIES`) leaves the id as
    typed, so a prefix typed into the id itself still counts."""
    fam = str(family or "auto").strip().lower()
    text = str(model_id or "").strip()
    if fam not in MODEL_FAMILIES:
        return text
    bare = split_model_family(text)[1]
    return f"{fam}:{bare}" if bare else ""


def model_family(model_id: Optional[str]) -> str:
    """``"anthropic"`` or ``"openai"``: the family the id states, else
    ``"anthropic"`` when the id names a Claude model, else ``"openai"``."""
    fam, bare = split_model_family(model_id)
    if fam != "auto":
        return fam
    low = bare.lower()
    if "anthropic" in low or _CLAUDE_NAME_WORD.search(_model_name_part(low)):
        return "anthropic"
    return "openai"


def is_foundry_model_id(model_id: Optional[str]) -> bool:
    """True for a Foundry model RID (``ri.…``), with or without a family."""
    return split_model_family(model_id)[1].startswith("ri.")


def _resolve_palantir_sdk(model_id: str) -> "EngineResolution":
    """Build the in-platform ``palantir_models`` engine for a model API name.

    On a Foundry deployment, a model id that is NOT an ``ri....`` RID (e.g.
    ``GPT_5_1``) is treated as a Palantir model API name and served through the
    ``palantir_models`` SDK — auth is handled by the workspace itself, so no
    host/token/proxy is involved. Never raises.
    """
    try:
        from webapp.palantir_sdk_engine import PalantirSdkChatModel, \
            sdk_available
    except Exception as exc:
        return EngineResolution(
            None, "error", model_id,
            f"Could not load the palantir_models engine wrapper: "
            f"{type(exc).__name__}: {exc}")
    if not sdk_available():
        return EngineResolution(
            None, "error", model_id,
            f"Model id '{model_id}' is not an 'ri....' RID, so it is treated "
            "as a Palantir model API name — but the 'palantir-models' package "
            "is not installed. Add 'palantir-models' and "
            "'language-model-service-api' in the workspace Libraries panel, "
            "or use an 'ri....' RID for the LLM-proxy route.")
    try:
        # GEOTECH_FOUNDRY_ROUTE: auto (Responses where the SDK can send it),
        # responses, or chat — see webapp/palantir_sdk_engine.py.
        model = PalantirSdkChatModel(
            model_api_name=model_id, max_tokens=_default_max_tokens(),
            route=(os.environ.get("GEOTECH_FOUNDRY_ROUTE") or "auto"))
    except Exception as exc:
        return EngineResolution(
            None, "error", model_id,
            f"Could not construct the palantir_models engine: "
            f"{type(exc).__name__}: {exc}")
    return EngineResolution(
        model, "foundry_sdk", model_id,
        f"Using the in-platform palantir_models SDK ({model_id}).")


#: Prompt-cache marks on the Foundry Anthropic proxy: on by default, as marks
#: on content blocks only; ``0`` / ``off`` sends none (see
#: :func:`foundry_chat_anthropic_class`).
FOUNDRY_PROMPT_CACHE_ENV = "GEOTECH_FOUNDRY_PROMPT_CACHE"

_FOUNDRY_CHAT_ANTHROPIC = None


def _strip_cache_marks(obj) -> None:
    """Remove every ``cache_control`` key from a request payload, in place."""
    if isinstance(obj, dict):
        obj.pop("cache_control", None)
        for value in obj.values():
            _strip_cache_marks(value)
    elif isinstance(obj, list):
        for value in obj:
            _strip_cache_marks(value)


class FoundryProxyLimit(ValueError):
    """A request the Foundry Anthropic proxy is known to refuse, stopped
    before it is sent, with what to do instead."""


def sanitize_foundry_anthropic_payload(payload: dict) -> list:
    """Make a Messages API request one the Foundry Anthropic proxy takes
    (INF-36), in place; returns what was changed, for the log.

    Measured on the proxy with both Claude 5.5 RIDs (the AI FDE, 2026-10-20),
    each refusal a 400 whose body names the field:

    - a top-level ``cache_control``: moved onto the last system block (when
      no system block has a mark yet), where the proxy takes it and caches;
    - ``cache_control`` on a tool definition: removed (the system and message
      block marks still cache the prompt);
    - ``service_tier``: removed;
    - ``thinking`` of type ``enabled``: sent as ``adaptive`` (the model's own
      error names adaptive as the form it takes).

    ``tool_choice`` ``any`` or a named tool is refused too, and no rewrite
    keeps its meaning, so it raises :class:`FoundryProxyLimit` here instead
    of a 400 from the proxy. Marks on system and message content blocks are
    accepted and cached, and are left alone. A new refusal names its field
    in the 400 body: add it here.
    """
    changed = []
    top = payload.pop("cache_control", None)
    if top is not None:
        changed.append("cache_control (top level)")
        system = payload.get("system")
        if isinstance(system, str) and system:
            system = payload["system"] = [{"type": "text", "text": system}]
        if (isinstance(system, list) and system
                and isinstance(system[-1], dict)
                and not any(isinstance(b, dict) and "cache_control" in b
                            for b in system)):
            system[-1] = {**system[-1], "cache_control": top}
    for tool in payload.get("tools") or []:
        if isinstance(tool, dict) and tool.pop("cache_control", None) \
                is not None:
            changed.append(f"cache_control (tool {tool.get('name')})")
    if payload.pop("service_tier", None) is not None:
        changed.append("service_tier")
    thinking = payload.get("thinking")
    if isinstance(thinking, dict) and thinking.get("type") == "enabled":
        payload["thinking"] = {"type": "adaptive"}
        changed.append("thinking enabled -> adaptive")
    choice = payload.get("tool_choice")
    kind = choice.get("type") if isinstance(choice, dict) else choice
    if kind in ("any", "tool"):
        raise FoundryProxyLimit(
            "The Foundry Anthropic proxy refuses tool_choice "
            f"'{kind}' (it forces a tool call, which these models do not "
            "support there); only 'auto' or none is accepted. Bind the tools "
            "with tool_choice='auto', or ask for structured output with "
            "method='json_schema'.")
    return changed


def foundry_chat_anthropic_class():
    """``ChatAnthropic`` as the Foundry Anthropic proxy takes it (INF-25,
    INF-36).

    The agent's prompt-caching middleware (langchain-anthropic's, which
    deepagents 0.6 and 0.7 both add) marks the system prompt's last block and
    the last tool definition, and asks for a TOP-LEVEL ``cache_control``, a
    field only Anthropic's own API accepts. This class names a non-direct
    ``_llm_type``, so langchain-anthropic moves that mark onto the last
    message block, and then every request passes
    :func:`sanitize_foundry_anthropic_payload` on the final payload, the one
    step every call takes (plain, streamed, with tools, from any middleware):
    no tool mark, no ``service_tier``, adaptive thinking, no forced tool.
    ``GEOTECH_FOUNDRY_PROMPT_CACHE=0`` sends no cache marks at all.
    Connection diagnostics send none either way: only the agent's own turns
    carry them. When a CA bundle is named (:func:`foundry_ca_bundle`) the
    client checks the proxy's TLS against it.
    """
    global _FOUNDRY_CHAT_ANTHROPIC
    if _FOUNDRY_CHAT_ANTHROPIC is None:
        import logging
        from functools import cached_property

        from langchain_anthropic import ChatAnthropic

        log = logging.getLogger(__name__)

        class FoundryChatAnthropic(ChatAnthropic):
            """ChatAnthropic against the Foundry Anthropic-messages proxy."""

            @property
            def _llm_type(self) -> str:
                return "anthropic-chat-foundry-proxy"

            def _get_request_payload(self, input_, *, stop=None, **kwargs):
                payload = super()._get_request_payload(
                    input_, stop=stop, **kwargs)
                if str(os.environ.get(FOUNDRY_PROMPT_CACHE_ENV, "")).strip() \
                        .lower() in ("0", "off", "false", "no"):
                    _strip_cache_marks(payload)
                changed = sanitize_foundry_anthropic_payload(payload)
                if changed:
                    log.debug("Foundry proxy request: %s", "; ".join(changed))
                return payload

            @cached_property
            def _client(self):
                ca = foundry_ca_bundle()
                if ca is None:
                    return super()._client
                import anthropic
                return anthropic.Client(**self._client_params,
                                        http_client=foundry_http_clients(ca)[0])

            @cached_property
            def _async_client(self):
                ca = foundry_ca_bundle()
                if ca is None:
                    return super()._async_client
                import anthropic
                return anthropic.AsyncClient(
                    **self._client_params,
                    http_client=foundry_http_clients(ca)[1])

        _FOUNDRY_CHAT_ANTHROPIC = FoundryChatAnthropic
    return _FOUNDRY_CHAT_ANTHROPIC


def _resolve_foundry(model_id: str) -> "EngineResolution":
    """Build a chat model against the Foundry LLM proxy for a model id.

    Routes by :func:`model_family`: the family the id states
    (``anthropic:`` / ``openai:``), else read off the id -- a Claude model →
    the Anthropic-messages proxy via ``ChatAnthropic``; anything else → the
    OpenAI-chat-completions proxy via ``ChatOpenAI``. The proxy is sent the
    id without its family prefix. Never raises.
    """
    stated = split_model_family(model_id)[0]
    family = model_family(model_id)
    model_id = split_model_family(model_id)[1]
    how = ("family chosen" if stated != "auto" else "family read off the id")
    token = foundry_token()
    route = foundry_proxy_route(family)
    if not token or route is None:
        missing = []
        if not token:
            missing.append("token (GEOTECH_FOUNDRY_TOKEN or FOUNDRY_TOKEN)")
        if route is None:
            missing.append("proxy address (FOUNDRY_PROXY_URL, the provider "
                           "base URL the container provides, "
                           "GEOTECH_FOUNDRY_HOST or FOUNDRY_HOSTNAME)")
        what = ("looks like a Foundry RID" if model_id.startswith("ri.")
                else "is a Claude model, served by the Foundry Anthropic proxy")
        return EngineResolution(
            None, "error", model_id,
            f"Model id '{model_id}' {what}, but the Foundry "
            f"{' and '.join(missing)} env var(s) are not set. See docs/FOUNDRY.md.")

    ca = foundry_ca_bundle()
    where = f"route: {route.label}" + ("; CA bundle set" if ca else "")
    if family == "anthropic":
        try:
            chat_anthropic = foundry_chat_anthropic_class()
        except Exception as exc:
            return EngineResolution(
                None, "error", model_id,
                f"'langchain_anthropic' is not installed ({type(exc).__name__}) "
                "— add it in the workspace Libraries panel.")
        try:
            # The proxy authenticates with "Authorization: Bearer <token>"; the
            # Anthropic client's own x-api-key header is sent too and ignored.
            model = chat_anthropic(
                model=model_id,
                max_tokens=_default_max_tokens(),
                anthropic_api_url=route.url,
                anthropic_api_key=token,
                default_headers={"Authorization": f"Bearer {token}"},
                timeout=MODEL_REQUEST_TIMEOUT_S,
            )
        except Exception as exc:
            return EngineResolution(
                None, "error", model_id,
                f"Could not construct ChatAnthropic for the Foundry proxy: "
                f"{type(exc).__name__}: {exc}")
        return EngineResolution(
            model, "foundry", model_id,
            f"Using the Foundry Anthropic proxy ({model_id}; {how}; {where}).",
            route=route.describe())

    try:
        from langchain_openai import ChatOpenAI
    except Exception as exc:
        return EngineResolution(
            None, "error", model_id,
            f"'langchain_openai' is not installed ({type(exc).__name__}) — add "
            "'langchain-openai' in the workspace Libraries panel (needed for "
            "GPT-family Foundry RIDs).")
    try:
        # The OpenAI client sends the api_key as "Authorization: Bearer …",
        # which is exactly what the Foundry proxy expects.
        #
        # max_completion_tokens, NOT max_tokens: the GPT-5/o-series reasoning
        # models REJECT max_tokens, and a Foundry RID gives langchain-openai no
        # model-name hint to translate it. max_completion_tokens is accepted by
        # every current OpenAI chat model, so send that unconditionally.
        #
        # GEOTECH_FOUNDRY_DISABLE_STREAMING=1 makes .stream() fall back to a
        # single non-streaming request — for proxies/models that reject
        # streaming (e.g. reasoning models behind an unverified org).
        _disable_stream = str(os.environ.get(
            "GEOTECH_FOUNDRY_DISABLE_STREAMING", "")).strip().lower() in (
            "1", "true", "yes", "on")
        _clients = ({} if ca is None else dict(
            zip(("http_client", "http_async_client"),
                foundry_http_clients(ca))))
        model = ChatOpenAI(
            model=model_id,
            api_key=token,
            base_url=route.url,
            disable_streaming=_disable_stream,
            max_completion_tokens=_default_max_tokens(),
            timeout=MODEL_REQUEST_TIMEOUT_S,
            **_clients,
        )
    except Exception as exc:
        return EngineResolution(
            None, "error", model_id,
            f"Could not construct ChatOpenAI for the Foundry proxy: "
            f"{type(exc).__name__}: {exc}")
    return EngineResolution(
        model, "foundry", model_id,
        f"Using the Foundry OpenAI proxy ({model_id}; {how}; {where}).",
        route=route.describe())


class EngineUnavailable(RuntimeError):
    """A builder's POLITE refusal: nothing is broken, this caller simply has
    no engine yet — a Tiny Apps tester with no AI key of their own
    (:mod:`webapp.tinyapps_engine`), a deployment whose settings are
    incomplete. :func:`resolve_engine` shows the message exactly as written
    (no exception name, no trace) with ``source="unavailable"``. The message
    must never carry a key."""


def register_model_builder(builder: Optional[Callable[..., object]]) -> None:
    """Install (or clear, with ``None``) the deployment model builder.

    The deployment supplies a Prompter-backed LangChain chat model by calling
    this at startup::

        from webapp.engine_config import register_model_builder
        register_model_builder(lambda: PrompterChatModel(fh_prompter))

    ``resolve_engine`` then prefers that model over the ANTHROPIC_API_KEY path.

    A builder that accepts an argument gets the in-app model-picker selection
    (a model id string, or ``None`` on first build)::

        register_model_builder(
            lambda model_id=None: PrompterChatModel(fh_prompter,
                                                    model=model_id or DEFAULT))

    Pair with ``GEOTECH_PROMPTER_MODELS`` (``Label=id,...``) so the sidebar
    picker offers the choices (see ``core.prompter_model_choices``). Zero-arg
    builders keep the pre-5.11 behavior: model fixed by the deployment.

    A builder with a parameter NAMED ``identity`` is also handed the
    signed-in caller (:class:`webapp.identity.Identity`) when the app knows
    it — the Tiny Apps builder picks that tester's own Prompter key with it.
    Such a builder may raise :class:`EngineUnavailable` to refuse politely.
    """
    global _MODEL_BUILDER
    _MODEL_BUILDER = builder


def has_model_builder() -> bool:
    """True when a deployment model builder is registered."""
    return _MODEL_BUILDER is not None


def _builder_takes_model_id(builder: Callable) -> bool:
    """True when the registered builder can accept the picker model id."""
    import inspect
    try:
        return bool(inspect.signature(builder).parameters)
    except (TypeError, ValueError):
        return False


def _builder_takes_identity(builder: Callable) -> bool:
    """True when the registered builder has a parameter named ``identity``
    (only an explicit name counts: a ``**kwargs`` builder is not handed one
    it never asked for)."""
    import inspect
    try:
        return "identity" in inspect.signature(builder).parameters
    except (TypeError, ValueError):
        return False


@dataclass
class EngineResolution:
    """The outcome of :func:`resolve_engine`.

    Attributes
    ----------
    model : object | None
        A LangChain-compatible chat model to hand to ``build_deep_agent``, or
        ``None`` when no engine is configured / an error occurred.
    source : str
        ``"prompter"`` | ``"anthropic"`` | ``"none"`` | ``"error"`` |
        ``"unavailable"`` (a builder's polite refusal: :class:`EngineUnavailable`
        — e.g. a Tiny Apps tester with no AI key; ``message`` is for the user).
    model_name : str
        Human-readable model identifier (e.g. the model id, or the builder's
        class name), for display.
    message : str
        A one-line status/banner message.
    route : str
        The Foundry proxy route (setting and URL), for diagnostics.
    """

    model: object
    source: str
    model_name: str
    message: str
    #: Where requests go, for Connection diagnostics: ``<setting>: <URL>``
    #: on the Foundry proxy routes (never a token), else empty.
    route: str = ""

    @property
    def ok(self) -> bool:
        """True when a usable model was resolved."""
        return self.model is not None


def _default_max_tokens() -> int:
    raw = os.environ.get(MAX_TOKENS_ENV)
    if raw:
        try:
            return max(256, int(raw))
        except ValueError:
            pass
    return DEFAULT_MAX_TOKENS


def resolve_engine(model_id: Optional[str] = None,
                   identity: Optional[object] = None) -> EngineResolution:
    """Resolve the chat engine from the environment. Never raises.

    ``model_id`` (optional) is the in-app model-picker selection: it OVERRIDES
    ``GEOTECH_WEBAPP_MODEL`` and the default for the ANTHROPIC_API_KEY path,
    and is forwarded to a deployment builder that accepts an argument (Prompter
    model switching; zero-arg builders stay deployment-fixed).
    ``None`` (the default) is byte-identical to the pre-picker behaviour.

    ``identity`` (optional) is the signed-in caller; it is forwarded to a
    builder with an ``identity`` parameter (Tiny Apps per-tester keys) and
    ignored by every other path. The result is built for THAT caller and
    must not be handed to anyone else.
    """
    # 1) Deployment-injected builder (Prompter hook). A builder that accepts
    #    an argument gets the picker selection (Prompter model switching); a
    #    zero-arg builder keeps the model deployment-fixed.
    if _MODEL_BUILDER is not None:
        # A family prefix means nothing to a deployment's own builder.
        _picked = split_model_family(model_id)[1] or None
        _kw = ({"identity": identity}
               if identity is not None and _builder_takes_identity(_MODEL_BUILDER)
               else {})
        try:
            if _builder_takes_model_id(_MODEL_BUILDER):
                model = _MODEL_BUILDER(_picked, **_kw)
            else:
                model = _MODEL_BUILDER()
                _picked = None            # selection cannot apply
        except EngineUnavailable as exc:  # a polite refusal, said as written
            return EngineResolution(None, "unavailable", "", str(exc))
        except Exception as exc:  # a bad builder must not crash the app
            return EngineResolution(
                None, "error", "prompter",
                f"Injected model builder failed: {type(exc).__name__}: {exc}")
        name = _picked or type(model).__name__
        return EngineResolution(
            model, "prompter", name,
            f"Using the deployment-provided engine ({name}).")

    # 2) Foundry model RID -> the stack's LLM proxy (routes by provider).
    _mid = model_id or os.environ.get(MODEL_ENV)
    if is_foundry_model_id(_mid):
        return _resolve_foundry(_mid)

    # 2b) Foundry deployment, non-RID model id -> the in-platform
    #     palantir_models SDK (model API name, e.g. "GPT_5_1"). No host, token,
    #     or proxy involved — the workspace authenticates the SDK itself.
    #     That engine speaks to GPT models only, so a Claude model named by
    #     its API name goes to the Anthropic-messages proxy instead.
    if is_foundry_deployment() and _mid:
        if model_family(_mid) == "anthropic":
            return _resolve_foundry(_mid)
        return _resolve_palantir_sdk(split_model_family(_mid)[1])

    # On a Foundry deployment the Anthropic-key path does not exist: the key
    # env is never read and no message references it — the only engine surface
    # is a model RID / API name (or the deployment Prompter hook above).
    if is_foundry_deployment():
        return EngineResolution(
            None, "none", "",
            "No model configured. Enter a model in the sidebar ('Model RID or "
            "API name') — an 'ri....' RID uses the LLM proxy, a model API name "
            "like GPT_5_1 uses the in-platform palantir_models SDK, and a "
            "Claude model (Model family: Anthropic) the Anthropic proxy — or "
            "set GEOTECH_FOUNDRY_MODELS (Label=id, comma-separated; "
            "Label=anthropic:id for Claude) on the deployment and republish. "
            "The app is running, but cannot answer questions until a model is "
            "selected.")

    # 2c) Tiny Apps deployment with no Prompter builder registered: the four
    #     PROMPTER_* settings are missing or incomplete. Name THEM — the
    #     personal-key path does not exist on this deployment either.
    if is_tinyapps_deployment():
        return EngineResolution(
            None, "none", "",
            "No model configured. This Tiny Apps deployment reads "
            "PROMPTER_URL, PROMPTER_MODEL and PROMPTER_API_KEY (and the "
            "optional PROMPTER_CA_BUNDLE) from Key Vault or the App Service "
            "app settings — locally, from the .env file. The app is running, "
            "but cannot answer questions until those are set and the app is "
            "restarted.")

    # 3) ANTHROPIC_API_KEY -> ChatAnthropic (local / dev path).
    if os.environ.get(KEY_ENV):
        model_id = split_model_family(
            model_id or os.environ.get(MODEL_ENV) or DEFAULT_MODEL)[1]
        try:
            from langchain_anthropic import ChatAnthropic
        except Exception as exc:
            return EngineResolution(
                None, "error", model_id,
                "ANTHROPIC_API_KEY is set but 'langchain_anthropic' is not "
                f"installed ({type(exc).__name__}). Install the [deep] extra: "
                "pip install 'geotech-staff-engineer[deep,webapp]'.")
        try:
            model = ChatAnthropic(model=model_id,
                                  max_tokens=_default_max_tokens(),
                                  timeout=MODEL_REQUEST_TIMEOUT_S)
        except Exception as exc:
            return EngineResolution(
                None, "error", model_id,
                f"Could not construct ChatAnthropic: {type(exc).__name__}: {exc}")
        return EngineResolution(
            model, "anthropic", model_id,
            f"Using Claude via the Anthropic API ({model_id}).")

    # 3) Nothing configured.
    return EngineResolution(
        None, "none", "",
        "No engine configured. Set ANTHROPIC_API_KEY (local/dev) or have the "
        "deployment register a model builder (Prompter). The app is running, "
        "but cannot answer questions until an engine is provided.")
