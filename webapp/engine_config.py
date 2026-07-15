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
DEFAULT_MAX_TOKENS = 8192

#: Deployment-supplied model builder (the Prompter hook). Set via
#: register_model_builder; None means "not injected".
_MODEL_BUILDER: Optional[Callable[[], object]] = None

# --- Palantir Foundry LLM proxy path (FOUNDRY_APP_PLAN.md) -------------------
# Foundry exposes provider-compatible proxies on the stack itself:
#   OpenAI shape:    https://<host>/api/v2/llm/proxy/openai/v1/chat/completions
#   Anthropic shape: https://<host>/api/v2/llm/proxy/anthropic/v1/messages
# Auth is the FOUNDRY token (bearer); the model is a Foundry RID like
# "ri.language-model-service..language-model.gpt-5-2". A model id starting with
# "ri." routes here automatically — by provider, from the RID text — so new
# model RIDs (e.g. Claude, once enabled on the enrollment) plug in with NO code
# change: type the RID in the app or list it in GEOTECH_FOUNDRY_MODELS.

FOUNDRY_TOKEN_ENVS = ("GEOTECH_FOUNDRY_TOKEN", "FOUNDRY_TOKEN")
FOUNDRY_HOST_ENVS = ("GEOTECH_FOUNDRY_HOST", "FOUNDRY_HOSTNAME", "FOUNDRY_URL")


def foundry_token() -> Optional[str]:
    """The Foundry bearer token from the first set env var, else None."""
    for env in FOUNDRY_TOKEN_ENVS:
        val = os.environ.get(env)
        if val and val.strip():
            return val.strip()
    return None


def foundry_base_url() -> Optional[str]:
    """The stack base URL (``https://<host>``) from the first set env var,
    normalized (scheme added, trailing slash stripped), else None."""
    for env in FOUNDRY_HOST_ENVS:
        val = os.environ.get(env)
        if val and val.strip():
            host = val.strip().rstrip("/")
            if not host.startswith(("http://", "https://")):
                host = "https://" + host
            return host
    return None


def is_foundry_model_id(model_id: Optional[str]) -> bool:
    """True for a Foundry model RID (``ri.…``)."""
    return bool(model_id) and str(model_id).startswith("ri.")


def _resolve_foundry(model_id: str) -> "EngineResolution":
    """Build a chat model against the Foundry LLM proxy for a model RID.

    Routes by provider inferred from the RID text: ``anthropic`` in the RID →
    the Anthropic-messages proxy via ``ChatAnthropic``; anything else → the
    OpenAI-chat-completions proxy via ``ChatOpenAI``. Never raises.
    """
    token = foundry_token()
    base = foundry_base_url()
    if not token or not base:
        missing = []
        if not token:
            missing.append("token (GEOTECH_FOUNDRY_TOKEN or FOUNDRY_TOKEN)")
        if not base:
            missing.append("host (GEOTECH_FOUNDRY_HOST or FOUNDRY_HOSTNAME)")
        return EngineResolution(
            None, "error", model_id,
            f"Model id '{model_id}' looks like a Foundry RID, but the Foundry "
            f"{' and '.join(missing)} env var(s) are not set. See docs/FOUNDRY.md.")

    if "anthropic" in model_id.lower():
        try:
            from langchain_anthropic import ChatAnthropic
        except Exception as exc:
            return EngineResolution(
                None, "error", model_id,
                f"'langchain_anthropic' is not installed ({type(exc).__name__}) "
                "— add it in the workspace Libraries panel.")
        try:
            # The proxy authenticates with "Authorization: Bearer <token>"; the
            # Anthropic client's own x-api-key header is sent too and ignored.
            model = ChatAnthropic(
                model=model_id,
                max_tokens=_default_max_tokens(),
                anthropic_api_url=f"{base}/api/v2/llm/proxy/anthropic",
                anthropic_api_key=token,
                default_headers={"Authorization": f"Bearer {token}"},
            )
        except Exception as exc:
            return EngineResolution(
                None, "error", model_id,
                f"Could not construct ChatAnthropic for the Foundry proxy: "
                f"{type(exc).__name__}: {exc}")
        return EngineResolution(
            model, "foundry", model_id,
            f"Using the Foundry Anthropic proxy ({model_id}).")

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
        model = ChatOpenAI(
            model=model_id,
            max_tokens=_default_max_tokens(),
            api_key=token,
            base_url=f"{base}/api/v2/llm/proxy/openai/v1",
        )
    except Exception as exc:
        return EngineResolution(
            None, "error", model_id,
            f"Could not construct ChatOpenAI for the Foundry proxy: "
            f"{type(exc).__name__}: {exc}")
    return EngineResolution(
        model, "foundry", model_id,
        f"Using the Foundry OpenAI proxy ({model_id}).")


def register_model_builder(builder: Optional[Callable[[], object]]) -> None:
    """Install (or clear, with ``None``) the deployment model builder.

    The TinyApp deployment supplies a Prompter-backed LangChain chat model by
    calling this at startup::

        from webapp.engine_config import register_model_builder
        register_model_builder(lambda: PrompterChatModel(fh_prompter))

    ``resolve_engine`` then prefers that model over the ANTHROPIC_API_KEY path.
    """
    global _MODEL_BUILDER
    _MODEL_BUILDER = builder


@dataclass
class EngineResolution:
    """The outcome of :func:`resolve_engine`.

    Attributes
    ----------
    model : object | None
        A LangChain-compatible chat model to hand to ``build_deep_agent``, or
        ``None`` when no engine is configured / an error occurred.
    source : str
        ``"prompter"`` | ``"anthropic"`` | ``"none"`` | ``"error"``.
    model_name : str
        Human-readable model identifier (e.g. the model id, or the builder's
        class name), for display.
    message : str
        A one-line status/banner message.
    """

    model: object
    source: str
    model_name: str
    message: str

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


def resolve_engine(model_id: Optional[str] = None) -> EngineResolution:
    """Resolve the chat engine from the environment. Never raises.

    ``model_id`` (optional) is the in-app model-picker selection: it OVERRIDES
    ``GEOTECH_WEBAPP_MODEL`` and the default for the ANTHROPIC_API_KEY path.
    The deployment Prompter engine is fixed by the deployment and ignores it.
    ``None`` (the default) is byte-identical to the pre-picker behaviour.
    """
    # 1) Deployment-injected builder (Prompter hook). Model is deployment-fixed;
    #    the picker selection does not apply here.
    if _MODEL_BUILDER is not None:
        try:
            model = _MODEL_BUILDER()
        except Exception as exc:  # a bad builder must not crash the app
            return EngineResolution(
                None, "error", "prompter",
                f"Injected model builder failed: {type(exc).__name__}: {exc}")
        name = type(model).__name__
        return EngineResolution(
            model, "prompter", name,
            f"Using the deployment-provided engine ({name}).")

    # 2) Foundry model RID -> the stack's LLM proxy (routes by provider).
    _mid = model_id or os.environ.get(MODEL_ENV)
    if is_foundry_model_id(_mid):
        return _resolve_foundry(_mid)

    # 3) ANTHROPIC_API_KEY -> ChatAnthropic (local / dev path).
    if os.environ.get(KEY_ENV):
        model_id = model_id or os.environ.get(MODEL_ENV) or DEFAULT_MODEL
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
                                  max_tokens=_default_max_tokens())
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
