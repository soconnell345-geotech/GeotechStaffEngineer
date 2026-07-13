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

    # 2) ANTHROPIC_API_KEY -> ChatAnthropic (local / dev path).
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
