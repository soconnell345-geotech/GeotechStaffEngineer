"""Universal Foundry entry/launcher for the geotech web app (one file).

TWO WAYS TO USE IT:

* **JupyterLab Code Workspace (the app path)** — make this file the
  workspace's ``app.py`` (or point Applications → Preview / Publish at it).
  Streamlit executes it directly; it enables Foundry deployment mode,
  registers the in-platform ``palantir_models`` engine, and runs the
  packaged app.

* **VS Code workspace / any terminal** — ``python foundry_app_launcher.py``.
  It re-launches itself under ``streamlit run`` on port 8501 (``PORT`` env
  overrides).

WHAT IT DOES beyond a plain launch:

1. Marks the process as a Foundry deployment (``GEOTECH_DEPLOYMENT=foundry``)
   — same no-key surface as the published app.
2. If the ``palantir_models`` SDK is importable, it registers an in-platform
   engine for the model API name in ``GEOTECH_SDK_MODEL`` (default
   ``GPT_5_1``) through the app's ``register_model_builder`` hook. This works
   on the CURRENT pip release (5.9.1) — the engine wrapper is embedded below,
   so no upgrade is needed to test the SDK route end-to-end.
   * Set ``GEOTECH_SDK_MODEL=`` (empty) to skip this and use the normal
     RID / LLM-proxy surface instead.
3. If ``palantir_models`` is NOT installed, it prints a note and boots the
   normal app (RID box + proxy route). Install ``palantir-models`` and
   ``language-model-service-api`` to enable the SDK route.

Errors from the model (403 access-not-granted, 404 wrong name, ...) appear
verbatim in the app's sidebar "Connection diagnostics" panel — run that first.

HOW IT WORKS: when streamlit is executing this file (detected via the script
run context, or the sentinel env var set by the self-relaunch), it registers
the engine and runs the installed app in-process. Run as plain ``python``, it
re-launches itself under ``streamlit run``.
"""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys

_SENTINEL = "_GEOTECH_VSCODE_BOOTSTRAP"
_DEFAULT_MODEL = os.environ.get("GEOTECH_SDK_MODEL", "GPT_5_1")
_PORT = os.environ.get("PORT", "8501")


# ===========================================================================
# Embedded palantir_models engine (mirror of webapp/palantir_sdk_engine.py,
# master 4449b6f — remove once running a release that ships that module).
# ===========================================================================

def _build_sdk_model(model_api_name: str):
    """Return a LangChain chat model over the palantir_models SDK."""
    try:  # a release that ships the packaged wrapper wins
        from webapp.palantir_sdk_engine import PalantirSdkChatModel
        return PalantirSdkChatModel(model_api_name=model_api_name,
                                    max_tokens=_max_tokens())
    except ImportError:
        pass
    return _InlineSdkChatModel(model_api_name=model_api_name,
                               max_tokens=_max_tokens())


def _max_tokens() -> int:
    try:
        return max(256, int(os.environ.get("GEOTECH_WEBAPP_MAX_TOKENS", "")))
    except ValueError:
        return 8192


def _sdk():
    from palantir_models.models import OpenAiGptChatLanguageModel
    import language_model_service_api.languagemodelservice_api as lms_base
    import language_model_service_api.languagemodelservice_api_completion_v3 \
        as lms_v3
    return OpenAiGptChatLanguageModel, lms_base, lms_v3


def _make_inline_class():
    """Define the embedded chat model (imports deferred to call time)."""
    from typing import Any, Optional, Sequence

    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import (AIMessage, BaseMessage,
                                         SystemMessage, ToolMessage)
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from pydantic import ConfigDict, Field, PrivateAttr

    from funhouse_agent.deep.databricks_bridge import (
        _FINISH_REASON_MAP, _text_content, _to_usage_metadata, _usage_to_dict)

    def _lc_messages_to_sdk(messages, lms_base, lms_v3):
        Role = lms_base.ChatMessageRole
        out = []
        for message in messages:
            if isinstance(message, SystemMessage):
                out.append(lms_base.ChatMessage(
                    Role.SYSTEM, _text_content(message.content)))
            elif isinstance(message, ToolMessage):
                out.append(lms_base.ChatMessage(
                    Role.TOOL, _text_content(message.content),
                    tool_call_id=message.tool_call_id))
            elif isinstance(message, AIMessage):
                tool_calls = None
                if message.tool_calls:
                    tool_calls = [
                        lms_v3.GptToolCall(
                            id=tc.get("id") or "",
                            tool_call=lms_v3.GptToolCallInfo(
                                function=lms_v3.FunctionToolCallInfo(
                                    arguments=json.dumps(
                                        tc.get("args", {}) or {}),
                                    name=tc.get("name", ""))))
                        for tc in message.tool_calls
                    ]
                text = _text_content(message.content)
                out.append(lms_base.ChatMessage(
                    Role.ASSISTANT, text or None, tool_calls=tool_calls))
            else:
                out.append(lms_base.ChatMessage(
                    Role.USER, _text_content(message.content)))
        return out

    def _openai_tools_to_sdk(openai_tools, lms_v3):
        out = []
        for tool in openai_tools:
            fn = tool.get("function", tool)
            out.append(lms_v3.GptTool(function=lms_v3.GptFunctionTool(
                name=fn.get("name", ""),
                parameters=fn.get("parameters") or {},
                description=fn.get("description"))))
        return out

    def _sdk_message_to_ai_message(msg) -> AIMessage:
        content = getattr(msg, "content", "") or ""
        tool_calls = []
        for tc in getattr(msg, "tool_calls", None) or []:
            info = getattr(tc, "tool_call", None)
            fn = getattr(info, "function", None) if info is not None else None
            if fn is None:
                fn = getattr(tc, "function", None)
            name = getattr(fn, "name", "") if fn is not None else ""
            raw_args = getattr(fn, "arguments", "") if fn is not None else ""
            try:
                args = json.loads(raw_args) if raw_args else {}
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append({"name": name, "args": args,
                               "id": getattr(tc, "id", None),
                               "type": "tool_call"})
        return AIMessage(content=content, tool_calls=tool_calls)

    class _InlineSdkChatModel(BaseChatModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)

        model_api_name: str
        max_tokens: Optional[int] = None
        temperature: Optional[float] = None
        openai_tools: Optional[list] = Field(default=None, exclude=True)
        _sdk_model: Any = PrivateAttr(default=None)

        @property
        def _llm_type(self) -> str:
            return "palantir-models-chat-inline"

        def _model(self):
            if self._sdk_model is None:
                OpenAiGptChatLanguageModel, _, _ = _sdk()
                self._sdk_model = OpenAiGptChatLanguageModel.get(
                    self.model_api_name)
            return self._sdk_model

        def bind_tools(self, tools: Sequence[Any], **kwargs: Any):
            openai_tools = [convert_to_openai_tool(t) for t in tools]
            return self.model_copy(update={"openai_tools": openai_tools})

        def _generate(
            self,
            messages: "list[BaseMessage]",
            stop: "Optional[list[str]]" = None,
            run_manager: "Optional[CallbackManagerForLLMRun]" = None,
            **kwargs: Any,
        ) -> ChatResult:
            _, lms_base, lms_v3 = _sdk()
            request_kwargs: dict = {}
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            if max_tokens is not None:
                request_kwargs["max_tokens"] = max_tokens
            temperature = kwargs.get("temperature", self.temperature)
            if temperature is not None:
                request_kwargs["temperature"] = temperature
            if stop:
                request_kwargs["stop"] = list(stop)
            tools = kwargs.get("tools") or self.openai_tools
            if tools:
                request_kwargs["tools"] = _openai_tools_to_sdk(tools, lms_v3)

            request = lms_v3.GptChatCompletionRequest(
                _lc_messages_to_sdk(messages, lms_base, lms_v3),
                **request_kwargs)
            response = self._model().create_chat_completion(request)

            choice = response.choices[0]
            ai_message = _sdk_message_to_ai_message(choice.message)
            finish_reason = getattr(choice, "finish_reason", None)
            generation_info = {"finish_reason": _FINISH_REASON_MAP.get(
                finish_reason, finish_reason)}
            model_name = (getattr(response, "model", None)
                          or self.model_api_name)
            if model_name:
                generation_info["model_name"] = model_name
            usage = getattr(response, "usage", None)
            if usage is not None:
                generation_info["usage"] = _usage_to_dict(usage)
                ai_message.usage_metadata = _to_usage_metadata(usage)
            return ChatResult(generations=[ChatGeneration(
                message=ai_message, generation_info=generation_info)])

    return _InlineSdkChatModel


class _LazyInline:
    """Defer class creation so plain-launcher mode never imports langchain."""

    _cls = None

    def __call__(self, **kwargs):
        if _LazyInline._cls is None:
            _LazyInline._cls = _make_inline_class()
        return _LazyInline._cls(**kwargs)


_InlineSdkChatModel = _LazyInline()


# ===========================================================================
# Bootstrap (runs INSIDE streamlit) / launcher (plain python)
# ===========================================================================

def _bootstrap() -> None:
    os.environ.setdefault("GEOTECH_DEPLOYMENT", "foundry")
    from webapp.foundry_entry import app_path

    model_name = _DEFAULT_MODEL.strip()
    if model_name:
        try:
            _sdk()  # availability probe
            from webapp.engine_config import register_model_builder
            register_model_builder(
                lambda: _build_sdk_model(model_name))
            print(f"[launcher] palantir_models SDK engine registered "
                  f"({model_name}).", flush=True)
        except ImportError as exc:
            print("[launcher] palantir_models not importable "
                  f"({exc}) — booting the normal RID/proxy surface. "
                  "Install 'palantir-models' + 'language-model-service-api' "
                  "to test the SDK route.", flush=True)
    else:
        print("[launcher] GEOTECH_SDK_MODEL empty — normal RID/proxy "
              "surface.", flush=True)

    runpy.run_path(app_path(), run_name="__main__")


def _launch() -> None:
    env = dict(os.environ)
    env[_SENTINEL] = "1"
    env.setdefault("GEOTECH_DEPLOYMENT", "foundry")
    print(f"[launcher] starting streamlit on port {_PORT} "
          f"(open http://localhost:{_PORT} — VS Code should offer to "
          "forward/open it) ...", flush=True)
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run",
         os.path.abspath(__file__),
         "--server.port", _PORT,
         "--server.address", "0.0.0.0",
         "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        env=env, check=False)


def _under_streamlit() -> bool:
    """True when streamlit itself is executing this file (Jupyter Preview /
    published app / the self-relaunch) rather than a plain ``python`` run."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        return False


if os.environ.get(_SENTINEL) or _under_streamlit():
    _bootstrap()
elif __name__ == "__main__":
    _launch()
