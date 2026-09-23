"""``LangChainVisionEngine`` — the v5.0 vision adapter for the deepagents port.

The deepagents vision tools (``analyze_image`` / ``analyze_pdf_page`` /
``read_reference_figure`` in :mod:`funhouse_agent.deep.tools`) call
``engine.analyze_image(image_bytes, prompt) -> str`` exactly like the v1
``GenAIEngine`` protocol (see :class:`funhouse_agent.engine.ClaudeEngine`). v5
drives a single LangChain :class:`~langchain_core.language_models.chat_models.BaseChatModel`
for both text and vision, so this thin adapter lets that one model object satisfy
the vision surface — no separate Anthropic/OpenAI SDK client required.

It mirrors ``ClaudeEngine.analyze_image`` but emits the **standard LangChain
multimodal message shape** instead of a provider-specific one: a
:class:`~langchain_core.messages.HumanMessage` whose ``content`` is a list of
blocks ``[{"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
{"type": "text", "text": prompt}]``. LangChain's provider integrations
(``langchain_anthropic``, ``langchain_openai``, …) translate that shape to the
right native format, so the same adapter works across providers.

Dependency-light: only ``langchain_core`` (for ``HumanMessage``) and stdlib
``base64``.
"""

from __future__ import annotations

import base64
from typing import Optional


class LangChainVisionEngine:
    """Adapt a LangChain chat model to the minimal vision surface v5 needs.

    Implements only ``analyze_image`` — the single method the deepagents vision
    tools call through. Construct it with any LangChain
    :class:`~langchain_core.language_models.chat_models.BaseChatModel` (or
    anything exposing a compatible ``.invoke([messages]) -> AIMessage``):

        from langchain_anthropic import ChatAnthropic
        from funhouse_agent.deep.vision_engine import LangChainVisionEngine

        engine = LangChainVisionEngine(ChatAnthropic(model="claude-sonnet-4-6"))
        text = engine.analyze_image(png_bytes, "Read Kp off this chart.")

    Parameters
    ----------
    model : BaseChatModel
        The LangChain chat model that performs the vision call. It must be
        vision-capable for real images; for offline tests a fake model whose
        ``.invoke`` returns an ``AIMessage`` works.
    media_type : str, optional
        MIME type embedded in the data URI. ``None`` (default) reads it from
        the bytes — a render can be PNG or JPEG (see
        :mod:`funhouse_agent.vision_view`).
    detail : str, optional
        OpenAI's image ``detail`` level (``"high"``, ``"original"``...). The
        default follows the app's image budget
        (:func:`funhouse_agent.vision_view.detail`); ``""`` omits the field.
        A model that rejects the value is asked once more without it.
    """

    # Marks this as a vision-capable engine (parity with the v1 engines, which
    # expose feature sentinels like ``native_tool_calling``). It is NOT a
    # native-tool-calling engine — deepagents handles tool calling itself.
    native_tool_calling = False

    #: The data URI is labelled with the bytes' real type, so a JPEG render
    #: may be sent (see ``vision_view.render_view(allow_jpeg=...)``).
    accepts_jpeg = True

    def __init__(self, model, media_type: Optional[str] = None,
                 detail: Optional[str] = None):
        self._model = model
        self._media_type = media_type
        self._detail = detail

    @property
    def model(self):
        """The wrapped LangChain chat model."""
        return self._model

    def analyze_image(
        self,
        image_input,
        user_prompt: str = "Describe this image.",
    ) -> str:
        """Analyze an image with the wrapped LangChain model.

        Parameters
        ----------
        image_input : bytes, bytearray, or str
            Raw image bytes, or a path to an image file. Mirrors
            ``ClaudeEngine.analyze_image`` / the ``GenAIEngine`` contract.
        user_prompt : str
            What to extract / describe from the image.

        Returns
        -------
        str
            The model's text response. The ``content`` of the returned
            ``AIMessage`` is normalized to plain text (it may come back as a
            string or as a list of content blocks depending on the provider).
        """
        from langchain_core.messages import HumanMessage

        from funhouse_agent import vision_view

        if isinstance(image_input, (bytes, bytearray)):
            data = bytes(image_input)
        elif isinstance(image_input, str):
            with open(image_input, "rb") as f:
                data = f.read()
        else:
            raise TypeError(
                f"image_input must be bytes or file path, got {type(image_input)}"
            )

        media_type = self._media_type or vision_view.image_media_type(data)
        data_uri = f"data:{media_type};base64,{base64.b64encode(data).decode()}"
        detail = (self._detail if self._detail is not None
                  else vision_view.detail()) or None

        def message(with_detail: bool) -> "HumanMessage":
            image_url = {"url": data_uri}
            if with_detail and detail:
                image_url["detail"] = detail
            return HumanMessage(content=[
                {"type": "image_url", "image_url": image_url},
                {"type": "text", "text": user_prompt},
            ])

        try:
            response = self._model.invoke([message(True)])
        except Exception as exc:
            # An older model (GPT-4.1, GPT-5.2) has no "original" detail; ask
            # once more at its default rather than fail the read.
            if not detail or "detail" not in str(exc).lower():
                raise
            response = self._model.invoke([message(False)])
        return _content_to_text(getattr(response, "content", response))


def _content_to_text(content) -> str:
    """Normalize an ``AIMessage.content`` (str or list of blocks) to text.

    LangChain providers may return ``content`` as a plain string or as a list
    of content blocks (e.g. ``[{"type": "text", "text": "..."}, ...]``). This
    flattens any text blocks and ignores non-text blocks.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Standard LangChain text block, or a Claude-style {"type":"text"}.
                if block.get("type") == "text" and "text" in block:
                    parts.append(block["text"])
                elif "text" in block and isinstance(block["text"], str):
                    parts.append(block["text"])
        return "".join(parts)
    return str(content)


__all__ = ["LangChainVisionEngine"]
