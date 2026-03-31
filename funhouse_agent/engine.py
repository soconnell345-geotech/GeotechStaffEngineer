"""
GenAIEngine protocol and adapter implementations.

Defines the interface any AI backend must satisfy to work with GeotechAgent.
PrompterAPI already satisfies this interface natively — no wrapper needed.

NativeToolEngine wraps a PrompterAPI instance and uses its underlying
OpenAI client directly, enabling native function-calling (tools parameter)
instead of text-based ReAct parsing.
"""

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class GenAIEngine(Protocol):
    """Interface that any AI backend must satisfy.

    PrompterAPI already has these methods with compatible signatures.
    For other backends, create a thin adapter (see ClaudeEngine).
    """

    def chat(
        self,
        user: str,
        system: str = "You are a helpful assistant.",
        temperature: float = 0,
    ) -> str:
        """Generate a text response.

        Parameters
        ----------
        user : str
            User message / prompt.
        system : str
            System prompt.
        temperature : float
            Sampling temperature.

        Returns
        -------
        str
            Model response text.
        """
        ...

    def analyze_image(
        self,
        image_input,
        user_prompt: str = "Describe this image.",
    ) -> str:
        """Analyze an image using vision capabilities.

        Parameters
        ----------
        image_input : bytes, str, or path-like
            Image data (bytes) or file path.
        user_prompt : str
            Prompt describing what to extract from the image.

        Returns
        -------
        str
            Model response text about the image.
        """
        ...

    def get_embedding(self, text) -> list:
        """Generate vector embeddings.

        Parameters
        ----------
        text : str or list of str
            Text to embed.

        Returns
        -------
        list
            Embedding vector(s).
        """
        ...


class ClaudeEngine:
    """Adapter wrapping Anthropic SDK to match the GenAIEngine interface.

    Requires: pip install anthropic (optional dependency).

    Parameters
    ----------
    api_key : str, optional
        Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
    model : str
        Claude model ID.
    max_tokens : int
        Maximum response tokens.
    """

    def __init__(
        self,
        api_key=None,
        model="claude-sonnet-4-6",
        max_tokens=8192,
    ):
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required for ClaudeEngine. "
                "Install with: pip install anthropic"
            )
        self._client = Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def chat(
        self,
        user: str,
        system: str = "You are a helpful assistant.",
        temperature: float = 0,
    ) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            temperature=temperature,
        )
        return response.content[0].text

    def analyze_image(
        self,
        image_input,
        user_prompt: str = "Describe this image.",
    ) -> str:
        import base64

        if isinstance(image_input, (bytes, bytearray)):
            b64 = base64.b64encode(image_input).decode()
        elif isinstance(image_input, str):
            with open(image_input, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
        else:
            raise TypeError(
                f"image_input must be bytes or file path, got {type(image_input)}"
            )

        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }],
        )
        return response.content[0].text

    def get_embedding(self, text) -> list:
        raise NotImplementedError(
            "Claude API does not provide embeddings. "
            "Use a separate embedding service."
        )


class NativeToolEngine:
    """Engine wrapping a PrompterAPI's OpenAI client for native tool calling.

    Instead of text-based ReAct (``<tool_call>`` XML), this engine tells
    GeotechAgent to use the OpenAI ``tools`` parameter so the model invokes
    functions natively.  This avoids the "tools haven't been configured on
    the back end" error that newer GPT models produce when tools are only
    described in the system prompt.

    The model name is read from ``prompter.chat_model`` at each call, so
    model changes in the notebook take effect immediately without rebuilding
    the engine or agent.

    Parameters
    ----------
    prompter : PrompterAPI
        Initialized Funhouse PrompterAPI instance.  Only ``.client`` and
        ``.chat_model`` are used — no dependency on ``prompter.chat()``.
    model : str, optional
        Override model name.  If *None*, reads ``prompter.chat_model`` each
        call.
    max_tokens : int, optional
        Max response tokens per LLM call.  *None* = API default.

    Usage::

        from funhouse_agent import GeotechAgent, NativeToolEngine

        engine = NativeToolEngine(fh_prompter)
        agent  = GeotechAgent(genai_engine=engine)
        result = agent.ask("Calculate bearing capacity ...")
    """

    # Sentinel so GeotechAgent can detect this engine type cheaply.
    native_tool_calling = True

    def __init__(
        self,
        prompter,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ):
        self._prompter = prompter
        self._model_override = model
        self._max_tokens = max_tokens

    # -- properties --------------------------------------------------------

    @property
    def client(self):
        """The underlying OpenAI SDK client."""
        return self._prompter.client

    @property
    def model(self) -> str:
        """Active model — override or live from PrompterAPI."""
        return self._model_override or self._prompter.chat_model

    # -- GenAIEngine protocol ----------------------------------------------

    def chat(
        self,
        user: str,
        system: str = "You are a helpful assistant.",
        temperature: float = 0,
    ) -> str:
        """Plain text chat (no tools).

        Used by the reviewer agent and any non-tool-calling path.
        """
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
        }
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def analyze_image(
        self,
        image_input,
        user_prompt: str = "Describe this image.",
    ) -> str:
        """Analyze an image via the OpenAI vision API."""
        import base64

        if isinstance(image_input, (bytes, bytearray)):
            b64 = base64.b64encode(image_input).decode()
        elif isinstance(image_input, str):
            with open(image_input, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
        else:
            raise TypeError(
                f"image_input must be bytes or file path, got {type(image_input)}"
            )

        kwargs = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }],
        }
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def get_embedding(self, text) -> list:
        """Delegate embeddings to the PrompterAPI."""
        return self._prompter.get_embedding(text)
