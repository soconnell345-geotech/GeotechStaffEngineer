# Funhouse Agent — Design Notes

## Purpose

Engine-agnostic geotechnical agent that works with any AI backend providing
text generation and optional vision capabilities. Designed for:

1. **Funhouse/Databricks** — PrompterAPI already satisfies the interface
2. **Claude** — ClaudeEngine adapter wraps the Anthropic SDK
3. **Any future backend** — implement the GenAIEngine protocol

## Architecture

```
GenAIEngine (Protocol)
├── PrompterAPI   — Databricks (satisfies interface natively)
├── ClaudeEngine  — Anthropic SDK adapter
└── Future        — Any chat+vision provider

GeotechAgent
├── Uses GeotechChatAgent internally for ReAct loop
├── Adds vision tools via engine.analyze_image()
├── Attachments dict for image/PDF data
└── Extended tool dispatch (standard + vision tools)
```

## Key Design: Dependency Injection

The agent accepts a `genai_engine` object rather than importing any specific SDK.
This means:
- No hard dependency on anthropic, openai, or any AI SDK
- PrompterAPI works with zero wrapping
- Tests use mock engines — no API keys needed

## Engine Interface (duck-typed)

```python
class GenAIEngine(Protocol):
    def chat(self, user: str, system: str, temperature: float) -> str: ...
    def analyze_image(self, image_input, user_prompt: str) -> str: ...  # optional
    def get_embedding(self, text) -> list: ...  # optional
```

## Vision Tools

Extended beyond the standard 4 ReAct tools:
- `analyze_image` — analyze attached image via vision
- `analyze_pdf_page` — render PDF page and analyze

These are dispatched within the agent's ReAct loop, not through the
standard Foundry agent registry.

## References

- chat_agent/agent.py — GeotechChatAgent base implementation
- chat_agent/parser.py — ToolCall parsing with extensible valid_tools
