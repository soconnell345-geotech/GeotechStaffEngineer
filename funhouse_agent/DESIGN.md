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
Similarly, `save_fn` abstracts file output so the same agent works on local
filesystem, DBFS, Unity Catalog Volumes, or S3.

This means:
- No hard dependency on anthropic, openai, or any AI SDK
- No hard dependency on dbutils or any storage SDK
- PrompterAPI works with zero wrapping
- Tests use mock engines and mock save functions — no API keys needed

## Engine Interface (duck-typed)

```python
class GenAIEngine(Protocol):
    def chat(self, user: str, system: str, temperature: float) -> str: ...
    def analyze_image(self, image_input, user_prompt: str) -> str: ...  # optional
    def get_embedding(self, text) -> list: ...  # optional
```

## Save Function Interface

```python
def save_fn(path: str, content: bytes | str) -> str:
    """Save content to a file. Returns the saved file path."""
    ...
```

Default: local filesystem (`open()` + `os.makedirs`).
Inject a custom function for other backends:

```python
# Databricks DBFS
agent = GeotechAgent(genai_engine=prompter_api,
                     save_fn=lambda p, c: dbutils.fs.put(p, c, True) or p)

# Unity Catalog Volumes (works with plain open, just set the path prefix)
agent = GeotechAgent(genai_engine=prompter_api)  # default save_fn works
```

## Extended Tools

Beyond the standard 4 ReAct tools:
- `analyze_image` — analyze attached image via vision
- `analyze_pdf_page` — render PDF page and analyze
- `save_file` — save content to a file (calc packages, HTML, PDF, plots)

These are dispatched within the agent's ReAct loop, not through the
standard Foundry agent registry.

## References

- chat_agent/agent.py — GeotechChatAgent base implementation
- chat_agent/parser.py — ToolCall parsing with extensible valid_tools
