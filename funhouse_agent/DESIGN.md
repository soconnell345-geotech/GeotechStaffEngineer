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
├── Self-contained ReAct loop (own dispatch, own system prompt)
├── 50 adapter modules (36 analysis + 14 reference) bridging flat JSON → module APIs
├── Vision tools via engine.analyze_image()
├── Attachments dict for image/PDF data
└── Extended tool dispatch (standard + vision tools)

Dispatch Chain:
  agent.py → dispatch.py → adapters/<module>.py → analysis modules
  (No dependency on foundry/ files — those are Palantir Foundry artifacts)
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

## Adapter Layer

Each adapter in `adapters/` bridges flat JSON dicts (what LLMs produce) to
structured Python objects (Footing, SoilLayer, SlopeGeometry, etc.) needed by
analysis modules. Three patterns:

1. **Builder + construction** (most modules) — helper functions build objects from dict
2. **Direct passthrough** — `module_fn(**params)` when signatures match
3. **Inline pop + passthrough** — `params.pop()` for special args, pass rest through

Each adapter exports:
- `METHOD_REGISTRY`: `{method_name: callable}` — the execution functions
- `METHOD_INFO`: `{method_name: {category, brief, parameters, returns}}` — LLM docs

Adapters are lazy-loaded: `dispatch.py` only imports an adapter when first called.

## Extended Tools

Beyond the standard 4 ReAct tools:
- `analyze_image` — analyze attached image via vision
- `analyze_pdf_page` — render PDF page and analyze
- `save_file` — save content to a file (calc packages, HTML, PDF, plots)

These are dispatched within the agent's ReAct loop, separate from the
standard geotechnical tool dispatch.

## Notebook Chat (notebook.py)

`NotebookChat` wraps a `GeotechAgent` with ipywidgets for Jupyter/Databricks.
Injects into `agent._on_tool_call` to capture tool calls and detect output files
(from `save_file` and `calc_package` adapter results). Uses `HTML` widgets for
styled chat history with collapsible `<details>` for tool calls.

Import explicitly to avoid pulling ipywidgets for non-notebook users:
```python
from funhouse_agent.notebook import NotebookChat
```

## References

- funhouse_agent/react_support.py — AgentResult, ConversationHistory, ToolCall parsing (self-contained)
