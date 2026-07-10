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
├── PrompterAPI      — Databricks (text-based ReAct, satisfies interface natively)
├── ClaudeEngine     — Anthropic SDK adapter (text-based ReAct)
├── NativeToolEngine — Wraps PrompterAPI.client for OpenAI-native tool calling
└── Future           — Any chat+vision provider

GeotechAgent
├── Two tool-calling modes (auto-detected from engine type):
│   ├── Text-based ReAct  — <tool_call> XML parsing (ClaudeEngine, PrompterAPI.chat)
│   └── Native tool calling — OpenAI `tools` parameter (NativeToolEngine)
├── 50 adapter modules (36 analysis + 14 reference) bridging flat JSON → module APIs
├── Real-FS discovery (list_files) + PDF text extraction (read_pdf_text) + vision tools
├── Attachments dict for image/PDF data — accepts an attachment key OR a real path
├── Verified saves (write-then-read-back + /tmp rescue; Databricks /Workspace API)
└── Extended tool dispatch (4 standard + 6 extended tools)

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

## Native Tool Calling (NativeToolEngine)

Newer GPT models behind PrompterAPI require tools to be registered via the
OpenAI `tools` API parameter rather than described in the system prompt.
`NativeToolEngine` wraps `fh_prompter.client` (the raw OpenAI SDK client)
and enables native function calling:

```python
from funhouse_agent import GeotechAgent, NativeToolEngine

engine = NativeToolEngine(fh_prompter)           # reads model from prompter
engine = NativeToolEngine(fh_prompter, model="gpt-4o-mini")  # or override
agent  = GeotechAgent(genai_engine=engine)
result = agent.ask("Calculate bearing capacity ...")
```

Key properties:
- **Model read at call time** — changing `fh_prompter.chat_model` takes
  effect immediately, no library update needed.
- **Auto-detected** — `GeotechAgent` checks for `engine.native_tool_calling`
  and switches to the native loop automatically.
- **Same adapters** — dispatches to the same 50 adapter modules; only the
  conversation format changes (OpenAI messages + `tool_calls` vs text XML).
- **NotebookChat compatible** — `_on_tool_call` hook fires for each tool call.

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
4. **Parse + cache** (subsurface) — `parse_diggs` parses a file, caches the
   `SiteModel` in a module-level dict, and returns a `site_key`. Subsequent
   plot calls reference the cached site by key instead of re-sending the full
   data dict, keeping tool call payloads small.

Each adapter exports:
- `METHOD_REGISTRY`: `{method_name: callable}` — the execution functions
- `METHOD_INFO`: `{method_name: {category, brief, parameters, returns}}` — LLM docs

Adapters are lazy-loaded: `dispatch.py` only imports an adapter when first called.

### Subsurface Adapter (8 methods)

`subsurface_adapter.py` provides DIGGS XML ingestion and all visualization:

| Method | Purpose |
|--------|---------|
| `parse_diggs` | Parse DIGGS 2.6/2.5.a XML → cached SiteModel, returns `site_key` + summary |
| `load_site` | Load site from nested dict → cached SiteModel, returns `site_key` |
| `plot_parameter_vs_depth` | Scatter plot of any parameter vs depth (Plotly) |
| `plot_atterberg_limits` | LL-PL bracket plot with natural moisture overlay |
| `plot_multi_parameter` | Side-by-side subplots with shared depth axis |
| `plot_plan_view` | Map of investigation locations |
| `plot_cross_section` | Lithology column profile along a transect |
| `compute_trend` | Linear/log-linear regression with R², COV |

All plot methods accept `site_key` (preferred, from cache) or `site_data` (dict).
Pass **`output_path`** to SAVE the self-contained Plotly HTML to a real path
(verified write via `_fileio.save_verified`, `/Workspace` through the Databricks
workspace API) and get back a compact `{output_path, file_exists, size,
renderer_note}` confirmation instead of the ~MB blob — the shared
`figure_output_format` / `save_html_output` helpers in `adapters/__init__.py`
handle this uniformly. Set `output_format: "html"` (no `output_path`) to receive
the HTML inline instead; the default (`"metadata"`) returns only counts/stats.

## Extended Tools

Beyond the 4 standard ReAct tools (`call_agent`, `list_methods`,
`describe_method`, `list_agents`), 6 extended tools:

- `list_files` — read-only listing of a REAL directory (name, type, size,
  mtime per entry). The DISCOVERY tool: the agent's scratch filesystem
  (`ls`/`read_file`) cannot see real paths, so `list_files` is how it finds an
  uploaded report, confirms a path, or picks a save destination. Dirs sorted
  first; optional bounded recursion (`depth` 0–2); output self-limited to stay
  valid JSON under the result cap, with a "narrow to a subdirectory" nudge.
- `read_pdf_text` — PyMuPDF text-layer extraction (no vision engine): read a
  real report's TOC, boring logs, lab summary and recommendations as TEXT.
  Cheap first pass for text-based PDFs. Flags any page with no text layer
  ("no text layer — use analyze_pdf_page") so scanned pages fall through to
  vision. Accepts an attachment key OR a real path (`/tmp/...`, `/Volumes/...`).
- `analyze_image` — analyze an attached image via vision
- `analyze_pdf_page` — render one PDF page and analyze it via vision (for
  scanned pages, figures, plotted cross-sections; heavier than `read_pdf_text`)
- `read_reference_figure` — read a value off a reference chart/figure via vision
- `save_file` — save content to a file (calc packages, HTML, PDF, plots),
  **write-verified** (see below)

`read_pdf_text`, `analyze_image`, and `analyze_pdf_page` each resolve their
source via `_resolve_attachment_or_path`: attachment-dict key first, then an
`os.path.isfile` real-path fallback, with an error that lists the available
keys and notes real paths are accepted. `read_pdf_text`, `read_reference_figure`,
and `list_files` get a larger truncation cap (16k chars) than the default 8k so
long text extracts, dense figure read-offs, and directory dumps survive.

These are dispatched within the agent's ReAct loop (and via the native
`tools` schema in `native_tools.py` / the deep `StructuredTool` factories in
`deep/tools.py`), separate from the standard geotechnical tool dispatch.

### Verified saves (`_fileio.py`)

`save_file` (and `calc_package`) write through helpers in `_fileio.py` that
**verify what actually landed on disk** — a plain write to Databricks
`/Workspace` can "succeed" while the workspace stores a literal `PLACEHOLDER`
(confirmed live 2026-06-12). `written_file_problem` reads the file back and
compares size + head; on a mismatch, `rescue_write` stages a verified copy to
the temp dir and the tool returns a structured `error` naming both the failed
target and the `rescue_path`. The same rescue fires if the writer raises.

For the DEFAULT writer with a `/Workspace` target, `save_file` first attempts a
durable write through the authenticated Databricks workspace API
(`workspace_api_upload` → `WorkspaceClient`), falling back to the plain
verified write when `databricks-sdk` is not importable. The SDK stays an
OPTIONAL dependency (preinstalled on Databricks runtimes, guarded everywhere); a
custom `save_fn` is the caller's explicit backend and is never rerouted.

## Notebook Chat (notebook.py, deep/notebook.py)

`NotebookChat` (v1 `GeotechAgent`) and `DeepNotebookChat` (deep/LangGraph agent)
wrap an agent with ipywidgets for Jupyter/Databricks. Both inject into the
agent's tool-call hook to capture tool calls and detect output files (from
`save_file` and `calc_package` adapter results), and use `HTML` widgets for
styled chat history with collapsible tool-call details.

Import explicitly to avoid pulling ipywidgets for non-notebook users:
```python
from funhouse_agent.notebook import NotebookChat       # v1 agent
chat = NotebookChat(agent); chat.display()

from funhouse_agent.deep.notebook import DeepNotebookChat  # deep agent
DeepNotebookChat.from_model(model).display()           # builds the agent for you
```

### File-attachment upload widget

Both chats expose an **Attach** `FileUpload` button that writes uploaded bytes
straight into the agent's **live attachments dict**, so a file dropped mid-chat
is immediately reachable by `read_pdf_text` / `analyze_pdf_page` / `analyze_image`
under its sanitized filename (e.g. `Mali Report v2.pdf` → key `Mali_Report_v2.pdf`).
Re-uploading the same name overwrites and says so; an attachments indicator shows
the current keys. A system-style chat line names each attached key so the user
knows what to reference.

The dict must be **one shared live object** end to end. `build_deep_agent`
materializes a single attachments dict (even when called with none), closures it
into every tool factory, and exposes it as `agent.geotech_attachments`;
`DeepNotebookChat` resolves its dict as explicit-arg → `agent.geotech_attachments`
→ fresh, and `from_model` threads the *same* dict into both the build and the
chat. v1 `NotebookChat` writes through `agent.add_attachment`. This is why an
upload reaches tools even when the agent was built without any attachments.

**Databricks ceiling:** the ipywidgets `FileUpload` transport caps at ~10 MB.
For larger files, read the bytes in a cell into an attachments dict and pass it
to the agent (`DeepNotebookChat.from_model(model, attachments=attachments)`), or
reference a driver-local real path (`/tmp/...`, `/Volumes/...`) directly.

**Note:** Panel (`pn.chat.ChatInterface`) and Gradio (`gr.ChatInterface`) were
evaluated as alternative chat UIs (Mar 2026). Gradio is blocked on the org
network. Panel installs but its Bokeh JS bundle fails to load in the locked-down
Funhouse/Databricks environment — widgets render as blank. ipywidgets is the
only framework that works reliably in Funhouse.

## References

- funhouse_agent/react_support.py — AgentResult, ConversationHistory, ToolCall parsing (self-contained)
