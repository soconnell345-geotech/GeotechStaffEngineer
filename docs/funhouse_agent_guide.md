# Funhouse Agent — Usage Guide

Engine-agnostic geotechnical AI agent with text + vision capabilities.
Works with any AI backend (PrompterAPI, Claude, custom). Provides access
to 18 geotechnical analysis modules plus vision tools for images and PDFs.

Fully self-contained — dispatches directly to analysis modules via an
internal adapter layer. No dependency on `foundry/` files or Palantir Foundry.

## Quick Start

```python
from funhouse_agent import GeotechAgent, ClaudeEngine

engine = ClaudeEngine()  # uses ANTHROPIC_API_KEY env var
agent = GeotechAgent(genai_engine=engine)

result = agent.ask("Calculate bearing capacity of a 2m footing on sand, phi=35")
print(result.answer)
```

## Engine Setup

### Claude (Anthropic SDK)

Requires `pip install anthropic`.

```python
from funhouse_agent import ClaudeEngine

engine = ClaudeEngine(
    api_key=None,               # falls back to ANTHROPIC_API_KEY env var
    model="claude-sonnet-4-6",  # model ID
    max_tokens=4096,            # response token limit
)
```

### Databricks / Funhouse PrompterAPI

PrompterAPI satisfies the GenAIEngine interface natively — no adapter needed.
It uses the OpenAI API under the hood (Prompter NTLM backend, Azure OpenAI,
or Grok/X.AI depending on configuration).

```python
from funhouse.prompter import PrompterAPI

# Default: auto-detects backend from config/env vars
fh_prompter = PrompterAPI()

# Or specify backend explicitly
fh_prompter = PrompterAPI(backend="openai", api_key="...", base_url="...")
fh_prompter = PrompterAPI(backend="grok", api_key="xai-...")

# Pass directly as the engine — chat(), analyze_image(), get_embedding() all match
agent = GeotechAgent(genai_engine=fh_prompter)
```

**PrompterAPI methods used by GeotechAgent:**

| Method | Signature | Notes |
|--------|-----------|-------|
| `chat()` | `(user, system, temperature) -> str` | Returns `None` on failure (handled gracefully) |
| `analyze_image()` | `(image_input: bytes\|str, user_prompt) -> str` | Vision via base64 + OpenAI multimodal |
| `get_embedding()` | `(text) -> list` | Optional, not used by ReAct loop |

### Custom Engine

Implement the `GenAIEngine` protocol:

```python
class MyEngine:
    def chat(self, user: str, system: str = "", temperature: float = 0) -> str:
        return "response from your backend"

    def analyze_image(self, image_input, user_prompt: str = "") -> str:
        return "image analysis"  # optional — omit if no vision needed
```

## Constructor

```python
agent = GeotechAgent(
    genai_engine,           # Required: PrompterAPI, ClaudeEngine, or custom
    save_fn=None,           # Custom file save function (default: local filesystem)
    max_rounds=10,          # Max ReAct loop iterations
    temperature=0.1,        # Sampling temperature
    verbose=False,          # Print round-by-round progress
    on_tool_call=None,      # Callback: fn(tool_name, args, result)
)
```

## Public Methods

### ask(question) -> AgentResult

Run a multi-step analysis. The agent discovers and calls analysis modules automatically.

```python
result = agent.ask("Calculate settlement of a 3m footing on clay")

print(result.answer)              # final answer text
print(result.tool_calls)          # list of tool calls made
print(result.rounds)              # number of ReAct rounds
print(result.total_time_s)        # elapsed time in seconds
print(result.conversation_turns)  # total messages in history
```

### add_attachment(key, data)

Attach image or PDF bytes for vision analysis.

```python
with open("site_plan.png", "rb") as f:
    agent.add_attachment("site_plan", f.read())

result = agent.ask("Extract slope geometry from site_plan")
```

### extract_geometry_from_image(image_bytes, prompt=None) -> PdfParseResult

Direct geometry extraction from an image (bypasses ReAct loop).

```python
result = agent.extract_geometry_from_image(png_data)
print(result.surface_points)       # [[x1, y1], [x2, y2], ...]
print(result.boundary_profiles)    # {"Layer1": [[...]], "Layer2": [[...]]}
```

### extract_geometry_from_pdf(pdf_bytes, page=0) -> PdfParseResult

Direct geometry extraction from a PDF page.

```python
with open("cross_section.pdf", "rb") as f:
    result = agent.extract_geometry_from_pdf(f.read(), page=0)
```

### analyze_pdf_report(pdf_bytes, prompt=None) -> str

Render and analyze a PDF page using vision. Returns text analysis.

```python
with open("geotech_report.pdf", "rb") as f:
    analysis = agent.analyze_pdf_report(f.read(), prompt="Summarize soil profiles")
```

### reset()

Clear conversation history and attachments.

```python
agent.reset()
```

### has_vision (property)

Check if the engine supports image analysis.

```python
if agent.has_vision:
    agent.add_attachment("photo", image_data)
```

## Available Tools

The agent uses these tools automatically during the ReAct loop:

### Standard Tools (Geotechnical Analysis)

| Tool | Purpose |
|------|---------|
| `call_agent(agent, method, params)` | Execute a geotechnical calculation |
| `list_agents()` | List all 18 available analysis modules |
| `list_methods(agent, category)` | List methods for a module |
| `describe_method(agent, method)` | Get parameter details |

### Available Modules (18)

| Module | Description |
|--------|-------------|
| `bearing_capacity` | Shallow foundation bearing capacity (Vesic/Meyerhof/Hansen) |
| `settlement` | Foundation settlement (elastic, Schmertmann, consolidation) |
| `slope_stability` | Slope stability (Fellenius/Bishop/Spencer, circular+noncircular, grid search) |
| `seismic_geotech` | Seismic evaluation (site class, M-O pressure, liquefaction) |
| `retaining_walls` | Retaining walls (cantilever + MSE, GEC-11) |
| `axial_pile` | Driven pile axial capacity (Nordlund/Tomlinson/Beta) |
| `drilled_shaft` | Drilled shaft capacity (GEC-10 alpha/beta/rock socket, LRFD) |
| `sheet_pile` | Sheet pile walls (cantilever and anchored) |
| `lateral_pile` | Lateral pile analysis (COM624P, 8 p-y models, FD solver) |
| `pile_group` | Pile group analysis (rigid cap, 6-DOF, Converse-Labarre) |
| `ground_improvement` | Ground improvement (aggregate piers, wick drains, vibro, GEC-13) |
| `wave_equation` | Smith 1-D wave equation (bearing graph, drivability) |
| `downdrag` | Pile downdrag (Fellenius neutral plane, UFC 3-220-20) |
| `wind_loads` | ASCE 7-22 wind loads on freestanding walls and fences (Ch 29.3) |
| `soe` | Support of excavation (braced/cantilever walls, stability, anchors) |
| `geolysis` | Soil classification (USCS/AASHTO) + SPT corrections + bearing capacity |
| `dxf_export` | Export cross-section geometry to DXF file format |
| `calc_package` | Generate Mathcad-style calc packages (HTML/LaTeX/PDF) for 13 modules |

### Vision Tools

| Tool | Purpose |
|------|---------|
| `analyze_image` | Analyze an attached image via engine vision |
| `analyze_pdf_page` | Render a PDF page and analyze it |
| `save_file` | Save content to a file (text or base64 binary) |

## File Output (save_fn)

By default, files save to the local filesystem. Inject a custom function for
other backends:

```python
# Databricks DBFS
agent = GeotechAgent(
    genai_engine=engine,
    save_fn=lambda path, content: dbutils.fs.put(path, content, True) or path,
)

# The agent can now save files via the save_file tool
result = agent.ask("Generate an HTML report and save to output/report.html")
```

The save function signature: `save_fn(path: str, content: bytes | str) -> str`

## Usage Patterns

### Multi-Turn Conversation

The agent retains context across questions:

```python
r1 = agent.ask("What is the bearing capacity for phi=30, B=2m, D=1m?")
r2 = agent.ask("Now estimate settlement at 80% of that capacity")

agent.reset()  # start fresh
```

### Image + Analysis Pipeline

```python
with open("slope_photo.png", "rb") as f:
    agent.add_attachment("slope", f.read())

result = agent.ask(
    "From the slope photo, extract layer geometry and calculate "
    "factor of safety using Bishop's method"
)
```

### Verbose Mode + Callbacks

```python
calls = []

agent = GeotechAgent(
    genai_engine=engine,
    verbose=True,                    # prints each round
    on_tool_call=lambda t, a, r: calls.append(t),  # log tool names
)

result = agent.ask("Calculate bearing capacity")
print(f"Tools used: {calls}")
```

### Deterministic Production Use

```python
agent = GeotechAgent(
    genai_engine=engine,
    temperature=0.0,   # deterministic
    max_rounds=5,      # bounded execution
)
```

## Notebook Chat (ipywidgets)

Interactive chat interface for Jupyter/Databricks notebooks using ipywidgets.

```python
from funhouse_agent import GeotechAgent
from funhouse_agent.notebook import NotebookChat

agent = GeotechAgent(genai_engine=prompter_api)
chat = NotebookChat(agent)
chat.display()
```

Features:
- Text input with Send/Reset buttons (Enter key to send)
- Scrollable chat history with styled messages and collapsible tool calls
- File upload widget + `chat.attach(path)` for loading files from DBFS/workspace
- Output file detection and display for calc packages
- Token/stats tracking bar
- `chat.output_files` — list of produced file paths
- `chat.preview_file(path)` — inline HTML calc package display

```python
# Attach files programmatically (easier than widget in Databricks)
chat.attach("/dbfs/project/site_plan.png")
chat.attach("/dbfs/project/report.pdf", key="geotech_report")

# After running queries
chat.output_files          # list of calc package paths
chat.preview_file(path)    # render HTML inline
```

## Error Handling

- **Missing attachment**: Error fed back to agent, which reports it gracefully
- **No vision support**: Agent reports vision unavailable
- **Max rounds reached**: Agent returns best answer so far
- **Malformed tool calls**: Agent recovers and retries

## Architecture

The funhouse agent has its own self-contained dispatch layer that routes
tool calls directly to analysis modules. It does NOT use the `foundry/`
files (which are Palantir Foundry deployment artifacts, excluded from pip).

```
User Question → GeotechAgent.ask()
  → LLM generates <tool_call> → parser extracts ToolCall
  → funhouse_agent/dispatch.py routes to adapter
  → funhouse_agent/adapters/<module>.py converts flat dict → structured objects
  → analysis module computes → result dict returned to LLM
  → LLM formulates final answer
```

## Files

```
funhouse_agent/
  __init__.py          # Exports: GeotechAgent, GenAIEngine, ClaudeEngine, AgentResult
  agent.py             # GeotechAgent class (ReAct loop + vision dispatch)
  react_support.py     # AgentResult, ConversationHistory, ToolCall parser (self-contained)
  engine.py            # GenAIEngine Protocol + ClaudeEngine adapter
  dispatch.py          # Tool dispatch — routes to adapters (not foundry)
  system_prompt.py     # Self-contained system prompt (18 modules)
  vision_tools.py      # Vision tool definitions and dispatch
  notebook.py          # NotebookChat — ipywidgets chat interface
  DESIGN.md            # Architecture and design decisions
  adapters/
    __init__.py        # MODULE_REGISTRY + clean_value/clean_result helpers
    bearing_capacity.py, settlement.py, slope_stability.py,
    seismic_geotech.py, retaining_walls.py, axial_pile.py,
    drilled_shaft.py, sheet_pile.py, lateral_pile.py,
    pile_group.py, ground_improvement.py, wave_equation.py,
    downdrag.py, wind_loads.py, soe.py, geolysis.py,
    dxf_export.py, calc_package.py
    (18 adapter modules — one per analysis module)
  tests/
    test_agent.py               # 35 agent tests
    test_engine.py              # 9 engine tests
    test_calc_package_adapter.py # 24 calc package tests
    test_notebook.py            # 38 notebook widget tests
    (106 total — mock engines, no API key needed)
```
