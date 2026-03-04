# Funhouse Agent — Usage Guide

Engine-agnostic geotechnical AI agent with text + vision capabilities.
Works with any AI backend (PrompterAPI, Claude, custom). Provides access
to all 48 Foundry geotechnical agents plus vision tools for images and PDFs.

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

### Databricks PrompterAPI

PrompterAPI satisfies the GenAIEngine interface natively — no adapter needed.

```python
agent = GeotechAgent(genai_engine=prompter_api)
```

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

Run a multi-step analysis. The agent discovers and calls Foundry tools automatically.

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
| `call_agent(agent, method, params)` | Execute any Foundry geotechnical function |
| `list_agents()` | List all 48 available agents |
| `list_methods(agent, category)` | List methods for an agent |
| `describe_method(agent, method)` | Get parameter details |

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

## Error Handling

- **Missing attachment**: Error fed back to agent, which reports it gracefully
- **No vision support**: Agent reports vision unavailable
- **Max rounds reached**: Agent returns best answer so far
- **Malformed tool calls**: Agent recovers and retries

## Files

```
funhouse_agent/
  __init__.py          # Exports: GeotechAgent, GenAIEngine, ClaudeEngine, AgentResult
  agent.py             # GeotechAgent class (ReAct loop + vision dispatch)
  engine.py            # GenAIEngine Protocol + ClaudeEngine adapter
  vision_tools.py      # Vision tool definitions and dispatch
  DESIGN.md            # Architecture and design decisions
  tests/
    test_agent.py      # 32 tests (mock engines, no API key needed)
```
