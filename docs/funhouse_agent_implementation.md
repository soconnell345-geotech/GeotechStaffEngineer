# Funhouse Agent Implementation Guide

## What It Is

The funhouse agent (`funhouse_agent/`) is an engine-agnostic geotechnical AI agent. You give it an AI backend (Claude, Databricks PrompterAPI, anything), and it gets access to all 44 geotechnical calculation agents plus vision capabilities for analyzing images and PDFs.

It works through a **ReAct loop**: the AI thinks about what to do, calls a tool, reads the result, thinks again, and repeats until it has an answer. You don't need to know anything about the internals — you just call `agent.ask("your question")` and get back an engineering answer.

## Quick Start

### Option 1: Databricks / PrompterAPI (simplest)

PrompterAPI already satisfies the engine interface — no wrapper needed.

```python
from funhouse_agent import GeotechAgent

agent = GeotechAgent(genai_engine=prompter_api, verbose=True)
result = agent.ask("Calculate bearing capacity of a 2m square footing at 1.5m depth, phi=30, gamma=18 kN/m3")
print(result.answer)
```

### Option 2: Claude (Anthropic API)

Requires `pip install anthropic` and an `ANTHROPIC_API_KEY` environment variable.

```python
from funhouse_agent import GeotechAgent, ClaudeEngine

engine = ClaudeEngine()  # uses ANTHROPIC_API_KEY env var
agent = GeotechAgent(genai_engine=engine, verbose=True)
result = agent.ask("Calculate bearing capacity of a 2m square footing at 1.5m depth, phi=30, gamma=18 kN/m3")
print(result.answer)
```

### Option 3: Any Other Backend

Implement the `GenAIEngine` protocol (just a `chat()` method — vision is optional):

```python
class MyEngine:
    def chat(self, user: str, system: str = "", temperature: float = 0) -> str:
        # Call your AI backend here
        return my_backend.generate(prompt=user, system_prompt=system, temp=temperature)

    def analyze_image(self, image_input, user_prompt: str = "Describe this image.") -> str:
        # Optional — only needed for vision features
        return my_backend.vision(image=image_input, prompt=user_prompt)

agent = GeotechAgent(genai_engine=MyEngine(), verbose=True)
```

## Constructor Parameters

```python
GeotechAgent(
    genai_engine,           # Required. AI backend (PrompterAPI, ClaudeEngine, or custom).
    save_fn=None,           # Optional. File save function (default: local filesystem).
    max_rounds=10,          # Optional. Max ReAct loop iterations before giving up.
    temperature=0.1,        # Optional. Sampling temperature (low = deterministic).
    verbose=False,          # Optional. Print each tool call to stdout.
    on_tool_call=None,      # Optional. Callback(tool_name, arguments, result_str) for logging.
)
```

## What the Agent Can Do

### Text queries (all backends)

The agent has access to 7 tools via the ReAct loop:

| Tool | What it does |
|------|-------------|
| `list_agents` | List all 44 available agents with descriptions |
| `list_methods` | List methods for a specific agent (e.g., "bearing_capacity") |
| `describe_method` | Get full parameter docs before calling a method |
| `call_agent` | Run a calculation (bearing capacity, settlement, pile capacity, etc.) |
| `analyze_image` | Analyze an attached image using vision |
| `analyze_pdf_page` | Render a PDF page and analyze it using vision |
| `save_file` | Save output files (HTML reports, PDFs, plots) |

The first 4 are standard tools (available in all agents). The last 3 are extended tools specific to the funhouse agent.

### How the agent discovers what to calculate

The system prompt contains a **compact routing table** — a Quick Reference that maps problem types to agent names (e.g., "Shallow foundation capacity → bearing_capacity"). When the agent gets your question, it:

1. Looks up the right agent from the routing table
2. Calls `list_methods` to see what methods are available
3. Calls `describe_method` to get exact parameter names and types
4. Calls `call_agent` with your values
5. Interprets the result and gives you an engineering summary

For simple problems this takes 3-4 rounds. For complex multi-step problems (e.g., "classify the soil, then estimate bearing capacity, then check settlement"), it chains multiple agent calls.

## Multi-Turn Conversations

The agent remembers context between calls to `.ask()`:

```python
agent = GeotechAgent(genai_engine=engine, verbose=True)

# First question
result = agent.ask("Calculate bearing capacity of a 2m footing, phi=30, gamma=18, depth=1.5m")
print(result.answer)

# Follow-up — agent remembers the footing from above
result2 = agent.ask("Now estimate settlement for 150 kPa applied pressure, Es=25000 kPa")
print(result2.answer)

# Start fresh
agent.reset()
```

## Vision: Analyzing Images and PDFs

### Attach-and-ask (inside the ReAct loop)

```python
agent = GeotechAgent(genai_engine=engine)

# Attach an image
with open("site_plan.png", "rb") as f:
    agent.add_attachment("site_plan", f.read())

# Ask about it — the agent can call analyze_image internally
result = agent.ask("Look at the attached site_plan and extract the cross-section geometry")
```

### Direct extraction (bypasses the ReAct loop)

For when you just want geometry out of an image or PDF without the full agent loop:

```python
# From an image
with open("cross_section.png", "rb") as f:
    geometry = agent.extract_geometry_from_image(f.read())
# Returns a PdfParseResult with layers, points, dimensions

# From a PDF page
with open("drawings.pdf", "rb") as f:
    geometry = agent.extract_geometry_from_pdf(f.read(), page=2)

# General PDF analysis
with open("geotech_report.pdf", "rb") as f:
    summary = agent.analyze_pdf_report(f.read(), prompt="Summarize the soil boring data")
```

Note: PDF features require PyMuPDF (`pip install PyMuPDF`).

## Saving Output Files

By default, files save to the local filesystem. For cloud storage, inject a custom `save_fn`:

```python
# Databricks DBFS
agent = GeotechAgent(
    genai_engine=prompter_api,
    save_fn=lambda path, content: dbutils.fs.put(path, content, True) or path,
)

# S3
import boto3
s3 = boto3.client("s3")

def save_to_s3(path, content):
    key = f"geotech-output/{path}"
    if isinstance(content, str):
        content = content.encode("utf-8")
    s3.put_object(Bucket="my-bucket", Key=key, Body=content)
    return f"s3://my-bucket/{key}"

agent = GeotechAgent(genai_engine=engine, save_fn=save_to_s3)
```

The agent can then save calc packages, HTML reports, and Plotly figures through the `save_file` tool during its ReAct loop.

## Logging and Callbacks

### Verbose mode

```python
agent = GeotechAgent(genai_engine=engine, verbose=True)
result = agent.ask("Calculate bearing capacity...")
```

Prints each round to stdout:
```
  Round 1: describe_method({"agent_name": "bearing_capacity", "method": "bearing_cap...)
  Round 2: call_agent({"agent_name": "bearing_capacity", "method": "bearing_capacity_...)
  Round 3: FINAL ANSWER
```

### Custom callback

```python
def my_logger(tool_name, arguments, result_str):
    print(f"[LOG] {tool_name}: {list(arguments.keys())} -> {len(result_str)} chars")

agent = GeotechAgent(genai_engine=engine, on_tool_call=my_logger)
```

### Result object

```python
result = agent.ask("...")
result.answer             # str — the final engineering answer
result.tool_calls         # list[dict] — log of every tool call with arguments and result preview
result.rounds             # int — how many ReAct rounds it took
result.total_time_s       # float — wall clock time
result.conversation_turns # int — total conversation history length
```

## Common Patterns

### Databricks notebook setup

```python
# Cell 1: Setup
from funhouse_agent import GeotechAgent

agent = GeotechAgent(
    genai_engine=prompter_api,
    save_fn=lambda p, c: dbutils.fs.put(f"/FileStore/{p}", c, True) or f"/FileStore/{p}",
    verbose=True,
    max_rounds=15,  # bump for complex multi-step problems
)

# Cell 2: Ask questions
result = agent.ask("""
I have a 3m x 3m spread footing at 2m depth.
Soil: phi=32 degrees, gamma=19 kN/m3, c=5 kPa.
Groundwater at 4m depth.
Calculate bearing capacity and estimate settlement for 200 kPa applied pressure.
""")
print(result.answer)

# Cell 3: Follow-up
result2 = agent.ask("What if I increase the footing to 4m x 4m?")
print(result2.answer)
```

### Processing a geotech report PDF

```python
agent = GeotechAgent(genai_engine=engine, verbose=True)

with open("boring_log.pdf", "rb") as f:
    agent.add_attachment("boring_log", f.read())

result = agent.ask("""
Look at the boring_log attachment. Extract the soil layers and SPT N-values.
Then classify the site per AASHTO/NEHRP and check for liquefaction
assuming M=7.5 earthquake with amax=0.3g.
""")
print(result.answer)
```

### Checking what agents are available

```python
# The agent can discover this itself, but you can also check programmatically:
from chat_agent.agent_registry import list_agents, list_methods, describe_method

# All agents
print(list_agents())

# Methods for one agent
print(list_methods("bearing_capacity"))

# Full docs for one method
print(describe_method("bearing_capacity", "bearing_capacity_analysis"))
```

## Troubleshooting

**"Reached maximum of 10 tool rounds"** — The problem needed more steps than `max_rounds` allows. Increase it: `GeotechAgent(..., max_rounds=15)`. This is common for multi-step problems (classify soil → look up parameters → calculate → check limits).

**Agent calls the wrong method or agent** — This usually means the question was ambiguous. Be specific: "Calculate Vesic bearing capacity" rather than "What's the capacity?" If it persists, the compact system prompt might not have enough context — you can pass a full prompt: `GeotechAgent(..., system_prompt=build_system_prompt())` (import from `chat_agent.react_prompt`).

**Vision not available error** — Your engine doesn't implement `analyze_image()`. Either add vision to your engine or don't use image/PDF features.

**Tool parse errors** — The AI model generated malformed `<tool_call>` tags. Usually self-corrects in the next round. If persistent, try a different model or lower temperature.

## Files

| File | What it does |
|------|-------------|
| `__init__.py` | Exports: `GeotechAgent`, `GenAIEngine`, `ClaudeEngine`, `AgentResult` |
| `agent.py` | Main `GeotechAgent` class with ReAct loop and vision dispatch |
| `engine.py` | `GenAIEngine` protocol + `ClaudeEngine` adapter |
| `vision_tools.py` | Extended tool definitions (image, PDF, save) and dispatch |
| `DESIGN.md` | Internal architecture notes |
| `tests/test_agent.py` | 32 agent tests (mock engines, no API key needed) |
| `tests/test_engine.py` | 11 engine protocol and adapter tests |
