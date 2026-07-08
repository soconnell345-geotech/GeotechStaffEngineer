# Funhouse Agent — Usage Guide

Engine-agnostic geotechnical AI agent with text + vision capabilities.
Works with any AI backend (PrompterAPI, Claude, custom). Provides access
to **50 geotechnical modules** (~901 methods) plus vision tools for images and PDFs.

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
| `list_agents()` | List all available modules |
| `list_methods(agent, category)` | List methods for a module |
| `describe_method(agent, method)` | Get parameter details |

### Available Modules (43)

#### Core Analysis (17)

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
| `soe` | Support of excavation (braced/cantilever walls, stability, anchors) |
| `dxf_export` | Export cross-section geometry to DXF file format |
| `calc_package` | Generate Mathcad-style calc packages (HTML/LaTeX/PDF) for 13 modules |

#### External Library Adapters (6)

| Module | Description |
|--------|-------------|
| `opensees` | OpenSees FE analyses (PM4Sand DSS, 1D site response) |
| `pystrata` | 1D site response (equivalent-linear and linear elastic, SHAKE-type) |
| `liquepy` | CPT-based liquefaction triggering (Boulanger & Idriss 2014) |
| `seismic_signals` | Earthquake signal processing (response spectra, intensity measures, RotD) |
| `pystra` | Structural reliability analysis (FORM/SORM/Monte Carlo) |
| `salib` | Sensitivity analysis (Sobol variance-based and Morris screening) |

#### File/Data Import Adapters (2)

| Module | Description |
|--------|-------------|
| `dxf_import` | DXF CAD import for slope stability + FEM |
| `pdf_import` | PDF cross-section import (PyMuPDF vector extraction) |

> GEF/BRO-XML CPT & borehole parsing (pygef), AGS4 read/validate (python-ags4),
> and DIGGS schema/dictionary validation (pydiggs) were folded into the
> `subsurface` adapter as format-adapter methods — see below.

#### FEM & Visualization (2)

| Module | Description |
|--------|-------------|
| `fem2d` | 2D plane-strain FEM (gravity, foundation, slope SRM, excavation, seepage, consolidation) |
| `subsurface` | Subsurface data I/O — DIGGS parse + Plotly viz (parameter vs depth, Atterberg, trends) + folded format adapters: GEF/BRO-XML CPT/borehole parse (pygef), AGS4 read/validate (python-ags4), DIGGS schema/dictionary validation (pydiggs) |

#### Additional Analysis (3)

| Module | Description |
|--------|-------------|
| `gstools` | Geostatistical kriging, variogram fitting, and random field generation |
| `hvsrpy` | HVSR site characterization from ambient noise |
| `swprocess` | MASW surface wave dispersion analysis |

#### Geotech-References (14)

| Module | Description |
|--------|-------------|
| `dm7` | NAVFAC DM7 equations (340+): soil classification, stresses, settlement, seepage, foundations |
| `gec6` | GEC-6 shallow foundations reference (tables/figures/text retrieval) |
| `gec7` | GEC-7 soil nail walls reference (tables/figures/text retrieval) |
| `gec10` | GEC-10 drilled shafts reference (tables/figures/text retrieval) |
| `gec11` | GEC-11 MSE walls & reinforced soil slopes (tables/figures/text retrieval) |
| `gec12` | GEC-12 driven piles reference (tables/figures/text retrieval) |
| `gec13` | GEC-13 ground modification reference (tables/figures/text retrieval) |
| `micropile` | Micropile design reference (bond stress/tables/text retrieval) |
| `fema_p2192` | FEMA P-2192 seismic design category, site class, Fa/Fv coefficients |
| `noaa_frost` | NOAA frost depth equations (Stefan/Berggren) and soil thermal properties |
| `ufc_backfill` | UFC 3-220-04N backfill design (compaction, filter criteria, drainage) |
| `ufc_dewatering` | UFC 3-220-05 dewatering (Thiem, Dupuit, Sichardt, superposition) |
| `ufc_expansive` | UFC 3-220-07 expansive soils (swell potential, heave, pier design) |
| `ufc_pavement` | UFC 3-260-02 airfield pavement design (CBR, thickness, ESWL) |

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

## Narrow reviewer agents

A **narrow reviewer** is a `GeotechAgent` whose direct tool surface is scoped
(via `allowed_agents`) to ONE geotechnical domain, and whose system prompt is
re-cast into "review mode" with a domain checklist. You chat with it
independently to have a calc CHECKED — it re-runs the domain tools to verify
numbers and cites the governing references — rather than to produce a design.

The first one is the **seismic reviewer** (v5.4 item D6):

```python
from funhouse_agent import make_seismic_reviewer, NativeToolEngine

rev = make_seismic_reviewer(NativeToolEngine(fh_prompter))
print(rev.ask("Review this liquefaction calc: N1_60cs=12, CSR=0.22, M=7.5, "
              "computed FS=0.9 with NCEER rd but a Boulanger-Idriss MSF ...").answer)
```

It is scoped to the seismic analysis modules it may re-run —
`seismic_geotech`, `liquefaction`, `liquepy`, `slope_stability` (pseudo-static /
Newmark), `opensees`, `pystrata`, `seismic_signals`, `hvsrpy`, `swprocess`
(MASW → Vs30), `fem2d` — plus the seismic reference modules it cites from — `fema_p2082`, `dm7`, `gec5`,
`gec7`, `gec11`, and `reference_db` / `figure_db` (which still search the whole
library). The checklist covers units (g vs m/s²), total-vs-effective stress, the
liquefaction CSR/CRR chain (NCEER/Youd-2001 vs Boulanger-Idriss-2014 are NOT
interchangeable), Mononobe-Okabe sign/battered-wall conventions (KAE→Ka at
kh=0), site-class boundary cases (Vs30 at 180/360/760), Newmark polarity, and
cross-method consistency.

There are **two thin surfaces over one shared playbook** so they cannot drift:

| Surface | What it is | How you use it |
|---------|-----------|----------------|
| Funhouse sub-agent | `funhouse_agent.make_seismic_reviewer(engine)` (this module) | Chat with it in a notebook / Databricks |
| Claude Code agent | `.claude/agents/seismic-reviewer.md` (version-controlled) | Spawn it as a reviewer teammate in Claude Code |

The checklist text lives once in `funhouse_agent/review_checklists.py`
(`SEISMIC_CHECKLIST` / `SEISMIC_REVIEWER_PREAMBLE`); the factory imports it and
the `.md` playbook pastes it with a sync pointer. The scope sets live in
`funhouse_agent/dispatch.py` (`SEISMIC_MODULES`, `SEISMIC_REFERENCES`).

A **deepagents** variant is available too (planning + scratch filesystem +
optional persistent memory; needs the `[deep]` extra):

```python
from funhouse_agent import make_seismic_reviewer_deep

agent = make_seismic_reviewer_deep(model)   # a compiled deep-agent graph
agent.invoke({"messages": [{"role": "user", "content": "Review ..."}]},
             config={"configurable": {"thread_id": "review-1"}})
```

Both factories set `reference_mode="off"` deliberately: the seismic references
are already directly in scope, so the general `consult_references` tool (which
would reach ALL 21 references) is omitted to keep the scoping tight.

### The reviewer family (v5.4 F8)

Three more reviewers follow the same two-surface pattern (F8). Each has a
Funhouse factory (`make_*_reviewer` + a `_deep` variant), a
`.claude/agents/<domain>-reviewer.md` Claude Code agent, a shared checklist in
`review_checklists.py`, and scope sets in `dispatch.py`:

| Reviewer | Factory | Analysis scope | Reference scope | Key checks |
|----------|---------|----------------|-----------------|-----------|
| **seismic** | `make_seismic_reviewer` | seismic_geotech, liquefaction, liquepy, slope_stability, opensees, pystrata, seismic_signals, hvsrpy, swprocess, fem2d | fema_p2082, dm7, gec5/7/11 | g vs m/s², total/effective stress, NCEER-vs-B&I chain, M-O KAE→Ka, Vs30 boundaries, Newmark polarity |
| **foundations** | `make_foundations_reviewer` | bearing_capacity, settlement, axial_pile, drilled_shaft, pile_group, lateral_pile, wave_equation, downdrag, ground_improvement | dm7, gec6/8/9/10/12/13, micropile, ufc_expansive | GWT-in-wedge bearing, method-per-soil settlement/pile, pile-group RH sign + Converse-Labarre, drivability damping default, downdrag neutral plane |
| **earth-retention** | `make_earth_retention_reviewer` | sheet_pile, soe, retaining_walls, seismic_geotech (M-O only) | dm7, gec4/7/11, california_trenching | Ka/K0/Kp state, single-FOS embedment basis, apparent-vs-triangular envelope, MSE LRFD CDRs, battered-wall M-O |
| **slope / FEM** | `make_slope_fem_reviewer` | slope_stability, fem2d, reliability, dxf_import, pdf_import, drawing_ir | dm7, gec7/11 | LE method ordering + Bishop fixed point, noncircular rejection diagnostics, SRM-vs-LE + mesh convergence, T6-not-CST, reliability COV basis, ingest scale/provenance |

```python
from funhouse_agent import make_foundations_reviewer, NativeToolEngine
rev = make_foundations_reviewer(NativeToolEngine(fh_prompter))
rev.ask("Review this bearing-capacity calc: B=2 m, Df=1 m, sand phi=34, GWT at "
        "0.5 m, q_ult=... using Vesic Nq with total unit weight throughout ...")
```

`seismic_geotech` legitimately appears in TWO scopes — the seismic reviewer owns
liquefaction / site class / Newmark, while the earth-retention reviewer owns only
its Mononobe-Okabe seismic earth pressure (each checklist says so). Note
`geotech_common` is a shared library, not a registered agent, so it is NOT a
scope member (adding it would be a no-op). More families (e.g. deep-foundations
split out, characterization) can be added by the same recipe: a scope pair in
`dispatch.py`, a checklist + preamble in `review_checklists.py`, a factory pair
in `reviewers.py`, and a `.md` agent def that pastes the checklist with a sync
pointer.

## Model Setup (v5 deep agent)

The v5 deep agent can attach a `model_setup` sub-agent that builds 2D LE/FEM
slope models in STAGES with human confirmation gates and an echo-back
cross-section render (the agent never trusts its own read of a drawing —
the human confirms the rendered numbers visually). Off by default:

```python
from funhouse_agent.deep.agent import build_deep_agent
from funhouse_agent.deep.setup_tools import ProjectStore

store = ProjectStore()                      # inspect store.project anytime
agent = build_deep_agent(model, enable_setup_agent=True, setup_store=store,
                         setup_render_dir="renders")
```

See `geo_project/DESIGN.md` (schema + staged protocol + confirmation
mechanics) and `docs/examples/model_setup_walkthrough.md` for a full
scripted transcript.

## Installing in Databricks / Funhouse (and the gotchas)

```python
# from PyPI (released versions):
%pip install "geotech-staff-engineer[deep]==5.0.0"
# or from a test wheel -- upload to /tmp or a UC Volume (NOT /Workspace,
# which mangles the filename):
%pip install "/tmp/geotech_staff_engineer-5.1.0rc5-py3-none-any.whl[deep]"

# No dbutils.library.restartPython() needed in the normal flow (see below).
import funhouse_agent.deep   # auto-repairs the stale typing_extensions on import
```

- The `[deep]` extra is required for the v2 deepagents loop (`funhouse_agent.deep`);
  without it you get `ModuleNotFoundError: No module named 'deepagents'`.
  Add `[interactive]` for the plotly single-file viewers, `[plot]` for matplotlib figures.
- **Full eval coverage (100 questions as of v5.4) needs the optional-dependency extras:** install
  `%pip install "/tmp/...whl[deep,full]"`. With `[deep]` alone, ~12 questions (the
  gstools/pygef/ags4/pydiggs/ezdxf/SALib/pystrata/eqsig/liquepy/openseespy modules)
  fail honestly with "not installed" errors. `run_suite` runs an optional-dependency
  preflight and prints a "Missing optional packages" banner at the top of the `.md`
  when any are absent.
- **`dbutils.library.restartPython()` is no longer required in the normal flow.**
  The cluster runtime pre-imports an old `typing_extensions` (<4.13) that used to
  shadow the freshly-installed one until a kernel restart; importing anything from
  `funhouse_agent.deep` then failed with
  `TypeError: ... unexpected keyword argument 'extra_items'` (langgraph/langchain
  build PEP 728 `TypedDict`s at import, which need `typing_extensions>=4.13`).
  `funhouse_agent/runtime_check.py` now runs at the top of `funhouse_agent.deep`
  and **reloads the already-installed `typing_extensions` in place** (`importlib.reload`
  re-executes the same module object, so `typing_extensions.TypedDict` is rebound to
  the PEP 728 implementation before langchain imports it). You can just
  `import funhouse_agent.deep` after `%pip install`.
  - **Fallbacks** (only if the auto-fix reports it could not help): it raises a clear
    error telling you either to `%pip install --upgrade "typing_extensions>=4.13"`
    (if the on-disk copy is itself too old) or to
    `dbutils.library.restartPython()` (if an even older copy is still winning at
    cluster/library scope and the in-place reload did not take). Restart therefore
    remains the guaranteed fallback, not a routine step.
  - **To avoid the issue entirely, install `typing_extensions` as a cluster-scoped
    library** (Compute → cluster → Libraries → Install `typing_extensions>=4.13`, or a
    cluster init script). A cluster-scoped install is present *before* the runtime
    pre-imports its old copy, so the new version wins from the start and no reload or
    restart is involved.
- Smoke test, then chat:

```python
from funhouse_agent.deep.databricks_bridge import PrompterChatModel
from funhouse_agent.deep.selfcheck import run_selfcheck
from funhouse_agent.deep.agent import build_deep_agent
from funhouse_agent.deep.notebook import DeepNotebookChat

model = PrompterChatModel(prompter=fh_prompter, model="funhouse-gpt-high")
run_selfcheck(model)              # expect 2/2 PASS
agent = build_deep_agent(model)   # enable_setup_agent=True for the staged model-setup sub-agent
DeepNotebookChat(agent).display()
```

### Wheel health check + eval suite (for validating a test wheel)

Two purpose-built entry points ship in the wheel:

```python
# 1. One-cell health check — offline regression (calc-package/file-write fixes)
#    + the live 2-proof self-check. Prints PASS/FAIL per check.
from funhouse_agent.deep.rc_wheel_check import run_rc_check
run_rc_check(fh_prompter)          # run_rc_check() with no arg = offline section only

# 2. The geotech eval suite (100 questions as of v5.4) through the deep agent.
from funhouse_agent.deep.eval_harness import run_suite
res = run_suite(model, limit=5, out="/tmp/geotech_eval_smoke")   # subset first (real API calls)
res = run_suite(model, out="/tmp/geotech_eval_rc")               # full run; writes .json + .md
print(res["metrics"])              # P1 hallucination rate, tool-error rate, rounds, latency, tokens
```
`run_suite` writes both a results JSON and a readable markdown review at
`<out>.md`. Correctness is **partly auto-scored** (questions carrying an
`expected` answer key are numeric-tolerance/keyword graded) and **partly
eyeballed** from the `.md`; the process metrics are always computed. Copy the
`.md` out of `/tmp` with `dbutils.fs.cp` to read/share it.

### Saving outputs (calc packages, DXF, plots): do NOT use /Workspace paths

Plain Python file writes to `/Workspace/...` go through the workspace FUSE
mount, which on many compute/access modes does **not** durably store the
content: the write call "succeeds" but the workspace keeps a literal
11-byte `PLACEHOLDER` file (confirmed live 2026-06-12 with a 241 kB calc
package), and binary files such as PDFs come out corrupt. The same applies
to bare filenames on DBR 14+, where the notebook working directory IS a
/Workspace folder.

- Pass `output_path="/tmp/<name>.html"` (or a UC Volume path
  `/Volumes/<catalog>/<schema>/<vol>/<name>.html`), then copy/download:
  `dbutils.fs.cp("file:/tmp/<name>.html", "/Volumes/...")` or
  `dbutils.fs.cp("file:/tmp/<name>.html", "dbfs:/FileStore/<name>.html")`
  (downloadable at `https://<workspace>/files/<name>.html`).
- As of 5.1.0rc3 the tools defend themselves: auto-generated output paths go
  to `/tmp` on Databricks, every save response carries `file_exists` /
  `file_size_bytes` verified against the written content, and if the target
  did not store the content the tool returns an error plus a `rescue_path`
  with a verified copy in `/tmp`.
- PDF calc packages need `pdflatex`, which Databricks clusters do not have —
  generate HTML (self-contained) and print to PDF from the browser instead.

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
  system_prompt.py     # Self-contained system prompt (50 modules)
  vision_tools.py      # Vision tool definitions and dispatch
  notebook.py          # NotebookChat — ipywidgets chat interface
  DESIGN.md            # Architecture and design decisions
  adapters/
    __init__.py              # MODULE_REGISTRY + clean_value/clean_result
    _reference_common.py     # Shared factory for 14 reference adapters
    bearing_capacity.py ... soe.py, dxf_export.py, calc_package.py
    opensees_adapter.py ... salib_adapter.py        # external library adapters
    dxf_import_adapter.py, pdf_import_adapter.py     # file/data import adapters
    fem2d_adapter.py, subsurface_adapter.py  # FEM/viz (subsurface also holds the
                                             #   folded pygef/ags4/pydiggs format
                                             #   adapters as methods)
    gstools_adapter.py, hvsrpy_adapter.py, swprocess_adapter.py  # additional
    dm7_adapter.py                                  # DM7 (340+ methods, collision handling)
    gec6_adapter.py ... micropile_adapter.py        # 7 GEC/micropile adapters
    fema_adapter.py, noaa_frost_adapter.py          # FEMA + NOAA
    ufc_backfill_adapter.py ... ufc_pavement_adapter.py  # 4 UFC adapters
    (50 adapter modules total)
  tests/
    test_agent.py               # 35 agent tests
    test_engine.py              # 9 engine tests
    test_calc_package_adapter.py # 24 calc package tests
    test_notebook.py            # 38 notebook widget tests
    test_new_adapters.py        # 104 Phase 1-2 adapter tests
    test_phase34_adapters.py    # 45 Phase 3-4 adapter tests
    test_reference_adapters.py  # 163 reference adapter tests
    (418 total — mock engines, no API key needed)
```
