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
├── 69 adapter modules (36 analysis + 33 reference; `dispatch.ANALYSIS_MODULES` / `REFERENCE_MODULES`) bridging flat JSON → module APIs
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
- **Same adapters** — dispatches to the same 69 adapter modules; only the
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

### Subsurface Adapter (17 methods)

`subsurface_adapter.py` provides DIGGS XML ingestion and writing, the folded
format adapters, and all visualization:

| Method | Purpose |
|--------|---------|
| `parse_diggs` | Parse DIGGS 2.6/2.5.a XML → cached SiteModel, returns `site_key` + summary |
| `load_site` | Load site from nested dict → cached SiteModel, returns `site_key` |
| `write_diggs` | Write DIGGS 2.6 from logs and lab results the agent has read, checked against the bundled schema and read back. Records come inline or, since RI-31 (2026-10-14), from JSON files a helper saved in the conversation's working folder (`investigations_file` / `lab_tests_file`; an id in two inputs is refused). Returns `cross_checks` (lab links, summary tables against sheets, drives against the sample interval and the log's form, readings below the hole's total depth, records citing pages no tool opened) and never changes a value. Groundwater "not encountered" is never written as a reading, an unknown position is written as NaN, and a printed refusal (`>50`, `50/5"`) becomes a refusal row (RI-27) |
| `plot_parameter_vs_depth` | Scatter plot of any parameter vs depth (Plotly) |
| `plot_atterberg_limits` | LL-PL bracket plot with natural moisture overlay |
| `plot_multi_parameter` | Side-by-side subplots with shared depth axis |
| `plot_plan_view` | Map of investigation locations; since GM-76 the axes are labelled from the stated coordinate system (longitude / latitude, easting / northing, or plain x / y with the system named) |
| `plot_cross_section` | Lithology column profile along a transect |
| `compute_trend` | Linear/log-linear regression with R², COV |
| `parse_cpt`, `parse_bore` | GEF or BRO-XML CPT and borehole files (pygef) |
| `read_ags4`, `validate_ags4` | AGS4 files (python-ags4; from the release after 5.33.0 an optional extra, `[ags4]`) |
| `validate_diggs_schema`, `validate_diggs_dictionary` | DIGGS XML against the schema (2.6 bundled; 2.5.a and the dictionary need pydiggs) |
| `spt_correction`, `stress_dilatancy` | SPT N → N60 → (N1)60; Bolton (1986) dilatancy |

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
- `analyze_image` — analyze an attached image via vision. Since VIS-09 the
  image's own pixels are a frame like a page's: the result gives its `view`
  and 0-999 boxes, an image too large for one look is also read in tiles cut
  from the original, and `render_region` takes the image as
  `attachment_key` to zoom (`funhouse_agent/image_view.py`). Since VIS-26
  `annotate_document` and `find_like` take an image in the same pixels:
  PyMuPDF opens an image as a page sized by the dpi the file declares (0.75
  pt a pixel with none), so the app puts the image on its own page at one
  point a pixel (`image_view.as_pdf`) for a new marked copy and for the
  search, converts a mark exactly to whatever copy it goes on, and refuses
  a mark centred off the image (or off a PDF's page)
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

## Deep agent: long turns, helpers and the record (on master after 5.33.0, unreleased)

The deep agent (`funhouse_agent/deep/`) is described from the code in
`module_work/harness_theory/` (`shared_machinery.md`: the loop, the limits,
the eyes). What changed on master after 5.33.0, in short:

- **An errored turn keeps its work** (`webapp/turn_record.py`, TA-19): every
  reply and tool result is recorded as it arrives; on an error the record
  becomes replayable history, so "continue" picks up from the tool results.
  Very large results are kept whole in the conversation's `kept/` folder. An
  interrupted turn's new files get download cards and are named in the next
  turn's files note (TA-28).
- **Helpers** (`deep/keep_work.py`, TA-21, TA-22): at most
  `GEOTECH_MAX_HELPERS` (2) at once; a general-purpose helper has a 30-call
  budget (`GEOTECH_HELPER_MAX_MODEL_CALLS`) whose last call hands over what
  it gathered; a helper that stops writes what it read to
  `kept/helper_*.md`; a `task` result whose helper saved files names them,
  so the primary passes the file instead of retyping it (RI-31).
- **The `reviewer` helper** (TA-25) has a 10-call budget
  (`DEFAULT_REVIEWER_MAX_MODEL_CALLS`), its last call answering from what it
  has; its description says to use it only after a draft answer that
  involved calculations, and that it sees only its brief: it cannot open
  documents, pages or files.
- **Busy refusals** (`deep/busy_retry.py`, `AgentBusyRetry`, TA-20): every
  agent model call retries a 429 / 529 / 5xx / overloaded / connection /
  timeout refusal on the provider's retry-after or 15/30/60/120 s,
  `GEOTECH_AGENT_BUSY_TRIES` (5) tries within `GEOTECH_BUSY_TURN_MAX_S`
  (900 s) a turn. An optional per-key token pacer:
  `GEOTECH_TOKENS_PER_MINUTE` (`deep/pacing.py`; unset = off).
- **Request time limits** (TA-26): a model request gives up after 180 s on
  the Prompter key route (`webapp/tinyapps_engine.REQUEST_TIMEOUT_S`) and
  600 s on the Foundry proxy and Anthropic routes
  (`webapp/engine_config.MODEL_REQUEST_TIMEOUT_S`); before, langchain handed
  the SDK no limit at all. A timeout is then retried as busy.
- **`kept_readings`** (`vision_tools.py`, TA-28): a read-only tool that lists
  the conversation's kept page readings by reference and returns one whole
  by page, as `reuse=true` does. On the geotech page for the primary and its
  helpers; on the review page only with `GEOTECH_REVIEW_KEPT_READINGS` (OFF).
- **Cited pages** (`cited_pages.py`, EV-18): after each turn the pages an
  answer cites are checked against the pages any tool or helper read or
  looked at, and a `cited_pages` event is recorded; `GEOTECH_CITED_PAGES`
  (`note` / `retry`; OFF) acts on it. `write_diggs` always adds a
  `page_not_read` cross-check for a record citing pages no tool opened, and
  `record_finding` returns `pages_not_opened`.
- **Cited documents** (`cited_documents.py`, EV-21; live wave w8b, where a
  brief made up "GEC-11 / AASHTO LRFD 3.11.5" and the PDF and the answer
  carried it as the user's): every published document an answer or a
  deliverable cites (GEC-n and FHWA report numbers as one document, AASHTO
  LRFD with its article, UFC / DM 7, EM, ASTM, ACI, ASCE 7, Caltrans, NCHRP,
  EN 1997 ... and author-and-year works) is checked against what the
  conversation's own record says was looked up: the user's messages, a
  references result or a reference module called, a worked example, a
  document opened (not one the conversation wrote), a calculation's own
  result or method documentation; never a brief, a helper's hand-back or
  the module catalog. An author the record names without that year (Rankine,
  when the user asked for Rankine pressure) is `year_not_checked`, not
  flagged. The answer's check is always a `cited_documents` event;
  `GEOTECH_CITED_DOCUMENTS` (`note` / `retry`; OFF) notes the answer or
  tells the model once (in one note with the page check), and adds
  `cited_not_looked_up` and a `citation_note` to `html_to_pdf`, the
  `*_package` methods, `write_docx`, a text `save_file` and
  `annotate_document`; the file is always written.
- **The review page** (DR-04, INF-10) no longer offers
  `read_reference_figure` / `view_worked_example_source`, and deepagents
  0.7's `delete` scratch tool is hidden there (and in the references
  consult) with the other scratch tools.
- **Misspelt keys, the app's side** (DR-32; 2026-10-16): `annotate_document`
  shows the model planlens' own markup fields, read off the installed spec
  (`document_tools.annotate_markups_schema`: `pdf_page` added, `author`
  withheld, closed to other keys), where it showed an array of anything.
  Nothing checks a call against that schema, so `color`, `box`, a nested
  `anchor` and a kind's other name still reach planlens, which forgives
  them. A `remove` object with a key other than `id`/`text` is refused with
  a did-you-mean; a markup's own `author` is dropped with a note (the app
  signs); written marks with no comment are named in `empty_comments`;
  `search_document` says when `min_score` came without `fuzzy`
  (`min_score_unused`).
- **What a result chose** (GM-115; 2026-10-16; no switch, results only):
  `call_agent` adds `defaults_used` to a successful dict result, the
  declared parameters (METHOD_INFO, GM-110) the call left out with the
  default each took; a name given under a spelling the method reads
  (`parameter_aliases`) counts as given, a default of None is left out, and
  an error, a list or a string result is returned as it was. The module
  results carry `choices` (`geotech_common/choices.py`): what `method="auto"`
  resolved to, each p-y model's source, the stress method, the M-O basis,
  the liquefaction procedure with rd and MSF per layer, the slope search's
  families tried and not tried. The liquefaction tool marks its procedure
  "default" or "given" and names the other; the axial adapter marks its
  own delta/phi default. Since 2026-10-17 the two are true and agree:
  `call_agent` runs the method inside `geotech_common.choices.call_inputs`
  with the names the call gave and the declared defaults it left in force,
  and a module's `how_given(..., name=)` answers from them, so a typed value
  equal to the default is "given" (GM-118; nested values, such as a
  layer's field, are still compared by value). `defaults_used` lists only
  the defaults the run used: a parameter's `applies_when` in METHOD_INFO
  (`adapters.declare_applies_when`; a dict of other parameters' values,
  "set" or "unset", a list of such dicts for "any", or False) says when it
  is read, and `dispatch.applies` tests it on the call's values over the
  declared defaults; any name `how_given` called "default" is listed
  whatever its condition, so the two never contradict (GM-119).
  `describe_method` leaves `applies_when` out. Tagged: liquefaction
  (CPT-only keys, C0, gamma_w), lateral pile (axis, f'c, E_steel),
  axial pile (cohesive_phi, the uplift fraction), settlement and the
  settlement package, the seismic package (KA, kv, M), the slope report
  package and the slope tools (crack water, reverse curvature, each search
  setting, the Bishop screen, the interslice function, the 3-stage normal,
  a Monte Carlo search budget, the infinite slope's water inputs; the kh
  yield_acceleration finds itself). The choice-bearing methods name
  `choices` in their returns, and `call_agent`'s description names both
  (GM-125). A default that depends on another input (Terzaghi's N-gamma
  is Kumbhojkar's, every other factor method's Vesic's; GM-158) is
  declared `default_depends_on` = `{"parameter", "values", "otherwise"}`
  in place of one `default`, its description says "Default depends on
  X", and `defaults_used` and the modules' `choices` take it at the
  call's value of X (`dispatch.default_in_force`). The guard
  (`tests/test_defaults_used.py` with `tests/method_probes.py`) runs every
  analysis-side method on its smallest inputs, gives the run its own
  `defaults_used` back by name and fails if the result moves or a result
  that echoes a default echoes another value; the same probes check that
  every declared name is one the adapter's own check accepts
  (`test_top_level_params.py`, GM-150).
  An input that changes nothing is not advertised (GM-160): the guard
  also gives every default a probe's run lists another value (the next
  allowed value, the other boolean, a number moved by half) and fails
  where the result does not move past the input's own echo and the
  `choices`; a label-only input is in `method_probes.LABEL_ONLY` and one
  whose effect the probe does not reach in `NOT_REACHED`, each with why.
  The file readers are probed on `eval_samples/` (`"<eval_samples>/..."`;
  a probe needing an optional backend names it under `"needs"`).
- **Vision** (VIS-01, VIS-04, VIS-05): an optional `turn` (0/90/180/270) on
  `analyze_pdf_page` / `render_region`; boxes right on sideways sheets; a
  zoom that cuts lines of lettering widens to take them in or says which
  edge cuts them; the turned re-read behind `GEOTECH_VISION_TURN_RETRY`
  (OFF). The markup check keeps the agent correcting through checks 1-3,
  names unconfirmed marks at check 4, and returns `found_bbox` for a
  misplaced mark (DR-15, DR-16).
- **The record** (`activity.jsonl`, EV-14): token use on the Prompter route,
  `task_id` and the brief on every event inside a helper, a refusal's status
  and rate-limit headers, the engine and retry policy, queue times, and cut
  payloads kept whole in `activity_full/`.
- **Figures** (GM-90, GM-95, GM-92; 2026-10-15): `plot_data` draws plans
  and sections to scale (`equal_aspect`), a plan as a drawing (`plan_view`:
  no axes, a scale bar, a north arrow from `north_deg` or up with a note),
  filled and hatched outlines, named points placed clear of each other,
  hover text, separate cases as markers along x, `xlim` / `ylim` and
  `legend_position` (`profile_figure/data_plot.py`); its description says
  its PNG and chart are one figure. The `calc` helper, which draws the
  deliverable's figures, gets `analyze_image` to look at them. `save_file`
  adds a `chart_note` when it writes a `*.plotly.json` by hand
  (`vision_tools.HAND_CHART_NOTE`): no PNG of that figure exists, and a PNG
  beside it is now a different figure. Behind `GEOTECH_FIGURE_HELPER` (OFF;
  arm `figure_helper`): a `figure` helper (`deep/figure_helper.py`) with an
  eleven-item figure checklist after Tufte, six worked examples
  (`figure_examples`, from `profile_figure/figure_examples.py`), `plot_data`,
  `analyze_image`, `calculate` and a 14-call budget; both pages, not the
  lean review agent. The checklist, `plot_data`'s `depth_axis` and
  `plot_code`'s description carry the owner's conventions (GM-97): elevation
  up a profile by default, kPa and kN, test points against an assumed
  design line, the wall drawn; with `plot_code` the examples tool also lists
  three worked `plot_code` scripts and says which kind each example is.
- **Plotting code in a locked box** (GM-98; 2026-10-15; `plot_box/`):
  behind `GEOTECH_PLOT_CODE` (OFF; arm `plot_code`; it brings the figure
  helper with it) the figure helper and the `calc` helper get
  `plot_code`, which runs matplotlib or Plotly code the model writes in a
  fresh Python process (`python -I -B plot_box/child.py`). The layers, in
  order: a static check of the code (imports only from numpy, pandas,
  matplotlib, mpl_toolkits, plotly, scipy and the standard maths and text
  modules; no `eval` / `exec` / `getattr`, no dunder attributes, no
  library's own `os` or `sys` handed on by name), run in the parent too so
  a refused script starts no process; a scrubbed environment (no key,
  token or app setting reaches the child); OS limits (Linux `resource`:
  address space, CPU time, file size, open files, no new processes;
  Windows: a job object with a memory and CPU-time ceiling and one
  process); the libraries the code names imported and drawn with once;
  then a PEP 578 audit hook, bound to immutable defaults and with no off
  switch, over the code AND the libraries it calls: reads only in the
  conversation folder, the installed libraries and font / time zone
  files; writes only in the conversation folder and only figure, chart and
  data files (no overwriting a file that was there before, except an image
  or chart being redrawn; at most 40 files); no process, socket, `ctypes`,
  registry, pickle or database. The code gets `data`, `save_figure`,
  `save_chart` and the plan furniture (`scale_bar`, `north_arrow`,
  `nice_length` from `geotech_common/plan_furniture.py`); a figure left
  open is saved. Limits: 600 s unless the call asks for more, up to 3,600 s
  (`GEOTECH_PLOT_CODE_SECONDS`, `GEOTECH_PLOT_CODE_MAX_SECONDS`; the owner:
  "I want it to cook"), 2,048 MB (`GEOTECH_PLOT_CODE_MEMORY_MB`), two at
  once per process (`GEOTECH_PLOT_CODE_AT_ONCE`). While it runs the app's
  "Working..." label reads "drawing with code: 2m 10s so far, stops at
  10m 00s" (`funhouse_agent/progress_notes.py`, carried on the heartbeat).
  Measured against an escape battery on Windows and Linux
  (`funhouse_agent/tests/test_plot_box.py`). Left as it is: a Python
  sandbox is not a kernel one; on Linux a kernel layer (Landlock or
  namespaces) could be added under the hook (BACKLOG INF-16).
- **Figures after live wave w9** (GM-107; 2026-10-16;
  `module_work/live_smoke/runs/w9-figures/TRACE_REVIEW.md`). Unswitched: the
  chat's card for a figure is the NEWER of its picture and its chart
  (`webapp/core.collect_turn_artifacts`), and a figure file removed during
  the turn gets no card; `plot_data`'s `xlim` / `ylim` keep the order given
  and `x_on_top` puts the x axis along the top. Behind the figure switches:
  a `plot_code` draw removes a chart an earlier draw this turn left under
  its PNG's name; `replaces=` on `plot_code` and the figure helper's
  `plot_data` retires drafts this turn drew under another name (figure files
  in the working folder only, newer than the start of the turn, else of the
  running helper, and nothing at all with neither; `plot_box.retire_drafts`);
  draws are counted across `plot_code` and `plot_data` and across names
  within one FIGURE helper run (`plot_box.draws_so_far`, from the helper's
  record; the calc helper draws a package's several figures and is not
  capped), the third asking for the best version to be handed back; a
  draw's `next` asks the look what the
  figure shows against the data before the lettering; the box reports the
  lettering's printed size (`lettering`: "5.6 pt at 6.5 in";
  `print_width_in`) and saves the report PNG at the dpi asked, with a
  2,048 px copy in `.scratch/` for looks (`look_file`). On the main agent
  (`figure_helper.FigureTasks`, inside `KeepWork`): each figure brief carries
  the nearest worked example (`nearest_example`: every word of its name in
  the brief), and the helper's result ends with what it drew and what each
  look found, written in code from its record (`looks_report`); the main
  prompt names the helper and says not to look again at what it looked at
  (`FIGURE_PRIMARY_NOTE`). Tests: `funhouse_agent/tests/
  test_plot_box_drafts.py`, `deep/tests/test_figure_runs.py`.
- **A figure is one thing across a turn, whoever draws it** (GM-151;
  2026-10-18; `module_work/live_smoke/runs/w12-figures-judgment/
  TRACE_REVIEW.md` F55j, F65, fixes 1, 2, 7). `plot_box/figures.py`. Where
  `plot_code` or (behind the switches) `plot_data` runs, every draw is
  booked on the turn (`TurnScope.figures`; else the helper's record):
  under its name, a version of it (`plan_v2`) or a name its draw
  `replaces`, so the cap of three counts per figure per turn by any helper
  (the figure helper's per-run count still applies), and `replaces=`
  reaches a file another helper drew this turn. The code that drew each
  figure is kept in the working folder's hidden `.figure_code/`
  (`<name>.plot.py` + `<name>.data.json`, or `<name>.plot_data.json`);
  as a dot-folder it gets no card, is not mirrored to SharePoint, and
  neither the prompt's file list nor `list_files` shows it; a retired
  figure's code goes with it, and a module redraw of the figure drops it.
  A file is removed only by a draw under its own name or by `replaces=`:
  a version name only counts on the draws and says the earlier file still
  shows ("keep both" is a brief too; `boring_V-1` is no version). Behind
  the switches: `FigureTasks` serves the figure and calc helpers: a brief
  that names a figure with kept code (or a version of its name) gets a
  "FIGURE TO REDRAW" block with the code and data and is asked to keep the
  name; both helpers' results end with their record, naming each figure's
  code file. The main agent's own `plot_data` takes `replaces=`. On the
  calc helper `FigureLooks` (just inside `DeliveredToolResults`, so
  activity.jsonl shows its notes) asks for a look at a figure a module
  method drew (booked like a draw) and has `html_to_pdf` name embedded
  figures no look read. Tests: `deep/tests/test_figure_turn.py`,
  `webapp/tests/test_figure_code_cards.py`.
- **Notes on `plot_data`** (GM-109; 2026-10-18; w9 TRACE_REVIEW §2.7, fix
  10): `annotations` places free text where the caller puts it, in data
  units or fractions of the plot, with an optional leader to the point it
  marks, alignment and plain box, the same in the PNG and the Plotly twin
  (`profile_figure/DESIGN.md`). The adapter refuses a misspelt note key with
  a did-you-mean in the note's own words (`ha` -> `align`, `xy` -> `point`,
  `bbox` -> `box`), returns the notes as drawn, and the layout check's
  findings come back in `warnings`. `plot_code`'s description no longer
  lists leaders among what only it can draw. The notes are checked by the
  plotting box's own pixel check, moved for this to
  `geotech_common/layout_ink.py` (the box loads it by its file path, so the
  rest of `geotech_common` stays out of the box; behaviour and budgets
  unchanged). In the app's process it runs one at a time under a lock, and
  a draw on any other thread passes straight through (the review saw 300 of
  300 figures saved by another thread lose their line while a check ran);
  leader arrows now count as lines, in `plot_code` too. Tests:
  `profile_figure/tests/test_data_plot_notes.py`,
  `geotech_common/tests/test_layout_ink.py`,
  `funhouse_agent/tests/test_nested_keys.py`,
  `deep/tests/test_figure_runs.py`.
- **The step cap** (TA-33; 2026-10-16): the message for a turn that runs
  out of steps is plain words (no LangGraph error name, no documentation
  link; the raw text goes to the activity log). Behind
  `GEOTECH_STEP_CAP_ANSWER` (OFF; arm `step_cap_answer`) such a turn
  answers from what it gathered instead: one last model call with no
  tools over the turn's own replies and tool results (each cut to 4,000
  characters), told to say what was found, what was not finished and
  that it stopped at its step limit (`webapp/core.answer_at_step_cap`).
- **Large results and the record** (TA-30, EV-20; 2026-10-15/16): a JSON
  result of 60,000 characters or more opens with a map of its keys (the
  line each starts on, what it holds), so the preview deepagents shows of
  a result moved to scratch says where each key is
  (`deep/result_map.py`); each turn's engine record lists the tools,
  helpers and switches it ran with; an errored turn keeps its token
  count; the vision probe's calls are labelled `probe`.
- **`calculate`** (`deep/calculate_tool.py`, GM-94; 2026-10-15): a variable
  may be a list of numbers, used through count, sum, mean, median, std
  (sample), min, max, `percentile(L, p)`, `count_above(L, v)` and
  `count_below(L, v)`; a list in arithmetic is refused with a plain message
  and a long list is summarised in the echo. erf, erfc and the standard
  normal's norm_cdf / norm_pdf / norm_ppf join the shared math list
  (`adapters/reliability_adapter._MATH_FUNCS`), so `reliability`'s
  expressions have them too.
- **Judgment calls** (GM-116; 2026-10-16; the owner: "User's judgement is
  the one that matters"). Behind `GEOTECH_JUDGMENT_CALLS` (OFF; arm
  `judgment_calls`), on the geotech page only. The main prompt gets a
  "Judgment Calls" section (`deep/prompt.judgment_calls_section`, 963
  characters; 897 without the clause naming the `references` helper when
  that helper is not built): make each call as an experienced engineer
  would and give the answer, then end with a short "Judgment calls" list
  (a method, and whether the default analysis suits the geometry and
  loading; a value, factor of safety or load factor the user did not give,
  assumed inputs among them; drained or undrained; which stratum governs;
  search or mesh settings; how ambiguous data was read), each with its
  reason and reference, the alternative set aside and what would change
  the answer (a number only from a tool run); no list when there is no
  call. The `calc` helper ends its reply with the calls it made
  (`deep/agent._CALC_JUDGMENT_CALLS`); the references helper returns a
  reference's several methods or values as cited options, says which fits
  the question as asked, and leaves the choice to the user
  (`_REFERENCES_OPTIONS`). The review page's prompt, the figure, reviewer
  and general-purpose helpers and every tool description are unchanged,
  and off the build is byte-for-byte as before
  (`deep/tests/test_judgment_calls.py`). Measured by live-smoke flows
  F61 / F61j and `expect.answer_names` (EV-25, EV-26).
  **GM-155 (2026-10-18, after live wave w12 measured four pairs):** no ON
  answer cited a reference, assumptions reached two PDFs as "Given", K0
  (a tool default named only in its notes) fell out of one list, and one
  ON answer ran fewer alternatives than OFF. The section (now 1,651
  characters; 1,544 without the helper) makes a tool's defaults calls
  (`choices` not marked given, `defaults_used` where the answer turns on
  them, what its notes say it assumed); each call's "why" names its
  source -- the basis in a tool's
  `choices`, a reference already looked up, or one consult of the
  `references` helper covering every call the answer turns on that has
  none -- or says it rests on judgment with no reference cited; "what
  would change the answer" is one tool run (one call; no second helper
  task or deliverable) for the call the answer turns on most and a
  direction for the rest, and that run adds to the
  alternatives, never replaces them; rounding and presentation are left
  out. The calc helper's text and the librarian's (several choices at
  once, a line or two each) follow. Unswitched, as plain corrections: the
  base prompt's confidence sentence asks for a number from a tool run or
  a reference looked up, else a direction (both arms invented
  sensitivity percentages); the delegation nudge and the calc helper keep
  the user's inputs apart from the assumptions; and every `*_package`
  opens with "Given Inputs" (what the call passed, less what it stated as
  assumptions) and "Assumed Inputs" (`defaults_used`, plus a new
  `assumptions` input, {input name: basis}, refused with a did-you-mean
  when a name is not one of the package's inputs), then the module's echo,
  retitled "Values Used" because it mixes given, assumed and derived
  values (`calc_package.tables.given_section` / `assumed_section`,
  `adapters/calc_package._given_inputs` / `_assumed_inputs`).

Every switch and its default: `review_flags.py` and
`module_work/REVIEW_HARNESS.md`.

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
