---
name: run-geotech-staff-engineer
description: Run, smoke-test, launch, or screenshot GeotechStaffEngineer — the analysis library (direct class-API calls), the LLM dispatch layer, and the Streamlit chat webapp. Use when asked to run the app, verify a module change works end-to-end, or drive the webapp.
---

# Run GeotechStaffEngineer

All paths are relative to the repo root (`GeotechStaffEngineer/`). This is a
Python package (31 analysis modules + `funhouse_agent/` LLM dispatch) with a
Streamlit chat webapp (`webapp/app.py`) on top. Nothing needs building — the
repo runs from source in the existing venv. Most PRs touch analysis modules or
adapters; verify those with the driver + direct invocation. The webapp is the
interactive surface; drive it with the AppTest harness (offline) or a real
`streamlit run` (live, needs the API key).

## Prerequisites (already satisfied on this machine)

- venv at `.venv/` (Python 3.14), all deps installed.
- `geotech-references/` submodule is an **editable install** (`pip install -e
  geotech-references/`) — reference changes are live from source.
- Live LLM runs read `ANTHROPIC_API_KEY` from the Windows User env at runtime.
  Never echo it into chat or transcripts.

## Run (agent path) — the smoke driver, FIRST

```bash
.venv/Scripts/python .claude/skills/run-geotech-staff-engineer/driver.py all
```

Three subcommands, each also runnable alone (`library` / `dispatch` /
`apptest`); prints `PASS`/`FAIL` per smoke, exit 0/1. All offline — no API key
read, no network:

- **library** — direct class-API call into `bearing_capacity` (the layer most
  PRs touch), asserts qult ≈ 1158.8 kPa.
- **dispatch** — the same calc through `funhouse_agent.dispatch.call_agent`
  (the flat-JSON surface the LLM tools use), asserts identical numbers.
- **apptest** — boots `webapp/app.py` headless via
  `streamlit.testing.v1.AppTest` with the engine + stream mocked, sends one
  chat message, asserts the transcript persisted. ~30 s; a
  "missing ScriptRunContext" warning is normal noise.

## Direct invocation (no app, no key)

Analysis modules export **classes**, not top-level `analyze_*()` functions
(`compute()` → result dataclass with `.summary()` / `.to_dict()`):

```bash
.venv/Scripts/python -c "
from bearing_capacity import BearingCapacityAnalysis, BearingSoilProfile, Footing, SoilLayer
soil = BearingSoilProfile(layer1=SoilLayer(friction_angle=32.0, unit_weight=19.0))
r = BearingCapacityAnalysis(footing=Footing(width=2.0, depth=1.0, shape='square'),
                            soil=soil, vertical_load=500.0).compute()
print(r.summary())"
```

The LLM tool surface (what the agent actually calls):

```bash
.venv/Scripts/python -c "
from funhouse_agent.dispatch import call_agent, list_methods
print(list_methods('bearing_capacity'))
print(call_agent('bearing_capacity', 'bearing_capacity_analysis',
      {'width': 2.0, 'depth': 1.0, 'shape': 'square',
       'friction_angle': 32.0, 'unit_weight': 19.0, 'vertical_load': 500.0}))"
```

Tests: `pytest <module>/ -v` per module; `pytest webapp/tests -q` for the app;
full gate `pytest -q` (~8,600 tests, long — run per-module while iterating).

## Run (live webapp)

```bash
.venv/Scripts/streamlit.exe run webapp/app.py --server.headless true --server.port 8501
```

Then open http://localhost:8501 (humans double-click `Start Geotech App.bat`
instead — same thing plus auto-opening the browser). First render takes
~5–10 s. With `ANTHROPIC_API_KEY` present the sidebar shows "Using Claude via
the Anthropic API (…)" and chat turns run the real agent; with no key it boots
to a "No engine configured" banner by design. Verified live: a bearing-capacity
question through the chat returns the same qult/qall the driver computes.

Conversation data (transcripts, uploads) persists under `~/.geotech_webapp/`
— set `GEOTECH_WEBAPP_DATA=<tmpdir>` to sandbox a session away from the
owner's real conversations. Stop with Ctrl-C. If launched as a background
shell task, killing the wrapper shell leaves `streamlit.exe` alive on Windows
— finish with `Get-Process | Where-Object { $_.ProcessName -match 'streamlit' }
| Stop-Process -Force` and confirm the port is closed.

## Gotchas (all hit for real)

- **`GEOTECH_WEBAPP_MODEL` won't stick on a machine with history.** It only
  sets the *startup default*; the app resumes the most recent saved
  conversation and that conversation's remembered model wins, and new
  conversations inherit the sidebar's sticky selection. To actually change
  model, use the sidebar **Model** picker (Session → Model), or sandbox with
  `GEOTECH_WEBAPP_DATA` so there is no history to resume.
- **Dispatch wants exact module names** — `call_agent('bearing', ...)` returns
  an "Unknown module" error listing valid names; `bearing_capacity` works.
  `list_methods(<module>)` first.
- **Result dict keys are unit-suffixed** (`q_ultimate_kPa`, not `qult`).
  `.summary()` is the human view; `.to_dict()` for asserting values.
- **AppTest needs the engine mocked** (see `driver.py` or
  `webapp/tests/test_app_smoke.py`: monkeypatch
  `webapp.engine_config.resolve_engine`, `core.build_agent`,
  `core.stream_turn`) — otherwise a present API key builds a real engine and a
  chat turn spends money.
- **Headless screenshots need real wall-clock wait.** Streamlit renders over a
  websocket; `chrome --headless --screenshot --virtual-time-budget=…` captures
  only the loading skeleton. Use CDP with a ~15 s sleep before
  `Page.captureScreenshot` (working script pattern: launch chrome with
  `--remote-debugging-port`, PUT `/json/new?<url>`, connect the tab's
  websocket with the venv's `websockets` lib).
- The app's sidebar version caption can lag the repo (`5.9.0` shown while
  `pyproject.toml` said 5.9.1) — it reads installed-package metadata, not the
  source tree. Cosmetic.

## Troubleshooting

- `ImportError: cannot import name 'analyze_bearing_capacity'` — that function
  doesn't exist; use the class API above.
- `KeyError: 'qult'` on a result dict — keys are unit-suffixed
  (`q_ultimate_kPa`).
- `BearingSoilProfile(layers=[...])` → TypeError — the kwargs are `layer1`
  (and optional `layer2`, `gwt_depth`), not a list.
- Blank dark page in a fresh browser hit — first render still in flight; wait
  ~10 s and reload.
