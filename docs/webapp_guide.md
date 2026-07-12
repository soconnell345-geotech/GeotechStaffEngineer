# Web Chat App — Owner Guide

A browser-based chat over the GeotechStaffEngineer deep agent. It's the same
agent you drive from the Databricks notebook (`funhouse_agent.deep`), wrapped in
a web UI so you (or a reviewer) can use it from a plain URL — locally today, and
in the State Dept **TinyApp** environment for hosting.

The app lives in `webapp/`. It runs standalone with one command and needs no
analysis code of its own — it's UI and glue over the shipped modules.

---

## What it does

- **Chat** with the geotech agent; answers stream in token-by-token, with the
  tool activity (which analysis module ran, with what inputs) shown in a
  collapsible status area.
- **Upload files** — PDF cross-sections, DXF drawings, CSV/DIGGS data, images.
  Each upload is both registered for the vision tools *and* written to a
  session temp folder so the real-path importers (`pdf_import`, `dxf_import`,
  `drawing_ir`, `read_pdf_text`) can open it. The agent is told the attachment
  key and the file path automatically.
- **Download artifacts** — when the agent produces a calc package (HTML), a DXF,
  or a plot, it appears as a download button in the sidebar.
- **Token counts** — per-turn and running conversation totals (the same metering
  the notebook chat uses), so you can watch spend.
- **Persistent conversations** — every chat is **saved automatically** and
  **survives restarts**. The sidebar **Conversations** list shows your saved chats
  (most-recent first, auto-titled from the first message); click one to resume it
  (the agent remembers the thread, your uploads and the artifacts it produced),
  **➕ New conversation** to start fresh, ✏️ to rename, 🗑️ to delete (a soft
  delete — moved to a `.trash/` folder, not erased). Conversations live under
  `~/.geotech_webapp/` by default, or wherever you point `GEOTECH_WEBAPP_DATA`.
- **Prominent professional-use disclaimer** at the top of every session.

---

## Run it locally (today)

You already have the agent installed in the project `.venv`. Add streamlit once,
set your key, and run:

```powershell
.venv\Scripts\pip install streamlit
$env:ANTHROPIC_API_KEY = "sk-..."
.venv\Scripts\streamlit run webapp\app.py
```

Streamlit opens `http://localhost:8501` in your browser. That's it.

- Default model is `claude-opus-4-8`. Change it with `GEOTECH_WEBAPP_MODEL`
  (e.g. `claude-sonnet-5`) if you want a cheaper/faster run.
- If you launch it with **no** key set, it still opens and shows a clear
  "no engine configured" banner — nothing crashes.

Full run details (including the Databricks driver-proxy recipe, which is
flagged *needs live verification*) are in `webapp/README.md`.

---

## Screenshots

Deferred — capture these on first real run and drop them here:

- the chat with a streamed answer + tool-activity panel,
- a file upload + the agent reading it,
- an artifact download button.

---

## TinyApp submission checklist

TinyApp (Azure Government) hosts application code you submit; Streamlit/Python is
a listed stack, and **Prompter** is a listed data connection.

1. **Submit** `webapp/app.py`, the `webapp/` package (`core.py`,
   `engine_config.py`), and `webapp/requirements.txt`.
2. **Wire the engine.** Either the platform exposes `ANTHROPIC_API_KEY` (nothing
   else to do), or register the Prompter-backed model at startup:
   `webapp.engine_config.register_model_builder(lambda: PrompterChatModel(fh_prompter))`.
   We do not ship a Prompter HTTP client — the deployment provides the SDK.
3. **Cold-open test.** Confirm the app renders (banner + disclaimer) before any
   engine is wired — reviewers may click it cold.
4. **Smoke test.** Ask a simple question (e.g. "Bearing capacity of a 2 m strip
   footing, phi=30, gamma=18"), upload a small PDF and ask the agent to read it,
   and confirm an artifact download appears when you ask for a calc package.

---

## How it's built (for maintenance)

- `webapp/core.py` and `webapp/engine_config.py` hold **all** the logic and are
  import-testable without streamlit or a live model. `webapp/app.py` is a thin
  streamlit view over them.
- The streaming/token parsing is **reused** from
  `funhouse_agent/deep/notebook.py` (the notebook chat's pure formatters), so
  the two front-ends stay in sync.
- Offline tests: `pytest webapp/tests/ -q` (attachment staging, artifact
  capture, engine resolution incl. the no-engine path, disclaimer, a fake-agent
  stream). No API key needed.
- The app imports from `funhouse_agent` but changes nothing in it.
