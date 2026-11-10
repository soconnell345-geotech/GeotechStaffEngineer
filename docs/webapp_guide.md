# Web Chat App — Owner Guide

A browser-based chat over the GeotechStaffEngineer deep agent. It's the same
agent you drive from the Databricks notebook (`funhouse_agent.deep`), wrapped in
a web UI so you (or a reviewer) can use it from a plain URL — locally, on a
Databricks cluster and on Palantir Foundry. It has two pages: Document Review
(the app root) and GeotechStaffEngineer (`/geotech`). The **Tiny Apps** host
runs Document Review alone, from its own package `aec-document-review`.

The app lives in `webapp/`, on the shared core `aec-agent-core`
(`packages/aec-agent-core/`). It runs standalone with one command and needs no
analysis code of its own — it's UI and glue over the shipped modules.

---

## What it does

- **Chat** with the geotech agent; answers stream in token-by-token, with the
  tool activity (which analysis module ran, with what inputs) shown in a
  collapsible status area.
- **Upload files** — PDF cross-sections, DXF drawings, CSV/DIGGS data, images.
  Each upload is both registered for the vision tools *and* written to a
  session temp folder so the real-path importers (`open_document` and the
  other document-review tools, `pdf_import`, `dxf_import`, `drawing_ir`,
  `read_pdf_text`) can open it. The agent is told the attachment
  key and the file path automatically. Uploads are capped at **25 MB** per
  file (the app's own websocket-transport limit, not Streamlit's); for bigger
  documents put the file in SharePoint and ask the agent to fetch it by name —
  see "Large files" in `webapp/README.md`.
- **Download artifacts** — when the agent produces a calc package (HTML), a DXF,
  or a plot, it appears as a card under the reply, with a download button
  (a chart as an interactive chart, a figure as an image).
- **Model picker** — a sidebar **Model** selector: Opus 4.8 (deepest reasoning,
  default), Sonnet 5 (fast + capable), Haiku 4.5 (quick questions). Switch to a
  cheaper/faster model any time — the choice applies to the current conversation
  going forward (history/uploads/artifacts are kept) and is remembered on resume.
  Per-model speed and price are the point: use Haiku/Sonnet for quick lookups,
  Opus for the hard reasoning.
- **Token counts** — per-turn and running conversation totals (the same metering
  the notebook chat uses), so you can watch spend.
- **Persistent conversations** — every chat is **saved automatically** and
  **survives restarts**. The sidebar **Conversations** list shows your saved chats
  (most-recent first, auto-titled from the first message); click one to resume it
  (the agent remembers the thread, your uploads and the artifacts it produced),
  **➕ New conversation** to start fresh, ✏️ to rename, 🗑️ to delete (a soft
  delete — moved to a `.trash/` folder, not erased). Conversations live under
  `~/.geotech_webapp/` by default, or wherever you point `GEOTECH_WEBAPP_DATA`.
- **Find and restore** — a filter box searches every saved conversation by
  title once there are more than a few; on Databricks, **Find a past
  conversation** under Permanent storage lists what is mirrored in SharePoint
  and **Restore** pulls a conversation (record and files) back after a
  cluster restart has wiped the driver disk.
- **Feedback** — a sidebar box saves a note with the conversation
  (`FEEDBACK.md`); the agent has a matching `record_feedback` tool it uses
  when it lacks a capability the task needs, so exports carry the gaps in its
  own words.
- **Full activity archive** — every tool call, result and model call of every
  turn, sub-agents included, goes to `activity.jsonl` in the conversation
  folder regardless of the "Show turn details" toggle (which only controls the
  on-screen summary).
- **Long turns survive errors** (from the release after 5.33.0) — a turn that
  ends on an error or a rate limit keeps its tool results, so asking the
  agent to continue picks up where it stopped; busy refusals are waited out,
  and every model request has a time limit. Details: "Long turns, errors and
  rate limits" in `webapp/README.md`.
- **Prominent professional-use disclaimer** at the top of every session.

---

## Run it locally (today)

You already have the agent installed in the project `.venv` (once: the core,
then the app, `.venv\Scripts\pip install -e packages\aec-agent-core -e .`;
Python 3.11 or newer). Set your key and run:

```powershell
$env:ANTHROPIC_API_KEY = "sk-..."
.venv\Scripts\streamlit run webapp\app.py          # the geotech page alone
.venv\Scripts\streamlit run webapp\pages_entry.py  # both pages
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

## Tiny Apps deployment

Tiny Apps hosts Document Review alone. The deployment is a thin wrapper repo
over the released package `aec-document-review` (with `aec-agent-core` and
`planlens`, pinned exactly): `tinyapps/wrapper_repo/` (`app.py`,
`packages.txt`, `requirements-thirdparty.txt`, `run.sh`, `.env.example`).
Settings, per-tester keys, storage, the dosdev recipe, the owner's
first-round checklist and the questions for CfA are in
`tinyapps/TINYAPPS.md`, the plan of record; open work is in `BACKLOG.md`
(TA-nn, OWN-44, INF-34). The older submit-`app.py`-and-`requirements.txt`
checklist that stood here was retired on 2026-10-12.

---

## How it's built (for maintenance)

- `webapp/core.py` and the core's
  `packages/aec-agent-core/aec_agent_core/engine_config.py` (still importable
  as `webapp.engine_config`) hold **all** the logic and are import-testable
  without streamlit or a live model. `webapp/app.py` is a thin streamlit view
  over them.
- A change to the Document Review page goes to
  `packages/aec-document-review/aec_document_review/`; this repository's
  copy of the review files is frozen (`webapp/tests/test_review_freeze.py`)
  until the split's step 15.
- The streaming/token parsing is **reused** from
  `funhouse_agent/deep/notebook.py` (the notebook chat's pure formatters), so
  the two front-ends stay in sync.
- Offline tests: `pytest webapp/tests/ -q` (attachment staging, artifact
  capture, engine resolution incl. the no-engine path, disclaimer, a fake-agent
  stream). No API key needed.
- The app imports from `funhouse_agent` but changes nothing in it.
