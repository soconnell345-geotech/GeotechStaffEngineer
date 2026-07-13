# GeotechStaffEngineer — Web Chat App

A [Streamlit](https://streamlit.io) chat interface over the GeotechStaffEngineer
deep agent. Ask geotechnical questions in plain language; the agent drives the
industry-standard analysis modules and reference lookups and streams its answer
back, with file upload (PDF/DXF/CSV/images), artifact downloads (calc packages,
DXFs, plots), and live token counts.

It runs **standalone** (one command, on your machine) and is the submission
target for the State Dept **TinyApp** hosting environment (Streamlit / Python).

> **Analysis/research aid — not a design deliverable.** Every result must be
> independently reviewed by a licensed professional engineer familiar with the
> site. The full professional-use disclaimer is shown at the top of the app.

---

## 1. Run it locally (one command after install)

From a published install:

```bash
pip install "geotech-staff-engineer[deep,full,webapp]"
export ANTHROPIC_API_KEY=sk-...          # Windows PowerShell: $env:ANTHROPIC_API_KEY="sk-..."
streamlit run webapp/app.py
```

Streamlit prints a local URL (default `http://localhost:8501`) and opens it in
your browser. That's the whole thing.

**From this repo (development)** — the app imports from the source tree, so just
add streamlit to the project venv and run:

```bash
.venv/Scripts/pip install streamlit          # one-time; the [webapp] extra
.venv/Scripts/streamlit run webapp/app.py
```

(`langchain-anthropic` and the analysis backends are already in the project
`.venv`; only `streamlit` is new.)

### Choosing the model

Pick the model **in the sidebar** ("Model"): **Opus 4.8** (deepest reasoning,
default), **Sonnet 5** (fast + capable), **Haiku 4.5** (quick questions) — trading
speed/cost against depth. The choice applies to the **current conversation going
forward** (your history, uploads and artifacts are kept; the next turn replays the
conversation into the new model) and is remembered when you resume it.

`GEOTECH_WEBAPP_MODEL` still sets the startup default (and, if it names a model not
in the curated list, adds it to the picker). `GEOTECH_WEBAPP_MAX_TOKENS` caps the
per-response output.

```bash
export GEOTECH_WEBAPP_MODEL=claude-sonnet-5      # startup default (any Claude id)
export GEOTECH_WEBAPP_MAX_TOKENS=8192            # per-response output cap
```

### No key configured?

If neither `ANTHROPIC_API_KEY` nor a deployment-provided engine is present, the
app **still boots** and shows a clear "No engine configured" banner in the
sidebar (important: TinyApp reviewers may open it cold). It just can't answer
until an engine is provided.

---

## 2. Conversations (auto-saved, resumable)

Every conversation is **saved automatically** as you go and **survives app
restarts** — nothing to click. The sidebar **Conversations** list shows your
saved chats (most-recent first, auto-titled from the first message); click one to
resume it, **➕ New conversation** to start fresh, ✏️ to rename, 🗑️ to delete.
Delete is a **soft delete** — the conversation directory is moved to a `.trash/`
folder under the data root, not erased.

**Where it's stored.** Under `$GEOTECH_WEBAPP_DATA` if set, else
`~/.geotech_webapp/`. Each conversation is a directory
`conversations/<thread_id>/` holding the display transcript, the agent-facing
message history (replayed on resume so the model remembers the thread), the
staged uploads, the produced artifacts, and a small metadata file.

```bash
export GEOTECH_WEBAPP_DATA=/data/geotech_webapp   # PowerShell: $env:GEOTECH_WEBAPP_DATA="D:\geotech_webapp"
```

**Durability mechanism (honest note).** Resume works by **persisting and
replaying the agent's message history** — dependency-free, using tools already in
our tree. We do *not* depend on a durable LangGraph checkpointer: `langgraph`'s
SQLite saver is a separate optional package not currently pinned. `build_agent`
accepts a `checkpointer=` argument so a durable saver (e.g.
`langgraph-checkpoint-sqlite`) can be dropped in later without other changes; the
in-tree `InMemorySaver` covers within-session thread state.

> ⚠️ **TinyApp storage caveat.** Whether the TinyApp environment gives the app a
> **persistent, writable** filesystem is unknown until the platform answer. If it
> does, set `GEOTECH_WEBAPP_DATA` to that persistent volume and conversations
> survive restarts/redeploys. If storage is **ephemeral**, conversations live
> only for the container's lifetime (the app still works; it just won't persist
> across a redeploy). Confirm with the hosting team.

---

## 3. Databricks (notebook-launched, via the driver proxy)

Run the app on a Databricks cluster and reach it through the **driver proxy
URL** — and drive it with the **Funhouse Prompter** engine, no Anthropic key.
This is the TinyApp dress rehearsal: same Prompter data connection, same app.

> ⚠️ **Needs live verification.** The exact proxy path and server flags vary by
> workspace/runtime; the launcher encodes the recipe below, but confirm on your
> cluster.

### Preferred: the one-cell launcher

```python
# In a Databricks notebook cell:
%pip install "geotech-staff-engineer[deep,full,webapp]"
dbutils.library.restartPython()
```

```python
from webapp.databricks_launcher import run_on_databricks

handle = run_on_databricks(port=8501, model="funhouse-gpt-high")
print(handle.url)          # open this in your browser (prints automatically too)
# ... when you're done:
# handle.stop()
```

That's the whole thing. `run_on_databricks` reads the driver-proxy identifiers
from spark, writes a small **bootstrap script**, and launches streamlit as a
background process on the driver. The bootstrap runs *in the streamlit process*
and, before the app resolves its engine:

1. **reconstructs a `PrompterAPI` on the driver** and registers a
   `PrompterChatModel` on it as the app's engine, then
2. starts streamlit under the proxy base path.

**Why reconstruct instead of passing `fh_prompter`?** The streamlit process is a
separate process — it never sees the notebook's `register_model_builder(...)`
call, and the notebook's live `fh_prompter` object can't cross the process
boundary. But `PrompterAPI` **self-configures** from the driver's Funhouse config
/ environment (the SDK's own code constructs `PrompterAPI()` with no arguments),
so the bootstrap rebuilds an equivalent client on the driver. If that fails (no
Funhouse config present), it **falls back automatically to `ANTHROPIC_API_KEY`**.

**Using the Anthropic key instead of (or as a fallback to) Prompter:**

```python
handle = run_on_databricks(
    port=8501,
    anthropic_key=dbutils.secrets.get("scope", "anthropic_key"),
)
```

The key is threaded into the streamlit subprocess env; if the Prompter can't be
built on the driver, the app uses the key. Other options: `spark=spark` (if not
auto-detected), `org_id=`/`cluster_id=` (override the proxy ids),
`workspace_host=` (if the host can't be read from spark, so the URL prints).

### Fallback: manual subprocess (loses the Prompter registration)

If you can't use the launcher, run streamlit directly — but note this plain
subprocess **does not** register the Prompter (a fresh process can't reach the
notebook's `fh_prompter`), so you must supply `ANTHROPIC_API_KEY`:

```python
import os, subprocess, sys
os.environ["ANTHROPIC_API_KEY"] = dbutils.secrets.get("scope", "anthropic_key")
org_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterOwnerOrgId")
cluster_id = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
port = 8501
base = f"/driver-proxy/o/{org_id}/{cluster_id}/{port}"
subprocess.Popen([
    sys.executable, "-m", "streamlit", "run", "webapp/app.py",
    "--server.port", str(port), "--server.address", "0.0.0.0",
    "--server.baseUrlPath", base, "--server.enableCORS", "false",
    "--server.enableXsrfProtection", "false", "--server.headless", "true",
])
print(f"Open: https://<your-workspace-host>{base}/")
```

If the page doesn't load, check: the port isn't already in use; the
`baseUrlPath` matches the proxy path exactly (trailing slash on the opened URL);
and the cluster allows the driver-proxy for your workspace. The local path
(section 1) is the reliable fallback.

---

## 4. TinyApp deployment (Azure Government)

TinyApp hosts application code you submit (Streamlit/Python is a first-class
stack). Submit:

- `webapp/app.py` (+ the `webapp/` package: `core.py`, `engine_config.py`)
- `webapp/requirements.txt` — pins `geotech-staff-engineer[deep,full,webapp]`

**Engine via the "Prompter" data connection.** TinyApp lists *Prompter* as an
allowed data connection. When the deployment provides a Prompter/`fh_prompter`
LLM client, register it as the app's model at startup (e.g. in a small
`startup`/`bootstrap` hook the platform runs before `app.py`):

```python
from webapp.engine_config import register_model_builder
# The deployment supplies a LangChain-compatible chat model wrapping the Prompter
# client. We do NOT ship a Prompter HTTP client — the environment provides the SDK.
register_model_builder(lambda: PrompterChatModel(fh_prompter))
```

`resolve_engine()` then prefers that model over `ANTHROPIC_API_KEY`. If the
platform instead exposes an Anthropic key as an env var, set `ANTHROPIC_API_KEY`
and no registration is needed. If neither is present the app boots into the
"no engine configured" banner rather than erroring — safe for a cold reviewer
click.

**Checklist**
- [ ] `requirements.txt` present and installs cleanly.
- [ ] Engine wired: Prompter builder registered **or** `ANTHROPIC_API_KEY` set.
- [ ] App opens cold (no engine) without a traceback — shows the banner.
- [ ] Disclaimer renders at the top.
- [ ] File upload → the agent can read the staged file (see `webapp_guide.md`).
- [ ] `GEOTECH_WEBAPP_DATA` pointed at a **persistent** volume if conversations
      should survive redeploys (else they are per-container; see section 2).

---

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit shell — a thin view over `core.py` (no logic). |
| `core.py` | All logic: attachment staging, artifact capture, streaming, disclaimer, and conversation **persistence** (per-conversation dir under `GEOTECH_WEBAPP_DATA`: transcript / messages / attachments / artifacts / meta; list / resume / rename / delete-to-trash). Import-testable without streamlit. |
| `engine_config.py` | Env-driven engine resolution (Anthropic key / Prompter hook / no-engine). |
| `databricks_launcher.py` | One-cell Databricks launcher (`run_on_databricks`): reconstructs the Prompter engine on the driver and runs streamlit under the driver proxy (§3). |
| `requirements.txt` | TinyApp dependency pin. |
| `tests/` | Offline tests for `core.py` + `engine_config.py` (no streamlit, no live model). |

Owner-facing overview: `docs/webapp_guide.md`.

Run the tests: `pytest webapp/tests/ -q`
