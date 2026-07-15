# Foundry deployment plan — the webapp on Palantir Foundry (DOS OBO)

**Status: research memo (2026-07-15), owner-requested exploration. Nothing
built yet.** Goal: run the Streamlit chat app on the DOS OBO Foundry instance
with MINIMAL work living in Foundry itself — same philosophy as the Databricks
launcher (everything ships in the pip package; Foundry holds a stub + config).

## TL;DR — it fits, with two load-bearing platform features

1. **Hosting: Code Workspaces publishes Streamlit apps natively.** Foundry's
   JupyterLab Code Workspaces supports building, previewing, and PUBLISHING
   Streamlit applications as first-class Foundry apps (shareable, behind
   Foundry auth). No container engineering, no Compute Modules needed. The
   Foundry-side footprint is: one Code Workspace, our package installed via
   the Libraries panel (pip — already proven on this instance), and a ~5-line
   `app.py` stub that launches the packaged webapp.
2. **LLM access: Foundry exposes an Anthropic-Messages-compatible proxy.**
   `POST /api/v2/llm/proxy/anthropic/v1/messages` — same wire shape as the
   Anthropic API, auth = `Authorization: Bearer {FOUNDRY_TOKEN}`, model =
   a Foundry RID (e.g. `ri.language-model-service..language-model.
   anthropic-claude-4-6-opus`). Because our whole deepagents stack runs on
   `langchain-anthropic` (ChatAnthropic), the agent — tool calling, streaming,
   sub-agents, vision — should run UNMODIFIED against this proxy with only a
   base_url + bearer-token + model-RID override. There is also an
   OpenAI-compatible proxy (`/api/v2/llm/proxy/openai/v1/chat/completions`)
   as fallback, which our Databricks `PrompterChatModel` bridge pattern
   already knows how to drive.

Rejected/irrelevant paths, for the record: **Compute Modules** (container
hosting, but function/pipeline execution modes only — no browser-facing HTTP
mode); **Workshop/Slate** (their own UI builders, not Python servers; Workshop
CAN iframe-embed the published Streamlit app later if wanted); the
`palantir_models`/`language_model_service_api` transform bindings (aimed at
pipelines; the proxy endpoints are the right surface for an interactive app).

## Structural differences vs the local app

| Concern | Local (today) | Foundry (planned) |
|---|---|---|
| Model + auth | ChatAnthropic + Windows-env ANTHROPIC_API_KEY | ChatAnthropic + proxy base_url + Foundry bearer token + model RID (no external egress needed) |
| Model picker | Claude model ids | Foundry model RIDs (catalog names differ per enrollment; DOS OBO list TBD) |
| App serving | `streamlit run` on localhost | Code Workspaces "Publish application" (Foundry-hosted, Foundry auth) |
| Package install | local .venv | Workspace Libraries panel (pip; proven) or wheel via an Artifacts repo |
| Conversations (`~/.geotech_webapp`) | user home, durable | container FS — durability TBD; `GEOTECH_WEBAPP_DATA` already relocates it; v1 may be ephemeral |
| Reference PDFs (figure read-off) | repo `docs/` | upload PDFs to workspace files + `GEOTECH_REFERENCES_DOCS` (same as Databricks) |
| Tracing | GEOTECH_TRACE JSONL | works as-is (local file) |

## Auth wrinkle (known, solvable)

The Anthropic SDK sends `x-api-key` by default; the proxy wants
`Authorization: Bearer`. The anthropic client supports bearer via
`auth_token=` (and/or default_headers), so the engine shim sets that. A
Palantir community thread confirms this exact mismatch is the common failure
mode — ~15 lines in our engine_config, not a redesign.

## Build items (all in OUR repo; Foundry holds ~5 lines)

- **F-A — Foundry engine path in `webapp/engine_config.py`**: detect Foundry
  (env: `FOUNDRY_TOKEN` / hostname vars present in workspaces), build
  ChatAnthropic with `base_url=https://<stack>/api/v2/llm/proxy/anthropic`,
  bearer auth, RID model ids; model picker entries from a small RID map env/
  config. Offline tests with a fake proxy server.
- **F-B — `webapp/foundry_entry.py` + `docs/FOUNDRY.md`**: the stub the
  Foundry workspace's `app.py` calls (locates the installed `webapp/app.py`
  and execs it), plus a click-path doc (create workspace → Libraries install →
  stub → Preview → Publish).
- **F-C (fallback, only if the Anthropic proxy is disabled on DOS OBO)** —
  OpenAI-proxy engine reusing the `NativeToolEngine`/OpenAI-client bridge.
- **F-D — storage**: point `GEOTECH_WEBAPP_DATA` at the workspace persistent
  volume if the published app shares it (verify on-platform); else document
  ephemeral-conversations for v1; dataset-backed export = later item.

Effort guess: F-A + F-B are small (the Databricks launcher was the same
shape); the risk is entirely in platform-side unknowns below.

## What only the owner can verify on DOS OBO (before we build)

1. **Model catalog**: which Claude (or GPT) models are enabled, and their
   RIDs (Model catalog app). Gov enrollments often have a restricted list.
2. **Proxy availability**: is the LLM-provider-compatible API
   (`/api/v2/llm/proxy/anthropic/...`) enabled? (Settings/AIP enablement;
   a 1-line curl from a workspace terminal answers it.)
3. **Code Workspaces app publishing** enabled on the enrollment.
4. **Package route**: does the Libraries panel reach a PyPI mirror with
   `geotech-staff-engineer`, or do we upload the wheel to an Artifacts repo?
5. **Published-app runtime**: do installed libraries + files persist into the
   published app container, and is there a writable durable path?

## Relationship to the existing foundry/ wrappers

The 48 `foundry/` agent-wrapper files (tools fed to a prepackaged AIP Agent)
are unchanged and orthogonal — that path gives Foundry's agent our TOOLS;
this plan hosts OUR agent + UI as a Foundry app. They can coexist.

## Sources

- Code Workspaces overview (Streamlit/Dash publishing):
  https://www.palantir.com/docs/foundry/code-workspaces/overview and
  https://www.palantir.com/docs/foundry/code-workspaces/jupyterlab
- Build-with-AIP example (Streamlit in Jupyter Code Workspaces):
  https://build.palantir.com/platform/578b38f7-cfaf-4ea1-817e-0b1189a62ac0
- Anthropic Messages proxy:
  https://www.palantir.com/docs/foundry/api/v1/llm-apis/models/anthropic-messages-proxy
- OpenAI chat-completions proxy:
  https://www.palantir.com/docs/foundry/api/llm-apis/models/openai-chat-completions-proxy
- LLM-provider-compatible APIs:
  https://www.palantir.com/docs/foundry/aip/llm-provider-compatible-apis
- Compute modules (rejected for hosting):
  https://www.palantir.com/docs/foundry/compute-modules/overview and
  https://www.palantir.com/docs/foundry/compute-modules/execution-modes
- Community: Anthropic SDK auth vs the proxy:
  https://community.palantir.com/t/auth-failing-for-anthropic-agents-sdk-and-llm-proxy/6255
