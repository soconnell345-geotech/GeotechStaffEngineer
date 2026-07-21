# Running the geotech web chat app on Palantir Foundry (Code Workspaces)

Design + rationale: `module_work/FOUNDRY_APP_PLAN.md`. This is the do-it page.
Everything lives in the pip package; the Foundry side is one workspace, a
2-line file, and a few environment variables.

## 1. Create the workspace and install the package

1. In Foundry: **Code Workspaces → New workspace → JupyterLab**.
2. In the workspace **Libraries** panel, install (PyPI):
   - `geotech-staff-engineer`  — as of the release after 5.8.0, this single
     package brings everything: all analysis backends, the deep agent,
     Streamlit, PDF reporting, and both LLM clients (`langchain-openai` for
     GPT-family RIDs, `langchain-anthropic` for Claude RIDs).
   - (On 5.8.0 or older, use the old form: `geotech-staff-engineer[deep,full,pdf]`
     plus `streamlit` plus `langchain-openai`.)

## 2. The app file (the only code that lives in Foundry)

Create `app.py` at the repo root of the workspace:

```python
from webapp.foundry_entry import main
main()
```

## 3. Point the app at your models (Foundry RIDs)

Find the model RIDs in the **Model catalog** app (each enabled model shows a
resource id like `ri.language-model-service..language-model.gpt-5-2`). Then
set environment variables for the workspace/app:

| Variable | Value | Why |
|---|---|---|
| `GEOTECH_FOUNDRY_MODELS` | `GPT 5.2=ri.language-model-service..language-model.gpt-5-2` (comma-separate several; `Label=RID` or bare RID) | Populates the in-app model picker; the FIRST entry is the default |
| `FOUNDRY_TOKEN` / `GEOTECH_FOUNDRY_TOKEN` | your Foundry token (Code Workspaces usually provides `FOUNDRY_TOKEN` already — check `env \| grep -i foundry` in the terminal) | Auth to the LLM proxy |
| `FOUNDRY_HOSTNAME` / `GEOTECH_FOUNDRY_HOST` | the stack host, e.g. `yourstack.palantirfoundry.com` (often preset too) | Where the proxy lives |
| `GEOTECH_WEBAPP_DATA` | a durable writable folder, e.g. a path under the workspace files | Saved conversations survive restarts |
| `GEOTECH_REFERENCES_DOCS` | folder of reference PDFs (optional) | Enables figure read-off, same as Databricks |

Any model id starting with `ri.` routes through the Foundry LLM proxy
automatically — RIDs containing `anthropic` use the Anthropic-messages proxy,
everything else uses the OpenAI-chat-completions proxy. **When Claude models
are enabled later, no code change is needed**: add the RID to
`GEOTECH_FOUNDRY_MODELS` (or paste it into the app's "Custom model id
(advanced)" box in the sidebar).

### 3b. Alternative engine: the in-platform `palantir_models` SDK (no proxy, no token)

On a Foundry deployment, a model id that is **not** an `ri....` RID is treated
as a **Palantir model API name** (e.g. `GPT_5_1`) and served through the
in-platform `palantir_models` SDK instead of the LLM proxy. The workspace
authenticates the SDK itself — **no token, host, or proxy enrollment is
involved**, which makes this the route of choice when the proxy returns
401/403 (`LlmProxyNotEnabled` and friends). Discovered live 2026-07-21: the
name resolves against `/llm/v3/completion/<modelApiName>`; a **404
`LanguageModelNotFound`** means a wrong name, a **403** means the name is right
but model access isn't granted yet (an AIP/model-permissions grant, not the
proxy enrollment).

Setup: add `palantir-models` and `language-model-service-api` in the workspace
**Libraries** panel, then either list the name in the picker env —
`GEOTECH_FOUNDRY_MODELS=GPT 5.1=GPT_5_1` — or type `GPT_5_1` into the sidebar
"Model RID or API name" box. Full tool calling is supported (the v3 request
carries OpenAI-style `tools`); wrapper: `webapp/palantir_sdk_engine.py`.
Current limit: image inputs are flattened to text on this route (the vision
tools' image legs need the proxy route or a future `MultiContentChatMessage`
leg).

Notebook smoke test for the SDK route (also how to find the right name — 404 =
wrong name, 403 = needs a model-access grant, answer text = working):

```python
from palantir_models.models import OpenAiGptChatLanguageModel
import language_model_service_api.languagemodelservice_api as base
import language_model_service_api.languagemodelservice_api_completion_v3 as v3
m = OpenAiGptChatLanguageModel.get("GPT_5_1")
r = m.create_chat_completion(v3.GptChatCompletionRequest(
    [base.ChatMessage(base.ChatMessageRole.USER, "Say OK")], max_tokens=20))
print(r.choices[0].message.content)
```

## 4. The 30-second proxy smoke test ("one curl")

Before publishing, open the workspace **Terminal** and paste (substitute your
RID; `$FOUNDRY_TOKEN` is usually already set):

```bash
curl -s "https://$FOUNDRY_HOSTNAME/api/v2/llm/proxy/openai/v1/chat/completions" \
  -H "Authorization: Bearer $FOUNDRY_TOKEN" -H "Content-Type: application/json" \
  -d '{"model":"ri.language-model-service..language-model.gpt-5-2","messages":[{"role":"user","content":"Say ok."}]}'
```

A JSON reply containing the model's answer = the proxy is enabled and the RID
is right. (For a Claude RID later, the same test against
`/api/v2/llm/proxy/anthropic/v1/messages` with the Anthropic request shape.)
An error page or 404 = the LLM-provider-compatible API isn't enabled on the
enrollment — ask the platform admin.

## 5. Preview, then publish

1. In JupyterLab, use the Streamlit **preview** on `app.py` to try it in the
   workspace.
2. **Publish application** → choose a Files-and-Projects location → file name
   `app.py`. The app gets a Foundry URL behind Foundry auth, shareable like
   any resource.
3. To upgrade later: bump `geotech-staff-engineer` in the Libraries panel and
   republish. The 2-line `app.py` never changes.

## Foundry deployment mode (5.9.1+): no API-key surface

The Foundry entry (`webapp/foundry_entry.py`) marks the process as a Foundry
deployment, and in that mode the app **never reads or mentions the Anthropic
API key** (enclave-IT requirement): the model surface is Foundry RIDs only —
the `GEOTECH_FOUNDRY_MODELS` entries populate the picker, and with nothing
configured the sidebar shows a single **"Model RID or API name"** input (no "advanced"
expander, no built-in Claude model list, no key references in banners or the
diagnostics report). Local/dev behaviour is unchanged.

## Troubleshooting a live deployment (learned 2026-07-17/18, State gov enclave)

Field notes from the first real publish (PDCS Sandbox, stateobo.palantirgov.com):

- **Upgrade recipe** (the only steps that matter):
  `maestro env pip install "geotech-staff-engineer==X.Y.Z"` in the workspace
  terminal → **Publish and sync** in the Applications tab → hard-reload the app
  (Ctrl+F5). Never `pip` directly (breaks the maestro lockfile), never
  uninstall first, and note "Restart workspace" does NOT update the published
  app — workspace and app are separate containers with separate lifecycles.
- **Which version is the app running?** The sidebar footer prints
  `geotech-staff-engineer X.Y.Z · engine: <source>` (5.8.2+). If the
  "Connection diagnostics" expander is missing, the app is on old code.
- **Publish fails with "failed to run startup scripts"**: read the log — the
  likely cause is an environment-restore pip conflict. Seen live: the
  Streamlit base image ships conda websockets 16.x, which leaked into the
  lockfile and clashed with langgraph-sdk (<16). Fixed by
  `maestro env pip install "websockets>=14,<16"` (and the package now pins
  this range itself).
- **Empty answers / flashing errors**: run the sidebar **Connection
  diagnostics** — it tests resolve / plain / streaming / tool-calling
  requests and prints failures verbatim.
- **401 Unauthorized from the LLM proxy on a gov enclave**: the container's
  `FOUNDRY_HOSTNAME` may point at a local sidecar (seen: `localhost:8080`)
  that rejects LLM-proxy paths, and direct egress to the stack domain is
  DNS-blocked. This is an enrollment/permission question, not an app bug.
  Ask the platform admin: (1) is the LLM provider-compatible API enabled for
  the enrollment; (2) is this account/project authorized for the enabled
  models; (3) what base URL + auth should Code Workspace containers use for
  the LLM proxy in this enclave. Their answer goes into
  `GEOTECH_FOUNDRY_HOST` (explicit `http://`/`https://` schemes are honored)
  in the app file — no package change needed.
- **Non-secret config lives in the app file** (there is no env-var panel):
  ```python
  import os
  os.environ["GEOTECH_FOUNDRY_MODELS"] = "GPT 5.1=ri.language-model-service..language-model.gpt-5-1"
  os.environ["GEOTECH_WEBAPP_MAX_TOKENS"] = "32000"
  os.environ["GEOTECH_TRACE"] = "1"
  from webapp.foundry_entry import main
  main()
  ```
  Secrets (tokens) must NOT go there — Foundry injects `FOUNDRY_TOKEN`.

## Notes / current limits

- Tool calling, streaming, and sub-agents ride the provider-compatible proxy
  unchanged. Vision (attachment analysis, figure read-off) depends on the
  enrollment's model supporting images through the proxy — try it, and prefer
  a multimodal RID for the default model.
- Conversations persist under `GEOTECH_WEBAPP_DATA`; if the published app's
  filesystem turns out not to be durable, point it at a durable mount (or
  accept per-session conversations until we wire a dataset-backed store).
- The old `foundry/` wrapper files (tools for the prepackaged AIP agent) are
  RETIRED and unrelated to this deployment.
