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
3. **Test wheels** (built from master, not on PyPI; e.g. the INF-25 set in
   `../foundry_handoff/gse_app_2026-10-20/`). Their versions carry a local
   label (`5.33.0+foundry.20261020`), which the sidebar footer shows. What
   the first run found (the AI FDE, 2026-10-20):
   - **Rename the files after a zip upload.** Unpacking a zip in the
     workspace dropped the `+` from the wheel names
     (`geotech_staff_engineer-5.33.0foundry.20261020-py3-none-any.whl`), and
     pip refuses a wheel whose name does not match its version. Put the `+`
     back (`5.33.0+foundry.20261020`) before installing; the version inside
     the wheel (its METADATA) is the one to copy.
   - **maestro does not take file paths** (`maestro env pip install ./x.whl`
     answers "No valid versions found"). What worked: every third-party
     dependency through the managed environment as usual, and the three
     local wheels installed by the app file at start-up with
     `pip install --no-deps --no-index --target <folder>` into a folder
     outside the managed environment, that folder first on `sys.path` and
     `PYTHONPATH`. Then the lockfile never lists the local builds and the
     published app installs them the same way. The clean route is a Foundry
     Artifacts repository (or conda channel) holding the three packages, so
     the environment can list them like any other dependency.
   - **Three pins the managed environment needed:** `websockets>=14,<16`,
     `polars>=1,<2` and `tenacity<9.2`. Streamlit's and Foundry's own
     libraries otherwise resolve newer versions than langgraph-sdk
     (websockets), pygef (polars) and google-genai (tenacity) accept, and
     `pip check` reports the conflict (`websockets 17.3`, `polars 2.0.0`).
     Add them on the conda side beside the app's dependencies.
   - The app needs numpy 2 (`np.trapezoid`); a fresh install of the INF-25
     set resolved numpy 2.5.4 on Python 3.12 and 2.4.6 on 3.11 (WSL,
     2026-10-20), and Foundry resolved 2.5.4 on Python 3.12.15.

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
| `FOUNDRY_PROXY_URL` | preset by Foundry in Code Workspaces and published apps (a `<proxy host>/…/foundry-proxy` address); nothing to set | Where the proxy lives (§3c) |
| `GEOTECH_FOUNDRY_HOST` | only to override the route on purpose, e.g. the stack host `yourstack.palantirfoundry.com` an admin names | Where the proxy lives (§3c) |
| `GEOTECH_WEBAPP_DATA` | a durable writable folder, e.g. a path under the workspace files | Saved conversations survive restarts |
| `GEOTECH_REFERENCES_DOCS` | folder of reference PDFs (optional; §3d) | Enables figure read-off and worked-example pages, same as Databricks |

Any model id starting with `ri.` routes through the Foundry LLM proxy
automatically, by **model family**: a Claude model uses the
Anthropic-messages proxy, everything else the OpenAI-chat-completions proxy.
The family is the one the id states, else read off the id:

- **Stated** (since the INF-25 test wheel, 2026-10-20): prefix the id with
  `anthropic:` or `openai:` — `Opus 5.5=anthropic:ri.…` in
  `GEOTECH_FOUNDRY_MODELS`, or pick **Model family: Anthropic (Claude)**
  beside the "Model RID or API name" box in the sidebar. The proxy is sent
  the id without the prefix. Use this whenever a Claude RID does not say
  what it is.
- **Read off the id** (Model family: Auto): an id containing `anthropic`
  anywhere, or naming `claude`, `opus`, `sonnet`, `haiku` or `fable` as a
  word of the model's name (a RID's last field; `claude-opus-5-5`,
  `Claude_Opus_5_5`, `opus4`), is a Claude model; anything else is
  OpenAI-shaped, so a random chunk such as `x7opus3k` stays OpenAI. (Before
  the INF-25 wheel only `anthropic` counted; the INF-25 test wheel of
  2026-10-20 matched the words anywhere in the id.)

The sidebar's success line names the route and how the family was decided,
e.g. "Using the Foundry Anthropic proxy (ri.…; family chosen; route:
FOUNDRY_PROXY_URL)", and so does **Connection diagnostics**.

### 3a. Claude on Foundry (Opus 5.5 / Sonnet 5.5)

```python
import os
os.environ["GEOTECH_FOUNDRY_MODELS"] = (
    "Opus 5.5=anthropic:<Opus 5.5 RID>,Sonnet 5.5=anthropic:<Sonnet 5.5 RID>")
from webapp.foundry_entry import main
main()
```

The first entry is the default; the sidebar picker switches between them.
Or leave the env line out and type the RID in the sidebar with Model family
**Anthropic (Claude)**, then **Use this model**. Claude goes through the
Anthropic-messages proxy (§3c says which address) with `FOUNDRY_TOKEN` as
the bearer; a Claude model **API name** typed instead of a RID goes there
too, because the `palantir_models` engine below is GPT-only. The RIDs must
be imported in the workspace's Models panel, or the proxy answers 403
`UseModelPermissionDenied`. The OpenAI-shaped door does not serve Claude
RIDs (404 `LanguageModelNotAvailable`), so a Claude RID must be family
Anthropic. If the proxy refuses (401/403), see the troubleshooting entry
below.

Page images go the same way: the vision tools send each image as a base64
data URI, which `langchain-anthropic` turns into an Anthropic `image` block
(`source.type: base64`), with the bare RID and a `max_tokens` on every
request. Images worked on Foundry on 2026-10-20 (Opus 5.5 and Sonnet 5.5).
The vision tools size Claude's images by the model's name (`claude-hires`
for the 5.5 models; Connection diagnostics show it), not by the token-count
probe the GPT models need, which labelled Opus 5.5 `gpt-5.2-high` in the
first run.

**What the proxy refuses, and what the app sends instead** (measured with
both Claude 5.5 RIDs on 2026-10-20; each refusal is a 400 whose body names
the field, e.g. `unrecognizedProperty=cache_control`). Every request passes
`engine_config.sanitize_foundry_anthropic_payload` on its final payload, so
this holds for plain, streamed and tool calls and whatever middleware asked:

| Field | Proxy | The app |
|---|---|---|
| `cache_control` at the top of the request (deepagents' caching middleware asks for it) | refused | moved onto the last message block, or the last system block |
| `cache_control` on a tool definition (the same middleware marks the last tool) | refused | removed |
| `cache_control` on system and message content blocks | accepted; a repeated prompt reads from the cache | kept |
| `service_tier` | refused | removed |
| `thinking: {"type": "enabled", ...}` | refused ("use adaptive") | sent as `{"type": "adaptive"}` |
| `tool_choice` `any` or a named tool | refused for these models | stopped before sending, with an error that says so; `auto` and none are unchanged |
| `anthropic-beta` headers, `metadata.user_id` | accepted | unchanged |

A new refusal names its field in the 400 body: add it to that function. If
chat questions still fail with a 400 that names `cache_control` while the
diagnostics pass, add `os.environ["GEOTECH_FOUNDRY_PROMPT_CACHE"] = "0"` to
the app file: no cache marks are sent at all. In the first run's checks a
repeated prompt did not always read from the cache; a miss costs tokens,
not answers. Checked
offline against a stand-in proxy that refuses those fields
(`webapp/tests/test_foundry_model_family.py`).

### 3c. Which proxy address the app uses

The first that is set wins:

1. `GEOTECH_FOUNDRY_HOST` + `/api/v2/llm/proxy/anthropic` (or
   `/openai/v1`): the app's own setting, for a route an admin names on
   purpose. Leave it unset otherwise.
2. `ANTHROPIC_BASE_URL` (Claude) or `OPENAI_BASE_URL` (GPT): some Foundry
   containers are given these ready-made.
3. `https://` + `FOUNDRY_PROXY_URL` + `/language-model-service/api/proxy/anthropic`
   (or `/openai/v1`): every Code Workspace and published app seen so far
   carries `FOUNDRY_PROXY_URL`, so this is the usual route.
4. `FOUNDRY_HOSTNAME` (or `FOUNDRY_URL`) + `/api/v2/llm/proxy/...`: the route
   before the release after 5.33.0. In a Code Workspace and a published app
   `FOUNDRY_HOSTNAME` is a local data sidecar (`localhost:8080`) that answers
   every LLM-proxy path with 401 and an empty body, so this works only where
   it is the real stack host.

The sidebar's success line names the route (`route: FOUNDRY_PROXY_URL`,
`route: the base URL the container provides`, ...), and **Connection
diagnostics** print the setting and the URL (never the token). TLS is checked
against the first of `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE` and
`MAESTRO_CERT_PEM_PATH` that names a file (the line then says "CA bundle
set"), else the SDK's own default.

### 3d. Reference PDFs on Foundry

There is no SharePoint on Foundry, so the reference PDFs (chart read-offs,
worked-example source pages) sit in a folder committed beside the app file,
under the same file names as the SharePoint `primary_references` folder
(Connection diagnostics list the names it looks for), and the app file points
at it:

```python
import os
os.environ.setdefault("GEOTECH_REFERENCES_DOCS", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "references"))
```

Where the PDFs should live for good (a media set or a dataset, with a fetcher
like the SharePoint one) is BACKLOG INF-39 (e) and OWN-80.

### 3b. Alternative engine: the in-platform `palantir_models` SDK (no proxy, no token)

On a Foundry deployment, a model id that is **not** an `ri....` RID is treated
as a **Palantir model API name** (e.g. `GPT_5_1`) and served through the
in-platform `palantir_models` SDK instead of the LLM proxy. This engine speaks
to **GPT models only** (`OpenAiGptChat…` classes); a Claude API name (family
Anthropic, stated or read off the name) goes to the Anthropic proxy of §3a
instead. The workspace
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
Images (since the release after 5.31.0): a call that carries an image goes
through `OpenAiGptChatWithVisionLanguageModel` with
`GptChatWithVisionCompletionRequest` / `MultiContentChatMessage` /
`Base64ImageContent(image_url=..., detail=ImageDetail.X)` — the shape in the
Model Catalog's own "Vision" example. Before that, image inputs were
flattened to text on this route and the vision tools were blind. Image
`detail`: HIGH caps an image at about 768 px on Foundry; full size is AUTO, so
the app's "original" goes as AUTO.

**Route** (5.32): by default (`GEOTECH_FOUNDRY_ROUTE=auto`) every call goes as
a **Responses** request (`CompletionRequestV3.open_ai_responses`) where the
SDK can send one. On Foundry, GPT-5.4 is served by Bedrock and answers only
that route (the chat doors return 404 LanguageModelNotAvailable), and it takes
GPT-5.6 Sol's images to at least 4096 px where the chat door stops near
2048 px. `GEOTECH_FOUNDRY_ROUTE=chat` forces the chat doors above. Connection
drops, timeouts, rate limits and 503s are retried (up to 5 minutes of waiting
a call); anything else is raised at once. Folded in from the AI FDE's glue
that ran the 5.32 test runs.

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
Claude RID; `$FOUNDRY_TOKEN` and `$FOUNDRY_PROXY_URL` are usually already
set). This is the route of §3c step 3:

```bash
curl -s "https://$FOUNDRY_PROXY_URL/language-model-service/api/proxy/anthropic/v1/messages" \
  -H "Authorization: Bearer $FOUNDRY_TOKEN" -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"<Claude RID>","max_tokens":20,"messages":[{"role":"user","content":"Say ok."}]}'
```

(If curl reports a certificate problem, add `--cacert "$REQUESTS_CA_BUNDLE"`
or the file `MAESTRO_CERT_PEM_PATH` names.) The stack-host route, where
`FOUNDRY_HOSTNAME` is the real stack host:

```bash
curl -s "https://$FOUNDRY_HOSTNAME/api/v2/llm/proxy/openai/v1/chat/completions" \
  -H "Authorization: Bearer $FOUNDRY_TOKEN" -H "Content-Type: application/json" \
  -d '{"model":"ri.language-model-service..language-model.gpt-5-2","messages":[{"role":"user","content":"Say ok."}]}'
```

For a Claude RID, the Anthropic shape:

```bash
curl -s "https://$FOUNDRY_HOSTNAME/api/v2/llm/proxy/anthropic/v1/messages" \
  -H "Authorization: Bearer $FOUNDRY_TOKEN" -H "Content-Type: application/json" \
  -H "anthropic-version: 2023-06-01" \
  -d '{"model":"<Claude RID>","max_tokens":20,"messages":[{"role":"user","content":"Say ok."}]}'
```

A JSON reply containing the model's answer = the proxy is enabled and the RID
is right. An error page or 404 = the LLM-provider-compatible API isn't enabled
on the enrollment — ask the platform admin. (If `FOUNDRY_HOSTNAME` already
carries `http://` or `https://`, drop the `https://` in the URL.)

## 4a. Downloads and the PDF preview in the published app

In the published app Streamlit's own download button saves an HTML error
page instead of the file (Chrome: `<id>.htm`, "File wasn't available on
site"), and a PDF shown inline as a `data:` frame is blocked ("This content
is blocked"). Since the release after 5.33.0, on Foundry
(`GEOTECH_DEPLOYMENT=foundry`, which the entry sets) every download is two
buttons the page makes in the browser: the download itself, saved under the
file's own name, and **Open in new tab**, the browser's own viewer (a file
saved from there gets a random name; use the first button to keep it). The
file travels inside the page, so files over 20 MB show a message instead;
`GEOTECH_FOUNDRY_DOWNLOAD_MAX_MB` changes the limit. A PDF card's preview
shows the first ten pages as images, with a note to open the whole file in a
new tab. Elsewhere both are unchanged (`webapp/file_widgets.py`).

## 4b. Retiring the workspace glue

The first run (INF-25) needed two files the AI FDE wrote in the workspace:
`foundry_proxy_shim.py` (a local forwarder that fixed the proxy address and
dropped the refused fields) and `foundry_ui_shim.py` (the download buttons,
and a PDF preview attempt that stayed blocked). The release after 5.33.0
does both in the package (§3a, §3c, §4a). Once it is installed, delete the
two files and their `install()` calls from the app file, republish, and run
Connection diagnostics: the engine line should say `route:
FOUNDRY_PROXY_URL` (or `route: the base URL the container provides`). Keep
the wheel installation and the model list in the app file until the
packages come from an Artifacts repository (§1). Nothing breaks if the new
wheel arrives before the two files are deleted: the forwarder sets
`GEOTECH_FOUNDRY_HOST`, which the app then follows (§3c step 1), and the
package's own download buttons and preview no longer call what the UI file
replaces.

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
configured the sidebar shows a single **"Model RID or API name"** input with a
**Model family** choice beside it (Auto / Anthropic (Claude) / OpenAI (GPT);
§3) — no "advanced" expander, no built-in Claude model list, no key
references in banners or the diagnostics report. Local/dev behaviour is
unchanged.

## Troubleshooting a live deployment (learned 2026-07-17/18, a government enclave)

Field notes from the first real publish (a sandbox project on `<your-enclave>.palantirgov.com`):

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
  `FOUNDRY_HOSTNAME` points at a local sidecar (`localhost:8080`) that
  answers every LLM-proxy path with 401 and an empty body, and direct egress
  to the stack domain is DNS-blocked. Since the release after 5.33.0 the
  app goes through `FOUNDRY_PROXY_URL` instead (§3c); check the engine line
  in Connection diagnostics says so, and that `GEOTECH_FOUNDRY_HOST` is not
  set to something older. If the proxy route answers 401/403 too, it is an
  enrollment or permission question: is the LLM provider-compatible API
  enabled; is this account authorized for the models, and are the RIDs
  imported in the workspace's Models panel (403 `UseModelPermissionDenied`
  otherwise); what base URL should containers use. An admin's answer goes
  into `GEOTECH_FOUNDRY_HOST` in the app file (explicit
  `http://`/`https://` schemes are honoured). `palantir_models` has a Claude
  class too (`AnthropicClaudeLanguageModel`, by model API name; worked for
  both 5.5 models on 2026-10-20), so the SDK engine can gain a Claude leg if
  the proxy is ever closed.
- **The upload line in Connection diagnostics** on Foundry reads `PUT -> 403;
  POST -> 405`: expected, because the probe has no browser session.
  Uploading from the browser worked in the published app (2026-10-20).
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
  unchanged. From the release after 5.33.0 each model request on the proxy
  route gives up after 600 s (before, langchain set no limit), and the
  agent's own busy ladder retries a timeout, a 429 or a 5xx on every route
  (the provider's retry-after, else 15/30/60/120 s, 5 tries); the SDK
  route (§3b) keeps its own retries as well. Vision (attachment analysis, figure read-off) depends on the
  enrollment's model supporting images through the proxy — try it, and prefer
  a multimodal RID for the default model.
- Conversations persist under `GEOTECH_WEBAPP_DATA` (default
  `~/.geotech_webapp`). In a published app that is inside the app's own
  container: not visible from the workspace, and lost when the container is
  rebuilt (BACKLOG INF-38).
- The old `foundry/` wrapper files (tools for the prepackaged AIP agent) are
  RETIRED and unrelated to this deployment.
