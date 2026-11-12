# Running the geotech web chat app on Palantir Foundry (Code Workspaces)

Design + rationale: `module_work/FOUNDRY_APP_PLAN.md`. This is the do-it page.
Everything lives in the pip package; the Foundry side is one workspace, a
2-line file, and a few environment variables. Document Review on its own,
from its own package `aec-document-review`, is published the same way:
section 6.

## 1. Create the workspace and install the package

1. In Foundry: **Code Workspaces → New workspace → JupyterLab**.
2. In the workspace **Libraries** panel, install (PyPI):
   - `geotech-staff-engineer`  — as of the release after 5.8.0, this single
     package brings everything: all analysis backends, the deep agent,
     Streamlit, PDF reporting, and both LLM clients (`langchain-openai` for
     GPT-family RIDs, `langchain-anthropic` for Claude RIDs). From 5.35.0 it
     also brings the shared core `aec-agent-core`, which the app pins
     exactly (`aec-agent-core==1.0.0` on 5.35.0), and it needs Python 3.11
     or newer (Foundry runs 3.12).
   - (On 5.8.0 or older, use the old form: `geotech-staff-engineer[deep,full,pdf]`
     plus `streamlit` plus `langchain-openai`.)
3. **Test wheels** (built from master, not on PyPI; e.g. the INF-25 set in
   `../foundry_handoff/gse_app_2026-10-20/`). Their versions carry a local
   label (`5.33.0+foundry.20261020`), which the sidebar footer shows. What
   worked in the first run (the AI FDE, 2026-10-20), in order:
   1. **The other packages first.** Everything the app needs except its own
      four packages goes into the managed environment as usual, plus three
      pins: `websockets>=14,<16`, `polars>=1,<2` and `tenacity<9.2`
      (Streamlit's and Foundry's own libraries otherwise pull in newer
      versions than langgraph-sdk, pygef and google-genai accept, and
      `pip check` reports the conflict, e.g. `websockets 17.3`,
      `polars 2.0.0`). Add the pins on the conda side beside the other
      dependencies. Foundry resolved numpy 2.5.4 on Python 3.12.15 in the
      Code Workspace; a transforms repo holds numpy at 1.26.4 (next item).
   2. **Upload the four wheels**: the app, its core `aec-agent-core`,
      planlens and geotech-references. A set for 5.35.0 or later (or built
      from master since the app split's step 11 (a)) needs the core's wheel: the
      app imports its model connections and vision tools from it, and pins
      it exactly (`aec-agent-core==<version>`; with `--no-deps` pip does
      not say it is missing, the app fails at import). Sets for 5.34.0 and
      earlier have three. Check their names. Unpacking a zip
      in the workspace dropped the `+` from the names
      (`geotech_staff_engineer-5.33.0foundry.20261020-py3-none-any.whl`),
      and pip refuses a wheel whose name does not match its version. Put the
      `+` back (`5.33.0+foundry.20261020`); the version inside the wheel
      (its METADATA) is the one to copy.
   3. **Install them from the app file at start-up.** maestro does not take
      file paths (`maestro env pip install ./x.whl` answers "No valid
      versions found"), so the app file runs
      `pip install --no-deps --no-index --target <folder>` on the four
      wheels, into a folder outside the managed environment, and puts that
      folder first on `sys.path` and `PYTHONPATH`. The lockfile then never
      lists the local builds, and the published app installs them the same
      way when it starts.

   The clean route is a Foundry Artifacts repository (or conda channel)
   holding the four packages, so the environment lists them like any other
   dependency and step 3 goes away.
4. **Transforms repos: numpy 1.26** (BACKLOG INF-53; where the evals run).
   The platform's `transforms` pins a pyspark that needs numpy < 2, so the
   environment keeps numpy 1.26.4 from the conda layer (Python 3.12), and
   pip does not move it. The app runs on numpy 1.26, but the newest scipy,
   pandas, contourpy, OpenCV, SALib, liquepy and pygef need numpy 2: left
   unpinned, pip put scipy 1.18.1 there, which cannot import its
   `optimize`, `interpolate`, `integrate` or `stats` on numpy 1.26
   (`np.long`), and 15 questions of the geotech eval failed on it. So:
   1. install the four wheels with `--no-deps`, as above;
   2. `python constraints/foundry_requirements.py > requirements.txt`
      prints what they need, numpy and PyNiteFEA left out;
   3. install that list with the constraints:
      `pip install -r requirements.txt -c constraints/foundry-numpy126.txt`
      (the constraints file caps each of the seven at the newest release
      that works on numpy 1.26, pandas below 3 for SALib 1.5.1, and pins
      numpy at 1.26.4). conda cannot read a pip constraints file: on the
      conda side, add the same caps as conda pins. conda-forge spells them
      the same in lower case (SALib is `salib`); liquepy and pygef are not
      on conda-forge, so cap them wherever they come from.
      pip then reports that PyNiteFEA is not installed,
      which is expected. (planlens 0.13 also drew a warning, for declaring
      numpy >= 2.0; from 0.14, and so on 5.34.0 and later, planlens
      declares numpy >= 1.26 and the warning is gone.)
   4. to keep the `pynite` tool (frame and continuous-beam analysis),
      install PyNiteFEA without its dependencies, then its one pure-Python
      requirement under the constraints:
      `pip install --no-deps PyNiteFEA==3.2.0`, then
      `pip install prettytable -c constraints/foundry-numpy126.txt`
      (PyNiteFEA declares numpy >= 2.4 but runs on 1.26.4, and its suites
      pass there). Check that the tool is listed:
      `python -c "from funhouse_agent.adapters import MODULE_REGISTRY as R; assert 'pynite' in R"`
      exits 0 (it exits 1 and the registry's `ABSENT_MODULES` names the
      missing package when the step was left out).

   PyNiteFEA declares numpy >= 2.4 from release 2.1, so no release the app
   accepts installs there; the app then leaves its `pynite` tool (frame and
   continuous-beam analysis) out of the agents' tool list, and everything
   else runs. (PyNiteFEA 3.2.0 does run on numpy 1.26.4: installed with
   `pip install --no-deps PyNiteFEA==3.2.0`, then
   `pip install prettytable -c constraints/foundry-numpy126.txt`, the tool
   is back and its tests pass.) CI checks this install on pushes to master
   that touch code, and on pull requests (the `numpy 1.26` job in
   `.github/workflows/tests.yml`).

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
| `GEOTECH_WEBAPP_DATA` | a writable folder, e.g. a path under the workspace files | Where saved conversations are kept. In a published app that folder is inside the app's own container, so they are lost when it is rebuilt (Notes below; BACKLOG INF-38) |
| `GEOTECH_REFERENCES_DOCS` | folder of reference PDFs (optional; §3d) | Enables figure read-off and worked-example pages, same as Databricks |
| `GEOTECH_HELPER_MODELS` | optional, e.g. `references=Sonnet 5.5,calc=main` | A helper's own model (§3a); on Foundry the references helper defaults to `Sonnet 5.5` |

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

On Foundry the main agent defaults to the entry labelled `Opus 5.5`, else
the list's first entry (`GEOTECH_WEBAPP_MODEL` overrides both); the sidebar
picker switches between them.

**The references helper runs on Sonnet 5.5** (the owner, 2026-10-20; BACKLOG
INF-41). On Foundry the helper that searches and reads the reference library
defaults to the model list's `Sonnet 5.5` entry; the main agent and every
other helper (calculations, figures, the reviewer, page reading) run on the
main model, Opus 5.5 by default. Pictures are read on the main model too:
when the references helper reads a value off a chart
(`read_reference_figure`), the image goes to the main model (Opus 5.5 by
default), and Sonnet 5.5 does the helper's searching, reading and writing.
The sidebar shows
a **References helper model** picker under **Model** ("Same as the main
model" keeps it on the main model), and `GEOTECH_HELPER_MODELS` sets any
helper by name: `references=Sonnet 5.5`, `calc=main`, or a RID
(`references=anthropic:<RID>`). A model is named by its label in
`GEOTECH_FOUNDRY_MODELS` or by its id. The helper's model is built exactly
as the main one, through the same proxy address and request fixes (§3a,
§3c). If `Sonnet 5.5` is not in the list, the helper stays on the main model
and the sidebar says so. **Connection diagnostics** print an `agent models`
line (the main agent, each helper with a model of its own, "every other
helper: the main agent's model" and the model that reads images, each with
where it came from: this host's
default, `GEOTECH_HELPER_MODELS`, `GEOTECH_WEBAPP_MODEL` or the sidebar) and send a plain and a tool-calling request
to each helper model; each turn's `activity.jsonl` names the model on every
`model_start` beside the agent that made the call, and `meta.json`'s engine
record lists `helper_models`. Other hosts have no helper defaults.
Or leave the env line out and type the RID in the sidebar with Model family
**Anthropic (Claude)**, then **Use this model**. Claude goes through the
Anthropic-messages proxy (§3c says which address) with `FOUNDRY_TOKEN` as
the bearer; a Claude model **API name** typed instead of a RID goes there
too, unless `GEOTECH_FOUNDRY_CLAUDE_SDK` is on (§3b, "Claude through the
SDK"). The RIDs must
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
`engine_config.sanitize_foundry_anthropic_payload`
(`packages/aec-agent-core/aec_agent_core/engine_config.py`, the shared
core) on its final payload, so
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

### 3b. Alternative engine: the in-platform `palantir_models` SDK (no proxy, no token)

On a Foundry deployment, a model id that is **not** an `ri....` RID is treated
as a **Palantir model API name** (e.g. `GPT_5_1`) and served through the
in-platform `palantir_models` SDK instead of the LLM proxy. For GPT models
it uses the `OpenAiGptChat…` classes; a Claude API name (family Anthropic,
stated or read off the name) goes to the Anthropic proxy of §3a, unless
`GEOTECH_FOUNDRY_CLAUDE_SDK` is on (below). The workspace
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
carries OpenAI-style `tools`); wrapper: `packages/aec-agent-core/aec_agent_core/palantir_sdk_engine.py`.
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

**Claude through the SDK** (BACKLOG INF-27; **off by default**, the proxy of
§3a stays Claude's route until it has been tested on Foundry). Turn it on in
the app file and name the Claude models by their **API names**, not RIDs:

```python
import os
os.environ["GEOTECH_FOUNDRY_CLAUDE_SDK"] = "1"
os.environ["GEOTECH_FOUNDRY_MODELS"] = (
    "Opus 5.5=AnthropicClaude_5_5_Opus,Sonnet 5.5=AnthropicClaude_5_5_Sonnet")
from webapp.foundry_entry import main
main()
```

Each call then goes through `AnthropicClaudeLanguageModel.get(<API name>)`
and `create_chat_completion(ClaudeChatCompletionRequest, max_rate_limit_retries=4)`
(wrapper: `PalantirClaudeChatModel` in `packages/aec-agent-core/aec_agent_core/palantir_sdk_engine.py`): no
token, no proxy address. Text and tool calls come back as the agent expects,
tool results go back as `tool_result` blocks, page images go as base64 PNG or
JPEG. No request asks for thinking; if a reply carries thinking blocks anyway,
they go back with the rest of that tool loop, as Anthropic's API requires.
Two blocks are marked for caching, as the proxy route's caching middleware
marks them: the system prompt (one block, which caches the tools with it) and
the last message's last block (the conversation so far). Where this SDK's
classes take no cache field, that block goes unmarked; the tools and system
prompt are then cached alone, or nothing is. `GEOTECH_FOUNDRY_PROMPT_CACHE=0`
sends no marks. A 400 whose body names a cache field (`cache_control`,
`system_blocks`) is asked once more with no marks, and if that works no marks
are sent for that model until the app restarts; any other error, a 5xx
included, is not resent for this;
`tool_choice` is never sent; connection drops, timeouts, rate limits and 503s
are retried as on the GPT leg. The labels keep the main agent on Opus 5.5 and
the references helper on Sonnet 5.5 (§3a). A model RID still uses the proxy,
and the engine message in the sidebar says so. Connection diagnostics show
`GEOTECH_FOUNDRY_CLAUDE_SDK=1` on the environment line, and the engine
resolution line reads `source=foundry_sdk` and "palantir_models SDK for
Claude". If a model is not imported in the workspace's **Models** panel,
Foundry answers 403 `UseModelPermissionDenied` and the app says, in plain
words, to import it there, restart the workspace and publish again. Turn the switch off (or
delete the line) to go back to the proxy.

### 3c. Which proxy address the app uses

The first that is set wins:

1. `GEOTECH_FOUNDRY_HOST` + `/api/v2/llm/proxy/anthropic` (or
   `/openai/v1`): the app's own setting, for a route an admin names on
   purpose. Leave it unset otherwise.
2. `ANTHROPIC_BASE_URL` (Claude) or `OPENAI_BASE_URL` (GPT): some Foundry
   containers are given these ready-made.
3. `https://` + `FOUNDRY_PROXY_URL` + `/language-model-service/api/proxy/anthropic`
   (or `/openai/v1`): every Code Workspace and published app seen so far
   carries `FOUNDRY_PROXY_URL`, so this is the usual route
   (`MAESTRO_FOUNDRY_PROXY_URL` is read when `FOUNDRY_PROXY_URL` is not
   set, as the first run's workspace glue did).
4. `FOUNDRY_HOSTNAME` (or `FOUNDRY_URL`) + `/api/v2/llm/proxy/...`: the route
   before the release after 5.33.0. In a Code Workspace and a published app
   `FOUNDRY_HOSTNAME` is a local data sidecar (`localhost:8080`) that answers
   every LLM-proxy path with 401 and an empty body, so this works only where
   it is the real stack host.

The sidebar's success line names the route (`route: FOUNDRY_PROXY_URL`,
`route: the base URL the container provides`, ...), and **Connection
diagnostics** print the setting and the URL (never the token). TLS is checked
against the first of `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`,
`MAESTRO_CERT_PEM_PATH` and `FOUNDRY_CLI_CERT_PEM_PATH` that names a file
(the line then says "CA bundle set"), else the SDK's own default, certifi's
public CA list. Connection diagnostics' **TLS certificates** line says which
(by variable, never the path), and warns when a variable names a file that
does not exist.

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

**A deployment's own references (local reference packs).** A deployment can
add its own manuals beside the built-in library, so the references helper
searches them, cites them and reads their figures like the built-in ones.
Each manual is a *pack*: a folder named by its id (which starts with
`local_`) holding `pack.json`, `text/chapterNN.json` in the library's chapter
format and, optionally, `figures_catalog.json` and the PDF. The format, the
rules (bodies in the print's own wording with page markers) and the check
(`python -m geotech_references.packs check <pack folder>`, exit 0 or 1) are
in the geotech-references repo's
[`docs/REFERENCE_PACKS.md`](https://github.com/soconnell345-geotech/geotech-references/blob/main/docs/REFERENCE_PACKS.md). Packs
need geotech-references 1.6.0 or later. On Foundry, commit the packs in a
folder beside the reference PDFs and point the setting at it in the app file:

```python
os.environ.setdefault("GEOTECH_REFERENCES_PACKS", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "references", "packs"))
```

A pack's PDF may sit in its own folder or in the `references/` folder above
(by the file name its catalog gives). A deployment's packs are its own: never
commit one to the public repos (a test fails on a tracked `pack.json`). With
packs set, the references helper is told which manuals the deployment adds,
the turn record's `turn_start` lists them (`reference_packs`: id, title,
files, a sha256 of their contents) and the Connection diagnostics' reference
PDF row lists their PDFs and names any pack the library skipped.

**A deployment's own instructions.** A deployment can add its own
conventions to every agent and helper: preferred methods (cited from its own
reference packs), units, formats, naming. `GEOTECH_INSTRUCTIONS` names a
folder holding `all.md` (appended to every agent and helper) and one
`<role>.md` per role (`main`, `references`, `reviewer`, `calc`, `figure`,
`general-purpose`, `model_setup`, `report_library`, `report_ingest`). The
built-in rules win: the app puts its protected integrity rules and a fixed
heading above the deployment's text, and the guards in code are never
touched by text. At most 4,000 characters per role, cut at a paragraph past
that; an unreadable folder adds nothing and the record says why. On Foundry,
commit an `instructions/` folder beside the app file and point the setting at
it there:

```python
os.environ.setdefault("GEOTECH_INSTRUCTIONS", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "instructions"))
```

The turn record's `turn_start` lists what each role got (`engine.build.
appended`: the files, the characters, a sha256 and the first 2,000
characters), and the Connection diagnostics' "deployment instructions" line
shows what each role gets, or "none". An example folder with made-up content is in
`docs/examples/deployment_instructions/`. A deployment's instructions are its
own: keep them in the deployment's repository, not the public ones.

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
`GEOTECH_FOUNDRY_DOWNLOAD_MAX_MB` changes the limit. The page is sent again
on every action, so a file over 2 MB first shows **Prepare download**: the
file goes into the page only once pressed, and until the next turn
(`GEOTECH_FOUNDRY_DOWNLOAD_PREPARE_MB` changes the 2 MB). An HTML or SVG file
opened in a new tab is shown inside a sandboxed frame, so a page the agent
wrote runs with an origin of its own, not the app's; the HTML preview on a
card is framed the same way on every host. A PDF card has a
**Preview the first 10 pages** switch: turned on, it shows those pages as
images, with a note to open the whole file in a new tab (the pages are drawn
only when asked for, so a long conversation with many PDFs stays quick).
Elsewhere both are unchanged (`webapp/file_widgets.py`).

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

## 4c. Conversations that survive a rebuild

A published app keeps its conversations inside its own container, so they
are lost when the container is rebuilt, and every push to the workspace's
master branch republishes the apps (a restart). Since the release after
5.33.0 the app saves each conversation to a Foundry dataset after every
turn, the way it saves to SharePoint on Databricks, and brings past ones
back from the sidebar (BACKLOG INF-38). With it in place a restart is
harmless: the conversation list starts empty, and **Find a past
conversation** brings any one back.

The app reaches the dataset through the workspace dataset library
(`foundry.transforms`) that comes with the Code Workspace, so there is
nothing to install. The platform's datasets API (and with it
`foundry-platform-sdk`) is turned off inside app containers.

1. **Create a dataset of raw files** (no schema), e.g. `app_conversations`,
   used for nothing else, in a project only the right people can see
   (below, "One user"). Leave it empty: it needs **no placeholder file**.
   A new dataset with nothing committed yet reads as an empty store, and
   the app's first save makes its first commit. (Before INF-65 it answered
   `FailedToResolveDataset`, "does not have any committed transaction", and
   a placeholder upload also made Foundry guess a table schema for it.)
2. **Import it into the workspace**: in the workspace's **Data** panel,
   import the dataset with **read and write** access. That names it in
   `.foundry/aliases.yml`; the name there is its alias. Commit and push.
3. **Set the alias in the app file**, before `main()`:
   ```python
   os.environ["GEOTECH_FOUNDRY_CONVERSATION_DATASET"] = "app_conversations"  # the alias, not the ri. ID
   ```
   A resource ID (`ri.…`) does not work here; Connection diagnostics and
   the sidebar say so.
4. **Push** (which republishes), then open **Connection diagnostics**: the
   line *conversation storage* should PASS and name the alias (on a new
   dataset, "readable, 0 files: a new dataset with nothing committed
   yet"). When it fails it says why: a resource ID instead of the alias,
   an alias the workspace does not define (`AliasNotDefined`: import the
   dataset and push), the library missing (the app was not published from
   a Code Workspace), or the dataset refusing the read (the error as
   Foundry gave it). Until it works, the sidebar's **Permanent storage**
   block says **Conversations are NOT kept across restarts**, with the
   same reason and fix. The sidebar and the diagnostics line share one
   record of the dataset's latest read, so they agree: on opening, the
   sidebar says "Foundry storage: checking the conversation dataset…"
   while it reads the file list once in the background (about 3-4 s), and
   says storage is on only once that read has worked.

**After the first turn**, the sidebar shows **Permanent storage** with
"Foundry storage: saving the latest turn…", then, without a click, "Foundry
storage: saved N file(s) just now, …" (the sidebar looks again every 2 s
while a save runs, and stops when it ends), and the dataset holds
`GeotechStaffEngineer/conversations/<title>_<date>/` with `meta.json`,
the transcript, the activity log `activity.jsonl`, the turn traces
`trace.jsonl`, any feedback (`FEEDBACK.md`, `feedback.jsonl`) and `files/`
(the Document Review page's under `conversations/document_review/`). The
dataset is where those records can be read from outside the app's
container. To check a restore, open
**Find a past conversation (permanent storage)**, press **Show past
conversations** (or type part of its title), and press **Restore** (or
open the app after a push: the list is read from the dataset).

How it saves: once the turn's record is complete, the files the turn
changed are copied under the outputs folder (`/foundry/outputs`), at their
paths in the dataset, and uploaded in **one** transaction. This runs in
the background, so the chat does not wait for it (a save takes about 5-7
s); saves run one at a time, in order. A dataset takes one open
transaction at a time, so a save that meets another one open waits and
tries again for about 30 s; if it still cannot, the sidebar says another
save holds the dataset, the turn is unaffected, and the files go up with
the next save. The app uploads the outputs folder only while it holds
nothing but the app's own `GeotechStaffEngineer/` folder; if anything else
is there, the app saves file by file (one transaction each, slower) and
says so once, so **keep `/foundry/outputs` for the app**. A dataset has no
link per folder, so the sidebar shows the folder name but no "Open session
folder" link. Reading the dataset's file list takes about 3-4 s, so the
page never waits for it on opening: **Find a past conversation** reads it
when asked (**Show past conversations**, a search or **Refresh**), then
keeps it for the browser session (after **Sync now** it is read again when
next shown); a restore reads it once and then downloads up to six files at
a time.

**One user.** On Foundry the store is built for a single user: the app's
owner. Every conversation goes in one folder, `conversations/`, and
**Find a past conversation** lists all of them. The app writes as the
publisher's account (as far as the AI FDE's test could tell), and whoever
can open the dataset can read every conversation in it, so keep it in a
project only the right people can see. Folders per viewer wait on the app
knowing who is viewing, which a Foundry app does not tell it today. (Where
a host does pass the viewer, as Tiny Apps does, each person's
conversations already go in a folder of their own.)

## 5. Preview, then publish

1. In JupyterLab, use the Streamlit **preview** on `app.py` to try it in the
   workspace.
2. **Publish application** → choose a Files-and-Projects location → file name
   `app.py`. The app gets a Foundry URL behind Foundry auth, shareable like
   any resource.
3. To upgrade later: bump `geotech-staff-engineer` in the Libraries panel and
   republish. The 2-line `app.py` never changes.

## 6. Document Review on Foundry (a second host)

Document Review, the document review page, can be published from a Foundry
Code Workspace on its own, from its own package `aec-document-review`, as a
second host in case Tiny Apps is withdrawn (BACKLOG INF-74). It is the same
Streamlit page the testers use on Tiny Apps, with the geotech app's Foundry
behaviour: the models are Foundry model RIDs, and the conversations can be
saved to a Foundry dataset. It needs the first `aec-document-review`
release after 0.1.1, which has the Foundry entry, with the core release it
pins; 0.1.1 and earlier cannot save to a Foundry dataset. Until that
release, use test wheels as in section 1, step 3, with three wheels
instead of four: `aec-document-review`, `aec-agent-core` and `planlens`
(nothing geotechnical).

1. **Install the package.** In the workspace's **Libraries** panel, install
   `aec-document-review` from PyPI. It brings the shared core
   `aec-agent-core` and `planlens` at the exact versions it was tested
   with, and Streamlit. For Claude models also install `langchain-anthropic`
   (the core's `anthropic` extra); GPT models need nothing more. It needs
   Python 3.11 or newer. If the panel reports a conflict with Foundry's own
   libraries, the pins in section 1, step 3.1 apply here too.
2. **Create the app file.** Create `app.py` at the root of the workspace
   repository, with these two lines:
   ```python
   from aec_document_review.foundry_entry import main
   main()
   ```
   `main()` marks the process as a Foundry deployment
   (`GEOTECH_DEPLOYMENT=foundry`) and runs the page. As with the geotech
   app, the page then offers only Foundry model RIDs and never reads or
   mentions a personal API key.
3. **Set the models.** Set `GEOTECH_FOUNDRY_MODELS` to the model RIDs, as
   in section 3 (`Label=RID`, comma-separated; the first one is the
   default). Choose models that accept images: the page looks at drawing
   pages. The Foundry token and the proxy address are found as for the
   geotech app (section 3); a Code Workspace provides them.
4. **Keep the conversations (recommended).** Follow section 4c, steps 1 to
   4, with a dataset used only for Document Review (for example
   `review_conversations`), and set its alias in the app file before
   `main()`:
   ```python
   import os
   os.environ["GEOTECH_FOUNDRY_CONVERSATION_DATASET"] = "review_conversations"  # the alias, not the ri. ID
   from aec_document_review.foundry_entry import main
   main()
   ```
   After the first turn the dataset holds one folder per conversation,
   `Document Review/conversations/document_review/<title>_<date>/`: the
   top folder is named after the application (`GEOTECH_SHAREPOINT_ROOT`
   names another). Without the alias the conversations live in the app's
   container and are lost when it is rebuilt; the sidebar and Connection
   diagnostics say so. SharePoint (`GRAPH_*` and `SHAREPOINT_SITE_URL`, as
   on Tiny Apps) is used only when the alias is not set, as for the geotech
   app.
5. **Preview, then publish** as in section 5, naming `app.py`. To upgrade
   later, bump `aec-document-review` in the Libraries panel and republish.

Other settings it reads, all optional: `GEOTECH_WEBAPP_DATA` (the folder in
the app's container where conversations are kept while it runs),
`GEOTECH_INSTRUCTIONS` (the deployment's own instructions to the agents,
section 3d), `GEOTECH_MARKUP_AUTHOR` (who the markups it writes on a PDF are
signed by) and the model settings of section 3 (`GEOTECH_HELPER_MODELS`,
`GEOTECH_FOUNDRY_DISABLE_STREAMING` and the others). The Tiny Apps model
settings (`PROMPTER_*`) are not used on Foundry.

**Who is viewing, and what keeps viewers apart: nothing.** A Foundry app is
not told who is viewing it. On Tiny Apps a sign-in header names each tester,
and each one's conversations and files go in a folder of their own. On
Foundry there is no such header, so Document Review runs as it does on a
laptop, for one unnamed user, exactly as the geotech app does on Foundry
today. Nothing keeps one viewer's conversations and files apart from
another's: everyone who can open the published app sees the same list of
conversations, can open, rename or delete any of them and download their
files, and **Find a past conversation** lists and restores every
conversation in the dataset. The **Working folder** box is shown, as for the
geotech app, so a viewer can point the agent's saves at another folder in
the app's container. Dataset writes run as the publisher's account (as far
as the AI FDE's test could tell), and whoever can open the dataset can read
every conversation in it. So share the published app only with people who
may see each other's documents, and keep the dataset in a project only they
can see (section 4c, "One user"). The page refuses no one on Foundry: its
refusal of a caller it cannot name applies only on the Tiny Apps App
Service.

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
  `http://`/`https://` schemes are honoured). Claude can also go through
  `palantir_models` (`AnthropicClaudeLanguageModel`, by model API name;
  worked for both 5.5 models on 2026-10-20), which needs no proxy at all:
  `GEOTECH_FOUNDRY_CLAUDE_SDK=1` with API names (§3b, "Claude through the
  SDK").
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
  agent's own busy ladder retries a timeout, a 429, a 5xx, or a 400 that
  names a rate limit (`errorName` `RateLimitsExceeded` or a QoS exception,
  which is how Foundry's model service sends one) on every route (the
  provider's retry-after, else 15/30/60/120 s, 5 tries); the SDK
  route (§3b) keeps its own retries as well. Vision (attachment analysis, figure read-off) depends on the
  enrollment's model supporting images through the proxy — try it, and prefer
  a multimodal RID for the default model.
- Conversations persist under `GEOTECH_WEBAPP_DATA` (default
  `~/.geotech_webapp`). In a published app that is inside the app's own
  container: not visible from the workspace, and lost when the container is
  rebuilt, unless a conversation dataset is set (§4c, BACKLOG INF-38). To
  keep one without it, use **Download this conversation**
  in the sidebar: **Prepare download** zips its record (`meta.json`,
  `activity.jsonl`, transcript), files and figures, leaving out caches and
  anything that may hold a key; over the 20 MB download limit (§4a) it
  offers the record alone, with the deliverables that fit. **Restore a
  downloaded conversation** in the sidebar takes that zip back (up to 200 MB
  unpacked, `GEOTECH_RESTORE_MAX_MB`): the conversation returns under its own
  id and opens, a record-alone zip says in the conversation what it left
  out, and one already here is replaced only when **Replace the existing
  copy** is ticked (BACKLOG INF-45).
- The old `foundry/` wrapper files (tools for the prepackaged AIP agent) are
  RETIRED and unrelated to this deployment.
