# Tiny Apps: Document Review, plan of record and working notes

**Status (2026-11-10, the app split's step 14, BACKLOG INF-34).** Tiny Apps
hosts **Document Review alone**. The geotechnical page left this host at the
app split (owner, 2026-10-20) and runs on its own elsewhere. Document Review
is its own public package, `aec-document-review` 0.1.1, which pins the
shared core `aec-agent-core` 1.0.1 and the document layer `planlens` 0.15.1
exactly (`packages/aec-document-review/`; built at step 13,
`module_work/inf33/CHECK_STEP13.md`). All three are on PyPI (these versions
since 2026-11-10: they carry the fixes of the deployment review of that
day, `module_work/docreview_review_2026-11-10/REVIEW.md`, and replace the
first releases 0.1.0, 1.0.0 and 0.15.0);
the company's Nexus mirror carries a release a day or two later. The wrapper
template in `tinyapps/wrapper_repo/` installs them from Nexus, or from two
wheels in the repo as the backup. Next: the owner's first round on the test
machine (dosdev), with the checklist below. The end-to-end check of the
wheels route is in `module_work/inf34/CHECK_STEP14.md`.

The platform team that runs Tiny Apps is called **CfA** in these notes.
Their documents (the Tiny Apps user guide, the pilot guidelines and their
example code) are kept only under the gitignored `tinyapps/reference/`;
nothing from them is committed to this public repository. Support: CfA's
Tiny Apps and GitHub support mailboxes (addresses in the user guide).

## What the app is for

A general chatbot is already available to the users. Document Review exists
for what that cannot do: **vision** (looking at drawings, scans, figures and
reviewer markups) and **document creation** (Word memos and marked-up PDFs
handed back). It is one page, at the app's root, for architects,
construction managers and engineers of any discipline: drawings,
specifications, submittals, RFIs, reports, calculation packages. On upload
it orients itself automatically (one cheap turn: what the document is, how
it is organised, what is text and what is picture, existing markups, what it
could do next); then it is a normal chat with the document, vision and file
tools, and a habit of handing back a `.docx` or a marked-up PDF.

Where the code is (all in `packages/aec-document-review/aec_document_review/`):

| File | What it does |
|---|---|
| `tinyapps_entry.py` | `main()`, which the wrapper's `app.py` calls: marks the process as a Tiny Apps deployment, installs the Prompter engine, runs the page |
| `pages_entry.py`, `profiles.py`, `review_page.py`, `app.py` | the one page and its chat shell |
| `tinyapps_settings.py` | the settings: environment, then Key Vault by `KV_NAME`, then the local `.env` |
| `tinyapps_keys.py` | whose Prompter key serves a caller (per-tester keys) |
| `identity.py` | the signed-in user from the identity header, or `DEV_IDENTITY` locally |
| `graph_sharepoint.py`, `sharepoint_store.py` | the SharePoint mirror over an app registration |
| `notice.py` | the page's notice (OWN-78) |

The Prompter engine itself is in the core:
`packages/aec-agent-core/aec_agent_core/tinyapps_engine.py`.

## Environment facts (from CfA's guide and example code; verified where marked)

- **Azure App Service, Linux, behind an IIS/ARR tier** that performs Windows
  authentication.
- **Identity**: IIS passes `X-Windows-Auth-Header: DOMAIN\user`; Streamlit
  reads it via `st.context.headers`. Trusted because the App Service accepts
  traffic only from the IIS servers and the IIS module replaces any client
  value; the app never verifies it and never lets a user type one. Locally
  `DEV_IDENTITY` stands in. On the App Service a request with no usable
  header is refused with a message, and `DEV_IDENTITY` is not read (until
  2026-11-10 such a request was served as one unnamed user).
- **Secrets**: the Key Vault named by the `KV_NAME` app setting, read with
  the managed identity; secret names hyphenated, environment names
  underscored; a real environment variable always wins.
- **Prompter** = an Azure OpenAI-style chat-completions API: `PROMPTER_URL`
  (the full URL ending `/chat/completions`), `PROMPTER_MODEL` (the
  deployment name), `PROMPTER_API_KEY` sent as both `api-key` and
  `Authorization: Bearer`, optional `PROMPTER_CA_BUNDLE` (the internal CA,
  PEM text in Key Vault). On the published app each tester has their own key
  (below). 180 s per call; the proxy times HTTP out near 120 s (turns run
  detached from the request). Strict `json_schema` works.
- **The pilot key serves ONE model, $50/month.** The deployment behind it
  MUST accept image inputs, or the app's reason to exist fails.
- **Packages** from the Nexus mirror only (`PIP_INDEX_URL`); no outbound
  internet from the App Service. The Nexus firewall quarantines packages
  with CVEs, so a build that installed last month can fail today. CfA's
  fleet pins (2026-09-17): `streamlit==1.62.0`, `pandas==3.0.2`,
  `PyJWT[crypto]==2.13.0`, `msal==1.37.0`, `azure-identity==1.21.0`,
  `azure-keyvault-secrets==4.9.0`. All six are in
  `wrapper_repo/requirements-thirdparty.txt`, pandas included: nothing in
  Document Review, the core or planlens caps pandas.
- **Streamlit on App Service**: bind `--server.port ${PORT:-8000}` (the
  probe is on 8000; 8501 gets the container recycled);
  `--server.corsAllowedOrigins <published URL>` per origin (`run.sh` builds
  them from the app setting `APP_ORIGINS`, else its CHANGE-ME line), or Streamlit
  refuses the websocket behind IIS/ARR and the page never loads (confirmed
  live 2026-09-17; keeps CORS and XSRF on); health path `/_stcore/health`;
  `STREAMLIT_SERVER_BASE_URL_PATH` if published under a sub-path.
- **SharePoint** from a server: an Entra app registration (client
  credentials, `Sites.Selected`, one per team). Uploads act as the app. Any
  Microsoft cloud, a secret or a certificate, several sites; the setup for
  CfA's administrator is `docs/SHAREPOINT_SETUP.md`.
- **Persistence**: App Service keeps `/home` across restarts; the wrapper
  sets `GEOTECH_WEBAPP_DATA=/home/data/geotech_webapp`. Every conversation
  is also mirrored to SharePoint after each turn, to
  `Shared Documents/GeotechStaffEngineer/conversations/<tester>/document_review/<title>_<date>/`
  on the published app (the owner's folder, 2026-10-21;
  `GEOTECH_SHAREPOINT_ROOT` changes the first part). With one person and no
  sign-in (dosdev) the `<tester>/` segment is left out.
- Python on dosdev: 3.11.9 (the package requires 3.11 or newer).
- Approval path: MOU, then the GHE repo, then the SIA (1-2 weeks), then a
  change board only for connections outside the pilot's boundary (the
  SharePoint tenant: ask), then the PCR (1-2 days). Code must run locally
  first; the App Services team deploys and pushes updates.

## The wrapper repo (`tinyapps/wrapper_repo/`)

```
app.py           sets GEOTECH_DOTENV (the local .env) and GEOTECH_WEBAPP_DATA,
                 then runs aec_document_review.tinyapps_entry.main()
packages.txt     aec-document-review, aec-agent-core and planlens, pinned
                 exactly, + -r requirements-thirdparty.txt
requirements-thirdparty.txt
                 CfA's fleet pins (pandas==3.0.2 included) + Key Vault libraries
run.sh           CfA's Streamlit template (PORT; the published address from
                 the app setting APP_ORIGINS, else its CHANGE-ME line); picks
                 the code route (below); stops with a plain error when the
                 install failed and nothing is installed
.env.example     the settings, for local runs
.gitattributes   keeps run.sh LF
.gitignore       keeps .env (the keys), .venv and caches out of commits
README.md        routes, local run, update, settings
```

An app update is a pin bump in `packages.txt` plus an App Services sync.
The engineers set `APP_ORIGINS` and provision Key Vault; nothing
in our code changes between dosdev and production: the same names are read
from `.env` there and from Key Vault here.

### How the code arrives

`run.sh` takes any of three routes. The app setting `APP_SOURCE` names the
route; unset, `run.sh` takes the first route whose files the repo holds, in
the order A, B, C below, so the bare template takes route C. It prints one
line naming the route and stops with an error if `APP_SOURCE` names a route
whose files are missing. `APP_DRY_RUN=1 bash run.sh` prints the route and
its commands without installing or starting anything.

| Route | Put in the repo | Setting |
| --- | --- | --- |
| **C. Nexus** (the normal one; owner, 2026-11-09) | Only the template: `packages.txt` pins the three packages and includes the third-party pins. | `APP_SOURCE=nexus` |
| **B. Wheels** (the backup, if Nexus lacks a package) | `wheels/aec_agent_core-1.0.1-py3-none-any.whl` and `wheels/aec_document_review-0.1.1-py3-none-any.whl` (the lead builds them, `python -m build --wheel packages/<name>`). They install with `requirements-thirdparty.txt`; planlens 0.15.0, which Document Review's wheel pins exactly, and every other dependency resolve from Nexus. A planlens wheel may sit beside them. | `APP_SOURCE=wheel` |
| **A. Source folder** (development copies) | `app_src/aec_document_review/` (copied from `packages/aec-document-review/`) and `app_src/requirements.txt` holding `aec-agent-core==1.0.1`, `planlens==0.15.1` and `websockets>=14,<16`. It goes first on `PYTHONPATH` (`app.py` also adds it for a plain `streamlit run app.py`). For an unreleased core as well, add `app_src/aec_agent_core/` and list the core's own dependencies in place of its pin. | `APP_SOURCE=source` |

Tests: `webapp/tests/test_tinyapps_wrapper.py` (each route's dry run, the
import line, the pins against the package's pyproject, pandas==3.0.2).

### A deployment's own instructions (REF-64 (B))

A deployment's conventions for the agents (units, formats, naming, house
style) live in a folder in the wrapper repo, for example `instructions/`
with `all.md` (every agent and helper) and one `<role>.md` per role:
Document Review's roles are `main`, `general-purpose` (the reading helper)
and `figure`. `GEOTECH_INSTRUCTIONS` names the folder by its full path. On
the App Service it is an app setting (the repo sits at `/home/site/wwwroot`,
so `/home/site/wwwroot/instructions`), not a Key Vault secret: the core
reads only the environment. Locally a `.env` line does it. Each role's text
goes last in its prompt, under protected rules it cannot switch off, at
most 4,000 characters a role; Connection diagnostics says what each role
gets. Details: `webapp/README.md` §3 "Your deployment's own instructions";
example folder `docs/examples/deployment_instructions/`.

A deployment's own references (REF-64 (A), local reference packs named by
`GEOTECH_REFERENCES_PACKS`) belong to the geotechnical page's reference
library. Document Review has no reference library, so a Tiny Apps
deployment leaves that setting unset.

## Per-tester Prompter keys

Each tester on the published app gets their own Prompter key, so each has
their own budget, and the list of keys is the pilot's access list.

**Who counts as a tester.** A caller identified by the IIS header.
Single-user hosts keep the shared `PROMPTER_API_KEY`: dosdev with
`DEV_IDENTITY`, and a laptop. On the App Service a request with no header
is not served.

**Resolution order** for a tester (`aec_document_review/tinyapps_keys.py`,
`resolve_for`):

1. **The per-tester secret** `PROMPTER-API-KEY--<TESTER>`. The optional
   `PROMPTER-MODEL--<TESTER>` and `PROMPTER-URL--<TESTER>` fall back to the
   shared `PROMPTER-MODEL` / `PROMPTER-URL`.
2. **The JSON secret `PROMPTER-KEYS`**:
   `{"CORP\\jdoe": {"key": "...", "model": "...", "url": "..."}}`. The
   identity is matched case-insensitively, `DOMAIN\user` or `user@domain`.
   `model` and `url` are optional, and a bare string value is taken as the
   key.
3. **The shared `PROMPTER-API-KEY`**, only when
   `PROMPTER_SHARED_KEY_FALLBACK` is `1`/`true`/`yes`/`on` (an app setting,
   a Key Vault secret `PROMPTER-SHARED-KEY-FALLBACK`, or a `.env` line).
   **It is off by default.**
4. If none of these applies, the app refuses politely in the sidebar and
   chat: "No AI key is set up for **jdoe** yet; ask the app owner. (For the
   app owner: this person's key goes in the Key Vault secret
   `PROMPTER-API-KEY--CORP-JDOE`.)"

**Naming `<TESTER>`.** Take the sign-in `DOMAIN\user`, put it in upper case
and turn every run of characters other than a letter or digit into ONE
hyphen: `CORP\jdoe` becomes `CORP-JDOE`, `CORP\j.doe` becomes `CORP-J-DOE`,
`jdoe@example.org` becomes `EXAMPLE-ORG-JDOE`. As an app setting or a `.env`
line the same name uses underscores: `PROMPTER_API_KEY__CORP_JDOE`. A name
longer than 127 characters is never looked up; two people whose names differ
only in punctuation would share a secret name; give either a `PROMPTER-KEYS`
entry instead.

**Behaviour to know.**

- **Tell CfA to set `PROMPTER_SHARED_KEY_FALLBACK=1` until the per-tester
  keys exist,** with the shared `PROMPTER-API-KEY` in Key Vault: the
  fallback does nothing without it. On the published app with only the
  shared key, every tester is refused while the fallback is off.
- A secret added for a new tester takes effect at their next new
  conversation or page reload, with no restart. A rotated key is used only
  after an app restart.
- No key is ever logged or shown. **Connection diagnostics** has a "Prompter
  key source" line with the caller, the source and the per-tester secret's
  NAME.
- To get the exact name for a tester, have them open the app: the refusal
  message shows it.

## Long turns and rate limits

- A turn that ends on an error keeps its tool results, so "continue" picks
  up from them; a helper that stops writes what it read to the
  conversation's `kept/` folder.
- Model calls wait out busy refusals (retry-after, else 15/30/60/120 s); at
  most two helpers run at once; every model request has a 180 s time limit
  and is retried like a busy refusal when it times out.

Settings (environment or app settings, not Key Vault; all optional, listed
in `wrapper_repo/.env.example`): `GEOTECH_TOKENS_PER_MINUTE` (pace calls a
little under the deployment's limit; unset means no pacing),
`GEOTECH_MAX_HELPERS` (2), `GEOTECH_HELPER_MAX_MODEL_CALLS` (30),
`GEOTECH_AGENT_BUSY_TRIES` (5), `GEOTECH_BUSY_TURN_MAX_S` (900),
`GEOTECH_PROMPTER_STREAM_USAGE` (on), and the figure helper's two switches,
`GEOTECH_FIGURE_HELPER` and `GEOTECH_PLOT_CODE` (on; `0` turns each off).
On the App Service the wrapper's `app.py` sets `GEOTECH_PLOT_CODE=0` unless
the app setting says `1`: the box that code runs in limits Python, not the
operating system (INF-16), and one process holds every tester's files. The
figure helper still draws, with the fixed plotting tool.

## The owner's first round on the test machine (paste-ready checklist)

Do these in order. Each step says what you should see. If a step fails,
stop and copy the last screen's text into a note for the lead.

**Before you start.** The three packages are on PyPI
(`aec-document-review` 0.1.1, `aec-agent-core` 1.0.1 and
`planlens` 0.15.1, all since 2026-11-10). The company's Nexus mirror carries a
release a day or two after PyPI; step 2 shows whether it has them. You need
from CfA: the Prompter values (URL, model, key, CA bundle) and, for step 7,
the four SharePoint values. If the SharePoint values have not arrived, do
steps 1-6 and leave step 7 for later.

1. **Make the wrapper repo.** On GitHub Enterprise, create a new private
   repository (for example `document-review-tinyapp`). Copy into it every
   file in `tinyapps/wrapper_repo/` of this public repository: `app.py`,
   `packages.txt`, `requirements-thirdparty.txt`, `run.sh`,
   `.env.example`, `.gitattributes`, `.gitignore`, `README.md`. Do not add
   `app_src/` or `wheels/`. Commit and push. (`.gitignore` keeps the `.env`
   of step 3, which holds your keys, and the `.venv` folder of step 2 out
   of every later commit.) If you already have a wrapper repository from
   the September round, or CfA made one for the pilot, use that one
   instead: delete its old files, put these in their place, commit and
   push.
2. **Get it onto the test machine.** On dosdev, open a terminal in VS Code
   and run, with your repository's address:
   ```
   git clone <your GHE repository address> document-review-tinyapp
   cd document-review-tinyapp
   python -m venv .venv
   .venv\Scripts\python -m pip install -r packages.txt
   ```
   Every command below starts with `.venv\Scripts\python`, the folder's own
   Python, so nothing needs "activating" (PowerShell often refuses the
   activate script). Run them in the `document-review-tinyapp` folder; in a
   new terminal, `cd` there first. You should see "Successfully installed ... aec-document-review-0.1.1 ...
   aec-agent-core-1.0.1 ... planlens-0.15.1 ... pandas-3.0.2 ...". If pip
   says it cannot find `aec-document-review` (or the core or planlens), the
   mirror does not carry it yet: wait a day, or ask CfA (ask 1 below). If
   pip prints a conflict, copy it into the note and stop.
3. **Fill in the settings.** Run `copy .env.example .env`, then open `.env`
   in VS Code and fill in `PROMPTER_URL`, `PROMPTER_MODEL`,
   `PROMPTER_API_KEY` and `PROMPTER_CA_BUNDLE` (from CfA). Fill in the four
   SharePoint lines (`GRAPH_TENANT_ID`, `GRAPH_CLIENT_ID`,
   `GRAPH_CLIENT_SECRET`, `SHAREPOINT_SITE_URL`) if you have them; leave
   them empty if not. Set `DEV_IDENTITY` to your own `DOMAIN\user`. Save.
   Never commit `.env` (it holds keys).
4. **Start the app.** Run `.venv\Scripts\python -m streamlit run app.py`.
   The first time Streamlit runs on a machine it may stop in the terminal
   at "Welcome to Streamlit! ... Email:"; just press Enter. A browser tab
   opens at `http://localhost:8501`. You should see the **Document Review**
   page, "Signed in as <you>" in the sidebar, and the model's name. In the
   sidebar, open **Connection diagnostics** and press **Run connection
   tests**: the plain, stream and tool-call checks should all pass. If the
   page instead shows the "No model configured" banner, a Prompter
   value in `.env` is missing or misspelt.
5. **Run one short review.** Drag a small PDF (a public drawing sheet or a
   few pages of a public report) into the upload box. The orientation turn
   appears by itself within a minute or two. Then type: *"Mark up the
   first page with two review comments and give me a one-page Word memo of
   them."* You should get two download cards: a `..._marked.pdf` and a
   `.docx`. Open both.
6. **Stop the app.** Press Ctrl+C in the terminal.
7. **Check the SharePoint copy** (only with the four SharePoint values).
   First run
   `.venv\Scripts\python -m aec_document_review.graph_sharepoint --check`:
   every line should say PASS (or SKIP), none FAIL. Start the app again
   (step 4), open the conversation from step 5 in the sidebar, and press
   **Sync now** under **Permanent storage** in the sidebar. On the
   SharePoint site, open
   `Shared Documents/GeotechStaffEngineer/conversations/document_review/`
   (on the published app, with sign-in, it is
   `.../conversations/<you>/document_review/`): a folder named after the
   conversation and its date holds the transcript, the upload, the
   marked-up PDF and the Word memo.
8. **Deploy.** Send CfA asks 2-5 below, with your repository's address. When
   they report it is deployed, open the published address. You should see
   the same page with your own sign-in name; repeat step 5 there, and step 7
   with the `conversations/<you>/document_review/` folder. If the page says
   "No AI key is set up for **<you>** yet", CfA has set neither the shared
   `PROMPTER-API-KEY` (with the fallback) nor your own key: send them ask 4
   again, with the secret name that message shows.
9. **Send the lead your note**: which steps passed, and the text of anything
   that failed.

## What to ask CfA (paste-ready)

1. **Nexus carries our three public packages.** "Please make sure the Nexus
   mirror carries `aec-document-review==0.1.1`, `aec-agent-core==1.0.1` and
   `planlens==0.15.1` from PyPI, and clears them through the firewall if it
   holds them back. They are our own packages, public on PyPI." (If this
   cannot be done, the backup is route B: "May our repo carry two small
   pure-Python wheels of ours in `wheels/`, installed by path?")
2. **Deploy the repo.** "Please deploy `<your GHE repository address>` as a
   Streamlit Tiny App. The startup command is `bash run.sh`; it installs
   `packages.txt` from Nexus with `--index-url $PIP_INDEX_URL`. Does your
   deploy run `run.sh`'s install, or does it install `packages.txt` itself?
   Either works with this repo."
3. **Fill in the published address.** "Please set the app setting
   `APP_ORIGINS` to the app's published address (several addresses:
   separate them with a space), and tell us the address." (Replacing
   `https://CHANGE-ME` in `run.sh` does the same, but is lost when the
   file is updated.)
4. **Settings in Key Vault.** "Please put these in the app's Key Vault and
   set `KV_NAME`: `PROMPTER-URL`, `PROMPTER-MODEL`, `PROMPTER-CA-BUNDLE`;
   the shared pilot key as `PROMPTER-API-KEY`, with
   `PROMPTER-SHARED-KEY-FALLBACK` = `1`, so a tester without a key of their
   own uses it until theirs exists (the fallback does nothing without the
   shared key); my own key as `PROMPTER-API-KEY--<MYDOMAIN>-<MYUSER>` (my
   sign-in `<MYDOMAIN>\<myuser>` in capitals, each run of other characters
   one hyphen: `CORP\jdoe` becomes `PROMPTER-API-KEY--CORP-JDOE`); one
   `PROMPTER-API-KEY--<TESTER>` per further tester, named the same way (the
   app's refusal message shows each name); and the SharePoint registration's
   `GRAPH-TENANT-ID`, `GRAPH-CLIENT-ID`, `GRAPH-CLIENT-SECRET` (or a
   certificate) and `SHAREPOINT-SITE-URL`. The registration's setup is in
   `docs/SHAREPOINT_SETUP.md` of our public repository."
5. **Health check.** "Please set the App Service health check path to
   `/_stcore/health`."

Still open with CfA from earlier rounds (BACKLOG OWN-44): is every tester's
key on a deployment with image input, passed through at full resolution;
does the gateway pass streaming; is the SharePoint tenant inside the pilot's
boundary; does IIS pass `X-Windows-Auth-Header` and replace any value a
browser sends; what the Prompter returns when a key's budget is spent, its
tokens-per-minute limit and its per-request time limit; are the fleet pins
still the six above.

## Monthly obligations

The pilot's monthly check-in (performance, friction, feature asks) and
cooperation with CfA's telemetry and optimisation. Put findings in this
file.
