# Document Review: Tiny Apps wrapper

A document-review assistant for engineers, architects and construction
managers. It **looks at the pages** (drawings, scans, figures, reviewer
markups), not only their text, and **hands documents back** (Word memos,
marked-up PDFs, figures). It complements a general chatbot rather than
duplicating it: the value is vision and deliverables.

This repository holds only the launch shell. The application is the
released `aec-document-review` package, installed from the Nexus mirror
with its two companions, the shared core `aec-agent-core` and the document
layer `planlens`.

```
app.py                       sets two environment defaults, then runs the app
packages.txt                 route C: the three package pins + the third-party pins
requirements-thirdparty.txt  CfA fleet pins + Key Vault libraries (every route)
run.sh                       CfA's Streamlit startup template; picks the route
.env.example                 the settings, for local runs
.gitattributes               keeps run.sh LF (bash on App Service fails on CRLF)
.gitignore                   keeps .env (the keys) and .venv out of commits
wheels/                      route B only: the wheels
app_src/                     route A only: a development copy of the source
```

## Choosing how the code arrives

`run.sh` takes one of three routes. The app setting `APP_SOURCE` names it;
without it, `run.sh` takes the first route whose files the repo holds, in
this order: `app_src/` (A), then `wheels/` (B), then `packages.txt` (C).
So a repo holding only the template takes route C. It prints one line naming the route, and it stops with an
error if `APP_SOURCE` names a route whose files are missing.

| Route | Put in the repo | Setting |
| --- | --- | --- |
| **C. Nexus** (the normal one) | Nothing beyond this template. `packages.txt` pins `aec-document-review==0.1.1`, `aec-agent-core==1.0.1` and `planlens==0.15.1`, and includes `requirements-thirdparty.txt`; `run.sh` installs it all from Nexus. | `APP_SOURCE=nexus` |
| **B. Wheels** (the backup) | The two wheels in `wheels/`: `aec_agent_core-1.0.1-py3-none-any.whl` and `aec_document_review-0.1.1-py3-none-any.whl`. `run.sh` installs them with `requirements-thirdparty.txt`; planlens 0.15.1 (pinned exactly by Document Review's wheel) and every other dependency resolve from Nexus. A planlens wheel may sit in `wheels/` too. | `APP_SOURCE=wheel` |
| **A. Source folder** (development copies) | `app_src/aec_document_review/` (from `packages/aec-document-review/` in the source repository) and `app_src/requirements.txt`, below. `run.sh` puts `app_src/` first on `PYTHONPATH` and installs `requirements-thirdparty.txt` plus that file from Nexus. | `APP_SOURCE=source` |

**Route A's `app_src/requirements.txt`** lists what the copied source needs
from Nexus:

```
aec-agent-core==1.0.1
planlens==0.15.1
websockets>=14,<16
```

To try an unreleased core as well, copy its `aec_agent_core/` folder (from
`packages/aec-agent-core/` in the source repository) into `app_src/` beside
`aec_document_review/`, and in `app_src/requirements.txt` replace the
`aec-agent-core` line with the core's own dependencies (the `dependencies`
list in its `pyproject.toml`).

To check a choice without installing anything, run
`APP_DRY_RUN=1 bash run.sh`: it prints the route and the commands it would
run, then exits.

## Run locally (the test machine)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -r packages.txt
cp .env.example .env          # fill in the Prompter and SharePoint values
python -m streamlit run app.py
```

On route B, install the wheels with `requirements-thirdparty.txt`, naming
both files (Windows does not expand `wheels/*.whl`):

```bash
python -m pip install wheels/aec_agent_core-1.0.1-py3-none-any.whl wheels/aec_document_review-0.1.1-py3-none-any.whl -r requirements-thirdparty.txt
```

On route A, install `requirements-thirdparty.txt` and `app_src/requirements.txt`;
`app.py` finds `app_src/` itself.

Set `DEV_IDENTITY` in `.env` to stand in for the signed-in user; deployed,
IIS supplies it in the `X-Windows-Auth-Header` header and the value is ignored.
A request that reaches the deployed app with no header (the App Service's
own address opened directly, say) is refused with a message: one process
serves every tester there, and it serves nobody it cannot name.

## Update the deployed app

On route C, bump the three pins in `packages.txt` to a release tested on
the test machine (the new `aec-document-review` release names its core and
planlens versions exactly; copy those); on route B, replace the wheels.
Push, and ask the App Services team to sync.

## Settings

The package reads these itself, in this order: a real environment variable
or app setting, then the Key Vault named by `KV_NAME` (secret names use
hyphens: `PROMPTER-API-KEY`), then the local `.env`. Several names begin
`GEOTECH_`: Document Review was built inside a larger project and kept its
setting names, so existing deployments keep working.

| Value | Purpose |
| --- | --- |
| `PROMPTER_URL`, `PROMPTER_MODEL`, `PROMPTER_API_KEY`, `PROMPTER_CA_BUNDLE` | the model endpoint. The deployment must accept image inputs: the app's vision tools depend on it |
| `PROMPTER-API-KEY--<TESTER>` per tester (optional `PROMPTER-MODEL--<TESTER>`, `PROMPTER-URL--<TESTER>`) | each tester's OWN key on the published app. `<TESTER>` is the sign-in `DOMAIN\user` in upper case, every run of other characters one hyphen: `CORP\jdoe` becomes `PROMPTER-API-KEY--CORP-JDOE` |
| `PROMPTER-KEYS` (optional) | the alternative: one JSON secret, `{"CORP\\jdoe": {"key": "...", "model": "...", "url": "..."}}` (`model`, `url` optional) |
| `PROMPTER_SHARED_KEY_FALLBACK` (optional) | `1` lets a signed-in tester with no key of their own use the shared `PROMPTER_API_KEY`. Off by default: a tester without a key is told politely to ask the app owner |
| `GRAPH_TENANT_ID`, `GRAPH_CLIENT_ID`, `GRAPH_CLIENT_SECRET`, `SHAREPOINT_SITE_URL` | the app registration that mirrors conversations to the team's SharePoint site; a certificate may replace the secret (`GRAPH_CLIENT_CERT`, `GRAPH_CLIENT_KEY`, see `.env.example`). Test with `python -m aec_document_review.graph_sharepoint --check` |
| `GEOTECH_SHAREPOINT_ROOT` (optional) | the folder inside the site; default `Shared Documents/GeotechStaffEngineer`. A tester's conversations go to `<root>/conversations/<tester>/document_review/` (one person with no sign-in: `<root>/conversations/document_review/`) |
| `SHAREPOINT_SITE_URLS`, `GRAPH_CLOUD` (optional) | more sites; the Microsoft cloud when the site address does not say it |
| `KV_NAME` (app setting) | the Key Vault the values above live in |
| `GEOTECH_WEBAPP_DATA` (optional) | where conversations live on the machine; `app.py` sets `/home/data/geotech_webapp` on the App Service |
| `GEOTECH_INSTRUCTIONS` (optional; an app setting or a `.env` line, not Key Vault) | the full path of a folder of this deployment's own instructions for the agents (units, formats, house style): `all.md` for every agent and helper, then `main.md`, `general-purpose.md` (the reading helper) and `figure.md`. In an `instructions/` folder of this repository it is `/home/site/wwwroot/instructions` on the App Service. Each role's text goes last in its prompt, under rules it cannot switch off, at most 4,000 characters a role; Connection diagnostics says what each role gets |
| `DEV_IDENTITY` (local only) | stands in for the signed-in user; not read on the App Service |
| `APP_ORIGINS` (app setting) | the published address(es) of the app, as the browser shows them; several are separated by a space or a comma. `run.sh` passes each to Streamlit as `--server.corsAllowedOrigins`. Unset, the `https://CHANGE-ME` line in `run.sh` is used, and until one of the two is set the start log carries a warning and the page does not load through IIS |
| `GEOTECH_PLOT_CODE` (app setting; optional) | `1` lets the figure helper run plotting code the model writes. `app.py` turns it off on the App Service: one process holds every tester's conversations there, and the box the code runs in limits Python, not the operating system. Figures are still drawn, with the fixed plotting tool |

The optional tuning settings (pacing, helpers, retries, the figure helper)
are listed, commented out, in `.env.example`.

Support: CfA's Tiny Apps support mailbox (its address is in the Tiny Apps
user guide).
