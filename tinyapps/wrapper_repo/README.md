# GeotechStaffEngineer — Tiny Apps wrapper

A document-review assistant for engineers, architects and construction
managers that can **look at the pages** (drawings, scans, figures, reviewer
markups) and **produce documents back** (Word memos, marked-up PDFs,
calculation packages, figures), with the geotechnical staff engineer as its
deepest worked example. It complements the Department's chatbot rather than
duplicating it: the value is vision and deliverables.

This repository holds the launch shell and, depending on the route chosen
below, our code. By default the application is the released
`geotech-staff-engineer` package from the Data.State Nexus mirror.

```
app.py                       sets two environment defaults, then runs the app
packages.txt                 route C: our package pin + the third-party pins
requirements-thirdparty.txt  CfA fleet pins + Key Vault libraries (every route)
run.sh                       CfA's Streamlit startup template; picks the route
.env.example                 the settings the engineers provide, for local runs
.gitattributes               keeps run.sh LF (bash on App Service fails on CRLF)
app_src/                     route A only: our source
wheels/                      route B only: our wheels
```

## Choosing how the code arrives (CfA's answer)

`run.sh` takes one of three routes. The app setting `APP_SOURCE` names it;
without it, `run.sh` takes the first route whose files the repo holds, in
the order below. It prints one line naming the route, and it stops with an
error if `APP_SOURCE` names a route whose files are missing.

| Route | Put in the repo | Setting |
| --- | --- | --- |
| **A. Source folder** | `app_src/` holding our importable packages (for example `app_src/webapp/`), and optionally `app_src/requirements.txt` listing their own dependencies. `run.sh` puts `app_src/` first on `PYTHONPATH` and installs `requirements-thirdparty.txt` plus that file from Nexus. | `APP_SOURCE=source` |
| **B. Wheels** | Our wheel(s) in `wheels/*.whl`. `run.sh` installs them with `requirements-thirdparty.txt`; their dependencies resolve from Nexus. | `APP_SOURCE=wheel` |
| **C. Nexus** (today's) | Nothing beyond this template: `packages.txt` names our package, and `run.sh` installs it with the third-party pins from Nexus. | `APP_SOURCE=nexus` |

Today's single package (the GSE app with both pages) works under all three:
its packages in `app_src/` with its dependencies in `app_src/requirements.txt`,
its wheel in `wheels/`, or its pin in `packages.txt`. After the app split,
route A carries Document Review's source.

A build from master since the app split's step 11 (a) (BACKLOG INF-32)
also needs the shared core `aec-agent-core`, which the app imports its model
connections and vision tools from: on route A its `aec_agent_core/` folder
(from `packages/aec-agent-core/`) in `app_src/` beside the app's packages;
on route B its wheel beside the app's in `wheels/`; on route C it comes from
Nexus once it is published (step 12).

To check a choice without installing anything, run
`APP_DRY_RUN=1 bash run.sh`: it prints the route and the commands it would
run, then exits.

## Run locally (dosdev)

```bash
python -m pip install -r packages.txt
cp .env.example .env          # fill in the Prompter and SharePoint values
python -m streamlit run app.py
```

On route A, install `requirements-thirdparty.txt` (and
`app_src/requirements.txt`) instead of `packages.txt`; `app.py` finds
`app_src/` itself. On route B, install the wheels with
`requirements-thirdparty.txt`.

Set `DEV_IDENTITY` in `.env` to stand in for the signed-in user; deployed,
IIS supplies it in the `X-Windows-Auth-Header` header and the value is ignored.

## Update the deployed app

On route C, bump the `geotech-staff-engineer` pin in `packages.txt` to a
release tested on dosdev; on route A or B, replace `app_src/` or the wheels
with the tested ones. Push, and ask the App Services team to sync.

## Settings the engineers provide

| Value | Purpose |
| --- | --- |
| `PROMPTER_URL`, `PROMPTER_MODEL`, `PROMPTER_API_KEY`, `PROMPTER_CA_BUNDLE` | the model endpoint (the deployment must accept image inputs — the app's vision tools depend on it) |
| `PROMPTER-API-KEY--<TESTER>` per tester (optional `PROMPTER-MODEL--<TESTER>`, `PROMPTER-URL--<TESTER>`) | each tester's OWN key on the published app. `<TESTER>` is the sign-in `DOMAIN\user` in upper case, every run of other characters one hyphen: `CORP\jdoe` → `PROMPTER-API-KEY--CORP-JDOE` |
| `PROMPTER-KEYS` (optional) | the alternative: one JSON secret, `{"CORP\\jdoe": {"key": "…", "model": "…", "url": "…"}}` (`model`, `url` optional) |
| `PROMPTER_SHARED_KEY_FALLBACK` (optional) | `1` lets a signed-in tester with no key of their own use the shared `PROMPTER_API_KEY`. Off by default: a tester without a key is told politely to ask the owner |
| `GRAPH_TENANT_ID`, `GRAPH_CLIENT_ID`, `GRAPH_CLIENT_SECRET`, `SHAREPOINT_SITE_URL` | the app registration that mirrors conversations to the team's SharePoint site |
| `KV_NAME` (app setting) | the Key Vault the values above live in |
| `--server.corsAllowedOrigins` in `run.sh` | the published URL of the app |

Support: CfA's Tiny Apps support mailbox (its address is in the Tiny Apps user guide).
