# GeotechStaffEngineer — Tiny Apps wrapper

A document-review assistant for engineers, architects and construction
managers that can **look at the pages** (drawings, scans, figures, reviewer
markups) and **produce documents back** (Word memos, marked-up PDFs,
calculation packages, figures), with the geotechnical staff engineer as its
deepest worked example. It complements the Department's chatbot rather than
duplicating it: the value is vision and deliverables.

This repository holds only the launch shell. The application is the released
`geotech-staff-engineer` package from the Data.State Nexus mirror.

```
app.py            sets two environment defaults, then runs the packaged app
packages.txt      the package pin + CfA fleet pins + Key Vault libraries
run.sh            CfA's Streamlit startup template
.env.example      the settings the engineers provide, for local runs
```

## Run locally (dosdev)

```bash
python -m pip install -r packages.txt
cp .env.example .env          # fill in the Prompter and SharePoint values
python -m streamlit run app.py
```

Set `DEV_IDENTITY` in `.env` to stand in for the signed-in user; deployed,
IIS supplies it in the `X-Windows-Auth-Header` header and the value is ignored.

## Update the deployed app

Bump the `geotech-staff-engineer` pin in `packages.txt` to a release tested on
dosdev, push, and ask the App Services team to sync.

## Settings the engineers provide

| Value | Purpose |
| --- | --- |
| `PROMPTER_URL`, `PROMPTER_MODEL`, `PROMPTER_API_KEY`, `PROMPTER_CA_BUNDLE` | the model endpoint (the deployment must accept image inputs — the app's vision tools depend on it) |
| `GRAPH_TENANT_ID`, `GRAPH_CLIENT_ID`, `GRAPH_CLIENT_SECRET`, `SHAREPOINT_SITE_URL` | the app registration that mirrors conversations to the team's SharePoint site |
| `KV_NAME` (app setting) | the Key Vault the values above live in |
| `--server.corsAllowedOrigins` in `run.sh` | the published URL of the app |

Support: CfATinyAppsSupport@state.gov.
