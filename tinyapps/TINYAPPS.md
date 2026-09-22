# Tiny Apps — plan of record and working notes

**Status (2026-09-21): building.** The pilot was awarded 2026-09-03; on
2026-09-21 the owner has GitHub Enterprise access and the **dosdev**
development environment (VS Code there, push to GHE), and
`pip install geotech-staff-engineer` WORKED on dosdev (Python 3.11.9). The
Prompter key is being requested. The Funhouse/Databricks app stays the fast
tester and backup; both deployments share one PyPI package and one `webapp`.

Sources, all local-only under `tinyapps/reference/` (gitignored — Department
documents are never committed): `Data.State Tiny Apps User Guide v1.1`
(2026-03-06, 20 pp, text extract beside it), `TinyApps Pilot Guidelines`, and
CfA's **`exampleCode`** repo (module versions 2026-08-28 → 2026-09-17: the
`settings.py` / `prompter.py` / `sharepoint.py` / `auth.py` patterns and a
Streamlit starter). Support: CfATinyAppsSupport@state.gov; GitHub:
CFAGitHubSupport@state.gov.

## What the app is FOR on this host (owner, 2026-09-21)

The Department already has a widely used, user-customisable chatbot (Palantir
Foundry AIP Chatbot). This app exists for what that cannot do: **vision** —
looking at drawings, scans, figures and reviewer markups — and **document
creation** — Word memos, marked-up PDFs, calculation packages, figures. Two
pages, one app, no duplicate:

| Page | URL | What it is |
|---|---|---|
| **Document Review** (default) | `/review` | a general document-review agent for architects, construction managers and engineers of any discipline: drawings, specifications, submittals, RFIs, reports, calc packages. On upload it orients itself automatically (one cheap turn: what this is, how it is organised, what is text vs picture, existing markups, what it could do next), then it is a normal chat with the full document, vision and file tools and a default habit of handing back a `.docx` or a marked-up PDF |
| **GeotechStaffEngineer** | `/geotech` | the geotechnical staff engineer exactly as on Databricks — the deepest worked example of both habits |

Implementation: `webapp/profiles.py` (an `AppProfile` per page; the review
page builds `build_deep_agent(allowed_agents=(), reference_mode="off",
system_prompt=build_document_review_prompt())`) and `webapp/tinyapps_entry.py`
(`st.navigation` over two `st.Page`s that both run `webapp/app.py`).

## Environment facts (from the guide + exampleCode; verified where marked)

- **Azure App Service (Azure Government), Linux, behind an IIS/ARR tier** that
  performs Windows authentication. No driver proxy — the Databricks websocket
  saga does not apply.
- **Identity**: IIS passes `X-Windows-Auth-Header: DOMAIN\user`; Streamlit reads
  it via `st.context.headers`. Trusted because the App Service accepts traffic
  only from the IIS servers and the IIS module replaces any client value — the
  app never verifies it and never lets a user type one. Locally `DEV_IDENTITY`
  stands in. → `webapp/identity.py`.
- **Secrets**: Key Vault named by the `KV_NAME` app setting, read with the
  managed identity (`IDENTITY_ENDPOINT` present ⇒ `APP_ENV=azure`);
  `<name>.vault.usgovcloudapi.net`; secret names hyphenated, env names
  underscored; a real env var always wins. → `webapp/tinyapps_settings.py`.
- **Prompter** = "an Azure OpenAI-style chat-completions API": `PROMPTER_URL`
  (full URL ending `/chat/completions`), `PROMPTER_MODEL` (the deployment
  name), `PROMPTER_API_KEY` sent as BOTH `api-key` and `Authorization: Bearer`,
  optional `PROMPTER_CA_BUNDLE` (internal CA, PEM text in Key Vault). Keys are
  per deployment. 180 s per call; the proxy times HTTP out near 120 s (our
  turns run detached from the request). Strict `json_schema` works. →
  `webapp/tinyapps_engine.py` (`ChatOpenAI`, `max_completion_tokens`).
- **The pilot key serves ONE model, $50/month.** The deployment behind it
  MUST accept image inputs or the app's reason to exist fails — the first
  question to the team.
- **Packages** from the Nexus mirror only (`PIP_INDEX_URL`); no outbound
  internet from the App Service. The Nexus firewall quarantines packages with
  CVEs — a build that installed last month can fail today. CfA fleet pins
  (2026-09-17): `streamlit==1.62.0`, `pandas==3.0.2`, `PyJWT[crypto]==2.13.0`,
  `msal==1.37.0`, `azure-identity==1.21.0`, `azure-keyvault-secrets==4.9.0`.
- **Streamlit on App Service**: bind `--server.port ${PORT:-8000}` (the probe
  is on 8000; 8501 gets the container recycled); `--server.corsAllowedOrigins
  <published URL>` per origin, or Streamlit refuses the websocket behind
  IIS/ARR and the page never loads (confirmed live 2026-09-17; keeps CORS and
  XSRF ON); health path `/_stcore/health`; `STREAMLIT_SERVER_BASE_URL_PATH` if
  published under a sub-path.
- **SharePoint** from a server: an Entra app registration (client credentials,
  `Sites.Selected`, one per team) on the PUBLIC Graph endpoints — the M365
  tenant is commercial-side. Uploads act as the app. → `webapp/graph_sharepoint.py`
  (SDK-free; the same six methods the app already calls).
- **Persistence**: App Service keeps `/home` across restarts; the wrapper sets
  `GEOTECH_WEBAPP_DATA=/home/data/geotech_webapp`. Conversations also mirror
  to SharePoint as before.
- Multi-tenant shared VM, one slot; CfA monitors usage; heavy fem2d / Monte
  Carlo runs may need throttling courtesy. Python on dosdev: 3.11.9 (we
  require ≥3.10).
- Approval path: MOU → GHE repo → SIA (1–2 wk; business-justification script
  in guide §5.3.1) → DT CAB only for connections outside the Data.State
  boundary (usdos.sharepoint.com — ask) → PCR (1–2 d). Code must run locally
  first; the App Services team deploys and pushes updates (§7.2).

## Deployment architecture: THIN WRAPPER repo (`tinyapps/wrapper_repo/`)

```
app.py           sets GEOTECH_DOTENV (local .env) + GEOTECH_WEBAPP_DATA, runs
                 webapp.tinyapps_entry.main()
packages.txt     geotech-staff-engineer pin + CfA fleet pins + Key Vault libs
run.sh           CfA's Streamlit template (PORT, corsAllowedOrigins CHANGE-ME)
.env.example     the eight settings the engineers provide + DEV_IDENTITY
README.md
```

Everything else arrives from Nexus as the released package. An app update =
bump the pin + ask App Services to sync. The engineers fill in
`corsAllowedOrigins` and provision Key Vault; **nothing in our code changes
between dosdev and production** — the same names are read from `.env` there
and Key Vault here.

## Built 2026-09-21 (on master, unreleased — candidate 5.26.0)

- `webapp/tinyapps_settings.py` — CfA's `get_setting` pattern.
- `webapp/identity.py` — header / `DEV_IDENTITY` / `GEOTECH_USER_EMAIL` →
  `Identity` (folder key, display name, markup author).
- `webapp/tinyapps_engine.py` — Prompter `ChatOpenAI` from `PROMPTER_*`;
  registers the model builder; `GEOTECH_PROMPTER_DISABLE_STREAMING`.
- `webapp/graph_sharepoint.py` — SDK-free Graph file manager;
  `sharepoint_store` prefers it when `GRAPH_*` + `SHAREPOINT_SITE_URL` are set.
- `webapp/engine_config.is_tinyapps_deployment` / `is_keyless_deployment` —
  no personal key read or named; diagnostics show the Prompter settings.
- `webapp/profiles.py` + `webapp/tinyapps_entry.py` — the two pages.
- `funhouse_agent/deep/prompt.DOCUMENT_REVIEW_PROMPT`;
  `build_deep_agent(system_prompt=…)`; an empty `allowed_agents` drops the
  dispatch tools.
- `webapp/core.register_thread_root` — per-user, per-page conversation roots
  looked up by thread id (the detached worker needs no session).
- Document OUTPUT (Opus build, same night): `write_docx` (Markdown → Word,
  `calc_package/docx_renderer.py`, python-docx) and `annotate_document`
  (planlens `markup_writer` + toolkit tool + app bridge: notes, highlights,
  boxes, callouts, replies onto a COPY of the PDF, readable back by planlens'
  own `markups()`).
- Tests: `webapp/tests/test_tinyapps_wiring.py`, `test_tinyapps_entry.py`.

## dosdev test recipe (owner, once the values arrive)

1. Clone the wrapper repo; `python -m pip install -r packages.txt`
   (proves the whole tree clears Nexus — pandas 3.0.2 and streamlit 1.62.0
   included); `cp .env.example .env` and fill it in.
2. `python -m streamlit run app.py` → Document Review page; sidebar shows
   "Signed in as …" (`DEV_IDENTITY`) and the model; Connection diagnostics
   → the Prompter self-tests (plain / stream / tool call).
3. Upload a PDF → the orientation turn appears on its own. Ask for a
   marked-up copy and a Word summary → two cards.
4. Switch to the GeotechStaffEngineer page → its own conversation list.
5. Sidebar Permanent storage → Sync now → the folder appears on the site.
6. `pytest webapp/tests -q` in the same venv.

## Office-hours / App Services questions (revised 2026-09-21)

1. **Does the model deployment behind our key accept IMAGE inputs, and is it
   GPT-4.1-class or better?** (make-or-break) Does the gateway pass streaming?
2. `PROMPTER_URL` / `PROMPTER_MODEL` / key; is `PROMPTER_CA_BUNDLE` needed?
3. The published origin(s) for `corsAllowedOrigins`.
4. SharePoint app registration for the team site (the four `GRAPH_*` /
   `SHAREPOINT_SITE_URL` values); is usdos.sharepoint.com inside the boundary
   (CAB)?
5. Is the identity header passed for pilot Streamlit apps (the starter says
   yes)?
6. Any objection to installing our package from Nexus vs vendoring; the
   fleet-pin expectations for our dependency tree.
7. Resource envelope on the shared VM (fem2d meshes, Monte Carlo runs).

## Monthly obligations

AI Strikeforce check-in (performance, friction, feature asks) + cooperate
with CfA telemetry/optimization. Put findings in this file.
