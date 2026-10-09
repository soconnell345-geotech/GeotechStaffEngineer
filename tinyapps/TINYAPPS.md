# Tiny Apps — plan of record and working notes

**Status (2026-09-23): 5.26.0 RELEASED and BOOTED ON DOSDEV; waiting on
CfA for the Prompter and SharePoint values.** The pilot was awarded
2026-09-03; the owner has GitHub Enterprise access and the **dosdev**
development environment (a virtual desktop, Python 3.11.9, VS Code, push to
GHE). On 2026-09-23 `pip install geotech-staff-engineer==5.26.0` resolved on
dosdev and `python -m streamlit run <site-packages>/webapp/tinyapps_entry.py`
started the app with no Prompter or SharePoint values: the "No model
configured" banner, the chat box disabled — Streamlit itself, the package
import on 3.11 and the two-page entry are proven on the target machine;
nothing model-side is. (The first attempt showed the single geotech page
because the older version was still installed — a symptom to remember.)
The owner is creating a VS Code venv on dosdev and installing from
`packages.txt`, which is also the first test of the CfA fleet pins beside
our tree. The Funhouse/Databricks app stays the fast tester and backup; both
deployments share one PyPI package and one `webapp`.

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
| **Document Review** (default) | `/` | a general document-review agent for architects, construction managers and engineers of any discipline: drawings, specifications, submittals, RFIs, reports, calc packages. On upload it orients itself automatically (one cheap turn: what this is, how it is organised, what is text vs picture, existing markups, what it could do next), then it is a normal chat with the full document, vision and file tools and a default habit of handing back a `.docx` or a marked-up PDF |
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
  per deployment; on the published app each TESTER has their own (see
  "Per-tester Prompter keys" below). 180 s per call; the proxy times HTTP out near 120 s (our
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

## Per-tester Prompter keys (built 2026-10-09, unreleased)

Each tester on the published app gets their own Prompter key, so each has
their own budget, and the list of keys is the pilot's access list.

**Who counts as a tester.** A caller identified by the IIS header
(`Identity.multi_user`). Single-user hosts keep the shared
`PROMPTER_API_KEY` exactly as before: dosdev with `DEV_IDENTITY`, a laptop,
Databricks, and any request with no header.

**Resolution order** for a tester (`webapp/tinyapps_engine.resolve_for`):

1. **The per-tester secret** `PROMPTER-API-KEY--<TESTER>`. The optional
   `PROMPTER-MODEL--<TESTER>` and `PROMPTER-URL--<TESTER>` fall back to the
   shared `PROMPTER-MODEL` / `PROMPTER-URL`. A tester's own model wins over
   the picker's choice, because a key serves one deployment.
2. **The JSON secret `PROMPTER-KEYS`**:
   `{"CORP\\jdoe": {"key": "…", "model": "…", "url": "…"}}`. The identity is
   matched case-insensitively, `DOMAIN\user` or `user@domain`. `model` and
   `url` are optional, and a bare string value is taken as the key.
3. **The shared `PROMPTER-API-KEY`**, only when
   `PROMPTER_SHARED_KEY_FALLBACK` is `1`/`true`/`yes`/`on` (an app setting,
   a Key Vault secret `PROMPTER-SHARED-KEY-FALLBACK`, or a `.env` line).
   **It is off by default.**
4. If none of these applies, the app refuses politely in the sidebar and
   chat: "No AI key is set up for **jdoe** yet — ask the app owner. (For the
   app owner: this person's key goes in the Key Vault secret
   `PROMPTER-API-KEY--CORP-JDOE`.)" There is no error and no trace.

**Naming `<TESTER>`** (`tester_suffix`). Start from the identity key
(`CORP\jdoe` → `corp__jdoe`). Put it in upper case and turn every run of
characters other than a letter or digit into ONE hyphen: `CORP-JDOE`,
`CORP\j.doe` → `CORP-J-DOE`, `jdoe@state.gov` → `STATE-GOV-JDOE`.

- The result is valid for Key Vault: letters, digits and hyphens, starting
  with a letter.
- The only `--` in a name is the separator.
- The app asks the vault for the upper-case name. Key Vault names are
  case-insensitive, but the upper-case spelling is the one that is
  guaranteed to match.
- As an App Service app setting or a `.env` line, the same name uses
  underscores: `PROMPTER_API_KEY__CORP_JDOE`.
- A name longer than 127 characters is never looked up. Such a tester needs
  a `PROMPTER-KEYS` entry.
- Two people whose names differ only in punctuation (`j.doe` / `j_doe`)
  would share a secret name. Give them `PROMPTER-KEYS` entries instead.

**Behaviour to know.**

- **Tell CfA to set `PROMPTER_SHARED_KEY_FALLBACK=1` until the per-tester
  keys exist.** On the published app with only the shared key, every tester
  is refused while it is off.
- Per-tester keys need the identity header (open question 5). Without the
  header, everyone is single-user and uses the shared key. To make sure
  nobody falls back to the shared key, leave `PROMPTER-API-KEY` out of the
  production vault.
- **New testers:** missing secrets are not cached. A secret added for a new
  tester takes effect at their next new conversation or page reload, with
  no restart.
- **Rotated keys:** a key that is found is cached for the life of the
  process. After CfA rotates a key, the new one is used only after an app
  restart, the same as the shared key.
- **Isolation:** the builder resolves the key for the caller it is handed
  every time it is called and caches no client. `app.py` passes the
  session's `Identity` to `engine_config.resolve_engine(…, identity=…)`, and
  the engine lives in that session's state only.
- **Secrecy:** no key is ever logged or shown. **Connection diagnostics** has
  a "Prompter key source" line with the caller, the source (per-tester
  secret / PROMPTER-KEYS mapping / shared key) and the per-tester secret's
  NAME. `PROMPTER_KEYS` appears only as a length.
- **Budget messages** speak of the user's own budget ("Your AI budget is
  used up …", `funhouse_agent/error_text.py`).

**What to tell CfA.** For each tester, create a Key Vault secret named
`PROMPTER-API-KEY--<TESTER>` whose value is that tester's Prompter key.
Build `<TESTER>` from the tester's sign-in `DOMAIN\user`: upper case, with
every run of other characters turned into one hyphen. For example,
`CORP\jdoe` → `PROMPTER-API-KEY--CORP-JDOE`. If a tester's key is for a
different model deployment, add `PROMPTER-MODEL--<TESTER>` and, if needed,
`PROMPTER-URL--<TESTER>`. Keep the shared `PROMPTER-URL`, `PROMPTER-MODEL` and
`PROMPTER-CA-BUNDLE`. Alternatively, CfA can provide a single JSON secret,
`PROMPTER-KEYS`. To get the exact name for a tester, have them open the
app: the refusal message shows it.

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
