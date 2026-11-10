# INF-34, the app split's step 14: the Tiny Apps wrapper for Document Review

Branch `claude/inf-34-step14`, from `claude/inf-33-step13` at `291fd34`
(step 13, under review; it gained no commits while this was built). Nothing
in `packages/aec-document-review/` or the core was touched. Nothing tagged
or released.

## 1. What changed

| File | Change |
|---|---|
| `tinyapps/wrapper_repo/app.py:41` | `from aec_document_review.tinyapps_entry import main` (was `webapp.tinyapps_entry`); docstring rewritten for Document Review. The `app_src/` path insert, `GEOTECH_DOTENV` and `GEOTECH_WEBAPP_DATA` defaults are unchanged. |
| `tinyapps/wrapper_repo/packages.txt:12-14` | `aec-document-review==0.1.0`, `aec-agent-core==1.0.0`, `planlens==0.15.0` (was `geotech-staff-engineer>=5.26`), then `-r requirements-thirdparty.txt` as before. |
| `tinyapps/wrapper_repo/requirements-thirdparty.txt:16` | `pandas==3.0.2` restored (it was commented out because GSE's python-ags4 capped pandas below 3). Nothing in Document Review, the core or planlens requires or caps pandas (`packages/*/pyproject.toml`; the core names pandas only in a comment); the end-to-end install below resolved it. |
| `tinyapps/wrapper_repo/run.sh` | unchanged (route choice and `APP_DRY_RUN=1` kept, as plan Step 14 says). |
| `tinyapps/wrapper_repo/.env.example` | header names Document Review; SharePoint block says where conversations go; the connection test is `python -m aec_document_review.graph_sharepoint --check`; `GEOTECH_WEBAPP_DATA` listed; `GEOTECH_JUDGMENT_CALLS` (GSE-only) removed; the "from the release after 5.33.0" note removed. Placeholders only (`CORP\jdoe`). |
| `tinyapps/wrapper_repo/README.md` | rewritten: route C (Nexus) the normal one, route B (two wheels) the backup, route A (`app_src/`) for development copies with what goes in it; local run; update; the settings table (CHECK_STEP13 §10). |
| `tinyapps/TINYAPPS.md` | rewritten for Document Review alone: the two-page table, "switch to the geotech page", the geotech-only behaviour switch, fem2d/Monte Carlo resource notes and the GSE build history removed; code paths point at `packages/aec-document-review/`; the test-machine recipe became the owner's numbered first-round checklist (§4 below); the CfA questions rewritten for the owner's 2026-11-09 route (public packages via Nexus; wheels the backup) as five paste-ready asks, plus the still-open list from OWN-44. |
| `webapp/tests/test_tinyapps_wrapper.py` | the `app_src` stand-in is now `aec_document_review/`; new tests: the import line (`:171`), the three pins against `packages/aec-document-review/pyproject.toml`'s version and exact core/planlens pins (`:183`), the fleet pins incl. `pandas==3.0.2` and none of ours in the third-party file (`:193`), no pandas line in the package's or core's dependencies (`:206`), and the shipped template's dry run takes route C and starts Streamlit on `app.py` (`:217`). |

**What goes in `app_src/` now (route A):** `app_src/aec_document_review/`
(copied from `packages/aec-document-review/`) and `app_src/requirements.txt`
with `aec-agent-core==1.0.0`, `planlens==0.15.0`, `websockets>=14,<16` (the
package's own non-fleet dependencies; streamlit is a fleet pin, requests
comes with it). For an unreleased core, add `app_src/aec_agent_core/` and
list the core's `dependencies` in place of its pin.

## 2. Each route's dry run (the real template, `APP_DRY_RUN=1`, `PIP_INDEX_URL=https://nexus.example/simple`)

```
=== nexus   (template only)
Code route: nexus (packages.txt from the feed; chosen by what the repo holds)
Installing Python packages from packages.txt...
DRY RUN: python -m pip install -r <repo>/packages.txt --index-url https://nexus.example/simple
DRY RUN: PYTHONPATH= python -m streamlit run <repo>/app.py --server.port 8000 ...
=== wheel   (the two wheels in wheels/)
Code route: wheel (aec_agent_core-1.0.0-py3-none-any.whl aec_document_review-0.1.0-py3-none-any.whl; chosen by what the repo holds)
Installing Python packages: our wheels + third-party pins...
DRY RUN: python -m pip install <repo>/wheels/aec_agent_core-1.0.0-py3-none-any.whl <repo>/wheels/aec_document_review-0.1.0-py3-none-any.whl -r <repo>/requirements-thirdparty.txt --index-url https://nexus.example/simple
DRY RUN: PYTHONPATH= python -m streamlit run <repo>/app.py --server.port 8000 ...
=== source  (app_src/aec_document_review/ + app_src/requirements.txt)
Code route: source (<repo>/app_src first on PYTHONPATH; chosen by what the repo holds)
Installing Python packages: third-party pins (+ app_src/requirements.txt)...
DRY RUN: python -m pip install -r <repo>/requirements-thirdparty.txt -r <repo>/app_src/requirements.txt --index-url https://nexus.example/simple
DRY RUN: PYTHONPATH=<repo>/app_src python -m streamlit run <repo>/app.py --server.port 8000 ...
```

## 3. End-to-end check of the wheels route

Python 3.11.18 (dosdev runs 3.11.9), a fresh venv, a throwaway copy of the
template. No Nexus in the cloud, so `PIP_INDEX_URL` was unset and pip used
PyPI through the session's proxy (`run.sh` warns and goes on); that is the
only difference from the test machine.

```bash
# wheels from the branch
python -m build --wheel --outdir dist packages/aec-agent-core      # aec_agent_core-1.0.0-py3-none-any.whl
python -m build --wheel --outdir dist packages/aec-document-review # aec_document_review-0.1.0-py3-none-any.whl
# throwaway template + fresh venv
cp -r tinyapps/wrapper_repo wrapper_e2e && mkdir wrapper_e2e/wheels && cp dist/*.whl wrapper_e2e/wheels/
python3.11 -m venv e2e311 && export PATH=$PWD/e2e311/bin:$PATH
cd wrapper_e2e
APP_SOURCE=wheel PORT=8765 GEOTECH_WEBAPP_DATA=<scratch> bash run.sh > e2e_run.log 2>&1 &
```

`e2e_run.log` (trimmed):

```
WARNING: PIP_INDEX_URL not set; using default package source
Successfully installed ... aec-agent-core-1.0.0 aec-document-review-0.1.0 ... deepagents-0.7.24 ...
  langchain-1.4.5 ... msal-1.37.0 ... openai-3.29.0 ... pandas-3.0.2 ... planlens-0.15.0 ...
  PyJWT-2.13.0 ... streamlit-1.62.0 ... azure-identity-1.21.0 azure-keyvault-secrets-4.9.0 ...
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8765
```

No resolver conflict; no Traceback or Error in the log. `pip list` in the
venv: no geotech-staff-engineer, no python-ags4.

```
$ curl -s -w " %{http_code}" http://localhost:8765/_stcore/health
ok 200
$ curl -s -o page.html -w "%{http_code}" http://localhost:8765/
200
```

The page in headless Chromium (Playwright, 8 s after load), its text:

```
Conversations / ➕ New conversation / Session
No model configured. This Tiny Apps deployment reads PROMPTER_URL, PROMPTER_MODEL and
PROMPTER_API_KEY (and the optional PROMPTER_CA_BUNDLE) from Key Vault or the App Service
app settings — locally, from the .env file. ...
Connection diagnostics
aec-document-review 0.1.0 · engine: none
Attachments / Upload ...
📄 Document Review
Reads drawings, specifications, submittals, reports and calculation packages — looking at
the pages, not only the text — and hands back Word memos and marked-up PDFs. ...
```

Then `pkill -f "streamlit run .../wrapper_e2e"`; stopped. The banner is the
"No model configured" one the plan's Step 14 test names.

Not tested in the cloud: the Nexus mirror itself (the packages are not
published yet), the App Service, IIS and Key Vault. Route C was not run for
real for the same reason (its install is route B's resolve with the three
pins from the index instead of two of them from files).

## 4. Tests (judged by pytest's exit code)

GSE venv (`pip install -e packages/aec-agent-core -e .`, Python 3.12):

| Suite | Exit | Result |
|---|---|---|
| `webapp/tests/test_tinyapps_wrapper.py webapp/tests/test_public_text.py webapp/tests/test_docs_currency.py` | 0 | 26 passed |

## 5. The owner's checklist (as in `tinyapps/TINYAPPS.md`)

### The owner's first round on the test machine (paste-ready checklist)

Do these in order. Each step says what you should see. If a step fails,
stop and copy the last screen's text into a note for the lead.

**Before you start.** The lead tells you the three packages are on PyPI
(`aec-document-review` 0.1.0, `aec-agent-core` 1.0.0, `planlens` 0.15.0).
Wait a day or two for the company's Nexus mirror to carry them. You need
from CfA: the Prompter values (URL, model, key, CA bundle) and, for step 7,
the four SharePoint values. If the SharePoint values have not arrived, do
steps 1-6 and leave step 7 for later.

1. **Make the wrapper repo.** On GitHub Enterprise, create a new private
   repository (for example `document-review-tinyapp`). Copy into it every
   file in `tinyapps/wrapper_repo/` of this public repository: `app.py`,
   `packages.txt`, `requirements-thirdparty.txt`, `run.sh`,
   `.env.example`, `.gitattributes`, `README.md`. Do not add `app_src/` or
   `wheels/`. Commit and push.
2. **Get it onto the test machine.** On dosdev, open a terminal in VS Code
   and run, with your repository's address:
   ```
   git clone <your GHE repository address> document-review-tinyapp
   cd document-review-tinyapp
   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install -r packages.txt
   ```
   You should see "Successfully installed ... aec-document-review-0.1.0 ...
   aec-agent-core-1.0.0 ... planlens-0.15.0 ... pandas-3.0.2 ...". If pip
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
4. **Start the app.** Run `python -m streamlit run app.py`. A browser tab
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
   First run `python -m aec_document_review.graph_sharepoint --check`:
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
   with the `conversations/<you>/document_review/` folder.
9. **Send the lead your note**: which steps passed, and the text of anything
   that failed.

### What to ask CfA (paste-ready)

1. **Nexus carries our three public packages.** "Please make sure the Nexus
   mirror carries `aec-document-review==0.1.0`, `aec-agent-core==1.0.0` and
   `planlens==0.15.0` from PyPI, and clears them through the firewall if it
   holds them back. They are our own packages, public on PyPI." (If this
   cannot be done, the backup is route B: "May our repo carry two small
   pure-Python wheels of ours in `wheels/`, installed by path?")
2. **Deploy the repo.** "Please deploy `<your GHE repository address>` as a
   Streamlit Tiny App. The startup command is `bash run.sh`; it installs
   `packages.txt` from Nexus with `--index-url $PIP_INDEX_URL`. Does your
   deploy run `run.sh`'s install, or does it install `packages.txt` itself?
   Either works with this repo."
3. **Fill in the published address.** "Please replace `https://CHANGE-ME` in
   `run.sh`'s `--server.corsAllowedOrigins` with the app's published
   address (one flag per address), and tell us the address."
4. **Settings in Key Vault.** "Please put these in the app's Key Vault and
   set `KV_NAME`: `PROMPTER-URL`, `PROMPTER-MODEL`, `PROMPTER-CA-BUNDLE`;
   one `PROMPTER-API-KEY--<TESTER>` per tester (the app's refusal message
   shows each name); `PROMPTER-SHARED-KEY-FALLBACK` = `1` until every
   tester's key exists; and the SharePoint registration's
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


## 6. What still needs CfA or the owner

- **The lead:** publish the three packages (CHECK_STEP13 §9), then tell the
  owner; the mirror follows a day or two later.
- **CfA (asks 1-5 in §5):** the three packages on Nexus and cleared through
  the firewall (else may the repo carry two wheels); deploy the repo and say
  whether their deploy runs `run.sh`'s install or reads `packages.txt`
  itself (both work); the published address in `corsAllowedOrigins`; the
  settings in Key Vault (Prompter values, per-tester keys, the shared-key
  fallback, the SharePoint registration per `docs/SHAREPOINT_SETUP.md`); the
  health path `/_stcore/health`.
- **CfA, still open from OWN-44:** image input at full resolution on every
  tester's deployment; streaming through the gateway; the SharePoint tenant
  inside the pilot's boundary; the identity header; what a spent budget
  returns and the rate and time limits; the fleet pins still current.
- **The owner:** the first round (§5); OWN-78's notice (`notice.py` is a
  placeholder in the package, not changed here).

## 7. Noted, not fixed

- `run.sh`'s comment still says "our package" for route C (now three); left
  so `run.sh` stays byte-for-byte unchanged, as the plan asks.
- The README and `.env.example` explain that several setting names begin
  `GEOTECH_`; renaming them is the package's change, not the wrapper's.
- The package's prompt still points geotechnical reports to "the
  GeotechStaffEngineer page" (CHECK_STEP13 §11); a package change.
