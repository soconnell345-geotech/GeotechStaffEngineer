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

## 5. The owner's checklist

The one current copy is in `tinyapps/TINYAPPS.md`: "The owner's first
round on the test machine (paste-ready checklist)" and "What to ask CfA
(paste-ready)". This section held a copy as built; the review fixes (§8)
changed steps 1, 2, 4, 7 and 8 and ask 4, so the copy was dropped rather
than kept in step by hand.

## 6. What still needs CfA or the owner

- **The lead:** publish the three packages (CHECK_STEP13 §9), then tell the
  owner; the mirror follows a day or two later.
- **CfA (asks 1-5 in §5):** the three packages on Nexus and cleared through
  the firewall (else may the repo carry two wheels); deploy the repo and say
  whether their deploy runs `run.sh`'s install or reads `packages.txt`
  itself (both work); the published address in `corsAllowedOrigins`; the
  settings in Key Vault (Prompter values, the shared key with the
  shared-key fallback, the owner's own key and the other per-tester keys,
  the SharePoint registration per `docs/SHAREPOINT_SETUP.md`); the
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

## 8. Review fixes (2026-11-09, branch `merge/step14b`)

The cloud branch sat on step 13's pre-squash history, so step 14 was
re-made on master `972ffad` as one commit (`b181798`). `tinyapps/TINYAPPS.md`
conflicted with master's REF-64 (B) section, "A deployment's own
instructions". The section is kept and adapted to Document Review: its
roles are `main`, `general-purpose` and `figure`, and `GEOTECH_INSTRUCTIONS`
is an app setting holding the full path (`/home/site/wwwroot/instructions`),
not a Key Vault secret, because the core reads only the environment. The
Python floor reads "3.11 or newer", matching the package's pyproject.

An independent review found four fixes, each its own commit:

| Fix | Commit | What changed |
|---|---|---|
| F1 | `82a9eb9` | CfA ask 4 asked for `PROMPTER-SHARED-KEY-FALLBACK` = `1` but not for the shared `PROMPTER-API-KEY`. Without that key, `tinyapps_keys.resolve_for` refuses every tester, the owner included, so step 8 would fail. The ask now names the shared key with the fallback and the owner's own `PROMPTER-API-KEY--<MYDOMAIN>-<MYUSER>` (generic placeholders). Step 8 says what the "No AI key is set up" refusal means, and the "Behaviour to know" bullet says the fallback needs the shared key. |
| F2 | `fc83722` | `tinyapps/wrapper_repo/.gitignore` (`.env`, `.venv/`, `__pycache__/`, `*.pyc`), because steps 2-3 create `.venv` and `.env` (the keys) inside the clone. `wheels/` is not ignored, because route B commits its wheels. It is named in step 1 and in both file trees. The test `test_template_keeps_the_keys_and_the_venv_out_of_commits` pins it. |
| F3 | `d8a773d` | Steps 2, 4 and 7 run `.venv\Scripts\python -m ...` instead of activating the venv. On Windows, VS Code's terminal is PowerShell, which often refuses the activate script. |
| F4 | `fd98ffb` | The README's route B install names the two wheels, because Windows does not expand `wheels/*.whl`. Step 1 reuses a wrapper repository that already exists. Step 4 says to press Enter at Streamlit's first-run e-mail prompt. "Both wheels pin planlens" became "Document Review's wheel pins it exactly", because the core asks `>=0.15`. |

§5's copy of the checklist was replaced with a pointer to `TINYAPPS.md`, so
only one copy exists.

Re-checks after the fixes (Windows 11, judged by exit codes):

| Check | Exit | Result |
|---|---|---|
| `webapp/tests/test_tinyapps_wrapper.py webapp/tests/test_public_text.py` | 0 | 21 passed |
| Wheels route end to end: the template as committed (`git archive HEAD`), wheels built from this branch, a fresh venv on Python 3.12.15, `PIP_INDEX_URL` unset (so PyPI), `bash run.sh` in Git Bash. The throwaway copy listens on `127.0.0.1` instead of `0.0.0.0`, to avoid a firewall prompt. | 0 | Installed aec-agent-core 1.0.0, aec-document-review 0.1.0, planlens 0.15.0, pandas 3.0.2, streamlit 1.62.0, PyJWT 2.13.0, msal 1.37.0, azure-identity 1.21.0 and azure-keyvault-secrets 4.9.0. `pip check` clean. `/_stcore/health` returned `ok` 200 and `/` returned 200. |
| The page script in Streamlit's `AppTest` | 0 | No exceptions. The title is "Document Review", with the "No model configured" banner and "aec-document-review 0.1.0 · engine: none". |

Not run: Nexus, the App Service, IIS, Key Vault, real Prompter or
SharePoint values, and route C (its packages are not on PyPI yet).
