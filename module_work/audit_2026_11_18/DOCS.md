About 560 statements checked, 15 false, 6 corrected here

Audit of 2026-11-19 (03:10 to 03:20 UTC) of the Document Review documents against master 3319d3a and the package index. The index holds, as latest: aec-agent-core 1.3.0; aec-document-review 1.3.0 (requires `aec-agent-core==1.3.0`, `planlens==0.17.0`); planlens 0.17.0; geotech-staff-engineer 5.36.0 (requires `aec-agent-core==1.1.0`, `planlens>=0.16.0`, `geotech-references>=1.8.0`). The count of statements is approximate (two helper sessions counted their own parts).

## False statements

| Document | The statement | What is true | Corrected here or report only |
|---|---|---|---|
| `module_work/cloud_briefs/COMMON_DOCREVIEW.md` | here a `v*`, `core-v*` or `document-review-v*` tag publishes | `publish-core.yml` and `publish-document-review.yml` run on `aec-v*` only; `publish.yml` on `v*` (the app). `core-v*` and `document-review-v*` start nothing | corrected (7bf9dd8) |
| the same | Document Review pins `aec-agent-core==1.0.1`, `planlens==0.15.1` on master | `==1.3.0` and `==0.17.0` (`packages/aec-document-review/pyproject.toml`) | corrected (7bf9dd8) |
| `packages/aec-document-review/README.md`, "Settings" | read from the environment, then a Key Vault, then a `.env` file in the working folder | after the environment: the Key Vault only on an Azure host, else the `.env` only on a local run, and the file is the one `GEOTECH_DOTENV` names, else the working folder's (`tinyapps_settings.get_setting`, `dotenv_path`) | corrected (9d9ace1) |
| `tinyapps/wrapper_repo/README.md`, "Settings" | the same three-step order | the same | corrected (96f9abb) |
| `tinyapps/TINYAPPS.md`, the table of files | the same three-step order | the same | corrected (2778084) |
| `docs/FOUNDRY.md` section 6 | "Until 1.1.0 is on PyPI, use test wheels" | 1.1.0, 1.2.0 and 1.3.0 are on the index | corrected (e2319a9) |
| `docs/DATABRICKS_INSTALL.md` lines 52 and 337 (two statements) | a 5.36.0 install lists `planlens-0.16.0` | 5.36.0 requires `planlens>=0.16.0` and the index's latest is 0.17.0, so a fresh resolve takes 0.17.0 once the host's mirror has it | report only (a page of the geotechnical app) |
| `CLAUDE.md`, "Fingerprints" | the fingerprints keep one entry per planlens version | the entries are keyed by a hash of planlens' tool specifications (`packages/aec-document-review/tests/test_review_fingerprints.py` lines 79 to 80); 0.17.0 has the specifications of 0.16.0 and needed no entry | report only |
| the same | a planlens release turns CI red at once | only a release whose tool specifications changed; 0.17.0 did not | report only |
| `HANDOFF.md` line 3 (the header above the first block) | last updated for 1.2.0, "the first block of section 0a-current" | the first block is the 1.3.0 state | report only |
| `BACKLOG.md` section 0, TA-42 | the first live check of the 1.1.0 release on the testers' host | the wrapper pins 1.3.0 and 0.17.0, and HANDOFF says the testers' pins move to those | report only |
| the same, OWN-147 | before the testers get Document Review 1.1.0 | the same | report only |
| `module_work/aec_1_3_0/RELEASE_NOTE.md` line 70 | planlens 0.17.0 will need its own entry in both fingerprint files | neither file has a 0.17.0 entry; the same paragraph goes on to say none was needed | report only (a released note) |

True, but worth a line:
- `packages/aec-document-review/README.md`: "The settings in the last eleven rows are read from the environment only" is true and incomplete: the nineteen rows from `GEOTECH_SHAREPOINT_MAX_UPLOAD_MB` down are, and so are the six above them (`os.environ` in `graph_sharepoint.py`, `turn_jobs.py`, `budget_panel.py`, `identity.py`, `core.py`). Not changed: the code does not prove it false.
- The same README says "The 55 review tasks", and the release note "54 tasks with 147 checks". Both are true of what they describe: master has 55 (151 own checks) since the unreleased `coord-second-issue`; release 1.3.0, and the README on the index, have 54.

## Item 3: released and unreleased

- No document presents one of the seven pieces merged after `aec-v1.3.0` as released, and none names a version the index lacks. `CLAUDE.md`, HANDOFF's first block and BACKLOG section 0 all say "merged, not released".
- There is NO CHANGELOG file under `packages/` (`find packages -iname "CHANGELOG*"` finds nothing), so nothing says per package what is unreleased. The only place that does is HANDOFF's first block. The READMEs' per-version paragraphs stop at 1.3.0 and do not mention the seven pieces.

## Item 1: settings (only the rows with a gap)

The two packages and the wrapper name 113 settings beginning `GEOTECH_`, `AEC_` or `PLANLENS_`. No setting documented in `.env.example`, `TINYAPPS.md`, the wrapper's README, the two package READMEs or the release note's table is missing from the code (`GEOTECH_REFERENCES_PACKS` is named only to say Document Review leaves it unset). Every default those files state matches the code: helpers 2, helper calls 30, busy tries 5, busy wait 900 s, retries 1, reading timeout 90 s, slot wait 600 s, upload 2048 MB, image ceiling none, the page's budget 40. The eight settings of the release note's table are in all three host files, each with a true sentence.

Read by the page on the testers' host, and in none of `.env.example`, `TINYAPPS.md` or the wrapper's README:

| Setting | Default in the code | Where it is read | Gap |
|---|---|---|---|
| `GEOTECH_CITED_PAGES`, `GEOTECH_CITED_DOCUMENTS` | off (`record` or `retry` turns each on) | `review_flags.py`, the page's `core.py` | named in the release note ("still off by default") and nowhere an administrator looks |
| `GEOTECH_STEP_CAP_ANSWER` | off | `review_flags.py`, `turn_jobs.py` | not documented for the host |
| `GEOTECH_REVIEW_MAX_MODEL_CALLS` | 40 | `review_agent.py` | `TINYAPPS.md` states the budget (40), not the setting that changes it |
| `GEOTECH_VISION_BUSY_TRIES` | 3 | `deep/vision_engine.py` | `TINYAPPS.md` says a busy reading "is asked at most six times" and names no setting (see "not checked") |
| `GEOTECH_UPLOAD_MODE` | `http` | `ws_upload.py` | not documented for the host |
| `GEOTECH_VISION_MAX_INFLIGHT`, `GEOTECH_VISION_MAX_PER_CONVERSATION`, `GEOTECH_VISION_BUDGET`, `GEOTECH_VISION_REPEAT_CACHE`, `GEOTECH_VISION_INLINE`, `GEOTECH_VISION_INLINE_KEEP` | not read off here | the core's vision tools | not documented for the host |
| `GEOTECH_PLOT_CODE_SECONDS`, `_MAX_SECONDS`, `_MEMORY_MB`, `_AT_ONCE` | 600 s, 3600 s; the other two not read off here | `plot_box/__init__.py` | matter only where `GEOTECH_PLOT_CODE=1`; not documented for the host |
| `GEOTECH_PROMPTER_DISABLE_STREAMING`, `GEOTECH_SHAREPOINT_LOG` | off; on | `tinyapps_engine.py`, `sharepoint_log.py` | in `.env.example` only, not in `TINYAPPS.md` |
| `GEOTECH_DOTENV`, `APP_ENV` | the working folder's `.env`; `azure` where `IDENTITY_ENDPOINT` is set, else `local` | `tinyapps_settings.py` | `app.py` sets the first; `APP_ENV` is documented nowhere |

About fifty more names are read by the core for other hosts (`GEOTECH_FOUNDRY_*`, `GEOTECH_SHAREPOINT_TOKEN*`, `GEOTECH_HELPER_MODELS`, `GEOTECH_WEBAPP_MODEL` and the like) or by the suite; they are not the Tiny Apps administrator's and were not followed.

## Item 2: the set-up a new host follows

Followed on a Linux machine with Python 3.12 and no account:
- Step 1: the eight files `TINYAPPS.md` names are the eight in `tinyapps/wrapper_repo/`.
- Step 2: the three pins in `packages.txt` are on the index, agree with each other and with Document Review's `pyproject.toml` and its metadata on the index; the six third-party pins installed together with the two packages (pandas 3.0.2, streamlit 1.62.0) with no conflict. `webapp/tests/test_tinyapps_wrapper.py`: 25 passed.
- `APP_DRY_RUN=1 bash run.sh` on the bare template takes route C and prints the install and start commands; exit 0.
- `from aec_document_review.tinyapps_entry import main` imports from the wrapper folder; `aec_document_review.graph_sharepoint` (`--check`), `aec_document_review.tinyapps_keys` and `aec_agent_core.plot_box.kernel_probe` all exist as `python -m` modules.
- Every piece of page text the checklist quotes is in the code ("No model configured", "Run connection tests", "Sync now", "Permanent storage", "Prompter key source", "Signed in as", "No AI key is set up for").

Where a person would be stuck:
1. Step 2, if the company mirror does not yet hold the three pins: the text says so and gives the way out (wait a day, ask 1, or route B). Not a defect.
2. Step 3 onward needs the Prompter values, and step 7 the SharePoint values: nothing can be checked without them. The text says to stop at step 6 without the SharePoint values.
3. Nothing in steps 1 to 6 names a file, command or setting the code lacks. No stuck point was found that the documents cause.

## Item 4: the two cloud briefs

Both now say that one session at a time works in the repository, the lead, on a small usage allowance, and hands a build or a review to a cloud session; the review brief no longer says a branch is always another helper's. Every rule is kept (7bf9dd8, f15b692).

## Tests

`webapp/tests/test_public_text.py` and `webapp/tests/test_docs_currency.py`, with `--noconftest`: 12 passed, 0 failed, after the last correction and again with this file.

## Not checked

- planlens' own facts: it was not installed from source and its repository is outside this session. So: the DXF plot's bounds (16 MB, 50,000 entities, 30 s), what `plan_symbols`, `plan_grid` and `drawing_geometry` return, the planlens half of the release note, and the planlens set-up in `COMMON_DOCREVIEW.md` (`pip install -e ".[mcp]"`).
- "A busy reading is asked at most six times" (`TINYAPPS.md`): the code's default is 3 tries, and the client asks each once more; whether that makes six on every path was not traced.
- Measurements and dates: sizes and timings in the run-report section, the 4 s and 6 to 7 thousand characters of the sheet index, the release note's counts from runs, every date.
- Everything that needs the host: the identity header, Key Vault, the mirror, the Prompter, SharePoint, the published address.
- The task-count line of the release note's suite against the release commit itself (the tags are not in this clone; the release commit 9367934 was used).
- HANDOFF below its first block, BACKLOG outside section 0, the generated user manual, and the pages of the geotechnical app.
