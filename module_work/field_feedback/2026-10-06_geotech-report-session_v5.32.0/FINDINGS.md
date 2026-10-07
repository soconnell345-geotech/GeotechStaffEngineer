# Field session 2026-10-06 — geotech page: a report, its borings, DIGGS (5.32.0)

**What ran.** The owner, on the GEOTECH page of 5.32.0 (`funhouse-gpt-high` =
GPT-5.4 on Funhouse), in one conversation of 11 turns: summarise an attached
geotechnical report, extract its subsurface data as DIGGS, a liquefaction
question, then — with a SharePoint link to the full report — DIGGS again, a
"DIGGS-like package", a second pass on the lab data, and questions about why
the newest borings were missing. "Show turn details" was on.

**How it was reviewed** (CLAUDE.md, "Review every model run in full"): every
line of `activity.jsonl` (1,022 events: 268 model calls, 232 tool calls,
primary and sub-agents, every tool argument and result), `transcript.jsonl`,
`messages.json` (what is replayed to the model), `trace.jsonl`, `meta.json`,
`FEEDBACK.md`, the files the agent produced, and the owner's screenshot. The
report's own pages (129, in two parts) were checked with PyMuPDF for which
pages carry a text layer. The raw record is under `raw/` (gitignored;
Department material). Below, `L<n>` is a line of that `activity.jsonl`; page
numbers are the tools' 0-based numbers with the PDF viewer's number in
brackets. This version of the log does not record the model's own text
(added in `f405765`, after 5.32.0), so "why" is inferred from what each call
was shown and what came next. The conversation cost about 3.75 M tokens;
turn 4 alone took 656 s and 1.0 M tokens.

## 1. The conversation, turn by turn

| Turn | Asked | What the agent did (evidence) | What it claimed — and was it so |
|---|---|---|---|
| 1 | Summarise the attached report | Opened the 52-page attachment (main text of both volumes + a calculations appendix), read pages 1–41, looked at the three figure pages (L4–L61) | A sound summary. True. |
| 2 | Extract DIGGS | Re-read the text; found "boring logs in Appendix A" but only calculation sheets in the attachment (L72–L137) | "The attached PDF does not include the boring logs." True. |
| 3 | Why was liquefaction dismissed? | Searched the text (L148–L169); delegated to `references` (L172) | Grounded in the report's numbers. The sub-agent's "no matches" is P6. |
| 4 | Full report via a SharePoint link — extract DIGGS | Downloaded the 129-page, 37 MB report through the link (L204–L205, 1.0 s). Opened it (L213), searched "Appendix A" (L217), looked at page 26 — the first *scanned* page (L220) — had a helper inventory pages 26–59 and 91–128 only (L242), then extract the 2011 logs on 26–37 (L548). Hand-wrote XML (L610–L611) | "A best-effort partial DIGGS XML" — not DIGGS (P4). 2026 borings never looked at, not mentioned (P3). |
| 5 | "A DIGGS-like package" | Downloaded the report again (L626–L629), re-read 2011 logs and four lab sheets, saved six files (L740–L753) | Accurate, incl. "1981 and 2026 borings are inventory only". |
| 6 | "Second pass, bulk up the lab table" | Every SharePoint call failed within 8 s (20:26:19–27 UTC): link → "file not found" (L768–L769); folder ×3 → "(empty or missing folder)" (L772–L781); name search → "No files matching" (L784–L785); `tool_error` feedback (L788) | "I could not access the full PDF … do not have it." False — downloaded twice that afternoon and still in the working folder (P1, P2). |
| 7 | "How did you create the earlier DIGGS files without the PDF?" | No tool calls (L797–L800) | "I had access only to the original attached PDF." False confession (P2). |
| 8 | Folder and file name typed out | Folder listed and showed the file (L808, L810); exact-name search found nothing (L809, L811); download by path worked (L814–L815); reading hit the token rate limit (L863–L864) | Ended on the raw RateLimitError (P8). |
| 9 | "Can you continue?" | The same relative path that failed in turn 6 downloaded at once (L878–L879); read the 2026 lab pages and six 2011 lab sheets; saved a table and a README under turn 5's /tmp name (L942–L947) | Accurate. |
| 10 | "Confirm the 2026 borings are in DIGGS" | No tool calls | "No, I cannot confirm." Honest. |
| 11 | "They're in the PDF" | Downloaded the report a third time under a new name (L974–L975 — the screenshotted duplicate); looked at pages 21–24 [22–25] for the first time (L994–L1015) | Read the 2026 logs. "I had not actually opened … the full appendix PDF yet" is false (turns 4, 5, 8, 9); "these are scanned sheets" is wrong (vector pages). |

## 2. Problems, mechanisms and fixes

### P1. "It can't handle the links I paste" — the link was fine; a refused request was reported as "not found"
**What happened.** The address was a "copy link" to a PDF (`/:b:/r/sites/<site>/Shared%20Documents/General/<app>/uploaded%20references/<file>.pdf?d=w…&csf=1&web=1&e=…`); `browser_url_to_path` (v5.32.0 `webapp/sharepoint_tools.py:74-112`) already converted it, and it downloaded in turns 4 and 5. In turn 6 the link, three listings and a search all came back negative in 8 s, 0.11–0.28 s each (L768–L785); 16 minutes later the same calls worked (L808–L815, L878–L879).
**Mechanism.** The Funhouse SDK turns a refused request into an empty answer: `ls` returns `[]` on any non-200 (SDK reference copy `funhouse/services/sharepoint/graph/graph_file_manager.py:1661`), `search_filenames` skips a non-200 (`:2758`), `download_file` raises `FileNotFoundError("… Download failed: <status> …")` for any failure (`:1235`). The tools rendered those as facts (v5.32.0 `sharepoint_tools.py:191-192`, `:243-245` discarding the status, `:355`). No status was recorded; sub-second failures rule out throttling and the same paths working before and after rule out the path. Likeliest: an expired sign-in — the refresher slept a fixed 30 min (v5.32.0 `webapp/databricks_launcher.py:497, 541-543`) while the silent MSAL flow returns the same cached token until ~5 min before expiry. Also: the exact long, underscored file name found nothing in the search index three times.
**Fixes** (`webapp/sharepoint_tools.py`, `webapp/databricks_launcher.py`): download errors classified by HTTP status (401/403 refused, 429 throttled, else failed — never "not found"); an empty listing, a "not found" or an empty search is checked against the base folder (`_probe`) and, when that cannot be read either, reported as SharePoint not answering — an access problem — pointing at the copies already in the working folder; the SDK's error text with its status now reaches the tool result and `activity.jsonl`. Link forms: every `/:x:/r/` kind, `?id=`/`RootFolder=` addresses (folder or file preview), plain library addresses (now converted, so the Tiny Apps client gets a path), `/teams/`, another library of the site (kept absolute), the breadcrumb spelling "Documents/General/…"; token links, Office viewer links (`Doc.aspx?sourcedoc=…&file=…`) and OneDrive go to the SDK's sharing API; `link_target_name` reads the name out of any of them. When a path is not found the download tries the address itself, then finds the file by its name (named folder → search → walk of ≤40 folders, the app's own mirror last), downloads it if exactly one matches and says so in plain words; several matches are listed, not guessed. Search retries without the extension and with the longest word, then walks folders by name. The refresher checks every minute and re-mints when the interval has passed or the token is within 10 minutes of expiry.

### P2. "Must not still have had it in memory" — a forgotten file, then false confessions
**What happened.** Turn 7: "I had access only to the original attached PDF"; turn 11: "I had not actually opened … the full appendix PDF yet". Both false (downloaded and read in turns 4, 5, 8, 9).
**Mechanism.** Earlier turns are replayed as user messages and final answers only (`messages.json`, 22 entries; v5.32.0 `webapp/turn_jobs.py:235`; `harness_theory/shared_machinery.md` §3.3). No tool result — not the download, not its path — reaches a later turn; only the first turn's attachment note survives. Turn 6's false negatives then read as proof it never had the file; turn 7 made no tool call that could check.
**Fix.** Every turn starts with a note of the files the conversation already holds, in front of that turn's message only and never saved into history (`webapp/core.py` `working_files_note`, `with_turn_note`, `stream_turn(turn_note=)`; `webapp/app.py`, `webapp/turn_jobs.py`): name, size, path and origin (attached; fetched from SharePoint '<path>' in turn N; produced in turn N), from the attachments index, the transcript and a new downloads ledger (`downloads.json`). It says the model sees earlier turns only through its answers and must not call a listed file unavailable. Fresh uploads are not listed twice; missing files are not named; newest 30 kept. Recorded as `context_note` on `turn_start` in `activity.jsonl`.

### P3. The 2026 borings were missed
**What happened.** Appendix A holds three sets: the 2026 logs on pages 21–24 [22–25] are vector pages with a text layer (kind `form`); the 2011 (26–37) and 1981 (39–59) logs are scans (`scanned`); markup-only dividers on pages 20, 25 and 38 head each set. In turn 4 the agent had all of it in one result (L213: `"form":"17,19,21-24,…"`, `"scanned":"26-37,39-59,…"`, `"pages_to_view":"17-19,21-24,26-37,…"`) and found the page-20 divider (L217), but looked first at page 26 (L220) and had a helper inventory 26–59 and 91–128 only (L242). The helper reported "no pages in these requested ranges that appear to be 2026 subsurface logs" (L541); the primary did not widen the search and its answer did not say the 2026 logs were unread.
**Mechanism.** "Boring log" was equated with "scanned page"; a page's kind says how it was made, not what it is. The geotech prompt already says to look at every page in `pages_to_view` (`funhouse_agent/deep/prompt.py`, reading rules); the delegated `general-purpose` helper has none of those rules (`harness_theory/geotech_agent.md` §3.4).
**Not fixed by a prompt rule** (owner's rule). Helping: `write_diggs` (P4), the per-turn file note (P2), and the whole-report ingest, which labels every page (§4). For decision (not built): (a) a measurement task on a synthetic report with vector new logs and scanned old ones; (b) planlens: a markup-only divider page should start and title a segment (`document_structure` showed pages 20–61 as one segment titled "6"); (c) give the `general-purpose` helper the reading rules (measure first).

### P4. "It overstated its DIGGS progress" — hand-typed XML called DIGGS
**What happened.** Turn 4 saved 12,952 bytes of hand-written XML under the DIGGS 2.6 namespace as "a best-effort partial DIGGS XML" (L610–L611); against the bundled DIGGS 2.6 schema it fails at its root element. Turn 5's XML uses invented elements.
**Mechanism.** No DIGGS writer was reachable: `subsurface` only parses/validates; the real writer and its gates (`report_ingest/diggs_writer.py`) are reachable only through the report-ingest sub-agent, off by default. The agent never validated its own file.
**Fix.** `subsurface.write_diggs` (`funhouse_agent/adapters/subsurface_adapter.py`): explorations and lab results as printed, in the record format (described in the method's own description) → DIGGS 2.6 via the existing writer, schema gate and read-back gate, a plain `verdict`, and what was left out; data that does not fit writes nothing and names each field. Catalog brief and the prompt's DIGGS section point to it. A synthetic boring + two lab tests come out schema-valid, read back equal, and parse in `parse_diggs`. (`list_agents` sits 12 characters under its 8,000-character cap; the brief was kept the same length; a test catches growth.)

### P5. The report saved twice
**What happened.** Two 35.5 MB copies under two names (turn 11, L974–L975).
**Mechanism.** The reuse check was skipped whenever `refresh=true` (passed on every call) and a new `save_as` picked a new name (v5.32.0 `sharepoint_tools.py:227, 232`); a same-named file was overwritten blindly (`:236`); reuse memory was in-process only.
**Fix.** One local copy per SharePoint file: refresh re-downloads into it; `save_as` applies only to a new download; the ledger survives a restart; a different same-named file is never overwritten (`_1`); identical bytes are the same file; downloads go through a hidden part file and a download that writes nothing is a failure. (Unchanged: turn 9's README with the same /tmp name and different content became `…_README_1.md` — correct.)

### P6. The citation gap (Youd et al. 2001, ASCE 7)
**What happened.** The `references` sub-agent recorded a capability gap (L175) and said "tool searches returned no matches" (L193); its searches were `ls('/')` and four `grep`s over the empty deepagents scratch space (L179–L190). It never called a reference tool.
**Mechanism.** The scratch guard lets the scratch root through (v5.32.0 `funhouse_agent/deep/scratch_guard.py:204`), so an empty scratch search read as "the library has nothing".
**Fix.** An empty scratch `ls`/`glob`/`grep` now says it did not search the reference library, the user's documents or SharePoint, and names the tools to use.
**Owner decision.** ASCE 7 and Youd et al. (2001) are not in the library (GEC-5 cites Youd in two chapters; `liquefaction` and `subsurface.spt_correction` implement the procedures).

### P7. "Make showing turn details the default" — done (`core.tracing_enabled`; off with `GEOTECH_TRACE=0` or the sidebar box).

### P8. A rate limit ended turn 8 on a raw error — `core.friendly_turn_error` now explains the throttle, that nothing is lost, and to ask the agent to continue. Retrying inside the model client is left for a separate change.

## 3. Job 2 — "Find a past conversation" empty on the geotech page
**Cause.** The store lists each page's folder correctly; the two pages share ONE `st.session_state` (Streamlit's multipage design) and the sidebar cached the listing under one key, `sp_remote_list` (v5.32.0 `webapp/app.py:950, 958-960`; restore result `:996, 1001`). Document Review is the root (`webapp/pages_entry.py:67`), so its list was cached first and the geotech page showed it; a restore from it then looked in the geotech folder and failed. Before 5.32.0 both pages listed the same folder, so the shared key was harmless.
**Fix.** Listing and restore result cached per person and page (`sharepoint_store.listing_cache_key`); search box keyed per page. Layouts (flat `conversations/` for geotech single-user, `conversations/<owner>/` multi-user, `[<owner>/]document_review/` for the review page) are already covered by the store's tests.
**Tests.** `webapp/tests/test_past_conversations_pages.py` (AppTest): review page first, then geotech in the same session — each asks for and shows its own list; Refresh re-lists only the current page; a geotech Restore uses the geotech folder and its error stays on that page. Failed before the fix.

## 4. Report ingest on the geotech page — not wired; default stays OFF
For: exactly this session's job — every page labelled (P3 solved), a log floor under the model, bound reports split out, DIGGS 2.6 with both gates. Against, today: 60–897 model calls and 8 min–3 h per report on Foundry (`harness_theory/report_ingest.md` §7); Funhouse's token rate limit (P8); blind accuracy mixed (logs 73 %, lab 82 %, DIGGS read-back equal on 3 of 12); and not plumbed — `core.build_agent` passes no `engine` (so `engine_for` finds no Prompter) and its output folder is not the conversation's. To settle: one measured Funhouse run on a report like this, then an opt-in sidebar switch (off by default) with engine and output folder plumbed, then a default decision after use. Meanwhile `write_diggs` gives real DIGGS for data the agent reads, at no extra model cost.

## 5. Record gaps now filled
| Question the record could not answer | Now |
|---|---|
| Why did every SharePoint call fail in turn 6? | Error text and status reach the tool result and `activity.jsonl` |
| What did the model say between calls? | `model_end.text` since `f405765` |
| What was it told about its files? | `turn_start.context_note` |

## 6. Files changed and tests
Code: `webapp/sharepoint_tools.py`, `webapp/core.py`, `webapp/app.py`, `webapp/turn_jobs.py`, `webapp/activity_log.py`, `webapp/sharepoint_store.py`, `webapp/databricks_launcher.py`, `webapp/README.md`, `funhouse_agent/deep/scratch_guard.py`, `funhouse_agent/adapters/subsurface_adapter.py`, `funhouse_agent/adapters/__init__.py`, `funhouse_agent/system_prompt.py`. New tests: `webapp/tests/test_past_conversations_pages.py`, `webapp/tests/test_sharepoint_links_and_failures.py`, `webapp/tests/test_working_files_note.py`, `funhouse_agent/tests/test_subsurface_write_diggs.py`; additions to `test_databricks_launcher.py`, `test_core.py`, `test_scratch_guard_offline.py`; updated expectations in `test_sharepoint_tools.py`, `test_app_smoke.py`, `test_phase34_adapters.py`, `test_subsurface_adapter.py`. All synthetic.

## 7. For the owner to decide
1. Measure, then offer, the whole-report ingest on the geotech page (§4).
2. Add ASCE 7 and Youd et al. (2001) to the library (P6).
3. The P3 measurement task and the planlens divider-title improvement.
4. Live check after release: paste a link, go past an hour (refresher), and confirm a fetched file appears in the next turn's details (`context_note`).

## 8. Value check of the extracted data (lead, 2026-10-07)

§§1–7 judge the agent's claims and its process. This section checks the NUMBERS in the turn-5 and turn-9 files against the report's own pages:
- two 2011 borings (four scanned log sheets, BH-1 and BH-3);
- two 2011 lab sheets (one grain-size, one Atterberg);
- the 2026 lab summary table and the 2026 log pages.

Layer boundaries on the scans were measured from the drawn lines against each sheet's depth frame. That check was done in code; it was not eyeballed. Counts only below; the values and the report stay under `raw/`.

### What the agent read right

| Item | Result |
|---|---|
| **SPT records** (blows per 15 cm, N, "50 for x cm" refusals) | **48 of 48 rows exactly right** across the four sheets |
| **Log headers** (water level, total depth, drilling method, dates) | All right on both borings |
| **2011 lab sheets** | Both checked sheets exactly right: grain-size fractions, Atterberg limits, and sample depths |
| **2026 lab values** for the boring it did extract | All 18 on the summary table right. Two "clay 0" entries are not on the summary. |

On one row the sample type was written as uncertain ("D[C/G]") where the sheet clearly prints "DC", which is the cautious failure.

**Reading printed numbers off these scans was not the problem.**

### Depths read off the drawing were rounded

- **Sample depths are not printed on these logs.** The agent snapped each sample to the nearest depth label. The rows are drawn 0.15–0.35 m from those depths. The agent did mark the column "approx".
- **Layer boundaries were rounded to whole or half metres,** and are 0.1–0.7 m from the drawn lines:
  - one boring's gravel top is about 0.6–0.7 m too deep;
  - the other boring's organic layer is placed about 0.5 m deeper than drawn.
- **Adjacent, differently described layers were merged:** once across a page break, and once three layers into one.

This is the same failure as the circle task (`module_work/harness_theory/locating_things_on_a_page.md`): a position read off an image by eye rather than measured. The general capability is depth measured in code, from the drawn line and the sheet's own depth scale. That is the raster counterpart of `log_grid` and a planlens candidate. It is not a prompt rule.

### Coverage was the bigger loss

**2011 lab appendix:**

| Sheets | Extracted |
|---|---|
| Grain size | 4 of 12 |
| Atterberg limits | 6 of 11 |
| In-place density, compaction, specific gravity, soil chemistry, water chemistry | None |
| The older campaign's lab and chemistry sheets | None |

**2026 data:**
- The second 2026 boring's lab results (about 25 values) are absent entirely. They include the only fine-grained sample and the groundwater chemistry.
- The 2026 logs are clean vector pages with dates, elevation, SPT records and USCS classes. The files carry only total depth (rounded) and first water level (right).

### A source error nobody flagged

On the 2026 summary table, the fine-grained sample's plastic limit is larger than its liquid limit, and the PI equals their difference. The two rows look swapped in the report itself. The agent never reached that column, and nothing in the app would have flagged it.

The general capability is a consistency gate on Atterberg results: PL ≤ LL, and PI = LL − PL. It belongs in `report_ingest`'s lab gates and in `subsurface.write_diggs`.

### The package's own labels

The README rated the SPT values "moderate" and the stratigraphy "high". The check found the reverse.

It also points the reader to "boring logs on pages 91–102". Those pages are grain-size sheets.

It did say plainly that only selected lab sheets were read, and that the 1981 and 2026 borings were inventory only.

### For the to-do list
- Depth from a raster log's drawn scale, measured in code (planlens, beside `log_grid`).
- Atterberg consistency gate.
- Count coverage in the answer: "10 of 23 classification sheets read", not just "selected".
- Both strengthen §4's case for measuring the whole-report ingest, which labels every page and has a lab reader and a log floor.
