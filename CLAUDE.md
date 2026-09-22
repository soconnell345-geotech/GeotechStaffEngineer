# GeotechStaffEngineer

## READ-FIRST TRIGGERS (this file is 73 KB — these four lines are the ones that matter)

This file and `MEMORY.md` are the only docs auto-loaded into your context.
Everything else below is reached ONLY if you go and read it. When one of these
triggers fires, open the named file BEFORE you answer — each exists because the
answer was re-derived from scratch at least twice:

| When the owner... | Read this FIRST |
|---|---|
| pastes a `%pip install` / cluster install log, or asks "any concerns?" about one | `docs/DATABRICKS_INSTALL.md` — do NOT re-derive the numpy cascade, the seven conflict warnings, or whether a new major version is safe. They are settled and written down. |
| asks what state things are in, or you are picking up work | `HANDOFF.md` §0a-current — authoritative pickup list |
| asks about a number published in planlens docs/README | `module_work/drawing_ground_truth/doc_claims_check.py` — if a number disagrees with that script, the DOCUMENT is wrong |
| bumps the version / cuts a release | `webapp/tests/test_docs_currency.py` will fail until CLAUDE.md's state block, HANDOFF §0a-current and the install guide's history table each name the new version. Refresh the prose, don't just paste the version in. |

**Why this block is here.** After a context compaction a fresh agent keeps only
this file and `MEMORY.md`. Three releases running, an install log arrived and
the same conclusions were rebuilt from the raw log instead of read off the
guide; separately, this file told every agent "BOTH TREES UNCOMMITTED" for two
days after 5.14.0 shipped. Pointers buried deeper in a 73 KB file get skimmed —
these are at the top on purpose.

Python toolkit for LLM-based geotechnical engineering agents.
32 analysis modules (incl. pavement_design AASHTO 1993 + structural section/RC/frame analysis: section_props, concrete_props, pynite) + OpenSees agent + pyStrata agent + seismic signals agent + liquepy agent + GSTools agent + SALib agent + pystra agent + subsurface characterization (DIGGS/GEF/AGS4 data I/O — folds in the former pygef/ags4/pydiggs wrappers as format adapters) + DXF import + DXF export + PDF import + fem2d (2D plane-strain FEM: T6 quadratic elements, 3D-principal MC return, GL99 strength reduction, staged construction) + reliability (FOSM/PEM/Monte Carlo/native FORM + published COV database) + geo_project (staged, human-gated LLM model setup) + funhouse_agent (engine-agnostic agent with vision).

## What this is for (framing — use this voice in user-facing docs)

Geotechnical engineering is the *practice* of building **on and in the ground** —
foundations, retaining walls, slopes, excavations, embankments. The engineer's
material is the earth itself: heterogeneous, layered, partly saturated, and
sampled at only a handful of points across a site. Because the ground is variable
and only partly known, **a single number is never the answer**. Geotechnical
analysis is **repeated calculation across plausible subsurface and loading
conditions** — running a chained assortment of industry-standard formulas,
empirical and numerical methods, then comparing against the **design
requirements** and varying assumptions until the design is robust across the
uncertainty. The engineer's job is to understand the **range and spread** of
answers, not one point estimate; the **true answer is a distribution**.

This toolkit packages those methods as clean, machine-callable Python
(deterministic `analyze_*()` functions → dataclasses), wraps them in a
probabilistic variability engine (`reliability/` — FOSM/PEM/MC/FORM → β, P_f),
and drives them with an engine-agnostic LLM agent (`funhouse_agent/`). It follows
the lineage of every tool that amplified the engineer — slide rule → spreadsheet
/ FEM → Monte Carlo → LLM agent — **not a replacement for judgment, but a
multiplier for it.** (Plain-language framing lives in `README.md` /
`docs/overview.html`; keep that practitioner voice — "practice," "on and in the
ground," "design requirements," "subsurface and loading conditions" — in
user-facing copy.)


## Architecture Patterns

Every analysis module follows this structure:
```
module_name/
  __init__.py          # exports analyze_*() + result classes
  <domain>.py          # core computation functions
  results.py           # @dataclass with summary() -> str, to_dict() -> dict
  tests/
    test_<module>.py   # pytest suite
  DESIGN.md            # theory, sign conventions, edge cases (read when working on this module)
```

Key conventions:
- **All units SI**: meters, kPa, kN, kN/m, degrees
- **Dict-based I/O** for LLM agents: analyze_*() returns dataclass, .to_dict() for JSON
- **No cross-module imports** between analysis modules (geotech_common is the exception)
- **SoilProfile adapters** in `geotech_common/soil_profile.py` bridge SoilProfile -> module inputs
- **Foundry wrappers** (`foundry/` dir + `geotech-references/agents/`): 32 + 14 = 46 agents, 3 functions each (agent/list/describe). NOT part of the pip package, and RETIRED as a deployment route (real Foundry deployment = `webapp/foundry_entry.py` + docs/FOUNDRY.md). Deleting them is NOT quick housekeeping: a 2026-07-18 attempt found 7 agent-wrapper test suites (opensees/pystrata/gstools/salib/liquepy/seismic_signals/pystra) import `foundry.*` throughout — excise those TestFoundry sections first, then delete foundry/ + foundry_test_harness/.


## CURRENT WORKING STATE (2026-09-22) — 5.25.0 RELEASED; on master UNRELEASED: the Tiny Apps build (two pages, one app) + document OUTPUT (Word, marked-up PDFs) — candidate 5.26.0 with planlens 0.7.0

- **THE TINY APPS BUILD IS ON MASTER (2026-09-21/22), UNRELEASED — master
  `40efccb`, three commits over `v5.25.0`; planlens master `47ceaa2` over
  `v0.6.0` (its `annotate_document` needs a planlens 0.7.0 release FIRST,
  refs pattern, then the app pin `planlens>=0.7`).** Plan of record
  `tinyapps/TINYAPPS.md` (rewritten from CfA's own `exampleCode` repo and the
  User Guide, both local-only under the gitignored `tinyapps/reference/`);
  pickup list `HANDOFF.md` §0a-current. **Positioning (owner):** the
  Department already has AIP Chatbot; this app exists for VISION (looking at
  pages) and DOCUMENT CREATION (files back). **What exists:**
  (1) **Two pages, one shell** — `webapp/profiles.py` (an `AppProfile` per
  page) + `webapp/tinyapps_entry.py` (`st.navigation`; Document Review is the
  root-served default, GeotechStaffEngineer is `/geotech`; both pages run
  `webapp/app.py`). The review page builds
  `build_deep_agent(allowed_agents=(), reference_mode="off",
  system_prompt=DOCUMENT_REVIEW_PROMPT)` — a new `system_prompt` override, an
  empty scope drops the dispatch tools — hides the specialist picker and the
  geotech-only behaviour controls, and after an upload sends ONE automatic
  orientation turn as the user's own message. With nobody signed in on the
  geotech page the app is byte-for-byte the old one (AppTest-pinned).
  (2) **Tiny Apps plumbing** — `webapp/tinyapps_settings.py` (CfA's
  `get_setting`: env → Key Vault by `KV_NAME` via managed identity → local
  `.env`), `webapp/tinyapps_engine.py` (Prompter as `ChatOpenAI` from
  `PROMPTER_URL/MODEL/API_KEY/CA_BUNDLE`, key sent as `api-key` AND Bearer),
  `webapp/identity.py` (IIS `X-Windows-Auth-Header` via `st.context.headers`,
  `DEV_IDENTITY` locally, `GEOTECH_USER_EMAIL` on Databricks → folder key,
  display name, markup author), `webapp/graph_sharepoint.py` (SDK-free Graph
  file manager over an app registration; `sharepoint_store` prefers it when
  `GRAPH_*` + `SHAREPOINT_SITE_URL` are set), `GEOTECH_DEPLOYMENT=tinyapps`
  keyless like Foundry, per-user/per-page conversation roots
  (`core.register_thread_root`, looked up by thread id) and SharePoint
  folders (`conversations/<owner>/<page>/…` via `core.tag_conversation`).
  Wrapper-repo templates in `tinyapps/wrapper_repo/` follow CfA's Streamlit
  starter. (3) **Document OUTPUT** (Opus build, lead-reviewed): `write_docx`
  (Markdown → Word, `calc_package/docx_renderer.py`; NEW deps `python-docx`,
  `markdown-it-py`; hides itself where python-docx is missing) and
  `annotate_document` (planlens `document/markup_writer.py` — notes,
  highlights, boxes, callouts, replies onto a COPY, anchored by quote / box /
  point / reply_to, read back by planlens' own `markups()`; toolkit tool +
  app bridge, `<document>_marked.pdf` in the working folder, signed
  "<user> via GeotechStaffEngineer (AI draft)" — `markup_author` threads
  from `build_deep_agent`). **Gate 2026-09-22:** planlens 1,331 passed; app
  2,283 passed / 8 skipped over webapp + deep + agent + calc_package, plus a
  real `streamlit run` of the wrapper (both pages render, switch, tab titles
  follow). **Blocked on the team:** the Prompter values (URL, model, key,
  CA bundle — the deployment MUST accept image inputs), the published origin
  for `corsAllowedOrigins`, the SharePoint app registration. **Owner's next
  step:** the dosdev recipe in `TINYAPPS.md`.

- **THE REPORT-INGEST TRAIN IS PARKED at 5.25.0 (2026-09-21).** The release is
  cut — master `46ae949` is tag `v5.25.0` and the version is on PyPI — and the
  train stops there by the owner's word, with nothing half-built. **The pickup
  list is `HANDOFF.md` §0a-current**, top entry: what exists, the install and
  upload steps, the first live checks in order with their costs, the items only
  the owner can do, what is still unmeasured and the last measured numbers.
  **The next session's work is Tiny Apps:** plan of record `tinyapps/TINYAPPS.md`,
  the "TinyApps pilot" section of this file, and the Tiny Apps items in
  `HANDOFF.md`.

- **app 5.25.0 (RELEASED 2026-09-21; tag `v5.25.0`, master `46ae949`, on
  PyPI — the Nexus mirror delivers it to the cluster a day or two later)** —
  **the pages nobody was
  reading get read, a report bound inside a report becomes its own record, the
  page labels become a vote, and the reports already read become a library that
  can be asked questions.** Five builds landed on master on 2026-09-21 and this
  is the one release that carries them. **No dependency change** (the pin stays
  `planlens>=0.6`), no schema bump, seven new modules inside `report_ingest`
  (`label_vote.py`, `bound.py`, `calc_reader.py` + `calc_scoring.py`,
  `sounding_reader.py` + `sounding_scoring.py`, `library.py` +
  `library_agent.py`), one new reader in `subsurface_characterization/diggs26.py`
  and one more data file in the wheel
  (`report_ingest/library_questions.EXAMPLE.json`).

  **1. The page labels are a VOTE, and the expensive review goes only where the
  voters disagree** (`label_vote.py`). The corpus run of 2026-09-20 said the old
  single-voter path was wrong twice over: planlens' rules and a cheap vision pass
  are COMPLEMENTARY by label class rather than one being better — rules 0.908 in
  sample / **0.767 honest blind**, vision in sheet mode 0.655 / **0.867** at about
  **$0.05 a report**, the rules owning `appended_report` (494 in-sample pages
  vision never emits), `other` (216) and `calculation` (1,009, 41 % missed) while
  vision owns `plan`, `profile`, `photos`, `cover`, `toc` and `figure` recall —
  and the label review, at about **$0.45 a report** over every page, broke nearly
  as many labels as it fixed on the reports its prompt was tuned against. So three
  cheap voters (the rules, one sheet-mode vision pass on the cheap tier, and the
  printed FORM where a private fingerprint file is in force) label every page under
  one combiner that both the `vote` scoring stage and `graph.ingest_report` call,
  so a policy cannot mean one thing in the measurement and another in the run.
  `label_policy` defaults to `structural`; `rules` reproduces the old behaviour and
  calls no vision pass. `review_mode` defaults to `disagreements`: the review sees
  ONLY the split pages plus two either side, on a budget of
  `max(20, 0.5 × split pages)` instead of `max(60, 0.25 × pages)`.
  `ReportRecord.page_labels` carries, per page, the chosen label, its confidence,
  an `agreed` flag, the policy, what settled it and every voter's own label and
  confidence; a split the review did not settle is a
  `QAEntry(kind="label_disagreement")`. Two leftovers from the durable-mirror train
  came with it: `out_dir` no longer refuses a `/Workspace` path (refusing it was
  refusing the durable option; a note is printed instead), and a `durable_dir`
  whose own name already IS the run's name is not nested inside itself.

  **2. A report bound inside a report is its OWN record** (`bound.py`). Triage
  calls **19 of the 38 corpus reports `multi_document`**, one to four bound
  documents each — an earlier firm's whole investigation reproduced as an
  appendix, a bridging report bound into the design-build report that answers it.
  Those pages were listed in a QA entry and never read, and reading them into the
  SAME record would have put two firms' B-1 in one list of investigations. Each
  bound range (the UNION of triage's claim and the record's own `appended_report`
  runs, floor four pages) goes round the same loop into `out_dir/bound/<id>/` with
  its own record, summary, library page, DIGGS file and library row. One extra
  structured call reads the child's own title, firm and date. Nothing crosses the
  boundary: the parent's counts stay the parent's, and the child's narrative reader
  is windowed so the front-fifty-pages lever cannot hand it the parent's cover.
  `ingest_bound=False` restores the old listed-and-skipped behaviour.

  **3. The calculations are read** (`calc_reader.py`). A quarter of the corpus's
  4,300 hand-labelled pages are calculation printouts — **1,009 of them over eight
  of the fourteen labelled reports** — and every one was a QA entry saying the
  pages existed and had been skipped. They are the pages a reviewer most wants: a
  boring log says what the ground IS, a lab sheet what a specimen DID, a
  calculation what the engineer ASSUMED, WORKED OUT and CONCLUDED. One run of
  `calculation` pages in, one `Calculation` out: the kind from a controlled list of
  thirteen, the program and version AS PRINTED, the method, the subject, the
  labelled values it was given and the ones it worked out, a sixty-word summary,
  its pages and what it could not settle. A deterministic floor of
  (label, value, unit) pairs runs before a call is spent; the budget is two calls
  and most printouts cost one; three Python gates catch a result printed on no
  page (kept, dropped to confidence 0.3, listed), a kind outside the list and a
  unit nothing converts. The DIGGS writer ignores calculations deliberately —
  DIGGS 2.6 has no element for a method or a chosen thickness — and its docstring
  says why. **The floor alone, measured 2026-09-21 with no model and no network:
  program 80 % (8/10), inputs 73 % (64/88), results 48 % (36/75)**, over ten
  hand-truthed runs across six reports and seven kinds, 44 pages.

  **4. The test pits, the cone soundings and the dynamic probes are read**
  (`sounding_reader.py`). `InvestigationKind` has said `test_pit`, `cpt` and `dcp`
  since the first version of the record and every one of them was read as though
  it were a boring; the corpus carries **251 test pit pages, 54 DCP pages and 23
  CPT pages** in nine of the fourteen labelled reports. A pit is a LOG and the log
  reader reads it, now including its dimensions — no pit in the corpus prints a
  labelled plan size, but every one names its BUCKET, and a trench dug with a
  90 cm bucket is 90 cm wide. A SOUNDING is not a log: a tabulated sheet is read
  deterministically with the one call spent on the header, a plotted one is
  digitised through `zoom_plot` against the axis ranges read off the plot's own
  text, with the confidence falling where two traces cross. Four gates refuse a
  depth outside the sheet's own depth axis, a channel outside its printed range, a
  negative tip resistance and a series with no depth unit. DIGGS 2.6 gets
  `StaticConePenetrationTest` for a cone and `DynamicProbeTest` for a dynamic probe
  (2.6 declares no `DynamicConePenetrometerTest` at all), read back row by row with
  the round-trip gate checking every reading. The calculation GROUPING defect found
  by the calc truth was fixed here: a run splits on a changed running header or a
  page that opens with a program banner. **The floor alone, measured 2026-09-21:
  73 % overall (257/353) — 96 % on the four TABULATED sheets (217/226) and 31 % on
  the nine PLOTTED ones (40/127)**, over thirteen hand-truthed sheets across six
  reports.

  **5. The reports already read become a LIBRARY that can be asked questions**
  (`library.py`, `library_agent.py`). The ingest reads ONE report; every question
  the owner actually asks spans many, and `run_folder` had written a `reports.db`
  since the first version that nothing read back. `Library(root)` sits over the
  folder the ingest writes: the index is DERIVED and the records are the truth, so
  a folder restored from SharePoint with no database beside it rebuilds on the next
  question, keyed off each report's own front matter so it cannot acquire a second
  identity, and nothing is written back into a record. Ten query functions answer
  across reports and every row carries the report id and the PDF pages. Search is
  FTS5 the way the reference layer does it, with a rapidfuzz fallback on the field
  VALUES rather than on whole chunks. `report_library` is a `CompiledSubAgent` plus
  one primary tool, both **OFF by default** and feature-detected on the FOLDER
  rather than on a package version, bound to the app's own chat model, with the
  ceilings in Python (eight queries an answer, 25 rows and 4,000 characters a
  result) and the citations CHECKED rather than copied — a `(report, page)` the
  answer claims that no query returned is left out of `citations` and named in
  `gaps`. **Measured 2026-09-21 with no model and no network**, twenty hand-written
  questions over a synthetic library of six records: **the chosen query scores
  reports 0.949 precision / 1.000 recall and pages 0.939 / 1.000**, against
  **search alone — the question's own prose into `find()` — at reports 0.647 /
  0.943 and pages 0.417 / 0.323**. The two remaining false positives are honest
  ones.

  **WHAT IS UNMEASURED UNTIL THE CLUSTER RUNS.** Every number above that involves
  a model is missing, and this release ships without it:
  (a) **the vote in production** — the policies have been scored on saved runs,
  but `structural` as the default inside `ingest_report`, the split fraction per
  report, what the disagreement-only review touches and what it costs have not;
  (b) **the calculation reader itself** — only its floor is measured, and **there
  is no blind set**: all ten hand-truthed runs were read while the prompt was
  written, so every number the `calc` stage prints is in sample and the scorecard
  says so in place of the open/blind line;
  (c) **the sounding and pit readers themselves** — again only the floor, and
  again **no blind set**;
  (d) **the narrative levers that shipped in 5.24.0** — the front-matter union, the
  glossary conventions, per-question retrieval, the deterministic exploration
  fields and the quote gate have never been run against the tier that will do the
  work, so the 64 % recall / 74 % precision of the eight-report run still stands as
  the last real number;
  (e) **the library's model half** — the deterministic retrieval is measured, the
  sub-agent answering real questions over the real corpus is not.
  The first live checks, in order, are in `HANDOFF.md` §0a-current.

  **Gate (2026-09-21, three chunks in the foreground, each on pytest's own exit
  code): 13,108 passed / 33 skipped / 0 failed** (3,598/28 + 8,086/5 +
  1,424/0). Wheel built and checked: no corpus name, no truth file,
  no `raw/`, no tests and no `module_work` in it;
  `report_ingest/templates.json.EXAMPLE` and
  `report_ingest/library_questions.EXAMPLE.json` both present; metadata requires
  `planlens>=0.6` and does not require `anthropic`.
  **TAGGED, PUSHED AND PUBLISHED 2026-09-21** — the owner cut the release.
  Nothing in it has run on the cluster yet; the first live checks are in
  `HANDOFF.md` §0a-current.

- **app 5.24.0** (2026-09-20, on master, NOT yet tagged or released) —
  **a log knows what FORM it was printed on, and the narrative reader is
  shown the pages that answer the questions it was failing.** No dependency
  change; pure Python inside `report_ingest`, no new model call anywhere,
  and one new data file in the wheel
  (`report_ingest/templates.json.EXAMPLE`).
  **(1) The log-template recogniser** (`report_ingest/log_templates.py`).
  The owner's observation, 2026-09-20: two firms' logs are so standard that
  simple rules would almost always catch them, and one of the two templates
  has shifted over the years. The SHIPPED code is generic machinery; the
  FINGERPRINTS are DATA in a private JSON file that travels with the truth
  folder (`<truth_dir>/../templates.json`, or `templates_path=` on
  `score_on_cluster`) and is **never committed**, because this repo is
  public and a fingerprint names a firm. `recognise(doc, page, grid=None)`
  scores a page's located text against every fingerprint with rapidfuzz —
  the footer stamp worth half the score, the title block three tenths, the
  column headings two tenths, over the groups a fingerprint declares — and
  returns the family, a confidence 0-1, the margin over the best fingerprint
  of a DIFFERENT family (a sibling scoring as well is the form drifting, not
  a doubt about whose form it is) and the list of what matched. **Two
  uses:** a VOTER (`template <family> (0.92)` beside the rules' label and
  the vision label, and on the label review's page ledger via
  `annotate_ledger`) and a KEY TO THE GRID (a fingerprint's `column_map`
  names the columns `log_grid` cannot — a form that stacks the sample id,
  the sampler code, the blow record and the recovery under one heading of
  `DATA` gives the general header vocabulary nothing to classify — and
  `log_floor.seed_from_grid(..., template=match)` then reads them, keeping
  `method="grid"` because the value was still placed by geometry, with the
  template named in the note and on the investigation's own provenance).
  With no fingerprint file anywhere every entry point is a **no-op**.
  **Measured locally** (no model, no network;
  `module_work/report_ingest_harness/measure_templates.py`): **100 % recall
  and 100 % precision on both families** over the fifteen hand-truthed logs,
  none of the five logs printed on undescribed forms claimed, and **no false
  positive on thirty pages the hand says are not logs** — the threshold sits
  at 0.65, between the lowest true match at 0.97 and the one near miss at
  0.59 (a laboratory sheet from the same gINT project, carrying the title
  block's words and none of the footer).
  **And the floor now reads what those columns print.** Four generic cell
  readers went into `log_floor` in the same train, because the column map was
  worth +1 value without them: a blow record printed with `+` separators
  (`2+1+2`, gINT's own spelling), a blow record printed one increment to a
  line each in its own cell (which both of the corpus's commonest forms do),
  a cell that names its own result (`MC = 12.0%`, `LL = 38`,
  `REC=29cm, 64%`), and a sample named and typed in one cell (`S-1, SPT`).
  The grid's floor over the fifteen truthed logs goes **283/521 → 373/521
  with no template at all, and 386/521 with the fingerprints** (`n_value`
  8/49 → 48/49, `blows` 13/58 → 56/58, `recovery` 16/47 → 26/47, `index`
  13/48 → 23/48); **no log loses ground.**
  **(2) The narrative reader's first round of accuracy levers**, from the
  per-question table of the eight-report run (recall 64 %, precision 74 %),
  all default-on and each switchable. **(a) The front matter always goes
  in**: `reading_pages` is the union of every page labelled cover, letter,
  contents or narrative with the first `front_pages=50` pages of the report
  whatever they were labelled, deduplicated, in page order, capped by a
  token budget the FRONT wins — `postName`, `primeAe`, `primeContractor` and
  `projectNumber` are answered on the transmittal letter and the cover and
  neither is a "narrative" page, which is why `postName` was missed on three
  of the four reports that have one; a page in the set with under 120
  characters of text is sent as a PICTURE instead, up to eight.
  **(b) A glossary and conventions block** (`narrative_glossary.py`) held as
  DATA the owner edits rather than prose inside a prompt, carrying six rules
  **marked DRAFT and awaiting the owner's confirmation**: counts of
  explorations the report says it did not do are `0` not null; `null` means
  the report does not say; the four "mention" questions answer on whether the
  report DISCUSSES the topic anywhere; `postName` is the city of the
  diplomatic post; `primeAe` is the architect-engineer of record; and
  `earthHazardsExposed` uses the listed phrases only and NEVER includes
  seismic shaking. Each rule records the evidence it came from.
  **(c) Per-question retrieval over the WHOLE report** for the seven
  questions as often answered in an appendix table as in prose (siteClass,
  asceSevenVersion, seismicCodeUsed, soilCorrosion, bearingCapacity,
  liquefactionPotential, reportDate) — exact search first and planlens'
  fuzzy search only where exact found nothing, because a fuzzy pass costs a
  full scan per phrase; the passages travel with the brief carrying their
  page numbers, and those pages become citable. **(d) Deterministic
  answers**: pass `investigations=` and boringCount, testPitCount, cptCount,
  boringDictionary and testPitDictionary come from the logs the labeller
  found, every difference recorded as an `unresolved` row the reconciler
  raises again — but a ZERO never overrules a stated number, because no cone
  logs read is not the same as no cones pushed. **(e) A quote gate**: an
  answer none of whose citation quotes can be found on the page it cites
  (fuzzy ≥ 85) drops to confidence **0.3** and is listed, never deleted; one
  with no citation at all keeps its confidence and is flagged separately;
  `result.confidence` is the map. **(f) Scoring** gains a LENIENT view of the
  four long free-text fields (recommendedFoundations, structureList,
  bearingCapacity, strata — partial ratio 70, or a number and its unit in
  common) printed BESIDE the strict one, neither being "the" score, and the
  per-question table gains a **why** column whose `convention` value means a
  house rule rather than a reading failure. `measure_wp4_narrative.py` gains
  `--front-pages` and `--lenient`.
  **THE NARRATIVE LEVERS ARE UNMEASURED** — nothing in (2) has been run
  against the tier that will do the work, and the next cluster run is what
  says whether they move 64/74. The template numbers ARE measured, because
  that half calls no model.
  Suites: `report_ingest` **833**, harness **229**, docs-currency green.
  **Next:** the owner confirms or overrules the six DRAFT conventions
  (`REPORT_INGEST_PLAN` §7 now points at `narrative_glossary.py` as where
  the rulings go), then `stages=("narrative",)` on the cluster. 5.23.0
  follows.
- **app 5.23.0** (2026-09-20, on master, NOT yet tagged or released) —
  **the readers vote, and the whole pipeline runs on the cluster.** No
  dependency change; pure Python inside `report_ingest`, and no new model
  call per item beyond ONE optional follow-up per log. **(1) The floor**
  (`report_ingest/floor.py`, `log_floor.py`, `lab_floor.py`). The first
  full cluster run (ledger runs 6–7, gpt-5.4, 2026-09-18) showed the log
  reader LOSING to the grid on seven of ten blind logs (73 % → 62 %;
  recovery 15/15 → 3/15, index 7/16 → 0/16) and the lab reader below the
  tables on six sheets, because each re-emitted the record from the model's
  answer. Now the deterministic pass is the FIRST voter and its values are
  the floor: `log_floor.seed_from_grid` turns the grid's rows into an
  `Investigation` before any call (samples, blows, N, recovery, index
  values, layer tops with any printed USCS symbol, header fields, a water
  level off the groundwater field, each at the grid's own confidence);
  `lab_floor.floor_from_tables` reads the sheet's title (kind), the printed
  link and every labelled table value, grading series and summary row into
  typed `LabTest`s; both are shown to the model as THE STARTING RECORD, and
  the answer is merged back under one rule — the model may ADD, may CORRECT
  only with evidence (a box and a note, or a note on a picture-read value),
  and may NEVER DROP: an omitted value is kept with a QA note, a
  contradiction keeps BOTH (the floor's in the slot, the model's in
  `prov.alternatives`, or the reverse with evidence) and raises a
  `QAEntry(kind="disagreement")` with both values and both confidences. A
  difference inside the scorer's tolerance (0.15 m depth, 0.30 m layer, N
  exact, lab index 0.01, grading 1 %) is `reconciled`. `Provenance.method`
  now names the voter (`grid`, `tables`, `model`, `model_from_picture`,
  `reconciled`, plus the old ones); every value carries a confidence. The
  log reader also spends ONE follow-up call on its own unsettled list when
  the budget allows, with the rows magnified through the ruler. The scorers
  keep before/after and add `model_alone` (the model's answer scored
  without the floor) plus disagreement/kept/added/reconciled counts; the
  graph turns disagreements into QA. **(2) The `ingest` stage**
  (`stages=("ingest",)`, the seventh): `graph.ingest_report` end to end
  per report into `out_dir/ingest/<ID>/` — record, summary, library page,
  DIGGS with both gates, `qa.json`, `run.json` — reusing a saved label
  run's triage and review from `runs/` (or `review_dir`) so the review is
  never paid twice, resumable per report and per item, mirrored like every
  run file, and scoring the record against whatever hand truth `truth_dir`
  holds with the SAME scorers (the whole-pipeline score). RESULTS gains
  `# Ingest: the record and its exports`. The graph accepts an OPEN planlens
  document as `source`, so the stage hands it the corpus's DI-aware one.
  Suites: `report_ingest` **777**, harness **229**, docs-currency green.
  **Next:** run `stages=("ingest",)` with `review_dir` pointing at the saved
  label run and read the floor's effect on the blind logs. 5.22.1 follows.
- **app 5.22.1** (tag `v5.22.1`, 2026-09-21) — the same release as 5.22.0
  with one wording fix. **5.22.0 on PyPI DOES NOT IMPORT** — a docstring
  edit of the lead's landed inside a function of `report_ingest/mirror.py`
  after the builder's gate, a piped test run hid the SyntaxError, and the
  tag went out on it. Never install 5.22.0; 5.22.1 replaces it. Lesson
  recorded: never edit a file by line number, and never read a suite's
  result through a pipe that swallows the exit code.
- **app 5.22.0** (tag `v5.22.0`, 2026-09-21, BROKEN on PyPI, see above) —
  **the run's output stops living only in `/tmp`, and a disagreement between
  the voters becomes a measurement.** No dependency change; pure Python
  inside `report_ingest`, and no new model call anywhere. **(1) The durable
  mirror** (`report_ingest/mirror.py`). The first full cluster run put 38
  label reviews — about **$17 of model calls** — in `/tmp/report_ingest_520`
  and a restart wiped them; the owner's words, 2026-09-20: *"We should really
  be saving the files somewhere other than tmp. Big waste of money. You know
  we've had this issue elsewhere."* The app has mirrored conversations to
  SharePoint after every turn since 5.10.0, so this is that, for the scoring
  runs, in the shipped package and with no dependency on the app.
  `score_on_cluster(..., sharepoint=fh_sp_client,
  sharepoint_folder="GeotechStaffEngineer/report_ingest", durable_dir=None)`.
  The run's remote folder is `<sharepoint_folder>/<basename(out_dir)>` with
  the same layout inside — which is where the owner has been copying runs by
  hand. **At the START** whatever the mirror holds and `out_dir` does not is
  restored, so a wiped `/tmp` resumes instead of paying twice; **after every
  run file written by any stage** the out_dir is mirrored incrementally (a
  local `mirror_manifest.json` of (size, mtime), so the 38th mirror sends one
  file rather than 38); and again at the end for `RESULTS.md` and
  `results.json`. **Two backends behind one duck type**: a SharePoint file
  manager — the `fh_sp_client` itself is unwrapped, and so is the app's
  `SharePointStore` whose `file_manager` is a METHOD — and a plain folder
  (`durable_dir`: the owner's workspace folder under `geotech_app/`, which
  persists; never `/tmp` or DBFS — the owner's rule, 2026-09-20). Both at
  once is allowed and each keeps its own
  manifest. **A mirror failure never stops the run**: one warning line per
  distinct error and a count at the end. `out_dir` still refuses
  `/Workspace`; `durable_dir` does not, because whether a path is durable on
  this cluster is the owner's finding rather than this package's.
  **(2) The `vote` stage** (`report_ingest/vote.py`, the sixth stage). The
  owner's standing direction of 2026-09-18 — *"even if something scores
  worse, it could still be useful ... if multiple methods say different
  things, it could trigger an extra review ... would be good to have
  confidence values associated with the classifications"* — and what run 10
  actually showed: the rules and the vision pass are **complementary by label
  class**, the rules owning `appended_report` (494 in-sample pages vision
  never emits), `other` (216, the same) and `calculation` (1,009, 41 % missed)
  while vision owns `plan`, `profile`, `photos`, `cover`, `toc` and `figure`
  recall. `stages=("vote",)` calls **no model at all**: it recomputes the
  rules with planlens where the PDF is to hand and reads them off the saved
  run where it is not, takes the vision labels from `vision_dirs` (default
  this run's `vision/`) and the review from `review_dir` (default `runs/`),
  and writes `# Vote: rules, vision and the review as voters` into
  `RESULTS.md` in five parts — **(a)** the agreement rate over every page
  (the number a production run can compute with no hand labels) and the
  accuracy of agreed against disagreed pages, which is the confidence claim;
  **(b)** a per-label trust table learned on the **in-sample reports only**,
  keyed by the RULES' label because that is what a run has before it knows
  the answer, ties to vision so "believe rules" always means strictly better,
  applied to the OOS sets and never learned on the blind one; **(c)** the
  combined labels under three policies — `trust` (the table), `structural`
  (vision except the four the rules own; it learns nothing, so it is the one
  to beat) and `confidence` (planlens' own rule confidence against the vision
  pass's, ties to the rules) — each scored by the SAME scorer beside rules /
  vision / +review; **(d)** the disagreement set: how many pages per set would
  go to a targeted review and the accuracy that review would need **on those
  pages alone** to carry the set over the 0.98 gate, printed `>1.000` when
  even a perfect review cannot because the AGREED pages already carry more
  error than the gate allows; **(e)** per report, blind sets excepted. Every
  split is listed in `vote/<ID>.json` — page, both voters, both confidences,
  the hand label, each policy's choice, labels and numbers only. The label and
  vision run files now also save **`rules_confidence`**, planlens' own
  per-page confidence, so the `confidence` policy does not need the PDF.
  Harness twin: `module_work/report_ingest_harness/measure_wp6_vote.py`.
  Suites: `report_ingest` **746** (56 new), harness **229** (16 new),
  docs-currency green. **Install 5.22.1** (not 5.22.0); it supersedes
  5.21.2 and everything back to 5.20.1. **Next:** run the vote over the 38
  saved sheet-mode vision runs beside the label runs and read off which policy
  to build on. 5.21.2 follows.

- **app 5.21.2** (2026-09-20, on master) — **the `document` vision mode after
  its first cluster run.** No dependency change; pure Python inside
  `report_ingest`. **What the 5.21.1 run measured** (gpt-4.1-mini): sheet mode
  with number-only sheets is the winner and is DONE — **0.932 strict on 307
  pages** (0.947 and 0.917 on the two reports read separately) in **26 calls and ~$0.04 a report**, level
  with page mode's 0.928 at a sixth of the calls. Document mode scored
  **0.927** on the one 151-page report of text pages it finished, in 5
  windows, for **254,724 input tokens (~51,000 a call — the pages are A4, and
  A4 at 100 dpi is six 512 px tiles where a letter page is four)**; three
  pages came back unresolved because the model skipped them and the
  overlapping window skipped them too; and **every window of the 156-page
  SCANNED report failed** with `The page was not displayed because the request
  entity is too large` — the gateway's REQUEST-BODY limit, which the
  50-image probe (tiny images) never reached. **The three fixes.** (1) Every
  page picture travels as **JPEG at quality 80**
  (`vision_labels.encode_for_vision`, `VISION_JPEG_QUALITY`), at the render's
  own pixel size, so **the token count does not change** and only the bytes
  do: a scan-like page is 780 KB of PNG against 250 KB of JPEG, about a third.
  On a crisp vector text page PNG already wins and JPEG runs 0.84–1.20 of it;
  the rule is unconditional because the pages that refuse a request are the
  scanned ones, and `engine.image_block` now takes either with both engines
  reading the media type off the bytes (`image_media_type`), so
  smaller-of-the-two is a one-line change if a text-page run ever wants those
  bytes back. (2) A window the gateway still refuses **halves itself**, puts
  both halves at the front of the queue and **re-cuts every window still
  queued** to the new size, so one refusal is paid for once; a refused request
  bills nothing, so a split costs time and no money, and a window at or under
  `DOCUMENT_MIN_SPLIT` (4) that is still refused raises. (3) **`fallback=True`
  / `vision_fallback=True`**: every page still unresolved after a sheet or
  document pass gets ONE page-mode call on the same budget — a page goes
  unresolved because the REPLY left it out, and one page alone is the mode
  that cannot skip it. `cost["splits"]` and `cost["fallback_pages"]` are
  counted, printed on the progress line and carried into the RESULTS
  per-report table as a `split` column (dash when none). Suites:
  `report_ingest` 690, harness 213, docs-currency green. **Install 5.21.2**;
  it supersedes 5.21.1 and everything back to 5.20.1. **Next:** re-run
  document mode on the scanned report to see whether JPEG alone clears the
  gateway or the split has to fire, then measure `vision_detail="low"` for
  accuracy. 5.21.1 follows.

- **app 5.21.1** (tag `v5.21.1`, 2026-09-18, on master) — **the vision
  passes never see a page's rule-derived kind.** The corpus run of sheet
  mode scored 0.557 strict on 4,147 in-sample pages where page mode had
  scored 0.928 on the same two reports, with narrative precision 0.43 and
  figure precision 0.06: the contact sheets were planlens' own, which
  caption every tile "<index> <kind>" (text, figure, form, scanned) for the
  label review's benefit, and the prompt passed planlens' legend through,
  so the model read "text" as narrative and "figure" as figure. The
  document mode's strip used the same sheets. `vision_labels.render_number_sheets`
  now draws the sheets itself, captioned "p. N" and nothing else, for sheet
  and document mode; three tests, one of which forbids `render_thumbnails`
  inside any vision call. **The 2026-09-18 sheet-mode numbers are VOID**
  (ledger run 8); page mode's stand. No dependency change; **install this
  one** when it clears the mirror, it supersedes 5.20.1–5.21.0. Owner's
  standing direction the same evening: a method that scores worse can
  still be useful as a VOTE — rules, review and vision disagreeing is a
  signal to escalate, and every classification and extracted value should
  carry a confidence (`FUTURE_IDEAS`, "Disagreement as a signal").
- **app 5.21.0** (tag `v5.21.0`, 2026-09-18, on master, released the same
  evening) — **a third vision mode,
  `document`, and dollars for the Funhouse tiers.** No dependency change.
  **Why:** page mode's cluster run (307 pages) matched rules-plus-review
  overall and beat both on narrative and figure recall, but LOST on `plan`
  and `lab_test` — the two labels a reader settles by knowing which appendix
  the page sits in, which is exactly what a page seen alone cannot say. The
  owner's own framing: *"I was mainly thinking about loading the full
  context up with all pages, not going one-by-one. Because the full context
  of the report is often needed to understand what's happening."*
  **What `mode="document"` does:** each call carries a STRIP of planlens
  contact sheets covering the whole report (48 thumbnails a sheet) plus a
  WINDOW of 36 consecutive full-size pages, each stamped `p. N` in a box at
  its top-left corner with PIL — a report restarts its printed numbering in
  every appendix, so the index the pass scores by is drawn ON the picture,
  and the page size (and therefore the token count) is unchanged. From the
  second window on the call also carries the labels decided so far as RUNS
  (`61-118: boring_log`), which is what an appendix looks like written down.
  Windows overlap by 3, so a page skipped in one is answerable by the next;
  where both answer, the LATER answer wins. **The binding constraint is the
  provider's image cap, measured on the cluster this day: 50 images in one
  request, a 51st refused with "Too many images in request: 51, maximum
  allowed: 50".** Not the context window — GPT-4.1 on `funhouse-gpt-low` has
  1M tokens, which would hold a 400-page report and the image count would
  still refuse it. So the strip is capped first (12 sheets, or what leaves
  the window 12 pages) and the window takes what is left: 151 pp → 4 sheets
  + 36 pages = 40 images over 5 calls; 426 pp → 9 + 36 = 45 over 13; 729 pp
  → **12** (not 16) + 36 = 48 over 22, the long report giving up seeing all
  of itself and keeping the sheets NEAREST the window. **Also:
  `vision_detail`** ("auto"/"low"/"high") threads OpenAI's
  `image_url.detail` through all three modes and the cluster stage;
  `"low"` is ~85 tokens an image instead of a page's four tiles, which is
  ~16,000 input tokens per 100 pages against ~97,000 — untested for
  accuracy, and the next thing to run. **And dollars:**
  `engine.PROMPTER_PRICES` carries the owner's own four Funhouse rates read
  from the budget page 2026-09-18 (gpt-5.4 $2.50/$15.00, gpt-5.1
  $1.25/$10.00, gpt-4.1-mini $0.40/$1.60, ada-002 $0.10), keyed by the
  DEPLOYMENT that answered rather than by the tier asked for — a tier is an
  alias and the model behind it changes — so every RESULTS `## Cost` block
  now prints dollars beside its tokens, and a call whose deployment has no
  rate on file still adds tokens and no dollars (`CostMeter.unpriced_calls`
  says how many, so a total reads as a FLOOR). A run where nothing is priced
  prints tokens alone exactly as before. **PIL is not a new dependency, but
  the route is not what it looked like:** Pillow comes from `matplotlib>=3.8`
  and `streamlit>=1.39`, both core app requirements — planlens and PyMuPDF
  declare no Pillow at all (verified against the installed metadata).
  Suites: `report_ingest` 660 (52 new), harness 213, docs-currency green.
  **5.21.0 supersedes 5.20.1–5.20.5 and is the one to install.** 5.20.5
  follows.

- **app 5.20.5** (tag `v5.20.5`, 2026-09-18, on master; superseded by 5.21.0) — **the full
  cluster run happened** (38 reports through triage + review, 15 logs, 31
  lab sheets, 8 narratives; ledger "CLUSTER RUN 1", run 6) and it showed
  two more things. (1) The provider's per-minute rate limit on the high
  tier is shared and the SDK's client retries once with a sub-second
  pause, so 10 of 38 label reviews and 5 of 15 logs died on 429s:
  `PrompterEngine` now waits on a ladder (15/30/60/120/120/120 s, or the
  provider's retry-after) and tries again, recording each wait. (2) The
  results file printed an empty "the same set minus ," heading when no
  checkpoint report had to be excluded. Three tests. **What the numbers
  say:** label review 0.907 → 0.915 strict on 4,082 in-sample pages
  (fixed 156 / broke 122 / still wrong 88), 0.76 → 0.78 on the open
  out-of-sample pages, 0.92 → 0.92 blind — well short of the 0.98 gate,
  and R24 sits at 0.48 whatever the review does; lab reader 68 % → 87 %
  over 31 sheets (blind 55 % → 87 %; kind and link 94 %); log reader
  LOSES ground on 7 of 10 logs because it re-emits the record and drops
  values the grid already had (recovery, layer tops, index); narrative
  recall 64 % / precision 74 % over 8 reports with the misses concentrated
  in conventions (null vs 0 counts, the yes/no mention fields, hazards
  vocabulary, post name). **Next train, in this order:** a floor under
  both readers (never lose what the grid or the tables had), the
  narrative conventions (owner's call on the schemas), then the review's
  precision. **Install this one**; 5.20.1–5.20.4 are superseded.
- **app 5.20.4** (tag `v5.20.4`, 2026-09-18, on master) — `strict_schema`
  also rewrites a fixed-length tuple (the log and lab readers' four-number
  provenance bbox, which pydantic writes as `prefixItems`, tuple validation
  strict mode does not implement) into a plain array with the length in
  the description; `prefixItems`/`additionalItems` join `STRICT_UNSUPPORTED`.
  Found by the offline scan before the cluster could; one test. The owner's
  third reader failure of the day turned out to be the first shim having
  detached when the setup cell re-created `fh_prompter` — the notebook shim
  now attaches to whatever client the engine picks up. **Install this one**;
  5.20.1–5.20.3 are superseded.
- **app 5.20.3** (tag `v5.20.3`, 2026-09-18, on master) — the third and, for
  the readers, the decisive one. With the parameter shim in place the log,
  lab and narrative readers still died with **0 model calls, 0 tokens**:
  OpenAI's strict mode checks the response-format schema before the call
  and REFUSES any keyword it does not implement, and pydantic emits
  `default` for every optional field and `minItems`/`maxItems` for every
  bounded list — 63, 149 and 52 such keywords in the three readers'
  schemas, none in triage, review or vision, which is exactly the split
  between what ran and what did not. `engine.strict_schema` now strips
  `STRICT_UNSUPPORTED` and folds each dropped limit into the field's
  description; a test walks EVERY model a pass sends and fails on any
  refused keyword. `report_ingest` 604. No dependency change. **First
  reader numbers (2026-09-18, 5.20.0 + the notebook shim, gpt-5.4, two
  items a stage):** log reader 9/17 → 11/17 on two blind logs in ONE call
  each (it follows continuation pages, not its own unsettled list — a
  lever); lab reader 132/132 (kind, link, every index, series and curve) on
  an open and a blind sheet; narrative reader recall 64 % / precision 73 %
  on R05 and R06 — identity, code and summary fields essentially solved,
  the misses being the null-vs-0 convention for counts of absent things,
  the yes/no mention fields and long free text scored at partial ratio 85.
  Ledger "CLUSTER RUN 1", run 5. Full run pending; vision in page mode is
  ~5 h for the corpus, so it runs apart from the rest, in sheet mode.
- **app 5.20.2** (tag `v5.20.2`, 2026-09-18, on master) — the second thing
  the first cluster run showed. The log, lab and narrative scorers store a
  failed model call ON THE SCORE (`after.error`, `score.error`), the stage
  wrote that blob like any finished run, and the next run skipped all six
  failed items as "already done" and printed their errors as results.
  `cluster_scoring._saved_failure` now reads a saved run for a recorded
  failure and the stage RETRIES it ("previous attempt failed (...);
  retrying"); `redo=True` is no longer needed to recover from a bad call.
  Two tests. **First real numbers, 5.20.0 + shim, two reports (307 pages):**
  label review 0.902 → 0.928 strict (fixed 9, broke 1; served by gpt-5.4
  review, gpt-5.1 triage); vision-first labels (gpt-4.1-mini, one call a
  page) 0.928 — equal to rules + review, better on narrative and figure
  recall, worse on plan and lab_test recall, 2.3 s and ~2,500 input tokens a
  page. Ledger: `module_work/field_feedback/2026-09-16_report-ingest/MEASUREMENTS.md`.
- **app 5.20.1** (tag `v5.20.1`, 2026-09-18, on master) — the fix the first
  cluster run demanded. The model behind a Funhouse tier refused `max_tokens`
  (it wants `max_completion_tokens`); the SDK's `chat()` logged that, returned
  None, and the triage FAILED on the first report. `PrompterEngine` now adapts
  a refused `max_tokens` or `temperature` on the raw client the way the app's
  Databricks bridge already did, keeps the lesson for the rest of the run, and
  falls back from `chat()` to the raw client when `chat()` swallows a refusal
  (`engine.adaptations` lists what it learned). Nine tests; `report_ingest`
  600; docs-currency green; no dependency change. 5.20.0 below is otherwise
  what ships. **A release reaches the cluster a day or two after PyPI** (the
  Nexus mirror), so the 2026-09-18 scoring runs are **5.20.0 + a notebook
  shim** that does the same renaming; 5.20.0 itself installed and imported
  on the cluster that day.
- **app 5.20.0** (tag `v5.20.0`, 2026-09-17, merge of `feature/report-ingest-wp2`
  into master; planlens 0.6.0 tagged `v0.6.0` the same day, first) — published
  to PyPI by the tag workflow on the owner's word. **What shipped:**
  the whole of `report_ingest` — WP2 through WP4 on top of 5.19.0's WP0/WP1b
  library. `ReportRecord` is the product and everything else is an export of
  it: a number keeps the unit it was printed in, every value carries the page
  and the box it came off and how it was read, and what could not be read is
  recorded rather than guessed. Three readers feed it — the **log reader**
  (one `Investigation` per boring, continuation sheets folded in, gated so a
  depth off the sheet's own ruler is refused rather than accepted, and
  nothing computed: an N the log never printed stays `None`), the **lab
  reader** (a typed result per test kind, refused at construction if the
  result does not match the kind), and the **narrative reader** (the owner's
  two standing query schemas answered field for field, cited, `None` where
  the report is silent). The **reconciler** links the lab tests to the ground
  and RECORDS a disagreement instead of settling it. The **writers** produce
  the record, a summary page, a WikiLLM-style library page with a SQLite
  index over many reports, and **real DIGGS 2.6** with both gates, which the
  app's existing subsurface reader now reads back. `graph.ingest_report` is
  the deterministic resumable loop, `run_folder` drives it over a folder, and
  `subagent` puts the whole thing on the app as one `CompiledSubAgent` plus
  one primary `report_ingest` tool — **OFF by default**
  (`build_deep_agent(enable_report_ingest=False)`), because nothing about it
  has been checked on the cluster yet. `cluster_scoring.score_on_cluster` now
  takes **four stages** (`labels`, `logs`, `lab`, `narrative`) and **ONE
  truth root**: `truth_dir` may hold `logs/`, `lab/` and `narrative/`, so one
  folder is uploaded before a run instead of three paths that must each be
  right. **Pin raised to `planlens>=0.6`** (0.6.0, 2026-09-17): `log_grid`,
  which reads a boring log as the coordinate system it is, and the text-
  extraction fix that returns an overprinted line ONCE — not a nicety here,
  because a doubled depth scale ("5, 5, 10, 10") holds no strictly rising run
  of three, so the ruler was refused and the sheet came back with no depths
  at all. **No new third-party package:** planlens 0.6.0 declares exactly
  what 0.4.0 did (numpy, ezdxf, PyMuPDF, opencv-python-headless, rapidfuzz).
  `anthropic` stays OPTIONAL and is not installed on the cluster. Release
  gate **12,381 passed / 33 skipped / 0 failed** (three chunks, each gated on
  pytest's exit code, on the final tree). Cluster install **NOT yet confirmed** (install guide
  §11). **First live check:** the owner's own five-stage scoring run —
  `%pip install "geotech-staff-engineer==5.20.5"`, one
  `score_on_cluster(stages=("labels","logs","lab","narrative","vision_labels"), truth_dir=…)`
  cell with `max_reports=2`, and `RESULTS.md` comes back. **The release also
  carries a vision-first page-classification experiment**
  (`report_ingest/vision_labels.py`): each page as a PICTURE to GPT-4.1 on
  the cheap tier (`funhouse-gpt-low`) with structured output, one call per
  page or one per six-page contact sheet, added as the fifth stage
  `vision_labels` and scored against the SAME hand labels with the SAME
  scorer as the rules and the review, so `RESULTS.md` prints the three
  answers side by side. **Raised and ruled:** `report_ingest/model.py` publishes the
  owner's own answer vocabulary, and two values name a building type; it is
  generic taxonomy rather than text out of a private report, it stays
  verbatim so the hand answers still match, and the owner can object before
  the tag (HANDOFF §0a-current). Plan and parked work:
  `module_work/REPORT_INGEST_PLAN.md` (WP5). 5.19.0 follows.

- **app 5.19.0** (tag `v5.19.0`, 2026-09-17) — the report-ingest package, and
  **no new direct dependency**. **What shipped:** `report_ingest/`, a LIBRARY
  in the wheel with **nothing on the app's tool surface yet** — two model
  passes that need a whole report in view rather than one page, plus the way
  to score them where the work actually happens. `triage()` is one structured
  call over the per-page ledger, the printed outline, the front matter and the
  first contact sheet, and returns a `DocumentProfile` whose `workflow` picks
  the readers that follow; `review_labels()` is an agent loop with four tools
  that checks planlens' rule labels against the pages and returns only its
  CHANGES, which Python applies — a model re-emitting 455 page-to-label pairs
  would drop a page silently and the scorecard would score the slip. Every
  change is graded `fixed` / `broke` / `still_wrong` / `disputed`, because a
  review that raises accuracy while breaking three correct labels has not
  earned the raise. `PrompterEngine` is the engine that counts (Funhouse
  OpenAI tiers, how the app really runs); `ClaudeEngine` stays as a
  DEVELOPMENT engine for prompt iteration and its numbers are a checkpoint,
  never a result. `cluster_scoring.score_on_cluster(...)` is the owner's
  notebook cell: it reads the reports **under their own file names** from the
  folder they are already in, resolving IDs through the manifest's source-file
  column, reads Azure DI results as either `<ID>.json.gz` or the uncompressed
  `DI_data_<stem>.json` Funhouse wrote, refuses a `/Workspace` out_dir up
  front, resumes report by report, and brings back one `RESULTS.md` carrying
  IDs, labels, counts and rates and nothing that names a firm or a person.
  **Pin is now plain `planlens>=0.5`** (planlens 0.5.0 on PyPI 2026-09-17):
  0.5.0 adds `planlens.document.roles` — what each page of a report IS over
  eighteen roles with its evidence, the work items its pages make, the
  document's own printed outline and the per-page ledger these passes read —
  plus `text_reliable` / `unmapped_fraction` (a text layer that is THERE and
  wrong now becomes `needs_ocr` instead of being quoted as prose) and the fix
  for a scanned appendix reported as 69 duplicates of its first page. **No new
  third-party package:** planlens 0.5.0 declares exactly what 0.4.0 did
  (numpy, ezdxf, PyMuPDF, opencv-python-headless, rapidfuzz). `anthropic`
  stays OPTIONAL and is not installed on the cluster; `openpyxl` was already
  arriving as a hard requirement of `python-ags4` and is imported lazily
  anyway. Two pyproject lines mattered more than the code: `report_ingest*` in
  `[tool.setuptools.packages.find]` (without it the wheel builds happily with
  no package in it and fails on the cluster as an import error — verified by
  listing the built wheel) and `report_ingest` in pytest `testpaths`. Release
  gate **11,890 passed / 33 skipped / 0 failed** (three chunks). Cluster
  install **NOT yet confirmed** (install guide §11). **First live check:** the
  owner's own scoring run — `%pip install "geotech-staff-engineer==5.19.0"`,
  then one `score_on_cluster(...)` cell with `max_reports=2`, and
  `/tmp/report_ingest_wp1b/RESULTS.md` comes back. Plan and parked work:
  `module_work/REPORT_INGEST_PLAN.md` (WP2–WP5). 5.18.0 follows.

- **app 5.18.0** (tag `v5.18.0`, 2026-09-16) — two feature branches merged, no
  new direct dependency. **Charts the reader can use:** `plot_data` now draws
  an interactive Plotly twin from the SAME cleaned series as the PNG
  (`profile_figure.build_data_plot_figure`; `interactive: false` opts out),
  written as a `*.plotly.json` sidecar the app has rendered with
  `st.plotly_chart` since 5.4 but that only `subsurface.plot_*` ever produced.
  The owner asked for an interactive plot of PYWall lateral pressures and got a
  static image; the agent had picked the right tool, the generic x/y tool was
  matplotlib-only. The app now also copies a sidecar written to `/tmp` into the
  conversation folder (the Nairobi failure mode) and **hides the PNG card when
  a sidecar stands beside it**, so the reader sees one card, the interactive
  one — the CARD list only, so the SharePoint mirror, the sidebar downloads and
  `html_to_pdf` still get the image. The deep prompt's figure wording is now
  honest: a figure is a card BELOW the reply, some tools show as interactive
  charts and some as images, and a markdown link to a local path still cannot
  display. **Eighth document tool:** `find_quantities` — every number the
  document STATES with a unit, with its wording, qualifier, page and box, so a
  report's claims can be set beside what a drawing measures — plus `fuzzy` /
  `min_score` on `search_document` for text read optically or plotted as
  strokes. Both are FEATURE-DETECTED from the installed planlens' own published
  specs, because the cluster installs from PyPI: on an older planlens
  `find_quantities` is never advertised and a `fuzzy=true` search returns a
  JSON error instead of calling the toolkit. **Pin is now plain
  `planlens>=0.4`** (planlens 0.4.0 on PyPI 2026-09-16): that release moved
  `opencv-python-headless` and `rapidfuzz` into planlens' CORE dependencies and
  left `[raster]` / `[text]` as EMPTY ALIAS extras, so the extra is gone from
  the pin and **the app adds no new direct dependency** — rapidfuzz is new on
  the cluster but arrives through planlens (it cleared the Nexus probe
  2026-09-16 as 3.14.6). planlens 0.4.0 also brings, unused by this app so far:
  the scale Bluebeam/Acrobat already store in the PDF (`/VP` viewports and
  measurement markups), PDF layer names and path fill in the IR,
  duplicate-scan detection by image hash, and an MCP server over its own
  toolkit. Release gate **11,773 passed / 33 skipped / 0 failed** (run in three
  chunks). Cluster install NOT yet confirmed (install guide §11). **First live
  checks:** ask for a plot and get an interactive chart card with zoom and
  hover and NO second PNG card; `find_quantities` on a geotechnical report;
  a `fuzzy=true` search on a drawing sheet whose text was plotted as strokes.
  5.17.1 follows.

- **app 5.17.1** (tag `v5.17.1`, 2026-09-15) — one fix on 5.17.0, found by the
  suite's ordering minutes after 5.17.0 went out: the deliverable import
  compared files with `filecmp.cmp`, which caches by (size, mtime), so a
  rebuilt file of the SAME byte size written in the same clock tick read as
  unchanged and was not re-copied — the conversation would have kept showing
  the previous version, the very failure the import exists to prevent. Now a
  byte comparison (`webapp/core._same_bytes`), with the case pinned (same
  size, same timestamp, different bytes). 5.17.0 follows.

- **app 5.17.0** (tag `v5.17.0`, 2026-09-15) — the NAIROBI SOE FIX TRAIN, out
  of a real review session (ledger:
  `module_work/field_feedback/2026-09-15_nairobi-soe-rerun_v5.15.0/FINDINGS.md`,
  17 items, commit per item in its "Train record").
  **Analysis-module defects fixed:** `soe/free_earth.py` replaces the
  cantilever and braced-embedment routines. The cantilever used the FIRST
  LAYER ONLY and put the passive resultant at H + D/3 about the wall base, so
  embedment came out about a third of the correct value (0.60 m on the
  session's profile, ~9.7 m correct); `soe.embedment.compute_embedment` had
  the same class of error. Both are now layered effective-stress free earth
  support with water (Caltrans T&S Simplified Method for cantilevers, hinge
  method about the lowest support for braced walls), **pinned to Caltrans
  Example 8-1** (D 6.09 ft, D′ 4.89 ft, T 14,254 lb/ft, M 22,494 ft-lb/ft, all
  within 1 %) and cross-checked against `sheet_pile.analyze_cantilever` (D0
  within 0.5 %). Found in the same pass: the braced span above the first
  support was treated as simply supported (p·d²/8 instead of the cantilever
  p·d²/2), surcharge and water were never added to braced loads (GEC-4 5.2.4),
  and the FHWA single-anchor result stopped at the upper tributary load — it
  now returns the TOTAL anchor load, D and the wall moment (closes the V-013
  residual). **Numbers change** for cantilever embedment, braced support loads
  and braced embedment.
  **App:** files a tool writes anywhere (e.g. `/tmp`) are copied into the
  conversation folder, so they get a download card, an inline image and the
  SharePoint mirror — the owner's "where is the PDF?"; a scratch-filesystem
  guard (`funhouse_agent/deep/scratch_guard.py`) answers `read_file`/`grep`/
  `ls` on a REAL path with the real-disk tool instead of "not found" (a calc
  sub-agent had rebuilt a report with placeholders that way, losing every
  number); new `read_text_file`; SharePoint uploads default to this
  conversation's folder and path doubling is gone; reference PDFs are fetched
  from SharePoint `GSE_app/primary_references` on first use
  (`funhouse_agent/reference_docs.py` + `webapp/reference_fetch.py`);
  `record_feedback` on every sub-agent and specialist; the activity log
  attributes parallel sub-agents by run ancestry; reviewer prompts forbid
  citing what the tools did not return.
  Release gate **11,743 passed / 33 skipped / 0 failed** (run in batches).
  **No dependency changes.** Cluster install NOT yet confirmed (install guide
  §11). Prior state (5.16.0) follows.

- **app 5.16.0** (tag `v5.16.0`, 2026-09-14) — the DOCUMENT-REVIEW train:
  seven whole-document tools on the primary agent (`open_document`,
  `document_structure`, `document_page_map`, `read_document`,
  `search_document`, `document_markups`, `render_page_thumbnails`) served by
  **planlens 0.3.0** (pin `planlens[raster]>=0.3`; planlens gained
  `planlens.document` — page map with structure from the pages' own
  headers/footers/printed numbering, located text, tables, review markups,
  hidden CAD text, Azure DI as an optional text source — and
  `planlens.tools`, the framework-neutral LLM tool layer; the drawing-IR text
  leg fixed: real rotation, reviewer comments no longer read as drawing
  text). The agent is told when text is not the page (`! look:` cues) and
  looks with `analyze_pdf_page` / `render_region` / `analyze_image`. Cluster
  install NOT yet confirmed; no new third-party dependencies (planlens 0.3's
  additions are pure Python over PyMuPDF). Full record: HANDOFF §0a-current
  "DOCUMENT-REVIEW RESET". Prior state (5.15.0, 2026-09-11) follows.


**`HANDOFF.md` §0a-current is authoritative; read it before touching either
repo.** Short version:

> **This block is gate-enforced.** `webapp/tests/test_docs_currency.py` fails
> if the version in `pyproject.toml` is bumped without this section,
> `HANDOFF.md` §0a-current and `docs/DATABRICKS_INSTALL.md` §11 each naming
> the new version. It exists because this file told every agent "BOTH TREES
> UNCOMMITTED" for two days after 5.14.0 shipped — written truthfully at
> 5.13.0, then left behind when the release touched `HANDOFF.md` and not this
> one. If that test is red, refresh the prose; do not just paste the version
> in.


- **app 5.15.0** (tag `v5.15.0`, 2026-09-11) **released** — the app feedback
  train plus the ultra code-review fixes, 12 commits over 5.14.0, **no
  dependency changes**. Release gate 11,651 passed / 33 skipped / 0 failed (2026-09-11). Ledger:
  `module_work/field_feedback/2026-09-11_app-usage_v5.14.0/FINDINGS.md`.
  Cluster: `%pip install "geotech-staff-engineer==5.15.0"` — first
  cluster install NOT yet confirmed (install guide §11 row says so; same
  log as 5.14.0 expected). First live check: a calc-package question →
  figures in the PDF; `activity.jsonl` + `FEEDBACK.md` in the SharePoint
  folder; a pre-restart conversation restored from the sidebar.
- **app 5.14.0** (`47f3c8a`, tag `v5.14.0`) and **planlens 0.2.0** (`f0a2b9d`,
  tag `v0.2.0`) are both **released to PyPI and committed**, published in that
  order because the app pins `planlens[raster]>=0.2`. Release-tree gate
  **11,594 passed / 33 skipped / 0 failed**; planlens suite 804. The
  `geotech-references` pointer is at 1.4.0. Both trees are clean of release
  work — what remains untracked is the owner's scratch (plans, PDFs, a .bat,
  `bamako_agent_cell.py`) and one planlens screenshot, all deliberately never
  staged.
- **Installed and verified on the cluster** (owner confirmed the app runs).
  **Before triaging any `%pip install` log, read
  `docs/DATABRICKS_INSTALL.md`** — it is the standing answer to "any
  concerns?": what good looks like in three lines, every expected-and-benign
  warning with what would change its verdict, the real red flags, and
  copy-paste checks. Three releases running, an agent re-derived the same
  conclusions from scratch; that file exists so it stops happening.
- **Two standing traps that file records.** (1) The local gate resolves the
  **floor** of our agent-stack ranges while the cluster resolves the
  **ceiling** — 5.14.0 shipped tested on deepagents 0.6.8 against a cluster
  running 0.7.13. It holds because `build_deep_agent` checks the *compiled
  agent* for `write_todos` rather than sniffing versions. (2) Pins without an
  upper bound let a **new major version** in unannounced: OpenCV 5.0.0.93
  arrived this way and was verified safe after the fact.
- **The staging trap is now historical but the rule stands.** Six planlens
  paths (`planlens/testing/*`, `tests/test_packaging.py`,
  `tests/test_readme_claims.py`, `ir/tests/test_text_bearing_scenes.py`) had to
  enter the SAME commit or the source tree breaks; they did. If you ever split
  them again the app repo collects **0 tests**.
- A **six-round** builder+independent-verifier remediation of a 15-finding
  `/code-review` of the planlens Phase-3.2 work is complete and shipped.
  Ledger: `module_work/code_review/2026-09-06_planlens_phase32/FINDINGS.md`
  plus both agents' per-round reports. Standing lesson: the Mecklenburg corpus
  **cannot** falsify changes to terminator styles it does not contain (an 18 pt
  tolerance hides a 9 pt error), so synthetic fixtures must sit beside it —
  `round4_repro.py`, `round5_attack.py`.
- **Published numbers have a command.** Two rounds running, docs carried
  figures no run reproduced.
  `module_work/drawing_ground_truth/doc_claims_check.py` regenerates every
  corpus figure the planlens docs publish, and ten guards in
  `planlens/tests/test_readme_claims.py` pin them. If a number disagrees with
  that script, the DOCUMENT is wrong — fix the prose, never the script.

**What 5.15.0 carries (the 2026-09-11 app feedback train):** feedback capture (sidebar box + `record_feedback` tool, saved with
the conversation), the always-on `activity.jsonl` archive (every tool call /
result / model call incl. sub-agents — read THAT when a session went wrong),
the calc sub-agent's own prompt with the figure rules (it had carried the
references LIBRARIAN preamble all along), `calc_package.render_figures` +
`profile_figure.plot_data`, restore-from-SharePoint + a conversation filter,
one disclaimer widget, large-upload docs; plus the ultra code-review's five
fixes (headline: `render_figures` mode moved off the params dict onto a
contextvar — strictly-validating packages would have rejected the injected
key). Ledger:
`module_work/field_feedback/2026-09-11_app-usage_v5.14.0/FINDINGS.md`; next
train = Ensoft-style calculation TABLES (HANDOFF §0a). Owner rules for app
trains: sequential, no parallel agents. (The "planlens = todo list only"
rule was lifted by the owner on 2026-09-13 for the document-review train; a
subagent for a narrow task is fine — one at a time, never concurrent.)

**OWNER CORRECTION — the goal is DOCUMENT REVIEW, not CAD-object recognition.**
Engineers reviewing design and construction documents; an upload is usually a
geotechnical REPORT (narrative, then tables, then figures), not a sheet; and
the user must not have to declare whether they want a language review or a
visual one. Approved plan of record:
`C:/Users/socon/.claude/plans/delightful-swinging-sky.md` — vertical slice on
*"what is the average spacing of the borings in this plan?"*, PDF-first,
findings as data rather than a fixed deliverable.

**2026-09-13/14 — the document-review reset, RELEASED as 5.16.0 / planlens
0.3.0.** A high-level review found planlens had drifted into arrowhead tuning
on ten no-text-layer sheets with no document layer at all; the owner approved
a reset. planlens now carries `planlens.document` (page map + structure from
the pages' own headers/footers/printed numbering, located text, tables,
review markups, hidden CAD text, Azure DI as an optional text source) and
`planlens.tools` (the framework-neutral LLM tool layer); the app's primary
agent has seven document tools via `funhouse_agent/document_tools.py` and a
text-first-then-look policy (`! look:` cues → `analyze_pdf_page` /
`render_region` / `analyze_image`). The boring-spacing vertical slice above
remains the driving example and is still blocked on the owner's real plans.
Pickup list: HANDOFF §0a-current ("DOCUMENT-REVIEW RESET").

Green and now **committed** under that plan: `planlens/ir/measure.py` — the
`Quantity` envelope, where units are mandatory, confidence composes by `min`
rather than a product, and a page-point value REFUSES to become feet without a
resolved scale; and `planlens/ir/spatial.py` — point-pattern maths, numpy only,
with no key named `average_spacing` at any depth (conventions differ 2.45x on a
regular grid). Still blocked on the owner for real ground truth: the Langan
Subsurface Investigation Plan is in neither repo.

## Post-5.11.2 on master (UNRELEASED — the structural + drawing-intelligence trains, 2026-09-03/04)

Everything below is committed on master awaiting the next owner-gated
release (candidate 5.12.0 given the scope):

**Security/lean-and-clean (pre-TinyApps posture):** g-expression eval
AST-whitelisted (8fb7565); geophysics excision — hvsrpy + swprocess
modules DELETED outright (owner: field-geophysics processing out of
scope; sheds obspy + PyQt5(GPL)+Qt5 ~90 MB; eval suite 108→106 Qs;
openseespy KEPT — PM4Sand + 1D site response + structural future);
wheel audited clean; GUI archaeology confirmed nothing survives.

**STRUCTURAL STACK (owner-directed; Sonnet waves, lead-QC'd each):**
(1) Analysis engines e36a0ae: section_props_agent (mm-units
exception), concrete_props_agent (Mn within 0.06% of the ACI hand calc),
pynite_agent (PyNiteFEA 3.0.0; beams textbook-exact) — 32 analysis
modules now. NOTE: the first two wrapped sectionproperties /
concreteproperties until 5.13.0, when both were re-implemented natively
because they require the proxy-quarantined cytriangle (see the 5.13.0
note).
(2) Structural calc specialist 2a95e6f (webapp Agent picker "Structural
calc specialist" + .claude/agents twin; nominal-capacities/no-sign-off
language). (3) SIX public-domain reference modules in the refs
submodule (tip ead7573; refs suite 5,793): em_2104 (USACE concrete
LRFD), em_2107 (USACE steel/gates), ufc_structural (UFC 3-301-01
loads/seismic incl. the 92-system Table 3-1 replacement),
ufc_collapse + gsa_collapse (progressive collapse, DoD + GSA, with
lead-adjudicated genuine inter-document m-factor divergences noted in
both), wood_handbook (USDA FPL-GTR-282; 104 MB PDF local-only w/
download URL in manifest) — 387 functions, ~895 tests, ~10 source-
document errata caught+documented; wired into the agent 3b6ce28
(6 adapters, 407 dispatch methods, catalog 7,837/8,000 + budget-guard
test). Survey + links: module_work/STRUCTURAL_REFS_SURVEY.md. FEMA
P-2192 download still blocked (fema.gov 502s).

**FIELD-FEEDBACK TRAIN (2026-09-05, all six items FIXED same night —
ledger: module_work/field_feedback/2026-09-04_praia-downdrag_v5.11.2/
FINDINGS.md):** calc sub-agent gained working-folder READ access (the
"sources unavailable" trust fix); **profile_figure = the 33rd module**
(subsurface schematic renderer — layers/GWT/pile overlay → PNG;
general-pool, deliberately NOT in specialist scopes, see its
DESIGN.md; a rendering module like calc_package, not in the analysis
inventory table); calc packages now DEFAULT to a profile figure;
html_to_pdf auto-embeds local images and refuses un-embeddable ones
LOUDLY; **downdrag gained the CGPR #56 method family** (Endo, Poulos,
Fellenius-CGPR, PILENEG, groups + method_comparison; §3.4 worked
example fully reproduced; methods-only, no digitization — CGPR is not
public-domain); conversations-tab icons fixed for streamlit 1.62+
(icon= slot; floor now >=1.39); saved files announce themselves as
chat-attached; SharePoint mirror folders named
<title>_<date>. Bonus fix: _fileio.py verified BINARY saves always
reported corruption (CRLF-normalized comparison vs PNG signature
bytes). Catalog 7,909/8,000.

**PLANLENS SPLIT (2026-09-04, owner-named):** the entire drawing stack
now lives in the SEPARATE package repo `C:/Users/socon/OneDrive/dev/
planlens` (import as `planlens.ir` / `planlens.pdf` / `planlens.dxf`;
former drawing_ir/, pdf_import/, dxf_import/units.py — every mention of
those module paths below is historical). It is the owner's TinyApp
"banner application" for office-wide architects/engineers (funding
pitch; geotech package rides along). App depends on `planlens[raster]
>=0.1` (editable install in dev; NOT on PyPI yet — 5.12.0 CANNOT ship
until planlens is published, refs-pattern, owner-gated). The
geotech-facing `to_dxf_parse_result` bridge moved app-side to
`dxf_import/pdf_bridge.py`. Agent names "drawing_ir"/"pdf_import" in
the dispatch registry are UNCHANGED (user-facing strings, not
imports). GitHub repo + submodule wiring pending owner; NO OBO
branding in planlens by owner instruction.

**DRAWING INTELLIGENCE Phase 1 (d6bccf4; plan of record =
module_work/DRAWING_INTELLIGENCE_DESIGN.md + _TASK.md):** chatbot-first
generic find-text-X on PDF-vector drawings — drawing_ir/render.py
render_region (bbox clip + set-of-marks; top-left-PDF-pts contract),
entities_ending_near / text_anchored_geometry / find_leaders
(proposal-only, documented confidence formula); synthetic fixtures
100% recall, 100% precision@0.5, dimension-arrowhead decoys pinned
~0.3 as THE documented false-positive source. Vision glue deliberately
NOT in the agent catalog — that is Phase 2 (wiring + find_dimensions/
title-blocks/bubbles/rev-clouds + drawing-set operations + real
DWG+PDF ground-truth harvest from Mecklenburg NC / Jacksonville FL).
Process note: Fable for code builds, Sonnet for document digitization
(owner); builder + independent-verifier pattern caught a 5-meter
radius-floor unit bug pre-commit.

## TinyApps pilot (AWARDED 2026-09-03; BUILT 2026-09-21/22 — the strategic hosting path)

CfA's Azure Web Apps pilot selected the team: a shared Azure App Service
slot behind an IIS/ARR tier doing Windows auth (NO driver proxy — the
websocket saga does not apply), a free Prompter API key ($50/mo, ONE model
deployment — it must accept image inputs), packages from Nexus only, 30
days to POC, 6-month pilot, successful apps stay hosted. **Plan of record:
`tinyapps/TINYAPPS.md`** — environment facts from CfA's `exampleCode` repo
(their `settings.py` / `prompter.py` / `sharepoint.py` / identity-header
patterns, all reproduced in `webapp/tinyapps_*.py`, `webapp/identity.py`,
`webapp/graph_sharepoint.py`), the thin-wrapper repo (`tinyapps/wrapper_repo/`
= `app.py` / `packages.txt` / `run.sh` / `.env.example`; the toolkit arrives
from Nexus as the released package), the two-page architecture
(`webapp/profiles.py` + `webapp/tinyapps_entry.py`), the dosdev test recipe
and the question list. Department documents (the guide, the guidelines, the
example code) live ONLY under the gitignored `tinyapps/reference/`. The
owner has GHE + dosdev (Python 3.11.9) and the package pip-installs there;
the Prompter values, the published origin and the SharePoint app
registration are the team's to provide. Funhouse/Databricks stays the fast
tester + backup; a plain `streamlit run webapp/app.py` and the Databricks
launcher are unchanged (geotech profile, default root).

## v5.11.2 status (RELEASED 2026-09-03 to PyPI; owner word "cut the release" — websocket FINAL root cause + detached turns)

**FINAL websocket verdict (2026-09-03, closed by lab ws probes on the
exact cluster stack): the 60 s deaths were OUR OWN SERVER all along — the
5.11.1 ping fix never activated because `bootstrap.run()` NEVER APPLIES
`flag_options` (it only installs config watchers; the streamlit CLI
applies flags via `load_config_options()` first, which the launcher
bootstrap never called). Every `_FLAGS` entry has been silently ignored
since the launcher was born** — masked by the env-var backups and by
server_port coinciding with the 8501 default; corroborated by telemetry
POSTs firing despite gatherUsageStats False. Probe evidence: owner's
live-app localhost probe = PING 30.0 s → close 60.0 s code 1011; lab
repro of the bootstrap path identical; adding ONE line
(`bootstrap.load_config_options(flag_options=_FLAGS)`, commit 76e21ff)
→ flags applied, zero pings, socket open past 75 s. The interim
"hard proxy TTL" theory is RETIRED (the funhouse dev's ping-free game
app surviving long sessions was the tell). Probe tooling worth keeping:
aiohttp ws_connect(autoping=False) against `/_stcore/stream` simulates
the proxy swallowing control frames. **Plus, belt-and-braces regardless:
`webapp/turn_jobs.py` detached turn execution.** The
entire turn pipeline (stream consumption, partial checkpoints, artifact
diffing, transcript/messages persistence, meta/title, tracing, SharePoint
mirror) runs in a daemon worker thread keyed by thread_id
(process-global, one per conversation); app.py only FOLLOWS the job's
recorded events (`_follow_turn_job`) and RE-ATTACHES after every
reconnect (resume block before chat_input; recover_partial skipped while
a job is live; busy-guard on double submit; same-session sync is in-place
list mutation, new-session sync reloads transcript/messages/artifacts
from disk). A reconnect now costs a ~1 s display blink — never the
analysis. `_persist_turn` folded into the worker. Chrome-AI's 405-POST
theory in the owner's console = streamlit telemetry noise, NOT the reload
cause (fires after each reconnect, doesn't cause it). webapp suite 239.

## v5.11.1 status (RELEASED 2026-09-02 to PyPI; owner word "Yes, release" — websocket root cause + upload workarounds + groundhog removal)

**"Connecting"-flap ROOT CAUSE FOUND AND FIXED (2026-09-02, corroborated by
the funhouse dev's WS proof-of-concept):** the Databricks driver proxy
SWALLOWS WebSocket control frames (ping/pong). Streamlit 1.59's uvicorn
server defaults to a 30 s protocol ping + 30 s pong timeout
(starlette_server_config DEFAULT_WEBSOCKET_PING_INTERVAL/TIMEOUT = 30) —
pings go unanswered through the proxy, so OUR OWN SERVER hung up every
socket ~60 s after open (matches the owner's DevTools capture: each
`stream` websocket lives exactly ~1 min, then `WebSocket onclose`, forever
cycling; the dev's POC hit 1011 at ping_interval+ping_timeout ~30 s with
the raw `websockets` lib). Fix: launcher bootstrap `_FLAGS` now sets
`server_websocketPingInterval: 3600` (streamlit sets timeout=interval),
verified end-to-end vs `_get_websocket_settings()`. Plus
`_install_idle_keepalive()` in app.py — a `st.fragment(run_every=20s)`
invisible re-render generating server->browser DATA frames while idle
(`GEOTECH_IDLE_KEEPALIVE_S`, 0 disables) in case the proxy also runs a
data-idle timer; the 5.11.0 turn heartbeat covers in-turn liveness.
App-level DATA frames traverse the proxy fine — the dev's POC keepalive
finding validates the 5.11.0 heartbeat design. The dev's probe rig lives
at the owner's upload (ws_poc); its README documents proxy launch flags
and close-code diagnostics. DevTools console noise that is NOT the
problem: data.streamlit.io metrics CSP blocks + `POST .../8501/ 405`
(streamlit telemetry; gatherUsageStats already off via _FLAGS).

**Upload-403 workarounds (same train, 2026-09-02):** (1) **websocket
uploader** — `webapp/ws_upload.py` + `webapp/ws_upload_component/index.html`
(hand-rolled component protocol, no build toolchain; wheel ships the HTML
via package-data): file bytes ride the app websocket as a base64 component
value, so the driver proxy's suspected PUT-method block never applies;
25 MB/file cap; `GEOTECH_UPLOAD_MODE` env (http default; launcher bootstrap
sets ws on Databricks — override =http to A/B); app.py falls back to the
native uploader on component errors. Evidence basis: owner's DevTools
showed `POST .../8501/ 405` — POST bodies reach Streamlit through the
proxy, so the wholesale-write-block theory is dead and PUT (what
st.file_uploader uses) is the prime suspect. (2) **diagnostics upload
probe** — `_upload_probe_check` in webapp/diagnostics.py PUT/POSTs the
localhost `_stcore/upload_file` endpoint (no proxy in path) and prints the
verdict in the sidebar Connection-diagnostics panel: non-403 locally =
proxy is the blocker; 403 locally = Streamlit-side XSRF/origin. NOTE:
stale `build/` dir was contaminating local wheel checks (groundhog_agent.py
ghost) — deleted; CI builds clean.

## (same train — groundhog removal + native correlations)

**groundhog dependency DELETED (owner-directed, 2026-09-02):** DT's Nexus
security sweep (end of Sept) removes groundhog 0.15.0 — its ONLY release —
flagged HIGH (bundled-jQuery CVEs), so `pip install geotech-staff-engineer`
would fail on the cluster once it's gone. Capability audit (90 wrapped
methods vs our 31+27 modules): 7 of 11 categories already covered; the real
gaps were RE-IMPLEMENTED NATIVELY from the original published sources
(groundhog is GPL-3 — no code copied; its outputs used only as numerical
test oracles, pinned before deletion):
- `geotech_common.soil_properties`: `spt_energy_correction` /
  `spt_overburden_correction` / `spt_n1_60` (Youd et al. 2001 Table 2 +
  Liao-Whitman/ISO C_N — the repo previously CONSUMED N1_60 everywhere but
  nothing produced it) + `stress_dilatancy_bolton` (Bolton 1986).
- `seismic_geotech.dynamic_properties`: Gmax family (Vs, Rix-Stokoe CPT
  sand, Mayne-Rix CPT clay, Hardin-Black, Andersen 2015) + Ishibashi &
  Zhang (1993) G/Gmax + damping. First Gmax functions in the repo.
- `fem2d.hs_correlations.estimate_hs_parameters_sand` (Brinkgreve et al.
  2010) — feeds the HS model that previously had no parameter estimator.
- Agent surface: subsurface `spt_correction`/`stress_dilatancy`, seismic
  `gmax`/`modulus_reduction`, fem2d `fem2d_hs_parameters`.
Deleted: groundhog_agent.py (+test), foundry/groundhog_agent_foundry.py,
SoilProfile.to_groundhog_profile()/to_logplot(); tier3 crosschecks
repointed native; `[groundhog]` extra now an empty alias; py-modules line
dropped. Deliberately NOT replaced: API RP2GEO offshore pile/bearing,
Alm & Hamre driving fatigue (offshore, out of scope), pumping-test k,
LogPlot boring-log plots. Coverage map in the 2026-09-02 session; DT
package-violation list also noted pandas-2.x critical-flag watch item
(pandas 3 jump risk vs streamlit pin — drift canary covers) and matplotlib
"License-Threat Not Assigned" (scanner artifact, owner to confirm w/ DT).

## v5.11.0 status (RELEASED 2026-09-01 to PyPI; owner OK'd — the connection-stability + capability train)

All four parked branches merged + the live-evidence fixes, packed per owner
("mirror takes a day or two — pack more in"). **Connection stability** (the
"Connecting"-flap fix, two layers): (1) `core.with_heartbeat` wraps
stream_turn — worker thread + queue; >15s silences (GEOTECH_HEARTBEAT_S)
emit heartbeat items the app renders as status-label updates = websocket
traffic through the driver proxy (covers silent tool phases AND model
thinking); ALWAYS on. (2) PrompterChatModel `_stream` (token streaming) —
clean unwrapped OpenAI client over the NTLM transport (the SDK wrapper
injects collect_usage=True on stream=True → TypeError), meter_log re-added
(Terms), tool-call delta chunks, default OFF via `streaming_enabled` /
`GEOTECH_PROMPTER_STREAMING=1`, auto-fallback to _generate. **Model
picker**: registered builders that accept a model id get the picker
selection; `GEOTECH_PROMPTER_MODELS` populates the sidebar;
run_on_databricks publishes [launch model, funhouse-gpt-medium] by default.
**Budget/email**: sidebar "AI budget: $X of $Y" (once/session + refresh,
warn ≥90%), BudgetExceeded friendly error, `email_file` agent tool
(.gov/.mil/.sbu guard, to="me" resolves the session user email captured at
launch into GEOTECH_USER_EMAIL). **SharePoint**: fix_web_url repairs the
SDK's single-slash redaction-dodging URLs (dead sidebar link); list-vs-
search nudge (search index lags ~5 min). **Das ergonomics**: Schmertmann
peak-kink-exact integrator + adapter gamma_soil; sheet-pile D bracket+1mm
bisection (kills ~3% grid quantization + the 0.5 m floor);
validation_examples ADDED TO GATE testpaths (a stale V-006 pin of
pre-5.9.1 beta had hidden there). **Launcher**: adb-dp- proxy-host rewrite
(*.databricks.azure.us), port=None → free-port scan from 8501,
auto_shutdown_min=, openai<3 cap (SDK pins openai==2.28; 3.x brings the
httpx2 fork). **Version guard** (webapp/version_guard.py): runtime
agent-stack drift warnings in sidebar + diagnostics, lockstep-tested
against the pyproject caps; weekly cloud drift canary armed (Mondays).
Env notes: cluster idle timeout is 30 min (documented — the "random
detach" ghost); Plotly static export broken in tenant. Funhouse SDK
reference tree (source+23 example topics+docs) at
Funhouse_for_Reference/funhouse-sdk-python; survey ledger in
FUTURE_IDEAS. Tiny Apps = sanctioned durable tier (supports Streamlit),
owner pursuing via Funhouse office hours.

## v5.10.2 status (RELEASED 2026-08-03 to PyPI; owner OK'd — deepagents-drift hotfix)

**Every-question GraphRecursionError on the cluster (2026-08), root-caused
and fixed:** unpinned drift resolved deepagents 0.7.11, which no longer
auto-attaches the todo/planning middleware — while our domain prompt still
tells the model to plan with `write_todos`, so the model looped calling a
nonexistent tool until the recursion cap (25 AND 75), on even trivial
questions. `build_deep_agent` now checks the COMPILED agent and rebuilds
once with `TodoListMiddleware` (from `deepagents.middleware` or its new
`langchain.agents.middleware` home) when `write_todos` is missing —
verified green on BOTH stacks (0.6.8/1.3.4 baseline and
0.7.11/1.3.18/streamlit-1.62 drifted). Plus: agent-stack UPPER BOUNDS in
pyproject (deepagents<0.8, langchain<1.4, langgraph<1.3 — raise only after
running deep+webapp suites on the newer version); recursion_limit default
25→50 (SharePoint+PDF turns are step-hungry); `friendly_turn_error` (raw
GraphRecursionError + plain-language sidebar advice). Reproduce-drift
recipe: pip install the cluster's versions, `pytest funhouse_agent/deep/tests
webapp/tests`. NOTE: deepagents 0.7.11 also adds ls/glob/grep/delete/execute
tools; `execute` fails safely on our StateBackend (not a sandbox backend).

## v5.10.1 status (RELEASED 2026-07-31 to PyPI; owner OK'd — Databricks stability + agent SharePoint tools)

**The detach root cause, fixed:** pip-installing the app upgraded the notebook
env's Pygments (2.15.1→2.20.0) via `pydiggs → myst-parser → sphinx`;
Databricks flags UNSUPPORTED_IPYTHON_DEPENDENCY_VERSION and SIGTERMs (143)
the REPL minutes later — presenting as "random detaches" / Py4J "Object ID
unknown" all week. pydiggs demoted to the `[pydiggs]` extra (DIGGS validation
degrades gracefully via has_pydiggs; native DIGGS parser unaffected). Also:
**agent-facing SharePoint tools** (`webapp/sharepoint_tools.py` — list/
search/download-to-working-folder/upload; injected via new
`build_deep_agent(extra_tools=…)` ONLY when SharePoint is configured; zero
catalog cost); launcher hardening (app log at /tmp/geotech_webapp_<port>.log
+ handle.log_path, POSIX setsid, streamlit CORS/XSRF env force-off, 502-boot
banner note). Live-verified on the owner's cluster 2026-07-31: mirror
uploaded a conversation to GSE_app ✅. Still open: browser upload 403 (PUT
probe pending on a stable kernel; SharePoint-fetch is the fallback attach
path), Prompter model picker TODO (FUTURE_IDEAS §6).

## v5.10.0 status (RELEASED 2026-07-30 to PyPI; owner OK'd — the Funhouse/Databricks train)

Owner pivot: their Palantir team retired the Foundry option — **Databricks/
Funhouse is THE deployment target.** (1) **SharePoint permanent storage**
(`webapp/sharepoint_store.py`): after every turn the conversation dir (meta/
transcript/trace/uploads/artifacts) mirrors incrementally (local manifest) to
`<ROOT>/conversations/<thread_id>/`; sidebar "Permanent storage" block +
Sync now; best-effort, never breaks a turn. Owner target: site
CSEGeotechGroup, root "Shared Documents/General/GSE_app". Auth =
`stage_sharepoint(site, root)` in `databricks_launcher` (delegated-OAuth:
writes the Graph token to a driver-local file + 30-min silent-refresh daemon
via MSALAuth secret_manager cache; store reads via
create_sharepoint_client_from_token_provider). NO client_id/secret in this
org; the SDK's default graph backend is device-code INTERACTIVE — never use
it in the app process. (2) **Databricks launcher**:
`run_on_databricks(prompter=fh_prompter)` threads the Prompter's NTLM creds
(plain strings on the live object) via GEOTECH_FH_* envs — bare
`PrompterAPI()` self-config dies with Py4J "Object ID unknown"
(live-verified). (3) **In-app trace toggle** (Behavior > "Show turn
details"; behavior["trace"] tri-state, None = GEOTECH_TRACE env). (4)
**axial_pile beta fix**: per-layer friction_angle on a cohesive layer WINS
over global cohesive_phi (Das-sweep +12.4% trap); global stays the fallback.
(5) st.iframe deprecation fix; Foundry-side work (palantir_sdk_engine,
foundry_app_launcher, "Model RID or API name") ships dormant/parked. Gate
10,063 passed / 48 skipped. 6.0 restructure still PARKED
(module_work/V6.0_RESTRUCTURE_PLAN.md — dedicated session).

## v5.9.1 status (RELEASED 2026-07-20 to PyPI; owner OK'd — defect fixes + figure retrieval + Foundry mode)

The sample-calc-as-defect-detector train (doctrine: FUTURE_IDEAS.md header;
ledger: module_work/wiki_verification/TIER_A_LEDGER.md). (1) **Two more REAL
defect fixes**: drilled_shaft depth-beta unit-mixing (metric 0.245 coeff was
applied to feet-converted depth — 1.81x-too-fast decay, -64% skin on NHI Ex
9-5; GEC-10 rational path unaffected) and axial_pile Nordlund/Meyerhof qL
re-digitized from printed GEC-12 Fig 7-15 (old table ~10x UNCONSERVATIVE at
phi=30, wrong 40-deg cap; now the page-QC'd refs digitization, chart visually
re-confirmed). (2) **view_worked_example_source** vision tool — renders a
worked example's printed source page (220 dpi, same docs-folder resolution as
read_reference_figure); FHWA NHI-06-088/089 Soils & Foundations manuals
onboarded (refs docs/, submodule 669bae0) with 10 execution-verified entries
(corpus 27, 21 with page retrieval). (3) **Foundry deployment mode** — the
published app never reads or mentions ANTHROPIC_API_KEY (enclave-IT ask):
RID-only model surface, promoted "Model RID" input, key-free diagnostics;
local/dev unchanged (webapp/tests/test_foundry_mode.py). (4) Das 6e solutions
sweep (internal-only): 8/8 reconciled <=0.3% — clean bill; six-item ergonomics
backlog recorded (top: beta global cohesive_phi trap). Handoff pickup list:
HANDOFF.md §0a.

## v5.9.0 status (RELEASED 2026-07-19 to PyPI; owner OK'd — worked examples + wiki verification)

Two owner-driven trains. (1) **worked_examples** (32nd module):
`funhouse_agent/worked_examples.json` — 17 validated calculations from real
published design reports (GEC-12/10/11/6, Caltrans, GEC-4, GEC-13, Slide2
benchmarks incl. Pilarcitos, FLAC, AASHTO/UFC pavements), each with a
self-contained problem, runnable dispatch calls, published-vs-computed results
and report-writing notes; adapter `find_worked_examples`/`get_worked_example`;
prompt nudge; every entry verified-by-execution in the gate. (2) **WikiLLM
verification campaign** (owner's 7,300-record library index over ~8k PDFs used
to verify module content vs the provenance-audit wishlist; ledger
`module_work/wiki_verification/TIER_A_LEDGER.md`): 2 REAL defects fixed
(lateral_pile Reese sand A/B cyclic tables were reconstructions with a wrong
asymptote — re-digitized from COM624P Figs 3.12/3.13; rapid_drawdown Kf now
implements printed EM Eq. G-8 for c'>0), Duncan-2000 citation family corrected
(Table 1→3; fabricated "Kulhawy & Trautmann" attribution), 6 zero-discrepancy
primary confirmations (B&T incl. rigid a0, Bishop/Spencer/M-P/App-G kernels,
AASHTO site factors 75/75, ACI Ec, ASTM A615 11/11, Das aniso-su), V-052b
Pilarcitos on the TRUE 2:1/3:1 section, and a to-the-pound Whitman M-O anchor
(oldest audit gap closed). Also: webapp/tests joined the gate; FOUNDRY.md
troubleshooting field guide; FUTURE_IDEAS.md; HANDOFF delta. Foundry app
remains blocked on the enclave LLM-proxy 401 (admin ticket pending).

## v5.8.2 status (RELEASED 2026-07-17 to PyPI; owner OK'd — Foundry diagnostics)

Owner's live Foundry debugging (GPT-5.1 RID → "(no answer text)", error
flashing one render): sidebar **Connection diagnostics** panel (staged
resolve/plain/stream/tool-call self-tests, failures printed verbatim,
webapp/diagnostics.py is streamlit-free and terminal-callable); turn errors
persist on transcript replay; empty answers get an actionable note; Foundry
OpenAI path sends **max_completion_tokens** (GPT-5/o-series reject max_tokens,
RID names give langchain-openai no hint) + `GEOTECH_FOUNDRY_DISABLE_STREAMING=1`
fallback; version+engine caption. Model switches ride on_change only (stale
proxy echoes can't revert). Foundry ops: workspace "Restart" beats pkill;
app view ≠ Jupyter container (files invisible across); engine via
GEOTECH_FOUNDRY_MODELS env line in the owner's app.py.

## v5.8.1 status (RELEASED 2026-07-17 to PyPI; owner OK'd — Foundry-deployment fixes)

Patch driven by the owner's first live Foundry publish (same night as 5.8.0):
(1) custom-model-RID clobber FIXED — the keyed Model selectbox's sticky state
reverted programmatic model changes (custom RID box AND conversation resume)
within one render; `_model_dirty` flag now syncs the widget pre-instantiation
(AppTest regression tests in webapp/tests/test_custom_model_rid.py). (2)
Extras consolidation (owner request): plain `pip install geotech-staff-engineer`
now brings the WHOLE stack (deep agent, webapp, all backends, PDF, both LLM
clients incl. langchain-openai); the 23 old extra names remain as empty
aliases. That still holds in 5.13.0: sectionproperties and
concreteproperties were REMOVED rather than made optional (cytriangle is
quarantined by the corporate proxy and no waiver was available), so there is
nothing left to opt into. (3) `websockets>=14,<16` pin — Foundry's Streamlit base image ships
websockets 16.x, which leaked into the app lockfile and made the publish-time
reinstall unsatisfiable vs langgraph-sdk (<16). Also: eval suite now 108 Q /
68 keyed (PAV-1..PAV-8, ground truth run on v5.8.0), eval_harness `--ids`
prefix filter, description says 31 modules. Foundry ops notes: published-app
"failed to run startup scripts" = read the env-restore log (the websockets
conflict presented there); preview needs `streamlit run <file> --server.port
8501` in the terminal first. Single-namespace package restructure (kill the 35
top-level modules) assessed and PARKED as the 6.0.0 candidate.

## v5.8.0 status (RELEASED 2026-07-17 to PyPI; owner OK'd — UFC alternative method + pavement specialist)

UFC 3-250-01 (2016) roads/parking design as a full alternative method in
`pavement_design/ufc.py`: `design_flexible_pavement_ufc` (Corps CBR / Figure E-1
cover cascade + Table 7-2 minimums + Ch 19 reduced-subgrade-strength frost),
`design_rigid_pavement_ufc` (Figure F-1 Westergaard-based + Eq 13-1 stabilized
foundation), `compare_flexible_pavement_methods` (AASHTO vs UFC side by side,
FHWA Mr=2555·CBR^0.64 bridge, caveats echoed), `ufc_mixed_traffic` (Table G-1
controlling-vehicle equivalent-18-kip passes; light vehicles below controlling
thickness → unlimited passes; reproduces printed G-1 within 6%). Plots module
(nomograph-overlay design charts, layer sections, UFC charts, method
comparison); calc-package design_type flexible_ufc/rigid_ufc/compare; pavement
specialist agent (app picker + reviewer + Claude Code twin). F-1 stands at
exact-at-anchors ±10% — no printed equation exists; N-densification provably
fails (documented in refs docstring). Ships with **geotech-references 1.3.3**
(pin >=1.3.3): ufc_pavement REBUILT from the real UFC 3-250-01 (closing the old
airfield-source audit gap) + all 30 Appendix E vehicle curves + 3 companion
practice modules (refs now 27 modules). Airfield (UFC 3-260-02) and rigid
vehicle curves F-2..F-31 PARKED.

## v5.7.0 status (RELEASED 2026-07-16 to PyPI; owner OK'd — pavement design)

New `pavement_design/` analysis module (31st): complete AASHTO 1993 flexible
(Fig 3.1 SN + Fig 3.2 layer split + §3.1.4 minimums + forward check) and rigid
(Fig 3.7; k = direct | MR/19.4 | full §3.2 composite-k worksheet, LS-corrected,
iterated with D) design, ESAL traffic (full Appendix D LEF tables incl. triple
axles), Appendix G swelling/frost-heave serviceability loss (printed G.4/G.8
equations) + Table 3.1 performance-period iteration, calc-package template +
`pavement_design_package`, adapter with 5 methods. US customary (documented
exception). Validation V-055 (guide worked examples SN 5.0 / D 10.0 / MR 5000
end-to-end); DESIGN.md carries the full ledger + chart-read tolerances. Ships
with **geotech-references 1.3.2** (pin >=1.3.2): EC7-1/EC7-2/AASHTO-1993 +
lef.py (~5,850 cells, lead visual QC) + composite_k.py (read-grid digitization)
+ environmental.py. The 1993 Guide is a SINGLE volume — the D-2 "Appendix MM of
Volume 2" citation is stale 1986 text (owner-verified). Overlays (Part III)
PARKED. list_agents catalog at 7,995/8,000 chars — rebalance briefs before
adding the next module.

## v5.6.0 status (RELEASED 2026-07-15 to PyPI; owner OK'd — the app train)

Owner pivot to app-heavy work (`webapp/`); wiki integration PARKED. All A-items
A1–A8 built: inline Plotly, calc sub-agent context isolation
(default ON, −84% measured), crash-proof turn persistence, durable files/links,
Agent + Analysis-depth + model pickers, per-conversation working folder
(`GEOTECH_DEFAULT_OUTPUT_DIR`), optional tracing (`GEOTECH_TRACE=1` / LangSmith
envs), industry memo. Owner-session fixes: bounded auto-continue for
mid-turn stops, download MIME types, recursion-cap visibility. Also in 5.6.0:
generic `calc_package.html_to_pdf` (verified HTML→PDF for reports no canned
package covers) + calc-agent REAL-DISK-ONLY framing; retaining_walls
`delta_base`/`base_adhesion` direct base-interface overrides + `sliding_basis`
echo (double-2/3 trap resolved — verdict in retaining_walls/DESIGN.md) +
widened `retaining_wall_package` params; **Palantir Foundry deployment path**
(`docs/FOUNDRY.md`: Code-Workspaces publish, LLM-proxy engine with `ri.…` RID
auto-routing OpenAI/Anthropic, `GEOTECH_FOUNDRY_MODELS`, in-app custom-RID box,
`webapp/foundry_entry.py` 2-line stub). Owner decisions: summarization backstop
SKIPPED, durable checkpointer PARKED, API thinking layer DEFERRED, `foundry/`
AIP-wrapper route RETIRED. Plans: `module_work/APP_PLAN.md`,
`module_work/FOUNDRY_APP_PLAN.md`. Webapp tests: 115 (`pytest webapp/tests -q`).

## v5.5.0 status (RELEASED 2026-07-12 to PyPI; owner OK'd)

The post-5.4.1 train, merged and released; close-out gate **8539 passed /
48 skipped**. Adds: correlated scalar-pair probabilistic variables
(`correlations=[(k1,k2,rho)]` on fosm_fos/monte_carlo_fos, V-045; overlapping
pairs rejected), Bray & Travasarou (2007) seismic displacement
(`bray_travasarou_2007`, paper anchors reproduced + hand-verified),
Lowe-Karafiath rapid drawdown (`method='lowe_karafiath'`, V-048/052/053;
"Corps #1/#2" = interslice-function menu-name collision, intentionally NOT a
drawdown method), anisotropic undrained strength (`strength_model='anisotropic'`,
su_active/su_dss/su_passive, V-054, exact isotropic identity), the one-call
`slope_report_package` (search story + rejection diagnostics + method table +
thrust line + FOSM/MC annex; hardened: param rejection, renderer surfaced,
Story PDFs ~1 MB, wide tables fit, thrust-line spike fixed source+display),
and native inline Plotly in `webapp/` (.plotly.json sidecar). Lead close-out
QC ledger: `module_work/V5.4_PLAN.md` §"Close-out QC" (7 findings, all fixed).
Deferred: toe-circle search under-sampling; steep-φ' Kc sensitivity.
5.4.1 (2026-07-10) carried the file round-trip (read_pdf_text, list_files,
verified saves, plot output_path, Attach widget, PDF fallback) + `webapp/`.

## v5.4.0 status (RELEASED 2026-07-08 to PyPI; owner OK'd pre-live-eval)

Merged to master and released; release gate **8436 passed / 48 skipped**.
On top of the six owner directives (below), 5.4.0 carries the FULL E1–E11 QC
backlog (rapid-drawdown search + validated #98, stage-3 'gle' option — default
STAYS 'fellenius' per owner, pore-pressure TIN grids, exit-side/truncation
tension cracks — V-026 <0.1%, composite pile EI — V-017 PASS, pile passive
convention, SRM mesh-refinement study, multi-surcharge zones, perf cleanups,
eval suite 71→**100** keyed questions, cross-module did-you-mean redirects,
fem2d `analyze_footing_capacity` — Prandtl −0.4%, ufc_expansive figures
complete in geotech-references ≥1.3.1) and F1/F2/F5/F8 (correlated `linear_su`
+ `ru`/`gamma_sat` probabilistic variables — Duncan Pf anchor, V-042/043/044,
search-admissibility guard — zero-shift proven, local-FOS heatmap + gallery
Exhibit 13, reviewer family: seismic/foundations/earth-retention/slope-fem).
DEFERRED to next train: correlated c'-φ' scalar-pair + Slide2 #33/#34, F3
Bray-Travasarou, F4 CoE/Lowe-Karafiath, F6 anisotropic su, F7 slope
calc-package. Owner-gated: live 100-Q eval run.

All six owner directives DONE (2026-07-07):
**D1** 132-pp PDF user manual (`docs/GeotechStaffEngineer_User_Manual_v5.3.pdf`,
regenerable builder, catalog auto-generated from METHOD_INFO); **D2** Databricks
no-restart flow (`funhouse_agent/runtime_check.py` hot-reloads a stale
`typing_extensions` before langchain imports); **D3** layered disclaimers
(DISCLAIMER.md ships in the wheel, PyPI README section, one-time first-import
notice w/ `GEOTECH_NO_DISCLAIMER=1`, `geotech-disclaimer` script, calc-package
basis blocks — pip runs no code on wheel install, these are the equivalents);
**D4** 12-exhibit visualization gallery (`docs/gallery/`, real validated runs);
**D5** `drawing_ir/` module (LLM-ready drawing IR: exact-coordinate entities w/
provenance + confidence, DXF / PDF-vector / raster-opencv legs, agent query
surface via handles; `[raster]` extra; geo_project wiring = flagged follow-up);
**D6** seismic reviewer (`.claude/agents/seismic-reviewer.md` +
`funhouse_agent.make_seismic_reviewer(engine)` / `make_seismic_reviewer_deep`;
shared checklist `funhouse_agent/review_checklists.py` — reviewer-family
template). Open backlog: `module_work/V5.4_PLAN.md` E1–E11 (QC carryovers) +
F1–F8 (creative proposals). **HANDOFF.md §3 is the authoritative v5.4 summary.**

## v5.3.0 status (RELEASED 2026-07-06 to PyPI; geotech-references stays 1.3.0)

**5.3.0 adds** (on top of the 5.2.0 line): Batch-2 coverage 5/5 (drilled_shaft rational
GEC-10 chains, MSE LRFD external-stability CDRs, soe basal-heave-sidewall-shear + FHWA
apparent-pressure anchored walls + log-spiral Caquot-Kerisel Kp, full Reese-1974 sand
p-y, fem2d monolithic Taylor-Hood u-p Biot consolidation); slope_stability round 2
(15 new Slide2/ACADS/Duncan validation problems V-026..V-040, noncircular-search
robustness fix + rejection diagnostics, rapid drawdown 2/3-stage, Newmark + Jibson,
infinite slope, Ito-Matsui stabilizing piles verified vs the original 1975 paper);
pdf_import round 2 (scale calibration, label→region, cleanup, vision grid overlay,
vision↔vector cross-check); and 12 adversarial-review fixes (headline: log-spiral Kp
δ=0 Rankine anchor). Plan of record + deferred follow-ups: `module_work/V5.3_PLAN.md`.
The owner-gated publish rule still applies to FUTURE releases: no version bump / tag /
publish without explicit owner OK (a `v*` tag push auto-publishes via
.github/workflows/publish.yml).

**>> For a full cross-session handoff read `HANDOFF.md` (repo root) first. <<**

Everything since v5.0.0 is summarized in `docs/V5.1_SUMMARY.html` (one page; validation
tables). Master carries: the v5.1 backlog (token round-cap, calc-QC round 3, adapter
ergonomics, eval answer-keys/sample files, foundry audit, references review); the LE+FEM
modernization (branches `le-modern`/`fem-modern`, merged); the `reliability/` module
(consolidation #7 DONE); calc-package figures/tables + plotly interactive viewers
(`calc-viz`); the staged model-setup agent (`geo_project/` + `deep/setup_agent.py`,
OFF by default via `build_deep_agent(enable_setup_agent=True)`); the post-v5.0
lateral-pile calc-package fix + adapter-ergonomics sweep + the Databricks /Workspace
placeholder-write fix (`funhouse_agent/_fileio.py`); **Phase E** published-example
validation (25 problems as 87+ offline tests in `validation_examples/`, `RESULTS.md`,
0 analysis bugs -- also added general fem2d `roller_base` BC + `initial_stress_relaxation`);
and **v5.2 coverage Batch 1** (four additive, default-preserving capability adds:
`settlement` Hough, `pile_group` Meyerhof group settlement, `axial_pile` per-layer
`toe_friction_angle` + `head_depth`, `retaining_walls` MSE bar-mat Kr/F* curves -- see
`module_work/V5.2_COVERAGE.md`; Batch 2 = the bigger builds, NOT yet started).
Intentional behavior changes vs 5.0 (battered-wall KPE, pile_group +Mx sense, sheet-pile
embedment basis, wave-equation damping default, fem2d T6 default) are listed in the
summary page.
Databricks install: from /tmp or a UC Volume, `%pip install "geotech-staff-engineer"`
(the `[full]` extra covers the optional analysis backends — without it ~12 of the 97 eval
questions fail honestly with "not installed"), then
`dbutils.library.restartPython()` -- a stale runtime `typing_extensions` (<4.13) otherwise
breaks langgraph imports with "unexpected keyword argument 'extra_items'". Funhouse health
check: `funhouse_agent/deep/rc_wheel_check.py` (`run_rc_check(fh_prompter)`); the 100-question eval
suite: `funhouse_agent/deep/eval_harness.py` (`run_suite(model, out=...)`). Save outputs to
`/tmp` or `/Volumes`, NOT `/Workspace` (FUSE writes are non-durable / permission-blocked).

## Module Inventory (31 analysis + geo_project setup layer + 30 reference, + foundry harness; reference layer fully QC'd, all figure catalogs 100% page-accurate)

| Module | Tests | Purpose |
|--------|-------|---------|
| bearing_capacity | 83 | Shallow foundations (CBEAR/Vesic/Meyerhof, load-spread two-layer, GWT-in-wedge) |
| settlement | 53 | Consolidation & immediate (CSETT, Schmertmann, shape-factored elastic) |
| axial_pile | 65 | Driven pile capacity (Nordlund/Tomlinson/Beta, GWT-split integration, uplift) |
| sheet_pile | 36 | Cantilever/anchored walls (Rankine/Coulomb w/ wall friction; single FOS basis) |
| soe | 111 | Support of excavation (braced/cantilever, stability, anchors) |
| lateral_pile | 12+97 | Lateral pile (COM624P p-y models, banded FD solver, above-ground stickup; validation.py oracle suite) |
| pile_group | 85 | Rigid cap groups (6-DOF, one RH sign convention end-to-end, Converse-Labarre) |
| wave_equation | 60 | Smith 1-D wave equation (elasto-plastic springs, smith/smith_viscous damping, bearing graph) |
| drilled_shaft | 60 | GEC-10 alpha/beta/rock socket (clay end-bearing cap, N60 side reduction) |
| seismic_geotech | 82 | Site class, M-O pressures (battered-wall-correct), Fpga(PGA), SPT liquefaction (NCEER/Youd-2001) |
| retaining_walls | 93 | Cantilever + MSE walls (GEC-11; thrust decomposition, coherent-gravity MSE, Meyerhof bearing) |
| pavement_design | 42+ | AASHTO 1993 pavement design (US customary): flexible SN + Fig 3.2 layer split, rigid slab D (direct/MR-19.4/composite-k), ESAL traffic (Appendix D LEFs), calc-package template; orchestrates geotech_references.aashto_1993 |
| ground_improvement | 49 | Aggregate piers (incl. Priebe n0), wick drains, surcharge, vibro (GEC-13) |
| slope_stability | 384+17skip | Rigorous GLE/M-P (Fredlund-Krahn) + Bishop/Janbu/Spencer/OMS, entry-exit + DE noncircular search, nails/anchors/geosynthetics, SHANSEP/Hoek-Brown, ponded water, probabilistic FOS (FOSM/MC), SLOPE/W-grade plots; validated vs F&K-1977/ACADS/Duncan (VALIDATION.md) |
| reliability | 176 | Probabilistic geotech engines (FOSM/PEM/Monte Carlo/native FORM), published COV knowledge base (Duncan 2000/TC304/Phoon-Kulhawy), Vanmarcke spatial averaging, bearing/pile/slope wrappers |
| downdrag | 53 | Fellenius neutral plane, UFC 3-220-20 downdrag |
| geotech_common | 288 | SoilProfile (82) + checks (93) + adapters (89) + plots (21) |
| opensees_agent | 106 | PM4Sand cyclic DSS, 1D site response |
| pystrata_agent | 60 | 1D EQL site response (SHAKE-type, Darendeli/Menq/custom) |
| seismic_signals | 74 | Earthquake signal processing (eqsig/pyrotd) |
| liquepy_agent | 59 | Boulanger & Idriss (2014) liquefaction triggering — CPT (LPI/LSN/LDI) + SPT |
| gstools_agent | 69 | Geostatistical kriging, variogram fitting, random fields |
| salib_agent | 35 | Sobol & Morris sensitivity analysis |
| pystra_agent | 43 | FORM/SORM/Monte Carlo structural reliability analysis |
| section_props_agent | 70 | Cross-section properties, native (A, I, Z, S, J, warping; steel shapes + polygons, mm units; exact Green's-theorem integration + closed-form torsion) |
| concrete_props_agent | 47 | RC rectangular sections, native ACI strain compatibility (cracked/gross Ixx, M_cr, nominal Mn, N-M interaction) |
| pynite_agent | 15 | Elastic 2D/3D frames + continuous beams via PyNiteFEA (reactions, M/V/deflection envelopes) |
| subsurface_characterization | 231 | Subsurface data I/O: DIGGS parser (20 test types) + Plotly plots + trend stats; PLUS folded format adapters — GEF/BRO-XML CPT/borehole parse (pygef), AGS4 read/validate (python-ags4), DIGGS schema/dictionary validation (pydiggs) |
| dxf_import | 97 | DXF CAD import for slope stability + FEM (discover layers, parse geometry, build SlopeGeometry/FEM inputs) |
| dxf_export | 37 | DXF export for cross-section geometry (surface, boundaries, GWT, nails, annotations) |
| pdf_import → `planlens.pdf` | (planlens) | HISTORICAL PATH. The PDF cross-section importer ships in the separate `planlens` package since 2026-09-04 (`planlens.pdf`; app-side bridge `dxf_import/pdf_bridge.py`). |
| drawing_ir → `planlens.ir` + `planlens.document` + `planlens.tools` | (planlens, 1,307 tests at 0.6.0) | HISTORICAL PATH. The drawing IR (DXF / vector-PDF / raster ingest, slice queries, leader / dimension / title-block / bubble / cloud finders, `render_region`) is `planlens.ir`; since planlens 0.3.0 the WHOLE-DOCUMENT layer (`planlens.document`: page map + structure, located text, tables, review markups, hidden CAD text, Azure DI text source) and the LLM tool layer (`planlens.tools`) sit beside it. See "Document review & drawing geometry (planlens)" below. |
| fem2d | 353 | 2D plane-strain FEM (T6 default + CST/Q4/beam, 3D-principal MC return, HS, GL99 SRM, seepage, consolidation, staged construction, PLAXIS-style calc-package plots); validated vs Griffiths-Lane/Prandtl (VALIDATION.md) |
| geo_project | 89 | Canonical Project document for staged, human-gated LE/FEM model setup (schema+validators, builders, templates, DXF/PDF/vision ingest w/ provenance quarantine, echo-back renderer) |
| report_ingest | 1043 | One geotechnical report PDF → one organised, cited record and its four exports. Document triage and label review over planlens' page roles; three readers (boring/test-pit log, laboratory sheet, narrative against the owner's two standing query schemas, every answer cited); a reconciler that links lab tests to the ground and RECORDS disagreements rather than settling them; writers for `report.record.json`, a summary page, a WikiLLM library page + a SQLite index of many reports, and DIGGS 2.6 with both gates. `graph.ingest_report` is the deterministic, resumable loop. Since the second unreleased train on master a REPORT BOUND INSIDE A REPORT is its OWN record (`bound.py`): the union of triage's `bound_together` and the record's own `appended_report` runs, floor four pages, goes round the same loop into `out_dir/bound/<id>/` with `ReportRecord.bound_documents` on the parent and `ReportRecord.parent` on the child, its own library row carrying a new `parent` column -- so an earlier firm's borings are that firm's and the parent's counts are the parent's (`ingest_bound=False` restores the old listed-and-skipped behaviour). And since the first unreleased train its PAGE LABELS ARE A VOTE (`label_vote.py`): planlens' rules, one vision pass over the pages as pictures on the cheap tier (~$0.05 a report) and the printed form where a private fingerprint file is in force are combined under `label_policy` (default `structural`; `rules` reproduces the old single-voter path), `review_mode="disagreements"` gives the ~$0.45 label review ONLY the pages they split on plus two either side on a budget of `max(20, 0.5 x split pages)`, every page's label in the record carries its confidence, its voters and an `agreed` flag, and a split the review did not settle is a `label_disagreement` QA entry. The same combiner is what the `vote` stage scores, so a policy cannot mean two things. `run_folder` drives the loop over a folder; `subagent` puts it on the app as one `CompiledSubAgent` + one `report_ingest` tool, **OFF by default** (`build_deep_agent(enable_report_ingest=True)`) and feature-detected on the installed planlens. Engine-agnostic (`PrompterEngine` counts, `ClaudeEngine` is for development); `cluster_scoring.score_on_cluster(stages=("labels","logs","lab","calc","narrative","vision_labels","vote","ingest"), truth_dir=…, sharepoint=fh_sp_client)` is the owner's notebook cell; `truth_dir` is ONE root holding `logs/`, `lab/`, `calc/` and `narrative/` so one folder is uploaded rather than four paths kept in step, and `sharepoint=`/`durable_dir=` (5.22.0, `mirror.py`) copy every run file somewhere a cluster restart cannot reach and restore a wiped `out_dir` at the start. The seventh stage is `ingest` (5.23.0): the whole graph per report into `out_dir/ingest/<ID>/` with the DIGGS file gated, a saved label run reused, and the record scored against the hand truth — the whole-pipeline score; Since the calculation train a quarter of the corpus's pages stopped being skipped: `calc_reader.py` reads one run of `calculation` pages into one `Calculation` -- the kind it works out from a controlled list, the program and version as printed, the method, the subject, the labelled values it was given and the ones it worked out -- on a deterministic floor of (label, value) pairs, one model call, and three gates (a result printed on no page drops to confidence 0.3, a kind outside the list becomes `other`, a unit with no conversion is kept as printed and listed). DIGGS ignores calculations by design; the record, the summary's Calculations section and the library page carry them. Cluster stage `calc`. Since 5.24.0 a log is also matched against the printed FORM it came off
(`log_templates.py`): a page's footer stamp, title-block labels and column
headings are scored against fingerprints kept in a PRIVATE file that travels
with the truth folder and is never committed (`templates.json.EXAMPLE` ships
with invented firms), giving a `template <family> (0.92)` line for the vote
and a `column_map` that names the columns `log_grid` cannot — measured 100 %
recall and precision on both corpus families with no false positive off the
logs. The narrative reader gained its first accuracy levers in the same
release: the front matter and the front fifty pages always in, a DRAFT
conventions glossary the owner edits (`narrative_glossary.py`), per-question
retrieval over the whole report, the five exploration answers taken from the
logs, a quote gate that downgrades an answer whose quote is not on the page it
cites, and a lenient scoring view of the four free-text fields printed beside
the strict one. And since 5.23.0 the log and lab readers VOTE: the grid / the tables are the first voter and the floor (`floor.py`, `log_floor.py`, `lab_floor.py`), the model may add, correct with evidence, never drop, and every split is a `disagreement` QA entry with both values and confidences. The sixth stage is `vote` (5.22.0, `vote.py`): NO model — the rules, the vision labels and the review set against each other and against the hand labels, giving the agreement rate, a per-label trust table learned in sample only, three combining policies scored beside the voters, and the accuracy a targeted review of the disagreements would need. The fifth stage is the 5.20.0 EXPERIMENT: `vision_labels.py` sends each page as a picture to GPT-4.1 on the cheap tier with structured output, scored against the same hand labels by the same scorer as the rules and the review — one call a page, one a six-page contact sheet, or (5.21.0) one per window of 36 stamped full-size pages with contact sheets of the WHOLE report beside them (`vision_mode="document"`), which is capped by the endpoint's measured 50-image-per-request limit rather than by its context window. `report_ingest/README.md`, plan in `module_work/REPORT_INGEST_PLAN.md` |

Other components: geotech-references submodule (382 DM7 + 95 GEC/micropile + 10 FEMA + 9 NOAA + 35 UFC functions + DM7 figure catalogs, 3529 tests), foundry_test_harness (142 tests), funhouse_agent (106 + 149 + 163 + 25 + 31 + 5 = 479 tests)

## Foundry Test Harness

`foundry_test_harness/` validates all Foundry agent functions via JSON-in/JSON-out:

| File | Tests | Purpose |
|------|-------|---------|
| test_tier1_textbook.py | 73 | Individual functions vs textbook/published answers |
| test_tier2_workflows.py | 14 | Multi-function engineering workflows (e.g., classify → SPT → bearing → settlement) |
| test_tier3_crosscheck.py | 10 | Cross-agent consistency (same problem, different agents) |
| test_tier4_error_handling.py | 41 | Bad JSON, unknown methods, missing params, invalid values |

Supporting files: `harness.py` (FoundryAgentHarness class), `scenarios.py` (reusable problem definitions)

Run: `pytest foundry_test_harness/ -v`

## Document review & drawing geometry (planlens)

Everything that reads a PDF, an image or a DXF lives in the separate,
published package **planlens** (`C:/Users/socon/OneDrive/dev/planlens`,
PyPI `planlens`, editable-installed in dev; app pin `planlens>=0.6` since
5.20.0, raised from `planlens>=0.5` at 5.19.0 — plain, no extra, because 0.4.0
moved `opencv-python-headless` and `rapidfuzz` into planlens' core and left
`[raster]` / `[text]` as empty alias extras, and 0.5.0 and 0.6.0 each declare
exactly the same dependencies).
It is the owner's TinyApp "banner application" for any architect or engineer
reviewing documents; the geotech package rides along. No OBO branding, and do
not pitch it as CAD-object recognition — the goal is document review.

| Layer | What it gives the agent |
|-------|-------------------------|
| `planlens.document` | Any PDF (or image) as review-ready data: a page map (kinds text / drawing_sheet / form / figure / scanned / blank / mixed, with evidence, word density, the page number PRINTED on the page, sheet refs, scale notes); the document's STRUCTURE (segments from running headers/footers, printed numbering, dividers, duplicates); text lines with exact boxes and reading direction; tables; review markups (author, date, says/shows, the point a callout or arrow aims at, reply links); AutoCAD hidden SHX text; search; contact sheets. Azure Document Intelligence results as an optional text source (never called by planlens). One frame: displayed-page points, top-left origin — `render_region`'s frame. |
| `planlens.ir` | Drawing geometry: DXF / vector-PDF / raster ingest into a provenance-and-confidence-bearing IR, slice queries, the construct finders (leaders, dimensions, title blocks, bubbles, clouds), `render_region` (zoom with set-of-marks), `measure` / `spatial`. |
| `planlens.tools` | The framework-neutral LLM tool layer (`ReviewToolkit`): ten tools, JSON-Schema specs in Anthropic / OpenAI style, every result valid JSON inside a size limit, cursors for anything longer. |
| `planlens.pdf` | The PDF ingest leg + the geotechnical cross-section importer (role mappings, soil-layer vision prompts) that predates the split; app-side bridge `dxf_import/pdf_bridge.py` → `build_slope_geometry()` / `build_fem_inputs()`. Moving the geotech part back here is on planlens' open list. |

**In this app** (5.19.0; the surface is unchanged from 5.18.0 — planlens
0.5.0's `document_roles` is NOT yet wired, and will arrive with the ingest
sub-agent): `funhouse_agent/document_tools.py` bridges
`ReviewToolkit` to the deep agent — eight tools on the PRIMARY agent
(`open_document`, `document_structure`, `document_page_map`,
`read_document`, `search_document`, `document_markups`,
`render_page_thumbnails`, `find_quantities`; the toolkit's own `render_page` /
`render_region` stay off the app surface because the app has its own vision
tools).
Attachments resolve through a contextvar per call; one process-wide toolkit
keeps handles across turns; planlens budgets each result just under the
reference cap so nothing is string-truncated mid-JSON. Every `! look:` cue
names the app's eyes — `analyze_pdf_page` (page), `render_region` (spot),
`analyze_image` (the thumbnail sheets) — and the deep-agent prompt states the
policy: text first, then look whenever a result carries a look cue or seems
wrong for the page kind, and say what was read vs seen. The drawing-geometry
tools (`digitize_drawing` / `query_drawing` / `get_entities` / `snip_region` /
`search_drawing_set`) are still served by
`funhouse_agent/adapters/drawing_ir_adapter.py`. The tools hide themselves on
a planlens older than 0.3 rather than failing.

**The eighth tool and fuzzy search need planlens 0.4** (they arrived with the
5.18.0 pin; the floor is 0.5 since 5.19.0). `find_quantities` returns every number the document STATES with a
unit, with its wording, qualifier, page and box, so the agent can set what a
report says beside what a drawing measures; `search_document` gains `fuzzy` /
`min_score` (default 80, about 75 for a single word under eight letters) for
text read optically or plotted as strokes. Both stay FEATURE-DETECTED from
the installed package's own specs (`document_tools.has_tool` /
`document_tools.search_supports_fuzzy`, the surface built by
`document_tools.document_tool_names()`) rather than trusting the pin: the
cluster installs planlens from PyPI and has resolved older than the pin
before, and on an older planlens `find_quantities` is never advertised and a
`fuzzy=true` search returns a JSON error instead of calling the toolkit.
**This added no new third-party dependency to the app** — fuzzy search rides
on `rapidfuzz`, which planlens 0.4.0 carries in its own core alongside
`opencv-python-headless`.

Run: `cd ../planlens && pytest planlens -q` (planlens' own suite, which that
repo gates — it grew past 1,100 at 0.4.0; do not quote a count from here) and
`pytest funhouse_agent/deep/tests/test_document_tools_offline.py -q` (the
app-side wiring). Design notes: `planlens/document/DESIGN.md`,
`planlens/ir/DESIGN.md`; changelog `planlens/CHANGELOG.md`.

## Funhouse Agent (Engine-Agnostic Geotechnical Agent)

`funhouse_agent/` provides an engine-agnostic geotechnical agent with text + vision capabilities. Works with any AI backend satisfying the `GenAIEngine` protocol. Self-contained dispatch layer routes tool calls directly to the analysis + reference modules via internal adapters — no dependency on `foundry/` files. Analysis-module adapters + 16 geotech-references adapters (DM7 340+ equations, 7 GEC/micropile references with text retrieval, FEMA P-2192, NOAA frost, 4 UFC standards, the cross-reference text-search DB `reference_db`, and the figure-catalog search DB `figure_db` — which pairs with the `read_reference_figure` vision tool to find an engineering chart by meaning and read a value off it). The `subsurface` adapter now also exposes the folded GEF/AGS4/DIGGS-validation format-adapter methods (parse_cpt/parse_bore/read_ags4/validate_ags4/validate_diggs_schema/validate_diggs_dictionary).

| File | Purpose |
|------|---------|
| `__init__.py` | Exports: `GeotechAgent`, `GenAIEngine`, `ClaudeEngine`, `NativeToolEngine`, `PrompterBridgeEngine`, `USING_SDK_ENGINES`, `AgentResult` |
| `engine.py` | `GenAIEngine` Protocol + engines. **Hybrid**: prefers the Funhouse SDK's `funhouse.services.prompter.engine` (`NativeToolEngine`/`PrompterBridgeEngine`) when importable, falls back to local classes (`USING_SDK_ENGINES` flag). Local `ClaudeEngine` retained. |
| `agent.py` | `GeotechAgent` (native + text-ReAct loops, vision dispatch, `allowed_agents` scoping, `reference_mode` + `consult_references`) |
| `dispatch.py` | Tool dispatch + shared `REFERENCE_MODULES`/`ANALYSIS_MODULES` constants + `allowed_agents` scoping |
| `reviewer.py` | Reference consult sub-agent (`consult_references`) + legacy post-hoc reviewer (`run_review`/`needs_revision`) |
| `system_prompt.py` | Self-contained system prompt (50 modules) |
| `native_tools.py` | OpenAI tool schemas + dispatch for NativeToolEngine |
| `vision_tools.py` | File and vision tools: `list_files`, `read_pdf_text`, `analyze_image`, `analyze_pdf_page`, `render_region` (zoom + set-of-marks), `read_reference_figure` (render a catalogued figure + read a value off it), `view_worked_example_source`, `save_file` |
| `document_tools.py` | The whole-document review tools (planlens `ReviewToolkit` bridge, 5.16.0): `open_document`, `document_structure`, `document_page_map`, `read_document`, `search_document`, `document_markups`, `render_page_thumbnails` — attachments via contextvar, one process-wide toolkit, look cues naming the app's vision tools |
| `notebook.py` | `NotebookChat` — ipywidgets chat interface for Jupyter/Databricks |
| `adapters/` | analysis-module adapters + 16 reference adapters bridging flat JSON → module APIs (the former pygef/ags4/pydiggs adapters were folded into `subsurface_adapter`) |
| `tests/` | 106 tests (mock engines, no API key needed) |

Usage:
```python
from funhouse_agent import GeotechAgent, ClaudeEngine, NativeToolEngine

# With PrompterAPI (Databricks) — native OpenAI tool calling (recommended)
agent = GeotechAgent(genai_engine=NativeToolEngine(fh_prompter))

# With PrompterAPI — text-based ReAct (legacy, may not work with newer GPT models)
agent = GeotechAgent(genai_engine=fh_prompter)

# With Claude
agent = GeotechAgent(genai_engine=ClaudeEngine())

result = agent.ask("Calculate bearing capacity of 2m footing, phi=30")

# Interactive notebook chat (ipywidgets)
from funhouse_agent.notebook import NotebookChat
chat = NotebookChat(agent)
chat.display()
```

Run: `pytest funhouse_agent/ -v`

### Unified liquefaction tool (`liquefaction`)

Liquefaction triggering is consolidated into ONE discoverable agent-layer tool,
`liquefaction` (adapter `adapters/liquefaction_adapter.py`, method
`liquefaction_analysis`). It auto-routes by **input type** and **method** (no
cross-module imports — the routing lives at the agent layer, which is allowed to
call multiple modules):

- **CPT** (`q_c`/`f_s` present) → `liquepy` Boulanger & Idriss (2014) CPT
  (`liquepy_agent.analyze_cpt_liquefaction`), with LPI / LSN / LDI.
- **SPT** (`N160` present), `method="bi2014"` (**DEFAULT**) → `liquepy` B&I-2014
  SPT (`liquepy_agent.analyze_spt_liquefaction`).
- **SPT**, `method="nceer2001"` → legacy NCEER / Youd et al. (2001) simplified
  procedure (`seismic_geotech.evaluate_liquefaction`), for code-compliance work.

B&I-2014 is the default for both SPT and CPT. The per-module functions and the
direct `liquepy`/`seismic_geotech` adapters remain intact and callable.

**liquepy SPT note:** `liquepy` ships a packaged B&I-2014 CPT triggering object
(`run_bi2014`) but **no packaged SPT triggering object** — its only SPT entry
points are *field correlations* (Vs/Dr/G0 from N). It DOES expose every B&I-2014
SPT *building block* as tested module-level functions
(`calc_crr_m7p5_from_n1_60cs`, `calc_rd`, `calc_csr`, `calc_k_sigma_w_n1_60cs`),
so `liquepy_agent.analyze_spt_liquefaction` composes those into a full SPT
triggering procedure (adding only the B&I SPT fines correction Eq 2.23 and the
SPT MSF, which liquepy lacks for SPT). **seismic_geotech citation fix:** its
SPT procedure is NCEER / Youd-2001 (NCEER CRR fit, NCEER MSF, Youd fines, Liao &
Whitman rd) — the docstrings that previously cited "Boulanger & Idriss (2014)"
were corrected; the numerical code was unchanged.

### Reference consult-agent (`reference_mode`)

The primary agent no longer calls the 21 reference modules directly — reference access is routed
through a single `consult_references` tool backed by a **reference-scoped sub-agent** (a
`GeotechAgent` restricted to `REFERENCE_MODULES` via `allowed_agents`, generalized from
`reviewer.py`; uses the same engine, so it gets native tool calling). `GeotechAgent(reference_mode=...)`:
- `"anytime"` (**default**) — `consult_references` always offered; the primary is auto-scoped to
  the analysis modules (`ANALYSIS_MODULES`), shrinking its tool surface (which cut method/module-
  name guessing in live testing).
- `"after_calc"` — `consult_references` offered only after a `call_agent` has run this `ask()`.
- `"off"` — legacy: the reference modules stay directly callable, no consult tool.

`allowed_agents` (new `GeotechAgent` param) scopes the system-prompt catalog AND
`list_agents`/`list_methods`/`describe_method`/`call_agent`. `REFERENCE_MODULES` (21) and
`ANALYSIS_MODULES` (36) are shared constants in `dispatch.py`. The consultant holds `figure_db` +
`read_reference_figure`, so it does figure read-off too. Verified live in Databricks (5.1 deployment).

## Module-Improvement Agent Team

A standing, domain-organized agent team improves the 30 analysis modules over time, fed by the
agent test-suite feedback and other tasks. Claude Code teammates are ephemeral, so the team's
identity and memory live as **version-controlled files**:

- `.claude/agents/geotech-team-lead.md` — the team-lead playbook (run by the main session):
  triage feedback, maintain the board, dispatch specialists, review every diff, gate on tests,
  serialize `geotech_common` edits.
- `.claude/agents/geotech-module-specialist.md` — the reusable specialist (spawn one per domain,
  named by domain, e.g. `seismic`): reads its ledger → edits its module + adapter → runs
  `pytest <module>/ -v` → updates its ledger → returns a diff (does NOT commit; the lead reviews).
- `module_work/BOARD.md` — backlog + status across all domains; the single source of truth, with
  the triaged feedback categories and the 28-clean / 40-recovered / 0-failed agent-suite baseline.
- `module_work/<domain>.md` — per-domain progress ledgers (owned modules, reference map, backlog,
  log). A specialist reads its ledger at task start and updates it at the end.
- `module_work/triage_feedback.py` + `module_feedback.json` — turn a Databricks
  `geotech_test_suite_results.json` run into per-domain/per-module work orders.
- `funhouse_agent/geotech_test_suite.json` + `funhouse_agent/smoke_test_native.py` — the
  68-question eval set and the Databricks smoke/probe script that generate the feedback.

Domains (in the board): foundations, deep-foundations, earth-retention, slope-fem, seismic,
characterization, io-cad, references, common (lead-serialized).

**Status: the module-fix backlog is CLEARED (v5.1, 2026-06-10).** Phase 0 ergonomics, the
allowed_values rollout (23 adapters), the param-name hotspot fixes, and calc-QC Round 3 are all
on master; `module_work/BOARD.md` and `module_work/calc_qc/FINDINGS.md` carry the close-out logs.
The board remains the template for future feedback-driven rounds. `.claude/agents/` is
committed (the repo `.gitignore` was changed to `.claude/*` + `!.claude/agents/`).

## Reference-Layer Build Work (figure catalogs + new references)

A separate ongoing effort builds/QCs the geotech-references library. Infrastructure:
- `.claude/agents/figure-catalog-builder.md` — reusable subagent that drives any catalog's
  `page_estimated → 0` (finds each figure's true PDF page via caption/footer/List-of-Figures
  search) and builds catalogs for new references; can fan out one worker per reference.
- `reference_work/BOARD.md` — running ledger of the reference build/QC backlog + status.

**Working model (user preference, 2026-06-05): autonomous, milestone-level — NOT per-step.**
Drive the multi-agent work to completion; pick sensible defaults for ordering / which-first and
proceed without asking; commit + push freely as milestones land; report at milestones. The user
reviews this module in big batches (~weekly), not step-by-step. **No version bumps / PyPI
publishes until the user explicitly says so.** Still surface genuinely consequential / irreversible
decisions. See memory `feedback-reference-layer-autonomy`.

**Done (2026-06-05, released as geotech-references 1.2.5 / geotech-staff-engineer 4.6.5):**
- Deleted low-value/incorrect refs: `noaa_frost`, `ufc_dewatering`, `fema_p2192`.
- Built `ufc_backfill` + `ufc_expansive` figure catalogs; **ALL 20 figure catalogs at 100% page
  accuracy** (`page_estimated=0`).
- Three NEW references, full pipeline (text JSON + python lookups + figure catalog + tests +
  registry wiring in both repos): **`fema_p2082`** (FEMA P-2082 / 2020 NEHRP — site classes
  BC/CD/DE with BC baseline, Fa/Fv removed), **`california_trenching`** (Caltrans T&S Manual —
  shoring/excavation), **`fhwa_pavements`** (FHWA-NHI-05-037 — Mr/CBR/frost/drainage; distinct
  from `ufc_pavement`). Reference layer is now **30 modules** (EC7-1/EC7-2/AASHTO-1993 added 2026-07-15; `em_2104`/`em_2107` — USACE EM 1110-2-2104 RC hydraulic structures / EM 1110-2-2107 hydraulic steel structures — `ufc_structural` — UFC 3-301-01 DoD structural engineering, IBC/ASCE 7-22 modifications — the progressive-collapse pair `ufc_collapse`/`gsa_collapse` — UFC 4-023-03 / GSA Alternate Path 2016 — and `wood_handbook` — USDA Wood Handbook FPL-GTR-282 — all added 2026-09-03/04) (`dispatch.REFERENCE_MODULES`).
- **Deep QC** (4-agent fan-out) found & fixed 3 critical + 4 major content/functional errors
  (incl. a broken figure `pdf_path` that disabled vision for 2 refs, and the FEMA Ch-19 SSI
  equations that a build agent had mis-reconstructed); ~640 reference methods verified 0-bug.
- **Ergonomics:** semantic aliases for reference methods (`_reference_common.register_semantic_aliases`)
  + consult round-budget fix; **smart method resolution + analysis-method aliases** in `dispatch.py`
  (`_METHOD_ALIASES`, selector-value directives, fuzzy did-you-mean) driven by the agent-suite triage
  (`module_work/module_feedback.json`). (A `fdm2d` wall-clock guard was added here too; `fdm2d`
  has since been removed in the Phase 1 consolidation — see `CONSOLIDATION_CHANGES.md`.)

**Open backlog:** the ~23 per-module param-name/value bugs and the `retaining_walls`
earth-pressure-coefficient gap were FIXED in v5.1 (adapter-ergonomics stream). Still open:
module-*selection* mis-routing (e.g. Rankine guessed on the wrong module) and a `fem2d`
footing-SRM convenience method.
`ufc_expansive` has ~22 more figures behind an OCR pass (scanned PDF). The weekend QC routine's
gec_6/7/12/13 chapter-text edits are a separate uncommitted workstream.

## HANDOFF — Figure Read-Off: Status & Remaining Work

> Handoff note for the next team (agentic pickup). The figure retrieval + vision
> read-off feature is **built, working, and committed**; what remains is listed
> as a TODO with priorities. Read this whole section before touching the figure
> subsystem, `reviewer.py`, `agent.py`, or `system_prompt.py`.

### Done (committed `fef450c` on `master`, pushed)
- **Autonomy gap CLOSED** — the agent now finds a chart via `figure_db.figure_search`
  **and renders + reads it** with `read_reference_figure`, instead of answering chart
  values from the caption + model memory. Verified live: DM7.2 **Figure 4-12**, passive
  coefficient **Kp ≈ 5.3** (correct ≈ 5.5; it previously confabulated ≈ 9.0 from memory).
- **The fix = "starve-the-shortcut signpost" (the agentic option, see below).** Each
  `figure_search` hit now carries a `read_value` next-step instruction pointing at
  `read_reference_figure` and forbidding from-caption/from-memory values. The search
  result identifies the figure but exposes no values to guess from, so the vision read-off
  is the path of least resistance. (`funhouse_agent/adapters/figure_db_adapter.py`)
- **Param robustness** — `figure_search` tolerates common result-cap aliases
  (`top_k`/`k`/`n`/`max_results`/…) instead of hard-crashing on an unexpected kwarg; and
  `extract_method_info` (`adapters/_reference_common.py`) no longer advertises
  `**kwargs`/`*args` catch-alls as required parameters.
- Routing fix (`read_reference_figure` in the agent dispatch tuple), strengthened tool #7
  wording, a local `ClaudeEngine` runner `funhouse_agent/run_local.py`, offline adapter
  tests (`tests/test_figure_db_adapter.py`), and an opt-in live end-to-end test
  (`tests/test_live_figure_readoff.py`, formerly `xfail`).

### Design context (why it's built this way)
- **Architecture: find-with-text, read-with-pixels.** Retrieve figures lexically (SQLite
  FTS5 over captions + chapter-cross-linked descriptions), then hand the *actual rendered
  page* to a vision model to read values. CLIP-style **image embeddings were rejected** —
  poor fit for axis-labeled line charts. Do not pursue image embeddings.
- **Autonomy approaches weighed:** *Soft* (prompt-only — rejected, unreliable); **Medium
  (agentic retrieval + starve-the-shortcut — CHOSEN)**; *Hard* (deterministic auto-render —
  rejected: discards the agent's judgment and over-fires the costly vision call on any
  question that merely mentions a figure). Medium keeps the agent *reasoning* while making
  the from-memory shortcut useless. **Caveat:** Medium makes cheating useless, not
  impossible — if the agent is ever still seen reporting chart values without rendering,
  escalate to the Hard backstop (TODO P4).

### TODO — remaining work (prioritized)
- [ ] **P1 — Hallucination-on-tool-error (concerning, general, NOT figure-specific).**
      When a tool call errors, the ReAct loop may **fabricate success** instead of
      retrying/reporting. Observed: after a failed `figure_search`, the agent invented
      "Figure 3-5, score 19.0" and wrote *"the search and figure read both completed
      successfully."* Fix in the ReAct loop (`funhouse_agent/agent.py`) and/or system
      prompt: a tool error MUST be retried with corrected args or reported — never
      reported as success. Consider a loop-level guard that blocks a final answer
      contradicting an error result.
- [~] **P2 — Partially addressed by the reference consult-agent.** The shared `REFERENCE_MODULES`
      set (now in `dispatch.py`) DOES include `figure_db`, and the consult sub-agent runs a full
      `GeotechAgent`, so it can `read_reference_figure`. Only the legacy post-hoc `reviewer.py`
      (`review=True`) still lacks `figure_db`/`read_reference_figure` — wire it there too if used.
- [x] **P3 — DONE (2026-06-03): catalog recipe rolled out to ALL 15 references** (2491 figures):
      DM7 (dm7_1/2) + GEC 4–14 + micropile + ufc_pavement. `build_figure_catalog.py` was
      generalized to handle dot/dash/spaced-dash ids, heading + heading-less (dotted-leader
      density) + no-List-of-Figures (body-caption extraction) + sequential "Figure N" numbering
      (manifest `"figure_numbering":"sequential"`) + multi-volume (manifest `"volumes"` list →
      per-figure `pdf_path`). `body_start` is derived from the LoF span, so a manifest needs only a
      `pdf_path` (or `volumes`). Released in geotech-references 1.2.3 / geotech-staff-engineer 4.6.3.
      Quality: most ≥94%; gec_8 37% and gec_11 42% are partial (correct figures, offset-estimated
      pages, flagged `page_estimated`).
- [ ] **P4 — (conditional) Hard/deterministic backstop.** Only if Medium proves
      insufficient in practice: a deterministic auto-render for detected chart-value
      questions. Do not build pre-emptively.
- [ ] **P5 — (optional) Lazy description enrichment.** On first read of a figure, capture
      a one-line semantic description + axis/variable list and write it back into
      `figures_catalog.json` to improve future `figure_search` recall at zero bulk cost.
      Planned in the original design, not built.
- [~] **P6 — Cheaper recall lever LANDED (2026-06-09); text embeddings still deferred.**
      Built the lexical **synonym query-expansion** layer first
      (`geotech_references/_query_expansion.py`) wired into `reference_search`/`figure_search`,
      selected by `EXPANSION_STRATEGY` (env `GEOTECH_RETRIEVAL_EXPANSION`, default `auto`).
      Eval (`scripts/eval_retrieval_recall.py`): recall@5 **11%→44%** with **0** top-1
      disturbance (`auto` = rerank the literal+synonym union but pin the literal top-1).
      27 new tests + full geotech-references suite (3702) pass. **Merged and live on master**;
      reaches the Funhouse consultant for free (shared `reference_db`/`figure_db` adapters).
      **v5.1 follow-ups:** live Funhouse reviewer-agent eval (owner-gated), synonym-map
      curation, larger gold set, keep-`auto`-default-vs-opt-in decision — see
      geotech-references `## Lexical Query Expansion`. **Text embeddings remain DEFERRED**
      (only if expansion+lexical still prove insufficient); **image embeddings remain rejected.**
- [ ] **P7 — (deferred) Figure cropping.** Full-page render is robust; revisit only if
      multi-figure pages hurt read-off accuracy.
- [x] **P8 — DONE: figure catalogs are packaged.** `"*/figures_catalog.json"` added to
      `geotech-references/pyproject.toml` `[tool.setuptools.package-data]`. So `figure_search`/
      `figure_get` work from a clean install. Released 1.2.1+.
- [x] **P9 — DONE: configurable PDF location.** `_figures_db.resolve_pdf()` now honors the
      `GEOTECH_REFERENCES_DOCS` env var (folder holding the source PDFs), falling back to the
      repo-relative `docs/`. On Databricks: copy a `docs/` folder of PDFs in and set that env var.
      PDFs are still NOT shipped in the wheel (large + license). Released 1.2.1+.

### Gotchas the next team MUST know
- **(RESOLVED 2026-06-03) The native-tool-calling work in `agent.py` & `system_prompt.py` is now
  COMMITTED** (`b163750`): the `## Available Modules` regex fix, the Tool Discipline section, and
  the reference consult-agent (`reference_mode`, `allowed_agents` scoping, `consult_references`).
  Confirmed working live in Databricks on the 5.1 deployment; released in 4.6.2/4.6.3.
- **Active adapter-generation churn.** `funhouse_agent/adapters/` is modified by a
  background process during sessions (new `gec*_adapter.py`, `ufc_pavement_adapter.py`,
  `adapters/__init__.py`, `tests/test_reference_adapters.py`, `module_work/`). Method
  counts can flip mid-run. **Stage by exact path and verify `git diff --cached` before
  committing.**
- **`geotech_references` is an editable install** (`pip install -e geotech-references/`),
  so reference/figure changes are live from source.
- **Live verification (opt-in, costs API).** Set `RUN_LIVE_TESTS=1` and provide
  `ANTHROPIC_API_KEY` (shell env or Windows *User* env), then:
  `.venv/Scripts/python -m pytest funhouse_agent/tests/test_live_figure_readoff.py -v -s`,
  or `.venv/Scripts/python -m funhouse_agent.run_local --demo`. The key is read at runtime
  from the Windows User env — never pass it through chat/transcripts.

## Working on a Module

1. Read the module's `DESIGN.md` first for theory and conventions
2. Read `__init__.py` for the public API
3. Run that module's tests: `pytest module_name/ -v`
4. Full regression: `pytest -q` (testpaths configured in pyproject.toml)

## Environment

- Windows 11, Python 3.14.3, venv at `.venv/`
- Git repo: github.com/soconnell345-geotech/GeotechStaffEngineer (private)
- Git submodule: `geotech-references/` → github.com/soconnell345-geotech/geotech-references (DM7 + future GEC refs)
- numpy >=2.0: use `np.trapezoid` (was `np.trapz`)
