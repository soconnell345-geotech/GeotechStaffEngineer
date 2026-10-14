# BACKLOG — GeotechStaffEngineer, planlens, geotech-references

THE one list of open work across the three repos: the app, `planlens` and
`geotech-references`. Owner's rule, 2026-10-12: earlier sessions kept starting
narrower to-do lists, and whole plans dropped off. There is no other to-do
list:
- HANDOFF §0a-current names only this week's slice, by ID.
- Plans, design notes, TRACE_REVIEWs, smoke REVIEWs, field-feedback FINDINGS and `LIVE_TEST_QUEUE.md` keep the detail and the evidence.
- Every open item in those documents has a line here pointing back to it.

## How every session uses this file

1. **Starting work.**
   - Read the owner's answers on the **Backlog Desk** (https://claude.ai/artifact/9M7saxwayRQ6c7v1TeDjyQ: `ArtifactData` list of collection `responses`) and fold any new ones in here (`module_work/backlog_dashboard/README.md`).
   - Then read §0 and the tier you are working in, and say which IDs you are taking.
2. **While working.** Anything new goes HERE the moment it is found, under its area with the next free ID: a bug, a gap, a follow-up, an unverified claim, an owner question, a fix list from a review or a field session. The next free ID is the highest number in use for that prefix, plus one; never reuse one.
   - One line per item: what it is and why it matters, then `*Src:*` the document holding the detail.
   - A detailed write-up elsewhere is fine. A to-do list elsewhere is not.
3. **Finishing an item.** Delete its line and add it to "Done since the last sweep" with the commit. If a §0 item changed, update HANDOFF §0a-current's slice too.
4. **Releasing.** Sweep: every item the release closed moves to Done. Raise or lower tiers for the next train.

After any change to this file, rebuild and republish the Backlog Desk (`module_work/backlog_dashboard/build.py`, same URL). Every `[owner]` item needs a plain-English card in `module_work/backlog_dashboard/context/`; the guard test checks this.
5. **A full sweep** (on the owner's ask, or every few weeks): re-read the plan documents against this file, as on 2026-10-12. Clear "Done since the last sweep" and record the date below.

`webapp/tests/test_backlog.py` (part of the gate) checks four things:
- the IDs are unique;
- each ID has a tier;
- every ID cited here, or in HANDOFF §0a-current, exists;
- CLAUDE.md and HANDOFF still point here.

**Key:**
- **Tiers:** **Now** = the current focus, named in §0. **Next** = the next train. **Later**. **Parked** = the owner decided to wait.
- **[owner]** = needs the owner: a decision, an action, data, or CfA.
- *was:* = the item's id in an older list: HANDOFF `A1`–`E6`, or "Where things stand" `0`–`3e`. That old text is in git history at 414525a.
- **Short names:**
  - `H` = `HANDOFF.md` (line numbers as of e329b22)
  - `FI` = `module_work/FUTURE_IDEAS.md`
  - `B5` = `module_work/review_eval_results/2026-10-08_foundry_5.33.0rc1/` (Foundry brief 5 TRACE_REVIEW and slices)
  - `LTQ` = `module_work/LIVE_TEST_QUEUE.md`
  - `RA` = `module_work/REVIEW_ARCHITECTURE.md`
  - `RH` = `module_work/REVIEW_HARNESS.md`
  - `HT` = `module_work/harness_theory/`
  - smoke reviews = `module_work/live_smoke/runs/<wave>/REVIEW.md` (gitignored)
- **Last full sweep:** 2026-10-12 (app master 414525a, planlens f92541d = v0.13.0, refs c53e587 = v1.4.1).

## 0. Now (from 2026-10-14): core improvements while CfA sets up Tiny Apps

CfA says the SharePoint connection and the per-user Prompter keys are a few weeks out (owner, 2026-10-14). Until they arrive, Now is core work on the app itself. The Tiny Apps production items (OWN-29, OWN-30, OWN-43–OWN-51, TA-03) wait in Next and come back when CfA delivers.

- **Engineering answers that err on the unsafe side** (geotech page; each gets a warning in the result first, then the fix): GM-20 (free-earth returns a depth with no root). Then GM-42 (catalog headroom, needed before any new method) and GM-30 (Terzaghi factors).
- **Document Review accuracy** (testers' page; general fixes, no single-example rules): PL-07 (distinct sheets flagged as duplicates), VIS-01 (turned lettering read upright), VIS-04 (a zoom that cuts text lines), VIS-05 (boxes on sideways sheets), DR-15–DR-17 (markups), DR-04 (geotech tools still on the page), VIS-09 (zoom for pasted screenshots and image uploads).
- **Release readiness:** INF-10 (module suites on a fresh resolve, since the next release admits pandas 3; no release is planned (OWN-55), but these keep master releasable), EV-13 with DR-05 (one Foundry suite run that measures this train, tonight's default-page changes and the lean build).
- **Calculation tables** (OWN-10, 2026-10-14: yes, the next module train): GM-39, after the tracks above.
- **Owner:** OWN-31 (the checklist session, perhaps next week), OWN-58 (downdrag: the pile's own weight in the load curve?). Answered 2026-10-14 and folded: OWN-01 (became GM-48), OWN-02, OWN-10, OWN-55 (hold the release; the Prompter key works on Funhouse this month), OWN-57. Report ingest stays parked, so its questions (OWN-07, OWN-08, OWN-09, OWN-11, OWN-25) wait in Next and Later.
- **Rollout model (owner, 2026-10-12):** testers get ONE deployment, GPT-5.6 Sol or higher, for every call; not the pilot's `tinyapp-gpt-medium` (GPT-5.1, 768 px). The app measures whatever answers on first use (`funhouse_agent/vision_probe.py`), and planlens maps `gpt-5.6` to `openai-high` / `openai-original`. Sol has run the suite at full resolution on Foundry (5.32.0rc3; B5 39/42).

## 1. Owner: decisions and actions

### Decisions

- **OWN-05** [Next] [owner] **Pick uses for the promo API credit before it expires 2026-11-05** (~$35 left, $20 reserved for touch-up together): EV-03, waves after EV-04/EV-05, a cross-family overfitting check, Batch API probes. *was:* H 0, 3c
- **OWN-06** [Next] [owner] **Decide the OFF review switches** (table in §2): measure, flip or retire each. The M1 exit step "flip winning switches to defaults" never ran. *Src:* RA:206-213; *was:* H C10, 8a(vii)
- **OWN-07** [Next] [owner] **The seven DRAFT narrative conventions** in `report_ingest/narrative_glossary.py`. *was:* H B4
- **OWN-08** [Next] [owner] **Define `structureCount`; does `figureCount` include appendix figures?** *was:* H B5
- **OWN-09** [Later] [owner] **Report-ingest `label_policy` / `review_mode`**: unsettled (rules win in sample, the vote by 2 of 60 blind pages, the review lost 1.5 points); needs more blind hand labels or R39 (OWN-39). *was:* H B6
- **OWN-11** [Later] [owner] **Report ingest on the geotech page**: measure on one run, then an opt-in switch; RI-01–RI-03 first. *was:* H 8a(i)
- **OWN-12** [Later] [owner] **ASCE 7 and Youd et al. (2001) licensing.** Public routes exist: GEC-5 Fig 5-20 (Youd CPT charts), FEMA P-2082 / UFC 3-301-01 (ASCE 7-22 site classes), USGS design maps (OWN-53). *was:* H 8a(ii)
- **OWN-13** [Later] [owner] **Ito–Matsui stabilizing-pile force as its own method?** (eval SPL-3). *was:* H B8
- **OWN-14** [Later] [owner] **The five REVIEW_ARCHITECTURE questions**: the page threshold between shapes and when to ask before paid page notes; first disciplines and where their checklists come from; large-review deliverables; "results by morning"; public or private large test sets. *Src:* RA:256-265
- **OWN-15** [Later] [owner] **Ingest visual-scales switch**: unmeasurable on a corpus with DI text; measure on a no-DI scan set, or decide "never". *Src:* B5 TRACE_REVIEW:72-76, 112
- **OWN-16** [Later] [owner] **Does the slope "B" row count as a target in `produce-markup`?** *Src:* `review_eval/tasks.py:236`
- **OWN-17** [Later] [owner] **Rapid-drawdown stage-3 default** kept `'fellenius'` on 2026-07-08 (1.273 vs published 1.443, conservative; `'gle'` 1.370) while the owner checks notes; record the outcome. *Src:* `module_work/V5.4_PLAN.md:68-79`
- **OWN-18** [Later] [owner] **Move PyNiteFEA to an optional extra?** Removes the numpy 2.4 / numba cascade on the cluster; may unblock EV-08. *Src:* `docs/DATABRICKS_INSTALL.md:81-84`
- **OWN-19** [Later] [owner] **Lateral pile cracked / moment-curvature EI** (V-017 residual +9 %), pending since v5.3. *Src:* `validation_examples/RESULTS.md:365-367`
- **OWN-20** [Later] [owner] **Reference-matching opt-in defaults**: drilled-shaft large-D base reduction (V-008b); bearing depth factor and GW defaults (V-021); Tomlinson α low in stiff clay (V-002, conservative). *Src:* `module_work/V5.2_COVERAGE.md:69-75`
- **OWN-21** [Later] [owner] **Scope calls**: soldier-pile SOE mode; pavement parked scope (UFC 3-260-02 airfield, rigid curves F-2..F-31, AASHTO Part III; DESIGN says "out of scope", CLAUDE.md "parked"); frost / dewatering references dropped in June with no replacement; GEC-13 column-supported embankments and deep mixing; RAP upper zone; diesel hammer; structural future rounds; FEMA P-2192.
- **OWN-22** [Later] [owner] **Model-setup agent (`geo_project`)**: OFF until the owner tests it; test or park formally. *Src:* `HANDOFF_2026-06-14.md:93-101`
- **OWN-23** [Later] [owner] **Multi-user UI (A7 rec 6)** was "N/A until multi-user"; Tiny Apps is multi-user now. Re-decide or close. *Src:* `module_work/A7_INDUSTRY_MEMO.md:61-63`
- **OWN-24** [Parked] [owner] **Vision efficiency dial-back**: robust first (2026-09-25), "do not act until asked" (VIS-20).
- **OWN-25** [Later] [owner] **Published answer vocabulary** in public `report_ingest/model.py` carries a building-type value from the owner's wording: object, or load it from a private file? *Src:* H:1482-1497
- **OWN-26** [Later] [owner] **Repo visibility**: three of seven tracked field-feedback folder names carry a place name. Private, or scrub? *Src:* `module_work/REPORT_INGEST_PLAN.md:906-910`
- **OWN-27** [Later] [owner] **OK to delete the stray root files and stale branches** (INF-01, INF-08); archive `suite_trial.json` (the only copy of a June baseline) to a gitignored folder.
- **OWN-28** [Next] [owner] **Review the unmerged refs branch `rescue/reference-wip-2026-06-02`** (7fb6fe8, on origin): merge or drop (REF-04). A lead can review and propose.

### Actions and data

- **OWN-29** [Next] [owner] **Install 5.33.0** when Nexus delivers it (wheels in `../foundry_handoff/` otherwise, planlens first), then the dosdev round (`tinyapps/TINYAPPS.md`). *was:* H A1 **packages.txt (2026-10-12):** its pandas==3.0.2 pin is commented out, because 5.33.0's python-ags4 caps pandas below 3 (TA-02, in Done); restore it with the release that makes python-ags4 optional (OWN-55). **2026-10-14:** back to Next (§0); the dosdev round goes on as time allows, and what it turns up goes into §4.
- **OWN-30** [Next] [owner] **First-hour check on Tiny Apps** (two people, orientation time, note + PDF, SharePoint link, Word title, `vision_model` / `budget` / `cut_off`; steps in `live_smoke/runs/w3-confirm-sonnet/REVIEW.md`). Feeds TA-03. *was:* H 3e GO 3
- **OWN-31** [Now] [owner] **The checklist session**: `funhouse_agent/report_review_checklist.json` as a plain list, markup in chat or Word, transferred keeping each `id`. *was:* H 8a(vi), 3d
- **OWN-32** [Next] [owner] **LTQ check 5 on Funhouse** (SharePoint links, remembered files, past conversations) when the budget renews; cheap model, short PDF. *was:* H A2
- **OWN-33** [Next] [owner] **Check the scheduled routines** at claude.ai/code/routines: the weekend refs QC routine (appears never to have run: `qc_progress.json` `last_run: null`, no `qc/*` branches, an old Sonnet on a Pro plan) and the Monday drift canary. Fix or retire (REF-05, INF-03).
- **OWN-34** [Next] [owner] **Suite inputs**: which document types and questions testers care about; tester transcripts with known answers as private `extra_tasks`; a second author for a blind half (DR-24). *Src:* RH:134-137
- **OWN-35** [Next] [owner] **Ask the Funhouse developer for the Nexus approved/blocked list** (CSV) for INF-02. *Src:* `PLANLENS_PACKAGE_SURVEY.md:16-23`
- **OWN-36** [Later] [owner] **Cluster checks never recorded**: Prompter metering (`module_work/prompter_metering/DIAGNOSIS.md` §8; compliance; fix shipped 5.14.0) and DXF block explosion (`n_block_entities`). *Src:* `docs/DATABRICKS_INSTALL.md:216-222`
- **OWN-37** [Later] [owner] **DT scan items**: the pandas-2.x critical-flag watch; matplotlib "License-Threat Not Assigned"; release of `cytriangle` from quarantine (low).
- **OWN-38** [Later] [owner] **Real documents**: subsurface-investigation plans (the driving sheet plus 2–4) for DR-12; one Bluebeam-calibrated PDF; 2–3 scrubbed team sheets incl. a scan; a real doubled scan (PL-12).
- **OWN-39** [Later] [owner] **Report-ingest data**: Azure DI for the scanned corpus reports (R02, R31, R32, R38, scanned pages of R03 / R33, a fresh R17; DI is the only OCR on FIPS hosts); the R39 PDF (153 hand-labelled pages never opened: the cheapest blind set); blind calc / pit / sounding truth; a private library question set (RI-10). *was:* H B7
- **OWN-40** [Later] [owner] **Reference PDFs**: the Wood Handbook PDF (104 MB, local-only) to SharePoint `GSE_app/primary_references`; FEMA P-2192 Vol 1 by browser if OWN-21 says use it.
- **OWN-41** [Later] [owner] **The owner's own reports as worked-example exemplars** (Phase 2, local-only `GEOTECH_EXAMPLES_DIR`). *Src:* FI:127-136
- **OWN-42** [Later] [owner] **Funhouse admin**: send or drop the drafted 401 admin note; `funhouse-*` aliases price at $0.00 in the pricing table. *Src:* H:2142-2144

- **OWN-58** [Now] [owner] **Downdrag: should the load curve include the pile's own weight?** UFC 3-220-20 Fig 6-26, GEC-12 §7.3.6.1 and both worked examples leave it out; the module includes it (DD-1: +24 kN; V-004: 53.6 ft / 490 / 289 kips with it, 54.0 / 486.6 / 285.6 without, published 54 / 486 / 285). Pairs with GM-50. *Src:* `downdrag/DESIGN.md` (comparison item 2; V-004)
- **OWN-56** [Later] [owner] **Department addresses in the public app repo's git history**: your work address (prompter_metering DIAGNOSIS), CfA support mailboxes, the SharePoint host and site name, the published-app domains. Current files are scrubbed (24b46df); history keeps them. Purge (rewrite + force-push) or accept? Pairs with OWN-26.

### CfA and Funhouse questions

- **OWN-43** [Next] [owner] **Per-tester Prompter keys** (Key Vault `PROMPTER-API-KEY--<TESTER>`); `PROMPTER_SHARED_KEY_FALLBACK=1` until all exist. **CfA, 2026-10-14:** each user gets their own key, in a few weeks. *was:* H A1, 0
- **OWN-44** [Next] [owner] **SharePoint app registration** (four `GRAPH-*` / `SHAREPOINT-SITE-URL` values, `Sites.Selected`); reference PDFs reach Tiny Apps only this way (REF-20). **CfA, 2026-10-14:** the SharePoint connection is a few weeks out. *was:* H A1, 0
- **OWN-45** [Next] [owner] **The published address** for `corsAllowedOrigins`. *was:* H A1, 0
- **OWN-46** [Next] [owner] **Confirm the identity header** ("Signed in as <name>"). *was:* H A1, 0
- **OWN-47** [Next] [owner] **An image-capable Prompter deployment** (make-or-break): now the GPT-5.6 Sol-or-higher deployment testers will get (owner, 2026-10-12). *was:* H 3e GO 2
- **OWN-48** [Next] [owner] **Does the App Service enforce FIPS?** Assume yes: OpenCV cannot load there, so OCR and the raster drawing leg are unavailable; `find_like` still works on its numpy backend (planlens 0.12+). *was:* H A3
- **OWN-49** [Next] [owner] **Questions that fell off A1**: does the gateway pass streaming; Nexus install vs vendoring and the fleet pins CfA expects of our tree; the shared VM's resource envelope (fem2d meshes, Monte Carlo). *Src:* `tinyapps/TINYAPPS.md:236-253` **Fleet pins (TA-02, 2026-10-12):** our tree cannot install beside pandas==3.0.2 until the release after 5.33.0; ask whether a pandas 2.x from the mirror is acceptable meanwhile.
- **OWN-50** [Next] [owner] **What does the Prompter return when a key's budget is spent** (status and body)? The D2 classifier is built on guesses (TA-06).
- **OWN-51** [Next] [owner] **Is the pilot Prompter key the sanctioned, metered route?** (the Funhouse SDK Terms say direct Azure AI calls bypass metering).
- **OWN-52** [Next] [owner] **Process questions**: whether the Department SharePoint tenant is inside the pilot's data boundary (a DT CAB item); approval-path status (MOU → GHE → SIA → CAB → PCR); a monthly CfA check-in with telemetry noted in `TINYAPPS.md`. *Src:* `TINYAPPS.md:91-94, 242-253`
- **OWN-53** [Later] [owner] **Tiny Apps API-first egress** for the keyless USGS design-maps services (ASCE 7 values without licensing). *Src:* FI:343-355
- **OWN-54** [Later] [owner] **Funhouse questions**: an in-enclave MCP host (PL-15, LT-06); vendored model weights through Nexus (VIS-11); who reconciles the org skill that names a `geotechnical_engineering_query` tool. *Src:* FI:250-255, 398-414

## 2. Document Review harness and eval suite

### Default page fixes

Testers get the legacy deepagents build (`GEOTECH_REVIEW_AGENT` unset); the plan's Phase-1 fixes exist only behind `lean`, OFF.

- **DR-04** [Now] **The default page still offers `read_reference_figure` / `view_worked_example_source`** (`make_vision_tools` called without `include`, `deep/agent.py` ~665) and deepagents' coding prompt (~7.7K chars). *Src:* HT `document_review.md:122`
- **DR-05** [Now] **Re-measure `GEOTECH_REVIEW_AGENT=lean`** at full resolution and on GPT after 5.32–5.33 (last 34/35, Sol, 768 px); until lean or legacy wins, DR-01–DR-04 stay on the legacy build. *Src:* RH:35, 239

### Review architecture (M2–M4 and the owner's questions)

- **DR-06** [Next] **Phase 5: rewrite the review prompt around the review method**, dropping incident rules the suite does not need; after OWN-31. *Src:* RH:166-171; *was:* H C14
- **DR-07** [Later] **Cross-reference resolver and revision compare (planlens A4)**, only if they earn a suite task; the digest layer already finds cross-references (OFF). *was:* H C14
- **DR-08** [Later] **M2 remainder**: on-demand page notes with a cost estimate and consent; roll-up summaries; verify-on-page; 2–3 large (200–500 page) suite sets with planted problems. *Src:* RA:112-149
- **DR-09** [Later] **M3 mega-reviews**: editable plan, background job runner (checkpoints, resume, cost meter), discipline reviewers, consolidation, a seeded mega-set; needs OWN-14. *Src:* RA:151-181
- **DR-10** [Later] **M4 cross-document coordination.** *Src:* RA:170-173
- **DR-11** [Later] **Recompute-from-report QC mode**: pull a report's claimed inputs and results, re-run them through the modules, return a claimed-vs-recomputed table ("killer app" for the owner's job; extraction exists). *Src:* FI:151-165
- **DR-12** [Later] [owner] **Boring-spacing vertical slice** ("average spacing of the borings on this plan"), the 2026-09-13 reset's driving example; `planlens/ir/spatial.py` is reached by no agent; W3 `measure` now exists. Needs OWN-38; its plan file is missing (DOC-15). *Src:* H:2550-2578
- **DR-13** [Later] [owner] **Standards-conformance scenarios** (the owner supplies the standards). *Src:* `module_work/DRAWING_INTELLIGENCE_DESIGN.md:217-232`

### Switches

All OFF unless marked. Decisions: OWN-06.

| Switch | Default | Awaits |
|---|---|---|
| `GEOTECH_REVIEW_AGENT=lean` | unset | DR-05 (34/35 on Sol at 768 px, 5.31; never on GPT or after 5.32) |
| `GEOTECH_REVIEW_AGENT=minimal` | unset | Not a candidate (33/35 at ~4x tokens) |
| `GEOTECH_VISION_TEXT_CONTEXT` / `_STRUCTURED` | off | 34/35; structured partly superseded by pixel boxes; the agent-supplied `context` parameter never built |
| `GEOTECH_VISION_INLINE` | off | Lost 3 locate tasks; DR-28, VIS-13 |
| `GEOTECH_REVIEW_SWEEP` | off | Ties baseline at 2.4x tokens; needs a task it wins (DR-25) |
| `GEOTECH_REVIEW_FINDINGS` | off | **Never measured** (DR-27) |
| `GEOTECH_REVIEW_OVERVIEW` | off | Lost on UFC (9/10 vs 10/10); keep off |
| `GEOTECH_REVIEW_GEOMETRY` | off | Tied 35/35 at 768 px; the construct finders reach no user while off |
| `GEOTECH_REVIEW_DIGEST` | off | 34/35; needs large-set tasks (DR-08) |
| `GEOTECH_COVERAGE` | off | CV1–CV4 fixed, not re-measured; DR-23, then EV-02 |
| `GEOTECH_REVIEW_CHECKLIST` | off | OWN-31, DR-20, DR-21, then a measurement |
| `GEOTECH_VISION_PATCH_ALIGN` | off | EV-02 (≥10 repeats; brief 5 part F undecided) |
| `GEOTECH_INGEST_VISUAL_SCALES` | off | OWN-15 |
| `GEOTECH_CHART_CODE_VOTER` | **on** | Inert on real reference charts (`no_scale` 9/9) until PL-03, VIS-10 |
| `GEOTECH_VISION_POLICY` | robust | OWN-24, VIS-20 |
| `GEOTECH_VISION_MAX_PX` | 2048 | VIS-16 |
| `GEOTECH_FOUNDRY_REASONING_SUMMARY` | auto | L1 unverified live (EV-02) |
| `PROMPTER_SHARED_KEY_FALLBACK` | off | Set to 1 until OWN-43 |
| `enable_report_ingest` / `report_library` / `setup_agent` | off | OWN-11, RI-10, OWN-22 |

### Markups and checks

- **DR-15** [Now] **GPT-5.4 stops after two rejected markups**: tell it to keep correcting until confirmed, up to N tries. *Src:* check1 TRACE_REVIEW C6
- **DR-16** [Now] **Return the corrected box**, not only "misplaced" (the crop look already finds the thing; `markup_check.py:775-778`). *Src:* HT `locating_things_on_a_page.md:1326-1329`
- **DR-17** [Now] **B4: the duplicate-mark note should say a duplicate may hide an unmarked thing** (`planlens/tools/toolkit.py:926-933`). *Src:* B5 `TRACE_REVIEW_B_AEFG.md:250-270`
- **DR-18** [Later] **A rotated mark's label should follow the text under it** (F7/B8; 83a77f4 only suggests `label_reads`). *was:* H 3e "rotated labels"
- **DR-19** [Later] **DXF viewing**: render a DXF to PDF so the vision tools can look at it. *Src:* w2b REVIEW:124, 176-180

### Coverage and checklist

- **DR-20** [Next] **N10: read the summary row from the page text**, and say "not run" when no summary was written (GPT-5.4 "corrected" it 2/3, dropped it 1/3; the checklist passed). *Src:* B5 `TRACE_REVIEW_new_tasks.md:559`
- **DR-21** [Next] **N11 / N12**: false fail "sheets not on the summary" for kinds the summary omits; "explorations have logs" stays unsure after the scans were read. *Src:* `TRACE_REVIEW_new_tasks.md:374-381`
- **DR-22** [Later] **Checklist consistency items cannot run on the review page** (they need `write_diggs`). *Src:* `TRACE_REVIEW_new_tasks.md:385`
- **DR-23** [Next] **Coverage gaps before `GEOTECH_COVERAGE` goes on**: hidden CAD text counted as "no text layer" (MK9); report_ingest / `find_like` / contact-sheet reads not counted; a markup-only divider leaves scans as "other" (PL-02). *Src:* `module_work/COVERAGE_AND_CHECKLIST.md:188-202`

### Eval suite and measurement

- **DR-24** [Next] [owner] **A blind half of the suite**, written by someone else and kept off the repo; all 42 tasks are open. *Src:* RH:136-137
- **DR-25** [Next] **Suite gaps**: no spec section (UFGS); no real long drawing set (make it private); the orientation turn never runs (`review_eval/runner.py:159`). *Src:* HT `document_review.md:436-440`
- **DR-26** [Next] **S1: judge run-to-run variable tasks over repeats** (`set-long-rare-tag`, circles); score each turned look-alike from the trace. *Src:* B5 `TRACE_REVIEW_suite_rest.md:206-221`
- **DR-27** [Next] **An arm and tasks for `GEOTECH_REVIEW_FINDINGS`** (`funhouse_agent/review_flags.py:206-229`). *Src:* RA:101-103, 228
- **DR-28** [Later] **The inline arm stays a candidate**: probe it on the rollout model (GPT-5.6 Sol or higher); switch it to pixel boxes first (VIS-13). *was:* H C10

## 3. Vision

- **VIS-01** [Now] **R: render turned or sideways lettering upright** (6 of 15 turned look-alikes misread vs 0 of 11 upright). *was:* H E3
- **VIS-02** [Next] **`found` B1 / B2**: judge region vs text line by aspect; carry the text read (`vision_view.py:775, 804`). *was:* H E3
- **VIS-03** [Next] **Notes MK1 / MK2 / MK3 / MK5**: `aim_note` false alarms; `found_note` noise; say when looks disagree or a zoom is coarser than the tiles. *was:* H E3
- **VIS-04** [Now] **MK4: a zoom that cuts text lines invents the rest**; one general sentence: say "cut". *Src:* B5 `TRACE_REVIEW_suite_meck.md:264-271`
- **VIS-05** [Now] **MK6: boxes stay in the image as sent when the drawing is sideways** (one answer ~300 pt off). *Src:* `TRACE_REVIEW_suite_meck.md:275-279`
- **VIS-06** [Next] **E4: `code_reading` should flag a READ input outside the fitted axis** (110 tsf at φ 28° on a 30–45° axis; `funhouse_agent/chart_reading.py`). *Src:* `TRACE_REVIEW_B_AEFG.md:495-499`
- **VIS-07** [Next] **Zoom latency 30–50 s (F1)**: A/B a reasoning setting on vision calls on GPT; a ~4k zoom cap (still 8k). *Src:* w3 REVIEW:220; *was:* H E3, 3e
- **VIS-08** [Next] **GPT-5.4 reads log depths by eye (M1)**: lead the `measure` / `log_grid` descriptions with the reviewer's questions; measure ×3 per wording. *Src:* `TRACE_REVIEW_new_tasks.md:402-426`; *was:* H E5
- **VIS-09** [Now] **A crop/zoom tool for raster uploads and pasted screenshots.** *Src:* FI "VISION HARNESS BENCHMARK"; *was:* H C9
- **VIS-10** [Next] **Per-figure crop boxes in the reference catalogs (P7)**: whole-page renders blind `find_scales` on two-figure pages, so `code_reading` never measured Fig 7-15. Find each box once at build time. *Src:* FI:540-543; B5 TRACE_REVIEW:82
- **VIS-11** [Next] [owner] **Free OCR on FIPS hosts**: only paid DI works where the app runs (RapidOCR imports OpenCV, which aborts). First a child-process probe of `onnxruntime` under FIPS, then approval of vendored weights (OWN-54). Merges Tier-2 engines and PL-21. *Src:* FI:384-386, 489-494
- **VIS-12** [Later] **MK7: offer `measure` / `log_grid` only when a scale or log page exists** (~930 tokens a call). *Src:* `TRACE_REVIEW_suite_meck.md:61-67`
- **VIS-13** [Later] **The inline route still asks for 0-999 boxes** (`deep/inline_images.py:71`; measured 57–91 pt off). *Src:* HT `locating_things_on_a_page.md:144-147`
- **VIS-14** [Later] **`find_like`'s example box can come from a wide view**: apply the 300 pt rule. *Src:* same, :145-147
- **VIS-15** [Later] **Drop `page_bbox` figures a side call computes itself** (status unclear). *Src:* same, :1332-1334
- **VIS-16** [Later] **9b's open calls**: 300 pt view limit or ~450 pt; the 2,048 px cap on Foundry (`none` for full size). *was:* H 9b
- **VIS-17** [Later] **Chart truth**: Chartography (check licence), a small geotech chart truth set, public chart/lab truth (vector lab sheets with table and plot; charts with printed equations). *Src:* FI:520-547; *was:* H C9
- **VIS-18** [Later] **BenchCAD Vision2Code benchmark.** *Src:* FI:533-535
- **VIS-19** [Later] **Set-of-marks for `measure`'s alternatives** (the model picks a number); the old SoM A/B harness calls the Claude API and would need porting. *Src:* FI:478-482; `drawing_ground_truth/som_ab_check.py`
- **VIS-20** [Parked] [owner] **Efficiency knobs** (`GEOTECH_VISION_POLICY=efficient`, 2x2 tiles first, crop to the drawing, persist probe profiles); first record brief 4's tiling cost (90 of 100 looks tiled; long-set input doubled) and MK5 in FI. *Src:* FI:437-464
- **VIS-21** [Later] **Drawing truth**: the public county corpus is SHX-only, so text-gated behaviour is unmeasured; needs a text-bearing multi-firm set, scanned sheets (OWN-38), more DWG+PDF pairs. *Src:* `DRAWING_INTELLIGENCE_DESIGN.md:251-295, 499-523`

## 4. Tiny Apps, Funhouse/Databricks and hosting

- **TA-03** [Next] **What only live GPT can show**: the rollout model's vision (no `GEOTECH_VISION_BUDGET` / `GEOTECH_CHART_BUDGET` lines; diagnostics should name GPT-5.6 Sol or higher at `openai-high`, charts at `openai-original`, and the host's edge; a half-size sheet answered from the text layer); spend per turn against each tester's budget, since full-size images cost several times gpt-5.1's 768 px (OWN-50), latency vs caps, budget and rate-limit responses, separate keys, links behind IIS, `reuse=true`. *was:* H E5, 0, 3e
- **TA-18** [Later] **The 401 retry reuses the client's pooled connection** (TA-01's retry, 0562ddf): if a refusal survives the retry live, rebuild the model client for the retry (the 2026-09-09 hypothesis was a stale NTLM connection). *Src:* HANDOFF 2026-09-09 field feedback
- **TA-04** [Next] **Not verified live**: the D2 budget message; busy retries; restore after restart; the Excel writer; two testers saving at once (H12); `annotate_dxf` in real CAD; D3 (harness copies the old sidebar). *was:* H E6
- **TA-05** [Next] **An SDK-free spend line in the sidebar** for testers on a $50/month key (`webapp/budget_panel.py` is SDK-only). *Src:* H:681-683
- **TA-06** [Next] **Pin the budget-spent response in a test** after OWN-50.
- **TA-07** [Later] **"Attach from SharePoint" box** (no model turn, no size cap); `webapp/app.py:793-794` says it is recorded in HANDOFF; it is recorded here. *Src:* FI:299-300
- **TA-08** [Later] **Chunked websocket upload past 25 MB** (a Databricks proxy limit; check Tiny Apps first). *Src:* FI:301-302
- **TA-09** [Later] **The email tool is SDK-only and off on Tiny Apps.** *Src:* H:683-685
- **TA-10** [Later] **A re-imported Plotly sidecar becomes `plot.plotly_1.json`** and degrades to a static card (`webapp/core.py:1509-1518`). *Src:* FI:379-383
- **TA-11** [Later] **Plotly → PNG twin for `subsurface.plot_*`**, blocked by broken static export in the tenant. *Src:* FI:305-306
- **TA-12** [Later] **Funhouse bridge retry on rate limit** (Tiny Apps got retries in 8ef04ad) (status unclear). *Src:* `module_work/field_feedback/2026-10-06_geotech-report-session_v5.32.0/FINDINGS.md:75`
- **TA-13** [Later] **Metering residue** (Funhouse is the backup host): `unknown_user` attribution; streamed rows labelled `ChatCompletionChunk`; `@check_budget` never runs; SDK bug reports. *Src:* `prompter_metering/DIAGNOSIS.md:201-385`
- **TA-14** [Later] **Auto-continue evidence audits** (false completion claims) from the Funhouse SDK (status unclear). *Src:* FI:231-234
- **TA-15** [Later] **Old live checks never recorded** (largely overtaken; status unclear): `email_file`, opt-in streaming, the model picker (5.11.x); 5.15 PDF figures and restore; 5.16–5.18 prompt-level checks. *Src:* H:1710-1749, 2695-2730
- **TA-16** [Parked] **Durable LangGraph checkpointer** and the **per-model API thinking layer**. *Src:* `module_work/APP_PLAN.md:83-103`

## 5. planlens

- **PL-01** [Next] **Rectangles read as x, y, w, h** (`loggrid.py:906-909`, `document.py:137-141`); measure the report-ingest log floor before fixing. *was:* H 8a(viii), E3
- **PL-02** [Next] **A markup-only divider page should start and title a segment** (`document/structure.py:130`). *was:* H 8a(iii)
- **PL-03** [Next] **Chart reading E1 / E2b / E3**: raster decision per region; a y scale labelled every 4th line; follow the input line on a single curve. *was:* H E3
- **PL-04** [Next] **`log_grid` N7 / N8**: multi-log header fields; table row rules read as strata. *was:* H E3
- **PL-05** [Next] **N14: `measure`'s `to` across a second zoom.** *was:* H E3
- **PL-06** [Next] **C4 public API**: `ReviewToolkit.forget(path)`, atomic save in `write_markups`, stable markup ids (positional today, `annotations.py:194`). *was:* H E3
- **PL-07** [Now] **Distinct stroke-drawn sheets are flagged as duplicates** (22 of 24; `document/document.py:436-441`), inviting skipped pages. *Src:* HT `document_review.md:335-341`
- **PL-08** [Later] **`log_grid` blind gaps**: index values 8/17 (plotted column), header fields 33/54, sheets with no depth unit, one zero-scoring log unexamined. *Src:* `planlens/document/DESIGN.md:1286-1310`
- **PL-09** [Later] **Reading gaps**: multi-column / visual reading order; tables spanning pages; captions on non-figure pages. *Src:* DESIGN.md:71-72, 1318-1327
- **PL-10** [Later] **Page roles**: `dcp_log` on scans (0.54); no pressuremeter, vane or dilatometer roles. *Src:* DESIGN.md:977-980
- **PL-11** [Later] **`find_quantities` misses `Label (unit): value`** (`scan_text("Pc (kip): 437.0")` returns `[]`); all 15 real-corpus misses. *Src:* DESIGN.md:294-300
- **PL-12** [Later] [owner] **Data-gated**: stored scale never seen on a real calibrated page; duplicate scans only synthetic; `/GEO` viewports detected, not parsed. Needs OWN-38. *Src:* DESIGN.md:320-376
- **PL-13** [Later] **Port the drawing tools into `planlens.tools`** (`digitize_drawing`, `query_drawing`, `get_entities`, `snip_region`, `search_drawing_set`). *Src:* `planlens/ir/DESIGN.md:509-511`
- **PL-14** [Later] **Move the geotech cross-section importer out of `planlens.pdf`** with a deprecation path (incl. `candidate_ground_surface`); it also ignores the new `Frame` (no vertical exaggeration or datum). *Src:* `planlens/pdf/DESIGN.md:3-10`
- **PL-15** [Later] [owner] **The MCP server has never run in the enclave**: check `mcp` 2.2.0's pins against the streamlit stack; it authenticates nobody. Waits on OWN-54. *Src:* planlens README:613-623
- **PL-16** [Later] [owner] **Bound OpenCV `<6`** (`planlens/pyproject.toml:52`) and re-run the suite on OpenCV 5. *Src:* `docs/DATABRICKS_INSTALL.md:158`
- **PL-17** [Later] **Multi-subpath PDF drawings fused into one polyline** (`pdf/extractor.py:473-505`); a re-baselining milestone. *Src:* H:2574-2579
- **PL-18** [Parked] **Construct-finder sharp edges** (frozen 2026-09-13): continuous-leg dimension ends; round-4 findings 5–6; mirror ties; MINSERT ingests only the first copy (`ir/ingest.py:539-548`). *Src:* `module_work/code_review/2026-09-06_planlens_phase32/`
- **PL-19** [Later] **Round-4 process gaps**: `doc_claims_check.py` re-implements ingest's layer walk; README guards pin constants, not script output. *Src:* `ROUND4_VERIFICATION.md:214-244`
- **PL-20** [Later] **Raster leg limits**: arcs lost, thick strokes trace double, a second OCR via pytesseract; the OCR extra needs GUI OpenCV (recipe pinned to 1.2.3); shapely `STRtree` unused; raster vectorization frozen. *Src:* `planlens/ir/DESIGN.md:252-274`
- **PL-21** [Later] **`planlens.ocr` as a document text source** (image-only logs read as `figure`); blocked by VIS-11. *Src:* DESIGN.md:988-991

## 6. Report ingest

Parked at 5.25.0; every number since is GPT-5.4 on Foundry. Code items were confirmed open at master 414525a by a code check on 2026-10-12. RI-01–RI-03 block OWN-11.

- **RI-01** [Later] **Summary and library citations print 0-based pages** (`report_ingest/writers.py:358-361`, `library_agent.py:696`); Document Review fixed the same in 5.30.0. *Src:* HT `report_ingest.md:352`
- **RI-02** [Later] **Sub-agent middleware is declared, not enforced** (`report_ingest/subagent.py:428-453`). *Src:* FI:622-629
- **RI-03** [Later] **The ingest engine needs a Funhouse SDK prompter** (`engine.py:1123-1172`), so ingest cannot run on Tiny Apps. *was:* H 8a(i)
- **RI-04** [Later] **Log floor**: G2, bind index cells to the nearest sample (blind index 0/16); the floor keeps the grid's mistakes (R30; HT Q1); N not seeded from drives, a model-added N unchecked (`log_floor.py`, `floor.settle`); R25_p19 reads nothing. *was:* H E4
- **RI-05** [Later] **G3: RESULTS prints the visual-scales counts.** *was:* H E4
- **RI-06** [Later] **Lab links**: gradation links flip between runs (measure over repeats); per-specimen hole-match and depth bands; maybe a deterministic link floor. The cause of Run 1's lab drop (83 → 46 %) was never confirmed (status unclear). *Src:* MEASUREMENTS.md:4196-4312; *was:* H E4
- **RI-07** [Later] **Lab reading gaps**: chemical 54 %, compaction 1/10, consolidation 10 %, a gradation at 21 %; `lab_reader.py` has no page window. *Src:* MEASUREMENTS.md:4267-4268
- **RI-08** [Later] **Small items**: ~1,080 unranked QA entries a report; leading zeros ("B-07" vs "B-7"); R16 continuation sheets; `score_on_cluster` demands a prompter for `vote`; `figureCount` from page labels. *was:* H C13
- **RI-09** [Later] **Re-run the whole-pipeline `ingest` stage** (once, at 5.31.0; since then 1ef959b, f61505d, `triage_model`, visual scales); include a public report end to end into the subsurface plots. *Src:* LTQ:344-358
- **RI-10** [Later] [owner] **The library's model half was never measured**; write the private question set first. *Src:* HT `report_ingest.md:36`
- **RI-11** [Later] **The ingest ledger (owner's design)**: one row per report per ingest to JSONL mirrored to SharePoint, a Delta table, a SharePoint list. *Src:* FI:802-845
- **RI-12** [Later] **Narrative cheap-vision second voter**, unblocked now the levers are measured. *Src:* FI:725-739
- **RI-13** [Later] **Weak narrative fields beyond B5**: `earthHazardsExposed` 0/3, `structureList` 3/7 (+1 invented), strata 4/8, `recommendedFoundations` 4/8, list recall 51 %. *Src:* MEASUREMENTS.md:4265
- **RI-14** [Later] **Pipeline structure**: narrative read once per item, last answer wins (`graph.py:1363`); bound children overlap parent pages (`graph.py:557-566`, `bound.py:87`); narrative counts from prose only (no `investigations=`); no lab `hint_kind`.
- **RI-15** [Later] **Label review**: breaks about as many labels as it fixes; R24 ~0.48; count fixed / broke over stage b (HT Q3); requests outgrow the tokens-per-minute limit; every `appended_report` page is a forced split (`label_vote.py:87`); changes on ledger evidence only (`label_review.py:454`). *Src:* H:1390-1394
- **RI-16** [Later] **Unmeasured cost levers**: `vision_detail="low"` (~85 tokens an image); `vision_outline_context`. *Src:* `report_ingest/README.md:2046-2078`
- **RI-17** [Later] **Calc reader**: `has_evidence` accepts any box plus note (`floor.py:135`; R20, R30 lost ground); runs split into one-page items with only the first page as an image (`calc_reader.py`). *Src:* MEASUREMENTS.md:4275-4276
- **RI-18** [Later] **Logs with no fitted ruler lose every depth** (`log_reader.py:788`).
- **RI-19** [Later] **The 16k output cap on reader calls** (`engine.py:137`); it killed R21's review. *Src:* HT `report_ingest.md:261-270`
- **RI-20** [Later] **Units and validation**: blank unit read as dimensionless (`model.py:190`; 17 %); calc units unconvertible (13 %); triage workflow unvalidated; a schema check that could not run recorded as failed (`writers.py:1075`); "mg/L unconverted" noise in `write_diggs`.
- **RI-21** [Later] **Cost records**: failed and rate-limited calls omitted; engine adaptations and reasoning tokens not persisted; bound documents take 206–897 calls each: acceptable? *Src:* HT `report_ingest.md:324-328`
- **RI-22** [Later] [owner] **Triage never scored against a hand verdict** (its `multi_document` call, 19 of 38). *Src:* `REPORT_INGEST_PLAN.md:587-611`
- **RI-23** [Later] **Votes not yet votes**: the narrative-vs-appendix count is a `count_mismatch`; the template voter, vote stage and trust table never ran in a pipeline run. *Src:* FI:712-714
- **RI-24** [Later] **WP5 tail**: boring locations off the plan figure (W3 enables it); the sounding floor's digitised series (OWN-15). *Src:* FI:600-601
- **RI-25** [Later] **Low**: the image-duplicate rule has no real positive; DIGGS dictionary / 2.5.a need pydiggs; the Funhouse gateway body limit (unclear); Spark packaging and Word reports parked. *Src:* `REPORT_INGEST_PLAN.md:796-905`

## 7. Geotech analysis modules

### Downdrag
- **GM-02** [Next] **Endo ratios unverified.** CGPR #56 IS on disk in a downdrag field session's gitignored `raw/` (2026-09-04): verify locally, commit nothing (not public domain). *was:* H E2
- **GM-03** [Next] **Post-liquefaction drag (GEC-10 §10.7)** not implemented; pair with a liquefaction worked-example anchor (NCHRP downdrag Appendix G CPT example). *was:* H E2
- **GM-49** [Next] **Settlement below a floating pile's toe is counted only within 3B' of the toe**: deep compressible soil below is under-counted. UFC goes down until the stress increase falls below 10 % of the effective stress, and a fill's stress increase does not decay with depth. *Src:* `downdrag/DESIGN.md` (settlement); GM-48 report 2026-10-14
- **GM-50** [Next] **Pile unit-weight defaults disagree**: the downdrag tool defaults to concrete (24 kN/m3) with steel's stiffness (200 GPa); the calc package defaults to steel (78.5). Make the default follow the pile material, or require it; settle with OWN-58.
- **GM-51** [Next] **No warning when nothing settles below the toe**: without a soil modulus on the bearing layer the settlement below the toe is about zero (DD-1: 0.1 mm), and the result says nothing.
- **GM-04** [Later] **Downdrag limits**: Poulos / PILENEG outside the comparison; single pile, ultimate consolidation only; β shaft far below Nordlund in dense gravel (V-004); β = 0.3 default uncited. *Src:* `downdrag/DESIGN.md:61-64, 317-325`

### fem2d
- **GM-07** [Next] **θ = 0.5 rings ±5 kPa at drained boundaries.** *was:* H E2
- **GM-09** [Next] **Consolidation fill under water is refused.** *was:* H E2
- **GM-61** [Next] **Staged analysis: a wall or beam activated in a later stage carries the movement its nodes made before it existed**: a gravity-then-install stage with no change shows 186 kN/m axial and 9 kNm/m moment with zero stage movement (same on the old code). The same class as the fill fix in 4bf710c. *Src:* fem2d agent report 2026-10-14
- **GM-62** [Next] **`analyze_staged` `max_iter` default 100 is low for plastic stages** (a 260 kPa footing converges with 500, not 100); documented, not changed.
- **GM-63** [Next] **Deep undrained slope: SRM 0.648 on every mesh against limit equilibrium 0.719 / 0.722 (−10 %)** (Duncan Ex 1 with a 2H foundation); Taylor's deep-foundation value is about 0.66-0.69. Settle which side is off.
- **GM-64** [Later] **The column mesher gives every column `ny` rows**, so a thin base layer gets flat elements; and `nr_method='tangent'` / `nr_fallback` are still unvalidated.
- **GM-10** [Later] **fem2d backlog**: structured slope mesher (toe slivers); GL99 Ex3 / Ex5; Q8; flag cohesionless chatter; T6 seepage / Biot; HS σzz elastic in SRM; SHANSEP / Hoek-Brown pre-linearised; `analyze_cavity` / `analyze_tunnel`. *Src:* `fem2d/UPGRADE_PLAN.md:158-248`
- **GM-11** [Later] **`analyze_footing_capacity` ignores side surcharge / embedment** (`fem2d/analysis.py:225-227`).
- **GM-12** [Later] **Slow rigorous searches and SRM runtimes** (G18; a module item, not a reference one). *was:* H E2

### slope_stability
- **GM-59** [Next] **Bishop counts a concentrated support only through its moment about the circle centre**, so a small toe circle whose centre sits near the support's level counts almost none of it: Slide2 #85 (active support) Bishop search 1.304 against Slide2's 1.531, while on Slide2's own circle the module gives 1.564. Review how simplified Bishop applies reinforcement forces, or warn when a Bishop critical surface crosses reinforcement with a short arm; meanwhile use a rigorous method on anchored slopes.
- **GM-60** [Next] **GLE / Spencer often fail to converge with large point loads** (Slide2's own #85 circle; anchored toe circles), so rigorous searches lose those surfaces and the toe-circle family found nothing for anchored #85 with Spencer.
- **GM-56** [Later] **Spencer / GLE reject thin planar polylines** (no F_m = F_f crossing), so noncircular searches can lose shallow surfaces; Janbu's f0 comes out 1.0025 there (≤ +0.25 %).
- **GM-57** [Next] **Automatic entry/exit search ranges are still fractions of the model width**; the new edge-of-range warnings flag surfaces they cut off, and the toe-circle / infinite-slope families cover toes and c′ = 0 faces only. Also check the Taylor φ = 0 benchmark (fine grid 2.039 vs the chart's 1.92).
- **GM-58** [Next] **Survey every adapter for nested dicts read with `.get`** (the slope adapter silently dropped a misspelt layer cohesion until GM-52); reuse `check_nested_keys`.
- **GM-15** [Next] **Steep-φ′ Kc rapid-drawdown overshoot ~22 %** (Lowe-Karafiath, DWW; V-052b, φ′ = 45, a failed case). *Src:* `V5.4_PLAN.md:325-332`
- **GM-16** [Next] **Five method descriptions exceed the 8,000-char result cap.** *Src:* HT `geotech_agent.md`; *was:* H C11
- **GM-17** [Next] **Jibson (2007) caveat** in Newmark docs and results (≥82 % unconservative bias over 42 dams). *Src:* `wiki_verification/TIER_A_LEDGER.md:89-92`
- **GM-18** [Later] **Slope residuals**: single-anchor with-support FOS (#85); degenerate free Spencer on ponded geometry (V-044); clay-crest water crack (V-043); Slide2 #33 (unclear); no cos²β seepage (conservative); the drawdown search hotspot; DXF slip export (unclear). *Src:* `validation_examples/INVENTORY.md:716-805`
- **GM-19** [Later] **Methods never built**: Janbu rigorous, Sarma, Barton-Bandis, a liquefied strength, a curved envelope, Hassiotis / Kumar & Hall piles. *Src:* `slope_stability/IMPROVEMENT_PLAN.md:170-176`

### SOE and sheet pile
- **GM-20** [Now] **`soe/free_earth.py` returns a value when no root is found**; return null plus a warning (G16). *was:* H E2
- **GM-21** [Next] **Caquot-Kerisel Kp0 and R(δ/φ) verified only at φ = 30°**; they drive every passive result; cross-read the DM7.2 Kp chart in the catalog. *Src:* `soe/DESIGN.md:190-200`
- **GM-22** [Later] **Ground-anchor design has no benchmark** (GEC-4 Table 4 nominal; a GEC-4-vs-PTI exhibit planned). *Src:* `soe/DESIGN.md:202-212`
- **GM-23** [Later] **SOE "Future Work"**: ACI 318 secant / diaphragm walls; wale and strut connections; beam-spring SSI; lagging; rakers and corners; seismic apparent pressure. *Src:* `soe/DESIGN.md:214-222`
- **GM-24** [Later] [owner] **SOE / sheet-pile modes**: soldier-pile (arching width, simplified toe moment; OWN-21); braced loads put the bottom span on the lowest support (conservative); a balanced-water sheet-pile mode (Das #3b). *Src:* `validation_examples/RESULTS.md:236-246`

### Deep foundations
- **GM-25** [Later] **Nordlund chart fits** (`nordlund_Kd` ignores V/V0, `alpha_t` ignores φ, `CF` linear) and GEC-12 cross-reads (Figs 7-5/7-8/7-13/7-14, Tables 7-1/9-2). *Src:* `axial_pile/DESIGN.md:42-51`
- **GM-26** [Later] **No λ (Vijayvergiya-Focht) pile method** (Das #6).
- **GM-27** [Later] **Drilled-shaft flags**: GEC-10 fs cap and RQD schedule uncited; rock socket drops the top 1.5 m even at ground surface (DS-3); sand tip independent of σ′v (DS-2, check). *Src:* w1-geotech REVIEW:194-216
- **GM-28** [Later] **Lateral and group**: φ = 40 k-table extrapolated past COM624P; pile_group coupled 6-DOF (PG-1) needs a CPGA benchmark; p-y group multipliers uncorroborated. *Src:* `pile_group/DESIGN.md:43-45`
- **GM-29** [Later] **wave_equation**: no real diesel ram cycle or GRLWEAP helmet data (V-003 blow count −60 %); quake / damping defaults unchecked vs GEC-12 Table 12-3. *Src:* `wave_equation/DESIGN.md:12-20`

### Shallow foundations, settlement, ground improvement
- **GM-30** [Now] **Terzaghi bearing-capacity factors are missing** ("testers ask for Terzaghi often"; a memo shipped with N-gamma blank). *Src:* w1-flows REVIEW:213-217
- **GM-31** [Later] **Ergonomics**: moist / saturated γ about the GWT (Das #4); Highter-Anders (Das #5); the FHWA Schmertmann variant; GEC-6 Fig 5-19 / T6-1 cross-read; vibro spacing unsourced; RAP upper zone (OWN-21). *Src:* `TIER_A_LEDGER.md:110, 133-148`

### Earth retention and seismic
- **GM-32** [Later] **retaining_walls**: the sliding display omits Pp with `include_passive=True`; a descending backfill is silently level; MSE surcharge side (unclear); representative reinforcement products. *Src:* `module_work/calc_qc/FINDINGS.md:596-602`
- **GM-33** [Later] **seismic_geotech**: SG-2 prints "f(Site Class, Ss)" when `pga` is given (`calc_steps.py:266-267`); Fpga ≈ Fa(Ss) without `pga`.

### Reliability, pavement, structural, setup, common
- **GM-34** [Later] **Reliability**: ETL 1110-2-547 examples as a second anchor; importance sampling for Pf < 1e-6; a TC304 COV re-audit.
- **GM-35** [Later] **Pavement's weakest chart reads**: Fig 3.3 (±20–25 %), Fig G.3 (±20–40 %). *Src:* `pavement_design/DESIGN.md:79-80`
- **GM-36** [Later] **Structural engines**: rectangular RC only, one bar layer a face, Mn without φ; no warping constant for rectangle / box / polygon; pynite P-Delta and plates hidden.
- **GM-37** [Later] [owner] **geo_project drift**: no anisotropic su, pore-pressure points, stabilizing piles or tension cracks; the LE builder maps only the first surcharge; no confirmed-IR ingest (OWN-22). *Src:* `geo_project/UPGRADE_PLAN.md:100-105`
- **GM-38** [Later] **Common**: Peck-Hanson-Thornburn N→φ unverified; interpreted cross-sections; gstools anisotropy hidden; unit constants (unclear). *Src:* `geotech_common/DESIGN.md:42-52`

### Calc packages and the geotech page
- **GM-39** [Now] **Ensoft-style calculation TABLES**: `calc_package.export_tables`, a `get_tables()` per module, CSV + HTML (per-depth p-y, per-slice, per-layer, iteration histories). The owner's 2026-09-11 "next train"; confirmed as the next module train 2026-10-14 (OWN-10). *Src:* FI:292-298
- **GM-40** [Next] **Specialists cannot draw or report**: foundations, earth-retention, slope-fem and seismic scopes exclude `calc_package` and `profile_figure` (`funhouse_agent/dispatch.py:122-178`); no SharePoint or email tools either. *Src:* FI:303-304
- **GM-41** [Next] **`soe` and `reliability` have no calc package**, so SOE reports are bespoke `html_to_pdf` (behind the 2026-09-15 "numbers gone" failure). *Src:* `calc_package/__init__.py`
- **GM-42** [Now] **The `list_agents` catalog is at 7,988 of 8,000 chars**: the next brief fails five tests; rebalance first. *Src:* H:1978-1979
- **GM-43** [Later] **The geotech `reviewer` sub-agent is never invoked (0/106)**; `CONSULTANT_FRAMING` ends with a dangling "Question: " (`reviewer.py:274`). *Src:* HT `geotech_agent.md:168-170, 315-358`
- **GM-44** [Later] **Playbooks**: standard multi-step workflows as data. *Src:* FI:138-149
- **GM-45** [Later] **Document work on the geotech page is token-heavy** (~2.25 M tokens in five turns): measure; turn the SOE re-run's agent errors (wrong stratum in basal heave; a real match dismissed) into suite tasks. *Src:* an SOE re-run session (2026-09-15) FINDINGS:319-352
- **GM-46** [Later] **Wave-1 leftovers** (unclear): a SoilProfile stresses tool (G3(d)); unique-name describe (G12); capped "Available" (G13); PAV-6 / PAV-7. *Src:* w1-geotech REVIEW:110-120
- **GM-47** [Later] **Validation gaps**: GEC-8 CFA (scanned); DM7 worked examples by vision; V-053 zoned dam; V-027 and Slide2 #63 (need sharper sources). Convention residuals kept on record: V-002, V-019, V-026, V-040, V-043, V-049/050, V-052. *Src:* `validation_examples/INVENTORY.md:835-918`

## 8. References and the librarian

- **REF-01** [Next] **G14: `retrieve_section`'s example id '5.7.2' does not exist** (`adapters/_reference_common.py:321`). *was:* H E2
- **REF-02** [Next] **G15: `.html.png` names** (`adapters/profile_figure_adapter.py:47-48`); a module item. *was:* H E2
- **REF-03** [Next] **Synonym-map curation** (`_query_expansion.py`, unchanged since 40f8a06): remove bare `cu` and `compaction`; fix the "negative skin friction" substring hit; prune dead CRR forms; add ~16 groups (PVD, CSR, (N1)60, p-y, t-z, CPT, SPT, DMT, PMT, UCS, RQD, geofoam, stone column, vibro, DSM, CFA, `mse`). The harmful entries ship under `auto`. *Src:* `module_work/references_review_v5.1.md:99-129`
- **REF-04** [Next] [owner] **Rescue branch 7fb6fe8**: GEC-7 ch 6–10, GEC-12 Vol II ch 9–18, summaries for GEC-6/7/12/13, `add_gec13_summaries.py` (~3,300 lines, unreviewed). Main has GEC-7 ch 1–5, GEC-12 ch 1–8. OWN-28.
- **REF-05** [Next] **Let reference QC pass**: `scripts/audit_chapter_text.py` hard-requires `summary` (gec_10 273 errors, gec_11 392, gec_13 188). OWN-33. *Src:* `references_review_v5.1.md:151-154`
- **REF-06** [Later] **No section summaries for six GECs** (0 of 670 sections; search shows bare titles); the rescue branch covers four.
- **REF-07** [Later] **Retrieval measurement**: a bigger gold set (9 cases); a live librarian eval of expansion (Foundry); top-N tuning. *Src:* refs `CLAUDE.md:260-269`
- **REF-08** [Later] **GEC-13 catalog ~14 % complete** (33 of ~238 figures).
- **REF-09** [Later] **~2,000 figures lack a `description`**, so concept search misses them; P5 lazy enrichment would chip at it. *Src:* `references_review_v5.1.md:158-160`
- **REF-10** [Later] **Lookups never written**: gec_7 T4.2b; gec_8 T5.1, Figs 5.2–5.5, T5.3 (CFA axial); gec_9 T3-1/3-2, 11-1, 8-1; gec_12 T7-32; gec_10 T10-1, 8-2/8-3; micropile T4-3/4-4; Tier 2–3 incl. gec_5 Fig 5-20. *Src:* `references_review_v5.1.md:42-73`
- **REF-11** [Later] **Data hygiene**: GEC-10 tables never cited from its text; mixed `equations` entry shapes; duplicated GEC-12 manifests. *Src:* `references_review_v5.1.md:29-31, 90-92, 161-163`
- **REF-12** [Later] **Scan finds**: `docs/UFC_4-023-03_2024_c4.pdf` is a committed 1.3 KB HTML page; legacy `esal_*` raise off SN = 5 / pt = 2.5 though `lef.py` has every table.
- **REF-13** [Later] **FHWA NHI-06-088/089 are PDFs only** (no text, no catalog), so the librarian cannot search them.
- **REF-14** [Later] [owner] **Wood Handbook local-only; FEMA P-2192 never fetched** (OWN-40).
- **REF-15** [Later] **The librarian refuses chapter text on GPT** (2026-07-31): collect a failing exchange first. *Src:* FI:275-280
- **REF-16** [Later] **Measure the librarian call budget** (raised to 10 in 3195ede, unmeasured). *Src:* `funhouse_agent/deep/limits.py:45`
- **REF-17** [Later] **Conditional retrieval**: a deterministic chart backstop (P4) only if needed; text embeddings deferred (image embeddings rejected). *Src:* CLAUDE.md P4, P6
- **REF-18** [Later] **Provenance cross-reads** (mostly in-house): Peck N→φ, vibro spacing, downdrag β, GEC-12 T7-1/T9-2, GEC-10 Fig 13-5, GEC-6 Fig 5-19, TC304 rows; with GM-21, GM-22, GM-25, GM-27. Slope wishlist sources not in the library. *Src:* `module_work/provenance_audit_other.md:150-263`
- **REF-19** [Later] [owner] **Structural future rounds**, pelicun, an AISC letter only if steel modules are built. *Src:* `module_work/STRUCTURAL_REFS_SURVEY.md:38-68`
- **REF-20** [Later] **Chart read-off on Tiny Apps** needs OWN-44 and has never been measured on gpt-5.1.

## 9. Measurement beds: Foundry, LIVE_TEST_QUEUE, the live smoke harness

- **EV-02** [Next] [owner] **Foundry brief 6 (not written)**: brief 5 groups 3–4 with repeats; Part B ×3 and Part E on both models; patch alignment (≥10 repeats, spread targets, second aspect ratio; LTQ check 7); an L1 reasoning call; coverage after CV1–CV4; GPT habits M1–M4; VIS-08; LTQ check 8 step 3 (DM7.2 Fig 4-12); the inline probe. *Src:* B5 `TRACE_REVIEW_B_AEFG.md:583-595`; *was:* H D, E5
- **EV-03** [Next] **Claude-credit items before brief 6**: the B3 Sonnet check (<$1; the fix landed in da28a69, so this now checks it); R on Haiku if needed; a Haiku smoke of the next rc. *was:* H 3c
- **EV-04** [Next] **Smoke-harness H-items**: SharePoint classifier wordings (`detectors.py:86`); exclude non-app files; recovery grading; isolation by bytes and author, with forced overlapping writes; missing detectors (truncation, output at cap, slow-turn cause, labels far from marks); ledger integrity on OneDrive; `session.py` mirroring `app.py` (D3), short sandbox root, HEAD recorded, separate worktree. *Src:* smoke REVIEWs w1–w3; *was:* H E6
- **EV-05** [Next] **Turn on prompt caching on the Claude engine path** (no `cache_control` anywhere) before more waves.
- **EV-06** [Next] **Record gaps**: LangGraph step count at `turn_end`; side-call prompt, detail, image size and kind on `model_end`; a prompt / tool fingerprint in `run.json`; per-mark truth rows. *Src:* check1 TRACE_REVIEW:503-520
- **EV-07** [Next] **The geotech eval runs a configuration users never get** (no calc sub-agent, step cap or history); align or document. *was:* H C12
- **EV-08** [Later] [owner] **Geotech eval rerun in a numpy-2 Foundry repo** (numpy 1.26 broke 17 questions). *was:* H D
- **EV-09** [Later] **Eval keys**: 38 of 106 questions unkeyed; module-derived keys cannot catch module defects, so add physical-bound checks; a `propose_role_mapping` question. *Src:* w1-geotech REVIEW:242-248
- **EV-10** [Later] **CI mock-eval subset** in GitHub Actions. *Src:* FI:286-288
- **EV-11** [Later] **Open eval questions**: what the references sub-agent did inside consults; numbers traceable to no `call_agent`; per-call overhead; a pinpoint citation spot check. *Src:* HT `geotech_agent.md:335-349`
- **EV-12** [Later] **Read `module_work/harness_theory/` open questions before changing a harness.** *was:* H "Open questions"
- **EV-13** [Now] **The next suite run measures tonight's default-page changes** (2026-10-12): the 40-call budget whose last call answers (DR-03; GEOTECH_REVIEW_MAX_MODEL_CALLS=off restores the old cap), the helper with the reading rules (DR-02), the zoom tool's wording (DR-01) and the markup check against the printed thing (DR-14), against brief 5's baseline (Sol 39/42). Read every run in full. *Src:* ae50842, da28a69

## 10. Infrastructure, releases, dependencies, repo hygiene

- **INF-01** [Later] [owner] **Stray untracked root files** (nothing the tracked versions lack): `A2_CONTEXT_DESIGN_copy.md`, `A7_INDUSTRY_MEMO_copy.md`, `APP_PLAN_copy.md`, `FOUNDRY_APP_PLAN_copy.md`, `V5.4_PLAN_copy.md`, `FOUNDRY_setup_copy.md`, `provenance_audit_other.md`, `provenance_audit_slope.md`, `CONSOLIDATION_PLAN.md.local-bak`, `suite_trial.json.md` (+ `suite_trial.json`: archive). Also `docs/geotech_eval_20260713.md`. OWN-27.
- **INF-02** [Next] **`check_pins.py` against the Nexus list** (after OWN-35).
- **INF-03** [Next] **Agent-stack caps** (`deepagents<0.8`, `langchain<1.4`, `langgraph<1.3`): raise only after the deep and webapp suites pass; read the canary first (OWN-33).
- **INF-04** [Later] **The `openai<3` pin comment is wrong** (httpx2 arrives via langsmith; `pyproject.toml:46-50`); does the pin still earn its place?
- **INF-05** [Later] **Retire the v1 `GeotechAgent` loop** (`agent.py`, `react_support.py`); with it the uncapped `_render_pdf_page` caller (`agent.py:732`), `write_docx` missing from native tools, `reviewer.py` without figure tools, `test_scenarios.json`. *Src:* `docs/V5.0_TODO.md:72`
- **INF-06** [Later] **Delete `foundry/` and `foundry_test_harness/`** after excising TestFoundry from 7 wrapper suites; also refs `agents/`.
- **INF-07** [Later] **Is master CI green?** ("should self-heal … (VERIFY)", H:2585).
- **INF-08** [Later] [owner] **Leftovers**: gitignored `build/`; old release worktrees and `.claude/worktrees/agent-*`; branches app `hotfix/5.12.1-structural-extra`, planlens `fix/advice-names-no-missing-tools` (shipped as 0d9787b), refs `origin/feature/reference-query-expansion` (merged). OWN-27.
- **INF-09** [Later] **Rebuild the private Harness Map** (outside the repos; last built 2026-09-29).
- **INF-10** [Now] **Run the geotech module suites on a fresh resolve** (numpy 2.5.4, plotly 7.1.0, deepagents 0.7.13, langchain 1.3.18 on 2026-10-12): TA-02 ran only what pandas and Streamlit reach. Recipe: WSL Ubuntu, `~/.local/bin/uv`, Python 3.12, repo on PYTHONPATH, GIT_OPTIONAL_LOCKS=0 (Smart App Control blocks fresh DLLs on Windows). Also `report_ingest/tests/test_cluster_scoring.py::test_a_workspace_out_dir_runs_rather_than_raising` assumes `/Workspace` can be created (fails on Linux as a non-root user).

## 11. Long-term programs and parked ideas

- **LT-01** [Later] **Reliability program** (the owner's prime interest; "the true answer is a distribution"): FORM/SORM on any module's limit state, system reliability, random-field FEM (gstools + fem2d), LRFD φ calibration, target-β design. Untouched since June. *Src:* FI:177-184
- **LT-02** [Parked] [owner] **6.0.0 single-namespace restructure** (name-squat risk); plan ready, needs a dedicated session. *Src:* `module_work/V6.0_RESTRUCTURE_PLAN.md`
- **LT-03** [Parked] [owner] **Repo slimming** (a Department reviewer's "millions of tokens"): inventory review; INF-06; drop result JSONs; trim CLAUDE.md (169 KB) and HANDOFF (222 KB) into history files; a home for `module_work/`. Parked 2026-09-22; both have grown. *Src:* FI:26-115
- **LT-04** [Parked] [owner] **Reference-wiki integration** (the owner's private library as a retrieval layer; parked 2026-07-13). *Src:* `module_work/V5.4_PLAN.md:308-313`
- **LT-05** [Later] **Sample-calc defect detector, next targets**: SCDOT, USACE EM appendices, remaining NHI manuals, program comparisons; corpus frozen at 27; 4 real defects found in July. *Src:* FI:3-18
- **LT-06** [Parked] [owner] **A toolkit MCP server over the dispatch layer**; waits on OWN-54, PL-15. *Src:* FI:398-414
- **LT-07** [Parked] [owner] **A full modelling GUI** (browser LE editor, 2–4 weeks).
- **LT-08** [Parked] **ML for scales and scans**: a tick-label digit reader; chart-to-table benchmark; learned combiners; PyTorch only on a measured gap. *Src:* FI:466-518
Long items kept in their own sections: DR-08–DR-13, GM-39, VIS-11, VIS-20, TA-16, REF-17, OWN-21.

## 12. Stale docs to fix

- **DOC-01** [Next] **HANDOFF**: §1–§6 (~2833-3088) not marked HISTORICAL: "(private)" repo, master v5.6.0, the `v5.1-todos` worktree, V5.4_PLAN current, eval 97/100.
- **DOC-03** [Next] **CLAUDE.md**: "this file is 73 KB" (169 KB); :251 "committed, not released" (0.11.0); :1237 missing plan (DOC-15); :1819 "fully QC'd"; :1872 "3529 tests"; :1909 "ten tools" (15); :2109-2112 footing method (done 0aea590), "~22 more figures" (done 3b25e0e), chapter text "uncommitted" (1176620; rest REF-04).
- **DOC-06** [Next] **`LTQ`**: :20 "check 5 only, after 5.32.1 is released"; "Done" omits brief 5; setup :36-42 installs 5.32.1.
- **DOC-07** [Next] **`HT/README.md:55-56`** marks W3 and brief-4 fixes "uncommitted"; silent on brief 5 and the smoke-wave changes.
- **DOC-08** [Next] **`RH:141-171`** "Next" stale; :151 points at "Open to-dos after 5.32.0".
- **DOC-09** [Later] **Counts and installs**: `README.md:5` "30 analysis modules · 21 digitized references", :60 "48 standalone tool agents"; `docs/funhouse_agent_guide.md` deleted refs (239-256), hvsrpy / swprocess (515, 560), `[deep]==5.0.0` (606); `webapp/README.md:23, 251-252`; `funhouse_agent/DESIGN.md:25` "50 adapter modules"; `SKILL.md:69` "~8,600 tests".
- **DOC-10** [Later] **User manual PDF is v5.3** (30 releases behind; old SOE example); `docs/V5.1_SUMMARY.html`; deferred webapp-guide screenshots.
- **DOC-11** [Later] **Module DESIGN docs**: `slope_stability/DESIGN.md:13` "nails.py, currently disconnected", :97 "no adapter yet", :359-367 Spencer "APPROXIMATE"; `geotech_common/DESIGN.md:30` "no adapter yet"; `seismic_geotech/DESIGN.md:15-21` (both checks closed); `lateral_pile/DESIGN.md:53` "machine precision" (test rel=5e-3); `pile_group/DESIGN.md` lacks CPGA / EM 1110-2-2906 refs; `subsurface_characterization/DESIGN.md:97-111` DIGGS out of date.
- **DOC-12** [Later] **Validation ledgers**: RESULTS V-038 "flip owner-gated" (decided); V-052 "assumed 2.5:1" (V-052b); `TIER_A_LEDGER.md:109` qL row "OPEN" (d9dab7a).
- **DOC-13** [Next] **planlens docs (public)**: README silent on 0.12–0.13 (scales, numpy `find_like`, FIPS); :127-129 raster extra; :552-567 OCR recipe wrong on FIPS hosts; `document/DESIGN.md:16-35` map stops at 0.4, no scales section, :1294 vs :1301 two vs four sheets, :669-670 stale Azure line; `ir/DESIGN.md` 13 tools (15).
- **DOC-14** [Next] **geotech-references docs (public; the PyPI page)**: README 559 functions, deleted modules, ufc_pavement "Airfields", gec_10 NHI-10-016, 16 modules missing, "all 14 references"; refs CLAUDE.md 3,529 tests (5,833), expansion "uncommitted", ufc_pavement "pending audit"; `scripts/README.md`; `_retrieval_db.py:26`; pyproject description; `reference_work/BOARD.md` abandoned.
- **DOC-15** [Next] **A missing plan file is the cited plan of record**: `delightful-swinging-sky.md` (CLAUDE.md:1237, H:2555, `DRAWING_INTELLIGENCE_DESIGN.md:15`); recover or repoint.
- **DOC-16** [Later] **`report_ingest/README.md`**: "counts from the parent's own logs", "calcs recorded as skipped", "narrative one call", the workflow table, "parent keeps appended_report".
- **DOC-17** [Later] **Pointers**: `pavement_design/DESIGN.md:157-163` vs CLAUDE.md "PARKED"; `.claude/agents/structural-calc-specialist.md:69-72` "not yet onboarded"; `webapp/app.py:793-794` → TA-07; FI "VISION EFFICIENCY" lacks brief 4's numbers.

## Done since the last sweep

Older docs still list these as open. Delete this section at the next sweep.

- **GM-05** (SRM factor of safety depended on the mesh): two causes. (1) `generate_slope_mesh` kept the whole Delaunay hull, so the notch above the toe was meshed as sliver "soil" that propped the face (25 % extra area on GL99 Ex 1); triangles outside the section are now dropped. (2) A residual that stalled while the displacements stood still was read as collapse; failure now needs non-convergence with displacements still growing (Griffiths & Lane's definition). Every benchmark now varies ≤ 1.7 % across meshes once ≥ 4 elements span the slope and lies within +2.5 / −5.4 % of limit equilibrium: the 10 m 2:1 slope 3.000 / 1.803 / 1.147 on three meshes → 1.909 / 1.897 / 1.884 (LE 1.887); GL99 Ex 1 1.331 on every mesh (published 1.4, LE 1.368); ACADS 1(a) 0.949 (published 1.00). Results warn plainly on too few elements across the slope, an unmeshed or too-thin layer, a dry cohesionless face above tan φ′/tan β, a range that did not bracket failure; ν ≥ 0.5 is refused; `mesh_check=True` re-runs 1.5x finer and reports both.
- **GM-06** (undrained locking): measured, there is none: GL99 Ex 4 gives 1.419 at ν 0.30 through 0.4999 (Taylor 1.47, LE 1.486). The old +19-26 % was the soil-filled notch. B-bar was built, measured worse (Ex 4 1.33, Prandtl Nc 4.52 against 5.14) and removed; no ν warning, since ν does not move the answer.
- **GM-08** (`analyze_staged` ignored `n_steps`): a stage now ramps from its start state to its full load over `n_steps` (a 360 kPa footing: one step finds no equilibrium, 10 steps settle 68 mm, matching ten 36 kPa stages to 0.1 %). Elastic stages unchanged. Also fixed in passing: the Duncan Ex 1 test compared a deep-foundation model with a shallow-base publication (rewritten: SRM 0.883 = non-circular Spencer 0.883); a calc-steps table title no longer says "Judgment". fem2d suite 461 passed.
- **GM-53** (the critical-surface search ignored reinforcement): every trial surface in every search kind (grid, entry/exit, random and DE noncircular, PSO, weak layer, the toe and infinite-slope families, fixed-surface FOSM / Monte Carlo, all three rapid-drawdown stages) now carries nails, anchors, geosynthetics and stabilizing piles exactly as `analyze_slope` does; the result lists the reinforcement crossing the critical surface. A search rejects a reinforced surface on which Spencer / GLE does not converge (the old fallback dropped the reinforcement and could win). Toe circles may touch a rigid base but not dip under it. ACADS 1(a) + 200 kN/m geosynthetic 0.990 → 1.123; Slide2 #86 with tiebacks now reproduces (Bishop 1.612, Spencer 1.606, GLE 1.606 against Slide2 1.629 / 1.620 / 1.622 and Duncan & Wright 1.61); #85 GLE 1.581 against 1.575 (its Bishop is GM-59). V-043's 44-56 kN/m is a single-surface quantity and its test stands. Runtime on reinforced slopes +11-41 %; unreinforced unchanged.
- **GM-54** (drawdown searches missed toe circles): `search_rapid_drawdown` runs the toe-circle and infinite-slope families with the drawdown FOS and respects given ranges; a free-draining c′ = 0 shell gives the planar closed form (0.942). At 2H:1V Pilarcitos now finds the shallow face slide classic under rapid drawdown (DWW 0.860, LK 1.017); the published values describe the observed deep slide, so `test_v052_pilarcitos_method_ordering_is_slope_invariant` runs with `infinite_slope=False` (explained in the test and INVENTORY). Automatic-range drawdown searches +40-60 % runtime.
- **GM-55** (the slope calc package dropped inputs silently): `_build_slope_geom` builds layers through the slope adapter's `build_soil_layer` (unknown keys refused; ru, strength_model and the SHANSEP / Hoek-Brown / anisotropic / R-envelope fields pass through), the report checks its variables, and the response and report show the governing surface, each family's minimum, the reinforcement crossing it, the infinite-slope checks and the search warnings; `geo_project` stores a planar critical surface by its points instead of xc = yc = R = 0. The fixed-circle `slope_stability_package` is strict about layer keys too now.
- **GM-14** (c′ = 0 slopes: the circle search missed the planar failure): wherever c′ = 0 soil forms a face the search now also scores shallow planar surfaces parallel to it (Janbu simplified, 0.1-0.4 % above the infinite-slope closed form with ru and kh). #86 without supports: 1.566 → 0.944 (true 0.942). Duncan Ex 2/3/4 now governed by the infinite slope (1.682, 1.197, 2.347 against closed forms 1.678, 1.194, 2.344). Results name the governing surface (`governing_surface`, `candidates`, `infinite_slope_checks`) and warn plainly. `slope_stability/search_supplements.py`.
- **GM-13** (small toe circles under-sampled): circles through each slope face's toe are scored after the main search (Bishop scan, the best three re-optimised in the chosen method). Steep φ = 0 case 2.784 → 2.286 (true 2.287); coarse-grid tests now match fine grids (Duncan Ex 6 1.716, V-029 1.130). Every number that moved went down. V-032 (Slide2 #61) re-pinned to 1.315 with brackets: Slide2's own printed circle gives 1.324 here and 1.3244 by an independent hand Bishop, so the published 1.366 is not reproducible from the stated inputs (recorded in INVENTORY/RESULTS). Runtime +0-12 %.
- **GM-52** (found and fixed 2026-10-14: the slope adapter silently dropped unknown keys inside soil layers, so `"c": 10` became c′ = 0; seen live in smoke wave 1, where an agent's Bishop check reported 0.97 against the true 1.89): `check_nested_keys` now refuses unknown keys in layers, surcharges, nails, anchors, geosynthetics, stabilizing piles and the probabilistic variable / correlation specs, naming the item and a did-you-mean (`c` → c_prime or cu, never guessed). All 13 shipped slope parameter sets still pass.
- **GM-48** (downdrag as the owner asked, OWN-01): every method computed on every run and tabulated side by side, each named as its reference names it, with its source (`to_dict()["neutral_plane_comparison"]`, `summary()`, the calc package); one plain sentence gives the spread (`methods_range`); the headline is UFC 3-220-20 §6-7.4's procedure (`headline_method`). The coined wording is gone from results, adapter, calc package and DESIGN, and a test keeps it out. Skin-friction stress defaults to final: GEC-12 §7.3.6.1 gives no rule (its only remark is that stress changes after driving move the neutral plane), so UFC §6-7.4 step 2 governs; both quotes are in DESIGN.md. The geotechnical check below the neutral plane therefore uses the final-stress resistance too. Pile head settlement is now the settlement at the neutral plane plus the elastic compression above it (UFC Fig 6-19); before, it left out the downdrag itself. B-5.5 still matches (47.70 / 50.14 / 52.48 ft against 48 / 50 / 52); V-004 53.61 ft / 490.1 / 289.1 kips with the pile weight in, 54.00 / 486.6 / 285.6 without (published 54 / 486 / 285). Follow-ups GM-49 to GM-51, OWN-58.
- **GM-01** (the ~4 % mismatch between the neutral-plane search and the reported drag force): three integrations of the same friction profile disagreed; now one exact integral of the piecewise-linear profile, and the neutral-plane load equals toe plus positive friction to rounding (pinned by `TestQuadratureConsistency`).
- **INF-11** (post and project names out of the shipped code): 13 comments reworded in `webapp/`, `funhouse_agent/` and `soe/`, plus two DESIGN notes; the "Department's ..." remarks made neutral. Guard: `webapp/tests/test_public_text.py` walks the wheel's packages and fails on any guarded name (stored as hashes, so the test does not repeat them). Not covered: the report-ingest answer vocabulary (OWN-25) and test files (not shipped; repo visibility is OWN-26).
- **OWN-01** (downdrag defaults): the owner, 2026-10-14: "I think presenting all of them is fine ... For stress, see what FHWA GEC 12 says about initial vs final stress and go with that; if it's silent, use final stress like UFC." Also: the card's coined wording made no sense to them. The work is GM-48.
- **OWN-02** (fem2d staged movements): confirmed, measure from the end of stage 0: "this is standard stuff. Don't measure the gravity stage." Default as built in 4bf710c.
- **OWN-10** (calculation tables): "Yes, the next module train"; GM-39 is in Now.
- **OWN-55** (release timing): hold: "We can use the prompter key in Funhouse this month. We can play it by ear if anything warrants a release before then."
- **OWN-57** (published PyPI files): the owner deleted the `geotech-report-extraction` project from PyPI on 2026-10-14 (confirmed: its JSON API returns 404; the account holds three projects). App releases stay; INF-11 cleans master, including the "Department's ..." remarks.
- **DR-01** (the default review page's zoom tool named tools the page lacks): ae50842. Unreleased (OWN-55); measured by EV-13.
- **DR-02** (the default page's helper now carries the reading rules, read-only tools and a 14-call budget): ae50842.
- **DR-03** (the default page's 40-call budget whose last call answers; `GEOTECH_REVIEW_MAX_MODEL_CALLS=off` restores the old cap): ae50842.
- **DR-14** (markup check measured against the printed thing; a rename keeps its verdict): da28a69.
- **TA-01** (a refused key said plainly, one retry on a 401): 0562ddf.
- **TA-17** (`.env` exported on a local run, so env-only settings work on dosdev): 0562ddf.
- **TA-02** (suites on CfA's fleet pins): run 2026-10-12 in WSL (Python 3.12, pandas 3.0.2, streamlit 1.62.0). Found python-ags4 blocks the install; fixed on master (a229090). Follow-ups OWN-55, INF-10.
- **DOC-04** (`docs/DATABRICKS_INSTALL.md` current for 5.33.0): 24b46df.
- **DOC-05** (`TINYAPPS.md` current; the retired TinyApp checklists point at it): 24b46df.
- **OWN-04** (PyMuPDF's AGPL licence on Tiny Apps): the owner, 2026-10-12: "We're good to use pymupdf. The funhouse sdk uses it, so we can too."

- **EV-01** (a rehearsal smoke wave at the gpt-5.1 768 px budget): closed by the owner, 2026-10-12, on the Backlog Desk. Testers get one GPT-5.6 Sol-or-higher deployment, not gpt-5.1. Sol already ran the suite at full resolution on Foundry (5.32.0rc3; B5 39/42), the app measures the model on first use and planlens maps `gpt-5.6` to full-size budgets, so a Claude stand-in wave would cover no gap.
- **OWN-03** (three non-government standards PDFs in the public refs repo): the owner kept them, since copies are openly available online. A provenance note is in the refs README (refs 3e582d8).

- **DOC-02** (HANDOFF §0a stale lines: 3c "NOT written yet", 3e "a builder is on it", the E2 label) and DOC-01's "Last updated" line: fixed in e329b22.

- **Releases and trains**: W1–W4 (5.33.0); brief 4 fix list A–M (cc36945, planlens cbca2ec); brief 5 groups 1, 2, 5 (233a40e, b6d7ffd); smoke-wave fixes w1–w3 incl. F2–F8 polish (3195ede … 83a77f4; planlens f679daf, f92541d); "release planlens then the app" (GO 1); the activity-log lock (6c81cc5); 9a `find_like` without OpenCV; 9b locations (cf43c90, planlens 08a1d53); 8a(iv) candidates resolved (W2, W3, W4; the Atterberg gate dropped); 8a(v) GEC-12 Fig 7-15 (refs d9dab7a, app 7d8771e); 8a(vii) brief 5 ran; 8a(viii) speed and mismatch.
- **Document Review**: Phase 0 suite (42 tasks); Phase 1 small fixes; M0 runs; the two scale tasks, N1, N5 (39716a8); peer-session prompt notes (ad83eb2); feedback becomes suite tasks (`extra_tasks=`); LTQ checks 1–4, 6, 8 (steps 1, 2, 4).
- **Vision**: code-on-image and a zoom inside `read_reference_figure`, superseded by `code_reading` (c0ecc9f) and `render_region` on the returned page.
- **planlens**: advice wording (0d9787b); survey Tier 1 (0.4.0); Word in and out; interactive `plot_data`; a live run on a real review question.
- **Hosting**: two pages on Databricks (b289d83), links behind the driver proxy (used 2026-10-06); per-owner restore; render cap on the deep path; the 5.11.x picker, uploader, streaming, budget sidebar.
- **Report ingest**: live checks a–e (Foundry runs 1–3; vote locally 6143a9d); DIGGS without pydiggs (499e1cf); WP5 calc reader; floor / model in RESULTS (cc36945); R34; folded hole ids (5e70823); recovery as length (46be026); R21 crash contained (1ef959b) and DIGGS round trip (f61505d), both unmeasured (RI-09).
- **Modules**: downdrag CGPR family; soe free earth (5.17.0); Das #1–#3a; fem2d footing method (0aea590) and calc figures; did-you-mean redirects (abbe1ad); `pore_pressure_points` in the adapter; SoilProfile → slope adapter; M-O anchor and AASHTO site factors (wave 7); 5.5.0 deferred items; consolidation #6, #7 (f7639e5), #9.
- **References**: G3, G13, G22 (3195ede); G4 and the FTS hyphen crash (5569c5b); G20 (eef1264); ufc_expansive figures (3b25e0e); all 32 catalogs page-accurate; ufc_pavement rebuilt (47be106); `auto` expansion default (fe7fe9d); part of the GEC chapter text (1176620).
- **Other**: `feature/subagent-feedback-reference-pdfs` merged (5c1058b); the calc sub-agent's file reads (planlens document tools not given).
