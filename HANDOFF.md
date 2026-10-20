# HANDOFF — GeotechStaffEngineer (current state, read this first)

**Last updated: 2026-10-15** (the 2026-10-14/15 train on master; Now is core improvements; open work is in `BACKLOG.md`). This is the authoritative handoff for a fresh LLM
session (any model). The older `HANDOFF_2026-06-14.md` is kept only for the
detailed Phase-E history; this file supersedes it.

---

## 0a-current. PICKUP LIST (2026-10-15, supersedes everything below)

### Where things stand (5.33.0 released 2026-10-09; master through 2026-10-15)

**0. 5.33.0 RELEASED 2026-10-09 with planlens 0.13.0 and geotech-references 1.4.1.** The owner's word, 2026-10-09: "go ahead and release when you're done".
- **What it is for:** the Document Review page goes to OTHER testers on Tiny Apps.
  - The owner first runs a dosdev round (recipe in `tinyapps/TINYAPPS.md`).
  - Then CfA gives a published address.
  - The owner takes the remaining CfA questions to the CfA team the week of 2026-10-12:
    - per-tester Prompter keys: Key Vault `PROMPTER-API-KEY--<TESTER>`, `CORP\jdoe` → `PROMPTER-API-KEY--CORP-JDOE`; set `PROMPTER_SHARED_KEY_FALLBACK=1` until every tester has one;
    - the SharePoint app registration: the four `GRAPH-*` / `SHAREPOINT-SITE-URL` values and the `Sites.Selected` grant on the team site;
    - the published address for `corsAllowedOrigins`;
    - confirming the identity header (each tester's sidebar shows "Signed in as <their name>").
- **How it was hardened:** live smoke waves (section 3e below) through the app's real turn path with Claude standing in for GPT. About $45 of the owner's API credit was spent, every run was read in full, and wave 3's verdict was GO.
- **Tags:**
  - planlens `v0.13.0` (f92541d), 1,686 tests on the clean tree;
  - geotech-references `v1.4.1` (c53e587), 5,833 tests;
  - app `v5.33.0` on branch `release/5.33.0` (3ec1dc4). **Release gate on the exact tree, with planlens 0.13.0: 14,624 passed / 34 skipped / 0 failed.**

    | Chunk | Result |
    |---|---|
    | analysis modules | 1,555, and 1,708 with 24 skipped |
    | fem2d | 274 + 135 = 409 |
    | report_ingest + foundry harness | 1,505 |
    | webapp + live_smoke harness + validation_examples | 822 |
    | funhouse_agent | 2,792 with 10 skipped |
    | geotech-references | 5,833 |

    The wheel was checked: `planlens>=0.13` and `geotech-references>=1.4.1`; the new modules are present; no `module_work`, `raw/` or `tinyapps/reference` files. The release wheels are in `../foundry_handoff/`.
- **Unverified live on GPT:**
  - vision on the rollout model (one GPT-5.6 Sol-or-higher deployment, owner 2026-10-12; the app measures it on first use);
  - latency against the new per-kind caps;
  - real budget and rate-limit responses;
  - testers on separate keys;
  - links behind the IIS proxy.

  First-hour check: `module_work/live_smoke/runs/w3-confirm-sonnet/REVIEW.md`.
- **Owner's answers, 2026-10-14** (BACKLOG "Done"): downdrag shows every neutral-plane method side by side in the references' own terms, with final effective stress by default (GEC-12 is silent; UFC 3-220-20 §6-7.4 step 2); fem2d measures construction movements from the end of stage 0.
- **Spend left:** $24.83 of the $80 of API credit for smoke work: the ledger (`module_work/live_smoke/spend_ledger.json`) stood at $55.17 on 2026-10-15 after waves w4–w8b (w7 / w7b measured the figure helper, w8 / w8b plotting code in the locked box), against the account limit of $100; the promo credit expires 2026-11-05. The owner's $20 is reserved for touch-up testing together.
- **Master after 5.33.0: the next train, unreleased.** The owner: "that's stuff that can just roll into the next release".
  - **a282d48, downdrag**, reworked 2026-10-14 after the owner's answer. Every cited neutral-plane method (UFC 3-220-20, GEC-12, CGPR #56) is computed on every run and tabulated side by side, each named as its reference names it and with its source; the methods table, with sections and pages, is in `downdrag/DESIGN.md`. The headline stays UFC's procedure; the spread over the methods is stated in one plain sentence. Default skin-friction stress is final effective stress. UFC App. B-5.5 and V-004 are reproduced.
    - The owner's direction: "Downdrag has many methods and it should be open to multiple assumptions", and on 2026-10-14: "presenting all of them is fine".
  - **4bf710c, fem2d.** Per-stage delta, cumulative and since-reference movements (`reset_displacements_after`, default after the gravity stage). Phasing decides self-weight. Three latent staged-analysis bugs fixed.
    - The owner's direction: "Construction phasing will determine output."
    - Stage 0 as the gravity stage: confirmed by the owner, 2026-10-14. Open: θ=0.5 ringing at drained boundaries (GM-07). `n_steps` is honoured since 2026-10-14.
  - **The references sub-agent.** The GSE main agent's `references` sub-agent was already wired to handle method and parameter calls (owner, 2026-10-09). Modules must not hide method choices; the plain `judgment` data in results makes a choice visible to it.

**1. 5.32.1 RELEASED 2026-10-08 with planlens 0.12.0.** The owner said, the night of 2026-10-07: "Ok to release however you see best".
- **planlens 0.12.0:** tag `v0.12.0` on branch `release/0.12.0` (c58f7ed), cut from 08a1d53, which Foundry measured as 0.12.0rc1. 1,500 tests passed; no dependency change.
- **App 5.32.1:** tag `v5.32.1` on branch `release/5.32.1` (a8114a2). That is a4ef417, which Foundry measured as 5.32.1rc3, plus the version, `planlens>=0.12` and its docs. Released as measured; the activity-log lock goes in the next release.
- **Release gate on the exact tree, with planlens 0.12.0:** 13,790 passed / 38 skipped / 0 failed.

  | Chunk | Result |
  |---|---|
  | analysis modules | 1,091, and 2,108 with 28 skipped |
  | fem2d | 399 |
  | funhouse_agent | 2,268 with 10 skipped |
  | webapp + validation_examples + foundry harness | 768 |
  | report_ingest | 1,363 |
  | geotech-references | 5,793 |

- **The wheels are in `../foundry_handoff/`.** The owner installed the 5.32.1 wheel on Funhouse on 2026-10-08, before the tag, for real work and to reload an old geotech conversation. The Nexus mirror delivers PyPI a day or two later.
- **Master** was then at version 5.32.1 with the same pin, carrying the train released as 5.33.0.

**What 5.32.1 carries**, all since 5.32.0:
- **The markup check measures size** (cc697e6). Live check 1 on 2026-10-07 found 5.32.0's "drawn close" question rejecting correct circles on GPT-5.4.
- **The SharePoint copy fix** (4334fce).
- **The model's own words in `activity.jsonl`** (f405765).
- **`find_like` without OpenCV on FIPS hosts** (planlens d7d6657, app ba12760).
- **The 2026-10-06 geotech-session fixes** (3d1cd44): SharePoint failures said for what they are, a per-turn note of the conversation's files, `subsurface.write_diggs`, turn details on by default, and the past-conversations list cached per page.
- **The location build** (cf43c90, planlens 08a1d53): pixel boxes converted in code, a 2,048 px cap, padded zooms, marks refused from wide views, quotes on their printed row.

**How it was measured: Foundry brief 4, read in full** (`module_work/review_eval_results/2026-10-07_foundry_5.32.1rc3/TRACE_REVIEW.md`; REVIEW_HARNESS run 6; MEASUREMENTS "FOUNDRY RUN 2").
- **The location fix held** on GPT-5.4 and Sol.
- **Circles:** 2/3 passes on GPT-5.4 (0/3 on Funhouse before), 3/3 on Sol.
- **Suite:** 37/37 on Sol, with no task changing outcome against run 4.
- **Report-ingest narrative:** 83/85 %.

**2. The 5.33 train as it stood on 2026-10-08 (released in 5.33.0; switches OFF).** Plan of record: `module_work/SCALES_COVERAGE_CROSSCHECKS.md`.
- **W1** (499e1cf): the DIGGS 2.6 schema is bundled and checked with lxml on every host, so pydiggs is no longer needed.
- **W2** (438430f): `write_diggs` runs report ingest's reconciler and returns `cross_checks`.
- **W3, visual scales.**
  - Design: `module_work/VISUAL_SCALES_DESIGN.md`; the owner accepted all nine recommendations.
  - Steps 0–5 BUILT in planlens main a3f2a1d: `find_scales`, `measure`, `log_grid` on scans.
  - Harness: `module_work/scales_harness/`; GATE PASS.
  - Private spot check: 32/32 stratum lines, median 0.007 m.
  - App side (steps 6–9) NOT built.
- **W4** (6c81cc5, c9d5c8d): the coverage ledger, gate and `document_coverage`, and the DRAFT report-review checklist, behind `GEOTECH_COVERAGE` / `GEOTECH_REVIEW_CHECKLIST`.
  - Step limits under the switch: an ordinary turn keeps the app's cap; an extraction turn may run to 150.
  - Design note: `module_work/COVERAGE_AND_CHECKLIST.md`.
- **Also on master:** the activity-log lock (6c81cc5).

**3. The work of 2026-10-08/09 (record; what is still open is in `BACKLOG.md`):**
- **3a. The brief 4 fix list**, TRACE_REVIEW §7, items A–M:
  - comments judged by what they point at;
  - a geometric "encloses";
  - a `dpi` floor;
  - forgiving markup fields;
  - tile-only finds;
  - `find_like` flood stop;
  - `markups_point_at`;
  - reasoning summaries;
  - report-ingest `floor_alone`/`model_alone`.

  BUILT 2026-10-08 (app cc36945, planlens cbca2ec; lead re-gated). Nearly
  every item changes what an agent sees, so all of it goes into brief 5;
  TRACE_REVIEW §9 lists what to measure. Note J: GPT-5.4 baseline's
  brief-4 markup now fails the new `markups_point_at` check (2/3).
- **3b. W3 app side (steps 6–9):** steps 6–8 BUILT and committed 2026-10-08,
  lead re-gated (`module_work/VISUAL_SCALES_DESIGN.md` §12):
  - `measure` and `log_grid` on the geotech page and the Document Review
    legacy and lean builds (not the minimal one), with the label-reading
    side call;
  - `read_reference_figure`'s second voter (`code_reading`);
  - report ingest's voters behind `GEOTECH_INGEST_VISUAL_SCALES`, OFF;
  - planlens: the speed fix and `log_grid(values=)`, unreleased on main.

  Step 9's live checks are `module_work/LIVE_TEST_QUEUE.md` check 8.
- **3c. Foundry brief 5: BACK and READ IN FULL 2026-10-08.**
  - **The readings:** `module_work/review_eval_results/2026-10-08_foundry_5.33.0rc1/TRACE_REVIEW.md`, plus four slice files. Also REVIEW_HARNESS run 7 and MEASUREMENTS "FOUNDRY RUN 3". Raw traces sit in `module_work/field_feedback/2026-10-08_foundry_brief5_5.33.0rc1/raw/`.
  - **Verdict:** all four switches stay OFF and nothing is released from rc1.
  - **The fix list** is TRACE_REVIEW groups 1–6. **E5 is DONE on master** (refs `d9dab7a`, app `7d8771e`): GEC-12 Fig 7-15 was re-measured (the released table is +41 % at φ 30°). Owner's rule: the reference refuses outside 30–43.75°, and Nordlund completes with a warning on `AxialPileResult.warnings`. It reaches users with the next release.
  - **Claude API credit** (owner, $100 promotional, expires 2026-11-05): spend it only on B3, on R if orientation needs a vision call, and on a Haiku smoke run before brief 6. Everything else replays offline from the traces.
  - The brief was `../foundry_handoff/AI_FDE_BRIEF_5.md`, with test wheels 5.33.0rc1 and planlens 0.13.0rc1 from master 39716a8 and main c5bdb8d.
  - Its original contents:
  - 3a's fixes and 3b;
  - the coverage arms `("baseline", "coverage", "checklist")`;
  - the report-ingest rescore questions (logs `index`, gradation links);
  - **W3, from LIVE_TEST_QUEUE check 8:**
    1. GPT-5.4 and Sol reading numbered label crops;
    2. whether the agents use `measure` from its description alone, and how
       often a measurement from a real zoom snaps rather than lists. Two
       suite TASKS were needed for this (written since, 39716a8): a log depth
       on a scanned log, and a distance on a plan (synthetic fixtures with
       truth, `planlens.testing.visual_scale_fixtures`);
    3. `code_reading` on real reference charts (GEC-12 Fig 7-15, decision
       8; DM7.2 Fig 4-12);
    4. report ingest with `GEOTECH_INGEST_VISUAL_SCALES` off and on, in
       separate `out_dir`s.
- **3d. The owner's checklist session, the week of 2026-10-12** (BACKLOG OWN-31).
- **3e. Live smoke before the Tiny Apps tester deploy (2026-10-08/09).**
  - **The owner's goal:** the Document Review page goes to OTHER testers on Tiny Apps in the week of 2026-10-12, and it is the primary focus. The owner gave $80 of the $100 promotional Anthropic API credit for plumbing smoke tests, with $20 kept for touch-up with the owner. The credit expires 2026-11-05.
  - **The harness:** `module_work/live_smoke/` (`run.py` docstring). It runs 55 flows (F01–F53; F29–F50 are Document Review; F52 / F53 draw report figures, and F52h / F53h repeat them with the figure helper on, a flow's own `env` switches restored after it), the geotech questions and the suite tasks through the app's real turn path, with Claude standing in for GPT. It checks each turn for files outside the conversation folder, links, cards, mirror gaps and tool errors. The spend ledger sits beside it (gitignored).
  - **Runs and reviews** are in `module_work/live_smoke/runs/<wave>/` (gitignored), each wave with a REVIEW.md from an Opus full read:
    - w1 (Haiku): flows, geotech, suite;
    - w2a (Sonnet): review flows;
    - w2b (Sonnet): 14 of the new flows;
    - w2c (Sonnet): the 8 untested flows plus re-runs.
  - **Fixed and committed** (3195ede, e01db59, cada395, 7a8b979, 9445781, 8ef04ad; planlens f679daf):
    - per-user isolation: SharePoint tools, disk reads, one document toolkit per conversation, the working folder bound per turn, owner = unique key;
    - the final answer only; files and links with conversation-relative names; the SharePoint link points at the mirrored copy;
    - Word/Excel in and out (`office_text.py`, `write_xlsx`); `plot_data` on both pages; `calculate`;
    - tool errors never end a turn; an output cap of 32k with cut-call detection; auto-continue only on real stops;
    - vision: concurrent tiles, retries, fairness between conversations, an already-read note;
    - markups on rotated pages (needs the planlens release) and safe re-marking;
    - module fixes: the sheet-pile water table at 0 m, fem2d consolidation units and load-only transient, the downdrag neutral-plane basis, and others.
  - **Owner's judgment:** given 2026-10-14 (item 0 above).
  - **Before the testers arrive:** release (on the owner's word) planlens (rotated markups, `label_reads`) and the app, then deploy to Tiny Apps. (Released 2026-10-09 as planlens 0.13.0 and app 5.33.0; the deploy waits on CfA.)
  - **2026-10-09, later: waves 2c and 3 and their fixes.**
    - Commits: 5dee6a6 (D1: concurrent testers crossed folders; context copied into the heartbeat pump, shared hosts fail closed), f4849a6 + a87d5a3 (D2–D7, E3, E7–E10), c05e56b (E1, E2, E4–E6).
    - **Wave 3 (final confirmation, 10 flows, Sonnet): GO, with no high findings.** Review: `runs/w3-confirm-sonnet/REVIEW.md`. Tiled whole-page reads are down to a median of 11.8 s (from 17.6 s), and the DXF note takes 8 s (from 179 s).
    - **Spend:** $44.59 of the $80 smoke budget; the account limit was raised to $100.
    - **Remaining polish** (F1–F8; done in 83a77f4 and planlens f92541d except F1, zoom latency = BACKLOG VIS-07, and F7, rotated labels = DR-18): zoom latency 30–50 s, Word title punctuation, 0-mark copies, notes on blank paper, the planlens callout aim, zoom readings kept, rotated labels, `downloads.json` paths.
  - **GO conditions:**
    1. Release planlens (f679daf and later) first, then the app.
    2. The CfA items: image-capable Prompter values, SharePoint app registration, identity header, published origin.
    3. A first-hour check on Tiny Apps (steps in the wave 3 REVIEW.md).
  - **Only live GPT on Tiny Apps can show:** vision on the rollout model (GPT-5.6 Sol or higher since 2026-10-12; gpt-5.1 at 768 px when this was written), GPT latency against the caps, the real budget and rate-limit responses, several testers on one key, and links behind the proxy.

Machine note: heavy test suites must not overlap on the owner's laptop. Background runs were stopped twice for low memory on 2026-10-07/08.

### Open to-dos → `BACKLOG.md` (2026-10-12)

**All open work across the app, planlens and geotech-references is in
`BACKLOG.md` at the repo root: one list, stable IDs, tiers Now / Next /
Later / Parked.** Add new to-dos THERE. Never start a new to-do list here or
in a plan document; plans keep the detail, the backlog keeps one line and a
pointer.

**Why (owner, 2026-10-12):** "the recent sessions lost track of some of our
future plans and started new, more-narrow todo lists in the handoff doc". A
sweep that day read every plan, design note, ledger, smoke review and memory
file across the three repos. It found about 280 distinct open items, and only
about 60 of them were on the list that stood here ("Open to-dos (refreshed
2026-10-10)", A1–E6).
- That list was folded into the backlog item by item.
- Each old id is noted on its new item as *was: H A1* and so on.
- The old text is in git history (commit 414525a, HANDOFF.md lines 192–487).

**Now (from 2026-10-14): core improvements while CfA sets up Tiny Apps** (details in BACKLOG §0). CfA says the SharePoint connection and the per-user Prompter keys (one per user) are a few weeks out, so the Tiny Apps production items (OWN-29, OWN-30, OWN-43–OWN-51, TA-03) moved to Next until they arrive.
- **Done on master 2026-10-14 (unreleased; the owner holds the release):** see CLAUDE.md's state block and BACKLOG "Done since the last sweep" for every item and commit. In short:
  - unsafe-side module answers: downdrag (every method side by side, final stress), the slope search (sand slopes, toe circles, reinforcement), fem2d (the slope mesher's soil-filled notch, the failure criterion, staged `n_steps`, straight T6 excavation walls, a staged excavation that unloads in lifts), SOE free earth with no solution, the anchored sheet-pile moment, Terzaghi factors;
  - Ensoft-style calculation tables in 13 modules' calc packages;
  - Document Review: markups keep correcting and return the found box, geotech tools off the page, turned lettering, cut lines, sideways boxes, zoom and tiles on images; planlens: distinct sheets no longer called duplicates (reaches the app with PL-22);
  - long turns keep their work and wait out busy refusals; a model request gives up after its time limit (it waited 3.5+ hours on Funhouse); the record answers cost and cause; DIGGS writer and reader fixes;
  - CI: green again on 2026-10-15 for the first time in months (the five old failures, the workflow, and an OpenSees test that hung every run).
  - 2026-10-14/15 continued: OpenSees site response fixed (seven defects, then two more found by pinning it to the OpenSees wiki's own example); fem2d excavations dewatered by default (larger wall forces below the water table: a release note); slope supports as Wright & Duncan apply them, GLE / Spencer with large anchor loads, and no silent unreinforced fallback; log depths read off the drawn scale (planlens, reaches the app with PL-22) and helpers' records handed to write_diggs as files; cited pages checked against pages read; earlier turns' files and page readings reach the helpers (`kept_readings`, switched OFF on the review page); the Funhouse launcher's Prompter-key cell and lost-app helpers; the drift canary's latest report is read at the start of every session (BACKLOG "How every session uses this file", INF-12); slope and fem2d model-width checks against Taylor's problem, and a slope search that finds the deep base circle; the excavation result gives the depth of its peak moment; marks and find_like on images in their own pixels (VIS-26); the reviewer helper's model-call budget; pystrata equivalent-linear runs that converge; plan-view axes named from the stated coordinate system; a fresh resolve with pandas 3 passes, and streamlit is tested below 1.67; fem2d strength reduction over deep undrained soil reaches Taylor's deep circle by default (release note: a slightly higher FOS, about twice the run time, for those runs only).
  - The owner's 2026-10-14 field session (a report's borings to DIGGS on Funhouse) is filed and read in full: `module_work/field_feedback/2026-10-14_report-to-diggs_v5.33.0/`.
  - Its continuation on 2026-10-15 (kriging and a joint-field Monte Carlo of rock under a footprint, then the plots) is filed at `module_work/field_feedback/2026-10-15_rock-excavation-kriging-mc_v5.33.0/`: the crash that ended it and the chat's unreadable chart text are fixed, plans are drawn to scale with named points and filled outlines, and the conditional simulation the app could not run is built (`gstools.conditional_simulation`, conditioning by kriging, checked against conditioning by its definition; follow-ups GM-96). `calculate` counts, adds up and averages lists and has the normal distribution (the session counted footprint fractions by eye); the helper that draws the figures can look at them, and a chart written by hand is flagged. The owner chose a figure helper (Tufte checklist plus worked examples): built and measured behind `GEOTECH_FIGURE_HELPER` (OFF); whether it goes on is the owner's call (GM-92). Plans are now drawn as drawings (scale bar, north arrow, property line; the owner's comments on the figures page). The owner then approved plotting code run in a locked box: built behind `GEOTECH_PLOT_CODE` (OFF) for the figure and `calc` helpers (f0bc000; follow-up INF-16), and named the figures engineers make most (built 2026-10-16 with the owner's conventions: elevation by default, kPa and kN, the wall drawn). Downdrag now leaves the pile's own weight out of the load curve by default, as the references do (867b08b; the calc package takes a steel section since).
  - The night of 2026-10-15/16 (the owner asleep: "keep working through the backlog on things you can handle yourself"; every item in BACKLOG "Done" with its commit): plotting code in a locked box measured live twice and fixed after each full read (waves w8 / w8b; `module_work/live_smoke/runs/w8*/TRACE_REVIEW.md`; the switch now brings the figure helper; a layout check measured from the drawing); downdrag leaves the pile's own weight out by default, its calc package takes a steel section, and it warns when nothing below the toe can settle; a wall check not run says "not checked" and failed checks come first, bearing on the larger edge pressure (release note); no placeholder FOS reported as a number; the Backlog Desk takes several answers per card; a look waits while a figure is drawn and names no zoom tool the caller lacks; large JSON results open with a key map; `write_xlsx` refuses unknown sheet keys; method docs come back whole; the Jibson range (GM-17, part); staged FEM at 500 iterations a step (release note); `html_to_pdf` says what landed on each page; the record keeps an errored turn's tokens, labels probe calls and lists each turn's tools (EV-20, part); the step cap in plain words and, behind `GEOTECH_STEP_CAP_ANSWER` (OFF), an answer from what was gathered; stale docs. The figure conventions and worked code examples are committed (the owner's answers: elevation by default, kPa and kN, past pressure as points or a design line by stage, the wall drawn). The cited-documents check is built behind GEOTECH_CITED_DOCUMENTS (OFF; follow-ups EV-23). Downdrag counts settlement below a pile's toe by the references' own rule (UFC 3-220-20's 10 % of the bearing stress by default, GEC-12's 10 % of the effective stress as the option; release note: more toe settlement where soft soil continues below the toe; the default is the owner's call, OWN-62). Wave w9 (flows F56-F60, five figures the tools do not name, $1.54) ran and was read in full (`module_work/live_smoke/runs/w9-figures/TRACE_REVIEW.md`). Flows F61 / F61j ran live as wave w10 ($0.82; `module_work/live_smoke/runs/w10-judgment/TRACE_REVIEW.md`): with the switch off the answer named its search choice, why, the alternative and what would change it; with it on the turn ran out of time in two long Spencer grid searches (since fixed: a time budget), so that arm was unmeasured (measured since in wave w12). Re-run 2026-10-17 as wave w11 ($0.35; `module_work/live_smoke/runs/w11-judgment/TRACE_REVIEW.md`): both arms found 0.935 in the clay by default and named their calls (3 of 3 off, 4 of 4 on); with the switch on the user also got a short list of judgment calls (another reading of the layer, a target FOS, a cohesion check, what would change the answer); neither said why it chose Spencer or cited a reference (GM-155). In flight now: nothing. Owner 2026-10-18: no more slope_stability fixes; every slope item waits for one integral review (GM-140). Wave w12 (16 runs, $5.05; ledger $62.93 of $80) was read in full (`module_work/live_smoke/runs/w12-figures-judgment/TRACE_REVIEW.md`): one figure card per helper holds, but a second helper asked to redraw started over (fixed since, below); the judgment-call switch is not ready to turn on (same engineering at nearly twice the cost, no reference cited, assumptions passed on as facts; GM-155); the A-line drawn from memory again (fixed since); the wall's heel surcharge counted as resisting (fixed since). Tool descriptions and the defaults a result reports now say what the code does, checked by running 165 of the 189 analysis methods. Weak-rock p-y curves follow Reese (1997) as printed in GEC 9 App. A.5 (90e23b7). A figure is one thing across a turn, whoever draws it, with its code kept beside it (e593029; behind the figure switches; a live wave must confirm). Standard chart lines (the plasticity chart, the AASHTO chart, sieve sizes) are drawn from the references (refs 6ad9fbb + a5ab30e). A live surcharge over a cantilever wall's heel no longer resists sliding or overturning (2844f1e; owner question OWN-64 on a permanent one). The MSE wall's ASD path does the same (8960ecd). 2026-10-19: judgment calls cite their source and calc packages keep given and assumed apart (8b9abba); wave w13 ($3.10, `module_work/live_smoke/runs/w13-judgment/TRACE_REVIEW.md`) kept the switch OFF: no answer in either arm made up a percentage and ON cost 0.92x OFF, but the ON answers never consulted the references and one printed c = 0 as given; round 2 (006ed10) and wave w13b ($2.55, ON only) got one references consult in three of four answers and real citations for 12 of 15 calls, still NO: one answer never consulted and one printed c = 0 as given (GM-155: the turn-end check and the given-inputs fix, both merged; then the delegation fills `given`, merged; the rerun w14-judgment ($1.79) said NO on cost only: the consult works, citations reach the PDFs, Given vs Assumed is right; the switch stays OFF until the owner rules how cost is judged, OWN-65); filed ten follow-ups (most now done), EV-29, REF-29. PyMuPDF 2.0 (released 2026-10-19) broke six report_ingest tests in CI: capped below 2 (3cd8fc1, planlens eeb278b); planlens main now reads 2.0 correctly (1e8e328), and the app stays below 2 until planlens 0.14 is released and the hosts pass an import probe (INF-18). Cloud sessions (the owner starts them from `dev/cloud_prompts/`): driven piles merged (4aafc4d, reviewed and fixed); six more ran (DM 7.1 edition, reference data, module inputs, figure tools, shoring, FEM wall friction; all finished 2026-10-19, all merged), each on its own `cloud/` branch. In flight (helpers in worktrees, up to four at once, owner 2026-10-19): GM-219 (header slivers); faster CI runs (a side branch, tried on GitHub); a backlog tidy (the Done list to an archive file, the Later items triaged; owner 2026-10-20); Figure tools merged (581a49f); shoring passive coefficients and calc packages merged; CI's matplotlib 3.11 failure fixed (a haloed note measured as drawn); the judgment-call turn-end check, hidden unrunnable tools and the calc helper's page tool merged; PDF figures move whole to the next page (no more 0.1-2 in figures); given vs assumed inputs merged (inputs left out say what they stand for; Given Inputs only what the call names as given); the four agent fixes merged (the calc helper's call budget, one page-number convention, a list for `calculate`, pyplot warmed before parallel tools). Judgment calls under the switch: an answer still unsourced after a consult goes back once, and the consult is held to 6 calls (switch OFF unchanged); the harness's detectors corrected and a check for a number no tool gave added. Next: INF-19, GM-196, GM-205, GM-211, GM-212, GM-221; owner OWN-66, OWN-67 (earth-pressure defaults); the second cloud batch is running (the calc-step citations merged; REF-29, PL-25/02/06, GM-79/80, GM-07/09/161, REF-03/07, started 2026-10-20). Owner: OWN-63 (five GEC-12 readings for driven piles), OWN-64 (a permanent surcharge over a cantilever wall's heel). Every tool's nested inputs (layers, supports, loads) now refuse a misspelt key with a did-you-mean (a misspelt key used to be dropped silently), and so do top-level names, with a reason where a tool deliberately does not take one. MSE walls take a sloping or broken backslope per GEC-11, pinned to its Examples E1 and E3; traffic is left out of pullout (release note: a lower pullout FOS near the top of a wall with a traffic surcharge). The figures page shows the conventions applied to the worked examples (v7).
- 2026-10-16 (every item in BACKLOG "Done" with its commit): the cited-documents check (OFF switch `GEOTECH_CITED_DOCUMENTS`); downdrag counts settlement below the toe by UFC 3-220-20's rule with GEC-12's as an option (the default is OWN-62); wave w9 (five figures the tools do not name) read in full and its fixes built (the chat shows the newest figure, a look checks content before lettering, the redraw cap holds); every tool refuses a misspelt input name, nested or top-level, with a did-you-mean, and gives a reason where it deliberately does not take one (a survey found planlens already does); MSE walls take a sloping or broken backslope per GEC-11, pinned to its Examples E1 and E3; `annotate_document` refuses misspelt remove keys, shows the model the markup fields and keeps its own signature; the slope search sets its automatic ranges from the slope's own geometry, labels an infinite-slope sliver for what it is and reports the lower surfaces it refused and where the governing surface lies (a drawdown pond-pressure error fixed on the way); behind `GEOTECH_JUDGMENT_CALLS` (OFF) the agent puts every engineering judgment call in front of the user (owner: "User's judgement is the one that matters"; flows F61 / F61j; measurement EV-25 and wave w12, GM-155); each module's result says what it chose (the method, given or default, the basis with its reference, the alternatives) for axial and lateral piles, settlement, seismic earth pressure, liquefaction and slopes, tool results list the defaults used, calc packages print a Method Choices table, and the citations were checked against the printed manuals (several were wrong); the judgment-call flows ran live (wave w10, read in full; follow-up EV-27). Owner rulings the same day: engineering judgment about when an analysis is the wrong tool stays with the agent and its references, never hard-coded in a module; one helper at a time.
- 2026-10-17 (every item in BACKLOG "Done" with its commit): a slip surface below the ground in more than one stretch is refused, as Slide2 does, instead of being scored across the air between; that showed the default search never tried circles coming out on the face above the toe, so it now does (FHWA's "circles passing above the toe"), leaving out circles Slide2 also refuses (m_alpha below 0.2, reverse curvature); a 10 m cut with a weak clay over rock now reads 0.936 by default, where it read 1.16 on circles scored across air; reversed search windows are swapped with a warning (follow-up GM-126); a slope search the agent runs stops at 180 s (the agent may give up to 600 s), refines around the best points before it goes finer so a cut search still finds a narrow valley, and opens its warnings with "the reported FOS may be too high" when cut; each method's cost is in its description, and a Bishop screen for rigorous methods is an option; the judgment-call flows re-ran live (wave w11, read in full): both arms found the clay by default and finished in two to four minutes; every slope search leaves out surfaces Slide2 refuses (m_alpha below 0.2 on a slice with friction; slices with no friction are exempt, since the check does not bear on them there), a single surface the user gives is reported with a warning, and the critical surface is checked at twice the slices (follow-up GM-130); `choices` and `defaults_used` agree with each other and list only what the run used (a value typed equal to the default is "given"), the slope results name only the surface families scored, and the tool descriptions tell the agent the choices are there (follow-up GM-132); a noncircular slip surface is sliced at its corners and at layer and water-table crossings, so its FOS no longer depends on the number of slices (follow-ups GM-135, GM-136); CI was red for four pushes (two tests from this work) and is green again; every liquefaction route requires the water table and the PGA (a CPT run used to take 1.0 m and 0.25 g without saying so). 2026-10-18: circles are sliced at layer, water-table, surcharge and crack edges and every slice takes its chord, so a FOS no longer depends on the slice count and the pinned published checks land on their converged values (follow-up GM-137); the rigorous methods' circular search refines on their own FOS, not Bishop's, and finds a published slope's critical circle 1.6 % below Slide2's circular value, matching Slide2's own noncircular result (follow-up GM-139); a slip surface's depth below the model's base is measured on the surface, so a circle tangent to the base is accepted at any slice count. The owner then stopped slope fixes: nine in three days, each started by one case, is fitting narrow problems; the rest waits for one integral review (GM-140). Calc packages no longer use a fixed constant when the inputs given determine it (seismic Ka from phi, settlement L = B) and print the values the run used; then every calc package was moved onto its tool's own code path, inputs and parameter docs, so a package and its tool cannot drift again. The driven-pile module was audited against the whole of GEC-12 chapter 7 and now reads its charts, tables and limits as printed (five judgment calls to the owner as OWN-63). The reference library's section numbers were checked against every printed manual and 39 renumbered (geotech-references 4dc357f; a test keeps them so), and every section level became searchable and fetchable, so the librarian reaches GEC-12's pile methods in one search and a small fetch (d5f7f52); and every reference tool's result now fits the app's size limit by its shape, with a note on how to get the rest, instead of arriving cut mid-JSON (db84be2). The figure helper's layout check now judges by what is drawn (false reports in the w9 figures 22 of 36 -> 1 of 16), and `plot_data` places notes with optional leaders, checked by the same rule. Module defaults that were silent are now right or stated, and the wall and settlement calc packages print the values the analysis used.
- Release notes to carry (everything since 5.33.0 is unreleased; the owner holds the release): fem2d excavation dewatered by default (larger wall forces with a water table above the excavation level); fem2d SRM over deep undrained soil a slightly higher FOS at about twice the run time; gstools conditioning by kriging by default; downdrag leaves the pile's own weight out; wall bearing on the larger edge pressure and "not checked" without q_allowable; staged FEM 500 iterations a step; liquefaction refuses amax 0; downdrag counts more toe settlement where soft soil continues below the toe; tools refuse misspelt input names instead of ignoring them (a call that worked by accident may now be refused with a did-you-mean); MSE pullout leaves traffic out (a lower pullout FOS near the top of a wall with a traffic surcharge) and strip F* reaches its deep value at 20 ft; the slope search sets its automatic ranges from the slope's own geometry (several pinned FOS move toward the published values); module results gain a `choices` list and tool results a `defaults_used` list (more text per result, no number changes), settlement warns when the secondary-compression time is an assumed 1 year, and seismic earth pressure warns when Mononobe-Okabe has no solution and the Seed & Whitman estimate is used; the slope search refuses a surface below the ground in more than one stretch and tries circles coming out on the face above the toe (lower FOS where a weak layer outcrops on the face; V-042's pinned circle 1.325 to 1.336 against a published 1.334; a default search about 1 s longer); a slope search the agent runs stops at 180 s unless it asks for up to 600 s, and a cut search says its FOS may be too high (the library itself has no time limit); searches leave out surfaces with m_alpha below 0.2 on a slice with friction, as Slide2 does (a Janbu example 1.076 to 1.199; a reinforced slope's discard-curvature case 1.467 to 1.432, the safe side); a noncircular surface is sliced at its corners and at layer and water-table crossings (noncircular and fixed-polyline FOS move, mostly up by 1-3 %, to where the old code converged at very fine slicing; a published cut-slope check 1.190 to 1.218); a CPT liquefaction call without the water table or the PGA is refused (it used to take 1.0 m and 0.25 g); circular-surface FOS move to their converged values, mostly up by 0.5-3 % (a published reinforced-slope check 1.180 to 1.210 against 1.209; a vertical cut in clay 0.800 to 0.842 against Taylor's 0.851), and rigorous searches on clay slopes take about 20 % longer; a Spencer, M-P or GLE circular search can now report a lower FOS where its old refinement followed Bishop (a published reinforced slope 1.238 to 1.186, against Slide2's circular 1.209 and noncircular 1.188); a given slip surface within 1/1000 of the model's height below its base is taken as touching the base, and one deeper is refused whatever the slice count; a seismic calc package without KA takes Rankine Ka from phi (at phi 34 degrees the static thrust 113.9 to 96.7 kN/m, the seismic increment up by the same, the total unchanged), and a settlement package without L takes L = B as the settlement tool does (a 3 m footing 39 to 87 mm); every calc package now runs its tool's own code and inputs: a pile-group package computes the group efficiency when the layout is given, a downdrag package without the fill unit weight takes the tool's 19 kN/m3 (was 20), a settlement package without B and an axial one without pile_type are refused as the tools are, and `axial_pile_capacity` itself refuses a call with no pile_type; driven-pile capacities follow GEC-12 chapter 7 as printed: in clay up by about half where the printed Fig. 7-17 replaces one simplified curve (a concrete pile 1,327 to 2,043 kN), deep sand toes down by Step 8's 3 ksf limit, an open pipe's unplugged case down (inside friction 1/3, plug weight subtracted), a beta-method clay toe far lower (Nt 3), and uplift the full shaft resistance (was 75 %); the reference library's section numbers follow the printed manuals (39 renumbered in six references, so a citation the librarian returns matches the page; needs a geotech-references release); the librarian finds and fetches subsections (a parent comes back with its own text and a list of its children instead of its whole subtree; geotech-references d5f7f52, same release); no reference tool's result is cut mid-JSON any more: long searches, chapter listings and chapters come back shorter with a note on how to get the rest (geotech-references db84be2; the app's geotech-references pin must rise with that release); settlement: a strip given with no length is a strip (2:1 stress at q 150 kPa, B 2 m, z 4 m 16.7 -> 50 kPa, consolidation 56 -> 139 mm), secondary compression uses each layer's e0 (20.83 -> 21.92 mm in the test case), the library's no-L default is L = B; a drilled shaft's cohesionless tip with no N60 is refused (it was a silent 15); the wall calc package's printed M_stab and x_R now match the analysis (419.2 -> 414.5 kN-m/m, FOS unchanged); section_properties accepts r=0 (it was refused); lateral piles in weak rock follow GEC 9 App. A.5 (alpha_r from RQD, p_ur up to three times lower; initial slope 100-500 E_m, was 0.25 E_m; 1.2 m shaft at 500 kN 3.9 -> 0.23 mm, at 8,000 kN 70 -> 90 mm and 9,555 -> 16,467 kN-m; re-run weak-rock analyses); `plot_data`'s `reference_chart` draws standard chart lines from the references (needs the next geotech-references release, after 6ad9fbb, and a pin bump; hidden on 1.4.1); cantilever walls: a surcharge over the heel no longer resists sliding, overturning or the middle-third check by default (F55: sliding 1.17 -> 1.07, overturning 3.06 -> 2.78; bearing unchanged; `live_load_resists=true` gives the old results); MSE walls, ASD path: the same (6 m wall, q 12 kPa: sliding 2.599 -> 2.339, overturning 4.333 -> 3.900; GEC-11 E4 inputs 2.268 -> 2.104); calc packages print Given Inputs, Assumed Inputs and Values Used, and the agent states a sensitivity number only from a tool run or a reference; driven piles: options for GEC-12 Fig. 7-18 (or the smaller of 7-17/7-18), a shrink-swell depth, Table 7-9 beta/Nt and the toe phi over 3b, a custom pile section on every axial tool, `capacity_vs_depth` refuses a call with no pile_type (no default number moves); figures: the layout check reports hatching and arrows through lettering, names keep clear of reference lines and hatching, a plan view takes in every outline, the plotting box allows `time` (figures with names or plans may draw differently and carry new warnings); the calc helper is told its call budget and warned from four calls left; `calculate` takes a list or a name-to-expression map; `read_pdf_text` takes 1-based `pdf_pages` and labels each page `pdf_page`; a failed scratch edit says nothing changed; PDF reports: a figure that does not fit the rest of a page starts the next page at full size with its heading, one taller than a page is scaled to fit, html_to_pdf no longer drops the text after such a figure (reports can run longer), `page_contents` numbers pages as `pdf_page` and gives each figure's printed size, figure files are named from their titles; new: post-liquefaction downdrag (GEC-10 §10.7): the neutral plane, drag force and settlement before the earthquake and after the liquefied layers reconsolidate, every method side by side with the cited side-resistance options (50 % default, zero, su,r, a ratio), taking the liquefaction tool's output (a CPT's depth profile too); a short result by default (tables as columns and rows, more than 6 zones summarised), `detail='full'` for the whole; the review page's prompt names its one calculation tool, `calculate`; bearing: Meyerhof with a load inclined at phi or more is refused (it gave a capacity that rose again with the angle), as are a negative or 90 deg load inclination, a ground slope outside 0-45 deg and a base tilt past alpha tan phi = 1 (the module's own limits); at Df/B = 1 exactly Vesic's depth factors use arctan(1) as UFC prints (about 4 % lower there only), `depth_factor_method` gec6 or none, an inclined load on a soil with cohesion needs `vertical_load`, Vesic's inclination factors now apply on a strip (an error on the unsafe side: -34 % in the example); elastic settlement with no Iw takes GEC-6 Table 5-12 (12 % higher for a flexible square centre); a shaft capacity-by-length sweep lists the lengths it skipped and why; a Method Choices table whose rows run taller than a third of a page, or past the right margin, prints one block per choice (on master the retaining wall with surcharge package dropped 18 words and the slope report 37, with 10 more cut at the margin; now none) and a table ending at a page foot no longer fails the PDF; Story PDFs no longer leave out what starts at a page foot (a section, figure, table row, list item, step result or a paragraph's last lines; a PDF may be a page longer); a call whose `parameters` is not an object of named inputs is refused with the expected shape and input names, not run; SoilProfile's lateral-pile input follows GEC 9 p. 219 (stiff clay takes Reese et al. 1975 only where free water is at the ground surface, Welch & Reese otherwise, below the water table too; deep-water results move, 600 kN: 10.9 -> 15.5 mm; `free_water_at_surface=True` restores the old choice) and passes `n_cycles`; lateral piles: stiff clay below the water table and Matlock soft clay under cyclic loading follow the printed curves (stiff clay, 0.9 m shaft, cyclic 600 kN: 19.4 -> 63.0 mm; cyclic loading there takes `kc`), and a load beyond what the pile and soil can carry fails with that capacity instead of returning numbers; lateral piles in stiff clay above the water table under cyclic loading use the printed cyclic curve with the number of cycles N (required; before, the static curve: a 0.9 m shaft at 300 kN went 4.47 mm / 387 kN m to 6.03 mm / 430 kN m at N = 100), static unchanged; log-spiral Kp from a chosen printed source (Caquot-Kerisel 1948 as Caltrans prints it, the default; UFC 3-220-20 Fig 4-13 or Eq 4-12) and sloping ground for Caquot-Kerisel and for K0 (Caltrans or EN 1997-1, named), Kp between the chart's 5 deg lines read on its log scale (up to about 5 % lower; log-spiral sheet pile embedment about 2 % deeper, 2.895 -> 2.958 m), a keyed wall's passive as UFC Fig 4-29 draws it (Pp and FOS sliding go up: 9.7 -> 29.2 kN/m, 1.31 -> 1.57 for a typical wall); the cantilever wall figure draws the active pressure where the analysis applies it (the vertical plane through the heel), with the surcharge, and draws a shear key (position optional); the pile group plan takes an optional cap outline and property line; downdrag profiles take an optional ground elevation (results unchanged), and the library's `plot_neutral_plane` no longer draws depth upward; calc packages list their own left-out inputs under Assumed Inputs; calc text cites GEC-6 chapter 5 and FHWA-SA-02-054, and the settlement, pile-group, drilled-shaft and wave-equation citations match the printed GEC-6/10/12, with a guard that every cited manual number exists (no numbers change); fem2d wall, strut, spring and joint forces are per metre whatever the slice thickness `t` (at t other than 1 they no longer scale with it; t = 1 unchanged); a ground anchor given a bond stress or load transfer needs `soil_type` ('soil', 'rock' or a GEC-4 table row; it was taken as sand); drilled shafts take `fractured_rock_thickness` (GEC-10 p. 8-15) and aggregate piers take `settlement_method` from GEC-13 Fig. 5-21 with `E_column` and `E_soil` (defaults unchanged); `mse_lrfd_external_stability` works the bearing resistance out from `gamma_foundation` and `phi_foundation` by GEC-11 Eq. 4-22 (qR = 0.65 qn; before, bearing went unchecked), ground-improvement feasibility holds top + thickness against GEC-13's typical depths (aggregate piers feasible as stone columns to 30.5 m, was excluded past 12 m; vibro-compaction excluded past about 50 m), inputs that changed nothing are refused with their reason (axial E, socket_length, E_column, gwt_depth_m, i_c_limit, s_g, beam A, gamma_foundation on mse_wall); `earth_pressure_coefficient` refuses wall friction, a battered back or a slope its coefficient does not take and names the alternative (before, Rankine Ka with delta 20 gave 0.333, and Rankine Kp, Caquot-Kerisel or K0 with a slope gave the level-ground value), Caquot-Kerisel echoes delta = phi, `composite_section_ei` refuses bars the section would drop (use `cased_concrete`), `capacity_vs_depth` gives `Q_allowable_kN` and `make_pile_section` gives `E_kPa`, combined settlement with no Es says no immediate settlement was computed; `plot_data` and the plotting box refuse a figure that draws nothing, or one whose title, axes or series call it a placeholder, before any file is written; calc-package figures are drawn for the 5.0 in width they print at, so their lettering prints at 7 pt or more (it was 3-6.5 pt), and a figure that still prints small is named in `figure_warnings` as the module's own; calc-package figures follow the figure rules (M-O pressure in kPa; the wall's toe ground where the analysis takes it; downdrag depth axes say no elevation was given; the pile group drawn as a plan with a scale bar and a north arrow from the new `north_deg` or up, tension hatched; no numbers change); keys and notes sit under the drawing (the figure grows to hold them), labels beside what they name, values on lines kept clear (no results change); LaTeX packages widen a figure, up to the text width, where its set width would print lettering under 7 pt; ground anchors sized for `test_load_factor` x DL, default 1.33 (was a fixed 1.50; 1.20 temporary SOE and 1.50 creeping soils as options): DL 300 kN on 15 mm strand 3 strands -> 2, `performance_test_kN` 1.33 DL, the 32 mm bar holds up to 501 kN DL (was 509) and the 1-3/4 in bar 1,067 kN (was 960) per GEC-4 Table 9; `braced_excavation` / `apparent_pressure` refuse a layer above the base with no `soil_type`; a multi-anchor FHWA wall refuses `FOS_embedment` and `Kp`; ground anchor bond length uses GEC-4's printed ultimate load transfer per metre (Tables 6 and 8), not a stress over a 150 mm hole: soil bonds were about twice too long and rock ones too short (DL 445 kN: medium sand 13.03 -> 6.14 m; the old rock rows 3.00 -> 4.50 to 8.90 m), six old names refused with their replacement, rock FS 3.0, minimum bond 4.5 m (3 m for a bar in rock), bar unbonded minimum 3 m, bond zone max(1.5 m, H/5) behind the failure surface measured normal to it; bearing with `factor_method='meyerhof'` uses Meyerhof's own N_gamma, (Nq-1)tan(1.4 phi): q_ult falls below phi about 46 deg (a 2 m square footing at Df 1 m, phi 32: 1,465.9 -> 1,241.0 kPa, -15 %), and under an inclined load its shape factors are 1 (UFC 3-220-20 Table 5-6 note A; the same footing at 10 deg 894.9 -> 594.9 kPa); `ngamma_method='vesic'` restores the old value under a vertical load only; the bearing result gives the net allowable pressure and the allowable structural load (for an eccentric footing q_all A' - q A); module choices read "given" only for inputs the call names as given; seismic PAE carries (1 - kv) per UFC Eq. 4-16 (kv still defaults to 0; at kv 0.1, 134.8 -> 121.3 kN/m); anchor tendons sized so the 1.50 DL test load is at most 0.80 SMTS (a 500 kN 32 mm bar becomes 36 mm); a variogram nugget of 0 fixes the nugget and a negative one is refused; cushion_cor changes the default cushion; AGS4 files with rule errors are invalid (before, every file passed) and a DIGGS file that passes the dictionary check is valid; concrete_fc is refused on the drilled-shaft tools (it was ignored); the drilled-shaft package prints the Nc cap, F_r and the bell's 1.27/D; fem2d_excavation `wall_contact='interface'` adds Mohr-Coulomb wall friction (default smooth, no number moves); deepagents' `execute` and `delete`, which cannot run here, are no longer offered on the geotech page; the calc helper can look at the pages of the PDFs it builds; behind `GEOTECH_JUDGMENT_CALLS` (OFF) an answer whose judgment calls name no source, in a turn with no references consult, is sent back once; Caquot-Kerisel passive coefficients follow the printed chart and table over phi 0-45 deg (Kp0 at phi 35 9.20 -> 10.2, at 40 14.0 -> 17.7; at delta/phi = 2/3, 35 7.20 -> 8.24, 40 10.96 -> 13.26; the log-spiral options of sheet_pile and retaining_walls only, SOE uses Rankine and is unchanged); new SOE and reliability calc packages; tool results and calc packages say what a left-out input stands for (no water table, one soil layer, bearing not checked, foundation soil taken as the backfill, with its basis), a calc package's Given Inputs hold only the inputs the call names in `given` in the user's words, every other input prints under Assumed Inputs and a choice the user did not give reads "assumed" (no computed number changes); PyMuPDF floor 1.27.2 (cap <2 kept), and planlens main reads 2.0 correctly with one PDF read at a time across threads (two reads at once could leave box heights wrong and printing lost for the rest of the process; that race is in released planlens 0.13) — a planlens release note; references (refs 9cd4228, needs a geotech-references release + pin bump for the cluster, with the plasticity and grain-size chart lines): DM 7.1 cited as UFS 3-220-10 (16 January 2026) with its printed pages, GEC-13 chapters 3-10 rebuilt from the printed manual, GEC-12 ch 7, micropile ch 5 and five DM 7.2 sections follow the print, run-on figure captions fixed, the ground-improvement package cites GEC-13 as printed (no numbers change).
- **Next:** EV-13 with DR-05: Foundry brief 6 is drafted (`../foundry_handoff/AI_FDE_BRIEF_6.md`: part A the image zoom check, `image_zoom_check_foundry.py`, on the rollout model; part C the suite with arms `baseline`, `lean`, `turn_retry`, and `kept_readings` as an optional arm). Its rc test wheels, built off master as for brief 5, are not built while the owner holds the release. Module follow-ups in Next: GM-79, GM-81, GM-82, GM-96. Also EV-23.
- **Owner:** GM-92 (the figure helper and plotting code: on by default?), OWN-31 (the checklist session, perhaps next week), OWN-59 (slope supports: driving or slack), OWN-60 (photos of Duncan & Wright §6.5.3 and ch. 8). Answered on the desk 2026-10-14 (BACKLOG "Done"): hold the release and use the Prompter key on Funhouse this month; downdrag shows every method, skin-friction stress per GEC-12; fem2d measures from the end of stage 0; calculation tables are the next module train; the old `geotech-report-extraction` project is deleted from PyPI. Answered 2026-10-15: the QC routine's token revoked and the routine deleted; downdrag leaves the pile's own weight out by default.
- **Rollout model (owner, 2026-10-12):** one GPT-5.6 Sol-or-higher deployment for every call, not gpt-5.1; the app measures it on first use. So the gpt-5.1 rehearsal wave is dropped, and the standards PDFs stay with a provenance note (BACKLOG "Done").
- **Done on master 2026-10-12 (unreleased):** the default review page's zoom wording, reading helper and answering budget; the markup check against the printed thing; a plain message and one retry on a refused key; `.env` working on dosdev; the suites run on CfA's pins (WSL), which found the python-ags4 install conflict; the install guide and `TINYAPPS.md` current; Department addresses scrubbed from tracked files (history: OWN-56). Commits ae50842, da28a69, 0562ddf, a229090, 24b46df. On dosdev, 5.33.0 installs with pandas left unpinned (pip takes 2.x) until the next release.

**Open questions** about how the harnesses behave are listed at the end of
each document in `module_work/harness_theory/`. Read that folder before
changing a harness.

### 5.32.0 RELEASED 2026-10-04 (tag `v5.32.0`) with planlens 0.11.0 (tag `v0.11.0`)

**Released on the owner's word** ("When you're done with code tonight, go
ahead and publish. I'd like to give it a test run tomorrow"), after the
Foundry measurements of 5.31 and of 5.32 itself (rc2/rc3). **Pin raised to
`planlens>=0.11`; no new third-party package.**

**Release gate, in CLEAN worktrees of the release commits (app d903d54,
planlens 059bc79 = PyPI 0.11.0), exit codes 0:** deep + suite + digest +
webapp + `geo_project` 1,180 passed / 4 skipped; the rest of
`funhouse_agent` 1,508 / 6; `report_ingest` 1,361; planlens 1,450. Wheels:
app requires `planlens>=0.11`, carries `markup_check`, `minimal_agent`,
`palantir_sdk_engine`, no tests / `module_work` / `raw`; planlens 0.11.0
declares the same five dependencies as 0.10.1 and carries
`planlens/opencv.py`. planlens 0.11.0 was on PyPI before the app's tag.

**Decided from the Foundry runs** (`module_work/review_eval_results/
2026-10-04_foundry_5.32.0rc3/NOTES.md`, GPT-5.6 Sol at full resolution):
review switches stay OFF — baseline 35/35 on the original tasks and 5/6 on
the two new ones; `sweep` ties at 2.4x the tokens and 2.7x the time; the
`minimal` measuring stick loses 2 originals (no text search, no markup data)
at ~4x the tokens; full-size images kept (HIGH added false pages on the
24-sheet task).

**What 5.32.0 carries.** From the 2026-10-01 field session (tags missed
across an 85-sheet set; circles drawn in empty paper — record
`module_work/field_feedback/2026-10-01_doc-review-penetrations_v5.31.0/`):
review prompts LOOK at every page in scope (text tools are evidence, never a
filter; say what was covered); `annotate_document` checks every mark placed by
location (`funhouse_agent/markup_check.py`); planlens `circle` markups,
visible `label`s, `view` + `image_box` anchors converted in the writer,
`box`/`page_bbox` accepted for `bbox`; screenshot paste; SharePoint browser
addresses; no auto-orientation for a mid-conversation screenshot; "Find a
past conversation" per page; the Foundry vision door; suite tasks
`set-long-rare-tag` and `produce-circle-tags`, the `minimal` arm, a "failed
calls" column; stale prompt notes fixed; the ingest stage honours
`triage_model`. **After rc2/rc3, from the Foundry runs:** a cap of 8 vision calls
in flight per process (`GEOTECH_VISION_MAX_INFLIGHT`); planlens test-loads
OpenCV in a child process and the app hides `find_like` where it cannot load
(a FIPS host aborted the interpreter); the markup check also asks whether a
box/circle is drawn CLOSE round the thing (an agent had widened misplaced
rings until confirmed); the Foundry engine's Responses route + infra retries
(`GEOTECH_FOUNDRY_ROUTE`, folded in from the FDE's glue); fixes for the 31
real tool errors of the 106-question geotech eval (category-as-topic
`list_methods`, closest-method hints, parameter aliases, scratch-guard and
large-result reads); report ingest: lab scorer folds hole IDs, run files keep
records + misses (`rescore_saved`), recovery printed as a length scores,
DIGGS writes past one bad-unit log and round-trips unnamed holes / pit water,
the label review survives running out of output room, no Funhouse dollars on
Foundry, narrative conventions re-read from the owner's answer keys (DRAFT).
Suite checks: LaTeX numbers, "pages:" and bulleted page lists;
`score_review_suite(rescore=True)`.

What is not yet verified live, and every owner item, is in `BACKLOG.md`
(since 2026-10-12). **Also written 2026-10-04:** `module_work/harness_theory/` — a theory
of operation of the three harnesses (shared machinery, Document Review, the
geotech agent, report ingest), from the code and the recorded Foundry runs,
by an independent agent at the owner's request; its evidence is kept locally
under `module_work/field_feedback/2026-10-04_foundry_evidence/raw/`
(gitignored).

**How 5.32 was built and measured** (the record, not to-dos): branches
`release/5.32.0` / `release/0.11.0` from 2026-10-01; nothing published until
Foundry had measured it — the AI FDE (Claude inside the owner's Foundry) ran
three briefs (`../foundry_handoff/AI_FDE_BRIEF*.md`): the review suite on
5.31 (runs 2–3) and on 5.32.0rc2/rc3 (run 4) — `module_work/REVIEW_HARNESS.md`
results ledger; report-ingest stages b–e and the 106-question geotech eval on
5.31 — `module_work/field_feedback/2026-09-16_report-ingest/MEASUREMENTS.md`
"FOUNDRY RUN 1". Foundry facts: memory `reference-foundry-model-route` (HIGH
caps images at ~768 px, AUTO is full size; GPT-5.4 only on the Responses
route; numpy 1.26 there; OpenCV aborts under FIPS). Day-by-day state:
`module_work/SESSION_2026-10-02_doc-review-5.32.md`.

### 5.31.0 RELEASED 2026-09-29 (tag `v5.31.0`) — DOCUMENT REVIEW M1, SWITCHED

**Released on the owner's word ("Yeah you can release"); no dependency
change** (pin stays `planlens>=0.10`). **Release gate, in a CLEAN
worktree of the release commit (cc4bb67; planlens at 0.10.1 = PyPI),
exit codes:** deep + suite + digest + webapp + `geo_project` 1,100 passed /
4 skipped; the rest of `funhouse_agent` 1,458 / 6; `report_ingest` 1,297
(two halves). Wheel: `review_findings`, `review_digest`, `findings_tools`,
`geometry_tools`, `digest_tools` present, no tests, requires
`planlens>=0.10`. **Lesson:** never hand pytest a file list built from
`ls tests/*.py`: it collects `live_reviewer_test.py`, which pytest's own
pattern skips, and that script makes a live Anthropic call at import (it was
refused for zero credit; nothing charged). Split by directory or `-k` instead.

**Plan of record: `module_work/REVIEW_ARCHITECTURE.md`** (three job shapes;
milestones M0-M4; decisions and open questions). 5.31.0 is its M1: what does
not wait on the owner's suite numbers, each behind a switch, OFF by default:
findings (`GEOTECH_REVIEW_FINDINGS`), contact-sheet overview
(`GEOTECH_REVIEW_OVERVIEW`), geometry on the review page
(`GEOTECH_REVIEW_GEOMETRY`), the digest's free layer
(`GEOTECH_REVIEW_DIGEST`); arms `overview`, `geometry`, `digest`; suite 35
tasks. Switch table: `module_work/REVIEW_HARNESS.md`.

**Owner's next steps (Oct 1, when the Funhouse budget renews; the mirror
needs a day or two for 5.31.0):** install 5.31.0; `probe(...)`; a small
`score_review_suite(... arms=("baseline", "lean"), ids=[...])` run to see cost
per task; then the arms that matter. The results decide the look primitive
(REVIEW_ARCHITECTURE S1.0), which the digest's page notes (M2) reuse.

**Not released:** planlens branch `fix/advice-names-no-missing-tools`
(e1caadc, drawing-sheet advice no longer names geometry tools); the app
does not need it.

### 5.30.0 RELEASED 2026-09-27 (tag `v5.30.0`) — THE DOCUMENT REVIEW EVAL SUITE + SWITCHED HARNESS CHANGES

**Released on the owner's word ("push and release as 5.30"); no dependency
change** (pin stays `planlens>=0.10`). **Release gate, in a CLEAN
worktree of exactly the release content** (`git worktree add` at 8cf0af0,
`PYTHONPATH` pointed at it, planlens at 0.10.1 = PyPI), foreground chunks on
pytest's exit code: `funhouse_agent/deep` + `review_eval` + webapp +
`geo_project` 892 passed / 1 skipped; the rest of `funhouse_agent` 1,458 / 6;
`report_ingest` 1,296 / 0. Only `funhouse_agent` and `webapp` changed since
5.29.1, and the rest of the gate does not import them (5.29.1's 13,332 / 36
stands for it). The new tests also pass on the drift stack (deepagents
0.7.13 / langchain 1.3.18). Wheel: `review_eval`, `review_flags`,
`review_agent`, `sweep`, `inline_images` present; no tests; requires
`planlens>=0.10`.

**Why.** The owner asked for a big-picture review of the Document Review
harness because sessions kept overfitting it to single examples. The review
(plan `C:/Users/socon/.claude/plans/we-ve-been-working-on-synchronous-lemon.md`)
found no evaluation set, vision by one-shot side calls the reasoning model
never sees, a page agent that was the geotech builder minus modules (29
tools, a coding-agent prompt), ~17 model calls per turn before a
GraphRecursionError, and citations one page early.

**What shipped.** `funhouse_agent/review_eval/` — 29 tasks over 16 public
documents (ten Mecklenburg stroke-lettered standard details + the ten bound as
a set, three UFCs incl. a scanned one, two calc packages, two small planlens
fixtures), deterministic checks, and the notebook cell `score_review_suite`.
Behaviour changes are SWITCHES, all OFF by default
(`funhouse_agent/review_flags.py`, arms `baseline` / `lean` / `grounded` /
`inline` / `sweep`): the lean review agent, text-layer context and structured
locations for vision calls, the main model looking at images itself, and
`sweep_pages`. Unswitched: `pdf_page` in results, the citation rule, the
`[G/Q]CE` example removed, tile reads in the run's context. An independent
review of the build found 9 bugs; all fixed and regression-tested.

**Owner's next steps (Funhouse, once the mirror has 5.30.0).** Full cells in
`module_work/REVIEW_HARNESS.md`:
1. Upload the public PDFs (`collect_public_docs()` on a checkout) to a
   cluster folder.
2. `probe(PrompterChatModel(prompter=fh_prompter, model="funhouse-gpt-high"))`
   from `funhouse_agent.deep.inline_images` — `ok: True` before trusting the
   `inline` arm.
3. A small `score_review_suite(... arms=("baseline", "lean"), ids=[...])`
   run, then the full arms. Record RESULTS.md in `REVIEW_HARNESS.md`'s ledger.
4. Tester feedback becomes a private TASK (JSON on SharePoint), not a prompt
   rule. A switch goes on by default only when the suite says so.

**Follow-ups noted by a peer session (not in this release):** the deep
reviewer prompt still says "You have 4 tools"/"<tool_call> tags"
(`reviewer.py`); structural references called "still being onboarded"
(`review_checklists.py:662`, `.claude/agents/structural-calc-specialist.md`);
dev-team lines in three specialist preambles; the geotech prompt tells the
agent to delegate to `calc` even when that sub-agent is off;
`.claude/agents/seismic-reviewer.md` lists deleted hvsrpy/swprocess.

### 5.29.1 RELEASED 2026-09-26 (tag `v5.29.1`) with planlens 0.10.1 (tag `v0.10.1`) — FIND_LIKE KEPT OPTIONAL

**Released on the owner's word ("Yeah go ahead"); no dependency change.**
**Release gate: 13,332 passed / 36 skipped / 0 failed** in foreground chunks
(same count as 5.29.0 — wording only). **A second session
("document-review-architecture") was editing uncommitted files in the same
working tree during the gate**, so the packages that import `funhouse_agent`
were re-run in a CLEAN worktree of exactly the tagged content (`git worktree
add` at d92122f + the four release files, `PYTHONPATH` pointed at it):
`funhouse_agent` 1,820 passed / 7 skipped, webapp + `geo_project` 453,
`report_ingest` 1,296 — all green. The rest of the gate does not import
`funhouse_agent`. planlens 0.10.1: 1,430 passed. Wheel: no `find_like` in the
prompts, requires `planlens>=0.10`. **Lesson:** two sessions sharing one
checkout must branch; release commits stage by exact path.

**Why.** 5.29.0 wired `find_like` into both system prompts, the `! look:`
hint on every drawing/scan result and the note on every stroke-lettered
sheet, and turned the IZD case's specifics into rules. The owner: "This seems
kinda niche. The example I gave was one of hundreds that could be run. I
think you may have overfit a solution to this one example. Not that we need
to get rid of the tool, but can you confirm that the setup is such that this
is an optional tool amongst many, and not viewed as a standard part of the
workflow?" Memory: `feedback-no-overfitting-to-one-example`.

**What changed.** `find_like` is known to the agent only through its own tool
description ("use it when EVERY occurrence of one repeated mark across many
sheets is wanted"). Out of the prompts, the look hint
(`document_tools.VISION_HINT`), the stroke-lettered note
(`vision_view.view_payload`) and planlens' render note. The rebar search
terms and "instances means callouts" became general rules (the text layer
is often where CAD words are; zoom before calling anything unreadable; zoom
closer on a bracketed character; never conclude absence from whole-sheet
views; say how an ambiguous request was read). Everything else in 5.29.0
stands. **Testers should install 5.29.1, not 5.29.0.**


### 5.29.0 RELEASED 2026-09-25 (tag `v5.29.0`) with planlens 0.10.0 (tag `v0.10.0`) — FIND_LIKE AND ROBUST-FIRST VISION (superseded by 5.29.1 above)

**Released on the owner's word ("Let it rip"), planlens first; no new
package** (pin `planlens>=0.10`). **Release gate on this tree: 13,332 passed /
36 skipped / 0 failed** in foreground chunks on pytest's exit code (1,944/14
fast analysis modules + `calc_package` + wrapped agents and I/O; 944/0 the
other analysis modules; 453/17 `slope_stability` alone; 399/0 `fem2d` in four
pieces; 1,822/5 `funhouse_agent`; 1,296/0 `report_ingest`; 486/0 webapp +
foundry harness; 195/0 `validation_examples` in two pieces; 5,793/0
geotech-references). planlens 0.10.0: 1,430 passed. Wheel: `find_like.py`,
`vision_probe.py`, `vision_view.py` present, no tests or dev folders,
requires `planlens>=0.10`. Also in this release: the two vision tools
(`analyze_pdf_page`, `find_like`) get a 32,000-character result cap
(owner OK'd raising caps; everything else unchanged).

**What happened.** A tester ran the Document Review page on Funhouse (5.28.0)
against an 85-sheet 90 % submission (SBU; the owner shared only the
transcript + a screenshot — the sheets stay private): lettering 0.06 in,
DRAWN AS LINES (no text layer), a penetration legend on every sheet. Asked
for every "GCE" penetration, the agent read the legend as "QCE" from a
whole-sheet image, then ran 52 whole-sheet vision calls for QCE and found
none of 34 callouts; it never zoomed. The images were 1187 x 768: the probe
had read Funhouse GPT-5.4 (`high` capped like tiles, `original` honoured —
owner's probe: 714 / 714 / 4,234 image tokens) as a tile model and never
used `original`. The owner then clarified "instances" meant the CALLOUTS
(tag + leader), not the legend rows — and set the direction: **robust first,
cost is fine, dial back for efficiency later** (knobs recorded in
`module_work/FUTURE_IDEAS.md` "VISION EFFICIENCY").

**What shipped.** planlens 0.10.0: `find_like` / `like_sheets`
(`planlens/document/findlike.py`; image-matching from one example box, 13
scales x 4 rotations, legend / callout (+ `points_to`) / unanchored from the
geometry), the `find_like` toolkit/MCP tool, the stroke-lettered fixture
`planlens.testing.tag_fixtures`, `budget_from_probe` using `original`
whenever honoured, `text_size` ignoring sub-2 pt text, renders reporting
`text_chars`. App: `funhouse_agent/find_like.py` + the `find_like` tool
(every candidate READ by vision on contact sheets; bracketed confusables ->
uncertain; look-alikes rejected; instances by page, callouts with leader
targets, legend counted apart; contact sheets saved to the conversation
folder), `GEOTECH_VISION_POLICY` (`robust` default: every image at the
largest honoured detail; `efficient` = the old split), `analyze_pdf_page`
`tiles="auto"` (overview + N x N tiles when lettering < 12 px), the
confusable instruction on every vision prompt, the stroke-lettered-sheet note,
and prompt rules (instances = callouts; zoom one copy then `find_like`; never
conclude absence from whole-sheet views).

**Owner's next checks (Funhouse, once the mirror has 5.29.0).** Connection
diagnostics: `vision model: gpt-5.4-...: images at gpt-4.1-high, charts at
openai-original (from probe)` — and under the robust policy every image goes
at `original`. Re-run the IZD question ("find every GCE penetration callout,
not the legend"): the agent should zoom to the legend, confirm GCE, call
`find_like`, and report ~34 callouts by sheet with contact sheets in the
folder. If it reports far more candidates than expected, the dense plan
lettering is producing look-alikes — the contact sheets show what was read.


### 5.28.0 RELEASED 2026-09-24 (tag `v5.28.0`) with planlens 0.9.0 (tag `v0.9.0`) — THE VISION MODEL IS MEASURED (superseded by 5.29.0 above)

**Released on the owner's word, planlens first; no new package** (pin
`planlens>=0.9`). **Release gate on this tree: 13,321 passed / 36 skipped /
0 failed** in foreground chunks on pytest's exit code (1,944/14 fast
analysis modules + `calc_package` + wrapped agents and I/O; 1,397/17 the
slower analysis modules; 399/0 `fem2d` in five pieces; 1,811/5
`funhouse_agent`; 1,296/0 `report_ingest`; 486/0 webapp + foundry harness;
195/0 `validation_examples` in four pieces — it outlasts a 10-minute call on
a loaded machine; 5,793/0 geotech-references). planlens 0.9.0: 1,417 passed.
Wheel: `vision_probe.py` + `vision_view.py` present, no tests or dev
folders, requires `planlens>=0.9`.

**What happened (the owner's first Tiny Apps run, dosdev, 5.26.0).** Uploaded
`BD-AB_01-26_0.pdf` (9 half-size 17x11 bridge standard sheets, 5 pt CAD
lettering, a full text layer). The automatic orientation turn was good (the
owner: "which I love"); asked for "a page-by-page summary of the rebar
requirements" the agent said the sheets were too blurry and asked for a
higher-resolution file. Diagnosis: (1) all 75 rebar callouts ("#5 BARS @
1'-0" MAX. SPACING", "#8 BARS (TOP)", "#4(E) BARS @ 8" MAX.") are EXACT
TEXT — the agent never needed to look; (2) `tinyapp-gpt-medium` answers as
**`gpt-5.1-2025-11-13`** (API `model` field; the model's self-report said
"GPT-4.1 family"), a TILE model: a whole sheet arrives 1187x768, the 5 pt
lettering ~4.8 px — genuinely unreadable; it ACCEPTS `detail="original"` and
ignores it (630 image tokens either way), so 5.27.0's retry-without-detail
never fires; (3) the agent gave up instead of zooming.

**What shipped.** planlens 0.9.0: `budget_for_model`, `budget_from_probe`
(tile / 2,500-patch / 6,144-patch and whether `original` is honoured, from
token RATIOS), budget `gpt-5.2-high`, `Document.text_size`, render
`text_px`, `legible_window`, toolkit legibility note. App:
`funhouse_agent/vision_probe.py` (the probe; cached per model per process;
`GEOTECH_VISION_PROBE=0` off), `vision_view` choosing env override >
measured profile > defaults for views AND charts, results carrying
`vision_model` / `text_px` / `legibility`, the text-layer-first and
zoom-before-giving-up rules in both prompts, and a "vision model" line in
Connection diagnostics.

**Owner's next checks (Tiny Apps, once the mirror has 5.28.0).** Remove any
`GEOTECH_VISION_BUDGET` / `GEOTECH_CHART_BUDGET` lines from `.env` (the probe
now decides; keep them only to override). Connection diagnostics should show
`vision model: gpt-5.1-2025-11-13: images at gpt-4.1-high (from probe)`.
Re-ask the rebar question on BD-AB_01-26: the answer should come from the
text layer, with zooms to see where the bars go, and no "too blurry".
`which_model.py` (the owner's one-off script) is superseded by that line.


### 5.27.0 RELEASED 2026-09-23 (tag `v5.27.0`) with planlens 0.8.0 (tag `v0.8.0`) — VISION SIZED TO THE MODEL (superseded by 5.28.0 above)

**Released on the owner's word ("go"), planlens first.** No new third-party
package: planlens 0.8.0 declares exactly what 0.7.0 did; the pin moves to
`planlens>=0.8`. Everything else from the 5.26.0 entry below still stands
(the Tiny Apps items are unchanged).

**Release gate on this tree: 13,309 passed / 36 skipped / 0 failed**, run in
the foreground in chunks each judged on pytest's own exit code (the machine
was short of memory for one long run): 2,165/17 analysis modules + 142/3
`calc_package` (NOT in pyproject `testpaths` — run it by hand) + 1,034/11
wrapped agents and I/O + 399/0 `fem2d` (three pieces; it outlasts a
10-minute call) + 1,799/5 `funhouse_agent` + 1,296/0 `report_ingest` +
681/0 webapp, foundry harness and validation examples + 5,793/0
geotech-references. planlens 0.8.0: 1,393 passed. Wheel checked:
`funhouse_agent/vision_view.py` present, no tests / module_work / tinyapps /
raw content, metadata requires `planlens>=0.8`; dependencies otherwise
identical to 5.26.0.

**Why.** The Opus 5.5 system card (§8.13: Chartography 64 % → 89 % with a
container + an image-cropping tool) and Anthropic's zoom-tool cookbook,
checked against OpenAI's "Images and vision" guide. A vision model SHRINKS
every image to its own limits before it looks, and the app never set the
`detail` level, so on GPT-5.4 (`funhouse-gpt-high` — owner, 2026-09-23)
every image was capped at ~1.6 MP, and a chart read-off rendered the whole
page and then lost resolution to the server's resize.

**What shipped.**
- **planlens 0.8.0** — `planlens.document.budget` (named budgets:
  `openai-high`, `openai-original`, `gpt-4.1-high`, `claude`,
  `claude-hires`; `fit_size`; `image_box_to_page`);
  `Document.render(budget=, fmt=)` renders to exactly the largest image the
  model reads unshrunk and re-draws a zoom from the PDF to fill it;
  `render_region(image=, image_box=, box_units=px|norm1000)`; `fmt="auto"`
  (PNG for drawings, JPEG for scans — measured: on a vector sheet PNG is
  about HALF a q92 JPEG); MCP `--image-budget` / `--image-format`; a render
  is now the size it reports (the "2000 px" page was 2016 px).
- **app** — `funhouse_agent/vision_view.py` is the one place for size,
  detail and location convention. `analyze_pdf_page`, `render_region`,
  `read_reference_figure` and `view_worked_example_source` render through
  it; every vision prompt asks for locations as 0-999 boxes and every result
  returns a `view`; `render_region(view=, image_box=)` zooms on what the
  vision call found (the agent never sees images itself, so this is its
  zoom loop). `LangChainVisionEngine` labels bytes by their real type (it
  always said PNG), sends `detail`, and retries ONCE without it if a model
  refuses the value. Settings: `GEOTECH_VISION_BUDGET` (default
  `openai-high`), `GEOTECH_CHART_BUDGET` (default `openai-original` — chart
  read-offs, owner-approved: "only a few readoffs every conversation"),
  `GEOTECH_VISION_DETAIL` (override; `none` omits the field).

**First live checks (owner, cluster).** (1) A chart read-off
(`read_reference_figure`, e.g. DM7.2 Fig 4-12 Kp) — the result's `view_px`
should be about 2800 x 3600 on a letter page, and the value should be at
least as good as before. (2) `analyze_pdf_page` on a drawing sheet, then ask
the agent to zoom on something the analysis located — it should call
`render_region` with `view` + `image_box`. (3) On Tiny Apps the Prompter
deployment is unknown: if it rejects `detail="original"` the engine retries
without it and nothing fails.

**Parked from this train** (`module_work/FUTURE_IDEAS.md`, "VISION HARNESS
BENCHMARK"): per-figure BOXES in the geotech-references figure catalog, so a
chart read-off renders just the chart instead of the page; a code-on-image
path for raster charts; scoring the harness on Chartography and BenchCAD
and on a small geotech chart truth set.


### 5.26.0 RELEASED 2026-09-22 (tag `v5.26.0`) with planlens 0.7.0 (tag `v0.7.0`) — THE TINY APPS BUILD (superseded by 5.27.0 above)

**Released on the owner's word ("you can release"), planlens first.** Two
new dependencies (`python-docx`, `markdown-it-py`); pin `planlens>=0.7`.
Nothing has yet run on the cluster or on dosdev — the owner's test list and
the questions for CfA are (b)–(c) below and `tinyapps/TINYAPPS.md`.

**What happened.** The owner opened the Tiny Apps session with three
uploads — CfA's `exampleCode` repo, the Tiny Apps User Guide v1.1 and the
Pilot Guidelines (all local-only under the gitignored `tinyapps/reference/`;
Department documents are NEVER committed, owner's rule) — and two decisions:
the app on this host exists for what AIP Chatbot cannot do (**vision** and
**document creation**), and it is **two pages over ONE app**, never a second
copy. Everything below was built in one night: the plumbing by the lead, the
document-output tools by an Opus builder the lead reviewed, then the shell.
Commits on master: `8315581` (plumbing + pages), `15c291a` (document output +
shell), `40efccb` (routing fix); planlens `47ceaa2` (markup writer).
`tinyapps/TINYAPPS.md` is the plan of record and was rewritten from the
sources — read it before touching any of this.

**(a) What exists — app, on master.**

| Piece | Where | Note |
|---|---|---|
| Two pages, one shell | `webapp/profiles.py`, `webapp/tinyapps_entry.py`, `webapp/app.py` | Document Review = root-served default (`st.Page(default=True)` takes NO `url_path` — Streamlit says "Page not found" otherwise); GeotechStaffEngineer = `/geotech`. Each page function calls `set_page_config` (tab title follows), flags `_page_config_set`, records the profile, `runpy`-runs `app.py`. A page switch parks and resumes each page's conversation (`_page_threads`). |
| Review agent | `funhouse_agent/deep/prompt.DOCUMENT_REVIEW_PROMPT`; `build_deep_agent(system_prompt=…)`; empty `allowed_agents` → no dispatch tools | Profile kwargs: `allowed_agents=()`, `reference_mode="off"`, `enable_calc_subagent=False`, `extra_system_prompt=None`. Specialist picker and the three geotech behaviour controls hidden; placeholder text differs. |
| Orientation turn | `app.py` upload handler → `ss.pending_orientation` → the chat-input block sends `profile.orientation_request` as the USER's message | Automatic (owner: "run it automatically"); only when an agent exists and no turn is running; ignored on a page whose profile does not orient (test-pinned). |
| Settings | `webapp/tinyapps_settings.py` | CfA's `get_setting`: env var → Key Vault (`KV_NAME`, managed identity, `IDENTITY_ENDPOINT` ⇒ azure) → `.env` (`GEOTECH_DOTENV`). Hyphen/underscore names equivalent. |
| Engine | `webapp/tinyapps_engine.py` | `ChatOpenAI(base_url=<PROMPTER_URL minus /chat/completions>, api_key, default_headers={"api-key"}, http_client=httpx(verify=CA ctx), max_completion_tokens)`; `register()` installs the model builder and publishes the model to the picker. `GEOTECH_PROMPTER_DISABLE_STREAMING=1`. |
| Identity | `webapp/identity.py` | `X-Windows-Auth-Header` via `st.context.headers` (two values ⇒ trust neither), `DEV_IDENTITY`, `GEOTECH_USER_EMAIL`, else `ANONYMOUS`. `Identity.key` (folder), `display_name`, `markup_author` ("<user> via GeotechStaffEngineer (AI draft)", or `GEOTECH_MARKUP_AUTHOR`). |
| Per-user roots | `core.register_thread_root` / `thread_root`; `profiles.session_root` | `<data root>/users/<key>/<profile>`; anonymous on the geotech page = the data root (legacy layout). The detached worker finds the root by thread id. Sidebar lists `list_conversations(_ROOT)`; delete/rename pass `_ROOT`. |
| SharePoint | `webapp/graph_sharepoint.py`; `sharepoint_store` prefers it | SDK-free Graph over an app registration (`GRAPH_TENANT_ID/CLIENT_ID/CLIENT_SECRET`, `SHAREPOINT_SITE_URL`), same six methods. Mirror folder `conversations/<owner>/<page>/<name>` when the meta is tagged (`core.tag_conversation`, done at the first turn on an identified or non-default session); untagged = old layout. |
| Keyless posture | `engine_config.is_tinyapps_deployment` / `is_keyless_deployment`; `diagnostics` | No personal key read or named; the banner names `PROMPTER_*`. |
| Word output | `calc_package/docx_renderer.py`; `write_docx` in `vision_tools` + `deep/tools` | Markdown → .docx via python-docx (NEW dep) parsed with markdown-it-py (NEW explicit dep); saves through `save_file`'s writer. Hides itself without python-docx. `.docx/.xlsx/.pptx` MIME in `app.py`. |
| Marked-up PDFs | planlens `document/markup_writer.py` + toolkit `annotate_document`; app bridge in `document_tools` + `deep/tools` | note / highlight / box / callout / reply onto a COPY; anchors quote (exact→fuzzy, narrowed to matched words) / bbox / point / reply_to (/IRT + /RT /R); read back by planlens' own `markups()` incl. rotated pages; `<document>_marked.pdf` in the working folder; `markup_author` threads `build_deep_agent → build_primary_tools → make_vision_tools`. Pages are 0-based. |
| Wrapper repo | `tinyapps/wrapper_repo/` | `app.py` (sets `GEOTECH_DOTENV`, `GEOTECH_WEBAPP_DATA=/home/data/…` on App Service), `packages.txt` (pin ≥5.26 + CfA fleet pins + Key Vault libs), `run.sh` (CfA template: `$PORT`, `corsAllowedOrigins CHANGE-ME`), `.env.example`, `README.md`. |
| Classic Funhouse host | `webapp/pages_entry.py`; `databricks_launcher.run_on_databricks(pages=True)` (commit `b289d83`) | The same two pages on Databricks: the launcher boots `pages_entry.py` by default; `pages=False` boots the single `app.py` as before 5.26. `Identity.multi_user` (the proxy header only) is what namespaces folders — the launcher's `GEOTECH_USER_EMAIL` shows "Signed in as" and signs markups but leaves the owner's geotech conversations where they are; the review page lives in `<data root>/pages/document_review` and mirrors to `conversations/document_review/…`. **UNVERIFIED LIVE: page links behind the driver proxy** (the proxy strips the prefix; `server_baseUrlPath` cannot be set) — the first cluster run of 5.26 should click between the pages and reload `/geotech`; if the links break, `pages=False`. |

**(b) Gate, 2026-09-22 (foreground, pytest exit codes):** planlens **1,331
passed**; app **2,283 passed / 8 skipped** over `webapp/tests`,
`funhouse_agent/deep/tests`, `funhouse_agent/tests`, `calc_package`; plus a
real `streamlit run tinyapps/wrapper_repo/app.py` with `DEV_IDENTITY` set,
checked in Chrome: both pages render, switch, tab titles follow, "Signed in
as jdoe", the specialist picker only on the geotech page. (Locally the
wrapper needs `PYTHONPATH=<repo>` because the app is not pip-installed into
the dev venv; on dosdev it is.)

**(c) PICKUP LIST, in order.**

1. **RELEASED 2026-09-22:** planlens **0.7.0** first (`annotate_document`,
   `Document.tobytes`, `frame.from_display_*`, `ReviewToolkit(author=,
   output_root=)`), then the app **5.26.0** with the pin `planlens>=0.7` —
   `annotate_document` is feature-detected, so a 5.26.0 on a 0.6.0 planlens
   simply lacks it. Two NEW dependencies (`python-docx`, `markdown-it-py`) —
   watch the Nexus install log; the mirror delivers a day or two after PyPI.
2. **The owner's dosdev run** (recipe in `TINYAPPS.md`). **DONE 2026-09-23
   as far as it can go without values:** `geotech-staff-engineer==5.26.0`
   installed from the mirror on dosdev and `streamlit run …/webapp/
   tinyapps_entry.py` booted the two-page app with the "No model
   configured" banner. Still to do there, once the values arrive: install
   from `packages.txt` in a venv (the CfA fleet pins — pandas 3.0.2,
   streamlit 1.62.0 — beside our tree; our suite has NOT run on pandas 3),
   fill `.env`, Connection diagnostics, upload a PDF → the orientation
   turn, ask for a marked-up copy and a Word summary, Sync now.
3. **Blocked on CfA:** Prompter URL / model / key / CA bundle — **the
   deployment must accept IMAGE inputs** (make-or-break; ask first); the
   published origin for `corsAllowedOrigins`; the SharePoint app
   registration; confirmation the identity header is passed for pilot
   Streamlit apps; whether the gateway streams.
4. **First cluster run of 5.26 (owner):** `run_on_databricks(prompter=fh_prompter)`
   as always — it now opens on Document Review; click to GeotechStaffEngineer
   and back, reload on `/geotech`, confirm the old conversations are still
   listed on the geotech page and "Signed in as <you>" shows. If the page
   links go wrong behind the proxy, relaunch with `pages=False` and report.
5. **Repo slimming + example-product inventory: PARKED behind Tiny Apps**
   (`module_work/FUTURE_IDEAS.md`, section "REPO SLIMMING"). The DOS
   reviewer's "millions of tokens" is the REPO (14.6M by bytes, 60 % of it
   generated artefacts and dev corpora); the code is ~2M and the wheel
   ~1.8M, and none of the example / ground-truth folders ship in the wheel.
   Owner rulings 2026-09-22: `salib_agent` and `pystra_agent` stay; the
   worked-example extracts stay on GitHub; the example-product inventory
   gets a proper review first, then the generated files find a home.
6. **Not built, noted:** an SDK-free budget/spend line for the sidebar
   (`report_ingest.engine.CostMeter` + `PROMPTER_PRICES` are reusable; the
   Funhouse budget panel needs the SDK); the email tool is SDK-only and off
   on Tiny Apps; `report_ingest`'s `PrompterEngine` still expects an SDK
   prompter object (the ingest/library sub-agents are OFF by default anyway);
   restoring from SharePoint lists every owner's folders (permissions are
   per team site — acceptable for the pilot, filter by owner later); the
   Databricks launcher could adopt the two-page entry but does not yet;
   `write_docx` is on the deep agent and the v1 text-ReAct surface but NOT
   in `native_tools.OPENAI_TOOLS` (the legacy v1 native path the app does
   not use); a highlight's read-back box is ~0.84 IoU by design (the quads
   are the ink — see `markup_writer`'s module docstring), note/box exact.

**(d) Standing rules added on this train.** Department documents (guides,
example code, reports, the Funhouse SDK) are local-only under a gitignored
path — add the `.gitignore` line BEFORE copying. The identity header is
trusted on the network restriction alone and is never typed into the app.
`st.Page(default=True)` takes no `url_path`.

### PARKED 2026-09-21 — report-ingest train at 5.25.0; NEXT: Tiny Apps (new session)

**The owner's word parks the train here**, at a released version, with nothing
half-built. Start from this entry; the 5.25.0 entry immediately below it is the
detail behind every claim made here.

**(a) What exists.** One pipeline reads a geotechnical report PDF into a
`ReportRecord` and writes four things out of it: a one-page summary, a
WikiLLM-style library page, a DIGGS 2.6 file and a library row. Every page gets
its label from a VOTE of three cheap voters, with the expensive review going
only where they split; a report bound inside another becomes its own linked
record; the boring logs, the laboratory sheets, the calculation printouts, the
test pits, the cone soundings and the dynamic probes are all read, each over a
deterministic floor that runs BEFORE a model call is spent; the narrative reader
answers the owner's two schemas under the conventions in
`report_ingest/narrative_glossary.py`; and the reports already read become a
library that can be asked questions across them. `score_on_cluster` has **nine
stages** — `labels`, `logs`, `lab`, `calc`, `soundings`, `narrative`,
`vision_labels`, `vote`, `ingest` — over ONE truth root. Two sub-agents ship
**OFF by default** (`build_deep_agent(enable_report_ingest=False)`, and
`report_library` feature-detected on the FOLDER rather than on a package
version), because none of it has run on the cluster yet. The hand truth, by
count: **4,300 labelled pages** over 14 reports, **15** logs, **31** laboratory
sheets, **10** calculation runs, **13** pits and soundings, **8** narratives —
all private, all gitignored. The upload folder is
`module_work/field_feedback/2026-09-16_report-ingest/raw/cluster_upload/` (about
2 MB, re-staged 2026-09-21 for this release), with `cluster_upload.zip` beside
it and its own README naming every path and what a run loses without it.

**(b) PICKUP LIST, in order.**

1. **Install 5.25.0 when the mirror delivers it.** It went to PyPI on
   2026-09-21 and reaches the cluster a day or two later:
   `%pip install "geotech-staff-engineer==5.25.0"` then
   `dbutils.library.restartPython()`. The install log has a standing answer in
   `docs/DATABRICKS_INSTALL.md` — do not re-derive the numpy cascade.
2. **Upload the private files.** Unzip `cluster_upload.zip` into the workspace
   folder the cells call `HOME`
   (`/Workspace/Users/<you>/geotech_app/report_ingest`), keeping its shape:
   `MANIFEST.md`, the hand label spreadsheet, `oos_labels.json`,
   `templates.json` **beside** `truth/` rather than inside it, and `truth/` with
   its five subfolders (`logs`, `lab`, `calc`, `soundings`, `narrative`). Then
   `%run` the Funhouse setup notebook — the restart wipes `fh_prompter` and
   `fh_sp_client` — and run the gather cell that copies `HOME` to
   `WORK = "/tmp/ri_files"`.
3. **The first live checks, exactly as the 5.25.0 entry lists them**, with its
   cost estimates: (a) `stages=("vote",)` over the 38 saved sheet-mode vision
   runs, **$0 — the stage calls no model**; (b) `stages=("ingest",)` with
   `max_reports=2`, `label_policy="structural"` and
   `review_mode="disagreements"`, order **$2–6**; (c)
   `stages=("calc", "soundings")`, the first model numbers either reader has
   ever had, order **$1–3**; (d) `stages=("logs", "lab", "narrative")`, with the
   5.24.0 floor, the templates and the narrative levers all in force for the
   first time, order **$3–6**; (e) `stages=("vision_labels",)` with
   `vision_mode="document"` over the corpus, order **$4–6**. The cells
   themselves are in `report_ingest/README.md` under "Running it on the
   cluster", in this same order. **Bring home `RESULTS.md` only.**

**(c) OWNER ITEMS — none of these can be done without the owner.**

1. **Rule on the six DRAFT conventions** in
   `report_ingest/narrative_glossary.py`. Confirming one is changing its
   `status` to `CONFIRMED`; overruling one is editing the sentence. They are:
   a count of explorations the report says it did NOT do is `0`, not null;
   `null` means THE REPORT DOES NOT SAY, never zero and never a word like
   "none" or "N/A"; the four "mention" fields answer on whether the report
   DISCUSSES the topic anywhere, not on whether the work was done; `postName` is
   the CITY of the post and nothing else; `primeAe` is the architect-engineer OF
   RECORD, not the geotechnical firm and not the contractor; and
   `earthHazardsExposed` uses the listed phrases only and never includes seismic
   shaking.
2. **Azure Document Intelligence is still owed** for the scanned reports, and
   only the owner can run it: R02, R31, R32 (58 pages in R32 alone read as
   confident nonsense without it), R38, the scanned pages of R03 and R33, and a
   fresh export of R17's truncated result. The WP0 harness prints the exact page
   ranges.
3. **BLIND truth sets are owed for the calculations and for the soundings.**
   All ten calc runs and all thirteen pit and sounding sheets were read while
   the prompts were being written, so `OPEN.txt` names every one of them and the
   scorecard prints that there is no blind set where the open/blind line would
   go. No number either stage reports is evidence about a report nobody looked
   at.
4. **The ingest ledger is the owner's design and is NOT built** — one function
   the ingest calls at the end of every report, writing one append-only row to
   every sink it can reach. Written up in full in `module_work/FUTURE_IDEAS.md`
   under "The ingest ledger"; nothing of it exists in code.
5. **`pydiggs` is not on the cluster**, deliberately — it is an optional extra
   demoted in 5.13.0 because it dragged Sphinx in. A DIGGS file a cluster run
   writes is checked by the ROUND TRIP only, and `RESULTS.md` prints `not
   checked here` where the XSD line would be. The schema gate runs in the
   offline gate, where pydiggs is installed. **DIGGS is validated locally only.**

**(d) WHAT IS UNMEASURED, and what the last measured numbers actually are.**

Still unmeasured: **the vote in production** (the policies were scored on saved
runs, never as the default inside `ingest_report`); **the calculation reader
itself**; **the sounding and pit readers themselves**; **the 5.24.0 narrative
levers**, which have never met the tier that will do the work; **the library's
model half**; and **document-mode vision over the corpus**.

| What | Last measured number | When |
|---|---|---|
| Page labels, rules alone | 0.908 in sample / **0.767 honest blind** | 2026-09-18, run 7 |
| Page labels, sheet-mode vision | 0.655 in sample / **0.867 honest blind**, about **$0.05 a report** | 2026-09-20, run 10 |
| Whole-page label review | about **$0.45 a report**; honest blind 0.767 → 0.850, but on in-sample reports it broke nearly as many labels as it fixed | 2026-09-18, run 7 |
| Log reader, 15 logs | all 83 % → 80 %; **blind 73 % → 62 %** — the reader lost to the grid, which is the argument for the floor | 2026-09-18, run 7 |
| Log floor over the grid | 283/521 → 373/521 with no template, **386/521 with the fingerprints** | 2026-09-20, no model |
| Log-template recogniser | 100 % recall and 100 % precision over both families, and no false positive on 30 pages the hand says are not logs | 2026-09-20, no model |
| Lab reader, 31 sheets | tables alone 68 % (452/667) → reader **87 % (716/825)** | 2026-09-18, run 6 |
| Narrative, 8 reports | **recall 64 % (151/235), precision 74 % (147/200)** | 2026-09-18, run 6 |
| Calculation reader FLOOR | program 80 % (8/10), inputs 73 % (64/88), results 48 % (36/75) | 2026-09-21, no model |
| Pit and sounding FLOOR | 73 % overall (257/353): **96 % tabulated (217/226), 31 % plotted (40/127)** | 2026-09-21, no model |
| Library retrieval | chosen query: reports 0.949 / 1.000, pages 0.939 / 1.000; search alone: reports 0.647 / 0.943, pages 0.417 / 0.323 | 2026-09-21, no model |

Every row marked "no model" is a floor or a rule and stands on its own. Every
row naming a run is from the cluster runs recorded in the private ledger,
`module_work/field_feedback/2026-09-16_report-ingest/MEASUREMENTS.md`.

**(e) Standing rules learned on this train.**

- **A release reaches the cluster a day or two after PyPI** (the Nexus mirror
  lag). Never assume a version is installable the day it is cut.
- **Never leave a paid output only in `/tmp`.** A cluster restart takes it and
  the money spent on it: mirror every run to the workspace folder or SharePoint.
- **Edit by exact text, and gate on pytest's own exit code** — never on a piped
  tail. That is how a SyntaxError shipped as 5.22.0 and had to be replaced the
  same hour.
- **One writer per file.** Two agents editing one document lose each other's
  work.
- **A cheap model reads whatever it is shown.** A whole corpus vision run was
  void because planlens captions each contact-sheet tile with the page kind and
  the model read the caption as the answer.

**(f) NEXT: Tiny Apps (a new session).** Plan of record `tinyapps/TINYAPPS.md`;
the "TinyApps pilot" section of `CLAUDE.md`; the Tiny Apps items further down
this file. The stubs exist: `webapp/tinyapps_entry.py` and
`tinyapps/wrapper_repo/` (`app.py`, `packages.txt`, `run.sh`). Engine wiring is
blocked on office-hours answer #1, the key-auth Prompter client. Nothing about
Tiny Apps is rewritten in this entry — those documents are the plan.

### app 5.25.0 — RELEASED 2026-09-21 (tag `v5.25.0`, master `46ae949`, on PyPI)

**One release carrying five builds that landed on master on 2026-09-21.** No
dependency change — the pin stays `planlens>=0.6` — no schema bump, seven new
modules inside `report_ingest`, one new reader in
`subsurface_characterization/diggs26.py`, and one more data file in the wheel
(`report_ingest/library_questions.EXAMPLE.json`, which moved out of the dev
harness so the cluster cell can read the shape from an installed copy).
**The owner cut the release on 2026-09-21: master `46ae949` is tag `v5.25.0`
and the version is on PyPI. It reaches the cluster a day or two later, when
the Nexus mirror picks it up.**

**1. The page labels are a VOTE, and the expensive review goes only where the
voters disagree** (`label_vote.py`). Rules 0.908 in sample / **0.767 honest
blind**; sheet-mode vision 0.655 / **0.867** at about **$0.05 a report**; the
whole-page label review about **$0.45 a report**, breaking nearly as many
labels as it fixed on the reports its prompt was tuned against. The two cheap
voters are COMPLEMENTARY by label class — the rules own `appended_report` (494
in-sample pages vision never emits), `other` (216) and `calculation` (1,009,
41 % missed); vision owns `plan`, `profile`, `photos`, `cover`, `toc` and
`figure` recall. So three cheap voters (rules, one sheet-mode vision pass on
the cheap tier, and the printed FORM where a private fingerprint file is in
force) label every page through ONE combiner that both `vote.py` (the scoring
stage) and `graph.ingest_report` (production) call. `label_policy` defaults to
`structural`; `rules` reproduces the old single-voter path and calls no vision
pass. `review_mode` defaults to `disagreements`: the review sees only the split
pages plus two either side, on `max(20, 0.5 × split pages)` instead of
`max(60, 0.25 × pages)`. `ReportRecord.page_labels` carries the chosen label,
its confidence, `agreed`, the policy, what settled it and every voter;
an unsettled split is a `QAEntry(kind="label_disagreement")`. Two leftovers
came with it: `out_dir` no longer refuses a `/Workspace` path, and a
`durable_dir` whose own name already IS the run's name is not nested inside
itself.

**2. A report bound inside a report is its OWN record** (`bound.py`). Triage
calls **19 of the 38 corpus reports `multi_document`**, one to four bound
documents each. Each bound range — the UNION of triage's claim and the record's
own `appended_report` runs, floor `MIN_BOUND_PAGES = 4` — goes round the same
loop into `out_dir/bound/<id>/` with its own record, summary, library page,
DIGGS file and library row (`parent` column). One extra structured call reads
the child's own title, firm and date. Nothing crosses the boundary: the
parent's counts stay the parent's, and the child's narrative reader is windowed
so the front-fifty-pages lever cannot hand it the parent's cover.
`ingest_bound=False` restores the old listed-and-skipped behaviour.

**3. The calculations are read** (`calc_reader.py`, `calc_scoring.py`). A
quarter of the corpus's 4,300 hand-labelled pages are calculation printouts —
**1,009 over eight of the fourteen labelled reports** — and every one was a QA
entry saying the pages existed and had been skipped. One run of `calculation`
pages in, one `Calculation` out: kind from a controlled list of thirteen,
program and version AS PRINTED, method, subject, the labelled values it was
given and the ones it worked out, a sixty-word summary, its pages and what it
could not settle. A deterministic floor of (label, value, unit) pairs before a
call is spent, a budget of two calls with most printouts costing one, three
Python gates. DIGGS ignores calculations deliberately. **Floor alone,
2026-09-21, no model and no network: program 80 % (8/10), inputs 73 % (64/88),
results 48 % (36/75)** over ten hand-truthed runs, six reports, seven kinds,
44 pages.

**4. The test pits, the cone soundings and the dynamic probes are read**
(`sounding_reader.py`, `sounding_scoring.py`). The corpus carries **251 test
pit pages, 54 DCP pages and 23 CPT pages** in nine of the fourteen labelled
reports, and all three were read as though they were borings. A pit is a LOG
and the log reader reads it, now including its dimensions off the BUCKET its
equipment field names. A sounding is not a log: tabulated sheets are read
deterministically with the one call on the header, plotted ones digitised
through `zoom_plot` against the axis ranges read off the plot's own text. Four
gates. DIGGS 2.6 gets `StaticConePenetrationTest` and `DynamicProbeTest` (2.6
declares no `DynamicConePenetrometerTest`), read back row by row. The
calculation GROUPING defect was fixed here. **Floor alone, 2026-09-21: 73 %
overall (257/353) — 96 % on the four TABULATED sheets (217/226), 31 % on the
nine PLOTTED ones (40/127)** over thirteen hand-truthed sheets, six reports.

**5. The reports already read become a LIBRARY that can be asked questions**
(`library.py`, `library_agent.py`). `Library(root)` over the folder the ingest
writes; the index is DERIVED and the records are the truth, so a folder
restored from SharePoint with no database rebuilds on the next question, keyed
off each report's own front matter, and nothing is written back into a record.
Ten query functions, every row carrying the report id and the PDF pages. FTS5
the way the reference layer does it, with a rapidfuzz fallback on the field
VALUES. `report_library` is a `CompiledSubAgent` plus one primary tool, **OFF
by default**, feature-detected on the FOLDER rather than on a package version,
bound to the app's own chat model, ceilings in Python (eight queries an answer,
25 rows and 4,000 characters a result), citations CHECKED rather than copied.
**Measured 2026-09-21, no model and no network**, twenty questions over a
synthetic library of six records: **the chosen query scores reports 0.949 /
1.000 and pages 0.939 / 1.000** (precision / recall); **search alone — the
question's own prose into `find()` — scores reports 0.647 / 0.943 and pages
0.417 / 0.323.**

**WHAT IS UNMEASURED UNTIL THE CLUSTER RUNS.** (a) **The vote in production** —
the policies were scored on saved runs, but `structural` as the default inside
`ingest_report`, the split fraction per report, what the disagreement-only
review touches and what it costs have not been run. (b) **The calculation
reader itself** — only its floor is measured, and **there is no blind set**:
all ten runs were read while the prompt was written, so every number the `calc`
stage prints is in sample and the scorecard says so in place of the open/blind
line. (c) **The sounding and pit readers themselves** — again only the floor,
again no blind set. (d) **The narrative levers that shipped in 5.24.0** — the
front-matter union, the glossary conventions, per-question retrieval, the
deterministic exploration fields and the quote gate have never met the tier
that will do the work, so the eight-report run's 64 % recall / 74 % precision
is still the last real number. (e) **The library's model half** — the
deterministic retrieval is measured, the sub-agent answering real questions
over the real corpus is not.

**Gate (2026-09-21, three chunks in the foreground, each gated on pytest's own
exit code): 13,108 passed / 33 skipped / 0 failed** (3,598/28 + 8,086/5 +
1,424/0). Wheel built and checked: no corpus name, no truth
file, nothing from `raw/`, no tests and no `module_work`;
`report_ingest/templates.json.EXAMPLE` and
`report_ingest/library_questions.EXAMPLE.json` both present; metadata requires
`planlens>=0.6` and does not require `anthropic`.

#### First live checks, in order

The cells themselves are in `report_ingest/README.md` under "Running it on the
cluster", in this same order, with the owner's own paths (`HOME` in the
workspace folder, files gathered into `WORK = "/tmp/ri_files"`, `out_dir` under
`/tmp`, `durable_dir = HOME + "/results/<run>"` and `sharepoint = fh_sp_client`
on every scoring cell). The **install log** question has a standing answer in
`docs/DATABRICKS_INSTALL.md` — do not re-derive the numpy cascade.

| # | Cell | What it proves | Cost estimate |
|---|---|---|---|
| 0 | `%pip install "geotech-staff-engineer==5.25.0"` then `dbutils.library.restartPython()` | the release installs; `planlens-0.6.0` in place, nothing else moved, no `anthropic` | none |
| 0b | `%run` the Funhouse setup notebook | the restart wiped `fh_prompter` and `fh_sp_client`; this puts them back | none |
| 0c | the gather cell | the private files land in `WORK = "/tmp/ri_files"` from `HOME`, `truth/` with its five subfolders and `templates.json` beside it | none |
| a | `stages=("vote",)` over the 38 saved sheet-mode vision runs | which policy the corpus actually supports, and whether `structural` stays the default; writes `vote/trust_table.json` | **$0 — the stage calls NO model**, it reads run files already on disk |
| b | `stages=("ingest",)`, `max_reports=2`, `label_policy="structural"`, `review_mode="disagreements"` | the vote in production: `## The page labels` prints the split fraction, the review mode, pages touched, splits left, and the final labels' accuracy beside the rules' alone; the `bound` column prints a line per child | **order $2–6 for the two reports.** By the run-7/run-10 rates: rules $0, the sheet-mode vision pass $0.05 a report, a disagreement-only review roughly a fifth of the $0.45 whole-page review, the readers the rest |
| c | `stages=("calc", "soundings")` | the two new readers against their hand truth — the first model numbers either of them has ever had | **order $1–3.** Ten calc runs at ≤ 2 calls each over 44 pages, thirteen sounding sheets at one call plus `zoom_plot` on the nine plotted, on `funhouse-gpt-high` (gpt-5.4 rates, $2.50 / $15.00 per M) |
| d | `stages=("logs", "lab", "narrative")` | the 5.24.0 floor and templates and the narrative levers, all in force for the first time; `templates_path` defaults beside `truth_dir` | **order $3–6.** Run 7 put 15 logs + 31 lab sheets + 8 narratives at about $3 in readers; the levers add front matter and retrieval passages to the narrative prompt |
| e | `stages=("vision_labels",)`, `vision_mode="document"` over the corpus | whether whole-report context wins back `appended_report`, `other` and the missed calculations that sheet mode cannot see | **order $4–6 for 38 reports.** run 9 measured one 151-page report at 5 windows / 255 k input tokens, about $0.10–0.15 a report on `gpt-4.1-mini` ($0.40 / $1.60 per M) |

Bring home **`RESULTS.md` only**. It carries IDs, labels, counts and rates. The
per-report runs, the triage profiles and the library answers stay on the
cluster, because a change's reason, a triage rationale or a cited quote can
name a firm, a project or a person.

**Then**, once those numbers exist: the blind truth sets owed for the
calculations and the soundings, the owner's confirmation or overrule of the six
DRAFT narrative conventions in `narrative_glossary.py`, and the ingest ledger
(the owner's design, written up in `module_work/FUTURE_IDEAS.md`, not built).

### 5.24.0 (2026-09-20, on master, NOT tagged) — the log-template recogniser and the narrative levers

**No dependency change.** Pure Python inside `report_ingest`, and no new model
call anywhere: both halves are rapidfuzz over text planlens has already
extracted.

**1. The log-template recogniser (`report_ingest/log_templates.py`).** The
owner's observation, 2026-09-20: two firms' logs are so standard that simple
rules would almost always catch them, and one of the two templates has shifted
over the years. The SHIPPED code is generic machinery; the FINGERPRINTS are
data in a private JSON file that travels with the truth folder
(`<truth_dir>/../templates.json`, or `templates_path=` on
`score_on_cluster`) and is never committed, because this repo is public and a
fingerprint names a firm. `recognise(doc, page, grid=None)` scores a page's
located text against every fingerprint — the footer stamp worth half, the
title block three tenths, the column headings two tenths — and returns the
family, a confidence, the margin over the best fingerprint of a DIFFERENT
family (a sibling scoring as well is the form drifting, not a doubt) and what
matched. Two uses: a VOTER (`template <family> (0.92)` beside the rules' and
the vision labels, and on the label review's ledger) and a KEY TO THE GRID (a
fingerprint's `column_map` names the columns `log_grid` cannot — a form that
stacks the sample id, the sampler code, the blow record and the recovery under
one heading of `DATA` gives the general vocabulary nothing to classify — and
`log_floor.seed_from_grid(..., template=match)` then reads them, keeping
`method="grid"` with the template in the note). `templates.json.EXAMPLE` ships
in the wheel with three INVENTED firms. With no fingerprint file anywhere,
every entry point is a no-op.

**Measured locally, no model and no network** (harness
`module_work/report_ingest_harness/measure_templates.py`, ledger
"Log-template recogniser"): 100 % recall and 100 % precision on both families
over the fifteen hand-truthed logs, none of the five logs on undescribed forms
claimed, and **no false positive on thirty pages the hand says are not logs**
(the threshold sits at 0.65, between the lowest true match at 0.97 and the one
near miss at 0.59, a laboratory sheet from the same gINT project).

**And the floor now reads what those columns print.** Four generic cell
readers were added to `log_floor` in the same train, because the column map
was worth almost nothing without them: a blow record printed with `+`
separators (`2+1+2`), a blow record printed one increment to a line in its own
cell (which both of the corpus's commonest forms do), a cell that names its own
result (`MC = 12.0%`, `LL = 38`, `REC=29cm, 64%`), and a sample named and typed
in one cell (`S-1, SPT`). The grid's floor over the fifteen truthed logs goes
**283/521 → 373/521 with no template at all, and 386/521 with the
fingerprints** — `n_value` 8/49 → 48/49, `blows` 13/58 → 56/58, and no log
loses ground.

**2. The narrative reader's accuracy levers.** From the per-question table of
the eight-report run (recall 64 %, precision 74 %), all default-on and each
switchable. (a) **The front matter always goes in**: `reading_pages` is the
union of every page labelled cover, letter, contents or narrative with the
first `front_pages=50` pages of the report, deduplicated, in page order,
capped by a token budget the front wins — `postName`, `primeAe`,
`primeContractor` and `projectNumber` live on the letter and the cover and
neither is a "narrative" page; a page in the set with under 120 characters of
text is sent as a PICTURE. (b) **A glossary and conventions block**
(`report_ingest/narrative_glossary.py`) as DATA the owner edits, carrying six
rules **marked DRAFT and awaiting the owner's confirmation** — counts of
explorations the report has none of are 0 not null; null means the report does
not say; the four "mention" fields answer on whether the report DISCUSSES the
topic; `postName` is the city of the post; `primeAe` is the architect-engineer
of record; `earthHazardsExposed` uses the listed phrases only and never
includes seismic shaking. (c) **Per-question retrieval over the whole report**
for the seven questions usually answered in a table, exact search first and
planlens' fuzzy search where exact found nothing, the passages carrying their
page numbers and those pages becoming citable. (d) **Deterministic answers**:
the five exploration fields come from the logs the labeller found when
`investigations=` is passed, with every difference recorded — but a ZERO never
overrules a stated number, because no cone logs read is not no cones pushed.
(e) **A quote gate**: an answer whose citation quote is not findable on the
page it cites (fuzzy ≥ 85) drops to confidence 0.3 and is listed, never
deleted; one with no citation keeps its confidence and is flagged.
(f) **Scoring** gains a LENIENT view of the four long free-text fields
(partial ratio 70, or a number and its unit in common) printed beside the
strict one, and the per-question table gains a **why** column whose
`convention` value means a house rule rather than a reading failure.
`measure_wp4_narrative.py` gains `--front-pages` and `--lenient`.

**The narrative levers are UNMEASURED.** Nothing here has been run against the
tier that will do the work; the next cluster run is what says whether they
move the 64/74. The template numbers above ARE measured, because that half
calls no model.

Suites: `report_ingest` **833**, harness **229**, docs-currency green.
`REPORT_INGEST_PLAN` §7 — the owner's standing review of the two query schemas
— now points at `narrative_glossary.py` as the place every ruling goes.
**Next:** the owner confirms or overrules the six DRAFT conventions, then
`stages=("narrative",)` on the cluster with `--front-pages` at its default.

### 5.23.0 (2026-09-20, on master, NOT tagged) — the readers' floor and the `ingest` stage

Built from what the first full cluster run showed (ledger runs 6–7): the log
reader re-emitted the record from the model's answer and LOST to the grid on
seven of ten blind logs (73 % → 62 %), and the lab reader came back below the
tables on six sheets. No dependency change; pure Python inside `report_ingest`.

**1. The floor** (`report_ingest/floor.py`, `log_floor.py`, `lab_floor.py`).
The deterministic pass is the FIRST voter and its values are the floor:
`seed_from_grid` (samples, blows, N, recovery, index values, layers with any
printed USCS symbol, header fields, the groundwater field) and
`floor_from_tables` (the title's kind, the printed link, every labelled table
value, grading series and summary row) are built BEFORE any call, shown to the
model as THE STARTING RECORD, and the answer is merged back: ADD accepted;
CORRECT only with evidence (a box and a note, or a note on a picture-read
value) — then the model's value takes the slot and the floor's stands in
`prov.alternatives`; a contradiction without evidence keeps the floor's value
in the slot with the model's beside it; NEVER DROP — an omitted value is kept
with a QA note. Every split is `QAEntry(kind="disagreement")` with both values
and both confidences. Within the scorer's tolerance = `reconciled`.
`Provenance.method` now names the voter; `Alternative` is the new model. The
log reader spends one follow-up call on its own unsettled list (rows
magnified through the ruler) when the budget allows. Scorers add
`model_alone` + disagreement/kept/added/reconciled counts; `after` IS
floor+model. Tests: a reply that drops everything scores no lower than the
grid/tables alone; a contradiction keeps both; evidence overrules; a symbol
the model adds is added.

**2. The `ingest` stage** (`stages=("ingest",)`, the seventh). The whole
graph per report into `out_dir/ingest/<ID>/` (record, summary, page, DIGGS
with both gates, `qa.json`, `run.json`; `ingest/reports.db`), reusing a saved
label run's triage and review from `runs/` or `review_dir`, resumable per
report and per item, mirrored, and scoring the record against the hand truth
in `truth_dir` with the SAME scorers. `RESULTS.md` gains `# Ingest: the
record and its exports` (per report: pages, workflow, investigations by kind,
samples, SPT, lab tests by kind, narrative answered/null, QA
disagreement/partial/out_of_range, DIGGS written/schema/read back, cost;
totals; scored-against-truth table). `graph.ingest_report` accepts an open
planlens document. README cell in "Ingesting whole reports".

Suites: `report_ingest` 777, harness 229, docs-currency green. Stage by
exact path; nothing under `raw/`. **Next:** the owner runs
`stages=("ingest",)` with `review_dir` at the saved label run (`max_reports=2`
first), then reads the blind-log floor effect and the whole-pipeline score.

### 5.22.1 RELEASED (2026-09-21) — 5.22.0 with its wording fixed; 5.22.0 on PyPI does not import

The lead's docstring edit by line number landed inside
`_FileManagerBackend.get` in `report_ingest/mirror.py` after the builder's
gate; the check that should have caught it read pytest through `| tail`, so
the exit code was tail's, and the tag went out. 5.22.1 is the builder's
5.22.0 tree with the wording done properly. **Install 5.22.1, never 5.22.0.**
Two rules from it: edit by exact text, never by line number; and read a
suite's exit code directly, never through a pipe.

### 5.22.0 (2026-09-21, tagged, BROKEN on PyPI) — the output survives a restart, and the disagreement becomes the answer

Two things, both in `report_ingest`, no dependency change and no model call
added anywhere.

**1. A durable mirror (`report_ingest/mirror.py`).** The first full cluster
run wrote 38 label reviews — about $17 of calls — into `/tmp/report_ingest_520`
and a restart wiped them. The owner, 2026-09-20: *"We should really be saving
the files somewhere other than tmp. Big waste of money. You know we've had
this issue elsewhere."* The app has mirrored conversations to SharePoint after
every turn since 5.10.0; this is the same for the scoring runs, in the shipped
package, with no dependency on the app. `score_on_cluster(...,
sharepoint=fh_sp_client, sharepoint_folder="GeotechStaffEngineer/report_ingest",
durable_dir=None)`:

- the run's remote folder is `<sharepoint_folder>/<basename(out_dir)>` with
  the SAME layout inside, so `/tmp/report_ingest_522` mirrors to
  `GeotechStaffEngineer/report_ingest/report_ingest_522/{runs,triage,vision,
  vote,RESULTS.md,results.json}` — where the owner has been copying runs by
  hand;
- **at the START** anything the mirror holds and `out_dir` does not is
  restored, so a wiped `/tmp` resumes rather than paying twice. The first
  printed line says where the mirror points and how many files came back;
- **after every run file, by every stage**, the out_dir is mirrored
  incrementally — a local `mirror_manifest.json` of (size, mtime), so the
  38th mirror sends one file, not 38 — and again at the end for `RESULTS.md`
  and `results.json`;
- **two backends behind one duck type**: a SharePoint file manager (the
  `fh_sp_client` itself is unwrapped, and so is the app's `SharePointStore`
  whose `file_manager` is a method) and a plain folder (`durable_dir` — the
  owner's workspace folder under `geotech_app/`, which persists; never `/tmp`
  or DBFS, the owner's rule of 2026-09-20). Both may be given at once;
- **a failure never stops the run**: one warning line per distinct error, and
  a count at the end. `out_dir` still refuses `/Workspace`; `durable_dir` does
  not, because whether a path is durable here is the owner's finding.

**2. The `vote` stage (`report_ingest/vote.py`, WP6).** The owner's standing
direction of 2026-09-18 — *"if multiple methods say different things, it could
trigger an extra review ... would be good to have confidence values"* — and
run 10's reading: rules and vision are COMPLEMENTARY by label class, not one
better than the other. `stages=("vote",)` calls **no model**; it reads the
rules (recomputed with planlens where the PDF is to hand, off the saved run
where it is not), the vision labels from `vision_dirs` (default this run's
`vision/`), and the review from `review_dir` (default `runs/`), and writes
five things into `RESULTS.md` under `# Vote: rules, vision and the review as
voters`: (1) the agreement rate over every page, and the accuracy of agreed
against disagreed pages — the confidence claim; (2) a per-label trust table
learned on the IN-SAMPLE reports only and applied to the OOS sets, keyed by
the RULES' label, ties to vision so "believe rules" always means strictly
better; (3) combined labels under `trust`, `structural` (vision except
appended_report / other / calculation / lab_test) and `confidence`, each
scored by the same scorer beside rules / vision / +review; (4) the
disagreement set — how many pages would go to a targeted review and the
accuracy that review would need to carry the set over 0.98, `>1.000` when it
is out of reach; (5) per report, blind sets excepted. Every split is listed
per report in `vote/<ID>.json` (page, both voters, both confidences, the hand
label, each policy's choice — labels and numbers only). The run files now also
save `rules_confidence`, planlens' own per-page confidence, so the
`confidence` policy does not need the PDF. Harness twin:
`module_work/report_ingest_harness/measure_wp6_vote.py`.

Suites: `report_ingest` 746 (56 new), harness 229 (16 new), docs-currency
green. **Install 5.22.1** (5.22.0 does not import); it supersedes 5.21.2 and
everything back to 5.20.1.

**Next on this stage**: run the vote over the 38 saved sheet-mode vision runs
the owner copied to SharePoint (`vision_dirs=[...]`) beside the label runs, if
those survived, and read off which policy to build on. Then the document-mode
re-run on the scanned report, and `vision_detail="low"` for accuracy.

### 5.21.2 (2026-09-20) — the whole-report vision mode, after its first cluster run

The 5.21.1 run answered the sheet-vs-page question and opened two new ones.
**Sheet mode is the winner and needs no further work:** with number-only
contact sheets it scored **0.932 strict on 307 pages** (0.947 and 0.917
on the two reports read separately) in **26 calls and about $0.04 a report** — level with page mode's 0.928 at a
sixth of the calls. **Document mode scored 0.927** on the one report of text
pages it finished (151 pp, 5 windows) but showed two defects, and both are
fixed here:

- **It cost 254,724 input tokens, ~51,000 a call.** Not a defect: the pages
  are A4, and A4 at 100 dpi scales to 768 x 1086, which is six 512 px tiles
  where a letter page is four. Recorded so nobody re-derives it.
- **Three pages came back unresolved** — the model skipped them and the
  window that overlaps each one skipped them too.
- **Every window of the 156-page SCANNED report failed** with
  `APIStatusError: The page was not displayed because the request entity is
  too large`. That is the gateway's REQUEST-BODY limit, which the 50-image
  probe (tiny images) never came near. A scanned page at 100 dpi is a few
  hundred KB of PNG; 36 of them plus four contact sheets, base64'd, is tens
  of megabytes.

**What 5.21.2 does about it.** (1) **Every page picture travels as JPEG** at
quality 80 (`vision_labels.encode_for_vision`, `VISION_JPEG_QUALITY`), at the
render's own pixel size — the provider scales and tiles the pixels, so **the
token count is unchanged** and only the bytes move. Measured: a scan-like page
is 780 KB of PNG and 250 KB of JPEG, a factor of about three. **On a crisp
vector text page PNG already wins** and JPEG runs 0.84 to 1.20 of it; the rule
is unconditional anyway, because the pages that refuse a request are the
scanned ones and a per-page choice would make one report's windows a different
size from another's for reasons no run file records. `engine.image_block` now
takes PNG or JPEG and **both engines read the media type off the bytes**
(`image_media_type`), so switching to smaller-of-the-two later is a one-line
change. (2) **A window the gateway still refuses splits itself**: the pass
halves it, puts both halves at the front of the queue, and **re-cuts every
window still queued** to the new size, so one refusal is paid for once. A
refused request bills nothing, so a split costs time and no money.
`DOCUMENT_MIN_SPLIT` is 4 — a window at or under it that is still refused
raises, because at four pages the body is small and the refusal means
something else. (3) **`fallback=True`** gives every page still unresolved ONE
page-mode call, on the same budget, after the sheet or document pass
(`vision_fallback` on `score_on_cluster`). A page goes unresolved because the
REPLY left it out; asking about that page alone is the mode that cannot skip
it. `cost["splits"]` and `cost["fallback_pages"]` are counted, printed on the
progress line and carried into the RESULTS per-report table as a `split`
column (a dash when none).

Suites: `report_ingest` 690, harness 213, docs-currency green. No dependency
change. **Install 5.21.2**; it supersedes 5.21.1 and everything back to
5.20.1.

**Next on this stage**, in order: re-run document mode on the scanned report
to see whether the JPEG alone clears the gateway or the split has to fire;
then `vision_detail="low"` for accuracy, which is still unmeasured and is the
only lever that moves a document run's token bill materially.

### 5.21.1 RELEASED (2026-09-18) — the vision passes see page numbers, not kinds

The sheet-mode corpus run (38 reports, 1,321 calls, ~$1.50) came back at
0.557 strict on 4,147 in-sample pages; page mode had scored 0.928 on two of
the same reports (R09 0.947 → 0.245, R11 0.910 → 0.308). Cause, found in the
code rather than the model: sheet mode used planlens' `render_thumbnails`,
whose sheets caption every tile "<index> <kind>" — the page's rule-derived
shape (text, figure, form, scanned) that the label review wants beside the
picture — and the vision prompt passed planlens' legend through. The model
did the sensible thing with a caption that said "text": narrative precision
0.43 with recall 0.89, figure precision 0.06 with recall 0.69, appended
reports never. The document mode's strip used the same sheets. Fix:
`vision_labels.render_number_sheets` draws the sheets in the package,
captioned "p. N" and nothing else; sheet and document mode use it; three
tests, one forbidding `render_thumbnails` in any vision call. **The
sheet-mode numbers of 2026-09-18 are void** (ledger run 8). Install 5.21.1
when it clears the mirror; it supersedes everything since 5.20.0.

**Owner's direction, same evening, now standing:** "even if something
scores worse, it could still be useful ... if multiple methods say
different things, it could trigger an extra review ... would be good to
have confidence values associated with the classifications and data
extractions." Rules, review and vision are three voters; agreement is
confidence, disagreement is the trigger for a second look. Every value in
the record already carries a `Provenance` with confidence and method; the
page labels need the same, and the vote is the design in
`module_work/FUTURE_IDEAS.md`, "Disagreement as a signal".

### 5.21.0 RELEASED (2026-09-18) — whole-report vision, and dollars

Tagged `v5.21.0` and on PyPI the same evening (the owner's standing word
covers releases). No dependency change. Superseded by 5.21.1 within the
hour, before it could reach the cluster.

**1. `mode="document"` in `report_ingest/vision_labels.py`.** The owner asked
for the whole report in context rather than a page at a time, and page mode's
own numbers said the same thing from the other side: it matched
rules-plus-review overall but lost on `plan` and `lab_test`, the two labels
that are settled by knowing which appendix a page sits in. So each call now
carries a strip of planlens contact sheets covering the whole report plus a
window of 36 consecutive full-size pages, each **stamped `p. N`** at its
top-left corner (a report restarts its printed numbering in every appendix, so
the index the pass scores by is drawn on the picture; the page size, and the
token count, are unchanged). From the second window on, the call also carries
the labels already decided as runs — `61-118: boring_log` — which is what an
appendix looks like written down. Windows overlap by 3; a page skipped in one
is answerable by the next, and where both answer the LATER answer wins.

**The cap that shapes it was measured on the cluster today: 50 images in one
request** ("Too many images in request: 51, maximum allowed: 50"). Not the
context window — 1M tokens would hold a 400-page report at ~770 tokens a page
and the image count would still refuse it. The strip is capped first (12
sheets, or whatever leaves the window 12 pages) and the window takes the rest:

| report | sheets it has | strip sent | window | images a call | calls |
|---|---|---|---|---|---|
| 151 pp | 4 | 4 | 36 | 40 | 5 |
| 426 pp | 9 | 9 | 36 | 45 | 13 |
| 729 pp | 16 | **12** | 36 | 48 | 22 |

The 729-page report is the one that gives up seeing all of itself, and it
gives up the far end: the twelve sheets NEAREST the window are sent.

**2. `vision_detail`.** `image_block(png, detail=...)` now carries OpenAI's
`image_url.detail` into `PrompterEngine._content` (only when set; the Claude
engine has no such key and ignores it), threaded through all three modes and
through `score_on_cluster(..., vision_detail=...)`. `"low"` is ~85 tokens an
image instead of a page's four tiles: ~16,000 input tokens per 100 pages
against ~97,000 at the default. **What it costs in accuracy is not known** —
that is the next run, and it is cheap.

**3. Dollars.** `engine.PROMPTER_PRICES` holds the owner's own four rates from
the Funhouse budget page, read 2026-09-18, keyed by the DEPLOYMENT that
answered (`served_by`) and never by the tier, because a tier is an alias and
the model behind it changes. Every RESULTS `## Cost` block prints dollars
beside its tokens; a call served by an unpriced deployment adds tokens and no
dollars, `CostMeter.unpriced_calls` counts those, and a total is therefore a
FLOOR. A run where nothing is priced still prints tokens alone.

**Estimated cost of a 100-page report at the `gpt-4.1-mini` rate:** page mode
~$0.11, sheet ~$0.02, document ~$0.05, document at low detail ~$0.01. Only the
page-mode token figure is measured (~2,500 a page on the cluster); the rest is
that overhead through the provider's tile arithmetic.

**Pillow is not a new dependency, but not for the reason assumed:** it comes
from `matplotlib>=3.8` and `streamlit>=1.39`, both core app requirements.
planlens and PyMuPDF declare no Pillow at all (checked against the installed
metadata, not inferred).

**What to run first on the cluster:** one `score_on_cluster` cell with
`stages=("vision_labels",)`, `vision_mode="document"`, its **own `out_dir`**
(a run file is keyed by report, not by mode, so a second mode written into the
first one's folder is skipped as already done) and `max_reports=2`. Then the
same again with `vision_detail="low"` in a third folder. Three RESULTS.md
files, three columns each, one scorer.

Suites: `report_ingest` 660 (52 new), harness 213, docs-currency green.

### 5.20.5 RELEASED (2026-09-18) — a rate-limited call waits; the first full run's numbers

The owner ran every stage but vision over the whole corpus on 5.20.0 plus
the notebook shim. Mechanically: 10 of 38 label reviews and 5 of 15 logs
died on the provider's 429 ("requests to gpt-5.4 for funhouse-gpt-high in
eastus2 have exceeded rate limit"); the SDK's client retries once with a
sub-second pause and the engine never waited. `PrompterEngine` now waits on
`RATE_LIMIT_WAITS` (15/30/60/120/120/120 s) or the provider's retry-after,
retries, and records each wait in `adaptations`; three tests. The results
file no longer prints "the same set minus ," when nothing is excluded.

The numbers (ledger "CLUSTER RUN 1", run 6; every model-dependent figure
in the plan is now REAL, gpt-5.4 for review and readers, gpt-5.1 triage):

- **Label review**, 13 in-sample reports, 4,082 pages: 0.907 → 0.915 strict
  (fixed 156, broke 122, still wrong 88). Open out-of-sample (10 reports,
  50 pages): 0.76 → 0.78 strict, 0.84 → 0.90 with alternates. Blind (5
  reports, 25 pages): 0.92 → 0.92. The Claude dev-engine checkpoint's
  0.889 → 0.945 did not transfer. Narrative recall rose (0.91 → 0.985) at
  the cost of precision (0.97 → 0.88); profile and figure improved a lot;
  toc, divider, field_test, appended_report precision fell; R24 is 0.48
  before and after. Triage called 17 of 28 reports `multi_document`.
- **Lab reader**, 31 sheets: 68 % → 87 % (blind 55 % → 87 %; kind and
  link 94 %; summary tables 100 %, gradation 86 %, triaxial 0 → 60 %,
  chemical 54 %, compaction 10 %). Six sheets REGRESSED below the tables
  alone (e.g. one gradation 28/28 → 11/30).
- **Log reader**, 10 of 15 logs: 7 of 10 lost 1–3 items against the grid
  alone (blind 73 % → 64 %); recovery 15/15 → 3/15 and layer tops are what
  it drops. The reader re-emits the record instead of building on the grid.
- **Narrative reader**, 8 reports: recall 64 %, precision 74 % (open 71/76,
  blind 62/73); identity, code and summary fields 8/8; misses are
  conventions — cptCount 7 missed (null vs 0), testPitCount 5,
  siteResponseMention 7 WRONG, earthHazardsExposed 5 invented,
  propertyType 4, postName 3, boringDictionary 4 — plus free-text lists
  scored at Jaccard 0.6.
- **Cost**: 4.3 M input tokens for the 28 reviews (~154 k a report, 37
  min); a log 25 s; a sheet 30 s; a narrative 54 s.

**Next train (owner informed 2026-09-18):** (1) a FLOOR under the log and
lab readers — the record starts from the grid/tables and the model may add
or correct with evidence, never drop; (2) the narrative conventions, which
are the owner's schema decisions; (3) the review's precision and R24-class
reports. Vision sheet-mode run over the corpus still to come.

### 5.20.4 RELEASED (2026-09-18) — a tuple in a schema becomes an array

Same day, same class of fault, found offline before the cluster could:
`ReadProv.bbox` is `Tuple[float, float, float, float]` in the log and lab
readers, pydantic writes it as `prefixItems` (tuple validation), and strict
mode does not implement tuple validation. `strict_schema` now turns a
`prefixItems` array into a plain array whose `items` is the arm type (or an
`anyOf` of the distinct arm types) with "exactly N items: ..." in the
description; pydantic still checks the tuple when the answer is parsed.
`prefixItems` and `additionalItems` join `STRICT_UNSUPPORTED`, so the
walk-every-model test covers them. One test. Also learned today: the
owner's third reader failure carried the ORIGINAL `max_tokens` refusal,
because the first notebook shim had patched the Prompter object that
existed at the time and the setup cell later made a new one; the shim now
patches the engine's `client` property so it attaches to whichever client
is in use. **5.20.4 is the one to install**; 5.20.1–5.20.3 are superseded.

### 5.20.3 RELEASED (2026-09-18) — the readers' schemas pass strict mode

Third lesson of the first cluster day, and the one that was actually
stopping the readers. After the parameter shim, the six reader items ran
and every one died with 0 model calls and 0 tokens. Cause: OpenAI's strict
`json_schema` response format validates the schema BEFORE the call and
refuses any keyword it does not implement; pydantic emits `default` for
every optional field and `minItems`/`maxItems` for every bounded list.
Counted with `strict_schema` on every model a pass sends: triage 0, review
0, vision 0 refused keywords; log reader 63, lab reader 149, narrative
reader 52 — precisely the passes that ran and the passes that did not.
`engine.strict_schema` now removes `STRICT_UNSUPPORTED` (default, min/max
items, numeric bounds, length bounds, pattern, format, ...) and appends the
limit to the field's description in words ("(1 to 3 items)", "(between 0
and 100)"), so the model still reads it and the Python gates still enforce
it; a single-value `const` becomes a one-entry `enum`. Two tests, one of
which walks every `output_format` model in the package and fails on any
refused keyword, so this class of failure cannot return unnoticed.
`report_ingest` 604. Until 5.20.3 clears the mirror the owner's notebook
carries a second shim that wraps `strict_schema` the same way.

### 5.20.2 RELEASED (2026-09-18) — a saved failure is retried, not skipped

The first cluster run's second lesson. The log, lab and narrative scorers
catch a failed model call and store it on the score (`after.error` for the
log and lab readers, `score.error` for the narrative reader); the stage then
wrote the blob as if the item had finished, and the NEXT run skipped all six
failed items as "already done" and rendered their errors in the results
tables. `cluster_scoring._saved_failure(blob)` reads a saved run for a
recorded failure and each stage retries such an item, printing "previous
attempt failed (...); retrying". Two tests (one on the logs stage end to
end, one on the three blob shapes). Also in this release: the first real
numbers are in the private ledger (`MEASUREMENTS.md`, "CLUSTER RUN 1") and
summarised in CLAUDE.md's state block — labels 0.902 → 0.928 on 307 pages,
vision-first 0.928. Like 5.20.1 this reaches the cluster a day or two after
PyPI; until then the owner clears frozen failures with a one-off notebook
cell that deletes any run file carrying an error.

### 5.20.1 RELEASED (2026-09-18) — the first cluster run's one failure, fixed

The owner's first `score_on_cluster` run on 5.20.0 stopped on its first
report: the model behind `funhouse-gpt-medium` (triage) refused `max_tokens`
and wanted `max_completion_tokens`; the SDK's `prompter.chat()` logged the
refusal and returned None, and `PrompterEngine` raised "Prompter returned
nothing". The fix is in `report_ingest/engine.py` and mirrors what
`funhouse_agent/deep/databricks_bridge.py` has done since the GPT-5 tiers:
a refused `max_tokens` is resent as `max_completion_tokens`, a refused
`temperature` is dropped, one retry per refused parameter, and the lesson is
kept on the engine (`token_key`, `send_temperature`) so later calls send the
accepted form first. When `chat()` swallows the refusal the call is retried
on the raw client and `chat()` is not offered again that run.
`engine.adaptations` records every lesson. Nine tests in
`report_ingest/tests/test_prompter_engine.py`, one of which replays the
cluster failure verbatim. No dependency change; the wheel is 5.20.0 plus
this. **The Nexus mirror delivers a PyPI release a day or two later**, so
the owner's 2026-09-18 runs are 5.20.0 plus a notebook shim (a wrapper on the
Prompter client that renames `max_tokens` and drops a refused `temperature`,
with `prefer_chat` forced off); label their results that way. 5.20.0
installed and imported on the cluster that day, which closes the
"cluster install unconfirmed" item for the 5.19/5.20 dependency set.

### 5.20.0 RELEASED (2026-09-17) — the report read end to end, with planlens 0.6.0

**Released on the owner's word.** `feature/report-ingest-wp2` merged into
master (no-ff) and tagged `v5.20.0`; planlens `feature/log-grid` fast-forwarded
into main and tagged `v0.6.0` FIRST, as for 0.5.0. Both tags pushed; the
`publish.yml` workflow on each repo builds and uploads to PyPI. Cluster install
of 5.20.0 is NOT yet confirmed (install guide §11).

**The published answer vocabulary — DECIDED, not a blocker.** The owner's own
answer vocabulary (compound and phase types) ships verbatim in the package;
the lead ruled it generic taxonomy, not corpus content, and the owner was
informed before the tag. It lives in `report_ingest/model.py` (commit
`5628ce1`), read off the hand answers rather than guessed.
(`report_ingest/tests/test_narrative_reader.py` repeats the same strings but
does NOT ship — the wheel carries no tests, verified by listing it — so GitHub
is its only exposure.) The values stay VERBATIM because the narrative scorer
matches strings at a rapidfuzz partial ratio of 85 and the hand answers are
the long form; shortening them would turn correct readings into misses and
move the numbers. **An option, not a blocker:** a GENERIC default list in
`model.py` with the real wording loaded at runtime from a private file
travelling with the truth folders. The two fields are already plain `str`
rather than `Literal`, so the vocabulary is only a prompt hint and a folding
table and nothing refuses an answer either way; it would be a follow-up
commit, not a history edit, and `5628ce1` and `17f728a` stand as they are.
Nothing else on either branch trips the privacy word list.

**What shipped.** WP2 through WP4 of the report-ingest train, on top of the
WP0/WP1b library that went out in 5.19.0. `ReportRecord` is the product and
the summary page, the library page and the DIGGS file are exports of it. A
number keeps the unit it was printed in; every value carries the page, the
box and how it was read (`text` / `di` / `ocr` / `grid` / `vision` /
`derived`); what could not be read is recorded rather than guessed. The three
readers: the **log reader**, one `Investigation` per boring with its
continuation sheets folded in, gated on the sheet's own fitted ruler so a
depth off the paper is refused rather than accepted, and computing nothing —
an N value the log never printed stays `None`, because half the templates
print it and half do not and adding drives would invent a number; the **lab
reader**, a typed result per test kind, refused at construction when the
result does not match the kind, because `kind` is what every consumer
dispatches on and a mis-filed result would be invisible in the JSON; the
**narrative reader**, the owner's two standing query schemas answered field
for field and cited, `None` where the report is silent. The **reconciler**
links lab tests to the ground and RECORDS a disagreement rather than settling
it. The **writers** produce `report.record.json`, a summary page, a
WikiLLM-style library page with a SQLite index over many reports, and **real
DIGGS 2.6** through both gates, which the app's existing subsurface reader
now reads back. `graph.ingest_report` is the deterministic resumable loop,
`run_folder` drives it over a folder, and `subagent` puts the whole thing on
the app as one `CompiledSubAgent` plus one primary `report_ingest` tool,
**OFF by default** — `build_deep_agent(enable_report_ingest=False)` — because
none of it has been checked on the cluster yet.

**The scoring cell changed shape.** `score_on_cluster` now takes five stages
(`labels`, `logs`, `lab`, `narrative`, `vision_labels`) and **one truth
root**: `truth_dir` may be a folder holding `logs/`, `lab/` and `narrative/`,
named after the stages that read them. The fifth stage needs no hand truth of
its own — it scores against the label spreadsheet and the out-of-sample file,
exactly as the `labels` stage does. The hand truth is private and is uploaded
by hand before every run, and three Volume paths that must each be right is
three chances for one to be stale while the run starts anyway and scores
against it. The older
three-argument form still works and still wins where it is passed. The folder
to upload is staged, with its own README naming every Volume path:
`module_work/field_feedback/2026-09-16_report-ingest/raw/cluster_upload/`
(gitignored; about 2 MB).

**Dependencies.** Pin raised to **`planlens>=0.6`** (0.6.0, 2026-09-17: the
log grid, and the text-extraction fix that returns an overprinted line once —
which was fatal rather than untidy, since a doubled depth scale holds no
strictly rising run of three and the ruler was refused outright). planlens
0.6.0 declares exactly what 0.4.0 declared, so **no new third-party package
reaches the cluster**. `anthropic` stays optional and uninstalled.

Release gate **12,381 passed / 33 skipped / 0 failed** (3,598/28 + 8,070/5 +
713/0, run by the lead on the final tree with the vision stage in), three
chunks, each gated on pytest's exit code and never on a piped tail.

**First live checks:** (1) `%pip install "geotech-staff-engineer==5.20.5"`
with `planlens-0.6.0` in the same `Successfully installed` line and nothing
new beside it; (2) one four-stage `score_on_cluster(...)` cell with
`max_reports=2` and `truth_dir` pointing at the uploaded `truth/` root, and
`RESULTS.md` comes back.

**Parked, deliberately:** WP5, the calculation reader — the one work package
of the plan not built. The Azure Document Intelligence results the owner does
not yet have for four reports. The one hand label sheet that matches no
corpus report and stays unmatched rather than being force-fitted. The
development-engine checkpoints were never run for the logs, the lab or the
narrative: every number in the ledger for those three is either the grid
alone, the tables alone, or nothing yet, and the model numbers arrive with
the owner's cluster run. And the sub-agent's flag default stays `False` until
that run says the readers work on the tier that will do the work.

**Parked follow-up — the sub-agent's middleware enforces nothing.** deepagents
0.6.8 reads only name/description/runnable from a `CompiledSubAgent` spec, so
`ScratchFilesystemGuard` and `ModelCallBudgetMiddleware` declared on the
`report_ingest` sub-agent enforce nothing; the real ceilings are the
per-reader `Budgets` in Python and the graph exposes no filesystem tool to any
model. Follow-up: enforce the guard and the budget inside the compiled graph
(or wrap the runnable) so the sub-agent has the same protections as the calc
and references sub-agents.

**The vision-first experiment, landed before the tag.**
`report_ingest/vision_labels.py` sends each page as a PICTURE to GPT-4.1 on
the cheap tier (`funhouse-gpt-low`) with structured output and the same
eighteen-label vocabulary the review uses — one call per page, or one per
six-page contact sheet, optionally with the document's own outline prepended
— and `score_on_cluster` gains a FIFTH stage, `vision_labels`, that runs it
over the same reports as the `labels` stage and scores it against the SAME
hand labels with the SAME scorer. `RESULTS.md` then carries one table with
three columns where the runs exist: the rules alone, the rules the review
corrected, and the picture. A page the vision pass left unresolved is scored
as `other` — a non-answer is scored, not excused — and the blind set prints a
summary only. Nothing on the app's tool surface changed and no model was
called here: the numbers come from the owner's cluster run. Local twin:
`module_work/report_ingest_harness/measure_wp5_vision.py` (`--reuse`
re-scores the saved runs with no model, no key and no network).

### 5.19.0 RELEASED (2026-09-17, tag `v5.19.0`) — the `report_ingest` package, with planlens 0.5.0

**Both packages verified LIVE on PyPI 2026-09-17: planlens 0.5.0 (11:03 UTC)
and geotech-staff-engineer 5.19.0 (requires `planlens>=0.5`, no `anthropic`).
Cluster install NOT yet confirmed (`%pip install
"geotech-staff-engineer==5.19.0"`, install guide §1 says what a good log looks
like). Plan, work packages and everything parked:
`module_work/REPORT_INGEST_PLAN.md`. Private measurements, ID-only:
`module_work/field_feedback/2026-09-16_report-ingest/MEASUREMENTS.md`.**

**What shipped.** `report_ingest/` — WP0 through WP1b of the report-ingest
train — as a LIBRARY in the wheel. **Nothing is on the app's tool surface
yet**: no prompt, no tool, no chat behaviour changed, and planlens 0.5.0's
`document_roles` is not wired either (it goes in with the ingest sub-agent).
The package holds the two passes that need the whole document in view, which
is the thing per-page rules cannot do: `triage()`, one structured call over
the per-page ledger, the printed outline, the front matter and the first
contact sheet, returning a `DocumentProfile` whose `workflow` chooses the
readers that follow; and `review_labels()`, an agent loop with `read_page` /
`render_page` / `contact_sheet` / `outline`, told the rules' own weak spots
first. The review returns only CHANGES and Python applies them — a model
re-emitting 455 page-to-label pairs drops one silently and the scorecard then
scores the slip rather than the judgement — and each change is graded
`fixed` / `broke` / `still_wrong` / `disputed`.

**Why it is shaped this way.** The app runs in Funhouse against OpenAI models
through Prompter, so a number measured anywhere else measures a model that
will never do the work (owner's correction, 2026-09-17). `PrompterEngine` is
therefore the engine that counts and `ClaudeEngine` is a DEVELOPMENT engine
whose numbers are a checkpoint, never a result.
`cluster_scoring.score_on_cluster(...)` is the owner's notebook cell: reports
read **under their own file names** from the folder they are already in (IDs
resolved through the manifest's source-file column), Azure DI results read as
either `<ID>.json.gz` or the uncompressed `DI_data_<stem>.json`, a
`/Workspace` out_dir refused up front, resumable report by report, and one
`RESULTS.md` to bring home carrying IDs, labels, counts and rates only.

**Dependencies.** Pin raised to plain **`planlens>=0.5`** (0.5.0 on PyPI
2026-09-17: page roles, work items, the printed outline, the per-page ledger,
`text_reliable`, and the fix for a scanned appendix read as 69 duplicates).
planlens 0.5.0 declares exactly what 0.4.0 declared, so **no new third-party
package reaches the cluster**. `anthropic` stays optional and uninstalled
there; `openpyxl` was already a hard requirement of `python-ags4` (a direct
app dependency) and the label reader imports it lazily anyway. Two
`pyproject.toml` lines were the release-critical part: `report_ingest*` in
`[tool.setuptools.packages.find]` — without it the wheel builds clean and
simply has no package in it, failing on the cluster as an import error, so
the built wheel was listed to prove it — and `report_ingest` in pytest
`testpaths`, or its 117 tests never run in the gate.

Release gate **11,890 passed / 33 skipped / 0 failed**, run in three chunks
and gated on pytest's exit code, never on a piped tail.

**First live checks:** (1) `%pip install "geotech-staff-engineer==5.19.0"` on
the cluster, `planlens-0.5.0` in the same `Successfully installed` line and no
new package beyond it; (2) one `score_on_cluster(...)` cell with
`max_reports=2` against the reports already in Funhouse — it finds them by
their own names, writes `runs/`, `triage/` and `RESULTS.md` under
`/tmp/report_ingest_wp1b`, and `RESULTS.md` comes back with the WP1b numbers
on the model that will do the work.

**Parked, and deliberately so:** WP2–WP5 of the plan (boring logs →
investigations → DIGGS, lab tests, the narrative question set, the record and
its exports). The four disputed hand labels stay disputed — the spreadsheet is
never edited, because a hand label records what a person decided. Four
reports (R02, R31, R32, R38) would be worth Azure DI results the owner does
not yet have. One label sheet matches no corpus report and stays unmatched
rather than being force-fitted.

### 5.18.0 RELEASED (2026-09-16, tag `v5.18.0`) — interactive `plot_data`, planlens 0.4 quantities + fuzzy search

**Both packages verified LIVE on PyPI 2026-09-16: planlens 0.4.0 (18:14 UTC) and
geotech-staff-engineer 5.18.0 (19:45 UTC, requires `planlens>=0.4`). Cluster install
NOT yet confirmed (`%pip install "geotech-staff-engineer==5.18.0"`, install guide §1
says what a good log looks like). Parked from the same train, recorded in
`module_work/FUTURE_IDEAS.md`: Tier 2 scanned-page engines (need Funhouse to OK
vendored model files), drawing revision comparison, a toolkit MCP server (after
Funhouse names an in-enclave host). Private measurements + probe scripts:
`module_work/field_feedback/2026-09-09_nairobi-soe_v5.11.2/PLANLENS_SURVEY_MEASUREMENTS.md`.**

Two feature branches merged into master, no conflicts, **no new direct
dependency**. `feature/plotly-plot-data`: `plot_data` draws an interactive
Plotly twin from the same cleaned series as the PNG and writes it as a
`*.plotly.json` sidecar (the format the app has rendered with
`st.plotly_chart` since 5.4, which only `subsurface.plot_*` ever produced);
`output_capture` now copies a sidecar written to `/tmp` into the conversation
folder, and the PNG **card** is suppressed when a sidecar stands beside it, so
the reader sees one card — the interactive one. Only the card list is
filtered: the artifact list keeps the image, so the SharePoint mirror, the
sidebar downloads and `html_to_pdf` are unaffected. The deep prompt's figure
wording is now honest about what the chat actually does with a saved figure.
`feature/planlens-survey-wiring`: `find_quantities` as the EIGHTH document
tool, plus `fuzzy` / `min_score` on `search_document`, both feature-detected
from the installed planlens rather than trusted from the pin (details in the
"APP WIRING" entry below).

Pin raised to plain **`planlens>=0.4`** (0.4.0 on PyPI 2026-09-16). No extra:
0.4.0 folded `opencv-python-headless` and `rapidfuzz` into planlens' core and
kept `[raster]` / `[text]` as empty alias extras. `rapidfuzz` is therefore new
on the cluster but arrives through planlens, not as an app dependency; it
cleared the Nexus probe 2026-09-16 as 3.14.6. planlens 0.4.0 also carries, and
this app does NOT yet use: the scale Bluebeam/Acrobat store in the PDF, PDF
layer names and path fill in the IR, duplicate-scan detection, and an MCP
server over its own toolkit.

Release gate **11,773 passed / 33 skipped / 0 failed**, run in three chunks
(analysis modules incl. the slow FEM and slope suites, 3,793 / 28; the
references submodule, 5,793; then `funhouse_agent` + `foundry_test_harness` +
`webapp/tests`, 2,187 / 5) and gated on pytest's exit code, never on a piped
tail — the 5.17.0 lesson. The pin change broke exactly one test and it was
fixed:
`test_drawing_ir_adapter.py::TestExplodeFlagAgainstAnOlderPlanlens::test_pin_floor_is_covered_or_the_degrade_is_in_place`
read the floor with a regex that hard-coded `planlens[raster]>=`, so dropping
the extra made it fail its own "dependency line not found" assertion. The
extras bracket is now optional in that regex. Worth knowing: had the assert
not been there, the guard would have silently stopped comparing floors.
Cluster install of 5.18.0 **NOT yet confirmed** (install guide §11).

**First live checks:** (1) ask for a plot of lateral pressures — one card, an
interactive chart with zoom and hover, no second PNG card, no broken image
link, both files in the SharePoint folder, and the PDF still embeds the image;
(2) `find_quantities` on a geotechnical report — every stated value with its
unit, wording and page; (3) a `fuzzy=true` search on a drawing sheet whose
text was plotted as strokes, with the answer saying the match was approximate.
Edge noted during the build and deliberately NOT fixed (pre-existing since
5.4): a re-imported sidecar that collides with different content is renamed
`plot.plotly_1.json` by `_unique_dest`, which stops classifying as plotly and
stops suppressing the PNG — it degrades to a static card, no crash.

### 5.17.1 RELEASED (2026-09-15, tag `v5.17.1`) — one fix on 5.17.0

`webapp/core.import_reported_outputs` compared files with `filecmp.cmp`, which
caches by (size, mtime): a rebuilt deliverable of the same byte size written in
the same clock tick counted as unchanged, was not re-copied, and the
conversation kept the earlier version — the failure the import exists to
prevent. It now compares bytes (`_same_bytes`), and the test pins the case
(same size, same timestamp, different content). Caught by the webapp suite's
ordering minutes after 5.17.0 was tagged; a release command whose pytest output
went through `tail` had hidden the failing status. **Lesson for the next
release: gate on pytest's exit code, never on a piped tail.**

### 5.17.0 RELEASED (2026-09-15, tag `v5.17.0`) — the Nairobi SOE fix train

`module_work/field_feedback/2026-09-15_nairobi-soe-rerun_v5.15.0/FINDINGS.md`,
17 items from one real review session, fixed the same day. The FINDINGS "Train
record" has the commit per item, what to check on the next live run, and what
was noticed but not done. Ten commits `5c1058b`..`380e3f8` on top of 5.16.0;
release gate **11,743 passed / 33 skipped / 0 failed** (run in eight batches —
the whole suite in one process ran the machine out of memory). **No dependency
changes**, so the cluster install should read exactly like 5.16.0's
(§11 row: not yet confirmed).

**First live checks:** a report or plot written anywhere shows as a download
card and reaches the conversation's SharePoint folder; "save it to SharePoint"
lands in that folder; a chart question fetches its PDF from SharePoint
`GSE_app/primary_references` (`webapp/reference_fetch.py`); a rebuild
delegation carries its numbers; the reviewer cites only what it looked up. The
last three are prompt-level, so they need a live run to confirm.

**Numbers change** (analysis): cantilever embedment (about 3x longer),
braced support loads (surcharge and water now included) and braced embedment.

Three wrong things had reached the owner:
1. The rebuilt calc package lost every number. The delegation carried no data,
   and deepagents' scratch `read_file` reported the real earlier file "not found".
2. `soe.cantilever_excavation` is defective. It uses the first layer only, and
   its passive moment arm is H + D/3, so embedment comes out about a third of the
   correct value (`sheet_pile.cantilever_wall` is right).
   `soe.embedment.compute_embedment` is suspect for the same reason.
3. FHWA single-anchor results are partial, but nothing in the result says so.

App side: deliverables written to `/tmp` never become download cards or reach
SharePoint, which is the owner's main complaint. Also the SharePoint upload
path doubling and an `html_to_pdf` `file:///tmp` bug.

Branch `feature/subagent-feedback-reference-pdfs` (unreleased) carries:
`record_feedback` on every sub-agent and specialist, the reference-PDF
diagnostics row, and the WE-PILE-1 filename-case fix.

### 5.16.0 RELEASED (2026-09-14, tag `v5.16.0`) with planlens 0.3.0 (tag `v0.3.0`) — the document-review train

Owner word "push and publish" 2026-09-14. planlens 0.3.0 published FIRST
(tag-triggered trusted publishing), then the app with the pin raised to
`planlens[raster]>=0.3`. Both feature branches fast-forwarded into
main/master. Cluster install of 5.16.0 NOT yet confirmed (install guide §11);
no new third-party dependencies. First live checks: upload a real submittal
and ask a review question — does the agent open_document first, follow the
`! look:` cues into vision, and cite printed page numbers?

### DOCUMENT-REVIEW RESET (2026-09-13) — the train's record (released as 5.16.0 / planlens 0.3.0 above)

**Why.** Owner asked for a high-level review of planlens against the goal: a
flexible package an LLM harness uses to review ANY AEC document — geometry into
hard data, common drawing objects, text with location (Azure DI allowed).
Verdict: mostly drifted. Months of effort had gone into arrowhead/leader/
dimension tuning on ten no-text-layer Mecklenburg sheets, while the package had
no document layer at all. Measured on the real 260-page Nairobi submittal
(field_feedback 2026-09-09): every PDF text item said rotation 0 (215 of 245
wrong on a /Rotate 270 sheet); reviewer comments were read as drawing text; the
submittal's whole review record (Bluebeam callouts, replies, stamps, checker
calcs) and 320 hidden AutoCAD SHX-text annotations were invisible; on a
text-bearing sheet 24 of 33 default-confidence "dimensions" were bound to note
text. Owner approved the reset plan the same day ("don't move too quick, no
parallel agents"; one subagent at a time is fine for narrow tasks).
**Supersedes the "planlens = todo list only" rule for this work.**

**planlens (was branch `feature/document-layer`; now main, released 0.3.0, suite
875):**
- `c54d6c0` `planlens.document` — `open_document` / `Document`: page map
  (kinds with evidence), text lines with true direction + exact boxes, tables
  (`find_tables`), review markups (author, date, the point a callout/arrow aims
  at, /IRT links, reply-aimed-at-comment, stamp wording), hidden CAD text,
  search; `AzureLayout` reads an Azure DI prebuilt-layout result as an optional
  text source (never calls Azure). ONE frame: displayed-page points, top-left —
  the frame `render_region` uses. Design: `planlens/document/DESIGN.md`.
- `fc7c104` drawing-IR text fixes: real rotation, annotation text excluded,
  opt-in `include_cad_hidden_text`. `doc_claims_check.py` re-run: published
  corpus figures UNCHANGED.
- `a56dd93` `planlens.tools.ReviewToolkit` — five LLM tools, framework-neutral
  specs, every result valid JSON inside a per-call size limit, lossless paging.

**App (branch `feature/document-review-tools`):** `funhouse_agent/document_tools.py`
bridges the toolkit (attachments via contextvar, one process-wide toolkit so
handles survive turns); `make_vision_tools` adds `open_document`,
`document_page_map`, `read_document`, `search_document`, `document_markups` to
the PRIMARY agent's default set only when `planlens.tools` is importable (calc
sub-agent and figure reader unchanged); prompt now says open_document first for
reviews; attachment note mentions it. Real submittal through the app surface:
open 1,281 chars, all 7 drawing sheets with locations in 7 results each
<= 15,438 chars (cap 16,000), 40 markups in one result.

**Release order (done 2026-09-14):** planlens 0.3.0 first, then the app pin
`planlens[raster]>=0.3`; on an older planlens the app hides the tools rather
than failing.

**Owner gut check, 2026-09-14 (Fable): "we may be undervaluing the agent's
own vision."** Confirmed by probe — the scanned figure, the boring log and the
plan sheet of the real submittal gave the model no cue to look. Fixed in
planlens (`page_advice` + `! look:` lines / `pages_to_view` / `look` /
`pages_not_searchable_as_text` in every tool result, `render_page` /
`render_region` tools, image files as one-page documents) and in the app:
`document_tools.VISION_HINT` names `analyze_pdf_page` / `render_region` (the
app keeps its own vision tools; planlens' render tools are NOT on the app
surface), and the prompt states the policy — text first, then look whenever a
result carries a look cue or seems wrong for the page kind, and say what was
read vs seen. The `source` a caller opened with is echoed back so the model can
pass it to `analyze_pdf_page(attachment_key=...)`.

**Page map deepened (2026-09-14, owner's ask "what else could the map
say?"):** planlens `structure.py` reads each page's header/footer bands, the
page number PRINTED there, sheet refs, dividers, duplicates, and cuts the
document into SEGMENTS (constituent documents) — on the real submittal 29 of
them, incl. the 7-sheet drawing set as one, Appendices A-E, the boring logs as
one `form` run and four nested calc sheets with their own "Page 1 of 4". New
`form` page kind (ruled grid). Metrics on every row: words, density, ruling
lines, printed_page/of, sheet, scales, divider_title, duplicate_of, segment.
Contact sheets (`render_page_thumbnails`, like Bluebeam's page panel, red frame
= marked up; 3.2 s / 260 pages). App surface now has seven document tools
(`document_structure`, `render_page_thumbnails` added; thumbnails are viewed
with `analyze_image(attachment_key=<image_path>)`, which accepts a real path).

**Open, in order:** live cluster run on a real review question; port the
drawing tools (digitize/query/get_entities/snip/search_drawing_set) from
`drawing_ir_adapter.py` into `planlens.tools`; move the geotech cross-section
code (`planlens.pdf` vision prompts, `candidate_ground_surface`) back into the
app with a deprecation path; a multi-firm TEXT-BEARING evaluation set (the
Nairobi submittal first — private, never into planlens' public repo);
`planlens.ocr` as a text source; a visual-reading-order option for form pages
(boring logs read in drafting order today); give the calc sub-agent read-only
document tools. Arrowhead tuning is FROZEN.

**planlens survey train BUILT 2026-09-16 — planlens branch `feature/survey-step1`
(10 commits on 0.3.0, NOT pushed, NOT released; suite 875 → 1,109; corpus
figures identical; core deps unchanged; new OPTIONAL extras `[text]`=rapidfuzz and
`[mcp]`=mcp>=2,<3).** What landed: stored PDF scale (`/VP` viewports + measurement
markups → `Viewport`, `Markup.measure`, bridge to `measure.Quantity`; NO real
Bluebeam-calibrated page exists in our corpus yet — re-probe when one arrives);
PDF layer names + fill in the IR (`include_hidden_layers`; page map faster);
`Document.search(fuzzy=True)` default 80 (measured); `Document.quantities` +
`find_quantities` tool (regex; quantulum3 REJECTED on measurement);
image-hash duplicates on scans (dHash, distance 2); `open_document` size-budget
bug fixed; `planlens.mcp_server` / `planlens-mcp` (mcp 2.2 low-level Server, ten
tools generated from the plain specs, images attached, stdio + `--http`).
Private measurements: `module_work/field_feedback/2026-09-09_nairobi-soe_v5.11.2/
PLANLENS_SURVEY_MEASUREMENTS.md` (+ four probe scripts, untracked).
**Nexus probe DONE 2026-09-16 (owner ran `nexus_pip_install` on the cluster,
858 blocked versions loaded): both cleared.** Reproducible pins the helper
printed: `rapidfuzz==3.14.6`; `mcp==2.2.0 idna==3.19 pydantic==2.12.5
pyjwt==2.10.1 starlette==1.6.0` — the helper pins a transitive dep only when
some of its versions are blocked, so idna / pydantic / pyjwt / starlette each
have quarantined versions. BEFORE the app pins `[mcp]`, check those four
against the streamlit / langchain stack the cluster resolves (install guide
§7 procedure); `[text]` has no such coupling.
**NEXT (owner-gated):** review + push branch;
app wiring for `find_quantities` and the `fuzzy`/`min_score` search params in
`funhouse_agent/document_tools.py`; then planlens 0.4.0. Two follow-ups noted by
the agents: calc-printout quantity form `Pc (kip): 437.0` is not extracted;
the positive duplicate-scan case is exercised only synthetically.

**APP WIRING — DONE AND SHIPPED in 5.18.0** (branch
`feature/planlens-survey-wiring`, merged to master 2026-09-16; the pin bump
this note was waiting on is done too, see the 5.18.0 entry at the top).
`find_quantities` and `search_document`'s `fuzzy` / `min_score` reach the
primary agent, both FEATURE-DETECTED from the installed planlens' own specs —
`document_tools.has_tool()` / `search_supports_fuzzy()`, surface built by
`document_tools.document_tool_names()` — so a PyPI planlens 0.3 never sees
them and a `fuzzy=true` search there returns a JSON error instead of calling
the toolkit. Files: `funhouse_agent/document_tools.py`,
`funhouse_agent/deep/tools.py`, `funhouse_agent/deep/prompt.py` (two policy
sentences: stated numbers vs what the drawing measures; retry a missed exact
search with `fuzzy=true` and say the match was approximate), plus
`funhouse_agent/deep/tests/test_document_tools_offline.py` (11 → 16; the
older-planlens path is proven by monkeypatched specs, not by reinstalling) and
one exact-surface assertion in `test_deep_tools_offline.py`. Deep suite 316
passed / 1 skipped. The pin landed as plain **`planlens>=0.4`**, NOT the
`planlens[raster,text]>=0.4` this note originally predicted: 0.4.0 moved
`opencv-python-headless` and `rapidfuzz` into planlens' core and kept
`[raster]` / `[text]` as empty alias extras, so there is nothing to opt into
(rapidfuzz cleared the Nexus probe 2026-09-16 as 3.14.6).

**Earlier same day — APPROVED IN PART 2026-09-16 (owner word: "move through
the step-one changes, small add-ons, and then the MCP plug"; Opus 5 subagents,
one at a time, Fable reviews).** In progress on planlens branch
`feature/survey-step1`: (1) stored PDF scale from `/VP` viewports and measurement
markups, PDF layer names and path fill in the IR; (2) fuzzy search, quantities
from prose, duplicate-scan detection; (3) an MCP server over `ReviewToolkit`.
**TODO, parked:** the scanned-page engines ("local AI models for OCR / tables /
layout", survey Tier 2) — owner may take them next week IF the Funhouse team
confirms model files can be vendored through Nexus; and drawing revision
comparison (survey Part A4). Original parking note follows.

**PARKED (2026-09-14, owner pivoting to app work): planlens package survey.**
`module_work/PLANLENS_PACKAGE_SURVEY.md` — a web-verified survey of packages
that could extend planlens (three no-new-dependency wins first: calibrated page
scale from PDF viewport/measure dictionaries, PDF layer names, path fill; then
fuzzy search, quantity extraction from prose, scanned-table structure, an MCP
server). Two owner conditions before it is picked up: (1) **plain-English summary for the owner — DONE 2026-09-16**, at the top of
the survey file; (2) **it was NOT
checked against the organisation's banned-package list** — the Nexus inventory
404s from outside the enclave and no local copy exists; every new pin must first
pass the on-cluster `nexus_pip_install` probe in the survey's Part E. Also
recorded there: the cluster egress firewall makes any package that downloads
model weights at runtime unusable as published (models must be vendored, as
`planlens.ocr` already does).

### 5.15.0 RELEASED (2026-09-11, tag `v5.15.0`) — the app feedback train + ultra-review fixes

Release gate 11,651 passed / 33 skipped / 0 failed (2026-09-11); no dependency changes; published by the `v*` tag
workflow. **Cluster:** `%pip install "geotech-staff-engineer==5.15.0"` then
`%restart_python` — first cluster install not yet confirmed (install guide
§11). The owner's `/code-review ultra` on the branch found five things,
all applied in `21e0465`: the `render_figures` mode rode a private key
inside the package params, which the strictly-validating packages
(`slope_report_package`, `pavement_design_package`) would have rejected —
now a contextvar set around the call, with tests for exactly those two; a
duplicated `if` in the uploader; an O(n²) folder-name loop in the restore
block; `tomllib` in the docs-currency gate (3.11+ while pyproject says
>=3.10 — the gate would have vanished at collection on 3.10); and a
`GEOTECH_DEFAULT_OUTPUT_DIR` leak from AppTest boots that made a
funhouse_agent test order-dependent (`webapp/tests/conftest.py`).

Six owner items from real 5.13/5.14 sessions, all landed as separate commits
(`8bde466` disclaimer, `7681352` large-upload docs, `0dbec49` feedback,
`af76521` activity log, `eaced20` figure tools, `6150236` calc-prompt figure
rules, `7ff13cf` restore + filter). Ledger with diagnosis per item:
`module_work/field_feedback/2026-09-11_app-usage_v5.14.0/FINDINGS.md`. Owner
instructions that governed it: app only (planlens = todo list), sequential
work, no parallel agents.

What the next agent must know:

* **`activity.jsonl` is now the archive** (`webapp/activity_log.py`, always
  on, sub-agents attributed by `task` nesting). When a session went wrong,
  read that file from the SharePoint folder — not `trace.jsonl`, which is
  the on-screen summary and only exists when the toggle was on.
* **`FEEDBACK.md` / `feedback.jsonl`** travel with every conversation:
  sidebar notes (`user`) and the agent's own `record_feedback` calls
  (`agent`: capability gaps, tool errors). Triage reads them first.
* **The calc sub-agent has its own prompt now** (`_CALC_PREAMBLE` +
  `_CALC_FIGURES` in `deep/agent.py`). It was prefixed with the references
  LIBRARIAN framing for its entire life — never again put
  `CONSULTANT_FRAMING` on a calculating agent. Figures for bespoke reports
  come from `calc_package.render_figures` and `profile_figure.plot_data`.
* **Restore** = the mirror backwards (`sharepoint_store.restore_conversation`).
  The Graph token refresher lives in the launching notebook; after it
  clears, mirror AND restore fail until `stage_sharepoint(...)` is re-run.
* Catalog is at **7,965 / 8,000** chars — the next new method needs a brief
  trimmed somewhere.

Suites at the tip before the review: webapp 308; deep 284/1 skipped;
calc_package + profile_figure + funhouse_agent/tests 1,552/7 skipped; after
the review fixes, webapp + calc_package adapter 364. Full gate at release:
11,651 passed / 33 skipped / 0 failed (2026-09-11).

**NEXT TRAIN (owner ask 2026-09-11): Ensoft-style TABLES of calculation
data.** "Punch up the documentation we get from the python calcs — tables
saved of calculation data, similar to the way an Ensoft program would handle
it" (LPILE/APILE/GROUP-style per-depth p-y / load-transfer tables, per-slice
tables, per-layer capacity contributions, iteration histories). Shape:
`calc_package.export_tables` as a sibling of `render_figures`; each module's
`calc_steps` gains `get_tables(result, analysis)` (several already build
`TableData`); CSV + an HTML table block per report; the calc prompt's
deliverable skeleton gains a Tables section. Plan it as its own train.

### 5.14.0 RELEASED with planlens 0.2.0 (2026-09-11) — both trees COMMITTED

The publish order finally ran as designed: **planlens 0.2.0** (`f0a2b9d`, tag
`v0.2.0`, trusted-publisher workflow, live on PyPI) then **app 5.14.0**
restoring `planlens[raster]>=0.2`. Gate on the release tree 11,594 / 33
skipped / 0 failed; real PyPI resolve of the built wheel: 149 packages,
planlens 0.2.0, refs 1.4.0, no cytriangle chain anywhere.

**Cluster:** `%pip install "geotech-staff-engineer==5.14.0"` then
`%restart_python`. Carries, on top of 5.13.0: the Prompter METERING fix
(`_child_env` threads the kernel's meter config to the app subprocess — see
the metering section; verify with §8 of
`module_work/prompter_metering/DIAGNOSIS.md`), the OCR render cap (planlens)
and OCR-skipped-when-the-page-already-has-text (adapter, `ocr_force` escape
hatch), the SharePoint base-folder de-dup, and DXF block explosion back via
planlens 0.2.0. The PyNiteFEA numpy>=2.4 cascade is unchanged (owner's
call to keep it in core; site response verified numerically identical on
numba 0.67).

**Both working trees are now clean of release work.** The app tree still has
the owner's untracked scratch (the `*_copy.md` plans, docs PDFs, the .bat,
`bamako_agent_cell.py`, `suite_trial.json`) — deliberately never staged.
planlens has the one stray screenshot, never staged.

**Open, in priority order:**
1. Cluster verification of the metering fix (§8 snippet) and of DXF block
   explosion (`blocks_exploded` no longer appears; `n_block_entities` does).
2. The Prompter 401 — send the revised admin note; run the confirming test
   when convenient. Two app-side items still PLANNED: retry-once on
   `AuthenticationError`, `friendly_turn_error` auth case.
3. Owner decision: PyNiteFEA -> extra to undo the numpy cascade (TF /
   mosaicml / ydata-profiling broken in that kernel). CONFIRMED from published
   metadata that PyNiteFEA 3.1.0 requires numpy>=2.4.0 and is the SOLE driver,
   so this one line removes all four warnings. `docs/DATABRICKS_INSTALL.md` §4.
4. The `openai<3` pin comment now lies — httpx2/httpcore2 arrive via
   langsmith; fix the comment (and decide whether the pin still earns its
   place). This is also what makes the install slow (resolver backtracking):
   `docs/DATABRICKS_INSTALL.md` §5.
4b. Optional: bound planlens' raster extra to `opencv-python-headless>=4.8,<6`
   so the next OpenCV major is a decision rather than a discovery (§7).
5. planlens findings 5 (60-deg+ triangles) and 6 (concave dart), documented
   not fixed; a synthetic-terminator fixture set alongside the corpus is the
   standing lesson of rounds 4-6 (`round4_repro.py`, `round5_attack.py`).
5b. Figure follow-ups (2026-09-11 train): specialist agents (foundations /
   earth-retention / slope-fem / seismic, `dispatch.py:119-205`) scope out
   BOTH `calc_package` and `profile_figure` — they cannot produce a report
   or a figure; decide whether specialists carry the rendering modules.
   Plotly->PNG twin for `subsurface.plot_*` stays blocked (static export
   broken in the tenant). `fem2d` HAS calc_steps/get_figures and so works
   with `render_figures`; `soe` has no package at all.
6. Large uploads (owner feedback 2026-09-11 — decided: DOCUMENT the route,
   build nothing yet). The 25 MB cap is OURS (`ws_upload.MAX_FILE_MB`), not
   Streamlit's (200 MB default, untouched): one base64 message must cross
   the driver proxy in one socket lifetime. Documented in `webapp/README.md`
   "Large files" + the Attachments caption: put the file in SharePoint and
   ask the agent to fetch it (`sharepoint_download_file`). Backlog, in order:
   (a) sidebar "Attach from SharePoint" box — filename/path in, staged like
   an upload, no LLM turn, no cap; (b) chunked websocket upload (N component
   values reassembled server-side — each chunk is widget state that reruns,
   so it needs the one-shot-key dance per chunk; real risk of re-opening the
   22 MB loop). WON'T: browser->SharePoint direct upload — the delegated
   Graph token would have to live in page JS.

### 5.14.0 INSTALLED AND VERIFIED ON THE CLUSTER (2026-09-11) — install-log triage now has a home

`%pip install "geotech-staff-engineer==5.14.0"` **succeeds and the app runs**
(owner confirmed live). No 403, no quarantine, no cytriangle chain. planlens
0.2.0 and refs 1.4.0 both landed.

**READ `docs/DATABRICKS_INSTALL.md` BEFORE TRIAGING ANY INSTALL LOG.** Three
releases running, an agent re-derived the same answers from scratch — which
package drives the numpy cascade, whether the seven conflict warnings matter,
whether a new major version is safe. That file is now the standing answer:
what "good" looks like in three lines, every expected-and-benign log line with
what would change its verdict, the real red flags, and copy-paste snippets to
re-check a numpy driver or a major bump. Keep its §11 history table current.

Two findings from the 5.14.0 log that were NOT previously recorded:

**OpenCV arrived at a new MAJOR version, unbounded and unannounced.** The
cluster installed `opencv-python-headless 5.0.0.93` because planlens pins
`>=4.8` with no ceiling. **Verified safe**: the full planlens suite (792
passed / 1 skipped) was run against OpenCV 5.0.0, and the local gate machine
turns out to have been on cv2 5.0.0 already — so it was tested, but by
accident rather than by design. The cv2 surface we use is small and
long-stable, and `findContours` is already unpacked version-agnostically.
Optional hardening: bound the planlens raster extra to `>=4.8,<6` so the next
major version is a decision, not a discovery — needs a planlens point release,
so it is the owner's call. Watch table: `docs/DATABRICKS_INSTALL.md` §7.

**The gate runs the FLOOR of our agent-stack ranges; the cluster runs the
CEILING.** 5.14.0 shipped with the gate on deepagents **0.6.8** and the cluster
resolving **0.7.13** — the exact shape of the 5.10.2 outage. It held, and by
design rather than luck: `build_deep_agent` inspects the **compiled agent** for
`write_todos` and re-attaches `TodoListMiddleware` when missing, instead of
sniffing versions, so it self-corrects on deepagents releases that did not
exist when the guard was written. The red flag to watch for is the agent
failing *every* question with a recursion error right after an install.

**PyNiteFEA's numpy floor is now CONFIRMED, not inferred**: `PyNiteFEA 3.1.0`
declares `numpy>=2.4.0` in its published metadata. With numba capping
numpy<2.5 the resolver lands on exactly 2.4.6 every time, and nothing else in
the tree wants numpy above 2.1.3 — so moving PyNiteFEA to an extra removes
**all four** numpy/numba conflict lines, not merely some of them. The watch
item written into `pyproject.toml` at the 5.13.0 pin is hereby closed.

### FUNHOUSE METERING WAS SILENTLY DROPPING EVERY APP RECORD (2026-09-10)

The owner noticed Funhouse's token counter was not picking up app usage.
Diagnosed with executed proof; full report committed at
`module_work/prompter_metering/DIAGNOSIS.md`.

**Two plausible theories were DISPROVEN by execution before the real one was
found** — worth knowing, because both are the kind of thing a reader would
accept on reasoning alone. LangChain is NOT building its own unwrapped OpenAI
client (we use our own `PrompterChatModel`,
`funhouse_agent/deep/databricks_bridge.py:397`, which calls the SDK's wrapped
client directly), and the `is_logging_active` ContextVar guard is innocent.
`wrap_all_openai_methods` really does fire `meter_log` with real usage.

**The cause is one layer below the wrapper.** `meter_log` writes to a LOCAL
SQLite file whose path exists only in the notebook kernel's in-memory
`FunhouseConfig` singleton. The app is `Popen`'d as a fresh process
(`databricks_launcher.py:695`) and builds `PrompterAPI` with no `config=`, so
it sees DEFAULTS: `should_write_sqlite_for_user -> False`,
`get_budget_sqlite_store -> None`, every record dropped. The one diagnostic it
emits is INFO, below the default WARNING stdout level, so nothing ever said
so. Gap is TOTAL for app traffic (every turn, sub-agent, tool call, streamed
or not); the notebook kernel's own usage still meters, which is why the
counter read low rather than empty.

**Fixed ours-side, no SDK or admin change:** `_child_env` in
`webapp/databricks_launcher.py` now threads the kernel's meter config to the
child as `FUNHOUSE_BUDGET__*` / `FUNHOUSE_SESSION__USER_NAME` /
`CURRENT_USER_NAME` (`FunhouseConfig._load_env_vars` maps `FUNHOUSE_A__B` ->
`a.b`). Five tests in `webapp/tests/test_databricks_launcher.py`, including
that an explicit override wins, a broken kernel config never blocks a launch,
and that it is a no-op off-cluster where no SDK exists.

**Deliberately NOT done, and why.** The `user_name` column may read
`unknown_user` in a subprocess; the report's fix for that pokes a private SDK
attribute. Skipped — the user's identity is in the sqlite PATH
(`/Workspace/Users/<user>/.funhouse_meter`), which is what the admin
cross-user report keys on, so compliance is satisfied without touching
internals. Only worth revisiting if the column itself matters.

**Still theirs, not ours:** the pricing table matches models by exact string
and carries no `funhouse-*` chat aliases, so rows may meter tokens at cost
0.00 — numbers WRONG rather than missing. Raise it once tokens are landing.
Also the SDK's `collect_usage` streaming injection raises TypeError before the
wire (we already route around it).

**Verify on the cluster:** §8 of the diagnosis is a 30-second notebook snippet
contrasting kernel vs child, plus a post-fix check that the money column is
right too.

### 5.13.0 INSTALLED ON THE CLUSTER — what the numpy cascade actually did (2026-09-10)

`%pip install "geotech-staff-engineer==5.13.0"` **succeeds**. No cytriangle, no
waiver, planlens 0.1.0 as designed. App boots and answers.

**The PyNiteFEA numpy watch item fired, exactly as predicted.** numpy went
2.1.3 -> 2.4.6, and because numba 0.61 caps numpy<2.2 it cascaded: numba
0.61 -> 0.67, llvmlite 0.44 -> 0.49. Three Databricks runtime packages are now
unsatisfied IN THAT KERNEL: `tensorflow 2.19` , `mosaicml-streaming 0.12`,
`ydata-profiling 4.16` (all want numpy<2.2; ydata also numba<=0.61). Harmless
unless that kernel needs them. PyNiteFEA is the SOLE cause — nothing else in
core wants numpy above 2.1.3 — so moving it to an extra reverts the whole
cascade in one line if the owner ever wants the runtime left alone.

**numba 0.67 is VERIFIED SAFE, not assumed.** pystrata JITs its wave
propagation core (`propagation.py:67`, `@numba.jit(nopython=True)`), and the
gate had only ever run on numba 0.65.1. Live comparison on the cluster vs the
gate machine, same profile (30 m Vs=180 PI=15 over Vs=760 rock,
`synthetic_long`):

| | cluster (numba 0.67) | local (numba 0.65.1) |
|---|---|---|
| EQL surface PGA / amp / max strain | 0.1736 g / 1.157 / 0.2218% | identical to 4 dp |
| Linear surface PGA / amp | 0.5564 g / 3.709 | identical once the same 2% linear damping is used |

The linear case first looked like a mismatch (3.709 vs 3.988) — it was the
agent substituting `linear, damping=2%` for the requested Darendeli PI=15,
which it disclosed in its answer. Re-running locally with that substitution
reproduces 0.5564/3.709 exactly. Numerics are unchanged across the numba
upgrade.

**Still open from that install:** `httpx2`/`httpcore2` reached the cluster via
`langsmith 0.12.4` (a deepagents dependency), NOT via openai. The `openai<3`
pin was written specifically to keep that fork stack off the cluster and no
longer achieves it. Not currently breaking anything — the gate machine runs the
same stack, and the Prompter's NTLM path still uses the untouched httpx 0.28.1
— but the pin's rationale comment is now wrong about what protects what.

### FIELD FEEDBACK 2026-09-09 — Nairobi SoE: the 401 was the MODEL, not SharePoint

Drop triaged into `module_work/field_feedback/2026-09-09_nairobi-soe_v5.11.2/`
(FINDINGS.md committed, raw/ gitignored). The owner's read was that the agent
"did some search it didn't have credentials for". The evidence says otherwise
and the correction matters: `AuthenticationError` is the **openai** exception
class, every SharePoint tool catches and stringifies its own errors (so none
can kill a turn), and the file was fetched successfully at its true
23,006,108 bytes. The agent's own model call to the Funhouse proxy came back
as an **IIS 401 HTML page**, which `friendly_turn_error` then dumped raw into
the chat — and that raw page is what made a model-proxy failure look like a
SharePoint permissions failure.

Transient, not mis-permissioned: turn 2 succeeded seconds later on the same
credentials. Working hypothesis is NTLM's per-connection auth going stale
during the 23 MB download that preceded the model call.

Two planned fixes, neither landed (mid-release, and neither is verifiable off
the cluster): (1) retry the main model call once on `AuthenticationError` with
a fresh client — there is currently NO retry on that path, so one 401 ends the
turn; (2) give `friendly_turn_error` an auth case that names the failing
surface and strips an HTML body to its title. Backlog: `_render_pdf_page` has
no output-size cap (measured 2.13 MB base64 for a D-size sheet at 220 dpi —
fine here, but unbounded by design).

---

### THE 22 MB UPLOAD LOOP — root-caused and fixed (2026-09-09)

Live on 5.11.2: uploading a 22 MB file put the app into a permanent
websocket reconnect loop, sockets dying every ~14 s (owner's devtools trace:
`stream` 101 at 14.37 / 13.48 / 17.30 / 14.28 s, each followed by the stock
`health` + `host-config` refetch). Everything was stable before the upload --
this is NOT the old ping/TTL family, which 5.11.2 closed.

**Root cause: a component value is WIDGET STATE, and widget state is re-sent
by the browser on every rerun and every reconnect.** Confirmed in Streamlit's
own source, not inferred:

* frontend bundle: `createWidgetStatesMsg(){ let e=new be; return
  this.widgetStates.forEach(t=>e.widgets.push(t)), e }` -- the WHOLE map, no
  delta, no filtering.
* `runtime/app_session.py`: each rerun BackMsg is read as
  `widget_states=client_state.widget_states`.

So the ws uploader (`webapp/ws_upload.py`, the proxy-safe attach path the
Databricks launcher forces via `GEOTECH_UPLOAD_MODE=ws`) leaves the file's
base64 in its component value, and the browser re-uploads it forever. 22 MB
inflates to ~29 MB of base64; that cannot cross the driver proxy inside one
socket lifetime, so the socket dies mid-send, the client reconnects, re-sends
the same 29 MB, and dies again. Unbreakable by design -- the payload rides
the very reconnect that is supposed to recover it.

Note the asymmetry that hid this: the DOWNWARD direction is protected.
`runtime/forward_msg_cache.py` (`populate_hash_if_needed` /
`create_reference_msg` / `global.minCachedMessageSize`) re-sends a large
repeated ForwardMsg as a hash reference, which is why `st.download_button`
with big artifact bytes has never behaved this way. There is no equivalent
cache browser -> server.

**Fix (in the tree, unreleased):** retire the uploader's widget id the moment
its bytes are in hand. `ws_upload.next_upload_key(thread_id, epoch)` +
`ss.upload_epoch`, bumped in `app.py` whenever the ws uploader yields pairs
(including a duplicate filename, which is still 29 MB sitting in state). A
new widget id makes the old one inactive, and the client's
`WidgetStateManager.removeInactive` deletes its state -- so the file crosses
the wire exactly once. Guarded by 6 tests in
`webapp/tests/test_ws_upload.py::TestOneShotWidgetKey`, one of which asserts
the bump still exists in `app.py` (the component cannot be driven through
AppTest, and losing that one line brings the loop straight back).

**Immediate workaround on the running 5.11.2, no release needed:** reload the
app tab. Widget state lives in the page's JS memory, so a reload starts an
empty state map and the loop stops. Until the fix ships, keep ws-mode
attachments small -- the 25 MB cap is about a single crossing, and anything
over roughly 10 MB is re-sent on every interaction.

**Still worth doing later:** chunk the ws upload (send the file in N
messages, reassemble server-side) so the per-file cap stops being a function
of what survives one socket lifetime.

---

### 5.13.0 — the quarantined structural libraries are GONE (2026-09-08)

`%pip install "geotech-staff-engineer==5.12.0"` died on the cluster with a
Nexus **403 "Requested item is quarantined"** on `cytriangle`. That package is
a transitive dep of BOTH `sectionproperties` and `concreteproperties`
(verified from installed metadata -- not concreteproperties-only, which the
pip traceback makes it look like), and both were core deps, so one blocked
wheel failed the WHOLE install: no agent stack, no planlens, nothing.

**Owner's call:** no security waiver is available for a project this size, so
the two libraries were REMOVED and their capability re-implemented natively --
the groundhog precedent from 5.11.2. A 5.12.1 hotfix that merely moved them to
an opt-in extra was built and verified first, but the owner chose to wait for
the rebuild rather than release it (branch `hotfix/5.12.1-structural-extra`
survives as a fallback; it is cut from the v5.12.0 tag so it can still ship
without planlens 0.2.0).

**What was built (all native, numpy/scipy only):**

* `section_props_agent/polygon_props.py` -- exact Green's-theorem integration
  of area, centroid, Ixx/Iyy/Ixy, principal axes, elastic moduli; plastic
  moduli by half-plane clipping + area-halving bisection; native
  self-intersection check replacing shapely.
* `section_props_agent/torsion.py` -- published closed forms (exact St.
  Venant series for rectangles, polar for circular, Bredt for closed boxes,
  El Darwish & Johnston/AISC for I-sections) plus a finite-difference Prandtl
  solve, Richardson-extrapolated, for arbitrary outlines.
* `concrete_props_agent/rc_native.py` -- transformed and cracked sections,
  cracking moment, ultimate capacity by ACI strain compatibility, and an
  N-M interaction diagram sampled uniformly in axial force.

**Verification.** The libraries' outputs were pinned as oracles BEFORE
deletion (`module_work/structural_native/pin_oracles.py` -> `oracles.json`,
23 cases); no library code was copied. Measured against them: every RC scalar
within **0.037%**; all section geometry within **0.5%**; polygon torsion
within **0.5%** (and within 0.013% of the exact series where one exists).
Two deliberate differences are declared in the oracle tests with reasons --
the warping constant is no longer reported for solid rectangles, closed boxes
or arbitrary polygons (no defensible closed form; design practice neglects
it), and the interaction curve differs above 70% of the squash load, which is
above ACI's own 0.80*P0 cap. Structural suites went 58 -> **150 tests** and
got ~17x faster (no meshing).

**Packaging.** `sectionproperties` and `concreteproperties` are gone from
`pyproject.toml`; `[structural]` is an empty alias like every other retired
extra name. **PyNiteFEA stays in core** (owner's call -- it is on no block
list and installs through the proxy fine). Watch one thing on the first
cluster install: PyNite 3.x requires `numpy>=2.4` and the runtime boots with
2.1.3, so pip will upgrade numpy under the kernel -- the failure mode that
SIGTERMed the REPL via Pygments in 5.10.0. If that bites, moving PyNiteFEA to
an extra is the one-line fix.

**Still open:** version is bumped to 5.13.0 in the working tree but NOT
released, and the tree still carries the uncommitted planlens Phase-3.2 work
plus the `planlens[raster]>=0.2` pin -- so the publish order (planlens 0.2.0
first, then the app) still applies. Ask the DT Nexus admins to release
`cytriangle` from quarantine anyway; it is a Cython wrapper around Shewchuk's
Triangle and the flag looks like the usual binary-wheel false positive.

---

### The staging trap (read first)

Six planlens paths MUST enter the same commit or the SOURCE tree breaks --
the tracked-and-modified shims `planlens/ir/tests/{leader,construct}_fixtures.py`
already delegate to them, and the guard that would catch the mistake is
itself in the untracked set:

    planlens/testing/{__init__,leader_fixtures,construct_fixtures}.py
    planlens/tests/{test_packaging,test_readme_claims}.py
    planlens/ir/tests/test_text_bearing_scenes.py

Measured consequence of missing them: planlens 412 -> 369 collected + 4
collection errors; the app repo 58 -> **0 collected**. Also untracked and
wanted: `planlens/ir/{measure,spatial}.py` + their tests (new work, below),
and app-side `module_work/code_review/2026-09-06_planlens_phase32/FINDINGS.md`
+ `module_work/drawing_ground_truth/doc_claims_check.py`.
Do NOT stage the stray `Screenshot 2026-09-07 204229.png` in planlens.

### Publish order: DECOUPLED (owner's call, 2026-09-09)

The app no longer waits on planlens. The pin is back to
`planlens[raster]>=0.1` and **app 5.13.0 publishes alone**, because its two
fixes -- the cytriangle install failure and the 22 MB websocket loop -- are
urgent, unrelated to planlens, and the cluster is broken in both ways today.
planlens 0.2.0 is HELD pending the round-4 repair (see the section above).

Verified before relaxing the pin, so this is not a hope: every planlens symbol
the app imports at RUNTIME exists in the published 0.1.0 wheel (checked by
parsing the wheel, not by installing it), and `planlens.testing` is imported
only by tests, which the wheel does not ship. The single live cost is that DXF
block line-work is absent, which `_dxf_supports_explode`
(`adapters/drawing_ir_adapter.py:118`) already handles by falling back to an
unexploded ingest.

**The follow-up release must restore `planlens[raster]>=0.2`** once 0.2.0
publishes. The comment at that pin in `pyproject.toml` says so too.

The `geotech-references` pointer still moves to `d8ff52e` (1.4.0, already on
PyPI) in the same commit.

### Round 6 — independently verified SHIP (2026-09-11 morning); release sequence in progress

Round 6 (same Fable builder, resumed after a session-limit reset on a tree
verified byte-identical to its freeze) answered every round-5 defect, and a
fresh pass by the same independent verifier measured it: **SHIP.** Reports:
`ROUND6_BUILDER_REPORT.md`, `ROUND6_VERIFICATION.md` beside the ledger.

- **D1 closed the way the tip had it right.** A fill cluster seats at its
  NEAREST MEMBER (was centroid); between the two sound tiers the better-seated
  candidate wins outright, tie -> directional; `_SEAT_DOMINANCE` REMOVED (a
  test asserts its absence). Verifier: cluster keeps its end in all 42
  shift x seat cells; the leader owning the foreign chevron survives with the
  CORRECT reading at every shift. Rationale: at shift 2.0 the cluster is
  0.5 pt off and the chevron 0.94, so any ratio > 1.9 fails and a scale
  threshold would need a 0.01 pt margin — nearer-wins has no constant.
- **D3:** oriented now also requires arrow scale (`_MIN_ARROW_SIZE_SCALE`
  0.5x, one home with the open-3 gate) AND a taper toward the end in the
  shape's own PCA frame. Every rectangle is blunt however turned (closes the
  scale-bar hole); every former "oriented" corpus member was a sub-scale
  fragment. **Oriented corpus population 0/0/0, so the README blunt figure is
  BACK at 19/17/0 = round 4 exactly; round 5's 18/16/0 is superseded.**
  `doc_claims_check.py` now prints the ORIENTED row too; both pinned.
- **D2 became a large win:** cProfile put 64% of the round-5 cost in the
  witness search building ~1,050 rounded dicts per tip to keep 50;
  `_ending_near_from_grid(limit=50)` is proven output-inert (vcheck
  byte-identical) and takes 3001 from 3.6 s to 0.68 s. Verifier's own
  timing vs round 4: -54% / -35% / -10% on the three totals (round 5 was
  +34/+15/+16). Exact prune kept, now bound against the best SOUND seat with
  the cluster's REAL radius (a zero radius had been pruning real clusters at
  shift 1-2 — caught by the verifier's D1 table).
- **D4:** tie-break `(seat, -alignment, d)`, order-independent, pinned.
- **Corpus:** every acceptance number, all 41 residuals, the 322-row called
  set AND the 1662-row leader set byte-identical to 1f6551c. Suites 798 / 61
  / 11. Observational churn vs round 4: +57 at <=0.25 and 15 rows at 0.3, all
  0.45 junk >= 30 pt from truth, per-row accounted in the builder report.
- **R1, the one accepted residual (documented, pinned, not a code change):**
  a stipple splash centred on a shaft end takes the end from a drawn arrow at
  ANY non-zero crookedness (the README had said "> ~9.6 deg"; measured 0.5 deg
  already loses, publishing 94.3 for 100 at 0.947). Corpus-inert, identical to
  the tip, a regression vs round 4 only for 0.3-9.6 deg arrows. Shipped as-is
  because a splash on the end IS a real stipple arrowhead's anatomy and the
  scene needs two terminators at one end; README sentence corrected and the
  0.5/1.0/2.4/5.0 deg rows pinned as the accepted residual
  (`TestTheAcceptedResidualAtItsRealThreshold`). Two low observations added to
  the README sharp edges (size floor is a sheet statistic; mirror-image ties
  resolve by grid order).
- Findings 5 and 6 remain documented, not fixed.

Final tree: `queries.py` md5 `e2b2b3de5820fe12f8400348108e098b` (the verdict's
file, unchanged through the doc pass); README `83685e33…`,
`test_end_ownership.py` `c40e863c…`. Release sequence from here: lead's own
planlens full suite + app drawing tests (61 green) -> commit planlens (six
staging-trap paths in ONE commit, screenshot excluded) -> tag v0.2.0 -> push
(trusted-publisher workflow) -> app: restore `planlens[raster]>=0.2`, version
bump, final gate, commit, tag, push.

### Round 5 (the repair) was independently verified: DO NOT SHIP as-is — round 6 in flight (2026-09-10 evening)

Builder (Fable, fresh) implemented Change A + Change B; a fresh Fable verifier
then measured the frozen tree (`queries.py` md5 `c0c67b34…`). Both reports are
committed beside the ledger: `ROUND5_BUILDER_REPORT.md`,
`ROUND5_VERIFICATION.md`, `round5_attack.py` (three-tree probes) in
`module_work/code_review/2026-09-06_planlens_phase32/`.

**What verified sound (RUN by the verifier):** every headline number
reproduces; findings 1, 3, 4 fixed on the fixtures; the restored pre-prune is
proven exact (triangle inequality) and measured inert on all ten sheets; 41
residuals and the 322-row called set byte-identical; suites 724 / 61 green.

**D1 — BLOCKER.** Finding 2 (the cluster anatomy, the highest-risk line) is
fixed only inside a 0.05-0.4 pt window. The 10x seat-dominance ratio needs
`seat_cluster*10 < seat_chevron`; real clusters seat 0.5-2 pt and an attached
foreign chevron seats up to ~4.6 pt, so at every realistic offset the foreign
chevron still takes the end and `exclude_dimensions` still deletes the owning
leader. Tip 1f6551c gets every shift right; round 4 none; the repair moved the
boundary from "never" to "sub-half-point". Corpus-inert — the corpus has no
such case, the same non-evidence three rounds already recorded. The builder
had flagged this exact judgement call (ratio vs absolute) as unsure; the
measurement settled it. Fix shape (verifier + lead agree): a cluster's seat
should be its REACH on the ray (`_reach_on_ray` exists), i.e. 0.0 whenever the
end lies inside the splash's axial extent, so real clusters always win and
round 4's splash-5-pt-off case stays capped.

**D3 — policy.** The "oriented" rank's real corpus population is 6/6 sub-2-pt
SHX glyph fragments, 0/6 terminators; with other rungs absent it CALLS a
[true arrow + 0.8 pt oblong fragment] at 1.0 and a graphic SCALE BAR with 2:1
end blocks as a dimension at 0.944 (tip: no proposal). Decision: oriented must
be commensurate with the arrowhead scale; scale-bar anatomy pinned as not a
dimension; fallback is clamp-only with all tipless capped.

**D2 cost** (+34% @0.0, +15-16% at the default, driven by 3001; 4.3 s absolute
max — the builder never timed the shipped tree). **D4** entity-order tie
(2.5 pt swing). **D5** docs: "+60 from the prune" wrong (60 removed / 120
added by seat ordering un-colliding the DISTINCT rule), "6 rows" is 8, the
"six oriented" README figure has no command.

Round 6 routed to the builder with all of the above and the verifier's drop-in
pytest; verifier re-runs after. Ledger correction applied: the two stale "316"
test counts now read 350.

### The remediation train: the caveat is CLOSED, and it found two regressions

The round-4 adversarial pass was run (2026-09-09, independent verifier, no
build context). Full report + runnable repro fixtures for all three trees:
`module_work/code_review/2026-09-06_planlens_phase32/ROUND4_VERIFICATION.md`
and `round4_repro.py` alongside it.

**Verdict: DO NOT SHIP as 0.2.0** until findings 1 and 2 are fixed.

What held up: **every measured number in the ledger reproduces** (632 / 58 /
25/25 / 16/16 / 23/25 / FPs 2·10·6·19·23·20·38 / precision 5·10·12·10 / blunt
19·17·0 / layer-0 557·158·1696), no README figure disagrees with
`doc_claims_check.py`, and — the strongest evidence for the tree, which the
ledger never claimed — **all 41 matched corpus defpoint residuals are identical
to `1f6551c` to three decimals.** On the corpus the remediation is provably
inert.

What broke, in six-line fixtures, on terminator styles the corpus does not
contain:

1. **Box-like (diamond / trapezoid) terminators**: `_arrow_geometry` computes
   the apex correctly and the blunt branch then discards it for a centroid
   projection — length 100.0 -> **94.0**, confidence 0.952 -> **0.45**, never
   called. Clean on v0.1.0 AND on the tip: a regression from the uncommitted
   work.
2. **A fill-cluster arrowhead loses its end to a farther foreign chevron** (the
   cluster is 0.04 pt away, the chevron 3.24 pt) because distance is only
   consulted *within* a tier. No cap, no flag, confidence stays 0.944 — and the
   foreign arrowhead lands in `arrowhead_ids`, so `exclude_dimensions=True`
   deletes whichever leader owns it. That is the arbitration steal round 4's
   headline claims to have closed, reached by a route its fixture does not
   cover. **On a corpus whose real arrowheads ARE clusters, this is the
   highest-risk line in round 4.** Also clean on 0.1.0 and the tip.

Findings 3-5 are regressions vs the published 0.1.0 but pre-existing at the
tip; 6 is pre-existing everywhere (a concave dart is deleted at every
threshold — document it, do not fix it); 7-10 are docs/staging, minutes each.
Finding 7: `"632 passed (from 316)"` — the tip actually collects **350**.

**Repair is TWO changes, not four (~1.5 days):**
- **Change A — finding 1.** Independent and corpus-inert (0 blunt proposals at
  0.5). One-line coordinate substitution at `queries.py:1275` using the
  primitive the cluster leg already has (`_reach_on_ray`, which returns
  `(100,0)` exactly on the diamond), plus a policy call on the unconditional
  0.45 blunt cap. Half a day. Land it first.
- **Change B — findings 2, 3 and 4 together.** Three halves of one question:
  who wins the end (2), what a win costs (4), whether the winner's coordinate
  is publishable (3). They pull against each other — repairing 3 makes 4 worse;
  repairing 4 re-opens the steal unless 2's distance-dominance rule lands with
  it. Four sites: `_end_tier` `:1360-1366`, the per-end comparison
  `:2111-2130`, the continuous apex publication `:2227`, the cap ladder
  `:2264-2272`. A day, with fixtures pinning BOTH directions plus a corpus and
  residual re-run.

**The structural lesson, which outlives these fixes:** the corpus cannot
falsify a change to a terminator style it does not contain, and its 18 pt match
tolerance would hide a 9 pt error. Five of the six code findings live in
exactly that blind spot. Any future gate on this code needs synthetic
terminator fixtures alongside the corpus — `round4_repro.py` is the start of
that set.

### Published numbers now have a command

Two rounds running, prose was edited with figures no run reproduced. The fix:
`module_work/drawing_ground_truth/doc_claims_check.py` regenerates every
corpus figure the planlens docs publish, and ten guards in
`planlens/tests/test_readme_claims.py` pin them. **If a number disagrees with
that script, the DOCUMENT is wrong.** Current measured values: blunt
terminators fire on 19 proposals @0.0, 17 @0.3, **0 @0.5** (only sheets 11.01
and 21.01); layer-0 inheritance re-homes 557/158/1696 entities on
10.17a/11.01/5003 -- and note the `n_layers` METADATA field counts something
different and does NOT follow the inheritance.

### NEW DIRECTION (owner correction, approved plan)

Recognition of CAD constructs was never the goal -- the goal is engineers
reviewing design and construction documents, and an uploaded artifact is
usually a REPORT (thirty pages of narrative, then tables, then figures), not
a sheet. The user must not have to say whether they want a language review or
a visual review.

**Plan of record then: `C:/Users/socon/.claude/plans/delightful-swinging-sky.md`.
That file is gone (not in `~/.claude/plans/`, never committed).** It was
replaced by the 2026-09-13 document-review reset ("DOCUMENT-REVIEW RESET",
above), then `module_work/REVIEW_ARCHITECTURE.md`; open work is in
`BACKLOG.md`, and this slice is DR-12.
Driving example: *"what is the average spacing of the borings in this plan?"*
over a Langan Subsurface Investigation Plan. Surveyed verdict: today that
fails as a plausible wrong number rather than a refusal.

Owner decisions taken: source documents are **PDF nearly always**; build the
**vertical slice** (one question end to end); output shape is "whatever the
reviewer asked for", so findings are data, not a fixed deliverable.

Started and green (new, untracked): `planlens/ir/measure.py` (the `Quantity`
envelope -- units mandatory, confidence composes by `min` not product, and a
page-point value REFUSES to become feet without a resolved scale) and
`planlens/ir/spatial.py` (point-pattern maths, numpy only, no scipy). 37
tests. Notable measured result: on a perfectly regular 3x3 grid the three
spacing conventions differ by **2.45x**, which is why the API deliberately
has no key named `average_spacing` at any depth.

**Blocked on the owner:** the Langan sheet is in NEITHER repo. Real ground
truth needs it plus 2-4 more subsurface investigation plans.

Next in the plan's own order: the two ingest representation fixes (PDF fill
is absent from the IR entirely, and multi-subpath drawings are fused into one
polyline) -- both are load-bearing for symbol work and both move every
published corpus number, so they are a re-baselining milestone, not a patch.

### Still open from before

Owner's live 5.12.0 cluster shakedown; SoM A/B live run (recipe in
`som_ab_check.py` docstring); planlens submodule wiring in the app repo;
TinyApps onboarding; cluster OCR needs the planlens README headless recipe;
master CI Tests should have self-healed now planlens is on PyPI (VERIFY).

## 0a-prev-c. PICKUP LIST (2026-09-05) [HISTORICAL]

**5.12.0 RELEASED TO PYPI 2026-09-05, VERIFIED LIVE** (with planlens
0.1.0 + geotech-references 1.4.0, both first published the same hour;
full gate 11,468/0 beforehand). Cluster install:
`%pip install "geotech-staff-engineer==5.12.0"` — the launcher
baseUrlPath fix ships, so NO notebook patch cell. Release gotcha for
the future: PyPI enforces a 512-char cap on the pyproject description
SERVER-SIDE only (twine check passes longer) — the 5.12.0 publish
400'd twice on a 513-char description. OPEN after release: owner's
live 5.12.0 cluster shakedown; SoM A/B live run (Funhouse recipe in
som_ab_check.py docstring); drawing Phase 3.2 list (design memo
banner); planlens submodule wiring; master CI Tests should self-heal
now that planlens is on PyPI (was red only for that reason — VERIFY);
TinyApps onboarding; cluster OCR needs the planlens README headless
recipe.

## 0a-prev-b. PICKUP LIST (2026-09-04 EVENING) [HISTORICAL]

Current release: **5.11.2** (PyPI; refs 1.3.3). Master carries the big
UNRELEASED 5.12.0-candidate train: security hardening, geophysics
excision, structural stack, drawing Phases 1+2 (verified ship-as-is),
ground-truth harvest, the **planlens split**, and the launcher
baseUrlPath fix. CLAUDE.md "Post-5.11.2" + "PLANLENS SPLIT" = summary.

DONE TODAY (2026-09-04): drawing Phase 2 COMPLETE + independently
verified; Mecklenburg ground truth harvested (10 DWG+PDF pairs + truth
JSON; SHX no-text-layer finding); 5.11.2 live shakedown PASSED after
the "Not Found" root cause (launcher baseUrlPath vs prefix-stripping
proxy — fixed b6683a6; live via notebook template patch) — heartbeat
holds, uploads work; **planlens split executed** (owner-named).

1. **planlens finish-out:** owner creates
   github.com/soconnell345-geotech/planlens → push the local repo
   (C:/Users/socon/OneDrive/dev/planlens, branch main) → `git submodule
   add https://github.com/soconnell345-geotech/planlens.git planlens`
   in the app repo (refs pattern). **5.12.0 CANNOT ship until planlens
   0.1.0 is on PyPI** (app now depends on planlens[raster]>=0.1;
   publish owner-gated). Dev environments: `pip install -e ../planlens`
   (already done in .venv).
2. **Drawing Phase 3 (owner GO 2026-09-04, build in planlens):**
   fill-cluster arrowhead detection (micro-dot fills — THE real-sheet
   leader/dimension recall gap) + B7 raster/OCR leg (REQUIRED even for
   vector PDFs: SHX sheets have no text layer); then B3 set-of-marks
   A/B + B1 DXF-native LEADER/DIMENSION ingest + set-level IR caching.
   Ground truth stays app-side: module_work/drawing_ground_truth/.
3. **Next release (5.12.0)** — owner decision 2026-09-04: HOLD until
   Phase 3 is built and tested. Order: planlens 0.1.0 to PyPI → refs
   release (1.4.0?) + pin bump → 5.12.0. Until then the cluster needs
   the notebook baseUrlPath patch cell each fresh kernel.
4. **TinyApps onboarding** (tinyapps/TINYAPPS.md): MOU → GitHub license
   → wrapper repo; planlens is the pitch's banner app (OBO branding at
   app-setup time, NOT in the package).
5. **FEMA P-2192**: fema.gov 502s persist — owner browser fallback;
   it's a worked-examples corpus play (discuss usage before building).
6. **FIELD FEEDBACK — ALL SIX FIXED 2026-09-05 (owner ordered
   execution; commits + details in the FINDINGS.md disposition table:
   module_work/field_feedback/2026-09-04_praia-downdrag_v5.11.2/).
   Ships with 5.12.0. Original items for reference:**
   (a) false "sources unavailable" in calc reports — calc sub-agent
       isolation likely drops working-folder visibility (TRUST issue);
   (b) subsurface `profile_figure` tool (matplotlib schematic: layers +
       GWT + pile/wall overlay → PNG) + prompt default "calc packages
       include a profile viz" + html_to_pdf fails LOUDLY on
       un-embeddable figures (model shipped "[image]" placeholders and
       a color-table fake figure);
   (c) CGPR #56 downdrag method family — downdrag.py has ONLY Fellenius
       neutral plane; implement the report's other methods w/ citations
       (CHECK CGPR distribution terms before any digitization);
   (d) conversations-tab rename/delete icons lost (streamlit 1.63
       glyph regression — repro, fix, consider version cap);
   (e) attachment awareness — model disclaims "can't attach in chat"
       while writing into the conversation files dir (which IS the
       attach surface): prompt + tool-result note;
   (f) SharePoint mirror folders named by custom conversation name +
       date stamp instead of thread-id hex.
7. Old backlog unchanged: axial_pile cohesive_phi trap; SCDOT
   onboarding; 108-Q eval rerun; auto-continue taxonomy; λ-method;
   Highter-Anders; 6.0 restructure (dedicated session); foundry/
   cleanup; Monday drift canary; pandas-2.x DT-sweep watch; deferred
   structural: pelicun + USGS design-maps route (FUTURE_IDEAS).

## 0a-prev. PICKUP LIST (2026-09-03) [HISTORICAL]

Current release: **5.11.2** (websocket saga CLOSED + upload workarounds +
detached turns + groundhog removal; refs 1.3.3). **TINYAPPS PILOT AWARDED
2026-09-03** — the strategic hosting path is live; plan + wrapper-repo
skeleton + office-hours question list = `tinyapps/TINYAPPS.md`. The
Funhouse/Databricks app stays as fast tester + backup (owner directive).

**The websocket saga, final ledger (do not relitigate):** three stacked
causes, all fixed — (1) the proxy swallows ws ping/pong control frames;
(2) streamlit's server pings 30 s / hangs up 60 s (code 1011) when pongs
never return; (3) the launcher's `bootstrap.run(flag_options)` NEVER
APPLIED flags — `load_config_options()` was the missing call (76e21ff), so
the 5.11.1 ping fix was inert and EVERY launcher flag had been ignored
since birth. Lab probe rig: aiohttp `ws_connect(autoping=False)` against
`/_stcore/stream` (session tmp scripts; recipe in CLAUDE.md). Plus
`webapp/turn_jobs.py`: turns now run in a socket-independent worker and
reattach across reconnects — belt and braces. "Hard proxy TTL" was an
interim theory, RETIRED.

1. **TinyApps onboarding (owner + agents):** MOU → GitHub Enterprise
   license → init wrapper repo (thin: app.py/packages.txt/run.sh from
   tinyapps/wrapper_repo/, package via Nexus). Blockers to resolve at
   office hours: key-auth Prompter client, SharePoint-from-App-Service
   auth, boundary/CAB status. `webapp/tinyapps_entry.py` stub exists;
   engine wiring lands after answer #1.
2. **5.11.2 live shakedown on Funhouse:** the 60 s flap should now be
   GONE (probe: PING should never arrive). Then: ws uploader, diagnostics
   upload-probe verdict, model picker, budget line, email_file,
   SharePoint link, streaming opt-in.
3. **Old backlog (unchanged):** axial_pile beta cohesive_phi trap; SCDOT
   examples onboarding; GPT-5.x 108-Q eval rerun; auto-continue taxonomy
   port; Document Intelligence bake-off; λ-method; Highter-Anders;
   6.0 restructure (dedicated session); foundry/ cleanup; Monday drift
   canary reports; pandas-2.x DT-sweep watch.

Standing rules: releases/tags ONLY on owner word (v* tag auto-publishes);
additive default-preserving changes; validate vs published values; never
unpinned installs on the cluster; owner is a practicing engineer, not a
developer — actionable results, no lectures; use Fable subagents freely
for build work (owner directive, Max plan).

## 0a-prev. PICKUP LIST (2026-09-01) [HISTORICAL]

Current release: **5.11.0** (PyPI, verified live; refs 1.3.3). Master ==
released (`ae126e8`). Gate green (1,778 on the release chunk; full gate needs
detached >10-min chunks). CLAUDE.md release sections + auto-memory carry the
narrative; `module_work/FUTURE_IDEAS.md` holds the **Funhouse SDK survey
ledger** (9 ranked findings — streaming landmines, Tiny Apps, auto-continue
taxonomy, Document Intelligence, environment facts) and the forward specs.

**OWNER PIVOT (2026-07-30): Foundry is RETIRED** (their Palantir team shut it
down) — **Databricks/Funhouse is THE deployment target.** Ignore the
Foundry-401 items in the old §0a; Foundry code ships dormant/parked.

1. **5.11.0 live shakedown (owner's next cluster session):** pinned install
   `%pip install "geotech-staff-engineer==5.11.0"` (no extra pins — openai<3
   is in core; mirror lags PyPI 1–2 days). Verify: heartbeat survives
   follow-up questions (the "Connecting" flap), SharePoint sidebar link now
   clickable, model picker (funhouse-gpt-medium), budget line, email_file,
   opt-in streaming (`GEOTECH_PROMPTER_STREAMING=1`), and the localhost PUT
   probe for the still-open browser-upload 403 (SharePoint-fetch = fallback).
2. **Funhouse office hours follow-ups (owner):** Tiny Apps intake status
   (form submitted >1 month ago; Tiny Apps = sanctioned durable tier,
   SUPPORTS Streamlit), the vanished `Tiny App Development/` example folder,
   streaming/metering blessing, 30-min cluster idle timeout.
3. **Monday drift canary:** `geotech-drift-canary` cloud routine (Mondays
   10:00 UTC) force-upgrades the agent stack past the pyproject caps in a
   scratch venv and runs the deep+webapp gate — read its report before
   raising any cap.
4. **axial_pile beta cohesive_phi trap** — still item #1 in the ledger's
   Das-sweep ergonomics section (per-layer fix, global fallback).
5. **Sample-calc detector next targets:** SCDOT design examples, USACE EM
   appendices, more NHI manuals (doctrine in FUTURE_IDEAS header; ledger =
   module_work/wiki_verification/TIER_A_LEDGER.md).
6. **Eval:** GPT-5.x live rerun of the 108-Q suite (`eval_harness --ids PAV`
   for the pavement subset).
7. **Parked/major:** 6.0.0 restructure (module_work/V6.0_RESTRUCTURE_PLAN.md,
   dedicated session); auto-continue gap-taxonomy port; Document Intelligence
   OCR bake-off (needs live tenant); λ-method; Highter-Anders; foundry/ dir
   cleanup (NOT quick — 9 test suites import it).

Standing rules (unchanged, restated in the old §0a below): releases/tags ONLY
on the owner's word (v* tag push auto-publishes); additive default-preserving
changes; validate against published values, never tune; never run unpinned
installs on the cluster; owner is a practicing geotech engineer, not a
developer — actionable results, no git lectures.

## 0a. FOR THE NEXT AGENT (written 2026-07-20 at the Fable→Opus handoff) — prioritized pickup list [HISTORICAL — Foundry items superseded]

Current release: **5.9.1** (PyPI; refs 1.3.3). Master == released. Full gate
green (~10k tests). Auto-memory carries the running narrative; CLAUDE.md
release sections carry per-release detail; `module_work/wiki_verification/
TIER_A_LEDGER.md` is the verification/defect master ledger.

1. **Foundry 401 follow-up (owner-blocking):** the published app awaits the
   enclave admins' answer on LLM-proxy access (ticket text in docs/FOUNDRY.md
   troubleshooting). When it arrives, the fix is likely ONE
   `GEOTECH_FOUNDRY_HOST` line in the owner's app file. Then: upgrade app to
   5.9.1 (`maestro env pip install` → Publish and sync), run the sidebar
   Connection diagnostics, then the pavement shakedown prompts.
2. **axial_pile beta cohesive_phi trap** (nearest-to-defect ergonomics item,
   +12.4% naive-user error; doc claims tip-only, behavior is all cohesive
   layers) — fix per-layer, keep global as fallback; details in the ledger's
   Das-sweep section.
3. **Continue the sample-calc detector** (owner standing directive; doctrine +
   next targets in module_work/FUTURE_IDEAS.md header): SCDOT design examples
   next; then USACE EM appendices; more NHI manuals live in the owner's
   OneDrive Lib (search the WikiLLM at C:\Users\socon\OneDrive\Lib\_WikiLLM,
   library.db — copy WITHOUT its stale journal to read).
4. **Ergonomics backlog items 2-6** (ledger): Schmertmann gamma_soil +
   Iz subdivision; anchored_wall embedment-grid refinement + waterfront mode;
   two-gamma GWT bearing; Highter-Anders option; lambda method.
5. **Eval:** owner's GPT-5.x live rerun of the 108-Q suite (PAV questions
   new; `eval_harness --ids PAV` for the subset); triage vs the 68 keys.
6. **Parked/major:** 6.0.0 single-namespace restructure — now FULLY PLANNED
   with a measured surface (2,795 imports/517 files + enumerated string/path
   traps) and an owner-approved step-by-step:
   **module_work/V6.0_RESTRUCTURE_PLAN.md** (one dedicated session); worked-examples phase-2 (owner's firm reports, private);
   playbooks + recompute-from-report QC specs; Databricks launcher
   live-verify; foundry/ dir cleanup (NOT quick — 9 test suites import it).

Standing rules that must survive the handoff: releases/tags ONLY on the
owner's word (a v* tag push auto-publishes via OIDC); additive
default-preserving changes; validate against published values, never tune;
verify against the printed page before changing chart-derived code; ONE
agent per wave (parallel fan-outs burn the session limit); owner is a
practicing geotech engineer, not a developer — actionable results, no git
lectures; wiki/Das content = internal anchors only, never shipped.

## 0. Delta since the table below (2026-07-15 → 07-18; table not yet rewritten) [HISTORICAL]

Everything here is also in CLAUDE.md's release sections (read those) and the
auto-memory. Headlines:

- **Releases 5.7.0 → 5.8.0 → 5.8.1 → 5.8.2** (all owner-OK'd, all on PyPI, tags
  `v5.7.0`..`v5.8.2`; refs 1.3.2 + 1.3.3 released too): the complete pavement
  stack (AASHTO 1993 module + UFC 3-250-01 alternative method + pavement
  specialist agent), then two same-night Foundry-deployment patch releases
  (custom-RID clobber fix; Connection-diagnostics panel + persistent errors +
  `max_completion_tokens` for GPT-5 RIDs; extras folded into the core install —
  plain `pip install geotech-staff-engineer` now brings everything).
- **Owner's Foundry app IS PUBLISHED and running** (State gov enclave,
  stateobo.palantirgov.com, PDCS Sandbox / geotechStaffChatbot) but **blocked on
  a 401 from the LLM proxy = enrollment/permissions**, NOT our code. Full field
  notes + the admin ticket text: `docs/FOUNDRY.md` troubleshooting section. When
  the admin answers, the fix is likely one `GEOTECH_FOUNDRY_HOST` line in the
  owner's app file.
- **UNRELEASED on master (5.9.0 candidate, owner-gated):** the
  `worked_examples` system — 17 verified-by-execution exemplar calculations
  from real published design reports (`funhouse_agent/worked_examples.json`,
  adapter `find_worked_examples`/`get_worked_example`, prompt wiring, gate
  tests); webapp/tests added to the pytest gate; FOUNDRY.md troubleshooting;
  `module_work/FUTURE_IDEAS.md` (8 forward specs incl. phase-2 harvesting of
  the owner's own reports, playbooks, recompute-from-report QC).
- **Eval suite is now 108 questions / 68 keyed** (PAV-1..PAV-8 added, ground
  truth run on v5.8.0) and `eval_harness --ids PAV` runs a subset.
- **Do NOT casually delete `foundry/`** — attempted 2026-07-18, reverted: 9
  agent-wrapper test suites import it throughout (scope in CLAUDE.md).
- Gate at handoff: 9,169 + 844 (slope/fem2d) + wrapper suites green, ~9,950
  tests + 48-ish skips.

---

## 1. One-screen status (2026-07-15) [HISTORICAL]

> **HISTORICAL, §0 to the end of this file (written 2026-07-15/18).** Since then the repo is public, the release and master are in §0a-current, the `v5.1-todos` worktree is gone, `module_work/V5.4_PLAN.md` is closed (open work is in `BACKLOG.md`), and the eval suite has 106 questions (`funhouse_agent/geotech_test_suite.json`).

| Item | Value |
|------|-------|
| Repo | github.com/soconnell345-geotech/GeotechStaffEngineer (private) |
| **master HEAD** | **v5.6.0 release** (2026-07-15, owner OK'd) = 5.5.2 + the app A-workstream (A1–A8, `module_work/APP_PLAN.md`) + html_to_pdf + retaining_walls base-interface overrides (double-2/3 verdict) + the Foundry deployment path (`docs/FOUNDRY.md`). Prior tags: `b81ddaa` = v5.5.2 | — PUBLISHED to PyPI 2026-07-13. Line: 5.5.0 (post-5.4.1 train: correlated pairs, Bray-Travasarou, Lowe-Karafiath, aniso su, slope_report_package, inline Plotly) → 5.5.1 (provenance-audit doc fixes, persistent conversations, model picker, save hardening) → 5.5.2 (Databricks Prompter launcher run_on_databricks, eval-suite fixes: self-contained questions + DIR-1 path + discoverability aliases, fem2d schema backfill + positive-depth guards). Eval: owner ran the 100-Q suite live (GPT-5.1 driver, docs/geotech_eval_20260713.json) — 72% graded pass, 0 exceptions, all misses triaged (2 real fem2d issues FIXED, 1 suite bug FIXED, 5 questions rewritten, rest = weak-model behavior) |
| **Branch `v5.4`** | tip `eb38615`, PUSHED, fully MERGED into master (`aad984e`); stays checked out in the worktree for the next train |
| Submodule `geotech-references` | `3b25e0e` = **v1.3.1 on PyPI** (owner OK'd 2026-07-08; ufc_expansive figures complete, 42/42 page-accurate); parent pin `>=1.3.1` |
| Version string | `5.6.0` in `pyproject.toml` (master) |
| Validation suite | `validation_examples/` — 191+ passed (offline; V-001..V-054) |
| Full repo suite | **8557 passed / 48 skipped** (5.5.2 release gate, 2026-07-13) |
| **Publish status** | 5.5.2 + refs 1.3.1 PUBLISHED. Owner-gated next: GPT-5.4 eval rerun (quota); TinyApp hosting feedback (form submitted 2026-07-10) → fold env answers into webapp/; APP WORKSTREAM (plan of record: module_work/APP_PLAN.md, A1-A8); reference-wiki integration PARKED (owner 2026-07-13). Deferred: toe-circle search under-sampling, steep-phi Kc sensitivity, E2 default stays fellenius (owner). |

**⚠️ Release gate (still applies):** a `v*` git tag push **auto-publishes to
PyPI** via `.github/workflows/publish.yml` (OIDC trusted publishing). Do **not**
push a tag, bump the version, or merge `v5.4` to master without the owner's
explicit OK.

---

## 1b. Post-5.5.2 app workstream (2026-07-13/14; released in 5.6.0) [HISTORICAL]

Owner pivoted to app-heavy work (wiki integration PARKED). **All A-items built +
deployed locally** — plan of record + close-out: `module_work/APP_PLAN.md`
(owner copy `APP_PLAN_copy.md` at repo root). Headlines: calc sub-agent context
isolation (default ON, −84% measured, `module_work/A2_CONTEXT_DESIGN.md`),
crash-proof turn persistence, Agent/Analysis-depth/model pickers, per-conversation
working folder (`GEOTECH_DEFAULT_OUTPUT_DIR`), optional tracing (`GEOTECH_TRACE=1`
+ LangSmith envs, webapp/README §Tracing), industry memo
(`module_work/A7_INDUSTRY_MEMO.md`). Post-A owner-session fixes: bounded
auto-continue for mid-turn "Let me…" stops, download-button MIME types
(.md-as-.bin), recursion-cap visibility. Owner decisions: summarization backstop
SKIPPED; durable checkpointer PARKED; API thinking layer DEFERRED (per-model
gating needed); "Analysis depth" naming reserved "Thinking" for future API
control. Open backlog (next cycle): retaining_walls sliding-check convention vs
free-body discrepancy (owner wall session); CI mock eval subset. Webapp tests:
105 (`pytest webapp/tests -q`). Launch: `"Start Geotech App.bat"` /
`streamlit run webapp\app.py --server.headless true`.

---

## 2. What's on master (done + durable, to 5.3.0) [HISTORICAL]

Chronological, all merged and pushed. Full detail: `docs/V5.1_SUMMARY.html`,
the per-module `DESIGN.md`/`VALIDATION.md`/`UPGRADE_PLAN.md`, and the memory
note `project_le_fem_modernization` (auto-loaded).

- **v5.1 line** (the bulk of the work): v5.1 TODO sweep; LE+FEM modernization
  (rigorous GLE/Morgenstern-Price `slope_stability`; T6 + 3D-MC-return + GL99
  SRM `fem2d`); the `reliability/` module (FOSM/PEM/MC/FORM + COV database);
  calc-package figures/tables + plotly viewers (`calc_package/interactive.py`);
  the staged model-setup agent (`geo_project/` + `deep/setup_agent.py`, OFF by
  default). Intentional behavior changes vs 5.0 are listed in the summary page.
- **Post-v5.0 field-failure fixes** (from a real Funhouse session): the
  lateral-pile calc-package bug (package re-ran the analysis; an invented
  `E_GPa` was silently dropped → steel-default; the deep agent's virtual FS
  couldn't see real files) → adapter-ergonomics sweep across ~30 adapters
  (`require_params`/`reject_unknown_params`, documented params, self-verifying
  file writes); and the **Databricks `/Workspace` placeholder-write** fix
  (`funhouse_agent/_fileio.py`: content-verified writes + rescue-to-`/tmp`).
- **Phase E — published-example validation (DONE):** 25 worked examples from
  GEC/Caltrans/FLAC implemented as **87 offline pytest checks**
  (`validation_examples/test_published_v0*.py` + `RESULTS.md`). **Zero
  analysis-result bugs** — every discrepancy resolved to units / method-variant
  / convention. One additive `fem2d` capability gained en route (`roller_base`
  BC + `initial_stress_relaxation` for excavation/cavity unloading; 369 fem2d
  tests, 0 regressions).
- **v5.2 coverage Batch 1 (DONE):** four additive, strictly default-preserving
  capability adds, each validated against its published example and flipping a
  `RESULTS.md` row to PASS — see `module_work/V5.2_COVERAGE.md`:
  - Q1 `settlement/hough.py` — Hough granular (C′-index) settlement.
  - Q2 `pile_group.meyerhof_group_settlement` — Meyerhof (1976) SPT group settlement.
  - Q3 `axial_pile` — per-layer `toe_friction_angle` + `head_depth` offset.
  - Q4 `retaining_walls.mse` — steel bar-mat/welded-grid Kr (2.5→1.2) + F* curves.
- **5.2.0 released 2026-07-05** (with geotech-references 1.3.0; 5.1.0 never
  shipped) after the rc5 71-Q eval review (archived
  `docs/geotech_eval_20260705.json`/`.md`; fixes: drilled_shaft per-layer
  breakdown, +6 dispatch aliases, P1 recovered-split, optional-dep preflight).
- **v5.3 train (released 2026-07-06 as 5.3.0):** Batch-2 coverage 5/5
  (drilled_shaft rational GEC-10 chains; MSE LRFD external-stability CDRs; soe
  basal-heave-sidewall-shear + FHWA apparent-pressure anchored + log-spiral
  Caquot-Kerisel Kp; full Reese-1974 sand p-y; fem2d monolithic Taylor-Hood
  u-p Biot consolidation); slope_stability round 2 (15 new Slide2/ACADS/Duncan
  validation problems V-026..V-040; SS-6 noncircular-search robustness fix +
  rejection diagnostics; rapid drawdown 2/3-stage; Newmark + Jibson; infinite
  slope; Ito-Matsui piles verified vs the ORIGINAL 1975 paper); pdf_import
  round 2 (scale calibration, label→region, cleanup, vision grid overlay,
  vision↔vector cross-check); + **12 adversarial-review fixes** (headline:
  log-spiral Kp δ=0 Rankine anchor — the clamp was ~44% unconservative).
  Plans: `module_work/V5.3_PLAN.md` (+ its review-outcome section).

---

## 3. What's on branch `v5.4` (BUILT + gated; released as 5.4.0 on 2026-07-08) [HISTORICAL]

All six owner directives (2026-07-06) are DONE on `v5.4` (13 commits over
master; final gate 8279 passed / 48 skipped). Plan of record + per-item log:
**`module_work/V5.4_PLAN.md`** (read it before continuing v5.4 work).

- **D1 PDF user manual** — `docs/GeotechStaffEngineer_User_Manual_v5.3.pdf`
  (132 pp) + regenerable `docs/user_manual/build_manual.py`; problem catalog is
  auto-generated from MODULE_REGISTRY/METHOD_INFO so it can't drift from code.
- **D2 no-restart Databricks** — `funhouse_agent/runtime_check.py` hot-reloads
  a stale pre-imported `typing_extensions` before any langchain import;
  `dbutils.library.restartPython()` is now only the documented fallback;
  cluster-scoped install documented as the avoid-entirely alternative.
- **D3 layered disclaimers** — DISCLAIMER.md (ships in wheel), prominent README/
  PyPI section, ONE-TIME first-import stderr notice (marker
  `~/.geotech_staff_engineer/disclaimer_ack`; suppress `GEOTECH_NO_DISCLAIMER=1`;
  silent under pytest), `geotech-disclaimer` console script, and a standing
  basis-&-limitations block in both calc-package templates. Honest constraint:
  pip runs NO code on wheel install — these are the legitimate equivalents.
- **D4 visualization gallery** — `docs/gallery/index.html` + `build_gallery.py`:
  12 exhibits, every figure from a REAL validated run (ACADS search, drawdown
  3-stage, Newmark polarity, p-y families, GL99 SRM, V-023 consolidation,
  Duncan reliability, GEC-11 MSE CDRs, bearing/settlement, bearing graph,
  pdf_import demo).
- **D5 `drawing_ir/` module** — LLM-ready drawing digitization: unified IR
  (Line/Polyline/Arc/Circle/Text/Region w/ coords, layer/color, provenance
  dxf|pdf_vector|raster_trace, per-entity confidence), OpenCV raster leg
  (`[raster]` extra), and an agent query surface (digitize_drawing → handle;
  query_drawing: entities_in_bbox / lines_by_angle / text_near /
  candidate_ground_surface(proposal) / …; get_entities). Deterministic
  extractor owns coordinates; the LLM asks for slices. Deferred follow-up:
  geo_project ingestion wiring (flagged in drawing_ir/DESIGN.md).
- **D6 seismic reviewer** (first narrow reviewer) — shared checklist in
  `funhouse_agent/review_checklists.py`; surfaces: `.claude/agents/
  seismic-reviewer.md` (Claude Code) and `funhouse_agent.make_seismic_reviewer
  (engine)` / `make_seismic_reviewer_deep(model)` (Funhouse; scope = 10 seismic
  modules + 7 seismic references via allowed_agents). Template for the
  reviewer-family rollout (V5.4_PLAN F8).

**NEXT:**
1. **Owner Funhouse feedback on 5.3** (and optionally the v5.4 pieces). On the
   owner's OK: merge `v5.4`→master, bump 5.4.0, tag (auto-publishes).
2. **E1–E11 QC carryovers + F1–F8 creative builds** — all scoped in
   `module_work/V5.4_PLAN.md` (rapid-drawdown search wrapper, #96 Kc, pore-
   pressure grid, composite-EI, eval refresh w/ new-tool questions, more Slide2
   problems, Bray-Travasarou, reviewer family…).
3. 71-Q eval re-runs: `docs/geotech_eval_20260705.json`/`.md` hold the 5.1rc5
   baseline; re-run on 5.3+ with `[deep,full]` clears the 4 env-blocked fails.

---

## 4. How to run things (all offline unless noted) [HISTORICAL; current: CLAUDE.md]

**Environment:**
- Work in the git worktree `C:\Users\socon\OneDrive\dev\GeotechStaffEngineer\.claude\worktrees\v5.1-todos`
  (branch `v5.2-coverage`), NOT the main checkout. Merge to master with
  `git merge --ff-only` from the main checkout, as every milestone has.
- **Venv python (system `python` has no pytest):**
  `C:\Users\socon\OneDrive\dev\GeotechStaffEngineer\.venv\Scripts\python.exe`
- Git identity: `-c user.name=soconnell345 -c user.email=soconnell345@gmail.com`;
  end commit messages with the `Co-Authored-By` trailer.
- Tests: `pytest <module>/ -q`; validation: `pytest validation_examples/ -q`.

**Rebuild the test wheel** (from the main checkout, after merging to master):
`python -m build --wheel` → copy to `v5_test_wheel/`. **CRITICAL: bump the
version first** (`pyproject.toml`) — pip on Databricks skips a same-version
reinstall ("already installed with the same version… use `--force-reinstall`").
Also **verify the wheel contents** after building (a past rc shipped without a
just-added file): unzip and confirm the new files/tokens are present.

**Funhouse (Databricks) — notebook, needs API/`fh_prompter`:**
- Install: `%pip install "/tmp/...whl[deep]"`, then just `import funhouse_agent.deep`.
  `dbutils.library.restartPython()` is **no longer required** in the normal flow:
  `funhouse_agent/runtime_check.py` (run at the top of `funhouse_agent.deep`) reloads
  the freshly-installed `typing_extensions>=4.13` in place, curing the old
  `typing_extensions`/`extra_items` PEP 728 error without a restart. Restart is kept
  only as the fallback the auto-fix points to if an even older copy is winning at
  cluster scope; installing `typing_extensions>=4.13` as a **cluster-scoped library**
  avoids the situation entirely.
- Health check: `from funhouse_agent.deep.rc_wheel_check import run_rc_check; run_rc_check(fh_prompter)`.
- Eval suite (100 questions as of v5.4): `from funhouse_agent.deep.eval_harness import run_suite;
  run_suite(model, out="/tmp/eval")` (writes `.json` + a readable `.md`). Model =
  `PrompterChatModel(prompter=fh_prompter, model="funhouse-gpt-high")`. Real API
  calls; use `limit=` for a subset first. Correctness is PARTLY auto-scored
  (questions with `expected` keys) + partly eyeballed from the `.md`; process
  metrics (P1 hallucination rate, tool-error rate, rounds, latency, tokens) are
  always computed.
  - **Full coverage needs the optional-dependency extras:** install
    `%pip install "/tmp/...whl[deep,full]"` — with `[deep]` alone, ~12 questions
    (the gstools/pygef/ags4/pydiggs/ezdxf/SALib/pystrata/eqsig/liquepy/openseespy
    modules) fail honestly with "not installed" errors. `run_suite` runs an
    optional-dependency preflight and prints a "Missing optional packages" banner
    at the top of the `.md` when any are absent.
- Save outputs to `/tmp` or `/Volumes`, then `dbutils.fs.cp` out — NOT
  `/Workspace` (FUSE writes are non-durable / permission-blocked on the cluster).

---

## 5. Standing constraints & gotchas (carry forward) [HISTORICAL; current: CLAUDE.md and §0a-current]

- **No version bump→tag→publish without explicit owner OK.** A `v*` tag push
  auto-publishes to PyPI. rc strings are committed without tags on purpose.
- **`ANTHROPIC_API_KEY`** is read from the Windows *User* env at runtime — never
  pass it through chat/transcripts. Live tests are opt-in (`RUN_LIVE_TESTS=1`).
- **Owner is not a developer** — skip git/dev-procedure rationale; report
  actionable results. Prefers autonomous milestone-level operation, big batches.
- **Stagger subagents** — one at a time (usage-window limits); commit at
  milestones so a cutoff is cheap. Every module change must be **additive +
  default-preserving** (mirror the v5.2 Batch-1 pattern): new params/methods
  default to prior behavior; existing module tests stay green byte-for-byte.
- **Validate against published targets, don't tune to them.** A discrepancy is
  far more likely units / method-variant / convention than a module bug;
  investigate before "fixing." Record CONVENTION / N-A(scope) when the module is
  defensibly correct.
- **Edit/Write tools are pinned to the worktree** — to change a main-checkout
  file, author in the worktree + `cp`, or use a Python heredoc via Bash.
- **geotech-references is an editable install** resolving to the *main*
  checkout's submodule; after a submodule pointer change, `git submodule update`
  the main checkout.
- **`np.trapezoid`** (not `np.trapz`); SI units throughout.

---

## 6. File map (where the important things live) [HISTORICAL; open work: `BACKLOG.md`]

- `HANDOFF.md` (this file) — current authoritative handoff.
- **`module_work/V5.4_PLAN.md`** — CURRENT plan of record (owner directives
  D1–D6 all done; QC carryovers E1–E11 + creative F1–F8 = the open backlog).
- `module_work/V5.3_PLAN.md` — v5.3 board incl. the adversarial-review outcome.
- `module_work/V5.2_COVERAGE.md` / `module_work/WEEKEND_QC_2026-06-13.md` —
  historical boards (complete).
- `validation_examples/INVENTORY.md` + `RESULTS.md` — published problems,
  verdicts, owner notes (coverage-gap backlog lives in the notes).
- `validation_examples/test_published_v0*.py` — the 136+ offline validation tests.
- **planlens** (`C:/Users/socon/OneDrive/dev/planlens`, PyPI `planlens`,
  0.3.0) — the document-review + drawing-geometry package: `planlens/document/`
  (page map, structure, text, tables, markups, thumbnails; DESIGN.md),
  `planlens/tools/` (the LLM tool layer), `planlens/ir/` (the drawing IR;
  DESIGN.md), `CHANGELOG.md`. App-side bridge: `funhouse_agent/document_tools.py`
  (the seven document tools) + `funhouse_agent/adapters/drawing_ir_adapter.py`
  (the drawing tools).
- `drawing_ir/` — HISTORICAL PATH (v5.4 → split out as planlens 2026-09-04);
  the geo_project-wiring follow-up flag now lives in `planlens/ir/DESIGN.md`.
- `docs/GeotechStaffEngineer_User_Manual_v5.3.pdf` + `docs/user_manual/` —
  the 132-pp manual + regenerable builder (rebuild each release).
- `docs/gallery/` — 12-exhibit module visualization gallery + build_gallery.py.
- `DISCLAIMER.md` + `funhouse_agent/_disclaimer` surfaces — professional-use
  terms (README/PyPI section, first-import notice, geotech-disclaimer script).
- `funhouse_agent/runtime_check.py` — Databricks no-restart typing_extensions fix.
- `funhouse_agent/review_checklists.py` + `funhouse_agent/reviewers.py` +
  `.claude/agents/seismic-reviewer.md` — the narrow-reviewer pattern (seismic
  first; family rollout = F8).
- `funhouse_agent/deep/rc_wheel_check.py` — one-cell Funhouse health check.
- `funhouse_agent/deep/eval_harness.py` — `run_suite` (100-question eval) + scorers.
- `funhouse_agent/geotech_test_suite.json` — the 97 eval questions (71 base + 26 v5.3/v5.4, E10).
- `funhouse_agent/_fileio.py` — verified-write + `/Workspace` rescue helpers.
- `docs/V5.1_SUMMARY.html` — everything-since-5.0 summary (owner-facing).
- `docs/funhouse_agent_guide.md` — install + Databricks gotchas + snippets +
  reviewer-agent usage.
- `CLAUDE.md` — project instructions (auto-loaded); its status block is kept
  current and points here.
- Reliability / geo_project / slope_stability / fem2d each have their own
  `DESIGN.md` + `VALIDATION.md` + `UPGRADE_PLAN.md`.
