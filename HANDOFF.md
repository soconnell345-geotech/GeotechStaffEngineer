# HANDOFF — GeotechStaffEngineer (current state, read this first)

**Last updated: 2026-11-10** (release 5.35.0 with planlens 0.15.0, geotech-references 1.6.0, the new shared core `aec-agent-core` 1.0.0 and Document Review alone as `aec-document-review` 0.1.0, item 0 of "Where things stand"; open work is in `BACKLOG.md`). This is the authoritative handoff for a fresh LLM
session (any model). The older `HANDOFF_2026-06-14.md` is kept only for the
detailed Phase-E history; this file supersedes it.

---

## 0a-current. PICKUP LIST (2026-10-15, supersedes everything below)

### Start here (written 2026-10-21 evening; work restarted 2026-10-22 after the weekly limit reset; updated 2026-10-23)

- **Order of work (owner, 2026-10-21; BACKLOG §0 and its sections follow it):** report ingest and GSE on Foundry first, then the geotech calc modules, then Document Review and Tiny Apps last. Everything on master through ed929c7 (2026-11-10) is released as 5.35.0 with planlens 0.15.0, geotech-references 1.6.0 and the shared core `aec-agent-core` ("Where things stand", item 0); 5.34.0 (2026-11-04, through d57f28d) is item 0a.
- **First on Friday:**
  1. CI was green on e7159c0 (all of today's code); the notes commits after it were pushed the same evening. Check the run on the newest commit.
  2. Done 2026-10-22: report ingest on the geotech page behind `GEOTECH_REPORT_INGEST` (off) and the helpers' call budgets, reviewed and merged (7476767).
  3. Done 2026-10-22: the ingest engine on the page's own chat model, reviewed and merged (ae74d3f); next, OWN-11's one-run check on Foundry (in the item), in the next wheel.
  4. Done 2026-10-23: the log floor reads a column holding sample numbers and blows by its printed lines, checks a printed N against the drives and gives no N under 50 on a record that ends in a halt (five review rounds, merge 2322e0d); follow-ups RI-46 (Later) and one closed in the 2026-11-04 sweep.
  5. Done 2026-10-23: the log floor's N past a hard seating drive and its N-column marks (merge 380d07a): N = 2nd + 3rd on three whole drives however hard a drive was (p. 3-51), printed refusal marks kept as printed, N60 kept out of N; follow-ups closed in the 2026-11-04 sweep (today's floor, not run under model first).
  6. Done 2026-10-23: the ingest pipeline reads the narrative once over all its pieces, reads a triage-drawn bound document's pages on one side only, and has counts from the logs behind a switch, off (merge dd276db); follow-ups RI-53 (Later) and a Foundry measurement (closed in the 2026-11-04 sweep).
  7. Done 2026-10-23: lab tests reach the hole their sample label names (558 of 1,064 linked on the saved stage-b records, from 427; none lost), lab items over 12 pages read in windows (merge 03370ad); follow-ups a Foundry measurement (closed in the 2026-11-04 sweep) and OWN-88 (two lab sheet kinds, for the owner).
  8. Done 2026-10-23: the calc reader keeps every printed value by its place (two load cases under one label are both kept) and a printout as one item up to 22 pages, every scanned page shown as a picture (merge 835ceef; hand-truthed calc runs 96 -> 108 of 145 offline, items 611 -> 148, calls at most 1,134 -> 296); follow-ups a live run on Foundry (closed in the 2026-11-04 sweep), RI-58, RI-59.
  9. Done 2026-10-23: every model request in the cost record, failed and retried ones by kind, waits and reasoning tokens where reported; RESULTS prints them, the calls by step and the visual-scales counts; nothing sent to the model changes (merge 62dd373); follow-up RI-60. The calls per report with a bound document go to the owner with that live run.
  10. Done 2026-10-23: every QA entry ranked open, changed or note (about 156 open a report out of about 1,100), open-first listing behind a switch, off; a log's second sheet read as its own log folded into its hole (9 on stage b, all checked); squashed to b78f012 because its tests had copied values from saved records (replaced before merge, never pushed); follow-ups RI-61, PL-30.
  11. Done 2026-10-23: the pystra reliability tools on pystra 2.x with the cap lifted, and a uniform variable built on its bounds (it was built as mean a, std b) (merge 9f1b394).
  12. Done 2026-10-23: a privacy audit of the week's pushes found a few values copied from saved report records (no names, no contract number), now replaced with made-up values (merge dbe98a7); the owner chose not to rewrite old commits (2026-10-26); INF-52 (Later).
  13. Done 2026-10-23: the label review broke more page labels than it fixed (186 against 135 on stage b, in-sample); a change it makes without looking at the page is now refused, which on the same pages lifts accuracy from .864 to .888 (merge b966969); its follow-up closed in the 2026-11-04 sweep.
  14. Done 2026-10-23: the vote stage and the narrative-vs-appendix count mismatches measured offline: no voting rule beats the rules out of sample (rules .896, trust .895), and in all 24 mismatches the narrative stands, which is what the record already does; nothing wired; its follow-up closed in the 2026-11-04 sweep.
  15. Done 2026-10-23: the reconciler's check of the narrative's hole names against the logs read now treats 'B-1' and 'B-01' as one hole (merge 7892943); on stage b the names reported unread fall from 98 to 70; the other places that match hole names followed (item 18).
  16. Done 2026-10-23: the label review's rule (refuse a change made without a look at the page) now also applies to a resumed review's saved changes with no evidence and to a reused labels run's changes; the measure's rule column uses its own page count (merge 97179c5); the rest closed in the 2026-11-04 sweep.
  17. Done 2026-10-23: the weak narrative fields diagnosed on the open reports (2 misses the scorer's, 4 field definitions, settled by the owner's question list, RI-101); the lenient list score leaves out a piece of a matched truth item (strict unchanged); run files keep the whole answer so saved runs can be re-scored with no model call (577761f); follow-up RI-66.
  18. Done 2026-10-23: the remaining places that match a hole by name (the narrative's hole count, the dictionary scorer, the calc-subject link, the summary row against its lab sheet) treat 'B-01' and 'B-1' as one hole (merge 5c990f5); on stage b, 3 more calc links and 15 more row-to-sheet matches.
  19. Done 2026-10-23: the report-ingest measurement can read on a chat model, so page labels and the readers can be measured on Claude through Foundry's own classes (the prompter path is unchanged); the calc page-pictures and lab-link switches are recorded in the run files (merge 753fb49); the owner's Opus 5.5 trial goes in the next Foundry build; follow-up RI-69.
  20. Done 2026-10-23: `di_dir=False` withholds Document Intelligence text with no fallback folder (None falls back to a `di/` folder); chat-model runs' notes no longer speak of Funhouse; a label run reused across engines is named (merge a5355b0); the rest of RI-69 stays open.
  21. Done 2026-10-23: the SPT halt rule reads a record's own drive lengths and keeps the N past an occasional fourth drive; one merge row per halted N; the refusal field's wording from the printed FHWA page (fedf9c8; the reader's prompt unchanged); follow-up RI-70.
  22. Done 2026-10-24: lab values carry their own page's boring and depth, a table value that pins no specimen never overrules the model, sample links follow the printed label, summary rows linked, compaction named by its parts with a field-density guard (8ab414b, reviewed four rounds); new switch `GEOTECH_INGEST_LAB_FLOOR_PER_PAGE` (off); follow-up RI-71; the owner keeps the stand-in borings (2026-10-26), and a stand-in now links to a log spelled another way or is written as a test pit when named as one.
  23. Done 2026-10-24: a four-drive SPT record's N is the 2nd + 3rd in the merge (Fig. 3-20's occasional fourth increment), never replacing a printed N when it rests on the interval alone; a lone count with a unitless length stays checked; QA 'the record carries both' ranks open (merge a49a494); the rest of RI-70 (model-facing) stays open.
  24. Done 2026-10-24: lab links and the per-page lab floor tightened (hole numbers in sample labels, the mass guard with ratio units per FHWA Eq. 2-4, a one-letter C a cohesion only on shear sheets or with a stress unit, shared sample names by depth, a note for every unwritten summary row; 7cbb8a1, reviewed); offline no link or DIGGS value moved, 9 false shear tests gone; the rest of RI-71 stays open.
  25. Done 2026-10-24: the lab floor's second batch: range labels, several-sample labels, rows stopped at the next value's label, held moisture only on compaction sheets, no chemical value from header text, trial columns skipped (a96c8f8, reviewed); offline no link or DIGGS value moved, disagreements 225 -> 217; the hydrometer Gs stays out of DIGGS as a result (the owner, 2026-10-27: it is the value the calculation used).
  26. Done 2026-10-24: report ingest measured on Claude 5.5 locally on the API credit (MEASUREMENTS 'CLAUDE LOCAL RUN 1'): Sonnet ties GPT-5.4 on logs and narrative, Opus reads lab 768 / 826 against GPT's 622; the engine took a wrapped answer as empty (fixed the same day); the range labeller built and measured; a public blind set downloaded (RI-79).
  27. Done 2026-10-24: the engine unwraps an answer sent inside one extra key and no longer takes an answer the unknown-key drop left empty (2fd41c9, reviewed twice); the public set's AGS answer sheets merged (RI-79).
  28. Done 2026-10-24: the calc merge stops writing floor values over right readings and the calc scorer reads labels by their words and values by unit kind (RI-74, 0c8dc2a, reviewed twice): calc 175 / 181 / 184 of 203 offline for Sonnet / Opus / GPT-5.4; test values copied from answer sheets replaced.
  29. Done 2026-10-24: the plain-Opus baseline the owner asked for (250aebb; MEASUREMENTS 'CLAUDE LOCAL RUN 2'): handed each item's pages and our answer schema, one call reads blind logs 175 / 214 against the pipeline's 182 and blind lab 420 / 459 against 417, for about half the cost; the pipeline adds boxes, checks and fields the schema lacks; the owner chose model-first ingest.
  30. Done 2026-10-25: page labels by whole ranges from a description of every page, behind a switch that stays off (49f522f, reviewed twice): on the 13 labelled reports 50-56 pages go to the wrong reader against 290 today; it is the first piece of the model-first ingest.
  31. Done 2026-10-26 evening (session '5.34 reboot', picked up from the notes after the 10-24 history loss), the four fixes due before the model-first brief, each built by one helper and reviewed by another, fixed and re-checked before merge (BACKLOG Done has each with its commits): the calc reader no longer lets a zoom spend the continuation, drops stale 'pages not shown' notes, keeps each call's answer and maps cited positions to report pages under `refuse` (merge 2828e5e); a token brake for Foundry's $0 meter (`max_report_tokens_in`, `max_total_tokens`; a braked report is retried on resume), placement counts from `placement.json` and `arm.json` per report folder (merge 7436a09; the brief's run-5 fixes as stop point 7 and a counts block in MEASUREMENTS, ca77af2); an item skipped for error keeps its report unfinished, counted failed and retried on resume, RESULTS covers every finished report, and `max_request_tokens_in` trims and splits a log conversation (merge c09f630). The local gate caught the calc fix loading model-first code on the off path (the guard test); fixed by moving the shared page mapping to `report_ingest/checks.py`, and the off-path snapshot re-recorded for the deliberate change to today's calc reader. Filed the model-first request-limit item (done in item 33) and RI-122.
  32. Done 2026-10-26 night: wave 2's second hand-back (the R37 resume, the R28 re-read and the three copy arms zeros / fill / skipsbound) read in full by an Opus reviewer (private `module_work/field_feedback/2026-10-24_foundry_brief_1023/raw/wave2_part2/TRACE_REVIEW.md`; counts in MEASUREMENTS 'FOUNDRY RUN 5, second hand-back'). R37 read clean. R28's lost log item was re-read (3 calls, 0 failed), but its pages hold seven boring logs and the reader returned one hole; R28's record lacks 18 of the 77 holes it prints and R13 5 of its own, so several holes read as one became Now (since done), with the depths above the ground surface that passed every check (since done) and a retried report keeping only its last attempt's cost (done in item 33). The limit that lost the item was per minute, not per request (the AI FDE was asked, and answered on 2026-10-27; pacing built the same night). The copy arms changed nothing on these six reports (RI-53's notes). Also filed eight follow-ups. The brief's stop points 5 and 7 to 9 rewritten; the off arm now waits for the split by hole and the ground-surface check (both since done). Next: the ground-surface check (since done), then the brief, then INF-40.
  33. Done 2026-10-26 late night: model-first page groups sized against `max_request_tokens_in` before sending and cut by `_halves` (a log within its own pages; each cut a QA note), and today's log reader counting its system prompt and schema (merge b7b4e73); every attempt's cost kept (run.json `earlier_attempts` / `all_attempts`, RESULTS and results.json over every attempt, unfinished reports' spend a Cost line of its own and all of them under failures, the per-report brake counting every attempt; merge 72b361d). The brief's stop points 3 and 8 read the new fields. Filed a wave's ceiling across builds (since done) and RI-134. CI green on 6759183. In flight then: one log item per hole (since done).
  34. Done 2026-10-27 small hours (several helpers at once, owner's go-ahead): merged and gated on master: units with Unicode dashes and superscripts read, the DIGGS read-back gate's printed-N column (each column checked once) and a printed '-' N; narrative citations to twin fields kept; RESULTS names the ingest switches and counts a re-read narrative or split lab item once; `lab_sheets_per_call` (off by default) for long Atterberg runs; every cost row's start time and a failed request's size; `tokens_per_minute` pacing (off by default; Foundry's 400-as-rate-limit held like a 429); gstools tests write under tmp_path. The AI FDE answered the rate-limit question (limits are set by Foundry's admins, likely one shared pool; the owner has since committed Haiku 5.5 in the workspace; its RID is in the brief). Merged since and pushed (2378a86, CI green; the red CI on f1b7a00 was the DIGGS file's creation date in the off-path snapshot, now taken out): the lab hole read from the label link, a resumed record's cost and reuse flags, the harness guard refusing when git cannot answer, the linked hole on the summary page, the summary-row match by test kind, and the wave's token ceiling counting the folder. Merged after, not yet pushed: pacing per call (no cross-thread rows; retry-after 0 takes the ladder), the lab checklist per test kind, and one log item per hole (squash, reviewed twice; a follow-up since closed, and the model-first log and two-holes-one-name fixes merged below). Then the lab rows and sheets the match could not place (squash, reviewed three rounds; its follow-ups on turned sheets and grading sheets now merged below, and RI-151 open). Then depths above the ground surface written as 0 within 0.10 m and refused beyond, through to the DIGGS file (reviewed; its follow-up closed in the 2026-11-04 sweep). Then the log floor's blow records (more than four counts refused, side-by-side bands unassigned), echoed depths as one reading and sample tops from their drawn boxes within 24 in, and a letterhead's telephone line never a hole name (planlens too); all reviewed. Then turned pages read in their upright frame and grading sheets checked against their own legend rows (two reviews). Then two holes under one printed name kept apart by what the pages print, each result written in the log it was linked to (squash, reviewed three rounds). Then Foundry's own rate-limit error name (every one of wave 2's 221 rate limits was a 400 by that name, the AI FDE's table) read as a rate limit, so a give-up after the app's five minutes of retries is recorded and paced as one. The Later list triaged against the owner's rule: 125 items to Parked with a reason each. Merged too: the first cloud session's work (a package installed but not importing says why, not 'not installed'), reviewed. Merged: model-first logs get the missing-holes check too, and another hole's rows are dropped on today's reader and moved to that hole's log on model first (reviewed). Merged from a cloud session: the agent-stack caps raised to the versions tested (langchain 1.4.5, openai 3.29.0; 0140df2; release note: a fresh install now gets these), and page fingerprints that record the reference library they read. Merged: tick-flanked depth labels on a scanned log form no longer read as negative depths while planlens 0.13 runs (0778335; R14's refused rows 48 to 4); planlens's root fix is merged into planlens main (82660d3; R14's refused rows 48 to 0, R11 read off its real depth scale) and goes into build f's planlens wheel. Build f is with the owner (2026-10-28, INF-40): Foundry's Claude route sends is_error on every tool result, tokens in counts cache tokens, failures reach the totals, the model-first narrative goes in one call and calc groups cut at printout boundaries (1c0c3ae, CI green); `foundry_handoff/gse_app_2026-10-28.zip`; it re-runs MF-1's spoiled arms and all of MF-2. Before it, MF-1's trace review (`module_work/field_feedback/2026-10-27_foundry_build_e/TRACE_REVIEW.md`): the today arms lost 47 items to a field Foundry's Claude SDK requires on every tool result, and the run totals hid it; model-first ahead on logs, level on lab and calc, behind on narrative because a picture cap split each report; MF-2 on build e stopped itself at R36 on the same bug, as planned. Build e's first hand-back is in (evening): MF-0 and MF-1 done on Foundry (12 arms, all items, no give-ups; pictures route, Sonnet the vision voter, Haiku the describer); the package's token meter misses Foundry's cache tokens, so its spend brakes never fired (the AI FDE metered it itself); MF-2 next, in a fresh FDE session; the MF-1 trace review is running. Build e is with the owner (INF-40): the model-first measurement on Claude in Foundry, `foundry_handoff/gse_app_2026-10-27.zip`, app `5.33.0+foundry.20261027.1` from ea373e8; it waits on the owner's run. Earlier: the owner imported Haiku 5.5 into the FDE's workspace, and the FDE has four no-build asks (picture and tool calls on each Claude model, an optional large-request timing, the cache bypass, a table of run 5's rate-limit events). **The owner's rule from 2026-10-27: stop chasing edge cases.** A fix's follow-up becomes a backlog item only if it changes a value in the saved reports' outputs; otherwise it is a "noted, not fixed" line in the parent's Done entry. Report ingest starts no new floor follow-ups until the model-first numbers are in; then the report-ingest backlog sweep (applied 2026-10-29) swept the Later items against them. Owner questions on technical points stay welcome; record the answer, code it only if it changes a result.
  35. Done 2026-10-29/30: build g's robustness half merged (45712b7, CI green; reviewed by Opus, the explicit-off snapshot byte-equal to 25c30de): report ingest is model-first by default for every stage (page labels and grouping, logs, lab, calc, narrative), set stage by stage, so `logs=off,lab=off` sends those stages back to today's readers; a page group never read is a failed item and re-asked on resume; a server error is sent again, a cut-off group halved down to one page, a list sent as JSON text read (a refused answer kept in the run files), dividers set aside not listed as unread, the DIGGS writer reads a resistivity printed in Mohm/cm; the model-first narrative follows `narrative_skips_bound` (R16's parent narrative reads 78 pages, was 3). Then the narrative half merged (7de1f84, reviewed twice; the off-path snapshots change request text only): the owner's narrative rulings of 2026-10-29 and the companion phase question, repeated list items dropped and recounted, an over-long summary sent back once, one item per foundation type and bearing value, an incomplete copy flagged (R13 keeps its mat pressure and clears the check; R05 and R28 still flag), the other-client rule in the model-first narrative, the model-first narrative told the bound documents, and a caller's questions asked of it. The owner's question list v3 is private; the app reads it at run time. The range labeller now reads under the owner's conventions B by default (it had run A; 0fa2ed5). MF-3 (build g) is delivered to the owner for Monday 2026-11-02. The tool tests now skip an absent optional package by name, the Foundry install adds the frame tool's package without its dependencies, and fem2d's mesh-refinement study is exposed to the agent (15fadcb). The slope account is merged (b2602ea; an independent check confirmed its headline findings and corrected two) and waits on OWN-101 to OWN-104; GM-140 stays open. The prompt-clarity pass merged (4eb0780): the geotech page's tool text and prompts now say what the run did (45 fixes; no tool number changed; release note: `defaults_used` lists only the inputs a run used, so a clay-only pile package no longer lists Nordlund's settings; the unsourced-range line is applied, its measurement still open). Bearing capacity now combines the factors as printed (5108f75): at phi = 0 the Vesic set adds Brinch Hansen's d_c as UFC 3-220-20 Eq. 5-21 does, pinned to Example B-3 (9.969 ft against 9.97 printed, 9.751 before); Meyerhof refuses a base tilt or ground slope; release note: undrained footings with shape and depth factors come out about 3 to 4 % lower, and eval key BC-2's q_ult goes from 582 to 567 kPa. The reference library's rescue chapters were checked against the print (about 300 errors fixed; GEC-7 ch 1-5 and part of GEC-12 ch 7 rebuilt; the GEC-13 summaries dropped; app pointer 874bfb0); then the rest of GEC-12 ch 7 rebuilt from the print (refs 8ed9e19, 2026-10-31; the old text had made-up downdrag and lateral squeeze equations and a 0.008 distortion limit where Table 7-18 prints 0.005); a reference-search ranking slip found in its review was then fixed in general: when no section holds every word of a query, the matches are ordered by how many of its words each holds (refs 1f7cac5). The driven-pile module was then checked against GEC-12 ch 7 as printed (merge c513235, verified by hand). Release notes: toe resistance in sand rises 1-3.5 % from 29 deg (Fig. 7-16(b) re-read; GEC-12's North Abutment example 359.7 -> 371.5 kips); the beta method no longer has a phi-based default (Table 7-9 by soil type, or the layer's own beta and Nt); the result now carries factored resistances by Table 7-1. The reference library's other original-build chapters were then checked against the print: GEC-12 Vol I, GEC-6 and the Micropile manual (refs efd9b82, 63fb31f, 1c9a642; Micropile's ten chapters and GEC-12's ch 5, 6 and 8 rebuilt), and the figure catalogs now read captions from the caption block (refs ed8f3c1), and the other figure catalogs were then checked against the print too (refs f86addd and db3e5c4: 811 captions lose a List of Figures page number or running head). The reference library's digitised ch 7 charts were then read off the print (refs 295be94): every one but one had values not from the figure, e.g. C_F rose with phi where the print falls, and the agent's reference tool returned them. GEC-7's 66 printed headings with no section were added from the print (refs dd09a27). The drilled-shaft module was then checked against GEC-10 (2018) as printed (merge 6b2e7a4, reviewed; GEC-10 Example 10-5 853.0 against 855.4 kips printed). Release notes: the sand base takes N60 as the mean over two diameters below the base, as GEC-10 p. 10-15 prints (a single N60 at the base before); rock qu is held no higher than the concrete strength when `concrete_fc` is given; a new `rock_side_method='caving'` follows Eq. 10-22 and Table 10-3. Downdrag's toe and sand side were then checked against GEC-12 Table 7-9 and UFC 3-220-20 as printed (merge ce26958, reviewed, numbers checked by hand). Release notes: when not given, the sand beta and the toe's Nt come from Table 7-9 by the layer's `soil_class` (silt, sand or gravel), with `Nt_method` offering UFC Table 6-17's Cheng (2004) equation instead; between the printed range ends the module takes a straight line in phi', and the result says that is its own reading; in the worked example the curves now meet at full toe resistance, so the neutral plane moves 5.99 -> 17.66 m and the drag 226 -> 1,308 kN (OWN-105 asks which neutral plane should be the headline when they meet in soil that does not settle). The lateral-pile module was then checked against GEC-9, COM624P and API RP 2A as printed (merge f3553a9, reviewed; GEC-9 Example B.1 735.6 against 727 kip-ft printed). Release notes: Reese sand's ultimate resistance now follows GEC-9 Eqs. A-22 to A-24, about twice the old table at 30 deg near the surface, and GEC-9's four-segment curve is the default; API sand's C1-C3 and k come off the API figures; sand k goes by relative density (Table A-3); a partly fixed head now sits between free and fixed (the spring's sign was reversed); 1,506 of 5,475 recorded results moved, every one a sand case, none in clay or rock; the manual's lateral example moves 4.56 -> 4.04 mm and 382.6 -> 348.5 kN-m; V-017 reads 3.32 mm against 3.3. The pile-group module was then checked against GEC-12 ch 7 and UFC 3-220-20 ch 6 as printed (merge c1b8bbc, reviewed; UFC App. B-5.4's block 14,494 kips reproduced). Release notes: Converse-Labarre is gone, group efficiency now comes from GEC-12 and UFC Table 6-26 by soil type and spacing (the calc package gives none without `soil_type`); block failure is factored by the printed resistance factors; the AASHTO p-multipliers at 5b are 0.85 and 0.70 for rows 2 and 3 (1.0 before); the equivalent footing sums settlement to UFC's 10 % depth (the worked raft 16.45 -> 13.74 mm), with GEC-12's rule as the option; a p-multiplier below 3b spacing is refused. The settlement module was then checked against GEC-6, UFC 3-220-20, UFS 3-220-10 and FHWA NHI-06-088/089 as printed (merge a79cdb0, reviewed twice). Release notes: Westergaard stresses rise 11 % to 81 % (now read off UFS Fig. 4-10; the old formula was not Westergaard's); Schmertmann needs `gamma_soil` and runs a little higher between square and strip (+4.7 % at L/B 1.5); secondary compression with cv rises 8-12 % (t1 = t90 as printed); the elastic and time-rate results move under 1 %; an underconsolidated layer now warns (OWN-107 asks whether to refuse it). The ground-improvement module was then checked against GEC-13 as printed (merge e1c20e6, reviewed; GEC-13 Sec. 4.4's wick-drain example 299 and 503 days against 300 and 500). Release notes: wick drains now ignore smear unless a smear ratio is given, as GEC-13 says is typical, so spacings come out a little wider and times shorter (90 % in half a year: 1.30 -> 1.46 m); vibro-compaction reads the probe spacing off Fig. 4-34 by target relative density and refuses D50 and a target SPT N; the settlement ratio is named as GEC-13 names it (OWN-108 asks about the stress ratio default). The retaining-wall module was then checked against UFC 3-220-20, FHWA NHI-06-089 and GEC-6 as printed (merge cb48c8c, reviewed and re-checked; UFC App. B-1 reproduced exactly, NHI Example 10-3 within 0.5 %). Release notes: Coulomb walls with level backfill now match Rankine (wall friction follows the backslope, as NHI and UFC print), so their sliding factor drops (the owner's 2026-07-14 wall 1.339 -> 0.945, and it now just fails overturning at 1.996 against 2.0); cohesive backfill gets a tension zone (more thrust); bearing on soil uses the uniform pressure over B - 2e, which alone raises the bearing factor up to a third; concrete 23.6 kN/m3; FS sliding 2.0 when passive is counted; a descending backfill is refused; the backfill is taken as drained (a water table behind the wall is a new backlog item). The reference library's GEC-9 was rewritten from the print (refs 9a3b3ad). The excavation support (soe) and sheet-pile modules were then checked against the Caltrans Trenching and Shoring Manual, UFC 3-220-20 and GEC-4 as printed (merge f86c4bd, reviewed and re-checked). Release notes: braced support loads now follow the Caltrans hinge method, so the first support rises 15-25 % and the lowest falls 30-40 %; clay pressure envelopes rise (stiff clay 0.4 gamma H); the soe cantilever moment is taken at FS = 1.0 (12-53 % lower) while sheet_pile keeps UFC's factored diagram; sections are sized at 0.60 Fy (about 10 % more section); sheet_pile embedment is 1.2 D0 by default (+20 %); Terzaghi's basal heave shows UFC's chart beside it. Owner questions OWN-109 to OWN-111. The library's GEC-10 review fixes and GEC-4, rewritten from the print, followed (refs 0e74e70). Each copy of the library now keeps its own search databases, so two installed versions sharing a temp folder no longer serve each other's text (refs 51af51b). The library's GEC-5 (site characterization) was then rewritten from the print, every printed heading its own section (refs 39f779f, reviewed). GEC-8 (continuous flight auger piles) followed (refs 11b2f5b, reviewed). EN 1997-2's sections 1-6 followed, paraphrased with tables exact (refs 158e408, reviewed). Its annexes followed (refs a0f2cf4, reviewed). Release notes: the agent's EN 1997-2 tool returns G.1's well-graded DPH density index and F.3's Burland and Burbridge settlement for over-consolidated sand as printed (the old settlement fell as the pressure passed the preconsolidation pressure), and seven table lookups as printed. Release notes from it (refs f9471b8): GEC-8's p-multipliers by soil type, its allowable capacity at a factor of safety of 2.5 unless p. 138's four conditions are met (was 2.0), and its grout volume factor at 1.15 minimum (was 1.0). The soil correlations were then checked against GEC-5, UFC 3-220-10, GEC-12, NHI-06-088 and EN 1997-2 as printed (merge d7058fa, reviewed). Release notes: friction angle from SPT by Peck now uses N1,60 as UFC prints it (up to 2.4 deg lower at the same number; more through a soil profile's fill), Meyerhof's follows GEC-5's table (4-7 deg higher), relative density from SPT works (it had returned 100 %), and the SPT correction tool lists its choices with unchanged numbers (OWN-112 asks which overburden correction it should use by default). The wave equation module was then checked against GEC-12 as printed (merge 896d1a7, reviewed and re-checked). Release notes: friction piles no longer read refusal (GEC-12 Example 6: 92.6 blows/ft against GRLWEAP's 94, was 305); the built-in hammers take GEC-12 Appendix C's data, so blow counts drop 6-20 % (about half for the ICE I-30V2, a diesel); diesel runs warn that they are likely low; the pile-top driving stress no longer includes the helmet's inertia (Example 6: 52.5 -> 28.8 ksi); refusal is 120 blows/ft. Seismic geotechnics followed (reviewed and re-checked). Release notes: below 30 m the stress reduction factor is 0.5 as GEC-13 prints it (lower liquefaction FS at depth); site class follows GEC-12 Table 7-26 (some sites move one class softer); site factors and Fpga need a PGA; Idriss & Boulanger residual strength comes off GEC-12 Fig. 7-78 (lower); seismic passive pressure takes wall friction 0.4 phi by default (KPE about 25 % lower than at 2/3 phi); Hardin's Gmax uses A = 625. Owner questions OWN-113 and OWN-114. MSE walls were then checked against GEC-11 as printed (merge f2a1f69, reviewed and re-checked). Release notes: geosynthetic walls are checked for sliding on the lowest sheet (sliding FOS 11-30 % lower where it governs); a level with less than 3 ft beyond the failure surface fails; on a clay foundation the undrained strength counts in full (higher sliding FOS), with Eurocode 7's 0.4 V cap offered; the unprinted 2/3 c L term is gone. Pavement design followed (merge af5a33e, reviewed). Release notes: UFC flexible and rigid thicknesses come off Figs E-1 and F-1 as printed (often thicker: required thickness -6 % to +48 % flexible, -19 % to +88 % slabs), a 4 in subbase minimum and 6 in slab minimum; AASHTO treated bases from Figs 2.8/2.9, the 40,000 psi rule, check mode per layer, composite k off Fig 3.3; the eval keys PAV-4 to PAV-6 moved to the print's values. Owner question OWN-115. A water table behind cantilever and gravity walls followed (merge 3f8f835, reviewed). Release notes: walls take an optional water table, with the buoyant weight and the water pressure behind the wall and the uplift under the base as EM 1110-2-2104 prints it; walls without one are unchanged. Owner question OWN-117. The wave equation's drivability inputs, embedded length and shaft distribution followed (merge 96a2ff9, reviewed). Release notes: a drivability study runs one pile embedded to each depth, as GEC-12 does, so with no pile length given the shallow depths carry the pile above grade and their stresses rise (a made-up study's tension at 5 m 43 -> 150 MPa); the drivability tool takes quakes and damping; an embedded length and a triangular shaft distribution are offered. The reference library's pavement charts followed (refs 22c11ed, reviewed twice). Release notes: the agent's AASHTO and UFC pavement chart tools read off the print (a composite k that gave 3,026 pci now 487); pavement design's frost option refuses soils UFC Table 19-2 does not print unless a laboratory frost group is given, and clayey gravel with 20 % or less fines is F3 (was F2). Group uplift followed: a pile group with any pile in tension gets the group uplift check GEC-12 Sec. 7.2.3.2 and UFC 3-220-20 print, over the piles in tension from the outer row inward, with the cap's share (d14c0e2, reviewed twice; owner question OWN-118). Release note: a pile group with a pile in tension now reports the group uplift check. The MSE checks followed (f3cd92a, reviewed): the Service I eccentricity, Eq. 4-24 local shear and a grid's St limit as GEC-11 prints them, and a left-out retained or foundation friction angle is 30 deg as p. 4-10 prints for no data. Release note: walls that leave those angles out over a 34 deg fill show sliding and overturning about 15 % lower; 10 of 452 recorded calls go from pass to fail. The pavement charts left unchecked followed (58c389d, refs d8183a1, reviewed): UFC Figs E-2 to E-31, AASHTO Fig 3.4, Table 3.1 Step 7 and UFC Figs 19-5/19-6 as printed, read from the reference library only. Release note: UFC mixed-traffic passes (Table G-1's example 1,316,228 -> 1,296,044), AASHTO rigid designs with bedrock under 10 ft (V-074 k 522 -> 519) and the Table 3.1 performance period (13.21 -> 13.11 yr) move slightly. MF-3 (build g on Foundry, 2026-11-02) was read in full (`module_work/field_feedback/2026-11-02_foundry_build_g/TRACE_REVIEW.md`): model first, the default, reads better or level on every stage with truth over R15, R30 and R21 (logs 105/108 against 99/108, lab 75/75 against 71/75, labels 0.983 against 0.962) with 45 % fewer Opus and Sonnet tokens in; the two failed items share one cause, the model's parameter markup inside an answer with no re-ask. The owner answered the review's six questions on 2026-11-03: the narrative rulings are in both readers (b4acd8f, model-first prompt mf-8) and the truth corrections in the private truth, where MF-3 rescores with arm A's harmful labels on R21 15 to 0 and on R30 6 to 1; the Foundry copy of the truth is re-uploaded with the next brief (RI-203); the appendix-question search went ahead and is merged. The four fixes to make first are merged: the lost bound test pits, the pages handed back and the pit width (0484519; prompts mf-7 and ri73-4, so the next live run starts in a fresh folder), and the model's parameter markup put back or the answer asked for again once (75bd5c7). Four more are merged (2026-11-05); the last, RI-203, is open. After the hand-back the FDE's resume-once recovered both R21 failed items (2 Sonnet calls each), and the R15 question-list answer turned out to hold bare values; the resume's own hand-back arrived 2026-11-03 and is being read. 2026-11-03, GSE on Foundry: the owner wants the app opened on Foundry only once conversations survive a rebuild, so the next app wheel waits for INF-38: a raw-files dataset behind the app's conversation mirror (the AI FDE's answer), merged (eeff9dd, checked against the SDK's source), its calls still to be confirmed on Foundry; build h (`foundry_handoff/gse_app_2026-11-03.zip`, local) carries it with the round trip in the published app, the build c/d checklist and the audit's checks. The FDE's test from a published app then found the SDK route closed in app containers and the workspace dataset library working, so the backend was rebuilt on that library (4779f31: saves in the background, a sidebar warning when conversations are not kept, single user) and build i (`foundry_handoff/gse_app_2026-11-03b.zip`, local) replaced build h: on Foundry it installed and published with no workaround, every diagnostics line passed but the reference PDFs (none uploaded yet), and the first conversation saved to the dataset in the background; the restart-and-restore test is next; a new empty dataset now reads as an empty store and the sidebar shows each save's result by itself (abb4e2e, for the next build). The Tiny Apps wrapper takes our code from a source folder, wheels or Nexus, so CfA's answer to the code-route questions 6a-6d (OWN-44) is a file drop on dosdev (a894ce6, INF-34); OWN-44 needs no code. An audit of the Funhouse wiring on Foundry (`module_work/foundry_parity/FINDINGS.md`) found, besides storage: a rate limit on the Claude proxy stopped a turn (now waits and retries, merged 6b26d2f); the reference PDFs reach the chart tools only from the workspace repo's `references/` folder (INF-63); the token may expire on a long-open tab (INF-64); no viewer identity, so one shared conversation list. Ten checks only Foundry can settle (U1-U10) go to the FDE with INF-40's brief. The numpy-2 measurement (closed 2026-11-04: Foundry runs numpy 2.5.4) is `module_work/foundry_env_gate/2026-10-29_REPORT.md` (a follow-up for the frame tool is since done, below).
- **Waiting on the owner:** (Haiku 5.5 is committed in the Foundry workspace and in the app's model list, 2026-10-27; the rate limits are settled as the FDE found them, unreadable and one shared pool) the Foundry build c results (INF-40; read every run in full); the Claude-on-Foundry check, which needs the next wheel (INF-40); marking up the Report Questions Desk, https://claude.ai/artifact/RuUVqrbNnaHPaX1YaCyEjd (folded 2026-10-26); the core's name (decided 2026-11-04: `aec-agent-core`).
- **Held until the Foundry results:** EV-07, GM-196, GM-239. Still no slope_stability changes (GM-140).
- **Housekeeping:** the workspace moved to `C:\dev` on 2026-10-22 (OneDrive's sync was competing for the laptop's memory); start Claude Code there. The gitignored private folders are junctions to the old copy under `C:\Users\socon\OneDrive\dev`, which stays until INF-50. The merged helper worktrees were deleted; the unmerged ones stayed in the OneDrive copy, and no branch holds unmerged work from this week. `wip/ev06` must never be pushed (Department text in its history).

### Where things stand (5.35.0 released 2026-11-10; 5.34.0 on 2026-11-04, 5.33.0 on 2026-10-09)

**Module audit, stage 1 (2026-11-12; open in BACKLOG: GM-316 to GM-331, INF-77, DOC-24 and OWN-144; write-up `module_work/module_audit_2026-11-12/AUDIT.md`).** The owner asked for every calculation module to be run through for incompleteness, as the reference review did for the library. Ten read-only scouts read the modules by group against one yardstick (the agent can do everything it is told it can, and a result says what it is); a coordinator re-opened every severity-A finding and a sample of the rest. Confirmed: 32 A, 152 B, 186 C; none of the A findings was in the backlog. The registry itself is sound: all 195 methods the agent is told of have a function behind them. The gaps are of four kinds: a failure or a missing input hidden (Python warnings never reach the agent, GM-316; a pile longer than the soil profile runs on, GM-318; settlement drops layers, GM-320); the agent told of a parameter or output that is not there (support spacing and angle, GM-321; liquefaction outputs, GM-324); a result that does not say what it is (a failed SORM returned as SORM, GM-322; pystrata's effective stress, GM-325); and a value outside a tool's stated list running as something else (GM-317). Stage 2 is planned in waves (the audit's section 6): quick fixes (DOC-24); the two cross-cutting fixes and the standing tests (GM-316, GM-317, INF-77); interface fixes by module group; print checks for the modules never checked against their references (GM-329); and calc package gaps with rebuilt sample packages (GM-328, OWN-144). slope_stability was read for the record only. The same day: report ingest's structured export for Foundry is being built (RI-209, branch `feat/ri-report-tables`).

**Foundry brief 7 came back (2026-11-12; open in BACKLOG: EV-43, EV-44, PL-39, VIS-31, DR-43, INF-74, INF-75 and INF-76; the run itself and the scorer fix are in its Done list).** The fourteen coordination tasks ran on GPT-5.6 Sol in five set-ups and on GPT-5.4 as shipped: 84 runs, no errors. The scorer was reading each run's own page notes as if delivered on the two whole-set review tasks; that is fixed and the runs re-scored with no model call (`module_work/review_eval_results/2026-11-12_foundry_brief7/RESCORED.md`): Sol 87 of 95 checks as shipped, 91 `show_me`, 89 `method`, 92 `sheet_tools`, 92 all three; GPT-5.4 74. One run each, so no switch goes on; the model matters far more than the set-up, and the single-question tasks are at the ceiling on Sol. Every run has been read in full (`module_work/review_eval_results/2026-11-12_foundry_brief7/TRACE_REVIEW.md`, the per-run detail in `parts/` beside it; raw runs under the gitignored `module_work/field_feedback/2026-11-12_foundry_brief7/raw/`). `show_me` was offered and never used, so it against the page as shipped is run-to-run chance (87 against 91 checks), and no difference between set-ups is larger; the callout check is the one change with a visible effect (the missing detail delivered in 4 of 4 whole-set runs with it, 2 of 6 without). Sol read every planted problem; what it lost, it lost between a helper's readings and its reply or at the write. Four wrong statements in 282 comment-log rows. Its follow-ups are defects in the checks (EV-44), planlens' tools (PL-39), the vision calls (VIS-31), the page (DR-43) and the record (INF-75). The owner's steer for what follows: do not fit the harness to this model. Next, in order: Document Review on Foundry as a trial (INF-74), then one brief to the FDE for the stand-up, a probe of the Google model route and standing tests for each new model (EV-43); the owner answered the package question on 2026-11-12: the core and Document Review release as one unit (INF-76). First pass and plan: `module_work/review_eval_results/2026-11-12_foundry_brief7/FIRST_PASS_AND_PLAN.md`.

**Master since the 2026-11-10 package releases, the review's recommendations (unreleased; worked in the order the owner agreed).** First, harder test tasks (DR-25): the review suite has fourteen coordination tasks, `coord-*`, on one made-up building's documents (a 15-sheet set with 13 planted problems, the same set one issue earlier, the set as a scan, a 40-sheet version, a project manual, a geotechnical memorandum, a concrete submittal), 52 tasks in all. The core's checks gain `stated_together` (the two sides of a finding side by side, or under one heading) and read a delivered Excel file. A blind attempt checked the tasks, not the app (`module_work/review_eval_results/2026-11-10_coordination_tasks_blind/README.md`). No model had run them when they were built: the first run, Foundry brief 7, is back (the paragraph above), one brief for the page as shipped and each change that waits on these tasks (DR-38, DR-06, PL-37). Second, the main model looks for itself where it asks to (DR-38): behind `GEOTECH_VISION_SHOW_ME=1`, off by default, the page's three looking tools take `show_me`, which shows the image to the agent's own model at its next step instead of running a vision call; without it a look is today's. Suite arm `show_me`; pinned in `packages/aec-document-review/tests/test_review_show_me_offline.py`. Third in hand, the review method (DR-06): the draft's two files are package data (`aec_document_review/review_method/`), loaded only when `GEOTECH_INSTRUCTIONS` names that folder, and a suite arm (`review_eval.method_arm()`). Fourth, a sheet index and a callout check (PL-37, DR-07): planlens main d7f43d8 (unreleased) reads a drawing set's sheets from their title blocks and checks each callout against the sheet it names; behind `GEOTECH_REVIEW_SHEET_TOOLS=1`, off by default, the page and its reading helper get the two tools (`sheet_index`, `check_cross_references`; suite arm `sheet_tools`), and on the pinned planlens 0.15.1 the switch does nothing. With every switch unset no prompt, tool or build changed, and the page fingerprints are untouched. The run that measures the three changes was Foundry brief 7 (one zip in the local hand-off folder, test wheels built from app b95e5b5 and planlens d7f43d8); the owner sent it and it came back on 2026-11-12 (the paragraph above).

**Master since 5.35.0, the deployment review of Document Review (2026-11-10; its fixes released for Tiny Apps the same day, below; BACKLOG Done, `module_work/docreview_review_2026-11-10/REVIEW.md`).** The owner asked for a review of `aec-document-review` and the code under it before the testers' round: six read-only reviewers by slice (isolation on the shared host, the agent loop, the model connection and vision, SharePoint, saved files and the page shell, `planlens.review`), every finding checked against the code. Fixed on master with tests, in `aec-document-review`, `aec-agent-core` and `tinyapps/wrapper_repo/` (branch `review/docreview-fable`): on the App Service a request with no sign-in header is refused, a tester has no Working folder box, a restored zip or a path quoted in a tool result cannot reach outside the conversation, and the wrapper turns model-written plotting code off (OWN-133, the owner's call); a refused SharePoint upload raises (it was counted as saved), throttled calls wait, a 401 gets a new token; a pass the model-call budget ended is not auto-continued; control characters no longer lose a Word or Excel file; `run.sh` reports a failed install and takes the published address from the app setting `APP_ORIGINS`. planlens' four fixes (a marked copy that could replace the original, exact search missing text a reader can see, an over-limit error for a search with many hits, protected or empty PDFs) are on planlens' main (merge de93bc1, pushed). **Released 2026-11-10 (the owner's go, BACKLOG Done):** planlens 0.15.1 (tag `v0.15.1`, planlens 660a608), then `aec-agent-core` 1.0.1 (tag `core-v1.0.1`), then `aec-document-review` 0.1.1 pinning both exactly (tag `document-review-v0.1.1`); the wrapper's `packages.txt`, its README and `tinyapps/TINYAPPS.md` name the three. GeotechStaffEngineer itself is not released: 5.35.0 on PyPI keeps the core at 1.0.0, and master pins 1.0.1 for the app's next release. The Nexus mirror carries a release a day or two after PyPI. Open from the review: TA-37 to TA-41, DR-41, DR-42, VIS-30, PL-37, PL-38, INF-71, and a draft review method for DR-06 (`module_work/docreview_review_2026-11-10/review_method_draft/`, deployment instructions, not measured).

**Master since 5.35.0, before the review (2026-11-10; docs and tests only, no package change):** field-session place names out of the current files (BACKLOG Done: three field-feedback folders renamed, `webapp/tests/test_public_text.py` guards the place words in every tracked file and path; 7a2cb48), and every doc reviewed against the app split by four helpers (merges c36b2bc, 9029b50, 505e482; BACKLOG items the split made wrong corrected in 96164fb; the reference library's Tiny Apps row, refs 88bf0bc, submodule 9ba77d1; planlens' docs 9dad90d). The open split work is INF-33's exit (GSE on Foundry from the 5.35.0 wheels), INF-34 (the owner's first Tiny Apps round) and INF-35 (steps 15-16).

**The chart suite on Foundry (BACKLOG Done, 2026-11-09, merged with its review fixes; run 2026-11-06, every one of the 1,000 runs read in full 2026-11-08, `module_work/review_eval_results/2026-11-08_chart_suite_ev35/TRACE_REVIEW.md`; raw traces gitignored in `module_work/field_feedback/2026-11-08_chart_suite_ev35/raw/`).** 100 CharXiv chart questions, Opus 5.5 and GPT-5.6 Sol, five setups each, Sonnet 5.5 judging. On Opus the setups tie (95 to 97 of 99 after three wrong answer keys are set aside), so Document Review's vision side calls cost Opus nothing on charts. On Sol, the testers' model, Document Review as shipped scores 84 against 95 for the chart with a crop tool and Python (p = 0.003): Sol's main model never sees the chart and repeats the side call's reading, including its mistakes. Fixed 2026-11-09 (merge 111f3a9; BACKLOG Done): a failed vision probe no longer lowers the image detail for the process, and the vision side calls see the user's own question; letting the main model see an uploaded image is parked (DR-38). No rerun (owner 2026-11-09: "We need to start wrapping this stuff up"). **The wrap-up (owner 2026-11-09): a clean Document Review package for Tiny Apps.** The review experiments that never earned a place are stripped (BACKLOG Done 2026-11-09, merge b9c2843, about 9,500 lines; the rest went with the planlens binding, merge 25c16ea); then the split, with Document Review published as the public package `aec-document-review` on PyPI so Tiny Apps installs it from Nexus, a wheel by path the backup: released 2026-11-10 (item 0); still open are GSE on Foundry from the new wheels (INF-33), the owner's first Tiny Apps round (INF-34) and steps 15-16 (INF-35). Both names were first held on PyPI by empty uploads (2026-11-09, `1.0.0.dev0` and `0.1.0.dev0`). **planlens 0.15.0 is released** (2026-11-09, the owner's word; planlens 7e0bdeb, tag `v0.15.0`, on PyPI; suite on the release tree 2,097 passed): `planlens.review` holds the app's document tools, with the host's parts passed in one `ReviewHost`; `ReviewToolkit` takes a `look`; `Label (unit): value` reads as a quantity. The planlens binding is merged (2026-11-09, merge 25c16ea, BACKLOG Done): the app binds to `planlens.review` and requires `planlens>=0.15`; with it went the last off-by-default switches (vision text context, structured readings, turn retry, patch align, coverage). The core is filled (app split step 11 (a), (b) and (c), merges 798b0cc, f9c170f and 020fb1f, 2026-11-09; BACKLOG Done); Document Review is its own public package, `aec-document-review` 0.1.0 (app split step 13, merge 9411d1a, 2026-11-10), with its Tiny Apps template (step 14, merge 47db5a3); the releases followed the same day (item 0), pre-approved by the owner for the whole split (2026-11-09). New from the owner the same day: local deployments add their own references and instructions (BACKLOG Done, plan `module_work/REFERENCE_PACKS_PLAN.md`: reference packs, geotech-references 1.6.0 on PyPI 2026-11-09 with the app's half dd21205; appended instructions under built-in rules, merge 4f22c00).

**0. 5.35.0 (2026-11-10) with planlens 0.15.0, geotech-references 1.6.0 and the new shared core `aec-agent-core` 1.0.0: everything on master since 5.34.0 (through ed929c7), ending with a deployment's own instructions (merge 4f22c00), Document Review's own package (the app split's step 13, merge 9411d1a; INF-33) and its Tiny Apps template (step 14, merge 47db5a3; INF-34).** The owner pre-approved the app split's releases on 2026-11-09 (the core, `aec-document-review`, planlens and the GSE release that pins the core). planlens 0.15.0 and geotech-references 1.6.0 went to PyPI on 2026-11-09, and the releases went out in this order on 2026-11-10: the core, tag `core-v1.0.0` at 972ffad (`publish-core.yml`; on PyPI); `aec-document-review` 0.1.0, tag `document-review-v0.1.0` at 2237b31 (`publish-document-review.yml`, which needs only the core and planlens on PyPI; the first upload, from 972ffad, was refused by PyPI for the classifier 'Framework :: Streamlit', which it does not list, nothing was uploaded, and `webapp/tests/test_classifiers.py` now checks every classifier; a clean venv installs it from PyPI with nothing geotech); then the app, tag `v5.35.0` on `release/5.35.0` (`publish.yml` checks the core is on PyPI). The Nexus mirror carries them a day or two later.
- **The three items merged last (2026-11-10):**
  - a deployment's own instructions (merge 4f22c00, BACKLOG Done; `module_work/ref64/CHECK_INSTRUCTIONS.md`): `GEOTECH_INSTRUCTIONS` names a folder of `all.md` (every agent and helper) and one `<role>.md` per role (`main`, `references`, `reviewer`, `calc`, `figure`, `general-purpose`, `model_setup`, `report_library`, `report_ingest`); the text goes last in each prompt, once, under the built-in protected integrity rules and a fixed heading, so the built-in rules win and the guards in code are untouched; at most 4,000 characters a role, cut at a paragraph; an unreadable folder adds nothing and the record says why; report ingest gets the block on every model call; each `turn_start` (`engine.build.appended`) and the Connection diagnostics show what each role got, unused or misspelt files included. Unset, every prompt is byte-identical (fingerprints unchanged); no answer changes. Host setup: `docs/FOUNDRY.md`, `docs/DATABRICKS_INSTALL.md`, `webapp/README.md` §3, `tinyapps/TINYAPPS.md`; example folder `docs/examples/deployment_instructions/`.
  - Document Review's own package (the app split's step 13, squash merge 9411d1a; INF-33; `module_work/inf33/CHECK_STEP13.md`): `aec-document-review` 0.1.0 in `packages/aec-document-review`, a copy of the review page, its shell, Tiny Apps entry, tests and 38 suite tasks with nothing geotech, pinning `aec-agent-core==1.0.0` and `planlens==0.15.0`; the core at 1.0.0 names no application (each page hands in its own name), and GSE pins it exactly. GSE's two pages are unchanged (fingerprints byte-identical) and GSE still carries both; its copy of the review files is frozen by hash (`webapp/tests/test_review_freeze.py`) until step 15, so review-page changes go to the package. The package's page differs from GSE's review page only in `plot_data` (no `reference_chart`) and `plot_code` (no standard chart lines). Neither sdist ships tests; a `document-review` CI job builds and tests the package on Python 3.11 and 3.12 with nothing geotech installed. Not a GSE dependency. Its follow-up (field-session place names left in current files) is done, BACKLOG Done 2026-11-10: the names replaced, three field-feedback folders renamed, and `test_public_text` now guards the place words in every tracked file and path.
  - the Tiny Apps template for Document Review alone (the app split's step 14, merge 47db5a3; INF-34, open for the owner's first round; `module_work/inf34/CHECK_STEP14.md`): template and docs only, no package change; `tinyapps/TINYAPPS.md` has the first-round checklist.
- **Branches and tags:**
  - geotech-references 1.6.0: tag `v1.6.0` (76f494d), on PyPI; the app's submodule already points there, so nothing to cut;
  - planlens 0.15.0: tag `v0.15.0` (7e0bdeb), on PyPI; nothing to cut;
  - the core 1.0.0: `packages/aec-agent-core`, tag `core-v1.0.0`, published first (its files are the same on master and the release branch);
  - app `release/5.35.0` (on 1caf160, master through ed929c7 merged in): the version, the user manual's version row and rebuild, the BACKLOG sweep and these docs. The pins (`planlens>=0.15`, `geotech-references>=1.6.0`, `aec-agent-core==1.0.0`) came with the merges;
  - `aec-document-review` 0.1.0: `packages/aec-document-review`, tag `document-review-v0.1.0`, after the app.
- **Release gate on the exact tree:** GitHub CI on the release commit c5ad543 (run 42748165640, every job green; for speed this release used CI, not 5.34.0's local six chunks): the app's suite with the reference library 25,877 passed / 84 skipped / 2 xfailed on Python 3.12 and 25,878 passed / 83 skipped / 2 xfailed on 3.13, 0 failed; Foundry's numpy 1.26 job 2,676 passed / 7 skipped; core-only (nothing geotech) 238 passed on 3.11 and 3.12; document-review (the package alone) 816 passed / 2 skipped on 3.11 and 3.12.
- **Wheels and sdists:** CI's Wheels job on the same commit: the app's and the core's wheels built as a release builds them and installed together in a fresh venv (11 passed), neither package's sdist carries tests (2 passed), every classifier one PyPI accepts (3 passed). The app's wheel now installs only beside the core's (four wheels in a clean venv: the app, the core, planlens and the library).
- **Release notes: what changes a calculation's answer, by module.** Compare a 5.34.0 answer only with these in hand. Each is in `BACKLOG_DONE.md` (swept 2026-11-10) with its evidence, found there by its commit; the "Master after 5.34.0" record just below has every change in full. The app split (steps 6 to 11, 13 and 14), a deployment's instructions left unset and the Document Review strip change no answer: the page fingerprints stayed byte-identical through every step.
  - **fem2d:** seepage reports the flow through the section, `flow_m3_per_s_per_m`, from the fixed-head reactions; the old `total_flow_m3_per_s_per_m` was about 29 times too high on the printed check and grew with t, so every saved seepage flow changes (UFS 3-220-10 Fig. 6-5 prints 34 ft3/day/ft, the module gives 32.4; 63c7027). Footing capacity: a `surcharge` loads the ground beside the footing and `q_ult_kPa` is the gross pressure (GEC-6 Eq. 5-1; c = 50 kPa clay with q = 36 kPa +14 %, a weightless sand with q = 20 kPa 0 -> 131 kPa; 66c16cd); with no surcharge the soil's weight goes on first whenever gamma > 0 (c = 20, phi = 25: 433.9 -> 455.6 kPa on the default mesh; `max_displacement_m` includes the soil's own settlement, 31.6 -> 38.4 mm on c = 50, phi = 0; d45e509); the default iteration limit grows with the mesh, so fine meshes rise (c-phi, 80x40: 173.6 -> 455.6 kPa; the default 40x20 unchanged on the 15 cases checked) and a run stopped at the limit says its q_ult is a lower bound (db75981); a cohesionless soil with no surcharge warns that its q_ult is set by the mesh. New options, defaults unchanged: `footing='rigid'` (85ea79a) and a graded mesh for clay. `fem2d_local_fos` uses the `t`, `gamma_w`, `max_iter`, `tol` and `n_gp` it is given (same defaults; 2aef04d).
  - **drilled_shaft:** with the base in rock and `concrete_fc` given, `lrfd_capacity`'s factored base is the smaller of 0.50 Q_tip and the concrete's bearing 0.70 [0.85 f'c A1 m] (GEC-10 Eq. 12-7), so `phi_Qt_kN` and `phi_Qn_kN` drop where f'c is low against the rock; nominal values and runs without `concrete_fc` are unchanged (fa5528f).
  - **sheet_pile:** a `gwt_depth_passive` above the dredge line is water standing in the excavation, pushing on the wall's face from its level down (UFC 3-220-20 Sec. 4-4.1); only calls that gave one move, to a shorter embedment, a smaller moment and a smaller anchor force (a 5 m wall in sand with water 2 m down on both sides: D 14.5 -> 8.6 m, M 1120 -> 324 kN-m/m, anchored T 105 -> 43 kN/m; UFC Fig. 4-43's example D0/H 1.461 -> 0.868; 45c894e).
  - **wave_equation:** a run that leaves out the hammer cushion and helmet no longer takes 500,000 kN/m, COR 0.80 and a 5 kN helmet; `hammer_name` 'Vulcan 014' takes GEC-12 Vol 2 Table 12-2's driving system (a made-up 18 m pipe at 900 kN: 45.1 -> 46.3 blows/m, 145.4 -> 184.5 MPa), and any other hammer is refused until the cushion inputs or `driving_system` ('Vulcan 014' or 'ICE 42S') are given; library callers pass `cor` and `helmet_weight` (2d6a836).
  - **axial_pile (driven piles):** with `adhesion_chart` "fig_7_18" or "smaller", a clay under a softer clay is no longer read on Fig. 7-18's bottom graph unasked: where the middle graph would give another Ca, the run stops and asks for the stratum's case. The default Fig. 7-17 answers are unchanged; new, a layer's own `alpha` or `adhesion` with its basis, and its own `highly_overconsolidated` (e88a275).
  - **retaining_walls (MSE):** the critical-combination bearing takes the live load over the reinforced zone at 1.75, as GEC-11 Example E4 prints it (E4 5.75 -> 5.86 ksf, CDR 1.83 -> 1.79; no pass or fail changes; 05138e0). New answers only: the MSE calc package (`retaining_wall_package`, `wall_type="mse"`), a sloping or broken backslope, a battered face from 10 deg (refused before) and `wall_type="mse_lrfd"` (50b097a); vertical walls are unchanged.
  - **bearing_capacity:** the default is unchanged (350 cases); new option `depth_factor_method="linear"` with `factor_method="meyerhof"`, k = Df/B at every depth as UFC 3-220-20 Table 5-6 prints it (a square 1.6 m footing at Df 4.0 m, phi 34 deg: 4,373 -> 5,253 kPa with the option; 4093116).
  - **site response (pystrata, OpenSees curve checks):** unchanged on Python 3.12 with pystrata 0.5; on Python 3.13 the tools now install and run on pystrata 0.6 and give the 0.5.4 answers to round-off (51dbde5).
  - **reference tools the agent calls (geotech-references 1.6.0):** GEC-13's deep dynamic compaction depth is Eq. 4-1 as printed, D = n (W H)^0.5 with n = 0.5 (0.4 for a landfill), Fig. 4-21's 0.3 and 0.8 beside it (20 Mg dropped 20 m: 10.0 to 12.0 m -> 10.0 m, range 6.0 to 16.0 m; the four soil types it took before are refused; refs 8a09592); the DDC energy and jet grouting lookups give Tables 4-3 and 8-1 as printed, a new lookup gives Table 8-6's soilcrete strengths, and the deep mixing strength lookup is gone (refs 3748893); GEC-13's soil-nail bond, micropile grout bond and geosynthetic reduction factor lookups matched no printed table and are gone (an unknown-method error), with GEC-7, the micropile manual's Table 5-3 (dense sand Type B 120 to 360 kPa, where the GEC-13 alias gave 95 to 180) and GEC-11 answering instead (refs c0ed2b9); a groutability ratio between 24 and 25 reads "Feasible" (p. 8-29), and the suitability number's ratings use Table 4-7's words, with no rating between 40 and 50 (refs f62a1d1); the AASHTO 1993 ESAL lookups answer any printed SN, D and pt (unchanged at SN 5, D 9 in, pt 2.5), an off-grid pt raises ValueError, and pavement_design's design ESALs keep the off-table note (numbers unchanged; refs e4b0fa1, 8c597ae); auto search fuses the literal and synonym hits (gold set R@5 38 -> 41 of 54; refs 1427409); worked examples from GEC-9, GEC-10, GEC-11, UFC 3-220-20 and GEC-12 Vol III are searchable (refs 32bbd77); NHI-06-088/089's equations show their printed formulas (refs c76a404).
  - **report_ingest:** a log that prints the water level was not measured gets a `not_measured` water row, never 'not encountered' (a remark in DIGGS), and a lab result on combined samples is linked to every sample it combined (bf90a33); a log sheet with no fitted depth ruler keeps its depths within the hole (b47d77d); a calculation's printed table keeps every row (805 last-row labels to 0 on the saved records; 7179081); the narrative counts a bound Volume 1's or data report's holes as this report's, and the model-first narrative answers the appendix-answered questions from anywhere in the report (ad0a112); on model first, a lab test with no printed boring carries the one printed nearest before it, and a sounding printed over two pages is one investigation, so hole counts drop by one per pair (d9abfed); DIGGS points with a printed vertical datum are 3-D under DIGGS' compound coordinate system (f40474e); a resumed report's record states the tokens paid (47d3fc9); cited boxes move to a sample's column or a value's block, values unchanged (be9a354). Prompts mf-9 and mf-10, so a live run starts in a fresh folder.
  - **unchanged:** slope_stability (frozen for GM-140; its `yield_acceleration` tool now refuses a `kh` it used to ignore), settlement, lateral_pile, pile_group, downdrag, soe, seismic_geotech, ground_improvement, pavement_design's numbers, the soil correlations and reliability.
- **Behaviour and install changes (not a number):**
  - install: the app requires the shared core, `aec-agent-core==1.0.0`; it brings no third-party package the app did not already require, so an install log adds one line, `aec-agent-core-1.0.0`; Python 3.11 or newer, as the core and deepagents need (`requires-python` raised from 3.10, the 3.10 classifier dropped; every host runs 3.12; `aec-document-review` needs 3.11 too); `planlens>=0.15` (0.15.0 declares the same packages as 0.4.0) and `geotech-references>=1.6.0` (no new package); pystrata by Python version (`pystrata>=0.5,<0.6` and `pyrvt<0.9` on Python 3.12, `pystrata>=0.6,<0.7` on 3.13); a checkout installs the core first (`pip install -e packages/aec-agent-core -e .`), and Foundry's test wheels add the core's. Install guide §11.
  - the app split, steps 6 to 11 (2026-11-05 to 2026-11-09): the model connections, settings, identity, vision calls, Tiny Apps key routes, figure helper, plotting box, chart stack, worked examples, guards and records, Word and Excel, the checks, the review suite's framework and the shared assembly live in `aec-agent-core`; every old import path still works and gives the same module; a host that builds the review page without filling the key hook refuses a multi-user caller unless `PROMPTER_SHARED_KEY_FALLBACK` is on (GSE fills it); a core-only CI job builds the Document Review page with nothing geotech installed; the app's document tools run from `planlens.review`.
  - Document Review: the experiments that never measured a gain are removed with their settings (`GEOTECH_REVIEW_AGENT=lean`, `GEOTECH_VISION_INLINE`, `GEOTECH_REVIEW_SWEEP`, `_FINDINGS`, `_OVERVIEW`, `_GEOMETRY`, `_DIGEST`, `_KEPT_READINGS`, `GEOTECH_VISION_TEXT_CONTEXT`, `_STRUCTURED`, `_TURN_RETRY`, `_PATCH_ALIGN` and `GEOTECH_COVERAGE`); a setting still set is ignored, a removed suite arm is refused, and `build_deep_agent` and `build_review_page_agent` no longer take `coverage_dir`. A vision model busy during its first measurement no longer drops every later look to a smaller image, and the vision side calls see the user's own question.
  - the geotech page keeps 15 document and drawing tools off the model's list until the conversation holds a document (about 6,260 tokens off each call); `activity.jsonl` keeps each call's cache-read and cache-write tokens.
  - SharePoint: `GRAPH_CLOUD` (commercial, GCC High or DoD; unset, read off the site's address), a certificate or a secret, `SHAREPOINT_SITE_URLS` for several sites, a connection test (`python -m webapp.graph_sharepoint --check`, also in the sidebar diagnostics; `docs/SHAREPOINT_SETUP.md`); a pasted address on another granted site opens there; errors go to `_app_log/sharepoint-YYYY-MM.log` in the app's folder (`GEOTECH_SHAREPOINT_LOG=0` turns it off). Today's four settings send the same requests.
  - new settings, each changing nothing when unset: `GEOTECH_REFERENCES_PACKS` (a deployment's own references; `python -m geotech_references.packs check <dir>`), `GEOTECH_INSTRUCTIONS` (a deployment's own instructions, above), `GSE_REPORT_QUESTIONS` and `GSE_REPORT_POSTS` (the owner's question list inside report ingest) and `GEOTECH_RECOMPUTE_GAMMA_PRIME` (OWN-130); `GEOTECH_INGEST_WATER_NOT_MEASURED` is off; the new tool `recompute_report_calcs` (bearing, settlement, wall earth-pressure coefficients, AASHTO rigid slab thickness; DR-11) comes with report ingest, behind `GEOTECH_REPORT_INGEST` (off). `GEOTECH_FIGURE_HELPER`, `GEOTECH_PLOT_CODE` and `GEOTECH_JUDGMENT_CALLS` stay on by default.
  - tool refusals: a list item with a wrong key names the keys it takes, and a guess that starts like a parameter gets it as a did-you-mean.
  - the user manual's HTML is rebuilt from master and its module prose checked against the code and the prints (the PDF stays with DOC-10).
- **geotech-references 1.6.0** (released 2026-11-09; CHANGELOG 1.6.0): local reference packs (`GEOTECH_REFERENCES_PACKS`, `python -m geotech_references.packs`), and everything on refs main since 1.5.0 in the reference-tools bullet above, with search databases built under a temporary name and moved in whole (refs 9a198ff) and the manuals cited by their covers (refs 50e6af4).
- **planlens 0.15.0** (released 2026-11-09; CHANGELOG 0.15.0): `planlens.review`, the app's document tools with the host's parts in one `ReviewHost`; `ReviewToolkit` takes a `look`; `Label (unit): value` reads as a quantity. No new package; suite on the release tree 2,097 passed.
- **Not verified live:** the core and the app installed together from PyPI on a host (install guide §11); a deployment's instructions on a host; `aec-document-review` installed on Tiny Apps (INF-34); the Foundry restore round trip (INF-38); the SharePoint sharing-link call under Sites.Selected on a real tenant (`module_work/LIVE_TEST_QUEUE.md` check 13); the Tiny Apps tester deploy (OWN-44, INF-34).

**Master after 5.34.0: the train released as 5.35.0 (item 0 has the release notes by module); the record, change by change.** Built from 2026-11-05 by one-time cloud sessions, each reviewed independently before the lead merged it.
- install and library (BACKLOG Done, 2026-11-09; `module_work/pl29/CHECK_B.md`, `module_work/dr40/CHECK.md`): the app now requires planlens 0.15 or later; its document tools run from `planlens.review`, with no change to what either page offers or answers. Removed, all off by default: `GEOTECH_VISION_TEXT_CONTEXT`, `GEOTECH_VISION_STRUCTURED`, `GEOTECH_VISION_TURN_RETRY`, `GEOTECH_VISION_PATCH_ALIGN` and `GEOTECH_COVERAGE` (a setting still set is ignored; the `turn_retry` and `coverage` suite arms are refused). `build_deep_agent` and `build_review_page_agent` no longer take `coverage_dir`.
- Document Review (BACKLOG Done, 2026-11-09; `module_work/dr39/CHECK.md`): the review experiments that were off by default and never measured a gain are gone (sweep, findings, overview, geometry, digest, checklist, kept readings, the lean build; about 9,500 lines with their tests, suite arms and docs). The default pages are unchanged (fingerprints byte-identical); a removed `GEOTECH_REVIEW_*` setting that is still set is ignored, and a removed suite arm name is refused.
- vision (BACKLOG Done, 2026-11-09; `module_work/dr36_37/CHECK.md`): a vision model that answers "busy" while the app first measures it no longer drops every later look to a smaller image for the life of the process: the measurement is retried, a failed one is not kept, and each result made on the fallback says "the vision model could not be measured" until it is measured again; Claude is measured by name and unaffected. Every reading of an image, page, zoom or chart is now given the user's own question as context beside the agent's request. No calculated value changes.
- fem2d seepage (63c7027; BACKLOG Done): the result now reports the flow through the section, `flow_m3_per_s_per_m`, from the fixed-head nodal reactions, with the outflow and a balance check. The old `total_flow_m3_per_s_per_m` summed |v| x area x t; it was about 29 times too high on the printed check and grew with t. Every saved seepage flow changes. UFS 3-220-10 Fig. 6-5 prints q = 34 ft3/day/ft; the module gives 32.4.
- drilled_shaft rock socket (BACKLOG Done, 2026-11-05): with the base in rock and `concrete_fc` given, `lrfd_capacity`'s factored base is the smaller of 0.50 Q_tip and the concrete's bearing 0.70 [0.85 f'c A1 m] (GEC-10 Eq. 12-7, p. 12-8; p. 10-25), so `phi_Qt_kN` and `phi_Qn_kN` drop where f'c is low against the rock; the result says which governs. Nominal values and runs without `concrete_fc` are unchanged.
- geotech-references main after 1.5.0 (1427409, app submodule pinned; released in geotech-references 1.6.0): the default (auto) reference search fuses the literal and synonym hits instead of keeping only the literal top hit, so expansion no longer drops what the plain search ranked well (gold set R@5 38 -> 41 of 54); synonym pairs that name a different quantity are removed. Search results only; no calculation changes.
- report ingest, resumed runs (BACKLOG Done, 2026-11-05): a resumed report's record and summary now state the tokens paid, cache writes included, as run.json, results.json, RESULTS.md and the cost meter already did (one resumed run had said 2.31M in where 9.72M were paid). A fresh run's record is unchanged; folders saved before this resume but stay short. The report_ingest README is rewritten to match the code.
- wave_equation (BACKLOG Done, 2026-11-05): a run that leaves out the hammer cushion and helmet no longer takes 500,000 kN/m, COR 0.80 and a 5 kN helmet, which no held reference prints. With `hammer_name` 'Vulcan 014' it takes GEC-12 Vol 2 Table 12-2's driving system (1,420,629 kN/m, COR 0.91, helmet 7.43 kN; a made-up 18 m pipe at 900 kN goes from 45.1 to 46.3 blows/m and 145.4 to 184.5 MPa). For any other hammer the run is refused with the printed options: the cushion inputs with COR and helmet weight, or `driving_system` 'Vulcan 014' / 'ICE 42S'. Library callers must now pass `cor` and `helmet_weight`.
- geotech-references main 9a198ff (app submodule pinned; released in geotech-references 1.6.0): the reference search and figure databases are built under a temporary name and moved in whole, so several processes starting at once (users' first calls on a shared cluster, or a parallel test run) never read a half-built database; they stay readable by other users of a shared temp folder. No search result changes.
- fem2d footing capacity (BACKLOG Done, 2026-11-05): `analyze_footing_capacity` (`fem2d_footing_capacity`) now applies its `surcharge` as a pressure on the ground beside the footing (it only sized the load ramp before): the soil's weight and q first, then the footing raised from q, and `q_ult_kPa` is the gross pressure (GEC-6 Eq. 5-1). A c = 50 kPa clay with q = 36 kPa gives +14 %; a weightless sand with q = 20 kPa goes from 0 to 131 kPa; both within 1-2.3 % of c Nc + q Nq. Only calls that give a surcharge change. The staged excavation's `choices` state each layer's soil model and K0 (no value changes).
- report ingest, groundwater (BACKLOG Done, 2026-11-05): a boring whose log prints that the water level was not measured (or not recorded, or hidden by drilling fluid) gets a water row `not_measured` with no depth, instead of no row or 'not encountered'; a boring drilled with fluid that has no water row, no well and no piezometer gets a derived `not_measured` row (confidence 0.5). The DIGGS file says it in a remark (never `notEncountered`), the app's DIGGS reader returns `gwl_not_measured`, and the report gains one QA note listing the holes changed. The readers' prompt sentence for it stays behind `GEOTECH_INGEST_WATER_NOT_MEASURED` (off).
- report ingest, combined samples (BACKLOG Done, 2026-11-05): a laboratory result run on samples combined ('S-3 & S-4', 'S3+S4', 'S-1/S-2', 'composite of', or an unlabelled sheet whose printed depth range holds two or more of the log's samples) is linked to every sample it combined (`linked_sample_ids`, `combined: true`; `linked_sample_id` the shallowest). Before, it was linked to one sample or, with a conflict entry, to none.
- report ingest, log sheets with no depth ruler (BACKLOG Done, 2026-11-05): a boring log whose first or last sheet has no fitted depth ruler keeps the depths read on that sheet when they lie within the hole (between its other sheets, the ground and the printed total depth). Today's readers no longer drop those samples, layers and SPT records, and model first no longer writes `out_of_range` for them or lowers them to confidence 0.5. A hole with a ruler on every sheet is judged exactly as before. The incomplete-copy note no longer flags a photo appendix's 'Page X of Y' run, and RESULTS.md's calls-by-step table gains `describe` and `ranges` columns (totals unchanged).
- report ingest, question list (BACKLOG Done, 2026-11-05): the record now keeps the document type, its printed page numbers, its own list of appendices, the caption labels and each investigation's dates and firm, so code can answer three more rows of the owner's list once they are mapped in the private code map. Which document types mean yes on row 2 is OWN-127; row 29 stays open (RI-207). No saved value changes.
- citations and prompts (BACKLOG Done, 2026-11-05): drilled-shaft side resistance is cited as GEC-10 Eq. 10-3 (p. 10-9) in the tool texts and manual, and GEC-12 citations name their volume; no calculated value changes. The agent's prompts now say a coloured table's fills never reach the PDF.
- sheet_pile, water in the excavation (BACKLOG Done, 2026-11-05): a `gwt_depth_passive` above the dredge line now means water standing in the excavation, pushing on the wall's face from its level down (UFC 3-220-20 Sec. 4-4.1, Figs. 4-5/4-6). Before, it was taken at the dredge line and its pressure was left out. Only calls that gave such a level move, and they get a shorter embedment, a smaller moment and a smaller anchor force: a 5 m wall in sand with water 2 m down on both sides goes from D 14.5 m to 8.6 m and M 1120 to 324 kN-m/m (cantilever), T 105 to 43 kN/m (anchored); UFC's Fig. 4-43 example from D0/H 1.461 to 0.868.
- report ingest, narrative (BACKLOG Done, 2026-11-05; prompts mf-9, so the next live run re-reads every saved model-first result): a report with a Volume 1 or data report of the same investigation bound inside it now tells its narrative that the bound document's borings, test pits and soundings are this report's, so previousInvestigationCount should drop by one for each and boringCount include those holes. Model-first narratives can now answer the seven appendix-answered questions (for example an ASCE 7 edition printed only on a calculation page) from passages anywhere in the report, never from a bound previous investigation's pages. The narrative glossary's last DRAFT mark is off.
- fem2d footing capacity, soil weight first (BACKLOG Done, 2026-11-05): with no surcharge the soil's weight is now applied before the footing load whenever gamma > 0. Every no-surcharge answer with gamma > 0 and phi > 0 changes: c = 20, phi = 25 goes from 433.9 to 455.6 kPa on the default mesh (less than its spread with the mesh, 499.0 to 433.9 kPa). A cohesionless soil with no surcharge now returns a non-zero q_ult, but that value is set by the mesh and is not a bearing capacity; the tool says so in a warning (the rigid footing and fine meshes below). `max_displacement_m` now includes the soil's own settlement on every gamma > 0 call without a surcharge (31.6 to 38.4 mm on c = 50, phi = 0, gamma = 18; q_ult unchanged).
- reference library, worked examples (BACKLOG Done, 2026-11-05; refs 32bbd77): the agent can now search and fetch GEC-9 Appendix A, GEC-10 Appendix A (design example), GEC-11 Vol I Eqs. 4-21 to 4-24 and Table 4-6, UFC 3-220-20 Appendix B Example 3 and GEC-12 Vol III Appendix D (comprehensive design examples), all as printed. Reference answers that needed those pages can change; search on the other manuals is unchanged.
- tool refusals (BACKLOG Done, 2026-11-05): `fem2d_local_fos` now uses the `t`, `gamma_w`, `max_iter`, `tol` and `n_gp` it is given (it accepted and ignored them; the defaults are the same, so a call that left them out is unchanged). `yield_acceleration` refuses a `kh` (it was ignored; the method finds kh itself). Refusal messages name the keys a list item takes and suggest the parameter a guess starts like. No calculated value changes for a call that leaves out what it did not set.
- app split, steps 6-8 (BACKLOG Done, 2026-11-05): the Document Review page now loads one geotech package and four app modules instead of four packages and 16-17 (the chart stack moved to `funhouse_agent/charts`, the review page builds itself in `funhouse_agent/review_build.py`). No answer, prompt or tool text changes, and every old import path still works. Importing `profile_figure` or drawing a module figure loads nothing of the agent; a figure that uses the layout-ink check still does. Step 9, the planlens binding, is done (2026-11-09); steps 10-11, the core, are done (2026-11-09).
- report ingest, where a value is shown on its page (BACKLOG RI-99 (1)-(2), 2026-11-05): on model first, a lab result on a table printed one column per sample is now boxed in its sample's column at the property's row (anchor `sample column`), where before it got its sample label in the header row; a lab test not found on its cited page can move to its sample column on another page of its call. A calculation value printed in repeated blocks (per boring, per load case) now takes the block whose own heading names it (anchor `number in its block`), where before it was `several` (page only). A value that names no block, and a table whose layout is unclear, are placed as before. Only the cited boxes move; no value changes.
- the activity record (BACKLOG INF-66, code part, 2026-11-05): each model call in `activity.jsonl` now keeps the tokens read from the prompt cache and written to it (`cache_read`, `cache_creation`), and each turn adds them up beside input and output, on Claude, Foundry GPT and the Funhouse Prompter. `input_tokens` means what it did (cache included), so uncached input is input minus the two. Old activity files read as before. The next Foundry turn will show whether calls 2-4 read the cache (about $0.23 a turn if they do, about $0.60 if not).
- report ingest, the owner's question list (BACKLOG RI-101 (1), and the post match in BACKLOG Done, 2026-11-05): with `GSE_REPORT_QUESTIONS` naming the private list (its code map beside it), every report's run writes the answer sheet beside its record, with two model calls (up to four when one is asked again) counted in the record's cost; a list that cannot be read is a QA note, never a failed run. With `GSE_REPORT_POSTS` naming a posts file (JSON or CSV with name and full name), the post-name row is matched to one post (exact, then full name, then a close spelling of the short name) and says which; no match, or two posts that close, is an open QA note. With neither setting, nothing changes.
- report ingest, lab links and two-page soundings (BACKLOG Done, 2026-11-05): on model first, a lab test whose pages print no boring or sample now carries the one its run printed nearest before it (a cut part of a lab item takes the item's first part's), never into pages described as naming a hole of their own; so `linked_investigation_id`, `linked_sample_id` and the lab results written into the logs gain entries and those tests' `unlinked` QA entries go. A dynamic probe or cone sounding printed over two pages, the second carrying on from the first, is now one investigation instead of two of one name: hole counts and the "two holes under one name" notes drop by one per pair, and the DIGGS file has one test where it had two. No request a model is sent changes.
- fem2d footing capacity, a rigid footing as a choice (BACKLOG Done, 2026-11-05): `fem2d_footing_capacity` takes `footing='rigid'`. The footing is pushed down in settlement steps and q is read from the reactions (GEC-6 p. 55: Eq. 5-7 assumes a rigid footing), with `base='rough'` (the default) or `'smooth'`, and the tool returns the pressure-settlement curve. The default stays `footing='flexible'`, so no saved answer changes. On the module's uniform meshes the rigid footing reads high and falls as the mesh is refined (c = 50 kPa, phi = 0, B = 2 m: 347.9 to 283.1 kPa on 20x10 to 80x40, against (2 + pi) c = 257.1), and on a cohesionless soil it gives no plateau. Every result now lists the footing model in `choices`. Fine meshes: the bullet below.
- the geotech page's document tools wait for a document (BACKLOG Done, 2026-11-05): on a geotech turn with no document, 15 document and drawing tools are kept off the model's list, about 6,260 tokens off every main-agent call (about a quarter of the prefix). They come back as soon as the conversation holds a document: an upload, a SharePoint download (in the same turn too), a working folder the user chose, and always on Databricks or wherever extra read roots are set. `read_pdf_text`, `analyze_pdf_page` and `analyze_image` stay listed. The page fingerprints were re-recorded; the Document Review page is unchanged. No calculation's answer changes.
- report ingest, recomputing a report's calculations (BACKLOG DR-11, first slice, 2026-11-05): new tool `recompute_report_calcs` on the geotech page (it comes with report ingest, so off by default) and `report_ingest.recompute.recompute_record`. They re-run the bearing-capacity and settlement calculations an ingested report claims, through the app's own modules, and return a table: claimed, recomputed, difference, and a flag. 'differs' only past 10 % (this table's own QC threshold, not an engineering criterion) outside every option the module ran; 'within_options' inside their range; 'below' for an allowable under q_ult/FS (UFC 3-220-20 Sec. 5-2.4 allows it where settlement controls). Each row gives the method used and why, the options run, the assumptions and the inputs with their pages. An input the report did not print is named, never filled in; other calculation kinds are listed with the reason they were not recomputed. No existing output changes.
- report ingest, DIGGS with a printed vertical datum (BACKLOG Done, 2026-11-05): where a log prints its elevation datum (NAVD 88, CGVD28, mean sea level), the DIGGS file now writes the point in 3-D under DIGGS' own compound coordinate system for that datum, latitude, longitude, height, as DIGGS' 2.6 documentation example writes it; before, every point with a height was 2-D with no datum. NGVD 29, local or assumed benchmarks and a datum the report says it is not using stay 2-D. The app's own DIGGS reader and plan figures read the new form. DIGGS 2.5.a validation stays an option (pydiggs 1.x ships its schema).
- driven piles, a layer's own adhesion and the Fig. 7-18 cases (BACKLOG Done, 2026-11-05): a clay layer takes a site-specific `alpha` or `adhesion`, with its `adhesion_basis`, in place of the alpha method's chart (GEC-12 Eq. 7-10, p. 246; NHI-06-089 Example 9-3), and a layer's own `highly_overconsolidated` decides API alpha past the chart (p. 247). **Release note:** with `adhesion_chart` "fig_7_18" or "smaller", a clay under a softer clay is no longer read on Fig. 7-18's bottom graph unasked; where the middle graph would give another Ca, the run stops and asks for the stratum's case on its top clay layer. The default Fig. 7-17 answers are unchanged.
- the user manual (BACKLOG DOC-10, the HTML part, 2026-11-05): `docs/user_manual/manual.html` is rebuilt from today's master (it said 5.33.0 and showed old anchor inputs, a fixed 1.50 test load and old drilled-shaft defaults). Every optional parameter now shows its registered default, and the drilled-shaft and pile-group prose says what the code does. The PDF, Appendix B and the screenshots are still open (DOC-10); the rest of the hand-written module prose was checked 2026-11-06 (the bullet "The user manual checked against the code").
- planlens main (unreleased, so the app's pin `>=0.14` is unchanged), quantities printed as `Label (unit): value` (BACKLOG Done, 2026-11-05): `find_quantities` now reads "Pc (kip): 437.0" as 437 kip with its label; hand-calc lines, ratios and a label running onto the line above are refused, and every quantity planlens read before comes back the same, except that the next field's label is no longer read as a unit ("437.0 M (kN-m)" gave 437 m). Released in planlens 0.15.0 (2026-11-09).
- planlens 0.15.0 (released 2026-11-09), app split step 9a (BACKLOG Done, 2026-11-05): the app's document review tools that only wrapped planlens now also live in `planlens.review`, with a host that can be set per conversation (look, signer, readers, budgets). The app switched over on 2026-11-09 (merge 25c16ea; the binding table in `module_work/pl29/CHECK.md`).
- reference library, citations and test hygiene (BACKLOG Done, 2026-11-05; refs pin 912a99d): GEC-13's stone-column improvement factor now cites Eq. 5-4 (p. 5-32, the stress reduction method) instead of Fig. 5-5 and Priebe; UFC 3-250-01 is cited by its printed title, "Pavement Design for Roads and Parking Areas"; the references agent's catalog is read from the package; the refs tests no longer rebuild the shared search databases from every test, which ends their errors when run in parallel. No calculation's answer changes. The other GEC-13 citations were fixed next (the bullet below).
- pystrata held below 0.6 (2026-11-05): pystrata 0.6.0, out that day for Python 3.13+, renamed the classes the site-response tools use, and CI's 3.13 job failed 23 tests; master now requires `pystrata>=0.5,<0.6` and `pyrvt<0.9` (pyrvt 0.9.0, out the same day, drops a class pystrata 0.5.4 imports). The released 5.34.0 does not have the caps, so a Python 3.13 install of it breaks site response; every host runs Python 3.12 today (Databricks, Foundry, CfA's fleet), so none is affected and the caps ship with the next release.
- The app split's step 10, the shared core (BACKLOG Done, 2026-11-06): `aec-agent-core`, empty (1.0.0.dev0), lives in `packages/aec-agent-core` with its own build, CI job and wheel test, and the app pins it exactly. **Release note:** no behaviour changes; the app now installs with the core: install `aec-agent-core` first (CI, nightly and the cloud briefs do: `pip install -e packages/aec-agent-core -e .`). `publish.yml` refuses an app release until the pinned core version is on PyPI; `publish-core.yml` publishes the core alone from a `core-v*` tag. The name is held: `aec-agent-core` 1.0.0.dev0, empty, is on PyPI (tag `core-v1.0.0.dev0`, 2026-11-09, after the owner added the pending publisher). Foundry's test wheels need a fourth wheel, the core's, from step 11 (a). `module_work/inf32/CHECK_STEP10.md`.
- The app split's step 11 (a), the core's first fill (BACKLOG Done, merge 798b0cc, 2026-11-09): the model connections, settings, identity, vision calls and the Tiny Apps key routes move into `aec-agent-core`, now 1.0.0.dev1; GSE hands in its Key Vault lookup and per-tester keys through `aec_agent_core.hooks`. **Release note:** no behaviour change on either page (fingerprints byte-identical; a 13-scenario key probe identical); every old import path still works and gives the same module; a host that builds the review page without filling the key hook now refuses a multi-user caller instead of serving the shared key, unless `PROMPTER_SHARED_KEY_FALLBACK` is on. The app cannot be released until the core 1.0.0.dev1 or later is on PyPI (`publish.yml` checks). `module_work/inf32/CHECK_STEP11A.md`.
- The app split's step 11 (b) (BACKLOG Done, merge f9c170f, 2026-11-09): the figure helper, the plotting box, the chart stack and the nine geotech worked examples move into `aec-agent-core`; GSE hands in its reference chart lines by a registration. **Release note:** no behaviour change (fingerprints byte-identical; the plotting box's rules and twelve escape probes identical); every old import path still works. `module_work/inf32/CHECK_STEP11B.md`.
- The app split's step 11 (c), the core's last fill (BACKLOG Done, merge 020fb1f, 2026-11-09): guards and records, Word and Excel, the checks, the review suite's framework (each page hands in its own turn), the document tools' binding and the shared assembly move into `aec-agent-core`. **Release note:** no behaviour change (fingerprints byte-identical; guards refuse exactly what they refused); every old import path still works; a new CI job, core-only, builds the Document Review page with nothing geotech installed; the app's tests keep its data folder in a temporary folder, so a test run no longer writes into `~/.geotech_webapp`. `module_work/inf32/CHECK_STEP11C.md`.
- Local reference packs (BACKLOG Done, merge dd21205, 2026-11-09; geotech-references 1.6.0). **Release note:** the app requires `geotech-references>=1.6.0`. A deployment can add its own references: an FDE writes each manual as a pack in the library's format (`pack.json`, chapter files, an optional figure catalog with an optional crop box and reading notes per figure, the PDF) and points `GEOTECH_REFERENCES_PACKS` at the folder; `python -m geotech_references.packs check <dir>` checks it first; the librarian then searches, cites and reads its figures beside the built-in references, and each turn's record names the packs in force. With the setting unset nothing changes. Host notes in `docs/FOUNDRY.md`, `docs/DATABRICKS_INSTALL.md` and `webapp/README.md`; the format in the refs repo's `docs/REFERENCE_PACKS.md`. `module_work/ref64/CHECK_APP.md`.
- Deployment instructions (BACKLOG Done, merge 4f22c00, 2026-11-10): `GEOTECH_INSTRUCTIONS` names a folder of `all.md` plus one `<role>.md` per role; the text is appended once, last, to every agent and helper (report ingest and the report library included), under a protected list of integrity rules and a fixed heading: built-in rules win, and code guards are untouched by any text. **Release note:** unset, nothing changes (every prompt byte-identical); set, each turn's record and the diagnostics show what was appended and any file not used (a misspelt role); 4,000 characters a role. Where the folder goes on each host: `docs/FOUNDRY.md`, `docs/DATABRICKS_INSTALL.md`, `webapp/README.md`, `tinyapps/TINYAPPS.md`; an example in `docs/examples/deployment_instructions/`. `module_work/ref64/CHECK_INSTRUCTIONS.md`.
- The app split's step 13, Document Review's package (BACKLOG INF-33, merge 9411d1a, 2026-11-10): `aec-document-review` 0.1.0 in `packages/aec-document-review`, a copy of the review page with nothing geotech, on `aec-agent-core` 1.0.0 and planlens 0.15.0 (both pinned exactly); GSE pins the core at 1.0.0. **Release note:** GSE's two pages are unchanged (fingerprints byte-identical); the core names no application, each page hands in its own; the package's page differs from GSE's review page only in `plot_data` (no `reference_chart`) and `plot_code` (no standard chart lines); from now on review-page changes go to the package, GSE's copy frozen by `webapp/tests/test_review_freeze.py` until step 15; neither sdist ships tests. `module_work/inf33/CHECK_STEP13.md`.
- The app split's step 14, the Tiny Apps wrapper (BACKLOG INF-34, merge 47db5a3, 2026-11-10): the template in `tinyapps/wrapper_repo/` installs Document Review alone from Nexus (`aec-document-review==0.1.0`, the core and planlens pinned exactly), or the two wheels by path; `tinyapps/TINYAPPS.md` holds the owner's numbered first-round checklist and the platform team's asks. **Release note:** template and docs only; no package changes. `module_work/inf34/CHECK_STEP14.md`.
- pystrata 0.6 on Python 3.13 (BACKLOG Done, 2026-11-06): the site-response tools (`pystrata_agent`'s equivalent-linear and linear runs, `opensees_agent`'s curve checks) build the same Darendeli, Menq, linear and custom soils on pystrata 0.5 or 0.6 (`pystrata_agent/compat.py`); the requirement is by Python version: `pystrata>=0.5,<0.6` and `pyrvt<0.9` below 3.13, `pystrata>=0.6,<0.7` from 3.13. **Release note:** the site-response tools now install and run on Python 3.13 with pystrata 0.6 as well as on Python 3.12 with pystrata 0.5; no answer changes on 3.12; on 3.13 the runs give the 0.5.4 answers to round-off. `module_work/inf68/CHECK.md`.
- Bearing capacity: Meyerhof's own depth factors as an option (BACKLOG Done, 2026-11-06; `module_work/gm309/CHECK.md`; owner question OWN-131): `depth_factor_method="linear"` with `factor_method="meyerhof"` takes k = Df/B at every depth as UFC 3-220-20 Table 5-6 (p. 294) prints it; the default (arctan beyond Df/B = 1, printed for Brinch Hansen) is unchanged over 350 cases; the recompute check runs the option beside the default for a Meyerhof printout beyond Df/B = 1. Release note: default unchanged; new option (square 1.6 m at Df 4.0 m, phi 34 deg: q_ult 4,373 to 5,253 kPa); 2 saved recompute rows move from 'differs' to 'within options'.
- GEC-13's deep dynamic compaction depth as Eq. 4-1 prints it (BACKLOG Done, 2026-11-06; refs 8a09592; `module_work/ref63/CHECK.md`): `equation_4_1_ddc_depth` (old name and alias kept) gives D = n (W H)^0.5 at n = 0.5 for most soils or 0.4 for a landfill, with the depths at Fig. 4-21's n = 0.3 and 0.8 beside it; the four soil types it took before (Lukas 1995, not printed in GEC-13) are refused. Release note: for a 20 Mg tamper dropped 20 m the answer moves from 10.0 to 12.0 m to 10.0 m (range 6.0 to 16.0 m); a waste fill's 7.0 to 10.0 m becomes a landfill's 8.0 m.
- GEC-13's deep dynamic compaction and jet grouting lookups as printed (BACKLOG Done, 2026-11-06; refs 3748893; `module_work/ref62/CHECK.md`): the DDC lookup gives Table 4-3's unit applied energy by deposit type, the jet grouting lookup Table 8-1's advantages and disadvantages, a new lookup Table 8-6's soilcrete strengths (Burke 2004); the deep mixing strength lookup is gone (chapter 7 prints no table; p. 7-23's 100 to 500 psi is text); Vol II is cited as FHWA-NHI-16-028. Release note: these three lookups' answers move (the old energies per pass, passes, waiting times, depths and per-system diameters and strengths were printed nowhere); the refs release needs a version bump.
- Report recompute check covers rigid pavement thickness (BACKLOG DR-11 slice 4, 2026-11-06; `module_work/dr11/CHECK_SLICE4.md`): an AASHTO 1993 rigid slab thickness D a printout claims is recomputed through `pavement_design.rigid_pavement_design` from the printed W18, reliability, So, serviceability loss, S'c, Ec, k, J and Cd; a missing So, J, Cd or k is refused by name, never taken from the module's defaults; a D rounded up to a construction increment agrees only when the printout says it rounds. On the saved records 24 of 24 recomputed thicknesses agree (median 0.05 %). Release note: `pavement` calculations claiming a rigid D now recompute; a flexible design printed beside it is read as flexible (not yet checked).
- Report recompute check covers wall earth-pressure coefficients (BACKLOG DR-11 slice 3, 2026-11-06; `module_work/dr11/CHECK_SLICE3.md`): Ka, Kp and K0 (Rankine, Coulomb, log-spiral Kp, Jaky) and Mononobe-Okabe KAE/KPE a wall printout claims are recomputed through the wall and seismic modules' own functions, one row per coefficient and case; kh is used as printed, or from a printed PGA only by a rule the printout states. Method names one letter off a known method (two off a long one) are read as it, and the row says so. On the saved records 86 of 107 wall coefficients agree (median 0.09 %); the 21 that differ are reading errors or real departures in the reports. Release note: retaining-wall calculations that claim a coefficient now recompute; a misspelled Meyerhof is judged as Meyerhof.
- Calculation tables keep every row (BACKLOG Done, 2026-11-06; `module_work/ri208/CHECK.md`): a printed table inside a calculation printout reaches the record as one value per row and column, labelled by its column and row ("Unit weight (pcf), Layer 2"), on both the floor and model-first readers, where it kept only the last row; a row settles only onto a reading that names it; row numbers compare as numbers; the scorer matches row-qualified labels. Offline on the saved records: 805 last-row labels to 0, table values 1,828 to 4,208, hand values found 2,238 to 2,237 (the one was a wrong-row credit). Release note: the model-first calc prompt is mf-10, so the next live run re-reads saved calc answers; the recompute check declines a value printed once per layer rather than using the last row's.
- fem2d footing capacity, a faster iteration tried and not merged (BACKLOG GM-308, part (a), 2026-11-06; `module_work/gm308/CHECK_A.md`): Anderson acceleration and the other schemes the cloud session tried did not beat the plain iteration on every case, and with the option on the answers near collapse varied by machine (far below the default on four of eleven cases on Linux), so only the notes are on master and the code stays on its branch. Frictional footing answers remain lower bounds, as the warning says; GM-308 (c) is the next idea. No saved answer changes.
- Report recompute check reads printouts with several cases (BACKLOG DR-11 slice 2, 2026-11-06; `module_work/dr11/CHECK_SLICE2.md`): a bearing or settlement calculation splits by its printed case (scenario, stratum by number or soil name, load case) and each case is recomputed with its own inputs only; more label spellings are read and ratios, water pressure, drain thickness and unconfined strength are no longer taken as inputs; the shape comes from a case's own L and B. A printout giving only gamma' is still refused by default ('water level not printed'); `GEOTECH_RECOMPUTE_GAMMA_PRIME=submerged` runs it as the submerged case, which is OWN-130. On the saved records: 1 bearing claim checked by default, 167 of 183 within 10 % with the submerged setting. Release note: rows that were not recomputed or not compared now recompute, one row per case.
- AASHTO 1993 ESAL lookups read the full tables (BACKLOG Done, 2026-11-06; refs pin e4b0fa1): the four `esal_*` lookups answer any printed SN (1-6) or slab thickness D (6-14 in) at pt 2.0, 2.5 or 3.0 from Appendix D, where they refused all but SN 5 / D 9 in at pt 2.5 (unchanged there); a load of zero or less, or an off-grid pt, is refused with a ValueError. A load, SN or D off the printed table is read at the nearest printed value with a note, and design ESALs now carry that note in their notes (they dropped it; numbers unchanged). A committed HTML page posing as a UFC PDF is gone. A review matched 3,554 of 3,555 cells to the print (the one is the cell the scan cannot read). Release note: the refs release changes the error type for an off-grid pt. `module_work/ref12/CHECK.md`.
- The user manual checked against the code (BACKLOG Done, 2026-11-06): about 60 module sentences fixed against the code and the prints (the false "no group uplift method" among them), the references table from the covers, and 13 reference modules listed as references rather than as analysis modules with a table per lookup (the manual is about 1,400 lines shorter; no chapter or module page lost). Rebuilt against refs c76a404. Release step added to CLAUDE.md's task table: rebuild the manual after the version bump and commit it with the release. Release note: the manual's module descriptions match the code and the printed references; no calculation changes. `module_work/doc23/CHECK.md`.
- Soils and Foundations manual equations show their formulas (BACKLOG Done, 2026-11-06; refs pin c76a404): reference fetches of FHWA NHI-06-088/089 now show the printed formula, symbols, units and page for 52 equations, every one the app cites among them (drilled-shaft Eqs. 9-37a/b and 9-39 to 9-47, Schmertmann Eqs. 8-16 to 8-18, SPT Eqs. 3-2 and 3-3, earth pressure Eqs. 10-2 to 10-5, 10-10 and 10-11). A review checked all 52 against the rendered pages. No calculation changes. `module_work/ref38/CHECK.md`.
- Cited report numbers match the manuals' covers (BACKLOG Done, 2026-11-06; refs pin 50e6af4): GEC-10's 2018 data cites FHWA-NHI-18-024 (not the 2010 NHI-10-016), GEC-13 Vol II cites FHWA-NHI-16-028, and GEC-13's figures report their volume. ufc_backfill keeps UFC 3-220-04N: the held 04FA prints none of its tables. No calculation changes. `module_work/ref31/CHECK.md`.
- MSE walls, the rest of the calc package (BACKLOG Done, 2026-11-06): the MSE calc package takes a sloping or broken backslope and prints GEC-11 Vol I Sec. 4.4.4's terms; `mse_wall` and the package take `face_batter`, and from 10 deg the wall is battered (Coulomb Kab with theta = 90 + omega, the thrust at delta - omega, Fig. 4-5's thrust plane under a slope, Eq. 4-26 internally with its bracket squared, Fig. 4-9(b) for geosynthetics); `wall_type="mse_lrfd"` writes an LRFD package with Strength I (min). **Release note:** new answers only (a battered face was refused before); vertical walls do not change. A battered face under a broken backslope, or with LRFD, is refused. `module_work/gm307/CHECK.md`.
- GEC-13 table lookups GEC-13 never printed, removed (BACKLOG Done, 2026-11-06; refs pin c0ed2b9; its follow-up, three more mis-cited lookups, done the same day): soil-nail bond strength, micropile grout bond strength and geosynthetic reduction factors matched no printed table and are gone from GEC-13 with their aliases. **Release note:** an old GEC-13 call to them now gets an unknown-method error; GEC-7 Tables 4.4a, 4.4b and 4.5 answer soil-nail bond, the micropile manual's Table 5-3 answers grout bond (`micropile_grout_bond_strength` on `micropile` now returns it: dense sand Type B 120-360 kPa, where GEC-13's alias gave 95-180), GEC-11 Vol I Tables 3-9 and 3-11 and the p. 3-33 creep list answer reduction factors; a GEC-11 'geotextile' durability query now returns both printed rows (1.6/2.0 and 1.15/1.3). The refs release needs a version bump (three public functions removed: 1.6.0 or 2.0.0, the owner's call) and a raised `geotech-references>=` pin. `module_work/ref61/CHECK.md`.
- fem2d footing domain width (BACKLOG GM-308 (b), 2026-11-06): the default half-width of 5 B does not cut off the failure mechanism on sands (phi 30-40, 5-15 B, by runs); no code change, no answer moves. Frictional answers stay iteration-limited lower bounds (GM-308 (a)). `module_work/gm308/CHECK.md`.
- MSE walls, calc package and Strength I (min) (BACKLOG Done, 2026-11-05): `retaining_wall_package` with `wall_type="mse"` writes the MSE wall's calc package, each step with its GEC-11 page (level backfill and vertical face only), and MSE LRFD reports Strength I (min) eccentricity and bearing apart. **Release note:** the critical-combination bearing now takes a live load over the reinforced zone at 1.75, as GEC-11 Example E4 prints it; E4's critical bearing stress goes from 5.75 to 5.86 ksf and its CDR from 1.83 to 1.79, with no pass or fail changing. A sloping backfill, a battered face and an LRFD package followed (the bullet above).
- fem2d footing capacity on fine meshes (BACKLOG Done, 2026-11-05): the low answers on fine meshes came from the solver stopping at its iteration limit; the default limit now grows with the mesh. **Release note:** fine-mesh answers rise (c-phi flexible footing, 80x40 mesh: 173.6 to 455.6 kPa); the default 40x20 mesh is unchanged on every case checked (15). A run that stops at the limit says its q_ult is a lower bound. A mesh graded towards the footing's edges is offered for clay (phi = 0) and refused on frictional soils. Frictional answers stay limited by the iteration count even on the default mesh: GM-308.
- SharePoint set up by the host team themselves (BACKLOG Done, 2026-11-05; owner: "build it so that CfA can flexibly set up the sharepoint registration themselves"): the Microsoft cloud is a setting (commercial, GCC High, DoD); a certificate works as well as a client secret; several sites can be listed; a connection test (`python -m webapp.graph_sharepoint --check`, also in the sidebar diagnostics) says which step failed and what the admin does; `docs/SHAREPOINT_SETUP.md` is the admin's guide. Nothing changes for a setup with today's four settings. The registration itself is now CfA's to do (OWN-44).
- SharePoint for testers (BACKLOG Done, 2026-11-06; owner: "They use some Azure gov cloud ... might be helpful for a user to point at another one ... get those error messages to pass to Sharepoint"; the testers' tenant is GCC on a `.sharepoint.com` address, so the commercial endpoints, no setting needed): the cloud comes from the site's address when `GRAPH_CLOUD` is unset; a user can paste an address on another site and the app opens it on its own host if that site was granted, else says plainly to ask the administrator (never the same path on the default site); a "Copy link" sharing link goes through Microsoft's sharing-link call, and if the app's Sites.Selected permission refuses it the user is asked for the browser address; every SharePoint error and each connection test's report goes to `_app_log/sharepoint-YYYY-MM.log` in the app's SharePoint folder (`GEOTECH_SHAREPOINT_LOG=0` off), closed to the agent and downloadable from the sidebar diagnostics with only the user's own entries. Release note: `module_work/ta35/CHECK.md`. First live check with the testers: `module_work/LIVE_TEST_QUEUE.md` check 13.
- GEC-13 lookups cite their printed equations (BACKLOG Done, 2026-11-06; refs pin f62a1d1): the backfill suitability number is Eq. 4-3, the area replacement ratio Eq. 5-1, the groutability ratio Eq. 8-2, the long-term geosynthetic strength GEC-11 Eq. 3-12, and the deep-mixing composite modulus is marked as not printed in GEC-13. **Release note:** one answer moves: a groutability ratio between 24 and 25 now reads "Feasible", as GEC-13 p. 8-29 prints (N > 24). The suitability number itself does not change; its rating uses Table 4-7's own words (above 50 "Unsuitable", was "Not suitable"), and between 40 and 50, where Table 4-7 prints no band, it gives no rating and says so (was "Poor"). `module_work/ref60/CHECK.md`.

**0a. 5.34.0 RELEASED 2026-11-04 with planlens 0.14.0 and geotech-references 1.5.0: everything on master since 5.33.0 (through d57f28d).** The owner held the release from 2026-10-14; it publishes on the owner's word, refs first, then planlens, then the app (each tag push publishes to PyPI).
- **Branches** (each cut from the head of main or master, one commit on top):
  - geotech-references `release/1.5.0` (25f7d7f, on d8183a1): the version only;
  - planlens `release/0.14.0` (d10d078, on 82660d3): the version, `__version__` and the CHANGELOG's 0.14.0 section;
  - app `release/5.34.0` (on d57f28d): the version, pins `planlens>=0.14` and `geotech-references>=1.5.0`, the submodule at 25f7d7f, the numpy 1.26 CI job's planlens pin, and these docs.
- **Release gate on the exact tree:** the app's six chunks 18,172 passed / 117 skipped / 2 xfailed / 0 failed, and with the reference library's suite (in the app's testpaths) 24,641 passed; planlens 1,783 passed. A fresh venv with the main `.venv`'s exact package versions (Python 3.14.5; pytest-xdist added) and the three release trees installed editable; each chunk judged on pytest's own exit code, four workers (two for report_ingest, when free memory was under 3 GB).

  | Chunk | Result |
  |---|---|
  | geotech-references 1.5.0 (its own tree) | 6,469 passed, 2 xfailed |
  | planlens 0.14.0 (its own tree) | 1,783 passed |
  | analysis modules, part 1 (bearing_capacity to ground_improvement, 13 folders) | 2,712 |
  | analysis modules, part 2 (slope_stability to geo_project, 19 folders) | 3,176, 54 skipped (36 are OpenSees: openseespy has no working build on this Python 3.14, as in the main `.venv`) |
  | fem2d | 583 |
  | report_ingest + its harness + foundry harness | 4,702, 33 skipped |
  | webapp + validation_examples | 1,398, 11 skipped, 2 xfailed (one docs test first failed on closed backlog IDs this item cited; fixed, and the three docs test files re-run green) |
  | funhouse_agent | 5,601, 19 skipped |

- **Wheels and sdists** in `../foundry_handoff/release_2026-11/`: built from the release trees with `python -m build`. Metadata: the app requires `planlens>=0.14`, `geotech-references>=1.5.0`, `numpy>=1.26`, `PyMuPDF>=1.27.2,<2`, `deepagents<0.7.25`, `langchain<1.4.6`, `openai<3.29.1` and `openpyxl>=3.1`; planlens `numpy>=1.26`, `PyMuPDF>=1.27.2`; the library `numpy>=1.24`. No `module_work`, `raw/`, `tinyapps/reference`, test or truth file in any wheel (planlens' `dxf/truth.py` is its ground-truth extractor code and the eval sample PDF is package data, both shipped before), and no planlens or library code inside the app wheel. Present: the report-ingest EXAMPLE files and review checklist, the DIGGS 2.6 schema zip with its licence and README, the eval question set, samples and worked examples, the upload component; the library's 33 figure catalogs and its text JSON. (`calc_package/templates` in the package data matches nothing, as in 5.33.0.) The library's sdist carries its `tests/`, as before. The three wheels installed together in a clean venv (`--find-links`, dependencies held to the main `.venv`'s versions), `pip check` clean, and 29 packages imported from site-packages, versions 5.34.0 / 0.14.0 / 1.5.0.
- **Release notes: what changes a calculation's answer, by module.** Compare a 5.33.0 answer only with these in hand. Each is in BACKLOG_DONE.md with its evidence (found there by its commit); item 35 of "Start here" above has the train's record.
  - **bearing_capacity:** Meyerhof takes its own N_gamma, (Nq-1)tan(1.4 phi) (UFC Table 5-6), and shape factors of 1 under an inclined load (F18 1,465.9 -> 1,241.0 kPa; f40d04b); the factors combine as printed, so at phi = 0 the Vesic set adds d_c and undrained footings with shape and depth factors come out about 3-4 % lower (UFC Example B-3 9.969 ft against 9.97 printed, was 9.751; eval key BC-2 582 -> 567 kPa); Meyerhof refuses a base tilt or ground slope (5108f75); new `q_allowable_net` and `allowable_load` (f40d04b).
  - **settlement:** Westergaard read off UFS Fig. 4-10 (11-81 % higher); Schmertmann needs `gamma_soil` (+4.7 % at L/B 1.5); secondary compression with cv +8-12 % (t1 = t90); elastic and time-rate under 1 %; an underconsolidated layer warns (a79cdb0). A strip with no L is a strip (2:1 stress at z = 2B: 16.7 -> 50 kPa), a rectangle with no L is refused, a square or circle takes L = B; secondary compression uses each layer's own e0 (0814d9f).
  - **axial_pile (driven piles), GEC-12 ch 7 as printed:** Nordlund, Tomlinson, the beta toe, the open-pipe plug and uplift on the full shaft step by step (deep sand H-pile 5,834.5 -> 5,717.2 kN, its uplift 4,015 -> 5,236; concrete pile in clay 1,326.8 -> 2,043.2; beta with a clay toe 1,222.7 -> 753.7; V-002 +13-14 %; 079fed0); the alpha method off Fig. 7-17 with no cap (alpha to about 1.1; API's 1.0 only on the API route; bc2f038); sand toe +1-3.5 % from 29 deg (Fig. 7-16(b) re-read; GEC-12 North Abutment 359.7 -> 371.5 kips); the beta method has no phi-based default (Table 7-9 by soil type, or the layer's own beta and Nt); factored resistances by Table 7-1 (c513235).
  - **drilled_shaft, GEC-10 (2018):** sand side by Eq. 10-13 by default (DS-2's tool result 7,328 -> 4,196 kN; clay unchanged; O'Neill-Reese an option; ec552a3); clay alpha Eq. 10-18, N*c 9.0, rock side C 1.0 by default, rock base 2.5 qu only where p. 10-22's conditions are stated (V-007 9,414 -> 14,690 kN; dfda54f); a cohesionless tip with no N60 refused (was a silent 15; 0814d9f); the sand base takes N60 averaged over two diameters below it, rock qu held to `concrete_fc` when given, `rock_side_method='caving'` (Example 10-5 853.0 against 855.4 kips; 6b2e7a4); su averaged over 2B and a Hoek-Brown base for fractured rock (c95a0e1, c95a0e1).
  - **downdrag:** every cited neutral-plane method side by side, UFC's the headline, final effective stress by default (a282d48); the pile's own weight out of the load curve by default (DD-1 174.9 -> 150.8 kN; `include_pile_weight=True` for the old basis; 867b08b); settlement below the toe to UFC's 10 % of the bearing stress (floating pile head 305.4 -> 353.9 mm; 857b668); alpha by su when not given (UFC B-5.4 223.5 -> 132.9 kips; 6a6f11f); sand beta and toe Nt from GEC-12 Table 7-9 by `soil_class` (the manual's example: neutral plane 5.99 -> 17.66 m, drag 226 -> 1,308 kN; ce26958).
  - **lateral_pile:** sand p-y curves by GEC-9 Eqs. A-22 to A-24 (about twice the old table at 30 deg near the surface), API C1-C3 and k off the API figures, sand k by relative density, a partly fixed head now between free and fixed (its spring's sign was reversed): 1,506 of 5,475 recorded results moved, all sand; the manual's example 4.56 -> 4.04 mm (f3553a9); weak-rock p-y as GEC-9 App. A.5 prints (working loads much stiffer, near ultimate softer; 90e23b7), weak rock needs qu and Es with RQD 100 by default (73da7b4); stiff clay below the water table (static at 900 kN 25.2 -> 40.8 mm; a96663e).
  - **pile_group:** Converse-Labarre removed; efficiency from GEC-12 / UFC Table 6-26 by soil type and spacing (none without `soil_type`); block failure factored; p-multipliers at 5b 0.85 and 0.70 for rows 2 and 3 (were 1.0); the equivalent footing to UFC's 10 % depth (worked raft 16.45 -> 13.74 mm); a p-multiplier below 3b refused (c1b8bbc); a group with a pile in tension reports the group uplift check (d14c0e2).
  - **wave_equation:** permanent set as GEC-12 prints it, so friction piles no longer read refusal (Example 6: 92.6 blows/ft against GRLWEAP's 94, was 305); built-in hammers from App. C (blow counts 6-20 % lower, about half for the ICE I-30V2); pile-top stress without the helmet's inertia (52.5 -> 28.8 ksi); refusal 120 blows/ft (896d1a7); a drivability study runs one pile embedded to each depth, so with no pile length the shallow depths carry the pile above grade and their stresses rise (96a2ff9).
  - **retaining_walls:** Coulomb takes delta = beta, so a level backfill matches Rankine (the owner's 2026-07-14 wall: sliding 1.339 -> 0.945, overturning 1.996, fails 2.0); a cohesive backfill gets its tension zone; bearing on B - 2e (bearing factor up to +33 %); concrete 23.6 kN/m3; FS sliding 2.0 when passive counts; a descending backfill refused (cb48c8c); sliding by friction or adhesion, not both (H 5 m, phi' 28, c' 10: 1.507 -> 1.233; 4c54e9f); a live surcharge over the heel no longer resists (2844f1e); bearing on the larger edge pressure and "not checked" with no `q_allowable` (718cd62); an optional water table with the base uplift of EM 1110-2-2104 (3f8f835); shear-key passive (ca988da). Earth pressure coefficients: the passive log-spiral default is UFC 3-220-20 by geometry (phi' 30/35/40 at 0.66 phi': 5.36/8.18/13.1 -> 5.06/7.53/11.7; a left-out delta is 0.66 phi'; Caquot-Kerisel 1948 a named option, re-read off the Caltrans figure; fa9ec9b, 700fa15); at-rest K0 behind a slope always EN 1997-1 Eq. 9.2 (1558af9).
  - **MSE walls:** sliding on the lowest geosynthetic sheet (11-30 % lower where it governs); a level with less than 3 ft beyond the failure surface fails; a clay foundation's undrained strength counts in full; the unprinted 2/3 c L term gone (f2a1f69); a left-out retained or foundation friction angle is 30 deg (p. 4-10): sliding and overturning about 15 % lower over a 34 deg fill, 10 of 452 recorded calls pass -> fail (f3cd92a); traffic out of pullout and out of the ASD resisting forces (022a1b7, 8960ecd); the coverage ratio Rc = b/Sh in pullout and rupture (a 9 m strip wall's pullout FOS about a tenth of before; 7b4c0f4); bar-mat and strip Tal, geosynthetic rupture at 1.5 and per-level reinforcement as GEC-11 prints them (1729383, 287cdf1, 31bd1bf, a689ec6).
  - **soe and sheet_pile:** braced loads by the Caltrans hinge method (first support +15-25 %, lowest -30-40 %); clay envelopes higher (stiff clay 0.4 gamma H); the soe cantilever moment at FS 1.0 (12-53 % lower); sections at 0.60 Fy (+10 % section); sheet_pile embedment 1.2 D0 by default (+20 %); Terzaghi heave with UFC's chart beside it (f86c4bd); anchor bond from GEC-4 Table 6 read as kN/m, not kPa (at 445 kN, medium sand 13.03 -> 6.14 m, rock 3.00 -> 4.50-8.90 m; b1613af); anchor test load 1.33 DL and the bar table per GEC-4 Table 9 (da0b2a3); the anchored-wall moment no longer overstated (V-013 596.5 -> 187.1 kN-m/m; 7e8e273).
  - **slope_stability (as of 2026-10-18; frozen since for GM-140):** a surface below the ground in more than one stretch refused and circles above the toe searched (123cfab); surfaces with m_alpha < 0.2 on a frictional slice left out (654b144); noncircular surfaces sliced at corners and crossings (ca8f08f); circles sliced at every edge, each slice on its chord (553aa58); the rigorous methods' search refines on their own FOS (V-043 Spencer 1.2375 -> 1.1859; 04f4759); supports per Wright & Duncan (729c511, 5fc997e); toe circles and Taylor's deep circle found (dbf1cdd, 014ef09).
  - **fem2d:** movements from the end of the gravity stage (4bf710c); excavations dewatered by default with seepage under the wall (with a water table above the excavation level the wall forces are larger; GM-73); strength reduction no longer mesh-dependent (c21c8cc) and over deep undrained soil slightly higher and about twice as slow (ad114e9); 500 iterations a stage step; straight T6 walls (429f62d).
  - **ground_improvement:** wick drains ignore smear unless a smear ratio is given (90 % in half a year: 1.30 -> 1.46 m); vibro-compaction spacing off Fig. 4-34 (e1c20e6); stone columns on GEC-13's printed Priebe curve only (062f140).
  - **seismic_geotech:** rd 0.5 below 30 m; site class by GEC-12 Table 7-26 (some sites one class softer); site factors and Fpga need a PGA; Idriss & Boulanger residual strength off Fig. 7-78 (lower); seismic passive with wall friction 0.4 phi (KPE about 25 % lower); Hardin's Gmax A = 625 (0b36ef7); M-O PAE carries (1 - kv) (a8b00b6); liquefaction requires the water table and PGA on every route (6319836).
  - **soil correlations:** Peck's phi from N1,60 (up to 2.4 deg lower), Meyerhof's from GEC-5's table (4-7 deg higher), relative density from SPT fixed (it returned 100 %) (d7058fa).
  - **pavement_design:** UFC thicknesses off Figs E-1 and F-1 as printed (flexible -6 % to +48 %, slabs -19 % to +88 %), 4 in subbase and 6 in slab minimums, AASHTO treated bases and composite k off Fig. 3.3; eval keys PAV-4 to PAV-6 moved (af5a33e); frost groups only for soils UFC Table 19-2 prints, clayey gravel with 20 % fines or less F3 (refs 22c11ed); UFC mixed traffic, AASHTO rigid over shallow bedrock and the Table 3.1 performance period move slightly (58c389d).
  - **site response and signals:** OpenSees site response applies its input at the column's real weight (surface PGA was about 1e-6 g), clay ru by the SSPquadUP formula (7bc7540); spectra by Nigam-Jennings at every period and RotD in the time domain (up to 10.7 % and -11.6 to +29.1 % off before; 56d65e3, 9752270).
  - **reliability:** a uniform variable is built on its bounds (it was built as mean a, std b; 9f1b394).
  - **reference tools the agent calls (geotech-references 1.5.0):** GEC-8's p-multipliers by soil type, CFA allowable at FS 2.5 unless p. 138's four conditions hold, grout factor 1.15 minimum (refs f9471b8); EN 1997-2's G.1 DPH density index, F.3 Burland and Burbridge and seven table lookups as printed; AASHTO and UFC pavement charts off the print (a composite k of 3,026 pci now 487).
  - **report_ingest:** a value with no printed unit stays unknown and stays out of DIGGS (a78c275); depths above the ground surface set to 0 within 0.10 m and refused beyond (ea93a2d); the owner's narrative rulings (soil corrosion's worst sample governs, landslide only when named; b4acd8f).
- **Behaviour and install changes (not a number):**
  - report ingest is model-first by default for every stage, today's readers one setting away per stage (`logs=off,lab=off`); a page group never read is a failed item re-asked on resume; prompts mf-8 and ri73-4, so a live run starts in a fresh folder; on the geotech page it stays behind `GEOTECH_REPORT_INGEST` (off);
  - `GEOTECH_FIGURE_HELPER`, `GEOTECH_PLOT_CODE` and `GEOTECH_JUDGMENT_CALLS` on by default (since 2026-10-20; a judgment-calls turn spends one references look-up); new switches all off; the review switches stay off;
  - a misspelt input name is refused with a did-you-mean; results list `choices`, and `defaults_used` only the inputs a run used; calc packages run their tool's own code path and carry calculation tables in 13 modules;
  - Foundry: installs on numpy 1.26 (`constraints/foundry-numpy126.txt`, no `pynite` tool there; 74c9cc3), no workspace glue, the references helper on Sonnet 5.5 and the rest on Opus 5.5, rate-limit 400s wait and retry, conversations saved to a workspace dataset (INF-38 open until the restore round trip in the published app);
  - install: agent-stack caps deepagents <0.7.25, langchain <1.4.6, openai <3.29.1 (0140df2); python-ags4 an extra and openpyxl core (beside CfA's pandas 3.0.2); pyrotd dropped; PyMuPDF `>=1.27.2,<2` (INF-18 lifts the cap after a host probe). Install guide §11.
- **geotech-references 1.5.0:** the manuals checked against the print and rebuilt where they had drifted (GEC-4 to GEC-10, GEC-12 Vol I and ch 7, GEC-13, Micropile; GEC-5, GEC-8 and EN 1997-2 with its annexes rewritten; GEC-12 ch 7 lost made-up downdrag and lateral squeeze equations and prints Table 7-18's 0.005 distortion limit); GEC-12 ch 7's digitised charts read off the print (C_F falls with phi, as printed); figure catalogs read from caption blocks; every section level searchable and fetchable, results shaped to the size limit, ranking by how many query words a section holds when none holds all, and search databases per installed copy (refs 51af51b). Suite 6,469 passed / 2 xfailed.
- **planlens 0.14.0** (CHANGELOG 0.14.0): a tick-flanked depth scale reads as depths (planlens 8c359fa); a letterhead's telephone line never names a hole (planlens 04ca6a7); numpy floor 1.26 (74c9cc3); tables read the same on PyMuPDF 2.0 and the planlens cap is lifted (floor 1.27.2), PDF reads one at a time across threads (planlens 0fae05d); `log_grid` sample boxes and whole long logs (PL-24); distinct stroke-drawn sheets not duplicates (planlens 89711ea); `plan_symbols` / `plan_grid` (planlens c4330b9); markup-only dividers and stable markup ids (planlens c4330b9, planlens c4330b9). Suite 1,783 passed / 0 skipped / 0 failed.
- **Not verified live:** the Foundry restore round trip (INF-38); the model-first ingest beyond MF-3's three truthed reports; the Tiny Apps tester deploy (waits on CfA, OWN-44); the cluster install (§11 row).

**0b. 5.33.0 RELEASED 2026-10-09 with planlens 0.13.0 and geotech-references 1.4.1.** The owner's word, 2026-10-09: "go ahead and release when you're done".
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
- **Master after 5.33.0: the next train, released as 5.34.0 (item 0a).** The owner: "that's stuff that can just roll into the next release".
  - **a282d48, downdrag**, reworked 2026-10-14 after the owner's answer. Every cited neutral-plane method (UFC 3-220-20, GEC-12, CGPR #56) is computed on every run and tabulated side by side, each named as its reference names it and with its source; the methods table, with sections and pages, is in `downdrag/DESIGN.md`. The headline stays UFC's procedure; the spread over the methods is stated in one plain sentence. Default skin-friction stress is final effective stress. UFC App. B-5.5 and V-004 are reproduced.
    - The owner's direction: "Downdrag has many methods and it should be open to multiple assumptions", and on 2026-10-14: "presenting all of them is fine".
  - **4bf710c, fem2d.** Per-stage delta, cumulative and since-reference movements (`reset_displacements_after`, default after the gravity stage). Phasing decides self-weight. Three latent staged-analysis bugs fixed.
    - The owner's direction: "Construction phasing will determine output."
    - Stage 0 as the gravity stage: confirmed by the owner, 2026-10-14. θ=0.5 ringing at drained boundaries fixed 2026-10-20 (Rannacher start-up). `n_steps` is honoured since 2026-10-14.
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
- **3d. The owner's checklist session** (done 2026-10-26 on the Report Questions Desk and folded the same day: the checklist is draft 2, 17 items).
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
  - **Owner's judgment:** given 2026-10-14 (item 0b above).
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

**Then (2026-10-14 to 2026-10-21; the order of work now is "Start here" above and BACKLOG §0): core improvements while CfA sets up Tiny Apps.** CfA says the SharePoint connection and the per-user Prompter keys (one per user) are a few weeks out, so the Tiny Apps production items (OWN-30, OWN-44, TA-03; the rest merged into OWN-44 on 2026-11-04) moved to Next until they arrive, and on 2026-10-21 the owner deferred them to Later; the FIPS and metered-route questions are answered.
- **Done on master 2026-10-14 (released in 5.34.0, 2026-11-04):** see CLAUDE.md's state block and BACKLOG "Done since the last sweep" for every item and commit. In short:
  - unsafe-side module answers: downdrag (every method side by side, final stress), the slope search (sand slopes, toe circles, reinforcement), fem2d (the slope mesher's soil-filled notch, the failure criterion, staged `n_steps`, straight T6 excavation walls, a staged excavation that unloads in lifts), SOE free earth with no solution, the anchored sheet-pile moment, Terzaghi factors;
  - Ensoft-style calculation tables in 13 modules' calc packages;
  - Document Review: markups keep correcting and return the found box, geotech tools off the page, turned lettering, cut lines, sideways boxes, zoom and tiles on images; planlens: distinct sheets no longer called duplicates (reached the app with planlens 0.14.0 in 5.34.0);
  - long turns keep their work and wait out busy refusals; a model request gives up after its time limit (it waited 3.5+ hours on Funhouse); the record answers cost and cause; DIGGS writer and reader fixes;
  - CI: green again on 2026-10-15 for the first time in months (the five old failures, the workflow, and an OpenSees test that hung every run).
  - 2026-10-14/15 continued: OpenSees site response fixed (seven defects, then two more found by pinning it to the OpenSees wiki's own example); fem2d excavations dewatered by default (larger wall forces below the water table: a release note); slope supports as Wright & Duncan apply them, GLE / Spencer with large anchor loads, and no silent unreinforced fallback; log depths read off the drawn scale (planlens, reached the app with planlens 0.14.0 in 5.34.0) and helpers' records handed to write_diggs as files; cited pages checked against pages read; earlier turns' files and page readings reach the helpers (`kept_readings`, switched OFF on the review page); the Funhouse launcher's Prompter-key cell and lost-app helpers; the drift canary's latest report is read at the start of every session (BACKLOG "How every session uses this file"); slope and fem2d model-width checks against Taylor's problem, and a slope search that finds the deep base circle; the excavation result gives the depth of its peak moment; marks and find_like on images in their own pixels (VIS-26); the reviewer helper's model-call budget; pystrata equivalent-linear runs that converge; plan-view axes named from the stated coordinate system; a fresh resolve with pandas 3 passes, and streamlit is tested below 1.67; fem2d strength reduction over deep undrained soil reaches Taylor's deep circle by default (release note: a slightly higher FOS, about twice the run time, for those runs only).
  - The owner's 2026-10-14 field session (a report's borings to DIGGS on Funhouse) is filed and read in full: `module_work/field_feedback/2026-10-14_report-to-diggs_v5.33.0/`.
  - Its continuation on 2026-10-15 (kriging and a joint-field Monte Carlo of rock under a footprint, then the plots) is filed at `module_work/field_feedback/2026-10-15_rock-excavation-kriging-mc_v5.33.0/`: the crash that ended it and the chat's unreadable chart text are fixed, plans are drawn to scale with named points and filled outlines, and the conditional simulation the app could not run is built (`gstools.conditional_simulation`, conditioning by kriging, checked against conditioning by its definition; its follow-ups done 2026-10-21). `calculate` counts, adds up and averages lists and has the normal distribution (the session counted footprint fractions by eye); the helper that draws the figures can look at them, and a chart written by hand is flagged. The owner chose a figure helper (Tufte checklist plus worked examples): built and measured behind `GEOTECH_FIGURE_HELPER` (OFF then; the owner ruled it on everywhere, on by default since 2026-10-20). Plans are now drawn as drawings (scale bar, north arrow, property line; the owner's comments on the figures page). The owner then approved plotting code run in a locked box: built behind `GEOTECH_PLOT_CODE` (OFF then, on by default since 2026-10-20) for the figure and `calc` helpers (f0bc000; follow-up INF-16), and named the figures engineers make most (built 2026-10-16 with the owner's conventions: elevation by default, kPa and kN, the wall drawn). Downdrag now leaves the pile's own weight out of the load curve by default, as the references do (867b08b; the calc package takes a steel section since).
  - The night of 2026-10-15/16 (the owner asleep: "keep working through the backlog on things you can handle yourself"; every item in BACKLOG "Done" with its commit): plotting code in a locked box measured live twice and fixed after each full read (waves w8 / w8b; `module_work/live_smoke/runs/w8*/TRACE_REVIEW.md`; the switch now brings the figure helper; a layout check measured from the drawing); downdrag leaves the pile's own weight out by default, its calc package takes a steel section, and it warns when nothing below the toe can settle; a wall check not run says "not checked" and failed checks come first, bearing on the larger edge pressure (release note); no placeholder FOS reported as a number; the Backlog Desk takes several answers per card; a look waits while a figure is drawn and names no zoom tool the caller lacks; large JSON results open with a key map; `write_xlsx` refuses unknown sheet keys; method docs come back whole; the Jibson range (GM-17, part); staged FEM at 500 iterations a step (release note); `html_to_pdf` says what landed on each page; the record keeps an errored turn's tokens, labels probe calls and lists each turn's tools (EV-20, part); the step cap in plain words and, behind `GEOTECH_STEP_CAP_ANSWER` (OFF), an answer from what was gathered; stale docs. The figure conventions and worked code examples are committed (the owner's answers: elevation by default, kPa and kN, past pressure as points or a design line by stage, the wall drawn). The cited-documents check is built behind GEOTECH_CITED_DOCUMENTS (OFF; follow-ups EV-23). Downdrag counts settlement below a pile's toe by the references' own rule (UFC 3-220-20's 10 % of the bearing stress by default, GEC-12's 10 % of the effective stress as the option; release note: more toe settlement where soft soil continues below the toe; the owner kept both with a warning when they differ, built 2026-10-20). Wave w9 (flows F56-F60, five figures the tools do not name, $1.54) ran and was read in full (`module_work/live_smoke/runs/w9-figures/TRACE_REVIEW.md`). Flows F61 / F61j ran live as wave w10 ($0.82; `module_work/live_smoke/runs/w10-judgment/TRACE_REVIEW.md`): with the switch off the answer named its search choice, why, the alternative and what would change it; with it on the turn ran out of time in two long Spencer grid searches (since fixed: a time budget), so that arm was unmeasured (measured since in wave w12). Re-run 2026-10-17 as wave w11 ($0.35; `module_work/live_smoke/runs/w11-judgment/TRACE_REVIEW.md`): both arms found 0.935 in the clay by default and named their calls (3 of 3 off, 4 of 4 on); with the switch on the user also got a short list of judgment calls (another reading of the layer, a target FOS, a cohesion check, what would change the answer); neither said why it chose Spencer or cited a reference (the judgment-calls switch answers it, on since 2026-10-20). In flight now: nothing. Owner 2026-10-18: no more slope_stability fixes; every slope item waits for one integral review (GM-140). Wave w12 (16 runs, $5.05; ledger $62.93 of $80) was read in full (`module_work/live_smoke/runs/w12-figures-judgment/TRACE_REVIEW.md`): one figure card per helper holds, but a second helper asked to redraw started over (fixed since, below); the judgment-call switch is not ready to turn on (same engineering at nearly twice the cost, no reference cited, assumptions passed on as facts; the judgment-calls switch, on since 2026-10-20); the A-line drawn from memory again (fixed since); the wall's heel surcharge counted as resisting (fixed since). Tool descriptions and the defaults a result reports now say what the code does, checked by running 165 of the 189 analysis methods. Weak-rock p-y curves follow Reese (1997) as printed in GEC 9 App. A.5 (90e23b7). A figure is one thing across a turn, whoever draws it, with its code kept beside it (e593029; behind the figure switches; a live wave must confirm). Standard chart lines (the plasticity chart, the AASHTO chart, sieve sizes) are drawn from the references (refs 6ad9fbb + a5ab30e). A live surcharge over a cantilever wall's heel no longer resists sliding or overturning (2844f1e; a permanent one stays out too, owner 2026-10-20; the wording merged e86c8fa). The MSE wall's ASD path does the same (8960ecd). 2026-10-19: judgment calls cite their source and calc packages keep given and assumed apart (8b9abba); wave w13 ($3.10, `module_work/live_smoke/runs/w13-judgment/TRACE_REVIEW.md`) kept the switch OFF: no answer in either arm made up a percentage and ON cost 0.92x OFF, but the ON answers never consulted the references and one printed c = 0 as given; round 2 (006ed10) and wave w13b ($2.55, ON only) got one references consult in three of four answers and real citations for 12 of 15 calls, still NO: one answer never consulted and one printed c = 0 as given (the turn-end check and the given-inputs fix, both merged; then the delegation fills `given`, merged; the rerun w14-judgment ($1.79) said NO on cost only: the consult works, citations reach the PDFs, Given vs Assumed is right; the owner then ruled, 2026-10-20, that it ships ON regardless of the cost test; on by default since 2026-10-20); filed ten follow-ups (most now done) and one merged into EV-23; the NHI manual it found unindexed was indexed 2026-10-20. PyMuPDF 2.0 (released 2026-10-19) broke six report_ingest tests in CI: capped below 2 (3cd8fc1, planlens eeb278b); planlens main now reads 2.0 correctly (1e8e328), and the app stays below 2 until planlens 0.14 is released and the hosts pass an import probe (INF-18). Cloud sessions (the owner starts them from `dev/cloud_prompts/`): driven piles merged (4aafc4d, reviewed and fixed); six more ran (DM 7.1 edition, reference data, module inputs, figure tools, shoring, FEM wall friction; all finished 2026-10-19, all merged), each on its own `cloud/` branch. In flight on 2026-10-20 (helpers in worktrees, up to four at once, owner 2026-10-19): faster CI runs (a side branch, tried on GitHub); planlens main 3b08f11 has plan locations, markup-only dividers and stable markup ids from the cloud batch (unreleased; app wiring PL-27); the backlog was tidied 2026-10-20 (closed items now in `BACKLOG_DONE.md`, the Later items triaged, the nine old items dropped by the owner 2026-10-20); Figure tools merged (581a49f); shoring passive coefficients and calc packages merged; CI's matplotlib 3.11 failure fixed (a haloed note measured as drawn); the judgment-call turn-end check, hidden unrunnable tools and the calc helper's page tool merged; PDF figures move whole to the next page (no more 0.1-2 in figures); given vs assumed inputs merged (inputs left out say what they stand for; Given Inputs only what the call names as given); the four agent fixes merged (the calc helper's call budget, one page-number convention, a list for `calculate`, pyplot warmed before parallel tools). Judgment calls under the switch: an answer still unsourced after a consult goes back once, and the consult is held to 6 calls (switch OFF unchanged); the harness's detectors corrected and a check for a number no tool gave added. Next: GM-196 (the calc-package numbers item, the post-liquefaction zones, the helpers' records as files and the MSE F* at the top done 2026-10-20); owner OWN-73 (the wall-sliding default; the stone-column default answered and built 2026-10-21), the rock-socket side resistance factor (answered 2026-10-21: Table 8-4's 0.50) and the app split's names and route (OWN-44) and the testers' notice (the release before the split's file moves answered 2026-10-21: release first, timing the owner's) (where the shared code lives and whether Document Review is private, decided 2026-10-21: a thin public core, the document-tool wrappers into planlens, Document Review private as source in the GHE repo; the plan rewritten 2026-10-21, with the planlens binding new (done 2026-11-09); the private copy on a private GitHub repository, the owner's choice) (plan `module_work/APP_SPLIT_PLAN.md`, merged 2026-10-20; its first five steps, since done, needed no decision); the passive coefficients from UFC by geometry merged (owner OWN-79 on the step between equation and charts); at-rest K0 behind a slope to EN 1997-1 Eq 9.2 always (merged 2026-10-20); GeotechStaffEngineer on Foundry is the owner's daily tool and comes first (owner 2026-10-20), Document Review and Tiny Apps lower; the second cloud batch is running (the calc-step citations, the planlens plan locations and the OpenSees laboratory curves, the reference-search synonyms, the fem2d consolidation start-up and the NHI Soils and Foundations manual merged; started 2026-10-20). Every tool's nested inputs (layers, supports, loads) now refuse a misspelt key with a did-you-mean (a misspelt key used to be dropped silently), and so do top-level names, with a reason where a tool deliberately does not take one. MSE walls take a sloping or broken backslope per GEC-11, pinned to its Examples E1 and E3; traffic is left out of pullout (release note: a lower pullout FOS near the top of a wall with a traffic surcharge). The figures page shows the conventions applied to the worked examples (v7).
- 2026-10-16 (every item in BACKLOG "Done" with its commit): the cited-documents check (OFF switch `GEOTECH_CITED_DOCUMENTS`); downdrag counts settlement below the toe by UFC 3-220-20's rule with GEC-12's as an option (the owner kept UFC as the default with a warning when they differ, built 2026-10-20); wave w9 (five figures the tools do not name) read in full and its fixes built (the chat shows the newest figure, a look checks content before lettering, the redraw cap holds); every tool refuses a misspelt input name, nested or top-level, with a did-you-mean, and gives a reason where it deliberately does not take one (a survey found planlens already does); MSE walls take a sloping or broken backslope per GEC-11, pinned to its Examples E1 and E3; `annotate_document` refuses misspelt remove keys, shows the model the markup fields and keeps its own signature; the slope search sets its automatic ranges from the slope's own geometry, labels an infinite-slope sliver for what it is and reports the lower surfaces it refused and where the governing surface lies (a drawdown pond-pressure error fixed on the way); behind `GEOTECH_JUDGMENT_CALLS` (OFF then, on by default since 2026-10-20) the agent puts every engineering judgment call in front of the user (owner: "User's judgement is the one that matters"; flows F61 / F61j; measurement EV-25 and wave w12); each module's result says what it chose (the method, given or default, the basis with its reference, the alternatives) for axial and lateral piles, settlement, seismic earth pressure, liquefaction and slopes, tool results list the defaults used, calc packages print a Method Choices table, and the citations were checked against the printed manuals (several were wrong); the judgment-call flows ran live (wave w10, read in full; follow-up merged into EV-04). Owner rulings the same day: engineering judgment about when an analysis is the wrong tool stays with the agent and its references, never hard-coded in a module; one helper at a time.
- 2026-10-17 (every item in BACKLOG "Done" with its commit): a slip surface below the ground in more than one stretch is refused, as Slide2 does, instead of being scored across the air between; that showed the default search never tried circles coming out on the face above the toe, so it now does (FHWA's "circles passing above the toe"), leaving out circles Slide2 also refuses (m_alpha below 0.2, reverse curvature); a 10 m cut with a weak clay over rock now reads 0.936 by default, where it read 1.16 on circles scored across air; reversed search windows are swapped with a warning (follow-up GM-126); a slope search the agent runs stops at 180 s (the agent may give up to 600 s), refines around the best points before it goes finer so a cut search still finds a narrow valley, and opens its warnings with "the reported FOS may be too high" when cut; each method's cost is in its description, and a Bishop screen for rigorous methods is an option; the judgment-call flows re-ran live (wave w11, read in full): both arms found the clay by default and finished in two to four minutes; every slope search leaves out surfaces Slide2 refuses (m_alpha below 0.2 on a slice with friction; slices with no friction are exempt, since the check does not bear on them there), a single surface the user gives is reported with a warning, and the critical surface is checked at twice the slices (follow-up GM-130); `choices` and `defaults_used` agree with each other and list only what the run used (a value typed equal to the default is "given"), the slope results name only the surface families scored, and the tool descriptions tell the agent the choices are there (follow-up GM-132); a noncircular slip surface is sliced at its corners and at layer and water-table crossings, so its FOS no longer depends on the number of slices (follow-ups GM-135, GM-136); CI was red for four pushes (two tests from this work) and is green again; every liquefaction route requires the water table and the PGA (a CPT run used to take 1.0 m and 0.25 g without saying so). 2026-10-18: circles are sliced at layer, water-table, surcharge and crack edges and every slice takes its chord, so a FOS no longer depends on the slice count and the pinned published checks land on their converged values (follow-up GM-137); the rigorous methods' circular search refines on their own FOS, not Bishop's, and finds a published slope's critical circle 1.6 % below Slide2's circular value, matching Slide2's own noncircular result (follow-up GM-139); a slip surface's depth below the model's base is measured on the surface, so a circle tangent to the base is accepted at any slice count. The owner then stopped slope fixes: nine in three days, each started by one case, is fitting narrow problems; the rest waits for one integral review (GM-140). Calc packages no longer use a fixed constant when the inputs given determine it (seismic Ka from phi, settlement L = B) and print the values the run used; then every calc package was moved onto its tool's own code path, inputs and parameter docs, so a package and its tool cannot drift again. The driven-pile module was audited against the whole of GEC-12 chapter 7 and now reads its charts, tables and limits as printed (the owner's rulings on its five judgment calls followed 2026-10-20). The reference library's section numbers were checked against every printed manual and 39 renumbered (geotech-references 4dc357f; a test keeps them so), and every section level became searchable and fetchable, so the librarian reaches GEC-12's pile methods in one search and a small fetch (d5f7f52); and every reference tool's result now fits the app's size limit by its shape, with a note on how to get the rest, instead of arriving cut mid-JSON (db84be2). The figure helper's layout check now judges by what is drawn (false reports in the w9 figures 22 of 36 -> 1 of 16), and `plot_data` places notes with optional leaders, checked by the same rule. Module defaults that were silent are now right or stated, and the wall and settlement calc packages print the values the analysis used.
- The night of 2026-10-20/21 (the owner away until the Foundry build c test at about 06:00 UTC 2026-10-21; every item in BACKLOG "Done" with its commit): downdrag in clay follows the manuals as printed (beta where a layer asks for it; alpha by su from UFC Sec. 6-5.4.2 when not given, reproducing UFC's drag example to 0.1 %), and the post-liquefaction result lists the zones that matter; a helper that stops keeps its unsaved records as files; the reference library's GEC-10 chapter 10 was rebuilt from the 2018 print and GEC-11 Table 3-6 and DM7.2 Eq. 6-9 corrected to the printed values; the earth-pressure coefficient tool refuses a left-out wall friction where the manual prints no default; drilled shafts take the Hoek-Brown base for fractured rock, the clay base's su averaged over 2B, and refuse rock-base inputs when the base is in soil; OpenSees site response reports ru properly, puts a node at the water table, takes permeability in m/s, and stabilizes pore pressure per element by the SSPquadUP formula so a clay between liquefied sands keeps its own ru (the result now carries the surface spectrum; a dense sand's PGA is a short spike that rises with mesh refinement); specialists in the Agent picker build calc packages, figures and PDFs and get SharePoint and email where the full agent has them; MSE strip and grid walls take the coverage ratio and geosynthetics their scale-effect factor as GEC-11 prints them (Examples E1 and E3 reproduced; strip pullout had been overstated about tenfold); then the bar mat's and strip's tensile capacity as printed (Example E4 reproduced) and spacing per level; and a geosynthetic's rupture held to GEC-11's factor (it had passed at two thirds of the manual's resistance); Example E1 reproduced through the module; bespoke reports can carry a package's calculation tables; report ingest keeps an unprinted unit unknown and converts many more units; the conversation record fills its gaps (steps, prompt and tools, vision asks, files written, restarts, restores, page crashes); the report library's page lookups fixed; every test run names tests skipped for a missing dependency; DIGGS positions named for the axis order they are written in; the MSE built-ins are GEC-11's printed strip and geogrids (strip walls carry two thirds of the old resistance); OpenSees reports a low-passed peak beside the raw one; OpenSees spectra by Nigam and Jennings at every period; MSE walls take a reinforcement per level; one response-spectrum code for the OpenSees and seismic-signals tools; RotD from time-domain responses; pyrotd dropped; DIGGS points with a height claim no datum; the reference library's README measured and current; MSE walls take the levels' depths; drilled shafts in rock at 0.50 only; the app-split plan rewritten for the owner's decisions; stone columns on the printed curve only; Claude through Foundry's own classes, behind a switch; MSE walls' bottom level to the base; the Report Questions Desk for the owner's mark-up (folded 2026-10-26); 2026-10-22: report ingest on the geotech page behind its switch and the helpers' budgets enforced, reviewed; the Nightly's two failures fixed; report ingest on the page's own chat model, so on Foundry, reviewed; logs without a fitted ruler keep their printed depths, reviewed; the log floor's index binding and N from drives, reviewed; the log floor's stacked sample/blows columns and printed-N check, reviewed five rounds; N past a hard seating drive, printed refusal marks and N60 kept out of N, reviewed five rounds; the narrative read once and bound pages read on one side only, reviewed twice; lab tests linked from their sample labels, reviewed twice; calc values kept by place and printouts kept whole, reviewed three rounds; every model request in the cost record, reviewed twice; QA entries ranked and continuation sheets folded, reviewed three rounds; pystra 2.x; copied test values replaced after a privacy audit; the label review's unlooked changes refused, reviewed; the vote measured, nothing to wire; zero-padded hole names matched; the label rule on resumed and reused reviews; the narrative fields diagnosed and the lenient list score fixed, reviewed three rounds; zero-padded hole names matched everywhere; the Foundry report-ingest run package (build d) sent to the owner; the measurement can run on Claude; DI text can be withheld cleanly; the SPT halt rule reads drive lengths; lab values by page and sample links by label; four-drive SPT records; lab link and mass-guard edge cases; Foundry wave 1 back and reviewed (the lab label-link switch wins); report ingest measured on Claude Sonnet and Opus locally (Sonnet for logs and narrative, Opus for lab) and page labels by section from a description of every page (50-56 harmful pages against 290 under the hand labels' own conventions, Haiku describing as good as Sonnet); a plain Opus handed each item's pages reads about as well as the whole pipeline; the owner chose to turn report ingest model-first, behind a switch (design in the report-ingest plan §8; the engine sends PDFs where the route takes them and a two-call check tells whether a route reads a PDF whole; pages grouped one call per log, lab window or calc printout, the hole named by the page describer; the answer schemas gain the log header's project, client and other fields, recovery as a length and a lab row's bottom elevation; under the switch the builders keep and flag a doubtful value and list an impossible one instead of dropping it; the model-first readers, one call per page group with cut-off replies asked again in halves, are built in report_ingest/model_first_read.py and wired into the graph behind the same switch (piece 8, 7f5f17d) and measured by score_on_cluster in either arm (piece 9, e092f16; the Foundry brief stayed a draft until the four fixes of item 31, merged 2026-10-26; the rate limits it waited on are settled and Haiku is committed in the workspace, 2026-10-27; the ground-surface check it also waited on is done); placement by printed words, report_ingest/locate.py, places 93 to 98 % of saved answers' values at a spot in every stage except Opus's held-out lab answers, 71 %: one blind sheet where Opus wrote the sample ids in a shape the sheet does not print, RI-99); the owner's report question list runs as report_ingest/question_list.py on a saved record, answer sheet in the owner's layout, not wired into the ingest yet, RI-101; the app installs and runs on Foundry's numpy 1.26 (the platform pins numpy below 2): a constraints file and a numpy-1.26 CI job, docs/FOUNDRY.md, the frame tool absent there; misspelt-input hints that never matched were fixed with a guard; the geostatistics module found a GSTools defect behind its all-NaN realization and works around it (other draws unchanged), and kriging draws a plan map of the mean and its standard deviation; geostatistics results list their choices; the app and planlens import PyMuPDF by its current name. The owner's order from 2026-10-21 (BACKLOG §0, the sections reordered and re-tiered to match): report ingest and GSE on Foundry first (Now: INF-40; OWN-11 kept parked by the owner until Foundry testing; Claude through Foundry's own classes built 2026-10-21, behind a switch), then the geotech calc modules, then Document Review and Tiny Apps last. In flight now (2026-10-22): nothing. Next for the owner: OWN-11's one-run check of report ingest on Foundry, which needs the next wheel (INF-40). New owner questions: OWN-83, OWN-84, OWN-85, OWN-86.
- Release notes carried by 5.34.0 (released 2026-11-04; item 0a above): report ingest keeps a lab water content or Atterberg limit over 100 instead of refusing it (the FHWA soils manual and DM 7.1 print such values) and lists a log recovery over 100 instead of silently cutting it to 100; on the Foundry SDK route a file or image the model cannot be sent is named in the tool result instead of vanishing, and a user turn carrying one is refused before sending; report ingest can label pages by whole sections from a short description of every page, behind GEOTECH_INGEST_RANGE_LABELS (off); the calc merge no longer writes the page's own values over right readings, and the calc scorer matches labels by their words and values only in units of the same kind (RI-74); report ingest unwraps an answer a model sends inside one extra key and no longer takes an answer left empty by dropping unknown keys; the corpus also finds P-numbered PDFs (RI-79); the lab floor reads range labels, stops a row at the next value's label, skips trial columns and reads no chemical value from header text; report ingest's lab floor no longer reads a temperature, a chart legend or a hydrometer reading headed 'C' as a cohesion, reads a water content printed as a ratio of masses, and links a sample name several samples share by depth; report ingest's log merge computes a four-drive SPT record's N from its 2nd and 3rd drives (FHWA Fig. 3-20's occasional fourth increment) and keeps a printed N it cannot confirm; report ingest's lab values carry their own page's boring and depth, a sample link follows its printed label, summary-table rows are linked to their holes, and a compaction sheet is named by its parts (new switch `GEOTECH_INGEST_LAB_FLOOR_PER_PAGE`, off; older code cannot load a record with linked summary rows); report ingest's log merge reads a record's own drive lengths for an SPT halt (a short fourth drive keeps the N) and leaves one row per halted N; `score_on_cluster(di_dir=False)` withholds Document Intelligence text (None still falls back to a `di/` folder); `score_on_cluster` can read on a LangChain chat model (`chat_model=`, `vision_chat_model=`), for measuring report ingest on Claude through Foundry's own model classes; the calc stage and the ingest run record the page-pictures and lab-link switches; report ingest treats a zero-padded hole name as the same hole wherever it matches holes by name (the narrative's hole count, the calc-subject link, the summary row against its lab sheet); report ingest's narrative run files keep the reader's whole answer and saved runs can be re-scored with no model call; the lenient narrative score no longer counts a piece of a matched truth item against a list answer (the strict score is unchanged); a resumed or reused label review applies the same rule to its saved changes as a fresh one; report ingest's check of the narrative's boring and test-pit names against the logs read treats a zero-padded name as the same hole ('B-01' and 'B-1'); report ingest's label review no longer applies a page-label change it made without looking at the page (on the saved runs those changes broke 105 labels and fixed 24); such a page keeps the vote's label and is listed as a label disagreement; every report-ingest QA entry carries a rank (open, changed or note) and the record counts them; listing them open first is behind `GEOTECH_INGEST_QA_BY_RANK`, off; a boring log's second sheet that planlens returned as a log of its own is folded into its hole; records written now carry fields an older wheel refuses to load; report ingest's cost record counts every model request, failed and retried ones by kind, rate-limit waits and reasoning tokens where the provider reports them, with one row per request in each run file; RESULTS prints them, the calls by step for each report (with what no step file accounts for) and the visual-scales counts; the vote stage runs without a prompter; nothing sent to the model changes; the calc reader keeps two values printed under one label at different places (two load cases, two footings) instead of keeping one, keeps a printout as one item up to 22 pages (a longer one in balanced parts) where it used to cut at every changed heading, and shows every scanned page as a picture; on the saved Foundry run the hand-truthed calc values found rise from 96 to 108 of 145 and the model calls fall from 1,134 to at most 296; every page of a calc window as a picture is behind `GEOTECH_INGEST_CALC_PAGE_IMAGES`, off; a lab test whose sheet prints the hole only inside the sample label ("BN-3 | 5.0 ft") is now linked to that logged boring, test pit or well, leading zeros folded, never across a depth or for a composite, and DIGGS places it there (on the saved records 558 of 1,064 lab tests linked, from 427, none lost); lab items over 12 pages are read in windows; a prompt paragraph for the same link is behind `GEOTECH_INGEST_LAB_LINK_FROM_LABEL`, off; report ingest reads a report's narrative once over all its pieces (one call where there were up to six, the first answer kept and disagreements recorded); pages of a separately bound document that only triage found are read once, in that document, with the reader the main report chose, and any such page left unread is flagged; two new switches, both off: `counts_from_logs` (boring/pit/CPT counts from the logs read) and `narrative_skips_bound`; each ingest score row names the record it matched; the log floor gives N = 2nd + 3rd for three whole 6 in drives however hard a drive was (FHWA NHI-06-088 p. 3-51 defines N that way; the test only may be halted past 50), and no N only where the record ends in a halt; a printed refusal mark (R, REF, 50+, N=50/3") is kept as printed and is never read as an exact N or as a recovery; an N60 or (N1)60 value goes to the sample's note, never to N; the log floor reads a column that holds both the sample numbers and the blows by its printed lines (number, up to three drives, then what the header names), keeps WOH where printed, reads a stated 6-inch increment in the header, checks a printed N against the drives (a mismatch is kept beside it with a QA row) and never takes an N under 50 on a record that ends in a halt, from the floor or the model (FHWA NHI-06-088 p. 3-51); measured floor scores unchanged on the blind set (139 of 214), open set 265 -> 266 of 307; the log floor ties index values to the sample clearly nearest (else leaves them unbound with a QA row), catches the grid's misplaced values ("#200" as LL, sample numbers as RQD and the like) and computes N from the printed drives for a standard SPT, per FHWA NHI-06-088 p. 3-51, saying so (never after WOH); measured floor scores rise (open set 245 -> 265 of 307, blind 138 -> 139 of 214); the pystra reliability tools run on pystra 1.x and 2.x (FORM results agree to about 1e-15) and a uniform random variable is built on its bounds a, b (it was read as mean a, standard deviation b, so earlier uniform results were wrong); a boring log whose depth ruler can't be fitted keeps the depths it prints (a rising depth column, sample intervals with a unit, description intervals that chain, the header's groundwater and total depth), each marked with how it was read, and a QA entry lists the depths to check against the printed log; elevations, dates, years, blow counts and thicknesses are never read as depths; logs whose ruler fits are unchanged; report ingest runs on the page's own chat model (on Foundry, GPT or Claude), behind `GEOTECH_REPORT_INGEST`; reader calls get the model's full output room (32,000 tokens; the Funhouse engines' default rose from 16,000 to 32,000), and a reply cut at that limit is re-asked or listed in the QA log, never taken as an empty reading; when the 1,000-call ceiling stops a run, every call made is counted and a step cut short is read again on the next ask; each report bound inside another gets its own checklist; report ingest on the geotech page behind `GEOTECH_REPORT_INGEST` (off by default, never on the Document Review page); with `GEOTECH_REVIEW_CHECKLIST` also on, the checklist's code checks run on each record and write `report.checklist.json`; the ingest and report-library helpers now hold their model-call budgets (the ingest 1,000 calls a report by default; asking again resumes) and the ingest writes into the conversation's folder; an expression given to `calculate`, a reliability `g_expression` or a pystra limit state that nests past 200 levels (a chain of more than about 200 terms) is refused with a plain message, on every Python version (Python 3.11 crashed in its compiler); add long lists with sum(); MSE walls without level_depths: the bottom level now carries the wall down to the base, as GEC-11 Fig. 4-14 does; where the height is not a whole number of spacings its Tmax rises and its FOS falls, and walls a whole number of spacings high no longer lose a level to rounding; stone columns read only GEC-13's printed Priebe curve (Fig. 5-21 curve (4), 'PRIEBE phi = 37.5 deg'), the result leading with the curve value and GEC-13's own words beside it; Priebe's formula and its 42.5 deg default are removed, and the old inputs are refused; on Foundry, Claude can run through Foundry's own model classes with no proxy or token (`GEOTECH_FOUNDRY_CLAUDE_SDK=1`, off by default); drilled-shaft side resistance in rock uses GEC-10 Table 8-4's 0.50; the 0.55 from the text of p. 10-21 is no longer offered; `mse_wall` takes `level_depths`, the levels' depths as GEC-11's examples print them; each level carries its contributory height (Vol I pp. 4-46, 4-47), with the horizontal stress averaged by the tributary trapezoid (default, as Examples E3 and E4) or at the midpoint; Example E4 now matches Table E4-7.4 within 0.2 % at all ten levels; without depths nothing changes; a DIGGS point with a height is written two-dimensional with the height in the centre line, so no vertical datum is claimed that the record never gave (DIGGS' own compound name is read back); pyrotd is no longer a dependency (RotD is computed by the app's own code); RotD50/RotD100 are computed from rotated time-domain oscillator responses, as Boore (2010) defines them, by the same code as the response spectrum (the old frequency-domain route read up to 30 % high at long periods and high at short ones); the seismic-signals tool's response spectrum uses the same code as the OpenSees tool's (Sa below ten record steps a period rises by up to 2.5 %); MSE walls take a reinforcement per level (`reinforcement_per_level`), as GEC-11 Example E4 changes to 4W15 mats below Level 4, with E4's mats as built-ins and a wall mixing steel and geosynthetic levels refused; the earth-pressure coefficient tool says what a left-out wall friction means for each theory (no number changes); OpenSees response spectra are computed by Nigam and Jennings at every period, as the seismic-signals tool's are (Sa between 0.02 and 0.2 s moves by up to about 10 %, longer periods by up to about 3 %); OpenSees site response gives, beside the raw surface PGA, the record low-passed at 10 Hz and Sa(0.01 s), says when motion above 10 Hz sets the raw peak, and its spectrum at the shortest periods is fixed; MSE walls: the built-in strip is GEC-11 Example E3's 50 x 4 mm strip (it was a 75 mm strip, so strip walls at the same spacing now have two thirds of the pullout and tensile resistance), and the built-in geogrids are Example E1's GG-I to GG-III (Tal 15.83, 31.66, 51.45 kN/m); the old names are refused, saying what to give instead, and results say these are worked-example values and ask for the specified product's Tal; a DIGGS file names plain WGS 84 positions CRS84, longitude then latitude as written (EPSG:4326 means latitude first), and reads an EPSG:4326 file the right way round; the report library's page lookups land on the cited page again (they were one late since pages count from 1); the conversation record now shows graph steps against the cap, the prompt and tools given, what each vision call was asked, results moved to scratch, the files each turn wrote, restarts, restores and page crashes (recording only); bespoke reports can carry a calc package's calculation (`calc_package` `render_tables`, placed by `html_to_pdf`); report ingest keeps a value printed with no unit as unknown instead of reading it as metres or kPa, converts many more units (forces, moments, kcf, kg/m3 with g stated, mg/L, conductivity, mm/blow), and says "not run" for a DIGGS schema check that could not run; `mse_wall` refuses `reinforcement_Fy`, which no check used; MSE walls: a geosynthetic's rupture check now requires a factor of 1.5 on its long-term strength Tult/RF (GEC-11's 0.90 resistance factor against 1.35 Tmax); before it took 1.0 and passed geosynthetic walls at two thirds of GEC-11's resistance, so some geogrid walls that passed now fail rupture; steel is unchanged; geostatistics results list their choices (covariance model, variogram fitted or set, nugget, kriging type, simulation method, realizations, seed), each given or default; MSE walls: the steel strip's and bar mat's Tal come from GEC-11 as printed and the facing reading is stated; the built-in welded-wire grid is GEC-11 Example E4's 4W11 bar mat (rupture FOS about three times what it was), the built-in strip's rupture FOS rises about 11 %, the spacing or coverage ratio can be given per level, and a rupture failure says one reinforcement is taken at every level; a built-in reinforcement name now refuses a strength, type or section given beside it (before, such a value was silently ignored and the built-in ran); kriging draws a plan map of the kriged mean and its standard deviation, both geostatistics maps take outlines such as the property line, and `fit_nugget=false` holds the nugget; Exponential random fields no longer come out all NaN on the rare draw a GSTools defect turned infinite, and every other draw is unchanged; the app and planlens import PyMuPDF by its current name (`pymupdf`), ready for PyMuPDF 2.0; a misspelt layer key in the downdrag, frame and subsurface tools gets its did-you-mean suggestion again; OpenSees site response takes the pore-pressure stabilization per element from the SSPquadUP formula, so a clay keeps its own ru instead of the sand's spread into it; in dense sand the surface PGA now rises with mesh refinement (a short dilation spike), and results include the surface spectrum, which does not; MSE walls with steel strips or grids take the coverage ratio Rc = b/Sh in pullout and rupture (GEC-11 Eq. 3-1) and are refused without the reinforcement's spacing; before, Rc was 1, which overstated strip pullout about tenfold (a 9 m strip wall's pullout FOS 29-42 -> 2.9-4.2); geogrids take alpha 0.8 and geotextiles 0.6 (Table 3-6), so their pullout FOS falls 20 % and 40 %; tested F* and alpha are accepted; specialists from the Agent picker build calculation packages, figures, subsurface profiles and PDFs, delegate calculations to the calc helper under the same Behavior setting, and get the SharePoint and email tools where the full agent has them; OpenSees site response reports ru from the pore pressure (it can exceed 1), puts a node at the water table, and takes permeability as hydraulic conductivity in m/s (sand had drained about 10 times too fast), so effective-stress results move; drilled shafts: a clay base takes GEC-10's mean su over 2B below the base (a weaker clay within that depth lowers the base), and rock that does not meet p. 10-22's conditions takes GEC-10's Hoek-Brown base from a layer's GSI and m_i instead of being refused; downdrag takes a clay's alpha off UFC 3-220-20 Fig. 6-14 and Table 6-16 by default, with UFC's transition zone (GEC-12 Fig. 7-17 as an option, a given alpha kept), so the drag force falls where firm to very stiff clay drags (UFC's own drag example now reproduced: 52 ft, 133 kips; OWN-83 asks whether that stays the default); MSE ribbed strips take F* = 1.2 + log Cu at the top of the wall (GEC-11 p. 3-20), with Cu = 4 (F* 1.8) when it is left out, so pullout in the top 6 m drops by up to about 10 % (give `Cu_backfill`); the coefficient tool asks for `delta_deg` with the 1948 log-spiral chart or Caltrans Eq. 4-12 and shows Kp at delta = 0 beside every result; the reference library's GEC-10 chapter 10 follows the 2018 print (Eqs. 10-2 to 10-29, the printed resistance factors) and its GEC-11 pullout defaults follow Eqs. 3-4 to 3-8 (refs a6a08d1, a refs release); downdrag takes the Beta method in clay as a named layer option and refuses inputs a layer would not use (Beta on an Alpha clay, phi on clay, alpha on sand); a phi given on a clay toe had made the toe resistance too large; a conversation downloaded as a zip can be restored from the sidebar (on Foundry too); sheet piles analysed with the log-spiral passive take UFC's passive wall friction (17 deg for clean sand unless a layer names its row of Table 4-1), so their embedment is shorter (a 4 m cantilever in sand 7.21 -> 5.06 m), and a retaining wall with a shear key but no key depth is refused; `soe` and the wall toe offer a log-spiral passive (Rankine stays the default); drilled shafts follow GEC-10 (2018) as printed: in rock the side factor C is 1.0 by default (0.65 and 1.9 as options) and the base takes 2.5 qu only where the engineer states the manual's three conditions or the rock below is RQD 100, otherwise the call is refused; clay alpha per Eq. 10-18 and base N*c 9.0 (capacities change: V-007 9,414 -> 14,690 kN, a clay case 8,995 -> 7,810 kN); report summaries and the report library cite pages as a PDF viewer numbers them (from 1); calc-package input labels are the quantity's plain name and units; on Foundry, files over 2 MB download after a Prepare press, and agent-written HTML opens and previews in a sandboxed frame on every host (a page relying on the app's own storage no longer works in the preview); report records never fill a missing total depth or water level silently ("not given", no water table placed, stated), WOH/WOR count as 0 blows with a note, and SPT refusals are marked on the SPT profile as printed; geotech page: "Download this conversation" zips a conversation's record, files and figures (on Foundry, over 20 MB, the record alone with the deliverables that fit); the seismic, ground-improvement and wave-equation calc packages now return the numbers their PDFs show, not only the PDF; on Foundry, Claude is reached through the container's proxy with no workspace glue, downloads and PDF page previews work in the published app, and the references helper runs on Sonnet 5.5 with everything else on Opus 5.5 (changeable in the sidebar); at-rest K0 takes an OCR (EN 1997-1 Eq. 9.1, default 1, stated); report records count structures from their list and figures in the main body only; at-rest K0 behind sloping ground is EN 1997-1 Eq. 9.2 always: level-ground K0 times (1 + sin beta) (at phi' 30: 0.500 level, 0.724 behind 2H:1V), the Caltrans form (which lowered K0) removed; drilled shafts in sand follow GEC-10 (2018) Eq. 10-13 step by step by default (O'Neill-Reese's depth method by name; the top 1.5 m left out in clay and rock, not in sand): eval case DS-2 7,328 -> 4,196 kN, a sand-bearing worked example 10,440 -> about 9,400 kN, clay shafts unchanged, and calc packages now print the depth formula the code computes; downdrag cites GEC-12's 10 % depth step for the soil below the toe (§7.3.5.3 clay, §7.3.5.6 sand, §7.3.5.5 both; no number changes), and the post-liquefaction result fits its size again with the rules warning whole; passive log-spiral coefficients from UFC 3-220-20 by default, by geometry (Eq. 4-12 vertical and level, Fig. 4-12 battered, Fig. 4-13 sloping; Caquot-Kerisel 1948 by name): at delta 0.66 phi' and phi' 30/35/40, 5.36/8.18/13.1 -> 5.06/7.53/11.7; sheet-pile embedment +3.5 to +5.6 %; the coefficient tool's left-out delta is now 0.66 phi' (was phi', unreported), so at phi' 30 6.30 -> 5.06; Document Review no longer offers geotechnical report tools (`log_grid` and `plot_data`'s soil-classification charts stay on GeotechStaffEngineer), and the DRAFT report checklist (off) moved to report_ingest and runs on its record; downdrag runs FHWA GEC-12's depth below the toe beside UFC 3-220-20's (the default) and warns when the two differ by more than 25 % or a check passes under one and fails under the other (headline numbers unchanged); geotech answers end with their judgment calls, each with its source, after one references look-up (about $0.20-0.53 a turn; Behavior's Consult references stops it and the consult budget caps it); the figure helper and plotting code are on for both pages (`=0` turns each off); fem2d excavation dewatered by default (larger wall forces with a water table above the excavation level); fem2d SRM over deep undrained soil a slightly higher FOS at about twice the run time; gstools conditioning by kriging by default; downdrag leaves the pile's own weight out; wall bearing on the larger edge pressure and "not checked" without q_allowable; staged FEM 500 iterations a step; liquefaction refuses amax 0; downdrag counts more toe settlement where soft soil continues below the toe; tools refuse misspelt input names instead of ignoring them (a call that worked by accident may now be refused with a did-you-mean); MSE pullout leaves traffic out (a lower pullout FOS near the top of a wall with a traffic surcharge) and strip F* reaches its deep value at 20 ft; the slope search sets its automatic ranges from the slope's own geometry (several pinned FOS move toward the published values); module results gain a `choices` list and tool results a `defaults_used` list (more text per result, no number changes), settlement warns when the secondary-compression time is an assumed 1 year, and seismic earth pressure warns when Mononobe-Okabe has no solution and the Seed & Whitman estimate is used; the slope search refuses a surface below the ground in more than one stretch and tries circles coming out on the face above the toe (lower FOS where a weak layer outcrops on the face; V-042's pinned circle 1.325 to 1.336 against a published 1.334; a default search about 1 s longer); a slope search the agent runs stops at 180 s unless it asks for up to 600 s, and a cut search says its FOS may be too high (the library itself has no time limit); searches leave out surfaces with m_alpha below 0.2 on a slice with friction, as Slide2 does (a Janbu example 1.076 to 1.199; a reinforced slope's discard-curvature case 1.467 to 1.432, the safe side); a noncircular surface is sliced at its corners and at layer and water-table crossings (noncircular and fixed-polyline FOS move, mostly up by 1-3 %, to where the old code converged at very fine slicing; a published cut-slope check 1.190 to 1.218); a CPT liquefaction call without the water table or the PGA is refused (it used to take 1.0 m and 0.25 g); circular-surface FOS move to their converged values, mostly up by 0.5-3 % (a published reinforced-slope check 1.180 to 1.210 against 1.209; a vertical cut in clay 0.800 to 0.842 against Taylor's 0.851), and rigorous searches on clay slopes take about 20 % longer; a Spencer, M-P or GLE circular search can now report a lower FOS where its old refinement followed Bishop (a published reinforced slope 1.238 to 1.186, against Slide2's circular 1.209 and noncircular 1.188); a given slip surface within 1/1000 of the model's height below its base is taken as touching the base, and one deeper is refused whatever the slice count; a seismic calc package without KA takes Rankine Ka from phi (at phi 34 degrees the static thrust 113.9 to 96.7 kN/m, the seismic increment up by the same, the total unchanged), and a settlement package without L takes L = B as the settlement tool does (a 3 m footing 39 to 87 mm); every calc package now runs its tool's own code and inputs: a pile-group package computes the group efficiency when the layout is given, a downdrag package without the fill unit weight takes the tool's 19 kN/m3 (was 20), a settlement package without B and an axial one without pile_type are refused as the tools are, and `axial_pile_capacity` itself refuses a call with no pile_type; driven-pile capacities follow GEC-12 chapter 7 as printed: in clay up by about half where the printed Fig. 7-17 replaces one simplified curve (a concrete pile 1,327 to 2,043 kN), deep sand toes down by Step 8's 3 ksf limit, an open pipe's unplugged case down (inside friction 1/3, plug weight subtracted), a beta-method clay toe far lower (Nt 3), and uplift the full shaft resistance (was 75 %); the reference library's section numbers follow the printed manuals (39 renumbered in six references, so a citation the librarian returns matches the page; needs a geotech-references release); the librarian finds and fetches subsections (a parent comes back with its own text and a list of its children instead of its whole subtree; geotech-references d5f7f52, same release); no reference tool's result is cut mid-JSON any more: long searches, chapter listings and chapters come back shorter with a note on how to get the rest (geotech-references db84be2; the app's geotech-references pin must rise with that release); settlement: a strip given with no length is a strip (2:1 stress at q 150 kPa, B 2 m, z 4 m 16.7 -> 50 kPa, consolidation 56 -> 139 mm), secondary compression uses each layer's e0 (20.83 -> 21.92 mm in the test case), the library's no-L default is L = B; a drilled shaft's cohesionless tip with no N60 is refused (it was a silent 15); the wall calc package's printed M_stab and x_R now match the analysis (419.2 -> 414.5 kN-m/m, FOS unchanged); section_properties accepts r=0 (it was refused); lateral piles in weak rock follow GEC 9 App. A.5 (alpha_r from RQD, p_ur up to three times lower; initial slope 100-500 E_m, was 0.25 E_m; 1.2 m shaft at 500 kN 3.9 -> 0.23 mm, at 8,000 kN 70 -> 90 mm and 9,555 -> 16,467 kN-m; re-run weak-rock analyses); `plot_data`'s `reference_chart` draws standard chart lines from the references (needs the next geotech-references release, after 6ad9fbb, and a pin bump; hidden on 1.4.1); cantilever walls: a surcharge over the heel no longer resists sliding, overturning or the middle-third check by default (F55: sliding 1.17 -> 1.07, overturning 3.06 -> 2.78; bearing unchanged; `live_load_resists=true` gives the old results); MSE walls, ASD path: the same (6 m wall, q 12 kPa: sliding 2.599 -> 2.339, overturning 4.333 -> 3.900; GEC-11 E4 inputs 2.268 -> 2.104); calc packages print Given Inputs, Assumed Inputs and Values Used, and the agent states a sensitivity number only from a tool run or a reference; driven piles: options for GEC-12 Fig. 7-18 (or the smaller of 7-17/7-18), a shrink-swell depth, Table 7-9 beta/Nt and the toe phi over 3b, a custom pile section on every axial tool, `capacity_vs_depth` refuses a call with no pile_type (no default number moves); figures: the layout check reports hatching and arrows through lettering, names keep clear of reference lines and hatching, a plan view takes in every outline, the plotting box allows `time` (figures with names or plans may draw differently and carry new warnings); the calc helper is told its call budget and warned from four calls left; `calculate` takes a list or a name-to-expression map; `read_pdf_text` takes 1-based `pdf_pages` and labels each page `pdf_page`; a failed scratch edit says nothing changed; PDF reports: a figure that does not fit the rest of a page starts the next page at full size with its heading, one taller than a page is scaled to fit, html_to_pdf no longer drops the text after such a figure (reports can run longer), `page_contents` numbers pages as `pdf_page` and gives each figure's printed size, figure files are named from their titles; new: post-liquefaction downdrag (GEC-10 §10.7): the neutral plane, drag force and settlement before the earthquake and after the liquefied layers reconsolidate, every method side by side with the cited side-resistance options (50 % default, zero, su,r, a ratio), taking the liquefaction tool's output (a CPT's depth profile too); a short result by default (tables as columns and rows, more than 6 zones summarised), `detail='full'` for the whole; the review page's prompt names its one calculation tool, `calculate`; bearing: Meyerhof with a load inclined at phi or more is refused (it gave a capacity that rose again with the angle), as are a negative or 90 deg load inclination, a ground slope outside 0-45 deg and a base tilt past alpha tan phi = 1 (the module's own limits); at Df/B = 1 exactly Vesic's depth factors use arctan(1) as UFC prints (about 4 % lower there only), `depth_factor_method` gec6 or none, an inclined load on a soil with cohesion needs `vertical_load`, Vesic's inclination factors now apply on a strip (an error on the unsafe side: -34 % in the example); elastic settlement with no Iw takes GEC-6 Table 5-12 (12 % higher for a flexible square centre); a shaft capacity-by-length sweep lists the lengths it skipped and why; a Method Choices table whose rows run taller than a third of a page, or past the right margin, prints one block per choice (on master the retaining wall with surcharge package dropped 18 words and the slope report 37, with 10 more cut at the margin; now none) and a table ending at a page foot no longer fails the PDF; Story PDFs no longer leave out what starts at a page foot (a section, figure, table row, list item, step result or a paragraph's last lines; a PDF may be a page longer); a call whose `parameters` is not an object of named inputs is refused with the expected shape and input names, not run; SoilProfile's lateral-pile input follows GEC 9 p. 219 (stiff clay takes Reese et al. 1975 only where free water is at the ground surface, Welch & Reese otherwise, below the water table too; deep-water results move, 600 kN: 10.9 -> 15.5 mm; `free_water_at_surface=True` restores the old choice) and passes `n_cycles`; lateral piles: stiff clay below the water table and Matlock soft clay under cyclic loading follow the printed curves (stiff clay, 0.9 m shaft, cyclic 600 kN: 19.4 -> 63.0 mm; cyclic loading there takes `kc`), and a load beyond what the pile and soil can carry fails with that capacity instead of returning numbers; lateral piles in stiff clay above the water table under cyclic loading use the printed cyclic curve with the number of cycles N (required; before, the static curve: a 0.9 m shaft at 300 kN went 4.47 mm / 387 kN m to 6.03 mm / 430 kN m at N = 100), static unchanged; log-spiral Kp from a chosen printed source (Caquot-Kerisel 1948 as Caltrans prints it, the default; UFC 3-220-20 Fig 4-13 or Eq 4-12) and sloping ground for Caquot-Kerisel and for K0 (Caltrans or EN 1997-1, named), Kp between the chart's 5 deg lines read on its log scale (up to about 5 % lower; log-spiral sheet pile embedment about 2 % deeper, 2.895 -> 2.958 m), a keyed wall's passive as UFC Fig 4-29 draws it (Pp and FOS sliding go up: 9.7 -> 29.2 kN/m, 1.31 -> 1.57 for a typical wall); driven piles, alpha method: Ca read off GEC-12 Fig. 7-17 as drawn, so alpha can exceed 1.0 (a worked example's shaft 715.3 -> 761.9 kN); clay past the chart's curves takes API alpha at or above the engineer's su/sigma'vo line and Ca held at the curve's end below it, and a run that needs the line and lacks it is refused; fem2d consolidation with theta < 1 starts with Rannacher half-steps (`startup_steps=0` for the old stepping) and accepts fill placed under water; SoilProfile weak rock needs qu and Es (no unprinted defaults) and its RQD defaults to 100 (a 0.9 m shaft at 2,000 kN: 29.3 -> 41.7 mm); on Foundry a model's family can be named (an `anthropic:` / `openai:` prefix or the sidebar's Model family), so a Claude RID reaches the Anthropic proxy, and prompt-cache marks sit on message blocks (`GEOTECH_FOUNDRY_PROMPT_CACHE=0` drops them); the references helper can read FHWA NHI-06-088/089 (Soils and Foundations), which the tools cite (refs 6880c88); reference search: curated synonyms (wrong-meaning hits removed; in-situ test, ground-improvement and pile-type groups added) and a query none of whose words is indexed keeps its own words (refs 6f63619); a cantilever wall's sliding check no longer adds two-thirds of the foundation cohesion to friction (GEC-6 p. 100: the two are not combined): drained friction by default, cu times B where phi = 0, UFC's friction plus adhesion as an option (a 5 m wall: FOS 1.507 -> 1.233, which fails; 1.553 with UFC's rule); a drilled shaft tipped in rock takes its base on the socket diameter when one is given (GEC-10 Example 10-4 now as printed) and takes the rock side factor C as an input; stone columns use only GEC-13's printed Priebe curve; OpenSees site response can take Darendeli or Menq laboratory curves as its backbone (default unchanged) and reports its solver steps, with Poisson's ratio, element count, sand constants and Rayleigh frequencies now settable; Story PDFs draw no background shading, so the grey bars that repeated on later pages are gone (heading rows get a rule, PASS/FAIL prints in bold colour; `html_to_pdf` says a user's backgrounds are not drawn); the cantilever wall figure draws the active pressure where the analysis applies it (the vertical plane through the heel), with the surcharge, and draws a shear key (position optional); the pile group plan takes an optional cap outline and property line; downdrag profiles take an optional ground elevation (results unchanged), and the library's `plot_neutral_plane` no longer draws depth upward; calc packages list their own left-out inputs under Assumed Inputs; calc text cites GEC-6 chapter 5 and FHWA-SA-02-054, and the settlement, pile-group, drilled-shaft and wave-equation citations match the printed GEC-6/10/12, with a guard that every cited manual number exists (no numbers change); fem2d wall, strut, spring and joint forces are per metre whatever the slice thickness `t` (at t other than 1 they no longer scale with it; t = 1 unchanged); a ground anchor given a bond stress or load transfer needs `soil_type` ('soil', 'rock' or a GEC-4 table row; it was taken as sand); drilled shafts take `fractured_rock_thickness` (GEC-10 p. 8-15) and aggregate piers take `settlement_method` from GEC-13 Fig. 5-21 with `E_column` and `E_soil` (defaults unchanged); `mse_lrfd_external_stability` works the bearing resistance out from `gamma_foundation` and `phi_foundation` by GEC-11 Eq. 4-22 (qR = 0.65 qn; before, bearing went unchecked), ground-improvement feasibility holds top + thickness against GEC-13's typical depths (aggregate piers feasible as stone columns to 30.5 m, was excluded past 12 m; vibro-compaction excluded past about 50 m), inputs that changed nothing are refused with their reason (axial E, socket_length, E_column, gwt_depth_m, i_c_limit, s_g, beam A, gamma_foundation on mse_wall); `earth_pressure_coefficient` refuses wall friction, a battered back or a slope its coefficient does not take and names the alternative (before, Rankine Ka with delta 20 gave 0.333, and Rankine Kp, Caquot-Kerisel or K0 with a slope gave the level-ground value), Caquot-Kerisel echoes delta = phi, `composite_section_ei` refuses bars the section would drop (use `cased_concrete`), `capacity_vs_depth` gives `Q_allowable_kN` and `make_pile_section` gives `E_kPa`, combined settlement with no Es says no immediate settlement was computed; `plot_data` and the plotting box refuse a figure that draws nothing, or one whose title, axes or series call it a placeholder, before any file is written; calc-package figures are drawn for the 5.0 in width they print at, so their lettering prints at 7 pt or more (it was 3-6.5 pt), and a figure that still prints small is named in `figure_warnings` as the module's own; calc-package figures follow the figure rules (M-O pressure in kPa; the wall's toe ground where the analysis takes it; downdrag depth axes say no elevation was given; the pile group drawn as a plan with a scale bar and a north arrow from the new `north_deg` or up, tension hatched; no numbers change); keys and notes sit under the drawing (the figure grows to hold them), labels beside what they name, values on lines kept clear (no results change); LaTeX packages widen a figure, up to the text width, where its set width would print lettering under 7 pt; ground anchors sized for `test_load_factor` x DL, default 1.33 (was a fixed 1.50; 1.20 temporary SOE and 1.50 creeping soils as options): DL 300 kN on 15 mm strand 3 strands -> 2, `performance_test_kN` 1.33 DL, the 32 mm bar holds up to 501 kN DL (was 509) and the 1-3/4 in bar 1,067 kN (was 960) per GEC-4 Table 9; `braced_excavation` / `apparent_pressure` refuse a layer above the base with no `soil_type`; a multi-anchor FHWA wall refuses `FOS_embedment` and `Kp`; ground anchor bond length uses GEC-4's printed ultimate load transfer per metre (Tables 6 and 8), not a stress over a 150 mm hole: soil bonds were about twice too long and rock ones too short (DL 445 kN: medium sand 13.03 -> 6.14 m; the old rock rows 3.00 -> 4.50 to 8.90 m), six old names refused with their replacement, rock FS 3.0, minimum bond 4.5 m (3 m for a bar in rock), bar unbonded minimum 3 m, bond zone max(1.5 m, H/5) behind the failure surface measured normal to it; bearing with `factor_method='meyerhof'` uses Meyerhof's own N_gamma, (Nq-1)tan(1.4 phi): q_ult falls below phi about 46 deg (a 2 m square footing at Df 1 m, phi 32: 1,465.9 -> 1,241.0 kPa, -15 %), and under an inclined load its shape factors are 1 (UFC 3-220-20 Table 5-6 note A; the same footing at 10 deg 894.9 -> 594.9 kPa); `ngamma_method='vesic'` restores the old value under a vertical load only; the bearing result gives the net allowable pressure and the allowable structural load (for an eccentric footing q_all A' - q A); module choices read "given" only for inputs the call names as given; seismic PAE carries (1 - kv) per UFC Eq. 4-16 (kv still defaults to 0; at kv 0.1, 134.8 -> 121.3 kN/m); anchor tendons sized so the 1.50 DL test load is at most 0.80 SMTS (a 500 kN 32 mm bar becomes 36 mm); a variogram nugget of 0 fixes the nugget and a negative one is refused; cushion_cor changes the default cushion; AGS4 files with rule errors are invalid (before, every file passed) and a DIGGS file that passes the dictionary check is valid; concrete_fc is refused on the drilled-shaft tools (it was ignored); the drilled-shaft package prints the Nc cap, F_r and the bell's 1.27/D; fem2d_excavation `wall_contact='interface'` adds Mohr-Coulomb wall friction (default smooth, no number moves); deepagents' `execute` and `delete`, which cannot run here, are no longer offered on the geotech page; the calc helper can look at the pages of the PDFs it builds; behind `GEOTECH_JUDGMENT_CALLS` (OFF then, on by default since 2026-10-20) an answer whose judgment calls name no source, in a turn with no references consult, is sent back once; Caquot-Kerisel passive coefficients follow the printed chart and table over phi 0-45 deg (Kp0 at phi 35 9.20 -> 10.2, at 40 14.0 -> 17.7; at delta/phi = 2/3, 35 7.20 -> 8.24, 40 10.96 -> 13.26; the log-spiral options of sheet_pile and retaining_walls only, SOE uses Rankine and is unchanged); new SOE and reliability calc packages; tool results and calc packages say what a left-out input stands for (no water table, one soil layer, bearing not checked, foundation soil taken as the backfill, with its basis), a calc package's Given Inputs hold only the inputs the call names in `given` in the user's words, every other input prints under Assumed Inputs and a choice the user did not give reads "assumed" (no computed number changes); PyMuPDF floor 1.27.2 (cap <2 kept), and planlens main reads 2.0 correctly with one PDF read at a time across threads (two reads at once could leave box heights wrong and printing lost for the rest of the process; that race is in released planlens 0.13) — a planlens release note; references (refs 9cd4228, needs a geotech-references release + pin bump for the cluster, with the plasticity and grain-size chart lines): DM 7.1 cited as UFS 3-220-10 (16 January 2026) with its printed pages, GEC-13 chapters 3-10 rebuilt from the printed manual, GEC-12 ch 7, micropile ch 5 and five DM 7.2 sections follow the print, run-on figure captions fixed, the ground-improvement package cites GEC-13 as printed (no numbers change).
- **Next:** EV-02: Foundry brief 6 is drafted (`../foundry_handoff/AI_FDE_BRIEF_6.md`: part A the image zoom check, `image_zoom_check_foundry.py`, on the rollout model; part C the suite with arms `baseline` and `kept_readings` as an optional arm; its `lean`, `turn_retry` and `coverage` arms were removed 2026-11-09). Its rc test wheels are not built (EV-02 is Later). Module follow-ups in Next: GM-79, GM-81, GM-82. Also EV-23.
- **Owner:** OWN-101 (with it, photos of Duncan & Wright §6.5.3 and ch. 8). Answered on the desk 2026-10-14 (BACKLOG "Done"): hold the release and use the Prompter key on Funhouse this month; downdrag shows every method, skin-friction stress per GEC-12; fem2d measures from the end of stage 0; calculation tables are the next module train; the old `geotech-report-extraction` project is deleted from PyPI. Answered 2026-10-15: the QC routine's token revoked and the routine deleted; downdrag leaves the pile's own weight out by default; the figure helper and plotting code on everywhere (done 2026-10-20). Answered 2026-10-20 (folded that day): slope supports offer both, slack the default (in GM-140); downdrag keeps both toe rules with a warning (done 2026-10-20); driven piles' five readings (done 2026-10-20) and a note on clearer writing; pavements and slabs modelled as retained soil (done 2026-10-20); passive coefficients from UFC 3-220-20 by default (done 2026-10-20); the judgment-calls switch on (done the same day).
- **Rollout model (owner, 2026-10-12):** one GPT-5.6 Sol-or-higher deployment for every call, not gpt-5.1; the app measures it on first use. So the gpt-5.1 rehearsal wave is dropped, and the standards PDFs stay with a provenance note (BACKLOG "Done").
- **Done on master 2026-10-12 (released in 5.34.0):** the default review page's zoom wording, reading helper and answering budget; the markup check against the printed thing; a plain message and one retry on a refused key; `.env` working on dosdev; the suites run on CfA's pins (WSL), which found the python-ags4 install conflict; the install guide and `TINYAPPS.md` current; Department addresses scrubbed from tracked files (history: OWN-56). Commits ae50842, da28a69, 0562ddf, a229090, 24b46df. On dosdev, 5.33.0 installs with pandas left unpinned (pip takes 2.x) until the next release.

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
Foundry, narrative conventions re-read from the narrative answer sheets (transcribed by a Claude session, not the owner; RI-72) (DRAFT).
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
sheet, and turned the penetrations case's specifics into rules. The owner: "This seems
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
at `original`. Re-run the penetrations question ("find every GCE penetration callout,
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
| Wrapper repo | `tinyapps/wrapper_repo/` | `app.py` (sets `GEOTECH_DOTENV`, `GEOTECH_WEBAPP_DATA=/home/data/…` on App Service), `packages.txt` (pin ≥5.26 + CfA fleet pins + Key Vault libs), `run.sh` (CfA template: `$PORT`; the published address from the app setting `APP_ORIGINS` since 2026-11-10, else its `CHANGE-ME` line), `.env.example`, `README.md`. |
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
`module_work/field_feedback/2026-09-09_soe-review_v5.11.2/PLANLENS_SURVEY_MEASUREMENTS.md`.**

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

### 5.17.0 RELEASED (2026-09-15, tag `v5.17.0`) — the SOE review fix train

`module_work/field_feedback/2026-09-15_soe-review-rerun_v5.15.0/FINDINGS.md`,
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
no document layer at all. Measured on the real 260-page SOE submittal
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
SOE submittal first — private, never into planlens' public repo);
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
Private measurements: `module_work/field_feedback/2026-09-09_soe-review_v5.11.2/
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
an agent-cell script, `suite_trial.json`) — deliberately never staged.
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

### FIELD FEEDBACK 2026-09-09 — SOE review: the 401 was the MODEL, not SharePoint

Drop triaged into `module_work/field_feedback/2026-09-09_soe-review_v5.11.2/`
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
   (C:/dev/planlens, branch main) → `git submodule
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
   module_work/field_feedback/2026-09-04_downdrag_v5.11.2/).
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
- **Owner's Foundry app IS PUBLISHED and running** (a government enclave, a sandbox project) but **blocked on
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
- **`trapezoid` from `geotech_common.numpy_compat`** (not `np.trapezoid` or `np.trapz`: the code also runs on numpy 1.26, INF-53); SI units throughout.

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
- **planlens** (`C:/dev/planlens`, PyPI `planlens`,
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
