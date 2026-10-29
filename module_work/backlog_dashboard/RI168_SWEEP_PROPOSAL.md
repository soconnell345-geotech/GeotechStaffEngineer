# RI-168: report ingest's open items, swept against the model-first default

Proposal only, 2026-10-29 (app master 25c30de). BACKLOG.md is not edited; the lead folds this in.

**The yardstick.** The owner ruled on 2026-10-29 (RI-186) that report ingest reads model-first by default at every stage: page labels and grouping, logs, lab, calc and narrative. Today's rule-based readers (the log grid and floor, the lab floor, form templates, the label vote and review, today's narrative reader) stay one setting away. So an item stays only if it changes a saved report's output or what a live run returns under that default; work that only hardens today's readers is parked as an idea; anything done or replaced is closed. Evidence: Foundry builds e and f (`module_work/field_feedback/2026-10-27_foundry_build_e/TRACE_REVIEW.md`, `module_work/field_feedback/2026-10-28_foundry_build_f/TRACE_REVIEW.md`, cited below as "build e §n" and "build f §n"), `module_work/field_feedback/2026-09-16_report-ingest/MEASUREMENTS.md` and `module_work/REPORT_INGEST_PLAN.md` §8.7-§8.8.

## Summary

102 open RI items: **KEEP 38** (15 of them in flight, including this sweep), **PARK 44**, **CLOSE 14**, **MERGE 6**.

| current tier | items | KEEP | PARK | CLOSE | MERGE |
|---|---:|---:|---:|---:|---:|
| Now | 11 | 7 | 0 | 3 | 1 |
| Next | 47 | 26 | 9 | 8 | 4 |
| Later | 12 | 4 | 7 | 1 | 0 |
| Parked | 32 | 1 | 28 | 2 | 1 |

What changes most:
- Seventeen Next items (logs, lab, calc and label follow-ups from the GPT and local Claude runs) are parked or closed, because they hardened today's floors, merges, vote or review, or because model-first already reads the case right (the R15_p46 water depths, the log header fields, limits on gradation sheets).
- Four Now items (RI-73, RI-78, RI-86, RI-87) are finished or replaced by RI-186.
- Three Parked items come back or close: RI-157 returns narrowed to soundings, which still use today's reader under model-first; RI-132 and RI-148 close as superseded.
- The kept work is the in-flight model-first fixes plus: model-first placement (RI-99), the page-label conventions (RI-104), the public held-out set (RI-79), the text source for scans (RI-91), triage's bound ranges (RI-113, RI-22), and the reader-independent record, question-list and DIGGS items.

**Owner questions** (judgment calls; one sentence each):
1. **RI-86:** Does build f's whole-report comparison (MF-2) answer your own whole-report check on the Foundry FDE, or do you still want to run it yourself?
2. **RI-104 (with RI-73):** In a test-pit appendix, is a page of foundation sketches a test-pit log or a photo page? 11 of the 17 pages model-first labels sent to the wrong reader on R28 are that one appendix (build f §3.3).
3. **RI-157:** Cone and probe soundings still go through today's reader under model-first; should they move to a model-first reader too, or should we only fix the two-page probe sheet that becomes two records?

## Every open RI item

| ID | current tier | proposed | reason |
|---|---|---|---|
| RI-73 | Now | **MERGE** | Into RI-104. The range labeller is built (switch merged 2026-10-24) and is now the default page labeller under RI-186; MF-2 measured it (32 pages to the wrong reader against today's 26, build f §3.3). What is left is RI-104's conventions and the held-out pages. |
| RI-78 | Now | **CLOSE** | Done: the local Claude arms were measured 2026-10-24 and their follow-ups filed (RI-74, RI-81 to RI-83); its part (3) was measured on Foundry by MF-2 (build f §3.3). |
| RI-86 | Now | **CLOSE** | Answered: handed the right pages a plain call reads about as well (logs, lab, calc, narrative measured 2026-10-24); the owner chose model-first (OWN-96, RI-186) and MF-2 ran the whole-report comparison (build f §3). Owner question below. |
| RI-87 | Now | **CLOSE** | Superseded by RI-186: all nine pieces merged, measured on Foundry in MF-1/MF-2 (builds e and f), and the default decided by the owner 2026-10-29; MF-3 and the bug fixes live in RI-186 to RI-194. |
| RI-101 | Now | **KEEP** | Question list on a live report: the owner's own questions, independent of which reader is default; first live run still to come. |
| RI-182 | Now | **KEEP** | In flight (narrative): the owner's 2026-10-29 rulings and answer-sheet corrections; build f §4.1. |
| RI-186 | Now | **KEEP** | In flight: model-first the default for every stage (owner 2026-10-29); MF-3 runs both arms on held-out reports. |
| RI-187 | Now | **KEEP** | In flight (all model-first stages): an unread group is a failed item; 45 pages unread on MF-2's on arm, build f §1. |
| RI-188 | Now | **KEEP** | In flight (narrative): R16's parent answered 12 of 29 against 25; build f §3.4. |
| RI-189 | Now | **KEEP** | In flight (narrative): another client's report rule, 6 misses; build f §4. |
| RI-190 | Now | **KEEP** | In flight (every stage): a server error re-sent; one log item lost in MF-1b, build f §1. |
| RI-06 | Next | **CLOSE** | Superseded: RI-07's label link and Claude's 79 of 79 links (RI-55); the GPT run-1 lab drop is moot on Claude. |
| RI-09 | Next | **CLOSE** | Done: MF-2 re-ran the whole-pipeline ingest stage on six reports in both arms (build f §3); a public report end to end is RI-79. |
| RI-11 | Next | **KEEP** | The owner's ingest-ledger design; a product feature unaffected by model-first. |
| RI-12 | Next | **CLOSE** | Moot: a cheap vision second voter was a lever for today's narrative reader; the model-first narrative reads the pages in one Opus call (+10 recall, build f §2.4). |
| RI-16 | Next | **PARK** | Cost levers of today's vision voter; under model-first the describer runs on Haiku. Idea only. |
| RI-22 | Next | **KEEP** | Triage decides bound documents on both routes; owner chose 'agent drafts, you confirm bound calls' (2026-10-26). |
| RI-24 | Next | **PARK** | Ideas without evidence: boring locations off the plan figure, and the sounding floor's digitised series (OWN-15). |
| RI-25 | Next | **PARK** | Low bundle; the Funhouse gateway part is moot now ingest runs on Foundry. |
| RI-29 | Next | **KEEP** | Report to DIGGS through the app is the end product; unaffected by which reader is default. |
| RI-38 | Next | **CLOSE** | Measured 2026-10-24 (wave 1); its follow-ups were RI-76, closed above. |
| RI-42 | Next | **KEEP** | The owner's marks on the Report Questions Desk still to fold into checklists and schemas (reader-independent). |
| RI-54 | Next | **PARK** | Hardens today's narrative-plus-log counts; wave 2 showed zeros and fill change nothing when the narrative answers every count, and RI-182's ruling 3 now settles what the counts cover. |
| RI-55 | Next | **PARK** | The label-link switch lives in today's lab reader only; Claude links 79 of 79 with it on, and model-first lab linked 76 of 79 (build e §3.2). |
| RI-57 | Next | **PARK** | Today's calc reader (continuation and fold); the owner's calls-per-report question was answered by wave 2 (about 0.5 to 0.6 calls a page). |
| RI-64 | Next | **PARK** | The label vote does not run under model-first (RI-186); its policy ideas stay as notes for today's label path. |
| RI-68 | Next | **CLOSE** | Done by MF-2: today's vision voter and label review ran on Claude through Foundry and were scored on pages sent to the wrong reader (build f §3.3: 26 off, 32 on). |
| RI-72 | Next | **KEEP** | Measurement basis for MF-3: every score is agreement with Claude-made answer sheets; the glossary wording that may reach the model-first instruction (RI-189) still says 'the owner's answer keys'. |
| RI-74 | Next | **PARK** | Parts (2)-(4) and the floor-label filter harden today's calc floor and merge, which model-first does not run; calc reads level on both routes (build f §2.3). Scorer one-to-one credit noted for MF-3. |
| RI-75 | Next | **MERGE** | Into RI-81. Parts (1) and (2) were GPT reading omissions that Claude does not make (the item's own 2026-10-24 correction); only the scorer part (3) remains. |
| RI-76 | Next | **CLOSE** | Superseded by model-first logs: the R15_p46 water depths read 64 of 64 (build e §3.1); (2) does not occur on Claude (RI-82); (1) and (4) are today's floor and merge; layer merging (3) goes to RI-103. |
| RI-77 | Next | **CLOSE** | Superseded: counts and structure lists by RI-178 and RI-182's rulings, the answer sheets checked by RI-182's corrections, figure counts by RI-184, invented hazards by ruling 4; scorer leftovers go to RI-66. |
| RI-79 | Next | **KEEP** | The only answers not made by a Claude session (AGS data); held-out evidence for the model-first default. Take in RI-84 and RI-85. |
| RI-81 | Next | **KEEP** | Lab (both routes): Spanish units unconverted and a redox sign dropped change the record on every arm; the scorer double-counts sieve points (130 of 826). |
| RI-82 | Next | **CLOSE** | Superseded by RI-87 piece 4: the model-first log schema holds project name and number, other header fields, recovery as a length and bottom elevation (header fields +12, build e §3.1); strata are RI-103. |
| RI-83 | Next | **MERGE** | Into RI-66. Part (2) (one list item twice) is RI-178; (1) the 'word - reason' fold, (3) the scorer and (4) citations in run files join the narrative scorer item. |
| RI-84 | Next | **MERGE** | Into RI-79. The model-first log builder keeps depths with no ruler (flag mode, RI-87 piece 6), so first measure the public scans on model-first; visual scales only if depths still fail. |
| RI-85 | Next | **MERGE** | Into RI-79: the answer-sheet and scorer conventions are needed to score the public set. |
| RI-91 | Next | **KEEP** | Every stage: model-first places values by matching printed words and sends pictures with their text; scanned pages without DI text have neither (owner direction). |
| RI-94 | Next | **KEEP** | Narrative and bound records: a bound Volume 1 counted as a previous investigation changes the count and the boring list on either reader. |
| RI-99 | Next | **KEEP** | Model-first placement is the default: lab placed 52 of 70 values at a spot (74 %) against the 90 % gate (build f §2.2). |
| RI-102 | Next | **KEEP** | Record: save what the question list's code rows need (RI-101); independent of the reader. |
| RI-103 | Next | **KEEP** | Model-first log reader (owner OWN-95): strata as the narrative defines them; also takes RI-76 (3). |
| RI-104 | Next | **KEEP** | Page labels are model-first by default now: the conventions go into the range labeller; MF-2's 11 R28 misroutes are a convention question (build f §3.3). Owner question below. |
| RI-106 | Next | **KEEP** | Both log readers and DIGGS: 'not measured' kept apart from 'not encountered' changes the record (owner note OWN-31). |
| RI-107 | Next | **KEEP** | Lab links on both routes (owner note OWN-31); first step a $0 count of printed combined samples on the open sheets. |
| RI-113 | Next | **KEEP** | Triage runs on both routes and alone decides bound ranges under model-first (R16 starts a page later than off, build f §1). |
| RI-116 | Next | **PARK** | Built on today's lab reader (a sheets-per-call setting, off); model-first lab windows are separate. Re-check R28's Atterberg flow curves on MF-2's saved on-arm records at $0 before reopening. |
| RI-168 | Next | **KEEP** | This sweep; close it when the lead folds this proposal into BACKLOG. |
| RI-170 | Next | **KEEP** | Question list (RI-101): the printed post name matched against Foundry's list of posts (OWN-98). |
| RI-178 | Next | **KEEP** | In flight (narrative, both readers): repeated list items dropped, structures recounted; build e §11. |
| RI-179 | Next | **KEEP** | In flight (narrative): an over-long summary sent back once; build e §11. |
| RI-181 | Next | **KEEP** | In flight (narrative): one item per foundation type and bearing value; build e §11. |
| RI-184 | Next | **KEEP** | In flight (record): an incomplete copy flagged; build f §3.5 found R05 and R28's attachments. |
| RI-191 | Next | **KEEP** | In flight (lab, calc): a cut-off group split to one sheet; R28_p163 0 of 47, build f §3.4. |
| RI-192 | Next | **KEEP** | In flight (calc): a list sent as text accepted; 28 calc pages refused, build f §3.4. |
| RI-193 | Next | **KEEP** | In flight (run record): 19 false 'skipped' entries for dividers; build f §3.4. |
| RI-194 | Next | **KEEP** | In flight (DIGGS export): a summary row's resistivity written; R13 read-back differs, build f §3.4. |
| RI-40 | Later | **KEEP** | DIGGS export, unaffected by model-first; waits on DIGGS' list of coordinate-system names. |
| RI-41 | Later | **KEEP** | DIGGS 2.5.a validation fails on CI's pydiggs; reader-independent (could move to §3). |
| RI-51 | Later | **PARK** | The N60 slot fight is in today's floor-and-model merge; on model-first only the schema wording matters, and MF-1 shows no N60 misread. |
| RI-60 | Later | **PARK** | Cost-record gaps; change no value (RI-171 and RI-173 covered the Foundry totals). |
| RI-66 | Later | **KEEP** | Narrative scorer: MF-3 decides narrative on recall; take in RI-83 and RI-77's scorer leftovers and RI-181's noted scorer change. Changes the measurement, not the record: decide after MF-3. |
| RI-69 | Later | **PARK** | Measurement plumbing follow-ups; change no value. |
| RI-134 | Later | **PARK** | Spend record of a killed attempt; changes no value. |
| RI-146 | Later | **CLOSE** | Superseded by RI-175: the model-first lab instruction writes limits on a gradation sheet as their own test (R17_p114 56 of 56, R28_p163 47 of 47 in MF-1, build f §2.2). |
| RI-160 | Later | **PARK** | Measured on today's grid and floor; take in RI-153. A $0 count of first-stratum tops refused on MF-1/MF-2's model-first records decides whether it reopens. |
| RI-165 | Later | **KEEP** | Lab, model-first (now default): a cut lab part with no printed hole gets no carried link on that route. |
| RI-167 | Later | **PARK** | Idea without evidence: the transform route may not carry the proxy's retry header; check first. |
| RI-180 | Later | **PARK** | Today's narrative reader only (its own 2026-10-29 note). |
| RI-43 | Parked | **PARK** | Today's depth gate; needs a misread first. |
| RI-46 | Parked | **PARK** | Today's log floor. |
| RI-49 | Parked | **PARK** | Today's log floor (stack reader). |
| RI-52 | Parked | **PARK** | Today's log floor. |
| RI-53 | Parked | **PARK** | Bound-pass details; each narrow. Part (4), logs counted instead of holes, is noted with RI-157. |
| RI-58 | Parked | **PARK** | Today's calc splitter; model-first groups calc pages from page heads and descriptions (RI-87 piece 3). |
| RI-59 | Parked | **PARK** | Today's calc floor and merge. |
| RI-61 | Parked | **PARK** | QA ranking and single-report folds; part (8) belongs in the release notes. |
| RI-62 | Parked | **PARK** | Today's label review, which does not run under model-first. |
| RI-70 | Parked | **PARK** | Today's SPT halt rule; part (1) wording would also suit the model-first instruction if a case shows. |
| RI-71 | Parked | **PARK** | Today's lab floor and links. |
| RI-100 | Parked | **PARK** | Ambiguous unit spellings left on purpose. |
| RI-118 | Parked | **PARK** | Today's floor reading one text-layer character on one sheet family. |
| RI-119 | Parked | **PARK** | Floor-versus-model QA noise on one report; the floor does not run under model-first. |
| RI-122 | Parked | **PARK** | Brief wording only. |
| RI-132 | Parked | **CLOSE** | Superseded by RI-87 piece 4: the model-first log schema holds any printed header field (its own park note). |
| RI-147 | Parked | **PARK** | Checklist count direction; changes no value. |
| RI-148 | Parked | **CLOSE** | Superseded by RI-87 pieces 2-3: model-first groups logs by the describer's hole field; MF-2 kept every printed hole in both arms (build f §3.2). |
| RI-151 | Parked | **PARK** | One wrong join on one report; owner called it an edge case. |
| RI-152 | Parked | **PARK** | Today's log floor on one page. |
| RI-153 | Parked | **MERGE** | Into RI-160 (same gate, tops just above the ground surface). |
| RI-154 | Parked | **PARK** | Today's lab floor; cosmetic. |
| RI-155 | Parked | **PARK** | Today's log floor, one record. |
| RI-157 | Parked | **KEEP** | Narrowed to soundings: cone and probe sheets still go through today's sounding reader under model-first, and MF-2 split two two-page probe sheets into two records each in both arms (build f §3.2, §7). The log part is superseded by model-first's join by hole. Owner question below. |
| RI-158 | Parked | **PARK** | One sideways summary page; RI-149's upright turn may already fix it. |
| RI-161 | Parked | **PARK** | Today's log floor, one record. |
| RI-162 | Parked | **PARK** | Hypothetical; all 41 firings were real. |
| RI-163 | Parked | **PARK** | Never seen in the 38 reports. |
| RI-164 | Parked | **PARK** | 2 of 135 turned pages on one report. |
| RI-166 | Parked | **PARK** | Today's lab splitter. |
| RI-183 | Parked | **PARK** | Owner's idea, no time now (2026-10-29). |
| RI-185 | Parked | **PARK** | Owner's idea after RI-184; wait for RI-184's note to land. |
