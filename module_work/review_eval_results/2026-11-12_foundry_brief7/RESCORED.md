# Foundry brief 7, re-scored on what each run delivered

The 84 saved runs, scored again with no model call after EV-41: a check reads the reply and the files the run delivered, and no longer the run's own page readings or kept helper work. "Recorded" is the score as the run wrote it on Foundry; "delivered" is the corrected one. Only the two whole-set review tasks read files, so only they can move.

## GPT-5.6 Sol

| set-up | checks, recorded | checks, delivered | tasks passed, recorded | tasks passed, delivered |
|---|---|---|---|---|
| `baseline` | 92 of 95 | 87 of 95 | 12 of 14 | 12 of 14 |
| `show_me` | 93 of 95 | 91 of 95 | 12 of 14 | 12 of 14 |
| `method` | 91 of 95 | 89 of 95 | 12 of 14 | 12 of 14 |
| `sheet_tools` | 92 of 95 | 92 of 95 | 12 of 14 | 12 of 14 |
| `all_three` | 93 of 95 | 92 of 95 | 12 of 14 | 12 of 14 |

The two whole-set reviews, checks passed (recorded, then delivered):

| set-up | coord-full-review | coord-long-review |
|---|---|---|
| `baseline` | 13 then 13 of 15 | 20 then 15 of 21 |
| `show_me` | 14 then 12 of 15 | 20 then 20 of 21 |
| `method` | 13 then 13 of 15 | 19 then 17 of 21 |
| `sheet_tools` | 15 then 15 of 15 | 19 then 19 of 21 |
| `all_three` | 14 then 14 of 15 | 20 then 19 of 21 |

## GPT-5.4

| set-up | checks, recorded | checks, delivered | tasks passed, recorded | tasks passed, delivered |
|---|---|---|---|---|
| `baseline` | 80 of 95 | 74 of 95 | 10 of 14 | 10 of 14 |

The two whole-set reviews, checks passed (recorded, then delivered):

| set-up | coord-full-review | coord-long-review |
|---|---|---|
| `baseline` | 11 then 9 of 15 | 12 then 8 of 21 |

## Checks whose verdict changed

- GPT-5.6 Sol `baseline` coord-long-review: callout-missing-detail — passed as recorded, fails on what was delivered
- GPT-5.6 Sol `baseline` coord-long-review: door-not-scheduled — passed as recorded, fails on what was delivered
- GPT-5.6 Sol `baseline` coord-long-review: door-not-on-plan — passed as recorded, fails on what was delivered
- GPT-5.6 Sol `baseline` coord-long-review: callout-missing-ceiling-detail — passed as recorded, fails on what was delivered
- GPT-5.6 Sol `baseline` coord-long-review: door-not-on-plan-5 — passed as recorded, fails on what was delivered
- GPT-5.6 Sol `show_me` coord-full-review: callout-missing-detail — passed as recorded, fails on what was delivered
- GPT-5.6 Sol `show_me` coord-full-review: spec-section-missing — passed as recorded, fails on what was delivered
- GPT-5.6 Sol `method` coord-long-review: door-not-on-plan — passed as recorded, fails on what was delivered
- GPT-5.6 Sol `method` coord-long-review: door-not-on-plan-5 — passed as recorded, fails on what was delivered
- GPT-5.6 Sol `all_three` coord-long-review: door-not-on-plan-5 — passed as recorded, fails on what was delivered
- GPT-5.4 `baseline` coord-full-review: callout-missing-sheet — passed as recorded, fails on what was delivered
- GPT-5.4 `baseline` coord-full-review: door-not-on-plan — passed as recorded, fails on what was delivered
- GPT-5.4 `baseline` coord-long-review: door-not-scheduled — passed as recorded, fails on what was delivered
- GPT-5.4 `baseline` coord-long-review: door-not-on-plan — passed as recorded, fails on what was delivered
- GPT-5.4 `baseline` coord-long-review: footing-not-scheduled — passed as recorded, fails on what was delivered
- GPT-5.4 `baseline` coord-long-review: door-not-on-plan-5 — passed as recorded, fails on what was delivered

## After EV-44: the scorer's five fixes

The same 84 saved runs, scored again with no model call after BACKLOG EV-44 (TRACE_REVIEW section 10, F-1 to F-5):

- **F-1.** A delivered file's text starts a fresh section, so a heading at the end of the reply no longer heads the file's rows.
- **F-2.** A door number is not read at the front of a decimal (door 109 is not in "EL. 109.30").
- **F-3.** A new scored check, `no-false-alarm`, on the three callout and the three schedule tasks: it fails when the answer states a callout that resolves, a mark that is both drawn and scheduled, or a door number no sheet carries as broken or missing, in one table row or one sentence, where the word that says so is nearest to that item (or to a list it is in) rather than to a planted problem.
- **F-4.** `index-title` wants its three terms in one line or one list item.
- **F-5.** Two truth texts corrected (coord-long-callouts' sheet count, coord-drawings-vs-manual's sections); they are read by people, not scored.

Each set-up now has 101 scored checks, not 95: six tasks gained one check each. The check totals before and after are therefore not the same measure; the tasks passed are.

### GPT-5.6 Sol

| set-up | checks, before EV-44 | checks, after EV-44 | tasks passed, before | tasks passed, after |
|---|---|---|---|---|
| `baseline` | 87 of 95 | 93 of 101 | 12 of 14 | 12 of 14 |
| `show_me` | 91 of 95 | 97 of 101 | 12 of 14 | 12 of 14 |
| `method` | 89 of 95 | 95 of 101 | 12 of 14 | 12 of 14 |
| `sheet_tools` | 92 of 95 | 98 of 101 | 12 of 14 | 12 of 14 |
| `all_three` | 92 of 95 | 98 of 101 | 12 of 14 | 12 of 14 |

### GPT-5.4

| set-up | checks, before EV-44 | checks, after EV-44 | tasks passed, before | tasks passed, after |
|---|---|---|---|---|
| `baseline` | 74 of 95 | 74 of 101 | 10 of 14 | 7 of 14 |

### The two whole-set reviews: planted problems delivered

Checks passed less the two automatic checks, against the readers' hand count of the delivered logs (TRACE_REVIEW section 2). They now agree in all twelve runs.

| run | coord-full-review (13): scorer / hand | coord-long-review (19): scorer / hand |
|---|---|---|
| Sol `baseline` | 11 / 11 | 13 / 13 |
| Sol `show_me` | 10 / 10 | 18 / 18 |
| Sol `method` | 11 / 11 | 15 / 15 |
| Sol `sheet_tools` | 13 / 13 | 17 / 17 |
| Sol `all_three` | 12 / 12 | 17 / 17 |
| GPT-5.4 `baseline` | 6 / 6 | 6 / 6 |

### Verdicts that changed

- GPT-5.4 `baseline` coord-sheet-index: index-title — passed before, fails now: the reply says the index matches the set; the check had passed on two neighbouring lines of its list of the sheets (S-201's own title, then S-202 ROOF FRAMING PLAN). Now the terms must share one line or list item (F-4).
- GPT-5.4 `baseline` coord-full-review: door-not-scheduled — passed before, fails now: the delivered workbook never names door 109; the check had passed on the parapet level EL. 109.30 beside an unrelated "not" (F-2).

### The new false-alarm check

- GPT-5.6 Sol passes it in all 30 runs of the six tasks, as the readers found no false alarm in any Sol chat answer there.
- GPT-5.4 coord-callouts: fails; 4/A-501 and 6/A-501 are listed under the callouts that do not point to a view ("does not point to a view that exists on A-501") before the same lines call them valid; the readers list this among GPT-5.4's false statements (TRACE_REVIEW section 5).
- GPT-5.4 coord-schedules: passes.
- GPT-5.4 coord-scan-callouts: passes.
- GPT-5.4 coord-scan-schedules: fails; doors 201, 202, 203, 205 and 206, all tagged on A-102, are said to be scheduled but not shown on the plan it checked (A-101 only).
- GPT-5.4 coord-long-callouts: fails; 18 statements call good callouts broken (1/A-301, 2/A-301, 3/A-501, 5/A-501 on the floor plans, 4/S-501 on the framing plans), all because their sheets are "not in set".
- GPT-5.4 coord-long-schedules: fails; door 409, which no sheet carries, is said to be scheduled on A-602 and not found on A-104.

GPT-5.4's other false statements on these tasks are not about a sound callout or mark and stay unscored: on coord-scan-callouts it says no callout points to a missing view, lists 7/A-501 as resolving and then flags it, and sets 2/A-302 aside as not counted; on coord-scan-schedules, "no second-floor plan" (A-102 is in the set); on coord-callouts, sheets named as carrying callouts that carry none. Its statements on coord-revisions and coord-sheet-index are outside the check's six tasks (the index one now fails `index-title`).
