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
