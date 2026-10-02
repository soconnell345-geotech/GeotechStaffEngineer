# Document Review suite — results

Run 2026-10-01T13:40:33 · model `GPT_5_4` · tasks 8 · arms baseline, lean
Versions: geotech-staff-engineer 5.31.0, planlens 0.10.1, deepagents 0.7.13, langchain 1.3.18

| arm | tasks passed | checks passed | model calls | tool calls | tokens (in/out) | minutes | errors | step caps |
|---|---|---|---|---|---|---|---|---|
| baseline | 8/8 | 28/28 | 111 | 43 | 587,615/18,160 | 5.3 | 0 | 0 |
| lean | 8/8 | 28/28 | 115 | 46 | 515,508/17,451 | 4.9 | 0 | 0 |

## By category (tasks passed)

| category | baseline | lean |
|---|---|---|
| check | 1/1 | 1/1 |
| count | 1/1 | 1/1 |
| locate | 3/3 | 3/3 |
| markups | 1/1 | 1/1 |
| produce | 1/1 | 1/1 |
| summarize | 1/1 | 1/1 |

## By document type (tasks passed)

| doc_type | baseline | lean |
|---|---|---|
| calc_package | 1/1 | 1/1 |
| criteria_scanned | 1/1 | 1/1 |
| criteria_text | 1/1 | 1/1 |
| drawing_set | 1/1 | 1/1 |
| drawing_stroke | 3/3 | 3/3 |
| markup_set | 1/1 | 1/1 |

## Per task

| task | category | doc type | baseline | lean |
|---|---|---|---|---|
| meck-ramp-slopes | locate | drawing_stroke | ✓ 3/3 | ✓ 3/3 |
| meck-revision-block | summarize | drawing_stroke | ✓ 3/3 | ✓ 3/3 |
| set-3600-psi | count | drawing_set | ✓ 3/3 | ✓ 3/3 |
| ufc04-table-5-1 | locate | criteria_text | ✓ 3/3 | ✓ 3/3 |
| ufc07-figure-1-1 | locate | criteria_scanned | ✓ 4/4 | ✓ 4/4 |
| calc-wall-check | check | calc_package | ✓ 5/5 | ✓ 5/5 |
| fixture-markups | markups | markup_set | ✓ 3/3 | ✓ 3/3 |
| produce-markup | produce | drawing_stroke | ✓ 4/4 | ✓ 4/4 |

## Changes against `baseline`

- no task changed outcome

## Failed checks

