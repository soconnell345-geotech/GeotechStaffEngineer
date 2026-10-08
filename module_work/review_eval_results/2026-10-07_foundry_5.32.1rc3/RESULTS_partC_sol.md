# Document Review suite — results

Run 2026-10-07T20:49:03 · model `GPT_5_6_SOL` · tasks 37 · arms baseline
Versions: geotech-staff-engineer 5.32.1rc3, planlens 0.12.0rc1, deepagents 0.7.13, langchain 1.3.18

| arm | tasks passed | checks passed | model calls | tool calls | tokens (in/out) | minutes | errors | failed calls | step caps |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 37/37 | 130/130 | 977 | 317 | 5,370,128/629,269 | 76.5 | 0 | 1 | 0 |

## By category (tasks passed)

| category | baseline |
|---|---|
| check | 4/4 |
| compare | 1/1 |
| count | 2/2 |
| locate | 15/15 |
| markups | 1/1 |
| orient | 3/3 |
| produce | 3/3 |
| summarize | 8/8 |

## By document type (tasks passed)

| doc_type | baseline |
|---|---|
| calc_package | 2/2 |
| criteria_scanned | 2/2 |
| criteria_text | 4/4 |
| drawing_set | 7/7 |
| drawing_stroke | 16/16 |
| long_text | 4/4 |
| markup_set | 1/1 |
| submittal | 1/1 |

## Per task

| task | category | doc type | baseline |
|---|---|---|---|
| meck-driveway-notes | locate | drawing_stroke | ✓ 3/3 |
| meck-ramp-slopes | locate | drawing_stroke | ✓ 3/3 |
| meck-ramp-warning-mat | locate | drawing_stroke | ✓ 4/4 |
| meck-pavement-section | locate | drawing_stroke | ✓ 3/3 |
| meck-revision-block | summarize | drawing_stroke | ✓ 3/3 |
| meck-row-sidewalk | locate | drawing_stroke | ✓ 3/3 |
| meck-underdrain | locate | drawing_stroke | ✓ 4/4 |
| meck-bioretention-access | summarize | drawing_stroke | ✓ 3/3 |
| meck-sediment-trap-criteria | summarize | drawing_stroke | ✓ 4/4 |
| meck-monument | summarize | drawing_stroke | ✓ 4/4 |
| meck-curb-types | summarize | drawing_stroke | ✓ 3/3 |
| meck-trap-dimensions | locate | drawing_stroke | ✓ 4/4 |
| meck-bioretention-section-dims | locate | drawing_stroke | ✓ 3/3 |
| meck-ramp-detail-callouts | locate | drawing_stroke | ✓ 4/4 |
| set-sheet-index | summarize | drawing_set | ✓ 3/3 |
| set-3600-psi | count | drawing_set | ✓ 3/3 |
| set-cross-references | compare | drawing_set | ✓ 3/3 |
| set-find-bioretention | locate | drawing_set | ✓ 3/3 |
| set-ncdot-vs-county | check | drawing_set | ✓ 4/4 |
| ufc04-density | locate | criteria_text | ✓ 4/4 |
| ufc04-table-5-1 | locate | criteria_text | ✓ 3/3 |
| ufc04-supersedes | orient | criteria_text | ✓ 3/3 |
| ufc04-confined-zones | locate | criteria_text | ✓ 4/4 |
| ufc07-figure-1-1 | locate | criteria_scanned | ✓ 4/4 |
| ufc07-drilled-shaft-table | locate | criteria_scanned | ✓ 3/3 |
| ufc301-changes | orient | long_text | ✓ 4/4 |
| ufc260-appendices | orient | long_text | ✓ 3/3 |
| ufc260-ch12-tables | summarize | long_text | ✓ 3/3 |
| ufc301-asce7-chapters | summarize | long_text | ✓ 3/3 |
| calc-wall-check | check | calc_package | ✓ 5/5 |
| calc-bearing-consistency | check | calc_package | ✓ 3/3 |
| fixture-markups | markups | markup_set | ✓ 3/3 |
| fixture-duplicate-page | check | submittal | ✓ 6/6 |
| produce-markup | produce | drawing_stroke | ✓ 4/4 |
| set-long-rare-tag | count | drawing_set | ✓ 3/3 |
| produce-circle-tags | produce | drawing_stroke | ✓ 4/4 |
| produce-memo | produce | drawing_set | ✓ 4/4 |

## Failed checks

