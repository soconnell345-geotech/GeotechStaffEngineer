# Document Review suite — results

Run 2026-10-04T13:19:44 · model `GPT_5_6_SOL` · tasks 35 · arms baseline, sweep, minimal
Versions: geotech-staff-engineer 5.32.0rc3, planlens 0.11.0rc2, deepagents 0.7.13, langchain 1.3.18

| arm | tasks passed | checks passed | model calls | tool calls | tokens (in/out) | minutes | errors | failed calls | step caps |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 35/35 | 123/123 | 464 | 309 | 4,530,060/359,960 | 45.5 | 0 | 0 | 0 |
| sweep | 35/35 | 123/123 | 1033 | 285 | 10,809,320/1,116,121 | 124.6 | 0 | 0 | 0 |
| minimal | 33/35 | 121/123 | 1045 | 2705 | 18,450,830/310,380 | 67.6 | 0 | 0 | 0 |

## By category (tasks passed)

| category | baseline | sweep | minimal |
|---|---|---|---|
| check | 4/4 | 4/4 | 4/4 |
| compare | 1/1 | 1/1 | 1/1 |
| count | 1/1 | 1/1 | 0/1 |
| locate | 15/15 | 15/15 | 15/15 |
| markups | 1/1 | 1/1 | 0/1 |
| orient | 3/3 | 3/3 | 3/3 |
| produce | 2/2 | 2/2 | 2/2 |
| summarize | 8/8 | 8/8 | 8/8 |

## By document type (tasks passed)

| doc_type | baseline | sweep | minimal |
|---|---|---|---|
| calc_package | 2/2 | 2/2 | 2/2 |
| criteria_scanned | 2/2 | 2/2 | 2/2 |
| criteria_text | 4/4 | 4/4 | 4/4 |
| drawing_set | 6/6 | 6/6 | 5/6 |
| drawing_stroke | 15/15 | 15/15 | 15/15 |
| long_text | 4/4 | 4/4 | 4/4 |
| markup_set | 1/1 | 1/1 | 0/1 |
| submittal | 1/1 | 1/1 | 1/1 |

## Per task

| task | category | doc type | baseline | sweep | minimal |
|---|---|---|---|---|---|
| meck-driveway-notes | locate | drawing_stroke | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| meck-ramp-slopes | locate | drawing_stroke | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| meck-ramp-warning-mat | locate | drawing_stroke | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 |
| meck-pavement-section | locate | drawing_stroke | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| meck-revision-block | summarize | drawing_stroke | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| meck-row-sidewalk | locate | drawing_stroke | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| meck-underdrain | locate | drawing_stroke | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 |
| meck-bioretention-access | summarize | drawing_stroke | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| meck-sediment-trap-criteria | summarize | drawing_stroke | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 |
| meck-monument | summarize | drawing_stroke | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 |
| meck-curb-types | summarize | drawing_stroke | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| meck-trap-dimensions | locate | drawing_stroke | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 |
| meck-bioretention-section-dims | locate | drawing_stroke | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| meck-ramp-detail-callouts | locate | drawing_stroke | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 |
| set-sheet-index | summarize | drawing_set | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| set-3600-psi | count | drawing_set | ✓ 3/3 | ✓ 3/3 | ✗ 2/3 |
| set-cross-references | compare | drawing_set | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| set-find-bioretention | locate | drawing_set | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| set-ncdot-vs-county | check | drawing_set | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 |
| ufc04-density | locate | criteria_text | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 |
| ufc04-table-5-1 | locate | criteria_text | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| ufc04-supersedes | orient | criteria_text | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| ufc04-confined-zones | locate | criteria_text | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 |
| ufc07-figure-1-1 | locate | criteria_scanned | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 |
| ufc07-drilled-shaft-table | locate | criteria_scanned | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| ufc301-changes | orient | long_text | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 |
| ufc260-appendices | orient | long_text | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| ufc260-ch12-tables | summarize | long_text | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| ufc301-asce7-chapters | summarize | long_text | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| calc-wall-check | check | calc_package | ✓ 5/5 | ✓ 5/5 | ✓ 5/5 |
| calc-bearing-consistency | check | calc_package | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| fixture-markups | markups | markup_set | ✓ 3/3 | ✓ 3/3 | ✗ 2/3 |
| fixture-duplicate-page | check | submittal | ✓ 6/6 | ✓ 6/6 | ✓ 6/6 |
| produce-markup | produce | drawing_stroke | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 |
| produce-memo | produce | drawing_set | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 |

## Changes against `baseline`

- `minimal` BREAKS set-3600-psi (count, drawing_set)
- `minimal` BREAKS fixture-markups (markups, markup_set)

## Failed checks

- `minimal` set-3600-psi: set_match — recall 0.33 (>= 1.0), precision 1.00 (>= 0.75); missing ['10.25A', '20.00B']; extra []
- `minimal` fixture-markups: contains_all — missing: ['contractor b']


---
## Foundry run notes (added by the Foundry runner, not by the package)
- Package: geotech-staff-engineer 5.32.0rc2 + planlens 0.11.0rc1 from the owner's wheels, loaded ahead of the
  repo's 5.31.0 on sys.path (no package edits).
- numpy: the wheels declare numpy>=2.0; this environment has numpy 1.26.4 (owner-approved; same as the 5.31 runs).
- Model: GPT-5.6 Sol via the Responses route (FullResResponsesChatModel): detail "original" is sent as AUTO = full
  size (measured to >= 4096 px); "high" passes through as HIGH (~692 image tokens). Retry layer on infrastructure
  errors only. Not comparable with the 5.31 runs (chat route, ~768 px).
- Concurrency cap applied in GLUE (rc2 has no GEOTECH_VISION_MAX_INFLIGHT): at most 8 model calls in flight per
  process, and the LMS client's connection pool raised from 10 to 64. Added after the two new tasks aborted their
  process (OpenSSL FIPS self-test abort, exit -6) under unbounded fan-out; each task runs in its own child process.

- rc3 / planlens 0.11.0rc2: OpenCV is test-loaded in a throwaway child; on this host loading it kills the child
  (OpenSSL FIPS self-test abort), so find_like is HIDDEN from the agent in every new-task run below.
- Package cap of 8 vision calls in flight (GEOTECH_VISION_MAX_INFLIGHT) plus the glue cap of 8 model calls in flight.
- Original-task runs were saved under rc2 and RESCORED here with rescore=True (no model calls).
