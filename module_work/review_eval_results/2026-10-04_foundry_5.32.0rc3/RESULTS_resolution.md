# Document Review suite — results

Run 2026-10-04T13:19:44 · model `GPT_5_6_SOL` · tasks 15 · arms baseline, baseline_high
Versions: geotech-staff-engineer 5.32.0rc3, planlens 0.11.0rc2, deepagents 0.7.13, langchain 1.3.18

| arm | tasks passed | checks passed | model calls | tool calls | tokens (in/out) | minutes | errors | failed calls | step caps |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 15/15 | 52/52 | 167 | 74 | 1,032,968/78,994 | 16.1 | 0 | 0 | 0 |
| baseline_high | 15/15 | 52/52 | 227 | 73 | 876,974/134,709 | 18.6 | 0 | 0 | 0 |

## By category (tasks passed)

| category | baseline | baseline_high |
|---|---|---|
| locate | 9/9 | 9/9 |
| produce | 1/1 | 1/1 |
| summarize | 5/5 | 5/5 |

## By document type (tasks passed)

| doc_type | baseline | baseline_high |
|---|---|---|
| drawing_stroke | 15/15 | 15/15 |

## Per task

| task | category | doc type | baseline | baseline_high |
|---|---|---|---|---|
| meck-driveway-notes | locate | drawing_stroke | ✓ 3/3 | ✓ 3/3 |
| meck-ramp-slopes | locate | drawing_stroke | ✓ 3/3 | ✓ 3/3 |
| meck-ramp-warning-mat | locate | drawing_stroke | ✓ 4/4 | ✓ 4/4 |
| meck-pavement-section | locate | drawing_stroke | ✓ 3/3 | ✓ 3/3 |
| meck-revision-block | summarize | drawing_stroke | ✓ 3/3 | ✓ 3/3 |
| meck-row-sidewalk | locate | drawing_stroke | ✓ 3/3 | ✓ 3/3 |
| meck-underdrain | locate | drawing_stroke | ✓ 4/4 | ✓ 4/4 |
| meck-bioretention-access | summarize | drawing_stroke | ✓ 3/3 | ✓ 3/3 |
| meck-sediment-trap-criteria | summarize | drawing_stroke | ✓ 4/4 | ✓ 4/4 |
| meck-monument | summarize | drawing_stroke | ✓ 4/4 | ✓ 4/4 |
| meck-curb-types | summarize | drawing_stroke | ✓ 3/3 | ✓ 3/3 |
| meck-trap-dimensions | locate | drawing_stroke | ✓ 4/4 | ✓ 4/4 |
| meck-bioretention-section-dims | locate | drawing_stroke | ✓ 3/3 | ✓ 3/3 |
| meck-ramp-detail-callouts | locate | drawing_stroke | ✓ 4/4 | ✓ 4/4 |
| produce-markup | produce | drawing_stroke | ✓ 4/4 | ✓ 4/4 |

## Changes against `baseline`

- no task changed outcome

## Failed checks



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
