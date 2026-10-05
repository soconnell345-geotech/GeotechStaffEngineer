# Document Review suite — results

Run 2026-10-04T14:51:58 · model `GPT_5_6_SOL` · tasks 2 · arms baseline, baseline_r2, baseline_r3, sweep, sweep_r2, sweep_r3, minimal, minimal_r2, minimal_r3, baseline_high
Versions: geotech-staff-engineer 5.32.0rc3, planlens 0.11.0rc2, deepagents 0.7.13, langchain 1.3.18

| arm | tasks passed | checks passed | model calls | tool calls | tokens (in/out) | minutes | errors | failed calls | step caps |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 2/2 | 7/7 | 76 | 52 | 552,115/34,783 | 3.1 | 0 | 0 | 0 |
| baseline_r2 | 1/2 | 6/7 | 127 | 62 | 1,174,002/59,038 | 22.0 | 0 | 0 | 0 |
| baseline_r3 | 2/2 | 7/7 | 92 | 59 | 654,757/65,017 | 5.4 | 0 | 0 | 0 |
| sweep | 2/2 | 7/7 | 137 | 28 | 1,197,651/52,566 | 4.7 | 0 | 0 | 0 |
| sweep_r2 | 2/2 | 7/7 | 71 | 12 | 636,174/43,952 | 3.5 | 0 | 0 | 0 |
| sweep_r3 | 1/2 | 6/7 | 76 | 19 | 664,342/23,805 | 2.9 | 0 | 0 | 0 |
| minimal | 1/2 | 6/7 | 151 | 359 | 3,180,263/40,375 | 5.6 | 0 | 0 | 0 |
| minimal_r2 | 1/2 | 6/7 | 155 | 347 | 2,710,834/39,827 | 5.6 | 0 | 0 | 0 |
| minimal_r3 | 2/2 | 7/7 | 226 | 461 | 3,564,702/58,442 | 23.0 | 0 | 0 | 0 |
| baseline_high | 1/2 | 6/7 | 317 | 60 | 700,286/117,921 | 20.8 | 0 | 0 | 0 |

## By category (tasks passed)

| category | baseline | baseline_r2 | baseline_r3 | sweep | sweep_r2 | sweep_r3 | minimal | minimal_r2 | minimal_r3 | baseline_high |
|---|---|---|---|---|---|---|---|---|---|---|
| count | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 | 1/1 | 0/1 |
| produce | 1/1 | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |

## By document type (tasks passed)

| doc_type | baseline | baseline_r2 | baseline_r3 | sweep | sweep_r2 | sweep_r3 | minimal | minimal_r2 | minimal_r3 | baseline_high |
|---|---|---|---|---|---|---|---|---|---|---|
| drawing_set | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 0/1 | 0/1 | 0/1 | 1/1 | 0/1 |
| drawing_stroke | 1/1 | 0/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |

## Per task

| task | category | doc type | baseline | baseline_r2 | baseline_r3 | sweep | sweep_r2 | sweep_r3 | minimal | minimal_r2 | minimal_r3 | baseline_high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| set-long-rare-tag | count | drawing_set | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✗ 2/3 | ✗ 2/3 | ✗ 2/3 | ✓ 3/3 | ✗ 2/3 |
| produce-circle-tags | produce | drawing_stroke | ✓ 4/4 | ✗ 3/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 |

## Changes against `baseline`

- `baseline_r2` BREAKS produce-circle-tags (produce, drawing_stroke)
- `sweep_r3` BREAKS set-long-rare-tag (count, drawing_set)
- `minimal` BREAKS set-long-rare-tag (count, drawing_set)
- `minimal_r2` BREAKS set-long-rare-tag (count, drawing_set)
- `baseline_high` BREAKS set-long-rare-tag (count, drawing_set)

## Failed checks

- `baseline_r2` produce-circle-tags: the rings sit on the GCE callouts — tag_set_GCE_penetrations_marked.pdf: 0/7 targets marked (recall 0.00 >= 0.8), 0/7 marks on a target (precision 0.00 >= 0.8)
- `sweep_r3` set-long-rare-tag: names pages 4, 12 and 20 and little else — pages named []; expected [4, 12, 20]: recall 0.00 (>= 1.0), precision 0.00 (>= 0.75)
- `minimal` set-long-rare-tag: names pages 4, 12 and 20 and little else — pages named [2, 3, 4, 8, 12, 20]; expected [4, 12, 20]: recall 1.00 (>= 1.0), precision 0.50 (>= 0.75)
- `minimal_r2` set-long-rare-tag: names pages 4, 12 and 20 and little else — pages named [4, 20]; expected [4, 12, 20]: recall 0.67 (>= 1.0), precision 1.00 (>= 0.75)
- `baseline_high` set-long-rare-tag: names pages 4, 12 and 20 and little else — pages named [4, 9, 12, 19, 20, 22]; expected [4, 12, 20]: recall 1.00 (>= 1.0), precision 0.50 (>= 0.75)


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
