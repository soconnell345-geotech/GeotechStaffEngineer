# Document Review suite — results

Run 2026-10-07T19:25:25 · model `GPT_5_4` · tasks 2 · arms baseline, baseline_r2, baseline_r3
Versions: geotech-staff-engineer 5.32.1rc3, planlens 0.12.0rc1, deepagents 0.7.13, langchain 1.3.18

| arm | tasks passed | checks passed | model calls | tool calls | tokens (in/out) | minutes | errors | failed calls | step caps |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 1/2 | 7/8 | 82 | 33 | 485,422/8,627 | 3.7 | 0 | 0 | 0 |
| baseline_r2 | 2/2 | 8/8 | 62 | 23 | 353,884/7,598 | 2.9 | 0 | 0 | 0 |
| baseline_r3 | 2/2 | 8/8 | 67 | 23 | 398,824/7,665 | 3.2 | 0 | 0 | 0 |

## By category (tasks passed)

| category | baseline | baseline_r2 | baseline_r3 |
|---|---|---|---|
| produce | 1/2 | 2/2 | 2/2 |

## By document type (tasks passed)

| doc_type | baseline | baseline_r2 | baseline_r3 |
|---|---|---|---|
| drawing_stroke | 1/2 | 2/2 | 2/2 |

## Per task

| task | category | doc type | baseline | baseline_r2 | baseline_r3 |
|---|---|---|---|---|---|
| produce-markup | produce | drawing_stroke | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 |
| produce-circle-tags | produce | drawing_stroke | ✗ 3/4 | ✓ 4/4 | ✓ 4/4 |

## Changes against `baseline`

- `baseline_r2` FIXES produce-circle-tags (produce, drawing_stroke)
- `baseline_r3` FIXES produce-circle-tags (produce, drawing_stroke)

## Failed checks

- `baseline` produce-circle-tags: the rings sit on the GCE callouts — tag_set_marked_GCE_callouts.pdf: 3/7 targets marked (recall 0.43 >= 0.8), 3/3 marks on a target (precision 1.00 >= 0.8)
