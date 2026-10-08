# Document Review suite — results

Run 2026-10-07T19:30:43 · model `GPT_5_6_SOL` · tasks 2 · arms baseline, baseline_r2, baseline_r3
Versions: geotech-staff-engineer 5.32.1rc3, planlens 0.12.0rc1, deepagents 0.7.13, langchain 1.3.18

| arm | tasks passed | checks passed | model calls | tool calls | tokens (in/out) | minutes | errors | failed calls | step caps |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 2/2 | 8/8 | 51 | 17 | 250,564/20,534 | 3.2 | 0 | 0 | 0 |
| baseline_r2 | 2/2 | 8/8 | 66 | 22 | 339,603/22,023 | 4.4 | 0 | 0 | 0 |
| baseline_r3 | 2/2 | 8/8 | 95 | 28 | 527,492/63,310 | 7.6 | 0 | 0 | 0 |

## By category (tasks passed)

| category | baseline | baseline_r2 | baseline_r3 |
|---|---|---|---|
| produce | 2/2 | 2/2 | 2/2 |

## By document type (tasks passed)

| doc_type | baseline | baseline_r2 | baseline_r3 |
|---|---|---|---|
| drawing_stroke | 2/2 | 2/2 | 2/2 |

## Per task

| task | category | doc type | baseline | baseline_r2 | baseline_r3 |
|---|---|---|---|---|---|
| produce-markup | produce | drawing_stroke | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 |
| produce-circle-tags | produce | drawing_stroke | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 |

## Changes against `baseline`

- no task changed outcome

## Failed checks

