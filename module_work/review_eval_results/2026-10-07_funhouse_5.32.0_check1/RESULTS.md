# Document Review suite — results

Run 2026-10-07T13:36:14 · model `funhouse-gpt-high` · tasks 2 · arms baseline, baseline_r2, baseline_r3
Versions: geotech-staff-engineer 5.32.0, planlens 0.11.0, deepagents 0.7.8, langchain 1.3.16

| arm | tasks passed | checks passed | model calls | tool calls | tokens (in/out) | minutes | errors | failed calls | step caps |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 1/2 | 6/8 | 55 | 29 | 296,988/9,041 | 3.4 | 0 | 0 | 0 |
| baseline_r2 | 1/2 | 6/8 | 32 | 26 | 197,641/5,095 | 2.0 | 0 | 0 | 0 |
| baseline_r3 | 1/2 | 7/8 | 36 | 23 | 213,516/5,723 | 2.1 | 0 | 0 | 0 |

## By category (tasks passed)

| category | baseline | baseline_r2 | baseline_r3 |
|---|---|---|---|
| produce | 1/2 | 1/2 | 1/2 |

## By document type (tasks passed)

| doc_type | baseline | baseline_r2 | baseline_r3 |
|---|---|---|---|
| drawing_stroke | 1/2 | 1/2 | 1/2 |

## Per task

| task | category | doc type | baseline | baseline_r2 | baseline_r3 |
|---|---|---|---|---|---|
| produce-markup | produce | drawing_stroke | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 |
| produce-circle-tags | produce | drawing_stroke | ✗ 2/4 | ✗ 2/4 | ✗ 3/4 |

## Changes against `baseline`

- no task changed outcome

## Failed checks

- `baseline` produce-circle-tags: the rings sit on the GCE callouts — tag_set_marked.pdf: 0/7 targets marked (recall 0.00 >= 0.8), 0/7 marks on a target (precision 0.00 >= 0.8)
- `baseline` produce-circle-tags: no 0-based page citation — contains: ['/\\b(?:page|p\\.|pg\\.?)\\s*0\\b/']
- `baseline_r2` produce-circle-tags: the rings sit on the GCE callouts — tag_set_marked.pdf: 1/7 targets marked (recall 0.14 >= 0.8), 1/1 marks on a target (precision 1.00 >= 0.8)
- `baseline_r2` produce-circle-tags: no 0-based page citation — contains: ['/\\b(?:page|p\\.|pg\\.?)\\s*0\\b/']
- `baseline_r3` produce-circle-tags: the rings sit on the GCE callouts — tag_set_marked.pdf: 2/7 targets marked (recall 0.29 >= 0.8), 2/3 marks on a target (precision 0.67 >= 0.8)
