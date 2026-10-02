# Document Review suite — results

Run 2026-10-02T02:46:23 · model `GPT_5_6_SOL` · tasks 10 · arms grounded, overview
Versions: geotech-staff-engineer 5.31.0, planlens 0.10.1, deepagents 0.7.13, langchain 1.3.18

| arm | tasks passed | checks passed | model calls | tool calls | tokens (in/out) | minutes | errors | step caps |
|---|---|---|---|---|---|---|---|---|
| grounded | 10/10 | 34/34 | 180 | 189 | 1,383,499/87,139 | 11.7 | 0 | 0 |
| overview | 9/10 | 33/34 | 181 | 199 | 1,488,267/88,815 | 9.1 | 0 | 0 |

## By category (tasks passed)

| category | grounded | overview |
|---|---|---|
| locate | 5/5 | 4/5 |
| orient | 3/3 | 3/3 |
| summarize | 2/2 | 2/2 |

## By document type (tasks passed)

| doc_type | grounded | overview |
|---|---|---|
| criteria_scanned | 2/2 | 2/2 |
| criteria_text | 4/4 | 3/4 |
| long_text | 4/4 | 4/4 |

## Per task

| task | category | doc type | grounded | overview |
|---|---|---|---|---|
| ufc04-density | locate | criteria_text | ✓ 4/4 | ✓ 4/4 |
| ufc04-table-5-1 | locate | criteria_text | ✓ 3/3 | ✗ 2/3 |
| ufc04-supersedes | orient | criteria_text | ✓ 3/3 | ✓ 3/3 |
| ufc04-confined-zones | locate | criteria_text | ✓ 4/4 | ✓ 4/4 |
| ufc07-figure-1-1 | locate | criteria_scanned | ✓ 4/4 | ✓ 4/4 |
| ufc07-drilled-shaft-table | locate | criteria_scanned | ✓ 3/3 | ✓ 3/3 |
| ufc301-changes | orient | long_text | ✓ 4/4 | ✓ 4/4 |
| ufc260-appendices | orient | long_text | ✓ 3/3 | ✓ 3/3 |
| ufc260-ch12-tables | summarize | long_text | ✓ 3/3 | ✓ 3/3 |
| ufc301-asce7-chapters | summarize | long_text | ✓ 3/3 | ✓ 3/3 |

## Changes against `grounded`

- `overview` BREAKS ufc04-table-5-1 (locate, criteria_text)

## Failed checks

- `overview` ufc04-table-5-1: contains_all — missing: ['/(?<![\\w.,/-])6\\s*(?:\\"|inch(?:es)?\\b|-inch\\b|in\\.|-in\\b|in\\b(?!\\s+(?:the|a|an|this|that|these|those|which|each|all|order|addition|accordance|place|front)\\b))/']
