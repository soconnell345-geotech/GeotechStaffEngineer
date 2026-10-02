# Document Review suite — results

Run 2026-10-02T00:30:48 · model `GPT_5_6_SOL` · tasks 35 · arms baseline, lean, grounded, inline, sweep, geometry, digest
Versions: geotech-staff-engineer 5.31.0, planlens 0.10.1, deepagents 0.7.13, langchain 1.3.18

| arm | tasks passed | checks passed | model calls | tool calls | tokens (in/out) | minutes | errors | step caps |
|---|---|---|---|---|---|---|---|---|
| baseline | 33/35 | 121/123 | 960 | 316 | 3,617,588/632,677 | 49.3 | 0 | 0 |
| lean | 34/35 | 122/123 | 1263 | 314 | 3,610,794/913,482 | 85.5 | 0 | 0 |
| grounded | 32/35 | 120/123 | 1245 | 325 | 4,307,587/1,334,229 | 75.4 | 0 | 0 |
| inline | 31/35 | 119/123 | 1252 | 567 | 5,083,393/1,131,461 | 54.2 | 0 | 0 |
| sweep | 35/35 | 123/123 | 1154 | 361 | 4,851,205/1,199,965 | 85.6 | 0 | 0 |
| geometry | 33/35 | 121/123 | 1512 | 374 | 5,172,393/1,444,055 | 69.5 | 0 | 0 |
| digest | 34/35 | 122/123 | 1672 | 385 | 5,848,360/1,699,208 | 88.9 | 0 | 0 |

## By category (tasks passed)

| category | baseline | lean | grounded | inline | sweep | geometry | digest |
|---|---|---|---|---|---|---|---|
| check | 4/4 | 4/4 | 3/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| compare | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| count | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| locate | 14/15 | 14/15 | 14/15 | 11/15 | 15/15 | 14/15 | 14/15 |
| markups | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| orient | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| produce | 1/2 | 2/2 | 1/2 | 2/2 | 2/2 | 1/2 | 2/2 |
| summarize | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |

## By document type (tasks passed)

| doc_type | baseline | lean | grounded | inline | sweep | geometry | digest |
|---|---|---|---|---|---|---|---|
| calc_package | 2/2 | 2/2 | 1/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| criteria_scanned | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 |
| criteria_text | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| drawing_set | 6/6 | 6/6 | 6/6 | 5/6 | 6/6 | 6/6 | 6/6 |
| drawing_stroke | 13/15 | 14/15 | 13/15 | 12/15 | 15/15 | 13/15 | 14/15 |
| long_text | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| markup_set | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |
| submittal | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 | 1/1 |

## Per task

| task | category | doc type | baseline | lean | grounded | inline | sweep | geometry | digest |
|---|---|---|---|---|---|---|---|---|---|
| meck-driveway-notes | locate | drawing_stroke | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| meck-ramp-slopes | locate | drawing_stroke | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| meck-ramp-warning-mat | locate | drawing_stroke | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 |
| meck-pavement-section | locate | drawing_stroke | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| meck-revision-block | summarize | drawing_stroke | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| meck-row-sidewalk | locate | drawing_stroke | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| meck-underdrain | locate | drawing_stroke | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✗ 3/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 |
| meck-bioretention-access | summarize | drawing_stroke | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| meck-sediment-trap-criteria | summarize | drawing_stroke | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 |
| meck-monument | summarize | drawing_stroke | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 |
| meck-curb-types | summarize | drawing_stroke | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| meck-trap-dimensions | locate | drawing_stroke | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✗ 3/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 |
| meck-bioretention-section-dims | locate | drawing_stroke | ✓ 3/3 | ✗ 2/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| meck-ramp-detail-callouts | locate | drawing_stroke | ✗ 3/4 | ✓ 4/4 | ✗ 3/4 | ✗ 3/4 | ✓ 4/4 | ✗ 3/4 | ✗ 3/4 |
| set-sheet-index | summarize | drawing_set | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| set-3600-psi | count | drawing_set | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| set-cross-references | compare | drawing_set | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| set-find-bioretention | locate | drawing_set | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✗ 2/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| set-ncdot-vs-county | check | drawing_set | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 |
| ufc04-density | locate | criteria_text | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 |
| ufc04-table-5-1 | locate | criteria_text | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| ufc04-supersedes | orient | criteria_text | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| ufc04-confined-zones | locate | criteria_text | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 |
| ufc07-figure-1-1 | locate | criteria_scanned | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 |
| ufc07-drilled-shaft-table | locate | criteria_scanned | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| ufc301-changes | orient | long_text | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 |
| ufc260-appendices | orient | long_text | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| ufc260-ch12-tables | summarize | long_text | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| ufc301-asce7-chapters | summarize | long_text | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| calc-wall-check | check | calc_package | ✓ 5/5 | ✓ 5/5 | ✓ 5/5 | ✓ 5/5 | ✓ 5/5 | ✓ 5/5 | ✓ 5/5 |
| calc-bearing-consistency | check | calc_package | ✓ 3/3 | ✓ 3/3 | ✗ 2/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| fixture-markups | markups | markup_set | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 | ✓ 3/3 |
| fixture-duplicate-page | check | submittal | ✓ 6/6 | ✓ 6/6 | ✓ 6/6 | ✓ 6/6 | ✓ 6/6 | ✓ 6/6 | ✓ 6/6 |
| produce-markup | produce | drawing_stroke | ✗ 3/4 | ✓ 4/4 | ✗ 3/4 | ✓ 4/4 | ✓ 4/4 | ✗ 3/4 | ✓ 4/4 |
| produce-memo | produce | drawing_set | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 | ✓ 4/4 |

## Changes against `baseline`

- `lean` BREAKS meck-bioretention-section-dims (locate, drawing_stroke)
- `lean` FIXES meck-ramp-detail-callouts (locate, drawing_stroke)
- `lean` FIXES produce-markup (produce, drawing_stroke)
- `grounded` BREAKS calc-bearing-consistency (check, calc_package)
- `inline` BREAKS meck-underdrain (locate, drawing_stroke)
- `inline` BREAKS meck-trap-dimensions (locate, drawing_stroke)
- `inline` BREAKS set-find-bioretention (locate, drawing_set)
- `inline` FIXES produce-markup (produce, drawing_stroke)
- `sweep` FIXES meck-ramp-detail-callouts (locate, drawing_stroke)
- `sweep` FIXES produce-markup (produce, drawing_stroke)
- `digest` FIXES produce-markup (produce, drawing_stroke)

## Failed checks

- `baseline` meck-ramp-detail-callouts: contains_all — missing: ['edge of pavement']
- `baseline` produce-markup: file_produced — no .pdf named *marked* among ['10.31A_designer_draft_markup.pdf']
- `lean` meck-bioretention-section-dims: contains_all — missing: ['/(?<![\\w.,/-])10\\s*(?:\'|ft\\b|-ft\\b|feet|-foot|foot)(?:\\s*-?\\s*0\\s*(?:\\"|inch(?:es)?\\b|in\\.|in\\b))?(?!\\s*-?\\s*\\d)[^\\w\\n]{0,4}(?:min\\b|min\\.|minimum)|(?:\\bmin\\b\\.?|\\bminimum|\\bat least|\\bno less than|\\bnot less than)[^.,;)\\]\\n\\d]{0,25}(?<![\\w.,/-])10\\s*(?:\'|f
- `grounded` meck-ramp-detail-callouts: contains_all — missing: ['edge of pavement']
- `grounded` calc-bearing-consistency: contains_all — missing: ['/(?<![\\w.,/-])1195(?!\\d)/ | /(?<![\\w.,/-])1,195(?!\\d)/']
- `grounded` produce-markup: file_produced — no .pdf named *marked* among ['10.31A_designer_draft_markup.pdf']
- `inline` meck-underdrain: contains_all — missing: ['m278 | m 278']
- `inline` meck-trap-dimensions: the two unlettered minimums — missing: ['/(?<![\\w.,/-])21\\s*(?:\\"|inch(?:es)?\\b|-inch\\b|in\\.|-in\\b|in\\b)[^\\w\\n]{0,4}(?:min\\b|min\\.|minimum)|(?:\\bmin\\b\\.?|\\bminimum|\\bat least|\\bno less than|\\bnot less than)[^.,;)\\]\\n\\d]{0,25}(?<![\\w.,/-])21\\s*(?:\\"|inch(?:es)?\\b|-inch\\b|in\\.|-in\\b|in\\b)(?![^\\w\\n]{
- `inline` meck-ramp-detail-callouts: contains_all — missing: ['edge of pavement']
- `inline` set-find-bioretention: contains_all — missing: ["/(?<![\\w.,/-])4'\\-0(?!\\d)/ | /(?<![\\w.,/-])(?<!note )(?<!notes )4\\s*(?:'|ft\\b|-ft\\b|feet|-foot|foot)/"]
- `geometry` meck-ramp-detail-callouts: contains_all — missing: ['edge of pavement']
- `geometry` produce-markup: file_produced — no .pdf named *marked* among ['10.31A_designer_review_draft.pdf']
- `digest` meck-ramp-detail-callouts: does not read the stacked 3/4 as 34 — contains: ['/(?<![\\d/.-])34\\s*(?:\\"|in\\b|inch)/']
