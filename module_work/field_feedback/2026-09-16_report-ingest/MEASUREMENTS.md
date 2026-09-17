# Report-ingest measurements (private ledger)

Measurements behind the planlens `feature/report-ingest-wp0` branch. Documents
are named by corpus ID only. Page indexes are 0-based.

Runner: `GeotechStaffEngineer/.venv/Scripts/python`, editable planlens
checkout. Corpus: R15 (202 pages), R28 (455), R36 (94), R04 (97). The
260-page submittal from the 2026-09-09 drop is used as the regression
document, because the published thresholds were measured on it.

---

## Fix 1 — image-duplicate over-claim on scanned pages

### 1.1 What the old rule claimed

| document | pages | image-rule claims | text-rule claims |
|---|---|---|---|
| R15 | 202 | **69** (pages 112-183, all onto 111 or onto another page in the same run) | 0 |
| R28 | 455 | 0 | 27 |
| R36 | 94 | 0 | 0 |
| R04 | 97 | 0 | 0 |
| submittal | 260 | 0 | 0 |

R15 pages 111-183 are a 73-page laboratory appendix, A4 (595x842), kind
`scanned`. Rendered at 100 dpi and inspected: 111 and 150 are sieve /
Atterberg result sheets for DIFFERENT samples and depths, with different
tabulated values and different particle-size curves (150's chart carries no
plotted line at all). They share the printed form, the logo band, the heading,
the table ruling and a large diagonal grey watermark.

The claim set is pages 112-183 inclusive (72 pages) less 116, 120 and 133,
which happened to sit 3 or more bits from every earlier sheet: 69 claims. 53
of them name page 111; the other 16 name a nearer earlier sheet (115 eight
times, 129 three, 113 twice, 116 / 122 / 141 once each). Page 184 is outside
the set, 17 bits from 111.

### 1.2 Why the 9x8 grid cannot do it

dHash distance to page 111, 64-bit grid, R15 111-183: 0 on 20 pages, 1-2 on
most of the rest, maximum 5. Grid spread 47-53 of 255 (the old
`MIN_GRID_SPREAD` floor is 32, so every page passed). Ink fraction at 50 dpi:
0.067-0.073 below grey 200, 0.018-0.020 below grey 128.

All 2,628 pairs the 73 sheets make, by candidate rule:

| candidate | pairs within 2 bits | closest pair |
|---|---|---|
| 8x8 = 64 bits, plain (the shipped rule) | 1,260 / 2,628 | 0 |
| 8x8, ink-normalised (threshold at mid-range, then dHash) | 1,763 / 2,628 | 0 |
| 16x16 = 256 bits, plain | **0 / 2,628** | **5** |
| 16x16, ink-normalised | 1 / 2,628 | 2 |
| 24x24 = 576 bits, plain | 0 / 2,628 | 11 |
| block-wise ink correlation, 16x16 | n/a | correlation 0.79-0.99, no usable floor |

Rejected candidates and why:

- **(a) ink-normalised hash.** Worse than doing nothing at 8x8 (1,763 against
  1,260) and worse than plain at 16x16 (closest 2 against 5). Thresholding
  removes the watermark and the paper tone, but those were never the problem:
  the template's rules and boxes survive the threshold at full weight, and the
  handwritten/typed data does not. Contrast is not the axis that separates
  these pages.
- **(b) two independent signals.** Not needed once the grid is finer, and it
  doubles the render cost for a decision one signal already makes with 3 bits
  of margin.
- **(c) ink-fraction floor alone.** The lab sheets sit at 0.019 ink below grey
  128 and near-blank divider pages at 0.003. A floor between them withholds
  the hash from the dividers but leaves the lab sheets colliding at 64 bits, so
  it does not fix the reported problem. Retained in a different form as the
  confident-bit floor below.
- **(d) tighter distance for scanned pages only.** The closest different pair
  at 64 bits is 0. There is no tighter threshold than 0.

Chosen: **16x16**. First grid that separates the population; 24x24 buys 6 more
bits of margin for 2.25x the pixels and no decision change.

### 1.3 The emptiness gate had to change too

At 16x16 the old grey-RANGE floor lets near-blank pages through, because a
printed border spans the full range by itself. New claims the finer grid would
have made with the old floor:

| document | pair | distance (256) | what they are |
|---|---|---|---|
| R36 | 73 & 90 | 0 | appendix divider "APPENDIX D" vs "APPENDIX E" — different pages |
| R36 | 26 & 50 | 1 | two more dividers |

Replacement gate: count bits whose two grid cells differ by more than 16 grey
levels (`INK_CONTRAST`), require at least `MIN_CONFIDENT_BITS`. Distribution
over all 198 pages the hash gate admits across R15, R28, R36, R04, the
submittal and the ten public Mecklenburg sheets:

| population | confident bits (of 256) |
|---|---|
| near-blank (R36 26/50/64/73/90; R04 29/57/69/77/85; R15 110/201) | 0, 1, 2, 2, 2, 2, 2, 2, 3, 4, 4, 9 |
| everything else (186 pages) | 10 (R36 82) and up, median 25 |

Gap is 4 to 9 with one page at 9 (R15 110, ink 0.0003 below grey 128 — a
near-blank page that matches nothing). Floor set at **8**.

### 1.4 After the fix

| document | image-rule claims before | after |
|---|---|---|
| R15 | 69 | **0** |
| R28 | 0 | 0 |
| R36 | 0 | 0 (the two divider pairs above are withheld, not claimed) |
| R04 | 0 | 0 |
| submittal | 0 | 0 |

Text-rule claims unchanged everywhere (R28 keeps its 27).

Regression checks on the published figures:

- Submittal closest different pair reproduces the documented 4 bits at 64 bits
  (pages 96/115 and 163/256). At 256 bits that pair is 24.
- Synthetic submittal fixture: the two D-size sheets stay 0 bits apart at 256
  as at 64, so "the picture never overrules the words" still has its case.
- /Rotate 180 on a public sheet: 42 bits of 64 before, 130 of 256 now.
- Cost, 260-page submittal: all 260 pages 1.26 s at 9x8 and 1.26 s at 17x16;
  the 13 gated pages 0.27-0.29 s either way. The render dominates, not the
  grid.
- `doc_claims_check.py` output unchanged (19 / 17 / 0 blunt-terminator
  proposals, 0 / 0 / 0 oriented) — it measures the IR layer and does not touch
  the page hash.

### 1.5 No real positive available

None of the five documents contains a page placed twice among the pages the
hash gate admits, so the positive side of the rule is pinned only by synthetic
fixtures (the same pixmap placed on two pages, at 0 bits). Worth watching for
a corpus document that has one.

---

## Fix 2 — `text_reliable`

### 2.1 The two populations

Unmapped characters (U+FFFD) as a fraction of a page's characters. Scanned
across every page of R15, R28, R36, R04 and the 260-page submittal — 1,108
pages, of which only 52 carry any unmapped character at all.

| population | pages | fraction |
|---|---|---|
| a stray glyph (a bullet, a degree sign, a logo character in a heading) | 30 | 0.0007, 0.0010 x5, 0.0011, 0.0039-0.0064 |
| a broken encoding | 22 | 0.248; 0.436, 0.436, 0.440, 0.455, 0.461, 0.461, 0.468, 0.469; 0.968-0.994 (13 pages) |

Nothing lands between 0.0064 and 0.248 — a 38x empty span. Threshold set at
**0.10**: 15x above the worst benign page, 2.5x below the mildest broken one.

The two named populations the brief asked for:

| set | pages | fraction |
|---|---|---|
| R28 calc printouts 280-289 | 8 of 10 | 0.436 to 0.469 |
| R28 calc printouts 280-289 | 2 of 10 (284, 285, kind `form`) | 0.0000 — correctly NOT flagged |
| R28 narrative 3-44 | 42 | 0.0000, every page |
| R36 narrative 2-25 | 24 | 0.0000, every page |

Neither narrative range contains a single unmapped character, so the margin
between the populations is the full width of the scale.

Text inspected at each band to confirm the call before the threshold was set:

- 0.44 (R28 280): every space is U+FFFD; the surviving words are an analysis
  program's banner and headings, unusable as a transcript.
- 0.99 (R28 223): the page extracts as nothing but replacement characters.
- 0.248 (R15 102): a laboratory receipt checklist; words readable, every space
  and every tick box U+FFFD.
- 0.006 (R28 266): a calculation cover sheet, perfectly readable.

### 2.2 What the flag catches

| document | pages flagged `text_reliable = False` |
|---|---|
| R15 | 1 (page 102, 0.248) |
| R28 | 22 (142; 169-172; 217-225; 280-283; 286-289) |
| R36 | 0 |
| R04 | 0 |
| submittal | 0 |

No false positives on the two narrative documents or on the submittal.

### 2.3 Side effects checked

Setting `needs_ocr` on an unreliable page also feeds the picture-hash gate.
Verified after the change: image-duplicate claims stay 0 on all five
documents and text-duplicate claims are unchanged (R28 keeps its 27).
`pages_without_text_layer` and `pages_with_unreliable_text` are disjoint by
construction — the first now excludes pages flagged by the second.

### 2.4 Fixture note

A fixture could NOT be built by deleting a `ToUnicode` entry: this MuPDF
build answers a missing one by substituting a font and guessing, returning
confident nonsense (glyph indices as characters) rather than U+FFFD. Same for
a Type1 font with unresolvable glyph names in `/Differences`, and for a
non-Identity CID ordering. What works is drawing CID 0xFFFD through an
Identity-H font, which the extractor decodes to the replacement character —
and the proportion of the extracted string that is undecodable is exactly the
signal the rule reads. Shipped as
`planlens.testing.build_unmapped_text_pdf(fraction=..., n_pages=...)`.

---

## WP0 -- corpus harness

Measured 2026-09-16 with the app venv and the editable planlens checkout, every report opened `di="auto"`. `needs OCR` is what the PDF text layer cannot deliver; `left` is what remains after `auto` attached a DI result; `DI used` is the pages whose text actually came from DI.

| ID | pages | planlens kinds | img dup | text dup | text_reliable False | needs OCR | left | DI | DI used | s |
|---|---|---|---|---|---|---|---|---|---|---|
| R01 | 85 | text 41, form 36, mixed 6, figure 2 | 0 | 0 | 0 | 0 | 0 | n | 0 | 6.9 |
| R02 | 296 | form 186, text 55, mixed 27, figure 24, drawing_sheet 3, blank 1 | 0 | 0 | 34 | 34 | 34 | n | 0 | 26.0 |
| R03 | 322 | form 103, scanned 80, figure 73, text 40, mixed 25, drawing_sheet 1 | 0 | 1 | 0 | 80 | 80 | n | 0 | 25.7 |
| R04 | 97 | mixed 26, text 25, form 23, blank 15, figure 4, scanned 4 | 0 | 0 | 0 | 4 | 4 | n | 0 | 1.8 |
| R05 | 15 | text 14, mixed 1 | 0 | 0 | 0 | 0 | 0 | n | 0 | 0.1 |
| R06 | 124 | form 52, text 48, mixed 10, blank 7, figure 7 | 0 | 0 | 0 | 0 | 0 | n | 0 | 2.9 |
| R07 | 34 | text 16, form 9, mixed 6, scanned 3 | 0 | 0 | 0 | 3 | 3 | n | 0 | 1.0 |
| R08 | 25 | text 18, mixed 3, figure 2, form 2 | 0 | 0 | 0 | 0 | 0 | n | 0 | 0.5 |
| R09 | 151 | figure 143, scanned 5, mixed 3 | 0 | 0 | 0 | 5 | 0 | y 151 | 5 | 4.9 |
| R10 | 155 | figure 108, mixed 45, scanned 2 | 0 | 0 | 0 | 2 | 0 | y 155 | 2 | 3.9 |
| R11 | 156 | figure 146, drawing_sheet 6, mixed 4 | 0 | 1 | 0 | 6 | 0 | y 156 | 6 | 4.2 |
| R12 | 159 | figure 129, mixed 30 | 0 | 0 | 0 | 0 | 0 | y 159 | 0 | 2.8 |
| R13 | 197 | scanned 197 | 0 | 0 | 0 | 197 | 0 | y 197 | 197 | 9.0 |
| R14 | 48 | figure 31, mixed 16, scanned 1 | 0 | 0 | 0 | 1 | 0 | y 48 | 1 | 0.8 |
| R15 | 202 | scanned 77, text 50, form 42, figure 17, mixed 16 | 0 | 0 | 1 | 78 | 0 | y 202 | 78 | 13.9 |
| R16 | 426 | text 280, form 61, scanned 53, mixed 23, figure 9 | 0 | 1 | 0 | 53 | 0 | y 426 | 53 | 17.3 |
| R17 | 221 | form 98, text 69, mixed 41, figure 10, scanned 3 | 0 | 0 | 0 | 3 | 3 | n | 0 | 7.5 |
| R18 | 64 | text 35, mixed 20, form 5, figure 4 | 0 | 0 | 0 | 0 | 0 | y 64 | 0 | 1.4 |
| R19 | 275 | scanned 264, figure 8, mixed 2, form 1 | 0 | 0 | 0 | 265 | 0 | y 275 | 265 | 15.7 |
| R20 | 370 | text 231, form 95, mixed 39, figure 5 | 0 | 0 | 0 | 0 | 0 | y 370 | 0 | 14.2 |
| R21 | 729 | text 383, form 173, mixed 95, scanned 52, figure 26 | 0 | 40 | 0 | 52 | 0 | y 729 | 52 | 27.8 |
| R22 | 163 | figure 57, text 48, mixed 33, form 21, scanned 4 | 0 | 13 | 1 | 5 | 0 | y 163 | 5 | 5.1 |
| R23 | 397 | text 148, form 96, scanned 55, figure 46, mixed 44, drawing_sheet 5, blank 3 | 0 | 0 | 1 | 56 | 0 | y 397 | 56 | 27.7 |
| R24 | 154 | text 72, form 51, mixed 14, figure 11, drawing_sheet 6 | 0 | 0 | 0 | 1 | 0 | y 154 | 1 | 3.9 |
| R25 | 152 | form 91, text 37, drawing_sheet 11, figure 10, mixed 3 | 0 | 0 | 0 | 0 | 0 | y 152 | 0 | 15.1 |
| R26 | 112 | text 92, form 16, mixed 4 | 0 | 0 | 0 | 0 | 0 | y 112 | 0 | 1.6 |
| R27 | 92 | form 43, text 33, mixed 13, figure 2, scanned 1 | 0 | 0 | 0 | 1 | 0 | y 92 | 1 | 5.6 |
| R28 | 455 | text 200, form 135, mixed 69, figure 46, scanned 5 | 0 | 27 | 22 | 28 | 0 | y 455 | 28 | 16.7 |
| R29 | 131 | text 43, figure 26, mixed 26, form 20, scanned 16 | 0 | 0 | 0 | 16 | 0 | y 131 | 16 | 2.6 |
| R30 | 556 | form 254, text 170, figure 72, mixed 57, scanned 3 | 0 | 1 | 0 | 3 | 0 | y 556 | 3 | 36.1 |
| R31 | 499 | text 227, form 158, figure 50, mixed 42, scanned 22 | 0 | 1 | 15 | 37 | 37 | n | 0 | 16.0 |
| R32 | 254 | text 92, form 80, mixed 40, figure 31, scanned 11 | 0 | 0 | 58 | 69 | 69 | n | 0 | 6.6 |
| R33 | 74 | scanned 39, text 18, form 9, mixed 7, figure 1 | 0 | 0 | 0 | 39 | 39 | n | 0 | 2.8 |
| R34 | 312 | text 127, form 119, mixed 46, figure 19, scanned 1 | 0 | 0 | 0 | 1 | 1 | n | 0 | 10.7 |
| R35 | 153 | figure 118, mixed 29, scanned 6 | 0 | 0 | 0 | 6 | 6 | n | 0 | 2.1 |
| R36 | 94 | text 48, form 29, mixed 9, figure 6, scanned 2 | 0 | 0 | 0 | 2 | 2 | n | 0 | 1.5 |
| R37 | 48 | text 19, mixed 12, form 12, figure 5 | 0 | 1 | 0 | 0 | 0 | n | 0 | 0.9 |
| R38 | 32 | scanned 32 | 0 | 0 | 0 | 32 | 32 | n | 0 | 1.5 |

38 reports, 7829 pages, 21 with a usable DI result, 769 pages read by DI under `auto`, 310 pages still without readable text, 345 s in total.

Notes:

- **R19**: page kinds moved once DI text was attached (scanned 265, figure 8, mixed 2)
- **R24**: page kinds moved once DI text was attached (text 72, form 51, mixed 13, figure 11, drawing_sheet 6, scanned 1)
- **R28**: 27 text-rule pairs; first two rendered at 40 dpi (R28_textdup_290_304.png, R28_textdup_294_308.png) -- AWAITING a visual check by the lead. Pair(s) 290/304 have identical text lines but DIFFERENT markup text, so the claim is wrong on the page a reader sees; page kinds moved once DI text was attached (text 200, form 135, mixed 69, figure 45, scanned 6)

### Hand labels

| label | pages |
|---|---|
| narrative | 433 |
| figure | 61 |
| plan | 18 |
| profile | 28 |
| boring_log | 273 |
| test_pit_log | 251 |
| cpt_log | 23 |
| dcp_log | 54 |
| lab_test | 992 |
| field_test | 19 |
| calculation | 1009 |
| appended_report | 494 |
| photos | 132 |
| divider | 83 |
| cover | 30 |
| letter | 6 |
| toc | 25 |
| other | 216 |

4147 labelled pages over 14 mapped reports (R09, R11, R12, R13, R15, R16, R18, R20, R21, R23, R24, R28, R29, R30).

### Sheet-to-ID matching

- sheet 1: R23 (397 rows, matched by name) -- rows == pages
- sheet 2: R20 (370 rows, matched by name) -- rows == pages
- sheet 3: R16 (426 rows, matched by name) -- rows == pages
- sheet 4: R18 (64 rows, matched by name) -- rows == pages
- sheet 5: R30 (556 rows, matched by name) -- rows == pages
- sheet 6: R13 (197 rows, matched by name) -- rows == pages
- sheet 7: R15 (202 rows, matched by name) -- rows == pages
- sheet 8: R28 (455 rows, matched by name) -- rows == pages
- sheet 9: R21 (729 rows, matched by name) -- rows == pages
- sheet 10: R24 (154 rows, matched by name) -- rows == pages
- sheet 11: R11 (156 rows, matched by name) -- rows == pages
- sheet 12: R12 (159 rows, matched by name) -- rows == pages
- sheet 13: NO MATCH (153 label rows) -- no corpus PDF matches its report name, and nothing with that page count shares a word with it
- sheet 14: R09 (151 rows, matched by name) -- rows == pages
- sheet 15: R29 (131 rows, matched by name) -- rows == pages

1 sheet(s) unmatched. The spreadsheet covers a report whose PDF was never copied into the corpus, so its rows are out of the scorecard until the PDF arrives. It is NOT matched on page count alone: an unrelated report of exactly that length exists, and matching it would have scored those labels against the wrong document.

### Lead's visual check of the R28 text-rule pairs (2026-09-16)

- 290 / 304: two lateral-pile deflection plots whose curves are the same; the
  only difference is the plot title ("Single Row" vs "Lead Row"), which is
  drawn as graphics, not text, so the text rule cannot see it. Same picture,
  different label.
- 294 / 308: two program input-echo pages from two runs with identical inputs
  on that page. The content really is the same; the runs are not.

Rule for the ingest that follows from both: a duplicate claim is recorded,
never acted on. No page is skipped because of it; the calc reader treats each
program run as its own item.

---

## WP1 -- page roles -- round 1

Measured 2026-09-16 with the app venv and the editable planlens checkout. 14 reports with hand labels, 4147 pages, opened `di="auto"`, scored against `labels.labels_for`. 181 s. Gate: precision AND recall >= 0.90 on `boring_log`, `test_pit_log`, `lab_test`, `narrative`, `calculation`.

**This round:** first measurement of the rules on the live documents

### Gated roles, round by round

| round | boring_log P/R | test_pit_log P/R | lab_test P/R | narrative P/R | calculation P/R | accuracy | what changed |
|---|---|---|---|---|---|---|---|
| 1 | 0.970 / 0.941 | 0.959 / 0.920 | 0.936 / 0.953 | 0.942 / 0.933 | 0.997 / 0.974 | 0.900 | first measurement of the rules on the live documents |

### Every role, this round

| role | precision | recall | F1 | hand-labelled |
|---|---|---|---|---|
| `narrative` **(gated)** | 0.942 | 0.933 | 0.937 | 433 |
| `figure` | 0.276 | 0.393 | 0.324 | 61 |
| `plan` | 0.381 | 0.444 | 0.410 | 18 |
| `profile` | 0.889 | 0.571 | 0.696 | 28 |
| `boring_log` **(gated)** | 0.970 | 0.941 | 0.955 | 273 |
| `test_pit_log` **(gated)** | 0.959 | 0.920 | 0.939 | 251 |
| `cpt_log` | 0.697 | 1.000 | 0.821 | 23 |
| `dcp_log` | 0.929 | 0.241 | 0.382 | 54 |
| `lab_test` **(gated)** | 0.936 | 0.953 | 0.944 | 992 |
| `field_test` | 0.613 | 1.000 | 0.760 | 19 |
| `calculation` **(gated)** | 0.997 | 0.974 | 0.985 | 1009 |
| `appended_report` | 0.990 | 1.000 | 0.995 | 494 |
| `photos` | 0.715 | 0.932 | 0.809 | 132 |
| `divider` | 0.573 | 0.855 | 0.686 | 83 |
| `cover` | 1.000 | 0.367 | 0.537 | 30 |
| `letter` | 1.000 | 0.667 | 0.800 | 6 |
| `toc` | 0.947 | 0.720 | 0.818 | 25 |
| `other` | 0.492 | 0.417 | 0.451 | 216 |

Overall page accuracy **0.900** over 4147 pages. The gate PASSES.

### Confusion matrix (rows = hand label, columns = predicted)

| hand \ pred | narr | figu | plan | prof | bori | test | cpt_ | dcp_ | lab_ | fiel | calc | appe | phot | divi | cove | lett | toc | othe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `narrative` | 404 | . | 1 | . | . | 1 | 1 | . | 8 | . | . | 1 | . | 1 | . | . | 1 | 15 |
| `figure` | 10 | 24 | 3 | . | 1 | . | . | . | 2 | . | . | . | . | . | . | . | . | 21 |
| `plan` | 1 | 1 | 8 | 2 | . | . | . | . | . | 2 | 1 | . | 1 | 1 | . | . | . | 1 |
| `profile` | 2 | 2 | . | 16 | 1 | . | . | . | . | . | 2 | . | . | . | . | . | . | 5 |
| `boring_log` | . | 9 | . | . | 257 | 6 | . | . | . | . | . | . | 1 | . | . | . | . | . |
| `test_pit_log` | . | . | 5 | . | 1 | 231 | . | . | 2 | . | . | . | 11 | 1 | . | . | . | . |
| `cpt_log` | . | . | . | . | . | . | 23 | . | . | . | . | . | . | . | . | . | . | . |
| `dcp_log` | . | . | . | . | . | . | . | 13 | . | . | . | . | 10 | . | . | . | . | 31 |
| `lab_test` | . | 33 | . | . | . | . | . | . | 945 | 4 | . | . | . | 1 | . | . | . | 9 |
| `field_test` | . | . | . | . | . | . | . | . | . | 19 | . | . | . | . | . | . | . | . |
| `calculation` | . | . | . | . | . | . | . | . | 15 | . | 983 | . | . | 11 | . | . | . | . |
| `appended_report` | . | . | . | . | . | . | . | . | . | . | . | 494 | . | . | . | . | . | . |
| `photos` | . | . | . | . | . | . | . | . | 5 | . | . | . | 123 | 4 | . | . | . | . |
| `divider` | . | 1 | 1 | . | . | . | . | . | 3 | . | . | 4 | . | 71 | . | . | . | 3 |
| `cover` | 3 | 6 | . | . | . | . | . | . | 1 | . | . | . | . | 1 | 11 | . | . | 8 |
| `letter` | 2 | . | . | . | . | . | . | . | . | . | . | . | . | . | . | 4 | . | . |
| `toc` | 5 | . | 1 | . | . | . | . | . | 1 | . | . | . | . | . | . | . | 18 | . |
| `other` | 2 | 11 | 2 | . | 5 | 3 | 9 | 1 | 28 | 6 | . | . | 26 | 33 | . | . | . | 90 |

Column keys: `narr` = narrative, `figu` = figure, `plan` = plan, `prof` = profile, `bori` = boring_log, `test` = test_pit_log, `cpt_` = cpt_log, `dcp_` = dcp_log, `lab_` = lab_test, `fiel` = field_test, `calc` = calculation, `appe` = appended_report, `phot` = photos, `divi` = divider, `cove` = cover, `lett` = letter, `toc` = toc, `othe` = other.

### Per-report accuracy

| ID | pages | accuracy |
|---|---|---|
| R09 | 151 | 0.934 |
| R11 | 156 | 0.904 |
| R12 | 159 | 0.698 |
| R13 | 197 | 0.883 |
| R15 | 202 | 0.886 |
| R16 | 426 | 0.939 |
| R18 | 64 | 0.953 |
| R20 | 370 | 0.984 |
| R21 | 729 | 0.966 |
| R23 | 397 | 0.942 |
| R24 | 154 | 0.468 |
| R28 | 455 | 0.886 |
| R29 | 131 | 0.664 |
| R30 | 556 | 0.941 |

### Misses on the gated roles

216 pages where a gated role was involved and the rules and the hand label disagree, by report. `rule` and `why` are the evidence planlens itself recorded; the page HEADING is omitted on purpose -- the largest type on a log or a laboratory sheet is a firm's title block, and this file is tracked in a public repository. The same list WITH headings is written to `raw/checks/wp1_misses.txt`, which is gitignored.

**R09** (6)

```
R09 p11   hand=other           pred=narrative       kind=figure        rule="prose before the report's first appendix tab" why=''
R09 p13   hand=figure          pred=narrative       kind=figure        rule="prose before the report's first appendix tab" why=''
R09 p14   hand=figure          pred=narrative       kind=figure        rule="prose before the report's first appendix tab" why=''
R09 p15   hand=figure          pred=narrative       kind=figure        rule="prose before the report's first appendix tab" why=''
R09 p17   hand=other           pred=narrative       kind=mixed         rule="prose before the report's first appendix tab" why=''
R09 p150  hand=other           pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'laboratory testing'"
```

**R11** (4)

```
R11 p49   hand=other           pred=lab_test        kind=figure        rule='the page names itself, and its appendix expects it' why="laboratory test title 'direct shear' beside a reference to t"
R11 p50   hand=lab_test        pred=divider         kind=figure        rule='tab or cover page naming what follows' why=''
R11 p146  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R11 p147  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
```

**R12** (42)

```
R12 p3    hand=toc             pred=narrative       kind=mixed         rule="prose before the report's first appendix tab" why=''
R12 p54   hand=figure          pred=boring_log      kind=figure        rule='the page names itself' why="log title 'borehole no', 2 log form fields"
R12 p61   hand=test_pit_log    pred=lab_test        kind=figure        rule='the page names itself' why="laboratory test title 'permeability' beside a reference to t"
R12 p62   hand=test_pit_log    pred=lab_test        kind=figure        rule='the page names itself' why="laboratory test title 'permeability' beside a reference to t"
R12 p64   hand=other           pred=lab_test        kind=figure        rule='its appendix says what it holds' why="['lab_test']"
R12 p78   hand=other           pred=lab_test        kind=figure        rule='its appendix says what it holds' why="['lab_test']"
R12 p96   hand=lab_test        pred=field_test      kind=figure        rule='its appendix says what it holds' why="['field_test', 'lab_test']"
R12 p102  hand=lab_test        pred=field_test      kind=figure        rule='its appendix says what it holds' why="['field_test', 'lab_test']"
R12 p108  hand=lab_test        pred=field_test      kind=figure        rule='its appendix says what it holds' why="['field_test', 'lab_test']"
R12 p114  hand=lab_test        pred=field_test      kind=figure        rule='its appendix says what it holds' why="['field_test', 'lab_test']"
R12 p123  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R12 p124  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R12 p125  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R12 p126  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R12 p127  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R12 p128  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R12 p129  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R12 p130  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R12 p131  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R12 p132  hand=other           pred=lab_test        kind=figure        rule='the page names itself' why="laboratory test title 'laboratory test'"
R12 p133  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R12 p134  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R12 p135  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R12 p136  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R12 p137  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R12 p138  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R12 p140  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R12 p141  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R12 p142  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R12 p143  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R12 p144  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R12 p145  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R12 p146  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R12 p147  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R12 p148  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R12 p149  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R12 p150  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R12 p152  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R12 p153  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R12 p154  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R12 p155  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R12 p156  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
```

**R13** (19)

```
R13 p1    hand=cover           pred=narrative       kind=scanned       rule="prose before the report's first appendix tab" why=''
R13 p2    hand=letter          pred=narrative       kind=scanned       rule="prose before the report's first appendix tab" why=''
R13 p3    hand=letter          pred=narrative       kind=scanned       rule="prose before the report's first appendix tab" why=''
R13 p16   hand=narrative       pred=lab_test        kind=scanned       rule='the page names itself' why="laboratory test title 'corrosivity'"
R13 p33   hand=figure          pred=narrative       kind=scanned       rule="prose before the report's first appendix tab" why=''
R13 p34   hand=figure          pred=narrative       kind=scanned       rule="prose before the report's first appendix tab" why=''
R13 p35   hand=figure          pred=narrative       kind=scanned       rule="prose before the report's first appendix tab" why=''
R13 p36   hand=figure          pred=narrative       kind=scanned       rule="prose before the report's first appendix tab" why=''
R13 p37   hand=figure          pred=narrative       kind=scanned       rule="prose before the report's first appendix tab" why=''
R13 p38   hand=figure          pred=narrative       kind=scanned       rule="prose before the report's first appendix tab" why=''
R13 p39   hand=figure          pred=narrative       kind=scanned       rule="prose before the report's first appendix tab" why=''
R13 p43   hand=other           pred=lab_test        kind=scanned       rule='the page names itself' why="laboratory test title 'organic matter'"
R13 p54   hand=test_pit_log    pred=plan            kind=scanned       rule='the page names itself, and its appendix expects it' why="plan title 'boring location plan'"
R13 p55   hand=test_pit_log    pred=plan            kind=scanned       rule='the page names itself, and its appendix expects it' why="plan title 'boring location plan'"
R13 p56   hand=test_pit_log    pred=plan            kind=scanned       rule='the page names itself, and its appendix expects it' why="plan title 'boring location plan'"
R13 p57   hand=test_pit_log    pred=plan            kind=scanned       rule='the page names itself, and its appendix expects it' why="plan title 'boring location plan'"
R13 p59   hand=test_pit_log    pred=plan            kind=scanned       rule='the page names itself, and its appendix expects it' why="plan title 'boring location plan'"
R13 p118  hand=profile         pred=calculation     kind=scanned       rule='its appendix says what it holds' why="['calculation']"
R13 p119  hand=profile         pred=calculation     kind=scanned       rule='its appendix says what it holds' why="['calculation']"
```

**R15** (19)

```
R15 p3    hand=toc             pred=narrative       kind=text          rule="prose before the report's first appendix tab" why=''
R15 p5    hand=narrative       pred=plan            kind=mixed         rule='the page names itself' why="plan title 'site plans'"
R15 p8    hand=narrative       pred=other           kind=mixed         rule='mixed page with no title of its own' why=''
R15 p19   hand=narrative       pred=other           kind=mixed         rule='mixed page with no title of its own' why=''
R15 p31   hand=narrative       pred=lab_test        kind=text          rule='the page names itself' why="laboratory test title 'laboratory testing' beside a referenc"
R15 p39   hand=other           pred=boring_log      kind=text          rule='the page names itself' why="log title 'test borings', 13 log form fields"
R15 p42   hand=other           pred=test_pit_log    kind=mixed         rule='the page names itself' why="log title 'test pit', 5 log form fields"
R15 p63   hand=boring_log      pred=photos          kind=form          rule='the page names itself' why='photograph caption (1)'
R15 p67   hand=test_pit_log    pred=boring_log      kind=form          rule='the page names itself' why="log title 'boring number', 6 log form fields"
R15 p76   hand=divider         pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'laboratory test'"
R15 p90   hand=lab_test        pred=other           kind=text          rule='text page with no title of its own' why=''
R15 p92   hand=lab_test        pred=other           kind=text          rule='text page with no title of its own' why=''
R15 p102  hand=lab_test        pred=other           kind=form          rule='form page with no title of its own' why=''
R15 p103  hand=lab_test        pred=other           kind=text          rule='text page with no title of its own' why=''
R15 p104  hand=lab_test        pred=other           kind=mixed         rule='mixed page with no title of its own' why=''
R15 p107  hand=lab_test        pred=other           kind=text          rule='text page with no title of its own' why=''
R15 p108  hand=lab_test        pred=other           kind=text          rule='text page with no title of its own' why=''
R15 p109  hand=lab_test        pred=other           kind=text          rule='text page with no title of its own' why=''
R15 p110  hand=lab_test        pred=other           kind=mixed         rule='mixed page with no title of its own' why=''
```

**R16** (8)

```
R16 p1    hand=cover           pred=narrative       kind=text          rule="prose before the report's first appendix tab" why=''
R16 p7    hand=narrative       pred=other           kind=text          rule='text page with no title of its own' why=''
R16 p27   hand=profile         pred=boring_log      kind=form          rule='the page names itself' why='log form shape, 6 log form fields'
R16 p29   hand=other           pred=boring_log      kind=mixed         rule='its appendix says what it holds' why="['boring_log', 'photos']"
R16 p79   hand=other           pred=lab_test        kind=mixed         rule='the page names itself, and its appendix expects it' why="laboratory test title 'laboratory test'"
R16 p128  hand=cover           pred=lab_test        kind=mixed         rule='its appendix says what it holds' why="['lab_test']"
R16 p418  hand=calculation     pred=divider         kind=text          rule='tab or cover page naming what follows' why=''
R16 p420  hand=calculation     pred=divider         kind=text          rule='tab or cover page naming what follows' why=''
```

**R18** (2)

```
R18 p19   hand=narrative       pred=other           kind=mixed         rule='mixed page with no title of its own' why=''
R18 p63   hand=calculation     pred=divider         kind=mixed         rule='tab or cover page naming what follows' why=''
```

**R20** (4)

```
R20 p6    hand=narrative       pred=other           kind=mixed         rule='mixed page with no title of its own' why=''
R20 p29   hand=narrative       pred=appended_report kind=text          rule='inside a report bound into this one' why=''
R20 p115  hand=test_pit_log    pred=divider         kind=mixed         rule='tab or cover page naming what follows' why=''
R20 p354  hand=calculation     pred=divider         kind=text          rule='tab or cover page naming what follows' why=''
```

**R21** (20)

```
R21 p33   hand=plan            pred=narrative       kind=form          rule="prose before the report's first appendix tab" why=''
R21 p35   hand=profile         pred=narrative       kind=form          rule="prose before the report's first appendix tab" why=''
R21 p36   hand=profile         pred=narrative       kind=form          rule="prose before the report's first appendix tab" why=''
R21 p41   hand=other           pred=lab_test        kind=form          rule='the page names itself' why="laboratory test title 'atterberg limits'"
R21 p115  hand=other           pred=lab_test        kind=text          rule='the page names itself, and its appendix expects it' why="laboratory test title 'laboratory test' beside a reference t"
R21 p714  hand=calculation     pred=divider         kind=mixed         rule='tab or cover page naming what follows' why=''
R21 p715  hand=calculation     pred=lab_test        kind=text          rule='the page names itself' why='3 laboratory test terms on the page'
R21 p716  hand=calculation     pred=lab_test        kind=text          rule='the page names itself' why='3 laboratory test terms on the page'
R21 p717  hand=calculation     pred=lab_test        kind=text          rule='the page names itself' why='3 laboratory test terms on the page'
R21 p718  hand=calculation     pred=lab_test        kind=text          rule='the page names itself' why='3 laboratory test terms on the page'
R21 p719  hand=calculation     pred=lab_test        kind=text          rule='the page names itself' why='3 laboratory test terms on the page'
R21 p720  hand=calculation     pred=lab_test        kind=text          rule='the page names itself' why='3 laboratory test terms on the page'
R21 p721  hand=calculation     pred=lab_test        kind=text          rule='the page names itself' why='3 laboratory test terms on the page'
R21 p722  hand=calculation     pred=lab_test        kind=text          rule='the page names itself' why='3 laboratory test terms on the page'
R21 p723  hand=calculation     pred=lab_test        kind=text          rule='the page names itself' why='3 laboratory test terms on the page'
R21 p724  hand=calculation     pred=lab_test        kind=text          rule='the page names itself' why='3 laboratory test terms on the page'
R21 p725  hand=calculation     pred=lab_test        kind=text          rule='the page names itself' why='3 laboratory test terms on the page'
R21 p726  hand=calculation     pred=lab_test        kind=text          rule='the page names itself' why='3 laboratory test terms on the page'
R21 p727  hand=calculation     pred=lab_test        kind=text          rule='the page names itself' why='3 laboratory test terms on the page'
R21 p728  hand=calculation     pred=lab_test        kind=text          rule='the page names itself' why='3 laboratory test terms on the page'
```

**R23** (12)

```
R23 p4    hand=narrative       pred=other           kind=mixed         rule='mixed page with no title of its own' why=''
R23 p5    hand=narrative       pred=other           kind=text          rule='text page with no title of its own' why=''
R23 p170  hand=boring_log      pred=test_pit_log    kind=figure        rule='its appendix says what it holds' why="['test_pit_log', 'photos']"
R23 p171  hand=boring_log      pred=test_pit_log    kind=mixed         rule='its appendix says what it holds' why="['test_pit_log', 'photos']"
R23 p184  hand=boring_log      pred=test_pit_log    kind=mixed         rule='its appendix says what it holds' why="['test_pit_log', 'photos']"
R23 p191  hand=boring_log      pred=test_pit_log    kind=text          rule='its appendix says what it holds' why="['test_pit_log', 'photos']"
R23 p192  hand=boring_log      pred=test_pit_log    kind=figure        rule='its appendix says what it holds' why="['test_pit_log', 'photos']"
R23 p194  hand=boring_log      pred=test_pit_log    kind=figure        rule='its appendix says what it holds' why="['test_pit_log', 'photos']"
R23 p197  hand=other           pred=test_pit_log    kind=figure        rule='its appendix says what it holds' why="['test_pit_log', 'photos']"
R23 p255  hand=other           pred=lab_test        kind=text          rule='its appendix says what it holds' why="['lab_test']"
R23 p375  hand=calculation     pred=divider         kind=figure        rule='tab or cover page naming what follows' why=''
R23 p376  hand=calculation     pred=divider         kind=text          rule='tab or cover page naming what follows' why=''
```

**R24** (30)

```
R24 p4    hand=narrative       pred=toc             kind=text          rule='table of contents heading' why=''
R24 p5    hand=narrative       pred=divider         kind=text          rule='tab or cover page naming what follows' why=''
R24 p6    hand=narrative       pred=other           kind=text          rule='text page with no title of its own' why=''
R24 p7    hand=narrative       pred=other           kind=text          rule='text page with no title of its own' why=''
R24 p8    hand=narrative       pred=other           kind=text          rule='text page with no title of its own' why=''
R24 p9    hand=narrative       pred=lab_test        kind=text          rule='the page names itself' why='6 laboratory test terms on the page'
R24 p10   hand=narrative       pred=lab_test        kind=text          rule='the page names itself' why="laboratory test title 'grain size distribution'"
R24 p11   hand=narrative       pred=cpt_log         kind=text          rule='the page names itself, and its appendix expects it' why="log title 'cone penetration', 2 log form fields"
R24 p12   hand=narrative       pred=lab_test        kind=text          rule='the page names itself' why="laboratory test title 'consolidation'"
R24 p13   hand=narrative       pred=lab_test        kind=text          rule='the page names itself' why="laboratory test title 'laboratory testing'"
R24 p14   hand=narrative       pred=lab_test        kind=text          rule='the page names itself' why="laboratory test title 'laboratory testing'"
R24 p15   hand=narrative       pred=other           kind=text          rule='text page with no title of its own' why=''
R24 p16   hand=narrative       pred=other           kind=text          rule='text page with no title of its own' why=''
R24 p18   hand=figure          pred=lab_test        kind=form          rule='the page names itself' why="laboratory test title 'water content'"
R24 p20   hand=figure          pred=lab_test        kind=form          rule='the page names itself' why="laboratory test title 'plasticity index'"
R24 p26   hand=divider         pred=lab_test        kind=mixed         rule='the page names itself' why='4 laboratory test terms on the page'
R24 p29   hand=other           pred=boring_log      kind=text          rule='the page names itself' why="log title 'boring profiles', 6 log form fields"
R24 p45   hand=other           pred=lab_test        kind=text          rule='the page names itself' why="laboratory test title 'classification test'"
R24 p47   hand=other           pred=lab_test        kind=text          rule='the page names itself' why="laboratory test title 'classification test'"
R24 p55   hand=other           pred=lab_test        kind=text          rule='the page names itself, and its appendix expects it' why="laboratory test title 'consolidation test'"
R24 p56   hand=other           pred=lab_test        kind=text          rule='the page names itself, and its appendix expects it' why="laboratory test title 'consolidation test'"
R24 p57   hand=other           pred=lab_test        kind=text          rule='the page names itself, and its appendix expects it' why="laboratory test title 'consolidation test'"
R24 p75   hand=other           pred=lab_test        kind=text          rule='the page names itself' why="laboratory test title 'triaxial'"
R24 p76   hand=other           pred=lab_test        kind=text          rule='the page names itself' why="laboratory test title 'triaxial'"
R24 p77   hand=other           pred=lab_test        kind=text          rule='the page names itself' why="laboratory test title 'triaxial'"
R24 p89   hand=other           pred=lab_test        kind=text          rule='the page names itself' why="laboratory test title 'compaction test'"
R24 p90   hand=other           pred=lab_test        kind=text          rule='the page names itself' why="laboratory test title 'compaction test'"
R24 p97   hand=divider         pred=lab_test        kind=mixed         rule='its appendix says what it holds' why="['lab_test']"
R24 p121  hand=other           pred=lab_test        kind=text          rule='the page names itself' why='2 laboratory test terms on the page'
R24 p122  hand=other           pred=lab_test        kind=text          rule='the page names itself' why="laboratory test title 'unconfined compression'"
```

**R28** (18)

```
R28 p1    hand=cover           pred=narrative       kind=mixed         rule="prose before the report's first appendix tab" why=''
R28 p4    hand=toc             pred=narrative       kind=text          rule="prose before the report's first appendix tab" why=''
R28 p5    hand=toc             pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'laboratory test'"
R28 p25   hand=narrative       pred=other           kind=mixed         rule='mixed page with no title of its own' why=''
R28 p28   hand=narrative       pred=other           kind=mixed         rule='mixed page with no title of its own' why=''
R28 p53   hand=other           pred=boring_log      kind=text          rule='the page names itself' why="log title 'test borings', 15 log form fields"
R28 p56   hand=other           pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'sieve' beside a reference to the expl"
R28 p233  hand=test_pit_log    pred=photos          kind=figure        rule='the page names itself' why='photograph caption (1)'
R28 p236  hand=test_pit_log    pred=photos          kind=figure        rule='the page names itself' why='photograph caption (1)'
R28 p238  hand=test_pit_log    pred=photos          kind=figure        rule='the page names itself' why='photograph caption (1)'
R28 p253  hand=test_pit_log    pred=photos          kind=figure        rule='its appendix says what it holds' why="['photos', 'test_pit_log']"
R28 p254  hand=test_pit_log    pred=photos          kind=figure        rule='its appendix says what it holds' why="['photos', 'test_pit_log']"
R28 p255  hand=test_pit_log    pred=photos          kind=figure        rule='its appendix says what it holds' why="['photos', 'test_pit_log']"
R28 p256  hand=test_pit_log    pred=photos          kind=figure        rule='its appendix says what it holds' why="['photos', 'test_pit_log']"
R28 p257  hand=test_pit_log    pred=photos          kind=figure        rule='its appendix says what it holds' why="['photos', 'test_pit_log']"
R28 p258  hand=test_pit_log    pred=photos          kind=figure        rule='its appendix says what it holds' why="['photos', 'test_pit_log']"
R28 p259  hand=test_pit_log    pred=photos          kind=figure        rule='its appendix says what it holds' why="['photos', 'test_pit_log']"
R28 p260  hand=test_pit_log    pred=photos          kind=figure        rule='its appendix says what it holds' why="['photos', 'test_pit_log']"
```

**R29** (14)

```
R29 p3    hand=toc             pred=narrative       kind=text          rule="prose before the report's first appendix tab" why=''
R29 p5    hand=toc             pred=narrative       kind=text          rule="prose before the report's first appendix tab" why=''
R29 p24   hand=narrative       pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'corrosivity'"
R29 p40   hand=narrative       pred=other           kind=mixed         rule='mixed page with no title of its own' why=''
R29 p53   hand=boring_log      pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R29 p54   hand=boring_log      pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R29 p55   hand=boring_log      pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R29 p56   hand=boring_log      pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R29 p57   hand=boring_log      pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R29 p58   hand=boring_log      pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R29 p59   hand=boring_log      pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R29 p60   hand=boring_log      pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R29 p61   hand=boring_log      pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R29 p130  hand=calculation     pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'gradation'"
```

**R30** (18)

```
R30 p5    hand=narrative       pred=test_pit_log    kind=text          rule='the page names itself' why="log title 'test pits', 3 log form fields"
R30 p49   hand=plan            pred=calculation     kind=form          rule='the page names itself' why="calculation heading 'global stability'"
R30 p58   hand=other           pred=boring_log      kind=mixed         rule='its appendix says what it holds' why="['boring_log']"
R30 p131  hand=other           pred=test_pit_log    kind=mixed         rule='its appendix says what it holds' why="['test_pit_log']"
R30 p228  hand=other           pred=lab_test        kind=mixed         rule='the page names itself, and its appendix expects it' why="laboratory test title 'laboratory test'"
R30 p296  hand=other           pred=lab_test        kind=mixed         rule='the page names itself, and its appendix expects it' why="laboratory test title 'laboratory test'"
R30 p297  hand=other           pred=lab_test        kind=mixed         rule='the page names itself, and its appendix expects it' why="laboratory test title 'laboratory testing'"
R30 p385  hand=other           pred=lab_test        kind=mixed         rule='the page names itself, and its appendix expects it' why="laboratory test title 'laboratory testing'"
R30 p418  hand=photos          pred=lab_test        kind=figure        rule='its appendix says what it holds' why="['lab_test']"
R30 p419  hand=photos          pred=lab_test        kind=figure        rule='its appendix says what it holds' why="['lab_test']"
R30 p430  hand=other           pred=lab_test        kind=mixed         rule='the page names itself, and its appendix expects it' why="laboratory test title 'laboratory testing'"
R30 p471  hand=photos          pred=lab_test        kind=figure        rule='its appendix says what it holds' why="['lab_test']"
R30 p472  hand=photos          pred=lab_test        kind=figure        rule='its appendix says what it holds' why="['lab_test']"
R30 p473  hand=photos          pred=lab_test        kind=figure        rule='its appendix says what it holds' why="['lab_test']"
R30 p523  hand=calculation     pred=divider         kind=mixed         rule='tab or cover page naming what follows' why=''
R30 p543  hand=calculation     pred=divider         kind=text          rule='tab or cover page naming what follows' why=''
R30 p545  hand=calculation     pred=divider         kind=text          rule='tab or cover page naming what follows' why=''
R30 p547  hand=calculation     pred=divider         kind=text          rule='tab or cover page naming what follows' why=''
```

---

## WP1 -- page roles -- round 2

Measured 2026-09-16 with the app venv and the editable planlens checkout. 14 reports with hand labels, 4147 pages, opened `di="auto"`, scored against `labels.labels_for`. 181 s. Gate: precision AND recall >= 0.90 on `boring_log`, `test_pit_log`, `lab_test`, `narrative`, `calculation`.

**This round:** dynamic-probing abbreviations (DPL/DPM/DPH/DPSH/DPT/LCPT) as DCP titles; chemical and grading laboratory titles; a tab of logs no longer claims a page of prose

### Gated roles, round by round

| round | boring_log P/R | test_pit_log P/R | lab_test P/R | narrative P/R | calculation P/R | accuracy | what changed |
|---|---|---|---|---|---|---|---|
| 1 | 0.970 / 0.941 | 0.959 / 0.920 | 0.936 / 0.953 | 0.942 / 0.933 | 0.997 / 0.974 | 0.900 | first measurement of the rules on the live documents |
| 2 | 0.970 / 0.941 | 0.959 / 0.920 | 0.936 / 0.984 | 0.942 / 0.933 | 0.997 / 0.974 | 0.912 | dynamic-probing abbreviations (DPL/DPM/DPH/DPSH/DPT/LCPT) as DCP titles; chemical and grading laboratory titles; a tab of logs no longer claims a page of prose |

### Every role, this round

| role | precision | recall | F1 | hand-labelled |
|---|---|---|---|---|
| `narrative` **(gated)** | 0.942 | 0.933 | 0.937 | 433 |
| `figure` | 0.444 | 0.393 | 0.417 | 61 |
| `plan` | 0.381 | 0.444 | 0.410 | 18 |
| `profile` | 0.889 | 0.571 | 0.696 | 28 |
| `boring_log` **(gated)** | 0.970 | 0.941 | 0.955 | 273 |
| `test_pit_log` **(gated)** | 0.959 | 0.920 | 0.939 | 251 |
| `cpt_log` | 0.700 | 0.913 | 0.792 | 23 |
| `dcp_log` | 0.968 | 0.556 | 0.706 | 54 |
| `lab_test` **(gated)** | 0.936 | 0.984 | 0.959 | 992 |
| `field_test` | 0.613 | 1.000 | 0.760 | 19 |
| `calculation` **(gated)** | 0.997 | 0.974 | 0.985 | 1009 |
| `appended_report` | 0.990 | 1.000 | 0.995 | 494 |
| `photos` | 0.715 | 0.932 | 0.809 | 132 |
| `divider` | 0.573 | 0.855 | 0.686 | 83 |
| `cover` | 1.000 | 0.367 | 0.537 | 30 |
| `letter` | 1.000 | 0.667 | 0.800 | 6 |
| `toc` | 0.947 | 0.720 | 0.818 | 25 |
| `other` | 0.538 | 0.421 | 0.473 | 216 |

Overall page accuracy **0.912** over 4147 pages. The gate PASSES.

### Confusion matrix (rows = hand label, columns = predicted)

| hand \ pred | narr | figu | plan | prof | bori | test | cpt_ | dcp_ | lab_ | fiel | calc | appe | phot | divi | cove | lett | toc | othe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `narrative` | 404 | . | 1 | . | . | 1 | 1 | . | 8 | . | . | 1 | . | 1 | . | . | 1 | 15 |
| `figure` | 10 | 24 | 3 | . | 1 | . | . | . | 2 | . | . | . | . | . | . | . | . | 21 |
| `plan` | 1 | 1 | 8 | 2 | . | . | . | . | . | 2 | 1 | . | 1 | 1 | . | . | . | 1 |
| `profile` | 2 | 2 | . | 16 | 1 | . | . | . | . | . | 2 | . | . | . | . | . | . | 5 |
| `boring_log` | . | 9 | . | . | 257 | 6 | . | . | . | . | . | . | 1 | . | . | . | . | . |
| `test_pit_log` | . | . | 5 | . | 1 | 231 | . | . | 2 | . | . | . | 11 | 1 | . | . | . | . |
| `cpt_log` | . | . | . | . | . | . | 21 | . | . | . | . | . | . | . | . | . | . | 2 |
| `dcp_log` | . | . | . | . | . | . | . | 30 | . | . | . | . | 10 | . | . | . | . | 14 |
| `lab_test` | . | 2 | . | . | . | . | . | . | 976 | 4 | . | . | . | 1 | . | . | . | 9 |
| `field_test` | . | . | . | . | . | . | . | . | . | 19 | . | . | . | . | . | . | . | . |
| `calculation` | . | . | . | . | . | . | . | . | 15 | . | 983 | . | . | 11 | . | . | . | . |
| `appended_report` | . | . | . | . | . | . | . | . | . | . | . | 494 | . | . | . | . | . | . |
| `photos` | . | . | . | . | . | . | . | . | 5 | . | . | . | 123 | 4 | . | . | . | . |
| `divider` | . | 1 | 1 | . | . | . | . | . | 3 | . | . | 4 | . | 71 | . | . | . | 3 |
| `cover` | 3 | 6 | . | . | . | . | . | . | 1 | . | . | . | . | 1 | 11 | . | . | 8 |
| `letter` | 2 | . | . | . | . | . | . | . | . | . | . | . | . | . | . | 4 | . | . |
| `toc` | 5 | . | 1 | . | . | . | . | . | 1 | . | . | . | . | . | . | . | 18 | . |
| `other` | 2 | 9 | 2 | . | 5 | 3 | 8 | 1 | 30 | 6 | . | . | 26 | 33 | . | . | . | 91 |

Column keys: `narr` = narrative, `figu` = figure, `plan` = plan, `prof` = profile, `bori` = boring_log, `test` = test_pit_log, `cpt_` = cpt_log, `dcp_` = dcp_log, `lab_` = lab_test, `fiel` = field_test, `calc` = calculation, `appe` = appended_report, `phot` = photos, `divi` = divider, `cove` = cover, `lett` = letter, `toc` = toc, `othe` = other.

### Per-report accuracy

| ID | pages | accuracy |
|---|---|---|
| R09 | 151 | 0.934 |
| R11 | 156 | 0.904 |
| R12 | 159 | 0.893 |
| R13 | 197 | 0.883 |
| R15 | 202 | 0.886 |
| R16 | 426 | 0.939 |
| R18 | 64 | 0.953 |
| R20 | 370 | 0.978 |
| R21 | 729 | 0.966 |
| R23 | 397 | 0.942 |
| R24 | 154 | 0.474 |
| R28 | 455 | 0.923 |
| R29 | 131 | 0.664 |
| R30 | 556 | 0.941 |

### Misses on the gated roles

187 pages where a gated role was involved and the rules and the hand label disagree, by report. `rule` and `why` are the evidence planlens itself recorded; the page HEADING is omitted on purpose -- the largest type on a log or a laboratory sheet is a firm's title block, and this file is tracked in a public repository. The same list WITH headings is written to `raw/checks/wp1_misses.txt`, which is gitignored.

**R09** (6)

```
R09 p11   hand=other           pred=narrative       kind=figure        rule="prose before the report's first appendix tab" why=''
R09 p13   hand=figure          pred=narrative       kind=figure        rule="prose before the report's first appendix tab" why=''
R09 p14   hand=figure          pred=narrative       kind=figure        rule="prose before the report's first appendix tab" why=''
R09 p15   hand=figure          pred=narrative       kind=figure        rule="prose before the report's first appendix tab" why=''
R09 p17   hand=other           pred=narrative       kind=mixed         rule="prose before the report's first appendix tab" why=''
R09 p150  hand=other           pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'laboratory testing'"
```

**R11** (4)

```
R11 p49   hand=other           pred=lab_test        kind=figure        rule='the page names itself, and its appendix expects it' why="laboratory test title 'direct shear' beside a reference to t"
R11 p50   hand=lab_test        pred=divider         kind=figure        rule='tab or cover page naming what follows' why=''
R11 p146  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R11 p147  hand=lab_test        pred=figure          kind=figure        rule='figure page with no title of its own' why=''
```

**R12** (13)

```
R12 p3    hand=toc             pred=narrative       kind=mixed         rule="prose before the report's first appendix tab" why=''
R12 p54   hand=figure          pred=boring_log      kind=figure        rule='the page names itself' why="log title 'borehole no', 2 log form fields"
R12 p61   hand=test_pit_log    pred=lab_test        kind=figure        rule='the page names itself' why="laboratory test title 'permeability' beside a reference to t"
R12 p62   hand=test_pit_log    pred=lab_test        kind=figure        rule='the page names itself' why="laboratory test title 'permeability' beside a reference to t"
R12 p64   hand=other           pred=lab_test        kind=figure        rule='its appendix says what it holds' why="['lab_test']"
R12 p78   hand=other           pred=lab_test        kind=figure        rule='its appendix says what it holds' why="['lab_test']"
R12 p96   hand=lab_test        pred=field_test      kind=figure        rule='its appendix says what it holds' why="['field_test', 'lab_test']"
R12 p102  hand=lab_test        pred=field_test      kind=figure        rule='its appendix says what it holds' why="['field_test', 'lab_test']"
R12 p108  hand=lab_test        pred=field_test      kind=figure        rule='its appendix says what it holds' why="['field_test', 'lab_test']"
R12 p114  hand=lab_test        pred=field_test      kind=figure        rule='its appendix says what it holds' why="['field_test', 'lab_test']"
R12 p122  hand=other           pred=lab_test        kind=figure        rule='the page names itself, and its appendix expects it' why='2 laboratory test terms on the page'
R12 p132  hand=other           pred=lab_test        kind=figure        rule='the page names itself' why="laboratory test title 'water extract'"
R12 p139  hand=other           pred=lab_test        kind=figure        rule='the page names itself, and its appendix expects it' why='2 laboratory test terms on the page'
```

**R13** (19)

```
R13 p1    hand=cover           pred=narrative       kind=scanned       rule="prose before the report's first appendix tab" why=''
R13 p2    hand=letter          pred=narrative       kind=scanned       rule="prose before the report's first appendix tab" why=''
R13 p3    hand=letter          pred=narrative       kind=scanned       rule="prose before the report's first appendix tab" why=''
R13 p16   hand=narrative       pred=lab_test        kind=scanned       rule='the page names itself' why="laboratory test title 'corrosivity'"
R13 p33   hand=figure          pred=narrative       kind=scanned       rule="prose before the report's first appendix tab" why=''
R13 p34   hand=figure          pred=narrative       kind=scanned       rule="prose before the report's first appendix tab" why=''
R13 p35   hand=figure          pred=narrative       kind=scanned       rule="prose before the report's first appendix tab" why=''
R13 p36   hand=figure          pred=narrative       kind=scanned       rule="prose before the report's first appendix tab" why=''
R13 p37   hand=figure          pred=narrative       kind=scanned       rule="prose before the report's first appendix tab" why=''
R13 p38   hand=figure          pred=narrative       kind=scanned       rule="prose before the report's first appendix tab" why=''
R13 p39   hand=figure          pred=narrative       kind=scanned       rule="prose before the report's first appendix tab" why=''
R13 p43   hand=other           pred=lab_test        kind=scanned       rule='the page names itself' why="laboratory test title 'organic matter'"
R13 p54   hand=test_pit_log    pred=plan            kind=scanned       rule='the page names itself, and its appendix expects it' why="plan title 'boring location plan'"
R13 p55   hand=test_pit_log    pred=plan            kind=scanned       rule='the page names itself, and its appendix expects it' why="plan title 'boring location plan'"
R13 p56   hand=test_pit_log    pred=plan            kind=scanned       rule='the page names itself, and its appendix expects it' why="plan title 'boring location plan'"
R13 p57   hand=test_pit_log    pred=plan            kind=scanned       rule='the page names itself, and its appendix expects it' why="plan title 'boring location plan'"
R13 p59   hand=test_pit_log    pred=plan            kind=scanned       rule='the page names itself, and its appendix expects it' why="plan title 'boring location plan'"
R13 p118  hand=profile         pred=calculation     kind=scanned       rule='its appendix says what it holds' why="['calculation']"
R13 p119  hand=profile         pred=calculation     kind=scanned       rule='its appendix says what it holds' why="['calculation']"
```

**R15** (19)

```
R15 p3    hand=toc             pred=narrative       kind=text          rule="prose before the report's first appendix tab" why=''
R15 p5    hand=narrative       pred=plan            kind=mixed         rule='the page names itself' why="plan title 'site plans'"
R15 p8    hand=narrative       pred=other           kind=mixed         rule='mixed page with no title of its own' why=''
R15 p19   hand=narrative       pred=other           kind=mixed         rule='mixed page with no title of its own' why=''
R15 p31   hand=narrative       pred=lab_test        kind=text          rule='the page names itself' why="laboratory test title 'laboratory testing' beside a referenc"
R15 p39   hand=other           pred=boring_log      kind=text          rule='the page names itself' why="log title 'test borings', 13 log form fields"
R15 p42   hand=other           pred=test_pit_log    kind=mixed         rule='the page names itself' why="log title 'test pit', 5 log form fields"
R15 p63   hand=boring_log      pred=photos          kind=form          rule='the page names itself' why='photograph caption (1)'
R15 p67   hand=test_pit_log    pred=boring_log      kind=form          rule='the page names itself' why="log title 'boring number', 6 log form fields"
R15 p76   hand=divider         pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'laboratory test'"
R15 p90   hand=lab_test        pred=other           kind=text          rule='text page with no title of its own' why=''
R15 p92   hand=lab_test        pred=other           kind=text          rule='text page with no title of its own' why=''
R15 p102  hand=lab_test        pred=other           kind=form          rule='form page with no title of its own' why=''
R15 p103  hand=lab_test        pred=other           kind=text          rule='text page with no title of its own' why=''
R15 p104  hand=lab_test        pred=other           kind=mixed         rule='mixed page with no title of its own' why=''
R15 p107  hand=lab_test        pred=other           kind=text          rule='text page with no title of its own' why=''
R15 p108  hand=lab_test        pred=other           kind=text          rule='text page with no title of its own' why=''
R15 p109  hand=lab_test        pred=other           kind=text          rule='text page with no title of its own' why=''
R15 p110  hand=lab_test        pred=other           kind=mixed         rule='mixed page with no title of its own' why=''
```

**R16** (8)

```
R16 p1    hand=cover           pred=narrative       kind=text          rule="prose before the report's first appendix tab" why=''
R16 p7    hand=narrative       pred=other           kind=text          rule='text page with no title of its own' why=''
R16 p27   hand=profile         pred=boring_log      kind=form          rule='the page names itself' why='log form shape, 6 log form fields'
R16 p29   hand=other           pred=boring_log      kind=mixed         rule='its appendix says what it holds' why="['boring_log', 'photos']"
R16 p79   hand=other           pred=lab_test        kind=mixed         rule='the page names itself, and its appendix expects it' why="laboratory test title 'laboratory test'"
R16 p128  hand=cover           pred=lab_test        kind=mixed         rule='its appendix says what it holds' why="['lab_test']"
R16 p418  hand=calculation     pred=divider         kind=text          rule='tab or cover page naming what follows' why=''
R16 p420  hand=calculation     pred=divider         kind=text          rule='tab or cover page naming what follows' why=''
```

**R18** (2)

```
R18 p19   hand=narrative       pred=other           kind=mixed         rule='mixed page with no title of its own' why=''
R18 p63   hand=calculation     pred=divider         kind=mixed         rule='tab or cover page naming what follows' why=''
```

**R20** (4)

```
R20 p6    hand=narrative       pred=other           kind=mixed         rule='mixed page with no title of its own' why=''
R20 p29   hand=narrative       pred=appended_report kind=text          rule='inside a report bound into this one' why=''
R20 p115  hand=test_pit_log    pred=divider         kind=mixed         rule='tab or cover page naming what follows' why=''
R20 p354  hand=calculation     pred=divider         kind=text          rule='tab or cover page naming what follows' why=''
```

**R21** (20)

```
R21 p33   hand=plan            pred=narrative       kind=form          rule="prose before the report's first appendix tab" why=''
R21 p35   hand=profile         pred=narrative       kind=form          rule="prose before the report's first appendix tab" why=''
R21 p36   hand=profile         pred=narrative       kind=form          rule="prose before the report's first appendix tab" why=''
R21 p41   hand=other           pred=lab_test        kind=form          rule='the page names itself' why="laboratory test title 'atterberg limits'"
R21 p115  hand=other           pred=lab_test        kind=text          rule='the page names itself, and its appendix expects it' why="laboratory test title 'laboratory test' beside a reference t"
R21 p714  hand=calculation     pred=divider         kind=mixed         rule='tab or cover page naming what follows' why=''
R21 p715  hand=calculation     pred=lab_test        kind=text          rule='the page names itself' why='3 laboratory test terms on the page'
R21 p716  hand=calculation     pred=lab_test        kind=text          rule='the page names itself' why='3 laboratory test terms on the page'
R21 p717  hand=calculation     pred=lab_test        kind=text          rule='the page names itself' why='3 laboratory test terms on the page'
R21 p718  hand=calculation     pred=lab_test        kind=text          rule='the page names itself' why='3 laboratory test terms on the page'
R21 p719  hand=calculation     pred=lab_test        kind=text          rule='the page names itself' why='3 laboratory test terms on the page'
R21 p720  hand=calculation     pred=lab_test        kind=text          rule='the page names itself' why='3 laboratory test terms on the page'
R21 p721  hand=calculation     pred=lab_test        kind=text          rule='the page names itself' why='3 laboratory test terms on the page'
R21 p722  hand=calculation     pred=lab_test        kind=text          rule='the page names itself' why='3 laboratory test terms on the page'
R21 p723  hand=calculation     pred=lab_test        kind=text          rule='the page names itself' why='3 laboratory test terms on the page'
R21 p724  hand=calculation     pred=lab_test        kind=text          rule='the page names itself' why='3 laboratory test terms on the page'
R21 p725  hand=calculation     pred=lab_test        kind=text          rule='the page names itself' why='3 laboratory test terms on the page'
R21 p726  hand=calculation     pred=lab_test        kind=text          rule='the page names itself' why='3 laboratory test terms on the page'
R21 p727  hand=calculation     pred=lab_test        kind=text          rule='the page names itself' why='3 laboratory test terms on the page'
R21 p728  hand=calculation     pred=lab_test        kind=text          rule='the page names itself' why='3 laboratory test terms on the page'
```

**R23** (12)

```
R23 p4    hand=narrative       pred=other           kind=mixed         rule='mixed page with no title of its own' why=''
R23 p5    hand=narrative       pred=other           kind=text          rule='text page with no title of its own' why=''
R23 p170  hand=boring_log      pred=test_pit_log    kind=figure        rule='its appendix says what it holds' why="['test_pit_log', 'photos']"
R23 p171  hand=boring_log      pred=test_pit_log    kind=mixed         rule='its appendix says what it holds' why="['test_pit_log', 'photos']"
R23 p184  hand=boring_log      pred=test_pit_log    kind=mixed         rule='its appendix says what it holds' why="['test_pit_log', 'photos']"
R23 p191  hand=boring_log      pred=test_pit_log    kind=text          rule='its appendix says what it holds' why="['test_pit_log', 'photos']"
R23 p192  hand=boring_log      pred=test_pit_log    kind=figure        rule='its appendix says what it holds' why="['test_pit_log', 'photos']"
R23 p194  hand=boring_log      pred=test_pit_log    kind=figure        rule='its appendix says what it holds' why="['test_pit_log', 'photos']"
R23 p197  hand=other           pred=test_pit_log    kind=figure        rule='its appendix says what it holds' why="['test_pit_log', 'photos']"
R23 p255  hand=other           pred=lab_test        kind=text          rule='its appendix says what it holds' why="['lab_test']"
R23 p375  hand=calculation     pred=divider         kind=figure        rule='tab or cover page naming what follows' why=''
R23 p376  hand=calculation     pred=divider         kind=text          rule='tab or cover page naming what follows' why=''
```

**R24** (30)

```
R24 p4    hand=narrative       pred=toc             kind=text          rule='table of contents heading' why=''
R24 p5    hand=narrative       pred=divider         kind=text          rule='tab or cover page naming what follows' why=''
R24 p6    hand=narrative       pred=other           kind=text          rule='prose page inside an appendix of boring_log pages' why=''
R24 p7    hand=narrative       pred=other           kind=text          rule='prose page inside an appendix of boring_log pages' why=''
R24 p8    hand=narrative       pred=other           kind=text          rule='prose page inside an appendix of boring_log pages' why=''
R24 p9    hand=narrative       pred=lab_test        kind=text          rule='the page names itself' why='6 laboratory test terms on the page'
R24 p10   hand=narrative       pred=lab_test        kind=text          rule='the page names itself' why="laboratory test title 'grain size distribution'"
R24 p11   hand=narrative       pred=cpt_log         kind=text          rule='the page names itself, and its appendix expects it' why="log title 'cone penetration', 2 log form fields"
R24 p12   hand=narrative       pred=lab_test        kind=text          rule='the page names itself' why="laboratory test title 'consolidation'"
R24 p13   hand=narrative       pred=lab_test        kind=text          rule='the page names itself' why="laboratory test title 'laboratory testing'"
R24 p14   hand=narrative       pred=lab_test        kind=text          rule='the page names itself' why="laboratory test title 'laboratory testing'"
R24 p15   hand=narrative       pred=other           kind=text          rule='prose page inside an appendix of boring_log pages' why=''
R24 p16   hand=narrative       pred=other           kind=text          rule='prose page inside an appendix of boring_log pages' why=''
R24 p18   hand=figure          pred=lab_test        kind=form          rule='the page names itself' why="laboratory test title 'water content'"
R24 p20   hand=figure          pred=lab_test        kind=form          rule='the page names itself' why="laboratory test title 'plasticity index'"
R24 p26   hand=divider         pred=lab_test        kind=mixed         rule='the page names itself' why='4 laboratory test terms on the page'
R24 p29   hand=other           pred=boring_log      kind=text          rule='the page names itself' why="log title 'boring profiles', 6 log form fields"
R24 p45   hand=other           pred=lab_test        kind=text          rule='the page names itself' why="laboratory test title 'classification test'"
R24 p47   hand=other           pred=lab_test        kind=text          rule='the page names itself' why="laboratory test title 'classification test'"
R24 p55   hand=other           pred=lab_test        kind=text          rule='the page names itself, and its appendix expects it' why="laboratory test title 'consolidation test'"
R24 p56   hand=other           pred=lab_test        kind=text          rule='the page names itself, and its appendix expects it' why="laboratory test title 'consolidation test'"
R24 p57   hand=other           pred=lab_test        kind=text          rule='the page names itself, and its appendix expects it' why="laboratory test title 'consolidation test'"
R24 p75   hand=other           pred=lab_test        kind=text          rule='the page names itself' why="laboratory test title 'triaxial'"
R24 p76   hand=other           pred=lab_test        kind=text          rule='the page names itself' why="laboratory test title 'triaxial'"
R24 p77   hand=other           pred=lab_test        kind=text          rule='the page names itself' why="laboratory test title 'triaxial'"
R24 p89   hand=other           pred=lab_test        kind=text          rule='the page names itself' why="laboratory test title 'compaction test'"
R24 p90   hand=other           pred=lab_test        kind=text          rule='the page names itself' why="laboratory test title 'compaction test'"
R24 p97   hand=divider         pred=lab_test        kind=mixed         rule='its appendix says what it holds' why="['lab_test']"
R24 p121  hand=other           pred=lab_test        kind=text          rule='the page names itself' why='2 laboratory test terms on the page'
R24 p122  hand=other           pred=lab_test        kind=text          rule='the page names itself' why="laboratory test title 'unconfined compression'"
```

**R28** (18)

```
R28 p1    hand=cover           pred=narrative       kind=mixed         rule="prose before the report's first appendix tab" why=''
R28 p4    hand=toc             pred=narrative       kind=text          rule="prose before the report's first appendix tab" why=''
R28 p5    hand=toc             pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'laboratory test'"
R28 p25   hand=narrative       pred=other           kind=mixed         rule='mixed page with no title of its own' why=''
R28 p28   hand=narrative       pred=other           kind=mixed         rule='mixed page with no title of its own' why=''
R28 p53   hand=other           pred=boring_log      kind=text          rule='the page names itself' why="log title 'test borings', 15 log form fields"
R28 p56   hand=other           pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'sieve' beside a reference to the expl"
R28 p233  hand=test_pit_log    pred=photos          kind=figure        rule='the page names itself' why='photograph caption (1)'
R28 p236  hand=test_pit_log    pred=photos          kind=figure        rule='the page names itself' why='photograph caption (1)'
R28 p238  hand=test_pit_log    pred=photos          kind=figure        rule='the page names itself' why='photograph caption (1)'
R28 p253  hand=test_pit_log    pred=photos          kind=figure        rule='its appendix says what it holds' why="['photos', 'test_pit_log']"
R28 p254  hand=test_pit_log    pred=photos          kind=figure        rule='its appendix says what it holds' why="['photos', 'test_pit_log']"
R28 p255  hand=test_pit_log    pred=photos          kind=figure        rule='its appendix says what it holds' why="['photos', 'test_pit_log']"
R28 p256  hand=test_pit_log    pred=photos          kind=figure        rule='its appendix says what it holds' why="['photos', 'test_pit_log']"
R28 p257  hand=test_pit_log    pred=photos          kind=figure        rule='its appendix says what it holds' why="['photos', 'test_pit_log']"
R28 p258  hand=test_pit_log    pred=photos          kind=figure        rule='its appendix says what it holds' why="['photos', 'test_pit_log']"
R28 p259  hand=test_pit_log    pred=photos          kind=figure        rule='its appendix says what it holds' why="['photos', 'test_pit_log']"
R28 p260  hand=test_pit_log    pred=photos          kind=figure        rule='its appendix says what it holds' why="['photos', 'test_pit_log']"
```

**R29** (14)

```
R29 p3    hand=toc             pred=narrative       kind=text          rule="prose before the report's first appendix tab" why=''
R29 p5    hand=toc             pred=narrative       kind=text          rule="prose before the report's first appendix tab" why=''
R29 p24   hand=narrative       pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'corrosivity'"
R29 p40   hand=narrative       pred=other           kind=mixed         rule='mixed page with no title of its own' why=''
R29 p53   hand=boring_log      pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R29 p54   hand=boring_log      pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R29 p55   hand=boring_log      pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R29 p56   hand=boring_log      pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R29 p57   hand=boring_log      pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R29 p58   hand=boring_log      pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R29 p59   hand=boring_log      pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R29 p60   hand=boring_log      pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R29 p61   hand=boring_log      pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R29 p130  hand=calculation     pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'gradation'"
```

**R30** (18)

```
R30 p5    hand=narrative       pred=test_pit_log    kind=text          rule='the page names itself' why="log title 'test pits', 3 log form fields"
R30 p49   hand=plan            pred=calculation     kind=form          rule='the page names itself' why="calculation heading 'global stability'"
R30 p58   hand=other           pred=boring_log      kind=mixed         rule='its appendix says what it holds' why="['boring_log']"
R30 p131  hand=other           pred=test_pit_log    kind=mixed         rule='its appendix says what it holds' why="['test_pit_log']"
R30 p228  hand=other           pred=lab_test        kind=mixed         rule='the page names itself, and its appendix expects it' why="laboratory test title 'laboratory test'"
R30 p296  hand=other           pred=lab_test        kind=mixed         rule='the page names itself, and its appendix expects it' why="laboratory test title 'laboratory test'"
R30 p297  hand=other           pred=lab_test        kind=mixed         rule='the page names itself, and its appendix expects it' why="laboratory test title 'laboratory testing'"
R30 p385  hand=other           pred=lab_test        kind=mixed         rule='the page names itself, and its appendix expects it' why="laboratory test title 'laboratory testing'"
R30 p418  hand=photos          pred=lab_test        kind=figure        rule='its appendix says what it holds' why="['lab_test']"
R30 p419  hand=photos          pred=lab_test        kind=figure        rule='its appendix says what it holds' why="['lab_test']"
R30 p430  hand=other           pred=lab_test        kind=mixed         rule='the page names itself, and its appendix expects it' why="laboratory test title 'laboratory testing'"
R30 p471  hand=photos          pred=lab_test        kind=figure        rule='its appendix says what it holds' why="['lab_test']"
R30 p472  hand=photos          pred=lab_test        kind=figure        rule='its appendix says what it holds' why="['lab_test']"
R30 p473  hand=photos          pred=lab_test        kind=figure        rule='its appendix says what it holds' why="['lab_test']"
R30 p523  hand=calculation     pred=divider         kind=mixed         rule='tab or cover page naming what follows' why=''
R30 p543  hand=calculation     pred=divider         kind=text          rule='tab or cover page naming what follows' why=''
R30 p545  hand=calculation     pred=divider         kind=text          rule='tab or cover page naming what follows' why=''
R30 p547  hand=calculation     pred=divider         kind=text          rule='tab or cover page naming what follows' why=''
```

---

### XGBoost benchmark

Measured 2026-09-16. The page-type classifier from the owner's earlier attempt, scored on the SAME 4,147 hand-labelled pages as the rules above. xgboost 3.4.1 in a scratch venv under the scratchpad (nothing installed into the app venv; the venv reads the app venv's numpy and scikit-learn through a `.pth` because this machine's application-control policy blocks a freshly downloaded numpy DLL in any user-writable path -- three locations were tried). The model is `models/page_type_classifier.json` in the private clone, 16 classes, 134 features (74 hand-written regex and layout counts plus 60 TF-IDF terms). Input is the page text planlens extracts, page by page, through the clone's own `ml_features.extract_features`. Nothing from that clone was copied into either repository.

**Read this first.** The classifier was trained on this very spreadsheet -- its training script loads `trial_pages_working_r2.xlsx` and fits on every sheet in it, with no held-out split at fit time (a leave-one-report-out path exists in that script but is not what produced the shipped model). So every number below is a TRAINING-SET number for the model and a first-sight number for the rules. It is an upper bound on the model, not a like-for-like comparison.

Two features of the 134 could not be supplied exactly: `drawing_count` (pdfplumber's count of drawn lines) is given planlens' count of long axis-aligned rules, and `di_has_tables` is 0 throughout.

#### Class mapping

The model's 16 classes onto the 18-label vocabulary. It has no class for `cpt_log`, `dcp_log` or `appended_report`, so those 571 hand-labelled pages can never come out right whatever it predicts.

| model class | label |
|---|---|
| `appendix_cover_page` | `divider` |
| `boring_log` | `boring_log` |
| `calculation` | `calculation` |
| `cover_letter` | `letter` |
| `cover_page` | `cover` |
| `field_photos` | `photos` |
| `figure` | `figure` |
| `infiltration_testing` | `field_test` |
| `informational_appendix` | `other` |
| `lab_testing` | `lab_test` |
| `main_report_narrative` | `narrative` |
| `other` | `other` |
| `subsurface_profile` | `profile` |
| `table_of_contents` | `toc` |
| `test_location_plan` | `plan` |
| `test_pit_log` | `test_pit_log` |

#### The five gated roles, whole scorecard

All 14 reports, 4,147 pages. 4147 pages.

| role | rules P | rules R | XGBoost P | XGBoost R | hand-labelled |
|---|---|---|---|---|---|
| `boring_log` | 0.970 | 0.941 | 0.761 | 0.934 | 273 |
| `test_pit_log` | 0.959 | 0.920 | 0.892 | 0.956 | 251 |
| `lab_test` | 0.936 | 0.984 | 0.925 | 0.971 | 992 |
| `narrative` | 0.942 | 0.933 | 0.874 | 0.979 | 433 |
| `calculation` | 0.997 | 0.974 | 0.805 | 0.944 | 1009 |
| **page accuracy** | **0.912** | | **0.812** | | |

#### The same, without R21

R21 alone holds 472 of the 494 `appended_report` pages -- one report bound inside another -- and the model has no class that can express them, which costs it 0.325 accuracy on that report alone. Dropping R21 is the most generous reading of the model available. 3418 pages.

| role | rules P | rules R | XGBoost P | XGBoost R | hand-labelled |
|---|---|---|---|---|---|
| `boring_log` | 0.966 | 0.935 | 0.958 | 0.927 | 246 |
| `test_pit_log` | 0.949 | 0.903 | 0.947 | 0.947 | 206 |
| `lab_test` | 0.950 | 0.984 | 0.996 | 0.972 | 989 |
| `narrative` | 0.945 | 0.929 | 0.978 | 0.978 | 408 |
| `calculation` | 0.997 | 0.987 | 0.974 | 0.951 | 873 |
| **page accuracy** | **0.900** | | **0.916** | | |

#### Verdict

On the whole scorecard the rules beat the classifier on precision for every one of the five gated roles and on page accuracy (0.912 against 0.812); the classifier is ahead on recall for `test_pit_log` and `narrative`. It does NOT reach the WP1 gate: `boring_log`, `test_pit_log`, `narrative`, `calculation` fall below 0.90 on precision or recall. Without R21 it is ahead on page accuracy (0.916 against 0.900) and on `narrative` and `lab_test` -- on pages it was fitted to.

Per the plan (section 4, "The old XGBoost page classifier"): the rules match or beat it, so it is NOT adopted and adds no dependency. The one place it is worth remembering is recall on `narrative` and `test_pit_log`, where it finds pages the rules miss -- but on its own training data, so that is a hypothesis for a later look, not a measured advantage. Nothing here is a dependency of the ingest.

#### Per-report accuracy, both

| ID | pages | rules | XGBoost |
|---|---|---|---|
| R09 | 151 | 0.934 | 0.940 |
| R11 | 156 | 0.904 | 0.968 |
| R12 | 159 | 0.893 | 0.943 |
| R13 | 197 | 0.883 | 0.995 |
| R15 | 202 | 0.886 | 0.975 |
| R16 | 426 | 0.939 | 0.939 |
| R18 | 64 | 0.953 | 0.938 |
| R20 | 370 | 0.978 | 0.859 |
| R21 | 729 | 0.966 | 0.325 |
| R23 | 397 | 0.942 | 0.894 |
| R24 | 154 | 0.474 | 0.916 |
| R28 | 455 | 0.923 | 0.866 |
| R29 | 131 | 0.664 | 0.710 |
| R30 | 556 | 0.941 | 0.962 |
