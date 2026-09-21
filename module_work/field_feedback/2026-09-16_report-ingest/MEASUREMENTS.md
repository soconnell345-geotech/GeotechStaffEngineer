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

---

### WP1 -- how these numbers were arrived at (read before trusting them)

The two measured rounds above are NOT held-out numbers, and the point
deserves to be as plain here as it is about the classifier.

Before round 1 the page text, page map and hand label of all 4,147 pages
were cached to a file once, and the rules were written and rewritten against
that cache -- roughly twenty passes, each one looking at the misses and
changing a rule. Round 1 is the first run of the finished rules on the live
documents; it is not the first time those rules met those pages. So the
numbers say the rules DESCRIBE this corpus well. They do not, on their own,
say the rules will hold on the next report.

Three things are held out, and they are the only evidence of generalisation
there is at this point:

1. **A synthetic report the rules never saw** — 21 pages built page by page
   from the wording of the trade (`planlens.testing.build_synthetic_report`):
   a cover, a contents page, a narrative run whose prose discusses Atterberg
   limits and test pits, an appendix tab, two sheets of one boring log, a
   test pit log, a photograph page, two laboratory sheets, a whole report
   bound inside an appendix with its own cover and its own APPENDIX A, and a
   two-page program printout. 21 of 21 pages right, and the work items come
   out as stated. It was written to the RULES' vocabulary, so it proves the
   rules are consistent, not that they are complete.
2. **Two corpus reports with no hand labels** (R36, 94 pages; R18's structure
   was in the labelled set but R36 was not): the roles and the work items
   were eyeballed and the log runs group correctly by boring id, with the
   "Page 1 of 3" continuations folded in.
3. **What the rules do NOT key on.** No rule mentions a firm, a project, a
   place or a report id; the keyword tables are trade vocabulary in four
   languages and the structural rules read running headers and appendix
   lettering. That is a property of the code, checkable by reading it, not a
   measurement.

The honest next measurement is a report outside this corpus, hand-labelled
after the rules were frozen. Until then these numbers are a description of
14 documents.

## WP1 -- the lead's out-of-sample check (rules as of planlens 4e7ed85)

120 pages: five drawn at random (seed 20260916) from each of the 24 reports
that have NO spreadsheet labels, hand-labelled by the lead from 36-dpi
contact sheets BEFORE the rules were run on them (labels and alternates in
`raw/checks/oos/labels.json`; script `report_ingest_harness/measure_oos.py`;
raw output `raw/checks/oos/result_round2.txt`). The rules never saw these
reports (the builder's cache held only the 14 labelled ones; R36 was
eyeballed once).

| set | pages | accuracy |
|---|---|---|
| in-sample, 14 labelled reports (builder's round 2) | 4,147 | 0.912 |
| out-of-sample, 24 unlabelled reports, strict | 120 | 0.742 |
| out-of-sample, accepting the lead's alternates | 120 | 0.792 |

Key content, out of sample (support / recall / precision): boring_log 15 /
0.80 / 0.80; lab_test 20 / 0.75 / 0.83; narrative 30 / 0.87 / 0.76;
calculation 13 / 1.00 / 0.81; test_pit_log 3 / 0.67 / 1.00; plan 3 / 0.33 /
1.00; profile 1 / 0.00; cpt_log 1 / 0.00 (alternate accepted).

The 31 strict misses fall into four classes, and the first two are the
rules' own doing:

1. **Appendix inheritance overrides the page** (13): rule "its appendix says
   what it holds" gives a page the label its appendix tab implies even when
   the page itself says otherwise -- logs, lab sheets and photos mixed in
   one appendix (R02 x5, R19 x2, R01 x2, R02/R33/R31/R34 with alternates).
2. **No appendix tab found, so everything is "prose before the first tab"**
   (8): a 1991 typed data report with no dividers has every lab sheet called
   narrative (R35 x5); a table of contents, a geologic-map figure and a
   figure-plus-table page inside the narrative likewise (R22 p4, R14 p23,
   R10 p19).
3. **Scanned pages with no text source** (4, R38): nothing to read; a DI
   result or OCR is the fix, not a rule.
4. **Single-page cue misfires** (6): a distribution page called a divider
   (R07 p11), an aerial-photo site plan called photos (R08 p22), a flood-zone
   map called calculation (R36 p33), a core-photo log called figure (R22
   p71), a guide-spec page called divider (R06 p117), a lab slip-sheet called
   lab_test (R32 p111).

Read together with the builder's own note above: the in-sample 0.91 is a
description of 14 documents; 0.74 is what the rules do on the next report.
The owner's guard rails (triage + model label review) are therefore
necessary, not optional, and the rules' inheritance must carry LOW
confidence so the review pass looks at exactly those pages. For the next
round the builder gets the diagnostic from ten of these reports (R01, R02,
R03, R04, R05, R06, R07, R08, R10, R14 -- 50 pages); the other fourteen
(R17, R19, R22, R25, R26, R27, R31-R38 -- 70 pages) stay blind for the
re-score.

---

## WP1 -- page roles -- round 3

Measured 2026-09-16 with the app venv and the editable planlens checkout. 4147 hand-labelled pages, opened `di="auto"`, scored against `labels.labels_for`, split into a development set (R09, R12, R15, R16, R20, R23, R28, R29, R30) and a held-out set (R11, R13, R18, R21, R24). **The gate is on the held-out set**: precision AND recall >= 0.90 on `boring_log`, `test_pit_log`, `lab_test`, `narrative`, `calculation`.

**This round:** the dev / held-out split, and document_outline + page_ledger added; NO rule changed this round

**Development Set**, round by round

| round | boring_log P/R | test_pit_log P/R | lab_test P/R | narrative P/R | calculation P/R | accuracy | what changed |
|---|---|---|---|---|---|---|---|
| 1 | 0.970 / 0.941 | 0.959 / 0.920 | 0.936 / 0.953 | 0.942 / 0.933 | 0.997 / 0.974 | 0.900 | first measurement of the rules on the live documents (all 14 reports; measured before the split) |
| 2 | 0.970 / 0.941 | 0.959 / 0.920 | 0.936 / 0.984 | 0.942 / 0.933 | 0.997 / 0.974 | 0.912 | dynamic-probing abbreviations (DPL/DPM/DPH/DPSH/DPT/LCPT) as DCP titles; chemical and grading laboratory titles; a tab of logs no longer claims a page of prose (all 14 reports; measured before the split) |
| 3 | 0.963 / 0.919 | 0.949 / 0.925 | 0.967 / 0.984 | 0.965 / 0.959 | 0.999 / 0.987 | 0.923 | the dev / held-out split, and document_outline + page_ledger added; NO rule changed this round |

**Held-Out Set**, round by round

| round | boring_log P/R | test_pit_log P/R | lab_test P/R | narrative P/R | calculation P/R | accuracy | what changed |
|---|---|---|---|---|---|---|---|
| 1 | - | - | - | - | - | - | first measurement of the rules on the live documents (all 14 reports; measured before the split) |
| 2 | - | - | - | - | - | - | dynamic-probing abbreviations (DPL/DPM/DPH/DPSH/DPT/LCPT) as DCP titles; chemical and grading laboratory titles; a tab of logs no longer claims a page of prose (all 14 reports; measured before the split) |
| 3 | 0.987 / 1.000 | 1.000 / 0.902 | 0.813 / 0.983 | 0.851 / 0.832 | 0.992 / 0.938 | 0.887 | the dev / held-out split, and document_outline + page_ledger added; NO rule changed this round |

### Where held-out lags development

- **`lab_test` lags by more than 0.05**: precision 0.967 to 0.813. The confusions that cost it on the held-out set: `other -> lab_test` (16 pages), `calculation -> lab_test` (14 pages), `narrative -> lab_test` (6 pages).
- **`narrative` lags by more than 0.05**: precision 0.965 to 0.851; recall 0.959 to 0.831. The confusions that cost it on the held-out set: `figure -> narrative` (7 pages), `narrative -> lab_test` (6 pages), `narrative -> other` (6 pages).

### This round in full

#### Held-Out Set

5 reports (R11, R13, R18, R21, R24), 1300 pages, 48 s.

| role | precision | recall | F1 | hand-labelled |
|---|---|---|---|---|
| `narrative` **(gated)** | 0.851 | 0.831 | 0.841 | 89 |
| `figure` | 0.250 | 0.217 | 0.233 | 23 |
| `plan` | 0.250 | 0.500 | 0.333 | 6 |
| `profile` | 0.875 | 0.467 | 0.609 | 15 |
| `boring_log` **(gated)** | 0.987 | 1.000 | 0.993 | 75 |
| `test_pit_log` **(gated)** | 1.000 | 0.902 | 0.948 | 51 |
| `cpt_log` | 0.438 | 1.000 | 0.609 | 7 |
| `dcp_log` | 0.000 | 0.000 | 0.000 | 0 |
| `lab_test` **(gated)** | 0.813 | 0.983 | 0.890 | 177 |
| `field_test` | 0.250 | 1.000 | 0.400 | 2 |
| `calculation` **(gated)** | 0.992 | 0.938 | 0.964 | 259 |
| `appended_report` | 1.000 | 1.000 | 1.000 | 472 |
| `photos` | 0.000 | 0.000 | 0.000 | 0 |
| `divider` | 0.318 | 0.824 | 0.459 | 17 |
| `cover` | 1.000 | 0.250 | 0.400 | 8 |
| `letter` | 1.000 | 0.500 | 0.667 | 4 |
| `toc` | 0.875 | 1.000 | 0.933 | 7 |
| `other` | 0.513 | 0.227 | 0.315 | 88 |

Page accuracy **0.887** over 1300 pages. The gate does NOT pass.

Confusion matrix (rows = hand label, columns = predicted):

| hand \ pred | narr | figu | plan | prof | bori | test | cpt_ | dcp_ | lab_ | fiel | calc | appe | phot | divi | cove | lett | toc | othe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `narrative` | 74 | . | . | . | . | . | 1 | . | 6 | . | . | . | . | 1 | . | . | 1 | 6 |
| `figure` | 7 | 5 | 1 | . | . | . | . | . | 2 | . | . | . | . | . | . | . | . | 8 |
| `plan` | 1 | . | 3 | 1 | . | . | . | . | . | . | . | . | . | 1 | . | . | . | . |
| `profile` | 2 | 2 | . | 7 | . | . | . | . | . | . | 2 | . | . | . | . | . | . | 2 |
| `boring_log` | . | . | . | . | 75 | . | . | . | . | . | . | . | . | . | . | . | . | . |
| `test_pit_log` | . | . | 5 | . | . | 46 | . | . | . | . | . | . | . | . | . | . | . | . |
| `cpt_log` | . | . | . | . | . | . | 7 | . | . | . | . | . | . | . | . | . | . | . |
| `dcp_log` | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| `lab_test` | . | 2 | . | . | . | . | . | . | 174 | . | . | . | . | 1 | . | . | . | . |
| `field_test` | . | . | . | . | . | . | . | . | . | 2 | . | . | . | . | . | . | . | . |
| `calculation` | . | . | . | . | . | . | . | . | 14 | . | 243 | . | . | 2 | . | . | . | . |
| `appended_report` | . | . | . | . | . | . | . | . | . | . | . | 472 | . | . | . | . | . | . |
| `photos` | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| `divider` | . | . | 1 | . | . | . | . | . | 2 | . | . | . | . | 14 | . | . | . | . |
| `cover` | 1 | 2 | . | . | . | . | . | . | . | . | . | . | . | . | 2 | . | . | 3 |
| `letter` | 2 | . | . | . | . | . | . | . | . | . | . | . | . | . | . | 2 | . | . |
| `toc` | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | 7 | . |
| `other` | . | 9 | 2 | . | 1 | . | 8 | 1 | 16 | 6 | . | . | . | 25 | . | . | . | 20 |

Column keys: `narr` = narrative, `figu` = figure, `plan` = plan, `prof` = profile, `bori` = boring_log, `test` = test_pit_log, `cpt_` = cpt_log, `dcp_` = dcp_log, `lab_` = lab_test, `fiel` = field_test, `calc` = calculation, `appe` = appended_report, `phot` = photos, `divi` = divider, `cove` = cover, `lett` = letter, `toc` = toc, `othe` = other.

Per-report accuracy:

| ID | pages | accuracy |
|---|---|---|
| R11 | 156 | 0.904 |
| R13 | 197 | 0.883 |
| R18 | 64 | 0.953 |
| R21 | 729 | 0.966 |
| R24 | 154 | 0.474 |

75 pages where a gated role is involved and the rules and the hand label disagree. They are NOT listed page by page: closing a gap by reading the held-out pages is how a held-out set stops being one. The confusions, by size:

| hand -> predicted | pages |
|---|---|
| `other -> lab_test` | 16 |
| `calculation -> lab_test` | 14 |
| `figure -> narrative` | 7 |
| `narrative -> lab_test` | 6 |
| `narrative -> other` | 6 |
| `test_pit_log -> plan` | 5 |
| `lab_test -> figure` | 2 |
| `letter -> narrative` | 2 |
| `profile -> calculation` | 2 |
| `calculation -> divider` | 2 |
| `profile -> narrative` | 2 |
| `figure -> lab_test` | 2 |
| `divider -> lab_test` | 2 |
| `lab_test -> divider` | 1 |
| `cover -> narrative` | 1 |
| `plan -> narrative` | 1 |
| `narrative -> toc` | 1 |
| `narrative -> divider` | 1 |
| `narrative -> cpt_log` | 1 |
| `other -> boring_log` | 1 |

#### Development Set

9 reports (R09, R12, R15, R16, R20, R23, R28, R29, R30), 2847 pages, 145 s.

| role | precision | recall | F1 | hand-labelled |
|---|---|---|---|---|
| `narrative` **(gated)** | 0.965 | 0.959 | 0.962 | 344 |
| `figure` | 0.559 | 0.500 | 0.528 | 38 |
| `plan` | 0.556 | 0.417 | 0.476 | 12 |
| `profile` | 0.900 | 0.692 | 0.783 | 13 |
| `boring_log` **(gated)** | 0.963 | 0.919 | 0.941 | 198 |
| `test_pit_log` **(gated)** | 0.949 | 0.925 | 0.937 | 200 |
| `cpt_log` | 1.000 | 0.875 | 0.933 | 16 |
| `dcp_log` | 1.000 | 0.556 | 0.714 | 54 |
| `lab_test` **(gated)** | 0.967 | 0.984 | 0.976 | 815 |
| `field_test` | 0.739 | 1.000 | 0.850 | 17 |
| `calculation` **(gated)** | 0.999 | 0.987 | 0.993 | 750 |
| `appended_report` | 0.815 | 1.000 | 0.898 | 22 |
| `photos` | 0.715 | 0.932 | 0.809 | 132 |
| `divider` | 0.713 | 0.864 | 0.781 | 66 |
| `cover` | 1.000 | 0.409 | 0.581 | 22 |
| `letter` | 1.000 | 1.000 | 1.000 | 2 |
| `toc` | 1.000 | 0.611 | 0.759 | 18 |
| `other` | 0.546 | 0.555 | 0.550 | 128 |

Page accuracy **0.923** over 2847 pages.

Confusion matrix (rows = hand label, columns = predicted):

| hand \ pred | narr | figu | plan | prof | bori | test | cpt_ | dcp_ | lab_ | fiel | calc | appe | phot | divi | cove | lett | toc | othe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `narrative` | 330 | . | 1 | . | . | 1 | . | . | 2 | . | . | 1 | . | . | . | . | . | 9 |
| `figure` | 3 | 19 | 2 | . | 1 | . | . | . | . | . | . | . | . | . | . | . | . | 13 |
| `plan` | . | 1 | 5 | 1 | . | . | . | . | . | 2 | 1 | . | 1 | . | . | . | . | 1 |
| `profile` | . | . | . | 9 | 1 | . | . | . | . | . | . | . | . | . | . | . | . | 3 |
| `boring_log` | . | 9 | . | . | 182 | 6 | . | . | . | . | . | . | 1 | . | . | . | . | . |
| `test_pit_log` | . | . | . | . | 1 | 185 | . | . | 2 | . | . | . | 11 | 1 | . | . | . | . |
| `cpt_log` | . | . | . | . | . | . | 14 | . | . | . | . | . | . | . | . | . | . | 2 |
| `dcp_log` | . | . | . | . | . | . | . | 30 | . | . | . | . | 10 | . | . | . | . | 14 |
| `lab_test` | . | . | . | . | . | . | . | . | 802 | 4 | . | . | . | . | . | . | . | 9 |
| `field_test` | . | . | . | . | . | . | . | . | . | 17 | . | . | . | . | . | . | . | . |
| `calculation` | . | . | . | . | . | . | . | . | 1 | . | 740 | . | . | 9 | . | . | . | . |
| `appended_report` | . | . | . | . | . | . | . | . | . | . | . | 22 | . | . | . | . | . | . |
| `photos` | . | . | . | . | . | . | . | . | 5 | . | . | . | 123 | 4 | . | . | . | . |
| `divider` | . | 1 | . | . | . | . | . | . | 1 | . | . | 4 | . | 57 | . | . | . | 3 |
| `cover` | 2 | 4 | . | . | . | . | . | . | 1 | . | . | . | . | 1 | 9 | . | . | 5 |
| `letter` | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | 2 | . | . |
| `toc` | 5 | . | 1 | . | . | . | . | . | 1 | . | . | . | . | . | . | . | 11 | . |
| `other` | 2 | . | . | . | 4 | 3 | . | . | 14 | . | . | . | 26 | 8 | . | . | . | 71 |

Column keys: `narr` = narrative, `figu` = figure, `plan` = plan, `prof` = profile, `bori` = boring_log, `test` = test_pit_log, `cpt_` = cpt_log, `dcp_` = dcp_log, `lab_` = lab_test, `fiel` = field_test, `calc` = calculation, `appe` = appended_report, `phot` = photos, `divi` = divider, `cove` = cover, `lett` = letter, `toc` = toc, `othe` = other.

Per-report accuracy:

| ID | pages | accuracy |
|---|---|---|
| R09 | 151 | 0.934 |
| R12 | 159 | 0.893 |
| R15 | 202 | 0.886 |
| R16 | 426 | 0.939 |
| R20 | 370 | 0.978 |
| R23 | 397 | 0.942 |
| R28 | 455 | 0.923 |
| R29 | 131 | 0.664 |
| R30 | 556 | 0.941 |

112 pages where a gated role is involved and the rules and the hand label disagree. `rule` and `why` are the evidence planlens itself recorded; the page HEADING is omitted on purpose -- the largest type on a log or a laboratory sheet is a firm's title block, and this file is tracked in a public repository. The same list WITH headings is written to `raw/checks/wp1_misses.txt`, which is gitignored.

**R09** (6)

```
R09 p11   hand=other           pred=narrative       kind=figure        rule="prose before the report's first appendix tab" why=''
R09 p13   hand=figure          pred=narrative       kind=figure        rule="prose before the report's first appendix tab" why=''
R09 p14   hand=figure          pred=narrative       kind=figure        rule="prose before the report's first appendix tab" why=''
R09 p15   hand=figure          pred=narrative       kind=figure        rule="prose before the report's first appendix tab" why=''
R09 p17   hand=other           pred=narrative       kind=mixed         rule="prose before the report's first appendix tab" why=''
R09 p150  hand=other           pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'laboratory testing'"
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

**R20** (4)

```
R20 p6    hand=narrative       pred=other           kind=mixed         rule='mixed page with no title of its own' why=''
R20 p29   hand=narrative       pred=appended_report kind=text          rule='inside a report bound into this one' why=''
R20 p115  hand=test_pit_log    pred=divider         kind=mixed         rule='tab or cover page naming what follows' why=''
R20 p354  hand=calculation     pred=divider         kind=text          rule='tab or cover page naming what follows' why=''
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

## WP1 -- page roles -- round 4

Measured 2026-09-16 with the app venv and the editable planlens checkout. 4147 hand-labelled pages, opened `di="auto"`, scored against `labels.labels_for`, split into a development set (R09, R12, R15, R16, R20, R23, R28, R29, R30) and a held-out set (R11, R13, R18, R21, R24). **The gate is on the held-out set**: precision AND recall >= 0.90 on `boring_log`, `test_pit_log`, `lab_test`, `narrative`, `calculation`.

**This round:** a role a page did NOT name itself now carries INHERITED_CONFIDENCE and says so; no prediction changed, so every rate is identical to round 3

**Development Set**, round by round

| round | boring_log P/R | test_pit_log P/R | lab_test P/R | narrative P/R | calculation P/R | accuracy | what changed |
|---|---|---|---|---|---|---|---|
| 1 | 0.970 / 0.941 | 0.959 / 0.920 | 0.936 / 0.953 | 0.942 / 0.933 | 0.997 / 0.974 | 0.900 | first measurement of the rules on the live documents (all 14 reports; measured before the split) |
| 2 | 0.970 / 0.941 | 0.959 / 0.920 | 0.936 / 0.984 | 0.942 / 0.933 | 0.997 / 0.974 | 0.912 | dynamic-probing abbreviations (DPL/DPM/DPH/DPSH/DPT/LCPT) as DCP titles; chemical and grading laboratory titles; a tab of logs no longer claims a page of prose (all 14 reports; measured before the split) |
| 3 | 0.963 / 0.919 | 0.949 / 0.925 | 0.967 / 0.984 | 0.965 / 0.959 | 0.999 / 0.987 | 0.923 | the dev / held-out split, and document_outline + page_ledger added; NO rule changed this round |
| 4 | 0.963 / 0.919 | 0.949 / 0.925 | 0.967 / 0.984 | 0.965 / 0.959 | 0.999 / 0.987 | 0.923 | a role a page did NOT name itself now carries INHERITED_CONFIDENCE and says so; no prediction changed, so every rate is identical to round 3 |

**Held-Out Set**, round by round

| round | boring_log P/R | test_pit_log P/R | lab_test P/R | narrative P/R | calculation P/R | accuracy | what changed |
|---|---|---|---|---|---|---|---|
| 1 | - | - | - | - | - | - | first measurement of the rules on the live documents (all 14 reports; measured before the split) |
| 2 | - | - | - | - | - | - | dynamic-probing abbreviations (DPL/DPM/DPH/DPSH/DPT/LCPT) as DCP titles; chemical and grading laboratory titles; a tab of logs no longer claims a page of prose (all 14 reports; measured before the split) |
| 3 | 0.987 / 1.000 | 1.000 / 0.902 | 0.813 / 0.983 | 0.851 / 0.832 | 0.992 / 0.938 | 0.887 | the dev / held-out split, and document_outline + page_ledger added; NO rule changed this round |
| 4 | 0.987 / 1.000 | 1.000 / 0.902 | 0.813 / 0.983 | 0.851 / 0.832 | 0.992 / 0.938 | 0.887 | a role a page did NOT name itself now carries INHERITED_CONFIDENCE and says so; no prediction changed, so every rate is identical to round 3 |

### Where held-out lags development

- **`lab_test` lags by more than 0.05**: precision 0.967 to 0.813. The confusions that cost it on the held-out set: `other -> lab_test` (16 pages), `calculation -> lab_test` (14 pages), `narrative -> lab_test` (6 pages).
- **`narrative` lags by more than 0.05**: precision 0.965 to 0.851; recall 0.959 to 0.831. The confusions that cost it on the held-out set: `figure -> narrative` (7 pages), `narrative -> lab_test` (6 pages), `narrative -> other` (6 pages).

### This round in full

#### Held-Out Set

5 reports (R11, R13, R18, R21, R24), 1300 pages, 47 s.

| role | precision | recall | F1 | hand-labelled |
|---|---|---|---|---|
| `narrative` **(gated)** | 0.851 | 0.831 | 0.841 | 89 |
| `figure` | 0.250 | 0.217 | 0.233 | 23 |
| `plan` | 0.250 | 0.500 | 0.333 | 6 |
| `profile` | 0.875 | 0.467 | 0.609 | 15 |
| `boring_log` **(gated)** | 0.987 | 1.000 | 0.993 | 75 |
| `test_pit_log` **(gated)** | 1.000 | 0.902 | 0.948 | 51 |
| `cpt_log` | 0.438 | 1.000 | 0.609 | 7 |
| `dcp_log` | 0.000 | 0.000 | 0.000 | 0 |
| `lab_test` **(gated)** | 0.813 | 0.983 | 0.890 | 177 |
| `field_test` | 0.250 | 1.000 | 0.400 | 2 |
| `calculation` **(gated)** | 0.992 | 0.938 | 0.964 | 259 |
| `appended_report` | 1.000 | 1.000 | 1.000 | 472 |
| `photos` | 0.000 | 0.000 | 0.000 | 0 |
| `divider` | 0.318 | 0.824 | 0.459 | 17 |
| `cover` | 1.000 | 0.250 | 0.400 | 8 |
| `letter` | 1.000 | 0.500 | 0.667 | 4 |
| `toc` | 0.875 | 1.000 | 0.933 | 7 |
| `other` | 0.513 | 0.227 | 0.315 | 88 |

Page accuracy **0.887** over 1300 pages. The gate does NOT pass.

Confusion matrix (rows = hand label, columns = predicted):

| hand \ pred | narr | figu | plan | prof | bori | test | cpt_ | dcp_ | lab_ | fiel | calc | appe | phot | divi | cove | lett | toc | othe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `narrative` | 74 | . | . | . | . | . | 1 | . | 6 | . | . | . | . | 1 | . | . | 1 | 6 |
| `figure` | 7 | 5 | 1 | . | . | . | . | . | 2 | . | . | . | . | . | . | . | . | 8 |
| `plan` | 1 | . | 3 | 1 | . | . | . | . | . | . | . | . | . | 1 | . | . | . | . |
| `profile` | 2 | 2 | . | 7 | . | . | . | . | . | . | 2 | . | . | . | . | . | . | 2 |
| `boring_log` | . | . | . | . | 75 | . | . | . | . | . | . | . | . | . | . | . | . | . |
| `test_pit_log` | . | . | 5 | . | . | 46 | . | . | . | . | . | . | . | . | . | . | . | . |
| `cpt_log` | . | . | . | . | . | . | 7 | . | . | . | . | . | . | . | . | . | . | . |
| `dcp_log` | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| `lab_test` | . | 2 | . | . | . | . | . | . | 174 | . | . | . | . | 1 | . | . | . | . |
| `field_test` | . | . | . | . | . | . | . | . | . | 2 | . | . | . | . | . | . | . | . |
| `calculation` | . | . | . | . | . | . | . | . | 14 | . | 243 | . | . | 2 | . | . | . | . |
| `appended_report` | . | . | . | . | . | . | . | . | . | . | . | 472 | . | . | . | . | . | . |
| `photos` | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| `divider` | . | . | 1 | . | . | . | . | . | 2 | . | . | . | . | 14 | . | . | . | . |
| `cover` | 1 | 2 | . | . | . | . | . | . | . | . | . | . | . | . | 2 | . | . | 3 |
| `letter` | 2 | . | . | . | . | . | . | . | . | . | . | . | . | . | . | 2 | . | . |
| `toc` | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | 7 | . |
| `other` | . | 9 | 2 | . | 1 | . | 8 | 1 | 16 | 6 | . | . | . | 25 | . | . | . | 20 |

Column keys: `narr` = narrative, `figu` = figure, `plan` = plan, `prof` = profile, `bori` = boring_log, `test` = test_pit_log, `cpt_` = cpt_log, `dcp_` = dcp_log, `lab_` = lab_test, `fiel` = field_test, `calc` = calculation, `appe` = appended_report, `phot` = photos, `divi` = divider, `cove` = cover, `lett` = letter, `toc` = toc, `othe` = other.

Per-report accuracy:

| ID | pages | accuracy |
|---|---|---|
| R11 | 156 | 0.904 |
| R13 | 197 | 0.883 |
| R18 | 64 | 0.953 |
| R21 | 729 | 0.966 |
| R24 | 154 | 0.474 |

75 pages where a gated role is involved and the rules and the hand label disagree. They are NOT listed page by page: closing a gap by reading the held-out pages is how a held-out set stops being one. The confusions, by size:

| hand -> predicted | pages |
|---|---|
| `other -> lab_test` | 16 |
| `calculation -> lab_test` | 14 |
| `figure -> narrative` | 7 |
| `narrative -> lab_test` | 6 |
| `narrative -> other` | 6 |
| `test_pit_log -> plan` | 5 |
| `lab_test -> figure` | 2 |
| `letter -> narrative` | 2 |
| `profile -> calculation` | 2 |
| `calculation -> divider` | 2 |
| `profile -> narrative` | 2 |
| `figure -> lab_test` | 2 |
| `divider -> lab_test` | 2 |
| `lab_test -> divider` | 1 |
| `cover -> narrative` | 1 |
| `plan -> narrative` | 1 |
| `narrative -> toc` | 1 |
| `narrative -> divider` | 1 |
| `narrative -> cpt_log` | 1 |
| `other -> boring_log` | 1 |

#### Development Set

9 reports (R09, R12, R15, R16, R20, R23, R28, R29, R30), 2847 pages, 141 s.

| role | precision | recall | F1 | hand-labelled |
|---|---|---|---|---|
| `narrative` **(gated)** | 0.965 | 0.959 | 0.962 | 344 |
| `figure` | 0.559 | 0.500 | 0.528 | 38 |
| `plan` | 0.556 | 0.417 | 0.476 | 12 |
| `profile` | 0.900 | 0.692 | 0.783 | 13 |
| `boring_log` **(gated)** | 0.963 | 0.919 | 0.941 | 198 |
| `test_pit_log` **(gated)** | 0.949 | 0.925 | 0.937 | 200 |
| `cpt_log` | 1.000 | 0.875 | 0.933 | 16 |
| `dcp_log` | 1.000 | 0.556 | 0.714 | 54 |
| `lab_test` **(gated)** | 0.967 | 0.984 | 0.976 | 815 |
| `field_test` | 0.739 | 1.000 | 0.850 | 17 |
| `calculation` **(gated)** | 0.999 | 0.987 | 0.993 | 750 |
| `appended_report` | 0.815 | 1.000 | 0.898 | 22 |
| `photos` | 0.715 | 0.932 | 0.809 | 132 |
| `divider` | 0.713 | 0.864 | 0.781 | 66 |
| `cover` | 1.000 | 0.409 | 0.581 | 22 |
| `letter` | 1.000 | 1.000 | 1.000 | 2 |
| `toc` | 1.000 | 0.611 | 0.759 | 18 |
| `other` | 0.546 | 0.555 | 0.550 | 128 |

Page accuracy **0.923** over 2847 pages.

Confusion matrix (rows = hand label, columns = predicted):

| hand \ pred | narr | figu | plan | prof | bori | test | cpt_ | dcp_ | lab_ | fiel | calc | appe | phot | divi | cove | lett | toc | othe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `narrative` | 330 | . | 1 | . | . | 1 | . | . | 2 | . | . | 1 | . | . | . | . | . | 9 |
| `figure` | 3 | 19 | 2 | . | 1 | . | . | . | . | . | . | . | . | . | . | . | . | 13 |
| `plan` | . | 1 | 5 | 1 | . | . | . | . | . | 2 | 1 | . | 1 | . | . | . | . | 1 |
| `profile` | . | . | . | 9 | 1 | . | . | . | . | . | . | . | . | . | . | . | . | 3 |
| `boring_log` | . | 9 | . | . | 182 | 6 | . | . | . | . | . | . | 1 | . | . | . | . | . |
| `test_pit_log` | . | . | . | . | 1 | 185 | . | . | 2 | . | . | . | 11 | 1 | . | . | . | . |
| `cpt_log` | . | . | . | . | . | . | 14 | . | . | . | . | . | . | . | . | . | . | 2 |
| `dcp_log` | . | . | . | . | . | . | . | 30 | . | . | . | . | 10 | . | . | . | . | 14 |
| `lab_test` | . | . | . | . | . | . | . | . | 802 | 4 | . | . | . | . | . | . | . | 9 |
| `field_test` | . | . | . | . | . | . | . | . | . | 17 | . | . | . | . | . | . | . | . |
| `calculation` | . | . | . | . | . | . | . | . | 1 | . | 740 | . | . | 9 | . | . | . | . |
| `appended_report` | . | . | . | . | . | . | . | . | . | . | . | 22 | . | . | . | . | . | . |
| `photos` | . | . | . | . | . | . | . | . | 5 | . | . | . | 123 | 4 | . | . | . | . |
| `divider` | . | 1 | . | . | . | . | . | . | 1 | . | . | 4 | . | 57 | . | . | . | 3 |
| `cover` | 2 | 4 | . | . | . | . | . | . | 1 | . | . | . | . | 1 | 9 | . | . | 5 |
| `letter` | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | 2 | . | . |
| `toc` | 5 | . | 1 | . | . | . | . | . | 1 | . | . | . | . | . | . | . | 11 | . |
| `other` | 2 | . | . | . | 4 | 3 | . | . | 14 | . | . | . | 26 | 8 | . | . | . | 71 |

Column keys: `narr` = narrative, `figu` = figure, `plan` = plan, `prof` = profile, `bori` = boring_log, `test` = test_pit_log, `cpt_` = cpt_log, `dcp_` = dcp_log, `lab_` = lab_test, `fiel` = field_test, `calc` = calculation, `appe` = appended_report, `phot` = photos, `divi` = divider, `cove` = cover, `lett` = letter, `toc` = toc, `othe` = other.

Per-report accuracy:

| ID | pages | accuracy |
|---|---|---|
| R09 | 151 | 0.934 |
| R12 | 159 | 0.893 |
| R15 | 202 | 0.886 |
| R16 | 426 | 0.939 |
| R20 | 370 | 0.978 |
| R23 | 397 | 0.942 |
| R28 | 455 | 0.923 |
| R29 | 131 | 0.664 |
| R30 | 556 | 0.941 |

112 pages where a gated role is involved and the rules and the hand label disagree. `rule` and `why` are the evidence planlens itself recorded; the page HEADING is omitted on purpose -- the largest type on a log or a laboratory sheet is a firm's title block, and this file is tracked in a public repository. The same list WITH headings is written to `raw/checks/wp1_misses.txt`, which is gitignored.

**R09** (6)

```
R09 p11   hand=other           pred=narrative       kind=figure        rule="prose before the report's first appendix tab" why=''
R09 p13   hand=figure          pred=narrative       kind=figure        rule="prose before the report's first appendix tab" why=''
R09 p14   hand=figure          pred=narrative       kind=figure        rule="prose before the report's first appendix tab" why=''
R09 p15   hand=figure          pred=narrative       kind=figure        rule="prose before the report's first appendix tab" why=''
R09 p17   hand=other           pred=narrative       kind=mixed         rule="prose before the report's first appendix tab" why=''
R09 p150  hand=other           pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'laboratory testing'"
```

**R12** (13)

```
R12 p3    hand=toc             pred=narrative       kind=mixed         rule="prose before the report's first appendix tab" why=''
R12 p54   hand=figure          pred=boring_log      kind=figure        rule='the page names itself' why="log title 'borehole no', 2 log form fields"
R12 p61   hand=test_pit_log    pred=lab_test        kind=figure        rule='the page names itself' why="laboratory test title 'permeability' beside a reference to t"
R12 p62   hand=test_pit_log    pred=lab_test        kind=figure        rule='the page names itself' why="laboratory test title 'permeability' beside a reference to t"
R12 p64   hand=other           pred=lab_test        kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['lab_test']"
R12 p78   hand=other           pred=lab_test        kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['lab_test']"
R12 p96   hand=lab_test        pred=field_test      kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['field_test', 'lab_test']"
R12 p102  hand=lab_test        pred=field_test      kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['field_test', 'lab_test']"
R12 p108  hand=lab_test        pred=field_test      kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['field_test', 'lab_test']"
R12 p114  hand=lab_test        pred=field_test      kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['field_test', 'lab_test']"
R12 p122  hand=other           pred=lab_test        kind=figure        rule='the page names itself, and its appendix expects it' why='2 laboratory test terms on the page'
R12 p132  hand=other           pred=lab_test        kind=figure        rule='the page names itself' why="laboratory test title 'water extract'"
R12 p139  hand=other           pred=lab_test        kind=figure        rule='the page names itself, and its appendix expects it' why='2 laboratory test terms on the page'
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
R16 p29   hand=other           pred=boring_log      kind=mixed         rule='INHERITED from its appendix tab; the page says nothing about' why="['boring_log', 'photos']"
R16 p79   hand=other           pred=lab_test        kind=mixed         rule='the page names itself, and its appendix expects it' why="laboratory test title 'laboratory test'"
R16 p128  hand=cover           pred=lab_test        kind=mixed         rule='INHERITED from its appendix tab; the page says nothing about' why="['lab_test']"
R16 p418  hand=calculation     pred=divider         kind=text          rule='tab or cover page naming what follows' why=''
R16 p420  hand=calculation     pred=divider         kind=text          rule='tab or cover page naming what follows' why=''
```

**R20** (4)

```
R20 p6    hand=narrative       pred=other           kind=mixed         rule='mixed page with no title of its own' why=''
R20 p29   hand=narrative       pred=appended_report kind=text          rule='inside a report bound into this one' why=''
R20 p115  hand=test_pit_log    pred=divider         kind=mixed         rule='tab or cover page naming what follows' why=''
R20 p354  hand=calculation     pred=divider         kind=text          rule='tab or cover page naming what follows' why=''
```

**R23** (12)

```
R23 p4    hand=narrative       pred=other           kind=mixed         rule='mixed page with no title of its own' why=''
R23 p5    hand=narrative       pred=other           kind=text          rule='text page with no title of its own' why=''
R23 p170  hand=boring_log      pred=test_pit_log    kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['test_pit_log', 'photos']"
R23 p171  hand=boring_log      pred=test_pit_log    kind=mixed         rule='INHERITED from its appendix tab; the page says nothing about' why="['test_pit_log', 'photos']"
R23 p184  hand=boring_log      pred=test_pit_log    kind=mixed         rule='INHERITED from its appendix tab; the page says nothing about' why="['test_pit_log', 'photos']"
R23 p191  hand=boring_log      pred=test_pit_log    kind=text          rule='INHERITED from its appendix tab; the page says nothing about' why="['test_pit_log', 'photos']"
R23 p192  hand=boring_log      pred=test_pit_log    kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['test_pit_log', 'photos']"
R23 p194  hand=boring_log      pred=test_pit_log    kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['test_pit_log', 'photos']"
R23 p197  hand=other           pred=test_pit_log    kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['test_pit_log', 'photos']"
R23 p255  hand=other           pred=lab_test        kind=text          rule='INHERITED from its appendix tab; the page says nothing about' why="['lab_test']"
R23 p375  hand=calculation     pred=divider         kind=figure        rule='tab or cover page naming what follows' why=''
R23 p376  hand=calculation     pred=divider         kind=text          rule='tab or cover page naming what follows' why=''
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
R28 p253  hand=test_pit_log    pred=photos          kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['photos', 'test_pit_log']"
R28 p254  hand=test_pit_log    pred=photos          kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['photos', 'test_pit_log']"
R28 p255  hand=test_pit_log    pred=photos          kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['photos', 'test_pit_log']"
R28 p256  hand=test_pit_log    pred=photos          kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['photos', 'test_pit_log']"
R28 p257  hand=test_pit_log    pred=photos          kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['photos', 'test_pit_log']"
R28 p258  hand=test_pit_log    pred=photos          kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['photos', 'test_pit_log']"
R28 p259  hand=test_pit_log    pred=photos          kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['photos', 'test_pit_log']"
R28 p260  hand=test_pit_log    pred=photos          kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['photos', 'test_pit_log']"
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
R30 p58   hand=other           pred=boring_log      kind=mixed         rule='INHERITED from its appendix tab; the page says nothing about' why="['boring_log']"
R30 p131  hand=other           pred=test_pit_log    kind=mixed         rule='INHERITED from its appendix tab; the page says nothing about' why="['test_pit_log']"
R30 p228  hand=other           pred=lab_test        kind=mixed         rule='the page names itself, and its appendix expects it' why="laboratory test title 'laboratory test'"
R30 p296  hand=other           pred=lab_test        kind=mixed         rule='the page names itself, and its appendix expects it' why="laboratory test title 'laboratory test'"
R30 p297  hand=other           pred=lab_test        kind=mixed         rule='the page names itself, and its appendix expects it' why="laboratory test title 'laboratory testing'"
R30 p385  hand=other           pred=lab_test        kind=mixed         rule='the page names itself, and its appendix expects it' why="laboratory test title 'laboratory testing'"
R30 p418  hand=photos          pred=lab_test        kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['lab_test']"
R30 p419  hand=photos          pred=lab_test        kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['lab_test']"
R30 p430  hand=other           pred=lab_test        kind=mixed         rule='the page names itself, and its appendix expects it' why="laboratory test title 'laboratory testing'"
R30 p471  hand=photos          pred=lab_test        kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['lab_test']"
R30 p472  hand=photos          pred=lab_test        kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['lab_test']"
R30 p473  hand=photos          pred=lab_test        kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['lab_test']"
R30 p523  hand=calculation     pred=divider         kind=mixed         rule='tab or cover page naming what follows' why=''
R30 p543  hand=calculation     pred=divider         kind=text          rule='tab or cover page naming what follows' why=''
R30 p545  hand=calculation     pred=divider         kind=text          rule='tab or cover page naming what follows' why=''
R30 p547  hand=calculation     pred=divider         kind=text          rule='tab or cover page naming what follows' why=''
```

---

### WP1 -- the held-out five were already looked at (declare it, do not hide it)

The split arrived after the rules were written, and by then every one of the
five held-out reports had been inspected. Naming them, because a held-out set
whose contamination is not written down is worse than no held-out set:

| ID | what was looked at, and what it changed |
|---|---|
| R11 | Its structure was printed and its misses read. Its cover page was being read as a tab that declared an appended report, which swallowed the whole document; that is why an appended report may now be opened only by a tab that carries an appendix letter inside a document that has already opened one. |
| R13 | Its dividers and its narrative block were printed. Its slope-stability printouts headed "SECTION A-A'" were being read as tabs; that is why "section" is not a divider word. Its first page was being read as a tab; that is why page 0 cannot be one. |
| R18 | Used as the end-to-end smoke test throughout. Its roles, its items and its segment profile were printed repeatedly. |
| R21 | The whole nested-document algorithm was designed against it: the appendix-letter stack, the point where the outer report resumes, and "a table of contents is not a tab" all come from reading its pages. |
| R24 | Its misses were read and four of its pages were RENDERED and looked at. Stripping the running header from a tab's declaration, reading sub-tabs ("PART B2"), and refusing to call a page of prose a log all come from it. |

So the held-out column in round 3 and round 4 is not a measurement of
generalisation. It is the same in-sample fit, reported separately. The
numbers are still worth having -- they show the five hardest documents
failing differently from the nine easier ones -- but they must not be quoted
as held-out performance.

What this makes true anyway, and what it does not:

- **True.** `boring_log` and `test_pit_log` hold up on the five hardest
  documents (0.987/1.000 and 1.000/0.902) while `lab_test` precision and
  `narrative` do not (0.813 and 0.851/0.832). The five were chosen for being
  unalike, and the roles that depend on a page naming itself survive that
  while the roles that lean on the appendix tab do not.
- **Not true.** That the rules will reach the gate on a report nobody here
  has seen. The lead's own out-of-sample check on 120 pages drawn from the
  24 unlabelled reports is the honest number, and it is far lower.

From here the discipline holds: nothing is tuned against R11, R13, R18, R21
or R24 again, and the next rule change is measured on a set that is genuinely
blind. The clean holdout available without any new labelling is the label
sheet whose PDF was never copied into the corpus (153 rows, plan section 7
item 3): drop that PDF in as R39 and it is a report with hand labels that
nothing here has ever opened.

---

## WP1 -- page roles -- round 5

Measured 2026-09-16 with the app venv and the editable planlens checkout. 4147 hand-labelled pages, opened `di="auto"`, scored against `labels.labels_for`, split into a development set (R09, R12, R15, R16, R20, R23, R28, R29, R30) and a held-out set (R11, R13, R18, R21, R24). **The gate is on the held-out set**: precision AND recall >= 0.90 on `boring_log`, `test_pit_log`, `lab_test`, `narrative`, `calculation`.

**This round:** the four principles: a page that names itself beats its tab, prose needs narrative evidence, a tab that names several things chooses on the page or says other, and a nested report records its inner role

**Development Set**, round by round

| round | boring_log P/R | test_pit_log P/R | lab_test P/R | narrative P/R | calculation P/R | accuracy | what changed |
|---|---|---|---|---|---|---|---|
| 1 | 0.970 / 0.941 | 0.959 / 0.920 | 0.936 / 0.953 | 0.942 / 0.933 | 0.997 / 0.974 | 0.900 | first measurement of the rules on the live documents (all 14 reports; measured before the split) |
| 2 | 0.970 / 0.941 | 0.959 / 0.920 | 0.936 / 0.984 | 0.942 / 0.933 | 0.997 / 0.974 | 0.912 | dynamic-probing abbreviations (DPL/DPM/DPH/DPSH/DPT/LCPT) as DCP titles; chemical and grading laboratory titles; a tab of logs no longer claims a page of prose (all 14 reports; measured before the split) |
| 3 | 0.963 / 0.919 | 0.949 / 0.925 | 0.967 / 0.984 | 0.965 / 0.959 | 0.999 / 0.987 | 0.923 | the dev / held-out split, and document_outline + page_ledger added; NO rule changed this round |
| 4 | 0.963 / 0.919 | 0.949 / 0.925 | 0.967 / 0.984 | 0.965 / 0.959 | 0.999 / 0.987 | 0.923 | a role a page did NOT name itself now carries INHERITED_CONFIDENCE and says so; no prediction changed, so every rate is identical to round 3 |
| 5 | 0.948 / 0.919 | 0.947 / 0.885 | 0.958 / 0.989 | 0.976 / 0.948 | 0.999 / 0.995 | 0.920 | the four principles: a page that names itself beats its tab, prose needs narrative evidence, a tab that names several things chooses on the page or says other, and a nested report records its inner role |

**Held-Out Set**, round by round

| round | boring_log P/R | test_pit_log P/R | lab_test P/R | narrative P/R | calculation P/R | accuracy | what changed |
|---|---|---|---|---|---|---|---|
| 1 | - | - | - | - | - | - | first measurement of the rules on the live documents (all 14 reports; measured before the split) |
| 2 | - | - | - | - | - | - | dynamic-probing abbreviations (DPL/DPM/DPH/DPSH/DPT/LCPT) as DCP titles; chemical and grading laboratory titles; a tab of logs no longer claims a page of prose (all 14 reports; measured before the split) |
| 3 | 0.987 / 1.000 | 1.000 / 0.902 | 0.813 / 0.983 | 0.851 / 0.832 | 0.992 / 0.938 | 0.887 | the dev / held-out split, and document_outline + page_ledger added; NO rule changed this round |
| 4 | 0.987 / 1.000 | 1.000 / 0.902 | 0.813 / 0.983 | 0.851 / 0.832 | 0.992 / 0.938 | 0.887 | a role a page did NOT name itself now carries INHERITED_CONFIDENCE and says so; no prediction changed, so every rate is identical to round 3 |
| 5 | 0.987 / 1.000 | 1.000 / 0.902 | 0.813 / 0.983 | 0.959 / 0.786 | 0.971 / 0.903 | 0.880 | the four principles: a page that names itself beats its tab, prose needs narrative evidence, a tab that names several things chooses on the page or says other, and a nested report records its inner role |

### Where held-out lags development

- **`lab_test` lags by more than 0.05**: precision 0.958 to 0.813. The confusions that cost it on the held-out set: `other -> lab_test` (14 pages), `calculation -> lab_test` (14 pages), `narrative -> lab_test` (7 pages).
- **`narrative` lags by more than 0.05**: recall 0.948 to 0.787. The confusions that cost it on the held-out set: `narrative -> lab_test` (7 pages), `narrative -> other` (7 pages), `narrative -> plan` (2 pages).
- **`calculation` lags by more than 0.05**: recall 0.995 to 0.903. The confusions that cost it on the held-out set: `calculation -> lab_test` (14 pages), `calculation -> profile` (8 pages), `figure -> calculation` (5 pages).

### This round in full

#### Held-Out Set

5 reports (R11, R13, R18, R21, R24), 1300 pages, 46 s.

| role | precision | recall | F1 | hand-labelled |
|---|---|---|---|---|
| `narrative` **(gated)** | 0.959 | 0.787 | 0.864 | 89 |
| `figure` | 0.273 | 0.261 | 0.267 | 23 |
| `plan` | 0.312 | 0.833 | 0.455 | 6 |
| `profile` | 0.438 | 0.467 | 0.452 | 15 |
| `boring_log` **(gated)** | 0.987 | 1.000 | 0.993 | 75 |
| `test_pit_log` **(gated)** | 1.000 | 0.902 | 0.948 | 51 |
| `cpt_log` | 0.368 | 1.000 | 0.538 | 7 |
| `dcp_log` | 0.000 | 0.000 | 0.000 | 0 |
| `lab_test` **(gated)** | 0.813 | 0.983 | 0.890 | 177 |
| `field_test` | 0.250 | 1.000 | 0.400 | 2 |
| `calculation` **(gated)** | 0.971 | 0.903 | 0.936 | 259 |
| `appended_report` | 1.000 | 1.000 | 1.000 | 472 |
| `photos` | 0.000 | 0.000 | 0.000 | 0 |
| `divider` | 0.333 | 0.824 | 0.475 | 17 |
| `cover` | 1.000 | 0.250 | 0.400 | 8 |
| `letter` | 1.000 | 0.500 | 0.667 | 4 |
| `toc` | 0.875 | 1.000 | 0.933 | 7 |
| `other` | 0.500 | 0.239 | 0.323 | 88 |

Page accuracy **0.880** over 1300 pages. The gate does NOT pass.

Confusion matrix (rows = hand label, columns = predicted):

| hand \ pred | narr | figu | plan | prof | bori | test | cpt_ | dcp_ | lab_ | fiel | calc | appe | phot | divi | cove | lett | toc | othe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `narrative` | 70 | . | 2 | . | . | . | 1 | . | 7 | . | . | . | . | 1 | . | . | 1 | 7 |
| `figure` | . | 6 | 3 | . | . | . | . | . | 3 | . | 5 | . | . | . | . | . | . | 6 |
| `plan` | . | . | 5 | 1 | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| `profile` | 2 | 2 | . | 7 | . | . | . | . | . | . | 2 | . | . | . | . | . | . | 2 |
| `boring_log` | . | . | . | . | 75 | . | . | . | . | . | . | . | . | . | . | . | . | . |
| `test_pit_log` | . | . | 5 | . | . | 46 | . | . | . | . | . | . | . | . | . | . | . | . |
| `cpt_log` | . | . | . | . | . | . | 7 | . | . | . | . | . | . | . | . | . | . | . |
| `dcp_log` | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| `lab_test` | . | 2 | . | . | . | . | . | . | 174 | . | . | . | . | 1 | . | . | . | . |
| `field_test` | . | . | . | . | . | . | . | . | . | 2 | . | . | . | . | . | . | . | . |
| `calculation` | . | . | 1 | 8 | . | . | 1 | . | 14 | . | 234 | . | . | 1 | . | . | . | . |
| `appended_report` | . | . | . | . | . | . | . | . | . | . | . | 472 | . | . | . | . | . | . |
| `photos` | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| `divider` | . | . | . | . | . | . | . | . | 2 | . | . | . | . | 14 | . | . | . | 1 |
| `cover` | . | 2 | . | . | . | . | . | . | . | . | . | . | . | . | 2 | . | . | 4 |
| `letter` | 1 | . | . | . | . | . | . | . | . | . | . | . | . | . | . | 2 | . | 1 |
| `toc` | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | 7 | . |
| `other` | . | 10 | . | . | 1 | . | 10 | 1 | 14 | 6 | . | . | . | 25 | . | . | . | 21 |

Column keys: `narr` = narrative, `figu` = figure, `plan` = plan, `prof` = profile, `bori` = boring_log, `test` = test_pit_log, `cpt_` = cpt_log, `dcp_` = dcp_log, `lab_` = lab_test, `fiel` = field_test, `calc` = calculation, `appe` = appended_report, `phot` = photos, `divi` = divider, `cove` = cover, `lett` = letter, `toc` = toc, `othe` = other.

Per-report accuracy:

| ID | pages | accuracy |
|---|---|---|
| R11 | 156 | 0.904 |
| R13 | 197 | 0.843 |
| R18 | 64 | 0.969 |
| R21 | 729 | 0.962 |
| R24 | 154 | 0.481 |

82 pages where a gated role is involved and the rules and the hand label disagree. They are NOT listed page by page: closing a gap by reading the held-out pages is how a held-out set stops being one. The confusions, by size:

| hand -> predicted | pages |
|---|---|
| `other -> lab_test` | 14 |
| `calculation -> lab_test` | 14 |
| `calculation -> profile` | 8 |
| `narrative -> lab_test` | 7 |
| `narrative -> other` | 7 |
| `figure -> calculation` | 5 |
| `test_pit_log -> plan` | 5 |
| `figure -> lab_test` | 3 |
| `lab_test -> figure` | 2 |
| `narrative -> plan` | 2 |
| `profile -> calculation` | 2 |
| `profile -> narrative` | 2 |
| `divider -> lab_test` | 2 |
| `lab_test -> divider` | 1 |
| `letter -> narrative` | 1 |
| `calculation -> plan` | 1 |
| `calculation -> cpt_log` | 1 |
| `calculation -> divider` | 1 |
| `narrative -> toc` | 1 |
| `narrative -> divider` | 1 |
| `narrative -> cpt_log` | 1 |
| `other -> boring_log` | 1 |

#### Development Set

9 reports (R09, R12, R15, R16, R20, R23, R28, R29, R30), 2847 pages, 140 s.

| role | precision | recall | F1 | hand-labelled |
|---|---|---|---|---|
| `narrative` **(gated)** | 0.976 | 0.948 | 0.962 | 344 |
| `figure` | 0.264 | 0.368 | 0.308 | 38 |
| `plan` | 0.294 | 0.417 | 0.345 | 12 |
| `profile` | 0.750 | 0.692 | 0.720 | 13 |
| `boring_log` **(gated)** | 0.948 | 0.919 | 0.933 | 198 |
| `test_pit_log` **(gated)** | 0.947 | 0.885 | 0.915 | 200 |
| `cpt_log` | 1.000 | 0.875 | 0.933 | 16 |
| `dcp_log` | 0.967 | 0.537 | 0.690 | 54 |
| `lab_test` **(gated)** | 0.958 | 0.989 | 0.973 | 815 |
| `field_test` | 0.882 | 0.882 | 0.882 | 17 |
| `calculation` **(gated)** | 0.999 | 0.995 | 0.997 | 750 |
| `appended_report` | 0.815 | 1.000 | 0.898 | 22 |
| `photos` | 0.904 | 0.856 | 0.879 | 132 |
| `divider` | 0.864 | 0.864 | 0.864 | 66 |
| `cover` | 1.000 | 0.409 | 0.581 | 22 |
| `letter` | 1.000 | 1.000 | 1.000 | 2 |
| `toc` | 1.000 | 0.611 | 0.759 | 18 |
| `other` | 0.509 | 0.648 | 0.570 | 128 |

Page accuracy **0.920** over 2847 pages.

Confusion matrix (rows = hand label, columns = predicted):

| hand \ pred | narr | figu | plan | prof | bori | test | cpt_ | dcp_ | lab_ | fiel | calc | appe | phot | divi | cove | lett | toc | othe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `narrative` | 326 | 2 | 4 | 1 | . | . | . | . | 5 | . | . | 1 | 1 | . | . | . | . | 4 |
| `figure` | 3 | 14 | 6 | . | 1 | . | . | . | . | . | . | . | . | . | . | . | . | 14 |
| `plan` | . | . | 5 | 1 | . | . | . | . | . | 2 | 1 | . | 1 | . | . | . | . | 2 |
| `profile` | . | . | . | 9 | 1 | . | . | . | . | . | . | . | . | . | . | . | . | 3 |
| `boring_log` | . | 9 | . | . | 182 | 3 | . | . | . | . | . | . | 1 | . | . | . | . | 3 |
| `test_pit_log` | . | . | . | . | 1 | 177 | . | . | 2 | . | . | . | 3 | 1 | . | . | . | 16 |
| `cpt_log` | . | . | . | . | . | . | 14 | . | . | . | . | . | . | . | . | . | . | 2 |
| `dcp_log` | . | 15 | . | . | . | . | . | 29 | . | . | . | . | . | . | . | . | . | 10 |
| `lab_test` | . | . | . | . | . | . | . | . | 806 | . | . | . | . | . | . | . | . | 9 |
| `field_test` | . | . | . | . | . | . | . | . | 2 | 15 | . | . | . | . | . | . | . | . |
| `calculation` | . | . | . | 1 | . | . | . | . | 2 | . | 746 | . | . | 1 | . | . | . | . |
| `appended_report` | . | . | . | . | . | . | . | . | . | . | . | 22 | . | . | . | . | . | . |
| `photos` | . | . | . | . | 1 | 2 | . | . | 5 | . | . | . | 113 | 4 | . | . | . | 7 |
| `divider` | . | 2 | . | . | . | . | . | . | 1 | . | . | 4 | . | 57 | . | . | . | 2 |
| `cover` | 1 | 3 | . | . | . | . | . | . | 1 | . | . | . | . | . | 9 | . | . | 8 |
| `letter` | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | 2 | . | . |
| `toc` | 4 | . | 1 | . | . | . | . | . | 2 | . | . | . | . | . | . | . | 11 | . |
| `other` | . | 8 | 1 | . | 6 | 5 | . | 1 | 15 | . | . | . | 6 | 3 | . | . | . | 83 |

Column keys: `narr` = narrative, `figu` = figure, `plan` = plan, `prof` = profile, `bori` = boring_log, `test` = test_pit_log, `cpt_` = cpt_log, `dcp_` = dcp_log, `lab_` = lab_test, `fiel` = field_test, `calc` = calculation, `appe` = appended_report, `phot` = photos, `divi` = divider, `cove` = cover, `lett` = letter, `toc` = toc, `othe` = other.

Per-report accuracy:

| ID | pages | accuracy |
|---|---|---|
| R09 | 151 | 0.901 |
| R12 | 159 | 0.899 |
| R15 | 202 | 0.871 |
| R16 | 426 | 0.948 |
| R20 | 370 | 0.968 |
| R23 | 397 | 0.937 |
| R28 | 455 | 0.901 |
| R29 | 131 | 0.733 |
| R30 | 556 | 0.944 |

121 pages where a gated role is involved and the rules and the hand label disagree. `rule` and `why` are the evidence planlens itself recorded; the page HEADING is omitted on purpose -- the largest type on a log or a laboratory sheet is a firm's title block, and this file is tracked in a public repository. The same list WITH headings is written to `raw/checks/wp1_misses.txt`, which is gitignored.

**R09** (10)

```
R09 p1    hand=narrative       pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R09 p2    hand=narrative       pred=other           kind=figure        rule='the page names itself' why="legend or notes sheet 'conversion factors'"
R09 p4    hand=narrative       pred=lab_test        kind=figure        rule='the page names itself' why='11 laboratory test terms on the page'
R09 p6    hand=narrative       pred=profile         kind=figure        rule='the page names itself' why="profile title 'probable soil'"
R09 p8    hand=narrative       pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R09 p13   hand=figure          pred=narrative       kind=figure        rule="prose carrying the narrative's running footer" why=''
R09 p14   hand=figure          pred=narrative       kind=figure        rule="prose carrying the narrative's running footer" why=''
R09 p15   hand=figure          pred=narrative       kind=figure        rule="prose carrying the narrative's running footer" why=''
R09 p17   hand=other           pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test name 'liquid limit' on a page of working"
R09 p150  hand=other           pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test name 'laboratory testing' on a page of worki"
```

**R12** (13)

```
R12 p1    hand=cover           pred=narrative       kind=figure        rule="prose carrying the narrative's running header" why=''
R12 p3    hand=toc             pred=narrative       kind=mixed         rule="prose carrying the narrative's running header" why=''
R12 p22   hand=narrative       pred=lab_test        kind=figure        rule='the page names itself' why='7 laboratory test terms on the page'
R12 p54   hand=figure          pred=boring_log      kind=figure        rule='the page names itself' why="log title 'borehole no', 2 log form fields"
R12 p61   hand=test_pit_log    pred=lab_test        kind=figure        rule='the page names itself' why="laboratory test title 'permeability' beside a reference to t"
R12 p62   hand=test_pit_log    pred=lab_test        kind=figure        rule='the page names itself' why="laboratory test title 'permeability' beside a reference to t"
R12 p64   hand=other           pred=lab_test        kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['lab_test']"
R12 p78   hand=other           pred=lab_test        kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['lab_test']"
R12 p116  hand=field_test      pred=lab_test        kind=figure        rule='INHERITED from a tab that names several things; chosen on th' why='a results form or a plotted result'
R12 p117  hand=field_test      pred=lab_test        kind=figure        rule='INHERITED from a tab that names several things; chosen on th' why='a results form or a plotted result'
R12 p122  hand=other           pred=lab_test        kind=figure        rule='the page names itself' why='2 laboratory test terms on the page'
R12 p132  hand=other           pred=lab_test        kind=figure        rule='the page names itself' why="laboratory test title 'water extract'"
R12 p139  hand=other           pred=lab_test        kind=figure        rule='the page names itself' why='2 laboratory test terms on the page'
```

**R15** (18)

```
R15 p3    hand=toc             pred=narrative       kind=text          rule="prose carrying the narrative's running header" why=''
R15 p5    hand=narrative       pred=plan            kind=mixed         rule='the page names itself' why="plan title 'site plans'"
R15 p11   hand=narrative       pred=photos          kind=text          rule='the page names itself' why='photograph caption (2)'
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

**R16** (4)

```
R16 p7    hand=narrative       pred=other           kind=text          rule='text page with no title of its own' why=''
R16 p27   hand=profile         pred=boring_log      kind=form          rule='the page names itself' why='log form shape, 6 log form fields'
R16 p79   hand=other           pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'laboratory test'"
R16 p128  hand=cover           pred=lab_test        kind=mixed         rule='INHERITED from its appendix tab; the page says nothing about' why="['lab_test']"
```

**R20** (4)

```
R20 p8    hand=narrative       pred=plan            kind=text          rule='the page names itself' why="plan title 'geologic map'"
R20 p29   hand=narrative       pred=appended_report kind=text          rule='inside a report bound into this one' why=''
R20 p115  hand=test_pit_log    pred=divider         kind=mixed         rule='tab or cover page naming what follows' why=''
R20 p365  hand=calculation     pred=profile         kind=form          rule='the page names itself' why="profile title 'cross section'"
```

**R23** (16)

```
R23 p5    hand=narrative       pred=other           kind=text          rule='text page with no title of its own' why=''
R23 p170  hand=boring_log      pred=test_pit_log    kind=figure        rule='its tab named several things and the page names none; the ru' why=''
R23 p171  hand=boring_log      pred=test_pit_log    kind=mixed         rule='its tab named several things and the page names none; the ru' why=''
R23 p184  hand=boring_log      pred=test_pit_log    kind=mixed         rule='its tab named several things and the page names none; the ru' why=''
R23 p186  hand=test_pit_log    pred=other           kind=figure        rule='its appendix tab names several things and the page names non' why=''
R23 p187  hand=test_pit_log    pred=other           kind=text          rule='its appendix tab names several things and the page names non' why=''
R23 p188  hand=test_pit_log    pred=other           kind=figure        rule='its appendix tab names several things and the page names non' why=''
R23 p189  hand=test_pit_log    pred=other           kind=text          rule='its appendix tab names several things and the page names non' why=''
R23 p190  hand=test_pit_log    pred=other           kind=figure        rule='its appendix tab names several things and the page names non' why=''
R23 p191  hand=boring_log      pred=other           kind=text          rule='its appendix tab names several things and the page names non' why=''
R23 p192  hand=boring_log      pred=other           kind=figure        rule='its appendix tab names several things and the page names non' why=''
R23 p193  hand=test_pit_log    pred=other           kind=text          rule='its appendix tab names several things and the page names non' why=''
R23 p194  hand=boring_log      pred=other           kind=figure        rule='its appendix tab names several things and the page names non' why=''
R23 p195  hand=test_pit_log    pred=other           kind=text          rule='its appendix tab names several things and the page names non' why=''
R23 p196  hand=test_pit_log    pred=other           kind=figure        rule='its appendix tab names several things and the page names non' why=''
R23 p255  hand=other           pred=lab_test        kind=text          rule='INHERITED from its appendix tab; the page says nothing about' why="['lab_test']"
```

**R28** (16)

```
R28 p4    hand=toc             pred=narrative       kind=text          rule="prose carrying the narrative's running header" why=''
R28 p5    hand=toc             pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'laboratory test'"
R28 p53   hand=other           pred=boring_log      kind=text          rule='the page names itself' why="log title 'test borings', 15 log form fields"
R28 p56   hand=other           pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'sieve' beside a reference to the expl"
R28 p233  hand=test_pit_log    pred=photos          kind=figure        rule='the page names itself' why='photograph caption (1)'
R28 p236  hand=test_pit_log    pred=photos          kind=figure        rule='the page names itself' why='photograph caption (1)'
R28 p238  hand=test_pit_log    pred=photos          kind=figure        rule='the page names itself' why='photograph caption (1)'
R28 p253  hand=test_pit_log    pred=other           kind=figure        rule='its appendix tab names several things and the page names non' why=''
R28 p254  hand=test_pit_log    pred=other           kind=figure        rule='its appendix tab names several things and the page names non' why=''
R28 p255  hand=test_pit_log    pred=other           kind=figure        rule='its appendix tab names several things and the page names non' why=''
R28 p256  hand=test_pit_log    pred=other           kind=figure        rule='its appendix tab names several things and the page names non' why=''
R28 p257  hand=test_pit_log    pred=other           kind=figure        rule='its appendix tab names several things and the page names non' why=''
R28 p258  hand=test_pit_log    pred=other           kind=figure        rule='its appendix tab names several things and the page names non' why=''
R28 p259  hand=test_pit_log    pred=other           kind=figure        rule='its appendix tab names several things and the page names non' why=''
R28 p260  hand=test_pit_log    pred=other           kind=figure        rule='its appendix tab names several things and the page names non' why=''
R28 p414  hand=calculation     pred=lab_test        kind=text          rule='the page names itself' why="laboratory test title 'consolidation'"
```

**R29** (16)

```
R29 p3    hand=toc             pred=narrative       kind=text          rule="prose carrying the narrative's running header" why=''
R29 p5    hand=toc             pred=lab_test        kind=text          rule='the page names itself' why="laboratory test name 'california bearing ratio' on a page of"
R29 p8    hand=narrative       pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'moisture content'"
R29 p11   hand=narrative       pred=plan            kind=text          rule='the page names itself' why="plan title 'geologic map'"
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

**R30** (24)

```
R30 p5    hand=narrative       pred=plan            kind=text          rule='the page names itself' why="plan title 'site and vicinity'"
R30 p49   hand=plan            pred=calculation     kind=form          rule='the page names itself' why="calculation heading 'global stability'"
R30 p58   hand=other           pred=boring_log      kind=mixed         rule='INHERITED from its appendix tab; the page says nothing about' why="['boring_log']"
R30 p81   hand=other           pred=boring_log      kind=mixed         rule='INHERITED from its appendix tab; the page says nothing about' why="['boring_log']"
R30 p82   hand=photos          pred=boring_log      kind=form          rule='INHERITED from its appendix tab; the page says nothing about' why="['boring_log']"
R30 p83   hand=other           pred=boring_log      kind=mixed         rule='INHERITED from its appendix tab; the page says nothing about' why="['boring_log']"
R30 p118  hand=other           pred=boring_log      kind=mixed         rule='INHERITED from its appendix tab; the page says nothing about' why="['boring_log']"
R30 p131  hand=other           pred=test_pit_log    kind=mixed         rule='INHERITED from its appendix tab; the page says nothing about' why="['test_pit_log']"
R30 p139  hand=other           pred=test_pit_log    kind=mixed         rule='INHERITED from its appendix tab; the page says nothing about' why="['test_pit_log']"
R30 p150  hand=other           pred=test_pit_log    kind=mixed         rule='INHERITED from its appendix tab; the page says nothing about' why="['test_pit_log']"
R30 p196  hand=other           pred=test_pit_log    kind=mixed         rule='INHERITED from its appendix tab; the page says nothing about' why="['test_pit_log']"
R30 p198  hand=photos          pred=test_pit_log    kind=mixed         rule='INHERITED from its appendix tab; the page says nothing about' why="['test_pit_log']"
R30 p199  hand=photos          pred=test_pit_log    kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['test_pit_log']"
R30 p228  hand=other           pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'laboratory test'"
R30 p296  hand=other           pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'laboratory test'"
R30 p297  hand=other           pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'laboratory testing'"
R30 p385  hand=other           pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'laboratory testing'"
R30 p418  hand=photos          pred=lab_test        kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['lab_test']"
R30 p419  hand=photos          pred=lab_test        kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['lab_test']"
R30 p430  hand=other           pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'laboratory testing'"
R30 p471  hand=photos          pred=lab_test        kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['lab_test']"
R30 p472  hand=photos          pred=lab_test        kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['lab_test']"
R30 p473  hand=photos          pred=lab_test        kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['lab_test']"
R30 p523  hand=calculation     pred=divider         kind=mixed         rule='tab or cover page naming what follows' why=''
```

---

### WP1 round 5 -- the four principles, and what they cost in sample

The lead's out-of-sample check found two failure classes that belong to the
rules rather than to the labels. The fix was made on principle, against the
OPEN half of that check only -- ten reports whose misses the lead handed
over. The other fourteen stayed shut; nothing here opened, rendered or scored
them.

#### The open half, before and after

Fifty pages, five from each of ten reports, hand-labelled by the lead with a
primary label and acceptable alternates.

| | strict | accepting alternates |
|---|---|---|
| round 4 | 0.74 | 0.80 |
| round 5 | **0.76** | **0.84** |

Of the thirteen misses the lead named, seven are fixed: a cross-section and a
boring log that their tabs were overriding, a guide-specification page and a
contents page that were being read as tabs, an aerial photograph under a plan,
and two figure pages the narrative block had swallowed. Three more became
alternates-accepted rather than strict passes. Three did not move and one is
by design:

- Two pages of a report bound inside another are still claimed by the outer
  tab, because detecting that nesting needs a phrase ("prior explorations")
  that would also flip a thirteen-page appendix in the labelled corpus the
  other way. Left alone rather than traded.
- One page inside a nested report that is a test pit log is still reported as
  `appended_report`. That is principle D: the role is the binding, and the
  log is recorded as `inner_role` in the evidence.
- Three new misses appeared where a page's own title now wins and the hand
  label disagrees (two logs in the body of a report that the narrative block
  still holds, one map page).

#### What each principle cost, in sample

Every gated rate that moved by more than 0.01 against round 4, and why.

| set | role | round 4 | round 5 | why |
|---|---|---|---|---|
| dev | `test_pit_log` recall | 0.925 | 0.885 | **Principle C.** A tab naming several things no longer falls back on the first thing it names. 28 pages that used to be claimed on the tab's word order now say `other` with the candidates listed. A run of such pages closed on both sides by one log is still filled in, at 0.55 confidence; a run that is not, is not. |
| dev | `boring_log` precision | 0.963 | 0.948 | The same run-filling, going the other way: it claims a few pages for a log that the hand labels call something else. |
| dev | `narrative` recall | 0.959 | 0.948 | **Principle B.** Prose must now carry the narrative's running band, its printed numbering or prose density. Sitting in front of the first tab is a position, not evidence. |
| dev | `narrative` precision | 0.965 | 0.976 | The same change, on the other side of the ledger. |
| held-out | `narrative` recall / precision | 0.832 / 0.851 | 0.787 / 0.959 | B again, and harder here: these five documents are the ones whose narrative is least conventional. Precision rose 0.11 and recall fell 0.05. The confusions are `narrative -> lab_test` (7) and `narrative -> other` (7). |
| held-out | `calculation` recall | 0.938 | 0.903 | **Principle A**, and this is its price. A settlement worksheet that titles itself "One Dimensional Consolidation" now beats its CALCULATIONS tab, because the rule cannot tell a test's name from the name of the method that uses it when the page is a ruled form rather than a page of prose. `calculation -> lab_test` (14) and `calculation -> profile` (8). A page of WORKING is already protected; a ruled worksheet is not. |
| held-out | `calculation` precision | 0.992 | 0.971 | `figure -> calculation` (5), the other side of the same rule. |

Page accuracy: development 0.923 to 0.920, held-out 0.887 to 0.880, open half
0.74 to 0.76 strict. The principles trade a little in-sample fit for
out-of-sample behaviour, which is what they were for, and the trade is small
in both directions. The held-out numbers remain contaminated -- see "the
held-out five were already looked at" above -- so the open half is the only
column here worth weighing.

#### What is now said out loud rather than guessed

- A role a page did not name itself carries `INHERITED_CONFIDENCE` and its
  evidence says `INHERITED`.
- A page its tab could not place says `other` with `candidates` listed and a
  confidence under 0.5, rather than a confident wrong role.
- A page filled in from the run it sits in says `between-pages-of-one-log` at
  0.55.
- A document that prints no appendix tab at all is marked `no_dividers` in
  its outline, and nothing in it inherits a role from anything.
- A page inside a report bound into this one records what it would have been
  as `inner_role`.

Every one of those is a row the model review pass should look at first.

## WP1 -- close-out: the blind half after round 5 (planlens 04aac36)

The 70 pages of the fourteen reports the builder never opened (R17, R19, R22,
R25, R26, R27, R31-R38), scored before and after round 5:

| blind half, 70 pages | strict | accepting alternates |
|---|---|---|
| round 2 rules (from the 120-page run above) | 0.714 | 0.771 |
| round 5 rules | 0.786 | 0.857 |
| round 5, excluding R38's four no-text scans | 0.833 | 0.909 |

Key content on the blind half after round 5 (support / recall / precision):
lab_test 18 / 1.00 / 0.95 (the 1991 typed data report is now right on all
five of its pages); calculation 11 / 1.00 / 0.92; boring_log 9 / 0.78 /
0.88; narrative 12 / 0.83 / 0.91; test_pit_log 2 / 1.00 / 1.00.

Every remaining strict miss is a page that says nothing about itself: two
photo pages and a figure under a tab that names several things (`other`
with candidates, as designed), four scanned pages with no text source
(R38), three INHERITED labels the alternates accept, a table of contents
carrying the narrative's running header, a lab slip-sheet that names the
lab, and a core-photo log with no caption. That is the population the model
review pass exists for. WP1 closes here; the rules are frozen for WP1b.

---

## WP1b -- triage and label review -- checkpoint: CLAUDE DEV ENGINE ONLY

**Not the production model.** These six reports were scored on the Claude API
against `claude-opus-5` (review) and `claude-sonnet-5` (triage), which is the
engine the two passes were BUILT and debugged against. The app runs in Funhouse
against OpenAI models through Prompter, so this is a development checkpoint and
nothing here is a result for the system as it will run. The numbers that count
come from `report_ingest.cluster_scoring.score_on_cluster` on the cluster.

**It is also ROUND 1, on the prompts as first written.** It is what diagnosed
the review's one large error class -- fifteen of its nineteen wrong changes were
an exploration's own results, plotted rather than tabulated, called `figure`, or
one kind of sounding called another -- and the prompts were changed afterwards
to name that case (commit `c9569fb`). Four of the six re-ran on the new prompts
before the Anthropic credit ran out; that partial round was never completed and
is deliberately not recorded as a result.

The four hand labels the lead confirmed as disputed are dropped from both the
before and the after score here, which is why 673 pages are scored rather than
677.

```
prompts: r1-preplotfix
checkpoint: 6 report(s), 673 scored pages
4 page(s) dropped from both scores as confirmed disputed hand labels
                            before     after
strict accuracy              0.889     0.945
accepting alternates         0.892     0.945

key content (the gate is 0.98 on both rates after review)
label                n  P before  R before  F1 before   P after   R after  F1 after
narrative           78     0.973     0.936      0.954     0.939     0.987     0.963
plan                 2     0.000     0.000       --       0.400     1.000     0.571
profile              5     0.833     1.000      0.909     1.000     1.000     1.000
boring_log          51     0.943     0.980      0.962     0.981     1.000     0.990
test_pit_log        53     0.976     0.774      0.863     1.000     0.774     0.872
cpt_log              0      --        --         --       0.000      --        --  
dcp_log             36     1.000     0.583      0.737     1.000     0.583     0.737
lab_test           174     0.971     0.948      0.959     1.000     0.994     0.997
calculation        196     1.000     0.995      0.997     1.000     1.000     1.000
below the gate after review: narrative, plan, test_pit_log, dcp_log

every label
label                n  P before  R before  F1 before   P after   R after  F1 after
narrative           78     0.973     0.936      0.954     0.939     0.987     0.963
figure              12     0.000     0.000       --       0.290     0.750     0.419
plan                 2     0.000     0.000       --       0.400     1.000     0.571
profile              5     0.833     1.000      0.909     1.000     1.000     1.000
boring_log          51     0.943     0.980      0.962     0.981     1.000     0.990
test_pit_log        53     0.976     0.774      0.863     1.000     0.774     0.872
dcp_log             36     1.000     0.583      0.737     1.000     0.583     0.737
lab_test           174     0.971     0.948      0.959     1.000     0.994     0.997
calculation        196     1.000     0.995      0.997     1.000     1.000     1.000
photos              38     0.846     0.868      0.857     1.000     1.000     1.000
divider              9     1.000     0.889      0.941     1.000     1.000     1.000
cover                3      --       0.000       --       1.000     1.000     1.000
letter               2     1.000     1.000      1.000     0.667     1.000     0.800
toc                  5     1.000     0.400      0.571     1.000     0.600     0.750
other                9     0.086     0.333      0.136     0.857     0.667     0.750

what the review's changes did, against the hand labels
  fixed            38
  broke             0
  still_wrong      19
  disputed          4
  unscored         50

hand labels disputed by the review (the spreadsheet is never edited; a confirmed dispute is dropped from both scores)
  R15 p8    hand=narrative    review=cover        CONFIRMED: a one-line volume title sheet inside the narrative run
  R15 p19   hand=narrative    review=cover        CONFIRMED: the second volume's title sheet, the same one-line form
  R28 p217  hand=lab_test     review=letter       CONFIRMED: a laboratory's transmittal cover letter inside the laboratory appendix
  R37 p47   hand=calculation  review=other        CONFIRMED: a web-tool disclaimer page inside the calculation appendix; no inputs or results on it

report    pages  scored   before   after  chg  tools  calls       $      s
R36          94       5    0.800   1.000   23  20/60      8   0.889    142
R37          48       5    0.800   0.800   13  12/60      7   0.516     93
R05          15       5    1.000   1.000    1   3/60      5   0.161     41
R28         455     455    0.899   0.932   32  38/113     12   2.141    257
R15         202     202    0.861   0.960   24  30/60     12   1.492    240
R14          48       5    0.400   0.800   18  22/60     10   0.673    112

what triage said (enumerated fields only; the rationale and anomalies stay in raw/checks/triage/)
report  document_type                 workflow         bound       toc   scan
R36     geotechnical report           standard             0   partial   0.02
R37     geotechnical report           standard             0   partial   0.00
R05     recommendation letter         standard             0    no_toc   0.00
R28     geotechnical report           standard             0   matched   0.01
R15     geotechnical report           multi_document       1   partial   0.38
R14     geotechnical report           standard             0   partial   0.02

cost: 54 model calls, 108 input tokens (+1,387,633 cached), 63,672 output, $5.87, 887 s -- $0.979 and 148 s a report
```

## WP2a -- log_grid

### log_grid scorecard, 2026-09-17

Tolerances: sample and index values 0.15 m, layer tops 0.3 m; depths compared in metres whatever the log prints. Open set = R36, R37, R06, R07, R15, R28.

```
log         set      ruler   unit      samples       layers        index       fields  cells unmatched
------------------------------------------------------------------------------------------------------
R06_p51     open       yes    yes     100% 4/4     100% 5/5     100% 4/4      75% 6/8     62        54
R07_p30     open       yes    yes     100% 8/8     100% 9/9     100% 2/2   100% 13/13    103        97
R15_p46     open       yes    yes     100% 8/8     100% 4/4   100% 14/14   100% 13/13     75        58
R28_p57     open       yes    yes   100% 16/16     100% 5/5      75% 3/4   100% 13/13     78        66
R36_p38     open       yes    yes     100% 8/8     100% 6/6     100% 8/8     82% 9/11     34        19
R37_p26     open       yes    yes   100% 13/13     100% 5/5            -     90% 9/10     55        42
ALL         open    100% 6/6 100% 6/6   100% 57/57   100% 34/34    97% 31/32    93% 63/68    407       336

R02_p123    blind      yes     NO            -     100% 1/1            -      33% 2/6     67        67
R04_p59     blind      yes     NO            -     100% 5/5            -      33% 2/6     42        42
R13_p45     blind      yes    yes    92% 24/26     100% 5/5     54% 7/13    83% 10/12    105        92
R25_p19     blind       NO     NO            -            -            -            -      0         0
R31_p278    blind      yes    yes            -     100% 2/2            -   100% 11/11     27        27
R34_p49     blind      yes    yes      0% 0/10       0% 0/8            -      50% 2/4    115       115
ALL         blind    83% 5/6  50% 3/6    67% 24/36    62% 13/21     54% 7/13    69% 27/39    356       343

warnings seen:
  R02_p123: the depth unit is not stated on these pages and could not be read from their text; depths are in whatever the ruler prints
  R04_p59: the depth unit is not stated on these pages and could not be read from their text; depths are in whatever the ruler prints
  R13_p45: page 45: the text was read optically (azure_di); boxes and column edges are softer than on an embedded text layer
  R13_p45: page 45: no ruled column edges were found; the columns below come from the header labels alone and their x bands are approximate
  R15_p46: page 46: 1 text line(s) run diagonally across the page (a watermark or a stamp) and were left out of the grid
  R25_p19: page 19: no ruled column edges were found; the columns below come from the header labels alone and their x bands are approximate
  R25_p19: page 19: no columns could be laid out — nothing on this page is placed
  R25_p19: no description column was identified, so no layers were read
  R34_p49: no column header states the depth unit; m was read off depths written into the log's own text

```

### How to read that table, and what it does not say

**The scoring rules, stated so they can be argued with.** Depths are compared
in metres whatever the log prints. A sample is an INTERVAL, not a point, and
half the templates print a blow record against the middle of that interval
rather than its top, so the window is the sample interval widened by 0.15 m
(and, where the truth states only a top, the top plus a 0.46 m drive). A blow
record counts as found when its drives appear IN ORDER among the numbers
standing in that window in a blows-family column, because some forms print
the record as one cell ("5-9-12") and some print each drive on its own line.
An N value counts as found when a cell carries it OR when the drives that
define it (the second and third six inches) stand at that depth — half the
templates print only the drives, the grid does no arithmetic by design, and
counting that as a miss would score a decision, not a defect. A column counts
as the right one when it carries the value's canonical name or one of its
family, because a form that heads one column "SAMPLING DATA" and prints the
sample id, the drives and the recovery inside it is not wrong.

**The unmatched-cell column is a precision PROXY, not a precision.** The grid
emits every text line on the page; the hand truth states only samples,
layers, index values and fields. A description line, an elevation, a date and
a ruler tick are all unmatched by construction. What the number is good for
is watching it move: it fell from 375 to 336 on the open six as the rules
improved, because values that used to land in no useful column started
landing in one.

**One of the blind six is not blind.** R13's scanned page is the page the
optical path (no ruled columns, ruler found before the columns, the monotone
run, the wider layer-binding slack) was built against, two hours before the
truth folder and OPEN.txt existed. Its 92 % on blow records should be read as
a development figure. The honest blind set is the other five.

**What the blind failures are, from the warnings alone.** R25 draws no ruled
columns and its header labels do not lay out either, so nothing is placed and
the module says so three times over. R02 and R04 find their rulers but no
header and no text on those pages states a unit, so depths are in whatever
the ruler prints and the unit check fails by design rather than by mistake.
R34 finds a ruler and reads its unit off the log's own text, and then scores
zero on samples and layers; that has NOT been looked into, because looking at
the page would spend the log.

**Thresholds, and where they came from.** Every number the rules lean on is
gathered at the top of `planlens/document/loggrid.py` with the measurement
that set it beside it. The four that did real work:

- a vertical rule is a column edge at 0.30 of the page height — real edges on
  the corpus run 0.39 to 0.73, the longest decoy (a box around a groundwater
  table) 0.14;
- the header band is the full-width rule whose band NAMES the most columns,
  not the first one and not the one above the first number — three of the six
  open templates draw a page frame, a title block or a groundwater table
  above their header, and one prints numeric axis labels inside it;
- a ruler is scored 2.0 for a header that says depth, 1.5 for even steps and
  0.1 per tick, because the tie that matters is a layer-contact column
  (called depth, uneven, few ticks, labels set against the contacts) against
  the printed ruler, and getting it wrong costs about 0.2 m;
- a label takes a value up to 70 pt to its right — a boring number is set in
  large type at the far end of its box (60 pt) while the nearest unrelated
  text on a crowded form footer was 134 pt away and had been read as the
  value before the cap.


### log_grid scorecard, revised after the lead confirmed the truth schema

The six BLIND logs are scored exactly as the truth states them: a sheet whose
`form` note says it carries no depth scale is scored on REFUSING (no ruler,
and a warning saying so), and its layers and samples are recorded as not
scored by depth rather than counted as misses; a sheet with several borings
on it (a tabular list) has its layers and samples flattened out of the
`investigations` list; `rqd` and `pp_kpa` joined the index values; and a sheet
the truth says samples nothing is checked for cells the grid put in a
blow-count column anyway. Nothing about the reader was tuned on a blind log.

### log_grid scorecard, 2026-09-17

Tolerances: sample and index values 0.15 m, layer tops 0.3 m; depths compared in metres whatever the log prints. Open set = R36, R37, R06, R07, R15, R28.

```
log         set      ruler   unit      samples       layers        index       fields  cells unmatched
------------------------------------------------------------------------------------------------------
R06_p51     open       yes    yes     100% 4/4     100% 5/5     100% 4/4      75% 6/8     62        54
R07_p30     open       yes    yes     100% 8/8     100% 9/9     100% 6/6   100% 13/13    103        93
R15_p46     open       yes    yes     100% 8/8     100% 4/4   100% 14/14   100% 13/13     75        58
R28_p57     open       yes    yes   100% 16/16     100% 5/5      75% 3/4   100% 13/13     78        66
R36_p38     open       yes    yes     100% 8/8     100% 6/6     100% 8/8     82% 9/11     34        19
R37_p26     open       yes    yes   100% 13/13     100% 5/5            -     90% 9/10     55        42
ALL         open    100% 6/6 100% 6/6   100% 57/57   100% 34/34    97% 35/36    93% 63/68    407       332

R02_p123    blind      yes     NO            -     100% 1/1            -      33% 2/6     67        67
R03_p135    blind      yes    yes            -     100% 1/1       0% 0/1      40% 2/5     65        65
R04_p59     blind      yes     NO            -     100% 5/5            -      33% 2/6     42        42
R13_p45     blind      yes    yes    92% 24/26     100% 5/5     54% 7/13    83% 10/12    105        92
R21_p96     blind       NO     NO       0% 0/6       0% 0/3       0% 0/2      40% 2/5     81         0
R25_p19     blind      yes     NO            -            -            -            -      0         0
R30_p65     blind       NO     NO       0% 0/8       0% 0/1            -      40% 2/5     74         0
R31_p278    blind      yes    yes            -     100% 2/2            -   100% 11/11     27        27
R34_p49     blind      yes    yes      0% 0/10       0% 0/8       0% 0/1      50% 2/4    115       115
ALL         blind    78% 7/9  50% 4/8    48% 24/50    54% 14/26     41% 7/17    61% 33/54    576       408

notes:
  R02_p123: the truth states no samples; the grid put 0 numeric cell(s) in a blow-count column
  R25_p19: tabular sheet with no depth scale: 10 layers and 1 samples not scored by depth
  R31_p278: the truth states no samples; the grid put 0 numeric cell(s) in a blow-count column

warnings seen:
  R02_p123: the depth unit is not stated on these pages and could not be read from their text; depths are in whatever the ruler prints
  R03_p135: no column header states the depth unit; ft was read off depths written into the log's own text
  R04_p59: the depth unit is not stated on these pages and could not be read from their text; depths are in whatever the ruler prints
  R13_p45: page 45: the text was read optically (azure_di); boxes and column edges are softer than on an embedded text layer
  R13_p45: page 45: no ruled column edges were found; the columns below come from the header labels alone and their x bands are approximate
  R15_p46: page 46: 1 text line(s) run diagonally across the page (a watermark or a stamp) and were left out of the grid
  R21_p96: page 96: no depth ruler was found — no column holds three or more numbers that fall on a straight line, so nothing on this page carries a depth
  R25_p19: page 19: no ruled column edges were found; the columns below come from the header labels alone and their x bands are approximate
  R25_p19: page 19: no columns could be laid out — nothing on this page is placed
  R25_p19: page 19: no depth ruler was found — nothing on this page carries a depth
  R25_p19: no description column was identified, so no layers were read
  R30_p65: page 65: no depth ruler was found — no column holds three or more numbers that fall on a straight line, so nothing on this page carries a depth
  R34_p49: no column header states the depth unit; m was read off depths written into the log's own text

```

### log_grid scorecard, round 2 (signed numbers + gINT header vocabulary)

Two generic defects the lead found by reading three low blind logs. A leading
dash was being STRIPPED, so a column of elevations below datum read as a
rising series and could be chosen as the depth scale; and what a header SAYS
was outweighed by tick count, so an unnamed column with more ticks beat a
column the form calls depth. Both are fixed on principle, with the gINT
default template's headers added to the vocabulary and the header matcher
changed so a header names itself before it qualifies itself.

The open six did not move by a single cell (407 placed, 332 unmatched, the
same as round 1). On the blind nine nothing moved either EXCEPT the failure
mode of one sheet, which is the whole point: it used to claim a ruler fitted
to the elevation column and read the page at the wrong datum, and it now
finds no ruler and says so.

So the ruler row is scored three ways from this round on -- yes, wrong, none
-- because a wrong ruler and a missing one are not the same failure. A
missing ruler withholds every depth; a wrong one hands back a page of
confident numbers. After this round there is no wrong ruler on any of the
fifteen logs, and the blind ruler figure FELL from 7/9 to 6/9 because the
sheet that used to be counted as "found" was found wrong.

The vocabulary additions changed no measured number. They are pinned by 18
new tests and will only show on the corpus once the three refused sheets get
a ruler.

### log_grid scorecard, 2026-09-17

Tolerances: sample and index values 0.15 m, layer tops 0.3 m; depths compared in metres whatever the log prints. Open set = R36, R37, R06, R07, R15, R28.

```
log         set      ruler   unit      samples       layers        index       fields  cells unmatched
------------------------------------------------------------------------------------------------------
R06_p51     open       yes    yes     100% 4/4     100% 5/5     100% 4/4      75% 6/8     62        54
R07_p30     open       yes    yes     100% 8/8     100% 9/9     100% 6/6   100% 13/13    103        93
R15_p46     open       yes    yes     100% 8/8     100% 4/4   100% 14/14   100% 13/13     75        58
R28_p57     open       yes    yes   100% 16/16     100% 5/5      75% 3/4   100% 13/13     78        66
R36_p38     open       yes    yes     100% 8/8     100% 6/6     100% 8/8     82% 9/11     34        19
R37_p26     open       yes    yes   100% 13/13     100% 5/5            -     90% 9/10     55        42
ALL         open    100% 6/6 100% 6/6   100% 57/57   100% 34/34    97% 35/36    93% 63/68    407       332

R02_p123    blind      yes     NO            -     100% 1/1            -      33% 2/6     67        67
R03_p135    blind      yes    yes            -     100% 1/1       0% 0/1      40% 2/5     65        65
R04_p59     blind      yes     NO            -     100% 5/5            -      33% 2/6     42        42
R13_p45     blind      yes    yes    92% 24/26     100% 5/5     54% 7/13    83% 10/12    105        92
R21_p96     blind     none     NO       0% 0/6       0% 0/3       0% 0/2      40% 2/5     81         0
R25_p19     blind  refused     NO            -            -            -            -      0         0
R30_p65     blind     none     NO       0% 0/8       0% 0/1            -      40% 2/5     74         0
R31_p278    blind      yes    yes            -     100% 2/2            -   100% 11/11     27        27
R34_p49     blind     none     NO      0% 0/10       0% 0/8       0% 0/1      50% 2/4    115         0
ALL         blind    67% 6/9  38% 3/8    48% 24/50    54% 14/26     41% 7/17    61% 33/54    576       293

notes:
  R02_p123: the truth states no samples; the grid put 0 numeric cell(s) in a blow-count column
  R25_p19: tabular sheet with no depth scale: 10 layers and 1 samples not scored by depth
  R31_p278: the truth states no samples; the grid put 0 numeric cell(s) in a blow-count column

warnings seen:
  R02_p123: the depth unit is not stated on these pages and could not be read from their text; depths are in whatever the ruler prints
  R03_p135: no column header states the depth unit; ft was read off depths written into the log's own text
  R04_p59: the depth unit is not stated on these pages and could not be read from their text; depths are in whatever the ruler prints
  R13_p45: page 45: the text was read optically (azure_di); boxes and column edges are softer than on an embedded text layer
  R13_p45: page 45: no ruled column edges were found; the columns below come from the header labels alone and their x bands are approximate
  R15_p46: page 46: 1 text line(s) run diagonally across the page (a watermark or a stamp) and were left out of the grid
  R21_p96: page 96: no depth ruler was found — no column holds three or more numbers that fall on a straight line, so nothing on this page carries a depth
  R25_p19: page 19: no ruled column edges were found; the columns below come from the header labels alone and their x bands are approximate
  R25_p19: page 19: no columns could be laid out — nothing on this page is placed
  R25_p19: page 19: no depth ruler was found — nothing on this page carries a depth
  R25_p19: no description column was identified, so no layers were read
  R30_p65: page 65: no depth ruler was found — no column holds three or more numbers that fall on a straight line, so nothing on this page carries a depth
  R34_p49: page 49: no depth ruler was found — no column holds three or more numbers that fall on a straight line, so nothing on this page carries a depth

```

---

## WP1 -- page roles -- round 6

Measured 2026-09-17 with the app venv and the editable planlens checkout. 4147 hand-labelled pages, opened `di="auto"`, scored against `labels.labels_for`, split into a development set (R09, R12, R15, R16, R20, R23, R28, R29, R30) and a held-out set (R11, R13, R18, R21, R24). **The gate is on the held-out set**: precision AND recall >= 0.90 on `boring_log`, `test_pit_log`, `lab_test`, `narrative`, `calculation`.

**Development Set**, round by round

| round | boring_log P/R | test_pit_log P/R | lab_test P/R | narrative P/R | calculation P/R | accuracy | what changed |
|---|---|---|---|---|---|---|---|
| 1 | 0.970 / 0.941 | 0.959 / 0.920 | 0.936 / 0.953 | 0.942 / 0.933 | 0.997 / 0.974 | 0.900 | first measurement of the rules on the live documents (all 14 reports; measured before the split) |
| 2 | 0.970 / 0.941 | 0.959 / 0.920 | 0.936 / 0.984 | 0.942 / 0.933 | 0.997 / 0.974 | 0.912 | dynamic-probing abbreviations (DPL/DPM/DPH/DPSH/DPT/LCPT) as DCP titles; chemical and grading laboratory titles; a tab of logs no longer claims a page of prose (all 14 reports; measured before the split) |
| 3 | 0.963 / 0.919 | 0.949 / 0.925 | 0.967 / 0.984 | 0.965 / 0.959 | 0.999 / 0.987 | 0.923 | the dev / held-out split, and document_outline + page_ledger added; NO rule changed this round |
| 4 | 0.963 / 0.919 | 0.949 / 0.925 | 0.967 / 0.984 | 0.965 / 0.959 | 0.999 / 0.987 | 0.923 | a role a page did NOT name itself now carries INHERITED_CONFIDENCE and says so; no prediction changed, so every rate is identical to round 3 |
| 5 | 0.948 / 0.919 | 0.947 / 0.885 | 0.958 / 0.989 | 0.976 / 0.948 | 0.999 / 0.995 | 0.920 | the four principles: a page that names itself beats its tab, prose needs narrative evidence, a tab that names several things chooses on the page or says other, and a nested report records its inner role |
| 6 | 0.948 / 0.919 | 0.947 / 0.885 | 0.958 / 0.989 | 0.976 / 0.948 | 0.999 / 0.995 | 0.920 |  |

**Held-Out Set**, round by round

| round | boring_log P/R | test_pit_log P/R | lab_test P/R | narrative P/R | calculation P/R | accuracy | what changed |
|---|---|---|---|---|---|---|---|
| 1 | - | - | - | - | - | - | first measurement of the rules on the live documents (all 14 reports; measured before the split) |
| 2 | - | - | - | - | - | - | dynamic-probing abbreviations (DPL/DPM/DPH/DPSH/DPT/LCPT) as DCP titles; chemical and grading laboratory titles; a tab of logs no longer claims a page of prose (all 14 reports; measured before the split) |
| 3 | 0.987 / 1.000 | 1.000 / 0.902 | 0.813 / 0.983 | 0.851 / 0.832 | 0.992 / 0.938 | 0.887 | the dev / held-out split, and document_outline + page_ledger added; NO rule changed this round |
| 4 | 0.987 / 1.000 | 1.000 / 0.902 | 0.813 / 0.983 | 0.851 / 0.832 | 0.992 / 0.938 | 0.887 | a role a page did NOT name itself now carries INHERITED_CONFIDENCE and says so; no prediction changed, so every rate is identical to round 3 |
| 5 | 0.987 / 1.000 | 1.000 / 0.902 | 0.813 / 0.983 | 0.959 / 0.786 | 0.971 / 0.903 | 0.880 | the four principles: a page that names itself beats its tab, prose needs narrative evidence, a tab that names several things chooses on the page or says other, and a nested report records its inner role |
| 6 | 0.987 / 1.000 | 1.000 / 0.902 | 0.813 / 0.983 | 0.959 / 0.786 | 0.971 / 0.903 | 0.880 |  |

### Where held-out lags development

- **`lab_test` lags by more than 0.05**: precision 0.958 to 0.813. The confusions that cost it on the held-out set: `other -> lab_test` (14 pages), `calculation -> lab_test` (14 pages), `narrative -> lab_test` (7 pages).
- **`narrative` lags by more than 0.05**: recall 0.948 to 0.787. The confusions that cost it on the held-out set: `narrative -> lab_test` (7 pages), `narrative -> other` (7 pages), `narrative -> plan` (2 pages).
- **`calculation` lags by more than 0.05**: recall 0.995 to 0.903. The confusions that cost it on the held-out set: `calculation -> lab_test` (14 pages), `calculation -> profile` (8 pages), `figure -> calculation` (5 pages).

### This round in full

#### Held-Out Set

5 reports (R11, R13, R18, R21, R24), 1300 pages, 99 s.

| role | precision | recall | F1 | hand-labelled |
|---|---|---|---|---|
| `narrative` **(gated)** | 0.959 | 0.787 | 0.864 | 89 |
| `figure` | 0.273 | 0.261 | 0.267 | 23 |
| `plan` | 0.312 | 0.833 | 0.455 | 6 |
| `profile` | 0.438 | 0.467 | 0.452 | 15 |
| `boring_log` **(gated)** | 0.987 | 1.000 | 0.993 | 75 |
| `test_pit_log` **(gated)** | 1.000 | 0.902 | 0.948 | 51 |
| `cpt_log` | 0.368 | 1.000 | 0.538 | 7 |
| `dcp_log` | 0.000 | 0.000 | 0.000 | 0 |
| `lab_test` **(gated)** | 0.813 | 0.983 | 0.890 | 177 |
| `field_test` | 0.250 | 1.000 | 0.400 | 2 |
| `calculation` **(gated)** | 0.971 | 0.903 | 0.936 | 259 |
| `appended_report` | 1.000 | 1.000 | 1.000 | 472 |
| `photos` | 0.000 | 0.000 | 0.000 | 0 |
| `divider` | 0.333 | 0.824 | 0.475 | 17 |
| `cover` | 1.000 | 0.250 | 0.400 | 8 |
| `letter` | 1.000 | 0.500 | 0.667 | 4 |
| `toc` | 0.875 | 1.000 | 0.933 | 7 |
| `other` | 0.500 | 0.239 | 0.323 | 88 |

Page accuracy **0.880** over 1300 pages. The gate does NOT pass.

Confusion matrix (rows = hand label, columns = predicted):

| hand \ pred | narr | figu | plan | prof | bori | test | cpt_ | dcp_ | lab_ | fiel | calc | appe | phot | divi | cove | lett | toc | othe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `narrative` | 70 | . | 2 | . | . | . | 1 | . | 7 | . | . | . | . | 1 | . | . | 1 | 7 |
| `figure` | . | 6 | 3 | . | . | . | . | . | 3 | . | 5 | . | . | . | . | . | . | 6 |
| `plan` | . | . | 5 | 1 | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| `profile` | 2 | 2 | . | 7 | . | . | . | . | . | . | 2 | . | . | . | . | . | . | 2 |
| `boring_log` | . | . | . | . | 75 | . | . | . | . | . | . | . | . | . | . | . | . | . |
| `test_pit_log` | . | . | 5 | . | . | 46 | . | . | . | . | . | . | . | . | . | . | . | . |
| `cpt_log` | . | . | . | . | . | . | 7 | . | . | . | . | . | . | . | . | . | . | . |
| `dcp_log` | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| `lab_test` | . | 2 | . | . | . | . | . | . | 174 | . | . | . | . | 1 | . | . | . | . |
| `field_test` | . | . | . | . | . | . | . | . | . | 2 | . | . | . | . | . | . | . | . |
| `calculation` | . | . | 1 | 8 | . | . | 1 | . | 14 | . | 234 | . | . | 1 | . | . | . | . |
| `appended_report` | . | . | . | . | . | . | . | . | . | . | . | 472 | . | . | . | . | . | . |
| `photos` | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . |
| `divider` | . | . | . | . | . | . | . | . | 2 | . | . | . | . | 14 | . | . | . | 1 |
| `cover` | . | 2 | . | . | . | . | . | . | . | . | . | . | . | . | 2 | . | . | 4 |
| `letter` | 1 | . | . | . | . | . | . | . | . | . | . | . | . | . | . | 2 | . | 1 |
| `toc` | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | 7 | . |
| `other` | . | 10 | . | . | 1 | . | 10 | 1 | 14 | 6 | . | . | . | 25 | . | . | . | 21 |

Column keys: `narr` = narrative, `figu` = figure, `plan` = plan, `prof` = profile, `bori` = boring_log, `test` = test_pit_log, `cpt_` = cpt_log, `dcp_` = dcp_log, `lab_` = lab_test, `fiel` = field_test, `calc` = calculation, `appe` = appended_report, `phot` = photos, `divi` = divider, `cove` = cover, `lett` = letter, `toc` = toc, `othe` = other.

Per-report accuracy:

| ID | pages | accuracy |
|---|---|---|
| R11 | 156 | 0.904 |
| R13 | 197 | 0.843 |
| R18 | 64 | 0.969 |
| R21 | 729 | 0.962 |
| R24 | 154 | 0.481 |

82 pages where a gated role is involved and the rules and the hand label disagree. They are NOT listed page by page: closing a gap by reading the held-out pages is how a held-out set stops being one. The confusions, by size:

| hand -> predicted | pages |
|---|---|
| `other -> lab_test` | 14 |
| `calculation -> lab_test` | 14 |
| `calculation -> profile` | 8 |
| `narrative -> lab_test` | 7 |
| `narrative -> other` | 7 |
| `figure -> calculation` | 5 |
| `test_pit_log -> plan` | 5 |
| `figure -> lab_test` | 3 |
| `lab_test -> figure` | 2 |
| `narrative -> plan` | 2 |
| `profile -> calculation` | 2 |
| `profile -> narrative` | 2 |
| `divider -> lab_test` | 2 |
| `lab_test -> divider` | 1 |
| `letter -> narrative` | 1 |
| `calculation -> plan` | 1 |
| `calculation -> cpt_log` | 1 |
| `calculation -> divider` | 1 |
| `narrative -> toc` | 1 |
| `narrative -> divider` | 1 |
| `narrative -> cpt_log` | 1 |
| `other -> boring_log` | 1 |

#### Development Set

9 reports (R09, R12, R15, R16, R20, R23, R28, R29, R30), 2847 pages, 346 s.

| role | precision | recall | F1 | hand-labelled |
|---|---|---|---|---|
| `narrative` **(gated)** | 0.976 | 0.948 | 0.962 | 344 |
| `figure` | 0.264 | 0.368 | 0.308 | 38 |
| `plan` | 0.294 | 0.417 | 0.345 | 12 |
| `profile` | 0.750 | 0.692 | 0.720 | 13 |
| `boring_log` **(gated)** | 0.948 | 0.919 | 0.933 | 198 |
| `test_pit_log` **(gated)** | 0.947 | 0.885 | 0.915 | 200 |
| `cpt_log` | 1.000 | 0.875 | 0.933 | 16 |
| `dcp_log` | 0.967 | 0.537 | 0.690 | 54 |
| `lab_test` **(gated)** | 0.958 | 0.989 | 0.973 | 815 |
| `field_test` | 0.882 | 0.882 | 0.882 | 17 |
| `calculation` **(gated)** | 0.999 | 0.995 | 0.997 | 750 |
| `appended_report` | 0.815 | 1.000 | 0.898 | 22 |
| `photos` | 0.904 | 0.856 | 0.879 | 132 |
| `divider` | 0.864 | 0.864 | 0.864 | 66 |
| `cover` | 1.000 | 0.409 | 0.581 | 22 |
| `letter` | 1.000 | 1.000 | 1.000 | 2 |
| `toc` | 1.000 | 0.611 | 0.759 | 18 |
| `other` | 0.509 | 0.648 | 0.570 | 128 |

Page accuracy **0.920** over 2847 pages.

Confusion matrix (rows = hand label, columns = predicted):

| hand \ pred | narr | figu | plan | prof | bori | test | cpt_ | dcp_ | lab_ | fiel | calc | appe | phot | divi | cove | lett | toc | othe |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `narrative` | 326 | 2 | 4 | 1 | . | . | . | . | 5 | . | . | 1 | 1 | . | . | . | . | 4 |
| `figure` | 3 | 14 | 6 | . | 1 | . | . | . | . | . | . | . | . | . | . | . | . | 14 |
| `plan` | . | . | 5 | 1 | . | . | . | . | . | 2 | 1 | . | 1 | . | . | . | . | 2 |
| `profile` | . | . | . | 9 | 1 | . | . | . | . | . | . | . | . | . | . | . | . | 3 |
| `boring_log` | . | 9 | . | . | 182 | 3 | . | . | . | . | . | . | 1 | . | . | . | . | 3 |
| `test_pit_log` | . | . | . | . | 1 | 177 | . | . | 2 | . | . | . | 3 | 1 | . | . | . | 16 |
| `cpt_log` | . | . | . | . | . | . | 14 | . | . | . | . | . | . | . | . | . | . | 2 |
| `dcp_log` | . | 15 | . | . | . | . | . | 29 | . | . | . | . | . | . | . | . | . | 10 |
| `lab_test` | . | . | . | . | . | . | . | . | 806 | . | . | . | . | . | . | . | . | 9 |
| `field_test` | . | . | . | . | . | . | . | . | 2 | 15 | . | . | . | . | . | . | . | . |
| `calculation` | . | . | . | 1 | . | . | . | . | 2 | . | 746 | . | . | 1 | . | . | . | . |
| `appended_report` | . | . | . | . | . | . | . | . | . | . | . | 22 | . | . | . | . | . | . |
| `photos` | . | . | . | . | 1 | 2 | . | . | 5 | . | . | . | 113 | 4 | . | . | . | 7 |
| `divider` | . | 2 | . | . | . | . | . | . | 1 | . | . | 4 | . | 57 | . | . | . | 2 |
| `cover` | 1 | 3 | . | . | . | . | . | . | 1 | . | . | . | . | . | 9 | . | . | 8 |
| `letter` | . | . | . | . | . | . | . | . | . | . | . | . | . | . | . | 2 | . | . |
| `toc` | 4 | . | 1 | . | . | . | . | . | 2 | . | . | . | . | . | . | . | 11 | . |
| `other` | . | 8 | 1 | . | 6 | 5 | . | 1 | 15 | . | . | . | 6 | 3 | . | . | . | 83 |

Column keys: `narr` = narrative, `figu` = figure, `plan` = plan, `prof` = profile, `bori` = boring_log, `test` = test_pit_log, `cpt_` = cpt_log, `dcp_` = dcp_log, `lab_` = lab_test, `fiel` = field_test, `calc` = calculation, `appe` = appended_report, `phot` = photos, `divi` = divider, `cove` = cover, `lett` = letter, `toc` = toc, `othe` = other.

Per-report accuracy:

| ID | pages | accuracy |
|---|---|---|
| R09 | 151 | 0.901 |
| R12 | 159 | 0.899 |
| R15 | 202 | 0.871 |
| R16 | 426 | 0.948 |
| R20 | 370 | 0.968 |
| R23 | 397 | 0.937 |
| R28 | 455 | 0.901 |
| R29 | 131 | 0.733 |
| R30 | 556 | 0.944 |

121 pages where a gated role is involved and the rules and the hand label disagree. `rule` and `why` are the evidence planlens itself recorded; the page HEADING is omitted on purpose -- the largest type on a log or a laboratory sheet is a firm's title block, and this file is tracked in a public repository. The same list WITH headings is written to `raw/checks/wp1_misses.txt`, which is gitignored.

**R09** (10)

```
R09 p1    hand=narrative       pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R09 p2    hand=narrative       pred=other           kind=figure        rule='the page names itself' why="legend or notes sheet 'conversion factors'"
R09 p4    hand=narrative       pred=lab_test        kind=figure        rule='the page names itself' why='11 laboratory test terms on the page'
R09 p6    hand=narrative       pred=profile         kind=figure        rule='the page names itself' why="profile title 'probable soil'"
R09 p8    hand=narrative       pred=figure          kind=figure        rule='figure page with no title of its own' why=''
R09 p13   hand=figure          pred=narrative       kind=figure        rule="prose carrying the narrative's running footer" why=''
R09 p14   hand=figure          pred=narrative       kind=figure        rule="prose carrying the narrative's running footer" why=''
R09 p15   hand=figure          pred=narrative       kind=figure        rule="prose carrying the narrative's running footer" why=''
R09 p17   hand=other           pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test name 'liquid limit' on a page of working"
R09 p150  hand=other           pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test name 'laboratory testing' on a page of worki"
```

**R12** (13)

```
R12 p1    hand=cover           pred=narrative       kind=figure        rule="prose carrying the narrative's running header" why=''
R12 p3    hand=toc             pred=narrative       kind=mixed         rule="prose carrying the narrative's running header" why=''
R12 p22   hand=narrative       pred=lab_test        kind=figure        rule='the page names itself' why='7 laboratory test terms on the page'
R12 p54   hand=figure          pred=boring_log      kind=figure        rule='the page names itself' why="log title 'borehole no', 2 log form fields"
R12 p61   hand=test_pit_log    pred=lab_test        kind=figure        rule='the page names itself' why="laboratory test title 'permeability' beside a reference to t"
R12 p62   hand=test_pit_log    pred=lab_test        kind=figure        rule='the page names itself' why="laboratory test title 'permeability' beside a reference to t"
R12 p64   hand=other           pred=lab_test        kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['lab_test']"
R12 p78   hand=other           pred=lab_test        kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['lab_test']"
R12 p116  hand=field_test      pred=lab_test        kind=figure        rule='INHERITED from a tab that names several things; chosen on th' why='a results form or a plotted result'
R12 p117  hand=field_test      pred=lab_test        kind=figure        rule='INHERITED from a tab that names several things; chosen on th' why='a results form or a plotted result'
R12 p122  hand=other           pred=lab_test        kind=figure        rule='the page names itself' why='2 laboratory test terms on the page'
R12 p132  hand=other           pred=lab_test        kind=figure        rule='the page names itself' why="laboratory test title 'water extract'"
R12 p139  hand=other           pred=lab_test        kind=figure        rule='the page names itself' why='2 laboratory test terms on the page'
```

**R15** (18)

```
R15 p3    hand=toc             pred=narrative       kind=text          rule="prose carrying the narrative's running header" why=''
R15 p5    hand=narrative       pred=plan            kind=mixed         rule='the page names itself' why="plan title 'site plans'"
R15 p11   hand=narrative       pred=photos          kind=text          rule='the page names itself' why='photograph caption (2)'
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

**R16** (4)

```
R16 p7    hand=narrative       pred=other           kind=text          rule='text page with no title of its own' why=''
R16 p27   hand=profile         pred=boring_log      kind=form          rule='the page names itself' why='log form shape, 6 log form fields'
R16 p79   hand=other           pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'laboratory test'"
R16 p128  hand=cover           pred=lab_test        kind=mixed         rule='INHERITED from its appendix tab; the page says nothing about' why="['lab_test']"
```

**R20** (4)

```
R20 p8    hand=narrative       pred=plan            kind=text          rule='the page names itself' why="plan title 'geologic map'"
R20 p29   hand=narrative       pred=appended_report kind=text          rule='inside a report bound into this one' why=''
R20 p115  hand=test_pit_log    pred=divider         kind=mixed         rule='tab or cover page naming what follows' why=''
R20 p365  hand=calculation     pred=profile         kind=form          rule='the page names itself' why="profile title 'cross section'"
```

**R23** (16)

```
R23 p5    hand=narrative       pred=other           kind=text          rule='text page with no title of its own' why=''
R23 p170  hand=boring_log      pred=test_pit_log    kind=figure        rule='its tab named several things and the page names none; the ru' why=''
R23 p171  hand=boring_log      pred=test_pit_log    kind=mixed         rule='its tab named several things and the page names none; the ru' why=''
R23 p184  hand=boring_log      pred=test_pit_log    kind=mixed         rule='its tab named several things and the page names none; the ru' why=''
R23 p186  hand=test_pit_log    pred=other           kind=figure        rule='its appendix tab names several things and the page names non' why=''
R23 p187  hand=test_pit_log    pred=other           kind=text          rule='its appendix tab names several things and the page names non' why=''
R23 p188  hand=test_pit_log    pred=other           kind=figure        rule='its appendix tab names several things and the page names non' why=''
R23 p189  hand=test_pit_log    pred=other           kind=text          rule='its appendix tab names several things and the page names non' why=''
R23 p190  hand=test_pit_log    pred=other           kind=figure        rule='its appendix tab names several things and the page names non' why=''
R23 p191  hand=boring_log      pred=other           kind=text          rule='its appendix tab names several things and the page names non' why=''
R23 p192  hand=boring_log      pred=other           kind=figure        rule='its appendix tab names several things and the page names non' why=''
R23 p193  hand=test_pit_log    pred=other           kind=text          rule='its appendix tab names several things and the page names non' why=''
R23 p194  hand=boring_log      pred=other           kind=figure        rule='its appendix tab names several things and the page names non' why=''
R23 p195  hand=test_pit_log    pred=other           kind=text          rule='its appendix tab names several things and the page names non' why=''
R23 p196  hand=test_pit_log    pred=other           kind=figure        rule='its appendix tab names several things and the page names non' why=''
R23 p255  hand=other           pred=lab_test        kind=text          rule='INHERITED from its appendix tab; the page says nothing about' why="['lab_test']"
```

**R28** (16)

```
R28 p4    hand=toc             pred=narrative       kind=text          rule="prose carrying the narrative's running header" why=''
R28 p5    hand=toc             pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'laboratory test'"
R28 p53   hand=other           pred=boring_log      kind=text          rule='the page names itself' why="log title 'test borings', 15 log form fields"
R28 p56   hand=other           pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'sieve' beside a reference to the expl"
R28 p233  hand=test_pit_log    pred=photos          kind=figure        rule='the page names itself' why='photograph caption (1)'
R28 p236  hand=test_pit_log    pred=photos          kind=figure        rule='the page names itself' why='photograph caption (1)'
R28 p238  hand=test_pit_log    pred=photos          kind=figure        rule='the page names itself' why='photograph caption (1)'
R28 p253  hand=test_pit_log    pred=other           kind=figure        rule='its appendix tab names several things and the page names non' why=''
R28 p254  hand=test_pit_log    pred=other           kind=figure        rule='its appendix tab names several things and the page names non' why=''
R28 p255  hand=test_pit_log    pred=other           kind=figure        rule='its appendix tab names several things and the page names non' why=''
R28 p256  hand=test_pit_log    pred=other           kind=figure        rule='its appendix tab names several things and the page names non' why=''
R28 p257  hand=test_pit_log    pred=other           kind=figure        rule='its appendix tab names several things and the page names non' why=''
R28 p258  hand=test_pit_log    pred=other           kind=figure        rule='its appendix tab names several things and the page names non' why=''
R28 p259  hand=test_pit_log    pred=other           kind=figure        rule='its appendix tab names several things and the page names non' why=''
R28 p260  hand=test_pit_log    pred=other           kind=figure        rule='its appendix tab names several things and the page names non' why=''
R28 p414  hand=calculation     pred=lab_test        kind=text          rule='the page names itself' why="laboratory test title 'consolidation'"
```

**R29** (16)

```
R29 p3    hand=toc             pred=narrative       kind=text          rule="prose carrying the narrative's running header" why=''
R29 p5    hand=toc             pred=lab_test        kind=text          rule='the page names itself' why="laboratory test name 'california bearing ratio' on a page of"
R29 p8    hand=narrative       pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'moisture content'"
R29 p11   hand=narrative       pred=plan            kind=text          rule='the page names itself' why="plan title 'geologic map'"
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

**R30** (24)

```
R30 p5    hand=narrative       pred=plan            kind=text          rule='the page names itself' why="plan title 'site and vicinity'"
R30 p49   hand=plan            pred=calculation     kind=form          rule='the page names itself' why="calculation heading 'global stability'"
R30 p58   hand=other           pred=boring_log      kind=mixed         rule='INHERITED from its appendix tab; the page says nothing about' why="['boring_log']"
R30 p81   hand=other           pred=boring_log      kind=mixed         rule='INHERITED from its appendix tab; the page says nothing about' why="['boring_log']"
R30 p82   hand=photos          pred=boring_log      kind=form          rule='INHERITED from its appendix tab; the page says nothing about' why="['boring_log']"
R30 p83   hand=other           pred=boring_log      kind=mixed         rule='INHERITED from its appendix tab; the page says nothing about' why="['boring_log']"
R30 p118  hand=other           pred=boring_log      kind=mixed         rule='INHERITED from its appendix tab; the page says nothing about' why="['boring_log']"
R30 p131  hand=other           pred=test_pit_log    kind=mixed         rule='INHERITED from its appendix tab; the page says nothing about' why="['test_pit_log']"
R30 p139  hand=other           pred=test_pit_log    kind=mixed         rule='INHERITED from its appendix tab; the page says nothing about' why="['test_pit_log']"
R30 p150  hand=other           pred=test_pit_log    kind=mixed         rule='INHERITED from its appendix tab; the page says nothing about' why="['test_pit_log']"
R30 p196  hand=other           pred=test_pit_log    kind=mixed         rule='INHERITED from its appendix tab; the page says nothing about' why="['test_pit_log']"
R30 p198  hand=photos          pred=test_pit_log    kind=mixed         rule='INHERITED from its appendix tab; the page says nothing about' why="['test_pit_log']"
R30 p199  hand=photos          pred=test_pit_log    kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['test_pit_log']"
R30 p228  hand=other           pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'laboratory test'"
R30 p296  hand=other           pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'laboratory test'"
R30 p297  hand=other           pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'laboratory testing'"
R30 p385  hand=other           pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'laboratory testing'"
R30 p418  hand=photos          pred=lab_test        kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['lab_test']"
R30 p419  hand=photos          pred=lab_test        kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['lab_test']"
R30 p430  hand=other           pred=lab_test        kind=mixed         rule='the page names itself' why="laboratory test title 'laboratory testing'"
R30 p471  hand=photos          pred=lab_test        kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['lab_test']"
R30 p472  hand=photos          pred=lab_test        kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['lab_test']"
R30 p473  hand=photos          pred=lab_test        kind=figure        rule='INHERITED from its appendix tab; the page says nothing about' why="['lab_test']"
R30 p523  hand=calculation     pred=divider         kind=mixed         rule='tab or cover page naming what follows' why=''
```

---

### log_grid scorecard, round 3 (overprinted lines + title-block headers)

Note on the WP1 block above it: `measure_wp1_labels.py` appends by default, so
re-running it to check the text-extraction change wrote a "round 6" block.
That block IS the check, and it is the answer: held-out page accuracy 0.880
over 1,300 pages, the same as the recorded baseline. Dropping overprinted
lines moved no page role.

Two defects the lead found by reading log pages I am not allowed to open.

**A line drawn twice.** Some forms draw a string a second time at the same
place to fake a bold weight. Fixed in planlens' TEXT EXTRACTION rather than in
log_grid, because it is not a log fact: counting a line twice doubles a page's
words, returns two search hits for one occurrence, and hands a reader "9 9 10
10" where the page reads 9, 10. Measured over the whole 7,829-page corpus:
4,751 overprinted lines on 523 pages of at least twelve reports (R28 922, R02
886, R31 684, R21 540, R23 504, R34 190, R29 180, R30 176, R20 159, R03 119,
R17 88, R16 73), up to 2.5 per cent of a report's lines. For a depth ruler it
was fatal rather than untidy: "5, 5, 10, 10, 15, 15" has no strictly rising
run of three in it, so three sheets refused their scale outright. The fitter
also collapses ticks sharing a value and a y, so a duplicate reaching it by
another route cannot do the same again.

**A title block naming a column.** A header candidate must now lie inside the
column it would name; a line crossing column boundaries is the form talking
about the sheet. It stays available to `fields`.

**Result.** The open six did not move by a single cell for the third round
running: 407 placed, 332 unmatched. On the blind nine the ruler went 6/9 to
9/9 -- the only sheet now without one is the tabular list that HAS no scale --
the unit 3/8 to 5/8, and layer tops 14/26 to 22/26. Samples, index values and
fields did not move.

**What is still short, and why (numbers only; no page was opened).** Three
sheets find their rulers and read their layers while every one of their 15 to
19 columns comes back `other`, which is why their samples and index values
score zero. The cause is geometric, not lexical: these forms draw NO rule
under their header row. The only rules crossing the whole form sit at y 86.2,
107.8 and 134.9 of a body running to 755.8, so every candidate band lies in
the top 50 pt and none of them can hold the column labels. Scored bands, per
sheet:

```
R34_p49  ruled y 86.2..755.8, 19 columns, full-width rules at 86.2/107.8/134.9
         candidate body_top=107.8  band=( 86.2,107.8)  4 lines  names 0 columns
         candidate body_top=134.9  band=(107.8,134.9)  4 lines  names 0 columns
R21_p96  ruled y 86.3..755.9, 15 columns, full-width rules at 86.3/107.9/134.9
         candidate body_top=107.9  band=( 86.3,107.9)  4 lines  names 0 columns
         candidate body_top=134.9  band=(107.9,134.9)  4 lines  names 0 columns
R15_p46  (an OPEN sheet, for contrast: the divider exists)
         candidate body_top=241.0  band=( 89.8,241.0) 58 lines  names 2 columns
         candidate body_top=280.6  band=(241.0,280.6) 13 lines  names 8 columns
```

Their rulers are right, which is the check that matters: R34 reads 6.00 to
13.50 m over 9 ticks with residual 0.001, R21 8.00 to 18.00 m over 11 ticks,
R30 5.00 to 11.00 over 7 ticks.

**Rejected, with the measurement.** Deriving the header band from the ruler's
own first tick when no drawn band names anything. It does not name those
columns either, and on the one blind sheet where it did fire it reached up
into the sheet's own fields: layer tops 5/5 to 3/5, fields 2/6 to 0/6, cells
42 to 71, to buy one depth unit. The drawn line stays the only evidence for
where a header band is.

### log_grid scorecard, 2026-09-17

Tolerances: sample and index values 0.15 m, layer tops 0.3 m; depths compared in metres whatever the log prints. Open set = R36, R37, R06, R07, R15, R28.

```
log         set      ruler   unit      samples       layers        index       fields  cells unmatched
------------------------------------------------------------------------------------------------------
R06_p51     open       yes    yes     100% 4/4     100% 5/5     100% 4/4      75% 6/8     62        54
R07_p30     open       yes    yes     100% 8/8     100% 9/9     100% 6/6   100% 13/13    103        93
R15_p46     open       yes    yes     100% 8/8     100% 4/4   100% 14/14   100% 13/13     75        58
R28_p57     open       yes    yes   100% 16/16     100% 5/5      75% 3/4   100% 13/13     78        66
R36_p38     open       yes    yes     100% 8/8     100% 6/6     100% 8/8     82% 9/11     34        19
R37_p26     open       yes    yes   100% 13/13     100% 5/5            -     90% 9/10     55        42
ALL         open    100% 6/6 100% 6/6   100% 57/57   100% 34/34    97% 35/36    93% 63/68    407       332

R02_p123    blind      yes     NO            -     100% 1/1            -      33% 2/6     89        89
R03_p135    blind      yes    yes            -     100% 1/1       0% 0/1      40% 2/5     69        69
R04_p59     blind      yes     NO            -     100% 5/5            -      33% 2/6     42        42
R13_p45     blind      yes    yes    92% 24/26     100% 5/5     54% 7/13    83% 10/12    105        92
R21_p96     blind      yes    yes       0% 0/6      67% 2/3       0% 0/2      40% 2/5     75        75
R25_p19     blind  refused     NO            -            -            -            -      0         0
R30_p65     blind      yes     NO       0% 0/8     100% 1/1            -      40% 2/5     71        71
R31_p278    blind      yes    yes            -     100% 2/2            -   100% 11/11     27        27
R34_p49     blind      yes    yes      0% 0/10      62% 5/8       0% 0/1      50% 2/4    112       112
ALL         blind   100% 9/9  62% 5/8    48% 24/50    85% 22/26     41% 7/17    61% 33/54    590       577

notes:
  R02_p123: the truth states no samples; the grid put 0 numeric cell(s) in a blow-count column
  R25_p19: tabular sheet with no depth scale: 10 layers and 1 samples not scored by depth
  R31_p278: the truth states no samples; the grid put 0 numeric cell(s) in a blow-count column

warnings seen:
  R02_p123: the depth unit is not stated on these pages and could not be read from their text; depths are in whatever the ruler prints
  R03_p135: no column header states the depth unit; ft was read off depths written into the log's own text
  R04_p59: the depth unit is not stated on these pages and could not be read from their text; depths are in whatever the ruler prints
  R13_p45: page 45: the text was read optically (azure_di); boxes and column edges are softer than on an embedded text layer
  R13_p45: page 45: no ruled column edges were found; the columns below come from the header labels alone and their x bands are approximate
  R15_p46: page 46: 1 text line(s) run diagonally across the page (a watermark or a stamp) and were left out of the grid
  R21_p96: no column header states the depth unit; m was read off depths written into the log's own text
  R25_p19: page 19: no ruled column edges were found; the columns below come from the header labels alone and their x bands are approximate
  R25_p19: page 19: no columns could be laid out — nothing on this page is placed
  R25_p19: page 19: no depth ruler was found — nothing on this page carries a depth
  R25_p19: no description column was identified, so no layers were read
  R30_p65: the depth unit is not stated on these pages and could not be read from their text; depths are in whatever the ruler prints
  R34_p49: no column header states the depth unit; m was read off depths written into the log's own text

```

---

### log_grid scorecard, round 4 (collinear rule segments)

The lead found it from geometry alone: on the gINT sheets the rule under the
header row IS there, drawn as four collinear segments with a 14 pt gap where
the depth-scale column's tick marks live. Measured stroke by stroke no piece
crosses the form, which is why round 3 concluded the rule did not exist.

Collinear pieces sharing a coordinate and leaving a gap no wider than one
narrow column (20 pt) are now joined before any length is measured -- column
edges, the header band and stratum lines alike. That alone named every column
on those sheets: MATERIAL SYMBOL, Elev. (m), USCS, Sample Description, Depth
Scale (m), Number, Type, Recov. (cm), Penetr. resist BL/15cm, Remarks. Ten of
fifteen columns each, from a vocabulary that already held the words.

Two scorer rules were wrong as well, and both were the scorer imposing
structure the grid never claims:

- **A blow record spread over one cell per drive has no order.** Each cell is
  an independent value with its own depth and its own box. Requiring page
  order failed a sheet whose drives read 4, 6, 8 down the column against a
  hand truth of 4, 8, 6. A record found inside ONE cell must still be in
  order -- there the order is part of what the cell says -- but a stack of
  cells is now matched as a multiset.
- **The blow-count column is tried before the wider family.** A sample id
  standing between two drives is in the family and its number is not a drive.
  (A dash after a letter is also no longer read as a minus, so "S-7" is seven
  and not minus seven: fixed in planlens, not here.)
- **The ruler verdict asks of each stated depth whether the page reaches it**,
  rather than asking whether the page covers the truth's RANGE. A range is
  meaningless where a truth states two depths a third of a metre apart on a
  sheet spanning eight, and asking it that way called a correct continuation
  sheet wrong.

**Result.** The open six did not move for the fourth round running: 407 cells
placed, all rates identical. On the blind nine, blow records and N values went
24/50 to 50/50, the unit 5/8 to 6/8 and index values 7/17 to 8/17; the ruler
stays 9/9 with no wrong ruler anywhere and layer tops 22/26. Every gINT sheet
that refused a ruler two rounds ago now reads its samples in full.

### log_grid scorecard, 2026-09-17

Tolerances: sample and index values 0.15 m, layer tops 0.3 m; depths compared in metres whatever the log prints. Open set = R36, R37, R06, R07, R15, R28.

```
log         set      ruler   unit      samples       layers        index       fields  cells unmatched
------------------------------------------------------------------------------------------------------
R06_p51     open       yes    yes     100% 4/4     100% 5/5     100% 4/4      75% 6/8     62        54
R07_p30     open       yes    yes     100% 8/8     100% 9/9     100% 6/6   100% 13/13    103        93
R15_p46     open       yes    yes     100% 8/8     100% 4/4   100% 14/14   100% 13/13     75        56
R28_p57     open       yes    yes   100% 16/16     100% 5/5      75% 3/4   100% 13/13     78        66
R36_p38     open       yes    yes     100% 8/8     100% 6/6     100% 8/8     82% 9/11     34        19
R37_p26     open       yes    yes   100% 13/13     100% 5/5            -     90% 9/10     55        42
ALL         open    100% 6/6 100% 6/6   100% 57/57   100% 34/34    97% 35/36    93% 63/68    407       330

R02_p123    blind      yes     NO            -     100% 1/1            -      33% 2/6     45        45
R03_p135    blind      yes    yes            -     100% 1/1     100% 1/1      40% 2/5     42        41
R04_p59     blind      yes     NO            -     100% 5/5            -      33% 2/6     32        32
R13_p45     blind      yes    yes   100% 26/26     100% 5/5     54% 7/13    83% 10/12    105        91
R21_p96     blind      yes    yes     100% 6/6      67% 2/3       0% 0/2      40% 2/5     48        45
R25_p19     blind  refused     NO            -            -            -            -      0         0
R30_p65     blind      yes    yes     100% 8/8     100% 1/1            -      40% 2/5     45        41
R31_p278    blind      yes    yes            -     100% 2/2            -   100% 11/11     27        27
R34_p49     blind      yes    yes   100% 10/10      62% 5/8       0% 0/1      50% 2/4     85        80
ALL         blind   100% 9/9  75% 6/8   100% 50/50    85% 22/26     47% 8/17    61% 33/54    429       402

notes:
  R02_p123: the truth states no samples; the grid put 0 numeric cell(s) in a blow-count column
  R25_p19: tabular sheet with no depth scale: 10 layers and 1 samples not scored by depth
  R31_p278: the truth states no samples; the grid put 0 numeric cell(s) in a blow-count column

warnings seen:
  R02_p123: the depth unit is not stated on these pages and could not be read from their text; depths are in whatever the ruler prints
  R03_p135: no column header states the depth unit; ft was read off depths written into the log's own text
  R04_p59: the depth unit is not stated on these pages and could not be read from their text; depths are in whatever the ruler prints
  R13_p45: page 45: the text was read optically (azure_di); boxes and column edges are softer than on an embedded text layer
  R13_p45: page 45: no ruled column edges were found; the columns below come from the header labels alone and their x bands are approximate
  R15_p46: page 46: 1 text line(s) run diagonally across the page (a watermark or a stamp) and were left out of the grid
  R25_p19: page 19: no ruled column edges were found; the columns below come from the header labels alone and their x bands are approximate
  R25_p19: page 19: no columns could be laid out — nothing on this page is placed
  R25_p19: page 19: no depth ruler was found — nothing on this page carries a depth
  R25_p19: no description column was identified, so no layers were read

```

## WP2b -- the record, the log reader and DIGGS

The log reader scorecard has TWO columns from here on. **before** is what
`log_grid` alone recovered; **after** is what the reader's record holds. The
grid runs once and is handed to the reader, so the difference between the
columns is the model and nothing else. Reporting only the after column would
credit the reader with everything the geometry already had.

The metrics are the plan's: N values exact; sample depths, index values and
water levels within 0.15 m; layer tops within 0.3 m; USCS symbols matched;
recovery and RQD where printed; header fields recovered. The matching rules
are the WP2a ones, restated in `report_ingest/log_scoring.py` so the two
measurements mean the same thing.

### The grid-only baseline, 2026-09-17 (no model, no key, no network)

Run with `measure_wp2b_logs --grid-only`. Worth re-running whenever planlens
changes; it is the floor every reader number is measured against.

### WP2b log scorecard, 2026-09-17 -- log_grid only

The floor the reader is measured against. NOTE: log_grid on the current feature/log-grid branch now finds a ruler on R21, R30 and R34, which the round-3 WP2a table above lists as having none -- the grid moved, the table did not.

Tolerances: samples, index values and water 0.15 m (0.15 m for water), layer tops 0.3 m; depths compared in metres whatever the log prints; N values exact. Open set = R36, R37, R06, R07, R15, R28.

```
open -- 6 log(s)
metric                  before         after
--------------------------------------------
n_value             100% 24/24             -
blows               100% 33/33             -
sample_depth        100% 44/44             -
layer_top           100% 34/34             -
uscs                 43% 12/28             -
water                 42% 5/12             -
recovery            100% 32/32             -
index                97% 31/32             -
fields               93% 63/68             -
OVERALL           91% 278/307   

blind -- 9 log(s)
metric                  before         after
--------------------------------------------
n_value              96% 24/25             -
blows               100% 25/25             -
sample_depth         90% 28/31             -
layer_top            61% 22/36             -
uscs                    0% 0/7             -
water                  40% 2/5             -
recovery            100% 15/15             -
index                 44% 7/16             -
fields               61% 33/54             -
OVERALL           73% 156/214   

all -- 15 log(s)
metric                  before         after
--------------------------------------------
n_value              98% 48/49             -
blows               100% 58/58             -
sample_depth         96% 72/75             -
layer_top            80% 56/70             -
uscs                 34% 12/35             -
water                 41% 7/17             -
recovery            100% 47/47             -
index                79% 38/48             -
fields              79% 96/122             -
OVERALL           83% 434/521   

log           set           before        after  calls  unres  look      s
--------------------------------------------------------------------------
R02_p123      blind        38% 3/8            -      -      -     -      -
R03_p135      blind        67% 6/9            -      -      -     -      -
R04_p59       blind       57% 8/14            -      -      -     -      -
R06_p51       open       77% 23/30            -      -      -     -      -
R07_p30       open       86% 55/64            -      -      -     -      -
R13_p45       blind      82% 62/76            -      -      -     -      -
R15_p46       open       89% 57/64            -      -      -     -      -
R21_p96       blind      73% 16/22            -      -      -     -      -
R25_p19       blind        0% 0/14            -      -      -     -      -
R28_p57       open       95% 56/59            -      -      -     -      -
R30_p65       blind      86% 19/22            -      -      -     -      -
R31_p278      blind      93% 13/14            -      -      -     -      -
R34_p49       blind      83% 29/35            -      -      -     -      -
R36_p38       open       95% 42/44            -      -      -     -      -
R37_p26       open       98% 45/46            -      -      -     -      -

warnings the grid raised:
  R02_p123: the depth unit is not stated on these pages and could not be read from their text; depths are in whatever the ruler prints
  R03_p135: no column header states the depth unit; ft was read off depths written into the log's own text
  R04_p59: the depth unit is not stated on these pages and could not be read from their text; depths are in whatever the ruler prints
  R13_p45: page 45: the text was read optically (azure_di); boxes and column edges are softer than on an embedded text layer
  R13_p45: page 45: no ruled column edges were found; the columns below come from the header labels alone and their x bands are approximate
  R15_p46: page 46: 1 text line(s) run diagonally across the page (a watermark or a stamp) and were left out of the grid
  R25_p19: page 19: no ruled column edges were found; the columns below come from the header labels alone and their x bands are approximate
  R25_p19: page 19: no columns could be laid out — nothing on this page is placed
  R25_p19: page 19: no depth ruler was found — nothing on this page carries a depth
  R25_p19: no description column was identified, so no layers were read

```

### WP3 lab scorecard, 2026-09-17 -- the page's tables only

The deterministic baseline for WP3, taken before any model ran: every number in the page's own detected tables, on all 31 hand-truthed sheets. 'kind' and 'link' have no before column because a table cannot answer them.

Tolerances: a depth links within 0.15 m, compared in metres whatever the sheet prints; an index value exact to 0.01; a grading within 1.0 percent; a curve within the tolerance its own truth file states. `kind` and `link` have no before column: a table cannot answer them. Open set = R36, R28, R17, R06.

```
open -- 16 sheet(s)
metric            before         after
--------------------------------------
index        87% 202/231             -
series         88% 38/43             -
curve          41% 11/27             -
OVERALL      83% 251/301

blind -- 15 sheet(s)
metric            before         after
--------------------------------------
index        45% 125/275             -
series         87% 76/87             -
curve             0% 0/4             -
OVERALL      55% 201/366

all -- 31 sheet(s)
metric            before         after
--------------------------------------
index        65% 327/506             -
series       88% 114/130             -
curve          35% 11/31             -
OVERALL      68% 452/667

kind                 sheets        before         after
-------------------------------------------------------
atterberg                 2    100% 12/12
chemical                  4     78% 21/27
compaction                1       50% 4/8
density                   1       88% 7/8
direct_shear              2      47% 8/17
gradation                 9   80% 214/269
moisture_content          1    100% 17/17
organic_content           1      100% 3/3
summary_table             4   72% 155/216
swell_consolidation       2     52% 11/21
triaxial                  3       0% 0/64
unconfined_rock           1        0% 0/5

sheet                       set          before       after  calls  zoom  unres  look      s
--------------------------------------------------------------------------------------------
atterberg__R25_p87          blind   100% 106/106           -      -     -      -     -      -
atterberg__R28_p198         open       100% 8/8           -      -     -      -     -      -
atterberg__R36_p52          open       100% 4/4           -      -     -      -     -      -
chemical__R17_p155          open              -           -      -     -      -     -      -
chemical__R28_p172          open       100% 3/3           -      -     -      -     -      -
chemical__R28_p210          open     100% 18/18           -      -     -      -     -      -
chemical__R36_p62           open         0% 0/6           -      -     -      -     -      -
compaction__R17_p136        open        50% 4/8           -      -     -      -     -      -
consolidation__R36_p54      open       38% 5/13           -      -     -      -     -      -
consolidation__R36_p56      open        75% 6/8           -      -     -      -     -      -
direct_shear__R25_p142      blind      40% 4/10           -      -     -      -     -      -
direct_shear__R36_p58       open        57% 4/7           -      -     -      -     -      -
gradation__R06_p67          open      56% 14/25           -      -     -      -     -      -
gradation__R15_p112         blind    100% 27/27           -      -     -      -     -      -
gradation__R15_p82          blind    100% 28/28           -      -     -      -     -      -
gradation__R17_p114         open     100% 46/46           -      -     -      -     -      -
gradation__R25_p56          blind      25% 8/32           -      -     -      -     -      -
gradation__R28_p163         open     100% 37/37           -      -     -      -     -      -
gradation__R28_p176         open      71% 25/35           -      -     -      -     -      -
gradation__R28_p177         open      83% 29/35           -      -     -      -     -      -
gradation__R35_p29          blind        0% 0/4           -      -     -      -     -      -
moisture_density__R03_p293  blind      100% 3/3           -      -     -      -     -      -
moisture_density__R15_p184  blind       88% 7/8           -      -     -      -     -      -
moisture_density__R27_p45   blind    100% 17/17           -      -     -      -     -      -
summary_table__R28_p159     open     100% 48/48           -      -     -      -     -      -
summary_table__R32_p114     blind       4% 1/23           -      -     -      -     -      -
summary_table__R35_p6       blind       0% 0/39           -      -     -      -     -      -
triaxial__R27_p85           blind       0% 0/12           -      -     -      -     -      -
triaxial__R35_p124          blind       0% 0/24           -      -     -      -     -      -
triaxial__R35_p45           blind       0% 0/28           -      -     -      -     -      -
unconfined__R22_p148        blind        0% 0/5           -      -     -      -     -      -

```

---

## CLUSTER RUN 1 -- 2026-09-18 -- app 5.20.0 + notebook shim, `max_reports=2`

The first production numbers. Everything below came through the Funhouse
Prompter API on the cluster; the owner ran the five-stage cell. Package was
5.20.0 with a notebook shim (client wrapper renaming `max_tokens` to
`max_completion_tokens`, `prefer_chat` off) because 5.20.1, which does the
same in the package, was still behind the Nexus mirror. Two reports per
stage. Tiers and what actually served them: `funhouse-gpt-high` =
gpt-5.4-2026-03-05 (review, readers), `funhouse-gpt-medium` =
gpt-5.1-2025-11-13 (triage), `funhouse-gpt-low` = gpt-4.1-mini-2025-04-14
(vision).

### Labels (WP1b): R09, R11 -- 307 hand-labelled pages

```
                            before     after
strict accuracy              0.902     0.928
accepting alternates         0.902     0.928
review verdicts: fixed 9, broke 1, still_wrong 0

label          n   P before  R before   P after  R after
narrative      9      0.571     0.444     0.700    0.778   <- below the 0.98 gate
plan           3      1.000     1.000     1.000    1.000
profile        8      0.889     1.000     1.000    1.000
boring_log    30      1.000     1.000     1.000    1.000
lab_test     227      0.982     0.987     0.987    0.996
figure         3      0.000     0.000     0.000    0.000
divider       11      0.421     0.727     0.421    0.727   (untouched by the review)
cover          2       --       0.000     1.000    1.000
other         14      0.000     0.000     0.333    0.071

report  pages  before  after  changes  tools   calls   in       out    s
R09       151   0.901  0.934        7  20/60       4   63,392  1,700  43
R11       156   0.904  0.923        3  40/60       4   86,790  2,172  43
```

Triage: R09 geotechnical report / standard / not bound / toc none / scan
0.03; R11 "report appendix or figure(s)" / appendix_only / bound 1 / no_toc.
Cost: 8 calls, 150,182 in (+2,432 cached), 3,872 out, 86 s.

### Vision-first labels (WP5): same 307 pages, page mode, 100 dpi, no outline

```
                        rules   +review   vision
strict accuracy         0.902     0.928    0.928
report R09              0.901     0.934    0.947
report R11              0.904     0.923    0.910

label          n   P vision  R vision
narrative      9      0.900     1.000   (review: 0.700 / 0.778)
plan           3      1.000     0.667
profile        8      1.000     1.000
boring_log    30      1.000     1.000
lab_test     227      0.996     0.978
figure         3      0.300     1.000
divider       11      0.421     0.727   (identical in all three columns)
other         14      1.000     0.071
```

Cost: 307 calls (one per page), 761,876 in (+327,680 cached), 11,025 out,
693 s (2.3 s/page). Sheet mode (six pages a call) is the untested cheaper
setting. Reading: looking alone equals rules + review on 307 pages, beats
both on narrative and figure recall, and loses on plan recall and lab_test
recall. The obvious next experiment is vision as a third voice the review
consults, not a replacement.

### Logs / lab / narrative: NO model numbers yet

All six items (R02_p123, R03_p135; atterberg R25_p87, R28_p198; R05, R06)
failed in the FIRST run on the same parameter refusal, the scorer stored the
failure on the score, the stage wrote the blob, and the second run skipped
all six as "already done". That is a resume defect, fixed in 5.20.2
(`_saved_failure`: a saved failure is retried). Grid-only baselines from the
run: logs 9/17 (sample_depth 1/1, layer_top 2/2, uscs 0/1, recovery 2/2,
fields 4/11); lab tables 114/114 (atterberg 8/8, summary_table 106/106).

### Run 3 (same day, after the frozen failures were cleared): readers still 0 calls

R02_p123 3/8 -> 0/0, R03_p135 6/9 -> 0/0, atterberg R25_p87 106/106 -> 0/0,
R28_p198 8/8 -> 0/0, R05 and R06 recall 0/0 -- every reader item "0 model
call(s), 0 in / 0 out", 20-25 s each (the document open and the grid, then
the refused request). Cause found offline: `strict_schema` on every
`output_format` model -- triage 0, review 0, vision 0 refused keywords; log
reader 63, lab reader 149, narrative reader 52 (`default`, `minItems`,
`maxItems`). Strict mode validates the schema before the call. Fixed in
5.20.3 (keywords stripped, limits folded into descriptions, a test walks
every model). Reader numbers remain unmeasured.

### Run 5 (shim on the engine's client property): ALL FIVE STAGES RAN -- first reader numbers

Package 5.20.0 + the order-proof notebook shim (equivalent to 5.20.4). Readers
served by gpt-5.4-2026-03-05. Two items per stage.

**Logs (WP2b), 2 blind logs, 1 model call each (budget 6):**

```
metric          grid only    reader
sample_depth     100% 1/1   100% 1/1
layer_top        100% 2/2   100% 2/2
uscs               0% 0/1     0% 0/1
recovery         100% 2/2   100% 2/2
fields            36% 4/11   55% 6/11
OVERALL           53% 9/17   65% 11/17
R02_p123   38% 3/8 -> 62% 5/8   1 call, 5 unresolved, 2 from the picture, 6,033 in / 526 out, 27 s
R03_p135   67% 6/9 -> 67% 6/9   1 call, 4 unresolved, 1 from the picture, 6,006 in / 527 out, 26 s
```
The reader took ONE call per log and left 4-5 items unresolved: it does not
go back for what it could not settle. Header fields and the USCS symbol are
where the misses are.

**Lab (WP3), 2 sheets (1 open, 1 blind), 1 call each:**

```
metric        tables only     reader
kind                   -     100% 9/9
link                   -     100% 9/9
index          100% 56/56   100% 56/56
series         100% 53/53   100% 53/53
curve            100% 5/5     100% 5/5
OVERALL      100% 114/114  100% 132/132
```
Every value, every test kind, every link to boring and depth. 10-11k input
tokens and 32-63 s a sheet; no zoom used.

**Narrative (WP4), R05 (open) + R06 (blind):**

```
                    open R05      blind R06     all
recall              63% 17/27     66% 19/29     64% 36/56
precision           68% 17/25     79% 19/24     73% 36/49
agreement           67% 22/33     70% 23/33     68% 45/66
recall by type: enum 59%, int 50%, string 88%, list 56%; list items P 75% R 83%
summaries: 8/8 written, 8/8 within limit
calls 1 / 4; 10.9k / 69.9k in; 2.9k / 7.9k out; 28 / 100 s
```
Per question (2 asked each): RIGHT on both -- asceSevenVersion, boringCount,
boringDictionary, documentType, geophysicalTestingMention,
geotechnicalEngineerFirm, liquefactionPotential, naturalHazardSummary,
outsideProject, projectName, projectNumber, quickSummary, reportDate,
seismicCodeUsed, seismicParameterSummary, siteClass, tableCount,
testingProgramSummary. MISSED (reader null, truth answered): cptCount 2,
testPitCount 2, propertyType 2, previousInvestigationCount 1, projectPhase 1
-- the "0 vs null" convention for counts of things the report has none of,
and the owner's propertyType vocabulary. WRONG: siteResponseMention 2,
strata 2 (free prose scored at partial ratio 85), bearingCapacity 1,
figureCount 1, hazardAnalysisMention 1, recommendedFoundations 1,
structureCount 1, structureList 1, soilCorrosion 1, earthHazardsExposed 1
(+1 invented). Reading: identity, code and summary fields are essentially
solved; the losses are conventions (null vs 0, yes/no mention fields) and
long free-text matching, i.e. prompt and scorer work, not model capacity.

**Cost per item, gpt-5.4:** log ~6k in / 26 s; lab sheet ~10.5k in / 48 s;
narrative ~40k in / 64 s; label review ~75k in / 43 s per report; vision
(gpt-4.1-mini) ~380k in / 346 s per report in page mode.

**Full-run estimate (38 reports, 7,829 pages):** labels ~30 min; 15 logs +
31 sheets + 8 narratives ~40 min; vision in PAGE mode ~5 h (2.3 s/page) --
run vision separately in sheet mode, or after the rest.

### Run 6 -- THE FULL RUN (labels, logs, lab, narrative; vision apart) -- 5.20.0 + shim

Review/readers gpt-5.4-2026-03-05, triage gpt-5.1-2025-11-13. 10 label
reports (R18, R22, R31-R38) and 5 logs (R04_p59, R06_p51, R07_p30,
R13_p45, R15_p46) FAILED on 429 rate limits and are outside every number
below (fixed in 5.20.5: wait-and-retry ladder).

**Labels.** In-sample 13 reports, 4,082 pages (1 disputed page dropped):
strict 0.907 -> 0.915; fixed 156, broke 122, still_wrong 88.
```
label            n   P before R before  P after  R after
narrative      411     0.971   0.910     0.884    0.985
plan            18     0.303   0.556     0.382    0.722
profile         28     0.593   0.571     0.852    0.821
boring_log     273     0.959   0.941     0.947    0.978
test_pit_log   251     0.957   0.888     0.980    0.976
cpt_log         23     0.636   0.913     0.767    1.000
dcp_log         54     0.935   0.537     0.912    0.574
lab_test       991     0.929   0.988     0.968    0.959
calculation    977     0.992   0.971     0.998    0.950
figure          57     0.225   0.281     0.644    0.667
field_test      19     0.680   0.895     0.360    0.947
appended_rep   494     0.990   1.000     0.915    1.000
photos         132     0.904   0.856     0.920    0.962
divider         81     0.651   0.852     0.464    0.790
cover           29     1.000   0.379     1.000    0.759
toc             23     0.941   0.696     0.529    0.783
other          216     0.510   0.481     0.942    0.301
per report before->after: R09 .901->.934, R11 .904->.923, R12 .899->.899,
R13 .843->.929, R15 .871->.851, R16 .948->.955, R20 .968->.976,
R21 .962->.951, R23 .937->.902, R24 .481->.494, R28 .901->.947,
R29 .733->.718 (60 changes, 10 calls, tool budget exhausted), R30 .944->.959
```
OOS open (10 reports, 50 pages): 0.760 -> 0.780 strict, 0.840 -> 0.900
with alternates (fixed 5, broke 4). OOS blind (5 reports, 25 pages): 0.920
-> 0.920 (fixed 2, broke 2). The Claude dev-engine checkpoint (0.889 ->
0.945 on 673 pages) did NOT transfer to gpt-5.4 on the full set.
Triage: 17 of 28 reports `multi_document` (bound 1-4), R13 `scanned`
(1.00), R19 multi_document at scan 0.96, R10 document_type `other`, R11
appendix_only. Cost 137 calls, 4.31 M in (+0.90 M cached), 118 k out,
2,229 s (~154 k in / 80 s per report).

**Logs.** 10 of 15 ran. Grid-only vs reader, per log:
```
R02_p123 blind  3/8 -> 5/8     R21_p96 blind 16/22 -> 14/22   R30_p65 blind 19/22 -> 16/22
R03_p135 blind  6/9 -> 6/9     R25_p19 blind  0/14 -> 0/14    R31_p278 blind 13/14 -> 12/14
R28_p57  open  56/59 -> 58/59  R34_p49 blind 29/35 -> 26/35   R36_p38 open 42/44 -> 40/44
R37_p26  open  45/46 -> 43/46
```
Blind (the 5 that ran): 73% -> 64%; recovery 15/15 -> 3/15, layer_top
22/36 -> 14/26, index 7/16 -> 0/3, uscs 0/7 -> 1/2. The reader RE-EMITS the
record and drops values the grid had. One call per log; 3-8 unresolved
each; 25 s and ~6 k in per log. R25_p19 reads nothing either way.

**Lab.** 31 sheets: tables-only 68% (452/667) -> reader 87% (716/825);
blind 55% -> 87%; kind 94%, link 94%; index 65 -> 85%; series 88 -> 92%;
curve 35 -> 65%. Per kind (after): summary_table 100% (260/260),
moisture_content 100%, unconfined_rock 100%, density 100%,
swell_consolidation 92%, gradation 86%, triaxial 60% (0 before),
direct_shear 57%, chemical 54%, compaction 10%. REGRESSIONS below the
tables: gradation R15_p82 28/28 -> 11/30, gradation R17_p114 46/46 ->
42/56, chemical R28_p210 18/18 -> 14/22, chemical R28_p172 3/3 -> 1/5,
compaction R17_p136 4/8 -> 1/10, atterberg R36_p52 4/4 -> 5/6. 1.1 calls,
~10.7 k in, 30 s per sheet; zoom used twice.

**Narrative.** 8 reports: recall 64% (151/235), precision 74% (147/200),
agreement 66%; open (R05, R36) 71/76; blind (6) 62/73. By type: enum 71%,
int 52%, string 78%, list 41%; list items P 64% R 48%; summaries 32/32
written and within limit. Per report recall/precision: R05 63/68, R06
66/79, R15 68/81, R21 70/77, R26 61/71, R28 56/55, R36 79/84, R37 52/76.
Always right (8/8): documentType, geophysicalTestingMention,
geotechnicalEngineerFirm, liquefactionPotential, naturalHazardSummary,
outsideProject, projectName, projectNumber, quickSummary,
seismicParameterSummary, testingProgramSummary. Systematic misses:
cptCount 7 missed, testPitCount 5 missed (null vs 0), siteResponseMention
7 WRONG, earthHazardsExposed 3 wrong + 5 invented, propertyType 4 missed,
boringDictionary 4 missed, postName 3/4 missed, primeAe 3/5 missed,
figureCount 4 wrong, structureCount 4 wrong, structureList 4 wrong,
recommendedFoundations 4 wrong, strata 4 wrong, bearingCapacity 3 wrong,
hazardAnalysisMention 3 wrong, soilCorrosion 3 wrong. 1.6 calls, 26 k
in, 54 s per report.

**Levers, in order:** (1) FLOOR under log and lab readers -- start from
the grid/tables, let the model add or correct with evidence, never drop;
(2) narrative conventions (null vs 0, mention fields yes/no, hazards
vocabulary, postName = city, list scoring) -- the owner's schema review;
(3) label review precision (it breaks nearly as much as it fixes on
gpt-5.4: divider, toc, field_test, appended_report) and the R24/R29 class.

### Run 7 -- the rate-limited items redone: the COMPLETE four-stage picture

All 38 label reports and all 15 logs now in. Labels total cost 191 calls,
6.01 M in (+1.29 M cached), 161 k out, 2,948 s (~$25 of the day's spend by
the owner's budget reading; readers ~$3 more).

**Labels, complete.** In-sample 14 reports, 4,146 pages: 0.908 -> 0.916
(fixed 158, broke 125, still_wrong 88). OOS open 10 reports, 50 pages:
0.760 -> 0.780 (alt 0.840 -> 0.900). OOS blind 14 reports, 69 pages:
0.783 -> 0.870 (alt 0.855 -> 0.928); HONEST BLIND (12 reports nobody
opened, 60 pages): **0.767 -> 0.850 strict, 0.833 -> 0.917 with
alternates** (fixed 11, broke 5). Reading: on reports the rules were tuned
on there is little left for the review to fix and it breaks as much as it
fixes; on unseen reports the rules start lower and the review recovers
+8 points. Per-report blind: R19 .60->1.0, R31 .60->1.0, R38 .20->.80,
R36 .80->1.0; R25 1.0->.60 (the one it damaged). New triage rows: R18
multi_document, R22 multi_document 2, R31 multi 2, R32 standard, R33
multi (scan .53), R34 multi 3, R35 "report appendix or figure(s)",
R36/R37 standard, R38 scanned 1.00. 19 of 38 multi_document.

**Logs, complete (15).** All 83% (434/521) -> 80% (418/521). Open 91% ->
93% (uscs 43% -> 96%, water 42 -> 50%, fields 93 -> 85%). BLIND 73% -> 62%:
n_value 24/25 -> 19/25, blows 25/25 -> 19/25, sample_depth 28/31 -> 23/31,
recovery 15/15 -> 3/15, index 7/16 -> 0/16; uscs 0/7 -> 6/7, layer_top
22/36 -> 24/36. Per log added this run: R04_p59 8/14 -> 8/14, R06_p51
23/30 -> 28/30, R07_p30 55/64 -> 61/64, R13_p45 (scanned, blind) 62/76 ->
45/76 (-17), R15_p46 57/64 -> 56/64 (22 unresolved). The reader adds USCS
symbols (the picture) and loses grid rows on templates it has not seen.
FLOOR under the reader stays lever #1; the blind loss is the argument.

### Run 8 -- vision SHEET mode over the corpus -- VOID (a design flaw, fixed in 5.21.1)

38 reports, 1,321 calls, 3.56 M in (+1.66 M cached), 241 k out, 5,043 s,
gpt-4.1-mini (~$1.50). In-sample 4,147 pages: strict 0.557 (rules 0.908);
OOS open 0.700 (rules 0.760), OOS blind 0.757 (0.786), honest blind 0.750
(0.767). Per report, the two page-mode reports: R09 0.947 -> 0.245, R11
0.910 -> 0.308; R21 0.187. Label pattern: narrative P 0.431 R 0.894,
figure P 0.063 R 0.689, appended_report R 0.000, other R 0.032, toc R 0.92
P 0.55. CAUSE: sheet mode used planlens' `render_thumbnails`, which
captions each tile "<index> <kind>" (text / figure / form / scanned), and
the prompt passed planlens' legend through; the model read the caption as
the answer. Document mode's strip used the same sheets. Fixed in 5.21.1
(`render_number_sheets`: "p. N" only; a test forbids `render_thumbnails` in
any vision call). These numbers measure the caption, not the model; the
page-mode figures (run 5) stand. Sheet mode must be re-run on 5.21.1.

### Run 9 -- 5.21.1 on the cluster: sheet mode validated, document mode hit the gateway

2026-09-20, gpt-4.1-mini, the first run in which no vision call saw a page's
rule-derived kind (run 8's defect, fixed in 5.21.1).

**Sheet mode, number-only sheets: 0.932 strict on 307 pages** (R09 0.947, R11
0.917), **26 calls and about $0.04 a report**. Page mode on the same two
reports (run 5) scored 0.928 in 307 calls at about $0.25. Same accuracy, a
sixth of the calls, a quarter of the money. Run 8's 0.557 measured the
"<index> <kind>" captions and is void. **Sheet mode is the mode to run and
needs no further work.**

**Document mode, R09 (151 pp, text pages): 0.927 in 5 windows**, 254,724 input
tokens -- about 51,000 a call. Not an anomaly: the pages are A4, and A4 at 100
dpi scales to 768 x 1086, which is SIX 512 px tiles where a letter page is
four. **Three pages ended unresolved** because the model skipped them and the
window overlapping each of them skipped them too.

**Document mode, R11 (156 pp, scanned pages with an OCR text layer): every
window FAILED** -- `APIStatusError: The page was not displayed because the
request entity is too large`. That is the gateway's limit on the REQUEST BODY.
The 50-image cap measured on 2026-09-18 was probed with tiny images and says
nothing about bytes: a scanned page at 100 dpi is a few hundred KB of PNG, and
36 of them plus four contact sheets, base64 at four bytes for three, is tens
of megabytes.

**Fixed in 5.21.2** (same day, offline): page pictures travel as JPEG at
quality 80 at the render's own pixel size, so the token count is unchanged and
only the bytes move (measured on a scan-like page: 780 KB PNG -> 250 KB JPEG,
ratio 0.32; on a crisp vector text page JPEG runs 0.84 to 1.20 of the PNG, and
the rule is unconditional anyway); a window the gateway still refuses halves
itself and re-cuts every window still queued, counted as `cost["splits"]`; and
every page left unresolved gets one page-mode call, counted as
`cost["fallback_pages"]`.

**What run 10 must answer:** does JPEG alone get R11's windows through, or
does the split have to fire? And `vision_detail="low"` on accuracy, which is
still unmeasured and is the only lever that moves a document run's token bill.

### Run 10 -- 5.21.1, SHEET mode over the corpus with number-only sheets (VALID)

38 reports, 1,321 calls, 3.41 M in (+1.64 M cached), 243 k out, 4,364 s,
**$1.82** ($0.05 a report), gpt-4.1-mini, one sheet of six pages a call.

```
set                      pages   rules  +review   vision(sheet)
in-sample (14)           4,147   0.908   0.916    0.655
OOS open (10)               50   0.760   0.780    0.780
OOS blind (14)              70   0.786   0.870    0.843
honest blind (12)           60   0.767   0.850    0.867
```
(+review from run 7; vision from this run.) Per in-sample report: R09 .947,
R11 .917, R15 .926, R30 .829, R16 .798, R12 .767, R23 .741, R20 .716,
R28 .703, R18 .656, R13 .589, R29 .580, R24 .409, **R21 .196**.

Why in-sample is low while blind is high -- the label classes, in-sample:
```
label            n   P rules R rules  P vision R vision
appended_report 494   0.990   1.000    0.000   0.000   <- never emitted; needs document context
other           216   0.507   0.481    0.083   0.005   <- never emitted
calculation    1009   0.992   0.971    0.808   0.587   <- printouts read as tables/narrative
narrative       433   0.973   0.915    0.472   0.910   <- the false-positive sink for the two above
lab_test        992   0.929   0.988    0.803   0.917
boring_log      273   0.959   0.941    0.784   0.919
test_pit_log    251   0.957   0.888    0.771   0.697
plan             18   0.303   0.556    0.567   0.944   <- vision better
profile          28   0.571   0.571    0.525   0.750   <- vision better
figure           61   0.267   0.328    0.156   0.721
photos          132   0.904   0.856    0.697   0.977   <- vision better
cover            30   1.000   0.367    0.595   0.833   <- vision better
toc              25   0.947   0.720    0.535   0.920   <- vision better
divider          83   0.657   0.855    0.458   0.783
```
R21 (729 pp) carries the bound-in bridging report -- hundreds of
`appended_report` pages a thumbnail cannot know are appended -- and the
494 appended + 216 other + 417 missed calculations are ~27% of the in-sample
pages, which is the whole gap between 0.655 and the rules. The blind sets
have few such pages, and there a $0.05 vision pass beats the rules by 10
points and equals the $0.45 review (0.867 vs 0.850).

Reading: rules and vision are COMPLEMENTARY by label class -- the rules own
the structural labels (appended_report, other, calculation, lab_test) and
vision owns the visual ones (plan, profile, photos, cover, toc, figure
recall). That is the vote the owner asked for, and it can be scored with NO
model calls from the run files already on disk (labels runs + vision runs):
per-label trust, agreement rate, and the disagreement set that would go to
review. The labels runs of 2026-09-18 live in /tmp/report_ingest_520 on the
cluster and may not have survived a restart.

Document mode (5.21.2, JPEG + splitting) is the pass that can recover
appended_report and calculation, because it sees the dividers and the
report's structure; R09 alone gave dividers R 0.857 (sheet 0.727).

## Log-template recogniser, 2026-09-20 -- 5.24.0, no model and no network

The owner's observation, 2026-09-20: two firms' logs are so standard that
simple rules would almost always catch them, and one of the two templates has
shifted over the years. `report_ingest/log_templates.py` is the generic
machinery; the FINGERPRINTS are data in `raw/truth/templates.json`, which is
gitignored and names the firms. **Families are letters below and nothing here
names a firm, a site or a form's own words** -- the evidence column prints
which GROUP of phrases matched and how many, never the phrases.

Five fingerprints over two families: family A has three forms (an imperial
boring log, its metric twin, a test-pit log -- one gINT report name each) and
family B has two (a 2008 data template still in use in 2023, and a 2021 gINT
library). Family B is the one the owner said had drifted.

The threshold is **0.65**. Every log a fingerprint describes scores 0.97 or
better because the footer stamp is there and the footer is half the score; the
one page off the logs that any fingerprint reached scored 0.59 (a laboratory
sheet from the same gINT project, carrying the title block's words and none of
the footer). The line is drawn between the two, nearer the false one: a page
whose footer did not survive into text is not a page to claim on the title
block alone.

```
5 fingerprint(s), 15 truthed log(s)
30 non-log page(s) drawn with seed 20260920
threshold 0.65; family names are letters here and the evidence names its GROUP and not the form's own words -- both live only in the private fingerprint file
log         want       got          conf  margin  cols  evidence
R02_p123    family A   family A     1.00    0.91     6  footer 1, title 5, columns 8
R03_p135    family A   family A     1.00    0.91     6  footer 1, title 5, columns 8
R04_p59     family A   family A     1.00    0.89     4  footer 1, title 5, columns 7
R06_p51     -          -            0.00    0.00     0
R07_p30     family B   family B     0.97    0.80     5  footer 1, title 6, columns 6
R13_p45     -          -            0.00    0.00     0
R15_p46     family B   family B     0.97    0.80     5  footer 1, title 6, columns 6
R21_p96     family A   family A     1.00    0.89     7  footer 1, title 5, columns 9
R25_p19     -          -            0.00    0.00     0
R28_p57     family B   family B     0.97    0.80     5  footer 1, title 6, columns 6
R30_p65     family A   family A     0.98    0.86     6  footer 1, title 5, columns 8
R31_p278    family B   family B     0.97    0.83     5  footer 1, title 6, columns 6
R34_p49     family A   family A     1.00    0.91     7  footer 1, title 5, columns 9
R36_p38     -          -            0.00    0.00     0
R37_p26     -          -            0.00    0.00     0

Recognition on the hand-truthed logs
family        logs        recall       precision
family A         6      100% 6/6        100% 6/6
family B         4      100% 4/4        100% 4/4
no template      5             -        100% 5/5

Pages the hand says are NOT logs: 30 drawn, 0 claimed by a template

The grid's floor, scored against the hand truth, on the 10 log(s) a template claimed
metric           no column map         with it
n_value             100% 28/28      100% 28/28
blows               100% 28/28      100% 28/28
sample_depth         95% 42/44       95% 42/44
layer_top            90% 35/39       90% 35/39
uscs                   6% 1/17         6% 1/17
water                  0% 0/11         0% 0/11
recovery             45% 18/40       65% 26/40
index                 26% 6/23       48% 11/23
fields               77% 62/81       77% 62/81
overall            71% 220/311     75% 233/311

Per log, overall floor score
log         family             no map      with map
R02_p123    family A          50% 4/8       50% 4/8
R03_p135    family A          56% 5/9       56% 5/9
R04_p59     family A         57% 8/14      57% 8/14
R07_p30     family B        78% 50/64     81% 52/64
R15_p46     family B        73% 47/64     73% 47/64
R21_p96     family A        59% 13/22     59% 13/22
R28_p57     family B        71% 42/59     90% 53/59
R30_p65     family A        68% 15/22     68% 15/22
R31_p278    family B        93% 13/14     93% 13/14
R34_p49     family A        66% 23/35     66% 23/35
```

**Recognition is solved on this corpus**: 10 of 10 logs claimed for the right
family, 5 of 5 logs on undescribed forms left alone, 0 of 30 non-log pages
claimed. The footer stamp is what does it -- it is printed by the template and
by nothing else -- and every match scored 0.97 or better.

**The column map is worth +13 values** on the ten logs it claimed (220/311 to
233/311), all of it in `recovery` (18/40 to 26/40) and `index` (6/23 to
11/23): the two columns a form that stacks its sampling data under one heading
gives the general header vocabulary no way to name. R28_p57 alone goes 42/59
to 53/59.

### What the map was worth BEFORE the floor could read those columns

Worth recording, because the first measurement of this build was **+1 value**
and it would have read as a failure of the recogniser rather than of the cell
readers. The fingerprint named the columns correctly from the first run; what
the floor could not do was PARSE what they print. Four generic readers went
into `log_floor` in the same train:

1. a blow record printed with `+` separators (`2+1+2`), gINT's own spelling
   and as common in this corpus as the hyphen;
2. a blow record printed one increment to a line, each in its own cell, which
   is what BOTH families' forms do (a lone number is still left alone: one
   number in a blows column is as likely to be an N value);
3. a cell that names its own result -- `MC = 10.1%`, `LL = 38`,
   `% Passing #200 = 68.7`, `REC=29cm, 56%` -- read by its label, and only in
   a column of the index / tests / recovery family;
4. a sample named and typed in one cell (`S-1, SPT`, `GR-3, GRAB`).

The grid's floor over all fifteen truthed logs, before any model:

```
                       5.23.0    + cell readers   + fingerprints
n_value                  8/49          48/49            48/49
blows                   13/58          56/58            56/58
sample_depth            72/75          72/75            72/75
layer_top               56/70          56/70            56/70
uscs                     9/35           9/35             9/35
water                    0/17           0/17             0/17
recovery                16/47          18/47            26/47
index                   13/48          18/48            23/48
fields                  96/122         96/122           96/122
overall                283/521        373/521          386/521
```

No log loses ground at either step. Per log, 5.23.0 to the end state:
R02 4/8 → 4/8, R03 5/9 → 5/9, R04 8/14 → 8/14, R06 19/30 → 22/30, R07 34/64 →
52/64, R13 28/76 → 52/76, R15 31/64 → 47/64, R21 7/22 → 13/22, R25 0/14 →
0/14, R28 35/59 → 53/59, R30 7/22 → 15/22, R31 13/14 → 13/14, R34 13/35 →
23/35, R36 42/44 → 42/44, R37 37/46 → 37/46.

The reading: the recogniser is a cheap and apparently reliable voter, and its
value to the RECORD is bounded by what the floor can do with a named column.
`uscs` (9/35) and `water` (0/17) are the two metrics the floor still cannot
reach at all and are where the model is still doing all the work.

### The narrative levers of 5.24.0 are UNMEASURED

The front-matter union, the DRAFT conventions glossary, per-question
retrieval, the deterministic exploration answers, the quote gate and the
lenient scoring view all ship in 5.24.0 and **none of them has been run
against the tier that will do the work**. The standing numbers are still run
7's: recall 64 % (151/235), precision 74 % (147/200) over eight reports. The
next cluster run with `stages=("narrative",)` is what says whether the levers
move them; `measure_wp4_narrative.py --front-pages --lenient` is the
development checkpoint and a checkpoint is never a result.

### WP5 calc reader -- the floor alone, 2026-09-21

The floor alone: what a pattern reads off a calculation printout before any model call. Ten hand-truthed runs, 44 pages, six reports, seven kinds. No blind set -- see raw/truth/calc/README.md.

A value matches when its printed LABEL matches at partial ratio 80 and its VALUE within 2% or the last printed digit, compared in SI wherever both units convert; a program name matches at 85. `kind`, `method`, `subject` have no before column: a pattern over a page cannot answer them. THERE IS NO BLIND SET -- all 6 report(s) (R16, R18, R20, R23, R29, R30) were read while the reader's prompt was written, so every number here is in sample.

```
open -- 10 run(s)
metric            before         after
--------------------------------------
program         80% 8/10
inputs         73% 64/88
results        48% 36/75
OVERALL      62% 108/173
(the before OVERALL leaves out the metrics the floor is not asked; read the metrics)

all -- 10 run(s)
metric            before         after
--------------------------------------
program         80% 8/10
inputs         73% 64/88
results        48% 36/75
OVERALL      62% 108/173
(the before OVERALL leaves out the metrics the floor is not asked; read the metrics)

kind                          runs  pages        before         after
---------------------------------------------------------------------
lateral_pile                     1      7     68% 13/19
pavement                         1      4     88% 21/24
retaining_wall                   1      7      50% 7/14
settlement                       2      9     78% 31/40
shallow_foundation_bearing       2      4     57% 28/49
site_response                    2      8      60% 6/10
slope_stability                  1      4      12% 2/17

calculation                             set      pp floor      before       after calls zoom unres misp     s
-------------------------------------------------------------------------------------------------------------
lateral_pile__R20_p344                  open      7    12   68% 13/19           -     -    -     -    -     -
pavement__R29_p127                      open      4    45   88% 21/24           -     -    -     -    -     -
retaining_wall__R18_p50                 open      7    54    50% 7/14           -     -    -     -    -     -
settlement__R16_p162                    open      3    47   64% 14/22           -     -    -     -    -     -
settlement__R18_p41                     open      6    48   94% 17/18           -     -    -     -    -     -
shallow_foundation_bearing__R23_p371    open      3    29   54% 14/26           -     -    -     -    -     -
shallow_foundation_bearing__R30_p542    open      1    29   61% 14/23           -     -    -     -    -     -
site_response__R18_p32                  open      7    85     50% 1/2           -     -    -     -    -     -
site_response__R30_p555                 open      1    11     62% 5/8           -     -    -     -    -     -
slope_stability__R23_p386               open      4     9    12% 2/17           -     -    -     -    -     -

```
