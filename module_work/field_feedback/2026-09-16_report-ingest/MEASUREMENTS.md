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
