# Reference library review, 2026-11-10

The owner asked for a review of the `geotech-references` library as the
`references` helper uses it inside GeotechStaffEngineer, looking for silent
errors: a wrong value, citation or statement that reaches the main agent with
nothing to show it is wrong. Much of the library was built in bulk by a
model, partly from the PDFs' text layer. The main agent sees only the helper's few sentences, never a tool
result, a docstring or the page that was read, so a function that is wrong
without raising or saying so travels to the engineer's answer unchecked.

Open work from this review is in `BACKLOG.md` under the IDs named here. This
file and the registers in `findings/` hold the detail; the working folder
with the scripts, renders and probe output stays local
(`C:/dev/tmp/refs_review_2026-11-10/`).

## 1. How it was done

- **Read-only reviewers in parallel**, each on one part: the DM 7.1 lookups,
  the DM 7.2 lookups, the eleven FHWA lookup packages, the pavement family
  (UFC pavements, AASHTO 1993, backfill, expansive soils), EN 1997 with
  Caltrans and FEMA P-2082, the six structural packages, the shared machinery
  (search, fetch, catalogs, local packs, interpolation), the stored chapter
  text, the recorded runs, and coverage (what is built against what is
  printed). Two more answered side questions: what changed between the held
  DM 7.2 (2025) and its 2026 reissue, and which public references are worth
  adding.
- **Every value against the print.** A finding about a printed object names
  the PDF file, the PDF page and the printed page; equations and tables were
  read from rendered pages, not from the PDF's text layer. Behaviour findings
  carry the call and its output. For most registers the lead re-checked
  a sample against the print, or reproduced it by a call, before anything
  was fixed (named at the top of the register).
- **The recorded runs.** Every unique `activity.jsonl` from the live smoke
  waves and the field sessions (1,169 records): 41 helper consults read in
  full and all 524 recorded library lookups run again on today's code. Field
  sessions are named by date and version only.
- **Fixes on branches.** Each reviewer then fixed its own A and B findings in
  its own worktree of the library, under one set of rules: every value,
  sign and citation from the rendered page; content the manual does not print
  is rebuilt from the print or removed, never kept behind a warning; tests
  assert the printed values (the manual's own worked examples where there
  are any); the chapter text corrected with the code. A helper who has not
  seen the fixer's notes then checks every changed value against the print
  before the lead merges.
- **Judgment calls.** Where the print itself looks wrong, the owner left the
  call to the lead and asked for a later item to revisit them (OWN-136). In
  every case the function's description and result state the printed form,
  the coded form and the page.

## 2. What was found

| Area | What was checked | A | B | C | Register |
|---|---|---|---|---|---|
| DM 7.1 lookups (UFS 3-220-10, 2026) | all 147 public functions; 121 equations read as typeset; 108 match the print | 17 | 8 | 5 | `dm71.md` |
| DM 7.2 lookups (UFC 3-220-20, 2025) | all 246 functions; 28 not as printed; all nine digitised charts and tables differ from the print | 11 | 3 | 3 | `dm72.md` |
| FHWA GEC packages (GEC-4 to -13, micropile, FHWA pavements) | 144 functions, 119 print-checked here, 25 checked earlier | 8 | 29 | 13 | `gec.md` |
| UFC pavements, AASHTO 1993, backfill, expansive soils, UFC practice manuals | 193 functions in seven packages | 8 | 12 | 13 | `pave.md` |
| EN 1997-1, EN 1997-2, Caltrans Trenching and Shoring, FEMA P-2082 | all 118 functions; every table, factor and sign | 6 | 10 | 5 | `ec7.md` |
| Structural packages (EM 2104, EM 2107, UFC structural, UFC and GSA progressive collapse, Wood Handbook) | 383 functions scanned; about 1,000 printed cells and expressions | 4 | 6 | 6 | `struct.md` |
| Shared machinery | search, fetch, catalogs, packs, interpolation | 3 | 10 | 6 | `core.md` |
| Stored chapter text | 574 of 7,668 sections; every DM 7 equation entry (344) | 12 | 6 | 2 | `text.md` |
| Recorded runs | 41 consults, 702 helper tool calls, 524 lookups re-run | 4 | 6 | 2 | `traces.md` |

A = a wrong or unsafe-side value, citation or statement returned with no
signal, or a failure hidden; B = misleading, catchable by a careful caller;
C = a note. A few findings appear in two registers (traces-01 is dm72-02,
traces-02 is core-07, text-06 is struct-16, text-07 is pave-04).

The worst, each read off the print: DM 7.1 Eq. 8-13 with both signs reversed
(PI 40: 45.6 deg against 26.8); DM 7.1 Figure 8-46 about 64 times too stiff;
DM 7.2 Figure 5-18 (strip, su2/su1 0.2, H/B 1.0: 3.80 against 2.53, and seen
in a recorded run); DM 7.2 Figure 4-36 (design moment 44 % of the print);
EN 1997-1 Annex C passive pressure with water 73 % high; UFC 3-250-01
Eq. 20-4 effective porosity (0.0353 against 0.2995, drainage times about 8
times too short); UFC 4-023-03 Table 4-1 acceptance criteria up to twice too
permissive; UFC 3-301-01 Eqs. 16-18a/b (150 mph converted to 9.49 mph); GEC-7
Figure 4.3 friction angle 2 to 4 deg high for loose sand. In the stored text,
38 DM 7 sections hold formulas that are not the printed ones, GEC-11's text
is a paraphrase with content from memory (19 A errors in 16 of 40 sections),
and GEC-10 chapters 6 and 8 are written from memory.

**Coverage** (`COVERAGE.md`): the mapping that exists is sound and not
complete. 181 printed chapters or appendices have no text; 904 printed
figures have no catalog row (GEC-5 stops at Figure 11-9; GEC-13's 207 are
REF-08); 3,716 of 3,736 catalog rows show their figure on their page (four
DM 7.1 rows are one page late); about 1,050 printed tables are neither a
lookup nor typed into the text, and the helper has no way to read one.

**DM 7.2's reissue** (`editions.md`): UFS 3-220-20 (26 March 2026) is the
2025 UFC text under a new designation and a "For Information Only" status;
one wording change in 686 body pages, no equation, value or figure changed.

**New references** (`CANDIDATES.md`): 58 public candidates, 12 of the first
priority. The USACE manuals' currency could not be confirmed by an automated
check (the official site refuses automated reads).

## 3. Patterns

1. **Equations retyped from a garbled text layer.** DM 7.1: 14 of 121 coded
   equations wrong, every one with a nested fraction, an exponent on a bracket
   or five or more terms. DM 7.2: 19 multi-factor equations shortened or
   re-derived. Stored text: 93 wrong-equation records.
2. **Charts and tables written from memory of the kind of chart.** All nine
   DM 7.2 digitised objects, 8 of 9 DM 7.1 table and chart lookups, GEC-6
   Figure 5-19, GEC-7 Figure 4.3, micropile Figure 5-23: axes, rows and keys
   that are not the print's.
3. **Tests that assert the code's own values.** The DM 7 suites were green on
   every finding.
4. **A result that does not say what it is.** Bare numbers (DM 7.2 on every
   method, DM 7.1 on 138 of 147, EN 1997 on 45, the GECs on 27), several in
   tsf, ksf or percent; 423 of 1,339 reference methods described to the
   helper only as "float"; US-unit manuals in a library that says "all SI".
5. **Silent edges.** A table end clamped with nothing said (42 of the 81
   functions that reach a clamp), a NaN through the same door, the first of
   several rows for a partial key (19 GEC lookups), an empty list for an
   unknown filter or scope (21 GEC lookups; the search's scopes).
6. **Summary text standing in for the manual.** In the recorded runs, 10 of
   the 11 consults where the library returned wrong content got it from
   model-written chapter text under section numbers the manual does not have.
   The references rewritten from the print in October hold (5 of 80 sampled
   sections with a substantive error, none a wrong number); the 19 never
   checked do not (77 of 301, 26 %).
7. **"Not found" that means "the lookup failed"** (12 of 41 consults), and
   the call budget ending most consults (28 of 39 on the forced last call).

What held: no value a consult received from a lookup function has changed
since it was recorded, apart from UFC 3-250-01 Figure E-1 (fixed in GM-298
and GM-303); the Opus and Sonnet helpers relayed the library word for word,
so every wrong statement of theirs was the library's; EN 1997-1 Tables A.1 to
A.17 match cell by cell; UFC 3-301-01 Table 3-1's 92 rows match; unit
handling was right in every conversion traced.

## 4. What is being fixed, and where

| Branch | Repository | What | Backlog |
|---|---|---|---|
| `fix/review-dm71` | geotech-references | every DM 7.1 A finding, range refusals, chapter text | REF-65 |
| `fix/review-dm72` | geotech-references | DM 7.2's 19 equations and nine digitised objects from the print | REF-66 |
| `fix/review-ec7` | geotech-references | EN 1997, Caltrans, FEMA P-2082 (checked; two items sent back) | REF-67 |
| `fix/review-struct` | geotech-references | the structural packages, with every formula re-read on the rendered page | REF-68 |
| `fix/review-gec` | geotech-references | the eleven FHWA packages, partial keys and filters | REF-69 |
| `fix/review-pave` | geotech-references | the pavement family, backfill and expansive soils rebuilt from the held prints | REF-70 |
| `fix/review-core` | geotech-references | search, fetch and catalog rows | REF-71 |
| `feat/em-1905` | geotech-references | EM 1110-1-1905 (2025) as a new package (a build, not a fix) | REF-72 |
| `fix/refs-helper-results` | the app | what the helper sees and hands back | REF-73 |

Each library branch merges after the independent check against the print.
The app's side of the fixes (removed or renamed functions, the pavement
module's handling of a refused input, calls that now raise, the page
fingerprints) follows the merges (REF-74), then the release (INF-72).

## 5. The owner's decisions (2026-11-10, evening)

In the owner's words, with where each is filed:

- New references: "for those 'references to add next', don't build those
  yet. Just put them on a 'later' backlog item." Nothing is built until the
  owner says (OWN-135, with the lead's suggested order). EM 1110-1-1905 was asked
  for separately and its build continues (REF-72). The held documents the
  same search reported superseded or archived are to be verified (REF-90).
- DM 7.2's reissue: "Agreed with not rebuilding DM 7.2 (I believe the same
  redesignation happened for 7.1)." The lead's reading: the citation moves to
  UFS 3-220-20 (26 March 2026), saying the content is the 16 January 2025
  UFC's, page for page, and nothing is rebuilt (recorded as OWN-137; work
  REF-75). DM 7.1 is already
  cited as UFS 3-220-10 (16 January 2026) (REF-28, done).
- The three copyrighted PDFs tracked in the reference repository
  (`docs/aashto1993.pdf` and the two EN 1997 parts): "The copyright PDFs are
  OK in the repo. I downloaded these freely off the web; my understanding is
  they're on there because they're adopted law." No action (recorded as
  OWN-138 in the backlog's Done list).
- Release: "You can release when everything is merged and checked." The
  lead releases the library's next version and the app's pin once every fix
  branch and the app-side branch is merged, checked and green, then tells
  the owner (recorded as OWN-139; work INF-72).
- Equation judgment calls: "for all equation judgement calls, go with
  whatever you think is best, but file away a 'later' backlog item for me to
  revisit it in the future" (OWN-136, one answer per call from
  `findings/RULINGS.md`).

## 6. What is left open

- **Not started, changes what the helper returns:** the out-of-range pass
  over the 81 clamping functions (REF-76); US-unit inputs that take SI values
  silently (REF-77); what a result says, its unit and its page (REF-78); the
  DM 7 chapter-text equations (REF-79); GEC-11's text rebuilt (REF-80);
  GEC-10 chapters 6 and 8 rewritten (REF-81); the FHWA pavements text rebuilt
  (REF-82); an every-equation pass over the other never-checked references
  (REF-83).
- **Coverage:** GEC-5's 90 and 135 appendix figures (REF-84; GEC-13's are
  REF-08); worked-example appendices and the two UFCs with no text (REF-85);
  printed tables the helper cannot reach (REF-34); table and equation entries
  outside the search (REF-86).
- **Measurement and checks:** the helper's call budgets measured again after
  the fixes (REF-87); two library checks that pass on wrong code (REF-88);
  chart code readings that never give a value (VIS-10).
- **Tidy:** the empty package folders, NOAA in the app's strings and the
  unused airfield PDF (REF-89); the retired `agents/references_agent.py`
  (INF-06); held documents reported superseded or archived, to verify
  (REF-90).

## 7. Files here

`findings/`: the registers (`dm71.md`, `dm72.md`, `gec.md`, `pave.md`,
`ec7.md`, `struct.md`, `core.md`, `text.md`, `traces.md`), the fix tables
(`dm71_fixes.md`, `ec7_fixes.md`, `struct_fixes.md`), the independent check of
the ec7 branch (`ec7_verify.md`), the lead's rulings (`RULINGS.md`), the DM
7.2 edition comparison (`editions.md`), the coverage audit (`COVERAGE.md`)
and the new-reference candidates (`CANDIDATES.md`). They are copies as of
2026-11-10; paths under `C:/dev/tmp/` in them are local. One row of the
candidates list that concerned no public reference was left out.
