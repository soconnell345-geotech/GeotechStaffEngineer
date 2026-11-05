# GM-223 and GM-225: the citation guard's loose ends, and one prompt claim

Branch `claude/gm-223-225-citations`. Citations and prompt wording only:
**no formula, default or calculated value changes** (shown under "Numbers
unchanged" below). No release note.

## GM-223 (1): bare numbers now checked against a file default manual

**The guard** is `funhouse_agent/tests/test_citations.py` (GM-178). It takes
each Eq./Table/Fig. number to belong to the last manual named before it in
the same string. A number with no manual named before it was put on an
"unparsed" list and never checked.

**What changed.** The guard now has `DEFAULT_MANUALS`, keyed by file path.
In a file listed there, a bare number at the start of a string, or at the
start of a sentence, is checked against the file's own manual:

| Files | Default manual (the one the module follows) |
|---|---|
| `axial_pile/` | GEC-12 (FHWA-NHI-16-009) |
| `drilled_shaft/` | GEC-10 (FHWA-NHI-18-024) |
| `retaining_walls/mse.py` | GEC-11 (FHWA-NHI-10-024) |
| `ground_improvement/` | GEC-13 (FHWA-NHI-16-027) |

Naming another source ends the default until the next sentence (Tomlinson,
API, NHI-06-089, EN 1997 and so on, from the guard's own `OTHER_SOURCES`
list). The report labels these citations "GEC-12 (file default)" and so on.

- **Before:** 1,601 citations checked; 969 numbers following no manual.
- **After:** 1,834 citations checked (233 more); 736 numbers following no manual.

**Failures.** None of the 233 newly checked numbers is missing from its
manual. Two land on numbers already listed in `REFERENCE_GAPS`, which were
checked against the print at GM-178/GM-241/GM-302: GEC-10 Eq. A-2 and
GEC-11 Eq. 4-22. So there was nothing to fix, and no citation needed marking
as unsettled.

**Files left without a default, on purpose.** A file that cites several
manuals side by side would have its bare numbers given to the wrong one:

- `soe/` (GEC-4, Caltrans T&S, UFC 3-220-20);
- `sheet_pile/` (EM 1110-2-2504, UFC, Caltrans);
- `downdrag/` (UFC 3-220-20, GEC-12, GEC-10, CGPR);
- `bearing_capacity/` (UFC 3-220-20, GEC-6);
- `retaining_walls/cantilever.py` and `calc_steps.py` (GEC-6, UFC, NHI-06-089, EN 1997-1);
- `pavement_design/` (AASHTO 1993 and UFC 3-250-01);
- `lateral_pile/` (a trial default put "GEC 9 App. A.4" numbers A-23, A-24, A-43, A-44 under GEC-9; the library lacks them and checking them against the print belongs with REF-36);
- the adapters, which mix sources.

## GM-223 (2): drilled shaft, "one diameter B"

GEC-10 (FHWA-NHI-18-024, 2018), Sec. 10.3.5, p. 10-9 (PDF p. 289), prints
Eq. 10-3, R_SN = pi B dz_i f_SN, "where: B = shaft diameter". Eq. 10-21
(p. 10-20) is the rock-socket unit side resistance f_SN/p_a = C sqrt(q_u/p_a),
which has no B in it.

The citation "Eq. 10-21" became "Eq. 10-3, p. 10-9" in four places:

- `drilled_shaft/shaft.py` (the `socket_diameter` docstring);
- `drilled_shaft/DESIGN.md`;
- the `socket_diameter` description in `funhouse_agent/adapters/drilled_shaft.py`;
- the matching rows of `docs/user_manual/manual.html`.

## GEC-12 page citations: the volume added

GEC-12 (FHWA-NHI-16-009, 2016) numbers its pages separately in each volume:

- **Vol I:** chapters 1-8, printed pp. 1-523.
- **Vol II:** chapters 9-24, pp. 1-547.
- **Vol III:** the appendices, pp. 1-271.

So a bare "p. 247" is ambiguous. Every GEC-12 citation by page in shipped
code (every string literal outside tests, docs, `slope_stability/` and
`foundry/`) now names the volume in front of the first page cited after
"GEC-12", for example "GEC-12 Vol I p. 407" or "GEC-12 Table 7-1 (Vol I
p. 224)". Later pages in the same clause are the same volume. That is 175
citations in 18 files.

**How the volume was settled, from the print.** Each volume PDF in
`geotech-references/docs/` was mapped from PDF page to printed page. Every
cited page was then read in both volumes:

- **The chapter 5-7 pages are all Vol I** (pp. 108-129, 224-408). Each was
  checked on the page, for example:
  - p. 247: the corrugated curve for H-piles;
  - p. 407: Eq. 7-90;
  - p. 291: Fig. 7-33;
  - p. 379: "Straight line interpolation";
  - p. 108: Eq. 5-1/5-3;
  - p. 367: Table 7-23.
- **The wave-equation pages are all Vol II** (chapter 12, pp. 158-226), for
  example:
  - p. 158: the permanent set by a weighted average of the quakes;
  - p. 182: "12.5.6 Example 6";
  - p. 226: the hammer cushion window, "Ideally, the contractor would provide...".
- **Pages left alone**, because they are UFC 3-220-20 pages: downdrag's
  "pp. 415-417", and pile_group's "p. 433", "p. 435", "p. 465" and "p. 641".
  Each follows a UFC name in its own clause.

**Tests that matched the old wording** were updated to the new wording only:

- `axial_pile/tests/test_choices.py`;
- `downdrag/tests/test_alpha_default.py`;
- `downdrag/tests/test_downdrag.py`;
- `downdrag/tests/test_toe_and_beta.py`.

**User manual.** The tool descriptions that changed were hand-edited in
`docs/user_manual/manual.html`: axial_pile, downdrag, pile_group,
wave_equation and drilled_shaft. There was no full rebuild.

## GM-225: the deep prompt's claim about html_to_pdf

`funhouse_agent/deep/prompt.py` said `html_to_pdf` "rejects" a coloured
table standing in for a figure. It makes no such check. What it does is
below (`funhouse_agent/adapters/calc_package.py`, `_embed_local_images` and
the render, and `calc_package/latex_renderer.py`, `render_html_pdf_story`):

- it refuses an "[image]"-style placeholder and an inline `<svg>` (unless
  `allow_unembeddable`);
- it draws no backgrounds since GM-219;
- its result's `fills_not_drawn` counts the fills left out.

The prompt now says exactly that. The same false claim was in the calc
sub-agent's figure framing (`funhouse_agent/deep/agent.py`), and it got the
same correction. The check itself was not built (the task said not to).

The page fingerprints were re-recorded in a commit of their own.

## Numbers unchanged

Every dispatch call in `funhouse_agent/worked_examples.json` was run on
master (1d39240) and on this branch (28 worked examples: axial pile,
drilled shaft, MSE, sheet pile, SOE, ground improvement, slopes,
pavements). **1,899 numeric leaves on each side; 0 keys differ, 0 values
differ.**

The calc_package tests: **264 passed / 3 skipped on master, and 264 passed
/ 3 skipped on the branch** (the 3 skips: pdflatex not installed).

## Tests (pytest exit codes, Python 3.12 venv)

- axial_pile, drilled_shaft, pile_group, downdrag, wave_equation,
  geotech_common, seismic_geotech, ground_improvement, retaining_walls,
  funhouse_agent/deep/tests, test_citations, test_reviewers,
  test_page_fingerprints, test_public_text: **3,603 passed, 4 skipped, 0
  failed** (exit 0), on the final tree.
- calc_package: 264 passed, 3 skipped (same on master).
- funhouse_agent/tests + validation_examples (run before the reviewer
  checklist fix): 4,519 passed, 30 skipped, 3 failed. One was the
  foundations reviewer's pasted checklist (fixed, test_reviewers now
  passes). The other two fail identically on master 1d39240 in this
  environment, so they are not this branch's:
  `test_calc_package_figure_conventions.py::test_large_pile_group_leaves_the_forces_to_the_table`
  and `test_defaults_used.py::...[profile_figure|plot_data]` (logx/logy
  read as inert).

## Noted, not fixed

- 736 numbers still follow no manual. Most are in files that cite several
  manuals (left without a default, above) or follow an author's name that
  the guard reads as another source. For example, "Hoek-Brown" and
  "Tomlinson's Eq. 7-90" are GEC-12/GEC-10 numbers, but "Brown" and
  "Tomlinson" are in `OTHER_SOURCES`.
- GEC-12 page citations in DESIGN.md files and docs (not shipped strings) were not given volumes.
- GEC-12 page citations in strings that do not name GEC-12 (for example a
  bare "Step 1 (p. 250)" in its own axial_pile string) carry no volume.
