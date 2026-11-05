# REF-44, DOC-09 (refs part), INF-55 part 2: CHECK

Branch `claude/ref-44-hygiene`, 2026-11-05. Python 3.13, pytest 9 with
pytest-xdist, PyMuPDF installed (so the PDF-reading tests ran, not skipped).
No search or ranking code was touched (`_retrieval_db.py`, `_figures_db.py`,
`_query_expansion.py`, `_atomic_db.py` are unchanged).

## 1. REF-44: the stone column settlement improvement factor

**Finding.** `gec_13/figures.py` `figure_5_5_settlement_improvement` computes
SRF = 1 / (1 + a_s(n - 1)) and called it "GEC-13 Figure 5-5" and "Priebe
(1995)". The repo's GEC-13 text settles it:

- `gec_13/text/chapter05.json` 5.4.1.2.4 "Stress Ratio" (printed 4.1.2.4)
  gives Eq. 5-4, sigma_soil = q / [1 + (n - 1) alpha_sc], p. 5-32. SRF =
  sigma_soil / q is that relation, so the source is **Eq. 5-4, p. 5-32**, the
  stress reduction method (GEC-13 4.1.2.6, p. 5-36). I also read it on the PDF page
  (`docs/GEC 13 Vol 1.pdf`, printed p. 5-32: Eqs. 5-2 to 5-5).
- Figure 5-5 is "Typical vibrator cross-section" (5.3.1.1.3 Equipment, p. 5-14).
- The Priebe method is a curve (Figure 5-21, applied with Figure 5-27).
  5.4.1.2.6 says the chapter prints no closed-form settlement improvement
  equation. So the function is not Priebe's.
- The text gives the relation as an equation, not a figure. No figure prints it.

**Changed.**
- The docstring, section comment and result `description` now cite GEC-13
  Eq. 5-4, p. 5-32, and the stress reduction method. They no longer mention
  Figure 5-5 or Priebe.
- The function **name stays** `figure_5_5_settlement_improvement`, because the
  app's alias table (`funhouse_agent/adapters/_reference_common.py`) calls it
  by that name. Renaming it needs a matching app change.
- Two tests in `tests/test_gec13_figures.py`: the result matches Eq. 5-4, and
  the description cites Eq. 5-4 with no "Priebe" and no "Figure 5-5".
- CLAUDE.md and the docstring in `agents/gec13_agent.py` follow.

**Noted, not changed (same file, same kind of mislabel; for a new backlog
item).** Checked against the repo's GEC-13 text:
- `figure_4_3_suitability_number`: Figure 4-3 is "SPT liquefaction chart". The
  suitability number is **Eq. 4-3** (4.3.2.2, Brown 1977). The printed formula
  also has a 1/(D10)^2 term, but the function takes only D50 and D20.
- `figure_5_2_area_replacement_ratio`: Figure 5-2 is "Top feed
  vibro-replacement". The ratio is **Eq. 5-1** (5.4.1.2.2).
- `equation_8_1_groutability_ratio`: Eq. 8-1 is Darcy's law. N = D15/D85 is
  **Eq. 8-2** (8.2.3.2.1, p. 8-29).
- `equation_11_1_ltds`: the manual has 10 chapters, so no Eq. 11-1 exists.
  This is a leftover of the chapters 9-11 that REF-25 found were not the
  manual's.
- `equation_7_4_composite_modulus`: Eq. 7-4 was not found in the chapter 7
  text; it needs a reader with the PDF.

## 2. DOC-09, the refs part

### 2a. CLAUDE.md test counts (measured)

Each package's own test files were run with `pytest -q`. All passed.

| Package | Was | Measured |
|---|---|---|
| gec_6 | 121 | 121 |
| gec_7 | 101 | 101 |
| gec_10 | 195 | 247 |
| gec_11 | 130 | 135 |
| gec_12 | 147 | 231 |
| gec_13 | 168 | 197 (195 before this branch's 2 REF-44 tests) |
| micropile | 108 | 108 |
| fhwa_soils_foundations | 12 | 15 (3 of them skip without PyMuPDF and the PDFs) |
| ufc_backfill | 87 | 75 |
| ufc_expansive | 55 | 63 |
| ufc_pavement | - | 339 (337 before this branch's 2 title tests) |
| eurocode_7_1 | 134 | 134 |
| eurocode_7_2 | 133 | 143 |
| aashto_1993 | 308+ | 425 |
| em_2104 | 190+ | 111 |
| em_2107 | 103 | 103 |
| ufc_structural | 200 | 200 |

- The GEC-7, GEC-10, GEC-11, GEC-12, GEC-13 and micropile values match the
  app's DOC-09 CHECK.
- em_2104 went down: "190+" was never a count.
- The per-reference headings follow the table. The suite total is now **6,540
  collected** (was 6,256).
- A line under the table's heading says how the column is counted.
- Function counts were not re-measured.
- Not done: `docs/agent_showcase.html` has the same old figures (the app's
  CHECK names it). It was not in this brief.

### 2b. `agents/`' old catalog

**Finding.** The catalog is **in use**. It is the Foundry dispatcher
(README "Agents"), and `tests/test_cover_citations.py` reads it. The app's
retrieval dev guide describes it. The hand-written `_REFERENCE_CATALOG` listed
12 of the package's 33 references and had drifted:
- GEC-10 was cited by the 2010 edition (FHWA-NHI-10-016), but the package is
  the 2018 edition.
- GEC-11 and UFC 3-250-01 were marked as having no text.
- (The old catalog already gave UFC 3-250-01 its printed title; what changed
  is that the catalog is now read from the package, so it cannot drift.)

**Changed.** `_REFERENCE_CATALOG` is now built at import from the installed
package:
- every subpackage;
- its public modules;
- whether it ships `text/*.json`;
- its citation: `reference_id` + `reference_title` from its chapter files, or
  the first line of its docstring when it has no text.

All 33 load, and the 31 with text answer `list_chapters`. The function
docstrings that listed 12 references now point to `list_reference_methods('')`.

New test `tests/test_references_agent.py`: the catalog equals the package,
each entry matches its package and loads, and the key citations are right.
`test_cover_citations` reads the agent's citation from the catalog instead of
by regex over the source. GEC-10 stays a strict xfail, because the manifest
and figure catalog still cite FHWA-NHI-10-016. README and CLAUDE.md say so.

The 19 per-reference `*_agent.py` wrappers were left as they are, apart from
the GEC-13 and UFC pavement docstring lines.

### 2c. The UFC 3-250-01 title

**Finding.** The chapter text files hold no cover. The repo's copy of the
document, `docs/ufc_3_250_01_2016.pdf`, prints **"PAVEMENT DESIGN FOR ROADS
AND PARKING AREAS"**:
- on its cover (page 1);
- on its revision summary sheet ("Document: UFC 3-250-01, PAVEMENT DESIGN FOR
  ROADS AND PARKING AREAS").

Its 1-1 scope reads "the pavement design of roads and parking areas". "Roads,
Streets, Walks, and Storage Areas" is printed nowhere in it. Other repo texts
that cite this UFC (ufc_concrete_practice ch. 18, ufc_stabilization ch. 1)
also print "Roads and Parking Areas".

**Changed.** These now use "Pavement Design for Roads and Parking Areas":
- the manifest, the figure catalog and all 21 chapter files (`reference_title`);
- the `ufc_pavement` package, equations and tables docstrings;
- `agents/ufc_pavement_agent.py`, the test docstring, README and CLAUDE.md.

The "covers roads, streets, walks, and open storage areas" lines now say
"roads and parking areas". Two tests: every data file carries the printed
title, and the title is on the cover (skips without the PDF).

## 3. INF-55 part 2: sqlite errors under pytest-xdist

**Finding.**
- INF-55 was measured on 2026-10-26 (refs at about ceeef53). REF-56, merged
  today, changed how the databases are built.
- Before REF-56, `rebuild_db()` deleted the one shared temp-folder database
  and rebuilt it in place.
- Module fixtures in `test_retrieval_db.py`, `test_query_expansion.py` and
  `test_figures_db.py` call `rebuild_db()`. Under `-n 4`, every worker deleted
  and rewrote that file in every module while other workers were writing or
  reading it.

Reproduced on the commit before REF-56 (424d5ae~1), `pytest -n 4`, exit 1
both times:
- cold temp folder: 2 failed and 3 errors ("disk I/O error", "unable to open
  database file");
- stale sources: 17 errors ("table sections already exists").

Neither run showed the "readonly database" wording. Which message appears
depends on timing; all are the same in-place race. The 79 "readonly database"
errors themselves were not reproduced here.

On this branch the suite was already clean under `-n 4` before any change.
REF-56's build-then-`os.replace` is what fixed it.

**Changed (hardening on top of REF-56, tests only):**
- The three fixtures call `_ensure_db()`, which builds only when the file is
  missing or stale. Workers no longer rewrite the shared file in every module.
- `test_subsection_retrieval` opens the shared database read-only.
- New `tests/test_shared_db_hygiene.py` fails if any test module except
  `test_atomic_db_build.py` calls `rebuild_db()` / `_build_db()` on the shared
  database again.
- New `tests/conftest.py` `pytest_configure` (review finding 4): calls
  `rebuild_db()` for the text and figure databases once per run, in the
  controlling process only (not in xdist workers, which have `workerinput`),
  so every session starts on databases built by the current indexing code.
  Staleness is otherwise detected only by a missing file, an older mtime, or a
  text-database schema mismatch, so an indexing-code change without a schema
  bump would be served from the old file. Cost: about 0.75 s per run (0.65 s
  text, 0.10 s figures).

## Test results (final tree, judged by pytest's exit code)

| Run | Result | Exit |
|---|---|---|
| `pytest -q` (serial) | 6538 passed, 2 xfailed | 0 |
| `pytest -q -n 4`, default temp folder | 6538 passed, 2 xfailed | 0 |
| `pytest -q -n 4`, fresh empty TMPDIR (cold build) | 6538 passed, 2 xfailed | 0 |
| `pytest -q -n 4`, sources touched newer than the database (stale rebuild) | 6538 passed, 2 xfailed | 0 |

No sqlite error of any kind appears in any of the four logs. The 2 xfails are
the strict GEC-10 and UFC backfill cover-citation marks in
`test_cover_citations.py`.
