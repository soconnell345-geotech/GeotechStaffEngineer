# DOC-10: user manual rebuild (CHECK)

Branch `claude/doc-10-manual-rebuild`, from master `d45380a` (version 5.34.0).
BACKLOG.md, HANDOFF.md and CLAUDE.md are not edited here; the lead writes the
notes at merge.

## What was rebuilt

| Step | Command (from `docs/user_manual/`) | Result |
|---|---|---|
| Worked examples | `python gen_worked_examples.py` | 18/18 calls ran. The output matched the committed `worked_examples.json` (last regenerated in 3cba0f6, GM-299) except for float noise in the 12th significant figure of lateral_pile, so the committed file was kept as it was. |
| HTML | `python build_manual.py` | `manual.html` + `toc.json` regenerated (never hand-edited). The build is deterministic: two runs give the same bytes. |
| PDF | `python render_pdf.py` | Builds on Linux with `/usr/bin/google-chrome`: 249 pages, 0 blank, 53 outline entries, cover "Version 5.34.0". **Not committed**: see "PDF" below. |

Environment: Python 3.11 venv, `pip install -e . -e geotech-references` (submodule
at 32bbd77), numpy 2.4.6.

## What the manual now shows

- **Version.** The cover and `<title>` say 5.34.0 (they said 5.33.0). The
  version is read from `pyproject.toml` by the build.
- **Every tool in the catalog.** All 70 modules in `MODULE_REGISTRY` are named;
  every one of the 195 non-alias methods of the analysis modules has its table,
  and every one of their 2,600 parameters (2,597 before the merge) is listed. Reference modules are listed
  by method count (as the build has always done; their 1,341 lookups are
  discoverable at run time). Checked by `module_work/doc10/check_manual.py`
  (exit 0; rerun it after any rebuild).
- **Defaults.** New in the build (`build_manual.py`, `_default_text`): every
  optional parameter with a registered default now shows `default: <value>` in
  its table, taken from the adapter's `METHOD_INFO` (1,087 such parameters on
  the analysis modules after the merge, 1,085 before; empty-string defaults are not shown). Before, a default
  appeared only where the description's prose happened to state it.
- **Anchor inputs and the test load.** `soe.design_ground_anchor` (14
  parameters, including `test_load_factor` with its 0.80 SMTS rule) and
  `soe.fhwa_apparent_pressure_anchored_wall` (10) are listed in full as the code
  has them. The "fixed 1.50 test load" DOC-19 named is gone: the tables show the
  input with `default: 1.33`; 1.50 appears only as one of the values the tool
  text quotes from GEC-4 (1.20 to 1.50).
- **Drilled-shaft defaults (GM-244, GM-284).** The parameter tables were already
  current (they come from the registry). The hand-written practitioner prose in
  `manual_content.py` was not, and was rewritten from the tool's own text:
  - clay side: GEC-10 (2018) Eq. 10-18 rational alpha by default (it said
    "AASHTO 0.55 default"); the AASHTO alpha by name;
  - sand base: Eq. 10-14 on the mean N60 over two diameters below the base (it
    said "0.60 N60 with the large-diameter reduction");
  - clay base N*c su (Eq. 10-19), su averaged over two diameters;
  - rock side Eq. 10-21 with C = 1.0 by default, 0.65 and 1.9 by name, Eq. 10-22
    for caving sockets, qu held to f'c when `concrete_fc` is given; rock base
    2.5 qu (Eq. 10-23) only with the p. 10-22 conditions stated or RQD 100 below
    the base, else the Hoek-Brown base or a refusal;
  - top exclusion 1.5 m in clay by default, none in sand or rock;
  - the "limits" bullet said the large-base reduction is applied by default; it
    is off by default (`large_base_reduction`), and a second bullet says qu is not
    checked against f'c without `concrete_fc`.
- **Changes picked up from master since the last rebuild:** wave_equation's
  `single_blow`, `bearing_graph`, `drivability` and `wave_equation_package`
  each show one more parameter;
  `write_diggs` names `not_measured` groundwater (RI-106); documented parameters
  5,764 -> 5,767.
- **Other prose fixed in `manual_content.py`:**
  - the subsurface entry said python-ags4 is "an optional extra from the release
    after 5.33.0"; it now says `[ags4]` since 5.34.0;
  - the retaining-wall example note quoted sliding FOS 1.27; the script's
    output is 1.143, so it now says 1.14;
  - the slope-search example note quoted 1,418 trial surfaces; the output is
    2,300 (`n_surfaces_evaluated`), so it now says 2,300;
  - Appendix B (version history) stopped at 5.3.0. One honest row was added
    for 5.4.0-5.34.0 pointing to HANDOFF.md and CLAUDE.md for the release notes,
    and the lead line no longer says V5.1_SUMMARY tracks the whole 5.x line.
    No history was written for the 30 releases; that is left (below).

## Review fixes and merge with master (2026-11-05)

Review: `REVIEW_DOC-10.md` (verdict: merge after fixes). Each sentence was checked
against the drilled_shaft adapter and `pile_group` before it was written; the code
and the review agreed in all four cases.

- Pile-group limits: the "No group uplift method" bullet is removed;
  `pile_group.group_uplift` (GEC-12 Sec. 7.2.3.2, since GM-289) is offered.
- Rock base: a rock layer given no RQD is taken at RQD 100, so a profile without
  RQD gets 2.5 qu (Eq. 10-23); the result's choices say so.
- Rock side: `rock_side_C` is a number (default 1.0). GEC-10 prints 0.65
  (Eq. 10-22 at alpha_E = 1) and 1.9 (roughened sockets); a C above 1.0 is to be
  verified by load tests or local experience (p. 10-22).
- Clay side: by default (`su_test_type="ciuc"`) cu is taken as su(CIUC),
  untransformed; the UU/UC to CIUC transform (Eqs. 10-16/10-17) applies only when
  `su_test_type` says so.

Master was merged into the branch (merge commit e8127dc). `manual.html` conflicted
(master's hand re-render of GM-305); it was taken from one side and rebuilt, not
hand-merged. The rebuild picked up GM-305 (rigid-footing rows: `footing`, `base`,
`settlement_max`) and GM-237 (driven-pile layer keys `alpha`, `adhesion`,
`adhesion_basis`, `highly_overconsolidated`). Parameters now documented: 2,600
(1,087 registered defaults, 195 methods, 70 modules; `check_manual.py` exit 0).
A second `build_manual.py` run leaves `manual.html` and `toc.json` unchanged.
The strong release check is: rebuild, then `git diff --exit-code
docs/user_manual/manual.html`.

## The SOE example

The manual's worked examples come from a script (`gen_worked_examples.py`), not
from hand-written numbers. The SOE call (`soe.braced_excavation`, 8 m cut, two
struts) was re-run: its output is the same as the committed JSON, so the HTML
quotes current numbers. The "old SOE example" DOC-10 named is the one in the
committed **v5.3 PDF**, which was built in July from the old code; it goes away
when the PDF is replaced.

## PDF

`render_pdf.py` was Windows-only (Chrome/Edge paths under `C:\Program Files`)
and wrote a fixed `..._v5.3.pdf` name. It now:

- also looks for `/usr/bin/google-chrome`, Chromium, `/opt/pw-browsers/chromium`
  and the same names on `PATH`, and passes `--no-sandbox`;
- names the output from the `pyproject.toml` version
  (`docs/GeotechStaffEngineer_User_Manual_v5.34.0.pdf`).

The Linux build is **not committed**, for two reasons the lead should weigh:

1. **Size.** It is about 15 MB (the v5.3 PDF is 5.5 MB), because the
   stylesheet asks for Georgia, Segoe UI and Consolas, which Linux does not have;
   Chrome substitutes Liberation/DejaVu and embeds about 3,000 Type 3 font
   pieces. `garbage=4, deflate=True` saves only 0.6 MB. A 15 MB binary per
   release in a public repo's history is a cost that cannot be taken back.
2. **Length.** 249 pages against 132 for v5.3: the HTML is now 2.0 MB (every
   parameter's full tool text, plus the new default lines), not just the fonts.

What it needs: run the three steps on the Windows laptop (Chrome or Edge,
PyMuPDF from the core install), which has the fonts the stylesheet names. A
copy of the Linux build is in this session's scratchpad for a look, not for
shipping. Then decide whether the old `docs/GeotechStaffEngineer_User_Manual_v5.3.pdf`
is removed or kept beside the new one; CLAUDE.md (line ~1975), HANDOFF.md (lines
~3045, ~3182) and `module_work/V5.4_PLAN.md` still name the v5.3 file. If 249
pages is too long, the bulk is Appendix A (every method's full tool text); the
20 headline methods' tables also appear twice, in Chapter 4 and Appendix A.

## `docs/V5.1_SUMMARY.html`

Nothing links to it as a hyperlink: no `href` in any HTML file, nothing in the
package data or the wheel (`docs*` is excluded in `pyproject.toml`). It is named
as text in CLAUDE.md (~2010), HANDOFF.md (~2989, ~3195),
`HANDOFF_2026-06-14.md`, `module_work/WEEKEND_QC_2026-06-13.md`,
`module_work/V5.4_PLAN.md`, BACKLOG DOC-10, the 2026-11-04 triage JSON, and the
manual's Appendix B lead (now "covers 5.0 to 5.1 only"). Not deleted.

## Tests

- `webapp/tests/test_docs_currency.py` + `webapp/tests/test_public_text.py`:
  10 passed, pytest exit 0 (before and after this CHECK).
- No manual test exists in the repo (nothing under `*/tests/` imports
  `build_manual` or reads `manual.html`). `module_work/doc10/check_manual.py` is
  the check used here; it is not a pytest test, because it would go red on every
  tool change until the manual is rebuilt. Whether to make it one is the lead's
  call.

## Left

- The distributed PDF, built on Windows (above), and what to do with the v5.3 file.
- Appendix B: one row per release since 5.3.0 (the release notes are in HANDOFF).
- The rest of the hand-written practitioner prose in `manual_content.py` was not
  audited module by module against the 5.34.0 changes (GM-281 to GM-303 moved
  many methods; e.g. Coulomb delta = beta and bearing on B' for retaining walls,
  Converse-Labarre removed from pile groups). Only the drilled-shaft entry
  (GM-244, named in DOC-10) and the stale lines above were fixed. The parameter
  tables are current by construction.
- The axial_pile limits bullet on Tomlinson alpha (`manual_content.py` ~435, "alpha
  0.42 vs DrivenPiles 0.76 at su 1.9 ksf") is unverified; it predates the
  GM-121/GM-144/GM-282 adhesion-chart work.
- The validation table (Chapter 8) is hand-written; V-008's title "Base
  resistance in sand (0.60 N60)" and `validation_examples/RESULTS.md` V-008b
  ("1.27/Db applied") describe the large-base reduction as applied, which has
  been off by default since GM-244.
- The deferred webapp-guide screenshots DOC-10 lists were not touched.
- The README in `docs/user_manual/` now says how the PDF is named and what the
  Linux build looks like.
