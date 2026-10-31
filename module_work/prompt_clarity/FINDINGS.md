# GM-230: the GSE page's agent-facing text, read as a geotechnical engineer

2026-10-30. Branch `wip/gm230` from master 15fadcb: six commits, 9dc6948 to
924db45, then five after an independent review (747dd02, dd79003, d1c30b5,
aa4d393, and the fingerprints in 0d9c015); not merged. Each finding is listed
in BACKLOG GM-230; this file keeps the detail.

## What was read

- The tool descriptions every GSE-page module gives the agent (method,
  parameter and result descriptions in each adapter's METHOD_INFO), dumped
  and de-duplicated.
- The `choices` each calculation module returns (what was chosen, how it was
  set, its basis and its alternatives), from runs on made-up cases: clay-only,
  sand-only and mixed pile and shaft profiles, rock sockets, phi = 0 and c = 0
  footings, and runs with every default left out.
- The `defaults_used` list each of those runs returns.
- The page's system prompt (`funhouse_agent/system_prompt.py`), the deep
  agent's prompt (`deep/prompt.py`), its helpers' prompts (`deep/agent.py`,
  `reviewer.py`) and the four specialist reviewers' checklists
  (`review_checklists.py`, with their `.claude/agents/*-reviewer.md` copies).

Out of scope, as briefed: the report-ingest and Document Review prompts, all
slope_stability text (GM-140), and anything a tool computes. No commit changes
a number a tool returns. Every section, equation, table and page the new text
names was already printed in the same module (its DESIGN.md, calc steps or
another basis string) or in the reference module the checklist points to;
none was added from memory. One was copied from a wrong cite already in the
module (Table 5-7's d_c, item 10 under wrong values); the independent review
caught it, and it is now checked against the printed page.

## What was found, by kind

| Kind | Items | Fixed |
|---|---|---|
| Text about a case the run did not have | 7 | 7 |
| Wrong method, value, unit or name | 10 | 10 |
| Backlog IDs, design IDs and change history in agent-facing text | 15 | 15 |
| Prompt statements that were untrue, or citations from memory | 8 | 8 |
| Coined terms | 2 | 2 |
| A method named loosely | 1 | 1 |
| A symbol with two meanings | 1 | 1 |
| A default with no reference, not said so | 1 | 1 |
| `defaults_used` listing inputs the run did not use | 3 methods | 3 |
| Noted, not fixed (no change to a saved report or a live run's output) | 10 | 0 |

### Text about a case the run did not have (7)

1. axial_pile, side-resistance basis: named both the Nordlund and the alpha
   method in every run. It now names only the methods the profile ran.
2. axial_pile, calculation-package method notes: named the Nordlund charts
   (K_delta, delta/phi, C_F) in clay-only runs and the adhesion chart in
   sand-only runs. They now name only the charts of the methods that ran.
3. bearing_capacity, phi = 0: the N-gamma choice offered alternative formulas.
   Every N-gamma formula gives 0 at phi = 0, so it now reads "0 at phi = 0"
   with no alternatives.
4. bearing_capacity, phi = 0: the depth-factor basis gave the phi > 0 forms.
   It now gives Table 5-7's d_c (the cite is item 10 under wrong values) and
   d_q = d_gamma = 1.
5. bearing_capacity, c = 0: the depth basis and its alternative discussed
   d_c, which does not enter. It now gives d_q and d_gamma only and drops the
   printed-d_c alternative.
6. drilled_shaft, top exclusion: the basis cited GEC-10's cohesionless-soil
   and rock sections in clay-only profiles. It is now built from the kinds of
   ground the profile has (`top_exclusion_basis`).
7. drilled_shaft, rock-base conditions: the choice repeated the whole
   rock-base basis (method, Hoek-Brown for fractured rock). It now gives the
   three Eq. 10-23 conditions and that they are the engineer's judgment
   (`ROCK_CONDITIONS_BASIS`); the base choice keeps the full basis.

### Wrong method, value, unit or name (10)

1. axial_pile: the toe alternative read "Nt sigma'v (method 'beta')"; the
   beta method's toe term is Nt sigma'p, the effective overburden at the toe.
2. axial_pile: the alpha-method clause named Tomlinson's Fig. 7-17 whatever
   adhesion chart was chosen. It now cites Sec. 7.2.1.3.2, fs = Ca (Eq. 7-10),
   and points to the 'adhesion chart (alpha-method)' choice.
3. soe: the log-spiral passive alternative said "wall friction left out".
   With passive_wall_friction_deg left out the module takes delta/phi' 0.66,
   none with cohesion (its own source line); the text now says so.
4. drilled_shaft adapter: bottom_exclusion's "5 ft (1.524)" had no unit; now
   "(1.524 m)".
5. Foundations checklist: "a critical-depth limit applies in sand". The
   module applies the Nordlund toe's limit, sigma'p at most 3 ksf (GEC-12
   Sec. 7.2.1.3.1, Step 8, p. 244, as axial_pile/DESIGN.md records), and no
   limit on side resistance.
6. Earth-retention checklist: the log-spiral passive Kp was named
   Caquot-Kerisel. The module's default is UFC 3-220-20 Eq. 4-12, with the
   Caquot-Kerisel chart as an option; at delta = 0 it must equal Rankine.
7. Earth-retention checklist: "the seismic thrust INCREMENT Delta KAE acts
   near 0.6H while the M-O TOTAL thrust acts near H/3". Now: Delta PAE at 0.6H
   above the base (Seed & Whitman 1970) and the static PA at H/3, as
   seismic_geotech's calc steps print.
8. Slope-FEM checklist: the mesh study was named `srm_mesh_refinement_study`;
   the agent calls it `fem2d_srm_mesh_refinement_study` (GM-280).
9. Earth-retention and slope-FEM checklists: the dm7 chapter anchors did not
   match the chapters the dm7 module holds; they now name DM7.2 Ch 2 and Ch 4
   (with Figs 4-12, 4-13) and DM7.1 Ch 7.
10. bearing_capacity, depth basis: "Table 5-7's 1 + 0.4 k" (in the phi > 0
    basis, already there) and "at phi = 0, d_c = 1 + 0.4 k, Table 5-7"
    (written in 9dc6948, caught by the independent review). Table 5-7 (p.
    295) prints d_c = 0.4 k, added in Eq. 5-21's 1 + s_c + d_c - i_c - b_c -
    g_c; the module multiplies by 1 + 0.4 k. Both bases, the depth_factors
    docstring and DESIGN.md now say so (747dd02).

### Backlog IDs, design IDs and change history (15)

- drilled_shaft adapter, six parameter descriptions: "the default until
  GM-241", "the default until GM-244" (twice), "the method until GM-244"
  (twice), "Applied by default until GM-244". The clause is gone: each
  description already says what the option does (large_base_reduction adds
  "true applies it"; d1c30b5). The first pass had put "the tool's earlier
  default" in their place, still history.
- retaining_walls: the refused k0_sloping_ground form said "... was removed
  (GM-243)"; now "tool has no other form".
- fem2d adapter: "Since 2026-10-09 the staggered scheme follows the reference
  too (it used to include self-weight)"; now "..., in both schemes".
- Checklists: "(DESIGN SG-3)"; "vs the 5.0 line" four times (the +Mx sense, a
  default that changed, the factor-of-safety basis, the battered-wall KPE);
  "(a v5.3 fix)"; "Since the 2026-10-14 fixes". Each now states the module's
  convention as it is.

### Prompt statements that were untrue, or citations from memory (8)

1. System prompt, CRITICAL section: told the agent to run "a reference
   module's own equation, table or chart function" itself. The deep agent has
   no reference modules; it now names the `references` helper (aa4d393).
   With that helper off, as for a specialist reviewer that holds its
   reference modules, the deep agent swaps the clause for one without the
   helper, so the prompt never names a helper it does not have.
2. System prompt: example citations "DM7.1 Figure 5", "GEC-10 Section 13.3"
   were written from memory. Now a format-only line: author and year, or the
   manual and the section, table or figure a tool returned.
3. System prompt: "unit weights between 14-24 kN/m^3" was a range from memory.
   Replaced by the GM-270 line: "A typical range you state comes from a
   reference looked up and cited; otherwise leave it out." (GM-270's
   measurement is still open.)
4. Reviewer rule 3: example citations "DM7.1 Table 3", "GEC-10 Section 13.3"
   from memory; removed.
5. The deep references helper's system prompt ended with a "Question:" label
   followed by the concision rule, not a question. `CONSULTANT_ROLE` has no
   label; v1 keeps `CONSULTANT_FRAMING = CONSULTANT_ROLE + "Question: "`.
   A stale comment that still called CONSULTANT_FRAMING the librarian's
   prompt is gone (aa4d393).
6. The same prompt listed NOAA, which the reference library does not hold,
   and not Eurocode 7, which it does. Now "UFC, Eurocode 7, micropile, FEMA
   and the others that list_agents names".
7. render_tables was missing from the deep prompt, the calc helper's preamble
   and the specialist deliverables, though the agent has it.
8. A checklist pointed to "the WORKFLOW note above", which is not there; now
   "reference content not in the reference modules above".

### Coined terms (2)

1. downdrag, two bases: "its headline procedure is UFC 3-220-20's"; now "as
   the module follows UFC 3-220-20's procedure by default". The downdrag
   adapter's tool docs said "the headline method" or "headline numbers" in
   13 places; they now say "the reported method" (d1c30b5).
2. Pavement checklist: "direct/MR-19.4/composite-k"; now "k from a direct
   value, MR/19.4 or the composite-k worksheet".

### A method named loosely (1)

- axial_pile adapter, `method`: "auto = Nordlund (sand) / Tomlinson (clay)".
  Now names the Nordlund method (GEC-12 Sec. 7.2.1.3.1), the alpha method
  with Tomlinson's adhesion (Sec. 7.2.1.3.2) and the beta method (Sec.
  7.2.1.3.4), as the module's DESIGN.md and calc steps cite them.

### A symbol with two meanings (1)

- Earth-retention checklist item 8: "Battered walls use Coulomb alpha = 90
  deg + beta", where item 2's beta is the backfill slope. It now names the
  wall batter (seismic_geotech's `beta`, from vertical) and says the
  backfill slope is its `i`, in review_checklists.py and the agent .md
  (dd79003).

### A default with no reference, not said so (1)

- ground_improvement, aggregate piers: E_s defaults to 5000 kPa with no
  reference behind it. When E_soil is 5000 the basis now adds "5000 kPa is
  the tool's default, not a value from a reference: give E_soil for the
  soil".

### defaults_used listing inputs the run did not use (3 methods)

- axial_pile: `displaced_volume` (Nordlund's V) was listed in clay-only runs.
  It is now listed only where a sand layer's choices report it.
- drilled_shaft `drilled_shaft_capacity` and `lrfd_capacity`: the side
  settings by soil (beta_method, K_over_Ko, alpha_method, su_test_type,
  rock_side_C, fractured_rock_thickness) were listed in runs without that
  soil. They are now listed where the run's choices report them.
  `capacity_vs_depth` gives no choices, so it keeps them.

## Before and after

1. drilled_shaft adapter, alpha_method:
   - Before: "...; GEC-10 recommends Eq. 10-18 'as a revision' to it; the
     default until GM-244."
   - After: "...; GEC-10 recommends Eq. 10-18 'as a revision' to it."
2. axial_pile side-resistance basis, clay-only run:
   - Before: "'auto' by soil type: Nordlund in cohesionless soil (GEC-12 Sec.
     7.2.1.3.1, Eq. 7-8: K_delta from Tables 7-6/7-7, C_F from Fig. 7-14, no
     limit on sigma'd or the unit side resistance), the alpha-method with
     Tomlinson's adhesion (Fig. 7-17) in cohesive soil (Sec. 7.2.1.3.2)"
   - After: "'auto' by soil type: the alpha method in cohesive soil (GEC-12
     Sec. 7.2.1.3.2: fs = Ca, Eq. 7-10, Ca from the chart in 'adhesion chart
     (alpha-method)')"
3. soe passive alternative:
   - Before: "... 3-220-20 Eq. 4-12, wall friction left out"
   - After: "... 3-220-20 Eq. 4-12, with passive_wall_friction_deg left out:
     delta/phi' 0.66, none with cohesion"
4. System prompt:
   - Before: "verify that input parameters are physically reasonable (e.g.,
     friction angles within expected ranges for the soil type, unit weights
     between 14-24 kN/m^3, SPT N-values consistent with described soil
     behavior)"
   - After: "Check that the inputs agree with each other and with the
     described soils (e.g., SPT N-values with the stated density) and flag
     any that seem unusual ... A typical range you state comes from a
     reference looked up and cited; otherwise leave it out."
5. downdrag basis:
   - Before: "The module's default: its headline procedure is UFC 3-220-20's."
   - After: "The module's default, as the module follows UFC 3-220-20's
     procedure by default."
6. Earth-retention checklist:
   - Before: "The seismic thrust INCREMENT Delta KAE acts near 0.6H while the
     M-O TOTAL thrust acts near H/3"
   - After: "The seismic thrust INCREMENT Delta PAE acts at 0.6H above the
     base (Seed & Whitman 1970) and the static PA at H/3"

## Prompt length

The domain prompt is 24,311 characters against 24,313 at 15fadcb; the page's
primary system prompt 26,215 against 26,217. Page fingerprints re-recorded in
924db45 and again in 0d9c015 (`python -m webapp.tests.page_builds record`).

## Tests (exit codes)

- drilled_shaft + axial_pile: 0 (354 passed).
- bearing_capacity: 0 (359).
- ground_improvement + soe + downdrag: 0 (606 passed, 1 skipped).
- sheet_pile + retaining_walls: 0 (482).
- validation_examples/test_published_v001_v005: 0 (15).
- funhouse_agent/tests/test_defaults_used.py: the full file exited 1 once,
  on `ground_improvement|aggregate_piers` (the E_s note first keyed on how the
  value was set; now keyed on the value). After the fix the touched modules'
  subset passed (324, and ground_improvement 106). The full file was not
  rerun.
- test_top_level_params: 0 (55).
- Deep prompt tests (nine files): 0 (150 passed, 1 skipped); after the trims,
  with the reviewer tests and test_public_text: 0 (192).
- test_reviewers + test_reviewer_citations: 0 (96).
- Adapter and calc-package tests (16 files): 0 (454).
- test_public_text: 0 (4).
- test_page_fingerprints: 0 (15) after the re-record.
- The independent review ran test_defaults_used in full: 0 (786 passed).
- After the review's fixes (747dd02 to 0d9c015): bearing_capacity 0 (359);
  the touched adapters' tests with test_defaults_used and
  test_top_level_params, -k by module, 0 (249 passed); the reviewer, deep
  prompt and references-off tests (21 files) 0 (457 passed, 1 skipped);
  test_public_text 0 (4); test_page_fingerprints 1 before the re-record
  (geotech_default, as expected), 0 (15) after.

## Noted, not fixed

None of these changes a saved report's output or what a live run returns.

- The system prompt says the agent works "in a Python Notebook environment".
- Tool Discipline asks for reference values by "semantic name" without
  saying what one is.
- The deep prompt's "canned" `*_package` and "bespoke" report.
- downdrag's result key `headline_method`, and "headline" in its calc
  package and post-liquefaction text and warnings (the tool docs now say
  "the reported method").
- The `choices` result description lists how a value was set as "given,
  default or module" and leaves out "assumed".
- axial_pile lists `shrink_swell_depth` in `defaults_used` for sand-only runs.
- The log-spiral basis that walks through Caltrans Table 4-2 is long for a
  choice.
- seismic_geotech's M-O choices for kv and wall friction delta carry no
  basis.
- The liquefaction adapter's list of B&I-2014 equations.
- The pavement specialist's playbook (`.claude/agents/pavement-design-
  specialist.md`) still says "MR-19.4".

## For the lead

- BACKLOG: GM-230 Done (9dc6948, c8e108b, 9cda9ab, 2526ed7, e598b14,
  924db45; after the review 747dd02, dd79003, d1c30b5, aa4d393, 0d9c015);
  GM-270's line applied, its measurement still open; GM-67's leftover done.
