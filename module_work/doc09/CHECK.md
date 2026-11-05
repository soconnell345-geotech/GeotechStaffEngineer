# DOC-09, DOC-11, DOC-12, DOC-17: stale documentation, checked against the repo (2026-11-05)

Documentation only. Branch `claude/doc-09-11-12-17-stale-docs`; master at the start. One code-file touch: a comment in `webapp/app.py` (see DOC-17). Not edited, as instructed: `slope_stability/DESIGN.md`, `docs/user_manual/`, BACKLOG.md, HANDOFF.md, CLAUDE.md.

## How each count was measured (run from the repo root, venv `/tmp/v`, `pip install -e .`)

| Count | Command | Output | Used in |
|---|---|---|---|
| Dispatch names: analysis / reference | `python -c "from funhouse_agent import dispatch as d; print(len(d.ANALYSIS_MODULES), len(d.REFERENCE_MODULES), len(d.MODULE_REGISTRY))"` | `36 34 70` | funhouse_agent/DESIGN.md lines 25, 69 (was 69 = 36 + 33); the 34 are 32 reference modules plus `reference_db` and `figure_db` |
| Adapter files | `ls funhouse_agent/adapters/*_adapter.py \| wc -l` | `54` | funhouse_agent_guide.md tree (was "50 adapter modules total") |
| Reference modules in the library | `ls -d geotech-references/geotech_references/*/ \| wc -l` | `33` (`dm7_1` and `dm7_2` are two directories; dispatch's single `dm7` makes 32) | README.md lines 5, 78; overview.html pill (was 21) |
| Foundry analysis tool agents | `ls foundry/*_foundry.py \| wc -l` | `25` | README.md line 60, overview.html (was "48 standalone tool agents") |
| Reference-library tool agents | `ls geotech-references/agents/*_agent.py \| wc -l` | `20` | README.md line 61, overview.html (was "9 library agents", and overview.html still said "groundhog +", a package deleted 2026-09) |
| Analysis modules | top-level packages with `__init__.py` and `DESIGN.md` (37), less the five infrastructure or rendering ones (`funhouse_agent`, `geotech_common`, `calc_package`, `profile_figure`, `geo_project`) | `32` | README.md lines 5, 71; overview.html pill and band (was 30). The 37: `for d in */; do d=${d%/}; [ -f $d/__init__.py ] && [ -f $d/DESIGN.md ] && echo $d; done`. `report_ingest` has no DESIGN.md and is not counted |
| Whole-repo tests | `python -m pytest --collect-only -q -p no:cacheprovider` | `24760 tests collected in 42.52s` (two packages all-skipped for a missing `ipywidgets` / `python_ags4`) | `.claude/skills/run-geotech-staff-engineer/SKILL.md` (the item's "~8,600" was already replaced by "about 15,400 ... 2026-10-15"; now 24,760) |
| Reference tests per manual | `cd geotech-references; for k in dm7_1 dm7_2 gec6 gec7 gec10 gec11 gec12 gec13 micropile; do pytest --collect-only -q $(ls tests/test_${k}*.py) \| grep -c "::"; done` | dm7_1 892, dm7_2 1122 (2,014), gec6 121, gec7 101, gec10 247, gec11 135, gec12 231, gec13 195, micropile 108; sum 3,152 | docs/agent_showcase.html table (was 2,008 / 121 / 101 / 175 / 130 / 147 / 105 / 108 / 2,810) |

The agent_showcase counts are the files named for each manual (`test_<ref>*.py`), so they include the `*_text.py` and printed-chapter files.

## DOC-09, what changed

- `README.md`: "30 analysis modules · 21 digitized references" is "32 · 33"; the diagram's "48 standalone tool agents / 9 library agents" is "25 / 20 reference-library agents"; "30 modules" is 32; the reference box is 33 and names EM and Eurocode 7.
- `docs/overview.html`: the same counts (pills, band title, the foundry box), and "groundhog + 9 lib agents" (groundhog is deleted) is "20 reference-library agents".
- `docs/funhouse_agent_guide.md`: the "Geotech-References (14)" table, which listed the deleted `fema_p2192`, `noaa_frost` and `ufc_dewatering` and an airfield `ufc_pavement`, is replaced by the current 32 modules plus two search tools, with the deletion noted. `hvsrpy` / `swprocess` are out of the seismic reviewer's scope text and its table, and out of the file tree (removed 2026-09; the guide's own line 180 already said so). `%pip install "geotech-staff-engineer[deep]==5.0.0"` is now the plain install (a plain install brings the whole stack since 2026-07-17; `pyproject.toml` keeps the old extras as empty aliases). The tree's `fema_adapter.py, noaa_frost_adapter.py` line is `fema_p2082_adapter.py`; "(50 adapter modules total)" is 54 files / 70 dispatch names.
- `webapp/README.md`: the install lines `[deep,full,webapp]` (two places, one in the Databricks cell) are the plain install, with a note that the old extras still work; the `[webapp]` and `[deep]` extra mentions say so.
- `funhouse_agent/DESIGN.md`: 69 adapter modules (36 + 33) is 70 adapter names (36 + 34).
- `SKILL.md` (the item's path is `.claude/skills/run-geotech-staff-engineer/SKILL.md`): the test count, above.
- `docs/agent_showcase.html` (this repo's copy): the test counts, above.

## DOC-11, what changed

- `seismic_geotech/DESIGN.md`: both "checks" are closed by BACKLOG_DONE.md **GM-293** (merge 0b36ef7, `module_work/gm293/CHECK.md`). The site-factor tables are no longer "candidate for wiki verification": site classes follow GEC-12 Table 7-26 and the 75 factor cells and interpolation Table 7-27, checked against the print. The Mononobe-Okabe "validation gap" is closed: Eq. 4-15 (UFC 3-220-20) and Eq. 44 (GEC-4), KPE wall friction 0.4 phi by default (phi/3 offered), GEC-11 Example E7 reproduced (V-062). Its static KA is Rankine, stated (BACKLOG_DONE.md GM-142).
- `lateral_pile/DESIGN.md`: "matches to machine precision at equal h" now says what the test asserts: `lateral_pile/tests/test_stickup_and_banded.py` compares the maximum moment and the embedded profile at relative tolerance 5e-3 (the grade-node shear carries an O(h) artifact).
- `pile_group/DESIGN.md`: already current. It cites USACE EM 1110-2-2906 and CPGA (ITL-89-4) at line 18, says neither is held and that the 6-DOF method is pinned only on GEC-12 Vol. III's vertical-pile case (V-060). Nothing to change; the open benchmark is BACKLOG GM-28.
- `subsurface_characterization/DESIGN.md`: already current. Its DIGGS section lists the dialect the parser reads (`diggs_parser.py` has loops for 16 test elements: SPT, Atterberg, static cone, vane, consolidation, triaxial, direct shear, unconfined, point load, specific gravity, gradation, permeability, compaction, dynamic probe, pressuremeter, dilatometer) and describes `diggs26.py`. Nothing to change.
- `slope_stability/DESIGN.md`: NOT edited (slope work is held, OWN-101 / GM-140). Its stale line still reads `:1717` "The legacy Spencer / Morgenstern-Price are APPROXIMATE (SS-1)" (the item's old `:359-367`). Noted, not fixed; the "currently disconnected" and "no adapter yet" lines are gone, as the item says.

## DOC-12, what changed

- `validation_examples/RESULTS.md` V-038: "default preserved, flip owner-gated" now says the default stays `'fellenius'` (owner decision at v5.4.0, recorded in `slope_stability/VALIDATION.md` and CLAUDE.md's v5.4.0 block) and points the rebuild of rapid drawdown at the open slope plan, BACKLOG OWN-101 step 9. BACKLOG.md's "merged" entry says OWN-17, the flip question, went into OWN-101. **No BACKLOG Done entry states the decision itself**, so the line cites the two places that do.
- RESULTS V-052: "assumed 2.5:1" now says the upstream section is published (2:1 upper / 3:1 lower) and was re-run as V-052b (`TIER_A_LEDGER.md`, "Follow-up tickets" item 2: Corps 0.726, LK 1.279, DWW 1.266 against published 0.823 / about 1.05; `slope_stability/VALIDATION.md` line 383 for the later automatic-window values). The 2.5:1 row stays as the representative-face ordering check.
- `module_work/wiki_verification/TIER_A_LEDGER.md` (qL row, now line 106; the item says 109): "OPEN" is "CLOSED", pointing at that file's own "Defect #4 ... CLOSED" section and geotech-references d9dab7a (2026-10-08, "gec_12 Fig 7-15: re-measured off the printed figure"; verified with `git -C geotech-references log -1 d9dab7a`).

## DOC-17, what changed

- `pavement_design/DESIGN.md`: the section "Scope limits (out of scope, by design)" is "(not built; parked, not ruled out)", with a note that the owner's call is open as BACKLOG OWN-21 (airfield, rigid curves F-2 to F-31, AASHTO Part III). That resolves the DESIGN-vs-CLAUDE.md contradiction ("out of scope" against "PARKED") by pointing at the item that owns it.
- `.claude/agents/structural-calc-specialist.md`: "not yet onboarded (see the Scope note above)" now says "outside the references listed under Scope above (the `References` bullet)", since `ufc_structural`, `em_2104`, `em_2107`, `ufc_collapse`, `gsa_collapse`, `wood_handbook` and `ufc_concrete_practice` are all onboarded.
- `webapp/app.py` comment only (line 1034; the item says 793-794): "a recorded future item in HANDOFF §0a" is "BACKLOG TA-07". No code line changed.
- `module_work/FUTURE_IDEAS.md` "VISION EFFICIENCY": added brief 4's measured numbers from `module_work/review_eval_results/2026-10-07_foundry_5.32.1rc3/TRACE_REVIEW.md` §3.2-3.3 (tokens, minutes, tiling 90 of 100 page looks against 1 of 120, the long rare-tag set 442 K to 997 K, probe image tokens).

## Noted, not fixed

- **In the geotech-references repo (a separate repo; this branch cannot change the submodule), with the measured values to use:**
  - its `CLAUDE.md` GEC table test counts: GEC-7 101 (right), GEC-10 195 (now 247), Micropile 108 (right), GEC-12 147 (now 231), GEC-11 130 (now 135), GEC-13 168 (now 195); DM7 2,014 is right (command above);
  - its `docs/agent_showcase.html` carries the same "2,008" and the table above;
  - `agents/references_agent.py` (its catalog block, lines 29-372) and `agents/ufc_pavement_agent.py` call UFC 3-250-01 "Pavement Design for Roads and Parking Areas", while the refs `CLAUDE.md` (line 146) titles it "Pavement Design for Roads, Streets, Walks, and Storage Areas". The printed cover is not held here, so I did not pick one; settle it against the cover before editing.
- `docs/funhouse_agent_guide.md`: the other hand-kept headings ("Core Analysis (17)", "External Library Adapters (6)", "Shared factory for 14 reference adapters", the 35 / 9 test counts in the tree) were not named by DOC-09 and are not re-measured; the guide says its list may lag `list_agents()`.
- `docs/overview.html` / README "32 analysis modules" is one definition (packages with a DESIGN.md less infrastructure); CLAUDE.md's inventory also says 32, but dispatch's `ANALYSIS_MODULES` is 36 because it includes `calc_package`, `profile_figure`, `worked_examples`, `drawing_ir`, `pdf_import` and the `liquefaction` router.
- `slope_stability/DESIGN.md` Spencer "APPROXIMATE" (above).

## Tests

- `python -m pytest -q webapp/tests/test_public_text.py webapp/tests/test_docs_currency.py webapp/tests/test_backlog.py`: 20 passed, 0 failed (exit 0).
- Tests that read files I changed (`webapp/app.py`, the agent definition, the reviewers): `webapp/tests/test_page_fingerprints.py`, `webapp/tests/test_record_ev06.py`, `funhouse_agent/tests/test_reviewer*.py`, `test_structural_adapters.py`: 176 passed, 0 failed (exit 0). No tool text or prompt changed, so no fingerprint re-record.

## Release note

No calculated value changes.
