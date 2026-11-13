# DOC-24: quick fixes from the module audit (Q1 to Q22)

Branch `claude/doc-24-audit-quick-fixes`. Wording and document changes only; no calculation changes, so no release note.

| Q | Result | Notes |
|---|---|---|
| Q1 | Done | `wave_equation/DESIGN.md`: Public API lists the real `__init__.py` exports with signatures read from the code; `results.py` is plotting only (no result classes). |
| Q2 | Done | `sheet_pile/DESIGN.md`: `analyze_cantilever`, `analyze_anchored`, `WallSoilLayer`, earth-pressure functions, signatures from the code; layers are `WallSoilLayer`, not dicts; no `results.py`. |
| Q3 | Done | `pile_group/DESIGN.md`: exports of `__init__.py`; `GroupLoad`/`PileGroupResult` live in `rigid_cap.py`; no `results.py`. |
| Q4 | Done | `ground_improvement/DESIGN.md`: aggregate piers, wick drains, surcharge and vibro signatures copied from the code. `evaluate_feasibility` was already right. |
| Q5 | Done | `retaining_walls/DESIGN.md`: `list_reinforcement()` removed from the Python API; a note says it is an agent-adapter method (the adapter and tests use it). |
| Q6 | Done | `soe/DESIGN.md`: secant/tangent and diaphragm walls moved to Excluded and Future Work; the basal heave / blowout / piping Future Work line removed (built). `soe/__init__.py`: the embedment line said "USACE method"; the code is Caltrans free earth support (`soe/embedment.py` docstring), so the line now says that. |
| Q7 | Done | `funhouse_agent/adapters/__init__.py` calc_package brief no longer gives a number ("every module that has a calc package (list_supported_modules)"). The registry has 17. |
| Q8 | Done (manual half) | `manual_content.py` already says 17, which matches `calc_package._IMPORT_MAP` (17 entries): nothing to change. CLAUDE.md half left to the lead, as the brief says. |
| Q9 | Done | `hvsrpy = []` and `swprocess = []` removed from `pyproject.toml`; their two empty rows removed from `docs/user_manual/manual.html` (the table is generated from pyproject by `build_manual.py`, so a rebuild gives the same). Other mentions (docs/funhouse_agent_guide.md, a test docstring, the eval JSON) are historical notes about the removed modules, not extras; left. |
| Q10 | Skipped | The lead does it. |
| Q11 | Skipped | Local folders, not in git. |
| Q12 | Done | `ground_improvement` adapter `feasibility` returns: the six keys `to_dict()` gives. |
| Q13 | Done | salib adapter: `sobol_analyze` `n_samples` and `morris_analyze` `n_trajectories` say "recorded only"; the count analysed is the length of Y (X and Y for Morris). The returns line for `n_samples` says the length of Y. The sample and one-call methods really use these counts and are unchanged. |
| Q14 | Done | `section_props` adapter: parametric `section_properties` says `mesh_size` is ignored (confirmed in `sections.py`: only the polygon torsion solve reads it); `polygon_section` says what it does. |
| Q15 | Done | pystra adapter: uniform takes `a` and `b` (checked in `pystra_utils.py`); `n_function_calls` says it is an estimate, iterations times variables (`reliability.py:137, 259`). |
| Q16 | Done | soe adapter `check_bottom_blowout`: `gamma_soil` is the submerged (buoyant) unit weight (`soe/stability.py`). |
| Q17 | Done | drilled_shaft adapter `get_resistance_factors`: says there is no uplift capacity run and the uplift factors are for hand calculation (the cited lines had moved; the text is on this method's brief). |
| Q18 | Done | subsurface adapter: `trend_results` (parameter-vs-depth plot only) and `skipped` added to the plot returns (both come from `PlotResult.to_dict()`, only when non-empty). |
| Q19 | Done | liquepy brief: LPI/LSN/LDI for CPT only. |
| Q20 | Done | concrete_props adapter returns: `ixx_gross_mm4`, `d_eff_mm`, `interaction_n_kN`, `interaction_m_kNm`, `note`, all confirmed in `concrete_props_agent/results.py`. |
| Q21 | Done | pynite adapter: `rho` added to the members description with "no self-weight applied" (no self-weight code exists in `pynite_agent`). |
| Q22 | Done | The claim is in `manual_content.py` (calc_package methods), not at the cited line 1838 (that text has none). Now says the Plotly viewers in `calc_package.interactive` are library functions with no agent tool (confirmed: only `docs/gallery/build_gallery.py` calls them). `manual.html` hand-edited to match. |

## Commits

1. Q1 to Q8, Q22
2. Q9
3. Q12 to Q21
4. Fingerprint re-record (the calc_package and liquepy briefs are in the system prompt)

## Tests

- `webapp/tests/test_page_fingerprints.py`, `test_public_text.py`, `test_docs_currency.py`: 24 passed (after the re-record; before it, the geotech_default system prompt hash moved by the two brief edits, as expected).
- 77 files under `funhouse_agent/tests` that name a touched adapter or the registry: 3394 passed, 26 skipped, 2 failed.
- The 2 failures also fail on unchanged master, so they are not from this branch: `test_calc_package_figure_conventions.py::test_large_pile_group_leaves_the_forces_to_the_table` and `test_defaults_used.py::test_every_default_a_run_lists_changes_its_result[profile_figure|plot_data]`. Not looked into (outside DOC-24); the lead may want a BACKLOG item.
- Each package's own suite was not run: only DESIGN.md text and `soe/__init__.py` (a docstring) changed there.

## Calls made

None; every fix states what the code already does.

## Noted, not fixed

- `retaining_walls/DESIGN.md` still lists `results.py` in Files (not part of Q5).
- The manual's extras table text in `manual_content.py` still has other empty extras (`groundhog` etc.); only the two named in Q9 were removed.
