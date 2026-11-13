# deep_axial

Paths are under C:\dev\GeotechStaffEngineer. Repository read only; nothing run.

Pattern across the group: the code puts its notes into `result.warnings` or `choices` in most places, but four places use Python `warnings.warn` instead, and nothing in `funhouse_agent` catches those (the only `catch_warnings` calls in `funhouse_agent` are in chart_eval and tests). Those four notes never reach the agent (axial_pile 1, drilled_shaft 1, pile_group 1 plus the simple method's batter case).

## axial_pile
Size: 9 python files, 8 public names exported, 3 adapter methods. The most carefully kept of the four: every documented parameter reaches the analysis, unknown inputs are refused, calc package with tables and factored steps exist, and three published examples pin it (V-001, V-002, V-059). The gaps are small and at the edges.

| # | Check | Sev | Finding | Evidence | Known? |
|---|---|---|---|---|---|
| 1 | 3 | A | `capacity_vs_depth` wraps each trial depth in `except Exception: pass`: a depth that fails for any reason except a judgment refusal is dropped with no row and no message, and if every depth fails the tool returns an empty curve with no error. Drilled shaft's version records each skipped depth and says so; this one does not. | axial_pile/capacity.py:2015-2018; adapter returns it as is, adapter axial_pile.py:274-291 | new |
| 2 | 3 | A | A pile tip deeper than the soil profile only raises a Python `warnings.warn` ("using full profile depth"); the result carries no warning, so the agent is never told the capacity assumes the last layer continues below the data. | axial_pile/capacity.py:341-346 (the "continues below" assumption stated at :797); no `catch_warnings` in funhouse_agent | new |
| 3 | 1 | B | The adapter tells the agent `capacity_vs_depth` returns rows `{depth, Q_ult, Q_skin, Q_tip}`; the real keys are `depth_m`, `Q_ultimate_kN`, `Q_skin_kN`, `Q_tip_kN`, plus `Q_factored_kN` and `warnings` when present. | funhouse_agent/adapters/axial_pile.py:389 vs axial_pile/capacity.py:2006-2014 | new |
| 4 | 1 | C | `axial_pile_capacity` "returns" lists 5 keys; `to_dict()` also gives `Q_uplift_kN`, `Q_factored_kN`, `Q_uplift_factored_kN`, `resistance_factors`, `toe_details`, `plug_weight_kN`, `warnings`, `pile_length_m`. The agent is not told the factored resistance exists. | adapter axial_pile.py:369 vs axial_pile/results.py:155-187 | new |
| 5 | 3 | C | `beta_coefficients` is offered as a parameter with one allowed value (`table_7_9`), so it selects nothing. | adapter axial_pile.py:329; axial_pile.py:50 | new |
| 6 | 4 | C | `capacity_vs_depth` rows carry no allowable capacity, no uplift and no resistance-factor detail beyond `Q_factored_kN`; drilled_shaft's curve rows carry `Q_allowable_kN`. | axial_pile/capacity.py:2006-2014 vs drilled_shaft/capacity.py:2576-2582 | new |
| 7 | 6 | C | `capacity_vs_depth` is exercised by one test in the module; `include_uplift` is documented to the agent with the pile-weight input but I did not find a published pin for uplift (V-001, V-002, V-059 are compression). Unverified beyond a name grep. | axial_pile/tests/test_axial_pile.py:509; validation_examples/INVENTORY.md:21,40,1031 | new |

Not checked: the 8 other test files' contents; whether `nordlund.py:52` and `tomlinson.py:98` ever reach `warnings.warn` (they do only when called with no notes list; I did not trace the callers); engineering values.

## drilled_shaft
Size: 9 python files, 5 public names exported, 4 adapter methods. Detailed, well-cited adapter text; the base and side options reach the analysis. The weak points are an unused uplift table, a Python-warning-only note, and the calc package leaving out the LRFD part the manual advertises.

| # | Check | Sev | Finding | Evidence | Known? |
|---|---|---|---|---|---|
| 1 | 3 | A | A shaft longer than the soil profile is run with the last layer taken to go on below the profile, announced only by a Python `warnings.warn`; `DrillShaftResult` has no `warnings` field at all, so the agent never learns the base sits in assumed soil. | drilled_shaft/capacity.py:981-985, :1397-1398; drilled_shaft/results.py (no warnings field; to_dict :123-162) | new |
| 2 | 1 | A | `get_resistance_factors` hands the agent three uplift resistance factors (0.35 clay, 0.45 sand, 0.40 rock), and the manual says the module covers LRFD, but the module computes no uplift (tension) capacity anywhere; nothing reads the uplift factors. | drilled_shaft/lrfd.py:24-26; grep "uplift" in drilled_shaft/*.py finds only lrfd.py and a comment at capacity.py:590; adapter drilled_shaft.py:189-190 | new |
| 3 | 5 | B | `drilled_shaft_package` runs only `drilled_shaft_capacity`; the LRFD factored resistances (`lrfd_capacity`) and the capacity-vs-depth curve are not in the calc package, though the manual lists "LRFD factored resistance" as a covered output. The only factored item in calc_steps is the concrete-bearing check. | funhouse_agent/adapters/calc_package.py:579-583; drilled_shaft/calc_steps.py:959-985; docs/user_manual/manual_content.py:606-609 | new |
| 4 | 1 | C | `drilled_shaft_capacity` "returns" lists 4 keys; the result also has `Q_allowable_kN`, `Q_side_clay/sand/rock_kN`, `side_resistance_by_layer`, `base_steps`, `choices`. | adapter drilled_shaft.py:245 vs drilled_shaft/results.py:125-161 | new |
| 5 | 4 | C | No settlement or load-transfer output and no tension capacity: the module is capacity only. Its DESIGN.md does not claim more; noted for the planning of a deeper effort. | drilled_shaft/DESIGN.md (no match for settlement, t-z, uplift capacity) | new |
| 6 | 3 | C | The two calc-package figures are skipped silently if matplotlib is missing (`except ImportError: pass`). | drilled_shaft/calc_steps.py:1083, :1103 | new |
| 7 | 7 | C | The "limits" of the manual say a missing RQD defaults to 100 (so 2.5*qu is taken); this is already stated to the user and is OWN-84 (open). Code agrees. | adapter drilled_shaft.py:70; docs/user_manual/manual_content.py (drilled_shaft limits) | OWN-84 open |

Not checked: the 14 test files' contents (only that published pins exist: V-006 to V-008, V-058 at INVENTORY.md:1008-1025); the LRFD arithmetic.

## downdrag
Size: 7 python files, 33 public names exported, 9 adapter methods. The largest module by far (8,080 lines). Every documented parameter of `downdrag_analysis` reaches the analysis and every accepted one is documented (cross-checked by script). Heavy on stated choices and warnings; few silent edges. Gaps are in outputs.

| # | Check | Sev | Finding | Evidence | Known? |
|---|---|---|---|---|---|
| 1 | 5 | B | The calc package covers only `downdrag_analysis`. Post-liquefaction downdrag and the six CGPR #56 methods (Endo, Poulos, Fellenius, PILENEG, rigid block, group reduction) have no calc package, and `PostLiquefactionResult` has no `get_tables`. | funhouse_agent/adapters/calc_package.py:584-587; downdrag/results.py:447 (get_tables only on DowndragResult); downdrag/liquefaction.py:666,750 | new |
| 2 | 1 | C | Three exports have no adapter method: `residual_strength_kramer_wang` (used internally at liquefaction.py:559 when a layer gives N1_60), `liquefied_layers_from_result` (used through the `liquefaction_result` input), `drag_load_reduction_factor` (called inside `drag_load_reduction_method`, cgpr56.py:1112). All reached indirectly; listed for completeness. | downdrag/__init__.py:48-62; adapter downdrag.py (no mention) | new |
| 3 | 3 | C | `downdrag_method_comparison` turns any exception from a method into a row `{"method", "error"}` instead of failing: the agent sees the message, which is the right behaviour, but the row has no neutral-plane keys, so a caller reading by key must check for it. | downdrag/cgpr56.py:1184-1188 | new |
| 4 | 4 | B | The downdrag module has no way to turn off or replace the "last settling layer" test of which layers drag: a layer is marked `settling` by the user and nothing checks that the settling soil lies above the neutral plane or that fill/drawdown loads match the flag. The module documents the settlement computation; I did not confirm it checks the flag against `fill_thickness` / `gw_drawdown`. Unverified. | downdrag/DESIGN.md:970 (Fill placement: uniform, 1-D assumption) | new |
| 5 | 4 | B | GEC-12 Fig. 7-18 for a stiff clay under sand or soft clay is not offered (UFC Fig. 6-14 default, Fig. 7-17 as a warned option). | BACKLOG.md:232 | GM-258 parked |
| 6 | 4 | B | The headline neutral plane can sit 11.7 m below where the soil stops settling more than the pile (worked example); the module warns and gives every method side by side. | BACKLOG.md:69 | OWN-105 open |
| 7 | 4 | C | Toe in a beta-method clay stays 9*cu. | BACKLOG.md:231 | GM-254 parked |
| 8 | 6 | C | Published pin is V-004 (neutral plane and drag, test_published_v001_v005.py) and UFC App. B-5.5 per the manual; each CGPR function appears in 2 or more test files, but the CGPR #56 methods are pinned to the report's own worked example (DESIGN/tests), not to a second independent source. | validation_examples/INVENTORY.md:75; downdrag/tests/test_cgpr56.py | new |

Not checked: finding 4 was not confirmed (listed as unverified); the 8,000 lines of analysis.py were scanned for the self-declared-gap words only (no TODO, stub, NotImplementedError in the module); engineering values.

## pile_group
Size: 5 python files, 14 public names exported, 5 adapter methods. Good print-checked group-effects code; the 6-DOF and the simple method are where the agent is told more than the code does.

| # | Check | Sev | Finding | Evidence | Known? |
|---|---|---|---|---|---|
| 1 | 1 | A | The user manual lists "an elastic equivalent-footing settlement (GEC-12 Sec. 7.3.5.3, Eq. 7-55 ...) ... as `influence_depth_m`" and the module docstring lists "Group settlement: Meyerhof (1976), the equivalent footing", but no adapter method reaches `group_settlement_equivalent_raft`; the only settlement tool is `meyerhof_group_settlement`, which itself says the equivalent footing is the method over cohesive deposits. The agent cannot run the method the manual says exists. | docs/user_manual/manual_content.py:682-686 (pile_group methods list); pile_group/group_efficiency.py:644; funhouse_agent/adapters/pile_group.py:758-808 (no call to it; grep for `equivalent_raft`, `influence_depth_m`, `soil_modulus` in adapters and calc_package finds none) | new |
| 2 | 1 | A | `pile_group_6dof` is described as "General 6-DOF rigid cap analysis" with batter angles accepted per pile, but the code leaves out the force-rotation coupling that batter introduces (kxz, kyz); the module's own docstring says a coupled analysis is preferred for heavily battered groups. Neither the tool description nor the result says so. | adapter pile_group.py:843-862 (no mention of limitation); pile_group/rigid_cap.py:422-427, :479-486 | GM-28 open (benchmark only; the missing statement to the agent is not covered) |
| 3 | 3 | A | `pile_group_simple` says its `piles[]` entries may carry `batter_x, batter_y`, and top-level `Vx`, `Vy`, `Mz` and stiffnesses are refused, but a batter or stiffness given on an explicit pile is accepted and silently ignored (all piles are taken vertical); cap settlement `dz` is read from pile 1's stiffness times n. | adapter pile_group.py:81, :832; pile_group/rigid_cap.py:337-345 (forces from x, y only), :371-376 (`piles[0].axial_stiffness`) | new |
| 4 | 2 | B | DESIGN.md "Public API" gives `analyze_pile_group(piles, loads, soil_params, ...) -> PileGroupResult`; no such function exists (the exports are `analyze_vertical_group_simple` and `analyze_group_6dof`). | pile_group/DESIGN.md (Public API section); pile_group/__init__.py:44-58; grep `analyze_pile_group` finds nothing in pile_group, funhouse_agent, geotech_common, calc_package | new |
| 5 | 1 | B | `block_failure_capacity` and `group_settlement_equivalent_raft` are exported with no adapter method; `block_failure_capacity` is a wrapper of the reached `block_failure_resistance`, the raft function is finding 1. | pile_group/__init__.py:26-27; adapter pile_group.py:7-12 | new |
| 6 | 5 | C | The calc package picks one of two tools; group uplift and the Meyerhof settlement are not sections of it (eta_g, block failure and the uplift block check are added when inputs are given). | funhouse_agent/adapters/calc_package.py:688-700; pile_group/calc_steps.py:370 | new |
| 7 | 3 | C | `analyze_vertical_group_simple` warns by Python `UserWarning` about ignored `Vx`, `Vy`, `Mz`; the adapter refuses them first, so the agent path is covered. Only library callers see the warning. | pile_group/rigid_cap.py:327-336; adapter pile_group.py:81 | new |
| 8 | 6 | C | Pinned to V-060 (GEC-12 Vol. III tables and UFC App. B-5.4) for the vertical, equal-spring case; the 6-DOF battered case has no published pin (DESIGN.md says so). Every exported function appears in at least two test files. | pile_group/DESIGN.md (References, 6-DOF paragraph); validation_examples/INVENTORY.md:1044-1064 | GM-28 open |

Not checked: the batter-case numbers; the calc package's uplift block logic; engineering values.
