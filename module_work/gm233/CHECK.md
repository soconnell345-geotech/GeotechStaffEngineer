# GM-233: drilled shaft base in rock against the concrete in bearing

## What the print says

- **GEC-10 (FHWA-NHI-18-024, 2018) p. 10-25**, after Fig. 10-8: "the
  resistance at the base of a drilled shaft in rock could be governed by
  either (a) the geotechnical strength of the rock mass beneath the tip ...
  or (b) the structural strength of the concrete in bearing. Equations for
  the strength of a concrete member in bearing are presented in Chapter 12
  ... the resistance factor applied to the structural strength for bearing
  (phi = 0.70) is greater than the geotechnical resistance factor
  (phi = 0.50). A comparison of the factored resistances must be carried out
  on a case-by-case basis to determine whether the geotechnical or
  structural resistance governs the design base resistance."
- **p. 12-8, Eq. 12-7** (AASHTO 5.6.5-2): with the tip possibly unreinforced
  (cage short of the tip, or a socket that may be deepened in the field),
  check plain concrete bearing: P_r = phi P_n = 0.70 [0.85 f'c A1 m], A1 =
  cross-sectional area of the rock socket, m < 2.0 for a uniformly
  distributed bearing surface; "For a socketed shaft bearing on competent
  rock the factor m can be taken equal to 2.0". With the reinforcement to
  the tip, Eq. 12-6 with phi = 0.70 for bearing.
- **Example 10-4, p. 10-28**: phi R_BN = 0.50 (25,447) = 12,723 kips;
  "Check of factored structural bearing resistance (see Chapter 12,
  Equation 12-7)": 0.70 [0.85 (576 ksf) pi/4 (6 ft)^2 x 2] = 19,380 kips >
  12,723 kips; "The factored geotechnical base resistance governs".

So the print gives a formula and a rule: compare FACTORED resistances and
take the smaller. It gives no rule for nominal or ASD values.

## What changed and why

- `drilled_shaft/capacity.py`: with the base in rock and `concrete_fc`
  given, `base_steps["concrete_bearing"]` holds Eq. 12-7 (f'c, A1 = the
  socket's area, m = 2.0, P_n, 0.70 P_n), the geotechnical 0.50 Q_tip
  (Table 8-4's base-in-rock factor, the module's `RESISTANCE_FACTORS`), and
  which governs. A new choice, "base resistance in rock, against the
  concrete in bearing", states the check, its basis (pp. 10-25, 12-8,
  Example 10-4), "not checked" when `concrete_fc` is not given, and Eq. 12-6
  as the alternative (not computed: the tool takes no f_y). Where the base
  is not taken as on competent rock (Hoek-Brown base or `rock_base_method`
  'rqd'), the basis says p. 12-8 prints only m < 2.0, so m = 2.0 is the
  least restrictive value the print allows.
- Q_tip, Q_ultimate and Q_allowable are unchanged: they stay geotechnical,
  since the print compares factored values only.
- `drilled_shaft/lrfd.py` (`apply_lrfd`, the `lrfd_capacity` tool):
  `phi_Qt_kN` = min(0.50 Q_tip, 0.70 P_n). The result also gives
  `phi_Qt_geotechnical_kN`, `phi_Pn_concrete_bearing_kN`,
  `phi_concrete_bearing` and `phi_Qt_governed_by`. `phi_Qn_kN` follows.
- Tool description (`concrete_fc`, `lrfd_capacity` returns) and the calc
  package (a "Base Against the Concrete in Bearing (factored)" step after
  End Bearing Capacity; the f'c input line) say the same.
- `defaults_used`: no new input. `concrete_fc` was already declared with no
  default; the check is on when it is given. m = 2.0 and phi 0.70 are
  printed constants, stated in the choice.

## Before and after

GEC-10 Example 10-4 (6 ft socket, qu 360 ksf, f'c 576 ksf = 27,579 kPa):

| | before | after |
|---|---|---|
| Q_tip (nominal) | 113,193 kN (25,447 kips) | same |
| 0.70 P_n concrete | not computed | 86,208 kN (19,380 kips, as printed) |
| lrfd phi_Qt | 56,596 kN (12,723 kips) | 56,596 kN, "geotechnical" |

Made-up case where the concrete governs (same socket, qu 17,000 kPa, f'c
10,000 kPa): Q_tip 111,638 kN in both. lrfd phi_Qt before 55,819 kN,
after 31,259 kN = 0.70 x 0.85 x 10,000 x 2.627 x 2, "concrete bearing
(Eq. 12-7)"; phi_Qn before 64,633 kN (side 8,814 kN, qu already limited to
f'c by GM-284), after 40,072 kN.

## Tests

- `drilled_shaft/` 199 passed, 0 failed (5 new in
  `tests/test_concrete_bearing_base.py`: Example 10-4 as printed, the
  concrete governing, A1 is the socket's area, no check without f'c, no
  check with the base in soil).
- `webapp/tests/test_page_fingerprints.py` + `test_public_text.py`: 19
  passed. The fingerprints did not change (the adapter's METHOD_INFO is not
  part of a page fingerprint), so no re-record was needed.
- calc_package/tests, validation_examples v006-v008 and v058 (GEC-10
  Ex 10-1), the 17 funhouse_agent test files that call drilled_shaft, and
  foundry_test_harness: 1129 passed, 3 skipped (pdflatex not installed),
  0 failed.
- `funhouse_agent/tests/test_defaults_used.py` + `test_render_tables.py`
  (method probes): 1 failed,
  `test_every_default_a_run_lists_changes_its_result[profile_figure|plot_data]`;
  it fails the same way on master without this change (checked by
  stashing), so it is not from GM-233. Noted, not fixed.

## Release note

drilled_shaft, `lrfd_capacity`: with the base in rock and `concrete_fc`
given, the factored base resistance is now the smaller of the geotechnical
0.50 Q_tip and the concrete's bearing 0.70 [0.85 f'c A1 m], m = 2.0, A1 the
socket's area (GEC-10 2018 p. 10-25, Eq. 12-7 p. 12-8, as in Example
10-4). Where f'c is low against 2.5 qu, `phi_Qt_kN` and `phi_Qn_kN` drop;
the result says which governs. Nominal Q_tip, Q_ultimate and Q_allowable
are unchanged. Without `concrete_fc` nothing changes, and the choices say
the check was not made.

## Noted, not fixed

- No input for m below 2.0 on rock that is not competent: p. 12-8 prints
  no value there. The choice says the check is the least restrictive one.
- Eq. 12-6 (cage to the tip) is not computed: the tool takes no f_y and
  cannot tell whether the cage reaches the tip. Named as the alternative.
- ASD (`Q_allowable`): the print gives the comparison for factored
  resistances only, so nothing is applied there.
- With a bell on rock the geotechnical base acts on the bell's area while
  A1 stays the socket's (p. 12-8 wording); bells in rock are not covered by
  the print.

## Review fixes (2026-11-04)

- `defaults_used` for a left-out `concrete_fc` now says the base in rock is
  not checked against the concrete in bearing either (p. 10-25), not only
  the side limit (p. 10-21).
- The choice, the calc-package note and the tool text say Eq. 12-7 takes
  the tip as plain concrete (p. 12-8). The Eq. 12-6 alternative (cage to
  the tip) says it can give the lower resistance, so there the concrete can
  govern where Eq. 12-7 says it does not.
- Notes folder renamed `module_work/gm233`; user manual `concrete_fc` text
  updated to the new tool text.
