# seismic_geotech — Seismic Geotechnical Analysis

## Purpose
Site classification (AASHTO LRFD / NEHRP), seismic earth pressures
(Mononobe-Okabe), liquefaction triggering (Youd et al. 2001), and residual
strength estimation.

## References
- AASHTO LRFD Bridge Design Specifications, 9th Ed., Section 3.10.3, with
  NEHRP Recommended Seismic Provisions (FEMA P-1050) — site classification
  (Vs30 / N-bar / su-bar boundaries and site factors). This is the standard the
  code implements (`calc_steps.py`, `__init__.py`); an earlier DESIGN.md note
  saying "ASCE 7" was inaccurate. **Source basis:** the Vs30/N/su class
  boundaries are text values (not a chart read). The site classes follow GEC-12
  Table 7-26 and the Fpga/Fa/Fv site-factor cells and their interpolation
  GEC-12 Table 7-27 (75 cells checked against the print; Fpga is used only when
  a PGA is given). That closes the earlier "candidate for wiki verification"
  note: BACKLOG_DONE.md GM-293 (merge 0b36ef7, `module_work/gm293/CHECK.md`).
- Mononobe-Okabe (1929) — seismic active/passive earth pressure. **Source basis:**
  the M-O closed form is coded as UFC 3-220-20 Eq. 4-15 and GEC-4 Eq. 44 print
  it, with the KPE wall friction at phi/3 by default (NHI-06-089 p. 10-19,
  above which it finds a planar passive wedge unreliable; GM-310, the call
  for OWN-114 of 2026-11-12) and UFC 3-220-20 Fig. 4-8's 0.4 phi offered; it
  was 0.4 phi from GM-293's review until GM-310. At phi 30 deg and kh 0.2
  KPE is 3.500 at phi/3 against 3.731 at 0.4 phi. It is anchored to a published worked example, GEC-11
  Example E7 (V-062), as well as to the internal trial-wedge cross-check, so
  the earlier "validation gap" is closed: BACKLOG_DONE.md GM-293. The static KA
  inside M-O is Rankine, a choice the result states (BACKLOG_DONE.md GM-142).
- **Youd et al. (2001) — SPT-based liquefaction triggering (NCEER/NSF workshop;
  the updated Seed-Idriss simplified procedure).** The triggering procedure in
  `liquefaction.py` is NCEER / Youd-2001 (NCEER CRR fit, NCEER MSF, Youd fines
  correction, Liao & Whitman rd) — it is NOT Boulanger & Idriss (2014). For
  B&I-2014 triggering use `liquepy_agent`. The unified agent-layer liquefaction
  tool defaults to B&I-2014 and exposes this NCEER-2001 procedure behind
  `method="nceer2001"`.
- Liao & Whitman (1986) — rd stress-reduction factor
- Seed & Harder (1990), Idriss & Boulanger (2008) — residual strength

## Files
- `site_class.py` — Site A-F classification, Fpga/Fa/Fv amplification tables
- `mononobe_okabe.py` — KAE, KPE, seismic earth pressure increment
- `liquefaction.py` — CSR, CRR, fines correction, factor of safety
- `residual_strength.py` — Seed-Harder, Idriss-Boulanger correlations
- `results.py` — SiteClassResult, MOResult, LiquefactionResult, etc.

## Public API
```python
# exports of seismic_geotech/__init__.py
classify_site(vs30=None, n_bar=None, su_bar=None) -> str
site_coefficients(site_class, Ss, S1, pga=None) -> SiteClassResult
mononobe_okabe_KAE(phi_deg, delta_deg, kh, kv=0.0, beta_deg=0.0, i_deg=0.0) -> float
mononobe_okabe_KPE(phi_deg, delta_deg, kh, kv=0.0, beta_deg=0.0, i_deg=0.0) -> float
seismic_earth_pressure(gamma, H, KAE, KA, kv=0.0) -> dict
evaluate_liquefaction(layer_depths, layer_N160, layer_FC, layer_gamma, amax_g, ...)
post_liquefaction_strength(N160cs, sigma_v_eff=None, method="seed_harder", ...)
SiteClassResult, SeismicEarthPressureResult, LiquefactionResult
```

## Key Notes
- N-bar: average SPT over top 30m (warns if profile < 30m)
- Liquefaction: CRR from N1_60cs (fines-corrected), CSR with rd depth factor.
  Since v5.1 (SG-1) `evaluate_liquefaction` integrates total overburden
  through the overlying layers (Σγᵢhᵢ; each point's γ applies from the
  previous evaluation depth down to its own) instead of γ(z)·z; depths must
  be increasing. Uniform profiles are unchanged.
  Since GM-324 (2026-11-13) the B&I (2014) SPT route in `liquepy_agent`
  uses the same rule (`TOTAL_STRESS_RULE`, the same text in both modules),
  the result's `choices` states it, and the K_sigma entry names the depths
  with sigma'v above 100 kPa, where leaving K_sigma out matters. The
  depths are where (N1)60 was measured, not layer mid-depths: a change in
  unit weight between two depths is taken at the upper one, so the tool
  text asks for a depth at each change in unit weight, and `choices` says
  so. A first depth of 0 or less is refused, as on the B&I route (it gave
  the FOS 99.9 placeholder).
- Site coefficients (SG-2): `site_coefficients(..., pga=)` interpolates Fpga
  against PGA per AASHTO Table 3.10.3.2-1. Without `pga`, Fpga ≈ Fa(Ss) — a
  documented approximation (same table values, wrong abscissa; exact only
  when Ss = 2.5·PGA).
- Mononobe-Okabe sign convention (SG-3, fixed v5.1): wall batter β positive
  when the back face leans toward the backfill (Coulomb α = 90° + β). At
  kh = kv = 0, KAE → Coulomb Ka and KPE → Coulomb Kp exactly for any β, i
  (previously xfail'd for battered passive). KAE was already exact; KPE had
  a flipped β sign in the numerator AND a flipped θ sign in the cos(δ+θ+β)
  terms — the latter affected vertical walls too (e.g. φ=30°, δ=15°, kh=0.2:
  KPE 3.35 → 4.13). Both coefficients verified against a numerical M-O wedge
  limit-equilibrium solution. Active uses inertia toward the wall; passive
  uses inertia away from the wall (worst case each).
- SoilProfile adapter: to_seismic_input(amax_g, magnitude)
- Static KA behind PA and delta-PAE (GM-120, GM-142): Rankine from phi by
  default (unchanged). With a sloping backfill, a batter or wall friction
  that is not the only reading, so `earth_pressure_choices(..., beta_deg=)`
  also states Coulomb's KA with KAE's own delta, i and beta (KAE at kh = 0)
  in the basis and as the first alternative; the tool passes its beta.
- PAE = (gamma H^2 / 2)(1 - kv) KAE, UFC 3-220-20 Eq. 4-16 (p. 215), with
  KAE from Eq. 4-15 (cos psi in its denominator, psi = atan(kh/(1 - kv)))
  (GM-149, 2026-10-19). Before, the (1 - kv) was left out, overstating
  PAE by 1/(1 - kv) wherever kv was not 0 (kv = 0, the default, is
  unchanged). Where M-O has no solution the Seed & Whitman KAE leaves kv
  out, and so does PAE; the `kv` choice says which.
