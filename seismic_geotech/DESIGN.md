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
  boundaries are text values (not a chart read); site-factor tables (Fpga/Fa/Fv)
  are transcribed — authoring-time in-hand status not recorded; **candidate for
  wiki verification against AASHTO LRFD 9th Ed §3.10.3 Tables 3.10.3.1-1/2/3.**
- Mononobe-Okabe (1929) — seismic active/passive earth pressure. **Source basis:**
  the M-O closed-form (AASHTO LRFD §11.6.5) is coded from the equation, and the
  implementation is anchored to an INTERNAL numerical trial-wedge cross-check,
  not to a published M-O coefficient table — noted honestly as a validation
  gap (a published K_AE table would be a stronger anchor).
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
classify_site(vs30=None, n_bar=None, su_bar=None) -> SiteClassResult
mononobe_okabe_active(phi, beta, theta, delta, kh, kv) -> MOResult
analyze_liquefaction(N160, FC, sigma_v, sigma_v_eff, amax, M, ...) -> LiquefactionResult
```

## Key Notes
- N-bar: average SPT over top 30m (warns if profile < 30m)
- Liquefaction: CRR from N1_60cs (fines-corrected), CSR with rd depth factor.
  Since v5.1 (SG-1) `evaluate_liquefaction` integrates total overburden
  through the overlying layers (Σγᵢhᵢ; each point's γ applies from the
  previous evaluation depth down to its own) instead of γ(z)·z; depths must
  be increasing. Uniform profiles are unchanged.
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
