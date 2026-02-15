# seismic_geotech — Seismic Geotechnical Analysis

## Purpose
Site classification (ASCE 7), seismic earth pressures (Mononobe-Okabe),
liquefaction triggering (Youd et al. 2001), and residual strength estimation.

## References
- ASCE 7-22 — site classification (Vs30, N-bar, su-bar)
- Mononobe-Okabe (1929) — seismic active/passive earth pressure
- Youd et al. (2001) — SPT-based liquefaction triggering (NCEER workshop)
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
- Liquefaction: CRR from N1_60cs (fines-corrected), CSR with rd depth factor
- SoilProfile adapter: to_seismic_input(amax_g, magnitude)
- 58 tests
