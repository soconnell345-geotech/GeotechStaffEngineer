"""V-062: site factor, kmax and Mononobe-Okabe KAE and PAE, one printed
example (GM-293).

FHWA GEC-11 Vol II (FHWA NHI-10-025, November 2009) Example E7, Seismic
Loading (pp. E7-3 to E7-6), read from
``geotech-references/docs/GEC 11 VOL 2.pdf``:

- Step 3 (p. E7-3): Site Class B, PGA = 0.206 g, S1 = 0.177 g; from the
  AASHTO tables Fpga = 1.00 and Fv = 1.0.
- Step 4 (p. E7-3): kmax = Fpga (PGA) = 1.00 (0.206 g) = 0.206 g (its Eq.
  7-1; GEC-12 Vol I Eq. 7-78 writes the same product As).
- Step 6 (pp. E7-5 and E7-6): kh = kmax = 0.206, kv = 0; phi'_b = 30 deg,
  delta = 30 deg, level backfill (I = 0), vertical face (theta = 90 deg);
  xi = atan(0.206) = 11.64 deg; KAE = 0.9008 / (0.7320 [1 + 0.6042]^2) =
  0.4782; PAE = 0.5 (0.4782)(125 pcf)(25.64 ft)^2 = 19.65 k/lft.

The example gives no Ss; Site Class B's Fa is 1.0 at every Ss (GEC-12
Table 7-28), so Ss = 0.5 here stands in. seismic_earth_pressure takes any
consistent units: pcf and ft give lb/ft.
"""

import pytest

from seismic_geotech.mononobe_okabe import (
    mononobe_okabe_KAE, seismic_earth_pressure,
)
from seismic_geotech.site_class import site_coefficients


def test_v062_site_factors_and_kmax():
    r = site_coefficients("B", Ss=0.5, S1=0.177, pga=0.206)
    assert r.Fpga == 1.00
    assert r.Fv == 1.0
    assert r.As == pytest.approx(0.206, abs=5e-4)


def test_v062_kae_to_the_printed_digit():
    kae = mononobe_okabe_KAE(30.0, 30.0, 0.206, kv=0.0)
    assert kae == pytest.approx(0.4782, abs=5e-5)


def test_v062_pae():
    kae = mononobe_okabe_KAE(30.0, 30.0, 0.206, kv=0.0)
    f = seismic_earth_pressure(125.0, 25.64, kae, KA=0.0, kv=0.0)
    assert f["PAE_total_kN_per_m"] / 1000.0 == pytest.approx(19.65, abs=0.005)


def test_v062_through_the_tool():
    from funhouse_agent.adapters.seismic_geotech import METHOD_REGISTRY as R
    r = R["seismic_earth_pressure"]({"phi": 30.0, "delta": 30.0,
                                     "kh": 0.206, "kv": 0.0})
    assert r["KAE"] == 0.4782
    s = R["site_classification"]({"vs30": 1000.0, "Ss": 0.5, "S1": 0.177,
                                  "pga": 0.206})
    assert s["site_class"] == "B"
    assert s["Fpga"] == 1.0
    assert s["As_g"] == 0.206
