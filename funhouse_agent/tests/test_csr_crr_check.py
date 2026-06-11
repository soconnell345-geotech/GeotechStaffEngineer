"""Regression for the csr_crr_check adapter (v5.1).

The adapter previously called stress_reduction_rd(z, M) (module takes z only),
passed rd where compute_CSR expects z, applied MSF twice (compute_CSR already
divides by MSF), and added fines_correction as a delta (it returns (N1)60cs).
"""
import pytest

from funhouse_agent.adapters.seismic_geotech import _run_csr_crr_check
from seismic_geotech.liquefaction import (
    CRR_from_N160cs,
    compute_CSR,
    fines_correction,
    stress_reduction_rd,
)


PARAMS = {
    "depth": 6.0,
    "N160": 12.0,
    "FC": 15.0,
    "amax_g": 0.25,
    "sigma_v": 110.0,
    "sigma_v_eff": 75.0,
    "magnitude": 6.75,
}


class TestCsrCrrCheck:
    def test_runs_and_matches_module_functions(self):
        out = _run_csr_crr_check(dict(PARAMS))
        N160cs = fines_correction(PARAMS["N160"], PARAMS["FC"])
        CSR = compute_CSR(
            PARAMS["amax_g"], PARAMS["sigma_v"], PARAMS["sigma_v_eff"],
            PARAMS["depth"], PARAMS["magnitude"],
        )
        CRR = CRR_from_N160cs(N160cs)
        assert out["N160cs"] == pytest.approx(N160cs, abs=0.05)
        assert out["CSR"] == pytest.approx(CSR, abs=5e-4)
        assert out["CRR"] == pytest.approx(CRR, abs=5e-4)
        assert out["FOS_liq"] == pytest.approx(CRR / CSR, abs=5e-3)
        assert out["rd"] == pytest.approx(stress_reduction_rd(PARAMS["depth"]), abs=5e-4)
        assert out["liquefiable"] == (out["FOS_liq"] < 1.0)

    def test_msf_applied_once(self):
        # M=7.5 -> MSF=1; lowering M must RAISE FOS (CSR_M7.5 = CSR/MSF, MSF>1 for M<7.5)
        base = _run_csr_crr_check(dict(PARAMS, magnitude=7.5))
        low_m = _run_csr_crr_check(dict(PARAMS, magnitude=6.0))
        assert low_m["FOS_liq"] > base["FOS_liq"]
        # CRR is reported on the M7.5 basis, independent of M
        assert low_m["CRR"] == base["CRR"]
