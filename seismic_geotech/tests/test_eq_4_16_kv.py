"""GM-149: PAE = (gamma H^2 / 2)(1 - kv) KAE, UFC 3-220-20 Eq. 4-16
(p. 215)."""

import pytest

from seismic_geotech.mononobe_okabe import (mononobe_okabe_KAE,
                                            seismic_earth_pressure)
from funhouse_agent import dispatch as D


def test_the_thrust_carries_one_minus_kv():
    KAE = mononobe_okabe_KAE(32, 2 / 3 * 32, 0.15, 0.1)
    f = seismic_earth_pressure(19.0, 6.0, KAE, 0.3073, kv=0.1)
    assert f["PAE_total_kN_per_m"] == pytest.approx(
        0.5 * 19 * 36 * 0.9 * KAE, abs=0.01)
    # kv = 0 (the default) is unchanged
    assert seismic_earth_pressure(19.0, 6.0, KAE, 0.3073)[
        "PAE_total_kN_per_m"] == pytest.approx(0.5 * 19 * 36 * KAE, abs=0.01)


def test_the_tool_and_its_choice_say_so():
    r = D.call_agent("seismic_geotech", "seismic_earth_pressure",
                     {"phi": 32, "kh": 0.15, "kv": 0.1, "gamma": 19, "H": 6})
    assert r["PAE_total_kN_per_m"] == pytest.approx(
        0.5 * 19 * 36 * 0.9 * r["KAE"], abs=0.02)
    kv = {c["what"]: c for c in r["choices"]}["kv"]
    assert "Eq. 4-16" in kv["basis"]


def test_no_mo_solution_leaves_kv_out_of_the_thrust():
    """phi - theta - i < 0: the Seed & Whitman KAE has no kv, nor PAE."""
    r = D.call_agent("seismic_geotech", "seismic_earth_pressure",
                     {"phi": 20, "kh": 0.5, "kv": 0.1, "gamma": 19, "H": 6})
    assert r["PAE_total_kN_per_m"] == pytest.approx(
        0.5 * 19 * 36 * r["KAE"], abs=0.02)
    kv = {c["what"]: c for c in r["choices"]}["kv"]
    assert kv["basis"].startswith("not used")


def test_the_calc_package_prints_one_minus_kv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from funhouse_agent.adapters.seismic_geotech import (
        run_seismic_earth_pressure)
    from seismic_geotech.calc_steps import get_calc_steps
    r, _, used = run_seismic_earth_pressure(
        {"phi": 32, "kh": 0.15, "kv": 0.1, "gamma": 19, "H": 6})
    steps = [i for s in get_calc_steps(r, used) for i in s.items
             if getattr(i, "title", "") == "Total Seismic Active Force"]
    (step,) = steps
    assert "(1 - k_v)" in step.equation and "(1 - 0.100)" in \
        step.substitution
    assert float(step.result_value) == pytest.approx(r.PAE_total)
