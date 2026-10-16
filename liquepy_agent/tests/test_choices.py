"""What the B&I-2014 liquefaction results say they chose (backlog GM-115),
and rd, MSF and K_sigma reported per SPT layer."""

import numpy as np
import pytest

from liquepy_agent.liquepy_utils import has_liquepy
from liquepy_agent.results import SPTLiquefactionResult
from liquepy_agent.spt_liquefaction import bi2014_spt_msf

requires_liquepy = pytest.mark.skipif(
    not has_liquepy(), reason="liquepy not installed"
)


@requires_liquepy
def test_spt_layers_carry_rd_msf_and_k_sigma():
    from liquepy_agent import analyze_spt_liquefaction
    r = analyze_spt_liquefaction(depth=[3, 6, 9], n1_60=[10, 15, 20],
                                 fc=[10, 10, 25], gamma=[18, 19, 19],
                                 amax_g=0.3, gwt_depth=2.0, m_w=7.0)
    rows = r.layer_results()
    for i, row in enumerate(rows):
        assert row["MSF"] == round(bi2014_spt_msf(7.0, r.n1_60cs[i]), 3)
        # CSR = 0.65 amax (sigma_v / sigma'v) rd
        csr = 0.65 * 0.3 * r.sigma_v[i] / r.sigma_veff[i] * r.rd[i]
        assert r.csr[i] == pytest.approx(csr, rel=1e-9)
        assert 0 < row["rd"] <= 1 and row["K_sigma"] > 0
    d = r.to_dict()
    procedure, c0, stress = d["choices"]
    assert procedure["chosen"] == "Boulanger & Idriss (2014) SPT"
    assert c0 == {**c0, "chosen": "2.8", "how": "default"}
    assert "not summed" in stress["basis"]


def test_a_result_built_without_the_factors_still_lists_its_layers():
    r = SPTLiquefactionResult(
        n_layers=1, depth=np.array([3.0]), n1_60=np.array([10.0]),
        n1_60cs=np.array([11.0]), fines_content=np.array([10.0]),
        sigma_v=np.array([54.0]), sigma_veff=np.array([44.0]),
        csr=np.array([0.2]), crr=np.array([0.15]),
        factor_of_safety=np.array([0.75]), liquefiable=np.array([True]))
    (row,) = r.layer_results()
    assert "rd" not in row and row["FOS_liq"] == 0.75
    assert "choices" not in r.to_dict()


@requires_liquepy
def test_cpt_choices():
    from liquepy_agent import analyze_cpt_liquefaction
    r = analyze_cpt_liquefaction(
        depth=[0.5 * i for i in range(1, 21)],
        q_c=[3000 + 400 * i for i in range(20)],
        f_s=[30 + 2 * i for i in range(20)], gwl=1.5, pga=0.3, m_w=7.0,
        i_c_limit=2.4)
    procedure, counted, strains = r.to_dict()["choices"]
    assert "B&I" in procedure["basis"] or "Boulanger" in procedure["chosen"]
    assert counted["chosen"] == "below the water table with Ic < 2.4"
    assert counted["how"] == "given"
    assert "Zhang" in strains["chosen"]
