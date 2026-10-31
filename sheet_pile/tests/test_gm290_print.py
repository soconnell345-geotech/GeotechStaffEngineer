"""GM-290: the cantilever sheet pile wall as the Caltrans Trenching and
Shoring Manual (2025) Sec. 7-5.02 and UFC 3-220-20 (2025) Fig. 4-41 print it.

The module's method is the Caltrans Simplified Method: moments about the
pivot O give D0 (step 2), D = 1.2 D0 (step 3: "The 20 percent increase is
not a factor of safety"), the passive pressure divided by the FS (UFC Fig.
4-41 step 2) and the maximum moment at zero shear on that diagram (UFC step
5). Uniform dry sand has closed forms for each (made-up wall: H = 4 m,
gamma 18 kN/m3, phi' 30 deg, Rankine Ka 1/3 and Kp 3, FS 1.5).
"""

import pytest

from sheet_pile import WallSoilLayer, analyze_cantilever
from sheet_pile.calc_steps import get_calc_steps

H, GAMMA = 4.0, 18.0
KA, KP_F = 1.0 / 3.0, 3.0 / 1.5


@pytest.fixture(scope="module")
def wall():
    return analyze_cantilever(
        H, [WallSoilLayer(30.0, GAMMA, friction_angle=30.0)],
        FOS_passive=1.5)


def test_D0_is_the_moment_balance_about_O(wall):
    # Ka gamma (H + D0)^3 / 6 = (Kp / FS) gamma D0^3 / 6
    D0 = H / ((KP_F / KA) ** (1 / 3) - 1)
    assert wall.embedment_converged == pytest.approx(D0, rel=0.005)


def test_D_is_1_2_D0(wall):
    assert wall.embedment_increase == 1.2
    assert wall.embedment_depth == pytest.approx(
        1.2 * wall.embedment_converged)
    assert wall.total_wall_length == pytest.approx(
        H + 1.2 * wall.embedment_converged)


def test_moment_at_zero_shear_on_the_factored_diagram(wall):
    # zero shear x below the excavation: Ka (H + x)^2 = (Kp / FS) x^2
    x = H / ((KP_F / KA) ** 0.5 - 1)
    M = GAMMA / 6 * (KA * (H + x) ** 3 - KP_F * x ** 3)
    assert wall.max_moment == pytest.approx(M, rel=0.005)
    # within two of the 500 integration steps
    assert wall.max_moment_depth == pytest.approx(H + x, abs=0.04)
    # the moment's body ends at O, where the moment closes
    assert wall.profile[-1][0] == pytest.approx(
        H + wall.embedment_converged, abs=0.02)
    assert abs(wall.profile[-1][6]) < 0.01 * M


def test_the_steps_cite_the_print(wall):
    analysis = {"wall_type": "cantilever", "excavation_depth": H,
                "soil_layers": [WallSoilLayer(30.0, GAMMA,
                                              friction_angle=30.0)],
                "FOS_passive": 1.5}
    steps = {getattr(it, "result_name", None): it
             for sec in get_calc_steps(wall, analysis) for it in sec.items}
    design = steps["D_design"]
    assert "1.20" in design.equation
    assert "7-5.02" in design.reference and "4-41" in design.reference
    assert "not a factor of safety" in design.notes
    assert "Figure 4-41 step 5" in steps["M_max"].notes
    assert "FS = 1.0" in steps["M_max"].notes
