"""GM-212: the cantilever_wall tool's sliding_cohesion (the printed rules
for the foundation cohesion in sliding), and UFC 3-220-20 Table 4-2 against
Fig. 4-13 and Eq. 4-12 for the log-spiral Kp."""

import warnings

import pytest

from funhouse_agent import dispatch as D
from funhouse_agent.adapters import retaining_walls as RW
from funhouse_agent.adapters import sheet_pile as SP
from geotech_common import log_spiral as LS

WALL = {"wall_height": 5.0, "gamma_backfill": 19.0, "phi_backfill": 32.0,
        "phi_foundation": 28.0, "c_foundation": 10.0}


def _wall(**kw):
    return D.call_agent("retaining_walls", "cantilever_wall", {**WALL, **kw})


# --- sliding_cohesion through the tool ----------------------------------

def test_the_default_is_drained_and_listed():
    r = _wall()
    assert r["FOS_sliding"] == pytest.approx(1.233, abs=0.0005)
    assert r["defaults_used"]["sliding_cohesion"] == "drained"
    assert r["sliding_basis"]["sliding_cohesion"] == "drained"
    assert r["sliding_basis"]["base_adhesion_kPa"] == 0.0
    assert "sliding" in r["checks_failed"]
    c = next(c for c in r["choices"] if "sliding_cohesion" in c["what"])
    assert c["how"] == "default"


def test_the_ufc_rule_given():
    r = _wall(sliding_cohesion="friction_plus_adhesion")
    assert r["FOS_sliding"] == pytest.approx(1.553, abs=0.0005)
    assert "sliding_cohesion" not in r["defaults_used"]
    b = r["sliding_basis"]
    assert b["base_adhesion_kPa"] == pytest.approx(10.0)
    assert b["adhesion_width_m"] == pytest.approx(2.335, abs=0.001)
    c = next(c for c in r["choices"] if "sliding_cohesion" in c["what"])
    assert c["how"] == "given"


def test_phi_foundation_zero_takes_the_undrained_rule():
    r = D.call_agent("retaining_walls", "cantilever_wall",
                     {**WALL, "phi_foundation": 0.0, "c_foundation": 40.0})
    assert r["defaults_used"]["sliding_cohesion"] == "undrained"
    assert r["sliding_basis"]["delta_base_deg"] is None
    assert r["sliding_basis"]["resisting_force_kN_per_m"] == \
        pytest.approx(120.0)
    assert "delta_base" not in r["defaults_used"]


def test_base_adhesion_without_a_rule_that_counts_it_is_refused():
    r = _wall(base_adhesion=5.0)
    assert "friction_plus_adhesion" in r["error"]
    assert "undrained" in r["error"]
    r = _wall(base_adhesion=5.0, sliding_cohesion="friction_plus_adhesion")
    assert "error" not in r
    assert r["sliding_basis"]["base_adhesion_kPa"] == 5.0


def test_an_unknown_rule_is_refused():
    r = _wall(sliding_cohesion="both")
    assert "error" in r


def test_the_description_names_the_printed_rules():
    spec = RW.METHOD_INFO["cantilever_wall"]["parameters"]
    d = spec["sliding_cohesion"]["description"]
    for words in ("GEC-6 Sec. 5.4 Eq. 5-38", "should not be combined",
                  "Any effective cohesion c' should be neglected",
                  "GEC-6 Eq. 5-39", "UFC 3-220-20 Table 4-1",
                  "UFC 3-220-20 Fig. 4-27", "Default depends on "
                  "phi_foundation"):
        assert words in d, words
    # the 2/3 c adhesion is gone
    assert "(2/3)*c_foundation" not in spec["c_foundation"]["description"]
    assert "2/3" not in spec["base_adhesion"]["description"]


def test_the_package_prints_the_rule(tmp_path, monkeypatch):
    """The calc package runs the tool's own code path (GM-141) and prints
    the rule's resistance."""
    monkeypatch.chdir(tmp_path)
    r = D.call_agent("calc_package", "retaining_wall_package",
                     {**WALL, "sliding_cohesion": "friction_plus_adhesion"})
    assert "error" not in r, r.get("error")
    html = "".join(f.read_text(encoding="utf-8", errors="ignore")
                   for f in tmp_path.rglob("*.htm*"))
    assert "B_e" in html and "UFC 3-220-20 Fig. 4-27" in html


# --- UFC 3-220-20 Table 4-2 (p. 194) against Fig. 4-13 and Eq. 4-12 -------

PHI = (30, 35, 40, 45)
TABLE_4_2_HALF_ROW = (5.3, 8.0, 12, 20)          # headed delta/phi' = 0.5
TABLE_4_2_HALF_ROW_COULOMB = (6.1, 9.8, 18, 44)
TABLE_4_2_ONE_ROW = (6.5, 10.5, 18, 35)          # delta/phi' = 1


def test_table_4_2_half_row_is_fig_4_13_at_0_66():
    for phi, printed in zip(PHI, TABLE_4_2_HALF_ROW):
        assert LS.dm7_2_fig_4_13_Kp(phi) == pytest.approx(printed,
                                                          rel=0.01)
        # Eq. 4-12 at the 0.5 the row is headed with gives far less
        assert LS.dm7_2_eq_4_12_Kp(phi, 0.5 * phi) < 0.9 * printed


def test_table_4_2_half_row_coulomb_is_coulomb_at_0_66():
    from retaining_walls.earth_pressure import coulomb_Kp
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for phi, printed in zip(PHI, TABLE_4_2_HALF_ROW_COULOMB):
            at_066 = coulomb_Kp(phi, 0.66 * phi)
            assert at_066 == pytest.approx(printed, rel=0.02)
            assert abs(coulomb_Kp(phi, 0.5 * phi) - printed) \
                > abs(at_066 - printed)


def test_table_4_2_one_row_agrees_with_eq_4_12():
    for phi, printed in zip(PHI, TABLE_4_2_ONE_ROW):
        assert LS.dm7_2_eq_4_12_Kp(phi, phi) == pytest.approx(printed,
                                                              rel=0.06)


def test_the_note_gives_the_numbers_the_tool_works_out():
    note = LS.TABLE_4_2_NOTE
    half = ", ".join(f"{LS.dm7_2_eq_4_12_Kp(p, 0.5 * p):.3g}" for p in PHI)
    assert half.replace(", 14.3", " and 14.3") in note
    assert "UFC 3-220-20 Table 4-2 (p. 194) is not used" in note


def test_the_result_and_descriptions_say_table_4_2_is_not_used():
    r = D.call_agent("retaining_walls", "earth_pressure_coefficient",
                     {"phi_deg": 30, "state": "passive",
                      "theory": "caquot_kerisel",
                      "log_spiral_source": "dm7_2_fig_4_13",
                      "delta_deg": 19.8})
    assert r["K"] == pytest.approx(5.3)
    assert LS.TABLE_4_2_NOTE in r["choices"][0]["basis"]
    d = RW.METHOD_INFO["earth_pressure_coefficient"]["parameters"][
        "log_spiral_source"]["description"]
    assert "Table 4-2 (p. 194) is not used" in d
    assert LS.TABLE_4_2_NOTE in SP._LOG_SPIRAL_SOURCE["description"]
