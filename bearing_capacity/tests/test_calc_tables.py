"""bearing_capacity calculation tables (backlog GM-39): the factor and term
breakdown is the result's own and multiplies out to its terms."""

import pytest

from bearing_capacity.capacity import BearingCapacityAnalysis
from bearing_capacity.footing import Footing
from bearing_capacity.soil_profile import BearingSoilProfile, SoilLayer


@pytest.fixture(scope="module")
def result():
    footing = Footing(width=2.0, depth=1.5, shape="square")
    soil = BearingSoilProfile(layer1=SoilLayer(cohesion=10.0,
                                               friction_angle=30.0,
                                               unit_weight=18.0))
    return BearingCapacityAnalysis(footing=footing, soil=soil,
                                   load_inclination=5.0,
                                   vertical_load=500.0).compute()


def _table(result, name):
    return {t.name: t for t in result.get_tables()}[name]


def test_factors_are_the_results_own(result):
    t = _table(result, "factors_and_terms")
    assert t.column("Bearing capacity factor N") == [
        result.Nc, result.Nq, result.Ngamma]
    assert t.column("Shape s") == [result.sc, result.sq, result.sgamma]
    assert t.column("Load inclination i") == [result.ic, result.iq,
                                              result.igamma]
    assert t.column("Term value") == [result.term_cohesion,
                                      result.term_overburden,
                                      result.term_selfweight]


def test_terms_multiply_out_and_sum_to_q_ult(result):
    t = _table(result, "factors_and_terms")
    c_row, q_row, g_row = t.rows

    def prod(row):
        p = 1.0
        for v in row[1:7]:
            p *= v
        return p
    assert 10.0 * prod(c_row) == pytest.approx(result.term_cohesion)
    assert result.q_overburden * prod(q_row) == pytest.approx(
        result.term_overburden)
    assert 0.5 * result.gamma_eff * result.B_eff * prod(g_row) == \
        pytest.approx(result.term_selfweight)
    assert sum(t.column("Term value")) == pytest.approx(result.q_ultimate)


def test_summary_csv_only(result):
    t = _table(result, "capacity_summary")
    assert not t.printed
    v = dict(zip(t.column("Symbol"), t.column("Value")))
    assert v["q_ult"] == result.q_ultimate and v["q_all"] == \
        result.q_allowable
