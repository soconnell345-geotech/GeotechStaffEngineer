"""``calculate`` over lists, and the normal distribution (GM-94).

Field session 2026-10-15: the agent read a 5 x 9 kriged grid and counted the
nodes above an excavation level by eye; 10 of its 40 counts were wrong (its
best estimate, 6 of 43, was 7 of 43 in the array it had been shown). It had
no tool that counts, and no normal CDF to turn a kriging variance into a
probability.
"""

import math

import pytest

from funhouse_agent.deep.calculate_tool import (CALCULATE_DESCRIPTION,
                                                evaluate)

Z = [870.1, 874.0, 875.2, 872.0, 873.6, 869.9, 876.1]


def _value(expr, **variables):
    out = evaluate(expr, variables)
    assert "error" not in out, out
    return out["value"]


def test_count_above_a_level():
    assert _value("count_above(z, 873.5)", z=Z) == 4.0
    assert _value("count_below(z, 873.5)", z=Z) == 3.0
    assert _value("count_above(z, 873.5) / count(z)", z=Z) == \
        pytest.approx(4 / 7)


def test_sums_and_averages():
    assert _value("sum(z)", z=Z) == pytest.approx(math.fsum(Z))
    assert _value("mean(z)", z=Z) == pytest.approx(math.fsum(Z) / 7)
    assert _value("median(z)", z=Z) == 873.6
    assert _value("max(z) - min(z)", z=Z) == pytest.approx(6.2)
    import statistics
    assert _value("std(z)", z=Z) == pytest.approx(statistics.stdev(Z))


def test_percentile_matches_numpy():
    np = pytest.importorskip("numpy")
    for p in (0, 10, 50, 90, 100):
        assert _value("percentile(z, p)", z=Z, p=p) == \
            pytest.approx(float(np.percentile(Z, p)))
    out = evaluate("percentile(z, 120)", {"z": Z})
    assert "error" in out


def test_the_normal_distribution():
    # the chance a kriged value 871 with kriging std 2 exceeds 873.5
    assert _value("1 - norm_cdf((873.5 - m) / s)", m=871.0, s=2.0) == \
        pytest.approx(0.10565, abs=1e-5)
    assert _value("norm_ppf(0.9)") == pytest.approx(1.28155, abs=1e-5)
    assert _value("norm_pdf(0)") == pytest.approx(1 / math.sqrt(2 * math.pi))
    assert _value("erf(1)") == pytest.approx(math.erf(1))


def test_a_list_in_arithmetic_is_explained():
    out = evaluate("z + 1", {"z": Z})
    assert "list variable goes into a list function" in out["error"]
    out = evaluate("count(n)", {"n": 3})
    assert "error" in out


def test_long_lists_are_summarised_in_the_echo():
    out = evaluate("mean(z)", {"z": list(range(30))})
    assert out["value"] == 14.5
    assert out["variables"]["z"] == "[30 values, 0 to 29]"


def test_bad_list_values_are_refused():
    out = evaluate("sum(z)", {"z": [1, "a", 3]})
    assert "not a number or a list of numbers" in out["error"]


def test_the_description_says_how():
    for word in ("count_above", "percentile", "norm_cdf", "never by eye"):
        assert word in CALCULATE_DESCRIPTION
