"""GM-160: the variogram's nugget is a value. Left out it is fitted with
the sill and range; a number fixes it (GSTools ``fit_variogram`` with a
fixed parameter)."""

import numpy as np
import pytest

from gstools_agent import has_gstools

pytestmark = pytest.mark.skipif(not has_gstools(), reason="gstools")


def _data(n=30, seed=1):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 100, n), rng.uniform(0, 100, n)
    return x, y, np.sin(x / 20) + 0.3 * rng.standard_normal(n)


def test_left_out_it_is_fitted_and_given_it_is_used():
    from gstools_agent import analyze_variogram
    x, y, v = _data()
    fitted = analyze_variogram(x, y, v)
    assert fitted.nugget_fitted
    fixed = analyze_variogram(x, y, v, nugget=0.05)
    assert not fixed.nugget_fitted and fixed.nugget == pytest.approx(0.05)
    none = analyze_variogram(x, y, v, nugget=0.0)
    assert none.nugget == 0.0
    assert fixed.to_dict()["nugget_fitted"] is False


def test_a_negative_nugget_is_refused():
    from gstools_agent import analyze_variogram
    x, y, v = _data()
    with pytest.raises(ValueError, match="0 or more"):
        analyze_variogram(x, y, v, nugget=-1)
