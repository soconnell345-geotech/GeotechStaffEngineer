"""QC regression: a variable may appear in at most ONE correlation pair.

The Monte Carlo sampler imposes each declared pair by overwriting the two
sample columns, so a variable in two pairs would silently lose its first
correlation. _parse_correlations must reject overlapping pairs up front
(for FOSM too, keeping both engines' accepted specs identical).
"""
import pytest

from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability.probabilistic import fosm_fos, monte_carlo_fos


def _geom():
    return SlopeGeometry(
        surface_points=[(0.0, 10.0), (10.0, 10.0), (30.0, 20.0), (40.0, 20.0)],
        soil_layers=[SlopeSoilLayer(
            name="soil", top_elevation=20.0, bottom_elevation=0.0,
            gamma=19.0, phi=25.0, c_prime=10.0)],
    )


VARS = {
    "phi": {"mean": 25.0, "cov": 0.10},
    "c_prime": {"mean": 10.0, "cov": 0.30},
    "gamma": {"mean": 19.0, "cov": 0.05},
}
OVERLAPPING = [("c_prime", "phi", -0.5), ("phi", "gamma", 0.3)]
SURF = dict(xc=18.0, yc=28.0, radius=16.0)


def test_fosm_rejects_overlapping_pairs():
    with pytest.raises(ValueError, match="more than one correlation pair"):
        fosm_fos(_geom(), VARS, correlations=OVERLAPPING, **SURF)


def test_monte_carlo_rejects_overlapping_pairs():
    with pytest.raises(ValueError, match="more than one correlation pair"):
        monte_carlo_fos(_geom(), VARS, correlations=OVERLAPPING, n=10,
                        seed=1, **SURF)


def test_disjoint_pairs_still_accepted():
    res = fosm_fos(_geom(), VARS,
                   correlations=[("c_prime", "phi", -0.5)], **SURF)
    assert res.sigma_f > 0.0
