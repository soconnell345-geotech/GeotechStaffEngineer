"""
The all-NaN realization of the 2026-10-15 field session (GM-91, GM-96).

GSTools 1.7.0's two-dimensional ``Exponential.spectral_rad_ppf`` returns an
infinite wavenumber for every uniform draw within 1e-8 of zero, and one
infinite mode makes the whole randomization-method field NaN. Seed 47863 is
the session's realization 863 of a run seeded 47000; seed 137080 is the
other seed below 200,000 that draws such a ``u``. ``gstools_agent.models``
gives the exact, finite inverse there and GSTools' own number everywhere
else.
"""

import numpy as np
import pytest

from gstools_agent.gstools_utils import has_gstools

pytestmark = pytest.mark.skipif(not has_gstools(),
                                reason="gstools not installed")

from gstools_agent.tests.test_conditional_simulation import (  # noqa: E402
    BX, BY, BZ, FOOT, NAMES,
)

#: The draw that made seed 47863's field NaN (mode 310 of 1,000).
U_FIELD_SESSION = 8.985714e-09


def _models():
    import gstools as gs
    from gstools_agent.models import model_class
    kw = dict(dim=2, var=2.0, len_scale=40.0)
    return gs.Exponential(**kw), model_class("Exponential")(**kw)


def test_ppf_is_gstools_own_above_1e8():
    own, fixed = _models()
    u = np.random.RandomState(3).uniform(size=100000)
    u = np.concatenate([u[u > 1e-8], [1e-8 * (1 + 1e-9), 2e-8, 0.5]])
    assert np.array_equal(own.spectral_rad_ppf(u), fixed.spectral_rad_ppf(u))


def test_ppf_is_the_exact_finite_inverse_near_zero():
    _own, fixed = _models()
    k = fixed.spectral_rad_ppf(np.array([U_FIELD_SESSION, 0.0]))
    assert np.all(np.isfinite(k))
    exact = np.sqrt(1.0 / U_FIELD_SESSION ** 2 - 1.0) / fixed.len_rescaled
    assert k[0] == pytest.approx(exact, rel=1e-12)


def test_the_fixed_class_is_still_gstools_exponential():
    import gstools as gs
    from gstools_agent.models import model_class
    _own, fixed = _models()
    assert isinstance(fixed, gs.Exponential) and fixed.name == "Exponential"
    assert fixed.has_ppf
    assert model_class("Gaussian") is gs.Gaussian
    one_d = model_class("Exponential")(dim=1, var=1.0, len_scale=10.0)
    u = np.array([0.1, 0.5, 0.9])
    assert np.array_equal(
        one_d.spectral_rad_ppf(u),
        gs.Exponential(dim=1, var=1.0, len_scale=10.0).spectral_rad_ppf(u))


@pytest.mark.parametrize("seed", [47863, 137080])
def test_the_field_sessions_seed_draws_a_finite_realization(seed):
    from gstools_agent import analyze_conditional_simulation
    r = analyze_conditional_simulation(
        BX, BY, BZ, [{"name": "b", "polygon": FOOT, "level": 97.0}],
        model_type="Exponential", variance=2.0, len_scale=40.0,
        cell_size=2.0, n_realizations=1, seed=seed,
        simulation_method="condsrf", names=NAMES)
    assert r.redrawn == []                 # the guard was not needed
    assert r.realization_ids.tolist() == [seed]


def test_an_unconditioned_field_at_that_seed_is_finite():
    from gstools_agent import generate_random_field
    rf = generate_random_field(model_type="Exponential", variance=2.0,
                               len_scale=40.0, x_min=-15, x_max=75,
                               y_min=0, y_max=50, n_x=46, n_y=26,
                               seed=47863)
    assert np.all(np.isfinite(np.asarray(rf.field, float)))
