"""The conditional simulation against conditioning by its definition.

"Conditioned on the borings" means: of all the fields the covariance model
can produce, only those that pass through the borings. That can be drawn
without any kriging at all: generate unconditioned fields and keep the ones
whose values at the borings fall within a small tolerance of the observed
values. This test does exactly that (rejection sampling, plain numpy) and
sets the spread of a total over a footprint against both of the tool's
methods, for the SAME covariance model, mean and borings.

Neither number is a statement about the ground: the Gaussian model, its
covariance and which borings count are assumptions, and they usually matter
more than the method. The question here is narrower: given those
assumptions, does the method draw the distribution they imply?
(Owner's question, 2026-10-15: "How do you know your model is the correct
one here?")

Measured when written (3 borings, 12 cells, Exponential var 4, len 15 m):
by definition 145-149, the formula 144, kriging_residual 142, condsrf 178
(about 1.2 times).
"""

import numpy as np
import pytest
from scipy.spatial.distance import cdist

from gstools_agent.gstools_utils import has_gstools

pytestmark = pytest.mark.skipif(not has_gstools(),
                                reason="gstools is not installed")

VAR, LEN = 4.0, 15.0
BORINGS = np.array([[5.0, 5.0], [25.0, 8.0], [14.0, 22.0]])
VALUES = np.array([102.0, 98.5, 100.4])
REGION = [{"name": "F", "polygon": [[8, 8], [22, 8], [22, 18], [8, 18]],
           "level": 90.0}]


def _run(method):
    from gstools_agent import analyze_conditional_simulation
    return analyze_conditional_simulation(
        list(BORINGS[:, 0]), list(BORINGS[:, 1]), list(VALUES), REGION,
        model_type="Exponential", variance=VAR, len_scale=LEN, nugget=0.0,
        kriging_type="simple", cell_size=3.5, n_realizations=4000, seed=1,
        simulation_method=method)


def _cells(result):
    mask = np.asarray(result.cell_masks[0])
    return np.c_[np.asarray(result.cell_x)[mask],
                 np.asarray(result.cell_y)[mask]]


def _cov(a, b):
    return VAR * np.exp(-cdist(a, b) / LEN)       # GSTools' Exponential


@pytest.fixture(scope="module")
def runs():
    return {m: _run(m) for m in ("kriging_residual", "condsrf")}


def test_by_definition(runs):
    r = runs["kriging_residual"]
    cells = _cells(r)
    mean = float(VALUES.mean())                   # the simple-kriging mean
    pts = np.vstack([BORINGS, cells])
    chol = np.linalg.cholesky(_cov(pts, pts) + 1e-10 * np.eye(len(pts)))
    rng = np.random.default_rng(2026)
    tol = 0.1 * np.sqrt(VAR)
    kept = []
    while sum(len(k) for k in kept) < 2000:
        z = chol @ rng.standard_normal((len(pts), 400_000)) + mean
        through = np.all(np.abs(z[:3] - VALUES[:, None]) < tol, axis=0)
        kept.append(r.cell_area * np.sum(z[3:, through] - 90.0, axis=0))
    by_definition = np.concatenate(kept).std(ddof=1)

    # the textbook formula for the same thing (Gaussian conditioning)
    c_dd, c_xd = _cov(BORINGS, BORINGS), _cov(cells, BORINGS)
    w = np.linalg.solve(c_dd, c_xd.T).T
    cond = _cov(cells, cells) - w @ c_xd.T
    formula = r.cell_area * np.sqrt(np.ones(len(cells)) @ cond
                                    @ np.ones(len(cells)))
    # rejection with a finite tolerance and 2,000 kept fields: a few %
    assert by_definition == pytest.approx(formula, rel=0.06)

    default = r.results[0]["volume_above"]["std"]
    condsrf = runs["condsrf"].results[0]["volume_above"]["std"]
    assert default == pytest.approx(formula, rel=0.05)
    assert default == pytest.approx(by_definition, rel=0.08)
    # GSTools 1.7's CondSRF: the kriged field plus an unconditioned field
    # scaled by the kriging std, so the cells keep the unconditioned
    # correlation and the total spreads too much
    assert condsrf > 1.12 * formula
    # both get the mean right
    assert r.results[0]["volume_above"]["mean"] == pytest.approx(
        runs["condsrf"].results[0]["volume_above"]["mean"], rel=0.01)
