"""BACKLOG GM-265: every geostatistics result lists its choices (covariance
model, variogram fit, nugget, kriging type, simulation method, number of
realizations, seed), each marked given or default, through the agent's own
call path."""

import pytest

pytest.importorskip("gstools")


@pytest.fixture(autouse=True)
def _outputs_in_tmp_path(tmp_path, monkeypatch):
    """INF-55: the adapter writes its CSVs and figures into the working
    folder; point it at this test's tmp_path so nothing lands in the repo."""
    monkeypatch.setenv("GEOTECH_DEFAULT_OUTPUT_DIR", str(tmp_path))

from funhouse_agent import dispatch as D

X = [0, 10, 20, 30, 40, 5, 15, 25]
Y = [0, 5, 10, 0, 10, 20, 25, 15]
V = [1, 2, 3, 2, 4, 3, 5, 4]
OBS = {"x": X, "y": Y, "values": V}
REGION = [{"name": "A", "polygon": [[0, 0], [30, 0], [30, 20], [0, 20]],
           "level": 3.0}]


def by(r):
    return {c["what"]: c for c in r["choices"]}


def test_kriging_default_versus_given():
    r = D.call_agent("gstools", "kriging", {**OBS, "figures": False})
    c = by(r)
    assert c["covariance model"]["how"] == "default"
    assert c["kriging type"]["how"] == "default"
    assert c["variogram"]["how"] == "default"
    assert "fitted" in c["variogram"]["chosen"]
    assert c["nugget"]["chosen"].startswith("fitted")
    assert "Exponential" in c["covariance model"]["alternatives"]
    # given, even where it equals the default
    r = D.call_agent("gstools", "kriging", {
        **OBS, "figures": False, "model_type": "Gaussian",
        "kriging_type": "simple"})
    c = by(r)
    assert c["covariance model"]["how"] == "given"
    assert c["kriging type"] == {**c["kriging type"], "how": "given",
                                 "chosen": "simple"}


def test_kriging_without_a_fit_states_the_values_and_the_held_nugget():
    r = D.call_agent("gstools", "kriging", {
        **OBS, "figures": False, "fit_variogram": False, "variance": 2.0})
    c = by(r)
    assert c["variogram"]["chosen"].startswith("not fitted")
    assert c["variogram"]["how"] == "given"
    assert c["variance (sill)"]["how"] == "given"
    assert c["correlation length"]["how"] == "module"
    assert c["nugget"]["how"] == "default" and c["nugget"]["chosen"] == "0"
    for name in ("nugget",):
        assert name in r["defaults_used"]
    r = D.call_agent("gstools", "kriging", {
        **OBS, "figures": False, "fit_nugget": False, "nugget": 0.1})
    c = by(r)
    assert "held at 0.1" in c["variogram"]["chosen"]
    assert c["nugget"]["how"] == "given"


def test_variogram_nugget_fitted_or_held():
    r = D.call_agent("gstools", "variogram", dict(OBS))
    assert by(r)["nugget"]["chosen"].startswith("fitted")
    assert by(r)["covariance model"]["how"] == "default"
    r = D.call_agent("gstools", "variogram", {**OBS, "nugget": 0.0,
                                              "model_type": "Spherical"})
    c = by(r)
    assert c["nugget"]["how"] == "given" and c["nugget"]["chosen"] == "0"
    assert c["covariance model"] == {**c["covariance model"],
                                     "how": "given", "chosen": "Spherical"}


def test_random_field_states_every_value_and_the_seed():
    r = D.call_agent("gstools", "random_field",
                     {"n_x": 8, "n_y": 8, "seed": 7, "variance": 2.0})
    c = by(r)
    assert c["random seed"]["how"] == "given"
    assert c["variance (sill)"]["how"] == "given"
    assert c["correlation length"]["how"] == "default"
    assert c["mean"]["how"] == "default"
    r = D.call_agent("gstools", "random_field", {"n_x": 8, "n_y": 8})
    assert by(r)["random seed"]["how"] == "default"
    assert "defaults_used" in r and "seed" in r["defaults_used"]


def test_conditional_simulation_states_method_count_and_seed():
    base = {**OBS, "regions": REGION, "n_realizations": 20, "figures": False}
    r = D.call_agent("gstools", "conditional_simulation", base)
    c = by(r)
    assert c["simulation method"]["how"] == "default"
    assert c["simulation method"]["chosen"] == "kriging_residual"
    assert "condsrf" in c["simulation method"]["alternatives"]
    assert c["number of realizations"]["how"] == "given"
    assert c["random seed"]["how"] == "default"
    assert c["variogram"]["how"] == "default"
    assert c["variogram"]["chosen"].startswith("not fitted")
    assert c["nugget"]["how"] == "module"
    assert c["variance (sill)"]["how"] == "module"
    r = D.call_agent("gstools", "conditional_simulation", {
        **base, "seed": 42, "simulation_method": "condsrf",
        "nugget": 0.0, "variance": 1.5, "len_scale": 20.0})
    c = by(r)
    assert c["random seed"]["how"] == "given"
    assert c["simulation method"]["how"] == "given"
    assert c["nugget"]["how"] == "given"
    assert c["correlation length"]["how"] == "given"
    for name in ("seed", "n_realizations", "kriging_type"):
        assert (name in r["defaults_used"]) == (
            c[{"seed": "random seed", "n_realizations":
               "number of realizations",
               "kriging_type": "kriging type"}[name]]["how"] == "default")
