"""GM-318: p-y layers that start below the ground surface or stop above
the pile toe are refused (they ran with no soil resistance there and only a
Python warning); ``extend_last_layer`` carries the last layer on to the toe
and the result says so. A cracked-section EI iteration that does not
converge is flagged in the result. Made-up values."""

import warnings

import pytest

from funhouse_agent.adapters.lateral_pile import METHOD_REGISTRY
from lateral_pile import (LateralPileAnalysis, Pile,
                          ReinforcedConcreteSection, SoilLayer)
from lateral_pile.py_curves import SoftClayMatlock


def _clay():
    return SoftClayMatlock(c=40.0, gamma=8.0, eps50=0.01, J=0.5)


def _pile(length=15.0):
    return Pile(length=length, diameter=0.6, thickness=0.012, E=200e6)


def test_layers_starting_below_the_ground_are_refused():
    with pytest.raises(ValueError) as e:
        LateralPileAnalysis(_pile(), [SoilLayer(2.0, 15.0, _clay())])
    msg = str(e.value)
    assert "start at 2 m" in msg and "stickup" in msg
    # review B2: stickup is the scour model (soil removed); soil that stays
    # in place but resists nothing is a weak top layer, and the two differ
    assert "Sec. 6.7.5, p. 91" in msg and "differs" in msg
    assert "top 2 m from the ground surface as its own layer of " \
           "negligible strength" in msg
    # the extension is of the last layer only
    with pytest.raises(ValueError, match="start at 2 m"):
        LateralPileAnalysis(_pile(), [SoilLayer(2.0, 15.0, _clay())],
                            extend_last_layer=True)


def test_layers_stopping_above_the_toe_are_refused():
    with pytest.raises(ValueError) as e:
        LateralPileAnalysis(_pile(), [SoilLayer(0.0, 10.0, _clay())])
    msg = str(e.value)
    assert "stop at 10 m" in msg and "toe is at 15 m" in msg
    assert "extend_last_layer" in msg


def test_extension_equals_layers_given_to_the_toe():
    ext = LateralPileAnalysis(_pile(), [SoilLayer(0.0, 10.0, _clay())],
                              extend_last_layer=True).solve(Vt=80.0)
    full = LateralPileAnalysis(_pile(),
                               [SoilLayer(0.0, 15.0, _clay())]).solve(Vt=80.0)
    assert ext.y_top == pytest.approx(full.y_top)
    assert ext.max_moment == pytest.approx(full.max_moment)
    d = ext.to_dict()
    assert next(iter(d)) == "warnings"
    assert "last layer's p-y model was taken to go on" in d["warnings"][0]
    assert d["choices"][0]["what"] == "soil below the layers given"
    # review C6: the p-y line marks the carried-on part
    assert "p-y curves 0-15 m (given to 10 m, assumed below)" in [
        c["what"] for c in d["choices"]]
    assert "p-y curves 0-15 m" in [c["what"] for c in full.choices]
    assert full.warnings == [] and "warnings" not in full.to_dict()
    assert "WARNING" in ext.summary()


def test_tool_refuses_and_extends():
    params = {"pile_diameter": 0.6, "pile_length": 15.0,
              "pile_thickness": 0.012, "Vt": 80.0,
              "layers": [{"top": 0.0, "bottom": 10.0,
                          "model": "SoftClayMatlock", "c": 40.0,
                          "gamma": 8.0, "eps50": 0.01}]}
    run = METHOD_REGISTRY["lateral_pile_analysis"]
    with pytest.raises(ValueError, match="extend_last_layer"):
        run(params)
    out = run({**params, "extend_last_layer": True})
    assert "nobody explored" in out["warnings"][0]


def _rc_analysis():
    rc = ReinforcedConcreteSection(diameter=0.9, fc=35000.0, fy=420000.0,
                                   n_bars=12, bar_diameter=0.02546,
                                   cover=0.075)
    pile = Pile.from_rc_section(length=15.0, rc_section=rc)
    return LateralPileAnalysis(pile, [SoilLayer(0.0, 15.0, _clay())])


def test_cracked_ei_that_does_not_converge_is_flagged():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = _rc_analysis().solve(Vt=400.0, max_ei_iterations=1)
    assert r.ei_converged is False and r.ei_iterations == 1
    assert r.ei_last_change > r.ei_tolerance == 1e-3
    d = r.to_dict()
    assert d["ei_converged"] is False and d["ei_iterations"] == 1
    assert d["ei_last_change"] == r.ei_last_change
    assert "did not converge" in d["warnings"][0]
    assert "after 1 passes" in d["warnings"][0]
    assert "EI converged:         False" in r.summary()


def test_no_ei_passes_is_refused():
    """Review C4: 0 passes gave a last change in EI of nan."""
    with pytest.raises(ValueError, match="max_ei_iterations must be 1"):
        _rc_analysis().solve(Vt=400.0, max_ei_iterations=0)


def test_cracked_ei_that_converges_says_so():
    r = _rc_analysis().solve(Vt=400.0)
    assert r.ei_converged is True and r.warnings == []
    assert r.ei_last_change < r.ei_tolerance
    d = r.to_dict()
    assert d["ei_converged"] is True and "warnings" not in d


def test_a_pile_with_no_ei_iteration_carries_no_flag():
    r = LateralPileAnalysis(_pile(),
                            [SoilLayer(0.0, 15.0, _clay())]).solve(Vt=80.0)
    assert r.ei_converged is None and "ei_converged" not in r.to_dict()
