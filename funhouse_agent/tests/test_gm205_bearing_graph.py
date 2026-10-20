"""GM-205: the wave equation package's bearing_graph reads as the PDF's
table does: a blow count of 1e5 per m or more is "refusal"."""

from types import SimpleNamespace

import numpy as np

from funhouse_agent.adapters.calc_package import (
    _MAX_LISTED_POINTS, _bearing_graph_numbers)


def _result(n, refusal_from=None):
    blows = np.linspace(10.0, 500.0, n)
    if refusal_from is not None:
        blows[refusal_from:] = 1e6
    return SimpleNamespace(
        R_values=np.linspace(200.0, 2000.0, n), blow_counts=blows,
        permanent_sets=np.full(n, 1e-3), max_comp_stresses=np.full(n, 3e5),
        max_tens_stresses=np.zeros(n))


def test_refusal_point_is_named_and_its_set_is_a_number():
    rows = _bearing_graph_numbers(_result(5, refusal_from=3))["bearing_graph"]
    assert [r["blow_count_per_m"] == "refusal" for r in rows] == [
        False, False, False, True, True]
    assert rows[4]["permanent_set_mm"] == 1.0
    assert isinstance(rows[0]["blow_count_per_m"], float)


def test_thinning_is_kept_with_refusal_points():
    out = _bearing_graph_numbers(_result(91, refusal_from=80))
    assert len(out["bearing_graph"]) == _MAX_LISTED_POINTS
    assert out["bearing_graph"][-1]["blow_count_per_m"] == "refusal"
    assert "of the 91 points" in out["bearing_graph_note"]
