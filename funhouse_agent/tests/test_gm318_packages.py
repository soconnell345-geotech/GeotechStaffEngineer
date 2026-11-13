"""GM-318: the pile calc packages say when the last layer was carried on
below the profile (a section of its own and ``warnings`` in the tool's
result), and refuse the uncovered pile as their tools do. Made-up values."""

import pytest

from funhouse_agent.adapters.calc_package import METHOD_REGISTRY

_SAND = {"thickness": 4.0, "soil_type": "cohesionless", "unit_weight": 18.0,
         "friction_angle": 30}
_CASES = {
    "axial_pile_package": {
        "pile_type": "pipe_closed", "diameter": 0.3239,
        "wall_thickness": 0.00953, "pile_length": 12.0,
        "layers": [_SAND, {"thickness": 6.0, "soil_type": "cohesive",
                           "unit_weight": 19.0, "cohesion": 60}]},
    "drilled_shaft_package": {
        "diameter": 1.0, "shaft_length": 15.0,
        "layers": [{"thickness": 5.0, "soil_type": "cohesive",
                    "unit_weight": 18.0, "cu": 60.0},
                   {"thickness": 7.0, "soil_type": "cohesive",
                    "unit_weight": 19.0, "cu": 90.0}]},
    "lateral_pile_package": {
        "pile_diameter": 0.6, "pile_length": 15.0, "pile_thickness": 0.012,
        "Vt": 80.0,
        "layers": [{"top": 0.0, "bottom": 10.0, "model": "SoftClayMatlock",
                    "c": 40.0, "gamma": 8.0, "eps50": 0.01}]},
}


@pytest.mark.parametrize("method", sorted(_CASES))
def test_package_refuses_then_says_the_layer_was_carried_on(method, tmp_path):
    out = tmp_path / "package.html"
    params = {**_CASES[method], "output_path": str(out), "format": "html"}
    with pytest.raises(ValueError, match="extend_last_layer"):
        METHOD_REGISTRY[method](params)
    r = METHOD_REGISTRY[method]({**params, "extend_last_layer": True})
    assert "nobody explored" in r["warnings"][0]
    html = out.read_text(encoding="utf-8")
    assert "Soil Below the Profile (Assumed)" in html
    assert "nobody explored" in html
    # review C6: the soil tables mark the carried-on part of the last layer
    mark = f"(given to {_GIVEN[method]} m, assumed below)"
    csv = "".join(p.read_text(encoding="utf-8-sig")
                  for p in tmp_path.glob("*soil_layers*.csv"))
    assert mark in csv
    if method != "drilled_shaft_package":    # its soil table is CSV only
        assert mark in html


_GIVEN = {"axial_pile_package": 10, "drilled_shaft_package": 12,
          "lateral_pile_package": 10}
