"""BACKLOG GM-237 through the agent's own call path: a clay layer's
site-specific alpha (or Ca) with its basis and its own highly
overconsolidated call reach the module, are stated in choices, and are
refused where they are incomplete. Values are made up."""

from funhouse_agent import dispatch as D

_BASIS = "made-up site load test"


def _layers(**lower):
    return [{"thickness": 4, "soil_type": "cohesive", "unit_weight": 17.5,
             "cohesion": 25},
            {"thickness": 12, "soil_type": "cohesive", "unit_weight": 19.5,
             "cohesion": 55, **lower}]


def _call(layers, **extra):
    return D.call_agent("axial_pile", "axial_pile_capacity", {
        "pile_type": "concrete_square", "width": 0.3, "pile_length": 13,
        "layers": layers, **extra})


def _by_what(r):
    return {c["what"]: c for c in r["choices"]}


def test_a_site_alpha_reaches_the_module_and_its_choices():
    chart = _call(_layers())
    site = _call(_layers(alpha=0.8, adhesion_basis=_BASIS))
    assert "error" not in site, site
    assert site["Q_skin_kN"] < chart["Q_skin_kN"]
    c = _by_what(site)["site-specific adhesion (alpha method)"]
    assert c["how"] == "given" and _BASIS in c["chosen"]


def test_a_site_alpha_without_its_basis_is_refused():
    r = _call(_layers(alpha=0.8))
    assert "error" in r and "adhesion_basis" in r["error"]


def test_a_layers_highly_overconsolidated_call_reaches_the_module():
    stiff = [{"thickness": 15, "soil_type": "cohesive", "unit_weight": 20,
              "cohesion": 200}]
    refused = D.call_agent("axial_pile", "axial_pile_capacity", {
        "pile_type": "pipe_closed", "diameter": 0.4, "wall_thickness": 0.0127,
        "pile_length": 12, "layers": stiff})
    assert "error" in refused
    stiff[0]["highly_overconsolidated"] = True
    r = D.call_agent("axial_pile", "axial_pile_capacity", {
        "pile_type": "pipe_closed", "diameter": 0.4, "wall_thickness": 0.0127,
        "pile_length": 12, "layers": stiff})
    assert "error" not in r, r
    assert _by_what(r)["highly overconsolidated clay (alpha method)"][
        "how"] == "given"


def test_fig_7_18_asks_for_the_case_under_a_softer_clay():
    r = _call(_layers(), adhesion_chart="fig_7_18")
    assert "error" in r and "fig_7_18_case" in r["error"]
    ok = _call(_layers(fig_7_18_case="soft_clay_over_stiff_clay"),
               adhesion_chart="fig_7_18")
    assert "error" not in ok, ok
