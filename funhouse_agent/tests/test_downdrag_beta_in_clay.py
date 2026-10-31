"""The downdrag tools and the Beta method on a cohesive layer (BACKLOG
GM-249): in a live test (w15-model-split, F67) two helpers passed ``beta`` on
a clay layer, the tool used alpha x cu and dropped it without a word, and
they rebuilt the layers by hand. The tool now refuses that ``beta``, naming
the input that takes effect, and takes the Beta method as a named option."""

import json

from funhouse_agent import dispatch as D

_F67 = {"pile_length": 22.0, "pile_diameter": 0.4, "pile_perimeter": 1.6,
        "pile_area": 0.16, "Q_dead": 800.0, "gwt_depth": 2.0,
        "fill_thickness": 2.5, "fill_unit_weight": 20.0}


def _layers(**clay):
    return [{"thickness": 2, "soil_type": "cohesionless", "unit_weight": 18,
             "phi": 32},
            {"thickness": 12, "soil_type": "cohesive", "unit_weight": 16,
             "cu": 25, "Cc": 0.6, "e0": 1.4, "settling": True, **clay},
            {"thickness": 15, "soil_type": "cohesionless", "unit_weight": 20,
             "phi": 38}]


def _run(method="downdrag_analysis", **clay):
    return D.call_agent("downdrag", method,
                        {**_F67, "layers": _layers(**clay)})


def _friction(r):
    return [c for c in r["choices"]
            if c["what"].startswith("skin friction method")]


def test_a_beta_the_tool_would_ignore_is_refused():
    err = _run(beta=0.25)["error"]
    for words in ("layers[1]", "does not take beta", "alpha x cu",
                  "skin_friction_method 'beta'", "Table 7-9"):
        assert words in err, words


def test_the_beta_method_is_taken_by_name():
    r = _run(skin_friction_method="beta", beta=0.25)
    assert "error" not in r, r
    # 654.6 until GM-283: the top sand's beta 0.294 -> 0.300 (Table 7-9)
    assert round(r["drag_force_kN"], 1) == 656.0
    (c,) = _friction(r)
    assert c["how"] == "given" and c["chosen"].endswith("beta 0.25")
    default = _run()
    assert round(default["drag_force_kN"], 1) == 545.1
    (c,) = _friction(default)
    assert c["how"] == "default" and c["chosen"].startswith("Alpha method")


def test_the_post_liquefaction_tool_takes_it_too():
    zone = [{"top": 4, "bottom": 8, "volumetric_strain": 0.02}]
    r = D.call_agent("downdrag", "post_liquefaction_downdrag", {
        **_F67, "layers": _layers(skin_friction_method="beta", beta=0.25),
        "liquefied_layers": zone})
    assert "error" not in r, r
    assert [c["how"] for c in _friction(r)] == ["given"]
    assert not any("lies in a cohesive layer (alpha method)" in w
                   for w in r["warnings"])


def test_the_descriptions_name_the_option():
    for method in ("downdrag_analysis", "post_liquefaction_downdrag"):
        info = D.describe_method("downdrag", method)
        layers = json.dumps(info["parameters"]["layers"])
        for words in ("skin_friction_method 'beta'", "Table 7-9",
                      "A cohesive layer takes no phi"):
            assert words in layers, (method, words)
        assert "skin friction method of the cohesive layers" in \
            info["returns"]["choices"]
