"""What a lateral pile result says it used (backlog GM-115): each layer's
p-y model with its source, and the loading."""

from lateral_pile import LateralPileAnalysis, Pile, SoilLayer
from lateral_pile.py_curves import SandReese, SoftClayMatlock


def _solve(sand_loading="static"):
    pile = Pile(length=20.0, diameter=0.6, thickness=0.012, E=200e6)
    layers = [
        SoilLayer(top=0.0, bottom=5.0,
                  py_model=SoftClayMatlock(c=25.0, gamma=8.0, eps50=0.02)),
        SoilLayer(top=5.0, bottom=20.0,
                  py_model=SandReese(phi=35.0, gamma=10.0, k=16000,
                                     loading=sand_loading)),
    ]
    return LateralPileAnalysis(pile, layers).solve(Vt=100.0)


def test_each_layer_names_its_model_and_source():
    r = _solve()
    clay, sand, loading = r.choices
    assert clay["what"] == "p-y curves 0-5 m"
    assert clay["chosen"] == "SoftClayMatlock" and clay["how"] == "given"
    assert "Matlock (1970)" in clay["basis"] and "J = 0.5" in clay["basis"]
    assert "SoftClayJeanjean" in clay["alternatives"]
    assert sand["chosen"] == "SandReese"
    assert "construction simplified" in sand["basis"]
    assert sand["alternatives"] == ["SandAPI"]
    assert loading == {**loading, "chosen": "static, every layer",
                       "how": "default", "alternatives": ["cyclic"]}
    assert r.to_dict()["choices"] == r.choices
    assert "Choices:" in r.summary()


def test_cyclic_loading_is_given_by_depth():
    loading = _solve("cyclic").choices[-1]
    assert loading["how"] == "given"
    assert loading["chosen"] == "static 0-5 m; cyclic 5-20 m"
