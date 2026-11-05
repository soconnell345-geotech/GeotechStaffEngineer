"""BACKLOG GM-113, the rest of the refusal hints after GM-110: a list
item's key given at the top level names its list; a did-you-mean finds the
leading words of a longer name; the inputs METHOD_INFO declared that no
handler read are refused with the reason; yield_acceleration refuses kh,
drilled_shaft refuses N1_60 and OCR under beta_method 'depth'. Made-up
inputs."""

import pytest

from funhouse_agent import dispatch as D
from funhouse_agent.adapters import suggest_params

_SLOPE = {"surface_points": [[0, 10], [20, 10], [40, 0], [60, 0]],
          "soil_layers": [{"top_elevation": 10, "bottom_elevation": -10,
                           "gamma": 19, "phi": 30, "c_prime": 5}]}


# -- 1. a list item's key at the top level -----------------------------------

def test_a_layer_key_at_the_top_level_names_the_layer_list():
    out = D.call_agent("slope_stability", "analyze_slope",
                       {**_SLOPE, "xc": 30, "yc": 25, "radius": 22,
                        "ru": 0.2})
    assert "unknown parameter(s) ['ru']" in out["error"]
    assert ("'ru' is a key of each item of soil_layers" in out["error"]
            and "soil_layers[i].ru" in out["error"])


def test_a_top_level_parameter_of_the_same_name_is_not_refused():
    # infinite_slope_fos takes ru at the top level
    info = D._load_adapter("slope_stability").METHOD_INFO
    assert "ru" in info["infinite_slope_fos"]["parameters"]


def test_the_hint_rides_with_a_did_you_mean_and_other_unknowns():
    out = D.call_agent("axial_pile", "axial_pile_capacity",
                       {"pile_type": "pipe", "pile_length": 10,
                        "layers": [], "friction_angle": 30, "bogus": 1})
    assert "'friction_angle' is a key of each item of layers" in out["error"]
    assert "'bogus' is a key" not in out["error"]


def test_every_item_keys_entry_names_a_declared_list_parameter():
    for agent in D.MODULE_REGISTRY:
        mod = D._load_adapter(agent)
        for method, params in (getattr(mod, "ITEM_KEYS", None) or {}).items():
            declared = mod.METHOD_INFO[method]["parameters"]
            for param, keys in params.items():
                assert param in declared and keys, (agent, method, param)


# -- 2. prefix matches -------------------------------------------------------

def test_the_leading_words_of_a_longer_name_are_offered():
    assert suggest_params("parameter", ["color_by", "parameter_for_color",
                                        "label_field"]) == [
        "parameter_for_color"]
    out = D.call_agent("subsurface", "plot_plan_view", {"parameter": "N"})
    assert "parameter -> parameter_for_color" in out["error"]


@pytest.mark.parametrize("unknown, valid", [
    ("E", ["E_s", "E_p"]),               # too short to be a word's guess
    ("para", ["parameter_for_color"]),   # not a whole word
    ("base", ["base_tilt"]),             # four letters, whole word: offered
])
def test_only_whole_words_of_four_letters_or_more(unknown, valid):
    expected = valid[:1] if unknown == "base" else []
    assert suggest_params(unknown, valid) == expected


# -- 3. declared but never read: fem2d_local_fos -----------------------------

_FIXED = ("gamma_w", "max_iter", "tol", "t", "n_gp")


def test_local_fos_no_longer_advertises_what_it_never_passed():
    info = D._load_adapter("fem2d").METHOD_INFO["fem2d_local_fos"]
    assert not set(_FIXED) & set(info["parameters"])
    assert set(_FIXED) == set(info["refused_parameters"])


@pytest.mark.parametrize("name, value", [
    ("gamma_w", 10.0), ("max_iter", 50), ("tol", 1e-3), ("t", 2.0),
    ("n_gp", 6)])
def test_local_fos_refuses_them_with_the_reason(name, value):
    out = D.call_agent("fem2d", "fem2d_local_fos",
                       {"surface_points": [[0, 10], [10, 10], [20, 0],
                                           [30, 0]],
                        "soil_layers": [{"bottom_elevation": -5}],
                        name: value})
    assert f"fem2d_local_fos does not take ['{name}']" in out["error"]
    assert "fem2d_slope_srm takes it" in out["error"]
    # a direct handler call says the same
    with pytest.raises(ValueError, match="does not take"):
        D._load_adapter("fem2d").METHOD_REGISTRY["fem2d_local_fos"](
            {"surface_points": [[0, 10], [30, 0]], "soil_layers": [],
             name: value})


# -- 4. yield_acceleration's kh ----------------------------------------------

_CIRCLE = {**_SLOPE, "xc": 30, "yc": 25, "radius": 22}


def test_yield_acceleration_refuses_a_kh_it_would_ignore():
    out = D.call_agent("slope_stability", "yield_acceleration",
                       {**_CIRCLE, "kh": 0.1})
    assert "yield_acceleration does not take ['kh']" in out["error"]
    assert "finds kh itself" in out["error"]
    info = D._load_adapter("slope_stability").METHOD_INFO
    assert "kh" not in info["yield_acceleration"]["parameters"]
    assert "kh" in info["analyze_slope"]["parameters"]


def test_yield_acceleration_takes_a_kh_of_zero_and_gives_the_same_ky():
    plain = D.call_agent("slope_stability", "yield_acceleration", _CIRCLE)
    zero = D.call_agent("slope_stability", "yield_acceleration",
                        {**_CIRCLE, "kh": 0})
    assert "error" not in plain and zero["ky"] == plain["ky"]


# -- 5. drilled_shaft N1_60 / OCR under the depth beta -----------------------

_SHAFT = {"diameter": 1.0, "shaft_length": 12.0, "layers": [
    {"thickness": 6, "soil_type": "cohesionless", "unit_weight": 19,
     "phi": 32, "N60": 20},
    {"thickness": 10, "soil_type": "cohesionless", "unit_weight": 19,
     "phi": 34, "N60": 30}]}


@pytest.mark.parametrize("key", ["N1_60", "N160", "OCR"])
@pytest.mark.parametrize("method", ["drilled_shaft_capacity",
                                    "capacity_vs_depth", "lrfd_capacity"])
def test_depth_beta_refuses_n1_60_and_ocr(method, key):
    import copy
    p = copy.deepcopy(_SHAFT)
    p["beta_method"] = "depth"
    p["layers"][0][key] = 2.0 if key == "OCR" else 18.0
    out = D.call_agent("drilled_shaft", method, p)
    assert "beta_method 'depth' does not take" in out.get("error", ""), out
