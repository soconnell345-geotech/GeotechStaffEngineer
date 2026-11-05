"""BACKLOG GM-113, the rest of the refusal hints after GM-110: a list
item's key given at the top level names its list; a did-you-mean finds the
leading words of a longer name; the inputs METHOD_INFO declared that no
handler read (fem2d_local_fos's) now reach the analysis; yield_acceleration
refuses kh; drilled_shaft takes N1_60 and OCR under beta_method 'depth' for
the result's rational alternative. Made-up inputs."""

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

_PASSED = ("t", "gamma_w", "max_iter", "tol", "n_gp")
_LOCAL_SLOPE = {"surface_points": [[0, 10], [10, 10], [20, 0], [30, 0]],
                "soil_layers": [{"name": "clay", "bottom_elevation": -5,
                                 "E": 20000, "nu": 0.3, "c": 10, "phi": 25,
                                 "psi": 0, "gamma": 19}]}


@pytest.fixture
def srm_kwargs(monkeypatch):
    """Replace analyze_slope_srm and keep what the handler passes it."""
    import fem2d
    seen = {}

    class _LocalFos:
        def to_dict(self):
            return {"min_local_fos": 1.2}

    class _Result:
        local_fos = _LocalFos()
        FOS = 1.2
        fos_basis = "nonconvergence"
        converged = True
        warnings = ()

    def fake(**kwargs):
        seen.update(kwargs)
        return _Result()

    monkeypatch.setattr(fem2d, "analyze_slope_srm", fake)
    return seen


def test_local_fos_passes_the_five_to_the_analysis(srm_kwargs):
    given = {"t": 2.0, "gamma_w": 10.0, "max_iter": 50, "tol": 1e-3,
             "n_gp": 6}
    out = D.call_agent("fem2d", "fem2d_local_fos", {**_LOCAL_SLOPE, **given})
    assert "error" not in out, out
    assert {n: srm_kwargs[n] for n in _PASSED} == given
    assert srm_kwargs["compute_local_fos"] is True


def test_local_fos_without_them_passes_the_analysis_defaults(srm_kwargs):
    import inspect

    from fem2d.analysis import analyze_slope_srm
    out = D.call_agent("fem2d", "fem2d_local_fos", dict(_LOCAL_SLOPE))
    assert "error" not in out, out
    own = inspect.signature(analyze_slope_srm).parameters
    assert {n: srm_kwargs[n] for n in _PASSED} == {
        n: own[n].default for n in _PASSED}


def test_local_fos_refuses_a_gauss_rule_it_has_not(srm_kwargs):
    out = D.call_agent("fem2d", "fem2d_local_fos", {**_LOCAL_SLOPE,
                                                    "n_gp": 4})
    assert "n_gp" in out["error"] and not srm_kwargs


def test_local_fos_describes_the_five_as_fem2d_slope_srm_does():
    info = D._load_adapter("fem2d").METHOD_INFO
    local = info["fem2d_local_fos"]
    assert "refused_parameters" not in local
    for n in _PASSED:
        assert local["parameters"][n] == info["fem2d_slope_srm"][
            "parameters"][n], n


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


def _depth_run(**first):
    import copy
    p = copy.deepcopy(_SHAFT)
    p["beta_method"] = "depth"
    p["layers"][0].update(first)
    out = D.call_agent("drilled_shaft", "drilled_shaft_capacity", p)
    assert "error" not in out, out
    rational = [a for c in out["choices"] for a in c.get("alternatives", [])
                if str(a).startswith("beta_method 'rational'")]
    assert len(rational) == 1, rational
    return out["Q_ultimate_kN"], rational[0]


@pytest.mark.parametrize("first", [{"OCR": 2.0},
                                   {"phi": 0, "N1_60": 18.0}])
def test_depth_beta_takes_n1_60_and_ocr_for_the_rational_alternative(first):
    q_plain, alt_plain = _depth_run()
    q_given, alt_given = _depth_run(**first)
    assert q_given == q_plain      # the depth beta uses neither
    assert alt_given != alt_plain  # the rational alternative does


def test_the_layers_text_says_where_n1_60_and_ocr_enter():
    desc = D._load_adapter("drilled_shaft").METHOD_INFO[
        "drilled_shaft_capacity"]["parameters"]["layers"]["description"]
    assert ("under beta_method 'depth' the depth beta uses neither N1_60 "
            "nor OCR: they enter only the result's beta_method 'rational' "
            "alternative (choices)") in desc
    assert "is refused: that beta does not use them" not in desc
