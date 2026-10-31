"""GM-155: a calc package keeps what was given apart from what was assumed.

Live wave w12 (F55 OFF, F55j): a wall calc printed the assumed concrete
unit weight and the required factors of safety as "Given"; F18j's brief
stated an assumed water table as found. A package already knows which
inputs its call left at the tool's default (``defaults_used``, GM-115 /
GM-158); it now prints the inputs given under "Given Inputs" and those
defaults under "Assumed Inputs", and a caller names its own assumptions
with ``assumptions`` ({input name: basis}), which move from Given to
Assumed with their basis. The module's echo, which mixes given, assumed
and derived values, is "Values Used". Pure rendering: no number changes.

GM-167 (waves w13, w13b): a method the agent chose, c = 0 and a height
read off a sheet printed as given; a refused assumption lost the valid
ones with it; and a left-out water table said nothing. Given Inputs now
holds only the inputs the call names in ``given`` (the user's own words
with them when the call has them); every other input passed prints under
Assumed Inputs as passed, not stated as given, so an input not known to
be the user's is assumed. An assumption refusal names only the bad keys;
a statement that is not an input ("drained") is its own row; and a
left-out input with no default prints as the case it stands for (GM-165:
"no water table").
"""

import re

import matplotlib
import pytest

matplotlib.use("Agg")

from funhouse_agent.dispatch import call_agent, describe_method  # noqa: E402
import funhouse_agent.adapters.calc_package as cp  # noqa: E402

_WALL = {"wall_type": "cantilever", "wall_height": 4.5,
         "gamma_backfill": 19, "phi_backfill": 32, "base_width": 2.8,
         "toe_length": 0.7, "stem_thickness_top": 0.35,
         "stem_thickness_base": 0.35, "base_thickness": 0.5,
         "surcharge": 10, "phi_foundation": 32}


def _text(path) -> str:
    html = open(path, encoding="utf-8").read()
    html = re.sub(r"<img[^>]*>|<style.*?</style>", " ", html, flags=re.S)
    return " ".join(re.sub(r"<[^>]+>", " ", html).split()).replace(
        "&#39;", "'")


#: The section after Assumed Inputs.
_NEXT = "Values Used"

#: The wall's and the bearing tool's defaults, as the packages print them:
#: none may appear under Given Inputs. (The earth pressure theory is no
#: longer among the wall's: at Coulomb's default wall friction, delta =
#: beta, its Ka is Rankine's, GM-288.)
_WALL_DEFAULTS = ("Concrete unit weight", "Required sliding FOS",
                  "Required overturning FOS")
_BEARING = {"width": 2.0, "depth": 1.0, "friction_angle": 32,
            "unit_weight": 19}
_BEARING_DEFAULTS = ("Soil cohesion", "Ngamma", "vesic", "Factor of safety")


def _section(text: str, title: str, end: str) -> str:
    start = text.index(title)
    return text[start:text.index(end, start)]


def test_defaults_used_print_as_assumed(tmp_path):
    out = str(tmp_path / "wall.html")
    r = call_agent("calc_package", "retaining_wall_package",
                   dict(_WALL, output_path=out, given=list(_WALL)))
    assert r["status"] == "success", r
    used = r["defaults_used"]
    assert used["gamma_concrete"] == 23.6
    assert used["FOS_sliding_required"] == 1.5
    text = _text(out)
    assumed = _section(text, "Assumed Inputs", _NEXT)
    assert assumed.startswith("Assumed Inputs Assumed, not given:")
    for words, value in (("Concrete unit weight (kN/m3)", "23.6"),
                         ("Required sliding FOS", "1.5"),
                         ("Required overturning FOS", "2")):
        assert f"{words} {value} the analysis's default" in assumed, words
    # a default whose basis the result's choices give points there
    assert "its basis is under Method Choices" in assumed
    # one row per default in force, nothing given among them; a left-out
    # input with no default is the case it stands for (GM-165)
    rows = assumed.split("Input Value used Basis", 1)[1]
    assert len(re.findall(r"the analysis's default(?!s)|not given: ",
                          rows)) == len(used)
    assert ("Allowable bearing pressure (kPa) bearing "
            "not checked not given: ") in rows
    for given in ("Wall height", "Backfill unit weight"):
        assert given not in assumed
    # given, then assumed, then every value used, then the choices
    assert text.index("Given Inputs") < text.index("Assumed Inputs")         < text.index("Values Used") < text.index(
            "Method Choices The choices this analysis made")
    assert "Input Parameters" not in text


@pytest.mark.parametrize("package, inputs, given, defaults", [
    ("retaining_wall_package", _WALL,
     ("Wall type cantilever", "Base width (m) 2.8",
      "Backfill unit weight (kN/m3) 19",
      "Foundation soil friction angle (deg) 32"),
     _WALL_DEFAULTS),
    ("bearing_capacity_package", _BEARING,
     ("Footing width B (m) 2", "Embedment depth Df (m) 1",
      "Friction angle phi (degrees) 32", "Soil unit weight gamma (kN/m3) 19"),
     _BEARING_DEFAULTS),
])
def test_given_inputs_hold_only_what_was_given(tmp_path, package, inputs,
                                               given, defaults):
    """w12 F55: the echo printed concrete 24, Rankine and FOS 1.5 / 2.0
    among the inputs; the bearing echo c = 0 and N-gamma by Vesic. Given
    Inputs holds the call's own inputs, one row each, and no default."""
    out = str(tmp_path / "p.html")
    r = call_agent("calc_package", package, dict(inputs, output_path=out,
                                                 given=list(inputs)))
    assert r["status"] == "success", r
    text = _text(out)
    part = _section(text, "Given Inputs", "Assumed Inputs")
    for row in given:
        assert row in part, row
    for words in defaults:
        assert words not in part, words
        assert words in _section(text, "Assumed Inputs", _NEXT), words
    # one row per input the call gave
    rows = part.split("Input Value", 1)[1]
    assert len(re.findall(r"\s\d+(\.\d+)?(?=\s|$)", rows)) >= len(given)
    assert "the analysis's default" not in rows


def test_a_callers_assumptions_print_with_their_basis(tmp_path):
    out = str(tmp_path / "wall.html")
    r = call_agent("calc_package", "retaining_wall_package", dict(
        _WALL, output_path=out,
        given=[k for k in _WALL if k != "phi_foundation"],
        assumptions={"phi_foundation": "not given; taken equal to the "
                                       "backfill",
                     "gamma_concrete": "not given; normal-weight concrete"}))
    assert r["status"] == "success", r
    assumed = _section(_text(out), "Assumed Inputs", _NEXT)
    assert ("Foundation soil friction angle (deg) 32 not given; "
            "taken equal to the backfill") in assumed
    # a stated assumption replaces the default's row, keeping its value
    assert "Concrete unit weight (kN/m3) 23.6 not given; normal-weight " \
        "concrete" in assumed
    assert "Concrete unit weight (kN/m3) 23.6 the analysis's default" \
        not in assumed
    # the stated ones come first
    assert assumed.index("Foundation soil") < assumed.index("Required "
                                                            "sliding FOS")
    # and a stated assumption is no longer among the given
    given = _section(_text(out), "Given Inputs", "Assumed Inputs")
    assert "Foundation soil friction angle" not in given
    assert "Base width (m) 2.8" in given


def test_same_numbers_with_and_without_assumptions(tmp_path):
    a = call_agent("calc_package", "retaining_wall_package",
                   dict(_WALL, output_path=str(tmp_path / "a.html")))
    b = call_agent("calc_package", "retaining_wall_package", dict(
        _WALL, output_path=str(tmp_path / "b.html"),
        assumptions=["phi_foundation"]))
    for k in ("FOS_sliding", "FOS_overturning", "verdict", "defaults_used"):
        assert a[k] == b[k], k
    assumed = _section(_text(str(tmp_path / "b.html")), "Assumed Inputs",
                       _NEXT)
    assert "32 stated as an assumption" in assumed


def test_assumptions_in_another_shape_are_refused(tmp_path):
    r = call_agent("calc_package", "retaining_wall_package", dict(
        _WALL, output_path=str(tmp_path / "x.html"), assumptions=3))
    assert "error" in r
    assert "assumptions: give {input name: its basis}" in r["error"]


def test_an_assumption_written_as_a_name_names_an_input(tmp_path):
    """A key written as an input's name that is not one, a word that is a
    misspelt input and a header field are refused before the run, with a
    did-you-mean (as an unknown input is, GM-110); GM-167: only those keys
    are named, and the message says the others are fine to send again."""
    out = tmp_path / "bc.html"
    r = call_agent("calc_package", "bearing_capacity_package", dict(
        _BEARING, output_path=str(out),
        assumptions={"unit_weight": "typical of the sand described",
                     "gwt_dept": "none found", "cohesoin": "sand",
                     "project_name": "x"}))
    assert "error" in r and not out.exists()
    err = r["error"]
    assert "bearing_capacity_package assumptions: unknown parameter(s) " \
        "['cohesoin', 'gwt_dept']" in err
    assert "gwt_dept -> gwt_depth" in err and "cohesoin -> cohesion" in err
    assert "['project_name']: header fields, not inputs" in err
    assert "The others ('unit_weight') are fine: send them again" in err
    # a name the package reads by another spelling is that input
    r = call_agent("calc_package", "bearing_capacity_package", dict(
        _BEARING, output_path=str(out),
        assumptions={"water_table_depth": "none found in the borings"}))
    assert r["status"] == "success", r
    assert ("Groundwater depth below surface (m) no water table none found "
            "in the borings") in _section(_text(out), "Assumed Inputs",
                                          _NEXT)


def test_an_assumption_in_words_is_its_own_row(tmp_path):
    """GM-167: "groundwater", "drained" and "c = 0 for sand" are not
    inputs; each is a row of its own under Assumed Inputs, with its
    basis, and the run is the same."""
    out = str(tmp_path / "bc.html")
    plain = call_agent("calc_package", "bearing_capacity_package", dict(
        _BEARING, output_path=str(tmp_path / "plain.html")))
    r = call_agent("calc_package", "bearing_capacity_package", dict(
        _BEARING, output_path=out,
        assumptions={"groundwater": "none found in the borings",
                     "drained": "long-term condition",
                     "c = 0 for sand": "clean sand"}))
    assert r["status"] == "success", r
    for k in ("q_ultimate_kPa", "q_allowable_kPa", "defaults_used"):
        assert r[k] == plain[k], k
    assumed = _section(_text(out), "Assumed Inputs", _NEXT)
    for row in ("groundwater none found in the borings",
                "drained long-term condition", "c = 0 for sand clean sand"):
        assert row in assumed, row
    # a word on its own with no basis says it was stated
    r = call_agent("calc_package", "retaining_wall_package", dict(
        _WALL, output_path=str(tmp_path / "w.html"), assumptions=["gwt"]))
    assert r["status"] == "success", r
    assert "gwt stated as an assumption" in _section(
        _text(str(tmp_path / "w.html")), "Assumed Inputs", _NEXT)


def test_a_left_out_water_table_prints_as_no_water_table(tmp_path):
    """GM-167 (c), with GM-165: bearing capacity with no gwt_depth is
    analysed with no water table, and the package says so under Assumed
    Inputs with its basis; given, the depth prints under Given Inputs."""
    out = str(tmp_path / "bc.html")
    r = call_agent("calc_package", "bearing_capacity_package", dict(
        _BEARING, output_path=out))
    assert r["defaults_used"]["gwt_depth"].startswith("no water table: ")
    assumed = _section(_text(out), "Assumed Inputs", _NEXT)
    assert ("Groundwater depth below surface (m) no water table not given: "
            "total unit weight above and below the base") in assumed
    r = call_agent("calc_package", "bearing_capacity_package", dict(
        _BEARING, gwt_depth=1.5, given=["gwt_depth"], output_path=out))
    text = _text(out)
    assert "Groundwater depth below surface (m) 1.5" in _section(
        text, "Given Inputs", "Assumed Inputs")
    assert "no water table" not in text


def test_only_the_inputs_named_as_given_print_as_given(tmp_path):
    """GM-167 (a), waves w13 / w13b: the helper's Vesic, c = 0 and a
    wall height read off a sheet printed as given. Only the inputs the call
    names in ``given`` print under Given Inputs; every other input passed
    prints under Assumed Inputs as passed, not stated as given."""
    out = str(tmp_path / "bc.html")
    r = call_agent("calc_package", "bearing_capacity_package", dict(
        _BEARING, cohesion=0, factor_method="meyerhof", shape="square",
        given=["width", "depth", "friction_angle", "shape"],
        output_path=out))
    assert r["status"] == "success", r
    text = _text(out)
    given = _section(text, "Given Inputs", "Assumed Inputs")
    assumed = _section(text, "Assumed Inputs", _NEXT)
    assert given.startswith("Given Inputs The inputs the user gave")
    for row in ("Footing width B (m) 2", "Embedment depth Df (m) 1",
                "Friction angle phi (degrees) 32", "Footing shape square"):
        assert row in given, row
    for words in ("meyerhof", "Soil cohesion", "Soil unit weight"):
        assert words not in given, words
    for row in ("Soil unit weight gamma (kN/m3) 19 passed, not stated as "
                "given",
                "Soil cohesion c (kPa) 0 passed, not stated as given",
                "Shape/depth/inclination-factor method meyerhof passed, not "
                "stated as given"):
        assert row in assumed, row
    # the numbers are the same whoever gave the inputs
    r2 = call_agent("calc_package", "bearing_capacity_package", dict(
        _BEARING, cohesion=0, factor_method="meyerhof", shape="square",
        output_path=str(tmp_path / "none.html")))
    assert r2["q_ultimate_kPa"] == r["q_ultimate_kPa"]


def test_with_no_given_list_every_input_passed_is_assumed(tmp_path):
    """Not knowing whose an input is, the package takes it as assumed, and
    the Given Inputs section says no input was named as the user's."""
    out = str(tmp_path / "bc.html")
    r = call_agent("calc_package", "bearing_capacity_package", dict(
        _BEARING, output_path=out))
    assert r["status"] == "success", r
    text = _text(out)
    assert ("Given Inputs No input was named as given by the user: each "
            "input passed is listed under Assumed Inputs.") in text
    assumed = _section(text, "Assumed Inputs", _NEXT)
    for row in ("Footing width B (m) 2 passed, not stated as given",
                "Friction angle phi (degrees) 32 passed, not stated as "
                "given"):
        assert row in assumed, row


def test_given_with_the_users_words(tmp_path):
    """``given`` as {input: the user's own words} prints them beside the
    value."""
    out = str(tmp_path / "bc.html")
    r = call_agent("calc_package", "bearing_capacity_package", dict(
        _BEARING, output_path=out,
        given={"width": "a 2 m square footing",
               "friction_angle": "phi = 32 deg from the boring logs",
               "depth": ""}))
    assert r["status"] == "success", r
    given = _section(_text(out), "Given Inputs", "Assumed Inputs")
    assert "Input Value As the user gave it" in given
    assert "Footing width B (m) 2 a 2 m square footing" in given
    assert ("Friction angle phi (degrees) 32 phi = 32 deg from the boring "
            "logs") in given
    assert "Embedment depth Df (m) 1" in given
    assert "Soil unit weight" not in given


def test_a_chosen_method_reads_assumed_in_method_choices(tmp_path):
    """The module's own Method Choices table says "assumed" for an input
    the call passed and did not name as the user's, never "given"."""
    out = str(tmp_path / "w.html")
    row = ("live-load surcharge over the heel, in sliding, overturning and "
           "the eccentricity check not a resisting force")
    r = call_agent("calc_package", "retaining_wall_package", dict(
        _WALL, live_load_resists=False, output_path=out))
    assert r["status"] == "success", r
    assert f"{row} assumed FHWA" in _text(out)
    r = call_agent("calc_package", "retaining_wall_package", dict(
        _WALL, live_load_resists=False, given=["live_load_resists"],
        output_path=out))
    assert f"{row} given FHWA" in _text(out)
    # left out, it is the default, as before
    r = call_agent("calc_package", "retaining_wall_package", dict(
        _WALL, output_path=out))
    assert f"{row} default FHWA" in _text(out)


def test_given_names_inputs_of_the_package(tmp_path):
    out = tmp_path / "bc.html"
    r = call_agent("calc_package", "bearing_capacity_package", dict(
        _BEARING, factor_method="vesic", given=["width", "factor_metod"],
        output_path=str(out)))
    assert "given: unknown parameter(s) ['factor_metod']" in r["error"]
    assert "factor_metod -> factor_method" in r["error"]
    assert "The others ('width') are fine" in r["error"]
    assert not out.exists()
    r = call_agent("calc_package", "bearing_capacity_package", dict(
        _BEARING, factor_method="vesic", given=["factor_method"],
        assumptions={"factor_method": "x"}, output_path=str(out)))
    assert "in both assumptions and given" in r["error"]
    r = call_agent("calc_package", "bearing_capacity_package", dict(
        _BEARING, given=3, output_path=str(out)))
    assert "given: give the names of the inputs you pass" in r["error"]


def test_given_names_only_inputs_the_call_passes(tmp_path):
    """A name the call does not pass is refused (its value would print as
    the analysis's default, the user's words lost), and a refusal lists
    the inputs passed: symbols such as B or phi are not the bearing
    package's names."""
    out = tmp_path / "bc.html"
    r = call_agent("calc_package", "bearing_capacity_package", dict(
        _BEARING, given=["width", "factor_of_safety"], output_path=str(out)))
    assert "given: ['factor_of_safety'] named but not passed" in r["error"]
    assert ("The inputs this call passes: depth, friction_angle, "
            "unit_weight, width.") in r["error"]
    assert not out.exists()
    r = call_agent("calc_package", "bearing_capacity_package", dict(
        _BEARING, given=["B", "phi"], output_path=str(out)))
    assert "unknown parameter(s) ['B', 'phi']" in r["error"]
    assert "The inputs this call passes: depth, friction_angle" in r["error"]
    r = call_agent("calc_package", "bearing_capacity_package", dict(
        _BEARING, factor_of_safety=3.0, output_path=str(out),
        given={"width": "2 m", "factor_of_safety": "FS = 3"}))
    assert r["status"] == "success", r
    given = _section(_text(out), "Given Inputs", "Assumed Inputs")
    assert "Factor of safety 3 FS = 3" in given


def test_with_nothing_named_as_given_no_choice_reads_given(tmp_path):
    """A module that says "given" for a passed input by its own rule (the
    seismic wall friction delta) says "assumed" in a package whose call
    names no input as the user's; named, it says "given"."""
    params = {"analysis_type": "seismic_earth_pressure", "phi": 32,
              "kh": 0.2, "delta": 21, "gamma": 19, "H": 5}
    out = str(tmp_path / "s.html")
    r = call_agent("calc_package", "seismic_package",
                   dict(params, output_path=out))
    assert r["status"] == "success", r
    text = _text(out)
    assert "wall friction delta 21 deg assumed" in text
    assert "wall friction delta 21 deg given" not in text
    r = call_agent("calc_package", "seismic_package", dict(
        params, output_path=out, given=["delta"]))
    assert r["status"] == "success", r
    assert "wall friction delta 21 deg given" in _text(out)


@pytest.mark.parametrize("package, extra", [
    ("slope_stability_package", {"xc": 5, "yc": 10, "radius": 8}),
    ("slope_report_package", {}),
])
def test_the_slope_packages_check_assumptions_before_running(
        tmp_path, package, extra):
    out = tmp_path / "s.html"
    r = call_agent("calc_package", package, dict(
        extra, surface_points=[[0, 0], [10, 5]], soil_layers=[],
        output_path=str(out), assumptions={"gwt_pionts": "not given"}))
    assert f"{package} assumptions: unknown parameter(s) ['gwt_pionts']" \
        in r["error"]
    assert "gwt_pionts -> gwt_points" in r["error"]
    assert not out.exists()


def test_every_package_takes_assumptions_and_says_why():
    for name in cp.METHOD_INFO:
        if not name.endswith("_package"):
            continue
        spec = describe_method("calc_package", name)["parameters"]
        assert "assumptions" in spec, name
        assert spec["assumptions"]["type"] == "dict"
        assert "so no assumption reads as given" in \
            spec["assumptions"]["description"]
        assert spec["given"]["type"] == "list|dict", name
        assert "Only these print under Given Inputs" in \
            spec["given"]["description"]


def test_the_tool_is_not_handed_assumptions(tmp_path, monkeypatch):
    """``assumptions`` is the package's, never passed to its tool (which
    would refuse it as unknown)."""
    seen = {}
    real = cp._run_tool

    def spy(package, params):
        run = real(package, params)
        seen.update(run.params)
        return run
    monkeypatch.setattr(cp, "_run_tool", spy)
    r = call_agent("calc_package", "bearing_capacity_package", {
        "width": 2.0, "depth": 1.0, "friction_angle": 32,
        "unit_weight": 19, "output_path": str(tmp_path / "bc.html"),
        "assumptions": {"unit_weight": "not given"}})
    assert r["status"] == "success", r
    assert "assumptions" not in seen and seen["unit_weight"] == 19


@pytest.mark.parametrize("name, words", [
    ("cohesion", "Soil cohesion c (kPa)"),
    ("factor_method", "Shape/depth/inclination-factor method"),
    ("nothing_declared", "nothing_declared"),
])
def test_an_input_prints_in_plain_words(name, words):
    from funhouse_agent.adapters import bearing_capacity as bc
    declared = bc.METHOD_INFO["bearing_capacity_analysis"]["parameters"]
    assert cp._input_words(name, declared) == words


def test_a_long_description_is_cut_at_its_first_clause():
    declared = {"q": {"description": (
        "Uniform DEAD-load (permanent) surcharge on the backfill (kPa), e.g. "
        "a pavement (FHWA NHI-06-089 Sec. 10.4.2). Its thrust drives.")}}
    assert cp._input_words("q", declared) == (
        "Uniform dead-load (permanent) surcharge on the backfill (kPa)")
