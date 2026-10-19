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
#: none may appear under Given Inputs.
_WALL_DEFAULTS = ("Concrete unit weight", "Required sliding FOS",
                  "Required overturning FOS", "Earth pressure theory",
                  "rankine")
_BEARING = {"width": 2.0, "depth": 1.0, "friction_angle": 32,
            "unit_weight": 19}
_BEARING_DEFAULTS = ("Soil cohesion", "Ngamma", "vesic", "Factor of safety")


def _section(text: str, title: str, end: str) -> str:
    start = text.index(title)
    return text[start:text.index(end, start)]


def test_defaults_used_print_as_assumed(tmp_path):
    out = str(tmp_path / "wall.html")
    r = call_agent("calc_package", "retaining_wall_package",
                   dict(_WALL, output_path=out))
    assert r["status"] == "success", r
    used = r["defaults_used"]
    assert used["gamma_concrete"] == 24.0
    assert used["FOS_sliding_required"] == 1.5
    text = _text(out)
    assumed = _section(text, "Assumed Inputs", _NEXT)
    assert assumed.startswith("Assumed Inputs Assumed, not given:")
    for words, value in (("Concrete unit weight (kN/m3)", "24"),
                         ("Required sliding FOS", "1.5"),
                         ("Required overturning FOS", "2"),
                         ("Earth pressure theory", "rankine")):
        assert f"{words} {value} the analysis's default" in assumed, words
    # a default whose basis the result's choices give points there
    assert "its basis is under Method Choices" in assumed
    # one row per default in force, nothing given among them
    assert len(re.findall(r"the analysis's default(?!s)", assumed)) == \
        len(used)
    for given in ("Wall height", "Backfill unit weight"):
        assert given not in assumed
    # given, then assumed, then every value used, then the choices
    assert text.index("Given Inputs") < text.index("Assumed Inputs")         < text.index("Values Used") < text.index(
            "Method Choices The choices this analysis made")
    assert "Input Parameters" not in text


@pytest.mark.parametrize("package, inputs, given, defaults", [
    ("retaining_wall_package", _WALL,
     ("The wall to analyse and report cantilever", "Base width (m) 2.8",
      "Backfill unit weight (kN/m3) 19",
      "Foundation soil friction angle (deg), FULL value 32"),
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
    r = call_agent("calc_package", package, dict(inputs, output_path=out))
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
        assumptions={"phi_foundation": "not given; taken equal to the "
                                       "backfill",
                     "gamma_concrete": "not given; normal-weight concrete"}))
    assert r["status"] == "success", r
    assumed = _section(_text(out), "Assumed Inputs", _NEXT)
    assert ("Foundation soil friction angle (deg), FULL value 32 not given; "
            "taken equal to the backfill") in assumed
    # a stated assumption replaces the default's row, keeping its value
    assert "Concrete unit weight (kN/m3) 24 not given; normal-weight " \
        "concrete" in assumed
    assert "Concrete unit weight (kN/m3) 24 the analysis's default" \
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
    assert "FULL value 32 stated as an assumption" in assumed


def test_assumptions_in_another_shape_are_refused(tmp_path):
    r = call_agent("calc_package", "retaining_wall_package", dict(
        _WALL, output_path=str(tmp_path / "x.html"), assumptions=3))
    assert "error" in r
    assert "assumptions: give {input name: its basis}" in r["error"]


def test_an_assumption_names_an_input_of_the_package(tmp_path):
    """{"gwt": "none"} printed a bare "gwt" row: a name that is not one of
    the package's inputs is refused before the run, with a did-you-mean
    where one is near (as an unknown input is, GM-110)."""
    out = tmp_path / "bc.html"
    r = call_agent("calc_package", "bearing_capacity_package", dict(
        _BEARING, output_path=str(out),
        assumptions={"water_table_depth": "none found"}))
    assert "error" in r and not out.exists()
    assert "bearing_capacity_package assumptions: unknown parameter(s) "         "['water_table_depth']" in r["error"]
    assert "water_table_depth -> gwt_depth" in r["error"]
    r = call_agent("calc_package", "retaining_wall_package", dict(
        _WALL, output_path=str(tmp_path / "w.html"),
        assumptions={"gwt": "none"}))
    assert "unknown parameter(s) ['gwt']" in r["error"]
    # the header fields are not inputs to assume
    r = call_agent("calc_package", "retaining_wall_package", dict(
        _WALL, output_path=str(tmp_path / "w.html"),
        assumptions={"project_name": "x"}))
    assert "unknown parameter(s) ['project_name']" in r["error"]


@pytest.mark.parametrize("package, extra", [
    ("slope_stability_package", {"xc": 5, "yc": 10, "radius": 8}),
    ("slope_report_package", {}),
])
def test_the_slope_packages_check_assumptions_before_running(
        tmp_path, package, extra):
    out = tmp_path / "s.html"
    r = call_agent("calc_package", package, dict(
        extra, surface_points=[[0, 0], [10, 5]], soil_layers=[],
        output_path=str(out), assumptions={"cohesoin": "not given"}))
    assert f"{package} assumptions: unknown parameter(s) ['cohesoin']"         in r["error"]
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
        "Uniform DEAD-load (permanent) surcharge on the backfill (kPa)")
