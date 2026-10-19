"""The calc package keeps what was given apart from what was assumed
(backlog GM-155): a "Given Inputs" section, then "Assumed Inputs", one row
per input that was not given -- the analysis's defaults and the caller's
own assumptions -- each with the value used and its basis. The module's
echo, which mixes given, assumed and derived values, follows as "Values
Used".

Live wave w12: two calc PDFs printed an assumed concrete unit weight and
the required factors of safety as "Given"."""

from calc_package import generate_calc_package, render_latex
from calc_package.data_model import CalcPackageData, TableData
from calc_package.tables import assumed_section, given_section

_ROWS = [
    {"input": "Concrete unit weight (kN/m3)", "value": 24.0,
     "basis": "the analysis's default"},
    {"input": "Foundation soil friction angle (deg)", "value": 32,
     "basis": "not given; taken equal to the backfill"},
    {"input": "Include a shear key", "value": False,
     "basis": "the analysis's default"},
]


def _bearing():
    from bearing_capacity import BearingCapacityAnalysis, Footing
    from bearing_capacity.soil_profile import BearingSoilProfile, SoilLayer
    analysis = BearingCapacityAnalysis(
        footing=Footing(width=2.0, depth=1.0, shape="square"),
        soil=BearingSoilProfile(layer1=SoilLayer(friction_angle=32,
                                                 unit_weight=19)),
        factor_of_safety=3.0)
    return analysis.compute(), analysis


def test_section_lists_each_assumed_input():
    section = assumed_section(_ROWS)
    assert section.title == "Assumed Inputs"
    intro, table = section.items
    assert intro.startswith("Assumed, not given:")
    assert "engineer's to accept or change" in intro
    assert isinstance(table, TableData)
    assert table.headers == ["Input", "Value used", "Basis"]
    assert table.rows == [
        ["Concrete unit weight (kN/m3)", "24", "the analysis's default"],
        ["Foundation soil friction angle (deg)", "32",
         "not given; taken equal to the backfill"],
        ["Include a shear key", "no", "the analysis's default"]]


def test_nothing_assumed_no_section():
    assert assumed_section(None) is None
    assert assumed_section([]) is None
    assert assumed_section([{"input": "", "value": 1}, "not a row"]) is None
    result, analysis = _bearing()
    html = generate_calc_package("bearing_capacity", result, analysis)
    assert "Assumed Inputs" not in html


def test_package_prints_given_then_assumed_then_values_used():
    result, analysis = _bearing()
    plain = generate_calc_package("bearing_capacity", result, analysis)
    html = generate_calc_package(
        "bearing_capacity", result, analysis, assumed_inputs=_ROWS,
        given_inputs=[{"input": "Footing width B (m)", "value": 2.0}])
    assert html.index("Given Inputs") < html.index("Footing width B (m)") \
        < html.index("Assumed Inputs") \
        < html.index("taken equal to the backfill") \
        < html.index("Values Used") \
        < html.index("Bearing Capacity Factor N_q")
    # the module's echo is retitled, its rows unchanged
    assert "Input Parameters" not in plain and "Input Parameters" not in html
    start = plain.index("Values Used")
    end = plain.index("</table>", start)
    assert plain[start:end] in html
    data = CalcPackageData(sections=[assumed_section(_ROWS)])
    latex = render_latex(data)
    assert "Assumed Inputs" in latex and "taken equal to the backfill" in latex


def test_given_section_lists_each_given_input():
    section = given_section([{"input": "Footing width B (m)", "value": 2.0},
                             {"input": "Layers", "value": [{"top": 0}]},
                             {"input": "", "value": 1}, "not a row"])
    assert section.title == "Given Inputs"
    intro, table = section.items
    assert intro.startswith("The inputs supplied for this calculation")
    assert table.headers == ["Input", "Value"]
    assert table.rows == [["Footing width B (m)", "2"],
                          ["Layers", '[{"top": 0}]']]
    assert given_section(None) is None and given_section([]) is None


def test_values_print_as_the_package_prints_cells():
    section = assumed_section([
        {"input": "a", "value": [1, 2]}, {"input": "b", "value": None},
        {"input": "c", "value": "rankine"}, {"input": "d", "value": 0.0}])
    _, table = section.items
    assert [r[1] for r in table.rows] == ["[1, 2]", "", "rankine", "0"]
    assert [r[2] for r in table.rows] == ["", "", "", ""]
