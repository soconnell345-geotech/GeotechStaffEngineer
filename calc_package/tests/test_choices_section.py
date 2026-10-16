"""The calc package prints what the analysis chose (backlog GM-115): a
"Method Choices" section beside the inputs, from the result's ``choices``
(or, for a package built on a search, the search's)."""

from calc_package import generate_calc_package, render_latex
from calc_package.data_model import CalcPackageData
from calc_package.tables import choices_section, package_choices


def _axial():
    from axial_pile import (
        AxialPileAnalysis, AxialSoilLayer, AxialSoilProfile, make_pipe_pile,
    )
    analysis = AxialPileAnalysis(
        pile=make_pipe_pile(0.6, 0.012, closed_end=True),
        soil=AxialSoilProfile(layers=[
            AxialSoilLayer(10, "cohesionless", 18.0, friction_angle=32),
            AxialSoilLayer(10, "cohesive", 17.0, cohesion=60)],
            gwt_depth=2.0),
        pile_length=15)
    return analysis.compute(), analysis


def test_section_lists_each_choice():
    result, _ = _axial()
    section = choices_section(result)
    assert section.title == "Method Choices"
    intro, table = section.items
    assert "engineer's to accept or change" in intro
    assert table.headers == ["Choice", "Used", "How", "Basis",
                             "Other options"]
    assert table.rows[0][:3] == ["side resistance method",
                                 "Nordlund 0-10 m; Tomlinson alpha 10-15 m",
                                 "module's choice"]


def test_package_prints_it_after_the_inputs():
    result, analysis = _axial()
    html = generate_calc_package("axial_pile", result, analysis)
    inputs = html.index("Input Parameters")
    choices = html.index("Method Choices")
    assert inputs < choices < html.index("Tomlinson alpha 10-15 m") \
        < html.index("Soil Profile")
    data = CalcPackageData(sections=[choices_section(result)])
    assert "Method Choices" in render_latex(data)


def test_no_choices_no_section():
    class Plain:
        pass
    assert choices_section(Plain()) is None
    result, analysis = _axial()
    result.choices = []
    assert "Method Choices" not in generate_calc_package(
        "axial_pile", result, analysis)


def test_the_table_is_plain_words():
    from settlement import ConsolidationLayer, SettlementAnalysis
    r = SettlementAnalysis(
        q_applied=150, q_overburden=20, B=3.0, L=3.0, Es_immediate=15000,
        C_alpha=0.01, t_secondary=10,
        consolidation_layers=[ConsolidationLayer(
            thickness=4.0, depth_to_center=4.0, e0=0.9, Cc=0.3, Cr=0.05,
            sigma_v0=60.0)]).compute()
    (_, table) = choices_section(r).items
    text = " ".join(" ".join(row) for row in table.rows)
    for key in ("stress_method", "immediate_method", "e0_secondary", "(cv)",
                "layer_results", "'boussinesq'"):
        assert key not in text, key
    stress = next(row for row in table.rows
                  if row[0] == "stress increase with depth")
    assert stress[4] == "Boussinesq; Westergaard"
    # the agent's own copy keeps the input names
    assert any("stress_method" in a for c in r.choices
               for a in c.get("alternatives", []))


def test_a_search_behind_the_package_supplies_them():
    class Search:
        choices = [{"what": "search (surface_type)", "chosen": "circular",
                    "how": "default"}]

    class Single:
        pass
    assert package_choices(Single(), {"search": Search()}) == Search.choices
    assert package_choices(Single(), None) == []
