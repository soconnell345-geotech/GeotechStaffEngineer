"""DR-11 slice 3: method names read with spelling tolerance, and
retaining-wall earth-pressure coefficients (static Ka, Kp, K0; Mononobe-Okabe
KAE, KPE) recomputed through the app's own module tools.

Offline, no model calls. Every label is made up in the shape of the generic
examples in ``module_work/dr11/LABELS_SUMMARY.md``; every number is
invented, and each expected value comes from the module itself.
"""

import pytest

from funhouse_agent.dispatch import call_agent
from report_ingest import recompute
from report_ingest.recompute import (NOT_RECOMPUTED,
                                     RECOMPUTE_TOOL_DESCRIPTION,
                                     recompute_calculation, table_markdown)


def nq(name, value, unit="", page=4):
    return {"name": name, "value": {"value": value, "unit": unit,
                                    "prov": {"page": page, "method": "text",
                                             "confidence": 1.0}}}


def wall(results, inputs, method="", summary="", subject="Wall W-7"):
    return {"kind": "retaining_wall", "method": method, "summary": summary,
            "subject": subject, "pages": [6], "inputs": list(inputs),
            "results": list(results)}


def K(**params):
    return call_agent("retaining_walls", "earth_pressure_coefficient",
                      params)["K"]


def MO(**params):
    return call_agent("seismic_geotech", "seismic_earth_pressure", params)


def one(calc):
    (row,) = recompute_calculation(calc)
    return row


PHI = nq("Backfill friction angle phi", 32, "deg")


# -- 1. method names read with spelling tolerance ------------------------------

def bearing_calc(method):
    return {"kind": "shallow_foundation_bearing", "method": method,
            "subject": "Square footing F-3", "pages": [3],
            "inputs": [nq("Footing width B", 2.0, "m"),
                       nq("Embedment depth Df", 1.0, "m"),
                       nq("Unit weight", 18.5, "kN/m3"),
                       nq("Friction angle", 30, "deg")],
            "results": [nq("Ultimate bearing capacity", 900, "kPa")]}


@pytest.mark.parametrize("printed, method", [
    ("Mayerhof (1963)", "meyerhof"),
    ("Terzhagi", "terzaghi"),
    ("Vesik", "vesic"),
])
def test_a_misspelled_bearing_method_is_read_as_that_method(printed, method):
    row = one(bearing_calc(printed))
    assert row["method_used"] == method
    assert "read as" in row["method_basis"]
    assert row["method_spelling"] and "off" in row["method_spelling"][0]
    expected = call_agent("bearing_capacity", "bearing_capacity_analysis", {
        "width": 2.0, "depth": 1.0, "unit_weight": 18.5,
        "friction_angle": 30, "shape": "square",
        "factor_method": method})["q_ultimate_kPa"]
    assert row["recomputed_si"] == pytest.approx(expected, rel=1e-6)


def test_a_correctly_spelled_method_says_nothing_about_spelling():
    row = one(bearing_calc("Meyerhof"))
    assert "method_spelling" not in row
    assert "read as" not in row["method_basis"]


def test_english_words_and_other_letters_are_never_read_as_a_method():
    text, notes = recompute._read_methods("ranking by column, basic case")
    assert text == "ranking by column, basic case" and notes == []


def test_a_word_as_close_to_two_methods_is_not_read(monkeypatch):
    monkeypatch.setattr(recompute, "_KNOWN_METHODS", ("hansen", "hanser"))
    text, notes = recompute._read_methods("hansex")
    assert text == "hansex" and notes == []


# Review fix F1: one letter off for a name of seven letters or fewer; two
# only for a long name, with its last letter and no English ending. Ordinary
# words, names and places are never read as a method.
@pytest.mark.parametrize("printed", [
    "values ranging from", "raking struts", "the adjacent ravine", "Racine",
    "Rankin County", "rankle", "ranting", "monopole", "monotone", "couloir",
    "Kessel", "Vesica", "Hanse", "coulombic", "Coloumb", "Columbia",
    "column", "ranking", "Hanson", "vessel",
])
def test_ordinary_words_names_and_places_are_never_read_as_a_method(printed):
    text, notes = recompute._read_methods(recompute._norm(printed))
    assert notes == []
    assert text == recompute._norm(printed)


@pytest.mark.parametrize("printed, name", [
    ("Mayerhof", "meyerhof"), ("Meyerhoff", "meyerhof"),
    ("Mayerhoff", "meyerhof"), ("Terzhagi", "terzaghi"),
    ("Terzagi", "terzaghi"), ("Coulumb", "coulomb"), ("Colomb", "coulomb"),
    ("Vesik", "vesic"), ("Mononabe", "mononobe"), ("Okabi", "okabe"),
    ("Schmertman", "schmertmann"), ("Caquote", "caquot"),
    ("Kerisal", "kerisel"), ("Rankines", "rankine"), ("Hansens", "hansen"),
])
def test_misspelled_method_names_are_still_read(printed, name):
    text, notes = recompute._read_methods(recompute._norm(printed))
    assert text == name and len(notes) == 1


def test_an_ordinary_word_in_the_summary_never_names_a_wall_method():
    row = one(wall([nq("Ka", 0.29)], [PHI], summary=(
        "Backfill friction angles ranging from 30 to 34 degrees, near the "
        "ravine, with raking struts.")))
    assert row["status"] == "not_recomputed"
    assert "names no method" in row["reason"]
    assert "method_spelling" not in row


def test_a_misspelled_wall_method_is_read_and_said():
    delta = nq("Wall friction angle delta", 20, "deg")
    expected = K(phi_deg=32, theory="coulomb", state="active",
                 delta_deg=20)
    row = one(wall([nq("Ka", round(expected, 3))], [PHI, delta],
                   method="Coulumb"))
    assert row["method_used"] == "coulomb" and row["flag"] == "agrees"
    assert "'coulumb' read as Coulomb" in row["method_basis"]


def test_a_misspelled_mononobe_okabe_is_read():
    row = one(wall([nq("KAE", 0.5)],
                   [PHI, nq("delta", 21, "deg"), nq("kh", 0.2)],
                   method="Mononabe-Okabe"))
    assert row["method_used"] == "mononobe_okabe"
    assert "'mononabe' read as Mononobe" in row["method_basis"]


# -- 2. static coefficients -----------------------------------------------------

def test_rankine_ka_kp_and_k0_are_recomputed():
    ka = K(phi_deg=32, theory="rankine", state="active")
    kp = K(phi_deg=32, theory="rankine", state="passive")
    k0 = K(phi_deg=32, state="at_rest")
    rows = recompute_calculation(wall(
        [nq("Active earth pressure coefficient Ka", round(ka, 3)),
         nq("Kp", round(kp, 2)), nq("At-rest coefficient K0", 0.47)],
        [PHI], method="Rankine"))
    assert [r["coefficient"] for r in rows] == ["Ka", "Kp", "K0"]
    assert all(r["status"] == "recomputed" and r["flag"] == "agrees"
               for r in rows)
    assert [r["recomputed_si"] for r in rows] == pytest.approx([ka, kp, k0],
                                                               abs=1e-4)
    assert rows[0]["module"] == "retaining_walls"
    assert rows[0]["module_method"] == "earth_pressure_coefficient"
    assert rows[2]["method_used"] == "jaky"


def test_rankine_ka_on_a_sloping_backfill():
    expected = K(phi_deg=32, theory="rankine", state="active", beta_deg=15)
    row = one(wall([nq("Ka", round(expected, 3))],
                   [PHI, nq("Backfill slope", 15, "deg")], method="Rankine"))
    assert row["flag"] == "agrees"
    assert row["recomputed_si"] == pytest.approx(expected, abs=1e-4)


def test_coulomb_with_a_batter_runs_both_senses():
    # the batter's sense is not printed: the default and the other run
    other = K(phi_deg=32, theory="coulomb", state="active", delta_deg=20,
              beta_deg=10, alpha_deg=100)
    row = one(wall([nq("Ka", round(other, 3))],
                   [PHI, nq("Wall friction", 20, "deg"),
                    nq("Backfill slope", 10, "deg"),
                    nq("Wall batter", 10, "deg")], method="Coulomb"))
    assert row["recomputed_si"] == pytest.approx(
        K(phi_deg=32, theory="coulomb", state="active", delta_deg=20,
          beta_deg=10, alpha_deg=80), abs=1e-4)
    assert row["flag"] == "within_options"
    assert any("alpha 100" in a["option"] for a in row["alternatives"])


# Review fix F4: the back-face batter acts on Ka and KAE only; Kp and KPE
# act on the front of the wall and run with a vertical face
def test_the_back_face_batter_is_not_passed_to_kp_or_kpe():
    batter = nq("Wall batter (back face)", 10, "deg")
    kp = K(phi_deg=32, theory="coulomb", state="passive", delta_deg=10)
    row = one(wall([nq("Kp", round(kp, 2))],
                   [PHI, nq("Wall friction", 10, "deg"), batter],
                   method="Coulomb"))
    assert row["flag"] == "agrees"
    assert row["recomputed_si"] == pytest.approx(kp, abs=1e-4)
    assert "alpha_deg" not in row["params"]
    assert not any("alpha" in a["option"] for a in row.get("alternatives", []))
    assert "front of the wall" in row["inputs_used"]["batter"]["note"]
    kpe = MO(phi=32, delta=10, kh=0.2, include_passive=True,
             delta_passive=10)["KPE"]
    row = one(wall([nq("KPE", round(kpe, 2))],
                   [PHI, nq("Wall friction", 10, "deg"), nq("kh", 0.2),
                    batter], method="Mononobe-Okabe"))
    assert row["flag"] == "agrees" and "beta" not in row["params"]
    # the active coefficient still takes the batter, both senses
    kae = one(wall([nq("KAE", 0.5)],
                   [PHI, nq("Wall friction", 10, "deg"), nq("kh", 0.2),
                    batter], method="Mononobe-Okabe"))
    assert kae["params"]["beta"] == -10


def test_coulomb_without_a_wall_friction_is_refused():
    row = one(wall([nq("Ka", 0.28)], [PHI], method="Coulomb"))
    assert row["status"] == "not_recomputed"
    assert "wall friction delta: not printed" in row["reason"]


def test_no_method_with_a_wall_friction_runs_rankine_and_coulomb():
    coulomb = K(phi_deg=32, theory="coulomb", state="active", delta_deg=21)
    row = one(wall([nq("Ka", round(coulomb, 3))],
                   [PHI, nq("Interface friction angle", 21, "deg")]))
    assert row["method_used"] == "rankine"
    assert "names no method" in row["method_basis"]
    assert row["flag"] == "within_options"


def test_no_method_and_no_wall_friction_is_refused():
    row = one(wall([nq("Ka", 0.31)], [PHI]))
    assert row["status"] == "not_recomputed"
    assert "names no method" in row["reason"]


def test_a_wall_friction_printed_as_a_ratio_of_phi():
    expected = K(phi_deg=30, theory="coulomb", state="active",
                 delta_deg=20.1)
    row = one(wall([nq("Ka", round(expected, 3))],
                   [nq("phi", 30, "deg"), nq("delta/phi", 0.67)],
                   method="Coulomb"))
    assert row["flag"] == "agrees"
    assert "0.67 x phi" in row["inputs_used"]["delta"]["note"]


def test_log_spiral_kp():
    expected = K(phi_deg=32, theory="caquot_kerisel", state="passive",
                 delta_deg=16)
    row = one(wall([nq("Passive earth pressure coefficient Kp",
                       round(expected, 2))],
                   [PHI, nq("Wall friction angle", 16, "deg")],
                   method="Log spiral"))
    assert row["method_used"] == "log_spiral" and row["flag"] == "agrees"


def test_k0_with_a_printed_ocr_runs_the_ocr_form_beside_jaky():
    with_ocr = K(phi_deg=32, state="at_rest", OCR=2.0)
    row = one(wall([nq("K0", round(with_ocr, 3))],
                   [PHI, nq("OCR", 2.0)]))
    assert row["recomputed_si"] == pytest.approx(
        K(phi_deg=32, state="at_rest"), abs=1e-4)
    assert row["flag"] == "within_options"


def test_a_wrong_coefficient_differs():
    row = one(wall([nq("Ka", 0.45)], [PHI], method="Rankine"))
    assert row["flag"] == "differs"


def test_a_pressure_in_kpa_is_not_a_coefficient():
    row = one(wall([nq("Active earth pressure at base", 40, "kPa")], [PHI],
                   method="Rankine"))
    assert row["status"] == "not_recomputed"
    assert row["reason"] == NOT_RECOMPUTED["retaining_wall"]


# -- 3. seismic coefficients ------------------------------------------------------

def test_mononobe_okabe_kae_with_kh_printed():
    expected = MO(phi=32, delta=21, kh=0.2)["KAE"]
    row = one(wall([nq("Seismic active coefficient KAE", round(expected, 3))],
                   [PHI, nq("delta", 21, "deg"),
                    nq("Horizontal seismic coefficient kh", 0.2)],
                   method="Mononobe-Okabe"))
    assert row["flag"] == "agrees"
    assert row["module"] == "seismic_geotech"
    assert row["module_method"] == "seismic_earth_pressure"


def test_kv_runs_with_both_signs():
    other = MO(phi=32, delta=21, kh=0.25, kv=-0.15)["KAE"]
    row = one(wall([nq("KAE", round(other, 3))],
                   [PHI, nq("delta", 21, "deg"), nq("kh", 0.25),
                    nq("kv", 0.15)], method="M-O"))
    assert row["recomputed_si"] == pytest.approx(
        MO(phi=32, delta=21, kh=0.25, kv=0.15)["KAE"], abs=1e-4)
    assert row["flag"] == "within_options"


@pytest.mark.parametrize("rule_label", ["Rule: kh = 0.5 PGA",
                                        "kh = PGA/2 rule"])
def test_kh_from_the_pga_by_the_printed_rule(rule_label):
    expected = MO(phi=32, delta=21, kh=0.15)["KAE"]
    calc = wall([nq("KAE", round(expected, 3))],
                [PHI, nq("delta", 21, "deg"), nq("PGA", 0.30, "g")],
                method="Mononobe-Okabe", summary=rule_label)
    row = one(calc)
    assert row["flag"] == "agrees"
    assert "by the rule the printout states" in row["inputs_used"]["kh"][
        "note"]


def _rule(summary):
    return recompute._kh_rule(recompute.Calculation.model_validate(wall(
        [], [], method="Mononobe-Okabe", summary=summary)))


@pytest.mark.parametrize("summary, factor, var", [
    ("kh = 0.5 PGA", 0.5, "pga"), ("kh = PGA/2", 0.5, "pga"),
    ("kh = 0.5 PGA = 0.15", 0.5, "pga"), ("kh = 0.5(As) = 0.2", 0.5, "As"),
    ("kh = 0.5 PGA (0.30 g)", 0.5, "pga"), ("kh = As.", 1.0, "As"),
])
def test_a_kh_rule_stated_as_factor_times_acceleration_is_read(
        summary, factor, var):
    (f, v, _text), why = _rule(summary)
    assert (f, v, why) == (factor, var, "")


# Review fix F2: a kh value printed with the PGA beside it, 'as' in a
# sentence, and a rule carrying the site factor are never read as a rule
@pytest.mark.parametrize("summary", [
    "seismic increments with kh=0.12 (PGA 0.24 g)",
    "kh = 0.12 (PGA = 0.24 g)", "kh = 0.12, PGA 0.24",
    "kh = 0.15 as recommended by the geotechnical engineer",
])
def test_a_kh_value_or_a_sentence_is_not_a_rule(summary):
    assert _rule(summary) == (None, "")


@pytest.mark.parametrize("summary", ["kh = 0.5(PGA)(Fpga)",
                                     "kh = 0.5 PGA x Fpga",
                                     "kh = 0.5 x Fpga x PGA"])
def test_a_kh_rule_with_the_site_factor_is_refused(summary):
    rule, why = _rule(summary)
    assert rule is None and "site factor Fpga" in why
    row = one(wall([nq("KAE", 0.45)],
                   [PHI, nq("delta", 21, "deg"), nq("PGA", 0.40, "g"),
                    nq("Fpga", 1.2)], method="Mononobe-Okabe",
                   summary=summary))
    assert row["status"] == "not_recomputed"
    assert "site factor Fpga" in row["reason"]


def test_a_kh_value_beside_the_pga_never_scales_the_pga():
    row = one(wall([nq("KAE", 0.45)],
                   [PHI, nq("delta", 21, "deg"), nq("PGA", 0.24, "g")],
                   method="Mononobe-Okabe",
                   summary="Seismic increments with kh=0.12 (PGA 0.24 g)."))
    assert row["status"] == "not_recomputed"
    assert "would be a guess" in row["reason"]


def test_a_pga_without_a_printed_rule_is_never_halved():
    row = one(wall([nq("KAE", 0.4)],
                   [PHI, nq("delta", 21, "deg"), nq("PGA", 0.30, "g")],
                   method="Mononobe-Okabe"))
    assert row["status"] == "not_recomputed"
    assert "would be a guess" in row["reason"]


def test_mononobe_okabe_kpe():
    expected = MO(phi=32, delta=10, kh=0.2, include_passive=True,
                  delta_passive=10)["KPE"]
    row = one(wall([nq("KPE", round(expected, 2))],
                   [PHI, nq("Wall friction", 10, "deg"), nq("kh", 0.2)],
                   method="Mononobe-Okabe"))
    assert row["coefficient"] == "KPE" and row["flag"] == "agrees"


def test_a_seismic_increment_is_not_compared():
    rows = recompute_calculation(wall(
        [nq("delta KAE", 0.08)],
        [PHI, nq("delta", 21, "deg"), nq("kh", 0.2)],
        method="Mononobe-Okabe"))
    assert rows[0]["status"] == "not_compared"


# Review fix F3: the increment sign U+2206 is read as delta, and 'Eq.' (an
# equation) does not make a static coefficient seismic
def test_an_increment_printed_with_the_increment_sign_is_not_compared():
    rows = recompute_calculation(wall(
        [nq("∆ KAE", 0.08)],
        [PHI, nq("delta", 21, "deg"), nq("kh", 0.2)],
        method="Mononobe-Okabe"))
    assert rows[0]["coefficient"] == "increment"
    assert rows[0]["status"] == "not_compared"


def test_an_equation_number_does_not_make_a_static_label_seismic():
    ka = K(phi_deg=32, theory="rankine", state="active")
    row = one(wall([nq("Ka (Eq. 3-1)", round(ka, 3))],
                   [PHI, nq("kh", 0.2)], method="Rankine"))
    assert row["coefficient"] == "Ka" and row["flag"] == "agrees"
    seismic = one(wall([nq("Ka (EQ)", 0.5)],
                       [PHI, nq("delta", 21, "deg"), nq("kh", 0.2)],
                       method="Mononobe-Okabe"))
    assert seismic["coefficient"] == "KAE"


# -- 4. refusals and cases ---------------------------------------------------------

def test_phi_not_printed_is_refused():
    row = one(wall([nq("Ka", 0.3)], [], method="Rankine"))
    assert row["status"] == "not_recomputed"
    assert "friction_angle (phi): not printed" in row["reason"]


def test_phi_printed_twice_is_refused():
    row = one(wall([nq("Ka", 0.3)],
                   [nq("phi", 30, "deg"), nq("Friction angle", 34, "deg")],
                   method="Rankine"))
    assert row["status"] == "not_recomputed"
    assert "printed more than once" in row["reason"]


def test_several_strata_are_recomputed_case_by_case():
    a = K(phi_deg=30, theory="rankine", state="active")
    b = K(phi_deg=36, theory="rankine", state="active")
    rows = recompute_calculation(wall(
        [nq("Ka, stratum 1", round(a, 3)), nq("Ka (Stratum 2)", round(b, 3))],
        [nq("phi, stratum 1", 30, "deg"), nq("Stratum 2 phi", 36, "deg")],
        method="Rankine"))
    assert [r["case"] for r in rows] == ["stratum 1", "stratum 2"]
    assert all(r["flag"] == "agrees" for r in rows)
    assert "Ka, stratum 1 [stratum 1]" in table_markdown(
        {"report_id": "X", "tolerance": 0.1, "rows": rows})


def test_soils_named_in_the_labels_split_the_cases():
    a = K(phi_deg=30, theory="rankine", state="active")
    b = K(phi_deg=34, theory="rankine", state="active")
    rows = recompute_calculation(wall(
        [nq("Ka (silty sand)", round(a, 3)), nq("Ka (gravel fill)", round(b, 3))],
        [nq("phi (silty sand)", 30, "deg"), nq("gravel fill phi", 34, "deg")],
        method="Rankine"))
    assert sorted(r["case"] for r in rows) == ["gravel fill", "silty sand"]
    assert all(r["flag"] == "agrees" for r in rows)


def test_a_wall_with_no_coefficient_says_why():
    row = one(wall([nq("Factor of safety against sliding", 1.6)], [PHI]))
    assert row["status"] == "not_recomputed"
    assert row["reason"] == NOT_RECOMPUTED["retaining_wall"]


def test_the_tool_description_names_the_wall_coefficients():
    assert "earth-pressure coefficients" in RECOMPUTE_TOOL_DESCRIPTION
