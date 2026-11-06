"""DR-11 slice 4: the AASHTO 1993 rigid pavement slab thickness D (and the
traffic a printed slab carries) recomputed through the app's own
``pavement_design`` tool.

Offline, no model calls. Every label is made up in the shape of the generic
examples in ``module_work/dr11/LABELS_SUMMARY.md``; every number is
invented, and each expected value comes from the module itself.
"""

import pytest

from funhouse_agent.dispatch import call_agent
from report_ingest.recompute import (NOT_RECOMPUTED,
                                     RECOMPUTE_TOOL_DESCRIPTION,
                                     recompute_calculation, table_markdown)


def nq(name, value, unit="", page=4):
    return {"name": name, "value": {"value": value, "unit": unit,
                                    "prov": {"page": page, "method": "text",
                                             "confidence": 1.0}}}


def rigid(params, **extra):
    return call_agent("pavement_design", "rigid_pavement_design",
                      dict(params, **extra))


W18 = nq("Design ESALs W18", 3.4e6, "ESALs")
R = nq("Reliability R", 90, "%")
SO = nq("Overall standard deviation So", 0.34)
PO = nq("Initial serviceability po", 4.5)
PT = nq("Terminal serviceability pt", 2.5)
SC = nq("Modulus of rupture S'c", 620, "psi")
EC = nq("Concrete elastic modulus Ec", 4.1e6, "psi")
J = nq("Load transfer coefficient J", 3.1)
CD = nq("Drainage coefficient Cd", 1.05)
K = nq("Effective modulus of subgrade reaction k", 110, "pci")
INPUTS = [W18, R, SO, PO, PT, SC, EC, J, CD, K]
PARAMS = dict(w18=3.4e6, reliability_pct=90, so=0.34, po=4.5, pt=2.5,
              sc_psi=620, ec_psi=4.1e6, j=3.1, cd=1.05, k_pci=110)


def pave(results, inputs=INPUTS, method="AASHTO 1993", summary="",
         subject="Rigid pavement, truck lane"):
    return {"kind": "pavement", "method": method, "summary": summary,
            "subject": subject, "pages": [8], "inputs": list(inputs),
            "results": list(results)}


def without(*names):
    return [n for n in INPUTS if n["name"] not in names]


def one(calc):
    (row,) = recompute_calculation(calc)
    return row


def D_REQ(**over):
    return rigid(PARAMS, **over)["d_required_in"]


# -- the design: D from the printed inputs -------------------------------------

def test_d_recomputed_through_the_module():
    d = D_REQ()
    row = one(pave([nq("Required slab thickness D", round(d, 2), "in")]))
    assert row["status"] == "recomputed"
    assert row["flag"] == "agrees"
    assert row["recomputed_si"] == pytest.approx(d)
    assert row["unit_si"] == "in"
    assert row["module"] == "pavement_design"
    assert row["module_method"] == "rigid_pavement_design"
    assert row["method_basis"] == "printed: AASHTO 1993"
    assert row["params"]["k_pci"] == 110
    assert row["assumed"] == {}


def test_wrong_d_differs():
    d = D_REQ()
    row = one(pave([nq("Slab thickness D", round(d * 1.3, 1), "in")]))
    assert row["flag"] == "differs"


def test_no_method_printed_reads_the_inputs():
    d = D_REQ()
    row = one(pave([nq("PCC thickness", round(d, 2), "in")], method=""))
    assert row["flag"] == "agrees"
    assert "names no method" in row["method_basis"]


def test_design_w18_printed_among_results_is_an_input():
    d = D_REQ()
    row = one(pave([W18, nq("D required", round(d, 2), "in")],
                   inputs=without("Design ESALs W18")))
    assert row["flag"] == "agrees"
    assert row["claim"] == "D"


# -- reliability ---------------------------------------------------------------

def test_zr_printed_is_used_and_a_positive_zr_read_negative():
    d = rigid({k: v for k, v in PARAMS.items() if k != "reliability_pct"},
              zr=-1.282)["d_required_in"]
    inputs = without("Reliability R") + [nq("Standard normal deviate ZR",
                                            1.282)]
    row = one(pave([nq("Slab thickness D", round(d, 2), "in")], inputs))
    assert row["flag"] == "agrees"
    assert row["params"]["zr"] == -1.282
    assert "read as -1.282" in row["inputs_used"]["zr"]["note"]


def test_reliability_as_a_fraction():
    d = D_REQ()
    inputs = without("Reliability R") + [nq("Reliability", 0.90)]
    row = one(pave([nq("Slab thickness D", round(d, 2), "in")], inputs))
    assert row["flag"] == "agrees"
    assert "90 %" in row["inputs_used"]["reliability"]["note"]


def test_no_reliability_refused():
    row = one(pave([nq("Slab thickness D", 9, "in")],
                   without("Reliability R")))
    assert row["status"] == "not_recomputed"
    assert "reliability R or ZR" in row["reason"]


# -- inputs never filled in ----------------------------------------------------

@pytest.mark.parametrize("drop, named", [
    ("Overall standard deviation So", "overall standard deviation So"),
    ("Load transfer coefficient J", "load transfer coefficient J"),
    ("Drainage coefficient Cd", "drainage coefficient Cd"),
    ("Modulus of rupture S'c", "modulus of rupture S'c"),
    ("Concrete elastic modulus Ec", "concrete modulus Ec"),
    ("Design ESALs W18", "design ESALs W18"),
])
def test_a_missing_input_is_named_never_assumed(drop, named):
    row = one(pave([nq("Slab thickness D", 9, "in")], without(drop)))
    assert row["status"] == "not_recomputed"
    assert named in row["reason"]
    assert "params" not in row


def test_k_missing_with_a_resilient_modulus_printed_says_why():
    inputs = without("Effective modulus of subgrade reaction k") + [
        nq("Roadbed resilient modulus MR", 6000, "psi")]
    row = one(pave([nq("Slab thickness D", 9, "in")], inputs))
    assert row["status"] == "not_recomputed"
    assert "MR/19.4" in row["reason"]


def test_an_input_printed_twice_is_refused():
    inputs = INPUTS + [nq("J (load transfer)", 3.8)]
    row = one(pave([nq("Slab thickness D", 9, "in")], inputs))
    assert row["status"] == "not_recomputed"
    assert "more than once" in row["reason"]


# -- serviceability --------------------------------------------------------------

def test_pt_alone_is_refused():
    row = one(pave([nq("Slab thickness D", 9, "in")],
                   without("Initial serviceability po")))
    assert row["status"] == "not_recomputed"
    assert "serviceability loss: only pt printed" in row["reason"]


def test_delta_psi_alone_runs_pt_options_beside_it():
    d = rigid({k: v for k, v in PARAMS.items() if k not in ("po", "pt")},
              delta_psi=1.7)["d_required_in"]
    inputs = without("Initial serviceability po",
                     "Terminal serviceability pt") + [
        nq("Design serviceability loss delta PSI", 1.7)]
    row = one(pave([nq("Slab thickness D", round(d, 2), "in")], inputs))
    assert row["flag"] == "agrees"
    assert row["assumed"] == {"pt": 2.5}
    assert [a["params"]["pt"] for a in row["alternatives"]] == [2.0, 3.0]
    assert all("recomputed_si" in a for a in row["alternatives"])


# -- units ---------------------------------------------------------------------

def test_si_printout_converted_once_and_said():
    sc_mpa, ec_mpa, k_mpa_m = 4.3, 28000.0, 30.0
    psi = 1 / 6.894757293168361
    params = dict(PARAMS, sc_psi=sc_mpa * 1000 * psi,
                  ec_psi=ec_mpa * 1000 * psi, k_pci=k_mpa_m / 0.27144710)
    d_in = rigid(params)["d_required_in"]
    inputs = without("Modulus of rupture S'c", "Concrete elastic modulus Ec",
                     "Effective modulus of subgrade reaction k") + [
        nq("Modulus of rupture S'c", sc_mpa, "MPa"),
        nq("Concrete modulus Ec", ec_mpa, "MPa"),
        nq("Modulus of subgrade reaction k", k_mpa_m, "MPa/m")]
    row = one(pave([nq("Slab thickness D", round(d_in * 25.4), "mm")],
                   inputs))
    assert row["flag"] == "agrees"
    assert row["recomputed_si"] == pytest.approx(d_in, rel=1e-6)
    assert row["claimed_si"] == pytest.approx(round(d_in * 25.4) / 25.4)
    assert "converted once" in row["units_note"]
    assert "MPa/m" in row["units_note"]


def test_w18_in_millions_and_as_a_log():
    d = D_REQ()
    for w in (nq("Design traffic (18-kip ESALs)", 3.4, "million"),
              nq("log W18", 6.531478917)):
        inputs = without("Design ESALs W18") + [w]
        row = one(pave([nq("Slab thickness D", round(d, 2), "in")], inputs))
        assert row["flag"] == "agrees", w["name"]


# -- rounding to a construction increment ------------------------------------------

THIN = dict(w18=2e5, sc_psi=700, ec_psi=4.2e6, zr=-1.282, so=0.35, po=4.5,
            pt=2.0, j=3.2, cd=1.1, k_pci=150)
THIN_INPUTS = [nq("W18", 2e5), nq("ZR", -1.282), nq("So", 0.35),
               nq("po", 4.5), nq("pt", 2.0), nq("S'c", 700, "psi"),
               nq("Ec", 4.2e6, "psi"), nq("J", 3.2), nq("Cd", 1.1),
               nq("k", 150, "pci")]


def test_rounded_claim_agrees_only_when_the_printout_says_it_rounds():
    d = rigid(THIN)["d_required_in"]
    claim = float(int(d) + 1)  # the next whole inch, more than 10 % up
    assert (claim - d) / claim > 0.10
    quiet = one(pave([nq("Design slab thickness D", claim, "in")],
                     THIN_INPUTS))
    assert quiet["flag"] == "differs"
    assert "does not say it rounds" in quiet["rounding"]
    said = one(pave([nq("Design slab thickness D", claim, "in")],
                    THIN_INPUTS,
                    summary="D rounded up to the next whole inch."))
    assert said["flag"] == "agrees"
    assert "says it rounds" in said["flag_basis"]


def test_a_claim_below_the_required_d_is_no_rounding():
    d = rigid(THIN)["d_required_in"]
    row = one(pave([nq("Slab thickness D", round(d * 0.7, 1), "in")],
                   THIN_INPUTS, summary="rounded to the nearest inch"))
    assert row["flag"] == "differs"
    assert "rounding" not in row


def test_the_module_refusing_is_said():
    inputs = [nq("W18", 5e4)] + THIN_INPUTS[1:]
    row = one(pave([nq("Slab thickness D", 4, "in")], inputs))
    assert row["status"] == "not_recomputed"
    assert row["reason"].startswith("the module refused")


# -- the inverse: the traffic a printed slab carries -------------------------------

def test_allowable_w18_for_a_printed_slab():
    cap = rigid(PARAMS, slab_thickness_in=10.0)["w18_capacity"]
    inputs = without("Design ESALs W18") + [nq("Slab thickness D", 10, "in")]
    row = one(pave([nq("Allowable ESALs W18", round(cap, -3), "ESALs")],
                   inputs))
    assert row["claim"] == "W18"
    assert row["flag"] == "agrees"
    assert row["unit_si"] == "ESALs"
    assert row["params"]["slab_thickness_in"] == 10
    assert abs(row["log10_difference"]) < 0.01


def test_w18_capacity_of_the_claimed_slab():
    cap = rigid(PARAMS, slab_thickness_in=9.5)["w18_capacity"]
    d = D_REQ()
    rows = recompute_calculation(pave([
        nq("Design slab thickness D", 9.5, "in"),
        nq("W18 capacity of the design slab", round(cap, -3))]))
    by = {r["claim"]: r for r in rows}
    assert by["D"]["recomputed_si"] == pytest.approx(d)
    assert by["W18"]["flag"] == "agrees"
    assert by["W18"]["inputs_used"]["slab"]["note"] == \
        "the slab the printout claims"


def test_allowable_w18_without_a_slab_refused():
    row = one(pave([nq("Allowable ESALs", 2e6)],
                   without("Design ESALs W18"), subject="Rigid pavement"))
    assert row["status"] == "not_recomputed"
    assert "slab thickness D" in row["reason"]


# -- what is not recomputed --------------------------------------------------------

def test_flexible_pavement_says_why():
    row = one({"kind": "pavement", "method": "AASHTO 1993",
               "subject": "Flexible pavement", "pages": [3],
               "inputs": [W18, R, SO],
               "results": [nq("Required structural number SN", 4.2)]})
    assert row["status"] == "not_recomputed"
    assert "flexible pavement" in row["reason"]


def test_ufc_rigid_design_refused():
    row = one(pave([nq("Slab thickness D", 8, "in")],
                   method="UFC 3-250-01 Figure F-1"))
    assert row["status"] == "not_recomputed"
    assert "not AASHTO 1993" in row["reason"]


def test_rigid_pavement_without_a_claim_not_compared():
    row = one(pave([nq("Joint spacing", 15, "ft")]))
    assert row["status"] == "not_compared"


def test_not_recomputed_reason_names_the_slice():
    assert "rigid slab thickness D" in NOT_RECOMPUTED["pavement"]


# -- cases -----------------------------------------------------------------------

def test_sections_by_name_split():
    d_truck = D_REQ()
    d_car = D_REQ(w18=4.0e5)
    inputs = without("Design ESALs W18") + [
        nq("Design ESALs W18 (truck lane)", 3.4e6),
        nq("Design ESALs W18 (parking area)", 4.0e5)]
    rows = recompute_calculation(pave([
        nq("Slab thickness D (truck lane)", round(d_truck, 2), "in"),
        nq("Slab thickness D (parking area)", round(d_car, 2), "in")],
        inputs, subject="Rigid pavement"))
    by = {r["case"]: r for r in rows}
    assert set(by) == {"truck lane", "parking area"}
    assert by["truck lane"]["recomputed_si"] == pytest.approx(d_truck)
    assert by["parking area"]["recomputed_si"] == pytest.approx(d_car)
    assert all(r["flag"] == "agrees" for r in rows)


def test_table_and_tool_description():
    d = D_REQ()
    row_calc = pave([nq("Slab thickness D", round(d, 2), "in")])
    md = table_markdown({"report_id": "X", "tolerance": 0.1,
                         "rows": recompute_calculation(row_calc)})
    assert "pavement" in md and "agrees" in md
    assert "rigid pavement" in RECOMPUTE_TOOL_DESCRIPTION
    assert "unprinted pt" in RECOMPUTE_TOOL_DESCRIPTION


# -- review fixes ------------------------------------------------------------------

def test_design_traffic_beside_a_printed_slab_is_no_capacity_claim():
    # a check printout: the slab is given, and the design traffic worked out
    # from a traffic count is printed among the results; it is not what the
    # slab carries, so it is no inverse claim (it is an input)
    inputs = without("Design ESALs W18") + [nq("Slab thickness D", 10, "in")]
    row = one(pave([nq("Design ESALs W18", 2.6e6, "ESALs")], inputs))
    assert row["status"] == "not_compared"
    cap = rigid(PARAMS, slab_thickness_in=10.0)["w18_capacity"]
    rows = recompute_calculation(pave([
        nq("Design ESALs W18", 2.6e6, "ESALs"),
        nq("Allowable ESALs", round(cap, -3), "ESALs")], inputs))
    assert [(r["claim"], r["quantity"]) for r in rows] == [
        ("W18", "Allowable ESALs")]
    assert rows[0]["flag"] == "agrees"


def test_a_total_thickness_is_not_the_slab():
    d = D_REQ()
    rows = recompute_calculation(pave([
        nq("Slab thickness D", round(d, 2), "in"),
        nq("Total pavement thickness", round(d, 2) + 6, "in")]))
    assert [r["quantity"] for r in rows] == ["Slab thickness D"]
    assert rows[0]["flag"] == "agrees"


@pytest.mark.parametrize("label", ["First-year ESALs", "Annual 18-kip ESALs",
                                   "ESALs/yr", "Daily ESALs"])
def test_a_year_or_a_day_of_traffic_is_not_w18(label):
    inputs = without("Design ESALs W18") + [nq(label, 1.7e5)]
    row = one(pave([nq("Slab thickness D", 9, "in")], inputs))
    assert row["status"] == "not_recomputed"
    assert "design ESALs W18: not printed" in row["reason"]


def test_a_dynamic_k_is_not_the_design_k():
    dyn = nq("Dynamic k-value from FWD", 230, "pci")
    row = one(pave([nq("Slab thickness D", 9, "in")],
                   without("Effective modulus of subgrade reaction k")
                   + [dyn]))
    assert row["status"] == "not_recomputed"
    assert "subgrade reaction k: not printed" in row["reason"]
    d = D_REQ()
    row = one(pave([nq("Slab thickness D", round(d, 2), "in")],
                   INPUTS + [dyn]))
    assert row["flag"] == "agrees"
    assert row["params"]["k_pci"] == 110


def test_an_asphalt_concrete_printout_is_flexible():
    row = one({"kind": "pavement", "method": "AASHTO 1993",
               "subject": "Access drive", "pages": [3],
               "inputs": [nq("W18", 2e5),
                          nq("Asphalt concrete wearing course layer "
                             "coefficient", 0.41)],
               "results": [nq("Required structural number SN", 2.6)]})
    assert row["status"] == "not_recomputed"
    assert "flexible pavement" in row["reason"]


def test_the_flexible_design_beside_the_rigid_one_is_not_read():
    d = D_REQ()
    flexible = [nq("W18, flexible alternative", 1.2e6),
                nq("Reliability, flexible alternative", 85, "%"),
                nq("Standard deviation So, flexible alternative", 0.47),
                nq("Initial serviceability, flexible alternative", 4.05),
                nq("Drainage coefficient, granular base layer", 0.85),
                nq("Asphalt concrete elastic modulus", 3.9e5, "psi")]
    row = one(pave([nq("Required PCC thickness", round(d, 2), "in")],
                   INPUTS + flexible))
    assert row["flag"] == "agrees"
    assert row["recomputed_si"] == pytest.approx(d)
    assert {k: row["params"][k] for k in ("w18", "reliability_pct", "so",
                                         "po", "cd", "ec_psi")} == {
        "w18": 3.4e6, "reliability_pct": 90, "so": 0.34, "po": 4.5,
        "cd": 1.05, "ec_psi": 4.1e6}
