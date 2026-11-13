"""BACKLOG GM-312: the cantilever wall's structural design through the
agent (``retaining_walls.cantilever_wall_structural_design``) and as a
calc package (``retaining_wall_package``, wall_type
``cantilever_reinforced``). Made-up walls."""

import os

import pytest

from funhouse_agent import dispatch as D

WALL = {"wall_height": 4.5, "base_width": 3.0, "toe_length": 0.6,
        "stem_thickness_top": 0.305, "stem_thickness_base": 0.45,
        "base_thickness": 0.45, "embedment_depth": 0.9, "surcharge": 12.0,
        "gamma_backfill": 19.0, "phi_backfill": 32.0, "q_allowable": 250.0,
        "fc_MPa": 28.0, "joint_spacing": 8.0}


def _design(**kw):
    return D.call_agent("retaining_walls",
                        "cantilever_wall_structural_design", {**WALL, **kw})


def test_the_tool_designs_the_wall_the_stability_tool_checks():
    r = _design()
    assert "error" not in r, r
    alone = D.call_agent("retaining_walls", "cantilever_wall", {
        k: v for k, v in WALL.items() if k not in ("fc_MPa",
                                                   "joint_spacing")})
    for key in ("FOS_sliding", "FOS_overturning", "FOS_bearing",
                "q_toe_kPa", "q_heel_kPa", "verdict"):
        assert r["stability"][key] == alone[key], key
    # every structural quantity in both unit systems
    for name in ("stem", "toe", "heel"):
        el = r[name]
        for key in ("Mu_kNm_per_m", "Mu_kipft_per_ft", "Vu_kN_per_m",
                    "Vu_kip_per_ft", "d_mm", "d_in",
                    "As_required_mm2_per_m", "As_required_in2_per_ft",
                    "As_provided_mm2_per_m", "phi_Mn_kNm_per_m",
                    "phi_Mn_kipft_per_ft", "phi_Vc_kN_per_m",
                    "phi_Vc_kip_per_ft", "phi_Vc_SI_form_kN_per_m"):
            assert el[key] is not None, (name, key)
        assert el["Mu_kipft_per_ft"] == pytest.approx(
            el["Mu_kNm_per_m"] / 4.448222, abs=0.002)
        assert all("p. " in c["reference"] or "pp. " in c["reference"]
                   for c in el["checks"])
    assert r["stem"]["Mu_kNm_per_m"] == pytest.approx(151.80, abs=0.01)
    assert r["reinforcement"]["schema"] == "reinforced_wall_section/1"
    assert [b["mark"] for b in r["reinforcement"]["bars"]] == list("ABCDEF")
    assert r["checks_failed"] == ["sliding"]
    assert any("3A and 3B" in n["what"] for n in r["not_designed"])
    # the stability choices and the design's, in one list
    whats = [c["what"] for c in r["choices"]]
    assert "strength reduction factors" in whats
    assert any("base friction" in w for w in whats)
    assert "choices" not in r["stability"]
    used = r["defaults_used"]
    assert used["fy_MPa"] == 414.0
    assert used["base_bottom_cover_mm"].startswith("100 mm (4 in.)")
    assert "fc_MPa" not in used and "joint_spacing" not in used


def test_the_tool_takes_the_common_names_and_refuses_the_rest():
    short = {"height": 4.5, "gamma": 19.0, "phi": 32.0, "fc": 28.0,
             "joint_spacing": 8.0, "stem_thickness_top": 0.305}
    r = D.call_agent("retaining_walls",
                     "cantilever_wall_structural_design", short)
    assert "error" not in r, r
    missing = _design(fc_MPa=None)
    bad = D.call_agent("retaining_walls",
                       "cantilever_wall_structural_design",
                       {k: v for k, v in WALL.items() if k != "fc_MPa"})
    assert "fc_MPa" in bad["error"]
    assert "error" in missing
    typo = _design(stem_bar_sise="No. 6")
    assert "stem_bar_size" in typo["error"]
    other = _design(fy_MPa=500.0)
    assert "modular_ratio" in other["error"]
    thin = _design(base_thickness=0.30)
    assert "base_top_cover_mm is required" in thin["error"]


def test_the_package_ends_with_the_structural_sections(tmp_path,
                                                        monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = str(tmp_path / "wall.html")
    r = D.call_agent("calc_package", "retaining_wall_package", {
        **WALL, "wall_type": "cantilever_reinforced", "output_path": out})
    assert "error" not in r, r
    assert r["analysis_type"].startswith("Retaining Wall (stability and "
                                         "structural")
    assert os.path.isfile(out)
    html = open(out, encoding="utf-8").read()
    order = [html.index(t) for t in (
        "Sliding Stability Check", "Structural Design: Basis and Load "
        "Cases", "Structural Design: Stem", "Structural Design: Toe",
        "Structural Design: Heel", "Structural Design: Reinforcement and "
        "Development", "Structural Design: Not Designed")]
    assert order == sorted(order)
    for words in ("Eq. D-8, D-9 (p. 72)", "Eq. 5-1 (p. 35)",
                  "Table E-2 (p. 93)", "Eq. B-49, B-50 (p. 51)",
                  "kip-ft/ft", "Bar schedule", "No. 5 @ 150 mm",
                  "not printed in a held reference", "EM 1110-2-2502"):
        assert words in html, words
    # the numbers beside the file
    assert r["stem"]["Mu_kNm_per_m"] == pytest.approx(151.80, abs=0.01)
    assert r["stem"]["bar"]["size"] == "No. 5"
    assert r["checks_failed"] == ["sliding"]
    assert r["reinforcement"]["schema"] == "reinforced_wall_section/1"
    assert "lap lengths and straight development lengths" in \
        r["not_designed"]
    # the tool's own name picks the same package
    again = D.call_agent("calc_package", "retaining_wall_package", {
        **WALL, "wall_type": "cantilever_wall_structural_design",
        "output_path": str(tmp_path / "again.html")})
    assert again["stem"] == r["stem"]


def test_the_plain_cantilever_package_is_unchanged_by_the_new_type(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wall = {k: v for k, v in WALL.items()
            if k not in ("fc_MPa", "joint_spacing")}
    r = D.call_agent("calc_package", "retaining_wall_package", {
        **wall, "output_path": str(tmp_path / "plain.html")})
    assert "error" not in r, r
    html = open(tmp_path / "plain.html", encoding="utf-8").read()
    assert "Structural Design" not in html
    assert "stem" not in r
    refused = D.call_agent("calc_package", "retaining_wall_package", {
        **wall, "fc_MPa": 28.0,
        "output_path": str(tmp_path / "refused.html")})
    assert "error" in refused and "cantilever_reinforced" in \
        refused["error"]
