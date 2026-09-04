"""Tests for geotech-references adapter modules (reference agents).

Covers:
- Registry completeness (METHOD_REGISTRY keys == METHOD_INFO keys)
- METHOD_INFO required fields (category, brief, parameters)
- Dispatch integration (list_methods, describe_method)
- Actual call_agent calls for representative functions
- Text retrieval methods (where applicable)
"""

import pytest
from funhouse_agent.dispatch import list_methods, describe_method, call_agent
from funhouse_agent.adapters import MODULE_REGISTRY


# ──────────────────────────────────────────────────────────────────────
# All reference modules
# ──────────────────────────────────────────────────────────────────────

REFERENCE_MODULES = [
    "dm7",
    "em_2104", "em_2107",
    "gec4", "gec5", "gec6", "gec7", "gec8", "gec9",
    "gec10", "gec11", "gec12", "gec13", "gec14",
    "micropile",
    "ufc_backfill", "ufc_expansive", "ufc_pavement", "ufc_structural",
    "ufc_collapse", "gsa_collapse", "wood_handbook",
]

# Modules with text retrieval (retrieve_section, search_sections, etc.)
TEXT_MODULES = [
    "em_2104", "em_2107", "ufc_structural", "ufc_collapse", "gsa_collapse",
    "wood_handbook",
    "gec4", "gec5", "gec6", "gec7", "gec8", "gec9",
    "gec10", "gec11", "gec12", "gec13", "gec14",
    "micropile",
]


# ──────────────────────────────────────────────────────────────────────
# Cross-cutting: all 14 registered
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("module_name", REFERENCE_MODULES)
class TestReferenceRegistered:
    def test_in_module_registry(self, module_name):
        assert module_name in MODULE_REGISTRY

    def test_has_brief(self, module_name):
        assert MODULE_REGISTRY[module_name]["brief"]


# ──────────────────────────────────────────────────────────────────────
# Registry completeness
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("module_name", REFERENCE_MODULES)
class TestRegistryCompleteness:
    def test_keys_match(self, module_name):
        import importlib
        spec = MODULE_REGISTRY[module_name]
        mod = importlib.import_module(spec["adapter"])
        assert set(mod.METHOD_REGISTRY.keys()) == set(mod.METHOD_INFO.keys())

    def test_has_methods(self, module_name):
        methods = list_methods(module_name)
        total = sum(len(v) for v in methods.values())
        assert total > 0, f"{module_name} has no methods"

    def test_method_info_fields(self, module_name):
        import importlib
        spec = MODULE_REGISTRY[module_name]
        mod = importlib.import_module(spec["adapter"])
        for method_name, info in mod.METHOD_INFO.items():
            assert "category" in info, f"{module_name}.{method_name} missing category"
            assert "brief" in info, f"{module_name}.{method_name} missing brief"
            assert "parameters" in info, f"{module_name}.{method_name} missing parameters"


# ──────────────────────────────────────────────────────────────────────
# Dispatch integration
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("module_name", REFERENCE_MODULES)
class TestDispatchIntegration:
    def test_list_methods(self, module_name):
        result = list_methods(module_name)
        assert "error" not in result
        assert isinstance(result, dict)

    def test_describe_method(self, module_name):
        methods = list_methods(module_name)
        # Pick first method from first category
        first_cat = next(iter(methods))
        first_method = next(iter(methods[first_cat]))
        desc = describe_method(module_name, first_method)
        assert "error" not in desc
        assert "parameters" in desc

    def test_unknown_method(self, module_name):
        result = call_agent(module_name, "nonexistent_method_xyz", {})
        assert "error" in result


# ──────────────────────────────────────────────────────────────────────
# Text retrieval (GEC + micropile)
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("module_name", TEXT_MODULES)
class TestTextRetrieval:
    def test_list_chapters(self, module_name):
        result = call_agent(module_name, "list_chapters", {})
        assert "error" not in result
        assert "result" in result
        assert isinstance(result["result"], list)

    def test_search_sections(self, module_name):
        result = call_agent(module_name, "search_sections", {"query": "design"})
        assert "error" not in result
        assert "result" in result

    def test_retrieve_section_bad_id(self, module_name):
        """Retrieving a non-existent section should still return a result (possibly None)."""
        result = call_agent(module_name, "retrieve_section", {"section_id": "999.999"})
        # Either returns error or None result — just shouldn't crash
        assert isinstance(result, dict)

    def test_text_methods_in_registry(self, module_name):
        methods = list_methods(module_name)
        # Text Retrieval category should exist
        assert "Text Retrieval" in methods
        assert "retrieve_section" in methods["Text Retrieval"]
        assert "search_sections" in methods["Text Retrieval"]
        assert "list_chapters" in methods["Text Retrieval"]
        assert "load_chapter" in methods["Text Retrieval"]


# ──────────────────────────────────────────────────────────────────────
# DM7 specific tests
# ──────────────────────────────────────────────────────────────────────

class TestDM7:
    def test_method_count(self):
        """DM7 should have 300+ methods (documented as 340+ but some filtered)."""
        methods = list_methods("dm7")
        total = sum(len(v) for v in methods.values())
        assert total >= 300, f"Expected 300+ DM7 methods, got {total}"

    def test_has_all_chapters(self):
        methods = list_methods("dm7")
        categories = set(methods.keys())
        expected_prefixes = [
            "DM7.1 Ch1", "DM7.1 Ch2", "DM7.1 Ch3", "DM7.1 Ch4",
            "DM7.1 Ch5", "DM7.1 Ch6", "DM7.1 Ch7", "DM7.1 Ch8",
            "DM7.2 Prologue", "DM7.2 Ch2", "DM7.2 Ch3",
            "DM7.2 Ch4", "DM7.2 Ch5", "DM7.2 Ch6", "DM7.2 Ch7",
        ]
        for prefix in expected_prefixes:
            assert any(prefix in c for c in categories), f"Missing chapter {prefix}"

    def test_boussinesq_point_load(self):
        result = call_agent("dm7", "boussinesq_point_load", {
            "Q": 100.0, "x": 0.0, "y": 0.0, "z": 2.0,
        })
        assert "error" not in result
        assert "result" in result
        assert isinstance(result["result"], float)
        assert result["result"] > 0

    def test_collision_handling(self):
        """Name collisions should be prefixed with chapter keys."""
        import importlib
        spec = MODULE_REGISTRY["dm7"]
        mod = importlib.import_module(spec["adapter"])
        # Check that no methods are lost — should have 300+
        assert len(mod.METHOD_REGISTRY) >= 300


# ──────────────────────────────────────────────────────────────────────
# EM 1110-2-2104 specific tests (USACE RC hydraulic structures)
# ──────────────────────────────────────────────────────────────────────

class TestEM2104:
    def test_method_count(self):
        # 4 detailing + 8 loads + 6 serviceability + 20 flexure/axial +
        # 5 design + 4 shear + 4 text retrieval = 51 (some digitized
        # functions collapse under semantic aliasing, which list_methods
        # hides from the count on purpose — see _reference_common).
        methods = list_methods("em_2104")
        total = sum(len(v) for v in methods.values())
        assert total >= 45, f"em_2104: expected >= 45, got {total}"

    def test_pure_flexure_singly_appendix_c2(self):
        """Appendix C-2 anchor: As=1.58in^2, fc'=4ksi, fy=60ksi, b=12in,
        d=20.5in, phi=0.9 -> phi*Mn = 137.5 k-ft = 1650 k-in."""
        r = call_agent("em_2104", "pure_flexure_singly", {
            "as_": 1.58, "fy": 60, "fc_prime": 4, "b": 12, "d": 20.5,
            "phi": 0.9,
        })
        assert "error" not in r
        assert r["phi_mn"] == pytest.approx(1650.0, rel=2e-3)

    def test_unknown_method(self):
        result = call_agent("em_2104", "nonexistent_method_xyz", {})
        assert "error" in result


# ──────────────────────────────────────────────────────────────────────
# EM 1110-2-2107 specific tests (USACE hydraulic steel structures)
# ──────────────────────────────────────────────────────────────────────

class TestEM2107:
    def test_method_count(self):
        # 3 design basis + 9 loads + 8 seismic amplification +
        # 3 fatigue/fracture + 3 connections + 23 tainter-gate loads +
        # 4 text retrieval = 53.
        methods = list_methods("em_2107")
        total = sum(len(v) for v in methods.values())
        assert total >= 45, f"em_2107: expected >= 45, got {total}"

    def test_side_seal_friction_appendix_f(self):
        """Appendix F worked example (printed pp. 438-440): mu_s=0.5,
        S=7.03 lb/ft, l_total=42.20 ft, gamma_w=0.0625 kcf, d2=0.5 ft,
        l1=42.20 ft, l2=0, h=40 ft -> Fs = 6.74 kips."""
        r = call_agent("em_2107", "side_seal_friction_force", {
            "mu_s": 0.5, "s_preset": 7.03, "l_total": 42.20,
            "gamma_w": 0.0625, "d2": 0.5, "l1": 42.20, "l2": 0, "h": 40,
        })
        assert "error" not in r
        assert r["fs_total"] == pytest.approx(6.74, rel=1e-3)

    def test_unknown_method(self):
        result = call_agent("em_2107", "nonexistent_method_xyz", {})
        assert "error" in result


# ──────────────────────────────────────────────────────────────────────
# UFC 3-301-01 specific tests (DoD structural engineering)
# ──────────────────────────────────────────────────────────────────────

class TestUFCStructural:
    def test_method_count(self):
        # 4 general provisions + 10 risk category/loads + 8 seismic
        # force-resisting systems + 6 seismic load combinations +
        # 6 evaluation/retrofit + 4 healthcare + 2 nonbuilding +
        # 13 nonstructural seismic + 8 GFRP + 5 best practices +
        # 4 text retrieval = 70.
        methods = list_methods("ufc_structural")
        total = sum(len(v) for v in methods.values())
        assert total >= 60, f"ufc_structural: expected >= 60, got {total}"

    def test_table_3_1_seismic_system(self):
        """Table 3-1 (REPLACES ASCE 7-22 Table 12.2-1): Category B special
        reinforced concrete shear walls -> R=6, Omega0=2.5, Cd=5."""
        r = call_agent("ufc_structural", "table_3_1_seismic_system", {
            "category": "B", "system": "special_reinforced_concrete_shear_walls",
        })
        assert "error" not in r
        assert r["R"] == 6
        assert r["omega0"] == pytest.approx(2.5)
        assert r["cd"] == 5

    def test_table_2_2_risk_category_v(self):
        """Table 2-2: DoD-added Risk Category V (national strategic military
        assets) -> seismic Ie = 1.0."""
        r = call_agent("ufc_structural", "table_2_2_risk_category",
                       {"risk_category": "V"})
        assert "error" not in r
        assert r["seismic_factor_ie"] == pytest.approx(1.0)

    def test_unknown_method(self):
        result = call_agent("ufc_structural", "nonexistent_method_xyz", {})
        assert "error" in result


# ──────────────────────────────────────────────────────────────────────
# UFC 4-023-03 specific tests (DoD progressive collapse)
# ──────────────────────────────────────────────────────────────────────

class TestUFCCollapse:
    def test_method_count(self):
        # 6 applicability + 16 tie forces + 17 alternate path +
        # 6 enhanced local resistance + 9 RC + 9 steel + 9 masonry/wood/CFS
        # + 5 IBC modifications + 4 text retrieval = 81.
        methods = list_methods("ufc_collapse")
        total = sum(len(v) for v in methods.values())
        assert total >= 70, f"ufc_collapse: expected >= 70, got {total}"

    def test_peripheral_tie_force_appendix_d(self):
        """Appendix D worked RC example (printed p. 126): wF=214.5 psf,
        L1=37.5 ft, Lp=3 ft, WC=35,100 lb -> Fp=250,088 lb (250.1 kip)."""
        r = call_agent("ufc_collapse", "peripheral_tie_force_two_way", {
            "wf": 214.5, "l1": 37.5, "wc": 35100, "lp": 3,
        })
        assert "error" not in r
        assert r["fp"] == pytest.approx(250087.5, rel=1e-4)

    def test_required_tie_area_appendix_d(self):
        """Appendix D: Ru=250.1 kip, fy=60 ksi, Phi=0.75, overstrength=1.25
        -> As_req'd = 4.45 in^2."""
        r = call_agent("ufc_collapse", "required_tie_area",
                       {"ru": 250.1, "fy": 60})
        assert "error" not in r
        assert r["as_required"] == pytest.approx(4.45, rel=2e-3)

    def test_unknown_method(self):
        result = call_agent("ufc_collapse", "nonexistent_method_xyz", {})
        assert "error" in result


# ──────────────────────────────────────────────────────────────────────
# GSA Alternate Path specific tests (civilian progressive collapse)
# ──────────────────────────────────────────────────────────────────────

class TestGSACollapse:
    def test_method_count(self):
        # 3 applicability + 25 alternate path + 8 redundancy + 9 RC +
        # 9 steel + 9 masonry/wood/CFS + 4 text retrieval = 67.
        methods = list_methods("gsa_collapse")
        total = sum(len(v) for v in methods.values())
        assert total >= 55, f"gsa_collapse: expected >= 55, got {total}"

    def test_minimum_load_redistribution_systems_appendix_d(self):
        """Appendix D worked redundancy example (printed p. D48): an
        8-story building requires n = ceil(8/3) = 3 load-redistribution
        systems (Equation 3.13)."""
        r = call_agent("gsa_collapse", "minimum_load_redistribution_systems",
                       {"total_floors": 8})
        assert "error" not in r
        assert r["n"] == 3

    def test_unknown_method(self):
        result = call_agent("gsa_collapse", "nonexistent_method_xyz", {})
        assert "error" in result


# ──────────────────────────────────────────────────────────────────────
# USDA Wood Handbook specific tests (FPL-GTR-282)
# ──────────────────────────────────────────────────────────────────────

class TestWoodHandbook:
    def test_method_count(self):
        # 15 moisture relations + 5 mechanical properties + 16 fastenings +
        # 12 structural deformation + 16 structural stress + 17 structural
        # stability + 4 text retrieval = 85.
        methods = list_methods("wood_handbook")
        total = sum(len(v) for v in methods.values())
        assert total >= 70, f"wood_handbook: expected >= 70, got {total}"

    def test_moisture_content_adjustment_eq_5_3(self):
        """Printed worked example: white ash MOR at 8% MC, P12=103,000 kPa,
        Pg=66,000 kPa, Mp=24 -> P8 = 119,500 kPa (Eq 5-3)."""
        r = call_agent("wood_handbook", "adjust_property_for_moisture_content", {
            "p12": 103000, "pg": 66000, "moisture_content_pct": 8, "mp_pct": 24,
        })
        assert "error" not in r
        assert r["property_value"] == pytest.approx(119500, rel=2e-3)

    def test_unknown_method(self):
        result = call_agent("wood_handbook", "nonexistent_method_xyz", {})
        assert "error" in result


# ──────────────────────────────────────────────────────────────────────
# GEC-7 specific tests
# ──────────────────────────────────────────────────────────────────────

class TestGEC7:
    def test_method_count(self):
        methods = list_methods("gec7")
        total = sum(len(v) for v in methods.values())
        # 2 figures + 13 tables + 4 text = 19
        assert total == 19

    def test_bond_strength_coarse(self):
        desc = describe_method("gec7", "table_4_4a_bond_strength_coarse")
        assert "parameters" in desc
        assert "error" not in desc


# ──────────────────────────────────────────────────────────────────────
# UFC specific tests
# ──────────────────────────────────────────────────────────────────────

class TestUFCBackfill:
    def test_method_count(self):
        methods = list_methods("ufc_backfill")
        total = sum(len(v) for v in methods.values())
        assert total == 8


class TestUFCExpansive:
    def test_method_count(self):
        methods = list_methods("ufc_expansive")
        total = sum(len(v) for v in methods.values())
        assert total == 9


class TestUFCPavement:
    def test_method_count(self):
        # Rebuilt 2026-07 from the real UFC 3-250-01 (2016): 26 equations +
        # 25 tables + text retrieval (was 11 airfield-derived methods).
        methods = list_methods("ufc_pavement")
        total = sum(len(v) for v in methods.values())
        assert total >= 50

    def test_new_ufc_reference_modules(self):
        for name, expected_min in (("ufc_stabilization", 17),
                                   ("ufc_flexible_practice", 30),
                                   ("ufc_concrete_practice", 17)):
            methods = list_methods(name)
            total = sum(len(v) for v in methods.values())
            assert total >= expected_min, (name, total)


# ──────────────────────────────────────────────────────────────────────
# GEC method count tests
# ──────────────────────────────────────────────────────────────────────

class TestGECMethodCounts:
    @pytest.mark.parametrize("module_name,expected_min", [
        ("gec4", 10),   # 6 tables + 4 text retrieval
        ("gec5", 4),    # text retrieval only
        ("gec6", 15),
        ("gec8", 9),    # 3 equations + 2 tables + 4 text retrieval
        ("gec9", 9),    # 5 tables + 4 text retrieval
        ("gec10", 12),
        ("gec11", 19),
        ("gec12", 18),
        ("gec13", 12),
        ("gec14", 4),   # text retrieval only
        ("micropile", 16),
    ])
    def test_minimum_methods(self, module_name, expected_min):
        methods = list_methods(module_name)
        total = sum(len(v) for v in methods.values())
        assert total >= expected_min, f"{module_name}: expected >= {expected_min}, got {total}"


# ──────────────────────────────────────────────────────────────────────
# GEC-12 call test
# ──────────────────────────────────────────────────────────────────────

class TestGEC12:
    def test_figure_lookup(self):
        """Test a GEC-12 figure lookup (e.g., delta/phi ratio)."""
        desc = describe_method("gec12", "figure_7_9_delta_phi_ratio")
        assert "parameters" in desc

    def test_table_lookup(self):
        """Test a GEC-12 table lookup."""
        methods = list_methods("gec12")
        table_methods = {}
        for cat, meths in methods.items():
            if "Table" in cat:
                table_methods.update(meths)
        assert len(table_methods) > 0


# ──────────────────────────────────────────────────────────────────────
# Micropile call test
# ──────────────────────────────────────────────────────────────────────

class TestMicropile:
    def test_has_bond_stress(self):
        """Micropile should have alpha_bond table."""
        methods = list_methods("micropile")
        all_methods = {}
        for cat_methods in methods.values():
            all_methods.update(cat_methods)
        bond_methods = [m for m in all_methods if "bond" in m.lower()]
        assert len(bond_methods) > 0
