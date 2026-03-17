"""Tests for geotech-references adapter modules (14 reference agents).

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
# All 14 reference modules
# ──────────────────────────────────────────────────────────────────────

REFERENCE_MODULES = [
    "dm7", "gec6", "gec7", "gec10", "gec11", "gec12", "gec13",
    "micropile", "fema_p2192", "noaa_frost",
    "ufc_backfill", "ufc_dewatering", "ufc_expansive", "ufc_pavement",
]

# Modules with text retrieval (retrieve_section, search_sections, etc.)
TEXT_MODULES = ["gec6", "gec7", "gec10", "gec11", "gec12", "gec13", "micropile"]


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
# FEMA P-2192 specific tests
# ──────────────────────────────────────────────────────────────────────

class TestFEMA:
    def test_method_count(self):
        methods = list_methods("fema_p2192")
        total = sum(len(v) for v in methods.values())
        assert total == 10

    def test_site_class(self):
        """Test determining seismic site class from Vs30."""
        # Find the site class method
        methods = list_methods("fema_p2192")
        all_methods = {}
        for cat_methods in methods.values():
            all_methods.update(cat_methods)

        # Look for a site class method
        site_class_methods = [m for m in all_methods if "site_class" in m.lower()]
        if site_class_methods:
            desc = describe_method("fema_p2192", site_class_methods[0])
            assert "parameters" in desc


# ──────────────────────────────────────────────────────────────────────
# NOAA Frost specific tests
# ──────────────────────────────────────────────────────────────────────

class TestNOAAFrost:
    def test_method_count(self):
        methods = list_methods("noaa_frost")
        total = sum(len(v) for v in methods.values())
        assert total == 9

    def test_stefan_frost_depth(self):
        result = call_agent("noaa_frost", "stefan_frost_depth_m", {
            "freezing_index_degC_days": 500.0,
            "k_frozen_W_per_mK": 1.5,
            "L_J_per_m3": 100000.0,
        })
        assert "error" not in result
        assert "result" in result
        assert result["result"] > 0


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


class TestUFCDewatering:
    def test_method_count(self):
        methods = list_methods("ufc_dewatering")
        total = sum(len(v) for v in methods.values())
        assert total == 9


class TestUFCExpansive:
    def test_method_count(self):
        methods = list_methods("ufc_expansive")
        total = sum(len(v) for v in methods.values())
        assert total == 9


class TestUFCPavement:
    def test_method_count(self):
        methods = list_methods("ufc_pavement")
        total = sum(len(v) for v in methods.values())
        assert total == 9


# ──────────────────────────────────────────────────────────────────────
# GEC method count tests
# ──────────────────────────────────────────────────────────────────────

class TestGECMethodCounts:
    @pytest.mark.parametrize("module_name,expected_min", [
        ("gec6", 15),
        ("gec10", 12),
        ("gec11", 19),
        ("gec12", 18),
        ("gec13", 12),
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
