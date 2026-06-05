"""Tests for semantic method aliases on document-id-prefixed reference methods.

Reference lookups are named after the source artifact (e.g. ``table_5_3_alpha_bond``).
``register_semantic_aliases`` adds guessable aliases (mechanical prefix-strip plus
a curated map) so an LLM can call the value by an intuitive name without knowing
the table/figure number. Aliases are callable + describable but hidden from
``list_methods``.
"""

import pytest

from funhouse_agent.adapters._reference_common import _mechanical_alias
from funhouse_agent.dispatch import call_agent, describe_method, list_methods


class TestMechanicalAlias:
    @pytest.mark.parametrize("name,expected", [
        ("table_5_3_alpha_bond", "alpha_bond"),
        ("figure_10_6_alpha_clay", "alpha_clay"),
        ("equation_10_21_rock_socket_side", "rock_socket_side"),
        ("figure_7_10_to_13_kd", "kd"),
        ("table_20_3_1_site_class_from_vs30", "site_class_from_vs30"),
        ("table_4_4a_bond_strength_coarse", "bond_strength_coarse"),
        ("table_n_factor", "n_factor"),          # single letter is semantic, not an id
        ("bearing_capacity_vesic", ""),          # no doc-id prefix -> no alias
        ("retrieve_section", ""),
    ])
    def test_strip(self, name, expected):
        assert _mechanical_alias(name) == expected


class TestAliasResolution:
    def test_curated_alias_matches_canonical(self):
        params = {"soil_type": "sand_coarse_dense", "micropile_type": "B"}
        canonical = call_agent("micropile", "table_5_3_alpha_bond", params)
        alias = call_agent("micropile", "grout_bond_strength", params)
        assert "error" not in alias
        assert alias == canonical
        assert alias["alpha_bond_min_kpa"] == 120

    def test_mechanical_alias_resolves(self):
        params = {"soil_type": "sand_coarse_dense", "micropile_type": "B"}
        assert "error" not in call_agent("micropile", "alpha_bond", params)

    def test_alias_describable_with_marker(self):
        d = describe_method("gec10", "alpha_adhesion_factor")
        assert "error" not in d
        assert d.get("alias_of") == "figure_10_6_alpha_clay"

    def test_aliases_hidden_from_list_methods(self):
        listed = {m for cat in list_methods("micropile").values() for m in cat}
        assert "table_5_3_alpha_bond" in listed       # canonical shown
        assert "grout_bond_strength" not in listed     # alias hidden
        assert "alpha_bond" not in listed

    def test_ufc_and_dm7_aliases_callable(self):
        from funhouse_agent.adapters import ufc_pavement_adapter, dm7_adapter
        assert "modulus_subgrade_reaction" in ufc_pavement_adapter.METHOD_REGISTRY
        assert "Tv_from_U" in dm7_adapter.METHOD_REGISTRY
