"""Argument types at the door for reference modules (2026-11-10 review,
item 4).

A reference parameter declared a bool refuses a string ("no", "false":
Python read any non-empty string as True), naming the parameter and the
values it takes; a call that leaves a required parameter out gets an error
that lists every parameter with its type and description, so the next call
can be right. Calculation modules are untouched. The module and its values
are made up.
"""

import types

import pytest

from funhouse_agent import dispatch
from funhouse_agent.adapters._reference_common import build_lookup_registry


def made_up_table(soil_type: str, detailed: bool = False,
                  depth_ft: float = 0.0) -> dict:
    """Made-up table of made-up values (Table 9-9).

    Parameters
    ----------
    soil_type : str
        'made_up_sand' or 'made_up_clay'.
    detailed : bool, optional
        Include the made-up detail columns.
    depth_ft : float, optional
        Made-up depth (ft).

    Returns
    -------
    dict
        The made-up row.
    """
    return {"soil_type": soil_type, "detailed": detailed,
            "value_ksf": 1.25}


def made_up_flag_from_default(include_notes=False) -> dict:
    """Made-up lookup whose flag has no annotation, only a bool default.

    Parameters
    ----------
    include_notes
        Include the made-up notes.
    """
    return {"include_notes": include_notes}


def _fake_module():
    src = types.ModuleType("made_up_reference")
    src.made_up_table = made_up_table
    src.made_up_flag_from_default = made_up_flag_from_default
    registry, info = build_lookup_registry([(src, "Made-Up Tables",
                                             "MADE-UP-9")])

    def keyed(params):
        return {"row": params["row_id"]}
    registry["made_up_keyed"] = keyed
    info["made_up_keyed"] = {
        "category": "Made-Up Tables", "brief": "A hand-described lookup.",
        "reference": "MADE-UP-9",
        "parameters": {"row_id": {"type": "str", "required": True,
                                  "description": "A made-up row id."}}}
    mod = types.ModuleType("made_up_adapter")
    mod.METHOD_REGISTRY = registry
    mod.METHOD_INFO = info
    return mod


@pytest.fixture
def reference_fake(monkeypatch):
    """The made-up adapter under a reference module's name."""
    assert "micropile" in dispatch.REFERENCE_MODULES
    monkeypatch.setitem(dispatch._loaded_adapters, "micropile",
                        _fake_module())
    return "micropile"


class TestBoolRefusesStrings:
    @pytest.mark.parametrize("word", ["no", "false", "False", "yes", "0"])
    def test_a_string_for_a_bool_is_refused(self, reference_fake, word):
        out = dispatch.call_agent(reference_fake, "made_up_table",
                                  {"soil_type": "made_up_sand",
                                   "detailed": word})
        assert "error" in out
        assert "'detailed'" in out["error"]
        assert "true or false" in out["error"]
        assert out["parameter"] == "detailed"
        assert out["accepted_values"] == [True, False]

    def test_a_json_boolean_runs(self, reference_fake):
        out = dispatch.call_agent(reference_fake, "made_up_table",
                                  {"soil_type": "made_up_sand",
                                   "detailed": False})
        assert out.get("detailed") is False
        assert out["value_ksf"] == 1.25

    def test_a_bool_default_declares_a_bool(self, reference_fake):
        out = dispatch.call_agent(reference_fake,
                                  "made_up_flag_from_default",
                                  {"include_notes": "no"})
        assert out["parameter"] == "include_notes"

    def test_a_string_for_a_str_parameter_runs(self, reference_fake):
        out = dispatch.call_agent(reference_fake, "made_up_table",
                                  {"soil_type": "made_up_clay"})
        assert out["soil_type"] == "made_up_clay"

    def test_a_real_text_tool_flag(self):
        """retrieve_section's include_subsections, on a real reference
        module: the string "false" no longer inlines the subsections."""
        out = dispatch.call_agent("gec12", "retrieve_section",
                                  {"section_id": "7",
                                   "include_subsections": "false"})
        assert out["parameter"] == "include_subsections"

    def test_a_calculation_module_is_untouched(self, monkeypatch):
        """The same made-up function under a calculation module's name:
        the reference check does not run there."""
        assert "bearing_capacity" not in dispatch.REFERENCE_MODULES
        monkeypatch.setitem(dispatch._loaded_adapters, "bearing_capacity",
                            _fake_module())
        out = dispatch.call_agent("bearing_capacity", "made_up_table",
                                  {"soil_type": "made_up_sand",
                                   "detailed": "no"})
        assert "error" not in out
        assert "missing_parameters" not in out


class TestMissingArgumentsAreDescribed:
    def test_no_arguments(self, reference_fake):
        out = dispatch.call_agent(reference_fake, "made_up_table", {})
        assert "error" in out
        assert out["missing_parameters"] == ["soil_type"]
        params = out["parameters"]
        assert params["soil_type"].startswith("str, required. ")
        assert "'made_up_sand' or 'made_up_clay'" in params["soil_type"]
        assert params["depth_ft"] == ("float, default 0.0. Made-up depth "
                                      "(ft).")
        assert "needs 'soil_type'" in out["error"]

    def test_a_wrapper_that_reads_the_key_itself(self, reference_fake):
        """A hand-described lookup that fails on the missing key (a
        KeyError, not a signature error) gets the same help."""
        out = dispatch.call_agent(reference_fake, "made_up_keyed", {})
        assert out["missing_parameters"] == ["row_id"]
        assert out["error"].startswith("micropile.made_up_keyed needs "
                                       "'row_id' (left out).")
        assert "A made-up row id." in out["parameters"]["row_id"]

    def test_descriptions_are_trimmed(self, reference_fake, monkeypatch):
        mod = dispatch._loaded_adapters[reference_fake]
        spec = mod.METHOD_INFO["made_up_table"]["parameters"]["soil_type"]
        monkeypatch.setitem(spec, "description", "word " * 200)
        out = dispatch.call_agent(reference_fake, "made_up_table", {})
        assert len(out["parameters"]["soil_type"]) < 240
        assert out["parameters"]["soil_type"].endswith("...")

    def test_a_successful_call_is_unchanged(self, reference_fake):
        out = dispatch.call_agent(reference_fake, "made_up_table",
                                  {"soil_type": "made_up_sand"})
        assert "missing_parameters" not in out
        assert "parameters" not in out

    def test_an_error_with_everything_given_is_unchanged(self,
                                                         reference_fake):
        out = dispatch.call_agent(reference_fake, "made_up_table",
                                  {"soil_type": "made_up_sand",
                                   "not_a_parameter": 1})
        assert "error" in out
        assert "missing_parameters" not in out
