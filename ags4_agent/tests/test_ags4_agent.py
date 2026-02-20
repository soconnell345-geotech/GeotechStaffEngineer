"""
Tests for ags4_agent — AGS4 data format reader/validator.

Tier 1: No python-ags4 required (result dataclasses, validation, utilities, Foundry metadata)
Tier 2: Requires python-ags4 (integration tests with sample AGS4 data)
"""

import json
import pytest

from ags4_agent.ags4_utils import has_ags4
from ags4_agent.results import AGS4ReadResult, AGS4ValidationResult

requires_ags4 = pytest.mark.skipif(
    not has_ags4(), reason="python-ags4 not installed"
)


# Minimal valid AGS4 content for testing
SAMPLE_AGS4 = '''"GROUP","PROJ"
"HEADING","PROJ_ID","PROJ_NAME","PROJ_LOC"
"UNIT","","",""
"TYPE","ID","X","X"
"DATA","ABC123","Test Project","Test Location"

"GROUP","HOLE"
"HEADING","HOLE_ID","HOLE_NATE","HOLE_NATN","HOLE_GL"
"UNIT","","m","m","m"
"TYPE","ID","2DP","2DP","2DP"
"DATA","BH-01","100.00","200.00","50.00"
"DATA","BH-02","150.00","250.00","48.50"

"GROUP","ISPT"
"HEADING","HOLE_ID","ISPT_TOP","ISPT_NVAL"
"UNIT","","m",""
"TYPE","ID","2DP","0DP"
"DATA","BH-01","1.50","8"
"DATA","BH-01","3.00","15"
"DATA","BH-01","4.50","22"
"DATA","BH-02","1.50","5"
"DATA","BH-02","3.00","12"
'''


# =====================================================================
# Tier 1: AGS4ReadResult defaults
# =====================================================================

class TestAGS4ReadResultDefaults:

    def test_default_construction(self):
        r = AGS4ReadResult()
        assert r.filepath == ""
        assert r.n_groups == 0
        assert r.group_names == []
        assert r.tables is None

    def test_construction_with_values(self):
        r = AGS4ReadResult(
            filepath="test.ags",
            n_groups=3,
            group_names=["PROJ", "HOLE", "ISPT"],
            group_row_counts={"PROJ": 1, "HOLE": 2, "ISPT": 5},
        )
        assert r.n_groups == 3
        assert "HOLE" in r.group_names

    def test_summary_contains_groups(self):
        r = AGS4ReadResult(
            filepath="test.ags",
            n_groups=2,
            group_names=["PROJ", "HOLE"],
            group_row_counts={"PROJ": 1, "HOLE": 3},
        )
        s = r.summary()
        assert "PROJ" in s
        assert "HOLE" in s
        assert "2" in s  # n_groups

    def test_to_dict_keys(self):
        r = AGS4ReadResult(
            filepath="test.ags",
            n_groups=2,
            group_names=["PROJ", "HOLE"],
            group_row_counts={"PROJ": 1, "HOLE": 3},
        )
        d = r.to_dict()
        assert "filepath" in d
        assert "n_groups" in d
        assert "group_names" in d
        assert "group_row_counts" in d

    def test_to_dict_includes_tables_when_present(self):
        r = AGS4ReadResult(
            filepath="test.ags",
            n_groups=1,
            group_names=["PROJ"],
            group_row_counts={"PROJ": 1},
            tables={"PROJ": [{"PROJ_ID": "ABC", "PROJ_NAME": "Test"}]},
        )
        d = r.to_dict()
        assert "tables" in d
        assert "PROJ" in d["tables"]

    def test_to_dict_no_tables_when_none(self):
        r = AGS4ReadResult(n_groups=1, group_names=["PROJ"])
        d = r.to_dict()
        assert "tables" not in d

    def test_to_dict_json_serializable(self):
        r = AGS4ReadResult(
            filepath="test.ags",
            n_groups=1,
            group_names=["PROJ"],
            group_row_counts={"PROJ": 1},
            tables={"PROJ": [{"PROJ_ID": "ABC"}]},
        )
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)


# =====================================================================
# Tier 1: AGS4ValidationResult defaults
# =====================================================================

class TestAGS4ValidationResultDefaults:

    def test_default_construction(self):
        r = AGS4ValidationResult()
        assert r.n_errors == 0
        assert r.is_valid is True

    def test_construction_with_errors(self):
        r = AGS4ValidationResult(
            filepath="test.ags",
            n_errors=3,
            n_warnings=1,
            n_fyi=0,
            is_valid=False,
        )
        assert r.n_errors == 3
        assert r.is_valid is False

    def test_summary_valid(self):
        r = AGS4ValidationResult(filepath="test.ags", is_valid=True)
        s = r.summary()
        assert "VALID" in s

    def test_summary_invalid(self):
        r = AGS4ValidationResult(filepath="test.ags", n_errors=2, is_valid=False)
        s = r.summary()
        assert "INVALID" in s

    def test_to_dict_keys(self):
        r = AGS4ValidationResult(filepath="test.ags")
        d = r.to_dict()
        assert "filepath" in d
        assert "n_errors" in d
        assert "is_valid" in d
        assert "errors" in d

    def test_to_dict_json_serializable(self):
        r = AGS4ValidationResult(
            filepath="test.ags",
            errors={"Rule 1": ["Error message"]},
        )
        s = json.dumps(r.to_dict())
        assert isinstance(s, str)


# =====================================================================
# Tier 1: Input validation
# =====================================================================

class TestInputValidation:

    def test_no_filepath_or_content(self):
        from ags4_agent.ags4_reader import read_ags4
        with pytest.raises(ValueError, match="Either filepath or content"):
            read_ags4()

    def test_both_filepath_and_content(self):
        from ags4_agent.ags4_reader import read_ags4
        with pytest.raises(ValueError, match="Provide either"):
            read_ags4(filepath="test.ags", content="data")

    def test_validate_no_input(self):
        from ags4_agent.ags4_reader import validate_ags4
        with pytest.raises(ValueError, match="Either filepath or content"):
            validate_ags4()

    def test_validate_both_inputs(self):
        from ags4_agent.ags4_reader import validate_ags4
        with pytest.raises(ValueError, match="Provide either"):
            validate_ags4(filepath="test.ags", content="data")

    def test_empty_content(self):
        from ags4_agent.ags4_reader import _validate_read_inputs
        with pytest.raises(ValueError, match="non-empty"):
            _validate_read_inputs("", is_content=True)

    def test_empty_filepath(self):
        from ags4_agent.ags4_reader import _validate_read_inputs
        with pytest.raises(ValueError, match="non-empty"):
            _validate_read_inputs("", is_content=False)


# =====================================================================
# Tier 1: Utilities
# =====================================================================

class TestUtilities:

    def test_has_ags4_returns_bool(self):
        assert isinstance(has_ags4(), bool)


# =====================================================================
# Tier 1: Foundry metadata
# =====================================================================

class TestFoundryMetadata:

    def test_list_methods_all(self):
        from ags4_agent_foundry import ags4_list_methods
        result = json.loads(ags4_list_methods(""))
        assert "Data Import" in result
        assert "Validation" in result

    def test_list_methods_filtered(self):
        from ags4_agent_foundry import ags4_list_methods
        result = json.loads(ags4_list_methods("Data Import"))
        assert "read_ags4" in result["Data Import"]

    def test_list_methods_bad_category(self):
        from ags4_agent_foundry import ags4_list_methods
        result = json.loads(ags4_list_methods("nonexistent"))
        assert "error" in result

    def test_describe_read(self):
        from ags4_agent_foundry import ags4_describe_method
        result = json.loads(ags4_describe_method("read_ags4"))
        assert "parameters" in result
        assert "filepath" in result["parameters"]

    def test_describe_validate(self):
        from ags4_agent_foundry import ags4_describe_method
        result = json.loads(ags4_describe_method("validate_ags4"))
        assert "parameters" in result

    def test_describe_unknown(self):
        from ags4_agent_foundry import ags4_describe_method
        result = json.loads(ags4_describe_method("nonexistent"))
        assert "error" in result

    def test_agent_invalid_json(self):
        from ags4_agent_foundry import ags4_agent
        result = json.loads(ags4_agent("read_ags4", "not json"))
        assert "error" in result

    def test_agent_unknown_method(self):
        from ags4_agent_foundry import ags4_agent
        result = json.loads(ags4_agent("nonexistent", "{}"))
        assert "error" in result


# =====================================================================
# Tier 2: Read integration (requires python-ags4)
# =====================================================================

@requires_ags4
class TestReadIntegration:

    def test_read_from_content(self):
        from ags4_agent import read_ags4
        r = read_ags4(content=SAMPLE_AGS4)
        assert r.n_groups == 3
        assert "PROJ" in r.group_names
        assert "HOLE" in r.group_names
        assert "ISPT" in r.group_names

    def test_read_row_counts(self):
        from ags4_agent import read_ags4
        r = read_ags4(content=SAMPLE_AGS4)
        assert r.group_row_counts["PROJ"] == 1
        assert r.group_row_counts["HOLE"] == 2
        assert r.group_row_counts["ISPT"] == 5

    def test_read_includes_tables(self):
        from ags4_agent import read_ags4
        r = read_ags4(content=SAMPLE_AGS4, include_data=True)
        assert r.tables is not None
        assert "PROJ" in r.tables
        assert len(r.tables["PROJ"]) > 0

    def test_read_excludes_tables(self):
        from ags4_agent import read_ags4
        r = read_ags4(content=SAMPLE_AGS4, include_data=False)
        assert r.tables is None

    def test_read_to_dict_json_serializable(self):
        from ags4_agent import read_ags4
        r = read_ags4(content=SAMPLE_AGS4)
        s = json.dumps(r.to_dict(), default=str)
        assert isinstance(s, str)

    def test_read_source_name_string(self):
        from ags4_agent import read_ags4
        r = read_ags4(content=SAMPLE_AGS4)
        assert r.filepath == "<string>"


# =====================================================================
# Tier 2: Validate integration (requires python-ags4)
# =====================================================================

@requires_ags4
class TestValidateIntegration:

    def test_validate_from_content(self):
        from ags4_agent import validate_ags4
        r = validate_ags4(content=SAMPLE_AGS4)
        assert isinstance(r.n_errors, int)
        assert isinstance(r.is_valid, bool)

    def test_validate_to_dict_json_serializable(self):
        from ags4_agent import validate_ags4
        r = validate_ags4(content=SAMPLE_AGS4)
        s = json.dumps(r.to_dict(), default=str)
        assert isinstance(s, str)

    def test_validate_source_name(self):
        from ags4_agent import validate_ags4
        r = validate_ags4(content=SAMPLE_AGS4)
        assert r.filepath == "<string>"


# =====================================================================
# Tier 2: Foundry integration (requires python-ags4)
# =====================================================================

@requires_ags4
class TestFoundryIntegration:

    def test_foundry_read(self):
        from ags4_agent_foundry import ags4_agent
        params = {"content": SAMPLE_AGS4}
        result = json.loads(ags4_agent("read_ags4", json.dumps(params)))
        assert "error" not in result
        assert result["n_groups"] == 3
        assert "PROJ" in result["group_names"]

    def test_foundry_validate(self):
        from ags4_agent_foundry import ags4_agent
        params = {"content": SAMPLE_AGS4}
        result = json.loads(ags4_agent("validate_ags4", json.dumps(params)))
        assert "error" not in result
        assert "is_valid" in result
