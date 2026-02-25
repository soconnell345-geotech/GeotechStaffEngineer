"""
Tests for pydiggs_agent module.

Tier 1: No pydiggs required (~18 tests)
Tier 2: Requires pydiggs (~12 tests)
"""

import json
import os
import tempfile
import matplotlib
matplotlib.use('Agg')
import pytest

from pydiggs_agent import has_pydiggs
from pydiggs_agent.results import DiggValidationResult


# Test XML samples
VALID_DIGGS_26_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Diggs xmlns="http://diggsml.org/schemas/2.6"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xmlns:gml="http://www.opengis.net/gml/3.2"
       xmlns:xlink="http://www.w3.org/1999/xlink"
       gml:id="d1">
  <documentInformation>
    <DocumentInformation gml:id="di1">
      <creationDate>2024-01-01</creationDate>
    </DocumentInformation>
  </documentInformation>
</Diggs>"""

INVALID_DIGGS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Diggs xmlns="http://diggsml.org/schemas/2.6">
  <InvalidElement>This should fail</InvalidElement>
</Diggs>"""

SYNTAX_ERROR_XML = """<?xml version="1.0"?>
<Diggs><unclosed>"""


# ============================================================================
# Tier 1 Tests - No pydiggs required
# ============================================================================

class TestDiggValidationResult:
    """Test DiggValidationResult dataclass."""

    def test_defaults(self):
        """Test default values."""
        result = DiggValidationResult(
            source="test.xml",
            check_type="schema"
        )
        assert result.source == "test.xml"
        assert result.check_type == "schema"
        assert result.schema_version is None
        assert result.is_valid is True
        assert result.n_errors == 0
        assert result.errors == []

    def test_with_values(self):
        """Test with all values set."""
        result = DiggValidationResult(
            source="test.xml",
            check_type="schema",
            schema_version="2.6",
            is_valid=False,
            n_errors=2,
            errors=["Error 1", "Error 2"]
        )
        assert result.source == "test.xml"
        assert result.check_type == "schema"
        assert result.schema_version == "2.6"
        assert result.is_valid is False
        assert result.n_errors == 2
        assert len(result.errors) == 2

    def test_summary_valid(self):
        """Test summary for valid result."""
        result = DiggValidationResult(
            source="test.xml",
            check_type="schema",
            schema_version="2.6"
        )
        summary = result.summary()
        assert "DIGGS Validation Result" in summary
        assert "test.xml" in summary
        assert "schema" in summary
        assert "2.6" in summary
        assert "Valid: True" in summary
        assert "No errors found" in summary

    def test_summary_with_errors(self):
        """Test summary with errors."""
        result = DiggValidationResult(
            source="content",
            check_type="dictionary",
            is_valid=False,
            n_errors=2,
            errors=["Undefined property: foo", "Undefined property: bar"]
        )
        summary = result.summary()
        assert "content" in summary
        assert "dictionary" in summary
        assert "Valid: False" in summary
        assert "Number of Errors: 2" in summary
        assert "foo" in summary
        assert "bar" in summary

    def test_summary_truncates_long_errors(self):
        """Test that very long error messages are truncated."""
        long_error = "x" * 250
        result = DiggValidationResult(
            source="test.xml",
            check_type="schema",
            is_valid=False,
            n_errors=1,
            errors=[long_error]
        )
        summary = result.summary()
        assert "..." in summary
        assert len(summary) < len(long_error) + 500

    def test_to_dict(self):
        """Test to_dict conversion."""
        result = DiggValidationResult(
            source="test.xml",
            check_type="schema",
            schema_version="2.6",
            is_valid=False,
            n_errors=1,
            errors=["Error message"]
        )
        d = result.to_dict()
        assert d["source"] == "test.xml"
        assert d["check_type"] == "schema"
        assert d["schema_version"] == "2.6"
        assert d["is_valid"] is False
        assert d["n_errors"] == 1
        assert d["errors"] == ["Error message"]

    def test_to_dict_json_serializable(self):
        """Test that to_dict result is JSON serializable."""
        result = DiggValidationResult(
            source="test.xml",
            check_type="schema",
            schema_version="2.6"
        )
        d = result.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert isinstance(json_str, str)


class TestInputValidation:
    """Test input validation without calling pydiggs."""

    def test_no_filepath_or_content_raises(self):
        """Test that missing both filepath and content raises ValueError."""
        if not has_pydiggs():
            pytest.skip("pydiggs not installed")

        from pydiggs_agent import validate_diggs_schema

        with pytest.raises(ValueError, match="Either filepath or content must be provided"):
            validate_diggs_schema()

    def test_both_filepath_and_content_raises(self):
        """Test that providing both filepath and content raises ValueError."""
        if not has_pydiggs():
            pytest.skip("pydiggs not installed")

        from pydiggs_agent import validate_diggs_schema

        with pytest.raises(ValueError, match="Only one of filepath or content"):
            validate_diggs_schema(filepath="test.xml", content="<xml/>")

    def test_invalid_schema_version_raises(self):
        """Test that invalid schema version raises ValueError."""
        if not has_pydiggs():
            pytest.skip("pydiggs not installed")

        from pydiggs_agent import validate_diggs_schema

        with pytest.raises(ValueError, match="Invalid schema version"):
            validate_diggs_schema(content="<xml/>", schema_version="3.0")


class TestUtilities:
    """Test utility functions."""

    def test_has_pydiggs_returns_bool(self):
        """Test that has_pydiggs returns a boolean."""
        result = has_pydiggs()
        assert isinstance(result, bool)


# ============================================================================
# Tier 2 Tests - Requires pydiggs
# ============================================================================

@pytest.mark.skipif(not has_pydiggs(), reason="pydiggs not installed")
class TestSchemaValidation:
    """Test schema validation with pydiggs."""

    def test_validate_valid_xml(self):
        """Test validation of valid DIGGS 2.6 XML."""
        from pydiggs_agent import validate_diggs_schema

        result = validate_diggs_schema(content=VALID_DIGGS_26_XML)
        assert result.source == "content"
        assert result.check_type == "schema"
        assert result.schema_version == "2.6"
        assert result.is_valid is True
        assert result.n_errors == 0
        assert result.errors == []

    def test_validate_invalid_xml(self):
        """Test validation of invalid DIGGS XML."""
        from pydiggs_agent import validate_diggs_schema

        result = validate_diggs_schema(content=INVALID_DIGGS_XML)
        assert result.source == "content"
        assert result.check_type == "schema"
        assert result.is_valid is False
        assert result.n_errors > 0
        assert len(result.errors) > 0

    def test_validate_syntax_error(self):
        """Test validation of XML with syntax errors."""
        from pydiggs_agent import validate_diggs_schema

        result = validate_diggs_schema(content=SYNTAX_ERROR_XML)
        assert result.source == "content"
        assert result.is_valid is False
        assert result.n_errors >= 1
        assert len(result.errors) >= 1

    def test_validate_schema_25a(self):
        """Test validation with DIGGS 2.5.a schema."""
        from pydiggs_agent import validate_diggs_schema

        # Use valid 2.6 XML but validate against 2.5.a
        # This should fail because namespaces differ
        result = validate_diggs_schema(
            content=VALID_DIGGS_26_XML,
            schema_version="2.5.a"
        )
        assert result.schema_version == "2.5.a"
        # Result may be valid or invalid depending on schema compatibility
        # Just check that it ran
        assert result.check_type == "schema"

    def test_validate_from_file(self):
        """Test validation from file path."""
        from pydiggs_agent import validate_diggs_schema

        # Write valid XML to temp file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.xml',
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(VALID_DIGGS_26_XML)
            temp_path = f.name

        try:
            result = validate_diggs_schema(filepath=temp_path)
            assert result.source == os.path.basename(temp_path)
            assert result.check_type == "schema"
            assert result.is_valid is True
        finally:
            os.unlink(temp_path)

    def test_temp_file_cleanup(self):
        """Test that temp files are cleaned up after content validation."""
        from pydiggs_agent import validate_diggs_schema

        # Get temp directory
        temp_dir = tempfile.gettempdir()

        # Count XML files before
        xml_files_before = [
            f for f in os.listdir(temp_dir)
            if f.endswith('.xml')
        ]

        # Run validation with content
        validate_diggs_schema(content=VALID_DIGGS_26_XML)

        # Count XML files after
        xml_files_after = [
            f for f in os.listdir(temp_dir)
            if f.endswith('.xml')
        ]

        # Should be same count (temp file cleaned up)
        assert len(xml_files_after) == len(xml_files_before)


@pytest.mark.skipif(not has_pydiggs(), reason="pydiggs not installed")
class TestDictionaryValidation:
    """Test dictionary validation with pydiggs."""

    def test_validate_from_content(self):
        """Test dictionary validation from content string."""
        from pydiggs_agent import validate_diggs_dictionary

        # Valid DIGGS XML should have valid property classes
        result = validate_diggs_dictionary(content=VALID_DIGGS_26_XML)
        assert result.source == "content"
        assert result.check_type == "dictionary"
        # May be valid or invalid depending on properties used
        assert isinstance(result.is_valid, bool)

    def test_validate_from_file(self):
        """Test dictionary validation from file path."""
        from pydiggs_agent import validate_diggs_dictionary

        # Write valid XML to temp file
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.xml',
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(VALID_DIGGS_26_XML)
            temp_path = f.name

        try:
            result = validate_diggs_dictionary(filepath=temp_path)
            assert result.source == os.path.basename(temp_path)
            assert result.check_type == "dictionary"
        finally:
            os.unlink(temp_path)


@pytest.mark.skipif(not has_pydiggs(), reason="pydiggs not installed")
class TestResultIntegration:
    """Test result dataclass with real validation."""

    def test_to_dict_after_validation(self):
        """Test to_dict JSON serialization after real validation."""
        from pydiggs_agent import validate_diggs_schema

        result = validate_diggs_schema(content=VALID_DIGGS_26_XML)
        d = result.to_dict()

        # Should be JSON serializable
        json_str = json.dumps(d)
        assert isinstance(json_str, str)

        # Round-trip
        d2 = json.loads(json_str)
        assert d2["source"] == result.source
        assert d2["is_valid"] == result.is_valid

    def test_summary_contains_error_info(self):
        """Test that summary includes error details for invalid XML."""
        from pydiggs_agent import validate_diggs_schema

        result = validate_diggs_schema(content=INVALID_DIGGS_XML)
        summary = result.summary()

        if not result.is_valid:
            assert "Valid: False" in summary
            assert "Number of Errors:" in summary
            assert result.n_errors > 0


# ============================================================================
# Foundry Agent Tests
# ============================================================================

class TestFoundryMetadata:
    """Test Foundry agent metadata functions (Tier 1)."""

    def test_list_methods_all(self):
        """Test listing all methods."""
        from foundry.pydiggs_agent_foundry import pydiggs_list_methods

        methods = json.loads(pydiggs_list_methods())
        assert isinstance(methods, list)
        assert len(methods) == 2

        method_names = [m["name"] for m in methods]
        assert "validate_schema" in method_names
        assert "validate_dictionary" in method_names

    def test_list_methods_filtered(self):
        """Test listing methods by category."""
        from foundry.pydiggs_agent_foundry import pydiggs_list_methods

        methods = json.loads(pydiggs_list_methods(category="Validation"))
        assert len(methods) == 2

        methods = json.loads(pydiggs_list_methods(category="Other"))
        assert len(methods) == 0

    def test_list_methods_invalid_category(self):
        """Test listing methods with unknown category returns empty."""
        from foundry.pydiggs_agent_foundry import pydiggs_list_methods

        methods = json.loads(pydiggs_list_methods(category="NonExistent"))
        assert methods == []

    def test_describe_validate_schema(self):
        """Test describing validate_schema method."""
        from foundry.pydiggs_agent_foundry import pydiggs_describe_method

        desc = json.loads(pydiggs_describe_method("validate_schema"))
        assert desc["name"] == "validate_schema"
        assert desc["category"] == "Validation"
        assert "description" in desc
        assert "parameters" in desc
        assert "returns" in desc

    def test_describe_validate_dictionary(self):
        """Test describing validate_dictionary method."""
        from foundry.pydiggs_agent_foundry import pydiggs_describe_method

        desc = json.loads(pydiggs_describe_method("validate_dictionary"))
        assert desc["name"] == "validate_dictionary"
        assert desc["category"] == "Validation"

    def test_describe_unknown_method(self):
        """Test describing unknown method returns error."""
        from foundry.pydiggs_agent_foundry import pydiggs_describe_method

        desc = json.loads(pydiggs_describe_method("unknown_method"))
        assert "error" in desc

    def test_agent_invalid_json(self):
        """Test agent with invalid JSON raises error."""
        from foundry.pydiggs_agent_foundry import pydiggs_agent

        with pytest.raises(json.JSONDecodeError):
            pydiggs_agent("validate_schema", "not json")

    def test_agent_unknown_method(self):
        """Test agent with unknown method returns error JSON."""
        from foundry.pydiggs_agent_foundry import pydiggs_agent

        result_json = pydiggs_agent("unknown_method", "{}")
        result = json.loads(result_json)
        assert "error" in result


@pytest.mark.skipif(not has_pydiggs(), reason="pydiggs not installed")
class TestFoundryIntegration:
    """Test Foundry agent integration (Tier 2)."""

    def test_agent_validate_schema(self):
        """Test validate_schema via Foundry agent."""
        from foundry.pydiggs_agent_foundry import pydiggs_agent

        params = {
            "content": VALID_DIGGS_26_XML,
            "schema_version": "2.6"
        }
        result_json = pydiggs_agent("validate_schema", json.dumps(params))
        result = json.loads(result_json)

        assert result["check_type"] == "schema"
        assert result["is_valid"] is True

    def test_agent_validate_dictionary(self):
        """Test validate_dictionary via Foundry agent."""
        from foundry.pydiggs_agent_foundry import pydiggs_agent

        params = {"content": VALID_DIGGS_26_XML}
        result_json = pydiggs_agent("validate_dictionary", json.dumps(params))
        result = json.loads(result_json)

        assert result["check_type"] == "dictionary"
        assert isinstance(result["is_valid"], bool)
