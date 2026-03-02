"""Tests for pdf_import.vision — LLM vision-based extraction.

All tests use mock image_fn — no API key needed.
"""

import json
import pytest

from pdf_import.vision import (
    extract_geometry_vision,
    _parse_vision_response,
    _extract_json,
)
from pdf_import.results import PdfParseResult


# ---------------------------------------------------------------------------
# Mock image functions
# ---------------------------------------------------------------------------

def _mock_vision_good(image_bytes, prompt):
    """Mock that returns valid JSON geometry."""
    return json.dumps({
        "surface_points": [[0, 10], [10, 10], [20, 5], [30, 5]],
        "boundary_profiles": {
            "Clay": [[0, 5], [10, 5], [20, 3], [30, 3]],
        },
        "gwt_points": [[0, 8], [10, 8], [20, 4], [30, 4]],
        "scale_info": "1 grid square = 5m",
    })


def _mock_vision_surface_only(image_bytes, prompt):
    """Mock that returns only surface points."""
    return json.dumps({
        "surface_points": [[0, 15], [20, 15], [40, 10]],
        "boundary_profiles": {},
        "gwt_points": None,
    })


def _mock_vision_markdown(image_bytes, prompt):
    """Mock that returns JSON wrapped in markdown code block."""
    return (
        "I can see a cross-section. Here is the geometry:\n\n"
        "```json\n"
        '{"surface_points": [[0, 12], [30, 8]], '
        '"boundary_profiles": {}, "gwt_points": null}\n'
        "```\n\n"
        "The drawing shows a simple slope."
    )


def _mock_vision_malformed(image_bytes, prompt):
    """Mock that returns invalid JSON."""
    return "I can see a slope but {invalid json here..."


def _mock_vision_empty(image_bytes, prompt):
    """Mock that returns no JSON at all."""
    return "I cannot identify any geometric features in this image."


def _mock_vision_partial(image_bytes, prompt):
    """Mock that returns JSON with some invalid coordinates."""
    return json.dumps({
        "surface_points": [[0, 10], "not a point", [20, 5]],
        "boundary_profiles": {
            "Sand": [[0, 7], [20, 4]],
            "BadLayer": "not a list",
        },
    })


# ---------------------------------------------------------------------------
# Tests: _extract_json
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_direct_json(self):
        result = _extract_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_markdown_code_block(self):
        text = 'Some text\n```json\n{"key": 1}\n```\nMore text'
        result = _extract_json(text)
        assert result == {"key": 1}

    def test_embedded_json(self):
        text = 'Here is the result: {"points": [1, 2, 3]} as you can see'
        result = _extract_json(text)
        assert result == {"points": [1, 2, 3]}

    def test_no_json(self):
        result = _extract_json("No JSON here at all")
        assert result is None

    def test_invalid_json(self):
        result = _extract_json("{broken: json}")
        assert result is None


# ---------------------------------------------------------------------------
# Tests: _parse_vision_response
# ---------------------------------------------------------------------------

class TestParseVisionResponse:
    def test_good_response(self):
        text = json.dumps({
            "surface_points": [[0, 10], [20, 5]],
            "boundary_profiles": {"Clay": [[0, 5], [20, 3]]},
            "gwt_points": [[0, 8], [20, 4]],
        })
        result = _parse_vision_response(text)
        assert len(result.surface_points) == 2
        assert "Clay" in result.boundary_profiles
        assert result.gwt_points is not None
        assert result.confidence >= 0.7

    def test_no_json_low_confidence(self):
        result = _parse_vision_response("No geometry found")
        assert result.confidence == 0.0
        assert len(result.surface_points) == 0

    def test_no_surface_low_confidence(self):
        text = json.dumps({
            "surface_points": [],
            "boundary_profiles": {},
        })
        result = _parse_vision_response(text)
        assert result.confidence <= 0.3

    def test_scale_applied(self):
        text = json.dumps({
            "surface_points": [[0, 100], [200, 50]],
        })
        result = _parse_vision_response(text, scale=0.01)
        assert abs(result.surface_points[0][1] - 1.0) < 0.01
        assert abs(result.surface_points[1][0] - 2.0) < 0.01

    def test_scale_info_warning(self):
        text = json.dumps({
            "surface_points": [[0, 10]],
            "scale_info": "1 inch = 10 feet",
        })
        result = _parse_vision_response(text)
        assert any("Scale info" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Tests: extract_geometry_vision
# ---------------------------------------------------------------------------

class TestExtractGeometryVision:
    def test_good_extraction(self):
        result = extract_geometry_vision(
            image_fn=_mock_vision_good,
            content=b"fake png data",
        )
        assert isinstance(result, PdfParseResult)
        assert result.extraction_method == "vision"
        assert len(result.surface_points) == 4
        assert "Clay" in result.boundary_profiles
        assert result.gwt_points is not None

    def test_surface_only(self):
        result = extract_geometry_vision(
            image_fn=_mock_vision_surface_only,
            content=b"fake image",
        )
        assert len(result.surface_points) == 3
        assert len(result.boundary_profiles) == 0
        assert result.gwt_points is None

    def test_markdown_wrapped_json(self):
        result = extract_geometry_vision(
            image_fn=_mock_vision_markdown,
            content=b"fake image",
        )
        assert len(result.surface_points) == 2

    def test_malformed_json(self):
        result = extract_geometry_vision(
            image_fn=_mock_vision_malformed,
            content=b"fake image",
        )
        assert result.confidence == 0.0
        assert any("Could not parse" in w for w in result.warnings)

    def test_empty_response(self):
        result = extract_geometry_vision(
            image_fn=_mock_vision_empty,
            content=b"fake image",
        )
        assert result.confidence == 0.0

    def test_partial_invalid_coords(self):
        result = extract_geometry_vision(
            image_fn=_mock_vision_partial,
            content=b"fake image",
        )
        # Should skip invalid points, keep valid ones
        assert len(result.surface_points) == 2  # [0,10] and [20,5]
        assert "Sand" in result.boundary_profiles
        # "BadLayer" should be skipped (not a list)
        assert "BadLayer" not in result.boundary_profiles

    def test_custom_prompt(self):
        calls = []

        def _capture_fn(image_bytes, prompt):
            calls.append(prompt)
            return json.dumps({"surface_points": [[0, 5]]})

        extract_geometry_vision(
            image_fn=_capture_fn,
            content=b"fake image",
            custom_prompt="Extract the slope geometry",
        )
        assert calls[0] == "Extract the slope geometry"

    def test_scale_factor(self):
        result = extract_geometry_vision(
            image_fn=_mock_vision_good,
            content=b"fake image",
            scale=0.1,
        )
        # Original points: [0,10], [10,10], etc.
        # Scaled: [0,1.0], [1.0,1.0], etc.
        assert abs(result.surface_points[1][0] - 1.0) < 0.01
        assert result.scale_factor == 0.1

    def test_no_input_raises(self):
        with pytest.raises(ValueError, match="Provide either"):
            extract_geometry_vision(image_fn=_mock_vision_good)
