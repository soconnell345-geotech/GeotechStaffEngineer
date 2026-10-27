"""RI-137: superscripts and the micro sign are read inside a unit token."""
from __future__ import annotations

import pytest

from report_ingest.calc_reader import _PAIR_RE, _VALUE_SPAN_RE
from report_ingest.model import _canonical_unit, to_si


@pytest.mark.parametrize("line,unit", [
    ("A = 2.5 m²", "m²"),
    ("V = 4.0 m³", "m³"),
    ("g = 18 kN/m³", "kN/m³"),
    ("k = 3 µm/s", "µm/s"),
    ("k = 3 μm/s", "μm/s"),
    ("c = 5 µg/kg", "µg/kg"),
    ("A = 2.5 m2", "m2"),
    ("A = 2.5 m^2", "m^2"),
    ("g = 18 kN/m^3", "kN/m^3"),
])
def test_pair_line_unit_read(line, unit):
    m = _PAIR_RE.match(line)
    assert m is not None
    assert m.group("unit") == unit


def test_value_span_with_superscript():
    assert _VALUE_SPAN_RE.match("2.5 m²").group("unit") == "m²"


@pytest.mark.parametrize("line", ["A = 12²", "A = 12 ²",
                                  "A = 12³"])
def test_superscript_after_number_alone_is_not_a_unit(line):
    # a footnote marker, not a unit: the line is not read as a value
    assert _PAIR_RE.match(line) is None


def test_superscript_spellings_convert_alike():
    assert (_canonical_unit("m²") == _canonical_unit("m^2")
            == _canonical_unit("m2"))
    assert _canonical_unit("kN/m³") == _canonical_unit("kN/m^3")
    assert _canonical_unit("m²") is not None
    assert _canonical_unit("kN/m³") is not None
    assert to_si(2.0, "m²") == to_si(2.0, "m^2") == to_si(2.0, "m2")
    assert to_si(2.0, "kN/m³") == to_si(2.0, "kN/m^3")


def test_both_mu_characters_convert_alike():
    assert (_canonical_unit("µg/kg") == _canonical_unit("μg/kg")
            == "ug/kg")
