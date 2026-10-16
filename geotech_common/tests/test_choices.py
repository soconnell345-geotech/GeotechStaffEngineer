"""The ``choices`` a result reports (backlog GM-115)."""

import numpy as np
import pytest

from geotech_common.choices import (
    DEFAULT, GIVEN, MODULE, choice, choice_rows, choices_from, choices_lines,
    how_given, plain,
)


def test_entry_keeps_what_chosen_how_and_drops_empty_parts():
    c = choice("stress increase with depth", "2:1", DEFAULT)
    assert c == {"what": "stress increase with depth", "chosen": "2:1",
                 "how": "default"}
    c = choice("search", "circular", GIVEN, "why", ["circular", "noncircular"])
    assert c["basis"] == "why"
    # the option chosen is not listed as its own alternative
    assert c["alternatives"] == ["noncircular"]


def test_how_must_be_one_of_three():
    with pytest.raises(ValueError, match="how must be one of"):
        choice("x", 1, "assumed")


def test_numpy_values_become_plain():
    c = choice("kh", np.float64(0.15), GIVEN)
    assert type(c["chosen"]) is float


def test_how_given_compares_with_the_default():
    assert how_given(2.8, 2.8) == DEFAULT
    assert how_given("beta", "auto") == GIVEN
    assert how_given(np.array([1, 2]), 1) == GIVEN   # no truth value


def test_printed_text_leaves_out_the_keys_named_for_the_agent():
    assert plain("Boussinesq (stress_method 'boussinesq')") == "Boussinesq"
    assert plain("rd and MSF are listed for each layer (layer_results)") \
        == "rd and MSF are listed for each layer"
    assert plain("windows given (x_entry_range, x_exit_range)") == \
        "windows given"
    assert plain("by a search with it (infinite_slope=True)") == \
        "by a search with it"
    assert plain("a drainage path given (Hdr)") == "a drainage path given"
    assert plain("circular, centre grid ('circular')") == \
        "circular, centre grid"
    # brackets an engineer reads stay
    for kept in ("GEC-12 Eq. 7-11 (Nc = 9)", "1 year (assumed)",
                 "Iw = 1.12 (flexible footing, centre)",
                 "Ka + 0.75 kh (Seed & Whitman 1970)"):
        assert plain(kept) == kept
    rows = choice_rows([choice("x (a_b)", "y (stress_method 'w')", GIVEN,
                               "z (layer_results)", ["w (method 'beta')"])])
    assert rows == [["x", "y", "given", "z", "w"]]


def test_lines_rows_and_reading_back():
    cs = [choice("side resistance method", "Nordlund 0-10 m", MODULE,
                 "auto by soil type", ["beta"])]
    lines = choices_lines(cs)
    assert lines[1].strip() == "Choices:"
    assert "Nordlund 0-10 m (module's choice)" in lines[2]
    assert choices_lines([]) == []
    assert choice_rows(cs) == [["side resistance method", "Nordlund 0-10 m",
                                "module's choice", "auto by soil type",
                                "beta"]]

    class R:
        choices = cs
    assert choices_from(R()) == cs
    assert choices_from({"choices": cs}) == cs
    assert choices_from(object()) == []
