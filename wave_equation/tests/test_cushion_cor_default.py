"""GM-160: cushion_cor applies to the default cushion too."""

from funhouse_agent import dispatch as D

_PROBE = {"R_total": 1000.0, "pile_area": 0.01, "pile_length": 20.0,
          "ram_weight": 30.0, "stroke": 2.5}


def test_cushion_cor_moves_the_blow_with_no_cushion_given():
    a = D.call_agent("wave_equation", "single_blow", dict(_PROBE))
    b = D.call_agent("wave_equation", "single_blow",
                     {**_PROBE, "cushion_cor": 0.5})
    assert "error" not in a and "error" not in b
    a.pop("defaults_used", None)
    b.pop("defaults_used", None)
    assert a != b
    c = D.call_agent("wave_equation", "single_blow",
                     {**_PROBE, "cushion_cor": 0.8})
    c.pop("defaults_used", None)
    assert c == a
