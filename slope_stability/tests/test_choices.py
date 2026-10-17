"""What a critical-surface search says it chose (backlog GM-115): the
search, the method it scored with, the surface families it did and did not
try, and how its ranges were set (pointing at ``search_ranges``)."""

from slope_stability import SlopeGeometry, SlopeSoilLayer
from slope_stability.analysis import search_critical_surface


def _geom():
    return SlopeGeometry(
        surface_points=[(0, 5), (5, 5), (15, 15), (20, 15)],
        soil_layers=[SlopeSoilLayer(name="m", top_elevation=15,
                                    bottom_elevation=0, gamma=18, phi=30,
                                    c_prime=10)])


def _by_what(result):
    return {c["what"]: c for c in result.choices}


def test_circular_search_names_what_it_did_not_try():
    r = search_critical_surface(_geom(), surface_type="entry_exit",
                                nx=4, ny=4)
    c = _by_what(r)
    assert c["search"]["chosen"] == \
        "circular, entry and exit windows ('entry_exit')"
    assert c["search"]["how"] == "given"
    assert c["FOS method of the search"] == {
        "what": "FOS method of the search", "chosen": "bishop",
        "how": "default"}
    fam = c["surface families tried"]
    assert fam["chosen"] == ("circular, toe circle, circle passing above "
                             "the toe, infinite slope")
    assert fam["alternatives"] == [
        "not tried: noncircular, by a noncircular search "
        "(surface_type 'noncircular_de')"]
    ranges = c["search ranges"]
    assert ranges["how"] == "module"
    assert r.search_ranges["automatic"] == ["x_entry_range", "x_exit_range"]
    assert ranges["chosen"] == "entry window, exit window"
    assert "(search_ranges)" in ranges["basis"]
    d = r.to_dict()
    assert d["choices"] == r.choices
    assert "Choices:" in r.summary()


def test_noncircular_bishop_is_scored_with_spencer():
    r = search_critical_surface(_geom(), surface_type="noncircular",
                                n_trials=20, seed=1,
                                x_entry_range=(0, 6), x_exit_range=(14, 20))
    c = _by_what(r)
    method = c["FOS method of the search"]
    assert method["chosen"] == "spencer" and method["how"] == "module"
    assert "not tried: toe circle, by a circular search" in \
        c["surface families tried"]["alternatives"]
    assert c["search ranges"] == {**c["search ranges"], "chosen": "given",
                                  "how": "given"}
