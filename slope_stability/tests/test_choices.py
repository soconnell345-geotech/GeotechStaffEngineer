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
    fam = c["surface families scored"]
    # c' = 10 everywhere: no planar surface, and the entry says so (GM-125)
    assert fam["chosen"] == ("circular, toe circle, circle passing above "
                             "the toe")
    assert fam["basis"].endswith(
        "; infinite slope: none scored, no face of c' = 0 soil")
    assert r.families_scored == ["circular", "toe_circle",
                                 "circle_above_toe"]
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
        c["surface families scored"]["alternatives"]
    assert c["search ranges"] == {**c["search ranges"], "chosen": "given",
                                  "how": "given"}


# ---------------------------------------------------------------------------
# GM-125: the families listed are the ones scored; pointers only at fields
# the result has
# ---------------------------------------------------------------------------

def _w10():
    """The live-run slope (wave w10): weak residual clay over sandstone."""
    return SlopeGeometry(
        surface_points=[(0, 0), (20, 0), (30, 10), (60, 10)],
        soil_layers=[
            SlopeSoilLayer("Residual clay", 10, 6.5, 18.5, phi=22,
                           c_prime=3),
            SlopeSoilLayer("Sandstone", 6.5, -40, 23, phi=40, c_prime=200)])


def test_a_cohesionless_face_lists_the_infinite_slope_as_scored():
    g = SlopeGeometry(
        surface_points=[(0, 0), (30, 24), (75, 24)],
        soil_layers=[SlopeSoilLayer(name="fill", top_elevation=24,
                                    bottom_elevation=0, gamma=20, phi=37,
                                    c_prime=0)])
    r = search_critical_surface(g, surface_type="entry_exit", nx=4, ny=4)
    fam = _by_what(r)["surface families scored"]
    assert "infinite slope" in fam["chosen"]
    assert "the infinite slope where c' = 0 soil forms the face" in (
        fam["basis"])
    assert "infinite_slope" in r.families_scored


def test_a_search_with_no_surface_says_none_and_points_at_nothing():
    """A short noncircular search with automatic windows on the live-run
    slope finds no admissible surface: the entry says none was scored and
    why, and the ranges point at no location."""
    r = search_critical_surface(_w10(), surface_type="noncircular",
                                n_trials=100, seed=1, n_slices=30)
    assert r.critical is None
    c = _by_what(r)
    fam = c["surface families scored"]
    assert fam["chosen"] == "none"
    assert fam["basis"].startswith(
        "no surface held; noncircular: none scored, none of the surfaces "
        "tried held")
    assert r.families_scored == []
    ranges = c["search ranges"]["basis"]
    assert "governing_location" not in ranges
    assert ranges.endswith("no surface is reported")
    assert "governing_location" not in r.to_dict()


def test_a_reported_surface_is_pointed_at():
    r = search_critical_surface(_geom(), surface_type="entry_exit",
                                nx=4, ny=4)
    basis = _by_what(r)["search ranges"]["basis"]
    assert r.to_dict()["governing_location"]
    assert "(governing_location)" in basis
    assert "refused_lowest" not in basis


def test_families_cut_off_by_the_windows_given_say_so():
    r = search_critical_surface(_geom(), x_entry_range=(0, 4),
                                x_exit_range=(14, 20))
    fam = _by_what(r)["surface families scored"]
    assert fam["chosen"] == "circular, toe circle"
    assert ("circle passing above the toe: none scored, none within the "
            "windows and ranges given") in fam["basis"]
    assert r.families_not_scored == {
        "circle_above_toe": "none within the windows and ranges given",
        "infinite_slope": "no face of c' = 0 soil"}
