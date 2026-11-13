"""GM-318: a pile toe below the soil profile is refused unless
``extend_last_layer`` carries the last layer on, and the result then says
so; ``capacity_vs_depth`` names the trial lengths it left out. Made-up
values."""

import pytest

from axial_pile import (AxialPileAnalysis, AxialSoilLayer, AxialSoilProfile,
                        make_pipe_pile)
from funhouse_agent.adapters.axial_pile import METHOD_REGISTRY


def _soil(thick=10.0):
    return AxialSoilProfile(layers=[
        AxialSoilLayer(4.0, "cohesionless", 18.0, friction_angle=30),
        AxialSoilLayer(thick - 4.0, "cohesionless", 19.0, friction_angle=34),
    ])


def _pile():
    return make_pipe_pile(0.3239, 0.00953, closed_end=True)


def test_toe_below_profile_refused_and_names_the_depth():
    with pytest.raises(ValueError) as e:
        AxialPileAnalysis(pile=_pile(), soil=_soil(), pile_length=12.0)
    msg = str(e.value)
    assert "ends at 10 m" in msg and "toe is at 12 m" in msg
    assert "nobody explored" in msg and "extend_last_layer" in msg


def test_head_depth_counts_toward_the_toe():
    with pytest.raises(ValueError, match="toe is at 11 m"):
        AxialPileAnalysis(pile=_pile(), soil=_soil(), pile_length=9.0,
                          head_depth=2.0)


def test_profile_depth_given_deeper_than_the_profile_is_refused():
    """Review C1: a library caller's profile_depth_given brought back the
    old silent basis (12 m pile, 10 m profile, run with no warning)."""
    with pytest.raises(ValueError, match="profile_depth_given = 20 m"):
        AxialPileAnalysis(pile=_pile(), soil=_soil(), pile_length=12.0,
                          profile_depth_given=20.0)


def test_toe_at_profile_bottom_runs_without_a_note():
    r = AxialPileAnalysis(pile=_pile(), soil=_soil(),
                          pile_length=10.0).compute()
    assert not any("nobody explored" in w for w in r.warnings)
    assert not any(c["what"] == "soil below the profile" for c in r.choices)


def test_extension_equals_a_profile_given_to_the_toe():
    ext = AxialPileAnalysis(pile=_pile(), soil=_soil(10.0), pile_length=12.0,
                            extend_last_layer=True).compute()
    full = AxialPileAnalysis(pile=_pile(), soil=_soil(12.0),
                             pile_length=12.0).compute()
    assert ext.Q_skin == pytest.approx(full.Q_skin)
    assert ext.Q_tip == pytest.approx(full.Q_tip)
    assert "last layer was taken to go on" in ext.warnings[0]
    c = [c for c in ext.choices if c["what"] == "soil below the profile"]
    assert len(c) == 1 and "from 10 m to the toe at 12 m" in c[0]["chosen"]


def test_curve_names_the_lengths_it_left_out():
    a = AxialPileAnalysis(pile=_pile(), soil=_soil(), pile_length=10.0)
    skipped = []
    curve = a.capacity_vs_depth(depth_min=6.0, depth_max=14.0, n_points=5,
                                skipped=skipped)
    assert [row["depth_m"] for row in curve] == [6.0, 8.0, 10.0]
    assert [m["depth_m"] for m in skipped] == [12.0, 14.0]
    # review C7: a short reason, not the whole refusal again for each
    assert [m["reason"] for m in skipped] == [
        "below the profile, which ends at 10 m"] * 2
    with pytest.warns(UserWarning, match="refused 2 of the 5"):
        a.capacity_vs_depth(depth_min=6.0, depth_max=14.0, n_points=5)


def test_curve_skipped_reason_under_a_buried_head_names_the_toe():
    a = AxialPileAnalysis(pile=_pile(), soil=_soil(), pile_length=8.0,
                          head_depth=1.0)
    skipped = []
    a.capacity_vs_depth(depth_min=7.0, depth_max=10.0, n_points=2,
                        skipped=skipped)
    assert skipped == [{"depth_m": 10.0, "reason": "toe at 11 m, below the "
                        "profile, which ends at 10 m"}]


def test_curve_with_every_length_refused_is_refused():
    a = AxialPileAnalysis(pile=_pile(), soil=_soil(), pile_length=10.0)
    with pytest.raises(ValueError, match="refused 3 of the 3"):
        a.capacity_vs_depth(depth_min=11.0, depth_max=13.0, n_points=3)


def test_extended_curve_marks_only_the_lengths_past_the_profile():
    a = AxialPileAnalysis(pile=_pile(), soil=_soil(), pile_length=12.0,
                          extend_last_layer=True)
    curve = a.capacity_vs_depth(depth_min=8.0, depth_max=12.0, n_points=3)
    noted = [any("nobody explored" in w for w in row.get("warnings", []))
             for row in curve]
    assert noted == [False, False, True]


def test_package_table_names_refused_lengths(monkeypatch):
    """Review C2: the package's capacity against pile length table names a
    refused length in its note, as the drilled shaft's does; with every
    length refused the table is left out with a warning, not an error."""
    from axial_pile.calc_steps import get_tables
    a = AxialPileAnalysis(pile=_pile(), soil=_soil(), pile_length=10.0)
    r = a.compute()
    real = AxialPileAnalysis.capacity_vs_depth

    def one_refused(self, *args, skipped=None, **kw):
        rows = real(self, *args, skipped=skipped, **kw)
        skipped.append({"depth_m": 5.0, "reason": "a made-up refusal"})
        return rows[:-1]

    monkeypatch.setattr(AxialPileAnalysis, "capacity_vs_depth", one_refused)
    t = next(t for t in get_tables(r, a) if t.name == "capacity_vs_length")
    assert "refused 1 of the 20" in t.note
    assert "5 m (a made-up refusal)" in t.note

    def all_refused(self, *args, **kw):
        raise ValueError("The analysis refused 20 of the 20 trial pile "
                         "lengths")

    monkeypatch.setattr(AxialPileAnalysis, "capacity_vs_depth", all_refused)
    with pytest.warns(UserWarning, match="table is left out: The analysis "
                                         "refused 20 of the 20"):
        tables = get_tables(r, a)
    assert not any(t.name == "capacity_vs_length" for t in tables)


_PARAMS = {
    "pile_type": "pipe_closed", "diameter": 0.3239, "wall_thickness": 0.00953,
    "layers": [
        {"thickness": 4.0, "soil_type": "cohesionless", "unit_weight": 18.0,
         "friction_angle": 30},
        {"thickness": 6.0, "soil_type": "cohesionless", "unit_weight": 19.0,
         "friction_angle": 34}],
}


def test_tool_refuses_and_extends():
    run = METHOD_REGISTRY["axial_pile_capacity"]
    with pytest.raises(ValueError, match="extend_last_layer"):
        run({**_PARAMS, "pile_length": 12.0})
    out = run({**_PARAMS, "pile_length": 12.0, "extend_last_layer": True})
    assert "nobody explored" in out["warnings"][0]
    assert out["choices"][0]["what"] == "soil below the profile"


def test_tool_curve_reports_skipped_depths():
    out = METHOD_REGISTRY["capacity_vs_depth"](
        {**_PARAMS, "depth_min": 6.0, "depth_max": 14.0, "n_points": 5})
    assert [m["depth_m"] for m in out["skipped_depths"]] == [12.0, 14.0]
    assert "refused 2 of the 5" in out["warning"]
    assert len(out["capacity_vs_depth"]) == 3 and out["choices"]


def test_tool_curve_default_length_under_a_buried_head():
    out = METHOD_REGISTRY["capacity_vs_depth"](
        {**_PARAMS, "head_depth": 1.5, "n_points": 4})
    assert "skipped_depths" not in out
    assert out["capacity_vs_depth"][-1]["depth_m"] == 8.5
