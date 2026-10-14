"""ground_improvement calculation tables (backlog GM-39): the time curves
are the result's own."""

from ground_improvement import (analyze_surcharge_preloading,
                                analyze_wick_drains)


def test_wick_drain_table_is_the_time_curve():
    r = analyze_wick_drains(spacing=1.5, ch=4.0, cv=2.0, Hdr=5.0, time=0.5)
    (t,) = r.get_tables()
    assert t.n_rows == len(r.time_settlement_curve)
    assert t.column("Time t") == [x for x, _ in r.time_settlement_curve]
    assert t.column("Combined degree of consolidation U") == [
        u for _, u in r.time_settlement_curve]


def test_surcharge_table_is_the_time_curve():
    r = analyze_surcharge_preloading(S_ultimate=0.4, surcharge_kPa=40.0,
                                     cv=2.0, Hdr=5.0)
    tables = r.get_tables()
    if r.time_settlement_curve:
        (t,) = tables
        assert t.column("Settlement S") == [s for _, s in
                                            r.time_settlement_curve]
    else:
        assert tables == []
