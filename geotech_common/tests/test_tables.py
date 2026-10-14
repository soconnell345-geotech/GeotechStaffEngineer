"""geotech_common.tables: the calculation-table type (backlog GM-39)."""

import csv
import io
import math

import numpy as np
import pytest

from geotech_common.tables import (
    CalcTable, TableColumn, csv_cell, display_rows, format_cell, tables_from,
)


def _table(n=5):
    return CalcTable(
        title="Pile response at each node",
        columns=[("Depth z", "m", ".2f"), ("Bending moment M", "kN-m"),
                 "Note"],
        rows=[[i * 0.5, float(i * i), f"n{i}"] for i in range(n)],
        note="test table",
    )


class TestCalcTable:
    def test_columns_from_tuples_and_names(self):
        t = _table()
        assert [c.header for c in t.columns] == [
            "Depth z (m)", "Bending moment M (kN-m)", "Note"]
        assert t.columns[0].fmt == ".2f"
        assert isinstance(t.columns[2], TableColumn)

    def test_name_defaults_to_a_slug_of_the_title(self):
        assert _table().name == "pile_response_at_each_node"
        assert CalcTable("x", ["a"], name="My Table!").name == "my_table"

    def test_row_length_is_checked(self):
        with pytest.raises(ValueError, match="row 0 has 2 values for 3"):
            CalcTable("t", ["a", "b", "c"], rows=[[1, 2]])

    def test_numpy_values_become_plain_python(self):
        t = CalcTable("t", ["a", "b"],
                      rows=[[np.float64(1.5), np.int64(2)]])
        assert type(t.rows[0][0]) is float and type(t.rows[0][1]) is int

    def test_column_by_name_or_header(self):
        t = _table()
        assert t.column("Bending moment M") == [0.0, 1.0, 4.0, 9.0, 16.0]
        assert t.column("Depth z (m)")[-1] == 2.0
        with pytest.raises(KeyError):
            t.column("Shear")

    def test_csv_round_trips(self):
        t = _table()
        rows = list(csv.reader(io.StringIO(t.to_csv())))
        assert rows[0] == t.headers
        assert len(rows) == t.n_rows + 1
        assert [float(r[1]) for r in rows[1:]] == t.column("Bending moment M")

    def test_to_dict(self):
        d = _table(2).to_dict()
        assert d["columns"][0] == {"name": "Depth z", "unit": "m"}
        assert d["rows"][1] == [0.5, 1.0, "n1"]

    def test_tables_from(self):
        class R:
            def get_tables(self):
                return [_table(), "not a table"]
        assert len(tables_from(R())) == 1
        assert tables_from(object()) == []


class TestCells:
    @pytest.mark.parametrize("v,out", [
        (None, ""), (True, "yes"), (0.1 + 0.2, "0.3"), (float("nan"), ""),
        (12, "12"), ("a", "a"), (1.23456789012345e-7, "1.23456789e-07"),
    ])
    def test_csv_cell(self, v, out):
        assert csv_cell(v) == out

    @pytest.mark.parametrize("v,fmt,out", [
        (None, "", "—"), (0.0, "", "0"), (361.04, "", "361"),
        (361.4, "", "361.4"), (3.7, "", "3.7"),
        (0.0052341, "", "0.005234"), (12345.6, "", "12346"),
        (4.5e-9, "", "4.500e-09"), (2.5e6, "", "2.500e+06"),
        (2.5, ".2f", "2.50"), (7, "", "7"), (float("inf"), "", "∞"),
    ])
    def test_format_cell(self, v, fmt, out):
        assert format_cell(v, fmt) == out


class TestDisplayRows:
    def test_short_table_prints_every_row(self):
        idx, note = display_rows(_table(10), max_rows=50)
        assert idx == list(range(10)) and note == ""

    def test_long_table_keeps_last_row_and_each_column_maximum(self):
        rows = [[float(i), math.sin(i / 10.0)] for i in range(101)]
        rows[37][1] = 99.0                      # off the regular stride
        t = CalcTable("t", [("z", "m"), ("M", "kN-m")], rows=rows)
        idx, note = display_rows(t, max_rows=20)
        assert 0 in idx and 100 in idx and 37 in idx
        assert len(idx) < 30
        assert note.startswith(f"{len(idx)} of 101 rows printed")
