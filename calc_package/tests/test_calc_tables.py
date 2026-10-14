"""Calculation tables in the calc package: CSVs beside the package and the
same tables printed in it (backlog GM-39)."""

import csv
import os

import pytest

from calc_package import (
    generate_calc_package, package_tables, render_latex,
)
from calc_package.data_model import CalcPackageData, CalcSection, TableData
from calc_package.tables import (
    INPUTS_NAME, MAX_PRINTED_ROWS, csv_bytes, csv_file_names, export_tables,
    tables_section, to_table_data,
)
from geotech_common.tables import CalcTable


def _lateral():
    from lateral_pile import LateralPileAnalysis, Pile
    from lateral_pile.py_curves import SandAPI, SoftClayMatlock
    from lateral_pile.soil import SoilLayer
    pile = Pile(length=20.0, diameter=0.6, thickness=0.0127, E=200e6)
    layers = [
        SoilLayer(top=0, bottom=5,
                  py_model=SoftClayMatlock(c=25, gamma=8, eps50=0.02)),
        SoilLayer(top=5, bottom=20,
                  py_model=SandAPI(phi=35, gamma=10, k=22000)),
    ]
    analysis = LateralPileAnalysis(pile, layers)
    return analysis.solve(Vt=150.0), analysis


def _read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.reader(fh))


class TestExportTables:
    def test_one_csv_per_table_named_by_stem(self, tmp_path):
        result, analysis = _lateral()
        paths = export_tables(result, str(tmp_path), module="lateral_pile",
                              analysis=analysis, stem="lp")
        names = [os.path.basename(p) for p in paths]
        assert names[0] == "lp_inputs.csv"
        assert names[1] == "lp_t1_pile_response.csv"
        assert all(os.path.isfile(p) for p in paths)

    def test_csv_is_utf8_with_bom_and_full_rows(self, tmp_path):
        result, analysis = _lateral()
        paths = export_tables(result, str(tmp_path), module="lateral_pile",
                              analysis=analysis, stem="lp")
        with open(paths[1], "rb") as fh:
            assert fh.read(3) == b"\xef\xbb\xbf"
        rows = _read_csv(paths[1])
        assert rows[0][0] == "Depth z (m)"
        assert len(rows) - 1 == len(result.z)          # every node

    def test_result_only_writes_the_results_own_tables(self, tmp_path):
        result, _ = _lateral()
        paths = export_tables(result, str(tmp_path))
        assert [os.path.basename(p) for p in paths] == [
            "calc_t1_pile_response.csv"]

    def test_long_paths_drop_the_name_then_cut_the_stem(self, tmp_path):
        t = CalcTable("A very long table title indeed", ["a"], rows=[[1]])
        deep = os.path.join(str(tmp_path), "d" * (200 - len(str(tmp_path))))
        names = csv_file_names([t], "s" * 80, deep)
        assert len(os.path.join(os.path.abspath(deep), names[0])) <= 220
        assert names[0].endswith("t1.csv")

    def test_csv_bytes_match_the_file(self, tmp_path):
        result, _ = _lateral()
        t = result.get_tables()[0]
        p = export_tables(tables=[t], out_dir=str(tmp_path), stem="x")[0]
        with open(p, "rb") as fh:
            assert fh.read() == csv_bytes(t)


class TestPrintedTables:
    def test_long_table_is_thinned_and_names_its_csv(self):
        result, _ = _lateral()
        t = result.get_tables()[0]
        td = to_table_data(t, "lp_t1_pile_response.csv")
        assert len(td.rows) <= MAX_PRINTED_ROWS + len(t.columns) + 1
        assert "of 101 rows printed" in td.notes
        assert "every row is in lp_t1_pile_response.csv" in td.notes
        # the row holding the maximum moment is printed
        i_max = max(range(t.n_rows),
                    key=lambda i: abs(t.rows[i][t.column_index(
                        "Bending moment M")]))
        depth = f"{t.rows[i_max][0]:.3f}"
        assert depth in [r[0] for r in td.rows]

    def test_round_off_near_zero_prints_as_zero(self):
        t = CalcTable("t", [("M", "kN-m")], rows=[[4.4e-9], [361.0]])
        assert to_table_data(t).rows[0] == ["0"]

    def test_section_skips_csv_only_tables(self):
        shown = CalcTable("Shown", ["a"], rows=[[1]])
        hidden = CalcTable("Hidden", ["a"], rows=[[1]], printed=False)
        sec = tables_section([hidden, shown], ["h.csv", "s.csv"])
        titles = [i.title for i in sec.items if isinstance(i, TableData)]
        assert titles == ["Shown"]
        assert tables_section([hidden]) is None


class TestGenerateCalcPackage:
    def test_html_prints_the_tables_and_writes_csvs(self, tmp_path):
        result, analysis = _lateral()
        out = tmp_path / "lp.html"
        report = {}
        html = generate_calc_package("lateral_pile", result, analysis,
                                     output_path=str(out),
                                     render_report=report)
        assert "Calculation Tables" in html
        assert "Pile response at each node" in html
        assert "lp_t1_pile_response.csv" in html
        written = {os.path.basename(t["path"]) for t in report["tables"]}
        assert {"lp_inputs.csv", "lp_t1_pile_response.csv",
                "lp_t2_py_curves.csv"} <= written
        for t in report["tables"]:
            assert os.path.isfile(t["path"])
        # the input echo is CSV-only: printed once, as section 1
        assert html.count("Input parameters") == 0

    def test_a_failing_table_costs_the_tables_not_the_package(
            self, tmp_path, monkeypatch):
        import calc_package
        result, analysis = _lateral()
        reg = calc_package._ensure_registered("lateral_pile")

        def boom(*a, **k):
            raise RuntimeError("bad table")
        monkeypatch.setitem(reg, "get_tables", boom)
        report = {}
        with pytest.warns(UserWarning, match="bad table"):
            html = generate_calc_package(
                "lateral_pile", result, analysis,
                output_path=str(tmp_path / "x.html"), render_report=report)
        assert "Pile Section Properties" in html
        assert "Calculation Tables" not in html
        assert "bad table" in report["tables_error"]
        # the input echo is still written
        assert [os.path.basename(t["path"]) for t in report["tables"]] == [
            "x_inputs.csv"]

    def test_tables_false_writes_and_prints_nothing(self, tmp_path):
        result, analysis = _lateral()
        html = generate_calc_package("lateral_pile", result, analysis,
                                     output_path=str(tmp_path / "x.html"),
                                     tables=False)
        assert "Calculation Tables" not in html
        assert [p.name for p in tmp_path.iterdir()] == ["x.html"]

    def test_no_output_path_prints_without_writing(self, tmp_path,
                                                   monkeypatch):
        monkeypatch.chdir(tmp_path)
        result, analysis = _lateral()
        html = generate_calc_package("lateral_pile", result, analysis)
        assert "Calculation Tables" in html
        assert list(tmp_path.iterdir()) == []

    def test_package_tables_starts_with_the_input_echo(self):
        result, analysis = _lateral()
        tables = package_tables("lateral_pile", result, analysis)
        assert tables[0].name == INPUTS_NAME and not tables[0].printed
        assert tables[0].column("Symbol")[:2] == ["L", "D"]

    def test_wide_table_gets_the_compact_block(self, tmp_path):
        result, analysis = _lateral()
        html = generate_calc_package("lateral_pile", result, analysis,
                                     output_path=str(tmp_path / "w.html"))
        assert '<div class="wide-table">' in html
        # the title sits above the table, not in a <caption>
        assert "<caption>" not in html

    def test_latex_sets_wide_tables_compact(self):
        t = CalcTable("Wide", [f"c{i}" for i in range(8)],
                      rows=[list(range(8))])
        data = CalcPackageData(sections=[CalcSection(
            "T", [to_table_data(t)])])
        tex = render_latex(data)
        assert r"\begingroup\footnotesize" in tex
        assert r"p{0.1120\textwidth}" in tex
