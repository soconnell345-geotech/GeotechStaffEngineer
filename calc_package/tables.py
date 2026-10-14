"""
Calculation tables in the calc package (backlog GM-39).

Engineers using LPILE, APILE, GROUP or SLIDE get, beside the answer, the
tables behind it. Each module that has them returns them as
:class:`geotech_common.tables.CalcTable` objects: from its result object's
``get_tables()``, or from its calc_steps' ``get_tables(result, analysis)``
when a table needs the inputs too (a p-y curve, capacity against length).

This file turns them into the two outputs the owner asked for:

* one CSV per table, saved beside the calc package (:func:`export_tables`),
  every row at full precision, plus the input echo as ``<stem>_inputs.csv``;
* the same tables printed in the calc package (:func:`tables_section`), a
  long table thinned to :data:`MAX_PRINTED_ROWS` rows with a note naming the
  CSV that holds them all.
"""

import os
from typing import List, Optional, Sequence

from geotech_common.tables import (
    CalcTable, display_rows, format_cell, tables_from,
)
from calc_package.data_model import CalcSection, InputItem, TableData

#: Rows of one table printed in the calc package; the CSV keeps every row.
MAX_PRINTED_ROWS = 50

#: Longest full path a table's CSV is given: Windows refuses a path over 260
#: characters unless long paths are enabled (the adapter's figure names keep
#: the same margin).
MAX_TABLE_PATH = 220

#: ``CalcTable.name`` of the input echo.
INPUTS_NAME = "inputs"


def input_table(items: Sequence) -> Optional[CalcTable]:
    """The calc package's input echo (its ``InputItem`` list) as a table.

    The values are the strings the package prints in its Input Parameters
    section, so the CSV and the package agree. Not printed again in the
    Tables section (``printed=False``)."""
    rows = [[i.description, i.name, i.value, i.unit]
            for i in items or () if isinstance(i, InputItem)]
    if not rows:
        return None
    return CalcTable(
        title="Input parameters",
        columns=["Parameter", "Symbol", "Value", "Unit"],
        rows=rows,
        note="The inputs of the analysis, as echoed in the calc package.",
        name=INPUTS_NAME,
        printed=False,
    )


def module_tables(reg: dict, result, analysis) -> List[CalcTable]:
    """A module's own tables: its calc_steps' ``get_tables(result,
    analysis)`` when it has one, else the result's ``get_tables()``."""
    fn = reg.get("get_tables")
    tables = fn(result, analysis) if fn else tables_from(result)
    return [t for t in (tables or []) if isinstance(t, CalcTable)]


def _short(stem: str, room: int) -> str:
    return stem[:max(8, room)]


def csv_file_names(tables: Sequence[CalcTable], stem: str,
                   out_dir: str = "") -> List[str]:
    """File names for ``tables``, in order: the input echo is
    ``<stem>_inputs.csv``; the k-th other table is ``<stem>_t<k>_<name>.csv``
    (the name is dropped, then the stem cut, to keep the full path under
    :data:`MAX_TABLE_PATH`)."""
    room = MAX_TABLE_PATH - len(os.path.abspath(out_dir or ".")) - 1
    names, k = [], 0
    for t in tables:
        if t.name == INPUTS_NAME:
            options = [f"{stem}_{INPUTS_NAME}.csv"]
            short = f"{INPUTS_NAME}.csv"
        else:
            k += 1
            options = [f"{stem}_t{k}_{t.name}.csv", f"{stem}_t{k}.csv"]
            short = f"t{k}.csv"
        for name in options:
            if len(name) <= room:
                break
        else:
            name = _short(stem, room - len(short) - 1) + "_" + short
        names.append(name)
    return names


def export_tables(result=None, out_dir: str = "", *, module: str = None,
                  analysis=None, tables: Sequence[CalcTable] = None,
                  stem: str = None) -> List[str]:
    """Write one CSV per calculation table; return the paths, in order.

    Parameters
    ----------
    result : object
        A module's result object. Ignored when ``tables`` is given.
    out_dir : str
        Folder for the CSV files (created if missing).
    module : str, optional
        The calc-package module name (``"lateral_pile"``). With it the
        input echo and the calc_steps tables are written too
        (:func:`calc_package.package_tables`); without it only
        ``result.get_tables()``.
    analysis : object, optional
        The module's analysis object, as for ``generate_calc_package``.
    tables : list of CalcTable, optional
        Write exactly these.
    stem : str, optional
        File-name stem; defaults to the module name, else ``"calc"``.

    The files are UTF-8 with a byte-order mark, so a spreadsheet opens the
    Greek letters in the headers (σ'v, φ) as written.
    """
    if tables is None:
        if module:
            from calc_package import package_tables
            tables = package_tables(module, result, analysis)
        else:
            tables = tables_from(result)
    tables = list(tables or [])
    if not tables:
        return []
    out_dir = out_dir or "."
    os.makedirs(out_dir, exist_ok=True)
    names = csv_file_names(tables, stem or module or "calc", out_dir)
    paths = []
    for t, name in zip(tables, names):
        path = os.path.abspath(os.path.join(out_dir, name))
        with open(path, "w", encoding="utf-8-sig", newline="") as fh:
            fh.write(t.to_csv())
        paths.append(path)
    return paths


def csv_bytes(table: CalcTable) -> bytes:
    """The bytes :func:`export_tables` writes for ``table``."""
    return table.to_csv().encode("utf-8-sig")


def to_table_data(table: CalcTable, csv_name: str = "",
                  max_rows: int = MAX_PRINTED_ROWS) -> TableData:
    """``table`` as the calc package prints it: formatted cells, at most
    ``max_rows`` rows (:func:`geotech_common.tables.display_rows`), and a
    note that says what the table is and which CSV holds every row."""
    idx, cut = display_rows(table, max_rows)
    # A float within 1e-9 of its column's largest magnitude prints as 0
    # (a free head's moment of 4e-9 kN-m is round-off); the CSV keeps it.
    scale = []
    for j in range(len(table.columns)):
        vals = [abs(r[j]) for r in table.rows
                if isinstance(r[j], float) and r[j] == r[j]]
        scale.append(max(vals) * 1e-9 if vals else 0.0)

    def cell(i, j, col):
        v = table.rows[i][j]
        if isinstance(v, float) and 0 < abs(v) <= scale[j]:
            v = 0.0
        return format_cell(v, col.fmt)

    rows = [[cell(i, j, c) for j, c in enumerate(table.columns)]
            for i in idx]
    notes = [table.note.rstrip(". ")] if table.note else []
    if cut:
        notes.append(cut + (f"; every row is in {csv_name}" if csv_name
                            else ""))
    elif csv_name:
        notes.append(f"CSV: {csv_name}")
    return TableData(title=table.title, headers=table.headers, rows=rows,
                     notes=". ".join(notes) + ("." if notes else ""))


def tables_section(tables: Sequence[CalcTable],
                   csv_names: Sequence[str] = None,
                   max_rows: int = MAX_PRINTED_ROWS) -> Optional[CalcSection]:
    """The calc package's "Calculation Tables" section, or ``None`` when no
    table is to be printed. ``csv_names`` (parallel to ``tables``) are the
    CSV files saved beside the package."""
    names = list(csv_names or [])
    names += [""] * (len(tables) - len(names))
    shown = [(t, n) for t, n in zip(tables, names) if t.printed]
    if not shown:
        return None
    intro = ("The tables below hold the analysis's own computed values, in "
             "the units named in each column.")
    if any(n for t, n in shown):
        intro += (" Each is also saved as a CSV file beside this package, "
                  "with every row")
        echo = [n for t, n in zip(tables, names)
                if n and t.name == INPUTS_NAME]
        intro += (f"; the input echo is {echo[0]}." if echo else ".")
    items = [intro] + [to_table_data(t, n, max_rows) for t, n in shown]
    return CalcSection(title="Calculation Tables", items=items)


__all__ = ["MAX_PRINTED_ROWS", "INPUTS_NAME", "input_table", "module_tables",
           "csv_file_names", "export_tables", "csv_bytes", "to_table_data",
           "tables_section"]
