"""
Calculation tables: the numbers behind a result (backlog GM-39).

Engineers using LPILE, APILE, GROUP or SLIDE get, beside the answer, the full
tables behind it: deflection, moment, shear and soil reaction at every node,
side and toe resistance per layer, the forces on every slice. A module's
result object returns those tables from ``get_tables()`` as
:class:`CalcTable` objects; ``calc_package`` writes one CSV per table beside
the calc package and prints the same tables in it.

Rules every table follows:

* the values are the module's own computed values, taken from the result as
  it holds them (never recomputed another way for display);
* units are SI as the module uses them, named in the column (``m``, ``kN``,
  ``kN-m``, ``kPa``);
* columns name quantities as the references and the Ensoft manuals do
  (deflection y, bending moment M, shear V, soil reaction p, unit side
  resistance f_s);
* ``note`` says in one line what the table is and which method produced it.

Pure data; no third-party import.
"""

import csv
import io
import math
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence


@dataclass
class TableColumn:
    """One column of a :class:`CalcTable`.

    Attributes
    ----------
    name : str
        The quantity, named as the references name it ("Deflection y").
    unit : str
        SI unit as the module uses it ("m"); empty for a label or a count.
    fmt : str
        Optional display format for the printed table (``".2f"``). The CSV
        always carries the full value.
    """
    name: str
    unit: str = ""
    fmt: str = ""

    @property
    def header(self) -> str:
        """``"Deflection y (m)"``, or the bare name when there is no unit."""
        return f"{self.name} ({self.unit})" if self.unit else self.name


def _as_column(spec) -> TableColumn:
    if isinstance(spec, TableColumn):
        return spec
    if isinstance(spec, str):
        return TableColumn(spec)
    if isinstance(spec, (tuple, list)) and 1 <= len(spec) <= 3:
        return TableColumn(*spec)
    raise TypeError(f"a table column is a TableColumn, a name, or "
                    f"(name, unit[, fmt]); got {spec!r}")


def _plain(value):
    """A cell as a plain Python value: numpy scalars become int/float/bool
    (np.float64 subclasses float, so the type is checked exactly)."""
    if value is None or type(value) in (str, bool, int, float):
        return value
    item = getattr(value, "item", None)          # numpy scalar
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    return value


def _slug(text: str, n: int = 32) -> str:
    t = re.sub(r"[^A-Za-z0-9]+", "_", str(text or "")).strip("_").lower()
    return (t[:n].strip("_")) or "table"


@dataclass
class CalcTable:
    """A named table of a module's computed values.

    Attributes
    ----------
    title : str
        "Pile response at each node".
    columns : list of TableColumn
        Column definitions; a ``(name, unit)`` tuple or a bare name is
        accepted and converted.
    rows : list of list
        One list per row, one value per column (numbers, strings or None).
    note : str
        One line: what the table is and its source/method.
    name : str
        Short file-name stem (``"pile_response"``); defaults to a slug of
        the title.
    printed : bool
        False: the table is written to CSV only, because the calc package
        already prints it elsewhere (the input echo, a layer table in its
        own section).
    """
    title: str
    columns: List[Any]
    rows: List[list] = field(default_factory=list)
    note: str = ""
    name: str = ""
    printed: bool = True

    def __post_init__(self):
        self.columns = [_as_column(c) for c in self.columns]
        n = len(self.columns)
        clean = []
        for i, row in enumerate(self.rows):
            row = list(row)
            if len(row) != n:
                raise ValueError(
                    f"table '{self.title}': row {i} has {len(row)} values "
                    f"for {n} columns")
            clean.append([_plain(v) for v in row])
        self.rows = clean
        if not self.name:
            self.name = _slug(self.title)
        else:
            self.name = _slug(self.name)

    # -- reading -----------------------------------------------------------
    @property
    def headers(self) -> List[str]:
        return [c.header for c in self.columns]

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    def column_index(self, key: str) -> int:
        """Index of a column given by its name or its full header."""
        for i, c in enumerate(self.columns):
            if key in (c.name, c.header):
                return i
        raise KeyError(f"table '{self.title}' has no column '{key}' "
                       f"(columns: {self.headers})")

    def column(self, key: str) -> list:
        """The values of one column, top to bottom."""
        i = self.column_index(key)
        return [row[i] for row in self.rows]

    # -- writing -----------------------------------------------------------
    def to_csv(self) -> str:
        """The table as CSV text: one header row, then the rows. Numbers
        keep ten significant digits; None is an empty cell."""
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(self.headers)
        for row in self.rows:
            w.writerow([csv_cell(v) for v in row])
        return buf.getvalue()

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "name": self.name,
            "columns": [{"name": c.name, "unit": c.unit} for c in self.columns],
            "rows": [list(r) for r in self.rows],
            "note": self.note,
        }


def csv_cell(value) -> str:
    """One cell as CSV text."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return format(value, ".10g")
    return str(value)


def format_cell(value, fmt: str = "") -> str:
    """One cell as it is printed in the calc package: ``fmt`` when given,
    else up to four significant figures without trailing zeros (scientific
    only below 1e-4 or from 1e6)."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return format(value, fmt) if fmt else str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "—"
        if math.isinf(value):
            return "∞" if value > 0 else "-∞"
        if fmt:
            return format(value, fmt)
        if value == 0:
            return "0"
        a = abs(value)
        if a >= 1e6 or a < 1e-4:
            return f"{value:.3e}"
        decimals = max(0, 3 - int(math.floor(math.log10(a))))
        text = f"{value:.{decimals}f}"
        # no trailing zeros the value does not have (3.7, not 3.700)
        return text.rstrip("0").rstrip(".") if "." in text else text
    return str(value)


def display_rows(table: CalcTable, max_rows: int = 50):
    """The rows of ``table`` to PRINT, and a note when some were left out.

    A table longer than ``max_rows`` keeps every k-th row, the last row and,
    for each numeric column, the row holding its largest magnitude (so the
    maximum moment or the largest settlement is never dropped from the
    printed table). Returns ``(row_indices, note)``; ``note`` is "" when
    every row is printed.
    """
    n = table.n_rows
    if n <= max_rows:
        return list(range(n)), ""
    step = int(math.ceil(n / float(max_rows)))
    keep = set(range(0, n, step))
    keep.add(n - 1)
    for j in range(len(table.columns)):
        best, best_i = None, None
        for i, row in enumerate(table.rows):
            v = row[j]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            if isinstance(v, float) and math.isnan(v):
                continue
            if best is None or abs(v) > best:
                best, best_i = abs(v), i
        if best_i is not None:
            keep.add(best_i)
    idx = sorted(keep)
    note = (f"{len(idx)} of {n} rows printed: one row in {step}, the last "
            "row, and the row holding each column's largest value")
    return idx, note


def tables_from(obj) -> List[CalcTable]:
    """``obj.get_tables()`` when ``obj`` has one, else ``[]``."""
    get = getattr(obj, "get_tables", None)
    if not callable(get):
        return []
    out = get() or []
    return [t for t in out if isinstance(t, CalcTable)]


__all__ = ["TableColumn", "CalcTable", "csv_cell", "format_cell",
           "display_rows", "tables_from"]
