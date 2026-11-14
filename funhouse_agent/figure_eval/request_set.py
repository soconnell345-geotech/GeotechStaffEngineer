"""The figure test set's requests and their data folder (BACKLOG GM-314).

``requests.json`` beside this file lists about twenty figures an engineer
asks the geotech page for, on made-up projects. A request carries no
numbers of its own: its inputs are a sample calc package's
(``docs/examples/sample_packages.json``, BACKLOG GM-328), named by the
package method in ``row``; ``replace`` swaps in a method probe's made-up
inputs (``funhouse_agent/tests/method_probes.json``) where the sample table
has none for the figure (without a probe, the request's own made-up
inputs), and ``patch`` changes single inputs.

Neither table ships in the app's wheel, so the set travels as a DATA FOLDER,
made where the repository is checked out (``python -m
funhouse_agent.figure_eval.request_set --out DIR``, no model, $0):

``requests.json``
    every request resolved: its inputs, the prompt the page is asked, the
    sample row it came from;
``references/<request id>/``
    GM-328's build of that row (``build_sample_packages.build_one``): the
    package, its tables, its own figures and ``reference.json`` with the
    numbers the calculation returned. The vision check compares the numbers
    on a figure with these.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REQUESTS_FILE = HERE / "requests.json"
SAMPLE_TABLE = REPO / "docs" / "examples" / "sample_packages.json"
PROBES_FILE = REPO / "funhouse_agent" / "tests" / "method_probes.json"
DATA_REQUESTS = "requests.json"
REFERENCES = "references"
REFERENCE_FILE = "reference.json"

#: What a figure is.
KINDS = ("profile", "wall", "time", "curve", "section", "plan", "cases",
         "sweep")

#: A sample row's inputs a request never passes: the package's header and
#: bookkeeping (the page writes its own files).
HEADER = ("project_name", "project_number", "given", "output_path",
          "format")


def _read(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def definitions(path: Path = REQUESTS_FILE) -> List[Dict[str, Any]]:
    """The requests as written in ``requests.json``."""
    return _read(path)["requests"]


def sample_rows(path: Path = SAMPLE_TABLE) -> Dict[str, Dict[str, Any]]:
    """GM-328's sample table, by package method."""
    return {r["method"]: r for r in _read(path)["packages"]}


def probes(path: Path = PROBES_FILE) -> Dict[str, Any]:
    return _read(path)


def sample_row(req: Dict[str, Any], rows: Dict[str, Dict[str, Any]],
               probe_table: Optional[Dict[str, Any]] = None
               ) -> Dict[str, Any]:
    """The sample row ``req`` runs on: GM-328's row, with a probe's inputs
    swapped in (``replace``) and single inputs changed (``patch``)."""
    row = copy.deepcopy(rows[req["row"]])
    rep = req.get("replace")
    if rep:
        rep = dict(rep)
        key = rep.pop("probe", None)
        params = {k: row["params"][k] for k in ("project_name",
                                                 "project_number")
                  if k in row["params"]}
        if key:
            table = probe_table if probe_table is not None else probes()
            params.update(copy.deepcopy(table[key]["params"]))
            row["origin"] = "method_probe"
            row["source"] = (f"Made-up inputs (the {key} probe of "
                             "funhouse_agent/tests/method_probes.json)")
            row["example_id"] = f"PROBE-{key}"
        else:
            row["origin"] = "made_up"
            row["source"] = (f"Made-up inputs (figure request {req['id']}, "
                             "funhouse_agent/figure_eval/requests.json)")
            row["example_id"] = f"MADE-UP-{req['id'].split('-')[0]}"
        row.pop("note", None)
        params.update(rep)
        row["params"] = params
    if req.get("patch"):
        row["params"].update(copy.deepcopy(req["patch"]))
    return row


def inputs_of(row: Dict[str, Any]) -> Dict[str, Any]:
    """The inputs a request passes: the row's call without its header."""
    return {k: v for k, v in row["params"].items() if k not in HEADER}


def prompt(req: Dict[str, Any], inputs: Dict[str, Any]) -> str:
    """What the page is asked: the engineer's request, then the inputs."""
    return (f"{req['ask']}\n\nInputs (SI units unless a name says "
            f"otherwise):\n{json.dumps(inputs)}")


def resolve(defs: Optional[List[Dict[str, Any]]] = None,
            rows: Optional[Dict[str, Dict[str, Any]]] = None,
            probe_table: Optional[Dict[str, Any]] = None
            ) -> List[Dict[str, Any]]:
    """Every request with its sample row, inputs and prompt."""
    defs = definitions() if defs is None else defs
    rows = sample_rows() if rows is None else rows
    out = []
    for req in defs:
        row = sample_row(req, rows, probe_table)
        inputs = inputs_of(row)
        out.append({
            **{k: v for k, v in req.items() if k not in ("replace", "patch")},
            "module": row["module"], "method": row["method"],
            "origin": row.get("origin"), "example_id": row.get("example_id"),
            "source": row.get("source"), "elevation": req.get("elevation"),
            "numpy_note": bool(req.get("numpy_note")),
            "inputs": inputs, "prompt": prompt(req, inputs),
            "sample_row": row})
    return out


def select(requests: List[Dict[str, Any]], ids: Optional[Iterable[str]]
           ) -> List[Dict[str, Any]]:
    """The requests whose id is, or starts with, one of ``ids`` (all when
    none)."""
    if not ids:
        return list(requests)
    ids = [str(i) for i in ids]
    picked = [r for r in requests
              if any(r["id"] == i or r["id"].startswith(i + "-")
                     for i in ids)]
    missing = [i for i in ids
               if not any(r["id"] == i or r["id"].startswith(i + "-")
                          for r in requests)]
    if missing:
        raise KeyError(f"no figure request {', '.join(missing)}")
    return picked


def build_reference(req: Dict[str, Any], out: Path) -> Dict[str, Any]:
    """GM-328's build of the request's row in ``out``: its index entry
    (status, the numbers the call returned, the package's figures)."""
    sys.path.insert(0, str(REPO / "docs" / "examples"))
    try:
        import build_sample_packages as bsp
    finally:
        sys.path.pop(0)
    out.mkdir(parents=True, exist_ok=True)
    try:
        entry = bsp.build_one(req["sample_row"], out, pdf=False)
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal
        entry = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    ref = {"request": req["id"], "method": req["method"],
           "status": entry.get("status"), "error": entry.get("error"),
           "numbers": entry.get("numbers") or {},
           "figures": entry.get("figures") or []}
    (out / REFERENCE_FILE).write_text(json.dumps(ref, indent=1, default=str),
                                      encoding="utf-8")
    return ref


def prepare(out_dir: Any, ids: Optional[Iterable[str]] = None,
            references: bool = True,
            build: Callable[[Dict[str, Any], Path], Dict[str, Any]] =
            build_reference, say: Callable[[str], None] = print
            ) -> List[Dict[str, Any]]:
    """Write the data folder (see the module's text) for ``ids`` (all when
    none); a reference already built is kept."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    reqs = select(resolve(), ids)
    if references:
        for req in reqs:
            ref_dir = out / REFERENCES / req["id"]
            if (ref_dir / REFERENCE_FILE).is_file():
                continue
            ref = build(req, ref_dir)
            say(f"{ref.get('status') or 'error':8s} reference {req['id']}"
                + (f"  {ref.get('error')}" if ref.get("error") else ""))
    about = _read(REQUESTS_FILE)["about"]
    (out / DATA_REQUESTS).write_text(json.dumps(
        {"about": about, "requests": reqs}, indent=1, default=str),
        encoding="utf-8")
    return reqs


def load_data(data_dir: Any) -> List[Dict[str, Any]]:
    """The resolved requests of a data folder, each with its reference
    (``None`` when none was built)."""
    root = Path(data_dir)
    reqs = _read(root / DATA_REQUESTS)["requests"]
    for req in reqs:
        path = root / REFERENCES / req["id"] / REFERENCE_FILE
        req["reference"] = _read(path) if path.is_file() else None
    return reqs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="the data folder to write")
    ap.add_argument("--ids", nargs="*", help="request ids (default: all)")
    ap.add_argument("--no-references", action="store_true",
                    help="write requests.json only")
    a = ap.parse_args(argv)
    reqs = prepare(a.out, a.ids, references=not a.no_references)
    print(f"{len(reqs)} requests written to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
