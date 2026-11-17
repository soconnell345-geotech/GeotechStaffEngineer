"""Build one sample calc package per ``*_package`` method (BACKLOG GM-328).

    python docs/examples/build_sample_packages.py [OUT_DIR] [--only NAME ...]
                                                  [--no-pdf] [--list]

For every ``*_package`` method in the calc_package adapter's registry this
builds one sample package through the dispatch call the agent makes
(``call_agent("calc_package", <method>, params)``), from the inputs of a
published worked example. The inputs live in ONE table beside this script,
``sample_packages.json``: the module, the method, the example's id and
printed source, and the call's parameters. Nothing about a package is
written in this file.

Output, in ``OUT_DIR`` (default ``docs/examples/out/``, which git ignores;
no output is committed):

    <module>__<method>.html          the package, as the agent's call saves it
    <module>__<method>.pdf           the same call with format="pdf", where
                                     the host can render one (--no-pdf: none)
    tables/<module>__<method>__*.csv the package's calculation tables
    figures/<module>__<method>__<nn>_<title>.png
                                     each figure the package draws, by
                                     calc_package.render_figures on the same
                                     inputs (nn: its order in the package)
    index.json                       per package: module, method, example id,
                                     origin, source, the files written, the
                                     numbers the call returned, the app
                                     version
    INDEX.md                         the same, for a person

The names are stable: they come from the table's module and method and
from each figure's order and title, never from the clock, and a rebuild
replaces a package's own files. A later task builds a figure test set on
them (BACKLOG GM-314).

A call that fails is written into the index with its error and the script
exits 1; the other packages are still built.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
TABLE = HERE / "sample_packages.json"
DEFAULT_OUT = HERE / "out"

#: Keys of a package response that are files, bookkeeping or long lists:
#: left out of the index's ``numbers``.
_NOT_NUMBERS = frozenset({
    "status", "output_path", "file_exists", "file_size_bytes", "format",
    "html_length", "tables", "tables_note", "defaults_used", "renderer",
    "renderer_warnings", "rescue_path"})


def load_table(path: Path = TABLE) -> list:
    """The sample table's rows."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["packages"]


def package_methods() -> list:
    """Every ``*_package`` method the calc_package adapter registers."""
    from funhouse_agent.adapters import calc_package as cp
    return sorted(m for m in cp.METHOD_REGISTRY if m.endswith("_package"))


def stem(row: dict) -> str:
    """A row's file name stem: ``<module>__<method>``."""
    return f'{row["module"]}__{row["method"]}'


def _slug(text: str, limit: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return slug[:limit].rstrip("_") or "figure"


def version_and_source() -> tuple:
    """(version, where it was read): the tree's own ``pyproject.toml``,
    the tree the samples are built from; the installed metadata only if
    that cannot be read (GM-328 review B3: the installed distribution can
    be another release)."""
    try:
        import tomllib
        with open(REPO / "pyproject.toml", "rb") as fh:
            return tomllib.load(fh)["project"]["version"], "pyproject.toml"
    except (OSError, KeyError, TypeError, ValueError, ImportError):
        pass
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("geotech-staff-engineer"), "installed metadata"
    except PackageNotFoundError:
        return "unknown", "none found"


def app_version() -> str:
    return version_and_source()[0]


def _version_said() -> str:
    ver, source = version_and_source()
    if source == "pyproject.toml":
        return ver
    return (f"{ver} (read from the {source}: the tree's pyproject.toml "
            f"could not be read)")


def call_package(row: dict, output_path: str, fmt: str = "html") -> dict:
    """The agent's own call for ``row``: the package saved at
    ``output_path``."""
    from funhouse_agent.dispatch import call_agent
    params = {**row["params"], "output_path": output_path}
    if fmt != "html":
        params["format"] = fmt
    return call_agent("calc_package", row["method"], params)


def call_figures(row: dict, output_dir: str) -> dict:
    """The agent's own call for ``row``'s figures alone."""
    from funhouse_agent.dispatch import call_agent
    return call_agent("calc_package", "render_figures", {
        **row["params"], "package": row["method"], "output_dir": output_dir})


def _clear(out: Path, name: str) -> None:
    """Remove the files an earlier build wrote for ``name``."""
    for folder in (out, out / "tables", out / "figures"):
        if folder.is_dir():
            for path in folder.iterdir():
                if path.is_file() and (path.stem == name or
                                       path.name.startswith(name + "__")):
                    path.unlink()


def _numbers(response: dict) -> dict:
    """What the call returned beside its files, small values only."""
    out = {}
    for key, value in response.items():
        if key in _NOT_NUMBERS:
            continue
        if value is None or isinstance(value, (bool, int, float)):
            out[key] = value
        elif isinstance(value, str) and len(value) <= 400:
            out[key] = value
    return out


def build_one(row: dict, out: Path, pdf: bool = True) -> dict:
    """Build ``row``'s sample in ``out``; its index entry."""
    name = stem(row)
    entry = {
        "module": row["module"], "method": row["method"],
        "example_id": row["example_id"], "origin": row["origin"],
        "source": row["source"], "app_version": app_version(),
        "status": "success", "files": {}, "figures": [],
    }
    if row.get("note"):
        entry["note"] = row["note"]
    _clear(out, name)
    with tempfile.TemporaryDirectory(prefix="sample_pkg_") as tmp:
        tmp = Path(tmp)
        html = tmp / f"{name}.html"
        response = call_package(row, str(html))
        if (not isinstance(response, dict) or "error" in response
                or response.get("status") != "success"
                or not response.get("file_exists")):
            entry["status"] = "error"
            entry["error"] = str(
                response.get("error") if isinstance(response, dict)
                else response)
            return entry
        shutil.copyfile(html, out / html.name)
        entry["files"]["html"] = html.name
        entry["numbers"] = _numbers(response)
        if response.get("tables_error"):
            entry["tables_error"] = response["tables_error"]
        if response.get("defaults_used"):
            entry["defaults_used"] = response["defaults_used"]

        tables = []
        for table in response.get("tables") or []:
            src = Path(table.get("output_path") or "")
            if not src.is_absolute():
                src = tmp / src
            if not src.is_file():
                continue
            rest = src.stem[len(name):].lstrip("_") or "table"
            dest = f"tables/{name}__{rest}.csv"
            (out / "tables").mkdir(exist_ok=True)
            shutil.copyfile(src, out / dest)
            tables.append({"title": table.get("title"), "file": dest,
                           "rows": table.get("rows")})
        if tables:
            entry["files"]["tables"] = tables

        fig_dir = tmp / "figs"
        fig_dir.mkdir()
        figs = call_figures(row, str(fig_dir))
        if isinstance(figs, dict) and "error" in figs:
            entry["figures_error"] = str(figs["error"])
        else:
            (out / "figures").mkdir(exist_ok=True)
            for i, fig in enumerate(figs.get("figures") or [], start=1):
                src = Path(fig.get("output_path") or "")
                if not src.is_absolute():
                    src = fig_dir / src
                if not src.is_file():
                    continue
                dest = (f"figures/{name}__{i:02d}_"
                        f"{_slug(fig.get('title'))}.png")
                shutil.copyfile(src, out / dest)
                entry["figures"].append({
                    "file": dest, "title": fig.get("title"),
                    "caption": fig.get("caption")})

        if pdf:
            pdf_path = tmp / f"{name}.pdf"
            try:
                made = call_package(row, str(pdf_path), fmt="pdf")
            except Exception as exc:                       # noqa: BLE001
                made = {"error": f"{type(exc).__name__}: {exc}"}
            if (isinstance(made, dict) and made.get("file_exists")
                    and pdf_path.is_file()):
                shutil.copyfile(pdf_path, out / pdf_path.name)
                entry["files"]["pdf"] = pdf_path.name
                if made.get("renderer"):
                    entry["pdf_renderer"] = made["renderer"]
                if made.get("figure_warnings"):
                    entry["pdf_figure_warnings"] = made["figure_warnings"]
            else:
                entry["pdf_error"] = str(
                    made.get("error") if isinstance(made, dict) else made)
    return entry


def index_markdown(entries: list) -> str:
    """INDEX.md: the index for a person."""
    lines = [
        "# Sample calc packages", "",
        f"Built by `docs/examples/build_sample_packages.py` from "
        f"`docs/examples/sample_packages.json`, app version "
        f"{_version_said()}. One package per `*_package` method, each through "
        f"the call the agent makes, from a published worked example's "
        f"inputs (made-up inputs where the table says so).", "",
        "| Module | Method | Example | Origin | Package | Figures | Tables |",
        "|---|---|---|---|---|---|---|"]
    for e in entries:
        files = e.get("files", {})
        package = ", ".join(
            f"[{kind}]({files[kind]})" for kind in ("html", "pdf")
            if files.get(kind)) or f"FAILED: {e.get('error', '')[:120]}"
        lines.append(
            f"| {e['module']} | `{e['method']}` | {e['example_id']} | "
            f"{e['origin'].replace('_', ' ')} | {package} | "
            f"{len(e.get('figures') or [])} | "
            f"{len(files.get('tables') or [])} |")
    lines += ["", "## Sources", ""]
    for e in entries:
        lines.append(f"- **{e['example_id']}** (`{e['method']}`): "
                     f"{e['source']}"
                     + (f" Note: {e['note']}" if e.get("note") else ""))
    problems = [e for e in entries
                if e["status"] != "success" or e.get("tables_error")
                or e.get("figures_error") or e.get("pdf_error")]
    if problems:
        lines += ["", "## Problems", ""]
        for e in problems:
            for key in ("error", "tables_error", "figures_error",
                        "pdf_error"):
                if e.get(key):
                    lines.append(f"- `{e['method']}` {key}: {e[key]}")
    return "\n".join(lines) + "\n"


def build(out: Path = DEFAULT_OUT, only=None, pdf: bool = True,
          say=print) -> list:
    """Build the samples of ``only`` (methods or example ids; all when
    empty) in ``out`` and write the index; the index entries."""
    os.environ.setdefault("GEOTECH_NO_DISCLAIMER", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    rows = load_table()
    if only:
        wanted = set(only)
        rows = [r for r in rows if r["method"] in wanted
                or r["example_id"] in wanted or r["module"] in wanted]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    for row in rows:
        entry = build_one(row, out, pdf=pdf)
        entries.append(entry)
        say(f"{entry['status']:8s} {stem(row)}  "
            f"({len(entry['figures'])} figures"
            + (", pdf" if entry["files"].get("pdf") else "") + ")"
            + (f"  {entry.get('error', '')}" if entry["status"] != "success"
               else ""))
    # a partial build (--only) keeps the other packages' entries
    index_path = out / "index.json"
    kept = []
    if only and index_path.is_file():
        with open(index_path, encoding="utf-8") as fh:
            built = {e["method"] for e in entries}
            kept = [e for e in json.load(fh).get("packages", [])
                    if e["method"] not in built]
    order = {r["method"]: i for i, r in enumerate(load_table())}
    listed = sorted(kept + entries, key=lambda e: order.get(e["method"], 1e9))
    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump({"app_version": app_version(),
                   "app_version_source": version_and_source()[1],
                   "packages": listed}, fh, indent=1)
        fh.write("\n")
    (out / "INDEX.md").write_text(index_markdown(listed), encoding="utf-8")
    return entries


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", nargs="?", default=str(DEFAULT_OUT),
                        help="output folder (default docs/examples/out/)")
    parser.add_argument("--only", nargs="+", metavar="NAME",
                        help="build only these package methods, modules or "
                             "example ids")
    parser.add_argument("--no-pdf", action="store_true",
                        help="HTML, tables and figures only")
    parser.add_argument("--list", action="store_true",
                        help="print the table's rows and exit")
    args = parser.parse_args(argv)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    if args.list:
        for row in load_table():
            print(f'{stem(row):48s} {row["origin"]:18s} {row["example_id"]}')
        return 0
    missing = sorted(set(package_methods())
                     - {r["method"] for r in load_table()})
    if missing:
        print("no row in sample_packages.json for: " + ", ".join(missing),
              file=sys.stderr)
    entries = build(Path(args.out), only=args.only, pdf=not args.no_pdf)
    failed = [e["method"] for e in entries if e["status"] != "success"]
    if failed:
        print("failed: " + ", ".join(failed), file=sys.stderr)
    # a package built with a part missing fails the run too (GM-328 review
    # C3); INDEX.md lists it under "Problems"
    flawed = [e["method"] for e in entries if e["status"] == "success"
              and (e.get("tables_error") or e.get("figures_error")
                   or e.get("pdf_error"))]
    if flawed:
        print("built with a part missing (INDEX.md, Problems): "
              + ", ".join(flawed), file=sys.stderr)
    return 1 if failed or missing or flawed else 0


if __name__ == "__main__":
    raise SystemExit(main())
