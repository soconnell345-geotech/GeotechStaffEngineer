"""The sample calc packages (BACKLOG GM-328): every ``*_package`` method
of the calc_package adapter has a row in ``docs/examples/
sample_packages.json``, each row's call runs and returns a package, and a
row taken from a worked example carries that example's inputs.

The whole set builds in about 20 s without PDFs, so every row runs in
every test run; the script's own end-to-end build (files, index, PDFs for
two packages) is one more test, and the full build with every PDF is
marked slow."""

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "docs" / "examples" / "build_sample_packages.py"


def _load():
    spec = importlib.util.spec_from_file_location("build_sample_packages",
                                                  SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


B = _load()
ROWS = B.load_table()
_HEADER = ("project_name", "project_number", "given")


def test_every_package_method_has_one_row():
    methods = [r["method"] for r in ROWS]
    assert sorted(methods) == B.package_methods()
    assert len(set(methods)) == len(methods)
    assert len({B.stem(r) for r in ROWS}) == len(ROWS)


def test_every_row_says_where_its_inputs_come_from():
    for r in ROWS:
        assert r["origin"] in ("worked_example", "validation_example",
                               "printed_example", "made_up"), r["method"]
        assert r["example_id"] and r["source"], r["method"]
        if r["origin"] == "made_up":
            assert r.get("note"), f"{r['method']}: say why no example fits"
            assert r["source"].startswith("Made-up inputs")
        # the example states every input passed
        assert r["params"]["given"] == sorted(
            k for k in r["params"] if k not in _HEADER)


def test_a_worked_example_row_carries_the_examples_inputs():
    with open(REPO / "funhouse_agent" / "worked_examples.json",
              encoding="utf-8") as fh:
        examples = {e["id"]: e for e in json.load(fh)}
    from funhouse_agent.adapters import calc_package as cp
    checked = 0
    for r in ROWS:
        if r["origin"] != "worked_example":
            continue
        entry = examples[r["example_id"]]
        call = entry["dispatch_calls"][r["example_call"]]
        assert r["example_tool"] == f'{call["module"]}.{call["method"]}'
        assert r["source"] == entry["source"]
        selector = (cp._PACKAGE_TOOLS.get(r["method"], {})
                    .get("selector") or [None])[0]
        own = {k: v for k, v in r["params"].items()
               if k not in _HEADER and k != selector}
        if selector not in call["params"]:
            assert own == call["params"], r["method"]
        else:                      # the selector is the tool's own input too
            assert {**own, selector: r["params"][selector]} == call["params"]
        checked += 1
    assert checked >= 10


@pytest.mark.parametrize("row", ROWS, ids=[r["method"] for r in ROWS])
def test_the_rows_call_returns_a_package(row, tmp_path):
    path = tmp_path / f"{B.stem(row)}.html"
    out = B.call_package(row, str(path))
    assert isinstance(out, dict) and "error" not in out, out
    assert out["status"] == "success" and out["file_exists"], out
    assert not out.get("tables_error"), out["tables_error"]
    html = path.read_text(encoding="utf-8")
    assert "Given Inputs" in html and "References" in html
    assert "No input was named as given" not in html
    assert row["example_id"] in html            # the header's number


def test_the_script_writes_the_files_the_index_names(tmp_path):
    only = ["settlement_package", "soe_package"]
    entries = B.build(tmp_path, only=only, pdf=True, say=lambda *_: None)
    assert [e["status"] for e in entries] == ["success", "success"]
    index = json.loads((tmp_path / "index.json").read_text("utf-8"))
    assert [e["method"] for e in index["packages"]] == only
    assert index["app_version"] == entries[0]["app_version"]
    for e in index["packages"]:
        files = e["files"]
        assert files["html"] == f'{e["module"]}__{e["method"]}.html'
        named = [files["html"]] + [t["file"] for t in files.get("tables", [])]
        named += [f["file"] for f in e["figures"]]
        if "pdf" in files:
            named.append(files["pdf"])
        else:
            assert e.get("pdf_error")
        for name in named:
            assert (tmp_path / name).is_file(), name
        for key in ("example_id", "origin", "source"):
            assert e[key]
    settlement = index["packages"][0]
    assert settlement["figures"][0]["file"].startswith(
        "figures/settlement__settlement_package__01_")
    text = (tmp_path / "INDEX.md").read_text("utf-8")
    assert "settlement_package" in text and "NHI-06-089" in text

    # a rebuild replaces a package's files under the same names
    before = sorted(p.name for p in tmp_path.rglob("*") if p.is_file())
    B.build(tmp_path, only=["settlement_package"], pdf=False,
            say=lambda *_: None)
    after = sorted(p.name for p in tmp_path.rglob("*") if p.is_file())
    assert after == [n for n in before
                     if n != "settlement__settlement_package.pdf"]
    index = json.loads((tmp_path / "index.json").read_text("utf-8"))
    assert [e["method"] for e in index["packages"]] == only


@pytest.mark.slow
def test_the_whole_set_builds_with_pdfs(tmp_path):
    entries = B.build(tmp_path, pdf=True, say=lambda *_: None)
    assert [e["method"] for e in entries
            if e["status"] != "success"] == []
    assert len(entries) == len(ROWS)
    assert all(e["files"].get("pdf") for e in entries), [
        (e["method"], e.get("pdf_error")) for e in entries
        if not e["files"].get("pdf")]


def test_the_output_folder_is_ignored_by_git():
    lines = (REPO / ".gitignore").read_text("utf-8").splitlines()
    assert "docs/examples/out/" in lines
