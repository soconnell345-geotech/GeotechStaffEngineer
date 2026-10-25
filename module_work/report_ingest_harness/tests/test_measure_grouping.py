"""Offline tests of ``measure_grouping`` (BACKLOG RI-87 piece 3), at $0.

Made-up answer-sheet file names, made-up range labels and made-up page
descriptions stand in for the private run folders. What is checked is what
the score relies on: an item's pages come from the stage's own rule, a blind
answer sheet is never opened, a group is scored whole, missed, split, mixed
or padded as the docstring says, and the results land where git ignores
them or nowhere.

Run::

    .venv/Scripts/python -m pytest module_work/report_ingest_harness/tests/test_measure_grouping.py -q
"""

from __future__ import annotations

import json
from pathlib import Path

from module_work.report_ingest_harness import measure_grouping as mg
from module_work.report_ingest_harness import run_claude_local as rcl
from report_ingest import model_first_pages as mfp


def _truth(root: Path) -> Path:
    """Made-up answer sheets: R01 open in every stage, R02 blind. A blind
    file holds no JSON at all, so opening it would fail the test."""
    truth = root / "truth"
    for stage in ("logs", "lab", "calc"):
        (truth / stage).mkdir(parents=True)
        (truth / stage / "OPEN.txt").write_text("R01\n", encoding="utf-8")
    (truth / "logs" / "R01_p5.json").write_text(json.dumps(
        {"id": "R01_p5", "pages": [5, 6], "layers": ["made up"]}),
        encoding="utf-8")
    (truth / "logs" / "R02_p9.json").write_text("not opened",
                                                encoding="utf-8")
    (truth / "lab" / "kind__R01_p20.json").write_text(json.dumps(
        {"id": "kind__R01_p20"}), encoding="utf-8")
    (truth / "lab" / "kind__R02_p3.json").write_text("not opened",
                                                     encoding="utf-8")
    (truth / "calc" / "kind__R01_p30.json").write_text(json.dumps(
        {"id": "kind__R01_p30", "pages": [30, 31, 32]}), encoding="utf-8")
    (truth / "calc" / "kind__R02_p40.json").write_text("not opened",
                                                       encoding="utf-8")
    return truth


def test_item_pages_follow_the_stage_rule_and_blind_files_stay_shut(
        tmp_path):
    truth = _truth(tmp_path)
    items = mg.load_items(truth, ["R01", "R02"])
    assert items["logs"] == {"R01": {"R01_p5": [5, 6]},
                             "R02": {"R02_p9": [9]}}
    assert items["lab"] == {"R01": {"kind__R01_p20": [20]},
                            "R02": {"kind__R02_p3": [3]}}
    assert items["calc"] == {"R01": {"kind__R01_p30": [30, 31, 32]},
                             "R02": {"kind__R02_p40": [40]}}
    assert mg.load_items(truth, ["R03"])["logs"] == {}


def test_a_held_out_report_stays_shut_even_if_a_stage_lists_it_open(
        tmp_path):
    held_out = mg.BLIND[0]
    truth = _truth(tmp_path)
    (truth / "logs" / "OPEN.txt").write_text(f"R01\n{held_out}\n",
                                             encoding="utf-8")
    (truth / "logs" / f"{held_out}_p9.json").write_text("not opened",
                                                        encoding="utf-8")
    items = mg.load_items(truth, ["R01", held_out])
    assert items["logs"][held_out] == {f"{held_out}_p9": [9]}


def test_a_group_is_scored_whole_missed_split_mixed_and_padded():
    items = {"a": [10, 11], "b": [12], "c": [20, 21], "d": [30]}
    units = [[10, 11, 12, 13], [20], [21], [40, 41]]
    labels = {30: "other"}
    scored = mg.score_stage("calc", units, items, labels)
    rows = {r["item"]: r for r in scored["items"]}
    assert rows["a"]["whole"] and rows["a"]["mixed_with"] == ["b"]
    assert rows["a"]["extra"] == [12, 13]
    assert rows["c"]["split"] and not rows["c"]["whole"]
    assert rows["d"]["missed"] == [{"page": 30, "label": "other"}]
    assert not rows["d"]["whole"]
    assert (scored["n_groups"], scored["empty_groups"],
            scored["empty_pages"]) == (4, 1, 2)


def test_a_long_log_s_parts_count_as_one_group():
    groups = [mfp.PageGroup("logs", [1, 2], key="log-1", part=(1, 2)),
              mfp.PageGroup("logs", [3, 4], key="log-1", part=(2, 2)),
              mfp.PageGroup("logs", [5], key="log-5")]
    assert mg.units_of("logs", groups) == [[1, 2, 3, 4], [5]]
    assert mg.units_of("lab", groups) == [[1, 2], [3, 4], [5]]


def _d(**fields):
    base = {"title": "", "kind": "made-up page", "names": "",
            "identifiers": [], "continues": "", "summary": "made up"}
    base.update(fields)
    return base


def _x(exploration, **fields):
    return dict(_d(**fields), exploration=exploration)


def _proxy_pages():
    said = {1: _x("QX-1", continues="Sheet 1 of 2"),
            2: _x("QX-1", continues="Sheet 2 of 2"),
            3: _x("QX-2", continues="Sheet 1 of 3"),
            4: _x("QX-3"),
            5: _x("QX-4", continues="Sheet 1 of 2"),
            6: _x("QX-5", continues="Sheet 2 of 2, continued from the "
                                    "page before"),
            7: _x("QX-2", continues="Sheet 2 of 3")}
    return {page: {"description": d} for page, d in said.items()}


def test_the_log_proxies_read_the_sheet_numbers_and_the_explorations():
    descriptions = _proxy_pages()
    groups = mfp.log_groups(sorted(descriptions), descriptions)
    assert [g.pages for g in groups] == [[1, 2], [3], [4], [5, 6], [7]]
    proxies = mg.log_proxies(groups, descriptions)
    assert proxies["counts"] == {
        "complete_set": 2, "cut_set": 2, "count_differs": 1,
        "no_sheet_numbers": 1, "holes>1": 1, "ids_many_groups": 1,
        "ids_many_groups_apart": 1}
    assert proxies["cut"] == [[3, 3, 1], [7, 7, 1]]
    assert proxies["count_differs"] == [[3, 3, 1, 3]]
    assert proxies["holes_many"] == [[5, 6, ["QX4", "QX5"]]]
    assert proxies["ids_many_groups"] == [["QX2", [3, 7], True]]
    assert proxies["broken"] == proxies["sheet1_many"] == []


def test_the_proxies_are_listed_by_pages_and_counts_only(tmp_path):
    descriptions = _proxy_pages()
    groups = mfp.log_groups(sorted(descriptions), descriptions)
    part = {"describer": "made-up", "folder": "made-up", "arm": "hinted_B",
            "reports": {"R01": {
                "n_pages": 8, "log_groups": [], "narrative": [],
                "log_proxies": mg.log_proxies(groups, descriptions),
                "log_boundaries": {}, **{
                    key: mg.score_stage("calc", [], {}, {})
                    for key, _title in mg.CALC_COLUMNS}}}}
    text = mg.write_results(tmp_path, {"arm": "hinted_B", "reports": ["R01"],
                                       "describers": {"made-up": part}})
    assert "R01 p7-7 (1 pp)" in text
    assert "R01 p3-3 (1 pp): its first sheet says 3" in text
    assert "R01 p5-6: 2 explorations" in text
    assert "R01: one exploration in groups starting p3, p7 (apart)" in text
    assert "QX" not in text


def _run_folder(root: Path) -> Path:
    """A made-up describer run: one report of eight pages."""
    folder = root / "describer"
    labels = {"0": "cover", "1": "narrative", "5": "boring_log",
              "6": "boring_log", "7": "boring_log"}
    (folder / "ranges" / "hinted_B").mkdir(parents=True)
    (folder / "ranges" / "hinted_B" / "R01.json").write_text(json.dumps(
        {"n_pages": 8, "labels": labels}), encoding="utf-8")
    pages = folder / "ranges" / "descriptions" / "R01"
    pages.mkdir(parents=True)
    said = {5: _d(identifiers=["QX-1"], continues="Sheet 1 of 2"),
            6: _d(continues="Sheet 2 of 2"),
            7: _d(identifiers=["QX-2"])}
    for page, description in said.items():
        (pages / f"p{page:04d}.json").write_text(json.dumps(
            {"page": page, "description": description}), encoding="utf-8")
    return folder


def test_a_describer_run_is_scored_and_written(tmp_path):
    folder = _run_folder(tmp_path)
    items = mg.load_items(_truth(tmp_path), ["R01"])
    part = mg.run_describer("made-up", folder, "hinted_B", ["R01"], items, {})
    report = part["reports"]["R01"]
    assert [g["pages"] for g in report["log_groups"]] == [[5, 6], [7]]
    assert report["logs"]["items"][0]["whole"]
    assert report["lab"]["items"][0]["missed"] == [{"page": 20,
                                                    "label": ""}]
    assert report["narrative"] == [[0, 7, 8]]
    out = tmp_path / "out"
    out.mkdir()
    text = mg.write_results(out, {"arm": "hinted_B", "reports": ["R01"],
                                  "describers": {"made-up": part}})
    assert (out / "results.json").is_file()
    assert "R01 (8 pp)" in text and "missed p20 (no label)" in text


def test_an_overlay_re_describes_some_pages_of_the_reports_it_covers(
        tmp_path):
    folder = _run_folder(tmp_path)
    items = mg.load_items(_truth(tmp_path), ["R01"])
    overlay = tmp_path / "overlay"
    (overlay / "R01").mkdir(parents=True)
    (overlay / "R01" / "p0006.json").write_text(json.dumps(
        {"page": 6, "description": _x("QX-3", continues="Sheet 2 of 2")}),
        encoding="utf-8")
    part = mg.run_describer("made-up", folder, "hinted_B", ["R01"], items,
                            {}, overlay=overlay)
    groups = [g["pages"] for g in part["reports"]["R01"]["log_groups"]]
    assert groups == [[5], [6], [7]]
    empty = tmp_path / "empty"
    empty.mkdir()
    assert mg.run_describer("made-up", folder, "hinted_B", ["R01"], items,
                            {}, overlay=empty)["reports"] == {}


def test_results_are_refused_where_git_would_track_them(tmp_path,
                                                        monkeypatch,
                                                        capsys):
    monkeypatch.setattr(rcl, "main_checkout", lambda start=None: tmp_path)
    assert mg.main(["--describers", "haiku", "--no-doc"]) == 2
    assert "not ignored by git" in capsys.readouterr().err
    assert not (tmp_path / rcl.OUT_REL / mg.OUT_NAME).exists()
