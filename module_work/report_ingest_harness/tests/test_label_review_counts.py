"""The RI-15 label-review counts, on run files written by hand.

Small enough to check on paper: one made-up report of eight pages, the vote's
labels, two voters, the hand labels, and five changes the review made with
the evidence each named.
"""

from __future__ import annotations

import json

from module_work.report_ingest_harness import measure_label_review as m

#: page: (rules, vision, the vote's label, hand label)
PAGES = {
    0: ("narrative", "narrative", "narrative", "narrative"),
    1: ("other", "figure", "other", "figure"),          # fixed by a look
    2: ("calculation", "figure", "calculation", "calculation"),  # broke, ledger
    3: ("other", "lab_test", "other", "other"),         # broke, ledger
    4: ("lab_test", "narrative", "lab_test", "toc"),    # wrong to wrong
    5: ("other", "photos", "other", "photos"),          # fixed, ledger
    6: ("plan", "plan", "plan", "plan"),
    7: ("figure", "plan", "plan", "figure"),            # fixed, to the rules'
}
CHANGES = [
    {"page": 1, "from": "other", "to": "figure", "evidence": "read_page",
     "reason": "never read"},
    {"page": 2, "from": "calculation", "to": "figure", "evidence": "ledger",
     "reason": "never read"},
    {"page": 3, "from": "other", "to": "lab_test", "evidence": "ledger",
     "reason": "never read"},
    {"page": 4, "from": "lab_test", "to": "narrative",
     "evidence": "render_page", "reason": "never read"},
    {"page": 5, "from": "other", "to": "photos", "evidence": "ledger",
     "reason": "never read"},
    {"page": 7, "from": "plan", "to": "figure", "evidence": "read page",
     "reason": "never read"},
]


def _write(folder):
    folder.mkdir()
    labels = {"split_pages": [1, 2, 3, 4, 5, 7], "pages": [
        {"page": p, "label": vote, "agreed": rules == vision,
         "voters": [{"voter": "rules", "label": rules},
                    {"voter": "vision", "label": vision}]}
        for p, (rules, vision, vote, _) in PAGES.items()]}
    review = {"changes": CHANGES, "asked_pages": list(range(8)),
              "cost": {"calls": 4, "input_tokens": 4000}}
    (folder / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    (folder / "review.json").write_text(json.dumps(review), encoding="utf-8")


def _run(tmp_path):
    _write(tmp_path / "R98")
    hand = {p: row[3] for p, row in PAGES.items()}
    return m.read_run(tmp_path / "R98", "R98", hand)


def test_each_change_gets_its_verdict_and_whose_label_it_took(tmp_path):
    run = _run(tmp_path)
    rows = {r["page"]: r for r in m.verdicts(run)}
    assert {p: r["verdict"] for p, r in rows.items()} == {
        1: "fixed", 2: "broke", 3: "broke", 4: "still_wrong", 5: "fixed",
        7: "fixed"}
    assert rows[1]["side"] == "vision's"
    assert rows[7]["side"] == "the rules'"
    assert rows[4]["side"] == "vision's"
    assert all("reason" not in r for r in rows.values())


def test_the_rule_keeps_only_the_changes_that_name_a_look(tmp_path):
    run = _run(tmp_path)
    kept = m.verdicts(run, lambda c: m.names_a_look(c["evidence"]))
    assert sorted(r["page"] for r in kept) == [1, 4, 7]


def test_accuracy_before_after_and_under_the_rule(tmp_path):
    run = _run(tmp_path)
    everything = m.accuracy(run, m.verdicts(run))
    ruled = m.accuracy(run, m.verdicts(
        run, lambda c: m.names_a_look(c["evidence"])))
    # Before: pages 0, 2, 3 and 6 are right. After all changes: 0, 1, 5, 6
    # and 7. Under the rule: 0, 1, 2, 3, 6 and 7.
    assert everything == {"scored": 8, "before": 4, "after": 5}
    assert ruled == {"scored": 8, "before": 4, "after": 6}


def test_the_printout_counts_by_evidence_and_under_the_rule(tmp_path):
    lines = m.report([_run(tmp_path)])
    text = "\n".join(lines)
    assert "ALL changes: 6; fixed 3, broke 2, still wrong 1" in text
    assert "ALL changes -- under the RI-15 rule: 3; fixed 2, broke 0, " \
           "still wrong 1" in text
    (ledger,) = [line for line in lines if line.startswith("ledger ")]
    assert ledger.split()[1:] == ["3", "1", "2", "0", "0"]
    (row,) = [line for line in lines if line.startswith("R98 ")]
    assert row.split()[8:11] == ["0.500", "0.625", "0.750"]
    assert row.split()[-1] == "1,000"


def test_a_blind_report_is_never_read(tmp_path, monkeypatch, capsys):
    from report_ingest.scoring import OOS_BLIND

    blind = OOS_BLIND[0]
    _write(tmp_path / blind)

    class NoSheets:
        def __init__(self, *a, **k):
            pass

        def mapped_ids(self):
            return [blind]

        def labels_for(self, rid):
            raise AssertionError("a blind report's labels were opened")

    monkeypatch.setattr("report_ingest.corpus.Corpus", NoSheets)
    assert m.main([str(tmp_path), "--raw-dir", str(tmp_path)]) == 0
    assert "in the blind set" in capsys.readouterr().err


def test_the_rule_column_divides_by_its_own_page_count(tmp_path, monkeypatch):
    run = _run(tmp_path)
    real = m.accuracy

    def fewer_under_the_rule(r, applied):
        out = real(r, applied)
        if len(list(applied)) < 6:           # the rule's subset
            out = dict(out, scored=4)
        return out

    monkeypatch.setattr(m, "accuracy", fewer_under_the_rule)
    (row,) = [line for line in m.report([run]) if line.startswith("R98 ")]
    # before and after over 8 pages; the rule's 6 hits over its own 4
    assert row.split()[8:11] == ["0.500", "0.625", "1.500"]
