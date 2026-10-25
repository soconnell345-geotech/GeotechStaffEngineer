"""The range voter offline (BACKLOG RI-73): describe, label by sections.

No model is called. The describer runs on planlens' synthetic report, the
labeller on made-up descriptions, both through scripted engines, so what is
tested is what the pass SENDS and what Python does with the answer: every
page covered once with no gap or overlap, long reports stitched from
overlapping windows, the switch off leaving the vote as it was, and a
resumed run paying nothing twice.
"""

from __future__ import annotations

import json
import os

import pytest

from report_ingest import range_labels as rl
from report_ingest.label_review import LABEL_DEFINITIONS
from report_ingest.label_vote import POLICIES, Voter, combine
from report_ingest.scoring import (consequence, harm_score, is_harmful,
                                   miss_structure, reader_for)
from report_ingest.tests.fake_engine import FakeEngine


@pytest.fixture(scope="module")
def synthetic():
    from planlens.document import open_document
    from planlens.testing import build_synthetic_report

    doc = open_document(build_synthetic_report().pdf, name="synthetic")
    try:
        yield doc
    finally:
        doc.close()


def _description(**over):
    answer = {"title": "Made-up title", "kind": "report text",
              "names": "", "identifiers": [], "continues": "",
              "header_footer": "", "issuer_date": "",
              "language": "English",
              "summary": "A page of made-up text."}
    answer.update(over)
    return rl.PageDescription(**answer)


def _records(n, **over):
    """``page -> saved record`` for ``n`` made-up pages."""
    return {page: {"page": page, "printed": page + 1, "printed_of": None,
                   "description": _description(**over).model_dump()}
            for page in range(n)}


def _answer(sections, exceptions=()):
    return rl.RangeAnswer(
        sections=[rl.RangeSection(first_page=a, last_page=b, label=label,
                                  reason="made up")
                  for a, b, label in sections],
        exceptions=[rl.RangeException(page=p, label=label, reason="made up")
                    for p, label in exceptions])


# -- the switch ----------------------------------------------------------------

def test_the_switch_is_off_by_default(monkeypatch):
    monkeypatch.delenv(rl.RANGE_LABELS_ENV, raising=False)
    assert rl.range_labels_arm() is None


@pytest.mark.parametrize("value, arm", [
    ("0", None), ("off", None), ("", None), ("1", "plain"), ("on", "plain"),
    ("plain", "plain"), ("hinted", "hinted"), ("HINTED", "hinted"),
])
def test_the_switch_names_an_arm_or_off(monkeypatch, value, arm):
    monkeypatch.setenv(rl.RANGE_LABELS_ENV, value)
    assert rl.range_labels_arm() == arm


def test_a_typo_in_the_switch_is_refused_not_read_as_off(monkeypatch):
    monkeypatch.setenv(rl.RANGE_LABELS_ENV, "hinterd")
    with pytest.raises(ValueError, match="plain or hinted"):
        rl.range_labels_arm()


def test_the_context_manager_overrides_the_environment(monkeypatch):
    monkeypatch.setenv(rl.RANGE_LABELS_ENV, "hinted")
    with rl.use_range_labels("off"):
        assert rl.range_labels_arm() is None
    with rl.use_range_labels("plain"):
        assert rl.range_labels_arm() == "plain"
    assert rl.range_labels_arm() == "hinted"


# -- the instructions ----------------------------------------------------------

def test_the_labeller_is_told_every_label_and_the_conventions():
    system = rl.labeller_system()
    for name in LABEL_DEFINITIONS:
        if name not in rl.LABELS_NOT_USED:
            assert f"  {name}: " in system
    assert rl.LABEL_CONVENTIONS.strip() in system
    assert "rules:" not in system


def test_the_labeller_is_not_offered_a_label_the_conventions_never_give():
    assert "divider" in rl.LABELS_NOT_USED
    for hinted in (False, True):
        system = rl.labeller_system(hinted=hinted)
        assert "  divider: " not in system
        assert "divider label" in system          # convention 3 says why
    assert "  divider: " in rl.labeller_system(not_used=())


def test_the_hint_note_does_not_say_the_rules_miss_photographs_and_plots():
    note = rl.labeller_system(hinted=True).split("THE RULES' LABEL")[1]
    assert "photograph" not in note and "plot" not in note


def test_each_arms_instructions_have_their_own_fingerprint():
    plain, hinted = rl.labeller_fingerprint(False), rl.labeller_fingerprint(
        True)
    assert plain != hinted and len(plain) == 10
    assert rl.labeller_fingerprint(False, conventions="Made up.") != plain


def test_only_the_hinted_arm_is_told_about_the_rules_label():
    assert "THE RULES' LABEL" in rl.labeller_system(hinted=True)
    assert "THE RULES' LABEL" not in rl.labeller_system(hinted=False)


def test_the_conventions_are_one_replaceable_text_and_a_run_records_which():
    other = "Label every page by its look alone."
    system = rl.labeller_system(conventions=other)
    assert other in system
    assert rl.LABEL_CONVENTIONS.strip() not in system
    assert rl.conventions_version(other) != rl.conventions_version()
    found = rl.label_by_ranges(_records(2), 2, FakeEngine(
        [{"final": _answer([(0, 1, "narrative")])}]), conventions=other)
    assert found.conventions == rl.conventions_version(other)


def test_a_description_line_carries_the_page_index_and_what_was_printed():
    record = _records(1, kind="laboratory summary table",
                      identifiers=[f"S-{i}" for i in range(15)],
                      language="Spanish")[0]
    line = rl.description_line(0, record, hint="lab_test")
    assert line.startswith("[0] | printed 1")
    assert "kind: laboratory summary table" in line
    assert "S-11" in line and "S-12" not in line and "(+3 more)" in line
    assert "language: Spanish" in line
    assert line.endswith("rules: lab_test")
    assert "English" not in rl.description_line(0, _records(1)[0])
    assert "could not be described" in rl.description_line(5, None)


# -- the describer -------------------------------------------------------------

def _describer(n):
    return FakeEngine([{"final": _description()} for _ in range(n)])


def test_one_call_per_page_each_with_its_picture_and_its_text(synthetic,
                                                               tmp_path):
    engine = _describer(3)
    found = rl.describe_pages(synthetic, engine, pages=[0, 1, 2],
                              folder=str(tmp_path))
    assert engine.n_calls == 3
    assert sorted(found.pages) == [0, 1, 2]
    for page, call in enumerate(engine.calls):
        assert call["n_images"] == 1
        assert call["output_format"] is rl.PageDescription
        assert call["system"] == rl.DESCRIBE_SYSTEM
        words = call["messages"][0]["content"][0]["text"]
        assert f"page index {page})" in words
        assert "PAGE TEXT" in words or "carries no text" in words


def test_each_description_is_saved_as_its_own_file_as_it_is_made(synthetic,
                                                                  tmp_path):
    rl.describe_pages(synthetic, _describer(2), pages=[3, 4],
                      folder=str(tmp_path))
    assert sorted(os.listdir(tmp_path)) == ["p0003.json", "p0004.json"]
    blob = json.loads((tmp_path / "p0003.json").read_text(encoding="utf-8"))
    assert blob["page"] == 3 and blob["model"] == "fake-model"
    assert blob["prompt"] == rl.DESCRIBE_PROMPT_VERSION
    assert blob["description"]["kind"] == "report text"
    assert blob["render"]["width_px"] > 0


def test_a_resumed_pass_asks_only_for_the_pages_it_does_not_have(synthetic,
                                                                 tmp_path):
    rl.describe_pages(synthetic, _describer(2), pages=[0, 1],
                      folder=str(tmp_path))
    again = _describer(1)
    found = rl.describe_pages(synthetic, again, pages=[0, 1, 2],
                              folder=str(tmp_path))
    assert again.n_calls == 1
    assert found.reused == 2 and sorted(found.pages) == [0, 1, 2]
    assert found.cost["calls"] == 1
    assert found.saved_cost["calls"] == 3


def test_redo_asks_again_for_every_page(synthetic, tmp_path):
    rl.describe_pages(synthetic, _describer(2), pages=[0, 1],
                      folder=str(tmp_path))
    again = _describer(2)
    rl.describe_pages(synthetic, again, pages=[0, 1], folder=str(tmp_path),
                      redo=True)
    assert again.n_calls == 2


def test_a_folder_of_another_models_descriptions_is_refused(synthetic,
                                                            tmp_path):
    rl.describe_pages(synthetic, _describer(1), pages=[0],
                      folder=str(tmp_path))
    with pytest.raises(ValueError, match="rather than mix them"):
        rl.describe_pages(synthetic, FakeEngine([], name="other-model"),
                          pages=[0, 1], folder=str(tmp_path))


def test_a_page_whose_call_fails_is_unresolved_and_not_saved(synthetic,
                                                             tmp_path):
    engine = FakeEngine([{"final": _description()},
                         {"raise": RuntimeError("made-up outage")},
                         {"final": _description()}])
    found = rl.describe_pages(synthetic, engine, pages=[0, 1, 2],
                              folder=str(tmp_path))
    assert sorted(found.pages) == [0, 2]
    assert [u["page"] for u in found.unresolved] == [1]
    assert not (tmp_path / "p0001.json").exists()
    assert found.cost["failed"] == 1


def test_a_page_with_no_answer_is_asked_once_more(synthetic, tmp_path):
    engine = FakeEngine([{"text": "no tool call"}, {"final": _description()}])
    found = rl.describe_pages(synthetic, engine, pages=[0],
                              folder=str(tmp_path))
    assert engine.n_calls == 2 and sorted(found.pages) == [0]


def test_a_stop_error_ends_the_pass_with_what_is_done_kept(synthetic,
                                                           tmp_path):
    class CallCeilingReached(RuntimeError):
        pass

    engine = FakeEngine([{"final": _description()},
                         {"raise": CallCeilingReached("made-up ceiling")}])
    found = rl.describe_pages(synthetic, engine, pages=[0, 1, 2, 3],
                              folder=str(tmp_path))
    assert engine.n_calls == 2
    assert found.stopped.startswith("CallCeilingReached")
    assert sorted(found.pages) == [0]


def test_several_pages_at_once_give_the_same_descriptions(synthetic,
                                                          tmp_path):
    found = rl.describe_pages(synthetic, _describer(4), pages=[0, 1, 2, 3],
                              folder=str(tmp_path), workers=3)
    assert sorted(found.pages) == [0, 1, 2, 3]
    assert found.cost["calls"] == 4


# -- the labeller: coverage ------------------------------------------------------

def test_sections_label_every_page_and_an_exception_overrides_its_section():
    engine = FakeEngine([{"final": _answer(
        [(0, 2, "narrative"), (3, 5, "boring_log")], [(1, "figure")])}])
    found = rl.label_by_ranges(_records(6), 6, engine)
    assert found.labels == {0: "narrative", 1: "figure", 2: "narrative",
                            3: "boring_log", 4: "boring_log",
                            5: "boring_log"}
    assert found.exceptions == [{"page": 1, "label": "figure",
                                 "reason": "made up"}]
    assert [(s["first_page"], s["last_page"], s["label"])
            for s in found.sections] == [(0, 0, "narrative"),
                                         (1, 1, "figure"),
                                         (2, 2, "narrative"),
                                         (3, 5, "boring_log")]
    assert engine.n_calls == 1 and not found.unresolved


def test_every_description_is_one_line_in_page_order():
    engine = FakeEngine([{"final": _answer([(0, 3, "narrative")])}])
    rl.label_by_ranges(_records(4), 4, engine, outline="MADE-UP OUTLINE")
    words = engine.calls[0]["messages"][0]["content"][0]["text"]
    lines = [x for x in words.splitlines() if x.startswith("[")]
    assert [x.split("]")[0] for x in lines] == ["[0", "[1", "[2", "[3"]
    assert "MADE-UP OUTLINE" in words
    assert "Label every page." in words


def test_the_hinted_arm_puts_the_rules_label_on_every_line():
    engine = FakeEngine([{"final": _answer([(0, 1, "narrative")])}])
    found = rl.label_by_ranges(_records(2), 2, engine,
                               hints={0: "cover", 1: "narrative"})
    words = engine.calls[0]["messages"][0]["content"][0]["text"]
    assert "rules: cover" in words and "rules: narrative" in words
    assert found.arm == "hinted"
    assert "THE RULES' LABEL" in engine.calls[0]["system"]


def test_a_gap_is_asked_about_once_more_naming_the_pages():
    engine = FakeEngine([
        {"final": _answer([(0, 2, "narrative"), (5, 7, "lab_test")])},
        {"final": _answer([(0, 2, "narrative"), (3, 7, "lab_test")])}])
    found = rl.label_by_ranges(_records(8), 8, engine)
    assert engine.n_calls == 2
    again = engine.calls[1]["messages"][0]["content"][0]["text"]
    assert "no section covers page(s) 3-4" in again
    assert len(engine.calls[1]["messages"]) == 1     # a fresh call
    assert found.labels[3] == "lab_test" and not found.unresolved
    assert found.windows[0]["repaired"] is True


def test_a_gap_the_reask_does_not_fill_is_unresolved_never_guessed():
    gap = _answer([(0, 2, "narrative"), (5, 7, "lab_test")])
    engine = FakeEngine([{"final": gap}, {"final": gap}])
    found = rl.label_by_ranges(_records(8), 8, engine)
    assert found.unresolved == [3, 4]
    assert 3 not in found.labels and 4 not in found.labels
    assert any(q["where"] == "coverage" for q in found.qa)


def test_two_sections_giving_one_page_two_labels_leave_it_unresolved():
    both = _answer([(0, 4, "narrative"), (4, 7, "lab_test")])
    found = rl.label_by_ranges(_records(8), 8,
                               FakeEngine([{"final": both},
                                           {"final": both}]))
    assert found.unresolved == [4]
    assert found.labels[3] == "narrative" and found.labels[5] == "lab_test"


def test_without_repair_a_faulty_answer_is_not_asked_about_again():
    engine = FakeEngine([{"final": _answer([(0, 2, "narrative")])}])
    found = rl.label_by_ranges(_records(4), 4, engine, repair=False)
    assert engine.n_calls == 1 and found.unresolved == [3]


def test_a_label_outside_the_vocabulary_is_other_with_a_note():
    found = rl.label_by_ranges(_records(2), 2, FakeEngine(
        [{"final": _answer([(0, 1, "made_up_label")])},
         {"final": _answer([(0, 1, "made_up_label")])}]))   # the re-ask
    assert found.labels == {0: "other", 1: "other"}
    assert any("not in the vocabulary" in q["note"] for q in found.qa)


def test_a_section_reaching_outside_the_report_is_cut_with_a_note():
    found = rl.label_by_ranges(_records(3), 3, FakeEngine(
        [{"final": _answer([(0, 9, "narrative")])}]))
    assert found.labels == {0: "narrative", 1: "narrative", 2: "narrative"}
    assert any("7 page(s) outside the pages" in q["note"] for q in found.qa)


def test_a_section_drawn_far_past_the_window_is_clamped_not_walked():
    claims, _e, problems, qa = rl.check_sections(
        _answer([(0, 10 ** 12, "narrative")]), [0, 1, 2])
    assert not problems
    assert {p: [l for l, _r in said] for p, said in claims.items()} == {
        0: ["narrative"], 1: ["narrative"], 2: ["narrative"]}
    assert any(f"{10 ** 12 - 2} page(s) outside" in q["note"] for q in qa)


def test_one_page_covered_twice_with_one_label_is_not_a_fault():
    engine = FakeEngine([{"final": _answer(
        [(0, 4, "narrative"), (4, 7, "narrative")])}])
    found = rl.label_by_ranges(_records(8), 8, engine)
    assert engine.n_calls == 1                     # no re-ask paid for it
    assert found.labels[4] == "narrative" and not found.unresolved


def test_a_page_its_exception_settles_is_not_a_fault():
    engine = FakeEngine([{"final": _answer(
        [(0, 4, "narrative"), (4, 7, "lab_test")], [(4, "figure")])}])
    found = rl.label_by_ranges(_records(8), 8, engine)
    assert engine.n_calls == 1
    assert found.labels[4] == "figure" and not found.unresolved


def test_a_reask_that_leaves_more_pages_unsettled_is_not_taken():
    # The first answer leaves one page uncovered; the second covers it but
    # leaves half the window without a section. One fault KIND each; the
    # first answer leaves far fewer pages unsettled, so it stands.
    first = _answer([(0, 40, "narrative"), (42, 99, "lab_test")])
    worse = _answer([(0, 41, "narrative")])
    engine = FakeEngine([{"final": first}, {"final": worse}])
    found = rl.label_by_ranges(_records(100), 100, engine)
    assert engine.n_calls == 2
    assert found.unresolved == [41]
    assert found.windows[0]["repaired"] is False
    assert found.labels[60] == "lab_test"


def test_a_reask_that_settles_more_pages_is_taken():
    first = _answer([(0, 40, "narrative")])
    better = _answer([(0, 40, "narrative"), (41, 98, "lab_test")])
    found = rl.label_by_ranges(_records(100), 100, FakeEngine(
        [{"final": first}, {"final": better}]))
    assert found.unresolved == [99]
    assert found.windows[0]["repaired"] is True


def test_the_pages_left_unsettled_are_counted_by_page():
    claims = {0: [("narrative", "")], 1: [], 2: [("a", ""), ("b", "")],
              3: [("a", ""), ("a", "")], 4: [], 5: [("a", ""), ("b", "")]}
    assert rl.unsettled(claims, {4: ("figure", ""), 5: ("a", "")}) == [1, 2]


def test_no_answer_at_all_is_asked_once_more_then_unresolved():
    engine = FakeEngine([{"text": "prose"}, {"text": "more prose"}])
    found = rl.label_by_ranges(_records(3), 3, engine)
    assert engine.n_calls == 2 and found.unresolved == [0, 1, 2]


# -- the labeller: long reports ----------------------------------------------------

@pytest.mark.parametrize("n, window, overlap", [
    (10, 6, 2), (100, 30, 5), (301, 300, 40), (1000, 300, 40), (7, 10, 3),
])
def test_windows_cover_every_page_overlap_and_stay_under_the_size(
        n, window, overlap):
    windows = rl.balanced_windows(list(range(n)), window, overlap)
    assert sorted({p for w in windows for p in w}) == list(range(n))
    assert all(len(w) <= window for w in windows)
    for a, b in zip(windows, windows[1:]):
        assert len(set(a) & set(b)) >= overlap
    if n <= window:
        assert windows == [list(range(n))]


def test_a_long_report_is_stitched_from_its_windows_each_page_once():
    # Ten pages in two windows, [0..5] and [4..9]; they disagree on 4 and 5.
    engine = FakeEngine([
        {"final": _answer([(0, 5, "narrative")])},
        {"final": _answer([(4, 9, "lab_test")])}])
    found = rl.label_by_ranges(_records(10), 10, engine, window=6,
                               overlap=2)
    assert engine.n_calls == 2
    first = engine.calls[0]["messages"][0]["content"][0]["text"]
    second = engine.calls[1]["messages"][0]["content"][0]["text"]
    assert "pages 0 to 5 only" in first and "[6]" not in first
    assert "pages 4 to 9 only" in second and "[3]" not in second
    # Each overlap page takes the window it is further inside: page 4 is
    # one from the first window's end and none from the second's start.
    assert found.labels[4] == "narrative"
    assert found.labels[5] == "lab_test"
    assert sorted(found.labels) == list(range(10))
    assert any("different labels" in q["note"] for q in found.qa)


def test_a_window_cut_at_the_output_cap_is_halved_and_asked_again(
        monkeypatch):
    monkeypatch.setattr(rl, "MIN_WINDOW", 2)
    engine = FakeEngine([
        {"text": "", "stop_reason": "max_tokens"},
        {"final": _answer([(0, 4, "narrative")])},
        {"final": _answer([(3, 7, "narrative")])}])
    found = rl.label_by_ranges(_records(8), 8, engine, window=8, overlap=2)
    assert engine.n_calls == 3
    assert sorted(found.labels) == list(range(8))
    assert any("output cap" in q["note"] for q in found.qa)


def test_a_cut_window_too_short_to_halve_is_unresolved(monkeypatch):
    monkeypatch.setattr(rl, "MIN_WINDOW", 60)
    engine = FakeEngine([{"text": "", "stop_reason": "max_tokens"}])
    found = rl.label_by_ranges(_records(8), 8, engine)
    assert found.unresolved == list(range(8))


# -- the vote --------------------------------------------------------------------

@pytest.mark.parametrize("policy", POLICIES)
def test_with_no_range_voter_the_vote_is_what_it_was(policy):
    for rules, vision in (("narrative", "figure"), ("boring_log", None),
                          ("lab_test", "lab_test"), ("", "plan")):
        before = combine(Voter("rules", rules, 0.6),
                         Voter("vision", vision, 0.7) if vision else None,
                         policy=policy, page=3)
        after = combine(Voter("rules", rules, 0.6),
                        Voter("vision", vision, 0.7) if vision else None,
                        policy=policy, page=3, ranges=None)
        assert after == before


@pytest.mark.parametrize("policy", [p for p in POLICIES if p != "rules"])
def test_the_range_label_settles_the_page_where_it_spoke(policy):
    choice = combine(Voter("rules", "narrative", 0.9),
                     Voter("vision", "narrative", 0.9), policy=policy,
                     ranges=Voter("ranges", "other"))
    assert choice.label == "other"
    assert choice.agreed is False                  # so the review sees it
    assert [v.name for v in choice.voters] == ["rules", "vision", "ranges"]
    assert choice.why.startswith("ranges:")


def test_the_range_voter_is_ignored_under_the_rules_policy():
    choice = combine(Voter("rules", "narrative", 0.9), policy="rules",
                     ranges=Voter("ranges", "other"))
    assert choice.label == "narrative"


def test_a_range_voter_that_agrees_keeps_the_page_agreed():
    choice = combine(Voter("rules", "lab_test", 0.9),
                     Voter("vision", "lab_test", 0.9),
                     ranges=Voter("ranges", "lab_test"))
    assert choice.label == "lab_test" and choice.agreed is True


# -- the graph -------------------------------------------------------------------

class _ByFormat(FakeEngine):
    """Answers by what is asked for: a description, sections, or the vision
    pass refused (so the test sees the range voter alone beside the rules)."""

    def __init__(self, n_pages, label="other"):
        super().__init__([])
        self.n_pages = n_pages
        self.label = label

    def complete(self, messages, **kwargs):
        wanted = getattr(kwargs.get("output_format"), "__name__", "")
        if wanted == "PageDescription":
            self.turns.append({"final": _description()})
        elif wanted == "RangeAnswer":
            self.turns.append({"final": _answer(
                [(0, self.n_pages - 1, self.label)])})
        else:
            self.turns.append({"raise": RuntimeError("no vision here")})
        return super().complete(messages, **kwargs)


def _vote(doc, engine, out, arm, resume=True, policy="structural"):
    from planlens.document.roles import document_outline, page_roles

    import report_ingest.graph as graph

    labelling = graph._Labelling(policy=policy, range_arm=arm,
                                 vision_engine=engine)
    qa = []
    choices, _vision, _templates = graph._vote_labels(
        doc, engine, graph.Budgets(), str(out), resume, page_roles(doc),
        labelling, graph._Spend(), qa, document_outline(doc))
    return choices, labelling, qa


def test_the_switch_off_makes_no_call_and_leaves_no_file(synthetic,
                                                         tmp_path):
    engine = _ByFormat(synthetic.n_pages)
    off, _labelling, _qa = _vote(synthetic, engine, tmp_path, None)
    asked = [c["output_format"] for c in engine.calls]
    assert rl.PageDescription not in asked and rl.RangeAnswer not in asked
    assert not (tmp_path / "descriptions").exists()
    assert not list(tmp_path.glob("ranges-*.json"))
    assert all("ranges" not in [v.name for v in c.voters] for c in off)


def test_the_switch_on_labels_every_page_and_a_resume_pays_nothing(
        synthetic, tmp_path):
    n = synthetic.n_pages
    engine = _ByFormat(n, label="other")
    choices, labelling, qa = _vote(synthetic, engine, tmp_path, "plain")
    assert [c.label for c in choices] == ["other"] * n
    described = [c for c in engine.calls
                 if c["output_format"] is rl.PageDescription]
    assert len(described) == n
    assert len(os.listdir(tmp_path / "descriptions")) == n
    assert (tmp_path / "ranges-plain.json").is_file()
    assert labelling.range_paid is True

    again = _ByFormat(n, label="narrative")
    resumed, labelling, _qa = _vote(synthetic, again, tmp_path, "plain")
    assert [c.label for c in resumed] == ["other"] * n
    assert not [c for c in again.calls
                if c["output_format"] in (rl.PageDescription,
                                          rl.RangeAnswer)]
    assert labelling.range_paid is False


def test_the_hinted_arm_is_its_own_file_over_the_same_descriptions(
        synthetic, tmp_path):
    n = synthetic.n_pages
    _vote(synthetic, _ByFormat(n), tmp_path, "plain")
    engine = _ByFormat(n)
    _vote(synthetic, engine, tmp_path, "hinted")
    asked = [c["output_format"] for c in engine.calls]
    assert rl.PageDescription not in asked
    assert asked.count(rl.RangeAnswer) == 1
    assert (tmp_path / "ranges-hinted.json").is_file()


def test_the_rules_policy_does_not_run_the_range_voter(synthetic, tmp_path):
    engine = _ByFormat(synthetic.n_pages)
    _choices, _labelling, qa = _vote(synthetic, engine, tmp_path, "plain",
                                     policy="rules")
    assert engine.n_calls == 0
    assert any(q.where == "labels.ranges" for q in qa)


# -- what a wrong label costs ------------------------------------------------------

def test_each_label_goes_to_the_reader_the_graph_sends_it_to():
    assert reader_for("boring_log") == reader_for("test_pit_log") == "log"
    assert reader_for("cover") == reader_for("toc") == "narrative"
    assert reader_for("appended_report") == "bound"
    assert reader_for("photos") is None and reader_for("other") is None


@pytest.mark.parametrize("hand, given, what", [
    ("boring_log", "test_pit_log", "same reader"),
    ("figure", "photos", "same (neither read)"),
    ("lab_test", "boring_log", "different reader"),
    ("dcp_log", "figure", "should be read, is not"),
    ("other", "lab_test", "read, should not be"),
    ("appended_report", "narrative", "bound/parent swap"),
    ("appended_report", "plan", "bound page not read"),
    ("plan", "appended_report", "page read as bound"),
])
def test_every_kind_of_miss_is_named(hand, given, what):
    assert consequence(hand, given) == what
    assert is_harmful(hand, given) is (what not in ("same reader",
                                                    "same (neither read)"))


def test_the_structure_of_the_misses_is_read_off_the_hand_runs():
    hand = {0: "narrative", 1: "narrative", 2: "narrative", 3: "narrative",
            4: "figure", 5: "lab_test", 6: "lab_test", 7: "plan",
            8: "plan"}
    given = {0: "cover", 1: "narrative", 2: "figure", 3: "narrative",
             4: "photos", 5: "lab_test", 6: "lab_test", 7: "other",
             8: "other"}
    assert miss_structure(hand, given) == {
        0: "edge, 1 page", 2: "inside, 1 page", 4: "one-page section",
        7: "whole section wrong", 8: "whole section wrong"}
    scored = harm_score(hand, given)
    assert scored["wrong"] == 5 and scored["harmful"] == 1     # page 2 only
    assert scored["harmful_moves"] == {"narrative -> figure": 1}


def test_a_page_with_no_label_counts_as_other():
    scored = harm_score({0: "lab_test", 1: "other"}, {})
    assert scored["right"] == 1 and scored["harmful"] == 1


@pytest.mark.parametrize("given, kept", [
    ('["B-1", "S-2"]', ["B-1", "S-2"]),
    ("B-1, S-2; Figure 3", ["B-1", "S-2", "Figure 3"]),
    ("B-1\nS-2", ["B-1", "S-2"]),
    ("", []),
    (None, []),
])
def test_identifiers_sent_as_one_string_are_read_not_refused(given, kept):
    assert _description(identifiers=given).identifiers == kept


def test_a_null_field_is_read_as_empty_not_refused():
    answer = rl.PageDescription(**{
        "title": None, "kind": "report text", "names": None,
        "identifiers": None, "continues": None, "header_footer": None,
        "issuer_date": None, "language": "English", "summary": None})
    assert answer.title == "" and answer.issuer_date == ""
    assert answer.identifiers == []
    mixed = _description(identifiers=["B-1", 2, None])
    assert mixed.identifiers == ["B-1", "2"]


def test_the_issuer_and_date_reach_the_labeller_and_its_instructions():
    record = _records(1, issuer_date="Made-Up Consultants, Job 123, 2001")[0]
    line = rl.description_line(0, record)
    assert "issued: Made-Up Consultants, Job 123, 2001" in line
    assert "'issued'" in rl.labeller_system()
    assert "issuer_date" in rl.DESCRIBE_SYSTEM
    assert rl.DESCRIBE_PROMPT_VERSION == "ri73-2"


def test_a_page_that_needed_a_second_call_says_why(synthetic, tmp_path):
    engine = FakeEngine([{"text": "prose, no tool call"},
                         {"final": _description()}])
    found = rl.describe_pages(synthetic, engine, pages=[0],
                              folder=str(tmp_path))
    (why,) = found.pages[0]["retried"]
    assert "no answer through the answer tool" in why
    saved = json.loads((tmp_path / "p0000.json").read_text(encoding="utf-8"))
    assert saved["retried"] == [why]


def test_a_page_described_first_time_carries_no_retry_note(synthetic,
                                                           tmp_path):
    found = rl.describe_pages(synthetic, _describer(1), pages=[0],
                              folder=str(tmp_path))
    assert "retried" not in found.pages[0]


def test_a_passed_deadline_sends_no_page_and_says_so(synthetic, tmp_path):
    import time

    engine = _describer(0)
    found = rl.describe_pages(synthetic, engine, pages=[0, 1],
                              folder=str(tmp_path), deadline=time.time() - 1)
    assert engine.n_calls == 0
    assert found.stopped.startswith("deadline")


def test_a_missing_field_is_not_filled_in():
    from pydantic import ValidationError

    answer = _description().model_dump()
    del answer["kind"]
    with pytest.raises(ValidationError):
        rl.PageDescription(**answer)


def test_a_ceiling_on_the_describers_calls_stops_the_pass(synthetic,
                                                          tmp_path):
    engine = _describer(5)
    found = rl.describe_pages(synthetic, engine, pages=[0, 1, 2, 3],
                              folder=str(tmp_path), max_calls=2)
    assert engine.n_calls == 2 and sorted(found.pages) == [0, 1]
    assert "ceiling of 2" in found.stopped
    assert not found.unresolved                    # left for later, unasked
    again = rl.describe_pages(synthetic, _describer(2), pages=[0, 1, 2, 3],
                              folder=str(tmp_path))
    assert again.reused == 2 and sorted(again.pages) == [0, 1, 2, 3]


def test_progress_is_told_of_each_page_saved(synthetic, tmp_path):
    seen = []
    rl.describe_pages(synthetic, _describer(3), pages=[0, 1, 2],
                      folder=str(tmp_path), workers=2,
                      progress=lambda done, wanted: seen.append(
                          (done, wanted)))
    assert sorted(seen) == [(1, 3), (2, 3), (3, 3)]


def test_the_describer_defaults_to_the_vision_engine(synthetic, tmp_path):
    from planlens.document.roles import document_outline, page_roles

    import report_ingest.graph as graph

    n = synthetic.n_pages
    vision, main = _ByFormat(n), _ByFormat(n)
    labelling = graph._Labelling(range_arm="plain", vision_engine=vision)
    graph._vote_labels(synthetic, main, graph.Budgets(), str(tmp_path), True,
                       page_roles(synthetic), labelling, graph._Spend(), [],
                       document_outline(synthetic))
    assert [c for c in vision.calls
            if c["output_format"] is rl.PageDescription]
    assert not [c for c in main.calls
                if c["output_format"] is rl.PageDescription]
    assert [c for c in main.calls if c["output_format"] is rl.RangeAnswer]


def test_saved_range_labels_from_other_instructions_are_not_used(
        synthetic, tmp_path):
    n = synthetic.n_pages
    _vote(synthetic, _ByFormat(n, label="other"), tmp_path, "plain")
    path = tmp_path / "ranges-plain.json"
    blob = json.loads(path.read_text(encoding="utf-8"))
    for key in ("describer", "labeller", "describe_prompt", "conventions",
                "labeller_prompt"):
        assert blob[key]
    blob["labeller_prompt"] = "made-up-old"
    path.write_text(json.dumps(blob), encoding="utf-8")
    engine = _ByFormat(n, label="narrative")
    choices, _labelling, qa = _vote(synthetic, engine, tmp_path, "plain")
    assert [c.label for c in choices] == ["narrative"] * n
    assert not [c for c in engine.calls
                if c["output_format"] is rl.PageDescription]
    assert any("not used" in q.detail for q in qa)


def test_a_describer_stopped_by_its_budget_labels_nothing(synthetic,
                                                          tmp_path):
    from planlens.document.roles import document_outline, page_roles

    import report_ingest.graph as graph

    engine = _ByFormat(synthetic.n_pages)
    labelling = graph._Labelling(range_arm="plain", vision_engine=engine)
    qa = []
    choices, _v, _t = graph._vote_labels(
        synthetic, engine, graph.Budgets(range_describe_calls=1),
        str(tmp_path), True, page_roles(synthetic), labelling,
        graph._Spend(), qa, document_outline(synthetic))
    asked = [c["output_format"] for c in engine.calls]
    assert asked.count(rl.PageDescription) == 1
    assert rl.RangeAnswer not in asked
    assert not (tmp_path / "ranges-plain.json").exists()
    assert all("ranges" not in [v.name for v in c.voters] for c in choices)
    assert any(q.kind == "partial" and "did not label" in q.detail
               for q in qa)


def test_a_saved_description_with_no_kind_and_no_summary_is_asked_again(
        synthetic, tmp_path):
    rl.describe_pages(synthetic, _describer(2), pages=[0, 1],
                      folder=str(tmp_path))
    path = tmp_path / "p0001.json"
    blob = json.loads(path.read_text(encoding="utf-8"))
    blob["description"] = {key: "" for key in blob["description"]}
    path.write_text(json.dumps(blob), encoding="utf-8")
    engine = _describer(1)
    again = rl.describe_pages(synthetic, engine, pages=[0, 1],
                              folder=str(tmp_path))
    assert engine.n_calls == 1 and again.reused == 1
    assert again.pages[1]["description"]["kind"] == "report text"


# -- conventions B ---------------------------------------------------------------

def test_conventions_are_chosen_by_name_and_a_typo_is_refused():
    assert rl.conventions_named("a") == (rl.LABEL_CONVENTIONS,
                                         rl.LABELS_NOT_USED)
    assert rl.conventions_named("B") == (rl.LABEL_CONVENTIONS_B,
                                         rl.LABELS_NOT_USED_B)
    with pytest.raises(ValueError, match="conventions must be one of"):
        rl.conventions_named("C")


def test_conventions_b_offers_the_divider_and_has_its_own_fingerprint():
    text, not_used = rl.conventions_named("B")
    for hinted in (False, True):
        system = rl.labeller_system(hinted, text, not_used)
        assert "  divider: " in system and text.strip() in system
        assert rl.labeller_fingerprint(hinted, text, not_used) != \
            rl.labeller_fingerprint(hinted)
    assert "labelled photos" in text and "is divider" in text
    assert rl.conventions_version(text) != rl.conventions_version()


def test_the_hint_note_defers_to_whichever_conventions_were_given():
    for name in rl.CONVENTIONS:
        system = rl.labeller_system(True, *rl.conventions_named(name))
        assert "the conventions decide" in system


def test_a_labelling_under_conventions_b_records_them():
    text, not_used = rl.conventions_named("B")
    engine = FakeEngine([{"final": _answer([(0, 1, "divider")])}])
    found = rl.label_by_ranges(_records(2), 2, engine, conventions=text,
                               not_used=not_used)
    assert found.conventions == rl.conventions_version(text)
    assert found.labels == {0: "divider", 1: "divider"}
    assert "  divider: " in engine.calls[0]["system"]


def test_a_label_the_conventions_do_not_offer_is_a_fault_and_a_note():
    answer = _answer([(0, 2, "divider")])
    claims, _e, problems, qa = rl.check_sections(
        answer, [0, 1, 2], rl.LABELS_NOT_USED)
    assert any("divider" in p for p in problems)
    assert any("divider" in q["note"] for q in qa)
    assert {l for said in claims.values() for l, _r in said} == {"other"}
    _c, _e, problems, qa = rl.check_sections(
        answer, [0, 1, 2], rl.LABELS_NOT_USED_B)
    assert not problems and not qa


def test_a_reply_without_the_optional_fields_is_taken():
    answer = _description().model_dump()
    for name in ("issuer_date", "continues", "identifiers", "names"):
        del answer[name]
    taken = rl.PageDescription(**answer)
    assert taken.issuer_date == "" and taken.continues == ""
    assert taken.identifiers == []
