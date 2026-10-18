"""Offline tests of the live smoke harness ($0: a scripted model, no API).

Run from the repo root::

    .venv\\Scripts\\python.exe -m pytest module_work/live_smoke/test_harness.py -q

* spend control: prices, cache token classes, the ledger's refusal, the
  wave cap stopping the next model call;
* the dry run: a scripted model does the bad things on purpose through the
  app's real turn path, and every detector fires (the clean control turn
  raises nothing);
* each detector on synthetic turns (download cards, the mirror, links);
* the session follows app.py: the module calls in each mirrored block of
  ``webapp/app.py`` appear in the same order in ``session.py``.
"""

from __future__ import annotations

import json
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (os.path.dirname(HERE), REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("langchain_core")

from live_smoke import detectors as det  # noqa: E402
from live_smoke import spend  # noqa: E402

TEST_PRICES = {"test-model": {"input": 1.0, "output": 2.0, "cache_read": 0.1,
                              "cache_write": 1.25}}


@pytest.fixture(autouse=True)
def _restore_env():
    keep = {k: os.environ.get(k) for k in ("GEOTECH_DEFAULT_OUTPUT_DIR",)}
    cwd = os.getcwd()
    yield
    os.chdir(cwd)
    for k, v in keep.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ---------------------------------------------------------------------------
# Spend
# ---------------------------------------------------------------------------

def test_prices_split_cache_tokens():
    usage = {"input_tokens": 10_000, "output_tokens": 1_000,
             "input_token_details": {"cache_read": 6_000,
                                     "cache_creation": 2_000}}
    got = spend.cost_of("claude-sonnet-5-5", usage)
    t = got["tokens"]
    assert (t["uncached"], t["cache_read"], t["cache_write_5m"]) == (
        2_000, 6_000, 2_000)
    want = (2_000 * 2.0 + 6_000 * 0.10 + 2_000 * 2.50 + 1_000 * 10.0) / 1e6
    assert got["usd"] == pytest.approx(want)
    assert t["cache_split"] is True


def test_haiku_tiers_switch_above_100k_prompt():
    small = spend.cost_of("claude-haiku-5-5", {"input_tokens": 100_000,
                                               "output_tokens": 0})
    big = spend.cost_of("claude-haiku-5-5", {"input_tokens": 100_001,
                                             "output_tokens": 0})
    assert small["usd"] == pytest.approx(100_000 * 0.10 / 1e6)
    assert big["usd"] == pytest.approx(100_001 * 0.50 / 1e6)


def test_one_hour_cache_writes_and_raw_usage():
    u = {"input_tokens": 1_000, "output_tokens": 0, "input_token_details": {
        "cache_creation": 0, "ephemeral_1h_input_tokens": 1_000}}
    got = spend.cost_of("claude-opus-5-5", u)
    assert got["usd"] == pytest.approx(1_000 * 4.0 * 2.0 / 1e6)
    raw = spend.usage_from_raw({"input_tokens": 10, "cache_read_input_tokens":
                                90, "cache_creation_input_tokens": 0,
                                "output_tokens": 5})
    assert raw["input_tokens"] == 100
    assert spend.split_usage(raw)["cache_read"] == 90


def test_unpriced_model_is_refused():
    with pytest.raises(spend.UnpricedModel):
        spend.rates_for("claude-unknown-9")


def test_ledger_refuses_a_wave_that_could_pass_the_hard_total(tmp_path):
    led = spend.SpendLedger(str(tmp_path / "l.json"), hard_total=80.0)
    led.data["cumulative_usd"] = 79.90
    with pytest.raises(spend.RefuseToStart):
        led.begin_wave("w", 0.20, model="m", mode="flow")
    led.begin_wave("w", 0.10, model="m", mode="flow")       # exactly 80.00
    assert json.loads((tmp_path / "l.json").read_text())["waves"]["w"][
        "cap_usd"] == 0.10


def _priced_model(meter, n_calls_box):
    from langchain_core.messages import AIMessage
    from live_smoke.models import scripted_model

    def policy(_msgs):
        n_calls_box.append(1)
        return AIMessage(content="ok")
    return scripted_model(policy, usage={"input_tokens": 5_000,
                                         "output_tokens": 2_500},
                          callbacks=[meter])


def test_wave_cap_stops_the_next_model_call(tmp_path):
    led = spend.SpendLedger(str(tmp_path / "l.json"))
    led.begin_wave("w", 0.015, model="test-model", mode="flow")
    meter = spend.SpendMeter(led, "w", 0.015, "test-model",
                             prices=TEST_PRICES)
    made = []
    model = _priced_model(meter, made)
    model.invoke("hi")                                 # $0.01
    assert led.wave_spent("w") == pytest.approx(0.01)
    model.invoke("hi")                                 # $0.02 total
    with pytest.raises(spend.WaveCapReached):
        model.invoke("hi")                             # refused, never made
    assert len(made) == 2
    assert meter.stopped and "wave cap reached" in meter.stopped
    data = json.loads((tmp_path / "l.json").read_text())
    assert data["waves"]["w"]["stopped_by_cap"] is True
    assert data["cumulative_usd"] == pytest.approx(0.02)


def test_meter_counts_unmetered_calls(tmp_path):
    from langchain_core.messages import AIMessage
    from live_smoke.models import scripted_model
    led = spend.SpendLedger(str(tmp_path / "l.json"))
    meter = spend.SpendMeter(led, "w", 1.0, "test-model", prices=TEST_PRICES)
    model = scripted_model(lambda m: AIMessage(content="x"), callbacks=[meter])
    model.invoke("hi")
    assert meter.summary()["unmetered"] == 1
    assert led.wave("w")["unmetered_calls"] == 1


# ---------------------------------------------------------------------------
# The dry run: every detector fires on purpose, the control stays clean
# ---------------------------------------------------------------------------

def test_dry_run_fires_every_detector(tmp_path):
    from live_smoke import dryrun
    real_ledger = os.path.join(HERE, "spend_ledger.json")
    before = os.path.getmtime(real_ledger) if os.path.exists(real_ledger) \
        else None
    res = dryrun.run(wave="dry", runs_dir=str(tmp_path / "runs"))
    assert res["missing"] == [], res["fired"]
    assert res["control"] == [], res["control"]
    # the control: save_file asked for /tmp/<name> lands in the conversation
    assert res["redirect"]["ok"], res["redirect"]
    assert res["ok"]
    # $0 and its own ledger: the real one is untouched.
    after = os.path.getmtime(real_ledger) if os.path.exists(real_ledger) \
        else None
    if before != after:
        # Another session's LIVE wave may be writing the real ledger while
        # this test runs; what matters is that the dry run recorded nothing
        # in it (no wave of its name, no scenario of its flow).
        with open(real_ledger, encoding="utf-8") as fh:
            text = fh.read()
        assert "dry" not in (json.loads(text).get("waves") or {})
        assert dryrun.FLOW["id"] not in text
    out = tmp_path / "runs" / "dry"
    assert (out / "dry_ledger.json").is_file()
    assert (out / "REPORT.md").is_file()
    scen = out / dryrun.FLOW["id"]
    for name in ("detectors.json", "transcript.md", "activity.jsonl",
                 "summary.json"):
        assert (scen / name).is_file(), name
    convs = list((scen / "conversations").iterdir())
    assert convs and (convs[0] / "meta.json").is_file()
    # the mirror carried every conversation file (the turn's (d) was clean)
    turn1 = json.loads((scen / "detectors.json").read_text(encoding="utf-8")
                       )["turns"][0]
    assert not [f for f in turn1["findings"]
                if f["code"] in ("file_not_mirrored", "mirror_error",
                                 "mirror_did_not_run")]


# ---------------------------------------------------------------------------
# Detectors on synthetic turns
# ---------------------------------------------------------------------------

def _conv(tmp_path):
    conv = tmp_path / "data" / "conversations" / "abc"
    files = conv / "files"
    files.mkdir(parents=True)
    (conv / "meta.json").write_text('{"title": "t"}', encoding="utf-8")
    return conv, files


def _ctx(rec, **kw):
    return det.turn_context(rec, **kw)


def codes(found):
    return {f["code"] for f in found}


def test_download_cards(tmp_path):
    conv, files = _conv(tmp_path)
    (files / "memo.docx").write_bytes(b"x")
    (files / "plot.png").write_bytes(b"x")
    (files / "plot.plotly.json").write_bytes(b"{}")
    outside = tmp_path / "elsewhere.pdf"
    outside.write_bytes(b"%PDF")
    rec = {"conv_dir": str(conv), "files_dir": str(files), "before": [],
           "after": [str(files / n) for n in ("memo.docx", "plot.png",
                                              "plot.plotly.json")],
           "staged_inputs": [], "final": "done",
           "entry": {"artifacts": [str(files / "plot.plotly.json"),
                                   str(files / "gone.pdf"), str(outside)]},
           "sidebar_after": {"downloads": [{"path": "x", "ok": False}]}}
    found = det.detect_download_cards(_ctx(rec))
    c = codes(found)
    assert {"file_without_card", "card_missing_file",
            "card_outside_conversation", "sidebar_download_unreadable"} <= c
    # the PNG behind an interactive sidecar is not "without a card"
    assert not any(f["code"] == "file_without_card" and
                   f["evidence"]["path"].endswith("plot.png") for f in found)


def test_sharepoint_mirror_and_links(tmp_path):
    from live_smoke.fake_sharepoint import LocalSharePointFM, ROOT
    conv, files = _conv(tmp_path)
    (files / "memo.docx").write_bytes(b"memo")
    fm = LocalSharePointFM(str(tmp_path / "lib"))
    remote = f"{ROOT}/conversations/x_2026-10-08"
    fm.seed(f"{remote}/meta.json", b'{"title": "t"}')        # memo missing
    rec = {"conv_dir": str(conv), "files_dir": str(files), "sp_sync": None,
           "sidebar_after": {"storage": {"web_url": "https:/contoso.share"
                                         "point.com/sites/LiveSmoke/x"}},
           "final": "x", "multi_user": True, "user": "LIVESMOKE\\alice"}
    found = det.detect_sharepoint(_ctx(rec, fm=fm, remote_folder=remote))
    c = codes(found)
    assert {"mirror_did_not_run", "file_not_mirrored",
            "folder_link_malformed", "mirror_not_per_user"} <= c
    # a good link to the right folder passes
    rec["sidebar_after"]["storage"]["web_url"] = fm.get_web_url(remote)
    rec["sp_sync"] = {"errors": []}
    ok = codes(det.detect_sharepoint(_ctx(rec, fm=fm, remote_folder=remote)))
    assert "folder_link_malformed" not in ok
    assert "folder_link_wrong_folder" not in ok


def test_upload_link_that_does_not_resolve(tmp_path):
    from live_smoke.fake_sharepoint import LocalSharePointFM, ROOT
    conv, files = _conv(tmp_path)
    fm = LocalSharePointFM(str(tmp_path / "lib"))
    act = conv / "activity.jsonl"
    act.write_text(json.dumps({"event": "tool_end",
                               "name": "sharepoint_upload_file",
                               "result": f"Uploaded {files}/m.docx -> {ROOT}"
                                         "/conversations/x/files/m.docx. Link:"
                                         " https://contoso.sharepoint.com/"
                                         "sites/LiveSmoke/nope.docx"}) + "\n",
                   encoding="utf-8")
    rec = {"conv_dir": str(conv), "files_dir": str(files), "sp_sync": {},
           "final": "x", "activity_offset": 0}
    c = codes(det.detect_sharepoint(_ctx(rec, fm=fm)))
    assert {"upload_not_in_library", "upload_link_dead"} <= c


def test_files_outside_from_the_watch(tmp_path):
    conv, files = _conv(tmp_path)
    rec = {"conv_dir": str(conv), "files_dir": str(files), "final": "",
           "t_start": 0}
    outside = [{"root": "home_webapp", "path": str(tmp_path / "a.json"),
                "bytes": 1, "change": "new"},
               {"root": "cwd", "path": str(tmp_path / "rel.pdf"), "bytes": 1,
                "change": "new"},
               {"root": "temp", "path": str(tmp_path / "noise.pdf"),
                "bytes": 1, "change": "new"}]
    found = det.detect_files_outside(_ctx(rec, outside=outside))
    c = codes(found)
    assert {"default_data_root_write", "file_in_process_cwd"} <= c
    # an unnamed file in the shared temp folder is not charged to the turn
    assert not any(f["evidence"].get("path", "").endswith("noise.pdf")
                   for f in found)


def test_health_codes():
    rec = {"final": "", "error": "GraphRecursionError: Recursion limit of 50 "
           "reached without hitting a stop condition."}
    assert "step_limit" in codes(det.detect_health(_ctx(rec)))
    rec = {"final": "(no answer text)", "error": None}
    assert "empty_answer" in codes(det.detect_health(_ctx(rec)))
    rec = {"final": "I can't create Excel files here; the tool is not "
           "available.", "error": None}
    assert "answer_says_unavailable" in codes(det.detect_health(_ctx(rec)))


def test_answer_links_accept_tool_urls(tmp_path):
    conv, files = _conv(tmp_path)
    url = "https://contoso.sharepoint.com/sites/LiveSmoke/Shared%20Documents/a.pdf"
    (conv / "activity.jsonl").write_text(json.dumps(
        {"event": "tool_end", "name": "sharepoint_upload_file",
         "result": f"Uploaded x -> y. Link: {url}"}) + "\n", encoding="utf-8")
    rec = {"conv_dir": str(conv), "files_dir": str(files),
           "final": f"Here is the link: {url}.", "entry": {"artifacts": []},
           "activity_offset": 0}
    assert not codes(det.detect_answer_links(_ctx(rec)))


def test_url_well_formed():
    assert det.url_well_formed("https://a.b/c")[0]
    assert not det.url_well_formed("https:/a.b/c")[0]
    assert not det.url_well_formed("https://a.b/c d")[0]


def test_public_document_guard():
    from live_smoke import docs
    with pytest.raises(docs.NotPublic):
        docs.assert_public(os.path.join(REPO, "module_work", "field_feedback",
                                        "x", "raw", "a.pdf"))
    with pytest.raises(docs.NotPublic):
        docs.assert_public(os.path.join(REPO, "geotech-references", "docs",
                                        "aashto1993.pdf"))
    docs.assert_public(os.path.join(REPO, "geotech-references", "docs",
                                    "ufc_3_220_07.pdf"))


def test_every_flow_is_well_formed():
    from live_smoke import runner
    flows = runner.load_flows()
    assert len(flows) >= 25
    ids = [f["id"] for f in flows]
    assert len(ids) == len(set(ids))
    steps = {"upload", "paste", "say", "new_conversation", "open", "restart",
             "restore"}
    pages = {"document_review", "geotech"}
    for f in flows:
        assert f.get("page", "document_review") in pages, f["id"]
        assert f.get("cost", "medium") in ("light", "medium", "heavy")
        for st in f["steps"]:
            assert steps & set(st), (f["id"], st)
        for seed in f.get("sharepoint_seed") or []:
            assert "doc" in seed
        for spec in (f.get("expect") or {}).get("cards") or []:
            spec = spec if isinstance(spec, dict) else {"kind": spec}
            assert set(spec) <= {"kind", "pattern", "count", "min", "max",
                                 "looked_at"}, (f["id"], spec)
            assert spec.get("kind") or spec.get("pattern"), f["id"]
            if spec.get("pattern"):
                re.compile(spec["pattern"])
            for k in ("count", "min", "max"):
                assert isinstance(spec.get(k, 0), int), (f["id"], spec)
            assert det.card_spec_bounds(spec)[1] >= 0
        assert all(isinstance(c, str) and c
                   for c in f.get("reader_checks") or []), f["id"]
        for item in (f.get("expect") or {}).get("answer_names") or []:
            label, patterns = det._answer_name(item)
            assert label and patterns, (f["id"], item)
            for p in patterns:
                re.compile(p)
    covered = {p for f in flows for p in [f.get("page", "document_review")]}
    assert covered == pages


def test_judgment_arms_pair_with_their_flows():
    """EV-26: each j-arm is its flow with GEOTECH_JUDGMENT_CALLS on and one
    more answer key, the "Judgment calls" list; the reader checks are the
    same, so the trace read asks both arms the same questions."""
    from live_smoke import runner
    flows = {f["id"]: f for f in runner.load_flows()}
    arms = [i for i in flows if re.match(r"F\d+j-", i)]
    assert {a.split("-")[0] for a in arms} >= {"F18j", "F51j", "F55j",
                                                "F61j"}
    for a in arms:
        num = re.match(r"F\d+", a).group(0)
        base, arm = flows[num + a[len(num) + 1:]], flows[a]
        assert "GEOTECH_JUDGMENT_CALLS" not in (base.get("env") or {}), a
        assert arm["env"] == dict(base.get("env") or {},
                                  GEOTECH_JUDGMENT_CALLS="1"), a
        for k in ("page", "cost", "steps", "reader_checks"):
            assert arm.get(k) == base.get(k), (a, k)
        be, ae = dict(base["expect"]), dict(arm["expect"])
        names, arm_names = be.pop("answer_names"), ae.pop("answer_names")
        assert ae == be, a
        assert arm_names[:-1] == names and len(names) >= 3, a
        assert det._answer_name(arm_names[-1]) == (
            'a "Judgment calls" list', ["judge?ment calls?"])
        assert base.get("reader_checks"), a


# ---------------------------------------------------------------------------
# The session follows app.py
# ---------------------------------------------------------------------------

_CALL = re.compile(r"(?<![\w.])(core|turn_jobs|profiles)\.(\w+)\(")


def _calls(text):
    return [f"{m.group(1)}.{m.group(2)}" for m in _CALL.finditer(text)]


def _block(text, start, end):
    i = text.index(start)
    j = text.index(end, i + len(start))
    return text[i:j]


def _func(text, name, indent=""):
    i = text.index(f"\n{indent}def {name}(")
    m = re.search(rf"\n{indent}(?:def |class |@|# ---)", text[i + 5:])
    return text[i: i + 5 + (m.start() if m else len(text))]


APP_VS_SESSION = [
    # (app.py function, session.py method)
    ("_build_agent_for_session", "_build_agent_for_session"),
    ("_new_conversation", "new_conversation"),
    ("_open_conversation", "open_conversation"),
    ("_stage_files", "_stage_files"),
    ("_queue_orientation", "_queue_orientation"),
    ("_follow_turn_job", "_follow_turn_job"),
]


@pytest.mark.parametrize("app_fn,sess_fn", APP_VS_SESSION)
def test_session_functions_call_what_app_py_calls(app_fn, sess_fn):
    app = open(os.path.join(REPO, "webapp", "app.py"), encoding="utf-8").read()
    ses = open(os.path.join(HERE, "session.py"), encoding="utf-8").read()
    assert _calls(_func(ses, sess_fn, "    ")) == _calls(_func(app, app_fn)), \
        f"session.{sess_fn} no longer follows app.py {app_fn}"


def test_turn_block_follows_app_py():
    app = open(os.path.join(REPO, "webapp", "app.py"), encoding="utf-8").read()
    ses = open(os.path.join(HERE, "session.py"), encoding="utf-8").read()
    app_turn = _block(app, "\nif prompt:\n", "_follow_turn_job(job)")
    ses_turn = _block(ses, "    def _send(", "self._follow_turn_job(job, rec)")
    assert _calls(ses_turn) == _calls(app_turn)
    app_orient = _block(app, '_orient = ss.pop("pending_orientation", None)',
                        "\nif prompt:\n")
    ses_orient = _block(ses, "_orient = ss.pending_orientation",
                        "if not prompt:\n            return []")
    assert _calls(ses_orient) == _calls(app_orient)
    app_paste = _block(app, "_chat_up = list(", "prompt = _chat_text.strip()")
    ses_paste = _block(ses, "    def paste(", "return self._rerun(prompt=(text")
    assert [c for c in _calls(ses_paste) if c.startswith("core.")] == \
        [c for c in _calls(app_paste) if c.startswith("core.")]


def test_upload_beside_the_mirrored_copy_is_flagged(tmp_path):
    """Live sanity 2026-10-08 (F19): 'save the memo to SharePoint' put a
    timestamped second copy next to the one the mirror already keeps."""
    from live_smoke.fake_sharepoint import LocalSharePointFM, ROOT
    conv, files = _conv(tmp_path)
    fm = LocalSharePointFM(str(tmp_path / "lib"))
    folder = f"{ROOT}/conversations/tester/x_2026-10-08/files"
    fm.seed(f"{folder}/memo.docx", b"v1")
    fm.seed(f"{folder}/memo_20261008_212756.docx", b"v1")
    link = fm.get_web_url(f"{folder}/memo_20261008_212756.docx")
    (conv / "activity.jsonl").write_text(json.dumps(
        {"event": "tool_end", "name": "sharepoint_upload_file",
         "result": f"Uploaded {files}/memo.docx -> {folder}/memo_20261008_"
                   f"212756.docx (this conversation's SharePoint folder). "
                   f"Link: {link}"}) + "\n", encoding="utf-8")
    rec = {"conv_dir": str(conv), "files_dir": str(files), "sp_sync": {},
           "final": "x", "activity_offset": 0}
    c = codes(det.detect_sharepoint(_ctx(rec, fm=fm)))
    assert "upload_duplicates_mirrored_file" in c
    assert "upload_link_dead" not in c and "upload_not_in_library" not in c


class _Sess:
    def __init__(self, turns):
        self.turns = turns


def test_answer_names_read_the_last_answer():
    """EV-26: a flow lists what its final answer must name (a phrase, a
    regex, or {name, any}); each one missing is a finding."""
    flow = {"expect": {"answer_names": [
        "factor of safety",
        {"name": "the search", "any": ["non-?circular", r"\bblock\b"]},
        {"name": "what would change it", "any": ["would change"]},
        "judge?ment calls", "[unclosed"]}}
    early = {"final": "nothing here", "t_end": 1.0}
    last = {"final": "Factor of safety 0.9 on a non‑circular surface. "
                     "Judgement calls: a wetter clay [unclosed bracket.",
            "t_end": 2.0}
    found = det.detect_expectations(flow, [_Sess([last]), _Sess([early])])
    assert [f["code"] for f in found] == ["expected_answer_name_missing"]
    assert found[0]["severity"] == "info"
    assert "what would change it" in found[0]["message"]
    assert found[0]["evidence"]["looked_for"] == ["would change"]
    # an empty answer misses every name, and says it was empty
    found = det.detect_expectations(flow, [_Sess([{"final": "",
                                                   "t_end": 3.0}])])
    assert len(found) == 5
    assert all("(the answer is empty)" in f["message"] for f in found)
    # no answer_names: nothing new
    assert det.detect_expectations({"expect": {}}, [_Sess([last])]) == []


def test_judgment_call_flows_run_both_arms():
    """EV-25's flow runs with GEOTECH_JUDGMENT_CALLS off and on, with the
    same question and the same checks (the ON arm also looks for the
    list)."""
    from live_smoke import runner
    flows = {f["id"]: f for f in runner.load_flows()}
    off = flows["F61-geotech-slope-weak-layer"]
    on = flows["F61j-geotech-slope-weak-layer"]
    assert "GEOTECH_JUDGMENT_CALLS" not in (off.get("env") or {})
    assert on["env"] == {"GEOTECH_JUDGMENT_CALLS": "1"}
    assert off["steps"] == on["steps"] and off["page"] == on["page"] == \
        "geotech"
    names_off = off["expect"]["answer_names"]
    assert on["expect"]["answer_names"][:len(names_off)] == names_off
    for f in (off, on):
        for item in f["expect"]["answer_names"]:
            for p in item["any"]:
                re.compile(p)


#: Seven phrasings of an answer to F61 and what each should be found to
#: leave out (the GM-116 review; four were mis-scored by the first regexes).
_F61_ANSWERS = [
    ("The automatic circular search gives FS = 6.3, but it misses the weak "
     "clay. A non‑circular (block) surface along the base of the "
     "residual clay gives FS ≈ 0.9 (Spencer). A rise in groundwater "
     "would lower it further.\n\nJudgment calls\n- Search type: noncircular "
     "...", set()),
    ("Bishop's method with a slip-circle search returns 6.3 for deep circles "
     "through the sandstone. The critical mechanism is a shallow "
     "translational slide confined to the residual clay on the upper face: "
     "FS = 0.7. Confidence: low; the result is heavily influenced by the "
     "assumed c′ of the clay.", {"list"}),
    ("Critical surface: a wedge daylighting on the face at the "
     "clay–sandstone contact (el. 6.5); FS = 0.93. A circular search "
     "alone would report 6.3. Lower φ′ in the residual clay would "
     "make it worse. One judgment call: I used Spencer.", set()),
    ("For the 10 m high 1H:1V cut (crest 10 m above the toe), Bishop's "
     "circular search gives FS = 6.3, with the critical circle passing "
     "through the toe. The slope is stable. If the groundwater rose the "
     "factor of safety would drop.", {"weak", "list"}),
    ("A noncircular search (Spencer) gives FS = 0.94 on a surface within the "
     "weak clay layer. The FS hinges on the clay's strength.",
     {"circular", "list"}),
    ("The deep-seated circle through the rock gives 6.3, but a shallow "
     "failure in the clay cap governs at FS 0.94. If the residual clay is "
     "weaker still, FS falls below 0.9.", {"list"}),
    ("Circular search: FS = 6.3. The sandstone outcrops on the lower face "
     "and the rock surface is at el. 6.5. No further comment.",
     {"weak", "change", "list"}),
]


@pytest.mark.parametrize("answer, missing", _F61_ANSWERS)
def test_f61_answer_names_score_seven_phrasings(answer, missing):
    from live_smoke import runner
    flow = {f["id"]: f for f in runner.load_flows()}[
        "F61j-geotech-slope-weak-layer"]
    key = {"the circular search": "circular",
           "a surface in or along the weak layer": "weak",
           "what would change the answer": "change",
           'a "Judgment calls" list': "list"}
    # the four keys these phrasings were written for (w11 added two more)
    names = [n for n in flow["expect"]["answer_names"] if n["name"] in key]
    assert len(names) == 4
    found = det.detect_expectations(
        {"expect": {"answer_names": names}},
        [_Sess([{"final": answer, "t_end": 1.0}])])
    got = {key[f["message"].replace("the final answer does not name ", "")]
           for f in found}
    assert got == missing


def test_f61_change_names_phi_written_as_a_symbol():
    from live_smoke import runner
    flow = {f["id"]: f for f in runner.load_flows()}[
        "F61-geotech-slope-weak-layer"]
    found = det.detect_expectations({"expect": {
        "answer_names": flow["expect"]["answer_names"]}}, [_Sess([{
            "final": "A circular search gives 6.3; a block along the base "
                     "of the clay 0.9. If φ′ is lower, so is FS.",
            "t_end": 1.0}])])
    # only w11's key (the method's reason)
    assert {f["message"].replace("the final answer does not name ", "")
            for f in found} == {"the limit-equilibrium method and why"}


def test_f61_w11_key_method_reason():
    """w11 §4: neither arm said why it chose Spencer; a key asks for it.
    The assumed model base is a reader check only (el. -30 m is the tool's
    default, and no wording of it is reliable)."""
    from live_smoke import runner
    flow = {f["id"]: f for f in runner.load_flows()}[
        "F61-geotech-slope-weak-layer"]
    names = [n for n in flow["expect"]["answer_names"]
             if n["name"] == "the limit-equilibrium method and why"]
    assert len(names) == 1
    assert not any("model base" in n["name"]
                   for n in flow["expect"]["answer_names"])
    assert any("model base" in c for c in flow["reader_checks"])

    def missing(text):
        return len(det.detect_expectations(
            {"expect": {"answer_names": names}},
            [_Sess([{"final": text, "t_end": 1.0}])]))
    assert missing("Spencer, because it satisfies both force and moment "
                   "equilibrium.") == 0
    assert missing("I used Spencer's method on the circle. FS = 0.94.") == 1


# ---------------------------------------------------------------------------
# EV-24: figure cards (k), card expectations, the commit, the estimate
# ---------------------------------------------------------------------------

def _at(path, t):
    """Set a file's time on disk (what (k) compares with the log's)."""
    os.utime(path, (t, t))


def _look(rid, key, t, size, *, kind="image", agent="figure"):
    """A look as the activity log records it (w9 F60 [10]-[13]): the
    tool's start, its vision call (tagged with the tool's run id), its end."""
    return [{"event": "tool_start", "name": "analyze_image", "run_id": rid,
             "agent": agent, "ts": t, "args": {"attachment_key": key,
                                               "prompt": "check"}},
            {"event": "model_start", "agent": agent, "ts": t + 0.1,
             "vision": {"kind": kind, "image_bytes": size},
             "tool": "analyze_image", "tool_run_id": rid},
            {"event": "tool_end", "name": "analyze_image", "run_id": rid,
             "agent": agent, "ts": t + 4.0,
             "result": json.dumps({"analysis": "looks fine"})}]


def _draw(rid, tool, name, t, *, look_file=None):
    """A drawing tool's end as w9 recorded it: plot_data names its PNG in
    ``output_path``, plot_code lists ``images`` (GM-107: with a
    ``look_file`` when the report copy is larger than a look takes)."""
    if tool == "plot_data":
        res = {"status": "success", "output_path": name,
               "file_size_bytes": 10}
    else:
        img = {"file": name, "output_path": f"C:\\sb\\files\\{name}"}
        if look_file:
            img["look_file"] = look_file
        res = {"seconds": 0.9, "images": [img]}
    return [{"event": "tool_start", "name": tool, "run_id": rid, "ts": t - 1,
             "args": {}},
            {"event": "tool_end", "name": tool, "run_id": rid, "ts": t,
             "result": json.dumps(res)}]


def _write_acts(conv, acts):
    (conv / "activity.jsonl").write_text(
        "\n".join(json.dumps(a) for a in acts) + "\n", encoding="utf-8")


def _fig_rec(conv, files, cards):
    return {"conv_dir": str(conv), "files_dir": str(files), "final": "x",
            "activity_offset": 0,
            "entry": {"artifacts": [str(files / c) for c in cards]}}


LOOKED = {"cards": [{"kind": "figure", "count": 1, "looked_at": True}]}


def test_stale_chart_card_after_a_redraw_w9_f56(tmp_path):
    """w9 F56: plot_data wrote the chart and its PNG; plot_code redrew the
    PNG 77 s later; the card was still the draft's chart."""
    conv, files = _conv(tmp_path)
    t0 = 1_792_114_530.0
    (files / "gsd.plotly.json").write_text("{}")
    (files / "gsd.png").write_bytes(b"p" * 175)
    _at(files / "gsd.plotly.json", t0)
    _at(files / "gsd.png", t0 + 77)
    _write_acts(conv, _draw("d1", "plot_data", "gsd.png", t0)
                + _look("l1", "gsd.png", t0 + 3, 87)       # the draft
                + _draw("d2", "plot_code", "gsd.png", t0 + 50)
                + _look("l2", "gsd.png", t0 + 52, 171)
                + _draw("d3", "plot_code", "gsd.png", t0 + 77)
                + _look("l3", "gsd.png", t0 + 78, 175))
    rec = _fig_rec(conv, files, ["gsd.plotly.json"])
    found = det.detect_figure_cards(_ctx(rec, expect=LOOKED))
    stale = [f for f in found if f["code"] == "stale_figure_card"]
    assert len(stale) == 1 and stale[0]["severity"] == "high"
    assert stale[0]["evidence"]["newer_shown"] is False
    assert stale[0]["evidence"]["newer_by_s"] == pytest.approx(77, abs=0.5)
    # the chart WAS read: its twin, drawn with it, was looked at before
    # the first redraw
    assert "figure_card_not_looked_at" not in codes(found)
    # a turn whose only looks came after the first redraw read the redraws,
    # never what the chart shows
    _write_acts(conv, _draw("d1", "plot_data", "gsd.png", t0)
                + _draw("d2", "plot_code", "gsd.png", t0 + 50)
                + _look("l2", "gsd.png", t0 + 52, 171)
                + _draw("d3", "plot_code", "gsd.png", t0 + 77)
                + _look("l3", "gsd.png", t0 + 78, 175))
    found = det.detect_figure_cards(_ctx(rec, expect=LOOKED))
    assert "figure_card_not_looked_at" in codes(found)
    # after GM-107 the card is the newer PNG: nothing is stale
    rec = _fig_rec(conv, files, ["gsd.png"])
    found = det.detect_figure_cards(_ctx(rec, expect=LOOKED))
    assert not codes(found)


def test_chart_drawn_in_code_is_not_read_by_a_look_at_its_png(tmp_path):
    """w8 F54: plot_code wrote a Plotly chart and its own matplotlib PNG
    under one name; the looks read the PNG, a different drawing, so the
    chart card was never looked at. Only plot_data draws both from one
    spec."""
    conv, files = _conv(tmp_path)
    t0 = 1_792_104_982.0
    (files / "prof.plotly.json").write_text("{}")
    (files / "prof.png").write_bytes(b"p")
    _at(files / "prof.png", t0 + 0.3)
    _at(files / "prof.plotly.json", t0 + 0.6)
    code = [{"event": "tool_end", "name": "plot_code", "run_id": "d1",
             "ts": t0 + 0.7, "result": json.dumps({"images": [
                 {"file": "prof.png", "output_path": "C:\\sb\\prof.png"}],
                 "files": ["prof.plotly.json", "prof.png"]})}]
    rec = _fig_rec(conv, files, ["prof.plotly.json"])
    _write_acts(conv, code + _look("l1", "prof.png", t0 + 13, 1))
    found = det.detect_figure_cards(_ctx(rec, expect=LOOKED))
    assert codes(found) == {"figure_card_not_looked_at"}
    # the same files written by plot_data: the PNG look reads the chart
    _write_acts(conv, _draw("d1", "plot_data", "prof.png", t0 + 0.7)
                + _look("l1", "prof.png", t0 + 13, 1))
    assert det.detect_figure_cards(_ctx(rec, expect=LOOKED)) == []
    # a writer the record does not name: no look is credited
    _write_acts(conv, _look("l1", "prof.png", t0 + 13, 1))
    assert codes(det.detect_figure_cards(_ctx(rec, expect=LOOKED))) == {
        "figure_card_not_looked_at"}


def test_numbered_siblings_are_not_versions(tmp_path):
    """sheet_1 and sheet_10, B1 and B12 are other figures; sheet_v2 and
    plan_2 are later versions."""
    conv, files = _conv(tmp_path)
    t0 = 1_792_104_000.0
    for name, dt in (("sheet_1.png", 0), ("sheet_10.png", 30),
                     ("B1.png", 0), ("B12.png", 30), ("plan.png", 0),
                     ("plan_2.png", 30), ("sheet_1_v2.png", 40)):
        (files / name).write_bytes(b"x")
        _at(files / name, t0 + dt)
    rec = _fig_rec(conv, files, ["sheet_1.png", "B1.png", "plan.png"])
    found = det.detect_figure_cards(_ctx(rec))
    sup = sorted((os.path.basename(f["evidence"]["card"]),
                  os.path.basename(f["evidence"]["newer"])) for f in found
                 if f["code"] == "superseded_figure_card")
    assert sup == [("plan.png", "plan_2.png"),
                   ("sheet_1.png", "sheet_1_v2.png")]


def test_picture_and_chart_of_one_draw_are_not_stale(tmp_path):
    conv, files = _conv(tmp_path)
    t0 = 1_792_114_600.0
    (files / "plan.plotly.json").write_text("{}")
    (files / "plan.png").write_bytes(b"x" * 50)
    _at(files / "plan.plotly.json", t0 + 0.4)          # one call writes both
    _at(files / "plan.png", t0)
    _write_acts(conv, _draw("d1", "plot_data", "plan.png", t0 + 0.4)
                + _look("l1", "plan.png", t0 + 2, 50))
    rec = _fig_rec(conv, files, ["plan.plotly.json"])
    assert det.detect_figure_cards(_ctx(rec, expect=LOOKED)) == []


def test_superseded_draft_card_w9_f57(tmp_path):
    """w9 F57: the draft chart and the final PNG under a longer name were
    both shown."""
    conv, files = _conv(tmp_path)
    t0 = 1_792_114_670.0
    for name in ("plasticity_chart.plotly.json", "plasticity_chart.png"):
        (files / name).write_bytes(b"{}")
        _at(files / name, t0)
    (files / "plasticity_chart_report.png").write_bytes(b"r" * 141)
    _at(files / "plasticity_chart_report.png", t0 + 152)
    (files / "plasticity_chart_notes.png").write_bytes(b"n")   # not a version
    _at(files / "plasticity_chart_notes.png", t0 + 160)
    _write_acts(conv, _draw("d1", "plot_data", "plasticity_chart.png", t0)
                + _look("l1", "plasticity_chart.png", t0 + 2, 2)
                + _draw("d2", "plot_code", "plasticity_chart_report.png",
                        t0 + 152)
                + _look("l2", "plasticity_chart_report.png", t0 + 154, 141))
    both = ["plasticity_chart.plotly.json", "plasticity_chart_report.png"]
    found = det.detect_figure_cards(_ctx(_fig_rec(conv, files, both)))
    sup = [f for f in found if f["code"] == "superseded_figure_card"]
    assert len(sup) == 1 and sup[0]["severity"] == "medium"
    assert sup[0]["evidence"]["newer"].endswith("plasticity_chart_report.png")
    # only the draft shown: high
    found = det.detect_figure_cards(_ctx(_fig_rec(
        conv, files, ["plasticity_chart.plotly.json"])))
    assert [f["severity"] for f in found
            if f["code"] == "superseded_figure_card"] == ["high"]
    # a later version drawn as picture and chart is one later draw (w7b
    # F52h, w8 F55); the shown one is named
    (files / "plasticity_chart_report.plotly.json").write_text("{}")
    _at(files / "plasticity_chart_report.plotly.json", t0 + 152)
    found = det.detect_figure_cards(_ctx(_fig_rec(
        conv, files, ["plasticity_chart.plotly.json",
                      "plasticity_chart_report.plotly.json"])))
    (sup,) = [f for f in found if f["code"] == "superseded_figure_card"]
    assert sup["severity"] == "medium"
    assert sup["evidence"]["newer"].endswith("report.plotly.json")
    assert sup["evidence"]["also"] == ["plasticity_chart_report.png"]
    os.remove(files / "plasticity_chart_report.plotly.json")
    # the card checks of the expectations see two figure cards for one
    flow = {"expect": LOOKED}
    found = det.detect_expectations(flow, [_Sess([_fig_rec(conv, files,
                                                           both)])])
    assert [f["code"] for f in found] == ["unexpected_card_count"]
    assert found[0]["severity"] == "info"
    assert found[0]["evidence"]["matched"] == sorted(both)


def test_look_copy_counts_as_a_look_and_failed_looks_do_not(tmp_path):
    """GM-107: a report PNG larger than a look takes gets a look copy in
    .scratch, and the look reads that copy."""
    conv, files = _conv(tmp_path)
    t0 = 1_792_114_700.0
    (files / "fig.png").write_bytes(b"big" * 100)
    _at(files / "fig.png", t0)
    (files / "memo.pdf").write_bytes(b"%PDF")
    rec = _fig_rec(conv, files, ["fig.png", "memo.pdf"])
    draw = _draw("d1", "plot_code", "fig.png", t0,
                 look_file=".scratch/fig_look.png")
    _write_acts(conv, draw + _look("l1", ".scratch/fig_look.png", t0 + 1, 9))
    assert det.detect_figure_cards(_ctx(rec, expect=LOOKED)) == []
    # no look at all
    _write_acts(conv, draw)
    found = det.detect_figure_cards(_ctx(rec, expect=LOOKED))
    assert codes(found) == {"figure_card_not_looked_at"}
    assert found[0]["evidence"]["card"].endswith("fig.png")
    # the vision probe reads nothing; a look that failed read nothing
    probe = _look("l2", "fig.png", t0 + 1, None, kind="probe")
    probe[0]["name"] = "other_tool"                 # only the probe tags it
    failed = _look("l3", "fig.png", t0 + 2, 300)
    failed[2]["result"] = json.dumps({"error": "image too large"})
    _write_acts(conv, draw + probe + failed)
    assert "figure_card_not_looked_at" in codes(
        det.detect_figure_cards(_ctx(rec, expect=LOOKED)))
    # a look before the last draw read an older figure
    _write_acts(conv, _look("l4", "fig.png", t0 - 30, 12) + draw)
    assert "figure_card_not_looked_at" in codes(
        det.detect_figure_cards(_ctx(rec, expect=LOOKED)))
    # without looked_at in the flow's expectations, no look is asked for
    _write_acts(conv, draw)
    assert det.detect_figure_cards(_ctx(rec)) == []
    assert det.detect_figure_cards(_ctx(rec, expect={"cards": [
        {"kind": "pdf", "looked_at": True}]})) == []


def test_figure_cards_detector_runs_with_the_others(tmp_path):
    assert det.detect_figure_cards in det.DETECTORS
    conv, files = _conv(tmp_path)
    (files / "memo.docx").write_bytes(b"x")
    rec = _fig_rec(conv, files, ["memo.docx"])
    assert det.detect_figure_cards(_ctx(rec)) == []


def test_card_expectations_count_kinds_and_patterns(tmp_path):
    conv, files = _conv(tmp_path)
    for n in ("calc.pdf", "wall.png", "wall_2.png", "data.csv"):
        (files / n).write_bytes(b"x")
    t1 = _fig_rec(conv, files, ["calc.pdf", "wall.png"])
    t2 = _fig_rec(conv, files, ["wall.png", "wall_2.png", "data.csv"])
    flow = {"expect": {"cards": [
        {"kind": "pdf", "count": 1},                       # met
        {"kind": "figure", "max": 1},                      # 2 figures
        {"kind": "chart"},                                 # none
        {"pattern": r"^data\.csv$", "min": 1, "max": 1},   # met
        "docx"]}}                                          # none
    found = det.detect_expectations(flow, [_Sess([t1, t2])])
    got = sorted((f["code"], f["evidence"]["spec"].get("kind")) for f in found)
    assert got == [("expected_cards_missing", "chart"),
                   ("expected_cards_missing", "docx"),
                   ("unexpected_card_count", "figure")]
    assert det.card_spec_bounds({"kind": "figure"}) == ("figure", 1, None)
    assert det.card_spec_bounds({"max": 2})[1:] == (0, 2)
    assert det.card_kinds("a/b/x.plotly.json") >= {"figure", "chart"}
    assert "image" in det.card_kinds("x.PNG") and \
        "figure" not in det.card_kinds("x.pdf")


def test_watch_ignores_claude_codes_own_state_file(tmp_path):
    """w9 F56/F57: both low unattributed_file_in_home findings were
    ~/.claude.json, rewritten by the Claude Code sessions beside the wave."""
    from live_smoke import watch
    home = tmp_path / "home"
    home.mkdir()
    (home / ".claude.json").write_text("{}")
    (home / "report.pdf").write_bytes(b"%PDF")
    roots = {"home": (str(home), 0)}
    before = watch.snapshot(roots)
    for name in (".claude.json", "report.pdf"):
        _at(home / name, os.path.getmtime(home / name) + 10)
    (home / ".claude.json.backup").write_text("{}")
    (home / "new.json").write_text("{}")
    changed = watch.diff(before, watch.snapshot(roots))
    assert sorted(os.path.basename(d["path"]) for d in changed) == [
        "new.json", "report.pdf"]


def test_wave_json_records_the_code(tmp_path):
    from live_smoke import report, runner
    import subprocess
    head = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    w = runner.Wave("cv", mode="flow", model_id="claude-haiku-5-5", cap=0,
                    fake=True, runs_dir=str(tmp_path), verbose=False)
    w.start()
    meta = json.loads((tmp_path / "cv" / "wave.json").read_text("utf-8"))
    app = meta["code"]["app"]
    assert app["commit"] == head and isinstance(app["dirty"], bool)
    assert isinstance(app["changed"], list) and meta["code"]["recorded"]
    assert meta["code_earlier"] == []
    w.finish()
    head_line = (tmp_path / "cv" / "REPORT.md").read_text("utf-8")
    assert f"app `{head[:10]}`" in head_line
    # a wave resumed on other code keeps what the earlier session ran
    meta["code"]["app"]["commit"] = "0" * 40
    (tmp_path / "cv" / "wave.json").write_text(json.dumps(meta), "utf-8")
    w2 = runner.Wave("cv", mode="flow", model_id="claude-haiku-5-5", cap=0,
                     fake=True, runs_dir=str(tmp_path), verbose=False)
    w2.start()
    meta2 = json.loads((tmp_path / "cv" / "wave.json").read_text("utf-8"))
    assert [c["app"]["commit"] for c in meta2["code_earlier"]] == ["0" * 40]
    assert meta2["code"]["app"]["commit"] == head
    w2.finish()
    text = (tmp_path / "cv" / "REPORT.md").read_text("utf-8")
    assert "Earlier session of this wave" in text and "`0000000000`" in text
    assert report.code_line({}) == ""
    assert runner.git_state(str(tmp_path)) == {}         # not a checkout


def test_estimate_from_recorded_runs(tmp_path):
    runs = tmp_path / "runs"

    def summ(wave, sid, usd, model="m", **kw):
        d = runs / wave / sid
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(
            {"id": sid, "model": model, "usd": usd, **kw}), "utf-8")
    summ("w1", "F90-a", 0.30)
    summ("w2", "F90-a", 0.50)
    summ("w2", "F91-b", 0.20)
    summ("w3", "F91-b", 9.0, incomplete=True)            # cut short
    summ("w3", "F92-c", 9.0, model="m (scripted fake)")  # a fake wave
    env = {"GEOTECH_PLOT_CODE": "1"}
    flows = [{"id": "F90-a", "page": "geotech", "cost": "light",
              "steps": [{"say": "x"}]},
             {"id": "F91-b", "page": "geotech", "cost": "light", "env": env,
              "steps": [{"say": "x"}]},
             {"id": "F92-c", "page": "geotech", "cost": "light", "env": env,
              "steps": [{"say": "x"}]},
             {"id": "F93-d", "page": "document_review", "cost": "heavy",
              "steps": [{"say": "x"}]}]
    est = spend.estimate_flows(flows + [{"id": "F90j"}], "m", str(runs))
    usd = {r["id"]: r["usd"] for r in est["rows"]}
    assert usd == {"F90-a": 0.50, "F91-b": 0.20, "F92-c": 0.20,
                   "F93-d": spend.FALLBACK_USD["heavy"], "F90j": 0.50}
    basis = {r["id"]: r["basis"] for r in est["rows"]}
    assert "other arm" in basis["F90j"] and "not in flows.json" in \
        basis["F90j"]
    assert "same page, cost tier and switches" in basis["F92-c"]
    assert est["total_usd"] == pytest.approx(3.90)
    assert est["suggested_cap_usd"] == 6.0               # 5.85 -> 6.00


def test_estimate_picks_what_the_flow_command_runs(capsys):
    """``estimate --ids F61`` prices F61 and F61j, as ``flow --ids F61``
    runs both; an id with no flow yet gets a row of its own."""
    from live_smoke import run as run_cli
    assert run_cli.main(["estimate", "--model", "claude-sonnet-5-5",
                         "--ids", "F61,F99z"]) == 0
    rows = [ln.split()[0] for ln in capsys.readouterr().out.splitlines()]
    assert rows == ["F61-geotech-slope-weak-layer",
                    "F61j-geotech-slope-weak-layer", "F99z", "total"]


def test_recheck_judges_a_recorded_wave_again(tmp_path):
    """The $0 recheck rebuilds each turn from detectors.json and the copied
    conversation (files keep their times) and runs (k) and the flow's
    expectations with the flows given."""
    from live_smoke import dryrun
    scen = tmp_path / "wave" / "F99-figure"
    conv = scen / "conversations" / "main_abcdef12"
    files = conv / "files"
    files.mkdir(parents=True)
    t0 = 1_792_114_530.0
    (files / "g.plotly.json").write_text("{}")
    (files / "g.png").write_bytes(b"p")
    _at(files / "g.plotly.json", t0)
    _at(files / "g.png", t0 + 60)
    acts = [dict(a, turn=1) for a in
            _draw("d1", "plot_data", "g.png", t0)
            + _draw("d2", "plot_code", "g.png", t0 + 60)
            + _look("l1", "g.png", t0 + 61, 1)]
    _write_acts(conv, acts)
    (scen / "detectors.json").write_text(json.dumps({
        "flow": {"id": "F99-figure"},
        "turns": [{"session": "main", "thread_id": "abcdef1234567890",
                   "cards": ["g.plotly.json"], "final": "done"}]}), "utf-8")
    flows = [{"id": "F99-figure", "expect": LOOKED, "steps": []}]
    res = dryrun.recheck(str(tmp_path / "wave"), flows=flows)
    turn = {f["code"] for f in res["F99-figure"]["turns"][0]}
    assert turn == {"stale_figure_card", "figure_card_not_looked_at"}
    assert res["F99-figure"]["scenario"] == []
    # with no expectation of looks, only the stale card
    res = dryrun.recheck(str(tmp_path / "wave"), flows=[
        {"id": "F99-figure", "steps": []}])
    assert {f["code"] for f in res["F99-figure"]["turns"][0]} == {
        "stale_figure_card"}


#: The EV-24 flows: figures a geotechnical engineer asks for in their own
#: words, one of each kind.
EV24_FLOWS = ("F62-geotech-figure-excavation-section",
              "F63-geotech-figure-consolidation-curve",
              "F64-geotech-figure-exploration-plan",
              "F65-geotech-figure-groundwater-record",
              "F66-geotech-figure-pile-options")


def test_ev24_figure_flows_name_no_tool_and_expect_one_looked_card():
    from live_smoke import runner
    flows = {f["id"]: f for f in runner.load_flows()}
    tools = re.compile(r"plot_code|plot_data|matplotlib|plotly|figure "
                       r"helper|render_figures|profile_figure|\btask\b",
                       re.IGNORECASE)
    for fid in EV24_FLOWS + ("F56-geotech-figure-grain-size",
                             "F57-geotech-figure-plasticity-chart",
                             "F60-geotech-figure-settlement-monitoring"):
        f = flows[fid]
        assert f["page"] == "geotech" and f["cost"] == "light"
        assert f["env"] == {"GEOTECH_PLOT_CODE": "1"}
        (step,) = f["steps"]
        assert not tools.search(step["say"]), fid
        assert f["expect"]["cards"] == [{"kind": "figure", "count": 1,
                                         "looked_at": True}]
        assert f["reader_checks"] and all(
            isinstance(c, str) and c for c in f["reader_checks"])


def test_recovered_tool_error_is_low_and_meter_gap_is_flagged(tmp_path):
    conv, files = _conv(tmp_path)
    acts = [{"event": "tool_end", "name": "call_agent",
             "result": json.dumps({"error": "ValueError: bad depth"})},
            {"event": "tool_end", "name": "call_agent",
             "result": json.dumps({"ok": 1})},
            {"event": "model_end"}, {"event": "model_end"}]
    (conv / "activity.jsonl").write_text(
        "\n".join(json.dumps(a) for a in acts) + "\n", encoding="utf-8")
    rec = {"conv_dir": str(conv), "files_dir": str(files),
           "final": "The answer is 42 kPa, computed with the module.",
           "activity_offset": 0}
    found = det.detect_health(_ctx(rec))
    err = [f for f in found if f["code"] == "tool_returned_error"]
    assert err and err[0]["severity"] == "low"
    cost = det.detect_cost(_ctx(rec, meter_calls=[{"usd": 0.0}]))
    assert "meter_activity_mismatch" in codes(cost)
    cost = det.detect_cost(_ctx(rec, meter_calls=[{"usd": 0.0}] * 2))
    assert "meter_activity_mismatch" not in codes(cost)
