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
        gi = (f.get("expect") or {}).get("given_inputs")
        if gi is not None:
            assert set(gi) <= {"package_given", "brief_given",
                               "not_given"}, (f["id"], gi)
            for item in gi.get("not_given") or []:
                assert set(item) <= {"name", "inputs", "any"}, f["id"]
                assert item["name"] and item["any"], (f["id"], item)
                assert all(isinstance(i, str) for i in item.get("inputs")
                           or []), f["id"]
                for p in item["any"]:
                    re.compile(p)
        assert isinstance((f.get("expect") or {}).get(
            "pdf_cites_consult", False), bool), f["id"]
    covered = {p for f in flows for p in [f.get("page", "document_review")]}
    assert covered == pages


#: Expectations only the switch can meet: the brief to the calc helper and
#: the PDF's citations follow the consult the switch asks for (EV-30: w14's
#: F18 OFF was scored on them).
_ON_ONLY = ("pdf_cites_consult",)
_ON_ONLY_GIVEN = ("brief_given",)


def test_judgment_arms_pair_with_their_flows():
    """EV-26: each j-arm is its flow with GEOTECH_JUDGMENT_CALLS on and one
    more answer key, the "Judgment calls" list; the reader checks are the
    same, so the trace read asks both arms the same questions. The keys only
    the switch can meet are on the arm alone (EV-30). The switch is on by
    default since GM-226, so the OFF flow sets it to 0."""
    from live_smoke import runner
    flows = {f["id"]: f for f in runner.load_flows()}
    arms = [i for i in flows if re.match(r"F\d+j-", i)]
    assert {a.split("-")[0] for a in arms} >= {"F18j", "F51j", "F55j",
                                                "F61j"}
    for a in arms:
        num = re.match(r"F\d+", a).group(0)
        base, arm = flows[num + a[len(num) + 1:]], flows[a]
        assert (base.get("env") or {}).get("GEOTECH_JUDGMENT_CALLS")             == "0", a
        assert arm["env"] == dict(base.get("env") or {},
                                  GEOTECH_JUDGMENT_CALLS="1"), a
        for k in ("page", "cost", "steps", "reader_checks"):
            assert arm.get(k) == base.get(k), (a, k)
        be, ae = dict(base["expect"]), dict(arm["expect"])
        names, arm_names = be.pop("answer_names"), ae.pop("answer_names")
        for k in _ON_ONLY:
            assert k not in be, (a, k)
            ae.pop(k, None)
        if "given_inputs" in ae:
            bg = dict(be.get("given_inputs") or {})
            ag = dict(ae["given_inputs"])
            for k in _ON_ONLY_GIVEN:
                assert k not in bg, (a, k)
                ag.pop(k, None)
            ae["given_inputs"] = ag
        assert ae == be, a
        assert arm_names[:-1] == names and len(names) >= 3, a
        assert det._answer_name(arm_names[-1]) == (
            'a "Judgment calls" list', ["judge?ment calls?"])
        assert base.get("reader_checks"), a


# ---------------------------------------------------------------------------
# The session follows app.py
# ---------------------------------------------------------------------------

# helper_models since EV-33: the session had silently lost INF-41's helper
# models, which this pin did not see.
_CALL = re.compile(r"(?<![\w.])(core|turn_jobs|profiles|helper_models)"
                   r"\.(\w+)\(")


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
    assert off["env"] == {"GEOTECH_JUDGMENT_CALLS": "0"}
    assert on["env"] == {"GEOTECH_JUDGMENT_CALLS": "1"}
    assert off["steps"] == on["steps"] and off["page"] == on["page"] == \
        "geotech"
    names_off = off["expect"]["answer_names"]
    assert on["expect"]["answer_names"][:len(names_off)] == names_off
    for f in (off, on):
        for item in f["expect"]["answer_names"]:
            for p in item["any"]:
                re.compile(p)


def test_figure_helper_pairs_set_their_switches():
    """GM-227: the figure helper and plotting code are on by default, so the
    "figure helper off" flows set both to 0 (plotting code brings the
    helper), and their ON twins keep plotting code off as when measured."""
    from live_smoke import runner
    flows = {f["id"]: f for f in runner.load_flows()}
    for num in ("F52", "F53"):
        off, = [f for i, f in flows.items() if i.startswith(num + "-")]
        on, = [f for i, f in flows.items() if i.startswith(num + "h-")]
        assert off["env"] == {"GEOTECH_FIGURE_HELPER": "0",
                              "GEOTECH_PLOT_CODE": "0"}, num
        assert on["env"] == {"GEOTECH_FIGURE_HELPER": "1",
                             "GEOTECH_PLOT_CODE": "0"}, num
        assert off["steps"] == on["steps"], num


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
    # only w11's key (the method's reason) and GM-155's (a reference),
    # the latter looked for in the whole answer, which has no list
    assert {f["message"].replace("the final answer does not name ", "")
            for f in found} == {
        "the limit-equilibrium method and why",
        'a reference cited for a judgment call (it has no "Judgment calls" '
        'list, so the whole answer was searched)'}


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
    # EV-28 (w12 F61j): agreement between methods is confidence, not why
    # the method was chosen; nor is naming the limit-equilibrium family
    assert missing("Moderate to high, because Spencer and Bishop agree to "
                   "within 0.001.") == 1
    assert missing("Spencer's limit-equilibrium analysis gives 0.94.") == 1
    assert missing("Spencer is the rigorous choice here.") == 0


#: The judgment-call pairs (EV-26) and the w12 sentences that hit a key
#: falsely (EV-28): each must now miss it.
_PAIRS = ("F18-geotech-calc-package", "F51-geotech-fem-excavation",
          "F55-geotech-calc-wall-figure", "F61-geotech-slope-weak-layer")
_REFERENCE_KEY = "a reference cited for a judgment call"


def _flow(fid):
    from live_smoke import runner
    return {f["id"]: f for f in runner.load_flows()}[fid]


def _names(fid, name):
    return [n for n in _flow(fid)["expect"]["answer_names"]
            if n["name"] == name]


def _found(names, text):
    return det.detect_expectations(
        {"expect": {"answer_names": names}},
        [_Sess([{"final": text, "t_end": 1.0}])])


def _misses(names, text):
    return sum(f["code"] == "expected_answer_name_missing"
               for f in _found(names, text))


def _listed(text):
    """``text`` as one call in an answer's "Judgment calls" list."""
    return ("q_ult = 1,159 kPa; q_all = 386 kPa.\n\n**Judgment calls**\n"
            f"- {text}\n")


@pytest.mark.parametrize("fid, name, text", [
    ("F18-geotech-calc-package",
     "general or local shear (how dense the sand is)",
     "phi = 32 suits a medium-dense sand; q_all = 386 kPa."),
    ("F55-geotech-calc-wall-figure", "where the Rankine pressure acts",
     "Ka = 0.3073, H = 4.5 m, so Pa = 59.1 kN/m."),
])
def test_ev28_keys_no_longer_hit_falsely(fid, name, text):
    for arm in (fid, fid.replace("-", "j-", 1)):
        names = _names(arm, name)
        assert len(names) == 1, arm
        assert _misses(names, text) == 1, arm
    assert _misses(_names(fid, name), {
        "general or local shear (how dense the sand is)":
            "General shear is assumed for a sand this dense.",
        "where the Rankine pressure acts":
            "Pa acts on the vertical plane through the back of the heel.",
    }[name]) == 0


@pytest.mark.parametrize("fid", _PAIRS)
def test_every_judgment_pair_looks_for_a_reference(fid):
    """GM-155: w12's four ON answers cited no reference for any call. Both
    arms of each pair now look for a cited one, before the j-arm's list
    key. Saying none was found is a reader check, not this key: the ON
    prompt's own words ("with no reference cited") would satisfy it."""
    for arm in (fid, fid.replace("-", "j-", 1)):
        names = _names(arm, _REFERENCE_KEY)
        assert len(names) == 1, arm
        assert names[0].get("within") == "judgment_calls", arm
        for text in ("Vesic (1973) as in GEC-6.", "per UFC 3-220-20",
                     "K0 from Jaky (1944)", "NAVFAC DM7.2 Table 5-3",
                     "FHWA NHI-06-089 Sec. 10.5.5",
                     "Duncan and Wright (2014) recommend Spencer.",
                     "K0 = 1 - sin phi' (Jaky, 1944).",
                     "N-gamma as tabulated (Vesic 1973).",
                     "N-gamma per Vesić (1973)."):
            assert _misses(names, _listed(text)) == 0, (arm, text)
            # with no list, the whole answer is searched, and said so
            found = _found(names, text)
            assert [f["code"] for f in found] == [
                "expected_answer_name_unscoped"], (arm, text)
            assert "so the whole answer was searched" in found[0]["message"]
        for text in ("Vesic, as the module's default.",
                     "Run on 2026-10-18; FS 0.94.",
                     "I used Spencer because it is rigorous.",
                     "Smooth wall: judgment, with no reference cited.",
                     "I assume no reference governs this.",
                     "q_ult = 1990 kPa (FS 3)."):
            assert _misses(names, _listed(text)) == 1, (arm, text)
            assert _misses(names, text) == 1, (arm, text)
        assert any("says so plainly" in c
                   for c in _flow(arm)["reader_checks"]), arm


def test_the_reference_key_looks_inside_the_list_only():
    """w13 §7 fix 13: F18j's key hit "Vesic 1973" in a results heading
    while every call in its list said "no reference basis". The key now
    looks in the "Judgment calls" list only."""
    names = _names("F18j-geotech-calc-package", _REFERENCE_KEY)
    answer = ("## Results (Vesic 1973 factors, GEC-6 Eq. 6-1)\n"
              "q_ult = 1,159 kPa.\n\n"
              "## Judgment calls\n"
              "1. **Vesic as the primary method.** No reference basis beyond "
              "common practice.\n"
              "2. **Gross against net.** Gross headlined.\n")
    found = _found(names, answer)
    assert [f["code"] for f in found] == ["expected_answer_name_missing"]
    assert found[0]["message"].endswith(
        'a reference cited for a judgment call in its "Judgment calls" '
        "list")
    assert _misses(names, answer.replace(
        "No reference basis beyond common practice.",
        "Vesic (1973), as GEC-6 Eq. 6-1 gives it.")) == 0
    # a citation after the list's own heading level ends is outside it
    assert _misses(names, answer + "\n## Sources\nVesic (1973)\n") == 1


@pytest.mark.parametrize("heading", ["## Judgment calls",
                                     "**Judgment calls**"])
def test_a_bold_references_block_after_the_list_is_outside_it(heading):
    """Round 2 review: a line that is only bold text ends the list, so a
    "**References**" block after it is not read as a call's citation; a
    call that opens in bold stays inside."""
    names = _names("F51j-geotech-fem-excavation", _REFERENCE_KEY)
    answer = (f"Wall moment 120 kN·m/m.\n\n{heading}\n"
              "- **Smooth wall contact.** Rests on judgment, no reference "
              "found.\n"
              "- **K0 = 0.658**, the module's default.\n\n"
              "**References**\n"
              "- Jaky (1944); UFC 3-220-20 Ch. 4.\n")
    part = det.answer_section(answer, "judgment_calls")
    assert "Smooth wall contact" in part and "K0 = 0.658" in part
    assert "References" not in part and "Jaky" not in part
    assert _misses(names, answer) == 1
    assert _misses(names, answer.replace("**References**",
                                         "References:")) == 0
    assert _misses(names, answer.replace("**References**\n", "")) == 0


@pytest.mark.parametrize("answer, inside, outside", [
    ("Intro.\n## Judgment calls\n- a\n### One call\n- b\n## Files\n- c",
     ("- a", "### One call", "- b"), ("Intro.", "- c")),
    ("Intro.\n**Judgment calls**\n- a\n\n**Method.** b\n", ("- a", "b"),
     ("Intro.",)),
    ("Intro.\nJudgment calls (yours to accept):\n1. a\n", ("1. a",),
     ("Intro.",)),
    ("Intro.\n### 3. Judgment Call\n- a\n## Next\n- z", ("- a",),
     ("- z", "Intro.")),
])
def test_answer_section_finds_the_list(answer, inside, outside):
    part = det.answer_section(answer, "judgment_calls")
    for t in inside:
        assert t in part, t
    for t in outside:
        assert t not in part, t


def test_answer_section_none_without_a_list():
    assert det.answer_section("No list here; judgment calls are below "
                              "nowhere.", "judgment_calls") is None
    assert det.answer_section("", "judgment_calls") is None


def test_every_within_names_a_known_part():
    from live_smoke import runner
    for f in runner.load_flows():
        for item in (f.get("expect") or {}).get("answer_names") or []:
            if isinstance(item, dict) and "within" in item:
                assert item["within"] in det._SCOPES, (f["id"], item)


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


def _judgment_event(action, unsourced, references_ran=False, mode="retry"):
    return {"turn": 1, "agent": "primary", "event": "judgment_check",
            "where": "answer", "mode": mode, "action": action,
            "has_list": True, "n_calls": 3, "unsourced": unsourced,
            "n_unsourced": len(unsourced), "references_ran": references_ran}


def test_judgment_check_retry_is_seen_and_counted(tmp_path):
    """GM-174: an answer the app sent back for calls with no source is a
    finding a wave can count, and a turn metric."""
    assert det.detect_judgment_check in det.DETECTORS
    conv, files = _conv(tmp_path)
    _write_acts(conv, [
        _judgment_event("fed_back", ["Wall contact. Smooth.",
                                     "Water. None given."]),
        {"turn": 1, "event": "tool_start", "name": "task"},
        _judgment_event("none", [], references_ran=True)])
    ctx = _ctx(_fig_rec(conv, files, []))
    found = det.detect_judgment_check(ctx)
    assert codes(found) == {"judgment_check_retry"}
    assert found[0]["severity"] == "info"
    assert found[0]["detector"] == "l_judgment_check"
    assert "2 of 3" in found[0]["message"]
    assert det.turn_metrics(ctx)["judgment_retries"] == 1
    # the answers it read, so a rerun sees the check ran (GM-181 rerun)
    assert det.turn_metrics(ctx)["judgment_checks"] == 2


def test_judgment_calls_left_unsourced_with_no_consult_are_seen(tmp_path):
    conv, files = _conv(tmp_path)
    _write_acts(conv, [_judgment_event("none", ["Water. None given."],
                                       mode="off")])
    ctx = _ctx(_fig_rec(conv, files, []))
    assert codes(det.detect_judgment_check(ctx)) == {
        "judgment_calls_unsourced"}
    assert det.turn_metrics(ctx)["judgment_retries"] == 0
    # calls whose list names a source: nothing
    _write_acts(conv, [_judgment_event("none", [])])
    assert det.detect_judgment_check(_ctx(_fig_rec(conv, files, []))) == []


def test_judgment_calls_unsourced_after_a_consult_are_seen(tmp_path):
    """EV-30 (w14 F18j: 2 of 5 calls went out with no source after the
    consult, recorded only as the check's numbers); GM-187's retry for them
    says why it went back."""
    conv, files = _conv(tmp_path)
    _write_acts(conv, [_judgment_event("none", ["Water. None given."],
                                       references_ran=True)])
    found = det.detect_judgment_check(_ctx(_fig_rec(conv, files, [])))
    assert codes(found) == {"judgment_calls_unsourced_after_consult"}
    assert found[0]["severity"] == "info"
    assert "after the references consult" in found[0]["message"]
    _write_acts(conv, [dict(_judgment_event("fed_back", ["Water."],
                                            references_ran=True),
                            reason="after_consult")])
    found = det.detect_judgment_check(_ctx(_fig_rec(conv, files, [])))
    assert codes(found) == {"judgment_check_retry"}
    assert "after the references consult" in found[0]["message"]
    assert found[0]["evidence"]["reason"] == "after_consult"


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
    # (e), (l) and (m) run again too (EV-30)
    _write_acts(conv, acts + [
        dict(_judgment_event("none", ["Water."], references_ran=True)),
        {"turn": 1, "event": "tool_error", "name": "call_agent",
         "run_id": "x", "error": "1 validation error\nmethod\n  Field "
         "required"}])
    dj = json.loads((scen / "detectors.json").read_text("utf-8"))
    dj["turns"][0]["final"] = "The settlement is 123.4 mm."
    (scen / "detectors.json").write_text(json.dumps(dj), "utf-8")
    res = dryrun.recheck(str(tmp_path / "wave"), flows=[
        {"id": "F99-figure", "steps": []}])
    assert {f["code"] for f in res["F99-figure"]["turns"][0]} >= {
        "stale_figure_card", "judgment_calls_unsourced_after_consult",
        "missing_argument", "answer_number_unsourced"}


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


def test_a_failed_scratch_write_is_not_a_write(tmp_path):
    """EV-28 (w12 F55j): an edit_file that failed "not found" wrote
    nothing; it is the tool error it returned, not a scratch write too."""
    conv, files = _conv(tmp_path)
    acts = [{"event": "tool_start", "name": "edit_file", "run_id": "r1",
             "args": {"file_path": "calc_v2.html"}},
            {"event": "tool_end", "name": "edit_file", "run_id": "r1",
             "result": "Error: File '/calc_v2.html' not found"},
            {"event": "tool_start", "name": "write_file", "run_id": "r2",
             "args": {"file_path": "/notes.md"}},
            {"event": "tool_end", "name": "write_file", "run_id": "r2",
             "result": "Updated file /notes.md"}]
    (conv / "activity.jsonl").write_text(
        "\n".join(json.dumps(a) for a in acts) + "\n", encoding="utf-8")
    rec = {"conv_dir": str(conv), "files_dir": str(files),
           "final": "Done; the notes are saved.", "activity_offset": 0}
    found = det.detect_health(_ctx(rec))
    writes = [f["evidence"]["path"] for f in found
              if f["code"] == "scratch_filesystem_write"]
    assert writes == ["/notes.md"]
    assert "tool_returned_error" in codes(found)


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


def test_argument_refusals_by_kind():
    """EV-30: pydantic's "Field required" is a missing argument, not an
    unknown one; a value of the wrong shape is its own kind."""
    r = det.argument_refusal
    assert r("ValidationError: 1 validation error for _CallAgentArgs\n"
             "method\n  Field required [type=missing]") == "missing_argument"
    assert r("ValidationError: 1 validation error for _CallAgentArgs\n"
             "parameters\n  Input should be a valid dictionary "
             "[type=dict_type]") == "invalid_argument"
    assert r("f() got an unexpected keyword argument 'x'") == \
        "unknown_argument_refusal"
    assert r("dm7.meyerhof_hansen_Nq: 'friction_angle' is not a parameter "
             "of meyerhof_hansen_Nq.") == "unknown_argument_refusal"
    assert r("read_document() missing 1 required positional argument: "
             "'document'") == "missing_argument"
    assert r("ValueError: bad depth") is None


def test_a_refusal_fixed_on_the_next_call_is_medium(tmp_path):
    """EV-30 (w14 F18j): a call refused over its arguments that the same
    agent put right on its next call of that method did not reach the user:
    medium, not high. Left unfixed, it stays high."""
    conv, files = _conv(tmp_path)
    refused = [{"event": "tool_start", "name": "call_agent", "run_id": "a",
                "agent": "references", "task_id": "T",
                "args": {"agent_name": "reference_get", "parameters": {}}},
               {"event": "tool_error", "name": "call_agent", "run_id": "a",
                "agent": "references", "task_id": "T",
                "error": "ValidationError: 1 validation error for "
                         "_CallAgentArgs\nmethod\n  Field required"}]
    fixed = [{"event": "tool_start", "name": "call_agent", "run_id": "b",
              "agent": "references", "task_id": "T",
              "args": {"agent_name": "reference_db",
                       "method": "reference_get", "parameters": {}}},
             {"event": "tool_end", "name": "call_agent", "run_id": "b",
              "agent": "references", "task_id": "T", "result": '{"ok": 1}'}]
    rec = {"conv_dir": str(conv), "files_dir": str(files),
           "final": "Done.", "activity_offset": 0}
    # the method is missing, so the module name stands in its place
    _write_acts(conv, refused + fixed)
    f = [x for x in det.detect_health(_ctx(rec))
         if x["code"] == "missing_argument"]
    assert f and f[0]["severity"] == "medium" and f[0]["evidence"][
        "recovered"]
    # an unknown parameter, fixed on the next call of the same method
    bad = _call("c", 1.0, "dm7", "meyerhof_hansen_Nq", {"friction_angle": 32},
                result=json.dumps({"error": "dm7.meyerhof_hansen_Nq: "
                                   "'friction_angle' is not a parameter of "
                                   "meyerhof_hansen_Nq."}))
    good = _call("d", 3.0, "dm7", "meyerhof_hansen_Nq", {"phi_deg": 32})
    _write_acts(conv, bad + good)
    f = [x for x in det.detect_health(_ctx(rec))
         if x["code"] == "unknown_argument_refusal"]
    assert f and f[0]["severity"] == "medium"
    # not fixed (another method came next, and the same one never again)
    other = _call("e", 3.0, "dm7", "bearing_capacity_factors", {"phi": 32})
    _write_acts(conv, bad + other)
    f = [x for x in det.detect_health(_ctx(rec))
         if x["code"] == "unknown_argument_refusal"]
    assert f and f[0]["severity"] == "high"
    # fixed by another agent's call: still high
    _write_acts(conv, refused + [dict(a, agent="calc", task_id="U")
                                 for a in fixed])
    f = [x for x in det.detect_health(_ctx(rec))
         if x["code"] == "missing_argument"]
    assert f and f[0]["severity"] == "high"


# ---------------------------------------------------------------------------
# (m) numbers in the answer that nothing holds (EV-30)
# ---------------------------------------------------------------------------

def _numbers(answer, *held):
    return [t["raw"] for t in det.unsourced_numbers(answer, held)]


def test_answer_numbers_flag_a_number_no_tool_holds():
    """w14 F18 OFF: the methods 'spread about ±15 %' where the runs gave
    -18 % and +7 %; the tool's own numbers, rounded, are not flagged."""
    tool = json.dumps({"q_ultimate_kPa": 1158.77, "q_allowable_kPa": 386.26,
                       "meyerhof": 1240.96, "terzaghi": 950.24})
    answer = ("| Method | q_ult (kPa) |\n|---|---|\n| Vesic | 1,159 |\n"
              "| Meyerhof | 1,241 |\n| Terzaghi | 950 |\n\n"
              "The methods spread about ±15% around Vesic.")
    assert _numbers(answer, tool) == ["15"]
    # held at the precision printed, with a hedge's 3 %
    assert _numbers("q_allow is about 390 kPa.", tool) == []
    assert _numbers("q_allow is 386.3 kPa.", tool) == []
    assert _numbers("q_allow is 386.4 kPa.", tool) == ["386.4"]


def test_answer_numbers_spare_the_user_ids_and_arithmetic():
    prompt = "A 2 m square footing 1 m deep, phi = 32 degrees, 19 kN/m3."
    tool = json.dumps({"q": 386.26, "t": 316.75})
    for answer in (
            "With phi = 32° and gamma = 19 kN/m3 as given.",     # the user's
            "See GEC-6 Sec. 5.2.3.2, UFC 3-220-20 Eq. 5-13 and Table 4-1.",
            "Terzaghi (1943) and Vesic (1973), page 245, Figure 7-15.",
            "1. First step\n2. Second step\n12. Twelfth step",
            "The file bearing_capacity_t17.csv and the B-17 log's N60.",
            "Use 3 layers and 7 slices.",                        # counts
            "The load is 386.3 kPa x 4 m² = 1,545 kN.",          # shown
            "Vesic gives 386 kPa, 22% above Terzaghi's 317 kPa.",
            "q is 8.07 ksf.",                                   # converted
            "phi is 0.5585 rad."):
        assert _numbers(answer, prompt, tool) == [], answer
    # a conversion counts only in the unit printed
    assert _numbers("q is 8.07 kPa.", prompt, tool) == ["8.07"]
    # arithmetic in another paragraph does not count
    assert _numbers("Vesic 386 kPa, Terzaghi 317 kPa.\n\nThat is 22% "
                    "more.", tool) == ["22"]


def test_answer_numbers_spare_arithmetic_read_off_a_table():
    """EV-30 review (w12 F66, F61j): a difference read off the answer's own
    table is spared; a rounded value a table matches only by chance is not
    (within 1 %), and neither is a number no table holds."""
    tool = json.dumps({"lengths": [21.1, 23.7], "fos": [0.937, 0.938],
                       "grid": [15, 20]})
    lengths = ("| Option | Length needed |\n|---|---|\n| Pipe | 21.1 m |\n"
               "| H-pile | 23.7 m |\n\n")
    assert _numbers(lengths + "- The H-pile is about 2.6 m longer.",
                    tool) == []
    fos = ("| Search | FOS |\n|---|---|\n| Spencer | 0.937 |\n"
           "| Bishop | 0.938 |\n\n")
    assert _numbers(fos + "The two agree to within 0.001.", tool) == []
    # 20/15 = 1.33 is a rounded 1.3, but 2.6 % off: chance, not arithmetic
    grids = ("| Grid | FOS |\n|---|---|\n| 15 by 15 | 0.937 |\n"
             "| 20 by 20 | 0.938 |\n\n")
    assert _numbers(grids + "A required FOS of 1.3 would call for a fix.",
                    tool) == ["1.3"]
    assert _numbers(lengths + "The lengths spread about ±15%.",
                    json.dumps({"lengths": [21.1, 23.7]})) == ["15"]


def test_answer_numbers_detector_reads_tools_and_earlier_turns(tmp_path):
    """The detector holds the turn's tool results and arguments, every
    prompt, and what earlier turns' tools and answers said; not the tools'
    own descriptions."""
    assert det.detect_answer_numbers in det.DETECTORS
    conv, files = _conv(tmp_path)
    acts = ([{"turn": 1, "event": "turn_start", "prompt": "B = 2.5 m"},
             {"turn": 1, "event": "model_end", "text": "Earlier: 77.7 kPa"}]
            + [dict(a, turn=1) for a in _call("a", 1.0, "bc", "run", {},
                                              result='{"q": 12.34}')]
            + [{"turn": 2, "event": "turn_start", "prompt": "and now?"},
               {"turn": 2, "event": "tool_end", "name": "describe_method",
                "result": '{"example": 55.5}'}])
    _write_acts(conv, acts)
    rec = {"conv_dir": str(conv), "files_dir": str(files),
           "activity_offset": 4, "activity_end": None,
           "final": "q = 12.34 kPa on B = 2.5 m, 77.7 before; 55.5 now."}
    found = det.detect_answer_numbers(_ctx(rec, prompts=["B = 2.5 m"]))
    assert codes(found) == {"answer_number_unsourced"}
    assert found[0]["severity"] == "low"
    assert [n["value"] for n in found[0]["evidence"]["numbers"]] == ["55.5"]
    rec["final"] = "q = 12.34 kPa."
    assert det.detect_answer_numbers(_ctx(rec)) == []


# ---------------------------------------------------------------------------
# Given inputs and the consult's citations in deliverables (GM-155, GM-167,
# GM-181): expect keys ``given_inputs`` and ``pdf_cites_consult``
# ---------------------------------------------------------------------------

#: w13b F18j's composed report, cut down: c = 0 under "Given by user".
_F18J_HTML = (
    "<h2>1. Inputs</h2><h3>Given by user</h3><table><tr><th>Input</th>"
    "<th>Value</th></tr><tr><td>Friction angle phi</td><td>32 deg</td></tr>"
    "<tr><td>Cohesion c</td><td>0 kPa</td></tr></table>"
    "<h3>Assumed (not given)</h3><ul><li>No groundwater within influence "
    "zone</li></ul><h2>3. Method</h2><p>Vesic factors per GEC-6.</p>")

#: w13b F55j's: a status column, H worked out but marked Given.
_F55J_HTML = (
    "<table><tr><th>Item</th><th>Value</th><th>Status</th></tr>"
    "<tr><td>Retained height</td><td>4.0 m</td><td>Given</td></tr>"
    "<tr><td>Total wall height H</td><td>4.5 m (= 4.0 + 0.5)</td>"
    "<td>Given (convention of the tool)</td></tr>"
    "<tr><td>Concrete unit weight</td><td>24 kN/m3</td>"
    "<td>Assumed, not given</td></tr></table>")


def test_given_rows_read_given_parts_and_status_cells():
    rows = det.given_rows(_F18J_HTML)
    assert "Cohesion c | 0 kPa" in rows
    assert not any("groundwater" in r.lower() for r in rows)
    assert not any("Vesic" in r for r in rows)
    rows = det.given_rows(_F55J_HTML)
    assert any(r.startswith("Total wall height H") for r in rows)
    assert not any("Concrete" in r for r in rows)
    assert det.given_rows("<h2>Inputs</h2><p>B = 2 m</p>") == []


def test_brief_given_block():
    brief = ("Build the package.\nGIVEN (the user's words): \"2 m square "
             "footing\", \"phi = 32 degrees\"\nASSUMPTIONS: c = 0 (sand), "
             "no water table\nRETURN the path.")
    block = det.brief_given_block(brief)
    assert "phi = 32" in block and "c = 0" not in block
    block = det.brief_given_block("**Given:** B = 2 m\n\nAssumed: c = 0")
    assert "B = 2 m" in block and "c = 0" not in block
    # "not given" and running text are no GIVEN block
    for text in ("ASSUMPTIONS (NOT GIVEN): c = 0",
                 "phi=32 deg (user-given), c=0", "as given by the user"):
        assert det.brief_given_block(text) is None, text


def _call(rid, t, agent_name, method, params, result='{"ok": 1}'):
    return [{"event": "tool_start", "name": "call_agent", "run_id": rid,
             "agent": "calc", "ts": t,
             "args": {"agent_name": agent_name, "method": method,
                      "parameters": params}},
            {"event": "tool_end", "name": "call_agent", "run_id": rid,
             "agent": "calc", "ts": t + 1, "result": result}]


def _task(rid, t, helper, brief, result, t_end=None):
    # the activity log keeps a task's arguments as a Python repr
    return [{"event": "tool_start", "name": "task", "run_id": rid,
             "agent": "primary", "ts": t, "subagent": helper,
             "args": repr({"subagent_type": helper, "description": brief})},
            {"event": "tool_end", "name": "task", "run_id": rid,
             "agent": "primary", "ts": t_end or t + 5, "subagent": helper,
             "result": result}]


_NOT_GIVEN = {"package_given": True, "brief_given": True, "not_given": [
    {"name": "c = 0", "inputs": ["cohesion"],
     "any": ["cohesion", r"(?<![\w.'])c'?\s*=\s*0(?![.\d])"]}]}


def test_given_inputs_expectations(tmp_path):
    """The F18j rerun sees whether the package call passed `given`, whether
    an input the user did not state (c = 0 for "sand") is under Given in
    the call, a composed report or a brief, and whether briefs have a
    GIVEN block; a call that failed is not counted."""
    conv, files = _conv(tmp_path)
    (files / "calc.html").write_text(_F18J_HTML, encoding="utf-8")
    acts = (_task("t1", 1.0, "calc", "Run it. phi=32 (user-given), c=0",
                  "done")
            + _call("p0", 2.0, "calc_package", "bearing_capacity_package",
                    {"width": 2}, result='{"error": "ValueError: given: x '
                    'named but not passed"}')
            + _call("p1", 3.0, "calc_package", "bearing_capacity_package",
                    {"width": 2, "cohesion": 0})
            + _call("p2", 4.0, "calc_package", "bearing_capacity_package",
                    {"width": 2, "cohesion": 0,
                     "given": {"width": "2 m square", "cohesion": "sand"}})
            + _call("h1", 5.0, "calc_package", "html_to_pdf",
                    {"html_path": "calc.html", "output_path": "calc.pdf"}))
    _write_acts(conv, acts)
    flow = {"expect": {"given_inputs": _NOT_GIVEN}}
    found = det.detect_expectations(flow, [_Sess([_fig_rec(conv, files,
                                                            [])])])
    got = [(f["code"], f["evidence"].get("where") or
            f["evidence"].get("package")) for f in found]
    assert ("expected_package_given_missing",
            "bearing_capacity_package") in got
    assert sum(1 for c, _w in got
               if c == "expected_package_given_missing") == 1   # not p0
    assert ("expected_brief_given_missing", None) in got
    assert ("assumed_input_given", "bearing_capacity_package") in got
    assert ("assumed_input_given", "calc.pdf") in got
    assert all(f["severity"] == "info" and f["detector"] == "i_expectations"
               for f in found)
    # a brief with a GIVEN block in the user's words, c = 0 under
    # assumptions, and `given` without cohesion: nothing to say
    acts = (_task("t1", 1.0, "calc", "GIVEN: \"2 m square footing\", "
                  "\"phi = 32\"\nASSUMPTIONS: c = 0 (sand)", "done")
            + _call("p2", 4.0, "calc_package", "bearing_capacity_package",
                    {"width": 2, "cohesion": 0, "given": ["width"]}))
    _write_acts(conv, acts)
    assert det.detect_expectations(flow, [_Sess([_fig_rec(conv, files,
                                                           [])])]) == []
    # no package built and no brief: the keys say they were not exercised
    _write_acts(conv, [])
    assert codes(det.detect_expectations(
        flow, [_Sess([_fig_rec(conv, files, [])])])) == {
        "expected_package_not_called", "expected_calc_brief_missing"}


_CONSULT = ("**Scope.** FHWA-NHI-06-089 is not in the indexed references. "
            "GEC-6 Sec. 5.2.3.2 gives the factors; UFC 3-220-20 Sec. 5-2.4 "
            "the factor of safety.")


def test_pdf_cites_consult(tmp_path):
    """w13b fix 10: a composed report cites what the consult named, the
    consult finished before it; a document a tool also named counts only
    with a section the consult gave (w13b F55j: the PDF's FHWA-NHI-06-089
    came from the tool's notes)."""
    conv, files = _conv(tmp_path)
    flow = {"expect": {"pdf_cites_consult": True}}

    def run(acts):
        _write_acts(conv, acts)
        return det.detect_expectations(flow, [_Sess([_fig_rec(conv, files,
                                                               [])])])
    tool = _call("w1", 0.5, "retaining_walls", "cantilever_wall", {},
                 result=json.dumps({"notes": "Per FHWA-NHI-06-089 Sec. "
                                    "10.4.2 and GEC-6."}))
    consult = _task("r1", 1.0, "references", "which method?", _CONSULT,
                    t_end=2.0)
    good = _call("h1", 3.0, "calc_package", "html_to_pdf",
                 {"html": "<p>Factors per GEC-6 Sec. 5.2.3.2.</p>",
                  "output_path": "a.pdf"})
    assert run(tool + consult + good) == []
    other = _call("h2", 3.0, "calc_package", "html_to_pdf",
                  {"html": "<p>FS per UFC 3-220-20.</p>",
                   "output_path": "b.pdf"})
    assert run(tool + consult + other) == []
    copied = _call("h3", 3.0, "calc_package", "html_to_pdf",
                   {"html": "<p>Per FHWA-NHI-06-089 Sec. 10.4.2; GEC-6.</p>",
                    "output_path": "c.pdf"})
    found = run(tool + consult + copied)
    assert codes(found) == {"expected_consult_citation_missing"}
    assert "also named by a tool" in found[0]["message"]
    # every citation is read, not each document's first (EV-30, w14 F18j:
    # GEC-6 named first with no section, its sections cited later)
    later = _call("h5", 3.0, "calc_package", "html_to_pdf",
                  {"html": "<p>Vesic factors (GEC-6).</p><p>Shape factors "
                   "per GEC-6 Sec. 5.2.3.2.</p>", "output_path": "e.pdf"})
    assert run(tool + consult + later) == []
    # an equation number the consult gave places a citation too
    eq_tool = _call("w2", 0.5, "bearing_capacity", "run", {},
                    result=json.dumps({"notes": "UFC 3-220-20."}))
    eq_consult = _task("r2", 1.0, "references", "which?",
                       "UFC 3-220-20 Eq. 5-13 gives the factor.", t_end=2.0)
    eq = _call("h6", 3.0, "calc_package", "html_to_pdf",
               {"html": "<p>N per DM7 (UFC 3-220-20) Eq. 5-13.</p>",
                "output_path": "f.pdf"})
    assert run(eq_tool + eq_consult + eq) == []
    plain = _call("h7", 3.0, "calc_package", "html_to_pdf",
                  {"html": "<p>N per UFC 3-220-20.</p>",
                   "output_path": "g.pdf"})
    assert codes(run(eq_tool + eq_consult + plain)) == {
        "expected_consult_citation_missing"}
    # written before the consult finished
    early = _call("h4", 1.5, "calc_package", "html_to_pdf",
                  {"html": "<p>GEC-6 Sec. 5.2.3.2</p>",
                   "output_path": "d.pdf"})
    found = run(tool + consult + early)
    assert codes(found) == {"expected_consult_before_deliverable"}
    assert "1 finished after it" in found[0]["message"]
    # a package's assumptions carry the citation in the basis
    pkg = _call("p1", 3.0, "calc_package", "bearing_capacity_package",
                {"assumptions": {"gwt_depth": "not given; deep, per "
                                 "UFC 3-220-20 Sec. 5-2.4"}})
    assert run(tool + consult + pkg) == []
    assert codes(run(tool)) == {"expected_deliverable_missing"}


def test_report_shows_the_judgment_check_and_given_keys(tmp_path):
    """REPORT.md and the transcript list the judgment-call check's info
    findings (did it send an answer back?) beside the flow expectations."""
    from live_smoke import report
    wave = tmp_path / "w"
    sc = wave / "F18j-x"
    sc.mkdir(parents=True)
    retry = det.finding("l_judgment_check", "judgment_check_retry", "info",
                        "the answer was sent back once")
    given = det.finding("i_expectations", "assumed_input_given", "info",
                        "c = 0 is under Given in calc.pdf")
    (sc / "summary.json").write_text(json.dumps({"id": "F18j-x"}),
                                     encoding="utf-8")
    (sc / "detectors.json").write_text(json.dumps(
        {"turns": [{"findings": [retry]}], "scenario": [given]}),
        encoding="utf-8")
    text = open(report.write_report(str(wave)), encoding="utf-8").read()
    assert "judgment_check_retry" in text and "assumed_input_given" in text
    path = tmp_path / "t.md"
    report.write_transcript(str(path), {"id": "F18j-x"}, [{
        "final": "ok", "findings": [retry],
        "metrics": {"judgment_checks": 2, "judgment_retries": 1}}], [given])
    text = path.read_text(encoding="utf-8")
    assert "read 2 answer(s), sent 1 back" in text
    assert "judgment_check_retry" in text and "assumed_input_given" in text
