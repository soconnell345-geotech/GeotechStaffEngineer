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
    assert res["ok"]
    # $0 and its own ledger: the real one is untouched.
    after = os.path.getmtime(real_ledger) if os.path.exists(real_ledger) \
        else None
    assert before == after
    out = tmp_path / "runs" / "dry"
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
    covered = {p for f in flows for p in [f.get("page", "document_review")]}
    assert covered == pages


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
