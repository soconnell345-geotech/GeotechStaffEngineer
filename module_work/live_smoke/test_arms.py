"""Offline tests of the EV-33 arms ($0: scripted models, no API).

Run from the repo root::

    .venv\\Scripts\\python.exe -m pytest module_work/live_smoke/test_arms.py -q

* the meter: a second model object priced at its own rates on the one
  meter and ledger, each call naming its agent;
* the arms: the app settings they name, the order, the wiring check that
  stops a mis-wired scenario before it pays;
* reading an answer against a flow's truth; the estimate;
* the EV-33 flows set their switches and carry a truth;
* the session's Behavior setting and per-id models follow the app;
* the $0 arms wave end to end.
"""

from __future__ import annotations

import json
import os
import re
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (os.path.dirname(HERE), REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("langchain_core")

from live_smoke import arms  # noqa: E402
from live_smoke import spend  # noqa: E402

ARM_PRICES = {"big": {"input": 4.0, "output": 20.0, "cache_read": 0.4,
                      "cache_write": 5.0},
              "small": {"input": 2.0, "output": 10.0, "cache_read": 0.2,
                        "cache_write": 2.5}}


def test_a_second_model_is_metered_at_its_own_price_on_one_meter(tmp_path):
    """The helper's model object gets for_model's handler: its calls are
    priced at its rates, land on the one meter and ledger under the one
    cap, and name the agent from the run's lc_agent_name."""
    from langchain_core.messages import AIMessage
    from live_smoke.models import scripted_model
    led = spend.SpendLedger(str(tmp_path / "l.json"))
    led.begin_wave("w", 6.0, model="big", mode="arms")
    meter = spend.SpendMeter(led, "w", 6.0, "big", prices=ARM_PRICES)
    usage = {"input_tokens": 1_000_000, "output_tokens": 0}
    big = scripted_model(lambda m: AIMessage(content="a"), usage=usage,
                         callbacks=[meter])
    small = scripted_model(lambda m: AIMessage(content="b"), usage=usage,
                           callbacks=[meter.for_model("small")])
    big.invoke("hi")
    small.invoke("hi", config={"metadata": {"lc_agent_name": "references"}})
    assert [(c["model"], c["agent"]) for c in meter.calls] == [
        ("big", "primary"), ("small", "references")]
    assert [c["usd"] for c in meter.calls] == [pytest.approx(4.0),
                                               pytest.approx(2.0)]
    assert led.wave_spent("w") == pytest.approx(6.0)
    by = meter.by_agent()
    assert by["references on small"]["usd"] == pytest.approx(2.0)
    assert by["primary on big"]["agent"] == "primary"
    with pytest.raises(spend.UnpricedModel):
        meter.for_model("no-such-model")
    with pytest.raises(spend.WaveCapReached):       # one cap for both
        small.invoke("hi")


def test_arms_name_the_app_settings_and_alternate():
    from webapp import helper_models
    assert arms.arm_env("split") == {
        "GEOTECH_HELPER_MODELS": "references=Sonnet 5.5"}
    assert arms.arm_env("opus_all") == {
        "GEOTECH_HELPER_MODELS": "references=main"}
    assert arms.expected_helper_models("split") == {
        "references": "claude-sonnet-5-5"}
    assert arms.expected_helper_models("opus_all") == {"references": None}
    assert arms.arm_models(["opus_all", "split"]) == [
        "claude-opus-5-5", "claude-sonnet-5-5"]
    # split names the references helper's model as Foundry's default does
    assert arms.ARMS["split"]["helpers"]["references"] == \
        helper_models.HOST_DEFAULTS["foundry"]["references"]
    assert arms.ARMS["split"]["main"] == arms.label_id(
        helper_models.HOST_MAIN_DEFAULTS["foundry"])
    f = [{"id": "A"}, {"id": "B"}, {"id": "C"}]
    assert [(x["id"], a) for x, a in arms.order(f, ["p", "q"])] == [
        ("A", "p"), ("A", "q"), ("B", "q"), ("B", "p"), ("C", "p"),
        ("C", "q")]
    sid = arms.scenario_id("F67-x", "split")
    assert arms.split_scenario_id(sid) == ("F67-x", "split")
    af = arms.arm_flow({"id": "F67-x", "env": {"GEOTECH_JUDGMENT_CALLS": "1"},
                        "steps": [{"say": "hi"}]}, "split")
    assert af["id"] == sid and af["arm"] == "split"
    assert af["env"] == {"GEOTECH_JUDGMENT_CALLS": "1",
                         "GEOTECH_HELPER_MODELS": "references=Sonnet 5.5"}
    with pytest.raises(ValueError):
        arms.arm("no-such-arm")


def _session(model, helpers, built):
    agent = types.SimpleNamespace(
        geotech_build={"helper_models": built} if built else {})
    return types.SimpleNamespace(ss=types.SimpleNamespace(
        agent=agent, agent_error=None, model=model,
        engine=types.SimpleNamespace(model=types.SimpleNamespace(
            model=model)), helper_engines=helpers))


def test_arm_wiring_check_catches_a_helper_on_the_wrong_model():
    """A scenario whose agent is not on the arm's models never pays."""
    from webapp.helper_models import HelperModel
    sonnet = HelperModel("references", "Sonnet 5.5", "env",
                         model_id="claude-sonnet-5-5", label="Sonnet 5.5",
                         model=object())
    on_main = HelperModel("references", "main", "env")
    ok = _session("claude-opus-5-5", [sonnet],
                  {"references": "claude-sonnet-5-5"})
    assert arms.wiring_findings(ok, "split") == []
    assert arms.wiring_findings(ok, "opus_all")       # Sonnet: wrong arm
    plain = _session("claude-opus-5-5", [on_main], None)
    assert arms.wiring_findings(plain, "opus_all") == []
    bad = arms.wiring_findings(plain, "split")
    assert bad and all(f["code"] == "arm_wiring" and f["severity"] == "high"
                       for f in bad)
    # the main agent on the wrong model
    assert arms.wiring_findings(_session(
        "claude-sonnet-5-5", [sonnet], {"references": "claude-sonnet-5-5"}),
        "split")


def test_arms_read_the_key_number_methods_and_cites():
    key = {"name": "side resistance", "near": "side resistance",
           "units": "kN\\b|MN\\b", "ranges": [[2900, 3500], [2.9, 3.5]]}
    got = arms.key_number("The nominal **side resistance** in the sand is "
                          "Q_s = 3,274 kN (beta 0.607).", key)
    assert got["in_range"] and got["value"] == 3274
    got = arms.key_number("side resistance about 4,611 kN by depth-beta",
                          key)
    assert not got["in_range"] and got["value"] == 4611
    assert arms.key_number("side resistance 3.27 MN", key)["in_range"]
    assert arms.key_number("no number here", key)["value"] is None
    truth = {"methods": [{"name": "log-spiral", "any": ["log[- ]?spiral"]},
                         {"name": "chart", "any": ["4-13"]}],
             "methods_min": 2,
             "cites": [{"ref": "UFC 3-220-20", "any": ["\\b193\\b"]}]}
    ans = ("Kp = 4.6 by the log‑spiral method, UFC 3-220-20 Fig. 4-13 "
           "(p. 193).")
    assert arms.methods_named(ans, truth)["ok"]
    assert arms.cites_check(ans, truth)["ok"]
    assert "p. 193" in arms.cited(ans) and "Fig 4-13" in arms.cited(ans)
    assert not arms.cites_check("UFC 3-220-20 p. 194", truth)["ok"]
    assert not arms.methods_named("log spiral", truth)["ok"]


def _write_run(root, wave, sid, usd, agents, model="claude-sonnet-5-5"):
    d = root / wave / sid
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps(
        {"id": sid, "model": model, "usd": usd}), encoding="utf-8")
    with open(d / "activity.jsonl", "w", encoding="utf-8") as fh:
        for agent, n_in, n_out in agents:
            fh.write(json.dumps({"event": "model_end", "agent": agent,
                                 "usage": {"input_tokens": n_in,
                                           "output_tokens": n_out}}) + "\n")


def test_arms_estimate_prices_the_helper_share_at_its_model(tmp_path):
    _write_run(tmp_path, "w13-judgment", "F1j-x", 1.0,
               [("primary", 600, 0), ("references", 400, 0)])
    _write_run(tmp_path, "w14-judgment", "F2-x", 0.2, [("primary", 900, 10)])
    _write_run(tmp_path, "w9-figures", "F3-x", 9.0,
               [("references", 900, 10)])            # not a basis wave
    _write_run(tmp_path, "w13-judgment", "F4-x", 9.0,
               [("references", 900, 10)], model="claude-opus-5-5")
    flows = [{"id": "N1", "truth": {"helpers": ["references"]}},
             {"id": "N2", "truth": {"helpers": []}}]
    est = arms.estimate(flows, ["opus_all", "split"], str(tmp_path),
                        cap=2.0)
    usd = {(r["flow"], r["arm"]): r["usd"] for r in est["rows"]}
    assert est["basis"]["references_share"] == pytest.approx(0.4)
    assert usd[("N1", "opus_all")] == pytest.approx(2.0)
    assert usd[("N1", "split")] == pytest.approx(0.6 * 2 + 0.4 * 1)
    assert usd[("N2", "opus_all")] == usd[("N2", "split")] == \
        pytest.approx(0.4)
    assert est["total_usd"] == pytest.approx(2.0 + 1.6 + 0.8)
    assert not est["fits"] and est["cut"] == ["N1"]


def test_ms_flows_set_their_switches_and_carry_a_truth():
    """EV-33's flows (GM-239): every one sets the judgment-call switch and
    "Consult references" itself, and says what a right answer contains."""
    from live_smoke import runner
    from webapp import core
    flows = [f for f in runner.load_flows()
             if re.match(r"F\d+-ms-", f["id"])]
    assert len(flows) == 8
    assert [bool(f["truth"]["helpers"]) for f in flows].count(False) >= 1
    for f in flows:
        assert f["page"] == "geotech", f["id"]
        assert f["env"].get("GEOTECH_JUDGMENT_CALLS") in ("0", "1"), f["id"]
        assert "GEOTECH_HELPER_MODELS" not in f["env"], f["id"]
        assert set(f["behavior"]) <= set(core.default_behavior()), f["id"]
        assert f["behavior"]["references"] in ("anytime", "off"), f["id"]
        t = f["truth"]
        assert set(t["helpers"]) <= {"references", "calc", "figure"}
        k = t["key"]
        assert k["value"] and k["source"] and k["ranges"], f["id"]
        assert all(lo < hi for lo, hi in k["ranges"]), f["id"]
        re.compile(k["near"])
        if k.get("units"):
            re.compile(k["units"])
        for m in t.get("methods") or []:
            assert m["name"] and m["any"], f["id"]
            for p in m["any"]:
                re.compile(p)
        assert int(t.get("methods_min", 0)) <= len(t.get("methods") or [])
        for c in t.get("cites") or []:
            re.compile(c["ref"])
            for p in c["any"]:
                re.compile(p)
        assert int(t.get("cites_needed", 0)) <= len(t.get("cites") or [])


def test_session_set_behavior_follows_the_sidebar(tmp_path):
    """A flow's ``behavior`` is set as the Behavior expander sets it: the
    agent is rebuilt when a setting other than the trace changes."""
    from live_smoke import models, session
    m = models.scripted_model(model="claude-haiku-5-5")
    with session.AppEnv(str(tmp_path / "sb"), m, model_id="claude-haiku-5-5",
                        sharepoint=False) as env:
        s = session.Session(env, "geotech", "LIVESMOKE\\tester")
        first = s.ss.agent
        assert first is not None, s.ss.agent_error
        assert s.set_behavior({"references": "anytime"}) is False
        assert s.ss.agent is first
        assert s.set_behavior({"references": "off"}) is True
        assert s.ss.behavior["references"] == "off"
        assert s.ss.agent is not None and s.ss.agent is not first
        with pytest.raises(ValueError):
            s.set_behavior({"no_such_setting": 1})


def test_app_env_hands_each_model_out_by_id(tmp_path):
    """An arms wave's builder returns the model asked for, as the Prompter
    builder does, and the model list names each by its label."""
    from live_smoke import models, session
    from webapp import core, engine_config
    big = models.scripted_model(model="claude-opus-5-5")
    small = models.scripted_model(model="claude-sonnet-5-5")
    with session.AppEnv(str(tmp_path / "sb"), big, model_id="claude-opus-5-5",
                        models={"claude-opus-5-5": big,
                                "claude-sonnet-5-5": small},
                        labels=arms.MODEL_LABELS, sharepoint=False):
        assert os.environ["GEOTECH_PROMPTER_MODELS"] == \
            "Opus 5.5=claude-opus-5-5,Sonnet 5.5=claude-sonnet-5-5"
        assert "Sonnet 5.5" in [c["label"] for c in core.model_choices()]
        assert engine_config.resolve_engine(
            model_id="claude-sonnet-5-5").model is small
        assert engine_config.resolve_engine(
            model_id="claude-opus-5-5").model is big
    with session.AppEnv(str(tmp_path / "sb2"), big,
                        model_id="claude-opus-5-5", sharepoint=False):
        # every earlier wave: the one model, by its id
        assert os.environ["GEOTECH_PROMPTER_MODELS"] == \
            "claude-opus-5-5=claude-opus-5-5"


def test_arms_dry_run_meters_each_agent_on_its_arms_model(tmp_path):
    """The $0 arms wave end to end: both arms through the app's helper
    wiring, the references helper metered on Sonnet in split and on Opus
    in opus_all, the control calling no helper, RESULTS.md written."""
    from live_smoke import dryrun
    res = dryrun.run_arms(wave="dry-arms", runs_dir=str(tmp_path / "runs"))
    assert res["ok"], res["problems"]
    out = tmp_path / "runs" / "dry-arms"
    assert (out / "RESULTS.md").is_file() and (out / "results.json").is_file()
    data = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert data["fake"] is True
    rows = {(r["flow"], r["arm"]): r for r in data["rows"]}
    kp = rows[("F68-ms-passive-kp-sloping-ground", "split")]
    assert kp["by_agent"]["references"]["models"] == ["claude-sonnet-5-5"]
    assert kp["by_agent"]["primary"]["models"] == ["claude-opus-5-5"]
    kp = rows[("F68-ms-passive-kp-sloping-ground", "opus_all")]
    assert kp["by_agent"]["references"]["models"] == ["claude-opus-5-5"]
    assert kp["wiring"]["built"] == {}
    summ = json.loads((out / "F68-ms-passive-kp-sloping-ground@split" /
                       "summary.json").read_text(encoding="utf-8"))
    assert summ["arm"] == "split" and "scripted fake" in summ["model"]
    assert summ["wiring"]["built"] == {"references": "claude-sonnet-5-5"}
    # a fake wave's runs never count as recorded costs
    assert spend.recorded_costs(str(tmp_path / "runs"),
                                "claude-opus-5-5") == {}
