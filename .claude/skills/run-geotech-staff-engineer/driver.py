"""Smoke driver for GeotechStaffEngineer — run with the project venv python.

    .venv/Scripts/python .claude/skills/run-geotech-staff-engineer/driver.py all

Subcommands (all offline; no API key is read or spent):
    library   direct class-API call into an analysis module (bearing_capacity)
    dispatch  flat-JSON LLM tool surface (funhouse_agent.dispatch.call_agent)
    apptest   boot webapp/app.py headless via streamlit AppTest with a mocked
              engine and drive one chat turn end-to-end
    all       the three above in order; nonzero exit on any failure

Each subcommand prints PASS/FAIL and returns exit code 0/1. The driver never
imports pytest — it is a standalone harness a future agent can run in seconds
before/after touching the analysis modules, the adapters, or the webapp.
"""

import os
import sys
import tempfile
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)
os.chdir(REPO)


def smoke_library():
    """Direct in-process call into an analysis module (the layer most PRs touch).

    NOTE the API shape: modules export analysis CLASSES (compute() -> result
    dataclass with summary()/to_dict()), not top-level analyze_*() functions.
    """
    from bearing_capacity import (
        BearingCapacityAnalysis, BearingSoilProfile, Footing, SoilLayer)

    soil = BearingSoilProfile(layer1=SoilLayer(friction_angle=32.0,
                                               unit_weight=19.0))
    analysis = BearingCapacityAnalysis(
        footing=Footing(width=2.0, depth=1.0, shape="square"),
        soil=soil, vertical_load=500.0)
    result = analysis.compute()
    qult = result.to_dict()["q_ultimate_kPa"]
    # Vesic, phi=32, 2 m square at 1 m: verified 1158.8 kPa when this driver
    # was written. Band guards against silent factor regressions, not noise.
    assert 1100.0 < qult < 1220.0, f"qult {qult:.1f} kPa outside sanity band"
    print(f"  qult = {qult:.1f} kPa (Vesic, 2 m square, D=1 m, phi=32)")


def smoke_dispatch():
    """The flat-JSON tool surface the LLM agent calls (adapters layer)."""
    from funhouse_agent.dispatch import call_agent, list_methods

    methods = list_methods("bearing_capacity")
    flat = {name for group in methods.values() for name in group}
    assert "bearing_capacity_analysis" in flat, f"methods changed: {flat}"

    out = call_agent("bearing_capacity", "bearing_capacity_analysis", {
        "width": 2.0, "depth": 1.0, "shape": "square",
        "friction_angle": 32.0, "unit_weight": 19.0, "vertical_load": 500.0})
    assert "error" not in out, f"dispatch error: {out.get('error')}"
    qult = out.get("q_ultimate_kPa")
    assert qult and 1100.0 < qult < 1220.0, f"dispatch qult {qult!r} off-band"
    print(f"  call_agent('bearing_capacity', ...) -> qult = {qult:.1f} kPa")


def smoke_apptest():
    """Boot webapp/app.py and drive one chat turn — engine + stream mocked, so
    no key is needed and no LLM is called. Same harness webapp/tests uses."""
    from streamlit.testing.v1 import AppTest

    import webapp.core as core
    import webapp.engine_config as engine_config
    from webapp.engine_config import EngineResolution

    tmp = tempfile.mkdtemp(prefix="geotech_driver_")
    os.environ["GEOTECH_WEBAPP_DATA"] = tmp

    # Mock the engine so the app "has a model" without touching the network.
    engine_config.resolve_engine = lambda *a, **k: EngineResolution(
        model=object(), source="anthropic", model_name="driver-mock",
        message="")
    core.build_agent = lambda *a, **k: object()
    core.build_reviewer_agent = lambda kind, *a, **k: f"reviewer:{kind}"

    def fake_stream(_agent, _messages, _thread_id, **_kw):
        yield {"kind": "token", "text": "Driver smoke "}
        yield {"kind": "token", "text": "answer."}
        yield {"kind": "turn_done", "answer": "Driver smoke answer.",
               "turn_tokens": 5}
    core.stream_turn = fake_stream

    at = AppTest.from_file(os.path.join(REPO, "webapp", "app.py"),
                           default_timeout=60).run()
    assert not at.exception, f"app boot raised: {at.exception}"
    assert at.session_state["initialized"] is True

    at.chat_input[0].set_value("What is bearing capacity?").run()
    assert not at.exception, f"chat turn raised: {at.exception}"
    transcript = at.session_state["transcript"]
    roles = [entry["role"] for entry in transcript]
    assert roles == ["user", "assistant"], f"unexpected transcript: {roles}"
    assert transcript[1]["text"] == "Driver smoke answer."
    print(f"  app booted, 1 mocked chat turn persisted (data root {tmp})")


SMOKES = {"library": smoke_library, "dispatch": smoke_dispatch,
          "apptest": smoke_apptest}


def main(argv):
    which = argv[1] if len(argv) > 1 else "all"
    names = list(SMOKES) if which == "all" else [which]
    if not all(n in SMOKES for n in names):
        print(f"usage: driver.py [{'|'.join(SMOKES)}|all]")
        return 2
    failed = False
    for name in names:
        print(f"[{name}]")
        try:
            SMOKES[name]()
            print(f"  PASS {name}")
        except Exception:
            traceback.print_exc()
            print(f"  FAIL {name}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
