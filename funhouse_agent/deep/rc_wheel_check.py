"""One-cell Funhouse/Databricks health check for the v5.1.0rc* wheel.

Paste the body of this file into a Databricks notebook cell AFTER installing the
wheel and running ``dbutils.library.restartPython()``. It runs:

  1. OFFLINE regression (no API) — the lateral-pile calc-package bug that
     prompted this rc: package-vs-analysis agreement, verified file write,
     unknown-param rejection, and the /Workspace placeholder rescue.
  2. LIVE self-check (needs ``fh_prompter`` in scope) — 2 proofs via the
     Funhouse model: the agent drives a real tool call, and cross-session
     memory save/recall.

Section 1 always runs and needs no API. Section 2 is skipped with a clear
message if ``fh_prompter`` is not defined.

You can also run it from the repo: ``from funhouse_agent.deep.rc_wheel_check
import run_rc_check; run_rc_check(fh_prompter)``.
"""

import os
import tempfile


def run_rc_check(fh_prompter=None, model_name="funhouse-gpt-high",
                 launch_chat=False):
    """Run the rc wheel health check. Returns True if all run checks passed."""
    from funhouse_agent.dispatch import call_agent

    ok = True
    print("=" * 64)
    try:
        from importlib.metadata import version
        print("geotech-staff-engineer version:", version("geotech-staff-engineer"))
    except Exception as e:
        print("version lookup unavailable (running from source?):", e)
    print("=" * 64)

    # ---- 1. OFFLINE regression — the lateral-pile calc-package bug ----------
    print("\n[1] OFFLINE regression (no API)")
    tmp = tempfile.gettempdir()
    out = os.path.join(tmp, "rc_check_lateral_pile.html")
    layers = [{"top": 0.0, "bottom": 15.0, "py_model": "SandAPI",
               "phi": 32.0, "gamma": 19.0, "k": 8150.0}]

    # 1a. Build the calc package (0.8 m drilled shaft, 15 m, dry sand, 30%
    #     cracked-concrete stiffness pile_E=7.5e6 kPa) and verify the WRITE.
    pkg = call_agent("calc_package", "lateral_pile_package", {
        "pile_length": 15.0, "pile_diameter": 0.8, "pile_E": 7.5e6,
        "soil_layers": layers, "Vt": 20.0, "head_condition": "free",
        "project_name": "rc wheel check", "output_path": out})
    wrote = (pkg.get("status") == "success" and pkg.get("file_exists")
             and pkg.get("file_size_bytes", 0) > 1000)
    print(f"  1a calc package written & verified: {_mark(wrote)} "
          f"(status={pkg.get('status')}, file_exists={pkg.get('file_exists')}, "
          f"{pkg.get('file_size_bytes')} bytes, y_top={pkg.get('y_top_mm')} mm)")
    ok &= wrote

    # 1b. The bug that started this: the calc package re-runs the analysis, so
    #     the package result MUST match a direct analysis with the SAME pile_E.
    ana = call_agent("lateral_pile", "lateral_pile_analysis", {
        "pile_type": "pipe", "pile_diameter": 0.8, "pile_length": 15.0,
        "pile_E": 7.5e6, "Vt": 20.0,
        "layers": [{"top": 0.0, "bottom": 15.0, "model": "SandAPI",
                    "phi": 32.0, "gamma": 19.0, "k": 8150.0}]})
    y_ana = ana.get("deflection_m", [None])[0]
    agree = y_ana is not None and abs(y_ana * 1000 - pkg.get("y_top_mm", 0)) < 0.05
    print(f"  1b package matches direct analysis: {_mark(agree)} "
          f"(package {pkg.get('y_top_mm')} mm vs analysis "
          f"{round(y_ana * 1000, 3) if y_ana else '?'} mm)")
    ok &= agree

    # 1c. An invented stiffness param must be REJECTED, not silently dropped
    #     (the old failure ran the 200 GPa steel default and was confidently
    #     wrong). Expect an error that names E_GPa.
    bad = call_agent("lateral_pile", "lateral_pile_analysis", {
        "pile_diameter": 0.8, "pile_length": 15.0, "Vt": 20.0, "E_GPa": 9.3,
        "layers": [{"top": 0, "bottom": 15, "model": "SandAPI",
                    "phi": 32, "gamma": 19, "k": 8150}]})
    rejected = "error" in bad and "E_GPa" in str(bad.get("error", ""))
    print(f"  1c unknown param rejected (not silently dropped): {_mark(rejected)}")
    ok &= rejected

    # 1d. /Workspace placeholder rescue (INFORMATIONAL — cluster-dependent, not
    #     gated). Writing to /Workspace on Databricks can silently store a
    #     PLACEHOLDER (or corrupt PDF). Any of: a verified real write, a
    #     rescue-to-/tmp, or a clean error is acceptable — only a silent
    #     PLACEHOLDER-without-rescue is bad, which the content check now
    #     prevents. So this reports the outcome but does not fail the gate.
    ws = "/Workspace/Users/_geotech_rc_check/rc_check_lateral_pile.html"
    try:
        wpkg = call_agent("calc_package", "lateral_pile_package", {
            "pile_length": 15.0, "pile_diameter": 0.8, "pile_E": 7.5e6,
            "soil_layers": layers, "Vt": 20.0, "head_condition": "free",
            "project_name": "rc /Workspace check", "output_path": ws})
        if wpkg.get("status") == "success" and wpkg.get("file_exists"):
            print(f"  1d /Workspace write durable on this env (INFO): "
                  f"{wpkg.get('output_path')}")
        elif wpkg.get("rescue_path"):
            print(f"  1d /Workspace not durable -> rescued to /tmp (INFO, "
                  f"correct): {wpkg.get('rescue_path')}")
        else:
            print(f"  1d /Workspace returned a clean error (INFO, acceptable): "
                  f"{str(wpkg.get('error'))[:90]}")
    except Exception as e:
        print(f"  1d /Workspace check skipped (INFO): {type(e).__name__}: {e}")

    # ---- 2. LIVE self-check (needs the Funhouse prompter) -------------------
    print("\n[2] LIVE self-check (Funhouse model)")
    if fh_prompter is None:
        print("  SKIPPED — pass fh_prompter to run the live proofs, e.g.")
        print("    run_rc_check(fh_prompter)")
    else:
        try:
            from funhouse_agent.deep.databricks_bridge import PrompterChatModel
            from funhouse_agent.deep.selfcheck import run_selfcheck
            model = PrompterChatModel(prompter=fh_prompter, model=model_name)
            results = run_selfcheck(model)  # prints its own PASS/FAIL summary
            live_ok = all(r.get("ok") for r in results.values())
            ok &= live_ok
            if launch_chat and live_ok:
                from funhouse_agent.deep.agent import build_deep_agent
                from funhouse_agent.deep.notebook import DeepNotebookChat
                agent = build_deep_agent(model)
                DeepNotebookChat(agent).display()
        except Exception as e:
            print(f"  LIVE check FAILED: {type(e).__name__}: {e}")
            ok = False

    print("\n" + "=" * 64)
    print("OVERALL:", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    print("=" * 64)
    return ok


def _mark(b):
    return "PASS" if b else "FAIL"


if __name__ == "__main__":
    run_rc_check()
