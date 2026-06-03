"""
Live smoke test for fh_prompter native (OpenAI-style) tool calling.

Run this in a Databricks / Funhouse environment where ``funhouse`` is installed
and the Prompter endpoint is reachable.  It walks three gates in order and stops
at the first hard failure:

    Step 1  is fh_prompter real?    -> PrompterAPI() constructs, has a client
    Step 2  does the endpoint take  -> fh.chat(tools=[...]) returns tool_calls
            the ``tools`` parameter?    (GO / NO-GO gate for native tool calling)
    Step 6  does the full agent     -> GeotechAgent + NativeToolEngine answers a
            loop work end to end?       tool-requiring question with no errors

Usage
-----
From a Databricks notebook cell:

    from funhouse_agent.smoke_test_native import main
    main()

Or pass an already-built PrompterAPI (e.g. ``fh_prompter`` you created earlier):

    from funhouse_agent.smoke_test_native import main
    main(fh_prompter)

Or from a shell where funhouse is importable:

    python -m funhouse_agent.smoke_test_native

The heavy ``funhouse`` import lives inside the functions, so this module imports
fine on a dev box without funhouse (it just can't run the live steps there).
"""

import json
import time


# ---------------------------------------------------------------------------
# Step 1 — confirm fh_prompter is real
# ---------------------------------------------------------------------------

def check_prompter(fh=None):
    """Construct (or accept) a PrompterAPI and report its basic shape.

    Parameters
    ----------
    fh : PrompterAPI, optional
        An existing instance.  If None, one is constructed via
        ``from funhouse import PrompterAPI``.

    Returns
    -------
    PrompterAPI
        The (possibly newly constructed) instance.
    """
    print("=" * 70)
    print("STEP 1  Is fh_prompter real?")
    print("=" * 70)
    if fh is None:
        from funhouse import PrompterAPI  # lazy attr — NOT funhouse.prompter
        fh = PrompterAPI()
        print("  Constructed PrompterAPI() from `from funhouse import PrompterAPI`")
    else:
        print("  Using the PrompterAPI instance passed in.")

    backend = getattr(fh, "backend", "<unknown>")
    chat_model = getattr(fh, "chat_model", "<unknown>")
    has_client = getattr(fh, "client", None) is not None

    print(f"  backend     : {backend}")
    print(f"  chat_model  : {chat_model}")
    print(f"  client set  : {has_client}")
    if not has_client:
        print("  WARNING: fh.client is None — native tool calling needs the raw")
        print("           OpenAI client (prompter/openai backends provide it; a")
        print("           bare Grok JSON backend does not).")
    print()
    return fh


# ---------------------------------------------------------------------------
# Step 2 — capability probe: does the endpoint accept `tools`?
# ---------------------------------------------------------------------------

def probe_tool_support(fh):
    """Send one tool-enabled chat() call and classify the result.

    Uses the budget/retry/logging-wrapped ``PrompterAPI.chat()`` path, which
    returns a *list* of normalized tool-call dicts when the model invokes a
    tool (see prompter_api.py).  This isolates "does the endpoint support
    native tools?" from the whole agent loop.

    Returns
    -------
    str
        "GO"        — endpoint returned tool_calls; native calling works.
        "TEXT"      — endpoint accepted ``tools`` but model replied with text
                      (tools probably supported; try a more forcing prompt).
        "NO_GO"     — chat() returned None (endpoint rejected tools / errored).
    """
    from funhouse_agent.native_tools import OPENAI_TOOLS

    print("=" * 70)
    print("STEP 2  Does the endpoint accept the `tools` parameter?  (GO / NO-GO)")
    print("=" * 70)

    out = fh.chat(
        user="List the available geotechnical analysis modules.",
        system=(
            "You are a tool-using assistant. When a tool can answer the "
            "request, you MUST call it rather than answering from memory."
        ),
        tools=[OPENAI_TOOLS[0]],  # just list_agents — minimal schema
        tool_choice="auto",
        temperature=0,
    )

    print(f"  return type : {type(out).__name__}")
    if isinstance(out, list):
        names = [
            (c.get("function") or {}).get("name")
            for c in out
            if isinstance(c, dict)
        ]
        print(f"  tool_calls  : {names}")
        print("  VERDICT: GO — endpoint supports native tool calling.")
        print()
        return "GO"
    if isinstance(out, str):
        print(f"  text reply  : {out[:200]!r}")
        print("  VERDICT: TEXT — `tools` was accepted (no error) but the model")
        print("           answered with text. Native calling is likely fine;")
        print("           the agent loop uses a directive system prompt that")
        print("           usually elicits calls. Proceeding to Step 6.")
        print()
        return "TEXT"
    print(f"  raw value   : {out!r}")
    print("  VERDICT: NO-GO — chat() returned None. The endpoint likely")
    print("           rejected the `tools` parameter or errored. Check the")
    print("           Funhouse logs; fall back to text-ReAct GeotechAgent(fh)")
    print("           and raise tool support with the platform team.")
    print()
    return "NO_GO"


# ---------------------------------------------------------------------------
# Step 6 — end-to-end agent smoke test
# ---------------------------------------------------------------------------

_TOOL_QUESTION = (
    "Calculate the ultimate and allowable bearing capacity of a 2 m wide "
    "square footing founded 1.5 m below grade in sand with a friction angle "
    "of 30 degrees, unit weight 18 kN/m^3, and zero cohesion. Use a factor "
    "of safety of 3. Show the key numbers."
)

_PLAIN_QUESTION = (
    "In one sentence, what is the difference between total stress and "
    "effective stress in soil?"
)


def run_smoke(fh, max_rounds=8, verbose=True):
    """Build the agent over a NativeToolEngine and run two questions.

    Returns
    -------
    dict
        {"tool_question": AgentResult, "plain_question": AgentResult}
    """
    from funhouse_agent import GeotechAgent, NativeToolEngine

    print("=" * 70)
    print("STEP 6  End-to-end agent smoke test")
    print("=" * 70)

    try:
        engine = NativeToolEngine(fh)
    except ValueError as exc:
        print(f"  Could not build NativeToolEngine: {exc}")
        print("  (This backend has no raw client — use text-ReAct instead.)")
        return {}

    agent = GeotechAgent(
        genai_engine=engine, max_rounds=max_rounds, verbose=verbose,
    )

    results = {}
    for label, question in (
        ("tool_question", _TOOL_QUESTION),
        ("plain_question", _PLAIN_QUESTION),
    ):
        print(f"\n  --- {label} ---")
        print(f"  Q: {question[:90]}{'...' if len(question) > 90 else ''}")
        t0 = time.time()
        result = agent.ask(question)
        elapsed = time.time() - t0

        tools_used = [tc.get("tool_name") for tc in result.tool_calls]
        print(f"  rounds      : {result.rounds}")
        print(f"  time (s)    : {elapsed:.1f}")
        print(f"  tools       : {tools_used}")
        print(f"  errors      : {result.errors}")
        print(f"  answer      : {result.answer[:400]}"
              f"{'...' if len(result.answer) > 400 else ''}")
        results[label] = result

    # Pass criteria for the tool question
    tq = results.get("tool_question")
    if tq is not None:
        called = [tc.get("tool_name") for tc in tq.tool_calls]
        ok = ("call_agent" in called) and not tq.errors
        print()
        print(f"  SMOKE RESULT: {'PASS' if ok else 'REVIEW'} "
              f"(call_agent used: {'call_agent' in called}, "
              f"errors: {len(tq.errors)})")
    print()
    return results


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def main(fh=None, max_rounds=8, verbose=True):
    """Run Steps 1, 2, and 6 in order, stopping at a hard NO-GO.

    Parameters
    ----------
    fh : PrompterAPI, optional
        Reuse an existing instance instead of constructing one.
    """
    fh = check_prompter(fh)

    verdict = probe_tool_support(fh)
    if verdict == "NO_GO":
        print("Stopping: native tool calling is unavailable on this endpoint.")
        print("Fallback:  agent = GeotechAgent(genai_engine=fh)  # text-ReAct")
        return {"step1_prompter": fh, "step2_verdict": verdict}

    results = run_smoke(fh, max_rounds=max_rounds, verbose=verbose)
    return {
        "step1_prompter": fh,
        "step2_verdict": verdict,
        "step6_results": results,
    }


if __name__ == "__main__":
    main()
