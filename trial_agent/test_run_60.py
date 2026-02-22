"""
30 more trial agent tests (41-70) — advanced scenarios, cross-agent, stress tests.

Tests 41-50: Cross-agent workflows and multi-step problems.
Tests 51-60: Edge cases, unusual parameters, boundary conditions.
Tests 61-70: Stress tests — ambiguous questions, unit confusion, missing data.

Usage:
    python trial_agent/test_run_60.py          # run all 30
    python trial_agent/test_run_60.py 45       # run just test 45
    python trial_agent/test_run_60.py 41-50    # run tests 41 through 50
"""
import json
import sys
import os
import time

# Fix Windows console encoding for Unicode (Greek letters, etc.)
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr.encoding != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anthropic

from trial_agent.system_prompt import SYSTEM_PROMPT
from trial_agent.tools import TOOLS
from trial_agent.agent_registry import (
    call_agent, list_methods, describe_method, list_agents,
)

MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 4096
MAX_TOOL_ROUNDS = 8


def dispatch_tool(tool_name: str, tool_input: dict) -> dict:
    """Route a tool call to the appropriate registry function."""
    if tool_name == "call_agent":
        return call_agent(
            tool_input["agent_name"],
            tool_input["method"],
            tool_input["parameters"],
        )
    elif tool_name == "list_methods":
        return list_methods(
            tool_input["agent_name"],
            tool_input.get("category", ""),
        )
    elif tool_name == "describe_method":
        return describe_method(
            tool_input["agent_name"],
            tool_input["method"],
        )
    else:
        return {"error": f"Unknown tool: {tool_name}"}


def _make_system_with_cache(text: str) -> list:
    return [
        {
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def run_single_question(question: str, test_num: int = 0):
    """Run a single question through the full agent pipeline."""
    client = anthropic.Anthropic(timeout=120.0)
    messages = [{"role": "user", "content": question}]
    system = _make_system_with_cache(SYSTEM_PROMPT)

    print(f"\n{'='*70}")
    print(f"TEST {test_num}: {question[:100]}...")
    print(f"{'='*70}")

    t_start = time.time()

    print("\n[Calling Claude API...]")
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        tools=TOOLS,
        messages=messages,
    )
    print(f"[Stop reason: {response.stop_reason}]")

    iteration = 0
    total_input_tokens = response.usage.input_tokens
    total_output_tokens = response.usage.output_tokens
    cache_read = getattr(response.usage, 'cache_read_input_tokens', 0) or 0
    cache_create = getattr(response.usage, 'cache_creation_input_tokens', 0) or 0

    while response.stop_reason == "tool_use":
        iteration += 1
        if iteration > MAX_TOOL_ROUNDS:
            print(f"\n[Reached {MAX_TOOL_ROUNDS} tool rounds, stopping]")
            break

        print(f"\n--- Tool Use Round {iteration} ---")

        tool_results = []
        for block in response.content:
            if block.type == "text":
                print(f"[Text] {block.text[:200]}")
            elif block.type == "tool_use":
                inp_str = json.dumps(block.input, separators=(",", ":"))
                print(f"  [Tool] {block.name}({inp_str[:300]})")

                t0 = time.time()
                result = dispatch_tool(block.name, block.input)
                t1 = time.time()
                result_str = json.dumps(result, indent=2)
                print(f"  [Result] ({t1-t0:.2f}s) {result_str[:500]}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result),
                })

        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": tool_results})

        print(f"\n[Calling Claude API again...]")
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=system,
            tools=TOOLS,
            messages=messages,
        )
        print(f"[Stop reason: {response.stop_reason}]")
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens
        cache_read += getattr(response.usage, 'cache_read_input_tokens', 0) or 0
        cache_create += getattr(response.usage, 'cache_creation_input_tokens', 0) or 0

    # Print final response
    print(f"\n{'='*70}")
    print("FINAL RESPONSE:")
    print(f"{'='*70}")
    for block in response.content:
        if hasattr(block, "text"):
            print(block.text)

    t_total = time.time() - t_start
    cost_input = total_input_tokens * 3 / 1_000_000
    cost_output = total_output_tokens * 15 / 1_000_000
    cost_cache_create = cache_create * 3.75 / 1_000_000
    cost_cache_read = cache_read * 0.30 / 1_000_000
    total_cost = cost_input + cost_output + cost_cache_create + cost_cache_read
    print(f"\n{'='*70}")
    print(f"STATS: {iteration} tool rounds | {t_total:.1f}s total")
    print(f"  Tokens: {total_input_tokens} in + {total_output_tokens} out"
          f" | cache: {cache_create} created, {cache_read} read")
    print(f"  Est. cost: ${total_cost:.4f}")
    print(f"{'='*70}")

    return {
        "test_num": test_num,
        "rounds": iteration,
        "time_s": t_total,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "cost": total_cost,
    }


# ---------------------------------------------------------------------------
# Test questions (numbered 41-70)
# ---------------------------------------------------------------------------

TESTS = {
    # =====================================================================
    # CROSS-AGENT WORKFLOWS AND MULTI-STEP PROBLEMS (41-50)
    # =====================================================================

    # 41: Pile design workflow — capacity + group + drivability
    41: (
        "I need to support a 4500 kN column load using HP14x73 piles (0.356m) "
        "driven into 20m of medium dense sand (phi=33, gamma=19 kN/m3). "
        "First estimate the single pile capacity, then tell me how many piles "
        "I need in a group (use FS=2.5), and check the group efficiency."
    ),

    # 42: Retaining wall + seismic — cantilever wall with earthquake loading
    42: (
        "I have a 5m cantilever retaining wall (3m base, 1.5m toe, 1.5m heel, "
        "0.4m stem, gamma_concrete=24) with granular backfill (phi=30, gamma=18). "
        "First check the static stability. Then calculate the seismic active "
        "earth pressure with kh=0.1 and compare KA vs KAE."
    ),

    # 43: Foundation on improved ground — bearing before and after
    43: (
        "A site has very soft clay (Su=12 kPa, gamma=15.5 kN/m3). I'm planning "
        "to preload with a 3m high surcharge fill (gamma_fill=20 kN/m3) for 6 months "
        "with wick drains (ch=2.5 m2/yr, drain spacing 1.5m triangular). "
        "What degree of consolidation will I achieve, and what will be the "
        "improved Su? (Assume Su/sigma'v = 0.22 for NC clay.)"
    ),

    # 44: Drilled shaft in 3 layers — clay/sand/rock
    44: (
        "Design a drilled shaft through a complex profile: 0-5m soft clay "
        "(Su=30 kPa, gamma=17), 5-12m dense sand (phi=38, gamma=20, "
        "N60=35), 12-15m limestone (qu=25 MPa, RQD=80%). Shaft diameter "
        "is 1.0m, total length 15m. What is the ultimate capacity?"
    ),

    # 45: Settlement comparison — elastic vs Schmertmann vs consolidation
    45: (
        "Compare three settlement methods for a 2.5m x 2.5m footing at 1m "
        "depth under 120 kPa net pressure on medium sand (Es=20 MPa, nu=0.3, "
        "gamma=18 kN/m3). Run: (1) elastic settlement, (2) Schmertmann with "
        "3 sublayers (Es = 15, 20, 30 MPa for 0-1m, 1-2.5m, 2.5-5m below "
        "footing), time=10 years. Which gives the larger estimate?"
    ),

    # 46: Liquefaction + residual strength + slope stability
    46: (
        "A 6m high embankment (2H:1V slopes, gamma=19 kN/m3) sits on a "
        "liquefiable sand layer at 3-8m depth (N1_60=8, sigma_v_eff=60 kPa). "
        "First check liquefaction (Mw=7.0, PGA=0.2g), then estimate the "
        "residual strength, and comment on post-earthquake stability."
    ),

    # 47: Sheet pile + groundwater — dewatering scenario
    47: (
        "A 5m deep excavation in sand (phi=30, gamma_sat=20 kN/m3, "
        "gamma_dry=17 kN/m3) has the water table at 2m below ground. "
        "Design an anchored sheet pile wall with anchor at 1m below the "
        "top. Account for the water pressures on both sides."
    ),

    # 48: Ground improvement comparison — wick drains vs surcharge alone
    48: (
        "A 4m thick soft clay layer (cv=1.5 m2/yr, ch=3.0 m2/yr, Cc=0.3, "
        "e0=1.0, sigma_v0=40 kPa) is loaded with 80 kPa. Compare: "
        "(1) surcharge alone for 12 months (Hdr=2m, double drainage), "
        "(2) wick drains at 1.5m triangular spacing for 6 months. "
        "Which achieves more consolidation?"
    ),

    # 49: Lateral pile — compare free vs fixed head
    49: (
        "Compare free-head vs fixed-head behavior for a 0.5m steel pipe pile "
        "(wall thickness 12mm, E=200 GPa) embedded 12m in stiff clay "
        "(Su=60 kPa, gamma=18 kN/m3, eps50=0.007). Apply 80 kN lateral load "
        "at the ground surface. How much does fixity reduce deflection?"
    ),

    # 50: Multi-agent: SPT corrections + bearing capacity + classification
    50: (
        "From a boring log at 3m depth: raw SPT N=18, borehole diameter 100mm, "
        "rod length 4m, country=United States. Effective stress is 54 kPa. "
        "Correct the N-value, estimate friction angle, then compute "
        "bearing capacity for a 1.5m square footing at 1m depth using "
        "the estimated phi. Also classify the soil (LL=28, PL=16, fines=25%)."
    ),

    # =====================================================================
    # EDGE CASES AND UNUSUAL PARAMETERS (51-60)
    # =====================================================================

    # 51: Bearing capacity — extremely deep footing (D/B > 2)
    51: (
        "Estimate the bearing capacity of a 1.0m square footing at 5.0m depth "
        "in dense sand (phi=40, gamma=20 kN/m3). The D/B ratio is 5.0, which "
        "is very deep. Use the Vesic method."
    ),

    # 52: Pile in very soft clay — very low Su
    52: (
        "A 0.3m square concrete pile is driven 25m into very soft marine clay "
        "with Su=8 kPa uniformly (gamma=14.5 kN/m3). What capacity can we "
        "expect? Is a single pile even useful here?"
    ),

    # 53: MSE wall — tall wall with surcharge
    53: (
        "Check a 9m high MSE wall with 0.75m reinforcement spacing, 6.3m "
        "reinforcement length, and a 20 kPa traffic surcharge. Reinforced "
        "soil: phi=34, gamma=20. Retained/foundation soil: phi=30, gamma=18. "
        "Geogrid Tult=80 kN/m, Rc=1.0."
    ),

    # 54: Slope stability with seismic loading (pseudo-static)
    54: (
        "Re-analyze a 10m high slope at 2H:1V in medium clay (Su=45 kPa, "
        "gamma=18.5 kN/m3) with a horizontal seismic coefficient kh=0.15. "
        "Compare the static FOS to the pseudo-static seismic FOS."
    ),

    # 55: Wave equation — very hard driving (should approach refusal)
    55: (
        "Can a Delmag D19-32 hammer drive a 0.3m HP pile through 12m of very "
        "dense sand (unit shaft resistance 100 kPa, unit toe 12000 kPa)? "
        "I'm worried about refusal. Check the bearing graph from 500 to 3000 kN."
    ),

    # 56: Drilled shaft — capacity vs depth curve
    56: (
        "Show me how the capacity of a 1.0m diameter drilled shaft builds up "
        "with depth in uniform stiff clay (Su=90 kPa, gamma=19). Plot from "
        "5m to 20m in 1m increments. At what depth do I reach 3000 kN?"
    ),

    # 57: Consolidation — thick layer with high OCR
    57: (
        "A 10m thick heavily overconsolidated clay (OCR=4) has Cc=0.5, "
        "Cr=0.06, e0=0.8. Current effective stress at midpoint is 80 kPa, "
        "preconsolidation pressure=320 kPa. Apply 200 kPa loading. "
        "How much settlement occurs in the OC range vs the NC range?"
    ),

    # 58: Bearing capacity — Hansen method (different from Vesic)
    58: (
        "Calculate the bearing capacity of a 2m x 4m rectangular footing at "
        "1.5m depth using the Hansen method (not Vesic). Soil is c-phi with "
        "c=10 kPa, phi=28, gamma=17.5 kN/m3. Compare with Vesic."
    ),

    # 59: Ground improvement — feasibility check for organic clay
    59: (
        "What ground improvement methods are feasible for a 5m thick organic "
        "clay layer (Su=10 kPa, water content=80%, organic content=15%)? "
        "The site needs to support a light warehouse (50 kPa)."
    ),

    # 60: Pile group 6-DOF with battered piles
    60: (
        "Analyze a pile group with 4 vertical piles and 2 battered piles "
        "(batter 1:4) using the 6-DOF method. All piles are 0.4m diameter, "
        "15m long, E=200 GPa. Vertical load 2000 kN, horizontal load 200 kN. "
        "Pile spacing is 1.2m."
    ),

    # =====================================================================
    # STRESS TESTS — AMBIGUITY, UNIT CONFUSION, MISSING DATA (61-70)
    # =====================================================================

    # 61: Vague question — should the LLM ask for more info?
    61: (
        "I have a clay site and need to know if I can put a footing there. "
        "The soil has an unconfined compressive strength of 50 kPa."
    ),

    # 62: Mixed units — some metric, some US
    62: (
        "Calculate bearing capacity for a 5-foot wide strip footing at 3 feet "
        "depth. The soil has phi=30 degrees and unit weight 120 pcf. "
        "Give the answer in both kPa and ksf."
    ),

    # 63: Contradictory data — phi=0 with sand (should flag this)
    63: (
        "Estimate the bearing capacity of a 2m square footing at 1m depth in "
        "sand with phi=0 degrees and gamma=18 kN/m3. Use Vesic."
    ),

    # 64: Incomplete pile problem — missing key data
    64: (
        "What is the capacity of a driven pile in clay? The pile is 12m long "
        "and 0.3m diameter."
    ),

    # 65: Very large numbers — stress test numerical limits
    65: (
        "Calculate the bearing capacity of a 0.5m square footing at 0.3m depth "
        "in rock (c=2000 kPa, phi=45 degrees, gamma=25 kN/m3). The capacity "
        "should be very high."
    ),

    # 66: Negative or zero values — should handle gracefully
    66: (
        "What happens if I try to calculate bearing capacity with zero footing "
        "width? Also what if friction angle is negative? I want to understand "
        "the error handling."
    ),

    # 67: Real-world messy problem — highway bridge foundation
    67: (
        "I'm designing foundations for a highway bridge pier. The boring log "
        "shows 3m of fill (gamma=17, phi=25), then 5m of soft clay (Su=20, "
        "gamma=16), then dense sand to depth (phi=35, gamma=20, N60=40). "
        "The pier load is 6000 kN. Should I use spread footings or deep "
        "foundations? If deep, recommend pile type and estimate capacity."
    ),

    # 68: Groundhog correlations — chain of SPT to multiple properties
    68: (
        "Given an SPT blow count of N=30 at 10m depth in sand with effective "
        "stress of 100 kPa, use the groundhog correlations to estimate: "
        "(1) relative density, (2) friction angle, (3) small-strain shear "
        "modulus Gmax."
    ),

    # 69: DM7 lookup — see if it can find a specific equation
    69: (
        "Using DM7, calculate the Boussinesq stress increase at 3m below "
        "the center of a 2m x 4m rectangular loaded area carrying 100 kPa."
    ),

    # 70: Capacity vs depth comparison — drilled shaft vs driven pile
    70: (
        "Compare a 0.6m diameter drilled shaft vs a 0.356m HP14x73 driven "
        "pile in the same soil profile: 8m medium clay (Su=50 kPa, gamma=18) "
        "over dense sand (phi=35, gamma=20). Both are 15m long. Which has "
        "higher capacity?"
    ),
}


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    # Parse arguments: single number, range (e.g. "41-50"), or all
    test_nums = sorted(TESTS.keys())
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if "-" in arg:
            start, end = arg.split("-")
            test_nums = [n for n in test_nums if int(start) <= n <= int(end)]
        else:
            test_nums = [int(arg)]

    results = []
    for i, num in enumerate(test_nums):
        print(f"\n\n{'#'*70}")
        print(f"# RUNNING TEST {num} ({i+1} of {len(test_nums)})")
        print(f"{'#'*70}")
        r = run_single_question(TESTS[num], test_num=num)
        results.append(r)

    # Print summary table
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'TEST':>5} {'ROUNDS':>6} {'TIME':>7} {'IN_TOK':>8} {'OUT_TOK':>8} {'COST':>8}")
    print(f"{'-'*5:>5} {'-'*6:>6} {'-'*7:>7} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8}")
    total_cost = 0
    for r in results:
        print(f"{r['test_num']:>5} {r['rounds']:>6} {r['time_s']:>6.1f}s "
              f"{r['input_tokens']:>8,} {r['output_tokens']:>8,} ${r['cost']:>.4f}")
        total_cost += r["cost"]
    print(f"{'-'*5:>5} {'-'*6:>6} {'-'*7:>7} {'-'*8:>8} {'-'*8:>8} {'-'*8:>8}")
    print(f"{'TOTAL':>5} {'':>6} {'':>7} {'':>8} {'':>8} ${total_cost:.4f}")
