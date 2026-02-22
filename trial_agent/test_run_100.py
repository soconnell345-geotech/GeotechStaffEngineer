"""
30 more trial agent tests (71-100) — DM7, OpenSees, pyStrata, advanced scenarios.

Tests 71-80: Untested/under-tested agents (DM7, groundhog, opensees, pystrata, seismic_signals, liquepy).
Tests 81-90: Multi-agent engineering workflows and verification problems.
Tests 91-100: Corner cases, tricky problems, and LLM reasoning challenges.

Usage:
    python trial_agent/test_run_100.py          # run all 30
    python trial_agent/test_run_100.py 75       # run just test 75
    python trial_agent/test_run_100.py 71-80    # run tests 71 through 80
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
    """Wrap system prompt with cache_control for prompt caching."""
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
    print(f"\n{'='*70}")
    print(f"STATS: {iteration} tool rounds | {t_total:.1f}s total")
    print(f"  Tokens: {total_input_tokens} in + {total_output_tokens} out"
          f" | cache: {cache_create} created, {cache_read} read")
    print(f"  Est. cost: ${cost_input + cost_output:.4f}")
    print(f"{'='*70}")

    return {
        "test": test_num,
        "rounds": iteration,
        "time": t_total,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "cost": cost_input + cost_output,
    }


# ---------------------------------------------------------------------------
# Test questions 71-100
# ---------------------------------------------------------------------------

TESTS = {
    # =====================================================================
    # UNDER-TESTED AGENTS: DM7, GROUNDHOG, OPENSEES, PYSTRATA (71-80)
    # =====================================================================

    # 71: DM7 — earth pressure at rest (K0)
    71: (
        "Using DM7 equations, calculate the coefficient of earth pressure at rest "
        "(K0) for a normally consolidated sand with phi=32 degrees. Then calculate "
        "K0 for the same soil if it's overconsolidated with OCR=3."
    ),

    # 72: DM7 — consolidation theory (Cv, time factor)
    72: (
        "Using DM7, calculate how long it takes for 90% consolidation of a 4m thick "
        "clay layer (double drainage) with Cv = 2.5 m2/year. What is the time factor Tv90?"
    ),

    # 73: DM7 — active earth pressure with surcharge
    73: (
        "Using DM7, calculate the total active earth pressure on a 6m retaining wall "
        "with phi=28 degrees, gamma=18 kN/m3, and a uniform surcharge of 20 kPa. "
        "Include both the soil self-weight and surcharge components."
    ),

    # 74: Groundhog — CPT correlations
    74: (
        "Using groundhog CPT correlations: at 5m depth, a CPT shows qc=8 MPa, "
        "fs=40 kPa, u2=50 kPa. Effective overburden is 50 kPa. "
        "Calculate: (1) friction ratio, (2) normalized cone resistance Qt, "
        "(3) estimated friction angle."
    ),

    # 75: Groundhog — phase relations
    75: (
        "A saturated clay sample has water content w=45%, specific gravity Gs=2.70. "
        "Using groundhog phase relation correlations, compute: "
        "(1) void ratio, (2) porosity, (3) saturated unit weight, (4) dry unit weight."
    ),

    # 76: Groundhog — Boussinesq stress from strip load
    76: (
        "Using groundhog, calculate the Boussinesq vertical stress increase "
        "at 3m depth directly below the edge of a 2m wide strip footing "
        "carrying 150 kPa."
    ),

    # 77: DM7 — slope stability chart (Taylor)
    77: (
        "Using DM7, for a 12m high slope with 30 degree inclination in "
        "undrained clay (Su=45 kPa, gamma=18 kN/m3), what is the stability "
        "number and factor of safety? Use the Taylor stability chart method."
    ),

    # 78: DM7 — pile group settlement
    78: (
        "Using DM7 methods, estimate the settlement of a 3x3 pile group with "
        "0.3m diameter piles at 0.9m spacing. Pile length is 10m. The soil below "
        "the pile tips has Es=30 MPa. The total group load is 1800 kN."
    ),

    # 79: Groundhog — K0 and earth pressure
    79: (
        "Using groundhog earth pressure methods, compute the active and passive "
        "earth pressure coefficients for a soil with phi=35 degrees. Then calculate "
        "the Rankine active pressure at 4m depth with gamma=19 kN/m3."
    ),

    # 80: DM7 — permeability and seepage
    80: (
        "Using DM7, calculate the seepage flow rate through a 10m wide earth dam "
        "with upstream head of 8m and downstream head of 2m. The dam is 20m long "
        "(perpendicular to flow) with permeability k=1e-5 m/s."
    ),

    # =====================================================================
    # MULTI-AGENT VERIFICATION WORKFLOWS (81-90)
    # =====================================================================

    # 81: Cross-check bearing capacity: bearing_capacity vs groundhog Nq/Ngamma
    81: (
        "For phi=30 degrees, compare the bearing capacity factors (Nq, Ngamma) "
        "from the bearing_capacity agent versus the groundhog correlations. "
        "Do they agree? Show the values side by side."
    ),

    # 82: Full foundation design — bearing + settlement + check
    82: (
        "Design a square footing for a 500 kN column load on medium dense sand "
        "(phi=33, gamma=18.5, Es=35 MPa, Poisson's ratio=0.3). Start with a "
        "1.5m wide footing at 1.0m depth and check both bearing capacity (FS>=3) "
        "and elastic settlement (<=25mm). Adjust the footing size if needed."
    ),

    # 83: Pile + downdrag analysis
    83: (
        "A 0.3m square concrete pile, 18m long, passes through 3m of recent fill "
        "(gamma=16, settling 200mm) and 5m of soft clay (Su=20 kPa, gamma=16, "
        "Cc=0.4, e0=1.2) before bearing in dense sand (phi=36, gamma=20). "
        "Calculate the axial capacity then check for downdrag. Is the pile adequate "
        "for a 400 kN service load?"
    ),

    # 84: Sheet pile + groundwater seepage
    84: (
        "Design a cantilever sheet pile wall with water on one side at 4m height "
        "and dry on the other. Soil is sandy gravel (phi=33, gamma_sat=20, "
        "gamma_dry=17 kN/m3). What embedment depth and maximum moment are needed?"
    ),

    # 85: Lateral pile + p-y curves
    85: (
        "Analyze a 0.6m diameter steel pipe pile (wall=12.7mm, E=200 GPa) under "
        "50 kN lateral load at the ground surface. Soil is stiff clay above water "
        "table (Su=75 kPa, gamma=18, epsilon_50=0.005). Free-head condition. "
        "Show the deflection at ground surface and maximum bending moment."
    ),

    # 86: Settlement comparison — elastic vs Schmertmann vs consolidation
    86: (
        "A 3m x 3m footing at 1.5m depth applies 200 kPa to medium dense sand "
        "(Es=25 MPa, Poisson's=0.3). Below is a 4m sand layer over 3m of clay "
        "(Cc=0.3, e0=0.9, gamma=17). Compare elastic settlement, Schmertmann "
        "settlement, and consolidation settlement of the clay layer."
    ),

    # 87: Retaining wall with seismic + water
    87: (
        "A 7m cantilever retaining wall retains sandy backfill (phi=32, gamma=19) "
        "with water table at 3m below the backfill surface. Design earthquake has "
        "kh=0.15. Calculate static sliding/overturning factors, then check the "
        "seismic earth pressure using Mononobe-Okabe. Is the wall adequate?"
    ),

    # 88: Ground improvement — compare aggregate piers vs vibro replacement
    88: (
        "Compare aggregate piers versus vibro-replacement for improving a 5m "
        "thick loose sand layer (N=5, phi=25, gamma=17) under a 100 kPa footing "
        "load. Target: increase bearing capacity to FS>=3 and reduce settlement. "
        "Which method is more effective for this soil?"
    ),

    # 89: Wave equation — compare two hammers on same pile
    89: (
        "Compare a Vulcan 06 and a Delmag D19-32 driving a 0.3m HP pile 12m into "
        "medium sand (unit shaft resistance 40 kPa, unit toe 5000 kPa). "
        "Which hammer is more efficient? Compare blow counts and pile stresses."
    ),

    # 90: DM7 + groundhog — double-check consolidation time
    90: (
        "Verify the time for 50% consolidation of a 6m clay layer (single drainage) "
        "with Cv=1.5 m2/year using both DM7 and groundhog methods. Do they give "
        "the same answer?"
    ),

    # =====================================================================
    # CORNER CASES AND LLM REASONING CHALLENGES (91-100)
    # =====================================================================

    # 91: Eccentricity — one-way vs two-way
    91: (
        "A 3m x 5m rectangular footing at 1m depth has a 600 kN vertical load "
        "with eccentricity ex=0.4m (along width) and ey=0.3m (along length). "
        "Soil: phi=30, gamma=18 kN/m3. Calculate the effective footing dimensions "
        "and bearing capacity considering two-way eccentricity."
    ),

    # 92: Very soft soil — should flag low capacity
    92: (
        "A 2m strip footing at 0.5m depth bears on soft clay with Su=8 kPa "
        "and gamma=14.5 kN/m3. What is the allowable bearing capacity? "
        "Can it support a 30 kPa line load?"
    ),

    # 93: Deep excavation — sheet pile with multiple bracing levels
    93: (
        "Design an anchored sheet pile wall for a 7m deep excavation in soft clay "
        "(Su=25 kPa, gamma=17 kN/m3) with anchor at 2m depth. What embedment "
        "is needed and what is the anchor force?"
    ),

    # 94: Pile in liquefiable soil — what happens to capacity?
    94: (
        "A 0.4m diameter pipe pile is 15m long in a profile with 5m of crust "
        "(Su=40 kPa, gamma=18), then 5m of liquefiable sand (phi=30, gamma=19, "
        "N1_60=10), then 5m of dense sand (phi=38, gamma=20). Calculate the "
        "pile capacity before liquefaction. Then estimate how much capacity "
        "is lost if the middle layer liquefies (assume residual strength Sr=5 kPa)."
    ),

    # 95: Tricky units — tonnage to kN
    95: (
        "A bridge pier applies 800 tonnes (metric) to a pile group. Convert this "
        "to kN and then determine how many HP14x73 piles are needed if each has "
        "an allowable capacity of 900 kN. Use a group efficiency of 0.8."
    ),

    # 96: Negative skin friction zone — how much pile length is "lost"?
    96: (
        "A 20m driven pile passes through 8m of settling fill and clay, then 12m "
        "of bearing sand. If the dragload from the settling zone is 300 kN and "
        "the ultimate skin friction in the sand is 150 kN/m, at what depth does "
        "the neutral plane occur? What is the structural load on the pile at "
        "the neutral plane if the service load is 500 kN?"
    ),

    # 97: Slope with tension crack
    97: (
        "Analyze a 6m high vertical cut (90 degree slope) in stiff clay with "
        "Su=60 kPa and gamma=19 kN/m3. What is the critical height for "
        "undrained failure? Is a 6m cut stable? Consider tension cracks."
    ),

    # 98: Two-layer bearing capacity (strong over weak)
    98: (
        "A 2m square footing at 1m depth sits on a 2m thick dense sand layer "
        "(phi=40, gamma=20) overlying soft clay (Su=15, gamma=16). Use the "
        "Meyerhof and Hanna two-layer method to check if the weak clay controls."
    ),

    # 99: Very long pile — check if capacity plateaus
    99: (
        "Plot the capacity of a 0.4m diameter closed-end pipe pile from 5m to "
        "40m length in uniform medium sand (phi=32, gamma=18.5, delta/phi=0.75). "
        "At what depth does the capacity essentially plateau due to the critical "
        "depth concept?"
    ),

    # 100: Grand finale — full geotechnical investigation
    100: (
        "A 4-story building (total load 4000 kN per column, 6m column spacing) "
        "is planned for a site with: 2m of fill (gamma=17, phi=25), 4m of medium "
        "clay (Su=40, gamma=17, Cc=0.25, e0=0.9), 3m of medium sand (phi=30, "
        "gamma=18.5, N60=20), and dense gravel below (phi=38, gamma=20, N60=50). "
        "Water table is at 2m. Evaluate: (1) shallow foundation feasibility, "
        "(2) settlement on shallow foundations, (3) driven pile option with "
        "capacity estimate. Recommend the best foundation type."
    ),
}


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    # Parse arguments: single number, range (e.g. "71-80"), or all
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
    print(f"{'-'*5} {'-'*6} {'-'*7} {'-'*8} {'-'*8} {'-'*8}")
    total_cost = 0
    for r in results:
        print(f"{r['test']:>5} {r['rounds']:>6} {r['time']:>6.1f}s {r['input_tokens']:>8,} {r['output_tokens']:>8,} ${r['cost']:.4f}")
        total_cost += r["cost"]
    print(f"{'-'*5} {'-'*6} {'-'*7} {'-'*8} {'-'*8} {'-'*8}")
    print(f"{'TOTAL':>5} {'':>6} {'':>7} {'':>8} {'':>8} ${total_cost:.4f}")
