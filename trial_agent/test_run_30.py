"""
30 additional trial agent tests — bread-and-butter + edge cases.

Tests 11-26: Untested agents and methods.
Tests 27-40: Edge cases and weird scenarios on previously-tested agents.

Usage:
    python trial_agent/test_run_30.py          # run all 30
    python trial_agent/test_run_30.py 15       # run just test 15
    python trial_agent/test_run_30.py 11-15    # run tests 11 through 15
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
        "pass": True,  # We'll visually inspect
    }


# ---------------------------------------------------------------------------
# Test questions (numbered 11-40, continuing from test_run.py's 1-10)
# ---------------------------------------------------------------------------

TESTS = {
    # =====================================================================
    # BREAD-AND-BUTTER: Untested agents and methods
    # =====================================================================

    # 11: Sheet pile — cantilever (sheet_pile agent)
    11: (
        "Design a cantilever sheet pile wall retaining 4m of sandy backfill "
        "(phi=30 degrees, gamma=18 kN/m3). The water table is at the bottom "
        "of the excavation. What embedment depth is required?"
    ),

    # 12: Sheet pile — anchored (sheet_pile agent)
    12: (
        "Design an anchored sheet pile wall retaining 6m of granular soil "
        "(phi=32 degrees, gamma=19 kN/m3). The anchor is at 1.5m below the "
        "top of the wall. What is the required embedment and anchor force?"
    ),

    # 13: Lateral pile analysis (lateral_pile agent)
    13: (
        "Analyze a 0.6m diameter steel pipe pile (wall thickness 12.7mm, "
        "E=200 GPa) embedded 15m in medium stiff clay (Su=50 kPa, "
        "gamma=18 kN/m3, epsilon50=0.01). A lateral load of 100 kN is "
        "applied at the ground surface with a free-head condition. "
        "What are the maximum deflection and bending moment?"
    ),

    # 14: Pile group — simple vertical (pile_group agent)
    14: (
        "A 3x3 pile group (9 piles) of 0.356m HP piles at 1.0m spacing "
        "supports a column load of 3600 kN plus a moment of 500 kN-m about "
        "one axis. What is the maximum and minimum pile load?"
    ),

    # 15: Wave equation — bearing graph (wave_equation agent)
    15: (
        "Generate a bearing graph for a 0.356m HP14x73 steel pile, 15m long, "
        "being driven by a Vulcan 06 hammer. The pile has E=200 GPa, "
        "unit weight 78.5 kN/m3. Use a quake of 2.5mm and Smith damping of "
        "0.5 s/m for side, 0.15 s/m for toe."
    ),

    # 16: Downdrag analysis (downdrag agent)
    16: (
        "A 0.356m steel H-pile is driven through 8m of settling fill "
        "(gamma=17 kN/m3, Su=20 kPa, Cc=0.3, e0=1.2) into 7m of dense "
        "sand bearing layer (phi=35, gamma=19.5 kN/m3). The fill is "
        "expected to settle under a 3m surcharge. Estimate the dragload "
        "and neutral plane depth."
    ),

    # 17: MSE wall (retaining_walls agent)
    17: (
        "Design check a 6m high MSE wall with geogrid reinforcement. "
        "Retained soil has phi=28 degrees, gamma=18 kN/m3. Reinforced soil "
        "has phi=34 degrees, gamma=20 kN/m3. Geogrid Tult=50 kN/m, "
        "coverage ratio Rc=1.0, pullout F*=0.67, alpha=1.0. "
        "Use 6 layers of reinforcement at 1m vertical spacing, each 4.2m long."
    ),

    # 18: Ground improvement — aggregate piers (ground_improvement agent)
    18: (
        "Design aggregate pier ground improvement for a 3m x 3m footing "
        "on soft clay (Su=25 kPa, gamma=17 kN/m3, Es=5 MPa). The pier "
        "diameter is 0.6m, aggregate phi=42 degrees, and the treatment "
        "depth is 6m. Target bearing capacity is 150 kPa."
    ),

    # 19: Seismic earth pressure — Mononobe-Okabe (seismic_geotech agent)
    19: (
        "Calculate the seismic active earth pressure coefficient (KAE) for "
        "a retaining wall. Backfill phi=32 degrees, wall friction delta=21 "
        "degrees, horizontal seismic coefficient kh=0.15, kv=0. The wall "
        "face is vertical and the backfill is level."
    ),

    # 20: Elastic settlement — standalone (settlement agent)
    20: (
        "Estimate the immediate elastic settlement of a 3m x 6m rectangular "
        "footing on dense sand (Es=40 MPa, Poisson's ratio=0.3) under a "
        "net contact pressure of 150 kPa."
    ),

    # 21: Schmertmann settlement (settlement agent)
    21: (
        "Estimate the settlement of a 2m wide square footing at 1m depth "
        "using the Schmertmann method. Net applied pressure is 100 kPa. "
        "The soil has 3 sublayers below the footing: "
        "Layer 1: 0-1m below footing, Es=15 MPa; "
        "Layer 2: 1-2m below footing, Es=25 MPa; "
        "Layer 3: 2-4m below footing, Es=35 MPa. "
        "Unit weight is 18 kN/m3. Time since loading is 5 years."
    ),

    # 22: Pile group efficiency (pile_group agent)
    22: (
        "Calculate the group efficiency of a 4x4 pile group (16 piles) "
        "with 0.4m diameter piles at 1.2m center-to-center spacing. "
        "Use the Converse-Labarre method."
    ),

    # 23: Drilled shaft LRFD (drilled_shaft agent)
    23: (
        "Evaluate a 1.2m diameter drilled shaft, 18m deep in stiff clay "
        "(Su=100 kPa, gamma=19 kN/m3) using AASHTO LRFD. The factored "
        "axial load is 4000 kN. Does it pass Strength I?"
    ),

    # 24: Ground improvement — vibro compaction (ground_improvement agent)
    24: (
        "Evaluate vibro compaction for densifying a 5m thick loose sand "
        "layer (initial Dr=35%, fines content=8%). Target relative density "
        "is 70%. Probe spacing is 2.5m in a triangular pattern. "
        "Is vibro compaction feasible for this soil?"
    ),

    # 25: Seismic residual strength (seismic_geotech agent)
    25: (
        "Estimate the post-liquefaction residual shear strength for a "
        "liquefiable sand layer with corrected N1_60 = 10 blows/ft. "
        "Use both the Seed-Harder and Idriss-Boulanger correlations."
    ),

    # 26: Combined settlement analysis (settlement agent)
    26: (
        "Run a full combined settlement analysis for a 3m x 3m footing at "
        "2m depth on a profile with: 2m sand (Es=30 MPa) over 4m of NC "
        "clay (Cc=0.25, e0=0.9, cv=2.0 m2/yr, gamma=17.5 kN/m3, "
        "Ca_Ce_ratio=0.04). Applied load is 500 kN. "
        "Estimate immediate + consolidation + secondary settlement at 10 years."
    ),

    # =====================================================================
    # EDGE CASES AND WEIRD SCENARIOS
    # =====================================================================

    # 27: Bearing capacity — eccentric + inclined load (tricky combo)
    27: (
        "A 2.5m x 2.5m square footing at 1.5m depth in sand (phi=28, "
        "gamma=17.5 kN/m3) has a vertical load of 600 kN applied with "
        "0.3m eccentricity in the B-direction and an inclined load at "
        "10 degrees from vertical. Use the Vesic method. "
        "What is the allowable capacity with FS=3?"
    ),

    # 28: Bearing capacity — very weak soil (should flag low capacity)
    28: (
        "Estimate the ultimate bearing capacity of a 1.2m wide strip "
        "footing at 0.5m depth on very soft clay (Su=10 kPa, phi=0, "
        "gamma=15 kN/m3). Is this soil suitable for a spread footing?"
    ),

    # 29: Slope stability — should be unstable (FS < 1.0)
    29: (
        "Analyze the stability of a 15m high slope at 1H:1V (45 degrees) "
        "in soft clay with Su=25 kPa and unit weight 19 kN/m3. "
        "Use the Bishop method. The slope should be unstable."
    ),

    # 30: Driven pile — all clay (no sand layer, alpha method only)
    30: (
        "Calculate the capacity of a 0.3m square precast concrete pile "
        "driven 20m into a uniform soft to medium clay profile. Su varies "
        "linearly from 20 kPa at the surface to 60 kPa at 20m depth. "
        "Unit weight is 18 kN/m3 throughout. Use the alpha method."
    ),

    # 31: Retaining wall — sloping backfill (uncommon parameter)
    31: (
        "Check the stability of a 4m high cantilever retaining wall with "
        "a 2.8m wide base (0.8m toe, 2.0m heel, 0.35m stem thickness). "
        "The backfill slopes upward at 15 degrees (beta=15), with "
        "phi=33 degrees and gamma=19 kN/m3. Concrete gamma=24 kN/m3."
    ),

    # 32: Consolidation — overconsolidated clay (uses Cr, not just Cc)
    32: (
        "A 5m thick clay layer has Cc=0.4, Cr=0.08, e0=1.3, "
        "preconsolidation pressure=150 kPa, current effective stress=100 "
        "kPa at midpoint, gamma=16.5 kN/m3. A 30 kPa load is applied. "
        "Calculate the consolidation settlement. The clay is overconsolidated."
    ),

    # 33: Drilled shaft — rock socket (different method than soil)
    33: (
        "Calculate the capacity of a 0.9m diameter drilled shaft with a "
        "3m rock socket in moderately weathered sandstone (qu=15 MPa, "
        "RQD=60%). Above the rock, there is 10m of stiff clay (Su=80 kPa, "
        "gamma=18.5 kN/m3). Total shaft length is 13m."
    ),

    # 34: Lateral pile — layered soil with water table
    34: (
        "Analyze a 0.5m diameter concrete pile (E=30 GPa, solid section) "
        "embedded 12m in a layered profile: 0-3m soft clay (Su=20 kPa, "
        "eps50=0.02), 3-6m medium sand (phi=30, gamma=18 kN/m3), 6-12m "
        "stiff clay (Su=80 kPa, eps50=0.005). Water table at 3m. "
        "Free-head, 50 kN lateral load at ground surface."
    ),

    # 35: Wave equation — drivability check (multi-depth analysis)
    35: (
        "Perform a drivability analysis for a 0.4m square concrete pile "
        "(20m long, E=30 GPa, unit weight 24 kN/m3) driven with an ICE 42S "
        "hammer. Soil is 10m of loose sand (unit shaft=20 kPa) over 10m of "
        "dense sand (unit shaft=80 kPa, unit toe=6000 kPa). "
        "Can the pile be driven to full depth without damage?"
    ),

    # 36: Pile group — eccentric biaxial loading
    36: (
        "A 2x3 pile group (6 piles) of 0.4m piles at 1.2m x 1.5m spacing "
        "carries V=2400 kN, Mx=300 kN-m, and My=200 kN-m simultaneously. "
        "What is the maximum pile load and which pile gets it?"
    ),

    # 37: Ground improvement — deep soil mixing (unusual method)
    37: (
        "Evaluate deep soil mixing to stabilize a 7m thick soft clay layer "
        "(Su=15 kPa, gamma=16 kN/m3). Target unconfined compressive strength "
        "of the mixed soil is 500 kPa. Column diameter 0.8m, "
        "square pattern at 1.5m spacing."
    ),

    # 38: Bearing capacity — circular footing (not square or strip)
    38: (
        "Calculate the bearing capacity of a 2m diameter circular footing "
        "at 1.0m depth in a c-phi soil (c=15 kPa, phi=25 degrees, "
        "gamma=18 kN/m3). The water table is at the footing base."
    ),

    # 39: US customary input (forces LLM to convert before calling)
    39: (
        "I have SPT data in US customary: N=22 blows/ft at 25 feet depth, "
        "hammer energy ratio 60%, effective stress 2100 psf. "
        "Correct the N-value for overburden (N1_60) and estimate the "
        "friction angle of the sand."
    ),

    # 40: Multi-agent complex — full foundation design workflow
    40: (
        "Complete foundation evaluation: A 2m x 3m rectangular footing at "
        "1.5m depth on medium dense sand (phi=30, gamma=18 kN/m3, Es=20 MPa, "
        "nu=0.3). Column load is 800 kN. The site is in a seismic zone with "
        "Vs30=250 m/s. Check: (1) bearing capacity with FS=3, "
        "(2) elastic settlement, and (3) site classification."
    ),
}


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    # Parse arguments: single number, range (e.g. "11-15"), or all
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
