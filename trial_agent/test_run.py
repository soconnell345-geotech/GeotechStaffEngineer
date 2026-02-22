"""
Non-interactive test of the trial agent pipeline.

Tests the full round-trip: user question -> Claude API -> tool_use -> agent dispatch -> response.
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
MAX_TOOL_ROUNDS = 8  # Prevent runaway exploration


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


def run_single_question(question: str):
    """Run a single question through the full agent pipeline."""
    client = anthropic.Anthropic(timeout=120.0)
    messages = [{"role": "user", "content": question}]
    system = _make_system_with_cache(SYSTEM_PROMPT)

    print(f"\n{'='*70}")
    print(f"QUESTION: {question}")
    print(f"{'='*70}")

    t_start = time.time()

    # First API call
    print("\n[Calling Claude API...]")
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        tools=TOOLS,
        messages=messages,
    )
    print(f"[Stop reason: {response.stop_reason}]")

    # Process tool calls in a loop
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


# ---------------------------------------------------------------------------
# Test questions
# ---------------------------------------------------------------------------

TESTS = [
    # Test 1: Bearing capacity (bearing_capacity agent)
    (
        "Calculate the ultimate bearing capacity of a 1.5m wide square footing "
        "at 1.0m depth in sand with phi=30 degrees and unit weight 18 kN/m3. "
        "Use the Vesic method. The water table is well below the footing."
    ),
    # Test 2: Multi-agent — bearing + settlement
    (
        "I have a 2.0m x 2.0m square footing at 1.5m depth on medium sand "
        "(phi=32, gamma=19 kN/m3, Es=25 MPa, Poisson's ratio=0.3). The applied "
        "load is 400 kN. First check the bearing capacity, then estimate the "
        "elastic settlement. Is the footing adequate?"
    ),
    # Test 3: Seismic — site classification + liquefaction (seismic_geotech agent)
    (
        "A site has an average SPT N-bar of 15 blows/ft in the upper 30m. "
        "Classify the site per AASHTO. Then check for liquefaction at 5m depth "
        "where total stress is 90 kPa, effective stress is 55 kPa, and the "
        "corrected N1_60 is 12. The design earthquake is Mw=7.5 with PGA=0.25g."
    ),
    # Test 4: Driven pile capacity (axial_pile agent)
    (
        "Estimate the axial capacity of a 0.356m (HP14x73) H-pile driven 15m "
        "into a profile with 10m of medium sand (phi=32, gamma=18.5 kN/m3) over "
        "5m of stiff clay (Su=100 kPa, gamma=19 kN/m3). Use the Nordlund method "
        "for sand and alpha method for clay."
    ),
    # Test 5: Slope stability (slope_stability agent)
    (
        "Analyze the stability of a 10m high slope with 2H:1V inclination using "
        "the Bishop method. The slope is uniform soft clay with Su=40 kPa and "
        "unit weight 18 kN/m3. Use a circular slip surface search."
    ),
    # Test 6: Retaining wall (retaining_walls agent)
    (
        "Design check a 5m high cantilever retaining wall with a 3m wide base "
        "(1.5m toe, 1.5m heel) and 0.4m thick stem. Backfill is granular with "
        "phi=30 degrees and gamma=18 kN/m3. The wall concrete has gamma=24 kN/m3. "
        "Check sliding, overturning, and bearing."
    ),
    # Test 7: Drilled shaft (drilled_shaft agent)
    (
        "Calculate the capacity of a 0.9m diameter drilled shaft, 12m deep. "
        "Upper 6m is stiff clay (Su=75 kPa, gamma=18 kN/m3), lower 6m is dense "
        "sand (phi=36, gamma=19 kN/m3). Water table is at 3m depth."
    ),
    # Test 8: Consolidation settlement (settlement agent)
    (
        "A 3m thick normally consolidated clay layer (Cc=0.35, e0=1.1, "
        "gamma=17 kN/m3) is at 5m depth below a proposed fill that adds 50 kPa. "
        "Estimate the primary consolidation settlement."
    ),
    # Test 9: Ground improvement — wick drains (ground_improvement agent)
    (
        "Design wick drains to accelerate consolidation of a 6m thick soft clay "
        "layer with ch=3.0 m2/yr. Target 90% consolidation in 6 months. "
        "Use a triangular drain pattern. What drain spacing is needed?"
    ),
    # Test 10: Soil classification (geolysis agent)
    (
        "Classify a soil sample with the following properties: liquid limit=45%, "
        "plastic limit=22%, percent passing #200 sieve=68%, percent passing #4=95%. "
        "Use the USCS classification system."
    ),
]


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    # Run all tests or just one (e.g., python test_run.py 3)
    if len(sys.argv) > 1:
        idx = int(sys.argv[1]) - 1
        run_single_question(TESTS[idx])
    else:
        for i, q in enumerate(TESTS):
            print(f"\n\n{'#'*70}")
            print(f"# TEST {i+1} of {len(TESTS)}")
            print(f"{'#'*70}")
            run_single_question(q)
