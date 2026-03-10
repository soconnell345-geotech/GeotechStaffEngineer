"""
Run funhouse_agent test scenarios against a live AI engine.

Usage
-----
# With PrompterAPI (Databricks / Funhouse):
    python -m funhouse_agent.run_scenarios

# With Claude (local):
    python -m funhouse_agent.run_scenarios --engine claude

# Run a single scenario:
    python -m funhouse_agent.run_scenarios --scenario BC-01

# Run with verbose agent output:
    python -m funhouse_agent.run_scenarios --verbose

# Dry-run (list scenarios without running):
    python -m funhouse_agent.run_scenarios --dry-run

Output
------
- Console: pass/fail summary per question
- output/ directory: calc package HTML/PDF files
- funhouse_agent/scenario_results.json: full results log
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
SCENARIOS_PATH = SCRIPT_DIR / "test_scenarios.json"
RESULTS_PATH = SCRIPT_DIR / "scenario_results.json"
OUTPUT_DIR = SCRIPT_DIR.parent / "output"


def load_scenarios(path: Path = SCENARIOS_PATH) -> list:
    """Load test scenarios from JSON file."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["scenarios"]


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

def make_engine(engine_name: str):
    """Create an AI engine by name."""
    if engine_name == "claude":
        from funhouse_agent import ClaudeEngine
        return ClaudeEngine()
    elif engine_name == "prompter":
        # Import from funhouse environment
        try:
            from funhouse.prompter import PrompterAPI
            return PrompterAPI()
        except ImportError:
            print("ERROR: funhouse.prompter not available.")
            print("  Install the funhouse package or use --engine claude")
            sys.exit(1)
    else:
        print(f"ERROR: Unknown engine '{engine_name}'. Use 'claude' or 'prompter'.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Result checking
# ---------------------------------------------------------------------------

def check_result(result, checks: dict, prev_results: list) -> dict:
    """Evaluate an AgentResult against expected checks.

    Returns dict with pass/fail and details.
    """
    report = {"passed": True, "details": []}
    answer = result.answer.lower()

    # Check must_contain keywords
    if "must_contain" in checks:
        for keyword in checks["must_contain"]:
            if keyword.lower() not in answer:
                report["details"].append(f"MISSING keyword: '{keyword}'")
                report["passed"] = False
            else:
                report["details"].append(f"OK keyword: '{keyword}'")

    # Check value ranges (look for numbers in the answer)
    if "value_ranges" in checks:
        for key, (lo, hi) in checks["value_ranges"].items():
            # Try to find a number near the key name in the answer
            found = _extract_value_near_key(result.answer, key, lo, hi)
            if found is None:
                report["details"].append(
                    f"WARN value '{key}': could not extract (range [{lo}, {hi}])"
                )
            elif found < lo or found > hi:
                report["details"].append(
                    f"FAIL value '{key}' = {found:.2f} outside [{lo}, {hi}]"
                )
                report["passed"] = False
            else:
                report["details"].append(
                    f"OK value '{key}' = {found:.2f} in [{lo}, {hi}]"
                )

    # Check output file exists
    if "output_file" in checks:
        fpath = OUTPUT_DIR.parent / checks["output_file"]
        if fpath.exists():
            size = fpath.stat().st_size
            report["details"].append(
                f"OK file '{checks['output_file']}' exists ({size:,} bytes)"
            )
        else:
            report["details"].append(
                f"FAIL file '{checks['output_file']}' not found"
            )
            report["passed"] = False

    # Check tool usage
    if "expect_module" in checks:
        modules_called = {
            tc.get("arguments", {}).get("agent_name", "")
            for tc in result.tool_calls
            if tc.get("tool_name") == "call_agent"
        }
        expected = checks["expect_module"]
        if expected in modules_called:
            report["details"].append(f"OK module '{expected}' was called")
        else:
            report["details"].append(
                f"WARN module '{expected}' not in called modules: {modules_called}"
            )

    # Logic check is informational — human reviews
    if "logic" in checks:
        report["details"].append(f"REVIEW logic: {checks['logic']}")

    return report


def _extract_value_near_key(text: str, key: str, lo: float, hi: float):
    """Try to extract a numeric value from agent answer near a key name."""
    # Normalize key: q_ultimate_kPa -> ["q", "ultimate", "kpa"]
    parts = key.lower().replace("_", " ").split()

    # Look for numbers in the answer
    numbers = re.findall(r"[\d,]+\.?\d*", text)
    for num_str in numbers:
        try:
            val = float(num_str.replace(",", ""))
            # Accept if in a plausible engineering range
            if lo * 0.1 <= val <= hi * 10:
                return val
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_scenario(agent, scenario: dict, verbose: bool = False) -> dict:
    """Run a single scenario (multiple questions) and return results."""
    sid = scenario["id"]
    title = scenario["title"]
    print(f"\n{'='*70}")
    print(f"  {sid}: {title}")
    print(f"{'='*70}")

    questions = scenario["questions"]
    question_results = []
    all_passed = True

    for i, q in enumerate(questions):
        qnum = i + 1
        print(f"\n  Q{qnum}: {q['text'][:100]}{'...' if len(q['text']) > 100 else ''}")

        t0 = time.time()
        try:
            result = agent.ask(q["text"])
            elapsed = time.time() - t0
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERROR ({elapsed:.1f}s): {type(e).__name__}: {e}")
            question_results.append({
                "question": q["text"],
                "error": str(e),
                "elapsed_s": round(elapsed, 1),
                "passed": False,
            })
            all_passed = False
            continue

        # Run checks
        checks = q.get("checks", {})
        check_report = check_result(result, checks, question_results)

        # Print summary
        status = "PASS" if check_report["passed"] else "FAIL"
        if not check_report["passed"]:
            all_passed = False

        print(f"  [{status}] {result.rounds} rounds, {elapsed:.1f}s")
        for detail in check_report["details"]:
            prefix = "    "
            if detail.startswith("FAIL") or detail.startswith("MISSING"):
                prefix = "  X "
            elif detail.startswith("WARN") or detail.startswith("REVIEW"):
                prefix = "  ? "
            else:
                prefix = "  + "
            print(f"{prefix}{detail}")

        # Show truncated answer
        answer_preview = result.answer[:300].replace("\n", " ")
        print(f"  Answer: {answer_preview}{'...' if len(result.answer) > 300 else ''}")

        # Log tools used
        tools_used = [tc.get("tool_name", "?") for tc in result.tool_calls]
        print(f"  Tools: {tools_used}")

        question_results.append({
            "question": q["text"],
            "answer": result.answer,
            "tool_calls": result.tool_calls,
            "rounds": result.rounds,
            "elapsed_s": round(elapsed, 1),
            "checks": check_report,
            "passed": check_report["passed"],
        })

    return {
        "id": sid,
        "title": title,
        "passed": all_passed,
        "questions": question_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run funhouse_agent integration test scenarios"
    )
    parser.add_argument(
        "--engine", default="prompter", choices=["prompter", "claude"],
        help="AI engine backend (default: prompter)"
    )
    parser.add_argument(
        "--scenario", default=None,
        help="Run a single scenario by ID (e.g., BC-01)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose agent output (round-by-round)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List scenarios without running them"
    )
    parser.add_argument(
        "--max-rounds", type=int, default=15,
        help="Max ReAct loop rounds per question (default: 15)"
    )
    parser.add_argument(
        "--scenarios-file", type=str, default=None,
        help="Path to scenarios JSON file (default: test_scenarios.json)"
    )
    args = parser.parse_args()

    # Load scenarios
    scenarios_path = Path(args.scenarios_file) if args.scenarios_file else SCENARIOS_PATH
    scenarios = load_scenarios(scenarios_path)

    # Filter by ID if requested
    if args.scenario:
        scenarios = [s for s in scenarios if s["id"] == args.scenario]
        if not scenarios:
            print(f"ERROR: Scenario '{args.scenario}' not found.")
            sys.exit(1)

    # Dry run: just list
    if args.dry_run:
        print(f"{'ID':<12} {'Title':<45} {'Questions'}")
        print("-" * 70)
        for s in scenarios:
            print(f"{s['id']:<12} {s['title']:<45} {len(s['questions'])}")
        print(f"\nTotal: {len(scenarios)} scenarios, "
              f"{sum(len(s['questions']) for s in scenarios)} questions")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create engine and agent
    print(f"Engine: {args.engine}")
    engine = make_engine(args.engine)

    from funhouse_agent import GeotechAgent
    agent = GeotechAgent(
        genai_engine=engine,
        max_rounds=args.max_rounds,
        verbose=args.verbose,
    )

    # Run scenarios
    print(f"\nRunning {len(scenarios)} scenarios "
          f"({sum(len(s['questions']) for s in scenarios)} questions)...")

    all_results = []
    total_pass = 0
    total_fail = 0
    total_questions = 0
    t_start = time.time()

    for scenario in scenarios:
        # Reset agent between scenarios (fresh conversation)
        agent.reset()

        result = run_scenario(agent, scenario, verbose=args.verbose)
        all_results.append(result)

        for qr in result["questions"]:
            total_questions += 1
            if qr.get("passed"):
                total_pass += 1
            else:
                total_fail += 1

    total_time = time.time() - t_start

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    for r in all_results:
        status = "PASS" if r["passed"] else "FAIL"
        n_q = len(r["questions"])
        n_pass = sum(1 for q in r["questions"] if q.get("passed"))
        print(f"  [{status}] {r['id']}: {r['title']} ({n_pass}/{n_q} questions)")

    print(f"\n  Total: {total_pass}/{total_questions} questions passed")
    print(f"  Time: {total_time:.0f}s")

    # Check for output files
    output_files = list(OUTPUT_DIR.glob("*"))
    if output_files:
        print(f"\n  Output files ({len(output_files)}):")
        for f in sorted(output_files):
            print(f"    {f.name} ({f.stat().st_size:,} bytes)")

    # Save results
    results_data = {
        "engine": args.engine,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_questions": total_questions,
        "passed": total_pass,
        "failed": total_fail,
        "total_time_s": round(total_time, 1),
        "scenarios": all_results,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, default=str)
    print(f"\n  Results saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
