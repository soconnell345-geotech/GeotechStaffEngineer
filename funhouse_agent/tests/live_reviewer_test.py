"""Live test of the reviewer agent with a real Claude API call.

Reads ANTHROPIC_API_KEY from Windows user environment variables.
Run with: .venv/Scripts/python funhouse_agent/tests/live_reviewer_test.py
"""

import os
import subprocess
import sys

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Pull the API key from Windows user env if not already in shell env
if not os.environ.get("ANTHROPIC_API_KEY"):
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command",
         "[Environment]::GetEnvironmentVariable('ANTHROPIC_API_KEY','User')"],
        capture_output=True, text=True,
    )
    key = result.stdout.strip()
    if key:
        os.environ["ANTHROPIC_API_KEY"] = key
        print(f"Loaded API key from Windows env: {key[:10]}...")
    else:
        print("ERROR: ANTHROPIC_API_KEY not found.")
        sys.exit(1)

from funhouse_agent import GeotechAgent, ClaudeEngine

print("\n" + "=" * 70)
print("LIVE REVIEWER TEST")
print("=" * 70)

engine = ClaudeEngine()
agent = GeotechAgent(genai_engine=engine, review=True, verbose=True)

question = (
    "Calculate the ultimate bearing capacity of a 2m wide strip footing "
    "at 1.5m depth in a cohesionless soil with phi=30 degrees and "
    "gamma=18 kN/m3. No water table."
)

print(f"\nQuestion: {question}\n")
print("-" * 70)

result = agent.ask(question)

print("-" * 70)
print(f"\nRounds: {result.rounds}")
print(f"Tool calls: {len(result.tool_calls)}")
print(f"Time: {result.total_time_s:.1f}s")
print(f"\n{'=' * 70}")
print("FINAL ANSWER:")
print("=" * 70)
print(result.answer)
