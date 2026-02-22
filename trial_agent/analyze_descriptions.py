"""
Analyze describe_method output size for every method across all 30 agents.

Measures character count, parameter count (required vs optional), and
identifies which fields contribute the most to response size.
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trial_agent.agent_registry import list_methods, describe_method, list_agents


def _extract_method_names(result) -> list:
    """Extract method names from any list_methods return format."""
    names = []

    if isinstance(result, list):
        # Direct list of method names or dicts
        for item in result:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, dict) and "name" in item:
                names.append(item["name"])
        return names

    if not isinstance(result, dict):
        return names

    # Check for explicit "methods" key
    if "methods" in result:
        methods = result["methods"]
        if isinstance(methods, list):
            for m in methods:
                if isinstance(m, dict) and "name" in m:
                    names.append(m["name"])
                elif isinstance(m, str):
                    names.append(m)
            if names:
                return names

    # Walk all values — categories map to lists or dicts of methods
    for key, val in result.items():
        if key in ("count", "error", "agent"):
            continue
        if isinstance(val, list):
            for m in val:
                if isinstance(m, str):
                    names.append(m)
                elif isinstance(m, dict) and "name" in m:
                    names.append(m["name"])
        elif isinstance(val, dict):
            # Category -> {method_name: description} format
            for method_name, desc in val.items():
                if isinstance(desc, str):
                    names.append(method_name)
                elif isinstance(desc, dict) and "name" in desc:
                    names.append(desc["name"])
        elif isinstance(val, str) and key not in ("count", "error"):
            # Single method as key: description
            names.append(key)

    return names


def analyze_all():
    agents = list_agents()
    rows = []
    agent_totals = {}

    for agent_name in sorted(agents.keys()):
        # Get all methods
        methods_result = list_methods(agent_name, "")
        if isinstance(methods_result, dict) and "error" in methods_result:
            print(f"ERROR listing {agent_name}: {methods_result['error']}")
            continue

        # Extract method names — agents return varied formats
        method_names = _extract_method_names(methods_result)

        if not method_names:
            print(f"WARNING: Could not extract methods from {agent_name}")
            continue

        agent_total_chars = 0
        agent_method_count = 0

        for method_name in method_names:
            desc = describe_method(agent_name, method_name)
            desc_str = json.dumps(desc)
            char_count = len(desc_str)

            # Count parameters
            params = desc.get("parameters", desc.get("params", []))
            if isinstance(params, list):
                n_params = len(params)
                n_required = sum(1 for p in params if isinstance(p, dict) and p.get("required", False))
                n_optional = n_params - n_required
            elif isinstance(params, dict):
                n_params = len(params)
                n_required = sum(1 for v in params.values() if isinstance(v, dict) and v.get("required", False))
                n_optional = n_params - n_required
            else:
                n_params = 0
                n_required = 0
                n_optional = 0

            # Measure field sizes
            field_sizes = {}
            for key, val in desc.items():
                field_sizes[key] = len(json.dumps(val))

            rows.append({
                "agent": agent_name,
                "method": method_name,
                "total_chars": char_count,
                "est_tokens": char_count // 4,
                "n_params": n_params,
                "n_required": n_required,
                "n_optional": n_optional,
                "field_sizes": field_sizes,
            })

            agent_total_chars += char_count
            agent_method_count += 1

        agent_totals[agent_name] = {
            "methods": agent_method_count,
            "total_chars": agent_total_chars,
            "avg_chars": agent_total_chars // max(agent_method_count, 1),
        }

    # Print summary by agent
    print("\n" + "=" * 90)
    print(f"{'AGENT':<22} {'METHODS':>7} {'TOTAL CHARS':>12} {'AVG CHARS':>10} {'AVG TOKENS':>10}")
    print("=" * 90)
    grand_total = 0
    grand_methods = 0
    for name in sorted(agent_totals.keys()):
        t = agent_totals[name]
        grand_total += t["total_chars"]
        grand_methods += t["methods"]
        print(f"{name:<22} {t['methods']:>7} {t['total_chars']:>12,} {t['avg_chars']:>10,} {t['avg_chars']//4:>10,}")
    print("-" * 90)
    print(f"{'TOTAL':<22} {grand_methods:>7} {grand_total:>12,} {grand_total//max(grand_methods,1):>10,} {grand_total//max(grand_methods,1)//4:>10,}")

    # Print top 20 largest methods
    rows.sort(key=lambda r: r["total_chars"], reverse=True)
    print("\n" + "=" * 100)
    print("TOP 30 LARGEST describe_method RESPONSES")
    print(f"{'AGENT':<22} {'METHOD':<35} {'CHARS':>7} {'TOKENS':>7} {'PARAMS':>6} {'REQ':>4} {'OPT':>4}")
    print("=" * 100)
    for r in rows[:30]:
        print(f"{r['agent']:<22} {r['method']:<35} {r['total_chars']:>7,} {r['est_tokens']:>7,} {r['n_params']:>6} {r['n_required']:>4} {r['n_optional']:>4}")

    # Print field size breakdown for top 5
    print("\n" + "=" * 100)
    print("FIELD SIZE BREAKDOWN — TOP 5 LARGEST METHODS")
    print("=" * 100)
    for r in rows[:5]:
        print(f"\n  {r['agent']}.{r['method']} ({r['total_chars']:,} chars)")
        for field, size in sorted(r["field_sizes"].items(), key=lambda x: -x[1]):
            pct = 100 * size / r["total_chars"]
            print(f"    {field:<25} {size:>6,} chars  ({pct:5.1f}%)")

    # Print smallest 10 for comparison
    print("\n" + "=" * 100)
    print("SMALLEST 10 describe_method RESPONSES")
    print(f"{'AGENT':<22} {'METHOD':<35} {'CHARS':>7} {'TOKENS':>7} {'PARAMS':>6}")
    print("=" * 100)
    for r in rows[-10:]:
        print(f"{r['agent']:<22} {r['method']:<35} {r['total_chars']:>7,} {r['est_tokens']:>7,} {r['n_params']:>6}")


if __name__ == "__main__":
    analyze_all()
