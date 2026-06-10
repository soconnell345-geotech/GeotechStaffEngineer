"""Triage tool — turn an agent suite-results JSON into per-domain work orders.

Usage:
    python module_work/triage_feedback.py <results.json> [out.json]

Reads a results file, classifies each record's errors, groups by domain +
module, prints a compact summary, and writes a machine-readable
module_feedback.json the team lead drops onto the BOARD / ledgers.

Accepts BOTH result formats (autodetected — see :func:`extract_records`):

* **v2 (current)** — the deepagents harness output
  (``funhouse_agent.deep.eval_harness``):

  - ``run_suite``: ``{"model", "n", "results": [...], "metrics", ...}``
  - ``run_ab``: ``{"v1": {"results": [...]}, "v2": {"results": [...]}, ...}``
    (the **v2 side** is triaged)

  v2 records are ``QAResult`` dicts: ``{qid, module, rounds,
  trace: [{name, args, errored, note}], errors: [{tool, type, message}],
  exception}``. The tool sequence is derived from the trace
  (``agent_name.method`` per call, like the suite markdown review).

* **v1 (legacy)** — ``docs/geotech_test_suite_results.json``: a flat JSON
  LIST of records ``{id, module, rounds, tool_sequence,
  errors: [{message, ...}], exception}``.
"""
import json
import re
import sys
from collections import defaultdict

DOMAINS = {
    "foundations": ["bearing_capacity", "settlement", "retaining_walls",
                    "ground_improvement", "downdrag"],
    "deep-foundations": ["axial_pile", "drilled_shaft", "lateral_pile",
                         "pile_group", "wave_equation"],
    "earth-retention": ["sheet_pile", "soe"],
    "slope-fem": ["slope_stability", "fem2d"],
    "seismic": ["seismic_geotech", "pystrata_agent", "opensees_agent",
                "liquepy_agent", "seismic_signals_agent"],
    "characterization": ["hvsrpy_agent", "swprocess_agent", "gstools_agent",
                         "subsurface_characterization", "salib_agent", "pystra_agent"],
    "io-cad": ["dxf_import", "dxf_export", "pdf_import"],
    "references": ["dm7",
                   "gec4", "gec5", "gec6", "gec7", "gec8", "gec9",
                   "gec10", "gec11", "gec12", "gec13", "gec14",
                   "micropile",
                   "ufc_backfill", "ufc_expansive", "ufc_pavement",
                   "fema_p2082", "california_trenching", "fhwa_pavements",
                   "reference_db", "figure_db"],
}
MOD_TO_DOMAIN = {m: d for d, mods in DOMAINS.items() for m in mods}


def classify(msg: str) -> str:
    m = msg.lower()
    if "unknown method" in m:
        return "ergonomics:method-name"      # guessed a method that doesn't exist
    if "unknown module" in m:
        return "ergonomics:module-name"
    if "must be" in m or ("got '" in m) or "allowed" in m or "one of" in m:
        return "ergonomics:enum-value"        # invalid enumerated value
    if "keyerror" in m:
        return "ergonomics:param-name"        # missing/misnamed parameter
    if "does not intersect" in m or "crossing" in m:
        return "input-geometry"               # bad trial geometry (slope)
    if "valueerror" in m or "typeerror" in m:
        return "possible-bug"
    return "other"


# ---------------------------------------------------------------------------
# Format autodetection / normalization (v1 list  |  v2 run_suite / run_ab)
# ---------------------------------------------------------------------------

def _trace_to_tool_sequence(trace):
    """Compact ``agent.method`` labels from a v2 QAResult trace.

    Mirrors the eval harness's markdown ``_trace_summary``: a ``call_agent`` /
    ``describe_method`` / ``list_methods`` call renders as
    ``<agent_name>.<method>``; any other tool by its bare name; an errored
    call gets an ``[errored]`` suffix.
    """
    seq = []
    for tc in trace or []:
        if not isinstance(tc, dict):
            continue
        name = tc.get("name", "?")
        args = tc.get("args", {}) or {}
        if name in ("call_agent", "describe_method", "list_methods") and isinstance(args, dict):
            agent_name = args.get("agent_name") or args.get("agent") or ""
            method = args.get("method") or ""
            label = ".".join(p for p in (agent_name, method) if p) or name
        else:
            label = name
        if tc.get("errored"):
            label += " [errored]"
        seq.append(label)
    return seq


def normalize_record(r: dict) -> dict:
    """Normalize one result record (v1 or v2) to the triage shape.

    Returns ``{id, module, rounds, tool_sequence, errors, exception}`` —
    ``errors`` stays a list of dicts carrying a ``message`` key (both formats
    already do).
    """
    if "qid" in r:  # v2 QAResult dict
        return {
            "id": r.get("qid"),
            "module": r.get("module", "?"),
            "rounds": r.get("rounds"),
            "tool_sequence": _trace_to_tool_sequence(r.get("trace")),
            "errors": r.get("errors") or [],
            "exception": r.get("exception"),
        }
    # v1 record — already in the triage shape.
    return {
        "id": r.get("id"),
        "module": r.get("module", "?"),
        "rounds": r.get("rounds"),
        "tool_sequence": r.get("tool_sequence"),
        "errors": r.get("errors") or [],
        "exception": r.get("exception"),
    }


def extract_records(data, agent: str = "v2") -> list:
    """Pull the record list out of any supported results payload.

    * JSON **list** → v1 legacy flat records.
    * Dict with ``"results"`` → v2 ``run_suite`` output.
    * Dict with ``"v1"``/``"v2"`` sides → v2 ``run_ab`` output; ``agent``
      picks the side (default the v2 agent's run).

    Every record is normalized via :func:`normalize_record`.
    """
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict) and isinstance(data.get("results"), list):
        records = data["results"]
    elif isinstance(data, dict) and isinstance(data.get(agent), dict):
        records = data[agent].get("results") or []
    else:
        raise ValueError(
            "Unrecognized results format: expected a v1 list, a run_suite "
            "dict with 'results', or a run_ab dict with 'v1'/'v2' sides."
        )
    return [normalize_record(r) for r in records if isinstance(r, dict)]


# ---------------------------------------------------------------------------
# Triage (pure — testable without file I/O)
# ---------------------------------------------------------------------------

def triage(records: list) -> tuple:
    """Classify + group normalized records.

    Returns ``(feedback, stats)`` where ``feedback`` is the per-domain
    per-module dict written to module_feedback.json and ``stats`` is
    ``{"n", "clean", "soft", "hard"}``.
    """
    per_mod = defaultdict(lambda: {"domain": None, "n_q": 0, "n_err_q": 0,
                                   "errors": [], "classes": defaultdict(int),
                                   "questions": []})
    clean = soft = hard = 0
    for r in records:
        mod = r.get("module", "?")
        dom = MOD_TO_DOMAIN.get(mod, "unassigned")
        pm = per_mod[mod]
        pm["domain"] = dom
        pm["n_q"] += 1
        errs = r.get("errors") or []
        exc = r.get("exception")
        if exc:
            hard += 1
        elif errs:
            soft += 1
        else:
            clean += 1
        if errs or exc:
            pm["n_err_q"] += 1
        pm["questions"].append({
            "id": r.get("id"), "rounds": r.get("rounds"),
            "tools": r.get("tool_sequence"), "n_errors": len(errs),
            "exception": (str(exc).splitlines()[0] if exc else None),
        })
        for e in errs:
            msg = str(e.get("message", "")).strip() if isinstance(e, dict) else str(e)
            cls = classify(msg)
            pm["classes"][cls] += 1
            short = re.sub(r"\s+", " ", msg)[:160]
            if short not in pm["errors"]:
                pm["errors"].append(short)

    feedback = {}
    for mod, pm in sorted(per_mod.items()):
        feedback.setdefault(pm["domain"], {})[mod] = {
            "n_questions": pm["n_q"], "n_with_errors": pm["n_err_q"],
            "classes": dict(pm["classes"]), "errors": pm["errors"],
            "questions": pm["questions"],
        }
    stats = {"n": len(records), "clean": clean, "soft": soft, "hard": hard}
    return feedback, stats


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    path = argv[0] if len(argv) > 0 else "docs/geotech_test_suite_results.json"
    out = argv[1] if len(argv) > 1 else "module_work/module_feedback.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    records = extract_records(data)
    feedback, stats = triage(records)

    print(f"records: {stats['n']}  |  clean {stats['clean']} · "
          f"recovered {stats['soft']} · failed {stats['hard']}\n")
    for dom in list(DOMAINS) + ["unassigned"]:
        mods = feedback.get(dom)
        if not mods:
            continue
        dom_err = sum(pm["n_with_errors"] for pm in mods.values())
        print(f"=== {dom}  ({dom_err} questions with errors) ===")
        for mod, pm in sorted(mods.items()):
            if pm["n_with_errors"] == 0:
                print(f"  [clean] {mod}  ({pm['n_questions']} q)")
                continue
            cls = ", ".join(f"{k}×{v}" for k, v in sorted(pm["classes"].items()))
            print(f"  [{pm['n_with_errors']}/{pm['n_questions']}] {mod}  -> {cls}")
            for em in pm["errors"]:
                print(f"        - {em}")
        print()

    with open(out, "w", encoding="utf-8") as f:
        json.dump(feedback, f, indent=2, default=str)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
