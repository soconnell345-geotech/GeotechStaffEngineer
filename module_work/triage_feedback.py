"""Triage tool — turn a geotech_test_suite_results.json into per-domain work orders.

Usage:
    python module_work/triage_feedback.py <results.json> [out.json]

Reads the full results file, classifies each record's errors, groups by domain +
module, prints a compact summary, and writes a machine-readable module_feedback.json
the team lead drops onto the BOARD / ledgers.
"""
import json
import re
import sys
from collections import defaultdict

DOMAINS = {
    "foundations": ["bearing_capacity", "settlement", "retaining_walls",
                    "ground_improvement", "downdrag", "wind_loads"],
    "deep-foundations": ["axial_pile", "drilled_shaft", "lateral_pile",
                         "pile_group", "wave_equation"],
    "earth-retention": ["sheet_pile", "soe"],
    "slope-fem": ["slope_stability", "fem2d", "fdm2d"],
    "seismic": ["seismic_geotech", "pystrata_agent", "opensees_agent",
                "liquepy_agent", "seismic_signals_agent", "pyseismosoil_agent"],
    "characterization": ["geolysis", "pygef_agent", "ags4_agent", "pydiggs_agent",
                         "hvsrpy_agent", "swprocess_agent", "gstools_agent",
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


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "docs/geotech_test_suite_results.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "module_work/module_feedback.json"
    records = json.load(open(path, encoding="utf-8"))

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
            "exception": (exc.splitlines()[0] if exc else None),
        })
        for e in errs:
            msg = str(e.get("message", "")).strip()
            cls = classify(msg)
            pm["classes"][cls] += 1
            short = re.sub(r"\s+", " ", msg)[:160]
            if short not in pm["errors"]:
                pm["errors"].append(short)

    # group by domain
    by_dom = defaultdict(list)
    for mod, pm in per_mod.items():
        by_dom[pm["domain"]].append((mod, pm))

    print(f"records: {len(records)}  |  clean {clean} · recovered {soft} · failed {hard}\n")
    feedback = {}
    for dom in list(DOMAINS) + ["unassigned"]:
        mods = sorted(by_dom.get(dom, []))
        if not mods:
            continue
        dom_err = sum(pm["n_err_q"] for _, pm in mods)
        print(f"=== {dom}  ({dom_err} questions with errors) ===")
        for mod, pm in mods:
            feedback.setdefault(dom, {})[mod] = {
                "n_questions": pm["n_q"], "n_with_errors": pm["n_err_q"],
                "classes": dict(pm["classes"]), "errors": pm["errors"],
                "questions": pm["questions"],
            }
            if pm["n_err_q"] == 0:
                print(f"  [clean] {mod}  ({pm['n_q']} q)")
                continue
            cls = ", ".join(f"{k}×{v}" for k, v in sorted(pm["classes"].items()))
            print(f"  [{pm['n_err_q']}/{pm['n_q']}] {mod}  -> {cls}")
            for em in pm["errors"]:
                print(f"        - {em}")
        print()

    json.dump(feedback, open(out, "w", encoding="utf-8"), indent=2, default=str)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
