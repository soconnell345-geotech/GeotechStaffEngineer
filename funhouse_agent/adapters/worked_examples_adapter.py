"""Worked-examples adapter — validated published calculations as exemplars.

Gives the agent a two-step consult: ``find_worked_examples`` returns compact
summaries (cheap to scan), ``get_worked_example`` returns one full entry with
the dispatch calls, published answer, and report-writing notes. See
``funhouse_agent/worked_examples.py`` for the corpus itself.
"""

from funhouse_agent import worked_examples as _we
from funhouse_agent.adapters import require_params


def _find(params: dict) -> dict:
    require_params(params, ["topic"], method="find_worked_examples")
    hits = _we.search_examples(params["topic"],
                               domain=params.get("domain"),
                               limit=int(params.get("limit", 3)))
    return {
        "n_matches": len(hits),
        "examples": [_we.summarize(e) for e in hits],
        "note": ("Call get_worked_example with an id for the full entry "
                 "(problem, dispatch calls with parameters, published vs "
                 "computed results, report notes)." if hits else
                 "No matching worked example — proceed from the module briefs; "
                 "consider consulting the reference agent for the method basis."),
    }


def _get(params: dict) -> dict:
    require_params(params, ["example_id"], method="get_worked_example")
    entry = _we.get_example(params["example_id"])
    if entry is None:
        known = [e.get("id") for e in _we.load_examples()]
        return {"error": f"Unknown worked example id "
                         f"'{params['example_id']}'. Known ids: {known}"}
    return dict(entry)


METHOD_REGISTRY = {
    "find_worked_examples": _find,
    "get_worked_example": _get,
}

METHOD_INFO = {
    "find_worked_examples": {
        "category": "worked examples",
        "brief": ("Search validated worked examples from real published design "
                  "reports (FHWA GEC, Caltrans, AASHTO, UFC, Slide2/FLAC "
                  "verification). Returns compact summaries; fetch the full "
                  "entry with get_worked_example. Consult BEFORE a nontrivial "
                  "multi-step design or calc report — the entry shows the "
                  "method calls, parameters, published answer, and what a good "
                  "report presents."),
        "parameters": {
            "topic": {"type": "str", "required": True,
                      "brief": "Free-text topic, e.g. 'MSE wall external "
                               "stability' or 'rapid drawdown dam'."},
            "domain": {"type": "str", "required": False,
                       "brief": "Optional module-name filter, e.g. "
                                "'slope_stability', 'pavement_design'."},
            "limit": {"type": "int", "required": False,
                      "brief": "Max results (default 3)."},
        },
    },
    "get_worked_example": {
        "category": "worked examples",
        "brief": ("Fetch one worked example by id: self-contained problem "
                  "statement, ready-to-run dispatch calls with parameters, "
                  "published vs computed results, and report-writing notes."),
        "parameters": {
            "example_id": {"type": "str", "required": True,
                           "brief": "Id from find_worked_examples, e.g. "
                                    "'WE-SLOPE-5'."},
        },
    },
}
