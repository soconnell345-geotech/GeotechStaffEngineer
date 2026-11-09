"""
Funhouse Agent — Engine-agnostic geotechnical agent with vision.

Self-contained dispatch: routes tool calls through internal adapters
directly to 50 modules (~901 methods). No dependency on foundry/ files.

Works with any AI backend satisfying the GenAIEngine protocol:
    - PrompterAPI (Databricks/Funhouse) — works natively
    - ClaudeEngine — Anthropic SDK adapter
    - Any future backend — implement chat() + analyze_image()

Usage:
    from funhouse_agent import GeotechAgent, ClaudeEngine

    # With PrompterAPI (Databricks)
    agent = GeotechAgent(genai_engine=prompter_api)

    # With Claude
    agent = GeotechAgent(genai_engine=ClaudeEngine())

    result = agent.ask("Calculate bearing capacity of 2m footing, phi=30")

The names below load on first use (a module ``__getattr__``), not when the
package is imported, so importing one of its modules (the vision tools,
the document tools, the switches) does not load the v1 agent, its engines
or the reviewers (BACKLOG INF-30, the app split's step 2).
"""

import importlib as _importlib
import importlib.util as _importlib_util

# The modules moved into the shared core (BACKLOG INF-32, the app split's
# step 11 (a); plan module_work/APP_SPLIT_PLAN.md section 2.3) keep their old
# names here until step 16: each old name IS the core's module (the same
# object, so a test that patches by the old name reaches the real module).
# Installed first, before anything below imports a moved name.
from aec_agent_core import install_aliases as _install_aliases

#: Old name -> the core's name. Lazy: nothing is imported until an old name
#: is looked up.
CORE_ALIASES = {
    "funhouse_agent._fileio": "aec_agent_core._fileio",
    "funhouse_agent.error_text": "aec_agent_core.error_text",
    "funhouse_agent.image_view": "aec_agent_core.image_view",
    "funhouse_agent.inline_store": "aec_agent_core.inline_store",
    "funhouse_agent.turn_question": "aec_agent_core.turn_question",
    "funhouse_agent.vision_probe": "aec_agent_core.vision_probe",
    "funhouse_agent.vision_tools": "aec_agent_core.vision_tools",
    "funhouse_agent.vision_view": "aec_agent_core.vision_view",
    "funhouse_agent.deep.databricks_bridge":
        "aec_agent_core.deep.databricks_bridge",
    "funhouse_agent.deep.inline_images": "aec_agent_core.deep.inline_images",
    "funhouse_agent.deep.pacing": "aec_agent_core.deep.pacing",
    "funhouse_agent.deep.vision_engine": "aec_agent_core.deep.vision_engine",
    # Step 11 (b): the figure helper, the plotting box and the chart stack
    # (plan 2.3, 2.8), with the two leaves plot_data's call goes through.
    # A package alias also covers its modules (funhouse_agent.charts.data_plot
    # IS aec_agent_core.charts.data_plot).
    "funhouse_agent.arg_checks": "aec_agent_core.arg_checks",
    "funhouse_agent.charts": "aec_agent_core.charts",
    "funhouse_agent.drawing_wait": "aec_agent_core.drawing_wait",
    "funhouse_agent.method_call": "aec_agent_core.method_call",
    "funhouse_agent.plot_box": "aec_agent_core.plot_box",
    "funhouse_agent.progress_notes": "aec_agent_core.progress_notes",
    "funhouse_agent.deep.figure_helper": "aec_agent_core.deep.figure_helper",
    # Step 11 (c): guards and records, Word and Excel, the checks, the
    # document tools' binding, the shared assembly (plan 2.3) and the review
    # suite's framework. funhouse_agent.review_eval stays GSE's package (its
    # tasks, its report fixture and its tests); only the three framework
    # modules under it are aliased.
    "funhouse_agent.cited_documents": "aec_agent_core.cited_documents",
    "funhouse_agent.cited_pages": "aec_agent_core.cited_pages",
    "funhouse_agent.coverage": "aec_agent_core.coverage",
    "funhouse_agent.document_tools": "aec_agent_core.document_tools",
    "funhouse_agent.docx_renderer": "aec_agent_core.docx_renderer",
    "funhouse_agent.dxf_notes": "aec_agent_core.dxf_notes",
    "funhouse_agent.measure_tool": "aec_agent_core.measure_tool",
    "funhouse_agent.office_text": "aec_agent_core.office_text",
    "funhouse_agent.review_flags": "aec_agent_core.review_flags",
    "funhouse_agent.run_mirror": "aec_agent_core.run_mirror",
    "funhouse_agent.runtime_check": "aec_agent_core.runtime_check",
    "funhouse_agent.safe_expr": "aec_agent_core.safe_expr",
    "funhouse_agent.scale_labels": "aec_agent_core.scale_labels",
    "funhouse_agent.xlsx_writer": "aec_agent_core.xlsx_writer",
    "funhouse_agent.deep.assembly": "aec_agent_core.deep.assembly",
    "funhouse_agent.deep.busy_retry": "aec_agent_core.deep.busy_retry",
    "funhouse_agent.deep.calculate_tool": "aec_agent_core.deep.calculate_tool",
    "funhouse_agent.deep.delivered_log": "aec_agent_core.deep.delivered_log",
    "funhouse_agent.deep.held_records": "aec_agent_core.deep.held_records",
    "funhouse_agent.deep.keep_work": "aec_agent_core.deep.keep_work",
    "funhouse_agent.deep.limits": "aec_agent_core.deep.limits",
    "funhouse_agent.deep.result_map": "aec_agent_core.deep.result_map",
    "funhouse_agent.deep.scratch_guard": "aec_agent_core.deep.scratch_guard",
    "funhouse_agent.deep.strict_args": "aec_agent_core.deep.strict_args",
    "funhouse_agent.deep.tool_guards": "aec_agent_core.deep.tool_guards",
    "funhouse_agent.deep.vision_kit": "aec_agent_core.deep.vision_kit",
    "funhouse_agent.deep.warm_imports": "aec_agent_core.deep.warm_imports",
    "funhouse_agent.review_eval.checks":
        "aec_agent_core.review_eval.checks",
    "funhouse_agent.review_eval.documents":
        "aec_agent_core.review_eval.documents",
    "funhouse_agent.review_eval.runner":
        "aec_agent_core.review_eval.runner",
}
_install_aliases(CORE_ALIASES)

# GSE fills the core's hooks for its own reference library (step 11 (a);
# aec_agent_core.hooks): the folders the read tools may also read, and the
# two readers of it. Strings, imported only when a hook is first asked for.
from aec_agent_core.hooks import set_hook as _set_hook

_set_hook("vision_tools.reference_read_roots",
          "funhouse_agent.reference_vision:reference_read_roots")
_set_hook("vision_tools.read_reference_figure",
          "funhouse_agent.reference_vision:_dispatch_read_reference_figure")
_set_hook("vision_tools.view_worked_example_source",
          "funhouse_agent.reference_vision:_dispatch_view_worked_example")


#: Each public name -> the module that defines it.
_LAZY = {
    "GenAIEngine": "funhouse_agent.engine",
    "ClaudeEngine": "funhouse_agent.engine",
    "NativeToolEngine": "funhouse_agent.engine",
    "PrompterBridgeEngine": "funhouse_agent.engine",
    "USING_SDK_ENGINES": "funhouse_agent.engine",
    "GeotechAgent": "funhouse_agent.agent",
    "AgentResult": "funhouse_agent.react_support",
    "make_seismic_reviewer": "funhouse_agent.reviewers",
    "make_seismic_reviewer_deep": "funhouse_agent.reviewers",
    "make_foundations_reviewer": "funhouse_agent.reviewers",
    "make_foundations_reviewer_deep": "funhouse_agent.reviewers",
    "make_earth_retention_reviewer": "funhouse_agent.reviewers",
    "make_earth_retention_reviewer_deep": "funhouse_agent.reviewers",
    "make_slope_fem_reviewer": "funhouse_agent.reviewers",
    "make_slope_fem_reviewer_deep": "funhouse_agent.reviewers",
    "make_pavement_specialist": "funhouse_agent.reviewers",
    "make_pavement_specialist_deep": "funhouse_agent.reviewers",
    "make_structural_specialist": "funhouse_agent.reviewers",
    "make_structural_specialist_deep": "funhouse_agent.reviewers",
    "disclaimer": "funhouse_agent._disclaimer",
}


def __getattr__(name):
    """A public name from its module, or a submodule by its name (as
    ``funhouse_agent.agent`` was reachable when this package imported it)."""
    module = _LAZY.get(name)
    if module is not None:
        value = getattr(_importlib.import_module(module), name)
        globals()[name] = value
        return value
    if not name.startswith("__"):
        try:
            spec = _importlib_util.find_spec(f"{__name__}.{name}")
        except (ImportError, ValueError):
            spec = None
        if spec is not None:
            return _importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))


__all__ = [
    "GeotechAgent",
    "GenAIEngine",
    "ClaudeEngine",
    "NativeToolEngine",
    "PrompterBridgeEngine",
    "USING_SDK_ENGINES",
    "AgentResult",
    "make_seismic_reviewer",
    "make_seismic_reviewer_deep",
    "make_foundations_reviewer",
    "make_foundations_reviewer_deep",
    "make_earth_retention_reviewer",
    "make_earth_retention_reviewer_deep",
    "make_slope_fem_reviewer",
    "make_slope_fem_reviewer_deep",
    "make_pavement_specialist",
    "make_pavement_specialist_deep",
    "make_structural_specialist",
    "make_structural_specialist_deep",
    "disclaimer",
]

# One-time (per user) professional-use disclaimer banner on first import. Fully
# self-guarding: stderr only, suppressed by GEOTECH_NO_DISCLAIMER=1 and under
# pytest, and it never raises. See funhouse_agent/_disclaimer.py.
try:
    from funhouse_agent._disclaimer import maybe_show_first_import_notice as _show_disclaimer

    _show_disclaimer()
except Exception:  # pragma: no cover - the notice must never break importing
    pass
