# GM-316 and GM-317: warnings reach the agent; `allowed_values` enforced

Branch `claude/gm-316-317-warnings-allowed-values`. Items: BACKLOG GM-316,
GM-317; module audit 2026-11-12 (`module_work/module_audit_2026-11-12/
AUDIT.md`) §2.1, §2.2 X1 and X2, §6 T1 (c) and T2. No print reference is
involved: both are dispatch-layer changes, and no formula, default or
coefficient changes.

## Piece 1, GM-316: a Python warning raised inside a method reaches the agent

### What changed

- New `packages/aec-agent-core/aec_agent_core/call_warnings.py`, used by
  `run_method` (`aec_agent_core/method_call.py`): the whole call (checks,
  handler, notes) runs inside `call_warnings.collecting()`, and the
  warnings collected are added to the result before working-folder paths
  are hidden. The old body of `run_method` is `_run_checked`, unchanged.
- One `warnings.showwarning` hook, installed on the first call. It
  appends to a `contextvars.ContextVar` list that `collecting()` sets, so
  calls running at once in threads keep their own; outside a call it hands
  each warning to the hook it replaced, unchanged. `catch_warnings` is not
  used round the call. When a `catch_warnings(record=True)` block (pytest
  puts one round every test) has swapped the hook out, the next call puts a
  new hook on top of whatever is there (the block puts its own back when it
  ends, so a hook installed earlier, e.g. `logging.captureWarnings`, is not
  lost). A hook that hands on to another of these hooks does not record
  twice.
- The core names no application: GSE hands in its calculation packages
  through the new hook `method_call.warning_packages`
  (`funhouse_agent/core_hooks.py: WARNING_PACKAGES`, filled in
  `funhouse_agent/__init__.py`; listed in `aec_agent_core/hooks.py`).

### The rule: which warnings reach the agent

A warning raised while a handler runs reaches the agent when

1. the code that issued it, or the code its `stacklevel` attributes it to,
   is in one of the app's calculation packages (GSE: the 34 calculation
   packages pyproject ships, `funhouse_agent.adapters` and
   `geotech_references`; a test checks the list against pyproject), or is
   `aec_agent_core.method_call` (a module's `stacklevel` one too high lands
   on the frame that called the handler); or
2. it is numpy's floating-point RuntimeWarning (`overflow`, `invalid
   value`, `divide by zero encountered in ...`), wherever it is attributed
   (scipy's internals included): it means a NaN or an infinity in a
   calculation;

and never when its category is DeprecationWarning, PendingDeprecationWarning,
FutureWarning, ImportWarning, ResourceWarning, SyntaxWarning, BytesWarning or
EncodingWarning, or when a plotting library issued it (matplotlib,
mpl_toolkits, plotly, PIL, kaleido, seaborn: matplotlib's `warn_external`
attributes its layout warnings to the caller's line, so attribution alone
would let them in). Anything else from a third-party library, attributed to
that library, stays on stderr only.

Improvement on the brief's starting point: the rule looks at the issuing
frame as well as the attributed one, so a scipy warning raised with
`stacklevel=2` against our own call (e.g. a fit that could not estimate its
covariance) does reach the agent (it is about our calculation), while
matplotlib's, attributed the same way, does not.

**Repeats.** Python's default action shows a warning once per location
(`__warningregistry__`). For a warning that reaches the agent, the hook takes
its key `(text, category, lineno)` back out of the registry of the module it
is attributed to, so the next call raising it reaches the hook again. No
filter is added and no other registry is touched, so third-party warnings
keep their once-per-location behaviour. Within one call, a warning goes on
to the replaced hook (stderr) once per location; on a later call it is shown
there again (before: once per process until the filters next changed).

**What the agent gets.** The message text only (no file path, no line
number); at most 20 a call, then "and N more". A dict result gets them under
`warnings`, after the entries the module already returns there, with no
duplicates (a module that both warns and lists the same text gets it once).
The `warnings` key is moved to the FRONT of the dict when anything is added:
a reference module's result is cut at its tail for size
(`funhouse_agent/deep/tools.py` `_result_cap_for_module` -> `_truncate`;
calc modules are uncapped) and a result of 60,000 characters or more is
shown to the model as a key map whose preview is its first 1,000 characters
(`aec_agent_core/deep/result_map.py`); at the front, the warnings survive
both. An error result (`{"error": ...}`) also carries the warnings raised
before the error. **A result that is not a dict is returned unchanged**:
no `call_agent` method returns one today (every probe returns a dict), and a
library caller of `run_method` that gets a list keeps a list rather than
finding it wrapped.

### Consumers checked

- `report_ingest/recompute.py:1854, 2356` copies a module result's
  `warnings` to the row's `module_warnings`: it now also holds the Python
  warnings the module raised, which is what the name says.
- `funhouse_agent/chart_eval/arms.py:370` prints a result's warnings: same.
- `funhouse_agent/tests/test_defaults_used.py` leaves `warnings` out of its
  comparisons (`_VOLATILE`), so a warning does not make an input look
  moved.
- The calc packages' METHOD_INFO says their `warnings` are "also printed in
  the package": a Python warning raised while a package runs is now added
  to that list but is not printed in the PDF. Noted, not fixed (the
  package's own list is still printed; no probe raises one).
- `funhouse_agent/deep/prompt.py` and the core's prompts say nothing about
  warnings, so no prompt text changed (page fingerprints unchanged).
- The webapp reads no tool result's `warnings`.

### Probes that now carry warnings

None. All 251 probes in `method_probes.json` were run with every warning
recorded (`simplefilter("always")`) before and after the change: they raise
no Python warning at all (they are the smallest happy-path inputs). So the
standing test (T2, `funhouse_agent/tests/test_probe_warnings.py`) also runs
four made-up calls that do raise the module warnings the audit named, each
twice and under both the "always" and the default filter:

| Call (made-up inputs) | Warning the agent now sees |
|---|---|
| axial_pile `axial_pile_capacity`, pile 40 m in a 30 m profile | "Pile tip (head_depth 0.0m + pile_length 40m) exceeds soil profile (30m); using full profile depth" |
| drilled_shaft `drilled_shaft_capacity`, length 20 m in 15 m | "Shaft length (20.0m) exceeds soil profile (15m); using full profile depth" |
| lateral_pile `lateral_pile_analysis`, soil to 10 m, pile 15 m | "Soil layers extend to 10 m but pile length is 15.0 m. No soil resistance will be applied below 10 m." |
| settlement `consolidation_settlement`, Cr 0.3 > Cc 0.05 | "Cr (0.3) > Cc (0.05); typically Cr < Cc" |

Each of these is a module defect the audit lists for wave 2 (a profile
shorter than the pile should be refused or stated, not extended); they are
not fixed here.

### Noted, not fixed

- `lateral_pile/solver.py:376` and `drilled_shaft/capacity.py:1867` put
  `catch_warnings()` + `simplefilter("ignore")` round code inside a
  handler, and `aec_agent_core/charts/layout_ink.py:1196, 1211` does the
  same round matplotlib's layout (under a lock that only orders the
  layout_ink callers). While any of those blocks is open, the "ignore" filter
  is process-wide, so a warning another thread's call raises in that window
  is dropped before any hook sees it (as it already was from stderr). The
  layout_ink catch still works as before (test
  `test_the_lock_based_catch_in_layout_ink_still_works`).
- A handler that starts worker threads of its own (a thread pool inside a
  module) runs them without the call's context, so their warnings stay on
  stderr only. No module does that through a probe today.
- If a warning was raised OUTSIDE any call earlier in the process (a
  library caller using a module directly) with the same text at the same
  line, Python's registry suppresses the next occurrence before the hook
  sees it, until the filters next change (any `catch_warnings` or
  `filterwarnings` clears every registry; the layout_ink catch does so on
  every figure).
- The very first installation of the hook, if it happens while another
  thread is inside a `catch_warnings` block, is undone when that block
  ends; the next call installs it again.
