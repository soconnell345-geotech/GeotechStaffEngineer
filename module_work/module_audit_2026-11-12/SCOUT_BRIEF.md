# Module incompleteness scout: brief

You are a scout. The repository is `C:\dev\GeotechStaffEngineer` (main checkout, branch master).

## Hard rules
- READ ONLY on the repository. Another session is working in this checkout. Do not edit, create, stage, commit, stash, checkout or run any git command that changes anything. `git log` / `git grep` / `git show` are fine.
- Do NOT run pytest or any test. Do NOT import the package in Python (memory on this machine is short). Use Read, Grep, Glob. A tiny `python -c` using only the standard library (`ast`, `json`, `re`) is allowed.
- Stay out of `geotech-references/`, `packages/aec-document-review/`, `packages/aec-agent-core/aec_agent_core/review_eval/`.
- Do not start other agents.
- Write exactly one file: your result file (path given in your task). Nothing else anywhere.

## Why
The owner asked: "run through all modules for incompleteness issues, similar to those found during the geotech-references run." That run (read `module_work/refs_review_2026-11-10/REVIEW.md` sections 2 and 3, lines 50-131, first) found a library that told its user it had things it did not have, results that did not say what they were, silent edges, and tests that asserted the code's own values. You find the OBVIOUS gaps of that kind in the calculation modules, so that a deeper effort can be planned. Breadth and speed over depth. You do NOT check engineering values against printed manuals; a later stage does that.

## What to check, per module, in this order

1. **Told versus reachable.** What the agent is told the module can do, against what the code does.
   - The module's adapter is `funhouse_agent/adapters/<name>.py` or `<name>_adapter.py`; the registry is `funhouse_agent/adapters/__init__.py`. In the adapter, `METHOD_INFO` lists each method with its parameters and what it returns; a dispatch dict maps method names to functions.
   - For each method: is there a dispatch function; does every documented parameter reach the module's function, or is one dropped, renamed or never read; does the described return agree with the result class's `to_dict()` keys?
   - Reverse: what the module's `__init__.py` exports (public `analyze_*` functions and classes) that no adapter method reaches. List them.
   - Other places that make claims about the module: its `DESIGN.md` and README, CLAUDE.md's module inventory (grep the module name there; the file is huge, never read it whole), `docs/user_manual/manual_content.py`, `funhouse_agent/deep/prompt.py`. A claim with no code behind it is a finding.
   - A module with no registry entry at all (the agent cannot reach it) is a finding: say what it is and whether anything else uses it.
2. **Self-declared gaps.** In the module's code and DESIGN.md, search for: `NotImplementedError`, `TODO`, `FIXME`, `XXX`, "not implemented", "not yet", "placeholder", "stub", "future", "simplified", "approximat", "limitation", "does not handle", "not supported", "ignored", "assume". Report the ones that matter to a user, with what it means for them.
3. **Silent edges.** A parameter in a public signature that the body never reads. An option where only some values are implemented and the rest fall through. A default applied with nothing said in the result. A clamp or extrapolation with no warning. A broad `except` that returns None, 0 or NaN. A result that carries no units or does not say which method ran.
4. **Method coverage, as the module itself states it.** From DESIGN.md and docstrings: the methods, cases and references it says it covers. Note a missing case only when the code or docs make it obvious (a compression function with no tension counterpart; "layered soil not supported"; one of a manual's three methods present). Do not speculate past what you can point to.
5. **Outputs.** Does the module have: a calc package (grep `_package` in its adapter and in `calc_package/`); calculation tables in that package; a figure; `summary()` and `to_dict()` on every result class? List what is missing.
6. **Tests.** Is the module pinned to a published example (`validation_examples/INVENTORY.md`, `validation_examples/test_published_*.py`, or tests in the module citing a page or example number), or do its tests only assert the code's own numbers? Which public functions appear in no test (grep the name under the module's `tests/`)? Do not run anything.
7. **Already known.** Before writing a finding, grep `BACKLOG.md` and `BACKLOG_DONE.md` for the module name and the topic. If an OPEN item covers it, give the ID and mark it "known". If a Done entry says it was fixed, look at the code and say whether it agrees.

## Result file format

```
# <group name>

## <module>
Size: N python files, N public functions/classes exported, N adapter methods. One or two sentences of overall impression.

| # | Check | Sev | Finding | Evidence | Known? |
|---|---|---|---|---|---|
| 1 | 1 | A | one plain sentence | path:line | new / GM-123 open |

Not checked: (what you skipped and why)
```

- Check = the number 1-7 above.
- Sev: **A** = the agent is told something that fails, or the module silently does something other than what it says; **B** = a gap a user would hit (a method, case or output that is missing or half-built); **C** = a note.
- Every finding carries a `path:line` you opened yourself. No guesses. If you could not confirm, write "unverified" in the finding.
- At most 25 findings per module, the most important first. Plain engineering language, no invented jargon.
- `slope_stability` is under a change hold (BACKLOG GM-140): audit it like the rest; nothing gets changed anyway.

## Your final message
The path of your result file, then one line per module: counts of A, B, C and the single most important finding. Fifteen lines at most. No other text.
