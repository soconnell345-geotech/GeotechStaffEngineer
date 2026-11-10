# REF-64 (B): instructions a deployment appends to every agent and helper

Branch `claude/ref-64-instructions` from master `1caf160`. Plan of record:
`module_work/REFERENCE_PACKS_PLAN.md` section 3 (owner's decision 3.3,
2026-11-09: the built-in rules win). All test content is made up.

## 1. What was built

**Core (`aec_agent_core`, no GSE import):**

- `packages/aec-agent-core/aec_agent_core/deep/instructions.py` (new)
  - `ENV = "GEOTECH_INSTRUCTIONS"` (:38), `CAP_CHARS = 4000` (:44).
  - `PROTECTED_RULES` (:49): the protected list in 3.3. It cites the printed
    page, takes formulas and defaults only from a printed source, states
    assumptions and the method, says when a reading failed or is uncertain,
    and never invents a value.
  - `HEADING` (:62): the fixed heading, word for word from 3.3.
  - `Instructions` (:95): one read of the folder. Each `*.md` is read as
    UTF-8 and stripped. An empty file is ignored. A file that cannot be read
    goes to `file_errors`, and an unreadable folder goes to `error`.
  - `Instructions.block(role)` gives `all.md` and then `<role>.md`, joined by
    a blank line. Past 4,000 characters the text is cut at the last paragraph
    break that fits (`_cut`, :77; else the last line break, else 4,000). The
    block is `"\n\n" + PROTECTED_RULES + "\n\n" + HEADING + "\n\n" + text`.
    With no text for the role, it returns nothing.
  - `apply(system_prompt, subagents)` (:156) appends to the main prompt (role
    `main`) and to each helper spec's `system_prompt`, picked by
    `spec["name"]`.
    - Compiled helpers (`"runnable"` in the spec) are skipped.
    - A spec that changes is copied, so the caller's dicts are untouched.
    - It returns the prompt, the helpers and the record. The record is `{}`
      when the setting is unset.
  - `block_for(role)` (:187) and `merge_record(agent, record)` (:199) are for
    the compiled helpers, which a builder hands their prompt itself.
  - `InstructedEngine` (:215) wraps an engine and appends the block to
    `system` on every `complete()`. Every other attribute (reads and writes)
    goes to the inner engine, and `_inner` is the name the ingest's pacer
    follows through wrappers. `.appended` carries the role's record entry.
- `packages/aec-agent-core/aec_agent_core/deep/assembly.py`
  - `compile_agent` (:486-488) calls `instructions.apply` at its top. By
    then both pages have their final prompt and their complete helper list,
    including the general-purpose helper that `guard_helpers` adds. The
    record goes on the agent as `geotech_appended` (:512-516).
  - `record_build` (:549-552) copies it into `geotech_build["appended"]`
    when it is not empty.
  - Neither page's builder changed for this.

**GSE (`funhouse_agent/deep/agent.py`):**

- `_report_ingest_engine_factory` (:807-833): with a `report_ingest` block,
  the engine `engine_for` built is wrapped in `InstructedEngine`. The block
  goes on every call's system prompt, with all.md first.
  - It is wrapped here, not in `engine_for`, so `score_on_cluster` keeps
    measuring the shipped prompts.
  - Unset, the factory returns the engine it returned before.
- Report library (:1361-1371): `block_for("report_library")` is passed as
  `build_report_library_subagent(extra_system_prompt=...)`. That function
  already appends it last (`library_agent.py:939-940`).
- After `record_build` (:1429-1437): the two compiled helpers' records
  (`report_library` when the library is built, `report_ingest` when the
  ingest tool is) are merged into `geotech_build["appended"]`.
- The record rides `webapp.core.engine_record` (`core.py:2659-2662`) into
  every `turn_start` as `engine.build.appended`. A test pins that.

**Record shape** (plan 3.4):

```
{"folder": <basename>,
 "roles": {role: {"files", "chars", "sha", "cut", "text"}},
 "error"?, "file_errors"?}
```

- `chars` is the deployment text appended, after any cut.
- `sha` is the sha256 of the whole appended block (protected list, heading
  and text).
- `text` is its first 2,000 characters.
- The record is absent when the setting is unset. When the setting names a
  folder with no text, it is present with empty `roles`, which shows the
  folder was found.

**Docs (3.5):**

- `docs/FOUNDRY.md` (an `instructions/` folder beside the app file, set in
  the app file).
- `docs/DATABRICKS_INSTALL.md` §9 (a Workspace folder set before
  `run_on_databricks`).
- `webapp/README.md` §3 "Your deployment's own instructions" (roles,
  protected rules, cap, record, and that on a two-page host `main.md`
  reaches both pages' main agents).
- `tinyapps/TINYAPPS.md` (a folder in the wrapper repo, exported in `run.sh`
  or set as an app setting).
- An example folder with made-up content:
  `docs/examples/deployment_instructions/` (`all.md`, `main.md`,
  `references.md`, `calc.md` and a README the app does not read).
- All paths are generic.

## 2. Role names found

Grepped and confirmed by building both default pages with a file for every
role:

| Role | Where it is named | Page |
|---|---|---|
| `main` | the main agent (`compile_agent`'s prompt) | both |
| `references` | `funhouse_agent/deep/agent.py:275` | geotech |
| `calc` | `funhouse_agent/deep/agent.py:654` | geotech (not built on the review page today) |
| `reviewer` | `funhouse_agent/deep/agent.py:711` | geotech |
| `figure` | `aec_agent_core/deep/figure_helper.py:56` (`FIGURE_HELPER_NAME`) | both |
| `general-purpose` | geotech: deepagents' spec re-declared in `assembly.guard_helpers` (:339-341); review: the reading helper `_review_reader_subagent` (`funhouse_agent/review_build.py:89-107`), same name, so it replaces the stock helper | both |
| `model_setup` | `funhouse_agent/deep/setup_agent.py:144` (when enabled) | geotech |
| `report_library` | `report_ingest/library_agent.py:56` (compiled) | geotech |
| `report_ingest` | `report_ingest/subagent.py:72` (compiled) | geotech |

These match plan 3.1 exactly. No other helper exists today.

A build with a file for every role (`/tmp` folder, made-up text) gave these
roles:

- `geotech_default`: main, references, reviewer, calc, figure,
  general-purpose (helpers built: references, reviewer, calc, figure,
  general-purpose).
- `review_default`: main, figure, general-purpose (helpers built: figure,
  general-purpose).

## 3. Unchanged when unset

- `apply` returns early with the prompt and the same helper list (the same
  objects) and `{}` when `GEOTECH_INSTRUCTIONS` is unset or empty. Nothing is
  appended and no attribute is set. This is pinned by
  `test_instructions.py::test_unset_changes_nothing`.
- `webapp/tests/page_fingerprints.json` is **unchanged**:
  `git diff 1caf160 -- webapp/tests/page_fingerprints.json` is empty.
- `webapp/tests/test_page_fingerprints.py` passes against it (results in
  section 4).
- `test_deployment_instructions.py::test_unset_build_has_no_appended_record`
  builds both default pages unset. It finds no `appended` key in
  `geotech_build` and no heading in any prompt.
- With `references.md` alone, the fingerprint's only difference is the
  librarian's `system_prompt` hash
  (`test_the_only_fingerprint_difference_is_the_prompts`), as plan 3.6 asks.

## 4. Tests

New tests:

- `packages/aec-agent-core/tests/test_instructions.py` (8 tests):
  - unset;
  - each role gets all.md and then its own file, last, under the protected
    list and the heading;
  - a missing role file gives all.md alone;
  - a compiled helper is skipped;
  - the caller's specs are untouched;
  - the record's files, chars, sha and text;
  - the cap cuts at a paragraph and is recorded;
  - an unreadable folder appends nothing and the record names the error;
  - an unreadable file is skipped and named;
  - `block_for` and `merge_record`;
  - the engine wrapper appends to `system` and forwards everything else.
- `webapp/tests/test_deployment_instructions.py` (8 tests, through
  `page_builds`):
  - unset on both pages;
  - with a folder, every role on both pages gets its block last;
  - the fingerprint difference is only the prompt;
  - an unreadable folder still builds and is recorded;
  - report ingest's and the report library's blocks are recorded, the
    library's prompt ends with its block, and the ingest engine appends to
    every call;
  - `engine_record` (each `turn_start`'s `engine`) carries `build.appended`,
    and it is absent unset.

Runs (Python 3.12 venv,
`pip install -e packages/aec-agent-core -e ../geotech-references -e . pytest pytest-xdist`;
`-n 4`; exit codes from pytest itself):

| Suite | Result | Exit |
|---|---|---|
| `funhouse_agent/tests` | 4233 passed, 2 failed, 30 skipped | 1 (both failures are on master too, see below) |
| `funhouse_agent/deep/tests` | 951 passed, 4 skipped | 0 |
| `webapp/tests` (includes fingerprints, boundary, public text, docs currency and the new file) | 1297 passed, 5 skipped | 0 |
| `report_ingest/tests` | 4856 passed, 23 skipped | 0 |
| `packages/aec-agent-core/tests` (includes `test_core_imports`, the no-GSE-import check, and the new file) | 232 passed | 0 |
| `webapp/tests/test_review_boundary.py`, `test_page_fingerprints.py`, `test_public_text.py` on their own | 25 passed | 0 |

The two `funhouse_agent/tests` failures are
`test_calc_package_figure_conventions.py::test_large_pile_group_leaves_the_forces_to_the_table`
and `test_defaults_used.py::test_every_default_a_run_lists_changes_its_result[profile_figure|plot_data]`.

- They also fail on unmodified master (`origin/master` 1caf160, checked out
  in a separate worktree and imported from it with
  `PYTHONPATH=<worktree>:<worktree>/packages/aec-agent-core`): 2 failed,
  exit 1.
- Neither touches a file this branch changes (calc_package and
  profile_figure).

## 5. Noted, not fixed

- **The review page's `minimal` arm** (`GEOTECH_REVIEW_AGENT=minimal`, off
  by default) builds with `create_agent` (`deep/review_agent.py`), not
  through `compile_agent`, so it gets no block. This was checked by a build:
  two `create_agent` calls, no `appended`. The default pages are covered.
- **"Last in its prompt"** means last in the prompt our code hands
  deepagents. deepagents then adds its own base prompt and middleware
  sections (todos, files, `task`), and the `/memories/` AGENTS.md when memory
  is on, after it, as for every built-in rule.
- **Report ingest's own run record** (plan 3.2's "also listed in its own run
  record under `appended`") is not written. The block is in
  `geotech_build["appended"]` (so in every `turn_start`) and on the wrapper
  engine as `.appended`, but `run_ingest`'s run files do not copy it. This
  does not change a value in a saved report's output.
- **The diagnostics line** (plan 3.6: "deployment instructions: roles,
  characters, folder found") is not built. The brief's list did not name it,
  and the turn record carries the same facts.
- The protected list and the heading are added only when there is text, as
  the brief asks (plan 3.3 put the protected list in the built-in prompt;
  the brief's rule keeps the fingerprints byte-identical).

## 6. Release note (for HANDOFF)

**Deployment instructions (REF-64 (B)).**

- A deployment can add its own conventions to the agents: preferred methods
  cited from its reference packs, units, formats, naming.
- `GEOTECH_INSTRUCTIONS` names a folder:
  - `all.md` is appended to every agent and helper;
  - `<role>.md` is appended to one role: `main`, `references`, `reviewer`,
    `calc`, `figure`, `general-purpose`, `model_setup`, `report_library`,
    `report_ingest`.
- The text goes last in each prompt, under the built-in protected rules and
  a fixed heading saying it cannot switch them off.
- Guards in code are untouched.
- The cap is 4,000 characters per role, cut at a paragraph past that.
- An unreadable folder adds nothing and is recorded.
- Report ingest gets one block on every model call (not in
  `score_on_cluster`).
- Each `turn_start` records what each role got (`engine.build.appended`:
  files, characters, sha256, first 2,000 characters).
- Unset, the agents are byte-identical (fingerprints unchanged). No answer
  changes.
- Host setup: `docs/FOUNDRY.md`, `docs/DATABRICKS_INSTALL.md`,
  `webapp/README.md` §3, `tinyapps/TINYAPPS.md`.
- Example: `docs/examples/deployment_instructions/`.

## 7. Review fixes

This section covers an independent review of 44c04c6. It used probes with
made-up text: the prompt each model call really received during fake-model
turns through `start_turn_job`, plus the `activity.jsonl` they wrote. Its
verdict was MERGE AFTER FIXES. Each fix below is its own commit on
`merge/ref64b`.

**F1 (High), commit ed5bed4: both routes of each compiled helper.** With
ingest and the library on, the main agent reaches each helper two ways: the
primary tool of that name, and `task` to the sub-agent. Only the sub-agent
routes carried the block, yet the record listed both roles. The fix:
- `build_deep_agent` reads each block once (`funhouse_agent/deep/agent.py:1216`,
  `:1246`) and hands the same block to the tool and to the sub-agent.
- The record entries come from those same reads (`:1445`).
- `make_report_ingest_tool(instructions=)` wraps both engines
  (`funhouse_agent/deep/tools.py:776`, `:782`).
- `_report_ingest_engine_factory(instructions=)` no longer re-reads the folder
  on each ingest run.
- `make_report_library_tool(extra_system_prompt=)` hands the library block to
  `answer_question(extra_system_prompt=)` (`report_ingest/library_agent.py`).
  That passes it on to `build_report_library_graph`.
- New core helper: `instructions.instructed()` (`instructions.py:308`).
- Test: `test_both_routes_of_each_compiled_helper_get_its_block` covers all
  four routes. Each gets the block read at build; an edit made after the build
  does not reach them. The test fails on 44c04c6's tool code.

**F2 (Medium), commit 8f3bc1c: files that did nothing.** The record gains two
lists:
- `unused`: files with text that no role of the build takes. `apply()` counts
  main and every helper spec's name, compiled helpers included.
- `empty`: the empty files.

Before, a misspelt `reviwer.md` gave only `"roles": {}`. Now the geotech page
records `"unused": ["reviwer.md"]`, and the review page also lists
`reviewer.md`, because it has no reviewer. Tests cover the core and both
pages, and the docs now say so.

**F3 (Low), commit f413679: no host paths in the record.** `error` and
`file_errors` are now the exception's type plus its `strerror` alone
(`_reason`, `instructions.py:99`). Before, they carried the full folder path,
while `folder` kept only the basename. A non-UTF-8 test now runs on Windows
too.

**F4 (Low), commit 9727792: `prompt_chars`.** `compile_agent` keeps the
compiled main prompt's length when it appended something
(`assembly.py:517`), and `record_build` records that instead
(`assembly.py:557`). Unset, nothing changes.

**Plan item, built: the diagnostics line (commit 8f3f450).** The Connection
diagnostics gain a "deployment instructions" line, from
`instructions.describe()` (`instructions.py:232`) and
`webapp/diagnostics.py:154`. Both pages show it.
It gives:
- "none" when unset;
- otherwise the folder's name, then for `all.md` and each role file the files
  and characters, and whether the text was cut;
- a warning when the folder or a file cannot be read, or a file is empty.

**Plan item, not built: `appended` in the ingest's own run record.** It is
more than small. The ingest has no run record apart from
`report.record.json`. Putting `appended` there means a new `DocumentFacts`
field, which:
- lands in every saved record;
- is read by the library and the DIGGS writer;
- is pinned by the schema snapshots.

So it changes a saved report's output. The turn record now carries the
ingest's block correctly for both routes.

A related limit, seen in the code and not tested: a resumed ingest checks a
saved step against its `made_by` (for example the range labels' engine name
and prompt versions, `report_ingest/graph.py:1554`). `made_by` does not hold
the deployment's block, so an edit between two runs of the same report does
not by itself redo a saved step.

**After the fixes.** I ran the review's probes again, in the same
environment.
- **Unset:** every prompt on both pages (geotech, also with ingest and the
  library on), the report library's prompt and `prompt_chars` equal master
  1caf160's exactly.
- **Fingerprints:** `page_fingerprints.json` is unchanged.
- **Routes:** all four routes carry the block (ingest tool and sub-agent are
  each an `InstructedEngine`; library tool and sub-agent each pass 1 of 1
  model calls).
- **Turn record:** a fake-model turn's `turn_start` reads, on geotech,
  `"unused": ["model_setup.md", "reviwer.md"]` (model_setup is off by
  default), and `prompt_chars` 28,020, the compiled prompt. On review it reads
  `"unused": ["calc.md", "model_setup.md", "references.md",
  "report_ingest.md", "report_library.md", "reviewer.md", "reviwer.md"]`.
- **Test run:** one run, Python 3.14 venv, serial; no other pytest running,
  at least 3.7 GB free; exit code from pytest. GitHub runs the full suite on
  the merge.

  | Tests | Result | Exit |
  |---|---|---|
  | `packages/aec-agent-core/tests`, `webapp/tests/test_deployment_instructions.py`, `funhouse_agent/deep/tests/test_report_ingest_chat_model.py`, `test_report_ingest_offline.py`, `test_report_library_offline.py`, `report_ingest/tests/test_library_agent.py`, `test_package.py`, `test_subagent.py`, `webapp/tests/test_page_fingerprints.py`, `test_review_boundary.py`, `test_public_text.py`, `test_diagnostics.py`, `test_docs_currency.py` | 414 passed, 1 skipped (POSIX permissions, on Windows) | 0 |

- `~/.geotech_webapp` has the same listing before and after.

**Notes, unchanged:**
- "Last in its prompt" means last in the prompt our code hands deepagents.
  deepagents then adds its base prompt and tool sections, about 12,100
  characters after the block on the geotech main agent and about 2,500 on a
  helper.
- The minimal review arm still gets no block.
