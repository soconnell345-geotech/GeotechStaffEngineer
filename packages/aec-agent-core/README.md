# aec-agent-core

The shared core of LLM agents that review and produce architecture,
engineering and construction (AEC) documents. It holds what an agent runs
below its chat page: the model connections, the vision calls and the file
tools beside them, the figure helper with its plotting box, the agent's
guards, limits and kept work, Word and Excel in and out, the run record and
its checks, and the switches. It has no user interface of its own.

Two applications are built on it:

- **GeotechStaffEngineer** (`pip install geotech-staff-engineer`), the
  geotechnical engineering agent, which pins this package exactly;
- **Document Review** (`pip install aec-document-review`), a document
  review application, which pins this package exactly.

## Status

**1.1.0 (not released yet; the first release with Document Review as one unit, BACKLOG INF-76).** One more module moved in:
`foundry_dataset_store`, the Foundry dataset store, from
GeotechStaffEngineer's `webapp` (BACKLOG INF-74), so that Document Review
can keep its conversations on Palantir Foundry the way the geotech page
does. On a Foundry deployment with the dataset's workspace alias set
(`GEOTECH_FOUNDRY_CONVERSATION_DATASET`), an application's conversation
store saves each conversation to that dataset after every turn and brings
past ones back. The folder inside the dataset is the name the application
handed in (`GeotechStaffEngineer` for the geotech app, as before;
`Document Review` for Document Review), unless `GEOTECH_SHAREPOINT_ROOT`
names another. GeotechStaffEngineer's old import path,
`webapp.foundry_dataset_store`, is an alias of the moved module.

**1.0.1 (2026-11-10).** Fixes from a deployment review of Document
Review (GeotechStaffEngineer's BACKLOG, TA-36): a control character in
text no longer loses a Word or Excel file, and Excel text that opens
with "=" is written as text; a file name that climbs out of the
working folder is not found; the primary agent's model-call budget is
noted on the turn, so a host does not start another pass with a new
one; a document-tools setting that is not a number falls back to its
default.

**1.0.0, the first release.** The app split's step 11 moved the
code in, in three merges (BACKLOG INF-32; plan
`module_work/APP_SPLIT_PLAN.md` in GeotechStaffEngineer's repository):

- **(a) model connections and vision calls:** `engine_config`,
  `palantir_sdk_engine`, `tinyapps_engine` (the Prompter connection),
  `http_trace`, `deep.databricks_bridge`, `deep.pacing`,
  `deep.vision_engine`, `deep.inline_images`, `vision_tools`,
  `vision_view`, `image_view`, `vision_probe`, `inline_store`,
  `turn_question`, `error_text`, `_fileio`;
- **(b) the figure helper:** `deep.figure_helper`, the plotting box
  (`plot_box`), the chart stack behind `plot_data` (`charts`) with its
  worked examples, `drawing_wait`, `progress_notes`, `arg_checks`,
  `method_call`;
- **(c) everything else both agents run:** the guards and records
  (`deep.keep_work`, `deep.limits`, `deep.tool_guards`,
  `deep.scratch_guard`, `deep.busy_retry`, `deep.delivered_log`,
  `deep.result_map`, `deep.strict_args`, `deep.warm_imports`,
  `deep.calculate_tool` with `safe_expr`, `deep.held_records`,
  `runtime_check`, `review_flags`); Word and Excel (`docx_renderer`,
  `xlsx_writer`, `office_text`); the run record and its checks
  (`activity_log`, `run_mirror`, `cited_pages`, `cited_documents`,
  `coverage`); the document tools' binding to `planlens.review`
  (`document_tools`, with `measure_tool`, `scale_labels`, `dxf_notes`);
  the tools' assembly (`deep.vision_kit`); the assembly both pages'
  agents share (`deep.assembly`: the guards round every helper, the
  general-purpose helper, the summarizer, one compile step); and the
  review suite's framework (`review_eval`: the runner, the checks, the
  `Task` type and task selection, document lookup).

The core imports nothing of either application. Where it needs something
only an application knows, it asks for a **hook**
(`aec_agent_core.hooks`), which the application fills:

| Hook | What the application hands in |
|---|---|
| `prompter.get_setting`, `prompter.resolve_for`, `prompter.tester_keys_possible` | how its settings are read, whose Prompter key serves a caller |
| `vision_tools.reference_read_roots`, `.read_reference_figure`, `.view_worked_example_source` | its reference library's folders and readers |
| `held_records.record_fields` | the shapes of the records a stopped helper held |
| `cited_documents.reference_modules` | its reference modules' names |
| `review_eval.page_host` | its chat page's turn path, for the review suite |
| `review_eval.open_tasks` | its review tasks |
| `review_eval.document_dirs`, `.fetch_document`, `.fixture` | where its tasks' documents are found |
| `app.details` | its name, distribution and package (`aec_agent_core.app_details.hand_in`): the run record's version, the default Word and markup authors, the runtime messages |

The chart lines `plot_data` and the plotting box can draw are handed in by a
registration, `aec_agent_core.charts.register_reference_charts`. Unfilled,
each place says what it does without the hook.

A deployment can add its own instructions to every agent and helper
(`deep.instructions`, applied by `deep.assembly` for both applications).
`GEOTECH_INSTRUCTIONS` names a folder: `all.md` goes to every agent and
helper, then one `<role>.md` per role (`main` for the main agent, else the
helper's name). Each role's text goes last in its prompt, under protected
rules it cannot switch off, at most 4,000 characters a role. The setting is
read from the environment only; unset, the prompts are unchanged.

`aec_agent_core.install_aliases` lets a module keep answering to its old
import path after it moved here, as the same module object (so an import or
a test patch by either name reaches the one copy of the code);
GeotechStaffEngineer keeps its old names this way until the split's step 16.

1.0.0 is the first real release (the split's step 12, on PyPI since
2026-11-10, published before the applications that pin it).
GeotechStaffEngineer's CI installs the core from its own repository
(`pip install -e packages/aec-agent-core`), so every change is tested with
the core beside it. Its "core-only" CI job installs this package with
planlens and nothing geotech, imports every module, runs these tests and
builds the document review page on it.

**Releases.** Since BACKLOG INF-76 this package and Document Review
(`aec-document-review`) release as one unit: one version number, and one
tag in GeotechStaffEngineer's repository, `aec-v<version>`, which publishes
both wheels, this one first (`publish-core.yml` and
`publish-document-review.yml`). Document Review pins this package at its
own version. GeotechStaffEngineer pins it exactly too, and its own release
(a `v<version>` tag) refuses while this package's folder has changed since
the tag of the version it pins: release the core first, then pin it. The
releases before (1.0.0, 1.0.1) had their own `core-v<version>` tags.

## Install

```
pip install aec-agent-core
pip install "aec-agent-core[anthropic]" # with Claude through LangChain
```

Python 3.11 or later. An application installs it for you: each pins the
core it was tested with.

## Licence

MIT, as GeotechStaffEngineer. See `LICENSE`.
