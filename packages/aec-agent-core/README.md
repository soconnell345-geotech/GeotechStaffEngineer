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

**1.3.0 (2026-11-18).** Every change in what an agent is offered, shown
or told is behind a setting that is off unless set: eight new ones, none
measured on a model yet (planlens' plan tools, its drawing tool and a DXF
opened as a plot, each through the host layer; worked examples for the
figure helper; a lean system prompt; an uploaded picture shown to the
main model; pixel boxes asked of a vision call; a kernel layer under the
plotting box). On from the start are repairs (a figure imported a second
time keeps its chart; a reading helper names a close-up it failed to
read; an image file is measured in its own pixels; a document's file
name is not masked as a key) and one change that is more than a repair:
when the summary of a long conversation fails, the turn goes on with the
whole history instead of ending on the provider's error. The cited-pages
and cited-documents checks, still off by default, fire less often on
what is not a citation. The drawing tool, the DXF plot and sheet numbers
as addresses need planlens 0.17.0; on 0.16.0 those settings change
nothing. The release note is `module_work/aec_1_3_0/RELEASE_NOTE.md` in
the repository.

**1.2.0 (2026-11-18).** Word out: `write_docx` stores a photograph as a
JPEG at print size, so a report of 20 photographs is about 4 to 13 MB
(line work, lettering and most charts and scans are stored as before; a chart whose pixels look like a photograph, such as a shaded heat map, becomes a full-colour JPEG). Word in: characters
from the Symbol font are read, equations in Word's own format are counted
as not read, and a file with tracked changes says so and gives its
deletions under a key of their own. A reading helper's open items name a
text read cut at its size limit and a failed zoom that was not redone. A
page read in tiles names a tag that the whole-page reading and a tile read
differently at one place. The turn counts the images it sends for reading
(`image_calls` in the record), with a ceiling when
`GEOTECH_TURN_MAX_IMAGE_CALLS` is set. Three switches, off by default and
not measured on a model yet: `GEOTECH_VISION_PHOTO_PAGE_ONE_LOOK`,
`GEOTECH_VISION_TILE_ANSWERS` and `GEOTECH_PHOTO_PAGES` (`review_flags`).
New: `run_report`, an HTML report of a saved run ("Reading a saved run"
below). The review suite's "found only in the working notes" line reads
only what a reading said, and a table column headed "NO." is no longer
taken for the word "no". Release note: GeotechStaffEngineer's
`module_work/aec_1_2_0/RELEASE_NOTE.md`.

**1.1.0 (2026-11-17; the first release with Document Review as one unit, BACKLOG INF-76).** One more module moved in:
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

Also in 1.1.0: a Python warning raised inside a method reaches the agent
(`call_warnings`), and a value outside a parameter's listed values is
refused with a did-you-mean (`method_call`); `plot_data` draws panels,
sections to scale and contours; vision readings are told they see one
image, a stalled reading is given up at 90 s and asked once more, and the
wait for a reading slot is bounded (`GEOTECH_VISION_SLOT_WAIT_S`); the Tiny
Apps client asks a request again once; the run record keeps stalled, late
and retried calls; a refused key's probe verdict stays with its own
caller; a reading helper's whole record is kept with what it left unread;
planlens' sheet tools are bound where the installed planlens has them; and
the review suite's checks read what a run delivered. Release note:
GeotechStaffEngineer's `module_work/inf76/RELEASE_NOTE.md`.

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
| `vision_tools.reference_read_roots`, `.read_reference_figure`, `.read_reference_table`, `.view_worked_example_source` | its reference library's folders and readers |
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

## Reading a saved run

```
python -m aec_agent_core.run_report <run folder>
python -m aec_agent_core.run_report <run folder> -o report.html --documents <folder with the PDFs>
```

This writes one HTML file, `run_report.html`, beside the run's record (or
where `-o` says), for an engineer to read in any browser without a network
or a developer. It shows what was asked, the model and the totals (model
calls, tokens, vision calls, helpers, wall time) and the files delivered;
then every step in order: the model's own words, each tool call with its
arguments and its result (long ones folded, with their size), and each
helper's steps under the brief it was given; then, per document, a grid of
its pages saying which were read as text, looked at whole, in tiles,
zoomed, shown on a contact sheet or never opened; and each page looked at,
with a box for every look, tile and zoom, numbered as the steps. A page
viewer inside the same file opens any page of the grid (below).

`<run folder>` is a page conversation's folder or one task folder of the
review suite (`runs/<arm>/<task>/`). The pages are drawn behind the boxes
when the documents' PDFs are in that folder or under `--documents` (the
suite deletes its uploads after scoring, so point `--documents` at its
documents); otherwise the page outlines are drawn alone. A missing or older
record file is named in the report. The file holds the run's text and page
images: treat it as you treat the documents.

Added in review: the questions and answers are read from `transcript.jsonl`
as the pages write it (a long question whole, the answer as the user
received it, or a line saying the record holds none); without an
`activity.jsonl` the page grid says "no look recorded", not "never opened";
a whole copy of a cut payload is read only from inside the run folder; a
record the report cannot be built from ends in one plain line and exit code
1, never a traceback; and the page names the run folder by its own name,
never its absolute path. Size, measured on a made-up run of 300 large
sheets and 600 tiled looks: the page is about 1.5 times the size of the
activity log (a 12 MB log gave a 17.7 MB page in about 12 s); the page
images are bounded as the next paragraph says.

The page viewer: choosing a square of the page grid (or "Open in the page
viewer" under a drawing) opens that page over the report, with its image
and a box for every look, tile and zoom. Each box says which call made it
(its number, tool, the main agent or which helper, and the step); choosing
a box or a numbered call shows what was asked and the first 1,500
characters of what came back, with a link to the call in the step list.
Previous / next page and previous / next page looked at (also the arrow
keys, with Shift for the pages looked at); Esc or the browser's Back closes
it. It is the file's own script reading a JSON block of the same file, so
it works opened from disk with no network. Page images are bounded so that
a long document still gives a file a browser opens: each page looked at is
drawn at 420 pixels on the long side behind its boxes (at most 300 pages;
a count of pages, not of bytes) and at 900 for the viewer while those stay
under about 8 MB of JPEG; the other pages at 420 for the viewer, the pages
taken in some other way first, under about 4 MB of JPEG; at most 800 pages
drawn in all (a page looked at, drawn at both sizes, counts once). The file
holds each image as base64, a third more than its JPEG. The report says how
many pages each cap left out and the images' size as they sit in the file,
and a page with no image still shows its outline where the record gives one
and lists its looks. Measured on made-up 200-page PDFs (drawing sheets,
report pages, scans) with every page looked at: a page of about 17.5 MB,
built in 11 to 17 s; with one page in five looked at, about 10.5 MB. A
noisier made-up raster with every page looked at gave a 33 MB page that
still opened. A record with a transcript and no `activity.jsonl` now lists each
turn's question and its answer (and the files shown with it).

Image calls: where a turn's record keeps its own count (`image_calls` on
its `turn_end`, from 1.2.0; with `image_limit` when a ceiling was set and
`image_calls_refused` when the ceiling refused any), the totals add an
"Image calls" row under "Vision calls" with one sentence saying what each
counts, and the notes list each turn's image calls against its ceiling,
beside a step cap reached. The two can differ: vision calls are the vision
model's answers in the activity log, the vision probe's included; image
calls are the images the turn counted as sent, each once with its retries
(a page looked at twice counts twice), refused ones not included. A record
without the count gives the page it gave before.

## Licence

MIT, as GeotechStaffEngineer. See `LICENSE`.
