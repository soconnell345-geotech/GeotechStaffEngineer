# `report_ingest` — the two model passes over a whole report

Part of the report-ingest train (`module_work/REPORT_INGEST_PLAN.md`). planlens
turns a PDF into pages with kinds, located text, an outline and a first draft of
what each page **is**; this package adds the judgement that needs the whole
document in view. Two passes ship here today. The readers, the record and the
DIGGS writer are later work packages.

| Pass | Module | Shape | Model in development |
|---|---|---|---|
| 0b. Document triage | `triage.py` | one structured call | `claude-sonnet-5` |
| 0c. Label review | `label_review.py` | an agent loop with four tools | `claude-opus-5` |

## Why they exist

The rules label a page from what that page prints about itself. Measured on the
corpus, they reach 0.91 accuracy across the fourteen reports they were built on
and 0.79 on the next report anyone opens. The gap is structural, not a missing
rule: a page only makes sense in the context of the rest of the report. An
appendix tab that says "LABORATORY TEST DATA" is right about most of its
appendix and wrong about the summary table, the legend sheet and the stray
calculation bound into it, and no amount of per-page cleverness fixes that.

So both passes are given everything the report says about **itself** before any
of it is read in detail.

## What each pass reads and returns

### `triage(doc, roles, outline, engine=...) -> DocumentProfile`

One call. It is given:

- the per-page ledger, one line a page (`planlens.document.roles.page_ledger`);
- the outline the document prints: contents, lists of figures, tables and
  appendices, every divider's text, figure captions, section headings;
- the front matter verbatim, through the end of the contents;
- the first contact sheet as a picture;
- **the facts Python already counted** — the scanned fraction, the
  unreliable-text fraction, where the printed page numbering restarts, the page
  kinds and the rule labels.

That last point is the rule of the module. A model asked to count 455 ledger
lines will be nearly right, which is the worst thing a scorecard field can be.
Those fields are measured by `document_facts()` and copied into the profile;
`TriageFindings` — the schema the model is held to — does not contain them.

It returns a `DocumentProfile`: `document_type` (the owner's enumeration),
`bound_together`, `has_narrative` / `has_logs` / `has_lab` / `has_calcs`,
`languages`, `toc_agreement`, `anomalies`, `rationale`, the measured fractions,
and a `workflow` in `standard`, `appendix_only`, `partial`, `multi_document`,
`scanned`, `needs_person`. The workflow chooses the readers that follow, so an
odd file is flagged rather than forced down the standard path.

### `review_labels(doc, roles, outline, profile, budget, engine=...) -> Review`

An agent loop. Its tools are `read_page(page)` (the toolkit's text for one
page), `render_page(page)` (the page as a picture at 80 dpi),
`contact_sheet(start_page)` (48 thumbnails, each labelled with its page index
and kind) and `outline()`.

Its brief carries the label vocabulary with one-line definitions, the ledger,
the outline, the triage profile, and **the rules' own weak spots as the list to
check first** — every page whose label came from an appendix tab rather than the
page, every page the tab left ambiguous, every page guessed from its shape,
every page with no readable text, and every page whose label contradicts both
its neighbours. A document with no appendix tab anywhere is said so in the first
line, because then nothing inherited and every label is weak.

**Budget:** `max(60, 0.25 × pages)` tool calls. When it is gone, further calls
come back as errors telling the model to report what it has, and
`Review.stopped_on_budget` records it. `max_model_calls` (default 60) stops the
loop itself.

It returns a `Review`. The model supplies `changes` (page, from, to, a reason of
25 words or fewer, and which tool showed it), `structure` (section or appendix
title → the page range it actually occupies) and `unresolved` (pages it could
not settle, with why).

`final_labels` is **not** asked of the model: it is the rules' labels with the
accepted changes applied, in Python. A model re-emitting 455 page-to-label pairs
can drop a page or shift a number silently, and the scorecard would then be
scoring the slip rather than the judgement. Four kinds of change are refused and
recorded in `rejected_changes` instead of applied: a page that does not exist, a
label outside the vocabulary, a second change to a page already changed, and a
change to the label the page already has. A change's `from` is always overwritten
with what the rules really said, so a model that misremembers is visible.

## The engine, and where it runs

Neither pass imports an SDK. Both take an `Engine` — one method:

```python
complete(messages, system=..., tools=..., images=..., output_format=...) -> Reply
```

Messages are neutral blocks (`text`, `image`, `tool_use`, `tool_result`,
`opaque`), and the engine translates to and from its provider's shape in both
directions. `opaque` is how a pass carries a thinking block back to the model
unchanged without ever reading it. So the same passes run on the Claude API here
and on the cluster's Prompter later, behind the app's own engine abstraction.

`ClaudeEngine` is the development implementation. **The credential comes from
the environment** — `ANTHROPIC_API_KEY`, or an `ant auth login` profile. It is
never an argument, never logged and never printed; only its presence is checked,
and a missing one fails at construction rather than deep inside a run. The
prompt prefix is cached by default (`auto_cache=True`), which on the checkpoint
turned roughly 1.4 M resent tokens into cache reads at a tenth the price.

`CostMeter` records calls, input and output tokens, cache reads and writes, wall
clock and dollars at the list prices in `MODEL_PRICES` (checked 2026-09-16).
Prices are pinned in the source rather than fetched: a scorecard has to be able
to say what a run cost months later.

## Running the scorecard

```
.venv/Scripts/python -m module_work.report_ingest_harness.measure_wp1b \
    --set checkpoint --note "what changed this round"
.venv/Scripts/python -m module_work.report_ingest_harness.measure_wp1b \
    --set insample --reuse --changes
.venv/Scripts/python -m module_work.report_ingest_harness.measure_wp1b \
    --set oos_blind --no-append
```

Sets are `insample` (the reports with spreadsheet labels), `oos_open` (the
lead's out-of-sample labels on ten reports), `oos_blind` (the other fourteen)
and `checkpoint` (six reports chosen to span short, long, scanned and old).
Running the first three covers all 38 corpus reports exactly once, which is also
how every report gets a triage profile. `--reuse` re-scores the saved runs
instead of calling the model, so tuning the scorecard is free.

Each run writes the full private detail to `raw/checks/wp1b/` and each report's
profile to `raw/checks/triage/`, both gitignored. The ledger gets IDs, labels,
counts and rates only, never a heading, a change's reason or a triage
rationale.

**Every change is graded** against the hand label as `fixed`, `broke`,
`still_wrong`, `disputed` or `unscored`. A review that raises accuracy while
breaking three correct labels has not earned the raise, and a total alone would
hide it.

**The blind set is protected and reported honestly.** `oos_blind` prints one
summary, writes no miss list and no change list, and refuses `--changes`. Two of
its fourteen reports were named in the cost checkpoint and their changes were
read, so the set is reported twice: over all fourteen, and over the twelve
nobody has opened. The second figure is the honest one and the output says so.

**Disputed hand labels.** When the review contradicts a hand label and the lead,
looking at the page, judges the hand label the doubtful one, the entry goes in
`DISPUTED`. The spreadsheet is never edited: a hand label records what a person
decided, and rewriting it to suit a model would destroy the only independent
thing in the measurement. A confirmed dispute drops the page from the before
**and** the after score, so it counts as neither a rule hit nor a review miss,
and the number of dropped pages is printed. It only drops while the review still
says what it said when the dispute was raised; a later run that moves the page
somewhere else is a new answer and is scored.

**Three guards, each from a run that went wrong.** A run in which every report
fails no longer overwrites the last good run's detail file. Every saved run
records a hash of the prompts that made it, and scoring a set whose saved runs
disagree refuses to stand behind the totals — half a set on new prompts and half
on old is a set that never existed. And `--max-report-dollars` (default $5) and
`--max-total-dollars` (default $45) stop the set rather than spend past them,
naming the report they stopped before.

## Tests

```
.venv/Scripts/python -m pytest report_ingest/tests -q
.venv/Scripts/python -m pytest module_work/report_ingest_harness/tests -q
```

Offline and free: no credential, no network, no corpus. `FakeEngine` replays a
script of turns in the engine's place, so the whole of both passes runs against
planlens' synthetic report — the prompt each builds, the tool loop, the budget
stop, the rules for applying a change. Running past the end of a script raises,
because a pass that makes one more call than the test expected is the bug the
test exists to catch. The harness suite adds the scorecard's own arithmetic, the
disputed-label rule and the prompt fingerprint; its corpus-dependent tests skip
cleanly on a machine without the private data.

## What the measurements have taught the prompts

Kept here because each line is a prompt rule that exists for a measured reason,
and a later editor who does not know why will delete it.

- **The four exploration labels say they may be PLOTTED, not just tabulated,
  and that each belongs to one named exploration.** On the cost checkpoint,
  fifteen of the review's nineteen wrong changes were an exploration's own
  results called `figure`, or one kind of sounding called another. The rules had
  been wrong about those pages and the review was right to move them; it moved
  them next door. A page showing depth against blow count looks like a graph,
  and nothing had said that a graph of one exploration's results is that
  exploration's log.
- **`figure` says it is the report's own numbered figure series and never an
  exploration's results**, for the same reason, from the other side.
- **The review is told to choose among the four logs on what is MEASURED, not on
  how it is drawn**: blows per increment of penetration is a DCP, continuous tip
  and sleeve and pore pressure is a CPT, a logged pit or trench face is a test
  pit, a drilled hole with driven samples and blow counts is a boring.
- **The brief leads with the rules' own weak spots.** The out-of-sample study
  sorted every rule miss into four classes and three are answerable only by
  opening the page, so those pages are named first and the rest of the document
  follows.
- **The deterministic facts are given, never asked.** A model asked to count 455
  ledger lines will be nearly right, which is the worst thing a scorecard field
  can be.

## What this package needs to ship

Nothing is wired into the app yet and `pyproject.toml` is untouched. When it
ships it will need:

- `report_ingest` added to `[tool.setuptools.packages.find]` and
  `report_ingest` (or `report_ingest/tests`) added to pytest's `testpaths`;
- `pydantic`, already a dependency;
- `anthropic` kept **optional**. Nothing outside `ClaudeEngine` imports it, and
  `report_ingest/__init__.py` reaches every entry point through a lazy import,
  so importing the package costs an app nothing at startup and the cluster,
  which runs the Prompter, never needs the package at all.
