# Document Review architecture: three job shapes (plan of record, 2026-09-28)

Owner discussion, 2026-09-28, after 5.30.0 shipped the review suite and the
switched harness changes (`module_work/REVIEW_HARNESS.md`). Nothing here is
built yet. This is the design agreed in conversation, to be built in the
milestones at the end, each measured before the next.

## Context

The 5.30 review showed the Document Review page was tuned one example at a
time and that one agent loop cannot carry a large review:

- The agent's limit is **model calls** (about 17 per turn shipped, 40 lean),
  not pages. Text pages are cheap (search and read cover many pages per call);
  pages that must be looked at go one per call.
- The harder limit is **context**. Everything read stays in the conversation
  and is re-sent on every call; a look at a small-lettered sheet can return
  ~30,000 characters. Dozens of looks degrade quality and raise cost before
  the call limit is reached.
- Between turns the app keeps only user messages and the agent's answers, not
  tool results, so a review split across turns forgets what it read unless it
  wrote it down.

Reviews at the office range from tens of pages to sets of 10+ PDFs of 100+
pages each. The owner expects wide use to bring the large ones.

## Terms

- **Agent**: loops (decide, call tools, read results, repeat). One per page
  today: the review agent.
- **Helper sub-agent**: an agent the review agent starts for one job; own loop
  and tools, sees nothing of the conversation, returns one report.
- **One-shot model call**: one question, one image or page, fresh call, no
  memory, no tools. Made by tools (vision reads, sweep, find_like, the probe).
- **Code**: planlens reading the PDF, rendering, storage. No AI.
- **Harness**: all of it together: prompt, tools, app, limits, logs.

## Principles

1. **The shapes stack.** Shape 2 = shape 1 + a digest. Shape 3 = several
   shape-2 reviews + coordination. One foundation serves all three.
2. **One front door.** The review agent stays the thing the user talks to.
   Code takes an inventory of the upload; the agent proposes a shape; the user
   confirms for shapes 2 and 3 (they cost real money). Page thresholds guide,
   they do not force: a targeted question on a big document stays shape 1.
3. **Code at the spine, agents at the leaves** for large work. A job runner
   (units, checkpoints, resume, cost meter) runs bounded agent jobs. No
   free-form multi-agent swarm. This is the report-ingest pattern.
4. **One finding format** everywhere (see Cross-cutting). Comparable,
   de-duplicable, checkable by code (a quote not on its cited page is flagged).
5. **Work saved as files**, never only in context or `/tmp`: conversation
   folder + SharePoint mirror. Survives turns and cluster restarts.
6. **Measure every shape** with its own suite tasks before and after each
   change. Tester feedback becomes a task, not a prompt rule.
7. **Special-purpose tools stay optional**, known only through their own
   descriptions (owner rule, 2026-09-26).

## Shape 1: small review (about 20 pages or fewer)

Today's loop is the right shape: at this size the agent can read every text
page and look at every drawing page itself. Refinements, in order:

**S1.0 Decide how pages are looked at, by measurement.** Owner runs the 5.30
suite on the cluster: `probe(...)` first, then baseline vs lean on a few tasks,
then all arms (baseline, lean, grounded, inline, sweep). The winner decides
the look primitive that shapes 2 and 3 reuse. Do this before designing page
notes.

**S1.1 Geometry for the review page.** planlens' drawing geometry
(`planlens.ir`: leaders, dimensions, title blocks, revision clouds, detail
bubbles; today reachable only through the geotech page's `call_agent` via
`funhouse_agent/adapters/drawing_ir_adapter.py`) has never been on the review
page. Add a small set of review tools with review-page descriptions:
- what a leader or callout at a spot points to;
- dimensions near a point, with their values and the geometry measured;
- the title block's fields;
- revision-clouded areas.
"Geometry says where, vision says what." Proposals carry confidence and can
be confidently wrong on real sheets, so they ship behind a switch and are
measured: add suite tasks that need geometry (a leader's target, a dimension
value, clouded changes) on the Mecklenburg sheets, whose DWG truth has the
leaders and dimensions.

**S1.2 Overview contact sheets standard above ~20 pages.** Today the automatic
orientation uses them only "if it is long", inconsistently. Make them
automatic above ~20 pages (one call per 48 pages; good for finding plans, logs,
marked-up pages, not for reading callouts), skip them for short documents.
If inline wins S1.0, show the contact sheet to the agent directly instead of
through `analyze_image`. Measure with the runner's `orientation=True`.

**S1.3 Describe grounded honestly.** Its text-layer half is belt-and-suspenders
next to the resolution fixes (image budget, tiling, zoom, "read the text layer
first") and adds nothing on stroke-lettered sheets; its located-items half
(page boxes) is new. Say so in `REVIEW_HARNESS.md` and the harness pages.

**S1.4 Look policy by purpose (if inline wins).** One-shot calls for bulk
passes (sweep, digest page notes: cheap, parallel, text anyway); inline for
close verification (zoom on one spot before reporting it). Not a user
choice: the switches are owner settings that the suite retires into defaults.

**S1.5 The finding format**, decided now because everything above reports
through it (see Cross-cutting). Shape-1 deliverables (Word memo, marked-up
PDF) are rendered from findings.

**S1.6 Small fix:** `funhouse_agent/find_like.py` verifies candidates in a
thread pool without copying the run context, so those model calls are missing
from `activity.jsonl` and token totals (same bug fixed for tiles in 5.30).

Everything else in shape 1 (orientation wording, answer style, clarifying
questions, memo layout) is independent and can come any time.

## Shape 2: large review, one scope (the common case)

Larger PDFs or several PDFs, not ~1,000 pages, not many disciplines. The lead
agent **holds summaries and pulls detail on demand**; it does not hold
everything.

**Stage A, the digest (built once per document, discipline-neutral):**
- **Free layer, at every upload (code, seconds):** inventory, page map, page
  kinds and roles, structure with printed page numbers, text with positions,
  tables, review markups, stated quantities, references to other sheets and
  sections, plus the overview contact sheets above ~20 pages.
- **Expensive layer, on demand:** page notes for pages that must be looked at
  (drawings, scans, figures), in a fixed schema: what the page is; key items
  and values with exact quotes and page boxes; what it references; what was
  unclear. Built the first time a question needs coverage, after the agent
  shows an estimate ("140 drawing sheets, about $X and N minutes; go
  ahead?"). Page notes roll up to section summaries and a document summary.
- **Coverage question** = one whose honest answer requires touching all or
  most relevant pages ("which sheets…", "every comment…", "where does X
  disagree with Y"). Targeted questions ("what bearing pressure is
  recommended?") are answered by search and read without a digest.
- Built with the sweep and page-reader machinery, writing to a store instead
  of the conversation. Keyed by document content hash, so a re-upload reuses
  it. Saved in the conversation folder and mirrored.

**Stage B, the lead agent reviews from the digest:**
- Starts with document summaries and an index in context; tools to search the
  page notes and fetch notes for pages.
- Cross-references and works checklists from the digest; **verifies on the
  page before reporting** (cheap zoom), which stops "telephone" loss from
  becoming a wrong finding.
- Follow-up questions are cheap and survive turns, because the digest does.

**Measure:** 2 to 3 sets of 200 to 500 pages with question tasks and planted
problems; score found vs planted, false alarms, cost, time.

**Reuse:** `report_ingest` (resumable runner, mirror, cost meter, record +
library with FTS search), `sweep_pages`, the page reader, planlens page roles.

## Shape 3: mega-reviews (a whole submission, many disciplines)

Not prescriptive yet (owner, 2026-09-28): expect a lot of back-and-forth
between user and orchestrator, like building a plan in Claude Code.

- **A plan document both edit.** The orchestrator drafts it from the
  inventory: disciplines, what each checks against, depth, deliverables,
  cost and time estimate. The user changes it, it revises, nothing expensive
  runs until approved, and it stays editable during the run ("skip the
  landscape sheets", "go deeper on the shoring submittal"). Same idea as the
  `geo_project` approval gates.
- **Split by discipline.** Each discipline focuses on certain documents, but a
  document is looked at by several disciplines, so the digest is built once
  per document and read by each discipline through its own checklist.
  Findings can carry more than one discipline tag.
- **One reviewer per discipline:** a shape-2 lead over its slice of the digest
  with its discipline's checklist, writing findings in the shared format.
  Parallel, limited mainly by Prompter rate limits (429s seen in
  report-ingest runs).
- **Coordination pass:** code lines up facts that should agree across
  documents and disciplines (a material strength in spec vs drawings, a
  referenced detail that should exist, a report recommendation vs the design
  value); an agent adjudicates only the candidate conflicts, against the pages.
- **Consolidate:** de-duplicate, rank by severity, render the comment log or
  matrix and a marked-up copy per PDF.
- **Runs as a background job,** not a chat turn: checkpoints per unit, resume
  after a cluster restart, progress and running cost shown in the app, stop
  and continue. When done, the user is back in shape-2 conversation over the
  digest and the findings.
- **Measure:** one seeded mega-set (known problems planted across documents
  and disciplines).

## Cross-cutting

**Finding (sketch):** id; statement; severity; confidence; disciplines;
citations (document, sheet or printed page, viewer page, page box, exact
quote); evidence kind (read from text / seen by looking / computed); status
(draft, confirmed, rejected by user); related findings; source (which worker
or reviewer). Code checks each quote against its cited page.

**Page note (sketch):** document, page, kind, title/sheet id; items (label,
value with unit, exact text, box, read-or-seen, sure); references out; notes;
unclear items; cost of the note.

**Review plan (sketch):** documents in scope; disciplines and their criteria
(spec sections, checklists, prior comments); depth; deliverables; estimate
(pages to look at, calls, dollars, hours); approvals and later edits.

**Hosting constraints:** Funhouse cluster idles out after 30 minutes and the
app lives in a notebook; Tiny Apps runs on App Service. Long jobs must be
detached, checkpointed and mirrored. Cost estimates use `PROMPTER_PRICES`.

**Privacy:** SBU tester documents never enter the repo; private suite tasks
live on SharePoint (`extra_tasks=`).

## Milestones (each ends with the owner's measurement on the cluster)

- **M0 (owner):** install 5.30.0; run the probe; run the suite, baseline vs
  lean first, then all arms. Decide the look primitive (S1.0).
- **M1, shape-1 refinements:** geometry tools behind a switch (S1.1),
  contact sheets standard above ~20 pages (S1.2), look policy (S1.4), the
  finding format (S1.5), find_like fix (S1.6), geometry and orientation suite
  tasks. Release, measure, flip winning switches to defaults.
- **M2, shape 2:** digest free layer, on-demand page notes with estimate and
  consent, lead-agent digest tools, cross-turn reuse, large-set suite tasks.
- **M3, shape 3:** the editable plan document, the background job runner,
  discipline reviewers, consolidation and deliverables, one seeded mega-set.
- **M4:** the cross-document coordination pass.

## Built in M1, RELEASED as 5.31.0 (2026-09-29)

Everything that does not wait on the M0 numbers, each behind a switch in
`funhouse_agent/review_flags.py`, OFF by default (with every switch off the
page is unchanged; tests pin it):

| Switch | What | Code |
|---|---|---|
| `GEOTECH_REVIEW_FINDINGS` | S1.5 finding format: `record_finding`, `update_finding`, `list_findings`, `findings_report` (Word comment log, `<doc>_findings.pdf`); quotes checked against the page (markup text counts; unreliable OCR and "seen" lettering are not held against a finding) | `review_findings.py`, `deep/findings_tools.py` |
| `GEOTECH_REVIEW_OVERVIEW` | S1.2 contact sheets requested above `GEOTECH_REVIEW_OVERVIEW_PAGES` (20); with inline too, the sheet is shown to the agent | `webapp/profiles.py orientation_request_for` |
| `GEOTECH_REVIEW_GEOMETRY` | S1.1 `drawing_callouts`, `drawing_dimensions`, `title_block`, `revision_clouds` over planlens.ir, displayed-frame coordinates, proposals with confidence | `deep/geometry_tools.py` |
| `GEOTECH_REVIEW_DIGEST` | Shape 2 free layer: per-document inventory, page rows, FTS5 index, cross-references with missing-sheet detection; `document_inventory`, `digest_search`, `digest_pages`, `digest_references`; 538 pages in ~3.5 s | `review_digest/`, `deep/digest_tools.py` |

Arms `overview`, `geometry`, `digest` added (existing arms unchanged; the
5.30 `inline` arm behaves as released). Also: S1.6 `find_like` context fix;
the working folder is bound at agent build (`build_deep_agent(working_dir=)`)
so findings and digests cannot cross conversations; `switches()` also resets
`SETTINGS_ENVS`; the `digest/` cache is not a download card. Suite: 35 tasks
(3 geometry, 3 long-document coverage with a `labelled_set` check).
Independent review: 8 bugs + 6 concerns, all fixed with tests. Gate:
1,103 / 1 and 1,460 / 4 (webapp, deep, suite, digest, geo_project, rest of
funhouse_agent).

## Decisions recorded (owner, 2026-09-28)

- Three job shapes as above; the shapes stack.
- Digest: split the difference (free layer at upload, expensive page notes on
  demand with an estimate).
- Mega-reviews split by discipline; documents shared across disciplines;
  planning is collaborative and iterative, not prescribed.
- Shape 1 is modular: refinements can come before or after shapes 2 and 3,
  except the look primitive, which is settled by the suite first.
- Grounded is described as belt-and-suspenders for its text half.
- Basic / grounded / inline are owner settings for measurement, not user
  choices.

## Open questions for the owner

1. Page threshold between shapes 1 and 2 (about 20?) and when to ask before
   building page notes (any cost, or above a dollar amount?).
2. The disciplines to plan for first, and where their checklists come from.
3. Deliverables for large reviews: comment log, compliance matrix, marked-up
   PDFs, memo; the office's usual format.
4. Turnaround: is "results by morning" acceptable for mega-reviews?
5. Large test sets for the suite: public sets that can be committed, or
   private ones kept on SharePoint?
