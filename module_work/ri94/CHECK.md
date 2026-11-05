# CHECK: RI-94 and RI-200 (with OWN-124's glossary mark), one prompt bump

Branch `claude/ri-94-200-narrative`. Model-first prompt version
**mf-8 -> mf-9** (`report_ingest/model_first.py:313`), once, for both items
and the glossary mark.

## 1. RI-94: a bound document is told to the narrative by its kind

### What changed and why

- `report_ingest/bound.py:508-525`: `CURRENT_KINDS = ("volume",
  "data_report")`, `PREVIOUS_KIND = "appended_prior_report"`,
  `bound_kind(row)` (a row naming no kind is triage's default, a prior
  report; an unknown kind is `other`) and `is_current_volume(row)`.
- `report_ingest/bound.py:527` `narrative_block` tells the narrative by
  kind. Only an appended prior report is a previous investigation and
  counts towards previousInvestigationCount. A volume or data report of
  this investigation is said to hold this report's own borings, test pits
  and cone soundings: count them in boringCount / testPitCount / cptCount,
  put their names in the dictionaries, and do not count it as a previous
  investigation. An `other` document is neither, unless the prose calls it
  a previous investigation.
  - When every bound document is a prior report (the common case, and the
    synthetic report's), the text is master's, byte for byte
    (`bound.py:568`). Today's readers' saved readings and the off snapshot
    stand for those.
  - Rows of another kind are tagged in their line, e.g.
    `pages 62-90 (a volume of THIS investigation): ...`.
- The counts take a current volume's holes in, while those holes stay in
  the child's own record.
  - `report_ingest/graph.py:1380`: when `told` names a current volume
    (`_current_bound_ids`, `graph.py:2721`), `_counts_from_logs` waits until
    the bound documents have been read. With no current volume nothing
    moves, so the off snapshot is unchanged.
  - `_current_volume_investigations` (`graph.py:2731`) collects the
    children's holes.
  - `_recheck_counts` (`graph.py:2743`): the reconciler ran before the
    children were read. Its `count_mismatch` entries on boringCount,
    testPitCount, cptCount, boringDictionary and testPitDictionary are
    replaced by the same checks (`reconciler._check_counts` /
    `_check_dictionaries`) over parent plus current-volume holes. One
    `note` (`where="bound.current_volume"`) says whose holes were counted.
  - `_counts_from_logs(..., also=)` (`graph.py:2810`) settles the five
    exploration answers over the same union when `counts_from_logs` is on.

### Made-up before/after

The parent's own logs are B-1 and TP-1. A Volume 1 bound at pp15-18 is
identified as kind `volume` and holds BH-1. The narrative states 4 borings.

| | master | branch |
|---|---|---|
| narrative told | "Each of them IS a previous investigation ... counts towards previousInvestigationCount" | "(a volume of THIS investigation) ... ARE this report's ... NOT previous investigations" |
| `count_mismatch` general.boringCount | `found 1` | `found 2` |
| `counts_from_logs="settle"`: boringCount / boringDictionary | 1 / `["B-1"]` | 2 / `["B-1", "BH-1"]` |

Kind `appended_prior_report` gives master's text and numbers on both.

### Tests

- `report_ingest/tests/test_bound_kinds.py`, 8 tests with made-up bound
  ranges for each kind:
  - which kinds are current, and the default kind;
  - the text for prior, volume, data_report and other;
  - prior-only gets master's words;
  - a file with every kind gives each rule for its own pages.
  - On master the module does not import (`CURRENT_KINDS` missing).
- `report_ingest/tests/test_graph_bound_kinds.py`, 5 end-to-end tests on
  the synthetic report:
  - volume and data_report are told as this investigation, and BH-1 stays
    in the child;
  - a volume's boring is settled into the parent's count and dictionary;
  - the count check finds 2 for a volume and 1 for a prior report.
  - 4 of the 5 fail on master. The prior-report test passes on master on
    purpose: it guards the old behaviour.

## 2. RI-200: model first gets today's whole-report search

### What changed and why

- `report_ingest/model_first_read.py:1146`: `read_narrative` runs
  `narrative_reader.retrieval_passages(doc, window=window)`. That is
  today's search for siteClass, asceSevenVersion, seismicCodeUsed,
  soilCorrosion, bearingCapacity, liquefactionPotential and reportDate,
  with the same phrases, exact first then fuzzy, and the same limits.
- The passages go into the instruction's addendum in today's words
  (`_retrieval_block`), after the bound documents and the caller's
  questions, before the last sentence.
- A searched page is citable:
  - in the builder (`:1171`);
  - in the call-page check (`:1180`), so no false "page numbers mix"
    entry;
  - in placement (`:1210`).
- New keyword arguments:
  - `window`: the search stays inside a bound document's pages, or the
    parent's own under `narrative_skips_bound`, as today's reader's does.
  - `retrieve=True`: `False` sends the instruction alone.
- `report_ingest/graph.py:3545, 3781`: `_read_model_first` hands `window`
  through `_read_unit` to the narrative. It is the same window today's
  narrative is searched in.

### Made-up before/after

The prose (pp0-3) names no ASCE 7 edition. p9, an appendix calculation,
prints "computed per ASCE 7-16".

- **Master:** model first is never shown p9, and asceSevenVersion is null
  (MF-3's R30 case).
- **Branch:** the instruction carries `PASSAGES FOUND ELSEWHERE IN THE
  REPORT ... asceSevenVersion: page 9: ... ASCE 7-16 ...`. The answer
  citing p9 keeps full confidence (the quote gate finds it there) and
  raises no page-number entry.

### Tests

`report_ingest/tests/test_model_first_retrieval.py`, 6 tests:

- the seven fields are today's;
- the appendix-only edition is shown, answered and citable, on both
  routes;
- the window keeps the search inside;
- `retrieve=False` sends the bare instruction;
- `_read_unit` forwards the window.

5 fail on master. The seven-field test is a guard.

## 3. OWN-124: the glossary's "(DRAFT)" mark

`report_ingest/narrative_glossary.py:295`: the recommendedFoundations /
bearingCapacity convention is `CONFIRMED`, with the owner's 2026-11-03
answer in its `why`. That is the half of 9abd7ae that 46a5252 reverted
for want of a bump. Nothing else from OWN-07 changed: the other conventions
stay DRAFT.

## Prompt versions

- **`READ_PROMPT_VERSION` mf-8 -> mf-9.**
  `test_model_first_schema.MODEL_FIRST_ASKS["mf-9"]` = `d529da01...`. The
  digest moved only for the glossary mark: the bound block and the passages
  are addenda the digest does not cover, but the bump re-reads every saved
  model-first narrative anyway.
- **Today's narrative reader** records no prompt version. It also sends the
  glossary, so its request text changed with the mark.
  - `report_ingest/tests/graph_off_snapshot.json` was re-recorded with
    `write_off_snapshot`, as RI-204 did.
  - Exactly four request digests changed, the narrative requests:
    `standard` 6 and 12, `rules` 4 and 10.
  - No output file's digest changed. `taken_with` was kept as it was.
  - With the glossary change stashed, the old snapshot passes unchanged.
    So RI-94's change (byte-identical for prior reports) and RI-200's
    (model first only) do not move the off path.
- `webapp/tests/test_page_fingerprints.py` passes unchanged, so nothing was
  re-recorded.

## Test runs

- `python -m pytest -q -n 4 report_ingest webapp/tests/test_public_text.py
  webapp/tests/test_page_fingerprints.py`: **4413 passed, 8 skipped (pydiggs
  absent), 0 failed**, exit 0.
- Consumers: `module_work/report_ingest_harness/tests`: 182 passed, 34
  skipped, 0 failed. No consumer in `webapp/` or `funhouse_agent/` reads
  the functions changed.

## Release note

- **Bound volumes:** a report with a Volume 1 or data report of the same
  investigation bound inside it (triage or the identity call says `volume`
  or `data_report`) now tells its narrative that the bound document's
  borings, test pits and soundings are this report's.
  - The narrative's previousInvestigationCount should drop by one for each
    such document, and its boringCount / dictionaries should include those
    holes.
  - The parent's `count_mismatch` entries count those holes too, with one
    `bound.current_volume` note. With `counts_from_logs` on, the settled
    counts and dictionaries include them.
  - The holes stay in the child's record.
- **Appendix-only answers:** model-first narratives can now answer the
  seven appendix-answered questions from passages anywhere in the report,
  e.g. an ASCE 7 edition printed only on a calculation page.
- **Re-reads:** every saved model-first result is re-read on resume (mf-9).

## To measure live

- **RI-94:** on a report with a bound Volume 1 / data report, check:
  - the identity call returns `volume` or `data_report` for it;
  - previousInvestigationCount comes out right in both arms;
  - boringCount / boringDictionary include the volume's holes;
  - the parent's count_mismatch entries are gone or right.
  - Also re-check a report with a prior report bound in (R21's pp120-591)
    to confirm nothing moved.
- **RI-200:**
  - Re-run R30 model first: asceSevenVersion should be filled, citing p555.
  - Compare the other six fields against today's reader on R15, R30 and
    R21.
  - Watch for passages pulling a bound or other report's values into the
    parent, under `narrative_skips_bound` off (no window, as today).
  - Record the narrative request's tokens in: up to 12,000 characters
    more.

## Noted, not fixed

- `model_first_read.narrative_plan` (the dry-run sizer) leaves out the
  addendum, as it already did for the bound block, so it under-estimates by
  up to the passages' size.
- A current volume's holes from child items that failed to read are not
  counted as "held back" in `_counts_from_logs`; only the parent's own
  items are checked for unread logs.
- `narrative_skips_bound` off: as in today's reader, the search window is
  the whole file, so a passage on a bound document's page can reach the
  parent's narrative.
