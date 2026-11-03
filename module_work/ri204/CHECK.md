# RI-204: the owner's narrative rulings of 2026-11-03 in both narrative readers

MF-3 (build g on Foundry) put R30's `soilCorrosion` wrong in both arms and
its `earthHazardsExposed` wrong in model first. The owner ruled on both on
2026-11-03 (OWN-119, OWN-120). This note gives IDs, counts and field names
only. The replay read the private hand-back in the gitignored raw folder.
The code is commit 8304f32 on `wip/ri204`.

## What changed

### 1. Two house conventions (`report_ingest/narrative_glossary.py`)
Both are `CONFIRMED`. Each `why` names the owner's ruling and the MF-3
evidence.
- **`soilCorrosion`, after the existing finding rule (OWN-119).** Across
  samples and strata the worst governs. One sample or stratum that tests
  corrosive makes the soil corrosive, even when the others test not
  corrosive. A corrosive fill sample counts above all, because fill touches
  the basements, footings, slabs, pavements and utilities. So one corrosive
  sample among non-corrosive ones answers `yes`, not `no` and not `mixed`.
  `mixed` stays only for soil corrosive to some materials and not to others,
  each material judged by its worst sample (RI-182's ruling of 2026-10-29).
- **`earthHazardsExposed`, after the two existing hazard rules (OWN-120).**
  Landslide only for an actual landslide or a landslide hazard the report
  names. A slope-stability analysis, a factor of safety or a concern about a
  slope's stability does not put landslide on the list.

The conventions block goes into both readers: today's narrative brief
(`narrative_reader._brief`) and model first's narrative instruction
(`model_first.instruction`). Nothing is enforced in Python.

### 2. Prompt version
- `READ_PROMPT_VERSION` is now `mf-8` (was `mf-7`). The new text reaches the
  model-first narrative instruction. A resume reads a narrative saved under
  `mf-7` again.
- `test_model_first_schema.MODEL_FIRST_ASKS` gains mf-8's digest.
- Today's narrative reader records no prompt version, so there was nothing
  else to bump.

### 3. The off-path snapshot
- `report_ingest/tests/graph_off_snapshot.json` was re-recorded with
  `write_off_snapshot`, for the conventions text only. Four request digests
  changed: the parent's and the bound report's narrative requests in both
  off runs (`standard` requests 6 and 12, `rules` requests 4 and 10).
- No output file's digest changed, and `taken_with` is unchanged.
- Proof (scratchpad `snapshot_check.py`): with the two new conventions taken
  out of `CONVENTIONS`, today's off runs equal the saved snapshot entry for
  entry. With them in, only those four narrative requests differ.

### 4. Page fingerprints
`webapp/tests/test_page_fingerprints.py` passes unchanged, so nothing was
re-recorded.

## Replay on the record ($0, no model calls)

The saved run files keep each call's `prompt_version`, not its prompt text.
So the replay rebuilds what each reader is told on R30 from the code, with
R30's saved narrative page lists. The conventions block is the same for
every report. Before is master's glossary, which is build g's (unchanged
since 0fa2ed5); after is this branch.

| R30, the two fields | before (build g) | after (wip/ri204) |
|---|---|---|
| convention lines on `soilCorrosion` (both readers) | 1 (rule 11, DRAFT) | 2 (rules 11 and 12) |
| convention lines on `earthHazardsExposed` (both readers) | 2 (rules 14, 15) | 3 (rules 15, 16, 17) |
| a rule on one corrosive sample among non-corrosive ones | none | rule 12: `yes` |
| a rule on landslide for a slope-stability concern | none | rule 17: not listed |
| model first's prompt version | mf-7 (build g ran mf-5) | mf-8 |

What MF-3 saved on R30:
- `soilCorrosion`: arm A `no`, arm B `mixed`. The owner's answer is `yes`.
- `earthHazardsExposed`: arm A listed landslide, erosion and flooding, citing
  a slope-stability concern. Arm B listed flooding and erosion. Per the owner,
  B is right.

Each wrong answer is the case one new rule names. No model was re-run, so
whether the readers now answer `yes` and leave landslide out is unmeasured.
The next Foundry run measures it.

## The owner's question this leaves open
OWN-119's words are: "If any samples come back corrosive, it should be yes."
Rule 12 reads this across samples and strata only, and keeps `mixed` for the
material split. Take a report where one sample attacks metal and no sample
attacks concrete. Under rule 12 that report is `mixed`. Read literally, the
owner's words make it `yes`. The owner has been asked. If `yes` governs,
rule 12's last sentence and rule 11's `mixed` clause change, and the
model-first prompt version moves again.

## Tests (pytest exit codes)
- `report_ingest/tests/test_narrative_levers.py`: new
  `test_the_owners_rulings_of_2026_11_03_reach_both_readers`. It checks
  that each ruling is a CONFIRMED rule on its field, that the soilCorrosion
  rule says the worst governs and keeps `mixed` for the material split, and
  that both rules are in today's brief and in model first's narrative
  instruction.
- `test_narrative_levers.py`, `test_model_first_schema.py`,
  `test_graph_model_first.py`, `test_narrative_reader.py`: 222 passed,
  exit 0.
- All of `report_ingest/tests`, in two chunks: 3,026 passed and 1,353
  passed, exit 0 and 0.
- `module_work/report_ingest_harness/tests`: 201 passed, 33 skipped, exit 0.
- `webapp/tests/test_page_fingerprints.py` and
  `webapp/tests/test_public_text.py`: 19 passed, exit 0.

## RI-205 alongside (private truth; counts only)
The owner's answers of 2026-11-03 went into the private truth, and MF-3 was
rescored at $0 with the review's scorers (`_score_ingest_record`,
`harm_score`).

**Changes:**
- Hand labels: R30 p418, p419, p471-p473 to `lab_test`; R21 p715-p728 to
  `lab_test`; R15 p67 to `boring_log`.
- R21 p714 is the lab report's appendix tab. It was set to `divider`, as
  R21's other four appendix tabs are and as conventions B rule 5 says, not
  to the listed `lab_test`.
- R15's `recommendedFoundations` was split, one item per type.
- R30 has no narrative answer sheet, so its `soilCorrosion` had nothing to
  change.

**Rescore results:**

| job | harmful labels | narrative recall |
|---|---|---|
| A R15 / R30 / R21 | 0 / 6 / 15 -> 0 / 1 / 0 | R15 30/31 -> 30/31; R21 32/32 |
| B R15 / R30 / R21 | 5 / 11 / 24 -> 5 / 8 / 9 | R15 30/31 -> 30/31; R21 31/32 |

- **Labels:** the prediction holds (A R21 15 -> 0, R30 6 -> 1).
- **Narrative:** the prediction (R15 31/31) does not hold. The split fixes
  the grain, but the strict scorer pairs list items by wording. Each arm
  pairs 1 of 3 items with the sheet's (jaccard 0.2 against 0.6).
- **Lenient view:** both arms reach 0.5. Arm B's lenient hit before the
  split (0.67, by RI-13's pieces rule) becomes a miss.

## Noted, not fixed
- The R15 narrative miss above is wording in the sheet and the scorer's
  strict item match. It changes a score, not a saved report's output.
- The draft `soilCorrosion` rule 11 stays beside the confirmed rule 12. They
  read the same way. Merging them waits on the owner's answer above.
- `soilCorrosion`'s schema description ("does it address soil corrosivity:
  yes / no / mixed / unclear") is unchanged. It is pinned in
  `MASTER_SCHEMAS`, and the convention governs.
