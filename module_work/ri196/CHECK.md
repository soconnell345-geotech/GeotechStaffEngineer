# RI-196, RI-197, RI-198: model first's log pages after the MF-3 trace review

Checked 2026-11-02 against arm A's saved MF-3 runs (build g on Foundry,
reports R15, R30 and R21, parent and bound windows), at $0: no model call,
replayed from the saved page descriptions, labels, item results and
records. The replay scripts and their outputs stay in the session's
scratchpad (they name explorations); this file keeps counts and field names
only. The trace review that filed the three items is
`module_work/field_feedback/2026-11-02_foundry_build_g/TRACE_REVIEW.md`
(sections 3.2, 3.3, 6 and 7).

Today's readers (`model_first="off"`) are unchanged: every change is on the
model-first path, and the graph's off-path snapshot test passes.

## What changed

| Item | Where | Before | Now |
|---|---|---|---|
| RI-196 | `model_first_pages.exploration_id` (the describer's field, not strict) | a field giving the name and then a note after a comma ("name, depth ..., elevation ...") named nothing, so `log_groups` added the page to the log before it | the name before the first comma or semicolon outside brackets, when every piece after it is a note; a comma between digits is a decimal one; a piece naming another exploration still makes the field name none |
| RI-196 | `range_labels` describer prompt, `exploration` field | asked for the name | asks for the name alone, no depth, elevation, water level or remark; `DESCRIBE_PROMPT_VERSION` ri73-3 -> ri73-4 |
| RI-197 (a) | `model_first.LogReading.other_log_pages`, `INSTRUCTIONS["logs"]` | no way to say a page is another log but prose | the answer lists such pages; `READ_PROMPT_VERSION` mf-5 -> mf-6 |
| RI-197 (a) | `model_first_read.read_logs` | a page in `pages_left` was "not read within its model-call budget", and nothing read it | a page in `other_log_pages` or `pages_left` with no row read off it goes back: the result's `returned` list, with the reader's reason; the log does not claim it |
| RI-197 (a) | `graph._read_model_first` | units read once, in order | a queue: each returned page is read as a one-page unit `log-<page>`, once, saved and resumed like any unit, joined to its hole's log by `_join_by_hole`; one QA note per handing-back group |
| RI-197 (b) | `model_first_read.read_logs` | | a page handed back by its own one-page call is a `skipped` entry ("could not be read"): an item error, not saved, asked again on resume |
| RI-197 (c) | `log_holes.missing_holes_qa(descriptions=)` | text-layer label names only, so a scan was never listed | on a page whose text names no hole under a label, the describer's exploration stands in (model first only) |
| RI-197 (d) | `model_first_pages.is_divider` | "divider" anywhere in the kind | "divider" in the kind's head, before a bracket, comma, semicolon, colon or spaced dash |
| RI-197 (e) | `graph._inside_dividers` | a divider page between two pages of one stage was set aside with no entry | one QA note per stage naming such pages |
| RI-198 | `model_first.LOG_PIT_SIZE` in `INSTRUCTIONS["logs"]` | no pit-size rule | today's rule: fill `pit` on a pit; the bucket in an equipment field is the width, in its unit; never a pit on a borehole; `READ_PROMPT_VERSION` mf-6 -> mf-7 |

## Replay, before and after

| Check | Report | Before | After |
|---|---|---|---|
| RI-196 log groups | R21 bound | 67 log pages, 42 groups; 0 of the 12 lost pit pages opens a group | 54 groups; all 12 open a one-page group |
| RI-196 log and calc groups | R15, R30, R21 parent | | unchanged |
| RI-196 lab windows | R30 parent, R21 bound | | the cuts move to the sheet start nearest the even split (more pages named); every window still covers the same run within 12 pages |
| RI-197 (a) pages handed back by the saved answers | R21 bound | 2 pages written as "model-call budget", never read | 2 pages read alone (each from a call of several pages) |
| RI-197 (a) | R15, R30, R21 parent | | none |
| RI-197 (b) one-page calls turned item errors | all | | none |
| RI-197 (c) missing-holes entries (describer part; the hand-back has no text layer) | R21 bound | 0 | 8 entries naming all 12 lost pit pages |
| RI-197 (c) | R15, R30, R21 parent | 0 | 0 (no false entry across 166 recorded logs) |
| RI-197 (d) dividers set aside | R21 (parent and bound windows) | 26 | 25: the photographed pit page opens a one-page log group |
| RI-197 (d) | R15, R30 | | unchanged |
| RI-197 (e) divider notes | R30 | none | 1 note, 6 calculation pages |
| RI-197 (e) | R21 parent, R21 bound | none | 1 note each, 6 and 5 calculation pages |
| RI-198 instruction | | no pit rule | carries `LOG_PIT_SIZE` |
| RI-198 test pits with a width, arm A vs arm B (today's reader) | R30 | 0 of 52 vs 45 of 52 | a live run shows the width read |
| RI-198 | R15 | 0 of 8 vs 8 of 8 | |
| RI-198 | R21 | 0 of 68 vs 0 of 80 | (its scanned pits name no bucket) |

## Tests run

| Commit | Tests | Result |
|---|---|---|
| RI-196 | `report_ingest` test_model_first_pages, test_range_labels | 236 passed, exit 0 |
| RI-196 | test_graph_model_first, test_model_first_read, test_ri187, harness test_measure_grouping | 116 passed, exit 0 |
| RI-197 | test_graph_model_first (off snapshot included), test_model_first, _schema, _read, _pages, _scoring, test_log_holes, test_ri187, harness test_measure_grouping | 473 passed, exit 0 |
| RI-197 | test_cluster_scoring, test_log_reader, harness test_run_plain_baseline | 392 passed, exit 0 |
| RI-197 | test_graph | 85 passed, exit 0 |
| RI-198 | test_model_first_schema, test_model_first, test_model_first_read | 92 passed, exit 0 |
| RI-198 | test_graph_model_first, test_ri187 | 67 passed, exit 0 |
| RI-198 | test_model_first_pages, test_model_first_scoring | 142 passed, exit 0 |
| each | `webapp` test_public_text | 4 passed, exit 0 |

## For a live run

- `READ_PROMPT_VERSION` is now mf-7: every saved model-first result is read
  again (as at every bump).
- `DESCRIBE_PROMPT_VERSION` is now ri73-4: a run resumed over a folder
  described under ri73-3 refuses the saved descriptions, so the range voter
  does not run there. Start the next run in a fresh folder.

## Noted, not fixed

- A describer field listing two lettered pits after one word ("Pit K, M")
  reads as one name with the letters joined. Older than RI-196; no saved
  report gives one.
- A describer field that is only a note (", depth 3 m") reads as a name
  ("DEPTH 3"). Older than RI-196; the describer is now asked for the name
  alone.
- The divider notes (RI-197 e) do not send the page to a reader. If a live
  run shows such a page holds results, that is a follow-up.
- A page a saved MF-3 answer named as another log only in prose is not
  re-read from that answer: mf-6 asks for the list, and the missing-holes
  check names the page meanwhile.
- The missing-holes replay had no text layer, so a text page's own label
  name was not exercised against the describer's; a page with a label name
  is never given the describer's.
