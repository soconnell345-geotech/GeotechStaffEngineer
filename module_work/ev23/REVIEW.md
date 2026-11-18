# EV-23 review: the cited-documents check repaired

## 1. Verdict

`MERGE AFTER FIXES`, reviewed head `cbfe267` (base `f0b371e`). The two fixes
are on this branch (`claude/rv-ev-23`): merge it in place of the reviewed
branch. Three recall questions are left for the lead (findings 3 to 5).

## 2. Findings

| # | Severity | Where | What is wrong | Input that shows it | Fix |
|---|---|---|---|---|---|
| 1 | must-fix (older than this branch, REF-114) | `cited_documents.py`, `_note_call` | A `read_reference_table` whose every page failed has no top-level `error`, so the call's `reference` was still added as looked up. | Result `{"reference": "gec_11", "pages_read": [{"error": ...}, {"error": ...}]}`, answer "(GEC-11, Table 9-9)": `looked_up`, "a references result". | **Fixed, `ed7057a`**: `_read_nothing` leaves the read out when every `pages_read` row has an `error` and no `analysis`. One page read still counts. |
| 2 | should-fix | `cited_documents.py`, `_SAYS` | "X, which ..." keeps a citation after a pointer verb or a negation; the same sentence with "whose" was dropped. | "Consult GEC-15, whose Table 4-1 gives this spacing." was a pointer; "I have not found any conflict with GEC-15, whose Table 4-1 gives this spacing." a denial. | **Fixed, `0725a58`**: `_SAYS` reads "which" and "whose". |
| 3 | should-fix, needs a design decision, not fixed | `_DENIAL_BEFORE` | A negated verb takes a document that is the subject of the next verb, where the sentence says what the document holds. Kept in the record as `denial`, `looked_up: false`. CHECK.md gives the rule but not this loss. | "We did not apply the reduction GEC-15 allows." "The analysis does not include the surcharge GEC-15 requires." | Needs a rule for "what follows the mention is a verb" (a verb list, or a word class). Either is a design choice. |
| 4 | note, the lead's call | `_pointed` | The commonest way an answer states its basis is lost when no value stands in the clause. CHECK.md says so for the first form; the record keeps each as `pointer`, `looked_up: false`. | "Use the GEC-15 method for the drains." "Drain spacing: use GEC-15." "Design basis\n- Use GEC-15 for drains\n- Use ASCE 7-22 for loads" "The method to use is GEC-15." "You should use GEC-15 because it covers drains in clay." "We recommend using the GEC-15 spacing of three metres." | None made. The hosted measurement should count these rows before the switch goes on anywhere. |
| 5 | note | `_number_pair` | An author and year with a page and no "p." is no longer read at all, so it is in no list of the record: this loss cannot be measured. CHECK.md says it is not checked. Matters little: the form is rare in answers, and "(Lind 2004, 12)" inside brackets still reads. | "Lind (2004, 12)", "Lind and Hale (2004, 12)", "Lind et al. (2004, 12)", "Lind (2004a, 12)", "Lind (2004, 3) and Hale (1999)" (Hale still reads). | A record row for a dropped pair would make it measurable. |
| 6 | note | `_note_call`, `Marks.add_text(not_from=)` | A designation or an author and year split over two arguments, each under 8 characters, is credited when the package prints them together. CHECK.md's "however short" is untrue for this. Needs a model to split its own citation over two fields. | `{"series": "GEC", "number": "15"}` printed "Reference GEC-15"; `{"by": "Okafor", "date": "2011"}` printed "By Okafor 2011", answer "Okafor (2011)". | None made. |
| 7 | note | `_decoded` | A string field that itself holds JSON text is not decoded a second time, so J still hides a designation after `\n` there. The logger keeps a result as one level of JSON, so no saved record is known to hold this. | `{"result": "{\"text\": \"soil\\nASTM D9999\"}"}`: not looked up, as on the base. | None made. |
| 8 | note | `cited_and_mentioned` | Reading each mention costs time that grows with the square of the citation count. | 3,000 lines, 9,000 mentions, 387,000 characters: 9.2 s (finding them alone: 1.1 s). At 900 mentions about 0.1 s. | None needed at real sizes. |
| 9 | note (older than this branch) | `_json`, `_feed` | JSON nested about 5,000 deep raises `RecursionError` in `json.loads`; `looked_up` swallows it after the offset has moved, so the rest of that chunk is lost. | `"[" * 5000 + '"x"' + "]" * 5000` as a result. | None made. |

## 3. Paths checked

1. **Recall** (56 made-up answers, `/tmp/rv/recall.py` and `recall2.py`).
   Still flagged: "Per X, verify ...", "X governs unless ...", "as X
   requires", a citation and a pointer to one document in either order, a
   denial of one document beside a citation of another, "Design the wall
   using X", "was designed using X", "We used X", "should follow X",
   "Check: ... meets the X minimum", "confirmed against X", "Unless noted,
   all criteria are from X", "I checked ... against X and it complies",
   "can be taken from X", "X is not used because ...", "not applied,
   following X", "As X requires, use a filter", "Confirm the spacing of
   3.0 m with X", an author and year after "Using" or "by using". Lost:
   findings 2 to 4, and "Whether or not the wall is braced GEC-15 requires
   drainage" (a condition with no comma) and "The design is not found
   wanting under GEC-15". Every lost one is a `mentioned_not_cited` row.
2. **Decoding.** No two fields join into a designation: `{"org": "ASTM",
   "number": "D9999"}`, `["ASTM", "D9999"]`, `{"ASTM": "D9999"}`, `{"GEC":
   15}`, `{"UFC": "3-999-01"}`, `{"ACI": 318}`, `["UFC", "3-999-01"]` and
   eight more name nothing, as on the base. A result that is not JSON is
   read as it came; a JSON string is decoded; a bare number is left. An
   author and a year pair as on the base for one-line JSON, and now also for
   indented JSON (two records within 90 characters). A 9 MB result decodes
   in 0.9 s (sidecars over 4 MB are not read); 400 levels deep is fine.
3. **Calculation packages** (16 cases). Not credited: a title of 8
   characters or more, a short "GEC-15", an alias in the arguments against
   the designation printed, a wrapped or upper-cased title, HTML in an
   argument printed as text, the package call's own echo, a reader's search
   query, another module's report with the model's title, a read by handle,
   a file `html_to_pdf` wrote over the package's. Credited: the package's
   own basis line. Let through: finding 6.
4. **Author and year.** Two authors, "et al.", a year with a letter, "p."
   after the year, two years, a page range: all read. Particle surnames
   read whole first or second, in brackets, with "et al." and with a
   lettered year. Lost: finding 5.
5. **Parts.** Whole from a part, a part from the whole: credited. Part 2
   from part 1, part 1 from part 10, "ISO 7777" from "ISO 77777-1": flagged.
   As CHECK.md says.
6. **Failed table read**: finding 1.
7. **Switch unset.** `finish` returns the answer itself and `feedback`
   `None`; `check_deliverable` adds nothing; both fingerprint files are
   unmoved and both tests pass. `find_citations` without `refined` uses the
   same patterns as the base, so the judgment-call check's note does not
   move (its two test files pass). Under `note` and `retry` a mention gets
   no note and a citation still does.
8. **Readers of the record.** No code reads the event's keys: `run_report`
   and `activity_log` only name the event. Nothing breaks on
   `mentioned_not_cited`, nor on `cited` being absent when a text only
   mentions.
9. **Merge.** Into `origin/claude/b13-integration` at `ce970b5` (merge base
   `f0b371e`): automatic, no conflicts; the four citation test files pass
   on the result (181). Not pushed.
10. **Shared host.** One new table, `_LookupState.app_written`: per
    activity file, cleared by `reset`, written under the state's own lock.
    No new lock, thread or wait.

## 4. Tests

Venv (b), repository root, on the reviewed head:
`pytest packages/aec-agent-core/tests/test_ev23_document_check_repairs.py
packages/aec-agent-core/tests/test_ref114_table_reads.py
funhouse_agent/tests/test_cited_documents.py
webapp/tests/test_cited_documents_turn.py
webapp/tests/test_judgment_check_turn.py
funhouse_agent/deep/tests/test_judgment_calls.py
webapp/tests/test_page_fingerprints.py`: 239 passed.

Venv (a), `packages/aec-document-review`: `tests/test_cited_documents_turn.py
tests/test_review_switches_offline.py tests/test_review_fingerprints.py
tests/test_boundary.py`: 27 passed. The new core file in venv (a): 56 passed.

On the base (`/tmp/base`): the new file as committed fails at collection
(`cd.POINTER` does not exist there), so all 56 error. With the four new
constants replaced by their strings: 32 failed, 24 passed, as CHECK.md
says. The 24 are the cases that must still fire, by design. The tests check
what the check returns, not text.

After the fixes, on this branch: the same venv (b) list plus
`test_rv_ev23_review_fixes.py`: 245 passed; venv (a)
`test_cited_documents_turn.py`, `test_review_fingerprints.py`,
`test_boundary.py`: 18 passed; the guards (`test_public_text.py`,
`test_review_freeze.py`): 13 passed. Fingerprints unmoved.

The new file `packages/aec-agent-core/tests/test_rv_ev23_review_fixes.py`
holds six tests. Without `ed7057a`,
`test_a_table_read_whose_every_page_failed_is_no_source` fails. Without
`0725a58`, the two "whose" cases fail. The other three pass either way and
guard what must not change (one page read, "which", a plain pointer).

## 5. Scope

Three files, all in scope: the core module, one new test file, CHECK.md. No
existing test edited. No file of the freeze list touched. No new setting.
Made-up values only. Nothing worded around one model. The parts rule and
the indented-JSON pairing go a little beyond J as the item words it; both
are in CHECK.md.

## 6. CHECK.md

- "nothing the call's or the reader's arguments cite counts ... however
  short": untrue for a citation split over two short arguments (finding 6).
- "On the starting code 32 fail": true only once the new constants are
  replaced; as committed the file does not collect there.
- "J lets nothing new through": holds for every join tried; it does not say
  that JSON text inside a string is still read raw (finding 7).
- The denial rule is stated, but not that it takes "the reduction X allows"
  (finding 3).
- The rest was supported: the 56 cases, the other suites, both
  fingerprints, the record key, the parts directions.
- Release note: "None" is right while the switch is unset. Under `note` or
  `retry` a tester would see fewer notes; CHECK.md's settings section says
  so.

## 7. Could not check

- How often findings 3 and 4 happen on real answers. Run the hosted
  measurement CHECK.md describes (`GEOTECH_CITED_DOCUMENTS=note`, the same
  tasks before and after) and read every `mentioned_not_cited` row with
  `looked_up: false` by hand.
- The table reader's result on a live host when every page fails: the fix
  reads the shape `funhouse_agent/reference_vision.py` builds (`pages_read`
  rows with `error`). Confirm on one saved record of such a read.
- The full suites: push this branch as `ci/<name>`.
