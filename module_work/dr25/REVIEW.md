# DR-25 and EV-49: recheck of `claude/dr-25-suite-gaps` (2026-11-18)

## 1. Verdict

`MERGE AFTER FIXES`, reviewed head `abeca33` (base `a7a48d8`). The fixes are
on `claude/rv-dr-25`, six commits before this file; merge that branch in
place of the reviewed one. Three things are left for a design decision
(findings 7 to 9). EV-49 holds as built.

## 2. Findings

| # | Severity | Where | What is wrong | Input that shows it | Fixed by |
|---|---|---|---|---|---|
| 1 | must-fix | core `review_eval/checks.py`, `_together_in_findings` | A term was first looked for in all pieces joined by spaces. A term that ends where its piece ends (a row's last cell answering its head with No) was then followed by the next row and called missing: a yes/no table passed only when its No row came last. | `\| Sheet \| Item \| Carried? \|` rule, `\| A-301 \| Parapet 109.30 \| No \|`, `\| A-101 \| Stair \| Yes \|`: parapet-twin failed; with the two rows swapped it passed. | `c25846d` |
| 2 | must-fix | `tasks.py`, `_LEFT_BEHIND`; `door-renumbered` | Plain right findings failed: parapet-twin 5 of my 11, door-renumbered 2 of 8. | "A-301 shows the old parapet level of 109.30"; "The parapet change on A-201 was not picked up on A-301"; "The sections (A-301) contradict the elevations"; "should read 109.60"; "A-101 shows door 106, which is no longer in the schedule"; "Door 106 on A-101 has no row in the door schedule" (the fixture's own consequence). | `e5b9f2c` |
| 3 | must-fix | `tasks.py`, `_NOT_GIVEN`, `_NOT_A_SECTION` | "Looked for and not found" failed in three checks: anchorage 3 of 9, section-missing 3 of 11, water-test 1 of 9. "Does not resolve" failed though the task's question asks what "does not resolve". | "I could not find the anchorage calculations anywhere in the package"; "Section 08 80 00 cannot be found in the manual's table of contents"; "The reference to Section 08 80 00 does not resolve"; "The laboratory report leaves out the water penetration test". | `fd8c704` |
| 4 | should-fix | `tasks.py`, `_NO_CHANGE` | "Backed", "supported" read only as a column head answered No. | Table row "A-102 \| DOOR 205 RELOCATED \| Entry is not backed by any change on the sheet". | `f206f00` |
| 5 | should-fix | `tasks.py`, `_UNLISTED` | A hedge passed cloud-not-listed: the "not" of "could not read" was taken as the absent entry. | "S-501: 12-20M at F3 is clouded; I could not read the revision block, so I cannot say whether it is listed." | `78128b3` |
| 6 | should-fix | `tasks.py`, `_LEFT_BEHIND` | "Missed", "overlooked", "left behind" had no guard against their denial. | "No sheet was left behind: the parapet on A-301 matches A-201"; "A-301 parapet: nothing was missed". | `8ed6039` |
| 7 | design decision, not fixed | all four word checks | A finding stated and then taken back in the same paragraph passes (5 of 9 checks on my replies). | "The water penetration result seemed to be missing, but it is on page 2 of the report: no leakage." Also "08 80 00 seemed to be missing from the contents, but it is there". | Open. A rule ("seemed ... but") would be fitted to wording; the docstring's own answer is the full read of the run. The lead decides whether the suite needs a taken-back guard. |
| 8 | design decision, not fixed | core `_findings` (and `_items`, same cut on the base) | Each plain line is its own finding. Label lines under a bold heading, a hard-wrapped paragraph and a list item continued on an unindented line are cut apart, and the heading rule helps only when the heading holds a term. | "**U-value**" / "Specified: 1.60 max." / "Submitted: 1.90 whole window." fails window-u-value; the same shape fails window-sov-total; "- A-301: the sections" + newline + "still show the parapet at 109.30" fails. | Open. Joining consecutive plain lines would also join "A-301: no change" to the next sheet's line, which the builder cut apart on purpose. Either answer changes scores; it is the lead's call. |
| 9 | note | `window-u-value`, `window-sov-total` | Two numbers side by side is the whole check: "1.90 is within the 1.60 limit" passes. Same design as the older `strength-record`. | as stated | Not fixed: nothing in the items asks for a verdict word. |
| 10 | note | `_NOT_GIVEN` | The right words about another thing on the same line pass. | "Section 08 80 00 glazing: the warranty was not included."; "The anchorage calculations do not show any overstress." | Not fixed: one finding is the scope by design. |
| 11 | note | `parapet-twin` | "A-301 was unchanged and shows the parapet at 109.30 where A-201 shows 109.60" fails: "unchanged" alone is left out on purpose (a list of unchanged sheets says it of each). | as stated | Not fixed; a reviewer who names both values and no fault word is the cost of that guard. |
| 12 | note | core `_closest` (unchanged since the base) | Time grows with the square of the matches: a 200,000-character reply of nothing but the two numbers takes 4.5 s per check. | `"109.30 1.90 1.60 178,500 174,500 " * 6100` | Not fixed: no real reply is that shape. |

## 3. The nine checks against my own replies

Replies in my words (a sentence, a list item, a nested list, a yes/no
table row, a table row with words, the finding among ten others, and free
variants; wrong: the opposite, another sheet or section, a denial, a hedge,
the question restated, a finding stated then taken back, the ten other
items alone, wrong yes/no rows).

| Check | Right passed, reviewed head | Wrong passed, reviewed head | Right passed, fixed | Wrong passed, fixed |
|---|---|---|---|---|
| parapet-twin | 6 of 11 | 1 of 11 | 10 of 11 | 0 of 11 |
| cloud-not-listed | 8 of 8 | 2 of 10 | 8 of 8 | 1 of 10 |
| entry-no-change | 8 of 9 | 1 of 11 | 9 of 9 | 1 of 11 |
| door-renumbered | 6 of 8 | 0 of 10 | 8 of 8 | 0 of 10 |
| window-u-value | 6 of 8 | 0 of 7 | 6 of 8 | 0 of 7 |
| window-anchorage | 6 of 9 | 2 of 10 | 9 of 9 | 2 of 10 |
| window-section-missing | 8 of 11 | 2 of 9 | 11 of 11 | 2 of 9 |
| window-water-test | 8 of 9 | 1 of 10 | 9 of 9 | 1 of 10 |
| window-sov-total | 6 of 7 | 0 of 7 | 6 of 7 | 0 of 7 |

What still fails or passes wrongly after the fixes is findings 7 (five
wrong replies), 8 (three right replies), 10 (two wrong) and 11 (one right).
The yes/no tables above happened to end on their No row; finding 1 came
from adding a row after it.

## 4. Paths checked

- **Regexes on a long reply.** Nine shapes of 200,000 characters (one
  paragraph, words with no punctuation, one word, a run of spaces, 5,000
  list lines, 6,500 table rows, digits, "carried: " repeated, "not been
  yet also" repeated) against all nine checks, before and after the fixes:
  nothing over about 1 s except finding 12. No runaway backtracking.
- **Shapes.** Windows line ends (list and table), a table with no header
  rule, empty cells, a short row, more cells than heads, an indented
  continuation, a numbered item, a heading that holds the sheet, a
  delivered file after the break mark, an empty reply and `None`: all as
  expected. A list item continued on an unindented line: finding 8.
- **Fitted to an example?** The checks name the fixture's sheets and
  values, as every older coordination check does; the words around them
  were too narrow for a plain finding (findings 2 to 4), not fitted to a
  model. No prompt or tool wording changed.
- **The scorer and the prompts.** `_PLANTED_CHECKS` is fixed text; no check
  reads the fixture or `truth` at run time. The two questions name no
  sheet, value or section. The reissue question lists four kinds of fault
  to look for (clouded, listed, entry matches a change, carried to twins)
  and the fixture plants one of each: a checklist, not an answer.
- **Older checks.** 52 older tasks: every field identical on `a7a48d8`,
  `abeca33` and the fixed head (one hash). Six made-up replies per task
  (the truth as prose, as a list, as a table row, reversed, an all-clear,
  a refusal), 828 check results over all 52 tasks: identical on all three.
  The seven older fixtures build to the same bytes on all three.
- **The three edited tests.** Only 52 to 54, 14 to 16, "fourteen" to
  "sixteen", and the planted-key set gaining the two new groups.
- **EV-49.** `SETTINGS_ENVS`, `ALL_ENVS` and `ARMS` hash the same on base
  and head. With the environment at 5: a baseline arm sees no ceiling
  (base: 5 leaks in); an arm naming 3 sees 3 and `describe()` ends with
  `GEOTECH_TURN_MAX_IMAGE_CALLS=3`; after the block, and after a block
  that raises, the environment is 5 again.
- **Settings the suite still does not clear between arms** and that change
  what a run returns: `GEOTECH_HELPER_MAX_MODEL_CALLS`,
  `GEOTECH_MAX_HELPERS`, `GEOTECH_VISION_POLICY`, `GEOTECH_VISION_BUDGET`,
  `GEOTECH_VISION_DETAIL`, `GEOTECH_VISION_MAX_PX`,
  `GEOTECH_VISION_MAX_TOKENS`, `GEOTECH_VISION_MAX_PER_CONVERSATION`,
  `GEOTECH_VISION_REPEAT_CACHE`, `GEOTECH_CHART_BUDGET`,
  `GEOTECH_ONLY_TOOLS`, `GEOTECH_INSTRUCTIONS`,
  `GEOTECH_FIND_LIKE_BUDGET_S`. (About 70 `GEOTECH_` names are read and not
  cleared; the rest are connections, storage and page furniture.) Not
  fixed: which of these an arm should inherit from the host is a decision.
- **Merge with `origin/claude/b13-integration`** (`5a69be8`), tried in a
  scratch worktree, not pushed: one conflict, `review_flags.py`'s
  `__all__`. Take the integration side and add `"LATER_SETTINGS_ENVS",`
  after `"LATER_ENVS",`. Document Review's README merged by itself (54
  tasks). `switches()` kept both sides. With that, 29 core and 83 package
  tests passed, the fingerprint test among them.

## 5. Tests

Venv (a) of COMMON_DOCREVIEW, Python 3.12.

- Reviewed head, from `packages/aec-document-review`: the builder's file,
  the three edited files, `test_review_fingerprints.py`,
  `test_boundary.py`: 204 passed, 2 skipped (public documents not in the
  checkout). Core: `test_ev49_image_ceiling_between_arms.py`,
  `test_review_checks_together.py`, and the two guards: 40 passed.
- Fixed head: the same package files plus mine, 235 passed, 2 skipped;
  core 29 passed; guards 13 passed before each commit.
- My tests, run against the reviewed head before each fix:
  `packages/aec-agent-core/tests/test_rv_dr25_findings_per_piece.py` 1 of 2
  failed (the other is the guard that a missing term is still named);
  `packages/aec-document-review/tests/test_rv_dr25_own_replies.py`: 6
  failed for `e5b9f2c`, 7 for `fd8c704`, 2 for `f206f00`, 2 for `78128b3`,
  3 for `8ed6039`; the rest are the wrong replies that must keep failing.
- Fingerprints: neither file is in the diff from `a7a48d8` to the fixed
  head; Document Review's test passes.

## 6. Scope

Files as the two items allow, plus my two new test files. The fixes touch
`review_eval/checks.py` (finding 1, `scope="finding"` only) and `tasks.py`
(the DR-25 words only). No older check, fixture, prompt or setting moved.

## 7. CHECK.md

- `module_work/dr25/CHECK.md`: its per-check counts are of the builder's
  own replies; with mine the reviewed head passed 62 of 80 right replies
  and 9 of 85 wrong ones (section 3). Its statement that a yes/no table
  row is read with its heads held only for a last row (finding 1).
- `module_work/ev49/CHECK.md`: supported as written.

## 8. Could not check

- The geotech page's fingerprint test (venv b, with the geotech app) was
  not run; its file is unmoved. The lead's `ci/` run covers it.
- Whether a model's real replies take the shapes of findings 7 and 8: read
  the first live runs of `coord-spec-submittal` and `coord-reissue` in
  full, every failing check against the reply's own text, before believing
  a score on either task.
