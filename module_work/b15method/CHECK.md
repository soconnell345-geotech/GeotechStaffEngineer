# B15: the review method's clause on revisions (BACKLOG DR-06, its stated next step)

Branch `claude/b15-method-revisions`, from `claude/b13-integration` at 9367934 (1.3.0). One commit of work, 9f6af0a, then this file. Model: claude-fable-5-2.

## Confirmed before changing
- The method is `packages/aec-document-review/aec_document_review/review_method/` (`main.md`, `general-purpose.md`, `__init__.py` with `FOLDER`, `ROLES`, `text()`).
- **The setting that turns it on:** `GEOTECH_INSTRUCTIONS`, naming that folder (`review_method.FOLDER`). The core's `aec_agent_core/deep/instructions.py` reads it when an agent is built and appends each role's file last in its prompt, under the protected rules, cut past `CAP_CHARS` = 4000. Unset, nothing is appended.
- **The suite arm:** `aec_document_review.review_eval.method_arm()` (`review_eval/__init__.py`), which returns `("method", {GEOTECH_INSTRUCTIONS: FOLDER})` for `score_review_suite(arms=[...])`.
- The clause was the last line of "Across sheets" in `main.md`, as DR-06 quotes it.

## What changed (`main.md`, +3 -1 lines; a change to text behind the setting, not a repair)
Before, in full:
> - a revision: what the cloud and the revision table say changed, and whether the sheets that depend on it changed too.

After, in full:
> - a revision: what the cloud and the revision table say changed, and whether the sheets that depend on it changed too;
> - two issues of a sheet or a set, both supplied: open both and compare them directly, sheet by sheet, the text of each and then a view of each. A change that no cloud and no revision-table row announces is a finding in itself;
> - one issue only: say that unflagged changes cannot be checked from it. Never say there are none.

**`general-purpose.md` is unchanged.** The helper is not the one who compares: its file ends "Report what the sheet shows, not whether it is right: the reviewer compares the sheets", and it already returns each sheet's revision and date, anything clouded, and each value stated more than once. The comparison of two issues is the page agent's.

Made-up example. Issues 0 and 1 of a 15-sheet set; note 2 on S-001 reads 150 kPa in issue 0 and 200 kPa in issue 1, with no cloud and S-001 still at revision 0. Before: the method sends the reviewer to the clouds and revision rows, which name A-401 and A-601 only. After: it reads S-001 from both files, sees 150 against 200, and logs "S-001 note 2: 150 kPa to 200 kPa, not clouded, no revision row". With issue 1 alone it writes, under the checks it could not make, that unflagged changes cannot be checked from one issue.

## Part 2: can the page's tools do it? Yes; nothing is missing and nothing refused
Run on the default build (`tests/page_builds.build("review_default")`, the offline stand-in model, planlens 0.17.0) with the suite's two made-up issues saved as `set_issue0.pdf` and `set_issue1.pdf` in one working folder. The calls a model would make, each of which returned a result:

1. `open_document(source="set_issue0.pdf")`, then `open_document(source="set_issue1.pdf")`: two handles, 15 pages each. The first handle still answered after the second file was opened, so both are open at once.
2. `search_document(handle=<issue 1>, pattern="kPa")`: one hit, page 10, "2. ALLOWABLE NET SOIL BEARING PRESSURE: 200 kPa."
3. `read_document(handle=<issue 0>, pages="10")` and `read_document(handle=<issue 1>, pages="10")`: the same sheet's text from each; 150 kPa in one, 200 kPa in the other.
4. `analyze_pdf_page(attachment_key="set_issue0.pdf", page=10, prompt=...)` and the same with `set_issue1.pdf`: a view of the same sheet from each; `render_region(attachment_key=..., page=10, bbox=[...], prompt=...)` zooms on either; `render_page_thumbnails(handle=..., pages="10")` and `document_markups(handle=...)` answered too.

What the walk shows about the work the clause asks for, none of it a refusal:
- No tool compares two files: the model reads each and compares. A view holds one file's page, so "the same sheet from each" is two calls, never one side-by-side image.
- `read_document(pages="0-14")` on a 15-sheet set returned pages 0 to 7 (13,665 characters): a whole-set comparison from text is at least two reads per issue.
- A cloud drawn on the sheet is not an annotation: `document_markups` returned 0 on issue 1, which has two clouds. Whether a change is clouded is known only by looking. The stand-in model reads no image, so what a real model sees in the two views is for the measurement to show.

## Part 3: which coordination tasks plant an unflagged change between two issues
The `coord-*` tasks are in Document Review (`aec_document_review/review_eval/tasks.py`, `COORDINATION_TASKS`), not in the core; the core holds the framework. Two of the 17 supply two issues:

- **`coord-revisions`** (issue 0 and issue 1): **one unflagged change**, S-001 note 2, 150 kPa to 200 kPa, no cloud, no revision row, the sheet still at revision 0 (`coordination_fixture.REVISIONS`, third row). Its check "bearing-pressure, and that it is not flagged" wants S-001, 200 kPa and a word for "not flagged" within 400 characters. The task's question already asks, for each change, whether it is clouded and listed, so the baseline arm is told to look too.
- **`coord-reissue`** (issue 1 and issue 2): **none.** Its four planted faults (`REISSUE_PLANTED`) are a flagged change not carried to its twin sheet (two of them), a clouded change with no revision row (half announced), and a revision row with no change. Its truth says every other sheet is as it was.

So a measurement of the method arm has one scored check, in one task, that the new lines bear on, and that task's question already prompts it. The other 15 tasks supply one issue: there the new third line can only add a sentence to the close.

## Tests (venv (a) of the common brief: Python 3.12, planlens 0.17.0)
- `git grep -l review_method -- packages | grep tests` names one file, `tests/test_review_method.py`.
- From `packages/aec-document-review`: `python -m pytest -q -p no:cacheprovider tests/test_review_method.py tests/test_review_fingerprints.py tests/test_boundary.py`: 15 passed, 0 failed, exit 0.
- From the root: `python -m pytest -q --noconftest -p no:cacheprovider webapp/tests/test_public_text.py webapp/tests/test_review_freeze.py`: 13 passed, 0 failed, exit 0; run again with this file staged before its commit.
- No test added and none edited. No test pins the method's wording. `test_the_two_files_ship_with_the_package_and_fit_the_loader` pins its length (at most `CAP_CHARS`) and still passes: `main.md` went from 3487 to 3812 characters.

## Fingerprints
Not moved, no re-record. `tests/review_fingerprints.json` holds `review_default` and `review_minimal` only, neither built with the method, and `test_unset_the_page_is_unchanged` passes. No prompt, tool description or schema of a build without the setting changed. `webapp/` and `funhouse_agent/` are untouched.

## Calls made
- Two bullets beside the old one, not a rewrite of it: the old clause (flagged changes and their dependants) still holds.
- "the text of each and then a view of each", following the method's own order on a sheet ("Read notes ... from text ..., then look"), and because a cloud is only seen.
- Kept to 325 characters: the loader cuts at 4000, and 188 are left. The next addition to `main.md` has to take something out, or the cap has to move in the core.
- No tool is named in the new lines (the method names only `write_todos`); the tools' own descriptions say how to open and view.

## Release note
None. Nothing changes unless a deployment names the method's folder in `GEOTECH_INSTRUCTIONS`.

## For the lead
- The measurement (three repeats of the `method` arm on `ids=["coord-"]`, the maintainer's run): read `coord-revisions` for the S-001 check against baseline, knowing the question prompts it in both arms. A fair measure of the new lines needs a two-issue task whose question does not ask about clouds; none exists, and none was added.
- In the single-issue tasks, read whether the arm's "cannot be checked" sentence lands near a sound mark and trips a `no-false-alarm` check (`coord-callouts`, `coord-schedules`, their scan and long twins): "cannot" and "never" are among that check's fault words (`_FAULTY` in `tasks.py`). Not run here.
- The prompt names the 1.3.0 head as the base; this branch is not from master.

## Noted, not fixed
- The task's wording places the `coord-*` tasks in the core; they are in Document Review.
