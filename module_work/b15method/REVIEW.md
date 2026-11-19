**Verdict: MERGE AFTER FIXES, the fixes on this branch** (`claude/rv-b15-method-revisions`: f03b46f and 00c02f2 on the reviewed head aa1e917). No blocking finding: the page's tools can do what the clause asks.

# Review of brief 15: the review method's clause on revisions (BACKLOG DR-06)

Reviewed: 9f6af0a and aa1e917 on 9367934 (`main.md` +3 -1, and the check file). Reviewer model: claude-fable-5-2; not the builder.

## Findings

| # | What | The made-up input | What the tools or the check returned | Fixed or noted |
|---|---|---|---|---|
| 1 | The page can compare two issues; nothing refuses. | `store_issue0.pdf` and `store_issue1.pdf`, three sheets each (G-001 cover with index, A-101 plan, S-001 notes). A-101: the second bay dimension 6000 to 6400, no cloud, the sheet still REV 0 with one revision row. S-001: note 2, 25 MPa to 30 MPa, inside a drawn cloud, row "1 REVISED CONCRETE STRENGTH". Both held as uploads (the attachments table), default build, stand-in model. | `open_document` twice: two handles, 3 pages each; the first handle answered `read_document` and `search_document` after the second file was open. `read_document(pages="1")` on each: "A-101 REV 0", one revision row in both, "6000 / 6000" against "6000 / 6400". `read_document(pages="0-2")` on each: the whole set in one call, with S-001's new row and 30 MPa. `search_document("6400")`: one hit on issue 1 (page 1, with its box), none on issue 0. `document_markups` on issue 1: 0 (the cloud is linework). `render_page_thumbnails`, `analyze_pdf_page` on the same page of each file and `render_region` on the clouded note all answered. | Noted. A careful reader of the two `read_document` results finds the unflagged change: the value differs and the sheet's revision and rows do not. |
| 2 | What the walk cannot show. | The same. | The stand-in model returns "done" for every view, so whether a model sees the cloud, or a change in linework alone, is not shown here. A change that is not in the text layer (a moved wall, a scan) is found only by the views. | Noted; see "Not checked". |
| 3 | "the text of each and then a view of each" reads as two views of every sheet (400 looks on a 200-sheet set). | Reading the clause as a model would. | n/a | Fixed, f03b46f: views where the text differs, where a cloud or a revision row points, or where there is no text layer; the sheets compared from text alone are named, so the answer cannot pass for a full comparison. |
| 4 | "one issue only: say that unflagged changes cannot be checked" applied to every review of one issue and gave the sentence no place. | Six made-up answers through `check_no_false_alarm` with the checks of `coord-callouts` and `coord-schedules`. | The sentence alone in the close: passes both. In the next sentence after a sound mark: passes. In one sentence with a sound mark ("Door 105 is tagged on A-101 and scheduled on A-601, though unflagged changes to it cannot be checked from one issue"): FAILS `coord-schedules`. In one sentence with a sound callout ("1/A-301 resolves, but whether it changed ... cannot be checked ..."): FAILS `coord-callouts`. In a table row of door 105: FAILS. The check reads one sentence or one table row at a time, not a character window; "cannot" and "never" are in `_FAULTY`. | Fixed, 00c02f2: said once, in the close, and only where no earlier issue was supplied and the user asks what changed or the sheets show a revision. A made-up answer with the new sentence in its close passes both checks. |
| 5 | "unflagged" is not a drawing's word. | The clause. | n/a | Fixed in 00c02f2: "changes no cloud marks". The rest keeps the method's terms: issue, cloud, revision table, revision row, text layer, the close. |
| 6 | The suite tasks that plant an unflagged change. | The fixture's issues 0, 1 and 2 built and their text and drawn paths compared sheet by sheet. | Issue 0 to 1: text changes on the cover, A-401 (7200 to 7500, row, 27 more paths), A-601 (HW-1 to HW-2, row, 18 more paths) and S-001 (150 to 200 kPa, no row, the same 12 paths: no cloud). Issue 1 to 2: the cover, A-102 (a row only), A-201 (cloud and row), A-601 (cloud and row), S-501 (10-20M to 12-20M, 35 more paths, no row); S-001 and every other sheet identical. Of the 17 `coord-*` tasks only `coord-revisions` and `coord-reissue` supply two issues. | Noted: the check file is right. One unflagged change, in `coord-revisions`; none in `coord-reissue`; none missed. The six tasks with a `no-false-alarm` check all supply one issue. |
| 7 | A build without the method is unchanged. | n/a | `git diff 9367934 HEAD --stat`: `main.md` and the two files under `module_work/b15method/` only. `review_fingerprints.json` not touched, no re-record, the fingerprint test passes. No test added or edited. | Noted. |
| 8 | The check file's numbers are now out of date. | n/a | `main.md` is 3967 characters, not 3812; 33 are left under `CAP_CHARS`, not 188. Its "After, in full" quote is the builder's wording, replaced below. | Noted: the next addition to `main.md` must take something out. |
| 9 | The check file says the two files were "saved in one working folder". | The two files written to the build's files folder and opened by bare name, not held as uploads. | `open_document` and `analyze_pdf_page`: "not an attachment key or a readable file path". Held as uploads, as the page holds them (`core.py`, `attachments[key] = data`), every call answered. | Noted. Not a defect of the page; the check file does not say how its files were held. |

A first wording of fix 4 brought `main.md` to 4131 characters and failed `test_the_two_files_ship_with_the_package_and_fit_the_loader`; both fixes were shortened before they were committed.

## The clause as it now stands, in full

> - a revision: what the cloud and the revision table say changed, and whether the sheets that depend on it changed too;
> - two issues supplied: compare the text of each sheet in both; view both where it differs, where a cloud or a revision row points, or where there is no text layer. A change that no cloud and no revision row announces is a finding in itself. Name the sheets compared from text alone;
> - no earlier issue supplied, and the user asks what changed or the sheets show a revision: say once, in the close, that changes no cloud marks cannot be found without it. Never say there are none.

Question 2, on this wording. Nothing asks for what one issue cannot give. No sentence allows "there are no unflagged changes": with one issue it is forbidden, with two the sheets compared from text alone are named. A review of one issue that is not about a revision and shows none says nothing; one that shows a revision says one sentence, at the close. Left open on purpose: two issues supplied still means the text of every sheet of both is read (by the builder's walk, not repeated here, one read returns about eight dense sheets); the method's step 3 already has the reviewer say how much of the set a turn can cover.

Change to the method's text in all: three lines in place of one (the builder's two new lines, both reworded here).

## What ran (venv (a) of the common brief: Python 3.12, planlens 0.17.0, PyMuPDF 1.28.2)

| Command | On | Result |
|---|---|---|
| From `packages/aec-document-review`: `python -m pytest -q -p no:cacheprovider tests/test_review_method.py tests/test_review_fingerprints.py tests/test_boundary.py` | aa1e917 | 15 passed, exit 0 |
| The same | 00c02f2 | 15 passed, exit 0 |
| From the root: `python -m pytest -q --noconftest -p no:cacheprovider webapp/tests/test_public_text.py webapp/tests/test_review_freeze.py` | aa1e917, 00c02f2, and with this file staged | 13 passed, exit 0, each time |
| `git grep -l "review_method\|method_arm" -- packages webapp funhouse_agent` | aa1e917 | one test file, `tests/test_review_method.py`; nothing in `webapp/` or `funhouse_agent/` |
| The tool walk of finding 1 and the check runs of findings 4 and 6 (scripts kept outside the repository, not committed) | aa1e917; the new close sentence on 00c02f2 | as the table says |

## Not checked

- No model ran. What a model does with the clause, and what it sees in a view of a clouded or an unclouded change, is for the measurement (three repeats of the `method` arm on `ids=["coord-"]`). Read `coord-revisions` for the S-001 check knowing its question asks about clouds in both arms, and read the six `no-false-alarm` tasks for the close sentence landing in a row.
- Neither fix is measured; both are wording, read as a model would read them.
- A scan (no text layer) and a set past eight dense sheets were not walked; the builder's note that `read_document(pages="0-14")` returns pages 0 to 7 was not repeated.
- The full suites and GSE's page fingerprints were not run (`webapp/` and the core are untouched); they run on GitHub when the lead pushes the branch as `ci/<name>`.
- `general-purpose.md` is unchanged and was read only for the builder's claim that the helper does not compare; that claim holds.
