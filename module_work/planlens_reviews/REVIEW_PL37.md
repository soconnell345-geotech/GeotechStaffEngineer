<!-- Filed here by the lead on 2026-11-19 from planlens' branch `claude/rv-pl-37` (planlens keeps no review file on its main); the branch was then deleted. -->

# Review: PL-37, a sheet number taken wherever a page is

**Verdict: MERGE AFTER FIXES** — reviewed head `abd142f` (base `6196f13`). No must-fix. Three should-fix findings, all fixed on this branch (`claude/rv-pl-37`); merge this branch, not `abd142f`.

## Findings

| # | Severity | Where (at `abd142f`) | What is wrong | Input that shows it | Fix |
|---|---|---|---|---|---|
| 1 | should-fix | `tools/toolkit.py:657`, `review/conversation.py` `_one_based_pages` | The refusal where a page argument and a sheet number disagree words the page 0-based for a caller who gave it 1-based: the host layer turns `pdf_pages` into `pages` before the toolkit compares. | Asking host, `read_document(sheets="A-101", pdf_pages="3")` returned `pages '2' and sheets A-101 name different pages`, hint `give one of them, pages or sheets`. Same for `pdf_page=3` on `plan_symbols` (`page 2 and sheet ...`). | **Fixed in `b5ece53`.** The host layer tells the toolkit which argument and value the caller gave (`pages_as_given`, a ContextVar beside `sheets_at_open`); the refusal and hint now say `pdf_pages '3'`. |
| 2 | should-fix | `tools/toolkit.py:665` | Wrong page with no word. On a scan the index takes a sheet's number from the PDF's page label. Where the file's labels are out of step with the sheets, the number resolves to a page that prints another number, and `sheet_pages` gives only sheet and page. | `build_sheet_set(scanned=[1, 2])` with pages 1 and 2 labelled the other way round: `sheets="A-101"` read page 2 (which prints A-102), `sheet_pages: [{"sheet": "A-101", "page": 2}]`. | **Fixed in `83039e1`.** The row carries `from` ("the PDF's page label, the title block not read"), the words `sheet_index` rows already use. Whether such a number should be refused instead is a design decision, left to the owner. |
| 3 | should-fix | `review/conversation.py:1260` `_side_document` | `sheets` is dropped without a word on a Word, Excel or DXF file: the call returns before the toolkit, so the whole file is read from the top. This contradicts "refused, never guessed", and a model will read `sheets` on a workbook as worksheets. | Asking host, `read_document(handle="plan.dxf", sheets="S-201")` returned the whole file; `search_document` likewise searched all of it. | **Fixed in `24b1e7e`.** Refused with words (no pages, no sheet index; worksheets are not sheets here), only for a host that asked. A host that did not ask is answered as at the base. |
| 4 | note | `tools/tests/test_sheet_tools.py:77` | The edit is forced, but it weakens the test: the `openai` style of an asking toolkit is no longer compared anywhere, and `with_sheet_args` (the code under test) is the oracle. | — | Not fixed (no behaviour to fail a test). Add `asked.specs("openai")[i]["function"] == asked.specs("plain")[i]` to the new spec test. |
| 5 | note | `tools/toolkit.py` `_sheet_addresses` | A number read by "largest lettering in the title-block corner" (index confidence 0.5) resolves with no word in `sheet_pages`. | Not reproduced on a made-up set. | Design decision: say it as finding 2 does, or leave to `sheet_index`. |
| 6 | note | `tools/sheet_addresses.py:59` `OTHER_WAYS` | Every refusal's hint offers `pdf_pages`, which a toolkit used without the host layer does not take. | `ReviewToolkit(sheet_tools=True)`, `sheets="Z-999"`. | Wording choice for the owner. |
| 7 | note | `tools/toolkit.py:686` `_with_sheet_pages` | `sheet_pages` is left out without a word when it would pass the size limit. The result's own page keys still say which pages were read. | Not reproduced (the limit floor is 1000). | None needed now. |
| 8 | note | `review/conversation.py` `_side_document` | `pages` on a Word, Excel or DXF file is also dropped without a word. Same at the base, so outside this branch. | `read_document(handle="plan.dxf", pages="7")` reads the whole file. | A backlog line, if wanted. |

## Paths checked

- **A host or toolkit that did not ask (look-hardest 1): holds.** `/tmp/rv/cmp.py` ran in two subprocesses, one on `6196f13` and one on the branch: every spec in three styles and 8,640 made-up calls (6,636 giving `sheets` or `sheet`) through `ReviewToolkit()`, `ReviewToolkit(sheet_tools=False)`, `ReviewHost()`, a plan-tools-only host, `sheet_tools: False`, and a plain host sharing a conversation with an asking one; through `call`, `call_json` with and without `max_chars`, `review.call` and `dispatch_document_tool`. The only differences were temp folder names, which also differ between two runs of the base. Identical again after the three fixes.
- **Wrong page, silently (2).** Matching is exact after `normalize_number`: with S-201, S-201A and S-2 in one index each resolves to its own page; S-20, S-201B and S201A are refused (the last names S-201A, does not take it). A number on two pages is refused with both pages. `"S-201 A-101"`, `"S-201/A-101"` and `"A-101 thru A-105"` are refused as one unknown number. The index is always built on the whole document, whatever `pages` the call gives. There is no list-of-drawings source; the one non-title-block source is the PDF page label (finding 2).
- **Disagreement with `pages` / `page` / `pdf_pages` / `pdf_page`:** refused in each form; all three of `pages`, `pdf_pages`, `sheets` is refused by the existing rule.
- **Tools reached:** all toolkit tools that take `pages` or `page`. `find_like` takes `sheet` for the example and `sheets` for the pages searched. A host is refused `sheets` on `annotate_document`, `document_structure`, `sheet_index`, `find_like`, and on plan tools (`sheet` only there).
- **Shared host:** both switches are ContextVars set per call; no new module-level table. The first sheet-addressed call reads every page under the document's lock, as `sheet_index` already does.
- **Merge with `origin/claude/b13-integration` (5):** `git merge --no-commit` is clean for `abd142f` and for this branch's head (auto-merged `CHANGELOG.md`, `document/DESIGN.md`, `review/conversation.py`). On the trial merge, `planlens/review/tests` and `planlens/tools/tests` pass. No merge was pushed.

## Tests

- Branch at `abd142f`: the two new files and the two edited files, 41 passed.
- On the base: `test_pl37_sheet_addresses.py` fails at import (the module does not exist), so none of its 9 tests can pass; `test_pl37_sheet_addresses_host.py` 4 of 4 failed.
- This branch's head: `pytest planlens -q`, exit 0, 2408 passed, 6 skipped.
- Each fix has a new test file that failed before its commit: `review/tests/test_pl37_review_pdf_pages_wording.py`, `tools/tests/test_pl37_review_page_label_said.py`, `review/tests/test_pl37_review_side_documents.py`. In the last, the second test (a host that did not ask) passes before and after by design.
- planlens has no fingerprint test of its own; the app's is under "Could not check".

## Scope

- Files changed are the ones the prompt names, plus `CHANGELOG.md`, two `DESIGN.md` files and two new test files. Nothing outside PL-37.
- Edited existing tests (3): `tools/tests/test_sheet_tools.py:77` is forced and weakened (finding 4). `review/tests/test_sheet_tools.py:57` is forced (it drops `sheets` before comparing; the new host test checks the `sheets` schema). Lines 70 and 76 are forced (the property lists gain `sheets`). Nothing else is weakened.
- My fixes touch `tools/toolkit.py`, `review/conversation.py` and three new test files. `CHANGELOG.md` and the `DESIGN.md` files do not yet mention `from` in `sheet_pages`, the side-document refusal or `pages_as_given`.

## CHECK.md

The branch has none. The prompt's claims were checked by running; all hold except "never guessed" on a Word, Excel or DXF file (finding 3) and the 0-based wording (finding 1).

## Could not check

- **The app as a consumer.** Not run: Document Review's tests and fingerprints against this branch. Build venv (a) from the GeotechStaffEngineer checkout, `pip install --no-deps --force-reinstall -e <this checkout>`, run `tests/test_review_sheet_tools_offline.py` and `tests/test_review_fingerprints.py` with `GEOTECH_REVIEW_SHEET_TOOLS` unset (fingerprints must not move) and set (tool schemas gain `sheets`; a re-record is expected).
- **A model's use of `sheets`**, and real drawing sets: every check here is on made-up sets.
