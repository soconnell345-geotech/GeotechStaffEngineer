# EV-18: an image file is not a document with pages (check file)

Branch `claude/ev-18-image-files`, from `claude/b13-integration` at f150fab. Fix and tests: cd6f0d5. Source: finding 4 of `module_work/b13_whole/REVIEW_B.md`.

## What was wrong

Confirmed on the head before changing it (`aec_agent_core/cited_pages.py`, `check_answer`). A `render_region` on an uploaded picture is recorded by `pages_opened` as page 0, reached by a look, of a document `site.png`. `check_answer` took that as one of the conversation's documents:
- "PDF page 1" resolves to page 0 of ANY document (`status(0, None)`), and page 0 of `site.png` was looked at, so it counted as opened;
- `labels()` returns nothing unless every document's printed labels can be read; the picture has none, so every document's labels came back unknown and plain "page N" citations went to `unverified`.

Made-up input: `open_document` of `made_up_report.pdf` (60 pages, each printing its viewer number), `read_document` of pages 4-5, and the answer "The bearing value is on PDF page 1 and the settlement on page 12."

| Record | Head | This branch |
|---|---|---|
| the report alone | `never_opened` = "PDF page 1", "page 12" | the same |
| plus one `render_region` on `site.png` | `never_opened` empty; `unverified` = "page 12" | `never_opened` = "PDF page 1", "page 12" |

## What changed

`packages/aec-agent-core/aec_agent_core/cited_pages.py`, 24 lines added and 1 changed, none in the region of `_described` and its helpers:
- new `IMAGE_SUFFIXES`, `_is_image`, `without_images(opened)` after the `PagesOpened` class: a copy of the record without the keys that are image files (`pages`, `names`, `sources`, `facts`, `n_pages`, `markups`, `sheet_seen`);
- `check_answer` reads `without_images(opened)`, whether the record was handed in or read there.

A repair, behind no setting. `pages_opened` itself is unchanged. No prompt, tool description, setting or fingerprint moved; the new names are not in `__all__`.

## The two neighbours

1. **Only an image file, and the answer says "page 1".** Head: nothing listed ("page 1" and "PDF page 1" both counted as opened by the look at the picture). Branch: "PDF page 1" is in `never_opened` (the picture is not a document whose pages can be cited so); plain "page 1" is in `unverified`, the same as a conversation that read no document at all. So with `GEOTECH_CITED_PAGES` on `note` or `retry`, "PDF page 1" said of a lone picture is now a listed citation (not run through a turn here).
2. **`cited_documents.py`: not the same defect, not changed.** It goes by what a tool result SAYS, never by which file was looked at. A look at `site.png` is a source for "GEC-11" only when the look's own result names GEC-11 (a picture of a cover); otherwise GEC-11 stays in `not_looked_up`. A test states both.

## Calls made

- The filter sits in `check_answer`, not in `pages_opened`: the photograph-page check builds its attachment list from the same record and reports an uploaded image under `left_out`; that stays as it was.
- An image file is told by its extension, the same eight as Document Review's `review_page.IMAGE_SUFFIXES` (the core cannot import that package). `.tif` / `.tiff` are in: a multi-page TIFF is treated as a picture.

## New tests (`packages/aec-agent-core/tests/test_ev18_image_files.py`, 7)

Run on the head with the fix stashed: 6 failed, 1 passed.
- `test_a_zoom_on_a_picture_opens_no_page_of_the_report`: the made-up case. Fails before (`never_opened` empty).
- `test_a_picture_does_not_make_the_printed_labels_unknown`: labels are asked of the report only. Fails before.
- `test_a_picture_opened_by_handle_is_left_out_too`: `open_document` on `Site.PNG`, then a zoom by handle. Fails before.
- `test_only_a_picture_pdf_page_1_is_no_page_of_it`, `test_only_a_picture_page_1_is_as_with_no_document`: neighbour 1. Both fail before.
- `test_the_record_of_pages_looked_at_still_holds_the_picture`: fails before only because `without_images` does not exist; its other lines hold on the head.
- `test_cited_documents_a_look_at_a_picture_names_no_other_document`: neighbour 2. PASSES before: it states behaviour, it does not show a defect.

## What I ran (all exit code 0)

- venv (a), from `packages/aec-agent-core`: `tests/test_ev18_page_check_repairs.py tests/test_rv_ev18_review_fixes.py tests/test_ev18_image_files.py`: 83 passed.
- venv (a), from `packages/aec-document-review`: `tests/test_cited_documents_turn.py tests/test_cited_pages_turn.py tests/test_review_fingerprints.py tests/test_boundary.py`: 26 passed; `tests/test_dr48_photo_pages_turn.py`: 7 passed.
- venv (a), root, `--noconftest`: `webapp/tests/test_public_text.py webapp/tests/test_review_freeze.py`: 13 passed.
- venv (b), root: `funhouse_agent/tests/test_cited_pages.py webapp/tests/test_cited_pages_turn.py webapp/tests/test_cited_documents_turn.py` with the three core files: 145 passed; `webapp/tests/test_page_fingerprints.py`: 13 passed (fingerprints not moved); `packages/aec-agent-core/tests/test_dr48_photo_pages.py test_vis33_photo_pages_one_look.py funhouse_agent/tests/test_cited_documents.py`: 157 passed.

No existing test was edited.

## Release note, noted and not fixed, not checked

- Release note: with `GEOTECH_CITED_PAGES` unset nobody sees a change; the `cited_pages` event of a conversation that looked at a picture now lists the cited pages no tool opened, where it listed none.
- Noted, not fixed: a picture looked at by a handle the record never named (no `open_document` in the log) keeps the handle as its key and is not told as an image; a look at a PNG that some tool rendered from a PDF page grants no page of that PDF (before, it granted page 1 only).
- Not checked: no model run and no hosted run; the full suites (GitHub, a `ci/` branch) are the lead's. The other branch's change to `_described` was not merged here: this branch adds lines only near 393 and 1997 of the head's file.
