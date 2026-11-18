# EV-18: repairs of the cited-pages check (2026-11-18)

Branch `claude/ev-18-page-check-repairs`. Code: `packages/aec-agent-core/aec_agent_core/cited_pages.py`.
Tests: `packages/aec-agent-core/tests/test_ev18_page_check_repairs.py`.

## What was asked

The turn-end page check (`TurnPageCheck` → `check_answer`) lists a page the
answer cites that no tool opened (`never_opened`). An offline pass over saved
runs found it calling pages "never opened" that the answer did not cite as a
source. The owner's ruling: repair the check itself, no new setting; recall
first (a page no tool opened and nothing described must still fire); for each
repair, a made-up case that must stop firing beside one that must still fire;
the two right firings (a tool's 0-based index cited as the PDF page) must
still fire. Order A, C, B, D, G, H; E and F noted, not fixed.

Measured offline on saved runs; the counts are in the lead's private notes.

## What was built

All of it is read only by `check_answer`. `GEOTECH_CITED_PAGES` stays `off`
by default; with it off `finish` returns the answer unchanged, so only the
turn's recorded event changes. With `note` or `retry` on, only a
`never_opened` page makes the note, as before. What `write_diggs` is told
(`record_entries`) and the photograph-page check (DR-48) read the same
record fields as before (`PagesOpened.status`, `documents`, `looked`); the
new data sits in new fields (`facts`, `markups`, `references`,
`reference_names`) that only `check_answer` reads. The geotech page runs the
same module, so the same rules hold there.

Where a citation now goes instead of `never_opened`:

| Repair | Goes to | Example (made up) |
|---|---|---|
| A | `page_map`, with `matched` (the words that matched) | "PDF page 4 is blank." where the page map said page 4 is blank |
| B | `named_unread`, with `said` (the cue) | "PDF pages 10 and 14 were not viewed." |
| C, printed number unknown | `unverified` | "The figure is on p. 1-3." where no page is known to print "1-3" |
| C, pair | nothing (the PDF page decides) | "printed page 1-3 (PDF page 13)" |
| D, G | counted as opened | a page whose markups were listed; a reference page a reader rendered |
| H | one citation of the whole run | "pp. 4-1 to 4-3" |

### A: the rule for "what the page map said"

For each cited page, the check keeps what the structure tools'
results said of it (`open_document`, `document_structure`, `document_roles`,
`document_page_map`, `document_inventory`, `digest_pages`): a row's kind,
title, heading, label, printed page, word count and flags; the key of a page
list it is on (`duplicate_pages`, `pages_by_kind`); an outline entry's title,
for the entry's page and the pages after it up to the next entry of the same
or a higher level (the last entry to the document's end when the page count
is known and the list was not cut short); for a duplicate and the page it
repeats, words such as "duplicate", "copy", "repeats", "same"; and, for a
page a structure result named, the document's page count ("last" for its
last page) and the printed numbers its page map gives it.

The statement is the citation's sentence part (to a sentence end, a
semicolon, a line break, or ", but", ", while", ", then"...), or its table
row, with every citation's own words blanked. A citation is filed under
`page_map` when, for its PDF reading or a printed reading:

1. every cited page (and the other pages of the same phrase, "PDF pages 6
   and 7") has a word or number of the statement among what was said of it;
2. every number or identifier in the statement ("MB-1", "3.2", "2026") is
   among what was said of those pages; and
3. at least half of the statement's other words of four letters or more
   (page vocabulary such as "page", "shows", "title" left out) are among
   those words (a plural matches its singular; a word matches another that
   starts with it when the shorter has four letters or more).

What it lets through: a statement made only of words the page map gave,
whatever it claims with them ("PDF page 22 covers the laboratory results"
inside a part titled "Laboratory results"), and, by rule 3, a short claim
whose words are half the page's title words. Any number or identifier the
page map did not give keeps it firing.

### B: a page named as unread

A cue in the statement: "not viewed / read / opened / looked at /
rendered", "did not view", "haven't read", "unread", "I can look", "want me
to open", "needs to be viewed", "should be viewed"; or right after the
citation, "was not needed / checked / reviewed / used", "should be checked".
A cue after the citation counts only with no other citation between ("PDF
page 4 is blank, and PDF page 14 was not viewed" names only page 14); a cue
before it only with no other citation before the cue. What it lets through:
"Since PDF page 14 was not viewed, the stamp on PDF page 10 is unconfirmed"
names page 10 as unread too.

### C: printed numbers

- A page number given next to a PDF page for the same place, with nothing
  between but punctuation and words such as "printed", "PDF", "in", "the",
  or in the same table row as a "PDF page" cell: only the PDF page is
  checked.
- A citation said to be printed ("printed page 28", "chapter page 12-16", a
  column headed "Printed page") that no page of the documents read is known
  to print: `unverified`, not a firing.
- A chapter-page form under a one-page lead ("p. 1-3", "page 4-2", a column
  headed "Page") in documents numbered by chapter, or whose printed numbers
  are not known (fewer than half of a document's pages print a plain
  number): read as the one printed page "1-3", never as PDF pages 1 to 3;
  `unverified` when no page is known to print it. In a plainly numbered
  document, "p. 12-16" is still read as a run.
- "pages 27 of 213 read" is a count, not a citation.

### D, G, H

- D: pages in a `document_markups` listing (the compact "p3 ..." rows or
  rows with a `page`) count as opened for a citation.
- G: pages a reference reader (`read_reference_figure`,
  `read_reference_table`, `view_worked_example_source`) rendered and read,
  with the printed label its result gives, count as opened, where the
  conversation has no document of its own or the citation's line names
  that reference (its figure or table number, example id, reference or
  file name). A render that failed does not count.
- H: printed labels joined by "to" / "through" ("pp. 4-1 to 4-3") are one
  citation of each label between.

Also: the PDF note "no document read here has that page" now counts a
document only mapped (opened but not read) as read here.

## Calls made

- The exempt citations are kept on the record: A under the existing
  `page_map` key (it already meant "where something is, read off the page
  map"), B under a new key `named_unread`. Nothing is dropped silently.
- A's "half of the other words" threshold, and the stoplist of page
  vocabulary, are mine; the identifier rule (2) carries the recall.
- G with a document of the conversation's own requires the line to name the
  reference, so a coincident page number does not count.
- C's "printed numbers not known" is "fewer than half of a document's pages
  print a plain number", so a front matter numbered 1-3 does not make a
  chapter-numbered manual "plain".
- The count rule and the label-run rule are parser repairs that fell out of
  C and H.

## Settings

None new. `GEOTECH_CITED_PAGES` unchanged (off by default).

## Tests

`test_ev18_page_check_repairs.py` (made-up documents and answers). The tests
that a citation stops firing fail on the code before this change and pass
after; the tests that a citation still fires pass before and after (they
guard recall):

- A: blank page, two logs named by their titles, a duplicate and the page
  it repeats, an outline part's pages, a chapter's first page with its
  printed number: filed under `page_map`. Still fire: a value, an
  identifier or a sample the page map never gave; a page no structure
  result named.
- B: six cue forms: filed under `named_unread`. Still fire: a plain
  citation; a citation beside another page named as unread, before or
  after it; "was not checked" said of the content.
- C: a printed number beside its PDF page (prose and table); a chapter-page
  number with unknown numbering; a number said to be printed that no page
  prints; a count. Still fire: the PDF page beside it when unopened; a
  plural run; "p. 12-16" in a plainly numbered document; the same number
  not said to be printed.
- D: a page whose markups were listed. Still fires: a page not listed.
  `write_diggs`' entries and `PagesOpened.status` are unchanged by a
  markup listing.
- G: a rendered reference page with no document of the conversation's own,
  and with one where the line names the figure. Still fire: a page nothing
  rendered; a line that does not name the reference.
- H: "pp. 4-1 to 4-3" is one citation, partly opened.
- A tool's 0-based index cited as the PDF page still fires.
- With the mode off the answer comes back unchanged and the event records
  `page_map` and `named_unread`; with `note` on only a page never opened is
  noted.

Existing tests run unchanged and pass: `funhouse_agent/tests/test_cited_pages.py`,
`packages/aec-agent-core/tests/test_dr48_photo_pages.py`,
`packages/aec-agent-core/tests/test_coverage_pages_offline.py`,
`webapp/tests/test_cited_pages_turn.py`, `webapp/tests/test_cited_documents_turn.py`,
and in the review package `tests/test_cited_pages_turn.py`,
`tests/test_dr48_photo_pages_turn.py`, `tests/test_cited_documents_turn.py`,
`tests/test_review_fingerprints.py`. No existing test was edited.

## Noted, not fixed

- E: a page cited from an earlier answer or a document of an earlier
  conversation that the record of this one does not hold.
- F: a printed page number read off a contents page ("the contents give it
  as page 4") checked as a page of the document.
- A printed label the page map reads wrong (a figure number taken for the
  page's printed number) makes a printed citation resolve to the wrong
  page; a planlens matter.
- An answer that says once, in a heading, that its page ranges come from
  the contents ("page numbers from the table of contents") is not read as
  naming those pages unread; B reads only the citation's own statement.
- A statement that mixes what the page map said with claims about other
  pages ("Appendix A (divider p. 7) holds the logs for MB-1 and MB-2")
  still fires when those identifiers are not among the cited page's facts.

## What a hosted measurement should compare

On the same saved or fresh runs (the check's event is recorded with
`GEOTECH_CITED_PAGES` off): the turns with a `never_opened` entry before and
after; each new
`page_map` row with `matched` (does the statement really only repeat the
page map?) and each `named_unread` row (does the answer really name the
page as unread?); `unverified` rows with "reads as a printed page number"
(was the page in fact one a tool read?); and that a page cited as a source
with nothing opened still fires.
