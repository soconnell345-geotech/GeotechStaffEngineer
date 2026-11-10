# The coordination tasks: a blind attempt (2026-11-10)

**What this is.** A check of the TASKS, not of Document Review. Before the
fourteen coordination tasks (BACKLOG DR-25) go to a model run, a reviewer who
had seen only the PDFs and the questions answered twelve of them, and the
answers were scored with the tasks' own checks. It answers three questions:
can each task be solved from the documents, is each question clear, and do
the checks take a right answer written in someone else's words.

**What this is not.** It is not a score for the app and must never be quoted
as one. The reviewer was a Claude Opus helper in a coding session, with
Python and PyMuPDF: it read every piece of text with its position and
compared the two issues of the set by program. The app's reviewer has the
page's tools and a budget of model calls, and the testers' model is a
different one.

## How it was run

- The six PDFs under their upload names and `questions.md` (the twelve
  questions, each naming its documents) in one folder; nothing else.
- The helper was told to stay inside that folder, to answer as it would to
  the engineer who asked, and not to report a problem it was not confident
  was real.
- Its answers are in `answers/`, one file per task, with its own notes
  (`answers/_notes.md`): how it worked each question, what it had to
  interpret, and what it saw and chose not to report.
- The two scanned-set tasks were added afterwards and were not attempted.

## Result

The reviewer found every planted problem: all 13 in the 15-sheet set, all 19
in the 40-sheet set, the three changes between the issues (and that the
third was made with no cloud and no revision entry), and the submittal's
four.

| | Checks passed |
|---|---|
| As first written | 84 of 86 |
| After the changes below | 86 of 86 |

The two misses were the checks, not the answers. The reviewer listed
`FCU-3-4` and `LP-4B` in a table headed "on a plan but not scheduled", with
rows that read "M-601 lists FCU-3-1 to FCU-3-3 only". The check wanted the
mark within 160 characters of a word such as "not" or "missing", and the row
had none: the heading carried it.

## What it changed

**In the checks.**
- A word said once in a heading counts for every line of the list or table
  under it (the core's `stated_together`).
- "Lists F1 to F3 only" says a mark is missing; "no discrepancies" and
  "nothing is missing" do not.
- Five checks that passed on a bare mention (a sheet number, a callout, a
  section number) now want a word that says it is not there beside it, so an
  answer that lists the index and calls it correct does not pass. The
  sheet-title check wants both titles.

**In the documents.** The reviewer's notes list things it saw that were not
planted. None was a mistake in the truth, but each could draw a comment from
a careful reviewer, and such a comment would then read as a false alarm.
They are gone:

| The reviewer saw | Now |
|---|---|
| Details called from places they do not suit (a storefront sill on a core wall, a parapet near the second floor). | Each detail is named for where it is called: the section's three callouts sit at the parapet, the second floor's slab edge and the base of the wall. |
| Borings drawn 4 to 7 m off the corners; the memorandum said "at the corners". | The memorandum says "near the corners". |
| The manual's contents list seven sections and bind three; steel, deck, ceilings, earthwork, mechanical and electrical have no section. | The contents list those trades' sections, say which three are bound in this volume, and put mechanical and electrical in Volume 2. |
| The panel schedule places panels in an "electrical room" no plan shows; the air handler sits in a storage room. | Panels are scheduled "in the core", where they are drawn, and the air handler is drawn there too. |
| The five-storey building has the two-storey building's footings (about half the area a rough load estimate asks for). | The five-storey set has its own, larger footing schedule. |
| The memorandum's fill removal, subgrade inspection and shoring are not on the drawings. | Two structural notes carry them. |
| A ten-day gap in the submittal's strength record. | The tests are a week apart. |

Left as they are, because a set drawn this simply has them and they are not
coordination problems: no door into the core above the first floor, no door
on the toilet, two of four elevations. A reviewer who comments on them is
not wrong.

## What it says about the tasks

- With a clean text layer and a program, these checks are mechanical: the
  reviewer paired each bubble's view number with the sheet number under it by
  position and looked the pair up. That is the case for a sheet index and a
  cross-reference check in code (BACKLOG PL-37, DR-07).
- The app's reviewer does not have that program. Whether it finds the same
  problems with the page's tools, in its budget of model calls, on a
  40-sheet set, is what the first model run measures (BACKLOG EV-40).
- On the scanned set there is no text to pair: the bubbles are read by
  looking, in a zoom. Those two tasks measure looking.

## Limits

- One attempt by one reviewer. It shows the tasks can be done and the
  questions are clear; it does not show how hard they are for the app.
- The checks read words. An answer that puts a mark beside "not" for some
  other reason passes; a full read of each run is where that shows.
- The checks do not score false alarms.
- The answers were written against the documents as they were before the
  changes in the table above; the planted problems are the same.
