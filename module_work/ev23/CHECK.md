# EV-23: the cited-documents check repaired (check notes)

## What the item asked

The turn-end check `aec_agent_core/cited_documents.py` lists a document an
answer (or a deliverable) cites that nothing in the conversation looked
up. An offline measurement on saved runs sorted its firings into right
ones and wrong ones, and the wrong ones into kinds. The item: stop the
wrong kinds from firing, for every model and every text, without losing
the right ones. They are repairs of the check itself, so no new setting.
With `GEOTECH_CITED_DOCUMENTS` unset (the default), what the model and the
user are handed must not change, on either page: only the turn's recorded
event may. The kinds, in the order repaired:

- **J**, a defect: a tool result was searched as its raw JSON text, so an
  escaped newline (`\n`) stood against the next word and a designation
  after it went unseen.
- **P**: a pointer ("verify it with X"), a denial ("I have not relied on
  X") or a condition ("if the specification calls for X, ...") was read
  as a citation.
- **K**: a surname with a one-letter particle ("O'Hara") was read without
  it in the answer, but whole in the sources.
- **L**: brackets holding a pair of numbers after a capitalised word
  ("Corner (1710, 1342)") were read as an author and a year.
- **M**: a calculation package's printed method basis, read back from the
  file the package wrote, was no source.
- **I**: a page read only as an image: noted, not fixed.

Measured offline on saved runs; the counts are in the lead's private
notes.

## What was built, and where

All of it is in `packages/aec-agent-core/aec_agent_core/cited_documents.py`.

- **J.** `_result_text` (with `_decoded`) renders a JSON result as the text
  it holds: strings decoded, one record's fields on one line, so an author
  and a year in one record still read as a pair. `_note_call` reads every
  result this way. A standard published in parts (ISO, BS, EN; `_PART`) is
  now the family of its parts, as EN 1997 already was (`covered`).
- **P.** `mention_kind` sorts each mention of a document into a citation or
  one of three kinds of mention (`POINTER`, `DENIAL`, `CONDITION`).
  `cited_and_mentioned` reads every mention, so a document named once in
  a pointer and once in a citation is cited. `check_text` keeps the
  mentions in the record under the new key `mentioned_not_cited` (rows
  `{"mentioned", "as", "document"?, "looked_up"}`) and never flags them;
  `n_cited` counts citations only. The key is new, so no reader expects
  it. `TurnCitationCheck` and `check_deliverable` record a turn that has
  only mentions.
- **K and L.** `find_citations(..., refined=True)` is the check's own
  reading: a particle surname read whole (`_AUTHORS_WHOLE`), and a list of
  numbers that are not all years taken as no citation (`_number_pair`).
  Without `refined` it reads as before: the judgment-call check
  (`funhouse_agent/judgment_check.py`, `has_source`) asks it whether a
  work is cited, and its note to the model must not move. `_name` treats
  a curly apostrophe as a straight one.
- **M.** `source_kind`: every `calc_package` method but `html_to_pdf`
  (`MODEL_TEXT_METHODS`) is a calculation. A file such a call wrote
  (`_LookupState.app_written`), read back, is a calculation as well. A
  later write to the same file by any other tool takes that away. For both,
  nothing the call's or the reader's arguments cite counts:
  `_without_given` removes each given string across line breaks, and
  `Marks.add_text(not_from=...)` leaves out every designation and author
  the arguments cite, however short.

## What each rule lets through

- **J** lets nothing new through: a result is read as what it says. The
  parts rule credits "ISO 77777" when only part 1 was read, and part 2
  when the whole was read; it never credits part 2 from part 1.
- **P, value test.** A clause that states a value keeps every mention in
  it a citation, whatever the verb: a sign (`=`, `≈`, `<`, `±`...), a
  number with a unit, or a decimal. Designations and years do not count as
  values. "Verify the 3.0 m spacing with UFC 3-999-01" still fires.
- **P, pointer.** These verbs, with nothing like "per", "from", "according
  to", "based on", "as", "which" or "because" between the verb and the
  mention, make a pointer:
  - check, cross-check, double-check, confirm, verify, compare, validate,
    consult or use;
  - in the plain form at a clause's start, after a comma or a bracket, or
    after a modal or "to";
  - "be" + the past form after a modal ("should be compared against");
  - "-ing" after "recommend", "suggest", "consider", "worth"...
  It therefore lets through an instruction that is really the basis, when
  no value stands in its clause: "Use the GEC-15 method for the drains."
  is a mention, not a citation, and is not flagged. Such a row stays in
  the record with `looked_up: false`, where a measurement can read it.
  Past and first-person statements stay citations ("were checked against
  X and agree", "We confirm ... against X"), and so do "X gives ...",
  "per X, ...", "(see X)", "X, which ...", and "as X recommends".
- **P, denial.** A negation, then within two words cite, use, rely on,
  apply, confirm, find, reference, include or follow, then the mention
  within three words. Or the mention, then within three words "is", "was",
  "were" or "has been", then "not" and such a verb. "I did not use the
  drainage chart in GEC-15" (four words between) stays a citation.
- **P, condition.** "If", "unless" or "whether" before the mention, with no
  comma or "then" between them. So "If the drains follow GEC-15 they
  work" is a condition and is not flagged; "If the yard drains east,
  GEC-15 sets ..." is a citation.
- **P, clauses** run between line ends, `;`, `!`, `?`, a colon followed by a
  space, and a full stop followed by a capital. The window is capped at
  600 characters on each side.
- **K** reads one capital and an apostrophe before a surname ("O'Hara",
  "D'Amato"). Particles of two words ("van der Berg") are as before.
- **L** drops brackets that hold only numbers when at least one is not a
  year from 1600 to 2099. So a Chicago-style "Lind (2004, 12)", where 12
  is a page with no "p.", is no longer read as a citation and is not
  checked. "(1850, 1956)" after a name is still read as two works, even
  when it is a pair of coordinates.
- **M** credits what a package printed from the app's template.
  `render_figures` and `render_tables` count as packages: they run a
  canned package's analysis, and any text of theirs taken from the
  arguments is removed as given. A new `calc_package` method that prints
  the model's own text would count as a calculation until it is added to
  `MODEL_TEXT_METHODS`.

## Calls made

- **What a citation is**: P keeps a value test, not a verb list alone. A
  citation is the source of a value, and a pointer seldom carries one.
  Where the two meet, recall wins.
- **"X, which ..."** is a citation even after a pointer verb. What follows
  says what the document holds.
- **Every mention is read**, not only the first. A pointer first and a
  citation later fires.
- **The parts rule** follows EN 1997, which the check already treated as a
  family.
- **calc_package**: every method but `html_to_pdf` is a calculation (see
  M). `html_to_pdf` renders the model's own HTML, so it stays the app's
  writer and no source.
- **The judgment-call check keeps the plain reading.** Moving it to the
  refined reading would change its note to the model, which this item may
  not do.

## Settings

None added. `GEOTECH_CITED_DOCUMENTS` keeps its three modes:
- off (the default): the event only;
- `note`: a note under the answer;
- `retry`: one retry, then the note.

A mention gets no note and no retry in any mode.

## Tests

`packages/aec-agent-core/tests/test_ev23_document_check_repairs.py` holds
56 cases with made-up documents, names and answers. On the starting code
32 fail: every case that must stop firing, plus the record key. The other
24 pass there too; they are the cases that must still fire, and they guard
recall. Two tests (`test_what_the_mention_rules_let_through` and
`test_what_the_number_rule_lets_through`) pin the limits listed under
"What each rule lets through".

- **J**:
  - a designation after an escaped newline is looked up, in a document's
    text and in a references result alike;
  - a standard no result names still fires;
  - an escaped letter ("Kovač") reads as the letter;
  - one record's author and year still pair;
  - the parts rule, as a unit test and through a read.
- **P**:
  - eleven mentions (pointers, denials, a condition) are kept in the record
    and not flagged;
  - fourteen citations still fire, among them "Per X, the spacing is 3.0
    m", "X gives ...", "Use 0.29 from X", "as X recommends", "X, which
    ...", a condition whose comma comes before the mention, and an author
    and year with a value;
  - a pointer and a citation of one document fire, in either order;
  - a pointer to a document that was read says `looked_up: true`.
- **Switch**:
  - unset: `feedback` is None and `finish` returns the answer unchanged,
    for a pointer-only answer and a citing one; the record keeps the
    mention and the citation;
  - `note` and `retry`: a mention gets no note, while a citation still
    gets its note or retry;
  - a writer's mention is recorded and adds nothing.
- **K and L**:
  - a particle surname is read whole, with either apostrophe and in
    brackets, and an unnamed one still fires;
  - a pair of numbers is no citation;
  - a second year or a page still fires;
  - the plain reading is unchanged, pinned for the judgment-call check.
- **M**:
  - a package's basis read back is credited;
  - the model's own `html_to_pdf` PDF is not;
  - nothing the model gave the package (a wrapped note, a short "GEC-15")
    is credited;
  - a file the model wrote over the package's is no source;
  - the package's own result is a calculation.

Also run, all green through the runner:
- `funhouse_agent/tests/test_cited_documents.py`
- `test_ref114_table_reads.py`
- `test_core_suite_and_records.py`
- `test_core_imports.py`
- `webapp/tests/test_cited_documents_turn.py`
- `test_judgment_check_turn.py`
- `test_core_move_11c.py`
- `funhouse_agent/deep/tests/test_judgment_calls.py`
- `funhouse_agent/deep/tests/test_review_switches_offline.py`
- Document Review's `test_cited_documents_turn.py` and
  `test_review_switches_offline.py`
- both fingerprint tests: `webapp/tests/test_page_fingerprints.py` and
  Document Review's `test_review_fingerprints.py`. Neither moved.

## Noted, not fixed

- **I.** A page read only as an image leaves no designation in the record's
  text, so a citation of what the page printed still fires. It would need
  the vision read's own text in the record.
- **An app-written file reaches the PDF unread.** The calculation HTML from
  `render_tables`, pasted into the model's report and printed by
  `html_to_pdf`, is credited only when that HTML file, or the result of
  `render_tables`, is read. The final PDF alone stays no source.
- **The old calculation path.** For calculation tools other than
  `calc_package`, `_without_echo` still keeps a given string under eight
  characters. A short designation the model passed and the tool echoed
  still counts as the calculation's.
- **Denials inside a source.** A source's own words such as "X was not
  used" still count as naming X.
- **The judgment-call check** could adopt the refined reading (K, L). It
  would change that check's note, so it is left for its own item.

## What a hosted measurement should compare

Run with `GEOTECH_CITED_DOCUMENTS=note`, on the same tasks before and after
this change, and compare per turn:
- `not_looked_up` rows: these should fall;
- `mentioned_not_cited` rows with `looked_up: false`: read each one by
  hand; each is either a fair pointer or a lost citation, and this is
  where recall would leak;
- whether the note appears under answers that only point to a document.
  It should not.

Under `retry`, also compare how many retries happen and what the retried
answer cites.
