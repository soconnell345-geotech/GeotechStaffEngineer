# VIS-32 and TA-43: check notes

Branch `claude/vis-32-ta-43`, cut from `claude/vis-33-34-tiles` (VIS-33, VIS-34). Two commits: VIS-32 (a repair, no switch) and TA-43 (a count always; a ceiling behind `GEOTECH_TURN_MAX_IMAGE_CALLS`, off by default).

## VIS-32: a page reading that the tiles contradict says so (a repair)

**Asked.** A review run (brief 8) asked about one door tag by name. The whole-page answer placed that tag where the tiles read the next door's tag, and the item was dropped between the two readings. `_merge_found` named things found only in the tiles, never a direct disagreement. Done when such a pair gets a note to zoom before relying on either reading, tested with made-up answers.

**Found on the parent.** Two shapes, depending on the tags:
- Tags with letters (D-509, D-508) carry codes, so the merge keeps two rows: one seen only on the page, one only in the tiles. The note named the tile row as found only in the tiles, and said nothing of the page row.
- Plain numbers (509, 508) carry no code, so the merge joined the two readings into one row, labelled with the tile's reading, and the note said the page answer and the tiles agree. This was the item's case.

**Built.** In `packages/aec-agent-core/aec_agent_core/vision_tools.py`:
- `_tags(label)` returns the words with a digit in them, two characters or more, in upper case.
- `_same_tag(a, b)` says whether two tags may be one tag written two ways, such as D-509 and 509.
- `_one_series(a, b)` says whether two tags are different tags of one series.
- `_place_disagreements(entries, whole, near)` finds the pairs.
- `_disagreement_line(pairs, whole)` writes the line.

`_merge_found` calls them after it builds its rows. It adds the line at the end of `found_note`, after the found-only-in-tiles text. If there is no such text, the line follows "'found' lists every located thing …", in place of the "agree" sentence, which would be false. The rows, the other keys and every prompt stay as they were. There is no new model call.

**When a pair is named (calls made, each conservative).**
- The page answer's tag is read by no tile anywhere, and the tile's tag is named nowhere in the page answer. If a tile reads the page's tag elsewhere, the page answer only placed it loosely, so nothing is said.
- The two boxes are within the merge's own reach. That is the distance at which the merge already calls two readings one thing. I added no new tolerance.
- The two tags are of one series: the same letters and the same count of digits, and not one written two ways. So 508 and 509 make a pair, as do D-508 and D-509. A-501 and 508 do not. Nor do D-509 and 509, which agree.
- Each page tag is paired with its nearest tile reading only. Each pair is named once, and at most six are named.
- The line says the tiles are the closer look because a tile's lettering is larger. That is the same reason the found-only-in-tiles note gives.

**Before and after (made up).** The page answer has "the requested door tag 509 [400, 400, 420, 415]". Tile r1c1 has "Door 508 [800, 800, 840, 830]", which is the same place.
- Before: one row, "Door 508", seen in the page and in r1c1. The note said "… (the page answer and the tiles agree on what is there)."
- After: the same row. The note reads: "'found' lists every located thing with its page box and where it was seen. The whole-page answer and the tiles DISAGREE at 1 place: 'the requested door tag 509' at [400.0, 400.0, 420.0, 415.0] in the page answer, where tile r1c1 read 'Door 508' at [400.0, 400.0, 420.0, 415.0]. The tiles are the closer look (the lettering larger): zoom on each such place before relying on either reading."

**Tests.** `packages/aec-agent-core/tests/test_vis32_page_tiles_disagree.py` has 17 tests.
- Six as-before cases, each compared as whole JSON with the result written out in the test: the same tag; a tile-only thing of another series; different series at one place; the page's tag read by a tile elsewhere; one series far apart; one tag written two ways. The no-tiles case is a seventh. All seven pass on the parent unchanged (run before the change: 7 passed, 10 failed). That run is the proof that a read with no pair is returned byte for byte.
- These fail on the parent and pass here:
  - the item's case (plain numbers, one merged row; only `found_note` moves);
  - the letter-code case (two rows; the line follows the found-only-in-tiles text);
  - an image read in tiles;
  - only the nearest reading is named;
  - a pair is named once;
  - `_tags`, five cases.
- Also run, all green: `funhouse_agent/tests/test_vision_brief4.py`, `test_vision_images.py`, and the core's `test_vis30`, `test_vis33` and `test_vis34` files.

**Noted, not fixed.**
- The merge still joins nearby readings that carry no code into one row, labelled with the smallest view's reading. In the item's case the row reads "Door 508" while the page answer said 509. The new line names the pair, but the row itself is unchanged, as the ruling asked (only a line is added). Splitting such rows would change saved results for every code-less label. It is a follow-up only if a run shows the agent working from the row and not the note.
- Tags of different series, such as a callout A-501 placed where a tile reads door 508, and a tag inside another tag (509 and 509A), are not named. This is conservative by design.

## TA-43: a count and a ceiling on a turn's image calls

**Asked.** A page look is up to 17 image calls, so a 50-sheet set read sheet by sheet is about 850. There was no count, no ceiling and no word to the tester. The design is in `module_work/ta37/CHECK.md`, "Left: a count and a ceiling of image calls per turn".

**Built (always on: a count, at no cost, that never fails a turn).**
- `deep/keep_work.py`, `TurnScope`: three new fields.
  - `image_calls`: the images sent this turn, every agent and helper together, under the scope's lock.
  - `image_refused`: the images refused at the ceiling.
  - `image_limit`: the ceiling, read from the setting when the turn's scope is made.
- `deep/keep_work.py`, `take_image_call()`: counts on the running turn and does nothing outside one. It also counts on the running tool call's `counting_look()`, which the tile threads share through their copied context.
- `deep/vision_engine.py`, `LangChainVisionEngine.analyze_image`: calls `_take_image_call()` once per image, before the slot wait and the asks. Only the ceiling's refusal raises; any other error in counting is swallowed.
- `activity_log.py`, `ActivityLogger`: keeps the turn's scope from its model starts (`_see_turn_scope`). `turn_end` then writes `image_calls`, plus `image_limit` and `image_calls_refused` when a ceiling was set or a look was refused (`keep_work.image_call_facts`).

**Built (with `GEOTECH_TURN_MAX_IMAGE_CALLS` = N > 0).**
- `deep/limits.py`: `TURN_MAX_IMAGE_CALLS_ENV`, `turn_max_image_calls()` (unset, 0, off or a bad value give 0, which means no ceiling) and `IMAGE_WARN_SHARE` (0.8). They sit beside the helper's model-call budget, the other turn ceiling there.
- Past N: `TurnScope.take_image_call` raises `ImageBudgetSpent` and the image is not sent. `describe_error` returns its own words: "this turn reached its ceiling of N image calls, so this image was NOT read. Finish with what you have: answer from what was already read, and say what was left unread."
  - The vision tools already report a failed look through `describe_error`. A look refused whole comes back as `{"error": …}`. A tiled look cut part way returns what it read, and the parts not read say why.
  - The turn goes on.
- `deep/vision_kit.py`, `_dispatch`: the old body is now `_dispatch_tool`, wrapped in `counting_look()`. A tool call that made image calls starts its result with `image_calls`: "this look made k image calls; this turn has made u of its ceiling of N". It adds "; r more were refused at the ceiling" when some were refused, and a warning once u reaches 80 percent of N (`keep_work.with_image_calls`). A result that is not a JSON object gets the same text as its first line.
- `aec_document_review/core.py`, `stream_turn`: after each tool result that moved the count, the turn yields a status line, "image calls this turn: u of N". It goes through the existing `tool_call` items, which the page writes into the running turn's status.
- `aec_document_review/turn_jobs.py`: a turn whose ceiling refused a look ends its answer with one line: "_This turn reached its ceiling of N image calls: r looks after that were not made, so the answer rests on what was read before. Ask to continue for more._" A turn that met no refusal, or that ended in an error, gets no line.

**Where the code moved from the design (intent kept).**
- The design counted in `invoke_side_call`. I count in `analyze_image` instead, the one caller of `invoke_side_call`. `analyze_image` calls it a second time when a model refuses the `detail` field, and the retry ladders inside it ask the same image again. Counting there would count one image two or more times, and the count would vary with how busy the server was.
- The design said `turn_end` should carry the count from the activity log. `turn_end` is written after the turn's scope has closed, so the logger keeps a reference to the scope from the turn's model starts. This also covers the review suite's runner, which writes its own `turn_end`.
- The design's 80 percent warning is part of each look's result note, not a separate exception. "A page look reports its own cost" is the same note.
- The tester is told once at the end by the line under the answer. A status line alone sits in the collapsed "Working…" box.

**Calls made.**
- The warning share is 0.8, as in the design.
- Refused looks are not added to `image_calls`; they are counted apart.
- The line to the tester is in italics under the answer. It is part of the saved answer and the next turn's history, so the agent knows the earlier turn was capped.

**Unset: nothing moves (proved).**
- No refusal, and no new key or line in any tool result. `with_image_calls` returns the very same string object.
- The model is handed what it is handed today: no tool description, schema or prompt changed. Both fingerprint tests still pass: `test_review_fingerprints.py` and `webapp/tests/test_page_fingerprints.py`.
- No status line, no line under the answer.
- The only change is the `image_calls` field on `turn_end`.

**Settings.** `GEOTECH_TURN_MAX_IMAGE_CALLS`: default unset, meaning no ceiling. A positive integer turns the ceiling on. `0`, `off` or a value that is not a number means none.

**Tests.** Each new test fails on the parent: both files import names that do not exist there (`limits.TURN_MAX_IMAGE_CALLS_ENV`, `keep_work.ImageBudgetSpent`).
- `packages/aec-agent-core/tests/test_ta43_image_ceiling.py`, 19 tests:
  - the setting is off unless set (7 values);
  - unset, 20 looks are all made and counted;
  - unset, a page read under a turn equals the same read outside any turn, character for character;
  - outside a turn, nothing is counted or refused;
  - past the ceiling the image is not sent, and the error text;
  - a refused page read is a plain `{"error": …}`, and the first read's note;
  - a 2x2 tiled read cut at 3 of 5 calls says what was not read;
  - the note and the 80 percent warning, and a result that is not JSON;
  - the status line moves only with the count;
  - the line to the tester appears only after a refusal;
  - the record, with no ceiling, with a ceiling and refusals, and with no scope.
- `packages/aec-document-review/tests/test_ta43_image_ceiling_page.py`, 3 tests. Each runs through the page's real compiled agent and turn worker with a scripted model whose tool looks through the core's vision engine:
  - unset: the turn is as before, and `turn_end` has `image_calls` = 3;
  - ceiling 2: the status lines "1 of 2" and "2 of 2", the third look refused with the tool error, the line under the answer, and the record;
  - ceiling 10: no line at the end.

**Noted, not fixed.**
- A helper's tool results do not reach the main stream, so the status line moves when the helper's `task` returns, not at each of its looks.
- `annotate_document`'s checking host is called outside `vision_kit._dispatch`. Its image calls are counted and refused like any others, but its result carries no `image_calls` note.
- Vision probes (`vision_probe`) call the model directly and are not counted. They run once per model per process.
- The geotech page also opens a turn scope. Its `turn_end` records gain `image_calls`, and the ceiling would apply there if a deployment set it. Its page shows no status line or line at the end, because the app's `webapp/core.py` is not changed.
- The item's "with DR-08's cost estimate" (a price per image call) is not built.

## For the lead

**Settings line for `tinyapps/wrapper_repo/.env.example`**, after `GEOTECH_BUSY_TURN_MAX_S`:

```
# Image calls one turn may make (each image sent to the vision model; a
# page read in tiles is up to 17), all its helpers together. Past it a look
# is refused and the turn finishes with what it read (0 = no ceiling).
# GEOTECH_TURN_MAX_IMAGE_CALLS=0
```

In `tinyapps/TINYAPPS.md`'s settings list: `GEOTECH_TURN_MAX_IMAGE_CALLS` (no ceiling; a number sets one).

**What a measurement on the hosted platform should compare.**
- VIS-32: re-run brief 8's coordination task on the 40-sheet set. On tiled reads, count the `found_note` lines that say DISAGREE, and check each one against the sheet. Look for any on a read where the page answer and the tiles agree, which would be a false line. Also check whether the agent zooms on a named place and whether door 509 survives into the comment log.
- TA-43, with the setting unset: read `turn_end.image_calls` across arms. That is the first measure of how many image calls a review turn takes.
- TA-43, with a ceiling under that number: check that the agent answers from what it read after its first refusal and does not try looks over and over. Check that the answer carries the line.
