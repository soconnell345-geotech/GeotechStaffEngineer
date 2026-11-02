# RI-195: the model's parameter markup is put back; a refused answer is asked for once more

MF-3 (build g on Foundry) lost a log, a model-first calculation window and a
page description to answers in which Claude wrote its own tool-call markup
inside a value. Four more page descriptions were taken with one field
swallowed into the one before it. This note gives IDs, counts and field names
only. The replay ran on the private hand-back in the gitignored raw folder.

## What changed

### 1. The engine puts the markup back (`report_ingest/engine.py`)
- `_mend_markup` takes the model's `<parameter name="...">` markup out of
  every object at any depth. It returns a copy and leaves the answer as it
  came unchanged. It handles two shapes:
  - A field swallowed into the one before it:
    `"pit": "null</pit>\n<parameter name=\"remarks\">..."` becomes `pit`
    None and `remarks` the words. Each piece is read as `null`, as JSON (a
    list or an object), or as its words.
  - A list spilled into numbered keys: `"inputs": "<parameter name=\"0\">{...}"`
    plus keys "1", "2", ... becomes the list, in order. A list spilled inside
    its own value is read the same way.
- Faults stop the repair, and the answer is then tried as it came:
  - a field named by the markup that the object already holds, not blank;
  - items numbered with a gap;
  - numbered keys before any list's markup;
  - two lists sharing the numbered keys. The provider keeps one value per
    number, so one list's items may stand in the other's place.
- `_validate` tries the mended answer first, even when the answer as it came
  would validate, because that answer can hold one field's words in another.
  Then it tries the answer as it came. Every case gets a note in the reply's
  `notes`, and the raw arguments are kept (`answer_args` on the cost row).
  A taken answer that still holds markup inside a list of words gets a note
  naming the value.
- The repair applies on `ChatModelEngine`, the Foundry route. `ClaudeEngine`
  and `PrompterEngine` do not validate through `_validate` and are unchanged,
  apart from counting the re-ask.

### 2. A refused answer is asked for once more
- The engine has a shared helper, `ask_again(messages, reply)`. It follows
  the calc reader's pattern: the reply's own turn, then a user turn
  `ASK_AGAIN` that gives the engine's reason. `refusal(reply)` reads that
  reason from the notes.
- Each engine marks a request that ends with this turn: the cost row says
  `"asked_again": true`, and `CostMeter.to_dict()` gives `asked_again` (a
  count) only when there was one.
- Three callers ask once more, only when the reply was not cut off:
  - Model first's `_Caller.read` (`model_first_read.py`). The re-ask row
    in `calls` says `asked_again`. A group is skipped only if the second
    reply also gives no answer ("..., and none when asked again").
  - Today's log reader, on its first call (`log_reader.py`), when its call
    budget allows. It warns "the first answer was not taken (...); the
    reader was asked for it again" and counts the call in `model_calls`.
  - `question_list._ask`. Each call's row in `run.calls` now keeps the
    engine's `notes`, `raw_answer` (the arguments where the engine kept
    them), or `text` (the prose of a reply with no answer). The re-ask is
    its own row with `asked_again`.
- No prompt version was bumped. The main instruction text is unchanged; the
  re-ask turn goes only on the second call, and its row is marked.

## Replay on the record ($0, no model calls)

Every answer or saved description in the MF-3 hand-back that shows the markup
went through the new `_validate`. The script and its output are in the
session scratchpad, not the repo.

| Case | Before | After |
|---|---|---|
| R21 arm A, call 257 (describer) | refused, `kind` missing (swallowed into `issuer_date`) | taken; `issuer_date` split into `issuer_date`, `kind` |
| R21 arm A, call 889 (model-first calc window) | refused: `inputs` and `results` not lists | still refused: two lists share the numbered keys, and one `inputs` item was overwritten. Model first now asks again |
| R21 arm B, call 222 (today's log reader) | refused: `pit` not an object; `remarks` swallowed | taken; `pit` None, `remarks` put back |
| R21 arm A, one description | taken, `language` swallowed into `issuer_date` | `language` put back |
| R30 arm A, description 1 | taken, `continues` swallowed into `exploration`; exploration unreadable | `exploration` reads as one hole id; `continues` put back |
| R30 arm A, description 2 | taken, `language` swallowed into `issuer_date` | `language` put back |
| R30 arm A, description 3 | taken, `exploration` swallowed into `identifiers` | `exploration` put back (blank, as the model gave it) |

6 of 7 repair cleanly, with no markup left in the taken answer. The seventh,
call 889, cannot be repaired without loss and goes to the re-ask. The four
descriptions were rebuilt from their saved files, because a taken answer's raw
arguments are not kept. One description's exploration changes from none to a
hole id, so its page can now be grouped with that hole.

## Tests (exit codes)

Free memory was checked before each run (2.9 to 5.0 GB). One run was held
until it was back over 3 GB.

- `report_ingest/tests/test_ri195.py` (new, 21 tests, made-up values): exit 0
- `report_ingest/tests` in pieces, all 71 files: exit 0 after the updates
  below. One `test_library.py` failure came from running with the tests
  folder as cwd. It passes from the repo root (exit 0).
- `module_work/report_ingest_harness/tests`: 201 passed, 33 skipped, exit 0
- `report_ingest/tests/test_graph.py` and `test_graph_model_first.py` (the
  off-path snapshot is in the second): exit 0
- `webapp/tests/test_public_text.py`: exit 0

Existing tests updated for the re-ask. Each script that gives no answer now
has a second no-answer turn, and the call counts went up by one:
- `test_ri192.py`: a refused-list example now uses plain words, since the
  markup shape is put back.
- `test_model_first_read.py`, `test_graph_model_first.py`, `test_ri186.py`,
  `test_model_first_scoring.py`, `test_log_reader.py`.

## Noted, not fixed
- A JSON-encoded empty string, `"\"\""`, was seen in one log answer's
  `coordinate_system`. It changes no saved value that matters.
- `cluster_scoring` totals do not sum `asked_again`. The per-call rows and
  the meter's own count have it.
- The describer (`range_labels.describe_pages`) does not ask again. Its one
  refusal in the replay (call 257) is now repaired by the engine.
