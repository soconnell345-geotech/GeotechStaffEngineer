# B15: the store of shown uploads (`GEOTECH_VISION_SHOW_UPLOADS`, DR-38)

Branch `claude/b15-shown-uploads`, from `claude/b13-integration` at 9367934 (1.3.0). Both limits are repaired. Three source files, 121 lines added; `webapp/` and `funhouse_agent/` untouched.

## Confirmed on the head, before any change

- **A second copy.** `aec_agent_core/inline_store.py`, `put`: every call made a new id and stored the bytes again. `deep/upload_view.py`, `show_upload` calls it once per look.
- **One store for the process.** `inline_store._STORE`, 96 images, oldest out first, shared by every conversation. `deep/inline_images.py`, `newest_image_ids` skips an id that `get` no longer finds, and `upload_view.left_out` skips it too (`got is None`), so the model call carried no image and no word; the tool result in the history still said "shown to you with your next step".

## What changed (all three are repairs; no setting)

- `inline_store.py`: inside a turn (`keep_work.turn_scope`) the images are held on the turn's scope (`scope.inline_images`, read through the turn's context variable), not in the process store. `put` returns the id an image already has where the same bytes with the same description are stored, and makes it the newest. An image pushed out leaves its description behind (new `pushed_out(image_id)`). `get` reads the turn's images, then the process store.
- `deep/keep_work.py`: `turn_scope` drops the turn's images when the turn ends; `image_call_facts` adds `uploads_not_shown` (a list of file names) only when the turn lost one.
- `deep/upload_view.py`: new `tell_not_shown(request)`, called first in `UploadImageMiddleware._with_images`. An upload's result whose image was pushed out before it was shown gets `not_shown` in place of its `note`, in the request, at every later call of the turn. An image is marked `shown` when a model call carries it, so one that falls out afterwards is not called unseen.

## The bound that remains, and why it is enough

- **Per turn: 96 images** (`MAX_IMAGES`, unchanged), each at most 4 MB for an upload. A model call carries the newest two, so only a turn that stores 97 or more between two model calls loses one, and it is now told which.
- **Per conversation: nothing between turns.** `newest_image_ids` reads only the results since the user's last message, so an image is never wanted after its turn; the turn's images go when it ends.
- **Per process: the turns running at that moment, times 96.** No count across conversations is kept, because any such count is what let one tester push out another. The process-wide store of 96 remains only for calls outside a turn (a library call, `inline_images.probe`); Document Review's page runs every turn in a scope (`core._turn_scope`).

## Before and after (made up)

Two testers on one host. A looks at `site_photo.png`; before A's next model call B's turn stores 100 page views.
- Before: A's model call carries no image; A's result still says it is shown.
- After: A's call carries `site_photo.png`; B's turn cannot read A's image id.

A alone looks at 97 uploads in one step: the first now reads `{"not_shown": "photo_000.png was NOT shown to you: this turn stored more than 96 images after it before your next step, ... say nothing about it from sight. To see it, call analyze_image on it again.", ...}` and the turn's record has `"uploads_not_shown": ["photo_000.png"]`.

## Tests

New file `packages/aec-agent-core/tests/test_b15_shown_uploads_store.py`, five tests. Each fails on 9367934 (run with the three source files stashed: 5 failed) and passes on the branch:
- `test_a_second_look_at_one_upload_stores_no_second_copy`
- `test_the_same_bytes_under_another_description_are_another_image`
- `test_another_conversation_s_images_cannot_push_an_upload_out`
- `test_a_turn_s_images_are_its_own_and_go_with_it`
- `test_an_upload_pushed_out_before_it_is_shown_is_said_not_to_be_shown`

Runs on the branch (each judged by pytest's exit code, all 0):
- venv (a), from the root, the new file with the core's five files that name shown uploads or the image ceiling (`test_dr48_photo_pages`, `test_ev17_run_report`, `test_ev49_image_ceiling_between_arms`, `test_rvb_image_read_once`, `test_ta43_image_ceiling`): 73 passed (the new file alone: 5 passed).
- venv (a), from `packages/aec-document-review`: `test_dr38_review_fixes`, `test_dr38_show_uploads`, `test_ta43_image_ceiling_page`, `test_review_fingerprints`, `test_boundary`, `test_review_show_me_offline`: 38 passed. The ten files there that name the inline store or `show_me`: 77 passed.
- venv (b), from the root: `funhouse_agent/deep/tests/test_review_harness_offline.py`, `test_review_minimal_offline.py`, `test_review_switches_offline.py`, `funhouse_agent/tests/test_vision_images.py`, `test_vision_turned.py`, `webapp/tests/test_page_fingerprints.py` and the new file: 168 passed.
- Guards: `webapp/tests/test_public_text.py`, `test_review_freeze.py`: 13 passed.
- No existing test was edited. The full suites were not run (the lead's `ci/` push).

## Fingerprints
Not moved: Document Review's and the geotech page's fingerprint tests pass unchanged. No prompt, tool description, schema or setting changed.

## Calls made

- **The turn, not the conversation, is the key.** A shown image is only wanted inside its turn, the turn's scope already reaches every tool and model call, and it frees the bytes by itself. A key by conversation would need its own clean-up.
- **The store changed for every user of it** (`show_me` looks, the `minimal` build, a contact sheet), not only for uploads: one shared store with a second one beside it for uploads would have left the other looks open to the same loss. Their ids, results and notes are unchanged.
- **One copy means the same bytes and the same description.** Two pages that render to the same bytes keep two ids, since each is labelled with its own page.
- **"Not shown" is written into the request, not the saved history**, as the images themselves are. The saved result keeps its first wording; the record's `uploads_not_shown` is the lasting statement.

## Release note

With `GEOTECH_VISION_SHOW_UPLOADS` on, a tester's shown upload can no longer be lost to another tester's images, and looking at the same upload twice keeps one copy. If one turn looks at more than 96 images between two model steps, the model is told which upload it did not see, and the turn's record lists it under `uploads_not_shown`. With the setting unset nothing a tester sees changes.

## For the lead

- Not run on a model or a live host. The measurement of the setting is still brief 9 (EV-45).
- `uploads_not_shown` is a new key of the turn's record, present only when it happens; `run_report` does not print it yet.
- The release note's two lines under "Known limits" and DR-38's "Noted, not fixed" can be struck when this merges.

## Noted, not fixed
- A second look at one upload still counts a second image call on the turn's ceiling.
- Only uploads are told "not shown"; a `show_me` page view pushed out by its own turn's 96 later images is still dropped without a word.
- A turn that runs with no scope (a host whose core cannot import `keep_work`) still uses the process-wide store.
