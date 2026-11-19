**Verdict: MERGE AS IS** (reviewed head a557449, base 9367934). No finding blocks; no fix was needed, so this branch adds only this file.

Every row below comes from a run on made-up images, on the base (a second worktree at 9367934) and on the branch, with the same scripts.

## Findings

| # | What | Made-up input | Before (9367934) | After (a557449) | Status |
|---|---|---|---|---|---|
| 1 | A second look stores no second copy | one turn looks at `kerb.png` twice | 2 ids, 2 copies; the next call carries the same picture twice | 1 id, 1 copy; the next call carries it once | repaired |
| 2 | Only the same bytes with the same description are one image | `kerb.png` at 64x48 and at 128x96; the same bytes as `kerb.png` and `kerb_copy.png` | distinct ids | distinct ids | unchanged, right |
| 3 | Another conversation cannot push an image out | A looks at `gate.png`; B stores 300 page views; A's next model call | A's call carries 0 images, A's result still says "shown" | A's call carries 1; A holds 1, B holds 96 | repaired |
| 4 | A turn over its own bound is told | one turn looks at 100 uploads in one step | `p000`..`p003` dropped without a word | their four results read `{"not_shown": "p000.png was NOT shown to you: ... It was never in front of you: say nothing about it from sight. ..."}`, the old `note` is gone, the same at a later call of the turn; record `uploads_not_shown: [p000..p003]` | repaired |
| 5 | No image crosses conversations (question 2) | A and B upload identical bytes under one name; two threads, 400 looks each, a model call after each look | every id readable by every conversation (one store) | ids differ; A cannot read B's id nor B A's, nor can a call outside a turn; 0 images lost or misrouted, 0 shared ids, each scope holds only its own; image calls 400 and 400; A's "shown" mark does not reach B's copy | holds |
| 6 | Nothing the head carried is lost (question 6) | 31 whole turns with the stand-in model, see below | images per model call recorded | identical in every turn but one: an upload looked at twice with `show_me` is carried once, not as two copies of the same picture (row 1) | holds |
| 7 | A host with no scope | two callers with no turn scope look at the same file; then 120 other images are stored | 2 ids; the lost image is dropped without a word | both callers get ONE id from the process store; the lost image is now said not shown | noted |
| 8 | The saved history keeps the first wording | row 4, then a later turn of the same conversation | the result of `p000` says "shown to you with your next step" | the same: `not_shown` is in the request of that turn only (the record's `uploads_not_shown` is the lasting statement) | noted, the builder says so too |

Row 7, in full: with no scope every conversation still shares the 96-image process store, so another conversation can still push an image out there (now told, for an upload). Because identical bytes with an identical description now share one id in that store, the "shown" mark is shared too: if A's call carried the image and B's copy is then pushed out before B's call, B is not told (B gets the head's silent drop, nothing worse). Document Review's page never runs this way: `core.stream_turn` opens a scope for every turn, and the page has no other entry point.

## Question 6: whole turns, head against branch

`build_review_page_agent` with a scripted model that records, at every model call, how many images the request carries. Each line ran on both trees; "same" means the per-call counts are equal.

- Default page, arm `show_me`, a look then a second look (calls carry 0, 1, 2): `agent.invoke` with no scope, `core.stream_turn`, `core.with_heartbeat(core.stream_turn(...))` (the page's real path: the turn runs in the pump thread, tools in the framework's worker threads), `agent.ainvoke` inside a scope, `agent.ainvoke` with none. Same, all five.
- Two looks in one step (two tool calls, run in two threads; the next call carries 2): `stream_turn`, heartbeat, `ainvoke` in a scope. Same.
- The reading helper (`task`) looking twice with `show_me` (the helper's calls carry 0, 1, 2; the main agent's 0): `invoke`, `stream_turn`, heartbeat, `ainvoke` in a scope. Same.
- `GEOTECH_VISION_SHOW_UPLOADS=1`, one upload, then the same upload and a second one in one step (1, then 2): four entry points. Same.
- `minimal` build: a look, then two looks in one step (1, then 2), five entry points; its helper looking twice, two entry points. Same.
- All settings unset, a look at an upload: 0 images, a vision call as before. Same.
- The page uses only the sync `agent.stream` (no `astream`/`ainvoke` in the package); the async rows show the store does not depend on it.
- Interrupted and resumed: the page has no path that resumes a turn. Its auto-continue and page-check passes run inside the same scope and each adds a user message, after which neither tree shows earlier images (`newest_image_ids` stops at the last user message). Read in `core.py`, not run.

## Question 4: what the process holds

- A running turn: at most 96 images and 384 descriptions on its scope. An upload is at most 4 MB (`INLINE_IMAGE_MAX_BYTES`), so the worst case is 96 x 4 MB = 384 MB per running turn; page views are far smaller. Measured: a turn that stored 200 held 96 images and 104 descriptions, process store 0.
- After a turn ends (returns, fails or is stopped): nothing; `turn_scope` drops them in its `finally`. A turn's generator that is abandoned and collected also frees them (measured).
- A turn whose thread never ends (a hung model or tool call) keeps its images until the thread does; nothing else frees them. Bound: 384 MB times the turns stuck at once.
- Between turns of a conversation: nothing (the head kept up to 96 images for the life of the process).
- Outside any turn: the process store, 96 images at most, never freed, as on the head.

## What I ran (pytest judged by exit code)

- `/tmp/rv/store_cases.py` (rows 1 to 5, 7), `turns.py` (row 6), `mem.py` (question 4): exit 0 on both trees; the tree under test proved by printing `aec_agent_core.__file__`.
- venv (a), from `packages/aec-document-review`: `test_review_fingerprints`, `test_boundary`, `test_dr38_review_fixes`, `test_dr38_show_uploads`, `test_ta43_image_ceiling_page`, `test_review_show_me_offline`, `test_vis26_minimal_images`, `test_review_minimal_offline`: 49 passed, exit 0. Fingerprints not moved.
- venv (a), the core: `test_b15_shown_uploads_store` with `test_dr48_photo_pages`, `test_ev17_run_report`, `test_ev49_image_ceiling_between_arms`, `test_rvb_image_read_once`, `test_ta43_image_ceiling`: 74 passed, exit 0.
- The builder's five tests on the base: 5 failed, exit 1 (each tests the change).
- `webapp/tests/test_public_text.py`, `test_review_freeze.py`: 13 passed, exit 0.
- Scope: three source files in the core, one new test file, the check file; no existing test edited; no prompt, tool description, setting or fingerprint moved.

## CHECK.md

Its claims held where tested. One count differs: its first core run says 73 passed; the same six files gave 74 here.

## Not checked

- venv (b): `webapp/tests/test_page_fingerprints.py` and the `funhouse_agent` tests the check file lists (the geotech app was not installed here). `webapp/core.py` also opens a turn scope, so the geotech page gets the same store; the lead's `ci/` push covers it.
- A turn's `uploads_not_shown` in a written `activity.jsonl`: checked at `image_call_facts` (which `activity_log` reads at turn end), not in a file.
- A live model, a live host, Streamlit itself, and memory under real 4 MB uploads.
