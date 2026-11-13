# Cloud reviewer brief: one branch of a Document Review round

Binding for a one-time cloud session the lead starts to review ONE branch that another cloud helper built. Your routine's prompt names the branch, the task file and section it was built from, the folder for your report and the branch your report goes on.

You are an independent reviewer. The lead has not read the code and will decide whether to merge from what you write. The builder's `CHECK.md` is a set of claims to check, not evidence. You change no code: your whole delivery is one report file.

## Read first

1. `module_work/cloud_briefs/COMMON_DOCREVIEW.md`: the rules the builder was bound by, and the setup commands. They bind you too, except "What you deliver" and "Finish with", which this file replaces.
2. The task file your prompt names: its top part (with the table "Who edits what") and the one section your prompt names.
3. The `BACKLOG.md` items that section closes, in full, and the evidence they cite.
4. The builder's `CHECK.md` on the branch.

A planlens review runs in the planlens repository: clone this repository read-only as COMMON_DOCREVIEW says and read these files from the clone.

## What to do

1. **The diff.** `git fetch origin <branch>`; the base is `git merge-base origin/master FETCH_HEAD` (`origin/main` in planlens). Read the whole of `git diff <base>..FETCH_HEAD`, then the code around it: every caller of a changed function and every reader of a changed result key, event, setting or file, across the whole repository. A review of the diff alone is not enough.
2. **Scope.** `git diff --name-only <base>..FETCH_HEAD`: every file must be one the section's row of "Who edits what" allows, a new test, or the CHECK file. Name any file outside that, any edit to an existing test (is the reason given true?), and any file COMMON_DOCREVIEW says is never edited.
3. **Does it do what each item asks**, on every path and for every model? Name each path the change misses and each it reaches by mistake. Where the section says "confirm first", confirm it yourself on the base.
4. **Defects.** Anything that returns a wrong or misleading result, raises, loses a file or part of one, leaks a private key or another tester's data into a result, waits without a bound, or changes behaviour the items did not ask to change. Give the file and line and a concrete made-up input that shows it. Where you suspect a defect, write a small test of your own in `/tmp` (never committed) and report what it did.
5. **A shared host.** For anything that runs in the testers' page: several testers share ONE process and its threads. Go through it with two testers acting at the same moment, a tester with two browser tabs, a turn that is stopped or fails halfway, a restart of the process mid-turn, and a slow or failing outside service. Look at every lock (what it guards, how long it is held, whether it is keyed per tester or per conversation or shared by all), every background thread (who ends it, what it does when the page is gone), and every module-level table (how it is keyed, when an entry leaves).
6. **The owner's rules.**
   - Nothing fitted to one model: is any new prompt or tool wording general, or written around one model's habit?
   - No switch on by default: is each piece the builder calls a repair truly a repair, as COMMON_DOCREVIEW defines one? Is each new behaviour off unless its setting is set, with the page and its fingerprints unchanged while it is unset? Is each setting documented beside the code with its default?
   - No edge-case chasing: did the change go beyond the item?
   - Public repositories: made-up values only; none of the words COMMON_DOCREVIEW lists.
7. **The tests.**
   - Run the tests the section names and the builder's new tests on the branch, in the venvs the section names. Report commands and passed/failed counts.
   - Prove each new test fails without the change: `git worktree add /tmp/base <base>`, copy the new test files into it, run them there. List every new test that passes on the base (it tests nothing new) and say whether that is by design.
   - Do the tests check behaviour, or only that some text is present?
   - Run both fingerprint tests. Moved or not? If moved, is the re-record in a commit of its own, and does it say what moved?
8. **CHECK.md.** List every statement in it that is untrue or that you could not support. Is the release note what a tester or a GSE user would actually notice?

Judge the branch as it stands. Do not propose redesigns, and do not list style points: a finding is worth a line only if it would change what a saved run or a live turn returns, lose or expose data, or break a rule above.

## What you deliver

ONE file, in ONE commit, on a NEW branch (named in your prompt) created from the head of the branch you reviewed, pushed to origin:

- in this repository `module_work/<folder your prompt names>/REVIEW.md`;
- in planlens `REVIEW_<ID>.md` at the repository root.

Under 150 lines, in this order:

1. **Verdict**, one line: `MERGE`, `MERGE AFTER FIXES` or `DO NOT MERGE`, and the reviewed head's short hash.
2. **Findings**, most serious first, as a table: severity (`must-fix`: a defect or a broken rule; `should-fix`: wrong but narrow; `note`), file and line, what is wrong, the input that shows it, the smallest fix.
3. **Paths checked**: each path you traced, and whether the change holds there.
4. **Tests**: the commands and counts on the branch; for each new test, failed or passed on the base; your own `/tmp` tests and what they showed; the fingerprints.
5. **Scope**: files outside the task's row, edited existing tests, anything the items did not ask for.
6. **CHECK.md**: statements untrue or unsupported.
7. **Could not check**: what needs a live host, a live service or a model, and exactly how the lead would check it.

The commit message ends with `Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>` (or the model your prompt names). Then a final message of under 100 words: the verdict, the count of must-fix findings, the review branch.

## Never

Edit, add or delete any file but your report. Push to the branch you reviewed, to `master` or `main`, or to a branch named `ci/...`. Open a pull request, create a tag, publish anything. Edit `BACKLOG.md`, `HANDOFF.md` or `CLAUDE.md`.
