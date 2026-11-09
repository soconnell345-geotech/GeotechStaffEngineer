# EV-35: the chart suite ready to merge (review fixes and notes)

Branch `merge/ev35`: the suite's branch with master merged in (bd736a4), whose
suite code is 8f1f8c3, the commit that ran on Foundry. The run's results were
read in full (`module_work/review_eval_results/2026-11-08_chart_suite_ev35/TRACE_REVIEW.md`).
The owner dropped the suite's record fixes and the rerun on 2026-11-09, so this
pass fixes only the public-repo findings (F1, F2) and the two small run
defects (F3, F6), and notes the rest. Nothing here changes how the run already
made was graded: it graded every task with CharXiv's own prompts, and those are
untouched.

Findings are from the independent code review (kept outside the repo; F1 to
F9, written on 60575d1, before the move from Chartography to CharXiv)
and TRACE_REVIEW §5. Each was checked against the current code. Line numbers
are on this branch's head.

## Fixed

| Finding | Commit | Where now | What changed |
|---|---|---|---|
| **F2** the verdict parser mirrored Surge's `_parse_verdict` (fence-stripping lines, the scores check and the boolean rejection copied) | 916926f | `funhouse_agent/chart_eval/judge.py:99-168` (`_outer_object`, `_decision`, `_part`, `parse_verdict`) | Written in this package's own form. It takes the JSON from the reply's first `{` to its last `}`, then reads one decision, or one per part when the object lists parts under `scores`. A decision is `correct` or `score`, as true/false or 1/0; anything else is no verdict and the judge is asked again. A verdict cut off part way never reads as its first part. The judge record calls the list `parts` (`part`, `answer`, `correct`, `reason`). Tests: `test_verdicts_parse_strictly` (`tests/test_chart_eval_offline.py:214`), `test_charxiv_verdicts_and_slots` (`:760`). |
| **F1** `OWN_TEMPLATE` paraphrased Surge's unlicensed judge prompt step by step, with one phrase and one example word for word | dd4fb2d | `judge.py:41-58`; docstring `judge.py:1-25`; `data.py:32`; `prepare.py:130` | Written afresh from what a grader of a short chart answer needs: the question, the reference and the answer to check; judge the result the answer settles on; the same value however written, any value in a stated range, the same things named, every part right. One paragraph rather than numbered steps, no examples, and its own reply form (`final_answer`, `correct`, `reason`). It grades only when the data folder holds no prompt. Nothing used it in the run, which graded with CharXiv's prompts (Apache 2.0, still used word for word, with `ATTRIBUTION.md` written by `charxiv.py:197,213-231`). Tests: `test_the_data_folders_judge_prompt_is_used_word_for_word` (`:249`) checks that the prompt asks for the reply its parser reads. `judge_fake` (`:86`) answers in the new form. |
| **F3** forked page children inherited the parent's live HTTP connection pools | 6671c1e | `runner.py:234-306` (`drop_inherited_clients`, `model_for_child`, `_fork_runs.child`); `__init__.py:47,69-77` (`_BUILT`, `rebuild_model`) | Before its run each child clears the process-wide client caches (langchain-anthropic's and langchain-openai's per-address clients, the app's CA-bundle clients). A model `foundry_model` built is then built again from its id, giving new clients and new `palantir_models` handles; any other model drops the clients it built on first use (`_client`, `_async_client`). `run.json` records it under `child_clients` (pid, rebuilt, dropped). Nothing inherited is closed, because the parent still uses it. Tests: `test_a_child_drops_the_clients_it_inherits` (`:595`), `test_a_model_foundry_model_built_is_built_again_in_the_child` (`:613`); `test_forked_page_runs_call_the_model_through_clients_of_their_own` (`:640`) is POSIX only, skipped here, and runs in CI. |
| **F6** a retried direct run kept the failed attempt's images | c7d983d | `runner.py:126-147` (`_set_aside`), `:153` | Before a run, whatever an earlier attempt left in its folder (transcript, `images/`, `box/`, `page/`, `run.json`) moves to `earlier_attempts/NN/` inside it, and `run.json` names that folder (`earlier_attempt`). The failed attempt's record is kept. Test: `test_a_retry_moves_the_failed_attempt_aside` (`:471`). |

Limits of the F3 fix, not chased:
- A model the caller built itself and handed a client or SDK handle keeps it
  in the child. Build it with `foundry_model` to give each child its own.
  Settings changed on a `foundry_model` object after it was built are not
  carried into the child's copy (`__init__.py:69-77` says so).
- Whether `palantir_models` itself keeps one process-wide session behind
  `.get()` cannot be checked off Foundry.

Limits of the F6 fix: with a mirror (`durable_dir`/`sharepoint`), the files a
failed attempt mirrored stay in the mirror at their old paths, and a later
restore into an empty `out_dir` brings them back beside the retry's files.

## Noted, not fixed (owner, 2026-11-09: no record fixes, no rerun)

From REVIEW.md:

- **F4** (GPT reasoning summaries missing from the direct arms' transcripts).
  Still as reviewed: `arms.py:127-144` reads content blocks only, and
  `arms.py:232,240` stores `reasoning or None`. It never reads
  `additional_kwargs["reasoning"]`, where the app's Foundry route puts it.
  TRACE_REVIEW §5.2 confirms: none of Sol's 358 direct-arm calls carries it.
- **F5** (`prepare.py` reads only parquet). Still `prepare.py:71-78`. This is
  the Chartography path only, which the owner cannot open; CharXiv's folder is
  made by `charxiv.py` from JSON.
- **F7** (Claude's direct arms see up to 2576 px, the page at most 2048 px).
  Still `imaging.py:72-74,136-137`. TRACE_REVIEW §5.6: every CharXiv chart is at
  most 1,024 px, so every arm saw it at its own size. Only zoom magnification
  differs, and no wrong run traces to it.
- **F8** (accuracy drops failed and unscored runs from the denominator).
  Still `runner.py:392-397`. TRACE_REVIEW §5.6: no run failed or went unscored,
  so all 100 count on every arm.
- **F9** notes:
  - The desk card still says "a gitignored raw/ folder"
    (`module_work/backlog_dashboard/context/other_owner.json:516`), but no
    top-level `raw/` is ignored. The card is the lead's and is not edited here.
  - Brief §7 (`module_work/FOUNDRY_BRIEF_CHART_SUITE.md:210-216`) says `page/`
    holds a `run.json`; `run_task` writes none there. The brief does not
    mention `earlier_attempts/` or `child_clients` (F3, F6); it is the record of
    the run that was made.
  - The judge retries at once on any error, rate limits included
    (`judge.py:190-200`). A run left unscored is graded again on the next run.

From TRACE_REVIEW §5:

- **5.1** The page arms' calls and tokens include the vision probe's 4 or 5
  calls per child. `run_page` takes `model_calls` and tokens as the page
  reports them (`runner.py:110-112`), and `summarize` averages them
  (`runner.py:408`). REPORT §1's cost lines overstate the page arms; no
  accuracy effect.
- **5.2** = F4.
- **5.3** `run.json` holds neither the question, the golden answer nor the
  judge's prompt (`runner.py:191-205` passes them to the judge, and `_save`
  writes the result without them). Direct transcripts do not keep the prompt
  text (`arms.py:293-295,421-427`). `page_minimal`'s `turn_start` records no
  engine class or model.
- **5.4** `run_python`'s description (`arms.py:65-80`) does not say each call
  starts a fresh process. Printed output keeps only its last 3,000 characters,
  with no mark (`funhouse_agent/plot_box/__init__.py:693-696`).
- **5.5** Neither `run.json` nor `RESULTS.md` counts errors, retries or vision
  fallbacks per run. That is why the eight lowered-detail Sol runs (§4.1) did
  not show in the hand-back.
- **5.6** The earlier review's flags: F3 and F6 are fixed above; F7 and F8 are
  noted above.

## Checks

- **The data folder is never committed.** `.gitignore:44-46` lists
  `chartography_data*` and `charxiv_data*`. `git check-ignore` confirms both
  folders, their zips, files inside them (`tasks.jsonl`, `judge_prompts.json`,
  `charts/*.jpg`, `judge_prompt.txt`) and the same folders nested anywhere
  (`sub/charxiv_data/...`, `module_work/charxiv_data/...`). `git ls-files`
  tracks no `tasks.jsonl`, `judge_prompt(s)`, `sample_N.txt`, `PROVENANCE.json`,
  `ATTRIBUTION.md`, CharXiv `reasoning_val`/`image_metadata`, `images.zip` or
  chart image. The run's results folder tracks only `TRACE_REVIEW.md`. The tests
  build their charts and questions on the spot (`SECRET QUESTION i?`,
  `GOLDEN i`, a red square; `charxiv_fetcher` with made-up questions).
- **No Foundry ids, workspace names or person names.** The branch's added
  lines against master (3,585 lines) were scanned:
  - None of these turned up: a RID (`ri.<service>.<...>`), a UUID, a Foundry
    workspace or project name, an e-mail address or a local user path.
  - None of the review's banned patterns turned up either: the owner's name,
    the AI FDE's folder and dataset names, the Department's domains and
    hosts, and the Foundry ids from earlier runs. They are not repeated here
    because this file is public.
  - "Workspace" appears only as the generic word ("Code Workspace",
    "the workspace's Models panel").
  - The only URLs are CharXiv's GitHub and Hugging Face, arXiv abstracts,
    Surge's public `scorer.py` (fetched only into the private folder), the
    CC BY-SA and Apache licence pages, and `proxy.example.invalid` in a test.
  - The only person's name is "Zirui Wang et al.", CharXiv's published
    citation in `charxiv.py:215`, which its CC BY-SA 4.0 licence requires.
- **Surge's prompt and code stay out of the package.** `prepare.py` still
  fetches Surge's `JUDGE_PROMPT_TEMPLATE` with `ast` into the private data
  folder only (`prepare.py:48-68,124-132`). That is the Chartography path, not
  used by the run.

## Tests

Run in a worktree of this branch with the main checkout's `.venv` python
(`python -m pytest`). Imports resolve to the worktree, which was checked by printing
`funhouse_agent.chart_eval.judge.__file__` and `webapp.engine_config.__file__`.
Each suite was started only with at least 2 GB of RAM free, and each is judged
by its exit code.

| Suite | Exit | Result |
|---|---|---|
| `funhouse_agent/chart_eval/tests` | 0 | 31 passed, 2 skipped (the two forked-children tests, POSIX only) |
| `webapp/tests/test_public_text.py` | 0 | 5 passed (run again after this file was committed, since it reads every tracked file) |
| `webapp/tests/test_page_fingerprints.py` | 0 | 15 passed |

The two POSIX-only tests (`tests/test_chart_eval_offline.py:557,640`) were
not run here; CI's Linux runners run them.
