# To-do-only steps in the brief 7 runs (DR-43, part 2)

Measured 2026-11-13 over the 84 saved runs, with no model call. A step is one model call by the main agent or a helper together with the tool calls it made; a to-do-only step is one whose only tool call was `write_todos`. Each such step sends the whole conversation again, so its cost is its input tokens. The share is given two ways: of the run's whole input (vision side calls included) and of the input of tool-calling steps alone.

## By model

| model | runs | tool-calling steps | to-do-only steps | input, all calls (M) | input, to-do-only steps (M) | share of all input | share of step input |
|---|---|---|---|---|---|---|---|
| Sol | 70 | 1176 | 308 | 122.7 | 16.08 | 13.1 % | 33.4 % |
| GPT-5.4 | 14 | 162 | 59 | 11.6 | 2.88 | 24.9 % | 48.2 % |

## By task, Sol (five set-ups each) and GPT-5.4

| task | Sol: share of all input, lowest to highest | Sol: median | Sol: to-do-only steps per run | GPT-5.4: share of all input | GPT-5.4: to-do-only steps |
|---|---|---|---|---|---|
| coord-sheet-index | 0 to 20 % | 0 % | 0 to 4 | 0 % | 0 |
| coord-callouts | 15 to 26 % | 21 % | 4 to 5 | 27 % | 4 |
| coord-schedules | 19 to 34 % | 27 % | 3 to 5 | 33 % | 5 |
| coord-levels-dimensions | 14 to 27 % | 23 % | 4 to 5 | 24 % | 3 |
| coord-drawings-vs-manual | 12 to 26 % | 23 % | 4 to 6 | 29 % | 5 |
| coord-drawings-vs-geotech | 17 to 28 % | 25 % | 4 to 5 | 40 % | 5 |
| coord-revisions | 4 to 18 % | 5 % | 3 to 4 | 37 % | 5 |
| coord-submittal | 25 to 42 % | 32 % | 5 to 5 | 42 % | 6 |
| coord-full-review | 8 to 30 % | 18 % | 4 to 7 | 31 % | 5 |
| coord-scan-callouts | 6 to 20 % | 14 % | 4 to 5 | 17 % | 4 |
| coord-scan-schedules | 18 to 29 % | 24 % | 4 to 6 | 24 % | 5 |
| coord-long-callouts | 4 to 15 % | 5 % | 4 to 5 | 16 % | 3 |
| coord-long-schedules | 6 to 19 % | 11 % | 4 to 6 | 23 % | 5 |
| coord-long-review | 3 to 5 % | 4 % | 3 to 6 | 16 % | 4 |

## By set-up, Sol

| set-up | to-do-only steps | share of all input | seconds in to-do-only steps |
|---|---|---|---|
| `baseline` | 57 | 11.7 % | 245 |
| `show_me` | 63 | 11.2 % | 241 |
| `method` | 67 | 16.2 % | 284 |
| `sheet_tools` | 60 | 14.3 % | 232 |
| `all_three` | 61 | 12.5 % | 231 |
