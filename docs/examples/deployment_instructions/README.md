# Example deployment instructions (made-up content)

A folder like this one, named by `GEOTECH_INSTRUCTIONS`, adds a deployment's
own conventions to the agents (BACKLOG REF-64 (B)). Everything here is
invented for illustration; replace it with your own.

- `all.md` is appended to every agent and helper, first.
- `<role>.md` is appended after it to one role: `main` (the main agent),
  `references`, `reviewer`, `calc`, `figure`, `general-purpose`,
  `model_setup`, `report_library`, `report_ingest`. The role is the helper's
  name, so a future helper is covered by adding its file.
- Files are UTF-8 text, read when an agent is built (the next conversation
  picks up an edit, no restart).
- At most 4,000 characters per role (`all.md` and the role's file together);
  past that the text is cut at a paragraph and the turn record says so.
- An empty or missing file adds nothing; an unreadable folder adds nothing
  and the record names the error.
- A file no agent or helper takes (a misspelt role name such as
  `reviwer.md`, or a role that page does not have) adds nothing; the turn
  record lists it under `unused`, and empty files under `empty`.

The app puts the built-in protected rules (cite the printed page, formulas
and defaults only from a printed source, state assumptions and the method,
say when a reading failed or is uncertain, never invent a value) and a fixed
heading above this text. Deployment text can set conventions, preferred
methods cited from the deployment's own reference packs, defaults, units and
formats; it cannot switch the protected rules or the guards in code off.

This README is not read by the app (only `all.md` and the role files are).
