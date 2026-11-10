# A review method for Document Review: draft

**Status: a draft, not measured.** BACKLOG DR-06. Nothing here is in the
package or the wrapper template.

The page's prompt is thorough on how to read a document and says nothing on
how to review one. `instructions/main.md` is a method for a set review: the
set before the sheets, the checks on one sheet, the checks that need two
places, what a comment may rest on, and the comment log. `instructions/
general-purpose.md` tells the reading helper what to bring back so the
reviewer can make the cross-sheet checks.

Both are written as deployment instructions (`GEOTECH_INSTRUCTIONS`; the
wrapper README, "Settings"): each role's text goes last in its prompt,
under rules it cannot switch off, at most 4,000 characters a role. So the
draft can be tried without a release:

1. Copy the `instructions/` folder into the wrapper repository (or anywhere
   on the test machine).
2. Set `GEOTECH_INSTRUCTIONS` to that folder's full path (a `.env` line on
   the test machine; an app setting on the App Service, where a folder in
   the repository is `/home/site/wwwroot/instructions`).
3. Connection diagnostics says what each role was given.

Before any of it enters the package's prompt it needs a suite run with
tasks that can show a difference (DR-25: the suite scores 33 to 34 of 35 on
every arm today).

What the draft deliberately does not do:
- name a discipline's checklist (the page reviews architectural, civil,
  structural and MEP sets alike; the user says what they need);
- cite any code or standard (the page has no reference library; DR-13);
- rank comments by severity (the judgement is the reader's).
