"""Live smoke-test harness for the GeotechStaffEngineer app.

Drives the app's real turn path (``webapp.core`` / ``webapp.turn_jobs`` /
``webapp.profiles`` in the order ``webapp/app.py`` calls them) with Claude
models through the Anthropic API, to find PLUMBING bugs before testers do:
files saved outside the conversation folder, links that do not resolve,
missing download cards, SharePoint mirror gaps, tool errors and crashes.
Claude stands in for the production GPT models; its answers are not scored.

Entry point: ``module_work/live_smoke/run.py`` (see its ``--help``).
"""
