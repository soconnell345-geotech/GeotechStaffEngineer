# User Manual — source & build

The distributed PDF is `../GeotechStaffEngineer_User_Manual_v5.3.pdf`. It is
regenerable from source each release with three steps (run from this folder,
using the project venv):

```bash
# 1. Capture real worked-example output from the live dispatch layer
python gen_worked_examples.py      # -> worked_examples.json

# 2. Build the print-styled HTML (introspects the dispatch registry for the
#    complete method/parameter catalog; pulls narrative from manual_content.py)
python build_manual.py             # -> manual.html + toc.json

# 3. Render to PDF with headless Chrome and attach a navigable outline
python render_pdf.py               # -> ../GeotechStaffEngineer_User_Manual_v5.3.pdf
```

## Files

| File | Role |
|------|------|
| `build_manual.py` | Renderer. Introspects `funhouse_agent.dispatch` for the complete, always-in-sync method/parameter catalog (Chapter 4 + Appendix A); assembles the document; emits `manual.html` + `toc.json`. |
| `manual_content.py` | All hand-written narrative: cover, disclaimer, chapter bodies, per-module practitioner prose (problems / methods+references / limits), worked-example prompts, validation table, version history. **Edit this to change the words.** |
| `gen_worked_examples.py` | Runs ~18 representative calls through the real dispatch layer and saves their output to `worked_examples.json`, so the worked examples quote authentic numbers. |
| `worked_examples.json` | Captured worked-example output (build input). |
| `render_pdf.py` | Headless-Chrome render + PyMuPDF outline/bookmark generation + a blank-page/outline verification. |
| `manual.html`, `toc.json` | Build artifacts. |

## Notes

- The method/parameter reference is generated from the shipped code, so it cannot
  drift from the installed package. When modules change, re-run all three steps.
- The disclaimer chapter mirrors the language in the repo-root `DISCLAIMER.md`.
- Rendering needs Google Chrome or Microsoft Edge (see `CHROME_CANDIDATES` in
  `render_pdf.py`) and PyMuPDF (`pip install PyMuPDF`, already in `[full]`/`[pdf]`).
