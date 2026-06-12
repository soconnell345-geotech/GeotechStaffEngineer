import fitz, glob, os, re, sys

docs = r"C:/Users/socon/OneDrive/dev/GeotechStaffEngineer/.claude/worktrees/v5.1-todos/geotech-references/docs"
pat = re.compile(r'(design example|example problem|sample problem|illustrative example|example calculation|EXAMPLE \d|Example \d+[-.:]|STEP 1\b)', re.I)

targets = sys.argv[1:] if len(sys.argv) > 1 else None
for pdf in sorted(glob.glob(docs + "/*.pdf")):
    name = os.path.basename(pdf)
    if targets and not any(t.lower() in name.lower() for t in targets):
        continue
    try:
        doc = fitz.open(pdf)
    except Exception as e:
        print("ERR", name, e); continue
    hits = []
    for i, page in enumerate(doc):
        txt = page.get_text()
        for m in pat.finditer(txt):
            # grab the line containing the match
            start = txt.rfind('\n', 0, m.start()) + 1
            end = txt.find('\n', m.end())
            line = txt[start:end if end > 0 else None].strip()
            hits.append((i + 1, line[:100]))
    print(f"\n===== {name} ({doc.page_count} pages, {len(hits)} hits) =====")
    seen = set()
    for p, line in hits:
        key = line.lower()
        tag = f"p{p}: {line}"
        print(tag)
    doc.close()
