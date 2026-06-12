import fitz, sys, os, glob, re

docs = r"C:/Users/socon/OneDrive/dev/GeotechStaffEngineer/.claude/worktrees/v5.1-todos/geotech-references/docs"
sys.stdout.reconfigure(encoding='utf-8')
# usage: find_kw.py "<pdf substring>" "<regex>"
sub, rx = sys.argv[1], re.compile(sys.argv[2], re.I)
matches = [p for p in glob.glob(docs + "/*.pdf") if sub.lower() in os.path.basename(p).lower()]
for pdf in matches:
    doc = fitz.open(pdf)
    print(f"--- {os.path.basename(pdf)} ---")
    for i, page in enumerate(doc):
        txt = page.get_text()
        for m in rx.finditer(txt):
            s = txt.rfind('\n', 0, m.start()) + 1
            e = txt.find('\n', m.end())
            print(f"p{i+1}: {txt[s:e if e>0 else None].strip()[:110]}")
    doc.close()
