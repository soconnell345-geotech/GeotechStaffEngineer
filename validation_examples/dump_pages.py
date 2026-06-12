import fitz, sys, os

docs = r"C:/Users/socon/OneDrive/dev/GeotechStaffEngineer/.claude/worktrees/v5.1-todos/geotech-references/docs"
# usage: dump_pages.py "<pdf substring>" start end [outfile]
sub, start, end = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
out = sys.argv[4] if len(sys.argv) > 4 else None
import glob
matches = [p for p in glob.glob(docs + "/*.pdf") if sub.lower() in os.path.basename(p).lower()]
assert len(matches) == 1, matches
doc = fitz.open(matches[0])
buf = []
for i in range(start - 1, min(end, doc.page_count)):
    buf.append(f"\n========== PAGE {i+1} ==========\n")
    buf.append(doc[i].get_text())
text = "".join(buf)
if out:
    open(out, 'w', encoding='utf-8').write(text)
    print(f"wrote {len(text)} chars to {out}")
else:
    sys.stdout.reconfigure(encoding='utf-8')
    print(text)
