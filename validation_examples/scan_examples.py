import json, glob, re, os, sys

base = r"C:/Users/socon/OneDrive/dev/GeotechStaffEngineer/.claude/worktrees/v5.1-todos/geotech-references/geotech_references"
pat = re.compile(r'\b(example|sample problem|illustrative|design example)\b', re.I)

def sections(d):
    """Yield (id, title, body) from whatever structure the JSON has."""
    if isinstance(d, dict):
        if 'sections' in d and isinstance(d['sections'], list):
            for s in d['sections']:
                if isinstance(s, dict):
                    yield s.get('id', s.get('section', '?')), s.get('title', ''), s.get('body', s.get('text', ''))
        else:
            # maybe dict of id -> {title, body}
            for k, v in d.items():
                if isinstance(v, dict) and ('body' in v or 'text' in v or 'title' in v):
                    yield k, v.get('title', ''), v.get('body', v.get('text', ''))
    elif isinstance(d, list):
        for s in d:
            if isinstance(s, dict):
                yield s.get('id', '?'), s.get('title', ''), s.get('body', s.get('text', ''))

hits_title = []
hits_body = []
for f in sorted(glob.glob(base + "/*/text/*.json")):
    ref = os.path.relpath(f, base).replace('\\', '/')
    try:
        d = json.load(open(f, encoding='utf-8'))
    except Exception as e:
        print("ERR", ref, e)
        continue
    for sid, title, body in sections(d):
        body = body or ''
        if pat.search(title or ''):
            hits_title.append((ref, sid, title, len(body)))
        elif body and re.search(r'(design example|example problem|sample problem|illustrative example|STEP 1[:.\s])', body, re.I):
            hits_body.append((ref, sid, title, len(body)))

print("=== TITLE HITS ===")
for r, s, t, n in hits_title:
    print(f"{r} :: {s} :: {t[:80]} :: bodylen={n}")
print(f"\n=== BODY HITS ({len(hits_body)}) ===")
for r, s, t, n in hits_body:
    print(f"{r} :: {s} :: {t[:80]} :: bodylen={n}")
