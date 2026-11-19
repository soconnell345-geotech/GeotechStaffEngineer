"""Audit of 2026-11-18, question 4: build every fixture document of one tree
and write its pages (words with positions, drawing and image counts) to the
JSON file named as the first argument; compare two such files by hand."""
import sys, json, hashlib, tempfile, os
import aec_document_review.review_eval
from aec_agent_core.review_eval import documents as D
import fitz
out={}
tmp=tempfile.mkdtemp()
for key,spec in D.DOCUMENTS.items():
    if "fixture" not in spec: continue
    try:
        p=D.resolve(key, tmp)
    except Exception as e:
        try: p=D.resolve(key)
        except Exception as e2: out[key]={"error":repr(e2)[:200]}; continue
    pass
    d=fitz.open(stream=p[1],filetype="pdf")
    pages=[]
    for pg in d:
        words=sorted((round(w[0]),round(w[1]),w[4]) for w in pg.get_text("words"))
        pages.append(hashlib.md5(json.dumps([words,len(pg.get_drawings()),len(pg.get_images())]).encode()).hexdigest()[:10])
    out[key]={"n":len(d),"pages":pages,"text":[pg.get_text() for pg in d]}
json.dump(out,open(sys.argv[1],"w"))
print({k:(v.get("n"),v.get("error")) for k,v in out.items()})
