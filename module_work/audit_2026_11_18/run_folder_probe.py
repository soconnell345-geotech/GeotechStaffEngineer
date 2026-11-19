"""Audit of 2026-11-18, question 5: made-up run folders scored as the runner
scores them, then re-scored offline (rescore=True). Run with one tree first on
the path. Made-up values and the suite's own truth text only."""
import json, os, sys, shutil, tempfile
from pathlib import Path
import aec_document_review.review_eval as RE
from aec_agent_core.review_eval import runner as R, checks as C
from aec_document_review.review_eval.tasks import OPEN_TASKS
import openpyxl, docx
T={t.id:t for t in OPEN_TASKS}
tid="coord-full-review"; task=T[tid]
truth=task.truth
class NoModel:
    model="none"
    def invoke(self,*a,**k): raise AssertionError("model called")
    bind_tools=invoke
def xlsx(path, rows, sheet2=None):
    b=openpyxl.Workbook(); 
    for r in rows: b.active.append(r)
    if sheet2:
        s=b.create_sheet("Log")
        for r in sheet2: s.append(r)
    b.save(path)
def folder(name, answer, make, files_rel=None, outcome=None, activity=()):
    base=Path(tempfile.mkdtemp(prefix="q5_"))
    rd=base/"runs"/"baseline"/tid; fd=rd/"files"; fd.mkdir(parents=True)
    make(fd)
    with open(rd/"activity.jsonl","w") as fh:
        for rec in activity: fh.write(json.dumps(rec)+"\n")
    produced=sorted(str(p) for p in fd.rglob("*") if p.is_file())
    records=R._activity_records(str(rd))
    live=[C.run_check(c,answer,files=produced,tool_calls=[],activity=records) for c in task.all_checks()]
    ls=C.score(live)
    res={"task":tid,"arm":"baseline","answer":answer,"error":None,"outcome_error":outcome,
         "files":files_rel if files_rel is not None else [os.path.relpath(p,rd) for p in produced],
         "tool_calls":[],"score":ls,"checks":live}
    (rd/"run.json").write_text(json.dumps(res))
    out=RE.score_review_suite(model=NoModel(),out_dir=base,arms=("baseline",),ids=[tid],rescore=True,verbose=False)
    rs=out["results"]["baseline"][tid]["score"]
    lp=[c["label"] for c in live if c["passed"] and not c["info"]]
    rp=[c["label"] for c in out["results"]["baseline"][tid]["checks"] if c["passed"] and not c["info"]]
    print(f"{name}: live {ls['checks_passed']}/{ls['checks_total']} rescored {rs['checks_passed']}/{rs['checks_total']}  {'SAME' if lp==rp else 'DIFFERENT: only live '+str(sorted(set(lp)-set(rp)))+' only rescored '+str(sorted(set(rp)-set(lp)))}")
    return live, out, base
rows=[["No.","Finding"]]+[[i+1,l] for i,l in enumerate(x for x in truth.replace(". ",".\n").split("\n") if x.strip())]
ptr="The review is complete. The findings are in the attached comment log."
print("truth as the chat answer:"); l,_,_=folder("A0 truth in chat", truth, lambda fd: None)
print("   failed:", [(c['label'],c['detail'][:150]) for c in l if not c['passed'] and not c['info']])
l,_,_=folder("A1 truth rows in comment_log.xlsx", ptr, lambda fd: xlsx(fd/"comment_log.xlsx", rows))
print("   failed:", [(c['label'],c['detail'][:120]) for c in l if not c['passed'] and not c['info']])
l,_,_=folder("A2 truth as one docx paragraph", ptr, lambda fd: (lambda d:(d.add_paragraph(truth),d.save(fd/"review.docx")))(docx.Document()))
l,_,_=folder("A3 truth on the workbook's 2nd sheet", ptr, lambda fd: xlsx(fd/"comment_log.xlsx", [["Cover"]], rows))
l,_,_=folder("B  corrupt xlsx (cannot be opened)", ptr, lambda fd: (fd/"comment_log.xlsx").write_bytes(b"not a workbook"))
print("   details:", sorted({c['detail'][-60:] for c in l if not c['passed']})[:3])
l,_,_=folder("B2 corrupt docx", ptr, lambda fd: (fd/"review.docx").write_bytes(b"not a document"))
l,_,_=folder("C  two logs: final (truth) + neutral draft", ptr, lambda fd: (xlsx(fd/"comment_log.xlsx", rows), xlsx(fd/"comment_log_draft.xlsx", [["Sheet","Note"],["A-101","door 101 is missing from the schedule"],["A-102","detail 4/A-501 does not exist"]])))
print("   failed:", [(c['label'],c['detail'][:160]) for c in l if not c['passed'] and not c['info']])
l,_,_=folder("D1 ended on a limit, no answer, no files", "", lambda fd: None, outcome="GraphRecursionError: limit")
print("   passed on nothing:", [c['label'] for c in l if c['passed'] and not c['info']])
l,_,_=folder("D2 ended on a limit, no answer, log delivered", "", lambda fd: xlsx(fd/"comment_log.xlsx", rows), outcome="GraphRecursionError: limit")
l,_,_=folder("E  files recorded with backslashes (saved on Windows)", ptr, lambda fd: xlsx(fd/"comment_log.xlsx", rows), files_rel=["files\\comment_log.xlsx"])
l,_,b=folder("F  files/ folder not copied with run.json", ptr, lambda fd: xlsx(fd/"comment_log.xlsx", rows))
shutil.rmtree(b/"runs"/"baseline"/tid/"files")
out=RE.score_review_suite(model=NoModel(),out_dir=b,arms=("baseline",),ids=[tid],rescore=True,verbose=False)
print("   after removing files/: rescored", out["results"]["baseline"][tid]["score"], "before", out["results"]["baseline"][tid]["score_before_rescore"]["checks_passed"])
l,_,_=folder("G  truth only in files/.scratch/notes.md", ptr, lambda fd: ((fd/".scratch").mkdir(), (fd/".scratch"/"notes.md").write_text(truth)))
print("   notes_only:", sum(1 for c in l if c.get('notes_only')))
l,_,_=folder("H  truth in files/out/log.md (subfolder)", ptr, lambda fd: ((fd/"out").mkdir(), (fd/"out"/"log.md").write_text(truth)))
l,_,_=folder("K  truth in review.pdf-like unread kinds (.html, .doc, .pptx name)", ptr, lambda fd: ((fd/"review.html").write_text(truth),(fd/"review.rtf").write_text(truth)))
try:
    RE.score_review_suite(model=None,out_dir=b,arms=("baseline",),ids=[tid],rescore=True,verbose=False); print("I  rescore with model=None: ok")
except Exception as e: print("I  rescore with model=None:", type(e).__name__, e)
# a saved run with error set: rescore tries to run it again
rd=b/"runs"/"baseline"/tid; d=json.loads((rd/"run.json").read_text()); d["error"]="TimeoutError: x"; (rd/"run.json").write_text(json.dumps(d))
try:
    out=RE.score_review_suite(model=NoModel(),out_dir=b,arms=("baseline",),ids=[tid],rescore=True,verbose=False); r=out["results"]["baseline"][tid]; print("J  rescore of a run saved with an error:", r.get("error"), r.get("score"), 'answer kept' if r.get('answer') else 'answer GONE')
except BaseException as e: print("J  rescore of a run saved with an error:", type(e).__name__, str(e)[:200])
