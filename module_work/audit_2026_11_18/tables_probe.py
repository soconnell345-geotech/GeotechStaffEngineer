"""Audit of 2026-11-18, question 6: the results table and the to-do-only
count on made-up runs and records. Run with one tree first on the path."""
import json
import aec_document_review.review_eval as RE
from aec_agent_core.review_eval import runner as R
from aec_document_review.review_eval.tasks import OPEN_TASKS
tasks=[t for t in OPEN_TASKS if t.id in ("coord-callouts","coord-schedules","coord-full-review","meck-underdrain")]
def run(p,t,**k):
    d={"score":{"checks_passed":p,"checks_total":t,"passed":p==t},"model_calls":10,"tool_calls":[{"name":"x"}]*5,"tokens_in":1000,"tokens_out":100,"seconds":60,"checks":[]}
    d.update(k); return d
runs={"baseline":{"coord-callouts":run(6,6,todo_only_steps=2,todo_only_input_tokens=200,model_input_tokens=1000),
                  "coord-schedules":run(4,6,todo_only_steps=1,todo_only_input_tokens=100,model_input_tokens=3000),
                  "coord-full-review":run(2,15,outcome_error="GraphRecursionError: x",answer=""),
                  "meck-underdrain":run(4,4)},
      "method":{"coord-callouts":run(5,6,todo_only_steps=0,todo_only_input_tokens=0,model_input_tokens=1000),
                # coord-schedules missing from this arm
                "coord-full-review":run(15,15),
                "meck-underdrain":{"skipped":True,"error":"missing document"},
                }}
runs["method"]["extra-not-in-tasks"]=run(0,9)
md=R.summarize(runs,["baseline","method"],tasks,{"updated":"x","model":"m","versions":{}})
print(md[:3200])
# activity record: to-do-only steps
def me(agent="main",names=None,tok=100,**k):
    d={"event":"model_end","agent":agent,"usage":{"input_tokens":tok}}
    if names is not None: d["tool_names"]=names; d["n_tool_calls"]=len(names)
    d.update(k); return d
ts=lambda a,n: {"event":"tool_start","agent":a,"name":n,"args":{}}
cases={
 "new record: todo-only, todo+tool, two todos, none, vision": [me(names=["write_todos"]),me(names=["write_todos","open_document"]),me(names=["write_todos","write_todos"]),me(names=[]),me(names=["write_todos"],vision=True)],
 "old record (pairing): todo-only, then todo+tool": [me(n_tool_calls=1),ts("main","write_todos"),me(n_tool_calls=2),ts("main","write_todos"),ts("main","open_document"),me(n_tool_calls=0)],
 "helper's todo-only step counted with main's": [me(names=["write_todos"]),me(agent="helper",names=["write_todos"])],
 "usage with prompt_tokens key (another provider's name)": [{"event":"model_end","agent":"main","tool_names":["write_todos"],"usage":{"prompt_tokens":500}},{"event":"model_end","agent":"main","tool_names":["x"],"usage":{"prompt_tokens":500}}],
}
for k,v in cases.items(): print(k,"->",R.todo_only_steps(v))
two=cases["new record: todo-only, todo+tool, two todos, none, vision"]
print("one run:",R.record_counts(two)["todo_only_steps"],"same record twice in one activity.jsonl:",R.record_counts(two+two)["todo_only_steps"])
import tempfile,os
d=tempfile.mkdtemp(); open(os.path.join(d,"activity.jsonl"),"w").write("".join(json.dumps(r)+"\n" for r in (two+[{"event":"model_end","agent":"main","usage":{"input_tokens":50,"output_tokens":5}}])*2))
print("_activity on doubled record:",{k:v for k,v in R._activity(d).items() if k!="tool_calls"})
