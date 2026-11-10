"""Copy GSE tests into the package (step 13): rename, then drop the test
functions (and classes) that still need GSE code; print what was dropped."""
import ast, sys, os, re, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from copy_step13 import rewrite, core_aliases
al = core_aliases()
dest = "packages/aec-document-review/tests"
GSE = re.compile(r"\b(funhouse_agent|webapp|geotech_references|geotech_common|report_ingest|calc_package|profile_figure|GEOTECH_PAGE|profiles\.GEOTECH|geotech_page)\b")
import io, tokenize
def _code_only(src):
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.STRING, tokenize.COMMENT):
                continue
            if getattr(tokenize, "FSTRING_MIDDLE", None) == tok.type:
                continue
            out.append(tok.string)
    except Exception:
        return src
    return " ".join(out)
report = {}
for src in sys.argv[1:]:
    t = open(src, encoding="utf-8").read()
    t = t.replace("webapp.tests.page_builds", "tests.page_builds").replace("from webapp.tests import page_builds", "from tests import page_builds")
    t = rewrite(t, al)
    t = t.replace("aec_document_review.tests", "tests")
    t = t.replace("funhouse_agent.review_eval", "aec_document_review.review_eval")
    def _multi(m):
        names = [n.strip() for n in m.group(1).replace("\n", " ").split(",") if n.strip()]
        if all(f"funhouse_agent.{n.split(' as ')[0]}" in al for n in names):
            return "from aec_agent_core import (" + m.group(1) + ")"
        return m.group(0)
    t = re.sub(r"from funhouse_agent import \(([^)]*)\)", _multi, t)
    t = re.sub(r"from funhouse_agent import \(  # noqa: E402\n([^)]*)\)", lambda m: "from aec_agent_core import (  # noqa: E402\n" + m.group(1) + ")" if all(f"funhouse_agent.{n.strip().split(' as ')[0]}" in al for n in m.group(1).replace("\n"," ").split(",") if n.strip()) else m.group(0), t)
    for a, b in [
        ("from funhouse_agent.deep.tools import make_vision_tools", "from aec_agent_core.deep.vision_kit import make_vision_tools"),
        ("from funhouse_agent.deep import agent as agent_mod", "from aec_document_review import review_build as agent_mod"),
        ("from funhouse_agent.deep import prompt as P", "from aec_document_review import review_prompt as P"),
        ("from funhouse_agent.deep.tests.test_review_harness_offline import", "from tests.test_review_harness_offline import"),
        ('os.path.join(_ROOT, "webapp", "app.py")', 'os.path.join(_ROOT, "aec_document_review", "app.py")'),
    ]:
        t = t.replace(a, b)
    t = t.replace("from funhouse_agent.deep.agent import build_deep_agent", "from aec_document_review.review_build import build_review_page_agent as build_deep_agent")
    t = t.replace("import funhouse_agent.deep.agent as deep_agent", "import aec_agent_core.deep.assembly as deep_agent")
    t = re.sub(r"^from aec_document_review import geotech_page.*\n", "", t, flags=re.M)
    t = re.sub(r"^from funhouse_agent\.deep import tools as tools_mod.*\n", "", t, flags=re.M)
    tree = ast.parse(t)
    lines = t.split("\n")
    drop = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            s = (node.decorator_list[0].lineno if node.decorator_list else node.lineno) - 1
            e = node.end_lineno
            seg = _code_only("\n".join(lines[s:e]))
            if GSE.search(seg):
                drop.append((s, e, node.name))
    for s, e, _ in reversed(drop):
        del lines[s:e]
    out = "\n".join(lines)
    name = os.path.basename(src)
    # module-level GSE reference left: report it
    left = [l for l in out.split("\n") if GSE.search(l) and not l.lstrip().startswith("#")]
    open(os.path.join(dest, name), "w", encoding="utf-8", newline="\n").write(out)
    report[name] = {"dropped": [d[2] for d in drop], "module_level_left": left[:6]}
json.dump(report, open("test_copy_report.json", "w"), indent=1)
for k,v in report.items():
    print(k, len(v["dropped"]), "LEFT:" if v["module_level_left"] else "", v["module_level_left"][:3])
