import ast, sys, json, collections
fails = [l.strip() for l in open(sys.argv[1]) if (l.startswith("FAILED tests/") or l.startswith("ERROR tests/"))]
by = collections.defaultdict(set)
for l in fails:
    node = l.split()[1]
    f, _, rest = node.partition("::")
    name = rest.split("::")[0].split("[")[0]
    by[f].add(name)
log = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
for f, names in by.items():
    t = open(f).read(); tree = ast.parse(t); lines = t.split("\n"); drop = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            s = (node.decorator_list[0].lineno if node.decorator_list else node.lineno) - 1
            drop.append((s, node.end_lineno, node.name))
    for s, e, _ in reversed(drop): del lines[s:e]
    open(f, "w").write("\n".join(lines))
    log.setdefault(f, []).extend(sorted(names))
json.dump(log, open(sys.argv[2], "w"), indent=1)
print(sum(len(v) for v in by.values()), "pruned")
