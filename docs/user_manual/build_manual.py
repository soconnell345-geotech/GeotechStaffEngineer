#!/usr/bin/env python
"""Build the GeotechStaffEngineer PDF user manual.

Regenerable per release. Two inputs:

1. The SHIPPED CODE — the method/parameter reference in Chapter 5 and the
   Appendix is generated PROGRAMMATICALLY by introspecting
   ``funhouse_agent.dispatch`` / the adapter ``METHOD_INFO`` registries, so the
   catalog can never drift from what the package actually exposes.
2. The HAND-WRITTEN NARRATIVE — practitioner prose, chapter bodies, worked
   examples, validation tables, version history — lives in ``manual_content.py``.

Real worked-example OUTPUT is read from ``worked_examples.json`` (captured by
running the representative calls through the live dispatch layer).

Usage:
    python build_manual.py            # writes manual.html
    # then render to PDF with headless Chrome (see README at bottom / task notes).
"""
from __future__ import annotations

import html
import json
import re
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parent.parent
sys.path.insert(0, str(WORKTREE))
sys.path.insert(0, str(HERE))

from funhouse_agent.adapters import MODULE_REGISTRY  # noqa: E402
from funhouse_agent import dispatch  # noqa: E402
import manual_content as C  # noqa: E402

OUT_HTML = HERE / "manual.html"
EXAMPLES_JSON = HERE / "worked_examples.json"


# ---------------------------------------------------------------------------
# Programmatic catalog extraction (the completeness guarantee)
# ---------------------------------------------------------------------------

def extract_catalog() -> dict:
    """Introspect every registered module -> methods, briefs, params."""
    cat = {"modules": {}, "stats": {}}
    n_methods = n_params = 0
    for name in sorted(MODULE_REGISTRY):
        spec = MODULE_REGISTRY[name]
        entry = {
            "brief": spec.get("brief", ""),
            "is_reference": name in dispatch.REFERENCE_MODULES,
            "methods": {},
        }
        try:
            mod = dispatch._load_adapter(name)
            for mname, info in getattr(mod, "METHOD_INFO", {}).items():
                if info.get("alias_of"):
                    continue
                params = info.get("parameters") or {}
                clean = {}
                for pn, pi in params.items():
                    if not isinstance(pi, dict):
                        clean[pn] = {"desc": str(pi)}
                        continue
                    clean[pn] = {
                        "desc": pi.get("desc") or pi.get("description") or "",
                        "type": pi.get("type", ""),
                        "required": bool(pi.get("required", False)),
                        "default": pi.get("default", None),
                        "allowed_values": pi.get("allowed_values") or [],
                    }
                entry["methods"][mname] = {
                    "brief": info.get("brief", ""),
                    "category": info.get("category", "General"),
                    "parameters": clean,
                }
                n_methods += 1
                n_params += len(clean)
        except Exception as e:  # pragma: no cover
            entry["load_error"] = f"{type(e).__name__}: {e}"
        cat["modules"][name] = entry
    cat["stats"] = {
        "n_modules": len(cat["modules"]),
        "n_analysis": len(dispatch.ANALYSIS_MODULES),
        "n_reference": len(dispatch.REFERENCE_MODULES),
        "n_methods": n_methods,
        "n_params": n_params,
    }
    return cat


def read_version() -> str:
    txt = (WORKTREE / "pyproject.toml").read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', txt, re.M)
    return m.group(1) if m else "?"


def read_extras() -> list[tuple[str, str]]:
    """Parse [project.optional-dependencies] into (extra, 'libs') pairs."""
    txt = (WORKTREE / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"\[project\.optional-dependencies\](.*?)(\n\[)", txt, re.S)
    out = []
    if block:
        for m in re.finditer(r'^([A-Za-z0-9_-]+)\s*=\s*\[(.*?)\]', block.group(1), re.S | re.M):
            name = m.group(1)
            libs = re.findall(r'"([^">=<]+)', m.group(2))
            out.append((name, ", ".join(s.strip() for s in libs)))
    return out


# ---------------------------------------------------------------------------
# Inline formatting: a tiny, safe markdown-ish -> HTML
# ---------------------------------------------------------------------------

_ENTITY = re.compile(r"&(?:[a-zA-Z][a-zA-Z0-9]*|#\d+|#x[0-9a-fA-F]+);")


def esc(s) -> str:
    """HTML-escape, but leave already-valid HTML entities (&mdash;, &phi;, &#8399;)
    intact so authored copy can use them freely."""
    s = str(s)
    out = []
    last = 0
    for m in _ENTITY.finditer(s):
        out.append(html.escape(s[last:m.start()], quote=False))
        out.append(m.group(0))
        last = m.end()
    out.append(html.escape(s[last:], quote=False))
    return "".join(out)


def inline(s: str) -> str:
    """Escape (entity-preserving), then re-enable `code`, **bold**, *italic*,
    and a small whitelist of inline tags (<sub>, <sup>) used for subscripts."""
    s = esc(s)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![*\w])\*([^*\n]+)\*(?![*\w])", r"<em>\1</em>", s)
    s = re.sub(r"&lt;(/?(?:sub|sup|code|strong|em|b|i|br\s*/?))&gt;", r"<\1>", s)
    return s


def bullets(items) -> str:
    if not items:
        return ""
    return "<ul>" + "".join(f"<li>{inline(x)}</li>" for x in items) + "</ul>"


# ---------------------------------------------------------------------------
# Document builder — accumulates body HTML + a numbered table of contents
# ---------------------------------------------------------------------------

class Doc:
    def __init__(self):
        self.body: list[str] = []
        self.toc: list[tuple[int, str, str, str]] = []  # (level, number, title, anchor)
        self._ch = 0
        self._sec = 0
        self._sub = 0
        self._anchors: set[str] = set()

    def _anchor(self, base: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", base.lower()).strip("-")[:60] or "sec"
        a = slug
        i = 2
        while a in self._anchors:
            a = f"{slug}-{i}"
            i += 1
        self._anchors.add(a)
        return a

    def add(self, htmltext: str):
        self.body.append(htmltext)

    def chapter(self, title: str, *, appendix: bool = False):
        self._ch += 1
        self._sec = 0
        num = f"Appendix {chr(ord('A') + self._ch - C.APPENDIX_START_CH)}" if appendix else f"{self._ch}"
        label = f"Appendix {num.split()[-1]}" if appendix else f"Chapter {num}"
        a = self._anchor(f"ch-{num}")
        self.toc.append((1, num, title, a))
        self.body.append(
            f'<h1 class="chapter" id="{a}">'
            f'<span class="ch-kicker">{esc(label)}</span>'
            f'<span class="ch-title">{inline(title)}</span></h1>'
        )
        return num

    def section(self, title: str, *, in_toc: bool = True) -> str:
        self._sec += 1
        self._sub = 0
        num = f"{self._ch}.{self._sec}"
        a = self._anchor(f"s-{num}")
        if in_toc:
            self.toc.append((2, num, title, a))
        self.body.append(f'<h2 class="section" id="{a}"><span class="num">{num}</span> {inline(title)}</h2>')
        return num

    def subsection(self, title: str, *, in_toc: bool = False, monospace: bool = False) -> str:
        self._sub += 1
        num = f"{self._ch}.{self._sec}.{self._sub}"
        a = self._anchor(f"ss-{num}")
        if in_toc:
            self.toc.append((3, num, title, a))
        cls = "subsection mono" if monospace else "subsection"
        self.body.append(f'<h3 class="{cls}" id="{a}"><span class="num">{num}</span> {inline(title)}</h3>')
        return num

    def html(self) -> str:
        return "\n".join(self.body)

    def toc_html(self) -> str:
        rows = []
        for level, num, title, anchor in self.toc:
            cls = f"toc-l{level}"
            rows.append(
                f'<div class="{cls}"><a href="#{anchor}">'
                f'<span class="toc-num">{esc(num)}</span>'
                f'<span class="toc-title">{inline(title)}</span></a></div>'
            )
        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def method_param_table(params: dict) -> str:
    if not params:
        return '<p class="muted">No parameters.</p>'
    rows = []
    for pn, pi in params.items():
        req = '<span class="req">required</span>' if pi.get("required") else '<span class="opt">optional</span>'
        typ = esc(pi.get("type", "") or "")
        desc = inline(pi.get("desc", "") or "")
        av = pi.get("allowed_values") or []
        if av:
            desc += ' <span class="allowed">allowed: ' + ", ".join(f"<code>{esc(v)}</code>" for v in av) + "</span>"
        rows.append(
            f"<tr><td class='pname'><code>{esc(pn)}</code></td>"
            f"<td class='ptype'>{typ}</td><td class='preq'>{req}</td>"
            f"<td class='pdesc'>{desc}</td></tr>"
        )
    return (
        "<table class='params'><thead><tr>"
        "<th>Parameter</th><th>Type</th><th></th><th>Description</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def methods_overview_table(methods: dict) -> str:
    # group by category
    by_cat: dict[str, list] = {}
    for mname, info in methods.items():
        by_cat.setdefault(info.get("category", "General"), []).append((mname, info))
    parts = []
    for cat in sorted(by_cat):
        rows = "".join(
            f"<tr><td class='mname'><code>{esc(mn)}</code></td>"
            f"<td class='mbrief'>{inline(info.get('brief',''))}</td>"
            f"<td class='mnp'>{len(info.get('parameters') or {})}</td></tr>"
            for mn, info in by_cat[cat]
        )
        parts.append(
            f"<table class='methods'><thead><tr><th colspan='3' class='catrow'>{esc(cat)}</th></tr>"
            f"<tr><th>Method</th><th>What it does</th><th>Params</th></tr></thead>"
            f"<tbody>{rows}</tbody></table>"
        )
    return "\n".join(parts)


def worked_example_block(key: str, ask: str, note: str = "") -> str:
    """Render a worked example from the captured live output."""
    data = WORKED.get(key)
    if not data:
        return ""
    mod, meth = key.split(".", 1)
    params = data.get("params", {})
    result = data.get("result", {})
    call = f"call_agent(\n    agent_name={mod!r},\n    method={meth!r},\n    parameters={_fmt_params(params)},\n)"
    out_lines = _fmt_result(result)
    parts = ['<div class="worked">']
    parts.append('<div class="worked-h">Worked example</div>')
    parts.append(f'<p class="ask"><span class="ask-label">Ask&nbsp;&rarr;</span> {inline(ask)}</p>')
    parts.append(f'<pre class="call"><code>{esc(call)}</code></pre>')
    parts.append('<div class="out-label">Representative output (real, from the shipped code)</div>')
    parts.append(f'<pre class="out"><code>{esc(out_lines)}</code></pre>')
    if note:
        parts.append(f'<p class="worked-note">{inline(note)}</p>')
    parts.append("</div>")
    return "\n".join(parts)


def _fmt_params(p: dict) -> str:
    try:
        s = json.dumps(p, indent=8, default=str)
        # tighten closing brace indent
        return s[:-1] + "    }"
    except Exception:
        return str(p)


def _fmt_result(res, max_items: int = 14) -> str:
    if not isinstance(res, dict):
        return str(res)[:600]
    lines = []
    for k, v in res.items():
        if isinstance(v, bool):
            lines.append(f"{k}: {v}")
        elif isinstance(v, (int, float, str)):
            sv = v if not isinstance(v, float) else round(v, 4)
            lines.append(f"{k}: {sv}")
        elif isinstance(v, dict):
            inner = ", ".join(
                f"{a}={round(b,4) if isinstance(b,float) else b}"
                for a, b in list(v.items())[:6] if isinstance(b, (int, float, str, bool))
            )
            lines.append(f"{k}: {{{inner}}}")
        elif isinstance(v, list):
            lines.append(f"{k}: [{len(v)} entries]")
        if len(lines) >= max_items:
            lines.append("… (additional fields in the full result dict)")
            break
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Load captured worked-example output
# ---------------------------------------------------------------------------

WORKED = {}
if EXAMPLES_JSON.exists():
    WORKED = json.loads(EXAMPLES_JSON.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

def css() -> str:
    return r"""
@page { size: Letter; margin: 20mm 18mm 20mm 18mm; }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  font-family: Georgia, "Times New Roman", serif;
  font-size: 10.6pt; line-height: 1.5; color: #1a1f27; margin: 0;
}
p { margin: 0 0 0.62em 0; }
a { color: #14507a; text-decoration: none; }
code, pre, .mono, .mono * { font-family: "Consolas", "SFMono-Regular", "Menlo", monospace; }
code { font-size: 0.86em; background: #f2f4f7; padding: 0.5px 3px; border-radius: 3px; color: #123; }
.muted { color: #6b7787; }
strong { color: #10151c; }

/* ---- cover ---- */
.cover {
  height: 246mm; display: flex; flex-direction: column; justify-content: center;
  page-break-after: always; text-align: center;
  background: linear-gradient(150deg, #0e1f2e 0%, #113247 55%, #0d2233 100%);
  color: #eef4f9; margin: -20mm -18mm 0 -18mm; padding: 0 24mm;
}
.cover .mark { font-size: 12pt; letter-spacing: 5px; text-transform: uppercase; color: #7fd3ff; margin-bottom: 26px; }
.cover h1 { font-family: "Segoe UI", Arial, sans-serif; font-size: 33pt; line-height: 1.12; margin: 0 0 10px 0; font-weight: 700; color: #ffffff; }
.cover .sub { font-size: 14.5pt; color: #bfe0f2; font-style: italic; margin: 8px 0 30px; }
.cover .tagline { font-size: 11pt; color: #d6e6f1; max-width: 150mm; margin: 0 auto; line-height: 1.6; }
.cover .meta { margin-top: 40px; font-size: 10.5pt; color: #9dc4dc; font-family: "Segoe UI", Arial, sans-serif; }
.cover .meta .ver { color: #ffffff; font-weight: 700; font-size: 12pt; }
.cover .rule { width: 70px; height: 3px; background: #37a0d8; margin: 22px auto; border: none; }

/* ---- generic pages ---- */
.page-break { page-break-after: always; }
h1.chapter {
  page-break-before: always; font-family: "Segoe UI", Arial, sans-serif;
  margin: 0 0 18px 0; padding-bottom: 10px; border-bottom: 2.5px solid #14507a;
}
h1.chapter .ch-kicker { display: block; font-size: 10.5pt; letter-spacing: 3px; text-transform: uppercase; color: #2a7db0; margin-bottom: 4px; }
h1.chapter .ch-title { display: block; font-size: 22pt; color: #0e2233; font-weight: 700; line-height: 1.15; }
h2.section {
  font-family: "Segoe UI", Arial, sans-serif; font-size: 14.5pt; color: #123a54;
  margin: 22px 0 8px; padding-top: 4px; page-break-after: avoid; border-bottom: 1px solid #d6dee6; padding-bottom: 3px;
}
h3.subsection { font-family: "Segoe UI", Arial, sans-serif; font-size: 12pt; color: #14507a; margin: 16px 0 6px; page-break-after: avoid; }
h3.subsection.mono { font-size: 12.5pt; color: #0e2233; }
.section .num, .subsection .num { color: #8aa3b5; font-weight: 600; margin-right: 6px; }
h4 { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.6pt; color: #2a3a48; margin: 12px 0 4px; page-break-after: avoid; }

/* ---- table of contents ---- */
.toc { page-break-after: always; }
.toc h1 { font-family: "Segoe UI", Arial, sans-serif; font-size: 20pt; color: #0e2233; border-bottom: 2.5px solid #14507a; padding-bottom: 8px; margin-bottom: 14px; }
.toc-l1, .toc-l2, .toc-l3 { margin: 1px 0; }
.toc-l1 a, .toc-l2 a, .toc-l3 a { display: flex; color: #1a1f27; }
.toc-l1 { margin-top: 9px; font-family: "Segoe UI", Arial, sans-serif; font-weight: 700; font-size: 11pt; color: #0e2233; }
.toc-l2 { padding-left: 16px; font-size: 10pt; }
.toc-l3 { padding-left: 34px; font-size: 9.2pt; color: #55636f; }
.toc-num { flex: 0 0 42px; color: #2a7db0; font-variant-numeric: tabular-nums; }
.toc-l1 .toc-num { color: #14507a; }
.toc-title { flex: 1; }

/* ---- tables ---- */
table { border-collapse: collapse; width: 100%; margin: 8px 0 14px; font-size: 9.1pt; page-break-inside: auto; }
thead { display: table-header-group; }
tr { page-break-inside: avoid; }
th { background: #123a54; color: #eaf3fa; text-align: left; padding: 5px 7px; font-family: "Segoe UI", Arial, sans-serif; font-weight: 600; font-size: 8.8pt; }
td { padding: 4px 7px; border-bottom: 1px solid #e3e8ee; vertical-align: top; }
tbody tr:nth-child(even) { background: #f6f8fa; }
table.methods th.catrow { background: #2a7db0; color: #fff; font-size: 9.4pt; letter-spacing: 0.3px; }
table.params .pname { white-space: nowrap; width: 20%; }
table.params .ptype { color: #6b7787; font-style: italic; width: 8%; }
table.params .preq { width: 9%; }
table.methods .mname { width: 26%; }
table.methods .mnp { width: 8%; text-align: center; color: #6b7787; }
.req { color: #b3421a; font-weight: 600; font-size: 8.2pt; font-family: "Segoe UI", Arial, sans-serif; }
.opt { color: #6b7787; font-size: 8.2pt; font-family: "Segoe UI", Arial, sans-serif; }
.allowed { color: #4a5a68; font-size: 0.94em; }

/* ---- callouts ---- */
.callout { border-left: 4px solid #2a7db0; background: #eef6fb; padding: 9px 13px; margin: 12px 0; border-radius: 0 4px 4px 0; page-break-inside: avoid; }
.callout.warn { border-left-color: #c9820a; background: #fdf6e9; }
.callout.limit { border-left-color: #9a4a2a; background: #fbefe9; }
.callout .co-h { font-family: "Segoe UI", Arial, sans-serif; font-weight: 700; font-size: 9.4pt; text-transform: uppercase; letter-spacing: 0.5px; color: #123a54; margin-bottom: 3px; }
.callout.warn .co-h { color: #8a5a06; }
.callout.limit .co-h { color: #7a2f16; }

/* ---- worked example ---- */
.worked { border: 1px solid #cfdae4; border-radius: 5px; margin: 12px 0 16px; page-break-inside: avoid; overflow: hidden; }
.worked-h { background: #123a54; color: #eaf3fa; font-family: "Segoe UI", Arial, sans-serif; font-weight: 600; font-size: 9pt; letter-spacing: 0.5px; text-transform: uppercase; padding: 4px 12px; }
.worked .ask { padding: 9px 12px 4px; margin: 0; font-size: 10pt; }
.ask-label { font-family: "Segoe UI", Arial, sans-serif; color: #2a7db0; font-weight: 700; font-size: 9pt; }
pre { margin: 0; padding: 9px 12px; background: #f6f8fa; overflow-x: auto; white-space: pre-wrap; word-break: break-word; font-size: 8.5pt; line-height: 1.42; }
pre.call { background: #0e2233; color: #d6e6f1; border-top: 1px solid #cfdae4; }
pre.call code { background: none; color: #d6e6f1; }
.out-label { font-family: "Segoe UI", Arial, sans-serif; font-size: 8.2pt; text-transform: uppercase; letter-spacing: 0.5px; color: #56707f; padding: 6px 12px 2px; background: #f0f4f7; }
pre.out { background: #f0f4f7; color: #14303f; }
pre.out code { background: none; }
.worked-note { padding: 6px 12px 10px; margin: 0; font-size: 9pt; color: #4a5a68; font-style: italic; }

/* ---- module card ---- */
.mod-purpose { font-size: 10.4pt; }
.mod-refs { font-size: 9.2pt; color: #4a5a68; }
.field-label { font-family: "Segoe UI", Arial, sans-serif; font-weight: 700; font-size: 9pt; text-transform: uppercase; letter-spacing: 0.4px; color: #2a7db0; margin: 12px 0 2px; }
.pill { display: inline-block; background: #e7eef4; color: #123a54; border-radius: 10px; padding: 1px 9px; font-family: "Segoe UI", Arial, sans-serif; font-size: 8pt; margin-right: 5px; }
.pill.ref { background: #efe7f4; color: #4a2a54; }
.lead { font-size: 11.4pt; color: #2a3a48; line-height: 1.55; }
hr.soft { border: none; border-top: 1px solid #e0e6ec; margin: 16px 0; }
.small { font-size: 9pt; color: #55636f; }
ul { margin: 4px 0 10px; padding-left: 20px; }
li { margin: 2px 0; }
.kpi { display: flex; gap: 10px; flex-wrap: wrap; margin: 10px 0; }
.kpi .box { flex: 1 1 120px; border: 1px solid #cfdae4; border-radius: 5px; padding: 8px 10px; background: #f8fafc; }
.kpi .box .n { font-family: "Segoe UI", Arial, sans-serif; font-size: 17pt; font-weight: 700; color: #14507a; }
.kpi .box .l { font-size: 8.4pt; color: #56707f; text-transform: uppercase; letter-spacing: 0.4px; }
"""


# ---------------------------------------------------------------------------
# Assemble the document
# ---------------------------------------------------------------------------

def _plain(s: str) -> str:
    """Strip inline tags + unescape entities to a plain text string for the PDF
    outline / bookmark search."""
    s = re.sub(r"</?(?:code|sub|sup|strong|em|b|i)>", "", str(s))
    s = (s.replace("&mdash;", "—").replace("&ndash;", "–")
           .replace("&amp;", "&").replace("&nbsp;", " ").replace("&hellip;", "…")
           .replace("&rsquo;", "’").replace("&prime;", "′"))
    s = re.sub(r"&#\d+;|&#x[0-9a-fA-F]+;|&[a-zA-Z]+;", "", s)  # drop remaining entities
    return re.sub(r"\s+", " ", s).strip()


def build():
    cat = extract_catalog()
    version = read_version()
    extras = read_extras()
    st = cat["stats"]
    d = Doc()

    # ---- content chapters (delegated to manual_content, which calls back into
    #      these render helpers via a small context object) ----
    ctx = RenderCtx(d, cat, version, extras, st)
    C.render(ctx)

    # ---- assemble ----
    cover = C.cover_html(version, st)
    disclaimer = C.disclaimer_html()
    toc = f'<section class="toc"><h1>Contents</h1>{d.toc_html()}</section>'
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>GeotechStaffEngineer User Manual v{esc(version)}</title>
<style>{css()}</style></head>
<body>
{cover}
{disclaimer}
{toc}
{d.html()}
</body></html>"""
    toc_entries = [(lvl, f"{num}  {_plain(title)}".strip(), _plain(title))
                   for (lvl, num, title, _a) in d.toc]
    return doc, toc_entries


class RenderCtx:
    """Passed to manual_content.render() so the content module can emit
    structured HTML through the Doc builder + the shared render helpers."""

    def __init__(self, doc: Doc, cat: dict, version: str, extras, stats):
        self.doc = doc
        self.cat = cat
        self.version = version
        self.extras = extras
        self.stats = stats

    # thin proxies
    def chapter(self, t, **k): return self.doc.chapter(t, **k)
    def section(self, t, **k): return self.doc.section(t, **k)
    def subsection(self, t, **k): return self.doc.subsection(t, **k)
    def add(self, h): self.doc.add(h)
    def para(self, t, cls=None):
        self.doc.add(f'<p class="{cls}">{inline(t)}</p>' if cls else f"<p>{inline(t)}</p>")
    def raw(self, h): self.doc.add(h)
    def inline(self, t): return inline(t)
    def bullets(self, items): self.doc.add(bullets(items))
    def callout(self, title, body, kind=""):
        cls = f"callout {kind}".strip()
        self.doc.add(f'<div class="{cls}"><div class="co-h">{esc(title)}</div>{body}</div>')
    def methods_table(self, module): self.doc.add(methods_overview_table(self.cat["modules"][module]["methods"]))
    def params_table(self, module, method):
        self.doc.add(method_param_table(self.cat["modules"][module]["methods"][method]["parameters"]))
    def worked(self, key, ask, note=""): self.doc.add(worked_example_block(key, ask, note))
    def module_methods(self, module): return self.cat["modules"].get(module, {}).get("methods", {})


def main():
    doc, toc_entries = build()
    OUT_HTML.write_text(doc, encoding="utf-8")
    (HERE / "toc.json").write_text(json.dumps(toc_entries, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    print(f"Wrote {OUT_HTML}  ({len(doc):,} bytes)")
    print(f"Wrote {HERE / 'toc.json'}  ({len(toc_entries)} entries)")


if __name__ == "__main__":
    main()
