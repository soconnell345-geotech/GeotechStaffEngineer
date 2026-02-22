"""
GeotechStaffEngineer - Live Project Dashboard

Run:  python dashboard_app.py
Open:  http://127.0.0.1:8050
"""

import os
import subprocess
from dash import Dash, html, dcc
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Data gathering
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

MODULE_META = [
    # Core analysis modules
    ("bearing_capacity",   "Shallow foundations (CBEAR/Vesic/Meyerhof)",                "Foundations"),
    ("settlement",         "Consolidation & immediate (CSETT)",                          "Foundations"),
    ("axial_pile",         "Driven pile capacity (Nordlund/Tomlinson/Beta)",              "Piles"),
    ("sheet_pile",         "Cantilever/anchored walls (Rankine/Coulomb)",                 "Walls"),
    ("lateral_pile",       "Lateral pile (COM624P, 8 p-y models, FD solver)",             "Piles"),
    ("pile_group",         "Rigid cap groups (6-DOF, Converse-Labarre)",                  "Piles"),
    ("wave_equation",      "Smith 1-D wave equation (bearing graph, drivability)",        "Piles"),
    ("drilled_shaft",      "GEC-10 alpha/beta/rock socket",                               "Piles"),
    ("seismic_geotech",    "Site class, M-O pressures, liquefaction",                     "Seismic"),
    ("retaining_walls",    "Cantilever + MSE walls (GEC-11)",                             "Walls"),
    ("ground_improvement", "Aggregate piers, wick drains, surcharge, vibro (GEC-13)",     "Ground"),
    ("slope_stability",    "Fellenius/Bishop/Spencer, circular slip, grid search",        "Ground"),
    ("downdrag",           "Fellenius neutral plane, UFC 3-220-20 downdrag",              "Piles"),
    ("geotech_common",     "SoilProfile + engineering checks + adapters",                 "Common"),
    # Library wrapper agents
    ("opensees_agent",     "PM4Sand cyclic DSS, BNWF lateral pile, 1D site response",    "Library"),
    ("pystrata_agent",     "1D EQL site response (SHAKE-type, Darendeli/Menq)",           "Library"),
    ("seismic_signals_agent", "Earthquake signal processing (eqsig/pyrotd)",              "Library"),
    ("liquepy_agent",      "CPT-based liquefaction triggering (B&I 2014)",                "Library"),
    ("pygef_agent",        "CPT/borehole file parser (GEF/BRO-XML)",                     "Data"),
    ("hvsrpy_agent",       "HVSR site characterization from ambient noise",              "Advanced"),
    ("gstools_agent",      "Geostatistical kriging, variogram fitting, random fields",   "Advanced"),
    ("ags4_agent",         "AGS4 geotechnical data format reader/validator",              "Data"),
    ("salib_agent",        "Sobol & Morris sensitivity analysis",                        "Advanced"),
    ("pyseismosoil_agent", "MKZ/HH nonlinear soil curve calibration + Vs profiles",      "Library"),
    ("swprocess_agent",    "MASW surface wave dispersion analysis",                      "Advanced"),
    ("geolysis_agent",     "USCS/AASHTO classification + SPT corrections + bearing",     "Data"),
    ("pystra_agent",       "FORM/SORM/Monte Carlo structural reliability",               "Advanced"),
    ("pydiggs_agent",      "DIGGS 2.6 XML schema and dictionary validation",             "Data"),
]

CATEGORY_COLORS = {
    "Foundations": "#38bdf8",
    "Piles":      "#818cf8",
    "Walls":      "#fb923c",
    "Seismic":    "#f472b6",
    "Ground":     "#34d399",
    "Common":     "#94a3b8",
    "Library":    "#a78bfa",
    "Data":       "#fbbf24",
    "Advanced":   "#f43f5e",
}


def _count_py_lines(directory):
    """Count total Python lines and files in a directory tree."""
    total_lines = 0
    total_files = 0
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for f in files:
            if f.endswith(".py"):
                fpath = os.path.join(root, f)
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                        total_lines += sum(1 for _ in fh)
                    total_files += 1
                except OSError:
                    pass
    return total_lines, total_files


def _count_tests(module_dir):
    """Count test functions in a module's test files."""
    count = 0
    test_dir = os.path.join(module_dir, "tests")
    search_dirs = [test_dir, module_dir]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if not f.startswith("test_") and f != "validation.py":
                continue
            fpath = os.path.join(d, f)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        stripped = line.strip()
                        if stripped.startswith("def test_"):
                            count += 1
            except OSError:
                pass
    return count


def gather_stats():
    """Scan the project and return all dashboard data."""
    modules = []
    for name, desc, cat in MODULE_META:
        mod_dir = os.path.join(PROJECT_ROOT, name)
        lines, files = _count_py_lines(mod_dir)
        tests = _count_tests(mod_dir)
        modules.append({
            "name": name,
            "description": desc,
            "category": cat,
            "lines": lines,
            "files": files,
            "tests": tests,
        })

    # Foundry wrappers
    foundry = []
    for f in sorted(os.listdir(PROJECT_ROOT)):
        if f.endswith("_agent_foundry.py"):
            agent_name = f.replace("_agent_foundry.py", "")
            fpath = os.path.join(PROJECT_ROOT, f)
            with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                lines = sum(1 for _ in fh)
            foundry.append({"name": agent_name, "file": f, "lines": lines})

    foundry_module_names = {fw["name"] for fw in foundry}

    # DM7
    dm7_dir = os.path.join(PROJECT_ROOT, "DM7Eqs")
    dm7_lines, dm7_files = _count_py_lines(dm7_dir)

    # Project totals
    total_lines, total_files = _count_py_lines(PROJECT_ROOT)

    # Git commit count
    try:
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=5,
        )
        git_commits = int(result.stdout.strip()) if result.returncode == 0 else 0
    except Exception:
        git_commits = 0

    module_tests = sum(m["tests"] for m in modules)

    return {
        "modules": modules,
        "foundry": foundry,
        "foundry_names": foundry_module_names,
        "dm7": {"lines": dm7_lines, "files": dm7_files, "tests": 2008, "functions": 382},
        "foundry_harness_tests": 121,
        "total_lines": total_lines,
        "total_files": total_files,
        "module_tests": module_tests,
        "total_tests": module_tests + 121 + 2008,
        "git_commits": git_commits,
    }


# ---------------------------------------------------------------------------
# Dash app
# ---------------------------------------------------------------------------

# Dark theme colors (matching static dashboard)
BG = "#0f172a"
CARD = "#1e293b"
CARD_BORDER = "#334155"
TEXT = "#e2e8f0"
TEXT_DIM = "#94a3b8"
ACCENT = "#38bdf8"
ACCENT2 = "#818cf8"
ACCENT3 = "#34d399"
ACCENT4 = "#fb923c"
GREEN = "#22c55e"
RED = "#ef4444"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=CARD,
    plot_bgcolor=CARD,
    font=dict(color=TEXT, family="Segoe UI, system-ui, sans-serif"),
    margin=dict(l=140, r=30, t=40, b=30),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.1)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
)


def stat_card(value, label, color):
    return html.Div([
        html.Div(f"{value:,}" if isinstance(value, int) else str(value),
                 style={"fontSize": "2.2rem", "fontWeight": "700", "color": color, "lineHeight": "1.2"}),
        html.Div(label, style={"color": TEXT_DIM, "fontSize": "0.85rem", "textTransform": "uppercase",
                                "letterSpacing": "0.05em", "marginTop": "0.25rem"}),
    ], style={
        "background": CARD, "border": f"1px solid {CARD_BORDER}", "borderRadius": "12px",
        "padding": "1.2rem 1.5rem", "textAlign": "center",
    })


def badge(text, ok):
    if ok is None:
        return html.Span("--", style={"color": TEXT_DIM})
    bg = f"rgba(34,197,94,0.15)" if ok else f"rgba(239,68,68,0.15)"
    fg = GREEN if ok else RED
    label = "Yes" if ok else "No"
    return html.Span(label, style={
        "display": "inline-block", "padding": "0.15rem 0.6rem", "borderRadius": "999px",
        "fontSize": "0.75rem", "fontWeight": "600", "background": bg, "color": fg,
    })


def build_layout(data):
    modules = data["modules"]
    foundry_names = data["foundry_names"]

    # Sort modules by test count descending for chart
    mods_by_tests = sorted(modules, key=lambda m: m["tests"])
    mods_by_lines = sorted(modules, key=lambda m: m["lines"])

    # --- Tests bar chart ---
    tests_fig = go.Figure()
    tests_fig.add_trace(go.Bar(
        y=[m["name"] for m in mods_by_tests],
        x=[m["tests"] for m in mods_by_tests],
        orientation="h",
        marker_color=[CATEGORY_COLORS.get(m["category"], TEXT_DIM) for m in mods_by_tests],
        text=[str(m["tests"]) for m in mods_by_tests],
        textposition="outside",
        textfont=dict(size=12, color=TEXT),
    ))
    tests_fig.update_layout(title="Tests by Module", height=450, showlegend=False, **PLOTLY_LAYOUT)
    tests_fig.update_yaxes(tickfont=dict(size=12))

    # --- Lines bar chart ---
    lines_fig = go.Figure()
    lines_fig.add_trace(go.Bar(
        y=[m["name"] for m in mods_by_lines],
        x=[m["lines"] for m in mods_by_lines],
        orientation="h",
        marker_color=[CATEGORY_COLORS.get(m["category"], TEXT_DIM) for m in mods_by_lines],
        text=[f"{m['lines']:,}" for m in mods_by_lines],
        textposition="outside",
        textfont=dict(size=12, color=TEXT),
    ))
    lines_fig.update_layout(title="Lines of Code by Module", height=450, showlegend=False, **PLOTLY_LAYOUT)
    lines_fig.update_yaxes(tickfont=dict(size=12))

    # --- Test breakdown pie ---
    pie_fig = go.Figure()
    pie_fig.add_trace(go.Pie(
        labels=["Module Tests", "Foundry Harness", "DM7 Tests"],
        values=[data["module_tests"], data["foundry_harness_tests"], data["dm7"]["tests"]],
        marker=dict(colors=[ACCENT, ACCENT4, ACCENT2]),
        textinfo="label+value",
        textfont=dict(size=14, color=TEXT),
        hole=0.5,
    ))
    pie_fig.update_layout(
        title="Test Distribution",
        height=350,
        paper_bgcolor=CARD, plot_bgcolor=CARD,
        font=dict(color=TEXT, family="Segoe UI, system-ui, sans-serif"),
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True,
        legend=dict(font=dict(color=TEXT_DIM)),
    )

    # --- Module table ---
    table_header = html.Tr([
        html.Th("Module", style={"textAlign": "left"}),
        html.Th("Purpose", style={"textAlign": "left"}),
        html.Th("Tests", style={"textAlign": "right"}),
        html.Th("Lines", style={"textAlign": "right"}),
        html.Th("Files", style={"textAlign": "right"}),
        html.Th("Foundry", style={"textAlign": "center"}),
    ])

    th_style = {"padding": "0.6rem 0.75rem", "color": TEXT_DIM, "fontSize": "0.75rem",
                "textTransform": "uppercase", "letterSpacing": "0.05em",
                "borderBottom": f"1px solid {CARD_BORDER}"}
    for th in table_header.children:
        th.style = {**th_style, **th.style}

    table_rows = []
    for m in modules:
        has_foundry = m["name"] in foundry_names
        # geotech_common doesn't need a Foundry agent
        foundry_cell = badge(None, None) if m["name"] == "geotech_common" else badge("", has_foundry)

        row_style = {"borderBottom": "1px solid rgba(255,255,255,0.04)"}
        td_style = {"padding": "0.55rem 0.75rem", "fontSize": "0.9rem"}
        table_rows.append(html.Tr([
            html.Td(m["name"], style={**td_style, "fontFamily": "Cascadia Code, Fira Code, monospace", "fontSize": "0.85rem"}),
            html.Td(m["description"], style=td_style),
            html.Td(str(m["tests"]), style={**td_style, "textAlign": "right"}),
            html.Td(f"{m['lines']:,}", style={**td_style, "textAlign": "right"}),
            html.Td(str(m["files"]), style={**td_style, "textAlign": "right"}),
            html.Td(foundry_cell, style={**td_style, "textAlign": "center"}),
        ], style=row_style))

    # Totals row
    table_rows.append(html.Tr([
        html.Td(html.Strong("Total"), style={"padding": "0.6rem 0.75rem"}),
        html.Td(""),
        html.Td(html.Strong(str(data["module_tests"])), style={"padding": "0.6rem 0.75rem", "textAlign": "right"}),
        html.Td(html.Strong(f"{sum(m['lines'] for m in modules):,}"), style={"padding": "0.6rem 0.75rem", "textAlign": "right"}),
        html.Td(html.Strong(str(sum(m["files"] for m in modules))), style={"padding": "0.6rem 0.75rem", "textAlign": "right"}),
        html.Td(""),
    ], style={"borderTop": f"2px solid {CARD_BORDER}"}))

    # --- Foundry agents table ---
    foundry_sorted = sorted(data["foundry"], key=lambda f: f["lines"], reverse=True)
    foundry_rows = []
    for fw in foundry_sorted:
        foundry_rows.append(html.Tr([
            html.Td(fw["name"], style={"padding": "0.55rem 0.75rem", "fontFamily": "Cascadia Code, Fira Code, monospace", "fontSize": "0.85rem"}),
            html.Td(f"{fw['lines']:,}", style={"padding": "0.55rem 0.75rem", "textAlign": "right"}),
        ], style={"borderBottom": "1px solid rgba(255,255,255,0.04)"}))
    foundry_rows.append(html.Tr([
        html.Td(html.Strong(f"{len(data['foundry'])} Agents"), style={"padding": "0.6rem 0.75rem"}),
        html.Td(html.Strong(f"{sum(f['lines'] for f in data['foundry']):,}"), style={"padding": "0.6rem 0.75rem", "textAlign": "right"}),
    ], style={"borderTop": f"2px solid {CARD_BORDER}"}))

    # --- Missing agents ---
    analysis_modules = {m["name"] for m in modules if m["name"] != "geotech_common"}
    missing = sorted(analysis_modules - foundry_names)

    missing_rows = []
    for name in missing:
        missing_rows.append(html.Tr([
            html.Td(name, style={"padding": "0.4rem 0.75rem", "fontFamily": "Cascadia Code, Fira Code, monospace", "fontSize": "0.85rem"}),
            html.Td("Not yet built", style={"padding": "0.4rem 0.75rem", "color": TEXT_DIM}),
        ], style={"borderBottom": "1px solid rgba(255,255,255,0.04)"}))

    # --- Layout ---
    card_style = {
        "background": CARD, "border": f"1px solid {CARD_BORDER}",
        "borderRadius": "12px", "padding": "1.5rem", "overflow": "auto",
    }

    section_title_style = {"fontSize": "1.3rem", "fontWeight": "600", "marginBottom": "1rem", "color": TEXT}

    return html.Div([
        # Header
        html.Div([
            html.H1("GeotechStaffEngineer", style={
                "fontSize": "2.2rem", "fontWeight": "700", "marginBottom": "0.3rem",
                "background": f"linear-gradient(135deg, {ACCENT}, {ACCENT2})",
                "WebkitBackgroundClip": "text", "WebkitTextFillColor": "transparent",
            }),
            html.Div("Python toolkit for LLM-based geotechnical engineering agents",
                      style={"color": TEXT_DIM, "fontSize": "1.05rem"}),
            html.Div(f"{data['git_commits']} commits",
                      style={"color": TEXT_DIM, "fontSize": "0.85rem", "marginTop": "0.5rem"}),
        ], style={"textAlign": "center", "marginBottom": "2.5rem", "paddingBottom": "1.5rem",
                   "borderBottom": f"1px solid {CARD_BORDER}"}),

        # Stat cards
        html.Div([
            stat_card(len(modules), "Analysis Modules", ACCENT),
            stat_card(data["total_tests"], "Total Tests", ACCENT3),
            stat_card(data["total_lines"], "Lines of Python", ACCENT2),
            stat_card(data["total_files"], "Python Files", ACCENT4),
            stat_card(len(data["foundry"]), "Foundry Agents", ACCENT3),
            stat_card(data["dm7"]["functions"], "DM7 Functions", ACCENT),
        ], style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
                   "gap": "1rem", "marginBottom": "2rem"}),

        # Charts row
        html.Div([
            html.Div([dcc.Graph(figure=tests_fig, config={"displayModeBar": False})], style={**card_style, "flex": "1"}),
            html.Div([dcc.Graph(figure=lines_fig, config={"displayModeBar": False})], style={**card_style, "flex": "1"}),
        ], style={"display": "flex", "gap": "1.5rem", "marginBottom": "2rem", "flexWrap": "wrap"}),

        # Module inventory table
        html.Div([
            html.Div("Module Inventory", style=section_title_style),
            html.Table([html.Thead(table_header), html.Tbody(table_rows)],
                       style={"width": "100%", "borderCollapse": "collapse"}),
        ], style={**card_style, "marginBottom": "2rem"}),

        # Bottom row: Foundry agents + DM7 + Pie
        html.Div([
            # Foundry table
            html.Div([
                html.Div("Foundry Agent Wrappers", style=section_title_style),
                html.Table([
                    html.Thead(html.Tr([
                        html.Th("Agent", style={**th_style, "textAlign": "left"}),
                        html.Th("Lines", style={**th_style, "textAlign": "right"}),
                    ])),
                    html.Tbody(foundry_rows),
                ], style={"width": "100%", "borderCollapse": "collapse"}),
                # Missing agents
                html.Div([
                    html.Div("Missing Agents", style={**section_title_style, "fontSize": "1rem", "marginTop": "1.5rem"}),
                    html.Table(html.Tbody(missing_rows), style={"width": "100%", "borderCollapse": "collapse"})
                    if missing_rows else html.Div("All modules covered!", style={"color": GREEN, "fontSize": "0.9rem", "marginTop": "0.5rem"}),
                ]),
            ], style={**card_style, "flex": "1"}),

            # DM7 + pie chart
            html.Div([
                html.Div("DM7 Equations Library", style=section_title_style),
                html.Table([
                    html.Thead(html.Tr([
                        html.Th("Metric", style={**th_style, "textAlign": "left"}),
                        html.Th("Count", style={**th_style, "textAlign": "right"}),
                    ])),
                    html.Tbody([
                        html.Tr([html.Td("Functions", style={"padding": "0.55rem 0.75rem"}), html.Td("382", style={"padding": "0.55rem 0.75rem", "textAlign": "right"})]),
                        html.Tr([html.Td("Tests", style={"padding": "0.55rem 0.75rem"}), html.Td("2,008", style={"padding": "0.55rem 0.75rem", "textAlign": "right"})]),
                        html.Tr([html.Td("Python files", style={"padding": "0.55rem 0.75rem"}), html.Td(str(data["dm7"]["files"]), style={"padding": "0.55rem 0.75rem", "textAlign": "right"})]),
                        html.Tr([html.Td("Lines of code", style={"padding": "0.55rem 0.75rem"}), html.Td(f"{data['dm7']['lines']:,}", style={"padding": "0.55rem 0.75rem", "textAlign": "right"})]),
                    ]),
                ], style={"width": "100%", "borderCollapse": "collapse"}),
                dcc.Graph(figure=pie_fig, config={"displayModeBar": False}),
            ], style={**card_style, "flex": "1"}),
        ], style={"display": "flex", "gap": "1.5rem", "marginBottom": "2rem", "flexWrap": "wrap"}),

        # Footer
        html.Div("GeotechStaffEngineer — Live Dashboard powered by Dash",
                  style={"textAlign": "center", "color": TEXT_DIM, "fontSize": "0.8rem",
                          "marginTop": "1rem", "paddingTop": "1rem",
                          "borderTop": f"1px solid {CARD_BORDER}"}),

    ], style={
        "fontFamily": "Segoe UI, system-ui, -apple-system, sans-serif",
        "background": BG, "color": TEXT, "padding": "2rem",
        "minHeight": "100vh", "lineHeight": "1.6",
    })


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

app = Dash(__name__)
app.title = "GeotechStaffEngineer Dashboard"

data = gather_stats()
app.layout = build_layout(data)

if __name__ == "__main__":
    print("\n  Dashboard running at: http://127.0.0.1:8050\n")
    app.run(debug=True, host="127.0.0.1", port=8050)
