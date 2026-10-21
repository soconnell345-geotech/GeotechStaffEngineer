"""
HTML template for calculation packages.

Single-file Jinja2 template with embedded CSS.
Designed for professional engineering output with print-friendly layout.

The package is three parts: its page (title block, footer), and between
them its sections and reference list (:data:`CALC_SECTIONS_TEMPLATE`,
styled by :data:`CALC_SECTIONS_CSS`). ``render_tables`` (BACKLOG GM-67)
prints the middle part alone, for a report of the caller's own, so the
package and the report print one rendering of the calculation.
"""

_PAGE_CSS = """
/* ---- Base ---- */
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: Georgia, "Times New Roman", serif;
    font-size: 11pt;
    line-height: 1.5;
    color: #1a1a1a;
    background: #fff;
    max-width: 8.5in;
    margin: 0 auto;
    padding: 0.75in 1in;
}

/* ---- Header / Title Block ---- */
.header-block {
    border: 2px solid #1a1a1a;
    padding: 16px 24px;
    margin-bottom: 24px;
}

.header-block h1 {
    font-size: 16pt;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
    color: #1a1a1a;
}

.header-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 4px 24px;
    font-size: 10pt;
}

.header-grid .label {
    color: #555;
    font-weight: bold;
}

"""

#: The CSS of the sections and the reference list: the package's, and
#: (scoped to ``.calc-tables``, with no fills) ``render_tables``'.
CALC_SECTIONS_CSS = """/* ---- Sections ---- */
.section {
    margin-bottom: 20px;
    page-break-inside: avoid;
}

.section h2 {
    font-size: 13pt;
    font-weight: bold;
    border-bottom: 1.5px solid #1a1a1a;
    padding-bottom: 3px;
    margin-bottom: 12px;
    text-transform: uppercase;
    letter-spacing: 0.3px;
}

/* ---- Input Table ---- */
.input-table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 12px;
    font-size: 10pt;
}

.input-table th {
    background: #f0f0f0;
    border: 1px solid #ccc;
    padding: 4px 10px;
    text-align: left;
    font-weight: bold;
}

.input-table td {
    border: 1px solid #ccc;
    padding: 4px 10px;
}

.input-table td:nth-child(3) {
    text-align: right;
    font-family: "Courier New", monospace;
    font-weight: bold;
}

.input-table td:nth-child(4) {
    text-align: left;
    color: #555;
    width: 60px;
}

/* ---- Calculation Step ---- */
.calc-step {
    margin-bottom: 16px;
    padding: 10px 14px;
    background: #fafafa;
    border-left: 3px solid #2563eb;
    page-break-inside: avoid;
}

.calc-step .step-title {
    font-weight: bold;
    font-size: 10.5pt;
    margin-bottom: 6px;
    color: #1e40af;
}

.calc-step .equation {
    font-family: "Courier New", monospace;
    font-size: 10.5pt;
    color: #333;
    margin-bottom: 4px;
    padding-left: 16px;
}

.calc-step .substitution {
    font-family: "Courier New", monospace;
    font-size: 10.5pt;
    color: #555;
    margin-bottom: 4px;
    padding-left: 16px;
}

.calc-step .result {
    font-family: "Courier New", monospace;
    font-size: 11pt;
    font-weight: bold;
    color: #1a1a1a;
    padding-left: 16px;
    margin-top: 4px;
}

.calc-step .result .computed {
    color: #2563eb;
    font-size: 11.5pt;
}

.calc-step .reference {
    font-size: 9pt;
    color: #888;
    font-style: italic;
    margin-top: 4px;
    padding-left: 16px;
}

.calc-step .notes {
    font-size: 9.5pt;
    color: #666;
    margin-top: 4px;
    padding-left: 16px;
}

/* ---- Check Items ---- */
.check-item {
    padding: 8px 14px;
    margin-bottom: 2px;
    border-radius: 4px;
    font-size: 10.5pt;
}

.check-item.pass {
    background: #f0fdf4;
    border-left: 4px solid #16a34a;
}

.check-item.fail {
    background: #fef2f2;
    border-left: 4px solid #dc2626;
}

.check-badge {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 3px;
    font-weight: bold;
    font-family: "Courier New", monospace;
    font-size: 9pt;
    margin-right: 8px;
}

.check-badge.pass {
    background: #16a34a;
    color: white;
}

.check-badge.fail {
    background: #dc2626;
    color: white;
}

.check-detail {
    font-family: "Courier New", monospace;
    font-size: 10pt;
    color: #555;
    padding-left: 28px;
    margin-top: 0;
    margin-bottom: 12px;
}

/* ---- Data Tables ---- */
.data-table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 12px;
    font-size: 9.5pt;
}

/* A table's title sits in its own block ABOVE the table: the PDF engine
   draws a table caption element below the table. */
.table-title {
    font-weight: bold;
    font-size: 10pt;
    text-align: left;
    margin-bottom: 4px;
}

.data-table th {
    background: #f0f0f0;
    border: 1px solid #ccc;
    padding: 3px 8px;
    text-align: center;
    font-weight: bold;
    font-size: 9pt;
}

.data-table td {
    border: 1px solid #ccc;
    padding: 3px 8px;
    text-align: center;
    font-family: "Courier New", monospace;
}

.table-notes {
    font-size: 8.5pt;
    color: #666;
    font-style: italic;
    margin-top: -8px;
    margin-bottom: 12px;
}

/* Tables of seven or more columns (the calculation tables, the slice
   forces) sit in a .wide-table block: smaller type and padding so they fit
   the page width. (A plain descendant selector: the PDF engine's CSS parser
   rejects :has().) */
.wide-table .data-table th,
.wide-table .data-table td {
    padding: 2px 3px;
    font-size: 7.5pt;
}

/* ---- Figures ---- */
.figure-block {
    text-align: center;
    margin: 16px 0;
    page-break-inside: avoid;
}

.figure-block img {
    border: 1px solid #ddd;
    padding: 4px;
    background: white;
}

.figure-block .caption {
    font-size: 9.5pt;
    color: #555;
    margin-top: 6px;
    font-style: italic;
}

/* ---- Text Paragraphs ---- */
.text-block {
    margin-bottom: 10px;
    font-size: 10.5pt;
}

/* ---- References ---- */
.references ol {
    padding-left: 24px;
    font-size: 9.5pt;
}

.references li {
    margin-bottom: 4px;
}

"""

_FOOTER_CSS = """/* ---- Footer ---- */
.footer {
    margin-top: 32px;
    padding-top: 8px;
    border-top: 1px solid #ccc;
    font-size: 8.5pt;
    color: #999;
    text-align: center;
}

.footer .basis {
    text-align: left;
    color: #666;
    margin: 8px 0;
    line-height: 1.4;
}

.footer .basis strong {
    color: #444;
}

/* ---- Print styles ---- */
@media print {
    body {
        padding: 0.5in 0.75in;
        max-width: none;
    }
    .section { page-break-inside: avoid; }
    .figure-block { page-break-inside: avoid; }
    .calc-step { page-break-inside: avoid; }
    .no-print { display: none; }
}
"""

CALC_PACKAGE_CSS = _PAGE_CSS + CALC_SECTIONS_CSS + _FOOTER_CSS

_PACKAGE_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ data.analysis_type }} &mdash; {{ data.project_name }}</title>
<style>
""" + CALC_PACKAGE_CSS + """
</style>
</head>
<body>

<!-- Header Block -->
<div class="header-block">
    <h1>{{ data.analysis_type }}</h1>
    <div class="header-grid">
        <div><span class="label">Project:</span> {{ data.project_name }}</div>
        <div><span class="label">Number:</span> {{ data.project_number }}</div>
        <div><span class="label">Prepared By:</span> {{ data.engineer }}</div>
        <div><span class="label">Date:</span> {{ data.date }}</div>
        {% if data.checker %}
        <div><span class="label">Checked By:</span> {{ data.checker }}</div>
        {% endif %}
        {% if data.company %}
        <div><span class="label">Company:</span> {{ data.company }}</div>
        {% endif %}
    </div>
</div>

"""

#: The sections and the reference list. ``heading`` is the headings' tag
#: and ``numbered`` numbers the sections: the package's are ``h2``,
#: numbered; ``render_tables``' sit under a heading of the report's own.
CALC_SECTIONS_TEMPLATE = """<!-- Sections -->
{% for section in sections %}
<div class="section">
    <{{ heading }}>{% if numbered %}{{ loop.index }}. {% endif %}{{ section.title }}</{{ heading }}>

    {% for item in section.items %}
    {% set itype = item_type(item) %}

    {% if itype == "input_table" %}
    <table class="input-table">
        <thead>
            <tr>
                {% for h in item.headers %}
                <th>{{ h }}</th>
                {% endfor %}
            </tr>
        </thead>
        <tbody>
            {% for row in item.rows %}
            <tr>
                {% for cell in row %}
                <td>{{ cell }}</td>
                {% endfor %}
            </tr>
            {% endfor %}
        </tbody>
    </table>

    {% elif itype == "calc_step" %}
    <div class="calc-step">
        <div class="step-title">{{ item.title }}</div>
        {# equation/substitution are trusted equation markup per the
           CalcStep contract ("HTML entities OK") — rendered raw; all
           other fields are autoescaped (CP-1). #}
        <div class="equation">{{ item.equation | safe }}</div>
        {% if item.substitution %}
        <div class="substitution">{{ item.substitution | safe }}</div>
        {% endif %}
        <div class="result">{{ item.result_name }} = <span class="computed">{{ item.result_value }}</span> {{ item.result_unit }}</div>
        {% if item.reference %}
        <div class="reference">{{ item.reference }}</div>
        {% endif %}
        {% if item.notes %}
        <div class="notes">{{ item.notes }}</div>
        {% endif %}
    </div>

    {% elif itype == "check" %}
    <div class="check-item {{ 'pass' if item.passes else 'fail' }}">
        <span class="check-badge {{ 'pass' if item.passes else 'fail' }}">{{ "PASS" if item.passes else "FAIL" }}</span>
        {{ item.description }}
    </div>
    <div class="check-detail">
        {{ item.demand_label }} = {{ item.demand }} {{ item.unit }}
        &nbsp;&nbsp;{{ ("&le;" if item.passes else "&gt;") | safe }}&nbsp;&nbsp;
        {{ item.capacity_label }} = {{ item.capacity }} {{ item.unit }}
        &nbsp;&nbsp;(D/C = {{ "%.2f"|format(item.demand / item.capacity if item.capacity != 0 else 0) }})
    </div>

    {% elif itype == "figure" %}
    <div class="figure-block">
        <img src="data:image/png;base64,{{ item.image_base64 }}"
             alt="{{ item.title }}"
             style="max-width: {{ item.width_percent }}%;">
        {% if item.caption %}
        <div class="caption">{{ item.caption }}</div>
        {% endif %}
    </div>

    {% elif itype == "table" %}
    {% if item.title %}
    <div class="table-title">{{ item.title }}</div>
    {% endif %}
    {% if item.headers|length >= 7 %}<div class="wide-table">{% endif %}
    <table class="data-table">
        <thead>
            <tr>
                {% for h in item.headers %}
                <th>{{ h }}</th>
                {% endfor %}
            </tr>
        </thead>
        <tbody>
            {% for row in item.rows %}
            <tr>
                {% for cell in row %}
                <td>{{ cell }}</td>
                {% endfor %}
            </tr>
            {% endfor %}
        </tbody>
    </table>
    {% if item.headers|length >= 7 %}</div>{% endif %}
    {% if item.notes %}
    <div class="table-notes">{{ item.notes }}</div>
    {% endif %}

    {% elif itype == "text" %}
    <div class="text-block">{{ item }}</div>

    {% endif %}
    {% endfor %}
</div>
{% endfor %}

<!-- References -->
{% if data.references %}
<div class="section references">
    <{{ heading }}>References</{{ heading }}>
    <ol>
        {% for ref in data.references %}
        <li>{{ ref }}</li>
        {% endfor %}
    </ol>
</div>
{% endif %}

"""

_PACKAGE_FOOT = """<!-- Footer -->
<div class="footer">
    Generated by GeotechStaffEngineer &mdash; {{ data.date }}
    <div class="basis">
        <strong>Basis &amp; limitations.</strong> This package applies published,
        industry-standard geotechnical methods (see the References section and each
        module's design documentation) to the inputs shown above. All quantities are
        SI. Every method has applicability limits, sign conventions, and edge cases;
        results are valid only where the selected method and the assumed subsurface
        and loading conditions apply.
    </div>
    <em>This document was generated using artificial intelligence under human
    guidance and is an analysis aid, not a design deliverable. It creates no
    engineer-of-record relationship. All inputs, method selections, and results
    must be independently reviewed by a licensed professional engineer familiar
    with the site before use. Provided AS IS, without warranty.</em>
</div>

</body>
</html>
"""

CALC_PACKAGE_TEMPLATE = _PACKAGE_HEAD + CALC_SECTIONS_TEMPLATE + _PACKAGE_FOOT

#: ``render_tables``' section headings (``h3``; the package's ``h2`` rule
#: is in :data:`CALC_SECTIONS_CSS`).
CALC_TABLES_HEADING_CSS = """
.section h3 {
    font-size: 11pt;
    font-weight: bold;
    border-bottom: 1px solid #1a1a1a;
    padding-bottom: 2px;
    margin: 14px 0 8px 0;
}
"""

#: ``render_tables``' file (BACKLOG GM-67): the sections and reference list
#: alone, inside ``.calc-tables``, for ``html_to_pdf``'s ``data-include`` to
#: place in a report. ``css``: the sections' CSS scoped to ``.calc-tables``
#: (``renderer.render_tables_html``).
CALC_TABLES_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{{ data.analysis_type }}</title>
<style>
{{ css | safe }}
</style>
</head>
<body>
<div class="calc-tables">
""" + CALC_SECTIONS_TEMPLATE + """</div>
</body>
</html>
"""
