"""
HTML template for calculation packages.

Single-file Jinja2 template with embedded CSS.
Designed for professional engineering output with print-friendly layout.
"""

CALC_PACKAGE_CSS = """
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

/* ---- Sections ---- */
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

.data-table caption {
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

/* ---- Footer ---- */
.footer {
    margin-top: 32px;
    padding-top: 8px;
    border-top: 1px solid #ccc;
    font-size: 8.5pt;
    color: #999;
    text-align: center;
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

CALC_PACKAGE_TEMPLATE = """<!DOCTYPE html>
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

<!-- Sections -->
{% for section in sections %}
<div class="section">
    <h2>{{ loop.index }}. {{ section.title }}</h2>

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
        <div class="equation">{{ item.equation }}</div>
        {% if item.substitution %}
        <div class="substitution">{{ item.substitution }}</div>
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
        &nbsp;&nbsp;{{ "&le;" if item.passes else "&gt;" }}&nbsp;&nbsp;
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
    <table class="data-table">
        {% if item.title %}
        <caption>{{ item.title }}</caption>
        {% endif %}
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
    <h2>References</h2>
    <ol>
        {% for ref in data.references %}
        <li>{{ ref }}</li>
        {% endfor %}
    </ol>
</div>
{% endif %}

<!-- Footer -->
<div class="footer">
    Generated by GeotechStaffEngineer &mdash; {{ data.date }}
</div>

</body>
</html>
"""
