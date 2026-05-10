"""
synthetic_tables.py
-------------------
Synthetic medical table generator + PubTabNet loader.
Used by rule_updater.py for the iterative rule refinement loop.

Place in: notebooks/synthetic_tables.py
"""

import json
import random
import re
from bs4 import BeautifulSoup

random.seed(42)


# ── value generators ───────────────────────────────────────────────────────────

def _rand_val(vtype="mixed"):
    if vtype == "num":
        choices = [
            f"{random.uniform(0.1, 99.9):.1f}",
            f"{random.randint(1, 500)}",
            f"{random.uniform(0,1):.3f}",
            f"{random.uniform(10,200):.1f} ({random.uniform(5,30):.1f})",
            f"{random.randint(1,200)}/{random.randint(200,500)} ({random.uniform(10,90):.1f}%)",
        ]
    elif vtype == "text":
        choices = [
            "Placebo", "Treatment", "Intervention", "Control",
            "Low dose", "High dose", "Total", "p value",
            "Relative risk", "Odds ratio", "Hazard ratio",
        ]
    else:
        choices = [
            f"{random.uniform(0.1,99):.1f}", "Yes", "No",
            f"{random.randint(1,500)}", "N/A", "--",
            f"{random.uniform(0.001,0.999):.3f}",
        ]
    return random.choice(choices)


# ── clean table makers ─────────────────────────────────────────────────────────

def make_simple_table(n_rows=6, n_cols=4) -> str:
    headers = ["Characteristic"] + [f"Group {i+1}" for i in range(n_cols - 1)]
    row_labels = ["Age, mean (SD)", "Male, n (%)", "BMI", "Duration (weeks)",
                  "Primary endpoint", "p-value", "HR (95% CI)", "Adverse events"]
    rows = []
    for i in range(min(n_rows, len(row_labels))):
        rows.append([row_labels[i]] + [_rand_val("num") for _ in range(n_cols - 1)])
    thead = "<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>"
    tbody = "<tbody>" + "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows
    ) + "</tbody>"
    return f"<table>{thead}{tbody}</table>"


def make_complex_table() -> str:
    return """<table>
      <thead>
        <tr>
          <th rowspan="2">Variable</th>
          <th colspan="2">Treatment (n=120)</th>
          <th colspan="2">Control (n=118)</th>
          <th rowspan="2">p-value</th>
        </tr>
        <tr><th>n (%)</th><th>Mean (SD)</th><th>n (%)</th><th>Mean (SD)</th></tr>
      </thead>
      <tbody>
        <tr><td>Age</td><td>—</td><td>54.3 (8.1)</td><td>—</td><td>52.9 (7.6)</td><td>0.18</td></tr>
        <tr><td>Male</td><td>71 (59.2%)</td><td>—</td><td>68 (57.6%)</td><td>—</td><td>0.83</td></tr>
        <tr><td>LDL-C (mmol/L)</td><td>—</td><td>3.82 (0.91)</td><td>—</td><td>3.79 (0.88)</td><td>0.76</td></tr>
        <tr><td>Primary endpoint</td><td>12 (10.0%)</td><td>—</td><td>22 (18.6%)</td><td>—</td><td>0.047</td></tr>
      </tbody>
    </table>"""


def make_adverse_events_table() -> str:
    events = [
        ("Any AE", "98 (81.7%)", "91 (77.1%)"),
        ("Serious AE", "14 (11.7%)", "12 (10.2%)"),
        ("Nausea", "23 (19.2%)", "8 (6.8%)"),
        ("Headache", "18 (15.0%)", "15 (12.7%)"),
        ("Dizziness", "11 (9.2%)", "9 (7.6%)"),
        ("Discontinued due to AE", "6 (5.0%)", "4 (3.4%)"),
    ]
    thead = "<thead><tr><th>Event</th><th>Treatment n=120 (%)</th><th>Placebo n=118 (%)</th></tr></thead>"
    rows  = "".join(f"<tr><td>{e}</td><td>{t}</td><td>{p}</td></tr>" for e, t, p in events)
    footnote = "<tr><td colspan='3'><i>AE = adverse event. Data shown as n (%).</i></td></tr>"
    return f"<table>{thead}<tbody>{rows}{footnote}</tbody></table>"


def make_dose_response_table() -> str:
    doses = ["5-10", "10-20", "20-40", "40-80", ">80"]
    thead = "<thead><tr><th>Dose (mg/day)</th><th>n</th><th>Response (%)</th><th>≥40% reduction</th></tr></thead>"
    rows  = "".join(
        f"<tr><td>{d}</td><td>{random.randint(20,80)}</td>"
        f"<td>{random.uniform(20,80):.1f}</td>"
        f"<td>{random.randint(5,50)} ({random.uniform(10,60):.1f}%)</td></tr>"
        for d in doses
    )
    return f"<table>{thead}<tbody>{rows}</tbody></table>"


# ── corruption functions ───────────────────────────────────────────────────────

def corrupt_ocr(html: str) -> str:
    return html.replace("1", "l").replace("0", "O").replace(".", "·")


def corrupt_missing_header(html: str) -> str:
    soup  = BeautifulSoup(html, "lxml")
    thead = soup.find("thead")
    if thead:
        thead.decompose()
    return str(soup)


def corrupt_empty_cells(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for td in soup.find_all(["td", "th"]):
        if random.random() < 0.60:
            td.string = ""
    return str(soup)


def corrupt_duplicate_rows(html: str) -> str:
    soup  = BeautifulSoup(html, "lxml")
    tbody = soup.find("tbody")
    if tbody:
        for row in list(tbody.find_all("tr")):
            for _ in range(2):
                clone = BeautifulSoup(str(row), "lxml").find("tr")
                if clone:
                    tbody.append(clone)
    return str(soup)


def corrupt_merged_cell_overflow(html: str) -> str:
    return html.replace("<td>", '<td colspan="99">', 1)


def corrupt_single_col_collapse(html: str) -> str:
    soup  = BeautifulSoup(html, "lxml")
    tbody = soup.find("tbody")
    if tbody:
        new_tbody = soup.new_tag("tbody")
        for tr in tbody.find_all("tr"):
            text   = " ".join(td.get_text(" ", strip=True) for td in tr.find_all(["td","th"]))
            new_tr = soup.new_tag("tr")
            new_td = soup.new_tag("td")
            new_td.string = text
            new_tr.append(new_td)
            new_tbody.append(new_tr)
        tbody.replace_with(new_tbody)
    return str(soup)


def corrupt_numeric_garble(html: str) -> str:
    return re.sub(r"(\d{2,})", lambda m: m.group(0)[:1] + "xyz" + m.group(0)[1:], html, count=5)


# ── dataset factory ────────────────────────────────────────────────────────────

CLEAN_MAKERS = [
    make_simple_table,
    make_complex_table,
    make_adverse_events_table,
    make_dose_response_table,
]

CORRUPTIONS = [
    ("ocr_artifacts",       corrupt_ocr),
    ("header_present",      corrupt_missing_header),
    ("empty_cell_ratio",    corrupt_empty_cells),
    ("duplicate_rows",      corrupt_duplicate_rows),
    ("span_overflow",       corrupt_merged_cell_overflow),
    ("single_col_collapse", corrupt_single_col_collapse),
    ("numeric_garble",      corrupt_numeric_garble),
]


def generate_dataset(n_tables=500):
    """
    60% clean, 30% single-corruption, 10% multi-corruption.
    Returns list of dicts: {id, html, label, injected}
    """
    tables   = []
    n_clean  = int(n_tables * 0.60)
    n_single = int(n_tables * 0.30)
    n_multi  = n_tables - n_clean - n_single

    for i in range(n_clean):
        maker = CLEAN_MAKERS[i % len(CLEAN_MAKERS)]
        html  = maker(n_rows=random.randint(4,10), n_cols=random.randint(3,6)) \
                if maker == make_simple_table else maker()
        tables.append({"id": f"clean_{i:04d}", "html": html, "label": "clean", "injected": []})

    for i in range(n_single):
        maker = random.choice(CLEAN_MAKERS)
        html  = maker(n_rows=random.randint(4,10), n_cols=random.randint(3,6)) \
                if maker == make_simple_table else maker()
        name, fn = random.choice(CORRUPTIONS)
        try:    html = fn(html)
        except: pass
        tables.append({"id": f"corrupt1_{i:04d}", "html": html, "label": "corrupted", "injected": [name]})

    for i in range(n_multi):
        maker = random.choice(CLEAN_MAKERS)
        html  = maker()
        chosen, injected = random.sample(CORRUPTIONS, k=random.randint(2,3)), []
        for name, fn in chosen:
            try:    html = fn(html); injected.append(name)
            except: pass
        tables.append({"id": f"corrupt2_{i:04d}", "html": html, "label": "corrupted", "injected": injected})

    random.shuffle(tables)
    return tables


# ── real PubTabNet loader ──────────────────────────────────────────────────────

def load_pubtabnet(path: str, max_tables=5000):
    """Load PubTabNet_2.0.0.jsonl, reconstruct HTML from structure tokens."""
    tables = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_tables:
                break
            try:
                row  = json.loads(line)
                html = _reconstruct_html(row)
                if html:
                    tables.append({
                        "id":       row.get("filename", f"ptn_{i:06d}"),
                        "html":     html,
                        "label":    "clean",
                        "injected": [],
                    })
            except Exception:
                continue
    return tables


def _reconstruct_html(row: dict) -> str:
    try:
        tokens   = row["html"]["structure"]["tokens"]
        cells    = row["html"]["cell"]
        cell_idx = 0
        parts    = ["<table>"]
        for tok in tokens:
            if tok in ("<td>", "<th>") or tok.startswith("<td ") or tok.startswith("<th "):
                parts.append(tok)
                if cell_idx < len(cells):
                    c = cells[cell_idx]
                    parts.append(c.get("tokens", [""])[0] if isinstance(c, dict) else str(c))
                    cell_idx += 1
            else:
                parts.append(tok)
        parts.append("</table>")
        return "".join(parts)
    except Exception:
        return ""