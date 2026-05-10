"""
rule_engine.py
--------------
Applies the structured rules from rules.json to a parsed HTML table.
Returns a RuleEngineResult with:
  - which rules fired
  - confidence score
  - routing tier (HIGH / MEDIUM / LOW)
  - flags for the downstream agent
"""


import re, json
from dataclasses import dataclass, field
from bs4 import BeautifulSoup
from pathlib import Path

RULES_PATH = Path(__file__).parent.parent / "shared" / "rules.json"
LOG_PATH   = Path(__file__).parent.parent / "results" / "rule_update_log.json"


def load_rules():
    with open(RULES_PATH) as f:
        return json.load(f)


# ── feature extraction ─────────────────────────────────────────────────────────

def extract_features(html: str) -> dict:
    soup  = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if not table:
        return {"parse_error": True}

    def rows_of(tag):
        sec = table.find(tag)
        if not sec: return []
        return [[td.get_text(" ", strip=True) for td in tr.find_all(["td","th"])]
                for tr in sec.find_all("tr")]

    hr = rows_of("thead")
    br = rows_of("tbody")
    if not hr and not br:
        all_r = [[td.get_text(" ", strip=True) for td in tr.find_all(["td","th"])]
                 for tr in table.find_all("tr")]
        hr, br = ([all_r[0]] if all_r else []), (all_r[1:] if len(all_r) > 1 else [])

    all_cells = [c for r in hr+br for c in r]
    body_cells = [c for r in br for c in r]

    # spans
    spans = []
    for td in table.find_all(["td","th"]):
        try:   cs = int(td.get("colspan", 1))
        except: cs = 1
        try:   rs = int(td.get("rowspan", 1))
        except: rs = 1
        spans.append((cs, rs))

    # column consistency
    all_rows = hr + br
    counts   = [len(r) for r in all_rows] if all_rows else [0]
    mode_val = max(set(counts), key=counts.count) if counts else 0
    consistency = counts.count(mode_val) / len(counts) if counts else 0

    # duplicate rows
    seen, dupes = set(), 0
    for r in br:
        k = tuple(r)
        if k in seen: dupes += 1
        seen.add(k)
    dup_ratio = dupes / len(br) if br else 0

    # empty cells
    empty_ratio = sum(1 for c in all_cells if not c.strip()) / len(all_cells) if all_cells else 0

    # OCR artifacts
    ocr_re   = re.compile(r"\bO\d|\bl\d|\b[Il]{2,}\b|[^\x00-\x7F]{3,}")
    ocr_hits = sum(1 for c in all_cells if ocr_re.search(c))
    ocr_ratio = ocr_hits / len(all_cells) if all_cells else 0

    # numeric garble
    garble_re   = re.compile(r"\d[A-Za-z]{2,}\d")
    garble_hits = sum(1 for c in body_cells if garble_re.search(c))
    garble_ratio = garble_hits / len(body_cells) if body_cells else 0

    # whitespace dominance
    junk_re   = re.compile(r"^[\s\-–—\.·_]{1,5}$")
    nonemp    = [c for c in all_cells if c.strip()]
    junk_ratio = sum(1 for c in nonemp if junk_re.match(c)) / len(nonemp) if nonemp else 0

    # single-col collapse
    single_frac = sum(1 for r in br if len(r) == 1) / len(br) if br else 0

    # max cell length
    max_len = max((len(c) for c in all_cells), default=0)

    # span extremes
    max_cs = max((cs for cs, _ in spans), default=1)
    max_rs = max((rs for _, rs in spans), default=1)

    return {
        "num_header_rows":        len(hr),
        "num_body_rows":          len(br),
        "all_header_cells_empty": all(not c.strip() for r in hr for c in r) if hr else True,
        "modal_col_consistency":  round(consistency, 3),
        "empty_cell_ratio":       round(empty_ratio, 3),
        "duplicate_body_row_ratio": round(dup_ratio, 3),
        "ocr_artifact_ratio":     round(ocr_ratio, 3),
        "numeric_garble_ratio":   round(garble_ratio, 3),
        "whitespace_cell_ratio":  round(junk_ratio, 3),
        "single_col_row_fraction": round(single_frac, 3),
        "max_cell_length":        max_len,
        "max_colspan":            max_cs,
        "max_rowspan":            max_rs,
    }


# ── rule evaluator ─────────────────────────────────────────────────────────────

def _eval_condition(rule: dict, feat: dict) -> bool:
    """Return True if the rule's condition is triggered (i.e. rule fires = problem found)."""
    rid = rule["id"]
    c   = rule["condition"]

    if rid == "R001":
        return feat["num_header_rows"] == 0 or feat["all_header_cells_empty"]
    if rid == "R002":
        return feat["modal_col_consistency"] < float(re.search(r"[\d.]+", c).group())
    if rid == "R003":
        return feat["empty_cell_ratio"] > float(re.search(r"[\d.]+", c).group())
    if rid == "R004":
        return feat["single_col_row_fraction"] > float(re.search(r"[\d.]+", c).group())
    if rid == "R005":
        m = re.findall(r"[\d.]+", c)
        return feat["max_colspan"] > float(m[0]) or feat["max_rowspan"] > float(m[1])
    if rid == "R006":
        return feat["duplicate_body_row_ratio"] > float(re.search(r"[\d.]+", c).group())
    if rid == "R007":
        return feat["ocr_artifact_ratio"] > float(re.search(r"[\d.]+", c).group())
    if rid == "R008":
        return feat["numeric_garble_ratio"] > float(re.search(r"[\d.]+", c).group())
    if rid == "R009":
        return feat["whitespace_cell_ratio"] > float(re.search(r"[\d.]+", c).group())
    if rid == "R010":
        return feat["max_cell_length"] > float(re.search(r"[\d.]+", c).group())

    # dynamically generated rules (R011+)
    if "threshold_key" in rule and "threshold_op" in rule:
        val = feat.get(rule["threshold_key"], 0)
        thr = rule["threshold_value"]
        op  = rule["threshold_op"]
        if op == ">":  return val > thr
        if op == "<":  return val < thr
        if op == ">=": return val >= thr
        if op == "<=": return val <= thr
    return False


@dataclass
class RuleEngineResult:
    table_id:    str
    confidence:  float
    tier:        str
    fired_rules: list = field(default_factory=list)   # rule ids that triggered
    flags:       list = field(default_factory=list)
    features:    dict = field(default_factory=dict)


HIGH_THRESHOLD   = 80
MEDIUM_THRESHOLD = 55
HARD_RULE_IDS    = {"R001", "R004", "R005"}   # grows as loop adds rules


def evaluate(html: str, table_id: str = "?", rules_data: dict = None) -> RuleEngineResult:
    if rules_data is None:
        rules_data = load_rules()

    active_rules = [r for r in rules_data["rules"] if r["status"] == "active"]
    feat         = extract_features(html)

    if feat.get("parse_error"):
        return RuleEngineResult(table_id, 0, "LOW", ["PARSE_ERROR"], ["parse_error"])

    fired, flags = [], []
    total_penalty = 0
    hard_fired    = False

    for rule in active_rules:
        if _eval_condition(rule, feat):
            fired.append(rule["id"])
            flags.extend(rule["action"].split(", "))
            total_penalty += rule["confidence_penalty"]
            if rule["id"] in HARD_RULE_IDS:
                hard_fired = True

    confidence = max(0, 100 - total_penalty)
    if hard_fired:
        confidence = min(confidence, MEDIUM_THRESHOLD - 1)

    tier = ("HIGH"   if confidence >= HIGH_THRESHOLD   else
            "MEDIUM" if confidence >= MEDIUM_THRESHOLD else
            "LOW")

    return RuleEngineResult(
        table_id   = table_id,
        confidence = confidence,
        tier       = tier,
        fired_rules = fired,
        flags      = list(set(flags)),
        features   = feat,
    )