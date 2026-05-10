"""
table_validator.py
------------------
10 structural checks on HTML table output.
Scoring model: each check has a weight; failed checks subtract that
weight from 100. Any HARD check failure immediately caps the tier.
"""

import re
from dataclasses import dataclass, field
from bs4 import BeautifulSoup

# ── thresholds (tuned by rule_updater) ────────────────────────────────────────
THRESHOLDS = {
    "max_empty_cell_ratio":       0.55,
    "max_whitespace_cell_ratio":  0.45,
    "min_col_consistency":        0.65,
    "max_ocr_artifact_ratio":     0.05,   # tightened
    "min_header_rows":            1,
    "max_duplicate_row_ratio":    0.20,   # tightened
    "max_single_col_fraction":    0.25,
    "max_outlier_cell_length":    800,
    "max_colspan_absolute":       15,     # NEW: absolute, not ratio
    "max_rowspan_absolute":       20,     # NEW
    "max_numeric_garble_ratio":   0.08,   # tightened
}

HIGH_THRESHOLD   = 80
MEDIUM_THRESHOLD = 55

# Check weights — must sum to 100
CHECK_WEIGHTS = {
    "column_consistency":    12,
    "header_present":        15,   # HARD — missing header = big problem
    "empty_cell_ratio":      10,
    "span_overflow":         12,   # HARD — colspan=99 is catastrophic
    "duplicate_rows":        10,
    "ocr_artifacts":          8,
    "single_col_collapse":   12,   # HARD — collapsed table is useless
    "whitespace_dominance":   8,
    "cell_length_outliers":   8,
    "numeric_garble":         5,
}

# These checks failing immediately drop tier to MEDIUM at best
HARD_CHECKS = {
    "header_present", "span_overflow", "single_col_collapse",
    "duplicate_rows", "ocr_artifacts", "numeric_garble", "empty_cell_ratio",
}


@dataclass
class CheckResult:
    name:   str
    passed: bool
    detail: str = ""


@dataclass
class ValidationResult:
    table_id:      str
    confidence:    float
    tier:          str
    checks:        list = field(default_factory=list)
    failure_modes: list = field(default_factory=list)


# ── parsing helpers ────────────────────────────────────────────────────────────

def _parse_rows(html):
    soup  = BeautifulSoup(html, "lxml")
    table = soup.find("table")
    if not table:
        return [], []

    def rows_from(tag):
        sec = table.find(tag)
        if not sec:
            return []
        return [[td.get_text(" ", strip=True) for td in tr.find_all(["td","th"])]
                for tr in sec.find_all("tr")]

    hr = rows_from("thead")
    br = rows_from("tbody")

    if not hr and not br:
        all_rows = [[td.get_text(" ", strip=True) for td in tr.find_all(["td","th"])]
                    for tr in table.find_all("tr")]
        if all_rows:
            hr, br = [all_rows[0]], all_rows[1:]

    return hr, br


def _get_spans(html):
    soup = BeautifulSoup(html, "lxml")
    spans = []
    for td in soup.find_all(["td","th"]):
        try:   cs = int(td.get("colspan", 1))
        except: cs = 1
        try:   rs = int(td.get("rowspan", 1))
        except: rs = 1
        spans.append((cs, rs))
    return spans


# ── checks ─────────────────────────────────────────────────────────────────────

def check_column_consistency(hr, br):
    all_rows = hr + br
    if not all_rows:
        return CheckResult("column_consistency", False, "no rows")
    counts   = [len(r) for r in all_rows]
    mode_val = max(set(counts), key=counts.count)
    frac     = counts.count(mode_val) / len(counts)
    passed   = frac >= THRESHOLDS["min_col_consistency"]
    return CheckResult("column_consistency", passed,
                       f"modal col={mode_val}, consistency={frac:.2f}")


def check_header_present(hr):
    ok = (len(hr) >= THRESHOLDS["min_header_rows"]
          and any(any(c.strip() for c in r) for r in hr))
    return CheckResult("header_present", ok, f"{len(hr)} header rows")


def check_empty_cell_ratio(hr, br):
    cells = [c for r in hr+br for c in r]
    if not cells:
        return CheckResult("empty_cell_ratio", False, "no cells")
    ratio = sum(1 for c in cells if not c.strip()) / len(cells)
    return CheckResult("empty_cell_ratio",
                       ratio <= THRESHOLDS["max_empty_cell_ratio"],
                       f"empty={ratio:.2f}")


def check_span_overflow(html):
    """Absolute check: any single cell with unrealistic colspan/rowspan fails."""
    spans = _get_spans(html)
    if not spans:
        return CheckResult("span_overflow", True, "no cells")
    bad = [(cs,rs) for cs,rs in spans
           if cs > THRESHOLDS["max_colspan_absolute"]
           or rs > THRESHOLDS["max_rowspan_absolute"]]
    passed = len(bad) == 0
    return CheckResult("span_overflow", passed,
                       f"bad spans: {bad[:3]}" if bad else "ok")


def check_duplicate_rows(br):
    if len(br) < 2:
        return CheckResult("duplicate_rows", True, "too few rows")
    seen, dupes = set(), 0
    for r in br:
        k = tuple(r)
        if k in seen: dupes += 1
        seen.add(k)
    ratio = dupes / len(br)
    return CheckResult("duplicate_rows",
                       ratio <= THRESHOLDS["max_duplicate_row_ratio"],
                       f"dup ratio={ratio:.2f}")


_OCR_RE = re.compile(
    r"\bO\d"          # O instead of 0
    r"|\bl\d"         # l instead of 1
    r"|\b[Il]{2,}\b"  # repeated I/l
    r"|[^\x00-\x7F]{3,}"  # mojibake run
)

def check_ocr_artifacts(hr, br):
    cells = [c for r in hr+br for c in r]
    if not cells:
        return CheckResult("ocr_artifacts", True, "no cells")
    hits  = sum(1 for c in cells if _OCR_RE.search(c))
    ratio = hits / len(cells)
    return CheckResult("ocr_artifacts",
                       ratio <= THRESHOLDS["max_ocr_artifact_ratio"],
                       f"artifact ratio={ratio:.3f}")


def check_single_col_collapse(br):
    if not br:
        return CheckResult("single_col_collapse", True, "no body rows")
    single = sum(1 for r in br if len(r) == 1)
    frac   = single / len(br)
    return CheckResult("single_col_collapse",
                       frac <= THRESHOLDS["max_single_col_fraction"],
                       f"single-col frac={frac:.2f}")


_JUNK_RE = re.compile(r"^[\s\-–—\.·_]{1,5}$")

def check_whitespace_dominance(hr, br):
    cells = [c for r in hr+br for c in r if c.strip()]
    if not cells:
        return CheckResult("whitespace_dominance", True, "all empty")
    junk  = sum(1 for c in cells if _JUNK_RE.match(c))
    ratio = junk / len(cells)
    return CheckResult("whitespace_dominance",
                       ratio <= THRESHOLDS["max_whitespace_cell_ratio"],
                       f"junk ratio={ratio:.2f}")


def check_cell_length_outliers(hr, br):
    cells = [c for r in hr+br for c in r]
    if not cells:
        return CheckResult("cell_length_outliers", True, "no cells")
    max_len = max(len(c) for c in cells)
    return CheckResult("cell_length_outliers",
                       max_len <= THRESHOLDS["max_outlier_cell_length"],
                       f"max_len={max_len}")


_GARBLE_RE = re.compile(r"\d[A-Za-z]{2,}\d")

def check_numeric_garble(br):
    cells = [c for r in br for c in r if c.strip()]
    if not cells:
        return CheckResult("numeric_garble", True, "no body cells")
    hits  = sum(1 for c in cells if _GARBLE_RE.search(c))
    ratio = hits / len(cells)
    return CheckResult("numeric_garble",
                       ratio <= THRESHOLDS["max_numeric_garble_ratio"],
                       f"garble ratio={ratio:.3f}")


# ── main validate ──────────────────────────────────────────────────────────────

def validate_table(html: str, table_id: str = "unknown") -> ValidationResult:
    hr, br = _parse_rows(html)

    checks = [
        check_column_consistency(hr, br),
        check_header_present(hr),
        check_empty_cell_ratio(hr, br),
        check_span_overflow(html),
        check_duplicate_rows(br),
        check_ocr_artifacts(hr, br),
        check_single_col_collapse(br),
        check_whitespace_dominance(hr, br),
        check_cell_length_outliers(hr, br),
        check_numeric_garble(br),
    ]

    # confidence = 100 minus weight of every failed check
    failed_weight = sum(
        CHECK_WEIGHTS[c.name] for c in checks if not c.passed
    )
    confidence = max(0, 100 - failed_weight)

    # hard check failure caps at MEDIUM
    hard_failed = any(
        not c.passed for c in checks if c.name in HARD_CHECKS
    )

    if hard_failed:
        cap = MEDIUM_THRESHOLD - 1   # force MEDIUM or LOW
        confidence = min(confidence, cap)

    tier = ("HIGH"   if confidence >= HIGH_THRESHOLD   else
            "MEDIUM" if confidence >= MEDIUM_THRESHOLD else
            "LOW")

    return ValidationResult(
        table_id      = table_id,
        confidence    = confidence,
        tier          = tier,
        checks        = checks,
        failure_modes = [c.name for c in checks if not c.passed],
    )