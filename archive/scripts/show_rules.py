"""
show_rules.py
-------------
Prints a clean, readable list of all active rules.
This is the living document that grows as rule_updater.py runs.

Usage:
    python show_rules.py               # print all active rules
    python show_rules.py --history     # include full update history from log
    python show_rules.py --md          # markdown output (for docs / Notion)
"""

import json, argparse, textwrap
from pathlib import Path

RULES_PATH = Path("rules.json")
LOG_PATH   = Path("results/rule_update_log.json")

CATEGORY_LABELS = {
    "structure": "STRUCTURAL",
    "content":   "CONTENT QUALITY",
    "learned":   "LEARNED (from PubTabNet)",
}

ACTION_ICONS = {
    "HIGH":   "✓",
    "MEDIUM": "~",
    "LOW":    "✗",
}


def load_history():
    if not LOG_PATH.exists():
        return {}
    log  = json.loads(LOG_PATH.read_text())
    hist = {}   # rule_id → list of changes
    for it in log.get("iterations", []):
        for change in it.get("rule_changes", []):
            rid = change.get("rule_id")
            if rid:
                hist.setdefault(rid, []).append({
                    "iteration": it["iteration"],
                    **change,
                })
    return hist


def print_rules(rules_data: dict, show_history: bool, md: bool):
    rules   = [r for r in rules_data["rules"] if r["status"] == "active"]
    version = rules_data.get("version", 1)
    updated = rules_data.get("last_updated", "")[:10]
    history = load_history() if show_history else {}

    # group by category
    from collections import defaultdict
    by_cat = defaultdict(list)
    for r in rules:
        by_cat[r["category"]].append(r)

    if md:
        _print_md(rules, version, updated, history, by_cat)
    else:
        _print_terminal(rules, version, updated, history, by_cat)


def _print_terminal(rules, version, updated, history, by_cat):
    total = len(rules)
    print()
    print(f"  TABLE EXTRACTION RULES  (v{version}  ·  {total} active rules  ·  updated {updated})")
    print(f"  {'═'*68}")

    for cat in ["structure", "content", "learned"]:
        cat_rules = by_cat.get(cat, [])
        if not cat_rules:
            continue
        label = CATEGORY_LABELS.get(cat, cat.upper())
        print(f"\n  ── {label} {'─'*(62-len(label))}")

        for r in cat_rules:
            source_tag = f"[{r['source']}]" if r.get("source","initial") != "initial" else ""
            print(f"\n  {r['id']}  {r['name'].upper()}  {source_tag}")
            # wrap description at 65 chars
            desc_lines = textwrap.wrap(r["description"], width=64)
            for line in desc_lines:
                print(f"      {line}")
            print(f"      condition : {r['condition']}")
            print(f"      action    : {r['action']}")
            print(f"      penalty   : -{r['confidence_penalty']} pts confidence")
            # table types
            if r.get("table_types") != ["all"]:
                print(f"      applies to: {', '.join(r['table_types'])}")

            # history
            if history and r["id"] in history:
                print(f"      history   :")
                for h in history[r["id"]]:
                    chg = h.get("change","")
                    if chg == "threshold_tightened":
                        print(f"        iter {h['iteration']}: tightened  {h['condition_before']}  →  {h['condition_after']}")
                        print(f"                  reason: {h['reason'][:80]}")
                    elif chg == "threshold_relaxed":
                        print(f"        iter {h['iteration']}: relaxed    {h['condition_before']}  →  {h['condition_after']}")
                    elif chg == "penalty_increased":
                        print(f"        iter {h['iteration']}: penalty    {h['old_penalty']} → {h['new_penalty']}  ({h['reason'][:60]})")
                    elif chg == "promoted_to_hard_check":
                        print(f"        iter {h['iteration']}: → HARD CHECK  ({h['reason'][:70]})")
                    elif chg == "new_rule_added":
                        print(f"        iter {h['iteration']}: ADDED  condition={h['condition']}")
                        print(f"                  reason: {h['reason'][:80]}")

    print(f"\n  {'─'*68}")
    # routing summary
    print(f"\n  ROUTING LOGIC")
    print(f"  ─────────────")
    print(f"  confidence = 100 − Σ(penalty for each fired rule)")
    print(f"  Any HARD rule firing caps confidence < 55 regardless of score")
    print()
    print(f"  ≥ 80   HIGH    → accept extraction, pass to RAG pipeline")
    print(f"  55–79  MEDIUM  → retry with Gemini fallback, take best result")
    print(f"  < 55   LOW     → Gemini required; if still LOW → human review queue")
    print()
    print(f"  HARD checks (single failure = instant LOW/MEDIUM cap):")
    try:
        from rule_engine import HARD_RULE_IDS
        hard_names = {r["id"]: r["name"] for r in rules}
        for rid in sorted(HARD_RULE_IDS):
            name = hard_names.get(rid, rid)
            print(f"    {rid}  {name}")
    except ImportError:
        pass
    print()


def _print_md(rules, version, updated, history, by_cat):
    print(f"# Table Extraction Rules  (v{version})")
    print(f"*{len(rules)} active rules · updated {updated}*\n")

    for cat in ["structure", "content", "learned"]:
        cat_rules = by_cat.get(cat, [])
        if not cat_rules:
            continue
        print(f"## {CATEGORY_LABELS.get(cat, cat.upper())}\n")
        for r in cat_rules:
            src = f" _{r['source']}_" if r.get("source","initial") != "initial" else ""
            print(f"### {r['id']} — `{r['name']}`{src}\n")
            print(f"{r['description']}\n")
            print(f"| Field | Value |")
            print(f"|-------|-------|")
            print(f"| Condition | `{r['condition']}` |")
            print(f"| Action | `{r['action']}` |")
            print(f"| Penalty | -{r['confidence_penalty']} pts |")
            if history and r["id"] in history:
                changes = history[r["id"]]
                hist_lines = []
                for h in changes:
                    if h.get("change") == "threshold_tightened":
                        hist_lines.append(f"iter {h['iteration']}: `{h['condition_before']}` → `{h['condition_after']}`")
                    elif h.get("change") == "promoted_to_hard_check":
                        hist_lines.append(f"iter {h['iteration']}: promoted to HARD")
                    elif h.get("change") == "new_rule_added":
                        hist_lines.append(f"iter {h['iteration']}: **added**")
                if hist_lines:
                    print(f"| History | {' · '.join(hist_lines)} |")
            print()

    print("## Routing Logic\n")
    print("```")
    print("confidence = 100 − Σ(penalty for each fired rule)")
    print("")
    print("≥ 80   HIGH    → accept, pass to RAG pipeline")
    print("55–79  MEDIUM  → retry with Gemini, take best result")
    print("< 55   LOW     → Gemini required; if still LOW → human review")
    print("```")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", action="store_true", help="Show rule change history")
    parser.add_argument("--md",      action="store_true", help="Markdown output")
    args = parser.parse_args()

    if not RULES_PATH.exists():
        print("rules.json not found. Run rule_updater.py first.")
    else:
        rules_data = json.loads(RULES_PATH.read_text())
        print_rules(rules_data, args.history, args.md)