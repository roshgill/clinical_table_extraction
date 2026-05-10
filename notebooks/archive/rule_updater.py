"""
rule_updater.py
---------------
Iterative rule refinement loop on PubTabNet (or synthetic data).

Each iteration:
  1. Run rule_engine on all tables
  2. Find false negatives (corrupted tables that passed as HIGH)
  3. Find false positives (clean tables flagged incorrectly)
  4. Generate new rules or tighten existing ones to fix failures
  5. Log every rule change with full rationale
  6. Write updated rules.json
  7. Repeat until convergence or max_iterations

Usage:
    python rule_updater.py --demo --n-tables 600 --iterations 4
    python rule_updater.py --data-path /path/to/PubTabNet_2.0.0.jsonl --iterations 3
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json, re, copy, argparse, datetime
from pathlib import Path
from collections import Counter, defaultdict

from shared.rule_engine import evaluate, extract_features, HARD_RULE_IDS
from synthetic_tables import generate_dataset, load_pubtabnet

RULES_PATH = Path(__file__).parent.parent / "shared" / "rules.json"
LOG_PATH    = Path("results/rule_update_log.json")


# ── run engine on all tables ───────────────────────────────────────────────────

def run_all(tables, rules_data):
    results = []
    for t in tables:
        r = evaluate(t["html"], t["id"], rules_data)
        results.append({
            "id":        r.table_id,
            "label":     t["label"],
            "injected":  t.get("injected", []),
            "tier":      r.tier,
            "confidence": r.confidence,
            "fired":     r.fired_rules,
            "flags":     r.flags,
            "features":  r.features,
        })
    return results


def compute_metrics(results):
    tp = fp = tn = fn = 0
    for r in results:
        clean   = r["label"] == "clean"
        is_high = r["tier"] == "HIGH"
        if clean and is_high:      tn += 1
        elif clean:                fp += 1
        elif not is_high:          tp += 1
        else:                      fn += 1
    total = len(results)
    prec  = tp/(tp+fp) if (tp+fp) else 0
    rec   = tp/(tp+fn) if (tp+fn) else 0
    f1    = 2*prec*rec/(prec+rec) if (prec+rec) else 0
    acc   = (tp+tn)/total if total else 0
    return {"accuracy": round(acc*100,1), "f1": round(f1*100,1),
            "precision": round(prec*100,1), "recall": round(rec*100,1),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": total}


# ── failure analysis ───────────────────────────────────────────────────────────

def get_false_negatives(results):
    """Corrupted tables that passed as HIGH — what did we miss and why."""
    fn = [r for r in results if r["label"] == "corrupted" and r["tier"] == "HIGH"]
    injected_counts = Counter(inj for r in fn for inj in r["injected"])
    # per injection: what feature values did the FN tables have?
    feature_by_injection = defaultdict(list)
    for r in fn:
        for inj in r["injected"]:
            feature_by_injection[inj].append(r["features"])
    return fn, injected_counts, feature_by_injection


def get_false_positives(results):
    """Clean tables that got flagged — which rules are over-firing."""
    fp = [r for r in results if r["label"] == "clean" and r["tier"] != "HIGH"]
    rule_counts = Counter(rid for r in fp for rid in r["fired"])
    return fp, rule_counts


# ── rule update logic ─────────────────────────────────────────────────────────

INJECTION_TO_FEATURE = {
    "ocr_artifacts":      ("ocr_artifact_ratio",      "R007", ">", -0.03),
    "numeric_garble":     ("numeric_garble_ratio",    "R008", ">", -0.04),
    "duplicate_rows":     ("duplicate_body_row_ratio","R006", ">", -0.08),
    "empty_cell_ratio":   ("empty_cell_ratio",        "R003", ">", -0.08),
    "span_overflow":      ("max_colspan",             "R005", ">", -8),
    "single_col_collapse":("single_col_row_fraction", "R004", ">", -0.10),
    "header_present":     (None,                      "R001", None, None),
}

FP_RULE_RELAX = {
    "R002": ("modal_col_consistency", "<",  +0.05),
    "R003": ("empty_cell_ratio",      ">",  +0.08),
    "R005": ("max_colspan",           ">",  +5),
    "R009": ("whitespace_cell_ratio", ">",  +0.08),
    "R010": ("max_cell_length",       ">",  +100),
}

NEXT_RULE_ID = {"counter": 11}   # auto-increment for new rules


def get_threshold_from_condition(condition_str: str):
    """Parse 'some_feature > 0.05' → ('some_feature', '>', 0.05)."""
    m = re.match(r"(\w+)\s*([<>]=?)\s*([\d.]+)", condition_str)
    if m:
        return m.group(1), m.group(2), float(m.group(3))
    return None, None, None


def update_rule_threshold(rule: dict, delta: float, reason: str, change_log: list):
    """Tighten or relax a threshold value embedded in rule['condition']."""
    feat_key, op, old_val = get_threshold_from_condition(rule["condition"])
    if feat_key is None:
        return
    new_val = round(old_val + delta, 4)
    old_cond = rule["condition"]
    rule["condition"] = re.sub(r"[\d.]+", str(new_val), rule["condition"], count=1)
    change_log.append({
        "rule_id":    rule["id"],
        "rule_name":  rule["name"],
        "change":     "threshold_tightened" if delta < 0 else "threshold_relaxed",
        "condition_before": old_cond,
        "condition_after":  rule["condition"],
        "delta":      delta,
        "reason":     reason,
    })
    print(f"    [{rule['id']}] {rule['name']}: {old_cond} → {rule['condition']}")
    print(f"          reason: {reason}")


def add_new_rule(rules_data: dict, feature_key: str, op: str, threshold: float,
                 name: str, description: str, action: str, penalty: int,
                 source: str, change_log: list):
    """Create a brand-new rule and append it."""
    rid = f"R{NEXT_RULE_ID['counter']:03d}"
    NEXT_RULE_ID['counter'] += 1
    new_rule = {
        "id":                rid,
        "name":              name,
        "category":          "learned",
        "description":       description,
        "condition":         f"{feature_key} {op} {threshold}",
        "action":            action,
        "confidence_penalty": penalty,
        "table_types":       ["all"],
        "status":            "active",
        "source":            source,
        # for dynamic evaluation
        "threshold_key":     feature_key,
        "threshold_op":      op,
        "threshold_value":   threshold,
    }
    rules_data["rules"].append(new_rule)
    change_log.append({
        "rule_id":    rid,
        "rule_name":  name,
        "change":     "new_rule_added",
        "condition":  new_rule["condition"],
        "reason":     source,
    })
    print(f"    [NEW {rid}] {name}: {feature_key} {op} {threshold}")
    print(f"          reason: {source}")
    return rid


def promote_to_hard(rule_id: str, reason: str, change_log: list):
    HARD_RULE_IDS.add(rule_id)
    change_log.append({
        "rule_id":   rule_id,
        "change":    "promoted_to_hard_check",
        "reason":    reason,
    })
    print(f"    [{rule_id}] promoted to HARD check — {reason}")



def _patch_ineffective_fires(results, rules_data, change_log, n_corrupt):
    """
    If a rule fires on corrupted tables but those tables still pass as HIGH,
    the penalty is too low — increase it or promote to hard.
    """
    rules_by_id = {r["id"]: r for r in rules_data["rules"]}
    ineffective = Counter()
    for r in results:
        if r["label"] == "corrupted" and r["tier"] == "HIGH":
            for fired_id in r["fired"]:
                ineffective[fired_id] += 1

    threshold = max(1, int(n_corrupt * 0.06))
    for rule_id, count in ineffective.most_common():
        if count < threshold:
            continue
        rule = rules_by_id.get(rule_id)
        if not rule:
            continue
        if rule_id not in HARD_RULE_IDS:
            old_pen = rule["confidence_penalty"]
            new_pen = min(30, old_pen + 5)
            if new_pen != old_pen:
                rule["confidence_penalty"] = new_pen
                msg = (f"Rule fires on {count} corrupted tables but confidence stays HIGH "
                       f"(penalty {old_pen} insufficient). Increasing penalty to {new_pen}.")
                change_log.append({"rule_id": rule_id, "rule_name": rule["name"],
                    "change": "penalty_increased", "old_penalty": old_pen,
                    "new_penalty": new_pen, "reason": msg})
                print(f"    [{rule_id}] {rule['name']}: penalty {old_pen} → {new_pen}")
                print(f"          reason: {msg}")
            if count / n_corrupt >= 0.10:
                promote_to_hard(rule_id,
                    f"Fires on {count} corrupted tables ({count/n_corrupt*100:.0f}%) "
                    f"but still passing as HIGH — promoting to hard check.", change_log)



def run_one_iteration(tables, rules_data, iteration: int,
                      n_clean: int, n_corrupt: int):
    results  = run_all(tables, rules_data)
    metrics  = compute_metrics(results)
    fn_list, fn_counts, fn_features = get_false_negatives(results)
    fp_list, fp_rule_counts          = get_false_positives(results)

    print(f"\n  accuracy={metrics['accuracy']}%  f1={metrics['f1']}%  "
          f"precision={metrics['precision']}%  recall={metrics['recall']}%")
    print(f"  FP={metrics['fp']}  FN={metrics['fn']}  "
          f"TP={metrics['tp']}  TN={metrics['tn']}")
    if fn_counts:
        print(f"  FN breakdown: {fn_counts.most_common(6)}")
    if fp_rule_counts:
        print(f"  FP-causing rules: {fp_rule_counts.most_common(4)}")

    change_log = []
    rules_by_id = {r["id"]: r for r in rules_data["rules"]}

    fn_thresh = max(1, int(n_corrupt * 0.06))
    fp_thresh = max(1, int(n_clean   * 0.04))

    # ── fix false negatives ────────────────────────────────────────────────
    for injection, count in fn_counts.most_common():
        if count < fn_thresh:
            continue
        if injection not in INJECTION_TO_FEATURE:
            continue

        feat_key, rule_id, op, delta = INJECTION_TO_FEATURE[injection]
        if feat_key is None:
            continue   # structural — no threshold to tune

        # find the median observed value for this feature in FN tables
        vals = [f.get(feat_key, 0) for f in fn_features[injection]]
        if not vals:
            continue
        median_val = sorted(vals)[len(vals)//2]

        if rule_id in rules_by_id:
            rule = rules_by_id[rule_id]
            _, _, cur_thr = get_threshold_from_condition(rule["condition"])
            if cur_thr is None:
                continue
            # only tighten if current threshold is above the median FN value
            if op == ">" and cur_thr > median_val:
                update_rule_threshold(
                    rule, delta,
                    f"Missed {count} corrupted tables via '{injection}' "
                    f"(median observed value {median_val:.3f} below threshold {cur_thr}). "
                    f"Tightening to reduce FN.",
                    change_log
                )
                # promote to hard if consistently causing many FNs
                if count / n_corrupt >= 0.12 and rule_id not in HARD_RULE_IDS:
                    promote_to_hard(rule_id,
                        f"'{injection}' caused {count} FNs ({count/n_corrupt*100:.0f}% of corrupted set)",
                        change_log)

    # ── fix false positives ────────────────────────────────────────────────
    for rule_id, count in fp_rule_counts.most_common():
        if count < fp_thresh:
            continue
        if rule_id not in FP_RULE_RELAX:
            continue
        rule = rules_by_id.get(rule_id)
        if not rule:
            continue
        feat_key, op, delta = FP_RULE_RELAX[rule_id]
        update_rule_threshold(
            rule, delta,
            f"Rule fired on {count} clean tables (false alarms). "
            f"Relaxing threshold to reduce FP.",
            change_log
        )

    # ── generate new rules for persistent FN patterns ─────────────────────
    for injection, count in fn_counts.most_common():
        if count < fn_thresh * 2:   # only for large persistent gaps
            continue
        # check if we already have a tight-enough rule for this
        mapped = INJECTION_TO_FEATURE.get(injection, (None,)*4)
        if mapped[0] is None:
            continue
        feat_key = mapped[0]
        vals = [f.get(feat_key, 0) for f in fn_features[injection] if f.get(feat_key, 0) > 0]
        if not vals:
            continue
        # if 25th percentile of FN vals is below current threshold, add a supplementary rule
        p25 = sorted(vals)[max(0, len(vals)//4 - 1)]
        existing_rule = rules_by_id.get(mapped[1])
        _, _, cur_thr = get_threshold_from_condition(existing_rule["condition"]) if existing_rule else (None, None, None)
        if cur_thr and p25 < cur_thr * 0.6:
            # the gap is large — add a secondary tighter rule
            new_thr = round(p25 * 1.5, 4)
            name    = f"{injection}_strict"
            if not any(r["name"] == name for r in rules_data["rules"]):
                add_new_rule(
                    rules_data,
                    feat_key, ">", new_thr,
                    name,
                    f"Stricter secondary rule for '{injection}' learned from PubTabNet FN analysis. "
                    f"25th-percentile of missed values was {p25:.3f}, well below primary threshold {cur_thr}.",
                    f"route=LOW, flag={injection}_strict",
                    25,
                    f"learned_iteration_{iteration}",
                    change_log
                )

    _patch_ineffective_fires(results, rules_data, change_log, n_corrupt)
    return metrics, change_log


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path")
    parser.add_argument("--demo",       action="store_true")
    parser.add_argument("--n-tables",   type=int, default=600)
    parser.add_argument("--iterations", type=int, default=4)
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)

    # ── load tables ────────────────────────────────────────────────────────
    if args.data_path and Path(args.data_path).exists():
        print(f"Loading PubTabNet from {args.data_path} ...")
        tables = load_pubtabnet(args.data_path, max_tables=5000)
    else:
        print(f"Generating {args.n_tables} synthetic PubTabNet-style tables ...")
        tables = generate_dataset(args.n_tables)

    n_clean   = sum(1 for t in tables if t["label"] == "clean")
    n_corrupt = sum(1 for t in tables if t["label"] == "corrupted")
    print(f"  {n_clean} clean  |  {n_corrupt} corrupted")

    # ── load rules ─────────────────────────────────────────────────────────
    rules_data = json.loads(RULES_PATH.read_text())
    all_iterations = []

    # ── iterative loop ─────────────────────────────────────────────────────
    for i in range(1, args.iterations + 1):
        print(f"\n{'═'*55}")
        print(f"  ITERATION {i}")
        print(f"{'═'*55}")

        metrics, changes = run_one_iteration(
            tables, rules_data, i, n_clean, n_corrupt
        )
        rules_data["version"] = i

        all_iterations.append({
            "iteration":   i,
            "metrics":     metrics,
            "rule_changes": changes,
            "rule_count":  len([r for r in rules_data["rules"] if r["status"]=="active"]),
        })

        if not changes:
            print(f"\n  ✓ Converged at iteration {i} — no further updates needed.")
            break

    # ── final eval ─────────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  SUMMARY")
    print(f"{'═'*55}")
    print(f"  {'iter':<8} {'accuracy':<12} {'f1':<10} {'precision':<12} {'recall':<10} {'FP':<6} {'FN':<6} {'rules'}")
    print(f"  {'─'*75}")
    for it in all_iterations:
        m = it["metrics"]
        print(f"  {it['iteration']:<8} {m['accuracy']:<12} {m['f1']:<10} "
              f"{m['precision']:<12} {m['recall']:<10} {m['fp']:<6} {m['fn']:<6} "
              f"{it['rule_count']}")

    first, last = all_iterations[0]["metrics"], all_iterations[-1]["metrics"]
    print(f"\n  DELTA  acc +{round(last['accuracy']-first['accuracy'],1)}pp  "
          f"f1 +{round(last['f1']-first['f1'],1)}pp  "
          f"recall +{round(last['recall']-first['recall'],1)}pp  "
          f"FN {first['fn']} → {last['fn']} "
          f"({first['fn']-last['fn']} caught)")

    # ── save ───────────────────────────────────────────────────────────────
    rules_data["last_updated"] = datetime.datetime.now().isoformat()
    RULES_PATH.write_text(json.dumps(rules_data, indent=2))

    full_log = {
        "run_timestamp": datetime.datetime.now().isoformat(),
        "n_tables":      len(tables),
        "n_clean":       n_clean,
        "n_corrupt":     n_corrupt,
        "iterations":    all_iterations,
        "final_rules":   rules_data["rules"],
    }
    LOG_PATH.write_text(json.dumps(full_log, indent=2))

    print(f"\n  Saved:")
    print(f"    rules.json          ← updated rule set (v{rules_data['version']})")
    print(f"    results/rule_update_log.json  ← full iteration log")


if __name__ == "__main__":
    main()


# ── patch: detect rules that fire but don't change tier ──────────────────────