"""
unitable_agent.py — rule engine evaluation on PubTabNet + real UniTable
Place in: agents/unitable_agent.py
"""

import sys, os, json, argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

sys.path.insert(0, os.path.abspath(".."))

from shared.rule_engine import evaluate as rule_evaluate, load_rules
from shared.teds import teds

RULES_PATH     = Path(__file__).parent.parent / "shared" / "rules.json"
TEDS_THRESHOLD = 0.85


# ── result ─────────────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    image_path:        str
    extraction_html:   str
    extractor_used:    str        # "unitable" | "gemini_fallback" | "gt_passthrough"
    rule_tier:         str
    confidence:        float
    fired_rules:       list
    teds_score:        Optional[float] = None
    teds_s_score:      Optional[float] = None
    routed_for_review: bool = False
    error:             Optional[str] = None


# ── PubTabNet HTML reconstruction ──────────────────────────────────────────────

def _reconstruct_html(row: dict) -> str:
    """
    Reconstruct HTML from PubTabNet JSONL format.
    Keys: html.structure.tokens (structure tags) + html.cells[].tokens (cell content)
    """
    try:
        html_data = row["html"]
        tokens    = html_data["structure"]["tokens"]
        cells     = html_data.get("cells", [])   # correct key

        cell_idx = 0
        parts    = ["<table>"]

        for tok in tokens:
            if tok in ("<td>", "<th>") or tok.startswith("<td ") or tok.startswith("<th "):
                parts.append(tok)
                if cell_idx < len(cells):
                    c = cells[cell_idx]
                    content = "".join(c.get("tokens", [])) if isinstance(c, dict) else str(c)
                    parts.append(content)
                    cell_idx += 1
            elif tok in ("</td>", "</th>"):
                parts.append(tok)
            else:
                parts.append(tok)

        parts.append("</table>")
        return "".join(parts)
    except Exception:
        return ""


# ── display helpers ────────────────────────────────────────────────────────────

def _print_header():
    print(f"\n  {'Image':<30} {'Extractor':<16} {'Tier':<8} {'Conf':<6} {'TEDs':<8} {'TEDs-S':<8} {'Fired'}")
    print(f"  {'─'*82}")

def _print_row(r: AgentResult):
    t    = f"{r.teds_score:.3f}"   if r.teds_score   is not None else "n/a"
    ts   = f"{r.teds_s_score:.3f}" if r.teds_s_score is not None else "n/a"
    name = Path(r.image_path).name[-28:]
    flag = " ⚠" if r.routed_for_review else ""
    err  = f" ERR:{r.error[:20]}" if r.error else ""
    print(f"  {name:<30} {r.extractor_used:<16} {r.rule_tier:<8} "
          f"{r.confidence:<6.3f} {t:<8} {ts:<8} {','.join(r.fired_rules[:2]) or '—'}{flag}{err}")

def _summarise(results):
    scored = [r for r in results if r.teds_score is not None]
    mt  = round(sum(r.teds_score   for r in scored) / len(scored), 4) if scored else None
    mts = round(sum(r.teds_s_score for r in scored) / len(scored), 4) if scored else None

    print(f"\n{'═'*55}")
    print(f"  {len(results)} tables processed")
    print(f"  HIGH={sum(1 for r in results if r.rule_tier=='HIGH')}  "
          f"MEDIUM={sum(1 for r in results if r.rule_tier=='MEDIUM')}  "
          f"LOW={sum(1 for r in results if r.rule_tier=='LOW')}")
    if mt is not None:
        print(f"  Mean TEDs={mt}  TEDs-S={mts}  "
              f"Below {TEDS_THRESHOLD}={sum(1 for r in scored if r.teds_score < TEDS_THRESHOLD)}")
    print(f"{'═'*55}\n")
    return {"n": len(results), "mean_teds": mt, "mean_teds_s": mts}


# ── batch runner ───────────────────────────────────────────────────────────────

def run_pubtabnet_jsonl(jsonl_path: str, n: int = 200, images_dir: Optional[str] = None) -> dict:
    rules_data = json.loads(Path(RULES_PATH).read_text())

    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            try:
                records.append(json.loads(line))
            except Exception:
                continue

    print(f"\n  Loaded {len(records)} tables from {Path(jsonl_path).name}")
    _print_header()

    results = []
    errors  = 0

    for rec in records:
        fname   = rec.get("filename", f"table_{len(results)}")
        gt_html = _reconstruct_html(rec)

        if not gt_html:
            errors += 1
            continue

        img_path = Path(images_dir) / fname if images_dir else None

        if img_path and img_path.exists():
            try:
                from PIL import Image
                from shared.unitable_endpoint import extract_html_remote
                image     = Image.open(str(img_path)).convert("RGB")
                pred_html = extract_html_remote(image)
                extractor = "unitable"
            except Exception as e:
                r = AgentResult(
                    image_path=fname, extraction_html="",
                    extractor_used="none", rule_tier="LOW",
                    confidence=0.0, fired_rules=[], error=str(e), routed_for_review=True
                )
                results.append(r)
                _print_row(r)
                errors += 1
                continue

            re_result    = rule_evaluate(pred_html, fname, rules_data)
            teds_score   = teds(pred_html, gt_html, structure_only=False)
            teds_s_score = teds(pred_html, gt_html, structure_only=True)

            r = AgentResult(
                image_path=fname, extraction_html=pred_html,
                extractor_used=extractor, rule_tier=re_result.tier,
                confidence=re_result.confidence, fired_rules=re_result.fired_rules,
                teds_score=teds_score, teds_s_score=teds_s_score,
                routed_for_review=(re_result.tier == "LOW"),
            )

        else:
            # GT passthrough — score annotation HTML through rule engine only
            re_result = rule_evaluate(gt_html, fname, rules_data)
            r = AgentResult(
                image_path=fname, extraction_html=gt_html,
                extractor_used="gt_passthrough", rule_tier=re_result.tier,
                confidence=re_result.confidence, fired_rules=re_result.fired_rules,
                teds_score=1.0, teds_s_score=1.0,
            )

        results.append(r)
        _print_row(r)

    summary = _summarise(results)

    candidates = [
        {"image": r.image_path, "teds": r.teds_score,
         "tier": r.rule_tier, "fired": r.fired_rules}
        for r in results if r.rule_tier == "LOW"
    ]
    if candidates:
        out = Path(__file__).parent.parent / "results" / "rule_update_candidates.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(candidates, indent=2))
        print(f"  {len(candidates)} LOW-tier candidates → results/rule_update_candidates.json")

    if errors:
        print(f"  {errors} rows skipped due to errors")

    return summary


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl",      required=True, help="PubTabNet JSONL path")
    ap.add_argument("--images-dir", help="PubTabNet val images folder (optional)")
    ap.add_argument("--n",          type=int, default=200, help="Number of tables to evaluate")
    args = ap.parse_args()

    run_pubtabnet_jsonl(args.jsonl, n=args.n, images_dir=args.images_dir)