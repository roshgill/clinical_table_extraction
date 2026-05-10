"""
unitable_agent.py — real UniTable integration via RunPod
Place in: agents/unitable_agent.py
"""

import sys, os, json, argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.abspath(".."))

from PIL import Image
from shared.rule_engine import evaluate as rule_evaluate, load_rules
from shared.teds import teds
from shared.unitable_endpoint import extract_html_remote

RULES_PATH     = Path(__file__).parent.parent / "shared" / "rules.json"
TEDS_THRESHOLD = 0.85


# ── Gemini fallback ────────────────────────────────────────────────────────────

def run_gemini(image: Image.Image) -> str:
    import google.generativeai as genai, base64, io
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    response = genai.GenerativeModel("gemini-1.5-pro").generate_content([
        {"mime_type": "image/png", "data": base64.b64encode(buf.getvalue()).decode()},
        "Extract this table to clean HTML with <thead>/<tbody>. "
        "Preserve all colspan and rowspan attributes exactly. Return only the HTML."
    ])
    return response.text.strip()


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


# ── single-table agent ─────────────────────────────────────────────────────────

class UniTableAgent:

    def __init__(self):
        self.rules_data = json.loads(RULES_PATH.read_text())

    def run(self, image_path: str, gt_html: Optional[str] = None) -> AgentResult:
        image = Image.open(image_path).convert("RGB")

        # 1 — UniTable via RunPod
        try:
            pred_html = extract_html_remote(image)
            extractor = "unitable"
        except Exception as e:
            return AgentResult(
                image_path=image_path, extraction_html="",
                extractor_used="none", rule_tier="LOW",
                confidence=0, fired_rules=[], error=str(e), routed_for_review=True
            )

        # 2 — rule engine
        result = rule_evaluate(pred_html, image_path, self.rules_data)

        # 3 — routing
        routed_for_review = False
        if result.tier in ("MEDIUM", "LOW"):
            try:
                gemini_html   = run_gemini(image)
                gemini_result = rule_evaluate(gemini_html, image_path, self.rules_data)
                if gemini_result.confidence > result.confidence:
                    pred_html, result, extractor = gemini_html, gemini_result, "gemini_fallback"
            except Exception:
                pass
            if result.tier == "LOW":
                routed_for_review = True

        # 4 — TEDs
        teds_score = teds_s_score = None
        if gt_html:
            teds_score   = teds(pred_html, gt_html, structure_only=False)
            teds_s_score = teds(pred_html, gt_html, structure_only=True)

        return AgentResult(
            image_path=image_path, extraction_html=pred_html,
            extractor_used=extractor, rule_tier=result.tier,
            confidence=result.confidence, fired_rules=result.fired_rules,
            teds_score=teds_score, teds_s_score=teds_s_score,
            routed_for_review=routed_for_review,
        )


# ── batch agent ────────────────────────────────────────────────────────────────

class UniTableBatchAgent:

    def __init__(self):
        self.agent = UniTableAgent()

    def run_folder(self, images_dir: str, gt_dir: Optional[str] = None) -> dict:
        paths = sorted(Path(images_dir).glob("*.png")) + sorted(Path(images_dir).glob("*.jpg"))
        _print_header()
        results = []
        for p in paths:
            gt = (Path(gt_dir) / (p.stem + ".html")).read_text() \
                 if gt_dir and (Path(gt_dir) / (p.stem + ".html")).exists() else None
            r = self.agent.run(str(p), gt)
            results.append(r)
            _print_row(r)
        return _summarise(results)

    def run_pubtabnet_jsonl(self, jsonl_path: str, n: int = 200,
                            images_dir: Optional[str] = None) -> dict:
        """
        Run on PubTabNet JSONL.
        - With images_dir: real UniTable extraction + TEDs vs annotation GT
        - Without images_dir: validates annotation HTML through rule engine only
        """
        records = []
        with open(jsonl_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= n: break
                try: records.append(json.loads(line))
                except: continue

        print(f"\n  Loaded {len(records)} tables from {Path(jsonl_path).name}")
        _print_header()

        results = []
        for rec in records:
            fname   = rec.get("filename", f"table_{len(results)}")
            gt_html = _reconstruct_html(rec)
            if not gt_html:
                continue

            img_path = Path(images_dir) / fname if images_dir else None
            if img_path and img_path.exists():
                r = self.agent.run(str(img_path), gt_html)
            else:
                # no image available — score GT HTML through rule engine
                re_result = rule_evaluate(gt_html, fname, self.agent.rules_data)
                r = AgentResult(
                    image_path=fname, extraction_html=gt_html,
                    extractor_used="gt_passthrough", rule_tier=re_result.tier,
                    confidence=re_result.confidence, fired_rules=re_result.fired_rules,
                    teds_score=1.0, teds_s_score=1.0,  # GT vs itself
                )
            results.append(r)
            _print_row(r)

        summary = _summarise(results)

        # save low-TEDs tables as candidates for rule_updater
        candidates = [
            {"image": r.image_path, "teds": r.teds_score,
             "tier": r.rule_tier, "fired": r.fired_rules}
            for r in results
            if r.teds_score is not None and r.teds_score < TEDS_THRESHOLD
        ]
        if candidates:
            out = Path(__file__).parent.parent / "results" / "rule_update_candidates.json"
            out.parent.mkdir(exist_ok=True)
            out.write_text(json.dumps(candidates, indent=2))
            print(f"\n  {len(candidates)} rule-update candidates → {out.name}")

        return summary


# ── PubTabNet HTML reconstruction ──────────────────────────────────────────────

def _reconstruct_html(row: dict) -> str:
    try:
        tokens, cells, idx, parts = row["html"]["structure"]["tokens"], row["html"]["cell"], 0, ["<table>"]
        for tok in tokens:
            if tok in ("<td>","<th>") or tok.startswith("<td ") or tok.startswith("<th "):
                parts.append(tok)
                if idx < len(cells):
                    c = cells[idx]
                    parts.append(c.get("tokens",[""])[0] if isinstance(c, dict) else str(c))
                    idx += 1
            else:
                parts.append(tok)
        parts.append("</table>")
        return "".join(parts)
    except:
        return ""


# ── display ────────────────────────────────────────────────────────────────────

def _print_header():
    print(f"\n  {'Image':<28} {'Extractor':<16} {'Tier':<8} {'Conf':<6} {'TEDs':<8} {'TEDs-S':<8} {'Fired'}")
    print(f"  {'─'*78}")

def _print_row(r: AgentResult):
    t    = f"{r.teds_score:.3f}"   if r.teds_score   is not None else "n/a"
    ts   = f"{r.teds_s_score:.3f}" if r.teds_s_score is not None else "n/a"
    name = Path(r.image_path).name[-26:]
    flag = " ⚠" if r.routed_for_review else ""
    err  = f" ERR:{r.error[:15]}" if r.error else ""
    print(f"  {name:<28} {r.extractor_used:<16} {r.rule_tier:<8} "
          f"{r.confidence:<6} {t:<8} {ts:<8} {','.join(r.fired_rules[:2]) or '—'}{flag}{err}")

def _summarise(results):
    scored = [r for r in results if r.teds_score is not None]
    mt  = round(sum(r.teds_score   for r in scored)/len(scored), 4) if scored else None
    mts = round(sum(r.teds_s_score for r in scored)/len(scored), 4) if scored else None
    print(f"\n{'═'*50}")
    print(f"  {len(results)} tables  |  "
          f"HIGH={sum(1 for r in results if r.rule_tier=='HIGH')}  "
          f"MEDIUM={sum(1 for r in results if r.rule_tier=='MEDIUM')}  "
          f"LOW={sum(1 for r in results if r.rule_tier=='LOW')}")
    print(f"  UniTable={sum(1 for r in results if r.extractor_used=='unitable')}  "
          f"Gemini={sum(1 for r in results if r.extractor_used=='gemini_fallback')}")
    if mt: print(f"  Mean TEDs={mt}  TEDs-S={mts}  "
                 f"Below {TEDS_THRESHOLD}={sum(1 for r in scored if r.teds_score < TEDS_THRESHOLD)}")
    print(f"{'═'*50}")
    return {"n": len(results), "mean_teds": mt, "mean_teds_s": mts}


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image",      help="Single image")
    ap.add_argument("--gt",         help="Ground truth HTML for single image")
    ap.add_argument("--batch",      help="Folder of images")
    ap.add_argument("--gt-dir",     help="Matching GT HTML folder")
    ap.add_argument("--jsonl",      help="PubTabNet JSONL path")
    ap.add_argument("--images-dir", help="PubTabNet images folder")
    ap.add_argument("--n",          type=int, default=200)
    args = ap.parse_args()

    if args.image:
        agent  = UniTableAgent()
        gt     = Path(args.gt).read_text() if args.gt else None
        result = agent.run(args.image, gt)
        _print_header()
        _print_row(result)

    elif args.batch:
        UniTableBatchAgent().run_folder(args.batch, args.gt_dir)

    elif args.jsonl:
        UniTableBatchAgent().run_pubtabnet_jsonl(args.jsonl, n=args.n, images_dir=args.images_dir)

    else:
        ap.print_help()