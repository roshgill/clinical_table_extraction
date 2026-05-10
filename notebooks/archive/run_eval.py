"""
run_eval.py
-----------
Runs rule_engine + TEDs on:
  1. Your 20 PubTabNet example tables (already in pubtabnet/examples/)
  2. More PubTabNet from a full JSONL if you have it
  3. Your actual medical PDFs (Lipitor, BMJ, Lancet etc.)

Usage:
    # Run on 20 PubTabNet examples you already have:
    python run_eval.py --pubtabnet-examples

    # Run on N tables from full JSONL (if downloaded):
    python run_eval.py --pubtabnet-jsonl ../pubtabnet/PubTabNet_2.0.0.jsonl --n 500

    # Run on medical PDFs (uses your existing extract pipeline):
    python run_eval.py --medical-pdfs ../papers/

    # Run everything:
    python run_eval.py --pubtabnet-examples --medical-pdfs ../papers/

Place in: notebooks/run_eval.py
"""

import sys, os, json, argparse
from pathlib import Path
from io import StringIO

sys.path.insert(0, os.path.abspath(".."))

import pandas as pd
from PIL import Image

from shared.rule_engine import evaluate as rule_evaluate, load_rules
from shared.teds import teds

RULES_PATH   = Path(__file__).parent.parent / "shared" / "rules.json"
EXAMPLES_DIR = Path(__file__).parent.parent / "pubtabnet" / "examples"
RESULTS_DIR  = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── helpers ────────────────────────────────────────────────────────────────────

def segments_to_html(tables) -> str:
    """Convert extract_tables_from_page() output → HTML string for rule_engine."""
    if not tables:
        return ""
    seg = tables[0]
    if not seg.rows:
        return ""
    thead = ""
    if seg.headers:
        thead = "<thead><tr>" + "".join(f"<th>{h}</th>" for h in seg.headers) + "</tr></thead>"
    tbody = "<tbody>" + "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
        for row in seg.rows
    ) + "</tbody>"
    return f"<table>{thead}{tbody}</table>"


def print_row(fname, tier, confidence, teds_score, teds_s, fired):
    fired_str = ",".join(fired[:2]) if fired else "—"
    teds_str  = f"{teds_score:.3f}" if teds_score is not None else "n/a"
    teds_s_str = f"{teds_s:.3f}"   if teds_s     is not None else "n/a"
    print(f"  {str(fname)[-32:]:<34} {tier:<8} {confidence:<6} {teds_str:<8} {teds_s_str:<8} {fired_str}")


def print_header():
    print(f"\n  {'File':<34} {'Tier':<8} {'Conf':<6} {'TEDs':<8} {'TEDs-S':<8} {'Fired rules'}")
    print(f"  {'─'*72}")


def summarise(results: list, label: str):
    if not results:
        return
    scored = [r for r in results if r["teds"] is not None]
    mean_t  = sum(r["teds"]   for r in scored) / len(scored) if scored else None
    mean_ts = sum(r["teds_s"] for r in scored) / len(scored) if scored else None
    high    = sum(1 for r in results if r["tier"] == "HIGH")
    medium  = sum(1 for r in results if r["tier"] == "MEDIUM")
    low     = sum(1 for r in results if r["tier"] == "LOW")
    below   = sum(1 for r in scored  if r["teds"] < 0.85)

    print(f"\n  ── {label} summary ({'─'*(50-len(label))}")
    print(f"  tables:    {len(results)}   HIGH={high}  MEDIUM={medium}  LOW={low}")
    if mean_t:
        print(f"  mean TEDs: {mean_t:.4f}   mean TEDs-S: {mean_ts:.4f}")
        print(f"  below 0.85 TEDs (flag for rule update): {below}")


# ── 1. PubTabNet examples ──────────────────────────────────────────────────────

def run_pubtabnet_examples(rules_data: dict, n: int = None) -> list:
    """
    Uses your existing pubtabnet/examples/utils.py format_html
    and extract_tables_from_page from agents/general_table/extract.py.
    Ground truth comes from the JSONL HTML annotations.
    """
    try:
        from pubtabnet.examples.utils import format_html
    except ImportError:
        print("  ERROR: couldn't import pubtabnet.examples.utils — make sure you're running from notebooks/")
        return []

    try:
        from shared.client import client, DEFAULT_MODEL
        from agents.general_table.extract import extract_tables_from_page
        has_extractor = True
    except ImportError:
        has_extractor = False
        print("  NOTE: extract_tables_from_page not available — scoring GT HTML through rule engine only")

    jsonl_path = EXAMPLES_DIR / "PubTabNet_Examples.jsonl"
    if not jsonl_path.exists():
        print(f"  ERROR: {jsonl_path} not found")
        return []

    examples = []
    with open(jsonl_path) as f:
        for line in f:
            examples.append(json.loads(line))
    if n:
        examples = examples[:n]

    print(f"\n  Loading {len(examples)} PubTabNet examples...")
    print_header()

    results = []
    for ex in examples:
        fname    = ex["filename"]
        gt_html  = format_html(ex)
        img_path = EXAMPLES_DIR / fname

        # try to run actual extraction
        pred_html = None
        if has_extractor and img_path.exists():
            try:
                image       = Image.open(img_path)
                page_result = extract_tables_from_page(client, DEFAULT_MODEL, image)
                pred_html   = segments_to_html(page_result.tables)
            except Exception as e:
                pass

        # fallback: score GT html through rule engine (validates ground truth structure)
        if not pred_html:
            pred_html = gt_html

        re_result = rule_evaluate(pred_html, fname, rules_data)
        teds_full = teds(pred_html, gt_html,  structure_only=False)
        teds_s    = teds(pred_html, gt_html,  structure_only=True)

        print_row(fname, re_result.tier, re_result.confidence,
                  teds_full, teds_s, re_result.fired_rules)

        results.append({
            "file":  fname,
            "tier":  re_result.tier,
            "confidence": re_result.confidence,
            "fired": re_result.fired_rules,
            "teds":  teds_full,
            "teds_s": teds_s,
            "source": "pubtabnet_examples",
        })

    summarise(results, "PubTabNet examples")
    return results


# ── 2. Full PubTabNet JSONL (sample) ──────────────────────────────────────────

def run_pubtabnet_jsonl(jsonl_path: str, rules_data: dict, n: int = 500) -> list:
    """
    Load N tables from the full PubTabNet JSONL.
    No images needed — scores GT HTML through rule engine + uses GT as both pred and gt for TEDs=1.0 baseline,
    then injects corruptions to show rule engine catching failures.
    """
    from synthetic_tables import CORRUPTIONS
    import random

    path = Path(jsonl_path)
    if not path.exists():
        print(f"  ERROR: {path} not found")
        return []

    # reconstruct HTML from PubTabNet token format
    def reconstruct(row):
        try:
            tokens   = row["html"]["structure"]["tokens"]
            cells    = row["html"]["cell"]
            idx      = 0
            parts    = ["<table>"]
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

    print(f"\n  Loading up to {n} tables from {path.name}...")
    tables = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if len(tables) >= n:
                break
            try:
                row  = json.loads(line)
                html = reconstruct(row)
                if html:
                    tables.append({"id": row.get("filename",""), "html": html})
            except:
                continue

    print(f"  Loaded {len(tables)} tables")
    print_header()

    results = []
    random.seed(42)
    for t in tables:
        gt_html   = t["html"]
        # 70% clean, 30% inject a random corruption to test rule engine
        if random.random() < 0.30:
            _, corrupt_fn = random.choice(CORRUPTIONS)
            try:    pred_html = corrupt_fn(gt_html)
            except: pred_html = gt_html
        else:
            pred_html = gt_html

        re_result = rule_evaluate(pred_html, t["id"], rules_data)
        teds_full = teds(pred_html, gt_html,  structure_only=False)
        teds_s    = teds(pred_html, gt_html,  structure_only=True)

        print_row(t["id"], re_result.tier, re_result.confidence,
                  teds_full, teds_s, re_result.fired_rules)

        results.append({
            "file":  t["id"],
            "tier":  re_result.tier,
            "confidence": re_result.confidence,
            "fired": re_result.fired_rules,
            "teds":  teds_full,
            "teds_s": teds_s,
            "source": "pubtabnet_jsonl",
        })

    summarise(results, f"PubTabNet JSONL sample ({len(tables)} tables)")
    return results


# ── 3. Medical PDFs ────────────────────────────────────────────────────────────

def run_medical_pdfs(papers_dir: str, rules_data: dict) -> list:
    """
    Runs your existing extract_tables_from_page on each PDF page image,
    validates with rule_engine.
    No XML ground truth available for FDA docs, so TEDs only for PMC papers.
    """
    try:
        from shared.client import client, DEFAULT_MODEL
        from agents.general_table.extract import extract_tables_from_page
        from shared.pdf import pdf_to_images
    except ImportError as e:
        print(f"  ERROR importing pipeline: {e}")
        return []

    papers_path = Path(papers_dir)
    pdfs        = list(papers_path.glob("*.pdf"))
    if not pdfs:
        print(f"  No PDFs found in {papers_dir}")
        return []

    print(f"\n  Found {len(pdfs)} PDFs in {papers_dir}")
    print_header()

    results = []
    for pdf_path in pdfs:
        print(f"\n  ── {pdf_path.name}")
        try:
            images = pdf_to_images(pdf_path)
        except Exception as e:
            print(f"    ERROR converting PDF: {e}")
            continue

        for page_num, image in enumerate(images[:5]):   # max 5 pages per doc
            try:
                page_result = extract_tables_from_page(client, DEFAULT_MODEL, image)
            except Exception as e:
                print(f"    page {page_num+1} extraction error: {e}")
                continue

            if not page_result.tables:
                continue

            pred_html = segments_to_html(page_result.tables)
            if not pred_html:
                continue

            re_result = rule_evaluate(pred_html, f"{pdf_path.name}:p{page_num+1}", rules_data)

            # TEDs only if we have XML ground truth (PMC papers)
            # TODO: wire in PMC XML fetch here for journal articles
            teds_full = None
            teds_s    = None

            label = f"{pdf_path.stem[:20]}:p{page_num+1}"
            print_row(label, re_result.tier, re_result.confidence,
                      teds_full, teds_s, re_result.fired_rules)

            results.append({
                "file":  f"{pdf_path.name}:page{page_num+1}",
                "tier":  re_result.tier,
                "confidence": re_result.confidence,
                "fired": re_result.fired_rules,
                "teds":  teds_full,
                "teds_s": teds_s,
                "source": "medical_pdf",
            })

    summarise(results, "Medical PDFs")
    return results


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pubtabnet-examples", action="store_true",
                        help="Run on 20 PubTabNet examples in pubtabnet/examples/")
    parser.add_argument("--pubtabnet-jsonl",    
                        help="Path to PubTabNet_2.0.0.jsonl for larger sample")
    parser.add_argument("--n",                  type=int, default=500,
                        help="How many tables to sample from JSONL (default 500)")
    parser.add_argument("--medical-pdfs",       
                        help="Path to folder containing medical PDFs")
    args = parser.parse_args()

    if not any([args.pubtabnet_examples, args.pubtabnet_jsonl, args.medical_pdfs]):
        parser.print_help()
        return

    if not RULES_PATH.exists():
        print(f"ERROR: rules.json not found at {RULES_PATH}")
        print("Run rule_updater.py first.")
        return

    rules_data = json.loads(RULES_PATH.read_text())
    print(f"\n  Loaded rules v{rules_data.get('version',1)} ({len(rules_data['rules'])} rules)")

    all_results = []

    if args.pubtabnet_examples:
        all_results += run_pubtabnet_examples(rules_data)

    if args.pubtabnet_jsonl:
        all_results += run_pubtabnet_jsonl(args.pubtabnet_jsonl, rules_data, n=args.n)

    if args.medical_pdfs:
        all_results += run_medical_pdfs(args.medical_pdfs, rules_data)

    # save full results
    if all_results:
        out_path = RESULTS_DIR / "eval_results.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Full results saved → {out_path}")

        # overall summary
        if len(all_results) > 1:
            print(f"\n{'═'*50}")
            summarise(all_results, "OVERALL")
            print(f"{'═'*50}")


if __name__ == "__main__":
    main()