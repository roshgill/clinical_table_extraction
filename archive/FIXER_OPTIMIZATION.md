# Fixer Agent Prompt Optimization

## Background

The extraction pipeline uses **UniTable** as a first pass on every cropped table image. UniTable reliably recovers table structure (rows, columns, spanning cells) but its OCR is imperfect — it misreads symbols like `±`, Greek letters, numeric values, and scientific notation. A second pass by a **Gemini Fixer Agent** corrects these cell-level errors while leaving structure untouched.

The fixer receives two inputs:
- The original table image
- The UniTable HTML (structure correct, text possibly wrong)

It returns either corrected HTML or `NO_CHANGES` if nothing needs fixing.

The fixer's behaviour is entirely controlled by its **system prompt**. A better system prompt means fewer OCR errors reaching the output. The question is: how do we find the best one?

---

## Why Automated Optimization

Writing a good system prompt manually is harder than it looks. The errors are subtle — a `1` misread as `l`, a `±` dropped, a p-value rounded differently. They vary by table type, font, and image quality. Manual iteration is slow, and a prompt that fixes one class of error often introduces another.

We have ground truth: PubTabNet, a large dataset of scientific table images paired with correct HTML. This is the same distribution as Huma's tables (generic scientific literature). We can score any candidate prompt against ground truth using **TEDS** (Tree Edit Distance Similarity), a standard metric that captures both structure and content accuracy.

This makes prompt selection a search problem — and we can automate it.

---

## Approach: DSPy-Style Optimization

The approach is inspired by DSPy (Stanford), which treats LLM prompts as learnable parameters. The core idea: instead of asking the model to rewrite its own prompt once and hoping for improvement, generate multiple diverse candidates, evaluate all of them empirically, and carry forward the best.

### The Loop

Each iteration does five things:

1. **Evaluate** the current best prompt on 13 training images. Score each with TEDS; compute mean and standard deviation.

2. **Select a composite score**: `composite = mean_TEDS − 0.5 × std_TEDS`
   This penalises high-variance prompts. A prompt that scores 0.99 on easy tables and 0.75 on hard ones is less trustworthy in production than one that scores 0.93 consistently.

3. **Analyse errors** — diff predicted cell text against ground truth for any image below 0.97 TEDS. Surfaces specific failure patterns (e.g. "drops ± sign", "misreads subscripts").

4. **Generate 5 candidate prompts in parallel** using Gemini as a meta-prompt engineer. Each candidate is generated at a different temperature (0.7 → 1.3) to encourage diversity. Candidates are required to be meaningfully different strategies — conservative correction, aggressive correction, few-shot-guided, symbol-focused, etc.

5. **Score all 5 candidates** on the full 13-image training set. Pick the best by composite score.

The loop runs for up to 8 iterations, stopping early if the target composite (0.97) is reached or if no candidate improves on the best-ever score for 3 consecutive iterations.

### Few-Shot Bootstrapping

After each iteration, training images where the current prompt scored ≥ 0.98 TEDS are collected as verified correction examples. These `(UniTable HTML → corrected HTML)` pairs are injected into candidate prompts as reference examples. The model learns from its own successes.

### Diversity Tracking

If all 5 candidates are semantically similar (mean pairwise cosine similarity > 0.95), the search has collapsed — Gemini is effectively paraphrasing the same prompt. This is flagged per iteration. If it happens, meta-prompt temperatures can be raised.

### Structured Outputs

All LLM outputs use Pydantic schemas and `response_mime_type="application/json"` — no text parsing, no fragile regex. Each candidate prompt returns `{system_prompt, rationale}`. The rationale field forces the model to reason about its strategy before writing the prompt, improving output quality.

---

## Training vs Test

The 20 PubTabNet examples are split 13/7 (train/test, fixed seed). The optimization loop never sees the test set. After the loop completes, the best-ever prompt is evaluated on the held-out 7 images to check for overfitting and confirm generalisation.

The final comparison also includes a **UniTable-only baseline** on the test set, showing the concrete accuracy improvement the fixer adds.

---

## Fixer Design

- **Temperature = 0.0** (fixed) — deterministic, reproducible. Removing temperature as a variable makes TEDS scores stable across runs, giving clean signal to the optimization loop.
- **System prompt / user message separation** — instructions are in `system_instruction`; the table image and HTML are in the user turn. This means the optimization loop only needs to evolve the instructions, not the data injection.
- **`NO_CHANGES` pass-through** — if the model sees no errors, it returns the exact string `NO_CHANGES` and the original UniTable HTML is used unchanged. No wasted correction on already-correct tables.
- **Parallel fixer calls** — within each candidate's 13-image eval, fixer calls run in a `ThreadPoolExecutor`. TEDS scorers are instantiated per thread (the library is not thread-safe).

---

## Output

The notebook produces:
- A **prompt evolution table** — best prompt per iteration with composite score, mean TEDS, std, diversity, and few-shot count
- A **candidate scatter plot** — all 5 candidates per iteration overlaid on the composite trajectory, with UniTable baseline and target line
- A **test set validation** with train/test gap
- A **baseline comparison** table: UniTable-only vs hybrid per test image
- The **final best system prompt**, ready to drop into the production pipeline

---

## PubTabNet Evaluation Results

### Setup
- **Dataset:** `eval_unitable_results/` — 500 triplets (image, UniTable pred HTML, GT HTML) from PubTabNet
- **Filter:** TEDS ≥ 0.85 → 302 eligible samples
- **Notebook:** `notebooks/dspy_fixer_pubtabnet.ipynb`

### Run 1 — gemini-3-flash-preview
Config: T=1, S=1, N=2, I=2 (smoke test scale)
**Result:** Hybrid worse than UniTable baseline (exact scores not retained).

### Run 2 — gemini-3.1-pro-preview
Config: T=1, S=1, N=5, I=8 (smoke test scale)

| Metric | Value |
|---|---|
| Best train composite | 0.8403 |
| UniTable-only mean TEDS (test) | 0.901 |
| Best hybrid mean TEDS (test) | 0.796 |
| Net improvement | **-0.105 (regression)** |
| NO_CHANGES rate | 0% — fixer always made changes |
| Stopped at | Iteration 3 (plateau 3/3) |
| Diversity | 0.963–0.974 — candidate collapse throughout |

**Failure modes identified:**
- **T=1 kills the signal** — with one training image, `std=0.0` on every score, the variance penalty is useless and the composite equals the mean. Optimization becomes random selection among near-identical prompts.
- **Candidate collapse** — diversity >0.963 (above 0.95 warning threshold) on every iteration. Meta-prompt generates near-identical candidates regardless of temperature variation.
- **Decimal comma bug** — fixer converted `22.7 → 22,7`, `31.2 → 31,2` (European locale format). A systematic wrong correction.
- **0% NO_CHANGES** — fixer always intervened; never conservative. On a dataset where UniTable already scores 0.901, this is fatal.

**Root cause:** T=1 is too small for any meaningful optimization. The next run should use T=16, S=30 (see calculation: ≈7 hours with EVAL_WORKERS=8, N=5, I=8).
