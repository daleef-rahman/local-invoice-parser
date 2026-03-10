# Local Invoice Parser

**Can invoice parsing run fully locally, on a laptop CPU, without immediately becoming too slow or too expensive?**

This project is a practical benchmark for local invoice parsing. The motivation is straightforward: many companies want document extraction systems that run entirely on-prem or on-device for compliance, privacy, and data-handling reasons. Shipping invoices to a third-party API is often the hard part organizationally, even before model quality is discussed.

The second constraint is cost. Large multimodal models are powerful, but they are also expensive to operate at scale and usually want more GPU memory than a typical developer machine has available. This experiment looks at a narrower question instead: how far can you get with smaller models, running locally, with `llama.cpp` and CPU-friendly setups?

The repository compares two families of approaches:

- **OCR + NER**: extract text first with PaddleOCR, then convert it into structured invoice fields.
- **VLM**: feed the invoice image directly to a vision-language model and ask for structured JSON.

The output schema is defined in [schema.py](/Users/daleefrahman/Documents/startup-experiments/local-invoice-parser/local-invoice-parser/schema.py) as `AdvancedReceiptData`, covering totals, taxes, merchant details, receipt numbers, line items, and related fields.

## TL;DR

- **Direct VLM extraction is currently ahead on quality in this repo.** `Qwen2.5-VL-7B` produced the best overall score in the current run.
- **OCR + GLiNER2 is the fastest setup in the current repo.** It gives the lowest average time per invoice, which makes it a useful baseline for local CPU-first deployments.
- **Local CPU-first experiments are viable for small evaluation sets.** On this machine, the implemented experiments land in roughly the 6-18 second range per invoice.

## The test machine

| Spec | Value |
|---|---|
| Laptop | MacBook Pro |
| Model identifier | Mac16,7 |
| Chip | Apple M4 Pro |
| CPU | 14 cores (10 performance + 4 efficiency) |
| Memory | 24 GB |
| GPU | Integrated Apple GPU (not the focus of this benchmark) |
| OS | macOS 15.6.1 |
| Kernel | Darwin 24.6.0 |

Everything in this repo is framed around local execution on a developer laptop, with CPU-feasible runtimes as the main constraint.

## The models and why they were chosen

The current catalog in [catalog.py](/Users/daleefrahman/Documents/startup-experiments/local-invoice-parser/local-invoice-parser/experiments/catalog.py) defines four active local experiments plus one cloud baseline reference in the README table.

**GLiNER2 -- the lightweight structured extraction baseline.** This is the simplest local NER-style baseline in the repo. It gives a useful floor for what a specialized extraction model can do after OCR, without requiring `llama.cpp` serving infrastructure.

**Qwen3-4B -- the stronger local text model for OCR + NER.** Once OCR text has been extracted, the next question is whether a small-but-capable instruction model can outperform a lighter NER baseline on structured field extraction. Qwen3-4B is a reasonable candidate because it is small enough to run locally in GGUF form while still being strong enough to test whether "OCR first, then LLM" remains competitive.

**Qwen2.5-VL-7B -- the direct image-to-JSON VLM candidate.** This experiment tests whether skipping OCR entirely gives better invoice understanding, especially for layout-sensitive fields and line items. It is larger than the text-only models in the repo, but still practical enough to run locally through `llama-server`.

**MiniCPM-V-4.5 -- an alternative local VLM path.** This serves as a second multimodal comparison point using a different model family and a different runtime path (`llama-mtmd-cli`). It helps separate "VLMs are better" from "one particular VLM happened to be better."

**Gemini 2.5 Flash -- a cloud baseline, not the target deployment.** The README keeps this as a reference point because it is useful to compare local approaches against a strong hosted multimodal model, even if the broader motivation of the project is to avoid relying on hosted APIs in production.

## Experiments

| # | Experiment ID | Approach | Model |
|---|---|---|---|
| 1 | `exp1_ocr_ner_gliner2` | OCR + NER | PaddleOCR + GLiNER2 |
| 2 | `exp2_ocr_ner_qwen3` | OCR + NER | PaddleOCR + Qwen3-4B via `llama-server` |
| 3 | `exp3_vlm_qwen25vl` | VLM | Qwen2.5-VL-7B via `llama-server` |
| 4 | `exp4_vlm_minicpmv` | VLM | MiniCPM-V-4.5 via `llama-mtmd-cli` |
| B1 | `baseline_vlm_gemini25flash.py` | VLM baseline | Gemini-2.5-Flash |

## Results

The current results come from 9 sample invoices in `data/sample-invoices/`, evaluated with partial credit scoring and saved in [all_experiments_simple_20260311_025303.json](/Users/daleefrahman/Documents/startup-experiments/local-invoice-parser/reports/all_experiments_simple_20260311_025303.json).

| # | Experiment | Avg time/invoice | Scalar acc | Line-item acc | Overall acc |
|---|---|---:|---:|---:|---:|
| 1 | PaddleOCR + GLiNER2 | 6.1s | 0.5278 | 0.4121 | 0.4660 |
| 2 | PaddleOCR + Qwen3-4B | 16.8s | 0.6250 | 0.4357 | 0.5127 |
| 3 | Qwen2.5-VL-7B | 15.3s | 0.6389 | 0.6233 | 0.6310 |
| 4 | MiniCPM-V-4.5 | 17.8s | 0.6389 | 0.5355 | 0.5853 |

Current read:

- `Qwen2.5-VL-7B` is the best overall performer in this run.
- `MiniCPM-V-4.5` is slightly behind on overall score, but still ahead of both OCR-based pipelines.
- `Qwen3-4B` improves meaningfully over `GLiNER2` in the OCR + NER setup, suggesting the OCR-first path still has room to improve with better text-side models.
- The biggest gap is on line-item extraction, where the VLM setups currently look stronger than OCR + NER.

## Setup

```bash
uv sync
```

Runtime responsibilities are split by backend:

- `gliner2` runs directly as a library backend
- `llama_server` is used for both OCR + NER and VLM experiments backed by `llama.cpp`
- `llama_mtmd_cli` is used for direct multimodal inference with `llama-mtmd-cli`

The evaluation runner in [eval.py](/Users/daleefrahman/Documents/startup-experiments/local-invoice-parser/local-invoice-parser/eval.py) handles the catalog-backed experiments end to end: it downloads missing Hugging Face assets when needed, starts `llama-server` for the relevant experiments, evaluates the invoice set, and shuts the runtime down afterward.

## Usage

### Run a pipeline directly

```python
from pipeline import close_pipeline, prepare_pipeline, run_pipeline

prepared = prepare_pipeline("exp1_ocr_ner_gliner2")
try:
    result = run_pipeline(prepared, "data/sample-invoices/1.png")
    print(result.receipt.model_dump())
finally:
    close_pipeline(prepared)
```

### Run evaluation

```bash
uv run python local-invoice-parser/eval.py --mode simple --experiment exp1_ocr_ner_gliner2
uv run python local-invoice-parser/eval.py --mode simple --experiment exp2_ocr_ner_qwen3
uv run python local-invoice-parser/eval.py --mode simple --experiment exp3_vlm_qwen25vl
uv run python local-invoice-parser/eval.py --mode simple --experiment exp4_vlm_minicpmv
uv run python local-invoice-parser/eval.py --mode simple --all
```

Reports are written to `reports/`.

## Repository structure

```text
local-invoice-parser/
  experiments/          # Experiment catalog and runtime metadata
  models/
    ner/                # GLiNER2 and llama-server NER backends
    ocr/                # OCR helpers
    vlm/                # llama-server and llama-mtmd-cli VLM backends
  pipeline.py           # Catalog-driven pipeline loader
  eval.py               # Evaluation runner
  schema.py             # Structured output schema
data/
  sample-invoices/      # Sample images and ground truth
reports/                # Evaluation outputs
```
