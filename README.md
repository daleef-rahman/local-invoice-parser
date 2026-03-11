# Local Invoice Parser

**Can invoice parsing run fully locally, on a laptop CPU, without immediately becoming too slow or too expensive?**

This project is a practical benchmark for local invoice parsing. The motivation is straightforward: many companies want document extraction systems that run entirely on-prem or on-device for compliance, privacy, and data-handling reasons. Shipping invoices to a third-party API is often the hard part organizationally, even before model quality is discussed.

The second constraint is cost. Large multimodal models are powerful, but they are also expensive to operate at scale and usually want more GPU memory than a typical developer machine has available. This experiment looks at a narrower question instead: how far can you get with smaller models, running locally, with `llama.cpp` and CPU-friendly setups?

The repository compares two families of approaches:

- **OCR + NER**: extract text first with PaddleOCR, then convert it into structured invoice fields.
- **VLM**: feed the invoice image directly to a vision-language model and ask for structured JSON.

The output schema is defined in [schema.py](/Users/daleefrahman/Documents/startup-experiments/local-invoice-parser/local-invoice-parser/schema.py) as `AdvancedReceiptData`, covering totals, taxes, merchant details, receipt numbers, line items, and related fields.

## TL;DR

- **Direct VLM extraction is currently ahead on quality in this repo.** In the latest reports folder outputs, `Qwen2.5-VL-3B` is the current top scorer overall.
- **OCR + GLiNER2 is the fastest setup in the current repo.** It gives the lowest average time per invoice, which makes it a useful baseline for local CPU-first deployments.
- **The quality/speed tradeoff is now wider than before.** `SmolVLM-256M` is extremely fast at about `1.1s` per invoice, but accuracy is currently too low to be useful.
- **Local CPU-first experiments are viable for small evaluation sets.** On this machine, the implemented experiments land in roughly the `1-16` second range per invoice.

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

The current catalog in [catalog.py](/Users/daleefrahman/Documents/startup-experiments/local-invoice-parser/local-invoice-parser/experiments/catalog.py) defines eight active local experiments plus one cloud baseline reference in the README table.

**GLiNER2 -- the lightweight structured extraction baseline.** This is the simplest local NER-style baseline in the repo. It gives a useful floor for what a specialized extraction model can do after OCR, without requiring `llama.cpp` serving infrastructure.

**Qwen3-4B -- the stronger local text model for OCR + NER.** Once OCR text has been extracted, the next question is whether a small-but-capable instruction model can outperform a lighter NER baseline on structured field extraction. Qwen3-4B is a reasonable candidate because it is small enough to run locally, and it has also shown up as one of the strongest small models in this benchmark [LocalLLaMA Round 2 benchmark](https://www.reddit.com/r/LocalLLaMA/comments/1r4ie8z/i_tested_21_small_llms_on_toolcalling_judgment/), where `qwen3:4b` tied for the top score.

**Qwen2.5-VL-7B -- the direct image-to-JSON VLM candidate.** This experiment tests whether skipping OCR entirely gives better invoice understanding. It is larger than the text-only models in the repo, but still practical enough to run locally through `llama-server`. The Qwen VL family ranks near the top of the [OCRBench v2 leaderboard](https://huggingface.co/spaces/ling99/OCRBench-v2-leaderboard)

**Qwen2.5-VL-3B and Qwen3-VL-2B -- smaller direct VLM variants.** These extend the main Qwen multimodal path downward in size to test whether a smaller local VLM can preserve most of the quality while cutting latency.

**MiniCPM-V-4.5 -- an alternative local VLM path.** This serves as a second multimodal comparison point using a different model family

**LFM2.5-VL-1.6B and SmolVLM-256M -- aggressive small-model probes.** These are included to test how far the repo can push fully local CPU inference toward lighter multimodal models, even if quality drops sharply.

**Gemini 2.5 Flash -- a cloud baseline, not the target deployment.** Used as a reference point for benchmarking local approaches against a strong closed-source multimodal model.

## Experiments

| # | Experiment ID | Approach | Model |
|---|---|---|---|
| 1 | `exp1_ocr_ner_gliner2` | OCR + NER | PaddleOCR + GLiNER2 |
| 2 | `exp2_ocr_ner_qwen3` | OCR + NER | PaddleOCR + Qwen3-4B via `llama-server` |
| 3 | `exp3_vlm_qwen25vl` | VLM | Qwen2.5-VL-7B via `llama-server` |
| 4 | `exp4_vlm_minicpmv` | VLM | MiniCPM-V-4.5 via `llama-mtmd-cli` |
| 5 | `exp5_vlm_qwen25vl3b` | VLM | Qwen2.5-VL-3B via `llama-server` |
| 6 | `exp6_vlm_qwen3vl2b` | VLM | Qwen3-VL-2B via `llama-server` |
| 7 | `exp7_vlm_lfm25vl16b` | VLM | LFM2.5-VL-1.6B via `llama-mtmd-cli` |
| 8 | `exp8_vlm_smolvlm256m` | VLM | SmolVLM-256M via `llama-mtmd-cli` |
| B1 | `baseline_vlm_gemini25flash.py` | VLM baseline | Gemini-2.5-Flash |

## Results

The current results come from 9 sample invoices in `data/sample-invoices/`, evaluated with partial credit scoring. The first four rows are from the aggregate run in [all_experiments_simple_20260311_053501.json](/Users/daleefrahman/Documents/startup-experiments/local-invoice-parser/reports/all_experiments_simple_20260311_053501.json), and rows `5` through `8` come from the latest standalone report files in `reports/`.

| # | Experiment | Avg time/invoice | Scalar acc | Line-item acc | Overall acc |
|---|---|---:|---:|---:|---:|
| 1 | PaddleOCR + GLiNER2 | 5.8s | 0.5208 | 0.4424 | 0.4790 |
| 2 | PaddleOCR + Qwen3-4B | 12.1s | 0.6250 | 0.4357 | 0.5127 |
| 3 | Qwen2.5-VL-7B | 13.9s | 0.6389 | 0.6233 | 0.6310 |
| 4 | MiniCPM-V-4.5 | 16.4s | 0.6389 | 0.5355 | 0.5853 |
| 5 | Qwen2.5-VL-3B | 5.5s | 0.6667 | 0.6536 | 0.6602 |
| 6 | Qwen3-VL-2B | 5.6s | 0.6493 | 0.5167 | 0.5816 |
| 7 | LFM2.5-VL-1.6B | 3.9s | 0.5035 | 0.4562 | 0.4786 |
| 8 | SmolVLM-256M | 1.1s | 0.0451 | 0.0000 | 0.0229 |

Current read:

- `Qwen2.5-VL-3B` is the strongest result currently recorded in `reports/`, beating the larger `Qwen2.5-VL-7B` aggregate run on both overall and line-item accuracy while also running much faster.
- `Qwen2.5-VL-7B` and `Qwen2.5-VL-3B` are the strongest line-item extractors in the current set, with the 3B variant now leading overall.
- `GLiNER2` remains the fastest credible OCR baseline, while `Qwen3-4B` improves OCR-first quality at roughly double the latency.
- `Qwen3-VL-2B` is competitive on scalar fields, but its latest run includes one hard parse failure on invoice `1`, which drags down the overall score.
- `LFM2.5-VL-1.6B` is interesting on speed, but not yet strong enough to displace the OCR baseline on quality.
- `SmolVLM-256M` is only useful as a speed floor at the moment; its extraction quality is effectively non-viable for this task.

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
uv run python local-invoice-parser/eval.py --mode simple --experiment exp5_vlm_qwen25vl3b
uv run python local-invoice-parser/eval.py --mode simple --experiment exp6_vlm_qwen3vl2b
uv run python local-invoice-parser/eval.py --mode simple --experiment exp7_vlm_lfm25vl16b
uv run python local-invoice-parser/eval.py --mode simple --experiment exp8_vlm_smolvlm256m
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
