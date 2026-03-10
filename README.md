# Local Invoice Parser

Experiments for locally-running invoice/receipt parsing. Compares two approaches:

- **OCR + NER**: PaddleOCR extracts text, a NER model extracts structured fields
- **VLM**: A vision-language model reads the image directly and returns structured fields

Extracted fields are defined in [schema.py](local-invoice-parser/schema.py) (`AdvancedReceiptData`): total amount, tax, merchant info, line items, and more.

## Experiments

| # | Experiment ID | Approach | Model |
|---|---------------|----------|-------|
| 1 | `exp1_ocr_ner_gliner2` | OCR + NER | PaddleOCR + GLiNER2 |
| 2 | `exp2_ocr_ner_qwen3` | OCR + NER | PaddleOCR + Qwen3-4B via `llama-server` |
| 3 | `exp3_vlm_qwen25vl` | VLM | Qwen2.5-VL-7B via `llama-server` |
| 4 | `exp4_vlm_minicpmv` | VLM | MiniCPM-V-4.5 via `llama-mtmd-cli` |
| B1 | `baseline_vlm_gemini25flash.py` | VLM (baseline-only) | Gemini-2.5-Flash |
| 5 | `exp5_vlm_qwen25_3b.py` | VLM | qwen2.5:3b |
| 6 | `exp6_vlm_qwen25_1_5b.py` | VLM | qwen2.5:1.5b |
| 7 | `exp7_vlm_lfm25_1_2b.py` | VLM | lfm2.5:1.2b |

## Results

Evaluated on 9 sample invoices (`data/sample-invoices/`) from [`reports/all_experiments_simple_20260311_025303.json`](/Users/daleefrahman/Documents/startup-experiments/local-invoice-parser/reports/all_experiments_simple_20260311_025303.json). Scores use partial credit (0/0.5/1.0 per field).

| # | Experiment | Avg time/invoice | Scalar acc | Line-item acc | Overall acc |
|---|-----------|-----------------|------------|---------------|-------------|
| 1 | PaddleOCR + GLiNER2 | 6.1s | 0.5278 | 0.4121 | 0.4660 |
| 2 | PaddleOCR + Qwen3-4B | 16.8s | 0.6250 | 0.4357 | 0.5127 |
| 3 | Qwen2.5-VL-7B | 15.3s | 0.6389 | 0.6233 | 0.6310 |
| 4 | MiniCPM-V-4.5 | 17.8s | 0.6389 | 0.5355 | 0.5853 |

## Setup

```bash
uv sync
```

Backend adapters are now organized by runtime instead of by model family:

- `GLiNER2` remains a direct library backend
- `llama_server` handles both OCR+NER and VLM extraction, with `task_type` selecting the prompt shape
- `llama_mtmd_cli` handles direct multimodal CLI inference

`eval.py` manages the runtime for catalog-backed experiments: it downloads missing model assets with `hf`, starts `llama-server` once per experiment when needed, evaluates all samples, and stops the server in `finally`.

Manual scripts are still available for debugging:

```bash
# Exp 2 — Qwen3 NER (default port 8080)
./scripts/serve_qwen3.sh

# Exp 3 — Qwen2.5-VL (default port 8080)
./scripts/serve_qwen25vl.sh

```

Each script downloads the GGUF model on first run (via `huggingface_hub`) and starts `llama-server`.

Exp 4 uses `llama-mtmd-cli` directly (no server needed). Ensure `llama-mtmd-cli` is installed and in `PATH`.

## Usage

### Direct

Use the helpers directly from Python:

```python
from pipeline import close_pipeline, prepare_pipeline, run_pipeline

prepared = prepare_pipeline("exp1_ocr_ner_gliner2")
try:
    result = run_pipeline(prepared, "data/sample-invoices/1.png")
    print(result.receipt.model_dump())
finally:
    close_pipeline(prepared)
```

### Eval

Evaluate against the sample invoices in `data/sample-invoices/` using `data/sample-invoices/ground_truth.json`:

```bash
uv run python local-invoice-parser/eval.py --mode simple --experiment exp1_ocr_ner_gliner2
uv run python local-invoice-parser/eval.py --mode simple --experiment exp2_ocr_ner_qwen3
uv run python local-invoice-parser/eval.py --mode simple --experiment exp3_vlm_qwen25vl
uv run python local-invoice-parser/eval.py --mode simple --experiment exp4_vlm_minicpmv
uv run python local-invoice-parser/eval.py --mode simple --all

```

Reports are saved to `reports/`.

## Structure

```
local-invoice-parser/   # Source code
  pipeline.py           # Catalog-driven pipeline helpers
  experiments/          # Experiment catalog and metadata
  pipelines/            # Shared OCR+NER and VLM pipeline logic
  models/ner/           # NER backends (GLiNER2, llama_server)
  models/vlm/           # VLM backends (llama_server, llama_mtmd_cli)
  eval.py               # Evaluation script
  schema.py             # Pydantic output schema
scripts/                # llama-server serve scripts
data/                   # Sample invoices and ground truth
reports/                # Eval output reports
```
