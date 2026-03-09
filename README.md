# Local Invoice Parser

Experiments for locally-running invoice/receipt parsing. Compares two approaches:

- **OCR + NER**: PaddleOCR extracts text, a NER model extracts structured fields
- **VLM**: A vision-language model reads the image directly and returns structured fields

Extracted fields are defined in [schema.py](local-invoice-parser/schema.py) (`AdvancedReceiptData`): total amount, tax, merchant info, line items, and more.

## Experiments

| # | File | Approach | Model |
|---|------|----------|-------|
| 1 | `exp1_paddleocr_gliner2ner.py` | OCR + NER | PaddleOCR + GLiNER2 |
| 2 | `exp2_paddleocr_qwen3ner.py` | OCR + NER | PaddleOCR + Qwen3-4B |
| 3 | `exp3_vlm_qwen25vl.py` | VLM | Qwen2.5-VL-7B |
| 4 | `exp4_vlm_minicpm.py` | VLM | MiniCPM-V-4.5 |

## Results

Evaluated on 5 sample invoices (`data/sample-invoices/`).

| # | Experiment | Avg time/invoice | Scalar acc | Line-item acc | Overall acc |
|---|-----------|-----------------|------------|---------------|-------------|
| 1 | PaddleOCR + GLiNER2 | 7.5s | 0.4250 | 0.4870 | 0.4615 |
| 2 | PaddleOCR + Qwen3-4B | 14.3s | 0.4625 | 0.2875 | 0.3458 |
| 3 | Qwen2.5-VL-7B | 10.9s | 0.6375 | 0.6900 | 0.6667 |
| 4 | MiniCPM-V-4.5 | 14.1s | 0.5875 | 0.6700 | 0.6333 |

## Setup

```bash
uv sync
```

Qwen3/Exp3 require a running [llama.cpp](https://github.com/ggerganov/llama.cpp) server. Use the provided scripts to download models and start servers:

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

Run a single experiment on one image:

```bash
uv run python local-invoice-parser/experiments/exp1_paddleocr_gliner2ner.py --image data/sample-invoices/1.png
uv run python local-invoice-parser/experiments/exp2_paddleocr_qwen3ner.py --image data/sample-invoices/1.png
uv run python local-invoice-parser/experiments/exp3_vlm_qwen25vl.py --image data/sample-invoices/1.png
uv run python local-invoice-parser/experiments/exp4_vlm_minicpm.py --image data/sample-invoices/1.png

# Optional explicit binary/model paths for Exp 4
uv run python local-invoice-parser/experiments/exp4_vlm_minicpm.py --image data/sample-invoices/1.png --mtmd-bin /path/to/llama-mtmd-cli --model-path /path/to/MiniCPM-V-4_5-Q4_K_M.gguf --mmproj-path /path/to/mmproj-model-f16.gguf
```

### Eval

Evaluate against the sample invoices in `data/sample-invoices/` using `data/sample-invoices/ground_truth.json`:

```bash
uv run python local-invoice-parser/eval.py --mode simple --experiment exp1_ocr_ner_gliner2
uv run python local-invoice-parser/eval.py --mode simple --experiment exp2_ocr_ner_qwen3
uv run python local-invoice-parser/eval.py --mode simple --experiment exp3_vlm_qwen25vl
uv run python local-invoice-parser/eval.py --mode simple --experiment exp4_vlm_minicpmv

# pass constructor params when needed
uv run python local-invoice-parser/eval.py --mode simple --experiment exp2_ocr_ner_qwen3 --experiment-param llama_url=http://localhost:8080/v1
```

Reports are saved to `reports/`.

## Structure

```
local-invoice-parser/   # Source code
  experiments/          # Class-based experiments + registry
  pipeline/             # Shared OCR+NER and VLM pipeline logic
  models/ner/           # NER backends (GLiNER2, Qwen3)
  models/vlm/           # VLM backends (Qwen2.5-VL, MiniCPM-V)
  eval.py               # Evaluation script
  schema.py             # Pydantic output schema
scripts/                # llama-server serve scripts
data/                   # Sample invoices and ground truth
reports/                # Eval output reports
```
