"""
Experiment 2: PaddleOCR + Qwen3 NER pipeline for invoice parsing.
Self-sufficient: starts llama-server automatically if not already running.

Run:
    python exp2_paddleocr_qwen3ner.py --image invoice.jpg
    python exp2_paddleocr_qwen3ner.py --image invoice.jpg --llama-url http://localhost:9090/v1
"""
import sys
import json
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from pipeline.ocr_ner import Config, run_pipeline, print_results, save_output

parser = argparse.ArgumentParser(description="Invoice parser: PaddleOCR + Qwen3 NER")
parser.add_argument("--image", required=True, help="Path to invoice image")
parser.add_argument("--llama-url", default="http://localhost:8080/v1", help="llama.cpp server URL")
args = parser.parse_args()

cfg = Config(ner_backend="qwen3", ner_backend_kwargs={"base_url": args.llama_url})
receipt, ocr_regions, text, timings = run_pipeline(args.image, cfg)

print_results(receipt, text, timings)

output_path = str(
    Path(__file__).resolve().parent.parent.parent
    / "reports"
    / f"exp2_paddleocr_qwen3ner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
)
save_output(output_path, args.image, receipt, ocr_regions, text, timings)

payload = {"image": args.image, "timings": timings, "receipt": receipt.model_dump()}
print(json.dumps(payload))
