"""
Experiment 1: PaddleOCR + GLiNER2 NER pipeline for invoice parsing.

Run:
    python exp1_paddleocr_gliner2ner.py --image invoice.jpg
"""
import sys
import json
from pathlib import Path
from datetime import datetime
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

from pipeline.ocr_ner import Config, run_pipeline, print_results, save_output

parser = argparse.ArgumentParser(description="Invoice parser: PaddleOCR + GLiNER2")
parser.add_argument("--image", required=True, help="Path to invoice image")
args = parser.parse_args()

cfg = Config(ner_backend="gliner2")
receipt, ocr_regions, text, timings = run_pipeline(args.image, cfg)

print_results(receipt, text, timings)

output_path = str(
    Path(__file__).resolve().parent.parent.parent
    / "reports"
    / f"exp1_paddleocr_gliner2ner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
)
save_output(output_path, args.image, receipt, ocr_regions, text, timings)

payload = {"image": args.image, "timings": timings, "receipt": receipt.model_dump()}
print(json.dumps(payload))
