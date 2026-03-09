"""
Experiment 1: OCR + NER pipeline for invoice parsing.
Uses PaddleOCR for text extraction and a pluggable NER backend for structured JSON extraction.

Run:
    python exp1_ocr_ner.py --image invoice.jpg
    python exp1_ocr_ner.py --image invoice.jpg --ner-backend qwen3
    python exp1_ocr_ner.py --image invoice.jpg --output results.json

NER backends:
    gliner2  (default)  GLiNER2 model from HuggingFace
    qwen3               Qwen3-4B via llama.cpp server (http://localhost:8080)
"""
import json
import time
import argparse
from dataclasses import dataclass, field
from pathlib import Path

from paddleocr import PaddleOCR

from schema import AdvancedReceiptData
from ner import BACKENDS, NERBackend, get_backend

# --- Config ---

@dataclass
class Config:
    # OCR
    ocr_lang: str = "en"
    ocr_use_textline_orientation: bool = True

    # NER
    ner_backend: str = "gliner2"
    ner_backend_kwargs: dict = field(default_factory=dict)

# --- Model loading ---

def load_ocr(cfg: Config) -> PaddleOCR:
    return PaddleOCR(
        use_textline_orientation=cfg.ocr_use_textline_orientation,
        lang=cfg.ocr_lang,
    )

def load_ner(cfg: Config) -> NERBackend:
    backend_cls = get_backend(cfg.ner_backend)
    return backend_cls(**cfg.ner_backend_kwargs)

# --- Pipeline ---

def extract_text(ocr: PaddleOCR, image_path: str) -> tuple[str, list[dict]]:
    """Run PaddleOCR on image, return (full_text, raw_regions)."""
    result = ocr.predict(image_path)
    lines, raw = [], []
    for page in result:
        for text, score, box in zip(page["rec_texts"], page["rec_scores"], page["rec_polys"]):
            lines.append(text)
            raw.append({"text": text, "confidence": round(float(score), 4), "box": box.tolist()})
    return "\n".join(lines), raw

def run_pipeline(image_path: str, cfg: Config) -> tuple[AdvancedReceiptData, list[dict], str, dict]:
    """Full OCR → NER pipeline for a single invoice image."""
    print("Loading models...")
    ocr = load_ocr(cfg)
    ner = load_ner(cfg)

    print(f"Extracting text from: {image_path}")
    t0 = time.perf_counter()
    text, ocr_regions = extract_text(ocr, image_path)
    t_ocr = time.perf_counter()
    print(f"  → {len(ocr_regions)} text regions, {len(text)} chars")

    print(f"Running structured extraction ({cfg.ner_backend})...")
    receipt = ner.extract(text)
    t_ner = time.perf_counter()
    print(f"  → {len(receipt.productLineItems)} line items found")

    ner.close()

    timings = {
        "ocr_s": round(t_ocr - t0, 3),
        "ner_s": round(t_ner - t_ocr, 3),
        "total_s": round(t_ner - t0, 3),
    }
    return receipt, ocr_regions, text, timings

# --- CLI ---

parser = argparse.ArgumentParser(description="Invoice OCR + NER pipeline (PaddleOCR + pluggable NER)")
parser.add_argument("--image", type=str, required=True, help="Path to invoice image")
parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
parser.add_argument("--ner-backend", type=str, default="gliner2", choices=list(BACKENDS), help="NER backend to use")
parser.add_argument("--ner-model", type=str, default=None, help="Model id/path for the NER backend (backend-specific)")
parser.add_argument("--llama-url", type=str, default="http://localhost:8080/v1", help="llama.cpp server URL (qwen3 backend)")
args = parser.parse_args()

ner_backend_kwargs = {}
if args.ner_backend == "gliner2" and args.ner_model:
    ner_backend_kwargs["model"] = args.ner_model
elif args.ner_backend == "qwen3":
    ner_backend_kwargs["base_url"] = args.llama_url
    if args.ner_model:
        ner_backend_kwargs["model"] = args.ner_model

cfg = Config(ner_backend=args.ner_backend, ner_backend_kwargs=ner_backend_kwargs)

receipt, ocr_regions, text, timings = run_pipeline(args.image, cfg)

# Print timings
print("\n--- Timings ---")
print(f"  ocr   : {timings['ocr_s']:.3f}s")
print(f"  ner   : {timings['ner_s']:.3f}s")
print(f"  total : {timings['total_s']:.3f}s")

# Print OCR text
print("\n--- OCR Text ---")
print(text)

# Print extracted fields
print("\n--- Receipt ---")
for field, value in receipt.model_dump().items():
    if field == "productLineItems":
        continue
    if value is not None:
        print(f"  {field}: {value}")

if receipt.productLineItems:
    print("\n--- Line Items ---")
    for i, item in enumerate(receipt.productLineItems, 1):
        print(f"  [{i}] {item.model_dump()}")

# Save output
if args.output:
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "image": args.image,
        "timings": timings,
        "text": text,
        "ocr_regions": ocr_regions,
        "receipt": receipt.model_dump(),
    }, indent=2))
    print(f"\nResults saved to {out_path}")
