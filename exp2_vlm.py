"""
Experiment 2: VLM pipeline for invoice parsing.
Feeds the invoice image directly to a Vision Language Model — no separate OCR step.

Run:
    python exp2_vlm.py --image invoice.jpg
    python exp2_vlm.py --image invoice.jpg --output results.json

VLM backends:
    llamacpp  (default)  Qwen2.5-VL-7B via llama.cpp server (./scripts/serve_qwen25vl.sh)
    minicpmv             MiniCPM-V-4.5 via llama.cpp server on :8081 (./scripts/serve_minicpmv.sh)
    qwen25vl             Qwen2.5-VL-7B-Instruct loaded locally via transformers
"""
import json
import time
import argparse
from dataclasses import dataclass, field
from pathlib import Path

from schema import AdvancedReceiptData
from vlm import BACKENDS, VLMBackend, get_backend

# --- Config ---

@dataclass
class Config:
    vlm_backend: str = "llamacpp"
    vlm_backend_kwargs: dict = field(default_factory=dict)

# --- Model loading ---

def load_vlm(cfg: Config) -> VLMBackend:
    backend_cls = get_backend(cfg.vlm_backend)
    return backend_cls(**cfg.vlm_backend_kwargs)

# --- Pipeline ---

def run_pipeline(image_path: str, cfg: Config) -> tuple[AdvancedReceiptData, str, dict]:
    """Full VLM pipeline for a single invoice image."""
    print(f"Loading {cfg.vlm_backend} model...")
    vlm = load_vlm(cfg)

    print(f"Extracting invoice data from: {image_path}")
    t0 = time.perf_counter()
    receipt = vlm.extract(image_path)
    t_vlm = time.perf_counter()
    print(f"  → {len(receipt.productLineItems)} line items found")

    vlm.close()

    timings = {
        "vlm_s": round(t_vlm - t0, 3),
        "total_s": round(t_vlm - t0, 3),
    }
    return receipt, timings

# --- CLI ---

parser = argparse.ArgumentParser(description="Invoice VLM pipeline (pluggable VLM backend)")
parser.add_argument("--image", type=str, required=True, help="Path to invoice image")
parser.add_argument("--output", type=str, default=None, help="Save results to JSON file")
parser.add_argument("--vlm-backend", type=str, default="llamacpp", choices=list(BACKENDS), help="VLM backend to use")
parser.add_argument("--vlm-model", type=str, default=None, help="HuggingFace model id (backend-specific)")
args = parser.parse_args()

vlm_backend_kwargs = {}
if args.vlm_model:
    vlm_backend_kwargs["model"] = args.vlm_model

cfg = Config(vlm_backend=args.vlm_backend, vlm_backend_kwargs=vlm_backend_kwargs)

receipt, timings = run_pipeline(args.image, cfg)

# Print timings
print("\n--- Timings ---")
print(f"  vlm   : {timings['vlm_s']:.3f}s")
print(f"  total : {timings['total_s']:.3f}s")

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
        "receipt": receipt.model_dump(),
    }, indent=2))
    print(f"\nResults saved to {out_path}")
