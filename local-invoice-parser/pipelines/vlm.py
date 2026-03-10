"""Shared VLM pipeline for invoice parsing."""

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from schema import AdvancedReceiptData
from models.modelbackend import ModelBackend
from models.vlm import get_backend


@dataclass
class Config:
    vlm_backend: str = "llama_server"
    backend_config: dict[str, Any] = field(default_factory=dict)


def load_vlm(cfg: Config) -> ModelBackend:
    backend_cls = get_backend(cfg.vlm_backend)
    return backend_cls(**cfg.backend_config)


def run_pipeline(image_path: str, cfg: Config) -> tuple[AdvancedReceiptData, dict]:
    """Full VLM pipeline for a single invoice image."""
    print(f"Loading {cfg.vlm_backend} model...", file=sys.stderr)
    vlm = load_vlm(cfg)

    print(f"Extracting invoice data from: {image_path}", file=sys.stderr)
    t0 = time.perf_counter()
    receipt = vlm.extract(image_path)
    t_vlm = time.perf_counter()
    print(f"  → {len(receipt.productLineItems)} line items found", file=sys.stderr)

    vlm.close()

    timings = {
        "vlm_s": round(t_vlm - t0, 3),
        "total_s": round(t_vlm - t0, 3),
    }
    return receipt, timings


def print_results(receipt: AdvancedReceiptData, timings: dict) -> None:
    p = lambda *a, **kw: print(*a, **kw, file=sys.stderr)
    p("\n--- Timings ---")
    p(f"  vlm   : {timings['vlm_s']:.3f}s")
    p(f"  total : {timings['total_s']:.3f}s")

    p("\n--- Receipt ---")
    for fname, value in receipt.model_dump().items():
        if fname == "productLineItems":
            continue
        if value is not None:
            p(f"  {fname}: {value}")

    if receipt.productLineItems:
        p("\n--- Line Items ---")
        for i, item in enumerate(receipt.productLineItems, 1):
            p(f"  [{i}] {item.model_dump()}")


def save_output(
    output_path: str,
    image_path: str,
    receipt: AdvancedReceiptData,
    timings: dict,
) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "image": image_path,
        "timings": timings,
        "receipt": receipt.model_dump(),
    }, indent=2))
    print(f"\nResults saved to {out}", file=sys.stderr)
