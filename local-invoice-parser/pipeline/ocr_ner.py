"""Shared PaddleOCR + NER pipeline for invoice parsing."""

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from paddleocr import PaddleOCR

from schema import AdvancedReceiptData
from models.ner import NERBackend, get_backend


@dataclass
class Config:
    ocr_lang: str = "en"
    ocr_use_textline_orientation: bool = True
    ner_backend: str = "gliner2"
    ner_backend_kwargs: dict = field(default_factory=dict)


def load_ocr(cfg: Config) -> PaddleOCR:
    return PaddleOCR(
        use_textline_orientation=cfg.ocr_use_textline_orientation,
        lang=cfg.ocr_lang,
    )


def load_ner(cfg: Config) -> NERBackend:
    backend_cls = get_backend(cfg.ner_backend)
    return backend_cls(**cfg.ner_backend_kwargs)


def extract_text(ocr: PaddleOCR, image_path: str) -> tuple[str, list[dict]]:
    result = ocr.predict(image_path)
    lines, raw = [], []
    for page in result:
        for text, score, box in zip(page["rec_texts"], page["rec_scores"], page["rec_polys"]):
            lines.append(text)
            raw.append({"text": text, "confidence": round(float(score), 4), "box": box.tolist()})
    return "\n".join(lines), raw


def run_pipeline(
    image_path: str,
    cfg: Config,
    ocr: PaddleOCR | None = None,
    ner: NERBackend | None = None,
) -> tuple[AdvancedReceiptData, list[dict], str, dict]:
    """Full OCR → NER pipeline for a single invoice image.

    If *ocr* and *ner* are provided they are reused (no reload); callers that
    pass pre-loaded models are responsible for closing *ner* themselves.
    """
    owns_models = ocr is None or ner is None
    if owns_models:
        print("Loading models...", file=sys.stderr)
        if ocr is None:
            ocr = load_ocr(cfg)
        if ner is None:
            ner = load_ner(cfg)

    print(f"Extracting text from: {image_path}", file=sys.stderr)
    t0 = time.perf_counter()
    text, ocr_regions = extract_text(ocr, image_path)
    t_ocr = time.perf_counter()
    print(f"  → {len(ocr_regions)} text regions, {len(text)} chars", file=sys.stderr)

    print(f"Running structured extraction ({cfg.ner_backend})...", file=sys.stderr)
    receipt = ner.extract(text)
    t_ner = time.perf_counter()
    print(f"  → {len(receipt.productLineItems)} line items found", file=sys.stderr)

    if owns_models:
        ner.close()

    timings = {
        "ocr_s": round(t_ocr - t0, 3),
        "ner_s": round(t_ner - t_ocr, 3),
        "total_s": round(t_ner - t0, 3),
    }
    return receipt, ocr_regions, text, timings


def print_results(receipt: AdvancedReceiptData, text: str, timings: dict) -> None:
    p = lambda *a, **kw: print(*a, **kw, file=sys.stderr)
    p("\n--- Timings ---")
    p(f"  ocr   : {timings['ocr_s']:.3f}s")
    p(f"  ner   : {timings['ner_s']:.3f}s")
    p(f"  total : {timings['total_s']:.3f}s")

    p("\n--- OCR Text ---")
    p(text)

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
    ocr_regions: list[dict],
    text: str,
    timings: dict,
) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "image": image_path,
        "timings": timings,
        "text": text,
        "ocr_regions": ocr_regions,
        "receipt": receipt.model_dump(),
    }, indent=2))
    print(f"\nResults saved to {out}", file=sys.stderr)
