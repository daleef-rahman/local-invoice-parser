"""Catalog-driven pipeline helpers."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.catalog import ExperimentSpec, get_experiment_spec
from models.ocr import extract_text as extract_ocr_text
from models.ocr import load_backend as load_ocr_backend
from models.modelbackend import ModelBackend
from models.ner import get_backend as get_ner_backend
from models.vlm import get_backend as get_vlm_backend
from schema import AdvancedReceiptData
from paddleocr import PaddleOCR


@dataclass
class PipelineResult:
    receipt: AdvancedReceiptData
    timings: dict[str, Any]
    artifacts: dict[str, Any]


@dataclass
class PreparedPipeline:
    spec: ExperimentSpec
    ocr_cfg: Any | None = None
    vlm_cfg: Any | None = None
    ocr: Any | None = None
    ner: Any | None = None
    vlm: Any | None = None


@dataclass
class OcrNerConfig:
    ocr_lang: str = "en"
    ocr_use_textline_orientation: bool = True
    ocr_max_image_side: int | None = 1600
    ner_backend: str = "gliner2"
    backend_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class VlmConfig:
    vlm_backend: str = "llama_server"
    backend_config: dict[str, Any] = field(default_factory=dict)


def load_ocr(cfg: OcrNerConfig) -> PaddleOCR:
    return load_ocr_backend(
        lang=cfg.ocr_lang,
        use_textline_orientation=cfg.ocr_use_textline_orientation,
    )


def load_ner(cfg: OcrNerConfig) -> ModelBackend:
    backend_cls = get_ner_backend(cfg.ner_backend)
    return backend_cls(**cfg.backend_config)


def load_vlm(cfg: VlmConfig) -> ModelBackend:
    backend_cls = get_vlm_backend(cfg.vlm_backend)
    return backend_cls(**cfg.backend_config)


def run_ocr_ner_pipeline(
    image_path: str,
    cfg: OcrNerConfig,
    ocr: PaddleOCR | None = None,
    ner: ModelBackend | None = None,
) -> tuple[AdvancedReceiptData, list[dict[str, Any]], str, dict[str, float]]:
    owns_models = ocr is None or ner is None
    if owns_models:
        print("Loading models...", file=sys.stderr)
        if ocr is None:
            ocr = load_ocr(cfg)
        if ner is None:
            ner = load_ner(cfg)

    print(f"Extracting text from: {image_path}", file=sys.stderr)
    t0 = time.perf_counter()
    text, ocr_regions = extract_ocr_text(
        ocr,
        image_path,
        max_image_side=cfg.ocr_max_image_side,
    )
    t_ocr = time.perf_counter()
    if ocr_regions and "meta" in ocr_regions[0]:
        meta = ocr_regions[0]["meta"]
        print(
            f"  -> resized image for OCR: {tuple(meta['source_size'])} -> {tuple(meta['ocr_input_size'])}",
            file=sys.stderr,
        )
        ocr_regions = ocr_regions[1:]
    print(f"  -> {len(ocr_regions)} text regions, {len(text)} chars", file=sys.stderr)

    print(f"Running structured extraction ({cfg.ner_backend})...", file=sys.stderr)
    receipt = ner.extract(text)
    t_ner = time.perf_counter()
    print(f"  -> {len(receipt.productLineItems)} line items found", file=sys.stderr)

    if owns_models:
        ner.close()

    timings = {
        "ocr_s": round(t_ocr - t0, 3),
        "ner_s": round(t_ner - t_ocr, 3),
        "total_s": round(t_ner - t0, 3),
    }
    return receipt, ocr_regions, text, timings


def run_vlm_pipeline(
    image_path: str,
    cfg: VlmConfig,
    vlm: ModelBackend | None = None,
) -> tuple[AdvancedReceiptData, dict[str, float]]:
    owns_model = vlm is None
    if owns_model:
        print(f"Loading {cfg.vlm_backend} model...", file=sys.stderr)
        vlm = load_vlm(cfg)

    print(f"Extracting invoice data from: {image_path}", file=sys.stderr)
    t0 = time.perf_counter()
    receipt = vlm.extract(image_path)
    elapsed = round(time.perf_counter() - t0, 3)
    print(f"  -> {len(receipt.productLineItems)} line items found", file=sys.stderr)

    if owns_model:
        vlm.close()

    timings = {
        "vlm_s": elapsed,
        "total_s": elapsed,
    }
    return receipt, timings


def print_results(result: PipelineResult) -> None:
    p = lambda *a, **kw: print(*a, **kw, file=sys.stderr)
    timings = result.timings

    p("\n--- Timings ---")
    for key in ("ocr_s", "ner_s", "vlm_s", "total_s"):
        if key in timings:
            p(f"  {key.removesuffix('_s'):<5}: {timings[key]:.3f}s")

    if "text" in result.artifacts:
        p("\n--- OCR Text ---")
        p(result.artifacts["text"])

    p("\n--- Receipt ---")
    for fname, value in result.receipt.model_dump().items():
        if fname == "productLineItems":
            continue
        if value is not None:
            p(f"  {fname}: {value}")

    if result.receipt.productLineItems:
        p("\n--- Line Items ---")
        for i, item in enumerate(result.receipt.productLineItems, 1):
            p(f"  [{i}] {item.model_dump()}")


def save_output(output_path: str, image_path: str, result: PipelineResult) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "image": image_path,
        "timings": result.timings,
        "receipt": result.receipt.model_dump(),
    }
    payload.update(result.artifacts)

    out.write_text(json.dumps(payload, indent=2))
    print(f"\nResults saved to {out}", file=sys.stderr)


def prepare_pipeline(experiment_id: str) -> PreparedPipeline:
    spec = get_experiment_spec(experiment_id)

    if spec.pipeline == "ocr_ner":
        ocr_cfg = OcrNerConfig(
            ner_backend=spec.backend,
            backend_config=dict(spec.backend_config),
            **spec.ocr_defaults,
        )
        return PreparedPipeline(
            spec=spec,
            ocr_cfg=ocr_cfg,
            ocr=load_ocr(ocr_cfg),
            ner=load_ner(ocr_cfg),
        )

    if spec.pipeline == "vlm":
        vlm_cfg = VlmConfig(
            vlm_backend=spec.backend,
            backend_config=dict(spec.backend_config),
        )
        return PreparedPipeline(
            spec=spec,
            vlm_cfg=vlm_cfg,
            vlm=load_vlm(vlm_cfg),
        )

    raise ValueError(f"Unsupported pipeline: {spec.pipeline}")


def run_pipeline(prepared: PreparedPipeline, image_path: str) -> PipelineResult:
    if prepared.spec.pipeline == "ocr_ner":
        receipt, ocr_regions, text, timings = run_ocr_ner_pipeline(
            image_path,
            prepared.ocr_cfg,
            ocr=prepared.ocr,
            ner=prepared.ner,
        )
        return PipelineResult(
            receipt=receipt,
            timings=timings,
            artifacts={"ocr_regions": ocr_regions, "text": text},
        )

    if prepared.spec.pipeline == "vlm":
        receipt, timings = run_vlm_pipeline(
            image_path,
            prepared.vlm_cfg,
            vlm=prepared.vlm,
        )
        return PipelineResult(
            receipt=receipt,
            timings=timings,
            artifacts={},
        )

    raise ValueError(f"Unsupported pipeline: {prepared.spec.pipeline}")


def close_pipeline(prepared: PreparedPipeline) -> None:
    if prepared.ner is not None:
        prepared.ner.close()
    if prepared.vlm is not None:
        prepared.vlm.close()
