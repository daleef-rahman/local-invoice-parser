"""Catalog-driven pipeline helpers."""

from __future__ import annotations

import json
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image
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
    image_max_side: int | None = 1600
    ner_backend: str = "gliner2"
    backend_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class VlmConfig:
    image_max_side: int | None = 1600
    vlm_backend: str = "llama_server"
    backend_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreparedImage:
    path: str
    temp_path: Path | None
    original_size: tuple[int, int]
    processed_size: tuple[int, int]


def _prepare_image(image_path: str, *, max_image_side: int | None) -> PreparedImage:
    src_path = Path(image_path)
    with Image.open(src_path) as img:
        original_size = img.size

        if max_image_side is None or max(original_size) <= max_image_side:
            return PreparedImage(
                path=image_path,
                temp_path=None,
                original_size=original_size,
                processed_size=original_size,
            )

        resized = img.convert("RGB")
        resized.thumbnail((max_image_side, max_image_side), Image.LANCZOS)
        processed_size = resized.size

        tmp = tempfile.NamedTemporaryFile(
            prefix="pipeline_input_",
            suffix=src_path.suffix or ".png",
            delete=False,
        )
        tmp_path = Path(tmp.name)
        tmp.close()
        resized.save(tmp_path)

    return PreparedImage(
        path=str(tmp_path),
        temp_path=tmp_path,
        original_size=original_size,
        processed_size=processed_size,
    )


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
) -> tuple[AdvancedReceiptData, list[dict[str, Any]], str, dict[str, Any]]:
    owns_models = ocr is None or ner is None
    if owns_models:
        print("Loading models...", file=sys.stderr)
        if ocr is None:
            ocr = load_ocr(cfg)
        if ner is None:
            ner = load_ner(cfg)

    t0 = time.perf_counter()
    prepared_image = _prepare_image(image_path, max_image_side=cfg.image_max_side)
    t_preprocess = time.perf_counter()
    try:
        print(f"Extracting text from: {prepared_image.path}", file=sys.stderr)
        text, ocr_regions = extract_ocr_text(
            ocr,
            prepared_image.path,
        )
        t_ocr = time.perf_counter()
        if prepared_image.original_size != prepared_image.processed_size:
            print(
                f"  -> resized pipeline input: {prepared_image.original_size} -> {prepared_image.processed_size}",
                file=sys.stderr,
            )
        print(f"  -> {len(ocr_regions)} text regions, {len(text)} chars", file=sys.stderr)

        print(f"Running structured extraction ({cfg.ner_backend})...", file=sys.stderr)
        receipt = ner.extract(text)
        t_ner = time.perf_counter()
        print(f"  -> {len(receipt.productLineItems)} line items found", file=sys.stderr)
    finally:
        if prepared_image.temp_path is not None:
            prepared_image.temp_path.unlink(missing_ok=True)

    if owns_models:
        ner.close()

    timings = {
        "preprocess_s": round(t_preprocess - t0, 3),
        "ocr_s": round(t_ocr - t_preprocess, 3),
        "ner_s": round(t_ner - t_ocr, 3),
        "total_s": round(t_ner - t0, 3),
    }
    artifacts = {
        "image_preprocessing": {
            "source_size": list(prepared_image.original_size),
            "model_input_size": list(prepared_image.processed_size),
        }
    }
    return receipt, ocr_regions, text, timings, artifacts


def run_vlm_pipeline(
    image_path: str,
    cfg: VlmConfig,
    vlm: ModelBackend | None = None,
) -> tuple[AdvancedReceiptData, dict[str, Any], dict[str, Any]]:
    owns_model = vlm is None
    if owns_model:
        print(f"Loading {cfg.vlm_backend} model...", file=sys.stderr)
        vlm = load_vlm(cfg)

    t0 = time.perf_counter()
    prepared_image = _prepare_image(image_path, max_image_side=cfg.image_max_side)
    t_preprocess = time.perf_counter()
    try:
        print(f"Extracting invoice data from: {prepared_image.path}", file=sys.stderr)
        receipt = vlm.extract(prepared_image.path)
        t_vlm = time.perf_counter()
        if prepared_image.original_size != prepared_image.processed_size:
            print(
                f"  -> resized pipeline input: {prepared_image.original_size} -> {prepared_image.processed_size}",
                file=sys.stderr,
            )
        print(f"  -> {len(receipt.productLineItems)} line items found", file=sys.stderr)
    finally:
        if prepared_image.temp_path is not None:
            prepared_image.temp_path.unlink(missing_ok=True)

    if owns_model:
        vlm.close()

    timings = {
        "preprocess_s": round(t_preprocess - t0, 3),
        "vlm_s": round(t_vlm - t_preprocess, 3),
        "total_s": round(t_vlm - t0, 3),
    }
    artifacts = {
        "image_preprocessing": {
            "source_size": list(prepared_image.original_size),
            "model_input_size": list(prepared_image.processed_size),
        }
    }
    return receipt, timings, artifacts


def print_results(result: PipelineResult) -> None:
    p = lambda *a, **kw: print(*a, **kw, file=sys.stderr)
    timings = result.timings

    p("\n--- Timings ---")
    for key in ("preprocess_s", "ocr_s", "ner_s", "vlm_s", "total_s"):
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
        ocr_defaults = dict(spec.ocr_defaults)
        ocr_cfg = OcrNerConfig(
            ner_backend=spec.backend,
            backend_config=dict(spec.backend_config),
            image_max_side=ocr_defaults.pop("ocr_max_image_side", 1600),
            **ocr_defaults,
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
        receipt, ocr_regions, text, timings, artifacts = run_ocr_ner_pipeline(
            image_path,
            prepared.ocr_cfg,
            ocr=prepared.ocr,
            ner=prepared.ner,
        )
        return PipelineResult(
            receipt=receipt,
            timings=timings,
            artifacts={**artifacts, "ocr_regions": ocr_regions, "text": text},
        )

    if prepared.spec.pipeline == "vlm":
        receipt, timings, artifacts = run_vlm_pipeline(
            image_path,
            prepared.vlm_cfg,
            vlm=prepared.vlm,
        )
        return PipelineResult(
            receipt=receipt,
            timings=timings,
            artifacts=artifacts,
        )

    raise ValueError(f"Unsupported pipeline: {prepared.spec.pipeline}")


def close_pipeline(prepared: PreparedPipeline) -> None:
    if prepared.ner is not None:
        prepared.ner.close()
    if prepared.vlm is not None:
        prepared.vlm.close()
