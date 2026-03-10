"""Catalog-driven pipeline helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from experiments.catalog import ExperimentSpec, get_experiment_spec
from schema import AdvancedReceiptData


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


def prepare_pipeline(experiment_id: str) -> PreparedPipeline:
    spec = get_experiment_spec(experiment_id)

    if spec.pipeline == "ocr_ner":
        from pipelines import ocr_ner as ocr_ner_pipeline

        ocr_cfg = ocr_ner_pipeline.Config(
            ner_backend=spec.backend,
            backend_config=dict(spec.backend_config),
            **spec.ocr_defaults,
        )
        return PreparedPipeline(
            spec=spec,
            ocr_cfg=ocr_cfg,
            ocr=ocr_ner_pipeline.load_ocr(ocr_cfg),
            ner=ocr_ner_pipeline.load_ner(ocr_cfg),
        )

    if spec.pipeline == "vlm":
        from pipelines import vlm as vlm_pipeline

        vlm_cfg = vlm_pipeline.Config(
            vlm_backend=spec.backend,
            backend_config=dict(spec.backend_config),
        )
        return PreparedPipeline(
            spec=spec,
            vlm_cfg=vlm_cfg,
            vlm=vlm_pipeline.load_vlm(vlm_cfg),
        )

    raise ValueError(f"Unsupported pipeline: {spec.pipeline}")


def run_pipeline(prepared: PreparedPipeline, image_path: str) -> PipelineResult:
    if prepared.spec.pipeline == "ocr_ner":
        from pipelines import ocr_ner as ocr_ner_pipeline

        receipt, ocr_regions, text, timings = ocr_ner_pipeline.run_pipeline(
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
        t0 = time.perf_counter()
        receipt = prepared.vlm.extract(image_path)
        elapsed = round(time.perf_counter() - t0, 3)
        return PipelineResult(
            receipt=receipt,
            timings={"vlm_s": elapsed, "total_s": elapsed},
            artifacts={},
        )

    raise ValueError(f"Unsupported pipeline: {prepared.spec.pipeline}")


def close_pipeline(prepared: PreparedPipeline) -> None:
    if prepared.ner is not None:
        prepared.ner.close()
    if prepared.vlm is not None:
        prepared.vlm.close()
