"""Experiment registry and factory."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any

from experiments.base import BaseExperiment

EXPERIMENTS: dict[str, str] = {
    "exp1_ocr_ner_gliner2": "experiments.exp1_paddleocr_gliner2ner:Exp1PaddleOCRGLiNER2",
    "exp2_ocr_ner_qwen3": "experiments.exp2_paddleocr_qwen3ner:Exp2PaddleOCRQwen3",
    "exp3_vlm_qwen25vl": "experiments.exp3_vlm_qwen25vl:Exp3VLMQwen25VL",
    "exp4_vlm_minicpmv": "experiments.exp4_vlm_minicpm:Exp4VLMMiniCPMV",
}

# Backward compatibility for old eval.py path-based names.
EXPERIMENT_ALIASES: dict[str, str] = {
    "exp1_paddleocr_gliner2ner": "exp1_ocr_ner_gliner2",
    "exp2_paddleocr_qwen3ner": "exp2_ocr_ner_qwen3",
    "exp3_vlm_qwen25vl": "exp3_vlm_qwen25vl",
    "exp4_vlm_minicpm": "exp4_vlm_minicpmv",
}


def _coerce_value(value: str) -> Any:
    lowered = value.strip().casefold()
    if lowered in {"true", "false"}:
        return lowered == "true"

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    if value.startswith(("{", "[", "\"")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    return value


def parse_constructor_kwargs(pairs: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid --experiment-param '{pair}'. Expected key=value.")
        key, value = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --experiment-param '{pair}'. Empty key.")
        out[key] = _coerce_value(value.strip())
    return out


def resolve_experiment_id(value: str) -> str:
    if value in EXPERIMENT_ALIASES:
        return EXPERIMENT_ALIASES[value]
    stem = Path(value).stem
    return EXPERIMENT_ALIASES.get(stem, value)


def create_experiment(experiment_id: str, **kwargs: Any) -> BaseExperiment:
    canonical = resolve_experiment_id(experiment_id)
    try:
        target = EXPERIMENTS[canonical]
    except KeyError as exc:
        choices = ", ".join(sorted(EXPERIMENTS))
        raise ValueError(f"Unknown experiment '{experiment_id}'. Choose from: {choices}") from exc
    module_name, class_name = target.split(":", 1)
    cls = getattr(import_module(module_name), class_name)
    return cls(**kwargs)
