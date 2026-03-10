"""Experiment registry and factory."""

from __future__ import annotations

import json
import inspect
from importlib import import_module
from pathlib import Path
from typing import Any

from experiments.base import BaseExperiment
from experiments.catalog import EXPERIMENT_ALIASES, EXPERIMENT_SPECS

EXPERIMENTS: dict[str, str] = {
    experiment_id: spec.target for experiment_id, spec in EXPERIMENT_SPECS.items()
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
        spec = EXPERIMENT_SPECS[canonical]
    except KeyError as exc:
        choices = ", ".join(sorted(EXPERIMENTS))
        raise ValueError(f"Unknown experiment '{experiment_id}'. Choose from: {choices}") from exc
    target = spec.target
    module_name, class_name = target.split(":", 1)
    cls = getattr(import_module(module_name), class_name)
    constructor_kwargs = dict(spec.constructor_defaults)
    constructor_kwargs.update(kwargs)
    signature = inspect.signature(cls.__init__)
    accepts_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if not accepts_var_kwargs:
        allowed = {
            name
            for name, param in signature.parameters.items()
            if name != "self"
            and param.kind in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        }
        constructor_kwargs = {
            key: value for key, value in constructor_kwargs.items() if key in allowed
        }
    return cls(**constructor_kwargs)
