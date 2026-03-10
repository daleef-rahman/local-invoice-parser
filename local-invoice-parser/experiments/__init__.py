"""Experiment metadata package."""

from experiments.catalog import (
    EXPERIMENT_ALIASES,
    EXPERIMENT_SPECS,
    get_experiment_spec,
    resolve_experiment_id,
)

__all__ = [
    "EXPERIMENT_ALIASES",
    "EXPERIMENT_SPECS",
    "get_experiment_spec",
    "resolve_experiment_id",
]
