"""Class-based experiment abstractions."""

from __future__ import annotations

import argparse
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from schema import AdvancedReceiptData


@dataclass
class ExperimentResult:
    receipt: AdvancedReceiptData
    timings: dict[str, Any]
    artifacts: dict[str, Any] = field(default_factory=dict)


class BaseExperiment(ABC):
    experiment_id: str
    description: str

    @abstractmethod
    def run(self, image_path: str) -> ExperimentResult:
        raise NotImplementedError

    @abstractmethod
    def print_result(self, result: ExperimentResult) -> None:
        raise NotImplementedError

    @abstractmethod
    def save_output(self, output_path: str, image_path: str, result: ExperimentResult) -> None:
        raise NotImplementedError


def make_default_report_path(experiment_id: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return repo_root / "reports" / f"{experiment_id}_{timestamp}.json"


def run_experiment_cli(
    experiment: BaseExperiment,
    image_path: str,
    output_path: str | None = None,
) -> None:
    result = experiment.run(image_path=image_path)
    experiment.print_result(result)

    out = output_path or str(make_default_report_path(experiment.experiment_id))
    experiment.save_output(output_path=out, image_path=image_path, result=result)

    payload = {"image": image_path, "timings": result.timings, "receipt": result.receipt.model_dump()}
    print(json.dumps(payload))


def build_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--image", required=True, help="Path to invoice image")
    parser.add_argument("--output", default=None, help="Optional output report JSON path")
    return parser
