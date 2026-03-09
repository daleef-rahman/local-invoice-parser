"""Experiment 1: PaddleOCR + GLiNER2 NER pipeline."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.base import BaseExperiment, ExperimentResult, build_common_parser, run_experiment_cli
from pipeline.ocr_ner import Config, print_results, run_pipeline, save_output


class Exp1PaddleOCRGLiNER2(BaseExperiment):
    experiment_id = "exp1_ocr_ner_gliner2"
    description = "Invoice parser: PaddleOCR + GLiNER2"

    def run(self, image_path: str) -> ExperimentResult:
        cfg = Config(ner_backend="gliner2")
        receipt, ocr_regions, text, timings = run_pipeline(image_path, cfg)
        return ExperimentResult(
            receipt=receipt,
            timings=timings,
            artifacts={"ocr_regions": ocr_regions, "text": text},
        )

    def print_result(self, result: ExperimentResult) -> None:
        print_results(result.receipt, result.artifacts["text"], result.timings)

    def save_output(self, output_path: str, image_path: str, result: ExperimentResult) -> None:
        save_output(
            output_path=output_path,
            image_path=image_path,
            receipt=result.receipt,
            ocr_regions=result.artifacts["ocr_regions"],
            text=result.artifacts["text"],
            timings=result.timings,
        )


def main() -> None:
    parser = build_common_parser(Exp1PaddleOCRGLiNER2.description)
    args = parser.parse_args()
    run_experiment_cli(Exp1PaddleOCRGLiNER2(), image_path=args.image, output_path=args.output)


if __name__ == "__main__":
    main()
