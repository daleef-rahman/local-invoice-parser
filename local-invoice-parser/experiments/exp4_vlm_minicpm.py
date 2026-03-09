"""Experiment 4: MiniCPM-V via llama.cpp."""

from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.base import BaseExperiment, ExperimentResult, build_common_parser, run_experiment_cli
from pipeline.vlm import Config, print_results, run_pipeline, save_output


class Exp4VLMMiniCPMV(BaseExperiment):
    experiment_id = "exp4_vlm_minicpmv"
    description = "Invoice parser: MiniCPM-V via llama.cpp"

    def __init__(self, llama_url: str = "http://localhost:8081"):
        self.llama_url = llama_url

    def run(self, image_path: str) -> ExperimentResult:
        cfg = Config(vlm_backend="minicpmv", vlm_backend_kwargs={"base_url": self.llama_url})
        receipt, timings = run_pipeline(image_path, cfg)
        return ExperimentResult(receipt=receipt, timings=timings)

    def print_result(self, result: ExperimentResult) -> None:
        print_results(result.receipt, result.timings)

    def save_output(self, output_path: str, image_path: str, result: ExperimentResult) -> None:
        save_output(
            output_path=output_path,
            image_path=image_path,
            receipt=result.receipt,
            timings=result.timings,
        )


def main() -> None:
    parser = build_common_parser(Exp4VLMMiniCPMV.description)
    parser.add_argument("--llama-url", default="http://localhost:8081", help="llama.cpp server URL")
    args = parser.parse_args()
    run_experiment_cli(
        Exp4VLMMiniCPMV(llama_url=args.llama_url),
        image_path=args.image,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
