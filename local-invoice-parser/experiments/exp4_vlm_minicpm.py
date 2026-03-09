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
    description = "Invoice parser: MiniCPM-V via llama-mtmd-cli"

    def __init__(
        self,
        mtmd_bin: str = "llama-mtmd-cli",
        model_path: str | None = None,
        mmproj_path: str | None = None,
        debug: bool = False,
    ):
        self.mtmd_bin = mtmd_bin
        self.model_path = model_path
        self.mmproj_path = mmproj_path
        self.debug = debug

    def run(self, image_path: str) -> ExperimentResult:
        cfg = Config(
            vlm_backend="minicpmv",
            vlm_backend_kwargs={
                "mtmd_bin": self.mtmd_bin,
                "model_path": self.model_path,
                "mmproj_path": self.mmproj_path,
                "debug": self.debug,
            },
        )
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
    parser.add_argument("--mtmd-bin", default="llama-mtmd-cli", help="Path to llama-mtmd-cli binary")
    parser.add_argument("--model-path", default=None, help="Path to MiniCPM-V gguf model")
    parser.add_argument("--mmproj-path", default=None, help="Path to MiniCPM-V mmproj file")
    parser.add_argument("--debug", action="store_true", help="Enable detailed backend diagnostics")
    args = parser.parse_args()
    run_experiment_cli(
        Exp4VLMMiniCPMV(
            mtmd_bin=args.mtmd_bin,
            model_path=args.model_path,
            mmproj_path=args.mmproj_path,
            debug=args.debug,
        ),
        image_path=args.image,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
