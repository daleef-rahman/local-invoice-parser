"""Central experiment catalog with runtime and backend metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HfAssetSpec:
    repo_id: str
    filenames: tuple[str, ...]
    local_dir: Path


@dataclass(frozen=True)
class RuntimeSpec:
    kind: str = "none"
    port: int | None = None
    model_path: Path | None = None
    mmproj_path: Path | None = None
    model_alias: str | None = None
    ctx_size: int = 4096
    hf_assets: HfAssetSpec | None = None


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    description: str
    pipeline: str
    backend: str
    backend_config: dict[str, Any] = field(default_factory=dict)
    ocr_defaults: dict[str, Any] = field(default_factory=dict)
    runtime: RuntimeSpec = field(default_factory=RuntimeSpec)
    aliases: tuple[str, ...] = ()


HOME = Path.home()

QWEN3_DIR = HOME / "models" / "qwen3-4b"
QWEN3VL2B_DIR = HOME / "models" / "qwen3vl-2b"
QWEN25VL3B_DIR = HOME / "models" / "qwen25vl-3b"
QWEN25VL_DIR = HOME / "models" / "qwen25vl-7b"
LFM25VL16B_DIR = HOME / "models" / "lfm25vl-1.6b"
SMOLVLM256M_DIR = HOME / "models" / "smolvlm-256m"
MINICPMV_DIR = HOME / "models" / "minicpmv-4.5"


EXPERIMENT_SPECS: dict[str, ExperimentSpec] = {
    "exp1_ocr_ner_gliner2": ExperimentSpec(
        experiment_id="exp1_ocr_ner_gliner2",
        description="Invoice parser: PaddleOCR + GLiNER2",
        pipeline="ocr_ner",
        backend="gliner2",
        ocr_defaults={"ocr_lang": "en", "ocr_use_textline_orientation": True},
        aliases=("exp1_paddleocr_gliner2ner",),
    ),
    "exp2_ocr_ner_qwen3": ExperimentSpec(
        experiment_id="exp2_ocr_ner_qwen3",
        description="Invoice parser: PaddleOCR + Qwen3 NER",
        pipeline="ocr_ner",
        backend="llama_server",
        backend_config={
            "base_url": "http://localhost:8081/v1",
            "task_type": "ner",
            "model": "qwen3",
            "model_path": str(QWEN3_DIR / "Qwen_Qwen3-4B-Q4_K_M.gguf"),
            "default_port": 8081,
            "ctx_size": 4096,
        },
        ocr_defaults={"ocr_lang": "en", "ocr_use_textline_orientation": True},
        runtime=RuntimeSpec(
            kind="llamacpp_server",
            port=8081,
            model_path=QWEN3_DIR / "Qwen_Qwen3-4B-Q4_K_M.gguf",
            model_alias="qwen3",
            ctx_size=4096,
            hf_assets=HfAssetSpec(
                repo_id="bartowski/Qwen_Qwen3-4B-GGUF",
                filenames=("Qwen_Qwen3-4B-Q4_K_M.gguf",),
                local_dir=QWEN3_DIR,
            ),
        ),
        aliases=("exp2_paddleocr_qwen3ner",),
    ),
    "exp3_vlm_qwen25vl": ExperimentSpec(
        experiment_id="exp3_vlm_qwen25vl",
        description="Invoice parser: Qwen2.5-VL via llama.cpp",
        pipeline="vlm",
        backend="llama_server",
        backend_config={
            "base_url": "http://localhost:8082/v1",
            "task_type": "vlm",
            "model": "qwen25vl",
            "model_path": str(QWEN25VL_DIR / "Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"),
            "mmproj_path": str(QWEN25VL_DIR / "mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf"),
            "default_port": 8082,
            "ctx_size": 4096,
        },
        runtime=RuntimeSpec(
            kind="llamacpp_server",
            port=8082,
            model_path=QWEN25VL_DIR / "Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
            mmproj_path=QWEN25VL_DIR / "mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf",
            model_alias="qwen25vl",
            ctx_size=4096,
            hf_assets=HfAssetSpec(
                repo_id="bartowski/Qwen_Qwen2.5-VL-7B-Instruct-GGUF",
                filenames=(
                    "Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
                    "mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf",
                ),
                local_dir=QWEN25VL_DIR,
            ),
        ),
        aliases=("exp3_vlm_qwen25vl",),
    ),
    "exp5_vlm_qwen25vl3b": ExperimentSpec(
        experiment_id="exp5_vlm_qwen25vl3b",
        description="Invoice parser: Qwen2.5-VL-3B via llama.cpp",
        pipeline="vlm",
        backend="llama_server",
        backend_config={
            "base_url": "http://localhost:8083/v1",
            "task_type": "vlm",
            "model": "qwen25vl3b",
            "model_path": str(QWEN25VL3B_DIR / "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"),
            "mmproj_path": str(QWEN25VL3B_DIR / "mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"),
            "default_port": 8083,
            "ctx_size": 4096,
        },
        runtime=RuntimeSpec(
            kind="llamacpp_server",
            port=8083,
            model_path=QWEN25VL3B_DIR / "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
            mmproj_path=QWEN25VL3B_DIR / "mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf",
            model_alias="qwen25vl3b",
            ctx_size=4096,
            hf_assets=HfAssetSpec(
                repo_id="ggml-org/Qwen2.5-VL-3B-Instruct-GGUF",
                filenames=(
                    "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
                    "mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf",
                ),
                local_dir=QWEN25VL3B_DIR,
            ),
        ),
        aliases=("exp5_vlm_qwen25vl3b",),
    ),
    "exp6_vlm_qwen3vl2b": ExperimentSpec(
        experiment_id="exp6_vlm_qwen3vl2b",
        description="Invoice parser: Qwen3-VL-2B via llama.cpp",
        pipeline="vlm",
        backend="llama_server",
        backend_config={
            "base_url": "http://localhost:8084/v1",
            "task_type": "vlm",
            "model": "qwen3vl2b",
            "model_path": str(QWEN3VL2B_DIR / "Qwen3VL-2B-Instruct-Q4_K_M.gguf"),
            "mmproj_path": str(QWEN3VL2B_DIR / "mmproj-Qwen3VL-2B-Instruct-F16.gguf"),
            "default_port": 8084,
            "ctx_size": 4096,
        },
        runtime=RuntimeSpec(
            kind="llamacpp_server",
            port=8084,
            model_path=QWEN3VL2B_DIR / "Qwen3VL-2B-Instruct-Q4_K_M.gguf",
            mmproj_path=QWEN3VL2B_DIR / "mmproj-Qwen3VL-2B-Instruct-F16.gguf",
            model_alias="qwen3vl2b",
            ctx_size=4096,
            hf_assets=HfAssetSpec(
                repo_id="Qwen/Qwen3-VL-2B-Instruct-GGUF",
                filenames=(
                    "Qwen3VL-2B-Instruct-Q4_K_M.gguf",
                    "mmproj-Qwen3VL-2B-Instruct-F16.gguf",
                ),
                local_dir=QWEN3VL2B_DIR,
            ),
        ),
        aliases=("exp6_vlm_qwen3vl2b",),
    ),
    "exp7_vlm_lfm25vl16b": ExperimentSpec(
        experiment_id="exp7_vlm_lfm25vl16b",
        description="Invoice parser: LFM2.5-VL-1.6B via llama.cpp",
        pipeline="vlm",
        backend="llama_mtmd_cli",
        backend_config={
            "task_type": "vlm",
            "mtmd_bin": "llama-mtmd-cli",
            "model_path": str(LFM25VL16B_DIR / "LFM2.5-VL-1.6b-Q4_0.gguf"),
            "mmproj_path": str(LFM25VL16B_DIR / "mmproj-LFM2.5-VL-1.6b-Q8_0.gguf"),
            "debug": False,
            "ctx_size": 4096,
        },
        runtime=RuntimeSpec(
            kind="hf_assets",
            model_path=LFM25VL16B_DIR / "LFM2.5-VL-1.6b-Q4_0.gguf",
            mmproj_path=LFM25VL16B_DIR / "mmproj-LFM2.5-VL-1.6b-Q8_0.gguf",
            ctx_size=4096,
            hf_assets=HfAssetSpec(
                repo_id="LiquidAI/LFM2.5-VL-1.6B-GGUF",
                filenames=(
                    "LFM2.5-VL-1.6b-Q4_0.gguf",
                    "mmproj-LFM2.5-VL-1.6b-Q8_0.gguf",
                ),
                local_dir=LFM25VL16B_DIR,
            ),
        ),
        aliases=("exp7_vlm_lfm25vl16b",),
    ),
    "exp8_vlm_smolvlm256m": ExperimentSpec(
        experiment_id="exp8_vlm_smolvlm256m",
        description="Invoice parser: SmolVLM-256M via llama.cpp",
        pipeline="vlm",
        backend="llama_mtmd_cli",
        backend_config={
            "task_type": "vlm",
            "mtmd_bin": "llama-mtmd-cli",
            "model_path": str(SMOLVLM256M_DIR / "SmolVLM-256M-Instruct-Q8_0.gguf"),
            "mmproj_path": str(SMOLVLM256M_DIR / "mmproj-SmolVLM-256M-Instruct-f16.gguf"),
            "debug": False,
            "ctx_size": 4096,
        },
        runtime=RuntimeSpec(
            kind="hf_assets",
            model_path=SMOLVLM256M_DIR / "SmolVLM-256M-Instruct-Q8_0.gguf",
            mmproj_path=SMOLVLM256M_DIR / "mmproj-SmolVLM-256M-Instruct-f16.gguf",
            ctx_size=4096,
            hf_assets=HfAssetSpec(
                repo_id="ggml-org/SmolVLM-256M-Instruct-GGUF",
                filenames=(
                    "SmolVLM-256M-Instruct-Q8_0.gguf",
                    "mmproj-SmolVLM-256M-Instruct-f16.gguf",
                ),
                local_dir=SMOLVLM256M_DIR,
            ),
        ),
        aliases=("exp8_vlm_smolvlm256m",),
    ),
    "exp4_vlm_minicpmv": ExperimentSpec(
        experiment_id="exp4_vlm_minicpmv",
        description="Invoice parser: MiniCPM-V via llama-mtmd-cli",
        pipeline="vlm",
        backend="llama_mtmd_cli",
        backend_config={
            "task_type": "vlm",
            "mtmd_bin": "llama-mtmd-cli",
            "model_path": str(MINICPMV_DIR / "MiniCPM-V-4_5-Q4_K_M.gguf"),
            "mmproj_path": str(MINICPMV_DIR / "mmproj-model-f16.gguf"),
            "debug": False,
        },
        runtime=RuntimeSpec(
            kind="hf_assets",
            model_path=MINICPMV_DIR / "MiniCPM-V-4_5-Q4_K_M.gguf",
            mmproj_path=MINICPMV_DIR / "mmproj-model-f16.gguf",
            hf_assets=HfAssetSpec(
                repo_id="openbmb/MiniCPM-V-4_5-gguf",
                filenames=("MiniCPM-V-4_5-Q4_K_M.gguf", "mmproj-model-f16.gguf"),
                local_dir=MINICPMV_DIR,
            ),
        ),
        aliases=("exp4_vlm_minicpm",),
    ),
}


EXPERIMENT_ALIASES: dict[str, str] = {
    alias: spec.experiment_id
    for spec in EXPERIMENT_SPECS.values()
    for alias in spec.aliases
}


def resolve_experiment_id(value: str) -> str:
    if value in EXPERIMENT_ALIASES:
        return EXPERIMENT_ALIASES[value]
    stem = Path(value).stem
    return EXPERIMENT_ALIASES.get(stem, value)


def get_experiment_spec(experiment_id: str) -> ExperimentSpec:
    canonical = resolve_experiment_id(experiment_id)
    try:
        return EXPERIMENT_SPECS[canonical]
    except KeyError as exc:
        choices = ", ".join(sorted(EXPERIMENT_SPECS))
        raise ValueError(f"Unknown experiment '{experiment_id}'. Choose from: {choices}") from exc
