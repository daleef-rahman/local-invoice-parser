"""VLM backend via llama-mtmd-cli."""

from __future__ import annotations

import subprocess
from pathlib import Path

from models.prompting import TASK_VLM, get_retry_prompts
from models.utils import parse_json_with_retries
from models.modelbackend import ModelBackend
from schema import AdvancedReceiptData, ProductLineItem

_TOP_P = 0.8
_TOP_K = 100
_REPEAT_PENALTY = 1.05


class LlamaMtmdCliVLMBackend(ModelBackend):
    """Image extraction through llama-mtmd-cli with a task-specific prompt."""

    _DEFAULT_MODEL = Path.home() / "models" / "minicpmv-4.5" / "MiniCPM-V-4_5-Q4_K_M.gguf"
    _DEFAULT_MMPROJ = Path.home() / "models" / "minicpmv-4.5" / "mmproj-model-f16.gguf"

    def __init__(
        self,
        *,
        task_type: str = TASK_VLM,
        mtmd_bin: str = "llama-mtmd-cli",
        model_path: str | None = None,
        mmproj_path: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        ctx_size: int = 4096,
        debug: bool = False,
    ):
        self.mtmd_bin = mtmd_bin
        self.model_path = Path(model_path) if model_path else self._DEFAULT_MODEL
        self.mmproj_path = Path(mmproj_path) if mmproj_path else self._DEFAULT_MMPROJ
        for path in (self.model_path, self.mmproj_path):
            if not path.exists():
                raise FileNotFoundError(f"Model asset not found at {path}")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ctx_size = ctx_size
        self.debug = debug
        self.prompts = get_retry_prompts(task_type)

    def _request_raw(self, image_path: str, prompt: str) -> str:
        cmd = [
            self.mtmd_bin,
            "-m",
            str(self.model_path),
            "--mmproj",
            str(self.mmproj_path),
            "-c",
            str(self.ctx_size),
            "--temp",
            str(self.temperature),
            "--top-p",
            str(_TOP_P),
            "--top-k",
            str(_TOP_K),
            "--repeat-penalty",
            str(_REPEAT_PENALTY),
            "--image",
            str(Path(image_path).resolve()),
            "-n",
            str(self.max_tokens),
            "-p",
            prompt,
        ]
        if self.debug:
            print(f"  [debug] mtmd_bin={self.mtmd_bin}")
            print(f"  [debug] model={self.model_path}")
            print(f"  [debug] mmproj={self.mmproj_path}")
            print(f"  [debug] cmd={' '.join(cmd)}")

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"llama-mtmd-cli failed (exit {proc.returncode}).\n"
                f"stderr:\n{proc.stderr[-1200:]}\nstdout:\n{proc.stdout[-1200:]}"
            )
        if self.debug:
            print(f"  [debug] mtmd stderr: {proc.stderr[-400:]}")
            print(f"  [debug] mtmd stdout head: {proc.stdout[:300]!r}")
        return proc.stdout

    def extract(self, image_path: str) -> AdvancedReceiptData:
        raw = parse_json_with_retries(
            lambda prompt: self._request_raw(image_path=image_path, prompt=prompt),
            self.prompts,
            error_prefix="Failed to parse JSON from llama-mtmd-cli response",
        )
        line_items = [
            ProductLineItem(**{k: item.get(k) for k in ProductLineItem.model_fields})
            for item in raw.get("productLineItems", [])
        ]
        return AdvancedReceiptData(
            **{k: raw.get(k) for k in AdvancedReceiptData.model_fields if k != "productLineItems"},
            productLineItems=line_items,
        )

    def close(self):
        pass
