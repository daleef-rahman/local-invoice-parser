"""VLM backend via an OpenAI-compatible llama.cpp server."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image
from openai import OpenAI

from models.prompting import TASK_VLM, get_retry_prompts
from models.utils import build_receipt_from_raw, ensure_llama_server, parse_json_with_retries
from models.modelbackend import ModelBackend
from schema import AdvancedReceiptData


class LlamaServerVLMBackend(ModelBackend):
    """Image extraction through llama-server with a task-specific prompt."""

    _DEFAULT_MODEL = Path.home() / "models" / "qwen25vl-7b" / "Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
    _DEFAULT_MMPROJ = Path.home() / "models" / "qwen25vl-7b" / "mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf"

    def __init__(
        self,
        *,
        task_type: str = TASK_VLM,
        base_url: str = "http://localhost:8080/v1",
        model: str = "qwen25vl",
        model_path: str | None = None,
        mmproj_path: str | None = None,
        default_port: int = 8080,
        ctx_size: int = 4096,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_image_size: int = 1024,
    ):
        resolved_model_path = model_path or str(self._DEFAULT_MODEL)
        resolved_mmproj_path = mmproj_path or str(self._DEFAULT_MMPROJ)
        for path in (resolved_model_path, resolved_mmproj_path):
            if not Path(path).exists():
                raise FileNotFoundError(f"Model asset not found at {path}")
        ensure_llama_server(
            base_url,
            default_port=default_port,
            model_args=[
                "--model", resolved_model_path,
                "--mmproj", resolved_mmproj_path,
                "--ctx-size", str(ctx_size),
            ],
        )
        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        self.model = model
        self.prompts = get_retry_prompts(task_type)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_image_size = max_image_size

    def _request_raw(self, image_data_url: str, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    def _encode_image(self, image_path: str) -> str:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((self.max_image_size, self.max_image_size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"

    def extract(self, image_path: str) -> AdvancedReceiptData:
        image_data_url = self._encode_image(image_path)
        raw = parse_json_with_retries(
            lambda prompt: self._request_raw(image_data_url=image_data_url, prompt=prompt),
            self.prompts,
            error_prefix="Failed to parse JSON from llama-server response",
        )
        return build_receipt_from_raw(raw)

    def close(self):
        self.client.close()
