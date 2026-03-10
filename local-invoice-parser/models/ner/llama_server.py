"""NER backend via an OpenAI-compatible llama.cpp server."""

from __future__ import annotations

import json
from pathlib import Path

from openai import OpenAI

from models.modelbackend import ModelBackend
from models.prompting import TASK_NER, get_prompt
from models.utils import ensure_llama_server
from schema import AdvancedReceiptData, ProductLineItem


class LlamaServerNERBackend(ModelBackend):
    """NER extraction through llama-server with a task-specific prompt."""

    _DEFAULT_MODEL = Path.home() / "models" / "qwen3-4b" / "Qwen_Qwen3-4B-Q4_K_M.gguf"

    def __init__(
        self,
        *,
        task_type: str = TASK_NER,
        base_url: str = "http://localhost:8080/v1",
        model: str = "qwen3",
        model_path: str | None = None,
        mmproj_path: str | None = None,
        default_port: int = 8080,
        ctx_size: int = 4096,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        resolved_model_path = model_path or str(self._DEFAULT_MODEL)
        if not Path(resolved_model_path).exists():
            raise FileNotFoundError(f"Model not found at {resolved_model_path}")
        ensure_llama_server(
            base_url,
            default_port=default_port,
            model_args=["--model", resolved_model_path, "--ctx-size", str(ctx_size), *(
                ["--mmproj", mmproj_path] if mmproj_path else []
            )],
        )
        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        self.model = model
        self.system_prompt = get_prompt(task_type)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def extract(self, text: str) -> AdvancedReceiptData:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )
        raw = json.loads(response.choices[0].message.content)
        line_items = [
            ProductLineItem(**{k: item.get(k) for k in ProductLineItem.model_fields})
            for item in raw.get("productLineItems", [])
        ]
        return AdvancedReceiptData(
            **{k: raw.get(k) for k in AdvancedReceiptData.model_fields if k != "productLineItems"},
            productLineItems=line_items,
        )

    def close(self):
        self.client.close()
