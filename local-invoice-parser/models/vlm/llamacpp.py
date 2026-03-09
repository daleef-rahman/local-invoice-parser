"""
llama.cpp VLM backend.
Runs any vision-capable GGUF model via the llama.cpp OpenAI-compatible server.

Start the server first:
    ./scripts/serve_qwen25vl.sh

Then run:
    python exp2_vlm.py --image invoice.jpg --vlm-backend llamacpp
"""

import base64
import io
from pathlib import Path

from PIL import Image
from openai import OpenAI

from schema import AdvancedReceiptData, ProductLineItem
from models.utils import ensure_llama_server, parse_json_with_retries
from models.vlm.common import VLMBackend

_DEFAULT_MODEL = Path.home() / "models" / "qwen25vl-7b" / "Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
_DEFAULT_MMPROJ = Path.home() / "models" / "qwen25vl-7b" / "mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf"
_CTX_SIZE = 4096


EXTRACTION_PROMPT = """Extract all invoice/receipt fields from this image.
Respond ONLY with a valid JSON object — no markdown, no explanation:

{
  "totalAmount": "string or null",
  "taxAmount": "string or null",
  "dateTime": "string or null",
  "merchantName": "string or null",
  "merchantAddress": "string or null",
  "currencyCode": "string or null",
  "merchantCountry": "string or null",
  "merchantState": "string or null",
  "merchantCity": "string or null",
  "merchantPostalCode": "string or null",
  "merchantPhone": "string or null",
  "merchantEmail": "string or null",
  "invoiceReceiptNumber": "string or null",
  "paidAmount": "string or null",
  "discountAmount": "string or null",
  "serviceCharge": "string or null",
  "productLineItems": [
    {
      "productName": "string or null",
      "quantity": "string or null",
      "unitPrice": "string or null",
      "totalPrice": "string or null",
      "productCode": "string or null"
    }
  ]
}

Use null for any field not found. Preserve original formatting for amounts and dates."""


class LlamaCppVLMBackend(VLMBackend):
    """
    VLM backend using a llama.cpp OpenAI-compatible server (vision-capable model).

    Args:
        base_url:    llama.cpp server base URL (default: http://localhost:8080/v1).
        model:       Model name as reported by the server.
        temperature: Sampling temperature. 0 for deterministic output.
        max_tokens:  Max tokens for the response.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        model: str = "qwen25vl",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_image_size: int = 1024,
    ):
        for path, label in [(_DEFAULT_MODEL, "model"), (_DEFAULT_MMPROJ, "mmproj")]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Qwen2.5-VL {label} not found at {path}. "
                    "Run scripts/serve_qwen25vl.sh first to download it."
                )
        ensure_llama_server(
            base_url,
            default_port=8080,
            model_args=[
                "--model", str(_DEFAULT_MODEL),
                "--mmproj", str(_DEFAULT_MMPROJ),
                "--ctx-size", str(_CTX_SIZE),
            ],
        )
        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        self.model = model
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
        """Resize image to max_image_size and return a JPEG data URL."""
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((self.max_image_size, self.max_image_size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/jpeg;base64,{b64}"

    def extract(self, image_path: str) -> AdvancedReceiptData:
        image_data_url = self._encode_image(image_path)

        prompts = [
            EXTRACTION_PROMPT,
            EXTRACTION_PROMPT + "\n\nRetry: Return ONLY minified valid JSON. Do not truncate strings.",
            EXTRACTION_PROMPT + "\n\nFinal retry: Return a compact JSON object only.",
        ]
        raw = parse_json_with_retries(
            lambda prompt: self._request_raw(image_data_url=image_data_url, prompt=prompt),
            prompts,
            error_prefix="Failed to parse JSON from llama.cpp response",
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
        self.client.close()
