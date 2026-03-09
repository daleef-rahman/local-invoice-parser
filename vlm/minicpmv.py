"""
MiniCPM-V backend via llama.cpp native /completion endpoint.

MiniCPM-V's chat template is not compatible with llama.cpp's OpenAI-layer,
so this backend calls the native /completion API directly with image_data.

Start the server first:
    ./scripts/serve_minicpmv.sh

Then run:
    python exp2_vlm.py --image invoice.jpg --vlm-backend minicpmv
"""

import base64
import io
import json
import re

import httpx
from PIL import Image

from schema import AdvancedReceiptData, ProductLineItem
from vlm.common import VLMBackend


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


class MiniCPMVBackend(VLMBackend):
    """
    VLM backend for MiniCPM-V via the llama.cpp native /completion endpoint.

    Args:
        base_url:       llama.cpp server base URL (default: http://localhost:8081).
        temperature:    Sampling temperature.
        max_tokens:     Max tokens for the response.
        max_image_size: Longest side of the image before encoding (pixels).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8081",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_image_size: int = 1024,
    ):
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_image_size = max_image_size
        self._client = httpx.Client(timeout=120.0)

    def _encode_image(self, image_path: str) -> str:
        """Resize to max_image_size and return raw base64 (no data-URL prefix)."""
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((self.max_image_size, self.max_image_size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode()

    @staticmethod
    def _extract_json_object(text: str) -> str:
        fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fence_match:
            text = fence_match.group(1)
        start = text.find("{")
        if start == -1:
            raise json.JSONDecodeError("No JSON object start found", text, 0)
        in_string = escape = False
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == "\"":
                    in_string = False
                continue
            if ch == "\"":
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start: i + 1]
        raise json.JSONDecodeError("No complete JSON object found", text, start)

    def _request_raw(self, b64_image: str, prompt: str) -> str:
        # MiniCPM-V 4.5 uses ChatML format; [img-1] is the llama.cpp image placeholder
        formatted = (
            f"<|im_start|>user\n[img-1]\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        payload = {
            "prompt": formatted,
            "image_data": [{"data": b64_image, "id": 1}],
            "temperature": self.temperature,
            "n_predict": self.max_tokens,
            "stop": ["<|im_end|>", "<|im_start|>"],
        }
        resp = self._client.post(f"{self.base_url}/completion", json=payload)
        resp.raise_for_status()
        data = resp.json()
        tokens_evaluated = data.get("tokens_evaluated", "?")
        tokens_predicted = data.get("tokens_predicted", "?")
        print(f"  [debug] prompt_tokens={tokens_evaluated} generated_tokens={tokens_predicted}")
        content = data.get("content", "")
        print(f"  [debug] raw response: {content[:300]!r}")
        return content

    def extract(self, image_path: str) -> AdvancedReceiptData:
        b64_image = self._encode_image(image_path)
        last_error: Exception | None = None
        prompts = [
            EXTRACTION_PROMPT,
            EXTRACTION_PROMPT + "\n\nRetry: Return ONLY minified valid JSON. Do not truncate strings.",
            EXTRACTION_PROMPT + "\n\nFinal retry: Return a compact JSON object only.",
        ]
        raw: dict | None = None
        for prompt in prompts:
            content = self._request_raw(b64_image=b64_image, prompt=prompt)
            try:
                raw = json.loads(content)
                break
            except json.JSONDecodeError:
                try:
                    raw = json.loads(self._extract_json_object(content))
                    break
                except json.JSONDecodeError as e:
                    last_error = e

        if raw is None:
            raise ValueError(f"Failed to parse JSON from MiniCPM-V response: {last_error}")

        line_items = [
            ProductLineItem(**{k: item.get(k) for k in ProductLineItem.model_fields})
            for item in raw.get("productLineItems", [])
        ]
        return AdvancedReceiptData(
            **{k: raw.get(k) for k in AdvancedReceiptData.model_fields if k != "productLineItems"},
            productLineItems=line_items,
        )

    def close(self):
        self._client.close()
