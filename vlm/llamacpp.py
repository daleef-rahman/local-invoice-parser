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
import json
import re
from pathlib import Path

from PIL import Image
from openai import OpenAI

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
        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_image_size = max_image_size

    @staticmethod
    def _extract_json_object(text: str) -> str:
        # Handle ```json ... ``` wrappers first.
        fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fence_match:
            text = fence_match.group(1)

        start = text.find("{")
        if start == -1:
            raise json.JSONDecodeError("No JSON object start found", text, 0)

        in_string = False
        escape = False
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
                    return text[start : i + 1]

        raise json.JSONDecodeError("No complete JSON object found", text, start)

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

        last_error: Exception | None = None
        prompts = [
            EXTRACTION_PROMPT,
            EXTRACTION_PROMPT + "\n\nRetry: Return ONLY minified valid JSON. Do not truncate strings.",
            EXTRACTION_PROMPT + "\n\nFinal retry: Return a compact JSON object only.",
        ]
        raw: dict | None = None
        for prompt in prompts:
            content = self._request_raw(image_data_url=image_data_url, prompt=prompt)
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
            raise ValueError(f"Failed to parse JSON from llama.cpp response: {last_error}")

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
