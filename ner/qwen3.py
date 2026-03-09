"""
Qwen3 NER backend via llama.cpp server.

Expects a llama.cpp server running with the OpenAI-compatible API, e.g.:
    llama-server --model Qwen3-4B-Q4_K_M.gguf --port 8080

Qwen3's thinking mode is disabled by default (add /no-think in the system
prompt) so the response is pure JSON.
"""

import json
from openai import OpenAI

from schema import AdvancedReceiptData, ProductLineItem
from ner.common import NERBackend


SYSTEM_PROMPT = """/no-think
You are an invoice data extraction assistant.
Extract all available fields from the invoice text the user provides.
Respond ONLY with a valid JSON object matching this exact structure — no markdown, no explanation:

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

Use null for any field not found in the text. Preserve original formatting for amounts and dates.
"""


class Qwen3Backend(NERBackend):
    """
    NER backend using Qwen3-4B via a llama.cpp OpenAI-compatible server.

    Args:
        base_url:    llama.cpp server base URL (default: http://localhost:8080/v1).
        model:       Model name as reported by the server (default: "qwen3").
        temperature: Sampling temperature. 0 for greedy/deterministic output.
        max_tokens:  Max tokens for the response.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        model: str = "qwen3",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):
        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def extract(self, text: str) -> AdvancedReceiptData:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
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
