"""Shared task prompts for runtime-based model backends."""

from __future__ import annotations

TASK_NER = "ner"
TASK_VLM = "vlm"
TASK_TYPES = {TASK_NER, TASK_VLM}

_SCHEMA_BLOCK = """{
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
}"""

_NER_SYSTEM_PROMPT = f"""/no-think
You are an invoice data extraction assistant.
Extract all available fields from the invoice text the user provides.
Respond ONLY with a valid JSON object matching this exact structure - no markdown, no explanation:

{_SCHEMA_BLOCK}

Use null for any field not found in the text. Preserve original formatting for amounts and dates.
"""

_VLM_PROMPT = f"""Extract all invoice/receipt fields from this image.
Respond ONLY with a valid JSON object - no markdown, no explanation:

{_SCHEMA_BLOCK}

Use null for any field not found. Preserve original formatting for amounts and dates."""


def normalize_task_type(task_type: str) -> str:
    if task_type not in TASK_TYPES:
        choices = ", ".join(sorted(TASK_TYPES))
        raise ValueError(f"Unknown task_type: {task_type!r}. Choose from: {choices}")
    return task_type


def get_prompt(task_type: str) -> str:
    task_type = normalize_task_type(task_type)
    if task_type == TASK_NER:
        return _NER_SYSTEM_PROMPT
    return _VLM_PROMPT


def get_retry_prompts(task_type: str) -> list[str]:
    base_prompt = get_prompt(task_type)
    return [
        base_prompt,
        base_prompt + "\n\nRetry: Return ONLY minified valid JSON. Do not truncate strings.",
        base_prompt + "\n\nFinal retry: Return a compact JSON object only.",
    ]
