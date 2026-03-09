"""
Qwen2.5-VL backend.
Runs Qwen2.5-VL locally via HuggingFace transformers.

Install dependencies:
    uv add transformers qwen-vl-utils
"""

import json
from pathlib import Path

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

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


class Qwen25VLBackend(VLMBackend):
    """
    VLM backend using Qwen2.5-VL loaded locally via transformers.

    Args:
        model:       HuggingFace model id (default: Qwen/Qwen2.5-VL-7B-Instruct).
        device_map:  Device placement strategy passed to from_pretrained.
        max_tokens:  Max new tokens to generate.
    """

    def __init__(
        self,
        model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device_map: str = "auto",
        max_tokens: int = 1024,
    ):
        self.max_tokens = max_tokens
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        self.processor = AutoProcessor.from_pretrained(model)

    def extract(self, image_path: str) -> AdvancedReceiptData:
        image_path = str(Path(image_path).resolve())
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": EXTRACTION_PROMPT},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_tokens)

        # Strip the input tokens, decode only the generated part
        generated_ids = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        raw = json.loads(output)
        line_items = [
            ProductLineItem(**{k: item.get(k) for k in ProductLineItem.model_fields})
            for item in raw.get("productLineItems", [])
        ]
        return AdvancedReceiptData(
            **{k: raw.get(k) for k in AdvancedReceiptData.model_fields if k != "productLineItems"},
            productLineItems=line_items,
        )

    def close(self):
        del self.model
        del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
