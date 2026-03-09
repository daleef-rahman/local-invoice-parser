"""MiniCPM-V backend via llama.cpp `llama-mtmd-cli`."""

import subprocess
from pathlib import Path

from schema import AdvancedReceiptData, ProductLineItem
from models.utils import parse_json_with_retries
from models.vlm.common import VLMBackend

_DEFAULT_MODEL = Path.home() / "models" / "minicpmv-4.5" / "MiniCPM-V-4_5-Q4_K_M.gguf"
_DEFAULT_MMPROJ = Path.home() / "models" / "minicpmv-4.5" / "mmproj-model-f16.gguf"
_CTX_SIZE = 4096
_TOP_P = 0.8
_TOP_K = 100
_REPEAT_PENALTY = 1.05


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
    VLM backend for MiniCPM-V via `llama-mtmd-cli`.

    Args:
        mtmd_bin:       Path to llama-mtmd-cli binary.
        model_path:     Path to MiniCPM-V gguf model.
        mmproj_path:    Path to mmproj file.
        temperature:    Sampling temperature.
        max_tokens:     Max tokens for the response.
        ctx_size:       Context size.
        debug:          Print payload/server diagnostics for troubleshooting.
    """

    def __init__(
        self,
        mtmd_bin: str = "llama-mtmd-cli",
        model_path: str | None = None,
        mmproj_path: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        ctx_size: int = _CTX_SIZE,
        debug: bool = False,
    ):
        self.mtmd_bin = mtmd_bin
        self.model_path = Path(model_path) if model_path else _DEFAULT_MODEL
        self.mmproj_path = Path(mmproj_path) if mmproj_path else _DEFAULT_MMPROJ
        for path, label in [(self.model_path, "model"), (self.mmproj_path, "mmproj")]:
            if not path.exists():
                raise FileNotFoundError(
                    f"MiniCPM-V {label} not found at {path}. "
                    "Run scripts/serve_minicpmv.sh first to download it."
                )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ctx_size = ctx_size
        self.debug = debug
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
        prompts = [
            EXTRACTION_PROMPT,
            EXTRACTION_PROMPT + "\n\nRetry: Return ONLY minified valid JSON. Do not truncate strings.",
            EXTRACTION_PROMPT + "\n\nFinal retry: Return a compact JSON object only.",
        ]
        raw = parse_json_with_retries(
            lambda prompt: self._request_raw(image_path=image_path, prompt=prompt),
            prompts,
            error_prefix="Failed to parse JSON from MiniCPM-V response",
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
