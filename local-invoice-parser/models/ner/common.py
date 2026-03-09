"""
Base class for NER (Named Entity Recognition) backends.
A NERBackend takes raw OCR text and returns structured invoice data.
Example backends: GLiNER2, Qwen3 via llama.cpp.
"""

from schema import AdvancedReceiptData


class NERBackend:
    """
    Base class for invoice extraction backends.
    Subclasses must implement extract().
    """

    def extract(self, text: str) -> AdvancedReceiptData:
        raise NotImplementedError

    def close(self):
        """Optional cleanup (e.g. close HTTP sessions, free model memory)."""
        pass
