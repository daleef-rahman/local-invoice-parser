"""
Base class for VLM (Vision Language Model) backends.
A VLMBackend takes a raw image path and returns structured invoice data directly,
bypassing a separate OCR step.
"""

from schema import AdvancedReceiptData


class VLMBackend:
    """
    Base class for VLM invoice extraction backends.
    Subclasses must implement extract().
    """

    def extract(self, image_path: str) -> AdvancedReceiptData:
        raise NotImplementedError

    def close(self):
        """Optional cleanup (e.g. free model memory)."""
        pass
