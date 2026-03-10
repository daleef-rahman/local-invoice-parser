"""
Shared base class for invoice extraction backends.

A ModelBackend takes a string input and returns structured invoice data.
The input may be OCR text, an image path, or another backend-specific string payload.
"""

from schema import AdvancedReceiptData


class ModelBackend:
    """
    Base class for invoice extraction backends.
    Subclasses must implement extract().
    """

    def extract(self, input_data: str) -> AdvancedReceiptData:
        raise NotImplementedError

    def close(self):
        """Optional cleanup (e.g. close HTTP sessions, free model memory)."""
        pass
