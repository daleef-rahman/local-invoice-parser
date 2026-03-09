"""
GLiNER2 NER backend.
Uses the GLiNER2 model for structured JSON extraction from invoice text.
"""

from gliner2 import GLiNER2

from schema import AdvancedReceiptData, ProductLineItem
from models.ner.common import NERBackend


# Format: "field_name::type::description"  (::str = single value)
INVOICE_SCHEMA = {
    "receipt": [
        "totalAmount::str::Total amount due on the invoice",
        "taxAmount::str::Tax or VAT amount",
        "dateTime::str::Invoice or receipt date and time",
        "merchantName::str::Business or vendor name",
        "merchantAddress::str::Full street address of the merchant",
        "currencyCode::str::ISO currency code e.g. USD GBP EUR",
        "merchantCountry::str::Country of the merchant",
        "merchantState::str::State or province of the merchant",
        "merchantCity::str::City of the merchant",
        "merchantPostalCode::str::Postal or ZIP code of the merchant",
        "merchantPhone::str::Phone number of the merchant",
        "merchantEmail::str::Email address of the merchant",
        "invoiceReceiptNumber::str::Invoice or receipt identifier number",
        "paidAmount::str::Amount already paid",
        "discountAmount::str::Discount applied",
        "serviceCharge::str::Service charge or tip",
    ],
    "productLineItem": [
        "productName::str::Name or description of the product or service",
        "quantity::str::Quantity purchased",
        "unitPrice::str::Price per unit",
        "totalPrice::str::Total price for this line item",
        "productCode::str::SKU barcode or product code",
    ],
}


class GLiNER2Backend(NERBackend):
    """
    NER backend using GLiNER2 for structured extraction.

    Args:
        model: HuggingFace model id, e.g. "fastino/gliner2-base-v1" or "fastino/gliner2-large-v1".
    """

    def __init__(self, model: str = "fastino/gliner2-base-v1"):
        self.ner = GLiNER2.from_pretrained(model)

    def extract(self, text: str) -> AdvancedReceiptData:
        raw = self.ner.extract_json(text, INVOICE_SCHEMA)
        receipt_list = raw.get("receipt", [{}])
        receipt = receipt_list[0] if receipt_list else {}
        line_items = [
            ProductLineItem(**{k: item.get(k) for k in ProductLineItem.model_fields})
            for item in raw.get("productLineItem", [])
        ]
        return AdvancedReceiptData(
            **{k: receipt.get(k) for k in AdvancedReceiptData.model_fields if k != "productLineItems"},
            productLineItems=line_items,
        )
