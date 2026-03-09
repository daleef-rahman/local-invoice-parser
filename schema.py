"""
Invoice extraction schema — Pydantic output models only.
"""
from typing import Optional
from pydantic import BaseModel

# Field value types — kept as str to preserve formatting (e.g. "$1,234.56")
StringFieldValue = Optional[str]
NumericFieldValue = Optional[str]


class ProductLineItem(BaseModel):
    productName: StringFieldValue = None
    quantity: NumericFieldValue = None
    unitPrice: NumericFieldValue = None
    totalPrice: NumericFieldValue = None
    productCode: StringFieldValue = None


class AdvancedReceiptData(BaseModel):
    totalAmount: NumericFieldValue = None
    taxAmount: NumericFieldValue = None
    dateTime: StringFieldValue = None
    merchantName: StringFieldValue = None
    merchantAddress: StringFieldValue = None
    currencyCode: StringFieldValue = None
    merchantCountry: StringFieldValue = None
    merchantState: StringFieldValue = None
    merchantCity: StringFieldValue = None
    merchantPostalCode: StringFieldValue = None
    merchantPhone: StringFieldValue = None
    merchantEmail: StringFieldValue = None
    invoiceReceiptNumber: StringFieldValue = None
    paidAmount: NumericFieldValue = None
    discountAmount: NumericFieldValue = None
    serviceCharge: NumericFieldValue = None
    productLineItems: list[ProductLineItem] = []
