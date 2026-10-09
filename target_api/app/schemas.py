"""Request and response models, and the documented error responses."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class OrderStatus(StrEnum):
    pending = "pending"
    paid = "paid"
    cancelled = "cancelled"
    shipped = "shipped"


class Customer(BaseModel):
    id: str = Field(json_schema_extra={"format": "uuid"})
    name: str
    email: str
    created_at: datetime


class Product(BaseModel):
    id: str = Field(json_schema_extra={"format": "uuid"})
    sku: str
    name: str
    price: str = Field(
        description="Unit price as a decimal string with up to 3 decimals", examples=["19.99"]
    )
    currency: str
    active: bool


class ProductPatch(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    price: str | None = Field(default=None, pattern=r"^\d{1,6}(\.\d{1,3})?$", examples=["21.50"])
    active: bool | None = None


class OrderItemIn(BaseModel):
    product_id: str = Field(json_schema_extra={"format": "uuid"})
    quantity: int = Field(description="Units ordered, 1..1000", examples=[2])


class OrderIn(BaseModel):
    customer_id: str | None = Field(
        default=None,
        json_schema_extra={"format": "uuid"},
        description="Required for staff/admin; customers always order for themselves",
    )
    items: list[OrderItemIn] = Field(min_length=1, max_length=50)


class OrderItem(BaseModel):
    product_id: str = Field(json_schema_extra={"format": "uuid"})
    quantity: int
    unit_price: str
    line_total: str = Field(description="unit_price * quantity rounded half-up to cents")


class Order(BaseModel):
    id: str = Field(json_schema_extra={"format": "uuid"})
    customer_id: str = Field(json_schema_extra={"format": "uuid"})
    status: OrderStatus
    items: list[OrderItem]
    total: str
    currency: str
    created_at: datetime


class OrderPage(BaseModel):
    items: list[Order]
    page: int
    page_size: int
    total: int


class Invoice(BaseModel):
    id: str = Field(json_schema_extra={"format": "uuid"})
    order_id: str = Field(json_schema_extra={"format": "uuid"})
    amount: str
    status: str
    issued_at: datetime


class PaymentEvent(BaseModel):
    event_id: str = Field(min_length=1, max_length=64)
    order_id: str = Field(json_schema_extra={"format": "uuid"})
    amount: str
    status: str = Field(description="'succeeded' or 'failed'")


class Error(BaseModel):
    detail: str


ERR: dict[int, dict[str, Any]] = {
    401: {"model": Error, "description": "Missing or invalid token"},
    403: {"model": Error, "description": "Role not allowed"},
    404: {"model": Error, "description": "Not found (or not visible to the caller)"},
    409: {"model": Error, "description": "Invalid state transition"},
    422: {"description": "Validation error"},
    503: {"model": Error, "description": "Database busy"},
}


def errs(*codes: int) -> dict[int | str, dict[str, Any]]:
    return {c: ERR[c] for c in codes}
