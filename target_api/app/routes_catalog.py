"""Health, customers and products."""

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from target_api.app import bugs
from target_api.app.db import execute, query
from target_api.app.deps import Auth, UuidPath, _customer_json, _product_json, parse_uuid, require
from target_api.app.schemas import Customer, Product, ProductPatch, errs
from target_api.app.state import db

router = APIRouter()


@router.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get(
    "/customers", response_model=list[Customer], tags=["customers"], responses=errs(401, 403)
)
def list_customers(p: Auth) -> Any:
    require(p, "admin", "staff")
    with db.connection() as conn:
        return [_customer_json(r) for r in query(conn, "SELECT * FROM customers ORDER BY name")]


@router.get(
    "/customers/{customer_id}",
    response_model=Customer,
    tags=["customers"],
    responses=errs(401, 403, 404, 422),
)
def get_customer(customer_id: UuidPath, p: Auth) -> Any:
    cid = parse_uuid(customer_id, "customer_id")
    if p.role == "customer" and p.customer_id != cid:
        raise HTTPException(404, "customer not found")
    with db.connection() as conn:
        rows = query(conn, "SELECT * FROM customers WHERE id = ?", (cid,))
    if not rows:
        raise HTTPException(404, "customer not found")
    return JSONResponse(_customer_json(rows[0]))  # JSONResponse: bypasses response_model filtering


@router.get("/products", response_model=list[Product], tags=["products"], responses=errs(401))
def list_products(p: Auth) -> Any:
    with db.connection() as conn:
        return [_product_json(r) for r in query(conn, "SELECT * FROM products ORDER BY sku")]


@router.get(
    "/products/{product_id}",
    response_model=Product,
    tags=["products"],
    responses=errs(401, 404, 422),
)
def get_product(product_id: UuidPath, p: Auth) -> Any:
    pid = parse_uuid(product_id, "product_id")
    with db.connection() as conn:
        rows = query(conn, "SELECT * FROM products WHERE id = ?", (pid,))
    if not rows:
        raise HTTPException(404, "product not found")
    return _product_json(rows[0])


@router.patch(
    "/products/{product_id}",
    response_model=Product,
    tags=["products"],
    responses=errs(401, 403, 404, 422),
)
def patch_product(product_id: UuidPath, body: ProductPatch, p: Auth) -> Any:
    """Update a product. Admin only."""
    if not bugs.on("B05"):
        require(p, "admin")
    pid = parse_uuid(product_id, "product_id")
    with db.write_lock, db.connection() as conn:
        rows = query(conn, "SELECT * FROM products WHERE id = ?", (pid,))
        if not rows:
            raise HTTPException(404, "product not found")
        for field in ("name", "price", "active"):
            value = getattr(body, field)
            if value is not None:
                execute(
                    conn,
                    f"UPDATE products SET {field} = ? WHERE id = ?",  # noqa: S608 - field comes from a fixed tuple of column names
                    (int(value) if field == "active" else value, pid),
                )
        conn.commit()
        return _product_json(query(conn, "SELECT * FROM products WHERE id = ?", (pid,))[0])
