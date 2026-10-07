# Customers and products

## Customers
- `GET /customers` (staff and admin) lists customers.
- `GET /customers/{customer_id}` returns `id`, `name`, `email` and `created_at` — **only these
  fields**. We hold internal risk information about customers that must never appear in any
  API response. A customer may read only their own record (`404` for anyone else's).

## Products
- Any authenticated caller can list and read products.
- `PATCH /products/{product_id}` changes name, price or active flag. **Admin only.** Staff and
  customers get `403`. Price changes by anyone else would be a fraud vector.
- Prices are decimal strings with up to three decimals.
