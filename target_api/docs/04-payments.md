# Payments and invoices

## Payment webhook
Our payment provider calls `POST /webhooks/payment` when a payment succeeds or fails. It does
**not** use a bearer token. Instead it signs the raw request body with our shared secret:
the `X-Signature` header is the hex-encoded **HMAC-SHA256** of the exact body bytes.

- A missing or wrong signature must be rejected with `401`. Anyone on the internet can call
  this URL; without the signature check, anyone could mark an order as paid.
- The body is `{event_id, order_id, amount, status}`. `status` is `succeeded` or `failed`.
- Events are delivered at least once. A repeated `event_id` is acknowledged and ignored.
- A successful payment moves a `pending` order to `paid` and issues an invoice for the order
  total. The amount must match the order total exactly, otherwise `409`.

## Invoices
- `GET /orders/{order_id}/invoice` returns the invoice for a paid order, with the same
  visibility rules as the order itself. `404` if there is no invoice yet.
