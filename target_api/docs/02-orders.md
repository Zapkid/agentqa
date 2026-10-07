# Orders

## Creating an order
- `POST /orders` with a list of items (`product_id`, `quantity`). Customers always order for
  themselves; staff and admins must say which `customer_id` they are ordering for.
- `quantity` is a whole number from **1 to 1000**. Zero, negative or larger quantities are
  rejected with `422`. We have been burned by negative quantities producing credit notes.
- Only active products can be ordered. Unknown or inactive products are `422`.
- A new order starts as `pending` and the response is `201` with the full order.

## Prices and totals
- Unit prices are decimal strings with up to **three** decimals (we sell some parts by the
  millimetre and the piece, e.g. a bolt at `0.125`).
- Each line total is `unit_price × quantity`, **rounded half-up to the cent**, computed with
  decimal arithmetic. Example: 1 × `0.125` is `0.13`; 1 × `2.345` is `2.35`.
- The order total is the sum of the line totals. Money is always returned as a string.

## Retries and duplicate submissions
Partner portals retry on timeouts. If a request carries an `Idempotency-Key` header and we have
already created an order for that key (for the same customer), we must return the **original**
order with `200` and must not create a second order.

## Listing orders
- `GET /orders` returns orders newest first, as `{items, page, page_size, total}`.
- Pages are 1-based. `page_size` defaults to 20 and may not exceed **100** (`422` above that).
  Page 2 with `page_size=10` must contain exactly the 11th to 20th orders — nothing skipped,
  nothing repeated.
- `status` filters by one of `pending`, `paid`, `cancelled`, `shipped`. Any other value is a
  client error (`422`); silently ignoring a typo would show a user the wrong orders.
- `created_from` filters to orders created at or after an ISO 8601 timestamp. Timestamps with
  an offset (e.g. `+02:00`) must be converted to UTC before comparing; timestamps without an
  offset are treated as UTC. Our Berlin office relies on this.
- Customers only ever see their own orders in the list.

## Reading an order
- `GET /orders/{order_id}`. A customer asking for someone else's order gets `404` (not `403`,
  so they can't even learn that the order exists).
- An `order_id` that is not a UUID is a client error (`422`), never a server error.

## Order lifecycle
```
pending --(payment webhook)--> paid --(ship, staff/admin)--> shipped
pending --(cancel)--> cancelled
paid    --(cancel)--> cancelled
```
- Cancelling is allowed for the owner, staff and admins, only from `pending` or `paid`;
  otherwise `409`.
- Shipping is staff/admin only, and **only from `paid`**. Shipping a cancelled or unpaid order
  is a `409` — a cancelled order leaving the warehouse is a real loss for us.
