# Acme Supply — Orders & Invoicing API: Requirements

Owner: Priya N. (Head of Operations, Acme Supply). Last reviewed: 2026-09-12.

## Why this API exists
Our trade customers place orders for hardware parts through partner portals. The API is the
single source of truth for customers, products, orders, payments and invoices. Mistakes here
cost us money directly (wrong totals, unpaid shipments) or expose one customer's data to another.

## Roles
- **Admin** — our back-office managers. Can do everything, including changing prices.
- **Staff** — warehouse and support agents. Can see all customers and orders and ship orders.
  Staff must **never** be able to change product prices.
- **Customer** — a trade customer's integration. Can only see and act on **their own** data.

Every call except the health check and the payment webhook needs `Authorization: Bearer <token>`.
A missing or unknown token is `401`. A known token with the wrong role is `403`.

## What matters most to us (in order)
1. Money is right to the cent.
2. Customers never see each other's data, and nothing internal leaks out of the API.
3. Orders move through their lifecycle correctly and payments can't be faked.
4. The API is fast enough for our partners' portals at peak (see the performance section).
