# Performance expectations

Our partner portals show the order list on their landing page, and the payment provider sends
bursts of webhooks at the end of each business day.

## Service levels (measured at the API, excluding network)
- **Order list** (`GET /orders`, default page size, typical filters): p95 under **300 ms** and
  p99 under **800 ms** with **20 concurrent users**, with up to 200,000 orders in the system.
- **Payment webhook** (`POST /webhooks/payment`): sustain at least **50 requests per second**.
- **Error rate** under 1% at the load levels above.
- Memory must stay flat over a working day; a slow leak forces restarts in the middle of
  trading.

## Expected traffic mix at peak
Roughly 60% order list, 15% single order reads, 10% product reads, 10% order creation,
5% webhooks.
