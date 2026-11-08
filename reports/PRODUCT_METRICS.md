# Product metrics audit: what FilmLook tracks, what it shows, what is missing (2026-11-08)

Source: read-only aggregate queries on the production database over SSH (counts only; no ids, addresses or
content were read), the dashboard code (`filmlook/usage_dashboard.py`, `product_analytics.py`, `llm_usage.py`,
`interest.py`) and `docs/usage-and-limits.md`.

## 1. What is tracked today

| metric | tracked | how | caveat |
|---|---|---|---|
| unique people | yes | `product_events.actor_key`: HMAC of the browser id or account id, rotated monthly | one person on two browsers counts twice; the hash changes each month by design, so return visits across months cannot be seen |
| daily / monthly active people | yes | distinct `actor_key` per day / month | |
| visits | yes | one `visit` event per browser session | |
| photo analyses started / succeeded / failed / limited | yes | `analysis_started`, `analysis_succeeded`, `analysis_failed`, `analysis_limited` with kind `dna` or `film` | |
| badge unlocks | yes (rows) | `user_badges` per identity; not as an event series | the dashboard shows counts, not a time series |
| share cards generated / downloaded / shared | yes | `card_generated`, `card_downloaded`, `share_initiated` | a share event is a button press |
| return visits | partly | "returning people" within a month (active on two or more days) | not across months |
| supporter-plan interest | yes (new today) | `supporter_interest` (one row per account or browser, hashed), `pricing_visits` (denominator); `/api/admin/interest` | 0 clicks and 1 pricing visit so far; nothing has been promoted yet |
| Gemini calls, tokens, latency, status | yes | `llm_api_requests` / `llm_api_attempts` per request, with estimated list-price cost and billing tier | tokens as reported by the provider; Gemini's OpenAI endpoint reports thinking only through totals |
| estimated cost per successful analysis | yes | dashboard "Cost per successful analysis" | estimate, not an invoice |
| test versus real activity | partly | the test account writes no product events, feedback, badges or results since 2026-11-06 (and since today's deployment its results live in a RAM-backed database); `llm_api_requests.tier = 'test'` marks its provider calls | rows written by the owner's own browsers and the smoke-test account before the rule are indistinguishable from real users (see `film-evaluation/reports/TEST_DATA_CLEANUP.md`) |

Not tracked: the funnel step between visit and upload (no "photo chosen" event), language of the session, device
class, the error reason behind a failed analysis, badge unlocks as events, and anything about who the person is
(by design: no email, no IP, no fingerprinting).

## 2. What the numbers say (2026-11-05 to 2026-11-08, all of it the launch week)

| figure | value |
|---|---|
| accounts | 2: the owner's `test` account and a smoke-test account |
| distinct people (hashed) with any event | 20 visitors across three days (7, 5, 11 per day); 1-2 people per day did anything beyond visiting |
| analyses started / succeeded / failed / limited | 7 / 5 / 1 / 1 (the failed one is today's synthetic probe; the limited one hit the one-a-day allowance) |
| cards generated / downloaded | 10 / 1 (one actor) |
| feedback rows / distinct identities | 7 / 5 |
| badges unlocked | 11 across 5 identities |
| DNA ratings | 5, mean fit 4.0 / 5 |
| Gemini requests | 8 (5 success, 2 cache hits, 1 blocked by the allowance); 10,379 + 23,627 input tokens, 4,236 + 1,277 output tokens |
| estimated Gemini cost | **$0.0168 in total; $0.0034 per successful analysis** (DNA reading about $0.0022, film analysis about $0.0078) |
| supporter interest | 0 interested, 1 pricing-page visitor |

Reading: there is no traction to measure yet. Every identity is consistent with the owner's own testing (the
cleanup report lists them); real user acquisition has not started, and the public domain changed today.

## 3. Conversion funnel as it can be built from existing events

visit → analysis_started → analysis_succeeded → feedback (rating) → card_generated → card_downloaded / share_initiated,
plus pricing_visits → supporter_interest. The only missing step is "photo chosen" between visit and analysis; adding
one `photo_selected` event (name only, same monthly hash) completes the funnel without new personal data.

## 4. API economics

- Observed: $0.0034 per successful analysis at Gemini 3.1 Flash-Lite list price (input $0.25 / M, output $1.50 / M);
  a DNA reading alone about $0.002, the optional film analysis about $0.008, so a photo that takes both costs about
  one US cent. These match the pre-launch estimate in `docs/usage-and-limits.md`.
- Guards in place: site-wide $3 / day and $30 / month estimated budgets, 1,500 requests / day, one analysis a day per
  free account and per network address, ten for supporters, atomic reservations, cache hits free. Worst case the
  free tier can cost $3 a day (about 300-900 analyses); actual spend so far is two cents.
- Supporter plan at ¥5 for 30 days (about $0.69): at ten analyses a day a heavy supporter could use 300 analyses
  = $0.65 (DNA only) to $3.0 (DNA + film) a month, so the price covers a light or moderate user and loses money on
  a heavy one. Break-even is about 200 DNA-only or 65 full analyses a month per supporter. Payment fees are not
  in this figure: on a ¥5 charge a fixed per-transaction fee would take a large share (Stripe's published fees for a
  Singapore account add a fixed component per transaction; the exact share must be read from Stripe's current
  pricing page and the approved account's payment methods before launch).
- Infrastructure: Oracle Always Free, $0; Gemini is the only variable cost.

## 5. Recommended minimal instrumentation (privacy-conscious, no new personal data)

1. `photo_selected` event (name only) to complete the funnel.
2. `badge_unlocked` event with the badge rarity only (not the badge id), so unlock rate per active person is a series.
3. Store the session language on `product_events` (two values), to split the funnel by language.
4. A `failure_reason` bucket on `analysis_failed` (image / content / provider / limit), already known to the API.
5. Keep the monthly hash rotation; report "returning within month" as the retention metric and say so on the dashboard.
6. Exclude the owner's identities going forward by using the test account only (its rows are not written), and clean
   the launch-week rows once the owner confirms the cleanup report.
