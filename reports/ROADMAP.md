# FilmLook: the next 14 days (proposal, 2026-11-08)

Priority order: real users and their judgement, then cost control, then monetisation readiness. No custom ML
training in this window.

## Days 1-2: clean baseline and instrumentation

- [ ] Owner confirms the test-data cleanup (`film-evaluation/reports/TEST_DATA_CLEANUP.md`); run it; counters start at zero.
- [ ] Add the four minimal events from `PRODUCT_METRICS.md` section 5 (photo selected, badge unlocked by rarity,
      language, failure reason) and show the funnel on `/admin/`.
- [ ] Convert the instance's ephemeral IP to a reserved one in the Oracle console after confirming it is free while
      assigned; re-create the old DuckDNS name pointing at the instance so the 301s work and Search Console's change
      of address can be filed.
- [ ] Search Console: verify the new property (the file is live), submit the sitemap, request indexing.

## Days 3-9: first real users through the Mini Tool and Xiaohongshu

- [ ] Publish the already-approved Mini Tool pointing people to `filmanalysis.duckdns.org` for the full analysis
      (no payment language, no external payment redirect in the Mini Tool).
- [ ] Two or three Xiaohongshu posts with share cards; measure visits → analyses → cards → ratings per day.
- [ ] Watch `/admin/`: cost per analysis, failure reasons, the one-a-day limit's hit rate (how often real people are
      turned away), and supporter interest clicks per pricing visitor.
- [ ] Collect the first human judgements of the analysis through the existing rating form (fit 1-5, "none" share);
      this is the only evidence that can later justify any ML work.

## Days 10-14: monetisation readiness, without charging anyone

- [ ] Decide the price from data: with Gemini at about $0.003-0.01 per analysis and a fixed per-transaction payment
      fee, ¥5 / 30 days only works for light use; evaluate ¥12-¥18 / 30 days, or ¥5 with five analyses a day, or
      a 90-day option, against the observed usage distribution. Keep "no automatic renewal".
- [ ] When Stripe approves the account: read the approved payment methods and the exact fees for CNY on the
      account; only then design Checkout (one-time, server-side price, webhook-verified activation) on top of the
      provider-neutral `payment_orders` core that already exists. Do not enable live payments before a test-mode run.
- [ ] If interest clicks stay near zero after a few hundred pricing visitors, keep the plan as a proposal and
      invest in the free product instead.

## ML in this window: pause (Option D), with one cheap optional pilot

- Custom training stays paused: the audit found no evidence that another run of the current recipe would change
  the outcome, and the product's value (judgement plus explanation) has no training label yet.
- Optional, only if a "find photos that look like this" feature is wanted for the Mini Tool or the site: measure the
  frozen student service on the E2 Micro itself (memory under 200 MB, latency), and run it behind the abstention
  gate as an exploratory feature with a "similar / not similar" button. Those votes become the first human
  similarity data. Gemini keeps producing the explanations (Option C in practice, with B's frozen DINOv2 excluded
  because it cannot run on the 1 GB instance).
- Resume training only if (a) a few hundred human similarity or film-look judgements exist, and (b) a pre-registered
  read on the untouched Met test half is defined before any new model is trained.

## What success looks like on day 14

A clean dashboard with real visitors, a measured analysis-to-card rate, a measured cost per analysis under one
US cent, a known interest-click rate on the pricing page, Stripe's approved capabilities and fees in hand, and a
price decision made from those numbers rather than from the ¥5 placeholder.
