# S2-xsec CHECKPOINT (2026-06-29)

- **Verdict: HONEST NEGATIVE on the pre-registered XS family** (2A funding, 2D vol-disp, 2B OI-growth, 2.3 cross-venue) — no new sleeve for the gem. Killer = two-leg turnover cost (~100%/day reshuffle); 2A's sign is also wrong-way (XS funding mean-reverts; gross annSh −1.38, MC-sign p=1.0).
- **Governance flag:** live `funding_squeeze` re-tested at 4h bull/bear w/ cluster-aware nulls = **suggestive-in-bull (excess +217bps i.i.d. p=1e-4 BUT macro-event t=1.50 NS, median +25bps), NOT-established-in-bear (n=35→21 events, t=1.23)**. Sign does NOT flip → not pure bull-beta, but NOT robustly confirmed. Don't treat the live edge as confirmed.
- **L0 leak test PASS**; engine + builders adversarially reviewed (no leak). One driver same-day regime look-ahead FOUND→FIXED→now tested (`daily_regime_on_grid`). Annualisation bug fixed (post-hoc inverted-2A lead is +0.58 true annual, not +2.16).
- **Exploratory (post-hoc, NOT a result):** inverted-slow XS funding reversion (+0.58 annual @14d) → needs fresh pre-registered + deflated OOS test.
- Code `src/hermes/research/s2_xsec.py` (22 tests, all green; 72 green incl. adjacent). Trials → `TRIAL_REGISTRY.md` (4 pre-reg + 18 exploratory, crypto book). Shadow size 0, nothing live. **STOP.**
