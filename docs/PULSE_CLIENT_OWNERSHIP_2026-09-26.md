# Pulse client ownership repair — 2026-09-26

## Evidence and cause

Operator-supplied container log `compose-connect-mobile-panel-4jhz46-hermes-agent-1-20260926_180229.log.txt` contains 21 market_analyst node failures across the 08:01, 12:01 and 16:01 UTC cycles on September 26. CCXT reports `ExchangeClosedByUser: binanceusdm instance was closed by the user` during fetch_ohlcv.

Previously pulse borrowed cached get_market(), then closed it. The trading graph borrows that same instance. The message refers to a programmatic close, not an operator action. Local CCXT 4.5.59 reopens sessions differently from the production traceback; the earlier fresh-process market smoke did not exercise this lifecycle and was insufficient.

## Narrow correction

Pulse now constructs its own MarketData(settings.exchange_primary) and closes it in finally, including cancellation while reading positions or candles. Shared trading-client ownership, strategy, risk, stage, schedule and model configuration are unchanged.

## Verification

- Three initial ownership/error/cancellation regression tests failed before the fix and passed after it.
- Added coverage for cancellation during open_positions, as requested by the independent read-only ownership/safety reviewer; no design blocker reported.
- Full local suite: 1315 passed, 4 skipped; 62.27 seconds.
- CI lint command scope `ruff check src tests`: passed.
- `uv build`: source distribution and wheel built; this is not a production Docker build.
- Actual public Binance reads with two real MarketData adapters: read shared, read owned, close owned, read shared again. Shared session stayed open and identical; each shared read returned two closed bars. No orders or LLM calls. This probes real adapter isolation, not the complete production pulse sequence.

## Release boundary

LOCAL ONLY. Existing authorization covers diagnostics in paper, not this repair deployment. Obtain explicit repair-only paper approval and isolate the release from unrelated local commits. Verify the deployed artifact and subsequent scheduled seven-symbol tick before declaring production recovery. Successful recovery is not profitability, an executed trade, or proof of the rolling-ten objective.
