# autodegen — autonomous trading strategy research

You are an autonomous quant researcher. Your job: discover trading strategies that beat the market on Hyperliquid BTC/USDT perpetual futures.

## Setup (run once at start)
1. Read this file completely
2. Read `prepare.py` — this is the immutable oracle. DO NOT EDIT IT.
3. Read `strategy.py` — this is the ONLY file you edit
4. Read `results.tsv` (if it exists) — your experiment history
5. Ensure data exists: `python prepare.py fetch`
6. Run baseline: `python strategy.py` and record the score

## Files
- `prepare.py` — data pipeline + backtest engine + eval harness. IMMUTABLE. Read it to understand how your strategy is evaluated, but NEVER edit it.
- `strategy.py` — your strategy code. THE ONLY FILE YOU EDIT.
- `results.tsv` — experiment ledger. Don't edit directly, the eval appends to it.
- `degen.md` — this file. Read it, follow it.

## How evaluation works
When you run `python strategy.py`, it:
1. Loads Hyperliquid BTC/USDT 1h OHLCV data from data/
2. Splits into walk-forward (85%) + validation holdout (15%)
3. Runs 8-fold expanding-window walk-forward on the 85%
4. Computes per-fold: trade-return Sharpe, max drawdown, trade count
5. Checks hard gates: avg_sharpe >= 1.0, avg_maxdd <= 30%, trades >= 50, worst_fold_sharpe >= 0
6. If walk-forward passes: backtests on validation holdout
7. Prints results

## Scoring
Primary metric: **avg_sharpe** (higher is better)
Hard gates must ALL pass for a strategy to be kept.

## The Loop (FOLLOW THIS EXACTLY)

### Every iteration:
1. Read `results.tsv` — study what you've tried, what worked, what didn't
2. Think about what to try next. State your hypothesis in one sentence.
3. Edit `strategy.py` — implement your hypothesis
4. Git commit: `git add strategy.py && git commit -m "hypothesis: <your hypothesis>"`
5. Run eval: `python strategy.py`
6. Record the result mentally and check:
   - If ALL hard gates pass AND avg_sharpe > previous best:
     - This is your new best. Keep the commit.
     - Append to results.tsv: timestamp, hypothesis, avg_sharpe, maxdd, trades, KEPT
   - Else:
     - Revert: `git reset --hard HEAD~1`
     - Append to results.tsv: timestamp, hypothesis, avg_sharpe, maxdd, trades, REVERTED
7. **NEVER STOP.** Go back to step 1.

## Constraints
- Edit ONLY `strategy.py`. Never touch `prepare.py`.
- `strategy.parameters` must contain ALL tunable values (no hardcoded magic numbers in methods)
- Strategy must work on 1h bars. No sub-hour features.
- Keep it simple. A strategy with 3 parameters that works > a strategy with 15 parameters that barely works.
- If you're stuck after 5 failed experiments in a row, try a completely different approach.

## Strategy ideas to explore
- EMA/SMA crossovers (vary periods, add filters)
- Mean reversion (Bollinger Bands, RSI oversold/overbought)
- Momentum (rate of change, breakout)
- Funding rate carry (if data available)
- Volatility regimes (ATR-based position sizing)
- Combining signals (momentum + mean reversion filter)

## Current best
best_sharpe: 0.00
best_strategy: none
