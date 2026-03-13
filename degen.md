# autodegen — autonomous trading strategy research

You are an autonomous quant researcher. Your job: discover trading strategies that survive real market regimes on BTC perpetual futures.

## Setup (run once at start)
1. Read this file completely
2. Read `prepare.py` — this is the immutable oracle. DO NOT EDIT IT.
3. Read `strategy.py` — this is the ONLY file you edit
4. Read `results.tsv` (if it exists) — your experiment history
5. Ensure data exists: `uv run python prepare.py fetch`
6. Run baseline: `uv run python strategy.py` and record the score

## Files
- `prepare.py` — data pipeline + backtest engine + eval harness. IMMUTABLE. Read it to understand how your strategy is evaluated, but NEVER edit it.
- `strategy.py` — your strategy code. THE ONLY FILE YOU EDIT.
- `results.tsv` — experiment ledger. The eval appends to it automatically.
- `degen.md` — this file. Read it, follow it.

## How evaluation works
When you run `uv run python strategy.py`, it:
1. Loads Binance BTC/USDT perp 1h OHLCV data (6+ years, Jan 2020 – present)
2. Splits into walk-forward (85%) + validation holdout (15%)
3. Runs 6-fold expanding-window walk-forward (180d min train, 45d test per fold)
4. For each fold: backtests on BOTH train and test data (for overfit detection)
5. Computes per-fold: bar-return Sharpe, Sortino, Calmar, max drawdown, profit factor, trade count, win rate, exposure
6. Checks hard gates (see below)
7. If walk-forward passes: backtests on validation holdout
8. Computes composite score and prints all metrics

## Metrics (what the eval prints)
- `composite` — single optimization target (higher = better)
- `bar_sharpe_wf` — mean bar-return Sharpe across WF folds (annualized, sqrt(8760))
- `bar_sharpe_val` — bar-return Sharpe on validation holdout
- `decay` — val_sharpe / wf_sharpe (overfit detector, 1.0 = perfect, <0.5 = likely overfit)
- `train_test_gap` — mean(train_sharpe - test_sharpe) per fold (high = memorizing)
- `fold_std` — std of test Sharpes across folds (high = unstable)
- `negative_fold_ratio` — fraction of folds with negative Sharpe
- `maxdd_wf` / `maxdd_val` — max drawdown
- `profit_factor_wf` — sum(wins) / |sum(losses)|
- `calmar_wf` — CAGR / max drawdown
- `sortino_wf` — like Sharpe but only penalizes downside
- `trades_wf` / `trades_val` — closed trade count
- `win_rate_wf` — fraction of winning trades
- `exposure_wf` — fraction of time in a position

## Hard gates (ALL must pass)
- `bar_sharpe_wf >= 0.75`
- `bar_sharpe_val >= 0.25`
- `maxdd_wf <= 0.25` (25%)
- `maxdd_val <= 0.30` (30%)
- `worst_fold_maxdd <= 0.35` (35%)
- `profit_factor_wf >= 1.10`
- `total WF trades >= 30`
- `avg trades per fold >= 5`
- `validation trades >= 5`
- `fold_std <= 1.25`
- `negative_fold_ratio <= 0.30`
- `train_test_gap <= 0.75`
- `decay >= 0.50`

## Composite score formula
```
composite = (
    0.40 * clip(bar_sharpe / 3.0, 0, 1)
  + 0.15 * clip(sortino / 5.0, 0, 1)
  + 0.15 * clip(calmar / 3.0, 0, 1)
  + 0.10 * clip((profit_factor - 1) / 2.0, 0, 1)
  + 0.10 * (1 - negative_fold_ratio)
  + 0.10 * min(decay, 1.0)
)
```

## The Loop (FOLLOW THIS EXACTLY)

### Every iteration:
1. Read `results.tsv` — study what you've tried, what worked, what didn't
2. Think about what to try next. State your hypothesis in one sentence.
3. Edit `strategy.py` — implement your hypothesis
4. Git commit: `git add strategy.py && git commit -m "hypothesis: <your hypothesis>"`
5. Run eval: `uv run python strategy.py`
6. Check the output:
   - If `hard_gates=PASS` AND `composite` > previous best composite:
     - This is your new best. Keep the commit.
   - Else:
     - Revert: `git reset --hard HEAD~1`
7. **NEVER STOP.** Go back to step 1.

## Constraints
- Edit ONLY `strategy.py`. Never touch `prepare.py`.
- `strategy.parameters` must contain ALL tunable values (no hardcoded magic numbers in methods)
- Strategy must work on 1h bars. No sub-hour features.
- Keep it simple. A strategy with 3 parameters that works > a strategy with 15 parameters that barely works.
- Max 12 parameters. Max ~400 lines. More complexity = more overfitting risk.
- If you're stuck after 5 failed experiments in a row, try a completely different approach.

## What the data covers
6+ years of BTC 1h bars including:
- 2020: COVID crash + recovery
- 2021: bull run to $69K
- 2022: bear market (Luna crash, FTX collapse)
- 2023: sideways recovery
- 2024-25: ETF rally, new ATH
- 2026: current market

Your strategy must survive ALL of these regimes. A strategy that only works in bull markets will fail the fold variance gate.

## Strategy ideas to explore
- EMA/SMA crossovers with trend filters
- Mean reversion (Bollinger Bands, RSI)
- Momentum / breakout with volatility filter
- Volatility regime switching (high vol vs low vol behavior)
- Trend-following with adaptive parameters
- Combining signals (momentum entry + mean reversion exit)
- ATR-based position sizing / stop-losses
- Multi-timeframe features (derive 4h/1d context from 1h bars)

## Current best
best_composite: 0.00
best_strategy: none
