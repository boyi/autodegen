# autodegen — autonomous trading strategy research

You are an autonomous quant researcher. Your job: discover trading strategies that survive real market regimes on BTC perpetual futures.

## Setup (run once at start)
1. Read this file completely
2. Read `prepare.py` — this is the immutable oracle. DO NOT EDIT IT.
3. Read `strategy.py` — this is the ONLY file you edit
4. Read `leaderboard.tsv` (if it exists) — best strategies across all agents. This is your benchmark to beat.
5. Read `results.tsv` (if it exists) — your local experiment history
6. Fetch the canonical benchmark dataset: `uv run python prepare.py fetch --exchange binance --pair BTC/USDT:USDT --timeframe 1h --start 2020-01-01T00:00:00Z`
7. Validate the dataset: `uv run python prepare.py validate --exchange binance --pair BTC/USDT:USDT --timeframe 1h`
8. Run baseline: `uv run python strategy.py` and record the score

## Files
- `prepare.py` — data pipeline + backtest engine + eval harness. IMMUTABLE. Read it to understand how your strategy is evaluated, but NEVER edit it.
- `strategy.py` — your strategy code. THE ONLY FILE YOU EDIT.
- `results.tsv` — experiment ledger. The eval appends to it automatically.
- `leaderboard.tsv` — hall of fame. PASS-only results across all agents/machines. Same columns as results.tsv + `source` column. Read this first to know the current best.
- `degen.md` — this file. Read it, follow it.

## How evaluation works
When you run `uv run python strategy.py`, it:
1. Validates and loads the canonical Binance BTC/USDT perpetual 1h OHLCV dataset from January 1, 2020 to present
2. Splits into walk-forward (85%) + validation holdout (15%)
3. Runs 6-fold expanding-window walk-forward (180d initial train, 45d test per fold)
4. For each fold: backtests on BOTH train and test data (for overfit detection)
5. Computes per-fold: bar-return Sharpe, Sortino, Calmar, max drawdown, profit factor, trade count, win rate, exposure
6. Checks hard gates (see below)
7. If walk-forward passes: backtests on validation holdout
8. Computes composite score and prints all metrics
9. Refuses evaluation if the real dataset is missing, stale, corrupted, or too short for the canonical benchmark

## Metrics (what the eval prints)
- `composite` — single optimization target (higher = better)
- `bar_sharpe_wf` — mean bar-return Sharpe across WF folds (annualized, sqrt(8760))
- `bar_sharpe_val` — bar-return Sharpe on validation holdout
- `decay` — val_sharpe / wf_sharpe (overfit detector, 1.0 = perfect, <0.5 = likely overfit)
- `fold_regime_gap` — mean(train_sharpe - test_sharpe) per fold; measures earlier-era vs later-era performance drift inside each WF split
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
- `fold_regime_gap <= 0.75`
- `decay >= 0.50`

## Composite score formula
```
composite = (
    0.35 * clip(bar_sharpe_wf / 3.0, 0, 1)
  + 0.10 * clip(bar_sharpe_val / 2.0, 0, 1)
  + 0.15 * clip(sortino_wf / 5.0, 0, 1)
  + 0.15 * clip(calmar_wf / 3.0, 0, 1)
  + 0.10 * clip((profit_factor_wf - 1.0) / 2.0, 0, 1)
  + 0.10 * (1 - negative_fold_ratio)
  + 0.05 * min(decay, 1.0)
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
     - Append the PASS row to `leaderboard.tsv` with your source name (e.g. `opus-manual`, `scout1-momentum`). Use the same columns as results.tsv + a `source` column at the end.
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
The canonical dataset is Binance `BTC/USDT:USDT` 1h bars from January 1, 2020 onward. Once `prepare.py validate` passes, it covers:
- 2020: COVID crash + recovery
- 2021: bull run to $69K
- 2022: bear market (Luna crash, FTX collapse)
- 2023: sideways recovery
- 2024-25: ETF rally, new ATH
- 2026: current market

Do not trust any result until `uv run python prepare.py validate` passes. A strategy that only works in bull markets will fail the fold variance gate.

## Strategy search doctrine
Do not anchor on canned indicator templates. Work backward from the evaluation metrics and search for simple trading rules that can make money robustly on BTC perpetual futures after fees and slippage.

Your job is to discover structural edges, not just remix common indicators.

For each new hypothesis:
- Start from the current bottleneck in `results.tsv`:
- weak walk-forward Sharpe,
- weak validation Sharpe,
- poor decay,
- high fold variance,
- excessive drawdown,
- too few trades,
- low profit factor.
- Ask what market behavior could fix that bottleneck while preserving profitability.
- Form one clear mechanism-level hypothesis before editing `strategy.py`.

Think in terms of edge archetypes, not indicator names:
- trend persistence,
- momentum ignition or continuation,
- breakout from compression,
- mean reversion after exhaustion,
- volatility expansion vs compression,
- regime switching,
- asymmetry between long and short behavior,
- path-dependent exits,
- risk management as edge,
- participation filters,
- market state filters derived from 1h bars.

Creative and degenerate ideas are allowed if they remain:
- simple,
- explainable,
- cost-aware,
- parameter-light,
- implementable from 1h bars only.

Avoid local search traps:
- do not spend many iterations only nudging thresholds or lookbacks,
- if several experiments fail in the same family, pivot to a different family,
- prefer changing one structural dimension over micro-tuning many parameters.

Judge ideas by these questions:
- Why should this make money on BTC perps specifically?
- What regime(s) should it exploit?
- Why should it survive unseen periods rather than only one era?
- Which evaluation bottleneck is it meant to improve?
- Is the rule simple enough to generalize?

## Leverage constraint
The backtest engine enforces a **max leverage of 3x** (`max_leverage=3.0` in `run_backtest()`). Before any buy signal executes, position size is capped so total position notional never exceeds 3× equity. This reflects real exchange margin limits and prevents unrealistic backtests.

Previous results (v1, unconstrained leverage) are archived in `results_v1_unconstrained.tsv` and `leaderboard_v1_unconstrained.tsv`.

## Current best
best_composite: 0.975
best_strategy: ema_20_50_hh_macro_smooth_v2
