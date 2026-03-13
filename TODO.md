# TODO.md — autodegen

## Phase 0 — Skeleton (get the loop running)
- [ ] Init repo: pyproject.toml, uv, project structure
- [ ] Data ingest: Hyperliquid BTC/USDT perps OHLCV (1h) via ccxt
- [ ] Store as parquet (monthly files)
- [ ] Data quality gate (monotonic timestamps, no dupes, OHLC checks)
- [ ] config.md + parser with defaults
- [ ] Basic Bar dataclass + data loader

## Phase 1 — Backtester core
- [ ] Event-driven backtest loop with pending order queue (one-bar delay)
- [ ] Slippage model (linear impact, base volume)
- [ ] Fee model (Hyperliquid maker/taker rates)
- [ ] Portfolio tracking (equity curve, positions, margin)
- [ ] Strategy interface (on_bar, initialize, parameters)
- [ ] Reference strategy: simple EMA crossover (agent's warm-start template)

## Phase 2 — Eval pipeline
- [ ] Three-way data split (walk-forward / validation / test)
- [ ] Walk-forward expanding window (7 folds)
- [ ] Trade-return Sharpe + max drawdown + Calmar
- [ ] Composite score with AST complexity check
- [ ] Hard gates (Sharpe > 1.0, MaxDD < 30%, trades > 50, worst fold > 0)
- [ ] Oracle hash + data snapshot hash + config hash in results
- [ ] results.tsv writer

## Phase 3 — Agent loop
- [ ] degen.md template
- [ ] Agent loop: read config → read degen.md → propose → edit → commit → eval → keep/revert
- [ ] Git commit enforcement (exactly 1 file changed)
- [ ] Error handling (try/except, auto-reset, never crash)
- [ ] SandboxRunner abstraction (systemd backend for us)
- [ ] First overnight run 🎉

## Phase 4 — Expand
- [ ] Add ETH/USDT, SOL/USDT
- [ ] Funding rate data + forward-fill
- [ ] Multi-pair config support
- [ ] Paper trader (websocket, simulated fills)

## Phase 5 — Polymarket domain plugin
- [ ] Polymarket CLOB data ingest
- [ ] Binary outcome eval (EV-based, not Sharpe)
- [ ] Prediction strategy interface
- [ ] Separate oracle module
