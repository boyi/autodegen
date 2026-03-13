# autodegen

Minimal autoresearch-style trading research loop.

## Architecture
- `prepare.py` — immutable oracle (fetch, backtest, evaluate)
- `strategy.py` — only mutable strategy file
- `degen.md` — agent firmware (loop instructions)

## Usage
```bash
uv run python prepare.py fetch --exchange hyperliquid --pair BTC/USDT --timeframe 1h
uv run python prepare.py eval
uv run python strategy.py
uv run pytest tests/ -v
```
