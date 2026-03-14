# autodegen

Minimal autoresearch-style trading research loop. Inspired by Andrej Karpathy autoresearch design and approach [here](https://github.com/karpathy/autoresearch).

## Architecture
- `prepare.py` — immutable oracle (fetch, backtest, evaluate)
- `strategy.py` — only mutable strategy file
- `degen.md` — agent firmware (loop instructions)

## Usage
```bash
uv run python prepare.py fetch --exchange binance --pair BTC/USDT:USDT --timeframe 1h --start 2020-01-01T00:00:00Z
uv run python prepare.py validate --exchange binance --pair BTC/USDT:USDT --timeframe 1h
uv run python prepare.py eval
uv run python strategy.py
```
