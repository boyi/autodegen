from __future__ import annotations

import argparse
import csv
import math
from collections import namedtuple
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import fmean

import ccxt  # type: ignore
import polars as pl

Bar = namedtuple("Bar", ["timestamp", "open", "high", "low", "close", "volume"])
REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class DataQualityError(ValueError):
    pass


@dataclass
class BacktestResult:
    fills: list[dict]
    equity_curve: list[float]
    cash: float
    position: float
    days_elapsed: float


def validate_ohlcv(df: pl.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataQualityError(f"missing required columns: {missing}")

    if len(df) == 0:
        return

    null_rows = df.filter(pl.any_horizontal([pl.col(c).is_null() for c in REQUIRED_COLUMNS]))
    if len(null_rows) > 0:
        raise DataQualityError("null values found in required columns")

    if not df["timestamp"].is_sorted():
        raise DataQualityError("timestamps must be monotonic ascending")

    dupes = df.group_by("timestamp").len().filter(pl.col("len") > 1)
    if len(dupes) > 0:
        raise DataQualityError("duplicate timestamps detected")

    if len(df.filter(pl.col("high") < pl.max_horizontal("open", "close"))) > 0:
        raise DataQualityError("OHLC violation: high < max(open, close)")

    if len(df.filter(pl.col("low") > pl.min_horizontal("open", "close"))) > 0:
        raise DataQualityError("OHLC violation: low > min(open, close)")

    if len(df.filter(pl.col("volume") < 0)) > 0:
        raise DataQualityError("negative volume detected")


def _target_dir(data_dir: str, exchange: str, pair: str) -> Path:
    return Path(data_dir) / exchange / pair.replace("/", "-")


def fetch_data(exchange: str = "hyperliquid", pair: str = "BTC/USDC:USDC", timeframe: str = "1h", data_dir: str = "data") -> list[Path]:
    """Fetch OHLCV data via ccxt, store as monthly parquet files. Incremental."""
    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    target = _target_dir(data_dir, exchange, pair)
    target.mkdir(parents=True, exist_ok=True)

    since_ms = None
    files = sorted(target.glob("*.parquet"))
    if files:
        latest = pl.concat([pl.read_parquet(f).select("timestamp") for f in files]).sort("timestamp")
        if len(latest) > 0:
            since_ms = int(latest["timestamp"][-1].timestamp() * 1000) + 1

    rows = ex.fetch_ohlcv(pair, timeframe=timeframe, since=since_ms, limit=1000)
    if not rows:
        return []

    df = pl.DataFrame(rows, schema=REQUIRED_COLUMNS, orient="row").with_columns(
        pl.from_epoch("timestamp", time_unit="ms").alias("timestamp"),
        *[pl.col(c).cast(pl.Float64) for c in ["open", "high", "low", "close", "volume"]],
    )
    df = df.select(REQUIRED_COLUMNS).sort("timestamp")
    validate_ohlcv(df)

    out_paths: list[Path] = []
    monthly = df.with_columns(pl.col("timestamp").dt.strftime("%Y-%m").alias("month")).partition_by("month")
    for part in monthly:
        month = part["month"][0]
        out = target / f"{month}.parquet"
        payload = part.drop("month").sort("timestamp")
        if out.exists():
            payload = pl.concat([pl.read_parquet(out), payload]).unique(subset=["timestamp"], keep="last").sort("timestamp")
        validate_ohlcv(payload)
        payload.write_parquet(out)
        out_paths.append(out)

    return out_paths


def load_bars(data_dir: str = "data", exchange: str = "hyperliquid", pair: str = "BTC/USDC:USDC") -> list[Bar]:
    """Load all parquet files, return list of Bar namedtuples sorted by timestamp."""
    target = _target_dir(data_dir, exchange, pair)
    files = sorted(target.glob("*.parquet"))
    if not files:
        return []
    df = pl.concat([pl.read_parquet(f) for f in files]).sort("timestamp")
    validate_ohlcv(df)
    return [Bar(*row) for row in df.select(REQUIRED_COLUMNS).iter_rows()]


def synthetic_bars(n: int = 500) -> list[Bar]:
    bars: list[Bar] = []
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    price = 100.0
    for i in range(n):
        drift = math.sin(i / 20) * 0.5 + 0.1
        o = price
        c = max(1.0, o + drift)
        h = max(o, c) + 0.5
        l = min(o, c) - 0.5
        v = 1000.0 + (i % 25) * 10
        bars.append(Bar(ts, o, h, l, c, v))
        ts = ts + timedelta(hours=1)
        price = c
    return bars


def _fill_price(side: str, size: float, bar: Bar, slippage_factor: float) -> float:
    participation = 0.0 if bar.volume <= 0 else size / bar.volume
    slip_pct = slippage_factor * participation
    raw = bar.open * (1 + slip_pct) if side == "buy" else bar.open * (1 - slip_pct)
    return max(bar.low, min(bar.high, raw))


def run_backtest(strategy, bars: list[Bar], initial_cash: float = 10000, maker_fee: float = 0.0002, taker_fee: float = 0.0005, slippage_factor: float = 0.1) -> BacktestResult:
    """Run backtest with honest execution. Returns BacktestResult."""
    if not bars:
        return BacktestResult([], [], initial_cash, 0.0, 0.0)

    strategy.initialize([])
    cash = float(initial_cash)
    position = 0.0
    equity_curve: list[float] = []
    fills: list[dict] = []
    pending: list[dict] = []

    for bar in bars:
        for signal in pending:
            side = signal["side"]
            size = float(signal["size"])
            price = _fill_price(side, size, bar, slippage_factor)
            signed = size if side == "buy" else -size
            notional = size * price
            fee = notional * taker_fee
            cash += (-notional if side == "buy" else notional) - fee
            prev_pos = position
            position += signed
            pnl = 0.0
            if prev_pos != 0 and (prev_pos > 0 > position or prev_pos < 0 < position or position == 0):
                pnl = (bar.open - price) * prev_pos
            fills.append({"timestamp": bar.timestamp, "side": side, "size": size, "price": price, "fee": fee, "pnl": pnl, "entry_value": abs(prev_pos) * price, "is_close": prev_pos != 0})

        pending = list(strategy.on_bar(bar, {"cash": cash, "position": position, "equity": cash + position * bar.close}) or [])
        equity_curve.append(cash + position * bar.close)

    days_elapsed = max(1e-9, (bars[-1].timestamp - bars[0].timestamp).total_seconds() / 86400) if len(bars) > 1 else 1e-9
    return BacktestResult(fills=fills, equity_curve=equity_curve, cash=cash, position=position, days_elapsed=days_elapsed)


def trade_return_sharpe(result: BacktestResult) -> float:
    closes = [f for f in result.fills if f.get("is_close") and f.get("entry_value", 0) > 0]
    if len(closes) < 2:
        return 0.0
    returns = [f["pnl"] / f["entry_value"] for f in closes]
    mean_r = fmean(returns)
    var = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
    std = math.sqrt(var)
    if std < 1e-12:
        return 0.0
    years = max(result.days_elapsed / 365.0, 1e-9)
    trades_per_year = len(returns) / years
    return (mean_r / std) * math.sqrt(trades_per_year)


def max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    mdd = 0.0
    for e in equity_curve:
        peak = max(peak, e)
        if peak > 0:
            mdd = max(mdd, (peak - e) / peak)
    return mdd


def walk_forward_splits(bars: list[Bar], n_folds: int) -> list[tuple[list[Bar], list[Bar]]]:
    fold_size = len(bars) // n_folds
    splits: list[tuple[list[Bar], list[Bar]]] = []
    for i in range(1, n_folds):
        train = bars[: i * fold_size]
        test = bars[i * fold_size : (i + 1) * fold_size]
        if test:
            splits.append((train, test))
    return splits


def _append_result_row(path: Path, hypothesis: str, avg_sharpe: float, avg_maxdd: float, avg_trades: float, status: str) -> None:
    header = ["timestamp", "hypothesis", "avg_sharpe", "maxdd", "trades", "status"]
    exists = path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        if not exists:
            w.writerow(header)
        w.writerow([datetime.now(UTC).isoformat(), hypothesis, f"{avg_sharpe:.6f}", f"{avg_maxdd:.6f}", f"{avg_trades:.2f}", status])


def evaluate(strategy_cls, bars: list[Bar], n_folds: int = 8, validation_pct: float = 0.15):
    if len(bars) < max(n_folds * 5, 20):
        bars = synthetic_bars(500)

    cut = int(len(bars) * (1 - validation_pct))
    wf_bars = bars[:cut]
    val_bars = bars[cut:]

    folds = walk_forward_splits(wf_bars, n_folds)
    fold_metrics = []
    for train, test in folds:
        strat = strategy_cls()
        strat.initialize(train)
        bt = run_backtest(strat, test)
        trades = len([f for f in bt.fills if f.get("is_close")])
        fold_metrics.append((trade_return_sharpe(bt), max_drawdown(bt.equity_curve), trades))

    avg_sharpe = fmean(m[0] for m in fold_metrics) if fold_metrics else 0.0
    avg_maxdd = fmean(m[1] for m in fold_metrics) if fold_metrics else 1.0
    avg_trades = fmean(m[2] for m in fold_metrics) if fold_metrics else 0.0
    worst_fold_sharpe = min((m[0] for m in fold_metrics), default=-1.0)

    gates_pass = (
        avg_sharpe >= 1.0
        and avg_maxdd <= 0.30
        and avg_trades >= 5
        and worst_fold_sharpe >= 0
    )

    print(f"avg_sharpe={avg_sharpe:.6f}")
    print(f"avg_maxdd={avg_maxdd:.6f}")
    print(f"avg_trades={avg_trades:.2f}")
    print(f"worst_fold_sharpe={worst_fold_sharpe:.6f}")
    print(f"hard_gates={'PASS' if gates_pass else 'FAIL'}")

    validation_sharpe = 0.0
    if gates_pass and val_bars:
        vbt = run_backtest(strategy_cls(), val_bars)
        validation_sharpe = trade_return_sharpe(vbt)
    print(f"validation_sharpe={validation_sharpe:.6f}")

    _append_result_row(Path("results.tsv"), getattr(strategy_cls, "name", "strategy"), avg_sharpe, avg_maxdd, avg_trades, "ok")

    return {
        "avg_sharpe": avg_sharpe,
        "avg_maxdd": avg_maxdd,
        "avg_trades": avg_trades,
        "worst_fold_sharpe": worst_fold_sharpe,
        "hard_gates": gates_pass,
        "validation_sharpe": validation_sharpe,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    fetch_p = sub.add_parser("fetch")
    fetch_p.add_argument("--exchange", default="hyperliquid")
    fetch_p.add_argument("--pair", default="BTC/USDC:USDC")
    fetch_p.add_argument("--timeframe", default="1h")

    sub.add_parser("eval")

    args = parser.parse_args()
    if args.cmd == "fetch":
        paths = fetch_data(args.exchange, args.pair, args.timeframe)
        print(f"wrote {len(paths)} parquet file(s)")
        for p in paths:
            print(p)
    elif args.cmd == "eval":
        from strategy import Strategy

        bars = load_bars()
        evaluate(Strategy, bars)
    else:
        parser.print_help()
