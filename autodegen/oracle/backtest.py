from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

import polars as pl


@dataclass(slots=True)
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    funding_rate: float | None = None
    quote_volume: float | None = None
    symbol: str = "BTC/USDT"


@dataclass(slots=True)
class Position:
    symbol: str
    size: float = 0.0  # signed base units (+ long, - short)
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    entry_value: float = 0.0


@dataclass(slots=True)
class FeeModel:
    maker_rate: float = 0.0002
    taker_rate: float = 0.0005


@dataclass(slots=True)
class Signal:
    symbol: str
    side: Literal["buy", "sell"]
    size: float
    order_type: Literal["market"] = "market"


@dataclass(slots=True)
class Fill:
    symbol: str
    side: Literal["buy", "sell"]
    size: float
    price: float
    fee: float
    timestamp: datetime
    pnl: float = 0.0
    is_close: bool = False
    entry_value: float = 0.0


@dataclass(slots=True)
class Portfolio:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    equity_curve: list[float] = field(default_factory=list)
    total_equity: float = 0.0

    def get_position(self, symbol: str) -> Position:
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        return self.positions[symbol]

    def update_equity(self, current_price: float) -> None:
        equity = self.cash
        for position in self.positions.values():
            position.unrealized_pnl = (current_price - position.entry_price) * position.size
            equity += position.size * current_price
        self.total_equity = equity
        self.equity_curve.append(equity)


@dataclass(slots=True)
class BacktestConfig:
    initial_cash: float = 10_000.0
    fee_model: FeeModel = field(default_factory=FeeModel)
    slippage_impact_factor: float = 0.1
    max_position_pct: float = 0.25


@dataclass(slots=True)
class BacktestResult:
    fills: list[Fill]
    portfolio: Portfolio
    bars_processed: int
    days_elapsed: float


class StrategyLike(Protocol):
    name: str
    parameters: dict

    def initialize(self, train_data: list[Bar]) -> None: ...

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Signal]: ...

    def on_fill(self, fill: Fill) -> None: ...


def load_bars(data_dir: Path, exchange: str, pair: str) -> list[Bar]:
    base = data_dir / exchange / pair.replace("/", "-")
    if not base.exists():
        return []

    files = sorted(base.glob("*.parquet"))
    if not files:
        return []

    df = pl.concat([pl.read_parquet(f) for f in files]).sort("timestamp")
    bars: list[Bar] = []
    for row in df.iter_rows(named=True):
        bars.append(
            Bar(
                timestamp=row["timestamp"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
                funding_rate=float(row["funding_rate"]) if row.get("funding_rate") is not None else None,
                quote_volume=float(row["quote_volume"]) if row.get("quote_volume") is not None else None,
                symbol=pair,
            )
        )
    return bars


def calculate_fill_price(signal: Signal, bar: Bar, impact_factor: float) -> float:
    if bar.volume <= 0:
        slippage_pct = 0.0
    else:
        participation_rate = signal.size / bar.volume
        slippage_pct = participation_rate * impact_factor

    raw = bar.open * (1 + slippage_pct) if signal.side == "buy" else bar.open * (1 - slippage_pct)
    return max(bar.low, min(bar.high, raw))


def _clip_signal_size(signal: Signal, bar: Bar, portfolio: Portfolio, max_position_pct: float) -> float:
    if signal.size <= 0:
        return 0.0

    position = portfolio.get_position(signal.symbol)
    equity_ref = portfolio.total_equity if portfolio.total_equity > 0 else portfolio.cash
    max_notional = max(0.0, equity_ref * max_position_pct)
    if max_notional <= 0 or bar.open <= 0:
        return 0.0

    max_abs_size = max_notional / bar.open
    signed_target = signal.size if signal.side == "buy" else -signal.size

    upper = max_abs_size - position.size
    lower = -max_abs_size - position.size
    clipped_signed = min(max(signed_target, lower), upper)
    return abs(clipped_signed)


def execute_order(
    signal: Signal,
    bar: Bar,
    portfolio: Portfolio,
    fee_model: FeeModel,
    impact_factor: float,
    max_position_pct: float,
) -> Fill | None:
    size = _clip_signal_size(signal, bar, portfolio, max_position_pct)
    if size <= 0:
        return None

    side_mult = 1.0 if signal.side == "buy" else -1.0
    signed_qty = side_mult * size
    fill_price = calculate_fill_price(Signal(signal.symbol, signal.side, size), bar, impact_factor)
    notional = size * fill_price
    fee = notional * fee_model.taker_rate

    cash_delta = -notional if signal.side == "buy" else notional
    portfolio.cash += cash_delta - fee

    position = portfolio.get_position(signal.symbol)
    prev_size = position.size
    prev_entry = position.entry_price

    realized_pnl = 0.0
    is_close = False
    entry_value = 0.0

    if prev_size != 0 and (prev_size > 0 > signed_qty or prev_size < 0 < signed_qty):
        closing_qty = min(abs(prev_size), abs(signed_qty))
        is_close = closing_qty > 0
        entry_value = closing_qty * prev_entry
        if prev_size > 0:
            realized_pnl += (fill_price - prev_entry) * closing_qty
        else:
            realized_pnl += (prev_entry - fill_price) * closing_qty

    new_size = prev_size + signed_qty

    if new_size == 0:
        position.size = 0.0
        position.entry_price = 0.0
        position.entry_value = 0.0
    elif prev_size == 0 or (prev_size > 0 and new_size < 0) or (prev_size < 0 and new_size > 0):
        position.size = new_size
        position.entry_price = fill_price
        position.entry_value = abs(new_size) * fill_price
    elif (prev_size > 0 and signed_qty > 0) or (prev_size < 0 and signed_qty < 0):
        total_abs = abs(prev_size) + abs(signed_qty)
        position.entry_price = ((abs(prev_size) * prev_entry) + (abs(signed_qty) * fill_price)) / total_abs
        position.size = new_size
        position.entry_value = abs(new_size) * position.entry_price
    else:
        position.size = new_size
        position.entry_value = abs(new_size) * position.entry_price

    return Fill(
        symbol=signal.symbol,
        side=signal.side,
        size=size,
        price=fill_price,
        fee=fee,
        timestamp=bar.timestamp,
        pnl=realized_pnl,
        is_close=is_close,
        entry_value=entry_value,
    )


def run_backtest(
    strategy: StrategyLike,
    bars: list[Bar],
    config: BacktestConfig,
    train_data: list[Bar] | None = None,
) -> BacktestResult:
    portfolio = Portfolio(cash=config.initial_cash, total_equity=config.initial_cash)
    fills: list[Fill] = []
    pending_signals: list[Signal] = []

    strategy.initialize(train_data or [])

    for bar in bars:
        for signal in pending_signals:
            fill = execute_order(
                signal=signal,
                bar=bar,
                portfolio=portfolio,
                fee_model=config.fee_model,
                impact_factor=config.slippage_impact_factor,
                max_position_pct=config.max_position_pct,
            )
            if fill is not None:
                fills.append(fill)
                strategy.on_fill(fill)
        pending_signals = []

        new_signals = strategy.on_bar(bar, portfolio)
        pending_signals = list(new_signals)

        portfolio.update_equity(bar.close)

    days_elapsed = 0.0
    if len(bars) >= 2:
        days_elapsed = (bars[-1].timestamp - bars[0].timestamp).total_seconds() / 86400

    return BacktestResult(
        fills=fills,
        portfolio=portfolio,
        bars_processed=len(bars),
        days_elapsed=days_elapsed,
    )
