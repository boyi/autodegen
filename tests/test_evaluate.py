from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from autodegen.oracle.backtest import BacktestConfig, BacktestResult, Bar, FeeModel, Fill, Portfolio, Signal
from autodegen.oracle.evaluate import (
    Metrics,
    WalkForwardResult,
    classify_regime,
    composite_score,
    compute_data_snapshot_id,
    compute_metrics,
    compute_oracle_hash,
    count_hardcoded_numeric_literals,
    passes_hard_gates,
    split_data_three_way,
    walk_forward_eval,
    write_result_row,
)
from autodegen.sandbox.strategy import Strategy


def _bars(n: int, start: float = 100.0, step: float = 1.0) -> list[Bar]:
    out = []
    t0 = datetime(2025, 1, 1, tzinfo=UTC)
    for i in range(n):
        px = start + (i * step)
        out.append(Bar(timestamp=t0 + timedelta(hours=i), open=px, high=px, low=px, close=px, volume=1000.0))
    return out


def _result_with(
    fills: list[Fill],
    equity: list[float],
    days_elapsed: float = 30.0,
) -> BacktestResult:
    p = Portfolio(cash=equity[-1] if equity else 0.0, total_equity=equity[-1] if equity else 0.0)
    p.equity_curve = equity
    return BacktestResult(fills=fills, portfolio=p, bars_processed=len(equity), days_elapsed=days_elapsed)


# --- Three-way split tests ---

def test_split_data_three_way_percentages_70_15_15() -> None:
    bars = _bars(100)
    wf, val, test = split_data_three_way(bars)
    assert len(wf) == 70
    assert len(val) == 15
    assert len(test) == 15


def test_split_data_three_way_no_overlap() -> None:
    bars = _bars(100)
    wf, val, test = split_data_three_way(bars)
    ids_wf = {id(x) for x in wf}
    ids_val = {id(x) for x in val}
    ids_test = {id(x) for x in test}
    assert ids_wf.isdisjoint(ids_val)
    assert ids_wf.isdisjoint(ids_test)
    assert ids_val.isdisjoint(ids_test)


def test_split_data_three_way_covers_all_rows() -> None:
    bars = _bars(137)
    wf, val, test = split_data_three_way(bars)
    assert len(wf) + len(val) + len(test) == 137


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7])
def test_split_data_three_way_small_dataset_edge_cases(n: int) -> None:
    bars = _bars(n)
    wf, val, test = split_data_three_way(bars)
    assert len(wf) + len(val) + len(test) == n


# --- Walk-forward tests ---

class TrackingStrategy(Strategy):
    train_lengths: list[int] = []
    test_start_end: list[tuple[datetime, datetime]] = []

    def __init__(self) -> None:
        super().__init__()
        self.name = "tracking"
        self.parameters = {"x": 1}
        self._seen: list[datetime] = []

    def initialize(self, train_data: list[Bar]) -> None:
        type(self).train_lengths.append(len(train_data))

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Signal]:
        self._seen.append(bar.timestamp)
        return []

    def on_fill(self, fill: Fill) -> None:
        return None


class BuyAndCloseEveryBar(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.name = "buy_close"
        self.parameters = {}
        self._has_pos = False

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Signal]:
        if self._has_pos:
            self._has_pos = False
            return [Signal(symbol="BTC/USDT", side="sell", size=1.0)]
        self._has_pos = True
        return [Signal(symbol="BTC/USDT", side="buy", size=1.0)]



def test_walk_forward_expanding_window_train_lengths() -> None:
    TrackingStrategy.train_lengths = []
    bars = _bars(80)
    walk_forward_eval(TrackingStrategy, bars, n_folds=8, backtest_config=BacktestConfig(slippage_impact_factor=0.0))
    assert TrackingStrategy.train_lengths == [10, 20, 30, 40, 50, 60, 70]


def test_walk_forward_produces_n_folds_minus_one_results() -> None:
    bars = _bars(80)
    res = walk_forward_eval(TrackingStrategy, bars, n_folds=8, backtest_config=BacktestConfig(slippage_impact_factor=0.0))
    assert len(res.fold_results) == 7


def test_walk_forward_no_lookahead_train_before_test() -> None:
    TrackingStrategy.train_lengths = []
    bars = _bars(80)
    fold_size = len(bars) // 8
    walk_forward_eval(TrackingStrategy, bars, n_folds=8, backtest_config=BacktestConfig(slippage_impact_factor=0.0))
    for i, train_len in enumerate(TrackingStrategy.train_lengths, start=1):
        assert train_len == i * fold_size


def test_walk_forward_with_known_strategy_runs() -> None:
    bars = _bars(96, start=100, step=0.5)
    res = walk_forward_eval(
        BuyAndCloseEveryBar,
        bars,
        n_folds=8,
        backtest_config=BacktestConfig(slippage_impact_factor=0.0, fee_model=FeeModel(0, 0)),
    )
    assert len(res.fold_results) == 7
    assert all(m.trade_count >= 1 for m in res.fold_results)


def test_walk_forward_invalid_n_folds_raises() -> None:
    with pytest.raises(ValueError):
        walk_forward_eval(TrackingStrategy, _bars(10), n_folds=1)


def test_walk_forward_bars_less_than_folds_raises() -> None:
    with pytest.raises(ValueError):
        walk_forward_eval(TrackingStrategy, _bars(5), n_folds=8)


# --- Metrics tests ---

def test_compute_metrics_sharpe_known_trade_returns() -> None:
    fills = [
        Fill("BTC/USDT", "sell", 1, 0, 0, datetime.now(UTC), pnl=10, is_close=True, entry_value=100),
        Fill("BTC/USDT", "sell", 1, 0, 0, datetime.now(UTC), pnl=20, is_close=True, entry_value=100),
        Fill("BTC/USDT", "sell", 1, 0, 0, datetime.now(UTC), pnl=0, is_close=True, entry_value=100),
    ]
    m = compute_metrics(_result_with(fills, [1000, 1010, 1020], days_elapsed=365.0))
    assert m.sharpe == pytest.approx(1.7320508, rel=1e-2)


def test_compute_metrics_sharpe_zero_when_std_near_zero() -> None:
    fills = [
        Fill("BTC/USDT", "sell", 1, 0, 0, datetime.now(UTC), pnl=10, is_close=True, entry_value=100),
        Fill("BTC/USDT", "sell", 1, 0, 0, datetime.now(UTC), pnl=10, is_close=True, entry_value=100),
    ]
    m = compute_metrics(_result_with(fills, [1000, 1001], days_elapsed=30.0))
    assert m.sharpe == 0.0


def test_compute_metrics_sharpe_zero_when_no_closing_trades() -> None:
    fills = [Fill("BTC/USDT", "buy", 1, 0, 0, datetime.now(UTC), pnl=0, is_close=False, entry_value=0)]
    m = compute_metrics(_result_with(fills, [1000, 1005], days_elapsed=30.0))
    assert m.sharpe == 0.0
    assert m.trade_count == 0


def test_compute_metrics_max_drawdown_known_curve() -> None:
    m = compute_metrics(_result_with([], [100, 120, 90, 110], days_elapsed=30.0))
    assert m.max_drawdown == pytest.approx(0.25)


def test_compute_metrics_calmar_computation() -> None:
    m = compute_metrics(_result_with([], [100, 150, 120], days_elapsed=365.0))
    assert m.calmar == pytest.approx(1.0)


def test_compute_metrics_win_rate_computation() -> None:
    fills = [
        Fill("BTC/USDT", "sell", 1, 0, 0, datetime.now(UTC), pnl=10, is_close=True, entry_value=100),
        Fill("BTC/USDT", "sell", 1, 0, 0, datetime.now(UTC), pnl=-5, is_close=True, entry_value=100),
        Fill("BTC/USDT", "sell", 1, 0, 0, datetime.now(UTC), pnl=0, is_close=True, entry_value=100),
    ]
    m = compute_metrics(_result_with(fills, [1000, 1002], days_elapsed=10.0))
    assert m.win_rate == pytest.approx(1 / 3)


def test_compute_metrics_empty_equity_curve_safe_defaults() -> None:
    m = compute_metrics(_result_with([], [], days_elapsed=0.0))
    assert m.total_return == 0.0
    assert m.max_drawdown == 0.0


# --- Composite score tests ---

class SimpleStrategy(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.parameters = {"a": 1}

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Signal]:
        return []


class ComplexStrategy(Strategy):
    def __init__(self) -> None:
        super().__init__()
        self.parameters = {f"p{i}": i for i in range(8)}

    def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Signal]:
        x = 3.14 + 9 + 21 + 42
        if bar.close > x:
            return [Signal("BTC/USDT", "buy", 1.0)]
        return []



def _wf(avg_sharpe: float, avg_maxdd: float, avg_calmar: float, avg_trade_count: float = 60, worst: float = 0.5) -> WalkForwardResult:
    return WalkForwardResult([], avg_sharpe, avg_maxdd, avg_calmar, avg_trade_count, worst)


def test_composite_score_in_0_1_range() -> None:
    s = composite_score(_wf(99, -1, 99), SimpleStrategy())
    assert 0.0 <= s <= 1.0


def test_composite_score_perfect_strategy_scores_high() -> None:
    s = composite_score(_wf(3.0, 0.01, 2.0), SimpleStrategy())
    assert s > 0.9


def test_composite_score_terrible_strategy_scores_low() -> None:
    s = composite_score(_wf(-2.0, 0.5, -1.0), ComplexStrategy())
    assert s < 0.2


def test_composite_score_simplicity_bonus_fewer_params_higher() -> None:
    wf = _wf(1.5, 0.1, 1.0)
    assert composite_score(wf, SimpleStrategy()) > composite_score(wf, ComplexStrategy())


def test_count_hardcoded_numeric_literals_penalizes_constants() -> None:
    assert count_hardcoded_numeric_literals(ComplexStrategy()) >= 4


def test_count_hardcoded_numeric_literals_excludes_common_idioms() -> None:
    class IdiomStrategy(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.parameters = {"a": 2, "b": 1, "c": -1, "d": 0}

        def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Signal]:
            x = 0 + 1 + 2 - 1
            if x > 0:
                return []
            return []

    assert count_hardcoded_numeric_literals(IdiomStrategy()) == 0


def test_count_hardcoded_numeric_literals_excludes_parameters_dict() -> None:
    class ParamHeavy(Strategy):
        def __init__(self) -> None:
            super().__init__()
            self.parameters = {"fast": 12, "slow": 26, "th": 0.7}

        def on_bar(self, bar: Bar, portfolio: Portfolio) -> list[Signal]:
            return []

    assert count_hardcoded_numeric_literals(ParamHeavy()) == 0


# --- Hard gates tests ---

def test_hard_gates_pass_with_good_metrics() -> None:
    ok, reason = passes_hard_gates(_wf(1.2, 0.2, 1.1, avg_trade_count=80, worst=0.1))
    assert ok is True
    assert reason == "ok"


def test_hard_gates_fail_low_sharpe() -> None:
    ok, reason = passes_hard_gates(_wf(0.9, 0.2, 1.1, avg_trade_count=80, worst=0.1))
    assert ok is False
    assert "avg_sharpe" in reason


def test_hard_gates_fail_high_drawdown() -> None:
    ok, reason = passes_hard_gates(_wf(1.2, 0.31, 1.1, avg_trade_count=80, worst=0.1))
    assert ok is False
    assert "avg_maxdd" in reason


def test_hard_gates_fail_low_trade_count() -> None:
    ok, reason = passes_hard_gates(_wf(1.2, 0.2, 1.1, avg_trade_count=49, worst=0.1))
    assert ok is False
    assert "avg_trade_count" in reason


def test_hard_gates_fail_negative_worst_fold_sharpe() -> None:
    ok, reason = passes_hard_gates(_wf(1.2, 0.2, 1.1, avg_trade_count=80, worst=-0.01))
    assert ok is False
    assert "worst_fold_sharpe" in reason


# --- Hash tests ---

def test_oracle_hash_deterministic(tmp_path: Path) -> None:
    d = tmp_path / "oracle"
    d.mkdir()
    (d / "a.py").write_text("print('a')\n")
    (d / "b.py").write_text("print('b')\n")
    h1 = compute_oracle_hash(d)
    h2 = compute_oracle_hash(d)
    assert h1 == h2


def test_oracle_hash_changes_on_content_change(tmp_path: Path) -> None:
    d = tmp_path / "oracle"
    d.mkdir()
    f = d / "a.py"
    f.write_text("x = 1\n")
    h1 = compute_oracle_hash(d)
    f.write_text("x = 2\n")
    h2 = compute_oracle_hash(d)
    assert h1 != h2


def test_data_snapshot_hash_deterministic(tmp_path: Path) -> None:
    d = tmp_path / "data"
    d.mkdir()
    (d / "2025-01.parquet").write_bytes(b"abc")
    h1 = compute_data_snapshot_id(d)
    h2 = compute_data_snapshot_id(d)
    assert h1 == h2


def test_data_snapshot_hash_changes_on_content_change(tmp_path: Path) -> None:
    d = tmp_path / "data"
    d.mkdir()
    f = d / "2025-01.parquet"
    f.write_bytes(b"abc")
    h1 = compute_data_snapshot_id(d)
    f.write_bytes(b"def")
    h2 = compute_data_snapshot_id(d)
    assert h1 != h2


# --- Results TSV tests ---

def test_write_result_row_creates_header_when_missing(tmp_path: Path) -> None:
    p = tmp_path / "results.tsv"
    write_result_row(
        p,
        datetime(2025, 1, 1, tzinfo=UTC),
        "h",
        0.5,
        Metrics(1.0, 0.1, 1.2, 0.6, 77, 0.4),
        True,
        "s",
        "o",
        "d",
        "c",
    )
    lines = p.read_text().splitlines()
    assert lines[0].startswith("timestamp\thypothesis\tcomposite")
    assert len(lines) == 2


def test_write_result_row_appends_to_existing(tmp_path: Path) -> None:
    p = tmp_path / "results.tsv"
    kwargs = dict(
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        hypothesis="h",
        composite=0.5,
        metrics=Metrics(1.0, 0.1, 1.2, 0.6, 77, 0.4),
        kept=True,
        strategy_hash="s",
        oracle_hash="o",
        data_snapshot_id="d",
        config_hash="c",
    )
    write_result_row(p, **kwargs)
    write_result_row(p, **kwargs)
    assert len(p.read_text().splitlines()) == 3


def test_write_result_row_writes_all_fields(tmp_path: Path) -> None:
    p = tmp_path / "results.tsv"
    write_result_row(
        p,
        datetime(2025, 1, 1, 12, 0, tzinfo=UTC),
        "hypo",
        0.1234567,
        Metrics(1.1111111, 0.2222222, 0.3333333, 0.4444444, 12, 0.5555555),
        False,
        "strat",
        "oracle",
        "data",
        "cfg",
        status="err",
        error="boom",
    )
    cols = p.read_text().splitlines()[1].split("\t")
    assert cols[1] == "hypo"
    assert cols[2] == "0.123457"
    assert cols[3] == "1.111111"
    assert cols[9] == "0"
    assert cols[14] == "err"
    assert cols[15] == "boom"


# --- Regime tests ---

def test_classify_regime_bull() -> None:
    bars = _bars(100, start=100, step=0.2)
    bars[-1].close = 120.0
    assert classify_regime(bars, window=90) == "bull"


def test_classify_regime_bear() -> None:
    bars = _bars(100, start=100, step=0.0)
    bars[-90].close = 100.0
    bars[-1].close = 80.0
    assert classify_regime(bars, window=90) == "bear"


def test_classify_regime_crab() -> None:
    bars = _bars(100, start=100, step=0.0)
    bars[-1].close = 105.0
    assert classify_regime(bars, window=90) == "crab"


def test_classify_regime_short_series_defaults_crab() -> None:
    assert classify_regime(_bars(1), window=90) == "crab"
