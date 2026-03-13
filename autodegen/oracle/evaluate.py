from __future__ import annotations

import ast
import csv
import hashlib
import inspect
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean

from autodegen.oracle.backtest import BacktestConfig, BacktestResult, Bar, run_backtest
from autodegen.sandbox.strategy import Strategy


@dataclass(slots=True)
class Metrics:
    sharpe: float
    max_drawdown: float
    calmar: float
    win_rate: float
    trade_count: int
    total_return: float


@dataclass(slots=True)
class WalkForwardResult:
    fold_results: list[Metrics]
    avg_sharpe: float
    avg_maxdd: float
    avg_calmar: float
    avg_trade_count: float
    worst_fold_sharpe: float


def split_data_three_way(
    bars: list[Bar],
    validation_pct: float = 0.15,
    test_pct: float = 0.15,
) -> tuple[list[Bar], list[Bar], list[Bar]]:
    n = len(bars)
    test_start = int(n * (1 - test_pct))
    val_start = int(n * (1 - validation_pct - test_pct))
    return bars[:val_start], bars[val_start:test_start], bars[test_start:]


def compute_metrics(result: BacktestResult) -> Metrics:
    close_fills = [f for f in result.fills if f.is_close]
    trade_count = len(close_fills)

    sharpe = 0.0
    win_rate = 0.0

    if trade_count > 0:
        wins = sum(1 for f in close_fills if f.pnl > 0)
        win_rate = wins / trade_count

    if trade_count >= 2:
        returns = [f.pnl / f.entry_value for f in close_fills if f.entry_value > 0]
        if len(returns) >= 2:
            mean_r = fmean(returns)
            variance = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
            std_r = variance**0.5
            if std_r >= 1e-8:
                days = max(result.days_elapsed, 1e-12)
                years = days / 365.0
                trades_per_year = len(returns) / years if years > 0 else 0.0
                if trades_per_year > 0:
                    sharpe = (mean_r / std_r) * (trades_per_year**0.5)

    equity = result.portfolio.equity_curve
    if not equity:
        return Metrics(
            sharpe=sharpe,
            max_drawdown=0.0,
            calmar=0.0,
            win_rate=win_rate,
            trade_count=trade_count,
            total_return=0.0,
        )

    peak = equity[0]
    max_drawdown = 0.0
    for e in equity:
        if e > peak:
            peak = e
        if peak > 0:
            dd = (peak - e) / peak
            if dd > max_drawdown:
                max_drawdown = dd

    first = equity[0]
    last = equity[-1]
    total_return = (last / first) - 1.0 if first != 0 else 0.0

    annualized_return = 0.0
    if result.days_elapsed > 0:
        annualized_return = (1 + total_return) ** (365.0 / result.days_elapsed) - 1

    calmar = annualized_return / max_drawdown if max_drawdown > 0 else 0.0

    return Metrics(
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        calmar=calmar,
        win_rate=win_rate,
        trade_count=trade_count,
        total_return=total_return,
    )


def walk_forward_eval(
    strategy_cls: type[Strategy],
    bars: list[Bar],
    n_folds: int = 8,
    backtest_config: BacktestConfig | None = None,
) -> WalkForwardResult:
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    if len(bars) < n_folds:
        raise ValueError("bars length must be >= n_folds")

    config = backtest_config or BacktestConfig()
    fold_size = len(bars) // n_folds
    results: list[Metrics] = []

    for i in range(1, n_folds):
        train_data = bars[: i * fold_size]
        test_data = bars[i * fold_size : (i + 1) * fold_size]
        if not test_data:
            continue
        strategy = strategy_cls()
        bt = run_backtest(strategy, test_data, config, train_data=train_data)
        results.append(compute_metrics(bt))

    if not results:
        return WalkForwardResult([], 0.0, 0.0, 0.0, 0.0, 0.0)

    return WalkForwardResult(
        fold_results=results,
        avg_sharpe=fmean(r.sharpe for r in results),
        avg_maxdd=fmean(r.max_drawdown for r in results),
        avg_calmar=fmean(r.calmar for r in results),
        avg_trade_count=fmean(r.trade_count for r in results),
        worst_fold_sharpe=min(r.sharpe for r in results),
    )


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def count_hardcoded_numeric_literals(strategy: Strategy) -> int:
    source = textwrap.dedent(inspect.getsource(type(strategy)))
    tree = ast.parse(source)

    excluded_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if not any(isinstance(t, ast.Attribute) and t.attr == "parameters" for t in node.targets):
                continue
            if isinstance(node.value, ast.Dict):
                for child in ast.walk(node):
                    lineno = getattr(child, "lineno", None)
                    if lineno is not None:
                        excluded_lines.add(lineno)

    count = 0
    for node in ast.walk(tree):
        val: int | float | None = None
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            val = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            if isinstance(node.operand.value, (int, float)):
                val = -node.operand.value

        if val is None:
            continue

        lineno = getattr(node, "lineno", None)
        if lineno in excluded_lines:
            continue
        if val in (0, 1, -1, 2):
            continue
        count += 1

    return count


def composite_score(metrics: WalkForwardResult, strategy: Strategy) -> float:
    sharpe_norm = _clamp(metrics.avg_sharpe, 0.0, 3.0) / 3.0
    dd_score = max(0.0, 1 - metrics.avg_maxdd / 0.30)
    calmar_norm = _clamp(metrics.avg_calmar, 0.0, 2.0) / 2.0

    complexity = len(strategy.parameters) + count_hardcoded_numeric_literals(strategy)
    simplicity_bonus = max(0.0, 1 - complexity / 10.0)

    score = (0.4 * sharpe_norm) + (0.3 * dd_score) + (0.2 * calmar_norm) + (0.1 * simplicity_bonus)
    return _clamp(score, 0.0, 1.0)


def passes_hard_gates(metrics: WalkForwardResult) -> tuple[bool, str]:
    if metrics.avg_sharpe < 1.0:
        return False, f"avg_sharpe {metrics.avg_sharpe:.2f} < 1.0"
    if metrics.avg_maxdd > 0.30:
        return False, f"avg_maxdd {metrics.avg_maxdd:.2%} > 30%"
    if metrics.avg_trade_count < 50:
        return False, f"avg_trade_count {metrics.avg_trade_count:.1f} < 50"
    if metrics.worst_fold_sharpe < 0:
        return False, f"worst_fold_sharpe {metrics.worst_fold_sharpe:.2f} < 0"
    return True, "ok"


def compute_oracle_hash(oracle_dir: Path) -> str:
    hasher = hashlib.sha256()
    for path in sorted(oracle_dir.rglob("*.py")):
        rel = path.relative_to(oracle_dir).as_posix().encode()
        hasher.update(rel)
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def compute_data_snapshot_id(data_dir: Path) -> str:
    hasher = hashlib.sha256()
    for path in sorted(data_dir.rglob("*.parquet")):
        rel = path.relative_to(data_dir).as_posix().encode()
        hasher.update(rel)
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def classify_regime(bars: list[Bar], window: int = 90) -> str:
    if len(bars) < 2:
        return "crab"
    segment = bars[-window:] if window > 0 else bars
    if len(segment) < 2:
        return "crab"
    first = segment[0].close
    last = segment[-1].close
    if first <= 0:
        return "crab"
    ret = (last / first) - 1.0
    if ret > 0.15:
        return "bull"
    if ret < -0.15:
        return "bear"
    return "crab"


def write_result_row(
    results_path: Path,
    timestamp: datetime,
    hypothesis: str,
    composite: float,
    metrics: Metrics,
    kept: bool,
    strategy_hash: str,
    oracle_hash: str,
    data_snapshot_id: str,
    config_hash: str,
    status: str = "ok",
    error: str = "",
) -> None:
    header = [
        "timestamp",
        "hypothesis",
        "composite",
        "sharpe",
        "max_drawdown",
        "calmar",
        "win_rate",
        "trade_count",
        "total_return",
        "kept",
        "strategy_hash",
        "oracle_hash",
        "data_snapshot_id",
        "config_hash",
        "status",
        "error",
    ]

    row = [
        timestamp.isoformat(),
        hypothesis,
        f"{composite:.6f}",
        f"{metrics.sharpe:.6f}",
        f"{metrics.max_drawdown:.6f}",
        f"{metrics.calmar:.6f}",
        f"{metrics.win_rate:.6f}",
        str(metrics.trade_count),
        f"{metrics.total_return:.6f}",
        "1" if kept else "0",
        strategy_hash,
        oracle_hash,
        data_snapshot_id,
        config_hash,
        status,
        error,
    ]

    results_path.parent.mkdir(parents=True, exist_ok=True)
    exists = results_path.exists()
    with results_path.open("a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        if not exists:
            writer.writerow(header)
        writer.writerow(row)
