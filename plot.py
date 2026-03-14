"""
plot.py — fold-aware visual diagnostic for the current strategy.

Usage:
    uv run python plot.py                   # saves chart.png
    uv run python plot.py --out out.png     # custom output path

Three panels (shared x-axis, dark theme):
  ① BTC close price (log) + buy/sell fill markers + in-position dots
  ② Strategy equity vs Buy-and-Hold baseline
  ③ Drawdown % (underwater plot)

Fold shading on every panel:
  ░░ orange  — WF test windows: out-of-sample folds the scorer actually measures
  ░░ green   — Validation holdout: truly held-out final 15% of history
     (unshaded = WF train regions: strategy has already seen these bars)

NOTE: the equity curve shown here is a single continuous run on ALL bars.
      That is NOT how the harness scores the strategy (it uses walk-forward
      folds). The fold shading is the honest reminder: orange/green regions
      are the parts that actually matter — treat unshaded equity as in-sample.
"""
from __future__ import annotations

import argparse
import sys

import matplotlib
matplotlib.use("Agg")  # headless-safe; no display required

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from prepare import (
    BacktestResult,
    load_bars,
    run_backtest,
    summarize_result,
    walk_forward_splits,
)
from strategy import Strategy

# ── colour palette (GitHub dark) ──────────────────────────────────────────────
_BG      = "#0d1117"
_GRID    = "#21262d"
_BORDER  = "#30363d"
_TEXT    = "#c9d1d9"
_MUTED   = "#8b949e"

_C_BTC   = "#58a6ff"  # BTC price line
_C_EQ    = "#e3b341"  # strategy equity
_C_BAH   = "#8b949e"  # buy-and-hold
_C_BUY   = "#3fb950"  # buy markers
_C_SELL  = "#f85149"  # sell markers / drawdown
_C_POS   = "#388bfd"  # in-position dots on price panel
_C_TEST  = "#f0a500"  # WF test window tint
_C_VAL   = "#3fb950"  # validation window tint


# ── helpers ───────────────────────────────────────────────────────────────────

def _fold_regions(
    bars,
    validation_pct: float = 0.15,
    n_folds: int = 6,
) -> tuple[int, list[tuple[int, int]]]:
    """
    Returns:
        val_start_idx  – index in `bars` where the validation holdout begins.
        test_windows   – list of (start_idx, end_idx) for each WF test window.

    walk_forward_splits is applied to wf_bars = bars[:val_start_idx], so the
    indices it returns (len(train), len(train)+len(test)) map directly onto
    the full `bars` list with no offset needed.
    """
    cut = int(len(bars) * (1 - validation_pct))
    folds = walk_forward_splits(bars[:cut], n_folds=n_folds)
    test_windows = [(len(tr), len(tr) + len(te)) for tr, te in folds]
    return cut, test_windows


def _drawdown_pct(equity: list[float]) -> list[float]:
    peak = equity[0]
    out: list[float] = []
    for e in equity:
        peak = max(peak, e)
        out.append(((e - peak) / peak) * 100.0)
    return out


def _shade(axes, timestamps, val_start: int, test_windows: list[tuple[int, int]]) -> None:
    """Apply fold / validation shading + divider lines to every panel."""
    for ax in axes:
        # WF test windows — out-of-sample folds
        for ts_i, te_i in test_windows:
            ax.axvspan(
                timestamps[ts_i], timestamps[te_i - 1],
                alpha=0.13, color=_C_TEST, lw=0,
            )
            ax.axvline(
                timestamps[ts_i],
                color=_C_TEST, alpha=0.35, linewidth=0.7, linestyle=":",
            )
        # Validation holdout
        ax.axvspan(
            timestamps[val_start], timestamps[-1],
            alpha=0.10, color=_C_VAL, lw=0,
        )
        ax.axvline(
            timestamps[val_start],
            color=_C_VAL, alpha=0.70, linewidth=1.3, linestyle="--",
        )


def _style(ax) -> None:
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_MUTED, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(_BORDER)
    ax.grid(True, color=_GRID, linewidth=0.4, linestyle="--")
    ax.yaxis.label.set_color(_MUTED)
    ax.xaxis.label.set_color(_MUTED)


# ── main chart builder ────────────────────────────────────────────────────────

def build_chart(out_path: str = "chart.png") -> None:
    # ── load data + run backtest ───────────────────────────────────────────────
    bars = load_bars()
    if not bars:
        print("No data found. Run:  uv run python prepare.py fetch")
        sys.exit(1)

    result: BacktestResult = run_backtest(Strategy(), bars)

    equity    = result.equity_curve
    positions = result.position_history
    fills     = result.fills
    summary   = summarize_result(result)

    timestamps = [b.timestamp for b in bars]
    closes     = [b.close for b in bars]
    bah        = [10_000.0 * (c / closes[0]) for c in closes]
    dd_series  = _drawdown_pct(equity)

    val_start, test_windows = _fold_regions(bars)

    # separate buy / sell fills for scatter markers
    buy_ts   = [f["timestamp"] for f in fills if f["side"] == "buy"]
    buy_px   = [f["price"]     for f in fills if f["side"] == "buy"]
    sell_ts  = [f["timestamp"] for f in fills if f["side"] == "sell"]
    sell_px  = [f["price"]     for f in fills if f["side"] == "sell"]

    # in-position dots: small dots on BTC price where strategy holds a position
    EPS = 1e-12
    pos_ts = [timestamps[i] for i, p in enumerate(positions) if abs(p) > EPS]
    pos_cl = [closes[i]     for i, p in enumerate(positions) if abs(p) > EPS]

    # ── figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 13), facecolor=_BG)
    gs  = GridSpec(3, 1, figure=fig, hspace=0.06, height_ratios=[3, 3, 2])

    ax_price  = fig.add_subplot(gs[0])
    ax_equity = fig.add_subplot(gs[1], sharex=ax_price)
    ax_dd     = fig.add_subplot(gs[2], sharex=ax_price)

    for ax in (ax_price, ax_equity, ax_dd):
        _style(ax)

    _shade([ax_price, ax_equity, ax_dd], timestamps, val_start, test_windows)

    # ── panel ① : BTC price ───────────────────────────────────────────────────
    ax_price.plot(timestamps, closes, color=_C_BTC, linewidth=0.8, zorder=2,
                  label="BTC close")
    ax_price.set_yscale("log")
    ax_price.set_ylabel("BTC Price (USDT)")

    # subtle dots where strategy holds a position (shows exposure visually)
    if pos_ts:
        ax_price.scatter(pos_ts, pos_cl, s=1.2, color=_C_POS,
                         alpha=0.20, zorder=1, linewidths=0)

    # trade fills
    if buy_ts:
        ax_price.scatter(buy_ts, buy_px, marker="^", color=_C_BUY,
                         s=25, zorder=5, alpha=0.85, linewidths=0, label="Buy fill")
    if sell_ts:
        ax_price.scatter(sell_ts, sell_px, marker="v", color=_C_SELL,
                         s=25, zorder=5, alpha=0.85, linewidths=0, label="Sell fill")

    plt.setp(ax_price.get_xticklabels(), visible=False)

    # ── panel ② : equity + buy & hold ────────────────────────────────────────
    ax_equity.plot(timestamps, equity, color=_C_EQ, linewidth=1.2, zorder=3,
                   label="Strategy equity")
    ax_equity.plot(timestamps, bah, color=_C_BAH, linewidth=0.7,
                   linestyle="--", alpha=0.55, zorder=2, label="Buy & Hold")
    ax_equity.set_yscale("log")
    ax_equity.set_ylabel("Portfolio Value (USDT, log)")

    plt.setp(ax_equity.get_xticklabels(), visible=False)

    # ── panel ③ : drawdown ────────────────────────────────────────────────────
    ax_dd.fill_between(timestamps, dd_series, 0,
                       color=_C_SELL, alpha=0.38, zorder=2)
    ax_dd.plot(timestamps, dd_series, color=_C_SELL, linewidth=0.6, zorder=3)
    ax_dd.axhline(0, color=_BORDER, linewidth=0.5)
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.set_xlabel("Date")

    fmt = mdates.DateFormatter("%b %Y")
    ax_dd.xaxis.set_major_formatter(fmt)
    ax_dd.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax_dd.get_xticklabels(), rotation=35, ha="right", fontsize=7,
             color=_MUTED)

    # ── title ─────────────────────────────────────────────────────────────────
    name   = getattr(Strategy, "name", "strategy")
    params = "  ".join(f"{k}={v}" for k, v in Strategy.parameters.items())
    n_folds_actual = len(test_windows)
    val_days = (len(bars) - val_start) / 24

    title_line = (
        f"Strategy: {name}   ║   "
        f"Sharpe (full-run*): {summary['bar_sharpe']:.2f}   "
        f"MaxDD: {summary['maxdd']*100:.1f}%   "
        f"Trades: {int(summary['trades'])}   "
        f"Final equity: ${equity[-1]:,.0f}   "
        f"WF folds: {n_folds_actual}   "
        f"Val holdout: {val_days:.0f}d\n"
        f"Params → {params}\n"
        f"* full-run Sharpe ≠ harness score (see orange/green regions for what actually counts)"
    )
    fig.suptitle(title_line, color=_TEXT, fontsize=8.5, y=0.998,
                 x=0.5, linespacing=1.6)

    # ── legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=_C_TEST, alpha=0.55,
                       label="WF test window — out-of-sample (harness scores here)"),
        mpatches.Patch(color=_C_VAL,  alpha=0.45,
                       label="Validation holdout — truly held-out 15%"),
        mpatches.Patch(color=_MUTED,  alpha=0.20,
                       label="WF train — strategy has seen these bars"),
        mpatches.Patch(color=_C_BTC,  label="BTC close"),
        mpatches.Patch(color=_C_POS,  alpha=0.55, label="In-position"),
        mpatches.Patch(color=_C_EQ,   label="Strategy equity (full-run)"),
        mpatches.Patch(color=_C_BAH,  label="Buy & Hold"),
        mpatches.Patch(color=_C_BUY,  label="Buy fill ▲"),
        mpatches.Patch(color=_C_SELL, label="Sell fill ▼"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        facecolor="#161b22",
        edgecolor=_BORDER,
        labelcolor=_TEXT,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.subplots_adjust(top=0.93, bottom=0.11)

    # ── save ──────────────────────────────────────────────────────────────────
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    plt.close()

    # print a quick text summary alongside the image path
    print(f"\n{'─'*60}")
    print(f"  chart saved → {out_path}")
    print(f"{'─'*60}")
    print(f"  strategy    : {name}")
    print(f"  bars        : {len(bars):,}  ({bars[0].timestamp:%Y-%m-%d} → {bars[-1].timestamp:%Y-%m-%d})")
    print(f"  WF folds    : {n_folds_actual}  (test windows shaded orange)")
    print(f"  val holdout : {val_days:.0f} days  (shaded green)")
    print(f"  sharpe*     : {summary['bar_sharpe']:.3f}   ← full-run, not harness score")
    print(f"  max DD      : {summary['maxdd']*100:.1f}%")
    print(f"  trades      : {int(summary['trades'])}")
    print(f"  final eq    : ${equity[-1]:,.2f}   (started $10,000)")
    print(f"  buy & hold  : ${bah[-1]:,.2f}")
    print(f"{'─'*60}\n")


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Fold-aware strategy diagnostic chart",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--out", default="chart.png",
        help="Output image path  (default: chart.png)",
    )
    args = ap.parse_args()
    build_chart(out_path=args.out)
