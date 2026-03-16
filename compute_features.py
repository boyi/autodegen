"""Self-contained feature computation module.

Computes all 10 OHLCV-derivable features used by the strategy, verified to
produce bit-identical values to the btc-prediction-py pipeline (see
validate_features.py for proof).

This module eliminates the dependency on the external pipeline for feature
computation. For live deployment, feed it spot OHLCV bars incrementally.

Usage:
    from compute_features import compute_all_features

    # From a pandas DataFrame with columns: close, open, volume
    features = compute_all_features(df)
    # Returns dict of {feature_name: pd.Series}

Validated against: btc-prediction-py factors_v3_24h.parquet
  - 9/10 features: correlation = 1.000000, MAE = 0.000000
  - positive_bar_ratio_72h: corr = 0.99999, MAE = 5.7e-6 (float rounding)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ── Transform primitives ─────────────────────────────────────────────────────
# Exact copies of btc-prediction-py/src/btc_pred/factors/transforms.py

def _rolling_return(close: pd.Series, lookback: int) -> pd.Series:
    return close / close.shift(lookback) - 1


def _realized_volatility(close: pd.Series, window: int) -> pd.Series:
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window, min_periods=max(1, window // 2)).std() * np.sqrt(window)


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(1, window // 2)).mean()
    std = series.rolling(window, min_periods=max(1, window // 2)).std()
    return (series - mean) / std.replace(0, np.nan)


def _percentile_rank(series: pd.Series, window: int) -> pd.Series:
    def _rank(x: np.ndarray) -> float:
        if len(x) < 2 or np.isnan(x[-1]):
            return np.nan
        valid = x[~np.isnan(x)]
        if len(valid) < 2:
            return np.nan
        return (valid < x[-1]).sum() / (len(valid) - 1)
    return series.rolling(window, min_periods=max(1, window // 2)).apply(_rank, raw=True)


def _slope(series: pd.Series, window: int) -> pd.Series:
    def _slope_fn(y: np.ndarray) -> float:
        if np.isnan(y).any():
            return np.nan
        x = np.arange(len(y), dtype=float)
        x -= x.mean()
        y_dm = y - y.mean()
        denom = (x * x).sum()
        if denom == 0:
            return 0.0
        return (x * y_dm).sum() / denom
    return series.rolling(window, min_periods=max(1, window // 2)).apply(_slope_fn, raw=True)


# ── Feature computation functions ────────────────────────────────────────────
# Each function matches the exact computation in btc-prediction-py.
# Parameters are hardcoded to match the YAML config values.

def vol_zscore_24h(close: pd.Series) -> pd.Series:
    """VolZScore: z-score of 24h realized vol over 720h window.
    Config: volatility.yaml → vol_window=24, zscore_window=720"""
    vol = _realized_volatility(close, 24)
    return _rolling_zscore(vol, 720)


def dist_from_low_360(close: pd.Series) -> pd.Series:
    """DistFromLow: (close - rolling_low_360) / rolling_low_360.
    Config: momentum_v2.yaml → window=360"""
    rolling_low = close.rolling(360, min_periods=180).min()
    return (close - rolling_low) / rolling_low


def vol_pctrank_720(close: pd.Series) -> pd.Series:
    """VolPctRank: percentile rank of 24h realized vol over 720h window.
    Config: volatility_v2.yaml → vol_window=24, rank_window=720"""
    vol = _realized_volatility(close, 24)
    return _percentile_rank(vol, 720)


def momentum_reversal_24h(close: pd.Series) -> pd.Series:
    """MomentumReversal: reversal after extreme 24h momentum.
    Config: momentum_v2.yaml → lookback=24, rank_window=720"""
    ret = _rolling_return(close, 24)
    prev_ret = ret.shift(24)
    extreme = prev_ret.abs() > ret.abs().rolling(720).quantile(0.9)
    return -ret * extreme.astype(float)


def sharpe_momentum_72h(close: pd.Series) -> pd.Series:
    """SharpeMomentum: 72h return / realized_vol.
    Config: directional.yaml → window=72"""
    ret = _rolling_return(close, 72)
    vol = _realized_volatility(close, 72)
    return ret / vol.replace(0, np.nan)


def net_volume_ratio_72h(close: pd.Series, open_: pd.Series, volume: pd.Series) -> pd.Series:
    """NetVolumeRatio: (up_vol - down_vol) / total_vol over 72h.
    Config: directional.yaml → window=72"""
    up = (volume * (close > open_).astype(float)).rolling(72).sum()
    dn = (volume * (close <= open_).astype(float)).rolling(72).sum()
    total = (up + dn).replace(0, np.nan)
    return (up - dn) / total


def price_efficiency_72h(close: pd.Series) -> pd.Series:
    """PriceEfficiency: net log move / total path over 72h.
    Config: momentum_v2.yaml → window=72"""
    log_ret = np.log(close / close.shift(1))
    net = log_ret.rolling(72).sum()
    total = log_ret.abs().rolling(72).sum().replace(0, np.nan)
    return net / total


def trend_slope_3d(close: pd.Series) -> pd.Series:
    """TrendSlope: linear regression slope of log(close) over 72h.
    Config: momentum.yaml → window=72"""
    return _slope(np.log(close), 72)


def trend_consistency_3d(close: pd.Series) -> pd.Series:
    """TrendConsistency: mean sign of 3 consecutive 24h segment returns.
    Config: directional.yaml → seg_len=24, n_segs=3"""
    segs = [np.sign(_rolling_return(close, 24).shift(i * 24)) for i in range(3)]
    return pd.concat(segs, axis=1).mean(axis=1)


def positive_bar_ratio_72h(close: pd.Series) -> pd.Series:
    """PositiveBarRatio: fraction of positive-return bars - 0.5 over 72h.
    Config: directional.yaml → window=72"""
    log_ret = np.log(close / close.shift(1))
    return (log_ret > 0).astype(float).rolling(72, min_periods=36).mean() - 0.5


# ── Enrichment-derived features (require non-OHLCV data) ────────────────────

def funding_cumsum_3d(funding_rate: pd.Series) -> pd.Series:
    """72h rolling sum of funding rate.
    Requires: binance_funding__funding_rate from enrichment data."""
    return funding_rate.rolling(72, min_periods=1).sum()


def oi_change_24h(open_interest: pd.Series) -> pd.Series:
    """24h change in open interest: (oi / oi.shift(24)) - 1.
    Requires: bybit_oi__open_interest from enrichment data."""
    return open_interest / open_interest.shift(24) - 1


# ── Public API ───────────────────────────────────────────────────────────────

OHLCV_FEATURES = {
    "vol_zscore_24h": vol_zscore_24h,
    "dist_from_low_360": dist_from_low_360,
    "vol_pctrank_720": vol_pctrank_720,
    "momentum_reversal_24h": momentum_reversal_24h,
    "sharpe_momentum_72h": sharpe_momentum_72h,
    "net_volume_ratio_72h": net_volume_ratio_72h,
    "price_efficiency_72h": price_efficiency_72h,
    "trend_slope_3d": trend_slope_3d,
    "trend_consistency_3d": trend_consistency_3d,
    "positive_bar_ratio_72h": positive_bar_ratio_72h,
}

# Minimum bars needed before features produce valid values
WARMUP_BARS = 744  # 720 (zscore window) + 24 (vol window) = 31 days


def compute_all_features(
    close: pd.Series,
    open_: pd.Series | None = None,
    volume: pd.Series | None = None,
    funding_rate: pd.Series | None = None,
    open_interest: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """Compute all strategy features from raw price data.

    Args:
        close: Spot close prices (hourly, DatetimeIndex).
        open_: Spot open prices. Required for net_volume_ratio_72h.
        volume: Spot volume. Required for net_volume_ratio_72h.
        funding_rate: Funding rate series (optional, for funding_cumsum_3d).
        open_interest: OI series (optional, for oi_change_24h).

    Returns:
        Dict mapping feature name to computed Series.
    """
    result: dict[str, pd.Series] = {}

    # Close-only features
    for name in ["vol_zscore_24h", "dist_from_low_360", "vol_pctrank_720",
                 "momentum_reversal_24h", "sharpe_momentum_72h",
                 "price_efficiency_72h", "trend_slope_3d",
                 "trend_consistency_3d", "positive_bar_ratio_72h"]:
        result[name] = OHLCV_FEATURES[name](close)

    # Volume feature
    if open_ is not None and volume is not None:
        result["net_volume_ratio_72h"] = net_volume_ratio_72h(close, open_, volume)

    # Enrichment features
    if funding_rate is not None:
        result["funding_cumsum_3d"] = funding_cumsum_3d(funding_rate)
    if open_interest is not None:
        result["oi_change_24h"] = oi_change_24h(open_interest)

    return result
