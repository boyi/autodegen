"""
autodegen strategy — THE ONLY FILE THE AGENT EDITS
Run: python strategy.py --source external
"""

import argparse

from prepare import evaluate, load_bars


class Strategy:
    name = "ema_20_50_hh_trendfilt_v1"
    description = (
        "EMA 20/50 + HH-only + trend_consistency filter on all entries. "
        "HH-only for decay, trend filter for fold stability."
    )
    parameters = {
        "ema_fast": 20,
        "ema_slow": 50,
        "structure_lookback": 8,
        "base_size": 0.30,
        "trail_pct": 0.019,
        "volz_scale": 3.00,
        "reentry_cooldown": 12,
        "reentry_trend_min": 0.33,
        "tp_pct": 0.04,
    }

    def initialize(self, train_data):
        self.close_history = []
        self.high_history = []
        self.low_history = []
        self.ema_fast_val = None
        self.ema_slow_val = None
        self.ema_macro_val = None
        self.prev_trend_up = False
        self.highest_since_entry = None
        self.bars_since_exit = 999
        self.trend_at_exit = False
        self.entry_price = None
        self.took_profit = False

    def _ema(self, prev, price, period):
        if prev is None:
            return price
        alpha = 2.0 / (period + 1)
        return alpha * price + (1.0 - alpha) * prev

    def on_bar(self, bar, portfolio):
        self.close_history.append(bar.close)
        self.high_history.append(bar.high)
        self.low_history.append(bar.low)

        self.ema_fast_val = self._ema(self.ema_fast_val, bar.close, self.parameters["ema_fast"])
        self.ema_slow_val = self._ema(self.ema_slow_val, bar.close, self.parameters["ema_slow"])
        self.ema_macro_val = self._ema(self.ema_macro_val, bar.close, 100)

        lookback = self.parameters["structure_lookback"]
        if len(self.close_history) < max(lookback * 2, self.parameters["ema_slow"]):
            return []

        current_pos = portfolio["position"]
        trend_up = self.ema_fast_val > self.ema_slow_val

        recent_high = max(self.high_history[-lookback:])
        prior_high = max(self.high_history[-lookback * 2:-lookback])
        hh = recent_high > prior_high
        recent_low = min(self.low_history[-lookback:])
        prior_low = min(self.low_history[-lookback * 2:-lookback])
        hl = recent_low > prior_low
        uptrend_structure = hh  # HH-only: more signals, better regime uniformity

        if current_pos == 0:
            self.bars_since_exit += 1

            extras = bar.extras or {}
            trend_c = extras.get("trend_consistency_3d")
            trend_ok = (trend_c is None or trend_c != trend_c
                        or trend_c > self.parameters["reentry_trend_min"])

            crossover_entry = (trend_up and not self.prev_trend_up
                               and uptrend_structure and trend_ok)

            trend_strong = (trend_c is not None and trend_c == trend_c
                            and trend_c > self.parameters["reentry_trend_min"])
            reentry = (self.trend_at_exit
                       and self.bars_since_exit >= self.parameters["reentry_cooldown"]
                       and trend_up and uptrend_structure and trend_strong)

            is_reentry = reentry and not crossover_entry
            if crossover_entry or reentry:
                size = self.parameters["base_size"]
                extras = bar.extras or {}
                volz = extras.get("vol_zscore_24h")
                if volz is not None and volz == volz:
                    scale = max(0.42, min(1.58, 1.0 - self.parameters["volz_scale"] * volz))
                    size = self.parameters["base_size"] * scale

                # Funding carry adjustment
                fc = extras.get("funding_cumsum_3d")
                if fc is not None and fc == fc:
                    size *= max(0.30, min(1.70, 1.0 - fc * 50.0))

                # OI conviction sizing
                oi = extras.get("oi_change_24h")
                if oi is not None and oi == oi:
                    size *= max(0.75, min(1.25, 1.0 + oi * 2.5))

                # Trend strength from distance
                dist = extras.get("dist_from_low_360")
                if dist is not None and dist == dist:
                    size *= max(0.70, min(1.30, 0.70 + dist * 0.60))

                # Vol percentile: size down when vol is historically extreme
                vpr = extras.get("vol_pctrank_720")
                if vpr is not None and vpr == vpr:
                    size *= max(0.10, min(1.55, 1.65 - vpr * 1.22))

                # Momentum reversal filter
                mr = extras.get("momentum_reversal_24h")
                if mr is not None and mr == mr:
                    size *= max(0.70, min(1.30, 1.0 - mr * 10.0))

                # Sharpe momentum quality: size up in clean trends
                sm = extras.get("sharpe_momentum_72h")
                if sm is not None and sm == sm:
                    size *= max(0.30, min(1.70, 0.65 + sm * 0.80))

                # Net volume ratio: size with buying pressure
                nvr = extras.get("net_volume_ratio_72h")
                if nvr is not None and nvr == nvr:
                    size *= max(0.20, min(1.80, 1.0 + nvr * 3.5))

                # Price efficiency: modest boost in clean moves
                pe = extras.get("price_efficiency_72h")
                if pe is not None and pe == pe:
                    size *= max(0.25, min(1.75, 0.25 + pe * 3.00))

                # Macro trend: reduce when slow EMA below 100 EMA
                if self.ema_slow_val < self.ema_macro_val:
                    size *= 0.96

                if is_reentry:
                    size *= 0.5
                self.highest_since_entry = bar.high
                self.entry_price = bar.close
                self.took_profit = False
                self.prev_trend_up = True
                self.trend_at_exit = False
                return [{"side": "buy", "size": size}]

        if current_pos > 0:
            self.highest_since_entry = max(self.highest_since_entry, bar.high)

            # Adaptive TP: tighter in low vol (lock gains), wider in high vol (let ride)
            extras = bar.extras or {}
            volz_tp = extras.get("vol_zscore_24h")
            tp_pct = self.parameters["tp_pct"]
            if volz_tp is not None and volz_tp == volz_tp:
                tp_pct = max(0.02, min(0.07, tp_pct + volz_tp * 0.0195))
            if (not self.took_profit
                    and self.entry_price is not None
                    and bar.close >= self.entry_price * (1.0 + tp_pct)):
                self.took_profit = True
                pbr = extras.get("positive_bar_ratio_72h")
                tp_frac = 0.40
                if pbr is not None and pbr == pbr:
                    tp_frac = max(0.01, min(0.90, 1.00 - pbr * 1.15))
                return [{"side": "sell", "size": abs(current_pos) * tp_frac}]

            # Trail stop (wider after TP)
            trail = 0.021 if self.took_profit else self.parameters["trail_pct"]
            trail_stop = self.highest_since_entry * (1.0 - trail)
            if bar.close <= trail_stop:
                self.highest_since_entry = None
                self.entry_price = None
                self.trend_at_exit = trend_up and uptrend_structure
                self.bars_since_exit = 0
                self.prev_trend_up = trend_up
                return [{"side": "sell", "size": abs(current_pos)}]

        self.prev_trend_up = trend_up
        return []


# ---- DO NOT EDIT BELOW THIS LINE ----
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["local", "external"], default="external")
    ap.add_argument("--external-dir", default=None)
    args = ap.parse_args()
    kwargs = {"source": args.source}
    if args.external_dir:
        kwargs["external_dir"] = args.external_dir
    bars = load_bars(**kwargs)
    evaluate(Strategy, bars)
