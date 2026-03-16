"""
autodegen strategy — THE ONLY FILE THE AGENT EDITS
Run: python strategy.py --source external
"""

import argparse

from prepare import evaluate, load_bars


class Strategy:
    name = "ema_20_50_hh_hl_volz_reentry_tp_v22"
    description = (
        "EMA 20/50 + HH/HL + volz sizing + filtered re-entry + partial TP. "
        "Sell half position when trade is +3% profitable. Locks in gains, "
        "reduces downside exposure, improves sortino."
    )
    parameters = {
        "ema_fast": 20,
        "ema_slow": 50,
        "structure_lookback": 8,
        "base_size": 0.96,
        "trail_pct": 0.019,
        "volz_scale": 0.50,
        "reentry_cooldown": 12,
        "reentry_trend_min": 0.33,
        "tp_pct": 0.05,
    }

    def initialize(self, train_data):
        self.close_history = []
        self.high_history = []
        self.low_history = []
        self.ema_fast_val = None
        self.ema_slow_val = None
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
        uptrend_structure = hh and hl

        if current_pos == 0:
            self.bars_since_exit += 1

            # Normal entry: EMA crossover + structure
            crossover_entry = trend_up and not self.prev_trend_up and uptrend_structure

            # Re-entry: stopped out of a still-bullish trend, cooldown elapsed,
            # AND enriched trend consistency confirms
            extras = bar.extras or {}
            trend_c = extras.get("trend_consistency_3d")
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
                    scale = max(0.50, min(1.50, 1.0 - self.parameters["volz_scale"] * volz))
                    size = self.parameters["base_size"] * scale

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

            # Partial profit-taking: sell half when +tp_pct%
            if (not self.took_profit
                    and self.entry_price is not None
                    and bar.close >= self.entry_price * (1.0 + self.parameters["tp_pct"])):
                self.took_profit = True
                return [{"side": "sell", "size": abs(current_pos) * 0.5}]

            trail_stop = self.highest_since_entry * (1.0 - self.parameters["trail_pct"])
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
