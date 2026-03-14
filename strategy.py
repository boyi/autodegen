"""
autodegen strategy — THE ONLY FILE THE AGENT EDITS
Run: python strategy.py

Simple EMA Crossover v10
- Simple dual EMA crossover
- Long only
- Minimal parameters for robustness
"""

from prepare import evaluate, load_bars


class Strategy:
    name = "simple_ema_crossover_v10"
    parameters = {
        "ema_fast": 21,
        "ema_slow": 50,
        "size": 0.04,
        "trail_pct": 0.018,         # Slightly tighter trail
    }

    def initialize(self, train_data):
        self.close_history = []
        self.ema_fast_val = None
        self.ema_slow_val = None
        self.prev_trend_up = False
        self.entry_price = None
        self.highest_since_entry = None

    def _ema(self, prev, price, period):
        if prev is None:
            return price
        alpha = 2.0 / (period + 1)
        return alpha * price + (1.0 - alpha) * prev

    def _target_order(self, target_position, portfolio):
        current = portfolio["position"]
        delta = target_position - current
        if abs(delta) < 1e-12:
            return []
        return [{"side": "buy" if delta > 0 else "sell", "size": abs(delta)}]

    def on_bar(self, bar, portfolio):
        self.close_history.append(bar.close)
        
        self.ema_fast_val = self._ema(self.ema_fast_val, bar.close, self.parameters["ema_fast"])
        self.ema_slow_val = self._ema(self.ema_slow_val, bar.close, self.parameters["ema_slow"])
        
        max_hist = 100
        if len(self.close_history) > max_hist:
            self.close_history = self.close_history[-max_hist:]
        
        if len(self.close_history) < self.parameters["ema_slow"]:
            return []
        
        current_pos = portfolio["position"]
        
        # Trend direction
        trend_up = self.ema_fast_val > self.ema_slow_val
        
        # Exit logic
        if current_pos > 0:
            self.highest_since_entry = max(self.highest_since_entry, bar.high)
            trail_stop = self.highest_since_entry * (1.0 - self.parameters["trail_pct"])
            
            if bar.close <= trail_stop:
                self.entry_price = None
                self.highest_since_entry = None
                return self._target_order(0.0, portfolio)
        
        # Entry logic - LONG ONLY on trend change to up
        if current_pos == 0:
            # Enter when trend turns from down to up
            if trend_up and not self.prev_trend_up:
                self.entry_price = bar.close
                self.highest_since_entry = bar.high
                self.prev_trend_up = True
                return self._target_order(self.parameters["size"], portfolio)
        
        # Update trend state
        self.prev_trend_up = trend_up
        
        return []


# ---- DO NOT EDIT BELOW THIS LINE ----
if __name__ == "__main__":
    bars = load_bars()
    evaluate(Strategy, bars)
