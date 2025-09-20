"""
swing_portfolio_manager.py - Modular SwingPortfolioManager class for better debugging and code organization
"""
import pandas as pd
import numpy as np
import os

RESULTS_DIR = "results"
class SwingPortfolioManager:
    def __init__(self,
                 initial_balance=100000,
                 max_positions=5,
                 min_hold_days=7,
                 max_hold_days=20,
                 cost_rate=0.001,
                 require_flat_before_new=True,
                 portfolio_increase_threshold=0.2,
                 allocation_increment=0.15,
                 trailing_sl_after_t3=False,):
        """
        trailing_sl_after_t3: if True, after T3 hit, instead of exiting all remaining shares,
        trail stop-loss upward by 1 ATR for each additional ATR price moves up.
        Also REMOVES max hold days constraint when beyond T3.
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_positions = max_positions
        self.min_hold_days = min_hold_days
        self.max_hold_days = max_hold_days
        self.cost_rate = cost_rate
        self.require_flat_before_new = require_flat_before_new
        
        # Dynamic allocation parameters
        self.portfolio_increase_threshold = portfolio_increase_threshold
        self.allocation_increment = allocation_increment
        self.current_allocation_multiplier = 1.0
        self.last_allocation_update_value = initial_balance
        
        # NEW: Trailing stop-loss after T3 parameter
        self.trailing_sl_after_t3 = trailing_sl_after_t3
        
        self.positions = {}
        self.trade_log = []
        self.portfolio_history = []

    def _check_and_update_allocation(self):
        """Check if portfolio has grown enough to increase allocation"""
        if not self.portfolio_history:
            return
        
        current_portfolio_value = self.portfolio_history[-1]["portfolio_value"]
        growth = (current_portfolio_value - self.last_allocation_update_value) / self.last_allocation_update_value
        
        if growth >= self.portfolio_increase_threshold:
            old_multiplier = self.current_allocation_multiplier
            self.current_allocation_multiplier += self.allocation_increment
            self.last_allocation_update_value = current_portfolio_value
            
            print(f"🚀 Portfolio grew {growth*100:.1f}% to {current_portfolio_value:,.0f}")
            print(f"💰 Allocation multiplier increased: {old_multiplier:.2f} → {self.current_allocation_multiplier:.2f}")

    def _has_partial_outstanding(self):
        return len(self.positions) > 0

    def _buy(self, date, symbol, row, capital_per_slot):
        self._check_and_update_allocation()
        
        price = row["close"]
        atr = row["ATR_14"]
        
        if pd.isna(price) or pd.isna(atr) or price <= 0:
            return False
        
        adjusted_capital = capital_per_slot * self.current_allocation_multiplier
        adjusted_capital = min(adjusted_capital, self.balance * 0.9)
        shares = int(adjusted_capital // price)
        
        if shares <= 0:
            return False
        
        stop_loss = price - 1.5 * atr
        t1 = price + 1.0 * atr
        t2 = price + 2.0 * atr
        t3 = price + 3.0 * atr
        
        cost = shares * price * self.cost_rate
        total_cash = shares * price + cost
        
        if self.balance < total_cash:
            return False
        
        self.balance -= total_cash
        
        if symbol in self.positions:
            existing_pos = self.positions[symbol]
            old_shares = existing_pos["shares"]
            old_price = existing_pos["entry_price"]
            total_shares = old_shares + shares
            weighted_avg_price = (old_shares * old_price + shares * price) / total_shares
            
            existing_pos["shares"] = total_shares
            existing_pos["entry_price"] = weighted_avg_price
            existing_pos["atr"] = atr
            existing_pos["stop_loss"] = weighted_avg_price - 1.5 * atr
            existing_pos["targets"] = [weighted_avg_price + 1.0 * atr,
                                      weighted_avg_price + 2.0 * atr,
                                      weighted_avg_price + 3.0 * atr]
        else:
            self.positions[symbol] = {
                "shares": shares,
                "entry_price": price,
                "entry_date": pd.to_datetime(date),
                "atr": atr,
                "stop_loss": stop_loss,
                "targets": [t1, t2, t3],
                "milestones_hit": 0
            }
        
        self.trade_log.append({
            "date": date, "action": "BUY", "symbol": symbol,
            "shares": shares, "entry_price": price, "exit_price": np.nan,
            "pnl": np.nan, "reason": "ENTRY", "holding_days": np.nan,
            "allocation_multiplier": self.current_allocation_multiplier,
            "adjusted_capital": adjusted_capital
        })
        
        return True

    def _sell_fraction(self, date, symbol, exit_price, fraction, reason):
        """
        Sell 'fraction' of the current shares (fraction between 0 and 1).
        If fraction == 1.0, sell remaining shares.
        """
        pos = self.positions.get(symbol)
        if pos is None:
            return
        
        shares_current = pos["shares"]
        shares_to_sell = int(round(shares_current * fraction))
        
        if shares_to_sell <= 0 and shares_current > 0 and fraction > 0:
            shares_to_sell = 1
        
        if shares_to_sell <= 0:
            return
        
        proceeds = shares_to_sell * exit_price
        cost = proceeds * self.cost_rate
        self.balance += (proceeds - cost)
        
        pnl = (exit_price - pos["entry_price"]) * shares_to_sell - cost
        
        self.trade_log.append({
            "date": date, "action": "SELL", "symbol": symbol,
            "shares": shares_to_sell, "entry_price": pos["entry_price"], "exit_price": exit_price,
            "pnl": pnl, "reason": reason, "holding_days": (pd.to_datetime(date) - pos["entry_date"]).days,
            "allocation_multiplier": self.current_allocation_multiplier,
            "adjusted_capital": np.nan
        })
        
        pos["shares"] = pos["shares"] - shares_to_sell
        if pos["shares"] <= 0:
            self.positions.pop(symbol, None)

    def manage_day(self, date, today_rows_dict, predictions, prob_threshold=0.55):
        """
        Manage existing positions (partial exits & trailing) and open new ones.
        """
        # ---------- EXITS: iterate over copy because we may modify positions ----------
        for symbol, pos in list(self.positions.items()):
            row = today_rows_dict.get(symbol)
            if row is None:
                continue

            high = row.get("high", row.get("close"))
            low = row.get("low", row.get("close"))
            close = row["close"]
            days_held = (pd.to_datetime(date) - pos["entry_date"]).days

            # 1) STOP-LOSS check (immediate always)
            if low <= pos["stop_loss"]:
                exit_price = pos["stop_loss"]
                self._sell_fraction(date, symbol, exit_price, 1.0, reason="STOPLOSS")
                continue

            # 2) Check target hits this day (use day's high)
            targets = pos["targets"]
            milestones = pos["milestones_hit"]
            fractions = [0.30, 0.30, 1.0]

            # Handle milestone progression
            for t_idx in range(milestones, len(targets)):
                target_price = targets[t_idx]
                if high >= target_price:
                    if t_idx < 2:  # T1 and T2 - partial exits as before
                        frac = fractions[t_idx]
                        self._sell_fraction(date, symbol, target_price, frac, reason=f"T{t_idx+1}_EXIT")

                        # Update trailing stops
                        if t_idx == 0:  # After T1: trail stop to entry (breakeven)
                            pos_local = self.positions.get(symbol)
                            if pos_local is not None:
                                pos_local["stop_loss"] = pos_local["entry_price"]
                                pos_local["milestones_hit"] = 1
                        elif t_idx == 1:  # After T2: trail stop to entry + 1 ATR
                            pos_local = self.positions.get(symbol)
                            if pos_local is not None:
                                pos_local["stop_loss"] = pos_local["entry_price"] + 1.0 * pos_local["atr"]
                                pos_local["milestones_hit"] = 2

                    else:  # t_idx == 2 (T3) - NEW LOGIC HERE
                        if self.trailing_sl_after_t3:
                            # TRAIL STOP-LOSS INSTEAD OF EXITING
                            pos_local = self.positions.get(symbol)
                            if pos_local is not None:
                                # Move stop-loss to entry + 2 ATR (T2 level) initially at T3
                                pos_local["stop_loss"] = pos_local["entry_price"] + 2.0 * pos_local["atr"]
                                pos_local["milestones_hit"] = 3
                                # Add tracking for trailing beyond T3
                                pos_local["trailing_level"] = 3

                                # Log T3 milestone reached but not exited
                                self.trade_log.append({
                                    "date": date, "action": "T3_TRAIL", "symbol": symbol,
                                    "shares": 0, "entry_price": pos_local["entry_price"],
                                    "exit_price": np.nan, "pnl": np.nan,
                                    "reason": "T3_MILESTONE_TRAIL", "holding_days": days_held,
                                    "allocation_multiplier": self.current_allocation_multiplier,
                                    "adjusted_capital": np.nan
                                })
                        else:
                            # ORIGINAL BEHAVIOR: exit all remaining at T3
                            self._sell_fraction(date, symbol, target_price, 1.0, reason="T3_EXIT")

                    # Refresh position (may have been removed)
                    if symbol not in self.positions:
                        break
                else:
                    # Target not reached, no higher targets reached either
                    break

            # NEW: Handle trailing beyond T3 if enabled
            if symbol in self.positions and self.trailing_sl_after_t3:
                pos_local = self.positions[symbol]
                if pos_local.get("milestones_hit", 0) >= 3:  # T3 already hit
                    # Check if price has moved up another ATR level
                    trailing_level = pos_local.get("trailing_level", 3)
                    next_trail_target = pos_local["entry_price"] + (trailing_level + 1) * pos_local["atr"]

                    if high >= next_trail_target:
                        # Price moved up another ATR - trail stop-loss up by 1 ATR
                        current_sl = pos_local["stop_loss"]
                        new_sl = current_sl + 1.0 * pos_local["atr"]

                        # Update stop-loss and trailing level
                        pos_local["stop_loss"] = new_sl
                        pos_local["trailing_level"] = trailing_level + 1

                        # Log the trailing action
                        self.trade_log.append({
                            "date": date, "action": "TRAIL_SL", "symbol": symbol,
                            "shares": 0, "entry_price": pos_local["entry_price"],
                            "exit_price": np.nan, "pnl": np.nan,
                            "reason": f"TRAIL_T{trailing_level+1}", "holding_days": days_held,
                            "allocation_multiplier": self.current_allocation_multiplier,
                            "adjusted_capital": np.nan
                        })

            # 3) UPDATED: Max hold days check - SKIP if trailing beyond T3
            if symbol in self.positions:
                pos2 = self.positions[symbol]
                days_held = (pd.to_datetime(date) - pos2["entry_date"]).days
                if days_held >= self.max_hold_days:
                    # Skip force exit if trailing_sl_after_t3 is enabled and beyond T3
                    if not (self.trailing_sl_after_t3 and pos2.get("milestones_hit", 0) >= 3):
                        # force exit remaining at close
                        self._sell_fraction(date, symbol, close, 1.0, reason="TIMEEXIT")

        # ---------- ENTRIES ----------
        if self.require_flat_before_new and len(self.positions) > 0:
            self.record_value(date, {s: today_rows_dict[s]["close"] for s in today_rows_dict})
            return

        preds_today = predictions.loc[predictions.index.isin(list(today_rows_dict.keys()))]
        if preds_today.empty:
            self.record_value(date, {s: today_rows_dict[s]["close"] for s in today_rows_dict})
            return

        #breakout_symbols = [s for s, r in today_rows_dict.items() if r.get("darvas_breakout_up", 0) == 1]
        #if not breakout_symbols:
        #    self.record_value(date, {s: today_rows_dict[s]["close"] for s in today_rows_dict})
        #    return

        #eligible = preds_today.loc[preds_today.index.isin(breakout_symbols)]
        eligible = preds_today.copy()
        eligible = eligible[eligible > prob_threshold].sort_values(ascending=False)
        if eligible.empty:
            self.record_value(date, {s: today_rows_dict[s]["close"] for s in today_rows_dict})
            return

        open_slots = max(0, self.max_positions - len(self.positions))
        for symbol, prob in eligible.items():
            if open_slots <= 0:
                break
            if symbol in self.positions:
                continue

            capital_per_slot = self.balance / open_slots if open_slots > 0 else 0
            bought = self._buy(date, symbol, today_rows_dict[symbol], capital_per_slot)
            if bought:
                open_slots -= 1

        self.record_value(date, {s: today_rows_dict[s]["close"] for s in today_rows_dict})

    def record_value(self, date, current_prices):
        holdings_val = sum(pos["shares"] * current_prices.get(sym, pos["entry_price"]) for sym, pos in self.positions.items())
        total = self.balance + holdings_val
        self.portfolio_history.append({"date": pd.to_datetime(date), "portfolio_value": total})

    def export_results(self, out_dir=RESULTS_DIR):
        os.makedirs(out_dir, exist_ok=True)
        trades_df = pd.DataFrame(self.trade_log)
        hist_df = pd.DataFrame(self.portfolio_history)
        trades_df.to_csv(os.path.join(out_dir, "trades.csv"), index=False)
        hist_df.to_csv(os.path.join(out_dir, "portfolio_history.csv"), index=False)
        return trades_df, hist_df
