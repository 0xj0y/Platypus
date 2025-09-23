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
                 trailing_sl_after_t3=False,
                 activate_drawdown_scaling=True,
                 position_scaling_increment=400000,
                 max_positions_limit=10,):
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

        # Emergency Drawdown Scaling
        self.activate_drawdown_scaling = activate_drawdown_scaling
        self.risk_multiplier = 1.0
        self.drawdown_scaling_active = False
        self.portfolio_peak = initial_balance
        self.current_trough = None
        
        # Position Scaling Logic - Fully Configurable
        self.initial_max_positions = max_positions
        self.position_scaling_increment = position_scaling_increment
        self.max_positions_limit = max_positions_limit
        self.last_position_scaling_value = initial_balance

    # ----------------- Allocation Growth -----------------
    def _check_and_update_allocation(self):
        """Check if portfolio has grown enough to increase allocation."""
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

    def _check_and_update_position_count(self):
        """Check if portfolio has grown enough to increase max_positions by 1."""
        if not self.portfolio_history:
            return
            
        current_portfolio_value = self.portfolio_history[-1]["portfolio_value"]
        portfolio_growth = current_portfolio_value - self.last_position_scaling_value
        
        if (portfolio_growth >= self.position_scaling_increment and 
            self.max_positions < self.max_positions_limit):
            
            old_max_positions = self.max_positions
            self.max_positions += 1
            self.last_position_scaling_value = current_portfolio_value
            
            print(f"📈 Portfolio grew by ${portfolio_growth:,.0f} to ${current_portfolio_value:,.0f}")
            print(f"🎯 Max positions increased: {old_max_positions} → {self.max_positions} (limit: {self.max_positions_limit})")

    
    
    # ----------------- Drawdown Scaling -----------------
    def _update_drawdown_scaling(self):
        """Update risk multiplier - BALANCED VERSION with aggressive recovery"""
        if not self.activate_drawdown_scaling:
            return
        
        if not self.portfolio_history:
            return
        
        current_value = self.portfolio_history[-1]["portfolio_value"]
        
        # Update peak logic (unchanged)
        if current_value > self.portfolio_peak:
            self.portfolio_peak = current_value
            if self.drawdown_scaling_active:
                print(f"🎉 NEW PEAK: ${current_value:,.0f} - Risk reset to full")
                self.drawdown_scaling_active = False
                self.risk_multiplier = 1.0
                self.current_trough = None
            return
        
        # Current drawdown
        current_drawdown = (self.portfolio_peak - current_value) / self.portfolio_peak
        print(f"DEBUG DD: Portfolio=${current_value:,.0f}, Peak=${self.portfolio_peak:,.0f}, "
            f"DD={current_drawdown*100:.1f}%, Risk={self.risk_multiplier:.2f}")
        
        # Track trough
        if self.current_trough is None or current_value < self.current_trough:
            self.current_trough = current_value
        
        old_multiplier = self.risk_multiplier
        
        # 🚨 BALANCED scaling - preserve more upside
        if current_drawdown >= 0.25 and self.risk_multiplier > 0.3:  # Only at 25%+
            self.risk_multiplier = 0.3  # 30% instead of 10%
            self.drawdown_scaling_active = True
            print(f"🚨 SEVERE DRAWDOWN: {current_drawdown*100:.1f}% DD → Risk cut to 30%")
            
        elif current_drawdown >= 0.15 and self.risk_multiplier > 0.5:  # 15% instead of 12%
            self.risk_multiplier = 0.5  # 50% instead of 20%
            self.drawdown_scaling_active = True
            print(f"⚠️ HIGH DRAWDOWN: {current_drawdown*100:.1f}% → Risk cut to 50%")
            
        elif current_drawdown >= 0.10 and self.risk_multiplier > 0.7:  # 10% instead of 8%
            self.risk_multiplier = 0.7  # 70% instead of 40%
            self.drawdown_scaling_active = True
            print(f"🔸 MODERATE DRAWDOWN: {current_drawdown*100:.1f}% → Risk cut to 70%")
        
        # 📈 AGGRESSIVE RECOVERY - Your requested scaling
        if self.drawdown_scaling_active:
            recovery_from_trough = 0
            if self.current_trough is not None and self.current_trough > 0:
                recovery_from_trough = (current_value - self.current_trough) / self.current_trough
                
            if current_drawdown < 0.05:  # 5% drawdown for full recovery
                self.risk_multiplier = 1.0
                self.drawdown_scaling_active = False
                self.current_trough = None
                print(f"✅ FULL RECOVERY: Drawdown {current_drawdown*100:.1f}% → Normal risk restored")
                
            elif recovery_from_trough >= 0.20:  # 20% recovery → 90% increase
                # Calculate 90% increase from current level
                target_multiplier = min(1.0, self.risk_multiplier * 1.9)  # 90% increase
                if target_multiplier > self.risk_multiplier:
                    self.risk_multiplier = target_multiplier
                    print(f"🚀 STRONG RECOVERY: +{recovery_from_trough*100:.1f}% from trough → Risk up 90% to {self.risk_multiplier:.2f}")
                
            elif recovery_from_trough >= 0.10:  # 10% recovery → 50% increase  
                # Calculate 50% increase from current level
                target_multiplier = min(1.0, self.risk_multiplier * 1.5)  # 50% increase
                if target_multiplier > self.risk_multiplier:
                    self.risk_multiplier = target_multiplier
                    print(f"📈 PARTIAL RECOVERY: +{recovery_from_trough*100:.1f}% from trough → Risk up 50% to {self.risk_multiplier:.2f}")
        
        # Log any change
        if abs(self.risk_multiplier - old_multiplier) > 0.01:
            print(f"🎯 RISK MULTIPLIER: {old_multiplier:.2f} → {self.risk_multiplier:.2f}")


    # ----------------- Position Management -----------------
    def _has_partial_outstanding(self):
        return len(self.positions) > 0

    def _buy(self, date, symbol, row, capital_per_slot):
        self._check_and_update_allocation()
        self._update_drawdown_scaling()

        price = row["close"]
        atr = row["ATR_14"]

        if pd.isna(price) or pd.isna(atr) or price <= 0:
            return False
        # Apply multipliers conditionally
        risk_multiplier = self.risk_multiplier if self.activate_drawdown_scaling else 1.0
        adjusted_capital = capital_per_slot * self.current_allocation_multiplier * risk_multiplier
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
        """Sell 'fraction' of the current shares (fraction between 0 and 1)."""
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

    # ----------------- Daily Portfolio Management -----------------
    def manage_day(self, date, today_rows_dict, predictions, prob_threshold=0.55, blacklisted_symbols=None):

        """Manage existing positions (partial exits & trailing) and open new ones."""
        
        self._check_and_update_position_count()
        if self.activate_drawdown_scaling:
            self._update_drawdown_scaling()

        # ---------- EXITS ----------
        for symbol, pos in list(self.positions.items()):
            row = today_rows_dict.get(symbol)
            if row is None:
                continue

            high = row.get("high", row.get("close"))
            low = row.get("low", row.get("close"))
            close = row["close"]
            days_held = (pd.to_datetime(date) - pos["entry_date"]).days

            # 1) STOP-LOSS check
            if low <= pos["stop_loss"]:
                exit_price = pos["stop_loss"]
                self._sell_fraction(date, symbol, exit_price, 1.0, reason="STOPLOSS")
                continue

            # 2) Targets check
            targets = pos["targets"]
            milestones = pos["milestones_hit"]
            fractions = [0.30, 0.30, 1.0]

            for t_idx in range(milestones, len(targets)):
                target_price = targets[t_idx]
                if high >= target_price:
                    if t_idx < 2:  # T1, T2 partial exits
                        frac = fractions[t_idx]
                        self._sell_fraction(date, symbol, target_price, frac, reason=f"T{t_idx+1}_EXIT")

                        if t_idx == 0:  # After T1
                            pos_local = self.positions.get(symbol)
                            if pos_local is not None:
                                pos_local["stop_loss"] = pos_local["entry_price"]
                                pos_local["milestones_hit"] = 1
                        elif t_idx == 1:  # After T2
                            pos_local = self.positions.get(symbol)
                            if pos_local is not None:
                                pos_local["stop_loss"] = pos_local["entry_price"] + 1.0 * pos_local["atr"]
                                pos_local["milestones_hit"] = 2

                    else:  # T3
                        if self.trailing_sl_after_t3:
                            pos_local = self.positions.get(symbol)
                            if pos_local is not None:
                                pos_local["stop_loss"] = pos_local["entry_price"] + 2.0 * pos_local["atr"]
                                pos_local["milestones_hit"] = 3
                                pos_local["trailing_level"] = 3

                                self.trade_log.append({
                                    "date": date, "action": "T3_TRAIL", "symbol": symbol,
                                    "shares": 0, "entry_price": pos_local["entry_price"],
                                    "exit_price": np.nan, "pnl": np.nan,
                                    "reason": "T3_MILESTONE_TRAIL", "holding_days": days_held,
                                    "allocation_multiplier": self.current_allocation_multiplier,
                                    "adjusted_capital": np.nan
                                })
                        else:
                            self._sell_fraction(date, symbol, target_price, 1.0, reason="T3_EXIT")

                    if symbol not in self.positions:
                        break
                else:
                    break

            # Trailing beyond T3
            if symbol in self.positions and self.trailing_sl_after_t3:
                pos_local = self.positions[symbol]
                if pos_local.get("milestones_hit", 0) >= 3:
                    trailing_level = pos_local.get("trailing_level", 3)
                    next_trail_target = pos_local["entry_price"] + (trailing_level + 1) * pos_local["atr"]

                    if high >= next_trail_target:
                        current_sl = pos_local["stop_loss"]
                        new_sl = current_sl + 1.0 * pos_local["atr"]

                        pos_local["stop_loss"] = new_sl
                        pos_local["trailing_level"] = trailing_level + 1

                        self.trade_log.append({
                            "date": date, "action": "TRAIL_SL", "symbol": symbol,
                            "shares": 0, "entry_price": pos_local["entry_price"],
                            "exit_price": np.nan, "pnl": np.nan,
                            "reason": f"TRAIL_T{trailing_level+1}", "holding_days": days_held,
                            "allocation_multiplier": self.current_allocation_multiplier,
                            "adjusted_capital": np.nan
                        })

            # 3) Max hold days check
            if symbol in self.positions:
                pos2 = self.positions[symbol]
                days_held = (pd.to_datetime(date) - pos2["entry_date"]).days
                if days_held >= self.max_hold_days:
                    if not (self.trailing_sl_after_t3 and pos2.get("milestones_hit", 0) >= 3):
                        self._sell_fraction(date, symbol, close, 1.0, reason="TIMEEXIT")

        # ---------- ENTRIES ----------
        if self.require_flat_before_new and len(self.positions) > 0:
            self.record_value(date, {s: today_rows_dict[s]["close"] for s in today_rows_dict})
            return

        preds_today = predictions.loc[predictions.index.isin(list(today_rows_dict.keys()))]
        if preds_today.empty:
            self.record_value(date, {s: today_rows_dict[s]["close"] for s in today_rows_dict})
            return

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
            if blacklisted_symbols and symbol in blacklisted_symbols:
                continue
            capital_per_slot = self.balance / open_slots if open_slots > 0 else 0
            bought = self._buy(date, symbol, today_rows_dict[symbol], capital_per_slot)
            if bought:
                open_slots -= 1

        self.record_value(date, {s: today_rows_dict[s]["close"] for s in today_rows_dict})

    # ----------------- Record & Export -----------------
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
