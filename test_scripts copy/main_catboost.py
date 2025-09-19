# main_catboost.py

"""
Swing backtest with:
- Entry only when darvas_breakout_up == 1 AND model prob > prob_threshold
- INCREASING ALLOCATION: Increase allocation by 15% whenever portfolio value increases by 20%
- Partial exits at ATR milestones:
  T1 = +1 ATR -> exit 30% -> trail SL to entry
  T2 = +2 ATR -> exit 30% -> trail SL to entry + 1 ATR
  T3 = +3 ATR -> exit remaining (40%) OR trail SL if trailing_sl_after_t3=True
- Initial SL = entry - 1.5 ATR (active immediately)
- Max hold = 20 days (force exit remaining, unless trailing beyond T3)
- Optional trailing SL after T3 - keeps position open and trails SL upward
- Labels: 1 if TP (3 ATR) touched before SL within horizon (keeps consistency)
- NEW: Stop-Loss Loss Report for trades that hit SL first and turned into loss
- NEW: Winning Trades Report for trades that hit T1 exit (first target)

Outputs in results/ and plots/catboost/
"""

import os
import glob
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, log_loss
import shap
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="catboost")
plt.rcParams["figure.figsize"] = (10, 5)

RESULTS_DIR = "results"
PLOTS_DIR = "plots/catboost"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# -------------------------
# Helper: label target with SL/TP-first logic
# -------------------------

def label_with_atr_first_touch(df, horizon=20, atr_col="ATR_14"):
    """
    For each symbol-row: label = 1 if, within next `horizon` days,
    the TP (entry + 3*ATR) is reached before SL (entry - 1.5*ATR).
    Otherwise label = 0.
    Uses daily high/low sequencing (best-effort).
    """
    df = df.copy()
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    labels = np.zeros(len(df), dtype=int)
    
    for sym, g in df.groupby("symbol", sort=False):
        idxs = g.index.to_numpy()
        highs = g["high"].to_numpy()
        lows = g["low"].to_numpy()
        closes = g["close"].to_numpy()
        atrs = g[atr_col].to_numpy()
        n = len(g)
        
        for local_i in range(n):
            global_i = idxs[local_i]
            entry = closes[local_i]
            atr = atrs[local_i]
            
            if pd.isna(entry) or pd.isna(atr):
                labels[global_i] = 0
                continue
            
            tp = entry + 3.0 * atr
            sl = entry - 1.5 * atr
            tp_hit = None
            sl_hit = None
            
            end_local = min(local_i + horizon, n - 1)
            for k in range(local_i + 1, end_local + 1):
                if tp_hit is None and highs[k] >= tp:
                    tp_hit = k
                if sl_hit is None and lows[k] <= sl:
                    sl_hit = k
                if tp_hit is not None and sl_hit is not None:
                    break
            
            if tp_hit is not None and (sl_hit is None or tp_hit < sl_hit):
                labels[global_i] = 1
            else:
                labels[global_i] = 0
    
    df["target"] = labels
    return df

# -------------------------
# NEW: Stop-Loss Loss Report Function
# -------------------------

def create_sl_loss_report(trades_df, master_df=None):
    """
    Create a report for trades that hit stop-loss first and turned into loss.
    Includes: date, symbol, entry_price, shares, RSI_14
    """
    # Filter trades that hit SL first and turned into loss
    loss_sl_trades = trades_df[(trades_df["action"] == "SELL") &
                               (trades_df["reason"] == "STOPLOSS") &
                               (trades_df["pnl"] < 0)].copy()
    
    if loss_sl_trades.empty:
        print("No stop-loss loss trades found.")
        return pd.DataFrame(columns=["date", "symbol", "entry_price", "shares", "RSI_14"])
    
    # Add RSI_14 if master_df provided
    if master_df is not None and "RSI_14" in master_df.columns:
        def find_closest_rsi(row):
            sym = row["symbol"]
            d = row["date"]
            df_sym = master_df[master_df["symbol"] == sym]
            if df_sym.empty:
                return np.nan
            # Find date closest to trade date
            diffs = (df_sym["date"] - d).abs()
            closest_idx = diffs.idxmin()
            return df_sym.loc[closest_idx, "RSI_14"]
        
        loss_sl_trades["RSI_14"] = loss_sl_trades.apply(find_closest_rsi, axis=1)
    else:
        loss_sl_trades["RSI_14"] = np.nan
    
    # Select relevant columns for report
    report = loss_sl_trades[["date", "symbol", "entry_price", "shares", "RSI_14"]].copy()
    return report

# -------------------------
# NEW: Winning Trades Report Function
# -------------------------

def create_win_report(trades_df, master_df=None):
    """
    Create a report for winning trades that hit T1 exit (first target).
    Includes: date, symbol, entry_price, shares, RSI_14
    """
    # Filter trades that have T1_EXIT as reason (winning trades)
    win_trades = trades_df[(trades_df["action"] == "SELL") &
                           (trades_df["reason"].str.startswith("T1_EXIT", na=False))].copy()
    
    if win_trades.empty:
        print("No winning trades found.")
        return pd.DataFrame(columns=["date", "symbol", "entry_price", "shares", "RSI_14"])
    
    # Add RSI_14 if master_df provided
    if master_df is not None and "RSI_14" in master_df.columns:
        def find_closest_rsi(row):
            sym = row["symbol"]
            d = row["date"]
            df_sym = master_df[master_df["symbol"] == sym]
            if df_sym.empty:
                return np.nan
            # Find date closest to trade date
            diffs = (df_sym["date"] - d).abs()
            closest_idx = diffs.idxmin()
            return df_sym.loc[closest_idx, "RSI_14"]
        
        win_trades["RSI_14"] = win_trades.apply(find_closest_rsi, axis=1)
    else:
        win_trades["RSI_14"] = np.nan
    
    # Select relevant columns for report
    report = win_trades[["date", "symbol", "entry_price", "shares", "RSI_14"]].copy()
    return report

# -------------------------
# Portfolio Manager with Partial Exits + Trailing + DYNAMIC ALLOCATION + NEW TRAILING SL
# -------------------------

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
                 trailing_sl_after_t3=False): # Change True to trail to ride the rally
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

        breakout_symbols = [s for s, r in today_rows_dict.items() if r.get("darvas_breakout_up", 0) == 1]
        if not breakout_symbols:
            self.record_value(date, {s: today_rows_dict[s]["close"] for s in today_rows_dict})
            return

        eligible = preds_today.loc[preds_today.index.isin(breakout_symbols)]
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

# -------------------------
# Data + Features
# -------------------------

def create_master_dataframe(folder_path, start_date_str="2019-01-01"):
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    csv_files = [f for f in csv_files if "master_with_indicators" not in os.path.basename(f)]
    
    if not csv_files:
        raise FileNotFoundError("No CSV files found in folder: " + folder_path)
    
    frames = []
    for fp in csv_files:
        try:
            df = pd.read_csv(fp, parse_dates=["date"])
            frames.append(df)
        except Exception as e:
            print("Skipping", fp, ":", e)
    
    master = pd.concat(frames, ignore_index=True)
    master["date"] = pd.to_datetime(master["date"])
    master = master[master["date"] >= pd.to_datetime(start_date_str)].copy()
    master.sort_values(["symbol", "date"], inplace=True)
    master.reset_index(drop=True, inplace=True)
    return master

def engineer_features_and_labels(df, horizon=20):
    df = df.copy()
    df["ema_9_21_diff"] = (df["EMA_9"] - df["EMA_21"]) / df["close"]
    df["ema_50_200_diff"] = (df["EMA_50"] - df["EMA_200"]) / df["close"]
    
    df_labeled = label_with_atr_first_touch(df, horizon=horizon, atr_col="ATR_14")
    
    features = [
        "RSI_14", "VOL_20", "NORM_ATR", "volume_ratio",
        "darvas_high", "darvas_low", "darvas_breakout_up",
        "ema_9_21_diff", "ema_50_200_diff"
    ]
    
    df_labeled.dropna(subset=features + ["target"], inplace=True)
    return df_labeled, features

# -------------------------
# Backtest Runner with CatBoost
# -------------------------
  
def run_backtest(folder,
                start_date="2019-01-01",
                initial_balance=100000,
                max_positions=5,
                prob_threshold=0.55,
                retrain_freq_days=20,
                train_window_days=365*2,
                val_window_days=60,
                horizon=20,
                cost_rate=0.001,
                require_flat_before_new=True,
                portfolio_increase_threshold=0.2,
                allocation_increment=0.15,
                trailing_sl_after_t3=False): # change to ride rally beyond T3

    master = create_master_dataframe(folder, start_date)
    df, features = engineer_features_and_labels(master, horizon=horizon)
    unique_days = sorted(df["date"].unique())
    
    if not unique_days:
        raise ValueError("No dates after filtering")

    pm = SwingPortfolioManager(
        initial_balance=initial_balance,
        max_positions=max_positions,
        min_hold_days=7,
        max_hold_days=20,
        cost_rate=cost_rate,
        require_flat_before_new=require_flat_before_new,
        portfolio_increase_threshold=portfolio_increase_threshold,
        allocation_increment=allocation_increment,
        trailing_sl_after_t3=trailing_sl_after_t3  # NEW PARAMETER
    )

    model = None
    explainer = None
    training_log = []
    day_map = {d: df[df["date"] == d].copy() for d in unique_days}

    for i, current_date in enumerate(unique_days):
        today_df = day_map[current_date]
        today_rows = {row["symbol"]: row for _, row in today_df.iterrows()}

        # retrain periodically
        if model is None or i % retrain_freq_days == 0:
            cutoff = current_date - pd.Timedelta(days=train_window_days)
            train_data = df[(df["date"] >= cutoff) & (df["date"] < current_date)].copy()
            
            if len(train_data) >= 200:
                X_train = train_data[features]
                y_train = train_data["target"]
                
                model = CatBoostClassifier(
                    iterations=200,
                    learning_rate=0.05,
                    depth=5,
                    subsample=0.8,
                    colsample_bylevel=0.8,
                    random_seed=42,
                    verbose=False
                )
                
                model.fit(X_train, y_train)
                explainer = shap.TreeExplainer(model)
                
                # validation metrics
                val_cut = current_date - pd.Timedelta(days=val_window_days)
                val_data = df[(df["date"] >= val_cut) & (df["date"] < current_date)].copy()
                
                if len(val_data) >= 50:
                    y_val = val_data["target"]
                    y_val_pred = model.predict_proba(val_data[features])[:, 1]
                    val_auc = roc_auc_score(y_val, y_val_pred)
                    val_logloss = log_loss(y_val, y_val_pred)
                else:
                    val_auc, val_logloss = np.nan, np.nan
                
                y_train_pred = model.predict_proba(X_train)[:, 1]
                train_auc = roc_auc_score(y_train, y_train_pred)
                train_logloss = log_loss(y_train, y_train_pred)
                
                training_log.append({
                    "date": current_date, "train_auc": train_auc, "train_logloss": train_logloss,
                    "val_auc": val_auc, "val_logloss": val_logloss,
                    "train_rows": len(train_data), "val_rows": len(val_data)
                })
                
                print(f"Retrained {current_date.date()} train_auc={train_auc:.3f} val_auc={val_auc if not np.isnan(val_auc) else 'na'}")

        if model is None:
            pm.record_value(current_date, {r["symbol"]: r["close"] for r in today_rows.values()})
            continue

        if today_df.empty:
            pm.record_value(current_date, {r["symbol"]: r["close"] for r in today_rows.values()})
            continue

        probs = model.predict_proba(today_df[features])[:, 1]
        preds = pd.Series(probs, index=today_df["symbol"].values)
        pm.manage_day(current_date, today_rows, preds, prob_threshold=prob_threshold)

    trades_df, hist_df = pm.export_results(out_dir=RESULTS_DIR)
    training_df = pd.DataFrame(training_log)
    return trades_df, hist_df, training_df, model, explainer, features, df

# -------------------------
# Plot & Report (UPDATED with WINNING TRADES REPORT + RSI DISTRIBUTION)
# -------------------------

def plot_and_report(trades_df, hist_df, training_df, model, explainer, features, master_df, trailing_sl_after_t3, out_dir=PLOTS_DIR):
    os.makedirs(out_dir, exist_ok=True)

    # equity & drawdown
    if not hist_df.empty:
        hist_df["date"] = pd.to_datetime(hist_df["date"])
        hist_df.sort_values("date", inplace=True)
        
        plt.figure()
        plt.plot(hist_df["date"], hist_df["portfolio_value"])
        plt.title("Equity Curve")
        plt.savefig(os.path.join(out_dir, "equity_curve.png"))
        plt.close()
        
        hist_df["cummax"] = hist_df["portfolio_value"].cummax()
        hist_df["drawdown"] = hist_df["portfolio_value"]/hist_df["cummax"] - 1.0
        
        plt.figure()
        plt.fill_between(hist_df["date"], hist_df["drawdown"], color="red", alpha=0.6)
        plt.title("Drawdown")
        plt.savefig(os.path.join(out_dir, "drawdown.png"))
        plt.close()

    # trades analysis
    if not trades_df.empty:
        sells = trades_df[trades_df["action"] == "SELL"].copy()
        
        if not sells.empty:
            plt.figure()
            sns.histplot(sells["pnl"].dropna(), bins=40, kde=True)
            plt.title("Trade PnL Distribution")
            plt.savefig(os.path.join(out_dir, "trade_pnl.png"))
            plt.close()
            
            if "holding_days" in sells.columns:
                plt.figure()
                sns.histplot(sells["holding_days"].dropna(), bins=30)
                plt.title("Holding Days Distribution")
                plt.savefig(os.path.join(out_dir, "holding_days.png"))
                plt.close()
            
            wins = (sells["pnl"] > 0).sum()
            losses = (sells["pnl"] <= 0).sum()
            
            plt.figure(figsize=(4,4))
            plt.pie([wins, losses], labels=["Wins","Losses"], autopct="%1.0f%%", colors=["green","red"])
            plt.title("Win/Loss Ratio")
            plt.savefig(os.path.join(out_dir, "win_loss.png"))
            plt.close()

        # NEW: Stop-Loss Loss Report
        print("📋 Generating Stop-Loss Loss Report...")
        sl_loss_report = create_sl_loss_report(trades_df, master_df)
        
        if not sl_loss_report.empty:
            sl_loss_report.to_csv(os.path.join(out_dir, "sl_loss_report.csv"), index=False)
            print(f"✅ Stop-Loss Loss Report saved: {len(sl_loss_report)} losing SL trades found")
            print(f"   Average RSI at entry for SL losses: {sl_loss_report['RSI_14'].mean():.1f}")
            
            # Plot RSI distribution for SL losses
            if not sl_loss_report["RSI_14"].isna().all():
                plt.figure()
                sns.histplot(sl_loss_report["RSI_14"].dropna(), bins=20, kde=True)
                plt.title("RSI Distribution for Stop-Loss Losses")
                plt.xlabel("RSI_14 at Entry")
                plt.ylabel("Frequency")
                plt.savefig(os.path.join(out_dir, "sl_loss_rsi_distribution.png"))
                plt.close()
        else:
            print("ℹ️  No stop-loss loss trades found for report")

        # NEW: Winning Trades Report
        print("📋 Generating Winning Trades Report... (T1 Exit)")
        win_report = create_win_report(trades_df, master_df)
        
        if not win_report.empty:
            win_report.to_csv(os.path.join(out_dir, "win_trade_report.csv"), index=False)
            print(f"✅ Winning Trades Report saved: {len(win_report)} winning trades found")
            print(f"   Average RSI at entry for winning trades: {win_report['RSI_14'].mean():.1f}")
            
            # Plot RSI distribution for winning trades
            if not win_report["RSI_14"].isna().all():
                plt.figure()
                sns.histplot(win_report["RSI_14"].dropna(), bins=20, kde=True)
                plt.title("RSI Distribution for Winning Trades (T1 Hit)")
                plt.xlabel("RSI_14 at Entry")
                plt.ylabel("Frequency")
                plt.savefig(os.path.join(out_dir, "win_trade_rsi_distribution.png"))
                plt.close()
        else:
            print("ℹ️  No winning trades found for report")

        # NEW: Trailing SL specific analysis
        if trailing_sl_after_t3:
            # Plot trailing actions over time
            trail_actions = trades_df[trades_df["action"].isin(["T3_TRAIL", "TRAIL_SL"])].copy()
            if not trail_actions.empty:
                trail_actions["date"] = pd.to_datetime(trail_actions["date"])
                
                plt.figure()
                trail_counts = trail_actions.groupby(trail_actions["date"].dt.to_period("M")).size()
                trail_counts.plot(kind='bar')
                plt.title("Trailing Stop-Loss Actions by Month")
                plt.xlabel("Month")
                plt.ylabel("Number of Trailing Actions")
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "trailing_actions_monthly.png"))
                plt.close()
                
                # Distribution of trailing reasons
                plt.figure()
                reason_counts = trail_actions["reason"].value_counts()
                plt.pie(reason_counts.values, labels=reason_counts.index, autopct="%1.1f%%")
                plt.title("Distribution of Trailing Stop-Loss Reasons")
                plt.savefig(os.path.join(out_dir, "trailing_reasons.png"))
                plt.close()

        # Exit reason analysis
        exit_reasons = sells["reason"].value_counts()
        
        plt.figure(figsize=(10, 6))
        exit_reasons.plot(kind='bar')
        plt.title("Exit Reasons Distribution")
        plt.xlabel("Exit Reason")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "exit_reasons.png"))
        plt.close()

        # Plot allocation multiplier over time
        buys = trades_df[trades_df["action"] == "BUY"].copy()
        if not buys.empty and "allocation_multiplier" in buys.columns:
            buys["date"] = pd.to_datetime(buys["date"])
            
            plt.figure()
            plt.plot(buys["date"], buys["allocation_multiplier"], marker='o', markersize=3)
            plt.title("Allocation Multiplier Over Time")
            plt.ylabel("Allocation Multiplier")
            plt.savefig(os.path.join(out_dir, "allocation_multiplier.png"))
            plt.close()
            
            plt.figure()
            sns.histplot(buys["allocation_multiplier"], bins=20)
            plt.title("Allocation Multiplier Distribution")
            plt.savefig(os.path.join(out_dir, "allocation_multiplier_dist.png"))
            plt.close()

    # training metrics
    if not training_df.empty:
        training_df["date"] = pd.to_datetime(training_df["date"])
        
        plt.figure()
        plt.plot(training_df["date"], training_df["train_auc"], label="train_auc")
        plt.plot(training_df["date"], training_df["val_auc"], label="val_auc")
        plt.legend()
        plt.title("AUC Over Retraining Periods")
        plt.savefig(os.path.join(out_dir, "train_val_auc.png"))
        plt.close()
        
        plt.figure()
        plt.plot(training_df["date"], training_df["train_logloss"], label="train_logloss")
        plt.plot(training_df["date"], training_df["val_logloss"], label="val_logloss")
        plt.legend()
        plt.title("LogLoss Over Retraining Periods")
        plt.savefig(os.path.join(out_dir, "train_val_logloss.png"))
        plt.close()

    # CatBoost feature importance
    try:
        feature_importances = model.get_feature_importance()
        feature_names = model.feature_names_
        
        plt.figure(figsize=(10, 8))
        indices = np.argsort(feature_importances)[::-1][:20]
        plt.title("CatBoost Feature Importance")
        plt.bar(range(len(indices)), feature_importances[indices])
        plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "catboost_fi.png"))
        plt.close()
    except Exception as e:
        print("Feature Importance failed:", e)

    # SHAP
    try:
        sample = master_df.sample(min(2000, len(master_df)), random_state=1)
        Xs = sample[features]
        shap_values = explainer(Xs)
        
        shap.summary_plot(shap_values, Xs, show=False)
        fig = plt.gcf()
        fig.savefig(os.path.join(out_dir, "shap_summary.png"))
        plt.close(fig)
    except Exception as e:
        print("SHAP failed:", e)

    # save CSVs
    trades_df.to_csv(os.path.join(out_dir, "trades.csv"), index=False)
    hist_df.to_csv(os.path.join(out_dir, "portfolio_history.csv"), index=False)
    training_df.to_csv(os.path.join(out_dir, "training_log.csv"), index=False)
    
    print("Plots and CSVs saved to", out_dir)

# -------------------------
# Run
# -------------------------

if __name__ == "__main__":
    # config
    analysis_folder = "analysis_data/ml_v1"
    prob_threshold = 0.15
    retrain_freq_days = 20
    train_window_days = 260
    val_window_days = 60
    horizon = 20
    initial_balance = 100000
    max_positions = 2
    cost_rate = 0.001
    require_flat_before_new = False # false to have position more than 1 in portfolio
    
    # Dynamic allocation parameters
    portfolio_increase_threshold = 0.2  # 20% portfolio growth triggers allocation increase
    allocation_increment = 0.15  # 15% increase in allocation multiplier
    
    # NEW: Trailing stop-loss parameter
    trailing_sl_after_t3 = True  # Set to False for original behavior, stopping all at t3, keep true to ride trend till end
    
    trades_df, hist_df, training_df, model, explainer, features, master_df = run_backtest(
        folder=analysis_folder,
        start_date="2019-01-01",
        initial_balance=initial_balance,
        max_positions=max_positions,
        prob_threshold=prob_threshold,
        retrain_freq_days=retrain_freq_days,
        train_window_days=train_window_days,
        val_window_days=val_window_days,
        horizon=horizon,
        cost_rate=cost_rate,
        require_flat_before_new=require_flat_before_new,
        portfolio_increase_threshold=portfolio_increase_threshold,
        allocation_increment=allocation_increment,
        trailing_sl_after_t3=trailing_sl_after_t3  # NEW PARAMETER
    )
    
    plot_and_report(trades_df, hist_df, training_df, model, explainer, features, master_df, trailing_sl_after_t3, out_dir=PLOTS_DIR)
    
    print("Done. Check results/ and plots/catboost/")
    print(f"Dynamic Allocation: +{allocation_increment*100}% every {portfolio_increase_threshold*100}% portfolio growth")
    print(f"Trailing SL after T3: {'ENABLED - No max hold constraint beyond T3' if trailing_sl_after_t3 else 'DISABLED - Original T3 exit behavior'}")
    
    # Print summary stats
    if not trades_df.empty:
        sells = trades_df[trades_df["action"] == "SELL"]
        if not sells.empty:
            total_pnl = sells["pnl"].sum()
            win_rate = (sells["pnl"] > 0).mean()
            avg_win = sells[sells["pnl"] > 0]["pnl"].mean() if (sells["pnl"] > 0).any() else 0
            avg_loss = sells[sells["pnl"] <= 0]["pnl"].mean() if (sells["pnl"] <= 0).any() else 0
            
            print(f"\n📊 BACKTEST SUMMARY:")
            print(f"Total P&L: {total_pnl:,.2f}")
            print(f"Win Rate: {win_rate:.1%}")
            print(f"Average Win: {avg_win:,.2f}")
            print(f"Average Loss: {avg_loss:,.2f}")
            print(f"Total Trades: {len(sells)}")
            
            if trailing_sl_after_t3:
                trail_actions = trades_df[trades_df["action"].isin(["T3_TRAIL", "TRAIL_SL"])]
                print(f"Trailing Actions: {len(trail_actions)}")
            
            # NEW: Stop-Loss Loss Summary
            sl_losses = trades_df[(trades_df["action"] == "SELL") &
                                 (trades_df["reason"] == "STOPLOSS") &
                                 (trades_df["pnl"] < 0)]
            if not sl_losses.empty:
                print(f"📉 Stop-Loss Losses: {len(sl_losses)} trades")
                print(f"   Total SL Loss Amount: {sl_losses['pnl'].sum():,.2f}")
                print(f"   Average SL Loss: {sl_losses['pnl'].mean():,.2f}")
            
            # NEW: Winning Trades Summary
            win_trades = trades_df[(trades_df["action"] == "SELL") &
                                  (trades_df["reason"].str.startswith("T1_EXIT", na=False))]
            if not win_trades.empty:
                print(f"🎯 Winning Trades (T1 Hit): {len(win_trades)} trades")
                print(f"   Total Win Amount: {win_trades['pnl'].sum():,.2f}")
                print(f"   Average Win: {win_trades['pnl'].mean():,.2f}")
