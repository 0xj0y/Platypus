# main.py
"""
Swing backtest with:
 - Entry only when darvas_breakout_up == 1 AND model prob > prob_threshold
 - Partial exits at ATR milestones:
     T1 = +1 ATR -> exit 30% -> trail SL to entry
     T2 = +2 ATR -> exit 30% -> trail SL to entry + 1 ATR
     T3 = +3 ATR -> exit remaining (40%)
 - Initial SL = entry - 1.5 ATR (active immediately)
 - Max hold = 20 days (force exit remaining)
 - No new entries allowed until portfolio is fully flat (configurable)
 - Labels: 1 if TP (3 ATR) touched before SL within horizon (keeps consistency)
Outputs in results/ and plots/v1/
"""

import os
import glob
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import roc_auc_score, log_loss
import shap
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
plt.rcParams["figure.figsize"] = (10, 5)

RESULTS_DIR = "results"
PLOTS_DIR = "plots/v1"
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
                # check high/low of day k
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
# Portfolio Manager with Partial Exits + Trailing
# -------------------------
class SwingPortfolioManager:
    def __init__(self,
                 initial_balance=100000,
                 max_positions=5,
                 min_hold_days=7,
                 max_hold_days=20,
                 cost_rate=0.001,
                 require_flat_before_new=True):
        """
        require_flat_before_new: if True, new entries are allowed ONLY when portfolio is completely flat.
        If False, opens up to max_positions concurrently (each gets its own partial exit schedule).
        """
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_positions = max_positions
        self.min_hold_days = min_hold_days
        self.max_hold_days = max_hold_days
        self.cost_rate = cost_rate
        self.require_flat_before_new = require_flat_before_new

        # positions: dict symbol -> position dict
        # pos dict contains:
        #   shares (int), entry_price, entry_date (pd.Timestamp), atr, stop_loss, targets [t1,t2,t3],
        #   milestones_hit (0..3)
        self.positions = {}
        self.trade_log = []         # list of dicts (BUY/SELL rows)
        self.portfolio_history = [] # list of dicts

    def _has_partial_outstanding(self):
        # Returns True if any position exists (i.e., not fully exited)
        return len(self.positions) > 0

    def _buy(self, date, symbol, row, capital_per_slot):
        price = row["close"]
        atr = row["ATR_14"]
        if pd.isna(price) or pd.isna(atr) or price <= 0:
            return False

        shares = int(capital_per_slot // price)
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
        self.positions[symbol] = {
            "shares": shares,
            "entry_price": price,
            "entry_date": pd.to_datetime(date),
            "atr": atr,
            "stop_loss": stop_loss,
            "targets": [t1, t2, t3],
            "milestones_hit": 0
        }

        # log buy
        self.trade_log.append({
            "date": date, "action": "BUY", "symbol": symbol,
            "shares": shares, "entry_price": price, "exit_price": np.nan,
            "pnl": np.nan, "reason": "ENTRY", "holding_days": np.nan
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
        # ensure at least 1 share if fraction > 0 and shares_current > 0
        if shares_to_sell <= 0 and shares_current > 0 and fraction > 0:
            shares_to_sell = 1

        if shares_to_sell <= 0:
            return

        proceeds = shares_to_sell * exit_price
        cost = proceeds * self.cost_rate
        self.balance += (proceeds - cost)

        pnl = (exit_price - pos["entry_price"]) * shares_to_sell - cost

        # append sell log
        self.trade_log.append({
            "date": date, "action": "SELL", "symbol": symbol,
            "shares": shares_to_sell, "entry_price": pos["entry_price"], "exit_price": exit_price,
            "pnl": pnl, "reason": reason, "holding_days": (pd.to_datetime(date) - pos["entry_date"]).days
        })

        # reduce remaining shares
        pos["shares"] = pos["shares"] - shares_to_sell
        if pos["shares"] <= 0:
            # fully closed
            self.positions.pop(symbol, None)

    def manage_day(self, date, today_rows_dict, predictions, prob_threshold=0.55):
        """
        Manage existing positions (partial exits & trailing) and open new ones.
        today_rows_dict: dict symbol -> row with at least keys ['close','high','low','ATR_14','darvas_breakout_up']
        predictions: pd.Series indexed by symbol with model probabilities
        """

        # ---------- EXITS: iterate over copy because we may modify positions ----------
        for symbol, pos in list(self.positions.items()):
            row = today_rows_dict.get(symbol)
            if row is None:
                # can't evaluate this position today, skip
                continue

            high = row.get("high", row.get("close"))
            low = row.get("low", row.get("close"))
            close = row["close"]
            days_held = (pd.to_datetime(date) - pos["entry_date"]).days

            # 1) STOP-LOSS check (immediate always)
            if low <= pos["stop_loss"]:
                # sell all remaining at stop price (use stop price to be conservative)
                exit_price = pos["stop_loss"]
                self._sell_fraction(date, symbol, exit_price, 1.0, reason="STOPLOSS")
                continue  # pos removed, move on

            # 2) Check target hits this day (use day's high)
            # We'll process targets in ascending order; if high surpasses multiple targets, we apply sequential sells
            targets = pos["targets"]
            milestones = pos["milestones_hit"]
            # fractions: T1 30%, T2 30%, T3 remaining (sell all remaining)
            fractions = [0.30, 0.30, 1.0]

            # For each target index > milestones_hit-1, if high >= target -> execute
            for t_idx in range(milestones, len(targets)):
                target_price = targets[t_idx]
                if high >= target_price:
                    # this target was reached today. Do partial sell
                    if t_idx < 2:
                        frac = fractions[t_idx]
                        # sell fraction
                        self._sell_fraction(date, symbol, target_price, frac, reason=f"T{t_idx+1}_EXIT")
                        # update milestones and trailing stop based on rules
                        if t_idx == 0:
                            # after T1: trail stop to entry (breakeven)
                            pos_local = self.positions.get(symbol)
                            if pos_local is not None:
                                pos_local["stop_loss"] = pos_local["entry_price"]
                                pos_local["milestones_hit"] = 1
                        elif t_idx == 1:
                            pos_local = self.positions.get(symbol)
                            if pos_local is not None:
                                # trail stop to entry + 1 ATR
                                pos_local["stop_loss"] = pos_local["entry_price"] + 1.0 * pos_local["atr"]
                                pos_local["milestones_hit"] = 2
                        # continue loop to see if higher milestones also hit today
                    else:
                        # final milestone -> sell all remaining
                        self._sell_fraction(date, symbol, target_price, 1.0, reason="T3_EXIT")
                        # milestones_hit becomes 3 but position is removed inside _sell_fraction
                    # refresh pos (it may be removed)
                    if symbol not in self.positions:
                        break
                else:
                    # target not reached, no further higher targets reached either
                    break

            # 3) If still holding and days_held >= max_hold_days => force exit at close
            if symbol in self.positions:
                pos2 = self.positions[symbol]
                days_held = (pd.to_datetime(date) - pos2["entry_date"]).days
                if days_held >= self.max_hold_days:
                    # force exit remaining at close
                    self._sell_fraction(date, symbol, close, 1.0, reason="TIMEEXIT")

        # ---------- ENTRIES ----------
        # If require_flat_before_new: don't open if any position exists
        if self.require_flat_before_new and len(self.positions) > 0:
            # simply record portfolio value and return
            self.record_value(date, {s: today_rows_dict[s]["close"] for s in today_rows_dict})
            return

        # Otherwise allow up to max_positions (equal allocation)
        preds_today = predictions.loc[predictions.index.isin(list(today_rows_dict.keys()))]
        if preds_today.empty:
            self.record_value(date, {s: today_rows_dict[s]["close"] for s in today_rows_dict})
            return

        # Filter by darvas breakout present today
        breakout_symbols = [s for s, r in today_rows_dict.items() if r.get("darvas_breakout_up", 0) == 1]
        if not breakout_symbols:
            self.record_value(date, {s: today_rows_dict[s]["close"] for s in today_rows_dict})
            return

        eligible = preds_today.loc[preds_today.index.isin(breakout_symbols)]
        eligible = eligible[eligible > prob_threshold].sort_values(ascending=False)

        if eligible.empty:
            self.record_value(date, {s: today_rows_dict[s]["close"] for s in today_rows_dict})
            return

        # If require_flat_before_new was False we may open multiple positions up to max_positions.
        # Calculate open slots and capital per slot.
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

        # record end-of-day total value
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
# Backtest Runner
# -------------------------
def run_backtest(folder,
                 start_date="2019-01-01",
                 initial_balance=100000,
                 max_positions=5,
                 prob_threshold=0.55,
                 retrain_freq_days=20,
                 train_window_days=365 * 2,
                 val_window_days=60,
                 horizon=20,
                 cost_rate=0.001,
                 require_flat_before_new=True):
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
        require_flat_before_new=require_flat_before_new
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
                model = XGBClassifier(
                    objective="binary:logistic", eval_metric="logloss",
                    n_estimators=200, learning_rate=0.05,
                    max_depth=5, subsample=0.8, colsample_bytree=0.8,
                    random_state=42, use_label_encoder=False
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

        # if model not ready still record portfolio value
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
# Plot & Report (basic)
# -------------------------
def plot_and_report(trades_df, hist_df, training_df, model, explainer, features, master_df, out_dir=PLOTS_DIR):
    os.makedirs(out_dir, exist_ok=True)

    # equity & drawdown
    if not hist_df.empty:
        hist_df["date"] = pd.to_datetime(hist_df["date"])
        hist_df.sort_values("date", inplace=True)
        plt.figure(); plt.plot(hist_df["date"], hist_df["portfolio_value"]); plt.title("Equity Curve")
        plt.savefig(os.path.join(out_dir, "equity_curve.png")); plt.close()

        hist_df["cummax"] = hist_df["portfolio_value"].cummax()
        hist_df["drawdown"] = hist_df["portfolio_value"]/hist_df["cummax"] - 1.0
        plt.figure(); plt.fill_between(hist_df["date"], hist_df["drawdown"], color="red", alpha=0.6); plt.title("Drawdown")
        plt.savefig(os.path.join(out_dir, "drawdown.png")); plt.close()

    # trades analysis
    if not trades_df.empty:
        sells = trades_df[trades_df["action"] == "SELL"].copy()
        if not sells.empty:
            plt.figure(); sns.histplot(sells["pnl"].dropna(), bins=40, kde=True); plt.title("Trade PnL Dist"); plt.savefig(os.path.join(out_dir, "trade_pnl.png")); plt.close()
            if "holding_days" in sells.columns:
                plt.figure(); sns.histplot(sells["holding_days"].dropna(), bins=30); plt.title("Holding Days Dist"); plt.savefig(os.path.join(out_dir, "holding_days.png")); plt.close()
            wins = (sells["pnl"] > 0).sum(); losses = (sells["pnl"] <= 0).sum()
            plt.figure(figsize=(4,4)); plt.pie([wins, losses], labels=["Wins","Losses"], autopct="%1.0f%%", colors=["green","red"]); plt.title("Win/Loss"); plt.savefig(os.path.join(out_dir, "win_loss.png")); plt.close()

    # training metrics
    if not training_df.empty:
        training_df["date"] = pd.to_datetime(training_df["date"])
        plt.figure(); plt.plot(training_df["date"], training_df["train_auc"], label="train_auc"); plt.plot(training_df["date"], training_df["val_auc"], label="val_auc"); plt.legend(); plt.title("AUC over retrains"); plt.savefig(os.path.join(out_dir, "train_val_auc.png")); plt.close()
        plt.figure(); plt.plot(training_df["date"], training_df["train_logloss"], label="train_logloss"); plt.plot(training_df["date"], training_df["val_logloss"], label="val_logloss"); plt.legend(); plt.title("LogLoss over retrains"); plt.savefig(os.path.join(out_dir, "train_val_logloss.png")); plt.close()

    # feature importance
    try:
        plt.figure(); plot_importance(model, max_num_features=20, importance_type="gain"); plt.title("XGB feature importance"); plt.savefig(os.path.join(out_dir, "xgb_fi.png")); plt.close()
    except Exception as e:
        print("FI failed:", e)

    # SHAP
    try:
        sample = master_df.sample(min(2000, len(master_df)), random_state=1)
        Xs = sample[features]
        shap_values = explainer(Xs)
        shap.summary_plot(shap_values, Xs, show=False)
        fig = plt.gcf(); fig.savefig(os.path.join(out_dir, "shap_summary.png")); plt.close(fig)
    except Exception as e:
        print("SHAP failed:", e)

    # save CSVs into out_dir
    trades_df.to_csv(os.path.join(out_dir, "trades.csv"), index=False)
    hist_df.to_csv(os.path.join(out_dir, "portfolio_history.csv"), index=False)
    training_df.to_csv(os.path.join(out_dir, "training_log.csv"), index=False)

    print("Plots and CSVs saved to", out_dir)


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    # config - flip require_flat_before_new to False if you want concurrency up to max_positions
    analysis_folder = "analysis_data/ml_v1"
    prob_threshold = 0.1
    retrain_freq_days = 20
    train_window_days = 260
    val_window_days = 60
    horizon = 20
    initial_balance = 100000
    max_positions = 5
    cost_rate = 0.001
    require_flat_before_new = False

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
        require_flat_before_new=require_flat_before_new
    )

    plot_and_report(trades_df, hist_df, training_df, model, explainer, features, master_df, out_dir=PLOTS_DIR)
    print("Done. Check results/ and plots/v1/")
