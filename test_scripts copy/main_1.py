# main.py

"""
Swing backtest with Darvas-breakout entry + ML filter + SL/TP + min-hold (7) + max-hold (20)
Uses MQL5 article features: ADX, ADX_Wilder, DeMarker, RSI, RVI, Stochastic, and Darvas-specific features
Outputs: trades.csv, portfolio_history.csv, plots in plots/v1/
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import roc_auc_score, log_loss
import shap
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
plt.rcParams["figure.figsize"] = (10, 5)

PLOTS_DIR = "plots/v2"
os.makedirs(PLOTS_DIR, exist_ok=True)

# -------------------------
# Portfolio Manager
# -------------------------

class SwingPortfolioManager:
    def __init__(self,
                 initial_balance=100000,
                 max_positions=5,
                 min_hold_days=7,
                 max_hold_days=20,
                 cost_rate=0.001):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_positions = max_positions
        self.min_hold_days = min_hold_days
        self.max_hold_days = max_hold_days
        self.cost_rate = cost_rate
        
        # positions: dict symbol -> position dict
        # position dict: {shares, entry_price, entry_date, stop_loss, target}
        self.positions = {}
        self.trade_log = []  # dict entries
        self.portfolio_history = []
    
    def _can_enter(self):
        return len(self.positions) < self.max_positions
    
    def _buy(self, date, symbol, row, capital_per_slot):
        price = row["close"]
        
        # Use normalized ATR equivalent for stop loss calculation
        # Since we don't have ATR_14, we'll estimate it from high-low range
        atr_estimate = (row["high"] - row["low"]) * 0.7  # Rough ATR estimate
        
        if pd.isna(price) or pd.isna(atr_estimate) or price <= 0:
            return False
        
        shares = int(capital_per_slot // price)
        if shares <= 0:
            return False
        
        stop_loss = price - 2 * atr_estimate
        target = price + 1.5 * atr_estimate
        
        cost = shares * price * self.cost_rate
        total_cash = shares * price + cost
        
        if self.balance < total_cash:
            return False
        
        self.balance -= total_cash
        
        self.positions[symbol] = {
            "shares": shares,
            "entry_price": price,
            "entry_date": date,
            "stop_loss": stop_loss,
            "target": target,
        }
        
        self.trade_log.append({
            "date": date, "action": "BUY", "symbol": symbol,
            "shares": shares, "entry_price": price, "exit_price": np.nan,
            "pnl": np.nan, "reason": "", "holding_days": np.nan
        })
        
        return True
    
    def _sell(self, date, symbol, exit_price, reason):
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return
        
        shares = pos["shares"]
        entry_price = pos["entry_price"]
        holding_days = (date - pos["entry_date"]).days
        
        proceeds = shares * exit_price
        cost = proceeds * self.cost_rate
        self.balance += proceeds - cost
        
        pnl = (exit_price - entry_price) * shares - cost  # include sell cost only here (we already deducted buy cost)
        
        self.trade_log.append({
            "date": date, "action": "SELL", "symbol": symbol,
            "shares": shares, "entry_price": entry_price, "exit_price": exit_price,
            "pnl": pnl, "reason": reason, "holding_days": holding_days
        })
    
    def manage_day(self, date, today_rows_dict, predictions, prob_threshold=0.55):
        """
        today_rows_dict: dict symbol -> row-dict with keys including 'close' and 'darvas_breakout_up'
        predictions: pd.Series indexed by symbol with model probability
        """
        
        # check exits for all positions (SL/TP immediate, or time exit if reached)
        to_exit = []
        
        for symbol, pos in list(self.positions.items()):
            # if missing today data, skip
            row = today_rows_dict.get(symbol)
            if row is None:
                continue
            
            price = row["close"]
            days_held = (date - pos["entry_date"]).days
            
            # Immediate SL/TP check (always active)
            if price <= pos["stop_loss"]:
                to_exit.append((symbol, price, "STOPLOSS"))
            elif price >= pos["target"]:
                to_exit.append((symbol, price, "TARGET"))
            else:
                # If neither hit, enforce max hold exit
                if days_held >= self.max_hold_days:
                    to_exit.append((symbol, price, "TIMEEXIT"))
                # otherwise do nothing (including model-based exit is disabled until min_hold_days passes)
        
        # Process exits
        for symbol, price, reason in to_exit:
            self._sell(date, symbol, price, reason)
        
        # --- 2) Entries: Only consider today's Darvas breakouts that pass ML filter
        open_slots = self.max_positions - len(self.positions)
        if open_slots <= 0:
            # no capacity
            self.record_value(date, {s: today_rows_dict[s]["close"] for s in today_rows_dict})
            return
        
        # Build candidates: must have darvas_breakout_up == 1 AND prob > threshold
        # predictions may include symbols not in today_rows_dict; intersect
        preds_today = predictions.loc[predictions.index.isin(list(today_rows_dict.keys()))]
        if preds_today.empty:
            self.record_value(date, {s: today_rows_dict[s]["close"] for s in today_rows_dict})
            return
        
        # filter breakouts
        breakout_symbols = [s for s, r in today_rows_dict.items() if r.get("darvas_breakout_up", 0) == 1]
        if not breakout_symbols:
            self.record_value(date, {s: today_rows_dict[s]["close"] for s in today_rows_dict})
            return
        
        eligible = preds_today.loc[preds_today.index.isin(breakout_symbols)]
        eligible = eligible[eligible > prob_threshold].sort_values(ascending=False)
        
        if eligible.empty:
            self.record_value(date, {s: today_rows_dict[s]["close"] for s in today_rows_dict})
            return
        
        # capital per slot = equal allocation of available cash to remaining open slots
        # note: if multiple buys, capital_per_slot is recalculated after each successful buy
        for symbol, prob in eligible.items():
            if open_slots <= 0:
                break
            if not self._can_enter():
                break
            if symbol in self.positions:
                continue  # already holding
            
            capital_per_slot = self.balance / open_slots
            success = self._buy(date, symbol, today_rows_dict[symbol], capital_per_slot)
            if success:
                open_slots -= 1
        
        # record portfolio value end of day
        self.record_value(date, {s: today_rows_dict[s]["close"] for s in today_rows_dict})
    
    def record_value(self, date, current_prices):
        holdings = sum(pos["shares"] * current_prices.get(sym, pos["entry_price"]) for sym, pos in self.positions.items())
        total = self.balance + holdings
        self.portfolio_history.append({"date": date, "portfolio_value": total})
    
    def export_results(self, out_dir="results"):
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
    
    dfs = []
    for fp in csv_files:
        try:
            df = pd.read_csv(fp, parse_dates=["date"])
            # ensure required columns exist
            dfs.append(df)
        except Exception as e:
            print("Skipping", fp, "because", e)
    
    master = pd.concat(dfs, ignore_index=True)
    master["date"] = pd.to_datetime(master["date"])
    master = master[master["date"] >= pd.to_datetime(start_date_str)].copy()
    master.sort_values(["symbol", "date"], inplace=True)
    master.reset_index(drop=True, inplace=True)
    
    return master

def engineer_features(df, horizon=20, threshold=0.07):
    """
    Creates features based on MQL5 article and target:
    target = 1 if price rises >= threshold within next horizon days
    """
    
    # Features list based on MQL5 article (12 features)
    features = [
        "ADX_14",           # Average Directional Movement Index
        "ADX_Wilder_14",    # Average Directional Movement Index by Welles Wilder
        "DeMarker_14",      # DeMarker
        "RSI_14",           # Relative Strength Index
        "RVI_10",           # Relative Vigor Index
        "Stochastic",       # Stochastic Oscillator
        "Stationary",       # Normalized return (current bar): 1000*(close-open)/close
        "BoxSize",          # Box size: 1000*(high-low)/close
        "Stationary2",      # Normalized return (2 bars ago)
        "Stationary3",      # Normalized return (3 bars ago)
        "DistanceHigh",     # Distance from close to darvas high: 1000*(close-high)/close
        "DistanceLow"       # Distance from close to darvas low: 1000*(close-low)/close
    ]
    
    # Add darvas_breakout_up for entry signal (not a feature for ML model)
    required_columns = features + ["darvas_breakout_up"]
    
    # Check if all required features exist
    missing_features = [f for f in required_columns if f not in df.columns]
    if missing_features:
        print(f"⚠️ Missing features: {missing_features}")
        print("💡 Make sure to run apply_technical_indicators first")
        return df, []
    
    # compute future max properly per symbol
    future_max = df.groupby("symbol")["close"].shift(-1).rolling(window=horizon, min_periods=1).max()
    df["target"] = ((future_max - df["close"]) / df["close"] >= threshold).astype(int)
    
    # drop rows missing indicators
    df.dropna(subset=features + ["target"], inplace=True)
    
    return df, features

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
                cost_rate=0.001):
    """
    Runs the whole backtest pipeline and returns portfolio manager + training metrics.
    """
    
    master = create_master_dataframe(folder, start_date)
    df, features = engineer_features(master)
    
    # prepare dates
    unique_days = sorted(df["date"].unique())
    if not unique_days:
        raise ValueError("No trading dates found after filtering.")
    
    portfolio = SwingPortfolioManager(initial_balance=initial_balance,
                                    max_positions=max_positions,
                                    min_hold_days=7, max_hold_days=20,
                                    cost_rate=cost_rate)
    
    model = None
    explainer = None
    training_log = []  # records train/val metrics at each retrain
    
    # Convert df to group-by-date dictionary for fast per-day access
    # day_map[date] -> dataframe slice for that day
    day_map = {d: df[df["date"] == d].copy() for d in unique_days}
    
    for i, current_date in enumerate(unique_days):
        today_df = day_map[current_date]
        
        # build dict symbol->row dict for today's rows (used for fast access)
        today_rows = {row["symbol"]: row for _, row in today_df.iterrows()}
        
        # decide if we retrain (first run and every retrain_freq_days)
        if model is None or i % retrain_freq_days == 0:
            # define training window: use last train_window_days before current_date
            cutoff = current_date - pd.Timedelta(days=train_window_days)
            train_data = df[(df["date"] >= cutoff) & (df["date"] < current_date)].copy()
            
            if len(train_data) >= 200:
                X_train = train_data[features]
                y_train = train_data["target"]
                
                model = XGBClassifier(
                    objective="binary:logistic", eval_metric="logloss",
                    n_estimators=400, learning_rate=0.04,
                    max_depth=5, subsample=0.8, colsample_bytree=0.8,
                    random_state=42, use_label_encoder=False
                )
                
                model.fit(X_train, y_train)
                explainer = shap.TreeExplainer(model)
                
                # validation window (recent val_window_days before current_date)
                val_cut = current_date - pd.Timedelta(days=val_window_days)
                val_data = df[(df["date"] >= val_cut) & (df["date"] < current_date)].copy()
                
                if len(val_data) >= 50:
                    y_val = val_data["target"]
                    y_val_pred = model.predict_proba(val_data[features])[:, 1]
                    val_auc = roc_auc_score(y_val, y_val_pred)
                    val_logloss = log_loss(y_val, y_val_pred)
                else:
                    val_auc, val_logloss = np.nan, np.nan
                
                # train metrics
                y_train_pred = model.predict_proba(X_train)[:, 1]
                train_auc = roc_auc_score(y_train, y_train_pred)
                train_logloss = log_loss(y_train, y_train_pred)
                
                training_log.append({
                    "date": current_date, "train_auc": train_auc, "train_logloss": train_logloss,
                    "val_auc": val_auc, "val_logloss": val_logloss,
                    "train_rows": len(train_data), "val_rows": len(val_data)
                })
                
                # print brief
                print(f"Retrained on {current_date.date()} | trainauc={train_auc:.3f} valauc={val_auc if not np.isnan(val_auc) else 'na'}")
        
        # if model not ready, skip day (we still record portfolio value)
        if model is None:
            portfolio.record_value(current_date, {r["symbol"]: r["close"] for r in today_rows.values()})
            continue
        
        # Prepare predictions for symbols available today
        X_today = today_df[features]
        if X_today.empty:
            portfolio.record_value(current_date, {r["symbol"]: r["close"] for r in today_rows.values()})
            continue
        
        probs = model.predict_proba(X_today)[:, 1]
        preds = pd.Series(probs, index=today_df["symbol"].values)
        
        # Manage trades (entries only for darvas breakout && prob > threshold)
        portfolio.manage_day(current_date, today_rows, preds, prob_threshold=prob_threshold)
    
    # finalize
    trades_df, history_df = portfolio.export_results(out_dir="results")
    training_df = pd.DataFrame(training_log)
    
    return trades_df, history_df, training_df, model, explainer, features, df

# -------------------------
# Reporting + Plots
# -------------------------

def plot_and_report(trades_df, history_df, training_df, model, explainer, features, master_df, out_dir="plots/v2"):
    os.makedirs(out_dir, exist_ok=True)
    
    # equity curve
    if history_df.empty:
        print("No portfolio history to plot.")
    else:
        history_df["date"] = pd.to_datetime(history_df["date"])
        history_df.sort_values("date", inplace=True)
        
        plt.figure()
        plt.plot(history_df["date"], history_df["portfolio_value"])
        plt.title("Equity Curve")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "equity_curve.png"))
        plt.close()
        
        # drawdown
        history_df["cummax"] = history_df["portfolio_value"].cummax()
        history_df["drawdown"] = history_df["portfolio_value"] / history_df["cummax"] - 1.0
        
        plt.figure()
        plt.fill_between(history_df["date"], history_df["drawdown"], color="red", alpha=0.6)
        plt.title("Drawdown")
        plt.savefig(os.path.join(out_dir, "drawdown.png"))
        plt.close()
    
    # trades PnL and holding days
    if not trades_df.empty:
        sells = trades_df[trades_df["action"] == "SELL"].copy()
        if not sells.empty:
            plt.figure()
            sns.histplot(sells["pnl"].dropna(), bins=40, kde=True)
            plt.title("Trade PnL Distribution (SELLs)")
            plt.savefig(os.path.join(out_dir, "trade_pnl_distribution.png"))
            plt.close()
            
            plt.figure()
            sns.histplot(sells["holding_days"].dropna(), bins=30)
            plt.title("Holding Days Distribution (SELLs)")
            plt.savefig(os.path.join(out_dir, "holding_days_distribution.png"))
            plt.close()
            
            wins = (sells["pnl"] > 0).sum()
            losses = (sells["pnl"] <= 0).sum()
            
            plt.figure(figsize=(4, 4))
            plt.pie([wins, losses], labels=["Wins", "Losses"], autopct="%1.0f%%", colors=["green", "red"])
            plt.title("Win vs Loss")
            plt.savefig(os.path.join(out_dir, "win_loss_pie.png"))
            plt.close()
    
    # training metrics (overfitting check)
    if not training_df.empty:
        training_df["date"] = pd.to_datetime(training_df["date"])
        
        plt.figure()
        plt.plot(training_df["date"], training_df["train_auc"], label="train_auc")
        plt.plot(training_df["date"], training_df["val_auc"], label="val_auc")
        plt.legend(); plt.title("AUC over retrains")
        plt.savefig(os.path.join(out_dir, "train_val_auc.png")); plt.close()
        
        plt.figure()
        plt.plot(training_df["date"], training_df["train_logloss"], label="train_logloss")
        plt.plot(training_df["date"], training_df["val_logloss"], label="val_logloss")
        plt.legend(); plt.title("LogLoss over retrains")
        plt.savefig(os.path.join(out_dir, "train_val_logloss.png")); plt.close()
    
    # feature importance (XGBoost)
    try:
        plt.figure()
        plot_importance(model, max_num_features=20, importance_type="gain")
        plt.title("Feature importance (gain)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "xgb_feature_importance.png"))
        plt.close()
    except Exception as e:
        print("Feature importance plot failed:", e)
    
    # SHAP global summary
    try:
        sample = master_df.sample(min(2000, len(master_df)), random_state=1)
        Xs = sample[features]
        shap_values = explainer(Xs)
        
        shap.summary_plot(shap_values, Xs, show=False)
        fig = plt.gcf()
        fig.savefig(os.path.join(out_dir, "shap_summary.png"))
        plt.close(fig)
    except Exception as e:
        print("SHAP plotting failed:", e)
    
    # Save CSVs to out_dir
    trades_df.to_csv(os.path.join(out_dir, "trades.csv"), index=False)
    history_df.to_csv(os.path.join(out_dir, "portfolio_history.csv"), index=False)
    training_df.to_csv(os.path.join(out_dir, "training_log.csv"), index=False)
    
    print(f"Saved plots and csvs to {out_dir}")

# -------------------------
# Run
# -------------------------

if __name__ == "__main__":
    # Config
    analysis_folder = "analysis_data/ml_v3"
    prob_threshold = 0.1
    retrain_freq_days = 20
    train_window_days = 260
    val_window_days = 60
    initial_balance = 100000
    max_positions = 5
    cost_rate = 0.001
    
    # Run
    trades_df, history_df, training_df, model, explainer, features, master_df = run_backtest(
        folder=analysis_folder,
        start_date="2019-01-01",
        initial_balance=initial_balance,
        max_positions=max_positions,
        prob_threshold=prob_threshold,
        retrain_freq_days=retrain_freq_days,
        train_window_days=train_window_days,
        val_window_days=val_window_days,
        cost_rate=cost_rate
    )
    
    # Reporting & plots
    plot_and_report(trades_df, history_df, training_df, model, explainer, features, master_df, out_dir=PLOTS_DIR)
    
    print("✅ Backtest finished. Check results/ and plots/v1/ for outputs.")
    print(f"📊 Features used from MQL5 article: {features}")
