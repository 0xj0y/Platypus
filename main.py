# main_catboost.py

"""
Swing backtest with:

- Entry only when darvas_breakout_up == 1 AND model prob > prob_threshold
- INCREASING ALLOCATION: Increase allocation by 15% whenever portfolio value increases by 20%
- Partial exits at ATR milestones:
  T1 = +1 ATR -> exit 30% -> trail SL to entry
  T2 = +2 ATR -> exit 30% -> trail SL to entry + 1 ATR
  T3 = +3 ATR -> exit remaining (40%) OR trail SL if trailing_sl_after_t3=True
- Initial SL = entry - 1 ATR (active immediately)
- Max hold = 20 days (force exit remaining, unless trailing beyond T3)
- Optional trailing SL after T3 - keeps position open and trails SL upward
- Labels: 1 if TP (3 ATR) touched before SL within horizon (keeps consistency)
- NEW: Enhanced Stop-Loss Loss Report with volume/ATR metrics
- NEW: Enhanced Winning Trades Report with volume/ATR metrics

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
import sys
from swing_portfolio_manager import SwingPortfolioManager

warnings.filterwarnings("ignore", category=UserWarning, module="catboost")
plt.rcParams["figure.figsize"] = (10, 5)

RESULTS_DIR = "results"
PLOTS_DIR = "plots/catboost/v13"
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
            sl = entry - 1 * atr
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
# ENHANCED: Stop-Loss Loss Report Function with Extended Metrics
# -------------------------
def create_sl_loss_report(trades_df, master_df=None):
    """
    Report for trades that exited via STOPLOSS with negative PnL,
    using entry-date metrics (RSI_14, VOL_20, NORM_ATR, volume_ratio,
    volume_percentile, ema_50_200_diff, ema_flag).
    Includes total_trade_pnl for consistency with win_report.
    """
    td = trades_df.copy()
    td["date"] = pd.to_datetime(td["date"])
    td = td.sort_values(["symbol", "date"]).copy()

    # Per-trade id: cumulative BUYs per symbol
    td["is_buy"] = (td["action"] == "BUY").astype(int)
    td["trade_id"] = td.groupby("symbol")["is_buy"].cumsum()

    # STOPLOSS SELL legs that lost money
    sl_legs = td[
        (td["action"] == "SELL") &
        (td["reason"] == "STOPLOSS") &
        (td["pnl"] < 0) &
        (td["trade_id"] > 0)
    ].copy()

    if sl_legs.empty:
        print("No stop-loss loss trades found.")
        return pd.DataFrame(columns=[
            "date","symbol","trade_id","entry_date","entry_price","shares",
            "total_trade_pnl","RSI_14","VOL_20","NORM_ATR","volume_ratio",
            "volume_percentile","ema_50_200_diff","ema_flag"
        ])

    # Derive entry_date per trade_id (first BUY date in that trade)
    buys = td[td["action"] == "BUY"][["symbol","trade_id","date","entry_price","shares"]]
    entry_per_trade = buys.groupby(["symbol","trade_id"], as_index=False)["date"].min().rename(columns={"date":"entry_date"})
    sl_legs = sl_legs.merge(entry_per_trade, on=["symbol","trade_id"], how="left")

    # Attach total_trade_pnl per trade
    pnl_per_trade = td[td["action"] == "SELL"].groupby(["symbol","trade_id"], as_index=False)["pnl"].sum()
    pnl_per_trade.rename(columns={"pnl": "total_trade_pnl"}, inplace=True)
    sl_legs = sl_legs.merge(pnl_per_trade, on=["symbol","trade_id"], how="left")

    # Attach entry-date metrics from master_df
    if master_df is not None:
        m = master_df.copy()
        m["date"] = pd.to_datetime(m["date"])

        def metrics_at_entry(row):
            df_sym = m[m["symbol"] == row["symbol"]]
            if df_sym.empty or pd.isna(row["entry_date"]):
                return pd.Series([np.nan]*6)
            closest_row = df_sym.iloc[(df_sym["date"] - row["entry_date"]).abs().argmin()]
            return pd.Series([
                closest_row.get("RSI_14", np.nan),
                closest_row.get("VOL_20", np.nan),
                closest_row.get("NORM_ATR", np.nan),
                closest_row.get("volume_ratio", np.nan),
                closest_row.get("volume_percentile", np.nan),
                closest_row.get("ema_50_200_diff", np.nan),
                closest_row.get("NORM_BB", np.nan),
                closest_row.get("BB_Squeeze", np.nan),
            ])

        sl_legs[["RSI_14","VOL_20","NORM_ATR","volume_ratio",
                 "volume_percentile","ema_50_200_diff","NORM_BB", "BB_Squeeze"]] = \
            sl_legs.apply(metrics_at_entry, axis=1)
    else:
        sl_legs["RSI_14"] = np.nan
        sl_legs["VOL_20"] = np.nan
        sl_legs["NORM_ATR"] = np.nan
        sl_legs["volume_ratio"] = np.nan
        sl_legs["volume_percentile"] = np.nan
        sl_legs["ema_50_200_diff"] = np.nan

    # Add ema_flag (+1 if positive, -1 if negative, 0 if zero, NaN otherwise)
    def flag_func(val):
        if pd.isna(val):
            return np.nan
        elif val > 0:
            return 1
        elif val < 0:
            return -1
        else:
            return 0

    sl_legs["ema_flag"] = sl_legs["ema_50_200_diff"].apply(flag_func)

    # Final report
    report = sl_legs[[
        "date","symbol","trade_id","entry_date","entry_price","shares",
        "total_trade_pnl","RSI_14","VOL_20","NORM_ATR","volume_ratio",
        "volume_percentile","ema_50_200_diff","ema_flag","NORM_BB", "BB_Squeeze"
    ]].copy()

    return report




# -------------------------
# ENHANCED: Winning Trades Report Function with Extended Metrics
# -------------------------
def create_win_report(trades_df, master_df=None):
    """
    Winning trades report using ENTRY-DATE metrics.
    A 'winning trade' is defined as a trade (symbol, trade_id) whose total realized PnL across all SELL legs > 0.
    For each winning trade, returns a single row anchored to the last SELL (exit) for reference,
    but enriches RSI/ATR/volume/EMA metrics at the trade's ENTRY date (first BUY in that trade).
    """
    td = trades_df.copy()
    td["date"] = pd.to_datetime(td["date"])
    td = td.sort_values(["symbol", "date"]).copy()

    # Per-trade id via cumulative BUYs per symbol
    td["is_buy"] = (td["action"] == "BUY").astype(int)
    td["trade_id"] = td.groupby("symbol")["is_buy"].cumsum()

    # SELL legs (must belong to an open trade_id)
    sells = td[(td["action"] == "SELL") & (td["trade_id"] > 0)].copy()
    if sells.empty:
        print("No SELL legs found.")
        return pd.DataFrame(columns=[
            "date","symbol","trade_id","entry_date","entry_price","shares",
            "total_trade_pnl","RSI_14","VOL_20","NORM_ATR","volume_ratio",
            "volume_percentile","ema_50_200_diff","ema_flag"
        ])

    sells["pnl"] = sells["pnl"].fillna(0.0)

    # Net PnL per trade
    pnl_per_trade = sells.groupby(["symbol","trade_id"], as_index=False)["pnl"].sum()
    winners_keys = pnl_per_trade[pnl_per_trade["pnl"] > 0][["symbol","trade_id"]]

    if winners_keys.empty:
        print("No winning trades found.")
        return pd.DataFrame(columns=[
            "date","symbol","trade_id","entry_date","entry_price","shares",
            "total_trade_pnl","RSI_14","VOL_20","NORM_ATR","volume_ratio",
            "volume_percentile","ema_50_200_diff","ema_flag"
        ])

    # One row per winning trade: last SELL as representative
    win_trades = sells.merge(winners_keys, on=["symbol","trade_id"], how="inner")
    win_trades = (win_trades
                  .sort_values(["symbol","trade_id","date"])
                  .groupby(["symbol","trade_id"], as_index=False)
                  .tail(1)
                  .copy())

    # Attach total_trade_pnl
    win_trades = win_trades.merge(
        pnl_per_trade.rename(columns={"pnl":"total_trade_pnl"}),
        on=["symbol","trade_id"], how="left"
    )

    # Derive entry_date (first BUY in the trade)
    buys = td[td["action"] == "BUY"][["symbol","trade_id","date","entry_price","shares"]].copy()
    entry_per_trade = buys.groupby(["symbol","trade_id"], as_index=False)["date"].min().rename(columns={"date":"entry_date"})
    win_trades = win_trades.merge(entry_per_trade, on=["symbol","trade_id"], how="left")

    # Metrics at ENTRY date
    if master_df is not None:
        m = master_df.copy()
        m["date"] = pd.to_datetime(m["date"])

        def metrics_at_entry(row):
            df_sym = m[m["symbol"] == row["symbol"]]
            if df_sym.empty or pd.isna(row["entry_date"]):
                return pd.Series([np.nan]*6)
            closest_row = df_sym.iloc[(df_sym["date"] - row["entry_date"]).abs().argmin()]
            return pd.Series([
                closest_row.get("RSI_14", np.nan),
                closest_row.get("VOL_20", np.nan),
                closest_row.get("NORM_ATR", np.nan),
                closest_row.get("volume_ratio", np.nan),
                closest_row.get("volume_percentile", np.nan),
                closest_row.get("ema_50_200_diff", np.nan),
                closest_row.get("NORM_BB", np.nan),
                closest_row.get("BB_Squeeze", np.nan),
            ])

        win_trades[["RSI_14","VOL_20","NORM_ATR","volume_ratio",
                    "volume_percentile","ema_50_200_diff","NORM_BB", "BB_Squeeze"]] = \
            win_trades.apply(metrics_at_entry, axis=1)
    else:
        win_trades["RSI_14"] = np.nan
        win_trades["VOL_20"] = np.nan
        win_trades["NORM_ATR"] = np.nan
        win_trades["volume_ratio"] = np.nan
        win_trades["volume_percentile"] = np.nan
        win_trades["ema_50_200_diff"] = np.nan

    # Add ema_flag (+1 if positive, -1 if negative, 0 if zero, NaN otherwise)
    def flag_func(val):
        if pd.isna(val):
            return np.nan
        elif val > 0:
            return 1
        elif val < 0:
            return -1
        else:
            return 0

    win_trades["ema_flag"] = win_trades["ema_50_200_diff"].apply(flag_func)

    # Final report columns
    report = win_trades[[
        "date","symbol","trade_id","entry_date","entry_price","shares",
        "total_trade_pnl","RSI_14","VOL_20","NORM_ATR","volume_ratio",
        "volume_percentile","ema_50_200_diff","ema_flag","NORM_BB", "BB_Squeeze"
    ]].copy()

    return report




# -------------------------
# Portfolio Manager with Partial Exits + Trailing + DYNAMIC ALLOCATION + NEW TRAILING SL
# -------------------------

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
    #df["ema_9_21_diff"] = (df["EMA_9"] - df["EMA_21"]) / df["close"]
    #df["ema_50_200_diff"] = (df["EMA_50"] - df["EMA_200"]) / df["close"]

    
    df_labeled = label_with_atr_first_touch(df, horizon=horizon, atr_col="ATR_14")
    
    features = [
        "RSI_14", "VOL_20", "NORM_ATR", "volume_ratio",
        "darvas_high", "darvas_low", #"darvas_breakout_up",
        "ema_9_21_diff", "ema_50_200_diff", "NORM_BB"
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
                trailing_sl_after_t3=False,
                activate_drawdown_scaling=True,
                position_scaling_increment=400000,
                max_positions_limit=10):

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
        trailing_sl_after_t3=trailing_sl_after_t3,
        activate_drawdown_scaling=activate_drawdown_scaling,
        position_scaling_increment=position_scaling_increment,
        max_positions_limit=max_positions_limit
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

    #save the model
    out_dir = "models"
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "catboost_latest.cbm")
    model.save_model(model_path)
    print(f"💾 Model saved at {model_path} (date={current_date.date()})")
    
    return trades_df, hist_df, training_df, model, explainer, features, df, master

# -------------------------
# Forward testing
# -------------------------

def forward_pick(master_df,prob_threshold=0.35, max_positions=2,horizon=20):
    """
    Use the most recently saved CatBoost model to generate forward picks
    from the latest available rows in master_df.
    Ensures feature engineering is applied.
    """
    model_path = "models/catboost_latest.cbm"
    if not os.path.exists(model_path):
        print("No saved model ffound.")
    model = CatBoostClassifier()
    model.load_model(model_path)
    print(f"Model loaded from {model_path}")

    # apply feature engineering ame as training
    df_labeled, features = engineer_features_and_labels(master_df,horizon)
    
    # get latest avvailable rows
    latest_date = df_labeled['date'].max()
    today_df = df_labeled[df_labeled["date"] == latest_date].copy()
    if today_df.empty:
        print("No data found for latest date after feature engineering")
        return pd.DataFrame

    # predict probabilities
    probs = model.predict_proba(today_df[features])[:, 1]
    today_df["probability"] = probs

    picks = (today_df[today_df["probability"] > prob_threshold]
            .sort_values("probability", ascending=False)
            .head(max_positions)[["symbol", "probability"]])
    
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    picks_path = os.path.join(out_dir, f"forward_picks_{latest_date.date()}.csv")
    picks.to_csv(picks_path, index=False)

    print(f"💾 Forward picks saved at {picks_path}")
    return picks



# -------------------------
# ENHANCED Plot & Report with Extended Win/Loss Analysis
# -------------------------

def plot_and_report(trades_df, hist_df, training_df, model, explainer, features, df, master_df, trailing_sl_after_t3, out_dir="plots"):
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

    # ENHANCED: Stop-Loss Loss Report
    print("📋 Generating Enhanced Stop-Loss Loss Report...")
    sl_loss_report = create_sl_loss_report(trades_df, master_df)
    
    if not sl_loss_report.empty:
        sl_loss_report.to_csv(os.path.join(out_dir, "sl_loss_report.csv"), index=False)
        print(f"✅ SL Loss Report saved: {len(sl_loss_report)} losing SL trades found")

        metrics_to_plot = ["RSI_14", "VOL_20", "NORM_ATR", "volume_ratio", "volume_percentile", "ema_50_200_diff","NORM_BB","BB_Squeeze"]
        
        for metric in metrics_to_plot:
            if not sl_loss_report[metric].isna().all():
                plt.figure()
                sns.histplot(sl_loss_report[metric].dropna(), bins=20, kde=True, color="red", alpha=0.7)
                plt.title(f"{metric} Distribution for Stop-Loss Losses")
                plt.xlabel(f"{metric} at Entry")
                plt.ylabel("Frequency")
                plt.savefig(os.path.join(out_dir, f"sl_loss_{metric.lower()}_distribution.png"))
                plt.close()
    else:
        print("ℹ️  No stop-loss loss trades found for report")

    # ENHANCED: Winning Trades Report
    print("📋 Generating Enhanced Winning Trades Report... (T1 Exit)")
    win_report = create_win_report(trades_df, master_df)
    
    if not win_report.empty:
        win_report.to_csv(os.path.join(out_dir, "win_trade_report.csv"), index=False)
        print(f"✅ Win Report saved: {len(win_report)} winning trades found")
        
        metrics_to_plot = ["RSI_14", "VOL_20", "NORM_ATR", "volume_ratio", "volume_percentile", "ema_50_200_diff","NORM_BB","BB_Squeeze"]
        
        for metric in metrics_to_plot:
            if not win_report[metric].isna().all():
                plt.figure()
                sns.histplot(win_report[metric].dropna(), bins=20, kde=True, color="green", alpha=0.7)
                plt.title(f"{metric} Distribution for Winning Trades (T1 Hit)")
                plt.xlabel(f"{metric} at Entry")
                plt.ylabel("Frequency")
                plt.savefig(os.path.join(out_dir, f"win_trade_{metric.lower()}_distribution.png"))
                plt.close()
        
        # Comparative analysis
        if not sl_loss_report.empty:
            fig, axes = plt.subplots(3, 2, figsize=(15, 18))
            axes = axes.flatten()
            
            for i, metric in enumerate(metrics_to_plot):
                if i < len(axes):
                    axes[i].hist(win_report[metric].dropna(), bins=15, alpha=0.7, color="green", label="Winners", density=True)
                    axes[i].hist(sl_loss_report[metric].dropna(), bins=15, alpha=0.7, color="red", label="Losers", density=True)
                    axes[i].set_title(f"{metric} Comparison: Winners vs Losers")
                    axes[i].set_xlabel(metric)
                    axes[i].set_ylabel("Density")
                    axes[i].legend()
            
            if len(metrics_to_plot) < len(axes):
                axes[-1].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "win_vs_loss_extended_comparative_analysis.png"), dpi=150)
            plt.close()
            
            # Summary stats
            print(f"\n📊 ENHANCED ANALYSIS SUMMARY:")
            print(f"{'='*70}")
            print(f"{'Metric':<20} {'Winners (Avg)':<15} {'Losers (Avg)':<15} {'Difference':<10}")
            print(f"{'-'*70}")
            
            for metric in metrics_to_plot:
                win_avg = win_report[metric].mean() if not win_report[metric].isna().all() else 0
                loss_avg = sl_loss_report[metric].mean() if not sl_loss_report[metric].isna().all() else 0
                diff = win_avg - loss_avg
                print(f"{metric:<20} {win_avg:<15.4f} {loss_avg:<15.4f} {diff:<10.4f}")

        # NEW: Scatter plots EMA vs PnL with regression line
        if "ema_50_200_diff" in win_report.columns and "total_trade_pnl" in win_report.columns:
            plt.figure(figsize=(8,6))
            # winners
            sns.scatterplot(x="ema_50_200_diff", y="total_trade_pnl", data=win_report,
                            alpha=0.6, color="green", label="Winners")
            # losers
            if not sl_loss_report.empty and "ema_50_200_diff" in sl_loss_report.columns:
                sns.scatterplot(x="ema_50_200_diff", y="total_trade_pnl", data=sl_loss_report,
                                alpha=0.6, color="red", label="Losers")
            
            # regression line (all trades combined)
            combined = pd.concat([
                win_report[["ema_50_200_diff","total_trade_pnl"]],
                sl_loss_report[["ema_50_200_diff","total_trade_pnl"]] if not sl_loss_report.empty else pd.DataFrame()
            ])
            combined = combined.dropna()
            if not combined.empty:
                sns.regplot(x="ema_50_200_diff", y="total_trade_pnl", data=combined,
                            scatter=False, color="blue", line_kws={"lw":2, "alpha":0.8}, label="Trendline")

            plt.axvline(0, color="black", linestyle="--", alpha=0.8)
            plt.xlabel("EMA 50/200 Difference")
            plt.ylabel("Total Trade PnL")
            plt.title("PnL vs EMA 50/200 Diff (Winners vs Losers)")
            plt.legend()
            plt.savefig(os.path.join(out_dir, "pnl_vs_ema_scatter.png"))
            plt.close()
    else:
        print("ℹ️  No winning trades found for report")
    # SHAP Feature Importance Plots
    if explainer is not None:
        try:
            print("Generating SHAP feature importance plots...")
            
            # Generate sample data for SHAP explanation
            sample_size = min(200, len(df))
            if sample_size > 50:
                # Use the last training data for explanation
                sample_df = df.sample(n=sample_size, random_state=42)
                X_sample = sample_df[features]
                
                # Calculate SHAP values
                shap_values = explainer.shap_values(X_sample)
                
                # SHAP Summary Plot (Bar chart - Feature Importance)
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "shap_feature_importance.png"), dpi=150, bbox_inches='tight')
                plt.close()
                
                # SHAP Summary Plot (Beeswarm)
                plt.figure(figsize=(10, 8))
                shap.summary_plot(shap_values, X_sample, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, "shap_summary_beeswarm.png"), dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"✅ SHAP plots saved to {out_dir}")
        except Exception as e:
            print(f"❌ Error generating SHAP plots: {e}")
    else:
        print("⚠️ SHAP explainer not available - skipping feature importance plots")

    # --- rest unchanged (trailing SL, allocation, training metrics, feature importance, SHAP) ---

    # save CSVs
    trades_df.to_csv(os.path.join(out_dir, "trades.csv"), index=False)
    hist_df.to_csv(os.path.join(out_dir, "portfolio_history.csv"), index=False)
    training_df.to_csv(os.path.join(out_dir, "training_log.csv"), index=False)
    
    print("✅ Enhanced plots and CSVs saved to", out_dir)


# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    # config
    analysis_folder = "analysis_data/darvas_5"
    prob_threshold = 0.35
    retrain_freq_days = 20
    train_window_days = 260
    val_window_days = 60
    horizon = 20
    initial_balance = 100000
    max_positions = 2
    cost_rate = 0.001
    require_flat_before_new = False  # false to have position more than 1 in portfolio

    # Dynamic allocation parameters
    portfolio_increase_threshold = 0.2  # 20% portfolio growth triggers allocation increase
    allocation_increment = 0.15  # 15% increase in allocation multiplier

    # Trailing stop-loss parameter
    trailing_sl_after_t3 = True  # Set to False for original behavior, stopping all at t3, keep true to ride trend till end
    activate_drawdown_scaling = False  # Set to False to disable drawdown controls
    
    # Position size tuning
    position_scaling_increment = 40000000  # $400k growth → +1 position
    max_positions_limit = 10             # Hard cap to prevent over-diversification
     
    if sys.argv[1] == "1":
        # orrect unpacking to handle all 8 returned values
        trades_df, hist_df, training_df, model, explainer, features, df, master_df = run_backtest(
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
            trailing_sl_after_t3=trailing_sl_after_t3,
            activate_drawdown_scaling=activate_drawdown_scaling,
            position_scaling_increment=position_scaling_increment,
            max_positions_limit=max_positions_limit
        )

        plot_and_report(trades_df, hist_df, training_df, model, explainer, features, df, master_df, trailing_sl_after_t3, out_dir=PLOTS_DIR)



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

                print(f"\n BACKTEST SUMMARY:")
                print(f"Total P&L: {total_pnl:,.2f}")
                print(f"Win Rate: {win_rate:.1%}")
                print(f"Average Win: {avg_win:,.2f}")
                print(f"Average Loss: {avg_loss:,.2f}")
                print(f"Total Trades: {len(sells)}")

                if trailing_sl_after_t3:
                    trail_actions = trades_df[trades_df["action"].isin(["T3_TRAIL", "TRAIL_SL"])]
                    print(f"Trailing Actions: {len(trail_actions)}")

                # Stop-Loss Trades Summary
                sl_losses = trades_df[(trades_df["action"] == "SELL") &
                                    (trades_df["reason"] == "STOPLOSS") &
                                    (trades_df["pnl"] < 0)]
                if not sl_losses.empty:
                    print(f"📉 Stop-Loss Losses: {len(sl_losses)} trades")
                    print(f"   Total SL Loss Amount: {sl_losses['pnl'].sum():,.2f}")
                    print(f"   Average SL Loss: {sl_losses['pnl'].mean():,.2f}")

                # Winning Trades Summary
                win_trades = trades_df[(trades_df["action"] == "SELL") &
                                    (trades_df["reason"].str.startswith("T1_EXIT", na=False))]
                if not win_trades.empty:
                    print(f"🎯 Winning Trades (T1 Hit): {len(win_trades)} trades")
                    print(f"   Total Win Amount: {win_trades['pnl'].sum():,.2f}")
                    print(f"   Average Win: {win_trades['pnl'].mean():,.2f}")

    if sys.argv[1] == "2":
        master_df = create_master_dataframe("analysis_data/darvas_5","2019-01-01")
        picks = forward_pick(master_df=master_df, prob_threshold=0.1, max_positions=2, horizon=20)
        if not picks.empty:
            print(picks)