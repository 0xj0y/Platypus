# Hybrid Momentum & Mean-Reversion ML Model For Portfolio prediction

This repository contains the code for a sophisticated swing trading backtesting engine. The system uses a `CatBoostClassifier` to identify and execute trades, and has demonstrated the ability to function as a hybrid model, successfully capitalizing on both momentum and mean-reversion opportunities.

## Overview

The core of this project is a Python-based script (`main.py`) that backtests a complex trading strategy on historical stock data. It was initially designed as a momentum system using Darvas box breakouts but evolved into a more robust hybrid model through machine learning. The model analyzes a variety of technical indicators to generate entry signals and manages trades with a detailed risk and trade management framework.

## Key Features

- **Machine Learning Core:** Utilizes a `CatBoostClassifier` to predict profitable trading setups with a given probability.
- **Hybrid Strategy:** The model has learned to trade both:
    - **Momentum:** Entering trades in the direction of the primary trend (e.g., when the 50-day EMA is above the 200-day EMA).
    - **Mean Reversion:** Identifying potential reversals or bounces when a stock is oversold within a larger downtrend.
- **Advanced Trade Management:** Implements a multi-stage profit-taking system and dynamic stop-loss adjustments.
- **Dynamic Position Sizing:** Includes logic to increase trade allocation as the portfolio value grows.
- **Comprehensive Reporting:** Generates detailed performance reports and visualizations to analyze the strategy's effectiveness.

## The Strategy

The trading logic is executed based on the following rules:

### Entry Conditions
- A trade is initiated only when two conditions are met simultaneously:
    1. A **Darvas Box breakout** signal occurs.
    2. The `CatBoostClassifier` model's predicted probability for a winning trade exceeds a predefined threshold.

### Trade & Risk Management
- **Initial Stop-Loss:** An initial stop-loss is immediately placed at 1.5x the Average True Range (ATR) below the entry price.
- **Partial Profit-Taking:**
    - **T1 (1x ATR profit):** Exit 30% of the position and move the stop-loss to the entry price (breakeven).
    - **T2 (2x ATR profit):** Exit another 30% of the position and trail the stop-loss up to the T1 level.
    - **T3 (3x ATR profit):** Exit the remaining 40% of the position.
- **Trailing Stop-Loss:** An optional trailing stop can be enabled after T3 is hit to allow profitable trades to continue running.
- **Maximum Hold Period:** Any open position is automatically closed after 20 trading days to avoid prolonged, non-performing trades.

## Technical Details

- **Model:** `CatBoostClassifier`
- **Primary Features:**
    - `ema_50_200_diff` (Difference between 50-day and 200-day EMA)
    - RSI (Relative Strength Index)
    - ATR (Average True Range)
    - Volume-based metrics
    - Darvas Box indicators

## How to Use

1.  **Install Dependencies:** Make sure you have the required Python libraries installed.
    ```
    pip install pandas numpy matplotlib seaborn catboost
    ```

2.  **Prepare Data:** Ensure your historical stock data is available in a CSV file with columns like `Date`, `Open`, `High`, `Low`, `Close`, `Volume`.

3.  **Configure the Script:** Open `main.py` and adjust the parameters at the top of the file, such as the input file path, stock symbol, and date ranges for training and testing.

4.  **Run the Backtest:**
    ```
    python main.py
    ```

## Outputs

The script will generate several files in a `results/` and `plots/` directory, including:
- **`equity_curve.jpg`:** A plot of the portfolio's equity over the backtest period.
- **`pnl_vs_ema_scatter.jpg`:** A scatter plot visualizing trade profitability against the EMA 50/200 difference, distinguishing between momentum and mean-reversion trades.
- **`win_loss_distribution.jpg`:** Histograms showing the EMA difference at entry for both winning and losing trades.
- **`win_loss_ratio.jpg`:** A pie chart illustrating the overall win rate.
- **Detailed CSV Reports:** Files containing exhaustive data on every winning and losing trade for further analysis.
```
