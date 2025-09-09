import pandas as pd
import os

file = 'stocks.csv'
stocks_list = pd.read_csv(file)['Symbol'].to_list()

for i in stocks_list:
    if not os.path.exists(f'stock_data/{i}_historical_1d.csv'):
        print(i)

from technical_indicators_safe import create_analysis_with_indicators, list_all_analysis_versions
# This does everything: copy raw data + add indicators + create master file
analysis_dir = create_analysis_with_indicators('swing_trading_analysis_v1')