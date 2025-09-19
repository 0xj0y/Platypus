This is a readme file for my ai/ml project.
----------------------------------------------------

```python
# Save the script as 'technical_indicators_safe.py' in your project folder
# Then import it
from technical_indicators_safe import create_analysis_with_indicators, list_all_analysis_versions
# This does everything: copy raw data + add indicators + create master file
analysis_dir = create_analysis_with_indicators('swing_trading_analysis_v1')
```

-----------------------------------------------------




# IDEAS for my ml project


# Basic idea:
    A trading bot , that will create portfolios independently trained with ML. The bot will not directly place order throgh broker as of now. 
# Trading Logic:
    1. Long only swing trade on daily timeframe
    2. Entry trigger is when 9 ema crosses over 21 ema
    3. Enntry time is next day or subsequent days when trading price is equal to close of the day when crossover happend.
    4. Also 50 ema is above 200 ema and rsi over 50
    5. Take profit level is above 7% and sl is based on atr
    6. if stock doesnt trigger sl or tp in 15 days, bot will exit the trade.
# some more thoughts:
    1. Bot will do everything independently. It will add or remove stocks aet eod after i provide with latest eod ohlcv of stocks. 
    2. The bot will make portfolio of at max 5 stocks at a time so there will no more than 5 stocks in portfolio at a time.
    3. Initial fund balance will be 100000 rs. Initially each stock can be allocated maximum 18000 rs so after 5 stocks, portfolio will have 10k free cash. Bot can allocate less than 18k rs to any sotck if it thinks so baszed on volatility and other aspects.
    4. In any case no stock will have morethan 18000 allocated to it. partial profit booking and trailing stop loss is permitted.
    5. Allocation of funds is valid only after 100% increase in fund balance. Only then allocation will be increased 100%.

# my stock data structure
    The stock data files looks like this
    ```csv
    date,symbol,raw_symbol,stock_name,open,high,low,close,volume,timestamp,RSI_14,EMA_9,EMA_21,EMA_50,EMA_200,ATR_14,returns,VOL_20,NORM_ATR,volume_ma_20,volume_ratio,volume_percentile,volume_spike
    2025-09-02,NSE:TECHM-EQ,TECHM,Tech Mahindra,1502.0,1519.9,1498.6,1512.8,1137022,1756771200,56.673684210526346,1502.8418478199758,1504.7069811161937,1527.43016627838,1547.3519733744101,30.77857142857145,0.004848887412819636,0.013359688520990148,0.020345433255269336,1619880.65,0.7019171443278862,0.1626984126984127,0
    2025-09-03,NSE:TECHM-EQ,TECHM,Tech Mahindra,1512.8,1516.1,1497.3,1508.2,1431929,1756857600,49.74287050023378,1503.9134782559809,1505.024528287449,1526.6760421106005,1546.962401500038,29.521428571428583,-0.0030407191961924207,0.012298878270655306,0.019573948131168667,1612403.45,0.8880711586172804,0.3373015873015873,0
    2025-09-04,NSE:TECHM-EQ,TECHM,Tech Mahindra,1506.1,1510.8,1495.3,1500.4,1038934,1756944000,48.651120256058555,1503.210782604785,1504.6041166249536,1525.6456090866554,1546.4990940224257,29.24285714285715,-0.00517172788754805,0.012295746824907498,0.019490040751037822,1517866.3,0.6844700353384221,0.1111111111111111,0
    2025-09-05,NSE:TECHM-EQ,TECHM,Tech Mahindra,1499.0,1507.4,1462.1,1477.9,1875965,1757030400,48.014440433213,1498.1486260838283,1502.1764696590485,1523.7732322597278,1545.8165159724515,29.271428571428583,-0.014996001066382325,0.012091643387138858,0.01980609552163785,1559665.9,1.2027992661761728,0.5753968253968254,0
    2025-09-08,NSE:TECHM-EQ,TECHM,Tech Mahindra,1482.4,1487.9,1459.0,1460.7,1308412,1757289600,47.648902821316604,1490.6589008670628,1498.4058815082258,1521.299772171111,1544.969585465263,29.05714285714287,-0.011638135191826215,0.011846418185265889,0.01989261508670012,1545712.05,0.8464784886680543,0.2698412698412698,0
    2025-09-09,NSE:TECHM-EQ,TECHM,Tech Mahindra,1466.6,1500.0,1465.7,1498.2,1749566,1757376000,50.318742031449204,1492.1671206936503,1498.387165007478,1520.3938987526362,1544.5042164556585,29.350000000000023,0.025672622715136484,0.013226186802373035,0.019590174876518502,1564607.1,1.1182142788435512,0.5079365079365079,0
    ```
Here many of the stock data starts from different dates though majority starts from 4th jan 2010. I have around 188 stocks data.some fin range 2010 to 2025(today), some maybe less thant that. all are up to date data. these data are in all seperate stock files named {stock_symbol}_historical_1d.csv and also all combined in a master_historical_data.csv with around 690000 rows.