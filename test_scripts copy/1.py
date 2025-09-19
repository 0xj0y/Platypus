import pandas as pd
file = "all_stocks.csv"
df = pd.read_csv(file)
from datetime import date
#from apply_technical_indicators import apply_technical_indicators
from fyers_apiv3 import fyersModel
stocks = df['Symbol'].to_list()
for stock in stocks:
    file = f'stock_data/{stock}_historical_1d.csv'
    data = pd.read_csv(file)
    last_date = pd.to_datetime(data['date'].iloc[-1]).date()
    if last_date != date(2025, 9, 18):
        print(f'{stock}  → last available: {last_date}")')
access_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhdWQiOlsiZDoxIiwiZDoyIiwieDowIiwieDoxIiwieDoyIl0sImF0X2hhc2giOiJnQUFBQUFCb3pWMDhvbGNzUmJsLWYxeE9mWkRYUDR1RGttdm1sUXRzNG4xRlRKZHBpTG9HTXdjR2QtMmZWMVo4SGdSd3VOSFFzajBqUG5rNG0wT2pma0RVYUVIYmUtYWZCQk1IUDhZZVJFQ1IxYl9SbFBKS2s3WT0iLCJkaXNwbGF5X25hbWUiOiIiLCJvbXMiOiJLMSIsImhzbV9rZXkiOiJlODdlMjBmODkwMjYxNTYzNTA2NjI1NTQ5NTUyZDk1ZGEyZmNmYzJmNjk1NGViZmU0ZmUwOGNlOCIsImlzRGRwaUVuYWJsZWQiOiJZIiwiaXNNdGZFbmFibGVkIjoiTiIsImZ5X2lkIjoiWEoxMDM1OCIsImFwcFR5cGUiOjEwMCwiZXhwIjoxNzU4MzI4MjAwLCJpYXQiOjE3NTgyODkyMTIsImlzcyI6ImFwaS5meWVycy5pbiIsIm5iZiI6MTc1ODI4OTIxMiwic3ViIjoiYWNjZXNzX3Rva2VuIn0.GYVaRfWiVeaduaO0a-ja1xxINUpq0Ua5RYCBLHbwMbI"    
client_id='XJ10358'
#apply_technical_indicators.create_analysis_with_indicators('ml_v1')
# Initialize the FyersModel instance with your client_id, access_token, and enable async mode
fyers = fyersModel.FyersModel(client_id=client_id, is_async=False, token=access_token, log_path="")
data = {
    "symbol":"NSE:SBIN-EQ",
    "resolution":"D",
    "date_format":"1",
    "range_from":"2025-09-19",
    "range_to":"2025-09-19",
    "cont_flag":"1"
}
response = fyers.history(data=data)
print(response)
