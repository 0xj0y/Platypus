from strategy_config import strategy_config
import pandas as pd
import os


class DataLoader:
    def __init__(self,data_path:str, config: strategy_config):
        self.data_path = data_path
        self.config = config

    
    def load_dataset(self, symbol:str, start_date:str = '2011-01-01') -> pd.DataFrame:

        """Load individual stock data
            Args: Symbol andd start date
            returns: stock data in a dataframe
        """
        try: #construct a file path
            file_path = os.path.join(self.data_path,f'{symbol}_historical_1d.csv')
            if not os.path.exists(file_path):
                print(f"❌ File not found: {file_path}")
                return pd.DataFrame
            #load data
            df = pd.read_csv(file_path)

            #filter the data bassed on start date
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['date'] >= start_date].copy()
            
            #check if data has required features
            required_cols = [
                'date', 'close', 'high', 'low', 'volume',
                'EMA_9', 'EMA_21', 'EMA_50', 'EMA_200',
                'RSI_14', 'ATR_14', 'NORM_ATR',
                'volume_ratio', 'volume_spike'
            ]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"⚠️ {file_path} has missing columns {missing_cols}")
                return pd.DataFrame
            
            df = df.sort_values('date').reset_index(drop=True) 
            df = self._apply_strategy_filters(df,symbol) #apply strategy filters to genrate signals 9later
            print(f'✅ {symbol}: {len(df)} records loaded')
            return df

        except Exception as e:
            print(f'{symbol}:{str(e)}')
            return pd.DataFrame

    def _apply_strategy_filters(self,df: pd.DataFrame,symbol: str) -> pd.DataFrame:
        """Apply strategy specific conditions to generate signals"""        
        original_count = len(df)
        
        #remove the rows with insufficient ema data
        df = df.dropna(subset=['EMA_200']).copy()
        #remove extreme outliers in data
        df = df[df['close'] > 0].copy()
        #filter out stocks with insufficient volatility data
        df = df.dropna(subset=['NORM_ATR', 'volume_ratio', 'VOL_20']).copy()
        #remove days with 0 volume
        df = df[df['volume'] > 0].copy()

        filtered_count = len(df)
        if filtered_count < original_count:
            print(f'{symbol}: Filterd out {original_count - filtered_count} polluted records')
        else:
            print(f'{symbol}: Data has all sufficient feature')
        return df
          


        