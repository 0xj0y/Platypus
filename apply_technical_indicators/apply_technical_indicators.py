"""
Safe Technical Indicators Module
- Preserves original raw data integrity
- Creates analysis copies with technical indicators
- Manages multiple analysis versions safely
"""

import os
import shutil
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
raw_data_dir = 'stock_data'
analysis_base_dir = 'analysis_data/'
class SafeDataManager:
    """Manages safe copying and analysis of stock data"""
    
    def __init__(self, raw_data_dir=raw_data_dir, analysis_base_dir=analysis_base_dir):
        self.raw_data_dir = raw_data_dir
        self.analysis_base_dir = analysis_base_dir
    
    def create_analysis_copy(self, analysis_name=None):
        """
        Create a safe copy of raw data for analysis
        """
        if analysis_name is None:
            analysis_name = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        analysis_dir = os.path.join(self.analysis_base_dir, analysis_name)
        
        # Create analysis directory
        os.makedirs(analysis_dir, exist_ok=True)
        
        print(f"📂 Creating analysis copy: {analysis_name}")
        print(f"📍 Location: {analysis_dir}")
        
        # Check if raw data directory exists
        if not os.path.exists(self.raw_data_dir):
            print(f"❌ Raw data directory not found: {self.raw_data_dir}")
            print("💡 Please ensure your stock data is downloaded first")
            return None
        
        # Find all CSV files in raw data directory
        csv_pattern = os.path.join(self.raw_data_dir, '*_historical_1d.csv')
        raw_files = glob.glob(csv_pattern)
        
        if not raw_files:
            print(f"❌ No historical CSV files found in {self.raw_data_dir}")
            print("💡 Expected files matching pattern: *_historical_1d.csv")
            return None
        
        print(f"🔍 Found {len(raw_files)} raw data files to copy")
        
        copied_count = 0
        skipped_count = 0
        
        for raw_file in raw_files:
            filename = os.path.basename(raw_file)
            dest_file = os.path.join(analysis_dir, filename)
            
            if not os.path.exists(dest_file):
                try:
                    shutil.copy2(raw_file, dest_file)
                    copied_count += 1
                    if copied_count <= 5:  # Show first 5 for brevity
                        print(f"   ✅ Copied: {filename}")
                except Exception as e:
                    print(f"   ❌ Error copying {filename}: {str(e)}")
            else:
                skipped_count += 1
        
        if copied_count > 5:
            print(f"   ✅ ... and {copied_count - 5} more files")
        
        print(f"📊 Copy summary: {copied_count} copied, {skipped_count} skipped")
        
        if copied_count == 0 and skipped_count == 0:
            print("❌ No files were processed")
            return None
        
        return analysis_dir
    
    def list_analysis_versions(self):
        """List all available analysis versions"""
        if not os.path.exists(self.analysis_base_dir):
            print("📂 No analysis directory found")
            return []
        
        versions = [d for d in os.listdir(self.analysis_base_dir) 
                   if os.path.isdir(os.path.join(self.analysis_base_dir, d))]
        
        if versions:
            print("📊 Available analysis versions:")
            for i, version in enumerate(versions, 1):
                version_path = os.path.join(self.analysis_base_dir, version)
                file_count = len(glob.glob(os.path.join(version_path, '*.csv')))
                print(f"   {i}. {version} ({file_count} files)")
        else:
            print("📂 No analysis versions found")
        
        return sorted(versions)
    
    def delete_analysis_version(self, version_name):
        """Safely delete an analysis version"""
        version_path = os.path.join(self.analysis_base_dir, version_name)
        
        if os.path.exists(version_path):
            try:
                shutil.rmtree(version_path)
                print(f"🗑️  Deleted analysis version: {version_name}")
                return True
            except Exception as e:
                print(f"❌ Error deleting {version_name}: {str(e)}")
                return False
        else:
            print(f"❌ Analysis version not found: {version_name}")
            return False

class TechnicalIndicators:
    """Calculate technical indicators on analysis data"""
    
    @staticmethod
    def calculate_RSI(df, period=14, column='close'):
        """Calculate Relative Strength Index"""
        if len(df) < period:
            return pd.Series(np.nan, index=df.index)
        
        delta = df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()
        
        # Handle division by zero
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_EMA(df, period, column='close'):
        """Calculate Exponential Moving Average"""
        if len(df) == 0:
            return pd.Series(dtype=float)
        return df[column].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_ATR(df, period=14):
        """Calculate Average True Range"""
        if len(df) < 2:
            return pd.Series(np.nan, index=df.index)
        
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period, min_periods=period).mean()
        return atr
    
    @staticmethod
    def normalized_ATR(df,period=14):
        """Normalized atr to provide more details about volatility
        """
        if len(df) < period:
            return pd.Series(np.nan, index=df.index)
        atr = TechnicalIndicators.calculate_ATR(df,period)
        normalized_atr = atr/df['close']
        return normalized_atr

    @staticmethod
    def rolling_stdev(df, period=20):
        """calculate rolling standard deviation to detrmine volatility"""
        df['returns'] = df['close'].pct_change()
        volatility_20d = df['returns'].rolling(period).std()
        return volatility_20d
    
    @staticmethod
    def volume_features(df, period=20):
        """
        Add normalized volume features to handle growth and splits
        """
        # 20-day volume moving average
        df['volume_ma_20'] = df['volume'].rolling(window=period).mean()
        
        # Volume ratio (current volume / 20-day average)
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        
        # Volume percentile over last 252 days (1 year)
        df['volume_percentile'] = df['volume'].rolling(window=252, min_periods=20).rank(pct=True)
        
        # Volume spike detection (volume > 2x average)
        df['volume_spike'] = (df['volume_ratio'] > 2.0).astype(int)
        
        return df
    
    @staticmethod
    def calculate_darvas_box(df, boxp=5):
        """
        Calculate Darvas Box high/low and breakout signals.
        Logic adapted from PineScript.
        
        Args:
            df (pd.DataFrame): Must contain 'high','low','close'
            boxp (int): Lookback length (default=5)
        
        Returns:
            pd.DataFrame: df with darvas_high, darvas_low, darvas_breakout_up
        """
        df = df.copy()

        # rolling highs and lows
        LL = df['low'].rolling(window=boxp).min()
        k1 = df['high'].rolling(window=boxp).max()
        k2 = df['high'].rolling(window=boxp-1).max()
        k3 = df['high'].rolling(window=boxp-2).max()

        # box condition
        box1 = k3 < k2
        cond_newhigh = df['high'] > k1.shift(1)

        NH = np.where(cond_newhigh, df['high'], np.nan)
        NH = pd.Series(NH).ffill().values

        barssince = (~cond_newhigh).groupby(cond_newhigh.cumsum()).cumcount()

        darvas_high = np.where((barssince == (boxp - 2)) & (box1), NH, np.nan)
        darvas_high = pd.Series(darvas_high).ffill().values

        darvas_low = np.where((barssince == (boxp - 2)) & (box1), LL, np.nan)
        darvas_low = pd.Series(darvas_low).ffill().values

        # breakout: only mark first breakout after box
        darvas_breakout_up = [0] * len(df)
        breakout_done = False
        for i in range(1, len(df)):
            if np.isnan(darvas_high[i]) or np.isnan(darvas_high[i - 1]):
                continue
            if not breakout_done and df['close'].iat[i] > darvas_high[i - 1]:
                darvas_breakout_up[i] = 1
                breakout_done = True
            if darvas_high[i] != darvas_high[i - 1]:
                breakout_done = False

        df['darvas_high'] = darvas_high
        df['darvas_low'] = darvas_low
        df['darvas_breakout_up'] = darvas_breakout_up
        # Clean up helper columns
        df.drop(columns=['LL', 'k1', 'k2', 'k3', 'box1', 'NH', 'barssince'], inplace=True, errors='ignore')
        return df



    
    @staticmethod
    def add_all_indicators(df):
        """Add all technical indicators to DataFrame"""
        if len(df) == 0:
            print("⚠️  Empty DataFrame - no indicators added")
            return df
        
        df_enhanced = df.copy()
        
        try:
            # RSI (14-period)
            df_enhanced['RSI_14'] = TechnicalIndicators.calculate_RSI(df_enhanced, 14)
            
            # EMAs (9, 21, 50, 200 periods)
            df_enhanced['EMA_9'] = TechnicalIndicators.calculate_EMA(df_enhanced, 9)
            df_enhanced['EMA_21'] = TechnicalIndicators.calculate_EMA(df_enhanced, 21)
            df_enhanced['EMA_50'] = TechnicalIndicators.calculate_EMA(df_enhanced, 50)
            df_enhanced['EMA_200'] = TechnicalIndicators.calculate_EMA(df_enhanced, 200)
            
            # ATR (14-period)
            df_enhanced['ATR_14'] = TechnicalIndicators.calculate_ATR(df_enhanced, 14)
            #rolling stdev 20 period
            df_enhanced['VOL_20'] = TechnicalIndicators.rolling_stdev(df_enhanced,20)
            #normalized atr
            df_enhanced['NORM_ATR'] = TechnicalIndicators.normalized_ATR(df_enhanced,14)
            #add volume features
            df_enhanced = TechnicalIndicators.volume_features(df_enhanced, 20)
            
            #add darvas box and breakouts
            df_enhanced = TechnicalIndicators.calculate_darvas_box(df_enhanced, boxp=5)
            df_enhanced["ema_9_21_diff"] = (df_enhanced["EMA_9"] - df_enhanced["EMA_21"]) / df_enhanced["close"]
            df_enhanced["ema_50_200_diff"] = (df_enhanced["EMA_50"] - df_enhanced["EMA_200"]) / df_enhanced["close"]
        
        except Exception as e:
            print(f"⚠️  Error adding indicators: {str(e)}")
            return df
        
        return df_enhanced

class AnalysisProcessor:
    """Process analysis data with technical indicators"""
    
    def __init__(self, analysis_dir):
        self.analysis_dir = analysis_dir
        
        if not os.path.exists(analysis_dir):
            raise ValueError(f"Analysis directory does not exist: {analysis_dir}")
    
    def add_indicators_to_all_files(self):
        """Add technical indicators to all CSV files in analysis directory"""
        print(f"🔧 Adding technical indicators to files in {os.path.basename(self.analysis_dir)}...")
        
        csv_files = glob.glob(os.path.join(self.analysis_dir, '*_historical_1d.csv'))
        
        if not csv_files:
            print("❌ No CSV files found in analysis directory")
            return False
        
        print(f"📊 Processing {len(csv_files)} files...")
        
        success_count = 0
        failed_count = 0
        
        for i, file_path in enumerate(csv_files, 1):
            filename = os.path.basename(file_path)
            stock_symbol = filename.replace('_historical_1d.csv', '')
            
            print(f"📈 [{i:3d}/{len(csv_files)}] {stock_symbol:<15}", end=" ")
            
            try:
                # Load data
                df = pd.read_csv(file_path)
                
                # Ensure date column is datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                original_cols = len(df.columns)
                
                # Add technical indicators
                df_with_indicators = TechnicalIndicators.add_all_indicators(df)
                
                # Save enhanced data back to same file
                df_with_indicators.to_csv(file_path, index=False)
                
                new_cols = len(df_with_indicators.columns)
                indicators_added = new_cols - original_cols
                
                print(f"✅ {indicators_added} indicators added ({len(df)} records)")
                
                success_count += 1
                
            except Exception as e:
                print(f"❌ Error: {str(e)}")
                failed_count += 1
        
        print(f"\n🎉 Processing complete!")
        print(f"✅ Successfully processed: {success_count} files")
        print(f"❌ Failed: {failed_count} files")
        print(f"📊 Success rate: {success_count/(success_count+failed_count)*100:.1f}%")
        
        return success_count > 0
    
    def create_master_file(self, output_filename=None):
        """Create master file combining all stocks with indicators"""
        if output_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"master_with_indicators_{timestamp}.csv"
        
        print(f"🔄 Creating master file with indicators...")
        
        csv_files = glob.glob(os.path.join(self.analysis_dir, '*_historical_1d.csv'))
        
        if not csv_files:
            print("❌ No CSV files found for master file creation")
            return None
        
        all_dataframes = []
        
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path, parse_dates=['date'])
                all_dataframes.append(df)
            except Exception as e:
                print(f"⚠️  Error reading {os.path.basename(file_path)}: {e}")
        
        if not all_dataframes:
            print("❌ No data loaded for master file")
            return None
        
        # Combine all data
        print(f"📊 Combining {len(all_dataframes)} stock files...")
        master_df = pd.concat(all_dataframes, ignore_index=True)
        master_df = master_df.sort_values(['raw_symbol', 'date']) if 'raw_symbol' in master_df.columns else master_df.sort_values(['symbol', 'date'])
        
        # Save master file
        master_path = os.path.join(self.analysis_dir, output_filename)
        master_df.to_csv(master_path, index=False)
        
        print(f"💾 Master file created: {output_filename}")
        print(f"📊 Total records: {len(master_df):,}")
        
        if 'raw_symbol' in master_df.columns:
            unique_stocks = master_df['raw_symbol'].nunique()
        elif 'symbol' in master_df.columns:
            unique_stocks = master_df['symbol'].nunique()
        else:
            unique_stocks = "Unknown"
        
        print(f"📈 Unique stocks: {unique_stocks}")
        print(f"📅 Date range: {master_df['date'].min().date()} to {master_df['date'].max().date()}")
        print(f"🔧 Technical indicators included: RSI_14, EMA_9/21/50/200, ATR_14")
        
        return master_path

# Easy-to-use workflow functions
def create_analysis_with_indicators(analysis_name=None, raw_data_dir=raw_data_dir):
    """Complete workflow: copy raw data + add indicators + create master file"""
    print("🚀 Starting complete safe analysis workflow...")
    print("🔒 Your raw data will remain untouched")
    
    # Step 1: Create safe copy
    dm = SafeDataManager(raw_data_dir=raw_data_dir)
    analysis_dir = dm.create_analysis_copy(analysis_name)
    
    if not analysis_dir:
        print("❌ Failed to create analysis copy")
        return None
    
    # Step 2: Add technical indicators
    try:
        processor = AnalysisProcessor(analysis_dir)
        
        print(f"\n🔧 Adding technical indicators...")
        success = processor.add_indicators_to_all_files()
        
        if not success:
            print("❌ Failed to add indicators")
            return analysis_dir
        
        # Step 3: Create master file
        print(f"\n📊 Creating master file...")
        master_file = processor.create_master_file()
        
        print(f"\n🎉 Complete analysis ready!")
        print(f"📂 Analysis directory: {analysis_dir}")
        if master_file:
            print(f"📊 Master file: {os.path.basename(master_file)}")
        
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
    
    return analysis_dir

def list_all_analysis_versions():
    """List all analysis versions"""
    dm = SafeDataManager()
    return dm.list_analysis_versions()

def delete_analysis_version(version_name):
    """Delete an analysis version"""
    dm = SafeDataManager()
    return dm.delete_analysis_version(version_name)

def load_analysis_stock(stock_symbol, analysis_version=None):
    """Load a specific stock from analysis directory"""
    if analysis_version is None:
        versions = list_all_analysis_versions()
        if not versions:
            print("❌ No analysis versions found")
            return None
        analysis_version = versions[-1]  # Use latest
    
    analysis_dir = os.path.join(analysis_base_dir, analysis_version)
    file_pattern = f"{stock_symbol.upper()}_historical_1d.csv"
    file_path = os.path.join(analysis_dir, file_pattern)
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_pattern} in {analysis_version}")
        return None
    
    df = pd.read_csv(file_path, parse_dates=['date'])
    print(f"✅ Loaded {stock_symbol} from {analysis_version} ({len(df)} records)")
    return df

# Quick demo function
def demo_safe_workflow():
    """Demonstrate the safe workflow"""
    print("🔬 Demo: Safe Technical Indicators Workflow")
    print("=" * 50)
    
    # List existing versions
    print("📋 Existing analysis versions:")
    versions = list_all_analysis_versions()
    
    # Create new analysis
    print("\n🚀 Creating new analysis with indicators...")
    analysis_dir = create_analysis_with_indicators('demo_analysis')
    
    if analysis_dir:
        print(f"\n✅ Demo complete! Check your analysis in:")
        print(f"   {analysis_dir}")

if __name__ == "__main__":
    print("📊 Safe Technical Indicators System")
    print("=" * 50)
    print("🔒 Protects your raw data from modification")
    print("🔧 Adds RSI, EMA, ATR indicators safely")
    print("📁 Creates organized analysis versions")
    print("=" * 50)
    
    print("\n📝 Quick commands:")
    print("1. create_analysis_with_indicators('my_analysis')")
    print("2. list_all_analysis_versions()")
    print("3. load_analysis_stock('RELIANCE', 'my_analysis')")
    print("4. demo_safe_workflow()")
