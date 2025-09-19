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

class SafeDataManager:
    """Manages safe copying and analysis of stock data"""
    
    def __init__(self, raw_data_dir='stock_data', analysis_base_dir='analysis_data'):
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
                        print(f"  ✅ Copied: {filename}")
                except Exception as e:
                    print(f"  ❌ Error copying {filename}: {str(e)}")
            else:
                skipped_count += 1
        
        if copied_count > 5:
            print(f"  ✅ ... and {copied_count - 5} more files")
        
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
                print(f"  {i}. {version} ({file_count} files)")
        else:
            print("📂 No analysis versions found")
        
        return sorted(versions)
    
    def delete_analysis_version(self, version_name):
        """Safely delete an analysis version"""
        version_path = os.path.join(self.analysis_base_dir, version_name)
        
        if os.path.exists(version_path):
            try:
                shutil.rmtree(version_path)
                print(f"🗑️ Deleted analysis version: {version_name}")
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
    def calculate_ADX(df, period=14):
        """Calculate Average Directional Movement Index"""
        if len(df) < period + 1:
            return pd.Series(np.nan, index=df.index)
        
        # Calculate True Range
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        up_move = df['high'] - df['high'].shift()
        down_move = df['low'].shift() - df['low']
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm = pd.Series(plus_dm, index=df.index)
        minus_dm = pd.Series(minus_dm, index=df.index)
        
        # Calculate smoothed averages
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    @staticmethod
    def calculate_ADX_Wilder(df, period=14):
        """Calculate ADX using Wilder's smoothing method"""
        if len(df) < period + 1:
            return pd.Series(np.nan, index=df.index)
        
        # Similar to ADX but with Wilder's smoothing (EMA with alpha = 1/period)
        alpha = 1.0 / period
        
        # Calculate True Range
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        up_move = df['high'] - df['high'].shift()
        down_move = df['low'].shift() - df['low']
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm = pd.Series(plus_dm, index=df.index)
        minus_dm = pd.Series(minus_dm, index=df.index)
        
        # Wilder's smoothing
        atr_wilder = tr.ewm(alpha=alpha).mean()
        plus_di_wilder = 100 * (plus_dm.ewm(alpha=alpha).mean() / atr_wilder)
        minus_di_wilder = 100 * (minus_dm.ewm(alpha=alpha).mean() / atr_wilder)
        
        # Calculate ADX with Wilder's method
        dx_wilder = 100 * np.abs(plus_di_wilder - minus_di_wilder) / (plus_di_wilder + minus_di_wilder + 1e-10)
        adx_wilder = dx_wilder.ewm(alpha=alpha).mean()
        
        return adx_wilder
    
    @staticmethod
    def calculate_DeMarker(df, period=14):
        """Calculate DeMarker oscillator"""
        if len(df) < period + 1:
            return pd.Series(np.nan, index=df.index)
        
        # DeMax and DeMin calculations
        demax = np.where(df['high'] > df['high'].shift(), df['high'] - df['high'].shift(), 0)
        demin = np.where(df['low'] < df['low'].shift(), df['low'].shift() - df['low'], 0)
        
        demax = pd.Series(demax, index=df.index)
        demin = pd.Series(demin, index=df.index)
        
        # Calculate DeMarker
        demax_sma = demax.rolling(window=period).mean()
        demin_sma = demin.rolling(window=period).mean()
        
        demarker = demax_sma / (demax_sma + demin_sma + 1e-10)
        
        return demarker
    
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
    def calculate_RVI(df, period=10):
        """Calculate Relative Vigor Index"""
        if len(df) < period:
            return pd.Series(np.nan, index=df.index)
        
        # Calculate numerator and denominator
        numerator = (df['close'] - df['open']) + 2 * (df['close'].shift() - df['open'].shift()) + \
                   2 * (df['close'].shift(2) - df['open'].shift(2)) + (df['close'].shift(3) - df['open'].shift(3))
        
        denominator = (df['high'] - df['low']) + 2 * (df['high'].shift() - df['low'].shift()) + \
                     2 * (df['high'].shift(2) - df['low'].shift(2)) + (df['high'].shift(3) - df['low'].shift(3))
        
        # Calculate RVI
        rvi_numerator = numerator.rolling(window=period).sum()
        rvi_denominator = denominator.rolling(window=period).sum()
        
        rvi = rvi_numerator / (rvi_denominator + 1e-10)
        
        return rvi
    
    @staticmethod
    def calculate_Stochastic(df, k_period=5, d_period=3, smooth_period=3):
        """Calculate Stochastic Oscillator"""
        if len(df) < k_period:
            return pd.Series(np.nan, index=df.index)
        
        # Calculate %K
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        
        k_percent = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low + 1e-10)
        
        # Smooth %K
        k_percent_smoothed = k_percent.rolling(window=smooth_period).mean()
        
        # Calculate %D
        d_percent = k_percent_smoothed.rolling(window=d_period).mean()
        
        return k_percent_smoothed  # Return %K (main line)
    
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
        """Add all technical indicators to DataFrame based on MQL5 article"""
        if len(df) == 0:
            print("⚠️ Empty DataFrame - no indicators added")
            return df
        
        df_enhanced = df.copy()
        
        try:
            # Technical Indicators from the article
            df_enhanced['ADX_14'] = TechnicalIndicators.calculate_ADX(df_enhanced, 14)
            df_enhanced['ADX_Wilder_14'] = TechnicalIndicators.calculate_ADX_Wilder(df_enhanced, 14) 
            df_enhanced['DeMarker_14'] = TechnicalIndicators.calculate_DeMarker(df_enhanced, 14)
            df_enhanced['RSI_14'] = TechnicalIndicators.calculate_RSI(df_enhanced, 14)
            df_enhanced['RVI_10'] = TechnicalIndicators.calculate_RVI(df_enhanced, 10)
            df_enhanced['Stochastic'] = TechnicalIndicators.calculate_Stochastic(df_enhanced, 5, 3, 3)
            
            # Price-based features from the article (normalized by 1000)
            # Stationary (current bar normalized return)
            df_enhanced['Stationary'] = 1000 * (df_enhanced['close'] - df_enhanced['open']) / df_enhanced['close']
            
            # Stationary2 (2 bars ago normalized return)
            df_enhanced['Stationary2'] = 1000 * (df_enhanced['close'].shift(2) - df_enhanced['open'].shift(2)) / df_enhanced['close'].shift(2)
            
            # Stationary3 (3 bars ago normalized return)  
            df_enhanced['Stationary3'] = 1000 * (df_enhanced['close'].shift(3) - df_enhanced['open'].shift(3)) / df_enhanced['close'].shift(3)
            
            # Add darvas box and breakouts
            df_enhanced = TechnicalIndicators.calculate_darvas_box(df_enhanced, boxp=5)
            
            # Box Size (normalized)
            df_enhanced['BoxSize'] = 1000 * (df_enhanced['darvas_high'] - df_enhanced['darvas_low']) / df_enhanced['close']
            
            # Distance High (normalized distance from close to darvas high)
            df_enhanced['DistanceHigh'] = 1000 * (df_enhanced['close'] - df_enhanced['darvas_high']) / df_enhanced['close']
            
            # Distance Low (normalized distance from close to darvas low)
            df_enhanced['DistanceLow'] = 1000 * (df_enhanced['close'] - df_enhanced['darvas_low']) / df_enhanced['close']
            
        except Exception as e:
            print(f"⚠️ Error adding indicators: {str(e)}")
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
                print(f"⚠️ Error reading {os.path.basename(file_path)}: {e}")
        
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
        print(f"🔧 Technical indicators included: ADX_14, ADX_Wilder_14, DeMarker_14, RSI_14, RVI_10, Stochastic, etc.")
        
        return master_path

# Easy-to-use workflow functions
def create_analysis_with_indicators(analysis_name=None, raw_data_dir='stock_data'):
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
    
    analysis_dir = os.path.join('analysis_data', analysis_version)
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
        print(f"  {analysis_dir}")

if __name__ == "__main__":
    print("📊 Safe Technical Indicators System")
    print("=" * 50)
    print("🔒 Protects your raw data from modification")
    print("🔧 Adds MQL5 article indicators safely")
    print("📁 Creates organized analysis versions")
    print("=" * 50)
    
    print("\n📝 Quick commands:")
    print("1. create_analysis_with_indicators('my_analysis')")
    print("2. list_all_analysis_versions()")
    print("3. load_analysis_stock('RELIANCE', 'my_analysis')")
    print("4. demo_safe_workflow()")
