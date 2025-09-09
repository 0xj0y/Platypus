"""
Enhanced Bulk Data Downloader with Standardized File Naming
- Files saved as {RAW_SYMBOL}_historical_1d.csv format
- Easy retrieval and automated processing
- IST timezone-aware cutoffs and smart resume functionality
"""

import pandas as pd
import json
import time
import os
import re
import pytz
from datetime import datetime, timedelta
from fyers_semi_auto_login import FyersAutoLogin

class BulkDataDownloader:
    def __init__(self):
        self.fyers = FyersAutoLogin()
        self.client = None
        self.downloaded_count = 0
        self.failed_stocks = []
        self.success_stocks = []
        self.skipped_stocks = []
        self.resumed_stocks = []
    
    def format_symbol_for_api(self, raw_symbol):
        """Format symbol for Fyers API (NSE:SYMBOL-EQ) - only for API calls"""
        clean_symbol = raw_symbol.replace('NSE:', '').replace('-EQ', '').strip()
        return f'NSE:{clean_symbol}-EQ'
    
    def generate_standard_filename(self, raw_symbol):
        """Generate standardized filename using ONLY the raw symbol from CSV"""
        # Clean the raw symbol only - remove special chars, keep alphanumeric
        cleaned_symbol = re.sub(r'[^A-Za-z0-9&-]', '_', raw_symbol.strip())
        # Convert to uppercase and add standard suffix
        return f"{cleaned_symbol.upper()}_historical_1d.csv"
    
    def get_file_path(self, raw_symbol, output_dir="stock_data"):
        """Get standardized file path using raw symbol only"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        filename = self.generate_standard_filename(raw_symbol)
        return os.path.join(output_dir, filename)
    
    def get_ist_current_and_cutoff(self):
        """Get current IST date and determine max download date based on 17:30 cutoff"""
        IST = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.now(IST)
        
        # 17:30 IST cutoff time
        cutoff_time = now_ist.replace(hour=17, minute=30, second=0, microsecond=0)
        
        if now_ist < cutoff_time:
            # Before 5:30 PM IST - download data up to previous day
            max_download_date = now_ist.date() - timedelta(days=1)
            print(f"⏰ Before 17:30 IST - will download data up to {max_download_date}")
        else:
            # After 5:30 PM IST - include today's data
            max_download_date = now_ist.date()
            print(f"⏰ After 17:30 IST - will download data up to {max_download_date}")
        
        return now_ist.date(), max_download_date
    
    def check_existing_data_status(self, file_path, required_start_date, max_download_date):
        """Check existing data and determine what needs to be downloaded"""
        if not os.path.exists(file_path):
            return {
                'status': 'new_download',
                'next_start_date': required_start_date,
                'message': 'File does not exist - full download needed',
                'existing_records': 0
            }
        
        try:
            df = pd.read_csv(file_path, parse_dates=['date'])
            
            if len(df) == 0:
                return {
                    'status': 'new_download',
                    'next_start_date': required_start_date,
                    'message': 'Empty file - full download needed',
                    'existing_records': 0
                }
            
            last_date_in_file = df['date'].max().date()
            first_date_in_file = df['date'].min().date()
            
            # Check if data is up to date
            if last_date_in_file >= max_download_date:
                return {
                    'status': 'up_to_date',
                    'next_start_date': None,
                    'message': f'Up to date ({first_date_in_file} to {last_date_in_file})',
                    'existing_records': len(df)
                }
            
            # Check if we can resume from last date
            if last_date_in_file >= required_start_date.date():
                # Resume from day after last date
                next_start = last_date_in_file + timedelta(days=1)
                return {
                    'status': 'resume',
                    'next_start_date': next_start,
                    'message': f'Resume from {next_start} (last data: {last_date_in_file})',
                    'existing_records': len(df)
                }
            else:
                # File exists but doesn't have enough coverage
                return {
                    'status': 'new_download',
                    'next_start_date': required_start_date,
                    'message': f'Insufficient coverage - restart from {required_start_date.date()}',
                    'existing_records': len(df)
                }
                
        except Exception as e:
            return {
                'status': 'new_download',
                'next_start_date': required_start_date,
                'message': f'Error reading file: {str(e)}',
                'existing_records': 0
            }
    
    def append_to_existing_data(self, existing_file_path, new_data_df):
        """Append new data to existing CSV file"""
        try:
            # Read existing data
            existing_df = pd.read_csv(existing_file_path, parse_dates=['date'])
            
            # Combine data
            combined_df = pd.concat([existing_df, new_data_df], ignore_index=True)
            
            # Remove duplicates and sort
            combined_df = combined_df.drop_duplicates(subset=['date', 'symbol'])
            combined_df = combined_df.sort_values('date')
            
            # Save back to file
            combined_df.to_csv(existing_file_path, index=False)
            
            print(f"   📄 Appended {len(new_data_df)} new records to existing {len(existing_df)} records")
            print(f"   📊 Total records now: {len(combined_df)}")
            return len(combined_df)
            
        except Exception as e:
            print(f"   ❌ Error appending data: {str(e)}")
            # Fallback - save new data only
            new_data_df.to_csv(existing_file_path, index=False)
            return len(new_data_df)
    
    def initialize_client(self):
        """Initialize Fyers client with enhanced token management"""
        print("🚀 Initializing Fyers client...")
        self.client = self.fyers.get_fyers_client()
        if self.client:
            print("✅ Client initialized successfully!")
            return True
        else:
            print("❌ Failed to initialize client")
            return False
    
    def generate_date_ranges(self, start_date, end_date):
        """Generate 365-day date ranges between start_date and end_date"""
        max_days = 365
        ranges = []
        current_start = start_date
        
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=max_days), end_date)
            ranges.append({
                'start': current_start.strftime("%Y-%m-%d"),
                'end': current_end.strftime("%Y-%m-%d")
            })
            current_start = current_end + timedelta(days=1)
        
        return ranges
    
    def download_single_range(self, symbol, start_date, end_date):
        """Download data for a single date range"""
        try:
            response = self.client.history({
                "symbol": symbol,
                "resolution": "D",
                "date_format": "1",
                "range_from": start_date,
                "range_to": end_date,
                "cont_flag": "1"
            })
            
            if response['s'] == 'ok':
                return response['candles']
            else:
                print(f"   ⚠️  API error: {response.get('message', 'Unknown error')}")
                return None
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
            return None
    
    def download_stock_data(self, raw_symbol, stock_name, required_start_year):
        """Enhanced download with corrected filename generation using only raw symbol"""
        # Format symbol for API calls ONLY
        formatted_symbol = self.format_symbol_for_api(raw_symbol)
        
        # Use raw symbol for filename generation - KEY FIX!
        file_path = self.get_file_path(raw_symbol)  # Use raw_symbol, not formatted_symbol
        
        # Get current time and determine max download date
        current_ist_date, max_download_date = self.get_ist_current_and_cutoff()
        
        # Set required start date
        required_start_date = datetime(required_start_year, 1, 1)
        max_end_date = datetime.combine(max_download_date, datetime.min.time())
        
        # Check existing data status
        data_status = self.check_existing_data_status(file_path, required_start_date, max_download_date)
        
        print(f"\n📊 {raw_symbol} → API: {formatted_symbol}")
        print(f"   📂 File: {os.path.basename(file_path)}")  # Now shows: ABB_historical_1d.csv
        print(f"   📋 Status: {data_status['message']}")
        
        if data_status['status'] == 'up_to_date':
            print(f"   ✅ Skipping - Data is up to date ({data_status['existing_records']} records)")
            self.skipped_stocks.append((raw_symbol, stock_name, os.path.basename(file_path)))
            return "skipped"
        
        # Determine actual download range
        if data_status['status'] == 'resume':
            actual_start_date = datetime.combine(data_status['next_start_date'], datetime.min.time())
            is_resume = True
            print(f"   🔄 Resuming from {data_status['next_start_date']}")
            self.resumed_stocks.append((raw_symbol, stock_name, os.path.basename(file_path)))
        else:
            actual_start_date = required_start_date
            is_resume = False
            print(f"   📥 Full download from {actual_start_date.date()}")
        
        # Generate date ranges for download
        date_ranges = self.generate_date_ranges(actual_start_date, max_end_date)
        all_candles = []
        
        print(f"   📅 Will download {len(date_ranges)} date ranges to {max_download_date}")
        
        for i, date_range in enumerate(date_ranges):
            print(f"   📈 Range {i+1}/{len(date_ranges)}: {date_range['start']} to {date_range['end']}", end=" ")
            
            # Use FORMATTED symbol for API calls
            candles = self.download_single_range(
                formatted_symbol,  # API needs NSE:ABB-EQ format
                date_range['start'], 
                date_range['end']
            )
            
            if candles:
                all_candles.extend(candles)
                print(f"✅ {len(candles)} days")
            else:
                print("❌ Failed")
            
            #time.sleep(1.5)  # Rate limiting
        
        if all_candles:
            # Convert to DataFrame
            new_df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            new_df['date'] = pd.to_datetime(new_df['timestamp'], unit='s')
            new_df['symbol'] = formatted_symbol  # Store full API format
            new_df['raw_symbol'] = raw_symbol    # Store original CSV symbol
            new_df['stock_name'] = stock_name
            new_df = new_df.sort_values('date')
            new_df = new_df.drop_duplicates(subset=['date', 'symbol'])
            
            # Standardized column order
            new_df = new_df[['date', 'symbol', 'raw_symbol', 'stock_name', 'open', 'high', 'low', 'close', 'volume', 'timestamp']]
            
            # Save or append data
            if is_resume and data_status['existing_records'] > 0:
                total_records = self.append_to_existing_data(file_path, new_df)
                print(f"   🎉 Resume successful! Total records: {total_records}")
            else:
                new_df.to_csv(file_path, index=False)
                print(f"   🎉 Download successful! New records: {len(new_df)}")
            
            print(f"   📅 Latest data range: {new_df['date'].min().strftime('%Y-%m-%d')} to {new_df['date'].max().strftime('%Y-%m-%d')}")
            return "success"
        else:
            print(f"   ❌ No data retrieved")
            return "failed"
    
    def download_all_stocks(self, stocks_csv="stocks.csv", start_year=2010, batch_size=5):
        """Enhanced bulk download with standardized filenames using raw symbols only"""
        if not self.initialize_client():
            return False
        
        # Get current IST info
        current_ist_date, max_download_date = self.get_ist_current_and_cutoff()
        
        # Load stock list
        print(f"📋 Loading stocks from {stocks_csv}...")
        try:
            stocks_df = pd.read_csv(stocks_csv)
            stock_list = stocks_df[['Symbol', 'Stock Name']].values.tolist()
            print(f"✅ Loaded {len(stock_list)} stocks")
        except Exception as e:
            print(f"❌ Error loading stocks CSV: {e}")
            return False
        
        print(f"\n🚀 Enhanced bulk download with standardized filenames...")
        print(f"📅 Target range: {start_year}-01-01 to {max_download_date}")
        print(f"📁 Filename format: {{RAW_SYMBOL}}_historical_1d.csv")
        print(f"⏰ IST time-based cutoff applied")
        print(f"🔄 Smart resume enabled for partial downloads")
        print("=" * 70)
        
        for i, (raw_symbol, stock_name) in enumerate(stock_list):
            print(f"\n📈 [{i+1}/{len(stock_list)}] Processing {raw_symbol}")
            
            try:
                result = self.download_stock_data(raw_symbol, stock_name, start_year)
                
                if result == "success":
                    filename = self.generate_standard_filename(raw_symbol)
                    self.success_stocks.append((raw_symbol, stock_name, filename))
                    self.downloaded_count += 1
                elif result == "skipped":
                    pass  # Already added to appropriate list
                else:
                    self.failed_stocks.append((raw_symbol, stock_name))
                
                # Progress summary
                total_processed = len(self.success_stocks) + len(self.skipped_stocks) + len(self.failed_stocks)
                print(f"   📊 Progress: {total_processed}/{len(stock_list)} | New: {len(self.success_stocks)} | Resumed: {len(self.resumed_stocks)} | Skipped: {len(self.skipped_stocks)} | Failed: {len(self.failed_stocks)}")
                
                # Batch checkpoint
                if (i + 1) % batch_size == 0:
                    print(f"\n🔄 BATCH CHECKPOINT ({i+1}/{len(stock_list)}):")
                    print(f"   ✅ New downloads: {len(self.success_stocks)}")
                    print(f"   🔄 Resumed: {len(self.resumed_stocks)}")
                    print(f"   ⏭️  Skipped (up-to-date): {len(self.skipped_stocks)}")
                    print(f"   ❌ Failed: {len(self.failed_stocks)}")
                    print(f"   📊 Success rate: {(len(self.success_stocks) + len(self.skipped_stocks))/(i+1)*100:.1f}%")
                    print("   ⏸️  Rate limit pause (5 seconds)...")
                    #time.sleep(5)
                
            except Exception as e:
                print(f"   ❌ Exception for {raw_symbol}: {str(e)}")
                self.failed_stocks.append((raw_symbol, stock_name))
        
        # Create file index and master file
        self.create_file_index()
        self.create_master_file(start_year, max_download_date)
        
        # Final summary with file examples
        print(f"\n🎉 ENHANCED BULK DOWNLOAD COMPLETE!")
        print("=" * 50)
        print(f"✅ New downloads: {len(self.success_stocks)}")
        print(f"🔄 Resumed partial downloads: {len(self.resumed_stocks)}")
        print(f"⏭️  Skipped (up-to-date): {len(self.skipped_stocks)}")
        print(f"❌ Failed: {len(self.failed_stocks)}")
        print(f"📊 Total processed: {len(stock_list)}")
        print(f"⏰ Data downloaded up to: {max_download_date}")
        
        # Show some example filenames
        if self.success_stocks:
            print(f"\n📁 Example standardized filenames:")
            for i, (symbol, name, filename) in enumerate(self.success_stocks[:3]):
                print(f"   {symbol} → {filename}")
            if len(self.success_stocks) > 3:
                print(f"   ... and {len(self.success_stocks)-3} more")
        
        return True
    
    def create_file_index(self):
        """Create an index file mapping symbols to filenames for easy lookup"""
        index_data = []
        
        # Add successful downloads
        for raw_symbol, stock_name, filename in self.success_stocks:
            formatted_symbol = self.format_symbol_for_api(raw_symbol)
            index_data.append({
                'raw_symbol': raw_symbol,
                'formatted_symbol': formatted_symbol,
                'stock_name': stock_name,
                'filename': filename,
                'status': 'downloaded'
            })
        
        # Add skipped files
        for raw_symbol, stock_name, filename in self.skipped_stocks:
            formatted_symbol = self.format_symbol_for_api(raw_symbol)
            index_data.append({
                'raw_symbol': raw_symbol,
                'formatted_symbol': formatted_symbol,
                'stock_name': stock_name,
                'filename': filename,
                'status': 'skipped_up_to_date'
            })
        
        # Add resumed files
        for raw_symbol, stock_name, filename in self.resumed_stocks:
            formatted_symbol = self.format_symbol_for_api(raw_symbol)
            index_data.append({
                'raw_symbol': raw_symbol,
                'formatted_symbol': formatted_symbol,
                'stock_name': stock_name,
                'filename': filename,
                'status': 'resumed'
            })
        
        if index_data:
            index_df = pd.DataFrame(index_data)
            index_filename = f"file_index_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            index_df.to_csv(index_filename, index=False)
            
            print(f"\n📋 File index created: {index_filename}")
            print(f"   📊 Total entries: {len(index_df)}")
            print(f"   🔍 Use this file to easily find data files for specific stocks")
    
    def create_master_file(self, start_year, max_date):
        """Create/update master file from all individual stock files"""
        print(f"\n🔄 Creating master database file...")
        
        all_files = []
        stock_data_dir = "stock_data"
        
        if os.path.exists(stock_data_dir):
            for filename in os.listdir(stock_data_dir):
                if filename.endswith('_historical_1d.csv'):
                    all_files.append(os.path.join(stock_data_dir, filename))
        
        if all_files:
            print(f"   📂 Found {len(all_files)} standardized stock files")
            all_dataframes = []
            
            for file_path in all_files:
                try:
                    df = pd.read_csv(file_path, parse_dates=['date'])
                    all_dataframes.append(df)
                except Exception as e:
                    print(f"   ⚠️  Error reading {file_path}: {e}")
            
            if all_dataframes:
                master_df = pd.concat(all_dataframes, ignore_index=True)
                master_df = master_df.sort_values(['raw_symbol', 'date'])
                
                master_filename = f"master_historical_data_{start_year}_to_{max_date.year}_{max_date.month:02d}_{max_date.day:02d}.csv"
                master_df.to_csv(master_filename, index=False)
                
                print(f"   💾 Master file created: {master_filename}")
                print(f"   📊 Total records: {len(master_df):,}")
                print(f"   📈 Unique stocks: {master_df['raw_symbol'].nunique()}")
                print(f"   📅 Date range: {master_df['date'].min().date()} to {master_df['date'].max().date()}")
        else:
            print("   ⚠️  No stock files found for master creation")

# Helper functions for easy data access
def load_stock_data(symbol, data_dir="stock_data"):
    """Load historical data using simplified filename"""
    # Clean the symbol for filename lookup
    cleaned_symbol = re.sub(r'[^A-Za-z0-9]', '_', symbol.strip()).upper()
    filename = f"{cleaned_symbol}_historical_1d.csv"
    filepath = os.path.join(data_dir, filename)
    
    if os.path.exists(filepath):
        return pd.read_csv(filepath, parse_dates=['date'])
    else:
        print(f"❌ File not found: {filename}")
        return None

def list_available_stocks(data_dir="stock_data"):
    """List all available stocks with their filenames"""
    if not os.path.exists(data_dir):
        print(f"❌ Directory not found: {data_dir}")
        return []
    
    files = [f for f in os.listdir(data_dir) if f.endswith('_historical_1d.csv')]
    
    stock_info = []
    
    for filename in files:
        # Extract symbol from filename
        symbol = filename.replace('_historical_1d.csv', '')
        stock_info.append({
            'symbol': symbol,
            'filename': filename,
            'filepath': os.path.join(data_dir, filename)
        })
    
    return sorted(stock_info, key=lambda x: x['symbol'])

# Usage
def start_standardized_bulk_download():
    """Start bulk download with standardized filenames"""
    downloader = BulkDataDownloader()
    downloader.download_all_stocks(
        stocks_csv="stocks.csv",
        start_year=2010,
        batch_size=5
    )

if __name__ == "__main__":
    print("🚀 Starting Standardized Bulk Data Download...")
    print("📁 All files will be saved as {RAW_SYMBOL}_historical_1d.csv")
    print("   Examples: ABB → ABB_historical_1d.csv")
    print("            RELIANCE → RELIANCE_historical_1d.csv")
    start_standardized_bulk_download()
