"""
Enhanced Bulk Data Downloader with Standardized File Naming
- Files saved as {RAW_SYMBOL}_historical_1d.csv format
- Easy retrieval and automated processing
- IST timezone-aware cutoffs and smart resume functionality
"""

import pandas as pd
import os
import re
import pytz
from datetime import datetime, timedelta
from fyers_semi_auto_login import FyersAutoLogin
import time
import sys
from apply_technical_indicators import apply_technical_indicators
import shutil


class BulkDataDownloader:
    def __init__(self):
        self.fyers = FyersAutoLogin()
        self.client = None
        self.downloaded_count = 0
        self.failed_stocks = []
        self.success_stocks = []
        self.skipped_stocks = []
        self.resumed_stocks = []

    # ---------------- SYMBOL + FILE HANDLING ----------------
    def format_symbol_for_api(self, raw_symbol):
        """Format symbol for Fyers API (NSE:SYMBOL-EQ)"""
        clean_symbol = raw_symbol.replace("NSE:", "").replace("-EQ", "").strip()
        return f"NSE:{clean_symbol}-EQ"

    def generate_standard_filename(self, raw_symbol):
        """Generate standardized filename using only raw symbol"""
        cleaned_symbol = re.sub(r"[^A-Za-z0-9&-]", "_", raw_symbol.strip())
        return f"{cleaned_symbol.upper()}_historical_1d.csv"

    def get_file_path(self, raw_symbol, output_dir="stock_data"):
        """Get standardized file path"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        return os.path.join(output_dir, self.generate_standard_filename(raw_symbol))

    # ---------------- DATE + CUTOFF HANDLING ----------------
    def get_ist_current_and_cutoff(self):
        """Get current IST date and determine max download date"""
        IST = pytz.timezone("Asia/Kolkata")
        now_ist = datetime.now(IST)

        cutoff_time = now_ist.replace(hour=17, minute=30, second=0, microsecond=0)
        if now_ist < cutoff_time:
            max_download_date = now_ist.date() - timedelta(days=1)
            print(f"⏰ Before 17:30 IST - will download data up to {max_download_date}")
        else:
            max_download_date = now_ist.date()
            print(f"⏰ After 17:30 IST - will download data up to {max_download_date}")

        return now_ist.date(), max_download_date

    # ---------------- DATA FILE STATUS ----------------
    def check_existing_data_status(self, file_path, required_start_date, max_download_date):
        """Check existing CSV file and decide what to do"""
        if not os.path.exists(file_path):
            return {"status": "new_download", "next_start_date": required_start_date,
                    "message": "File does not exist - full download needed", "existing_records": 0}

        try:
            df = pd.read_csv(file_path, parse_dates=["date"])
            if len(df) == 0:
                return {"status": "new_download", "next_start_date": required_start_date,
                        "message": "Empty file - full download needed", "existing_records": 0}

            last_date_in_file = df["date"].max().date()
            first_date_in_file = df["date"].min().date()

            if last_date_in_file >= max_download_date:
                return {"status": "up_to_date", "next_start_date": None,
                        "message": f"Up to date ({first_date_in_file} to {last_date_in_file})",
                        "existing_records": len(df)}

            if last_date_in_file >= required_start_date.date():
                next_start = last_date_in_file + timedelta(days=1)
                return {"status": "resume", "next_start_date": next_start,
                        "message": f"Resume from {next_start} (last data: {last_date_in_file})",
                        "existing_records": len(df)}

            return {"status": "new_download", "next_start_date": required_start_date,
                    "message": f"Insufficient coverage - restart from {required_start_date.date()}",
                    "existing_records": len(df)}

        except Exception as e:
            return {"status": "new_download", "next_start_date": required_start_date,
                    "message": f"Error reading file: {str(e)}", "existing_records": 0}

    def append_to_existing_data(self, existing_file_path, new_data_df):
        """Append new data to existing CSV"""
        try:
            existing_df = pd.read_csv(existing_file_path, parse_dates=["date"])
            combined_df = pd.concat([existing_df, new_data_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["date", "symbol"]).sort_values("date")
            combined_df.to_csv(existing_file_path, index=False)
            print(f"   📄 Appended {len(new_data_df)} new records (total {len(combined_df)})")
            return len(combined_df)
        except Exception as e:
            print(f"   ❌ Error appending data: {str(e)}")
            new_data_df.to_csv(existing_file_path, index=False)
            return len(new_data_df)

    # ---------------- CLIENT + DOWNLOAD ----------------
    def initialize_client(self):
        print("🚀 Initializing Fyers client...")
        self.client = self.fyers.get_fyers_client()
        if self.client:
            print("✅ Client initialized successfully!")
            return True
        print("❌ Failed to initialize client")
        return False

    def generate_date_ranges(self, start_date, end_date):
        """Break down into 365-day ranges"""
        ranges, current_start = [], start_date
        while current_start < end_date:
            current_end = min(current_start + timedelta(days=365), end_date)
            ranges.append({"start": current_start.strftime("%Y-%m-%d"),
                           "end": current_end.strftime("%Y-%m-%d")})
            current_start = current_end + timedelta(days=1)
        return ranges

    def download_single_range(self, symbol, start_date, end_date):
        try:
            response = self.client.history({
                "symbol": symbol, "resolution": "D", "date_format": "1",
                "range_from": start_date, "range_to": end_date, "cont_flag": "1"
            })
            #time.sleep()
            if response["s"] == "ok":
                return response["candles"]
            print(f"   ⚠️ API error: {response.get('message')}")
            return None
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")
            return None

    def download_stock_data(self, raw_symbol, required_start_year):
        """Download + save a single stock"""
        formatted_symbol = self.format_symbol_for_api(raw_symbol)
        file_path = self.get_file_path(raw_symbol)

        _, max_download_date = self.get_ist_current_and_cutoff()
        required_start_date = datetime(required_start_year, 1, 1)
        max_end_date = datetime.combine(max_download_date, datetime.min.time())

        data_status = self.check_existing_data_status(file_path, required_start_date, max_download_date)

        print(f"\n📊 {raw_symbol} → API: {formatted_symbol}")
        print(f"   📂 File: {os.path.basename(file_path)}")
        print(f"   📋 Status: {data_status['message']}")

        if data_status["status"] == "up_to_date":
            print(f"   ✅ Skipping (already up to date)")
            self.skipped_stocks.append((raw_symbol, os.path.basename(file_path)))
            return "skipped"

        if data_status["status"] == "resume":
            actual_start_date = datetime.combine(data_status["next_start_date"], datetime.min.time())
            is_resume = True
            print(f"   🔄 Resuming from {data_status['next_start_date']}")
            self.resumed_stocks.append((raw_symbol, os.path.basename(file_path)))
        else:
            actual_start_date, is_resume = required_start_date, False
            print(f"   📥 Full download from {actual_start_date.date()}")

        date_ranges = self.generate_date_ranges(actual_start_date, max_end_date)

        all_candles = []

        # If start and end date is equal
        if actual_start_date==max_end_date:
            #sys.exit()
            print(f"   📅 Will download letes eod data for date: {max_end_date}")
            candles = self.download_single_range(formatted_symbol, 
                                                 start_date=actual_start_date.strftime("%Y-%m-%d"), 
                                                 end_date=max_end_date.strftime("%Y-%m-%d"))

            if candles:
                all_candles = candles  # Assign (not extend) for single day
                print(f"✅ Got {len(candles)} records")
            else:
                print("❌ No data retrieved")
                return "failed"
        else:
            print(f"   📅 Will download {len(date_ranges)} ranges up to {max_download_date}")

            for i, dr in enumerate(date_ranges):
                print(f"   📈 Range {i+1}/{len(date_ranges)}: {dr['start']} → {dr['end']}", end=" ")
                candles = self.download_single_range(formatted_symbol, dr["start"], dr["end"])

                if candles:
                    all_candles.extend(candles)
                    print(f"✅ {len(candles)} days")
                else:
                    print("❌ Failed")

        if not all_candles:
            print("   ❌ No data retrieved")
            return "failed"

        new_df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        new_df["date"] = pd.to_datetime(new_df["timestamp"], unit="s")
        new_df["symbol"] = formatted_symbol
        new_df["raw_symbol"] = raw_symbol
        new_df = new_df.drop_duplicates(subset=["date", "symbol"]).sort_values("date")

        new_df = new_df[["date", "symbol", "raw_symbol", "open", "high", "low", "close", "volume", "timestamp"]]

        if is_resume and data_status["existing_records"] > 0:
            total_records = self.append_to_existing_data(file_path, new_df)
            print(f"   🎉 Resume successful! Total records: {total_records}")
        else:
            new_df.to_csv(file_path, index=False)
            print(f"   🎉 Download successful! Records: {len(new_df)}")

        print(f"   📅 Range: {new_df['date'].min().date()} → {new_df['date'].max().date()}")
        return "success"

    # ---------------- BULK DOWNLOAD ----------------
    def download_all_stocks(self, stocks_csv="stocks.csv", start_year=2010, batch_size=5):
        if not self.initialize_client():
            return False

        _, max_download_date = self.get_ist_current_and_cutoff()

        print(f"📋 Loading stocks from {stocks_csv}...")
        try:
            stocks_df = pd.read_csv(stocks_csv)
            stock_list = stocks_df["Symbol"].tolist()
            print(f"✅ Loaded {len(stock_list)} stocks")
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return False

        print(f"\n🚀 Bulk download started")
        print(f"📅 Target: {start_year}-01-01 → {max_download_date}")
        print(f"📁 File format: {{RAW_SYMBOL}}_historical_1d.csv")
        print("=" * 60)

        for i, raw_symbol in enumerate(stock_list):
            print(f"\n📈 [{i+1}/{len(stock_list)}] {raw_symbol}")
            try:
                result = self.download_stock_data(raw_symbol, start_year)
                #time.sleep(5)
                if result == "success":
                    filename = self.generate_standard_filename(raw_symbol)
                    self.success_stocks.append((raw_symbol, filename))
                    self.downloaded_count += 1
                elif result == "skipped":
                    pass
                else:
                    self.failed_stocks.append((raw_symbol,))
            except Exception as e:
                print(f"   ❌ Exception: {str(e)}")
                self.failed_stocks.append((raw_symbol,))

            if (i + 1) % batch_size == 0:
                print(f"\n🔄 Batch checkpoint ({i+1}/{len(stock_list)})")
                print(f"   ✅ New: {len(self.success_stocks)} | 🔄 Resumed: {len(self.resumed_stocks)} | ⏭️ Skipped: {len(self.skipped_stocks)} | ❌ Failed: {len(self.failed_stocks)}")
                #time.sleep(2)

        self.create_file_index()
        self.create_master_file(start_year, max_download_date)

        print(f"\n🎉 BULK DOWNLOAD COMPLETE")
        print(f"✅ New: {len(self.success_stocks)}")
        print(f"🔄 Resumed: {len(self.resumed_stocks)}")
        print(f"⏭️ Skipped: {len(self.skipped_stocks)}")
        print(f"❌ Failed: {len(self.failed_stocks)}")
        return True

    # ---------------- FILE INDEX + MASTER ----------------
    def create_file_index(self):
        index_data = []
        for raw_symbol, filename in self.success_stocks:
            index_data.append({"raw_symbol": raw_symbol,
                               "formatted_symbol": self.format_symbol_for_api(raw_symbol),
                               "filename": filename, "status": "downloaded"})
        for raw_symbol, filename in self.skipped_stocks:
            index_data.append({"raw_symbol": raw_symbol,
                               "formatted_symbol": self.format_symbol_for_api(raw_symbol),
                               "filename": filename, "status": "skipped"})
        for raw_symbol, filename in self.resumed_stocks:
            index_data.append({"raw_symbol": raw_symbol,
                               "formatted_symbol": self.format_symbol_for_api(raw_symbol),
                               "filename": filename, "status": "resumed"})

        if index_data:
            index_df = pd.DataFrame(index_data)
            idx_file = f"file_index_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            index_df.to_csv(idx_file, index=False)
            print(f"\n📋 File index created: {idx_file}")

    def create_master_file(self, start_year, max_date):
        print(f"\n🔄 Creating master file...")
        stock_dir = "stock_data"
        all_files = [os.path.join(stock_dir, f) for f in os.listdir(stock_dir)
                     if f.endswith("_historical_1d.csv")] if os.path.exists(stock_dir) else []

        if not all_files:
            print("   ⚠️ No stock files found")
            return

        dfs = []
        for fp in all_files:
            try:
                dfs.append(pd.read_csv(fp, parse_dates=["date"]))
            except Exception as e:
                print(f"   ⚠️ Error {fp}: {e}")

        if dfs:
            master_df = pd.concat(dfs, ignore_index=True).sort_values(["raw_symbol", "date"])
            master_file = f"master_historical_data_{start_year}_to_{max_date}.csv"
            master_df.to_csv(master_file, index=False)
            print(f"   💾 Master file: {master_file} ({len(master_df)} records)")


# ---------------- HELPER FUNCTIONS ----------------
def load_stock_data(symbol, data_dir="stock_data"):
    cleaned_symbol = re.sub(r"[^A-Za-z0-9]", "_", symbol.strip()).upper()
    filename = f"{cleaned_symbol}_historical_1d.csv"
    filepath = os.path.join(data_dir, filename)
    return pd.read_csv(filepath, parse_dates=["date"]) if os.path.exists(filepath) else None


def list_available_stocks(data_dir="stock_data"):
    if not os.path.exists(data_dir):
        return []
    files = [f for f in os.listdir(data_dir) if f.endswith("_historical_1d.csv")]
    return [{"symbol": f.replace("_historical_1d.csv", ""), "filename": f,
             "filepath": os.path.join(data_dir, f)} for f in files]


# ---------------- USAGE ----------------
def start_standardized_bulk_download():
    downloader = BulkDataDownloader()
    downloader.download_all_stocks(stocks_csv="all_stocks.csv", start_year=2010, batch_size=5)


if __name__ == "__main__":
    print("🚀 Starting Standardized Bulk Data Download...")
    start_standardized_bulk_download()
    folder_path = "analysis_data/darvas_5"

    try:
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' and its contents deleted successfully.")
    except OSError as e:
        print(f"Error deleting folder: {e}")

    apply_technical_indicators.create_analysis_with_indicators('darvas_5')

    ''' downloader = BulkDataDownloader()
    if downloader.initialize_client():
        result = downloader.download_stock_data("CEMPRO",required_start_year=2010)
        print(result)'''