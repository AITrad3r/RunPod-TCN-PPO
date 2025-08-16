import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DataQuery:
    def __init__(self,
                 vwap_window: int = 24,
                 rsi_period: int = 14,
                 use_bbands: bool = False,
                 bb_window: int = 20,
                 bb_k: float = 2.0):
        self.db_path = os.getenv('DATABASE_PATH', 'bitcoin_data.db')
        # Indicator configuration
        self.vwap_window = int(vwap_window)
        self.rsi_period = int(rsi_period)
        self.use_bbands = bool(use_bbands)
        self.bb_window = int(bb_window)
        self.bb_k = float(bb_k)
        
    def get_data(self, start_date=None, end_date=None, limit=None):
        """Get OHLCV data from database"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        
        # Detect if a 'datetime' column exists in the table
        try:
            has_datetime = bool(cur.execute("SELECT 1 FROM pragma_table_info('hourly_data') WHERE name='datetime'").fetchone())
        except Exception:
            has_datetime = False
        
        query = "SELECT * FROM hourly_data"
        conditions = []
        params = []
        order_col = 'timestamp'
        
        # Prefer filtering by ISO datetime strings to avoid timestamp unit ambiguity
        if has_datetime and (start_date or end_date):
            order_col = 'datetime'
            if start_date:
                if not isinstance(start_date, str):
                    start_date = datetime.fromtimestamp(start_date.timestamp()).strftime('%Y-%m-%d %H:%M:%S')
                conditions.append("datetime >= ?")
                params.append(start_date)
            if end_date:
                if not isinstance(end_date, str):
                    end_date = datetime.fromtimestamp(end_date.timestamp()).strftime('%Y-%m-%d %H:%M:%S')
                conditions.append("datetime <= ?")
                params.append(end_date)
        else:
            # Fallback: filter by numeric timestamp, auto-detect unit (s vs ms)
            ts_unit = 'ms'
            try:
                row = cur.execute("SELECT MAX(timestamp) FROM hourly_data").fetchone()
                max_ts = row[0] if row else None
                if max_ts is not None and max_ts < 1e11:
                    ts_unit = 's'  # seconds
                else:
                    ts_unit = 'ms'
            except Exception:
                ts_unit = 'ms'
            
            if start_date:
                if isinstance(start_date, str):
                    start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
                start_ts = int(start_date.timestamp() * (1000 if ts_unit == 'ms' else 1))
                conditions.append("timestamp >= ?")
                params.append(start_ts)
            if end_date:
                if isinstance(end_date, str):
                    end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
                end_ts = int(end_date.timestamp() * (1000 if ts_unit == 'ms' else 1))
                conditions.append("timestamp <= ?")
                params.append(end_ts)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += f" ORDER BY {order_col}"
        if limit:
            query += f" LIMIT {limit}"
        
        df = pd.read_sql_query(query, conn, params=params)
        
        # Ensure we have a proper pandas datetime column
        if 'datetime' in df.columns:
            # Parse if stored as string
            try:
                df['datetime'] = pd.to_datetime(df['datetime'])
            except Exception:
                # Fallback to timestamp parsing if needed
                unit_guess = 'ms'
                try:
                    row = cur.execute("SELECT MAX(timestamp) FROM hourly_data").fetchone()
                    max_ts = row[0] if row else None
                    if max_ts is not None and max_ts < 1e11:
                        unit_guess = 's'
                except Exception:
                    unit_guess = 'ms'
                if 'timestamp' in df.columns:
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit=unit_guess)
        elif 'timestamp' in df.columns:
            # Derive datetime from timestamp with unit detection
            unit_guess = 'ms'
            try:
                row = cur.execute("SELECT MAX(timestamp) FROM hourly_data").fetchone()
                max_ts = row[0] if row else None
                if max_ts is not None and max_ts < 1e11:
                    unit_guess = 's'
            except Exception:
                unit_guess = 'ms'
            df['datetime'] = pd.to_datetime(df['timestamp'], unit=unit_guess)
        
        conn.close()
        return df
    
    def get_recent_data(self, hours=96):
        """Get most recent N hours of data (default 96 for PPO model)"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        # Determine order column preference
        try:
            has_datetime = bool(cur.execute("SELECT 1 FROM pragma_table_info('hourly_data') WHERE name='datetime'").fetchone())
        except Exception:
            has_datetime = False
        order_col = 'datetime' if has_datetime else 'timestamp'
        query = f"SELECT * FROM hourly_data ORDER BY {order_col} DESC LIMIT {hours}"
        df = pd.read_sql_query(query, conn)
        # Determine timestamp unit for parsing if needed
        unit_guess = 'ms'
        try:
            row = cur.execute("SELECT MAX(timestamp) FROM hourly_data").fetchone()
            max_ts = row[0] if row else None
            if max_ts is not None and max_ts < 1e11:
                unit_guess = 's'
        except Exception:
            unit_guess = 'ms'
        conn.close()
        # Ensure datetime column exists
        if 'datetime' in df.columns:
            try:
                df['datetime'] = pd.to_datetime(df['datetime'])
            except Exception:
                if 'timestamp' in df.columns:
                    df['datetime'] = pd.to_datetime(df['timestamp'], unit=unit_guess)
        elif 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit=unit_guess)
        # Oldest first
        df = df.sort_values('datetime').reset_index(drop=True)
        return df
    
    def calculate_indicators(self, df):
        """Calculate VWAP, RSI, and optional Bollinger Bands indicators"""
        df = df.copy()
        
        # Calculate VWAP (Volume Weighted Average Price)
        df['typical_price'] = (df['high_price'] + df['low_price'] + df['close_price']) / 3
        df['vwap_numerator'] = df['typical_price'] * df['volume']
        df['vwap_denominator'] = df['volume']
        
        # Rolling VWAP calculation (configurable window)
        vw = max(1, int(self.vwap_window))
        df['vwap'] = df['vwap_numerator'].rolling(window=vw).sum() / df['vwap_denominator'].rolling(window=vw).sum()
        
        # Price relative to VWAP
        df['price_minus_vwap'] = df['close_price'] - df['vwap']
        df['above_vwap'] = (df['close_price'] > df['vwap']).astype(int)
        
        # Calculate RSI (configurable period)
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        rp = max(2, int(self.rsi_period))
        df['rsi'] = calculate_rsi(df['close_price'], window=rp)
        df['rsi_50_cross'] = ((df['rsi'] > 50) & (df['rsi'].shift(1) <= 50)).astype(int)

        # Optional Bollinger Bands
        if self.use_bbands:
            w = max(2, int(self.bb_window))
            k = float(self.bb_k)
            sma = df['close_price'].rolling(window=w).mean()
            std = df['close_price'].rolling(window=w).std()
            upper = sma + k * std
            lower = sma - k * std
            width = (upper - lower).replace(0, np.nan)
            # Store Bollinger band width (normalized by SMA to be scale-invariant)
            df['bb_width'] = ((upper - lower) / sma).replace([np.inf, -np.inf], np.nan)
            # %B: normalized position within bands
            df['bb_pctb'] = ((df['close_price'] - lower) / width).clip(0.0, 1.0)
        else:
            df['bb_pctb'] = np.nan  # placeholder for consistency
        
        return df
    
    def get_training_data(self, window_size=96):
        """Get data formatted for PPO training with indicators"""
        # Get all data
        df = self.get_data()
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Select features
        # Keep feature count constant (7). Use bb_pctb instead of rsi_50_cross when enabled.
        use_bb = bool(self.use_bbands)
        features = [
            'close_price',
            'volume', 
            'vwap',
            'price_minus_vwap',
            'above_vwap',
            'rsi',
            'bb_pctb' if use_bb else 'rsi_50_cross'
        ]
        
        # Remove rows with NaN values (from indicator calculations)
        df = df.dropna(subset=features)
        
        # Create sliding windows for training
        data_windows = []
        for i in range(window_size, len(df)):
            window_data = df.iloc[i-window_size:i][features].values
            data_windows.append(window_data)
        
        return np.array(data_windows), df.iloc[window_size:].reset_index(drop=True)
    
    def get_data_stats(self):
        """Get basic statistics about the data"""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        # Check if datetime column exists
        try:
            has_datetime = bool(cur.execute("SELECT 1 FROM pragma_table_info('hourly_data') WHERE name='datetime'").fetchone())
        except Exception:
            has_datetime = False
        if has_datetime:
            query = """
                SELECT 
                    COUNT(*) as total_records,
                    MIN(datetime) as earliest_date,
                    MAX(datetime) as latest_date,
                    MIN(close_price) as min_price,
                    MAX(close_price) as max_price,
                    AVG(close_price) as avg_price,
                    AVG(volume) as avg_volume
                FROM hourly_data
            """
            stats = pd.read_sql_query(query, conn)
            conn.close()
            return stats.iloc[0].to_dict()
        else:
            # Compute stats from timestamp with unit detection
            try:
                row = cur.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp), MIN(close_price), MAX(close_price), AVG(close_price), AVG(volume) FROM hourly_data").fetchone()
                total, min_ts, max_ts, min_p, max_p, avg_p, avg_v = row
            except Exception:
                conn.close()
                return {
                    'total_records': 0,
                    'earliest_date': None,
                    'latest_date': None,
                    'min_price': None,
                    'max_price': None,
                    'avg_price': None,
                    'avg_volume': None,
                }
            unit_guess = 'ms'
            if max_ts is not None and max_ts < 1e11:
                unit_guess = 's'
            earliest = pd.to_datetime(min_ts, unit=unit_guess) if min_ts is not None else None
            latest = pd.to_datetime(max_ts, unit=unit_guess) if max_ts is not None else None
            conn.close()
            return {
                'total_records': int(total or 0),
                'earliest_date': earliest.strftime('%Y-%m-%d %H:%M:%S') if earliest is not None else None,
                'latest_date': latest.strftime('%Y-%m-%d %H:%M:%S') if latest is not None else None,
                'min_price': float(min_p) if min_p is not None else None,
                'max_price': float(max_p) if max_p is not None else None,
                'avg_price': float(avg_p) if avg_p is not None else None,
                'avg_volume': float(avg_v) if avg_v is not None else None,
            }
    
    def export_to_csv(self, filename='bitcoin_hourly_data.csv', start_date=None, end_date=None):
        """Export data to CSV file"""
        df = self.get_data(start_date=start_date, end_date=end_date)
        df = self.calculate_indicators(df)
        
        # Select relevant columns
        export_columns = [
            'datetime', 'timestamp', 'open_price', 'high_price', 'low_price', 
            'close_price', 'volume', 'vwap', 'price_minus_vwap', 'above_vwap', 
            'rsi', 'rsi_50_cross'
        ]
        
        df[export_columns].to_csv(filename, index=False)
        print(f"Data exported to {filename}")
        return filename

def main():
    """Example usage"""
    dq = DataQuery()
    
    # Get basic stats
    stats = dq.get_data_stats()
    print("=== Database Statistics ===")
    print(f"Total records: {stats['total_records']:,}")
    print(f"Date range: {stats['earliest_date']} to {stats['latest_date']}")
    print(f"Price range: ${stats['min_price']:,.2f} - ${stats['max_price']:,.2f}")
    print(f"Average price: ${stats['avg_price']:,.2f}")
    print(f"Average volume: {stats['avg_volume']:,.2f}")
    
    # Get recent data with indicators
    print("\n=== Recent Data Sample (last 5 records) ===")
    recent_data = dq.get_recent_data(hours=5)
    recent_with_indicators = dq.calculate_indicators(recent_data)
    
    display_cols = ['datetime', 'close_price', 'volume', 'vwap', 'rsi']
    print(recent_with_indicators[display_cols].to_string(index=False))
    
    # Get training data shape
    print("\n=== Training Data Info ===")
    windows, metadata = dq.get_training_data(window_size=96)
    print(f"Training windows shape: {windows.shape}")
    print(f"Features per timestep: {windows.shape[2]}")
    print(f"Total training samples: {windows.shape[0]}")
    
    print("\nData is ready for PPO training!")

if __name__ == "__main__":
    main()