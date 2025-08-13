import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DataQuery:
    def __init__(self):
        self.db_path = os.getenv('DATABASE_PATH', 'bitcoin_data.db')
        
    def get_data(self, start_date=None, end_date=None, limit=None):
        """Get OHLCV data from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM hourly_data"
        conditions = []
        params = []
        
        if start_date:
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
            start_ts = int(start_date.timestamp() * 1000)
            conditions.append("timestamp >= ?")
            params.append(start_ts)
            
        if end_date:
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
            end_ts = int(end_date.timestamp() * 1000)
            conditions.append("timestamp <= ?")
            params.append(end_ts)
            
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
            
        query += " ORDER BY timestamp"
        
        if limit:
            query += f" LIMIT {limit}"
            
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        return df
    
    def get_recent_data(self, hours=96):
        """Get most recent N hours of data (default 96 for PPO model)"""
        conn = sqlite3.connect(self.db_path)
        
        query = f"""
            SELECT * FROM hourly_data 
            ORDER BY timestamp DESC 
            LIMIT {hours}
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Convert timestamp and reverse order (oldest first)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def calculate_indicators(self, df):
        """Calculate VWAP and RSI indicators as specified in the PPO spec"""
        df = df.copy()
        
        # Calculate VWAP (Volume Weighted Average Price)
        df['typical_price'] = (df['high_price'] + df['low_price'] + df['close_price']) / 3
        df['vwap_numerator'] = df['typical_price'] * df['volume']
        df['vwap_denominator'] = df['volume']
        
        # Rolling VWAP calculation (24-hour window)
        window = 24
        df['vwap'] = df['vwap_numerator'].rolling(window=window).sum() / df['vwap_denominator'].rolling(window=window).sum()
        
        # Price relative to VWAP
        df['price_minus_vwap'] = df['close_price'] - df['vwap']
        df['above_vwap'] = (df['close_price'] > df['vwap']).astype(int)
        
        # Calculate RSI(14)
        def calculate_rsi(prices, window=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df['rsi'] = calculate_rsi(df['close_price'])
        df['rsi_50_cross'] = ((df['rsi'] > 50) & (df['rsi'].shift(1) <= 50)).astype(int)
        
        return df
    
    def get_training_data(self, window_size=96):
        """Get data formatted for PPO training with indicators"""
        # Get all data
        df = self.get_data()
        
        # Calculate indicators
        df = self.calculate_indicators(df)
        
        # Select features as specified in PPO spec
        features = [
            'close_price',
            'volume', 
            'vwap',
            'price_minus_vwap',
            'above_vwap',
            'rsi',
            'rsi_50_cross'
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