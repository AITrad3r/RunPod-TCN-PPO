import os
import sqlite3
import pandas as pd

DB = os.getenv("DATABASE_PATH", "bitcoin_data.db")
print("DB Path:", os.path.abspath(DB))

if not os.path.exists(DB):
    print("ERROR: Database file does not exist")
    raise SystemExit(1)

conn = sqlite3.connect(DB)
cur = conn.cursor()

# Schema
try:
    ti = cur.execute("PRAGMA table_info(\"hourly_data\")").fetchall()
    print("\nSchema for hourly_data (cid, name, type, notnull, dflt_value, pk):")
    for col in ti:
        print(col)
except Exception as e:
    print("Failed to read schema:", e)

# Datetime stats
try:
    row = cur.execute("SELECT COUNT(*), MIN(datetime), MAX(datetime) FROM hourly_data").fetchone()
    print("\nCount, min(datetime), max(datetime):", row)
except Exception as e:
    print("Datetime stats err:", e)

# Timestamp stats
try:
    row = cur.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM hourly_data").fetchone()
    total, min_ts, max_ts = row
    unit = "s" if (max_ts is not None and max_ts < 1e11) else "ms"
    print("\nCount, min(timestamp), max(timestamp):", row, " unit_guess=", unit)
    if min_ts is not None:
        earliest = pd.to_datetime(min_ts, unit=unit)
        latest = pd.to_datetime(max_ts, unit=unit)
        print("Earliest (from timestamp):", earliest)
        print("Latest   (from timestamp):", latest)
except Exception as e:
    print("Timestamp stats err:", e)

# Latest rows
df = None
try:
    df = pd.read_sql_query("SELECT * FROM hourly_data ORDER BY datetime DESC LIMIT 10", conn)
    sort_col = "datetime"
except Exception:
    df = pd.read_sql_query("SELECT * FROM hourly_data ORDER BY timestamp DESC LIMIT 10", conn)
    sort_col = "timestamp"

print("\nLatest 10 rows (ascending by", sort_col, "):")
try:
    print(df.sort_values(sort_col).to_string(index=False))
except Exception as e:
    print("Could not print sample rows:", e)

conn.close()
