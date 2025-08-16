from data_query import DataQuery
from trading_environment import GridTradingEnvironment

print("=== DataQuery Stats ===")
dq = DataQuery()
print(dq.get_data_stats())

print("\n=== Recent 48h Sample (tail) ===")
recent = dq.get_recent_data(hours=48)
print(recent.tail().to_string(index=False))
print(f"Recent rows: {len(recent)}")

print("\n=== Environment Load ===")
env = GridTradingEnvironment(dq)
env.load_data()
print(f"Loaded rows: {len(env.data)} | max_steps: {env.max_steps}")
print("First 3 datetimes:")
try:
    print(env.data['datetime'].head(3).to_list())
except Exception:
    print("datetime column not present")
print("Last 3 datetimes:")
try:
    print(env.data['datetime'].tail(3).to_list())
except Exception:
    print("datetime column not present")
