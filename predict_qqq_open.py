import os
import sys
import datetime
import numpy as np
import pandas as pd
import torch
import torch_directml
import yfinance as yf
from huggingface_hub import snapshot_download

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 1. BIND TO AMD DIRECTML ENGINE
device = torch_directml.device()
print("DirectML Compute Node compiled successfully.")

# 2. SOURCE KRONOS BASE ENGINE
print("Initializing Kronos-base context graph...")
model_dir = snapshot_download(repo_id="NeoQuasar/Kronos-base")

from model import Kronos, KronosTokenizer, KronosPredictor
model = Kronos.from_pretrained(model_dir)
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
predictor = KronosPredictor(model, tokenizer, max_context=512)

# 3. LIVE QQQ MARKET DATA HARVESTING
print("Pulling intraday tracking sequences for QQQ ETF...")
ticker = "QQQ"
# Pulling 14 days guarantees we get 360+ pristine clean RTH bars after filtering weekends
raw_data = yf.download(tickers=ticker, period="14d", interval="5m")

if isinstance(raw_data.columns, pd.MultiIndex):
    raw_data.columns = raw_data.columns.get_level_values(0)

raw_data = raw_data.reset_index()
raw_data.rename(columns={'Datetime': 'timestamps', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)

# 4. TRIMMING EXTENDED HOURS (RTH Boundary Isolation)
# Convert timestamps to US Eastern Time to accurately slice the 09:30 to 16:00 window
raw_data['est_time'] = raw_data['timestamps'].dt.tz_convert('America/New_York')

# Filter for regular trading session only
rth_data = raw_data[
    (raw_data['est_time'].dt.time >= datetime.time(9, 30)) & 
    (raw_data['est_time'].dt.time <= datetime.time(16, 0))
].copy()

# Calculate standard equity capitalization turnover: Amount = Volume * Close
rth_data['amount'] = rth_data['volume'] * rth_data['close']
rth_data = rth_data.dropna(subset=['open', 'high', 'low', 'close', 'volume', 'amount'])

# Slice the trailing historical footprint to feed the Transformer
context_data = rth_data.tail(360).copy()

# 5. GENERATING FUTURE TIME STEPS FOR THE FIRST 2 HOURS SGT
# We define the upcoming opening sequence: 2 hours = 24 five-minute candles
pred_len = 24

# Set target calculation for the next opening bell day (detecting if today is weekend)
last_est = context_data['est_time'].iloc[-1]
if last_est.weekday() >= 4:  # If the last data point was Friday
    # Project target for Monday morning open
    next_open_day = last_est + pd.Timedelta(days=(7 - last_est.weekday()))
else:
    next_open_day = last_est + pd.Timedelta(days=1)

# Frame target execution times inside Eastern Standard trading blocks
target_start = datetime.datetime.combine(next_open_day.date(), datetime.time(9, 35))
future_est_times = pd.date_range(start=target_start, periods=pred_len, freq='5min', tz='America/New_York')
# Convert target outputs back to Singapore Local Time (SGT) for monitoring
future_sgt_times = future_est_times.tz_convert('Asia/Singapore')

# 6. ENCODING VECTOR AND EXECUTING FORECAST
x_df = context_data[['open', 'high', 'low', 'close', 'volume', 'amount']].reset_index(drop=True)
x_timestamp = context_data['timestamps'].reset_index(drop=True)
y_timestamp = pd.Series(future_est_times.tz_localize(None)) # Match local tracking format

print(f"Feeding {len(x_df)} pure RTH candlesticks into autoregressive sequence space...")
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp.dt.tz_localize(None),
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=0.7, 
    top_p=0.85,
    sample_count=1,
    verbose=False
)

# Align tracking frames back to local Singapore Time
pred_df.index = future_sgt_times

# 7. LOG OUTPUT MATRIX
print("\n=== PROJECTED QQQ OPENING SELECTION (FIRST 2 HOURS - SINGAPORE TIME) ===")
# Display columns with explicit SGT index map
print(pred_df[['open', 'high', 'low', 'close', 'volume']].head(pred_len))