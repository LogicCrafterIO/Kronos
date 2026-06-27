import os
import sys
import numpy as np
import pandas as pd
import torch
import torch_directml
import yfinance as yf
from huggingface_hub import snapshot_download

# Enforce local directory imports from Kronos root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 1. ACCELERATED HARDWARE SELECTION
device = torch_directml.device()
print(f"DirectML active. Compiling tensor layers via AMD Compute Array.")

# 2. LOAD KRONOS-BASE PIPELINE
print("Validating Kronos-base foundation paths...")
model_dir = snapshot_download(repo_id="NeoQuasar/Kronos-base")

from model import Kronos, KronosTokenizer, KronosPredictor

model = Kronos.from_pretrained(model_dir)
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
predictor = KronosPredictor(model, tokenizer, max_context=512)

# 3. EXTRACT LIVE MARKET STREAM FROM YFINANCE
print("Downloading live 5-minute bars for MNQ Futures...")
# MNQ=F is the continuous contract ticker for Micro E-mini Nasdaq-100 Futures
ticker = "MNQ=F"
raw_data = yf.download(tickers=ticker, period="5d", interval="5m")

if raw_data.empty:
    raise ValueError(f"Could not retrieve data for {ticker}. Ensure internet connection is active.")

# Clean up MultiIndex columns if present in newer yfinance versions
if isinstance(raw_data.columns, pd.MultiIndex):
    raw_data.columns = raw_data.columns.get_level_values(0)

raw_data = raw_data.reset_index()
raw_data.rename(columns={'Datetime': 'timestamps', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)

# 4. QUANT RE-STRUCTURING: CALCULATE FUTURES NOTIONAL TURNOVER
# Kronos needs an 'amount' column (Dollar/Fiat Volume).
# For MNQ, each point is worth $2. Notional Turnover = Volume * Close * $2
raw_data['amount'] = raw_data['volume'] * raw_data['close'] * 2.0

# Drop any incomplete rows or periods with structural gaps
raw_data = raw_data.dropna(subset=['open', 'high', 'low', 'close', 'volume', 'amount'])

# Slice out the exact tail context matching the max_context lookback window
context_data = raw_data.tail(360).copy()

# 5. PREPARE EXECUTION WINDOW FOR TARGET HORIZON
# We will project the next 1 hour of market execution (12 consecutive 5-minute candles)
pred_len = 12
last_timestamp = context_data['timestamps'].iloc[-1]
future_timestamps = pd.date_range(start=last_timestamp + pd.Timedelta(minutes=5), periods=pred_len, freq='5min')

# Split structures cleanly for the engine wrappers
x_df = context_data[['open', 'high', 'low', 'close', 'volume', 'amount']].reset_index(drop=True)
x_timestamp = context_data['timestamps'].reset_index(drop=True)
y_timestamp = pd.Series(future_timestamps)

# 6. COMPUTE INFERENCE GENERATION
print(f"Feeding {len(x_df)} bars of context into Kronos Autoregressive Engine...")

pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=0.75,         # Temperature scale balance (Lower = more tight/structural consistency)
    top_p=0.85,       # Nucleus sampling boundary limit
    sample_count=1,
    verbose=False
)

# Bind the calculated index tracking array back to regular times
pred_df.index = future_timestamps

# 7. PRINT STRUCTURAL PREDICTION MATRIX
print("\n=== LIVE MNQ FUTURE 1-HOUR FORWARD CANDLESTICK PROJECTIONS ===")
print(pred_df[['open', 'high', 'low', 'close', 'volume']].head(pred_len))