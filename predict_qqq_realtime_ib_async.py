import os
import sys
import asyncio
import pandas as pd
import torch
import torch_directml
from ib_async import IB, Stock, util
from huggingface_hub import snapshot_download

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 1. BIND TO AMD DIRECTML ENGINE
device = torch_directml.device()
print(f"DirectML active. Compiling tensor layers via AMD Compute Array.")

# 2. SOURCE KRONOS BASE ENGINE
print("Initializing Kronos-base foundation models...")
model_dir = snapshot_download(repo_id="NeoQuasar/Kronos-base")

from model import Kronos, KronosTokenizer, KronosPredictor
model = Kronos.from_pretrained(model_dir)
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
predictor = KronosPredictor(model, tokenizer, max_context=512)

# Global variables to track the rolling data array
bars_data = None 

def run_kronos_forecast(df):
    """Encodes the live matrix and generates a forward rolling 2-hour prediction."""
    # QQQ trades 2 hours = 24 five-minute candles
    pred_len = 24 
    
    # Ensure correct format for Kronos input
    x_df = df[['open', 'high', 'low', 'close', 'volume', 'amount']].tail(360).reset_index(drop=True)
    x_timestamp = df['date'].tail(360).reset_index(drop=True)
    
    # Calculate real-time target timestamps starting from the last confirmed candle
    last_timestamp = x_timestamp.iloc[-1]
    future_timestamps = pd.date_range(
        start=last_timestamp + pd.Timedelta(minutes=5), 
        periods=pred_len, 
        freq='5min'
    )
    
    # Run the autoregressive transformer forward pass
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp.dt.tz_localize(None),
        y_timestamp=pd.Series(future_timestamps),
        pred_len=pred_len,
        T=0.7, 
        top_p=0.85,
        sample_count=1,
        verbose=False
    )
    
    # Convert timestamps to local Singapore Time (SGT) for clear real-time monitoring
    # To this:
    pred_df.index = future_timestamps.tz_convert('Asia/Singapore')
    
    print(f"\n================= ROLLING 2-HOUR KRONOS FORECAST =================")
    print(f"Latest Bar Fed: {last_timestamp.strftime('%Y-%m-%d %H:%M:%S')} EST")
    print(f"Current Local Execution Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')} SGT")
    print("------------------------------------------------------------------")
    print(pred_df[['open', 'high', 'low', 'close', 'volume']].head(pred_len))
    print("==================================================================\n")

def on_bar_update(bars, has_new_bar):
    """Callback trigger that runs automatically every time a live 5-min bar closes."""
    global bars_data
    
    # Update our main DataFrame with the fresh IBKR data cache
    bars_data = util.df(bars)
    bars_data['amount'] = bars_data['volume'] * bars_data['close']
    
    # Only recalculate and output when a 5-minute candle fully locks/closes
    if has_new_bar:
        print("🔔 Live 5-minute bar closed. Recalibrating transformer context...")
        run_kronos_forecast(bars_data)

async def main():
    ib = IB()
    print("Connecting to live Interactive Brokers session...")
    await ib.connectAsync(host="127.0.0.1", port=7497, clientId=15)
    
    contract = Stock(symbol="QQQ", exchange="SMART", currency="USD")
    await ib.qualifyContractsAsync(contract)
    
    print("Sourcing historical context & subscribing to live streaming bars...")
    # keepUpToDate=True instructs the IB server to keep this data collection alive in memory.
    # useRTH=True isolates regular cash trading hours.
    bars = await ib.reqHistoricalDataAsync(
        contract,
        endDateTime="",
        durationStr="5 D",
        barSizeSetting="5 mins",
        whatToShow="TRADES",
        useRTH=True,
        keepUpToDate=True
    )
    
    # Initialize the baseline dataset and run the initial prediction
    global bars_data
    bars_data = util.df(bars)
    bars_data['amount'] = bars_data['volume'] * bars_data['close']
    run_kronos_forecast(bars_data)
    
    # Attach our custom callback event to the active IB live-stream object
    bars.updateEvent += on_bar_update
    
    print("Engine active. Script is running in live-listening loop. Press Ctrl+C to terminate.")
    while True:
        await asyncio.sleep(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSession terminated by user. Exiting gracefully.")