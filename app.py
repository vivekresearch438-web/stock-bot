import os
import io
import json
import time
import threading
from datetime import datetime
import pytz
import requests
import pandas as pd
import yfinance as yf
import mplfinance as mpf
from PIL import Image
import gradio as gr

# ==================== 🔑 CONFIGURATION ====================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

TELEGRAM_BOT_TOKEN = "8125553397:AAGEextoGrpeFoCgcUcG98owqECCe6xR9kU".strip()
TELEGRAM_CHAT_ID = "5912667880".strip()

# Watchlist for 15-min Intraday & Scalping
WATCHLIST = [
    ("^NSEI", "NIFTY 50"),
    ("^NSEBANK", "BANK NIFTY"),
    ("RELIANCE.NS", "RELIANCE"),
    ("HDFCBANK.NS", "HDFC BANK"),
    ("ICICIBANK.NS", "ICICI BANK"),
    ("TATAMOTORS.NS", "TATAMOTORS"),
    ("INFY.NS", "INFOSYS"),
    ("SBIN.NS", "SBIN")
]

# Watchlist for Daily Long-Term Swing / Investing
INVESTMENT_WATCHLIST = [
    ("RELIANCE.NS", "RELIANCE"),
    ("TCS.NS", "TCS"),
    ("INFY.NS", "INFOSYS"),
    ("ICICIBANK.NS", "ICICI BANK"),
    ("LT.NS", "LARSEN & TOUBRO"),
    ("HINDUNILVR.NS", "HINDUSTAN UNILEVER"),
    ("ITC.NS", "ITC")
]

SYSTEM_PROMPT = """
You are a top-tier Indian Stock Market Quant & Technical Analyst.
Analyze the provided candlestick price action, Support/Resistance, and momentum.
Evaluate whether there is a high-confidence setup (>80% confidence).
Return your decision strictly in valid JSON format:
{
    "has_signal": true/false,
    "confidence": 85,
    "action": "BUY" or "SELL" or "HOLD",
    "entry": 24500.00,
    "target": 24650.00,
    "stop_loss": 24420.00,
    "risk_reward": "1:2",
    "reason": "Clear explanation of pattern, breakout, volume, and confluence."
}
"""

logs = []

def log(msg):
    ts = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    print(entry)
    logs.append(entry)
    if len(logs) > 50:
        logs.pop(0)

# ==================== 🤖 GEMINI CALL VIA HTTP (AQ KEY COMPATIBLE) ====================
def ask_gemini_analysis(symbol_name, timeframe, price_summary):
    if not GEMINI_API_KEY:
        return None, "Error: GEMINI_API_KEY environment variable is empty."

    # URL without query parameters (AQ keys require x-goog-api-key header)
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    
    prompt_text = f"""
    {SYSTEM_PROMPT}

    Symbol: {symbol_name}
    Timeframe: {timeframe}
    Latest Price Summary:
    {price_summary}
    """

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt_text}]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json"
        }
    }

    # Pass the AQ key strictly via x-goog-api-key header
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code != 200:
            return None, f"{response.status_code} Error: {response.text}"
        
        result_json = response.json()
        raw_text = result_json["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(raw_text), None
    except Exception as e:
        return None, str(e)

# ==================== 📲 TELEGRAM SENDER ====================
def send_telegram_alert(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        log(f"Telegram dispatch error: {e}")

# ==================== 📈 SCANNING ENGINE ====================
def scan_symbol(ticker, name, period="5d", interval="15m", mode="INTRADAY"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty or len(df) < 15:
            return

        # Flatten multi-level columns if created by yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        last_row = df.iloc[-1]
        
        # Safely convert to scalar float regardless of yfinance series structure
        close_p = float(pd.Series(last_row['Close']).iloc[0])
        high_p  = float(pd.Series(last_row['High']).iloc[0])
        low_p   = float(pd.Series(last_row['Low']).iloc[0])
        vol_val = int(pd.Series(last_row['Volume']).iloc[0]) if 'Volume' in df else 0

        summary = (
            f"Current Price: {close_p:.2f}\n"
            f"High: {high_p:.2f}\n"
            f"Low: {low_p:.2f}\n"
            f"Volume: {vol_val}"
        )

        analysis, err = ask_gemini_analysis(name, interval, summary)
        if err:
            log(f"{mode} Analysis error on {name}: {err}")
            return

        if analysis and analysis.get("has_signal") and analysis.get("confidence", 0) >= 80:
            msg = (
                f"🚨 *NEW AI MARKET SIGNAL* 🚨\n\n"
                f"📊 *Asset:* {name}\n"
                f"🎯 *Mode:* {mode} ({interval})\n"
                f"⚡ *Action:* {analysis.get('action')}\n"
                f"💵 *Entry:* ₹{analysis.get('entry')}\n"
                f"🎯 *Target:* ₹{analysis.get('target')}\n"
                f"🛑 *Stop Loss:* ₹{analysis.get('stop_loss')}\n"
                f"⚖️ *R:R Ratio:* {analysis.get('risk_reward')}\n"
                f"🔥 *Confidence:* {analysis.get('confidence')}%\n\n"
                f"📝 *Reason:* {analysis.get('reason')}"
            )
            log(f"✅ HIGH PROBABILITY SIGNAL FOUND FOR {name}! Sending alert...")
            send_telegram_alert(msg)
        else:
            log(f"Scan complete for {name}: No high-probability setup.")
    except Exception as e:
        log(f"Error scanning {name}: {e}")

def run_market_engine():
    log("🚀 Background 24/7 Market Engine Started.")
    while True:
        try:
            # 1. Long-Term Daily Scan
            log("💎 Scanning Long-Term Investment Candidates...")
            for ticker, name in INVESTMENT_WATCHLIST:
                scan_symbol(ticker, name, period="1mo", interval="1d", mode="SWING/INVESTMENT")
                time.sleep(2)

            # 2. Intraday / F&O 15-Minute Scan
            log("⚡ Scanning Intraday & F&O Watchlist...")
            for ticker, name in WATCHLIST:
                scan_symbol(ticker, name, period="5d", interval="15m", mode="F&O")
                time.sleep(2)

            # Wait 15 minutes before the next scan cycle
            time.sleep(900)
        except Exception as e:
            log(f"Engine Loop Error: {e}")
            time.sleep(30)

# Start background thread immediately
engine_thread = threading.Thread(target=run_market_engine, daemon=True)
engine_thread.start()

# ==================== 🌐 GRADIO DASHBOARD ====================
def get_dashboard_state():
    now_ist = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S IST")
    log_text = "\n".join(reversed(logs[-20:]))
    output_text = (
        f"🟢 AI Trading Bot is Active\n"
        f"⏰ Time: {now_ist}\n"
        f"🎯 Active Positions: 0\n\n"
        f"--- RECENT ACTIVITY LOGS ---\n"
        f"{log_text}"
    )
    return output_text

def clear_logs():
    logs.clear()
    return ""

with gr.Blocks(title="NSE Stock Market AI Bot") as demo:
    gr.Markdown("# 🇮🇳 NSE Indian Stock Market AI Bot")
    gr.Markdown("Running 24/7 on Render Cloud. Signals and Charts are automatically pushed to your Telegram bot.")
    
    out = gr.Textbox(label="output", lines=18, interactive=False)
    
    with gr.Row():
        gen_btn = gr.Button("Generate", variant="primary")
        clr_btn = gr.Button("Clear")

    gen_btn.click(fn=get_dashboard_state, inputs=[], outputs=[out])
    clr_btn.click(fn=clear_logs, inputs=[], outputs=[out])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
