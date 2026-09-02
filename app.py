import io
import json
import os
import time
import threading
from datetime import datetime
import pytz
import requests
import pandas as pd
import yfinance as yf
import mplfinance as mpf
from PIL import Image
from google import genai
from google.genai import types
import gradio as gr

# ==================== 🔑 CONFIGURATION ====================
GEMINI_API_KEY = "AQ.Ab8RN6Ll9KCzPwjpw-AJH2sdP2sGAiqUR3uxulUJbJ9gKk3ADQ".strip()
TELEGRAM_BOT_TOKEN = "8125553397:AAGEextoGrpeFoCgcUc9G8owqECCX6BAtLk".strip()
TELEGRAM_CHAT_ID = "5912667880".strip()

# 1. Trading Watchlist (15-min Intraday, Scalping & F&O)
FO_WATCHLIST = [
    ("^NSEI", "NIFTY 50"),
    ("^NSEBANK", "BANK NIFTY"),
    ("RELIANCE.NS", "RELIANCE"),
    ("HDFCBANK.NS", "HDFC BANK"),
    ("ICICIBANK.NS", "ICICI BANK"),
    ("TATAMOTORS.NS", "TATAMOTORS"),
    ("INFY.NS", "INFOSYS"),
    ("SBIN.NS", "SBIN")
]

# 2. Investing Watchlist (Daily Candles: 1M, 6M, 1Y Horizons)
INVESTMENT_WATCHLIST = [
    ("RELIANCE.NS", "RELIANCE"),
    ("TCS.NS", "TCS"),
    ("INFY.NS", "INFOSYS"),
    ("ICICIBANK.NS", "ICICI BANK"),
    ("LT.NS", "L&T"),
    ("BHARTIARTL.NS", "BHARTI AIRTEL"),
    ("TITAN.NS", "TITAN"),
    ("ITC.NS", "ITC"),
    ("SUNPHARMA.NS", "SUN PHARMA"),
    ("TATAMOTORS.NS", "TATAMOTORS")
]

MIN_CONFIDENCE = 0.80
MULTI_DAY_CONFIDENCE = 0.85
# ==========================================================

client = genai.Client(api_key=GEMINI_API_KEY)
ACTIVE_POSITIONS = {}
CLOSED_JOURNAL = []
POST_MARKET_SCAN_DONE = False
LOG_HISTORY = []

def log(msg):
    global LOG_HISTORY
    timestamp = get_ist_time().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {msg}"
    print(entry)
    LOG_HISTORY.append(entry)
    if len(LOG_HISTORY) > 50:
        LOG_HISTORY.pop(0)

def resolve_active_model():
    preferred = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.0-flash", "gemini-2.5-flash", "gemini-2.0-flash"]
    try:
        available = [m.name.replace("models/", "") for m in client.models.list()]
        for p in preferred:
            if p in available:
                return p
        for m in available:
            if "flash" in m:
                return m
    except Exception:
        pass
    return "gemini-2.5-flash"

ACTIVE_MODEL = resolve_active_model()

def get_ist_time():
    return datetime.now(pytz.timezone('Asia/Kolkata'))

def is_market_open():
    now = get_ist_time()
    if now.weekday() >= 5:
        return False
    start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return start <= now <= end

def is_btst_window():
    now = get_ist_time()
    return now.weekday() < 5 and (now.hour == 15 and 0 <= now.minute <= 25)

def send_telegram(caption, image_bytes=None):
    if image_bytes:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        files = {"photo": ("chart.png", image_bytes, "image/png")}
        data = {"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "HTML"}
        try:
            requests.post(url, files=files, data=data, timeout=30)
        except Exception as e:
            log(f"Telegram photo error: {e}")
    else:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        data = {"chat_id": TELEGRAM_CHAT_ID, "text": caption, "parse_mode": "HTML"}
        try:
            requests.post(url, data=data, timeout=20)
        except Exception as e:
            log(f"Telegram text error: {e}")

def generate_technical_dataset(ticker, clean_name, timeframe="15m", period="5d"):
    df = yf.download(ticker, period=period, interval=timeframe, auto_adjust=True, progress=False)
    if df.empty or len(df) < 30:
        return None, None, None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()

    buf = io.BytesIO()
    add_plots = [
        mpf.make_addplot(df['SMA_20'].tail(60), color='blue', width=1),
        mpf.make_addplot(df['EMA_50'].tail(60), color='orange', width=1)
    ]
    
    mpf.plot(
        df.tail(60),
        type='candle',
        style='charles',
        title=f"\nNSE: {clean_name} ({timeframe.upper()})",
        addplot=add_plots,
        volume=True,
        savefig=dict(fname=buf, dpi=120, bbox_inches='tight')
    )
    buf.seek(0)
    img_bytes = buf.getvalue()

    latest = df.iloc[-1]
    stats = {
        "price": round(float(latest['Close']), 2),
        "rsi": round(float(latest['RSI']), 2) if pd.notna(latest['RSI']) else 50.0,
        "atr": round(float(latest['ATR']), 2) if pd.notna(latest['ATR']) else 5.0,
        "sma_20": round(float(latest['SMA_20']), 2),
        "ema_50": round(float(latest['EMA_50']), 2)
    }
    return img_bytes, stats, float(latest['Close'])

def analyze_trading_fno_setup(img_bytes, clean_name, stats, is_index, force_btst=False):
    prompt = f"""
    You are an institutional derivatives analyst for NSE India.
    Analyze this 15-minute candlestick chart for {clean_name}:
    - CMP: Rs {stats['price']} | RSI: {stats['rsi']} | 20 SMA: Rs {stats['sma_20']} | 50 EMA: Rs {stats['ema_50']} | ATR: Rs {stats['atr']}
    - Mode: {'3:00 PM BTST Screen' if force_btst else 'Intraday Scan'}

    Determine:
    1. Strategy Style: SCALPING, INTRADAY_FNO, or MULTI_DAY_CARRY.
    2. Suggested Option Strike & Expiry.
    3. Hedging / Spread plan to mitigate theta decay.
    4. Numeric Entry, Target 1, Target 2, and Stop Loss.
    5. Holding timeframe and trailing stop loss rule.

    Return ONLY a JSON matching:
    {{
      "action": "BUY_CE | BUY_PE | BUY_EQUITY | NO_TRADE",
      "strategy_style": "SCALPING | INTRADAY_FNO | MULTI_DAY_CARRY",
      "confidence": 0.85,
      "recommended_instrument": "Rs {round(stats['price']/50)*50} CE Current Expiry",
      "hedging_spread": "Bull Call Spread: Buy ATM CE, Sell OTM CE to cap theta risk",
      "holding_duration": "Exit in 45-60 mins OR Hold 2-3 days with spread",
      "entry": {stats['price']},
      "target_1": {round(stats['price'] * 1.008, 2)},
      "target_2": {round(stats['price'] * 1.018, 2)},
      "stop_loss": {round(stats['price'] * 0.994, 2)},
      "trailing_sl_rule": "Move SL to Entry at Target 1; trail remainder",
      "rationale": "Breakout above 50 EMA with volume expansion"
    }}
    """
    try:
        image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")
        response = client.models.generate_content(
            model=ACTIVE_MODEL,
            contents=[image_part, prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)
    except Exception as e:
        log(f"F&O Analysis error on {clean_name}: {e}")
        return None

def analyze_investment_multihorizon(img_bytes, clean_name, stats):
    prompt = f"""
    You are a long-term equity research analyst for Indian stocks.
    Analyze this daily chart for {clean_name}:
    - CMP: Rs {stats['price']} | RSI: {stats['rsi']} | 20 SMA: Rs {stats['sma_20']} | 50 EMA: Rs {stats['ema_50']}

    Provide multi-horizon investment targets (1 Month, 6 Months, 1 Year). Confidence >= 0.80.

    Return ONLY a JSON matching:
    {{
      "action": "ACCUMULATE | STRONG_BUY | NO_TRADE",
      "confidence": 0.88,
      "accumulation_range": "Rs {stats['price']} - Rs {round(stats['price']*0.98, 2)}",
      "stop_loss": {round(stats['price'] * 0.91, 2)},
      "targets": {{
          "1_month_target": {round(stats['price'] * 1.07, 2)},
          "6_month_target": {round(stats['price'] * 1.20, 2)},
          "1_year_target": {round(stats['price'] * 1.38, 2)}
      }},
      "trailing_rule": "Move SL to 1-Month Target level once 6-Month Target is achieved",
      "hedging_optional": "Optional OTM Put buy for downside hedge",
      "investment_thesis": "Long-term accumulation with strong trend continuation"
    }}
    """
    try:
        image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")
        response = client.models.generate_content(
            model=ACTIVE_MODEL,
            contents=[image_part, prompt],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)
    except Exception as e:
        log(f"Investment Analysis error on {clean_name}: {e}")
        return None

def track_live_market_triggers():
    global ACTIVE_POSITIONS, CLOSED_JOURNAL
    now_str = get_ist_time().strftime("%H:%M:%S IST")

    for key, pos in list(ACTIVE_POSITIONS.items()):
        try:
            ticker = pos['ticker']
            df_live = yf.download(ticker, period="1d", interval="1m", progress=False)
            if df_live.empty:
                continue
            if isinstance(df_live.columns, pd.MultiIndex):
                df_live.columns = df_live.columns.get_level_values(0)

            current_price = float(df_live.iloc[-1]['Close'])
            entry = pos['entry']
            sl = pos['sl']
            direction = pos['direction']
            pnl_pct = ((current_price - entry) / entry) * 100 if direction != "BUY_PE" else ((entry - current_price) / entry) * 100

            if pos['category'] == "TRADING":
                t1, t2 = pos['t1'], pos['t2']

                if direction in ["BUY_CE", "BUY_EQUITY"]:
                    if current_price >= t2:
                        msg = f"🚀🚀 <b>TARGET 2 ACHIEVED! ({pos['strategy']})</b>\n\n<b>Stock:</b> {pos['name']}\n<b>Entry:</b> ₹{entry} ➔ <b>CMP:</b> ₹{current_price} (<b>+{pnl_pct:.2f}%</b>)\n<b>Target 2:</b> ₹{t2}\n<b>Status:</b> FULL TARGET ACCOMPLISHED ✅"
                        send_telegram(msg)
                        CLOSED_JOURNAL.append({"name": pos['name'], "status": "WIN_T2", "pnl": pnl_pct})
                        del ACTIVE_POSITIONS[key]
                    elif current_price >= t1 and not pos.get('t1_hit', False):
                        msg = f"🎯 <b>TARGET 1 ACHIEVED! ({pos['strategy']})</b>\n\n<b>Stock:</b> {pos['name']}\n<b>Entry:</b> ₹{entry} ➔ <b>CMP:</b> ₹{current_price} (<b>+{pnl_pct:.2f}%</b>)\n<b>Target 1:</b> ₹{t1}\n<b>Action:</b> Shift SL to Cost (₹{entry})."
                        send_telegram(msg)
                        ACTIVE_POSITIONS[key]['t1_hit'] = True
                    elif current_price <= sl:
                        msg = f"🛑 <b>STOP LOSS TRIGGERED</b>\n\n<b>Stock:</b> {pos['name']} ({pos['strategy']})\n<b>Exit Price:</b> ₹{current_price} (<b>{pnl_pct:.2f}%</b>)\n<b>SL Level:</b> ₹{sl}\n<b>Status:</b> Closed."
                        send_telegram(msg)
                        CLOSED_JOURNAL.append({"name": pos['name'], "status": "LOSS_SL", "pnl": pnl_pct})
                        del ACTIVE_POSITIONS[key]

                elif direction == "BUY_PE":
                    if current_price <= t2:
                        msg = f"🚀🚀 <b>TARGET 2 ACHIEVED (PUT)!</b>\n\n<b>Stock:</b> {pos['name']}\n<b>CMP:</b> ₹{current_price} (<b>+{pnl_pct:.2f}%</b>)\n<b>Target 2:</b> ₹{t2}"
                        send_telegram(msg)
                        CLOSED_JOURNAL.append({"name": pos['name'], "status": "WIN_T2", "pnl": pnl_pct})
                        del ACTIVE_POSITIONS[key]
                    elif current_price <= t1 and not pos.get('t1_hit', False):
                        msg = f"🎯 <b>TARGET 1 ACHIEVED (PUT)!</b>\n\n<b>Stock:</b> {pos['name']}\n<b>CMP:</b> ₹{current_price} (<b>+{pnl_pct:.2f}%</b>)\n<b>Target 1:</b> ₹{t1}"
                        send_telegram(msg)
                        ACTIVE_POSITIONS[key]['t1_hit'] = True
                    elif current_price >= sl:
                        msg = f"🛑 <b>STOP LOSS HIT (PUT)</b>\n\n<b>Stock:</b> {pos['name']}\n<b>Exit:</b> ₹{current_price} (<b>{pnl_pct:.2f}%</b>)"
                        send_telegram(msg)
                        CLOSED_JOURNAL.append({"name": pos['name'], "status": "LOSS_SL", "pnl": pnl_pct})
                        del ACTIVE_POSITIONS[key]

            elif pos['category'] == "INVESTING":
                for horizon, t_val in pos['targets'].items():
                    tag = f"{horizon}_hit"
                    if current_price >= t_val and not pos.get(tag, False):
                        label = horizon.replace('_', ' ').title()
                        msg = f"🏆 <b>INVESTMENT MILESTONE ACHIEVED!</b>\n\n<b>Stock:</b> {pos['name']}\n<b>Milestone:</b> {label}\n<b>Entry:</b> ₹{entry}\n<b>Target:</b> ₹{t_val}\n<b>CMP:</b> ₹{current_price} (<b>+{pnl_pct:.2f}%</b>)\n<b>Time:</b> {now_str}"
                        send_telegram(msg)
                        ACTIVE_POSITIONS[key][tag] = True

        except Exception as e:
            log(f"Tracking error on {key}: {e}")

def run_fno_trading_scan(is_btst=False):
    mode_tag = "3:00 PM BTST" if is_btst else "Intraday"
    log(f"⚡ Scanning F&O Basket ({mode_tag})...")

    for ticker, clean_name in FO_WATCHLIST:
        try:
            is_index = ticker in ["^NSEI", "^NSEBANK"]
            img_bytes, stats, current_price = generate_technical_dataset(ticker, clean_name, timeframe="15m", period="5d")
            if not img_bytes:
                continue

            signal = analyze_trading_fno_setup(img_bytes, clean_name, stats, is_index, force_btst=is_btst)
            if not signal:
                continue

            action = signal.get("action", "NO_TRADE")
            confidence = float(signal.get("confidence", 0))
            strategy_style = signal.get("strategy_style", "INTRADAY_FNO")
            threshold = MULTI_DAY_CONFIDENCE if strategy_style == "MULTI_DAY_CARRY" else MIN_CONFIDENCE

            if action in ["BUY_CE", "BUY_PE", "BUY_EQUITY"] and confidence >= threshold:
                badge = "🟢 <b>BUY CE (CALL)</b>" if action == "BUY_CE" else ("🔴 <b>BUY PE (PUT)</b>" if action == "BUY_PE" else "📈 <b>BUY CASH</b>")
                style_badge = "⚡ <b>SCALP</b>" if strategy_style == "SCALPING" else ("🌙 <b>MULTI-DAY CARRY</b>" if strategy_style == "MULTI_DAY_CARRY" else "⏱️ <b>INTRADAY</b>")
                
                caption = (
                    f"🚨 <b>{style_badge} | {badge} ({confidence*100:.0f}% CONFIDENCE)</b>\n\n"
                    f"<b>Asset:</b> {clean_name} | <b>CMP:</b> ₹{stats['price']}\n"
                    f"<b>Suggested Strike:</b> {signal.get('recommended_instrument')}\n\n"
                    f"🛡️ <b>Hedging Strategy:</b>\n<i>{signal.get('hedging_spread')}</i>\n\n"
                    f"⏳ <b>Holding Time:</b> {signal.get('holding_duration')}\n\n"
                    f"🎯 <b>Entry:</b> ₹{signal.get('entry')}\n"
                    f"🛑 <b>Stop Loss:</b> ₹{signal.get('stop_loss')}\n"
                    f"🔄 <b>Trailing SL:</b> {signal.get('trailing_sl_rule')}\n"
                    f"🏁 <b>Target 1:</b> ₹{signal.get('target_1')}\n"
                    f"🚀 <b>Target 2:</b> ₹{signal.get('target_2')}\n\n"
                    f"📝 <b>Rationale:</b> {signal.get('rationale')}"
                )
                send_telegram(caption, img_bytes)
                log(f"✅ Dispatched {strategy_style} alert for {clean_name}")

                ACTIVE_POSITIONS[f"TRADE_{ticker}"] = {
                    "name": clean_name,
                    "ticker": ticker,
                    "category": "TRADING",
                    "strategy": strategy_style,
                    "direction": action,
                    "entry": float(signal.get('entry', current_price)),
                    "t1": float(signal.get('target_1', current_price * 1.01)),
                    "t2": float(signal.get('target_2', current_price * 1.02)),
                    "sl": float(signal.get('stop_loss', current_price * 0.994)),
                    "t1_hit": False
                }
            time.sleep(2)
        except Exception as e:
            log(f"Error in F&O item {clean_name}: {e}")

def run_daily_investment_scan():
    log("💎 Scanning Long-Term Investment Candidates...")
    for ticker, clean_name in INVESTMENT_WATCHLIST:
        try:
            img_bytes, stats, current_price = generate_technical_dataset(ticker, clean_name, timeframe="1d", period="1y")
            if not img_bytes:
                continue

            signal = analyze_investment_multihorizon(img_bytes, clean_name, stats)
            if not signal:
                continue

            action = signal.get("action", "NO_TRADE")
            confidence = float(signal.get("confidence", 0))

            if action in ["ACCUMULATE", "STRONG_BUY"] and confidence >= MIN_CONFIDENCE:
                targets = signal.get("targets", {})
                caption = (
                    f"💎 <b>LONG-TERM INVESTMENT PICK ({confidence*100:.0f}% CONFIDENCE)</b>\n\n"
                    f"<b>Stock:</b> {clean_name} | 🟢 <b>{action}</b>\n"
                    f"<b>CMP:</b> ₹{stats['price']}\n"
                    f"<b>Accumulation Zone:</b> {signal.get('accumulation_range')}\n"
                    f"<b>Invalidation / SL:</b> ₹{signal.get('stop_loss')}\n\n"
                    f"🎯 <b>1-Month Target:</b> ₹{targets.get('1_month_target')}\n"
                    f"🎯 <b>6-Month Target:</b> ₹{targets.get('6_month_target')}\n"
                    f"🚀 <b>1-Year Target:</b> ₹{targets.get('1_year_target')}\n\n"
                    f"🔄 <b>Trailing Rule:</b> {signal.get('trailing_rule')}\n"
                    f"🛡️ <b>Portfolio Hedge:</b> {signal.get('hedging_optional')}\n\n"
                    f"📝 <b>Thesis:</b> {signal.get('investment_thesis')}"
                )
                send_telegram(caption, img_bytes)
                log(f"✅ Dispatched Investment pick for {clean_name}")

                ACTIVE_POSITIONS[f"INV_{ticker}"] = {
                    "name": clean_name,
                    "ticker": ticker,
                    "category": "INVESTING",
                    "direction": "BUY_EQUITY",
                    "entry": stats['price'],
                    "sl": float(signal.get('stop_loss', stats['price'] * 0.91)),
                    "targets": targets
                }
            time.sleep(2)
        except Exception as e:
            log(f"Error in Investment item {clean_name}: {e}")

def automated_background_worker():
    global POST_MARKET_SCAN_DONE
    log("🚀 Background 24/7 Market Engine Started.")
    
    time.sleep(5)
    run_daily_investment_scan()
    run_fno_trading_scan()

    while True:
        try:
            now = get_ist_time()
            track_live_market_triggers()

            if is_market_open():
                POST_MARKET_SCAN_DONE = False
                if is_btst_window():
                    run_fno_trading_scan(is_btst=True)
                    time.sleep(900)
                else:
                    run_fno_trading_scan(is_btst=False)
                    time.sleep(900)
            else:
                if now.weekday() < 5 and now.hour == 15 and now.minute >= 35 and not POST_MARKET_SCAN_DONE:
                    run_daily_investment_scan()
                    POST_MARKET_SCAN_DONE = True
                time.sleep(60)

        except Exception as e:
            log(f"Loop error: {e}")
            time.sleep(30)

# Launch worker thread
threading.Thread(target=automated_background_worker, daemon=True).start()

# Live Gradio Interface
def get_status():
    ist = get_ist_time().strftime('%Y-%m-%d %H:%M:%S IST')
    status_text = f"🟢 AI Trading Bot is Active\n⏰ Time: {ist}\n🎯 Active Positions: {len(ACTIVE_POSITIONS)}\n\n"
    status_text += "--- RECENT ACTIVITY LOGS ---\n" + "\n".join(LOG_HISTORY[-15:])
    return status_text

demo = gr.Interface(
    fn=get_status,
    inputs=None,
    outputs="text",
    title="🇮🇳 NSE Indian Stock Market AI Bot",
    description="Running 24/7 on Render Cloud. Signals and Charts are automatically pushed to your Telegram bot."
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port)
