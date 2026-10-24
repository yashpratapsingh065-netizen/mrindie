import os
import sqlite3
import time
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

# Indian Standard Time (IST) Offset: UTC + 5:30
IST = timezone(timedelta(hours=5, minutes=30))

st.set_page_config(
    page_title="MR. INDIE — Indian Market Algorithmic Trading Platform",
    page_icon="🇮🇳",
    layout="wide"
)

DB_PATH = os.getenv("MRINDIE_DB_PATH", "mrindie_trading_bot.db")

INDIAN_SYMBOLS = {
    "NIFTY 50": {"yf": "^NSEI", "tv": "NSE:NIFTY", "lot": 25, "type": "INDEX"},
    "BANK NIFTY": {"yf": "^NSEBANK", "tv": "NSE:BANKNIFTY", "lot": 15, "type": "INDEX"},
    "RELIANCE": {"yf": "RELIANCE.NS", "tv": "NSE:RELIANCE", "lot": 1, "type": "EQUITY"},
    "TCS": {"yf": "TCS.NS", "tv": "NSE:TCS", "lot": 1, "type": "EQUITY"},
    "HDFC BANK": {"yf": "HDFCBANK.NS", "tv": "NSE:HDFCBANK", "lot": 1, "type": "EQUITY"}
}

INTERVALS = {
    "5 min": {"yf": "5m", "minutes": 5, "tv": "5"},
    "15 min": {"yf": "15m", "minutes": 15, "tv": "15"},
    "1 hour": {"yf": "60m", "minutes": 60, "tv": "60"},
    "1 day": {"yf": "1d", "minutes": 1440, "tv": "D"}
}

# --- DATABASE ENGINE ---
def connect_db():
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.execute("""
        CREATE TABLE IF NOT EXISTS mrindie_open_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at_ist TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            entry_price REAL,
            stop_loss REAL,
            take_profit REAL,
            quantity INTEGER DEFAULT 1,
            bias TEXT NOT NULL,
            mode TEXT NOT NULL,
            strategy_votes TEXT,
            candle_timestamp TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS mrindie_trade_history (
            id INTEGER PRIMARY KEY,
            created_at_ist TEXT NOT NULL,
            closed_at_ist TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            entry_price REAL,
            exit_price REAL,
            stop_loss REAL,
            take_profit REAL,
            quantity INTEGER,
            bias TEXT NOT NULL,
            mode TEXT NOT NULL,
            exit_reason TEXT NOT NULL,
            pnl_inr REAL NOT NULL,
            strategy_votes TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS mrindie_account_balance (
            id INTEGER PRIMARY KEY,
            balance_inr REAL NOT NULL
        )
    """)
    cursor = con.cursor()
    cursor.execute("SELECT balance_inr FROM mrindie_account_balance WHERE id = 1")
    row = cursor.fetchone()
    if not row:
        con.execute("INSERT INTO mrindie_account_balance (id, balance_inr) VALUES (1, 100000.0)")
    con.commit()
    return con

# --- MARKET HOURS CONTROL ---
def get_indian_market_status():
    now_ist = datetime.now(IST)
    current_time = now_ist.time()
    
    if now_ist.weekday() >= 5: # Saturday / Sunday
        return "WEEKEND_CLOSED", "🔴 Market Closed (Weekend)"
        
    m_open = datetime.strptime("09:15:00", "%H:%M:%S").time()
    m_close = datetime.strptime("15:30:00", "%H:%M:%S").time()
    sq_time = datetime.strptime("15:15:00", "%H:%M:%S").time()
    
    if sq_time <= current_time <= m_close:
        return "AUTO_SQUARE_OFF", "⚠️ Intraday Auto Square-off Window (3:15 PM+)"
    elif m_open <= current_time <= m_close:
        return "OPEN", "🟢 Market Live (NSE/BSE)"
    else:
        return "CLOSED", "🔴 Market Closed (Trading Hours: 9:15 AM - 3:30 PM IST)"

# --- DATA FETCHING ENGINE ---
def fetch_indian_market_data(symbol_label, timeframe_label):
    info = INDIAN_SYMBOLS[symbol_label]
    cfg = INTERVALS[timeframe_label]
    
    try:
        df = yf.download(info["yf"], period="5d", interval=cfg["yf"], progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0].lower() for col in df.columns]
            else:
                df.columns = [col.lower() for col in df.columns]
            df["datetime_ist"] = df.index.tz_convert(IST)
            return df[["open", "high", "low", "close", "datetime_ist"]].tail(200), "NSE Real-Time Feed"
    except Exception:
        pass
    return pd.DataFrame(), "No Data Available"

# --- PRICE LESSON SMC STRATEGY ENGINE ---
def evaluate_mrindie_smc_strategy(df):
    if len(df) < 30:
        return {"bias": "WAIT", "votes": "Insufficient Data", "bullish": 0, "bearish": 0, "entry": 0, "sl": 0, "tp": 0}
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    p_prev = df.iloc[-3]
    
    body = abs(last.close - last.open)
    rng = max(last.high - last.low, 1e-12)
    upper_wick = last.high - max(last.open, last.close)
    lower_wick = min(last.open, last.close) - last.low
    
    swing_high = float(df.high.tail(50).max())
    swing_low = float(df.low.tail(50).min())
    eq_50 = (swing_high + swing_low) / 2.0
    
    bullish_votes = []
    bearish_votes = []

    # 1. Premium vs Discount Zone (S7 Rule)
    if last.close < eq_50:
        bullish_votes.append("Discount Zone (Buy Bias)")
    else:
        bearish_votes.append("Premium Zone (Sell Bias)")

    # 2. Unmitigated Fair Value Gap (S5 FVG Rule)
    if last.low > p_prev.high and prev.close > prev.open:
        bullish_votes.append("Unmitigated Bullish FVG")
    elif last.high < p_prev.low and prev.close < prev.open:
        bearish_votes.append("Unmitigated Bearish FVG")

    # 3. Supply / Demand Wick Rejections
    if lower_wick >= body * 1.5 and last.close > last.open:
        bullish_votes.append("Demand Rejection Wick")
    elif upper_wick >= body * 1.5 and last.close < last.open:
        bearish_votes.append("Supply Rejection Wick")

    # 4. Breakout Confluence
    recent = df.iloc[-21:-1]
    if last.close > float(recent.high.max()):
        bullish_votes.append("Resistance High Breakout")
    elif last.close < float(recent.low.min()):
        bearish_votes.append("Support Low Breakdown")

    b_score = len(bullish_votes)
    s_score = len(bearish_votes)

    bias = "BUY" if b_score >= s_score else "SELL"
    curr_close = float(last.close)
    
    sl_points = max(rng * 1.5, 15.0 if curr_close > 1000 else 2.0)
    tp_points = max(rng * 3.0, 30.0 if curr_close > 1000 else 4.0)

    sl = curr_close - sl_points if bias == "BUY" else curr_close + sl_points
    tp = curr_close + tp_points if bias == "BUY" else curr_close - tp_points

    votes_summary = f"Bullish [{b_score}]: {', '.join(bullish_votes)} | Bearish [{s_score}]: {', '.join(bearish_votes)}"

    return {
        "bias": bias,
        "votes": votes_summary,
        "bullish": b_score,
        "bearish": s_score,
        "entry": round(curr_close, 2),
        "sl": round(sl, 2),
        "tp": round(tp, 2)
    }

def get_balance(con):
    cursor = con.cursor()
    cursor.execute("SELECT balance_inr FROM mrindie_account_balance WHERE id = 1")
    row = cursor.fetchone()
    return float(row[0]) if row else 100000.0

def update_balance(con, delta):
    new_bal = get_balance(con) + delta
    con.execute("UPDATE mrindie_account_balance SET balance_inr = ? WHERE id = 1", (new_bal,))
    con.commit()
    return new_bal

def execute_mrindie_trade(con, symbol, timeframe, result, qty, mode, candle_ts):
    cursor = con.cursor()
    cursor.execute("SELECT id FROM mrindie_open_trades WHERE symbol = ? AND timeframe = ? AND candle_timestamp = ?", (symbol, timeframe, str(candle_ts)))
    if cursor.fetchone():
        return False # Cooldown: Already traded on this candle

    created_ist = datetime.now(IST).strftime('%Y-%m-%d %I:%M:%S %p IST')
    con.execute(
        "INSERT INTO mrindie_open_trades (created_at_ist, symbol, timeframe, entry_price, stop_loss, take_profit, quantity, bias, mode, strategy_votes, candle_timestamp) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (created_ist, symbol, timeframe, result["entry"], result["sl"], result["tp"], qty, result["bias"], mode, result["votes"], str(candle_ts))
    )
    con.commit()
    return True

def close_mrindie_trade(con, trade_id, current_price, reason="Manual Exit 👤"):
    cursor = con.cursor()
    cursor.execute("SELECT created_at_ist, symbol, timeframe, entry_price, stop_loss, take_profit, quantity, bias, mode, strategy_votes FROM mrindie_open_trades WHERE id = ?", (trade_id,))
    row = cursor.fetchone()
    if not row:
        return 0.0

    created_ist, symbol, timeframe, entry_p, sl, tp, qty, bias, mode, strategy_votes = row
    diff = (current_price - entry_p) if bias == "BUY" else (entry_p - current_price)
    pnl_inr = round(diff * qty, 2)
    closed_ist = datetime.now(IST).strftime('%Y-%m-%d %I:%M:%S %p IST')

    con.execute(
        "INSERT INTO mrindie_trade_history (id, created_at_ist, closed_at_ist, symbol, timeframe, entry_price, exit_price, stop_loss, take_profit, quantity, bias, mode, exit_reason, pnl_inr, strategy_votes) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (trade_id, created_ist, closed_ist, symbol, timeframe, entry_p, round(current_price, 2), sl, tp, qty, bias, mode, reason, pnl_inr, strategy_votes)
    )
    con.execute("DELETE FROM mrindie_open_trades WHERE id = ?", (trade_id,))
    con.commit()
    update_balance(con, pnl_inr)
    return pnl_inr

# --- STREAMLIT DASHBOARD UI ---
st.title("🇮🇳 MR. INDIE — Indian Market Price Lesson Bot")
st.caption("Institutional SMC Strategy Engine • Anti-Duplicate Execution • Dhan API Ready")

con = connect_db()
m_code, m_status_msg = get_indian_market_status()

with st.sidebar:
    st.header("⚙️ Mr. Indie Control Panel")
    selected_symbol = st.selectbox("Select Instrument", list(INDIAN_SYMBOLS.keys()), index=0)
    selected_tf = st.selectbox("Timeframe", list(INTERVALS.keys()), index=1)
    
    default_qty = INDIAN_SYMBOLS[selected_symbol]["lot"]
    trade_qty = st.number_input("Order Quantity / Lots", min_value=1, max_value=1000, value=default_qty, step=1)
    
    st.markdown("---")
    st.subheader("🛡️ Execution Environment")
    trading_mode = st.radio("Select Mode", ["DEMO (Paper Trading)", "LIVE BROKER (Dhan API)"], index=0)
    
    if "LIVE BROKER" in trading_mode:
        st.warning("🔒 Live Mode Enabled. Valid Dhan Client ID & Access Token required in secrets.")
        dhan_client_id = st.text_input("Dhan Client ID", type="password")
        dhan_access_token = st.text_input("Dhan Access Token", type="password")
    
    st.markdown("---")
    auto_pilot = st.toggle("Enable Auto-Pilot Signal Execution", value=False)
    
    if st.button("➕ Execute Manual Strategy Trade", use_container_width=True):
        df_temp, _ = fetch_indian_market_data(selected_symbol, selected_tf)
        if not df_temp.empty:
            res_temp = evaluate_mrindie_smc_strategy(df_temp)
            candle_ts = df_temp.index[-1]
            mode_str = "LIVE" if "LIVE" in trading_mode else "DEMO"
            ok = execute_mrindie_trade(con, selected_symbol, selected_tf, res_temp, trade_qty, mode_str, candle_ts)
            if ok:
                st.sidebar.success(f"Trade placed on {selected_symbol}!")
            else:
                st.sidebar.warning("Duplicate trade blocked: Candle already traded!")

    st.markdown("---")
    if st.button("🔄 Reset Virtual Portfolio (₹1,00,000)", use_container_width=True):
        con.execute("UPDATE mrindie_account_balance SET balance_inr = 100000.0 WHERE id = 1")
        con.commit()
        st.sidebar.success("Virtual balance reset!")
        st.rerun()

# --- MAIN DASHBOARD BODY ---
st.info(f"Market Status: **{m_status_msg}**")

df_live, feed_source = fetch_indian_market_data(selected_symbol, selected_tf)

if not df_live.empty:
    smc_result = evaluate_mrindie_smc_strategy(df_live)
    live_p = float(df_live.close.iloc[-1])
    curr_candle_ts = df_live.index[-1]
    mode_str = "LIVE" if "LIVE" in trading_mode else "DEMO"

    if auto_pilot and m_code == "OPEN":
        execute_mrindie_trade(con, selected_symbol, selected_tf, smc_result, trade_qty, mode_str, curr_candle_ts)

    curr_bal = get_balance(con)
    st.markdown(f"### 💼 Portfolio Capital: **₹{curr_bal:,.2f} INR**")
    
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Live Spot Price ({selected_symbol})", f"₹{live_p:,.2f}")
    c2.metric("SMC Signal Bias", smc_result["bias"])
    c3.metric("Confluence Score", f"Bull: {smc_result['bullish']} | Bear: {smc_result['bearish']}")

st.divider()

col_chart, col_details = st.columns([1.8, 1])

with col_chart:
    st.subheader(f"📈 TradingView Live Chart ({selected_symbol})")
    tv_sym = INDIAN_SYMBOLS[selected_symbol]["tv"]
    tv_tf = INTERVALS[selected_tf]["tv"]
    
    tv_widget_html = f"""
    <div class="tradingview-widget-container" style="height:480px;width:100%">
      <div id="tv_chart_mrindie" style="height:460px;width:100%"></div>
      <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
      <script type="text/javascript">
      new TradingView.widget({{
        "autosize": true,
        "symbol": "{tv_sym}",
        "interval": "{tv_tf}",
        "timezone": "Asia/Kolkata",
        "theme": "dark",
        "style": "1",
        "locale": "en",
        "enable_publishing": false,
        "allow_symbol_change": true,
        "container_id": "tv_chart_mrindie"
      }});
      </script>
    </div>
    """
    components.html(tv_widget_html, height=490, scrolling=False)

with col_details:
    st.subheader("🧠 Price Lesson SMC Analysis")
    if not df_live.empty:
        st.write(f"**Confluence Breakdown:** {smc_result['votes']}")
        st.write(f"**Target SL:** `₹{smc_result['sl']:,.2f}` | **Target TP:** `₹{smc_result['tp']:,.2f}`")
        st.caption(f"Data Feed Source: {feed_source}")
        st.markdown("---")
        df_tail = df_live.tail(4).copy()
        df_tail.index = df_tail.index.strftime('%d %b %I:%M %p')
        st.dataframe(df_tail[["open", "high", "low", "close"]], use_container_width=True)

st.divider()

st.subheader("📋 Active Open Positions")
open_trades_df = pd.read_sql_query("SELECT * FROM mrindie_open_trades ORDER BY id DESC", con)

if open_trades_df.empty:
    st.info("Abhi koi open position active nahi hai.")
else:
    for idx, r in open_trades_df.iterrows():
        with st.expander(f"📌 Trade #{r['id']} | {r['symbol']} ({r['bias']} - {r['quantity']} Qty) - Mode: {r['mode']}"):
            c1, c2 = st.columns(2)
            c1.markdown(f"🕒 **Entry Time:** `{r['created_at_ist']}`\n📍 **Entry Price:** `₹{r['entry_price']:,.2f}`\n🛑 **Stop Loss:** `₹{r['stop_loss']:,.2f}`")
            c2.markdown(f"🎯 **Take Profit:** `₹{r['take_profit']:,.2f}`\n⚡ **Timeframe:** `{r['timeframe']}`\n📊 **Votes:** `{r['strategy_votes']}`")
            
            if st.button(f"Square Off Position #{r['id']}", key=f"sq_{r['id']}"):
                pnl = close_mrindie_trade(con, r['id'], live_p, reason="Manual Square-Off 👤")
                st.toast(f"Trade #{r['id']} closed! Realized PnL: ₹{pnl:+,.2f}")
                st.rerun()

st.divider()

st.subheader("📜 Completed Trade Audit History")
history_trades_df = pd.read_sql_query("SELECT * FROM mrindie_trade_history ORDER BY id DESC LIMIT 50", con)

if history_trades_df.empty:
    st.info("Abhi tak koi trade close nahi hui hai.")
else:
    for idx, r in history_trades_df.iterrows():
        pnl_flag = "🟢" if r['pnl_inr'] >= 0 else "🔴"
        with st.expander(f"{pnl_flag} Trade #{r['id']} | {r['symbol']} ({r['bias']}) | PnL: ₹{r['pnl_inr']:+,.2f} | Reason: {r['exit_reason']}"):
            c1, c2 = st.columns(2)
            c1.markdown(f"🕒 **Entry:** `{r['created_at_ist']}`\n🕒 **Exit:** `{r['closed_at_ist']}`\n📦 **Qty:** `{r['quantity']}`")
            c2.markdown(f"📍 **Entry Price:** `₹{r['entry_price']:,.2f}`\n🏁 **Exit Price:** `₹{r['exit_price']:,.2f}`\n💰 **Net PnL:** `₹{r['pnl_inr']:+,.2f}`")

con.close()
