# mcrf_auto_scanner.py
"""
MCRF — Auto Scanner Streamlit App (User-friendly edition)
Features:
- SCAN NOW (one-click) using built-in sample universes (micro/small/sector lists)
- Category-based scanner (AI, EV, Semiconductors, Biotech, Energy, Robotics)
- Auto-scan logic: volume anomaly, momentum, MA alignment, RSI, HL accumulation
- Picks Top-7 (ranked) and shows clean cards + mini charts
- Watchlist (session-based), export CSV
- Clean/simplified UI layout for all users
- Educational only — NOT investment advice

Dependencies:
pip install streamlit yfinance pandas numpy plotly
Optional for Colab/Sharing: pyngrok
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime
import math
from typing import List, Dict

st.set_page_config(page_title="MCRF Auto Scanner", layout="wide", initial_sidebar_state="expanded")

# ---------------------
# --- Utility funcs ---
# ---------------------
@st.cache_data(ttl=300)
def fetch_history(ticker: str, period="1y", interval="1d"):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, actions=False)
        if df.empty:
            return pd.DataFrame()
        df = df[['Open','High','Low','Close','Volume']].copy()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception:
        return pd.DataFrame()

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df = df.dropna(subset=['Close'])
    df['Volume'] = df['Volume'].fillna(0)
    # forward-fill small gaps (not ideal for production but OK for prototype)
    df = df.asfreq('D').fillna(method='ffill')
    df = df[df['Volume']>0]
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df['MA20'] = df['Close'].rolling(20, min_periods=5).mean()
    df['MA50'] = df['Close'].rolling(50, min_periods=10).mean()
    df['MA200'] = df['Close'].rolling(200, min_periods=50).mean()
    df['Momentum_21'] = df['Close'] / df['Close'].shift(21) - 1
    df['Vol_MA20'] = df['Volume'].rolling(20, min_periods=5).mean()
    delta = df['Close'].diff()
    up = delta.clip(lower=0); down = -1*delta.clip(upper=0)
    roll_up = up.rolling(14).mean(); roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['RSI14'] = 100 - (100/(1+rs))
    df['Vol_Spike'] = df['Volume'] / (df['Vol_MA20'] + 1e-9)
    df['HL'] = ((df['Low'] > df['Low'].shift(1)) & (df['Low'].shift(1) > df['Low'].shift(2))).astype(int)
    return df

def score_ticker(df: pd.DataFrame, info: dict) -> Dict:
    if df.empty:
        return {'score': -999, 'reason': 'no_data'}
    latest = df.iloc[-1]
    mom = float(latest.get('Momentum_21', 0))
    mom_s = np.tanh(mom*5)
    rsi = float(latest.get('RSI14', 50))
    rsi_s = 1 - abs((rsi-55)/55)
    vol_spike = float(latest.get('Vol_Spike', 1))
    vol_s = np.tanh((vol_spike-1.0)*1.2)
    hl = int(df['HL'].rolling(21).sum().iloc[-1]) if 'HL' in df.columns else 0
    hl_s = min(hl/5.0, 1.0)
    ma_bias = 0.0
    if latest['Close'] > latest.get('MA50', -1): ma_bias += 0.4
    if latest['Close'] > latest.get('MA200', -1): ma_bias += 0.6
    rev_growth = info.get('revenueGrowth') or 0.0
    rev_s = np.tanh(rev_growth*3)
    composite = 0.32*mom_s + 0.18*rsi_s + 0.20*vol_s + 0.10*hl_s + 0.12*ma_bias + 0.08*rev_s
    score = (composite + 1) / 2 * 100
    return {'score': round(float(score),2), 'mom': round(mom_s,3), 'rsi': round(rsi,2), 'vol_spike': round(vol_spike,2), 'hl21': hl}

# ---------------------
# --- Sample universes (small, micro, by-theme) ---
# (These are small sample lists for demo. For full auto-scan upload CSV or provide larger list.)
# ---------------------
SAMPLE_SMALLCAP = ["AAPL","MSFT","NVDA","TSM","AMD","INTC","QCOM","AVGO","CRM","ORCL"]
SAMPLE_MICRO = ["SOUN", "BBAI", "INDI", "VLD", "IRTC", "HIMX","ENPH","FSLR","PLUG","SPCE"]  # mix sample
THEME_AI = ["NVDA","AMD","SMCI","AVGO","MSFT","GOOGL","INTC"]
THEME_ROBOTICS = ["IRTC","FSLR","VLD","INDI","HIMX"]
THEME_BIO = ["IRTC","CRSP","BEAM","NTLA","ILMN"]  # examples
THEME_EV = ["TSLA","NIO","LI","XPEV","FSR"]

# ---------------------
# --- UI: Sidebar ---
# ---------------------
st.sidebar.title("MCRF — Easy Scanner")
st.sidebar.markdown("One-click scanner for users")

scan_mode = st.sidebar.radio("Scanner mode", ["SCAN NOW (Auto sample)", "Category Scanner", "Upload CSV"])
period = st.sidebar.selectbox("History period", ["6mo","1y","2y"], index=1)
topk = st.sidebar.slider("Top picks to show",  value=7, min_value=3, max_value=15)
max_check = st.sidebar.slider("Max tickers to attempt (only for CSV or custom)", min_value=50, max_value=1000, value=300, step=50)

st.title("MCRF — Auto Scanner (User Friendly)")
st.markdown("**One-click SCAN NOW** — system finds stocks for you from sample universes. Educational only.")

# ---------------------
# --- Main: SCAN NOW ---
# ---------------------
def run_scan(ticker_list: List[str]):
    results = []
    details = {}
    for i, t in enumerate(ticker_list):
        try:
            hist = fetch_history(t, period=period)
            df = clean_df(hist)
            df = compute_indicators(df)
            info = {}
            try:
                info = yf.Ticker(t).info
            except Exception:
                info = {}
            sc = score_ticker(df, info)
            results.append({'ticker': t, 'score': sc.get('score', -999)})
            details[t] = {'df': df, 'info': info, 'score': sc}
        except Exception as e:
            results.append({'ticker': t, 'score': -999})
    out = pd.DataFrame(results).sort_values('score', ascending=False).reset_index(drop=True)
    return out, details

# Main actions
if scan_mode == "SCAN NOW (Auto sample)":
    st.subheader("SCAN NOW")
    st.write("Auto-scan uses sample universes optimized for user simplicity.")
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        if st.button("Quick: SmallCap sample"):
            tickers = SAMPLE_SMALLCAP
            st.session_state['last_universe'] = "SmallCap sample"
            st.session_state['last_list'] = tickers
            with st.spinner("Scanning..."):
                out, details = run_scan(tickers)
                st.session_state['last_result'] = out; st.session_state['details']=details
    with col2:
        if st.button("Quick: Micro sample"):
            tickers = SAMPLE_MICRO
            st.session_state['last_universe'] = "Micro sample"
            st.session_state['last_list'] = tickers
            with st.spinner("Scanning..."):
                out, details = run_scan(tickers)
                st.session_state['last_result'] = out; st.session_state['details']=details
    with col3:
        if st.button("Quick: AI Theme"):
            tickers = THEME_AI
            st.session_state['last_universe'] = "AI theme"
            st.session_state['last_list'] = tickers
            with st.spinner("Scanning..."):
                out, details = run_scan(tickers)
                st.session_state['last_result'] = out; st.session_state['details']=details

    if 'last_result' in st.session_state:
        res = st.session_state['last_result']
        details = st.session_state['details']
        st.markdown("### Top picks")
        st.table(res.head(topk))
        # show cards for top-k
        picks = res.head(topk)['ticker'].tolist()
        for t in picks:
            sc = details[t]['score']
            df = details[t]['df']
            info = details[t]['info']
            colA, colB = st.columns([1,3])
            with colA:
                color = "🟢" if sc.get('score',-999)>=65 else ("🟡" if sc.get('score',-999)>=45 else "🔴")
                st.markdown(f"#### {t} {color} — Score {sc.get('score')}")
                st.write(info.get('shortName') or info.get('longName') or "")
                st.write(f"Sector: {info.get('sector','n/a')}")
                add = st.button(f"Add {t} to Watchlist", key=f"add_{t}")
                if add:
                    if 'watchlist' not in st.session_state: st.session_state['watchlist']=[]
                    if t not in st.session_state['watchlist']:
                        st.session_state['watchlist'].append(t)
                        st.success(f"{t} added to watchlist")
            with colB:
                if not df.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'))
                    if 'MA50' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50', line=dict(dash='dash')))
                    fig.update_layout(height=260, margin=dict(l=0,r=0,t=10,b=0))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.write("No price data")

elif scan_mode == "Category Scanner":
    st.subheader("Category Scanner")
    cat = st.selectbox("Choose category", ["AI","Robotics","Biotech","EV","Semiconductors","Micro-cap"])
    pre = []
    if cat=="AI": pre = THEME_AI
    if cat=="Robotics": pre = THEME_ROBOTICS
    if cat=="Biotech": pre = THEME_BIO
    if cat=="EV": pre = THEME_EV
    if cat=="Semiconductors": pre = ["TSM","ASML","NVDA","INTC","MU","AMD"]
    if cat=="Micro-cap": pre = SAMPLE_MICRO
    st.write("Selected sample universe:", pre)
    if st.button("Scan category"):
        with st.spinner("Scanning category..."):
            out, details = run_scan(pre)
            st.session_state['last_result']=out; st.session_state['details']=details
            st.table(out.head(topk))

else:  # upload CSV
    st.subheader("Upload CSV (column: ticker)")
    uploaded = st.file_uploader("Upload CSV file with 'ticker' column", type=["csv"])
    if uploaded is not None:
        df_up = pd.read_csv(uploaded)
        if 'ticker' not in df_up.columns:
            st.error("CSV must contain 'ticker' column")
        else:
            tickers = [str(x).upper() for x in df_up['ticker'].tolist()][:max_check]
            if st.button("Run CSV scan"):
                with st.spinner("Running CSV scan. This may take time for large lists."):
                    out, details = run_scan(tickers)
                    st.session_state['last_result']=out; st.session_state['details']=details
                    st.table(out.head(topk))

# Watchlist & export
st.sidebar.markdown("---")
st.sidebar.subheader("Watchlist")
if 'watchlist' not in st.session_state:
    st.session_state['watchlist'] = []
st.sidebar.write(st.session_state['watchlist'])
if st.sidebar.button("Scan my watchlist") and st.session_state['watchlist']:
    out, details = run_scan(st.session_state['watchlist'])
    st.session_state['last_result']=out; st.session_state['details']=details
    st.sidebar.success("Watchlist scanned — check main view")
if st.sidebar.button("Export last result CSV") and 'last_result' in st.session_state:
    st.sidebar.download_button("Download CSV", st.session_state['last_result'].to_csv(index=False), file_name=f"mcrf_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")

# Footer: quick help
st.markdown("---")
st.markdown("**How to use**: choose SCAN NOW or Category Scanner → click a quick sample button → view Top picks → Add to Watchlist if you like → Scan Watchlist. For a full universe scan, upload a CSV with tickers.")
st.caption("This tool uses limited sample universes by default. For broad-market auto-scan, upload a curated CSV (ex: Russell2000 tickers). Always cross-check outputs. Educational use only.")
