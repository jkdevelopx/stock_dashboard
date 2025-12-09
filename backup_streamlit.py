# streamlit_mcrf_dashboard.py
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
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from datetime import datetime
import math
from typing import List, Dict
import time

st.set_page_config(page_title="MCRF Auto Scanner", layout="wide", initial_sidebar_state="expanded")

# ---------------------
# --- Utility funcs ---
# ---------------------
@st.cache_data(ttl=300)
def fetch_history(ticker: str, period="1y", interval="1d"):
    """Fetch historical data with better error handling"""
    try:
        # Add small delay to avoid rate limiting
        time.sleep(0.1)
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, actions=False)
        if df.empty:
            print(f"No data returned for {ticker}")
            return pd.DataFrame()
        df = df[['Open','High','Low','Close','Volume']].copy()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataframe with better handling"""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df = df.dropna(subset=['Close'])
    df['Volume'] = df['Volume'].fillna(0)
    # Remove forward-fill as it can cause issues
    df = df[df['Volume'] > 0]
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators"""
    if df.empty or len(df) < 50:  # Need minimum data points
        return df
    df = df.copy()
    
    # Moving averages
    df['MA20'] = df['Close'].rolling(20, min_periods=5).mean()
    df['MA50'] = df['Close'].rolling(50, min_periods=10).mean()
    df['MA200'] = df['Close'].rolling(200, min_periods=50).mean()
    
    # Momentum
    df['Momentum_21'] = df['Close'] / df['Close'].shift(21) - 1
    
    # Volume indicators
    df['Vol_MA20'] = df['Volume'].rolling(20, min_periods=5).mean()
    
    # RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['RSI14'] = 100 - (100 / (1 + rs))
    
    # Volume spike
    df['Vol_Spike'] = df['Volume'] / (df['Vol_MA20'] + 1e-9)
    
    # Higher lows
    df['HL'] = ((df['Low'] > df['Low'].shift(1)) & 
                (df['Low'].shift(1) > df['Low'].shift(2))).astype(int)
    
    return df

def score_ticker(df: pd.DataFrame, info: dict) -> Dict:
    """Score ticker with better error handling"""
    if df.empty or len(df) < 50:
        return {'score': -999, 'reason': 'insufficient_data', 'mom': 0, 'rsi': 50, 'vol_spike': 1, 'hl21': 0}
    
    try:
        latest = df.iloc[-1]
        
        # Momentum score
        mom = float(latest.get('Momentum_21', 0))
        if pd.isna(mom):
            mom = 0
        mom_s = np.tanh(mom * 5)
        
        # RSI score
        rsi = float(latest.get('RSI14', 50))
        if pd.isna(rsi):
            rsi = 50
        rsi_s = 1 - abs((rsi - 55) / 55)
        
        # Volume spike score
        vol_spike = float(latest.get('Vol_Spike', 1))
        if pd.isna(vol_spike):
            vol_spike = 1
        vol_s = np.tanh((vol_spike - 1.0) * 1.2)
        
        # Higher lows
        hl = int(df['HL'].rolling(21).sum().iloc[-1]) if 'HL' in df.columns else 0
        if pd.isna(hl):
            hl = 0
        hl_s = min(hl / 5.0, 1.0)
        
        # MA alignment
        ma_bias = 0.0
        close = latest['Close']
        ma50 = latest.get('MA50', 0)
        ma200 = latest.get('MA200', 0)
        
        if not pd.isna(ma50) and close > ma50:
            ma_bias += 0.4
        if not pd.isna(ma200) and close > ma200:
            ma_bias += 0.6
        
        # Revenue growth
        rev_growth = info.get('revenueGrowth', 0.0)
        if rev_growth is None or pd.isna(rev_growth):
            rev_growth = 0.0
        rev_s = np.tanh(float(rev_growth) * 3)
        
        # Composite score
        composite = (0.32 * mom_s + 0.18 * rsi_s + 0.20 * vol_s + 
                    0.10 * hl_s + 0.12 * ma_bias + 0.08 * rev_s)
        score = (composite + 1) / 2 * 100
        
        return {
            'score': round(float(score), 2),
            'mom': round(mom_s, 3),
            'rsi': round(rsi, 2),
            'vol_spike': round(vol_spike, 2),
            'hl21': int(hl)
        }
    except Exception as e:
        print(f"Error scoring ticker: {e}")
        return {'score': -999, 'reason': 'scoring_error', 'mom': 0, 'rsi': 50, 'vol_spike': 1, 'hl21': 0}

# ---------------------
# --- Sample universes ---
# ---------------------
SAMPLE_SMALLCAP = ["AAPL", "MSFT", "NVDA", "TSM", "AMD", "INTC", "QCOM", "AVGO", "CRM", "ORCL"]
SAMPLE_MICRO = ["SOUN", "BBAI", "INDI", "VLD", "IRTC", "HIMX", "ENPH", "FSLR", "PLUG", "SPCE"]
THEME_AI = ["NVDA", "AMD", "SMCI", "AVGO", "MSFT", "GOOGL", "INTC"]
THEME_ROBOTICS = ["IRTC", "FSLR", "VLD", "INDI", "HIMX"]
THEME_BIO = ["IRTC", "CRSP", "BEAM", "NTLA", "ILMN"]
THEME_EV = ["TSLA", "NIO", "LI", "XPEV", "FSR"]

# ---------------------
# --- UI: Sidebar ---
# ---------------------
st.sidebar.title("MCRF — Easy Scanner")
st.sidebar.markdown("One-click scanner for users")

scan_mode = st.sidebar.radio("Scanner mode", ["SCAN NOW (Auto sample)", "Category Scanner", "Upload CSV"])
period = st.sidebar.selectbox("History period", ["6mo", "1y", "2y"], index=1)
topk = st.sidebar.slider("Top picks to show", value=7, min_value=3, max_value=15)
max_check = st.sidebar.slider("Max tickers to attempt (only for CSV or custom)", 
                               min_value=50, max_value=1000, value=300, step=50)

st.title("MCRF — Auto Scanner (User Friendly)")
st.markdown("**One-click SCAN NOW** — system finds stocks for you from sample universes. Educational only.")

# ---------------------
# --- Main: SCAN NOW ---
# ---------------------
def run_scan(ticker_list: List[str]):
    """Main scan function with progress tracking"""
    results = []
    details = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, t in enumerate(ticker_list):
        try:
            status_text.text(f"Scanning {t} ({i+1}/{len(ticker_list)})...")
            progress_bar.progress((i + 1) / len(ticker_list))
            
            hist = fetch_history(t, period=period)
            df = clean_df(hist)
            
            if df.empty or len(df) < 50:
                results.append({'ticker': t, 'score': -999})
                details[t] = {'df': pd.DataFrame(), 'info': {}, 
                            'score': {'score': -999, 'reason': 'no_data'}}
                continue
            
            df = compute_indicators(df)
            
            # Fetch info with error handling
            info = {}
            try:
                ticker_obj = yf.Ticker(t)
                info = ticker_obj.info or {}
            except Exception as e:
                print(f"Error fetching info for {t}: {e}")
                info = {}
            
            sc = score_ticker(df, info)
            results.append({'ticker': t, 'score': sc.get('score', -999)})
            details[t] = {'df': df, 'info': info, 'score': sc}
            
        except Exception as e:
            print(f"Error processing {t}: {e}")
            results.append({'ticker': t, 'score': -999})
            details[t] = {'df': pd.DataFrame(), 'info': {}, 
                        'score': {'score': -999, 'reason': str(e)}}
    
    progress_bar.empty()
    status_text.empty()
    
    out = pd.DataFrame(results).sort_values('score', ascending=False).reset_index(drop=True)
    return out, details

# Main actions
if scan_mode == "SCAN NOW (Auto sample)":
    st.subheader("SCAN NOW")
    st.write("Auto-scan uses sample universes optimized for user simplicity.")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("Quick: SmallCap sample"):
            tickers = SAMPLE_SMALLCAP
            st.session_state['last_universe'] = "SmallCap sample"
            st.session_state['last_list'] = tickers
            with st.spinner("Scanning..."):
                out, details = run_scan(tickers)
                st.session_state['last_result'] = out
                st.session_state['details'] = details
                st.rerun()
    
    with col2:
        if st.button("Quick: Micro sample"):
            tickers = SAMPLE_MICRO
            st.session_state['last_universe'] = "Micro sample"
            st.session_state['last_list'] = tickers
            with st.spinner("Scanning..."):
                out, details = run_scan(tickers)
                st.session_state['last_result'] = out
                st.session_state['details'] = details
                st.rerun()
    
    with col3:
        if st.button("Quick: AI Theme"):
            tickers = THEME_AI
            st.session_state['last_universe'] = "AI theme"
            st.session_state['last_list'] = tickers
            with st.spinner("Scanning..."):
                out, details = run_scan(tickers)
                st.session_state['last_result'] = out
                st.session_state['details'] = details
                st.rerun()

    if 'last_result' in st.session_state:
        res = st.session_state['last_result']
        details = st.session_state['details']
        
        # Filter out -999 scores
        valid_res = res[res['score'] > -999]
        
        if valid_res.empty:
            st.error("❌ No valid data retrieved. This could be due to:")
            st.markdown("""
            - **Network issues** - Check your internet connection
            - **Yahoo Finance API limits** - Wait a few minutes and try again
            - **Invalid tickers** - Some tickers may be delisted or incorrect
            """)
        else:
            st.markdown("### Top picks")
            st.table(valid_res.head(topk))
            
            # Show cards for top-k
            picks = valid_res.head(topk)['ticker'].tolist()
            for t in picks:
                if t not in details:
                    continue
                    
                sc = details[t]['score']
                df = details[t]['df']
                info = details[t]['info']
                
                colA, colB = st.columns([1, 3])
                with colA:
                    score_val = sc.get('score', -999)
                    color = "🟢" if score_val >= 65 else ("🟡" if score_val >= 45 else "🔴")
                    st.markdown(f"#### {t} {color} — Score {score_val}")
                    st.write(info.get('shortName') or info.get('longName') or "N/A")
                    st.write(f"Sector: {info.get('sector', 'n/a')}")
                    
                    add = st.button(f"Add {t} to Watchlist", key=f"add_{t}")
                    if add:
                        if 'watchlist' not in st.session_state:
                            st.session_state['watchlist'] = []
                        if t not in st.session_state['watchlist']:
                            st.session_state['watchlist'].append(t)
                            st.success(f"{t} added to watchlist")
                
                with colB:
                    if not df.empty:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close'))
                        if 'MA50' in df.columns and not df['MA50'].isna().all():
                            fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], 
                                                    name='MA50', line=dict(dash='dash')))
                        fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write("No price data available")

elif scan_mode == "Category Scanner":
    st.subheader("Category Scanner")
    cat = st.selectbox("Choose category", 
                      ["AI", "Robotics", "Biotech", "EV", "Semiconductors", "Micro-cap"])
    pre = []
    if cat == "AI": pre = THEME_AI
    if cat == "Robotics": pre = THEME_ROBOTICS
    if cat == "Biotech": pre = THEME_BIO
    if cat == "EV": pre = THEME_EV
    if cat == "Semiconductors": pre = ["TSM", "ASML", "NVDA", "INTC", "MU", "AMD"]
    if cat == "Micro-cap": pre = SAMPLE_MICRO
    
    st.write("Selected sample universe:", pre)
    if st.button("Scan category"):
        with st.spinner("Scanning category..."):
            out, details = run_scan(pre)
            st.session_state['last_result'] = out
            st.session_state['details'] = details
            
            valid_res = out[out['score'] > -999]
            if not valid_res.empty:
                st.table(valid_res.head(topk))
            else:
                st.error("No valid data retrieved. Please try again later.")

else:  # Upload CSV
    st.subheader("Upload CSV (column: ticker)")
    uploaded = st.file_uploader("Upload CSV file with 'ticker' column", type=["csv"])
    if uploaded is not None:
        df_up = pd.read_csv(uploaded)
        if 'ticker' not in df_up.columns:
            st.error("CSV must contain 'ticker' column")
        else:
            tickers = [str(x).strip().upper() for x in df_up['ticker'].tolist()][:max_check]
            st.write(f"Found {len(tickers)} tickers in CSV (limited to {max_check})")
            if st.button("Run CSV scan"):
                with st.spinner("Running CSV scan. This may take time for large lists."):
                    out, details = run_scan(tickers)
                    st.session_state['last_result'] = out
                    st.session_state['details'] = details
                    
                    valid_res = out[out['score'] > -999]
                    if not valid_res.empty:
                        st.table(valid_res.head(topk))
                    else:
                        st.error("No valid data retrieved. Check your tickers and try again.")

# Watchlist & export
st.sidebar.markdown("---")
st.sidebar.subheader("Watchlist")
if 'watchlist' not in st.session_state:
    st.session_state['watchlist'] = []

if st.session_state['watchlist']:
    st.sidebar.write(st.session_state['watchlist'])
    if st.sidebar.button("Scan my watchlist"):
        out, details = run_scan(st.session_state['watchlist'])
        st.session_state['last_result'] = out
        st.session_state['details'] = details
        st.sidebar.success("Watchlist scanned — check main view")
        st.rerun()
else:
    st.sidebar.write("(empty)")

if st.sidebar.button("Export last result CSV") and 'last_result' in st.session_state:
    csv_data = st.session_state['last_result'].to_csv(index=False)
    st.sidebar.download_button(
        "Download CSV",
        csv_data,
        file_name=f"mcrf_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("**How to use**: Choose SCAN NOW or Category Scanner → Click a quick sample button → View Top picks → Add to Watchlist → Scan Watchlist. For full universe scan, upload CSV with tickers.")
st.caption("⚠️ Educational use only. Not investment advice. Data from Yahoo Finance may have delays or errors.")