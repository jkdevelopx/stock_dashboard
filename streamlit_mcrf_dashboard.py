# streamlit_mcrf_enhanced.py
"""
MCRF Enhanced Scanner - NEW VERSION with Features 1,2,3,4
- Feature 1: Search individual stocks
- Feature 2: Click-to-explore detailed view
- Feature 3: Better/larger universe lists (S&P500, NASDAQ, sectors)
- Feature 4: Advanced charts (candlestick, volume bars, multiple indicators)

This is a NEW file - original streamlit_mcrf_dashboard.py is untouched
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime
import time
from typing import List, Dict

# Page configuration
st.set_page_config(
    page_title="StockSense Pro - Market Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ---------------------
# --- Utility funcs (same as original) ---
# ---------------------
@st.cache_data(ttl=300)
def fetch_history(ticker: str, period="1y", interval="1d"):
    try:
        time.sleep(0.1)
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, actions=False)
        if df.empty:
            return pd.DataFrame()
        df = df[['Open','High','Low','Close','Volume']].copy()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df = df.dropna(subset=['Close'])
    df['Volume'] = df['Volume'].fillna(0)
    df = df[df['Volume'] > 0]
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or len(df) < 50:
        return df
    df = df.copy()
    df['MA20'] = df['Close'].rolling(20, min_periods=5).mean()
    df['MA50'] = df['Close'].rolling(50, min_periods=10).mean()
    df['MA200'] = df['Close'].rolling(200, min_periods=50).mean()
    df['Momentum_21'] = df['Close'] / df['Close'].shift(21) - 1
    df['Vol_MA20'] = df['Volume'].rolling(20, min_periods=5).mean()
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['RSI14'] = 100 - (100 / (1 + rs))
    df['Vol_Spike'] = df['Volume'] / (df['Vol_MA20'] + 1e-9)
    df['HL'] = ((df['Low'] > df['Low'].shift(1)) & 
                (df['Low'].shift(1) > df['Low'].shift(2))).astype(int)
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

def score_ticker(df: pd.DataFrame, info: dict) -> Dict:
    if df.empty or len(df) < 50:
        return {'score': -999, 'reason': 'insufficient_data', 'mom': 0, 'rsi': 50, 'vol_spike': 1, 'hl21': 0}
    try:
        latest = df.iloc[-1]
        mom = float(latest.get('Momentum_21', 0))
        if pd.isna(mom): mom = 0
        mom_s = np.tanh(mom * 5)
        rsi = float(latest.get('RSI14', 50))
        if pd.isna(rsi): rsi = 50
        rsi_s = 1 - abs((rsi - 55) / 55)
        vol_spike = float(latest.get('Vol_Spike', 1))
        if pd.isna(vol_spike): vol_spike = 1
        vol_s = np.tanh((vol_spike - 1.0) * 1.2)
        hl = int(df['HL'].rolling(21).sum().iloc[-1]) if 'HL' in df.columns else 0
        if pd.isna(hl): hl = 0
        hl_s = min(hl / 5.0, 1.0)
        ma_bias = 0.0
        close = latest['Close']
        ma50 = latest.get('MA50', 0)
        ma200 = latest.get('MA200', 0)
        if not pd.isna(ma50) and close > ma50: ma_bias += 0.4
        if not pd.isna(ma200) and close > ma200: ma_bias += 0.6
        rev_growth = info.get('revenueGrowth', 0.0)
        if rev_growth is None or pd.isna(rev_growth): rev_growth = 0.0
        rev_s = np.tanh(float(rev_growth) * 3)
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
        return {'score': -999, 'reason': 'scoring_error', 'mom': 0, 'rsi': 50, 'vol_spike': 1, 'hl21': 0}

# ---------------------
# --- FEATURE 3: Enhanced Universe Lists ---
# ---------------------
# Small samples (original)
SAMPLE_SMALLCAP = ["AAPL", "MSFT", "NVDA", "TSM", "AMD", "INTC", "QCOM", "AVGO", "CRM", "ORCL"]
SAMPLE_MICRO = ["SOUN", "BBAI", "INDI", "VLD", "IRTC", "HIMX", "ENPH", "FSLR", "PLUG", "SPCE"]

# NEW: S&P 500 Top 50 by market cap (sample - you can expand this)
SP500_TOP50 = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "UNH", "JNJ",
    "V", "XOM", "WMT", "JPM", "MA", "PG", "AVGO", "HD", "CVX", "MRK",
    "ABBV", "LLY", "PEP", "COST", "KO", "ADBE", "TMO", "MCD", "CSCO", "ACN",
    "ABT", "NKE", "DHR", "CRM", "VZ", "WFC", "TXN", "CMCSA", "PM", "BMY",
    "INTC", "AMD", "NEE", "ORCL", "LIN", "UPS", "RTX", "HON", "QCOM", "AMGN"
]

# NEW: NASDAQ 100 Tech Leaders
NASDAQ_TECH = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AVGO", "ADBE", "CSCO",
    "NFLX", "INTC", "AMD", "QCOM", "TXN", "AMAT", "INTU", "ISRG", "MU", "LRCX"
]

# NEW: Sector-specific lists
SECTOR_TECH = ["AAPL", "MSFT", "NVDA", "AMD", "INTC", "QCOM", "AVGO", "CRM", "ADBE", "ORCL", "CSCO"]
SECTOR_HEALTHCARE = ["UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "PFE", "BMY", "AMGN"]
SECTOR_FINANCE = ["JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP", "USB"]
SECTOR_ENERGY = ["XOM", "CVX", "COP", "EOG", "SLB", "MPC", "PSX", "VLO", "OXY", "HAL"]
SECTOR_CONSUMER = ["AMZN", "TSLA", "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "TJX"]

# Themes (original + enhanced)
THEME_AI = ["NVDA", "AMD", "MSFT", "GOOGL", "META", "AVGO", "INTC", "SMCI", "ARM", "PLTR"]
THEME_EV = ["TSLA", "RIVN", "LCID", "NIO", "LI", "XPEV", "F", "GM", "STLA"]
THEME_BIOTECH = ["MRNA", "BNTX", "CRSP", "BEAM", "NTLA", "ILMN", "REGN", "VRTX", "GILD"]

UNIVERSE_OPTIONS = {
    "Quick Samples": {
        "SmallCap Sample (10)": SAMPLE_SMALLCAP,
        "Micro Sample (10)": SAMPLE_MICRO,
    },
    "Major Indices": {
        "S&P 500 Top 50": SP500_TOP50,
        "NASDAQ Tech Leaders": NASDAQ_TECH,
    },
    "By Sector": {
        "Technology": SECTOR_TECH,
        "Healthcare": SECTOR_HEALTHCARE,
        "Financial": SECTOR_FINANCE,
        "Energy": SECTOR_ENERGY,
        "Consumer": SECTOR_CONSUMER,
    },
    "By Theme": {
        "AI & Machine Learning": THEME_AI,
        "Electric Vehicles": THEME_EV,
        "Biotech": THEME_BIOTECH,
    }
}

# ---------------------
# --- FEATURE 4: Advanced Chart Function ---
# ---------------------
def create_advanced_chart(df: pd.DataFrame, ticker: str):
    """Create candlestick chart with volume and indicators"""
    if df.empty:
        return None
    
    # Create subplots: 3 rows
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{ticker} - Price & MAs', 'Volume', 'RSI'),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Row 1: Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add moving averages
    if 'MA20' in df.columns and not df['MA20'].isna().all():
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='MA20', 
                                line=dict(color='orange', width=1)), row=1, col=1)
    if 'MA50' in df.columns and not df['MA50'].isna().all():
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50', 
                                line=dict(color='blue', width=1)), row=1, col=1)
    if 'MA200' in df.columns and not df['MA200'].isna().all():
        fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], name='MA200', 
                                line=dict(color='red', width=1, dash='dash')), row=1, col=1)
    
    # Row 2: Volume bars
    colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' 
              for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors),
        row=2, col=1
    )
    
    # Row 3: RSI
    if 'RSI14' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI14'], name='RSI', line=dict(color='purple')),
            row=3, col=1
        )
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    
    return fig

# ---------------------
# --- FEATURE 2: Detailed View Function ---
# ---------------------
def show_detailed_view(ticker: str, df: pd.DataFrame, info: dict, score_data: dict):
    """Show detailed analysis for a single ticker"""
    st.markdown(f"## 🔍 Detailed Analysis: {ticker}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Score", f"{score_data.get('score', 'N/A')}", 
                 delta="Strong" if score_data.get('score', 0) >= 65 else "Weak")
        st.metric("Current Price", f"${df['Close'].iloc[-1]:.2f}" if not df.empty else "N/A")
        st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}" if not df.empty else "N/A")
    
    with col2:
        st.metric("RSI (14)", f"{score_data.get('rsi', 'N/A')}")
        st.metric("Momentum (21d)", f"{score_data.get('mom', 'N/A')}")
        st.metric("Volume Spike", f"{score_data.get('vol_spike', 'N/A')}x")
    
    with col3:
        st.write("**Company Info:**")
        st.write(f"Name: {info.get('shortName', 'N/A')}")
        st.write(f"Sector: {info.get('sector', 'N/A')}")
        st.write(f"Industry: {info.get('industry', 'N/A')}")
        st.write(f"Market Cap: ${info.get('marketCap', 0)/1e9:.2f}B" if info.get('marketCap') else "N/A")
    
    # Advanced chart
    st.markdown("### 📊 Technical Chart")
    chart = create_advanced_chart(df, ticker)
    if chart:
        st.plotly_chart(chart, use_container_width=True)
    
    # Technical indicators table
    if not df.empty:
        st.markdown("### 📈 Latest Indicators")
        latest = df.iloc[-1]
        indicators_df = pd.DataFrame({
            'Indicator': ['Close', 'MA20', 'MA50', 'MA200', 'RSI', 'Volume', 'Vol vs Avg'],
            'Value': [
                f"${latest['Close']:.2f}",
                f"${latest.get('MA20', 0):.2f}",
                f"${latest.get('MA50', 0):.2f}",
                f"${latest.get('MA200', 0):.2f}",
                f"{latest.get('RSI14', 0):.2f}",
                f"{latest['Volume']:,.0f}",
                f"{latest.get('Vol_Spike', 0):.2f}x"
            ]
        })
        st.table(indicators_df)

# ---------------------
# --- Scan Function (same as original) ---
# ---------------------
def run_scan(ticker_list: List[str], show_progress=True):
    results = []
    details = {}
    
    if show_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    for i, t in enumerate(ticker_list):
        try:
            if show_progress:
                status_text.text(f"Scanning {t} ({i+1}/{len(ticker_list)})...")
                progress_bar.progress((i + 1) / len(ticker_list))
            
            hist = fetch_history(t, period=st.session_state.get('period', '1y'))
            df = clean_df(hist)
            
            if df.empty or len(df) < 50:
                results.append({'ticker': t, 'score': -999})
                details[t] = {'df': pd.DataFrame(), 'info': {}, 
                            'score': {'score': -999, 'reason': 'no_data'}}
                continue
            
            df = compute_indicators(df)
            info = {}
            try:
                ticker_obj = yf.Ticker(t)
                info = ticker_obj.info or {}
            except Exception:
                info = {}
            
            sc = score_ticker(df, info)
            results.append({'ticker': t, 'score': sc.get('score', -999)})
            details[t] = {'df': df, 'info': info, 'score': sc}
            
        except Exception as e:
            results.append({'ticker': t, 'score': -999})
            details[t] = {'df': pd.DataFrame(), 'info': {}, 
                        'score': {'score': -999, 'reason': str(e)}}
    
    if show_progress:
        progress_bar.empty()
        status_text.empty()
    
    out = pd.DataFrame(results).sort_values('score', ascending=False).reset_index(drop=True)
    return out, details

# ---------------------
# --- MAIN UI ---
# ---------------------
# Header with branding
st.markdown('<h1 class="main-header">📊 StockSense Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="tagline">Intelligent Market Analytics Platform | Built by Wichaya Kanlaya</p>', unsafe_allow_html=True)

#st.markdown("**New Features**: Search stocks • Click to explore • Larger universes • Advanced charts")

# Sidebar
st.sidebar.title("⚙️ Settings")
period = st.sidebar.selectbox("History period", ["6mo", "1y", "2y"], index=1)
st.session_state['period'] = period

# Initialize session state
if 'watchlist' not in st.session_state:
    st.session_state['watchlist'] = []
if 'last_update' not in st.session_state:
    st.session_state['last_update'] = None

# ---------------------
# --- FEATURE 1: Search Individual Stocks ---
# ---------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Search Stock")
search_input = st.sidebar.text_input("Enter ticker (e.g., AAPL, TSLA)", key="search_ticker")
search_input = search_input.strip().upper()

if st.sidebar.button("🔎 Search & Analyze") and search_input:
    with st.spinner(f"Analyzing {search_input}..."):
        out, details = run_scan([search_input], show_progress=False)
        if search_input in details and details[search_input]['score'].get('score', -999) > -999:
            st.session_state['selected_ticker'] = search_input
            st.session_state['scan_results'] = out
            st.session_state['scan_details'] = details
            st.session_state['last_update'] = datetime.now()
        else:
            st.sidebar.error(f"❌ Could not fetch data for {search_input}")

# ---------------------
# --- Universe Scanner ---
# ---------------------
st.markdown("## 📊 Universe Scanner")

# Category and universe selection
col1, col2 = st.columns(2)

with col1:
    category = st.selectbox("Select Category", list(UNIVERSE_OPTIONS.keys()))

with col2:
    universe_name = st.selectbox("Select Universe", list(UNIVERSE_OPTIONS[category].keys()))

selected_universe = UNIVERSE_OPTIONS[category][universe_name]
st.info(f"📋 Selected universe: **{universe_name}** ({len(selected_universe)} tickers)")

if st.button(f"🚀 Scan {universe_name}", type="primary"):
    with st.spinner(f"Scanning {len(selected_universe)} tickers..."):
        out, details = run_scan(selected_universe)
        st.session_state['scan_results'] = out
        st.session_state['scan_details'] = details
        st.session_state['last_update'] = datetime.now()
        st.rerun()

# Show last update time
if st.session_state['last_update']:
    st.caption(f"⏰ Last updated: {st.session_state['last_update'].strftime('%Y-%m-%d %H:%M:%S')}")

# ---------------------
# --- FEATURE 2: Display Results with Click-to-Explore ---
# ---------------------
if 'scan_results' in st.session_state and 'scan_details' in st.session_state:
    res = st.session_state['scan_results']
    details = st.session_state['scan_details']
    
    valid_res = res[res['score'] > -999]
    
    if valid_res.empty:
        st.error("❌ No valid data retrieved. Check tickers or try again later.")
    else:
        # NEW: Quick Stats Dashboard
        st.markdown("### 📊 Scan Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_scanned = len(res)
        strong_signals = len(valid_res[valid_res['score'] >= 65])
        moderate_signals = len(valid_res[(valid_res['score'] >= 45) & (valid_res['score'] < 65)])
        weak_signals = len(valid_res[valid_res['score'] < 45])
        avg_score = valid_res['score'].mean()
        
        col1.metric("📈 Total Scanned", total_scanned)
        col2.metric("🟢 Strong", strong_signals)
        col3.metric("🟡 Moderate", moderate_signals)
        col4.metric("🔴 Weak", weak_signals)
        col5.metric("📊 Avg Score", f"{avg_score:.1f}")
        
        st.markdown("---")
        st.markdown("### 🏆 Top Results")
        
        # NEW: Smart Filters
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            signal_filter = st.multiselect(
                "Filter by Signal",
                options=["🟢 Strong (65+)", "🟡 Moderate (45-65)", "🔴 Weak (<45)"],
                default=["🟢 Strong (65+)", "🟡 Moderate (45-65)", "🔴 Weak (<45)"]
            )
        
        with col2:
            # Get unique sectors from details
            sectors = set()
            for ticker in valid_res['ticker']:
                if ticker in details:
                    sector = details[ticker]['info'].get('sector', 'Unknown')
                    if sector and sector != 'Unknown':
                        sectors.add(sector)
            
            sector_filter = st.multiselect(
                "Filter by Sector",
                options=sorted(list(sectors)) if sectors else ["All"],
                default=sorted(list(sectors)) if sectors else ["All"]
            )
        
        with col3:
            show_count = st.selectbox("Show Top", [10, 20, 50, 100], index=1)
        
        # Apply filters
        filtered_df = valid_res.copy()
        
        # Signal filter
        if "🟢 Strong (65+)" not in signal_filter:
            filtered_df = filtered_df[filtered_df['score'] < 65]
        if "🟡 Moderate (45-65)" not in signal_filter:
            filtered_df = filtered_df[(filtered_df['score'] < 45) | (filtered_df['score'] >= 65)]
        if "🔴 Weak (<45)" not in signal_filter:
            filtered_df = filtered_df[filtered_df['score'] >= 45]
        
        # Sector filter
        if sector_filter and sectors:
            filtered_tickers = []
            for ticker in filtered_df['ticker']:
                if ticker in details:
                    sector = details[ticker]['info'].get('sector', 'Unknown')
                    if sector in sector_filter or 'All' in sector_filter:
                        filtered_tickers.append(ticker)
            filtered_df = filtered_df[filtered_df['ticker'].isin(filtered_tickers)]
        
        # Add performance metrics (1-day, 1-week returns)
        display_df = filtered_df.head(show_count).copy()
        display_df['signal'] = display_df['score'].apply(
            lambda x: "🟢 Strong" if x >= 65 else ("🟡 Moderate" if x >= 45 else "🔴 Weak")
        )
        
        # Calculate performance metrics
        performance_data = []
        for ticker in display_df['ticker']:
            if ticker in details and not details[ticker]['df'].empty:
                df = details[ticker]['df']
                
                # 1-day return
                day_return = ((df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1) * 100 if len(df) >= 2 else 0
                
                # 1-week return (5 trading days)
                week_return = ((df['Close'].iloc[-1] / df['Close'].iloc[-6]) - 1) * 100 if len(df) >= 6 else 0
                
                # 1-month return (21 trading days)
                month_return = ((df['Close'].iloc[-1] / df['Close'].iloc[-22]) - 1) * 100 if len(df) >= 22 else 0
                
                sector = details[ticker]['info'].get('sector', 'N/A')
                
                performance_data.append({
                    '1D %': f"{day_return:+.2f}%",
                    '1W %': f"{week_return:+.2f}%",
                    '1M %': f"{month_return:+.2f}%",
                    'Sector': sector
                })
            else:
                performance_data.append({
                    '1D %': 'N/A',
                    '1W %': 'N/A',
                    '1M %': 'N/A',
                    'Sector': 'N/A'
                })
        
        perf_df = pd.DataFrame(performance_data)
        
        # Combine dataframes
        display_df = display_df[['ticker', 'signal', 'score']].reset_index(drop=True)
        final_df = pd.concat([display_df, perf_df], axis=1)
        
        st.dataframe(final_df, use_container_width=True, hide_index=True)
        
        # FEATURE 2: Click to explore with better layout
        st.markdown("---")
        st.markdown("### 🔍 Detailed Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            ticker_options = filtered_df['ticker'].tolist() if not filtered_df.empty else valid_res['ticker'].tolist()
            
            # Create better formatted options with emoji indicators
            def format_ticker_option(ticker):
                if not ticker:
                    return "-- Select a ticker to analyze --"
                score_df = filtered_df if not filtered_df.empty else valid_res
                score = score_df[score_df['ticker']==ticker]['score'].values[0] if ticker in score_df['ticker'].values else 0
                emoji = "🟢" if score >= 65 else ("🟡" if score >= 45 else "🔴")
                return f"{emoji} {ticker} — Score: {score}"
            
            selected = st.selectbox(
                "Choose a ticker for in-depth analysis:",
                options=[''] + ticker_options,
                format_func=format_ticker_option
            )
        
        with col2:
            if selected and selected in details:
                if st.button(f"⭐ Add {selected} to Watchlist", use_container_width=True):
                    if selected not in st.session_state['watchlist']:
                        st.session_state['watchlist'].append(selected)
                        st.success(f"✅ Added!")
                    else:
                        st.info(f"Already in watchlist")
                
                # Quick actions
                info = details[selected]['info']
                if info.get('website'):
                    st.link_button("🌐 Company Website", info['website'], use_container_width=True)
        
        if selected and selected in details:
            show_detailed_view(
                selected,
                details[selected]['df'],
                details[selected]['info'],
                details[selected]['score']
            )

# ---------------------
# --- Watchlist Management ---
# ---------------------
st.sidebar.markdown("---")
st.sidebar.subheader("⭐ My Watchlist")

if st.session_state['watchlist']:
    for ticker in st.session_state['watchlist']:
        col1, col2 = st.sidebar.columns([3, 1])
        col1.write(ticker)
        if col2.button("🗑️", key=f"remove_{ticker}"):
            st.session_state['watchlist'].remove(ticker)
            st.rerun()
    
    if st.sidebar.button("📊 Scan Watchlist"):
        with st.spinner("Scanning watchlist..."):
            out, details = run_scan(st.session_state['watchlist'])
            st.session_state['scan_results'] = out
            st.session_state['scan_details'] = details
            st.session_state['last_update'] = datetime.now()
            st.rerun()
    
    if st.sidebar.button("🗑️ Clear Watchlist"):
        st.session_state['watchlist'] = []
        st.rerun()
else:
    st.sidebar.write("(empty)")

# Export functionality
if 'scan_results' in st.session_state:
    st.sidebar.markdown("---")
    csv_data = st.session_state['scan_results'].to_csv(index=False)
    st.sidebar.download_button(
        "📥 Download Results (CSV)",
        csv_data,
        file_name=f"mcrf_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
**📚 How to use:**
1. **Search**: Enter any ticker in the sidebar to analyze a specific stock
2. **Universe Scan**: Choose a category and universe, then click Scan
3. **Explore**: Click on any ticker in the results to see detailed analysis
4. **Watchlist**: Add interesting stocks to your watchlist for quick access

⚠️ **Disclaimer**: Educational use only. Not financial advice. Always do your own research.
""")