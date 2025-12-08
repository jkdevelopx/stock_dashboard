"""
Streamlit MCRF Dashboard
Multi-Cycle Rotational Framework - Streamlit app

Features:
- Input a list of tickers (or upload CSV)
- Fetch pricing + volume + basic fundamentals (via yfinance)
- Compute technical & micro-volume signals: 50/200 MA, RSI, momentum, volume anomalies
- Compute composite "Potential Score" combining Momentum, Volume Accumulation, Fundamental proxies
- Show cards per ticker with green/red indicator, charts, metrics, short AI-prompt-ready summary (optional, requires OpenAI key)
- Export results as CSV

Note: This app is for EDUCATIONAL / SIMULATION use only. Not investment advice.

Dependencies:
- streamlit
- yfinance
- pandas
- numpy
- plotly

Run:
streamlit run streamlit_mcrf_dashboard.py

"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
import plotly.graph_objs as go
from typing import List, Dict, Any

st.set_page_config(page_title="MCRF - Micro Volume Scanner", layout="wide")

# ----------------------------- Utilities ---------------------------------

def fetch_price_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval, actions=False)
        if df.empty:
            return pd.DataFrame()
        df = df[['Open','High','Low','Close','Volume']].copy()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        st.error(f"Error fetching {ticker}: {e}")
        return pd.DataFrame()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['MA50'] = df['Close'].rolling(window=50, min_periods=10).mean()
    df['MA200'] = df['Close'].rolling(window=200, min_periods=50).mean()
    df['Return'] = df['Close'].pct_change()
    df['Momentum_21'] = df['Close'] / df['Close'].shift(21) - 1
    df['Vol_MA20'] = df['Volume'].rolling(window=20, min_periods=5).mean()
    # RSI simple
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['RSI14'] = 100.0 - (100.0 / (1.0 + rs))
    # Volume spike
    df['Vol_Spike'] = df['Volume'] / (df['Vol_MA20'] + 1e-9)
    # Higher lows detection (simple)
    df['HL'] = ((df['Low'] > df['Low'].shift(1)) & (df['Low'].shift(1) > df['Low'].shift(2))).astype(int)
    return df


def compute_scores(df: pd.DataFrame, ticker: str, fundamentals: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    if df.empty:
        out['ticker'] = ticker
        out['score'] = -999
        return out
    latest = df.iloc[-1]
    # Momentum score (21-day momentum normalized)
    mom = latest['Momentum_21'] if not np.isnan(latest['Momentum_21']) else 0
    mom_score = np.tanh(mom * 5)  # -1..1
    # RSI score (ideal 40-70)
    rsi = latest['RSI14'] if not np.isnan(latest['RSI14']) else 50
    rsi_score = 1 - abs((rsi - 55) / 55)  # peak near 55
    # Volume accumulation score: recent vol spike and higher lows
    vol_spike = latest['Vol_Spike'] if not np.isnan(latest['Vol_Spike']) else 1
    vol_score = np.tanh((vol_spike - 1) * 1.5)
    hl_signal = df['HL'].rolling(21).sum().iloc[-1] if 'HL' in df.columns else 0
    hl_score = min(hl_signal / 5.0, 1.0)
    # Moving average alignment
    ma_bias = 0
    if latest['Close'] > latest['MA50']:
        ma_bias += 0.5
    if latest['Close'] > latest['MA200']:
        ma_bias += 0.5
    # Fundamental proxies from 'fundamentals' dict (optional)
    revenue_growth = fundamentals.get('revenueGrowth', None)
    gross_margin = fundamentals.get('grossMargins', None)
    fcf_margin = fundamentals.get('freeCashflow', None)
    rev_score = 0
    if revenue_growth is not None:
        try:
            rev_score = np.tanh(float(revenue_growth) / 0.5)
        except: rev_score = 0
    gm_score = 0
    if gross_margin is not None:
        try:
            gm_score = np.tanh(float(gross_margin))
        except: gm_score = 0
    # Composite score weights — tweakable
    composite = (0.35 * mom_score) + (0.2 * rsi_score) + (0.2 * vol_score) + (0.1 * hl_score) + (0.15 * ma_bias) + (0.1 * rev_score) + (0.05 * gm_score)
    # normalize to 0-100
    score_norm = (composite + 2) / 4 * 100
    out['ticker'] = ticker
    out['score'] = round(score_norm, 2)
    out['mom_score'] = round(float(mom_score),3)
    out['rsi'] = round(float(rsi),2)
    out['vol_spike'] = round(float(vol_spike),2)
    out['ma_bias'] = ma_bias
    out['hl_count_21'] = int(hl_signal)
    out['rev_growth'] = revenue_growth
    out['gross_margin'] = gross_margin
    return out


def get_fundamentals(ticker: str) -> Dict[str, Any]:
    # yfinance has limited fundamentals; use info as proxy
    try:
        t = yf.Ticker(ticker)
        info = t.info if hasattr(t, 'info') else {}
        keys = ['sector','industry','marketCap','longBusinessSummary','forwardPE','trailingPE','grossMargins','revenueGrowth','freeCashflow']
        return {k: info.get(k, None) for k in keys}
    except Exception:
        return {}

# ----------------------------- App Layout ---------------------------------

st.title("MCRF — Micro-Volume Stock Discovery Dashboard (Streamlit)")
st.markdown("**Educational / Simulation Only. Not investment advice.**")

with st.sidebar:
    st.header("Controls")
    input_mode = st.radio("Input method", ['Manual tickers','Upload CSV','Example Archetypes'])
    if input_mode == 'Manual tickers':
        tickers_raw = st.text_area("Paste tickers separated by comma (e.g. AAPL,MSFT,TSM)", value="")
        tickers = [t.strip().upper() for t in tickers_raw.split(',') if t.strip()]
    elif input_mode == 'Upload CSV':
        uploaded = st.file_uploader("Upload CSV with column 'ticker'", type=['csv'])
        tickers = []
        if uploaded is not None:
            df_up = pd.read_csv(uploaded)
            if 'ticker' in df_up.columns:
                tickers = [str(x).upper() for x in df_up['ticker'].tolist()]
            else:
                st.error("CSV must contain a 'ticker' column")
    else:
        st.info("Example archetypes will populate sample tickers for demo. You can replace them.")
        example_list = st.selectbox('Choose archetype example set', ['AI Infra Micro-cap (sample)','Robotics & Automation (sample)'])
        if example_list.startswith('AI'):
            tickers = ['SOUN', 'BBAI', 'VLD', 'HIMX', 'INDI'][:7]
        else:
            tickers = ['INDI', 'TSM', 'VLD', 'HIMX', 'IRTC'][:7]

    period = st.selectbox('Price history period', ['6mo','1y','2y','5y'], index=2)
    run_scan = st.button('Run Scan')
    openai_key = st.text_input('Optional: OpenAI API Key (for auto summary)', type='password')
    st.markdown("---")
    st.caption("How to use: provide tickers or upload CSV. Click Run Scan to evaluate. Use OpenAI key to generate AI summaries for each stock (optional).")

if not tickers:
    st.warning('Provide tickers or upload a CSV to run the scan.')
    st.stop()

if run_scan:
    progress = st.progress(0)
    results = []
    all_details = {}
    n = len(tickers)
    for i,t in enumerate(tickers):
        progress.progress(int((i+1)/n*100))
        df = fetch_price_data(t, period=period)
        if df.empty:
            results.append({'ticker': t, 'score': -999})
            continue
        df = compute_indicators(df)
        fund = get_fundamentals(t)
        sc = compute_scores(df, t, fund)
        results.append(sc)
        all_details[t] = {'df': df, 'fund': fund}
    progress.empty()

    # results dataframe
    res_df = pd.DataFrame(results).sort_values('score', ascending=False)

    # Top summary
    st.subheader('Scan Results — Summary')
    col1, col2 = st.columns([3,1])
    with col1:
        st.dataframe(res_df.reset_index(drop=True))
    with col2:
        avg_score = res_df[res_df['score']>-900]['score'].mean()
        st.metric('Average Composite Score', round(float(avg_score),2) if not np.isnan(avg_score) else 'N/A')
        st.metric('Tickers Scanned', len(tickers))

    # Cards per ticker
    st.markdown('---')
    st.subheader('Ticker Cards')
    for idx, row in res_df.iterrows():
        t = row['ticker']
        sc = row['score']
        df = all_details.get(t,{}).get('df', pd.DataFrame())
        fund = all_details.get(t,{}).get('fund', {})
        col1, col2 = st.columns([1,3])
        with col1:
            # indicator
            if sc == -999:
                st.markdown(f"### {t} — No Data")
            else:
                color = 'green' if sc >= 60 else ('orange' if sc >=40 else 'red')
                st.markdown(f"### {t} — **{sc}**")
                st.markdown(f"**Sector:** {fund.get('sector','n/a')})**MarketCap:** {fund.get('marketCap','n/a')}")
                st.markdown(f"**Indicator:** <span style='color:{color};font-weight:bold'>{'Strong' if color=='green' else ('Watch' if color=='orange' else 'Weak')}</span>", unsafe_allow_html=True)
                st.write('Key fundamentals (proxy):')
                st.write({k:v for k,v in fund.items() if v is not None})
        with col2:
            if not df.empty:
                # price chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Close', line=dict(width=1.5)))
                fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='MA50', line=dict(width=1, dash='dash')))
                fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], name='MA200', line=dict(width=1, dash='dot')))
                # volume as bar
                fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', yaxis='y2', opacity=0.2))
                fig.update_layout(margin=dict(l=0,r=0,t=20,b=0), height=300, xaxis_rangeslider_visible=False,
                                  legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
                fig.update_yaxes(title_text='Price', secondary_y=False)
                # secondary y axis for volume
                fig.update_layout(yaxis2=dict(overlaying='y', side='right', showgrid=False, title='Volume'))
                st.plotly_chart(fig, use_container_width=True)

                # key indicators
                c1,c2,c3,c4 = st.columns(4)
                c1.metric('MA Bias', f"{row.get('ma_bias',0)}/1")
                c2.metric('RSI14', row.get('rsi','n/a'))
                c3.metric('Vol Spike', row.get('vol_spike','n/a'))
                c4.metric('HL Count(21)', row.get('hl_count_21',0))

                # optional AI summary
                if openai_key:
                    st.info('OpenAI summary enabled — generating...')
                    # Build prompt (client-side). For security, the app expects user to supply key and will call OpenAI via server-side
                    try:
                        from openai import OpenAI
                        client = OpenAI(api_key=openai_key)
                        prompt = f"You are an analyst. Provide a short educational summary for {t} given fundamentals {fund} and latest technicals RSI {row.get('rsi')} vol_spike {row.get('vol_spike')} score {sc}. Do not give investment advice."
                        resp = client.chat.completions.create(model='gpt-4o-mini', messages=[{"role":"user","content":prompt}], max_tokens=220)
                        summ = resp.choices[0].message.content
                        st.write(summ)
                    except Exception as e:
                        st.warning('OpenAI summary failed or OpenAI SDK not installed. Skipping AI summary.')
            else:
                st.write('No price data to chart.')
        st.markdown('---')

    # Download CSV
    csv = res_df.to_csv(index=False)
    st.download_button('Download Results CSV', csv, file_name='mcrf_scan_results.csv', mime='text/csv')

    # Save snapshot
    if st.button('Save Snapshot (local CSV)'):
        ts = dt.datetime.utcnow().strftime('%Y%m%d_%H%M')
        fn = f'mcrf_snapshot_{ts}.csv'
        res_df.to_csv(fn, index=False)
        st.success(f'Snapshot saved as {fn} (in server running Streamlit)')

else:
    st.info('Ready. Provide tickers and click Run Scan to start.')


# ---------------------------- Footer / Notes -------------------------------
st.markdown('\n---\n')
st.caption("This dashboard is a starting point: refine scoring weights, add data sources (e.g., EDGAR, alternative data, earnings revisions API), and create automated weekly runs on a server for continuous scanning.")
st.caption("Educational use only. Not financial advice.")
