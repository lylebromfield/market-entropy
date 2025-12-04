import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ripser import ripser
from sklearn.manifold import MDS

st.set_page_config(
    page_title="Market Entropy",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
    }
    .big-font {
        font-size: 30px !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def fetch_market_data(tickers, benchmark_ticker):
    data = yf.download(tickers, start="2000-01-01", interval="1d", auto_adjust=True)['Close']
    mag7_returns = np.log(data / data.shift(1)).dropna()
    
    sp500 = yf.download(benchmark_ticker, start="2000-01-01", interval="1d", auto_adjust=True)['Close']
    if isinstance(sp500, pd.DataFrame):
        sp500 = sp500.squeeze()
        
    return mag7_returns, sp500

@st.cache_data
def run_tda_pipeline(log_returns, window_size, stride=1):
    n_samples = len(log_returns)
    entropies = []
    dates = []
    
    for i in range(window_size, n_samples, stride):
        window_data = log_returns.iloc[i-window_size:i]
        corr_matrix = window_data.corr().values
        dist_matrix = np.sqrt(2 * (1 - corr_matrix))
        np.fill_diagonal(dist_matrix, 0)
        
        result = ripser(dist_matrix, maxdim=2, distance_matrix=True)
        diagrams = result['dgms']
        
        entropy = 0
        for dim_diagram in diagrams:
            if len(dim_diagram) > 0:
                births = dim_diagram[:, 0]
                deaths = dim_diagram[:, 1]
                lifetimes = deaths - births
                lifetimes = lifetimes[np.isfinite(lifetimes) & (lifetimes > 0)]
                if len(lifetimes) > 0:
                    lifetimes_norm = lifetimes / lifetimes.sum()
                    entropy -= np.sum(lifetimes_norm * np.log(lifetimes_norm + 1e-16))
        
        entropies.append(entropy)
        dates.append(log_returns.index[i])
    
    return pd.Series(entropies, index=dates)

@st.cache_data
def run_takens_pipeline(price_series, window_size, stride=1, dimension=3, delay=1):
    returns = np.log(price_series / price_series.shift(1)).dropna()
    n_samples = len(returns)
    entropies = []
    dates = []
    
    for i in range(window_size, n_samples, stride):
        window_data = returns.iloc[i-window_size:i].values
        
        # Takens embedding
        embedded = []
        for j in range(len(window_data) - (dimension - 1) * delay):
            embedded.append([window_data[j + k * delay] for k in range(dimension)])
        embedded = np.array(embedded)
        
        if len(embedded) > 1:
            result = ripser(embedded, maxdim=1)
            diagrams = result['dgms']
            
            entropy = 0
            for dim_diagram in diagrams:
                if len(dim_diagram) > 0:
                    births = dim_diagram[:, 0]
                    deaths = dim_diagram[:, 1]
                    lifetimes = deaths - births
                    lifetimes = lifetimes[np.isfinite(lifetimes) & (lifetimes > 0)]
                    if len(lifetimes) > 0:
                        lifetimes_norm = lifetimes / lifetimes.sum()
                        entropy -= np.sum(lifetimes_norm * np.log(lifetimes_norm + 1e-16))
            
            entropies.append(entropy)
            dates.append(returns.index[i])
    
    return pd.Series(entropies, index=dates)

def get_snapshot_topology(log_returns, target_date, window_size, tickers):
    if target_date not in log_returns.index:
        target_idx = log_returns.index.get_indexer([target_date], method='nearest')[0]
    else:
        target_idx = log_returns.index.get_loc(target_date)
        
    if target_idx < window_size:
        return None, None
        
    window_data = log_returns.iloc[target_idx-window_size : target_idx]
    
    corr_matrix = window_data.corr()
    dist_matrix = np.sqrt(2 * (1 - corr_matrix))
    np.fill_diagonal(dist_matrix.values, 0)
    
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, normalized_stress=False)
    coords = mds.fit_transform(dist_matrix)
    
    return coords, corr_matrix

with st.sidebar:
    st.header("⚙️ Settings")
    
    window_size = st.slider("Lookback Window", 30, 90, 60)
    smoothing_span = st.slider("Smoothing", 1, 50, 10)
    
    st.markdown("---")
    st.header("🚨 Thresholds")
    percentile_threshold = st.slider("Critical %", 1, 20, 5)
    
    st.markdown("---")
    st.header("📅 Date Range")
    min_year = 2000
    max_year = pd.Timestamp.now().year
    
    date_range = st.slider("Years", min_year, max_year, (2018, max_year))
    start_date = pd.to_datetime(f"{date_range[0]}-01-01")
    end_date = pd.to_datetime(f"{date_range[1]}-12-31")

st.title("Market Entropy")

with st.spinner(f"Computing topology..."):
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA']
    mag7_returns, sp500_price = fetch_market_data(tickers, "^GSPC")
    
    # Ensure dates are timezone-aware if needed
    if mag7_returns.index.tz is not None:
        start_date = pd.Timestamp(start_date).tz_localize(mag7_returns.index.tz)
        end_date = pd.Timestamp(end_date).tz_localize(mag7_returns.index.tz)
    
    # Find common date range between both datasets
    common_start = max(mag7_returns.index[0], sp500_price.index[0], start_date)
    common_end = min(mag7_returns.index[-1], sp500_price.index[-1], end_date)
    
    # Align both datasets to the same date range
    mag7_returns = mag7_returns.loc[common_start:common_end]
    sp500_price = sp500_price.loc[common_start:common_end]
    
    # Further align to exact same index to ensure stride produces matching dates
    common_index = mag7_returns.index.intersection(sp500_price.index)
    mag7_returns = mag7_returns.loc[common_index]
    sp500_price = sp500_price.loc[common_index]
    
    if len(mag7_returns) < window_size or len(sp500_price) < window_size:
        st.error(f"Insufficient data. Need at least {window_size} days in selected range")
        st.stop()
    
    corr_entropy = run_tda_pipeline(mag7_returns, window_size)
    takens_entropy = run_takens_pipeline(sp500_price, window_size)
    
    if len(corr_entropy) == 0 or len(takens_entropy) == 0:
        st.error("Insufficient data after calculations. Try a larger date range")
        st.stop()

# Processing
common_index = corr_entropy.index.intersection(takens_entropy.index)
corr_entropy = corr_entropy.loc[common_index]
takens_entropy = takens_entropy.loc[common_index]

norm_corr = (corr_entropy - corr_entropy.mean()) / corr_entropy.std()
norm_takens = (takens_entropy - takens_entropy.mean()) / takens_entropy.std()

combined_signal = (norm_corr + norm_takens) / 2.0
signal_smooth = combined_signal.ewm(span=smoothing_span).mean()

historical_threshold_val = signal_smooth.quantile(percentile_threshold / 100.0)
threshold_series = pd.Series(historical_threshold_val, index=signal_smooth.index)

view_signal = signal_smooth
view_threshold = threshold_series
view_sp500 = sp500_price.loc[view_signal.index]

if len(view_signal) == 0:
    st.error("No data in selected range")
    st.stop()

latest_val = float(view_signal.iloc[-1])
price_trend = view_sp500.pct_change(len(view_sp500)//10 if len(view_sp500) > 10 else 1).iloc[-1] 

if latest_val <= historical_threshold_val:
    if price_trend < 0:
        status = "Critical Risk"
        color = "inverse"
        msg = "Low entropy + falling price"
    else:
        status = "Euphoria"
        color = "normal"
        msg = "Low entropy + rising price"
elif latest_val <= historical_threshold_val * 1.05:
    status = "Caution"
    color = "off"
    msg = "Structure weakening"
else:
    status = "Healthy"
    color = "normal"
    msg = "Diverse structure"

tab1, tab2 = st.tabs(["Signal", "Inspector"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Status", status, msg, delta_color=color)
    c2.metric("Entropy (Z)", f"{latest_val:.2f}", f"{latest_val - float(view_signal.iloc[-2]):.2f}")
    c3.metric("Trend", f"{price_trend:.1%}", "Momentum")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=view_signal.index, y=view_signal, name="Entropy", line=dict(color='#00ff00', width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=view_threshold.index, y=view_threshold, name="Threshold", line=dict(color='red', dash='dot')), secondary_y=False)
    fig.add_trace(go.Scatter(x=view_sp500.index, y=view_sp500, name="S&P 500", line=dict(color='rgba(255,255,255,0.2)')), secondary_y=True)
    
    fig.update_layout(template="plotly_dark", height=500, hovermode="x unified", legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig, width='stretch')

with tab2:
    snap_date = st.select_slider("Date", options=view_signal.index, value=view_signal.index[-1], format_func=lambda x: x.strftime('%Y-%m-%d'))
    coords, corr = get_snapshot_topology(mag7_returns, snap_date, window_size, tickers)
    
    if coords is not None:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Correlation Map**")
            mds_fig = go.Figure()
            mds_fig.add_trace(go.Scatter(x=coords[:,0], y=coords[:,1], mode='markers+text', text=tickers,
                                       textposition='top center', textfont=dict(size=12),
                                       marker=dict(size=15, color='#00ff00')))
            for i in range(len(tickers)):
                for j in range(i+1, len(tickers)):
                    if corr.iloc[i,j] > 0.6:
                        mds_fig.add_trace(go.Scatter(x=[coords[i,0], coords[j,0]], y=[coords[i,1], coords[j,1]], 
                                                   mode='lines', line=dict(color=f'rgba(255,0,0,{corr.iloc[i,j]})', width=2), showlegend=False))
            mds_fig.update_layout(template="plotly_dark", showlegend=False, xaxis_visible=False, yaxis_visible=False)
            st.plotly_chart(mds_fig, width='stretch')
        
        with c2:
            st.markdown("**Takens Attractor**")
            idx = sp500_price.index.get_indexer([snap_date], method='nearest')[0]
            if idx > 60:
                win = sp500_price.iloc[idx-60:idx].values
                colors = np.linspace(0, 1, len(win)-2)
                takens_fig = go.Figure(data=[go.Scatter3d(
                    x=win[2:], y=win[1:-1], z=win[:-2], 
                    mode='lines',
                    line=dict(color=colors, colorscale='Viridis', width=4, showscale=True,
                             colorbar=dict(title="Time"))
                )])
                takens_fig.update_layout(template="plotly_dark", scene=dict(xaxis_title="t", yaxis_title="t-1", zaxis_title="t-2"))
                st.plotly_chart(takens_fig, width='stretch')