import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ripser import ripser
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import networkx as nx
from scipy.stats import entropy as scipy_entropy
try:
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import Amplitude
    GTDA_AVAILABLE = True
except ImportError:
    GTDA_AVAILABLE = False
from typing import Tuple, Optional
from validation_module import EntropyValidator, create_validation_plots
from optimizer_module import EntropyOptimizer
from takens_diagnostics import find_optimal_delay, false_nearest_neighbors, find_optimal_dimension, compute_autocorrelation

ENTROPY_MAX_DIM = 2
ENTROPY_REGULARIZATION = 1e-16
TAKENS_DIMENSION = 4
TAKENS_DELAY = 1
ROLLING_WINDOW_DAYS = 252
Z_SCORE_EXTREME_THRESHOLD = -2.0
Z_SCORE_SEVERE_THRESHOLD = -3.0
PRICE_DRAWDOWN_THRESHOLD = -0.05
PRICE_DRAWDOWN_60D_THRESHOLD = -0.05
FNN_DIMENSION_THRESHOLD = 1.0
CORRELATION_THRESHOLD_DEFAULT = 0.6
CORRELATION_DISTANCE_COEFFICIENT = np.sqrt(2)
P_VALUE_SIGNIFICANCE = 0.05
PROGRESS_BAR_MAX = 1.0

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
def fetch_market_data(tickers: list[str], benchmark_ticker: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Fetch OHLCV data for multiple tickers and compute log returns.
    
    Args:
        tickers: List of stock ticker symbols (11 SPDR sector ETFs)
        benchmark_ticker: Benchmark index ticker (e.g., '^GSPC' for S&P 500)
    
    Returns:
        Tuple of (log_returns DataFrame, benchmark price Series)
    """
    try:
        data = yf.download(tickers, start="2000-01-01", interval="1d", auto_adjust=True, group_by='column', progress=False)
        if 'Close' in data.columns:
            data = data['Close']
        log_returns = np.log(data / data.shift(1)).dropna()
        
        benchmark_data = yf.download(benchmark_ticker, start="2000-01-01", interval="1d", auto_adjust=True, progress=False)
        if 'Close' in benchmark_data.columns:
            benchmark_data = benchmark_data['Close']
        if isinstance(benchmark_data, pd.DataFrame):
            benchmark_data = benchmark_data.squeeze()
            
        return log_returns, benchmark_data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame(), pd.Series(dtype=float)

def _compute_data_hash(data: pd.DataFrame | pd.Series) -> int:
    """
    Compute hash of data for caching purposes.
    
    Uses index, shape, and first value to create a unique identifier.
    This enables fast cache invalidation when data changes.
    
    Args:
        data: DataFrame or Series to hash
        
    Returns:
        Integer hash of the data for cache key
    """
    return hash((tuple(data.index), data.shape, float(data.iloc[0].sum() if hasattr(data.iloc[0], 'sum') else data.iloc[0])))

@st.cache_data
def run_tda_pipeline(log_returns: pd.DataFrame, window_size: int, stride: int = 1) -> pd.Series:
    """
    Compute topological data analysis entropy using persistent homology.
    
    Wrapper for caching that calls the implementation function with data hash.
    
    Args:
        log_returns: DataFrame of log returns (shape: n_samples × n_assets)
        window_size: Sliding window size in days
        stride: Step size between windows
    
    Returns:
        Series of entropy values indexed by dates
    """
    data_hash = _compute_data_hash(log_returns)
    return _run_tda_pipeline_impl(log_returns, window_size, stride, data_hash)

@st.cache_data
def _run_tda_pipeline_impl(log_returns: pd.DataFrame, window_size: int, stride: int, _data_hash: int) -> pd.Series:
    """
    Implementation of TDA entropy computation using Rips persistent homology.
    
    For each sliding window:
    1. Compute correlation matrix of returns
    2. Convert to distance matrix via sqrt(2(1-correlation))
    3. Compute persistent homology with Ripser
    4. Calculate Shannon entropy of persistence lifetimes
    
    Args:
        log_returns: DataFrame of log returns
        window_size: Window size in days
        stride: Step size between windows
        _data_hash: Cache key (unused but required for Streamlit caching)
    
    Returns:
        Series of entropy values
    """
    n_samples = len(log_returns)
    entropies = []
    dates = []
    
    progress_bar = st.progress(0, text="Computing correlation-based topology...")
    total_windows = (n_samples - window_size) // stride
    
    for idx, i in enumerate(range(window_size, n_samples, stride)):
        window_data = log_returns.iloc[i-window_size:i]
        corr_matrix = window_data.corr().values
        dist_matrix = CORRELATION_DISTANCE_COEFFICIENT * np.sqrt(1 - corr_matrix)
        np.fill_diagonal(dist_matrix, 0)
        
        result = ripser(dist_matrix, maxdim=ENTROPY_MAX_DIM, distance_matrix=True)
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
                    entropy -= np.sum(lifetimes_norm * np.log(lifetimes_norm + ENTROPY_REGULARIZATION))
        
        entropies.append(entropy)
        dates.append(log_returns.index[i])
        
        if idx % max(1, total_windows // 20) == 0:
            progress_bar.progress(min(idx / total_windows, PROGRESS_BAR_MAX), text="Computing correlation-based topology...")
    
    progress_bar.empty()
    return pd.Series(entropies, index=dates)

@st.cache_data
def run_takens_pipeline(price_series: pd.Series, window_size: int, stride: int = 1, 
                        dimension: int = TAKENS_DIMENSION, delay: int = TAKENS_DELAY) -> pd.Series:
    """
    Compute topological data analysis entropy using Takens time-delay embedding.
    
    Wrapper for caching that calls the implementation function with data hash.
    
    Args:
        price_series: Price series for single asset
        window_size: Sliding window size in days
        stride: Step size between windows
        dimension: Embedding dimension (default 3)
        delay: Time delay between coordinates (default 1)
    
    Returns:
        Series of entropy values indexed by dates
    """
    data_hash = hash((tuple(price_series.index), len(price_series), float(price_series.iloc[0])))
    return _run_takens_pipeline_impl(price_series, window_size, stride, dimension, delay, data_hash)

@st.cache_data
def _run_takens_pipeline_impl(price_series: pd.Series, window_size: int, stride: int, 
                              dimension: int, delay: int, _data_hash: int) -> pd.Series:
    """
    Implementation of Takens embedding entropy computation.
    
    For each sliding window:
    1. Compute log returns
    2. Reconstruct attractor using Takens time-delay embedding: [y(t), y(t-τ), y(t-2τ), ...]
    3. Compute persistent homology
    4. Calculate Shannon entropy of persistence lifetimes
    
    Args:
        price_series: Price series for single asset
        window_size: Window size in days
        stride: Step size between windows
        dimension: Embedding dimension
        delay: Time delay between embedding coordinates
        _data_hash: Cache key (unused but required for Streamlit caching)
    
    Returns:
        Series of entropy values
    """
    returns = np.log(price_series / price_series.shift(1)).dropna()
    n_samples = len(returns)
    entropies = []
    dates = []
    
    progress_bar = st.progress(0, text="Computing Takens embedding topology...")
    total_windows = (n_samples - window_size) // stride
    
    for idx, i in enumerate(range(window_size, n_samples, stride)):
        window_data = returns.iloc[i-window_size:i].values
        
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
                        entropy -= np.sum(lifetimes_norm * np.log(lifetimes_norm + ENTROPY_REGULARIZATION))
            
            entropies.append(entropy)
            dates.append(returns.index[i])
        
        if idx % max(1, total_windows // 20) == 0:
            progress_bar.progress(min(idx / total_windows, PROGRESS_BAR_MAX), text="Computing Takens embedding topology...")
    
    progress_bar.empty()
    return pd.Series(entropies, index=dates)

@st.cache_data
def run_shannon_pipeline(price_series: pd.Series, window_size: int, stride: int = 1) -> pd.Series:
    """
    Compute Shannon entropy of return distribution (measures chaos/surprise).
    
    Wrapper for caching that calls the implementation function with data hash.
    
    Args:
        price_series: Price series for single asset
        window_size: Sliding window size in days
        stride: Step size between windows
    
    Returns:
        Series of Shannon entropy values indexed by dates
    """
    data_hash = hash((tuple(price_series.index), len(price_series), float(price_series.iloc[0])))
    return _run_shannon_pipeline_impl(price_series, window_size, stride, data_hash)

@st.cache_data
def _run_shannon_pipeline_impl(price_series: pd.Series, window_size: int, stride: int, _data_hash: int) -> pd.Series:
    """
    Implementation of Shannon entropy computation.
    
    For each sliding window:
    1. Compute log returns
    2. Create histogram to approximate probability distribution
    3. Calculate Shannon entropy: -sum(p * log(p))
    
    High entropy = fat tails, wild moves (crisis/crash)
    Low entropy = normal distribution, predictable (quiet market)
    
    Args:
        price_series: Price series for single asset
        window_size: Window size in days
        stride: Step size between windows
        _data_hash: Cache key (unused but required for Streamlit caching)
    
    Returns:
        Series of Shannon entropy values
    """
    returns = np.log(price_series / price_series.shift(1)).dropna()
    n_samples = len(returns)
    shannon_vals = []
    dates = []
    
    progress_bar = st.progress(0, text="Computing return distribution entropy...")
    total_windows = (n_samples - window_size) // stride
    
    for idx, i in enumerate(range(window_size, n_samples, stride)):
        window_data = returns.iloc[i-window_size:i].values
        
        # Create histogram to approximate probability distribution
        counts, _ = np.histogram(window_data, bins='auto', density=True)
        
        # Calculate Shannon entropy (add epsilon to avoid log(0))
        S = scipy_entropy(counts + 1e-10)
        
        shannon_vals.append(S)
        dates.append(returns.index[i])
        
        if idx % max(1, total_windows // 20) == 0:
            progress_bar.progress(min(idx / total_windows, PROGRESS_BAR_MAX), text="Computing return distribution entropy...")
    
    progress_bar.empty()
    return pd.Series(shannon_vals, index=dates)

@st.cache_data
def run_drift_pipeline(log_returns: pd.DataFrame, window_size: int, stride: int = 1) -> pd.Series:
    """
    Compute Wasserstein drift (topological speed) using giotto-tda.
    
    Measures the rate of change in market structure by comparing
    persistence diagrams at time T vs T-1. High drift indicates
    violent structural change (shocks), low drift indicates stable structure.
    
    Args:
        log_returns: DataFrame of log returns
        window_size: Sliding window size in days
        stride: Step size between windows
    
    Returns:
        Series of drift values indexed by dates
    """
    if not GTDA_AVAILABLE:
        st.warning("giotto-tda not available. Drift calculation disabled.")
        return pd.Series(dtype=float)
    
    data_hash = _compute_data_hash(log_returns)
    return _run_drift_pipeline_impl(log_returns, window_size, stride, data_hash)

@st.cache_data
def _run_drift_pipeline_impl(log_returns: pd.DataFrame, window_size: int, stride: int, _data_hash: int) -> pd.Series:
    """
    Implementation of Wasserstein drift computation.
    
    1. Generate distance matrices for all windows
    2. Compute persistence diagrams using Vietoris-Rips
    3. Calculate amplitude (structural magnitude) of each diagram
    4. Measure drift as absolute change in amplitude
    
    Args:
        log_returns: DataFrame of log returns
        window_size: Window size in days
        stride: Step size between windows
        _data_hash: Cache key
    
    Returns:
        Series of drift values
    """
    n_samples = len(log_returns)
    distance_matrices = []
    dates = []
    
    progress_bar = st.progress(0, text="Computing topological drift...")
    
    # Generate distance matrices for all windows
    for i in range(window_size, n_samples):
        window_data = log_returns.iloc[i-window_size:i]
        corr_matrix = window_data.corr().values
        dist_matrix = CORRELATION_DISTANCE_COEFFICIENT * np.sqrt(1 - corr_matrix)
        np.fill_diagonal(dist_matrix, 0)
        distance_matrices.append(dist_matrix)
        dates.append(log_returns.index[i])
        
        if i % max(1, (n_samples - window_size) // 20) == 0:
            progress_bar.progress(min((i - window_size) / (n_samples - window_size), PROGRESS_BAR_MAX), text="Computing topological drift...")
    
    if len(distance_matrices) < 2:
        progress_bar.empty()
        return pd.Series(0, index=dates)
    
    # Compute persistence diagrams
    vr = VietorisRipsPersistence(metric="precomputed", homology_dimensions=[0, 1, 2], n_jobs=-1)
    diagrams = vr.fit_transform(np.array(distance_matrices))
    
    # Calculate amplitude (structural magnitude)
    amp = Amplitude(metric='wasserstein', n_jobs=-1)
    amplitudes = amp.fit_transform(diagrams)
    
    # Sum amplitudes across dimensions
    total_amp = np.sum(amplitudes, axis=1)
    
    # Drift = absolute change in structural magnitude
    drift = np.abs(np.diff(total_amp, prepend=total_amp[0]))
    
    drift_series = pd.Series(drift, index=dates)
    
    progress_bar.empty()
    return drift_series.iloc[::stride]

def get_snapshot_topology(log_returns: pd.DataFrame, target_date: pd.Timestamp, 
                          window_size: int, tickers: list[str]) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame]]:
    """
    Compute correlation matrix and MDS projection for a specific date.
    
    Args:
        log_returns: DataFrame of log returns
        target_date: Date to create snapshot for
        window_size: Lookback window size in days
        tickers: List of ticker symbols
    
    Returns:
        Tuple of (MDS coordinates array, correlation matrix DataFrame), or (None, None) if insufficient data
    """
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
    signal_options = ["TDA", "Shannon"] + (["Wasserstein"] if GTDA_AVAILABLE else [])
    default_index = signal_options.index("Wasserstein") if GTDA_AVAILABLE else 0
    primary_signal = st.radio(
        "Primary Signal",
        signal_options,
        index=default_index,
        help="Select which signal drives alarms, validation, and optimization"
    )
    
    window_size = st.slider("Lookback Window (days)", 30, 120, 70, help="Correlation window. 70 days captures medium-term regimes.")
    smoothing_span = st.slider("Smoothing (EMA span)", 1, 50, 20, help="Exponential smoothing. 20 is moderate smoothing for clarity.")
    
    st.markdown("---")
    st.header("🔬 Takens Embedding")
    takens_delay = st.slider("Takens Delay", 1, 10, TAKENS_DELAY, help="Time lag for embedding. Use diagnostics to find optimal value.")
    takens_dimension = st.slider("Takens Dimension", 2, 6, 4, help="Embedding dimension. Use diagnostics (FNN) to find optimal value.")
    
    st.markdown("---")
    st.header("🚨 Thresholds")
    percentile_threshold = st.slider("Critical %", 1, 20, 10, help="Percentile threshold. 10th percentile marks elevated risk levels.")
    
    st.markdown("---")
    st.header("📅 Date Range")
    min_year = 2000
    max_year = pd.Timestamp.now().year
    
    date_range = st.slider("Years", min_year, max_year, (2018, max_year))
    start_date = pd.to_datetime(f"{date_range[0]}-01-01")
    end_date = pd.to_datetime(f"{date_range[1]}-12-31")

st.title("Market Entropy")

# The 11 SPDR Sector ETFs (Full US Economy Coverage)
tickers = [
    'XLK',  # Technology
    'XLF',  # Financials
    'XLV',  # Healthcare
    'XLE',  # Energy
    'XLC',  # Communication Services
    'XLI',  # Industrials
    'XLP',  # Consumer Staples
    'XLY',  # Consumer Discretionary
    'XLU',  # Utilities
    'XLB',  # Materials
    'XLRE'  # Real Estate
]
mag7_returns_full, sp500_price_full = fetch_market_data(tickers, "^GSPC")

if len(mag7_returns_full) == 0 or len(sp500_price_full) == 0:
    st.error("Failed to fetch market data. Please check your internet connection and try again.")
    st.stop()

if mag7_returns_full.index.tz is not None:
    start_date = pd.Timestamp(start_date).tz_localize(mag7_returns_full.index.tz)
    end_date = pd.Timestamp(end_date).tz_localize(mag7_returns_full.index.tz)

common_index_full = mag7_returns_full.index.intersection(sp500_price_full.index)
mag7_returns_full = mag7_returns_full.loc[common_index_full]
sp500_price_full = sp500_price_full.loc[common_index_full]

corr_entropy_full = run_tda_pipeline(mag7_returns_full, window_size)
takens_entropy_full = run_takens_pipeline(sp500_price_full, window_size, stride=1, dimension=takens_dimension, delay=takens_delay)
shannon_entropy_full = run_shannon_pipeline(sp500_price_full, window_size, stride=1)

# Compute Wasserstein drift if giotto-tda is available
if GTDA_AVAILABLE:
    drift_full = run_drift_pipeline(mag7_returns_full, window_size, stride=1)
else:
    drift_full = pd.Series(dtype=float)

if len(corr_entropy_full) == 0 or len(takens_entropy_full) == 0 or len(shannon_entropy_full) == 0:
    st.error("Insufficient data for TDA computation")
    st.stop()

if GTDA_AVAILABLE and len(drift_full) > 0:
    common_start = max(corr_entropy_full.index[0], takens_entropy_full.index[0], shannon_entropy_full.index[0], drift_full.index[0], start_date)
    common_end = min(corr_entropy_full.index[-1], takens_entropy_full.index[-1], shannon_entropy_full.index[-1], drift_full.index[-1], end_date)
else:
    common_start = max(corr_entropy_full.index[0], takens_entropy_full.index[0], shannon_entropy_full.index[0], start_date)
    common_end = min(corr_entropy_full.index[-1], takens_entropy_full.index[-1], shannon_entropy_full.index[-1], end_date)

mag7_returns = mag7_returns_full.loc[common_start:common_end]
sp500_price = sp500_price_full.loc[common_start:common_end]
corr_entropy = corr_entropy_full.loc[common_start:common_end]
takens_entropy = takens_entropy_full.loc[common_start:common_end]
shannon_entropy = shannon_entropy_full.loc[common_start:common_end]

if GTDA_AVAILABLE and len(drift_full) > 0:
    drift = drift_full.loc[common_start:common_end]
else:
    drift = pd.Series(dtype=float)

if len(mag7_returns) < window_size or len(sp500_price) < window_size:
    st.error(f"Insufficient data. Need at least {window_size} days in selected range")
    st.stop()

if len(corr_entropy) == 0 or len(takens_entropy) == 0 or len(shannon_entropy) == 0:
    st.error("Insufficient data after calculations. Try a larger date range")
    st.stop()

# Processing
if GTDA_AVAILABLE and len(drift) > 0:
    common_index = corr_entropy.index.intersection(takens_entropy.index).intersection(shannon_entropy.index).intersection(drift.index)
    drift = drift.loc[common_index]
    drift_smooth = drift.ewm(span=smoothing_span).mean()
    norm_drift = (drift_smooth - drift_smooth.mean()) / drift_smooth.std()
else:
    common_index = corr_entropy.index.intersection(takens_entropy.index).intersection(shannon_entropy.index)
    
corr_entropy = corr_entropy.loc[common_index]
takens_entropy = takens_entropy.loc[common_index]
shannon_entropy = shannon_entropy.loc[common_index]

norm_corr = (corr_entropy - corr_entropy.mean()) / corr_entropy.std()
norm_takens = (takens_entropy - takens_entropy.mean()) / takens_entropy.std()
norm_shannon = (shannon_entropy - shannon_entropy.mean()) / shannon_entropy.std()

signal_correlation = float(norm_corr.corr(norm_takens))
shannon_takens_corr = float(norm_shannon.corr(norm_takens))

stacked_signals = np.column_stack([norm_corr, norm_takens])
pca = PCA(n_components=1)
pca_weights = np.abs(pca.fit_transform(stacked_signals).flatten())
pca_weights = pca_weights / pca_weights.sum()

pca_weight_corr = float(pca.components_[0, 0] ** 2 / (pca.components_[0, 0] ** 2 + pca.components_[0, 1] ** 2))
pca_weight_takens = 1 - pca_weight_corr

# Smoothed versions for plotting/alerts
tda_signal_raw = pca_weight_corr * norm_corr + pca_weight_takens * norm_takens
tda_signal_smooth = tda_signal_raw.ewm(span=smoothing_span).mean()
shannon_smooth = norm_shannon.ewm(span=smoothing_span).mean()
drift_smooth = drift_smooth if GTDA_AVAILABLE and len(drift) > 0 else pd.Series(dtype=float)

# Effective primary (fallback if drift unavailable)
effective_primary = primary_signal
if primary_signal == "Wasserstein" and (not GTDA_AVAILABLE or len(drift_smooth) == 0):
    effective_primary = "TDA"
    st.sidebar.warning("Wasserstein unavailable for current selection; falling back to TDA.")

# Select which signal drives alarms/validation/optimization
if effective_primary == "TDA":
    active_display_signal = tda_signal_smooth
    active_threshold_signal = tda_signal_smooth
    active_label = "TDA Entropy"
elif effective_primary == "Shannon":
    active_display_signal = shannon_smooth
    active_threshold_signal = shannon_smooth
    active_label = "Shannon Entropy"
elif effective_primary == "Wasserstein" and GTDA_AVAILABLE and len(drift_smooth) > 0:
    active_display_signal = drift_smooth
    active_threshold_signal = -drift_smooth  # invert so low values still mean worse for threshold logic
    active_label = "Wasserstein Drift"
else:
    active_display_signal = tda_signal_smooth
    active_threshold_signal = tda_signal_smooth
    active_label = "TDA Entropy"

signal_sharpe = float(active_display_signal.mean() / active_display_signal.std() * np.sqrt(252)) if active_display_signal.std() > 0 else 0
signal_strength = float(np.abs(active_display_signal.mean()) / (active_display_signal.std() + 1e-8))

historical_threshold_val = active_threshold_signal.quantile(percentile_threshold / 100.0)
threshold_series = pd.Series(historical_threshold_val, index=active_threshold_signal.index)

view_signal = active_display_signal
view_threshold = threshold_series
view_sp500 = sp500_price.loc[view_signal.index]

if len(view_signal) == 0:
    st.error("No data in selected range")
    st.stop()

latest_val = float(active_threshold_signal.iloc[-1])
display_val = float(view_signal.iloc[-1])

rolling_mean = active_threshold_signal.rolling(ROLLING_WINDOW_DAYS).mean().iloc[-1]
rolling_std = active_threshold_signal.rolling(ROLLING_WINDOW_DAYS).std().iloc[-1]
z_score = (latest_val - rolling_mean) / (rolling_std + 1e-8) if rolling_std > 0 else 0

price_trend_5d = view_sp500.pct_change(5).iloc[-1]
price_trend_20d = view_sp500.pct_change(20).iloc[-1]
price_trend_60d = view_sp500.pct_change(60).iloc[-1]

below_threshold = latest_val <= historical_threshold_val
extreme_low = z_score < Z_SCORE_EXTREME_THRESHOLD
very_extreme_low = z_score < Z_SCORE_SEVERE_THRESHOLD

if very_extreme_low and price_trend_20d < PRICE_DRAWDOWN_THRESHOLD:
    status = "🚨 Critical Risk"
    color = "inverse"
    msg = f"Extreme entropy spike + -5% drawdown"
    risk_level = 5
elif very_extreme_low and price_trend_60d < 0:
    status = "🚨 Critical Risk"
    color = "inverse"
    msg = "Extreme entropy spike + declining prices"
    risk_level = 5
elif extreme_low and below_threshold and price_trend_20d < 0:
    status = "⚠️ High Risk"
    color = "inverse"
    msg = "Low entropy + negative momentum"
    risk_level = 4
elif extreme_low and below_threshold:
    status = "⚡ Elevated Risk"
    color = "off"
    msg = "Unusually low entropy"
    risk_level = 3
elif below_threshold and price_trend_60d < PRICE_DRAWDOWN_60D_THRESHOLD:
    status = "⚠️ Caution"
    color = "off"
    msg = "Low entropy + recent decline"
    risk_level = 2
elif below_threshold:
    status = "✓ Monitoring"
    color = "normal"
    msg = "Low entropy + stable prices"
    risk_level = 1
else:
    status = "✅ Healthy"
    color = "normal"
    msg = "Normal market structure"
    risk_level = 0

tab1, tab2, tab3, tab4 = st.tabs(["Signal", "Inspector", "Validation", "Optimization"])

with tab1:
    drift_msg = " **Wasserstein Drift** (cyan) measures the speed of structural change." if GTDA_AVAILABLE else ""
    st.info(f"📊 **Signal Methodology:** This dashboard analyzes market entropy using topological data analysis (TDA). The combined signal uses: (1) **Topological Entropy** from 11 SPDR sector correlation structure (measures systemic connectivity), (2) **Takens Embedding Entropy** from S&P 500 price dynamics (measures attractor complexity). **Shannon Entropy** (purple line) is shown for reference to compare structural topology vs. return distribution chaos.{drift_msg} Entropy and drift are visualization aids; alerts are based on topological entropy.")
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Status", status, msg, delta_color=color)
    c2.metric(f"{active_label} (Z-score)", f"{display_val:.2f}", f"Z-score: {z_score:.2f}")
    c3.metric("Risk Level", f"{risk_level}/5", "Multi-factor Risk Score")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Optional overlays (default to primary only)
    show_tda = st.checkbox("Show TDA", value=effective_primary == "TDA")
    show_shannon = st.checkbox("Show Shannon", value=effective_primary == "Shannon")
    show_drift = st.checkbox("Show Wasserstein", value=(effective_primary == "Wasserstein" and GTDA_AVAILABLE and len(drift) > 0))
    
    view_corr = norm_corr.loc[view_signal.index]
    view_shannon = norm_shannon.loc[view_signal.index]
    if show_tda:
        fig.add_trace(go.Scatter(x=view_corr.index, y=view_corr, name="TDA Entropy", line=dict(color='#00ff00', width=2)), secondary_y=False)
    if show_shannon:
        fig.add_trace(go.Scatter(x=view_shannon.index, y=view_shannon, name="Shannon Entropy", line=dict(color='rgba(189,0,255,0.4)', width=1.5)), secondary_y=False)
    
    if show_drift and GTDA_AVAILABLE and 'drift' in locals() and len(drift) > 0:
        view_drift = norm_drift.loc[view_signal.index]
        fig.add_trace(go.Scatter(x=view_drift.index, y=view_drift, name="Wasserstein Drift", line=dict(color='rgba(0,212,255,0.6)', width=1.5)), secondary_y=False)
    
    # Active threshold
    fig.add_trace(go.Scatter(x=view_threshold.index, y=view_threshold, name="Threshold", line=dict(color='red', dash='dot')), secondary_y=False)
    
    # S&P 500 price
    fig.add_trace(go.Scatter(x=view_sp500.index, y=view_sp500, name="S&P 500", line=dict(color='rgba(255,255,255,0.2)')), secondary_y=True)
    
    fig.update_layout(template="plotly_dark", height=500, hovermode="x unified", legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig, width='stretch')
    
    st.markdown("---")
    st.subheader("📊 Signal Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Z-Score",
            f"{z_score:.2f}",
            help="How many std devs from 1-year mean (< -2.0 is extreme)"
        )
    
    with col2:
        st.metric(
            "Signal Correlation",
            f"{signal_correlation:.3f}",
            help="Correlation between correlation-based and Takens-based entropy"
        )
    
    with col3:
        if GTDA_AVAILABLE and len(drift) > 0:
            drift_percentile = (drift_smooth <= drift_smooth.iloc[-1]).sum() / len(drift_smooth) * 100
            st.metric(
                "Wasserstein Drift",
                f"{drift_smooth.iloc[-1]:.3f}",
                f"{drift_percentile:.0f}th percentile",
                help="Topological speed (rate of structural change). High = shock"
            )
        else:
            st.metric(
                "Shannon Entropy",
                f"{norm_shannon.iloc[-1]:.2f}",
                help="Return distribution entropy (Z-score). High = chaos/fat-tails"
            )
    
    with col4:
        st.metric(
            "Signal Stability (Sharpe-style)",
            f"{signal_sharpe:.3f}",
            help="Risk-adjusted signal strength (not a P&L Sharpe ratio)"
        )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Price Trend (5d)",
            f"{price_trend_5d*100:.2f}%",
            help="5-day price change"
        )
    
    with col2:
        st.metric(
            "Price Trend (20d)",
            f"{price_trend_20d*100:.2f}%",
            help="20-day price change"
        )
    
    with col3:
        if primary_signal == "TDA":
            st.metric(
                "PCA Weights",
                f"TDA:{pca_weight_corr:.0%} / Tak:{pca_weight_takens:.0%}",
                help="PCA weights: TDA/Takens"
            )
        else:
            st.metric("PCA Weights", "N/A", help="Only applicable for TDA primary")
    
    with col4:
        st.metric(
            "Rolling Mean",
            f"{rolling_mean:.2f}",
            help="252-day rolling average entropy"
        )
    
    # Actionable summary based on regime
    if risk_level >= 3:
        summary_color = "🔴"
        summary_text = f"**{summary_color} Elevated Risk Regime:** Current entropy is {abs(z_score):.1f} standard deviations below normal. Historically, similar conditions have preceded increased volatility and drawdowns. See Validation tab for statistical evidence."
    elif risk_level >= 1:
        summary_color = "🟡"
        summary_text = f"**{summary_color} Monitoring:** Entropy shows some stress (Z={z_score:.1f}), but price action remains relatively stable. Watch for deterioration in trend metrics."
    else:
        summary_color = "🟢"
        summary_text = f"**{summary_color} Normal Market Structure:** Entropy levels within historical norms (Z={z_score:.1f}). No immediate structural concerns detected."
    
    st.markdown("---")
    st.markdown(summary_text)
    st.markdown("---")
    
    with st.expander("ℹ️ Risk Classification Details", expanded=False):
        st.markdown(f"""
        ### Statistical Risk Framework
        
        **Z-Score Analysis:**
        - Current Z-Score: {z_score:.2f}
        - Interpretation: {z_score:.2f} standard deviations from 252-day mean
        
        **Z-Score Thresholds:**
        - Greater than -1.0: Normal market conditions
        - -1.0 to -2.0: Below average entropy (monitoring)
        - -2.0 to -3.0: Extreme low entropy (elevated risk)
        - Less than -3.0: Severe low entropy (critical risk)
        
        **Price Momentum:**
        - 5-day change: {price_trend_5d*100:.2f}% (immediate)
        - 20-day change: {price_trend_20d*100:.2f}% (short-term)
        - 60-day change: {price_trend_60d*100:.2f}% (medium-term)
        
        **Risk Scoring:**
        - Level 0: Normal market structure (no alert)
        - Level 1: Low entropy with stable prices (monitoring)
        - Level 2: Low entropy + recent decline (caution)
        - Level 3: Extreme entropy shift (elevated risk)
        - Level 4: Extreme entropy + negative momentum (high risk)
        - Level 5: Severe entropy + major drawdown (critical risk)
        
        **How It Works:**
        1. We compare current entropy to its 252-day rolling statistics
        2. We measure statistical significance (Z-score)
        3. We combine with recent price action (5d, 20d, 60d trends)
        4. Multi-factor classification prevents false alarms from single metrics
        """)

with tab2:
    snap_date = st.select_slider("Date", options=view_signal.index, value=view_signal.index[-1], format_func=lambda x: x.strftime('%Y-%m-%d'))
    
    corr_threshold = st.slider("Correlation Edge Threshold", 0.0, 1.0, CORRELATION_THRESHOLD_DEFAULT, 0.05, help="Show edges for correlations above this value")
    
    coords, corr = get_snapshot_topology(mag7_returns, snap_date, window_size, tickers)
    
    if coords is not None:
        G = nx.Graph()
        for i in range(len(tickers)):
            G.add_node(i, ticker=tickers[i])
        for i in range(len(tickers)):
            for j in range(i+1, len(tickers)):
                if corr.iloc[i,j] > corr_threshold:
                    G.add_edge(i, j, weight=corr.iloc[i,j])
        
        clustering = nx.clustering(G)
        try:
            centrality = nx.betweenness_centrality(G)
        except:
            centrality = {i: 0 for i in range(len(tickers))}
        
        row1_cols = st.columns([2, 1])
        
        with row1_cols[0]:
            st.markdown("**Correlation Network Map**")
            mds_fig = go.Figure()
            
            for i in range(len(tickers)):
                for j in range(i+1, len(tickers)):
                    if corr.iloc[i,j] > corr_threshold:
                        corr_val = corr.iloc[i,j]
                        color_intensity = (corr_val - corr_threshold) / (1 - corr_threshold)
                        mds_fig.add_trace(go.Scatter(
                            x=[coords[i,0], coords[j,0]], 
                            y=[coords[i,1], coords[j,1]], 
                            mode='lines', 
                            line=dict(
                                color=f'rgba(255, {int(255*(1-color_intensity))}, 0, {0.3 + 0.7*color_intensity})', 
                                width=1 + 3*color_intensity
                            ), 
                            showlegend=False,
                            hoverinfo='skip'
                        ))
            
            node_sizes = [15 + 30 * centrality.get(i, 0) for i in range(len(tickers))]
            node_colors = [clustering.get(i, 0) for i in range(len(tickers))]
            
            mds_fig.add_trace(go.Scatter(
                x=coords[:,0], 
                y=coords[:,1], 
                mode='markers+text', 
                text=tickers,
                textposition='top center', 
                textfont=dict(size=12),
                marker=dict(
                    size=node_sizes, 
                    color=node_colors,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Clustering", thickness=10, len=0.5),
                    line=dict(color='white', width=1)
                ),
                hovertemplate='<b>%{text}</b><br>Clustering: %{marker.color:.2f}<extra></extra>'
            ))
            
            mds_fig.update_layout(
                template="plotly_dark", 
                showlegend=False, 
                xaxis=dict(visible=True, title="MDS 1"),
                yaxis=dict(visible=True, title="MDS 2"),
                height=400
            )
            st.plotly_chart(mds_fig, use_container_width=True)
        
        with row1_cols[1]:
            st.markdown("**Network Metrics**")
            avg_clustering = np.mean(list(clustering.values()))
            avg_centrality = np.mean(list(centrality.values()))
            density = nx.density(G)
            
            st.metric("Avg Clustering", f"{avg_clustering:.3f}", help="How tightly grouped stocks are")
            st.metric("Avg Centrality", f"{avg_centrality:.3f}", help="Average betweenness centrality")
            st.metric("Network Density", f"{density:.3f}", help="Ratio of actual to possible connections")
            
            # Warn about sparse networks
            if density < 0.2:
                st.warning(f"⚠️ **Network Sparsity Alert:** Current correlation threshold ({corr_threshold:.2f}) yields {density:.1%} network density. Centrality metrics may exhibit reduced reliability. Consider lowering the threshold for enhanced network analysis.")
            
            st.markdown("**Top Central Stocks**")
            sorted_centrality = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            for idx, (node, cent) in enumerate(sorted_centrality[:3]):
                st.caption(f"{idx+1}. {tickers[node]}: {cent:.3f}")
        
        row2_cols = st.columns(2)
        
        with row2_cols[0]:
            st.markdown("**Correlation Heatmap**")
            heatmap_fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=tickers,
                y=tickers,
                colorscale='RdYlGn',
                zmid=0,
                text=np.round(corr.values, 2),
                texttemplate='%{text}',
                textfont=dict(size=10),
                colorbar=dict(title="Correlation")
            ))
            heatmap_fig.update_layout(
                template="plotly_dark",
                height=360,
                xaxis=dict(side='bottom'),
                yaxis=dict(autorange='reversed'),
                margin=dict(t=30, b=20, l=10, r=10)
            )
            st.plotly_chart(heatmap_fig, use_container_width=True)
        
        with row2_cols[1]:
            st.markdown("**Takens Attractor**")
            idx = sp500_price_full.index.get_indexer([snap_date], method='nearest')[0]
            if idx > 60:
                win = sp500_price_full.iloc[idx-60:idx].values
                colors = np.linspace(0, 1, len(win)-2)
                takens_fig = go.Figure(data=[go.Scatter3d(
                    x=win[2:], y=win[1:-1], z=win[:-2], 
                    mode='lines',
                    line=dict(color=colors, colorscale='Viridis', width=4, showscale=True,
                             colorbar=dict(title="Time", thickness=10, len=0.5))
                )])
                takens_fig.update_layout(
                    template="plotly_dark", 
                    scene=dict(xaxis_title="t", yaxis_title="t-1", zaxis_title="t-2"),
                    height=360,
                    margin=dict(t=30, b=20, l=0, r=0)
                )
                st.plotly_chart(takens_fig, use_container_width=True)
        
        with st.expander("🔬 Takens Embedding Diagnostics"):
            st.markdown("**Determining Optimal Embedding Parameters**")
            
            try:
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                progress_text.text("Computing autocorrelation...")
                snap_idx = sp500_price_full.index.get_indexer([snap_date], method='nearest')[0]
                window_end = min(snap_idx + 1, len(sp500_price_full))
                window_start = max(0, window_end - 252)  # Reduced from 500 to 252 for performance
                
                diagnostic_prices = sp500_price_full.iloc[window_start:window_end]
                
                progress_bar.progress(0.2)
                progress_text.text("Finding optimal delay...")
                optimal_delay = find_optimal_delay(diagnostic_prices, method='first_zero')
                
                progress_bar.progress(0.4)
                progress_text.text("Computing false nearest neighbors, please wait...")
                fnn_percentages = false_nearest_neighbors(diagnostic_prices, max_dimension=10, delay=optimal_delay)
                
                progress_bar.progress(1.0)
                progress_text.text("✅ Diagnostics complete!")
                progress_bar.empty()
                progress_text.empty()
                
                if fnn_percentages:
                    optimal_dimension = find_optimal_dimension(fnn_percentages, threshold=FNN_DIMENSION_THRESHOLD)
                    autocorr = compute_autocorrelation(np.log(diagnostic_prices / diagnostic_prices.shift(1)).dropna().values)
                    
                    col_diag1, col_diag2 = st.columns(2)
                    
                    with col_diag1:
                        st.metric("Optimal Delay (days)", f"{optimal_delay}", help="Time lag for Takens embedding")
                        st.metric("Optimal Dimension", f"{optimal_dimension}", help="Embedding dimension from FNN")
                    
                    with col_diag2:
                        st.metric("Current Delay (app)", f"{takens_delay}", help="Active delay setting (adjust in sidebar)")
                        st.metric("Current Dimension (app)", f"{takens_dimension}", help="Active dimension setting (adjust in sidebar)")
                    
                    diag_col1, diag_col2 = st.columns(2)
                    
                    with diag_col1:
                        autocorr_fig = go.Figure()
                        autocorr_fig.add_trace(go.Scatter(
                            y=autocorr[:30],
                            mode='lines+markers',
                            line=dict(color='cyan'),
                            name='Autocorrelation'
                        ))
                        autocorr_fig.add_hline(y=0, line_dash='dash', line_color='white', opacity=0.3)
                        autocorr_fig.update_layout(
                            template='plotly_dark',
                            title='Autocorrelation of Returns',
                            xaxis_title='Lag (days)',
                            yaxis_title='Autocorrelation',
                            height=350
                        )
                        st.plotly_chart(autocorr_fig, use_container_width=True)
                    
                    with diag_col2:
                        fnn_fig = go.Figure()
                        fnn_fig.add_trace(go.Scatter(
                            y=fnn_percentages,
                            mode='lines+markers',
                            line=dict(color='#ff00ff'),
                            marker=dict(size=8),
                            name='FNN %'
                        ))
                        fnn_fig.add_hline(y=FNN_DIMENSION_THRESHOLD, line_dash='dash', line_color='red', annotation_text='Threshold (1%)')
                        fnn_fig.update_layout(
                            template='plotly_dark',
                            title='False Nearest Neighbors Analysis',
                            xaxis_title='Embedding Dimension',
                            yaxis_title='FNN %',
                            height=350
                        )
                        st.plotly_chart(fnn_fig, use_container_width=True)
                    
                    st.markdown(f"""
                    **Interpretation:**
                    
                    - **Optimal Delay**: {optimal_delay} days
                      - First point where autocorrelation crosses zero
                      - At this lag, returns are relatively independent
                      - Current setting: {takens_delay} day(s)
                    
                    - **Optimal Dimension**: {optimal_dimension}
                      - False Nearest Neighbors fall below 1% at dimension {optimal_dimension}
                      - This is the minimum dimension to unfold the attractor
                      - Current setting: {takens_dimension}
                    
                    **Recommendation:** {"✅ Current settings match diagnostic recommendations" if takens_delay == optimal_delay and takens_dimension == optimal_dimension else f"Consider adjusting sidebar to delay={optimal_delay}, dimension={optimal_dimension} for optimal reconstruction"}
                    """)
                else:
                    st.warning("Insufficient data for diagnostics computation")
            
            except Exception as e:
                st.warning(f"Diagnostic computation unavailable: {str(e)}")


with tab3:
    st.header("🔬 Signal Validation")
    st.caption(f"Validating primary signal: {active_label}")
    
    st.warning("⚠️ **Disclaimer:** All backtested results are hypothetical and do not reflect actual trading. Analysis excludes transaction costs, slippage, and market impact. Past performance does not guarantee future results.")
    
    st.markdown("""
    This validation suite tests whether low entropy signals actually predict market stress.
    We evaluate the signal using multiple statistical tests and backtesting metrics.
    """)
    
    with st.spinner("Running validation tests..."):
            validator = EntropyValidator(
                signal=active_threshold_signal,
                prices=sp500_price_full.loc[active_threshold_signal.index],
                threshold=historical_threshold_val
            )
            
            alerts = validator.generate_alerts(lookback=5)
            forward_returns = validator.compute_forward_returns([5, 10, 20, 60])
            
            perf_metrics = validator.alert_performance(forward_returns, alerts)
            dd_metrics = validator.drawdown_prediction(drawdown_threshold=-0.10)
            mc_results = validator.monte_carlo_test(n_simulations=1000, horizons=[20])
            regime_stats = validator.regime_analysis()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        precision = dd_metrics['precision']
        st.metric(
            "Precision", 
            f"{precision*100:.1f}%",
            help="Of all alerts, what % preceded a 10%+ drawdown?"
        )
    
    with col2:
        recall = dd_metrics['recall']
        st.metric(
            "Recall", 
            f"{recall*100:.1f}%",
            help="Of all 10%+ drawdowns, what % did we predict?"
        )
    
    with col3:
        twenty_day_diff = perf_metrics[20]['mean_diff']
        st.metric(
            "20d Return Impact", 
            f"{twenty_day_diff*100:.2f}%",
            help="Mean return difference: Alert vs No Alert"
        )
    
    with col4:
        mc_pval = mc_results[20]['p_value']
        mc_sig = "Yes" if mc_pval < P_VALUE_SIGNIFICANCE else "No"
        st.metric(
            "Better than Random?", 
            mc_sig,
            help=f"Monte Carlo p-value: {mc_pval:.3f}"
        )
    
    st.markdown("---")
    
    fig = create_validation_plots(validator, alerts, forward_returns)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        with st.expander("📊 Forward Return Statistics", expanded=False):
            st.markdown("**How returns differ when alerts are active vs. inactive**")
            
            comparison_data = []
            for horizon in [5, 10, 20, 60]:
                if horizon in perf_metrics:
                    m = perf_metrics[horizon]
                    comparison_data.append({
                        'Horizon': f'{horizon}d',
                        'Alert Mean': f"{m['alert_mean']*100:.2f}%",
                        'No Alert Mean': f"{m['no_alert_mean']*100:.2f}%",
                        'Difference': f"{m['mean_diff']*100:.2f}%",
                        'P-value': f"{m['p_value']:.4f}",
                        'Significant': '✓' if m['p_value'] < P_VALUE_SIGNIFICANCE else '✗'
                    })
            
            st.dataframe(comparison_data, use_container_width=True)
            
            st.markdown("""
            **Interpretation:**
            - Negative "Difference" means alerts predict lower returns (bearish signal)
            - P-value < 0.05 means statistically significant
            - Look for consistent patterns across horizons
            """)
    
    with col_right:
        with st.expander("🎯 Drawdown Prediction Quality", expanded=False):
            st.markdown(f"**Predicting 10%+ drawdowns within 60 days**")
            
            st.markdown(f"""
            - **Precision: {dd_metrics['precision']*100:.1f}%**  
              When we issue an alert, there's a {dd_metrics['precision']*100:.1f}% chance a significant drawdown follows
            
            - **Recall: {dd_metrics['recall']*100:.1f}%**  
              We successfully predict {dd_metrics['recall']*100:.1f}% of all major drawdowns
            
            - **F1 Score: {dd_metrics['f1_score']:.3f}**  
              Harmonic mean of precision and recall (0-1 scale, higher is better)
            
            - **Accuracy: {dd_metrics['accuracy']*100:.1f}%**  
              Overall correct predictions
            """)
            
            st.markdown("**Confusion Matrix:**")
            confusion_df = pd.DataFrame({
                '': ['Alert Issued', 'No Alert'],
                'Drawdown Occurred': [
                    dd_metrics['true_positives'], 
                    dd_metrics['false_negatives']
                ],
                'No Drawdown': [
                    dd_metrics['false_positives'], 
                    dd_metrics['true_negatives']
                ]
            })
            st.dataframe(confusion_df, use_container_width=True)
    
    with st.expander("🎲 Monte Carlo Randomness Test", expanded=False):
        st.markdown("""
        **Testing if our signal is better than random guessing**
        
        We randomly shuffle the alert timings 1,000 times and check if the actual signal
        performs better than these random shuffles. This tests whether the temporal
        relationship between entropy and future returns is real or just luck.
        """)
        
        for horizon, results in mc_results.items():
            st.markdown(f"**{horizon}-Day Forward Returns:**")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Actual Signal", f"{results['actual_diff']*100:.2f}%")
                st.metric("Random Average", f"{results['null_mean']*100:.2f}%")
            
            with col2:
                st.metric("P-value", f"{results['p_value']:.4f}")
                sig_text = "✓ SIGNIFICANT" if results['significant'] else "✗ Not Significant"
                sig_color = "green" if results['significant'] else "red"
                st.markdown(f":{sig_color}[**{sig_text}**]")
            
            st.markdown(f"""
            The actual signal produces a {results['actual_diff']*100:.2f}% return difference,
            while random shuffling produces {results['null_mean']*100:.2f}% on average.
            
            **P-value = {results['p_value']:.4f}** means that only {results['p_value']*100:.1f}% of random
            shuffles performed as well as the actual signal.
            """)
    
    with st.expander("📈 Entropy Regime Analysis", expanded=False):
        st.markdown("""
        **How returns vary across different entropy levels**
        
        We split the data into 5 regimes based on entropy percentiles and analyze
        forward returns for each regime.
        """)
        
        regime_rows = []
        for regime_name, stats in regime_stats.items():
            regime_rows.append({
                'Regime': regime_name,
                'Mean Return': f"{stats['mean']*100:.2f}%",
                'Sharpe': f"{stats['sharpe']:.2f}",
                'Negative %': f"{stats['negative_pct']*100:.1f}%",
                'Count': stats['count']
            })
        
        regime_df = pd.DataFrame(regime_rows)
        st.dataframe(regime_df, use_container_width=True)
        
        st.markdown("""
        **What to look for:**
        - Does "Very Low" entropy consistently show worse returns?
        - Is the relationship monotonic (returns improve as entropy increases)?
        - Are there enough observations in each regime for statistical validity?
        """)
    
    st.markdown("---")
    
    st.subheader("🎓 Validation Summary")
    
    score = 0
    max_score = 4
    feedback = []
    
    if perf_metrics[20]['p_value'] < P_VALUE_SIGNIFICANCE:
        score += 1
        feedback.append("✅ **Statistically significant return prediction** (20-day horizon)")
    else:
        feedback.append("⚠️ **Weak return prediction** - 20-day returns not significantly different")
    
    if dd_metrics['precision'] > 0.3 and dd_metrics['recall'] > 0.3:
        score += 1
        feedback.append("✅ **Decent drawdown prediction** - Balanced precision and recall")
    elif dd_metrics['precision'] > 0.4:
        score += 0.5
        feedback.append("⚠️ **High precision but low recall** - Few false alarms but misses many events")
    elif dd_metrics['recall'] > 0.4:
        score += 0.5
        feedback.append("⚠️ **High recall but low precision** - Catches most events but many false alarms")
    else:
        feedback.append("❌ **Poor drawdown prediction** - Both precision and recall are low")
    
    if mc_results[20]['significant']:
        score += 1
        feedback.append("✅ **Beats random chance** - Signal is statistically significant")
    else:
        feedback.append("❌ **Not better than random** - Could be due to luck or overfitting")
    
    regime_means = [stats['mean'] for stats in regime_stats.values()]
    if regime_means[0] < regime_means[-1]:
        score += 1
        feedback.append("✅ **Clear entropy-return relationship** - Returns improve with entropy")
    else:
        feedback.append("⚠️ **Unclear relationship** - Entropy levels don't show monotonic pattern")
    
    score_pct = (score / max_score) * 100
    
    if score_pct >= 75:
        assessment = "**Strong Signal** 🎯"
        color = "green"
        interpretation = "The entropy signal shows robust predictive power across multiple tests."
    elif score_pct >= 50:
        assessment = "**Moderate Signal** ⚡"
        color = "orange"
        interpretation = "The signal shows some predictive ability but has limitations."
    else:
        assessment = "**Weak Signal** ⚠️"
        color = "red"
        interpretation = "The signal's predictive power is questionable. Consider refinement."
    
    st.markdown(f"## :{color}[{assessment}]")
    st.markdown(f"**Overall Score: {score:.1f}/{max_score}** ({score_pct:.0f}%)")
    st.markdown(interpretation)
    
    st.markdown("---")
    
    for item in feedback:
        st.markdown(item)
    
    st.markdown("---")
    st.info("""
    **Analysis Guidelines:**
    
    1. **Score < 50%:** Consider adjusting window size, threshold, or smoothing parameters for improved signal quality
    2. **High Precision, Low Recall:** Threshold may be overly conservative—consider relaxation for broader coverage
    3. **Low Precision, High Recall:** Threshold may be insufficiently selective—consider tightening criteria
    4. **Underperformance vs. Random:** Potential overfitting to in-sample data—validate on out-of-sample periods
    
    **Note:** Historical validation metrics do not guarantee prospective performance. This analysis should complement, not replace, comprehensive risk assessment and portfolio construction frameworks.
    """)

with tab4:
    st.header("⚙️ Parameter Optimization")
    
    st.warning("⚠️ **Disclaimer:** Optimization results represent hypothetical backtests excluding transaction costs, slippage, and market impact. Optimized parameters may exhibit degraded out-of-sample performance due to overfitting.")
    st.header("🔧 Parameter Optimization")
    
    st.markdown("""
    Find optimal parameters by searching across multiple combinations.
    Choose between **grid search** (test all combinations) or **walk-forward** 
    (realistic out-of-sample testing).
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Optimization Method")
        opt_method = st.radio(
            "Method",
            ["Grid Search", "Walk-Forward"],
            help="""
            **Grid Search**: Test all parameter combinations on full dataset (faster, risk of overfitting)
            **Walk-Forward**: Train on past data, test on future (slower, more realistic)
            """
        )
        
        trading_default = "Short (Risk-Off)" if effective_primary == "Wasserstein" else "Long (Contrarian)"
        trading_direction = st.radio(
            "Trading Direction",
            ["Long (Contrarian)", "Short (Risk-Off)"],
            index=["Long (Contrarian)", "Short (Risk-Off)"].index(trading_default),
            help="""
            **Long**: Buy when signal is low (contrarian)
            **Short**: Sell/short when signal indicates stress (default for Wasserstein)
            """
        )
        
        optimize_for = st.selectbox(
            "Optimize For",
            ["Sharpe Ratio", "Total Return", "Max Drawdown (minimize)", "Win Rate"],
            help="Which metric to maximize/minimize"
        )
    
    with col2:
        st.subheader("Parameter Ranges")
        
        col2a, col2b = st.columns(2)
        
        with col2a:
            if primary_signal == "Wasserstein":
                window_default = (60, 110)
                smooth_default = (10, 25)
            else:
                window_default = (40, 100)
                smooth_default = (5, 20)

            window_min, window_max = st.slider(
                "Window Size Range", 20, 140, window_default, 10,
                help="Lookback period"
            )
            
            smooth_min, smooth_max = st.slider(
                "Smoothing Range", 1, 40, smooth_default, 5,
                help="EMA smoothing span"
            )
        
        with col2b:
            use_pca = st.checkbox("Use PCA Weighting", value=True, help="Use PCA to optimize signal combination weights")
            
            if use_pca:
                weight_options = ['pca']
            else:
                weight_options = st.multiselect(
                    "Correlation Weights",
                    [0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0],
                    default=[0.3, 0.5, 0.7],
                    help="Weight of correlation entropy (vs Takens)"
                )
            
            threshold_options = st.multiselect(
                "Threshold Percentiles",
                [1, 3, 5, 10, 15, 20, 25],
                default=[5, 10, 15],
                help="Alert threshold percentile"
            )
    
    param_grid = {
        'window_size': list(range(window_min, window_max + 1, 10)),
        'smoothing_span': list(range(smooth_min, smooth_max + 1, 5)),
        'corr_weight': weight_options if weight_options else [0.5],
        'threshold_pct': threshold_options if threshold_options else [5]
    }
    
    num_combinations = len(param_grid['window_size']) * \
                       len(param_grid['smoothing_span']) * \
                       len(param_grid['corr_weight']) * \
                       len(param_grid['threshold_pct'])
    
    st.info(f"**Total combinations to test: {num_combinations}**")
    
    if num_combinations > 500:
        st.error(f"⚠️ **Parameter Space Exceeds Limit:** {num_combinations} combinations exceeds the 500 configuration maximum. Please reduce parameter ranges or use Walk-Forward optimization with coarser grid spacing.")
        st.stop()
    elif num_combinations > 200:
        st.warning(f"⚠️ **Computation Time:** Testing {num_combinations} configurations may require approximately {num_combinations//10} minutes.")
    
    if opt_method == "Walk-Forward":
        st.markdown("---")
        wf_col1, wf_col2 = st.columns(2)
        
        with wf_col1:
            train_days = st.slider("Training Days", 252, 2520, 1260, 252, help="252 days ≈ 1 year")
        with wf_col2:
            test_days = st.slider("Testing Days", 63, 504, 252, 63, help="252 days ≈ 1 year")
        
        train_years = train_days / 252
        test_years = test_days / 252
        total_folds = max(1, (len(sp500_price_full) - train_days) // test_days)
        st.info(f"Will train on {train_years:.1f} years ({train_days} days), test on {test_years:.1f} years ({test_days} days). Estimated ~{total_folds} folds.")
    
    st.markdown("---")
    
    if st.button("▶️ Run Optimization", type="primary"):
        if effective_primary == "Shannon":
            st.error("Optimization for Shannon entropy is not supported. Please choose TDA or Wasserstein.")
            st.stop()
        
        with st.spinner("Running optimization... This may take several minutes."):
            
            if effective_primary == "Wasserstein" and GTDA_AVAILABLE:
                def inverted_drift_pipeline(log_returns, window_size):
                    d = run_drift_pipeline(log_returns, window_size, stride=1)
                    return -d  # invert so threshold logic (low=alert) still applies
                optimizer = EntropyOptimizer(
                    mag7_returns=mag7_returns_full,
                    sp500_price=sp500_price_full,
                    run_tda_pipeline=inverted_drift_pipeline,
                    run_takens_pipeline=lambda *args, **kwargs: pd.Series(dtype=float)
                )
                # Force drift-only weighting
                param_grid['corr_weight'] = [1.0]
            elif primary_signal == "Shannon":
                st.error("Optimization for Shannon entropy not supported in this release.")
                st.stop()
            else:
                optimizer = EntropyOptimizer(
                    mag7_returns=mag7_returns_full,
                    sp500_price=sp500_price_full,
                    run_tda_pipeline=run_tda_pipeline,
                    run_takens_pipeline=run_takens_pipeline
                )
            
            direction = 'long' if trading_direction == "Long (Contrarian)" else 'short'
            
            if opt_method == "Grid Search":
                results_df = optimizer.grid_search(
                    param_grid=param_grid,
                    direction=direction,
                    max_combinations=min(500, num_combinations)
                )
            else:
                results_df = optimizer.walk_forward_optimization(
                    param_grid=param_grid,
                    train_days=train_days,
                    test_days=test_days,
                    direction=direction,
                    min_date=start_date
                )
            
            if len(results_df) == 0:
                st.error("Optimization produced no valid results. Consider adjusting parameter ranges.")
                st.stop()
            
            st.session_state['opt_results'] = results_df
            st.session_state['opt_method'] = opt_method
            st.session_state['opt_direction'] = direction
            st.session_state['optimizer'] = optimizer
        
        st.success(f"✅ Optimization Complete: {len(results_df)} configurations evaluated successfully.")
        st.rerun()
    
    if 'opt_results' in st.session_state:
        results_df = st.session_state['opt_results']
        opt_method = st.session_state['opt_method']
        
        st.markdown("---")
        st.subheader("📊 Results")
        
        if opt_method == "Grid Search":
            if optimize_for == "Sharpe Ratio":
                results_df = results_df.sort_values('sharpe', ascending=False)
            elif optimize_for == "Total Return":
                results_df = results_df.sort_values('total_return', ascending=False)
            elif optimize_for == "Max Drawdown (minimize)":
                results_df = results_df.sort_values('max_drawdown', ascending=True)
            elif optimize_for == "Win Rate":
                results_df = results_df.sort_values('win_rate', ascending=False)
            
            st.markdown("### Top 10 Configurations")
            
            display_cols = [
                'window_size', 'smoothing_span', 'corr_weight', 'threshold_pct',
                'sharpe', 'annual_return', 'max_drawdown', 'win_rate', 
                'pct_invested', 'num_trades'
            ]

            available_cols = [c for c in display_cols if c in results_df.columns]
            top_10 = results_df.head(10)[available_cols].copy()

            if 'sharpe' in top_10.columns:
                top_10['sharpe'] = top_10['sharpe'].apply(lambda x: f"{x:.3f}")
            if 'annual_return' in top_10.columns:
                top_10['annual_return'] = top_10['annual_return'].apply(lambda x: f"{x*100:.2f}%")
            if 'max_drawdown' in top_10.columns:
                top_10['max_drawdown'] = top_10['max_drawdown'].apply(lambda x: f"{x*100:.2f}%")
            if 'win_rate' in top_10.columns:
                top_10['win_rate'] = top_10['win_rate'].apply(lambda x: f"{x*100:.1f}%")
            if 'pct_invested' in top_10.columns:
                top_10['pct_invested'] = top_10['pct_invested'].apply(lambda x: f"{x*100:.1f}%")
            if 'corr_weight' in top_10.columns:
                top_10['corr_weight'] = top_10['corr_weight'].apply(lambda x: x if x == 'pca' else f"{x:.1%}")
            
            st.dataframe(top_10, use_container_width=True)
            
            best = results_df.iloc[0]
            
            st.markdown("### 🏆 Best Configuration")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Window", f"{int(best['window_size'])}d")
            col2.metric("Smoothing", f"{int(best['smoothing_span'])}d")
            col3.metric("Corr Weight", best['corr_weight'] if best['corr_weight'] == 'pca' else f"{best['corr_weight']:.1%}")
            col4.metric("Threshold", f"{best['threshold_pct']:.0f}%ile")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Sharpe", f"{best['sharpe']:.3f}", 
                       delta=f"{(best['sharpe'] - best['bh_sharpe']):.3f}" if 'bh_sharpe' in best else None)
            col2.metric("Annual Return", f"{best['annual_return']*100:.2f}%",
                       delta=f"{(best['annual_return'] - best['bh_annual_return'])*100:.2f}%" if 'bh_annual_return' in best else None)
            col3.metric("Max DD", f"{best['max_drawdown']*100:.2f}%" if 'max_drawdown' in best else "N/A")
            col4.metric("Win Rate", f"{best['win_rate']*100:.1f}%" if 'win_rate' in best else "N/A")
            
            if 'optimizer' in st.session_state:
                optimizer = st.session_state['optimizer']
                fig = optimizer.visualize_results(results_df, top_n=20)
                st.plotly_chart(fig, use_container_width=True)
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                "📥 Download Full Results (CSV)",
                csv,
                "optimization_results.csv",
                "text/csv"
            )
            
        else:
            st.markdown("### Walk-Forward Test Results")
            
            st.markdown(f"""
            **Average Test Performance:**
            - Mean Sharpe: {results_df['test_sharpe'].mean():.3f}
            - Mean Annual Return: {results_df['test_annual_return'].mean()*100:.2f}%
            - Mean Max Drawdown: {results_df['test_max_drawdown'].mean()*100:.2f}%
            """)
            
            display_df = results_df[[
                'fold', 'test_start', 'test_end', 
                'window_size', 'smoothing_span', 'threshold_pct',
                'train_sharpe', 'test_sharpe', 'test_annual_return', 'test_max_drawdown'
            ]].copy()
            
            display_df['test_start'] = pd.to_datetime(display_df['test_start']).dt.date
            display_df['test_end'] = pd.to_datetime(display_df['test_end']).dt.date
            display_df['train_sharpe'] = display_df['train_sharpe'].apply(lambda x: f"{x:.3f}")
            display_df['test_sharpe'] = display_df['test_sharpe'].apply(lambda x: f"{x:.3f}")
            display_df['test_annual_return'] = display_df['test_annual_return'].apply(lambda x: f"{x*100:.2f}%")
            display_df['test_max_drawdown'] = display_df['test_max_drawdown'].apply(lambda x: f"{x*100:.2f}%")
            
            st.dataframe(display_df, use_container_width=True)
            
            st.markdown("### 📈 Consistency Analysis")
            
            positive_folds = (results_df['test_sharpe'] > 0).sum()
            total_folds = len(results_df)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Positive Folds", f"{positive_folds}/{total_folds}",
                       f"{positive_folds/total_folds*100:.0f}%")
            col2.metric("Sharpe Std Dev", f"{results_df['test_sharpe'].std():.3f}")
            col3.metric("Stability Score", 
                       f"{1 - (results_df['test_sharpe'].std() / (abs(results_df['test_sharpe'].mean()) + 0.1)):.2f}")
            
            st.markdown("""
            **Interpretation:**
            - **Positive Folds > 70%**: Strategy is consistently profitable
            - **Low Std Dev (<0.5)**: Performance is stable across periods
            - **Stability Score > 0.5**: Good robustness
            """)
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                "📥 Download Walk-Forward Results (CSV)",
                csv,
                "walk_forward_results.csv",
                "text/csv"
            )
    
    with st.expander("❓ How to Use Optimization"):
        st.markdown("""
        ### Grid Search vs Walk-Forward
        
        **Grid Search** (Faster, 2-5 minutes):
        - Tests all combinations on the full dataset
        - Good for initial exploration
        - ⚠️ Risk of overfitting - parameters may work on past but not future
        
        **Walk-Forward** (Slower, 10-30 minutes):
        - Trains on historical data, tests on future data
        - Rolls forward through time (realistic simulation)
        - ✅ More reliable - mimics real trading conditions
        
        ### Interpreting Results
        
        **Good signs:**
        - Sharpe > 1.0 (excellent risk-adjusted returns)
        - Max Drawdown < -30% (limited downside)
        - Win Rate > 55% (more winners than losers)
        - Consistent performance across periods (walk-forward)
        
        **Red flags:**
        - Only works with specific parameters (overfitting)
        - Walk-forward test performance << training performance
        - Too many trades (>200/year) = high transaction costs
        - Very high Sharpe (>3) = probably overfitting
        
        ### Next Steps
        
        1. Run grid search first to find promising ranges
        2. Narrow down parameters and run walk-forward
        3. If walk-forward performance holds up → signal is robust
        4. Use best parameters in your live dashboard
        5. Re-optimize quarterly to adapt to changing markets
        """)
