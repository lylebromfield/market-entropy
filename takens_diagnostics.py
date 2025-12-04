import numpy as np
from scipy import signal
import pandas as pd
from typing import Optional

def compute_autocorrelation(series: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """
    Compute autocorrelation function for a time series.
    
    Uses the standard autocorrelation estimator: autocov(k) / autocov(0),
    where autocov is computed via zero-padded convolution for efficiency.
    
    Args:
        series: 1D array of values (typically returns)
        max_lag: Maximum lag to compute (default 50)
        
    Returns:
        Autocorrelation values from lag 0 to max_lag-1
    """
    series_centered = series - series.mean()
    autocov = np.correlate(series_centered, series_centered, mode='full')
    autocov = autocov[len(autocov)//2:]
    autocorr = autocov / autocov[0]
    return autocorr[:max_lag]

def find_optimal_delay(price_series: pd.Series, method: str = 'first_zero', max_lag: int = 50) -> int:
    """
    Find optimal time delay for Takens embedding using autocorrelation.
    
    Two methods available:
    - 'first_zero': First lag where autocorrelation crosses zero (captures decorrelation point)
    - 'first_minimum': First lag where autocorrelation reaches local minimum
    
    These methods identify when returns lose dependence on past values,
    suitable for reconstructing the attractor in embedding space.
    
    Args:
        price_series: Price series (typically closing prices)
        method: 'first_zero' (default) or 'first_minimum'
        max_lag: Maximum lag to search (default 50)
        
    Returns:
        Optimal delay in days (minimum 1)
    """
    returns = np.log(price_series / price_series.shift(1)).dropna().values
    autocorr = compute_autocorrelation(returns, max_lag)
    
    if method == 'first_zero':
        zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
        if len(zero_crossings) > 0:
            return int(zero_crossings[0]) + 1
        else:
            return max(1, np.argmin(np.abs(autocorr)))
    elif method == 'first_minimum':
        return int(np.argmin(autocorr[1:])) + 1
    else:
        return 1

def false_nearest_neighbors(price_series: pd.Series, max_dimension: int = 10, delay: int = 1, window_size: int = 100) -> Optional[list]:
    """
    Compute false nearest neighbors percentage across embedding dimensions.
    
    False nearest neighbors (FNN) identifies spurious nearest neighbors that
    appear close in lower dimensions but separate in higher dimensions. As
    dimension increases, FNN % decreases. When FNN < ~1%, the dimension is
    sufficient to unfold the attractor.
    
    Uses Cao's algorithm with ratio test:
    rt = (dist(m+1) - dist(m)) / dist(m), where rt > threshold indicates FNN.
    
    Args:
        price_series: Price series (typically closing prices)
        max_dimension: Maximum embedding dimension to test (default 10)
        delay: Time delay for embedding (default 1)
        window_size: Minimum data points required (default 100)
        
    Returns:
        List of FNN percentages for dimensions 1 to max_dimension, or None if insufficient data
    """
    returns = np.log(price_series / price_series.shift(1)).dropna().values
    
    if len(returns) < window_size:
        return None
    
    fnn_percentages = []
    
    for d in range(1, max_dimension + 1):
        embedded_d = []
        embedded_d_plus_1 = []
        
        for i in range(len(returns) - (d - 1) * delay):
            point_d = [returns[i + k * delay] for k in range(d)]
            point_d_plus_1 = point_d + [returns[i + d * delay]] if i + d * delay < len(returns) else None
            
            embedded_d.append(point_d)
            if point_d_plus_1 is not None:
                embedded_d_plus_1.append(point_d_plus_1)
        
        embedded_d = np.array(embedded_d)
        embedded_d_plus_1 = np.array(embedded_d_plus_1)
        
        if len(embedded_d_plus_1) == 0:
            break
        
        fnn_count = 0
        rt_threshold = 10
        
        for i in range(min(len(embedded_d), len(embedded_d_plus_1))):
            distances_d = np.linalg.norm(embedded_d - embedded_d[i], axis=1)
            nn_idx = np.argsort(distances_d)[1]
            
            dist_d = distances_d[nn_idx]
            if dist_d > 0:
                dist_d_plus_1 = np.linalg.norm(embedded_d_plus_1[i] - embedded_d_plus_1[nn_idx])
                rt = (dist_d_plus_1 - dist_d) / dist_d
                
                if rt > rt_threshold:
                    fnn_count += 1
        
        fnn_pct = 100 * fnn_count / len(embedded_d_plus_1)
        fnn_percentages.append(fnn_pct)
    
    return fnn_percentages

def find_optimal_dimension(fnn_percentages: list, threshold: float = 1.0) -> int:
    """
    Find optimal embedding dimension where FNN drops below threshold.
    
    As embedding dimension increases, false nearest neighbors decrease.
    When FNN < threshold (typically 1%), further increases don't improve
    the embedding. Returns first dimension meeting criterion.
    
    Args:
        fnn_percentages: List from false_nearest_neighbors()
        threshold: FNN percentage threshold (default 1.0 for 1%)
        
    Returns:
        Optimal dimension (1-indexed, min 1, max 5)
    """
    for i, fnn_pct in enumerate(fnn_percentages):
        if fnn_pct < threshold:
            return i + 1
    return min(len(fnn_percentages), 5)
