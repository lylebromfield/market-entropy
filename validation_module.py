import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any

class EntropyValidator:
    """
    Statistical validator for entropy signal predictiveness.
    
    Analyzes whether entropy-based alerts correlate with forward returns
    and market conditions, with multiple statistical tests including
    t-tests, Monte Carlo shuffling, and regime analysis.
    """
    
    def __init__(self, signal: pd.Series, prices: pd.Series, threshold: float) -> None:
        """
        Initialize validator with entropy signal and price data.
        
        Args:
            signal: Time series of entropy values
            prices: Price series (typically close prices)
            threshold: Entropy level below which signals are considered alerts
        """
        self.signal = signal
        self.prices = prices
        self.threshold = threshold
        
    def generate_alerts(self, lookback: int = 5) -> pd.Series:
        """
        Generate binary alert signals based on entropy threshold.
        
        Args:
            lookback: Window size (days) to check for entropy below threshold
            
        Returns:
            Boolean Series indicating alert days (True when entropy dropped)
        """
        alerts = pd.Series(False, index=self.signal.index)
        for i in range(lookback, len(self.signal)):
            window = self.signal.iloc[i-lookback:i+1]
            if (window <= self.threshold).any():
                alerts.iloc[i] = True
        return alerts
    
    def compute_forward_returns(self, horizons: list[int]) -> Dict[int, pd.Series]:
        """
        Compute forward returns at specified horizons.
        
        Args:
            horizons: List of forward periods in days (e.g., [5, 10, 20])
            
        Returns:
            Dictionary mapping horizon to forward returns Series
        """
        forward_returns = {}
        for h in horizons:
            fwd = self.prices.pct_change(h).shift(-h)
            forward_returns[h] = fwd.loc[self.signal.index]
        return forward_returns
    
    def alert_performance(self, forward_returns: Dict[int, pd.Series], alerts: pd.Series) -> Dict[int, Dict[str, float]]:
        """
        Compute performance metrics comparing alert vs non-alert periods.
        
        Uses independent t-tests to assess statistical significance of
        mean return differences between alert and non-alert periods.
        
        Args:
            forward_returns: Dictionary from compute_forward_returns
            alerts: Boolean Series of alert signals
            
        Returns:
            Dictionary with horizon -> metrics dictionary containing:
            - alert_mean: Mean return during alerts
            - no_alert_mean: Mean return without alerts
            - mean_diff: Difference in means
            - p_value: t-test significance (lower = more significant)
            - t_statistic: t-test statistic
        """
        metrics = {}
        for horizon, returns in forward_returns.items():
            valid_idx = ~returns.isna()
            alert_rets = returns[alerts & valid_idx]
            no_alert_rets = returns[~alerts & valid_idx]
            
            if len(alert_rets) > 0 and len(no_alert_rets) > 0:
                t_stat, p_val = stats.ttest_ind(alert_rets, no_alert_rets)
                metrics[horizon] = {
                    'alert_mean': float(alert_rets.mean()),
                    'no_alert_mean': float(no_alert_rets.mean()),
                    'mean_diff': float(alert_rets.mean() - no_alert_rets.mean()),
                    'p_value': float(p_val),
                    't_statistic': float(t_stat)
                }
        return metrics
    
    def drawdown_prediction(self, drawdown_threshold: float = -0.10) -> Dict[str, Any]:
        """
        Evaluate alert ability to predict significant drawdowns.
        
        Computes confusion matrix (true positives, false positives, etc.)
        and derives precision, recall, F1-score, and accuracy.
        
        Args:
            drawdown_threshold: Percent drawdown threshold (e.g., -0.10 for 10% drop)
            
        Returns:
            Dictionary with:
            - precision: TP / (TP + FP)
            - recall: TP / (TP + FN)
            - f1_score: Harmonic mean of precision and recall
            - accuracy: (TP + TN) / total
            - confusion matrix counts (true_positives, false_positives, etc.)
        """
        drawdowns = self.prices.pct_change(60).shift(-60)
        drawdown_events = (drawdowns <= drawdown_threshold).loc[self.signal.index]
        
        alerts = self.generate_alerts(lookback=5)
        
        valid_idx = ~drawdowns.isna()
        alerts_valid = alerts[valid_idx]
        events_valid = drawdown_events[valid_idx]
        
        tp = ((alerts_valid) & (events_valid)).sum()
        fp = ((alerts_valid) & (~events_valid)).sum()
        tn = ((~alerts_valid) & (~events_valid)).sum()
        fn = ((~alerts_valid) & (events_valid)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        return {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'accuracy': float(accuracy),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }
    
    def monte_carlo_test(self, n_simulations: int = 1000, horizons: list[int] = None) -> Dict[int, Dict[str, Any]]:
        """
        Assess statistical significance of alert signal via Monte Carlo shuffling.
        
        Shuffles alert labels randomly 1000 times and computes the null
        distribution of mean return differences. Compares actual difference
        to null distribution to estimate p-value.
        
        Args:
            n_simulations: Number of random shuffles (default 1000)
            horizons: Forward return horizons to test
            
        Returns:
            Dictionary with horizon -> results dictionary containing:
            - actual_diff: Observed mean return difference
            - null_mean: Mean of null distribution
            - null_std: Std of null distribution
            - p_value: Two-tailed p-value from null distribution
            - significant: Boolean (True if p < 0.05)
        """
        if horizons is None:
            horizons = [20]
            
        alerts = self.generate_alerts(lookback=5)
        forward_returns = self.compute_forward_returns(horizons)
        
        actual_metrics = self.alert_performance(forward_returns, alerts)
        
        results = {}
        for horizon in horizons:
            actual_diff = actual_metrics[horizon]['mean_diff']
            
            null_diffs = []
            for _ in range(n_simulations):
                shuffled_alerts = alerts.sample(frac=1.0, replace=False)
                shuffled_alerts.index = alerts.index
                
                valid_idx = ~forward_returns[horizon].isna()
                alert_rets = forward_returns[horizon][shuffled_alerts & valid_idx]
                no_alert_rets = forward_returns[horizon][~shuffled_alerts & valid_idx]
                
                if len(alert_rets) > 0 and len(no_alert_rets) > 0:
                    null_diffs.append(alert_rets.mean() - no_alert_rets.mean())
            
            p_value = (np.abs(null_diffs) >= np.abs(actual_diff)).mean()
            
            results[horizon] = {
                'actual_diff': float(actual_diff),
                'null_mean': float(np.mean(null_diffs)),
                'null_std': float(np.std(null_diffs)),
                'p_value': float(p_value),
                'significant': p_value < 0.05
            }
        
        return results
    
    def regime_analysis(self, n_regimes: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Analyze market behavior across entropy quintiles.
        
        Partitions entropy signal into equal-frequency regimes and computes
        return statistics for each regime, including Sharpe ratio.
        
        Args:
            n_regimes: Number of entropy regimes (default 5 for quintiles)
            
        Returns:
            Dictionary with regime name -> statistics dictionary containing:
            - mean: Mean forward 20-day return in regime
            - std: Standard deviation of returns
            - sharpe: Sharpe ratio (annualized, 252-day convention)
            - negative_pct: Percentage of negative returns
            - count: Number of observations in regime
        """
        quintiles = pd.qcut(self.signal, q=n_regimes, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'], duplicates='drop')
        
        fwd_returns = self.prices.pct_change(20).shift(-20).loc[self.signal.index]
        
        regime_stats = {}
        for regime in quintiles.cat.categories:
            regime_mask = (quintiles == regime)
            regime_returns = fwd_returns[regime_mask].dropna()
            
            if len(regime_returns) > 0:
                regime_stats[regime] = {
                    'mean': float(regime_returns.mean()),
                    'std': float(regime_returns.std()),
                    'sharpe': float(regime_returns.mean() / regime_returns.std() * np.sqrt(252)) if regime_returns.std() > 0 else 0,
                    'negative_pct': float((regime_returns < 0).mean()),
                    'count': int(len(regime_returns))
                }
        
        return regime_stats

def create_validation_plots(validator: EntropyValidator, alerts: pd.Series, forward_returns: Dict[int, pd.Series]) -> go.Figure:
    """
    Create 4-panel visualization of validation results.
    
    Panels show:
    1. Box plots comparing alert vs non-alert return distributions
    2. Histogram overlay of alert vs non-alert returns
    3. Cumulative performance during alert vs non-alert periods
    4. Alert timeline with signal overlay and alert markers
    
    Args:
        validator: EntropyValidator instance with signal and threshold
        alerts: Boolean Series of alert signals
        forward_returns: Dictionary from compute_forward_returns
        
    Returns:
        Plotly Figure with 4 subplots
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Forward Returns by Alert Status',
            'Return Distribution: Alert vs No Alert',
            'Cumulative Alert Performance',
            'Alert Timeline'
        ),
        specs=[
            [{'type': 'box'}, {'type': 'histogram'}],
            [{'type': 'scatter'}, {'type': 'scatter'}]
        ]
    )
    
    h = 20
    returns = forward_returns[h].dropna()
    alert_rets = returns[alerts.loc[returns.index]]
    no_alert_rets = returns[~alerts.loc[returns.index]]
    
    fig.add_trace(
        go.Box(y=alert_rets*100, name='Alert', marker_color='red'),
        row=1, col=1
    )
    fig.add_trace(
        go.Box(y=no_alert_rets*100, name='No Alert', marker_color='green'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Histogram(x=alert_rets*100, name='Alert', marker_color='red', opacity=0.7, nbinsx=30),
        row=1, col=2
    )
    fig.add_trace(
        go.Histogram(x=no_alert_rets*100, name='No Alert', marker_color='green', opacity=0.7, nbinsx=30),
        row=1, col=2
    )
    
    cum_alert = (alert_rets + 1).cumprod()
    cum_no_alert = (no_alert_rets + 1).cumprod()
    
    fig.add_trace(
        go.Scatter(x=cum_alert.index, y=cum_alert, name='Alert Period', line=dict(color='red')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=cum_no_alert.index, y=cum_no_alert, name='No Alert Period', line=dict(color='green')),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=validator.signal.index,
            y=validator.signal,
            name='Signal',
            line=dict(color='cyan')
        ),
        row=2, col=2
    )
    
    alert_dates = alerts[alerts].index
    if len(alert_dates) > 0:
        fig.add_trace(
            go.Scatter(
                x=alert_dates,
                y=validator.signal.loc[alert_dates],
                mode='markers',
                name='Alerts',
                marker=dict(color='red', size=8, symbol='x')
            ),
            row=2, col=2
        )
    
    fig.add_hline(y=validator.threshold, line_dash="dash", line_color="red", row=2, col=2)
    
    fig.update_xaxes(title_text="", row=1, col=1)
    fig.update_yaxes(title_text="20d Return (%)", row=1, col=1)
    
    fig.update_xaxes(title_text="20d Return (%)", row=1, col=2)
    fig.update_yaxes(title_text="Frequency", row=1, col=2)
    
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Return", row=2, col=1)
    
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_yaxes(title_text="Signal Value", row=2, col=2)
    
    fig.update_layout(
        template="plotly_dark",
        height=700,
        showlegend=True,
        barmode='overlay'
    )
    
    return fig
