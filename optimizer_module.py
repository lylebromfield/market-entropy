import pandas as pd
import numpy as np
from itertools import product
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from typing import Dict, Any, Tuple, Optional, Callable

class EntropyOptimizer:
    """
    Parameter optimizer for entropy-based trading signals.
    
    Uses grid search or walk-forward optimization to find optimal
    combinations of window size, smoothing, signal weighting, and
    alert thresholds. Supports PCA-based signal combination.
    """
    
    def __init__(self, mag7_returns: pd.DataFrame, sp500_price: pd.Series, 
                 run_tda_pipeline: Callable, run_takens_pipeline: Callable,
                 run_shannon_pipeline: Optional[Callable] = None) -> None:
        """
        Initialize optimizer with market data and entropy functions.
        
        Args:
            mag7_returns: Log returns of sector ETFs
            sp500_price: S&P 500 price series
            run_tda_pipeline: Function to compute correlation-based entropy
            run_takens_pipeline: Function to compute Takens embedding entropy
            run_shannon_pipeline: Function to compute Shannon entropy (optional)
        """
        self.mag7_returns = mag7_returns
        self.sp500_price = sp500_price
        self.run_tda_pipeline = run_tda_pipeline
        self.run_takens_pipeline = run_takens_pipeline
        self.run_shannon_pipeline = run_shannon_pipeline
    
    def _compute_signal(self, window_size: int, smoothing_span: int, corr_weight: float | str, 
                       threshold_pct: float) -> Tuple[Optional[pd.Series], Optional[float]]:
        """
        Compute combined entropy signal with smoothing and threshold.
        
        Combines correlation and Takens entropy with optional PCA weighting,
        applies exponential smoothing, and computes alert threshold.
        
        Args:
            window_size: Rolling window for entropy computation (days)
            smoothing_span: EWM span for signal smoothing
            corr_weight: Weight for correlation entropy (0-1) or 'pca' for auto-weighting
            threshold_pct: Percentile for alert threshold (0-100)
            
        Returns:
            Tuple of (smoothed signal, threshold value) or (None, None) if error
        """
        try:
            corr_entropy = self.run_tda_pipeline(self.mag7_returns, window_size)
            takens_entropy = self.run_takens_pipeline(self.sp500_price, window_size)
            
            # If Shannon entropy is available, use 3-signal system
            if self.run_shannon_pipeline is not None:
                shannon_entropy = self.run_shannon_pipeline(self.sp500_price, window_size)
                # handle missing takens/shannon by intersecting only available
                common_index = corr_entropy.index
                if len(takens_entropy) > 0:
                    common_index = common_index.intersection(takens_entropy.index)
                if len(shannon_entropy) > 0:
                    common_index = common_index.intersection(shannon_entropy.index)
                corr_entropy = corr_entropy.loc[common_index]
                takens_entropy = takens_entropy.loc[common_index] if len(takens_entropy) > 0 else pd.Series(0, index=common_index)
                shannon_entropy = shannon_entropy.loc[common_index] if len(shannon_entropy) > 0 else pd.Series(0, index=common_index)
                
                norm_corr = (corr_entropy - corr_entropy.mean()) / (corr_entropy.std() + 1e-8)
                norm_takens = (takens_entropy - takens_entropy.mean()) / (takens_entropy.std() + 1e-8)
                norm_shannon = (shannon_entropy - shannon_entropy.mean()) / (shannon_entropy.std() + 1e-8)
                
                if corr_weight == 'pca':
                    stacked_signals = np.column_stack([norm_corr, norm_takens, norm_shannon])
                    pca = PCA(n_components=1)
                    pca.fit(stacked_signals)
                    denom = (pca.components_[0, 0] ** 2 + pca.components_[0, 1] ** 2 + pca.components_[0, 2] ** 2)
                    pca_weight_corr = float(pca.components_[0, 0] ** 2 / denom)
                    pca_weight_takens = float(pca.components_[0, 1] ** 2 / denom)
                    pca_weight_shannon = 1 - pca_weight_corr - pca_weight_takens
                    combined_signal = pca_weight_corr * norm_corr + pca_weight_takens * norm_takens + pca_weight_shannon * norm_shannon
                else:
                    takens_shannon_weight = (1 - corr_weight) / 2
                    combined_signal = corr_weight * norm_corr + takens_shannon_weight * norm_takens + takens_shannon_weight * norm_shannon
            else:
                # Original 2-signal system (or corr-only if takens missing)
                if len(takens_entropy) > 0:
                    common_index = corr_entropy.index.intersection(takens_entropy.index)
                    corr_entropy = corr_entropy.loc[common_index]
                    takens_entropy = takens_entropy.loc[common_index]
                    norm_corr = (corr_entropy - corr_entropy.mean()) / (corr_entropy.std() + 1e-8)
                    norm_takens = (takens_entropy - takens_entropy.mean()) / (takens_entropy.std() + 1e-8)
                    if corr_weight == 'pca':
                        stacked_signals = np.column_stack([norm_corr, norm_takens])
                        pca = PCA(n_components=1)
                        pca.fit(stacked_signals)
                        pca_weight_corr = float(pca.components_[0, 0] ** 2 / (pca.components_[0, 0] ** 2 + pca.components_[0, 1] ** 2))
                        combined_signal = pca_weight_corr * norm_corr + (1 - pca_weight_corr) * norm_takens
                    else:
                        combined_signal = corr_weight * norm_corr + (1 - corr_weight) * norm_takens
                else:
                    # corr-only path (used by drift-only pipeline)
                    norm_corr = (corr_entropy - corr_entropy.mean()) / (corr_entropy.std() + 1e-8)
                    combined_signal = norm_corr
            
            signal_smooth = combined_signal.ewm(span=smoothing_span).mean()
            
            threshold = signal_smooth.quantile(threshold_pct / 100.0)
            
            return signal_smooth, threshold
        except:
            return None, None
    
    def _backtest_signal(self, signal: pd.Series, threshold: float, direction: str = 'long') -> Dict[str, Any]:
        """
        Backtest signal and compute performance metrics.
        
        Generates alerts when signal falls below threshold, then computes
        strategy returns assuming long/short position when alert is active.
        
        Args:
            signal: Time series of signal values
            threshold: Alert threshold (enter position when signal <= threshold)
            direction: 'long' for long bias, 'short' for short bias
            
        Returns:
            Dictionary with metrics:
            - total_return: Cumulative return over period
            - annual_return: Annualized return
            - sharpe: Sharpe ratio (annualized)
            - max_drawdown: Maximum drawdown from peak
            - win_rate: Percentage of positive days
            - pct_invested: Percentage of days with active position
            - num_trades: Total number of entry/exit signals
        """
        alerts = signal <= threshold
        
        sp500_filt = self.sp500_price.loc[signal.index]
        returns = sp500_filt.pct_change()
        
        if direction == 'long':
            strategy_returns = returns * alerts.astype(int).shift(1)
        else:
            strategy_returns = returns * (-alerts.astype(int).shift(1))
        
        cum_strategy = (1 + strategy_returns).cumprod()
        cum_bh = (1 + returns).cumprod()
        
        if len(cum_strategy) == 0 or len(cum_bh) == 0:
            return {
                'total_return': 0.0,
                'annual_return': 0.0,
                'sharpe': 0.0,
                'max_dd': 0.0,
                'win_rate': 0.0,
                'num_trades': 0,
                'bh_return': 0.0,
                'bh_annual_return': 0.0
            }
        
        total_return = float(cum_strategy.iloc[-1] - 1)
        bh_return = float(cum_bh.iloc[-1] - 1)
        
        years = len(strategy_returns) / 252
        annual_return = (cum_strategy.iloc[-1] ** (1 / years) - 1) if years > 0 else 0
        bh_annual_return = (cum_bh.iloc[-1] ** (1 / years) - 1) if years > 0 else 0
        
        daily_returns = strategy_returns.dropna()
        sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
        bh_daily_returns = returns.dropna()
        bh_sharpe = (bh_daily_returns.mean() / bh_daily_returns.std() * np.sqrt(252)) if bh_daily_returns.std() > 0 else 0
        
        rolling_max = cum_strategy.expanding().max()
        drawdown = (cum_strategy - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min())
        
        win_rate = (strategy_returns > 0).sum() / len(strategy_returns) if len(strategy_returns) > 0 else 0
        
        pct_invested = alerts.astype(int).mean()
        num_trades = alerts.diff().abs().sum()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'bh_annual_return': bh_annual_return,
            'sharpe': sharpe,
            'bh_sharpe': bh_sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'pct_invested': pct_invested,
            'num_trades': num_trades
        }
    
    def grid_search(self, param_grid: Dict[str, list], direction: str = 'long', 
                   max_combinations: int = 500) -> pd.DataFrame:
        """
        Exhaustive grid search over parameter combinations.
        
        Tests all combinations of window_size, smoothing_span, corr_weight, and
        threshold_pct (up to max_combinations limit for speed).
        
        Args:
            param_grid: Dict with lists for each parameter:
                - window_size: List of rolling window sizes
                - smoothing_span: List of EWM spans
                - corr_weight: List of weights (0-1) or ['pca']
                - threshold_pct: List of percentiles (0-100)
            direction: 'long' or 'short' bias
            max_combinations: Maximum combinations to test
            
        Returns:
            DataFrame with all parameters and backtest metrics for each combo
        """
        results = []
        
        params_list = list(product(
            param_grid['window_size'],
            param_grid['smoothing_span'],
            param_grid['corr_weight'],
            param_grid['threshold_pct']
        ))
        
        for window_size, smoothing_span, corr_weight, threshold_pct in params_list[:max_combinations]:
            signal, threshold = self._compute_signal(window_size, smoothing_span, corr_weight, threshold_pct)
            
            if signal is not None and threshold is not None:
                metrics = self._backtest_signal(signal, threshold, direction)
                
                results.append({
                    'window_size': window_size,
                    'smoothing_span': smoothing_span,
                    'corr_weight': corr_weight,
                    'threshold_pct': threshold_pct,
                    **metrics
                })
        
        return pd.DataFrame(results)
    
    def walk_forward_optimization(self, param_grid: Dict[str, list], train_days: int = 1260, 
                                 test_days: int = 252, direction: str = 'long', 
                                 min_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """
        Walk-forward parameter optimization with rolling train/test windows.
        
        Performs out-of-sample testing by: (1) optimizing parameters on training
        data using best in-sample Sharpe, (2) applying those parameters to test
        period, (3) rolling windows forward and repeating.
        
        Args:
            param_grid: Parameter ranges to test
            train_days: Training window size in days (default 5 years)
            test_days: Testing window size in days (default 1 year)
            direction: 'long' or 'short' bias
            min_date: Optional minimum date for analysis start
            
        Returns:
            DataFrame with fold number, dates, parameters, and test performance
        """
        results = []
        
        all_dates = self.sp500_price.index
        if min_date is not None:
            all_dates = all_dates[all_dates >= min_date]
        
        start_idx = train_days
        fold = 0
        
        while start_idx + test_days <= len(all_dates):
            train_end_idx = start_idx
            test_end_idx = start_idx + test_days
            
            train_dates = all_dates[:train_end_idx]
            test_dates = all_dates[train_end_idx:test_end_idx]
            
            mag7_train = self.mag7_returns.loc[:train_dates[-1]]
            sp500_train = self.sp500_price.loc[:train_dates[-1]]
            
            mag7_test = self.mag7_returns.loc[test_dates]
            sp500_test = self.sp500_price.loc[test_dates]
            
            best_train_metrics = None
            best_params = None
            best_train_sharpe = -np.inf
            
            params_list = list(product(
                param_grid['window_size'],
                param_grid['smoothing_span'],
                param_grid['corr_weight'],
                param_grid['threshold_pct']
            ))
            
            for window_size, smoothing_span, corr_weight, threshold_pct in params_list:
                try:
                    corr_entropy = self.run_tda_pipeline(mag7_train, window_size)
                    takens_entropy = self.run_takens_pipeline(sp500_train, window_size)
                    
                    common_index = corr_entropy.index.intersection(takens_entropy.index)
                    corr_entropy = corr_entropy.loc[common_index]
                    takens_entropy = takens_entropy.loc[common_index]
                    
                    norm_corr = (corr_entropy - corr_entropy.mean()) / corr_entropy.std()
                    norm_takens = (takens_entropy - takens_entropy.mean()) / takens_entropy.std()
                    
                    combined_signal = corr_weight * norm_corr + (1 - corr_weight) * norm_takens
                    signal_smooth = combined_signal.ewm(span=smoothing_span).mean()
                    
                    threshold = signal_smooth.quantile(threshold_pct / 100.0)
                    
                    train_metrics = self._backtest_signal(signal_smooth, threshold, direction)
                    
                    if train_metrics['sharpe'] > best_train_sharpe:
                        best_train_sharpe = train_metrics['sharpe']
                        best_params = (window_size, smoothing_span, corr_weight, threshold_pct)
                        best_train_metrics = train_metrics
                except:
                    continue
            
            if best_params is not None:
                window_size, smoothing_span, corr_weight, threshold_pct = best_params
                
                signal_test, threshold_test = self._compute_signal(window_size, smoothing_span, corr_weight, threshold_pct)
                
                if signal_test is not None:
                    signal_test = signal_test.loc[test_dates]
                    test_metrics = self._backtest_signal(signal_test, threshold_test, direction)
                    
                    results.append({
                        'fold': fold,
                        'test_start': test_dates[0],
                        'test_end': test_dates[-1],
                        'window_size': window_size,
                        'smoothing_span': smoothing_span,
                        'corr_weight': corr_weight,
                        'threshold_pct': threshold_pct,
                        'train_sharpe': best_train_metrics['sharpe'],
                        'test_sharpe': test_metrics['sharpe'],
                        'test_annual_return': test_metrics['annual_return'],
                        'test_max_drawdown': test_metrics['max_drawdown'],
                        'test_win_rate': test_metrics['win_rate']
                    })
            
            fold += 1
            start_idx += test_days
        
        return pd.DataFrame(results)
    
    def visualize_results(self, results_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
        """
        Visualize top parameter combinations and their performance metrics.
        
        Creates 4-panel bar chart showing Sharpe, annual return, max drawdown,
        and win rate for the top N configurations.
        
        Args:
            results_df: DataFrame from grid_search or walk_forward_optimization
            top_n: Number of top results to display
            
        Returns:
            Plotly Figure with 4 performance subplots
        """
        top_results = results_df.head(top_n)
        # Ensure required columns exist; otherwise drop the corresponding subplot
        has_drawdown = 'max_drawdown' in top_results.columns
        has_winrate = 'win_rate' in top_results.columns
        has_return = 'annual_return' in top_results.columns
        
        specs = [[{'type': 'bar'}, {'type': 'bar'}], [{'type': 'bar'}, {'type': 'bar'}]]
        titles = ['Sharpe Ratio', 'Annual Return', 'Max Drawdown', 'Win Rate']
        if not has_return:
            specs[0][1] = None
            titles[1] = ''
        if not has_drawdown:
            specs[1][0] = None
            titles[2] = ''
        if not has_winrate:
            specs[1][1] = None
            titles[3] = ''

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=tuple(titles),
            specs=specs
        )
        
        labels = [f"W{int(r['window_size'])} S{int(r['smoothing_span'])}" for _, r in top_results.iterrows()]
        
        fig.add_trace(
            go.Bar(y=top_results['sharpe'], name='Sharpe', marker_color='#00ff00', text=[f"{x:.2f}" for x in top_results['sharpe']], textposition='outside'),
            row=1, col=1
        )

        if has_return:
            fig.add_trace(
                go.Bar(y=top_results['annual_return']*100, name='Return', marker_color='#00ffff', text=[f"{x:.1f}%" for x in top_results['annual_return']*100], textposition='outside'),
                row=1, col=2
            )
        if has_drawdown:
            fig.add_trace(
                go.Bar(y=top_results['max_drawdown']*100, name='Drawdown', marker_color='#ff0000', text=[f"{x:.1f}%" for x in top_results['max_drawdown']*100], textposition='outside'),
                row=2, col=1
            )
        if has_winrate:
            fig.add_trace(
                go.Bar(y=top_results['win_rate']*100, name='Win Rate', marker_color='#ffff00', text=[f"{x:.1f}%" for x in top_results['win_rate']*100], textposition='outside'),
                row=2, col=2
            )
        
        fig.update_yaxes(title_text="Sharpe", row=1, col=1)
        if has_return:
            fig.update_yaxes(title_text="Return %", row=1, col=2)
        if has_drawdown:
            fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
        if has_winrate:
            fig.update_yaxes(title_text="Win Rate %", row=2, col=2)
        
        fig.update_layout(
            template="plotly_dark",
            height=600,
            showlegend=False,
            xaxis_tickangle=-45
        )
        
        return fig
